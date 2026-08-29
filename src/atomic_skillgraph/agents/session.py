"""Client-managed replay sessions with fail-closed Agent protocol handling."""

from __future__ import annotations

import copy
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from ..core.errors import AgentProtocolError, AtomicSkillGraphError, FailureLayer
from .protocol import (
    AgentMessage,
    AgentProvider,
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    SchemaValidationError,
    parse_json_strict,
    validate_schema_instance,
)
from .usage import AgentBudget, BudgetTracker, LLMUsage, UsageBucket, UsageLedger


_PROTOCOL_REPAIR_LIMIT = 1


@dataclass(frozen=True)
class ProtocolFailureRecord:
    turn_index: int
    code: str
    message: str
    repair_attempted: bool
    rejected_turn: dict[str, Any] | None = None


class ReplayAgentSession:
    """Agent session for providers without server-side session state.

    The instance owns and replays its message history.  A tool result is accepted
    only when its call id is the one currently pending in this same instance.
    """

    def __init__(
        self,
        provider: AgentProvider,
        *,
        system_prompt: str,
        usage_ledger: UsageLedger,
        usage_bucket: UsageBucket | str,
        budget: AgentBudget | None = None,
        session_id: str | None = None,
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("Agent session requires a non-empty system_prompt")
        self._provider = provider
        self._session_id = session_id or f"session_{uuid.uuid4().hex}"
        if not self._session_id.strip():
            raise ValueError("session_id must be non-empty")
        self._messages: list[AgentMessage] = [
            {"role": "system", "content": system_prompt.strip()}
        ]
        self._usage_ledger = usage_ledger
        self._usage_bucket = UsageBucket(usage_bucket)
        self._budget_tracker = BudgetTracker(budget) if budget is not None else None
        self._pending_call: NativeToolCall | None = None
        self._seen_call_ids: set[str] = set()
        self._last_tools: list[NativeToolSpec] = []
        self._turn_index = 0
        self._protocol_failures: list[ProtocolFailureRecord] = []
        self._terminal_protocol_failure: AgentProtocolError | None = None
        self._finalized = False
        self._lock = threading.RLock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def usage_ledger(self) -> UsageLedger:
        return self._usage_ledger

    @property
    def usage_bucket(self) -> UsageBucket:
        return self._usage_bucket

    @property
    def pending_tool_call(self) -> NativeToolCall | None:
        return self._pending_call

    def set_usage_bucket(self, bucket: UsageBucket | str) -> None:
        """Switch the accounting stage within a shared Planner/Extractor session."""
        with self._lock:
            self._ensure_live()
            if self._pending_call is not None:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "cannot change usage bucket while a tool call is pending",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            self._usage_bucket = UsageBucket(bucket)

    def next_turn(
        self,
        user_input: str | None,
        *,
        tools: list[NativeToolSpec] | None = None,
        structured_output_schema: dict[str, Any] | None = None,
    ) -> AgentTurn:
        with self._lock:
            self._ensure_live()
            if self._pending_call is not None:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "pending native tool call requires submit_tool_result in the same session",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            if user_input is not None and not isinstance(user_input, str):
                raise TypeError("user_input must be a string or None")
            if user_input is None and self._messages[-1]["role"] != "system":
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "next_turn requires new user input unless this is the first session turn",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            normalized_tools = _normalize_tools(tools)
            schema = _normalize_schema(structured_output_schema)
            self._check_budget_before_call()
            if user_input is not None:
                self._messages.append({"role": "user", "content": user_input})
            self._last_tools = normalized_tools
            return self._request_valid_turn(normalized_tools, schema)

    def submit_tool_result(
        self,
        call_id: str,
        result: dict[str, Any],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        with self._lock:
            self._ensure_live()
            pending = self._pending_call
            if pending is None:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "this session has no pending native tool call",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            if call_id != pending.call_id:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "tool result call_id does not match this session's pending call",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            normalized_tools = self._last_tools if tools is None else _normalize_tools(tools)
            self._check_budget_before_call()
            self._append_tool_result(pending, result)
            self._last_tools = list(normalized_tools)
            return self._request_valid_turn(normalized_tools, None)

    def finalize_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        """Close a terminal session after returning its final native-tool result.

        This preserves the Assistant ToolCall → Tool result protocol without
        purchasing an unused extra provider turn or leaving a pending call.
        """
        with self._lock:
            self._ensure_live()
            pending = self._pending_call
            if pending is None or call_id != pending.call_id:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "final tool result call_id does not match this session's pending call",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            self._append_tool_result(pending, result)
            self._finalized = True

    def _append_tool_result(self, pending: NativeToolCall, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise TypeError("tool result must be a JSON object")
        try:
            encoded_result = json.dumps(
                result, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("tool result must be JSON serializable") from exc
        self._messages.append({
            "role": "tool", "tool_call_id": pending.call_id,
            "name": pending.name, "content": encoded_result,
        })
        self._pending_call = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_call
            return {
                "session_id": self._session_id,
                "messages": copy.deepcopy(self._messages),
                "pending_tool_call": _tool_call_dict(pending) if pending is not None else None,
                "finalized": self._finalized,
                "seen_call_ids": sorted(self._seen_call_ids),
                "turn_count": self._turn_index,
                "usage_bucket": self._usage_bucket.value,
                "budget": self._budget_tracker.snapshot() if self._budget_tracker else None,
                "protocol_failures": [
                    {
                        "turn_index": item.turn_index,
                        "code": item.code,
                        "message": item.message,
                        "repair_attempted": item.repair_attempted,
                        "rejected_turn": copy.deepcopy(item.rejected_turn),
                    }
                    for item in self._protocol_failures
                ],
                "terminal_protocol_failure": (
                    {
                        "code": self._terminal_protocol_failure.code,
                        "message": str(self._terminal_protocol_failure),
                    }
                    if self._terminal_protocol_failure is not None
                    else None
                ),
                "provider": self._provider.snapshot(),
            }

    def _request_valid_turn(
        self,
        tools: list[NativeToolSpec],
        structured_output_schema: dict[str, Any] | None,
    ) -> AgentTurn:
        repair_count = 0
        while True:
            self._check_budget_before_call()
            try:
                turn = self._provider.complete(
                    copy.deepcopy(self._messages),
                    tools=list(tools) or None,
                    structured_output_schema=copy.deepcopy(structured_output_schema),
                )
            except AgentProtocolError as exc:
                metering_turn = getattr(exc, "usage_turn", None)
                if not isinstance(metering_turn, AgentTurn):
                    # A provider may reject before it can construct AgentTurn.
                    # The call is still represented as unavailable usage.
                    metering_turn = _unavailable_turn(self._provider)
                self._record_provider_call(metering_turn)
                failure = exc
                rejected_turn = (
                    _turn_dict(metering_turn)
                    if getattr(exc, "usage_turn", None) is not None
                    else None
                )
            else:
                self._record_provider_call(turn)
                try:
                    self._validate_turn(turn, tools, structured_output_schema)
                except AgentProtocolError as exc:
                    failure = exc
                    rejected_turn = _turn_dict(turn)
                else:
                    self._append_assistant_turn(turn)
                    return turn

            can_repair = repair_count < _PROTOCOL_REPAIR_LIMIT
            self._protocol_failures.append(
                ProtocolFailureRecord(
                    turn_index=max(0, self._turn_index - 1),
                    code=failure.code,
                    message=str(failure),
                    repair_attempted=can_repair,
                    rejected_turn=rejected_turn,
                )
            )
            if not can_repair:
                self._terminal_protocol_failure = failure
                raise failure
            repair_count += 1
            self._check_budget_before_call()
            self._messages.append(
                {
                    "role": "user",
                    "content": _protocol_repair_message(
                        tools=tools,
                        structured_output_schema=structured_output_schema,
                        failure=failure,
                    ),
                }
            )

    def _validate_turn(
        self,
        turn: AgentTurn,
        tools: list[NativeToolSpec],
        structured_output_schema: dict[str, Any] | None,
    ) -> None:
        if len(turn.tool_calls) > 1:
            raise AgentProtocolError(
                "runtime_agent_multiple_tool_calls",
                "an Agent turn may contain at most one native tool call",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        if turn.tool_calls:
            call = turn.tool_calls[0]
            offered = {tool.name: tool for tool in tools}
            if call.name not in offered:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    f"native tool {call.name!r} was not offered in this turn",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            if call.call_id in self._seen_call_ids:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    f"native tool call_id {call.call_id!r} was already used in this session",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            try:
                validate_schema_instance(call.arguments, offered[call.name].input_schema)
            except SchemaValidationError as exc:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    f"native tool arguments failed schema validation: {exc}",
                    layer=FailureLayer.RUNTIME_AGENT,
                ) from exc
            return

        if structured_output_schema is None:
            raise AgentProtocolError(
                "agent_protocol_no_action",
                "assistant returned no native tool call or requested structured output",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        try:
            structured_value = parse_json_strict(turn.content)
        except (TypeError, ValueError) as exc:
            raise AgentProtocolError(
                "runtime_agent_schema_error",
                "assistant structured output is not a single valid JSON value",
                layer=FailureLayer.RUNTIME_AGENT,
            ) from exc
        try:
            validate_schema_instance(structured_value, structured_output_schema)
        except SchemaValidationError as exc:
            raise AgentProtocolError(
                "runtime_agent_schema_error",
                f"assistant structured output failed schema validation: {exc}",
                layer=FailureLayer.RUNTIME_AGENT,
            ) from exc

    def _append_assistant_turn(self, turn: AgentTurn) -> None:
        message: AgentMessage = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            call = turn.tool_calls[0]
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    },
                }
            ]
            self._pending_call = call
            self._seen_call_ids.add(call.call_id)
        self._messages.append(message)

    def _record_provider_call(self, turn: AgentTurn) -> None:
        self._usage_ledger.record_turn(
            session_id=self._session_id,
            turn_index=self._turn_index,
            bucket=self._usage_bucket,
            turn=turn,
        )
        self._turn_index += 1
        if self._budget_tracker is not None:
            try:
                self._budget_tracker.consume(LLMUsage.from_turn(turn))
            except AtomicSkillGraphError:
                raise

    def _check_budget_before_call(self) -> None:
        if self._budget_tracker is not None:
            self._budget_tracker.check_before_call()

    def _ensure_live(self) -> None:
        if self._finalized:
            raise AgentProtocolError(
                "runtime_agent_schema_error",
                "session was finalized after its terminal tool result",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        if self._terminal_protocol_failure is not None:
            raise AgentProtocolError(
                self._terminal_protocol_failure.code,
                "session is fail-closed after an unrepaired protocol violation",
                layer=FailureLayer.RUNTIME_AGENT,
            )


ClientManagedAgentSession = ReplayAgentSession


def _normalize_tools(tools: list[NativeToolSpec] | None) -> list[NativeToolSpec]:
    values = list(tools or [])
    if not all(isinstance(item, NativeToolSpec) for item in values):
        raise TypeError("tools must contain NativeToolSpec values")
    names = [item.name for item in values]
    if len(set(names)) != len(names):
        raise ValueError("native tool names must be unique within a turn")
    return values


def _normalize_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    if not isinstance(schema, dict):
        raise TypeError("structured_output_schema must be a mapping or None")
    try:
        json.dumps(schema, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("structured_output_schema must be JSON serializable") from exc
    return copy.deepcopy(schema)


def _protocol_repair_message(
    *,
    tools: list[NativeToolSpec],
    structured_output_schema: dict[str, Any] | None,
    failure: AgentProtocolError,
) -> str:
    if tools and structured_output_schema is not None:
        expected = (
            "Return either exactly one native tool call using an offered tool, or one raw JSON "
            "value conforming to the supplied structured-output schema."
        )
    elif tools:
        expected = "Return exactly one native tool call using one of the offered tools."
    else:
        expected = "Return one raw JSON value conforming to the supplied structured-output schema."
    return (
        "PROTOCOL REPAIR REQUIRED. The previous response was rejected and no action was executed. "
        f"Violation: {failure.code}. {expected} Do not encode action arguments in prose or Markdown."
    )


def _tool_call_dict(call: NativeToolCall) -> dict[str, Any]:
    return {"call_id": call.call_id, "name": call.name, "arguments": copy.deepcopy(call.arguments)}


def _turn_dict(turn: AgentTurn) -> dict[str, Any]:
    return {
        "content": turn.content,
        "tool_calls": [_tool_call_dict(call) for call in turn.tool_calls],
        "finish_reason": turn.finish_reason,
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "total_tokens": turn.total_tokens,
        "reasoning_tokens": turn.reasoning_tokens,
        "latency_ms": turn.latency_ms,
        "provider_metadata": _safe_provider_metadata(turn.provider_metadata),
    }


def _safe_provider_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    allowed_reasoning = {
        "reasoning_tokens_status",
        "reasoning_tokens_source",
        "reasoning_tokens_in_completion",
    }
    for key, value in metadata.items():
        lowered = str(key).lower()
        if "reasoning" in lowered and lowered not in allowed_reasoning:
            continue
        safe[str(key)] = copy.deepcopy(value)
    return safe


def _unavailable_turn(provider: AgentProvider) -> AgentTurn:
    metadata = dict(provider.snapshot())
    metadata.update(
        {
            "usage_status": "unavailable",
            "reasoning_tokens_status": "unavailable",
        }
    )
    return AgentTurn(
        content="",
        tool_calls=[],
        finish_reason="protocol_error",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        reasoning_tokens=None,
        latency_ms=0.0,
        provider_metadata=metadata,
    )


__all__ = [
    "ClientManagedAgentSession",
    "ProtocolFailureRecord",
    "ReplayAgentSession",
]
