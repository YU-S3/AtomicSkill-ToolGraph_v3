"""Client-managed replay sessions with fail-closed Agent protocol handling."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from ..core.errors import (
    AgentProtocolError,
    AtomicSkillGraphError,
    BudgetExhausted,
    FailureLayer,
)
from .protocol import (
    AgentMessage,
    AgentProvider,
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    SchemaValidationError,
    validate_schema_instance,
)
from .usage import AgentBudget, BudgetTracker, LLMUsage, UsageBucket, UsageLedger


PROTOCOL_REPAIR_LIMIT = 1
_POLICY_CONTEXT_SEPARATOR = "\n\nPOLICY_CONTEXT_JSON\n"


def structured_provider_turn_cap(semantic_max_turns: int) -> int:
    """Add the one session-wide protocol-repair call to a semantic turn cap."""

    if (
        isinstance(semantic_max_turns, bool)
        or not isinstance(semantic_max_turns, int)
        or semantic_max_turns < 0
    ):
        raise ValueError("semantic_max_turns must be a non-negative integer")
    return int(semantic_max_turns) + PROTOCOL_REPAIR_LIMIT


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
        semantic_max_turns: int | None = None,
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
        if (
            semantic_max_turns is not None
            and (
                isinstance(semantic_max_turns, bool)
                or not isinstance(semantic_max_turns, int)
                or semantic_max_turns < 0
            )
        ):
            raise ValueError("semantic_max_turns must be a non-negative integer or None")
        if semantic_max_turns is not None and budget is None:
            raise ValueError("semantic_max_turns requires an AgentBudget")
        self._semantic_max_turns = semantic_max_turns
        self._pending_call: NativeToolCall | None = None
        self._seen_call_ids: set[str] = set()
        self._last_tools: list[NativeToolSpec] = []
        self._turn_index = 0
        self._accepted_turn_count = 0
        self._protocol_repairs_used = 0
        self._context_compaction_count = 0
        self._replay_initial_catalog_compacted = False
        self._replay_full_catalog_count_at_last_request = 0
        self._replay_history_action_count = 0
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
            self._check_semantic_budget_before_call()
            self._check_budget_before_call()
            if user_input is not None:
                self._messages.append({"role": "user", "content": user_input})
            self._last_tools = normalized_tools
            return self._request_valid_turn(normalized_tools)

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
            self._check_semantic_budget_before_call()
            self._check_budget_before_call()
            self._append_tool_result(pending, result)
            self._last_tools = list(normalized_tools)
            return self._request_valid_turn(normalized_tools)

    def acknowledge_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        """Append a Tool result without purchasing another provider turn.

        Planner/Extractor structured submissions use this acknowledgement to
        keep one live replay session across their semantic stages while keeping
        protocol-format repair independent from P1R/P2R semantic repair.
        """
        with self._lock:
            self._ensure_live()
            pending = self._pending_call
            if pending is None or call_id != pending.call_id:
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "acknowledged tool result does not match this session's pending call",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            self._append_tool_result(pending, result)

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
            "content": encoded_result,
        })
        self._pending_call = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_call
            return {
                "session_id": self._session_id,
                "messages": _safe_messages_snapshot(self._messages),
                "pending_tool_call": _tool_call_dict(pending) if pending is not None else None,
                "finalized": self._finalized,
                "seen_call_ids": sorted(self._seen_call_ids),
                "turn_count": self._turn_index,
                "provider_call_count": self._turn_index,
                "accepted_turn_count": self._accepted_turn_count,
                "protocol_repairs_used": self._protocol_repairs_used,
                "context_compaction_count": self._context_compaction_count,
                "replay_catalog_compaction_count": self._context_compaction_count,
                "replay_initial_catalog_compacted": (
                    self._replay_initial_catalog_compacted
                ),
                "replay_full_catalog_count_at_last_request": (
                    self._replay_full_catalog_count_at_last_request
                ),
                "replay_history_action_count": self._replay_history_action_count,
                "semantic_budget": (
                    None
                    if self._semantic_max_turns is None
                    else {
                        "max_turns": self._semantic_max_turns,
                        "used_turns": self._accepted_turn_count,
                        "remaining_turns": max(
                            0, self._semantic_max_turns - self._accepted_turn_count,
                        ),
                    }
                ),
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
    ) -> AgentTurn:
        while True:
            self._check_budget_before_call()
            self._compact_superseded_action_catalogs()
            accepted_candidate: AgentTurn | None = None
            try:
                set_context = getattr(self._provider, "set_request_context", None)
                if callable(set_context):
                    set_context(session_id=self._session_id, stage=self._usage_bucket.value)
                turn = self._provider.complete(
                    copy.deepcopy(self._messages),
                    tools=list(tools) or None,
                )
            except AtomicSkillGraphError as exc:
                if not isinstance(exc, AgentProtocolError):
                    metering_turn = getattr(exc, "usage_turn", None)
                    if isinstance(metering_turn, AgentTurn):
                        self._record_provider_call(metering_turn)
                    # Provider failures are Infrastructure.  They are never
                    # protocol-repaired and never converted to Dynamic.
                    raise
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
                accepted_candidate = turn
                try:
                    self._validate_turn(turn, tools)
                except AgentProtocolError as exc:
                    failure = exc
                    rejected_turn = _turn_dict(turn)
                else:
                    self._append_assistant_turn(turn)
                    self._accepted_turn_count += 1
                    return turn

            can_repair = self._protocol_repairs_used < PROTOCOL_REPAIR_LIMIT
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
            self._protocol_repairs_used += 1
            self._check_budget_before_call()
            if accepted_candidate is not None:
                self._append_rejected_turn_for_replay(accepted_candidate, failure)
            self._messages.append(
                {
                    "role": "user",
                    "content": _protocol_repair_message(
                        tools=tools,
                        failure=failure,
                    ),
                }
            )

    def _validate_turn(
        self,
        turn: AgentTurn,
        tools: list[NativeToolSpec],
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

        raise AgentProtocolError(
            "agent_protocol_no_action",
            "assistant returned no native tool call",
            layer=FailureLayer.RUNTIME_AGENT,
        )

    def _append_assistant_turn(self, turn: AgentTurn) -> None:
        message = _assistant_replay_message(turn)
        if turn.tool_calls:
            call = turn.tool_calls[0]
            self._pending_call = call
            self._seen_call_ids.add(call.call_id)
        self._messages.append(message)

    def _append_rejected_turn_for_replay(
        self,
        turn: AgentTurn,
        failure: AgentProtocolError,
    ) -> None:
        """Preserve DeepSeek reasoning while explicitly rejecting every call.

        This is only a submission/protocol repair.  No environment action is
        executed and no semantic repair quota is consumed.
        """
        self._messages.append(_assistant_replay_message(turn))
        for call in turn.tool_calls:
            self._seen_call_ids.add(call.call_id)
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(
                        {
                            "accepted": False,
                            "executed": False,
                            "error": failure.code,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

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

    def _check_semantic_budget_before_call(self) -> None:
        if (
            self._semantic_max_turns is None
            or self._accepted_turn_count < self._semantic_max_turns
        ):
            return
        assert self._budget_tracker is not None
        raise BudgetExhausted(
            self._budget_tracker.budget.exhaustion_code,
            "agent semantic turn budget exhausted",
            layer=FailureLayer.RUNTIME_AGENT,
        )

    def _compact_superseded_action_catalogs(self) -> None:
        """Keep at most one current full catalog in Runtime replay.

        Assistant envelopes (including DeepSeek reasoning_content) and action
        ToolCall envelopes remain byte-for-byte structurally intact.  Only
        client-authored Runtime policy JSON is rewritten: stale catalogs become
        auditable markers and verbose budget snapshots become remaining-quota
        views.  Canonical Trace and GroundingEvidence keep the full records.
        """

        if not self._usage_bucket.value.startswith("runtime_"):
            self._replay_full_catalog_count_at_last_request = 0
            self._replay_history_action_count = 0
            return

        catalogs: list[dict[str, Any]] = []
        markers: list[dict[str, Any]] = []
        history_action_count = 0
        for index, message in enumerate(self._messages):
            role = message.get("role")
            if role == "user":
                decoded = _decode_policy_context(str(message.get("content", "")))
                if decoded is None:
                    continue
                prefix, payload = decoded
                for history_key in (
                    "relevant_action_history",
                    "relevant_real_action_history",
                ):
                    history = payload.get(history_key)
                    if isinstance(history, list):
                        history_action_count += len(history)
                catalog = payload.get("current_action_catalog")
                count = _full_catalog_entry_count(catalog)
                if count is not None:
                    catalogs.append({
                        "kind": "initial",
                        "index": index,
                        "prefix": prefix,
                        "payload": payload,
                        "field": "current_action_catalog",
                        "entry_count": count,
                        "revision": _catalog_revision(catalog),
                    })
                elif _is_superseded_catalog_marker(catalog):
                    markers.append({
                        "kind": "initial",
                        "index": index,
                        "prefix": prefix,
                        "payload": payload,
                        "field": "current_action_catalog",
                    })
                compact_budget = _compact_replay_budget(payload.get("remaining_budget"))
                if compact_budget is not None and compact_budget != payload.get(
                    "remaining_budget"
                ):
                    payload["remaining_budget"] = compact_budget
                    self._messages[index]["content"] = _encode_policy_context(
                        prefix, payload
                    )
                continue
            if role != "tool":
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            compact_budget = _compact_replay_budget(payload.get("remaining_budget"))
            if compact_budget is not None:
                payload["remaining_budget"] = compact_budget
            catalog = payload.get("action_catalog")
            count = _full_catalog_entry_count(catalog)
            if (
                "new_revision" in payload
                and "observation" in payload
                and (
                    count is not None
                    or _is_superseded_catalog_marker(catalog)
                )
            ):
                history_action_count += 1
            if count is not None:
                catalogs.append({
                    "kind": "tool",
                    "index": index,
                    "payload": payload,
                    "field": "action_catalog",
                    "entry_count": count,
                    "revision": _catalog_revision(
                        catalog, fallback=payload.get("new_revision")
                    ),
                })
            elif _is_superseded_catalog_marker(catalog):
                markers.append({
                    "kind": "tool",
                    "index": index,
                    "payload": payload,
                    "field": "action_catalog",
                })
            self._messages[index]["content"] = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        latest_revision = catalogs[-1]["revision"] if catalogs else None
        if latest_revision is not None:
            for marker in markers:
                payload = marker["payload"]
                payload[marker["field"]]["superseded_by_revision"] = latest_revision
                index = int(marker["index"])
                if marker["kind"] == "initial":
                    self._messages[index]["content"] = _encode_policy_context(
                        str(marker["prefix"]), payload
                    )
                else:
                    self._messages[index]["content"] = json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
        for item in catalogs[:-1]:
            payload = item["payload"]
            payload[item["field"]] = {
                "status": "superseded",
                "entry_count": item["entry_count"],
                "superseded_by_revision": latest_revision,
            }
            index = int(item["index"])
            if item["kind"] == "initial":
                self._messages[index]["content"] = _encode_policy_context(
                    str(item["prefix"]), payload
                )
                self._replay_initial_catalog_compacted = True
            else:
                self._messages[index]["content"] = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            self._context_compaction_count += 1
        self._replay_full_catalog_count_at_last_request = min(1, len(catalogs))
        self._replay_history_action_count = history_action_count

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


def _decode_policy_context(content: str) -> tuple[str, dict[str, Any]] | None:
    if _POLICY_CONTEXT_SEPARATOR not in content:
        return None
    prefix, encoded = content.split(_POLICY_CONTEXT_SEPARATOR, 1)
    try:
        payload = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return prefix, payload


def _encode_policy_context(prefix: str, payload: dict[str, Any]) -> str:
    return prefix + _POLICY_CONTEXT_SEPARATOR + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _full_catalog_entry_count(value: Any) -> int | None:
    """Return entry count only for a full policy catalog, never a marker."""

    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and isinstance(value.get("actions"), list):
        return len(value["actions"])
    return None


def _catalog_revision(value: Any, *, fallback: Any = None) -> Any:
    if isinstance(value, dict) and "revision" in value:
        return value.get("revision")
    if isinstance(value, list):
        revisions = {
            item.get("revision")
            for item in value
            if isinstance(item, dict) and "revision" in item
        }
        if len(revisions) == 1:
            return next(iter(revisions))
    return fallback


def _is_superseded_catalog_marker(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == "superseded"


def _compact_replay_budget(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    aliases = (
        ("remaining_global_actions", "remaining_global_actions"),
        ("task_actions_remaining", "remaining_global_actions"),
        ("remaining_node_actions", "remaining_node_actions"),
        ("node_actions_remaining", "remaining_node_actions"),
    )
    for source, target in aliases:
        if source not in value or target in result:
            continue
        candidate = value[source]
        if isinstance(candidate, bool):
            continue
        try:
            result[target] = max(0, int(candidate))
        except (TypeError, ValueError):
            continue
    return result or None


def _normalize_tools(tools: list[NativeToolSpec] | None) -> list[NativeToolSpec]:
    values = list(tools or [])
    if not all(isinstance(item, NativeToolSpec) for item in values):
        raise TypeError("tools must contain NativeToolSpec values")
    names = [item.name for item in values]
    if len(set(names)) != len(names):
        raise ValueError("native tool names must be unique within a turn")
    return values


def _protocol_repair_message(
    *,
    tools: list[NativeToolSpec],
    failure: AgentProtocolError,
) -> str:
    expected = (
        "Return exactly one native tool call using the only offered submit tool."
        if len(tools) == 1 and tools[0].name.startswith("submit_")
        else "Return exactly one native tool call using one of the offered tools."
    )
    return (
        "PROTOCOL REPAIR REQUIRED. The previous response was rejected and no action was executed. "
        f"Violation: {failure.code}. {expected} Do not encode action arguments in prose or Markdown."
    )


def _tool_call_dict(call: NativeToolCall) -> dict[str, Any]:
    return {"call_id": call.call_id, "name": call.name, "arguments": copy.deepcopy(call.arguments)}


def _turn_dict(turn: AgentTurn) -> dict[str, Any]:
    return {
        "content": turn.content,
        **_reasoning_summary(turn.reasoning_content),
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
        "reasoning_content_present",
        "reasoning_content_chars",
        "reasoning_content_sha256",
    }
    for key, value in metadata.items():
        lowered = str(key).lower()
        if "reasoning" in lowered and lowered not in allowed_reasoning:
            continue
        safe[str(key)] = copy.deepcopy(value)
    return safe


def _assistant_replay_message(turn: AgentTurn) -> AgentMessage:
    if turn.replay_assistant_message:
        message = copy.deepcopy(turn.replay_assistant_message)
        if message.get("role") != "assistant":
            raise AgentProtocolError(
                "provider_invalid_response",
                "provider replay envelope must have assistant role",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        # Parsed fields remain authoritative for consumers; the replay envelope
        # is provider-owned only for exact DeepSeek history transport.
        message["content"] = turn.content
        message["reasoning_content"] = turn.reasoning_content
        return message
    message: AgentMessage = {
        "role": "assistant",
        "content": turn.content,
        "reasoning_content": turn.reasoning_content,
    }
    if turn.tool_calls:
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
            for call in turn.tool_calls
        ]
    return message


def _reasoning_summary(reasoning_content: str) -> dict[str, Any]:
    encoded = reasoning_content.encode("utf-8")
    return {
        "reasoning_content_present": bool(reasoning_content),
        "reasoning_content_chars": len(reasoning_content),
        "reasoning_content_sha256": (
            hashlib.sha256(encoded).hexdigest() if reasoning_content else ""
        ),
    }


def _safe_messages_snapshot(messages: list[AgentMessage]) -> list[AgentMessage]:
    safe: list[AgentMessage] = []
    for original in messages:
        message = copy.deepcopy(original)
        reasoning = message.pop("reasoning_content", None)
        if isinstance(reasoning, str):
            message.update(_reasoning_summary(reasoning))
        safe.append(message)
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
