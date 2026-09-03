"""Deterministic, no-API Agent and Harness fixtures for the v3 full-chain smoke.

These fakes deliberately implement the same public protocols as production
components.  They do not bypass native ToolCall schemas, action-catalog
revisions, grounding evidence, validation, or usage accounting.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from atomic_skillgraph.agents.protocol import (
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    validate_schema_instance,
)
from atomic_skillgraph.agents.usage import UsageBucket, UsageLedger
from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
from atomic_skillgraph.core.contracts import ContractSource, SemanticPredicate, TaskContract
from atomic_skillgraph.core.errors import AtomicSkillGraphError, FailureLayer
from atomic_skillgraph.core.results import (
    AtomicEffectResolution,
    PrimitiveToolStep,
    ValidationResult,
)
from atomic_skillgraph.harness.action_catalog import HarnessActionCatalog
from atomic_skillgraph.harness.protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
    PredicateSpec,
)
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher


_LEARNED_TOOL = "$learned"
_RUNTIME_KINDS = {
    "runtime_preparation",
    "runtime_seeded",
    "runtime_dynamic",
    "runtime_provisional_seeded",
    "runtime_dynamic_cold_start_continuation",
}
_INITIAL_BUCKET = {
    "planner": UsageBucket.PLANNER_P1,
    "cold_start_c1": UsageBucket.COLD_START_C1,
    "cold_start_c1_repair": UsageBucket.COLD_START_C1_REPAIR,
    "runtime_preparation": UsageBucket.RUNTIME_PREPARATION,
    "runtime_seeded": UsageBucket.RUNTIME_SEEDED,
    "runtime_dynamic": UsageBucket.RUNTIME_DYNAMIC,
    "runtime_provisional_seeded": UsageBucket.RUNTIME_PROVISIONAL_SEEDED,
    "runtime_dynamic_cold_start_continuation": (
        UsageBucket.RUNTIME_DYNAMIC_COLD_START_CONTINUATION
    ),
    "extractor": UsageBucket.EXTRACTOR_E1,
    "failure_extractor_f1": UsageBucket.FAILURE_EXTRACTOR_F1,
    "failure_extractor_f2": UsageBucket.FAILURE_EXTRACTOR_F2,
}
_PROVIDER_STAGES = (
    "planner",
    "runtime_preparation",
    "runtime_seeded",
    "runtime_dynamic",
    "extractor",
)
_PROVIDER_STAGE_ALIASES = {
    "cold_start_c1": "planner",
    "cold_start_c1_repair": "planner",
    "runtime_provisional_seeded": "runtime_seeded",
    "runtime_dynamic_cold_start_continuation": "runtime_dynamic",
    "failure_extractor_f1": "extractor",
    "failure_extractor_f2": "extractor",
}


@dataclass(frozen=True)
class FakeProviderRequest:
    """Immutable capture of one :class:`AgentProvider.complete` request.

    The stored values are defensive copies of exactly the replay-session
    ``messages`` and native ``tools``.  A structured
    :class:`FakeReply` may use a request-aware callback when its value depends
    on code-generated authority, such as Extractor E2 occurrence ids.
    """

    messages: tuple[dict[str, Any], ...]
    tools: tuple[NativeToolSpec, ...]

    @property
    def last_user_input(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else ""
        return ""

    @property
    def policy_context(self) -> dict[str, Any]:
        """Return the newest ``POLICY_CONTEXT_JSON`` object in replay history."""

        marker = "\n\nPOLICY_CONTEXT_JSON\n"
        for message in reversed(self.messages):
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, str):
                continue
            if marker not in content:
                continue
            value = json.loads(content.split(marker, 1)[1])
            if not isinstance(value, dict):
                raise AssertionError("POLICY_CONTEXT_JSON must contain one JSON object")
            return value
        raise AssertionError("provider request has no POLICY_CONTEXT_JSON user message")


StructuredReplyFactory = Callable[[FakeProviderRequest], Any]


@dataclass(frozen=True)
class FakeReply:
    """One deterministic provider reply.

    ``tool_name="$learned"`` resolves to the single offered Implementation
    Invocation name, keeping fixtures independent of its content-derived hash.
    """

    structured_value: Any = None
    tool_name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    content: str = ""
    prompt_tokens: int = 7
    completion_tokens: int = 3
    reasoning_tokens: int | None = 1
    latency_ms: float = 1.0

    @classmethod
    def structured(
        cls,
        value: Any,
        *,
        prompt_tokens: int = 7,
        completion_tokens: int = 3,
        reasoning_tokens: int | None = 1,
    ) -> "FakeReply":
        return cls(
            structured_value=copy.deepcopy(value),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @classmethod
    def tool(
        cls,
        name: str,
        arguments: Mapping[str, Any],
        *,
        prompt_tokens: int = 7,
        completion_tokens: int = 3,
        reasoning_tokens: int | None = 1,
    ) -> "FakeReply":
        return cls(
            tool_name=name,
            arguments=copy.deepcopy(dict(arguments)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def materialize(
        self,
        *,
        call_id: str,
        tools: Sequence[NativeToolSpec],
        request: FakeProviderRequest | None = None,
    ) -> AgentTurn:
        if self.tool_name:
            name = self.tool_name
            if name == _LEARNED_TOOL:
                names = [item.name for item in tools if item.name.startswith("invoke_impl_")]
                if len(names) != 1:
                    raise AssertionError(
                        f"expected exactly one learned invocation, got {names!r}"
                    )
                name = names[0]
            offered = {item.name: item for item in tools}
            if name not in offered:
                raise AssertionError(f"scripted tool {name!r} was not offered: {sorted(offered)!r}")
            arguments = copy.deepcopy(dict(self.arguments))
        else:
            submit_tools = [item for item in tools if item.name.startswith("submit_")]
            if len(submit_tools) != 1:
                raise AssertionError(
                    "structured fake reply requires exactly one offered native submit tool"
                )
            name = submit_tools[0].name
            if callable(self.structured_value):
                if request is None:
                    raise AssertionError("request-aware fake reply requires a provider request")
                value = self.structured_value(request)
            else:
                value = self.structured_value
            arguments = copy.deepcopy(value)
        offered = {item.name: item for item in tools}
        if name not in offered:
            raise AssertionError(f"scripted tool {name!r} was not offered: {sorted(offered)!r}")
        if not isinstance(arguments, dict):
            raise AssertionError("native fake ToolCall arguments must be an object")
        validate_schema_instance(arguments, offered[name].input_schema)
        calls = [NativeToolCall(call_id, name, arguments)]
        content = self.content
        finish_reason = "tool_calls"
        reasoning_content = f"deterministic reasoning for {call_id}"
        replay = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False,
                    ),
                },
            }],
        }
        return AgentTurn(
            content=content,
            tool_calls=calls,
            finish_reason=finish_reason,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            latency_ms=self.latency_ms,
            provider_metadata={
                "provider": "fake",
                "model": "deterministic-v3",
                "reasoning_tokens_status": (
                    "available" if self.reasoning_tokens is not None else "unavailable"
                ),
                "reasoning_tokens_source": "fixture",
                "reasoning_tokens_in_completion": True,
            },
            reasoning_content=reasoning_content,
            replay_assistant_message=replay,
        )


def _capture_provider_request(
    messages: list[dict[str, Any]],
    tools: list[NativeToolSpec] | None,
) -> FakeProviderRequest:
    """Validate and copy the concrete request shape emitted by ReplayAgentSession."""

    if not isinstance(messages, list) or not messages:
        raise AssertionError("fake provider requires a non-empty replay message list")
    copied_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise AssertionError(f"provider message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise AssertionError(f"provider message {index} has invalid role {role!r}")
        if not isinstance(message.get("content"), str):
            raise AssertionError(f"provider message {index} content must be a string")
        if role == "tool":
            if not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"]:
                raise AssertionError("replay tool message requires tool_call_id")
            if "name" in message:
                raise AssertionError("DeepSeek replay tool messages must omit name")
            try:
                json.loads(message["content"])
            except (TypeError, ValueError) as exc:
                raise AssertionError("replay tool message content must be JSON") from exc
        if role == "assistant" and "tool_calls" in message:
            if not isinstance(message.get("reasoning_content"), str):
                raise AssertionError("DeepSeek assistant ToolCall replay requires reasoning_content")
            native_calls = message["tool_calls"]
            if not isinstance(native_calls, list) or len(native_calls) != 1:
                raise AssertionError("replay assistant message must contain exactly one native ToolCall")
            native = native_calls[0]
            function = native.get("function") if isinstance(native, dict) else None
            if (
                not isinstance(native, dict)
                or not isinstance(native.get("id"), str)
                or native.get("type") != "function"
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
            ):
                raise AssertionError("replay assistant ToolCall has an invalid provider schema")
            try:
                arguments = json.loads(function["arguments"])
            except (TypeError, ValueError) as exc:
                raise AssertionError("replay assistant ToolCall arguments must be JSON") from exc
            if not isinstance(arguments, dict):
                raise AssertionError("replay assistant ToolCall arguments must be a JSON object")
        copied_messages.append(copy.deepcopy(message))
    if copied_messages[0]["role"] != "system":
        raise AssertionError("replay provider history must begin with a system message")
    try:
        json.dumps(copied_messages, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AssertionError("replay provider messages must be JSON serializable") from exc

    if tools is None:
        copied_tools: tuple[NativeToolSpec, ...] = ()
    else:
        if not isinstance(tools, list) or not all(isinstance(item, NativeToolSpec) for item in tools):
            raise AssertionError("fake provider tools must be a list of NativeToolSpec")
        copied_tools = tuple(copy.deepcopy(tools))
    return FakeProviderRequest(tuple(copied_messages), copied_tools)


class ScriptedAgentProvider:
    """Deterministic :class:`AgentProvider` consumed by ReplayAgentSession.

    Replies are consumed globally in provider-call order, which is exactly what
    a stage-specific provider injected into :class:`AtomicSkillGraphSystem`
    observes across fresh replay sessions.
    """

    def __init__(
        self,
        replies: Iterable[FakeReply] = (),
        *,
        provider_id: str = "default",
    ) -> None:
        self.provider_id = str(provider_id)
        if not self.provider_id:
            raise ValueError("fake provider_id must be non-empty")
        self._safe_id = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in self.provider_id
        )[:48] or "default"
        self._replies: deque[FakeReply] = deque()
        self._requests: list[FakeProviderRequest] = []
        self.enqueue(*list(replies))

    @property
    def remaining_replies(self) -> int:
        return len(self._replies)

    @property
    def requests(self) -> tuple[FakeProviderRequest, ...]:
        return tuple(copy.deepcopy(self._requests))

    def enqueue(self, *replies: FakeReply) -> None:
        if not all(isinstance(reply, FakeReply) for reply in replies):
            raise TypeError("ScriptedAgentProvider accepts only FakeReply values")
        self._replies.extend(replies)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        request = _capture_provider_request(messages, tools)
        if not self._replies:
            raise AssertionError(
                f"no scripted provider reply remains for stage {self.provider_id!r}"
            )
        reply = self._replies.popleft()
        call_index = len(self._requests)
        self._requests.append(request)
        return reply.materialize(
            call_id=f"call_{self._safe_id}_{call_index:06d}",
            tools=request.tools,
            request=request,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "model": "deterministic-v3",
            "provider_id": self.provider_id,
            "call_count": len(self._requests),
            "remaining_replies": len(self._replies),
        }

    def assert_exhausted(self) -> None:
        if self._replies:
            raise AssertionError(
                f"unconsumed fake provider replies for {self.provider_id}: {len(self._replies)}"
            )


class FakeProviderSet(MappingABC[str, ScriptedAgentProvider]):
    """Stage-keyed provider Mapping accepted directly by AtomicSkillGraphSystem."""

    def __init__(
        self,
        scripts: Mapping[str, Iterable[FakeReply]] | None = None,
    ) -> None:
        self._providers = {
            stage: ScriptedAgentProvider(provider_id=stage)
            for stage in _PROVIDER_STAGES
        }
        for stage, replies in dict(scripts or {}).items():
            self.enqueue(stage, replies)

    def __getitem__(self, stage: str) -> ScriptedAgentProvider:
        canonical = _PROVIDER_STAGE_ALIASES.get(str(stage), str(stage))
        return self._providers[canonical]

    def __iter__(self) -> Iterator[str]:
        return iter(self._providers)

    def __len__(self) -> int:
        return len(self._providers)

    def provider(self, stage: str) -> ScriptedAgentProvider:
        try:
            return self[str(stage)]
        except KeyError as exc:
            raise KeyError(f"unknown fake provider stage: {stage!r}") from exc

    def enqueue(self, stage: str, replies: Iterable[FakeReply]) -> None:
        self.provider(stage).enqueue(*list(replies))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {stage: provider.snapshot() for stage, provider in self._providers.items()}

    def assert_exhausted(self) -> None:
        remaining = {
            stage: provider.remaining_replies
            for stage, provider in self._providers.items()
            if provider.remaining_replies
        }
        if remaining:
            raise AssertionError(f"unconsumed stage provider replies: {remaining!r}")


class ScriptedAgentSession:
    """Small strict AgentSession whose turns are supplied by :class:`FakeReply`."""

    def __init__(
        self,
        session_id: str,
        session_kind: str,
        replies: Iterable[FakeReply],
        usage_ledger: UsageLedger,
    ) -> None:
        self._session_id = session_id
        self.session_kind = session_kind
        self._replies = deque(replies)
        self._usage_ledger = usage_ledger
        self._usage_bucket = _INITIAL_BUCKET.get(session_kind, UsageBucket.UNATTRIBUTED)
        self._turn_index = 0
        self._pending: NativeToolCall | None = None
        self._seen_call_ids: set[str] = set()
        self._finalized = False
        self._messages: list[dict[str, Any]] = []
        self._tool_results: list[dict[str, Any]] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def remaining_replies(self) -> int:
        return len(self._replies)

    @property
    def tool_results(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._tool_results))

    def enqueue(self, *replies: FakeReply) -> None:
        self._replies.extend(replies)

    def set_usage_bucket(self, bucket: UsageBucket | str) -> None:
        if self._pending is not None:
            raise AssertionError("cannot change fake usage bucket with a pending ToolCall")
        self._usage_bucket = UsageBucket(bucket)

    def next_turn(
        self,
        user_input: str | None,
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        if self._finalized:
            raise AssertionError("fake AgentSession was finalized")
        if self._pending is not None:
            raise AssertionError("pending fake ToolCall requires submit_tool_result")
        if user_input is not None:
            self._messages.append({"role": "user", "content": str(user_input)})
        return self._issue(tools or [])

    def acknowledge_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        if self._pending is None or self._pending.call_id != call_id:
            raise AssertionError("fake acknowledgement does not match the pending ToolCall")
        self._append_tool_result(call_id, result)

    def finalize_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        self.acknowledge_tool_result(call_id, result)
        self._finalized = True

    def submit_tool_result(
        self,
        call_id: str,
        result: dict[str, Any],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        if self._pending is None or self._pending.call_id != call_id:
            raise AssertionError("fake ToolCall result does not match the pending call")
        self._append_tool_result(call_id, result)

        # A rejected learned invocation needs an actual repair turn.  A
        # non-terminal environment action also needs another decision.  The
        # Runtime currently asks for a return value even when it immediately
        # exits after terminal success; return a zero-cost sentinel in that case
        # rather than fabricating an extra provider call.
        repair = bool(result.get("repairable"))
        continue_environment = (
            "new_revision" in result
            and not bool(result.get("done"))
            and not bool(result.get("won"))
        )
        if repair or continue_environment:
            return self._issue(tools or [])
        return AgentTurn("", [], "stop", 0, 0, 0, 0, 0.0, {"provider": "fake"})

    def _append_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        if self._pending is None:
            raise AssertionError("fake session has no pending ToolCall")
        self._tool_results.append(
            {"call_id": call_id, "tool_name": self._pending.name, "result": copy.deepcopy(result)}
        )
        self._messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                result, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ),
        })
        self._pending = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "session_kind": self.session_kind,
            "turn_count": self._turn_index,
            "usage_bucket": self._usage_bucket.value,
            "pending_tool_call": (
                None
                if self._pending is None
                else {
                    "call_id": self._pending.call_id,
                    "name": self._pending.name,
                    "arguments": copy.deepcopy(self._pending.arguments),
                }
            ),
            "finalized": self._finalized,
            "messages": _sanitized_fake_messages(self._messages),
            "tool_results": copy.deepcopy(self._tool_results),
            "remaining_replies": len(self._replies),
            "provider": {"provider": "fake", "model": "deterministic-v3"},
        }

    def _issue(
        self,
        tools: Sequence[NativeToolSpec],
    ) -> AgentTurn:
        if not self._replies:
            raise AssertionError(
                f"no scripted reply remains for {self._session_id} ({self.session_kind})"
            )
        reply = self._replies.popleft()
        call_id = f"call_{self._session_id}_{self._turn_index:03d}"
        turn = reply.materialize(
            call_id=call_id,
            tools=tools,
            request=FakeProviderRequest(
                tuple(copy.deepcopy(self._messages)),
                tuple(copy.deepcopy(tools)),
            ),
        )
        self._usage_ledger.record_turn(
            session_id=self._session_id,
            turn_index=self._turn_index,
            bucket=self._usage_bucket,
            turn=turn,
            event_id=f"usage_{self._session_id}_{self._turn_index:03d}",
        )
        self._turn_index += 1
        self._messages.append(copy.deepcopy(turn.replay_assistant_message))
        if turn.tool_calls:
            call = turn.tool_calls[0]
            if call.call_id in self._seen_call_ids:
                raise AssertionError(f"duplicate fake ToolCall id: {call.call_id}")
            self._seen_call_ids.add(call.call_id)
            self._pending = call
        return turn


def _sanitized_fake_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for original in messages:
        message = copy.deepcopy(dict(original))
        reasoning = message.pop("reasoning_content", None)
        if isinstance(reasoning, str):
            message.update({
                "reasoning_content_present": bool(reasoning),
                "reasoning_content_chars": len(reasoning),
                "reasoning_content_sha256": (
                    sha256(reasoning.encode("utf-8")).hexdigest() if reasoning else ""
                ),
            })
        values.append(message)
    return values


class FakeAgentFactory:
    """Queue one script per future Planner/Runtime/Extractor session."""

    def __init__(self, usage_ledger: UsageLedger | None = None) -> None:
        self.usage_ledger = usage_ledger or UsageLedger()
        self._scripts: dict[str, deque[list[FakeReply]]] = defaultdict(deque)
        self._counts: dict[str, int] = defaultdict(int)
        self.sessions: list[ScriptedAgentSession] = []

    def enqueue(self, session_kind: str, replies: Iterable[FakeReply]) -> None:
        self._scripts[str(session_kind)].append(list(replies))

    def new_session(
        self,
        session_kind: str,
        replies: Iterable[FakeReply] = (),
    ) -> ScriptedAgentSession:
        session_kind = str(session_kind)
        self._counts[session_kind] += 1
        session = ScriptedAgentSession(
            f"fake_{session_kind}_{self._counts[session_kind]:03d}",
            session_kind,
            replies,
            self.usage_ledger,
        )
        self.sessions.append(session)
        return session

    def __call__(self, first: Any, second: Any) -> ScriptedAgentSession:
        session_kind = (
            first
            if isinstance(first, str) and first in _INITIAL_BUCKET
            else "planner"
        )
        try:
            replies = self._scripts[session_kind].popleft()
        except IndexError as exc:
            raise AssertionError(f"no queued fake session for {session_kind}") from exc
        return self.new_session(session_kind, replies)

    def sessions_of(self, session_kind: str) -> list[ScriptedAgentSession]:
        return [item for item in self.sessions if item.session_kind == session_kind]

    def assert_exhausted(self) -> None:
        queued = {key: len(value) for key, value in self._scripts.items() if value}
        remaining = {
            item.session_id: item.remaining_replies
            for item in self.sessions
            if item.remaining_replies
        }
        if queued or remaining:
            raise AssertionError(
                f"unconsumed fake scripts: queued={queued!r}, session_replies={remaining!r}"
            )


class FakeValidatorChannel:
    """Validator-only full-state channel for :class:`FakeHarness`."""

    validation_strength = "deterministic_full_state"

    def __init__(self) -> None:
        self.target_item = ""
        self.role_bindings: dict[str, Any] = {}
        self.facts: list[dict[str, Any]] = []
        self.revision = 0
        self.won = False

    def reset(self, task: HarnessTask) -> None:
        self.target_item = str(task.context["target_item"])
        self.role_bindings = dict(task.context.get("semantic_bindings", {}))
        self.role_bindings.setdefault("item", self.target_item)
        self.facts = []
        self.revision = 0
        self.won = False

    def record_fact(self, predicate: str, args: Mapping[str, Any], revision: int) -> None:
        fact = {
            "predicate": str(predicate),
            "args": copy.deepcopy(dict(args)),
            "revision": int(revision),
            "witness_ref": f"fake:{predicate}:{revision}:{len(self.facts)}",
        }
        if not any(
            item["predicate"] == fact["predicate"] and item["args"] == fact["args"]
            for item in self.facts
        ):
            self.facts.append(fact)
        self.revision = int(revision)

    def snapshot(self) -> dict[str, Any]:
        return {
            "facts": copy.deepcopy(self.facts),
            "revision": self.revision,
            "won": self.won,
        }

    def resolve_atomic_effect(
        self, request: dict[str, Any],
    ) -> AtomicEffectResolution:
        """Deterministic action-fact resolver matching the production contract."""

        effects = list(request.get("effects") or [])
        if not effects:
            return AtomicEffectResolution(
                False,
                failure_code="atomic_effect_violation",
                message="Atomic effect resolution requires at least one effect",
            )
        if "current_revision" not in request:
            return AtomicEffectResolution(
                False,
                failure_code="atomic_effect_revision_invalid",
                message="Atomic effect resolution requires an explicit revision",
            )
        try:
            if isinstance(request["current_revision"], bool):
                raise ValueError("boolean revision")
            requested_revision = int(request["current_revision"])
        except (TypeError, ValueError):
            return AtomicEffectResolution(
                False,
                failure_code="atomic_effect_revision_invalid",
                message="Atomic effect resolution revision is invalid",
            )
        if requested_revision != self.revision:
            return AtomicEffectResolution(
                False,
                failure_code="stale_atomic_effect_witness",
                message="Atomic effect resolution revision is stale",
            )
        known = {
            str(role): value
            for role, value in dict(request.get("known_bindings") or {}).items()
            if value not in (None, "")
        }
        anchors = {
            str(role): value
            for role, value in dict(request.get("semantic_anchors") or {}).items()
            if value not in (None, "")
        }
        preferred = [
            value for value in list(request.get("preferred_values") or [])
            if value not in (None, "")
        ]
        per_effect: list[list[tuple[dict[str, Any], list[str]]]] = []
        checks: dict[str, bool] = {}
        for index, raw_effect in enumerate(effects):
            predicate = (
                raw_effect.predicate
                if isinstance(raw_effect, SemanticPredicate)
                else str(raw_effect["predicate"])
            )
            raw_args = (
                raw_effect.args
                if isinstance(raw_effect, SemanticPredicate)
                else dict(raw_effect.get("args") or {})
            )
            cardinality = max(1, int(
                raw_effect.cardinality
                if isinstance(raw_effect, SemanticPredicate)
                else raw_effect.get("cardinality", 1)
            ))
            distinct_by = str(
                raw_effect.distinct_by
                if isinstance(raw_effect, SemanticPredicate)
                else raw_effect.get("distinct_by", "")
            )
            parsed_args: dict[str, tuple[str, Any]] = {}
            for predicate_role, raw in raw_args.items():
                expression: BindingExpression | None = None
                if isinstance(raw, BindingExpression):
                    expression = raw
                elif isinstance(raw, MappingABC) and "kind" in raw:
                    try:
                        expression = BindingExpression.from_dict(dict(raw))
                    except (KeyError, TypeError, ValueError):
                        return AtomicEffectResolution(
                            False,
                            checks=checks,
                            failure_code="atomic_effect_expression_invalid",
                            message="Atomic effect contains an invalid binding expression",
                        )
                if expression is not None:
                    if expression.kind is BindingExprKind.CONSTANT:
                        parsed_args[predicate_role] = (
                            "constant", expression.constant,
                        )
                    elif expression.kind is BindingExprKind.SKILL_INPUT:
                        parsed_args[predicate_role] = (
                            "role", str(expression.source_role),
                        )
                    else:
                        return AtomicEffectResolution(
                            False,
                            checks=checks,
                            failure_code="atomic_effect_expression_unsupported",
                            message=(
                                "Atomic effects may only use skill-input or "
                                "constant expressions"
                            ),
                        )
                elif isinstance(raw, str) and raw.startswith("$"):
                    parsed_args[predicate_role] = ("role", raw[1:])
                else:
                    parsed_args[predicate_role] = ("constant", raw)
            source_roles = {
                str(value)
                for kind, value in parsed_args.values()
                if kind == "role" and str(value)
            }
            if any(kind == "role" and not str(value) for kind, value in parsed_args.values()):
                return AtomicEffectResolution(
                    False,
                    checks=checks,
                    failure_code="atomic_effect_expression_invalid",
                    message="Atomic effect references an empty input role",
                )

            candidates: list[
                tuple[dict[str, Any], dict[str, Any], str]
            ] = []
            for fact in self.facts:
                if fact["predicate"] != predicate:
                    continue
                resolved: dict[str, Any] = {}
                compatible = True
                for predicate_role, (kind, expected) in parsed_args.items():
                    actual = fact["args"].get(predicate_role)
                    if actual in (None, ""):
                        compatible = False
                        break
                    if kind == "constant":
                        compatible &= actual == expected
                        continue
                    role = str(expected)
                    if role in known and known[role] != actual:
                        compatible = False
                    if role in anchors and anchors[role] != actual:
                        compatible = False
                    resolved[role] = actual
                if compatible:
                    candidates.append((
                        resolved,
                        dict(fact["args"]),
                        str(fact["witness_ref"]),
                    ))

            candidate_values = {
                role: {
                    candidate[0][role]
                    for candidate in candidates
                    if role in candidate[0]
                }
                for role in source_roles
            }
            filtered: list[
                tuple[dict[str, Any], dict[str, Any], str]
            ] = []
            for assignment, actual_args, witness in candidates:
                eligible = True
                for role, value in assignment.items():
                    if role in known or role in anchors:
                        continue
                    preferred_match = value in preferred
                    other_arguments = [
                        (kind, expected)
                        for kind, expected in parsed_args.values()
                        if not (kind == "role" and str(expected) == role)
                    ]
                    other_arguments_anchored = bool(other_arguments) and all(
                        kind == "constant"
                        or str(expected) in known
                        or str(expected) in anchors
                        for kind, expected in other_arguments
                    )
                    uniquely_constrained = (
                        len(candidate_values.get(role, set())) == 1
                        and other_arguments_anchored
                    )
                    if not preferred_match and not uniquely_constrained:
                        eligible = False
                        break
                if eligible:
                    filtered.append((assignment, actual_args, witness))

            groups_by_assignment: dict[
                tuple[tuple[str, str], ...],
                tuple[dict[str, Any], list[str]],
            ] = {}
            for group in combinations(filtered, cardinality):
                if distinct_by and len({
                    item[1].get(distinct_by)
                    for item in group
                    if item[1].get(distinct_by) not in (None, "")
                }) < cardinality:
                    continue
                merged: dict[str, Any] = {}
                refs: list[str] = []
                consistent = True
                for assignment, _arguments, witness in group:
                    refs.append(witness)
                    for role, value in assignment.items():
                        if role in merged and merged[role] != value:
                            consistent = False
                            break
                        merged[role] = value
                    if not consistent:
                        break
                if not consistent:
                    continue
                key = tuple(sorted(
                    (role, repr(value)) for role, value in merged.items()
                ))
                if key in groups_by_assignment:
                    prior, prior_refs = groups_by_assignment[key]
                    groups_by_assignment[key] = (
                        prior,
                        list(dict.fromkeys([*prior_refs, *refs])),
                    )
                else:
                    groups_by_assignment[key] = (
                        merged,
                        list(dict.fromkeys(refs)),
                    )
            groups = list(groups_by_assignment.values())
            checks[f"effect_{index}_witness_resolved"] = bool(groups)
            per_effect.append(groups)

        if not all(per_effect):
            return AtomicEffectResolution(
                False,
                checks=checks,
                failure_code="atomic_effect_violation",
                message="Fake action facts do not witness every Atomic effect",
            )

        assignments: list[tuple[dict[str, Any], list[str]]] = [({}, [])]
        for groups in per_effect:
            next_assignments: list[tuple[dict[str, Any], list[str]]] = []
            for existing, refs in assignments:
                for resolved, witness_refs in groups:
                    if any(
                        role in existing and existing[role] != value
                        for role, value in resolved.items()
                    ):
                        continue
                    next_assignments.append((
                        {**existing, **resolved},
                        list(dict.fromkeys([*refs, *witness_refs])),
                    ))
            assignments = next_assignments
        checks["joint_effect_assignment_exists"] = bool(assignments)
        if not assignments:
            return AtomicEffectResolution(
                False,
                checks=checks,
                failure_code="atomic_effect_violation",
                message="Atomic effects have no jointly consistent witness assignment",
            )
        unique: dict[
            tuple[tuple[str, str], ...],
            tuple[dict[str, Any], list[str]],
        ] = {}
        for assignment, refs in assignments:
            key = tuple(sorted(
                (role, repr(value)) for role, value in assignment.items()
            ))
            if key in unique:
                prior, prior_refs = unique[key]
                unique[key] = (
                    prior,
                    list(dict.fromkeys([*prior_refs, *refs])),
                )
            else:
                unique[key] = (assignment, refs)
        if len(unique) != 1:
            return AtomicEffectResolution(
                False,
                checks=checks,
                failure_code="atomic_effect_witness_ambiguous",
                message="Multiple concrete Atomic effect witnesses remain",
            )
        resolved, refs = next(iter(unique.values()))
        merged = {**known, **resolved}
        outputs = {
            str(item["output_role"]): merged[str(item["input_role"])]
            for item in request.get("output_identity") or []
            if str(item.get("input_role", "")) in merged
        }
        output_roles = {
            str(spec.get("name", ""))
            if isinstance(spec, MappingABC)
            else str(getattr(spec, "name", ""))
            for spec in request.get("output_specs") or []
        }
        witness_set = set(refs)
        witness_arguments = [
            dict(fact["args"])
            for fact in self.facts
            if str(fact["witness_ref"]) in witness_set
        ]
        for output_role in sorted(output_roles - set(outputs)):
            exact: dict[str, Any] = {}
            for arguments in witness_arguments:
                if output_role in arguments:
                    value = arguments[output_role]
                    exact.setdefault(repr(value), value)
            if len(exact) == 1:
                outputs[output_role] = next(iter(exact.values()))
        checks["assignment_unique"] = True
        return AtomicEffectResolution(
            True,
            resolved_bindings=resolved,
            output_candidates=outputs,
            witness_refs=refs,
            checks=checks,
        )

    def validate_atomic_effect(self, request: dict[str, Any]) -> ValidationResult:
        bindings = dict(request.get("bindings", {}))
        bindings.update(request.get("output_candidates", {}))
        passed, refs = self._effects_hold(request.get("effects", []), bindings)
        return ValidationResult(
            "atomic",
            passed,
            {"effects_witnessed": passed},
            [] if passed else ["atomic_effect_violation"],
            [] if passed else ["fake world does not witness the requested Atomic effect"],
            refs,
            f"revision:{max(0, self.revision - 1)}",
            f"revision:{self.revision}",
        )

    def validate_task_contract(self, contract: TaskContract) -> ValidationResult:
        passed, refs = self._effects_hold(contract.target_effects, self.role_bindings)
        return ValidationResult(
            "task_contract",
            passed,
            {"target_effects": passed},
            [] if passed else ["task_contract_mismatch"],
            [] if passed else ["fake task target effect is not yet satisfied"],
            refs,
            f"revision:0",
            f"revision:{self.revision}",
        )

    def _effects_hold(
        self,
        effects: Iterable[SemanticPredicate | Mapping[str, Any]],
        bindings: Mapping[str, Any],
    ) -> tuple[bool, list[str]]:
        effects = list(effects)
        if not effects:
            return False, []
        witnesses: list[str] = []
        for raw_effect in effects:
            predicate = (
                raw_effect.predicate
                if isinstance(raw_effect, SemanticPredicate)
                else str(raw_effect["predicate"])
            )
            raw_args = raw_effect.args if isinstance(raw_effect, SemanticPredicate) else raw_effect.get("args", {})
            cardinality = max(1, int(
                raw_effect.cardinality
                if isinstance(raw_effect, SemanticPredicate)
                else raw_effect.get("cardinality", 1)
            ))
            distinct_by = str(
                raw_effect.distinct_by
                if isinstance(raw_effect, SemanticPredicate)
                else raw_effect.get("distinct_by", "")
            )
            expected = {
                role: self._resolve(value, bindings)
                for role, value in raw_args.items()
            }
            matches = [
                fact
                for fact in self.facts
                if fact["predicate"] == predicate
                and all(fact["args"].get(role) == value for role, value in expected.items())
            ]
            if distinct_by:
                selected: list[dict[str, Any]] = []
                seen: set[Any] = set()
                for fact in matches:
                    value = fact["args"].get(distinct_by)
                    if value in (None, "") or value in seen:
                        continue
                    seen.add(value)
                    selected.append(fact)
                    if len(selected) == cardinality:
                        break
            else:
                selected = matches[:cardinality]
            if len(selected) < cardinality:
                return False, []
            witnesses.extend(str(fact["witness_ref"]) for fact in selected)
        return True, witnesses

    def _resolve(self, value: Any, bindings: Mapping[str, Any]) -> Any:
        if isinstance(value, Mapping) and "kind" in value:
            value = BindingExpression.from_dict(dict(value))
        if isinstance(value, BindingExpression):
            if value.kind is BindingExprKind.CONSTANT:
                return value.constant
            if value.kind is BindingExprKind.SKILL_INPUT:
                return bindings.get(
                    value.source_role,
                    self.role_bindings.get(value.source_role),
                )
            return f"__unsupported_effect_expression__:{value.kind.value}"
        if isinstance(value, str) and value.startswith("$"):
            role = value[1:]
            return bindings.get(role, self.role_bindings.get(role))
        return value


class FakeHarness:
    """Revisioned one-item world used by the four deterministic smoke episodes."""

    profile_name = "fake_v3"

    def __init__(self) -> None:
        self._catalog = HarnessActionCatalog(self._parse_action)
        self._validator = FakeValidatorChannel()
        self._task: HarnessTask | None = None
        self._revision = 0
        self._held = False
        self._observed = False
        self._done = False
        self._won = False

    @staticmethod
    def _parse_action(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        return (
            str(raw["action_type"]),
            copy.deepcopy(dict(raw.get("arguments", {}))),
            str(raw.get("display_text", raw["action_type"])),
            copy.deepcopy(dict(raw.get("metadata", {}))),
        )

    @property
    def current_task(self) -> HarnessTask | None:
        return self._task

    def reset(self, task: HarnessTask) -> HarnessActionResult:
        if "target_item" not in task.context:
            raise ValueError("FakeHarness task requires context.target_item")
        self._task = task
        self._revision = 0
        self._held = self._observed = self._done = self._won = False
        self._validator.reset(task)
        catalog = self._replace_catalog()
        return HarnessActionResult(
            True,
            f"Target {task.context['target_item']} is available.",
            False,
            False,
            self._revision,
            catalog,
            {"reset": True},
        )

    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog.items()

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        if self._task is None:
            raise RuntimeError("FakeHarness must be reset before action execution")
        if self._done or self._won:
            raise AtomicSkillGraphError(
                "harness_terminal_latched",
                "benchmark terminal is latched; no further env.step is allowed",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        spec = self._catalog.get(action_id, revision)
        target = str(self._task.context["target_item"])
        accepted = False
        observation = "Nothing happens."
        if spec.action_type == "TAKE" and spec.arguments == {"item": target} and not self._held:
            accepted = True
            self._held = True
            observation = f"You take {target}."
        elif spec.action_type == "EXAMINE" and spec.arguments == {"item": target} and self._held:
            accepted = True
            self._observed = True
            observation = f"You examine {target}."

        self._revision += 1
        if accepted and spec.action_type == "TAKE":
            self._validator.record_fact("agent.holds", {"object": target}, self._revision)
        if accepted and spec.action_type == "EXAMINE":
            self._validator.record_fact("object.observed", {"object": target}, self._revision)

        rescue_required = bool(self._task.context.get("requires_rescue"))
        self._won = self._held and (not rescue_required or self._observed)
        self._done = self._won
        self._validator.won = self._won
        self._validator.revision = self._revision
        catalog = self._replace_catalog()
        return HarnessActionResult(
            accepted,
            observation,
            self._done,
            self._won,
            self._revision,
            catalog,
            {"action_type": spec.action_type},
        )

    def task_contract(self, task: HarnessTask) -> TaskContract:
        return TaskContract(
            target_effects=[
                SemanticPredicate(
                    "agent.holds",
                    {
                        "object": BindingExpression(
                            BindingExprKind.SKILL_INPUT,
                            source_role="item",
                        )
                    },
                )
            ],
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="fake_v3_goal",
        )

    def contract_matcher(self) -> ExactContractMatcher:
        bindings = {}
        if self._task is not None:
            bindings["item"] = self._task.context.get("target_item")
        return ExactContractMatcher(bindings)

    def validator_channel(self) -> FakeValidatorChannel:
        return self._validator

    def semantic_predicate_schema(self) -> list[PredicateSpec]:
        return [
            PredicateSpec(
                "agent.holds", "world", ("object",),
                {"object": "entity"}, "fake_v3_action_facts",
            ),
            PredicateSpec(
                "object.observed", "evidence", ("object",),
                {"object": "entity"}, "fake_v3_action_facts",
            ),
        ]

    def primitive_action_schema(self) -> list[dict[str, Any]]:
        return [
            {"action_type": "TAKE", "argument_roles": ["item"]},
            {"action_type": "EXAMINE", "argument_roles": ["item"]},
            {"action_type": "GO_TO", "argument_roles": ["destination"]},
            {"action_type": "PUT", "argument_roles": ["item", "destination"]},
        ]

    def compile_primitive(
        self,
        primitive: PrimitiveToolStep,
        bindings: dict[str, Any],
    ) -> HarnessActionSpec:
        expected: dict[str, Any] = {}
        for role, raw_expression in primitive.argument_mapping.items():
            expression = raw_expression
            if isinstance(expression, Mapping) and "kind" in expression:
                expression = BindingExpression.from_dict(dict(expression))
            if isinstance(expression, BindingExpression):
                value = (
                    expression.constant
                    if expression.kind is BindingExprKind.CONSTANT
                    else bindings.get(expression.source_role)
                )
            else:
                value = expression
            expected[role] = value
        for spec in self.action_catalog():
            if spec.action_type == primitive.action_type and all(
                spec.arguments.get(role) == value for role, value in expected.items()
            ):
                return spec
        raise KeyError(f"no fake {primitive.action_type} affordance matches {expected!r}")

    def execute_primitive(
        self,
        primitive: PrimitiveToolStep,
        bindings: dict[str, Any],
    ) -> HarnessActionResult:
        spec = self.compile_primitive(primitive, bindings)
        return self.execute_action(spec.action_id, spec.revision)

    def replay_tool(
        self,
        task: HarnessTask,
        tool: Any,
        case: Mapping[str, Any],
    ) -> bool:
        """Replay a compiled Primitive IR at its recorded boundary state."""

        self.reset(task)
        bindings = copy.deepcopy(dict(case.get("bindings", {})))
        raw_steps = list(tool.artifact.get("steps", []))
        if raw_steps and raw_steps[0].get("action_type") == "EXAMINE":
            # EXAMINE rescue was extracted after the graph's TAKE boundary.
            # Reconstruct that validator-certified precondition without exposing
            # it to an Agent or using it as a Runtime binding oracle.
            target = str(task.context["target_item"])
            self._held = True
            self._validator.record_fact("agent.holds", {"object": target}, self._revision)
            self._replace_catalog()
        try:
            for raw in raw_steps:
                primitive = PrimitiveToolStep(
                    action_type=str(raw["action_type"]),
                    argument_mapping=copy.deepcopy(dict(raw.get("argument_mapping", {}))),
                )
                result = self.execute_primitive(primitive, bindings)
                if not result.accepted:
                    return False
        except Exception:
            return False
        return bool(raw_steps)

    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool:
        return kind in {
            "argument_exists",
            "argument_concrete",
            "harness_affordance",
            "current_context",
        } or bool(verifier_id)

    def _replace_catalog(self) -> list[HarnessActionSpec]:
        if self._task is None:
            return self._catalog.replace([], self._revision)
        target = str(self._task.context["target_item"])
        actions: list[dict[str, Any]] = []
        if not self._held:
            actions.append(
                {
                    "action_type": "TAKE",
                    "arguments": {"item": target},
                    "display_text": f"take {target}",
                }
            )
            actions.append(
                {
                    "action_type": "TAKE",
                    "arguments": {"item": "distractor_1"},
                    "display_text": "take distractor_1",
                }
            )
        elif bool(self._task.context.get("requires_rescue")) and not self._observed:
            actions.append(
                {
                    "action_type": "EXAMINE",
                    "arguments": {"item": target},
                    "display_text": f"examine {target}",
                }
            )
        return self._catalog.replace(actions, self._revision)


def fake_task(
    task_id: str,
    target_item: str,
    *,
    expose_binding: bool = True,
    requires_rescue: bool = False,
) -> HarnessTask:
    """Build one deterministic task while keeping concrete state task-local."""

    context: dict[str, Any] = {
        "target_item": target_item,
        "requires_rescue": bool(requires_rescue),
        "binding_types": {"item": "string"},
        "initial_observation": f"Target {target_item} is available.",
    }
    if expose_binding:
        context["semantic_bindings"] = {"item": target_item}
    return HarnessTask(
        task_id=task_id,
        goal=f"Hold the target item ({target_item}).",
        benchmark="fake",
        task_type="deterministic_fullchain",
        context=context,
        metadata={"task_signature": f"fake:{task_id}:{target_item}:{int(requires_rescue)}"},
    )


def planner_gap_replies() -> list[FakeReply]:
    """P1 and P1R replies that truthfully expose an empty-bank capability gap."""

    requirement = {
        "requirement_id": "req_hold",
        "intent": "hold the target item",
        "desired_effects": [{"predicate": "agent.holds", "args": {"object": "$item"}}],
        "expected_inputs": [],
        "expected_outputs": [],
        "precondition_hints": [],
        "semantic_variants": ["take target", "acquire item"],
        "required": True,
        "rationale": "The benchmark requires the target item to be held.",
    }
    payload = {"requirements": [requirement]}
    return [FakeReply.structured(payload), FakeReply.structured(payload)]


def knowledge_digest(database: Any) -> str:
    """Hash the same long-term facts as ``AtomicSkillGraphSystem.knowledge_digest``."""

    specs = {
        "metadata": ("key,value", "key"),
        "artifact_index": (
            "artifact_ref,artifact_kind,logical_id,version,content_hash,status,schema_version",
            "artifact_ref",
        ),
        "recommended_pointers": ("logical_id,artifact_ref", "logical_id"),
        "graph_edges": (
            "edge_id,source_ref,target_ref,relation,metadata_json",
            "edge_id",
        ),
        "evidence_events": (
            "event_id,schema_version,task_id,trace_id,occurrence_id,attempt_id,"
            "sequence_no,artifact_ref,artifact_kind,event_type,failure_layer,confidence,metadata_json",
            "event_id",
        ),
        "lifecycle_projection": (
            "artifact_ref,projection_json,last_event_rowid",
            "artifact_ref",
        ),
        "projection_checkpoints": (
            "projection_name,last_event_rowid",
            "projection_name",
        ),
        "provisional_artifacts": (
            "provisional_ref,contract_signature,canonical_intent,status,"
            "harness_profile,content_hash,source_trace_id,source_task_id,"
            "promoted_refs_json,schema_version,created_at,updated_at",
            "provisional_ref",
        ),
        "failure_experiences": (
            "experience_id,cluster_signature,divergence_signature,status,"
            "harness_profile,content_hash,support_count,resolved_count,"
            "schema_version,created_at,updated_at",
            "experience_id",
        ),
        "cold_start_evidence": (
            "event_id,task_id,trace_id,subject_ref,subject_kind,event_type,"
            "sequence_no,metadata_json",
            "event_id",
        ),
    }
    tables = {
        table: [list(row) for row in database.execute(
            f"SELECT {columns} FROM {table} ORDER BY {order}"
        ).fetchall()]
        for table, (columns, order) in specs.items()
    }

    data_dir = database.path.parent
    artifact_root = data_dir / "artifacts"
    files: list[dict[str, str]] = []
    if artifact_root.exists():
        for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(data_dir).as_posix(),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
    failure_root = data_dir / "failure_knowledge"
    if failure_root.exists():
        for path in sorted(item for item in failure_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(data_dir).as_posix(),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
    payload = {"files": files, "tables": tables}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "FakeAgentFactory",
    "FakeHarness",
    "FakeProviderRequest",
    "FakeProviderSet",
    "FakeReply",
    "FakeValidatorChannel",
    "ScriptedAgentProvider",
    "ScriptedAgentSession",
    "fake_task",
    "knowledge_digest",
    "planner_gap_replies",
]
