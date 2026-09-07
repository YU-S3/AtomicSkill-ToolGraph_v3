from __future__ import annotations

import copy
import hashlib
import json

import pytest

from atomic_skillgraph.agents.protocol import (
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
)
from atomic_skillgraph.agents.provider import AgentProviderError
from atomic_skillgraph.agents.session import ReplayAgentSession
from atomic_skillgraph.agents.usage import AgentBudget, UsageLedger
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.traces.schema import (
    AgentSessionRecord,
    TaskRecord,
    TraceRecord,
)
from atomic_skillgraph.agents.runtime_policy_projection import (
    FORMAT,
    canonical_bytes,
    pack_downstream_context,
    project_runtime_payload,
    unpack_downstream_context,
)


def _obligation(
    *,
    edge_id: str,
    producer_role: str,
    consumer_step: str = "observe",
    consumer_role: str,
    anchors: dict | None = None,
) -> dict:
    return {
        "producer_step": "prepare",
        "producer_output_role": producer_role,
        "edge_id": edge_id,
        "consumer_step": consumer_step,
        "consumer_input_role": consumer_role,
        "consumer_summary": (
            "Observe the prepared object under the selected light source while "
            "preserving the task identity constraints."
        ),
        "consumer_input_contract": {
            "name": consumer_role,
            "semantic_type": "entity",
            "required": True,
            "description": "A validated input to the downstream occurrence.",
        },
        "consumer_preconditions": [{
            "predicate": "object.at_location",
            "args": {"object": "object", "location": "source"},
            "cardinality": 1,
        }],
        "consumer_effects": [{
            "predicate": "object.observed_under_light",
            "args": {"object": "object", "source": "source"},
            "cardinality": 1,
        }],
        "consumer_known_semantic_anchors": copy.deepcopy(anchors or {}),
    }


def _repeated_consumer_context() -> dict:
    anchors = {
        "object": {
            "value": "apple",
            "semantic_type": "entity",
            "source": "task",
        }
    }
    return {
        "current_step": "prepare",
        "output_obligations": [
            _obligation(
                edge_id="edge-object",
                producer_role="placed_object",
                consumer_role="object",
                anchors=anchors,
            ),
            _obligation(
                edge_id="edge-source",
                producer_role="place_location",
                consumer_role="source",
                anchors=anchors,
            ),
        ],
        "remaining_method_outline": [
            {"step_id": "observe", "summary": "observe the prepared object"}
        ],
    }


def test_support_projection_removes_only_top_level_diagnostics() -> None:
    raw = {
        "support_atomic_candidates": [
            {
                "atomic_ref": "atomic:a",
                "score": 1.0,
                "supplied_roles": ["entity"],
                "output_roles": ["entity"],
                "effect_predicates": ["object.visible"],
                "role_mappings": [{
                    "producer_role": "entity",
                    "consumer_role": "object",
                    "diagnostics": "nested mapping data must remain",
                }],
                "diagnostics": {"why": "full retrieval diagnostics"},
                "future_policy_safe_field": {"keep": True},
            },
            "unexpected-but-policy-safe-shape",
        ],
        "current_action_catalog": {
            "revision": 4,
            "actions": [{
                "action_id": "r004_a001",
                "action_type": "OPEN",
                "arguments": {"object": "cabinet_1"},
            }],
        },
    }
    before = copy.deepcopy(raw)

    projected, audit = project_runtime_payload(raw)

    assert raw == before
    expected_candidate = copy.deepcopy(raw["support_atomic_candidates"][0])
    del expected_candidate["diagnostics"]
    assert projected["support_atomic_candidates"] == [
        expected_candidate,
        "unexpected-but-policy-safe-shape",
    ]
    assert projected["current_action_catalog"] == raw["current_action_catalog"]
    assert audit["removed_fields"] == [
        "support_atomic_candidates[0].diagnostics"
    ]
    assert audit["raw_support_candidates"] == raw["support_atomic_candidates"]
    assert audit["raw_support_candidates"] is not raw["support_atomic_candidates"]
    assert audit["downstream_projection"] == "not_present"
    assert audit["before_payload_utf8_bytes"] > audit["after_payload_utf8_bytes"]
    assert audit["byte_count_is_not_token_count"] is True


def test_repeated_consumer_contract_round_trips_exactly_and_keeps_edges() -> None:
    raw = _repeated_consumer_context()
    before = copy.deepcopy(raw)

    packed, reason = pack_downstream_context(raw)

    assert reason == "deduplicated"
    assert raw == before
    assert packed["context_format"] == FORMAT
    assert list(packed["consumer_contracts"]) == ["observe"]
    assert list(
        packed["consumer_contracts"]["observe"]["consumer_input_contracts"]
    ) == ["object", "source"]
    assert [item["edge_id"] for item in packed["output_obligations"]] == [
        "edge-object",
        "edge-source",
    ]
    assert unpack_downstream_context(packed) == raw
    assert len(canonical_bytes(packed)) < len(canonical_bytes(raw))

    packed_again, repeat_reason = pack_downstream_context(packed)
    assert repeat_reason == "already_packed"
    assert packed_again == packed
    assert packed_again is not packed


def test_downstream_projection_falls_back_without_losing_unknown_or_conflicting_data() -> None:
    unknown = _repeated_consumer_context()
    unknown["new_constraint"] = {"must_keep": True}
    returned, reason = pack_downstream_context(unknown)
    assert reason == "original_unrecognized_shape"
    assert returned == unknown
    assert returned is not unknown

    conflicting = _repeated_consumer_context()
    conflicting["output_obligations"][1][
        "consumer_known_semantic_anchors"
    ] = {"source": {"value": "lamp_2"}}
    returned, reason = pack_downstream_context(conflicting)
    assert reason == "original_conflicting_consumer_view"
    assert returned == conflicting

    distinct_consumers = _repeated_consumer_context()
    distinct_consumers["output_obligations"][1]["consumer_step"] = "observe-2"
    returned, reason = pack_downstream_context(distinct_consumers)
    assert reason == "original_no_repeated_consumer"
    assert returned == distinct_consumers


def test_payload_projection_combines_lossless_downstream_pack_and_separate_audit() -> None:
    downstream = _repeated_consumer_context()
    raw = {
        "task_goal": "observe an object",
        "current_state_snapshot": {
            "current_atomic": {"summary": "prepare"},
            "confirmed_bindings": {"object": "apple_1"},
            "candidate_bindings": {},
            "missing_bindings": [],
            "invalidated_bindings": {},
            "preconditions": [],
            "effect_witness_status": {},
            "learned_invocation_ready": True,
            "blocking_reasons": [],
            "downstream_obligations": downstream,
            "remaining_budget": {"remaining_node_actions": 35},
        },
        "support_atomic_candidates": [{
            "atomic_ref": "atomic:support",
            "score": 0.75,
            "role_mappings": [{"producer_role": "entity"}],
            "diagnostics": ["not Agent-facing"],
        }],
        "current_observation": "apple_1 is visible",
        "recent_accepted_actions": [{"action_type": "LOOK"}],
    }

    projected, audit = project_runtime_payload(raw)

    assert audit["projection_version"] == "v3.2-r5"
    assert audit["downstream_projection"] == "deduplicated"
    assert audit["raw_downstream_obligations"] == downstream
    assert unpack_downstream_context(
        projected["current_state_snapshot"]["downstream_obligations"]
    ) == downstream
    assert projected["current_observation"] == raw["current_observation"]
    assert projected["recent_accepted_actions"] == raw["recent_accepted_actions"]
    assert projected["current_state_snapshot"]["confirmed_bindings"] == {
        "object": "apple_1"
    }
    assert audit["before_payload_sha256"] != audit["after_payload_sha256"]


class _RepairThenSuccessProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[list[dict], list[NativeToolSpec]]] = []

    def set_request_context(self, **_context) -> None:
        return None

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        normalized_tools = list(tools or [])
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(normalized_tools)))
        if len(self.requests) == 1:
            calls: list[NativeToolCall] = []
            reasoning = "private rejected reasoning"
        else:
            calls = [NativeToolCall("call_repaired", "runtime_action", {"ok": True})]
            reasoning = "private accepted reasoning"
        return AgentTurn(
            content="",
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
            reasoning_tokens=1,
            latency_ms=1.0,
            provider_metadata={"provider": "r5-test"},
            reasoning_content=reasoning,
        )

    def snapshot(self) -> dict:
        return {"provider": "r5-test", "call_count": len(self.requests)}


class _TimeoutProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[list[dict], list[NativeToolSpec]]] = []

    def set_request_context(self, **_context) -> None:
        return None

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        self.requests.append((
            copy.deepcopy(messages),
            copy.deepcopy(list(tools or [])),
        ))
        raise AgentProviderError("provider_timeout", "provider timed out")

    def snapshot(self) -> dict:
        return {"provider": "timeout-test", "call_count": len(self.requests)}


def _runtime_action_tool() -> NativeToolSpec:
    return NativeToolSpec(
        "runtime_action",
        "Submit one deterministic Runtime action.",
        {
            "type": "object",
            "required": ["ok"],
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
        },
    )


def test_runtime_request_audit_captures_safe_actual_views_and_protocol_repair() -> None:
    provider = _RepairThenSuccessProvider()
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
        session_id="r5-runtime-audit",
    )

    turn = session.next_turn("start", tools=[_runtime_action_tool()])

    assert turn.tool_calls[0].name == "runtime_action"
    snapshot = session.snapshot()
    audits = snapshot["runtime_request_context_audits"]
    assert len(provider.requests) == snapshot["provider_call_count"] == 2
    assert [item["request_sequence"] for item in audits] == [0, 1]
    assert [item["repair_in_progress"] for item in audits] == [False, True]
    assert all(item["session_id"] == "r5-runtime-audit" for item in audits)
    assert all(item["usage_bucket"] == "runtime_dynamic" for item in audits)
    assert audits[1]["tools"] == [{
        "name": "runtime_action",
        "description": "Submit one deterministic Runtime action.",
        "input_schema": _runtime_action_tool().input_schema,
    }]
    encoded = json.dumps(
        {"messages": audits[1]["messages"], "tools": audits[1]["tools"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert audits[1]["safe_snapshot_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert audits[1]["safe_snapshot_utf8_bytes"] == len(encoded)
    serialized = json.dumps(audits, ensure_ascii=False)
    assert "private rejected reasoning" not in serialized
    assert "private accepted reasoning" not in serialized
    assert "reasoning_content_chars" in serialized
    assert "reasoning_content_sha256" in serialized
    assert provider.requests[1][0][2]["reasoning_content"] == (
        "private rejected reasoning"
    )


def test_runtime_budget_failure_before_dispatch_creates_no_request_audit() -> None:
    provider = _RepairThenSuccessProvider()
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
        budget=AgentBudget(0, 100, "runtime_node_token_budget_exhausted"),
    )

    with pytest.raises(BudgetExhausted) as exc_info:
        session.next_turn("start", tools=[_runtime_action_tool()])

    assert getattr(exc_info.value, "code", "") == "runtime_node_token_budget_exhausted"
    assert provider.requests == []
    assert session.snapshot()["runtime_request_context_audits"] == []


def test_runtime_timeout_attempt_audit_survives_trace_serialization() -> None:
    provider = _TimeoutProvider()
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_seeded",
        session_id="r5-timeout-audit",
    )

    with pytest.raises(AgentProviderError, match="provider timed out"):
        session.next_turn("start", tools=[_runtime_action_tool()])

    snapshot = session.snapshot()
    assert len(provider.requests) == 1
    assert snapshot["provider_call_count"] == 0
    assert len(snapshot["runtime_request_context_audits"]) == 1
    audit = snapshot["runtime_request_context_audits"][0]
    assert audit["request_sequence"] == 0
    assert audit["repair_in_progress"] is False
    assert audit["session_id"] == "r5-timeout-audit"
    trace = TraceRecord.create(
        TaskRecord("task", "fake", "goal", "type", "signature", {}),
        {},
        {},
        {},
    )
    trace.agent_sessions.append(AgentSessionRecord(
        session.session_id,
        "SeededSession",
        "occurrence",
        0.0,
        snapshot=snapshot,
    ))
    serialized = json.dumps(
        to_primitive(trace),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    restored = json.loads(serialized)

    persisted = restored["agent_sessions"][0]["snapshot"]
    assert persisted["runtime_request_context_audits"] == [audit]
