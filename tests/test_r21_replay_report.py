from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from atomic_skillgraph.agents import (
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    ReplayAgentSession,
    UsageLedger,
)
from experiments.fakes import FakeReply, ScriptedAgentProvider
from experiments.protocol import ProtocolError, validate_deepseek_formal_llm
from experiments.report import summarize_traces, trace_to_row


ROOT = Path(__file__).resolve().parents[1]
POLICY_SEPARATOR = "\n\nPOLICY_CONTEXT_JSON\n"


def _action_tool(action_id: str) -> NativeToolSpec:
    return NativeToolSpec(
        "environment_action",
        "Execute one current action.",
        {
            "type": "object",
            "required": ["action_id"],
            "additionalProperties": False,
            "properties": {
                "action_id": {"type": "string", "enum": [action_id]},
            },
        },
    )


def _projection(revision: int, recent: list[dict] | None = None) -> dict:
    return {
        "current_state_snapshot": {
            "revision": revision,
            "occurrence_id": "occ_runtime",
        },
        "exploration_memory": {"visited": []},
        "recent_accepted_actions": list(recent or []),
    }


def test_runtime_replay_compacts_initial_and_old_catalogs_without_touching_assistant() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.tool("environment_action", {"action_id": "r000_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r001_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r002_a001"}),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
    )
    prompt = "Choose one action." + POLICY_SEPARATOR + json.dumps({
        **_projection(0),
        "current_action_catalog": {
            "revision": 0,
            "actions": [{
                "action_id": "r000_a001",
                "action_type": "LOOK",
                "arguments": {},
            }],
        },
        "remaining_budget": {
            "global_action_budget": 100,
            "remaining_global_actions": 100,
            "used_global_actions": 0,
            "used_tokens": 0,
        },
    })
    first = session.next_turn(prompt, tools=[_action_tool("r000_a001")])
    second = session.submit_tool_result(
        first.tool_calls[0].call_id,
        {
            **_projection(1, [{
                "action_type": "LOOK",
                "arguments": {},
                "observation": "revision one",
                "revision": 1,
                "done": False,
                "won": False,
                "origin": "runtime_dynamic",
            }]),
            "accepted": True,
            "observation": "revision one",
            "new_revision": 1,
            "action_catalog": {
                "revision": 1,
                "actions": [{
                    "action_id": "r001_a001",
                    "action_type": "GO_TO",
                    "arguments": {"destination": "place_1"},
                }],
            },
            "remaining_budget": {
                "remaining_global_actions": 99,
                "used_global_actions": 1,
                "used_tokens": 999,
            },
        },
        tools=[_action_tool("r001_a001")],
    )
    session.submit_tool_result(
        second.tool_calls[0].call_id,
        {
            **_projection(2, [
                {
                    "action_type": "LOOK",
                    "arguments": {},
                    "observation": "revision one",
                    "revision": 1,
                    "done": False,
                    "won": False,
                    "origin": "runtime_dynamic",
                },
                {
                    "action_type": "GO_TO",
                    "arguments": {"destination": "place_1"},
                    "observation": "revision two",
                    "revision": 2,
                    "done": False,
                    "won": False,
                    "origin": "runtime_dynamic",
                },
            ]),
            "accepted": True,
            "observation": "revision two",
            "new_revision": 2,
            "action_catalog": {
                "revision": 2,
                "actions": [{
                    "action_id": "r002_a001",
                    "action_type": "TAKE",
                    "arguments": {"object": "thing_1"},
                }],
            },
            "remaining_budget": {
                "remaining_global_actions": 98,
                "used_global_actions": 2,
                "used_turns": 2,
            },
        },
        tools=[_action_tool("r002_a001")],
    )

    request = provider.requests[2]
    initial = next(
        message for message in request.messages if message["role"] == "user"
    )
    initial_payload = json.loads(initial["content"].split(POLICY_SEPARATOR, 1)[1])
    assert "current_action_catalog" not in initial_payload
    assert "remaining_budget" not in initial_payload
    assert "current_state_snapshot" not in initial_payload
    assert "exploration_memory" not in initial_payload
    tool_payloads = [
        json.loads(message["content"])
        for message in request.messages
        if message["role"] == "tool"
    ]
    assert len(tool_payloads) == 1
    assert tool_payloads[0]["action_catalog"]["revision"] == 2
    assert sum(
        int(
            isinstance(payload.get("action_catalog"), dict)
            and isinstance(payload["action_catalog"].get("actions"), list)
        )
        for payload in [initial_payload, *tool_payloads]
    ) == 1

    assistants = [
        message for message in request.messages if message["role"] == "assistant"
    ]
    assert len(assistants) == 1
    assert all("reasoning_content" in message for message in assistants)
    assert all("tool_calls" in message for message in assistants)
    prior_first_assistant = next(
        message
        for message in provider.requests[1].messages
        if message["role"] == "assistant"
    )
    assert assistants[0] != prior_first_assistant
    assert sum(
        "current_state_snapshot" in payload
        for payload in [initial_payload, *tool_payloads]
    ) == 1
    assert sum(
        "exploration_memory" in payload
        for payload in [initial_payload, *tool_payloads]
    ) == 1
    snapshot = session.snapshot()
    assert snapshot["replay_catalog_compaction_count"] == 2
    assert snapshot["replay_initial_catalog_compacted"] is True
    assert snapshot["replay_full_catalog_count_at_last_request"] == 1
    assert snapshot["replay_history_action_count"] == 2


def test_runtime_replay_keeps_one_envelope_and_five_structured_actions() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.tool("environment_action", {"action_id": f"r{i:03d}_a001"})
        for i in range(23)
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
    )
    turn = session.next_turn("start", tools=[_action_tool("r000_a001")])
    recent: list[dict] = []
    for index in range(22):
        next_action_id = f"r{index + 1:03d}_a001"
        recent.append({
            "action_type": "LOOK",
            "arguments": {},
            "observation": f"accepted-{index}",
            "revision": index + 1,
            "done": False,
            "won": False,
            "origin": "runtime_dynamic",
        })
        turn = session.submit_tool_result(
            turn.tool_calls[0].call_id,
            {
                **_projection(index + 1, recent[-5:]),
                "accepted": True,
                "observation": f"accepted-{index}",
                "new_revision": index + 1,
                "action_catalog": [{
                    "action_id": next_action_id,
                    "revision": index + 1,
                }],
            },
            tools=[_action_tool(next_action_id)],
        )

    replay = provider.requests[22].messages
    assistants = [message for message in replay if message["role"] == "assistant"]
    tool_results = [message for message in replay if message["role"] == "tool"]
    assert len(assistants) == 1
    assert len(tool_results) == 1
    assert [
        json.loads(message["tool_calls"][0]["function"]["arguments"])["action_id"]
        for message in assistants
    ] == ["r021_a001"]
    projected = json.loads(tool_results[0]["content"])
    projection_payloads = [projected]
    for message in replay:
        if message["role"] != "user" or POLICY_SEPARATOR not in message["content"]:
            continue
        projection_payloads.append(
            json.loads(message["content"].split(POLICY_SEPARATOR, 1)[1])
        )
    assert sum(
        "current_state_snapshot" in item for item in projection_payloads
    ) == 1
    assert sum("exploration_memory" in item for item in projection_payloads) == 1
    assert [
        item["observation"]
        for item in projected["recent_accepted_actions"]
    ] == [f"accepted-{i}" for i in range(17, 22)]

    # Retained provider envelopes, including reasoning_content, are untouched.
    assert assistants[0]["reasoning_content"] == (
        "deterministic reasoning for call_default_000021"
    )
    snapshot = session.snapshot()
    assert snapshot["replay_action_window_size"] == 5
    assert snapshot["replay_action_window_compaction_count"] == 21
    assert snapshot["replay_pruned_action_count"] == 21
    assert snapshot["replay_history_action_count"] == 5
    assert snapshot["pending_tool_call"] == {
        "call_id": turn.tool_calls[0].call_id,
        "name": "environment_action",
        "arguments": {"action_id": "r022_a001"},
    }
    assert any(
        message.get("role") == "assistant"
        and message.get("tool_calls", [{}])[0].get("id") == turn.tool_calls[0].call_id
        and message.get("reasoning_content_present") is True
        for message in snapshot["messages"]
    )


def test_runtime_replay_initial_history_and_new_pairs_share_the_five_action_window() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.tool("environment_action", {"action_id": "r000_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r001_a001"}),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
    )
    history = [
        {
            "action_type": f"ACTION_{index}",
            "arguments": {},
            "observation": f"history-{index}",
            "accepted": index not in {1, 4},
        }
        for index in range(8)
    ]
    prompt = "Choose one action." + POLICY_SEPARATOR + json.dumps({
        **_projection(0, history),
        "current_action_catalog": [{"action_id": "r000_a001", "revision": 0}],
        "remaining_budget": {"remaining_global_actions": 100},
    })
    first = session.next_turn(prompt, tools=[_action_tool("r000_a001")])
    first_history = provider.requests[0].policy_context["recent_accepted_actions"]
    assert [item["observation"] for item in first_history] == [
        "history-2", "history-3", "history-5", "history-6", "history-7",
    ]

    session.submit_tool_result(
        first.tool_calls[0].call_id,
        {
            **_projection(1, [
                *[item for item in first_history[-4:]],
                {
                    "action_type": "LOOK",
                    "arguments": {},
                    "observation": "new accepted action",
                    "revision": 1,
                    "done": False,
                    "won": False,
                    "origin": "runtime_dynamic",
                },
            ]),
            "accepted": True,
            "observation": "new accepted action",
            "new_revision": 1,
            "action_catalog": [{"action_id": "r001_a001", "revision": 1}],
        },
        tools=[_action_tool("r001_a001")],
    )
    second_payload = json.loads(next(
        message["content"]
        for message in provider.requests[1].messages
        if message["role"] == "tool"
    ))
    second_history = second_payload["recent_accepted_actions"]
    assert [item["observation"] for item in second_history] == [
        "history-3", "history-5", "history-6", "history-7",
        "new accepted action",
    ]
    snapshot = session.snapshot()
    assert snapshot["replay_history_action_count"] == 5
    assert snapshot["replay_action_window_compaction_count"] == 2
    assert snapshot["replay_pruned_action_count"] >= 4


def test_multiple_runtime_tool_calls_are_all_rejected_before_single_call_repair() -> None:
    def turn(call_ids: list[str]) -> AgentTurn:
        calls = [
            NativeToolCall(call_id, "environment_action", {"action_id": "r000_a001"})
            for call_id in call_ids
        ]
        reasoning = "private rejected reasoning" if len(calls) > 1 else "private repair reasoning"
        return AgentTurn(
            content="",
            tool_calls=calls,
            finish_reason="tool_calls",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            reasoning_tokens=1,
            latency_ms=0.0,
            provider_metadata={"provider": "test"},
            reasoning_content=reasoning,
            replay_assistant_message={
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in calls
                ],
            },
        )

    class MultipleThenSingleProvider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages, *, tools=None):
            self.calls += 1
            return turn(["rejected_a", "rejected_b"] if self.calls == 1 else ["accepted_c"])

        def snapshot(self):
            return {"provider": "multiple_then_single"}

    session = ReplayAgentSession(
        MultipleThenSingleProvider(),
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
    )
    repaired = session.next_turn("start", tools=[_action_tool("r000_a001")])
    assert repaired.tool_calls[0].call_id == "accepted_c"
    snapshot = session.snapshot()
    rejected_results = [
        json.loads(message["content"])
        for message in snapshot["messages"]
        if message.get("role") == "tool"
    ]
    assert rejected_results == [
        {
            "accepted": False,
            "error": "runtime_agent_multiple_tool_calls",
            "executed": False,
        },
        {
            "accepted": False,
            "error": "runtime_agent_multiple_tool_calls",
            "executed": False,
        },
    ]
    assert snapshot["protocol_repairs_used"] == 1
    assert snapshot["replay_history_action_count"] == 0


def test_all_runtime_system_prompts_start_with_single_toolcall_rule() -> None:
    from atomic_skillgraph.system import _SYSTEM_PROMPTS

    for stage in ("runtime_preparation", "runtime_seeded", "runtime_dynamic"):
        assert _SYSTEM_PROMPTS[stage].startswith(
            "Exactly ONE native ToolCall per turn."
        )


def test_r21_formal_runtime_parameters_and_action_budgets_are_frozen() -> None:
    config = yaml.safe_load((ROOT / "configs" / "alfworld_train_full_30.yaml").read_text(
        encoding="utf-8"
    ))
    validate_deepseek_formal_llm(config)
    assert config["llm"]["runtime"] == {
        "reasoning_effort": "low",
        "max_completion_tokens": 32768,
        "request_timeout_seconds": 180,
        "max_total_tokens_per_node": 100000,
        "max_total_tokens_per_task": 300000,
        "learned_toolcall_repair_limit": 2,
        "protocol_repair_limit": 1,
    }
    assert config["llm"]["planner"]["reasoning_effort"] == "high"
    assert config["llm"]["extractor"]["reasoning_effort"] == "high"
    assert config["runtime"]["global_action_budget"] == 100
    assert config["runtime"]["node_action_budget"] == 35

    invalid = copy.deepcopy(config)
    invalid["runtime"]["node_action_budget"] = 36
    with pytest.raises(ProtocolError, match="runtime.node_action_budget"):
        validate_deepseek_formal_llm(invalid)


def test_r21_report_derives_commit_conflict_replay_and_token_metrics() -> None:
    trace = {
        "trace_id": "trace-r21",
        "schema_version": 3,
        "task": {
            "task_id": "task-r21",
            "task_signature": "signature-r21",
            "benchmark": "alfworld",
            "task_type": "fixture",
        },
        "runtime_plan": {"source": "stored_composite"},
        "strict_task_success": True,
        "task_rescue_required": True,
        "native_tool_calls": [
            {
                "tool_name": "environment_action",
                "arguments": {"action_id": "a1", "intent": "explore"},
                "occurrence_id": "occ-1",
            },
            {
                "tool_name": "environment_action",
                "arguments": {
                    "action_id": "a2", "intent": "attempt_current_atomic",
                },
                "occurrence_id": "occ-1",
            },
            {
                "tool_name": "validate_current_atomic",
                "arguments": {"candidate_bindings": {"location": "place_1"}},
                "preflight_result": {"atomic_effect_passed": True},
                "occurrence_id": "occ-1",
            },
            {
                "tool_name": "invoke_impl_portable_action_1234",
                "call_kind": "implementation_invocation",
                "arguments": {},
                "occurrence_id": "occ-1",
            },
            {
                "tool_name": "report_runtime_status",
                "arguments": {"status": "plan_conflict"},
                "occurrence_id": "occ-1",
            },
        ],
        "agent_sessions": [{
            "session_id": "session-1",
            "session_type": "RuntimePreparationSession",
            "occurrence_id": "occ-1",
            "snapshot": {
                "usage_bucket": "runtime_preparation",
                "replay_catalog_compaction_count": 3,
                "replay_initial_catalog_compacted": True,
                "replay_full_catalog_count_at_last_request": 1,
                "replay_history_action_count": 2,
                "budget": {
                    "max_turns": 100,
                    "used_turns": 2,
                    "max_total_tokens": 100000,
                    "used_total_tokens": 140,
                },
            },
        }],
        "llm_usage": [{
            "session_id": "session-1",
            "bucket": "runtime_preparation",
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "reasoning_tokens": 30,
            "call_count": 1,
            "latency_ms": 1.0,
        }],
    }
    row = trace_to_row(trace)
    assert row["runtime_exploration_action_count"] == 1
    assert row["runtime_atomic_attempt_action_count"] == 1
    assert row["runtime_validate_current_atomic_success_count"] == 1
    assert row["learned_invocation_selected_count"] == 1
    assert row["runtime_plan_conflict_count"] == 1
    assert row["runtime_plan_conflict_rescue_success_count"] == 1
    assert row["replay_catalog_compaction_count"] == 3
    assert row["runtime_prompt_tokens"] == 100
    assert row["runtime_completion_tokens"] == 40
    assert row["runtime_reasoning_tokens"] == 30
    assert row["runtime_reasoning_share"] == 0.75
    stage = row["runtime_token_decomposition"]["runtime_preparation"]
    assert stage["calls"] == 1
    assert stage["average_tokens_per_call"] == 140.0
    assert stage["max_tokens_per_call"] == 140

    summary = summarize_traces([row])
    assert summary["runtime_plan_conflict_count"] == 1
    assert summary["runtime_reasoning_share"] == 0.75
    assert summary["runtime_token_decomposition"]["runtime_preparation"] == stage


def test_r21_report_defaults_new_metrics_to_zero_for_legacy_trace() -> None:
    row = trace_to_row({
        "trace_id": "legacy",
        "schema_version": 3,
        "task": {"task_id": "legacy"},
    })
    assert row["runtime_exploration_action_count"] == 0
    assert row["runtime_plan_conflict_count"] == 0
    assert row["runtime_prompt_tokens"] == 0
    assert row["runtime_reasoning_share"] == 0.0
    assert all(
        item["calls"] == 0
        for item in row["runtime_token_decomposition"].values()
    )


def test_r31_report_aggregates_only_structured_events() -> None:
    trace = {
        "trace_id": "trace-r31",
        "schema_version": 3,
        "task": {"task_id": "task-r31"},
        "metadata": {
            "r3_events": [
                {
                    "revision": 1,
                    "occurrence_id": "occ-1",
                    "event_type": "grounding_refresh",
                    "details": {},
                },
                {
                    "revision": 1,
                    "occurrence_id": "occ-1",
                    "event_type": "unique_binding_auto_confirm",
                    "details": {"role_count": 2},
                },
                {
                    "revision": 2,
                    "occurrence_id": "occ-1",
                    "event_type": "invocation_ready_transition",
                    "details": {"from": False, "to": True},
                },
                {
                    "revision": 3,
                    "occurrence_id": "occ-1",
                    "event_type": "effect_ready_transition",
                    "details": {"from": False, "to": True},
                },
                {
                    "revision": 4,
                    "occurrence_id": "occ-1",
                    "event_type": "runtime_context_projection",
                    "details": {
                        "current_state_snapshot_count": 1,
                        "exploration_memory_count": 1,
                        "recent_action_count": 5,
                        "action_window_size": 5,
                    },
                },
                {
                    "revision": 4,
                    "occurrence_id": "occ-1",
                    "event_type": "replay_action_window_compaction",
                    "details": {"pruned_action_count": 3},
                },
                {
                    "revision": 4,
                    "occurrence_id": "",
                    "event_type": "partial_atomic_admission",
                    "details": {
                        "admission_count": 1,
                        "alignment_reuse_count": 1,
                        "new_contract_count": 0,
                        "tool_admission_count": 1,
                        "implementation_admission_count": 1,
                    },
                },
            ],
            # Report must not infer R3 transitions from other metadata.
            "runtime_state_snapshots": [{"learned_invocation_ready": True}],
        },
    }
    row = trace_to_row(trace)
    assert row["runtime_grounding_refresh_count"] == 1
    assert row["runtime_unique_binding_auto_confirm_count"] == 1
    assert row["runtime_unique_binding_auto_confirm_role_count"] == 2
    assert row["runtime_invocation_ready_transition_count"] == 1
    assert row["runtime_effect_ready_transition_count"] == 1
    assert row["replay_action_window_size"] == 5
    assert row["replay_action_window_compaction_count"] == 1
    assert row["replay_pruned_action_count"] == 3
    assert row["runtime_context_snapshot_count"] == 1
    assert row["runtime_exploration_memory_projection_count"] == 1
    assert row["runtime_recent_action_projection_count"] == 5
    assert row["partial_atomic_admission_count"] == 1
    assert row["partial_atomic_alignment_reuse_count"] == 1
    assert row["partial_atomic_new_contract_count"] == 0
    assert row["partial_atomic_tool_admission_count"] == 1
    assert row["partial_atomic_implementation_admission_count"] == 1
    summary = summarize_traces([row])
    for name in (
        "runtime_grounding_refresh_count",
        "replay_action_window_compaction_count",
        "partial_atomic_admission_count",
    ):
        assert summary[name] == row[name]


def test_r21_report_uses_strict_exhaustion_and_per_event_token_authority() -> None:
    trace = {
        "trace_id": "strict-runtime-accounting",
        "schema_version": 3,
        "task": {"task_id": "strict-runtime-accounting"},
        "node_records": [{
            "occurrence_id": "occ-exhausted",
            "direct_result": {
                "failure_code": "runtime_node_token_budget_exhausted",
            },
        }],
        "agent_sessions": [
            {
                "session_id": "session-exact-cap-success",
                "session_type": "RuntimePreparationSession",
                "occurrence_id": "occ-success",
                "snapshot": {
                    "usage_bucket": "runtime_preparation",
                    "replay_initial_catalog_compacted": True,
                    # Reaching the cap is not exhaustion evidence by itself.
                    "budget": {
                        "max_turns": 10,
                        "used_turns": 2,
                        "max_total_tokens": 100,
                        "used_total_tokens": 100,
                    },
                },
            },
            {
                "session_id": "session-explicit-exhaustion",
                "session_type": "RuntimePreparationSession",
                "occurrence_id": "occ-exhausted",
                "snapshot": {
                    "usage_bucket": "runtime_preparation",
                    "replay_initial_catalog_compacted": True,
                    "budget": {
                        "max_turns": 10,
                        "used_turns": 3,
                        "max_total_tokens": 100,
                        "used_total_tokens": 101,
                    },
                },
            },
        ],
        "llm_usage": [
            {
                "session_id": "session-exact-cap-success",
                "bucket": "runtime_preparation",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 16,
                "reasoning_tokens": 3,
                "call_count": 1,
            },
            {
                "session_id": "session-explicit-exhaustion",
                "bucket": "runtime_preparation",
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "total_tokens": 28,
                "reasoning_tokens": None,
                "call_count": 1,
            },
        ],
    }
    row = trace_to_row(trace)
    stage = row["runtime_token_decomposition"]["runtime_preparation"]
    assert row["replay_initial_catalog_compacted_count"] == 2
    assert row["runtime_reasoning_tokens"] == 3
    assert stage["reasoning_tokens"] == 3
    assert stage["total_tokens"] == 44
    assert stage["average_tokens_per_call"] == 22.0
    assert stage["exhausted_session_count"] == 1

    summary_stage = summarize_traces([row])["runtime_token_decomposition"][
        "runtime_preparation"
    ]
    assert summary_stage["total_tokens"] == 44
    assert summary_stage["average_tokens_per_call"] == 22.0


def test_r21_report_does_not_guess_ambiguous_legacy_exhaustion_session() -> None:
    row = trace_to_row({
        "trace_id": "ambiguous-legacy-exhaustion",
        "schema_version": 3,
        "task": {"task_id": "ambiguous-legacy-exhaustion"},
        "agent_sessions": [
            {
                "session_id": "preparation",
                "session_type": "RuntimePreparationSession",
                "occurrence_id": "occ-1",
            },
            {
                "session_id": "seeded",
                "session_type": "SeededSession",
                "occurrence_id": "occ-1",
            },
        ],
        "failures": [{
            "code": "runtime_node_token_budget_exhausted",
            "occurrence_id": "occ-1",
        }],
    })
    assert all(
        item["exhausted_session_count"] == 0
        for item in row["runtime_token_decomposition"].values()
    )


def test_r21_report_maps_only_explicit_provisional_exhaustion_evidence() -> None:
    row = trace_to_row({
        "trace_id": "provisional-exhaustion",
        "schema_version": 3,
        "task": {"task_id": "provisional-exhaustion"},
        "agent_sessions": [
            {
                "session_id": "provisional-exact-cap-success",
                "session_type": "ProvisionalSeededSession",
                "occurrence_id": "cold::step-success",
                "snapshot": {
                    "usage_bucket": "runtime_provisional_seeded",
                    "budget": {
                        "max_turns": 10,
                        "used_turns": 2,
                        "max_total_tokens": 100000,
                        "used_total_tokens": 100000,
                    },
                },
            },
            {
                "session_id": "provisional-explicit-exhaustion",
                "session_type": "ProvisionalSeededSession",
                "occurrence_id": "cold::step-exhausted",
                "snapshot": {
                    "usage_bucket": "runtime_provisional_seeded",
                    "budget": {
                        "max_turns": 10,
                        "used_turns": 3,
                        "max_total_tokens": 100000,
                        "used_total_tokens": 100001,
                    },
                },
            },
        ],
        "cold_start_steps": [
            {"step_id": "step-success", "failure_code": ""},
            {
                "step_id": "step-exhausted",
                "failure_code": "provisional_seeded_budget_exhausted",
            },
        ],
    })
    assert row["runtime_token_decomposition"][
        "runtime_provisional_seeded"
    ]["exhausted_session_count"] == 1
