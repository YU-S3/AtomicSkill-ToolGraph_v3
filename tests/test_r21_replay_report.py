from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from atomic_skillgraph.agents import NativeToolSpec, ReplayAgentSession, UsageLedger
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
    assert initial_payload["current_action_catalog"] == {
        "status": "superseded",
        "entry_count": 1,
        "superseded_by_revision": 2,
    }
    assert initial_payload["remaining_budget"] == {
        "remaining_global_actions": 100,
    }
    tool_payloads = [
        json.loads(message["content"])
        for message in request.messages
        if message["role"] == "tool"
    ]
    assert tool_payloads[0]["action_catalog"]["status"] == "superseded"
    assert tool_payloads[0]["remaining_budget"] == {
        "remaining_global_actions": 99,
    }
    assert tool_payloads[1]["action_catalog"]["revision"] == 2
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
    assert len(assistants) == 2
    assert all("reasoning_content" in message for message in assistants)
    assert all("tool_calls" in message for message in assistants)
    prior_first_assistant = next(
        message
        for message in provider.requests[1].messages
        if message["role"] == "assistant"
    )
    assert assistants[0] == prior_first_assistant
    snapshot = session.snapshot()
    assert snapshot["replay_catalog_compaction_count"] == 2
    assert snapshot["replay_initial_catalog_compacted"] is True
    assert snapshot["replay_full_catalog_count_at_last_request"] == 1
    assert snapshot["replay_history_action_count"] == 2


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
