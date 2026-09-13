from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from atomic_skillgraph.agents import (
    NativeToolSpec,
    ReplayAgentSession,
    UsageLedger,
)
from atomic_skillgraph.agents.usage import (
    AgentBudget,
    BudgetTracker,
    LLMUsage,
    UsageBucket,
    resolve_tool_builder_usage_bucket,
)
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.system import AtomicSkillGraphSystem, _trace_release_metadata
from experiments.fakes import FakeReply, ScriptedAgentProvider
from experiments.report import (
    LEARNING_EXECUTED_R4,
    LEARNING_NOT_APPLICABLE_FROZEN,
    LEARNING_SKIPPED_BY_POLICY,
    LEGACY_DIAGNOSTICS_UNAVAILABLE,
    render_markdown,
    summarize_traces,
    trace_to_row,
)


def _usage_event(
    event_id: str,
    session_id: str,
    bucket: str,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "session_id": session_id,
        "turn_index": 0,
        "bucket": bucket,
        "prompt_tokens": total_tokens,
        "completion_tokens": 0,
        "total_tokens": total_tokens,
        "reasoning_tokens": 0,
        "call_count": 1,
        "latency_ms": 1.0,
        "provider_metadata": {"usage_status": "reported"},
    }


def _trace(
    task_id: str,
    *,
    success: bool,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    usage = list(events or [])
    return {
        "trace_id": f"trace-{task_id}",
        "schema_version": 3,
        "task": {
            "task_id": task_id,
            "task_signature": f"sig-{task_id}",
            "benchmark": "alfworld",
            "task_type": "pick_and_place_simple",
        },
        "runtime_plan": {"source": "full_dynamic", "control_sequence": []},
        "node_records": [],
        "agent_sessions": [],
        "benchmark_success": success,
        "strict_task_success": success,
        "llm_usage": usage,
        "metadata": {
            "usage_reconciliation": {
                "episode_total_tokens": sum(
                    int(item["total_tokens"]) for item in usage
                )
            }
        },
    }


def test_f01_cost_labels_include_failed_and_auxiliary_usage() -> None:
    solved = _trace(
        "solved",
        success=True,
        events=[
            _usage_event("u-prep", "s-prep", "runtime_preparation", 70),
            _usage_event(
                "u-builder", "s-builder", "tool_builder_runtime", 30
            ),
        ],
    )
    failed = _trace(
        "failed",
        success=False,
        events=[_usage_event("u-failed", "s-failed", "runtime_dynamic", 200)],
    )
    maintenance = _trace(
        "maintenance",
        success=False,
        events=[
            _usage_event(
                "u-maintenance", "s-maintenance", "evolution_repair", 50
            )
        ],
    )
    maintenance["metadata"]["trace_kind"] = "maintenance"
    maintenance["task"]["task_type"] = "maintenance"
    auxiliary_attempt = _trace(
        "failed-attempt",
        success=False,
        events=[
            _usage_event(
                "u-attempt-runtime", "s-attempt", "runtime_dynamic", 50,
            ),
        ],
    )

    summary = summarize_traces(
        [solved, failed],
        auxiliary_usage_traces=[maintenance, auxiliary_attempt],
    )
    assert summary["success_conditioned_tokens_per_solved_task"] == 100.0
    assert summary["all_run_tokens_per_solved_task"] == 400.0
    # Auxiliary attempts remain in all-run cost, but are not formal task rows.
    assert summary["runtime_all_tasks_tokens_per_solved_task"] == 300.0
    assert summary["runtime_tool_builder_tokens"] == 30
    assert summary["maintenance_tokens"] == 50
    # The historical comparison field remains byte-for-byte compatible.
    assert summary["tokens_per_solved_task"] == 200.0

    no_solved = summarize_traces([failed], auxiliary_usage_traces=[maintenance])
    assert no_solved["success_conditioned_tokens_per_solved_task"] is None
    assert no_solved["all_run_tokens_per_solved_task"] is None
    assert no_solved["runtime_all_tasks_tokens_per_solved_task"] is None


def test_f03_f04_budget_tracker_records_stage_reason_and_provider_usage() -> None:
    token_tracker = BudgetTracker(
        AgentBudget(3, 100, "runtime_node_token_budget_exhausted")
    )
    with pytest.raises(BudgetExhausted) as token_error:
        token_tracker.consume(LLMUsage(total_tokens=120, call_count=1))
    token_audit = token_error.value.budget_audit
    assert token_error.value.code == "runtime_node_token_budget_exhausted"
    assert token_audit["budget_reason"] == "token_limit"
    assert token_audit["budget_check_stage"] == "after_provider_call"
    assert token_audit["provider_call_recorded"] is True
    assert token_audit["actual_call_usage"]["total_tokens"] == 120
    assert token_tracker.snapshot()["used_total_tokens"] == 120

    turn_tracker = BudgetTracker(
        AgentBudget(1, 100, "runtime_node_token_budget_exhausted")
    )
    turn_tracker.consume(LLMUsage(total_tokens=10, call_count=1))
    with pytest.raises(BudgetExhausted) as turn_error:
        turn_tracker.check_before_call()
    turn_audit = turn_error.value.budget_audit
    assert turn_audit["budget_reason"] == "turn_limit"
    assert turn_audit["budget_check_stage"] == "before_call"
    assert turn_audit["provider_call_recorded"] is False


def _environment_action_tool() -> NativeToolSpec:
    return NativeToolSpec(
        "environment_action",
        "Execute one environment action.",
        {
            "type": "object",
            "required": ["action_id"],
            "additionalProperties": False,
            "properties": {"action_id": {"type": "string"}},
        },
    )


def test_f03_executed_result_is_persisted_before_next_turn_budget_check() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001"},
            prompt_tokens=2,
            completion_tokens=1,
            reasoning_tokens=0,
        ),
    ])
    ledger = UsageLedger()
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime budget boundary",
        usage_ledger=ledger,
        usage_bucket="runtime_preparation",
        budget=AgentBudget(
            2, 3, "runtime_node_token_budget_exhausted",
        ),
        session_id="runtime-executed-at-limit",
    )
    tool = _environment_action_tool()

    first = session.next_turn("act", tools=[tool])
    result = {
        "accepted": True,
        "executed": True,
        "observation": "at desk_1",
    }
    with pytest.raises(BudgetExhausted) as exhausted:
        session.submit_tool_result(
            first.tool_calls[0].call_id,
            result,
            tools=[tool],
            returned_action_executed=True,
        )

    assert exhausted.value.code == "runtime_node_token_budget_exhausted"
    assert len(provider.requests) == 1
    assert ledger.total().total_tokens == 3
    snapshot = session.snapshot()
    assert snapshot["pending_tool_call"] is None
    assert snapshot["messages"][-1]["role"] == "tool"
    assert json.loads(snapshot["messages"][-1]["content"]) == result
    audit = snapshot["budget"]["exhaustion_events"][-1]
    assert audit["budget_check_stage"] == "before_call"
    assert audit["provider_call_recorded"] is False
    assert audit["returned_action_executed"] is True


def test_f03_control_result_defaults_to_no_environment_action() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.tool(
            "propose_runtime_automation_atomic",
            {"draft_id": "rejected"},
            prompt_tokens=2,
            completion_tokens=1,
            reasoning_tokens=0,
        ),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime control boundary",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_preparation",
        budget=AgentBudget(
            2, 3, "runtime_node_token_budget_exhausted",
        ),
        session_id="runtime-control-at-limit",
    )
    tool = NativeToolSpec(
        "propose_runtime_automation_atomic",
        "Submit one task-local draft.",
        {
            "type": "object",
            "properties": {"draft_id": {"type": "string"}},
            "required": ["draft_id"],
            "additionalProperties": False,
        },
    )

    first = session.next_turn("consider automation", tools=[tool])
    with pytest.raises(BudgetExhausted) as exhausted:
        session.submit_tool_result(
            first.tool_calls[0].call_id,
            {"accepted": False, "error": "runtime_automation_r0_rejected"},
            tools=[tool],
        )

    audit = session.snapshot()["budget"]["exhaustion_events"][-1]
    assert audit["returned_action_executed"] is False
    assert audit["actual_call_usage"]["call_count"] == 0
    assert exhausted.value.budget_audit == audit


def test_f03_provider_overrun_does_not_claim_unexecuted_action() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001"},
            prompt_tokens=2,
            completion_tokens=1,
            reasoning_tokens=0,
        ),
    ])
    ledger = UsageLedger()
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime provider overrun",
        usage_ledger=ledger,
        usage_bucket="runtime_preparation",
        budget=AgentBudget(
            2, 2, "runtime_node_token_budget_exhausted",
        ),
        session_id="runtime-provider-overrun",
    )

    with pytest.raises(BudgetExhausted) as exhausted:
        session.next_turn("act", tools=[_environment_action_tool()])

    assert len(provider.requests) == 1
    assert ledger.total().total_tokens == 3
    snapshot = session.snapshot()
    assert snapshot["pending_tool_call"] is None
    audit = snapshot["budget"]["exhaustion_events"][-1]
    assert audit["budget_check_stage"] == "after_provider_call"
    assert audit["provider_call_recorded"] is True
    assert audit["returned_action_executed"] is False
    assert exhausted.value.budget_audit == audit


def test_f02_offered_automation_is_not_reported_as_a_proposal() -> None:
    trace = _trace("offered", success=True)
    trace["agent_sessions"] = [{
        "session_id": "runtime-session",
        "session_type": "RuntimePreparationSession",
        "occurrence_id": "occ-1",
        "snapshot": {
            "usage_bucket": "runtime_preparation",
            "runtime_request_context_audits": [{
                "runtime_automation_offered": True,
                "runtime_automation_interface_projected": True,
            }],
        },
    }]
    row = trace_to_row(trace)
    funnel = row["runtime_automation_funnel"]
    assert funnel["offered_request_count"] == 1
    assert funnel["interface_projected_request_count"] == 1
    assert funnel["proposal_count"] == 0
    assert funnel["r0_pass"] == 0
    assert funnel["r0_reject"] == 0


def test_f02_registered_tool_actions_are_not_automation_trial_actions() -> None:
    trace = _trace("registered-tool-only", success=True)
    trace["tool_executions"] = [{
        "execution_id": "registered-execution",
        "result": {
            "executed_action_count": 3,
            "executed_step_count": 3,
        },
    }]

    row = trace_to_row(trace)
    # Preserve the historical broad v3.2 Tool-action diagnostic.
    assert row["runtime_tool_internal_action_count"] == 3
    # The R9.2 funnel must not claim a task-local trial that never occurred.
    funnel = row["runtime_automation_funnel"]
    assert funnel["trial_started"] == 0
    assert funnel["trial_internal_action_count"] == 0
    assert funnel["trial_llm_bypassed_action_count"] == 0


def test_f02_trial_actions_require_matching_tool_execution_ids() -> None:
    trace = _trace("registered-and-trial-tools", success=True)
    trace["tool_executions"] = [
        {
            "attempt_id": "registered-execution",
            "result": {"executed_step_count": 3},
        },
        {
            "attempt_id": "runtime-trial-execution",
            "result": {"executed_step_count": 2},
        },
    ]
    trace["metadata"]["runtime_tool_trials"] = {
        "draft-1": {
            "tool_execution_ids": ["runtime-trial-execution"],
            # Embedded copies are not execution identity authority.
            "result": {"tool_results": [{"executed_step_count": 99}]},
        },
    }
    trace["metadata"]["runtime_automation_funnel"] = {
        # Explicit aggregates likewise cannot invent trial-owned actions.
        "trial_internal_action_count": 77,
        "trial_llm_bypassed_action_count": 77,
    }

    row = trace_to_row(trace)
    assert row["runtime_tool_internal_action_count"] == 5
    funnel = row["runtime_automation_funnel"]
    assert funnel["trial_started"] == 1
    assert funnel["trial_internal_action_count"] == 2
    assert funnel["trial_llm_bypassed_action_count"] == 2


def test_f03_session_snapshot_is_separate_from_final_task_outcome() -> None:
    trace = _trace(
        "terminal-after-budget",
        success=True,
        events=[
            _usage_event(
                "u-over", "runtime-session", "runtime_preparation", 120
            )
        ],
    )
    audit = {
        "session_token_limit_exceeded": True,
        "session_turn_limit_exceeded": False,
        "budget_reason": "token_limit",
        "budget_check_stage": "after_provider_call",
        "budget_error_code": "runtime_node_token_budget_exhausted",
        "used_before": {"turns": 0, "total_tokens": 0},
        "actual_call_usage": {"total_tokens": 120},
        "used_after": {"turns": 1, "total_tokens": 120},
        "configured_limit": {"max_turns": 3, "max_total_tokens": 100},
        "provider_call_recorded": True,
        "returned_action_executed": False,
        "terminal_reconciled_after_session_failure": False,
    }
    trace["agent_sessions"] = [{
        "session_id": "runtime-session",
        "session_type": "RuntimePreparationSession",
        "occurrence_id": "occ-1",
        "snapshot": {
            "usage_bucket": "runtime_preparation",
            "budget_scope": "session",
            "budget": {
                "max_turns": 3,
                "used_turns": 1,
                "max_total_tokens": 100,
                "used_total_tokens": 120,
                "exhaustion_events": [audit],
            },
        },
    }]
    trace["node_records"] = [{
        "occurrence_id": "occ-1",
        "direct_result": {
            "started": True,
            "terminal_interrupted": True,
            "terminal_effect_reconciled": True,
            "atomic_effect_passed": True,
        },
        "seeded_result": {},
    }]
    trace["metadata"]["v32_metrics"] = {
        "runtime_terminal_started_reconciliation_success_count": 1
    }

    row = trace_to_row(trace)
    assert row["benchmark_success"] is True
    assert row["session_token_limit_exceeded_count"] == 1
    assert row["runtime_token_decomposition"]["runtime_preparation"][
        "exhausted_session_count"
    ] == 1
    recorded = row["runtime_budget_exhaustion_audits"][0]
    assert recorded["provider_call_recorded"] is True
    assert recorded["returned_action_executed"] is False
    assert recorded["terminal_reconciled_after_session_failure"] is True


def test_f03_terminal_reconciliation_uses_mode_and_occurrence_authority() -> None:
    trace = _trace("terminal-occurrence-authority", success=True)

    def session(
        session_id: str,
        occurrence_id: str,
        bucket: str = "runtime_preparation",
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "session_type": (
                "RuntimeSeededSession"
                if bucket == "runtime_seeded"
                else "RuntimePreparationSession"
            ),
            "occurrence_id": occurrence_id,
            "snapshot": {
                "usage_bucket": bucket,
                "budget": {
                    "max_turns": 3,
                    "used_turns": 1,
                    "max_total_tokens": 100,
                    "used_total_tokens": 120,
                    "exhaustion_events": [{
                        "session_token_limit_exceeded": True,
                        "session_turn_limit_exceeded": False,
                        "budget_reason": "token_limit",
                        "budget_check_stage": "after_provider_call",
                        "provider_call_recorded": True,
                        "returned_action_executed": False,
                        "terminal_reconciled_after_session_failure": False,
                    }],
                },
            },
        }

    trace["agent_sessions"] = [
        session("session-reconciled", "occ-reconciled"),
        session("session-unrelated", "occ-unrelated"),
        session("session-seeded", "occ-seeded", "runtime_seeded"),
    ]
    trace["node_records"] = [
        {
            "occurrence_id": "occ-reconciled",
            "direct_result": {
                "started": True,
                "terminal_interrupted": True,
                "terminal_effect_reconciled": True,
                "atomic_effect_passed": True,
            },
        },
        {
            "occurrence_id": "occ-unrelated",
            "direct_result": {
                "started": True,
                "terminal_interrupted": False,
                "terminal_effect_reconciled": False,
                "atomic_effect_passed": False,
            },
        },
        {
            "occurrence_id": "occ-seeded",
            "direct_result": {},
            "seeded_result": {
                "started": False,
                "terminal_interrupted": False,
                "terminal_effect_reconciled": True,
                "atomic_effect_passed": True,
            },
        },
    ]
    trace["metadata"]["v32_metrics"] = {
        "runtime_terminal_started_reconciliation_success_count": 2,
    }

    audits = trace_to_row(trace)["runtime_budget_exhaustion_audits"]
    reconciled = {
        item["session_id"]: item[
            "terminal_reconciled_after_session_failure"
        ]
        for item in audits
    }
    assert reconciled == {
        "session-reconciled": True,
        "session-unrelated": False,
        "session-seeded": True,
    }


def _no_tool_trace(*, static_passed: bool, explicit_field: str) -> dict:
    trace = _trace("automation-no-tool", success=True)
    trace["metadata"]["runtime_automation_drafts"] = {
        "draft-no-tool": {
            "failure_code": "runtime_automation_no_tool",
            "stage": "builder_no_tool",
            "proposal": {"decision": "no_tool"},
            # Historical projections used both boolean values for NO_TOOL.
            "static_passed": static_passed,
        },
    }
    trace["metadata"]["runtime_automation_funnel"] = {
        "no_tool": 1,
        explicit_field: 1,
    }
    return trace


def test_f02_no_tool_is_not_counted_as_explicit_static_pass() -> None:
    trace = _no_tool_trace(static_passed=True, explicit_field="static_pass")

    funnel = trace_to_row(trace)["runtime_automation_funnel"]
    assert funnel["no_tool"] == 1
    assert funnel["static_pass"] == 0
    assert funnel["static_reject"] == 0


def test_f02_no_tool_is_not_counted_as_explicit_static_reject() -> None:
    trace = _no_tool_trace(static_passed=False, explicit_field="static_reject")

    funnel = trace_to_row(trace)["runtime_automation_funnel"]
    assert funnel["no_tool"] == 1
    assert funnel["static_pass"] == 0
    assert funnel["static_reject"] == 0


def test_f05_learning_status_does_not_call_frozen_or_policy_skip_legacy() -> None:
    frozen = _trace("frozen", success=True)
    frozen["extraction_policy"] = {
        "should_extract": False,
        "reasons": ["frozen_mode_disabled"],
    }
    skipped = _trace("skipped", success=True)
    skipped["extraction_policy"] = {
        "should_extract": False,
        "reasons": ["already_covered"],
    }
    legacy = _trace("legacy", success=True)
    executed = _trace("executed", success=True)
    executed_obj = SimpleNamespace(**executed)
    AtomicSkillGraphSystem._initialize_r4_learning_diagnostics(executed_obj)

    assert trace_to_row(frozen)["learning_diagnostics_status"] == (
        LEARNING_NOT_APPLICABLE_FROZEN
    )
    assert trace_to_row(skipped)["learning_diagnostics_status"] == (
        LEARNING_SKIPPED_BY_POLICY
    )
    assert trace_to_row(legacy)["learning_diagnostics_status"] == (
        LEGACY_DIAGNOSTICS_UNAVAILABLE
    )
    assert trace_to_row(executed_obj)["learning_diagnostics_status"] == (
        LEARNING_EXECUTED_R4
    )

    summary = summarize_traces([frozen, skipped, legacy, executed_obj])
    assert summary["learning_diagnostics_status_counts"] == {
        LEARNING_EXECUTED_R4: 1,
        LEARNING_NOT_APPLICABLE_FROZEN: 1,
        LEARNING_SKIPPED_BY_POLICY: 1,
        LEGACY_DIAGNOSTICS_UNAVAILABLE: 1,
    }
    assert {
        version: group["task_count"]
        for version, group in summary["learning_diagnostics_by_version"].items()
    } == {
        "legacy_mixed_or_incomplete": 1,
        "v3.2-r4": 1,
    }


def test_f06_runtime_tool_builder_aliases_share_one_bucket() -> None:
    assert resolve_tool_builder_usage_bucket("tool_builder_runtime") is (
        UsageBucket.TOOL_BUILDER_RUNTIME
    )
    assert resolve_tool_builder_usage_bucket("runtime_automation") is (
        UsageBucket.TOOL_BUILDER_RUNTIME
    )
    assert resolve_tool_builder_usage_bucket("tool_builder_evolution") is (
        UsageBucket.TOOL_BUILDER_EVOLUTION
    )


def test_r92_release_revision_is_explicit_without_changing_method_patch() -> None:
    assert _trace_release_metadata({
        "method_patch": "3.2",
        "repair_revision": "R9.2",
    }) == {
        "method_patch": "3.2",
        "repair_revision": "R9.2",
    }
    assert _trace_release_metadata({"method_patch": "3.2"}) == {
        "method_patch": "3.2",
    }


def test_f06_typed_tool_replay_diagnostics_reconcile_in_all_reports() -> None:
    trace = _trace("typed-replays", success=True)
    trace["metadata"]["tool_replay_results"] = [
        {
            "case_id": "case-source",
            "source_trace_id": "trace-source",
            "source_task_id": "task-source",
            "requested_task_id": "task-requested",
            "resolved_task_id": "task-other",
            "stage": "source_resolution",
            "passed": False,
            "failure_code": "replay_source_task_mismatch",
            "message": "wrong source task",
            "started": False,
            "executed_action_count": 0,
            "completed": False,
            "terminal_interrupted": False,
            "atomic_effect_passed": False,
            "output_validation_passed": False,
        },
        {
            "case_id": "case-prefix",
            "stage": "prefix",
            "passed": False,
            "failure_code": "replay_prefix_rejected",
            "terminal_interrupted": False,
        },
        {
            "case_id": "case-tool",
            "stage": "tool",
            "passed": False,
            "failure_code": "tool_ir_replay_terminal_interrupted",
            "terminal_interrupted": True,
        },
        {
            "case_id": "case-final-failure",
            "stage": "final_validation",
            "passed": False,
            "failure_code": "tool_ir_replay_atomic_effect_failed",
            "terminal_interrupted": False,
        },
        {
            "case_id": "case-pass",
            "stage": "final_validation",
            "passed": True,
            "failure_code": "",
            "terminal_interrupted": False,
        },
    ]
    legacy = _trace("legacy-replay", success=False)
    legacy["metadata"]["tool_replay_results"] = [{
        "case_id": "legacy-case",
        "stage": "legacy_callback",
        "passed": False,
        "failure_code": "legacy_replay_failed",
    }]

    row = trace_to_row(trace)
    assert row["tool_replay_results"] == trace["metadata"][
        "tool_replay_results"
    ]
    assert row["tool_replay_case_count"] == 5
    assert row["tool_replay_pass_count"] == 1
    assert row["tool_replay_failure_count"] == 4
    assert row["tool_replay_source_task_mismatch_count"] == 1
    assert row["tool_replay_prefix_failure_count"] == 1
    assert row["tool_replay_tool_execution_failure_count"] == 1
    assert row["tool_replay_final_validation_failure_count"] == 1
    assert row["tool_replay_terminal_interrupted_count"] == 1
    assert row["legacy_replay_reason_unavailable_count"] == 0
    assert row["tool_replay_stage_distribution"] == {
        "final_validation": 2,
        "prefix": 1,
        "source_resolution": 1,
        "tool": 1,
    }
    assert row["tool_replay_failure_code_distribution"] == {
        "replay_prefix_rejected": 1,
        "replay_source_task_mismatch": 1,
        "tool_ir_replay_atomic_effect_failed": 1,
        "tool_ir_replay_terminal_interrupted": 1,
    }

    summary = summarize_traces([trace, legacy])
    assert summary["tool_replay_case_count"] == 6
    assert summary["tool_replay_pass_count"] == 1
    assert summary["tool_replay_failure_count"] == 5
    assert summary["tool_replay_source_task_mismatch_count"] == 1
    assert summary["tool_replay_prefix_failure_count"] == 1
    assert summary["tool_replay_tool_execution_failure_count"] == 1
    assert summary["tool_replay_final_validation_failure_count"] == 1
    assert summary["tool_replay_terminal_interrupted_count"] == 1
    assert summary["legacy_replay_reason_unavailable_count"] == 1
    assert summary["tool_replay_stage_distribution"] == {
        "final_validation": 2,
        "legacy_callback": 1,
        "prefix": 1,
        "source_resolution": 1,
        "tool": 1,
    }
    assert summary["tool_replay_failure_code_distribution"][
        "legacy_replay_reason_unavailable"
    ] == 1

    markdown = render_markdown(summary, [row, trace_to_row(legacy)])
    assert "## Tool admission replay diagnostics" in markdown
    assert "| tool_replay_source_task_mismatch_count | 1 |" in markdown
    assert "| tool_replay_prefix_failure_count | 1 |" in markdown
    assert "| tool_replay_tool_execution_failure_count | 1 |" in markdown
    assert "| tool_replay_final_validation_failure_count | 1 |" in markdown
    assert "legacy_replay_reason_unavailable" in markdown


@pytest.mark.parametrize(
    "failure_code",
    [
        "replay_source_current_trace_conflict",
        "replay_source_task_conflict",
        "replay_source_manifest_conflict",
    ],
)
def test_replay_source_authority_production_conflicts_are_counted(
    failure_code: str,
) -> None:
    trace = _trace(f"source-{failure_code}", success=False)
    trace["metadata"]["tool_replay_results"] = [{
        "case_id": failure_code,
        "stage": "source_resolution",
        "passed": False,
        "failure_code": failure_code,
    }]

    row = trace_to_row(trace)
    assert row["tool_replay_source_task_mismatch_count"] == 1
    assert row["legacy_replay_reason_unavailable_count"] == 0


def test_auxiliary_attempt_replay_remains_visible_without_becoming_task_row() -> None:
    formal = _trace("formal", success=True)
    auxiliary = _trace("failed-attempt", success=False)
    auxiliary["metadata"]["tool_replay_results"] = [{
        "case_id": "auxiliary-case",
        "stage": "tool",
        "passed": False,
        "failure_code": "tool_ir_replay_failed",
    }]

    summary = summarize_traces(
        [formal],
        auxiliary_usage_traces=[auxiliary],
    )

    assert summary["task_count"] == 1
    assert summary["solved_task_count"] == 1
    assert summary["tool_replay_case_count"] == 1
    assert summary["tool_replay_failure_count"] == 1
    assert summary["tool_replay_stage_distribution"] == {"tool": 1}
