"""Deterministic JSONL, CSV, and Markdown reports derived from v3 traces.

The reporter never re-parses logs.  It consumes the structured ``TraceRecord``
schema (or its JSON representation), preserves every token-accounting bucket,
and exposes the reconciliation invariants required by the experiment design.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


USAGE_BUCKETS = (
    "planner_p1",
    "planner_p1_repair",
    "planner_p2",
    "planner_p2_repair",
    "cold_start_c1",
    "cold_start_c1_repair",
    "runtime_preparation",
    "runtime_seeded",
    "runtime_dynamic",
    "runtime_provisional_seeded",
    "runtime_dynamic_cold_start_continuation",
    "extractor_e1",
    "extractor_e2",
    "tool_builder_runtime",
    "tool_builder_evolution",
    "failure_extractor_f1",
    "failure_extractor_f2",
    "evolution_repair",
    "unattributed",
)

_RUNTIME_USAGE_BUCKETS = (
    "runtime_preparation",
    "runtime_seeded",
    "runtime_dynamic",
    "runtime_dynamic_cold_start_continuation",
    "runtime_provisional_seeded",
)

_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "call_count",
    "latency_ms",
)

_COMPLETED_NODE_STATUSES = frozenset(
    {
        "already_satisfied",
        "direct_autonomous_success",
        "direct_agent_prepared_success",
        "agent_completed_before_invocation",
        "seeded_success",
    }
)
_EXECUTED_NODE_STATUSES = _COMPLETED_NODE_STATUSES - {"already_satisfied"}
_FAILED_NODE_STATUSES = frozenset(
    {"failed_not_started", "direct_failed", "seeded_failed"}
)

EXTRACTOR_QUALITY_METRICS = (
    "extractor_e1_proposal_count",
    "extractor_e1_validated_occurrence_count",
    "extractor_e1_rejection_count",
    "portable_intent_pass_count",
    "portable_intent_fallback_count",
    "known_contract_name_reuse_count",
    "new_canonical_intent_count",
    "atomic_alignment_reuse_count",
    "atomic_new_contract_count",
    "composite_alignment_reuse_count",
    "artifact_label_concrete_term_violation_count",
)

V31_METHOD_METRICS = (
    "repeat_block_count",
    "expanded_requirement_instance_count",
    "repeated_atomic_occurrence_count",
    "runtime_distinctness_rejection_count",
    "runtime_shared_value_rejection_count",
    "cold_start_trigger_count",
    "cold_start_plan_valid_count",
    "cold_start_c1_validation_pass_count",
    "cold_start_executable_prefix_nonempty_count",
    "cold_start_executable_prefix_empty_count",
    "cold_start_admitted_prefix_step_count",
    "cold_start_executed_scaffold_step_count",
    "cold_start_continuation_only_count",
    "cold_start_scaffold_step_count",
    "cold_start_verified_step_success_count",
    "provisional_trial_count",
    "provisional_local_success_count",
    "provisional_local_failure_count",
    "cold_start_assisted_success_count",
    "runtime_dynamic_cold_start_continuation_count",
    "failure_extractor_f1_count",
    "failure_extractor_f2_count",
    "failure_extractor_eligible_count",
    "provisional_created_count",
    "provisional_trial_ready_count",
    "provisional_trial_supported_count",
    "provisional_promoted_count",
    "provisional_suppressed_count",
    "failure_experience_observed_count",
    "failure_experience_confirmed_count",
    "failure_experience_resolved_count",
    "failure_experience_retrieval_count",
    "verified_atomic_full_coverage_count",
    "planner_p2_count",
    "p0_exact_contract_rejection_count",
    "failure_side_read_count",
    "provisional_selected_count",
)

R21_RUNTIME_METRICS = (
    "runtime_exploration_action_count",
    "runtime_atomic_attempt_action_count",
    "runtime_validate_current_atomic_count",
    "runtime_validate_current_atomic_success_count",
    "learned_invocation_selected_count",
    "runtime_plan_conflict_count",
    "runtime_plan_conflict_occurrence_count",
    "runtime_plan_conflict_rescue_count",
    "runtime_plan_conflict_rescue_success_count",
    "replay_catalog_compaction_count",
    "replay_initial_catalog_compacted_count",
    "replay_history_action_count",
)

R31_RUNTIME_METRICS = (
    "runtime_grounding_refresh_count",
    "runtime_unique_binding_auto_confirm_count",
    "runtime_unique_binding_auto_confirm_role_count",
    "runtime_invocation_ready_transition_count",
    "runtime_effect_ready_transition_count",
    "replay_action_window_size",
    "replay_action_window_compaction_count",
    "replay_pruned_action_count",
    "runtime_context_snapshot_count",
    "runtime_exploration_memory_projection_count",
    "runtime_recent_action_projection_count",
    "partial_atomic_admission_count",
    "partial_atomic_alignment_reuse_count",
    "partial_atomic_new_contract_count",
    "partial_atomic_tool_admission_count",
    "partial_atomic_implementation_admission_count",
)

V32_METHOD_METRICS = (
    "planner_repairability_gate_count",
    "planner_repairability_repairable_count",
    "planner_hard_capability_gap_count",
    "planner_p1r_skipped_hard_gap_count",
    "planner_support_atomic_candidate_count",
    "planner_support_atomic_selected_count",
    "task_terminal_early_success_count",
    "task_terminal_during_tool_count",
    "task_terminal_with_remaining_occurrences_count",
    "terminal_skipped_occurrence_count",
    "extractor_noncontiguous_atomic_count",
    "extractor_support_event_count",
    "extractor_envelope_event_count",
    "extractor_redundant_envelope_event_excluded_count",
    "tool_builder_call_count",
    "tool_builder_no_tool_count",
    "tool_builder_proposal_count",
    "tool_builder_static_pass_count",
    "tool_builder_static_rejection_count",
    "runtime_automation_atomic_proposal_count",
    "runtime_automation_r0_pass_count",
    "runtime_automation_r0_reject_count",
    "runtime_tool_trial_count",
    "runtime_tool_trial_r1_pass_count",
    "runtime_tool_trial_r1_reject_count",
    "runtime_tool_internal_action_count",
    "runtime_tool_llm_bypassed_action_count",
    "runtime_support_retrieval_count",
    "runtime_support_candidate_count",
    "runtime_support_selected_count",
    "runtime_support_success_count",
    "runtime_graph_augmentation_count",
    "tool_validated_path_count",
    "tool_unvalidated_path_count",
    "tool_observed_loop_iteration_count",
    "tool_stop_condition_witness_count",
)

R22_FAILURE_EXTRACTOR_METRICS = (
    "failure_extractor_f1_input_event_count",
    "failure_extractor_f1_prompt_chars",
    "failure_extractor_f1_prompt_bytes",
    "failure_extractor_f1_tokens",
    "failure_extractor_f2_span_count",
    "failure_extractor_f2_source_event_count",
    "failure_extractor_f2_prompt_chars",
    "failure_extractor_f2_prompt_bytes",
    "failure_extractor_f2_tokens",
    "failure_extractor_budget_exhausted_count",
    "failure_extractor_skipped_after_budget_count",
    "failure_extractor_f1_provider_call_count",
    "failure_extractor_f2_provider_call_count",
    "failure_extractor_usage_persisted_after_rejection_count",
    "failure_extractor_remaining_budget_before_f2",
)

_FROZEN_FORBIDDEN_COLD_START_BUCKETS = (
    "cold_start_c1",
    "cold_start_c1_repair",
    "runtime_provisional_seeded",
    "runtime_dynamic_cold_start_continuation",
    "failure_extractor_f1",
    "failure_extractor_f2",
)

REPORT_COLUMNS = (
    "trace_id",
    "schema_version",
    "task_id",
    "task_signature",
    "benchmark",
    "task_type",
    "benchmark_success",
    "task_contract_success",
    "strict_task_success",
    "node_contract_success",
    "implementation_direct_success",
    "graph_self_sufficient_success",
    "graph_full_completion",
    "learning_eligible",
    "infrastructure_failure",
    "resource_usage_complete",
    "plan_source",
    "source_composite_ref",
    "planner_outcome",
    "planner_fallback_reason",
    "planner_requirement_repair_used",
    "planner_graph_repair_used",
    "planner_atomic_full_coverage",
    "planner_p2_used",
    "planner_p1r_reason_distribution",
    "p0_rejection_stage_distribution",
    "atomic_contract_mismatch_reason_distribution_p1",
    "atomic_contract_mismatch_reason_distribution_p1r",
    "confirmed_capability_gap",
    "full_dynamic",
    "task_rescue_required",
    "planned_node_count",
    "recorded_node_count",
    "completed_node_count",
    "executed_node_count",
    "already_satisfied_count",
    "direct_autonomous_success_count",
    "direct_agent_prepared_success_count",
    "agent_completed_before_invocation_count",
    "seeded_success_count",
    "failed_node_count",
    "skipped_goal_terminal_count",
    "implementation_invocation_count",
    "implementation_started_count",
    "implementation_completed_count",
    "implementation_preflight_rejected_count",
    "tool_execution_count",
    "tool_started_count",
    "tool_completed_count",
    "environment_action_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "call_count",
    "llm_latency_ms",
    "episode_total_tokens",
    "real_bucket_total_tokens",
    "unattributed_total_tokens",
    "token_mismatch",
    *(f"{bucket}_total_tokens" for bucket in USAGE_BUCKETS if bucket != "unattributed"),
    "usage_by_bucket",
    "duration_ms",
    "cost_usd",
    "failure_codes",
    "runtime_failure_diagnostic",
    "task_token_budget_exhausted_count",
    "node_token_budget_exhausted_count",
    *R21_RUNTIME_METRICS,
    *R31_RUNTIME_METRICS,
    *V32_METHOD_METRICS,
    "replay_full_catalog_count_at_last_request",
    "runtime_prompt_tokens",
    "runtime_completion_tokens",
    "runtime_reasoning_tokens",
    "runtime_reasoning_share",
    "runtime_token_decomposition",
    *R22_FAILURE_EXTRACTOR_METRICS,
    *V31_METHOD_METRICS,
    *EXTRACTOR_QUALITY_METRICS,
    "extraction_attempted",
    "extraction_stage",
    "extraction_prepared",
    "extraction_applied",
    "extraction_error_code",
    "e1_proposed",
    "e1_validated",
    "e1_rejected",
    "e1_contract_coverage_passed",
    "e2_attempted",
    "e2_selected_existing_edges",
    "e2_selected_new_edges",
    "artifact_growth",
    "artifact_lifecycle",
)


@dataclass(frozen=True)
class ReportPaths:
    jsonl: Path
    csv: Path
    markdown: Path


def trace_to_row(trace: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert one structured v3 trace into a stable experiment row."""

    task = _mapping(_field(trace, "task", {}))
    planner = _mapping(_field(trace, "planner_audit", {}))
    plan = _mapping(_field(trace, "runtime_plan", {}))
    metadata = _mapping(_field(trace, "metadata", {}))
    quality = _mapping(metadata.get("extractor_quality", {}))
    nodes = [_mapping(item) for item in _sequence(_field(trace, "node_records", []))]
    statuses = [_enum_value(item.get("status", "not_started")) for item in nodes]

    usage = _usage_report(trace, metadata)
    invocations = [
        _mapping(item)
        for item in _sequence(_field(trace, "implementation_invocations", []))
    ]
    executions = [
        _mapping(item) for item in _sequence(_field(trace, "tool_executions", []))
    ]
    planned = _planned_node_count(plan, nodes)
    started_at = _number(_field(trace, "started_at", 0.0), 0.0)
    ended_at = _number(_field(trace, "ended_at", 0.0), 0.0)

    benchmark_success = _boolean(
        _field(trace, "benchmark_success", False)
    )
    task_contract_raw = _field(trace, "task_contract_success", None)
    task_contract_success = (
        benchmark_success
        if task_contract_raw is None
        else _boolean(task_contract_raw)
    )
    strict_raw = _field(trace, "strict_task_success", None)
    strict_task_success = (
        benchmark_success
        if strict_raw is None
        else _boolean(strict_raw)
    )
    failure_codes = _failure_codes(trace, nodes, invocations, executions)
    planner_atomic_full_coverage = _planner_atomic_full_coverage(planner)
    planner_p2_used = (
        _has_value(planner.get("workflow_p2"))
        or _integer(
            _mapping(usage["by_bucket"].get("planner_p2", {})).get(
                "call_count", 0,
            )
        ) > 0
    )
    v31_metrics = _v31_method_metrics(
        trace,
        planner=planner,
        plan=plan,
        metadata=metadata,
        usage=usage,
        failure_codes=failure_codes,
        planner_atomic_full_coverage=planner_atomic_full_coverage,
        planner_p2_used=planner_p2_used,
    )
    extraction = _extraction_diagnostic(trace, metadata, quality, usage)
    failure_extractor = _failure_extractor_diagnostic(
        trace, metadata=metadata, usage=usage,
    )
    runtime_failure = _runtime_failure_diagnostic(
        trace,
        plan=plan,
        nodes=nodes,
        invocations=invocations,
        failure_codes=failure_codes,
        strict_task_success=strict_task_success,
    )
    r21_runtime = _r21_runtime_metrics(
        trace,
        usage=usage,
        strict_task_success=strict_task_success,
        task_rescue_required=_boolean(
            _field(trace, "task_rescue_required", False)
        ),
    )
    r31_runtime = _r31_event_metrics(trace, metadata=metadata)
    v32_metrics = _v32_method_metrics(
        trace,
        planner=planner,
        metadata=metadata,
        nodes=nodes,
        invocations=invocations,
        executions=executions,
    )
    row: dict[str, Any] = {
        "trace_id": str(_field(trace, "trace_id", "")),
        "schema_version": _integer(_field(trace, "schema_version", 0)),
        "task_id": str(task.get("task_id", "")),
        "task_signature": str(task.get("task_signature", "")),
        "benchmark": str(task.get("benchmark", "")),
        "task_type": str(task.get("task_type", "")),
        "benchmark_success": benchmark_success,
        "task_contract_success": task_contract_success,
        "strict_task_success": strict_task_success,
        "node_contract_success": _boolean(
            _field(trace, "node_contract_success", False)
        ),
        "implementation_direct_success": _boolean(
            _field(trace, "implementation_direct_success", False)
        ),
        "graph_self_sufficient_success": _boolean(
            _field(trace, "graph_self_sufficient_success", False)
        ),
        "graph_full_completion": _boolean(
            _field(trace, "graph_full_completion", False)
        ),
        "learning_eligible": _boolean(_field(trace, "learning_eligible", False)),
        "infrastructure_failure": _boolean(
            _field(trace, "infrastructure_failure", False)
        ),
        "resource_usage_complete": _boolean(
            _field(trace, "resource_usage_complete", True)
        ),
        "plan_source": str(plan.get("source", "")),
        "source_composite_ref": plan.get("source_composite_ref") or "",
        "planner_outcome": str(planner.get("final_outcome", "")),
        "planner_fallback_reason": str(planner.get("fallback_reason", "")),
        "planner_requirement_repair_used": _has_value(
            planner.get("requirements_p1r")
        )
        or _has_value(planner.get("atomic_search_p1r")),
        "planner_graph_repair_used": _has_value(planner.get("workflow_p2r"))
        or _has_value(planner.get("validation_p2r")),
        "planner_atomic_full_coverage": planner_atomic_full_coverage,
        "planner_p2_used": planner_p2_used,
        "planner_p1r_reason_distribution": _planner_p1r_reasons(planner),
        "p0_rejection_stage_distribution": _p0_rejection_stages(planner),
        "atomic_contract_mismatch_reason_distribution_p1": (
            _atomic_contract_mismatch_reasons(planner, "atomic_search_p1")
        ),
        "atomic_contract_mismatch_reason_distribution_p1r": (
            _atomic_contract_mismatch_reasons(planner, "atomic_search_p1r")
        ),
        "confirmed_capability_gap": _confirmed_capability_gap(trace, metadata),
        "full_dynamic": str(plan.get("source", "")) == "full_dynamic",
        "task_rescue_required": _boolean(
            _field(trace, "task_rescue_required", False)
        ),
        "planned_node_count": planned,
        "recorded_node_count": len(nodes),
        "completed_node_count": sum(
            status in _COMPLETED_NODE_STATUSES for status in statuses
        ),
        "executed_node_count": sum(
            status in _EXECUTED_NODE_STATUSES or status in _FAILED_NODE_STATUSES
            for status in statuses
        ),
        "already_satisfied_count": statuses.count("already_satisfied"),
        "direct_autonomous_success_count": statuses.count(
            "direct_autonomous_success"
        ),
        "direct_agent_prepared_success_count": statuses.count(
            "direct_agent_prepared_success"
        ),
        "agent_completed_before_invocation_count": statuses.count(
            "agent_completed_before_invocation"
        ),
        "seeded_success_count": statuses.count("seeded_success"),
        "failed_node_count": sum(status in _FAILED_NODE_STATUSES for status in statuses),
        "skipped_goal_terminal_count": statuses.count("skipped_goal_terminal"),
        "implementation_invocation_count": len(invocations),
        "implementation_started_count": sum(
            _boolean(_mapping(item.get("result", {})).get("started", False))
            for item in invocations
        ),
        "implementation_completed_count": sum(
            _boolean(_mapping(item.get("result", {})).get("completed", False))
            for item in invocations
        ),
        "implementation_preflight_rejected_count": sum(
            not _preflight_passed(item) for item in invocations
        ),
        "tool_execution_count": len(executions),
        "tool_started_count": sum(
            _boolean(_mapping(item.get("result", {})).get("started", False))
            for item in executions
        ),
        "tool_completed_count": sum(
            _boolean(_mapping(item.get("result", {})).get("completed", False))
            for item in executions
        ),
        "environment_action_count": len(
            _sequence(_field(trace, "environment_actions", []))
        ),
        **usage["episode_total"],
        "llm_latency_ms": usage["episode_total"]["latency_ms"],
        **usage["reconciliation"],
        "usage_by_bucket": usage["by_bucket"],
        "duration_ms": (
            round(max(0.0, ended_at - started_at) * 1000.0, 3)
            if ended_at > 0.0 and started_at > 0.0
            else 0.0
        ),
        "cost_usd": _trace_cost(trace, metadata, usage["events"]),
        "failure_codes": failure_codes,
        "runtime_failure_diagnostic": runtime_failure,
        "task_token_budget_exhausted_count": int(
            "runtime_task_token_budget_exhausted" in failure_codes
        ),
        "node_token_budget_exhausted_count": int(
            "runtime_node_token_budget_exhausted" in failure_codes
        ),
        **r21_runtime,
        **r31_runtime,
        **v32_metrics,
        **failure_extractor,
        **v31_metrics,
        **{
            name: _integer(quality.get(name, 0))
            for name in EXTRACTOR_QUALITY_METRICS
        },
        **extraction,
        "artifact_growth": _first_present(
            metadata, "artifact_growth", "artifact_growth_snapshot", default={}
        ),
        "artifact_lifecycle": _first_present(
            metadata, "artifact_lifecycle", "lifecycle_snapshot", default={}
        ),
    }
    # ``latency_ms`` is normalized to the report's explicit LLM-only name.
    row.pop("latency_ms", None)
    for bucket in USAGE_BUCKETS:
        if bucket == "unattributed":
            continue
        row[f"{bucket}_total_tokens"] = int(
            usage["by_bucket"].get(bucket, {}).get("total_tokens", 0)
        )
    return {column: row.get(column) for column in REPORT_COLUMNS}


def summarize_traces(
    traces_or_rows: Iterable[Mapping[str, Any] | Any],
    *,
    auxiliary_usage_traces: Iterable[Mapping[str, Any] | Any] = (),
    run_artifact_growth: Mapping[str, Any] | None = None,
    run_artifact_lifecycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate all v3 design metrics without hiding zero denominators."""

    items = list(traces_or_rows)
    rows = [
        dict(item)
        if isinstance(item, Mapping) and _looks_like_report_row(item)
        else trace_to_row(item)
        for item in items
    ]
    auxiliary_rows = [
        dict(item)
        if isinstance(item, Mapping) and _looks_like_report_row(item)
        else trace_to_row(item)
        for item in auxiliary_usage_traces
    ]
    resource_rows = [*rows, *auxiliary_rows]
    task_rows = rows
    task_count = len(task_rows)
    solved = [
        row for row in task_rows
        if _boolean(row.get("benchmark_success", False))
    ]
    strict = [
        row for row in task_rows
        if _boolean(row.get("strict_task_success", False))
    ]
    learning = [
        row for row in task_rows
        if _boolean(row.get("learning_eligible", False))
    ]
    completed_nodes = sum(
        _integer(row.get("completed_node_count", 0)) for row in task_rows
    )
    planned_nodes = sum(
        _integer(row.get("planned_node_count", 0)) for row in task_rows
    )
    total_tokens = sum(_integer(row.get("total_tokens", 0)) for row in resource_rows)
    total_latency = sum(
        _number(row.get("llm_latency_ms", 0.0), 0.0) for row in resource_rows
    )
    total_duration = sum(
        _number(row.get("duration_ms", 0.0), 0.0) for row in resource_rows
    )
    known_costs = [
        float(row["cost_usd"])
        for row in resource_rows
        if row.get("cost_usd") is not None
    ]
    complete_resource_cost = bool(resource_rows) and (
        len(known_costs) == len(resource_rows)
    )

    by_bucket: dict[str, dict[str, int | float | None]] = {}
    for bucket in USAGE_BUCKETS:
        bucket_items = [
            _mapping(_mapping(row.get("usage_by_bucket", {})).get(bucket, {}))
            for row in resource_rows
        ]
        by_bucket[bucket] = _sum_usage(bucket_items)

    p1r_reason_distribution: dict[str, int] = {}
    for row in task_rows:
        for reason, count in _mapping(
            row.get("planner_p1r_reason_distribution", {})
        ).items():
            p1r_reason_distribution[str(reason)] = (
                p1r_reason_distribution.get(str(reason), 0)
                + _integer(count)
            )

    def merged_distribution(key: str) -> dict[str, int]:
        merged: dict[str, int] = {}
        for row in task_rows:
            for name, count in _mapping(row.get(key, {})).items():
                merged[str(name)] = merged.get(str(name), 0) + _integer(count)
        return dict(sorted(merged.items()))

    p0_stage_distribution = merged_distribution(
        "p0_rejection_stage_distribution"
    )
    atomic_mismatch_p1 = merged_distribution(
        "atomic_contract_mismatch_reason_distribution_p1"
    )
    atomic_mismatch_p1r = merged_distribution(
        "atomic_contract_mismatch_reason_distribution_p1r"
    )
    atomic_mismatch_all = dict(atomic_mismatch_p1)
    for name, count in atomic_mismatch_p1r.items():
        atomic_mismatch_all[name] = atomic_mismatch_all.get(name, 0) + count
    strict_dynamic = [
        row for row in task_rows
        if _boolean(row.get("strict_task_success"))
        and _boolean(row.get("full_dynamic"))
    ]
    runtime_token_decomposition = _aggregate_runtime_token_decomposition(
        resource_rows
    )
    runtime_prompt_tokens = sum(
        _integer(item.get("runtime_prompt_tokens", 0))
        for item in resource_rows
    )
    runtime_completion_tokens = sum(
        _integer(item.get("runtime_completion_tokens", 0))
        for item in resource_rows
    )
    runtime_reasoning_tokens = sum(
        _integer(item.get("runtime_reasoning_tokens", 0))
        for item in resource_rows
    )

    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": task_count,
        "solved_task_count": len(solved),
        "official_alfworld_won_count": len(solved),
        "strict_task_success_count": len(strict),
        "learning_eligible_success_count": len(learning),
        "benchmark_success_rate": _rate(len(solved), task_count),
        "official_alfworld_won_rate": _rate(len(solved), task_count),
        "strict_task_success_rate": _rate(len(strict), task_count),
        "learning_eligible_success_rate": _rate(len(learning), task_count),
        "graph_self_sufficient_success_rate": _rate(
            sum(
                _boolean(row.get("graph_self_sufficient_success"))
                for row in task_rows
            ),
            task_count,
        ),
        "direct_autonomous_rate": _rate(
            sum(
                _integer(row.get("direct_autonomous_success_count", 0))
                for row in task_rows
            ),
            completed_nodes,
        ),
        "direct_agent_prepared_rate": _rate(
            sum(
                _integer(row.get("direct_agent_prepared_success_count", 0))
                for row in task_rows
            ),
            completed_nodes,
        ),
        "agent_completed_before_invocation_rate": _rate(
            sum(
                _integer(row.get("agent_completed_before_invocation_count", 0))
                for row in task_rows
            ),
            completed_nodes,
        ),
        "seeded_success_rate": _rate(
            sum(
                _integer(row.get("seeded_success_count", 0))
                for row in task_rows
            ),
            completed_nodes,
        ),
        "full_dynamic_rate": _rate(
            sum(_boolean(row.get("full_dynamic")) for row in task_rows), task_count
        ),
        "task_rescue_rate": _rate(
            sum(_boolean(row.get("task_rescue_required")) for row in task_rows),
            task_count,
        ),
        "planner_requirement_repair_rate": _rate(
            sum(
                _boolean(row.get("planner_requirement_repair_used"))
                for row in task_rows
            ),
            task_count,
        ),
        "planner_graph_repair_rate": _rate(
            sum(
                _boolean(row.get("planner_graph_repair_used"))
                for row in task_rows
            ),
            task_count,
        ),
        "confirmed_capability_gap_count": sum(
            _boolean(row.get("confirmed_capability_gap")) for row in task_rows
        ),
        "confirmed_capability_gap_rate": _rate(
            sum(
                _boolean(row.get("confirmed_capability_gap"))
                for row in task_rows
            ),
            task_count,
        ),
        "task_token_budget_exhausted_count": sum(
            _integer(row.get("task_token_budget_exhausted_count", 0))
            for row in task_rows
        ),
        "node_token_budget_exhausted_count": sum(
            _integer(row.get("node_token_budget_exhausted_count", 0))
            for row in task_rows
        ),
        **{
            name: sum(_integer(row.get(name, 0)) for row in task_rows)
            for name in R21_RUNTIME_METRICS
        },
        **{
            name: (
                max(
                    (_integer(row.get(name, 0)) for row in task_rows),
                    default=0,
                )
                if name == "replay_action_window_size"
                else sum(_integer(row.get(name, 0)) for row in task_rows)
            )
            for name in R31_RUNTIME_METRICS
        },
        **{
            name: sum(_integer(row.get(name, 0)) for row in task_rows)
            for name in R22_FAILURE_EXTRACTOR_METRICS
        },
        "replay_full_catalog_count_at_last_request": max(
            (
                _integer(row.get("replay_full_catalog_count_at_last_request", 0))
                for row in task_rows
            ),
            default=0,
        ),
        "runtime_prompt_tokens": runtime_prompt_tokens,
        "runtime_completion_tokens": runtime_completion_tokens,
        "runtime_reasoning_tokens": runtime_reasoning_tokens,
        "runtime_reasoning_share": _ratio(
            runtime_reasoning_tokens, runtime_completion_tokens
        ) or 0.0,
        "runtime_token_decomposition": runtime_token_decomposition,
        **{
            name: sum(
                _integer(row.get(name, 0)) for row in task_rows
            )
            for name in V31_METHOD_METRICS
        },
        **{
            name: sum(
                _integer(row.get(name, 0)) for row in task_rows
            )
            for name in EXTRACTOR_QUALITY_METRICS
        },
        "planner_atomic_full_coverage_count": sum(
            _boolean(row.get("planner_atomic_full_coverage"))
            for row in task_rows
        ),
        "planner_p1r_reason_distribution": dict(
            sorted(p1r_reason_distribution.items())
        ),
        "p0_rejection_stage_distribution": p0_stage_distribution,
        "atomic_contract_mismatch_reason_distribution": dict(
            sorted(atomic_mismatch_all.items())
        ),
        "atomic_contract_mismatch_reason_distribution_p1": atomic_mismatch_p1,
        "atomic_contract_mismatch_reason_distribution_p1r": atomic_mismatch_p1r,
        "strict_dynamic_success_count": len(strict_dynamic),
        "strict_dynamic_success_extraction_prepared_count": sum(
            _boolean(row.get("extraction_prepared")) for row in strict_dynamic
        ),
        "strict_dynamic_success_extraction_applied_count": sum(
            _boolean(row.get("extraction_applied")) for row in strict_dynamic
        ),
        "extractor_e1_contract_coverage_failure_count": sum(
            row.get("e1_contract_coverage_passed") is False
            or str(row.get("extraction_error_code", ""))
            == "extractor_e1_task_contract_coverage_incomplete"
            for row in task_rows
        ),
        "extractor_e2_content_rejection_count": sum(
            str(row.get("extraction_error_code", "")).startswith("extractor_e2_")
            for row in task_rows
        ),
        "failed_task_diagnostics": [
            dict(row.get("runtime_failure_diagnostic") or {})
            for row in task_rows
            if _has_value(row.get("runtime_failure_diagnostic"))
        ],
        "planned_node_count": planned_nodes,
        "completed_node_count": completed_nodes,
        "total_tokens": total_tokens,
        "tokens_per_task": _ratio(total_tokens, task_count),
        "tokens_per_solved_task": _ratio(
            sum(_integer(row.get("total_tokens", 0)) for row in solved)
            + sum(_integer(row.get("total_tokens", 0)) for row in auxiliary_rows),
            len(solved),
        ),
        "llm_latency_ms": round(total_latency, 3),
        "llm_latency_ms_per_solved_task": _ratio(
            sum(_number(row.get("llm_latency_ms", 0.0), 0.0) for row in solved)
            + sum(
                _number(row.get("llm_latency_ms", 0.0), 0.0)
                for row in auxiliary_rows
            ),
            len(solved),
        ),
        "wall_duration_ms": round(total_duration, 3),
        "wall_duration_ms_per_solved_task": _ratio(
            sum(_number(row.get("duration_ms", 0.0), 0.0) for row in solved)
            + sum(
                _number(row.get("duration_ms", 0.0), 0.0)
                for row in auxiliary_rows
            ),
            len(solved),
        ),
        "cost_usd": (
            round(sum(known_costs), 9) if complete_resource_cost else None
        ),
        "cost_usd_per_solved_task": (
            round(sum(known_costs) / len(solved), 9)
            if complete_resource_cost and solved
            else None
        ),
        "token_mismatch": sum(
            _integer(row.get("token_mismatch", 0)) for row in resource_rows
        ),
        "unattributed_total_tokens": sum(
            _integer(row.get("unattributed_total_tokens", 0))
            for row in resource_rows
        ),
        "usage_by_bucket": by_bucket,
        # Task rows retain task-local snapshots.  Run-level evidence must be
        # supplied explicitly by the formal runner from immutable Manifest
        # authority; the reporter never guesses it from the last task.
        "artifact_growth": dict(run_artifact_growth or {}),
        "artifact_lifecycle": dict(run_artifact_lifecycle or {}),
    }


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    payload = "".join(
        _canonical_json(dict(row)) + "\n" for row in rows
    ).encode("utf-8")
    _atomic_write(target, payload)
    return target


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    normalized = [dict(row) for row in rows]
    temporary = _temporary_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row in normalized:
                writer.writerow(
                    {
                        column: _csv_value(row.get(column))
                        for column in REPORT_COLUMNS
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def render_markdown(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "AtomicSkillGraph v3 Experiment Report",
) -> str:
    """Render a compact human audit while JSONL/CSV remain authoritative."""

    lines = [f"# {title}", "", "## Outcome", ""]
    outcome = (
        ("Tasks", summary.get("task_count")),
        ("Official ALFWorld won", _percent(summary.get("official_alfworld_won_rate"))),
        ("Strict TaskContract success", _percent(summary.get("strict_task_success_rate"))),
        ("Learning-eligible success", _percent(summary.get("learning_eligible_success_rate"))),
        (
            "Graph self-sufficient success",
            _percent(summary.get("graph_self_sufficient_success_rate")),
        ),
        ("Full Dynamic", _percent(summary.get("full_dynamic_rate"))),
        ("Task rescue", _percent(summary.get("task_rescue_rate"))),
        ("Confirmed capability gaps", summary.get("confirmed_capability_gap_count")),
    )
    lines.extend(_markdown_pairs(outcome))
    lines.extend(["", "## Node execution", ""])
    nodes = (
        ("Planned nodes", summary.get("planned_node_count")),
        ("Completed nodes", summary.get("completed_node_count")),
        ("Direct autonomous", _percent(summary.get("direct_autonomous_rate"))),
        ("Direct agent-prepared", _percent(summary.get("direct_agent_prepared_rate"))),
        (
            "Agent completed before invocation",
            _percent(summary.get("agent_completed_before_invocation_rate")),
        ),
        ("Seeded success", _percent(summary.get("seeded_success_rate"))),
        (
            "Planner requirement repair",
            _percent(summary.get("planner_requirement_repair_rate")),
        ),
        ("Planner graph repair", _percent(summary.get("planner_graph_repair_rate"))),
        ("Planner Atomic full coverage", summary.get("planner_atomic_full_coverage_count")),
        ("Planner P2", summary.get("planner_p2_count")),
        ("Task token exhaustion", summary.get("task_token_budget_exhausted_count")),
        ("Node token exhaustion", summary.get("node_token_budget_exhausted_count")),
    )
    lines.extend(_markdown_pairs(nodes))
    lines.extend(["", "## Extractor and knowledge quality", ""])
    quality = tuple(
        (name, summary.get(name, 0))
        for name in EXTRACTOR_QUALITY_METRICS
    ) + (
        (
            "planner_p1r_reason_distribution",
            _canonical_json(summary.get("planner_p1r_reason_distribution", {})),
        ),
        (
            "p0_rejection_stage_distribution",
            _canonical_json(summary.get("p0_rejection_stage_distribution", {})),
        ),
        (
            "atomic_contract_mismatch_reason_distribution_p1",
            _canonical_json(summary.get(
                "atomic_contract_mismatch_reason_distribution_p1", {}
            )),
        ),
        (
            "atomic_contract_mismatch_reason_distribution_p1r",
            _canonical_json(summary.get(
                "atomic_contract_mismatch_reason_distribution_p1r", {}
            )),
        ),
        ("strict_dynamic_success_count", summary.get("strict_dynamic_success_count", 0)),
        (
            "strict_dynamic_success_extraction_prepared_count",
            summary.get("strict_dynamic_success_extraction_prepared_count", 0),
        ),
        (
            "strict_dynamic_success_extraction_applied_count",
            summary.get("strict_dynamic_success_extraction_applied_count", 0),
        ),
        (
            "extractor_e1_contract_coverage_failure_count",
            summary.get("extractor_e1_contract_coverage_failure_count", 0),
        ),
        (
            "extractor_e2_content_rejection_count",
            summary.get("extractor_e2_content_rejection_count", 0),
        ),
    )
    lines.extend(_markdown_pairs(quality))
    lines.extend(["", "## v3.1 method patch", ""])
    lines.extend(_markdown_pairs(tuple(
        (name, summary.get(name, 0))
        for name in V31_METHOD_METRICS
    )))
    lines.extend(["", "## R2.2 Failure Extractor diagnostics", ""])
    lines.extend(_markdown_pairs(tuple(
        (name, summary.get(name, 0))
        for name in R22_FAILURE_EXTRACTOR_METRICS
    )))
    lines.extend(["", "## R2.1 Runtime decision and replay", ""])
    lines.extend(_markdown_pairs(tuple(
        (name, summary.get(name, 0))
        for name in R21_RUNTIME_METRICS
    ) + (
        (
            "replay_full_catalog_count_at_last_request",
            summary.get("replay_full_catalog_count_at_last_request", 0),
        ),
        ("runtime_prompt_tokens", summary.get("runtime_prompt_tokens", 0)),
        (
            "runtime_completion_tokens",
            summary.get("runtime_completion_tokens", 0),
        ),
        ("runtime_reasoning_tokens", summary.get("runtime_reasoning_tokens", 0)),
        ("runtime_reasoning_share", summary.get("runtime_reasoning_share", 0.0)),
    )))
    lines.extend(["", "## R3.1 Runtime context and observability", ""])
    lines.extend(_markdown_pairs(tuple(
        (name, summary.get(name, 0))
        for name in R31_RUNTIME_METRICS
    )))
    lines.extend(["", "### Runtime token decomposition", ""])
    lines.append(
        "| Bucket | Calls | Prompt | Completion | Reasoning | Avg total/call | "
        "Max total/call | Exhausted sessions |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    runtime_usage = _mapping(summary.get("runtime_token_decomposition", {}))
    for bucket in _RUNTIME_USAGE_BUCKETS:
        item = _mapping(runtime_usage.get(bucket, {}))
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                bucket,
                item.get("calls", 0),
                item.get("prompt_tokens", 0),
                item.get("completion_tokens", 0),
                item.get("reasoning_tokens", 0),
                _display(item.get("average_tokens_per_call", 0.0)),
                item.get("max_tokens_per_call", 0),
                item.get("exhausted_session_count", 0),
            )
        )
    lines.extend(["", "## Token, latency, and cost", ""])
    accounting = (
        ("Total tokens", summary.get("total_tokens")),
        ("Tokens / solved task", _display(summary.get("tokens_per_solved_task"))),
        (
            "LLM latency ms / solved task",
            _display(summary.get("llm_latency_ms_per_solved_task")),
        ),
        ("Cost USD / solved task", _display(summary.get("cost_usd_per_solved_task"))),
        ("Token mismatch", summary.get("token_mismatch")),
        ("Unattributed total tokens", summary.get("unattributed_total_tokens")),
    )
    lines.extend(_markdown_pairs(accounting))
    lines.extend(["", "### Per-agent usage buckets", ""])
    lines.append("| Bucket | Calls | Prompt | Completion | Total | Reasoning | Latency ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    usage = _mapping(summary.get("usage_by_bucket", {}))
    for bucket in USAGE_BUCKETS:
        item = _mapping(usage.get(bucket, {}))
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                bucket,
                item.get("call_count", 0),
                item.get("prompt_tokens", 0),
                item.get("completion_tokens", 0),
                item.get("total_tokens", 0),
                _display(item.get("reasoning_tokens")),
                _display(item.get("latency_ms", 0.0)),
            )
        )

    lifecycle = summary.get("artifact_lifecycle")
    growth = summary.get("artifact_growth")
    if _has_value(growth) or _has_value(lifecycle):
        lines.extend(["", "## Artifact growth and lifecycle", ""])
        if _has_value(growth):
            lines.extend(["### Growth", "", "```json", _pretty_json(growth), "```", ""])
        if _has_value(lifecycle):
            lines.extend(["### Lifecycle", "", "```json", _pretty_json(lifecycle), "```", ""])

    lines.extend(["", "## Per-task results", ""])
    lines.append(
        "| Task | Official won | Strict | Learning | Plan | Graph self-sufficient | Rescue | Tokens | LLM latency ms | Cost USD |"
    )
    lines.append("|---|:---:|:---:|:---:|---|:---:|:---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _markdown_cell(row.get("task_id", "")),
                _yes_no(row.get("benchmark_success")),
                _yes_no(row.get("strict_task_success")),
                _yes_no(row.get("learning_eligible")),
                _markdown_cell(row.get("plan_source", "")),
                _yes_no(row.get("graph_self_sufficient_success")),
                _yes_no(row.get("task_rescue_required")),
                row.get("total_tokens", 0),
                _display(row.get("llm_latency_ms", 0.0)),
                _display(row.get("cost_usd")),
            )
        )

    extraction_rows = [
        row for row in rows if _boolean(row.get("extraction_attempted"))
    ]
    if extraction_rows:
        lines.extend(["", "## Extraction diagnostics", ""])
        lines.append(
            "| Task | Stage | Prepared | Applied | Error | E1 proposed | E1 valid | E1 rejected | E1 coverage | E2 attempted | Existing edges | New edges |"
        )
        lines.append("|---|---|:---:|:---:|---|---:|---:|---:|:---:|:---:|---:|---:|")
        for row in extraction_rows:
            coverage = row.get("e1_contract_coverage_passed")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    _markdown_cell(row.get("task_id", "")),
                    _markdown_cell(row.get("extraction_stage", "")),
                    _yes_no(row.get("extraction_prepared")),
                    _yes_no(row.get("extraction_applied")),
                    _markdown_cell(row.get("extraction_error_code", "")),
                    row.get("e1_proposed", 0),
                    row.get("e1_validated", 0),
                    row.get("e1_rejected", 0),
                    "n/a" if coverage is None else _yes_no(coverage),
                    _yes_no(row.get("e2_attempted")),
                    row.get("e2_selected_existing_edges", 0),
                    row.get("e2_selected_new_edges", 0),
                )
            )

    failed = [
        _mapping(row.get("runtime_failure_diagnostic", {}))
        for row in rows
        if _has_value(row.get("runtime_failure_diagnostic"))
    ]
    if failed:
        lines.extend(["", "## Failed task diagnostics", ""])
        lines.append(
            "| Task | Plan | Exhaustion | Occurrence | Atomic | Last failure | Prep turns/actions | Seeded turns/actions | Unresolved roles | Progress snapshots/actions since change |"
        )
        lines.append("|---|---|---|---|---|---|---:|---:|---|---:|")
        for item in failed:
            preparation = _mapping(item.get("preparation", {}))
            seeded = _mapping(item.get("seeded", {}))
            binding = _mapping(item.get("binding", {}))
            progress = _mapping(item.get("progress", {}))
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {}/{} | {}/{} | {} | {}/{} |".format(
                    _markdown_cell(item.get("task_id", "")),
                    _markdown_cell(item.get("plan_source", "")),
                    _markdown_cell(item.get("exhaustion_scope", "none")),
                    _markdown_cell(item.get("occurrence_id", "")),
                    _markdown_cell(item.get("atomic_ref", "")),
                    _markdown_cell(item.get("last_runtime_failure_code", "")),
                    preparation.get("turn_count", 0),
                    preparation.get("environment_action_count", 0),
                    seeded.get("turn_count", 0),
                    seeded.get("environment_action_count", 0),
                    _markdown_cell(",".join(map(str, _sequence(binding.get("unresolved_roles", []))))),
                    progress.get("progress_snapshot_count", 0),
                    progress.get("actions_since_last_progress_change", 0),
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str = "AtomicSkillGraph v3 Experiment Report",
) -> Path:
    target = Path(path)
    _atomic_write(target, render_markdown(summary, rows, title=title).encode("utf-8"))
    return target


def write_reports(
    traces: Iterable[Mapping[str, Any] | Any],
    output_dir: str | Path,
    *,
    stem: str = "v3_results",
    title: str = "AtomicSkillGraph v3 Experiment Report",
    auxiliary_usage_traces: Iterable[Mapping[str, Any] | Any] = (),
    run_artifact_growth: Mapping[str, Any] | None = None,
    run_artifact_lifecycle: Mapping[str, Any] | None = None,
) -> ReportPaths:
    """Emit task rows plus resource summaries from auxiliary immutable traces."""

    if not stem or Path(stem).name != stem:
        raise ValueError("report stem must be a non-empty filename stem")
    rows = [trace_to_row(trace) for trace in traces]
    summary = summarize_traces(
        rows, auxiliary_usage_traces=auxiliary_usage_traces,
        run_artifact_growth=run_artifact_growth,
        run_artifact_lifecycle=run_artifact_lifecycle,
    )
    root = Path(output_dir)
    paths = ReportPaths(
        jsonl=root / f"{stem}.jsonl",
        csv=root / f"{stem}.csv",
        markdown=root / f"{stem}.md",
    )
    write_jsonl(rows, paths.jsonl)
    write_csv(rows, paths.csv)
    write_markdown(summary, rows, paths.markdown, title=title)
    return paths


def validate_formal_usage(traces: Iterable[Mapping[str, Any] | Any]) -> dict[str, Any]:
    """Fail closed unless every formal provider call is fully and exactly attributed."""
    items = list(traces)
    rows: list[dict[str, Any]] = []
    for trace in items:
        if _field(trace, "resource_usage_complete", True) is not True:
            raise ValueError(
                f"trace {_field(trace, 'trace_id', '<unknown>')} has incomplete provider usage"
            )
        for request in _sequence(_field(trace, "provider_requests", [])):
            if str(_field(request, "usage_status", "unavailable")) != "reported":
                raise ValueError(
                    f"trace {_field(trace, 'trace_id', '<unknown>')} has an unaudited provider request"
                )
        events = [_mapping(item) for item in _sequence(_field(trace, "llm_usage", []))]
        turns = _sequence(_field(trace, "agent_turns", []))
        if not events and turns:
            raise ValueError(f"trace {_field(trace, 'trace_id', '<unknown>')} has no LLM usage")
        for event in events:
            metadata = _mapping(event.get("provider_metadata", {}))
            if metadata.get("usage_status") != "reported":
                raise ValueError(
                    f"trace {_field(trace, 'trace_id', '<unknown>')} has unavailable/partial usage"
                )
            if str(event.get("bucket", "")) == "unattributed":
                raise ValueError(
                    f"trace {_field(trace, 'trace_id', '<unknown>')} has unattributed usage"
                )
        row = trace_to_row(trace)
        if int(row.get("token_mismatch", 0)) != 0:
            raise ValueError(
                f"trace {_field(trace, 'trace_id', '<unknown>')} has non-zero token_mismatch"
            )
        if int(row.get("unattributed_total_tokens", 0)) != 0:
            raise ValueError(
                f"trace {_field(trace, 'trace_id', '<unknown>')} has unattributed tokens"
            )
        rows.append(row)
    summary = summarize_traces(rows)
    if int(summary.get("token_mismatch", 0)) != 0:
        raise ValueError("formal report has non-zero token_mismatch")
    if int(summary.get("unattributed_total_tokens", 0)) != 0:
        raise ValueError("formal report has unattributed tokens")
    return summary


def validate_usage_event_persistence(
    usage_events: Iterable[Mapping[str, Any] | Any],
    traces: Iterable[Mapping[str, Any] | Any],
) -> dict[str, int]:
    """Prove every provider call in this process occurs in one immutable Trace."""

    trace_items = list(traces)
    validate_formal_usage(trace_items)
    persisted_counts: dict[str, int] = {}
    for trace in trace_items:
        for raw in _sequence(_field(trace, "llm_usage", [])):
            event_id = str(_field(raw, "event_id", ""))
            if not event_id:
                raise ValueError(
                    f"trace {_field(trace, 'trace_id', '<unknown>')} has usage without event_id"
                )
            persisted_counts[event_id] = persisted_counts.get(event_id, 0) + 1
    duplicates = sorted(
        event_id for event_id, count in persisted_counts.items() if count != 1
    )
    if duplicates:
        raise ValueError(
            "usage event_id is persisted in multiple Traces: " + ", ".join(duplicates[:10])
        )

    current_ids = [str(_field(event, "event_id", "")) for event in usage_events]
    if any(not event_id for event_id in current_ids):
        raise ValueError("in-process UsageLedger contains an event without event_id")
    if len(current_ids) != len(set(current_ids)):
        raise ValueError("in-process UsageLedger contains duplicate event_id values")
    missing = sorted(set(current_ids) - set(persisted_counts))
    if missing:
        raise ValueError(
            "in-process usage events are absent from task/maintenance Traces: "
            + ", ".join(missing[:10])
        )
    return {
        "in_process_event_count": len(current_ids),
        "persisted_event_count": len(persisted_counts),
        "trace_count": len(trace_items),
    }


def validate_frozen_v31_guards(
    traces: Iterable[Mapping[str, Any] | Any],
) -> dict[str, int]:
    """Reject failure-side reads, provisional selection, or cold token use."""

    rows = [trace_to_row(trace) for trace in traces]
    failure_side_reads = sum(
        _nonnegative_integer(row.get("failure_side_read_count", 0))
        for row in rows
    )
    provisional_selected = sum(
        _nonnegative_integer(row.get("provisional_selected_count", 0))
        for row in rows
    )
    cold_start_calls = 0
    for row in rows:
        by_bucket = _mapping(row.get("usage_by_bucket", {}))
        cold_start_calls += sum(
            _nonnegative_integer(
                _mapping(by_bucket.get(bucket, {})).get("call_count", 0)
            )
            for bucket in _FROZEN_FORBIDDEN_COLD_START_BUCKETS
        )
    if failure_side_reads:
        raise ValueError(
            "frozen v3.1 report has non-zero failure_side_read_count"
        )
    if provisional_selected:
        raise ValueError(
            "frozen v3.1 report has non-zero provisional_selected_count"
        )
    if cold_start_calls:
        raise ValueError("frozen v3.1 report contains cold-start token buckets")
    return {
        "failure_side_read_count": 0,
        "provisional_selected_count": 0,
        "cold_start_provider_call_count": 0,
    }


generate_reports = write_reports


def build_report_rows(
    traces: Iterable[Mapping[str, Any] | Any],
) -> list[dict[str, Any]]:
    return [trace_to_row(trace) for trace in traces]


def _usage_report(trace: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    # A runner-provided UsageLedger snapshot spans Planner/Runtime/Extractor
    # and is authoritative over TraceRecord.llm_usage, which older runtime
    # paths populated with Runtime turns only.
    source = _first_present(
        metadata,
        "usage_snapshot",
        "usage_ledger",
        "llm_usage",
        default=[],
    )
    if not _has_value(source):
        source = _field(trace, "llm_usage", [])
    events, provided_by_bucket, reconciliation = _normalize_usage_source(source)
    if not events and not provided_by_bucket:
        # Agent turns are structured provenance too, and are a safe fallback
        # when an older trace omitted its redundant llm_usage list.
        events = [_mapping(item.get("usage", {})) | {
            "session_id": item.get("session_id", ""),
            "turn_index": item.get("turn_index", 0),
            "provider_metadata": item.get("provider_metadata", {}),
        } for item in (
            _mapping(turn) for turn in _sequence(_field(trace, "agent_turns", []))
        )]

    session_buckets = _session_bucket_map(trace)
    normalized_events: list[dict[str, Any]] = []
    for raw in events:
        event = _mapping(raw)
        bucket = str(event.get("bucket") or event.get("usage_bucket") or "")
        provider_metadata = _mapping(event.get("provider_metadata", {}))
        bucket = bucket or str(provider_metadata.get("usage_bucket", ""))
        if bucket not in USAGE_BUCKETS:
            bucket = session_buckets.get(str(event.get("session_id", "")), "unattributed")
        normalized = _normalize_usage(event)
        normalized["bucket"] = bucket
        normalized["provider_metadata"] = provider_metadata
        for cost_key in ("cost_usd", "cost"):
            if cost_key in event:
                normalized[cost_key] = event[cost_key]
        normalized_events.append(normalized)

    by_bucket: dict[str, dict[str, int | float | None]] = {
        bucket: _zero_usage() for bucket in USAGE_BUCKETS
    }
    if normalized_events:
        for bucket in USAGE_BUCKETS:
            by_bucket[bucket] = _sum_usage(
                item for item in normalized_events if item["bucket"] == bucket
            )
    else:
        for bucket, raw in provided_by_bucket.items():
            normalized = _normalize_usage(_mapping(raw))
            target = bucket if bucket in by_bucket else "unattributed"
            by_bucket[target] = _sum_usage((by_bucket[target], normalized))

    episode = _sum_usage(by_bucket.values())
    real_total = sum(
        int(by_bucket[bucket]["total_tokens"])
        for bucket in USAGE_BUCKETS
        if bucket != "unattributed"
    )
    unattributed = int(by_bucket["unattributed"]["total_tokens"])
    expected = _episode_total_tokens(metadata, source, reconciliation, episode)
    return {
        "events": normalized_events,
        "by_bucket": by_bucket,
        "episode_total": episode,
        "reconciliation": {
            "episode_total_tokens": expected,
            "real_bucket_total_tokens": real_total,
            "unattributed_total_tokens": unattributed,
            "token_mismatch": int(episode["total_tokens"]) - expected,
        },
    }


def _normalize_usage_source(
    source: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if isinstance(source, Mapping):
        events = [_mapping(item) for item in _sequence(source.get("events", []))]
        by_bucket = _mapping(source.get("by_bucket", {}))
        reconciliation = _mapping(source.get("reconciliation", {}))
        if not events and not by_bucket and any(key in source for key in _USAGE_FIELDS):
            events = [_mapping(source)]
        return events, by_bucket, reconciliation
    events: list[dict[str, Any]] = []
    by_bucket: dict[str, Any] = {}
    reconciliation: dict[str, Any] = {}
    for item in _sequence(source):
        candidate = _mapping(item)
        if "events" in candidate or "by_bucket" in candidate:
            nested, nested_buckets, nested_reconciliation = _normalize_usage_source(candidate)
            events.extend(nested)
            by_bucket.update(nested_buckets)
            reconciliation.update(nested_reconciliation)
        else:
            events.append(candidate)
    return events, by_bucket, reconciliation


def _session_bucket_map(trace: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    type_map = {
        "RuntimePreparationSession": "runtime_preparation",
        "SeededSession": "runtime_seeded",
        "DynamicTaskSession": "runtime_dynamic",
        "ColdStartPlannerSession": "cold_start_c1",
        "ColdStartSession": "cold_start_c1",
        "ProvisionalSeededSession": "runtime_provisional_seeded",
        "DynamicColdStartContinuationSession": (
            "runtime_dynamic_cold_start_continuation"
        ),
        "ColdStartDynamicContinuationSession": (
            "runtime_dynamic_cold_start_continuation"
        ),
        "FailureExtractorSession": "failure_extractor_f1",
    }
    for raw in _sequence(_field(trace, "agent_sessions", [])):
        session = _mapping(raw)
        snapshot = _mapping(session.get("snapshot", {}))
        declared = str(
            snapshot.get("usage_bucket")
            or session.get("usage_bucket")
            or ""
        )
        bucket = (
            declared
            if declared in USAGE_BUCKETS
            else type_map.get(str(session.get("session_type", "")))
        )
        if bucket:
            mapping[str(session.get("session_id", ""))] = bucket
    return mapping


def _episode_total_tokens(
    metadata: Mapping[str, Any],
    source: Any,
    reconciliation: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> int:
    candidates = [
        reconciliation.get("episode_total_tokens"),
        metadata.get("episode_total_tokens"),
        _mapping(metadata.get("usage_reconciliation", {})).get("episode_total_tokens"),
    ]
    if isinstance(source, Mapping):
        candidates.append(_mapping(source.get("episode_total", {})).get("total_tokens"))
    for value in candidates:
        if value is not None:
            return _integer(value)
    return int(episode["total_tokens"])


def _normalize_usage(value: Mapping[str, Any]) -> dict[str, int | float | None]:
    reasoning = value.get("reasoning_tokens", 0)
    return {
        "prompt_tokens": _nonnegative_integer(value.get("prompt_tokens", 0)),
        "completion_tokens": _nonnegative_integer(value.get("completion_tokens", 0)),
        "total_tokens": _nonnegative_integer(value.get("total_tokens", 0)),
        "reasoning_tokens": (
            None if reasoning is None else _nonnegative_integer(reasoning)
        ),
        "call_count": _nonnegative_integer(value.get("call_count", 0)),
        "latency_ms": round(_nonnegative_number(value.get("latency_ms", 0.0)), 3),
    }


def _sum_usage(values: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    items = [_normalize_usage(_mapping(item)) for item in values]
    return {
        "prompt_tokens": sum(int(item["prompt_tokens"]) for item in items),
        "completion_tokens": sum(int(item["completion_tokens"]) for item in items),
        "total_tokens": sum(int(item["total_tokens"]) for item in items),
        "reasoning_tokens": (
            sum(int(item["reasoning_tokens"] or 0) for item in items)
            if all(item["reasoning_tokens"] is not None for item in items)
            else None
        ),
        "call_count": sum(int(item["call_count"]) for item in items),
        "latency_ms": round(sum(float(item["latency_ms"]) for item in items), 3),
    }


def _zero_usage() -> dict[str, int | float | None]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "call_count": 0,
        "latency_ms": 0.0,
    }


def _planned_node_count(plan: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> int:
    control = _sequence(plan.get("control_sequence", []))
    if control:
        return len(control)
    occurrences = _sequence(plan.get("occurrences", []))
    return len(occurrences) if occurrences else len(nodes)


def _preflight_passed(invocation: Mapping[str, Any]) -> bool:
    preflight = _mapping(invocation.get("preflight", {}))
    result = _mapping(invocation.get("result", {}))
    if "passed" in preflight:
        return _boolean(preflight["passed"])
    if "preflight_passed" in result:
        return _boolean(result["preflight_passed"])
    # No preflight information is not evidence of a rejection.
    return True


def _planner_atomic_full_coverage(planner: Mapping[str, Any]) -> bool:
    rows = _sequence(
        planner.get("atomic_search_p1r")
        or planner.get("atomic_search_p1")
        or []
    )
    required = [
        _mapping(item)
        for item in rows
        if _mapping(_mapping(item).get("requirement", {})).get(
            "required", True,
        )
    ]
    return bool(required) and all(
        _boolean(item.get("covered", False)) for item in required
    )


def _planner_p1r_reasons(planner: Mapping[str, Any]) -> dict[str, int]:
    if not (
        _has_value(planner.get("requirements_p1r"))
        or _has_value(planner.get("atomic_search_p1r"))
    ):
        return {}
    counts: dict[str, int] = {}
    rows = _sequence(planner.get("atomic_search_p1") or [])
    for raw in rows:
        row = _mapping(raw)
        if _boolean(row.get("covered", False)):
            continue
        reasons: list[str] = []
        for rejection in _sequence(row.get("rejection_reasons", [])):
            value = _mapping(rejection)
            for key in ("code", "reason"):
                if value.get(key):
                    reasons.append(str(value[key]))
            reasons.extend(
                str(item)
                for item in _sequence(value.get("reasons", []))
                if item
            )
        if not reasons:
            reasons = ["coverage_incomplete"]
        for reason in sorted(set(reasons)):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _p0_rejection_stages(planner: Mapping[str, Any]) -> dict[str, int]:
    """Count deterministic P0 rejections at their actual decision boundary."""

    rows = _sequence(planner.get("composite_rejections", []))
    if not rows:
        rows = _sequence(
            planner.get("p0_exact_contract_rejections")
            or planner.get("exact_contract_rejections")
            or []
        )
    counts: dict[str, int] = {}
    for raw in rows:
        item = _mapping(raw)
        stage = str(item.get("stage", ""))
        if not stage:
            reasons = {str(value) for value in _sequence(item.get("reasons", []))}
            if reasons & {"canonical_sequence_incomplete", "canonical_occurrence_ids_not_unique", "canonical_edges_invalid", "unvalidated_temporary_edge"}:
                stage = "retrieval_structure"
            elif reasons & {"candidate_bootstrap_not_top1", "candidate_exploration_quota"}:
                stage = "lifecycle_policy"
            elif "goal_contract_exact_mismatch" in reasons:
                stage = "retrieval_contract"
            else:
                stage = "legacy_unstaged"
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


_ATOMIC_MISMATCH_CODES = frozenset({
    "atomic_effect_predicate_missing",
    "atomic_effect_argument_role_missing",
    "atomic_effect_cardinality_insufficient",
    "atomic_required_input_type_unavailable",
})


def _atomic_contract_mismatch_reasons(
    planner: Mapping[str, Any], search_key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_result in _sequence(planner.get(search_key, [])):
        result = _mapping(raw_result)
        for raw_rejection in _sequence(result.get("rejection_reasons", [])):
            rejection = _mapping(raw_rejection)
            compatibility = _mapping(
                rejection.get("compatibility")
                or rejection.get("contract_diagnosis")
                or {}
            )
            codes = {
                str(value)
                for source in (rejection, compatibility)
                for value in _sequence(source.get("failure_codes", []))
            }
            codes.update(
                str(value)
                for value in _sequence(rejection.get("reasons", []))
                if str(value) in _ATOMIC_MISMATCH_CODES
            )
            for code in sorted(codes & _ATOMIC_MISMATCH_CODES):
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _failure_extractor_diagnostic(
    trace: Any,
    *,
    metadata: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> dict[str, int]:
    """Project R2.2 diagnostics without creating a second usage authority.

    Input-size and allocation facts are emitted by the bounded view/session
    producer.  Provider calls and tokens always come from the normalized
    provider-reported usage buckets.  Exhaustion is counted only from the
    formal failure-extraction rejection, never from a numeric cap comparison.
    """

    metrics = _mapping(metadata.get("failure_extractor_metrics", {}))
    failure_extraction = _mapping(_field(trace, "failure_extraction", {}))
    rejection = _mapping(failure_extraction.get("rejection", {}))

    def metric(name: str) -> int:
        if name in metrics:
            return _nonnegative_integer(metrics[name])
        return _nonnegative_integer(metadata.get(name, 0))

    by_bucket = _mapping(usage.get("by_bucket", {}))
    f1_usage = _mapping(by_bucket.get("failure_extractor_f1", {}))
    f2_usage = _mapping(by_bucket.get("failure_extractor_f2", {}))
    f1_calls = _nonnegative_integer(f1_usage.get("call_count", 0))
    f2_calls = _nonnegative_integer(f2_usage.get("call_count", 0))
    f1_tokens = _nonnegative_integer(f1_usage.get("total_tokens", 0))
    f2_tokens = _nonnegative_integer(f2_usage.get("total_tokens", 0))

    rejection_code = str(rejection.get("code", ""))
    rejection_stage = str(rejection.get("stage", ""))
    budget_exhausted = rejection_code == "failure_extractor_budget_exhausted"
    skipped_after_budget = bool(
        budget_exhausted
        and rejection_stage == "f2_not_started_no_remaining_budget"
    )
    if rejection_stage.startswith("f1"):
        rejected_usage_persisted = f1_calls > 0
    elif rejection_stage == "f2_not_started_no_remaining_budget":
        rejected_usage_persisted = f1_calls > 0
    elif rejection_stage.startswith("f2"):
        rejected_usage_persisted = f2_calls > 0
    else:
        rejected_usage_persisted = f1_calls + f2_calls > 0

    return {
        "failure_extractor_f1_input_event_count": metric(
            "failure_extractor_f1_input_event_count"
        ),
        "failure_extractor_f1_prompt_chars": metric(
            "failure_extractor_f1_prompt_chars"
        ),
        "failure_extractor_f1_prompt_bytes": metric(
            "failure_extractor_f1_prompt_bytes"
        ),
        "failure_extractor_f1_tokens": f1_tokens,
        "failure_extractor_f2_span_count": metric(
            "failure_extractor_f2_span_count"
        ),
        "failure_extractor_f2_source_event_count": metric(
            "failure_extractor_f2_source_event_count"
        ),
        "failure_extractor_f2_prompt_chars": metric(
            "failure_extractor_f2_prompt_chars"
        ),
        "failure_extractor_f2_prompt_bytes": metric(
            "failure_extractor_f2_prompt_bytes"
        ),
        "failure_extractor_f2_tokens": f2_tokens,
        "failure_extractor_budget_exhausted_count": int(budget_exhausted),
        "failure_extractor_skipped_after_budget_count": int(
            skipped_after_budget
        ),
        "failure_extractor_f1_provider_call_count": f1_calls,
        "failure_extractor_f2_provider_call_count": f2_calls,
        "failure_extractor_usage_persisted_after_rejection_count": int(
            budget_exhausted and rejected_usage_persisted
        ),
        "failure_extractor_remaining_budget_before_f2": metric(
            "failure_extractor_remaining_budget_before_f2"
        ),
    }


def _extraction_diagnostic(
    trace: Any,
    metadata: Mapping[str, Any],
    quality: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    extraction = _mapping(metadata.get("extraction", {}))
    applied_payload = _mapping(metadata.get("evolution_applied", {}))
    policy = _mapping(_field(trace, "extraction_policy", {}))
    e2_usage = _mapping(
        _mapping(usage.get("by_bucket", {})).get("extractor_e2", {})
    )
    attempted = _boolean(
        extraction.get("attempted", policy.get("should_extract", False))
    )
    prepared = _boolean(extraction.get("prepared", False))
    applied = _boolean(extraction.get("applied", bool(applied_payload)))
    stage = str(extraction.get("stage", ""))
    if not stage:
        stage = "applied" if applied else "prepared" if prepared else ""
    proposed = _integer(extraction.get(
        "e1_proposed",
        quality.get("extractor_e1_proposal_count", 0),
    ))
    validated = _integer(extraction.get(
        "e1_validated",
        quality.get("extractor_e1_validated_occurrence_count", 0),
    ))
    rejected = _integer(extraction.get(
        "e1_rejected",
        quality.get("extractor_e1_rejection_count", 0),
    ))
    coverage_value = extraction.get("e1_contract_coverage_passed")
    coverage = None if coverage_value is None else _boolean(coverage_value)
    e2_attempted = _boolean(extraction.get(
        "e2_attempted",
        _integer(e2_usage.get("call_count", 0)) > 0,
    ))
    return {
        "extraction_attempted": attempted,
        "extraction_stage": stage,
        "extraction_prepared": prepared,
        "extraction_applied": applied,
        "extraction_error_code": str(extraction.get("error_code", "")),
        "e1_proposed": proposed,
        "e1_validated": validated,
        "e1_rejected": rejected,
        "e1_contract_coverage_passed": coverage,
        "e2_attempted": e2_attempted,
        "e2_selected_existing_edges": _integer(
            extraction.get("e2_selected_existing_edges", 0)
        ),
        "e2_selected_new_edges": _integer(
            extraction.get("e2_selected_new_edges", 0)
        ),
    }


def _runtime_failure_diagnostic(
    trace: Any,
    *,
    plan: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    invocations: Sequence[Mapping[str, Any]],
    failure_codes: Sequence[str],
    strict_task_success: bool,
) -> dict[str, Any]:
    """Derive attribution from existing Trace facts; never change Runtime state."""

    # Recoverable preflight/runtime failures remain useful Trace evidence even
    # when the task ultimately succeeds.  They must not turn a successful task
    # into a row in the report's failed-task diagnostic section.
    if strict_task_success:
        return {}
    trace_failures = [
        _mapping(item) for item in _sequence(_field(trace, "failures", []))
    ]
    last_trace_failure = next(
        (
            item for item in reversed(trace_failures)
            if str(item.get("code") or item.get("failure_code") or "")
        ),
        {},
    )
    failed_statuses = _FAILED_NODE_STATUSES | {"not_started"}
    node = next(
        (
            item for item in reversed(nodes)
            if _enum_value(item.get("status", "")) in failed_statuses
            or _has_value(item.get("failure"))
        ),
        {},
    )
    failure_occurrence_id = str(last_trace_failure.get("occurrence_id", ""))
    if not node and failure_occurrence_id:
        node = next(
            (
                item for item in reversed(nodes)
                if str(item.get("occurrence_id", "")) == failure_occurrence_id
            ),
            {},
        )
    occurrence_id = str(
        node.get("occurrence_id", "") or failure_occurrence_id
    )
    atomic_ref = str(node.get("atomic_ref", ""))
    if not atomic_ref:
        atomic_ref = next(
            (
                str(value)
                for value in reversed(
                    _sequence(last_trace_failure.get("artifact_refs", []))
                )
                if str(value).startswith("skill://atomic")
            ),
            "",
        )
    node_invocations = [
        item for item in invocations
        if not occurrence_id or str(item.get("occurrence_id", "")) == occurrence_id
    ]
    direct_result = _mapping(node.get("direct_result", {}))
    seeded_result = _mapping(node.get("seeded_result", {}))
    direct_codes: set[str] = set()
    for value in [direct_result, *node_invocations]:
        _collect_codes(value, direct_codes)

    sessions = [
        _mapping(item) for item in _sequence(_field(trace, "agent_sessions", []))
    ]
    turns = [
        _mapping(item) for item in _sequence(_field(trace, "agent_turns", []))
    ]
    spans = [
        _mapping(item) for item in _sequence(_field(trace, "runtime_spans", []))
    ]

    def session_diagnostic(label: str) -> dict[str, Any]:
        selected = [
            item for item in sessions
            if label in str(item.get("session_type", "")).casefold()
            and (not occurrence_id or str(item.get("occurrence_id", "")) == occurrence_id)
        ]
        ids = {str(item.get("session_id", "")) for item in selected}
        action_count = sum(
            max(0, _integer(span.get("action_end", 0)) - _integer(span.get("action_start", 0)))
            for span in spans
            if label in str(span.get("kind", "")).casefold()
            and (not occurrence_id or str(span.get("occurrence_id", "")) == occurrence_id)
        )
        return {
            "entered": bool(selected),
            "turn_count": sum(str(item.get("session_id", "")) in ids for item in turns),
            "environment_action_count": action_count,
        }

    binding_changes = [
        _mapping(item) for item in _sequence(_field(trace, "binding_changes", []))
        if not occurrence_id or str(_mapping(item).get("occurrence_id", "")) == occurrence_id
    ]
    latest_binding_by_role: dict[str, dict[str, Any]] = {}
    for item in binding_changes:
        role = str(item.get("role", ""))
        if role:
            latest_binding_by_role[role] = _mapping(item.get("current", {}))
    resolved_roles = sorted(
        role
        for role, current in latest_binding_by_role.items()
        if str(current.get("status", "")).casefold() == "grounded"
        and current.get("value") not in (None, "")
    )
    explicit_unresolved_roles = {
        str(role)
        for source in (direct_result, seeded_result, _mapping(node.get("failure", {})))
        for role in _sequence(source.get("unresolved_roles", []))
        if str(role)
    }
    unresolved_roles = sorted(
        explicit_unresolved_roles
        | {
            role
            for role, current in latest_binding_by_role.items()
            if str(current.get("status", "")).casefold() != "grounded"
            or current.get("value") in (None, "")
        }
    )

    progress = [
        _mapping(item) for item in _sequence(_field(trace, "task_progress_records", []))
    ]
    last_digest = ""
    last_progress_world_revision = -1
    prior_digest: str | None = None
    for item in progress:
        snapshot = _mapping(item.get("snapshot", {}))
        digest = str(snapshot.get("progress_digest", ""))
        if digest != prior_digest:
            # The record revision is the progress ledger's own counter.  Only
            # snapshot.revision shares authority with EnvironmentActionRecord
            # new_revision.  Retain the outer value solely for legacy fixtures
            # that predate snapshot revision persistence.
            last_progress_world_revision = _integer(
                snapshot.get("revision", item.get("revision", -1))
            )
            prior_digest = digest
        last_digest = digest
    actions_since_change = sum(
        _boolean(_mapping(item).get("accepted", False))
        and _integer(_mapping(item).get("new_revision", -1))
        > last_progress_world_revision
        for item in _sequence(_field(trace, "environment_actions", []))
    )

    node_failure = _mapping(node.get("failure", {}))
    last_failure = str(
        last_trace_failure.get("code")
        or last_trace_failure.get("failure_code")
        or node_failure.get("code")
        or node_failure.get("failure_code")
        or (failure_codes[-1] if failure_codes else "")
    )
    exhaustion_scope = (
        "task" if "runtime_task_token_budget_exhausted" in failure_codes
        else "node" if "runtime_node_token_budget_exhausted" in failure_codes
        else "none"
    )
    preparation = session_diagnostic("preparation")
    seeded = session_diagnostic("seeded")
    return {
        "task_id": str(_mapping(_field(trace, "task", {})).get("task_id", "")),
        "plan_source": str(plan.get("source", "")),
        "failure_codes": list(failure_codes),
        "exhaustion_scope": exhaustion_scope,
        "occurrence_id": occurrence_id,
        "atomic_ref": atomic_ref,
        "direct": {
            "preflight_rejected_count": sum(
                not _preflight_passed(item) for item in node_invocations
            ),
            "started": any(
                _boolean(_mapping(item.get("result", {})).get("started", False))
                for item in node_invocations
            ),
            "atomic_effect_passed": _boolean(
                direct_result.get("atomic_effect_passed", False)
            ) or any(
                _boolean(_mapping(item.get("result", {})).get("atomic_effect_passed", False))
                for item in node_invocations
            ),
            "failure_codes": sorted(direct_codes),
        },
        "preparation": {
            "turn_count": preparation["turn_count"],
            "environment_action_count": preparation["environment_action_count"],
        },
        "seeded": seeded,
        "binding": {
            "resolved_roles": resolved_roles,
            "unresolved_roles": unresolved_roles,
        },
        "progress": {
            "progress_snapshot_count": len(progress),
            "last_progress_digest": last_digest,
            "actions_since_last_progress_change": actions_since_change,
        },
        "last_runtime_failure_code": last_failure,
    }


def _runtime_exhausted_session_counts(
    trace: Any,
    *,
    sessions: Sequence[Mapping[str, Any]],
    session_bucket_map: Mapping[str, str],
) -> dict[str, int]:
    """Count only sessions backed by an actual token-exhaustion result.

    A session reaching its numeric cap is not itself failure evidence: it may
    have completed successfully on that exact call.  New traces expose the
    route result that caught ``BudgetExhausted``.  Legacy FailureEnvelope-only
    traces are used only when occurrence/stage identity selects exactly one
    Runtime session; ambiguous evidence stays zero.
    """

    session_rows: list[tuple[str, str, str]] = []
    for session in sessions:
        session_id = str(session.get("session_id", ""))
        snapshot = _mapping(session.get("snapshot", {}))
        bucket = str(
            snapshot.get("usage_bucket")
            or session_bucket_map.get(session_id, "")
        )
        if bucket in _RUNTIME_USAGE_BUCKETS and session_id:
            session_rows.append((
                session_id,
                bucket,
                str(session.get("occurrence_id", "")),
            ))

    exhausted_ids: set[str] = set()

    def add_unique(bucket: str, occurrence_id: str | None = None) -> None:
        candidates = [
            session_id
            for session_id, candidate_bucket, candidate_occurrence in session_rows
            if candidate_bucket == bucket
            and (
                occurrence_id is None
                or candidate_occurrence == occurrence_id
            )
        ]
        if len(candidates) == 1:
            exhausted_ids.add(candidates[0])

    def code(value: Any) -> str:
        payload = _mapping(value)
        return str(payload.get("failure_code") or payload.get("code") or "")

    node_exhaustion = "runtime_node_token_budget_exhausted"
    for raw_node in _sequence(_field(trace, "node_records", [])):
        node = _mapping(raw_node)
        occurrence_id = str(node.get("occurrence_id", ""))
        if code(node.get("direct_result", {})) == node_exhaustion:
            add_unique("runtime_preparation", occurrence_id)
        if code(node.get("seeded_result", {})) == node_exhaustion:
            add_unique("runtime_seeded", occurrence_id)

    for raw_step in _sequence(_field(trace, "cold_start_steps", [])):
        step = _mapping(raw_step)
        if code(step) != "provisional_seeded_budget_exhausted":
            continue
        step_id = str(step.get("step_id", ""))
        if step_id:
            add_unique("runtime_provisional_seeded", f"cold::{step_id}")

    metadata = _mapping(_field(trace, "metadata", {}))
    task_exhaustion = "runtime_task_token_budget_exhausted"
    for key, default_bucket in (
        ("dynamic_result", "runtime_dynamic"),
        ("task_rescue", "runtime_dynamic"),
        (
            "cold_start_dynamic_continuation",
            "runtime_dynamic_cold_start_continuation",
        ),
    ):
        result = _mapping(metadata.get(key, {}))
        if code(result) != task_exhaustion:
            continue
        bucket = default_bucket
        if key == "dynamic_result" and _boolean(
            result.get("cold_start_continuation", False)
        ):
            bucket = "runtime_dynamic_cold_start_continuation"
        add_unique(bucket)

    # Legacy strict fallback: a formal FailureEnvelope is usable only when its
    # code and occurrence resolve to one and only one eligible session.
    for raw_failure in _sequence(_field(trace, "failures", [])):
        failure = _mapping(raw_failure)
        failure_code = code(failure)
        occurrence_id = str(failure.get("occurrence_id", ""))
        if failure_code == task_exhaustion:
            candidates = [
                session_id
                for session_id, bucket, _occurrence in session_rows
                if bucket in {
                    "runtime_dynamic",
                    "runtime_dynamic_cold_start_continuation",
                }
            ]
            if len(candidates) == 1:
                exhausted_ids.add(candidates[0])
        elif failure_code == node_exhaustion and occurrence_id:
            candidates = [
                session_id
                for session_id, bucket, candidate_occurrence in session_rows
                if bucket in {"runtime_preparation", "runtime_seeded"}
                and candidate_occurrence == occurrence_id
            ]
            if len(candidates) == 1:
                exhausted_ids.add(candidates[0])

    counts = {bucket: 0 for bucket in _RUNTIME_USAGE_BUCKETS}
    by_id = {session_id: bucket for session_id, bucket, _ in session_rows}
    for session_id in exhausted_ids:
        counts[by_id[session_id]] += 1
    return counts


def _r31_event_metrics(
    trace: Any,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, int]:
    """Aggregate only explicit R3 events; never infer Runtime decisions."""

    events = [
        _mapping(item)
        for item in _sequence(metadata.get("r3_events", []))
    ]
    result = {name: 0 for name in R31_RUNTIME_METRICS}
    result["replay_action_window_size"] = 5
    for event in events:
        event_type = str(event.get("event_type", ""))
        details = _mapping(event.get("details", {}))
        if event_type == "grounding_refresh":
            result["runtime_grounding_refresh_count"] += 1
        elif event_type == "unique_binding_auto_confirm":
            result["runtime_unique_binding_auto_confirm_count"] += 1
            result["runtime_unique_binding_auto_confirm_role_count"] += _integer(
                details.get("role_count", 0)
            )
        elif event_type == "invocation_ready_transition":
            result["runtime_invocation_ready_transition_count"] += 1
        elif event_type == "effect_ready_transition":
            result["runtime_effect_ready_transition_count"] += 1
        elif event_type == "replay_action_window_compaction":
            result["replay_action_window_compaction_count"] += 1
            result["replay_pruned_action_count"] += _integer(
                details.get("pruned_action_count", 0)
            )
        elif event_type == "runtime_context_projection":
            result["runtime_context_snapshot_count"] += _integer(
                details.get("current_state_snapshot_count", 0)
            )
            result["runtime_exploration_memory_projection_count"] += _integer(
                details.get("exploration_memory_count", 0)
            )
            result["runtime_recent_action_projection_count"] += _integer(
                details.get("recent_action_count", 0)
            )
        elif event_type == "partial_atomic_admission":
            result["partial_atomic_admission_count"] += _integer(
                details.get("admission_count", 0)
            )
            result["partial_atomic_alignment_reuse_count"] += _integer(
                details.get("alignment_reuse_count", 0)
            )
            result["partial_atomic_new_contract_count"] += _integer(
                details.get("new_contract_count", 0)
            )
            result["partial_atomic_tool_admission_count"] += _integer(
                details.get("tool_admission_count", 0)
            )
            result[
                "partial_atomic_implementation_admission_count"
            ] += _integer(details.get("implementation_admission_count", 0))
    return result


def _r21_runtime_metrics(
    trace: Any,
    *,
    usage: Mapping[str, Any],
    strict_task_success: bool,
    task_rescue_required: bool,
) -> dict[str, Any]:
    """Derive R2.1 diagnostics from formal Trace records, defaulting to zero.

    These values are observational only.  They never infer a Runtime decision
    from task type, object names, free text, or hidden validator state.
    """

    calls = [
        _mapping(item)
        for item in _sequence(_field(trace, "native_tool_calls", []))
    ]
    exploration = 0
    attempts = 0
    validations = 0
    validation_successes = 0
    selected_invocations = 0
    conflict_occurrences: set[str] = set()
    conflict_count = 0
    for call in calls:
        name = str(call.get("tool_name", ""))
        arguments = _mapping(call.get("arguments", {}))
        preflight = _mapping(call.get("preflight_result", {}))
        if name == "environment_action":
            intent = str(arguments.get("intent", preflight.get("intent", "")))
            exploration += int(intent == "explore")
            attempts += int(intent == "attempt_current_atomic")
        elif name == "validate_current_atomic":
            validations += 1
            validation_successes += int(_boolean(
                preflight.get(
                    "atomic_effect_passed",
                    preflight.get("passed", False),
                )
            ))
        elif name == "report_runtime_status" and str(
            arguments.get("status", "")
        ) == "plan_conflict":
            conflict_count += 1
            occurrence_id = str(call.get("occurrence_id", ""))
            if occurrence_id:
                conflict_occurrences.add(occurrence_id)
        elif (
            str(call.get("call_kind", "")).casefold()
            in {"implementation", "implementation_invocation", "learned_invocation"}
            or name.startswith("invoke_impl_")
        ):
            selected_invocations += 1

    sessions = [
        _mapping(item)
        for item in _sequence(_field(trace, "agent_sessions", []))
    ]
    replay_catalog_compactions = 0
    replay_initial_compacted_count = 0
    replay_last_full_catalogs = 0
    replay_history_actions = 0
    session_bucket_map = _session_bucket_map(trace)
    for session in sessions:
        snapshot = _mapping(session.get("snapshot", {}))
        replay_catalog_compactions += _integer(
            snapshot.get(
                "replay_catalog_compaction_count",
                snapshot.get("context_compaction_count", 0),
            )
        )
        replay_initial_compacted_count += int(_boolean(
            snapshot.get("replay_initial_catalog_compacted", False)
        ))
        replay_last_full_catalogs = max(
            replay_last_full_catalogs,
            _integer(snapshot.get("replay_full_catalog_count_at_last_request", 0)),
        )
        replay_history_actions += _integer(
            snapshot.get("replay_history_action_count", 0)
        )
    exhausted_by_bucket = _runtime_exhausted_session_counts(
        trace,
        sessions=sessions,
        session_bucket_map=session_bucket_map,
    )

    events = [
        _mapping(item) for item in _sequence(usage.get("events", []))
    ]
    decomposition: dict[str, dict[str, Any]] = {}
    for bucket in _RUNTIME_USAGE_BUCKETS:
        bucket_events = [item for item in events if item.get("bucket") == bucket]
        if bucket_events:
            totals = _sum_usage(bucket_events)
            calls_count = _integer(totals.get("call_count", 0))
            total_values = [_integer(item.get("total_tokens", 0)) for item in bucket_events]
            reasoning_tokens = sum(
                _integer(item.get("reasoning_tokens", 0))
                for item in bucket_events
            )
        else:
            totals = _normalize_usage(
                _mapping(_mapping(usage.get("by_bucket", {})).get(bucket, {}))
            )
            calls_count = _integer(totals.get("call_count", 0))
            total_values = []
            reasoning_tokens = _integer(totals.get("reasoning_tokens", 0))
        decomposition[bucket] = {
            "calls": calls_count,
            "prompt_tokens": _integer(totals.get("prompt_tokens", 0)),
            "completion_tokens": _integer(totals.get("completion_tokens", 0)),
            "total_tokens": _integer(totals.get("total_tokens", 0)),
            "reasoning_tokens": reasoning_tokens,
            "average_tokens_per_call": _ratio(
                _integer(totals.get("total_tokens", 0)), calls_count
            ) or 0.0,
            "max_tokens_per_call": max(total_values, default=0),
            "exhausted_session_count": exhausted_by_bucket[bucket],
        }

    runtime_prompt = sum(
        _integer(item["prompt_tokens"]) for item in decomposition.values()
    )
    runtime_completion = sum(
        _integer(item["completion_tokens"]) for item in decomposition.values()
    )
    runtime_reasoning = sum(
        _integer(item["reasoning_tokens"]) for item in decomposition.values()
    )
    return {
        "runtime_exploration_action_count": exploration,
        "runtime_atomic_attempt_action_count": attempts,
        "runtime_validate_current_atomic_count": validations,
        "runtime_validate_current_atomic_success_count": validation_successes,
        "learned_invocation_selected_count": selected_invocations,
        "runtime_plan_conflict_count": conflict_count,
        "runtime_plan_conflict_occurrence_count": len(conflict_occurrences),
        "runtime_plan_conflict_rescue_count": int(
            conflict_count > 0 and task_rescue_required
        ),
        "runtime_plan_conflict_rescue_success_count": int(
            conflict_count > 0 and task_rescue_required and strict_task_success
        ),
        "replay_catalog_compaction_count": replay_catalog_compactions,
        "replay_initial_catalog_compacted_count": replay_initial_compacted_count,
        "replay_full_catalog_count_at_last_request": replay_last_full_catalogs,
        "replay_history_action_count": replay_history_actions,
        "runtime_prompt_tokens": runtime_prompt,
        "runtime_completion_tokens": runtime_completion,
        "runtime_reasoning_tokens": runtime_reasoning,
        "runtime_reasoning_share": _ratio(
            runtime_reasoning, runtime_completion
        ) or 0.0,
        "runtime_token_decomposition": decomposition,
    }


def _aggregate_runtime_token_decomposition(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for bucket in _RUNTIME_USAGE_BUCKETS:
        values = [
            _mapping(_mapping(row.get("runtime_token_decomposition", {})).get(
                bucket, {}
            ))
            for row in rows
        ]
        calls = sum(_integer(item.get("calls", 0)) for item in values)
        total_tokens = sum(
            _integer(item.get("total_tokens", 0))
            for item in values
        )
        result[bucket] = {
            "calls": calls,
            "prompt_tokens": sum(
                _integer(item.get("prompt_tokens", 0)) for item in values
            ),
            "completion_tokens": sum(
                _integer(item.get("completion_tokens", 0)) for item in values
            ),
            "total_tokens": total_tokens,
            "reasoning_tokens": sum(
                _integer(item.get("reasoning_tokens", 0)) for item in values
            ),
            "average_tokens_per_call": _ratio(total_tokens, calls) or 0.0,
            "max_tokens_per_call": max(
                (_integer(item.get("max_tokens_per_call", 0)) for item in values),
                default=0,
            ),
            "exhausted_session_count": sum(
                _integer(item.get("exhausted_session_count", 0))
                for item in values
            ),
        }
    return result


def _confirmed_capability_gap(trace: Any, metadata: Mapping[str, Any]) -> bool:
    direct = _first_present(
        metadata,
        "confirmed_capability_gap",
        "capability_gap_confirmed",
        default=None,
    )
    if direct is not None:
        return _boolean(direct)
    policy = _mapping(_field(trace, "extraction_policy", {}))
    candidates = (
        policy.get("classification"),
        policy.get("gap_classification"),
        metadata.get("classification"),
        metadata.get("gap_classification"),
    )
    return any(value == "confirmed_capability_gap" for value in candidates)


def _v32_method_metrics(
    trace: Any,
    *,
    planner: Mapping[str, Any],
    metadata: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    invocations: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Derive v3.2 counters from structured Trace authority only."""

    explicit = _mapping(metadata.get("v32_metrics", {}))
    planner_repairability = _mapping(planner.get("repairability", {}))
    diagnostics = planner_repairability.get("diagnostics") or ()
    result: dict[str, int] = {
        name: _integer(explicit.get(name, 0))
        for name in V32_METHOD_METRICS
    }
    result.setdefault("planner_repairability_gate_count", int(bool(planner_repairability)))
    result.setdefault("planner_repairability_repairable_count", int(
        bool(planner_repairability)
        and bool(planner_repairability.get("repairable", False))
    ))
    result.setdefault("planner_hard_capability_gap_count", sum(
        1 for item in diagnostics if item.get("hard_capability_gap")
    ))
    result.setdefault("planner_p1r_skipped_hard_gap_count", int(
        bool(planner_repairability)
        and str(planner_repairability.get("reason_code", ""))
        == "planner_hard_capability_gap"
    ))
    result.setdefault("task_terminal_early_success_count", int(
        bool(_mapping(metadata.get("task_terminal", {})).get("during"))
    ))
    result.setdefault("task_terminal_during_tool_count", sum(
        1 for item in executions
        if _boolean(_mapping(item.get("result", {})).get("terminal_interrupted", False))
    ))
    result.setdefault("terminal_skipped_occurrence_count", sum(
        1 for item in nodes
        if str(item.get("status", "")) == "skipped_goal_terminal"
    ))
    result.setdefault("task_terminal_with_remaining_occurrences_count", int(
        result["terminal_skipped_occurrence_count"] > 0
    ))
    result.setdefault("tool_builder_proposal_count", int(
        result["tool_builder_call_count"] - result["tool_builder_no_tool_count"]
    ))
    result.setdefault("tool_validated_path_count", sum(
        len(_sequence(_mapping(item.get("result", {})).get("validated_paths", ())))
        for item in executions
    ))
    result.setdefault("tool_unvalidated_path_count", sum(
        len(_sequence(_mapping(item.get("result", {})).get("unvalidated_paths", ())))
        for item in executions
    ))
    result.setdefault("tool_observed_loop_iteration_count", sum(
        sum(_integer(value) for value in _mapping(
            _mapping(item.get("result", {})).get("loop_iteration_counts", {})
        ).values())
        for item in executions
    ))
    result.setdefault("tool_stop_condition_witness_count", sum(
        len(_sequence(_mapping(item.get("result", {})).get("stop_condition_witnesses", ())))
        for item in executions
    ))
    result.setdefault("runtime_tool_llm_bypassed_action_count", sum(
        _integer(_mapping(item.get("result", {})).get("executed_node_count", 0))
        for item in executions
        if _mapping(item.get("result", {})).get("path_id")
        or _mapping(item.get("result", {})).get("program_node_id")
    ))
    return result


def _v31_method_metrics(
    trace: Any,
    *,
    planner: Mapping[str, Any],
    plan: Mapping[str, Any],
    metadata: Mapping[str, Any],
    usage: Mapping[str, Any],
    failure_codes: Sequence[str],
    planner_atomic_full_coverage: bool,
    planner_p2_used: bool,
) -> dict[str, int]:
    """Derive frozen v3.1 counters only from structured Trace authority.

    Runtime components may publish an exact counter either directly in Trace
    metadata or in one of the scoped metric mappings below.  Those explicit
    values take precedence.  The fallbacks cover facts represented directly by
    the v3.1 Trace fields and never parse observations or free-form messages.
    """

    explicit_sources = (
        metadata,
        _mapping(metadata.get("v31_metrics", {})),
        _mapping(metadata.get("method_metrics", {})),
        _mapping(metadata.get("cold_start_metrics", {})),
        _mapping(metadata.get("failure_extractor_metrics", {})),
        _mapping(metadata.get("failure_knowledge_metrics", {})),
    )

    def value(name: str, derived: int | bool = 0) -> int:
        for source in explicit_sources:
            if name in source:
                return _nonnegative_integer(source[name])
        return _nonnegative_integer(derived)

    requirement_bundle = _mapping(_field(trace, "requirement_bundle", {}))
    if not requirement_bundle:
        requirement_bundle = _mapping(
            planner.get("requirements_p1r")
            or planner.get("requirements_p1")
            or {}
        )
    requirement_expansion = _mapping(
        _field(trace, "requirement_expansion", {})
    )
    if not requirement_expansion:
        requirement_expansion = _mapping(
            planner.get("requirement_expansion", {})
        )
    repeat_blocks = _sequence(
        requirement_bundle.get("repeat_blocks")
        or requirement_expansion.get("repeat_blocks")
    )
    instances = [
        _mapping(item)
        for item in _sequence(requirement_expansion.get("instances", []))
    ]
    repeat_step_ids = {
        str(step_id)
        for raw in _sequence(plan.get("repeat_constraints", []))
        for iteration in _sequence(
            _mapping(raw).get("iteration_steps", [])
        )
        for step_id in _sequence(iteration)
        if str(step_id)
    }
    runtime_occurrences = [
        _mapping(item)
        for item in _sequence(plan.get("occurrences", []))
    ]
    repeated_occurrences = sum(
        str(occurrence.get("step_id", "")) in repeat_step_ids
        for occurrence in runtime_occurrences
    )

    cold_plan = _mapping(_field(trace, "cold_start_plan", None))
    cold_proposal = _mapping(cold_plan.get("proposal", {}))
    cold_validation = _mapping(cold_plan.get("validation", {}))
    if not cold_proposal:
        cold_proposal = _mapping(
            planner.get("cold_start_repair")
            or planner.get("cold_start_plan")
            or {}
        )
    if not cold_validation:
        cold_validation = _mapping(
            planner.get("cold_start_repair_validation")
            or planner.get("cold_start_validation")
            or {}
        )
    cold_steps = [
        _mapping(item)
        for item in _sequence(_field(trace, "cold_start_steps", []))
    ]

    def candidate_source(step: Mapping[str, Any]) -> str:
        return str(step.get("candidate_source", "")).casefold()

    provisional_steps = [
        step for step in cold_steps
        if (
            "provisional" in candidate_source(step)
            or str(step.get("execution_mode", "")).casefold()
            == "provisional_seeded"
        )
    ]
    verified_steps = [
        step for step in cold_steps
        if "verified" in candidate_source(step)
        and "provisional" not in candidate_source(step)
    ]
    planned_steps = [
        _mapping(item) for item in _sequence(cold_proposal.get("steps", []))
    ]
    planned_provisional_steps = [
        step for step in planned_steps
        if "provisional" in candidate_source(step)
    ]

    failure_extraction = _mapping(
        _field(trace, "failure_extraction", None)
    )
    provisional_refs = {
        str(item)
        for item in _sequence(failure_extraction.get("provisional_refs", []))
        if str(item)
    }
    failure_experience_ids = {
        str(item)
        for item in _sequence(
            failure_extraction.get("failure_experience_ids", [])
        )
        if str(item)
    }
    promotions = [
        _mapping(item)
        for item in _sequence(_field(trace, "provisional_promotions", []))
    ]

    def lifecycle_status(item: Mapping[str, Any]) -> str:
        return str(
            item.get("status")
            or item.get("outcome")
            or item.get("new_status")
            or ""
        ).casefold()

    trial_ready = sum(lifecycle_status(item) == "trial_ready" for item in promotions)
    trial_supported = sum(
        lifecycle_status(item) == "trial_supported" for item in promotions
    )
    promoted = sum(
        lifecycle_status(item) == "promoted"
        or bool(_sequence(item.get("promoted_verified_refs", [])))
        for item in promotions
    )
    suppressed = sum(
        lifecycle_status(item) == "suppressed" for item in promotions
    )

    retrieval = _mapping(planner.get("cold_start_retrieval", {}))
    retrieved_experience_refs = {
        str(_mapping(item).get("experience_id") or item)
        for key in (
            "failure_experiences",
            "retrieved_failure_experiences",
            "failure_experience_candidates",
        )
        for item in (
            _sequence(cold_proposal.get(key, []))
            or _sequence(retrieval.get(key, []))
        )
        if str(_mapping(item).get("experience_id") or item)
    }
    retrieved_experience_refs.update(
        str(step.get("candidate_ref", ""))
        for step in cold_steps
        if "failure_experience" in candidate_source(step)
        and str(step.get("candidate_ref", ""))
    )

    sessions = [
        _mapping(item)
        for item in _sequence(_field(trace, "agent_sessions", []))
    ]
    continuation_sessions = sum(
        "cold_start_continuation" in str(
            session.get("session_type")
            or session.get("session_kind")
            or _mapping(session.get("snapshot", {})).get("session_kind")
            or ""
        ).casefold()
        for session in sessions
    )
    continuation_bucket = _mapping(
        _mapping(usage.get("by_bucket", {})).get(
            "runtime_dynamic_cold_start_continuation", {}
        )
    )
    if continuation_sessions == 0 and _integer(
        continuation_bucket.get("call_count", 0)
    ) > 0:
        continuation_sessions = 1

    f1_calls = _integer(
        _mapping(_mapping(usage.get("by_bucket", {})).get(
            "failure_extractor_f1", {}
        )).get("call_count", 0)
    )
    f2_calls = _integer(
        _mapping(_mapping(usage.get("by_bucket", {})).get(
            "failure_extractor_f2", {}
        )).get("call_count", 0)
    )

    trace_failures = [
        _mapping(item) for item in _sequence(_field(trace, "failures", []))
    ]

    def failure_count(*codes: str) -> int:
        wanted = set(codes)
        count = sum(str(item.get("code", "")) in wanted for item in trace_failures)
        return count or int(any(code in failure_codes for code in wanted))

    p0_rejections = _sequence(
        planner.get("p0_exact_contract_rejections", [])
    )
    if not p0_rejections:
        p0_rejections = _sequence(planner.get("exact_contract_rejections", []))
    if not p0_rejections:
        p0_rejections = [
            item
            for raw in _sequence(planner.get("composite_rejections", []))
            if "goal_contract_exact_mismatch" in {
                str(reason) for reason in _sequence(_mapping(raw).get("reasons", []))
            }
            for item in [raw]
        ]

    plan_source = str(plan.get("source", "")).casefold()
    cold_triggered = bool(
        cold_plan
        or cold_proposal
        or retrieval
        or cold_steps
        or "cold_start" in plan_source
    )
    cold_plan_valid = bool(cold_proposal) and _boolean(
        cold_validation.get("passed", cold_validation.get("valid", False))
    )
    executable_step_ids = [
        str(item)
        for item in _sequence(cold_plan.get("executable_step_ids", []))
        if str(item)
    ]
    executable_prefix_nonempty = cold_plan_valid and bool(executable_step_ids)
    executable_prefix_empty = cold_plan_valid and not executable_step_ids
    continuation_only = bool(
        executable_prefix_empty
        and plan_source == "full_dynamic"
        and cold_plan
    )

    derived = {
        "repeat_block_count": len(repeat_blocks),
        "expanded_requirement_instance_count": len(instances),
        "repeated_atomic_occurrence_count": repeated_occurrences,
        "runtime_distinctness_rejection_count": failure_count(
            "runtime_repetition_distinctness_violation",
            "provisional_repetition_distinctness_violation",
        ),
        "runtime_shared_value_rejection_count": failure_count(
            "runtime_repetition_shared_value_violation",
            "provisional_repetition_shared_value_violation",
        ),
        "cold_start_trigger_count": int(cold_triggered),
        "cold_start_plan_valid_count": int(cold_plan_valid),
        "cold_start_c1_validation_pass_count": int(cold_plan_valid),
        "cold_start_executable_prefix_nonempty_count": int(
            executable_prefix_nonempty
        ),
        "cold_start_executable_prefix_empty_count": int(
            executable_prefix_empty
        ),
        "cold_start_admitted_prefix_step_count": len(executable_step_ids),
        "cold_start_executed_scaffold_step_count": len(cold_steps),
        "cold_start_continuation_only_count": int(continuation_only),
        "cold_start_scaffold_step_count": len(cold_steps),
        "cold_start_verified_step_success_count": sum(
            _boolean(step.get("local_effect_passed", False))
            for step in verified_steps
        ),
        "provisional_trial_count": len(provisional_steps),
        "provisional_local_success_count": sum(
            _boolean(step.get("local_effect_passed", False))
            for step in provisional_steps
        ),
        "provisional_local_failure_count": sum(
            not _boolean(step.get("local_effect_passed", False))
            for step in provisional_steps
        ),
        "cold_start_assisted_success_count": int(_boolean(
            _field(trace, "cold_start_assisted_success", False)
        )),
        "runtime_dynamic_cold_start_continuation_count": continuation_sessions,
        "failure_extractor_f1_count": f1_calls,
        "failure_extractor_f2_count": f2_calls,
        # Eligibility is written by the failure-extraction coordinator from
        # its code-owned predicate.  Never infer it from whether F1 happened.
        "failure_extractor_eligible_count": int(_boolean(
            metadata.get("failure_extractor_eligible", False)
        )),
        "provisional_created_count": len(provisional_refs),
        "provisional_trial_ready_count": trial_ready,
        "provisional_trial_supported_count": trial_supported,
        "provisional_promoted_count": promoted,
        "provisional_suppressed_count": suppressed,
        "failure_experience_observed_count": len(failure_experience_ids),
        "failure_experience_confirmed_count": 0,
        "failure_experience_resolved_count": 0,
        "failure_experience_retrieval_count": len(retrieved_experience_refs),
        "verified_atomic_full_coverage_count": int(
            planner_atomic_full_coverage
        ),
        "planner_p2_count": int(planner_p2_used),
        "p0_exact_contract_rejection_count": len(p0_rejections),
        "failure_side_read_count": 0,
        "provisional_selected_count": max(
            len(provisional_steps), len(planned_provisional_steps)
        ),
    }
    return {name: value(name, derived[name]) for name in V31_METHOD_METRICS}


def _trace_cost(
    trace: Any,
    metadata: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> float | None:
    for key in ("cost_usd", "total_cost_usd", "cost"):
        if metadata.get(key) is not None:
            return round(_nonnegative_number(metadata[key]), 9)
    values: list[float] = []
    for event in events:
        provider = _mapping(event.get("provider_metadata", {}))
        value = _first_present(event, "cost_usd", "cost", default=None)
        if value is None:
            value = _first_present(provider, "cost_usd", "cost", default=None)
        if value is not None:
            values.append(_nonnegative_number(value))
    return round(sum(values), 9) if values else None


def _failure_codes(
    trace: Any,
    nodes: Sequence[Mapping[str, Any]],
    invocations: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
) -> list[str]:
    codes: set[str] = set()
    for failure in _sequence(_field(trace, "failures", [])):
        _collect_codes(_mapping(failure), codes)
    for node in nodes:
        _collect_codes(_mapping(node.get("failure", {})), codes)
    for item in (*invocations, *executions):
        _collect_codes(_mapping(item.get("preflight", {})), codes)
        _collect_codes(_mapping(item.get("result", {})), codes)
    return sorted(code for code in codes if code)


def _collect_codes(value: Any, codes: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"code", "failure_code", "error_code"} and item:
                codes.add(str(item))
            elif key == "failure_codes" and isinstance(item, (list, tuple)):
                codes.update(str(code) for code in item if code)
            elif isinstance(item, (Mapping, list, tuple)):
                _collect_codes(item, codes)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_codes(item, codes)


def _looks_like_report_row(value: Mapping[str, Any]) -> bool:
    return "task_id" in value and "usage_by_bucket" in value and "plan_source" in value


def _last_nonempty(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        value = row.get(key)
        if _has_value(value):
            return value
    return {}


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 9)


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return None if denominator == 0 else round(float(numerator) / denominator, 9)


def _markdown_pairs(values: Sequence[tuple[str, Any]]) -> list[str]:
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {_markdown_cell(name)} | {_display(value)} |" for name, value in values)
    return lines


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if _boolean(value) else "no"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pretty_json(value: Any) -> str:
    return json.dumps(_primitive(value), ensure_ascii=False, sort_keys=True, indent=2)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return _canonical_json(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _has_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, Mapping, Sequence, set, frozenset)):
        return len(value) > 0
    return True


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _primitive(getattr(value, item.name)) for item in fields(value)}
    return {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _boolean(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.casefold()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(value or 0)


def _nonnegative_integer(value: Any) -> int:
    result = _integer(value)
    if result < 0:
        raise ValueError("usage counts must be non-negative")
    return result


def _number(value: Any, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("report values must be finite")
    return result


def _nonnegative_number(value: Any) -> float:
    result = _number(value, 0.0)
    if result < 0:
        raise ValueError("usage/cost values must be non-negative")
    return result


def _primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_primitive(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=_canonical_json)
        return items
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values cannot enter a report")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))


__all__ = [
    "REPORT_COLUMNS",
    "R21_RUNTIME_METRICS",
    "R31_RUNTIME_METRICS",
    "R22_FAILURE_EXTRACTOR_METRICS",
    "ReportPaths",
    "USAGE_BUCKETS",
    "V31_METHOD_METRICS",
    "build_report_rows",
    "generate_reports",
    "render_markdown",
    "summarize_traces",
    "trace_to_row",
    "validate_frozen_v31_guards",
    "validate_formal_usage",
    "validate_usage_event_persistence",
    "write_csv",
    "write_jsonl",
    "write_markdown",
    "write_reports",
]
