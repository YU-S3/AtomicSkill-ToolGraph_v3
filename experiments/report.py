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
    "failure_extractor_f1",
    "failure_extractor_f2",
    "evolution_repair",
    "unattributed",
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
    "cold_start_scaffold_step_count",
    "cold_start_verified_step_success_count",
    "provisional_trial_count",
    "provisional_local_success_count",
    "provisional_local_failure_count",
    "cold_start_assisted_success_count",
    "runtime_dynamic_cold_start_continuation_count",
    "failure_extractor_f1_count",
    "failure_extractor_f2_count",
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
    "task_token_budget_exhausted_count",
    "node_token_budget_exhausted_count",
    *V31_METHOD_METRICS,
    *EXTRACTOR_QUALITY_METRICS,
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
        benchmark_success and task_contract_success
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
        "task_token_budget_exhausted_count": int(
            "runtime_task_token_budget_exhausted" in failure_codes
        ),
        "node_token_budget_exhausted_count": int(
            "runtime_node_token_budget_exhausted" in failure_codes
        ),
        **v31_metrics,
        **{
            name: _integer(quality.get(name, 0))
            for name in EXTRACTOR_QUALITY_METRICS
        },
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
        "artifact_growth": _last_nonempty(rows, "artifact_growth"),
        "artifact_lifecycle": _last_nonempty(rows, "artifact_lifecycle"),
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
    ) + ((
        "planner_p1r_reason_distribution",
        _canonical_json(summary.get("planner_p1r_reason_distribution", {})),
    ),)
    lines.extend(_markdown_pairs(quality))
    lines.extend(["", "## v3.1 method patch", ""])
    lines.extend(_markdown_pairs(tuple(
        (name, summary.get(name, 0))
        for name in V31_METHOD_METRICS
    )))
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
) -> ReportPaths:
    """Emit task rows plus resource summaries from auxiliary immutable traces."""

    if not stem or Path(stem).name != stem:
        raise ValueError("report stem must be a non-empty filename stem")
    rows = [trace_to_row(trace) for trace in traces]
    summary = summarize_traces(
        rows, auxiliary_usage_traces=auxiliary_usage_traces,
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
    repeated_occurrences = sum(
        bool(str(item.get("repeat_block_id", "")))
        or _integer(item.get("repeat_index", -1)) >= 0
        for item in instances
    )
    if not instances:
        repeated_step_ids = {
            str(step_id)
            for raw in _sequence(plan.get("repeat_constraints", []))
            for iteration in _sequence(_mapping(raw).get("iteration_steps", []))
            for step_id in _sequence(iteration)
            if str(step_id)
        }
        repeated_occurrences = len(repeated_step_ids)

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
