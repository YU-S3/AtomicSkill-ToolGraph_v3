"""Focused regressions for v3.2 report derivation and environment provenance."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.report import summarize_traces, trace_to_row, write_reports


def _v32_trace() -> dict[str, object]:
    return {
        "trace_id": "trace-v32-report",
        "schema_version": 3,
        "task": {
            "task_id": "task-v32-report",
            "task_signature": "alfworld:report",
            "benchmark": "alfworld",
            "task_type": "train",
        },
        "planner_audit": {
            "repairability": {
                "repairable": False,
                "reason_code": "planner_hard_capability_gap",
                "diagnostics": [{"hard_capability_gap": True}],
            },
            "support_atomic_candidates": [
                {"atomic_ref": "skill://atomic_locate@1.0.0"},
                {"atomic_ref": "skill://atomic_open@1.0.0"},
            ],
            "support_atomic_selected": [
                {"atomic_ref": "skill://atomic_locate@1.0.0"},
            ],
        },
        "runtime_plan": {
            "source": "online",
            "control_sequence": ["step-1", "step-2"],
        },
        "node_records": [
            {
                "occurrence_id": "occurrence-1",
                "step_id": "step-1",
                "status": "direct_autonomous_success",
            },
            {
                "occurrence_id": "occurrence-2",
                "step_id": "step-2",
                "status": "skipped_goal_terminal",
            },
        ],
        "tool_executions": [{
            "result": {
                "started": True,
                "completed": False,
                "terminal_interrupted": True,
                "executed_action_count": 1,
                # Control-node visits are deliberately larger than ACTIONs.
                "executed_node_count": 4,
                "path_id": "program/if-1:then/action-1/return-1",
                "validated_paths": ["branch:then"],
                "unvalidated_paths": ["branch:else"],
                "loop_iteration_counts": {"loop-1": 2},
                "stop_condition_witnesses": ["stop:loop-1"],
            },
        }],
        "benchmark_success": True,
        "strict_task_success": True,
        "metadata": {
            "environment": {"alfworld_version": "0.4.2"},
            "task_terminal": {"during": "tool"},
        },
    }


def test_v32_report_uses_structured_fallbacks_and_explicit_overrides() -> None:
    trace = _v32_trace()
    row = trace_to_row(trace)

    assert row["planner_repairability_gate_count"] == 1
    assert row["planner_repairability_repairable_count"] == 0
    assert row["planner_hard_capability_gap_count"] == 1
    assert row["planner_p1r_skipped_hard_gap_count"] == 1
    assert row["planner_support_atomic_candidate_count"] == 2
    assert row["planner_support_atomic_selected_count"] == 1
    assert row["task_terminal_early_success_count"] == 1
    assert row["task_terminal_during_tool_count"] == 1
    assert row["terminal_skipped_occurrence_count"] == 1
    assert row["task_terminal_with_remaining_occurrences_count"] == 1
    assert row["tool_validated_path_count"] == 1
    assert row["tool_unvalidated_path_count"] == 1
    assert row["tool_observed_loop_iteration_count"] == 2
    assert row["tool_stop_condition_witness_count"] == 1
    assert row["runtime_tool_internal_action_count"] == 1
    assert row["runtime_tool_llm_bypassed_action_count"] == 1

    summary = summarize_traces([row])
    assert summary["planner_repairability_gate_count"] == 1
    assert summary["planner_support_atomic_candidate_count"] == 2
    assert summary["planner_support_atomic_selected_count"] == 1
    assert summary["task_terminal_during_tool_count"] == 1
    assert summary["tool_observed_loop_iteration_count"] == 2
    assert summary["runtime_tool_internal_action_count"] == 1
    assert summary["runtime_tool_llm_bypassed_action_count"] == 1

    trace["metadata"]["v32_metrics"] = {  # type: ignore[index]
        "task_terminal_during_tool_count": 7,
    }
    overridden = trace_to_row(trace)
    assert overridden["task_terminal_during_tool_count"] == 7
    assert overridden["tool_observed_loop_iteration_count"] == 2


def test_report_formats_preserve_environment_metadata(tmp_path: Path) -> None:
    trace = _v32_trace()
    paths = write_reports([trace], tmp_path, stem="v32_environment")

    jsonl_row = json.loads(
        paths.jsonl.read_text(encoding="utf-8").strip()
    )
    assert jsonl_row["environment"] == {"alfworld_version": "0.4.2"}

    with paths.csv.open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert json.loads(csv_row["environment"]) == {
        "alfworld_version": "0.4.2",
    }

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "## Environment" in markdown
    assert "| alfworld_version | 0.4.2 |" in markdown
    assert "## v3.2 Agent-driven Tool evolution" in markdown
    assert "| planner_repairability_gate_count | 1 |" in markdown
