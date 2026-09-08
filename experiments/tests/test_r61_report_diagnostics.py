"""R6.1 reporting remains diagnostic-only and preserves exact E2 counts."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.report import summarize_traces, trace_to_row, write_reports


REASONING_AUDIT = {
    "configured_reasoning_effort": "high",
    "effective_reasoning_effort": "high",
    "reasoning_effort_source": "explicit_request",
}


def _trace() -> dict[str, object]:
    return {
        "trace_id": "trace-r61-report",
        "schema_version": 3,
        "task": {
            "task_id": "task-r61-report",
            "task_signature": "alfworld:r61-report",
            "benchmark": "alfworld",
            "task_type": "train",
        },
        "benchmark_success": False,
        "strict_task_success": False,
        "metadata": {
            "extractor_quality": {
                "extractor_e2_protocol_repair_count": 1,
                "extractor_e2_repair_attempt_count": 1,
                "extractor_e2_repair_success_count": 0,
                "extractor_e2_repair_failure_count": 1,
            },
            "extraction": {
                "attempted": True,
                "stage": "atomic_only",
                "prepared": True,
                "applied": True,
                "e2_attempted": True,
                "e2_initial_validation_error": (
                    "E2 reused required input must have explicit DataFlow"
                ),
                "e2_repair_attempted": True,
                "e2_repair_applied": False,
                "e2_repair_error": "repair remained invalid",
            },
        },
    }


def test_r61_e2_diagnostics_are_projected_and_aggregated(
    tmp_path: Path,
) -> None:
    trace = _trace()
    row = trace_to_row(trace)

    assert row["extractor_e2_protocol_repair_count"] == 1
    assert row["extractor_e2_repair_attempt_count"] == 1
    assert row["extractor_e2_repair_success_count"] == 0
    assert row["extractor_e2_repair_failure_count"] == 1
    assert row["e2_repair_attempted"] is True
    assert row["e2_repair_applied"] is False
    assert row["e2_repair_error"] == "repair remained invalid"

    summary = summarize_traces(
        [row], reasoning_effort_audit=REASONING_AUDIT,
    )
    assert summary["extractor_e2_protocol_repair_count"] == 1
    assert summary["extractor_e2_repair_attempt_count"] == 1
    assert summary["extractor_e2_repair_success_count"] == 0
    assert summary["extractor_e2_repair_failure_count"] == 1
    assert summary["configured_reasoning_effort"] == "high"
    assert summary["effective_reasoning_effort"] == "high"
    assert summary["reasoning_effort_source"] == "explicit_request"

    paths = write_reports(
        [trace], tmp_path, stem="r61",
        reasoning_effort_audit=REASONING_AUDIT,
    )
    persisted = json.loads(paths.jsonl.read_text(encoding="utf-8"))
    assert persisted["extractor_e2_protocol_repair_count"] == 1
    assert persisted["configured_reasoning_effort"] == "high"
    assert persisted["effective_reasoning_effort"] == "high"
    assert persisted["reasoning_effort_source"] == "explicit_request"
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "## Reasoning effort audit" in markdown
    assert "| configured_reasoning_effort | high |" in markdown
    assert "| effective_reasoning_effort | high |" in markdown
    assert "| reasoning_effort_source | explicit_request |" in markdown
    assert "| extractor_e2_protocol_repair_count | 1 |" in markdown
    assert "| extractor_e2_repair_attempt_count | 1 |" in markdown
    assert "| extractor_e2_repair_success_count | 0 |" in markdown
    assert "| extractor_e2_repair_failure_count | 1 |" in markdown
