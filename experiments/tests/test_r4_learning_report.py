from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from experiments.report import (
    LEGACY_LEARNING_DIAGNOSTICS_VERSION,
    R4_LEARNING_DIAGNOSTICS_VERSION,
    summarize_traces,
    trace_to_row,
    write_reports,
)


def _trace(task_id: str) -> dict[str, object]:
    return {
        "trace_id": f"trace-{task_id}",
        "schema_version": 3,
        "task": {
            "task_id": task_id,
            "task_signature": f"alfworld:{task_id}",
            "benchmark": "alfworld",
            "task_type": "train",
        },
        "runtime_plan": {"source": "full_dynamic", "control_sequence": []},
        "node_records": [],
        "benchmark_success": True,
        "strict_task_success": True,
        "metadata": {},
    }


def _r4_trace() -> dict[str, object]:
    trace = _trace("task-r4")
    trace["metadata"] = {
        "learning_diagnostics_version": R4_LEARNING_DIAGNOSTICS_VERSION,
        # These legacy projections deliberately disagree.  The versioned R4
        # authority below must win.
        "extractor_quality": {
            "extractor_e1_proposal_count": 99,
            "extractor_e1_validated_occurrence_count": 0,
            "extractor_e1_rejection_count": 99,
        },
        "extraction": {
            "attempted": True,
            "e1_proposed": 99,
            "e1_validated": 0,
            "e1_rejected": 99,
        },
        "v32_metrics": {
            "extractor_e1_proposal_count": 2,
            "extractor_e1_validated_occurrence_count": 1,
            "extractor_e1_rejection_count": 1,
            "tool_builder_call_count": 1,
            "tool_builder_proposal_count": 1,
            "tool_builder_no_tool_count": 0,
            "tool_builder_submission_rejection_count": 0,
            "tool_builder_static_pass_count": 0,
            "tool_builder_static_rejection_count": 1,
            "tool_builder_aborted_count": 0,
            "atomic_only_prepared_after_tool_rejection_count": 1,
            "atomic_only_retained_after_tool_rejection_count": 1,
            "atomic_staged_occurrence_count": 1,
        },
        "evolution_tool_builds": [{
            "occurrence_id": "occ-valid",
            "phase_id": "phase-valid",
            "atomic_ref": "skill://atomic_valid@1.0.0",
            "source": "success_evolution",
            "outcome": "static_rejected",
            "builder_entered": True,
            "session_id": "builder-session-1",
            "proposal_received": True,
            "static_checked": True,
            "static_passed": False,
            "atomic_only_prepared": True,
            "atomic_registered": True,
            "registered_atomic_ref": "skill://atomic_valid@1.0.0",
            "failure_stage": "tool_static",
            "error_code": "tool_builder_static_rejected",
            # Two failure codes still describe one rejected proposal.
            "failure_codes": [
                "tool_ir_final_effects_missing",
                "tool_ir_action_schema_invalid",
            ],
            "messages": ["final Effect does not match the Atomic contract"],
        }],
        "extraction_occurrence_rejections": [{
            "stage": "atomicizer_validation",
            "phase_id": "phase-invalid",
            "error_code": "atomic_precondition_entry_witness_missing",
            "failure_codes": ["atomic_precondition_entry_witness_missing"],
            "messages": ["Atomic precondition lacks before-state witness"],
        }],
        "tool_build_rejections": [{
            "stage": "tool_static",
            "phase_id": "phase-valid",
            "error_code": "tool_builder_static_rejected",
            "failure_codes": [
                "tool_ir_final_effects_missing",
                "tool_ir_action_schema_invalid",
            ],
            "messages": ["final Effect does not match the Atomic contract"],
        }],
        "knowledge_preparation_rejections": [],
    }
    return trace


def _legacy_trace() -> dict[str, object]:
    trace = _trace("task-legacy")
    trace["metadata"] = {
        "extractor_quality": {
            "extractor_e1_proposal_count": 2,
            "extractor_e1_validated_occurrence_count": 0,
            "extractor_e1_rejection_count": 2,
        },
        "v32_metrics": {
            "tool_builder_call_count": 2,
            "tool_builder_no_tool_count": 1,
            "tool_builder_static_pass_count": 1,
            "tool_builder_static_rejection_count": 0,
        },
    }
    return trace


def test_r4_report_does_not_mix_learning_stages(tmp_path: Path) -> None:
    r4 = _r4_trace()
    legacy = _legacy_trace()

    row = trace_to_row(r4)
    assert row["learning_diagnostics_version"] == R4_LEARNING_DIAGNOSTICS_VERSION
    assert row["extractor_e1_proposal_count"] == 2
    assert row["extractor_e1_validated_occurrence_count"] == 1
    assert row["extractor_e1_rejection_count"] == 1
    assert row["e1_proposed"] == 2
    assert row["e1_validated"] == 1
    assert row["e1_rejected"] == 1
    assert row["tool_builder_proposal_count"] == 1
    assert row["tool_builder_static_rejection_count"] == 1
    assert row["atomic_only_retained_after_tool_rejection_count"] == 1
    assert len(row["tool_build_rejections"]) == 1

    legacy_row = trace_to_row(legacy)
    assert legacy_row["learning_diagnostics_version"] == (
        LEGACY_LEARNING_DIAGNOSTICS_VERSION
    )
    assert legacy_row["extractor_e1_proposal_count"] == 2
    assert legacy_row["tool_builder_call_count"] == 2
    # The old call-minus-NO_TOOL formula is not evidence of a create proposal.
    assert legacy_row["tool_builder_proposal_count"] is None
    assert legacy_row["tool_builder_submission_rejection_count"] is None
    assert legacy_row["atomic_staged_occurrence_count"] is None
    assert legacy_row["evolution_tool_builds"] is None

    summary = summarize_traces([row, legacy_row])
    groups = summary["learning_diagnostics_by_version"]
    assert groups[R4_LEARNING_DIAGNOSTICS_VERSION]["task_count"] == 1
    assert groups[R4_LEARNING_DIAGNOSTICS_VERSION][
        "tool_builder_static_rejection_count"
    ] == 1
    assert groups[LEGACY_LEARNING_DIAGNOSTICS_VERSION]["task_count"] == 1
    assert groups[LEGACY_LEARNING_DIAGNOSTICS_VERSION][
        "tool_builder_submission_rejection_count"
    ] is None
    # A mixed-version top-level number would be a false common denominator.
    assert summary["extractor_e1_proposal_count"] is None
    assert summary["tool_builder_static_rejection_count"] is None

    paths = write_reports([r4, legacy], tmp_path, stem="r4")
    jsonl_rows = [
        json.loads(line)
        for line in paths.jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert jsonl_rows[0]["tool_build_rejections"][0]["phase_id"] == (
        "phase-valid"
    )
    assert jsonl_rows[1]["tool_builder_submission_rejection_count"] is None

    with paths.csv.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert json.loads(csv_rows[0]["tool_build_rejections"])[0][
        "error_code"
    ] == "tool_builder_static_rejected"
    assert csv_rows[1]["tool_builder_submission_rejection_count"] == ""

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "### E1 Atomic validation (v3.2-r4)" in markdown
    assert "### ToolBuilder construction (v3.2-r4)" in markdown
    assert "Legacy mixed or incomplete learning diagnostics" in markdown
    assert "no top-level learning-stage total" in markdown
    assert "tool_ir_final_effects_missing" in markdown
    assert "tool_ir_action_schema_invalid" in markdown


def test_r4_static_failure_codes_count_one_rejected_proposal() -> None:
    row = trace_to_row(_r4_trace())
    assert row["tool_builder_static_rejection_count"] == 1
    assert row["tool_builder_proposal_count"] == 1


def test_r4_report_rejects_incomplete_or_inconsistent_authority() -> None:
    missing_stage = _r4_trace()
    del missing_stage["metadata"]["tool_build_rejections"][0]["stage"]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing stage"):
        trace_to_row(missing_stage)

    inconsistent = copy.deepcopy(_r4_trace())
    inconsistent["metadata"]["v32_metrics"][  # type: ignore[index]
        "tool_builder_static_rejection_count"
    ] = 2
    with pytest.raises(ValueError, match="does not match occurrence records"):
        trace_to_row(inconsistent)


def test_r4_report_preserves_partial_builder_records_after_aborted_attempt() -> None:
    interrupted = _r4_trace()
    metadata = cast(dict[str, Any], interrupted["metadata"])
    metrics = metadata["v32_metrics"]
    metrics.update({
        "extractor_e1_proposal_count": 3,
        "extractor_e1_validated_occurrence_count": 3,
        "extractor_e1_rejection_count": 0,
        "tool_builder_call_count": 1,
        "tool_builder_proposal_count": 0,
        "tool_builder_no_tool_count": 0,
        "tool_builder_submission_rejection_count": 0,
        "tool_builder_static_pass_count": 0,
        "tool_builder_static_rejection_count": 0,
        "tool_builder_aborted_count": 1,
        "atomic_only_prepared_after_tool_rejection_count": 0,
        "atomic_only_retained_after_tool_rejection_count": 0,
        "atomic_staged_occurrence_count": 0,
    })
    record = metadata["evolution_tool_builds"][0]
    record.update({
        "outcome": "aborted",
        "proposal_received": False,
        "static_checked": False,
        "static_passed": None,
        "atomic_only_prepared": False,
        "atomic_registered": False,
        "registered_atomic_ref": "",
        "failure_stage": "tool_builder_submission",
        "error_code": "runtime_agent_token_budget_exhausted",
        "failure_codes": ["runtime_agent_token_budget_exhausted"],
        "messages": ["ToolBuilder token budget exhausted"],
    })
    metadata["extraction_occurrence_rejections"] = []
    metadata["tool_build_rejections"] = []

    row = trace_to_row(interrupted)
    assert row["extractor_e1_validated_occurrence_count"] == 3
    assert len(row["evolution_tool_builds"]) == 1
    assert row["tool_builder_call_count"] == 1
    assert row["tool_builder_aborted_count"] == 1
    assert row["tool_builder_proposal_count"] == 0
    assert row["atomic_staged_occurrence_count"] == 0

    # The reporter does not invent records for the two occurrences that were
    # never reached after the provider/budget interruption.
    summary = summarize_traces([row])
    assert summary["extractor_e1_validated_occurrence_count"] == 3
    assert summary["tool_builder_call_count"] == 1
    assert summary["tool_builder_aborted_count"] == 1

    unexplained = copy.deepcopy(interrupted)
    unexplained_record = unexplained["metadata"][  # type: ignore[index]
        "evolution_tool_builds"
    ][0]
    unexplained_record.update({
        "outcome": "no_tool",
        "builder_entered": True,
        "failure_stage": "",
        "error_code": "",
        "failure_codes": [],
        "messages": [],
        "atomic_only_prepared": True,
    })
    unexplained_metrics = unexplained["metadata"][  # type: ignore[index]
        "v32_metrics"
    ]
    unexplained_metrics.update({
        "tool_builder_no_tool_count": 1,
        "tool_builder_aborted_count": 0,
        "atomic_staged_occurrence_count": 1,
    })
    with pytest.raises(ValueError, match="without an aborted interruption"):
        trace_to_row(unexplained)


def test_r4_report_accepts_explicit_atomicizer_interruption_without_guessing() -> None:
    interrupted = _r4_trace()
    metadata = cast(dict[str, Any], interrupted["metadata"])
    metadata["v32_metrics"].update({
        "extractor_e1_proposal_count": 2,
        "extractor_e1_validated_occurrence_count": 0,
        "extractor_e1_rejection_count": 0,
        "tool_builder_call_count": 0,
        "tool_builder_proposal_count": 0,
        "tool_builder_no_tool_count": 0,
        "tool_builder_submission_rejection_count": 0,
        "tool_builder_static_pass_count": 0,
        "tool_builder_static_rejection_count": 0,
        "tool_builder_aborted_count": 0,
        "atomic_only_prepared_after_tool_rejection_count": 0,
        "atomic_only_retained_after_tool_rejection_count": 0,
        "atomic_staged_occurrence_count": 0,
    })
    metadata["evolution_tool_builds"] = []
    metadata["extraction_occurrence_rejections"] = []
    metadata["tool_build_rejections"] = []
    metadata["knowledge_preparation_rejections"] = [{
        "stage": "atomicizer",
        "phase_id": "atomicizer_batch",
        "error_code": "extractor_e1_atomicizer_unexpected",
        "failure_codes": [],
        "messages": ["unexpected Atomicizer failure"],
    }]

    row = trace_to_row(interrupted)
    assert row["extractor_e1_proposal_count"] == 2
    assert row["extractor_e1_validated_occurrence_count"] == 0
    assert row["extractor_e1_rejection_count"] == 0
    assert row["knowledge_preparation_rejections"] == [
        metadata["knowledge_preparation_rejections"][0]
    ]


def test_report_accepts_system_initialized_r4_record_schema() -> None:
    from atomic_skillgraph.system import AtomicSkillGraphSystem

    trace = SimpleNamespace(
        trace_id="trace-system-schema",
        schema_version=3,
        task={"task_id": "task-system-schema"},
        planner_audit={},
        runtime_plan={"source": "full_dynamic", "control_sequence": []},
        node_records=[],
        implementation_invocations=[],
        tool_executions=[],
        benchmark_success=False,
        metadata={},
    )
    AtomicSkillGraphSystem._initialize_r4_learning_diagnostics(trace)
    record = AtomicSkillGraphSystem._new_r4_tool_build_record(
        trace,
        SimpleNamespace(occurrence_id="occ-system", phase_id="phase-system"),
        SimpleNamespace(ref="skill://atomic_system@1.0.0"),
    )
    record["outcome"] = "exact_reuse"
    trace.metadata["v32_metrics"].update({
        "extractor_e1_proposal_count": 1,
        "extractor_e1_validated_occurrence_count": 1,
        "extractor_e1_rejection_count": 0,
    })
    AtomicSkillGraphSystem._finalize_r4_learning_metrics(
        trace,
        atomic_staged_occurrence_count=1,
    )

    row = trace_to_row(trace)
    assert row["learning_diagnostics_version"] == R4_LEARNING_DIAGNOSTICS_VERSION
    assert row["extractor_e1_validated_occurrence_count"] == 1
    assert row["atomic_staged_occurrence_count"] == 1
    assert row["evolution_tool_builds"][0]["outcome"] == "exact_reuse"


def _atomic_view_rejection_trace() -> dict[str, object]:
    trace = _r4_trace()
    metadata = cast(dict[str, Any], trace["metadata"])
    metadata["v32_metrics"].update({
        "extractor_e1_proposal_count": 2,
        "extractor_e1_validated_occurrence_count": 2,
        "extractor_e1_rejection_count": 0,
    })
    metadata["extraction_occurrence_rejections"] = []
    metadata["knowledge_preparation_rejections"] = [{
        "occurrence_id": "occ-before-builder",
        "stage": "atomic_view",
        "phase_id": "phase-before-builder",
        "error_code": "knowledge_preparation_failed",
        "failure_codes": [],
        "messages": ["canonical occurrence has no Atomic view"],
    }]
    return trace


def test_r4_atomic_view_rejection_exactly_explains_missing_build_record() -> None:
    row = trace_to_row(_atomic_view_rejection_trace())

    assert row["extractor_e1_validated_occurrence_count"] == 2
    assert len(row["evolution_tool_builds"]) == 1
    assert row["tool_builder_call_count"] == 1
    assert row["knowledge_preparation_rejections"][0] == {
        "occurrence_id": "occ-before-builder",
        "stage": "atomic_view",
        "phase_id": "phase-before-builder",
        "error_code": "knowledge_preparation_failed",
        "failure_codes": [],
        "messages": ["canonical occurrence has no Atomic view"],
    }


@pytest.mark.parametrize(
    "stage",
    ["atomicizer", "tool_static", "canonical_rewrite", "e2"],
)
def test_r4_non_atomic_view_rejection_does_not_explain_missing_build(
    stage: str,
) -> None:
    trace = _atomic_view_rejection_trace()
    metadata = cast(dict[str, Any], trace["metadata"])
    metadata["knowledge_preparation_rejections"][0]["stage"] = stage

    with pytest.raises(ValueError, match="exact atomic_view rejection"):
        trace_to_row(trace)


def test_r4_duplicate_atomic_view_occurrence_does_not_double_explain_gap() -> None:
    trace = _atomic_view_rejection_trace()
    metadata = cast(dict[str, Any], trace["metadata"])
    metadata["v32_metrics"].update({
        "extractor_e1_proposal_count": 3,
        "extractor_e1_validated_occurrence_count": 3,
    })
    metadata["knowledge_preparation_rejections"].append(copy.deepcopy(
        metadata["knowledge_preparation_rejections"][0]
    ))

    with pytest.raises(ValueError, match="exact atomic_view rejection"):
        trace_to_row(trace)
