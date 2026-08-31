from __future__ import annotations

from pathlib import Path

import yaml

from atomic_skillgraph.agents.usage import REAL_USAGE_BUCKETS, UsageBucket
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.traces.schema import (
    ColdStartPlanRecord,
    ColdStartStepRecord,
    FailureExtractionRecord,
    TaskProgressRecord,
    TaskRecord,
    TraceRecord,
)
from atomic_skillgraph.traces.store import TraceStore
from experiments.fakes import FakeAgentFactory, FakeProviderSet
from experiments.report import (
    USAGE_BUCKETS,
    V31_METHOD_METRICS,
    summarize_traces,
    trace_to_row,
)


ROOT = Path(__file__).resolve().parents[1]
NEW_BUCKETS = (
    "cold_start_c1",
    "cold_start_c1_repair",
    "runtime_provisional_seeded",
    "runtime_dynamic_cold_start_continuation",
    "failure_extractor_f1",
    "failure_extractor_f2",
)


def _trace() -> TraceRecord:
    return TraceRecord.create(
        TaskRecord(
            task_id="task-v31",
            benchmark="alfworld",
            goal="put two objects in a receptacle",
            task_type="pick_two_obj_and_place",
            task_signature="signature-v31",
        ),
        task_contract={"target_effects": []},
        planner_audit={},
        runtime_plan={"source": "full_dynamic"},
    )


def test_v31_configs_freeze_only_the_documented_patch_fields() -> None:
    default = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(
        encoding="utf-8"
    ))
    online = yaml.safe_load((
        ROOT / "configs" / "alfworld_train_full_30.yaml"
    ).read_text(encoding="utf-8"))
    frozen = yaml.safe_load((
        ROOT / "configs" / "alfworld_frozen_eval.yaml"
    ).read_text(encoding="utf-8"))

    expected_cold_start = {
        "enabled": True,
        "provisional_top_k_per_requirement": 3,
        "failure_experience_top_k": 2,
        "scaffold_max_steps": 8,
        "failure_extractor_enabled": True,
        "source_replay_required": True,
        "provisional_suppress_consecutive_failures": 3,
        "promotion_requires_strict_task_success": True,
        "experience_confirm_independent_tasks": 2,
    }
    for config in (default, online, frozen):
        assert config["schema_version"] == 3
        assert config["method_patch"] == "3.1"
        assert config["planner"]["max_repeat_count"] == 4
        assert config["planner"]["max_runtime_occurrences"] == 16
        assert config["planner"]["cold_start_c1_repair_limit"] == 1
    assert default["cold_start"] == expected_cold_start
    assert online["cold_start"] == expected_cold_start
    assert frozen["cold_start"] == {"enabled": False}


def test_trace_v31_records_defaults_metadata_and_roundtrip(tmp_path: Path) -> None:
    trace = _trace()
    assert trace.schema_version == 3
    assert trace.metadata == {"method_patch": "3.1"}
    assert trace.requirement_bundle == {}
    assert trace.requirement_expansion == {}
    assert trace.task_progress_records == []
    assert trace.cold_start_plan is None
    assert trace.cold_start_steps == []
    assert trace.failure_extraction is None
    assert trace.provisional_promotions == []
    assert trace.cold_start_assisted_success is False

    trace.task_progress_records.append(TaskProgressRecord(
        revision=2,
        source="environment_action",
        snapshot={"progress_digest": "digest-2"},
    ))
    trace.cold_start_plan = ColdStartPlanRecord(
        plan_id="cold-plan-1",
        proposal={"steps": []},
        validation={"passed": True},
        repair_used=False,
        executable_step_ids=["step-1"],
        first_unresolved_step_id="",
    )
    trace.cold_start_steps.append(ColdStartStepRecord(
        step_id="step-1",
        candidate_source="verified_atomic",
        candidate_ref="skill://atomic_probe@1.0.0",
        execution_mode="direct",
        outcome="success",
        local_effect_passed=True,
        action_start=0,
        action_end=1,
        progress_before="digest-1",
        progress_after="digest-2",
        failure_code="",
    ))
    trace.failure_extraction = FailureExtractionRecord(
        f1_alignment={"passed": True},
        f1_validation={"passed": True},
        f2_proposal={"kind": "provisional"},
        provisional_refs=["provisional://probe"],
        failure_experience_ids=["failure://probe"],
        rejection={},
    )
    trace.provisional_promotions.append({
        "provisional_ref": "provisional://probe",
        "status": "PROMOTED",
    })
    trace.cold_start_assisted_success = True

    expected = to_primitive(trace)
    assert expected["metadata"]["method_patch"] == "3.1"
    assert expected["task_progress_records"][0]["revision"] == 2
    assert expected["cold_start_plan"]["plan_id"] == "cold-plan-1"
    assert expected["failure_extraction"]["provisional_refs"] == [
        "provisional://probe"
    ]

    store = TraceStore(tmp_path)
    store.save_atomic(trace)
    assert to_primitive(store.load(trace.trace_id)) == expected


def test_v31_usage_buckets_are_real_report_buckets_and_fake_aliases() -> None:
    real = {bucket.value for bucket in REAL_USAGE_BUCKETS}
    assert set(NEW_BUCKETS) <= real
    assert set(NEW_BUCKETS) <= set(USAGE_BUCKETS)
    assert UsageBucket.UNATTRIBUTED.value not in real

    factory = FakeAgentFactory()
    for kind in NEW_BUCKETS:
        assert factory.new_session(kind).snapshot()["usage_bucket"] == kind

    providers = FakeProviderSet()
    assert providers["cold_start_c1"] is providers["planner"]
    assert providers["cold_start_c1_repair"] is providers["planner"]
    assert providers["runtime_provisional_seeded"] is providers["runtime_seeded"]
    assert (
        providers["runtime_dynamic_cold_start_continuation"]
        is providers["runtime_dynamic"]
    )
    assert providers["failure_extractor_f1"] is providers["extractor"]
    assert providers["failure_extractor_f2"] is providers["extractor"]


def test_report_preserves_new_buckets_and_derives_v31_metrics() -> None:
    trace = _trace()
    trace.planner_audit = {
        "atomic_search_p1": [{
            "requirement": {"required": True},
            "covered": True,
        }],
        "workflow_p2": {"steps": ["step-1"]},
        "p0_exact_contract_rejections": [{"ref": "a"}, {"ref": "b"}],
    }
    trace.runtime_plan = {"source": "cold_start", "repeat_constraints": []}
    trace.requirement_bundle = {"repeat_blocks": [{"block_id": "repeat-1"}]}
    trace.requirement_expansion = {
        "instances": [
            {"instance_id": "repeat-1::0::unit", "repeat_block_id": "repeat-1", "repeat_index": 0},
            {"instance_id": "repeat-1::1::unit", "repeat_block_id": "repeat-1", "repeat_index": 1},
            {"instance_id": "single::finish", "repeat_block_id": "", "repeat_index": -1},
        ]
    }
    trace.cold_start_plan = ColdStartPlanRecord(
        plan_id="cold-plan",
        proposal={"failure_experiences": [{"experience_id": "experience-1"}]},
        validation={"passed": True},
        repair_used=False,
        executable_step_ids=["verified", "provisional-ok", "provisional-fail"],
        first_unresolved_step_id="",
    )
    trace.cold_start_steps = [
        ColdStartStepRecord(
            "verified", "verified_atomic", "skill://verified", "direct",
            "success", True, 0, 1, "p0", "p1", "",
        ),
        ColdStartStepRecord(
            "provisional-ok", "provisional", "provisional://ok",
            "provisional_seeded", "success", True, 1, 2, "p1", "p2", "",
        ),
        ColdStartStepRecord(
            "provisional-fail", "provisional", "provisional://fail",
            "provisional_seeded", "failed", False, 2, 3, "p2", "p2",
            "provisional_atomic_effect_failed",
        ),
    ]
    trace.failure_extraction = FailureExtractionRecord(
        f1_alignment={},
        f1_validation={"passed": True},
        f2_proposal={},
        provisional_refs=["provisional://ok", "provisional://fail"],
        failure_experience_ids=["experience-1"],
        rejection={},
    )
    trace.provisional_promotions = [
        {"status": "TRIAL_READY"},
        {"status": "TRIAL_SUPPORTED"},
        {"status": "PROMOTED", "promoted_verified_refs": ["skill://new"]},
        {"status": "SUPPRESSED"},
    ]
    trace.cold_start_assisted_success = True
    trace.metadata["v31_metrics"] = {
        "failure_experience_confirmed_count": 2,
        "failure_experience_resolved_count": 1,
        "failure_side_read_count": 4,
    }
    events = []
    for index, bucket in enumerate(NEW_BUCKETS):
        events.append({
            "event_id": f"usage-{index}",
            "session_id": f"session-{index}",
            "turn_index": 0,
            "bucket": bucket,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "reasoning_tokens": 0,
            "call_count": 1,
            "latency_ms": 1.0,
            "provider_metadata": {"usage_status": "reported"},
        })
    trace.metadata["usage_snapshot"] = {
        "events": events,
        "reconciliation": {"episode_total_tokens": 2 * len(events)},
    }

    row = trace_to_row(trace)
    assert row["unattributed_total_tokens"] == 0
    assert row["token_mismatch"] == 0
    for bucket in NEW_BUCKETS:
        assert row[f"{bucket}_total_tokens"] == 2
        assert row["usage_by_bucket"][bucket]["call_count"] == 1

    expected_metrics = {
        "repeat_block_count": 1,
        "expanded_requirement_instance_count": 3,
        "repeated_atomic_occurrence_count": 0,
        "cold_start_trigger_count": 1,
        "cold_start_plan_valid_count": 1,
        "cold_start_scaffold_step_count": 3,
        "cold_start_verified_step_success_count": 1,
        "provisional_trial_count": 2,
        "provisional_local_success_count": 1,
        "provisional_local_failure_count": 1,
        "cold_start_assisted_success_count": 1,
        "runtime_dynamic_cold_start_continuation_count": 1,
        "failure_extractor_f1_count": 1,
        "failure_extractor_f2_count": 1,
        "provisional_created_count": 2,
        "provisional_trial_ready_count": 1,
        "provisional_trial_supported_count": 1,
        "provisional_promoted_count": 1,
        "provisional_suppressed_count": 1,
        "failure_experience_observed_count": 1,
        "failure_experience_confirmed_count": 2,
        "failure_experience_resolved_count": 1,
        "failure_experience_retrieval_count": 1,
        "verified_atomic_full_coverage_count": 1,
        "planner_p2_count": 1,
        "p0_exact_contract_rejection_count": 2,
        "failure_side_read_count": 4,
        "provisional_selected_count": 2,
    }
    for name, expected in expected_metrics.items():
        assert row[name] == expected

    summary = summarize_traces([row])
    for name in V31_METHOD_METRICS:
        assert summary[name] == row[name]


def test_report_does_not_infer_repeated_atomic_occurrences_from_requirement_ir() -> None:
    trace = _trace()
    trace.requirement_expansion = {
        "instances": [
            {
                "instance_id": "repeat::0::unit",
                "repeat_block_id": "repeat",
                "repeat_index": 0,
            },
            {
                "instance_id": "repeat::1::unit",
                "repeat_block_id": "repeat",
                "repeat_index": 1,
            },
        ]
    }
    trace.runtime_plan = {
        "source": "full_dynamic",
        "occurrences": [],
        "repeat_constraints": [],
    }

    row = trace_to_row(trace)
    assert row["expanded_requirement_instance_count"] == 2
    assert row["repeated_atomic_occurrence_count"] == 0


def test_report_counts_only_real_repeated_runtime_occurrences() -> None:
    trace = _trace()
    trace.requirement_expansion = {
        "instances": [
            {
                "instance_id": "repeat::0::unit",
                "repeat_block_id": "repeat",
                "repeat_index": 0,
            },
            {
                "instance_id": "repeat::1::unit",
                "repeat_block_id": "repeat",
                "repeat_index": 1,
            },
        ]
    }
    trace.runtime_plan = {
        "source": "atomic_composition",
        "occurrences": [
            {"step_id": "repeat-step-0"},
            {"step_id": "repeat-step-1"},
            {"step_id": "non-repeat-step"},
        ],
        "repeat_constraints": [{
            "block_id": "repeat",
            "iteration_steps": [
                ["repeat-step-0"],
                ["repeat-step-1"],
            ],
        }],
    }

    row = trace_to_row(trace)
    assert row["expanded_requirement_instance_count"] == 2
    assert row["repeated_atomic_occurrence_count"] == 2
