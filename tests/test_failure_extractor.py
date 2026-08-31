from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from atomic_skillgraph.evolution.failure_extraction_validator import (
    FailureAssetRecordBuilder,
    FailureAssetValidator,
    FailureAtomicSourceReplay,
    FailureExtractionCoordinator,
    FailureExtractionEligibility,
    FailurePlanAlignmentValidator,
    PreparedFailureExtraction,
)
from atomic_skillgraph.evolution.failure_extractor_session import (
    FailureAtomicProposal,
    FailureExtractionProposal,
    FailureExtractorSession,
    FailurePlanAlignment,
    PlanStepAlignment,
)
from atomic_skillgraph.core.contracts import (
    CapabilityRequirement,
    ContractSource,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.knowledge import FailureKnowledgeStore, StateDatabase
from atomic_skillgraph.knowledge.failure_knowledge_store import ProvisionalStatus
from atomic_skillgraph.planner.multiplicity import (
    RequirementExpansion,
    RequirementInstance,
)
from atomic_skillgraph.traces.schema import (
    EnvironmentActionRecord,
    RuntimeSpan,
    TaskRecord,
    TraceRecord,
    ValidationRecord,
)
from atomic_skillgraph.validation.tool_validator import ToolValidator


class _SubmissionQueue:
    def __init__(self, values: list[dict]) -> None:
        self.values = list(values)

    def request(self, *_args, **_kwargs):
        return SimpleNamespace(value=self.values.pop(0))


class _BucketSession:
    def __init__(self) -> None:
        self.buckets: list[str] = []

    def set_usage_bucket(self, value: str) -> None:
        self.buckets.append(value)


def _alignment_payload() -> dict:
    return {
        "alignment_id": "alignment",
        "step_alignments": [{
            "step_id": "acquire",
            "status": "achieved",
            "event_start": 0,
            "event_end": 1,
            "effect_witness_refs": ["effect:w1"],
            "rationale": "authoritative effect",
        }],
        "matched_prefix_step_ids": ["acquire"],
        "first_unrecovered_divergence": {
            "kind": "unresolved_requirement",
            "step_id": "place",
            "event_index": 1,
            "summary": "the next requirement remains unresolved",
        },
        "remaining_requirement_instance_ids": ["single::place"],
        "candidate_progress_spans": [{
            "step_id": "acquire",
            "event_start": 0,
            "event_end": 1,
            "effect_witness_refs": ["effect:w1"],
        }],
    }


def _extraction_payload() -> dict:
    return {
        "provisional_atomics": [{
            "atomic_proposal": {
                "phase_id": "acquire",
                "intent": "acquire_target_object",
                "event_start": 0,
                "event_end": 1,
                "input_roles": {"object": "apple_1", "source": "table_1"},
                "output_roles": {"held_object": "apple_1"},
                "preconditions": [],
                "effects": [{
                    "predicate": "agent.holds",
                    "args": {"object": "apple_1"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "rationale": "accepted transition",
            },
            "aligned_plan_step_ids": ["acquire"],
            "progress_relation": "consumed_prerequisite",
        }],
        "validated_plan_prefix": ["acquire"],
        "negative_method_suffix": {
            "summary": "avoid retrying an unresolved placement method",
        },
        "reusable_failure_summary": {
            "summary": "placement prerequisite remained unresolved",
        },
    }


def test_f1_and_f2_are_exactly_two_turns_in_the_same_session() -> None:
    raw = _BucketSession()
    extractor = FailureExtractorSession(raw)
    extractor.submissions = _SubmissionQueue([
        _alignment_payload(), _extraction_payload(),
    ])

    alignment = extractor.align(
        task_contract={}, requirement_expansion={}, cold_start_plan={},
        trace_events=[], task_progress=[], failures=[], candidate_contracts=[],
    )
    proposal = extractor.extract(
        validated_alignment=alignment,
        authoritative_trace=[],
        task_contract={},
    )

    assert raw.buckets == ["failure_extractor_f1", "failure_extractor_f2"]
    assert proposal.provisional_atomics[0].progress_relation == "consumed_prerequisite"
    with pytest.raises(RuntimeError, match="F1 may run exactly once"):
        extractor.align(
            task_contract={}, requirement_expansion={}, cold_start_plan={},
            trace_events=[], task_progress=[], failures=[], candidate_contracts=[],
        )
    with pytest.raises(RuntimeError, match="may run exactly once"):
        extractor.extract(
            validated_alignment=alignment,
            authoritative_trace=[],
            task_contract={},
        )


def _trace() -> SimpleNamespace:
    action = SimpleNamespace(
        accepted=True, revision=0, new_revision=1,
    )
    validation = SimpleNamespace(
        result={"passed": True, "witness_refs": ["effect:w1"]},
    )
    return SimpleNamespace(
        environment_actions=[action],
        validations=[validation],
    )


def _alignment() -> FailurePlanAlignment:
    return FailurePlanAlignment(
        alignment_id="alignment",
        step_alignments=[PlanStepAlignment(
            "acquire", "achieved", 0, 1, ["effect:w1"], "effect passed",
        )],
        matched_prefix_step_ids=["acquire"],
        first_unrecovered_divergence={
            "kind": "unresolved_requirement", "step_id": "place",
            "event_index": 1, "summary": "next requirement unresolved",
        },
        remaining_requirement_instance_ids=["single::place"],
        candidate_progress_spans=[{
            "step_id": "acquire", "event_start": 0, "event_end": 1,
            "effect_witness_refs": ["effect:w1"],
        }],
    )


def _proposal() -> FailureExtractionProposal:
    value = _extraction_payload()
    item = value["provisional_atomics"][0]
    return FailureExtractionProposal(
        provisional_atomics=[FailureAtomicProposal(
            item["atomic_proposal"], item["aligned_plan_step_ids"],
            item["progress_relation"],
        )],
        validated_plan_prefix=value["validated_plan_prefix"],
        negative_method_suffix=value["negative_method_suffix"],
        reusable_failure_summary=value["reusable_failure_summary"],
    )


def test_f1_keeps_later_independent_effect_span_after_first_divergence() -> None:
    cleaned, result = FailurePlanAlignmentValidator().validate(
        _alignment(),
        cold_start_plan=SimpleNamespace(control_sequence=["acquire", "place"]),
        trace=_trace(),
    )
    assert result.passed
    assert cleaned is not None
    assert cleaned.candidate_progress_spans == [{
        "step_id": "acquire", "event_start": 0, "event_end": 1,
        "effect_witness_refs": ["effect:w1"],
    }]


def test_source_replay_failure_rejects_all_assets_before_any_store_write() -> None:
    validation = FailureAssetValidator().validate(
        _proposal(),
        alignment=_alignment(),
        trace=_trace(),
        source_replay=lambda _item: {"passed": False},
    )
    assert not validation.result.passed
    assert "provisional_source_replay_failed" in validation.result.failure_codes
    assert validation.proposal.provisional_atomics == []

    class Extractor:
        def align(self, **_kwargs):
            return _alignment()

        def extract(self, **_kwargs):
            return _proposal()

    class RecordBuilder:
        def build(self, *_args):
            raise AssertionError("record builder must not run after content rejection")

    prepared = FailureExtractionCoordinator().prepare(
        eligibility=FailureExtractionEligibility(
            True, True, False, False, "online",
        ),
        extractor=Extractor(),
        task_contract={},
        requirement_expansion={},
        cold_start_plan=SimpleNamespace(
            control_sequence=["acquire", "place"],
        ),
        trace=_trace(),
        task_progress=[],
        failures=[],
        candidate_contracts=[],
        source_replay=lambda _item: {"passed": False},
        record_builder=RecordBuilder(),
    )
    assert not prepared.accepted
    assert prepared.rejection["code"] == "provisional_source_replay_failed"

    class Store:
        writes = 0

        def upsert_provisional(self, _record):
            self.writes += 1

        def upsert_failure_experience(self, _record):
            self.writes += 1

    store = Store()
    assert prepared.commit(store) == ([], [])
    assert store.writes == 0

    def programming_error(_item):
        raise RuntimeError("replay backend corrupted")

    with pytest.raises(RuntimeError, match="backend corrupted"):
        FailureAssetValidator().validate(
            _proposal(), alignment=_alignment(), trace=_trace(),
            source_replay=programming_error,
        )


def test_failure_extractor_eligibility_is_strict_and_online_only() -> None:
    valid = FailureExtractionEligibility(True, True, False, False, "online")
    assert valid.passed
    assert not FailureExtractionEligibility(
        True, False, False, False, "online",
    ).passed
    assert not FailureExtractionEligibility(
        True, True, True, False, "online",
    ).passed
    assert not FailureExtractionEligibility(
        True, True, False, False, "frozen",
    ).passed


def test_validated_f2_enters_trial_ready_only_after_fresh_source_replay(
    tmp_path: Path,
) -> None:
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "object.at_location",
            {"object": "apple", "location": "bowl"},
        )],
        source=ContractSource.ADAPTER_DERIVED,
        confidence=1.0,
        validator_id="fake",
    )
    trace = TraceRecord.create(
        TaskRecord(
            "failed-task", "fake", "place an apple", "place", "signature",
        ),
        to_primitive(contract), {}, {"source": "cold_start"},
    )
    trace.environment_actions = [EnvironmentActionRecord(
        "action-1", 0, "TAKE",
        {"object": "apple_1", "source": "table_1"},
        True, "taken", False, False, 1, "trial-span",
    )]
    trace.runtime_spans = [RuntimeSpan(
        "trial-span", "runtime_provisional_seeded", "cold::acquire",
        0, 1, None, True,
    )]
    trace.validations = [ValidationRecord(
        "cold::acquire", "atomic",
        {"passed": True, "witness_refs": ["effect:w1"]}, 1,
    )]
    acquire = CapabilityRequirement(
        "acquire", "acquire_target_object",
        [SemanticPredicate("agent.holds", {"object": "$object"})],
        [ParameterSpec("object", "entity"), ParameterSpec("source", "entity")],
        [ParameterSpec("held_object", "entity")],
        [], [], True, "required prerequisite",
    )
    place = CapabilityRequirement(
        "place", "place_target_object",
        [SemanticPredicate(
            "object.at_location", {"object": "$object", "location": "$location"},
        )],
        [ParameterSpec("object", "entity"), ParameterSpec("location", "entity")],
        [], [], [], True, "task target",
    )
    expansion = RequirementExpansion(
        templates=(acquire, place),
        repeat_blocks=(),
        instances=(
            RequirementInstance("single::acquire", "acquire", "", 0, acquire),
            RequirementInstance("single::place", "place", "", 0, place),
        ),
        instance_ids_by_template={
            "acquire": ("single::acquire",),
            "place": ("single::place",),
        },
    )
    plan = SimpleNamespace(
        control_sequence=["acquire", "place"],
        steps=[
            SimpleNamespace(step_id="acquire", requirement_instance_ids=["single::acquire"]),
            SimpleNamespace(step_id="place", requirement_instance_ids=["single::place"]),
        ],
    )

    class Harness:
        profile_name = "fake_v3"

        def __init__(self):
            self.replayed = 0

        def replay_tool(self, _task, _tool, case):
            self.replayed += 1
            assert case["trace_id"] == trace.trace_id
            return True

        def supports_constraint(self, _kind, _verifier_id=""):
            return True

    harness = Harness()
    source_replay = FailureAtomicSourceReplay(
        trace=trace,
        task=SimpleNamespace(task_id=trace.task.task_id),
        normalizer=TraceNormalizer(),
        atomicizer=Atomicizer(),
        tool_compiler=ToolCompiler(),
        admission=Admission(ToolValidator()),
        harness=harness,
    )
    f2 = FailureAssetValidator().validate(
        _proposal(), alignment=_alignment(), trace=trace,
        source_replay=source_replay,
    )
    assert f2.result.passed
    records, experience = FailureAssetRecordBuilder(
        source_replay,
        task_contract=contract,
        requirement_expansion=expansion,
        cold_start_plan=plan,
        trace=trace,
        harness_profile=harness.profile_name,
    ).build(_alignment(), f2)
    assert harness.replayed == 1
    assert records[0].status is ProvisionalStatus.TRIAL_READY
    assert records[0].source_replay["passed"] is True

    data_dir = tmp_path / "data_v3"
    with StateDatabase(data_dir / "state.sqlite3") as database:
        store = FailureKnowledgeStore(data_dir, database)
        staged = PreparedFailureExtraction(
            _alignment(), None, f2, records, experience, {},
        )
        refs, experience_ids = staged.commit(store)
        assert refs == [records[0].provisional_ref]
        assert experience_ids == [experience.experience_id]
        assert store.get_provisional(refs[0]).status is ProvisionalStatus.TRIAL_READY

    class BrokenStore:
        def upsert_provisional(self, _record):
            raise RuntimeError("database write failed")

    with pytest.raises(RuntimeError, match="database write failed"):
        staged.commit(BrokenStore())
