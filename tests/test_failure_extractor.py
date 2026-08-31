from __future__ import annotations

import json
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
    RepeatBlock,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.knowledge import FailureKnowledgeStore, StateDatabase
from atomic_skillgraph.knowledge.failure_knowledge_store import (
    FailureExperienceStatus,
    ProvisionalStatus,
)
from atomic_skillgraph.planner.cold_start_retriever import FailureExperienceRetriever
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


def _cold_plan(
    *,
    acquire_step: str = "acquire",
    place_step: str = "place",
    acquire_instance: str = "single::acquire",
    place_instance: str = "single::place",
) -> SimpleNamespace:
    return SimpleNamespace(
        control_sequence=[acquire_step, place_step],
        steps=[
            SimpleNamespace(
                step_id=acquire_step,
                requirement_instance_ids=[acquire_instance],
            ),
            SimpleNamespace(
                step_id=place_step,
                requirement_instance_ids=[place_instance],
            ),
        ],
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
        cold_start_plan=_cold_plan(),
        trace=_trace(),
    )
    assert result.passed
    assert cleaned is not None
    assert cleaned.candidate_progress_spans == [{
        "step_id": "acquire", "event_start": 0, "event_end": 1,
        "effect_witness_refs": ["effect:w1"],
    }]


def test_f1_allows_no_progress_span_and_uses_authoritative_remaining_suffix() -> None:
    alignment = _alignment()
    alignment.candidate_progress_spans = []
    cleaned, result = FailurePlanAlignmentValidator().validate(
        alignment,
        cold_start_plan=_cold_plan(),
        trace=_trace(),
    )

    assert result.passed
    assert result.checks["remaining_requirement_instances_authoritative"]
    assert cleaned is not None
    assert cleaned.candidate_progress_spans == []
    assert cleaned.remaining_requirement_instance_ids == ["single::place"]


def test_f1_rejects_model_authored_remaining_suffix_mismatch() -> None:
    alignment = _alignment()
    alignment.remaining_requirement_instance_ids = ["single::acquire"]
    cleaned, result = FailurePlanAlignmentValidator().validate(
        alignment,
        cold_start_plan=_cold_plan(),
        trace=_trace(),
    )

    assert cleaned is None
    assert not result.passed
    assert not result.checks["remaining_requirement_instances_authoritative"]
    assert result.failure_codes == ["failure_extractor_alignment_invalid"]


def test_failure_experience_is_admitted_without_provisional_atomic() -> None:
    proposal = _proposal()
    proposal.provisional_atomics = []

    def unexpected_replay(_item):
        raise AssertionError("an empty provisional list must not trigger replay")

    validation = FailureAssetValidator().validate(
        proposal,
        alignment=_alignment(),
        trace=_trace(),
        source_replay=unexpected_replay,
    )

    assert validation.result.passed
    assert validation.failure_experience_accepted
    assert validation.proposal.provisional_atomics == []
    assert validation.provisional_rejections == []


def test_valid_provisional_survives_invalid_failure_summary() -> None:
    proposal = _proposal()
    proposal.negative_method_suffix = {
        "summary": "avoid source object apple_1",
    }
    validation = FailureAssetValidator().validate(
        proposal,
        alignment=_alignment(),
        trace=_trace(),
        source_replay=lambda _item: {"passed": True},
    )

    assert validation.result.passed
    assert not validation.failure_experience_accepted
    assert len(validation.proposal.provisional_atomics) == 1
    assert validation.provisional_rejections == []


def test_invalid_failure_summary_and_no_provisional_produce_no_assets() -> None:
    proposal = _proposal()
    proposal.provisional_atomics = []
    proposal.reusable_failure_summary = {}
    validation = FailureAssetValidator().validate(
        proposal,
        alignment=_alignment(),
        trace=_trace(),
        source_replay=lambda _item: {"passed": True},
    )

    assert not validation.result.passed
    assert not validation.failure_experience_accepted
    staged = PreparedFailureExtraction(
        _alignment(), None, validation, [], None, {
            "code": "failure_extractor_atomic_invalid",
        },
    )
    assert not staged.accepted


def test_rejected_provisional_does_not_discard_valid_failure_experience() -> None:
    validation = FailureAssetValidator().validate(
        _proposal(),
        alignment=_alignment(),
        trace=_trace(),
        source_replay=lambda _item: {"passed": False},
    )
    assert validation.result.passed
    assert validation.failure_experience_accepted
    assert validation.proposal.provisional_atomics == []
    assert validation.provisional_rejections == [{
        "phase_id": "acquire",
        "code": "provisional_source_replay_failed",
    }]

    class Extractor:
        def align(self, **_kwargs):
            return _alignment()

        def extract(self, **_kwargs):
            return _proposal()

    class RecordBuilder:
        def build(self, *_args):
            return [], SimpleNamespace(experience_id="failure-exp")

    prepared = FailureExtractionCoordinator().prepare(
        eligibility=FailureExtractionEligibility(
            True, True, False, False, "online",
        ),
        extractor=Extractor(),
        task_contract={},
        requirement_expansion={},
        cold_start_plan=_cold_plan(),
        trace=_trace(),
        task_progress=[],
        failures=[],
        candidate_contracts=[],
        source_replay=lambda _item: {"passed": False},
        record_builder=RecordBuilder(),
    )
    assert prepared.accepted
    assert prepared.rejection == {}

    class Store:
        writes = 0

        def upsert_provisional(self, _record):
            self.writes += 1

        def upsert_failure_experience(self, _record):
            self.writes += 1
            return SimpleNamespace(experience_id=_record.experience_id)

    store = Store()
    assert prepared.commit(store) == ([], ["failure-exp"])
    assert store.writes == 1

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


def _renamed_failure_record(
    *,
    task_id: str,
    acquire_id: str,
    place_id: str,
    block_id: str,
    step_prefix: str,
    roles: tuple[str, str, str, str],
):
    object_role, source_role, held_role, location_role = roles
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "object.at_location",
            {"object": "$object", "location": "$location"},
            cardinality=2,
            distinct_by="object",
        )],
        cardinality_constraints=[{
            "constraint_id": "cc_delivery",
            "predicate": "object.at_location",
            "count": 2,
            "distinct_by": "object",
            "shared_roles": ["location"],
            "composition_mode": "repeat_unit",
        }],
        source=ContractSource.ADAPTER_DERIVED,
        confidence=1.0,
        validator_id="fake",
    )
    acquire = CapabilityRequirement(
        acquire_id,
        "free text must not enter the shape",
        [SemanticPredicate(
            "agent.holds", {"object": f"${object_role}"},
        )],
        [
            ParameterSpec(object_role, "entity"),
            ParameterSpec(source_role, "entity"),
        ],
        [ParameterSpec(held_role, "entity")],
        [],
        ["incidental variant"],
        True,
        "incidental rationale",
    )
    place = CapabilityRequirement(
        place_id,
        "another free-text label",
        [SemanticPredicate(
            "object.at_location",
            {
                "object": f"${held_role}",
                "location": f"${location_role}",
            },
        )],
        [
            ParameterSpec(held_role, "entity"),
            ParameterSpec(location_role, "entity"),
        ],
        [],
        [],
        [],
        True,
        "another rationale",
    )
    block = RepeatBlock(
        block_id=block_id,
        count=2,
        ordered_requirement_ids=(acquire_id, place_id),
        distinct_roles=(object_role,),
        shared_roles=(location_role,),
        basis_constraint_id="cc_delivery",
        basis_role_map={
            "object": object_role,
            "location": location_role,
        },
    )
    instances = tuple(
        RequirementInstance(
            f"{block_id}::{repeat_index}::{requirement.requirement_id}",
            requirement.requirement_id,
            block_id,
            repeat_index,
            requirement,
        )
        for repeat_index in range(2)
        for requirement in (acquire, place)
    )
    expansion = RequirementExpansion(
        templates=(acquire, place),
        repeat_blocks=(block,),
        instances=instances,
        instance_ids_by_template={
            acquire_id: (instances[0].instance_id, instances[2].instance_id),
            place_id: (instances[1].instance_id, instances[3].instance_id),
        },
    )
    step_ids = [f"{step_prefix}_{index}" for index in range(4)]
    plan = SimpleNamespace(
        control_sequence=step_ids,
        steps=[
            SimpleNamespace(
                step_id=step_id,
                requirement_instance_ids=[instance.instance_id],
            )
            for step_id, instance in zip(step_ids, instances)
        ],
    )
    alignment = FailurePlanAlignment(
        alignment_id=f"alignment_{step_prefix}",
        step_alignments=[PlanStepAlignment(
            step_ids[0], "achieved", 0, 1, ["effect:w1"], "effect passed",
        )],
        matched_prefix_step_ids=[step_ids[0]],
        first_unrecovered_divergence={
            "kind": "unsatisfied_precondition",
            "step_id": step_ids[1],
            "event_index": 1,
            "summary": "the next semantic requirement remains unresolved",
        },
        remaining_requirement_instance_ids=[
            item.instance_id for item in instances[1:]
        ],
        candidate_progress_spans=[],
    )
    proposal = FailureExtractionProposal(
        provisional_atomics=[],
        validated_plan_prefix=[step_ids[0]],
        negative_method_suffix={
            "summary": "avoid a method that omits a required precondition",
            "avoid_pattern_codes": ["unsatisfied_precondition"],
        },
        reusable_failure_summary={
            "summary": "a required semantic precondition remained unresolved",
        },
    )
    validation = FailureAssetValidator().validate(
        proposal,
        alignment=alignment,
        trace=_trace(),
        source_replay=lambda _item: {"passed": True},
    )
    trace = TraceRecord.create(
        TaskRecord(task_id, "fake", "deliver objects", "place", "signature"),
        to_primitive(contract),
        {},
        {"source": "cold_start"},
    )
    records, experience = FailureAssetRecordBuilder(
        SimpleNamespace(),
        task_contract=contract,
        requirement_expansion=expansion,
        cold_start_plan=plan,
        trace=trace,
        harness_profile="fake_v3",
    ).build(alignment, validation)
    assert records == []
    assert experience is not None
    return contract, expansion, experience


def test_semantically_equivalent_renamed_failures_share_identity_and_confirm(
    tmp_path: Path,
) -> None:
    contract_a, expansion_a, experience_a = _renamed_failure_record(
        task_id="task-a",
        acquire_id="req_acquire_one",
        place_id="req_place_one",
        block_id="repeat_delivery",
        step_prefix="source_step",
        roles=("object", "source", "held_object", "location"),
    )
    contract_b, expansion_b, experience_b = _renamed_failure_record(
        task_id="task-b",
        acquire_id="acquire_target",
        place_id="place_target",
        block_id="delivery_loop",
        step_prefix="renamed_step",
        roles=("item", "origin", "carried_item", "destination"),
    )

    assert experience_a.divergence_signature == experience_b.divergence_signature
    assert experience_a.experience_id == experience_b.experience_id
    assert (
        experience_a.metadata["remaining_requirement_shape_ids"]
        == experience_b.metadata["remaining_requirement_shape_ids"]
    )

    data_dir = tmp_path / "data_v3"
    with StateDatabase(data_dir / "state.sqlite3") as database:
        store = FailureKnowledgeStore(
            data_dir,
            database,
            experience_confirm_independent_tasks=2,
        )
        observed = store.upsert_failure_experience(experience_a)
        assert observed.status is FailureExperienceStatus.OBSERVED
        confirmed = store.upsert_failure_experience(experience_b)
        assert confirmed.experience_id == observed.experience_id
        assert confirmed.status is FailureExperienceStatus.CONFIRMED

        views = FailureExperienceRetriever(store).retrieve(
            contract_b,
            expansion_b,
            harness_profile="fake_v3",
        )
        assert len(views) == 1
        assert views[0].status == "confirmed"
        encoded = json.dumps(to_primitive(views[0]), sort_keys=True)
        for source_identifier in (
            "req_acquire_one",
            "req_place_one",
            "repeat_delivery",
            "source_step",
            "acquire_target",
            "place_target",
            "delivery_loop",
            "renamed_step",
        ):
            assert source_identifier not in encoded
        assert views[0].remaining_requirement_shape_ids


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
