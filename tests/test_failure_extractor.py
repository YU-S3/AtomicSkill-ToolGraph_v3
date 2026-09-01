from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from atomic_skillgraph.agents import LLMUsage, UsageBucket, UsageEvent, UsageLedger
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
    FailureExtractorBudgetUnavailable,
    FailureExtractorSession,
    FailureExtractorSessionAllocation,
    FailurePlanAlignment,
    PlanStepAlignment,
)
from atomic_skillgraph.evolution.failure_extraction_view import (
    FailureAlignmentView,
    FailureAssetExtractionView,
)
from atomic_skillgraph.core.contracts import (
    CapabilityRequirement,
    ContractSource,
    ParameterSpec,
    RepeatBlock,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.errors import BudgetExhausted, FailureLayer
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
from atomic_skillgraph.system import AtomicSkillGraphSystem
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
        self.sessions: list[object] = []

    def request(self, session, **_kwargs):
        self.sessions.append(session)
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


def test_f1_and_f2_use_fresh_sessions() -> None:
    f1 = _BucketSession()
    f2 = _BucketSession()
    extractor = FailureExtractorSession(
        f1,
        lambda: FailureExtractorSessionAllocation(f2, 1000),
    )
    extractor.submissions = _SubmissionQueue([
        _alignment_payload(), _extraction_payload(),
    ])

    alignment = extractor.align(
        alignment_view={},
    )
    proposal = extractor.extract(
        asset_view={},
    )

    assert extractor.submissions.sessions == [f1, f2]
    assert extractor.f1_session_id == ""
    assert extractor.f2_session_id == ""
    assert proposal.provisional_atomics[0].progress_relation == "consumed_prerequisite"
    with pytest.raises(RuntimeError, match="F1 may run exactly once"):
        extractor.align(alignment_view={})
    with pytest.raises(RuntimeError, match="may run exactly once"):
        extractor.extract(asset_view={})


def test_system_allocates_f2_only_from_authoritative_f1_usage() -> None:
    system = object.__new__(AtomicSkillGraphSystem)
    system.config = {
        "llm": {
            "extractor": {
                "max_completion_tokens": 131072,
                "max_total_tokens_per_task": 262144,
            },
        },
    }
    system.usage = UsageLedger()
    created: list[dict] = []

    def new_session(**kwargs):
        created.append(dict(kwargs))
        return SimpleNamespace(session_id=f"fresh-{len(created)}")

    system._new_session = new_session
    f1 = system._failure_extractor_f1_session("task")
    assert f1.session_id == "fresh-1"
    assert created[-1]["bucket"] is UsageBucket.FAILURE_EXTRACTOR_F1
    assert created[-1]["max_tokens"] == 262144
    assert created[-1]["semantic_max_turns"] == 1

    for index, total in enumerate((100000, 20000)):
        system.usage.append(UsageEvent(
            event_id=f"usage-{index}",
            session_id=f1.session_id,
            turn_index=index,
            bucket=UsageBucket.FAILURE_EXTRACTOR_F1,
            usage=LLMUsage(
                prompt_tokens=total - 1000,
                completion_tokens=1000,
                total_tokens=total,
                reasoning_tokens=500,
                call_count=1,
                latency_ms=1.0,
            ),
        ))

    allocation = system._failure_extractor_f2_allocation(
        "task", f1.session_id,
    )
    assert allocation.remaining_tokens == 142144
    assert allocation.session is not f1
    assert created[-1]["bucket"] is UsageBucket.FAILURE_EXTRACTOR_F2
    assert created[-1]["max_tokens"] == 142144
    assert created[-1]["semantic_max_turns"] == 1

    exhausted = object.__new__(AtomicSkillGraphSystem)
    exhausted.config = system.config
    exhausted.usage = UsageLedger()
    exhausted.usage.append(UsageEvent(
        event_id="usage-exhausted",
        session_id="f1-exhausted",
        turn_index=0,
        bucket=UsageBucket.FAILURE_EXTRACTOR_F1,
        usage=LLMUsage(
            prompt_tokens=200000,
            completion_tokens=62144,
            total_tokens=262144,
            reasoning_tokens=60000,
            call_count=1,
            latency_ms=1.0,
        ),
    ))
    exhausted._new_session = lambda **_kwargs: pytest.fail(
        "F2 provider session must not be created with zero remaining budget"
    )
    no_budget = exhausted._failure_extractor_f2_allocation(
        "task", "f1-exhausted",
    )
    assert no_budget == FailureExtractorSessionAllocation(None, 0)


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


def test_f1_deduplicates_overlapping_candidate_spans_before_f2() -> None:
    trace = _trace()
    trace.environment_actions = [
        SimpleNamespace(accepted=True, revision=index, new_revision=index + 1)
        for index in range(100)
    ]
    alignment = _alignment()
    span = {
        "step_id": "acquire", "event_start": 10, "event_end": 13,
        "effect_witness_refs": ["effect:w1"],
    }
    alignment.candidate_progress_spans = [
        span,
        dict(span),
        {
            "step_id": "place", "event_start": 12, "event_end": 15,
            "effect_witness_refs": ["effect:w1"],
        },
    ]

    cleaned, result = FailurePlanAlignmentValidator().validate(
        alignment,
        cold_start_plan=_cold_plan(),
        trace=trace,
    )

    assert result.passed
    assert result.messages == ["candidate_progress_span_rejected:2"]
    assert cleaned is not None
    assert cleaned.candidate_progress_spans == [span]
    assert sum(
        int(item["event_end"]) - int(item["event_start"])
        for item in cleaned.candidate_progress_spans
    ) == 3


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


def _coordinator_prepare(extractor, *, record_builder=None):
    class DefaultRecordBuilder:
        def build(self, *_args):
            return [], SimpleNamespace(experience_id="failure-exp")

    return FailureExtractionCoordinator().prepare(
        eligibility=FailureExtractionEligibility(
            True, True, False, False, "online",
        ),
        extractor=extractor,
        task_contract={},
        requirement_expansion={},
        cold_start_plan=_cold_plan(),
        trace=_trace(),
        task_progress=[],
        failures=[],
        candidate_contracts=[],
        source_replay=lambda _item: {"passed": False},
        record_builder=record_builder or DefaultRecordBuilder(),
    )


def test_coordinator_passes_compact_f1_and_f2_views_and_keeps_diagnostics() -> None:
    class Extractor:
        def __init__(self) -> None:
            self.seen_alignment_view = None
            self.seen_asset_view = None
            self._diagnostics = {"f1": 1}

        @property
        def diagnostics(self):
            return dict(self._diagnostics)

        def align(self, *, alignment_view):
            self.seen_alignment_view = alignment_view
            return _alignment()

        def extract(self, *, asset_view):
            self.seen_asset_view = asset_view
            self._diagnostics["f2"] = 1
            proposal = _proposal()
            proposal.provisional_atomics = []
            return proposal

    extractor = Extractor()
    prepared = _coordinator_prepare(extractor)

    assert isinstance(extractor.seen_alignment_view, FailureAlignmentView)
    assert isinstance(extractor.seen_asset_view, FailureAssetExtractionView)
    assert prepared.accepted
    assert prepared.diagnostics == {"f1": 1, "f2": 1}


@pytest.mark.parametrize("stage", ["f1", "f2"])
def test_coordinator_isolates_only_extractor_token_budget(stage: str) -> None:
    class Extractor:
        diagnostics = {"prompt_chars": 123}

        def align(self, *, alignment_view):
            assert isinstance(alignment_view, FailureAlignmentView)
            if stage == "f1":
                raise BudgetExhausted(
                    "extractor_token_budget_exhausted",
                    "F1 provider call exceeded its allocation",
                    layer=FailureLayer.RUNTIME_AGENT,
                )
            return _alignment()

        def extract(self, *, asset_view):
            assert isinstance(asset_view, FailureAssetExtractionView)
            raise BudgetExhausted(
                "extractor_token_budget_exhausted",
                "F2 provider call exceeded its allocation",
                layer=FailureLayer.RUNTIME_AGENT,
            )

    prepared = _coordinator_prepare(Extractor())

    assert not prepared.accepted
    assert prepared.rejection == {
        "code": "failure_extractor_budget_exhausted",
        "stage": stage,
        "source_code": "extractor_token_budget_exhausted",
    }
    assert prepared.diagnostics == {
        "prompt_chars": 123,
        "failure_extractor_budget_exhausted_count": 1,
    }
    assert (prepared.alignment is not None) is (stage == "f2")


def test_coordinator_does_not_swallow_other_budget_or_program_errors() -> None:
    class OtherBudget:
        diagnostics = {}

        def align(self, *, alignment_view):
            raise BudgetExhausted(
                "runtime_task_token_budget_exhausted",
                "not an extractor allocation",
                layer=FailureLayer.RUNTIME_AGENT,
            )

    with pytest.raises(BudgetExhausted) as budget:
        _coordinator_prepare(OtherBudget())
    assert budget.value.code == "runtime_task_token_budget_exhausted"

    class ProgrammingError:
        diagnostics = {}

        def align(self, *, alignment_view):
            raise RuntimeError("provider usage persistence corrupted")

    with pytest.raises(RuntimeError, match="usage persistence corrupted"):
        _coordinator_prepare(ProgrammingError())


def test_coordinator_records_f2_not_started_when_no_budget_remains() -> None:
    class Extractor:
        diagnostics = {"failure_extractor_skipped_after_budget_count": 1}

        def align(self, *, alignment_view):
            return _alignment()

        def extract(self, *, asset_view):
            raise FailureExtractorBudgetUnavailable()

    prepared = _coordinator_prepare(Extractor())

    assert prepared.rejection == {
        "code": "failure_extractor_budget_exhausted",
        "stage": "f2_not_started_no_remaining_budget",
        "source_code": "extractor_token_budget_exhausted",
    }
    assert prepared.diagnostics[
        "failure_extractor_skipped_after_budget_count"
    ] == 1


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
