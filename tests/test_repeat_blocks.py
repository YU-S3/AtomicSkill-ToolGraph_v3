from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    CompositeOccurrence,
    CompositeSkill,
    ParameterSpec,
    PlannerRequirementBundle,
    PlannerWorkflowProposal,
    ProposedOccurrence,
    RepeatBlock,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeRepeatConstraint
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.governance.lifecycle import CandidateUsePolicy
from atomic_skillgraph.planner.compiler import PlanCompiler
from atomic_skillgraph.planner.composite_retriever import CompositeRetriever
from atomic_skillgraph.planner.multiplicity import (
    RequirementBundleValidator,
    RequirementMultiplicityCompiler,
)
from atomic_skillgraph.planner.validator import PlannerValidator
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore


def _input(role: str) -> BindingExpression:
    return BindingExpression(
        BindingExprKind.SKILL_INPUT,
        source_role=role,
    )


def _constant(value: str) -> BindingExpression:
    return BindingExpression(
        BindingExprKind.CONSTANT,
        constant=value,
    )


def _atomic(
    logical_id: str,
    *,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...] = (),
    preconditions: tuple[SemanticPredicate, ...] = (),
    effects: tuple[SemanticPredicate, ...],
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=logical_id,
        inputs=[ParameterSpec(role, "entity") for role in inputs],
        outputs=[ParameterSpec(role, "entity") for role in outputs],
        preconditions=list(preconditions),
        effects=list(effects),
        validator_spec={"validator_id": "repeat_test"},
        failure_modes=[],
        guideline={},
        metadata={"harness_profiles": ["repeat_test"]},
        status=SkillStatus.ACTIVE,
    )


class _Skills:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self._atomics = {str(item.ref): item for item in atomics}

    def get_atomic(self, ref: SkillRef) -> AbstractAtomicSkill:
        return self._atomics[str(ref)]

    def implementations_for(
        self, _ref: SkillRef, *, mode: RuntimeMode | str,
    ) -> list[Any]:
        RuntimeMode(mode)
        return []


class _Graph:
    @staticmethod
    def existing_edge_by_id(
        _edge_id: str, *, mode: RuntimeMode | str,
    ) -> None:
        RuntimeMode(mode)
        return None


def _delivery_fixture() -> tuple[
    TaskContract,
    PlannerRequirementBundle,
    AbstractAtomicSkill,
    AbstractAtomicSkill,
]:
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "object.at_location",
            {"object": "target", "location": "destination"},
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
    )
    acquire_requirement = CapabilityRequirement(
        requirement_id="req_acquire",
        intent="acquire one object",
        desired_effects=[SemanticPredicate(
            "agent.holds", {"object": "$object"},
        )],
        expected_inputs=[ParameterSpec("object", "entity")],
        expected_outputs=[],
        precondition_hints=[],
        semantic_variants=[],
        required=True,
        rationale="one reusable acquisition",
    )
    place_requirement = CapabilityRequirement(
        requirement_id="req_place",
        intent="place one object",
        desired_effects=[SemanticPredicate(
            "object.at_location",
            {"object": "$object", "location": "$destination"},
        )],
        expected_inputs=[
            ParameterSpec("object", "entity"),
            ParameterSpec("destination", "entity"),
        ],
        expected_outputs=[],
        precondition_hints=[SemanticPredicate(
            "agent.holds", {"object": "$object"},
        )],
        semantic_variants=[],
        required=True,
        rationale="one reusable placement",
    )
    bundle = PlannerRequirementBundle(
        requirements=[acquire_requirement, place_requirement],
        repeat_blocks=[RepeatBlock(
            block_id="repeat_delivery",
            count=2,
            ordered_requirement_ids=("req_acquire", "req_place"),
            distinct_roles=("object",),
            shared_roles=("destination",),
            basis_constraint_id="cc_delivery",
            basis_role_map={
                "object": "object",
                "location": "destination",
            },
            execution_policy="serial",
        )],
    )
    acquire = _atomic(
        "acquire_one",
        inputs=("object",),
        effects=(SemanticPredicate(
            "agent.holds", {"object": _input("object")},
        ),),
    )
    place = _atomic(
        "place_one",
        inputs=("object", "destination"),
        preconditions=(SemanticPredicate(
            "agent.holds", {"object": _input("object")},
        ),),
        effects=(SemanticPredicate(
            "object.at_location",
            {
                "object": _input("object"),
                "location": _input("destination"),
            },
        ),),
    )
    return contract, bundle, acquire, place


def _proposal(
    acquire: AbstractAtomicSkill,
    place: AbstractAtomicSkill,
) -> PlannerWorkflowProposal:
    rows = [
        (
            "acquire_0", "occ_acquire_0", acquire.ref,
            "repeat_delivery::0::req_acquire",
            {"object": _constant("apple_1")},
            {"object": "object"},
        ),
        (
            "place_0", "occ_place_0", place.ref,
            "repeat_delivery::0::req_place",
            {
                "object": _constant("apple_1"),
                "destination": _constant("countertop_1"),
            },
            {"object": "object", "destination": "destination"},
        ),
        (
            "acquire_1", "occ_acquire_1", acquire.ref,
            "repeat_delivery::1::req_acquire",
            {"object": _constant("apple_2")},
            {"object": "object"},
        ),
        (
            "place_1", "occ_place_1", place.ref,
            "repeat_delivery::1::req_place",
            {
                "object": _constant("apple_2"),
                "destination": _constant("countertop_1"),
            },
            {"object": "object", "destination": "destination"},
        ),
    ]
    steps = [
        ProposedOccurrence(
            step_id=step_id,
            occurrence_id=occurrence_id,
            node_ref=node_ref,
            requirement_ids=[instance_id],
            binding_specs=bindings,
            requirement_instance_ids=[instance_id],
            repeat_role_bindings=role_bindings,
        )
        for (
            step_id, occurrence_id, node_ref, instance_id,
            bindings, role_bindings,
        ) in rows
    ]
    return PlannerWorkflowProposal(
        steps=steps,
        control_sequence=[item.step_id for item in steps],
        data_edges=[],
        dependency_edges=[],
        requirement_coverage={
            item.requirement_instance_ids[0]: [item.step_id]
            for item in steps
        },
    )


def test_single_and_multi_node_repeat_blocks_expand_stably() -> None:
    contract, bundle, _acquire, _place = _delivery_fixture()
    validation = RequirementBundleValidator().validate(
        bundle,
        contract,
        max_repeat_count=4,
        max_runtime_occurrences=16,
    )
    assert validation.passed is True
    expansion = RequirementMultiplicityCompiler().expand(bundle, contract)
    assert [item.instance_id for item in expansion.instances] == [
        "repeat_delivery::0::req_acquire",
        "repeat_delivery::0::req_place",
        "repeat_delivery::1::req_acquire",
        "repeat_delivery::1::req_place",
    ]

    transform = bundle.requirements[1]
    single_bundle = PlannerRequirementBundle(
        requirements=[replace(
            transform,
            requirement_id="req_transform",
            intent="transform one object",
        )],
        repeat_blocks=[replace(
            bundle.repeat_blocks[0],
            block_id="repeat_transform",
            ordered_requirement_ids=("req_transform",),
        )],
    )
    single = RequirementMultiplicityCompiler().expand(single_bundle)
    assert [item.instance_id for item in single.instances] == [
        "repeat_transform::0::req_transform",
        "repeat_transform::1::req_transform",
    ]


def test_compiler_and_validator_accept_same_atomic_ref_in_serial_instances() -> None:
    contract, bundle, acquire, place = _delivery_fixture()
    expansion = RequirementMultiplicityCompiler().expand(bundle, contract)
    skills = _Skills(acquire, place)
    proposal = _proposal(acquire, place)
    plan = PlanCompiler(skills).compile(
        proposal,
        SimpleNamespace(task_id="repeat_task"),
        contract,
        mode=RuntimeMode.ONLINE,
        audit={},
        expansion=expansion,
    )

    assert [str(item.node_ref) for item in plan.occurrences] == [
        str(acquire.ref), str(place.ref), str(acquire.ref), str(place.ref),
    ]
    assert plan.repeat_constraints == [RuntimeRepeatConstraint(
        block_id="repeat_delivery",
        count=2,
        iteration_steps=(("acquire_0", "place_0"), ("acquire_1", "place_1")),
        distinct_roles=("object",),
        shared_roles=("destination",),
        step_role_bindings={
            "acquire_0": {"object": "object"},
            "place_0": {"object": "object", "destination": "destination"},
            "acquire_1": {"object": "object"},
            "place_1": {"object": "object", "destination": "destination"},
        },
    )]
    instance_candidates = {
        instance.instance_id: {
            str(
                acquire.ref
                if instance.template_requirement_id == "req_acquire"
                else place.ref
            )
        }
        for instance in expansion.instances
    }
    report = PlannerValidator(
        skills, _Graph(), max_occurrences=16,
    ).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
        expansion=expansion,
        instance_candidates=instance_candidates,
    )
    assert report.passed is True, report
    assert report.checks["repeat_requirement_instances_exactly_once"] is True
    assert report.checks["repeat_serial_order"] is True
    assert report.checks["runtime_repeat_constraints_match"] is True


def test_validator_rejects_duplicate_occurrence_instance_and_bad_order() -> None:
    contract, bundle, acquire, place = _delivery_fixture()
    expansion = RequirementMultiplicityCompiler().expand(bundle, contract)
    skills = _Skills(acquire, place)
    proposal = _proposal(acquire, place)
    proposal.steps[2].occurrence_id = proposal.steps[0].occurrence_id
    proposal.steps[2].requirement_instance_ids = [
        "repeat_delivery::0::req_acquire"
    ]
    proposal.steps[2].requirement_ids = [
        "repeat_delivery::0::req_acquire"
    ]
    proposal.control_sequence = [
        "acquire_0", "place_0", "place_1", "acquire_1",
    ]
    proposal.requirement_coverage = {
        item.requirement_instance_ids[0]: [item.step_id]
        for item in proposal.steps
    }
    plan = PlanCompiler(skills).compile(
        proposal,
        SimpleNamespace(task_id="bad_repeat_task"),
        contract,
        mode=RuntimeMode.ONLINE,
        audit={},
        expansion=expansion,
    )
    report = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
        expansion=expansion,
        instance_candidates={
            instance.instance_id: {str(acquire.ref), str(place.ref)}
            for instance in expansion.instances
        },
    )
    assert report.passed is False
    assert report.checks["occurrence_ids_unique"] is False
    assert report.checks["repeat_requirement_instances_exactly_once"] is False
    assert report.checks["repeat_serial_order"] is False
    assert "planner_requirement_instance_uncovered" in report.failure_codes


def test_validator_rejects_unknown_candidate_and_repeat_role() -> None:
    contract, bundle, acquire, place = _delivery_fixture()
    expansion = RequirementMultiplicityCompiler().expand(bundle, contract)
    skills = _Skills(acquire, place)
    proposal = _proposal(acquire, place)
    proposal.steps[0].repeat_role_bindings = {"object": "invented_role"}
    plan = PlanCompiler(skills).compile(
        proposal,
        SimpleNamespace(task_id="bad_authority_task"),
        contract,
        mode=RuntimeMode.ONLINE,
        audit={},
        expansion=expansion,
    )
    candidates = {
        instance.instance_id: {
            str(
                acquire.ref
                if instance.template_requirement_id == "req_acquire"
                else place.ref
            )
        }
        for instance in expansion.instances
    }
    candidates["repeat_delivery::0::req_acquire"] = {str(place.ref)}
    report = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
        expansion=expansion,
        instance_candidates=candidates,
    )
    assert report.passed is False
    assert report.checks["requirement_instance_candidate_authority"] is False
    assert report.checks["repeat_role_bindings_valid"] is False
    assert "planner_requirement_instance_uncovered" in report.failure_codes
    assert "planner_repeat_role_invalid" in report.failure_codes


def _runtime_repeat_constraint() -> RuntimeRepeatConstraint:
    return RuntimeRepeatConstraint(
        block_id="repeat_delivery",
        count=2,
        iteration_steps=(("place_0",), ("place_1",)),
        distinct_roles=("object",),
        shared_roles=("destination",),
        step_role_bindings={
            "place_0": {"object": "item", "destination": "target"},
            "place_1": {"object": "item", "destination": "target"},
        },
    )


def test_repeat_binding_state_rejects_reused_distinct_and_changed_shared() -> None:
    store = RuntimeBindingStore()
    store.configure_repeat_constraints([_runtime_repeat_constraint()])
    first = store.commit_repeat_bindings(
        "place_0",
        {"item": "apple_1", "target": "countertop_1"},
        effect_passed=True,
    )
    assert first.passed is True

    reused = store.preflight_repeat_bindings(
        "place_1",
        {"item": "apple_1", "target": "countertop_1"},
    )
    assert reused.passed is False
    assert reused.failure_codes == [
        "runtime_repetition_distinctness_violation"
    ]

    changed_shared = store.preflight_repeat_bindings(
        "place_1",
        {"item": "apple_2", "target": "drawer_1"},
    )
    assert changed_shared.passed is False
    assert changed_shared.failure_codes == [
        "runtime_repetition_shared_value_violation"
    ]

    valid = store.commit_repeat_bindings(
        "place_1",
        {"item": "apple_2", "target": "countertop_1"},
        effect_passed=True,
    )
    assert valid.passed is True
    assert store.repeat_state.committed_distinct_values == {
        "repeat_delivery::object": {0: "apple_1", 1: "apple_2"},
    }
    assert store.repeat_state.committed_shared_values == {
        "repeat_delivery::destination": "countertop_1",
    }


def test_failed_effect_never_commits_repeat_values() -> None:
    store = RuntimeBindingStore()
    store.configure_repeat_constraints([_runtime_repeat_constraint()])
    skipped = store.commit_repeat_bindings(
        "place_0",
        {"item": "apple_1", "target": "countertop_1"},
        effect_passed=False,
    )
    assert skipped.passed is True
    assert store.repeat_state.committed_distinct_values == {}
    assert store.repeat_state.committed_shared_values == {}

    # Because the failed attempt left no state, the same concrete identity in
    # a later effect-success iteration is not falsely treated as a reuse.
    later = store.commit_repeat_bindings(
        "place_1",
        {"item": "apple_1", "target": "countertop_1"},
        effect_passed=True,
    )
    assert later.passed is True


class _CompositeSkills(_Skills):
    def __init__(
        self,
        atomic: AbstractAtomicSkill,
        composites: list[CompositeSkill],
    ) -> None:
        super().__init__(atomic)
        self._composites = composites

    def composites(
        self, *, mode: RuntimeMode | str,
    ) -> list[CompositeSkill]:
        selected_mode = RuntimeMode(mode)
        return [
            item for item in self._composites
            if item.status is SkillStatus.ACTIVE
            or selected_mode is RuntimeMode.ONLINE
        ]


def _complete_composite(
    logical_id: str,
    atomic: AbstractAtomicSkill,
    *,
    cardinality: int,
    status: SkillStatus,
    summary: str,
) -> CompositeSkill:
    constraint = ([{
        "constraint_id": "cc_delivery",
        "predicate": "object.at_location",
        "count": cardinality,
        "distinct_by": "object",
        "shared_roles": ["location"],
        "composition_mode": "repeat_unit",
    }] if cardinality > 1 else [])
    return CompositeSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=summary,
        occurrences=[CompositeOccurrence(
            "place", f"occ_{logical_id}", atomic.ref, {},
        )],
        control_sequence=["place"],
        data_edges=[],
        dependency_edges=[],
        goal_contract=TaskContract(
            target_effects=[SemanticPredicate(
                "object.at_location",
                {"object": "object", "location": "location"},
                cardinality=cardinality,
                distinct_by="object" if cardinality > 1 else "",
            )],
            cardinality_constraints=constraint,
        ),
        guideline={},
        insight={},
        validator_spec={},
        metadata={"harness_profiles": ["repeat_test"]},
        status=status,
    )


def test_p0_complete_match_is_exact_and_candidate_bootstrap_is_top_one() -> None:
    atomic = _atomic(
        "place_exact",
        inputs=("object", "location"),
        effects=(SemanticPredicate(
            "object.at_location",
            {"object": _input("object"), "location": _input("location")},
        ),),
    )
    aggregate = _complete_composite(
        "aggregate_two", atomic,
        cardinality=2,
        status=SkillStatus.ACTIVE,
        summary="place two objects",
    )
    unit_top = _complete_composite(
        "unit_top", atomic,
        cardinality=1,
        status=SkillStatus.CANDIDATE,
        summary="place object",
    )
    unit_other = _complete_composite(
        "unit_other", atomic,
        cardinality=1,
        status=SkillStatus.CANDIDATE,
        summary="unrelated capability",
    )
    retriever = CompositeRetriever(
        _CompositeSkills(atomic, [aggregate, unit_other, unit_top]),
        top_k=5,
        candidate_policy=CandidateUsePolicy(exploration_quota=0.0),
    )
    unit_contract = TaskContract(target_effects=[SemanticPredicate(
        "object.at_location",
        {"object": "object", "location": "location"},
    )])
    result = retriever.retrieve_complete(
        SimpleNamespace(task_id="p0_exact", goal="place object"),
        unit_contract,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert [str(item.ref) for item in result.candidates] == [str(unit_top.ref)]
    assert any(
        item["composite_ref"] == str(aggregate.ref)
        and "goal_contract_exact_mismatch" in item["reasons"]
        for item in result.rejections
    )
    assert any(
        item["composite_ref"] == str(unit_other.ref)
        and "candidate_bootstrap_not_top1" in item["reasons"]
        for item in result.rejections
    )
