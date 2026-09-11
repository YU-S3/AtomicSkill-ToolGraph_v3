"""R9 gates for Stored Repeat executable identity closure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.planner.compiler import PlanCompiler
from atomic_skillgraph.planner.repeat_constraints import (
    repeat_enforcement_input_role,
)
from atomic_skillgraph.planner.validator import PlannerValidator
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore


def _input(role: str) -> BindingExpression:
    return BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)


def _flow(step: str, role: str) -> BindingExpression:
    return BindingExpression(
        BindingExprKind.DATA_FLOW,
        source_step=step,
        source_role=role,
    )


def _contract() -> TaskContract:
    return TaskContract(
        target_effects=[SemanticPredicate(
            "object.at_location",
            {"object": "$object", "location": "$location"},
            cardinality=2,
            distinct_by="object",
        )],
        cardinality_constraints=[{
            "constraint_id": "repeat_two_placements",
            "predicate": "object.at_location",
            "count": 2,
            "distinct_by": "object",
            "shared_roles": ["location"],
            "composition_mode": "repeat_unit",
        }],
    )


def _take(
    name: str = "take",
    *,
    derivation_kind: str = "input_identity",
) -> AbstractAtomicSkill:
    derivation = (
        {"kind": "input_identity", "input_role": "object"}
        if derivation_kind == "input_identity"
        else {
            "kind": "effect_witness",
            "predicate": "agent.holds",
            "argument_role": "object",
        }
    )
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"),
        summary=name,
        inputs=[ParameterSpec(
            "object", "entity", runtime_resolvable=True,
        )],
        outputs=[ParameterSpec("held_object", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds", {"object": _input("held_object")},
        )],
        validator_spec={
            "output_derivations": {"held_object": derivation},
        },
        failure_modes=[],
        guideline={},
        metadata={"harness_profiles": ["r9_repeat"]},
        status=SkillStatus.ACTIVE,
    )


def _place(
    name: str = "place",
    *,
    object_derivation: str = "input_identity",
    direct_effect_inputs: bool = False,
) -> AbstractAtomicSkill:
    object_rule = (
        {"kind": "input_identity", "input_role": "object"}
        if object_derivation == "input_identity"
        else {
            "kind": "effect_witness",
            "predicate": "object.at_location",
            "argument_role": "object",
        }
    )
    effect_args = (
        {"object": _input("object"), "location": _input("destination")}
        if direct_effect_inputs
        else {
            "object": _input("placed_object"),
            "location": _input("placed_location"),
        }
    )
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"),
        summary=name,
        inputs=[
            ParameterSpec("object", "entity", runtime_resolvable=True),
            ParameterSpec("destination", "entity", runtime_resolvable=True),
        ],
        outputs=[
            ParameterSpec("placed_object", "entity"),
            ParameterSpec("placed_location", "entity"),
        ],
        preconditions=[SemanticPredicate(
            "agent.holds", {"object": _input("object")},
        )],
        effects=[SemanticPredicate("object.at_location", effect_args)],
        validator_spec={
            "output_derivations": {
                "placed_object": object_rule,
                "placed_location": {
                    "kind": "input_identity",
                    "input_role": "destination",
                },
            },
        },
        failure_modes=[],
        guideline={},
        metadata={"harness_profiles": ["r9_repeat"]},
        status=SkillStatus.ACTIVE,
    )


class _Skills:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self.by_ref = {str(item.ref): item for item in atomics}

    def get_atomic(self, ref: SkillRef | str) -> AbstractAtomicSkill:
        return self.by_ref[str(ref)]

    @staticmethod
    def implementations_for(
        _ref: SkillRef, *, mode: RuntimeMode | str,
    ) -> list[Any]:
        RuntimeMode(mode)
        return []


class _Graph:
    @staticmethod
    def existing_edge_by_id(*_args: Any, **_kwargs: Any) -> None:
        return None


def _composite(
    take: AbstractAtomicSkill,
    place: AbstractAtomicSkill,
    *,
    second_object_source: str = "take1",
) -> CompositeSkill:
    contract = _contract()
    occurrences = [
        CompositeOccurrence("take0", "occ_take0", take.ref, {}),
        CompositeOccurrence(
            "place0",
            "occ_place0",
            place.ref,
            {"object": _flow("take0", "held_object")},
        ),
        CompositeOccurrence("take1", "occ_take1", take.ref, {}),
        CompositeOccurrence(
            "place1",
            "occ_place1",
            place.ref,
            {
                "object": _flow(second_object_source, "held_object"),
                "destination": _flow("place0", "placed_location"),
            },
        ),
    ]
    edges = [
        GraphEdge(
            "take0_place0",
            GraphEdgeType.DATA_FLOW,
            "take0",
            "place0",
            "held_object",
            "object",
            "extractor_validated",
        ),
        GraphEdge(
            "take1_place1",
            GraphEdgeType.DATA_FLOW,
            second_object_source,
            "place1",
            "held_object",
            "object",
            "extractor_validated",
        ),
        GraphEdge(
            "shared_location",
            GraphEdgeType.DATA_FLOW,
            "place0",
            "place1",
            "placed_location",
            "destination",
            "extractor_validated",
        ),
    ]
    return CompositeSkill(
        ref=SkillRef("two_object_composite", "1.0.0"),
        summary="take/place/take/place",
        occurrences=occurrences,
        control_sequence=["take0", "place0", "take1", "place1"],
        data_edges=edges,
        dependency_edges=[],
        goal_contract=contract,
        guideline={},
        insight={},
        validator_spec={},
        metadata={"harness_profiles": ["r9_repeat"]},
        status=SkillStatus.ACTIVE,
    )


def _compile(
    composite: CompositeSkill,
    skills: _Skills,
):
    return PlanCompiler(skills).from_composite(
        SimpleNamespace(task_id="r9_repeat"),
        _contract(),
        composite,
        mode=RuntimeMode.ONLINE,
        audit={},
    )


def test_effect_outputs_normalize_to_executable_input_roles() -> None:
    take = _take()
    place = _place()
    skills = _Skills(take, place)
    plan = _compile(_composite(take, place), skills)

    assert repeat_enforcement_input_role(place, "placed_object") == "object"
    assert repeat_enforcement_input_role(
        place, "placed_location",
    ) == "destination"
    assert plan.repeat_constraints[0].iteration_steps == (
        ("take0", "place0"),
        ("take1", "place1"),
    )
    assert plan.repeat_constraints[0].step_role_bindings == {
        "take0": {"object": "object"},
        "place0": {"object": "object", "location": "destination"},
        "take1": {"object": "object"},
        "place1": {"object": "object", "location": "destination"},
    }
    report = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_repeat",
    )
    assert report.passed is True, report


def test_second_take_duplicate_is_rejected_before_environment_action() -> None:
    take = _take()
    place = _place()
    plan = _compile(_composite(take, place), _Skills(take, place))
    store = RuntimeBindingStore()
    store.configure_repeat_constraints(plan.repeat_constraints)
    assert store.commit_repeat_bindings(
        "take0", {"object": "object_1"}, effect_passed=True,
    ).passed

    environment_actions: list[str] = []
    preflight = store.preflight_repeat_bindings(
        "take1", {"object": "object_1"},
    )
    if preflight.passed:
        environment_actions.append("TAKE object_1")

    assert preflight.failure_codes == [
        "runtime_repetition_distinctness_violation",
    ]
    assert environment_actions == []


def test_placement_only_runtime_take_is_guarded_by_placement_input() -> None:
    place = _place("placement_only")
    contract = _contract()
    composite = CompositeSkill(
        ref=SkillRef("placement_only_repeat", "1.0.0"),
        summary="two runtime-grounded placements",
        occurrences=[
            CompositeOccurrence("place0", "occ_place0", place.ref, {}),
            CompositeOccurrence(
                "place1",
                "occ_place1",
                place.ref,
                {"destination": _flow("place0", "placed_location")},
            ),
        ],
        control_sequence=["place0", "place1"],
        data_edges=[GraphEdge(
            "shared_location",
            GraphEdgeType.DATA_FLOW,
            "place0",
            "place1",
            "placed_location",
            "destination",
            "extractor_validated",
        )],
        dependency_edges=[],
        goal_contract=contract,
        guideline={},
        insight={},
        validator_spec={},
        metadata={"harness_profiles": ["r9_repeat"]},
        status=SkillStatus.ACTIVE,
    )
    plan = _compile(composite, _Skills(place))
    assert plan.repeat_constraints[0].iteration_steps == (
        ("place0",), ("place1",),
    )
    store = RuntimeBindingStore()
    store.configure_repeat_constraints(plan.repeat_constraints)
    assert store.commit_repeat_bindings(
        "place0",
        {"object": "object_1", "destination": "table_1"},
        effect_passed=True,
    ).passed
    assert store.preflight_repeat_bindings(
        "place1",
        {"object": "object_1", "destination": "table_1"},
    ).failure_codes == ["runtime_repetition_distinctness_violation"]


def test_output_effect_witness_cannot_supply_repeat_input_identity() -> None:
    take = _take()
    place = _place(object_derivation="effect_witness")
    skills = _Skills(take, place)
    compiler = PlanCompiler(skills)
    plan = compiler.from_composite(
        SimpleNamespace(task_id="r9_repeat"),
        _contract(),
        _composite(take, place),
        mode=RuntimeMode.ONLINE,
        audit={},
    )

    assert repeat_enforcement_input_role(place, "placed_object") == ""
    assert plan.repeat_constraints == []
    assert (
        compiler.repeat_compiler
        .last_stored_identity_closure_compile_count
    ) == 0
    assert (
        compiler.repeat_compiler
        .last_stored_identity_closure_rejection_count
    ) == 1


def test_identity_producer_from_another_distinct_iteration_fails_closed() -> None:
    take = _take()
    place = _place()
    composite = _composite(
        take,
        place,
        second_object_source="take0",
    )
    # Remove the now-unused second TAKE to isolate the illegal cross-iteration
    # distinct identity source while preserving forward control order.
    composite.occurrences = [
        item for item in composite.occurrences if item.step_id != "take1"
    ]
    composite.control_sequence.remove("take1")
    plan = _compile(composite, _Skills(take, place))

    assert plan.repeat_constraints == []


def test_upstream_effect_witness_producer_fails_identity_closure() -> None:
    take = _take(derivation_kind="effect_witness")
    place = _place()
    plan = _compile(_composite(take, place), _Skills(take, place))

    assert plan.repeat_constraints == []


def test_direct_effect_input_roles_continue_to_compile() -> None:
    take = _take()
    place = _place(direct_effect_inputs=True)
    plan = _compile(_composite(take, place), _Skills(take, place))

    assert plan.repeat_constraints[0].step_role_bindings["place1"] == {
        "object": "object",
        "location": "destination",
    }
