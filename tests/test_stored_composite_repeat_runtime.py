from __future__ import annotations

from dataclasses import replace
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
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.planner.compiler import PlanCompiler
from atomic_skillgraph.planner.pipeline import PlannerPipeline
from atomic_skillgraph.planner.validator import (
    PlannerValidator,
    _project_plan_effects,
)
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore


def _input(role: str) -> BindingExpression:
    return BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)


def _repeat_contract() -> TaskContract:
    return TaskContract(
        target_effects=[SemanticPredicate(
            "P",
            {"x": "target", "y": "destination"},
            cardinality=2,
            distinct_by="x",
        )],
        cardinality_constraints=[{
            "constraint_id": "cc_repeat_p",
            "predicate": "P",
            "count": 2,
            "distinct_by": "x",
            "shared_roles": ["y"],
            "composition_mode": "repeat_unit",
        }],
    )


def _atomic(
    logical_id: str,
    *,
    cardinality: int = 1,
    distinct_by: str = "",
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=logical_id,
        inputs=[
            ParameterSpec("item", "entity"),
            ParameterSpec("destination", "entity"),
        ],
        outputs=[],
        preconditions=[],
        effects=[SemanticPredicate(
            "P",
            {"x": _input("item"), "y": _input("destination")},
            cardinality=cardinality,
            distinct_by=distinct_by,
        )],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={"harness_profiles": ["repeat_test"]},
        status=SkillStatus.ACTIVE,
    )


def _composite(
    logical_id: str,
    atomic: AbstractAtomicSkill,
    step_ids: tuple[str, ...],
    contract: TaskContract,
    *,
    summary: str = "repeat P",
) -> CompositeSkill:
    return CompositeSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=summary,
        occurrences=[CompositeOccurrence(
            step_id=step_id,
            occurrence_id=f"occ_{step_id}",
            node_ref=atomic.ref,
            binding_specs={
                "item": _input("x"),
                "destination": _input("y"),
            },
        ) for step_id in step_ids],
        control_sequence=list(step_ids),
        data_edges=[],
        dependency_edges=[],
        goal_contract=contract,
        guideline={},
        insight={},
        validator_spec={},
        metadata={"harness_profiles": ["repeat_test"]},
        status=SkillStatus.ACTIVE,
    )


class _Skills:
    def __init__(
        self,
        atomic: AbstractAtomicSkill,
        composites: list[CompositeSkill] | None = None,
    ) -> None:
        self.atomic = atomic
        self._composites = list(composites or [])

    def get_atomic(self, ref: SkillRef) -> AbstractAtomicSkill:
        if str(ref) != str(self.atomic.ref):
            raise KeyError(ref)
        return self.atomic

    @staticmethod
    def implementations_for(
        _ref: SkillRef, *, mode: RuntimeMode | str,
    ) -> list[Any]:
        RuntimeMode(mode)
        return []

    def composites(
        self, *, mode: RuntimeMode | str,
    ) -> list[CompositeSkill]:
        RuntimeMode(mode)
        return list(self._composites)


class _Graph:
    @staticmethod
    def existing_edge_by_id(
        _edge_id: str, *, mode: RuntimeMode | str,
    ) -> None:
        RuntimeMode(mode)
        return None


def _compile(
    composite: CompositeSkill,
    contract: TaskContract,
    skills: _Skills,
) -> Any:
    return PlanCompiler(skills).from_composite(
        SimpleNamespace(task_id="stored_repeat"),
        contract,
        composite,
        mode=RuntimeMode.ONLINE,
        audit={},
    )


def test_stored_composite_compiles_formal_repeat_authority() -> None:
    contract = _repeat_contract()
    atomic = _atomic("unit_p")
    composite = _composite(
        "repeat_p", atomic, ("A0", "A1"), contract,
    )
    skills = _Skills(atomic, [composite])

    plan = _compile(composite, contract, skills)

    assert len(plan.repeat_constraints) == 1
    repeat = plan.repeat_constraints[0]
    assert repeat.block_id == "stored::cc_repeat_p"
    assert repeat.basis_constraint_id == "cc_repeat_p"
    assert repeat.count == 2
    assert repeat.iteration_steps == (("A0",), ("A1",))
    assert repeat.distinct_roles == ("x",)
    assert repeat.shared_roles == ("y",)
    assert repeat.step_role_bindings == {
        "A0": {"x": "item", "y": "destination"},
        "A1": {"x": "item", "y": "destination"},
    }

    report = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert report.passed is True, report
    assert report.checks["runtime_repeat_basis_authority"] is True
    offered, _ = _project_plan_effects(
        plan, {item.step_id: atomic for item in plan.occurrences},
    )
    symbols = {step_id: arguments for _effect, step_id, arguments in offered}
    assert symbols == {
        "A0": {
            "x": "repeat:cc_repeat_p:x:0",
            "y": "repeat:cc_repeat_p:y:shared",
        },
        "A1": {
            "x": "repeat:cc_repeat_p:x:1",
            "y": "repeat:cc_repeat_p:y:shared",
        },
    }


def test_stored_repeat_runtime_enforces_distinct_and_shared_values() -> None:
    contract = _repeat_contract()
    atomic = _atomic("runtime_unit_p")
    composite = _composite(
        "runtime_repeat_p", atomic, ("A0", "A1"), contract,
    )
    repeat = _compile(composite, contract, _Skills(atomic)).repeat_constraints
    store = RuntimeBindingStore()
    store.configure_repeat_constraints(repeat)
    assert store.commit_repeat_bindings(
        "A0",
        {"item": "value_1", "destination": "dest_1"},
        effect_passed=True,
    ).passed is True
    assert store.preflight_repeat_bindings(
        "A1",
        {"item": "value_1", "destination": "dest_1"},
    ).failure_codes == ["runtime_repetition_distinctness_violation"]
    assert store.preflight_repeat_bindings(
        "A1",
        {"item": "value_2", "destination": "dest_2"},
    ).failure_codes == ["runtime_repetition_shared_value_violation"]


def test_runtime_repeat_roles_must_name_formal_distinct_and_shared_roles() -> None:
    contract = _repeat_contract()
    atomic = _atomic("formal_roles_unit_p")
    composite = _composite(
        "formal_roles_repeat_p", atomic, ("A0", "A1"), contract,
    )
    skills = _Skills(atomic)
    plan = _compile(composite, contract, skills)
    repeat = plan.repeat_constraints[0]

    plan.repeat_constraints = [replace(
        repeat,
        distinct_roles=("renamed_x",),
        step_role_bindings={
            step_id: {
                "renamed_x": bindings["x"],
                "y": bindings["y"],
            }
            for step_id, bindings in repeat.step_role_bindings.items()
        },
    )]
    renamed_distinct = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert renamed_distinct.passed is False
    assert renamed_distinct.checks[
        "runtime_repeat_formal_roles_declared"
    ] is False

    plan.repeat_constraints = [replace(
        repeat,
        shared_roles=("renamed_y",),
        step_role_bindings={
            step_id: {
                "x": bindings["x"],
                "renamed_y": bindings["y"],
            }
            for step_id, bindings in repeat.step_role_bindings.items()
        },
    )]
    renamed_shared = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert renamed_shared.passed is False
    assert renamed_shared.checks[
        "runtime_repeat_formal_roles_declared"
    ] is False


def test_unprovable_stored_repeat_is_typed_validation_rejection() -> None:
    contract = _repeat_contract()
    unit = _atomic("short_unit_p")
    short = _composite("short_repeat", unit, ("A0",), contract)
    skills = _Skills(unit, [short])
    short_plan = _compile(short, contract, skills)
    assert short_plan.repeat_constraints == []
    short_report = PlannerValidator(skills, _Graph()).validate(
        short_plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert short_report.passed is False
    assert "planner_repeat_block_invalid" in short_report.failure_codes

    aggregate = _atomic(
        "aggregate_p", cardinality=2, distinct_by="x",
    )
    aggregate_composite = _composite(
        "aggregate_repeat", aggregate, ("aggregate",), contract,
    )
    aggregate_skills = _Skills(aggregate, [aggregate_composite])
    aggregate_plan = _compile(
        aggregate_composite, contract, aggregate_skills,
    )
    assert aggregate_plan.repeat_constraints == []
    aggregate_report = PlannerValidator(
        aggregate_skills, _Graph(),
    ).validate(
        aggregate_plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert aggregate_report.passed is False
    assert "planner_repeat_block_invalid" in (
        aggregate_report.failure_codes
    )


def test_nonrepeat_stored_composite_behavior_is_unchanged() -> None:
    contract = TaskContract(target_effects=[SemanticPredicate(
        "P", {"x": "target", "y": "destination"},
    )])
    atomic = _atomic("single_p")
    composite = _composite("single", atomic, ("A0",), contract)
    skills = _Skills(atomic, [composite])
    plan = _compile(composite, contract, skills)
    assert plan.repeat_constraints == []
    report = PlannerValidator(skills, _Graph()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="repeat_test",
    )
    assert report.passed is True, report


def test_p0_repeat_rejection_is_audited_and_next_candidate_is_tried() -> None:
    contract = _repeat_contract()
    atomic = _atomic("pipeline_unit_p")
    invalid = _composite(
        "invalid_first", atomic, ("bad_only",), contract,
        summary="invalid first",
    )
    valid = _composite(
        "valid_second", atomic, ("A0", "A1"), contract,
        summary="valid second",
    )
    skills = _Skills(atomic, [invalid, valid])
    pipeline = PlannerPipeline(
        skills, _Graph(), lambda *_args: None,
    )
    harness = SimpleNamespace(
        profile_name="repeat_test",
        task_contract=lambda _task: contract,
    )

    plan = pipeline.build_plan(
        SimpleNamespace(task_id="p0_repeat", goal="invalid first"),
        harness,
        mode=RuntimeMode.ONLINE,
    )

    assert plan.source == "stored_composite"
    assert plan.source_composite_ref == str(valid.ref)
    rejection = next(
        item for item in plan.planner_audit["composite_rejections"]
        if item["composite_ref"] == str(invalid.ref)
    )
    assert rejection["stage"] == "plan_validation"
    assert "planner_repeat_block_invalid" in rejection["reasons"]
    assert rejection["checks"]["runtime_repeat_basis_authority"] is False
