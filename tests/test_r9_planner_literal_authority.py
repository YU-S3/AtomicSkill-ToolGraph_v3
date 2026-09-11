"""R9 gates for role-scoped Planner CONSTANT authority."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    PlannerWorkflowProposal,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
    ValidationResult,
)
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.planner.pipeline import PlannerPipeline
from atomic_skillgraph.planner.validator import (
    PlannerValidator,
    planner_constant_authority_rejections,
)
from atomic_skillgraph.planner.workflow_agent import WorkflowAgent
from atomic_skillgraph.system import AtomicSkillGraphSystem


def _atomic(role: str, semantic_type: str = "entity") -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(f"literal_{role}", "1.0.0"),
        summary=f"literal {role}",
        inputs=[ParameterSpec(
            role,
            semantic_type,
            runtime_resolvable=False,
        )],
        outputs=[],
        preconditions=[],
        effects=[],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={"harness_profiles": ["r9_literal"]},
        status=SkillStatus.ACTIVE,
    )


class _Skills:
    def __init__(self, atomic: AbstractAtomicSkill) -> None:
        self.atomic = atomic

    def get_atomic(self, ref: SkillRef | str) -> AbstractAtomicSkill:
        if str(ref) != str(self.atomic.ref):
            raise KeyError(ref)
        return self.atomic


class _Graph:
    @staticmethod
    def existing_edge_by_id(*_args: Any, **_kwargs: Any) -> None:
        return None


def _plan(
    atomic: AbstractAtomicSkill,
    expression: BindingExpression,
) -> RuntimeLinearPlan:
    role = atomic.inputs[0].name
    return RuntimeLinearPlan(
        task_id="r9_literal",
        source="atomic_composition",
        source_composite_ref=None,
        occurrences=[RuntimeOccurrence(
            "step",
            "occ_step",
            atomic.ref,
            [],
            {role: expression},
            [],
            [],
        )],
        control_sequence=["step"],
        data_edges=[],
        dependency_edges=[],
        task_contract=TaskContract(),
        planner_audit={},
    )


def _constant(value: Any) -> BindingExpression:
    return BindingExpression(BindingExprKind.CONSTANT, constant=value)


def test_empty_literal_authority_rejects_episode_entity_constants() -> None:
    for role, value in (
        ("station", "microwave 1"),
        ("light", "light 1"),
        ("destination", "countertop 1"),
    ):
        atomic = _atomic(role)
        plan = _plan(atomic, _constant(value))
        result = PlannerValidator(
            _Skills(atomic), _Graph(),
        ).validate(
            plan,
            mode=RuntimeMode.ONLINE,
            harness_profile="r9_literal",
            literal_authorities={},
        )

        assert result.passed is False
        assert result.failure_codes == [
            "planner_constant_authority_invalid",
        ]
        assert result.checks["planner_constant_authority_valid"] is False
        assert planner_constant_authority_rejections(plan, {}) == [{
            "step_id": "step",
            "target_role": role,
            "constant": value,
        }]


def test_exact_code_owned_scalar_literal_is_authorized_per_role() -> None:
    atomic = _atomic("threshold", "number")
    plan = _plan(atomic, _constant(0.5))
    validator = PlannerValidator(_Skills(atomic), _Graph())

    allowed = validator.validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_literal",
        literal_authorities={"threshold": [0.5]},
    )
    wrong_role = validator.validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_literal",
        literal_authorities={"mode": [0.5]},
    )

    assert allowed.passed is True, allowed
    assert wrong_role.passed is False
    assert "planner_constant_authority_invalid" in wrong_role.failure_codes


def test_exact_literal_authority_preserves_value_type() -> None:
    atomic = _atomic("threshold", "number")
    plan = _plan(atomic, _constant(1))

    result = PlannerValidator(
        _Skills(atomic), _Graph(),
    ).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_literal",
        literal_authorities={"threshold": [True]},
    )

    assert result.passed is False
    assert "planner_constant_authority_invalid" in result.failure_codes


def test_task_entity_value_does_not_authorize_constant() -> None:
    atomic = _atomic("object")
    result = PlannerValidator(
        _Skills(atomic), _Graph(),
    ).validate(
        _plan(atomic, _constant("apple_1")),
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_literal",
        task_binding_roles={"object"},
        literal_authorities={},
    )

    assert result.passed is False
    assert "planner_constant_authority_invalid" in result.failure_codes


def test_adapter_transform_authority_is_unchanged() -> None:
    atomic = _atomic("normalized_object")
    expression = BindingExpression(
        BindingExprKind.ADAPTER_TRANSFORM,
        source_role="object",
        transform_id="normalize_entity",
    )

    result = PlannerValidator(
        _Skills(atomic), _Graph(),
    ).validate(
        _plan(atomic, expression),
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_literal",
        literal_authorities={},
    )

    assert result.passed is True, result


class _PromptSession:
    def __init__(self) -> None:
        self.bucket = ""

    def set_usage_bucket(self, value: str) -> None:
        self.bucket = value


def _capture_prompt(agent: WorkflowAgent) -> dict[str, str]:
    captured: dict[str, str] = {}

    def capture(prompt: str, **_kwargs: Any) -> str:
        captured["prompt"] = prompt
        return "captured"

    agent._request_workflow = capture  # type: ignore[method-assign]
    return captured


def _line_payload(prompt: str, label: str) -> Any:
    line = next(item for item in prompt.splitlines() if item.startswith(label))
    return json.loads(line[len(label):])


def test_p2_and_p2r_prompts_display_literal_authorities() -> None:
    session = _PromptSession()
    agent = WorkflowAgent(session)
    captured = _capture_prompt(agent)
    expansion = SimpleNamespace(
        templates=[],
        repeat_blocks=[],
        instances=[],
        instance_ids_by_template={},
    )

    assert agent.propose(
        SimpleNamespace(goal="use an authorized threshold"),
        TaskContract(),
        expansion,
        {},
        [],
        [],
        literal_authorities={"threshold": [0.5]},
    ) == "captured"
    assert _line_payload(
        captured["prompt"], "Allowed literal authorities: ",
    ) == {"threshold": [0.5]}

    assert agent.repair(
        PlannerWorkflowProposal([], [], [], [], {}),
        ValidationResult.fail(
            "planner", "planner_constant_authority_invalid", "invalid",
        ),
        [],
        [],
        literal_authorities={},
    ) == "captured"
    assert _line_payload(
        captured["prompt"], "Allowed literal authorities: ",
    ) == {}
    assert "do not emit CONSTANT for any Atomic input" in captured["prompt"]


def test_formal_pipeline_default_literal_authority_is_empty() -> None:
    pipeline = PlannerPipeline(
        SimpleNamespace(),
        SimpleNamespace(),
        lambda *_args: None,
    )

    assert pipeline.literal_authorities == {}


def test_system_wires_code_owned_literal_authorities_into_planner(
    tmp_path,
) -> None:
    config = {
        "schema_version": 3,
        "data_dir": str(tmp_path / "data_v3"),
        "experiment": {
            "runtime_mode": "online",
            "freeze_skills": False,
            "output_dir": str(tmp_path / "runs"),
        },
        "planner": {
            "literal_authorities": {
                "threshold": [0.5],
                "mode": ["strict"],
            },
        },
    }

    with AtomicSkillGraphSystem(
        config,
        harness=SimpleNamespace(profile_name="r9_literal"),
    ) as system:
        assert system.planner.literal_authorities == {
            "threshold": (0.5,),
            "mode": ("strict",),
        }
