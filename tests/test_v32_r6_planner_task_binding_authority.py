"""R6-A: Planner task-binding role authority is explicit and fail closed."""

from __future__ import annotations

import json
from types import SimpleNamespace

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ContractSource,
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
from atomic_skillgraph.planner.pipeline import (
    PlannerPipeline,
    _task_binding_interface,
)
from atomic_skillgraph.planner.validator import PlannerValidator
from atomic_skillgraph.planner.workflow_agent import WorkflowAgent


TASK_BINDING_INTERFACE = {
    "object": {
        "semantic_value": "semantic_object_family",
        "semantic_type": "entity",
    },
    "destination": {
        "semantic_value": "semantic_destination_family",
        "semantic_type": "entity",
    },
}


class _Skills:
    def __init__(self, atomic: AbstractAtomicSkill) -> None:
        self.atomic = atomic

    def get_atomic(self, ref: SkillRef) -> AbstractAtomicSkill:
        assert ref == self.atomic.ref
        return self.atomic

    def list_refs(self, *_args, **_kwargs) -> list[SkillRef]:
        return []


class _Graph:
    def existing_edge_by_id(self, *_args, **_kwargs):
        return None


class _PromptSession:
    def __init__(self) -> None:
        self.bucket = ""

    def set_usage_bucket(self, value: str) -> None:
        self.bucket = value


def _atomic() -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef("generic_place", "1.0.0"),
        summary="place a generic entity",
        inputs=[ParameterSpec("item", "entity")],
        outputs=[],
        preconditions=[],
        effects=[],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )


def _plan(atomic: AbstractAtomicSkill, source_role: str) -> RuntimeLinearPlan:
    return RuntimeLinearPlan(
        task_id="generic_task",
        source="stored_composite",
        source_composite_ref="skill://generic_composite@1.0.0",
        occurrences=[RuntimeOccurrence(
            step_id="place",
            occurrence_id="place_occurrence",
            node_ref=atomic.ref,
            requirement_ids=[],
            binding_specs={
                "item": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role=source_role,
                ),
            },
            implementation_candidates=[],
            expected_effects=[],
        )],
        control_sequence=["place"],
        data_edges=[],
        dependency_edges=[],
        task_contract=TaskContract(source=ContractSource.ADAPTER_DERIVED),
        planner_audit={},
    )


def _validator(atomic: AbstractAtomicSkill) -> PlannerValidator:
    return PlannerValidator(_Skills(atomic), _Graph())


def _capture_prompt(agent: WorkflowAgent) -> dict[str, str]:
    captured: dict[str, str] = {}

    def capture(prompt: str, **_kwargs):
        captured["prompt"] = prompt
        return "captured"

    agent._request_workflow = capture  # type: ignore[method-assign]
    return captured


def _prompt_payload(prompt: str, label: str) -> dict[str, object]:
    line = next(item for item in prompt.splitlines() if item.startswith(label))
    return json.loads(line[len(label):])


def test_semantic_value_is_not_an_authorized_task_role() -> None:
    atomic = _atomic()
    report = _validator(atomic).validate(
        _plan(atomic, "semantic_object_family"),
        mode=RuntimeMode.ONLINE,
        task_binding_roles=set(TASK_BINDING_INTERFACE),
    )

    assert report.passed is False
    assert report.checks["skill_input_source_roles_authorized"] is False
    assert report.checks["required_inputs_closed"] is False
    assert report.failure_codes == ["planner_task_binding_role_invalid"]


def test_task_role_key_is_authorized() -> None:
    atomic = _atomic()
    report = _validator(atomic).validate(
        _plan(atomic, "object"),
        mode=RuntimeMode.ONLINE,
        task_binding_roles=set(TASK_BINDING_INTERFACE),
    )

    assert report.passed is True, report
    assert report.checks["skill_input_source_roles_authorized"] is True


def test_p2_prompt_carries_task_binding_interface() -> None:
    session = _PromptSession()
    agent = WorkflowAgent(session)
    captured = _capture_prompt(agent)

    result = agent.propose(
        SimpleNamespace(goal="place the requested object"),
        TaskContract(source=ContractSource.ADAPTER_DERIVED),
        SimpleNamespace(
            templates=[], instances=[], repeat_blocks=[],
            instance_ids_by_template={},
        ),
        {},
        [],
        [],
        task_binding_interface=TASK_BINDING_INTERFACE,
    )

    assert result == "captured"
    assert session.bucket == "planner_p2"
    prompt = captured["prompt"]
    assert _prompt_payload(prompt, "Task binding interface: ") == (
        TASK_BINDING_INTERFACE
    )
    assert "semantic_value is a value, never a role name" in prompt


def test_p2r_prompt_carries_same_task_binding_interface() -> None:
    session = _PromptSession()
    agent = WorkflowAgent(session)
    captured = _capture_prompt(agent)

    result = agent.repair(
        PlannerWorkflowProposal([], [], [], [], {}),
        ValidationResult(
            level="planner",
            passed=False,
            checks={"skill_input_source_roles_authorized": False},
            failure_codes=["planner_task_binding_role_invalid"],
        ),
        [],
        [],
        task_binding_interface=TASK_BINDING_INTERFACE,
    )

    assert result == "captured"
    assert session.bucket == "planner_p2_repair"
    prompt = captured["prompt"]
    assert _prompt_payload(prompt, "Task binding interface: ") == (
        TASK_BINDING_INTERFACE
    )
    assert "semantic_value is a value, never a role name" in prompt


def test_p0_rejects_stored_composite_with_unauthorized_task_role() -> None:
    atomic = _atomic()
    invalid_plan = _plan(atomic, "semantic_object_family")
    pipeline = object.__new__(PlannerPipeline)
    pipeline.skills = _Skills(atomic)
    pipeline.validator = _validator(atomic)
    pipeline.compiler = SimpleNamespace(
        from_composite=lambda *_args, **_kwargs: invalid_plan,
    )
    candidate = SimpleNamespace(ref=SkillRef("generic_composite", "1.0.0"))
    pipeline.composite_retriever = SimpleNamespace(
        retrieve_complete=lambda *_args, **_kwargs: SimpleNamespace(
            audit_candidates=[], rejections=[], candidates=[candidate],
        ),
        retrieve_terminal=lambda *_args, **_kwargs: SimpleNamespace(
            terminal_empirical_audit=[], terminal_empirical_candidates=[],
        ),
    )
    pipeline.cold_start_enabled = False
    task = SimpleNamespace(
        task_id="generic_task",
        goal="place the requested object",
        context={
            "semantic_bindings": {
                "object": "semantic_object_family",
                "destination": "semantic_destination_family",
            },
            "binding_types": {
                "object": "entity",
                "destination": "entity",
            },
        },
    )
    harness = SimpleNamespace(
        profile_name="generic",
        task_contract=lambda _task: TaskContract(
            source=ContractSource.ADAPTER_DERIVED
        ),
    )

    assert _task_binding_interface(task) == TASK_BINDING_INTERFACE
    fallback = pipeline.build_plan(task, harness, mode=RuntimeMode.ONLINE)

    assert fallback.source == "full_dynamic"
    rejection = fallback.planner_audit["composite_rejections"][0]
    assert rejection["stage"] == "plan_validation"
    assert rejection["reasons"] == ["planner_task_binding_role_invalid"]
    assert rejection["checks"]["skill_input_source_roles_authorized"] is False
