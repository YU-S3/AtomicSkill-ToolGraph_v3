from __future__ import annotations

import json
from types import SimpleNamespace

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    IdentityConstraint,
    IdentityRelation,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
)
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.plan_context import RuntimePlanContextBuilder
from atomic_skillgraph.runtime.task_progress import TaskProgressTracker


class _AtomicRegistry:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self._atomics = {atomic.ref: atomic for atomic in atomics}

    def get_atomic(self, ref: SkillRef) -> AbstractAtomicSkill:
        return self._atomics[ref]


class _ValidatorChannel:
    def __init__(self, facts: list[dict], revision: int = 1) -> None:
        self.facts = facts
        self.revision = revision

    def snapshot(self) -> dict:
        return {"revision": self.revision, "facts": list(self.facts)}


def _atomic(
    logical_id: str,
    summary: str,
    *,
    inputs: list[ParameterSpec],
    outputs: list[ParameterSpec],
    preconditions: list[SemanticPredicate] | None = None,
    effects: list[SemanticPredicate] | None = None,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=summary,
        inputs=inputs,
        outputs=outputs,
        preconditions=list(preconditions or []),
        effects=list(effects or []),
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )


def _occurrence(
    step_id: str,
    atomic: AbstractAtomicSkill,
    bindings: dict[str, BindingExpression] | None = None,
) -> RuntimeOccurrence:
    return RuntimeOccurrence(
        step_id=step_id,
        occurrence_id=f"occ::{step_id}",
        node_ref=atomic.ref,
        requirement_ids=[],
        binding_specs=dict(bindings or {}),
        implementation_candidates=[],
        expected_effects=list(atomic.effects),
    )


def _plan(
    occurrences: list[RuntimeOccurrence],
    data_edges: list[GraphEdge],
) -> RuntimeLinearPlan:
    return RuntimeLinearPlan(
        task_id="policy-context-fixture",
        source="stored_composite",
        source_composite_ref="skill://method@1.0.0",
        occurrences=occurrences,
        control_sequence=[item.step_id for item in occurrences],
        data_edges=data_edges,
        dependency_edges=[],
        task_contract=TaskContract(),
        planner_audit={},
    )


def test_nav_to_take_context_explains_consumer_without_source_guess() -> None:
    navigate = _atomic(
        "navigate",
        "navigate to a location",
        inputs=[ParameterSpec("destination", "location", runtime_resolvable=True)],
        outputs=[ParameterSpec("destination", "location")],
        effects=[SemanticPredicate("agent.at_location", {
            "destination": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="destination",
            ),
        })],
    )
    take = _atomic(
        "take",
        "take an object from a location",
        inputs=[
            ParameterSpec("object", "object"),
            ParameterSpec("source", "location", runtime_resolvable=True),
        ],
        outputs=[ParameterSpec("object", "object")],
        preconditions=[SemanticPredicate("object.at_location", {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="object",
            ),
            "location": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="source",
            ),
        })],
        effects=[SemanticPredicate("agent.holds", {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="object",
            ),
        })],
    )
    navigate_occurrence = _occurrence("navigate-step", navigate)
    take_occurrence = _occurrence("take-step", take, {
        "object": BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role="task_object",
        ),
        "source": BindingExpression(
            BindingExprKind.DATA_FLOW,
            source_step="navigate-step",
            source_role="destination",
        ),
    })
    edge = GraphEdge(
        "edge-nav-take",
        GraphEdgeType.DATA_FLOW,
        "navigate-step",
        "take-step",
        "destination",
        "source",
        "extractor_validated",
    )
    plan = _plan([navigate_occurrence, take_occurrence], [edge])
    bindings = RuntimeBindingStore()
    bindings.seed_task_bindings(
        SimpleNamespace(
            task_id="fixture",
            context={"semantic_bindings": {"task_object": "target_object"}},
        ),
        TaskContract(),
        0,
    )
    bindings.propose_agent_arguments(
        take_occurrence,
        {"source": "unvalidated_source_guess"},
        0,
    )

    context = RuntimePlanContextBuilder(
        _AtomicRegistry(navigate, take),
    ).build(plan, "navigate-step", bindings).policy_view()

    assert context["current_step"] == "navigate-step"
    assert context["remaining_method_outline"] == [
        {"step_id": "take-step", "summary": "take an object from a location"},
    ]
    assert len(context["output_obligations"]) == 1
    obligation = context["output_obligations"][0]
    assert obligation["producer_output_role"] == "destination"
    assert obligation["consumer_input_role"] == "source"
    assert obligation["consumer_known_semantic_anchors"] == {
        "object": {
            "value": "target_object",
            "semantic_type": "entity",
            "source": "task",
        },
    }
    serialized = json.dumps(context, sort_keys=True)
    assert "source_location_guess" not in serialized
    assert "unvalidated_source_guess" not in serialized
    assert "task_type" not in serialized
    assert "tool_body" not in serialized


def test_plan_context_does_not_invent_obligation_without_dataflow() -> None:
    first = _atomic(
        "first",
        "first portable step",
        inputs=[],
        outputs=[ParameterSpec("result", "entity")],
    )
    second = _atomic(
        "second",
        "second portable step",
        inputs=[ParameterSpec("input", "entity")],
        outputs=[],
    )
    plan = _plan([_occurrence("first", first), _occurrence("second", second)], [])

    context = RuntimePlanContextBuilder(
        _AtomicRegistry(first, second),
    ).build(plan, "first", RuntimeBindingStore())

    assert context.output_obligations == ()
    assert context.remaining_method_outline == (
        {"step_id": "second", "summary": "second portable step"},
    )


def test_remaining_outline_exposes_method_without_benchmark_defaults() -> None:
    navigate = _atomic(
        "outline_navigate",
        "navigate to a selected location",
        inputs=[ParameterSpec("destination", "location", runtime_resolvable=True)],
        outputs=[ParameterSpec("destination", "location")],
    )
    open_container = _atomic(
        "outline_open",
        "open a selected container",
        inputs=[ParameterSpec("container", "container")],
        outputs=[ParameterSpec("container", "container")],
    )
    transform = _atomic(
        "outline_transform",
        "transform an object with a selected device",
        inputs=[ParameterSpec("device", "device")],
        outputs=[],
    )
    occurrences = [
        _occurrence("navigate", navigate),
        _occurrence("open", open_container),
        _occurrence("transform", transform),
    ]
    edges = [
        GraphEdge(
            "navigate-open",
            GraphEdgeType.DATA_FLOW,
            "navigate",
            "open",
            "destination",
            "container",
        ),
        GraphEdge(
            "open-transform",
            GraphEdgeType.DATA_FLOW,
            "open",
            "transform",
            "container",
            "device",
        ),
    ]
    context = RuntimePlanContextBuilder(
        _AtomicRegistry(navigate, open_container, transform),
    ).build(
        _plan(occurrences, edges),
        "navigate",
        RuntimeBindingStore(),
    ).policy_view()

    assert context["remaining_method_outline"] == [
        {"step_id": "open", "summary": "open a selected container"},
        {
            "step_id": "transform",
            "summary": "transform an object with a selected device",
        },
    ]
    serialized = json.dumps(context, sort_keys=True).casefold()
    for forbidden_default in ("fridge", "microwave", "sinkbasin"):
        assert forbidden_default not in serialized


def test_plan_context_ignores_unverified_atomic_contracts() -> None:
    verified = _atomic(
        "verified",
        "verified step",
        inputs=[],
        outputs=[ParameterSpec("result", "entity")],
    )
    draft = _atomic(
        "draft",
        "draft private guess",
        inputs=[ParameterSpec("input", "entity")],
        outputs=[],
    )
    draft.status = SkillStatus.DRAFT
    edge = GraphEdge(
        "unverified-edge",
        GraphEdgeType.DATA_FLOW,
        "verified",
        "draft",
        "result",
        "input",
    )
    plan = _plan([_occurrence("verified", verified), _occurrence("draft", draft)], [edge])

    context = RuntimePlanContextBuilder(
        _AtomicRegistry(verified, draft),
    ).build(plan, "verified", RuntimeBindingStore())

    assert context.output_obligations == ()
    assert context.remaining_method_outline == ()


def test_task_progress_policy_view_reports_counts_without_witness_identity() -> None:
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "object.at_location",
            {
                "object": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="object",
                ),
                "location": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="location",
                ),
            },
            cardinality=2,
            distinct_by="object",
        )],
        cardinality_constraints=[{
            "constraint_id": "place-two",
            "predicate": "object.at_location",
            "count": 2,
            "distinct_by": "object",
            "shared_roles": ["location"],
        }],
        identity_constraints=[IdentityConstraint(
            "object",
            IdentityRelation.SAME_AS,
            "location",
            "task",
        )],
    )
    validator = _ValidatorChannel([])
    tracker = TaskProgressTracker(contract, validator)

    before = tracker.policy_view()
    assert before["targets"][0]["satisfied_count"] == 0
    assert before["targets"][0]["remaining_count"] == 2
    validator.facts.append({
        "predicate": "object.at_location",
        "args": {
            "object": "private_object_4",
            "location": "private_target_1",
        },
    })
    policy = tracker.policy_view()

    assert policy == {
        "targets": [{
            "constraint_id": "place-two",
            "predicate": "object.at_location",
            "required_count": 2,
            "satisfied_count": 1,
            "remaining_count": 1,
            "distinct_by": "object",
        }],
        "unsatisfied_identity_constraint_count": 1,
    }
    serialized = json.dumps(policy, sort_keys=True)
    assert "private_object_4" not in serialized
    assert "private_target_1" not in serialized
    assert "facts" not in serialized
    assert "witness" not in serialized
    assert "progress_digest" not in serialized


def test_task_progress_policy_view_can_reuse_an_audited_snapshot() -> None:
    contract = TaskContract(target_effects=[SemanticPredicate("goal.done", {})])
    validator = _ValidatorChannel([])
    tracker = TaskProgressTracker(contract, validator)
    snapshot = tracker.snapshot()
    validator.facts.append({"predicate": "goal.done", "args": {}})

    policy = tracker.policy_view(snapshot)

    assert policy["targets"][0]["satisfied_count"] == 0
    assert policy["targets"][0]["remaining_count"] == 1
