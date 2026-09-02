from __future__ import annotations

from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    BindingResolution,
    BindingSource,
    BindingStatus,
    EvidenceStability,
    GroundingEvidence,
    RuntimeBinding,
)
from atomic_skillgraph.core.contracts import ParameterSpec, TaskContract
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
    RuntimeRepeatConstraint,
)
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.plan_context import RuntimePlanContextBuilder


def _occurrence(
    step_id: str,
    occurrence_id: str,
    binding_specs: dict[str, BindingExpression] | None = None,
) -> RuntimeOccurrence:
    return RuntimeOccurrence(
        step_id=step_id,
        occurrence_id=occurrence_id,
        node_ref=SkillRef("semantic_anchor_fixture", "1.0.0"),
        requirement_ids=[],
        binding_specs=dict(binding_specs or {}),
        implementation_candidates=[],
        expected_effects=[],
    )


def _harness_binding(role: str, value: str, revision: int) -> RuntimeBinding:
    return RuntimeBinding(
        role=role,
        value=value,
        semantic_type="entity",
        source=BindingSource.HARNESS_EVIDENCE,
        status=BindingStatus.GROUNDED,
        resolution=BindingResolution.CONCRETE,
        evidence_refs=[f"evidence:{revision}"],
        world_revision=revision,
    )


def test_task_anchor_survives_concrete_overwrite_and_revision_invalidation() -> None:
    store = RuntimeBindingStore()
    store.bind_task_value("object", "pen", "entity", 0)
    occurrence = _occurrence(
        "locate",
        "occ_locate",
        {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="object",
            ),
        },
    )
    store.resolve_occurrence_specs(occurrence, 0)

    anchor = store.semantic_anchor_for(occurrence, "object")
    assert anchor is not None
    assert (anchor.value, anchor.source) == ("pen", BindingSource.TASK)

    store.commit_grounded(
        occurrence.occurrence_id,
        {"object": _harness_binding("object", "pen_1", 1)},
    )
    assert store.snapshot_for_node(occurrence)["object"].value == "pen_1"
    assert store.semantic_anchor_for(occurrence, "object").value == "pen"

    store.invalidate_revision(2)
    assert (
        store.snapshot_for_node(occurrence)["object"].status
        is BindingStatus.INVALIDATED
    )
    persistent = store.semantic_anchor_for(occurrence, "object")
    assert persistent is not None
    assert (persistent.value, persistent.source) == (
        "pen",
        BindingSource.TASK,
    )
    assert store.runtime_prompt_projection(
        occurrence,
        [ParameterSpec("object", "entity", required_resolution="concrete")],
    ) == {
        "task_semantic_context": {"object": "pen"},
        "occurrence_semantic_anchors": {"object": "pen"},
        "execution_ready_bindings": {},
        "missing_or_insufficient_bindings": ["object"],
    }


def test_validated_plan_constant_is_a_persistent_plan_anchor() -> None:
    store = RuntimeBindingStore()
    occurrence = _occurrence(
        "place",
        "occ_place",
        {
            "destination": BindingExpression(
                BindingExprKind.CONSTANT,
                constant="countertop_1",
            ),
        },
    )
    store.resolve_occurrence_specs(occurrence, 3)
    anchor = store.semantic_anchor_for(occurrence, "destination")
    assert anchor is not None
    assert (anchor.value, anchor.source) == (
        "countertop_1",
        BindingSource.RUNTIME_PLAN,
    )

    store.commit_grounded(
        occurrence.occurrence_id,
        {"destination": _harness_binding("destination", "countertop_1", 3)},
    )
    store.invalidate_revision(4)
    persistent = store.semantic_anchor_for(occurrence, "destination")
    assert persistent is not None
    assert (persistent.value, persistent.source) == (
        "countertop_1",
        BindingSource.RUNTIME_PLAN,
    )


def test_validated_dataflow_is_anchor_but_tool_output_alone_is_not() -> None:
    source = _occurrence("take", "occ_take")
    target = _occurrence(
        "place",
        "occ_place",
        {
            "object": BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_step="take",
                source_role="held_object",
            ),
        },
    )
    edge = GraphEdge(
        edge_id="take-to-place",
        edge_type=GraphEdgeType.DATA_FLOW,
        source_step="take",
        target_step="place",
        source_role="held_object",
        target_role="object",
        origin="planner_validated",
    )
    plan = RuntimeLinearPlan(
        task_id="dataflow-anchor",
        source="atomic_composition",
        source_composite_ref=None,
        occurrences=[source, target],
        control_sequence=["take", "place"],
        data_edges=[edge],
        dependency_edges=[],
        task_contract=TaskContract(),
        planner_audit={},
    )
    store = RuntimeBindingStore()
    store.apply_data_flow(plan, "take", revision=1)
    store.publish_validated_outputs(
        source,
        {"held_object": "pen_1"},
        ["validator:take-effect"],
        1,
    )
    assert store.semantic_anchor_for(source, "held_object") is None

    store.apply_data_flow(plan, "place", revision=1)
    anchor = store.semantic_anchor_for(target, "object")
    assert anchor is not None
    assert (anchor.value, anchor.source) == (
        "pen_1",
        BindingSource.DATA_FLOW,
    )
    store.invalidate_revision(2)
    assert (
        store.snapshot_for_node(source)["held_object"].status
        is BindingStatus.INVALIDATED
    )
    assert (
        store.snapshot_for_node(target)["object"].status
        is BindingStatus.INVALIDATED
    )
    assert store.semantic_anchor_for(source, "held_object") is None
    assert store.semantic_anchor_for(target, "object").value == "pen_1"

    # Re-resolving the formal edge in a later world may recover semantic
    # identity, but it must not resurrect the stale concrete proof.
    store.resolve_occurrence_specs(target, 2)
    refreshed = store.snapshot_for_node(target)["object"]
    assert refreshed.status is BindingStatus.GROUNDED
    assert refreshed.resolution is BindingResolution.SEMANTIC
    store.commit_grounded(
        target.occurrence_id,
        {"object": _harness_binding("object", "pen_1", 2)},
    )
    store.invalidate_revision(3)
    persistent = store.semantic_anchor_for(target, "object")
    assert persistent is not None
    assert (persistent.value, persistent.source) == (
        "pen_1",
        BindingSource.DATA_FLOW,
    )


def test_harness_and_incidental_tool_bindings_never_become_plan_anchors() -> None:
    store = RuntimeBindingStore()
    occurrence = _occurrence("inspect", "occ_inspect")
    store.commit_grounded(
        occurrence.occurrence_id,
        {"object": _harness_binding("object", "pen_1", 0)},
    )
    store.publish_validated_outputs(
        occurrence,
        {"destination": "drawer_1"},
        ["validator:output"],
        0,
    )

    assert store.semantic_anchor_for(occurrence, "object") is None
    assert store.semantic_anchor_for(occurrence, "destination") is None
    assert RuntimePlanContextBuilder._formal_anchor(
        occurrence,
        "object",
        store,
    ) is None
    assert RuntimePlanContextBuilder._formal_anchor(
        occurrence,
        "destination",
        store,
    ) is None
    assert store.runtime_prompt_projection(occurrence, [])[
        "occurrence_semantic_anchors"
    ] == {}


def test_repeat_commit_projects_stable_distinct_and_shared_anchors() -> None:
    steps = [
        _occurrence(
            step_id,
            f"occ_{step_id}",
            {
                "object": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="object",
                ),
                "destination": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="destination",
                ),
            },
        )
        for step_id in ("take_0", "place_0", "take_1", "place_1")
    ]
    constraint = RuntimeRepeatConstraint(
        block_id="repeat_delivery",
        count=2,
        iteration_steps=(("take_0", "place_0"), ("take_1", "place_1")),
        distinct_roles=("object",),
        shared_roles=("destination",),
        step_role_bindings={
            step_id: {"object": "object", "destination": "destination"}
            for step_id in ("take_0", "place_0", "take_1", "place_1")
        },
    )
    plan = RuntimeLinearPlan(
        task_id="repeat-anchor",
        source="atomic_composition",
        source_composite_ref=None,
        occurrences=steps,
        control_sequence=[item.step_id for item in steps],
        data_edges=[],
        dependency_edges=[],
        task_contract=TaskContract(),
        planner_audit={},
        repeat_constraints=[constraint],
    )
    store = RuntimeBindingStore()
    store.configure_repeat_constraints(plan.repeat_constraints)
    store.apply_data_flow(plan, "take_0", revision=0)
    store.bind_task_value("object", "apple", "entity", 0)
    store.bind_task_value("destination", "countertop", "entity", 0)
    for occurrence in steps:
        store.resolve_occurrence_specs(occurrence, 0)

    assert store.commit_repeat_bindings(
        "take_0",
        {"object": "apple_1", "destination": "countertop_1"},
        effect_passed=True,
    ).passed

    first_iteration = store.semantic_anchor_for("occ_place_0", "object")
    next_iteration = store.semantic_anchor_for("occ_take_1", "object")
    shared = store.semantic_anchor_for("occ_place_1", "destination")
    assert first_iteration is not None
    assert (first_iteration.value, first_iteration.source) == (
        "apple_1",
        BindingSource.REPEAT,
    )
    assert next_iteration is not None
    assert next_iteration.value == "apple"
    assert shared is not None
    assert (shared.value, shared.source) == (
        "countertop_1",
        BindingSource.REPEAT,
    )

    store.commit_grounded(
        "occ_place_0",
        {"object": _harness_binding("object", "apple_1", 0)},
    )
    store.invalidate_revision(1)
    assert store.semantic_anchor_for("occ_place_0", "object").value == "apple_1"


def test_state_scoped_evidence_fails_closed_after_world_change() -> None:
    state_scoped = GroundingEvidence(
        evidence_id="state",
        evidence_type="validated_tool_output",
        payload={"value": "pen_1"},
        source="tool_output",
        observed_at_revision=4,
        valid_from_revision=4,
        stability=EvidenceStability.STATE_SCOPED,
    )
    persistent = GroundingEvidence(
        evidence_id="task",
        evidence_type="task_binding",
        payload={"value": "pen"},
        source="task",
        observed_at_revision=4,
        valid_from_revision=4,
        stability=EvidenceStability.PERSISTENT,
    )

    assert state_scoped.valid_at(4) is True
    assert state_scoped.valid_at(5) is False
    assert persistent.valid_at(5) is True
