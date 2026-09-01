from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.bindings import (
    BindingExpression, BindingExprKind, BindingSource,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    ContractSource,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
    ToolAsset,
)
from atomic_skillgraph.core.edges import ExistingEdgeEvidence, GraphEdge, GraphEdgeType
from atomic_skillgraph.core.errors import (
    AgentProtocolError,
    AtomicSkillGraphError,
    FailureLayer,
    PlannerGraphValidationError,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
    RuntimeRepeatConstraint,
)
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.harness.protocol import HarnessActionSpec
from atomic_skillgraph.harness.protocol import HarnessActionResult
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.graph_store import GraphStore
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.planner.pipeline import (
    PlannerPipeline,
    _is_planner_content_failure,
    _require_supplied_atomic_refs,
)
from atomic_skillgraph.planner.validator import PlannerValidator
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
from atomic_skillgraph.runtime.loop_guard import ActionLoopGuard
from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.validation.engine import ValidationEngine
from atomic_skillgraph.validation.tool_validator import ToolValidator
from experiments.fakes import (
    FakeAgentFactory,
    FakeHarness,
    FakeReply,
    fake_task,
    planner_gap_replies,
)


def _atomic(
    logical_id: str,
    *,
    inputs: list[ParameterSpec] | None = None,
    outputs: list[ParameterSpec] | None = None,
    preconditions: list[SemanticPredicate] | None = None,
    effects: list[SemanticPredicate] | None = None,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef(logical_id, "1.0.0"),
        logical_id,
        list(inputs or []),
        list(outputs or []),
        list(preconditions or []),
        list(effects or []),
        {},
        [],
        {},
        {},
        SkillStatus.ACTIVE,
    )


class _SkillView:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self.by_ref = {item.ref: item for item in atomics}

    def get_atomic(self, ref: SkillRef) -> AbstractAtomicSkill:
        return self.by_ref[ref]


class _NoExistingEdges:
    def existing_edge_by_id(self, *_args, **_kwargs):
        return None


def _two_step_plan(
    source: AbstractAtomicSkill,
    target: AbstractAtomicSkill,
    edge: GraphEdge,
    *,
    binding_expression: BindingExpression | None = None,
    coverage: dict | None = None,
) -> RuntimeLinearPlan:
    target_effect = target.effects[0]
    return RuntimeLinearPlan(
        "task",
        "atomic_composition",
        None,
        [
            RuntimeOccurrence("s1", "occ_source", source.ref, ["r_source"], {}, [], source.effects),
            RuntimeOccurrence(
                "s2",
                "occ_target",
                target.ref,
                ["r_target"],
                {"item": binding_expression} if binding_expression is not None else {},
                [],
                target.effects,
            ),
        ],
        ["s1", "s2"],
        [edge],
        [],
        TaskContract([target_effect], source=ContractSource.ADAPTER_DERIVED),
        {"requirement_coverage": coverage or {"r_target": ["s2"]}},
    )


def test_planner_rejects_nonexistent_dataflow_role_and_forged_coverage() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("held", "entity")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "entity")],
        effects=[SemanticPredicate("object.at_location", {})],
    )
    edge = GraphEdge(
        "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
        "does_not_exist", "item", "planner_proposed",
    )
    plan = _two_step_plan(
        source,
        target,
        edge,
        coverage={"r_target": ["not_a_step"]},
    )
    result = PlannerValidator(
        _SkillView(source, target), _NoExistingEdges()
    ).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert result.passed is False
    assert result.checks["edge_roles_valid"] is False
    assert result.checks["requirement_coverage"] is False


def test_planner_coverage_cannot_claim_atomic_from_another_candidate_set() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("held", "entity")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "entity")],
        effects=[SemanticPredicate("object.at_location", {})],
    )
    edge = GraphEdge(
        "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
        "held", "item", "planner_proposed",
    )
    expression = BindingExpression(
        BindingExprKind.DATA_FLOW, source_role="held", source_step="s1"
    )
    result = PlannerValidator(_SkillView(source, target), _NoExistingEdges()).validate(
        _two_step_plan(source, target, edge, binding_expression=expression),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        requirement_candidates={
            "r_source": {str(source.ref)},
            "r_target": {str(source.ref)},
        },
        harness_profile="alfworld_v3",
    )
    assert result.passed is False
    assert result.checks["requirement_coverage"] is False


def test_planner_requires_dataflow_expression_to_match_explicit_edge() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("held", "entity")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "entity")],
        effects=[SemanticPredicate("object.at_location", {})],
    )
    edge = GraphEdge(
        "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
        "held", "item", "planner_proposed",
    )
    valid_expression = BindingExpression(
        BindingExprKind.DATA_FLOW, source_role="held", source_step="s1"
    )
    validator = PlannerValidator(_SkillView(source, target), _NoExistingEdges())
    valid = validator.validate(
        _two_step_plan(source, target, edge, binding_expression=valid_expression),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert valid.passed is True

    mismatched = BindingExpression(
        BindingExprKind.DATA_FLOW, source_role="other", source_step="s1"
    )
    invalid = validator.validate(
        _two_step_plan(source, target, edge, binding_expression=mismatched),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert invalid.passed is False
    assert invalid.checks["data_flow_expression_consistent"] is False
    assert "data_flow_error" in invalid.failure_codes


def test_planner_checks_optional_dataflow_expressions_too() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("held", "entity")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "string", required=False)],
        effects=[SemanticPredicate("object.at_location", {})],
    )
    edge = GraphEdge(
        "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
        "held", "item", "planner_proposed",
    )
    mismatched = BindingExpression(
        BindingExprKind.DATA_FLOW, source_role="other", source_step="s1"
    )
    result = PlannerValidator(_SkillView(source, target), _NoExistingEdges()).validate(
        _two_step_plan(source, target, edge, binding_expression=mismatched),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert result.passed is False
    assert result.checks["data_flow_expression_consistent"] is False


def test_planner_rejects_binding_source_that_conflicts_with_data_edge() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("held", "entity")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "string")],
        effects=[SemanticPredicate("object.at_location", {})],
    )
    edge = GraphEdge(
        "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
        "held", "item", "planner_proposed",
    )
    result = PlannerValidator(_SkillView(source, target), _NoExistingEdges()).validate(
        _two_step_plan(
            source, target, edge,
            binding_expression=BindingExpression(
                BindingExprKind.CONSTANT, constant="different_authority"
            ),
        ),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert result.passed is False
    assert result.checks["one_authoritative_producer"] is False
    assert result.checks["data_flow_expression_consistent"] is False


def test_planner_rejects_misfiled_edge_type_and_incompatible_semantic_types() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("held", "boolean")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "entity")],
        effects=[SemanticPredicate("object.at_location", {})],
    )
    expression = BindingExpression(
        BindingExprKind.DATA_FLOW, source_role="held", source_step="s1"
    )
    validator = PlannerValidator(_SkillView(source, target), _NoExistingEdges())
    incompatible = validator.validate(
        _two_step_plan(
            source,
            target,
            GraphEdge(
                "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
                "held", "item", "planner_proposed",
            ),
            binding_expression=expression,
        ),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert incompatible.passed is False
    assert incompatible.checks["edge_semantic_types_compatible"] is False

    source.outputs[0].semantic_type = "entity"
    misfiled = validator.validate(
        _two_step_plan(
            source,
            target,
            GraphEdge(
                "edge", GraphEdgeType.REQUIRES_SKILL, "s1", "s2",
                "held", "item", "planner_proposed",
            ),
            binding_expression=expression,
        ),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert misfiled.passed is False
    assert misfiled.checks["edge_types_valid"] is False


def test_dependency_roles_accept_documented_effect_precondition_aliases() -> None:
    heated = SemanticPredicate("object.heated", {"object": "target"})
    source = _atomic("heat", effects=[heated])
    target = _atomic(
        "place",
        preconditions=[heated],
        effects=[SemanticPredicate("object.at_location", {"object": "target"})],
    )
    dependency = GraphEdge(
        "dependency", GraphEdgeType.REQUIRES_SKILL, "s1", "s2",
        "object_heated", "preceded_by_heating", "planner_proposed",
    )
    plan = RuntimeLinearPlan(
        "task", "atomic_composition", None,
        [
            RuntimeOccurrence("s1", "occ_source", source.ref, ["r_source"], {}, [], source.effects),
            RuntimeOccurrence("s2", "occ_target", target.ref, ["r_target"], {}, [], target.effects),
        ],
        ["s1", "s2"],
        [],
        [dependency],
        TaskContract(target.effects, source=ContractSource.ADAPTER_DERIVED),
        {"requirement_coverage": {"r_target": ["s2"]}},
    )
    result = PlannerValidator(_SkillView(source, target), _NoExistingEdges()).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert result.passed is True
    assert result.checks["edge_roles_valid"] is True


def test_planner_allows_control_only_linear_occurrences() -> None:
    atomic = _atomic(
        "place",
        inputs=[ParameterSpec("object", "entity")],
        effects=[SemanticPredicate(
            "object.at_location",
            {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="object")},
        )],
    )
    occurrences = [
        RuntimeOccurrence(
            f"s{index}", f"occ{index}", atomic.ref, [],
            {"object": BindingExpression(BindingExprKind.CONSTANT, constant=f"object_{index}")},
            [], atomic.effects,
        )
        for index in (1, 2)
    ]
    plan = RuntimeLinearPlan(
        "task", "atomic_composition", None, occurrences, ["s1", "s2"], [], [],
        TaskContract([atomic.effects[0]], source=ContractSource.ADAPTER_DERIVED), {},
    )
    result = PlannerValidator(_SkillView(atomic), _NoExistingEdges()).validate(
        plan, mode=RuntimeMode.ONLINE, harness_profile="alfworld_v3"
    )
    assert result.passed is True
    assert "no_disconnected_occurrence" not in result.checks


def _pick_two_plan(atomic: AbstractAtomicSkill, *, second_source_role: str) -> RuntimeLinearPlan:
    effects = [SemanticPredicate(
        "object.at_location",
        {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="object")},
    )]
    dependency = GraphEdge(
        "dependency", GraphEdgeType.REQUIRES_SKILL, "s1", "s2",
        "object_at_location", "requires_location", "planner_proposed",
    )
    return RuntimeLinearPlan(
        "pick-two", "atomic_composition", None,
        [
            RuntimeOccurrence(
                "s1", "occ1", atomic.ref, [],
                {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="object_1")},
                [], effects,
            ),
            RuntimeOccurrence(
                "s2", "occ2", atomic.ref, [],
                {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role=second_source_role)},
                [], effects,
            ),
        ],
        ["s1", "s2"], [], [dependency],
        TaskContract(
            [SemanticPredicate(
                "object.at_location", {"object": "target"},
                cardinality=2, distinct_by="object",
            )],
            [{
                "constraint_id": "cc_pick_two",
                "predicate": "object.at_location", "role": "object",
                "count": 2, "distinct_by": "object",
                "shared_roles": [],
                "composition_mode": "repeat_unit",
            }],
            [],
            ContractSource.ADAPTER_DERIVED,
        ),
        {},
        [RuntimeRepeatConstraint(
            block_id="pick_two",
            basis_constraint_id="cc_pick_two",
            count=2,
            iteration_steps=(("s1",), ("s2",)),
            distinct_roles=("object",),
            shared_roles=(),
            step_role_bindings={
                "s1": {"object": "object"},
                "s2": {"object": "object"},
            },
        )],
    )


def test_pick_two_aggregates_occurrences_from_formal_repeat_authority() -> None:
    effect = SemanticPredicate(
        "object.at_location",
        {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="object")},
    )
    atomic = _atomic(
        "place", inputs=[ParameterSpec("object", "entity")],
        preconditions=[effect], effects=[effect],
    )
    validator = PlannerValidator(_SkillView(atomic), _NoExistingEdges())
    valid = validator.validate(
        _pick_two_plan(atomic, second_source_role="object_2"),
        mode=RuntimeMode.ONLINE, harness_profile="alfworld_v3",
    )
    assert valid.passed is True
    assert valid.checks["task_contract_effect_coverage"] is True
    assert valid.checks["identity_cardinality_preserved"] is True

    # Static proof trusts the validated RuntimeRepeatConstraint rather than
    # synthetic task-role spellings. Runtime preflight enforces concrete
    # distinctness across the two iterations.
    same_raw_source = validator.validate(
        _pick_two_plan(atomic, second_source_role="object_1"),
        mode=RuntimeMode.ONLINE, harness_profile="alfworld_v3",
    )
    assert same_raw_source.passed is True
    assert same_raw_source.checks["task_contract_effect_coverage"] is True

    missing_authority_plan = _pick_two_plan(
        atomic, second_source_role="object_2",
    )
    missing_authority_plan.repeat_constraints = []
    missing_authority = validator.validate(
        missing_authority_plan,
        mode=RuntimeMode.ONLINE,
        harness_profile="alfworld_v3",
    )
    assert missing_authority.passed is False
    assert "planner_repeat_block_invalid" in missing_authority.failure_codes


def test_existing_edge_id_cannot_be_reused_with_different_roles() -> None:
    source = _atomic(
        "source",
        outputs=[ParameterSpec("first", "entity"), ParameterSpec("second", "entity")],
        effects=[SemanticPredicate("agent.holds", {})],
    )
    target = _atomic(
        "target",
        inputs=[ParameterSpec("item", "entity")],
        effects=[SemanticPredicate("object.at_location", {})],
    )

    class KnownGraph:
        def existing_edge_by_id(self, *_args, **_kwargs):
            return ExistingEdgeEvidence(
                "known", "skill://composite@1.0.0", str(source.ref), str(target.ref),
                "data_flow", "first", "item", ("entity", "entity"), ("trace",),
            )

    edge = GraphEdge(
        "edge", GraphEdgeType.DATA_FLOW, "s1", "s2",
        "second", "item", "existing_active", "known",
    )
    result = PlannerValidator(_SkillView(source, target), KnownGraph()).validate(
        _two_step_plan(source, target, edge),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["r_target"],
        harness_profile="alfworld_v3",
    )
    assert result.passed is False
    assert result.checks["edge_origin_valid"] is False


def test_graph_store_existing_data_edge_includes_authoritative_semantic_types(tmp_path: Path) -> None:
    database, skills, _tools, graph = _empty_runtime(tmp_path)
    try:
        source = _atomic(
            "source", outputs=[ParameterSpec("held", "entity")],
            effects=[SemanticPredicate("agent.holds", {})],
        )
        target = _atomic(
            "target", inputs=[ParameterSpec("item", "string")],
            effects=[SemanticPredicate("object.at_location", {})],
        )
        skills.register_atomic(source)
        skills.register_atomic(target)
        skills.register_composite(CompositeSkill(
            SkillRef("workflow", "1.0.0"), "workflow",
            [
                CompositeOccurrence("s1", "occ1", source.ref, {}),
                CompositeOccurrence("s2", "occ2", target.ref, {}),
            ],
            ["s1", "s2"],
            [GraphEdge(
                "known", GraphEdgeType.DATA_FLOW, "s1", "s2",
                "held", "item", "extractor_validated", evidence_refs=("trace",),
            )],
            [], TaskContract(target.effects), {}, {}, {}, {}, SkillStatus.ACTIVE,
        ))
        evidence = graph.existing_edges(
            [str(source.ref), str(target.ref)], mode=RuntimeMode.ONLINE
        )
        assert len(evidence) == 1
        assert evidence[0].semantic_types == ("entity", "string")
    finally:
        database.close()


def test_dataflow_expression_resolves_step_id_to_occurrence_owner() -> None:
    source = RuntimeOccurrence("s1", "occ_source", SkillRef("source", "1"), [], {}, [], [])
    target = RuntimeOccurrence(
        "s2",
        "occ_target",
        SkillRef("target", "1"),
        [],
        {
            "item": BindingExpression(
                BindingExprKind.DATA_FLOW, source_role="held", source_step="s1"
            )
        },
        [],
        [],
    )
    plan = RuntimeLinearPlan(
        "task", "atomic_composition", None, [source, target], ["s1", "s2"],
        [GraphEdge("edge", GraphEdgeType.DATA_FLOW, "s1", "s2", "held", "item", "planner_proposed")],
        [], TaskContract(), {},
    )
    store = RuntimeBindingStore()
    # Register the plan's step/occurrence ownership without applying the edge
    # to its target yet, then resolve the explicit expression.
    store.apply_data_flow(plan, "s1", revision=1)
    store.publish_validated_outputs(source, {"held": "apple_1"}, ["validator:witness"], 1)
    store.resolve_occurrence_specs(target, 1)
    assert store.snapshot_for_node(target)["item"].value == "apple_1"
    assert store.snapshot_for_node(target)["item"].source is BindingSource.DATA_FLOW
    assert store.semantic_anchor_for(target, "item") is not None


def _empty_runtime(tmp_path: Path):
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    graph = GraphStore(database, skills)
    return database, skills, tools, graph


def test_dynamic_tool_result_carries_the_new_action_catalog(tmp_path: Path) -> None:
    database, skills, tools, graph = _empty_runtime(tmp_path)
    try:
        harness = FakeHarness()
        factory = FakeAgentFactory()
        factory.enqueue("planner", planner_gap_replies())
        factory.enqueue(
            "runtime_dynamic",
            [
                FakeReply.tool("environment_action", {"action_id": "r000_a001"}),
                FakeReply.tool("environment_action", {"action_id": "r001_a001"}),
            ],
        )
        runtime = RuntimeOrchestrator(
            PlannerPipeline(skills, graph, factory),
            harness,
            InvocationCompiler(skills, tools, harness),
            ValidationEngine(),
            factory,
            runtime_config={"global_action_budget": 10, "node_action_budget": 5},
        )
        trace = runtime.run_task(
            fake_task("catalog-probe", "apple_1", requires_rescue=True)
        )
        session = next(
            item for item in trace.agent_sessions
            if item.session_type == "DynamicTaskSession"
        )
        first_result = json.loads(next(
            message["content"] for message in session.snapshot["messages"]
            if message["role"] == "tool"
        ))
        assert first_result["new_revision"] == 1
        assert first_result["action_catalog"] == [
            {
                "action_id": "r001_a001",
                "action_type": "EXAMINE",
                "arguments": {"item": "apple_1"},
                "display_text": "examine apple_1",
                "revision": 1,
            }
        ]
        assert "r000_a001" not in {
            item["action_id"] for item in first_result["action_catalog"]
        }
        assert first_result["remaining_budget"]["used_global_actions"] == 1
        assert first_result["remaining_budget"]["remaining_global_actions"] == 9
    finally:
        database.close()


def test_environment_action_never_grounds_from_rejection_or_missing_current_witness(
    tmp_path: Path,
) -> None:
    database, skills, tools, graph = _empty_runtime(tmp_path)
    try:
        harness = FakeHarness()
        atomic = _atomic(
            "inspect_item",
            inputs=[ParameterSpec("item", "string", runtime_resolvable=True)],
            effects=[SemanticPredicate("object.observed", {})],
        )
        skills.register_atomic(atomic)
        occurrence = RuntimeOccurrence(
            "s1", "occ1", atomic.ref, [], {}, [], atomic.effects
        )
        plan = RuntimeLinearPlan(
            "grounding", "atomic_composition", None, [occurrence], ["s1"], [], [],
            TaskContract(atomic.effects), {},
        )
        runtime = RuntimeOrchestrator(
            PlannerPipeline(skills, graph, lambda *_args: None),
            harness,
            InvocationCompiler(skills, tools, harness),
            ValidationEngine(),
            lambda *_args: None,
            runtime_config={"global_action_budget": 10, "node_action_budget": 5},
        )
        ctx = runtime._create_context(
            fake_task("grounding", "apple_1", expose_binding=False), plan
        )
        spec = ctx.action_catalog[0]
        session = SimpleNamespace(session_id="session")

        harness.execute_action = lambda _action_id, _revision: HarnessActionResult(
            False, "Nothing happens.", False, False, 0, [spec]
        )
        span = ctx.trace_builder.start_span("probe", occurrence.occurrence_id)
        loop_guard = ActionLoopGuard()
        runtime.node_executor._execute_environment_call(
            SimpleNamespace(
                call_id="reject", name="environment_action",
                arguments={"action_id": spec.action_id},
            ),
            session, occurrence, ctx, span_id=span.span_id, origin="test",
            loop_guard=loop_guard,
        )
        assert "item" not in ctx.binding_store.snapshot_for_node(occurrence)

        harness.execute_action = lambda _action_id, _revision: HarnessActionResult(
            True, "Accepted but no current catalog witness.", False, False, 1, []
        )
        runtime.node_executor._execute_environment_call(
            SimpleNamespace(
                call_id="no-witness", name="environment_action",
                arguments={"action_id": spec.action_id},
            ),
            session, occurrence, ctx, span_id=span.span_id, origin="test",
            loop_guard=loop_guard,
        )
        assert "item" not in ctx.binding_store.snapshot_for_node(occurrence)
    finally:
        database.close()


class _InfrastructureSession:
    session_id = "infra_session"

    def next_turn(self, *_args, **_kwargs):
        raise AtomicSkillGraphError(
            "llm_error", "provider unavailable", layer=FailureLayer.INFRASTRUCTURE
        )

    def snapshot(self):
        return {"session_id": self.session_id}


def test_planner_provider_infrastructure_error_is_not_dynamic_fallback(tmp_path: Path) -> None:
    database, skills, _tools, graph = _empty_runtime(tmp_path)
    try:
        skills.register_atomic(_atomic(
            "unrelated_active_atomic",
            effects=[SemanticPredicate("unrelated.effect", {"value": "x"})],
        ))
        planner = PlannerPipeline(skills, graph, lambda *_args: _InfrastructureSession())
        harness = FakeHarness()
        task = fake_task("planner-infra", "apple_1")
        with pytest.raises(AtomicSkillGraphError, match="provider unavailable"):
            planner.build_plan(task, harness, initial_observation="initial")
    finally:
        database.close()


def test_untyped_planner_provider_error_is_not_dynamic_fallback(tmp_path: Path) -> None:
    database, skills, _tools, graph = _empty_runtime(tmp_path)
    try:
        skills.register_atomic(_atomic(
            "unrelated_active_atomic",
            effects=[SemanticPredicate("unrelated.effect", {"value": "x"})],
        ))
        class Session:
            def next_turn(self, *_args, **_kwargs):
                raise ConnectionError("provider transport failed")

        planner = PlannerPipeline(skills, graph, lambda *_args: Session())
        with pytest.raises(ConnectionError, match="provider transport failed"):
            planner.build_plan(
                fake_task("planner-untyped-infra", "apple_1"),
                FakeHarness(),
                initial_observation="initial",
            )
    finally:
        database.close()


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (KeyError, "planner fixture key"),
        (TypeError, "planner fixture type"),
        (ValueError, "planner fixture value"),
    ],
)
def test_bare_planner_programming_errors_are_not_dynamic_fallback(
    tmp_path: Path,
    error_type: type[Exception],
    message: str,
) -> None:
    database, skills, _tools, graph = _empty_runtime(tmp_path)
    try:
        skills.register_atomic(_atomic(
            "unrelated_active_atomic",
            effects=[SemanticPredicate("unrelated.effect", {"value": "x"})],
        ))

        class Session:
            def next_turn(self, *_args, **_kwargs):
                raise error_type(message)

        planner = PlannerPipeline(skills, graph, lambda *_args: Session())
        with pytest.raises(error_type, match=message):
            planner.build_plan(
                fake_task("planner-programming-error", "apple_1"),
                FakeHarness(),
                initial_observation="initial",
            )
    finally:
        database.close()


def test_explicit_planner_proposal_error_may_fallback_to_dynamic(
    tmp_path: Path,
) -> None:
    database, skills, _tools, graph = _empty_runtime(tmp_path)
    try:
        skills.register_atomic(_atomic(
            "unrelated_active_atomic",
            effects=[SemanticPredicate("unrelated.effect", {"value": "x"})],
        ))

        class Session:
            def next_turn(self, *_args, **_kwargs):
                raise AgentProtocolError(
                    "runtime_agent_schema_error",
                    "planner proposal failed its explicit schema",
                    layer=FailureLayer.RUNTIME_AGENT,
                )

        planner = PlannerPipeline(skills, graph, lambda *_args: Session())
        plan = planner.build_plan(
            fake_task("planner-explicit-proposal-error", "apple_1"),
            FakeHarness(),
            initial_observation="initial",
        )
        assert plan.source == "full_dynamic"
        assert plan.planner_audit["fallback_reason"] == (
            "planner_requirement_invalid"
        )
    finally:
        database.close()


def test_generic_error_cannot_masquerade_as_planner_content_failure(
    tmp_path: Path,
) -> None:
    database, skills, _tools, graph = _empty_runtime(tmp_path)
    try:
        skills.register_atomic(_atomic(
            "unrelated_active_atomic",
            effects=[SemanticPredicate("unrelated.effect", {"value": "x"})],
        ))

        class Session:
            def next_turn(self, *_args, **_kwargs):
                raise AtomicSkillGraphError(
                    "planner_graph_invalid",
                    "generic error with forged Planner attribution",
                    layer=FailureLayer.PLANNER_GRAPH,
                )

        planner = PlannerPipeline(skills, graph, lambda *_args: Session())
        with pytest.raises(
            AtomicSkillGraphError,
            match="generic error with forged Planner attribution",
        ):
            planner.build_plan(
                fake_task("planner-generic-forgery", "apple_1"),
                FakeHarness(),
                initial_observation="initial",
            )
    finally:
        database.close()


def test_unsupplied_workflow_ref_is_typed_fallback_eligible_content_failure() -> None:
    proposal = SimpleNamespace(steps=[
        SimpleNamespace(node_ref=SkillRef("invented_atomic", "1.0.0")),
    ])

    with pytest.raises(PlannerGraphValidationError) as error:
        _require_supplied_atomic_refs(
            proposal,
            {str(SkillRef("retrieved_atomic", "1.0.0"))},
        )

    assert error.value.code == "planner_graph_invalid"
    assert error.value.layer is FailureLayer.PLANNER_GRAPH
    assert _is_planner_content_failure(error.value) is True


def test_dynamic_provider_infrastructure_error_is_rethrown(tmp_path: Path) -> None:
    database, skills, tools, graph = _empty_runtime(tmp_path)
    try:
        scripted = FakeAgentFactory()
        scripted.enqueue("planner", planner_gap_replies())

        def factory(first, second):
            if first == "runtime_dynamic":
                return _InfrastructureSession()
            return scripted(first, second)

        runtime = RuntimeOrchestrator(
            PlannerPipeline(skills, graph, factory),
            FakeHarness(),
            InvocationCompiler(skills, tools, FakeHarness()),
            ValidationEngine(),
            factory,
            runtime_config={"global_action_budget": 10, "node_action_budget": 5},
        )
        with pytest.raises(AtomicSkillGraphError, match="provider unavailable"):
            runtime.run_task(fake_task("dynamic-infra", "apple_1", requires_rescue=True))
    finally:
        database.close()


def test_seeded_provider_infrastructure_error_is_rethrown(tmp_path: Path) -> None:
    database, skills, tools, graph = _empty_runtime(tmp_path)
    try:
        harness = FakeHarness()
        task = fake_task("seeded-infra", "apple_1")
        contract = harness.task_contract(task)
        atomic = _atomic(
            "hold_item",
            inputs=[ParameterSpec("item", "string", True, True, "concrete")],
            effects=[
                SemanticPredicate(
                    "agent.holds",
                    {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="item")},
                )
            ],
        )
        skills.register_atomic(atomic)
        skills.register_composite(CompositeSkill(
            SkillRef("hold_workflow", "1.0.0"),
            "hold target item",
            [CompositeOccurrence(
                "s1", "occ1", atomic.ref,
                {"item": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="item")},
            )],
            ["s1"],
            [],
            [],
            contract,
            {},
            {},
            {},
            {"harness_profiles": ["fake_v3"]},
            SkillStatus.ACTIVE,
        ))
        factory = lambda *_args: _InfrastructureSession()
        runtime = RuntimeOrchestrator(
            PlannerPipeline(skills, graph, factory),
            harness,
            InvocationCompiler(skills, tools, harness),
            ValidationEngine(),
            factory,
            runtime_config={"global_action_budget": 10, "node_action_budget": 5},
        )
        with pytest.raises(AtomicSkillGraphError, match="provider unavailable"):
            runtime.run_task(task)
    finally:
        database.close()


def test_tool_runner_does_not_relabel_harness_crash_as_tool_failure() -> None:
    tool = ToolAsset(
        ToolRef("probe", "1.0.0"),
        "probe",
        {"type": "object", "properties": {}},
        {"output_schema": {"type": "object", "properties": {}}},
        "primitive_ir",
        {"steps": [{"action_type": "TAKE", "argument_mapping": {}}]},
        [],
        {"reviewed": True},
        {},
        {},
        ToolStatus.ACTIVE,
    )

    class CrashingHarness:
        def compile_primitive(self, *_args):
            return HarnessActionSpec("a001", 0, "TAKE", {}, "take", "take", {})

        def execute_action(self, *_args):
            raise AtomicSkillGraphError(
                "infrastructure_failure", "environment crashed",
                layer=FailureLayer.INFRASTRUCTURE,
            )

    class TraceBuilder:
        trace = SimpleNamespace(environment_actions=[], tool_executions=[])

        def start_span(self, *_args, **_kwargs):
            return SimpleNamespace(span_id="span")

        def finish_span(self, *_args):
            return None

    context = SimpleNamespace(
        world_revision=0,
        harness=CrashingHarness(),
        budget=SimpleNamespace(consume_action=lambda: None),
        trace_builder=TraceBuilder(),
        update_after_action=lambda *_args: None,
    )
    with pytest.raises(AtomicSkillGraphError, match="environment crashed"):
        ToolRunner(ToolValidator()).run(
            tool, {}, context, occurrence_id="occurrence"
        )
