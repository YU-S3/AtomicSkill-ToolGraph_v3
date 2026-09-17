"""R10.2.1 release invariants exercised through the production node engine."""
from test_r10_runtime import setup, action
from fixtures.certified_values import concrete_entities
from atomic_skillgraph.runtime.composite_executor import VerifiedCompositeExecutor
from atomic_skillgraph.runtime.runtime_step import run_runtime_step
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind, GroundingConstraint, GroundingConstraintKind
from atomic_skillgraph.core.contracts import SemanticPredicate
from dataclasses import replace
import copy
import pytest


def test_C03_C05_agent_keeps_node_until_explicit_invocation(tmp_path):
    def choose(request, count):
        if count == 1:
            return action(request, "GO_TO", destination="cabinet_1")
        if count == 2:
            return action(request, "OPEN")
        assert count == 3, str(request.messages[-1])[-1800:]
        name = next(tool.name for tool in request.tools if tool.name.startswith("invoke_impl_"))
        return name, {"object": "egg_1", "source": "cabinet_1"}
    system, ctx, occurrence, _, provider = setup(tmp_path, choose)
    try:
        result = VerifiedCompositeExecutor(system.orchestrator.node_executor).run_occurrence(occurrence, ctx)
        assert result.atomic_effect_passed
        assert len(provider.requests) == 3
        assert not ctx.trace_builder.trace.metadata.get("r10_automatic_invocations")
    finally:
        system.close()


def _opened(tmp_path):
    system, ctx, occurrence, invocations, provider = setup(tmp_path, lambda r, n:
        action(r, "GO_TO", destination="cabinet_1") if n == 1 else action(r, "OPEN"))
    for _ in range(2):
        run_runtime_step(system.orchestrator.node_executor, "preparation", occurrence, ctx, invocations, [])
    return system, ctx, occurrence, invocations, provider


def _state(ctx):
    return copy.deepcopy((ctx.harness._runtime_state_digest(), ctx.binding_store._bindings,
        ctx.binding_store._semantic_anchors, ctx.binding_store._proposals, ctx.binding_store._outputs,
        ctx.binding_store.repeat_state, ctx.budget.snapshot(), ctx.validated_outputs))


def _prepare(system, ctx, occurrence, compiled):
    return system.invocation_compiler.preflight(compiled, call_name=compiled.spec.name,
        call_id="explicit", arguments={"object": "egg_1", "source": "cabinet_1"},
        occurrence=occurrence, binding_store=ctx.binding_store, evidence_store=ctx.evidence_store,
        revision=ctx.world_revision, task_contract=ctx.task_contract)


def test_V07_entry_inspection_does_not_commit_any_candidate(tmp_path):
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        before = _state(ctx)
        for compiled in [invocations[0], copy.deepcopy(invocations[0])]:
            report = _prepare(system, ctx, occurrence, compiled)
            assert report.passed, report
            assert report.binding_updates
            assert _state(ctx) == before
    finally:
        system.close()


def test_V02_explicit_closed_entry_rejects_open_container_without_action(tmp_path):
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        compiled = copy.deepcopy(invocations[0])
        compiled.tools[0].interface["entry_contract"]["conditions"] = to_primitive([
            SemanticPredicate("container.closed", {"container": "$source"})])
        before = _state(ctx)
        report = _prepare(system, ctx, occurrence, compiled)
        assert not report.passed
        assert report.failure_code == "tool_entry_conditions_unsatisfied"
        assert _state(ctx) == before
    finally:
        system.close()


def test_V08_explicit_attempt_completes_in_one_call_after_preparation(tmp_path):
    def choose(r, n):
        if n == 1:
            return action(r, "GO_TO", destination="cabinet_1")
        if n == 2:
            return action(r, "OPEN")
        assert n == 3
        name, args = action(r, "TAKE", object="egg_1")
        return name, {**args, "intent": "attempt_current_atomic",
            "candidate_bindings": {"object": "egg_1", "source": "cabinet_1"},
            "candidate_outputs": {"object": "egg_1"}}
    system, ctx, occurrence, _, provider = setup(tmp_path, choose)
    try:
        result = VerifiedCompositeExecutor(system.orchestrator.node_executor).run_occurrence(occurrence, ctx)
        assert result.atomic_effect_passed
        assert result.validated_outputs == {"object": "egg_1"}
        assert len(provider.requests) == 3
        assert not ctx.trace_builder.trace.implementation_invocations
    finally:
        system.close()


def test_C07_explore_terminal_never_fabricates_node_outputs(tmp_path):
    def choose(r, n):
        assert n <= 3
        return action(r, "GO_TO", destination="cabinet_1") if n == 1 else action(r, "OPEN" if n == 2 else "TAKE")
    system, ctx, occurrence, _, provider = setup(tmp_path, choose)
    ctx.harness.case = replace(ctx.harness.case, terminal_action="TAKE")
    try:
        result = VerifiedCompositeExecutor(system.orchestrator.node_executor).run_occurrence(occurrence, ctx)
        assert ctx.benchmark_terminal()
        assert not result.validated_outputs
        assert not ctx.validated_outputs
        assert len(provider.requests) == 3
    finally:
        system.close()


def test_D01_D02_identity_persists_but_current_existence_does_not(tmp_path):
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *_: pytest.fail("no Agent"))
    try:
        certified = concrete_entities({'object': 'certified_identity'}, ['validator:source'], 0)
        ctx.binding_store.publish_validated_outputs(occurrence, {"object": "certified_identity"}, ["validator:source"], 0,
            certified_bindings=certified)
        ctx.evidence_store.add_validated_tool_output("object", "certified_identity", ["validator:source"],
            certified_binding=certified['object'], occurrence_id=occurrence.occurrence_id)
        ctx.binding_store.invalidate_revision(1)
        ctx.evidence_store.replace_action_catalog([], 1)
        current = ctx.binding_store.snapshot_for_node(occurrence)["object"]
        assert current.value == "certified_identity"
        assert current.resolution.value == "concrete"
        assert current.evidence_refs == ["validator:source"]
        constraint = GroundingConstraint("given", GroundingConstraintKind.ARGUMENT_CONCRETE,
            argument_mapping={"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="object")})
        assert ctx.evidence_store.match_constraint(constraint, {"object": current.value}, 1)
        assert not ctx.evidence_store.match_constraint(replace(constraint, kind=GroundingConstraintKind.ARGUMENT_EXISTS), {"object": current.value}, 1)
    finally:
        system.close()


def test_D03_catalog_unique_does_not_assign_missing_input(tmp_path):
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        before = _state(ctx)
        atomic = invocations[0].atomic
        state = system.orchestrator.node_executor.grounding_authority.refresh(occurrence, atomic, invocations, ctx)
        assert "object" in state["missing_bindings"]
        assert "source" in state["missing_bindings"]
        assert not state["candidate_bindings"]
        assert _state(ctx) == before
    finally:
        system.close()


def test_V05_fresh_output_is_not_selected_from_fact(tmp_path):
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        atomic = replace(invocations[0].atomic, validator_spec={})
        result = system.validation.atomic.validate_execution_result(atomic, occurrence,
            {"object": "egg_1", "source": "cabinet_1"}, {}, ctx.harness.validator_channel(),
            current_revision=ctx.world_revision)
        assert not result.passed and result.failure_codes == ["missing_outputs"]
    finally:
        system.close()


def _declared_tool(system, name, inputs, outputs, effects, action_type, action_mapping,
                   returns, *, preconditions=(), program=None):
    from atomic_skillgraph.core.contracts import AbstractAtomicSkill
    from atomic_skillgraph.core.refs import SkillRef
    from atomic_skillgraph.core.status import SkillStatus, ToolStatus
    from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
    from atomic_skillgraph.tooling.proposal import tool_proposal_from_dict, ToolProvenance
    atomic = AbstractAtomicSkill(SkillRef(name, "1.0.0"), "generic declared test capability",
        inputs, outputs, list(preconditions), effects,
        {"output_derivations": {out: {"kind": "input_identity", "input_role": inp}
                                for out, inp in returns.items()}}, [],
        {"steps": ["Inspect current evidence, then choose a legal method."], "notes": []}, {}, SkillStatus.ACTIVE)
    system.skills.register_atomic(atomic)
    raw = {"proposal_version": "2", "decision": "create", "atomic_ref": str(atomic.ref),
        "summary": "generic declared test capability", "inputs": to_primitive(inputs),
        "outputs": to_primitive(outputs), "entry_contract": {"conditions": [], "grounding_constraints": []},
        "program": program or [{"op": "ACTION", "node_id": "act", "action_type": action_type,
            "argument_mapping": {key: {"kind": "skill_input", "source_role": role}
                                 for key, role in action_mapping.items()}, "expected_effects": to_primitive(effects)},
            {"op": "RETURN", "node_id": "end", "output_sources": {
                out: {"source": "tool_input", "field": inp} for out, inp in returns.items()}}],
        "max_actions": 3, "final_effects": to_primitive(effects), "evidence_outputs": [],
        "path_expectations": [], "rationale": "Deterministic production validation fixture"}
    proposal = tool_proposal_from_dict(raw)
    static = system.tool_static_validator.validate_proposal(proposal, atomic, system.harness)
    assert static.passed, static
    occurrence = CanonicalAtomicOccurrence(name, name, atomic.summary, 0, 0,
        {p.name: "cabinet_1" for p in inputs}, {p.name: "cabinet_1" for p in outputs},
        inputs, outputs, list(preconditions), effects, [], [], to_primitive(system.harness._current_task),
        "fixture", atomic.ref)
    compiled = system.tool_compiler.compile_proposal(occurrence, atomic, proposal,
        ToolProvenance(source="r102_fixture", atomic_ref=str(atomic.ref), source_trace_id="fixture", occurrence_id=name))
    compiled.implementation.status = SkillStatus.ACTIVE
    compiled.tool.status = ToolStatus.ACTIVE
    system.tools.register(compiled.tool)
    system.skills.register_implementation(compiled.implementation)
    return compiled


def test_V01_conditional_first_action_does_not_become_global_entry(tmp_path):
    from atomic_skillgraph.core.contracts import ParameterSpec
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    system, ctx, original, _, _ = _opened(tmp_path)
    try:
        role = lambda name: {'kind': 'skill_input', 'source_role': name}
        effects = [SemanticPredicate('agent.holds', {'object': '$object'})]
        program = [
            {'op': 'IF', 'node_id': 'optional_open', 'condition': {'match': {
                'source': 'action_catalog', 'where': {'action_type': 'OPEN'},
                'project': {'kind': 'argument', 'role': 'object'}}},
             'then_branch': [{'op': 'ACTION', 'node_id': 'open', 'action_type': 'OPEN',
                 'argument_mapping': {'object': role('source')}, 'expected_effects': to_primitive([SemanticPredicate('container.open', {'container': '$source'})])}], 'else_branch': []},
            {'op': 'ACTION', 'node_id': 'take', 'action_type': 'TAKE',
             'argument_mapping': {'object': role('object'), 'source': role('source')},
             'expected_effects': to_primitive(effects)},
            {'op': 'RETURN', 'node_id': 'end', 'output_sources': {'held': {'source': 'tool_input', 'field': 'object'}}}]
        parameters = [ParameterSpec(name, 'entity', runtime_resolvable=True, required_resolution='concrete') for name in ('object', 'source')]
        compiled = _declared_tool(system, 'conditional_take', parameters, [ParameterSpec('held', 'entity')],
            effects, 'TAKE', {}, {'held': 'object'}, program=program)
        occurrence = replace(original, node_ref=compiled.atomic.ref, implementation_candidates=[compiled.implementation.ref], expected_effects=effects)
        invocation = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=ctx.task_id)[0]
        checked = _prepare(system, ctx, occurrence, invocation)
        assert checked.passed, checked
        count = len(ctx.trace_builder.trace.environment_actions)
        result = execute_invocation(system.orchestrator.node_executor.implementation_runner,
            invocation, checked, occurrence, ctx, agent_prepared=True)
        assert result.atomic_effect_passed, result
        assert result.validated_outputs == {'held': 'egg_1'}
        assert [a.action_type for a in ctx.trace_builder.trace.environment_actions[count:]] == ['TAKE']
        assert 'open' not in result.tool_results[0].tool_path_evidence['executed_node_ids']
    finally:
        system.close()


@pytest.mark.parametrize("source", ["stored_composite", "atomic_composition"])
def test_C01_C10_actual_return_and_declared_dataflow_autoruns_next_node(tmp_path, source):
    from atomic_skillgraph.core.contracts import ParameterSpec
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
    def choose(request, count):
        assert count == 1, str(request.policy_context["execution_frame"])
        return next(t.name for t in request.tools if t.name.startswith("invoke_impl_")), {"where": "cabinet_1"}
    system, ctx, _, _, provider = setup(tmp_path, choose)
    try:
        system.harness.case = replace(system.harness.case, terminal_action="OPEN")
        param = lambda name: ParameterSpec(name, "entity", True, True, "concrete")
        upstream = _declared_tool(system, "r102_producer", [param("where")], [param("object"), param("location")],
            [SemanticPredicate("agent.at_location", {"location": "$where"})], "GO_TO", {"destination": "where"},
            {"object": "where", "location": "where"})
        downstream = _declared_tool(system, "r102_consumer", [param("target"), param("site")], [],
            [SemanticPredicate("container.open", {"container": "$target"})], "OPEN", {"object": "target"}, {},
            preconditions=[SemanticPredicate("agent.at_location", {"location": "$site"})])
        producer = RuntimeOccurrence("producer", "producer", upstream.atomic.ref, [], {},
            [upstream.implementation.ref], upstream.atomic.effects)
        consumer = RuntimeOccurrence("consumer", "consumer", downstream.atomic.ref, [], {},
            [downstream.implementation.ref], downstream.atomic.effects)
        plan = replace(ctx.plan, source=source, occurrences=[producer, consumer],
            control_sequence=["producer", "consumer"], data_edges=[
                GraphEdge("object_edge", GraphEdgeType.DATA_FLOW, "producer", "consumer", "object", "target"),
                GraphEdge("location_edge", GraphEdgeType.DATA_FLOW, "producer", "consumer", "location", "site")])
        system.planner.build_plan = lambda *args, **kwargs: plan
        trace = system.orchestrator.run_task(ctx.task)
        assert trace.benchmark_success
        assert len(provider.requests) == 1
        assert [a.action_type for a in trace.environment_actions] == ["GO_TO", "OPEN"]
        assert trace.node_records[0].direct_result["validated_outputs"] == {"object": "cabinet_1", "location": "cabinet_1"}
        realized = trace.node_records[1].direct_result["realized_bindings"]
        assert {k: v["value"] for k, v in realized.items()} == {"target": "cabinet_1", "site": "cabinet_1"}
        assert all(v["source"] == "data_flow" and v["evidence_refs"] for v in realized.values())
        assert trace.metadata["r10_metrics"]["composite_auto_node_count"] == 1
    finally:
        system.close()


def test_C09_ambiguous_active_and_candidate_are_not_automatically_selected(tmp_path):
    from atomic_skillgraph.core.refs import SkillRef
    from atomic_skillgraph.core.status import SkillStatus, ToolStatus
    system, ctx, occurrence, invocations, provider = _opened(tmp_path)
    try:
        selected = invocations[0]
        prepared = _prepare(system, ctx, occurrence, selected)
        ctx.binding_store.commit_grounded(occurrence.occurrence_id,
            {b.role: b for b in prepared.binding_updates})
        alternate = copy.deepcopy(selected)
        alternate.implementation.ref = SkillRef("independent_alternative", "1.0.0")
        system.skills.register_implementation(alternate.implementation)
        before = _state(ctx)
        assert system.orchestrator.node_executor.try_autonomous(occurrence, [selected, alternate], ctx) is None
        assert _state(ctx) == before
        candidate = copy.deepcopy(selected)
        candidate.implementation.status = SkillStatus.CANDIDATE
        candidate.tools[0].status = ToolStatus.CANDIDATE
        assert system.orchestrator.node_executor.try_autonomous(occurrence, [candidate], ctx) is None
        assert _state(ctx) == before
        assert len(provider.requests) == 2
    finally:
        system.close()


@pytest.mark.parametrize("won", [False, True])
def test_C07_done_even_without_won_stops_after_explore(tmp_path, won):
    system, ctx, occurrence, _, provider = setup(tmp_path,
        lambda r, n: action(r, "GO_TO", destination="cabinet_1") if n == 1 else pytest.fail("terminal work"))
    try:
        system.harness.case = replace(system.harness.case, terminal_action="GO_TO", terminal_won=won)
        result = system.orchestrator.node_executor.run_agent_node(occurrence, ctx)
        assert len(provider.requests) == 1
        assert ctx.execution_terminal()
        assert ctx.benchmark_terminal() is won
        assert not result.validated_outputs and not result.atomic_effect_passed
    finally:
        system.close()
