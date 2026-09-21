"""CM01--CM06: observer-only regression; no paid provider or validator stubs."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import uuid

import pytest

from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.traces.compiler_observer import initialize, initial_state, finalize, node_window, _dataflow_inputs
from atomic_skillgraph.traces.store import TraceStore
from atomic_skillgraph.runtime.runtime_step import run_runtime_step
from atomic_skillgraph.runtime.composite_executor import VerifiedCompositeExecutor
from experiments.compiler_metrics import trace_metrics, aggregate, write_reports
from test_r10_runtime import setup, action


def observe(ctx, system, provider):
    initialize(ctx.trace_builder, persistent_refs=[str(t.ref) for t in system.tools.tools()],
               request_snapshot=lambda: [f'request_{n}' for n in range(len(provider.requests))])
    initial_state(ctx, SimpleNamespace(observation=ctx.observation, catalog=ctx.action_catalog, new_revision=ctx.world_revision))


def test_CM01_observer_changes_no_messages_tools_decisions_actions_or_bank(tmp_path, monkeypatch):
    records = []
    for enabled in (False, True):
        ids = iter(range(1,10000))
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(ids)))
        system, ctx, occ, inv, provider = setup(tmp_path/str(enabled), lambda r,n: action(r,'GO_TO',destination='cabinet_1'))
        try:
            if enabled:
                observe(ctx,system,provider)
            digest = system.knowledge_digest()
            with node_window(occ,ctx):
                result = run_runtime_step(system.orchestrator.node_executor,'preparation',occ,ctx,inv,[])
            t=ctx.trace_builder.trace
            from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
            records.append(dict(requests=[(r.messages,[t.to_openai() for t in r.tools]) for r in provider.requests],
                actions=to_primitive(t.environment_actions), calls=to_primitive(t.native_tool_calls),
                validations=to_primitive(t.validations), result=to_primitive(result), world=ctx.harness._runtime_state_digest(),
                extractor_source=TraceNormalizer().build(t)))
            assert system.knowledge_digest() == digest
            assert 'compiler_observability' not in str(provider.requests)
        finally:
            system.close()
    assert records[0] == records[1]


@pytest.mark.parametrize('route,expected,accepted',[
    ('stored_composite','p0_module',True),('atomic_composition','p2_composition',True),
    ('full_dynamic','full_dynamic',False),('cold_start','cold_start',False)])
def test_CM02_final_route_only_and_no_assumed_novelty(tmp_path,route,expected,accepted):
    system,ctx,occ,inv,p=setup(tmp_path,lambda *a:None)
    try:
        ctx.plan.source=route
        ctx.trace_builder.trace.planner_audit={'p0_failed':{},'workflow_p2':{},'workflow_p2r':{}}
        observe(ctx,system,p)
        finalize(ctx.trace_builder.trace)
        m=trace_metrics(ctx.trace_builder.trace)
        assert (m['selected_route'],m['accepted_graph']) == (expected,accepted)
        assert m['structural_reuse_kind'] == ('existing_module' if accepted and route=='stored_composite' else 'unknown')
        assert aggregate([{'compiler_diagnostics':m}])['task_count'] == 1
    finally: system.close()


def test_CM03_actual_automatic_program_and_later_rollback_preserve_policy_work(tmp_path):
    from experiments.r10_world_checks import install_fixture, bind
    from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
    from atomic_skillgraph.core.contracts import SemanticPredicate
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from atomic_skillgraph.runtime.checkpoint import capture, restore
    system,ctx,_,_,p=setup(tmp_path,lambda *a:pytest.fail('automatic execution called model'))
    try:
        role=BindingExpression(BindingExprKind.SKILL_INPUT,source_role='target')
        atomic,impl=install_fixture(system,'open_test',[],[SemanticPredicate('container.open',{'container':role})],
            [('GO_TO','destination'),('OPEN','object')],source_target='cabinet_1')
        occ=RuntimeOccurrence('open_test','open_test',atomic.ref,[],{},[impl.ref],atomic.effects)
        ctx.plan.occurrences.append(occ)
        ctx.begin_occurrence(occ);ctx.budget.begin_node(occ.occurrence_id)
        bind(ctx,occ,'cabinet_1');ctx.graph_bootstrap_completed=True
        observe(ctx,system,p)
        saved=capture(ctx,occ.occurrence_id)
        result=VerifiedCompositeExecutor(system.orchestrator.node_executor).run_occurrence(occ,ctx)
        assert result.started and result.completed and result.atomic_effect_passed
        finalize(ctx.trace_builder.trace)
        m=trace_metrics(ctx.trace_builder.trace)
        assert m['program_canonical_actions']==2 and m['llm_free_complete_successors']==1
        links=ctx.trace_builder.trace.metadata['compiler_observability']['program_invocation_links']
        assert links[0]['origin']=='graph_entry_auto' and links[0]['asset_origin']=='persistent'
        # Nested association cannot multiply the same policy actions.
        links.append(copy.deepcopy(links[0]))
        assert trace_metrics(ctx.trace_builder.trace)['program_canonical_actions']==2
        restore(ctx,saved,'controlled_rollback')
        finalize(ctx.trace_builder.trace)
        m=trace_metrics(ctx.trace_builder.trace)
        assert m['program_canonical_actions']==0 and m['program_policy_actions']==2
        assert m['autonomous_complete_entries']==0
        assert all(x['rollback'] for x in links)
    finally: system.close()


def test_CM04_publication_is_not_consumption_and_override_loses_lineage(tmp_path):
    from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution
    from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
    system,ctx,occ,inv,p=setup(tmp_path,lambda *a:None)
    try:
        observe(ctx,system,p)
        producer=replace(occ,occurrence_id='producer',step_id='producer')
        ctx.plan.occurrences.insert(0,producer)
        ctx.plan.data_edges=[GraphEdge('flow',GraphEdgeType.DATA_FLOW,producer.step_id,occ.step_id,source_role='value',target_role='container')]
        binding=RuntimeBinding('value','cabinet_1','entity',BindingSource.TOOL_OUTPUT,
            BindingStatus.GROUNDED,BindingResolution.CONCRETE, ['witness'],ctx.world_revision)
        ctx.binding_store.publish_validated_outputs(producer.occurrence_id,{'value':'cabinet_1'},['witness'],ctx.world_revision,
            certified_bindings={'value':binding})
        assert not ctx.trace_builder.trace.metadata['compiler_observability']['dataflow_consumptions']
        ctx.binding_store.apply_data_flow(ctx.plan,occ.step_id,revision=ctx.world_revision)
        rows=_dataflow_inputs(ctx,occ,{'container':'cabinet_1'})
        assert len(rows)==1 and rows[0]['edge_id']=='flow'
        ctx.binding_store.commit_grounded(occ.occurrence_id,{'container':replace(binding,role='container',
            value='egg',source=BindingSource.HARNESS_EVIDENCE)})
        assert _dataflow_inputs(ctx,occ,{'container':'egg'})==[]
        assert _dataflow_inputs(ctx,occ,{'container':'cabinet_1'})==[]
    finally: system.close()


def test_CM05_roundtrip_old_trace_unknown_and_readonly_bank(tmp_path):
    system,ctx,occ,inv,p=setup(tmp_path,lambda *a:None)
    try:
        old=to_primitive(ctx.trace_builder.trace)
        assert trace_metrics(old) is None
        observe(ctx,system,p)
        digest=system.knowledge_digest()
        store=TraceStore(tmp_path/'output')
        store.save_atomic(ctx.trace_builder.trace)
        loaded=store.load(ctx.trace_builder.trace.trace_id)
        assert loaded.metadata['compiler_observability']==ctx.trace_builder.trace.metadata['compiler_observability']
        assert store.save_atomic(loaded).exists()
        rows=[{'task_id':ctx.task_id,'compiler_diagnostics':trace_metrics(loaded)}]
        write_reports(rows,tmp_path/'metrics')
        assert system.knowledge_digest()==digest
        assert len(list((tmp_path/'metrics').iterdir()))==3
    finally: system.close()


@pytest.mark.parametrize('entry', ['transaction', 'direct', 'runtime_trial'])
@pytest.mark.parametrize('update,consumed', [('unchanged', True), ('recertified', False),
                                           ('overwritten', False), ('identical', True)])
def test_CM04_effective_preflight_lineage_and_execution_invariance(tmp_path, monkeypatch, entry, update, consumed):
    from experiments.r10_world_checks import install_fixture
    from atomic_skillgraph.core.bindings import (BindingExpression, BindingExprKind, RuntimeBinding,
                                                BindingSource, BindingStatus, BindingResolution)
    from atomic_skillgraph.core.contracts import SemanticPredicate
    from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    from atomic_skillgraph.runtime.implementation_runner import ImplementationRunner
    records = []
    for enabled in (False, True):
        ids = iter(range(1, 10000))
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(ids)))
        system, ctx, _, _, provider = setup(tmp_path/str(enabled), lambda *a: pytest.fail('unexpected model call'))
        try:
            role = BindingExpression(BindingExprKind.SKILL_INPUT, source_role='target')
            atomic, impl = install_fixture(system, 'move_test', [],
                [SemanticPredicate('agent.at_location', {'location': role})], [('GO_TO', 'destination')],
                source_target='countertop_1')
            occ = RuntimeOccurrence('move_test', 'move_test', atomic.ref, [], {}, [impl.ref], atomic.effects)
            producer = replace(occ, occurrence_id='producer', step_id='producer')
            ctx.plan.occurrences.extend([producer, occ])
            ctx.plan.data_edges = [GraphEdge('flow', GraphEdgeType.DATA_FLOW, 'producer', occ.step_id,
                                            source_role='value', target_role='target')]
            ctx.begin_occurrence(occ); ctx.budget.begin_node(occ.occurrence_id)
            binding = RuntimeBinding('value', 'countertop_1', 'entity', BindingSource.TOOL_OUTPUT,
                BindingStatus.GROUNDED, BindingResolution.CONCRETE, ['witness'], ctx.world_revision)
            ctx.binding_store.publish_validated_outputs('producer', {'value': binding.value}, ['witness'],
                ctx.world_revision, certified_bindings={'value': binding})
            ctx.binding_store.apply_data_flow(ctx.plan, occ.step_id, revision=ctx.world_revision)
            original = ctx.binding_store.snapshot_for_node(occ)['target']
            compiled = system.invocation_compiler.compile_candidates(occ, ctx.binding_store, task_id=ctx.task_id)[0]
            # Certify the alternate input in a fresh scope. The normal Agent
            # path must still reject replacing a concrete DATA_FLOW anchor;
            # this case tests the transaction's already-certified update only.
            certification_scope = replace(occ, occurrence_id='fresh_scope') if update == 'overwritten' else occ
            preflight = system.invocation_compiler.preflight(compiled,
                call_name=compiled.spec.name, call_id='controlled',
                arguments={'target': 'countertop_2' if update == 'overwritten' else 'countertop_1'},
                occurrence=certification_scope, binding_store=ctx.binding_store, evidence_store=ctx.evidence_store,
                revision=ctx.world_revision, arguments_are_agent_proposals=update in {'recertified', 'overwritten'})
            assert preflight.passed, preflight
            if update == 'identical':
                preflight = replace(preflight, binding_updates=[original])
            elif update == 'unchanged':
                assert preflight.binding_updates == [original]
                # Registered runners can retain the already committed input;
                # trial requires its certified inputs in the preflight itself.
                if entry != 'runtime_trial':
                    preflight = replace(preflight, binding_updates=[])
            else:
                assert preflight.binding_updates[0].source == BindingSource.HARNESS_EVIDENCE
            if enabled:
                observe(ctx, system, provider)
            runner = ImplementationRunner(system.validation)
            if entry == 'transaction':
                result = execute_invocation(runner, compiled, preflight, occ, ctx, agent_prepared=True)
            else:
                result = runner.run(compiled, preflight, occ, ctx, agent_prepared=True,
                                    execution_scope='runtime_trial' if entry == 'runtime_trial' else 'registered')
            assert result.started and result.completed and result.atomic_effect_passed, result
            trace = ctx.trace_builder.trace
            if enabled:
                rows = trace.metadata['compiler_observability']['dataflow_consumptions']
                assert bool(rows) == consumed
                if rows:
                    assert rows[0]['edge_id'] == 'flow'
                    assert rows[0]['consumer_invocation_id'] == trace.implementation_invocations[0].attempt_id
            records.append(to_primitive(dict(requests=provider.requests, preflight=preflight,
                actions=trace.environment_actions, validations=trace.validations,
                invocations=trace.implementation_invocations, result=result,
                usage=[e.to_dict() for e in system.usage.events], world=ctx.harness._runtime_state_digest())))
        finally:
            system.close()
    assert records[0] == records[1]


def test_CM06_all_task_denominator_and_no_duplicate_cost_or_replay_deployment(tmp_path):
    from experiments.report import trace_to_row, summarize_traces
    system,ctx,occ,inv,p=setup(tmp_path,lambda r,n:action(r,'GO_TO',destination='cabinet_1'))
    try:
        observe(ctx,system,p)
        run_runtime_step(system.orchestrator.node_executor,'preparation',occ,ctx,inv,[])
        t=ctx.trace_builder.trace
        t.llm_usage=[e.to_dict() for e in system.usage.events]
        finalize(t)
        row=trace_to_row(t)
        assert row['compiler_diagnostics']['residual_runtime_decisions'] == 1
        assert row['compiler_diagnostics']['runtime_protocol_repairs'] == 0
        old=to_primitive(t);old['metadata'].pop('compiler_observability');old['trace_id']='other'
        oldrow=trace_to_row(old)
        summary=summarize_traces([row,oldrow])
        m=aggregate([row,oldrow],resource_summary={'total_tokens':summary['total_tokens']})
        assert m['task_count']==2 and m['unknown_tasks']==1
        assert summary['total_tokens']==2*sum(e.usage.total_tokens for e in system.usage.events)
        assert m['resource_accounting']['total_tokens']==summary['total_tokens']
        assert m['non_source_deployment'] is None
        assert m['autonomous_complete_success_rate'] is None
    finally: system.close()


def test_CM06_real_session_types_repairs_missing_and_deduplication(tmp_path):
    from atomic_skillgraph.traces.schema import AgentSessionRecord
    system, ctx, _, _, p = setup(tmp_path, lambda *a: None)
    try:
        observe(ctx, system, p)
        t = ctx.trace_builder.trace
        assert trace_metrics(t)['residual_runtime_decisions'] == 0
        t.agent_sessions = [AgentSessionRecord(str(i), kind, 'node', 0, snapshot={
            'runtime_request_context_audits': [{'request_sequence': 1, 'repair_in_progress': False}]})
            for i, kind in enumerate(('RuntimePreparationSession', 'SeededSession', 'DynamicTaskSession'))]
        t.metadata['runtime_steps'] = [{'session_id': s.session_id} for s in t.agent_sessions]
        assert trace_metrics(t)['residual_runtime_decisions'] == 3
        t.agent_sessions = t.agent_sessions[:1]
        t.metadata['runtime_steps'] = [{'session_id': '0'}]
        audits = t.agent_sessions[0].snapshot['runtime_request_context_audits']
        audits.extend([audits[0].copy(), {'request_sequence': 2, 'repair_in_progress': True}])
        assert trace_metrics(t)['residual_runtime_decisions'] == 1
        assert trace_metrics(t)['runtime_protocol_repairs'] == 1
        t.agent_sessions[0].snapshot = {}
        metrics = trace_metrics(t)
        assert metrics['residual_runtime_decisions'] is None
        assert metrics['runtime_protocol_repairs'] is None
        assert metrics['runtime_request_capture_status'] == 'partial'
        assert metrics['runtime_request_missing_reasons'] == ['0:runtime_request_audits_missing']
        t.agent_sessions = []
        assert trace_metrics(t)['runtime_request_missing_reasons'] == ['0:runtime_session_missing']
        t.metadata.pop('compiler_observability')
        assert trace_metrics(t) is None
    finally:
        system.close()
