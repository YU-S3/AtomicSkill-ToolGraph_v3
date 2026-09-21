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
