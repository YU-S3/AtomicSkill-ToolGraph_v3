"""Source wiring through real TraceBuilder, Normalizer, System and replay.

Only model choices and the public environment are fixtures; no evidence,
schema, compiler, source authority or execution validator is stubbed true.
"""
import copy
from dataclasses import replace

import pytest

from atomic_skillgraph.core.contracts import ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.errors import AtomicSkillGraphError, FailureLayer
from atomic_skillgraph.core.refs import canonical_json
from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.evolution.typed_boundary import public_value_authorities
from atomic_skillgraph.runtime.checkpoint import capture, restore
from atomic_skillgraph.runtime.runtime_step import run_runtime_step
from atomic_skillgraph.system import AtomicSkillGraphSystem
from test_r10_runtime import setup, action
from test_r1021_boundaries import typed_atomicizer


def source_case(tmp_path, *, sparse=False, support_late=False):
    system, ctx, owner, invocations, provider = setup(tmp_path, lambda r, n:
        action(r, 'GO_TO', destination='countertop_1') if n == 1 else
        action(r, 'GO_TO', destination='cabinet_1') if n <= 2+int(sparse) else action(r, 'OPEN'))
    step = lambda: run_runtime_step(system.orchestrator.node_executor, 'preparation', owner, ctx, invocations, [])
    step()  # genuine accepted prefix creates P, removed by capability navigation
    if sparse:
        saved = capture(ctx, owner.occurrence_id)
        step()
        restore(ctx, saved, 'declared_rollback_fixture')
    step(); step()
    trace = ctx.trace_builder.trace
    trace.metadata['method_patch'] = '3.2'
    n = TraceNormalizer().build(trace)
    authorities = public_value_authorities(trace)
    n['boundary_authorities']['inputs'].extend(authorities)
    source = next(a for a in authorities if a['value']=='cabinet_1' and a['available_revision']==0)
    fact = next(f for f in n['actions'][-1]['authoritative_positive_effects'] if f['predicate']=='container.open')
    n['boundary_authorities']['effects'] = [copy.deepcopy(fact)]
    start, end = n['actions'][1]['event_index'], n['actions'][-1]['event_index']
    p = AtomicOccurrenceProposal('access', 'make_container_accessible', start, end,
        {'container':'cabinet_1'}, {'accessible':'cabinet_1'}, [],
        [SemanticPredicate('container.open', {'container':'cabinet_1'})], 'accepted native preparation',
        support_event_ids=[a['action_id'] for a in (n['actions'][-1:] if support_late else n['actions'][1:])],
        effect_witness_refs=[fact['witness_ref']],
        input_provenance_refs={'container':{'authority_ref':source['authority_ref'],'source_role':source['role']}},
        output_derivations={'accessible':{'kind':'input_identity','input_role':'container'}},
        input_provenance_contract='code_authority_v3_2', boundary_schema_version='2',
        input_specs=[ParameterSpec('container','entity',required_resolution='concrete')],
        output_specs=[ParameterSpec('accessible','entity',required_resolution='concrete')])
    c, = typed_atomicizer().validate_and_canonicalize([p], n)
    return system, ctx, provider, c, n


def author(request, *, create=False):
    ctx=request.policy_context
    a=ctx['canonical_atomic']
    result=dict(proposal_version='2', decision='no_tool', summary='bounded source capability',
        atomic_ref=ctx['atomic_ref'], inputs=a['inputs'], outputs=a['outputs'],
        entry_contract={'conditions':[],'grounding_constraints':[]}, program=[], max_actions=1,
        final_effects=[], evidence_outputs=[], path_expectations=[], rationale='controlled model choice')
    if create:
        result.update(decision='create',max_actions=2, final_effects=copy.deepcopy(a['effects']),program=[
            {'node_id':'arrive','op':'ACTION','action_type':'GO_TO',
             'argument_mapping':{'destination':{'kind':'skill_input','source_role':'container'}},
             'expected_effects':[{'predicate':'agent.at_location','args':{'location':'$container'},'effect_domain':'world'}]},
            {'node_id':'open','op':'ACTION','action_type':'OPEN',
             'argument_mapping':{'object':{'kind':'skill_input','source_role':'container'}},
             'expected_effects':[{'predicate':'container.open','args':{'container':'$container'},'effect_domain':'world'}]},
            {'node_id':'return','op':'RETURN','output_sources':{'accessible':{'source':'tool_input','field':'container'}}}])
    return 'create_tool', result


@pytest.mark.parametrize('sparse,support_late', [(False,False),(False,True),(True,True)])
def test_S01_S02_S03_S06_S09_S10_system_request_uses_real_declared_entry(tmp_path,sparse,support_late):
    system,ctx,provider,c,n=source_case(tmp_path,sparse=sparse,support_late=support_late)
    try:
        before=copy.deepcopy(n)
        c.action_events.reverse()  # submitted support order is not source order
        calls=len(provider.requests); actions=len(ctx.trace_builder.trace.environment_actions)
        provider.choose=lambda r,_:author(r)
        compiled,_=system._build_tool_for_occurrence(c,system._canonical_atomic_for_occurrence(c),n,ctx.trace_builder.trace)
        assert compiled is None and len(provider.requests)==calls+1
        request=provider.requests[-1]; payload=request.policy_context
        delta=payload['semantic_delta']; entry=n['actions'][1]
        assert 'before_state_facts' not in n and n==before
        assert delta['source_boundary']['entry_event_index']==c.event_start==entry['event_index']
        assert delta['source_boundary']['entry_before_revision']==1
        assert delta['source_boundary']['last_support_event_index']==c.event_end
        if support_late: assert payload['atomic_evidence_support'][0]['event_index']>c.event_start
        if sparse: assert c.event_start==2 and len(n['actions'])==3
        assert delta['source_input_bindings']==c.input_bindings
        assert [e['event_index'] for e in payload['atomic_evidence_support']]==sorted(e['event_index'] for e in c.action_events)
        assert delta['before_facts'] and 'after_facts' not in delta
        assert len(ctx.trace_builder.trace.environment_actions)==actions
        assert ctx.world_revision==3  # source entry is r1, live state already r3
        assert 'per-event changes, not a complete final-state' in request.messages[-1]['content']
        assert payload['tool_ir_schema']['collection_sources']
        assert request.tools[0].to_openai()['function']['parameters']['properties']['atomic_ref']['enum']==[payload['atomic_ref']]
    finally: system.close()


@pytest.mark.parametrize('damage',['missing','format','duplicate','bool_index','trace','owner','parameter_type','prefix','task'])
def test_S05_S17_corrupt_source_is_infrastructure_before_builder(tmp_path,damage):
    system,ctx,provider,c,n=source_case(tmp_path)
    try:
        c=copy.deepcopy(c);n=copy.deepcopy(n)
        if damage=='missing': del n['actions'][1]['authoritative_before_state_facts']
        elif damage=='format': n['actions'][1]['authoritative_before_state_facts']=None
        elif damage=='duplicate': n['actions'].append(copy.deepcopy(n['actions'][0]))
        elif damage=='bool_index': n['actions'][0]['event_index']=False
        elif damage=='trace': c.source_trace_id='wrong'
        elif damage=='owner': c.action_events[0]['span_id']='other_owner'
        elif damage=='parameter_type': c.action_events[0]['arguments']['destination']=True
        elif damage=='prefix': c.prefix_events.append(copy.deepcopy(n['actions'][0]))
        supplied_task=replace(ctx.task,task_id='wrong_source_task') if damage=='task' else ctx.task
        count=len(provider.requests)
        with pytest.raises(AtomicSkillGraphError) as error:
            system._build_tool_for_occurrence(c,system._canonical_atomic_for_occurrence(c),n,ctx.trace_builder.trace,source_task=supplied_task)
        assert error.value.code=='builder_source_integrity' and error.value.layer is FailureLayer.INFRASTRUCTURE
        assert len(provider.requests)==count
        r=ctx.trace_builder.trace.metadata['evolution_tool_builds'][-1]
        assert r['outcome']=='aborted' and not r['builder_entered'] and r['session_id']==''
    finally: system.close()


def test_S04_S07_S08_projection_empty_negative_and_json_types(tmp_path):
    from atomic_skillgraph.agents.context_builder import _compact_tool_builder_evidence
    system,ctx,_,c,n=source_case(tmp_path)
    try:
        evidence=_compact_tool_builder_evidence(c.action_events)
        assert any(e['authoritative_negative_effects'] for e in evidence)
        for raw,visible in zip(c.action_events,evidence):
            for field in ('authoritative_positive_effects','authoritative_negative_effects'):
                assert [(x['predicate'],x['args'],x['revision'],x['witness_ref']) for x in raw[field]]==[
                    (x['predicate'],x['args'],x['revision'],x['witness_ref']) for x in visible[field]]
        # Explicit corruption-free empty recorded view, unlike the missing case.
        n['actions'][1]['authoritative_before_state_facts']=[]
        c.input_bindings.update(flag=True,count=1,text='1')
        c.input_specs += [ParameterSpec(k,'value') for k in ('flag','count','text')]
        view=system._build_builder_source_context(c,n)
        assert view['before_facts']==[] and view['source_boundary']['entry_facts_status']=='recorded'
        assert canonical_json(view['source_input_bindings'])==canonical_json(c.input_bindings)
        assert type(view['source_input_bindings']['flag']) is bool
        assert type(view['source_input_bindings']['count']) is int
    finally: system.close()


def test_S12_S16_S18_native_multistep_builder_compile_source_replay(tmp_path):
    system,ctx,provider,c,n=source_case(tmp_path)
    try:
        provider.choose=lambda r,_:author(r,create=True)
        before=len(provider.requests)
        compiled,_=system._build_tool_for_occurrence(c,system._canonical_atomic_for_occurrence(c),n,ctx.trace_builder.trace,source_task=ctx.task)
        assert compiled is not None and len(provider.requests)==before+1
        case=compiled.tool.tests[0]
        assert case['prefix']==[{'action_type':'GO_TO','arguments':{'destination':'countertop_1'}}] and case['event_range']==[1,2]
        assert canonical_json(case['bindings'])==canonical_json(provider.requests[-1].policy_context['semantic_delta']['source_input_bindings'])
        outcome=system._replay_case_with_source_authority(compiled.tool,case,current_task=ctx.task,current_trace=ctx.trace_builder.trace,audit_trace=ctx.trace_builder.trace)
        assert outcome.passed and outcome.completed and outcome.executed_action_count==2
        assert len(provider.requests)==before+1  # replay uses no Agent
        bundle=system.aligner.stage_atomic(compiled.atomic,compiled.tool,compiled.implementation)
        renamed=system.aligner.atomic_canonicalizer.rewrite_canonical_occurrence(c,bundle,atomic_ref=bundle.atomic.ref)
        from atomic_skillgraph.evolution.tool_compiler import build_occurrence_replay_case
        rewritten=build_occurrence_replay_case(renamed,bundle.atomic,source_task=ctx.task)
        assert canonical_json(rewritten['bindings'])==canonical_json({bundle.input_role_map.get(k,k):v for k,v in c.input_bindings.items()})
        assert rewritten['prefix']==case['prefix']
    finally: system.close()


def test_current_system_typed_directory_to_builder_replay_and_admission(tmp_path):
    from test_r1021_final import raw_proposal
    system,ctx,provider,c,n=source_case(tmp_path)
    try:
        proposal=AtomicOccurrenceProposal('access','make_container_accessible',c.event_start,c.event_end,
            dict(c.input_bindings),dict(c.output_bindings),[],
            [SemanticPredicate('container.open', {'container':'cabinet_1'})],'declared native capability',
            support_event_ids=list(c.support_event_ids),effect_witness_refs=list(c.effect_witness_refs),
            input_provenance_refs={'container':{'authority_ref':c.input_provenance_refs['container']['authority_ref'],
                'source_role':c.input_provenance_refs['container']['role']}},
            output_derivations={'accessible':{'kind':'input_identity','input_role':'container'}},
            input_provenance_contract='code_authority_v3_2',boundary_schema_version='2',
            input_specs=c.input_specs,output_specs=c.output_specs)
        submitted=raw_proposal(proposal)
        bad=copy.deepcopy(submitted);bad['phase_id']='bad_sibling'
        bad['input_provenance_refs']['container']['authority_ref']='not_supplied'
        def choose(request,_):
            if 'canonical_trace' in request.policy_context:
                actual=request.policy_context['canonical_trace']['boundary_authorities']['inputs']
                assert actual and all(a['kind'] in {'public_binding','public_catalog'} for a in actual)
                assert any(a['value']=='cabinet_1' and a['available_revision']==0 for a in actual)
                assert any(a['kind']=='public_binding' for a in actual)
                assert all(type(a['available_revision']) is int for a in actual)
                return 'submit_extractor_atomics',{'occurrences':[submitted,bad]}
            return author(request,create=True)
        provider.choose=choose
        trace=ctx.trace_builder.trace
        prepared=system._prepare_evolution(trace,ctx.task)
        assert len(prepared.compiled)==1
        assert trace.metadata['extraction']['e1_validated']==1
        assert trace.metadata['extraction']['e1_rejected']==1
        result=system._apply_evolution(prepared,trace,ctx.task)
        assert result['tool_refs'] and result['atomic_refs']
        tool=system.tools.get(result['tool_refs'][0])
        assert tool.tests and tool.tests[0]['prefix']==[{'action_type':'GO_TO','arguments':{'destination':'countertop_1'}}]
        # Same pending certificate is usable before publication, without a new
        # reset/action; distinct content/program/authority cannot borrow it.
        from atomic_skillgraph.evolution.replay_certificates import ReplayCertificates
        from atomic_skillgraph.evolution.aligner import _tool_signature
        calls=[]
        original_reset=system.harness.reset
        def counted_reset(task):
            calls.append(task.task_id)
            return original_reset(task)
        system.harness.reset=counted_reset
        repeated=system._replay_case_with_source_authority(tool,tool.tests[0],
            current_task=ctx.task,current_trace=trace,audit_trace=trace)
        assert repeated.passed and repeated.stage=='certificate_reuse' and not calls
        assert not repeated.started and repeated.executed_action_count==0
        # Completed source/Trace publication uses exactly the existing ledger.
        trace.finish();system.traces.save_atomic(trace);system._commit_replay_certificates(trace)
        cert=ReplayCertificates(system.ledger);signature=_tool_signature(tool)
        assert cert.lookup(signature,tool.tests[0]) is not None
        assert cert.lookup(signature+'changed',tool.tests[0]) is None
        changed=copy.deepcopy(tool.tests[0]);changed['prefix']=[]
        assert cert.lookup(signature,changed) is None
        assert ReplayCertificates(system.ledger,authority_version='different').lookup(signature,tool.tests[0]) is None
    finally: system.close()
