"""Author responsibility is a contract, not a historical-resolution heuristic."""
import copy
from dataclasses import replace

import pytest

from atomic_skillgraph.core.contracts import ParameterSpec, SemanticPredicate, TaskContract
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
from atomic_skillgraph.evolution.contract_canonicalizer import atomic_contract_signature
from test_stored_composite_binding_authority import _occurrence, _predicate


@pytest.mark.parametrize('resolvable', [False, True])
def test_resolution_responsibility_not_resolution_level(resolvable):
    from atomic_skillgraph.agents.structured_submission import PARAMETER_SPEC_SCHEMA
    from test_r1021_boundaries import typed_preparation_example, typed_atomicizer
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from atomic_skillgraph.evolution.extractor_session import ExtractorSession
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    from test_r1021_final import raw_proposal
    p,n=typed_preparation_example()
    p.input_specs=[replace(s,runtime_resolvable=resolvable) for s in p.input_specs]
    provider=ScriptedAgentProvider([FakeReply.structured({'occurrences':[raw_proposal(p)]})])
    session=ReplayAgentSession(provider,system_prompt='extractor',usage_ledger=UsageLedger(),usage_bucket='extractor_e1')
    proposed=ExtractorSession(session).propose_atomics(n)
    c,=typed_atomicizer().validate_and_canonicalize(proposed,n)
    assert all(s.runtime_resolvable is resolvable for s in c.input_specs)
    assert all(s['runtime_resolvable'] is resolvable for s in to_primitive(c.input_specs))
    assert not ParameterSpec('unchanged','entity').runtime_resolvable
    assert 'runtime_resolvable' not in PARAMETER_SPEC_SCHEMA['required']
    assert 'future invocation' in PARAMETER_SPEC_SCHEMA['properties']['runtime_resolvable']['description']
    # Permission cannot authenticate an input that did not exist at entry.
    n['boundary_authorities']['inputs'][0]['available_revision']=100
    with pytest.raises(ValueError): typed_atomicizer().validate_and_canonicalize(proposed,n)


def test_true_false_contracts_are_not_same_intent_aliases(tmp_path):
    from test_atomic_contract_canonicalizer import _take_candidate, _runtime
    from atomic_skillgraph.evolution.aligner import Aligner
    db,skills,tools,graph=_runtime(tmp_path)
    try:
        a,*_= _take_candidate('item','held','one')
        b=replace(a,inputs=[replace(s,runtime_resolvable=True) for s in a.inputs])
        assert atomic_contract_signature(a)!=atomic_contract_signature(b)
        aligner=Aligner(skills,tools)
        ar=aligner.align_atomic(a); br=aligner.align_atomic(b)
        assert ar!=br
        assert not skills.get_atomic(ar).inputs[0].runtime_resolvable
        assert skills.get_atomic(br).inputs[0].runtime_resolvable
    finally: db.close()


def test_composite_keeps_runtime_gap_and_required_edges():
    a=_occurrence('reach','reach',{'where':'cabinet_1'},{'reached':'cabinet_1'},
        _predicate('agent.at_location',location='where'),0)
    b=_occurrence('open','open',{'container':'cabinet_1'},{},
        _predicate('container.open',container='container'),1)
    edge=dict(edge_id='flow',edge_type='data_flow',source_step='reach',target_step='open',
        source_role='reached',target_role='container')
    proposal=CompositeExtractionProposal(['reach','open'],[],[edge],'access a container',{}, {})
    contract=TaskContract([SemanticPredicate('container.open',{'container':'cabinet_1'})])
    build=lambda:CompositeBuilder().validate_and_build(proposal,[a,b],contract)
    composite=build()
    assert composite.metadata['binding_origins']['reach']['where']=={'kind':'runtime'}
    assert not composite.occurrences[0].binding_specs
    b.input_specs=[replace(s,runtime_resolvable=False) for s in b.input_specs]
    assert build().occurrences[1].binding_specs['container'].source_step=='reach'
    a.input_specs=[replace(s,runtime_resolvable=False) for s in a.input_specs]
    with pytest.raises(ValueError,match='required non-runtime'):build()
    anchored=CompositeBuilder().validate_and_build(proposal,[a,b],
        TaskContract([SemanticPredicate('agent.at_location',{'location':'cabinet_1'}),
                      SemanticPredicate('container.open',{'container':'cabinet_1'})]),
        task_bindings={'location':'cabinet_1'})
    assert anchored.metadata['binding_origins']['reach']['where']['kind']=='task'
    a.input_specs=[replace(s,runtime_resolvable=True) for s in a.input_specs]
    proposal.new_edges=[]
    with pytest.raises(ValueError,match='explicit DataFlow'):build()


@pytest.mark.parametrize('output_resolution', ['semantic', 'concrete'])
def test_native_learning_to_persisted_composite_and_runtime(tmp_path, output_resolution):
    from test_r1021_i import source_case
    from test_r1021_final import raw_proposal
    from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal
    system,ctx,provider,_,n=source_case(tmp_path)
    try:
        # This fixture's goal is the actually observed OPEN effect, not a PASS stub.
        contract=TaskContract([SemanticPredicate('container.open',{'container':'cabinet_1'})])
        system.harness.task_contract=lambda task:contract
        authority=next(a for a in n['boundary_authorities']['inputs'] if a['value']=='cabinet_1' and a.get('available_revision')==0)
        proposals=[]
        for i,(intent,predicate,key) in enumerate([('reach_location','agent.at_location','location'),('open_container','container.open','container')],1):
            event=n['actions'][i]
            fact=next(f for f in event['authoritative_positive_effects'] if f['predicate']==predicate)
            p=AtomicOccurrenceProposal(intent,intent,event['event_index'],event['event_index'],
                {'target':'cabinet_1'},{'result':'cabinet_1'},[],
                [SemanticPredicate(predicate,{key:'cabinet_1'})],'actual public action',
                support_event_ids=[event['action_id']],effect_witness_refs=[fact['witness_ref']],
                input_provenance_refs={'target':{'authority_ref':authority['authority_ref'],'source_role':authority['role']}},
                output_derivations={'result':{'kind':'input_identity','input_role':'target'}},
                input_provenance_contract='code_authority_v3_2',boundary_schema_version='2',
                input_specs=[ParameterSpec('target','entity',runtime_resolvable=True,required_resolution='concrete')],
                output_specs=[ParameterSpec('result','entity',required_resolution=output_resolution)],
                output_semantic_constraints=({'result': {'compatible_with_input': 'target'}}
                    if output_resolution == 'concrete' else {}))
            proposals.append(raw_proposal(p))
        bad=copy.deepcopy(proposals[0]);bad['phase_id']='bad';bad['input_provenance_refs']['target']['authority_ref']='forged'
        def choose(request,_):
            payload=request.policy_context
            if 'canonical_trace' in payload:
                assert 'future use' in str(request.messages)
                return 'submit_extractor_atomics',{'occurrences':proposals+[bad]}
            if 'new_edge_candidates' in payload:
                ids=[c['candidate_id'] for c in payload['new_edge_candidates'] if c['edge_type']=='data_flow']
                return 'submit_extractor_composite',dict(selected_existing_edge_ids=[],selected_new_edge_candidate_ids=ids,
                    summary='reach and open a container',guideline={'steps':[],'notes':[]},insight={})
            if 'canonical_atomic' in payload:
                a=payload['canonical_atomic']; role=a['inputs'][0]['name']; out=a['outputs'][0]['name']
                nav=a['effects'][0]['predicate']=='agent.at_location'
                assert a['inputs'][0]['runtime_resolvable']
                return 'create_tool',dict(proposal_version='2',decision='create',summary='one public action',
                    atomic_ref=payload['atomic_ref'],inputs=a['inputs'],outputs=a['outputs'],
                    entry_contract={'conditions':[],'grounding_constraints':[]},
                    program=[dict(node_id='act',op='ACTION',action_type='GO_TO' if nav else 'OPEN',
                        argument_mapping={('destination' if nav else 'object'):{'kind':'skill_input','source_role':role}},expected_effects=a['effects']),
                        dict(node_id='return',op='RETURN',output_sources={out:{'source':'tool_input','field':role}})],
                    max_actions=1,final_effects=a['effects'],evidence_outputs=[],path_expectations=[],rationale='declared legal fixture')
            name=next(t.name for t in request.tools if t.name.startswith('invoke_impl_'))
            tool=next(t for t in request.tools if t.name==name)
            return name,{role:'cabinet_1' for role in tool.input_schema['properties']}
        provider.choose=choose
        trace=ctx.trace_builder.trace
        prepared=system._prepare_evolution(trace,ctx.task)
        assert trace.metadata['extraction']['e1_validated']==2 and trace.metadata['extraction']['e1_rejected']==1
        assert prepared.composite is not None,trace.metadata
        applied=system._apply_evolution(prepared,trace,ctx.task)
        stored=system.skills.get_composite(applied['composite_ref'])
        assert any(v=={'kind':'runtime'} for roles in stored.metadata['binding_origins'].values() for v in roles.values())
        assert len(stored.occurrences)==2 and stored.validator_spec['task_contract_covered']
        assert all(system.tools.get(ref).tests for ref in applied['tool_refs'])
        # Candidate remains Candidate. Explicit Agent invocations must work;
        # autonomous lifecycle activation is covered separately, not forged here.
        from atomic_skillgraph.core.results import RuntimeOccurrence
        from atomic_skillgraph.core.bindings import BindingExprKind
        occurrences=[RuntimeOccurrence(o.occurrence_id,o.step_id,o.node_ref,[],o.binding_specs,
            [r for r in system.skills.list_refs('implementation') if system.skills.get_implementation(r).abstract_ref==o.node_ref],
            system.skills.get_atomic(o.node_ref).effects) for o in stored.occurrences]
        plan=replace(ctx.plan,source='stored_composite',occurrences=occurrences,
            control_sequence=stored.control_sequence,data_edges=[e for e in stored.dependency_edges if e.edge_type.value=='data_flow'],task_contract=contract)
        system.harness.case=replace(system.harness.case,terminal_action='OPEN')
        system.planner.build_plan=lambda *args,**kwargs:plan
        runtime=system.orchestrator.run_task(ctx.task)
        assert [a.action_type for a in runtime.environment_actions]==['GO_TO','OPEN']
        assert runtime.node_records[0].direct_result['validated_outputs']
        second=runtime.node_records[1].direct_result['realized_bindings']
        # Candidate requires an explicit invocation and may refresh grounding;
        # its original DataFlow handoff and identity must nevertheless exist.
        assert all(v['value']=='cabinet_1' and v['evidence_refs'] for v in second.values())
        assert any('data_flow' in str(to_primitive(c)) for c in runtime.binding_changes)
    finally:system.close()
