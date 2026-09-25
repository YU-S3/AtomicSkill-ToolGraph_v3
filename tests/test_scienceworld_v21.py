import copy
import pytest
from atomic_skillgraph.agents.protocol import validate_schema_instance
from atomic_skillgraph.agents.structured_submission import TOOL_IR_COLLECTION_SOURCE_SCHEMA
from atomic_skillgraph.tooling.ir import COLLECTION_SOURCES, ToolExecutionState, resolve_collection, CONDITION_SOURCES
from atomic_skillgraph.tooling.runtime_interface import public_tool_ir_collection_sources
from atomic_skillgraph.tooling.ir_contract import collection_source_names
from atomic_skillgraph.harness.registry import resolve_benchmark_identity
from atomic_skillgraph.harness.scienceworld_public import container_evidence
from atomic_skillgraph.harness.public_discovery import PublicDiscoveryFrame, DiscoveryRecord
from atomic_skillgraph.harness.protocol import HarnessActionSpec
from atomic_skillgraph.deployment.scienceworld_reference import room_search, container_search


def test_collection_sources_have_one_authority():
    names = set(collection_source_names())
    assert names == set(TOOL_IR_COLLECTION_SOURCE_SCHEMA['properties']['source']['enum'])
    assert names == {r['source'] for r in public_tool_ir_collection_sources()} == COLLECTION_SOURCES
    state = ToolExecutionState(bindings={'xs':[1,2],'n':2},local={'ys':[3]},max_actions=2,
        catalog=[{'action_type':'OPEN','arguments':{'container':'box'}}],
        semantic_facts=[{'predicate':'seen','args':{'entity':'x'}}])
    sources = [{'source':'tool_input','field':'xs'}, {'source':'local_variable','field':'ys'},
        {'source':'action_catalog','project':{'kind':'argument','role':'container'},'refresh_each_iteration':True},
        {'source':'semantic_evidence','project':{'kind':'argument','role':'entity'}},
        {'source':'binding_evidence'}, {'source':'local_deterministic','values':[1]},
        {'source':'bounded_count','count':{'source':'tool_input','field':'n'}}]
    for source in sources:
        validate_schema_instance(source, TOOL_IR_COLLECTION_SOURCE_SCHEMA)
        assert isinstance(resolve_collection(source,state),list)
    for name in ('tool_input','local_variable','bounded_count','local_deterministic'):
        with pytest.raises(ValueError):
            validate_schema_instance({'source':name},TOOL_IR_COLLECTION_SOURCE_SCHEMA)


@pytest.mark.parametrize('config,expected', [({},'alfworld'),
    ({'experiment':{'benchmark':'scienceworld'},'harness':{'adapter':'scienceworld_v1'}},'scienceworld')])
def test_benchmark_identity(config,expected):
    assert resolve_benchmark_identity(config)['benchmark'] == expected


@pytest.mark.parametrize('benchmark,adapter', [('alfworld','scienceworld_v1'),
    ('scienceworld','alfworld_v3'),('unknown','alfworld_v3'),('scienceworld','unknown'),(None,'scienceworld_v1')])
def test_benchmark_mismatch_rejected(benchmark,adapter):
    with pytest.raises(ValueError):
        resolve_benchmark_identity({'experiment':{'benchmark':benchmark},'harness':{'adapter':adapter}})


def parse(observation,entities=('seed',)):
    action = HarnessActionSpec('look',0,'LOOK_IN',{'container':'box'},'', '', {})
    catalog = [HarnessActionSpec(str(i),1,'EXAMINE',{'entity':e},'', '', {}) for i,e in enumerate(entities)]
    frame = PublicDiscoveryFrame('fixture','episode',1,'hash',
        (DiscoveryRecord('box','room','public','r','in_room',1),),(),())
    return container_evidence(action,observation,catalog,frame,1,'episode')


@pytest.mark.parametrize('text,status', [('There is nothing in the box.','empty_listing'),
    ('Inside the box is: \n\tnothing','empty_listing'),
    ('Inside the box is:\n','unparsed'),
    ('Inside the other box is:\n\tnothing','unparsed'),
    ('Inside the box is:\n\ta seed','complete_listing'),
    ('Inside the box is:\n\tunknown prose','unparsed'),
    ('The box is closed.','inaccessible'),('Ambiguous request: choose','ambiguous')])
def test_container_inspection_status_is_not_inferred(text,status):
    facts,row = parse(text)
    assert row['status'] == status
    assert any(f['predicate']=='container.inspected' for f in facts) == (status in {'empty_listing','complete_listing'})
    assert any(f['predicate']=='entity.in_container' for f in facts) == (status=='complete_listing')


def test_container_ambiguous_aliases_do_not_prove_concrete_identity():
    facts,row = parse('Inside the box is:\n\ta seed, dry',('seed','seed, dry'))
    assert row['status'] == 'ambiguous'
    assert not facts


def test_failed_container_inspection_uses_revisions_not_trace_indices():
    from atomic_skillgraph.runtime.container_search_observer import checked_scopes
    _,_,tool = container_search()
    state = ToolExecutionState(bindings={'containers':['box'],'query':'seed'})
    # After a rollback, immutable action indices and live world revisions differ.
    state.iteration_observations = [{'node_id':'authorized_containers','value':'box',
        'action_start':100,'action_end':102,'revision_start':4,'revision_end':6,
        'condition_start':0,'condition_end':0}]
    frame = {'container':'box','revision':6,'status':'unparsed','source_ref':'public:6'}
    rows = checked_scopes(tool.artifact['program'],state,[frame])[3]
    assert rows[0]['inspection_status'] == 'unparsed'
    assert rows[0]['source_refs'] == ('public:6',)
    assert rows[0]['outcome'] == 'incomplete_inspection'
    assert rows[0]['action_indices'] == (100,101)
    stale = {**frame,'revision':4,'status':'empty_listing'}
    assert checked_scopes(tool.artifact['program'],state,[stale])[3][0]['source_refs'] == ()


def test_partial_container_listing_never_proves_absence():
    facts,row = parse('Inside the box is:\n\ta seed\n\tan unidentified object')
    assert row['status'] == 'unparsed'
    assert [f['predicate'] for f in facts] == ['entity.in_container','entity.discovered_at']


def test_room_and_container_input_authority():
    a,i,t = room_search()
    assert [p.name for p in a.inputs] == ['query','locations']
    assert t.signature['properties']['locations']['maxItems'] == 16
    validate_schema_instance({'query':'target','locations':[f'room{i}' for i in range(16)]},t.signature)
    with pytest.raises(ValueError):
        validate_schema_instance({'query':'target','locations':[f'room{i}' for i in range(17)]},t.signature)
    a,i,t = container_search()
    assert t.artifact['max_actions'] == 17
    assert t.signature['properties']['containers']['maxItems'] == 8


def test_exact_action_guard_resolves_current_local_without_fuzzy_matching():
    from atomic_skillgraph.tooling.ir import evaluate_condition
    from atomic_skillgraph.agents.structured_submission import TOOL_IR_CONDITION_SCHEMA
    condition = {'op':'exists','match':{'source':'action_catalog','where':{
        'action_type':'OPEN','container':{'source':'local_variable','field':'box'}},
        'project':{'kind':'argument','role':'container'}}}
    validate_schema_instance(condition,TOOL_IR_CONDITION_SCHEMA)
    state = ToolExecutionState(local={'box':'box'},catalog=[{'action_id':'red','revision':0,'action_type':'OPEN','arguments':{'container':'red box'}}])
    assert not evaluate_condition(condition,state)
    state.catalog.append({'action_id':'exact','revision':0,'action_type':'OPEN','arguments':{'container':'box'}})
    assert evaluate_condition(condition,state)
    state.local.clear()
    with pytest.raises(ValueError): evaluate_condition(condition,state)


def test_identity_bundle_zero_action_r1_preserves_semantic_resolution():
    from atomic_skillgraph.deployment.scienceworld_reference import workflow_entry
    from atomic_skillgraph.validation.atomic_validator import AtomicValidator
    from atomic_skillgraph.validation.tool_validator import ToolValidator
    from atomic_skillgraph.runtime.tool_runner import ToolRunner
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from test_r92_tool_ir_public_contract import _runtime_context
    _,ctx,_ = _runtime_context()
    for arity in (1,2,3):
        atomic,impl,tool = workflow_entry(arity)
        arguments = {p.name:'query' for p in atomic.inputs if p.name.startswith('query')}
        arguments.update(locations=['room'],wait_steps=1)
        occurrence = RuntimeOccurrence('bundle','bundle',atomic.ref,[],{},[],[])
        result = ToolRunner(ToolValidator()).run(tool,arguments,ctx,occurrence_id='bundle')
        assert result.completed and result.executed_action_count==0
        validator = AtomicValidator()
        r1 = validator.validate_execution_result(atomic,occurrence,arguments,result.output_candidates,
            ctx.harness.validator_channel(),current_revision=ctx.world_revision)
        assert r1.passed,r1
        assert all(b.resolution.value=='semantic' for b in r1.validated_output_bindings.values())
        assert r1.witness_refs[0].startswith('input_identity_validation:')
        ctx.binding_store.publish_validated_outputs(occurrence,result.output_candidates,r1.witness_refs,
            ctx.world_revision,certified_bindings=r1.validated_output_bindings)
        changed = {**result.output_candidates,'locations':['different room']}
        assert not validator.validate_execution_result(atomic,occurrence,arguments,changed,
            ctx.harness.validator_channel(),current_revision=ctx.world_revision).passed
        atomic.outputs[0].required_resolution = 'concrete'
        assert not validator.validate_execution_result(atomic,occurrence,arguments,result.output_candidates,
            ctx.harness.validator_channel(),current_revision=ctx.world_revision).passed
        # Empty effects never authorizes an arbitrary action/fresh output Tool.
        tool.artifact['program'].insert(0,{'op':'ACTION','node_id':'hidden','action_type':'WAIT1','argument_mapping':{}})
        assert not ToolValidator().validate_asset(tool).passed


def test_workflow_closure_requires_required_forward_explicit_dataflow(monkeypatch):
    from types import SimpleNamespace as NS
    from atomic_skillgraph.core.contracts import ParameterSpec
    from atomic_skillgraph.core.bindings import BindingExpression
    from atomic_skillgraph.deployment import workflow_entry_closure as audit
    output = ParameterSpec('value','entity',required_resolution='semantic')
    source = NS(inputs=[ParameterSpec('query','entity',required_resolution='semantic',runtime_resolvable=True)],outputs=[output])
    target = NS(inputs=[ParameterSpec('query','entity',required_resolution='semantic',runtime_resolvable=True)],outputs=[])
    source.ref,target.ref = 'a','b'
    skills = NS(get_atomic=lambda ref:{'a':source,'b':target}[ref])
    first = NS(step_id='first',node_ref='a',binding_specs={})
    second = NS(step_id='second',node_ref='b',binding_specs={
        'query':BindingExpression('data_flow',source_step='first',source_role='value')})
    edge = NS(target_step='second',target_role='query',source_step='first',source_role='value',origin='registered')
    graph = NS(ref='g',occurrences=[first,second],control_sequence=['first','second'],data_edges=[edge])
    monkeypatch.setattr(audit,'static_closure',lambda *args:{'program_static_closure':True,
        'nodes':[{'step_id':s,'available_implementations':['i'],'available_count':1} for s in ('first','second')]})
    result = audit.audit_workflow(skills,graph,None)
    assert result['post_bootstrap_program_closed']
    assert result['nodes'][0]['input_sources']=={'query':'bootstrap_declared'}
    assert result['nodes'][1]['input_sources']=={'query':'data_flow'}
    assert result['goal_completion_authority_not_inferred']
    output.required=False
    assert not audit.audit_workflow(skills,graph,None)['post_bootstrap_program_closed']
    output.required=True
    edge.origin='planner_proposed'
    assert not audit.audit_workflow(skills,graph,None)['post_bootstrap_program_closed']
    edge.origin='registered'
    graph.data_edges=[]
    assert not audit.audit_workflow(skills,graph,None)['post_bootstrap_program_closed']
    second.binding_specs={}
    assert audit.audit_workflow(skills,graph,None)['downstream_runtime_agent_required_count']==1
