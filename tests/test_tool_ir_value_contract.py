"""Public, benchmark-neutral IR projections/counts on the production interpreter."""
import copy
import pytest
from atomic_skillgraph.agents.protocol import validate_schema_instance
from atomic_skillgraph.agents.structured_submission import (
    TOOL_ACTION_BINDING_SCHEMA, TOOL_IR_COLLECTION_SOURCE_SCHEMA, TOOL_PROPOSAL_SCHEMA)
from atomic_skillgraph.core.contracts import ParameterSpec, ToolAsset
from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.validation.tool_validator import ToolValidator
from atomic_skillgraph.tooling.entry_contract import parameter_schema
from atomic_skillgraph.tooling.ir import ToolExecutionState, resolve_collection
from atomic_skillgraph.tooling.proposal import tool_proposal_from_dict
from atomic_skillgraph.tooling.value_reference import (
    resolve_tool_value_reference, validate_value_contract, worst_case_actions)
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from test_r92_tool_ir_public_contract import _runtime_context, _catalog_atomic_and_proposal


def fixture(kind='count'):
    atomic, proposal = _catalog_atomic_and_proposal()
    role, typ = ('steps','integer') if kind == 'count' else ('jobs','array')
    atomic.inputs.append(ParameterSpec(role,typ))
    proposal.inputs = list(atomic.inputs)
    proposal.input_schema = parameter_schema(atomic.inputs)
    loop = proposal.program[0]
    if kind == 'count':
        proposal.input_schema['properties']['steps'].update(minimum=0, maximum=2)
        loop['collection_source'] = {'source':'bounded_count','count':{'source':'tool_input','field':'steps'}}
        ref = {'kind':'skill_input','source_role':'destination'}
    else:
        proposal.input_schema['properties']['jobs'].update(maxItems=2, items={'type':'object',
            'required':['to'], 'additionalProperties':False, 'properties':{'to':{'type':'string'}}})
        loop['collection_source'] = {'source':'tool_input','field':'jobs'}
        ref = {'kind':'local_variable','source_role':'candidate_destination','field_path':['to']}
    loop['body'][0]['argument_mapping']['destination'] = ref
    loop['body'][0]['expected_effects'][0]['args']['location'] = copy.deepcopy(ref)
    tool = ToolAsset(ToolRef('bounded_navigation','1.0.0'), proposal.summary,
        proposal.input_schema, {'entry_contract':proposal.entry_contract,'output_schema':parameter_schema(atomic.outputs)},
        'tool_ir_v1', {'schema_version':1,'value_contract_version':2,'max_actions':2,
            'program':proposal.program,'final_effects':to_primitive(atomic.effects)}, [],
        {'reviewed':True,'zero_llm':True,'allowed_action_types':['GO_TO']}, {}, {})
    return atomic, proposal, tool


@pytest.mark.parametrize('kind', ['count','jobs'])
def test_full_static_and_runtime_value_contract(kind):
    _, ctx, occurrence = _runtime_context()
    atomic, proposal, tool = fixture(kind)
    report = ToolStaticValidator().validate_proposal(proposal,atomic,ctx.harness)
    assert report.passed, report
    assert ToolStaticValidator().validate_tool_asset(tool,atomic,ctx.harness).passed
    arguments = {'destination':'desk_1'}
    arguments.update(steps=1) if kind == 'count' else arguments.update(jobs=[{'to':'drawer_2'},{'to':'desk_1'}])
    result = ToolRunner(ToolValidator()).run(tool,arguments,ctx,occurrence_id=occurrence.occurrence_id)
    assert result.completed, result
    assert result.executed_action_count == (1 if kind == 'count' else 2)
    assert result.output_candidates['destination'] == 'desk_1'
    if kind == 'jobs':
        assert [v['value'] for v in result.tool_path_evidence['value_reference_observations']] == ['drawer_2','desk_1']


@pytest.mark.parametrize('kind,value', [('count',3),('count',100000),('count',-1),('count',True),
    ('count',1.5),('jobs',[{'to':'cabinet_1'}]*3),('jobs',[{'missing':'x'}])])
def test_invalid_inputs_rejected_before_any_environment_action(kind,value):
    _,ctx,occ = _runtime_context()
    _,_,tool = fixture(kind)
    before = len(ctx.trace_builder.trace.environment_actions)
    args = {'destination':'sidetable_1', 'steps' if kind=='count' else 'jobs':value}
    result = ToolRunner(ToolValidator()).run(tool,args,ctx,occurrence_id=occ.occurrence_id)
    assert not result.started
    assert result.failure_code == 'tool_input_schema_invalid'
    assert len(ctx.trace_builder.trace.environment_actions) == before


@pytest.mark.parametrize('path', [[],['..'],['*'],['x.y'],['x[0]'],[0],['0'],['x()'],['a']*5])
def test_unsafe_path_rejected_in_both_native_schema_and_runtime(path):
    ref = {'kind':'local_variable','source_role':'pair','field_path':path}
    with pytest.raises(ValueError): validate_schema_instance(ref,TOOL_ACTION_BINDING_SCHEMA)
    with pytest.raises(ValueError): resolve_tool_value_reference(ref,ToolExecutionState(local={'pair':{'left':'wire'}}))


def test_projection_is_object_only_fail_closed_and_legacy_scalar_unchanged():
    ref = {'kind':'local_variable','source_role':'pair','field_path':['nested','left']}
    for local in ({}, {'pair':{'nested':1}}, {'pair':{'nested':[]}}, {'pair':{'nested':{}}}):
        with pytest.raises(ValueError, match='tool_ir_value_reference_invalid'):
            resolve_tool_value_reference(ref,ToolExecutionState(local=local))
    state = ToolExecutionState(bindings={'input':'original'},local={'pair':{'nested':{'left':'wire'}}})
    assert resolve_tool_value_reference(ref,state) == 'wire'
    assert resolve_tool_value_reference({'kind':'skill_input','source_role':'input'},state) == 'original'
    assert resolve_tool_value_reference({'kind':'local_variable','source_role':'pair'},state) == state.local['pair']


def test_count_zero_and_no_clamp_or_boolean_conversion():
    source = {'source':'bounded_count','count':{'source':'tool_input','field':'steps'}}
    state = ToolExecutionState(bindings={'steps':0},max_actions=20)
    assert resolve_collection(source,state) == []
    state.bindings['steps'] = 8
    assert resolve_collection(source,state) == list(range(8))
    for bad in (True,3.5,-1,100000):
        state.bindings['steps'] = bad
        with pytest.raises(ValueError): resolve_collection(source,state)


@pytest.mark.parametrize('kind', ['count','jobs'])
def test_native_submission_roundtrip_and_input_schema_preservation(kind):
    atomic, proposal, _ = fixture(kind)
    payload = to_primitive(proposal)
    payload.pop('metadata')  # internal provenance is not an Agent-authored field
    validate_schema_instance(payload,TOOL_PROPOSAL_SCHEMA)
    parsed = tool_proposal_from_dict(payload)
    assert parsed.input_schema == proposal.input_schema
    assert parsed.program == proposal.program
    legacy = copy.deepcopy(payload)
    legacy.pop('input_schema')
    assert tool_proposal_from_dict(legacy).input_schema is None
    assert atomic.inputs == parsed.inputs


def test_static_missing_shape_scope_limits_and_worst_case_fail_closed():
    atomic, proposal, tool = fixture('jobs')
    schema, program = proposal.input_schema, proposal.program
    for mutate in (
        lambda s,p:s['properties']['jobs'].pop('maxItems'),
        lambda s,p:s['properties']['jobs'].update(maxItems=3),
        lambda s,p:p[0]['body'][0]['argument_mapping']['destination'].update(source_role='outside'),
        lambda s,p:p[0]['body'][0]['argument_mapping']['destination'].update(field_path=['missing']),
        lambda s,p:p[0]['body'].append({**copy.deepcopy(p[0]['body'][0]),'node_id':'another'}),
    ):
        s,p = copy.deepcopy(schema),copy.deepcopy(program)
        mutate(s,p)
        with pytest.raises(ValueError): validate_value_contract(p,s,2,strict=True)
    nested = [{'op':'FOR_EACH','max_iterations':3,'body':[{'op':'FOR_EACH','max_iterations':2,
        'body':[{'op':'ACTION'},{'op':'ACTION'}]}]}, {'op':'IF','then_branch':[{'op':'ACTION'}],
        'else_branch':[{'op':'ACTION'},{'op':'ACTION'}]}]
    assert worst_case_actions(nested) == 14


def test_count_schema_rejects_other_sources_or_extra_fields():
    for source in ({'source':'bounded_count'},
        {'source':'bounded_count','count':{'source':'local_variable','field':'steps'}},
        {'source':'bounded_count','count':{'source':'tool_input','field':'steps'},'values':[1]}):
        with pytest.raises(ValueError): validate_schema_instance(source,TOOL_IR_COLLECTION_SOURCE_SCHEMA)


def test_compiler_persists_input_contract_and_native_invocation_exposes_bounds():
    from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
    from atomic_skillgraph.tooling.proposal import ToolProvenance
    from atomic_skillgraph.core.status import SkillStatus, ToolStatus
    from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
    from test_r92_tool_ir_public_contract import _fixture_module
    _,ctx,_ = _runtime_context()
    atomic, proposal, _ = fixture('count')
    occurrence = _fixture_module()._occurrence('nav','navigate',
        {'destination':'desk_1','steps':1}, {'destination':'desk_1'}, atomic.effects[0],0)
    bundle = ToolCompiler().compile_proposal(occurrence,atomic,proposal,
        ToolProvenance('runtime_automation',str(atomic.ref),'trace','nav'), harness_profile='public_test_harness')
    assert bundle.tool.signature == proposal.input_schema
    assert bundle.tool.artifact['value_contract_version'] == 2
    assert bundle.implementation.compatibility['harness_profiles'] == ['public_test_harness']
    bundle.tool.status, bundle.implementation.status = ToolStatus.ACTIVE, SkillStatus.ACTIVE
    native = InvocationCompiler(None,None,ctx.harness).compile(atomic,bundle.implementation,[bundle.tool],{})
    validate_schema_instance({'destination':'desk_1','steps':1},native.input_schema)
    with pytest.raises(ValueError): validate_schema_instance({'destination':'desk_1','steps':3},native.input_schema)


def test_new_proposal_static_gate_cannot_skip_worst_case_via_small_input_value():
    _,ctx,_ = _runtime_context()
    atomic,proposal,tool = fixture('count')
    body = proposal.program[0]['body']
    body.append({**copy.deepcopy(body[0]),'node_id':'second_action'})
    report = ToolStaticValidator().validate_proposal(proposal,atomic,ctx.harness)
    assert report.failure_codes == ['tool_ir_worst_case_actions_exceeded']


def test_v2_declared_action_ceiling_must_fit_runtime_global_budget():
    _,ctx,occ = _runtime_context()
    _,_,tool = fixture('count')
    ctx.budget.global_action_budget = 1
    result = ToolRunner(ToolValidator()).run(tool,{'destination':'desk_1','steps':1},
        ctx,occurrence_id=occ.occurrence_id)
    assert not result.started
    assert result.failure_code == 'tool_input_schema_invalid'
    assert not ctx.trace_builder.trace.environment_actions


def test_empty_loop_body_cannot_hide_excessive_iteration_bound():
    _,proposal,_ = fixture('count')
    proposal.program[0]['body'] = []
    proposal.program[0]['max_iterations'] = 100000
    with pytest.raises(ValueError, match='max_iterations must fit max_actions'):
        validate_value_contract(proposal.program,proposal.input_schema,2,strict=True)


@pytest.mark.parametrize('extra', [{'source_step':'other_node'}, {'transform_id':'transform'}, {'constant':'hidden'}])
def test_projection_native_schema_and_resolver_reject_other_authorities(extra):
    ref = {'kind':'skill_input','source_role':'input','field_path':['left'],**extra}
    with pytest.raises(ValueError):
        validate_schema_instance(ref,TOOL_ACTION_BINDING_SCHEMA)
    with pytest.raises(ValueError):
        resolve_tool_value_reference(ref,ToolExecutionState(bindings={'input':{'left':'wire'}}))
