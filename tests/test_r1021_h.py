"""H: actual request wiring and finite schema contracts, not model compliance."""
import copy
import json
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents import ReplayAgentSession, UsageLedger
from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.agents.protocol import SchemaValidationError, validate_schema_instance
from atomic_skillgraph.agents.runtime_prompt_texts import SEARCH_POLICY, AUTOMATION_DRAFT_PROMPT
from atomic_skillgraph.agents.structured_submission import (
    TOOL_PROPOSAL_SCHEMA, BINDING_EXPRESSION_SCHEMA, TOOL_ACTION_BINDING_SCHEMA,
    COMPOSITE_EXTRACTION_SCHEMA, specialize_tool_proposal_schema,
    specialize_composite_selection_schema,
)
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.extractor_session import ExtractorSession, _composite_authority
from atomic_skillgraph.system import _SYSTEM_PROMPTS
from atomic_skillgraph.tooling.builder_session import ToolBuilderSession
from atomic_skillgraph.tooling.proposal import ToolProvenance, tool_proposal_from_dict
from atomic_skillgraph.tooling.runtime_interface import OUTPUT_SEMANTIC_CONSTRAINT_RULES
from experiments.fakes import FakeReply, ScriptedAgentProvider
from test_r1021_final import raw_proposal
from test_r1021_boundaries import typed_preparation_example, typed_atomicizer


def atomic(name='alpha', input_role='query', output_role='item'):
    return AbstractAtomicSkill(SkillRef(name, '1.0.0'), 'preserve a supplied identity',
        [ParameterSpec(input_role, 'entity', required_resolution='concrete')],
        [ParameterSpec(output_role, 'entity'), ParameterSpec('optional', 'boolean', required=False)],
        [], [SemanticPredicate('state.ready', {'object': '$'+output_role})],
        {'output_derivations': {output_role: {'kind': 'input_identity', 'input_role': input_role}}},
        [], {}, {})


INTERFACE = {'primitive_actions': [{'action_type':'ACT', 'argument_roles':['object']}],
             'predicate_vocabulary':[{'predicate':'state.ready', 'argument_roles':['object']}]}


def no_tool(a):
    return dict(proposal_version='2', decision='no_tool', summary='No justified program',
        rationale='No new safe bounded implementation is supported.', atomic_ref=str(a.ref),
        inputs=to_primitive(a.inputs), outputs=to_primitive(a.outputs),
        entry_contract={'conditions':[], 'grounding_constraints':[]}, program=[],
        max_actions=1, final_effects=[], evidence_outputs=[], path_expectations=[])


def session(provider, bucket='tool_builder_evolution'):
    return ReplayAgentSession(provider, system_prompt=_SYSTEM_PROMPTS['tool_builder'],
        usage_ledger=UsageLedger(), usage_bucket=bucket)


def test_H01_H02_H05_e1_actual_request_and_new_formal_role():
    p,n=typed_preparation_example()
    p.input_specs[0].name='new_query'
    p.input_roles['new_query']=p.input_roles.pop('target')
    p.input_provenance_refs['new_query']=p.input_provenance_refs.pop('target')
    for constraint in p.output_semantic_constraints.values():
        if constraint.get('compatible_with_input')=='target':
            constraint['compatible_with_input']='new_query'
    for authority in n['boundary_authorities']['inputs']:
        authority['source_role']='ancestral_role_not_the_port'
    provider=ScriptedAgentProvider([FakeReply.structured({'occurrences':[raw_proposal(p)]})])
    e=ExtractorSession(session(provider,'extractor_e1'))
    submitted=e.propose_atomics(n)
    assert typed_atomicizer().validate_and_canonicalize(submitted,n)
    request=provider.requests[0]
    text=request.messages[-1]['content']
    assert text.count(OUTPUT_SEMANTIC_CONSTRAINT_RULES)==1
    assert text.startswith('Extract independently useful')
    assert request.policy_context['canonical_trace']['boundary_authorities']==n['boundary_authorities']
    assert 'boundary_authorities' not in request.policy_context
    schema=request.tools[0].input_schema
    assert 'enum' not in schema['properties']['occurrences']['items']['properties']['input_specs']['items']['properties']['name']
    submitted[0].input_provenance_refs['new_query']['source_role']='ancestral_role_not_the_port'
    with pytest.raises(ValueError): typed_atomicizer().validate_and_canonicalize(submitted,n)
    empty=ScriptedAgentProvider([FakeReply.structured({'occurrences':[]})])
    assert ExtractorSession(session(empty,'extractor_e1')).propose_atomics(n)==[]
    assert len(empty.requests)==1


def test_H06_H09_H22_builder_request_isolation_and_complete_no_tool():
    base=copy.deepcopy(TOOL_PROPOSAL_SCHEMA)
    a,b=atomic(),atomic('beta','anchor','found')
    provider=ScriptedAgentProvider([FakeReply.tool('create_tool',no_tool(a)),FakeReply.tool('create_tool',no_tool(b))])
    s=session(provider);builder=ToolBuilderSession(s)
    for contract in (a,b):
        proposal=builder.build(atomic=contract, provenance=ToolProvenance(source='success_evolution',atomic_ref=str(contract.ref),source_trace_id='source',occurrence_id='owner'),harness_interface=INTERFACE)
        assert proposal.decision=='no_tool' and not proposal.program
    for request,contract in zip(provider.requests,(a,b)):
        schema=request.tools[0].input_schema
        assert schema['properties']['atomic_ref']['enum']==[str(contract.ref)]
        assert schema['properties']['inputs']['items']['properties']['name']['enum']==[contract.inputs[0].name]
        # ParameterSpec.name must not alias semantic_type or output names.
        assert 'enum' not in schema['properties']['inputs']['items']['properties']['semantic_type']
        assert schema['properties']['outputs']['items']['properties']['name']['enum']==[contract.outputs[0].name,'optional']
        validate_schema_instance(no_tool(contract),schema)
        assert request.messages[-1]['content'].count(OUTPUT_SEMANTIC_CONSTRAINT_RULES)==1
    assert TOOL_PROPOSAL_SCHEMA==base
    assert len(provider.requests)==2 and s.pending_tool_call is None


def test_H07_H08_H09_tool_syntax_optional_return_and_unchanged_graph():
    a=atomic();schema=specialize_tool_proposal_schema(TOOL_PROPOSAL_SCHEMA,a,INTERFACE)
    value=no_tool(a);value.update(decision='create',final_effects=to_primitive(a.effects),program=[
        {'node_id':'act','op':'ACTION','action_type':'ACT',
         'argument_mapping':{'object':{'kind':'skill_input','source_role':'query'}},
         'expected_effects':[{'predicate':'state.ready','args':{'object':'$query'}}]},
        {'node_id':'return','op':'RETURN','output_sources':{'item':{'source':'tool_input','field':'query'}}}])
    validate_schema_instance(value,schema)
    assert tool_proposal_from_dict(value).decision=='create'
    # No optional output is silently upgraded to required.
    for mutate in (
        lambda p:p.update(atomic_ref='skill://other@1.0.0'),
        lambda p:p['program'][0].update(action_type='NOT_PUBLIC'),
        lambda p:p['program'][0]['argument_mapping']['object'].update(kind='data_flow'),
        lambda p:p['program'][0]['expected_effects'][0].update(predicate='private.fact'),
        lambda p:p['program'][1]['output_sources'].update(undeclared={'source':'tool_input','field':'query'}),
        lambda p:p['program'][1]['output_sources'].pop('item'),
    ):
        bad=copy.deepcopy(value);mutate(bad)
        with pytest.raises(SchemaValidationError):validate_schema_instance(bad,schema)
    validate_schema_instance({'kind':'local_variable','source_role':'candidate'},TOOL_ACTION_BINDING_SCHEMA)
    validate_schema_instance({'kind':'constant','constant':True},TOOL_ACTION_BINDING_SCHEMA)
    with pytest.raises(SchemaValidationError):
        validate_schema_instance({'kind':'local_variable','source_role':'candidate'},BINDING_EXPRESSION_SCHEMA)
    assert 'data_flow' in BINDING_EXPRESSION_SCHEMA['properties']['kind']['enum']


def test_H12_composite_namespaces_empty_and_base_isolation():
    base=copy.deepcopy(COMPOSITE_EXTRACTION_SCHEMA)
    authority=SimpleNamespace(existing_edge_by_id={'existing':{}},new_edge_candidate_ids={'candidate'})
    schema=specialize_composite_selection_schema(base,authority)
    p={'selected_existing_edge_ids':['existing'],'selected_new_edge_candidate_ids':['candidate'],
       'summary':'Compose supported capabilities','guideline':{},'insight':{}}
    validate_schema_instance(p,schema)
    for existing,new in [(['candidate'],[]),([],['existing']),(['existing','existing'],[])]:
        with pytest.raises(SchemaValidationError):validate_schema_instance(dict(p,selected_existing_edge_ids=existing,selected_new_edge_candidate_ids=new),schema)
    empty=specialize_composite_selection_schema(base,_composite_authority([],[]))
    validate_schema_instance(dict(p,selected_existing_edge_ids=[],selected_new_edge_candidate_ids=[]),empty)
    with pytest.raises(SchemaValidationError): validate_schema_instance(p,empty)
    assert base==COMPOSITE_EXTRACTION_SCHEMA
    assert '"enum": []' not in json.dumps(empty)


def test_H07_H20_local_shape_is_not_a_scope_certificate():
    from test_r92_tool_ir_public_contract import _catalog_atomic_and_proposal, _runtime_context
    from atomic_skillgraph.tooling.validator import ToolStaticValidator
    from atomic_skillgraph.tooling.runtime_interface import public_primitive_action_schema, public_predicate_schema
    _,ctx,_=_runtime_context()
    a,p=_catalog_atomic_and_proposal()
    interface={'primitive_actions':public_primitive_action_schema(ctx.harness),
               'predicate_vocabulary':public_predicate_schema(ctx.harness)}
    schema=specialize_tool_proposal_schema(TOOL_PROPOSAL_SCHEMA,a,interface)
    raw={k:v for k,v in to_primitive(p).items() if k in schema['properties']}
    validate_schema_instance(raw,schema)
    assert ToolStaticValidator().validate_proposal(p,a,ctx.harness).passed
    # Same legal local operand shape outside the lexical body must still fail
    # the real static validator; shallow native schema is not an AST checker.
    p.program=[p.program[0]['body'][0],p.program[-1]]
    raw={k:v for k,v in to_primitive(p).items() if k in schema['properties']}
    validate_schema_instance(raw,schema)
    assert not ToolStaticValidator().validate_proposal(p,a,ctx.harness).passed


@pytest.mark.parametrize('scope',['node','initial','rescue','continuation'])
@pytest.mark.parametrize('lazy',[False,True])
def test_H14_H15_actual_context_keeps_scope_and_lazy_only(scope,lazy):
    c=ContextBuilder();interface={'source_occurrence_id':'owner','current_input_sources':[]}
    kw=dict(task_goal='generic goal',observation='public state',action_catalog=[],relevant_action_history=[],remaining_budget={'task_actions':7})
    if scope=='node':
        prompt=c.runtime_node(**kw,atomic_contract=atomic(),implementation_invocations=[],runtime_automation_interface=interface if lazy else {})
    else:
        prompt=c.dynamic_task(**kw,task_runtime_frame={'runtime_automation_interface':interface} if lazy else {},rescue_method_guidance={'history':scope} if scope!='initial' else None)
    # Release5 drafts have only their own submission scope, not node/task action instructions.
    assert prompt.count(SEARCH_POLICY)==int(not lazy)
    assert (AUTOMATION_DRAFT_PROMPT in prompt)==lazy
    assert prompt.count(OUTPUT_SEMANTIC_CONSTRAINT_RULES)==int(lazy)
    payload=json.loads(prompt.split('POLICY_CONTEXT_JSON\n')[1])
    assert bool(payload.get('runtime_automation_interface'))==lazy
    assert 'runtime_automation_interface' not in payload.get('task_runtime_frame',{})
    assert 'ACTION.argument_mapping' not in prompt


@pytest.mark.parametrize('invalid_first',[False,True])
def test_H19_specialized_schema_round_trip_through_real_provider_adapter(monkeypatch,invalid_first):
    from atomic_skillgraph.agents.provider import OpenAICompatibleProvider
    from test_deepseek_protocol import _config,_Response,_success
    monkeypatch.setenv('TEST_DEEPSEEK_KEY','test-only')
    a=atomic();body=no_tool(a);requests=[]
    def post(_url,*,headers,json,timeout):
        requests.append(json)
        result=_success('call_h_'+str(len(requests)),'create_tool')
        reply=dict(body,atomic_ref='skill://invented@1.0.0') if invalid_first and len(requests)==1 else body
        result['choices'][0]['message']['tool_calls'][0]['function']['arguments']=__import__('json').dumps(reply)
        return _Response(result)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    s=session(OpenAICompatibleProvider(_config()))
    proposal=ToolBuilderSession(s).build(atomic=a,provenance=ToolProvenance(source='runtime_automation',atomic_ref=str(a.ref),source_trace_id='source',occurrence_id='owner'),harness_interface=INTERFACE)
    sent=requests[0]['tools'][0]['function']
    assert sent['parameters']==specialize_tool_proposal_schema(TOOL_PROPOSAL_SCHEMA,a,INTERFACE)
    validate_schema_instance(body,sent['parameters'])
    assert 'strict' not in sent and 'response_format' not in requests[0]
    assert proposal.decision=='no_tool' and len(requests)==1+int(invalid_first)
    assert s.snapshot()['protocol_repairs_used']==int(invalid_first)
    assert s.pending_tool_call is None
