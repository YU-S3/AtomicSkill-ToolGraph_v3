"""V4 gates: intercepted Provider, actual production Broker, optional pinned Docker. No paid HTTP."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import digest, object_schema
from atomic_skillgraph.empirical.office_operation_mining import mine, bind_inputs, material, select, read, sha
from atomic_skillgraph.empirical.office_trace_program_induction import (RequestStore, ModelOutputError,
    validate_proposal, program_from_response, bundle)
from atomic_skillgraph.empirical.office_program_probe import (probe_cases, execute_probe, consumed,
    partial_workflow, verify_result, expected)
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.retrieval_chain import recover, authentic
from atomic_skillgraph.harness.simple_protocol import Broker
from atomic_skillgraph.experiments import run_office_trace_v4_pilot as runner
from atomic_skillgraph.experiments.office_v4_consumption import execute_action, snapshot, restore
from atomic_skillgraph.experiments.office_v4_offline import SEARCH_FIXTURE
from test_cf2_contracts import http, response
from test_office_trace_v3 import local_system, source


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs): raise AssertionError('Paid model HTTP is forbidden in this gate')
    monkeypatch.setattr('requests.sessions.Session.request', reject)


def fixture_case(tmp_path):
    system, task = local_system(tmp_path)
    trace = source(system, task)
    spans, opportunities = mine(trace, digest(trace), system.config['program_environment'])
    op = next(o for o in opportunities if o['origin'] == 'observed_local_step' and o['events'][-1]['name'] == 'read')
    return system, task, trace, spans, op


def proposal(op):
    return {'decision':'propose_program', 'source_opportunity_ids':[op['id']], 'evidence_limitations':'observed local read; task may be wrong',
        'atomic':{'goal':'Read a public corpus window','applicability':'known public path and Unicode offset',
            'preconditions':['path/offset obtained'],'result_role':'intermediate','stop_condition':'missing text'},
        'program_spec':{'input_schema':object_schema({'path':{'type':'string'},'offset':{'type':'integer'}},['path','offset']),
            'output_schema':object_schema({'text':{'type':'string'}},['text']), 'allowed_tools':['read'],
            'result_fields':['text'],'probe_contract':{'kind':'unverified'}}}


SOURCE = '''def run(ctx, inputs):
    result = ctx.call('read', inputs)
    if not result['accepted']: return {'status':'blocked','outputs':{},'detail':'read rejected'}
    return {'status':'ok','outputs':{'text':result['data']['text']}}
'''


def test_success_failure_projection_and_old_authentic_unchanged(tmp_path):
    s, task, trace, spans, op = fixture_case(tmp_path)
    original = deepcopy(trace)
    trace.update(score={'hard':False,'gold':'PRIVATE-GOLD'}, prediction='FAILED-ANSWER')
    trace['execution']['prediction'] = 'FAILED-ANSWER'
    projected, opportunities = mine(trace,digest(trace),s.config['program_environment'])
    encoded = json.dumps([projected,opportunities])
    assert 'PRIVATE-GOLD' not in encoded and 'FAILED-ANSWER' not in encoded
    assert any(o['origin']=='observed_local_step' for o in opportunities)
    chain = recover(task,original,s.config['program_environment'])[0]
    assert authentic(chain,{'events':original['tools']})
    mutated=deepcopy(original); mutated['tools'][-1]['arguments']['offset'] += 1
    assert not recover(task,mutated,s.config['program_environment'])
    assert any(o['origin']=='observed_local_step' for o in mine(mutated,digest(mutated),s.config['program_environment'])[1])
    bad=deepcopy(trace);bad['task']['split']='val'
    with pytest.raises(ValueError,match='Train'):mine(bad,digest(bad),{})
    s.close()


def test_exact_defaults_invalid_ids_extra_schema_and_api_feedback(tmp_path):
    s, task, trace, spans, op = fixture_case(tmp_path)
    p=proposal(op); candidate=validate_proposal(p,op,spans,['read'],[])
    assert candidate['bound_inputs']==spans[1]['arguments']
    p['program_spec']['input_bindings']={'path':{'span_id':'invented','section':'arguments','path':['path']}}
    with pytest.raises(ValueError,match='invalid_source_id'):validate_proposal(p,op,spans,['read'],[])
    p=proposal(op);p['program_spec']['input_schema']['properties']['query']={'type':'string'}
    p['program_spec']['input_schema']['required'].append('query')
    with pytest.raises(ValueError,match='missing_observed_control'):validate_proposal(p,op,spans,['read'],[])
    p=proposal(op);p['program_spec']['invented']=True
    with pytest.raises(ValueError,match='extra_schema_field'):validate_proposal(p,op,spans,['read'],[])
    with pytest.raises(ValueError,match='Context has no read_context'):
        program_from_response({'source':'def run(ctx, inputs):\n    return ctx.read_context(inputs)'},candidate,s.config['program_environment'],['read'])
    result=program_from_response({'source':SOURCE},candidate,s.config['program_environment'],['read'])
    assert result['allowed_tools']==['read']
    p=proposal(op)
    p['program_spec']['input_bindings']={'path':{'span_id':spans[1]['id'],'section':'arguments','path':['path'],'slice':{'start':0,'end':1}}}
    assert validate_proposal(p,op,spans,['read'],[])['bound_inputs']['path']=='a'
    for bad in (None, [], {'decision':'propose_program','source_opportunity_ids':[op['id']], 'evidence_limitations':'x','atomic':[]}):
        with pytest.raises(ValueError):validate_proposal(bad,op,spans,['read'],[])
    s.close()


def provider(tmp_path):
    p=OpenAICompatibleProvider(OpenAICompatibleConfig(base_url='https://example.invalid',model='deepseek-v4-flash',
        api_key_env='MODEL_API_KEY',max_completion_tokens=128,thinking_type='disabled',max_retries=0))
    p.budget_governor=BudgetGovernor(tmp_path/'budget.json',token_limit=100000,request_limit=20,finish_reserve=0)
    return p


def test_paid_response_resume_decode_failure_and_unknown_http(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response(finish='stop',content='{"decision":"no_change"}'),response(finish='stop',content='{invalid')])
    p=provider(tmp_path); client=RequestStore(tmp_path/'requests',p,phase='learn')
    first, receipt=client.ask('one','prompt',{'actual':'source'})
    assert first['decision']=='no_change' and receipt['requests'][0]['raw_usage']['total_tokens']==5
    assert client.ask('one','prompt',{'actual':'source'})[0]==first and len(seen)==1
    with pytest.raises(RuntimeError,match='identity_changed'):client.ask('one','other prompt',{'actual':'source'})
    for _ in range(2):
        with pytest.raises(ModelOutputError,match='invalid_json'):client.ask('two','prompt',{})
    assert len(seen)==2
    save_json(tmp_path/'requests/unresolved.json',{'payload_sha256':digest(p._build_payload([{'role':'system','content':'p'},
        {'role':'user','content':'{}'}],[])),'status':'prepared'})
    with pytest.raises(RuntimeError,match='unresolved_http'):client.ask('unresolved','p',{})
    assert len(seen)==2


@pytest.mark.parametrize('correct,decision,repair_case',[(True,'propose_program',False),(False,'propose_program',False),
    (False,'no_change',False),(False,'propose_program',True)])
def test_intercepted_production_learning_bundle_and_idempotency(tmp_path,monkeypatch,correct,decision,repair_case):
    s,task,trace,spans,op=fixture_case(tmp_path)
    trace['score']={'hard':correct}
    spans,opportunities=mine(trace,digest(trace),s.config['program_environment'])
    op=next(o for o in opportunities if o['origin']=='observed_local_step' and o['events'][-1]['name']=='read')
    root=tmp_path/'pilot';root.mkdir();(root/'pilot_bank').mkdir();(root/'programs').mkdir()
    cfg={'wire_max_bytes':65536,'source_window_chars':12000,'max_builder_generations':6}
    plan={'run_id':'fixture','config':cfg,'parent_bank':str(tmp_path/'oldbank'),
          'parent_frozen':runner.tree_identity(tmp_path/'oldbank')}
    save_json(root/'pilot_manifest.json',plan)
    save_json(root/'zero_http_check.json',{'passed':True,'manifest_sha256':sha(root/'pilot_manifest.json')})
    save_json(root/'authorization.json',{'approved':True,'user_authorization':'TEST ONLY intercepted transport',
        'manifest_sha256':sha(root/'pilot_manifest.json'),'phase_limits':runner.PHASE_LIMITS})
    ledger={'limits':{'token_limit':90000000,'request_limit':480000,'finish_reserve':0},'attempts':{},'unknown_billing':False}
    save_json(root/'budget/inherited.json',ledger);save_json(root/'budget/usage.json',ledger)
    runner.jsonl(root/'source_span_catalog.jsonl',spans);runner.jsonl(root/'opportunity_catalog.jsonl',[op])
    save_json(root/'selection.json',{'opportunities':[op['id']]});save_json(root/'parent_programs.json',[])
    monkeypatch.setattr(runner,'checked',lambda *a,**k:(plan,s.config))
    p=proposal(op) if decision=='propose_program' else {'decision':'no_change','source_opportunity_ids':[op['id']],
        'evidence_limitations':'no useful reusable operation','reason':'irrelevant'}
    responses=[response(finish='stop',content=json.dumps(p))]
    if decision=='propose_program':
        if repair_case:responses.append(response(finish='stop',content=json.dumps({'source':'def run(ctx, inputs):\n    return ctx.read_context(inputs)'})))
        responses.append(response(finish='stop',content=json.dumps({'source':SOURCE})))
    seen=http(monkeypatch,responses)
    if repair_case:
        original_bundle=runner.bundle
        def interrupted(*a,**k):raise RuntimeError('simulated interrupt after paid repair response')
        monkeypatch.setattr(runner,'bundle',interrupted)
        with pytest.raises(RuntimeError,match='simulated interrupt'):runner.learn(root)
        assert len(seen)==3
        monkeypatch.setattr(runner,'bundle',original_bundle)
    result=runner.learn(root)
    assert result['formal_promotions']==0
    before=len(seen);runner.learn(root);assert len(seen)==before
    bundles=runner.rows(root/'program_bundles.jsonl')
    assert len(bundles)==(decision=='propose_program')
    if bundles:
        assert bundles[0]['status']=='candidate_only' and bundles[0]['implementation']['status']=='candidate_only'
        assert bundles[0]['source_trace_sha256']==op['source_trace_sha256'] and not bundles[0]['outputs_consumed']
        altered=deepcopy(bundles[0]);altered['program']['source']+='\n# altered after generation'
        asset_path=root/'pilot_bank'/(altered['id']+'.json');save_json(asset_path,altered)
        with pytest.raises(RuntimeError,match='identity changed'):runner.verify_candidates(root)
        save_json(asset_path,bundles[0])
    s.close()


def test_four_paired_arms_no_false_consumption_and_zero_duplicate_http(tmp_path,monkeypatch):
    from atomic_skillgraph.experiments import office_v4_consumption as consumption
    s,task,trace,spans,op=fixture_case(tmp_path)
    root=tmp_path/'pairs';root.mkdir();(root/'pilot_bank').mkdir();(root/'programs').mkdir()
    tasks=[{**asdict(task),'task_id':'toy:'+str(i),'physical_key':'toy-key-'+str(i)} for i in range(2)]
    plan={'run_id':'paired_fixture','source':str(tmp_path/'source'), 'config':{'transfer_tasks':[t['task_id'] for t in tasks],
        'paired_arm_limits':{'tokens':200000,'http':10,'native_calls':48}},'parent_bank':str(tmp_path/'oldbank'),
        'parent_frozen':runner.tree_identity(tmp_path/'oldbank')}
    save_json(root/'pilot_manifest.json',plan)
    save_json(root/'authorization.json',{'approved':True,'user_authorization':'TEST intercepted only',
        'manifest_sha256':sha(root/'pilot_manifest.json'),'phase_limits':runner.PHASE_LIMITS})
    ledger={'limits':{'token_limit':90000000,'request_limit':480000,'finish_reserve':0},'attempts':{},'unknown_billing':False}
    save_json(root/'budget/inherited.json',ledger);save_json(root/'budget/usage.json',ledger)
    save_json(root/'source_records.json',[{'task':t} for t in tasks]);save_json(root/'parent_programs.json',[])
    labels=tmp_path/'private_labels.json';save_json(labels,{t['task_id']:{'answer':'7'} for t in tasks})
    save_json(tmp_path/'source/config.json',{'harness':{'evaluator_records':str(labels)}})
    candidate=validate_proposal(proposal(op),op,spans,['read'],[])
    program=program_from_response({'source':SOURCE},candidate,s.config['program_environment'],['read'])
    value=bundle(candidate,program,'fixture',{});value['status']='worker_probe_passed'
    save_json(root/'pilot_bank'/ (value['id']+'.json'),value);runner.index_outputs(root)
    monkeypatch.setattr(consumption,'checked',lambda *a,**k:(plan,s.config))
    # This Runtime test supplies an explicitly labelled fixture; learning provenance is tested above.
    monkeypatch.setattr(consumption,'verify_candidates',lambda *a:[value])
    seen=http(monkeypatch,[response(finish='stop',content='{"action":"finish","answer":"7"}') for _ in range(4)])
    result=consumption.consume(root)
    assert result['paired_arms']==4 and result['new_http']==4 and result['outputs_consumed']==0
    assert all(r['score']['hard'] for r in runner.rows(root/'pilot_consumption_traces.jsonl'))
    consumption.consume(root);assert len(seen)==4
    first=[json.loads(p['messages'][1]['content']) for p in seen[:2]]
    assert first[0]['tools']==first[1]['tools'] and not first[0]['state']['programs'] and first[1]['state']['programs']
    assert all('private_labels' not in json.dumps(p) for p in seen)
    s.close()


def test_phase_governor_no_borrow_and_no_unapproved_http(tmp_path):
    ledger={'limits':{'token_limit':90000000,'request_limit':480000,'finish_reserve':0},'attempts':{},'unknown_billing':False}
    save_json(tmp_path/'pilot_manifest.json',{'run_id':'r'})
    save_json(tmp_path/'budget/inherited.json',ledger);save_json(tmp_path/'budget/usage.json',ledger)
    with pytest.raises(FileNotFoundError):runner.PhaseGovernor(tmp_path,'learn')
    save_json(tmp_path/'authorization.json',{'approved':True,'user_authorization':'TEST','manifest_sha256':sha(tmp_path/'pilot_manifest.json'),
        'phase_limits':runner.PHASE_LIMITS})
    g=runner.PhaseGovernor(tmp_path,'learn')
    with pytest.raises(BudgetExhausted):g.admit('one',{'max_tokens':1000001},{})
    assert g.state['attempts']=={}
    g.admit('two',{'max_tokens':10},{})
    g.complete({'request_id':'two','raw_usage':{'total_tokens':999999},'outcome':'ok'})
    with pytest.raises(BudgetExhausted):g.admit('three',{'max_tokens':2},{})
    assert runner.PhaseGovernor(tmp_path,'consume').headroom({})['tokens']==800000


def test_oracles_abstain_and_consumption_requires_actual_reference(tmp_path):
    text='In millions\n| Period | Value |\n|---|---|\n| first | 12 |\n| second | 17 |'
    contract={'kind':'table_cell','text_input':'text','row_input':'row','column_input':'column','unit_input':'unit','output_field':'value'}
    values={'text':text,'row':'first','column':'Value','unit':'millions'}
    assert expected(contract,values)=={'value':12}
    for change in ({'row':'absent'},{'unit':'billions'},{'text':text+'\nIn billions'}):
        with pytest.raises(ValueError):expected(contract,{**values,**change})
    event={'name':'grep_page','result':{'data':{'complete':False}}}
    assert not verify_result({'kind':'retrieval'},{'label':'negative'},{'status':'not_found','detail':'none'},[event])[0]
    context=TaskContext();rid=context.register('producer',{'status':'ok','outputs':{'value':12}})
    producer={'result_id':rid,'program_id':'p','status':'ok'}
    assert not consumed(producer,[{'accepted':True,'content':'I used p'}])
    adapter=type('Adapter',(),{})();broker=type('Broker',(),{'context':context})()
    state={'actions':[],'producers':[]}
    action=execute_action({'action':'finish','answer_ref':{'result_id':rid,'path':['outputs','value']}},broker,[],None,state)
    assert consumed(producer,[action]) and state['prediction']=='12'
    assert not consumed({**producer,'status':'not_found'},[action])
    with pytest.raises(ValueError):execute_action({'action':'finish','answer_ref':{'result_id':'future','path':['value']}},broker,[],None,{})
    assert snapshot(restore(snapshot(context)))==snapshot(context)
    broker.remaining_calls=lambda:0
    exhausted={}
    record=execute_action({'action':'native','name':'read','arguments':{}},broker,[],None,exhausted)
    assert exhausted['terminal_reason']=='tool_budget_exhausted' and not record['accepted']
    parent={'kind':'program','status':'ok','accepted':True,'program_id':'p','result_id':rid,'resolved_refs':{},'bound_inputs':{},'result':{'outputs':{'value':12}}}
    child={'kind':'program','status':'ok','accepted':True,'program_id':'q','result_id':'child','resolved_refs':{'x':{'result_id':rid,'path':['outputs','value']}},
        'bound_inputs':{'x':12},'result':{'outputs':{'value':24}}}
    programs={'p':{'id':'p','source':'fixture','input_schema':object_schema(),'output_schema':object_schema({'value':{'type':'number'}},['value'])},
        'q':{'id':'q','source':'fixture','input_schema':object_schema({'x':{'type':'number'}},['x']),'output_schema':object_schema({'value':{'type':'number'}},['value'])}}
    assert partial_workflow([producer],[parent,child],programs)['partial']
    assert partial_workflow([producer],[parent,{**child,'accepted':False}],programs) is None


@pytest.mark.skipif(os.environ.get('OFFICE_V4_DOCKER')!='1',reason='explicit local pinned Docker gate')
def test_real_docker_search_variation_negative_and_illegal_scope(tmp_path):
    s,task,trace,spans,op=fixture_case(tmp_path)
    config=s.config;config['program_environment']['image_digest']='sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783'
    props={'pattern':{'type':'string'},'paths':{'type':'array','items':{'type':'string'}},'legacy':{'type':'boolean'}}
    spec={'input_schema':object_schema(props,list(props)),
          'output_schema':object_schema({'text':{'type':'string'},'source_path':{'type':'string'},'source_offset':{'type':'integer'},'complete':{'type':'boolean'}},
                                      ['text','source_path','source_offset','complete']),
          'allowed_tools':['grep','read','grep_page','read_context'],'probe_contract':{'kind':'retrieval'}}
    candidate={'program_spec':spec,'bound_inputs':{'pattern':'target','paths':['a.txt'],'legacy':False},'input_provenance':{},
        'atomic':proposal(op)['atomic'],'generation_basis':op}
    program=program_from_response({'source':SEARCH_FIXTURE},candidate,config['program_environment'],spec['allowed_tools'])
    value=bundle(candidate,program,'fixture',{});probes=probe_cases(value)
    assert probes['status']=='ready'
    receipts=[execute_probe(value,c,config,asdict(task)) for c in probes['cases']]
    assert all(r['passed'] for r in receipts)
    ignored={**probes['cases'][0],'inputs':{**probes['cases'][0]['inputs'],'pattern':'ignored-new-query'}}
    assert not verify_result({'kind':'retrieval'},ignored,receipts[0]['result'],receipts[0]['events'])[0]
    assert all(r['result']['execution_receipt']['image_digest']==config['program_environment']['image_digest'] for r in receipts)
    assert any(e['name']=='read_context' for e in receipts[0]['events'])
    bad={**probes['cases'][2],'inputs':{**candidate['bound_inputs'],'paths':['../outside.txt']}}
    receipt=execute_probe(value,bad,config,asdict(task));assert receipt['result']['status']=='blocked'
    s.close()
