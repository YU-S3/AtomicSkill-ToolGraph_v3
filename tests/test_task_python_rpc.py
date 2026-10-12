"""Temporary search uses the current owner, then normal source qualification."""
from copy import deepcopy
from atomic_skillgraph.empirical.contracts import object_schema
from atomic_skillgraph.empirical.local_validation import evidence_from_trace
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import office, http, response
from test_empirical import worker


SEARCH = '''import json
pattern = 'target'
paths = ctx.call('glob', {'pattern': '*.txt'})['data']
hits = ctx.call('grep', {'pattern': pattern, 'paths': paths})['data']
page = ctx.call('read', {'path': hits[0]['path'], 'offset': hits[0]['offset']})['data']
print(json.dumps({'path': hits[0]['path'], 'text': page['text']}))
'''


def test_empty_bank_searches_in_one_temporary_script_and_accounts_every_call(tmp_path, worker):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    snapshots = []
    broker = Broker(s.adapter, 5, journal=lambda events:snapshots.append(deepcopy(events)))
    try:
        result = broker.call('execute_python', {'source':SEARCH})
        assert result['accepted'] and result['data']['computed'] == {'path':'a.txt','text':'第二行 target\nlast'}
        assert [e['name'] for e in broker.events] == ['execute_python','glob','grep','read']
        assert all(e['state']=='finished' and e['backend_invoked'] for e in broker.events)
        parent = broker.events[0]['event_id']
        assert all(e['parent_event_id']==parent for e in broker.events[1:])
        assert snapshots[-1] == broker.events and broker.remaining_calls() == 1
        assert not s.bank.all('skill') and not s.bank.all('program')
        trace = {'tools':broker.events,'execution':{},'score':{'hard':False}}
        evidence = evidence_from_trace(s.adapter.task,trace,s.config['program_environment'])
        assert len(evidence)==1 and evidence[0]['prefix']==[]
        broker.call('glob', {'pattern':'*.txt'})
        evidence = evidence_from_trace(s.adapter.task,{'tools':broker.events},s.config['program_environment'])
        assert evidence[-1]['prefix'] == [{'name':'execute_python','arguments':{'source':SEARCH}}]
    finally: s.close()


def test_temporary_script_cannot_reset_shared_call_budget(tmp_path, worker):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    broker = Broker(s.adapter, 2)
    source = "ctx.call('glob', {'pattern':'*.txt'})\nctx.call('glob', {'pattern':'*.txt'})"
    try:
        result = broker.call('execute_python',{'source':source})
        assert not result['accepted'] and 'budget' in result['error'].lower()
        assert len(broker.events)==2 and broker.remaining_calls()==0 and not broker.unknown
        assert all(e['state']=='finished' for e in broker.events)
    finally: s.close()


def test_runtime_consumes_parent_result_with_no_existing_program(tmp_path, worker):
    from atomic_skillgraph.empirical.executor import Executor
    from atomic_skillgraph.empirical.model_view import expand_material
    from atomic_skillgraph.empirical.planner import dynamic
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    calls = []
    def agent(stage, prompt, material, *args, **kwargs):
        calls.append(expand_material(material))
        if len(calls)==1: return {'action':'call_tool','name':'execute_python','arguments':{'source':SEARCH}}
        actual = calls[-1]['memory']['recent_results']
        assert any(r.get('data',{}).get('computed',{}).get('text')=='第二行 target\nlast' for r in actual)
        return {'action':'finish','answer':'target'}
    try:
        broker = Broker(s.adapter,5)
        result = Executor(s.bank,agent,worker,s.planner).run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
        assert result['prediction']=='target' and len(calls)==2 and len(broker.events)==4
        assert not s.bank.all('program') and not s.bank.all('skill')
    finally: s.close()


def test_temporary_search_keeps_path_and_recursive_execution_rejections(tmp_path, worker):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    broker = Broker(s.adapter, 5)
    source = '''import json
bad_path = ctx.call('read', {'path':'../outside.txt','offset':0})
recursive = ctx.call('execute_python', {'source':'print(1)'})
print(json.dumps({'path_accepted':bad_path['accepted'], 'recursive_accepted':recursive['accepted']}))
'''
    try:
        result = broker.call('execute_python',{'source':source})
        assert result['accepted'] and result['data']['computed'] == {'path_accepted':False,'recursive_accepted':False}
        assert len(broker.events)==3 and not broker.events[-1]['backend_invoked']
        assert 'not authorized' in broker.events[-1]['result']['error']
        assert not s.bank.all('program')
    finally: s.close()


def test_search_on_failed_task_can_learn_only_through_real_source_trial(tmp_path, monkeypatch, worker):
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    s = office(tmp_path); s.worker = worker; s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    s.adapter_factory = lambda: OfficeAdapter({'office':{'answer':'private'}},s.config)
    broker = Broker(s.adapter, 24)
    broker.call('execute_python',{'source':SEARCH})
    trace = {'tools':broker.events,'execution':{'prediction':'wrong'},'score':{'hard':False}}
    evidence = s.learner._experience(s.adapter.task,trace)['local_evidence'][0]
    proposal = {'decision':'propose_skill_and_program_spec','skill':{
        'goal':'Locate and read a text match in an authorized corpus','execution_intent':'program_requested',
        'result_role':'intermediate','input_schema':object_schema({'pattern':{'type':'string'}},['pattern']),
        'output_schema':object_schema({'path':{'type':'string'},'text':{'type':'string'}},['path','text'])},
        'realization_request':{'skill_id':'$new','action':'build','case_bindings':[{
            'case_id':s.adapter.task.physical_key,'local_evidence_ref':evidence['id'],'inputs':{'pattern':'target'},
            'input_refs':{'pattern':{'kind':'source_literal','path':['body',1,'value']}},
            'reference_fields':{k:{'kind':'json','path':['data','computed',k]} for k in ['path','text']}}]}}
    generated = {'source':'''def run(ctx, inputs):
    paths = ctx.call('glob', {'pattern':'*.txt'})['data']
    hits = ctx.call('grep', {'pattern':inputs['pattern'], 'paths':paths})['data']
    if not hits: return {'status':'not_found','outputs':{}}
    page = ctx.call('read', {'path':hits[0]['path'], 'offset':hits[0]['offset']})['data']
    return {'status':'ok','outputs':{'path':hits[0]['path'],'text':page['text']}}
'''}
    sent = http(monkeypatch,[response([proposal],name='submit_learning'),response([generated],name='submit_program')])
    try:
        learned = s.learner.learn(s.adapter.task,trace)
        assert not learned['errors'] and len(sent)==2
        program = s.bank.all('program')[0]
        assert s.bank.program_eligible(program) and program['local_validation']['reason']=='source_output_match'
        result = worker.execute(program,{'pattern':'first'},Broker(s.adapter,24))
        assert result['status']=='ok' and result['outputs']['text'].startswith('first\n') and result['calls']==3
        # The fixture's private whole-task answer never becomes a Program output.
        assert 'private' not in generated['source'] and program['result_role']=='intermediate'
    finally: s.close()


def test_temporary_worker_obeys_the_owners_earlier_deadline(tmp_path, worker):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    broker = Broker(s.adapter,2,native_timeout=.4)
    try:
        result = broker.call('execute_python',{'source':'import time\ntime.sleep(10)'})
        assert not result['accepted'] and 'timeout' in result['error'].lower()
        assert result['worker_seconds'] < 5 and not broker.unknown and len(broker.events)==1
    finally: s.close()
