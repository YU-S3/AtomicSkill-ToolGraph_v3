"""Temporary search uses the current owner, then normal source qualification."""
from copy import deepcopy
import pytest
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


@pytest.mark.parametrize('span_handoff',[False,True])
def test_empty_bank_searches_in_one_temporary_script_and_accounts_every_call(tmp_path, worker, span_handoff):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    if span_handoff:
        from test_evidence_handoff import configure
        configure(s.config)
        s.adapter.config=s.config
        s.task_context.settings.update(s.config["runtime"])
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


@pytest.mark.parametrize('span_handoff',[False,True])
def test_temporary_script_cannot_reset_shared_call_budget(tmp_path, worker, span_handoff):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    if span_handoff:
        from test_evidence_handoff import configure
        configure(s.config)
        s.adapter.config=s.config
        s.task_context.settings.update(s.config["runtime"])
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


@pytest.mark.parametrize('span_handoff',[False,True])
def test_temporary_contract_reaches_real_runtime_payload_and_result(tmp_path, monkeypatch, worker, span_handoff):
    import json
    from atomic_skillgraph.empirical.executor import Executor
    from atomic_skillgraph.empirical.model_view import expand_material
    from atomic_skillgraph.empirical.planner import dynamic
    from atomic_skillgraph.empirical.program_worker import public_program_abi
    s = office(tmp_path); s.worker = worker; s.adapter.worker = worker
    if span_handoff:
        from test_evidence_handoff import configure
        configure(s.config)
        s.adapter.config=s.config
        s.task_context.settings.update(s.config["runtime"])
    worker.settings['wall_timeout_seconds'] = 20
    sent = http(monkeypatch, [response([{'action':'call_tool','name':'execute_python','arguments':{'source':SEARCH}}]),
                             response([{'action':'finish','answer':'target'}])])
    try:
        broker = Broker(s.adapter, 5, context=s.task_context)
        result = Executor(s.bank,s.agent,worker,s.planner).run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
        initial = expand_material(json.loads(sent[0]['messages'][1]['content']))
        contract = initial['calls']['task_local_program']
        assert contract['tool'] == 'execute_python' and contract['context_methods'] == public_program_abi()['methods']
        assert set(contract['allowed_tools']) == {'glob','grep','read','read_result'}
        assert contract['max_internal_calls'] == worker.settings['max_tool_calls_per_invocation']
        from atomic_skillgraph.empirical.program_worker import DIAGNOSTIC_MAX_BYTES
        assert contract['max_stdout_bytes'] == (65536 if span_handoff else DIAGNOSTIC_MAX_BYTES)
        assert contract['wall_timeout_seconds'] == 20 and initial['recovery']['remaining_calls'] == 5
        actual = expand_material(json.loads(sent[1]['messages'][1]['content']))
        assert actual['recovery']['remaining_calls'] == 1
        assert any(r.get('data',{}).get('computed',{}).get('text') == '第二行 target\nlast'
                   for r in actual['memory']['recent_results'])
        assert result['prediction']=='target' and len(sent)==2 and len(broker.events)==4
        assert not s.bank.all('skill') and not s.bank.all('program')
    finally: s.close()


def test_advertised_stdout_limit_matches_real_worker_capture(tmp_path, worker):
    import json
    s = office(tmp_path); s.adapter.worker = worker
    worker.settings['wall_timeout_seconds'] = 20
    broker = Broker(s.adapter,4)
    contract = s.adapter.task_program_contract(broker.available_tools())
    try:
        limit = contract['max_stdout_bytes']
        oversized = broker.call('execute_python',{'source':"import json\nprint(json.dumps({'text':'x'*6000}))"})
        assert oversized['accepted'] and len(oversized['observation'].encode())==limit
        assert 'computed' not in oversized['data']
        source = "import json\nresult={'text':'x'*2000}\nencoded=json.dumps(result)\nassert len(encoded.encode('utf-8'))<="+str(limit)+"\nprint(encoded)"
        bounded = broker.call('execute_python',{'source':source})
        assert bounded['accepted'] and bounded['data']['computed']=={'text':'x'*2000}
        assert len(json.dumps(bounded['data']['computed']).encode())<=limit
        assert not s.bank.all('program')
    finally: s.close()


def test_temporary_contract_requires_current_permission(tmp_path):
    from atomic_skillgraph.harness.benchmarks import SpreadsheetAdapter
    s = office(tmp_path)
    sheet = SpreadsheetAdapter({},s.config)
    try:
        assert s.adapter.task_program_contract([]) is None
        assert sheet.task_program_contract(sheet.available_tools()) is None
        # Turning off the declared bound execution permission removes the capability;
        # a description or benchmark/task name cannot grant it.
        s.adapter.broker_bound_tools = ()
        assert s.adapter.task_program_contract(s.adapter.available_tools()) is None
    finally: sheet.close(); s.close()


@pytest.mark.parametrize('span_handoff',[False,True])
def test_temporary_search_keeps_path_and_recursive_execution_rejections(tmp_path, worker, span_handoff):
    s = office(tmp_path); s.adapter.worker = worker; worker.settings['wall_timeout_seconds'] = 20
    if span_handoff:
        from test_evidence_handoff import configure
        configure(s.config)
        s.adapter.config=s.config
        s.task_context.settings.update(s.config["runtime"])
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


def test_nested_read_timeout_is_worker_failure_and_runtime_can_continue(tmp_path, worker, monkeypatch):
    import threading
    s = office(tmp_path); s.adapter.worker = worker
    worker.settings['wall_timeout_seconds'] = 4
    original = s.adapter.call
    release = threading.Event()
    def slow(name, arguments):
        if name == 'grep': release.wait(8)
        return original(name, arguments)
    monkeypatch.setattr(s.adapter, 'call', slow)
    broker = Broker(s.adapter, 4, native_timeout=15)
    try:
        result = broker.call('execute_python', {'source':"ctx.call('grep', {'pattern':'target'})"})
        assert not result['accepted'] and 'timeout' in result['error'].lower()
        assert not broker.unknown and all(e['state'] == 'finished' for e in broker.events)
        assert [e['name'] for e in broker.events] == ['execute_python', 'grep']
        assert broker.events[1]['result']['error_code'] == 'native_read_timeout'
        assert broker.remaining_calls() == 2
        assert broker.call('glob', {'pattern':'*.txt'})['accepted']
    finally:
        release.set()
        s.close()


def test_learning_directory_keeps_acquired_pages_after_discovery_scripts(tmp_path, worker):
    from atomic_skillgraph.empirical.model_view import learning_case
    from atomic_skillgraph.empirical.public_results import SCOPED_SEARCH, TEXT_PAGE
    s = office(tmp_path); s.adapter.worker = worker
    worker.settings['wall_timeout_seconds'] = 20
    broker = Broker(s.adapter, 12)
    discovery = "import json\nctx.call('glob', {'pattern':'*.txt'})\nif False: ctx.call('read', {'path':'a.txt','offset':0})\nprint(json.dumps({'listed':True}))"
    search = "import json\nr=ctx.call('grep', {'pattern':'target'})\nprint(json.dumps({'hits':r['data']}))"
    page = "import json\nr=ctx.call('read', {'path':'a.txt','offset':0})\nprint(json.dumps({'text':r['data']['text']}))"
    try:
        for source in (discovery, search, page):
            assert broker.call('execute_python', {'source':source})['accepted']
        trace = {'tools':broker.events, 'execution':{}, 'score':{'hard':False}}
        evidence = evidence_from_trace(s.adapter.task, trace, s.config['program_environment'])
        assert [e['context_result_protocols'] for e in evidence] == [[], [SCOPED_SEARCH], [TEXT_PAGE]]
        directory = learning_case({'task':s.adapter.model_task(), 'local_evidence':evidence},
                                  s.adapter.task.physical_key)['local_evidence']
        assert [r['local_evidence_ref'] for r in directory] == [evidence[i]['id'] for i in (2,1,0)]
        assert len(broker.events) == 6 and not s.bank.all('program')
    finally:
        s.close()


def test_later_compute_trial_replays_nested_calls_and_original_failure(tmp_path, monkeypatch, worker):
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    s=office(tmp_path);s.worker=worker;s.adapter.worker=worker
    s.adapter_factory=lambda:OfficeAdapter({'office':{'answer':'private'}},s.config)
    worker.settings['wall_timeout_seconds']=20
    broker=Broker(s.adapter,24)
    broker.call('execute_python',{'source':SEARCH})
    failure=broker.call('execute_python',{'source':"raise ValueError('recorded failure')"})
    assert not failure['accepted']
    source="import json\nvalues=[2,5,9]\nprint(json.dumps({'total':sum(values)}))"
    broker.call('execute_python',{'source':source})
    trace={'tools':broker.events,'execution':{'prediction':'wrong'},'score':{'hard':False}}
    evidence=s.learner._experience(s.adapter.task,trace)['local_evidence'][-1]
    spec={'goal':'Sum acquired numeric values','execution_intent':'program_requested','result_role':'intermediate',
        'input_schema':object_schema({'values':{'type':'array','items':{'type':'number'}}},['values']),
        'output_schema':object_schema({'total':{'type':'number'}},['total'])}
    binding={'case_id':s.adapter.task.physical_key,'local_evidence_ref':evidence['id'],'inputs':{'values':[2,5,9]},
        'input_refs':{'values':{'kind':'source_literal','path':['body',1,'value']}},
        'reference_fields':{'total':{'kind':'json','path':['data','computed','total']}}}
    proposal={'decision':'propose_skill_and_program_spec','skill':spec,
        'realization_request':{'skill_id':'$new','action':'build','case_bindings':[binding]}}
    generated={'source':"def run(ctx, inputs):\n    return {'status':'ok','outputs':{'total':sum(inputs['values'])}}"}
    sent=http(monkeypatch,[response([proposal],name='submit_learning'),response([generated],name='submit_program')])
    try:
        learned=s.learner.learn(s.adapter.task,trace)
        assert not learned['errors'] and len(sent)==2
        program=s.bank.all('program')[0]
        assert s.bank.program_eligible(program) and program['local_validation']['reason']=='source_output_match'
        trial=s.bank.attempts(program['id'])[0]
        assert [e['name'] for e in trial['tools']]==['execute_python','glob','grep','read','execute_python']
        assert trial['tools'][-1]['result']['accepted'] is False
        # A different rejection is a source mismatch, never a qualifying replay.
        from copy import deepcopy
        from atomic_skillgraph.empirical.local_validation import validate_on_source
        experience=s.learner._experience(s.adapter.task,trace)
        bad=deepcopy(experience);bad['events'][4]['result']['error']='a different recorded error'
        locked=s.bank.jobs()[0]['case_bindings'][0]
        import pytest
        with pytest.raises(ValueError,match='Source prefix cannot be reconstructed'):
            validate_on_source(s,program,locked,s.adapter.task,bad,'mismatched-replay')
    finally:s.close()
