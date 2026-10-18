"""Fixed real upstream text, native Banks, production transport and full intercepted run."""
import json
from pathlib import Path
import pytest
from atomic_skillgraph.empirical import human_frontier_runtime as rt
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask,validate_schema_instance
from atomic_skillgraph.experiments import run_human_frontier_r2 as run
from atomic_skillgraph.experiments.build_human_frontier_r2 import build,compile_variant
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http,response

LLM={'base_url':'https://fixture.invalid','model':'deepseek-v4-flash','api_key_env':'MODEL_API_KEY'}

@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Offline: real network forbidden')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)

@pytest.fixture
def banks(tmp_path):
    root=tmp_path/'run';build(root/'banks');return root

def view(root,arm='benchmark_prior'):
    variant='benchmark_prior' if arm=='NoSkill' else arm
    path=root/'banks/variants'/variant;bank=Bank(path,readonly=True)
    return bank,run.integration().HumanBankView(bank,run.read(path/'bank.json'),'off' if arm=='NoSkill' else 'manual','off')

def setup(root):
    task=PublicTask('fixture','fixture-key','A stronger statement is in the stem.',
        {'choices':[{'label':'A','text':'x is positive.'},{'label':'Q','text':'None of the above.'}]},'train')
    adapter=AnswerAdapter('livemath',{task.task_id:{'choices':task.inputs['choices'],'correct_choice':task.inputs['choices'][0]}})
    g=BudgetGovernor(root/'ledger.json',token_limit=2000000,request_limit=20,finish_reserve=0)
    return task,adapter,g

def test_lossless_ordered_bank_native_import(banks,tmp_path):
    f=run.integration();hashes=[]
    for name in ('upstream_skillopt','upstream_skillopt_lite','benchmark_prior'):
        bundle,raw=compile_variant(name);bank,v=view(banks,name)
        try:
            assert v.policy(None).encode()==raw
            assert [x['id'] for x in v.selected(None)]==bundle['assembly_order']
            assert not bank.attempts('anything') and not bank.train_cases()
            assert all(x['author_execution_permission']=='manual_eval' for x in v.selected(None))
            assert v.program_options()==[] and len(bank.all('workflow'))==1
            hashes.append(v.view_hash)
        finally:bank.close()
        f.native_import(bundle,tmp_path/name,{'backend':'sandbox_python_v1'},seed=42)
    assert len(set(hashes))==3
    assert compile_variant('benchmark_prior')[1]==compile_variant('upstream_skillopt_lite')[1]

@pytest.mark.parametrize('label',['A','B','C','D','E','ZZ'])
def test_meta_reporting_never_maps_to_answer(label):
    t=PublicTask('t','k','A stronger result can be proven.',{'choices':[
        {'label':label,'text':'One of the remaining options is correct, but a stronger result can be proven.'},
        {'label':'N','text':'None of the above.'},{'label':'X','text':'All of the above.'}]},'train')
    assert rt.meta_labels(t)==[label]
    t.inputs['choices']=t.inputs['choices'][1:]+[{'label':'Y','text':'A stronger bound holds.'}]
    assert rt.meta_labels(t)==[]

def test_three_arms_same_runtime_one_policy_and_scopes(banks,monkeypatch):
    t,a,g=setup(banks);calls=http(monkeypatch,[response(content='A',finish='stop')]*3)
    results=[]
    for arm in run.ARMS:
        bank,v=view(banks,arm)
        try:results.append(rt.solve(t,a,v,LLM,banks/arm,g,phase='development',arm=arm))
        finally:bank.close()
    for arm,wire in zip(run.ARMS,calls):
        assert wire['max_tokens']==65536 and wire['thinking']['type']=='enabled' and wire['reasoning_effort']=='high'
        assert not wire.get('tools')
        if arm!='NoSkill':assert wire['messages'][0]['content'].count(compile_variant(arm)[1].decode())==1
    assert all(c['messages'][1]==calls[0]['messages'][1] for c in calls)
    assert len({r['context']['episode_scope'] for r in g.state['attempts'].values()})==3
    assert all(x['valid'] and x['score']['hard'] and x['program_calls']==0 for x in results)
    assert rt.provider(LLM).config.request_timeout_seconds==600

def test_bounded_final_same_full_policy_once_and_not_censored(banks,monkeypatch):
    t,a,g=setup(banks);raw=response(content='',finish='length');raw['choices'][0]['message']['reasoning_content']='数学'*20000
    calls=http(monkeypatch,[raw,response(content='A',finish='stop')]);bank,v=view(banks)
    try:
        r=rt.solve(t,a,v,LLM,banks/'solve',g,phase='development',arm='benchmark_prior')
        assert r['first_truncated'] and r['finish_state']=='valid' and not r['budget_censored']
        assert len(calls)==2 and calls[1]['max_tokens']==2048 and calls[1]['thinking']['type']=='disabled'
        assert all(w['messages'][0]['content'].count(v.policy(t))==1 for w in calls)
        tail=json.loads(calls[1]['messages'][1]['content'])['previous_reasoning_tail']
        assert tail['truncated'] and len(tail['text'].encode())<=16384
        assert rt.solve(t,a,v,LLM,banks/'solve',g,phase='development',arm='benchmark_prior')==r
        assert len(calls)==2
    finally:bank.close()

@pytest.mark.parametrize('point',['after_main_response','after_score','before_trace','after_trace'])
def test_crash_recovery_never_repays(banks,monkeypatch,point):
    t,a,g=setup(banks);calls=http(monkeypatch,[response(content='A',finish='stop')]);bank,v=view(banks)
    def fault(p):
        if p==point:raise RuntimeError('injected crash')
    try:
        with pytest.raises(RuntimeError,match='injected'):
            rt.solve(t,a,v,LLM,banks/'solve',g,phase='development',arm='benchmark_prior',fault=fault)
        r=rt.solve(t,a,v,LLM,banks/'solve',g,phase='development',arm='benchmark_prior')
        assert r['valid'] and len(calls)==1 and len(g.state['attempts'])==1
    finally:bank.close()

def test_received_journal_recovery_and_unknown_stop(banks,monkeypatch):
    t,a,g=setup(banks);calls=http(monkeypatch,[response(content='A',finish='stop')]);bank,v=view(banks)
    try:
        rt.solve(t,a,v,LLM,banks/'solve',g,phase='development',arm='benchmark_prior')
        (banks/'solve/trace.json').unlink();(banks/'solve/main/response.json').unlink();(banks/'solve/main/requests.json').unlink()
        r=rt.solve(t,a,v,LLM,banks/'solve',g,phase='development',arm='benchmark_prior')
        assert r['valid'] and len(calls)==1
        save_json(banks/'unknown/main/started.json',{})
        with pytest.raises(RuntimeError,match='no blind'):
            rt.solve(t,a,v,LLM,banks/'unknown',g,phase='development',arm='benchmark_prior')
        assert len(calls)==1
    finally:bank.close()

def test_optional_digits_schema_and_malformed_actions(banks):
    optional=run.read(banks/'banks/optional_programs/bank.json')
    p=next(p for p in optional['program'] if 'exact_compute' in p['source_file'])
    assert p['input_schema']['properties']['digits']=={'type':'integer','minimum':0,'maximum':30}
    with pytest.raises(ValueError):validate_schema_instance({'expression':'1/3','digits':50},p['input_schema'])
    validate_schema_instance({'expression':'1/3','digits':30},p['input_schema'])
    module={};exec(p['source'],module);assert module['run'](None,{'expression':'1+','digits':12})['status']=='needs_input'
    schema=run.integration().action_schema();assert schema['type']=='object'
    with pytest.raises(ValueError):validate_schema_instance({'action':'finish','answer':'A','invented':True},schema)
    counter=next(p for p in optional['program'] if 'counterexample' in p['source_file'])
    assert 'unrelated toy' in counter['title'] and 'general proof' in counter['title']

def test_real_legacy_schema400_is_absent():
    p=Path('/home/yangchengyu/livemath_human_frontier_schema_v2_seed42_20261015/parent_rejected_request.json')
    if not p.exists():pytest.skip('Saved real schema 400 receipt unavailable')
    assert 'Invalid schema' in p.read_text()
    s=run.integration().action_schema();assert s['type']=='object'
    assert all(branch['type']=='object' for branch in s['oneOf'])


def test_optional_native_import_has_no_stale_workflow_references(banks,tmp_path):
    bundle=run.read(banks/'banks/optional_programs/bank.json')
    ids={p['id'] for p in bundle['program']}
    assert all(n['program_id'] in ids for w in bundle['workflow'] for n in w['nodes'] if n.get('program_id'))
    imported=run.integration().native_import(bundle,tmp_path/'native_optional',bundle['program'][0]['environment'])
    bank=Bank(tmp_path/'native_optional',readonly=True)
    try:
        assert len(bank.all('program'))==2 and len(bank.all('workflow'))==3
        assert all(bank.get(n['program_id']) for w in bank.all('workflow') for n in w['nodes'] if n.get('program_id'))
        assert all(p['state']=='candidate' and 'local_validation' not in p for p in bank.all('program'))
    finally:bank.close()

def fixture_run(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    for split,n in [('train',60),('val',17),('test',100)]:
        tasks=[];gold={}
        for i in range(n):
            task=PublicTask(split+str(i),split+'key'+str(i),'Public mathematical task.',{'choices':[
                {'label':'A','text':'x>0.'},{'label':'B','text':'One of the remaining options is correct, but a stronger result can be proven.' if i%2 else 'x<0.'}]},split)
            tasks.append(task.__dict__);gold[task.task_id]={'correct_choice':task.inputs['choices'][0],'choices':task.inputs['choices']}
        save_json(data/(split+'.json'),tasks);save_json(data/('evaluator_records_'+split+'.json'),gold)
    save_json(tmp_path/'base.json',{'llm':LLM,'program_environment':{'backend':'sandbox_python_v1'}})
    prefix={'limits':run.LIMITS,'attempts':{},'unknown_billing':False};save_json(tmp_path/'prefix.json',prefix)
    save_json(tmp_path/'history/campaign_budget_prefix.json',prefix)
    save_json(tmp_path/'history/budget.json',{'attempts':{'old':{'context':{'parent_task_id':'old'},'reserved_tokens':30000,'accounted_tokens':30000}}})
    monkeypatch.setattr(run,'code_identity',lambda:{'git_sha':'fixture','source_sha256':'fixture','tracked_dirty':False})
    root=tmp_path/'run';run.prepare(root,tmp_path/'base.json',data,tmp_path/'prefix.json',tmp_path/'history')
    return root

def test_full_36_then_two_test100_control_flow_resume(tmp_path,monkeypatch):
    root=fixture_run(tmp_path,monkeypatch);calls=http(monkeypatch,[response(content='A',finish='stop')]*236)
    saved=run.read(root/'schedule.json');seen=set()
    for i in range(12):
        rows=saved[3*i:3*i+3]
        assert [r['arm'] for r in rows]==run.ARMS[i%3:]+run.ARMS[:i%3]
        assert len({r['task']['physical_key'] for r in rows})==1
        seen.add(rows[0]['task']['physical_key'])
    assert len(seen)==12
    real=rt.solve;counter=0
    def interrupt(*a,**kw):
        nonlocal counter
        if counter==37:raise RuntimeError('fixture boundary interruption')
        counter+=1;return real(*a,**kw)
    with pytest.raises(RuntimeError,match='boundary'):run.execute(root,solve_fn=interrupt)
    assert len(calls)==37
    result=run.execute(root);assert result['status']=='completed_test' and result['observation_requirement_met']
    assert len(calls)==236 and result['new_http']==236
    assert run.execute(root)['status']=='completed_test' and len(calls)==236
    assert all(result['by_arm']['development/'+a]['normal']==12 for a in run.ARMS)
    assert all(result['by_arm']['test_'+a+'/'+a]['normal']==100 for a in run.TEST_ARMS)

def test_actual_legacy_truncated_responses(banks,monkeypatch):
    archive=Path('/home/yangchengyu/livemath_human_frontier_schema_v2_seed42_20261015')
    if not archive.exists():pytest.skip('Completed real HumanFrontier archive unavailable')
    evaluator=run.read('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/livemath/evaluator_records_test.json')
    chosen=[]
    for p in sorted((archive/'test/traces').glob('*.json')):
        trace=run.read(p)
        if trace['requests'][0].get('provider_finish_reason')=='length':chosen.append(trace)
    assert len(chosen)>=2
    bank,v=view(banks)
    try:
        for i,old in enumerate(chosen[:2]):
            responses=[]
            for request in old['requests']:
                for attempt in request.get('http_attempts',[]):
                    if attempt.get('public_response'):
                        data=json.loads(json.dumps(attempt['public_response']))
                        data['usage']=attempt['raw_usage'];responses.append(data)
            assert responses[0]['choices'][0]['finish_reason']=='length'
            calls=http(monkeypatch,responses)
            task=PublicTask(**old['task']);adapter=AnswerAdapter('livemath',evaluator)
            g=BudgetGovernor(banks/str(i)/'ledger.json',token_limit=128000,request_limit=2,finish_reserve=0)
            result=rt.solve(task,adapter,v,LLM,banks/str(i)/'solve',g,phase='test_benchmark_prior',arm='benchmark_prior')
            assert result['first_truncated'] and result['valid'] and len(calls)==2
            assert not result['budget_censored'] and result['raw_usage']==[r['usage'] for r in responses[:2]]
            assert calls[1]['max_tokens']==2048 and calls[1]['thinking']['type']=='disabled'
    finally:bank.close()

def test_native_config_does_not_inherit_legacy_32768():
    p=Path('/home/yangchengyu/livemath_human_frontier_schema_v2_seed42_20261015/config.json')
    if not p.exists():pytest.skip('Actual native config required')
    base=run.read(p);base['llm']['runtime']['max_completion_tokens']=32768
    bundle,_=compile_variant('benchmark_prior')
    cfg=run.integration().configure(base,bundle,Path('/tmp/offline-human'),'/tmp/tasks.json','/tmp/evaluator.json')
    assert cfg['llm']['runtime']['max_completion_tokens']==65536
    assert cfg['llm']['runtime']['max_total_tokens_per_task']==128000
    assert cfg['llm']['purpose_overrides']['finish_only']=={'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}


