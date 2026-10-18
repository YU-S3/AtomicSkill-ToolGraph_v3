"""Manual R31 native contracts, graph dataflow, billing and campaign recovery."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from atomic_skillgraph.experiments import run_human_frontier_r31 as run
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http, response

LLM={'base_url':'https://fixture.invalid','model':'deepseek-v4-flash','api_key_env':'MODEL_API_KEY'}
META='One of the remaining options is correct, but a stronger result can be proven.'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Real HTTP forbidden in offline tests')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


@pytest.fixture
def native(tmp_path):
    run.import_bank(run.PACKAGE,tmp_path/'bank')
    bank=Bank(tmp_path/'bank',readonly=True)
    yield bank
    bank.close()


def task(meta=True, split='train', key='fixture'):
    return PublicTask(key,key,'Which statement follows?',{'choices':[
        {'label':'A','text':'For every x, there exists a unique solution.'},
        {'label':'D','text':META if meta else 'There exists a solution.'}]},split)


def context(tmp_path,t):
    adapter=AnswerAdapter('livemath',{t.task_id:{'correct_choice':t.inputs['choices'][0],'choices':t.inputs['choices']}})
    governor=BudgetGovernor(tmp_path/'ledger.json',token_limit=1000000,request_limit=20,finish_reserve=0)
    return adapter,governor


def module(monkeypatch):
    m=run.bridge(run.PACKAGE)
    graph=__import__(m.__package__+'.native_programmatic_runtime',fromlist=['fixture_worker'])
    def worker(p,args):
        value=graph.fixture_worker(p,args)
        return {**value,'elapsed_seconds':.01,'cpu_seconds':.001,
                'execution_receipt':{'exit_code':0,'offline_fixture':True}}
    monkeypatch.setattr(m,'docker_worker',lambda adapter:worker)
    return m,worker


def test_native_assets_and_program_only_zero_http(native,tmp_path,monkeypatch):
    m,_=module(monkeypatch);t=task();a,g=context(tmp_path,t)
    before=native.digest();bundle=run.read(run.PACKAGE/'bank.json')
    r=m.solve_one(t,a,native,bundle,LLM,tmp_path/'episode',g)
    assert r['valid'] and r['answer']=='D' and not r['correct']
    assert r['programmatic_benchmark_prior'] and r['model_http']==0 and not g.state['attempts']
    assert r['program_invocations_total']==2 and r['program_dataflow_consumed']==2
    assert all(n['implementation_id'] for n in r['graph']['program_nodes'])
    assert m.solve_one(t,a,native,bundle,LLM,tmp_path/'episode',g)==r
    assert native.digest()==before and len(native.all('program'))==3


def test_dynamic_actual_new_atomic_payload_and_immutable_receipts(native,tmp_path,monkeypatch):
    m,_=module(monkeypatch);t=task(False);a,g=context(tmp_path,t)
    calls=http(monkeypatch,[response(content='A',finish='stop')])
    r=m.solve_one(t,a,native,run.read(run.PACKAGE/'bank.json'),LLM,tmp_path/'episode',g)
    guidance=native.get('skill_lm_r31_dynamic_theorem')['guidance']
    assert calls[0]['messages'][0]['content'].count(guidance)==1
    assert 'surface_differences_only_no_truth_verification' in calls[0]['messages'][0]['content']
    assert json.loads(calls[0]['messages'][1]['content'])['inputs']==t.inputs
    assert calls[0]['max_tokens']==65536 and calls[0]['thinking']['type']=='enabled'
    assert r['valid'] and r['model_http']==1 and r['program_invocations_total']==2
    assert r['graph']['program_nodes'][-1]['handoff_exposure']
    p=tmp_path/'episode/program_stages/scan.json';p.write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):
        m.solve_one(t,a,native,run.read(run.PACKAGE/'bank.json'),LLM,tmp_path/'episode',g)
    assert len(calls)==1


@pytest.mark.parametrize('valid_finish',[True,False])
def test_truncation_preserves_scored_failure_or_valid_finish(native,tmp_path,monkeypatch,valid_finish):
    m,_=module(monkeypatch);t=task(False);a,g=context(tmp_path,t)
    calls=http(monkeypatch,[response(content='',finish='length'),response(content='A' if valid_finish else '',finish='stop')])
    r=m.solve_one(t,a,native,run.read(run.PACKAGE/'bank.json'),LLM,tmp_path/'episode',g)
    assert r['status']=='completed' and r['valid']==valid_finish and r['first_truncated']
    assert not r['budget_censored'] and r['model_http']==2
    assert calls[1]['max_tokens']==2048 and calls[1]['thinking']['type']=='disabled'
    assert len(r['raw_usage'])==2


def test_actual_graph_references_and_missing_dependencies(native,tmp_path,monkeypatch):
    m,worker=module(monkeypatch)
    bundle=deepcopy(run.read(run.PACKAGE/'bank.json'))
    bundle['workflow'][0]['nodes'][1]['args']['meta_label']={'from':'scan','field':'invented'}
    class Altered:
        def all(self,k):return bundle[k]
        def get(self,k):return next((a for kind in bundle['counts'] for a in bundle[kind] if a['id']==k),None)
    with pytest.raises(ValueError,match='Unknown result field'):
        m.run_programmatic_graph(task(),Altered(),run_worker=worker)


def test_stage_crash_no_blind_replay_and_failure_fallback(native,tmp_path,monkeypatch):
    m,worker=module(monkeypatch);calls=[]
    def crash(p,args):
        calls.append(p['id'])
        if len(calls)==2:raise RuntimeError('crash without ended receipt')
        return worker(p,args)
    with pytest.raises(RuntimeError,match='crash'):
        m.run_programmatic_graph(task(),native,run_worker=crash,stage_dir=tmp_path/'stages')
    with pytest.raises(FileExistsError):
        m.run_programmatic_graph(task(),native,run_worker=crash,stage_dir=tmp_path/'stages')
    assert len(calls)==2
    seen={}
    def dynamic(t,h,n):seen.update(h);return 'A',{'tokens':0,'http':0}
    r=m.run_programmatic_graph(task(),native,run_worker=lambda p,a:{'status':'execution_error'},dynamic_answer=dynamic)
    assert r['branch']=='dynamic_fallback' and seen['choices']==task().inputs['choices']
    assert not seen['features'] and r['program_nodes'][0]['status']=='execution_error'


def test_crash_after_paid_response_reuses_both_worker_and_http(native,tmp_path,monkeypatch):
    m,_=module(monkeypatch);t=task(False);a,g=context(tmp_path,t)
    calls=http(monkeypatch,[response(content='A',finish='stop')]);save=m.save
    def crash(path,value):
        if Path(path).name=='r31_result.json':raise RuntimeError('top receipt interruption')
        save(path,value)
    monkeypatch.setattr(m,'save',crash)
    with pytest.raises(RuntimeError,match='interruption'):
        m.solve_one(t,a,native,run.read(run.PACKAGE/'bank.json'),LLM,tmp_path/'episode',g)
    monkeypatch.setattr(m,'save',save)
    r=m.solve_one(t,a,native,run.read(run.PACKAGE/'bank.json'),LLM,tmp_path/'episode',g)
    assert len(calls)==1 and r['program_worker_invocations']==0 and r['program_invocations_total']==2


def fixture_campaign(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir();previous=tmp_path/'previous'
    for split,n in [('train',60),('val',17),('test',100)]:
        tasks=[task(bool(i%2),split,split+str(i)).__dict__ for i in range(n)]
        save_json(data/f'{split}.json',tasks)
        save_json(data/f'evaluator_records_{split}.json',{t['task_id']:{'correct_choice':t['inputs']['choices'][0],
                  'choices':t['inputs']['choices']} for t in tasks})
    save_json(previous/'plan.json',{'data_root':str(data)})
    save_json(previous/'result.json',{'status':'completed_test'})
    save_json(previous/'config.json',{'llm':LLM})
    save_json(previous/'budget/ledger.json',{'limits':run.LIMITS,'attempts':{},'unknown_billing':False})
    monkeypatch.setattr(run,'code_identity',lambda:{'git_sha':'offline_fixture','tracked_dirty':False})
    root=tmp_path/'run';run.prepare(root,previous)
    m,_=module(monkeypatch);monkeypatch.setattr(run,'bridge',lambda p:m)
    run.check(root)
    return root,m


def test_full_train4_test100_and_resume_no_repayment(tmp_path,monkeypatch):
    root,m=fixture_campaign(tmp_path,monkeypatch)
    calls=http(monkeypatch,[response(content='A',finish='stop')]*52)
    count=0
    def stop(*a,**kw):
        nonlocal count
        count+=1
        if count==9:raise RuntimeError('boundary interruption')
        return m.solve_one(*a,**kw)
    with pytest.raises(RuntimeError,match='interruption'):run.execute(root,solve_fn=stop)
    r=run.execute(root)
    assert r['status']=='completed_test' and r['observation_requirement_met']
    assert r['by_phase']['test']['scored']==100 and r['by_phase']['test']['program_invocations']==200
    assert r['by_phase']['test']['branches']['meta_prior']['n']==50
    assert len(calls)==52 and r['new_http']==52
    assert run.execute(root)['status']=='completed_test' and len(calls)==52


def test_predecessor_budget_not_reset(tmp_path,monkeypatch):
    root,_=fixture_campaign(tmp_path,monkeypatch)
    predecessor=Path(run.read(root/'plan.json')['previous_run'])
    ledger=run.read(predecessor/'budget/ledger.json')
    ledger['attempts']['old']={'status':'metered','reserved_tokens':40000000,'accounted_tokens':40000000}
    save_json(predecessor/'budget/ledger.json',ledger)
    with pytest.raises(ValueError,match='cumulative token'):
        run.prepare(tmp_path/'over_budget',predecessor)
