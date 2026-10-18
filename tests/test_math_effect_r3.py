"""Protocol regression: learned predicates, actual Docker and intercepted model HTTP."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
import yaml

from atomic_skillgraph.empirical import math_effect_program_r3 as p, math_effect_runtime_r3 as r
from atomic_skillgraph.empirical import math_effect_runtime as base
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import run_math_effect_r3 as runner
from atomic_skillgraph.experiments.run_empirical import write_json
from test_cf2_contracts import http,response
from test_math_effect_p7_r2 import value

ENV={'backend':'sandbox_python_v1','python':'3.12','dependencies':['openpyxl==3.1.5','pandas==2.3.3','pillow==11.3.0'],
    'adapter_abi':'simple.v2','image_digest':'sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783'}

def config():
    return {'effect_p7':yaml.safe_load(Path('configs/livemath_learned_r3_seed42.yaml').read_text()),
        'llm':{'base_url':'https://fixture.invalid','model':'deepseek-v4-flash','api_key_env':'MODEL_API_KEY'},'program_environment':ENV}

def proposal(rule=True):
    v=value(ids=('s0','s1'));v['rule_basis']='benchmark_prior' if rule else 'reasoning_procedure'
    v['public_choice_rule']={'require_all':[r'\bblue\b'],'exclude_any':[]} if rule else None
    return v

def candidate(rule=True):
    return p.candidate(proposal(rule),['s'+str(i) for i in range(17)],{'response_hash':'intercepted_fixture'})

def tasks(split='train',n=4):
    return [PublicTask('t'+str(i),'s'+str(i),'Choose a property.',{'choices':[
        {'label':'A','text':'red'},{'label':'B','text':'blue'}]},split) for i in range(n)]

def adapter(ts):
    return AnswerAdapter('livemath',{t.task_id:{'correct_choice':t.inputs['choices'][1],'choices':t.inputs['choices']} for t in ts})

def test_contract_rejects_fake_proof_and_lookup():
    v=proposal();p.validate(v,['s0','s1'])
    for bad in ['s0',r'(a)\1',r'(?=a)b','test123','[']:
        v['public_choice_rule']['require_all']=[bad]
        with pytest.raises(ValueError):p.validate(v,['s0','s1'])
    v=proposal();v['rule_basis']='reasoning_procedure'
    with pytest.raises(ValueError):p.validate(v,['s0','s1'])
    v=proposal();v['public_choice_rule']=[]
    with pytest.raises(ValueError):p.validate(v,['s0','s1'])
    c=candidate();program=p.compile_program(c,ENV)
    assert 'stronger' not in program['source'] and 'meta' not in program['source']
    c['candidate']['goal']='Changed'
    assert not p.checked(c)

def test_one_repair_preserves_valid_sibling_and_no_change(tmp_path,monkeypatch):
    a=proposal(False);b=proposal();b['public_choice_rule']['require_all']=['[']
    calls=http(monkeypatch,[response([{'decision':'propose','candidates':[a,b]}],name='submit_effect'),
        response([{'decision':'repair','repairs':[{'index':1,'replacement':proposal()}]}],name='submit_effect')])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=600000,request_limit=12,finish_reserve=0)
    got=p.learn(tmp_path/'learn',config(),g,{},['s0','s1'])
    assert len(got['candidates'])==2 and got['structural_repairs']==1 and len(calls)==2
    assert p.learn(tmp_path/'learn',config(),g,{},['s0','s1'])==got and len(calls)==2
    calls=http(monkeypatch,[response([{'decision':'no_change','candidates':None}],name='submit_effect')])
    assert p.learn(tmp_path/'abstain',config(),g,{},['s0','s1'])['state']=='no_change' and len(calls)==1

@pytest.fixture
def qualified(tmp_path):
    import subprocess
    ok=subprocess.run(['docker','image','inspect',ENV['image_digest']],capture_output=True)
    if ok.returncode:pytest.skip('Pinned Docker image unavailable')
    c=candidate();ts=tasks();a=adapter(ts)
    try:q=p.qualify(c,ts,{t.physical_key:False for t in ts},a,ENV,tmp_path/'qualify')
    finally:a.close()
    assert q['qualified_for_trial'] and q['dispatched']==4 and q['correct']==4
    assert r.verify_train(q,c,ENV)
    return c,q

def test_actual_docker_invariance_takeover_abstention_replay(qualified,tmp_path,monkeypatch):
    c,q=qualified;ts=tasks();a=adapter(ts);calls=http(monkeypatch,[])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=1500000,request_limit=68,finish_reserve=0)
    trace=r.solve(ts[0],a,[c],{c['id']:q},config(),tmp_path/'task',g)
    assert trace['score']['hard'] and trace['http']==0 and trace['program_calls']==1 and trace['consumed_program_ids']
    assert r.solve(ts[0],a,[c],{c['id']:q},config(),tmp_path/'task',g)==trace and not calls
    ambiguous=PublicTask('a','a','ambiguous',{'choices':[{'label':'A','text':'blue'},{'label':'B','text':'blue'}]})
    missing=PublicTask('n','n','absent',{'choices':[{'label':'A','text':'red'},{'label':'B','text':'green'}]})
    out=p.execute(q['program'],[ambiguous,missing],tmp_path/'negative')
    assert out['valid'] and [d['label'] for d in out['decisions']]==['','']
    changed=deepcopy(trace);changed['prediction']='<answer>A</answer>';changed['trace_hash']=digest({k:v for k,v in changed.items() if k!='trace_hash'})
    with pytest.raises(ValueError):r.verify(changed)
    a.close()


def test_conflicting_programs_fall_back_to_metered_model(qualified,tmp_path,monkeypatch):
    c,q=qualified;v=proposal();v['public_choice_rule']['require_all']=[r'\bred\b']
    other=p.candidate(v,['s'+str(i) for i in range(17)],{'response_hash':'other_fixture'})
    train=[PublicTask(t.task_id,t.physical_key,t.goal,{'choices':[{'label':'A','text':'green'},{'label':'B','text':'red'}]}) for t in tasks()]
    a=adapter(train)
    try:q2=p.qualify(other,train,{t.physical_key:False for t in train},a,ENV,tmp_path/'other')
    finally:a.close()
    task=tasks('val')[0];a=adapter([task]);calls=http(monkeypatch,[response(content='<answer>B</answer>')])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=1500000,request_limit=68,finish_reserve=0)
    trace=r.solve(task,a,[c,other],{c['id']:q,other['id']:q2},config(),tmp_path/'conflict',g)
    assert trace['program_conflict'] and trace['answer_origin']=='dynamic' and not trace['consumed_program_ids']
    assert trace['http']==1 and trace['score']['hard'] and len(calls)==1
    r.verify(trace);a.close()

def test_native_val_publication_and_test_freeze_guard(qualified,tmp_path,monkeypatch):
    c,q=qualified;ts=tasks('val',17);a=adapter(ts);cfg=config()
    calls=http(monkeypatch,[response(content='<answer>A</answer>') for _ in ts])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=3000000,request_limit=68,finish_reserve=0)
    off=[];on=[]
    for task in ts:
        off.append(base.solve(task,a,[],cfg,tmp_path/'off'/task.physical_key,g,
            selection_override={'selected':[],'features':{},'skipped':[],'injection_tokens':0,'routing_http':0}))
        on.append(r.solve(task,a,[c],{c['id']:q},cfg,tmp_path/'on'/task.physical_key,g))
    receipt=r.pair_report(off,on,[c],'val');assert receipt['qualified'] and r.verify_utility(receipt)
    assert receipt['groups']['all']['program_answers']==17 and len(calls)==17
    root=tmp_path/'run';plan={'learning_source_keys':['s'+str(i) for i in range(17)]}
    result={'policy':[c],'qualifications':{c['id']:q},'val_receipt':receipt}
    write_json(root/'config.json',cfg);write_json(root/'selection.json',result)
    frozen=runner.freeze(root,plan,result)
    assert runner.frozen(root)==frozen
    from atomic_skillgraph.empirical.bank import Bank
    bank=Bank(root/'frozen_bank',readonly=True,environment=ENV)
    assert len(bank.all('skill'))==len(bank.all('program'))==len(bank.all('implementation'))==1
    assert bank.program_eligible(bank.all('program')[0]);bank.close()
    changed=root/'selection.json';x=json.loads(changed.read_text());x['policy']=[];write_json(changed,x)
    with pytest.raises(ValueError):runner.frozen(root)
    a.close()


@pytest.mark.skipif(not Path('/home/yangchengyu/livemath_effect_p7_r2_seed42_20261016').exists(),reason='Actual Train evidence required')
def test_real_source_and_baseline_zero_http_prepare(tmp_path,monkeypatch):
    calls=http(monkeypatch,[])
    monkeypatch.setattr(runner,'code_identity',lambda:{'git_sha':'intercepted_fixture','tracked_dirty':False})
    root=tmp_path/'prepared'
    result=runner.prepare(root,'configs/livemath_learned_r3_seed42.yaml',
        '/home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42',
        '/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1')
    assert result['status']=='passed' and result['baseline_reuse_tasks']==29 and not calls
    rows=runner.material(root)['source_materials'];assert len(rows)==60
    assert not any('correct_choice' in json.dumps(x) for x in rows)
    plan=runner.checked(root)
    assert plan['historical_tokens_or_reserved']==36926724 and plan['historical_http']==1897
    with pytest.raises(FileNotFoundError):runner.execute(root)
    assert not calls
    write_json(root/'paid_authorization.json',{'plan_hash':digest(plan),
        'human_instruction':'OFFLINE INTERCEPTED FIXTURE ONLY','phases':plan['settings']['phase_limits']})
    learned=[]
    def no_candidate(*args,**kwargs):learned.append(args);return {'candidates':[],'state':'no_change'}
    monkeypatch.setattr(p,'learn',no_candidate)
    assert runner.execute(root)['status']=='completed_no_candidate'
    assert len(learned)==1 and not (root/'val').exists() and not calls
    assert runner.execute(root)['status']=='completed_no_candidate' and len(learned)==1


def test_billing_refusal_does_not_start_http(tmp_path,monkeypatch):
    from atomic_skillgraph.core.errors import BudgetExhausted
    calls=http(monkeypatch,[]);g=BudgetGovernor(tmp_path/'budget.json',token_limit=1,request_limit=1,finish_reserve=0)
    with pytest.raises(BudgetExhausted):p.learn(tmp_path/'learning',config(),g,{},['s0','s1'])
    assert not calls and not (tmp_path/'learning/request/started.json').exists()


def test_same_payload_sampling_wins_do_not_qualify(tmp_path,monkeypatch):
    ts=tasks('val',17);a=adapter(ts)
    calls=http(monkeypatch,[response(content='<answer>'+label+'</answer>') for _ in ts for label in ('A','B')])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=4000000,request_limit=68,finish_reserve=0)
    off=[];on=[];selection={'selected':[],'features':{},'skipped':[],'injection_tokens':0,'routing_http':0}
    for task in ts:
        off.append(base.solve(task,a,[],config(),tmp_path/'off'/task.physical_key,g,selection_override=selection))
        on.append(base.solve(task,a,[],config(),tmp_path/'on'/task.physical_key,g,selection_override=selection))
    receipt=r.pair_report(off,on,[],'val')
    assert receipt['net_correct']==17 and receipt['groups']['intervened']['tasks']==0 and not receipt['qualified']
    assert len(calls)==34 and r.verify_utility(receipt);a.close()
