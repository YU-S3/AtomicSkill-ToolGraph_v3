"""V2 real Docker and production-provider intercepted regressions; no paid calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, validate_schema_instance
from atomic_skillgraph.empirical.manual_math_v2 import action_schema, action_errors, grounded_operations
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.experiments import run_manual_math_v2 as runner
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http, response
from test_public_process import task

PACKAGE=Path('/mnt/d/T3S_exp/resources/manual_livemath_bank_v2_release_20261015')
PREFLIGHT=Path('/home/yangchengyu/livemath_manual_bank_preflight_20261015')
SOURCE=Path('/home/yangchengyu/livemath_p4_seed42_20261014/plan.json')
HISTORY=Path('/home/yangchengyu/livemath_manual_bank_test_seed42_20261015')
LEDGERS=[HISTORY/'budget.json',Path('/home/yangchengyu/livemath_p5_trace_seed42_20261015/budget.json')]
pytestmark=pytest.mark.skipif(not PACKAGE.exists(),reason='Authored V2 package and local Docker required')


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    patch=pytest.MonkeyPatch();identity=runner.code_identity()
    patch.setattr(runner,'code_identity',lambda:{**identity,'tracked_dirty':False})
    root=tmp_path_factory.mktemp('v2')/'run'
    runner.prepare(PACKAGE,PREFLIGHT,SOURCE,LEDGERS,root)
    yield root
    patch.undo()


def system(root,tmp_path,t=None,enabled=True):
    t=t or task('x',split='test');cfg=runner.read(root/'test/config.json')
    cfg['experiment']['manual_reference_enabled']=enabled
    s=EmpiricalSystem(cfg,readonly=True,harness=AnswerAdapter('livemath',{'x':{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint')
    return s


def action(**kw):return response([kw],name='answer_step')


@pytest.mark.parametrize('value,field',[
    ({'action':'execute_python','arguments':{'code':'print(1)'}},'source'),
    ({'action':'execute_python','source':'x','arguments':{},'input_schema':[],'output_schema':{}},'$'),
    ({'action':'finish','answer':'A','source':'x'},'source'),
    ({'action':'call_program','arguments':{}},'name'),
    ({'action':'read_result','result_id':'x'},'path')])
def test_action_contract(value,field):
    assert any(r['field_path']==field for r in action_errors(value))
    with pytest.raises(ValueError):validate_schema_instance(value,action_schema())


def test_routing_and_unknown_premises(prepared,tmp_path,monkeypatch):
    calls=http(monkeypatch,[response(content='<answer>A</answer>')])
    t=task('x',split='test');t=PublicTask(t.task_id,t.physical_key,'What color is the car?',
        {'choices':[{'label':'A','text':'red'},{'label':'B','text':'blue'}]},t.split)
    s=system(prepared,tmp_path,t)
    try:r=s.run_task(t,learn=False)
    finally:s.close()
    wire=json.loads(calls[0]['messages'][1]['content'])
    assert wire['guidance']==wire['programs']==wire['workflows']==wire['reference_knowledge']==[]
    assert r['execution']['reference_routing']['injected']=={'skills':[],'programs':[],'workflows':[]}


def test_real_program_and_workflow_consumption(prepared,tmp_path,monkeypatch):
    t=task('x',split='test');t=PublicTask(t.task_id,t.physical_key,'Compute (1/2)+(1/3).',t.inputs,t.split)
    s=system(prepared,tmp_path,t)
    from atomic_skillgraph.empirical.task_context import TaskContext
    s.bank.router.bind(t,TaskContext(s.config['runtime']))
    wid=s.bank.router.workflows[0]
    final=action(action='finish',answer='<answer>A</answer>')
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments={}),response(content=json.dumps({
        'status':'ok','outputs':{'analysis':'The local sum is 5/6; no other theorem is established.'},'references':[]})),final])
    from atomic_skillgraph.agents import provider
    post=provider.requests.post
    def consume(*a,**kw):
        material=json.loads(kw['json']['messages'][1]['content'])
        if material.get('local_results'):
            final['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps({
                'action':'finish','answer':'<answer>A</answer>','used_results':[material['local_results'][0]['result_id']]})
        return post(*a,**kw)
    monkeypatch.setattr(provider.requests,'post',consume)
    try:r=s.run_task(t,learn=False)
    finally:s.close()
    w=r['execution']['workflow_attempts'][0]
    assert w['status']=='ok' and w['outputs_consumed'] and w['program_executions']==1
    assert w['nodes'][0]['result']['outputs']['exact']=='5/6'
    assert '--network=none' in w['nodes'][0]['result']['execution_receipt']['command']
    assert [c['max_tokens'] for c in calls]==[65536,4096,16384]
    assert r['execution']['reference_routing']['used']==[wid]


@pytest.mark.parametrize('override',[False,True])
def test_direct_program_readiness_and_consumption(prepared,tmp_path,monkeypatch,override):
    t=task('x',split='test');t=PublicTask(t.task_id,t.physical_key,'Compute gcd(84,30).',t.inputs,t.split)
    s=system(prepared,tmp_path,t)
    from atomic_skillgraph.empirical.task_context import TaskContext
    s.bank.router.bind(t,TaskContext(s.config['runtime']));pid=s.bank.router.programs[0]
    arguments=s.bank.router.binding(pid)['arguments']
    if override:arguments['a']=999
    final=action(action='finish',answer='<answer>A</answer>') if not override else response(content='<answer>A</answer>')
    calls=http(monkeypatch,[action(action='call_program',name=pid,arguments=arguments),final])
    from atomic_skillgraph.agents import provider
    post=provider.requests.post
    def consume(*a,**kw):
        material=json.loads(kw['json']['messages'][1]['content'])
        if material.get('local_results') and not override:
            final['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps({
                'action':'finish','answer':'<answer>A</answer>','used_results':[material['local_results'][0]['result_id']]})
        return post(*a,**kw)
    monkeypatch.setattr(provider.requests,'post',consume)
    try:r=s.run_task(t,learn=False)
    finally:s.close()
    assert len(calls)==2
    if override:assert not r['execution']['attempts']
    else:
        a=r['execution']['attempts'][0]
        assert a['program_executed'] and a['outputs_consumed']
        assert r['result_store']['program_results'][a['result_id']]['outputs']['value']==6


def test_invalid_action_one_feedback_then_finish(prepared,tmp_path,monkeypatch):
    invalid=action(action='execute_python',arguments={'code':'print(1)'})
    calls=http(monkeypatch,[invalid,deepcopy(invalid),response(content='<answer>A</answer>')])
    s=system(prepared,tmp_path)
    try:r=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    assert [c['max_tokens'] for c in calls]==[65536,4096,16384]
    assert len(r['execution']['action_contract_failures'])==2
    assert r['execution']['temporary_executions']==[]
    assert not r['execution'].get('budget_censored')
    assert json.loads(calls[1]['messages'][1]['content'])['action_contract_feedback'][0]['field_path']=='source'


def test_bad_worker_return_finalizes_without_second_64k(prepared,tmp_path,monkeypatch):
    calls=http(monkeypatch,[action(action='execute_python',source='def run(ctx, inputs):\n    return {"value":1}',
        arguments={},input_schema={'type':'object'},output_schema={'type':'object'}),response(content='<answer>A</answer>')])
    s=system(prepared,tmp_path)
    try:r=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    assert len(calls)==2 and calls[1]['max_tokens']==16384
    assert r['execution']['choice_bounded_finalization']['trigger']=='worker_failed'
    assert r['execution']['temporary_executions'][0]['status']!='ok'
    assert r['execution']['answer_status']=='valid' and not r['execution'].get('budget_censored')


@pytest.mark.parametrize('bad',['unknown','override','fake_truth'])
def test_unready_workflow_never_executes(prepared,tmp_path,monkeypatch,bad):
    t=task('x',split='test');t=PublicTask(t.task_id,t.physical_key,'Compute (1/2)+(1/3).',t.inputs,t.split)
    s=system(prepared,tmp_path,t)
    from atomic_skillgraph.empirical.task_context import TaskContext
    s.bank.router.bind(t,TaskContext(s.config['runtime']))
    wid=s.bank.router.workflows[0]
    if bad=='unknown':wid='invented'
    args={'goal':'overwrite'} if bad=='override' else {'domain_complete':True,'truth_values':[True,False]} if bad=='fake_truth' else {}
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments=args),response(content='<answer>A</answer>')])
    try:r=s.run_task(t,learn=False)
    finally:s.close()
    assert r['execution']['workflow_attempts'][0]['program_executions']==0
    assert calls[-1]['max_tokens']==16384


def test_truncation_and_recovery(prepared,tmp_path,monkeypatch):
    calls=http(monkeypatch,[response(finish='length'),response(content='<answer>A</answer>')])
    s=system(prepared,tmp_path)
    try:r=s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')
    finally:s.close()
    assert r['execution']['choice_bounded_finalization']['valid_answer']
    assert not r['execution'].get('budget_censored')
    s=system(prepared,tmp_path)
    try:second=s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')
    finally:s.close()
    assert r['execution']==second['execution'] and len(calls)==2


@pytest.mark.parametrize('boundary',['truncation','worker_failure'])
def test_finish_crash_recovers_paid_response_once(prepared,tmp_path,monkeypatch,boundary):
    first=response(finish='length') if boundary=='truncation' else action(action='execute_python',
        source='def run(ctx, inputs):\n    return {"value":1}',arguments={},input_schema={'type':'object'},output_schema={'type':'object'})
    calls=http(monkeypatch,[first,response(content='<answer>A</answer>')])
    commit=TaskCheckpoint.commit_decision;crashed=[]
    def interrupted(self,key,status,**values):
        if values.get('answer_executor',{}).get('finished') and not crashed:
            crashed.append(True);raise RuntimeError('fixture crash after final response')
        return commit(self,key,status,**values)
    monkeypatch.setattr(TaskCheckpoint,'commit_decision',interrupted)
    s=system(prepared,tmp_path)
    try:
        with pytest.raises(RuntimeError,match='fixture crash'):s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')
    finally:s.close()
    assert len(calls)==2
    s=system(prepared,tmp_path)
    try:r=s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')
    finally:s.close()
    assert r['execution']['answer_status']=='valid' and len(calls)==2


def test_final_budget_refusal_is_not_claimed_as_128k_spent(prepared,tmp_path,monkeypatch):
    calls=http(monkeypatch,[response(finish='length')]);s=system(prepared,tmp_path)
    original=s.agent
    def agent(*args,**kwargs):
        if kwargs.get('decision_purpose')=='finish_only':
            from atomic_skillgraph.core.errors import BudgetExhausted
            raise BudgetExhausted('parent_task_budget_exhausted','Fixture exact wire admission refused')
        return original(*args,**kwargs)
    monkeypatch.setattr(s,'agent',agent)
    try:r=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    assert len(calls)==1
    assert r['execution']['reason']=='finish_budget_inadmissible'
    assert r['execution']['choice_bounded_finalization']['budget_state']['metered_task_tokens']==5


def test_stage_preserves_finish_reservation(tmp_path):
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    ledger={'limits':{'token_limit':10000,'request_limit':10,'finish_reserve':0},'attempts':{},'unknown_billing':False}
    runner.write_json(tmp_path/'owner.json',ledger);runner.write_json(tmp_path/'phase/campaign_budget_prefix.json',ledger)
    owner=BudgetGovernor(tmp_path/'owner.json',**ledger['limits'])
    stage=runner.ReferenceStageGovernor(owner,{'new_limits':{'token_limit':5000,'request_limit':3}},tmp_path/'phase')
    assert stage.headroom({'parent_finish_reserve':2000})['tokens']==3000


def test_exact_operands_no_fabricated_certificate():
    t=PublicTask('probe','probe',r'Compute \frac{1}{2} + \frac{1}{3}.',{'choices':[]})
    assert grounded_operations(t)['exact_rational']['arguments']['a']['denominator']==2
    t=PublicTask('probe','probe','A relation on an infinite set is specified abstractly.',{'choices':[]})
    assert grounded_operations(t)=={}
    t=PublicTask('probe','probe',r'Compute \det\begin{pmatrix}1&2\\3&4\end{pmatrix}.',{'choices':[]})
    assert grounded_operations(t)['exact_matrix']['arguments']['matrix']==[['1','2'],['3','4']]


def test_registry_requires_premises_and_no_gold(prepared,tmp_path):
    t=task('x',split='test');t=PublicTask(t.task_id,t.physical_key,'A contraction mapping is defined. Which statement follows?',t.inputs,t.split)
    s=system(prepared,tmp_path,t)
    try:
        from atomic_skillgraph.empirical.task_context import TaskContext
        s.bank.router.bind(t,TaskContext(s.config['runtime']))
        row=s.bank.router.knowledge()[0]
        assert row['coverage_status']=='insufficient_evidence' and not row['verified_assumptions']
        assert row['unverified_assumptions'] and 'gold' not in json.dumps(row)
        assert all(not s.bank.bank.program_eligible(p) for p in s.bank.all('program'))
    finally:s.close()


def test_real_ten_failures_replay_without_http(prepared,tmp_path,monkeypatch):
    traces=[runner.read(p) for p in (HISTORY/'test/traces').glob('*.json')]
    failed=[t for t in traces if t['execution'].get('budget_censored')]
    assert len(failed)==10
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    evidence=[]
    for n,old in enumerate(failed):
        first=old['requests'][0];turn=first['response'];calls_raw=turn['tool_calls']
        # Original ToolCall and raw_usage are replayed unchanged; the final answer
        # is a labelled fixture, not a new model prediction or corrected old score.
        saved=response([c['arguments'] for c in calls_raw],name=calls_raw[0]['name'],finish=turn['finish_reason'],content=turn['content'])
        saved['usage']=first['http_attempts'][0]['raw_usage']
        original_errors=[action_errors(c['arguments']) for r in old['requests'] for c in r.get('response',{}).get('tool_calls',[])]
        assert any(original_errors)
        first_invalid=bool(action_errors(calls_raw[0]['arguments']))
        sequence=[saved,*([action(action='execute_python',arguments={})] if first_invalid else []),response(content='<answer>A</answer>')]
        calls=http(monkeypatch,sequence)
        t=PublicTask(**old['task']);s=system(prepared,tmp_path/str(n),t)
        s.adapter._records={t.task_id:{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}
        try:r=s.run_task(t,learn=False)
        finally:s.close()
        assert [c['max_tokens'] for c in calls]==([65536,4096,16384] if first_invalid else [65536,16384])
        assert len(r['execution']['action_contract_failures'])==(2 if first_invalid else 0)
        assert not r['execution'].get('budget_censored')
        assert len(r['execution']['temporary_executions'])==(0 if first_invalid else 1)
        wire=calls[-1];material=json.loads(wire['messages'][1]['content'])
        assert material['goal']==t.goal and material['inputs']==t.inputs
        assert saved['usage']['total_tokens']+input_token_bound(wire)+16384 < 128000
        evidence.append({'task_id':t.task_id,'source_request':first['id'],'raw_usage':saved['usage'],
            'errors':original_errors,'replay_caps':[c['max_tokens'] for c in calls],
            'final_wire_bytes':input_token_bound(wire),'final_answer':'synthetic_fixture_not_performance_evidence'})
    runner.write_json(prepared/'ten_failure_replay.json',{'new_http':0,'evidence':evidence})


def test_all_public_train_val_routing(prepared,tmp_path):
    from atomic_skillgraph.empirical.task_context import TaskContext
    tasks=runner.read(SOURCE)['tasks'];s=system(prepared,tmp_path)
    counts=[]
    try:
        for phase in ('train','val'):
            for row in tasks[phase]:
                s.bank.router.bind(PublicTask(**row),TaskContext(s.config['runtime']))
                audit=s.bank.router.audit
                assert len(audit['injected']['skills'])<=2 and len(audit['injected']['programs'])<=2 and len(audit['injected']['workflows'])<=1
                for pid in audit['injected']['programs']:
                    s.bank.router.validate_call(pid,s.bank.router.binding(pid)['arguments'])
                counts.append({'task_id':row['task_id'],'injected':audit['injected']})
    finally:s.close()
    assert len(counts)==77
    runner.write_json(prepared/'train_val_routing.json',{'new_http':0,'tasks':counts})


def test_full_control_flow_same_conditions_and_readonly(prepared,monkeypatch,tmp_path):
    calls=http(monkeypatch,[response(content='<answer>A</answer>') for _ in range(106)])
    monkeypatch.setattr(runner,'load_env',lambda _:None)
    before=tree_identity(prepared/'frozen_bank')
    result=runner.execute(prepared,'unused')
    assert result['status']=='completed' and result['tasks']==100 and len(calls)==106
    assert runner.execute(prepared,'unused')==result and len(calls)==106
    assert tree_identity(prepared/'frozen_bank')==before
    for off,on in zip(calls[:3],calls[3:6]):
        a,b=json.loads(off['messages'][1]['content']),json.loads(on['messages'][1]['content'])
        assert a['goal']==b['goal'] and a['inputs']==b['inputs']
        assert off['max_tokens']==on['max_tokens']==65536
        assert a['guidance']==a['programs']==a['workflows']==[]
