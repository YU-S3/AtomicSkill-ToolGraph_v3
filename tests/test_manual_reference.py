"""External catalog, real Docker dataflow and intercepted production HTTP control arm."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.manual_reference import ReferenceView, VERSION
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments import run_manual_reference_test as runner
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http, response
from test_public_process import task

PACKAGE=Path('/mnt/d/T3S_exp/resources/manual_livemath_bank_20261015')
PREFLIGHT=Path('/home/yangchengyu/livemath_manual_bank_preflight_20261015')
SOURCE=Path('/home/yangchengyu/livemath_p4_seed42_20261014/plan.json')
LEDGER=Path('/home/yangchengyu/livemath_p5_program_probe_seed42_20261015/budget.json')
pytestmark=pytest.mark.skipif(not PACKAGE.exists(),reason='Authored package and locked Docker receipts unavailable')


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    identity=runner.code_identity()
    monkeypatch.setattr(runner,'code_identity',lambda:{**identity,'tracked_dirty':False})
    root=tmp_path/'manual'
    result=runner.prepare(PACKAGE,PREFLIGHT,SOURCE,LEDGER,root)
    assert result['status']=='passed'
    return root


def system(root,t=None):
    t=t or task('x',split='test')
    cfg=runner.read(root/'config.json')
    s=EmpiricalSystem(cfg,readonly=True,harness=AnswerAdapter('livemath',{'x':{
        'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
    s.checkpoint=TaskCheckpoint(root/'fixture_checkpoint')
    return s


def node(outputs,*,references=True,status='ok'):
    return response(finish='stop',content=json.dumps({'status':status,'outputs':outputs,
        'references':[{'task_path':['goal'],'quote':'Suppose'}] if references else []}))


def action(**kwargs):return response([kwargs],name='answer_step')


def test_import_does_not_forge_ours_qualification(prepared):
    root=prepared;before=tree_identity(root/'frozen_bank')['sha256']
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        assert [len(bank.all(k)) for k in ('skill','program','implementation','workflow')]==[12,6,7,5]
        assert all(not bank.program_eligible(p) and p['state']=='candidate' for p in bank.all('program'))
        view=ReferenceView(bank,root/'reference_manifest.json')
        assert len(view.program_options(''))==6 and len(view.workflow_options())==5
        assert all(view.program_eligible(p) for p in bank.all('program'))
    finally:bank.close()
    assert before==tree_identity(root/'frozen_bank')['sha256']
    manifest=runner.read(root/'reference_manifest.json')
    ref=next(iter(manifest['program_receipts'].values()))[0]
    receipt=root/ref['file'];receipt.write_text('{}')
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        with pytest.raises(ValueError,match='receipt changed'):ReferenceView(bank,root/'reference_manifest.json')
    finally:bank.close()


CASES=[
    (0,{'premises':[[True,False]],'conclusion':[True,False],'domain_complete':True,'true_label':'A','false_label':'B'},[],2),
    (1,{'choices':[{'label':'A','truth_values':[True,True]},{'label':'B','truth_values':[False,True]}],
        'mode':'all','domain_complete':True},[],1),
    (2,{},[node({'symbols':['p'],'premises':[{'var':'p'}],'conclusion':{'var':'p'}}),
        node({'analysis':'The finite conclusion follows.'},references=False),node({'answer':'<answer>A</answer>'},references=False)],1),
    (3,{},[node({'relation':[[True,False],[False,True]]}),node({'analysis':'Every row has a witness, but no column covers all rows.'},references=False)],1),
    (4,{},[node({'operation':'add','a':{'numerator':1,'denominator':2},'b':{'numerator':1,'denominator':3}}),
        node({'analysis':'The exact sum is 5/6.'},references=False)],1),
]


@pytest.mark.parametrize('index,args,nodes,programs',CASES)
def test_all_saved_graphs_use_real_worker_dataflow(prepared,monkeypatch,index,args,nodes,programs):
    source=runner.read(PACKAGE/'banks/livemath/bank.json');wid=source['workflows'][index]['id']
    final=action(action='finish',answer='<answer>A</answer>')
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments=args),*nodes,final])
    from atomic_skillgraph.agents import provider
    post=provider.requests.post
    def consume(*a,**kw):
        material=json.loads(kw['json']['messages'][1]['content'])
        if material.get('local_results'):
            final['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps({
                'action':'finish','answer':'<answer>A</answer>',
                'used_results':[material['local_results'][0]['result_id']]})
        return post(*a,**kw)
    monkeypatch.setattr(provider.requests,'post',consume)
    s=system(prepared)
    try:trace=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    graph=trace['execution']['workflow_attempts'][0]
    assert graph['status']=='ok',graph
    assert graph['program_executions']==programs and graph['outputs_consumed'] and trace['score']['hard']
    assert len(calls)==len(nodes)+2
    assert len(trace['injected_guidance_ids'])==12
    first=json.loads(calls[0]['messages'][1]['content'])
    assert len(first['programs'])==6 and len(first['workflows'])==5 and len(first['guidance'])==12
    for row in graph['nodes']:
        if row['mode']=='program':
            receipt=row['result']['execution_receipt']
            assert receipt['exit_code']==0 and '--network=none' in receipt['command']
    if index==0:assert graph['nodes'][1]['arguments']['entailed'] is True
    if index==2:assert graph['nodes'][1]['arguments']['symbols']==['p']
    for wire in calls:
        material=json.loads(wire['messages'][1]['content'])
        assert material['goal']==task('x').goal and material['inputs']==task('x').inputs
        assert not any(k in material for k in ('gold','correct_choice','evaluator_records'))
        assert wire['max_tokens']==65536 and wire['thinking']=={'type':'enabled'} and wire['reasoning_effort']=='high'


@pytest.mark.parametrize('boundary',['dynamic','program'])
def test_dynamic_response_recovery_without_duplicate_http(prepared,monkeypatch,boundary):
    wid=runner.read(PACKAGE/'banks/livemath/bank.json')['workflows'][4]['id']
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments={}),*CASES[4][2],
        action(action='finish',answer='<answer>A</answer>')])
    commit=TaskCheckpoint.commit_decision;crashed=[]
    def interrupted(self,key,status,**values):
        states=values.get('manual_workflows',{})
        if states and any(s['nodes'] and s['nodes'][-1]['mode']==boundary for s in states.values()) and not crashed:
            crashed.append(True);raise RuntimeError('fixture crash after paid node')
        return commit(self,key,status,**values)
    monkeypatch.setattr(TaskCheckpoint,'commit_decision',interrupted)
    s=system(prepared)
    try:
        with pytest.raises(RuntimeError,match='fixture crash'):s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')
    finally:s.close()
    assert len(calls)==2
    s=system(prepared)
    try:r=s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')
    finally:s.close()
    assert r['score']['hard'] and r['execution']['workflow_attempts'][0]['status']=='ok'
    assert len(calls)==4
    s=system(prepared)
    try:assert s.run_task(task('x',split='test'),learn=False,attempt_id='x:1')['score']['hard']
    finally:s.close()
    assert len(calls)==4


def test_direct_program_submission_uses_supplied_source(prepared,monkeypatch):
    pid=runner.read(PACKAGE/'banks/livemath/bank.json')['programs'][5]['id']
    calls=http(monkeypatch,[action(action='call_program',name=pid,
        arguments={'entailed':True,'true_label':'A','false_label':'B','domain_complete':True})])
    s=system(prepared)
    try:r=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    assert r['score']['hard'] and len(calls)==1
    assert r['execution']['attempts'][0]['submission_by_program']


@pytest.mark.parametrize('invalid',['unknown_id','missing_args','ungrounded_source'])
def test_invalid_graph_inputs_stop_locally(prepared,monkeypatch,invalid):
    workflows=runner.read(PACKAGE/'banks/livemath/bank.json')['workflows']
    wid=workflows[2 if invalid=='ungrounded_source' else 0]['id'] if invalid!='unknown_id' else 'invented'
    nodes=[node({'symbols':['p'],'premises':[],'conclusion':{'var':'p'}},references=False)] if invalid=='ungrounded_source' else []
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments={}),*nodes,action(action='finish',answer='<answer>A</answer>')])
    s=system(prepared)
    try:r=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    w=r['execution']['workflow_attempts'][0]
    assert w['status']=='needs_input' and w['program_executions']==0 and len(calls)==2+len(nodes)


def test_truncation_finalizer_keeps_full_input_and_parent_state(prepared,monkeypatch):
    calls=http(monkeypatch,[response(finish='length'),response(finish='stop',content='<answer>A</answer>')])
    s=system(prepared)
    try:r=s.run_task(task('x',split='test'),learn=False)
    finally:s.close()
    assert r['score']['hard'] and not r['execution'].get('budget_censored')
    final=r['execution']['choice_bounded_finalization']
    assert final['attempted'] and final['valid_answer'] and not final['parent_budget_exhausted']
    assert calls[1]['max_tokens']==16384 and calls[1]['reasoning_effort']=='low'
    material=json.loads(calls[1]['messages'][1]['content'])
    assert material['goal']==task('x').goal and material['inputs']==task('x').inputs


def test_all_100_production_payloads_budgets_and_readonly_resume(prepared,monkeypatch,tmp_path):
    calls=http(monkeypatch,[response(finish='stop',content='<answer>A</answer>') for _ in range(100)])
    # Offline fixture: no credential or network access; actual canonical inputs and official scoring path.
    monkeypatch.setattr(runner,'load_env',lambda p:None)
    result=runner.execute(prepared,tmp_path/'unused.env')
    assert result['status']=='completed' and result['tasks']==100 and len(calls)==100
    assert runner.execute(prepared,tmp_path/'unused.env')==result and len(calls)==100
    audit=runner.read(prepared/'freeze_audit.json');assert audit['before']==audit['after']
    plan=runner.read(prepared/'plan.json');budget=runner.read(prepared/'budget.json')
    prefix=runner.read(prepared/'campaign_budget_prefix.json')
    assert all(budget['attempts'][k]==v for k,v in prefix['attempts'].items())
    assert len(budget['attempts'])-len(prefix['attempts'])==100
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    for task_row,wire in zip(plan['tasks'],calls):
        material=json.loads(wire['messages'][1]['content'])
        assert material['goal']==task_row['goal'] and material['inputs']==task_row['inputs']
        assert input_token_bound(wire)+wire['max_tokens']<=128000
    assert len(list((prepared/'test/traces').glob('*.json')))==100
