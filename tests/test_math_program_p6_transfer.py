"""Intercepted Provider plus actual Docker: P6 learning, transfer, readonly consumption."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import run_math_program_p6 as runner
from atomic_skillgraph.experiments.run_empirical import write_json
from test_cf2_contracts import http, response
from test_math_program_p6_contract import c,m,rows,proposal,config,FIXTURE


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    if not runner.p5.docker_ready()['ready']:pytest.skip('Locked Docker required')
    root=tmp_path/'run';cfg=config(tmp_path);cfg['data_dir']=str(root/'bank');cfg['budget']={'token_limit':45000000,'request_limit':9165,'finish_reserve':0}
    cfg['program_worker']['record_execution_receipt']=True
    bank=Bank(root/'bank',environment=cfg['program_environment']);initial=bank.digest();bank.close()
    units=m.catalog(rows())[1]
    owner=BudgetGovernor(root/'budget.json',**cfg['budget'])
    plan={'code':{'fixture':True},'run_id':'p6_fixture','initial_bank_digest':initial,
        'transfer_physical_keys':['2'],'smoke_physical_keys':['2'],
        'settings':{'cumulative_limits':cfg['budget'],'stage_limits_proposed_not_authorized':{
            'learning':{'token_limit':1000000,'request_limit':100},'smoke':{'token_limit':400000,'request_limit':30}}}}
    for name,obj in [('config.json',cfg),('source_catalog.json',rows()),('atomic_source_catalog.json',[]),
        ('units.json',units),('campaign_budget_prefix.json',owner.state),('learning/campaign_budget_prefix.json',owner.state)]:write_json(root/name,obj)
    monkeypatch.setattr(runner,'checked',lambda path:plan)
    return root,plan


def learn_fixture(prepared,monkeypatch):
    root,plan=prepared
    seen=http(monkeypatch,[response([proposal()],name='submit_math_program_spec'),response([{'source':FIXTURE}],name='submit_program')])
    result=runner.learn(root);assert result['status']=='completed_trial_candidates'
    assert result['units'][0]['task_transfer_checked'];assert result['usable_programs']==0
    assert runner.learn(root)==result and len(seen)==2
    manifest=runner.trial_bank(root);return manifest


def test_real_learn_bind_call_read_finish(prepared,monkeypatch):
    root,plan=prepared;manifest=learn_fixture(prepared,monkeypatch);pid=next(iter(manifest['program_hashes']))
    task=PublicTask(**rows()[2]['task']);cfg=runner.smoke_config(root,'program')
    adapter=AnswerAdapter('livemath',{task.task_id:{'correct_choice':task.inputs['choices'][0],'choices':task.inputs['choices']}})
    s=EmpiricalSystem(cfg,harness=adapter,readonly=True)
    from atomic_skillgraph.empirical.task_context import TaskContext
    # Use a known ID solely for deterministic protocol fixtures; the production context still stores actual Worker results.
    original=TaskContext.register
    def register(ctx,*args,**kwargs):
        rid=original(ctx,*args,**kwargs)
        if kwargs.get('name')==pid:
            register.rid=rid
        return rid
    monkeypatch.setattr(TaskContext,'register',register)
    def replies():
        yield response([{'action':'call_program','name':pid,'arguments':{'statement':task.goal}}],name='answer_step')
        yield response([{'action':'read_result','result_id':register.rid,'path':['outputs','quantifiers']}],name='answer_step')
        yield response([{'action':'finish','answer':'<answer>A</answer>','used_results':[register.rid]}],name='answer_step')
    seen=http(monkeypatch,replies())
    before=s.bank.digest()
    try:trace=s.run_task(task,learn=False)
    finally:s.close()
    execution=trace['execution'];assert trace['score']['hard']
    assert execution['attempts'][0]['status']=='ok' and execution['attempts'][0]['outputs_consumed']
    layer=execution['program_usage_layers'][0]
    assert layer['worker_invoked'] and layer['result_read'] and layer['result_exposed'] and layer['used_results_declared']
    assert layer['task_correct'] and not layer['downstream_handoff']
    assert s.request_attribution['parent_token_limit']==128000 and not s.request_attribution['train_parent']
    assert len(seen)==3
    bank=Bank(root/'trial_bank',readonly=True,environment=cfg['program_environment'])
    assert bank.digest()==before and not bank.program_eligible(bank.get(pid));bank.close()


def test_unbound_task_and_control_have_empty_programs(prepared,monkeypatch):
    root,plan=prepared;learn_fixture(prepared,monkeypatch)
    for arm,goal in [('program','There exists a unique solution of this infinite PDE.'),('off',rows()[2]['task']['goal'])]:
        cfg=runner.smoke_config(root,arm);task=PublicTask('fixture:unbound','unbound',goal,rows()[2]['task']['inputs'])
        seen=http(monkeypatch,[response([{'action':'finish','answer':'<answer>A</answer>'}],name='answer_step')])
        s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{task.task_id:{'correct_choice':task.inputs['choices'][0],'choices':task.inputs['choices']}}),readonly=True)
        try:trace=s.run_task(task,learn=False)
        finally:s.close()
        material=json.loads(seen[0]['messages'][1]['content'])
        assert material['programs']==[] and material['guidance']==[]
        assert trace['execution']['attempts']==[]


def test_action_error_one_feedback_then_valid_final(prepared,monkeypatch):
    root,plan=prepared;learn_fixture(prepared,monkeypatch);cfg=runner.smoke_config(root,'program');task=PublicTask(**rows()[2]['task'])
    seen=http(monkeypatch,[response([{'action':'execute_python','arguments':{'code':'print(1)'}}],name='answer_step'),
        response([{'action':'finish','answer':'<answer>A</answer>'}],name='answer_step')])
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{task.task_id:{'correct_choice':task.inputs['choices'][0],'choices':task.inputs['choices']}}),readonly=True)
    try:trace=s.run_task(task,learn=False)
    finally:s.close()
    assert len(trace['execution']['action_contract_failures'])==1 and not trace['execution']['temporary_executions']
    assert seen[1]['max_tokens']==4096 and trace['execution']['answer_status']=='valid'


def test_all_abstain_exits_without_smoke_http(prepared,monkeypatch):
    root,plan=prepared
    seen=http(monkeypatch,[response([{'contract_version':c.CONTRACT,'decision':'no_change'}],name='submit_math_program_spec')])
    assert runner.learn(root)['status']=='completed_no_candidate'
    assert runner.smoke(root)=={'status':'completed_no_bindable_program','new_http':0}
    assert len(seen)==1


def test_wrong_parameters_never_reach_worker(prepared,monkeypatch):
    root,plan=prepared;manifest=learn_fixture(prepared,monkeypatch);pid=next(iter(manifest['program_hashes']))
    cfg=runner.smoke_config(root,'program');task=PublicTask(**rows()[2]['task'])
    seen=http(monkeypatch,[response([{'action':'call_program','name':pid,'arguments':{'statement':r'\forall z made up'}}],name='answer_step'),
        response(content='<answer>A</answer>')])
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{task.task_id:{'correct_choice':task.inputs['choices'][0],'choices':task.inputs['choices']}}),readonly=True)
    monkeypatch.setattr(s.worker,'execute',lambda *a:pytest.fail('Unbound arguments reached Worker'))
    try:trace=s.run_task(task,learn=False)
    finally:s.close()
    assert not trace['execution']['attempts'] and trace['execution']['answer_status']=='valid'
    assert seen[-1]['max_tokens']==16384


def test_trial_resume_does_not_rewrite_modified_bank(prepared,monkeypatch):
    root,plan=prepared;manifest=learn_fixture(prepared,monkeypatch)
    from atomic_skillgraph.experiments.formal_log import tree_identity
    before=tree_identity(root/'trial_bank')
    assert runner.trial_bank(root)==manifest and tree_identity(root/'trial_bank')==before
    bank=Bank(root/'trial_bank',environment=runner.read(root/'config.json')['program_environment'])
    bank.put('skill',{'goal':'Foreign asset','state':'candidate'});bank.close()
    changed=tree_identity(root/'trial_bank')
    with pytest.raises(ValueError,match='trial Bank or evidence changed'):runner.trial_bank(root)
    assert tree_identity(root/'trial_bank')==changed


def test_unbound_smoke_skips_both_arms(prepared,monkeypatch):
    root,plan=prepared;learn_fixture(prepared,monkeypatch)
    data=runner.read(root/'source_catalog.json');data.append({**data[2],'task':{**data[2]['task'],'physical_key':'unbound','goal':'Abstract theorem'}})
    write_json(root/'source_catalog.json',data);plan['smoke_physical_keys']=['unbound']
    seen=http(monkeypatch,[])
    assert runner.smoke(root)=={'status':'completed_no_bindable_program','new_http':0} and not seen


def test_budget_stop_and_unresolved_http_never_send(prepared,monkeypatch):
    root,plan=prepared;seen=http(monkeypatch,[])
    plan['settings']['stage_limits_proposed_not_authorized']['learning']['token_limit']=1
    from atomic_skillgraph.core.errors import BudgetExhausted
    with pytest.raises(BudgetExhausted):runner.learn(root)
    assert not seen
    plan['settings']['stage_limits_proposed_not_authorized']['learning']['token_limit']=1000000
    owner=BudgetGovernor(root/'budget.json',**plan['settings']['cumulative_limits'])
    owner.admit('unresolved',{'messages':[],'max_tokens':10},{'parent_scope':'p6_fixture:unknown'})
    with pytest.raises(BudgetExhausted):runner.learn(root)
    assert not seen


def test_failed_transfer_is_preserved_and_runtime_rechecks(prepared,monkeypatch):
    root,plan=prepared
    wrong=FIXTURE.replace('    if not tokens:return',"    if text.endswith(' 2'):tokens=tokens[:-1]\n    if not tokens:return")
    http(monkeypatch,[response([proposal()],name='submit_math_program_spec'),response([{'source':wrong}],name='submit_program')])
    summary=runner.learn(root)
    assert summary['units'][0]['local_algorithm_checked'] and not summary['units'][0]['task_transfer_checked']
    manifest=runner.trial_bank(root);pid=next(iter(manifest['program_hashes']))
    task=PublicTask(**rows()[2]['task']);cfg=runner.smoke_config(root,'program')
    seen=http(monkeypatch,[response([{'action':'call_program','name':pid,'arguments':{'statement':task.goal}}],name='answer_step'),
        response(content='<answer>A</answer>')])
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{task.task_id:{'correct_choice':task.inputs['choices'][0],'choices':task.inputs['choices']}}),readonly=True)
    try:trace=s.run_task(task,learn=False)
    finally:s.close()
    assert trace['execution']['attempts'][0]['status']=='blocked'
    assert not trace['execution']['attempts'][0]['outputs_consumed'] and len(seen)==2
