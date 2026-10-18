"""Finite autonomous condition edits and exact physical-response reuse."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from atomic_skillgraph.empirical import math_predicate_evolution as e,math_effect_program_r3 as p
from atomic_skillgraph.empirical import math_effect_runtime_r3 as r,math_effect_runtime as base
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.experiments.run_empirical import write_json
from test_math_effect_r3 import config,proposal,candidate,tasks,adapter,ENV,qualified
from test_cf2_contracts import http,response


def test_generic_neighborhood_uses_only_parent_features():
    v=proposal();v['public_choice_rule']['exclude_any']=['navy']
    parent=p.candidate(v,['s0','s1'],{'response_hash':'fixture'})
    children=e.neighborhood(parent,['s0','s1'],config()['effect_p7']['tokenizer_path'])
    assert len(children)==2
    assert children[0]['candidate']['public_choice_rule']=={'require_all':[r'\bblue\b'],'exclude_any':[]}
    assert children[1]['candidate']['public_choice_rule']=={'require_all':['navy'],'exclude_any':[]}
    assert all(x['parent_ids']==[parent['id']] and not x['model_evidence']['natural_model_candidate'] for x in children)
    assert all(x['candidate']['rule_basis']=='benchmark_prior' for x in children)
    assert e.neighborhood(candidate(False),['s0','s1'],config()['effect_p7']['tokenizer_path'])==[]


def test_exact_reuse_no_http_and_no_test_replay(tmp_path,monkeypatch):
    task=tasks('val')[0];a=adapter([task]);cfg=config();c=candidate(False)
    calls=http(monkeypatch,[response(content='<answer>B</answer>')])
    original=BudgetGovernor(tmp_path/'original_budget.json',token_limit=150000,request_limit=2,finish_reserve=0)
    original_trace=r.solve(task,a,[c],{},cfg,tmp_path/'original',original)
    replay_budget=BudgetGovernor(tmp_path/'replay_budget.json',token_limit=0,request_limit=0,finish_reserve=0)
    replay=r.solve(task,a,[c],{},cfg,tmp_path/'replayed',replay_budget,
        reuse_trace=tmp_path/'original/trace.json',require_reuse=True)
    assert len(calls)==1 and replay['reused_model_response']['new_http']==0
    assert replay['request_ids']==original_trace['request_ids'] and replay['answer_origin']=='dynamic_reused'
    assert not replay_budget.state['attempts'];r.verify(replay)
    test=PublicTask(task.task_id,task.physical_key,task.goal,task.inputs,'test')
    with pytest.raises(ValueError,match='exact response reuse is required'):
        r.solve(test,a,[c],{},cfg,tmp_path/'test',replay_budget,reuse_trace=tmp_path/'original/trace.json',require_reuse=True)
    changed=deepcopy(cfg);changed['effect_p7']['runtime']['finish_completion']=2048
    with pytest.raises(ValueError,match='exact response reuse is required'):
        r.solve(task,a,[c],{},changed,tmp_path/'changed',replay_budget,reuse_trace=tmp_path/'original/trace.json',require_reuse=True)
    assert len(calls)==1;a.close()


def test_different_skill_card_never_reuses_old_response(tmp_path,monkeypatch):
    task=tasks('val')[0];a=adapter([task]);cfg=config();c=candidate(False)
    calls=http(monkeypatch,[response(content='<answer>B</answer>')])
    g=BudgetGovernor(tmp_path/'old_budget.json',token_limit=150000,request_limit=2,finish_reserve=0)
    r.solve(task,a,[c],{},cfg,tmp_path/'original',g)
    v=proposal(False);v['steps']=['A changed reasoning intervention.'];new=p.candidate(v,['s0','s1'],{'response_hash':'fixture2'})
    g2=BudgetGovernor(tmp_path/'budget.json',token_limit=0,request_limit=0,finish_reserve=0)
    with pytest.raises(ValueError,match='exact response reuse is required'):
        r.solve(task,a,[new],{},cfg,tmp_path/'changed',g2,reuse_trace=tmp_path/'original/trace.json',require_reuse=True)
    assert len(calls)==1;a.close()


def test_real_program_overrides_reused_wrong_response(qualified,tmp_path,monkeypatch):
    c,q=qualified;task=tasks('val')[0];a=adapter([task]);cfg=config()
    calls=http(monkeypatch,[response(content='<answer>A</answer>')])
    g=BudgetGovernor(tmp_path/'old_budget.json',token_limit=150000,request_limit=2,finish_reserve=0)
    old=r.solve(task,a,[],{},cfg,tmp_path/'original',g)
    g2=BudgetGovernor(tmp_path/'budget.json',token_limit=0,request_limit=0,finish_reserve=0)
    trace=r.solve(task,a,[c],{c['id']:q},cfg,tmp_path/'program',g2,
        reuse_trace=tmp_path/'original/trace.json',require_reuse=True)
    assert not old['score']['hard'] and trace['score']['hard'] and trace['answer_origin']=='learned_program'
    assert trace['http']==0 and len(calls)==1 and not g2.state['attempts'];r.verify(trace);a.close()


def test_candidate_requires_original_response_and_metered_charge(tmp_path):
    from atomic_skillgraph.experiments.run_math_predicate_evolution import evidence_files
    v=proposal();turn={'tool_calls':[{'arguments':{'candidates':[v]}}]}
    records=[{'request_id':'physical1','raw_usage':{'total_tokens':5}}]
    ev={'turn':turn,'response_hash':digest(turn),'request_ids':['physical1'],'request_hash':digest(records)}
    folder=tmp_path/'learning/0/request'
    write_json(folder/'response.json',ev);write_json(folder/'requests.json',records);write_json(folder/'request.json',{})
    c=p.candidate(v,['s0','s1'],ev)
    ledger={'attempts':{'physical1':{'status':'metered','actual_usage':records[0]['raw_usage']}}}
    assert len(evidence_files(tmp_path,[c],ledger))==3
    ledger['attempts']['physical1']['actual_usage']={'total_tokens':6}
    with pytest.raises(ValueError,match='metered charge'):evidence_files(tmp_path,[c],ledger)


def test_derived_train_val_freeze_flow_and_idempotence(qualified,tmp_path,monkeypatch):
    from atomic_skillgraph.experiments import run_math_predicate_evolution as runner
    from atomic_skillgraph.experiments import run_math_effect_r3 as original
    c,q=qualified;root=tmp_path/'derived';cfg=config();cfg['effect_p7']['derivation']={'version':e.VERSION}
    sets={'train':tasks('train',12),'val':tasks('val',17)};baselines={};paths={}
    calls=http(monkeypatch,[response(content='<answer>A</answer>') for _ in range(29)])
    g=BudgetGovernor(tmp_path/'historical_budget.json',token_limit=5000000,request_limit=100,finish_reserve=0)
    for split,ts in sets.items():
        phase='feedback' if split=='train' else 'val';baselines[phase]={};a=adapter(ts)
        for task in ts:
            path=tmp_path/'historical'/split/task.physical_key
            r.solve(task,a,[],{},cfg,path,g)
            baselines[phase][task.physical_key]={'path':str(path/'trace.json')}
            if split=='train':paths[task.physical_key]=str(path/'trace.json')
        a.close()
        write_json(tmp_path/'data/livemath'/('evaluator_records_'+split+'.json'),
            {t.task_id:{'correct_choice':t.inputs['choices'][1],'choices':t.inputs['choices']} for t in ts})
    plan={'method_version':e.VERSION,'datasets':str(tmp_path/'data'),'learning_source_keys':['s'+str(i) for i in range(17)],
        'settings':{'cumulative_limits':g.state['limits'],'phase_limits':{'feedback':{'token_limit':0,'request_limit':0},
            'val':{'token_limit':100000,'request_limit':2}}}}
    for name,value in [('config.json',cfg),('tasks.json',{k:[t.__dict__ for t in ts] for k,ts in sets.items()}),
        ('baseline.json',baselines),('budget/prefix.json',g.state),('budget/ledger.json',g.state),
        ('neighborhood_candidates.json',[c]),('text_policies.json',[{'policy':[],'source_run':None,'train_traces':paths}]),
        ('paid_authorization.json',{'plan_hash':digest(plan),'human_instruction':'intercepted offline fixture',
            'phases':{'val':plan['settings']['phase_limits']['val']}})]:write_json(root/name,value)
    monkeypatch.setattr(original,'checked',lambda root_:plan)
    monkeypatch.setattr(original,'qualify',lambda *args:{c['id']:q})
    result=runner.execute(root)
    assert result['status']=='val_passed' and result['frozen']['method_version']==e.VERSION
    assert result['val_receipt']['groups']['all']['program_answers']==17
    assert result['train_receipt']['groups']['all']['program_answers']==12
    assert len(calls)==29 and runner.execute(root)==result and len(calls)==29
    assert json.loads((root/'budget/ledger.json').read_text())['attempts']==g.state['attempts']
