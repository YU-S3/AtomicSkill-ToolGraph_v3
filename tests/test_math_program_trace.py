"""Paid calls are intercepted; candidate Worker and Val consumption tests use real Docker."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from atomic_skillgraph.empirical import math_program_induction as m
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.math_program_trial import TrialView, verified_bundles
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments import run_math_program_p5 as pilot
from atomic_skillgraph.experiments import run_math_program_trace as runner
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http, response
from test_math_program_p5 import TRAIN, P4, ROOT, FIXTURE, proposal
from test_public_process import task

SETTINGS=yaml.safe_load((ROOT/'configs/livemath_p5_trace_seed42.yaml').read_text())


def test_full_train_schedule_is_deterministic_and_distinct():
    if not TRAIN.exists():pytest.skip('Requires real historical Train60')
    rows,atoms,*_=m.load_sources(TRAIN,P4)
    opportunities,_=m.catalog(rows,atoms,42,120)
    selected,coverage=runner.schedule(rows,opportunities,SETTINGS)
    assert len(coverage)==60 and len(selected)>60
    assert [r['physical_key'] for r in coverage]==[r['task']['physical_key'] for r in rows]
    assert max(len(r['opportunity_ids']) for r in coverage)==2
    assert len({o['opportunity_id'] for o in selected})==len(selected)
    assert runner.schedule(rows,list(reversed(opportunities)),SETTINGS)==(selected,coverage)
    assert all(not o['observed_execution'] for o in selected)
    assert not any('correct_choice' in json.dumps(m.material(o,rows,atoms)) for o in selected)


def test_parallel_ceiling_and_physical_charge_union(tmp_path):
    concurrent=tmp_path/'manual';root=tmp_path/'natural'
    historical={'limits':{'token_limit':45000000,'request_limit':9165,'finish_reserve':0},
        'attempts':{'old':{'reserved_tokens':100,'accounted_tokens':80}}}
    settings={**deepcopy(SETTINGS),'concurrent_runs':[{'path':str(concurrent),'token_limit':13000000,'request_limit':5000}]}
    pilot.write_json(concurrent/'plan.json',{'test_limits':{'token_limit':13000000,'request_limit':5000}})
    pilot.write_json(concurrent/'campaign_budget_prefix.json',historical)
    current=deepcopy(historical);current['attempts']['live']={'reserved_tokens':200}
    pilot.write_json(concurrent/'budget.json',current)
    own=deepcopy(historical);own['attempts']['natural']={'reserved_tokens':90,'accounted_tokens':40}
    pilot.write_json(root/'budget.json',own);pilot.write_json(root/'formal_plan.json',{'settings':settings})
    assert runner.allocation(settings,historical)['maximum_joint_tokens']==20176080
    snapshot=runner.campaign_snapshot(root)
    assert snapshot['http']==3 and snapshot['tokens_including_reservations']==320 and snapshot['known_metered_tokens']==120
    current['attempts']['old']['accounted_tokens']=81;pilot.write_json(concurrent/'budget.json',current)
    with pytest.raises(ValueError,match='Conflicting'):runner.campaign_snapshot(root)
    with pytest.raises(ValueError,match='historical charge changed'):runner.allocation(settings,historical)


def test_trial_contract_isolated_from_other_benchmarks(tmp_path):
    if not P4.exists():pytest.skip('Requires saved runtime contract')
    cfg=runner.val_config(pilot.read(P4/'plan.json')['configs']['val'],tmp_path)
    assert cfg['runtime']['choice_bounded_finalization']['enabled']
    assert cfg['experiment']['runtime_mode']=='frozen'
    assert cfg['llm']['purpose_overrides']['finish_only']['max_completion_tokens']==16384
    from atomic_skillgraph.empirical.system import validate_config
    for key,value in [('benchmark','officeqa'),('math_program_method_version','other'),('runtime_mode','online')]:
        changed=deepcopy(cfg);changed['experiment'][key]=value
        with pytest.raises(ValueError):validate_config(changed)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'):pytest.skip('Requires actual locked Docker')
    contracts=tmp_path/'contracts';contracts.mkdir()
    monkeypatch.setattr(pilot,'CONTRACTS',contracts)
    monkeypatch.setattr(runner,'CONTRACT',contracts/'livemath_p5_trace_seed42.yaml')
    identity={**runner.code_identity(),'tracked_dirty':False}
    monkeypatch.setattr(pilot,'code_identity',lambda:identity)
    monkeypatch.setattr(runner,'code_identity',lambda:identity)
    old=ROOT/'configs/livemath_p5_program_probe_seed42.yaml'
    (contracts/old.name).write_text(old.read_text())
    reval=tmp_path/'revalidation';pilot.prepare(TRAIN,reval,contracts/old.name)
    replies=[];chosen=False
    for op in pilot.read(reval/'selected_opportunities.json'):
        if not chosen and op['requested_local_operation']=='finite_quantifiers':
            replies.extend([response([proposal(op)],name='submit_math_program_spec'),response([{'source':FIXTURE}],name='submit_program')])
            chosen=True
        else:replies.append(response([{'decision':'no_change'}],name='submit_math_program_spec'))
    http(monkeypatch,replies);pilot.learn(reval,approved=True);review=pilot.probe(reval)
    assert review['probe_passed']==1 and len(verified_bundles(reval))==1
    settings={**deepcopy(SETTINGS),'revalidation':str(reval),'ledger_source':str(reval/'budget.json')}
    runner.CONTRACT.write_text(yaml.safe_dump(settings))
    root=tmp_path/'formal'
    result=runner.prepare(TRAIN,root,runner.CONTRACT)
    assert result['status']=='passed' and result['covered_train_tasks']==60 and result['val_tasks']==17
    return root


def learn_one(root,monkeypatch):
    replies=[];chosen=False
    for op in pilot.read(root/'selected_opportunities.json'):
        if not chosen and op['requested_local_operation']=='finite_quantifiers':
            replies.extend([response([proposal(op)],name='submit_math_program_spec'),response([{'source':FIXTURE}],name='submit_program')])
            chosen=True
        else:replies.append(response([{'decision':'no_change'}],name='submit_math_program_spec'))
    calls=http(monkeypatch,replies);pilot.learn(root,approved=True);pilot.probe(root)
    assert len(verified_bundles(root))==1
    return calls


def test_full_trace_to_val_and_resume(prepared,monkeypatch,tmp_path):
    root=prepared;learned=learn_one(root,monkeypatch)
    calls=http(monkeypatch,[response(finish='stop',content='<answer>A</answer>') for _ in range(17)])
    monkeypatch.setattr(runner,'load_env',lambda _:None)
    original=runner.run;crashed=[]
    def after_val(*a,**kw):
        summary=original(*a,**kw)
        if not crashed:
            crashed.append(True);raise RuntimeError('fixture interruption after Val before final receipt')
        return summary
    monkeypatch.setattr(runner,'run',after_val)
    with pytest.raises(RuntimeError,match='fixture interruption'):runner.execute(root,tmp_path/'unused')
    result=runner.execute(root,tmp_path/'unused')
    assert result['status']=='completed' and result['tasks']==17 and len(calls)==17
    assert result['trial_programs']==1 and result['usable_programs']==0 and not result['test_executed']
    assert result['learning_http']==len(learned) and result['learning_tokens']==len(learned)*5
    assert runner.execute(root,tmp_path/'unused')==result and len(calls)==17
    audit=pilot.read(root/'freeze_audit.json');assert audit['before']==audit['after']
    plan=pilot.read(root/'formal_plan.json')
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    for t,wire in zip(plan['tasks'],calls):
        material=json.loads(wire['messages'][1]['content'])
        assert material['goal']==t['goal'] and material['inputs']==t['inputs']
        assert input_token_bound(wire)+wire['max_tokens']<=128000
        assert wire['max_tokens']==65536 and wire['thinking']=={'type':'enabled'} and wire['reasoning_effort']=='high'
        assert len(material['programs'])==1 and 'source_input' not in json.dumps(material['programs'])
    budget=pilot.read(root/'budget.json');prefix=pilot.read(root/'campaign_budget_prefix.json')
    assert all(budget['attempts'][k]==v for k,v in prefix['attempts'].items())
    assert len(budget['attempts'])-len(prefix['attempts'])==len(learned)+17
    assert pilot.report(root)['new_http']==len(learned)


def test_actual_candidate_consumption_and_source_tamper(prepared,monkeypatch):
    root=prepared;learn_one(root,monkeypatch);frozen=runner.freeze(root)
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        p=bank.all('program')[0]
        assert p['state']=='candidate' and not bank.program_eligible(p)
        assert TrialView(bank,root/'trial_manifest.json').program_eligible(p)
    finally:bank.close()
    finish=response([{'action':'finish','answer':'<answer>A</answer>'}],name='answer_step')
    calls=http(monkeypatch,[response([{'action':'call_program','name':p['id'],
        'arguments':{'relation':[[True,False],[False,True]]}}],name='answer_step'),finish])
    from atomic_skillgraph.agents import provider
    post=provider.requests.post
    def consume(*a,**kw):
        material=json.loads(kw['json']['messages'][1]['content'])
        if material.get('local_results'):
            finish['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps({
                'action':'finish','answer':'<answer>A</answer>','used_results':[material['local_results'][0]['result_id']]})
        return post(*a,**kw)
    monkeypatch.setattr(provider.requests,'post',consume)
    t=task('fixture',split='val');cfg=pilot.read(root/'val_config.json')
    s=EmpiricalSystem(cfg,readonly=True,harness=AnswerAdapter('livemath',{'fixture':{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
    s.checkpoint=TaskCheckpoint(root/'fixture_checkpoint')
    try:trace=s.run_task(t,learn=False)
    finally:s.close()
    assert trace['score']['hard'] and len(calls)==2
    attempt=trace['execution']['attempts'][0]
    assert attempt['outputs_consumed'] and s.worker.invocations[0]['execution_receipt']['exit_code']==0
    assert tree_identity(root/'frozen_bank')['sha256']==frozen['tree_hash']
    row_path=next(p for p in (root/'learning').glob('*/learning.json') if 'bundle' in pilot.read(p))
    row=pilot.read(row_path);row['bundle']['program']['source']+='\n# altered'
    pilot.write_json(row_path,row)
    with pytest.raises(ValueError,match='Program or Implementation changed'):verified_bundles(root)


def test_trial_finalization_preserves_parent_contract(prepared,monkeypatch):
    root=prepared;runner.freeze(root)
    calls=http(monkeypatch,[response(finish='length'),response(finish='stop',content='<answer>A</answer>')])
    t=task('fixture',split='val');cfg=pilot.read(root/'val_config.json')
    s=EmpiricalSystem(cfg,readonly=True,harness=AnswerAdapter('livemath',{'fixture':{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
    s.checkpoint=TaskCheckpoint(root/'fixture_checkpoint')
    try:trace=s.run_task(t,learn=False)
    finally:s.close()
    final=trace['execution']['choice_bounded_finalization']
    assert trace['score']['hard'] and not trace['execution'].get('budget_censored')
    assert final['attempted'] and final['valid_answer'] and not final['parent_budget_exhausted']
    assert calls[0]['max_tokens']==65536 and calls[1]['max_tokens']==16384 and calls[1]['reasoning_effort']=='low'
    material=json.loads(calls[1]['messages'][1]['content'])
    assert material['goal']==t.goal and material['inputs']==t.inputs


def test_no_program_is_truthfully_evaluated_and_no_promotion(prepared,monkeypatch,tmp_path):
    root=prepared;count=len(pilot.read(root/'selected_opportunities.json'))
    calls=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_math_program_spec') for _ in range(count)]+
        [response(finish='stop',content='<answer>A</answer>') for _ in range(17)])
    monkeypatch.setattr(runner,'load_env',lambda _:None)
    result=runner.execute(root,tmp_path/'unused')
    assert result['status']=='completed' and result['trial_programs']==result['usable_programs']==0
    assert len(calls)==count+17 and result['program_invocations']==0


def test_learning_budget_stop_never_enters_val(prepared,monkeypatch,tmp_path):
    from atomic_skillgraph.core.errors import BudgetExhausted
    from atomic_skillgraph.experiments.formal_choice_batch import BatchGovernor
    def exhausted(*a,**kw):raise BudgetExhausted('fixture_limit','No stage budget remaining')
    monkeypatch.setattr(BatchGovernor,'admit',exhausted)
    monkeypatch.setattr(runner,'load_env',lambda _:None)
    calls=http(monkeypatch,[])
    result=runner.execute(prepared,tmp_path/'unused')
    assert result['status']=='stopped' and result['stage']=='trace_learning'
    assert not calls and not (prepared/'val').exists() and not (prepared/'frozen_bank').exists()
