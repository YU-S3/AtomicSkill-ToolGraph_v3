"""P5 uses real source and production HTTP/Worker interfaces; fixtures are not learned assets."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from atomic_skillgraph.empirical import math_program_induction as m
from atomic_skillgraph.empirical import math_program_probe as probe
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import digest, program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import run_math_program_p5 as runner
from atomic_skillgraph.experiments.formal_choice_batch import BatchGovernor
from atomic_skillgraph.experiments.canonical_manifest import sha256
from test_cf2_contracts import http, response
from test_empirical import config_for

ROOT=Path(__file__).resolve().parents[1]
TRAIN=Path('/home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42')
P4=Path('/home/yangchengyu/livemath_p4_seed42_20261014')
SETTINGS=yaml.safe_load((ROOT/'configs/livemath_p5_program_probe_seed42.yaml').read_text())
# Explicit artificial test fixture; never supplied to the real Learner/Builder.
FIXTURE='''def run(ctx, inputs):
    rows = inputs.get('relation')
    if not rows or not all(isinstance(r,list) and r and len(r)==len(rows[0]) and all(type(v)==bool for v in r) for r in rows):
        return {'status':'needs_input','outputs':{}}
    witnesses = [j for j in range(len(rows[0])) if all(row[j] for row in rows)]
    return {'status':'ok','outputs':{'tested_cases':sum(map(len,rows)), 'forall_exists':all(any(r) for r in rows),
        'exists_forall':bool(witnesses),'common_witness_columns':witnesses,'limitation':'finite_explicit_input_only_not_general_proof'}}
'''


def op():
    return {'opportunity_id':'op_fixture','requested_local_operation':'finite_quantifiers',
        'source_case_ids':['physical_fixture'],'public_source_refs':[{'case_id':'physical_fixture','span_id':'span_fixture'}],
        'parent_outcome':'wrong','origin':'problem_derived_hypothesis','observed_execution':False,'atomic_source_ids':[]}


def proposal(o=None):
    o=o or op();io=probe.interface('finite_quantifiers')
    return {'decision':'propose_program','source_opportunity_ids':[o['opportunity_id']],
        'atomic':{'goal':'Compare explicit finite quantifiers','applicability':'Explicit Boolean relation supplied',
            'prerequisites':['The domain and relation are finite and complete'],'operation':'Inspect row and column witnesses',
            'outputs':['forall_exists','exists_forall'],'stop_condition':'Missing or infinite domain: needs_input','result_role':'intermediate'},
        'program_spec':{**io,'allowed_tools':[],'finite_scope':'Explicit input matrix only','negative_cases':['Missing relation'],
            'probe_kind':'finite_quantifiers','source_input':{'relation':[[True,False],[False,True]]},
            'input_derivation':'Artificial finite test instantiation, not a source theorem result'},
        'limitations':'Finite checks do not prove universal theorems'}


def cfg(tmp):
    c=config_for(tmp/'old');c['harness']['adapter']='livemath';c['experiment']['benchmark']='livemath'
    c['llm']['model']='deepseek-v4-flash';c['program_environment'].update(adapter_abi='simple.v2',image_digest=m.IMAGE)
    return runner.configure(c,tmp,SETTINGS)


def test_real_source_allowlist_and_feedback(monkeypatch):
    if not TRAIN.exists():pytest.skip('Real local Train60 is required')
    original=Path.open
    opened=[]
    def guarded(path,*a,**kw):
        if str(path).startswith(str(P4)) or str(path).startswith(str(TRAIN)):
            assert not set(path.parts)&{'val','test','consolidation','candidates'}
            opened.append(str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    rows,atoms,hashes,c,manifest=m.load_sources(TRAIN,P4)
    assert len(rows)==60 and len(atoms)==15 and sum(r['parent_outcome']=='correct' for r in rows)==13
    assert sum(a['category']=='finite_computation' for a in atoms)==2
    assert all(r['observed_execution'] is False for r in rows)
    assert 'correct_choice' not in json.dumps(rows)
    all_ops,chosen=m.catalog(rows,atoms)
    assert len(chosen)==6 and {o['parent_outcome'] for o in chosen}>={'correct','wrong'}
    assert all(o['origin']=='problem_derived_hypothesis' for o in chosen if o['parent_outcome']!='correct')
    assert all(m.material(o,rows,atoms)['train_atomic_hypotheses']==atoms for o in chosen)
    assert m.catalog(rows,atoms)==(all_ops,chosen)
    assert all(sha256(Path(p))==h for p,h in hashes.items())


def test_spec_source_schema_and_abstention():
    v=proposal();assert m.validate_spec(v,op())==v
    for mutation in [{'source_opportunity_ids':['invented']}, {'atomic':{**v['atomic'],'result_role':'final_answer'}},
        {'program_spec':{**v['program_spec'],'allowed_tools':['grader']}}]:
        with pytest.raises(ValueError):m.validate_spec({**v,**mutation},op())
    assert m.validate_spec({'decision':'no_change','unrelated':'ignored'},op())=={'decision':'no_change'}
    assert m.validate_spec({'decision':'guidance_only'},op())=={'decision':'guidance_only'}


@pytest.mark.parametrize('source',[
    "def run(ctx, inputs):\n return {'status':'ok','outputs':{'answer':'B'}}",
    "def run(ctx, inputs):\n return open('/hidden/gold').read()",
    "import requests\ndef run(ctx, inputs):\n return {}",
    "def run(ctx, inputs):\n return ctx.call('evaluate', {})"])
def test_wrong_answer_and_authority_rejected(tmp_path,source):
    v=proposal()
    with pytest.raises(ValueError):m.program(source,v['program_spec'],m.atomic(v,op()),cfg(tmp_path)['program_environment'])


def test_generated_syntax_is_a_model_rejection(tmp_path):
    v=proposal()
    with pytest.raises(ValueError,match='syntax'):m.program('def run(ctx, inputs)',v['program_spec'],m.atomic(v,op()),cfg(tmp_path)['program_environment'])


def test_oracles_variations_and_incomplete_inputs():
    for kind,source in [
        ('finite_quantifiers',{'relation':[[True,False],[False,True]]}),
        ('rational_compare',{'left':[[1,3],[-1,2]],'right':[[2,6],[1,-2]]}),
        ('finite_predicates',{'x_values':[1,2,3,4],'y_values':[1,2],'left':['eq',['mod','x',2],0],
            'right':['eq',['mod','x',4],0]})]:
        spec={'probe_kind':kind,'source_input':source};samples=probe.probe_inputs(spec)
        assert len(samples)==3 and samples[0][1]!=samples[1][1]
        assert probe.oracle(kind,source)['limitation']==probe.LIMITATION
        with pytest.raises(ValueError):probe.oracle(kind,{})
    with pytest.raises(ValueError):probe.oracle('finite_quantifiers',{'relation':[[True],[False,True]]})
    with pytest.raises(ValueError):probe.oracle('rational_compare',{'left':[[1,0]],'right':[[1,1]]})
    with pytest.raises(ValueError):probe.expression(['eq',True,1],1,1)


def test_real_provider_and_paid_response_recovery(tmp_path,monkeypatch):
    c=cfg(tmp_path);s=EmpiricalSystem(c,harness=AnswerAdapter('livemath',{}))
    plan={'code':{'git_sha':'fixture'},'policy_sha':'fixture','run_id':'p5_fixture'};o=op();v=proposal()
    seen=http(monkeypatch,[response([v],name='submit_math_program_spec'),response([{'source':FIXTURE}],name='submit_program')])
    material=m.material(o,[],[])
    a=runner.receive(s,plan,o,tmp_path/'unit','extractor',m.PROMPT,material,'submit_math_program_spec',m.PROPOSAL,
        'math_program_spec',lambda x:m.validate_spec(x,o));assert a==v
    b=runner.build(s,plan,o,v,tmp_path);assert b['state']=='code_generated'
    assert not s.bank.program_eligible(b['program']);s.close()
    s=EmpiricalSystem(c,harness=AnswerAdapter('livemath',{}))
    a=runner.receive(s,plan,o,tmp_path/'unit','extractor',m.PROMPT,material,'submit_math_program_spec',m.PROPOSAL,
        'math_program_spec',lambda x:m.validate_spec(x,o))
    b2=runner.build(s,plan,o,v,tmp_path);assert b2==b and len(seen)==2
    assert all(p['thinking']['type']=='disabled' for p in seen)
    assert [p['max_tokens'] for p in seen]==[4096,8192]
    assert all(x['state']=='candidate' for x in s.bank.all('program'))
    s.close()


def test_model_invalid_cached_no_extra_request(tmp_path,monkeypatch):
    s=EmpiricalSystem(cfg(tmp_path),harness=AnswerAdapter('livemath',{}));plan={'code':{},'policy_sha':'x','run_id':'p5_x'}
    seen=http(monkeypatch,[response([{'decision':'propose_program','source_opportunity_ids':['invented']}],name='submit_math_program_spec')])
    for _ in range(2):
        with pytest.raises(ValueError):runner.receive(s,plan,op(),tmp_path/'unit','extractor',m.PROMPT,{},
            'submit_math_program_spec',m.PROPOSAL,'math_program_spec',lambda x:m.validate_spec(x,op()))
    assert len(seen)==1;s.close()


def test_phase_cap_and_unknown_billing(tmp_path):
    from atomic_skillgraph.core.errors import BudgetExhausted
    owner=BudgetGovernor(tmp_path/'budget.json',token_limit=10000,finish_reserve=0,request_limit=20)
    runner.write_json(tmp_path/'learning/campaign_budget_prefix.json',deepcopy(owner.state))
    g=BatchGovernor(owner,{'run_id':'p5','new_limits':{'token_limit':100,'request_limit':2}},tmp_path/'learning')
    with pytest.raises(BudgetExhausted):g.admit('large',{'max_tokens':101},{'parent_scope':'p5:a'})
    g.admit('first',{'max_tokens':10},{'parent_scope':'p5:a'})
    g.complete({'request_id':'first','raw_usage':None,'outcome':'error'})
    assert g.headroom({})['unknown_billing']
    with pytest.raises(BudgetExhausted):g.admit('second',{'max_tokens':1},{'parent_scope':'p5:a'})


def test_p5_identity_does_not_affect_other_methods(tmp_path):
    c=cfg(tmp_path);validate_config(c)
    for k,v in [('benchmark','officeqa'),('math_program_method_version','made_up')]:
        changed=deepcopy(c);changed['experiment'][k]=v
        with pytest.raises(ValueError):validate_config(changed)
    c['experiment'].pop('math_program_method_version')
    with pytest.raises(ValueError):validate_config(c)


def test_actual_docker_bundle_and_no_promotion(tmp_path):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'):pytest.skip('Needs locked Linux Docker')
    assert os.environ['PROGRAM_IMAGE_DIGEST']==m.IMAGE
    v=proposal();c=cfg(tmp_path);bank=Bank(tmp_path/'bank',environment=c['program_environment'])
    a=bank.put('skill',m.atomic(v,op()));p=bank.put('program',m.program(FIXTURE,v['program_spec'],a,c['program_environment']))
    worker=ProgramWorker({'wall_timeout_seconds':10,'record_execution_receipt':True})
    result=probe.probe(worker,p,tmp_path/'receipts')
    assert result['state']=='train_trial_only' and result['probe_passed']
    assert len(result['receipts'])==3 and all(r['rpc_calls']==[] for r in result['receipts'])
    assert all(r['docker_exit_code']==0 for r in result['receipts'])
    assert result['receipts'][2]['result']['status']=='needs_input'
    assert not bank.program_eligible(p)
    from atomic_skillgraph.empirical.math_capability import checked_asset
    assert not checked_asset(a)
    assert probe.probe(worker,p,tmp_path/'receipts')==result and len(worker.invocations)==3
    bank.close()


def test_worker_unavailable_has_no_host_fallback(tmp_path,monkeypatch):
    v=proposal();c=cfg(tmp_path);p=m.program(FIXTURE,v['program_spec'],m.atomic(v,op()),c['program_environment'])
    p['id']='program_'+program_digest(p)
    monkeypatch.setattr('atomic_skillgraph.empirical.program_worker.shutil.which',lambda _:None)
    result=probe.probe(ProgramWorker({'record_execution_receipt':True}),p,tmp_path/'blocked')
    assert result['state']=='blocked' and len(result['receipts'])==1
    assert not result['probe_passed']


def prepared(tmp_path,monkeypatch):
    if not TRAIN.exists():pytest.skip('Real Train60 unavailable')
    monkeypatch.setattr(runner,'code_identity',lambda:{'git_sha':'offline-fixture','source_sha256':'offline-fixture','tracked_dirty':False})
    root=tmp_path/'prepared'
    result=runner.prepare(TRAIN,root,ROOT/'configs/livemath_p5_program_probe_seed42.yaml')
    assert result['status']=='passed' and result['new_model_http']==0
    return root


def test_full_no_candidate_flow_and_budget_inheritance(tmp_path,monkeypatch):
    root=prepared(tmp_path,monkeypatch);before=sha256(P4/'frozen_bank/bank.sqlite3')
    with pytest.raises(ValueError,match='not authorized'):runner.learn(root)
    seen=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_math_program_spec') for _ in range(6)])
    result=runner.learn(root,approved=True)
    assert result['status']=='completed_no_program' and result['new_http']==6 and result['new_tokens']==30
    assert runner.learn(root)['new_http']==6 and len(seen)==6
    assert runner.probe(root)['probe_passed']==0 and sha256(P4/'frozen_bank/bank.sqlite3')==before
    prefix=runner.read(root/'campaign_budget_prefix.json');ledger=runner.read(root/'budget.json')
    assert all(ledger['attempts'][k]==v for k,v in prefix['attempts'].items())
    assert ledger['inherited_reservation_recovery']==prefix['inherited_reservation_recovery']
    assert not (root/'val').exists() and not (root/'test').exists()


def test_candidate_local_rejection_and_single_builder_repair(tmp_path,monkeypatch):
    root=prepared(tmp_path,monkeypatch);ops=runner.read(root/'selected_opportunities.json')
    responses=[];expected=[]
    for i,o in enumerate(ops):
        if o['requested_local_operation']=='finite_quantifiers':
            v=proposal(o);responses.append(response([v],name='submit_math_program_spec'))
            if o['parent_outcome']=='correct':
                responses.append(response([{'source':FIXTURE}],name='submit_program'));expected.append('code_generated')
            else:
                for _ in range(2):responses.append(response([{'source':"def run(ctx, inputs):\n return {'status':'ok','outputs':{'answer':'B'}}"}],name='submit_program'))
                expected.append('model_invalid')
        elif i==0:
            responses.append(response([{'decision':'propose_program','source_opportunity_ids':['fake']}],name='submit_math_program_spec'))
            expected.append('model_invalid')
        else:
            responses.append(response([{'decision':'no_change'}],name='submit_math_program_spec'));expected.append('no_change')
    seen=http(monkeypatch,responses);result=runner.learn(root,approved=True)
    assert [r['state'] for r in result['rows']]==expected
    assert result['generated_programs']==1 and len(seen)==9
    runner.learn(root);assert len(seen)==9
    b=Bank(root/'bank',readonly=True)
    assert len(b.all('program'))==1 and not any(b.program_eligible(p) for p in b.all('program'));b.close()


def test_response_received_crash_reuses_physical_builder(tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
    c=cfg(tmp_path);plan={'code':{},'policy_sha':'x','run_id':'p5_crash'};o=op();v=proposal()
    seen=http(monkeypatch,[response([{'source':FIXTURE}],name='submit_program')])
    original=TaskCheckpoint.commit_decision;crashed=[]
    def interrupted(self,key,status,**kw):
        if status=='applied' and not crashed:
            crashed.append(True);raise RuntimeError('simulated exit after paid response')
        return original(self,key,status,**kw)
    monkeypatch.setattr(TaskCheckpoint,'commit_decision',interrupted)
    s=EmpiricalSystem(c,harness=AnswerAdapter('livemath',{}))
    with pytest.raises(RuntimeError,match='simulated exit'):runner.build(s,plan,o,v,tmp_path)
    s.close();monkeypatch.setattr(TaskCheckpoint,'commit_decision',original)
    s=EmpiricalSystem(c,harness=AnswerAdapter('livemath',{}))
    assert runner.build(s,plan,o,v,tmp_path)['state']=='code_generated' and len(seen)==1
    s.close()
