"""Full public-source scheduling, finite phase ownership and derived evaluation."""
from copy import deepcopy
from pathlib import Path
import sqlite3

import pytest
import yaml

from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.experiments import capability_schedule as scheduling
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.experiments import run_derived_choice as derived
from atomic_skillgraph.experiments.canonical_manifest import sha256
from atomic_skillgraph.experiments.formal_choice_batch import BatchGovernor
from skillcompiler_bench_contracts.livemath import NORMALIZATION_VERSION,CHOICE_PROJECTION_VERSION,CHOICE_SEED
from test_capability_batch import fixture_source,fixture_transport

ROOT=Path(__file__).parents[1]


def prepare_fixture(tmp_path,monkeypatch,*,unresolved=False,full=False,recovery=False):
    source,_,_,env=fixture_source(tmp_path,monkeypatch)
    identity={'git_sha':'intercepted-fixture','tracked_dirty':False,'source_sha256':'simulated-fixture-source'}
    monkeypatch.setattr(derived,'code_identity',lambda:identity)
    cfg=batch.read(source/'train/config.json');cfg['llm']['max_retries']=4
    cfg['experiment']['choice_method_version']='choice-capability-batch.v2'
    with sqlite3.connect(source/'train/bank/bank.sqlite3') as db:
        db.execute("INSERT OR REPLACE INTO metadata VALUES('choice_method_version','choice-capability-batch.v2')")
    bank=Bank(source/'train/bank',readonly=True)
    try:bank_hash=bank.digest()
    finally:bank.close()
    summary=batch.read(source/'train/summary.json');summary['knowledge_digest']=bank_hash
    batch.write_json(source/'train/summary.json',summary)
    batch.write_json(source/'train/config.json',cfg)
    execution=batch.read(source/'train/execution_manifest.json');execution['config']=cfg
    batch.write_json(source/'train/execution_manifest.json',execution)
    spec=yaml.safe_load((ROOT/('configs/livemath_p2_plan_b_seed42.yaml' if full else 'configs/livemath_p2_schedule_seed42.yaml')).read_text())
    spec['source']['historical_run_commit']=batch.read(source/'run_manifest.json')['source_commit']
    settings=tmp_path/'new-contract.yaml';settings.write_text(yaml.safe_dump(spec))
    data=tmp_path/'data';files={}
    for split in ('val','test'):
        rows=deepcopy(execution['tasks'][:17] if split=='val' else execution['tasks'])
        ids=batch.read(ROOT/'data/main_experiment_v1/livemath'/(split+'.json'))['tasks']
        rows=[{**deepcopy(rows[i%len(rows)]),'task_id':r['task_id'],'physical_key':digest(r['task_id']),'split':split} for i,r in enumerate(ids)]
        records={row['task_id']:{'choices':row['inputs']['choices'],'correct_choice':row['inputs']['choices'][0]} for row in rows}
        for name,value in [('livemath/'+split+'.json',{'tasks':rows}),('livemath/evaluator_records_'+split+'.json',records)]:
            batch.write_json(data/name,value);files[name]=sha256(data/name)
    batch.write_json(data/'materialization.json',{'authority_sha256':sha256(ROOT/'data/main_experiment_v1/manifest.json'),
        'files_sha256':files,'livemath_normalization_version':NORMALIZATION_VERSION,
        'livemath_choice_projection_version':CHOICE_PROJECTION_VERSION,'livemath_choice_seed':CHOICE_SEED})
    if unresolved:
        governor=BudgetGovernor(source/'budget.json',token_limit=23232000,request_limit=9165,finish_reserve=0)
        governor.admit('unfinished',{'messages':[],'max_tokens':2048},{'parent_scope':'old-val'})
    recovery_path=None
    if recovery:
        ledger=batch.read(source/'budget.json')
        recovery_path=tmp_path/'reserved-recovery.json'
        batch.write_json(recovery_path,{'schema':'billing-reservation-recovery.v1','authorization':'Explicit test fixture authorization',
            'source_ledger_sha256':sha256(source/'budget.json'),
            'attempts':{key:{'entry_hash':digest(row),'reserved_tokens':row['reserved_tokens']} for key,row in ledger['attempts'].items() if row['status']!='metered'}})
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    root=tmp_path/'new-version'
    receipt=derived.prepare(settings,source,root,data,reservation_recovery=recovery_path)
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    return source,root,receipt,env,before


def test_complete_grouping_uses_public_demands_and_is_immutable(tmp_path,monkeypatch):
    source,root,receipt,_,_=prepare_fixture(tmp_path,monkeypatch)
    plan=batch.read(root/'p2/plan.json');group=plan['source_schedule'];keys=[k for unit in plan['candidate_units'] for k in unit]
    assert len(group['units'])==12 and len(keys)==len(set(keys))==48
    heldout={row['physical_key'] for values in plan['selection_tasks'].values() for row in values}
    assert not set(keys)&heldout
    changed=deepcopy(plan['generation_tasks'])
    for row in changed:row.update(score={'hard':True},gold='A',metadata={'test_score':1})
    assert scheduling.schedule(list(reversed(changed)),plan['settings'])==group
    for unit in group['units']:
        assert unit['algorithm_hash']==group['algorithm_hash'] and len(unit['public_relation'])==3
    plan['candidate_units'][0].reverse();batch.write_json(root/'p2/plan.json',plan)
    batch.write_json(root/'p2/plan_identity.json',{'plan_hash':digest(plan)})
    with pytest.raises(ValueError,match='schedule changed'):derived.check(root)


def test_all_twelve_normal_negative_units_complete_with_no_paid_evaluation(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch,no_candidate=True)
    source,root,receipt,env,before=prepare_fixture(tmp_path,monkeypatch)
    result=derived.execute(root,env)
    assert result['status']=='awaiting_p2_review_no_usable_assets'
    p2=result['p2'];assert p2['status']=='completed_no_candidate' and p2['attempted_units']==12
    assert len(p2['seen_sources'])==48 and not p2['unattempted_source_keys'] and not p2['cited_support_sources']
    assert len(calls)==12 and not (root/'val').exists() and not (root/'test').exists()
    derived.execute(root,env);assert len(calls)==12
    with pytest.raises(ValueError,match='review authorization'):derived.execute(root,env,stage='test')
    assert len(calls)==12 and before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_eighty_http_censors_full_two_candidate_selection_without_publication(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch)
    _,root,_,env,_=prepare_fixture(tmp_path,monkeypatch)
    result=derived.execute(root,env)
    assert result['status']=='stopped_budget_censored' and result['new_http']==80 and len(calls)==80
    order=batch.read(root/'p2/candidate_order.json')
    assert len(order)==12 and sum(row['status']=='selected_for_routing' for row in order)==2
    assert sum(row['status']=='checked_not_selected' for row in order)==10
    assert result['source_checked_candidate_count']==12 and result['selected_for_routing_count']==2
    assert result['attempted_units']==12 and not result['unattempted_source_keys']
    assert not (root/'train/frozen_bank').exists() and not (root/'val').exists()
    bank=Bank(root/'train/bank',readonly=True)
    try:assert not bank.all('skill')
    finally:bank.close()
    derived.execute(root,env);assert len(calls)==80


def test_unknown_source_fee_allows_only_offline_prepare(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch)
    source,root,receipt,env,before=prepare_fixture(tmp_path,monkeypatch,unresolved=True)
    assert receipt['status']=='passed' and not receipt['billing_ready']
    assert batch.read(root/'budget.json')['unknown_billing']
    with pytest.raises(BudgetExhausted,match='billing receipt'):derived.execute(root,env)
    assert not calls and before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_checked_top_two_val_readonly_and_separate_test_review(tmp_path,monkeypatch):
    spec=yaml.safe_load((ROOT/'configs/livemath_p2_selection_seed42.yaml').read_text())['split']
    excluded=set(spec['selection_screen_ids'][2:]+spec['selection_confirmation_ids'][2:])
    calls=fixture_transport(monkeypatch,eligibility=lambda body:not any(' '+key+']' in body['task']['goal'] for key in excluded))
    source,root,_,env,before=prepare_fixture(tmp_path,monkeypatch)
    result=derived.execute(root,env)
    assert result['status']=='awaiting_test_review' and result['val']['tasks']==17
    assert batch.read(root/'p2/result.json')['new_http']==76 and not (root/'test').exists()
    assert all(trace['injected_guidance_ids'] for trace in batch.read(root/'val/summary.json')['cases'])
    phase_before=(root/'p2/budget.json').read_bytes()
    n=len(calls)
    with pytest.raises(ValueError,match='review authorization'):derived.execute(root,env,stage='test')
    assert len(calls)==n
    result=derived.execute(root,env,stage='test',reviewed=True)
    assert result['test']['tasks']==100 and result['status']=='completed'
    audit=batch.read(root/'test_freeze_audit.json');assert audit['hash_before']==audit['hash_after']
    assert batch.read(root/'test_review.json')['test_development_history_disclosed']
    assert (root/'p2/budget.json').read_bytes()==phase_before
    n=len(calls);derived.execute(root,env,stage='test',reviewed=True);assert len(calls)==n
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind',['capacity','unit_budget'])
def test_unit_failure_advances_fixed_sources_and_reports_actual_seen(tmp_path,monkeypatch,kind):
    calls=fixture_transport(monkeypatch,no_candidate=True)
    _,root,_,env,_=prepare_fixture(tmp_path,monkeypatch)
    from atomic_skillgraph.empirical.learner import Learner
    original=Learner.learn_capability_batch;visited=[]
    def learn(self,unit,selection,**kwargs):
        visited.append(unit)
        if len(visited)==1:
            if kind=='unit_budget':raise BudgetExhausted('parent_task_budget_exhausted','Unit budget admission refused')
            exc=ValueError('Required four-source material exceeds 16 KiB')
            exc.code='material_too_large';exc.learning_outcome='learning_capacity_rejected';raise exc
        return original(self,unit,selection,**kwargs)
    monkeypatch.setattr(Learner,'learn_capability_batch',learn)
    result=derived.execute(root,env)
    p2=result['p2'] if 'p2' in result else result
    assert p2['attempted_units']==12 and len(visited)==12 and len(calls)==11
    assert len(p2['seen_sources'])==44 and not p2['unattempted_source_keys']
    assert set(visited[0]).isdisjoint(p2['seen_sources'])
    if kind=='unit_budget':
        assert p2['status']=='stopped_budget_censored' and p2['learning_budget_censored']
        assert not (root/'train/frozen_bank').exists()
    else:assert p2['status']=='completed_no_candidate'


def test_stage_prefix_and_unit_global_authority_never_reset(tmp_path):
    owner=BudgetGovernor(tmp_path/'campaign.json',token_limit=100000,request_limit=10,finish_reserve=0)
    payload={'messages':[],'max_tokens':100}
    owner.admit('old-p2',payload,{'parent_scope':'old-unit'})
    owner.complete({'request_id':'old-p2','outcome':'success','raw_usage':{'total_tokens':30}})
    batch.write_json(tmp_path/'campaign_budget_prefix.json',owner.state)
    batch.write_json(tmp_path/'inherited_stage_attempts.json',owner.state['attempts'])
    plan={'run_id':'new-version','new_limits':{'token_limit':1000,'request_limit':2}}
    stage=BatchGovernor(owner,plan,tmp_path)
    assert stage.headroom({})['requests']==1 and stage.headroom({})['tokens']==970
    context={'parent_scope':'new-version:unit','parent_token_limit':120,'train_parent':False,
             'episode_scope':'new-version:unit','episode_request_limit':1}
    with pytest.raises(BudgetExhausted):stage.admit('too-big',payload,context)
    small={'messages':[],'max_tokens':1};stage.admit('one',small,context)
    stage.complete({'request_id':'one','outcome':'success','raw_usage':{'total_tokens':20}})
    assert stage.headroom({})['requests']==0
    owner.admit('val',small,{'parent_scope':'readonly-val'})
    owner.complete({'request_id':'val','outcome':'success','raw_usage':{'total_tokens':10}})
    restored=BatchGovernor(BudgetGovernor(owner.path,token_limit=100000,request_limit=10,finish_reserve=0),plan,tmp_path)
    assert restored.headroom({})['requests']==0 and restored.headroom({})['tokens']==950
    with pytest.raises(BudgetExhausted):restored.admit('two',small,context)
