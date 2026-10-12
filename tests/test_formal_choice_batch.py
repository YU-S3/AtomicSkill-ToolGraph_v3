"""Fresh formal P2, physical billing and resume through intercepted production HTTP."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.capability_batch import checked_asset
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import formal_choice_batch as formal
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.experiments import run_empirical, run_formal
from atomic_skillgraph.experiments.canonical_manifest import ordered_train, sha256
from atomic_skillgraph.experiments.formal_log import tree_identity
from skillcompiler_bench_contracts.livemath import NORMALIZATION_VERSION, CHOICE_PROJECTION_VERSION, CHOICE_SEED
from test_capability_batch import fixture_transport, settings
from test_public_process import config, task

ROOT = Path(__file__).parents[1]
AUTHORITY = ROOT/'data/main_experiment_v1'


def fixture(tmp_path, monkeypatch):
    identity = {'git_sha':'intercepted-formal-fixture','tracked_dirty':False,'source_sha256':'fixture-source'}
    for module in (formal,batch,run_empirical,run_formal):monkeypatch.setattr(module,'code_identity',lambda:identity)
    base = config(tmp_path/'source')
    base['llm'].update(model='deepseek-v4-flash',generation_seed_supported=False,input_modalities=['text'],
        token_limit_field='max_tokens',capability_profile={'tools':True,'thinking':True,'reasoning_effort':True,'generation_seed':False})
    base['budget'] = json.loads((ROOT/'configs/atomic_unified_seed42_budget.json').read_text())['cells']['livemath']
    spec = yaml.safe_load((ROOT/'configs/livemath_p2_formal_seed42.yaml').read_text())
    spec['choice_batch']['contract'] = str(ROOT/spec['choice_batch']['contract'])
    base = formal.configure(base,spec,'livemath',42,AUTHORITY)
    datasets = tmp_path/'resources';files = {};all_tasks = {}
    for split in ('train','val','test'):
        canonical = batch.read(AUTHORITY/'livemath'/(split+'.json'))['tasks']
        rows = [task(t['task_id'],split=split).__dict__ for t in canonical]
        for t in rows:
            t['physical_key'] = settings()['split']['selected_physical_keys'].get(t['task_id'],digest(t['task_id']))
            t['goal'] += ' [Fixture public demand '+t['task_id']+']'
        all_tasks[split] = rows
        evaluator = {t['task_id']:{'correct_choice':t['inputs']['choices'][0],'choices':t['inputs']['choices']} for t in rows}
        for name, value in [('livemath/'+split+'.json',{'tasks':rows}),('livemath/evaluator_records_'+split+'.json',evaluator)]:
            batch.write_json(datasets/name,value);files[name] = sha256(datasets/name)
    batch.write_json(datasets/'livemath_integrity.json',{'fixture':True})
    batch.write_json(datasets/'materialization.json',{'authority_sha256':sha256(AUTHORITY/'manifest.json'),'files_sha256':files,
        'livemath_normalization_version':NORMALIZATION_VERSION,'livemath_choice_projection_version':CHOICE_PROJECTION_VERSION,
        'livemath_choice_seed':CHOICE_SEED,'integrity_sha256':sha256(datasets/'livemath_integrity.json')})
    profiles = batch.read(ROOT/'benchmark_profiles.json')['profiles'];output = tmp_path/'formal'
    return base,profiles,output,datasets,all_tasks


@pytest.mark.parametrize('no_candidate',[True,False])
def test_fresh_train60_batch_frozen_val17_test100_and_resume(tmp_path,monkeypatch,no_candidate):
    calls = fixture_transport(monkeypatch,no_candidate=no_candidate)
    base,profiles,root,datasets,tasks = fixture(tmp_path,monkeypatch)
    # Stop at a real Train boundary; no early P2, Frozen or evaluation.
    first = run_formal.campaign(base,profiles,'livemath',42,root,datasets,AUTHORITY,max_new_tasks=5)
    assert first['train']['tasks'] == 5 and not (root/'p2').exists() and not (root/'train/frozen_bank').exists()
    result = run_formal.campaign(base,profiles,'livemath',42,root,datasets,AUTHORITY,resume=True)
    assert [result[s]['tasks'] for s in ('train','val','test')] == [60,17,100]
    assert result['p2']['status'] == ('completed_no_candidate' if no_candidate else 'completed_selection')
    plan = batch.read(root/'p2/plan.json')
    assert plan['candidate_units'] == [[t['physical_key'] for t in plan['generation_tasks'][i*2:i*2+2]] for i in range(2)]
    assert set(plan['candidate_units'][0]+plan['candidate_units'][1]).isdisjoint(base['experiment']['choice_selection_physical_keys'])
    assert batch.read(root/'train/execution_manifest.json')['tasks'] == ordered_train(tasks['train'],42)
    assert batch.read(root/'train/summary.json')['frozen'] is None
    assert [len(batch.read(root/s/'summary.json')['cases']) for s in ('train','val','test')] == [60,17,100]
    bank = Bank(root/'train/frozen_bank',readonly=True)
    try:
        assets = bank.all('skill')
        assert len(assets) == (0 if no_candidate else 2) and all(checked_asset(a) for a in assets)
    finally:bank.close()
    frozen = batch.read(root/'final_frozen_manifest.json')
    assert frozen['final_artifact_hash'] == frozen['hash_before_test'] == frozen['artifact_hash_after_test']
    p2_attempts = batch.read(root/'p2/budget.json')['attempts']
    ledger = batch.read(root/'budget.json')
    assert ledger['limits']['token_limit'] == 23232000 and len(ledger['attempts']) == len(calls)
    assert set(p2_attempts).issubset(ledger['attempts']) and len(p2_attempts) == result['p2']['new_http']
    assert result['p2']['new_http'] <= 80 and not result['p2']['unknown_billing']
    assert result['p2']['new_http'] == (2 if no_candidate else 78)
    logs = [json.loads(l) for l in (root/'llm_calls.jsonl').read_text().splitlines()]
    assert len(logs) == len(calls)
    train = batch.read(root/'train/summary.json')['cases']
    assert all(t['learning']['reason'] in {'held_out_train_selection','scheduled_post_train_capability_batch'} for t in train)
    assert sum(t['learning']['reason']=='held_out_train_selection' for t in train) == 12
    eval_traces = batch.read(root/'val/summary.json')['cases']+batch.read(root/'test/summary.json')['cases']
    assert all(bool(t['injected_guidance_ids']) == (not no_candidate) for t in eval_traces)
    assert all(not t.get('learning') and t['knowledge_before']==t['knowledge_after'] for t in eval_traces)
    before = tree_identity(root/'train/frozen_bank');paid = len(calls)
    run_formal.campaign(base,profiles,'livemath',42,root,datasets,AUTHORITY,resume=True)
    assert len(calls) == paid and tree_identity(root/'train/frozen_bank') == before


def test_shared_stage_budget_admission_and_carryforward_are_not_refilled(tmp_path):
    owner = BudgetGovernor(tmp_path/'campaign.json',token_limit=100000,request_limit=10,finish_reserve=0)
    payload = {'max_tokens':8,'messages':[]}
    owner.admit('old',payload,{})
    owner.complete({'request_id':'old','raw_usage':{'total_tokens':40},'outcome':'success'})
    batch.write_json(tmp_path/'p2/campaign_budget_prefix.json',deepcopy(owner.state))
    plan = {'new_limits':{'token_limit':10000,'request_limit':1}}
    limited = formal.BatchGovernor(owner,plan,tmp_path/'p2')
    limited.admit('new',payload,{})
    limited.complete({'request_id':'new','raw_usage':{'total_tokens':5},'outcome':'success'})
    limited = formal.BatchGovernor(owner,plan,tmp_path/'p2')
    assert limited.headroom({})['requests'] == 0
    with pytest.raises(BudgetExhausted):limited.admit('again',payload,{})
    assert set(owner.state['attempts']) == {'old','new'}
    assert list(batch.read(tmp_path/'p2/budget.json')['attempts']) == ['new']


def test_unknown_billing_and_p2_configuration_mismatch_stop_before_http(tmp_path,monkeypatch):
    base,profiles,root,datasets,tasks = fixture(tmp_path,monkeypatch)
    spec = yaml.safe_load((ROOT/'configs/livemath_p2_formal_seed42.yaml').read_text())
    spec['choice_batch']['contract'] = str(ROOT/spec['choice_batch']['contract'])
    for benchmark,seed in [('livemath',43),('officeqa',42)]:
        with pytest.raises(ValueError):formal.configure(base,spec,benchmark,seed,AUTHORITY)
    owner = BudgetGovernor(tmp_path/'unknown.json',token_limit=10000,request_limit=10,finish_reserve=0)
    batch.write_json(tmp_path/'p2/campaign_budget_prefix.json',deepcopy(owner.state))
    owner.state['unknown_billing'] = True
    limited = formal.BatchGovernor(owner,{'new_limits':{'token_limit':1000,'request_limit':5}},tmp_path/'p2')
    with pytest.raises(BudgetExhausted):limited.admit('never',{'max_tokens':1},{})
    assert not owner.state['attempts']


def test_interrupt_after_paid_unit_resumes_without_repeating_that_unit(tmp_path,monkeypatch):
    calls = fixture_transport(monkeypatch,no_candidate=True)
    base,profiles,root,datasets,tasks = fixture(tmp_path,monkeypatch)
    progress = batch.progress
    interrupted = False
    def stop(root,event,**values):
        nonlocal interrupted
        progress(root,event,**values)
        if event == 'learning_unit_completed' and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(batch,'progress',stop)
    with pytest.raises(KeyboardInterrupt):run_formal.campaign(base,profiles,'livemath',42,root,datasets,AUTHORITY)
    assert len(calls) == 61 and not (root/'train/frozen_bank').exists() and not (root/'val').exists()
    assert batch.read(root/'p2/result.json')['status'] == 'running'
    result = run_formal.campaign(base,profiles,'livemath',42,root,datasets,AUTHORITY,resume=True)
    assert result['p2']['new_http'] == 2 and len(calls) == 179
    assert result['test']['complete']


def test_carryforward_unions_physical_charges_and_keeps_original_limits(tmp_path):
    row = {'status':'metered','accounted_tokens':7,'reserved_tokens':10,'context':{}}
    old = {'limits':{'token_limit':1800000,'request_limit':80,'finish_reserve':0},'unknown_billing':False,'attempts':{'shared':row}}
    batch.write_json(tmp_path/'a/budget.json',old)
    old['attempts']['other'] = {**row,'accounted_tokens':3}
    batch.write_json(tmp_path/'b/budget.json',old)
    limits = {'token_limit':23232000,'request_limit':9165,'finish_reserve':0}
    ledger,lineage = formal.carryforward([tmp_path/'a',tmp_path/'b'],limits)
    assert ledger['limits'] == limits and lineage['inherited_tokens'] == 10 and lineage['inherited_http'] == 2
    assert lineage['bank_inherited'] is False
    old['attempts']['shared']['accounted_tokens'] = 8
    batch.write_json(tmp_path/'b/budget.json',old)
    with pytest.raises(ValueError,match='physical charge changed'):formal.carryforward([tmp_path/'a',tmp_path/'b'],limits)


def test_actual_eval_public_inputs_fit_routing_contract_without_paid_http(tmp_path,monkeypatch):
    data = os.environ.get('CF4_DATASETS');source = os.environ.get('LIVEMATH_P2_R5_SELECTION_RUN')
    if not data or not source:pytest.skip('Set the actual public resources and immutable source-checked candidate')
    calls = fixture_transport(monkeypatch,eligibility=False)
    base,_,_,_,_ = fixture(tmp_path,monkeypatch)
    cfg = batch.configuration(base,tmp_path/'public-input-check')
    system = EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    candidate = batch.read(Path(source)/'learning/0/learning.json')['candidate']
    try:
        for split in ('val','test'):
            for row in batch.read(Path(data)/'livemath'/(split+'.json'))['tasks']:
                t = PublicTask(**row);system.phase=split;system.budget_scope=split+':'+t.physical_key
                system.checkpoint=TaskCheckpoint(tmp_path/'routing'/system.budget_scope)
                system.audit_path=str(system.checkpoint.root/'requests.json')
                system._task_start=len(system.usage.events);system._request_start=len(system.requests)
                system.request_attribution={'parent_scope':system.budget_scope,'parent_token_limit':96000,'train_parent':False}
                assert system.learner.match_capability(candidate,t)['applicable'] is False
        assert len(calls)==117
        assert all(len(json.dumps(p).encode())<=16384 for p in calls)
    finally:system.close()


def test_held_out_program_outcome_never_updates_learning_statistics(tmp_path,monkeypatch):
    base,_,_,_,tasks = fixture(tmp_path,monkeypatch)
    cfg = batch.configuration(base,tmp_path/'statistics-check')
    system = EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    recorded = []
    monkeypatch.setattr(system.bank,'record',lambda attempt:recorded.append(deepcopy(attempt)))
    monkeypatch.setattr(system.learner,'learn',lambda task,trace:{'decision':'no_change'})
    selected = set(base['experiment']['choice_selection_physical_keys'])
    trace = {'score':{'hard':True},'execution':{'attempts':[{'status':'ok','output_contract_status':'valid',
        'outputs_consumed':True,'local_check':'unavailable'}]}}
    try:
        t = next(t for t in tasks['train'] if t['physical_key'] in selected)
        system.learn_trace(PublicTask(**t),deepcopy(trace))
        assert not recorded
        t = next(t for t in tasks['train'] if t['physical_key'] not in selected)
        system.learn_trace(PublicTask(**t),deepcopy(trace))
        assert len(recorded)==1 and recorded[0]['outcome']=='positive'
    finally:system.close()
