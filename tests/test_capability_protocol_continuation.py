"""Actual saved P2 responses enter the paid production path through intercepted HTTP."""
from copy import deepcopy
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.experiments import replay_choice_protocol as replay
from test_capability_batch import fixture_transport
from test_capability_protocol_r3 import original_run,forbid_http,replay_identity


def prepared(tmp_path,monkeypatch):
    replay_identity(monkeypatch)
    receipt=replay.replay(original_run()/'plan.json',tmp_path/'receipt')
    out=tmp_path/'paid'
    plan=batch.continue_protocol(tmp_path/'receipt/protocol_replay.json',out)
    env=tmp_path/'empty.env';env.write_text('')
    return receipt,out,plan,env


def test_prepare_preserves_budget_sources_and_selection_with_zero_http(tmp_path,monkeypatch):
    calls=forbid_http(monkeypatch);receipt,out,plan,_=prepared(tmp_path,monkeypatch)
    assert batch.check(out/'plan.json')['status']=='passed' and not calls
    assert plan['paid_prefix']['prior_p2_tokens']==42057 and plan['paid_prefix']['remaining_tokens']==1757943
    assert plan['new_limits']=={'token_limit':1800000,'request_limit':80}
    assert plan['inherited_tokens']==7137815 and plan['inherited_http']==372
    for field in ('candidate_units','selection_tasks','utility_policy'):assert plan[field]==receipt[field]
    assert plan['protocol_continuation']['proposals'][0]['original_response']['repair_index']==1
    assert plan['protocol_continuation']['proposals'][1] is None
    assert plan['config_transition']['current']['runtime']['choice_bounded_finalization']['enabled'] is False
    assert not (out/'bank').exists() and not (out/'budget.json').exists()
    ledger=batch.read(out/'inherited_p2_budget.json');ledger['attempts'][next(iter(ledger['attempts']))]['accounted_tokens']=0
    batch.write_json(out/'inherited_p2_budget.json',ledger)
    with pytest.raises(ValueError,match='Prepared material changed'):batch.check(out/'plan.json')


@pytest.mark.parametrize('grounding',['procedural_only','unsupported'])
def test_actual_paid_path_reuses_only_legal_outputs_and_runs_original_gates(tmp_path,monkeypatch,grounding):
    receipt,out,plan,env=prepared(tmp_path,monkeypatch)
    inherited=batch.read(out/'inherited_p2_budget.json')
    protected={str(p):p.read_bytes() for p in original_run().rglob('*') if p.is_file()}
    chosen=plan['selection_tasks']['screen'][:2]+plan['selection_tasks']['confirmation'][:2]
    goals={t['goal'] for t in chosen}
    evaluator=batch.evaluator_records(batch.read(Path(plan['source_run'])/'train/summary.json'))
    correct={t['goal']:evaluator[t['task_id']]['correct_choice']['label'] for t in chosen}
    def answer(body):
        label=correct[body['goal']]
        return label if body.get('guidance') else next(c['label'] for c in body['inputs']['choices'] if c['label']!=label)
    calls=fixture_transport(monkeypatch,eligibility=lambda b:b['task']['goal'] in goals,answers=answer,check_status=grounding)
    result=batch.execute(out/'plan.json',env)
    names=[p.get('tools',[{'function':{'name':''}}])[0]['function']['name'] for p in calls]
    assert names[:3]==['submit_capability_check','submit_learning','submit_capability_check']
    assert 'submit_capability' not in names and names.count('submit_learning')==1
    assert result['new_http']==len(calls) and result['total_p2_http']==8+len(calls)
    assert result['total_p2_tokens']==42057+result['new_tokens']
    assert result['cumulative_tokens']==7137815+result['new_tokens']
    assert result['revalidated_candidate_count']==1 and result['natural_candidate_count']==1
    assert not result['source_train_runtime_reexecuted']
    first=batch.read(out/'learning/0/learning.json')
    assert first['proposal']==receipt['units'][0]['proposal_responses'][-1]['arguments']
    assert first['proposal_origin']=='saved_paid_v1_revalidated'
    for attempt_id,entry in inherited['attempts'].items():assert batch.read(out/'budget.json')['attempts'][attempt_id]==entry
    frozen=Bank(out/'frozen_bank',readonly=True)
    try:assets=frozen.all('skill')
    finally:frozen.close()
    assert result['status']==('completed_selection' if grounding=='procedural_only' else 'completed_no_candidate')
    if grounding=='procedural_only':
        assert len(assets)==2 and all(cb.checked_asset(a) for a in assets)
        assert names.count('submit_capability_match')==24 and names.count('answer_step')==16
    else:assert not assets and 'answer_step' not in names and 'submit_capability_match' not in names
    assert protected=={str(p):p.read_bytes() for p in original_run().rglob('*') if p.is_file()}
    previous=len(calls);assert batch.execute(out/'plan.json',env)==result and len(calls)==previous


def test_changed_receipt_is_not_a_new_paid_candidate(tmp_path,monkeypatch):
    calls=forbid_http(monkeypatch);replay_identity(monkeypatch)
    receipt=replay.replay(original_run()/'plan.json',tmp_path/'receipt')
    changed=deepcopy(receipt);changed['units'][0]['proposal_responses'][-1]['arguments']['guidance']='Invented replacement'
    batch.write_json(tmp_path/'receipt/protocol_replay.json',changed)
    batch.write_json(tmp_path/'receipt/receipt_identity.json',{'receipt_hash':digest(changed)})
    with pytest.raises(ValueError,match='production revalidation'):batch.continue_protocol(tmp_path/'receipt/protocol_replay.json',tmp_path/'paid')
    assert not calls and not (tmp_path/'paid').exists()
