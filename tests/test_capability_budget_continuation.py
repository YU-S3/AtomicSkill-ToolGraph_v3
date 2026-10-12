"""Use the real costly prefix that exposed scope collisions, rather than five-token prefixes alone."""
import os
from pathlib import Path

import pytest

from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from test_capability_batch import fixture_transport
from test_capability_protocol_r3 import forbid_http,replay_identity


def original_run():
    value=os.environ.get('LIVEMATH_P2_R5_STOPPED_RUN')
    if not value:pytest.skip('Set the immutable actual locally refused R5 run')
    return Path(value)


def test_zero_http_continuation_keeps_induction_identity_and_every_charge(tmp_path,monkeypatch):
    source=original_run();before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls=forbid_http(monkeypatch);replay_identity(monkeypatch);out=tmp_path/'continued'
    plan=batch.continue_budget(source/'plan.json',out)
    assert batch.check(out/'plan.json')['status']=='passed' and not calls
    assert plan['scope_policy']==batch.SCOPE_POLICY
    assert plan['paid_prefix']['prior_p2_tokens']==89644 and plan['paid_prefix']['prior_p2_http']==17
    assert plan['paid_prefix']['remaining_tokens']==1710356 and plan['paid_prefix']['remaining_http']==63
    assert plan['inherited_tokens']==7185402 and plan['inherited_http']==381
    demand=plan['budget_continuation']['demands'][0]
    assert demand['http_attempt_id']=='provider_request_6988cb2e36f941e3804e762cbf20561b'
    assert demand['new_model_http']==0 and plan['budget_continuation']['demands'][1] is None
    assert batch.read(out/'inherited_p2_budget.json')==batch.read(source/'budget.json')
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('grounding',['procedural_only','unsupported'])
def test_real_high_cost_prefix_cannot_exhaust_new_unit_or_pair_pools(tmp_path,monkeypatch,grounding):
    source=original_run();replay_identity(monkeypatch);out=tmp_path/'continued'
    plan=batch.continue_budget(source/'plan.json',out)
    old=batch.read(source/'budget.json')
    assert sum(a['accounted_tokens'] for a in old['attempts'].values() if a['context'].get('parent_scope')=='candidate:0')==65347
    chosen=plan['selection_tasks']['screen'][:2]+plan['selection_tasks']['confirmation'][:2];goals={t['goal'] for t in chosen}
    evaluator=batch.evaluator_records(batch.read(Path(plan['source_run'])/'train/summary.json'))
    correct={t['goal']:evaluator[t['task_id']]['correct_choice']['label'] for t in chosen}
    def answer(body):
        label=correct[body['goal']]
        return label if body.get('guidance') else next(c['label'] for c in body['inputs']['choices'] if c['label']!=label)
    calls=fixture_transport(monkeypatch,check_status=grounding,eligibility=lambda b:b['task']['goal'] in goals,answers=answer)
    env=tmp_path/'empty.env';env.write_text('');result=batch.execute(out/'plan.json',env)
    names=[p.get('tools',[{'function':{'name':''}}])[0]['function']['name'] for p in calls]
    assert names[:5]==['submit_learning','submit_capability_check','submit_capability','submit_learning','submit_capability_check']
    assert names.count('submit_capability')==1
    assert result['status']==('completed_selection' if grounding=='procedural_only' else 'completed_no_candidate')
    assert result['new_http']==len(calls) and result['total_p2_http']==17+len(calls)<=80
    assert result['total_p2_tokens']==89644+result['new_tokens'] and result['cumulative_tokens']==7185402+result['new_tokens']
    ledger=batch.read(out/'budget.json');added=set(ledger['attempts'])-set(old['attempts'])
    assert all(ledger['attempts'][k]==v for k,v in old['attempts'].items())
    # Learning/routing use explicit run scopes; Runtime already scopes each arm by its fresh output directory.
    assert all(ledger['attempts'][k]['context']['parent_scope'].startswith((plan['run_id']+':',str(out/'candidates')+'/')) for k in added)
    for p in (out/'candidates').glob('*/selection/*/*/trace.json'):
        assert batch.read(p)['attempt_id'].startswith(plan['run_id']+':')
    assert batch.execute(out/'plan.json',env)==result
