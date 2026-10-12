"""Actual R4 failures, a bounded explicit retry, and the full production path with intercepted HTTP."""
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from test_capability_batch import fixture_transport
from test_capability_protocol_r3 import forbid_http,replay_identity


def original_run():
    value=os.environ.get('LIVEMATH_P2_R4_PAID_RUN')
    if not value:pytest.skip('Set the immutable actual completed P2-R4 run')
    return Path(value)


def test_actual_r4_failures_remain_real_and_all_costs_transfer(tmp_path,monkeypatch):
    source=original_run();before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls=forbid_http(monkeypatch);replay_identity(monkeypatch);out=tmp_path/'r5'
    plan=batch.revise(source/'plan.json',out,experimental_revision='P2-R5')
    assert batch.check(out/'plan.json')['status']=='passed' and not calls
    prefix=plan['paid_prefix']
    assert prefix['prior_p2_tokens']==84292 and prefix['prior_p2_http']==16
    assert prefix['remaining_tokens']==1715708 and prefix['remaining_http']==64
    assert plan['inherited_tokens']==7180050 and plan['inherited_http']==380
    original=batch.read(source/'plan.json')
    for k in ('candidate_units','selection_tasks','utility_policy','generation_tasks','source_run','new_limits'):
        assert plan[k]==original[k]
    review=batch.read(out/'prior_content_checks.json')['units']
    assert review[0]['reason_characters']==546 and review[0]['status']=='domain_fact'
    assert not review[0]['structure_valid'] and not review[0]['content_passed'] and not review[0]['published']
    assert review[1]['stage']=='capability_induction' and review[1]['status']=='not_run'
    assert batch.read(out/'inherited_p2_budget.json')==batch.read(source/'budget.json')
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('grounding',['procedural_only','unsupported'])
def test_full_paid_path_stays_bounded_and_only_model_verdicts_reach_pairs(tmp_path,monkeypatch,grounding):
    source=original_run();replay_identity(monkeypatch);out=tmp_path/'r5'
    plan=batch.revise(source/'plan.json',out,experimental_revision='P2-R5')
    chosen=plan['selection_tasks']['screen'][:2]+plan['selection_tasks']['confirmation'][:2]
    goals={t['goal'] for t in chosen}
    evaluator=batch.evaluator_records(batch.read(Path(plan['source_run'])/'train/summary.json'))
    correct={t['goal']:evaluator[t['task_id']]['correct_choice']['label'] for t in chosen}
    def answer(body):
        label=correct[body['goal']]
        return label if body.get('guidance') else next(c['label'] for c in body['inputs']['choices'] if c['label']!=label)
    calls=fixture_transport(monkeypatch,check_status=grounding,eligibility=lambda b:b['task']['goal'] in goals,answers=answer)
    env=tmp_path/'empty.env';env.write_text('');result=batch.execute(out/'plan.json',env)
    names=[p.get('tools',[{'function':{'name':''}}])[0]['function']['name'] for p in calls]
    assert names[:6]==['submit_capability','submit_learning','submit_capability_check']*2
    assert result['status']==('completed_selection' if grounding=='procedural_only' else 'completed_no_candidate')
    assert result['experimental_revision']=='P2-R5' and result['total_p2_http']==16+len(calls)<=80
    assert result['total_p2_tokens']==84292+result['new_tokens'] and result['cumulative_http']==380+len(calls)
    for p in calls:
        assert len(json.dumps(p,allow_nan=False).encode())<=16384
        if p.get('tools'):
            prompt=p['messages'][0]['content'];props=p['tools'][0]['function']['parameters']['properties']
            for field in ('required_conditions','guidance','reason'):
                if field in props:
                    assert props[field]['maxLength']==180 and 'aim <=140' in prompt
            if 'required_conditions' in props:assert 'No digits, formulas, LaTeX' in prompt
    frozen=Bank(out/'frozen_bank',readonly=True)
    try:assets=frozen.all('skill')
    finally:frozen.close()
    assert len(assets)==(2 if grounding=='procedural_only' else 0)
    if grounding=='procedural_only':
        assert names.count('submit_capability_match')==24 and names.count('answer_step')==16
        assert all(cb.checked_asset(a) for a in assets)
    else:assert 'submit_capability_match' not in names and 'answer_step' not in names
    old=batch.read(source/'budget.json');new=batch.read(out/'budget.json')
    assert all(new['attempts'][k]==v for k,v in old['attempts'].items())
    assert batch.execute(out/'plan.json',env)==result
    with pytest.raises(ValueError,match='no automatic further revision'):
        batch.revise(out/'plan.json',tmp_path/'another',experimental_revision='P2-R5')
