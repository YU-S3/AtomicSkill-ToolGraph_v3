"""An explicit next experiment inherits the complete R3 bill and never promotes old denials."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from test_capability_batch import fixture_transport
from test_capability_protocol_r3 import forbid_http,replay_identity


def original_run():
    value=os.environ.get('LIVEMATH_P2_R3_PAID_RUN')
    if not value:pytest.skip('Set the immutable actual completed P2-R3 run')
    return Path(value)


def test_zero_http_replay_and_plan_preserve_full_paid_prefix(tmp_path,monkeypatch):
    source=original_run();before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls=forbid_http(monkeypatch);replay_identity(monkeypatch);out=tmp_path/'r4'
    plan=batch.revise(source/'plan.json',out,experimental_revision='P2-R4')
    assert batch.check(out/'plan.json')['status']=='passed' and not calls
    original=batch.read(source/'plan.json')
    for k in ('candidate_units','selection_tasks','utility_policy','generation_tasks','source_run','new_limits'):
        assert plan[k]==original[k]
    prefix=plan['paid_prefix']
    assert prefix['prior_p2_tokens']==58444 and prefix['prior_p2_http']==11
    assert prefix['remaining_tokens']==1741556 and prefix['remaining_http']==69
    assert plan['inherited_tokens']==7154202 and plan['inherited_http']==375
    config=batch.read(out/'config.json');old_config=batch.read(source/'config.json')
    for section in ('llm','budget'):assert config[section]==old_config[section]
    assert config['runtime']['choice_bounded_finalization']==old_config['runtime']['choice_bounded_finalization']
    assert plan['prompt_revision']==cb.PROMPT_REVISION and 'protocol_continuation' not in plan and 'recovery' not in plan
    replay=batch.read(out/'prior_content_checks.json')
    assert [r['reason_characters'] for r in replay['units']]==[380,436]
    assert all(r['structure_valid'] and r['status']=='unsupported' and not r['content_passed'] and not r['published'] for r in replay['units'])
    assert replay['new_model_http']==0 and not (out/'bank').exists() and not (out/'budget.json').exists()
    assert batch.read(out/'inherited_p2_budget.json')==batch.read(source/'budget.json')
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    changed=deepcopy(plan);changed['paid_prefix']['prior_p2_tokens']-=1
    batch.write_json(out/'plan.json',changed);batch.write_json(out/'plan_identity.json',{'plan_hash':batch.digest(changed)})
    with pytest.raises(ValueError,match='provenance or allowance'):batch.check(out/'plan.json')
    assert not calls


def test_paid_path_generates_fresh_candidates_once_and_counts_every_old_attempt(tmp_path,monkeypatch):
    source=original_run();replay_identity(monkeypatch);out=tmp_path/'r4'
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    plan=batch.revise(source/'plan.json',out,experimental_revision='P2-R4')
    calls=fixture_transport(monkeypatch,check_status='unsupported');env=tmp_path/'empty.env';env.write_text('')
    result=batch.execute(out/'plan.json',env)
    names=[p['tools'][0]['function']['name'] for p in calls]
    assert names==['submit_capability','submit_learning','submit_capability_check']*2
    assert result['status']=='completed_no_candidate' and result['experimental_revision']=='P2-R4'
    assert result['new_http']==6 and result['total_p2_http']==17 and result['cumulative_http']==381
    assert result['new_tokens']==30 and result['total_p2_tokens']==58474 and result['cumulative_tokens']==7154232
    assert result['natural_candidate_count']==2 and result['revalidated_candidate_count']==0 and result['paired_tasks']==0
    for p in calls:
        props=p['tools'][0]['function']['parameters']['properties']
        for field in ('required_conditions','guidance'):
            if field in props:assert props[field]['maxLength']==180
        if 'reason' in props:assert props['reason']['maxLength']==512
        assert len(json.dumps(p,allow_nan=False).encode())<=16384
    old=batch.read(source/'budget.json');new=batch.read(out/'budget.json')
    assert all(new['attempts'][k]==v for k,v in old['attempts'].items())
    assert batch.execute(out/'plan.json',env)==result and len(calls)==6
    with pytest.raises(ValueError,match='no automatic further revision'):
        batch.revise(out/'plan.json',tmp_path/'another',experimental_revision='P2-R4')
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
