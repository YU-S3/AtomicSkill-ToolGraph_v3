"""P2-R3 admission and immutable paid-evidence replay; no live provider calls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import pytest
import requests

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import replay_choice_protocol as replay
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from test_capability_batch import candidate,fixture_source,fixture_transport


def original_run():
    value=os.environ.get('LIVEMATH_P2_R2_FINAL_RUN')
    if not value:pytest.skip('Set the immutable actual completed P2-R2 run')
    return Path(value)


def forbid_http(monkeypatch):
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(args);raise AssertionError('Offline protocol replay may not send HTTP')
    monkeypatch.setattr(requests.sessions.Session,'request',forbidden)
    monkeypatch.setattr(requests,'post',forbidden)
    return calls


def replay_identity(monkeypatch):
    monkeypatch.setattr(batch,'code_identity',lambda:{'git_sha':'explicit-intercepted-r3-fixture',
        'tracked_dirty':False,'source_sha256':'fixture-only'})


def test_exact_field_and_card_limits_and_legacy_policy():
    c=candidate();p=deepcopy(c['proposal']);rows=c['source_records']
    p.update(guidance='检'*255,applicability='查'*80)
    legal=cb.validate_candidate(p,c['capability'],rows)
    assert len(json.dumps(cb.render(legal),ensure_ascii=False))+2==512
    p['guidance']+='检'
    with pytest.raises(ValueError,match='512 Unicode characters; actual 513'):cb.validate_candidate(p,c['capability'],rows)
    p.update(guidance='检'*256,applicability='查'*79)
    assert cb.validate_candidate(p,c['capability'],rows)['method_version']==cb.VERSION
    with pytest.raises(ValueError,match='maxLength; expected <= 180'):cb.validate_candidate(p,c['capability'],rows,method_version=replay.LEGACY)
    demand=deepcopy(c['capability']['definition']);demand['required_conditions']='检'*256
    cb.validate_capability(demand,rows)
    with pytest.raises(ValueError,match='expected <= 180'):cb.validate_capability(demand,rows,method_version=replay.LEGACY)
    demand['required_conditions']+='检'
    with pytest.raises(ValueError,match='expected <= 256'):cb.validate_capability(demand,rows)


def test_all_structural_errors_are_reported_together():
    c=candidate();p=deepcopy(c['proposal']);p.update(guidance='检'*257,applicability='查'*81)
    p['references'][0]['span_id']='x'*81;p['references'][1].pop('case_id')
    with pytest.raises(ValueError) as error:cb.validate_candidate(p,c['capability'],c['source_records'])
    message=str(error.value)
    for required in ('$.guidance','$.applicability','$.references[0].span_id','$.references[1]','case_id','512 Unicode characters'):
        assert required in message
    p.pop('guidance')
    with pytest.raises(ValueError) as error:cb.validate_candidate(p,c['capability'],c['source_records'])
    assert 'missing required' in str(error.value) and '$.applicability' in str(error.value)


def test_production_repair_receives_all_errors_once_and_prompts_declare_limits(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch);post=requests.post;altered=False
    def invalid_first(url,**kwargs):
        nonlocal altered
        response=post(url,**kwargs)
        if kwargs['json']['tools'][0]['function']['name']=='submit_learning' and not altered:
            body=response.json();args=json.loads(body['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            args.update(guidance='检'*257,applicability='查'*81)
            body['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps(args)
            response.json=lambda:body;altered=True
        return response
    monkeypatch.setattr(requests,'post',invalid_first)
    _,out,plan,_=fixture_source(tmp_path,monkeypatch);system=EmpiricalSystem(batch.read(out/'config.json'),harness=AnswerAdapter('livemath',{}))
    try:
        for row in batch.read(out/'source_cases.json'):system.bank.save_case(PublicTask(**row['task']),row['experience'])
        batch.learning_boundary(system,out/'learning/0','candidate:0')
        selected=[t['physical_key'] for stage in plan['selection_tasks'].values() for t in stage]
        log=system.learner.learn_capability_batch(plan['candidate_units'][0],selected)
        assert log['decision']=='candidate' and len(calls)==4
        assert 'required_conditions must be 1-256 Unicode characters' in calls[0]['messages'][0]['content']
        prompt=calls[1]['messages'][0]['content']
        assert 'guidance must be 1-256' in prompt and 'applicability must be 1-80' in prompt and '512 Unicode' in prompt
        repair=json.dumps(calls[2]['messages'],ensure_ascii=False)
        assert '$.guidance' in repair and '$.applicability' in repair and '512 Unicode characters' in repair
        audit=batch.read(out/'learning/0/requests.json')['requests']
        assert [q['repair'] for q in audit if q['purpose']=='guidance_learning']==[0,1]
        assert not system.bank.all('skill')
    finally:system.close()


def test_actual_r2_revalidation_is_zero_http_and_preserves_all_paid_records(tmp_path,monkeypatch):
    source=original_run();before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    calls=forbid_http(monkeypatch);replay_identity(monkeypatch);out=tmp_path/'r3'
    receipt=replay.replay(source/'plan.json',out)
    assert replay.check(out/'protocol_replay.json')['status']=='passed'
    assert receipt['method_version']!=receipt['source_method_version'] and receipt['new_model_http']==0 and not calls
    first,second=receipt['units']
    assert first['status']=='ready_for_independent_content_check' and first['card_characters']==500
    assert len(first['candidate']['guidance'])==248 and len(first['candidate']['applicability'])==75
    assert first['proposal_responses'][-1]['repair_index']==1
    assert second['status']=='ready_for_guidance_generation' and len(second['capability']['definition']['required_conditions'])==231
    assert not second['proposal_responses'] and second['candidate'] is None
    assert all(r['independent_content_check']=='not_run' and r['utility_check']=='not_run' and not r['published'] for r in receipt['units'])
    assert receipt['paid_prefix']['prior_p2_tokens']==42057 and receipt['paid_prefix']['prior_p2_http']==8
    assert receipt['paid_prefix']['remaining_tokens']==1757943 and receipt['paid_prefix']['remaining_http']==72
    assert receipt['paid_prefix']['inherited_tokens']==7137815 and receipt['paid_prefix']['inherited_http']==372
    assert not (out/'bank').exists() and not (out/'frozen_bank').exists() and not (out/'plan.json').exists()
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    for field in ('candidate_units','selection_tasks','utility_policy'):
        assert receipt[field]==batch.read(source/'plan.json')[field]
    # The old bounds remain auditable; their first-error ordering is not replayed as a new model turn.
    raw=first['proposal_responses'][0]['arguments']
    errors=cb.field_errors(raw,cb.schema_for(cb.PROPOSAL,replay.LEGACY))
    assert any('$.guidance' in e for e in errors) and any('$.applicability' in e for e in errors)
    ledger=batch.read(out/'inherited_p2_budget.json');ledger['attempts'][next(iter(ledger['attempts']))]['accounted_tokens']=0
    batch.write_json(out/'inherited_p2_budget.json',ledger)
    with pytest.raises(ValueError,match='allowance'):replay.check(out/'protocol_replay.json')


@pytest.mark.parametrize('damage',['response','wire','schema','logical_id','usage','unknown_billing'])
def test_revalidation_refuses_altered_provenance_without_output_or_http(tmp_path,monkeypatch,damage):
    source=tmp_path/'copied-original';shutil.copytree(original_run(),source)
    calls=forbid_http(monkeypatch);replay_identity(monkeypatch)
    if damage=='unknown_billing':
        ledger=batch.read(source/'budget.json');ledger['unknown_billing']=True;batch.write_json(source/'budget.json',ledger)
    else:
        path=source/'learning/0/requests.json';audit=batch.read(path)
        q=next(q for q in audit['requests'] if q.get('http_attempts'));a=q['http_attempts'][0]
        if damage=='response':q['response']['tool_calls'][0]['arguments']={'decision':'no_change'}
        elif damage=='wire':a['final_payload_audit']['messages'][1]['content']='{}'
        elif damage=='schema':a['final_payload_audit']['tools'][0]['function']['parameters']['properties']['required_conditions']['maxLength']=999
        elif damage=='logical_id':q['logical_decision_id']='different'
        else:a['raw_usage']['total_tokens']+=1
        batch.write_json(path,audit)
    with pytest.raises(ValueError):replay.replay(source/'plan.json',tmp_path/'r3')
    assert not calls and not (tmp_path/'r3').exists()
