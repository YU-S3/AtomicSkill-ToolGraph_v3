"""Schema-only P3 continuation preserves paid responses, negatives and stage charges."""
from copy import deepcopy
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.experiments import math_continuation as continuation
from atomic_skillgraph.experiments import run_math_p3 as p3
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from test_math_p3 import prepared,transport


def stopped(tmp_path,monkeypatch):
    _,parent,env=prepared(tmp_path,monkeypatch);calls=transport(monkeypatch)
    save=TaskCheckpoint.save_response;runtime=[0]
    def interrupt(self,key,value):
        save(self,key,value)
        if '/selection/' in str(self.root):
            runtime[0]+=1
            if runtime[0]==2:raise SystemExit('stop after metered response, before trace')
    with monkeypatch.context() as patch:
        patch.setattr(TaskCheckpoint,'save_response',interrupt)
        with pytest.raises(SystemExit):p3.execute(parent,env)
    batch.write_json(parent/'stop_receipt.json',{'status':'stopped_for_explicit_schema_continuation'})
    monkeypatch.setattr(p3,'code_identity',lambda:{'git_sha':'p3-schema-fixed','tracked_dirty':False,'source_sha256':'p3-schema-fixed-source'})
    return parent,env,calls


def test_continuation_replays_paid_terminal_and_continues_only_unfinished(tmp_path,monkeypatch):
    parent,env,calls=stopped(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()};count=len(calls)
    root=tmp_path/'child';result=continuation.prepare(parent,root)
    assert result['status']=='passed' and len(calls)==count
    plan=batch.read(root/'plan.json');old=batch.read(parent/'plan.json');inherited=plan['continuation']
    assert plan['code']!=old['code'] and plan['run_id']!=old['run_id']
    assert plan['identity']['parent_initial_artifact_hash']==old['identity']['initial_artifact_hash']
    assert plan['identity']['initial_artifact_hash']==continuation.tree_identity(root/'learning/bank')['sha256']
    assert inherited['candidate_hashes']==batch.read(parent/'learning/result.json')['candidate_hashes']
    arms=[r for rows in inherited['arms'].values() for r in rows.values()]
    assert len(arms)==2 and sum('recovered' in r['path'] for r in arms)==1
    old_attempts=batch.read(parent/'budget.json')['attempts']
    assert batch.read(root/'budget.json')['attempts']==old_attempts
    result=p3.execute(root,env);assert result['status']=='awaiting_test_review' and not result['test_started']
    assert all(w.get('tools',[{'function':{'name':''}}])[0]['function']['name']!='submit_learning' for w in calls[count:])
    assert len(calls)==len(batch.read(root/'budget.json')['attempts'])
    assert all(batch.read(root/'budget.json')['attempts'][k]==v for k,v in old_attempts.items())
    for stage in ('learning','val'):
        ledger=batch.read(root/stage/'budget.json');old_stage=batch.read(parent/stage/'budget.json')
        assert ledger['limits']==old_stage['limits']
        assert all(ledger['attempts'][k]==v for k,v in old_stage['attempts'].items())
    assert len(batch.read(root/'learning/budget.json')['attempts'])==len(batch.read(parent/'learning/budget.json')['attempts'])
    assert before=={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    count=len(calls);p3.execute(root,env);assert len(calls)==count
    # The child may not silently discard the imported stage cost on a later resume.
    batch.write_json(root/'val/inherited_stage_attempts.json',{})
    with pytest.raises(ValueError,match='Prepared P3 input changed'):p3.check(root)


def test_continuation_rejects_unknown_cost_changed_source_or_new_thresholds(tmp_path,monkeypatch):
    parent,_,calls=stopped(tmp_path,monkeypatch);count=len(calls)
    ledger=batch.read(parent/'budget.json');changed=deepcopy(ledger)
    next(iter(changed['attempts'].values()))['status']='admitted'
    batch.write_json(parent/'budget.json',changed)
    with pytest.raises(ValueError,match='metered'):continuation.prepare(parent,tmp_path/'unknown')
    batch.write_json(parent/'budget.json',ledger)
    settings=batch.read(p3.CONTRACT);settings['selection_policy']['screen']['min_exposed']=1
    contract=tmp_path/'changed_contract.json';batch.write_json(contract,settings)
    with monkeypatch.context() as patch:
        patch.setattr(p3,'CONTRACT',contract)
        with pytest.raises(ValueError,match='Only the declared'):continuation.prepare(parent,tmp_path/'different_policy')
    learned=batch.read(parent/'learning/result.json');learned['candidates'][0]['guidance']='Changed'
    batch.write_json(parent/'learning/result.json',learned)
    with pytest.raises(ValueError,match='candidate identity'):continuation.prepare(parent,tmp_path/'changed_candidate')
    assert len(calls)==count


def test_missing_response_cache_key_cannot_repeat_paid_http(tmp_path,monkeypatch):
    parent,_,calls=stopped(tmp_path,monkeypatch);count=len(calls)
    partial=next(p.parent for p in parent.glob('val/candidates/*/selection/*/*/checkpoint') if not (p.parent/'trace.json').exists())
    response=next((partial/'checkpoint/responses').glob('*.json'))
    response.rename(response.with_name('not-the-original-cache-key.json'))
    with pytest.raises(ValueError,match='paid replay is forbidden'):continuation.prepare(parent,tmp_path/'cache_miss')
    assert len(calls)==count


@pytest.mark.skipif(not Path('/home/yangchengyu/livemath_p3_seed42_20261014/stop_receipt.json').exists(),reason='saved P3 stop not on this host')
def test_actual_metered_boundary_zero_http(tmp_path,monkeypatch):
    parent=Path('/home/yangchengyu/livemath_p3_seed42_20261014')
    before={str(p):batch.sha256(p) for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(p3,'code_identity',lambda:{'git_sha':'offline-test-only','tracked_dirty':False,'source_sha256':'offline-test-only'})
    monkeypatch.setenv('DEEPSEEK_API_KEY','offline-replay-no-network')
    def forbidden(*args,**kwargs):raise AssertionError('Offline HTTP transport called')
    monkeypatch.setattr('requests.post',forbidden)
    root=tmp_path/'actual';assert continuation.prepare(parent,root)['status']=='passed'
    imported=batch.read(root/'continuation_manifest.json')
    old_learning=batch.read(parent/'learning/budget.json')['attempts']
    assert imported['prior_stage_cost']['learning']=={'tokens':sum(a['accounted_tokens'] for a in old_learning.values()),'http':len(old_learning)}
    assert sum(len(a) for a in imported['arms'].values())==6
    assert sum(len(r) for r in imported['routes'].values())==6
    assert before=={str(p):batch.sha256(p) for p in parent.rglob('*') if p.is_file()}
