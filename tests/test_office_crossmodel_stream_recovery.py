"""Replay the actual interrupted prefix without backend replay or uncharged retries."""
import json
from pathlib import Path
import sys
import pytest
from atomic_skillgraph.empirical.checkpoint import native_event_restorable
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
import recover_office_crossmodel_stream as recovery
from test_office_v13_crossmodel import Stream


@pytest.mark.parametrize('event,allowed',[
    ({'state':'finished','backend_invoked':False,'name':'','result':{'accepted':False}},True),
    ({'state':'finished','backend_invoked':True,'name':'write','result':{'accepted':False}},False),
    ({'state':'intent','backend_invoked':False,'result':{'accepted':False}},False),
    ({'state':'unknown','backend_invoked':False,'result':{'accepted':False}},False),
    ({'state':'finished','name':'','result':{'accepted':False}},False),
    ({'state':'finished','backend_invoked':False,'local_result_read':True},True),
])
def test_restoration_requires_a_finished_nonexecuted_operation(event,allowed):
    assert native_event_restorable(event,{}) is allowed


def test_actual_stream_prefix_resumes_within_original_task_budget(tmp_path,monkeypatch):
    r=recovery.runner;c=r.common;r.fixed.bind_runner(True)
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(c,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(c,'load_env',lambda _:None)
    monkeypatch.setenv('ZHIPU_API_KEY','offline-fixture');monkeypatch.setenv('MOONSHOT_API_KEY','offline-fixture')
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**k:pytest.fail('Real HTTP forbidden'))
    monkeypatch.setattr('atomic_skillgraph.harness.benchmarks.OfficeAdapter.call',lambda *a,**k:pytest.fail('Prior backend must not replay'))
    root=tmp_path/'resume';before=c.sha(recovery.BASE/'budget.json')
    recovery.prepare(root);plan=c.verify(root)
    assert plan['recovery']['retained_completed_episodes']==23
    assert plan['recovery']['unknown_tokens_reserved']==86347
    assert plan['ledger_limits']==c.read(recovery.BASE/'budget.json')['limits']
    pending=next(e for e in plan['episodes'] if e.get('recovery_source'))
    path=Path(pending['path']);source=Path(pending['recovery_source'])
    events_path=next((path/'checkpoints').glob('*/*/native_events.json'))
    events_before=c.read(events_path);sent=[]
    def post(*a,**k):
        wire=k['json'];sent.append(wire)
        assert wire['model']=='glm-5.3-flash'
        assert not wire.get('tools'), 'Original task headroom should route to its reserved final answer'
        (root/'STOP_AFTER_EPISODE').write_text('offline test boundary')
        return Stream(wire['model'],content='offline fixture')
    monkeypatch.setattr('requests.post',post)
    assert r.execute(root,'unused')==75
    assert len(sent)==1 and c.read(events_path)==events_before
    summary=c.read(path/'summary.json');assert summary['complete'] and summary['total_tokens'] is None
    assert summary['unknown_billing_attempts']==1 and summary['known_total_tokens']==41789
    state=r.status(root);assert state['completed_episodes']==24 and state['new_http']==1
    assert state['models']['glm53_flash']['unknown_billing_attempts']==1
    ledger=c.read(root/'budget.json');prior=c.read(recovery.BASE/'budget.json')
    assert all(ledger['attempts'][k]==v for k,v in prior['attempts'].items())
    assert c.sha(recovery.BASE/'budget.json')==before
    assert c.read(path/'freeze_record.json')['unchanged']
    # Exact old requests and settled responses are retained, not reissued.
    old_audit=c.read(next((source/'requests').glob('*.json')))
    new_audit=c.read(next((path/'requests').glob('*.json')))
    assert new_audit['requests'][:len(old_audit['requests'])]==old_audit['requests']
