"""Actual 504 prefix: retain billing and completed computation, replay no backend."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
from atomic_skillgraph.empirical.checkpoint import native_event_restorable
from test_office_v13_crossmodel import Stream
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
import recover_office_v15_gateway as recovery


@pytest.fixture(autouse=True)
def isolate_runner_binding(monkeypatch):
    for obj,names in ((recovery.recovery,('runner','c')),
        (recovery.v15.common,('PACKAGE','frontier','VERSION')),
        (recovery.v15.previous.previous,('MODELS',))):
        for name in names:monkeypatch.setattr(obj,name,getattr(obj,name))


@pytest.mark.parametrize('change,allowed',[
    ({},True),({'state':'intent'},False),({'state':'unknown'},False),
    ({'backend_invoked':None},False),({'result.accepted':False},False),
    ({'result.data.files':['new.txt']},False),({'result.data.deleted_files':['old.txt']},False),
    ({'result.data.files':None},False),({'result.local_artifact_hashes':{'x':'hash'}},False),
    ({'result.local_artifact_identities':None},False)])
def test_only_explicit_finished_unchanged_computation_restores(change,allowed):
    event={'name':'execute_python','state':'finished','backend_invoked':True,'result':{
        'accepted':True,'data':{'files':[],'deleted_files':[]},'local_artifact_hashes':{},'local_artifact_identities':{}}}
    for key,value in change.items():
        target=event;parts=key.split('.')
        for part in parts[:-1]:target=target[part]
        target[parts[-1]]=value
    assert native_event_restorable(event,{'execute_python':{'effect':'sandbox_compute'}}) is allowed
    assert not native_event_restorable(event,{'execute_python':{'effect':'write'}})


def test_actual_gateway_prefix_continues_once_without_backend_replay(tmp_path,monkeypatch):
    recovery.bind();r=recovery.recovery.runner;c=r.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(c,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(c,'load_env',lambda _:None)
    monkeypatch.setenv('ROOTFLOWAI_API_KEY','offline-fixture')
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**k:pytest.fail('Real HTTP forbidden'))
    monkeypatch.setattr('atomic_skillgraph.harness.benchmarks.OfficeAdapter.call',lambda *a,**k:pytest.fail('Completed backend must not replay'))
    monkeypatch.setattr('atomic_skillgraph.harness.benchmarks.OfficeAdapter.call_with_broker',lambda *a,**k:pytest.fail('Completed sandbox must not replay'))
    base=recovery.BASE;before=c.sha(base/'budget.json');root=tmp_path/'resume'
    recovery.prepare(root);plan=c.verify(root)
    assert plan['recovery']['retained_completed_episodes']==4
    assert plan['recovery']['unknown_tokens_reserved']==82972
    assert plan['ledger_limits']==c.read(base/'budget.json')['limits']
    pending=next(e for e in plan['episodes'] if e.get('recovery_source'))
    path=Path(pending['path']);source=Path(pending['recovery_source'])
    events_path=next((path/'checkpoints').glob('*/*/native_events.json'));events_before=c.read(events_path)
    sent=[]
    def post(*a,**kw):
        wire=kw['json'];sent.append(wire)
        assert wire['model']=='gemini-3.8-flash'
        (root/'STOP_AFTER_EPISODE').write_text('offline boundary')
        return Stream(wire['model'],content='0.105')
    monkeypatch.setattr('requests.post',post)
    assert r.execute(root,'unused')==75
    assert len(sent)==1 and c.read(events_path)==events_before
    summary=c.read(path/'summary.json')
    assert summary['complete'] and summary['total_tokens'] is None and summary['unknown_billing_attempts']==1
    assert summary['known_total_tokens']==38400
    status=r.status(root);assert status['completed_episodes']==5 and status['new_http']==1
    ledger=c.read(root/'budget.json');prior=c.read(base/'budget.json')
    assert all(ledger['attempts'][k]==v for k,v in prior['attempts'].items())
    assert c.sha(base/'budget.json')==before and c.read(path/'freeze_record.json')['unchanged']
    old_audit=c.read(next((source/'requests').glob('*.json')))
    new_audit=c.read(next((path/'requests').glob('*.json')))
    assert new_audit['requests'][:len(old_audit['requests'])]==old_audit['requests']
