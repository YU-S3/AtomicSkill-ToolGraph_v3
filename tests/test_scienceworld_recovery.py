import json
import pytest
from experiments.scienceworld_recovery import read, sha, NAME


def test_recovery_preserves_original_identity_and_rejects_drift(tmp_path):
    output=tmp_path/'train'
    output.mkdir()
    manifest=tmp_path/'run/run_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'code_commit':'old'}))
    config={'experiment':{'output_dir':str(output),'name':'run'}}
    receipt={'run_id':'run','config_hash':'cfg','execution_code_hash':'new',
             'original_code_hash':'old','manifest_sha256':sha(manifest),'quarantined_files':[]}
    (output/NAME).write_text(json.dumps(receipt))
    assert read(config,'cfg','new')['original_code_hash']=='old'
    with pytest.raises(ValueError,match='exact run'):
        read(config,'cfg','other')
    with pytest.raises(ValueError,match='exact run'):
        read(config,'different','new')
    assert json.loads(manifest.read_text())=={'code_commit':'old'}


def test_recovery_quarantine_remains_integrity_checked(tmp_path):
    output=tmp_path/'train';output.mkdir()
    manifest=tmp_path/'run/run_manifest.json';manifest.parent.mkdir()
    manifest.write_text(json.dumps({'code_commit':'old'}))
    evidence=output/'unknown.json';evidence.write_text('unknown, not zero')
    receipt={'run_id':'run','config_hash':'cfg','execution_code_hash':'new',
        'original_code_hash':'old','manifest_sha256':sha(manifest),
        'quarantined_files':[{'path':'unknown.json','sha256':sha(evidence)}]}
    (output/NAME).write_text(json.dumps(receipt))
    config={'experiment':{'output_dir':str(output),'name':'run'}}
    assert read(config,'cfg','new')
    evidence.write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):
        read(config,'cfg','new')


def test_partial_maintenance_recovery_retains_known_usage_and_marks_unknown(tmp_path, monkeypatch):
    import shutil
    import experiments.scienceworld_recovery as recovery
    from experiments.protocol import AttemptTraceLedger
    from test_attempt_trace_ledger import _write_trace
    output=tmp_path/'train';output.mkdir()
    manifest=tmp_path/'run/run_manifest.json';manifest.parent.mkdir()
    manifest.write_text(json.dumps({'code_commit':'old'}))
    config={'experiment':{'output_dir':str(output),'name':'run'}}
    monkeypatch.setattr(recovery,'load_config',lambda _:config)
    monkeypatch.setattr(recovery,'hash_config',lambda _:'cfg')
    monkeypatch.setattr(recovery,'hash_code',lambda _:'new')
    ledger=AttemptTraceLedger(output/'attempt_history',output/'traces')
    attempt=ledger.begin(run_id='run',task_id='task',task_signature='sig',attempt_kind='task',sequence=1,
                         expected_periodic_milestone='online_success_10')
    _write_trace(output/'traces',trace_id='trace_partial',task_id='task',task_signature='sig',
                 event_id='paid_event',tokens=123,cost=.01,success=True)
    backup=tmp_path/'backup';shutil.copytree(output,backup)
    with pytest.raises(ValueError,match='explicit'):
        recovery.prepare('config',backup,'crash')
    receipt=recovery.prepare('config',backup,'crash',True)
    assert receipt['unknown_interrupted_attempts']==1
    assert not ledger.pending(run_id='run')
    assert not (output/'attempt_history'/f'{attempt.attempt_id}.capture.json').exists()
    rows=recovery.recovery_usage_traces(output,receipt)
    assert len(rows)==1 and rows[0]['llm_usage'][0]['total_tokens']==123
    assert (output/'traces/trace_partial.json').read_bytes()==(backup/'traces/trace_partial.json').read_bytes()
    assert recovery.read(config,'cfg','new')==receipt
