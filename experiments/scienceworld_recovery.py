"""Explicit recovery receipt; old manifests/traces are never relabelled.

Preparing recovery requires a separate full backup and explicit acknowledgement
of untraced interrupted requests. Those attempts are retained in quarantine,
not represented as successful tasks or zero-cost calls.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from atomic_skillgraph.core.serialization import atomic_create_json
from atomic_skillgraph.system import load_config
from experiments.protocol import AttemptTraceLedger, ProtocolError, hash_code, hash_config, _file_hashes

ROOT = Path(__file__).resolve().parents[1]
NAME = 'scienceworld_recovery.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(config):
    e = config['experiment']
    return Path(e['output_dir']).parent / e['name'] / 'run_manifest.json'


def read(config, config_digest, execution_hash):
    output = Path(config['experiment']['output_dir'])
    path = output / NAME
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    manifest = json.loads(manifest_path(config).read_text())
    expected = {'run_id': config['experiment']['name'], 'config_hash': config_digest,
                'execution_code_hash': execution_hash, 'original_code_hash': manifest['code_commit'],
                'manifest_sha256': sha(manifest_path(config))}
    if any(receipt.get(k) != v for k,v in expected.items()):
        raise ValueError('Recovery receipt does not match exact run/config/execution code')
    for row in [*receipt.get('quarantined_files', []), *receipt.get('partial_trace_files', [])]:
        if sha(output / row['path']) != row['sha256']:
            raise ValueError('Quarantined interrupted evidence changed')
    return receipt


def partial_periodic_traces(ledger, attempt):
    """Identify an interrupted maintenance tail without accepting it as complete."""
    start = ledger._read(ledger._start_path(attempt.attempt_id))
    baseline = ledger._validate_baseline_hashes(start)
    ids = sorted(set(ledger._trace_hashes()) - set(baseline))
    if not ids or not attempt.expected_periodic_milestone:
        return []
    for trace_id in ids:
        ledger._validate_owned_trace(start, trace_id)
    try:
        ledger._validate_periodic_expectation(start, ids)
    except ProtocolError as exc:
        if 'lacks expected periodic maintenance Trace' not in str(exc):
            raise
        return [ledger.trace_root / f'{trace_id}.json' for trace_id in ids]
    return []


def recovery_usage_traces(output, receipt):
    rows = []
    for item in (receipt or {}).get('partial_trace_files', []):
        path = Path(output) / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Interrupted Trace changed')
        rows.append(json.loads(path.read_text()))
    return rows


def prepare(config_path, backup, reason, allow_untraced=False):
    config = load_config(config_path)
    output = Path(config['experiment']['output_dir'])
    backup = Path(backup).resolve()
    if backup == output or output in backup.parents or backup in output.parents:
        raise ValueError('Backup must be independent of output')
    if not (backup/'traces').is_dir() or not (backup/'attempt_history').is_dir():
        raise ValueError('A full pre-recovery episode backup is required')
    if (output/NAME).exists():
        return read(config, hash_config(config_path), hash_code(ROOT))
    # Verify all immutable task evidence and the retained checkpoint before any move.
    for directory in ('traces', 'attempt_history', '.task_checkpoint'):
        for source in (output/directory).rglob('*'):
            if source.is_file() and sha(source) != sha(backup/source.relative_to(output)):
                raise ValueError(f'Backup differs: {source}')
    checkpoint = output/'.task_checkpoint'
    if checkpoint.exists():
        cp = json.loads((checkpoint/'checkpoint_manifest.json').read_text())
        if cp['files'] != _file_hashes(checkpoint, exclude={'checkpoint_manifest.json'}):
            raise ValueError('Checkpoint integrity mismatch')
    run_id = config['experiment']['name']
    ledger = AttemptTraceLedger(output/'attempt_history', output/'traces')
    quarantined=[]
    partial_files=[]
    partial_attempts=0
    for attempt in ledger.pending(run_id=run_id):
        paths = partial_periodic_traces(ledger, attempt)
        if not paths:
            continue
        if not allow_untraced:
            raise ValueError('Interrupted maintenance usage requires explicit --allow-untraced')
        target=output/'recovery_quarantine'/attempt.attempt_id
        target.mkdir(parents=True,exist_ok=True)
        source=ledger._start_path(attempt.attempt_id)
        destination=target/source.name
        if destination.exists():raise FileExistsError(destination)
        shutil.move(str(source),str(destination))
        quarantined.append({'path':str(destination.relative_to(output)), 'sha256':sha(destination)})
        partial_files.extend({'path':str(p.relative_to(output)), 'sha256':sha(p)} for p in paths)
        partial_attempts+=1
    ledger.recover_pending(run_id=run_id)
    unresolved = ledger.unresolved(run_id=run_id)
    if unresolved and not allow_untraced:
        raise ValueError('Untraced interrupted usage requires explicit --allow-untraced')
    for row in unresolved:
        attempt_id=row['attempt_id']
        target=output/'recovery_quarantine'/attempt_id
        target.mkdir(parents=True,exist_ok=True)
        for suffix in ('start','unresolved'):
            source=output/'attempt_history'/f'{attempt_id}.{suffix}.json'
            destination=target/source.name
            if destination.exists():raise FileExistsError(destination)
            shutil.move(str(source),str(destination))
            quarantined.append({'path':str(destination.relative_to(output)), 'sha256':sha(destination)})
    manifest=json.loads(manifest_path(config).read_text())
    receipt={'schema_version':1,'kind':'scienceworld_audited_software_recovery',
        'run_id':run_id,'config_hash':hash_config(config_path),'original_code_hash':manifest['code_commit'],
        'execution_code_hash':hash_code(ROOT),'manifest_sha256':sha(manifest_path(config)),
        'created_at':datetime.now(timezone.utc).isoformat(),'reason':reason,'backup':str(backup),
        'unknown_interrupted_attempts':len(unresolved)+partial_attempts,'quarantined_files':quarantined,
        'partial_trace_files':partial_files,
        'usage_policy':'Recorded totals only; untraced interrupted usage is unknown, never zero.',
        'task_results_policy':'Keep completed tasks; rerun interrupted task from its pre-task state.'}
    atomic_create_json(output/NAME,receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--backup',required=True)
    p.add_argument('--reason',required=True);p.add_argument('--allow-untraced',action='store_true')
    a=p.parse_args();print(json.dumps(prepare(a.config,a.backup,a.reason,a.allow_untraced),indent=2))
