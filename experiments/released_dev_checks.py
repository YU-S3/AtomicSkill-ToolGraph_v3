"""Read-only finite-release acceptance; never launches episodes or edits banks."""
import json
import sqlite3
from pathlib import Path
from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.deployment.release_protocol import ReleaseError, sha
from .protocol import hash_code, hash_config
from .compiler_metrics import search_metrics, task_attempt_costs

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def check_episode(output, *, code_hash, bank_digest, entry, profile):
    output = Path(output)
    config, manifest, summary = (read(output / name) for name in ('config.json', 'run_manifest.json', 'summary.json'))
    config.pop('_config_path', None)  # Loader provenance is not part of the authored config hash.
    progress, tasks = read(output/'progress.json'), read(output/'task_manifest.json')['tasks']
    if (progress.get('state') != 'completed' or progress.get('completed') != 1 or
        manifest['code_commit'] != code_hash or summary['code_hash'] != code_hash or
        read(output/'execution_code_manifest.json')['code_hash'] != code_hash or
        manifest['config_hash'] != hash_config(config) or summary['config_hash'] != hash_config(config) or
        manifest['knowledge_digest'] != bank_digest or summary['bank_digest'] != bank_digest or
        config['bank_release']['expected_bank_digest'] != bank_digest or
        summary['profile'] != profile or config['deployment']['presentation_profile'] != profile or
        len(tasks) != 1 or tasks[0]['task_id'] != entry['task_id'] or tasks[0]['task_signature'] != entry['task_signature']):
        raise ReleaseError(f'finite dev identity/completion mismatch: {output}')
    digest = read(output/'bank_digest_audit.json')
    if not digest['unchanged'] or digest['before'] != bank_digest or digest['after'] != bank_digest:
        raise ReleaseError('finite dev bank mutation')
    with sqlite3.connect(f'file:{output / "run_state.sqlite3"}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT task_id,trace_id,state FROM run_tasks').fetchall()
    if len(rows) != 1 or rows[0][0] != entry['task_id'] or rows[0][2] != 'completed':
        raise ReleaseError('finite dev task ledger not terminal')
    trace_id = rows[0][1]
    traces = [read(p) for p in sorted((output/'traces').glob('trace_*.json'))]
    final = next(t for t in traces if t['trace_id'] == trace_id)
    if final.get('infrastructure_failure') or summary.get('unknown_usage_requests') != 0:
        raise ReleaseError('finite dev unresolved infrastructure/usage')
    costs = task_attempt_costs(traces).get(entry['task_id'])
    if costs is None or costs['total_tokens'] is None or costs['total_tokens'] != summary['total_recorded_tokens']:
        raise ReleaseError('finite dev full-attempt cost mismatch')
    # Include every retained usage event, not only a final successful attempt.
    events = {e['event_id']: e for t in traces for e in t.get('llm_usage', [])}
    if {e['event_id']: e for e in read(output/'all_usage.json')} != events:
        raise ReleaseError('finite dev usage ledger mismatch')
    for start in (output/'attempt_history').glob('*.start.json'):
        if not start.with_name(start.name.replace('.start.json', '.capture.json')).is_file():
            raise ReleaseError('finite dev uncaptured attempt')
    from .protocol import AttemptTraceLedger
    ledger = AttemptTraceLedger(output/'attempt_history', output/'traces')
    if not ledger.owner_path.is_file() or not list((output/'attempt_history').glob('*.start.json')):
        raise ReleaseError('finite dev attempt ownership missing')
    ledger._validate_run_files(manifest['run_id'])
    for capture_path in (output/'attempt_history').glob('*.capture.json'):
        start_path = capture_path.with_name(capture_path.name.replace('.capture.json', '.start.json'))
        ledger._validate_captured_hashes(read(start_path), read(capture_path))
    if ledger.unresolved(run_id=manifest['run_id']):
        raise ReleaseError('finite dev unresolved attempt')
    metrics = search_metrics(final)
    if metrics.get('search_history_capture_status') != 'complete' or metrics.get('interrupted_native_calls_missing'):
        raise ReleaseError('finite dev missing runtime history/call capture')
    if any(not row['history_present_and_exact'] for row in metrics['search_history_request_checks']):
        raise ReleaseError('finite dev history missing from actual HTTP request')
    root_cost = read(output/'compiler_summary.json')['cost_distributions']
    if root_cost != read(output/'reports/compiler_summary.json')['cost_distributions'] or root_cost['total_tokens']['known'] != 1:
        raise ReleaseError('finite dev compiler cost reports disagree')
    return {'output': str(output), 'trace_id': trace_id, 'trace_sha256': sha(output/'traces'/f'{trace_id}.json'),
            'summary_sha256': sha(output/'summary.json'), 'costs': costs, 'search_metrics': metrics}


def verify_dev(root, *, write=False):
    from .run_v3_r103_validation import declared_entries
    root = Path(root)
    manifest = read(root/'seed42/frozen/release_manifest.json')
    code = hash_code(REPO)
    if manifest['code_hash'] != code:
        raise ReleaseError('finite dev release code differs')
    expected = {e['task_id']: e for e in declared_entries() if int(e['task_id'].split('_')[2]) in (36, 2, 34)}
    declared = read(root/'dev/declared_tasks.json')
    if len(declared) != 3 or {e['task_id']: e for e in declared} != expected:
        raise ReleaseError('finite dev declared task identities differ')
    auto = read(root/'dev/automation/acceptance.json')
    if (auto.get('passed') is not True or auto.get('code_hash') != code or
        auto.get('bank_digest_before') != manifest['knowledge_digest'] or
        auto.get('bank_digest_after') != manifest['knowledge_digest']):
        raise ReleaseError('automatic acceptance identity mismatch')
    trace_path = root/'dev/automation/traces'/f"{auto['trace_id']}.json"
    if sha(trace_path) != auto.get('trace_sha256'):
        raise ReleaseError('automatic acceptance Trace mismatch')
    rows = [check_episode(root/'dev'/profile/task, code_hash=code, bank_digest=manifest['knowledge_digest'],
        entry=entry, profile=profile) for task, entry in expected.items() for profile in ('current', 'lean')]
    comparison = read(root/'dev/comparison/summary.json')
    if len(comparison.get('pairs', [])) != 3 or {p['task_id'] for p in comparison['pairs']} != set(expected):
        raise ReleaseError('paired comparison incomplete')
    for pair in comparison['pairs']:
        for profile in ('current', 'lean'):
            if pair[profile] != read(root/'dev'/profile/pair['task_id']/'summary.json'):
                raise ReleaseError('paired comparison does not describe actual episodes')
    result = {'passed': True, 'code_hash': code, 'bank_digest': manifest['knowledge_digest'],
              'episodes': rows, 'automatic_acceptance_sha256': sha(root/'dev/automation/acceptance.json')}
    if write:
        atomic_write_json(root/'dev/final_acceptance.json', result)
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, required=True)
    args = p.parse_args()
    result = verify_dev(args.release_root, write=True)
    print(json.dumps({'passed': result['passed'], 'episodes': len(result['episodes'])}))
