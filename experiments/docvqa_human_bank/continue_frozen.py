"""Run one explicitly prepared continuation and merge disjoint completed results."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = read(args.job)
    root = Path(job['root'])
    sys.path.insert(0, str(Path(job['cwd']) / 'src'))
    from atomic_skillgraph.empirical.checkpoint import save_json
    from atomic_skillgraph.experiments.formal_log import tree_identity
    manifest = read(root / 'continuation_manifest.json')
    assert hashlib.sha256(Path(args.job).read_bytes()).hexdigest() == manifest['job_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=job['cwd'], text=True).strip() == manifest['source_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=job['cwd'], text=True).strip()
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / 'evaluation/result.json').exists():
            save_json(root / 'supervisor.json', {'status': 'running', 'source_commit': manifest['source_commit']})
            process = subprocess.run(job['command'], cwd=job['cwd'])
            if process.returncode:
                save_json(root / 'supervisor.json', {'status': 'stopped', 'exit_code': process.returncode})
                raise SystemExit(process.returncode)
        completed = {}
        for directory in [*[Path(p) for p in manifest['parent_runs']], root]:
            assert tree_identity(directory / 'evaluation/bank')['sha256'] == manifest['bank_hash']
            for line in (directory / 'evaluation/episodes.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row.get('task_infrastructure_error'):
                    continue
                assert row['task_id'] not in completed, 'A completed task was run twice'
                completed[row['task_id']] = row
        assert list(completed) == read(manifest['full_selection'])['task_ids']
        ledger = read(root / 'budget.json')
        attempts = list(ledger['attempts'].values())
        from atomic_skillgraph.empirical.budget_governor import billing_blocked
        assert not billing_blocked(ledger)
        known = sum(a.get('accounted_tokens', 0) for a in attempts)
        unknown = [a for a in attempts if a['status'] != 'metered']
        reserved = sum(a['reserved_tokens'] for a in unknown)
        assert known + reserved <= ledger['limits']['token_limit'] and len(attempts) <= ledger['limits']['request_limit']
        result = {'complete': True, 'tasks': len(completed), 'correct': sum(bool(r['success']) for r in completed.values()),
            'anls': sum(r['official_score'] for r in completed.values()) / len(completed),
            'empty': sum(bool(r['empty_answer']) for r in completed.values()),
            'known_tokens': known, 'total_tokens': None if unknown else known,
            'unknown_reserved_tokens': reserved, 'unknown_billing_attempts': len(unknown), 'http': len(attempts),
            'bank_hash': manifest['bank_hash'], 'bank_unchanged': True,
            'parent_runs': manifest['parent_runs'], 'continuation_run': str(root),
            'source_commit': manifest['source_commit'], 'all_completed_tasks_preserved': True}
        save_json(root / 'completion_report.json', result)
        save_json(root / 'supervisor.json', {'status': 'completed'})
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
