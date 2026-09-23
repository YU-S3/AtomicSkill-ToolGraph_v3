"""Cooperative operator stop at committed task/rep boundaries, not a task error."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
from atomic_skillgraph.core.serialization import atomic_write_json


class ReleaseStopRequested(Exception):
    pass


def check_stop(root, seed, output=None):
    root = Path(root).resolve()
    requests = [p for p in (root/'control/stop_all.json', root/f'control/stop_seed{seed}.json') if p.exists()]
    if not requests:
        return
    record = {'state': 'paused', 'seed': seed, 'utc': datetime.now(timezone.utc).isoformat(),
              'control_requests': [str(p) for p in requests], 'boundary': 'committed_task_or_rep',
              'resume': 'Clear the control request explicitly, then use same-version --resume.'}
    if output is not None:
        output = Path(output)
        atomic_write_json(output/'operator_stop.json', record)
        progress = output/'progress.json'
        if progress.exists():
            prior = json.loads(progress.read_text())
            atomic_write_json(progress, {**prior, 'state': 'paused', 'current_task': None, 'updated_at': record['utc']})
    atomic_write_json(root/f'control/seed{seed}_stopped.json', record)
    raise ReleaseStopRequested(f'seed{seed} paused at committed boundary')


def check_config_stop(config, output):
    release = config.get('bank_release', {})
    if not release.get('release_manifest'):
        return
    root = Path(release['release_manifest']).resolve().parents[2]
    check_stop(root, release['source_seed'], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('seed', choices=('all', '42', '43', '44'))
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/'evaluation_plan.json').is_file():
        raise ValueError('not a released evaluation root')
    # Old running interpreters do not support this protocol.
    from atomic_skillgraph.agents.native_call_contract import NATIVE_CALL_CONTRACT_VERSION
    audit = json.loads((root/'seed42/frozen/native_call_contract.json').read_text())
    if audit['version'] != NATIVE_CALL_CONTRACT_VERSION:
        raise ValueError('release does not support this stop interface')
    name = 'stop_all.json' if args.seed == 'all' else f'stop_seed{args.seed}.json'
    atomic_write_json(root/'control'/name, {'utc': datetime.now(timezone.utc).isoformat(), 'operator_requested': True})
    print(f'Stop requested at next committed task boundary: {root / "control" / name}')


if __name__ == '__main__':
    main()
