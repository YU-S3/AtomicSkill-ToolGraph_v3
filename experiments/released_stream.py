"""Locked, non-overwriting execution of a previously validated release matrix."""
import argparse
import json
from pathlib import Path

from atomic_skillgraph.deployment.release_protocol import ReleaseError
from .protocol import hash_config


def read(path):
    return json.loads(Path(path).read_text())


def validate_plan(root):
    root = Path(root).resolve()
    plan = read(root/'evaluation_plan.json')
    if len(plan['runs']) != 5 or {(r['seed'], r['rep']) for r in plan['runs']} != {
            (42, 1), (42, 2), (42, 3), (43, 1), (44, 1)}:
        raise ReleaseError('expected the fixed five-run release matrix')
    for row in plan['runs']:
        path = root/'configs'/f"seed{row['seed']}_rep{row['rep']:02}.json"
        config = read(path)
        output = root/'eval'/f"seed{row['seed']}"/f"rep{row['rep']:02}"
        if (Path(row['config']).resolve() != path or hash_config(config) != row['config_hash']
                or Path(row['output']).resolve() != output
                or Path(config['experiment']['output_dir']).resolve() != output
                or config['deployment']['presentation_profile'] != plan['profile']
                or Path(config['bank_release']['release_manifest']).resolve()
                   != root/f"seed{row['seed']}"/'frozen/release_manifest.json'):
            raise ReleaseError('release matrix config/output identity changed')
    return sorted(plan['runs'], key=lambda r: (r['seed'], r['rep']))


def run_state(row):
    output = Path(row['output'])
    if not output.exists():
        return 'fresh'
    if not (output/'progress.json').is_file() or read(output/'progress.json').get('state') != 'completed':
        return 'resume_required'
    summary = read(output/'summary.json')
    if summary['tasks'] != 134 or summary['config_hash'] != row['config_hash']:
        raise ReleaseError('completed release summary does not match plan')
    return 'completed'


def verify_gate(root):
    preparation = read(root/'seed42/prepare_manifest.json')
    if preparation['config'].get('runtime', {}).get('preparation_coverage_version'):
        from .release4_coverage import verify_coverage
        verify_coverage(root)
    else:
        from .released_dev_checks import verify_dev
        verify_dev(root)


def run_stream(root, seed, *, resume=False, first_only=False):
    import fcntl
    from .run_v3_released_frozen import run
    root = Path(root).resolve()
    with (root/f'seed{seed}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError(f'seed{seed} already running') from exc
        for row in validate_plan(root):
            if row['seed'] != seed or (first_only and row['rep'] != 1):
                continue
            state = run_state(row)
            if state == 'completed':
                print(f"seed{seed} rep{row['rep']:02} already complete; preserved", flush=True)
                continue
            if state == 'resume_required' and not resume:
                raise ReleaseError(f"{row['output']} is unfinished; use explicit --resume")
            print(f"seed{seed} rep{row['rep']:02}: {state}", flush=True)
            run(Path(row['config']), resume=state == 'resume_required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', required=True, type=Path)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--first-only', action='store_true')
    args = parser.parse_args()
    validate_plan(args.release_root)
    verify_gate(args.release_root)
    if args.seed is not None:
        run_stream(args.release_root, args.seed, resume=args.resume, first_only=args.first_only)


if __name__ == '__main__':
    main()
