"""Fixed Val22 control using an unchanged previous checkout and readonly Bank."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'source-sha', 'config', 'bank', 'output', 'env-file'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    repo, root, bank = (Path(x).resolve() for x in (args.repo_root, args.output, args.bank))
    sys.path.insert(0, str(repo / 'src'))
    sys.path.insert(0, str(repo / 'experiments/docvqa_human_bank/integration'))
    from human_view import HumanDocVQAView, METHOD, image_result_bridge
    from atomic_skillgraph.empirical.contracts import PublicTask, digest
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity
    from atomic_skillgraph.experiments.run_empirical import code_identity, load_env, run, write_json
    from atomic_skillgraph.experiments.canonical_manifest import verify, sha256
    code = code_identity()
    if code['git_sha'] != args.source_sha or code['tracked_dirty'] or METHOD != 'docvqa.human-ideal.programmatic.v1.1':
        raise ValueError('Previous source/control identity differs')
    config = deepcopy(json.loads(Path(args.config).read_text()))
    gold = Path(config['harness']['evaluator_records'])
    if gold.name != 'evaluator_records_val.json': raise ValueError('Val only')
    data = json.loads((gold.parent / 'val.json').read_text())
    authority = repo / 'data/main_experiment_v1'
    verify(authority)
    canonical = json.loads((authority / 'docvqa/val.json').read_text())['tasks']
    if data['split_sha256'] != sha256(authority / 'docvqa/val.json') or [x['task_id'] for x in data['tasks']] != [x['task_id'] for x in canonical]:
        raise ValueError('Canonical Val identity differs')
    tasks = [PublicTask(**x) for x in data['tasks']]
    assert len(tasks) == 22
    config['data_dir'] = str(bank)
    config['experiment']['output_dir'] = str(root / 'val')
    config['budget'].update(token_limit=22 * 96000, request_limit=22 * 9 * 5)
    manifest = json.loads((bank / 'manual_manifest.json').read_text())
    identity = {'method': 'human_bank_v11_fixed_val_control', 'benchmark': 'docvqa',
        'run_seed': 42, 'source': code, 'bank_digest': manifest['bank_digest'],
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'val_split_hash': data['split_sha256'], 'task_order': [x.task_id for x in tasks]}
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        load_env(args.env_file)
        log = FormalLog(root, identity, config, resume=args.resume)
        governor = BudgetGovernor(root / 'budget.json', **{k: config['budget'][k] for k in
            ('token_limit', 'request_limit', 'finish_reserve')})
        before = tree_identity(bank)['sha256']
        try:
            with image_result_bridge():
                summary = run(config, tasks, root / 'val', readonly=True, resume=args.resume,
                    bank_view_factory=lambda b: HumanDocVQAView(b, manifest), formal_log=log,
                    task_metadata={x['task_id']: x['task_metadata'] for x in canonical},
                    budget_governor=governor, freeze_on_complete=False)
            after = tree_identity(bank)['sha256']
            if before != after: raise RuntimeError('Previous Bank changed')
            result = {'complete': summary['complete'], 'tasks': summary['tasks'], 'correct': summary['successes'],
                'anls': sum(t['score']['soft'] for t in summary['cases']) / len(tasks),
                'tokens': summary['known_total_tokens'], 'unknown_billing': summary['unknown_billing_attempts'],
                'empty': sum(not (t['execution'].get('prediction') or '').strip() for t in summary['cases']),
                'bank_before': before, 'bank_after': after}
            write_json(root / 'result.json', result)
            log.finish('completed' if summary['complete'] else 'stopped')
            print(json.dumps(result), flush=True)
        except BaseException:
            log.finish('stopped'); raise


if __name__ == '__main__':
    main()
