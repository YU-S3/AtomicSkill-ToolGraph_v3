"""Independent human Bank: fixed Val2 interface check, then canonical Test201."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(ROOT / 'integration'))

from human_view import HumanDocVQAView, METHOD, image_result_bridge
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.experiments.canonical_manifest import sha256, verify
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity, utc
from atomic_skillgraph.experiments.run_empirical import code_identity, load_env, run, write_json
from atomic_skillgraph.experiments.run_formal import model_settings, resolved_config


def prepare(root, datasets, expected_sha):
    code = code_identity()
    if code['git_sha'] != expected_sha or code['tracked_dirty']:
        raise ValueError('Use the exact reviewed clean source')
    authority = REPO / 'data/main_experiment_v1'
    canonical = verify(authority)
    material = json.loads((datasets / 'materialization.json').read_text())
    if material['authority_sha256'] != sha256(authority / 'manifest.json'):
        raise ValueError('Public authority differs')
    spec = yaml.safe_load((REPO / 'configs/docvqa_deepseek_v41.yaml').read_text())
    model = json.loads((REPO / spec['model_lock']).read_text())['formal_models'][0]
    base = model_settings(yaml.safe_load((REPO / spec['base_config']).read_text()), model)
    profile = json.loads((REPO / spec['benchmark_profiles']).read_text())['profiles']['docvqa']
    # Same 96K per evaluation task, including two separate interface-check tasks.
    base['budget'].update(token_limit=203 * 96000, request_limit=203 * 9 * (base['llm']['max_retries'] + 1))
    configs, tasks, metadata = {}, {}, {}
    image_hashes = {}
    for phase, split, count in [('validation', 'val', 2), ('test', 'test', 201)]:
        public_path = datasets / 'docvqa' / (split + '.json')
        gold_path = datasets / 'docvqa' / ('evaluator_records_' + split + '.json')
        for path in (public_path, gold_path):
            if sha256(path) != material['files_sha256'][path.relative_to(datasets).as_posix()]:
                raise ValueError('Materialized file identity differs: ' + str(path))
        rows = json.loads(public_path.read_text())['tasks']
        public = json.loads((authority / 'docvqa' / (split + '.json')).read_text())['tasks']
        if [r['task_id'] for r in rows] != [r['task_id'] for r in public]:
            raise ValueError('Canonical task members/order differ')
        if len(rows) != {'val': 22, 'test': 201}[split]:
            raise ValueError('Canonical count differs')
        tasks[phase] = [PublicTask(**r) for r in rows[:count]]
        metadata.update({r['task_id']: r['task_metadata'] for r in public})
        for task in tasks[phase]:
            if set(task.inputs) != {'images'} or len(task.inputs['images']) != 1:
                raise ValueError('Only the unchanged original image is allowed')
            for image in task.inputs['images']:
                actual = sha256(image)
                if actual != material['external_files_sha256'][image]:
                    raise ValueError('Original image changed')
                image_hashes[image] = actual
        c = resolved_config(base, profile, 'docvqa', 42, split, root, datasets, authority, None)
        c['data_dir'] = str(root / 'bank')
        c['program_worker']['record_execution_receipt'] = True
        c['experiment'].update(output_dir=str(root / phase),
            docvqa_human_reference={'method': METHOD, 'programs': 'auto', 'workflow_execution': False})
        configs[phase] = c
    config_path = root / 'import_config.json'
    if not (root / 'bank').exists():
        write_json(config_path, configs['test'])
        sys.path.insert(0, str(ROOT / 'tools'))
        from import_native import prepare as import_bank
        import_bank(REPO, config_path, root / 'bank')
    manual = json.loads((root / 'bank/manual_manifest.json').read_text())
    with_bank = Bank(root / 'bank', readonly=True, seed=42, environment=configs['test']['program_environment'])
    try:
        view = HumanDocVQAView(with_bank, manual)
        if manual['source_bundle_hash'] != digest(json.loads((ROOT / 'bank/bank.json').read_text())):
            raise ValueError('Imported Bank source changed')
        view_hash = view.view_hash
    finally:
        with_bank.close()
    identity = {'method': 'human_authored_reference_not_learned_ours', 'method_version': METHOD,
        'benchmark': 'docvqa', 'run_seed': 42, 'split_seed': 42,
        'source_commit': code['git_sha'], 'source_sha256': code['source_sha256'],
        'model_name': model['display_name'], 'provider_model_id': model['model_id'],
        'generation_seed_status': 'unsupported', 'generation_seed': None,
        'split_manifest_hash': sha256(authority / 'manifest.json'),
        'split_hashes': {k: v['sha256'] for k, v in canonical['benchmarks']['docvqa']['splits'].items()},
        'public_materialization_sha256': sha256(datasets / 'materialization.json'),
        'bank_digest': manual['bank_digest'], 'bank_view_hash': view_hash,
        'bank_origin': 'external human authored', 'workflow_execution': False,
        'validation_policy': 'fixed canonical first two Val tasks; interface gate only, no score selection',
        'original_images_sha256': digest(image_hashes),
        'source_bundle': json.loads((ROOT / 'source_manifest.json').read_text())}
    plan = {'identity': identity, 'configs': configs,
        'task_ids': {k: [t.task_id for t in v] for k, v in tasks.items()}, 'limits': base['budget']}
    if (root / 'plan.json').exists() and json.loads((root / 'plan.json').read_text()) != plan:
        raise ValueError('Frozen experiment plan changed')
    write_json(root / 'plan.json', plan)
    return plan, tasks, metadata, manual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run', 'resume'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', default='/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1')
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, tasks, metadata, manual = prepare(root, Path(args.datasets).resolve(), args.expected_sha)
        if args.action == 'check':
            print(json.dumps({'ready': True, 'model_http': 0, 'counts': {k: len(v) for k, v in tasks.items()},
                              'bank': manual['counts'], 'limits': plan['limits']}))
            return
        if args.env_file:
            load_env(args.env_file)
        resume = args.action == 'resume'
        if resume and not (root / 'run_manifest.json').exists():
            raise ValueError('No started run to resume')
        log = FormalLog(root, plan['identity'], plan['configs'], resume=resume)
        governor = BudgetGovernor(root / 'budget.json', **{k: plan['limits'][k] for k in
            ('token_limit', 'request_limit', 'finish_reserve')})
        frozen = log.freeze(root / 'bank', newly_created=not (root / 'final_frozen_manifest.json').exists())
        try:
            for phase in ('validation', 'test'):
                if phase == 'test' and args.validation_only:
                    log.finish('validation_completed'); return
                existing = root / phase / 'summary.json'
                if existing.exists() and json.loads(existing.read_text()).get('complete'):
                    continue
                before = tree_identity(root / 'bank')['sha256']
                if before != frozen['final_artifact_hash']:
                    raise ValueError('Readonly human Bank changed')
                if phase == 'test':
                    frozen.update(hash_before_test=before, test_start_time=utc(), test_started_after_freeze=True)
                    write_json(root / 'final_frozen_manifest.json', frozen)
                with image_result_bridge():
                    summary = run(plan['configs'][phase], tasks[phase], root / phase,
                        resume=resume and (root / phase / 'execution_manifest.json').exists(), readonly=True,
                        formal_log=log, task_metadata=metadata,
                        bank_view_factory=lambda b: HumanDocVQAView(b, manual), budget_governor=governor,
                        freeze_on_complete=False)
                after = tree_identity(root / 'bank')['sha256']
                if before != after: raise RuntimeError('Human Bank mutated')
                if not summary['complete'] or summary['unknown_billing_attempts']:
                    raise RuntimeError('Incomplete or unaccounted run; inspect before continuation')
                report = {'phase': phase, 'tasks': summary['tasks'], 'hard_correct': summary['successes'],
                    'mean_anls': sum(t['score']['soft'] for t in summary['cases']) / summary['tasks'],
                    'tokens': summary['known_total_tokens'], 'program_invocations': summary['program_invocations'],
                    'bank_hash_before': before, 'bank_hash_after': after,
                    'workflow_invocations': 0, 'origin': 'human_authored_reference_not_learned_ours'}
                write_json(root / (phase + '_result.json'), report)
                print(json.dumps(report), flush=True)
                if phase == 'test':
                    frozen['artifact_hash_after_test'] = after
                    write_json(root / 'final_frozen_manifest.json', frozen)
            log.finish('completed')
            write_json(root / 'completion.json', {'status': 'completed', 'time': utc(),
                'method': METHOD, 'test': json.loads((root / 'test_result.json').read_text())})
        except BaseException:
            log.finish('stopped'); raise


if __name__ == '__main__':
    main()
