"""Canonical Ours campaigns: independent Train, frozen Val and frozen Test."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from ..empirical.contracts import PublicTask, digest
from ..empirical.system import validate_config
from ..harness.registry import create_simple_harness
from .canonical_manifest import ADAPTER_NAMES, BENCHMARKS, ordered_train, sha256, verify
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, resolve_alfworld_tasks, run, write_json
from skillcompiler_bench_contracts import source_identity
from ..empirical import IMPLEMENTATION_REVISION, POLICY_DEFAULTS


def resolved_config(base, profile, benchmark, seed, split, root, datasets, authority, corpus_root):
    config = deepcopy(base)
    phase = root / split
    config['data_dir'] = str(root / 'train' / ('bank' if split == 'train' else 'frozen_bank'))
    config['manifest'] = str(authority / benchmark / (split + '.json'))
    config['benchmark_profile'] = benchmark + '.main_experiment_v1'
    config['experiment'].update(seed=seed, benchmark=benchmark, output_dir=str(phase),
                                runtime_mode='online' if split == 'train' else 'frozen')
    config['harness'] = {'adapter': ADAPTER_NAMES.get(benchmark, benchmark),
        'evaluator_records': str(datasets / benchmark / ('evaluator_records_' + split + '.json'))}
    if benchmark == 'officeqa':
        config['harness']['corpus_root'] = str(corpus_root)
    if benchmark == 'alfworld':
        config['harness'] = {**base['harness'], 'adapter': 'alfworld_v3',
            'split': {'train': 'train', 'val': 'eval_in_distribution', 'test': 'eval_out_of_distribution'}[split],
            'max_steps': profile['environment_steps']}
    config['runtime']['global_action_budget'] = max(1, profile['native_calls'])
    config['runtime'].pop('environment_step_budget', None)
    if profile.get('environment_steps'):
        config['runtime']['environment_step_budget'] = profile['environment_steps']
    config['program_worker'].update(wall_timeout_seconds=profile['wall_seconds'], memory_limit_mb=profile['memory_mb'],
                                    max_tool_calls_per_invocation=profile['program_call_limit'])
    config['program_environment'].update(adapter_abi=profile['adapter_abi'], image_digest=profile['image_digest'])
    return validate_config(config)


def campaign(base, profiles, benchmark, seed, output, datasets, authority, corpus_root=None, *, resume=False, stop_after_val=False):
    root = Path(output).resolve()
    authority, datasets = Path(authority).resolve(), Path(datasets).resolve()
    manifest = verify(authority)
    if benchmark not in BENCHMARKS or seed not in (42, 43, 44): raise ValueError('Unknown benchmark or run seed')
    if benchmark == 'officeqa' and not corpus_root: raise ValueError('OfficeQA requires --corpus-root')
    materialized = json.loads((datasets / 'materialization.json').read_text())
    if materialized['authority_sha256'] != sha256(authority / 'manifest.json'):
        raise ValueError('Materialization belongs to a different public authority')
    for name, expected in materialized['files_sha256'].items():
        if sha256(datasets / name) != expected:
            raise ValueError('Materialized resource changed: ' + name)
    for name, expected in materialized.get('external_files_sha256', {}).items():
        if sha256(name) != expected: raise ValueError('Public input file changed: ' + name)
    from skillcompiler_bench_contracts.livemath import NORMALIZATION_VERSION
    if materialized.get('livemath_normalization_version') != NORMALIZATION_VERSION:
        raise ValueError('CF4 requires new public materialization; old checkpoints cannot be resumed')
    if sha256(datasets / 'livemath_integrity.json') != materialized['integrity_sha256']:
        raise ValueError('LiveMath integrity record changed')
    profile = profiles[ADAPTER_NAMES.get(benchmark, benchmark)]
    code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']:
        raise ValueError('Formal campaigns require committed clean source')
    identity = {'method': 'ours', 'model_name': base['llm'].get('display_name', base['llm']['model']),
        'provider_model_id': base['llm']['model'], 'benchmark': benchmark, 'run_seed': seed, 'split_seed': 42,
        'source_commit': code['git_sha'], 'source_sha256': code['source_sha256'],
        'materialization_sha256': sha256(datasets / 'materialization.json'),
        'benchmark_contracts_sha256': source_identity(),
        'public_materialization_sha256': sha256(datasets / 'materialization.json'),
        'implementation_revision': IMPLEMENTATION_REVISION,
        'model_view_version': POLICY_DEFAULTS['runtime']['model_view_version'],
        'split_manifest_hash': sha256(authority / 'manifest.json'),
        'split_hashes': {k: v['sha256'] for k, v in manifest['benchmarks'][benchmark]['splits'].items()},
        'generation_seed_status': 'supported' if base['llm'].get('generation_seed_supported') else 'unsupported',
        'generation_seed': seed if base['llm'].get('generation_seed_supported') else None,
        'initial_bank': 'empty independent Bank', 'local_rng_seed': seed,
        'candidate_sampling': 'deterministic existing algorithm; no random sampler'}
    configurations = {split: resolved_config(base, profile, benchmark, seed, split, root, datasets, authority, corpus_root)
                      for split in ('train', 'val', 'test')}
    if stop_after_val and (root / 'test/summary.json').exists():
        raise ValueError('--stop-after-val cannot relabel an existing Test run')
    log = FormalLog(root, identity, configurations, resume=resume)
    summaries, offset = {}, 0
    try:
        frozen_manifest = root / 'final_frozen_manifest.json'
        if frozen_manifest.exists():
            if tree_identity(root / 'train/frozen_bank')['sha256'] != json.loads(frozen_manifest.read_text())['final_artifact_hash']:
                raise RuntimeError('Frozen snapshot changed before resume')
        for split, settings in configurations.items():
            phase = root / split
            prior = phase / 'summary.json'
            if resume and prior.exists() and json.loads(prior.read_text()).get('complete'):
                summary = json.loads(prior.read_text())
                summaries[split] = {k: summary[k] for k in ('complete', 'tasks', 'successes', 'known_total_tokens', 'unknown_billing_attempts', 'knowledge_digest')}
                offset += summary['tasks']
                if split == 'val' and stop_after_val: break
                continue
            public = json.loads((authority / benchmark / (split + '.json')).read_text())['tasks']
            metadata = {r['task_id']: r['task_metadata'] for r in public}
            adapter = create_simple_harness(settings)
            try:
                rows = json.loads((datasets / benchmark / (split + '.json')).read_text())['tasks']
                if benchmark == 'alfworld':
                    tasks = resolve_alfworld_tasks(adapter, rows, canonical_split=split,
                        mapping_path=root / split / 'task_identity_resolution.json')
                else:
                    tasks = [PublicTask(**r) for r in rows]
                if [t.task_id for t in tasks] != [r['task_id'] for r in public]:
                    raise ValueError('Ours materialization changed canonical membership/order')
                if split == 'train':
                    tasks = ordered_train(tasks, seed)
                else:
                    frozen_path = root / 'train/frozen_bank'
                    frozen = json.loads((root / 'final_frozen_manifest.json').read_text())
                    if tree_identity(frozen_path)['sha256'] != frozen['final_artifact_hash']:
                        raise RuntimeError('Frozen snapshot changed before evaluation')
                    if split == 'test':
                        frozen.update(test_started_after_freeze=True, test_start_time=utc(),
                                      hash_before_test=tree_identity(frozen_path)['sha256'])
                        write_json(root / 'final_frozen_manifest.json', frozen)
                phase = root / split
                summary = run(settings, tasks, phase, resume=resume and (phase / 'execution_manifest.json').exists(),
                    readonly=split != 'train', adapter=adapter, adapter_factory=lambda settings=settings: create_simple_harness(settings),
                    formal_log=log, task_metadata=metadata, order_offset=offset)
                summaries[split] = {k: summary[k] for k in ('complete', 'tasks', 'successes', 'known_total_tokens', 'unknown_billing_attempts', 'knowledge_digest')}
                if split == 'val':
                    log.emit('validation_events', {'event_id': 'frozen-validation', 'validation_event_id': 'frozen-validation',
                        'candidate_or_snapshot_id': frozen['final_artifact_id'], 'validation_size': len(tasks),
                        'validation_task_ids': [t.task_id for t in tasks], 'validation_scores': [r['score']['raw_score'] for r in summary['cases']],
                        'validation_mean_score_raw': None, 'accepted_or_selected': None,
                        'selection_reason': None, 'purpose': 'read-only evaluation; no learning selection'})
                if split == 'test':
                    frozen['artifact_hash_after_test'] = tree_identity(frozen_path)['sha256']
                    write_json(root / 'final_frozen_manifest.json', frozen)
                    if frozen['artifact_hash_after_test'] != frozen['final_artifact_hash']:
                        raise RuntimeError('Frozen snapshot changed during Test')
                offset += len(tasks)
            finally:
                adapter.close()
            if split == 'val' and stop_after_val: break
        status = 'awaiting_test' if stop_after_val else 'completed'
        log.finish(status)
        write_json(root / 'completion.json', {'status': status, 'completed_splits': list(summaries), 'runs': summaries})
        return summaries
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            status = 'interrupted'
        else:
            status = 'infrastructure_failed' if log.task_error(exc) else 'execution_failed'
        log.finish(status)
        write_json(root / 'completion.json', {'status': status, 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def configured_models(lock):
    configured = [m for m in lock['formal_models'] if all(m.get(k) for k in ('model_id', 'endpoint', 'api_key_env', 'input_modalities'))]
    return configured or [lock['current_test']]


def model_settings(base, model):
    settings = deepcopy(base)
    settings['llm'].update(model=model['model_id'], display_name=model.get('display_name', model['model_id']),
        base_url=model['endpoint'], api_key_env=model['api_key_env'], dialect=model.get('dialect', base['llm']['dialect']),
        input_modalities=model['input_modalities'], token_limit_field=model.get('token_limit_field', 'max_tokens'),
        generation_seed_supported=model.get('generation_seed_supported', False))
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/main_experiment_v1.yaml')
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--corpus-root', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)
    spec = yaml.safe_load(Path(args.config).read_text())
    if spec['split_seed'] != 42 or spec['run_seeds'] != [42, 43, 44] or tuple(spec['benchmarks']) != BENCHMARKS:
        raise ValueError('Formal matrix must cover all six benchmarks and the three fixed run seeds')
    base = yaml.safe_load(Path(spec['base_config']).read_text())
    profiles = json.loads(Path(spec['benchmark_profiles']).read_text())['profiles']
    lock = json.loads(Path(spec['model_lock']).read_text())
    configured = [m for m in lock['formal_models'] if all(m.get(k) for k in ('model_id', 'endpoint', 'api_key_env', 'input_modalities'))]
    models = configured_models(lock)
    output = Path(args.output).resolve()
    matrix = {'schema': 'ours.formal-matrix.v1', 'authority_sha256': sha256(Path(spec['authority']) / 'manifest.json'),
        'configured_models': models, 'unconfigured_models': [m for m in lock['formal_models'] if m not in configured], 'cells': {}}
    path = output / 'matrix.json'
    if path.exists():
        if not args.resume:
            raise ValueError('Existing matrix requires --resume')
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != 'cells'} != {k: v for k, v in matrix.items() if k != 'cells'}:
            raise ValueError('Matrix identity changed')
        matrix = old
    for model in models:
        if Path(model['model_id']).name != model['model_id']:
            raise ValueError('Model ID must be a safe single directory name')
        for benchmark in spec['benchmarks']:
            for seed in spec['run_seeds']:
                cell = model['model_id'] + '/' + benchmark + '/seed' + str(seed)
                unsupported = benchmark == 'docvqa' and 'image' not in model['input_modalities']
                matrix['cells'].setdefault(cell, {'status': 'unsupported' if unsupported else 'queued',
                    'reason': 'Locked model has no image capability' if unsupported else None, 'score': None, 'path': str(output / cell)})
    write_json(path, matrix)
    for model in models:
        settings = model_settings(base, model)
        for benchmark in spec['benchmarks']:
            for seed in spec['run_seeds']:
                cell = model['model_id'] + '/' + benchmark + '/seed' + str(seed)
                if matrix['cells'][cell]['status'] in {'completed', 'unsupported', 'execution_failed'}:
                    continue
                matrix['cells'][cell].update(status='running', started_at=utc())
                write_json(path, matrix)
                try:
                    result = campaign(settings, profiles, benchmark, seed, output / cell, args.datasets, spec['authority'],
                                      args.corpus_root, resume=args.resume and (output / cell / 'run_manifest.json').exists())
                    matrix['cells'][cell].update(status='completed', runs=result, ended_at=utc())
                except Exception as exc:
                    completion = output / cell / 'completion.json'
                    status = json.loads(completion.read_text())['status'] if completion.exists() else 'infrastructure_failed'
                    matrix['cells'][cell].update(status=status, error_type=type(exc).__name__, error=str(exc), ended_at=utc())
                write_json(path, matrix)


if __name__ == '__main__':
    main()
