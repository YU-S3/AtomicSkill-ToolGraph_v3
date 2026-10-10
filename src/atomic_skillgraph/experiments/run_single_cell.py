"""Run one canonical model/benchmark/seed through the existing formal campaign."""
import argparse
import json
from pathlib import Path
import yaml
from .canonical_manifest import BENCHMARKS, verify
from .run_empirical import load_env, write_json
from .run_formal import campaign, configured_models, model_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/main_experiment_v1.yaml')
    parser.add_argument('--benchmark', required=True, choices=BENCHMARKS)
    parser.add_argument('--seed', required=True, type=int, choices=(42, 43, 44))
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--corpus-root')
    parser.add_argument('--env-file')
    parser.add_argument('--model-key')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-val', action='store_true')
    parser.add_argument('--max-new-tasks', type=int)
    parser.add_argument('--stop-at-task-id')
    args = parser.parse_args(argv)
    if args.env_file: load_env(args.env_file)
    spec = yaml.safe_load(Path(args.config).read_text())
    if spec['split_seed'] != 42 or spec['run_seeds'] != [42, 43, 44] or tuple(spec['benchmarks']) != BENCHMARKS:
        raise ValueError('Use the fixed main-experiment specification')
    verify(spec['authority'])
    if args.benchmark == 'officeqa' and not args.corpus_root: parser.error('OfficeQA requires --corpus-root')
    models = configured_models(json.loads(Path(spec['model_lock']).read_text()))
    if args.model_key:
        models = [m for m in models if args.model_key in (m.get('model_id'), m.get('display_name'))]
    if len(models) != 1: parser.error('Select exactly one configured model with --model-key')
    model = models[0]
    if args.benchmark == 'docvqa' and 'image' not in model['input_modalities']:
        root = Path(args.output)
        if root.exists() and any(root.iterdir()): raise ValueError('Unsupported run needs an empty output directory')
        write_json(root / 'completion.json', {'status': 'unsupported', 'benchmark': args.benchmark,
            'run_seed': args.seed, 'model_id': model['model_id'], 'score': None, 'reason': 'Locked model has no image capability'})
        return
    base = model_settings(yaml.safe_load(Path(spec['base_config']).read_text()), model)
    profiles = json.loads(Path(spec['benchmark_profiles']).read_text())['profiles']
    if spec.get('budget_allocation'):
        from .seed42_budget import allocation
        frozen=json.loads(Path(spec['budget_allocation']).read_text())
        expected=allocation(verify(spec['authority']),profiles,base['llm'].get('max_retries',4))
        if frozen!=expected:raise ValueError('Frozen single-seed budget allocation changed')
        if args.seed!=42 or args.benchmark not in frozen['cells']:
            raise ValueError('This allocation covers only the four authorized seed42 cells')
        base['budget']=frozen['cells'][args.benchmark]
    return campaign(base, profiles, args.benchmark, args.seed, args.output, args.datasets, spec['authority'],
                    args.corpus_root, resume=args.resume, stop_after_val=args.stop_after_val,
                    max_new_tasks=args.max_new_tasks, stop_at_task_id=args.stop_at_task_id)


if __name__ == '__main__': main()
