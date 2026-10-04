"""One fixed 12 Train + 6 valid_seen diagnostic, then a truthful mechanism release."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.harness.registry import create_simple_harness
from .run_empirical import code_identity, load_env, resolve_alfworld_tasks, run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/alfworld_empirical_seed42.yaml')
    parser.add_argument('--materials', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)
    acceptance = json.loads(Path(args.acceptance).read_text())
    if not acceptance.get('passed'):
        raise ValueError('Genuine generated-program acceptance has not passed; do not start pilot')
    selection = json.loads((Path(args.materials) / 'selection.json').read_text())
    if len(selection['train']) != 12 or len(selection['valid_seen']) != 6:
        raise ValueError('Pilot must use the previous fixed 12 + 6 selection')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_config = yaml.safe_load(Path(args.config).read_text())
    summaries = {}
    for split in ('train', 'valid_seen'):
        config = deepcopy(source_config)
        config['harness']['split'] = 'train' if split == 'train' else 'eval_in_distribution'
        config['experiment']['runtime_mode'] = 'online' if split == 'train' else 'frozen'
        config['experiment']['output_dir'] = str(output / split)
        config['data_dir'] = str(output / 'train' / ('bank' if split == 'train' else 'frozen_bank'))
        adapter = create_simple_harness(config)
        tasks = resolve_alfworld_tasks(adapter, selection[split])
        summary = run(config, tasks, output / split, resume=args.resume,
            readonly=split != 'train', adapter=adapter, adapter_factory=lambda: create_simple_harness(config))
        summaries[split] = summary
    invocations = [a for summary in summaries.values() for trace in summary['cases']
                   for a in trace['execution'].get('attempts', [])]
    reused = [a for a in invocations if a['status'] == 'ok' and a['calls'] >= 2]
    release = {'schema': 'skillcompiler.mechanism-release.v1', 'code': code_identity(),
        'profile': source_config['mechanism_profile'], 'selection_sha256': digest(selection),
        'acceptance': str(Path(args.acceptance).resolve()), 'acceptance_passed': True,
        'engineering_passed': all(s['complete'] for s in summaries.values()),
        'mechanism_evidence': {'generated_multi_action_frozen_invocation': acceptance['frozen_invocation']['program_id'],
            'pilot_multi_action_invocations': len(reused)},
        'pilot': {split: {k: s[k] for k in ('tasks', 'successes', 'total_tokens', 'program_invocations', 'knowledge_digest')}
                  for split, s in summaries.items()},
        'formal_score': False, 'stage_two_ready': bool(reused)}
    write_json(output / 'mechanism_release.json', release)
    print(json.dumps(release, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
