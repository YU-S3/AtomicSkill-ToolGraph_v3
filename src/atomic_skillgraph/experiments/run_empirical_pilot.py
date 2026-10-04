"""One fixed 12 Train + 6 valid_seen diagnostic, then a truthful mechanism release."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.harness.registry import create_simple_harness
from .run_empirical import code_identity, load_env, resolve_alfworld_tasks, run, write_json


def completed_train(config, entries, output):
    """Explicit loader-repair continuation; never resume Train under a new identity."""
    train = output / 'train'
    identity = json.loads((train / 'execution_manifest.json').read_text())
    summary = json.loads((train / 'summary.json').read_text())
    expected = {digest({'path': e['gamefile_rel'], 'sha256': e['gamefile_sha256']}) for e in entries}
    if (identity['config'] != config or not summary.get('complete') or summary['tasks'] != len(entries)
            or {t['task']['physical_key'] for t in summary['cases']} != expected):
        raise ValueError('Completed Train config or physical task selection differs')
    root = next((p for p in Path(__file__).resolve().parents if (p / '.git').exists()), None)
    if root is None or not identity['code'].get('git_sha') or identity['code'].get('tracked_dirty'):
        raise ValueError('Loader continuation requires a clean recorded Train source revision')
    changes = subprocess.check_output(['git', 'diff', '--name-only', identity['code']['git_sha'],
        '--', 'src', 'experiments'], cwd=root, text=True).splitlines()
    allowed = {'src/atomic_skillgraph/harness/alfworld.py',
        'src/atomic_skillgraph/experiments/run_empirical.py',
        'src/atomic_skillgraph/experiments/run_empirical_pilot.py',
        'src/atomic_skillgraph/experiments/run_empirical_acceptance.py'}
    if set(changes) - allowed:
        raise ValueError('Mechanism source changed; loader continuation cannot reuse this Train')
    seed = config['experiment']['seed']
    bank = Bank(train / 'bank', readonly=True, seed=seed)
    frozen = Bank(train / 'frozen_bank', readonly=True, seed=seed)
    try:
        freeze = json.loads((train / 'frozen_bank' / 'freeze.json').read_text())
        if (bank.digest() != summary['knowledge_digest'] or freeze != summary['frozen']
                or freeze['source_digest'] != bank.digest() or freeze['digest'] != frozen.digest()):
            raise ValueError('Completed Train or frozen Bank changed')
    finally:
        bank.close()
        frozen.close()
    write_json(output / 'continuation_manifest.json', {
        'reason': 'physical_task_loader_repair', 'train_code': identity['code'], 'val_code': code_identity(),
        'train_summary_sha256': hashlib.sha256((train / 'summary.json').read_bytes()).hexdigest(),
        'selection_digest': digest(entries), 'frozen_digest': freeze['digest'],
        'train_reused_without_execution': True})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/alfworld_empirical_seed42.yaml')
    parser.add_argument('--materials', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continue-val', action='store_true', help='Reuse completed Train after a verified loader-only repair')
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
        config['manifest'] = str(Path(args.materials)/'selection.json')
        config['harness']['split'] = 'train' if split == 'train' else 'eval_in_distribution'
        config['experiment']['runtime_mode'] = 'online' if split == 'train' else 'frozen'
        config['experiment']['output_dir'] = str(output / split)
        config['data_dir'] = str(output / 'train' / ('bank' if split == 'train' else 'frozen_bank'))
        if split == 'train' and args.continue_val:
            summaries[split] = completed_train(config, selection[split], output)
            continue
        adapter = create_simple_harness(config)
        tasks = resolve_alfworld_tasks(adapter, selection[split],
            mapping_path=output / split / 'task_identity_resolution.json')
        summary = run(config, tasks, output / split, resume=args.resume,
            readonly=split != 'train', adapter=adapter, adapter_factory=lambda: create_simple_harness(config))
        summaries[split] = summary
    invocations = [a for summary in summaries.values() for trace in summary['cases']
                   for a in trace['execution'].get('attempts', [])]
    reused = [a for a in invocations if a['status'] == 'ok' and a['calls'] >= 2]
    release = {'schema': 'skillcompiler.mechanism-release.v1', 'code': code_identity(),
        'run_codes': {split: json.loads((output / split / 'execution_manifest.json').read_text())['code']
                      for split in summaries},
        'profile': source_config['mechanism_profile'], 'selection_sha256': digest(selection),
        'acceptance': str(Path(args.acceptance).resolve()), 'acceptance_passed': True,
        'implementation_ready': all(s['complete'] for s in summaries.values()),
        'mechanism_demonstrated': bool(reused),
        'measured_effect': None,
        'mechanism_evidence': {'generated_multi_action_frozen_invocation': acceptance['frozen_invocation']['program_id'],
            'pilot_multi_action_invocations': len(reused)},
        'pilot': {split: {k: s[k] for k in ('tasks', 'successes', 'total_tokens', 'program_invocations', 'knowledge_digest')}
                  for split, s in summaries.items()},
        'formal_score': False}
    write_json(output / 'mechanism_release.json', release)
    print(json.dumps(release, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
