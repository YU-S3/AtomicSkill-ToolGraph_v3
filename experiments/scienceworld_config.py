"""Materialize independent ScienceWorld lane configs; never launches episodes."""
import argparse
import copy
from pathlib import Path
import yaml
from atomic_skillgraph.system import load_config

ROOT = Path(__file__).resolve().parents[1]

def make_config(root, *, seed=42, phase='train', repetition=1, frozen=None, diagnostic=False):
    if seed not in (42, 43, 44) or phase not in ('train', 'dev', 'test'):
        raise ValueError('Unsupported seed/phase')
    root = Path(root).expanduser().resolve()
    if not root.is_relative_to(Path.home()) or str(root).startswith('/mnt/'):
        raise ValueError('Use a fresh Linux home output root')
    output = root / f'seed{seed}' / (f'test_repeat{repetition}' if phase == 'test' else phase)
    config = copy.deepcopy(load_config(ROOT / f'configs/alfworld_train_full_120_r103_seed{seed}.yaml'))
    config['harness'] = {'adapter': 'scienceworld_v1', 'version': '1.2.3', 'split': phase,
        'simplification': 'easy', 'max_steps': 100, 'manifest': str(ROOT / 'data/scienceworld_manifests' /
        {'train': 'train_120.json', 'dev': 'dev_10.json', 'test': 'test_90.json'}[phase])}
    if phase != 'train' and frozen is None:
        raise ValueError('Readonly evaluation requires an explicit frozen bank')
    config['data_dir'] = str(output / 'data_v3' if phase == 'train' else Path(frozen).resolve())
    config['trace_data_dir'] = str(output)
    config['experiment'] = {'name': f'scienceworld_seed{seed}_{phase}_repeat{repetition}',
        'phase': phase, 'condition': 'full', 'runtime_mode': 'online' if phase == 'train' else 'frozen',
        'freeze_skills': phase != 'train', 'seed': seed, 'output_dir': str(output),
        'task_manifest_path': str(output / 'task_manifest.json'), 'max_task_attempts': 3,
        'initialize_v3_bank': 'empty' if phase == 'train' else 'frozen',
        'experiment_kind': 'diagnostic' if diagnostic else 'formal'}
    if phase != 'train':
        config['cold_start']['enabled'] = False
    config['lifecycle']['candidate_exploration_seed'] = seed
    destination = root / 'configs' / f'seed{seed}_{phase}_{repetition}.yaml'
    if destination.exists():
        if yaml.safe_load(destination.read_text()) != config:
            raise FileExistsError(f'Config already exists with different identity: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return destination

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--seed', type=int, choices=(42,43,44), default=42)
    p.add_argument('--phase', choices=('train','dev','test'), default='train')
    p.add_argument('--repetition', type=int, default=1)
    p.add_argument('--frozen')
    p.add_argument('--diagnostic', action='store_true')
    print(make_config(**vars(p.parse_args())))
