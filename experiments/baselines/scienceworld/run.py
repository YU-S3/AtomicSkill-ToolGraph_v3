"""Independent ScienceWorld baseline lane. No train/test barrier across seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml
from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.harness.scienceworld_resource import verify_resource
from experiments.scienceworld_manifest import load
from experiments.baselines.common.model_client import AuditedChatClient
from .runner import ScienceWorldTextEpisodeRunner
from .report import summarize

ROOT = Path(__file__).resolve().parents[3]
METHODS = ('b0_dynamic', 'b1_static_skill', 'b3_skillopt', 'b4_embodiskill', 'b5_gepa')

def run(root, method, seed, *, smoke=False, resume=False):
    root = Path(root).expanduser().resolve() / method / f'seed{seed}'
    if not root.is_relative_to(Path.home()) or str(root).startswith('/mnt/'):
        raise ValueError('Use Linux home for state and JVM experiment outputs')
    if root.exists() and not resume:
        raise FileExistsError(f'Use a fresh root or --resume: {root}')
    root.mkdir(parents=True, exist_ok=True)
    datasets = {s: load(ROOT / f'data/scienceworld_manifests/{s}_{n}.json')
                for s, n in [('train',120), ('dev',10), ('test',90)]}
    verify_resource(datasets['train']['resource_identity'])
    model = yaml.safe_load((ROOT / 'configs/baselines/common.yaml').read_text())['model']
    identity = {'method': method, 'run_id': root.parent.parent.name, 'run_seed': seed,
        'manifest_digests': {s: d['digest'] for s,d in datasets.items()}, 'model': model,
        'experiment_kind': 'diagnostic' if smoke else 'formal'}
    if (root / 'identity.json').exists() and json.loads((root / 'identity.json').read_text()) != identity:
        raise ValueError('Resume identity mismatch')
    atomic_write_json(root / 'identity.json', identity)
    def chat_factory(output, entry):
        client = AuditedChatClient(output=output / 'provider_calls.jsonl',
            identity={**identity, 'phase': entry['source_split'], 'task_id': entry['task_id']}, model=model)
        def chat(*, messages, repair, task_id):
            return client.chat(messages=messages, role='target',
                stage='scienceworld_protocol_repair' if repair else 'scienceworld_policy')
        return chat
    train, dev, test = [datasets[s]['tasks'] for s in ('train','dev','test')]
    if smoke:
        # Same fixed three physical test variations for every method.
        train, dev, test = train[:2], dev[:2], test[:3]
    frozen = root / 'frozen'
    if method == 'b4_embodiskill':
        from .embodiskill_lane import run_embodiskill
        return run_embodiskill(root, train, dev, test, model, identity, smoke=smoke, resume=resume)
    if not frozen.exists():
        if method == 'b0_dynamic':
            skill = ''
        elif method == 'b1_static_skill':
            skill = (ROOT / 'experiments/baselines/assets/scienceworld_initial.md').read_text()
        else:
            sys.path.insert(0, str(ROOT / '.external/skillopt'))
            sys.path.insert(0, str(ROOT / '.external/gepa/src'))
            from .learning import train_skillopt, train_gepa
            train_root = root / 'train'
            train_root.mkdir(exist_ok=True)
            if method == 'b3_skillopt':
                skill = train_skillopt(train_root, train, dev, chat_factory, model, seed, smoke=smoke)
            else:
                evolution = AuditedChatClient(output=train_root / 'provider_calls.jsonl',
                    identity={**identity, 'phase': 'train'}, model=model)
                skill = train_gepa(train_root, train, dev, chat_factory, evolution, seed, smoke=smoke)
        frozen.mkdir()
        (frozen / 'skill.md').write_text(skill, encoding='utf-8')
        atomic_write_json(frozen / 'manifest.json', {**identity, 'sha256': hashlib.sha256(skill.encode()).hexdigest(),
            'learning_enabled': False, 'source': 'no_persistent_state' if method == 'b0_dynamic' else method})
    skill = (frozen / 'skill.md').read_text()
    authority = json.loads((frozen / 'manifest.json').read_text())
    if hashlib.sha256(skill.encode()).hexdigest() != authority['sha256']:
        raise ValueError('Frozen skill digest mismatch')
    frozen_bytes = {p.name: p.read_bytes() for p in frozen.iterdir() if p.is_file()}
    for repeat in range(1, (3 if seed == 42 and not smoke else 1) + 1):
        scope, rows = root / f'test_repeat{repeat}', []
        for entry in test:
            output = scope / entry['task_id']
            print(json.dumps({'method': method, 'seed': seed, 'repeat': repeat, 'task': entry['task_id']}), flush=True)
            if resume and (output / 'result.json').exists():
                result = json.loads((output / 'result.json').read_text())
                if result['source_identity'] != entry:
                    raise ValueError('Resume episode source mismatch')
            else:
                result = ScienceWorldTextEpisodeRunner(chat_factory(output, entry)).run(entry, skill, output)
            rows.append(result)
            if frozen_bytes != {p.name: p.read_bytes() for p in frozen.iterdir() if p.is_file()}:
                raise RuntimeError('Readonly Test modified the frozen state')
        atomic_write_json(scope / 'summary.json', summarize(rows, scope))
    atomic_write_json(root / 'completion.json', {'completed': True, 'test_tasks': len(test), 'identity': identity})
    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, choices=(42,43,44), default=42)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    raise SystemExit(run(**vars(p.parse_args())))
