"""Original four-chunk manual learning, continuous Dev selection and frozen tests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import yaml
from atomic_skillgraph.core.serialization import atomic_write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.b4_embodiskill.state import copy_state, publish_checkpoint, load_checkpoint
from experiments.baselines.b4_embodiskill.manifest_adapter import train_chunks
from .report import summarize

ROOT = Path(__file__).resolve().parents[3]


def run_embodiskill(root, train, dev, test, model, identity, *, smoke, resume):
    seed = identity['run_seed']
    cfg = yaml.safe_load((ROOT / 'configs/baselines/b4_embodiskill.yaml').read_text())['embodiskill']
    cfg.update(max_trials=100, static_few_shots=0)
    dependencies = ROOT / '.venv_b4_embodiskill/lib/python3.12/site-packages'
    if not dependencies.is_dir():
        raise FileNotFoundError('Configure the pinned EmbodiSkill dependency environment first')

    def operation(name, phase, state, *, epoch, entry=None, train_score=None):
        receipt = load_checkpoint(root, name)
        if receipt:
            if not resume:
                raise FileExistsError(name)
            return Path(receipt['state']), receipt['result']
        output = root / 'operations' / name / uuid.uuid4().hex
        output.mkdir(parents=True)
        destination = output / 'state'
        before = copy_state(state, destination)
        job = {'operation': name, 'phase': phase, 'state': str(destination), 'output': str(output),
            'source': str(ROOT / '.external/embodiskill'), 'embedding_path': str(ROOT / '.external/embedding_models/all-MiniLM-L6-v2'),
            'seed': seed, 'epoch': epoch, 'entry': entry, 'train_score': train_score, 'model': model, 'config': cfg,
            'identity': {**identity, 'phase': phase, 'operation': name, 'task_id': (entry or {}).get('task_id')}}
        atomic_write_json(output / 'job.json', job)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT/'src'), str(ROOT), str(dependencies))))
        print(json.dumps({'method': 'b4_embodiskill', 'seed': seed, 'operation': name}), flush=True)
        with (output / 'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'experiments.baselines.scienceworld.embodiskill_worker',
                '--job', str(output / 'job.json')], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if state is not None and digest_directory(state) != before:
            raise RuntimeError('Worker mutated its source snapshot')
        receipt = publish_checkpoint(root, name, output, destination)
        return destination, receipt['result']

    frozen, state = root / 'frozen', None
    if not frozen.exists():
        chunks = train_chunks(train, seed=seed, epochs=1 if smoke else 4, chunk_size=len(train) if smoke else 30)
        best_score, best, selection = float('-inf'), None, []
        for epoch, chunk in enumerate(chunks, 1):
            scores = []
            for i, entry in enumerate(chunk):
                state, result = operation(f'train_e{epoch}_{i:03d}', 'train', state, epoch=epoch, entry=entry)
                scores.append(result['normalized_score'])
            state, _ = operation(f'revision_e{epoch}', 'revision', state, epoch=epoch, train_score=sum(scores)/len(scores))
            rows = [operation(f'dev_e{epoch}_{i:03d}', 'dev', state, epoch=epoch, entry=e)[1] for i,e in enumerate(dev)]
            score = sum(r['normalized_score'] for r in rows)/len(rows)
            selected = score > best_score
            if selected:
                best_score, best = score, state
            selection.append({'epoch': epoch, 'dev_mean_normalized_score': score, 'selected': selected, 'state': str(state)})
            atomic_write_json(root / 'selection.json', selection)
        frozen.mkdir()
        copy_state(best, frozen / 'state')
        atomic_write_json(frozen / 'manifest.json', {**identity, 'state_digest': digest_directory(frozen/'state'),
            'selection_metric': 'dev_mean_normalized_score', 'selection_tie': 'earlier', 'selected_state': str(best),
            'learning_enabled': False})
    authority = json.loads((frozen / 'manifest.json').read_text())
    if digest_directory(frozen/'state') != authority['state_digest']:
        raise RuntimeError('Frozen state digest mismatch')
    for repetition in range(1, (3 if seed == 42 and not smoke else 1) + 1):
        rows = [operation(f'test_r{repetition}_{i:03d}', 'test', frozen/'state', epoch=0, entry=e)[1] for i,e in enumerate(test)]
        scope = root / f'test_repeat{repetition}'
        atomic_write_json(scope / 'results.json', rows)
        # Operation logs remain immutable alongside their exact copied state.
        atomic_write_json(scope / 'summary.json', summarize(rows, root/'operations', operation_prefix=f'test_r{repetition}_'))
    atomic_write_json(root / 'completion.json', {'completed': True, 'test_tasks': len(test), 'identity': identity})
    return 0
