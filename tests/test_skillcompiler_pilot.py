import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.harness.protocol import HarnessTask
from experiments.run_skillcompiler_pilot import select
from experiments.run_v3_r103_validation import resolve_validation_tasks, verify_runner_recovery
from experiments.protocol import hash_code


def test_training_selection_is_source_order_two_per_family():
    path = Path(__file__).resolve().parents[1] / 'data/baseline_manifests/train_120.json'
    chosen, audit = select(path, 'train', 2)
    assert len(chosen) == 12 and set(audit['counts'].values()) == {2}
    source = json.loads(path.read_text(encoding='utf-8'))['tasks']
    assert chosen == [row for row in source if row['task_id'] in {x['task_id'] for x in chosen}]


def validation_fixture(tmp_path):
    relative = 'json_2.1.1/valid_seen/pick_and_place_simple-X/trial/game.tw-pddl'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"identity":"fixed-game"}', encoding='utf-8')
    task = HarnessTask(task_id='alfworld_eval_in_distribution_105_pick_and_place_simple',
        goal='place the object', benchmark='alfworld', task_type='pick_and_place_simple',
        context={'game_file': str(path), 'env_index': 105})
    task.metadata['task_signature'] = hashlib.sha256('\x1f'.join(
        ('eval_in_distribution', str(path), task.goal)).encode()).hexdigest()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {'task_id': 'alfworld_eval_in_distribution_122_pick_and_place_simple', 'env_index': 122,
        'source_split': 'valid_seen', 'task_type': task.task_type, 'gamefile_rel': relative,
        'gamefile_sha256': digest, 'task_signature': hashlib.sha256('\x1f'.join(
            ('alfworld', 'valid_seen', task.task_type, relative, digest)).encode()).hexdigest()}
    calls = []
    def load_tasks(*, limit):
        calls.append(limit)
        return [task]
    return SimpleNamespace(harness=SimpleNamespace(alfworld_data=str(tmp_path),
        split='eval_in_distribution', load_tasks=load_tasks)), entry, task, calls


def test_held_out_selection_preserves_game_despite_different_discovery_indices(tmp_path):
    system, entry, task, calls = validation_fixture(tmp_path)
    original_signature = task.metadata['task_signature']
    audit = tmp_path / 'selection_audit.json'
    assert resolve_validation_tasks(system, [entry], audit_path=audit) == [task]
    assert calls == [0]
    assert task.context['env_index'] == 105
    assert task.metadata['task_signature'] == original_signature != entry['task_signature']
    mapping, = json.loads(audit.read_text())['tasks']
    assert mapping['manifest_env_index'] == 122 and mapping['harness_env_index'] == 105
    assert mapping['gamefile_sha256'] == entry['gamefile_sha256']


@pytest.mark.parametrize('mutation', ['file_hash', 'manifest_signature', 'family',
    'harness_signature', 'wrong_split', 'duplicate', 'escape'])
def test_held_out_selection_rejects_changed_physical_authority(tmp_path, mutation):
    system, entry, task, _ = validation_fixture(tmp_path)
    if mutation == 'file_hash': entry['gamefile_sha256'] = '0' * 64
    elif mutation == 'manifest_signature': entry['task_signature'] = '0' * 64
    elif mutation == 'family': entry['task_type'] = 'another_family'
    elif mutation == 'harness_signature': task.metadata['task_signature'] = '0' * 64
    elif mutation == 'wrong_split': entry['source_split'] = 'valid_unseen'
    elif mutation == 'escape': entry['gamefile_rel'] = '../outside/game.tw-pddl'
    audit = tmp_path / 'selection_audit.json'
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        resolve_validation_tasks(system, [entry, entry] if mutation == 'duplicate' else [entry], audit_path=audit)
    assert not audit.exists()


@pytest.mark.parametrize('extra_change', [None, 'src/runtime.py', 'src/atomic_skillgraph/traces/store.py',
                                       'configs/source.yaml'])
def test_runner_recovery_proves_unchanged_execution_including_trace_writers(tmp_path, monkeypatch, extra_change):
    from experiments import run_v3_r103_validation as module
    paths = ['experiments/run_skillcompiler_pilot.py', 'experiments/run_v3_r103_validation.py',
             'src/runtime.py', 'src/atomic_skillgraph/traces/store.py', 'configs/source.yaml']
    original, current = tmp_path / 'original', tmp_path / 'current'
    for root in (original, current):
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original', encoding='utf-8')
    old_code = hash_code(original)
    (current / paths[0]).write_text('runner repaired', encoding='utf-8')
    if extra_change:
        (current / extra_change).write_text('changed execution', encoding='utf-8')
    monkeypatch.setattr(module, 'REPO', current)
    arguments = ({'code_hash': old_code}, {'provenance': {'source_code_commit': old_code}},
                 {'_config_path': str(original / 'configs/source.yaml')}, hash_code(current))
    if extra_change:
        with pytest.raises(RuntimeError):
            verify_runner_recovery(*arguments)
    else:
        audit = verify_runner_recovery(*arguments)
        assert audit['production_execution_package_identical']
        assert audit['changed_files'] == [paths[0]]


def test_validation_only_uses_completed_train_without_restarting_it(tmp_path, monkeypatch):
    from experiments import run_skillcompiler_pilot as module
    from atomic_skillgraph.agents.node_context import VERSION as node_version
    from atomic_skillgraph.evolution.extraction_view import VERSION as extraction_version
    from atomic_skillgraph.evolution.realization_queue import VERSION as realization_version
    from atomic_skillgraph.runtime.scope_diagnostics import OUTCOME_VERSION
    train, train_source = module.select(module.REPO / 'data/baseline_manifests/train_120.json', 'train', 2)
    val_manifest = tmp_path / 'val.json'
    val_manifest.write_text(json.dumps({'manifest_id': 'fixed_val', 'tasks': [
        {'task_id': f'val_{n}', 'task_type': family, 'source_split': 'valid_seen'}
        for n, family in enumerate(module.ALFWORLD_FORMAL_TASK_TYPES)]}), encoding='utf-8')
    val, val_source = module.select(val_manifest, 'valid_seen', 1)
    selection = {'train': train, 'valid_seen': val, 'sources': {'train': train_source, 'valid_seen': val_source},
        'protocol_versions': {'node_context': node_version, 'e1': extraction_version,
                              'realization': realization_version, 'search_outcome': OUTCOME_VERSION}}
    output = tmp_path / 'pilot'
    (output / 'train12').mkdir(parents=True)
    (output / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    (output / 'train12/summary.json').write_text(json.dumps({'complete': True, 'tasks': 12}), encoding='utf-8')
    monkeypatch.setattr(module, 'train_dev16', lambda *a, **k: pytest.fail('completed training restarted'))
    calls = []
    def deploy(*args, **kwargs):
        calls.append((args, kwargs))
        return {'complete': True}
    monkeypatch.setattr(module, 'deploy', deploy)
    monkeypatch.setattr('sys.argv', ['pilot', '--output', str(output), '--val-manifest', str(val_manifest),
        '--validation-only', '--runner-recovery', '--validation-output', str(tmp_path / 'valid_recovered')])
    module.main()
    assert len(calls) == 1
    assert calls[0][0][2] == output / 'train12'
    assert calls[0][1] == {'validation_entries': val, 'runner_recovery': True}
    assert json.loads((output / 'selection.json').read_text()) == selection
