"""Physical backend lifecycle; no evidence or result validator is bypassed."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from atomic_skillgraph.core.errors import AtomicSkillGraphError
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter


@pytest.fixture
def backend(monkeypatch, tmp_path):
    counts = dict(collect=0, create=0, reset=0, close=0)
    files = [str(tmp_path / f'json_2.1.1/train/pick_and_place_simple-{i}/game.tw-pddl') for i in range(5)]
    for file in files:
        path = Path(file)
        path.parent.mkdir(parents=True)
        path.write_text(file)

    class Environment:
        def __init__(self, files):
            self.files, self.index, self.fail, self.closed = files, 0, False, False
        def reset(self):
            counts['reset'] += 1
            if self.fail:
                raise RuntimeError('declared reset failure')
            file = self.files[self.index % len(self.files)]
            self.index += 1
            return ['Your task is to: put an apple in a bowl.'], {
                'extra.gamefile': [file], 'admissible_commands': [['look']]}
        def close(self):
            assert not self.closed
            self.closed = True
            counts['close'] += 1

    class Library:
        def __init__(self, config, train_eval):
            self.config = config
            self.collect_game_files()
        def collect_game_files(self, verbose=False):
            counts['collect'] += 1
            self.game_files, self.num_games = files[:], len(files)
        def init_env(self, batch_size):
            assert batch_size == 1
            counts['create'] += 1
            return Environment(self.game_files)

    parent = ModuleType('alfworld'); agents = ModuleType('alfworld.agents')
    module = ModuleType('alfworld.agents.environment')
    parent.agents = agents; agents.environment = module
    module.get_environment = lambda name: Library
    for name, value in [('alfworld',parent),('alfworld.agents',agents),('alfworld.agents.environment',module)]:
        monkeypatch.setitem(sys.modules, name, value)
    return counts


def test_discovery_exact_reuse_clean_state_and_return_to_discovery(backend):
    h = AlfWorldAdapter(split='train'); tasks = h.load_tasks()
    assert len(tasks) == 5
    assert backend == dict(collect=1, create=1, reset=5, close=0)
    for _ in range(3):
        result = h.reset(tasks[4])
        assert result.accepted and h._current_task.task_id == tasks[4].task_id
        assert h._revision == 0 and not h._runtime_accepted_prefix and not h._done
        h._revision = 99; h._runtime_accepted_prefix.append({'dirty':True}); h._done = True
    assert backend == dict(collect=1, create=2, reset=8, close=1)
    h.reset(tasks[1])
    assert backend == dict(collect=1, create=3, reset=9, close=2)
    again = h.load_tasks()
    assert again == tasks
    h.reset(tasks[3])
    balanced = h.load_balanced_tasks(['pick_and_place_simple'], 3)
    assert balanced == tasks[:3]
    h._close_backend(); h._close_backend()
    assert backend['close'] == backend['create']


def test_untrusted_mapping_and_identity_never_load_arbitrary_file(backend):
    h = AlfWorldAdapter(split='train'); tasks = h.load_tasks()
    wrong = copy.deepcopy(tasks[3]); wrong.context['game_file'] = tasks[1].context['game_file']
    before = dict(backend)
    with pytest.raises(AtomicSkillGraphError, match='file/index mismatch'):
        h.reset(wrong)
    assert backend == before
    wrong = copy.deepcopy(tasks[3]); wrong.goal = 'different goal'
    with pytest.raises(AtomicSkillGraphError, match='mapping changed'):
        h.reset(wrong)
    assert h._env is None
    h.reset(tasks[3]); assert h._env is not None
    h._env.fail = True
    with pytest.raises(AtomicSkillGraphError, match='reset failed'):
        h.reset(tasks[3])
    assert h._env is None and h._tw_env is None
    h.reset(tasks[3]); h._close_backend()
    assert backend['close'] == backend['create']


def test_canonical_id_wraps_verified_native_id_without_relaxing_reset(backend, tmp_path):
    from atomic_skillgraph.experiments.run_empirical import resolve_alfworld_tasks
    from atomic_skillgraph.harness.alfworld_simple import SimpleAlfWorld
    harness = AlfWorldAdapter(split='train', alfworld_data=str(tmp_path))
    task = harness.load_tasks()[2]
    path = Path(task.context['game_file'])
    entries = [{'task_id': 'alfworld:canonical', 'source_split': 'train', 'env_index': None,
                'task_type': task.task_type, 'gamefile_rel': path.relative_to(tmp_path).as_posix(),
                'gamefile_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
    adapter = SimpleAlfWorld(harness)
    public = resolve_alfworld_tasks(adapter, entries, canonical_split='train')[0]
    assert public.task_id == 'alfworld:canonical'
    assert public.inputs['environment_task']['native_task_id'] == task.task_id
    adapter.reset(public)
    assert harness._current_task.task_id == task.task_id
    assert harness._current_task.context['game_file'] == str(path)
    adapter.close()


def test_unknown_index_discovered_once_and_configuration_invalidates(backend):
    original = AlfWorldAdapter(split='train'); task = original.load_tasks()[4]
    original._close_backend()
    h = AlfWorldAdapter(split='train')
    h.reset(task)
    after = dict(backend)
    h.reset(task)
    assert backend['collect'] == after['collect'] and backend['reset'] == after['reset'] + 1
    h.max_steps += 1
    h.reset(task)
    assert backend['collect'] == after['collect'] + 1
    h._close_backend()


def test_physical_selection_scans_past_old_ordinal_and_keeps_reset_guards(backend, tmp_path):
    from atomic_skillgraph.experiments.run_empirical import resolve_alfworld_tasks
    h = AlfWorldAdapter(split='train', alfworld_data=str(tmp_path))
    entries = []
    for old_index, native_index in enumerate([4, 1]):
        path = tmp_path / f'json_2.1.1/train/pick_and_place_simple-{native_index}/game.tw-pddl'
        entries.append({'task_id': f'old_{old_index}', 'env_index': old_index, 'source_split': 'train',
            'task_type': 'pick_and_place_simple', 'gamefile_rel': path.relative_to(tmp_path).as_posix(),
            'gamefile_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    mapping = tmp_path / 'mapping.json'
    tasks = resolve_alfworld_tasks(SimpleNamespace(harness=h), entries, mapping_path=mapping)
    assert [t.inputs['environment_task']['context']['env_index'] for t in tasks] == [4, 1]
    assert backend['reset'] == 5
    recorded = json.loads(mapping.read_text())['tasks']
    assert [t['source_task_id'] for t in recorded] == ['old_0', 'old_1']
    assert [t['runtime_task_id'] for t in recorded] == [t.task_id for t in tasks]
    from atomic_skillgraph.harness.alfworld_simple import SimpleAlfWorld
    adapter = SimpleAlfWorld(h)
    adapter.reset(tasks[0])
    wrong = copy.deepcopy(tasks[0])
    wrong.inputs['environment_task']['context']['env_index'] = 1
    with pytest.raises(AtomicSkillGraphError, match='file/index mismatch'):
        adapter.reset(wrong)
    with pytest.raises(ValueError, match='Duplicate physical task'):
        resolve_alfworld_tasks(adapter, [entries[0], entries[0]])
    wrong_entry = dict(entries[0], gamefile_sha256='changed')
    with pytest.raises(ValueError, match='file hash mismatch'):
        resolve_alfworld_tasks(adapter, [wrong_entry])
    wrong_entry = dict(entries[0], gamefile_rel='../outside/game.tw-pddl')
    outside = tmp_path.parent / 'outside' / 'game.tw-pddl'
    outside.parent.mkdir(exist_ok=True)
    outside.write_text('outside')
    with pytest.raises(ValueError, match='within the dataset'):
        resolve_alfworld_tasks(adapter, [wrong_entry])
    adapter.close()


def test_targeted_discovery_restarts_and_stops_when_all_targets_found(backend, tmp_path):
    h = AlfWorldAdapter(split='train')
    targets = {str(tmp_path / f'json_2.1.1/train/pick_and_place_simple-{i}/game.tw-pddl') for i in [1, 3]}
    first = h.load_tasks(game_files=targets)
    assert [t.context['env_index'] for t in first] == [1, 3] and backend['reset'] == 4
    assert h.load_tasks(game_files=targets) == first and backend['reset'] == 8
    assert len(targets) == 2
    with pytest.raises(ValueError, match='ordinal scan limit'):
        h.load_tasks(game_files=targets, limit=2)
    h._close_backend()


def test_repeated_full_discovery_preserves_native_indices(backend):
    h = AlfWorldAdapter(split='train')
    first = h.load_tasks()
    assert h.load_tasks() == first
    assert backend['collect'] == 2 and backend['reset'] == 10
    h._close_backend()


def test_missing_physical_task_fails_after_bounded_scan_without_substitution(backend, tmp_path):
    h = AlfWorldAdapter(split='train')
    with pytest.raises(AtomicSkillGraphError, match='absent from ALFWorld discovery'):
        h.load_tasks(game_files={str(tmp_path / 'not-in-native-split/game.tw-pddl')})
    assert backend['reset'] == 5 and backend['close'] == 1
    assert h._env is None
