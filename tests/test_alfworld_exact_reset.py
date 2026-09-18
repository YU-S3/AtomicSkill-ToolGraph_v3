"""Physical backend lifecycle; no evidence or result validator is bypassed."""
import copy
import sys
from types import ModuleType

import pytest

from atomic_skillgraph.core.errors import AtomicSkillGraphError
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter


@pytest.fixture
def backend(monkeypatch):
    counts = dict(collect=0, create=0, reset=0, close=0)
    files = [f'/dataset/json_2.1.1/train/pick_and_place_simple-{i}/game.tw-pddl' for i in range(5)]

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
