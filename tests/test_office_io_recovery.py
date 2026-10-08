"""Corpus authorization and explicit, single-execution trial recovery."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillcompiler_bench_contracts.office import grep, select_paths
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.experiments.formal_log import FormalLog
from test_cf2_contracts import office
from test_empirical import program


def test_selection_resolves_shared_parent_once_and_preserves_aliases(tmp_path, monkeypatch):
    root = tmp_path / 'corpus'; root.mkdir()
    for i in range(20): (root / f'{i:02}.txt').write_text('α target\r\nsecond target', encoding='utf-8')
    (root / 'alias.txt').symlink_to(root / '00.txt')
    original, calls = Path.resolve, []
    def resolve(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    selected = select_paths(root)
    assert len(selected) == 20 and sum(p == root for p in calls) == 2
    assert all(p.name not in {f'{i:02}.txt' for i in range(20)} for p in calls)
    hits, audit = grep(root, 'target', ['alias.txt', './00.txt'])
    assert hits == [{'path': '00.txt', 'line': 1, 'offset': 0, 'text': 'α target'},
                    {'path': '00.txt', 'line': 2, 'offset': 9, 'text': 'second target'}]
    assert audit == {'execution_paths': ['00.txt'], 'truncated': False}


@pytest.mark.parametrize('kind', ['file', 'parent', 'dangling'])
def test_selection_keeps_symlink_authorization(tmp_path, kind):
    root = tmp_path / 'corpus'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir(); (outside / 'x.txt').write_text('secret')
    if kind == 'parent':
        (root / 'link').symlink_to(outside, target_is_directory=True); scope = ['link/x.txt']
    else:
        (root / 'x.txt').symlink_to(outside / ('x.txt' if kind == 'file' else 'absent.txt')); scope = ['x.txt']
    with pytest.raises(ValueError, match='authorized'): select_paths(root, scope)


def test_cached_learning_keeps_sent_schema_and_semantics(tmp_path):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    calls = []
    def agent(*args, **kwargs): calls.append(args); return {'skill_id': '$new'}
    s.agent = agent
    schema = {'type': 'object', 'properties': {'skill_id': {'enum': ['$new']}}}
    try:
        result = s.learner._receive('proposal', 'extractor', 'prompt', {'material': 1}, 'submit', schema)
        expanded = deepcopy(schema); expanded['properties']['skill_id']['enum'].append('persisted_skill')
        assert s.learner._receive('proposal', 'extractor', 'prompt', {'material': 2}, 'submit', expanded) == result
        assert len(calls) == 1
        with pytest.raises(ValueError, match='semantics changed'):
            s.learner._receive('proposal', 'extractor', 'changed prompt', {}, 'submit', expanded)
    finally: s.close()


def test_explicit_trial_authorization_is_consumed_once_and_logs_new_scope(tmp_path):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    s.adapter_factory = lambda: type(s.adapter)({'office': {'answer': '42'}}, s.config)
    p = s.bank.put('program', {**program('def run(ctx, inputs): raise AssertionError("inert source")'),
                              'allowed_tools': ['grep']})
    logical = TaskCheckpoint(s.checkpoint.root / 'trials/recover')
    logical.advance('trial_started', execution_id='old')
    old = TaskCheckpoint(logical.root / 'executions/old')
    old.advance('trial_exception', exception={'cause_type': 'UnknownSideEffect'})
    old_bytes = old.path.read_bytes()
    calls = []
    def literal(program, inputs, broker):
        calls.append(1); broker.call('grep', {'pattern': 'target', 'paths': ['a.txt']})
        return {'status': 'execution_error', 'detail': 'literal ordinary failure'}
    s.worker = SimpleNamespace(execute=literal, settings={'max_rpc_message_bytes': 10000})
    log = FormalLog(tmp_path / 'formal', {'fixture': True}, {})
    log.begin_task(s.adapter.task, 0, 'office:1', {})
    s.observer = log
    try:
        with pytest.raises(RuntimeError, match='explicit new execution'):
            s.test_program(p, {}, s.adapter.task, trial_id='recover')
        logical.advance('trial_started', new_execution_authorization={'recovery_id': 'literal-authorization'})
        result = s.test_program(p, {}, s.adapter.task, trial_id='recover')
        assert result['outcome'] == 'execution_failure' and result['basis'] is None
        assert result['id'] == 'recover:' + result['trial_execution_id']
        assert s.test_program(p, {}, s.adapter.task, trial_id='recover') == result and calls == [1]
        assert TaskCheckpoint(logical.root).state['new_execution_authorization'] is None
        assert old.path.read_bytes() == old_bytes and s.bank.get(p['id'])['state'] == 'candidate'
        assert {r['scope_id'] for r in log.rows('interactions')} == {result['id']}
        assert all(r['task_id'] == s.adapter.task.task_id for r in log.rows('interactions'))
    finally: s.close()
