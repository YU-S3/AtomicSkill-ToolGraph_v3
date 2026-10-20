"""Unchanged inference, exact Bank copies, and sequential budget admission."""
import importlib.util
from pathlib import Path

import pytest


def test_repeat_preparation_preserves_runtime_and_bounds(tmp_path, monkeypatch):
    path = Path(__file__).parents[1]/'experiments/run_office_humanstrong_v4_repeats.py'
    spec = importlib.util.spec_from_file_location('repeat_controller', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not m.RUNTIME.exists():
        pytest.skip('Immutable WSL runtime required')
    monkeypatch.chdir(m.RUNTIME)
    monkeypatch.setattr('requests.sessions.Session.request',
                        lambda *a, **kw: pytest.fail('No model HTTP in repeat preparation'))
    c = m.bind(m.RUNTIME)
    before = c.sha(m.PARENT/'budget.json')
    root = tmp_path/'repeat_2'
    plan = m.prepare(c, root, m.PARENT, 2)
    assert plan['code']['git_sha'] == m.RUNTIME_SHA
    assert plan['inherited_used_or_reserved'] == 121011205
    assert plan['ledger_limits']['token_limit'] == 143101957
    assert plan['new_token_cap'] == 22090752
    assert len(plan['episodes']) == 102 and all(e['arm'] == 'manual' for e in plan['episodes'])
    previous = c.read(m.PARENT/'plan.json')
    old = [e for e in previous['episodes'] if e['arm'] == 'manual']
    for current, original in zip(plan['episodes'], old):
        assert current['task_id'] == original['task_id']
        assert current['bank_identity'] == original['bank_identity']
        cfg = c.read(Path(current['path'])/'config.json')
        prior = c.read(Path(original['path'])/'config.json')
        for key in ('llm', 'runtime', 'harness', 'program_environment'):
            assert cfg[key] == prior[key]
    assert c.verify(root) == plan and c.sha(m.PARENT/'budget.json') == before
    with pytest.raises(ValueError, match='Billing parent must have completed'):
        m.prepare(c, tmp_path/'repeat_3', root, 3)
    c.save_json(root/'launch_status.json', {'status': 'completed'})
    ledger = c.read(root/'budget.json')
    ledger['attempts']['offline_expensive'] = {
        'status': 'metered', 'accounted_tokens': 12_000_000, 'reserved_tokens': 12_000_000}
    c.save_json(root/'budget.json', ledger)
    with pytest.raises(ValueError, match='Cumulative authority insufficient'):
        m.prepare(c, tmp_path/'repeat_3', root, 3)
    assert not (tmp_path/'repeat_3').exists()
