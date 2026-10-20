"""Zero-network production regressions, including real Docker and original refusal replay."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import sqlite3
import tempfile
import pytest
import campaign as c
import provider_outcomes
from provider_outcomes import retain_reservation
from atomic_skillgraph.agents.provider import AgentProviderError
from atomic_skillgraph.empirical.system import EmpiricalSystem

r, HERE = c.r, c.HERE
SPEC = r.read(HERE / 'spec.json')
SELECTED = r.read(SPEC['tasks'])['tasks'][:2]
USAGE = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}
BASELINE = Path('/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020/fulltest1400')
MESSAGES = {}
for arm in SPEC['arms']:
    with sqlite3.connect((BASELINE / arm / 'test/run.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        MESSAGES[arm] = {t['task_id']: json.loads(db.execute('SELECT result FROM tasks WHERE id=?',
            (t['task_id'],)).fetchone()[0])['requests'][0]['messages'] for t in SELECTED}


def exercise(mode):
    root = Path(tempfile.mkdtemp(prefix='glm_refusal_' + mode + '_', dir='/home/yangchengyu'))
    limits = {'token_limit': 2000000, 'request_limit': 20, 'finish_reserve': 0}
    ledger = {'limits': limits, 'attempts': {}, 'unknown_billing': False}
    spec = {**SPEC, 'tasks': str(root / 'tasks.json'), 'replay_sources': {}}
    if mode == 'original_replay':
        snap = r.read(HERE / 'parent_snapshot.json')
        parent_ledger = r.read(Path(snap['root']) / 'budget.json')
        ledger['attempts'] = {key: deepcopy(parent_ledger['attempts'][key]) for key in snap['refused_attempts']}
        ledger['unknown_billing'] = True
        spec['replay_sources'] = SPEC['replay_sources']
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, ledger)
    if mode == 'original_replay':
        governor = r.BudgetGovernor(root / 'budget.json', **limits)
        for attempt in snap['refused_attempts'].values():
            retain_reservation(governor, attempt)
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'tasks.json', {'tasks': SELECTED})
    r.write_json(root / 'spec.json', spec)
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'fixture_only': True,
        'driver_sha256': r.sha(c.__file__)})
    seen = []

    def post(*args, **kwargs):
        current = r.read(root / 'fixture/current.json')
        arm, task_id = current['arm'], current['task_id']
        wire = deepcopy(kwargs['json'])
        assert wire['messages'] == MESSAGES[arm][task_id]
        assert wire['model'] == 'glm-5.3-flash' and wire['max_tokens'] == 65536
        assert wire['stream'] and wire['thinking'] == {'type': 'enabled', 'clear_thinking': False}
        assert wire['reasoning_effort'] == 'high' and wire['temperature'] == 1.0 and wire['top_p'] == .95
        assert not {'seed', 'stream_options', 'tools', 'tool_choice'} & wire.keys()
        seen.append({'arm': arm, 'task_id': task_id})
        first = task_id == SELECTED[0]['task_id']
        abnormal = first and mode not in {'normal', 'original_replay'}
        finish = ('sensitive' if mode.startswith(('refusal', 'crash')) else
                  'network_error' if mode.startswith('network') else
                  'model_context_window_exceeded' if mode == 'context_error' else 'stop') if abnormal else 'stop'
        content = '' if abnormal else '<answer>fixture</answer>'
        chunk = {'id': 'fixture', 'model': 'glm-5.3-flash',
            'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': finish}]}
        if not abnormal or mode in {'refusal_metered', 'network_metered', 'context_error'}:
            chunk['usage'] = USAGE
        def lines(**unused):
            yield b'data: ' + json.dumps(chunk).encode()
            if not (abnormal and mode == 'broken_stream'):
                yield b'data: [DONE]'
        return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lines, close=lambda: None)

    succeeds = mode in {'normal', 'refusal_metered', 'refusal_unknown', 'original_replay',
                        'crash_before_reservation', 'crash_before_score'}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda path: None)
        patch.setenv('ZHIPU_API_KEY', 'offline-fixture')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Real network forbidden')))
        with (root / 'fixture.log').open('w') as output, redirect_stdout(output):
            if mode.startswith('crash'):
                target, attr = (provider_outcomes, 'retain_reservation') if mode == 'crash_before_reservation' else (EmpiricalSystem, 'complete_execution')
                real = getattr(target, attr)
                def crash(*args, **kwargs):
                    raise SystemExit('simulated process crash after recorded provider refusal')
                with pytest.MonkeyPatch.context() as crash_patch:
                    crash_patch.setattr(target, attr, crash)
                    with pytest.raises(SystemExit):
                        c.execute('fixture')
                assert len(seen) == 1
            if succeeds:
                value = c.execute('fixture')
                assert all(v['completed'] == 2 for v in value['arms'].values())
                expected = 3 if mode == 'original_replay' else 4
                assert len(seen) == expected
                c.execute('fixture')
                assert len(seen) == expected
                for arm in SPEC['arms']:
                    row = c.completed(root / 'fixture', arm)[SELECTED[0]['task_id']]
                    refusal = mode.startswith(('refusal', 'crash')) or (mode == 'original_replay' and arm == 'v7_text_compiler')
                    if refusal:
                        assert row['execution']['reason'] == 'provider_content_filter'
                        assert row['score']['hard'] is False and row['execution']['prediction'] is None
                        assert row['execution']['attempts'] == [] and len(row['requests']) == 1
                        assert len(row['requests'][0]['http_attempts']) == 1
                        assert not row['execution']['all_authored_nodes_completed']
                        assert sum(u['total_tokens'] for u in row['usage']) == (5 if mode == 'refusal_metered' else 0)
                        if mode != 'refusal_metered':
                            assert row['requests'][0]['http_attempts'][0]['raw_usage'] is None
                assert value['arms']['v7_text_compiler']['program_ok'] == (2 if mode == 'normal' else 1)
            else:
                with pytest.raises(AgentProviderError):
                    c.execute('fixture')
                assert len(seen) == 1 and not any(c.completed(root / 'fixture', a) for a in SPEC['arms'])
                if mode not in {'network_metered', 'context_error'}:
                    before = len(seen)
                    with pytest.raises(AssertionError, match='Unresolved billing'):
                        c.execute('fixture')
                    assert len(seen) == before
            assert all(r.read(root / 'fixture/freeze_audit.json').values())
        state = r.read(root / 'budget.json')
        assert len(state['attempts']) == len(seen) + (mode == 'original_replay')
        assert all(state['attempts'].get(k) == v for k, v in ledger['attempts'].items())
        assert sum(a.get('accounted_tokens', 0) for a in state['attempts'].values()) == {
            'normal': 20, 'refusal_metered': 20, 'refusal_unknown': 10, 'original_replay': 15,
            'crash_before_reservation': 10, 'crash_before_score': 10,
            'network_metered': 5, 'context_error': 5}.get(mode, 0)
    return {'mode': mode, 'passed': True, 'intercepted_http': len(seen), 'real_http': 0, 'fixture': str(root)}


checks = [exercise(mode) for mode in ('normal', 'refusal_metered', 'refusal_unknown', 'original_replay',
    'missing_usage', 'broken_stream', 'network_metered', 'network_unknown', 'context_error',
    'crash_before_reservation', 'crash_before_score')]
r.write_json(HERE / 'offline_acceptance.json', {
    'passed': True, 'real_http': 0, 'checks': checks, 'source': r.code_identity(),
    'driver_sha256': r.sha(c.__file__), 'guard_sha256': r.sha(HERE / 'provider_outcomes.py'),
    'test_sha256': r.sha(__file__), 'same_messages_and_model_parameters': True,
    'refusal_never_retried': True, 'unknown_usage_not_fabricated': True,
    'nonterminal_errors_not_scored': True, 'original_refusal_replayed_without_http': True,
    'normal_native_execution_and_resume_verified': True})
print(json.dumps(checks))
