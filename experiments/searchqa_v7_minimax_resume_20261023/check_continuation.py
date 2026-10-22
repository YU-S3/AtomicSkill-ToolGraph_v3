"""Zero-HTTP checks of inherited billing, real native execution and bounded recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import continue_minimax as run
from atomic_skillgraph.agents.provider import AgentProviderError
from atomic_skillgraph.empirical.budget_governor import billing_blocked

r, c, HERE = run.r, run.c, run.HERE
snap = r.read(HERE / 'parent_snapshot.json')
parent = r.read(run.PARENT / 'budget.json')
selected = r.read(HERE / 'remaining_tasks.json')['tasks'][:2]
old_request = r.read(snap['failed_audit_path'])['requests'][0]
checks = []
for mode in ('normal', 'new_overload'):
    root = Path(tempfile.mkdtemp(prefix='minimax_529_resume_' + mode + '_', dir='/home/yangchengyu'))
    inherited = run.recover_ledger(parent, snap['failed_attempt'])
    r.write_json(root / 'budget.json', inherited)
    r.write_json(root / 'inherited_budget.json', parent)
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'tasks.json', {'tasks': selected})
    r.write_json(root / 'spec.json', {**r.read(HERE / 'spec.json'), 'tasks': str(root / 'tasks.json')})
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'driver_sha256': r.sha(c.__file__)})
    seen = []

    def post(*args, **kwargs):
        wire = deepcopy(kwargs['json'])
        current = r.read(root / 'fixture/current.json')['task_id']
        seen.append({'task': current, 'wire': wire})
        assert current in {t['task_id'] for t in selected}
        assert wire['model'] == 'MiniMax-M3' and wire['max_completion_tokens'] == 65536
        assert wire['thinking'] == {'type': 'adaptive'} and wire['reasoning_split']
        if current == selected[0]['task_id']:
            assert wire['messages'] == old_request['messages']
        if mode == 'new_overload':
            body = {'error': {'type': 'overloaded_error', 'message': 'offline overload'}}
            return SimpleNamespace(status_code=529, ok=False, headers={}, content=json.dumps(body).encode(),
                text=json.dumps(body), json=lambda: body, close=lambda: None)
        chunk = {'id': 'offline', 'model': 'MiniMax-M3', 'created': 1,
            'choices': [{'index': 0, 'delta': {'content': '<answer>fixture</answer>'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
        def lines(**unused):
            yield b'data: ' + json.dumps(chunk).encode()
        return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lines, close=lambda: None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda p: None)
        patch.setenv('MINIMAX_API_KEY', 'offline-fixture')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        with (root / 'fixture.log').open('w') as log, redirect_stdout(log):
            if mode == 'normal':
                c.execute('fixture')
                rows = c.completed(root / 'fixture', run.ARM)
                assert set(rows) == {t['task_id'] for t in selected} and len(seen) == 2
                assert all(x['execution']['all_authored_nodes_completed'] for x in rows.values())
                assert all(a['status'] == 'ok' and a['outputs_consumed'] for x in rows.values() for a in x['execution']['attempts'])
                c.execute('fixture')
                assert len(seen) == 2, 'No duplicate HTTP on completed resume'
            else:
                with pytest.raises(AgentProviderError): c.execute('fixture')
                assert len(seen) == 1 and not c.completed(root / 'fixture', run.ARM)
                with pytest.raises(AssertionError, match='Unresolved billing'): c.execute('fixture')
                assert len(seen) == 1, 'No blanket authorization of future failures'
        ledger = r.read(root / 'budget.json')
        assert ledger['limits'] == parent['limits']
        assert all(ledger['attempts'][k] == v for k, v in parent['attempts'].items())
        assert billing_blocked(ledger) == (mode == 'new_overload')
        assert all(r.read(root / 'fixture/freeze_audit.json').values())
    checks.append({'mode': mode, 'passed': True, 'intercepted_http': len(seen), 'root': str(root)})

for status in (400, 401, 429, 500):
    bad = deepcopy(snap['failed_attempt']); bad['http_status'] = status
    with pytest.raises(AssertionError): run.recover_ledger(parent, bad)
receipt = {'passed': True, 'real_http': 0, 'checks': checks, 'source': r.code_identity(),
    'inherited_entry_unchanged': True, 'original_limits_unchanged': True,
    'original_messages_and_parameters_preserved': True, 'new_overload_stops_without_scoring': True,
    'completed_resume_zero_new_calls': True, 'native_docker_program_consumed': True,
    'driver_sha256': r.sha(c.__file__), 'launcher_sha256': r.sha(run.__file__), 'test_sha256': r.sha(__file__)}
r.write_json(HERE / 'offline_acceptance.json', receipt)
print(json.dumps(receipt))
