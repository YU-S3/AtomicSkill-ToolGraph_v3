"""Fault injection through the real provider, native graph, budget ledger and Docker worker."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import requests

import continue_test as x
from infrastructure_guard import DockerUnavailable
from provider_retry import ProviderRetryStopped, transient
from atomic_skillgraph.empirical.program_worker import ProgramWorker

c, r, HERE = x.c, x.r, x.HERE
BASE = Path(tempfile.mkdtemp(prefix='gemini_transport_offline_', dir='/home/yangchengyu'))
SNAP = r.read(HERE / 'parent_snapshot.json')
TASK = r.read(HERE / 'remaining_tasks.json')['tasks'][0]
checks = []


def response(status=200, incomplete=False, usage=True):
    if status != 200:
        data = {'error': {'type': 'gateway_timeout' if status == 504 else 'authentication_error', 'message': 'fixture'}}
        return SimpleNamespace(status_code=status, ok=False, headers={}, content=json.dumps(data).encode(),
                               text=json.dumps(data), json=lambda: data, close=lambda: None)
    chunk = {'id': 'offline', 'model': 'gemini-3.8-flash', 'choices': [
        {'index': 0, 'delta': {'content': '<answer>Fixture answer</answer>'}, 'finish_reason': 'stop'}]}
    if usage:
        chunk['usage'] = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}
    lines = [b'data: ' + json.dumps(chunk).encode()]
    if not incomplete:
        lines.append(b'data: [DONE]')
    return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lambda **kw: iter(lines), close=lambda: None)


def fixture(mode):
    root = BASE / mode
    state = r.read(x.PARENT / 'budget.json') if mode == 'historical_resume' else {
        'limits': {'token_limit': 110000 if mode == 'global_cap' else 2000000, 'request_limit': 30, 'finish_reserve': 0},
        'attempts': {}, 'unknown_billing': False}
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, state)
    if mode == 'historical_resume':
        governor = r.BudgetGovernor(root / 'budget.json', **state['limits'])
        x.migrate(governor, SNAP)
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'driver_sha256': r.sha(c.__file__)})
    r.write_json(root / 'tasks.json', {'tasks': [TASK]})
    spec = r.read(HERE / 'spec.json')
    spec['tasks'] = str(root / 'tasks.json')
    r.write_json(root / 'spec.json', spec)
    seen, context_seen = [], []
    stopped_once = False

    def post(*args, **kwargs):
        nonlocal stopped_once
        wire = deepcopy(kwargs['json'])
        material = json.loads(wire['messages'][1]['content'])
        assert material['goal'] == TASK['goal'] and material['inputs'] == TASK['inputs']
        assert 'gold' not in material and 'answers' not in material
        assert wire['max_tokens'] == 65536 and wire['model'] == 'gemini-3.8-flash'
        assert wire['reasoning_effort'] == 'high' and wire.get('tools', []) == []
        seen.append(wire)
        ledger = r.read(root / 'budget.json')
        context_seen.append(next(v['context'] for v in reversed(list(ledger['attempts'].values())) if v['status'] == 'admitted'))
        if mode == 'timeout' and len(seen) == 1:
            raise requests.exceptions.ReadTimeout('Offline read timeout fixture')
        if mode == 'restart' and not stopped_once:
            stopped_once = True
            (root / 'STOP_AFTER_TASK').touch()
            return response(504)
        fail = (mode == 'two_504' and len(seen) <= 2) or (mode == 'historical_resume' and len(seen) == 1)
        fail |= mode in ('exhaustion', 'global_cap')
        if fail:
            return response(504)
        if mode in ('partial_stream', 'unknown_stream') and len(seen) == 1:
            return response(incomplete=True, usage=mode == 'partial_stream')
        return response()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda p: None)
        patch.setenv('ROOTFLOWAI_API_KEY', 'offline')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        patch.setitem(x.POLICY, 'backoff_seconds', [0, 0, 0])
        original_execute = ProgramWorker.execute
        docker_failures = []
        if mode == 'docker_resume':
            def worker(*args, **kwargs):
                if not docker_failures:
                    docker_failures.append(True)
                    return {'status': 'execution_error', 'diagnostic': 'Cannot connect to the Docker daemon',
                            'calls': 0, 'cpu_seconds': None, 'execution_receipt': {'exit_code': 125}}
                return original_execute(*args, **kwargs)
            patch.setattr(ProgramWorker, 'execute', worker)
        with (root / 'fixture.log').open('w') as log, redirect_stdout(log):
            if mode in ('exhaustion', 'global_cap'):
                for _ in range(2):
                    with pytest.raises(ProviderRetryStopped):
                        c.execute('fixture')
                assert len(seen) == (4 if mode == 'exhaustion' else 1)
                assert not c.completed(root / 'fixture', 'gepa')
            else:
                if mode == 'restart':
                    with pytest.raises(ProviderRetryStopped):
                        c.execute('fixture')
                    (root / 'STOP_AFTER_TASK').unlink()
                if mode == 'docker_resume':
                    with pytest.raises(DockerUnavailable):
                        c.execute('fixture')
                c.execute('fixture')
                before_resume = len(seen)
                c.execute('fixture')
                assert len(seen) == before_resume
                rows = c.completed(root / 'fixture', 'gepa')
                assert len(rows) == 1 and next(iter(rows.values()))['execution']['all_authored_nodes_completed']
        assert all(wire == seen[0] for wire in seen), 'Retry changed serialized payload'
        ledger = r.read(root / 'budget.json')
        assert all(ledger['attempts'][k] == v for k, v in state['attempts'].items())
        assert ledger['limits'] == state['limits']
        assert all(r.read(root / 'fixture/freeze_audit.json').values())
        assert not r.BudgetGovernor(root / 'budget.json', **ledger['limits']).billing_blocked
        if mode == 'historical_resume':
            assert len(seen) == 2 and [v['infrastructure_retry_index'] for v in context_seen] == [2, 3]
            retry = ledger['provider_retries'][SNAP['binding']['retry_key']]
            assert retry['failures'][:2] == SNAP['binding']['initial_failures'] and len(retry['failures']) == 3
        added = [v for k, v in ledger['attempts'].items() if k not in state['attempts']]
        if mode == 'partial_stream':
            assert sum(v.get('accounted_tokens', 0) for v in added) == 10
            assert all(v['status'] == 'metered' for v in added)
        for key, receipt in ledger.get('transport_failures', {}).items():
            assert receipt['entry_hash'] == r.digest(ledger['attempts'][key])
        expected = {'normal': 1, 'two_504': 3, 'restart': 2, 'historical_resume': 2, 'docker_resume': 1,
                    'exhaustion': 4, 'global_cap': 1, 'partial_stream': 2, 'unknown_stream': 2, 'timeout': 2}[mode]
        assert len(seen) == expected
    r.write_json(root / 'intercepted_requests.json', seen)
    checks.append({'mode': mode, 'passed': True, 'intercepted_http': len(seen), 'fixture': str(root)})


if __name__ == '__main__':
    for mode in ('normal', 'two_504', 'restart', 'historical_resume', 'exhaustion', 'global_cap',
                 'partial_stream', 'unknown_stream', 'timeout', 'docker_resume'):
        fixture(mode)
    for status in (400, 401, 403, 422):
        assert not transient({'http_status': status}, RuntimeError())
    assert not transient({'http_status': 429, 'response_diagnostic': {'error': {'type': 'insufficient_quota'}}}, RuntimeError())
    assert not transient({'http_status': 200, 'public_response': {'choices': [{'finish_reason': 'content_filter'}]}}, RuntimeError())
    checks.append({'mode': 'no_retry_auth_quota_or_refusal', 'passed': True, 'intercepted_http': 0})
    receipt = {'passed': True, 'real_http': 0, 'source': r.code_identity(), 'driver_sha256': r.sha(c.__file__),
               'checks': checks, 'files': {name: r.sha(HERE / name) for name in x.CODE_FILES}, 'at': r.utc()}
    r.write_json(HERE / 'offline_acceptance.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))
