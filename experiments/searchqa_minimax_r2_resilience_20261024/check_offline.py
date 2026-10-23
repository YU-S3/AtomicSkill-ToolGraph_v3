"""Production-provider and native-graph fault injection; network is forbidden."""
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
from atomic_skillgraph.agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from provider_outcomes import last_refusal, refusal_evidence, retain_reservation
from provider_retry import ProviderRetryStopped, retry_guard, transient

c, r, HERE = x.c, x.r, x.HERE
BASE = Path(tempfile.mkdtemp(prefix='minimax_resilience_checks_', dir='/home/yangchengyu'))
snap = r.read(HERE / 'parent_snapshot.json')
tasks = r.read(HERE / 'remaining_tasks.json')['tasks'][:2]
checks = []


def response(status=200, message='input new_sensitive (1026)', error_type='unprocessable_entity_error'):
    error = {'error': {'type': error_type, 'message': message}}
    chunk = {'id': 'offline', 'model': 'MiniMax-M3', 'choices': [
        {'index': 0, 'delta': {'content': '<answer>A &amp; B</answer>'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
    return SimpleNamespace(status_code=status, ok=status == 200, headers={}, content=json.dumps(error).encode(),
        text=json.dumps(error), json=lambda: error,
        iter_lines=lambda **unused: iter([b'data: ' + json.dumps(chunk).encode()]), close=lambda: None)


def fixture(mode):
    root = BASE / mode
    initial = r.read(x.PARENT / 'budget.json') if mode == 'saved_refusal' else {
        'limits': {'token_limit': 1000000, 'request_limit': 20, 'finish_reserve': 0},
        'attempts': {}, 'unknown_billing': False}
    for name in ['budget.json', 'inherited_budget.json']:
        r.write_json(root / name, initial)
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'tasks.json', {'tasks': tasks})
    spec = {'tasks': str(root / 'tasks.json'), 'base_config': str(HERE / 'base_config.json'),
        'arms': {'r2': {'kind': 'native', 'package': str(x.PARENT / x.STAGE / 'r2/input'),
                        'frozen_source': str(x.PARENT / x.STAGE / 'r2/frozen_bank')}}, 'purpose': 'Zero-HTTP fault injection'}
    if mode == 'saved_refusal':
        audit = r.read(snap['failed_audit_path'])
        spec['replay_sources'] = {'r2': {tasks[0]['task_id']: {'path': snap['failed_audit_path'], 'audit_digest': r.digest(audit)}}}
        governor = r.BudgetGovernor(root / 'budget.json', **initial['limits'])
        retain_reservation(governor, snap['failed_attempt'])
    r.write_json(root / 'spec.json', spec)
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'driver_sha256': r.sha(c.__file__)})
    seen = []
    worker_failures = []
    original_worker = ProgramWorker.execute

    def post(*args, **kwargs):
        wire = deepcopy(kwargs['json'])
        current = r.read(root / 'fixture/current.json')['task_id']
        material = json.loads(wire['messages'][1]['content'])
        task = next(t for t in tasks if t['task_id'] == current)
        assert material['inputs'] == task['inputs'] and material['final_question_to_answer'] == task['goal']
        assert wire['model'] == 'MiniMax-M3' and wire['thinking'] == {'type': 'adaptive'}
        seen.append(wire)
        if len(seen) == 1:
            if mode == 'new_refusal': return response(422)
            if mode == 'native_refusal':
                value = response()
                value.iter_lines = lambda **unused: iter([b'data: ' + json.dumps(
                    {'base_resp': {'status_code': 1027, 'status_msg': 'output new_sensitive'}}).encode()])
                return value
            if mode == 'unknown_422': return response(422, 'invalid parameter')
            if mode == 'auth': return response(401, 'invalid API key', 'authentication_error')
            if mode == 'overload': return response(529, 'overloaded', 'overloaded_error')
            if mode == 'timeout': raise requests.exceptions.ReadTimeout('offline timeout')
        return response()

    def worker(self, *args, **kwargs):
        if mode == 'docker_after_response' and not worker_failures:
            worker_failures.append(True)
            return {'status': 'execution_error', 'diagnostic': 'Cannot connect to the Docker daemon',
                    'execution_receipt': {'exit_code': 125}, 'calls': 0, 'cpu_seconds': None}
        return original_worker(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda p: None)
        patch.setenv('MINIMAX_API_KEY', 'offline')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        patch.setattr('provider_retry.time.sleep', lambda seconds: None)
        patch.setattr(ProgramWorker, 'execute', worker)
        with (root / 'fixture.log').open('w') as log, redirect_stdout(log):
            if mode in ('unknown_422', 'auth'):
                with pytest.raises(Exception): c.execute('fixture')
            elif mode == 'docker_after_response':
                with pytest.raises(x.DockerUnavailable): c.execute('fixture')
                c.execute('fixture')
                c.execute('fixture')
            else:
                c.execute('fixture')
                c.execute('fixture')
        rows = c.completed(root / 'fixture', 'r2')
        ledger = r.read(root / 'budget.json')
        assert all(ledger['attempts'].get(k) == v for k, v in initial['attempts'].items())
        if mode in ('unknown_422', 'auth'):
            assert not rows and len(seen) == 1
        else:
            assert len(rows) == 2 and all(r.read(root / 'fixture/freeze_audit.json').values())
            if mode in ('new_refusal', 'native_refusal', 'saved_refusal'):
                first = rows[tasks[0]['task_id']]
                assert first['score']['hard'] is False and first['execution']['reason'] == 'provider_content_filter'
                assert rows[tasks[1]['task_id']]['execution']['all_authored_nodes_completed']
                assert len(seen) == (1 if mode == 'saved_refusal' else 2)
            else:
                assert all(row['execution']['all_authored_nodes_completed'] for row in rows.values())
                assert len(seen) == (3 if mode in ('overload', 'timeout') else 2)
            if mode in ('overload', 'timeout'):
                assert seen[0]['messages'] == seen[1]['messages']
                assert {k:v for k,v in seen[0].items() if k != 'max_completion_tokens'} == {k:v for k,v in seen[1].items() if k != 'max_completion_tokens'}
                assert 1024 <= seen[1]['max_completion_tokens'] < seen[0]['max_completion_tokens'] == 65536
                assert seen[2]['max_completion_tokens'] == 65536
                assert sum(a['status'] == 'unknown_billing' for a in ledger['attempts'].values()) == 1
                assert not r.BudgetGovernor(root / 'budget.json', **initial['limits']).billing_blocked
    r.write_json(root / 'intercepted_requests.json', seen)
    checks.append({'mode': mode, 'passed': True, 'intercepted_http': len(seen), 'fixture': str(root)})


def bounded_retries(mode):
    root = BASE / mode
    governor = r.BudgetGovernor(root / 'budget.json', token_limit=1000000, request_limit=20, finish_reserve=0)
    cfg = OpenAICompatibleConfig(base_url='https://api.minimax.cn/v1', model='MiniMax-M3', api_key_env='MINIMAX_API_KEY',
        max_completion_tokens=2048, token_limit_field='max_completion_tokens', dialect='minimax_m3_chat', stream=True, max_retries=0)
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        return response(503, 'busy', 'overloaded_error')
    def provider():
        p = OpenAICompatibleProvider(cfg)
        p.budget_governor = governor
        p.set_request_context(session_id='offline-session', stage='runtime', parent_scope=str(root), parent_task_id='fixture-task', logical_decision_id='same-decision',
                              purpose='runtime', parent_token_limit=128000 if mode == 'retry_limit' else 3000)
        return p
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('MINIMAX_API_KEY', 'offline')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr('provider_retry.time.sleep', lambda s: None)
        patch.setattr(socket.socket, 'connect', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        with retry_guard(governor, root / 'events'):
            with pytest.raises(ProviderRetryStopped): provider().complete([{'role': 'user', 'content': 'fixture'}])
            before = len(calls)
            with pytest.raises(ProviderRetryStopped): provider().complete([{'role': 'user', 'content': 'fixture'}])
            assert before == len(calls) == (4 if mode == 'retry_limit' else 1)
    assert all(v['status'] == 'unknown_billing' and 'accounted_tokens' not in v for v in governor.state['attempts'].values())
    checks.append({'mode': mode, 'passed': True, 'intercepted_http': len(calls), 'resume_additional_http': 0})


actual = snap['failed_attempt']
assert refusal_evidence(actual)['error_code'] == 1026
for code in [1026, 1027]:
    a = deepcopy(actual)
    a['response_diagnostic']['error']['message'] = ('input' if code == 1026 else 'output') + f' new_sensitive ({code})'
    assert refusal_evidence(a)['error_code'] == code
    assert not transient(a, RuntimeError())
a = deepcopy(actual)
a['response_diagnostic']['error']['message'] = 'other failure'
assert refusal_evidence(a) is None
a['response_diagnostic']['error']['message'] = 'insufficient balance (1008)'
a['http_status'] = 429
assert not transient(a, RuntimeError())
for mode in ['saved_refusal', 'new_refusal', 'native_refusal', 'overload', 'timeout', 'unknown_422', 'auth', 'docker_after_response']:
    fixture(mode)
for mode in ['retry_limit', 'retry_budget']:
    bounded_retries(mode)
files = ['campaign.py', 'runtime.py', 'provider_outcomes.py', 'provider_retry.py', 'infrastructure_guard.py',
         'continue_test.py', 'check_offline.py']
receipt = {'passed': True, 'real_http': 0, 'checks': checks, 'driver_sha256': r.sha(c.__file__),
           'files': {name: r.sha(HERE / name) for name in files}, 'source': r.code_identity(), 'at': r.utc()}
r.write_json(HERE / 'offline_acceptance.json', receipt)
print(json.dumps(receipt, ensure_ascii=False))
