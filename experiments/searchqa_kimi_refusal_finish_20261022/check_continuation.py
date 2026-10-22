"""Offline production replay and strict HTTP-refusal classification regressions."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import tempfile
import pytest
import requests
import continue_kimi as run
from atomic_skillgraph.agents.provider import AgentProviderError

r, c, HERE = run.r, run.c, run.HERE
SPEC = r.read(HERE / 'spec.json')
TASKS = r.read(SPEC['tasks'])['tasks'][:2]
SNAP = r.read(HERE / 'parent_snapshot.json')
ORIGINAL = SNAP['refused_attempt']
USAGE = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}


def classification_checks():
    before = r.digest(ORIGINAL)
    assert run.outcomes.refusal_evidence(ORIGINAL) == {
        'source': 'http_error', 'http_status': 400, 'error_type': 'content_filter',
        'error_code': '400', 'finish_reason': None}
    rejected = 0
    for status, dialect, error_type in [(400, 'moonshot_k2_chat', 'invalid_request_error'),
                                      (400, 'moonshot_k2_chat', 'rate_limit_error'),
                                      (401, 'moonshot_k2_chat', 'content_filter'),
                                      (500, 'moonshot_k2_chat', 'content_filter'),
                                      (400, 'glm_53_chat', 'content_filter')]:
        item = deepcopy(ORIGINAL)
        item['http_status'] = status
        item['final_payload_audit']['provider_snapshot']['dialect'] = dialect
        item['response_diagnostic']['error']['type'] = error_type
        assert run.outcomes.refusal_evidence(item) is None
        rejected += 1
    for dialect, finish in [('moonshot_k2_chat', 'content_filter'), ('glm_53_chat', 'sensitive')]:
        item = deepcopy(ORIGINAL)
        item.update(http_status=200, public_response={'choices': [{'finish_reason': finish}]},
                    response_diagnostic={'stream': {'completed': True}})
        item['final_payload_audit']['provider_snapshot']['dialect'] = dialect
        assert run.outcomes.refusal_evidence(item)['finish_reason'] == finish
        item['response_diagnostic']['stream']['completed'] = False
        assert run.outcomes.refusal_evidence(item) is None
    with pytest.raises(ValueError, match='identity mismatch'):
        run.outcomes.last_refusal({'requests': [{'http_attempts': [ORIGINAL]}]}, 'wrong-task')
    assert r.digest(ORIGINAL) == before
    return {'negative_http_cases': rejected, 'original_evidence_immutable': True,
            'existing_glm_and_kimi_stream_refusals_preserved': True}


def exercise(mode):
    root = Path(tempfile.mkdtemp(prefix='kimi_final10_' + mode + '_', dir='/home/yangchengyu'))
    limits = {'token_limit': 1000000, 'request_limit': 20, 'finish_reserve': 0}
    ledger = {'limits': limits, 'attempts': {}, 'unknown_billing': False}
    spec = {**SPEC, 'tasks': str(root / 'tasks.json'), 'replay_sources': {}}
    if mode == 'original_replay':
        parent = r.read(run.PARENT / 'budget.json')
        ledger['attempts'] = {ORIGINAL['request_id']: deepcopy(parent['attempts'][ORIGINAL['request_id']])}
        ledger['unknown_billing'] = True
        spec['replay_sources'] = SPEC['replay_sources']
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, ledger)
    if mode == 'original_replay':
        governor = r.BudgetGovernor(root / 'budget.json', **limits)
        run.outcomes.retain_reservation(governor, ORIGINAL)
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'tasks.json', {'tasks': TASKS})
    r.write_json(root / 'spec.json', spec)
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'fixture_only': True,
        'driver_sha256': r.sha(c.__file__)})
    seen = []

    def post(*args, **kwargs):
        current = r.read(root / 'fixture/current.json')
        wire = kwargs['json']
        assert wire['model'] == 'kimi-k2.6' and wire['max_tokens'] == 65536
        assert wire['stream'] and wire['thinking'] == {'type': 'enabled'}
        assert 'reasoning_effort' not in wire
        seen.append(current['task_id'])
        abnormal = len(seen) == 1 and mode not in ('normal', 'original_replay')
        if abnormal and mode != 'broken_stream':
            response = requests.Response()
            response.status_code = 500 if mode == 'server_error' else 400
            response._content = json.dumps({'error': {
                'code': str(response.status_code),
                'type': 'content_filter' if mode in ('http_refusal', 'server_error') else 'invalid_request_error',
                'message': 'Offline fixture error'}}).encode()
            response.headers = {'Content-Type': 'application/json'}
            return response
        chunk = {'id': 'offline-fixture', 'model': 'kimi-k2.6',
                 'choices': [{'index': 0, 'delta': {'content': '<answer>fixture</answer>'}, 'finish_reason': 'stop'}],
                 'usage': USAGE}
        if abnormal:
            chunk.pop('usage')
        def lines(**unused):
            yield b'data: ' + json.dumps(chunk).encode()
            if not abnormal:
                yield b'data: [DONE]'
        return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lines, close=lambda: None)

    succeeds = mode in ('normal', 'original_replay', 'http_refusal')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda path: None)
        patch.setenv('MOONSHOT_API_KEY', 'offline-fixture')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        with (root / 'fixture.log').open('w') as output, redirect_stdout(output):
            if succeeds:
                value = c.execute('fixture')
                assert value['arms'][run.ARM]['completed'] == 2
                expected = 1 if mode == 'original_replay' else 2
                assert len(seen) == expected
                c.execute('fixture')
                assert len(seen) == expected
                rows = c.completed(root / 'fixture', run.ARM)
                if mode != 'normal':
                    row = rows[TASKS[0]['task_id']]
                    assert row['score']['hard'] is False and row['execution']['prediction'] is None
                    assert row['execution']['provider_finish_reason'] is None
                    assert row['execution']['provider_refusal_evidence']['source'] == 'http_error'
                    assert row['usage'] == [] and len(row['requests']) == 1
                    attempt = row['requests'][0]['http_attempts'][0]
                    assert attempt['raw_usage'] is None and attempt['public_response'] is None
                    if mode == 'original_replay':
                        assert attempt == ORIGINAL and TASKS[0]['task_id'] not in seen
                assert rows[TASKS[1]['task_id']]['execution']['all_authored_nodes_completed']
                assert rows[TASKS[1]['task_id']]['execution']['attempts'][0]['status'] == 'ok'
                assert not value['billing_blocked']
            else:
                with pytest.raises(AgentProviderError):
                    c.execute('fixture')
                assert len(seen) == 1 and not c.completed(root / 'fixture', run.ARM)
                with pytest.raises(AssertionError, match='Unresolved billing'):
                    c.execute('fixture')
                assert len(seen) == 1
        assert all(r.read(root / 'fixture/freeze_audit.json').values())
        after = r.read(root / 'budget.json')
        assert all(after['attempts'][k] == v for k, v in ledger['attempts'].items())
        if mode == 'original_replay':
            entry = after['attempts'][ORIGINAL['request_id']]
            assert entry['reserved_tokens'] == 77429 and 'accounted_tokens' not in entry
    return {'mode': mode, 'passed': True, 'intercepted_requests': len(seen), 'real_http': 0, 'fixture': str(root)}


classification = classification_checks()
checks = [exercise(mode) for mode in ('normal', 'original_replay', 'http_refusal', 'invalid_request', 'server_error', 'broken_stream')]
receipt = {'passed': True, 'real_http': 0, 'checks': checks, 'classification': classification,
           'source': r.code_identity(), 'driver_sha256': r.sha(c.__file__),
           'launcher_sha256': r.sha(run.__file__), 'guard_sha256': r.sha(run.outcomes.__file__),
           'test_sha256': r.sha(__file__), 'original_refusal_replayed_without_request': True,
           'original_unknown_charge_preserved': 77429, 'actual_usage_not_fabricated': True}
r.write_json(HERE / 'offline_acceptance.json', receipt)
print(json.dumps(receipt))
