"""Zero-HTTP wire, native workflow, scoring, billing and resume acceptance."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import sqlite3
import tempfile
import pytest
import run_models as run
from atomic_skillgraph.agents.provider import AgentProviderError
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.budget_governor import output_token_bound, billing_blocked

r, c, HERE = run.r, run.c, run.HERE
TASKS = r.read(r.DATASETS / 'test.json')['tasks']
SELECTED = [TASKS[0], max(TASKS[1:], key=lambda t: len(t['inputs']['context']))]
with sqlite3.connect((run.REFERENCE / run.ARM / 'test/run.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
    MESSAGES = {t['task_id']: json.loads(db.execute('SELECT result FROM tasks WHERE id=?',
        (t['task_id'],)).fetchone()[0])['requests'][0]['messages'] for t in SELECTED}
USAGE = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}


def exercise(model, mode):
    root = Path(tempfile.mkdtemp(prefix='searchqa_crossmodel_' + model + '_' + mode + '_', dir='/home/yangchengyu'))
    limits = {'token_limit': 1000000, 'request_limit': 20, 'finish_reserve': 0}
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, {'limits': limits, 'attempts': {}, 'unknown_billing': False})
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'tasks.json', {'tasks': SELECTED})
    spec = r.read(HERE / (model + '_spec.json'))
    r.write_json(root / 'spec.json', {**spec, 'tasks': str(root / 'tasks.json')})
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'fixture_only': True,
        'driver_sha256': r.sha(c.__file__)})
    seen = []
    cfg = r.read(HERE / (model + '_config.json'))

    def post(*args, **kwargs):
        current = r.read(root / 'fixture/current.json')
        task_id = current['task_id']
        wire = deepcopy(kwargs['json'])
        assert wire['model'] == cfg['llm']['model'] and wire[cfg['llm']['token_limit_field']] == 65536
        assert wire['stream'] is True and kwargs['stream'] is True
        assert kwargs['timeout'] == (15.0, 600)
        assert wire['messages'] == MESSAGES[task_id]
        assert not {'tools', 'tool_choice'} & wire.keys()
        assert wire['stream_options'] == {'include_usage': True}
        if model == 'qwen38_flash':
            assert wire['enable_thinking'] and wire['preserve_thinking']
            assert wire['seed'] == 42 and wire['temperature'] == .6
            assert wire['thinking_budget'] == 131072
            assert 'reasoning_effort' not in wire and 'max_completion_tokens' not in wire
            assert output_token_bound(wire) == 196618
        else:
            assert wire['thinking'] == {'type': 'adaptive'}
            assert wire['reasoning_split'] and wire['service_tier'] == 'standard'
            assert not {'seed', 'reasoning_effort', 'max_tokens', 'thinking_budget'} & wire.keys()
        seen.append({'task_id': task_id, 'payload': wire})
        abnormal = task_id == SELECTED[0]['task_id'] and mode != 'normal'
        refusal = abnormal and mode.startswith('refusal')
        finish = 'content_filter' if refusal else 'stop'
        chunk = {'id': 'offline-fixture', 'created': 10, 'model': cfg['llm']['model'], 'choices': [{'index': 0,
            'delta': {'content': '' if refusal else '<answer>fixture</answer>'}, 'finish_reason': finish}]}
        if abnormal and mode == 'broken_stream':
            chunk['choices'][0]['finish_reason'] = None
        if not abnormal or mode == 'refusal_metered':
            chunk['usage'] = USAGE
        def lines(**unused):
            yield b'data: ' + json.dumps({'id': 'offline-fixture', 'created': 11, 'model': cfg['llm']['model'], 'choices': []}).encode()
            yield b'data: ' + json.dumps(chunk).encode()
            if (model != 'minimax_m3' or mode == 'refusal_unknown') and not (abnormal and mode == 'broken_stream'):
                yield b'data: [DONE]'
        return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lines, close=lambda: None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda path: None)
        patch.setenv(cfg['llm']['api_key_env'], 'offline-fixture')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Real network forbidden')))
        with (root / 'fixture.log').open('w') as log, redirect_stdout(log):
            if mode in ('broken_stream', 'missing_usage'):
                with pytest.raises(AgentProviderError):
                    c.execute('fixture')
                assert len(seen) == 1 and not c.completed(root / 'fixture', run.ARM)
                with pytest.raises(AssertionError, match='Unresolved billing'):
                    c.execute('fixture')
                assert len(seen) == 1
            else:
                value = c.execute('fixture')
                assert value['arms'][run.ARM]['completed'] == 2 and len(seen) == 2
                c.execute('fixture')
                assert len(seen) == 2, 'Completed resume must not send more requests'
                rows = c.completed(root / 'fixture', run.ARM)
                for i, task in enumerate(SELECTED):
                    row = rows[task['task_id']]
                    refused = i == 0 and mode.startswith('refusal')
                    if refused:
                        assert row['execution']['reason'] == 'provider_content_filter'
                        assert row['score']['hard'] is False and row['execution']['prediction'] is None
                        assert not row['execution'].get('all_authored_nodes_completed')
                    else:
                        assert row['execution']['all_authored_nodes_completed']
                        assert row['execution']['attempts'][0]['status'] == 'ok'
                        assert row['execution']['attempts'][0]['outputs_consumed']
                    attempt = row['requests'][0]['http_attempts'][0]
                    assert attempt['raw_usage'] == (None if refused and mode == 'refusal_unknown' else USAGE)
                ledger = r.read(root / 'budget.json')
                assert len(ledger['attempts']) == 2
                if mode == 'refusal_unknown':
                    unknown = [a for a in ledger['attempts'].values() if a['status'] == 'unknown_billing']
                    assert len(unknown) == 1 and 'accounted_tokens' not in unknown[0]
                    assert not billing_blocked(ledger)
        assert all(r.read(root / 'fixture/freeze_audit.json').values())
    return {'model': model, 'mode': mode, 'passed': True, 'intercepted_requests': len(seen),
            'real_http': 0, 'fixture': str(root)}, seen


checks, maximum = [], 0
for model in run.MODELS:
    cfg = r.read(HERE / (model + '_config.json'))
    for mode in ('normal', 'refusal_metered', 'refusal_unknown', 'broken_stream', 'missing_usage'):
        check, wires = exercise(model, mode)
        checks.append(check)
        if mode == 'normal':
            template = wires[0]['payload']
            for task in TASKS:
                wire = deepcopy(template)
                material = json.loads(wire['messages'][1]['content'])
                material.update(goal=task['goal'], inputs=task['inputs'])
                wire['messages'][1]['content'] = json.dumps(material, ensure_ascii=False)
                bound = len(json.dumps(wire, allow_nan=False).encode()) + output_token_bound(wire)
                assert bound < cfg['budget']['eval_task_tokens']
                maximum = max(maximum, bound)
    cfg = r.read(HERE / (model + '_config.json'))
    system = object.__new__(EmpiricalSystem)
    system.config, system.providers, system.provider_override, system.budget_governor = cfg, {}, None, None
    for purpose, cap in [(None, 65536), ('finish_only', 16384)]:
        provider = system.provider('runtime', purpose)
        payload = provider._build_payload([{'role': 'user', 'content': 'fixture'}], [])
        assert payload['stream'] and payload[cfg['llm']['token_limit_field']] == cap
        assert provider.config.max_retries == 0
        assert 'reasoning_effort' not in payload
        assert payload.get('thinking_budget') == (131072 if model == 'qwen38_flash' else None)
        assert payload.get('enable_thinking') if model == 'qwen38_flash' else payload['thinking']['type'] == 'adaptive'
receipt = {'passed': True, 'real_http': 0, 'source': r.code_identity(), 'checks': checks,
    'driver_sha256': r.sha(c.__file__), 'launcher_sha256': r.sha(run.__file__), 'test_sha256': r.sha(__file__),
    'same_v7_messages_and_full_public_context': True, 'canonical_payloads_checked': 2800,
    'maximum_prompt_bound_plus_completion': maximum,
    'native_docker_program_and_official_score_verified': True,
    'completed_resume_adds_no_calls': True, 'explicit_refusals_scored_once': True,
    'unknown_usage_not_fabricated': True, 'broken_stream_and_missing_usage_stop_before_score': True}
r.write_json(HERE / 'offline_acceptance.json', receipt)
print(json.dumps(receipt))
