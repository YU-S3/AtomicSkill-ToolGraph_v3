"""Actual serializers, native graph, Docker submit, refusals and persistent retry checks; zero HTTP."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import run_models as x
from provider_retry import ProviderRetryStopped

c, r, HERE = x.c, x.r, x.HERE
BASE = Path(tempfile.mkdtemp(prefix='searchqa_gepa_crossmodel_offline_', dir='/home/yangchengyu'))
TASK = r.read(r.DATASETS / 'test.json')['tasks'][0]
checks = []


def fixture(model, mode):
    root = BASE / (model + '_' + mode)
    cfg = r.read(HERE / model / 'base_config.json')
    state = {'limits': {'token_limit': 2000000, 'request_limit': 20, 'finish_reserve': 0},
             'attempts': {}, 'unknown_billing': False}
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, state)
    r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
    r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
    r.write_json(root / 'offline_acceptance.json', {'passed': True, 'driver_sha256': r.sha(c.__file__)})
    r.write_json(root / 'tasks.json', {'tasks': [TASK]})
    spec = r.read(HERE / model / 'spec.json')
    spec['tasks'] = str(root / 'tasks.json')
    r.write_json(root / 'spec.json', spec)
    seen = []

    def post(*args, **kwargs):
        wire = deepcopy(kwargs['json'])
        material = json.loads(wire['messages'][1]['content'])
        assert material['goal'] == TASK['goal'] and material['inputs'] == TASK['inputs']
        assert wire['model'] == cfg['llm']['model'] and wire['max_tokens'] == cfg['llm']['runtime']['max_completion_tokens']
        assert not wire.get('tools') and 'gold' not in material and 'answers' not in material
        assert material['reference_card_ids'] == ['supplied_gepa_seed44_train_facts']
        assert 'Context Snippet Mining' in wire['messages'][0]['content']
        if model == 'deepseek':
            assert wire['thinking'] == {'type': 'enabled'} and wire['reasoning_effort'] == 'high'
        else:
            assert wire['thinking'] == {'type': 'enabled'} and 'reasoning_effort' not in wire and wire['stream']
        seen.append(wire)
        if mode == 'exhaustion' or (mode == 'two_504' and len(seen) <= 2) or (mode == 'refusal' and model == 'kimi'):
            status = 400 if mode == 'refusal' else 504
            data = {'error': {'type': 'content_filter' if status == 400 else 'gateway_timeout', 'message': 'fixture'}}
            return SimpleNamespace(status_code=status, ok=False, headers={}, content=json.dumps(data).encode(),
                text=json.dumps(data), json=lambda: data, close=lambda: None)
        answer = {'role': 'assistant', 'content': '<answer>J. K. Ch\u00e1vez</answer>', 'reasoning_content': 'fixture reasoning'}
        finish = 'content_filter' if mode == 'refusal' else 'stop'
        data = {'id': 'fixture', 'model': cfg['llm']['model'], 'choices': [{'index': 0,
            'delta' if cfg['llm'].get('stream') else 'message': answer, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
        if cfg['llm'].get('stream'):
            lines = [b'data: ' + json.dumps(data).encode()]
            if not (mode == 'incomplete_stream' and len(seen) == 1):
                lines.append(b'data: [DONE]')
            return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lambda **kw: iter(lines), close=lambda: None)
        return SimpleNamespace(status_code=200, ok=True, headers={}, content=json.dumps(data).encode(), text=json.dumps(data), json=lambda: data, close=lambda: None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, 'CAMPAIGN', root)
        c.prepare('fixture', root / 'spec.json')
        patch.setattr(c, 'HERE', root)
        patch.setattr(r, 'load_env', lambda p: None)
        patch.setenv(cfg['llm']['api_key_env'], 'offline')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
        patch.setattr(socket.socket, 'connect', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        patch.setitem(x.POLICY, 'backoff_seconds', [0, 0, 0])
        with (root / 'fixture.log').open('w') as log, redirect_stdout(log):
            if mode == 'exhaustion':
                for _ in range(2):
                    with pytest.raises(ProviderRetryStopped):
                        c.execute('fixture')
                assert len(seen) == 4 and not c.completed(root / 'fixture', x.ARM)
            else:
                c.execute('fixture')
                c.execute('fixture')
                rows = c.completed(root / 'fixture', x.ARM)
                assert len(rows) == 1
                row = rows[TASK['task_id']]
                if mode == 'refusal':
                    assert row['execution']['reason'] == 'provider_content_filter' and not row['score']['hard']
                    assert len(seen) == 1
                else:
                    assert row['execution']['all_authored_nodes_completed']
                    assert row['execution']['prediction'] == '<answer>J.K. Chavez</answer>'
        assert all(wire == seen[0] for wire in seen), 'Retry changed actual payload'
        ledger = r.read(root / 'budget.json')
        assert not r.BudgetGovernor(root / 'budget.json', **ledger['limits']).billing_blocked
        assert all(r.read(root / 'fixture/freeze_audit.json').values())
        assert len(seen) == {'normal': 1, 'two_504': 3, 'exhaustion': 4, 'refusal': 1, 'incomplete_stream': 2}[mode]
        assert r.tree_identity(root / 'fixture' / x.ARM / 'frozen_bank') == r.read(HERE / 'campaign_plan.json')['frozen_identity']
    checks.append({'model': model, 'mode': mode, 'passed': True, 'intercepted_http': len(seen), 'fixture': str(root)})


if __name__ == '__main__':
    for model in x.MODELS:
        for mode in ('normal', 'two_504', 'exhaustion', 'refusal'):
            fixture(model, mode)
        if model == 'kimi':
            fixture(model, 'incomplete_stream')
    receipt = {'passed': True, 'real_http': 0, 'checks': checks, 'source': r.code_identity(),
        'driver_sha256': r.sha(c.__file__), 'files': {name: r.sha(HERE / name) for name in x.CODE_FILES}, 'at': r.utc()}
    r.write_json(HERE / 'offline_acceptance.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))
