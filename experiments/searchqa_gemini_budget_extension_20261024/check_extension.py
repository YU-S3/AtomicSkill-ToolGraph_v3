"""Zero-HTTP verification of the exact authorized budget change and continuation driver."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import budget_extension as x

c, r = x.c, x.r
root = Path(tempfile.mkdtemp(prefix='gemini_budget_extension_offline_', dir='/home/yangchengyu'))
old = r.read(x.PARENT / 'budget.json')
state = deepcopy(old)
assert state['limits']['token_limit'] == 10272289
state['limits']['token_limit'] = 10772289
assert state['limits']['request_limit'] == 2257
for name in ('budget.json', 'inherited_budget.json'):
    r.write_json(root / name, state)
r.write_json(root / 'authorization.json', {'status': 'approved', 'fixture_only': True})
r.write_json(root / 'cost_carryforward_manifest.json', {'fixture_only': True})
count = len(x.prefix())
tasks = r.read(r.DATASETS / 'test.json')['tasks']
assert set(x.prefix()) == {t['task_id'] for t in tasks[:count]}
task = tasks[count]
r.write_json(root / 'tasks.json', {'tasks': [task]})
spec = {'tasks': str(root / 'tasks.json'), 'base_config': str(x.PARENT / x.STAGE / x.ARM / 'config.json'),
    'arms': {x.ARM: {'kind': 'native', 'package': str(x.PARENT / x.STAGE / x.ARM / 'input'),
                     'frozen_source': str(x.PARENT / x.STAGE / x.ARM / 'frozen_bank')}},
    'purpose': 'Offline authorized cumulative budget amendment verification; no real HTTP.'}
r.write_json(root / 'spec.json', spec)
seen = []


def post(*args, **kwargs):
    wire = deepcopy(kwargs['json'])
    material = json.loads(wire['messages'][1]['content'])
    assert material['goal'] == task['goal'] and material['inputs'] == task['inputs']
    assert wire['max_tokens'] == 65536 and wire['model'] == 'gemini-3.8-flash'
    assert wire['reasoning_effort'] == 'high' and wire.get('tools', []) == []
    seen.append(wire)
    chunk = {'id': 'offline', 'model': wire['model'], 'choices': [
        {'index': 0, 'delta': {'content': '<answer>Fixture answer</answer>'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
    lines = [b'data: ' + json.dumps(chunk).encode(), b'data: [DONE]']
    return SimpleNamespace(status_code=200, ok=True, headers={}, iter_lines=lambda **kw: iter(lines), close=lambda: None)


with pytest.MonkeyPatch.context() as patch:
    patch.setattr(c, 'CAMPAIGN', root)
    c.prepare('fixture', root / 'spec.json')
    patch.setattr(r, 'load_env', lambda p: None)
    patch.setenv('ROOTFLOWAI_API_KEY', 'offline')
    patch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    patch.setattr(socket.socket, 'connect', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Network forbidden')))
    with (root / 'fixture.log').open('w') as log, redirect_stdout(log):
        c.execute('fixture')
        c.execute('fixture')
    assert len(seen) == 1
    rows = c.completed(root / 'fixture', x.ARM)
    assert len(rows) == 1 and rows[task['task_id']]['execution']['all_authored_nodes_completed']
    ledger = r.read(root / 'budget.json')
    assert all(ledger['attempts'][key] == value for key, value in old['attempts'].items())
    assert ledger['limits'] == state['limits']
    for key in ('provider_retries', 'transport_failures', 'transport_continuations', 'inherited_reservation_recovery'):
        assert ledger.get(key) == old.get(key)
    assert all(r.read(root / 'fixture/freeze_audit.json').values())
    cfg = r.read(root / 'fixture' / x.ARM / 'config.json')
    expected = r.read(x.PARENT / x.STAGE / x.ARM / 'config.json')
    expected['budget']['token_limit'] = 10772289
    expected['data_dir'] = cfg['data_dir']
    expected['experiment']['output_dir'] = cfg['experiment']['output_dir']
    assert cfg == expected

prior = r.read(x.RUNTIME / 'offline_acceptance.json')
for name, sha in prior['files'].items():
    assert r.sha(x.RUNTIME / name) == sha
receipt = {'passed': True, 'real_http': 0, 'intercepted_http': 1, 'fixture': str(root),
    'extension_sha256': r.sha(x.HERE / 'budget_extension.py'), 'runtime_files': prior['files'],
    'checks': ['exact_500000_token_increase', 'same_http_limit', 'old_attempts_unchanged',
               'retry_counters_and_unknown_reservations_preserved', 'canonical_prefix_and_remaining_ids',
               'identical_normal_request_parameters', 'actual_native_graph_and_docker_submit', 'completed_resume_zero_http', 'frozen_hash_unchanged'],
    'retained_completed': count, 'source': r.code_identity(), 'at': r.utc()}
r.write_json(x.HERE / 'offline_acceptance.json', receipt)
print(json.dumps(receipt, ensure_ascii=False))
