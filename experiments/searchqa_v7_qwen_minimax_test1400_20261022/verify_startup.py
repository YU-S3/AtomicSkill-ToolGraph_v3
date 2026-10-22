"""Verify real paid prefix identity, raw usage, Program consumption and billing."""
from collections import Counter
import json
import run_models as run

r, c = run.r, run.c
reference = c.completed(run.REFERENCE, run.ARM)
checked = {}
for model in run.MODELS:
    root = run.select(model)
    c.check(run.STAGE)
    value = run.monitor(model)
    assert value['status'] == 'running' and value['process_alive'] and value['error'] is None
    rows = c.completed(root / run.STAGE, run.ARM)
    assert len(rows) >= 3
    snapshot = r.read(root / run.STAGE / run.ARM / 'config.json')
    samples = []
    for task_id, row in list(rows.items())[:3]:
        assert row['requests'][0]['messages'] == reference[task_id]['requests'][0]['messages']
        attempts = [a for q in row['requests'] for a in q['http_attempts']]
        assert len(attempts) == 1
        attempt = attempts[0]
        params = attempt['final_payload_audit']['serialized_parameters']
        assert params['model'] == snapshot['llm']['model'] == attempt['public_response']['model']
        assert params[snapshot['llm']['token_limit_field']] == 65536
        assert 'reasoning_effort' not in params
        if model == 'qwen38_flash':
            assert params['thinking_budget'] == 131072 and params['seed'] == 42
            assert params['enable_thinking'] and params['preserve_thinking'] and params['temperature'] == .6
        else:
            assert params['thinking']['type'] == 'adaptive' and params['reasoning_split']
            assert params['service_tier'] == 'standard' and 'seed' not in params
        assert attempt['outcome'] == 'success' and attempt['http_status'] == 200
        assert attempt['raw_usage']['total_tokens'] == sum(u['total_tokens'] for u in row['usage'])
        assert row['execution']['all_authored_nodes_completed']
        assert all(a['status'] == 'ok' and a['outputs_consumed'] for a in row['execution']['attempts'])
        samples.append({'task_id': task_id, 'physical_request_id': attempt['request_id'],
            'response_model': attempt['public_response']['model'],
            'raw_usage': attempt['raw_usage'], 'actual_parameters': params,
            'same_messages_as_reference': True, 'native_program_consumed': True})
    ledger, inherited = r.read(root / 'budget.json'), r.read(root / 'inherited_budget.json')
    assert all(ledger['attempts'][k] == v for k, v in inherited['attempts'].items())
    added = {k: v for k, v in ledger['attempts'].items() if k not in inherited['attempts']}
    assert all(v['status'] in ('admitted', 'metered') for v in added.values())
    checked[model] = {'snapshot': value, 'samples': samples,
        'prior_billing_preserved': True, 'new_attempt_states': dict(Counter(v['status'] for v in added.values())),
        'plan_sha256': r.read(root / run.STAGE / 'plan_identity.json')['sha256']}
receipt = {'passed': True, 'at': r.utc(), 'real_http_from_this_check': 0,
           'source': r.code_identity(), 'models': checked}
r.write_json(run.HERE / 'startup_verification.json', receipt)
r.write_json(run.ROOT / 'startup_verification.json', receipt)
print(json.dumps({'passed': True, 'at': receipt['at'], 'completed': {
    model: value['snapshot']['arms'][run.ARM]['completed'] for model, value in checked.items()}}))
