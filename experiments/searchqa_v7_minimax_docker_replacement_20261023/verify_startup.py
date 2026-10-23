"""Check the first three real replacement tasks without additional HTTP."""
import json
import run_replacement as run

r,c = run.r,run.c
for stage in run.STAGES: c.check(stage)
manifest = r.read(run.HERE/'replacement_manifest.json')
old_rows = run.parent_rows()
assert r.digest(old_rows) == manifest['parent_rows_hash']
assert r.sha(run.PARENT/'budget.json') == manifest['parent_budget_sha256']
assert r.sha(run.FIRST/'budget.json') == manifest['first_budget_sha256']
assert r.sha(run.FIRST.parent/'qwen38_flash'/run.OLD_STAGE/'result.json') == manifest['qwen_result_sha256']
rows = c.completed(run.ROOT/'replacement200',run.ARM)
tasks = r.read(run.HERE/'replacement200_tasks.json')['tasks']
assert len(rows) >= 3, 'Wait for three completed real tasks.'
ledger = r.read(run.ROOT/'budget.json')
parent = r.read(run.PARENT/'budget.json')
assert ledger['limits'] == parent['limits']
assert all(ledger['attempts'][k] == v for k,v in parent['attempts'].items())
assert all(v['status'] in ('metered','admitted') for k,v in ledger['attempts'].items() if k not in parent['attempts'])
samples = []
for task in tasks[:3]:
    row = rows[task['task_id']]
    calls = [a for q in row['requests'] for a in q.get('http_attempts',[])]
    assert len(calls) == 1
    attempt = calls[0]
    wire = attempt['final_payload_audit']['serialized_parameters']
    assert attempt['http_status'] == 200 and attempt['outcome'] == 'success'
    assert wire['model'] == 'MiniMax-M3' and wire['max_completion_tokens'] == 65536
    assert wire['thinking'] == {'type':'adaptive'} and attempt['public_response']['model'] == 'MiniMax-M3'
    tokens = attempt['raw_usage']['total_tokens']
    assert tokens == sum(u['total_tokens'] for u in row['usage'])
    assert tokens == ledger['attempts'][attempt['request_id']]['accounted_tokens']
    assert row['requests'][0]['messages'] == old_rows[task['task_id']]['requests'][0]['messages']
    assert row['execution']['all_authored_nodes_completed']
    assert row['execution']['attempts'] and all(a['status']=='ok' and a['outputs_consumed'] for a in row['execution']['attempts'])
    samples.append({'task_id':task['task_id'],'request_id':attempt['request_id'],
                    'raw_usage':attempt['raw_usage'],'parameters':wire,'program_consumed':True})
value = run.status()
assert value['alive'] and value['stages']['replacement200']['status'] == 'running'
assert value['unknown_reserved_tokens'] == 154912
receipt = {'passed':True,'at':r.utc(),'real_http_from_check':0,'snapshot':value,'samples':samples,
           'parents_and_qwen_unchanged':True,'source':r.code_identity(),
           'same_public_messages_bank_model':True,'guard_sha256':r.sha(run.HERE/'infrastructure_guard.py')}
r.write_json(run.HERE/'startup_verification.json',receipt)
print(json.dumps({'passed':True,'progress':value['stages'],'effective':value['effective']}))
