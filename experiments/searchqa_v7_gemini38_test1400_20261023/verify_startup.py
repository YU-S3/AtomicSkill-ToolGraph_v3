"""Read-only real prefix verification without extra paid requests."""
import json
import run_gemini as run

r,c = run.r,run.c
c.check(run.STAGE)
rows = c.completed(run.ROOT/run.STAGE,run.ARM)
tasks = r.read(run.ROOT/run.STAGE/'tasks.json')['tasks']
assert len(rows)>=3,'Wait for the first three completed tasks'
reference = c.completed(run.REFERENCE,run.ARM)
ledger = r.read(run.ROOT/'budget.json')
assert all(a['status'] in ('metered','admitted') for a in ledger['attempts'].values())
samples = []
for task in tasks[:3]:
    row = rows[task['task_id']]
    calls = [a for q in row['requests'] for a in q.get('http_attempts',[])]
    assert len(calls)==1
    attempt = calls[0];wire = attempt['final_payload_audit']['serialized_parameters']
    assert attempt['http_status']==200 and attempt['outcome']=='success'
    assert wire['model']=='gemini-3.8-flash' and wire['max_tokens']==65536
    assert wire['reasoning_effort']=='high' and wire['stream']
    assert not {'thinking','thinking_budget','seed','max_completion_tokens'} & wire.keys()
    assert attempt['public_response']['model']=='gemini-3.8-flash'
    assert row['requests'][0]['messages']==reference[task['task_id']]['requests'][0]['messages']
    assert attempt['raw_usage']['total_tokens']==sum(u['total_tokens'] for u in row['usage'])
    assert attempt['raw_usage']['total_tokens']==ledger['attempts'][attempt['request_id']]['accounted_tokens']
    assert row['execution']['all_authored_nodes_completed']
    assert row['execution']['attempts'] and all(a['status']=='ok' and a['outputs_consumed'] for a in row['execution']['attempts'])
    samples.append({'task_id':task['task_id'],'request_id':attempt['request_id'],'parameters':wire,
                    'raw_usage':attempt['raw_usage'],'response_model':attempt['public_response']['model'],
                    'program_consumed':True})
value = run.status()
assert value['status']=='running' and value['process_alive'] and not value['error']
receipt = {'passed':True,'at':r.utc(),'real_http_from_check':0,'source':r.code_identity(),
           'snapshot':value,'samples':samples,'same_public_messages_bank_and_method':True}
r.write_json(run.HERE/'startup_verification.json',receipt)
print(json.dumps({'passed':True,'progress':value['arms'][run.ARM],'tokens':value['campaign_tokens'],'http':value['campaign_http']}))
