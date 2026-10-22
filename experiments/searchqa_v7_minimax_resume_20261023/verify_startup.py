"""Read-only paid continuation verification: no extra model calls."""
import json
import continue_minimax as run

r,c = run.r,run.c
c.check(run.STAGE)
snap = r.read(run.HERE / 'parent_snapshot.json')
assert r.sha(run.PARENT / 'budget.json') == snap['budget_sha256']
assert r.digest(c.completed(run.PARENT / run.STAGE, run.ARM)) == snap['result_digest']
assert r.sha(run.PARENT.parent / 'qwen38_flash' / run.STAGE / 'result.json') == snap['qwen_result_sha256']
rows = c.completed(run.ROOT / run.STAGE, run.ARM)
tasks = r.read(run.HERE / 'remaining_tasks.json')['tasks']
assert len(rows) >= 3
samples=[]
for task in tasks[:3]:
    row=rows[task['task_id']]
    calls=[a for q in row['requests'] for a in q.get('http_attempts',[])]
    assert len(calls)==1
    a=calls[0];wire=a['final_payload_audit']['serialized_parameters']
    assert a['http_status']==200 and a['outcome']=='success'
    assert wire['model']=='MiniMax-M3' and wire['max_completion_tokens']==65536
    assert wire['thinking']=={'type':'adaptive'} and a['public_response']['model']=='MiniMax-M3'
    assert a['raw_usage']['total_tokens']==sum(u['total_tokens'] for u in row['usage'])
    assert row['execution']['all_authored_nodes_completed']
    assert all(x['status']=='ok' and x['outputs_consumed'] for x in row['execution']['attempts'])
    if task==tasks[0]:
        assert row['requests'][0]['messages']==r.read(snap['failed_audit_path'])['requests'][0]['messages']
    samples.append({'task_id':task['task_id'],'request_id':a['request_id'],'raw_usage':a['raw_usage'],
                    'parameters':wire,'program_consumed':True})
ledger=r.read(run.ROOT/'budget.json');old=r.read(run.PARENT/'budget.json')
assert ledger['limits']==old['limits']
assert all(ledger['attempts'][k]==v for k,v in old['attempts'].items())
assert all(v['status'] in ('metered','admitted') for k,v in ledger['attempts'].items() if k not in old['attempts'])
value=c.status(run.STAGE)
assert value['status']=='running' and run.alive(run.ROOT)
receipt={'passed':True,'at':r.utc(),'real_http_from_check':0,'snapshot':value,'samples':samples,
         'preserved_parent_and_qwen':True,'unknown_reservation_unchanged':77459,
         'first_interrupted_task_same_messages':True,'source':r.code_identity()}
r.write_json(run.HERE/'startup_verification.json',receipt)
print(json.dumps({'passed':True,'at':receipt['at'],'progress':value['arms'][run.ARM]}))
