"""Read-only engineering verification of the actual paid prefix."""
from collections import Counter
import json
import run_experiment as run

c,r=run.c,run.r
run.check()
root=run.ROOT/run.STAGE
tasks={t['task_id']:t for t in r.read(root/'tasks.json')['tasks']}
audit={}
for arm in run.arms():
    rows=c.completed(root,arm)
    assert len(rows)>=3
    total=delivered=0
    models=Counter()
    for key,row in rows.items():
        assert row['execution']['all_authored_nodes_completed']
        assert all(a['status']=='ok' for a in row['execution'].get('attempts',[]))
        req=row['requests'][0]
        payload=json.loads(req['messages'][1]['content'])
        assert payload['goal']==tasks[key]['goal'] and payload['inputs']==tasks[key]['inputs']
        assert len(row['usage'])==1 and row['usage'][0]['model']=='deepseek-v4-flash'
        models[row['usage'][0]['provider_metadata']['response_model']]+=1
        assert req['provider_finish_reason']=='stop'
        for attempt in req['http_attempts']:
            settings=attempt['final_payload_audit']['serialized_parameters']
            assert settings['model']=='deepseek-v4-flash' and settings['max_tokens']==32768
            assert settings['reasoning_effort']=='high' and settings['thinking']['type']=='enabled'
            assert attempt['usage_status']=='reported'
        if arm.startswith('v8_'):
            for path in (run.HERE/'guidance').glob('*.txt'):
                assert req['messages'][0]['content'].count(path.read_text().strip())==1
        mapping=req['node_response_mapping']
        delivered+=bool(mapping.get('source_fields_delivered'))
        assert ('source_fields' in payload)==bool(mapping.get('source_fields_delivered'))
        total+=sum(u['total_tokens'] for u in row['usage'])
    audit[arm]={'completed':len(rows),'model_solve_calls_per_completed_task':1,'tokens':total,
        'source_fields_delivered':delivered,'all_native_nodes_completed':True,'all_programs_ok':True,
        'full_public_input_unchanged':True,'response_models':dict(models)}
assert audit['v8_programmatic']['source_fields_delivered']>=1
receipt={'passed':True,'at':r.utc(),'source':r.code_identity(),'plan_sha256':r.sha(root/'plan.json'),
    'controller_sha256':r.sha(run.__file__),'arms':audit,'run_status':run.status(),
    'notes':'Startup engineering validation only; no accuracy inference from prefix. An admitted request may be in flight.'}
r.write_json(run.ROOT/'startup_verification.json',receipt)
r.write_json(run.HERE/'startup_verification.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
