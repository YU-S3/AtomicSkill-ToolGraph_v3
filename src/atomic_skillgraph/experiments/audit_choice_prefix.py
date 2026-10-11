"""Engineering-only gate for a fresh public-process Train prefix, with no extra calls."""
import argparse
import json
from pathlib import Path
import sqlite3

from ..empirical.choice_guidance import checked_asset
from ..empirical.budget_governor import input_token_bound
from ..empirical.public_process import VERSION
from .audit_choice_process import audit as coverage
from .canonical_manifest import ordered_train
from .run_empirical import code_identity,write_json


def audit(root,count=6):
    root=Path(root);manifest=json.loads((root/'run_manifest.json').read_text());source=code_identity()
    assert not source['tracked_dirty'] and source['git_sha']==manifest['source_commit'], 'Pinned source changed'
    assert manifest['initial_bank']=='empty independent Bank' and manifest['choice_method_version']==VERSION and manifest['run_seed']==42
    config=json.loads((root/'train/config.json').read_text())
    assert config['runtime']['choice_guidance']['public_process'] and not config['runtime']['choice_bounded_finalization']['enabled']
    assert config['experiment']['choice_projection_version']==manifest['choice_projection_version']
    assert config['experiment']['public_materialization_sha256']==manifest['public_materialization_sha256']
    db=sqlite3.connect((root/'train/run.sqlite3').as_uri()+'?mode=ro',uri=True)
    rows=[json.loads(r[0]) for r in db.execute("SELECT result FROM tasks WHERE status='completed' ORDER BY rowid")];db.close()
    public=json.loads(Path(config['manifest']).read_text())['tasks']
    assert len(rows)==count and [r['task']['task_id'] for r in rows]==[t['task_id'] for t in ordered_train(public,42)[:count]]
    errors=[];http_ids=[]
    for trace in rows:
        error=trace.get('error') or {}
        assert error.get('code') in {None,'parent_task_budget_exhausted','empirical_token_budget_exhausted','runtime_finish_reserved'}, str(error)
        for request in trace.get('requests',[]):
            if request.get('error'):errors.append({'task_id':trace['task']['task_id'],'purpose':request['purpose'],'error':request['error']})
            if str(request.get('error','')).startswith('material_too_large'):
                receipt=request.get('material_refusal') or {}
                assert receipt.get('reason') and receipt['wire_bytes']>receipt['limit_bytes'] and not request.get('http_attempts'), 'Unexplained material refusal'
                assert receipt.get('required_fields') and input_token_bound(receipt['payload'])==receipt['wire_bytes']==sum(receipt['parts_bytes'].values())
            for attempt in request.get('http_attempts',[]):
                http_ids.append(attempt['request_id'])
                wire=attempt['final_payload_audit'];params=wire['serialized_parameters']
                if request['stage']=='extractor':
                    assert wire['serialized_payload_utf8_bytes']<=request['payload_max_bytes']
                    assert params['thinking']['type']=='disabled'
    ledger=json.loads((root/'budget.json').read_text());carry=json.loads((root/'cost_carryforward_manifest.json').read_text())
    assert ledger['limits']=={k:config['budget'][k] for k in ('token_limit','request_limit','finish_reserve')}
    assert carry['bank_inherited'] is False and carry['limits_unchanged'] is True
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())<=ledger['limits']['token_limit'] and len(ledger['attempts'])<=ledger['limits']['request_limit']
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    assert len(http_ids)==len(set(http_ids)) and set(ledger['attempts'])==set(http_ids)|set(carry['inherited_attempt_ids'])
    db=sqlite3.connect((root/'train/bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    assert db.execute("SELECT value FROM metadata WHERE key='choice_method_version'").fetchone()==(VERSION,)
    skills=[json.loads(r[0]) for r in db.execute("SELECT payload FROM assets WHERE kind='skill'")];db.close()
    assert all(checked_asset(a) for a in skills if a.get('guidance_policy_version'))
    report={'status':'passed','source_commit':source['git_sha'],'prefix_tasks':count,'performance_gate':False,
        'coverage':coverage(root),'ordinary_or_bounded_errors':errors,'extra_pilot_tasks':0,
        'continuation':'same source, directory and Bank; Train60 to Frozen to previously used Val17; no Test'}
    write_json(root/'prefix_engineering_audit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--count',type=int,default=6)
    args=p.parse_args();print(json.dumps(audit(args.root,args.count),ensure_ascii=False))
