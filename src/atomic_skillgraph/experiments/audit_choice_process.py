"""Read existing Train records without inventing public processes or making model calls."""
import argparse
import json
from pathlib import Path
import sqlite3

from .canonical_manifest import sha256
from .run_empirical import write_json


def audit(root):
    root=Path(root);db=sqlite3.connect((root/'train/run.sqlite3').as_uri()+'?mode=ro',uri=True)
    traces=[json.loads(row[0]) for row in db.execute("SELECT result FROM tasks WHERE status='completed' ORDER BY rowid")];db.close()
    rows=[]
    for t in traces:
        e=t.get('execution',{});process=e.get('public_process_evidence') or {}
        operations=[r for r in e.get('temporary_executions',[])+e.get('attempts',[]) if r.get('status')=='ok']
        rows.append({'task_id':t['task']['task_id'],'hard_success':t.get('score',{}).get('hard'),
            'answer_status':e.get('answer_status'),'public_process_present':bool(process.get('public_process')),
            'self_reported_steps':len(process.get('public_steps',[])),'source_compatible_process':process.get('verification')=='source_compatible',
            'real_successful_worker_operations':len(operations),'outputs_consumed':sum(bool(r.get('outputs_consumed')) for r in operations),
            'learning_decision':t.get('learning',{}).get('decision'),'learning_reason':t.get('learning',{}).get('reason')})
    return {'kind':'existing Train public evidence audit','source_run':str(root),'new_http':0,'inferred_public_processes':0,
        'train_config_sha256':sha256(root/'train/config.json'),'source_commit':json.loads((root/'run_manifest.json').read_text())['source_commit'],
        'counts':{'tasks':len(rows),'hard_success':sum(r['hard_success'] is True for r in rows),
            **{k:sum(r[k] for r in rows) for k in ('public_process_present','self_reported_steps','source_compatible_process','real_successful_worker_operations','outputs_consumed')}},
        'tasks':rows,'limitation':'Self-reported public steps are observable statements, not executed tools or mathematical proof.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.root);write_json(args.output,result)
    print(json.dumps(result['counts'],ensure_ascii=False))
