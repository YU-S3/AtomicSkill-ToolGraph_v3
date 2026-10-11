"""Read-only progress and cost snapshot for explicitly selected experiment cells."""
import argparse
import json
from pathlib import Path
import sqlite3


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot(cell):
    root=Path(cell);manifest=read(root/'run_manifest.json');completion=read(root/'completion.json')
    result={'cell':str(root),'status':manifest.get('run_status',completion.get('status','not_started')),'phases':{}}
    for split in ('train','val','test'):
        phase=root/split;summary=read(phase/'summary.json');dbpath=phase/'run.sqlite3'
        if not dbpath.exists():continue
        db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)
        statuses=dict(db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status'))
        active=[r[0] for r in db.execute("SELECT id FROM tasks WHERE status='running'")];db.close()
        result['phases'][split]={'completed':statuses.get('completed',0),'expected':summary.get('expected_tasks'),
            'successes_at_last_summary':summary.get('successes'),'active_tasks':active,'statuses':statuses}
        if result['phases'][split]['expected'] is None:
            config=read(phase/'config.json');manifest=config.get('manifest')
            if manifest and Path(manifest).exists():result['phases'][split]['expected']=len(read(Path(manifest))['tasks'])
    ledger=read(root/'budget.json');carry=read(root/'cost_carryforward_manifest.json')
    attempts=ledger.get('attempts',{})
    used=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in attempts.values())
    result['cost']={'tokens_including_carry_and_reservations':used,'new_run_tokens_or_reservations':used-carry.get('inherited_tokens',0),
        'http_including_carry':len(attempts),'new_run_http':len(attempts)-carry.get('inherited_http',0),
        'unknown_billing':ledger.get('unknown_billing'),'limits':ledger.get('limits')}
    result['prefix_gate']=read(root/'prefix_engineering_audit.json').get('status','pending')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cells',nargs='+')
    for cell in parser.parse_args().cells:print(json.dumps(snapshot(cell),ensure_ascii=False,indent=2))
