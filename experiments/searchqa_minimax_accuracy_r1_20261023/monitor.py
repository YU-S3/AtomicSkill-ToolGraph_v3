"""Read-only progress monitor."""
import json
from pathlib import Path
import run_comparison as x

s=x.c.status(x.STAGE)
before=x.r.read(x.ROOT/'inherited_budget.json')
ledger=x.r.read(x.ROOT/'budget.json')
added=[v for k,v in ledger['attempts'].items() if k not in before['attempts']]
pid=(x.ROOT/'pid').read_text().strip()
alive=Path('/proc')/pid
print(json.dumps({'status':s['status'],'alive':alive.exists(),'arms':s['arms'],
    'new_tokens':s['campaign_tokens']-sum(v.get('accounted_tokens',0) for v in before['attempts'].values()),
    'new_http':s['campaign_http']-len(before['attempts']),
    'new_unmetered_states':[v['status'] for v in added if v['status']!='metered'],
    'note':'admitted while running is an in-flight request, not an error',
    'current':x.r.read(x.ROOT/x.STAGE/'current.json')},ensure_ascii=False,indent=2))
