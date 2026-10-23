"""Read-only progress monitor."""
import json
from pathlib import Path
import run_comparison as x

stages=[(x.ROOT,x.STAGE),(Path('/home/yangchengyu/searchqa_minimax_accuracy_r2_20261023'),'followup96')]
values={}
for root,stage in stages:
    if not (root/stage/'plan.json').exists():continue
    x.c.CAMPAIGN=root;s=x.c.status(stage)
    before=x.r.read(root/'inherited_budget.json');ledger=x.r.read(root/'budget.json')
    added=[v for k,v in ledger['attempts'].items() if k not in before['attempts']]
    pid=(root/'pid').read_text().strip() if (root/'pid').exists() else ''
    values[stage]={'status':s['status'],'alive':bool(pid) and Path('/proc',pid).exists(),'arms':s['arms'],
        'new_tokens':sum(v.get('accounted_tokens',0) for v in added),'new_http':len(added),
        'new_unmetered_states':[v['status'] for v in added if v['status']!='metered']}
    if (root/stage/'current.json').exists():values[stage]['current']=x.r.read(root/stage/'current.json')
if (x.HERE/'r2_not_started.json').exists():values['r2_not_started']=x.r.read(x.HERE/'r2_not_started.json')
values['note']='admitted while running is an in-flight request, not an error'
print(json.dumps(values,ensure_ascii=False,indent=2))
