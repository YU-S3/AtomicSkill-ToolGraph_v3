"""Independent fixed 96-task follow-up; retains prior costs and the original 8M ceiling."""
from copy import deepcopy
import json
from pathlib import Path
import random
import time
import run_comparison as base

c,r,HERE=base.c,base.r,base.HERE
PARENT=base.ROOT
ROOT=Path('/home/yangchengyu/searchqa_minimax_accuracy_r2_20261023')
STAGE='followup96'


def prepare_spec():
    assert not (HERE/'r2_spec.json').exists()
    tasks=r.read(HERE/'tasks128_public.json')['tasks']
    indices=set(random.Random(20261024).sample(range(128),96))
    r.write_json(HERE/'tasks96_public.json',{'tasks':[t for i,t in enumerate(tasks) if i in indices]})
    r.write_json(HERE/'r2_spec.json',{'tasks':str(HERE/'tasks96_public.json'),'base_config':str(HERE/'base_config.json'),
        'arms':{'r2':{'kind':'native','package':str(HERE/'packages/minimax_r2')}},
        'purpose':'Fixed uniform 96 of the original uniform128, seed20261024. R2 follow-up development; compare with the exact prior V7/R1 rows, not simultaneous fresh controls. Same full public input/model/scorer. No automatic full Test1400.'})
    r.write_json(HERE/'r2_selection.json',{'seed':20261024,'indices_within_128':sorted(indices),'no_result_filter':True,
        'normal_new_http':96,'new_http_cap':110,'minimum_admission_headroom_to_start':400000,
        'original_cumulative_tokens':8000000,'original_cumulative_http':3000,
        'bank_sha256':r.sha(HERE/'packages/minimax_r2/bank/bank.json'),
        'scope':'One independent R2 follow-up within existing cumulative MiniMax allowance; retain every R1 result.'})


def prepare():
    assert not ROOT.exists()
    assert r.read(PARENT/base.STAGE/'result.json')['status']=='completed'
    c.CAMPAIGN=PARENT;c.check(base.STAGE)
    check=r.read(HERE/'r2_offline_acceptance.json')
    assert check['passed'] and check['driver_sha256']==r.sha(c.__file__)
    assert r.tree_identity(HERE/'packages/minimax_r2')==check['package']
    ledger=r.read(PARENT/'budget.json')
    assert ledger['limits']['token_limit']==8000000
    charged=sum(v.get('accounted_tokens',v['reserved_tokens']) for v in ledger['attempts'].values())
    if 8000000-charged<400000:
        r.write_json(HERE/'r2_not_started.json',{'reason':'Insufficient remaining original cumulative allowance',
            'headroom':8000000-charged,'required':400000,'no_new_http':True})
        return False
    inherited=r.read(PARENT/'inherited_budget.json')
    assert all(v['status']=='metered' for k,v in ledger['attempts'].items() if k not in inherited['attempts'])
    r.write_json(ROOT/'parent_budget.json',ledger)
    ledger=deepcopy(ledger)
    ledger['limits']['request_limit']=min(3000,len(ledger['attempts'])+110)
    for name in ['budget.json','inherited_budget.json']:r.write_json(ROOT/name,ledger)
    r.write_json(ROOT/'authorization.json',{'status':'approved','at':r.utc(),
        'user':base.AUTHORIZATION,'scope':'One follow-up96 Bank iteration within original8M/3000 limits; current user explicitly permits authored rules/fact cards and prior paid iterations.',
        'limits':ledger['limits']})
    r.write_json(ROOT/'cost_carryforward_manifest.json',{'parent':str(PARENT),'parent_budget_sha256':r.sha(PARENT/'budget.json'),
        'prior_searchqa_tokens':sum(v.get('accounted_tokens',0) for v in ledger['attempts'].values()),'prior_searchqa_http':len(ledger['attempts']),
        'prior_probe_task_scores_not_inherited_into_new_arm':True,'old_physical_and_unknown_entries_unchanged':True,
        'original_authority':{'token_limit':8000000,'request_limit':3000},'headroom_at_start':8000000-charged})
    c.CAMPAIGN=ROOT;c.prepare(STAGE,HERE/'r2_spec.json')
    plan=r.read(ROOT/STAGE/'plan.json')
    cfgpath=ROOT/STAGE/'r2/config.json';cfg=r.read(cfgpath)
    cfg['experiment']['authored_bank_runtime']=r.read(ROOT/STAGE/'r2/frozen_bank/imported_bank.json')['method_version']
    r.write_json(cfgpath,cfg);plan['files'][str(cfgpath)]=r.sha(cfgpath)
    for p in [Path(__file__),HERE/'r2_offline_acceptance.json',HERE/'check_revision2.py',HERE/'r2_selection.json',HERE/'infrastructure_guard.py',PARENT/base.STAGE/'result.json',PARENT/'budget.json']:
        plan['files'][str(p)]=r.sha(p)
    r.write_json(ROOT/STAGE/'plan.json',plan);r.write_json(ROOT/STAGE/'plan_identity.json',{'sha256':r.digest(plan)})
    c.check(STAGE)
    return True


def queue():
    import os
    import finalize_review
    while True:
        value=r.read(PARENT/base.STAGE/'result.json')
        if value['status']!='running':break
        pid=(PARENT/'pid').read_text().strip()
        if not Path('/proc',pid).exists():
            r.write_json(HERE/'r2_not_started.json',{'reason':'Parent exited without terminal result','no_new_http':True})
            return
        time.sleep(30)
    if value['status']!='completed':
        r.write_json(HERE/'r2_not_started.json',{'reason':'Parent did not complete','status':value['status'],'no_new_http':True})
        return
    finalize_review.finalize()
    if prepare():
        (ROOT/'pid').write_text(str(os.getpid()))
        c.execute(STAGE)
        r.write_json(HERE/'r2_completed.json',{'at':r.utc(),'status':'completed','root':str(ROOT)})
        import finalize_revision2
        finalize_revision2.finalize()


if __name__=='__main__':
    import sys
    if sys.argv[1:] == ['spec']:prepare_spec()
    else:queue()
