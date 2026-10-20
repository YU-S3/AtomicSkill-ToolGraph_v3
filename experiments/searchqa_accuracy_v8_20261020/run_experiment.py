"""Independent bounded V7/V8 comparison, using the unchanged native paired runner."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'searchqa_accuracy_v7_20261019'
sys.dont_write_bytecode=True
sys.path.insert(0,str(PRIOR))
import campaign as c
from atomic_skillgraph.empirical.budget_governor import billing_blocked
r=c.r
ROOT=Path('/home/yangchengyu/searchqa_accuracy_v8_seed42_20261020')
PARENT=Path('/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020')
STAGE='comparison192'
ALLOCATION={'tokens':4000000,'http':720}
c.CAMPAIGN=ROOT

def arms():
    return {
        'v7_text_compiler':{'package':str(PARENT/'fulltest1400/v7_text_compiler/input'),'kind':'native'},
        **{a:{'package':str(HERE/'packages'/a),'kind':'native'} for a in ('v8_guidance','v8_programmatic')}}

def prepare():
    assert not ROOT.exists(), 'Preserve existing run'
    receipt=r.read(HERE/'offline_acceptance.json')
    assert receipt['passed'] and receipt['controller_sha256']==r.sha(__file__)
    for a,item in arms().items():assert receipt['packages'][a]==r.tree_identity(item['package'])
    assert r.read(PARENT/'fulltest1400/result.json')['status']=='completed'
    parent=r.read(PARENT/'budget.json')
    assert not billing_blocked(parent)
    carry=r.read(PARENT/'cost_carryforward_manifest.json')
    charge=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in parent['attempts'].values())
    limits=dict(token_limit=charge+ALLOCATION['tokens'],request_limit=len(parent['attempts'])+ALLOCATION['http'],finish_reserve=0)
    authority,reserve=carry['original_authority_limits'],carry['r3_remaining_reservation']
    assert limits['token_limit']+reserve['tokens']<=authority['token_limit']
    assert limits['request_limit']+reserve['http']<=authority['request_limit']
    cfg=r.read(PARENT/'fulltest1400/v7_text_compiler/config.json')
    canonical=Path(cfg['manifest'])
    r.verify(canonical.parent.parent)
    tasks=r.read(r.DATASETS/'test.json')['tasks']
    assert len(tasks)==1400 and [t['task_id'] for t in tasks]==[t['task_id'] for t in r.read(canonical)['tasks']]
    chosen=set(t['task_id'] for t in random.Random(20261020).sample(tasks,192))
    selected=[t for t in tasks if t['task_id'] in chosen]
    ledger=deepcopy(parent);ledger['limits']=limits
    for name in ('budget.json','inherited_budget.json'):r.write_json(ROOT/name,ledger)
    r.write_json(ROOT/'authorization.json',dict(status='approved',at=r.utc(),
        user_instruction='按照上面的规划修改好bank，启动相关实验。',
        scope='One bounded same-model three-arm comparison; detailed authored experience, no factual answer memory.',
        allocation=ALLOCATION,automatic_fulltest=False,automatic_promotion=False))
    r.write_json(ROOT/'cost_carryforward_manifest.json',dict(parent=str(PARENT),
        parent_budget_sha256=r.sha(PARENT/'budget.json'),historical_charged_tokens=charge,historical_http=len(parent['attempts']),
        limits=limits,allocation=ALLOCATION,original_authority_limits=authority,r3_remaining_reservation=reserve,
        old_unknown_reservation_preserved=parent.get('inherited_reservation_recovery')))
    r.write_json(ROOT/'selection_public.json',dict(tasks=selected))
    r.write_json(HERE/'spec.json',dict(tasks=str(ROOT/'selection_public.json'),arms=arms(),
        purpose='Uniform seed20261020 sample of 192 canonical Test tasks, no outcome filtering. Paired V7 / V8 guidance / V8 conditional Program. Post-Test development, not a fresh held-out result. Same DeepSeek high/cap32768/input/scorer; one normal model call per arm/task.'))
    c.prepare(STAGE,HERE/'spec.json')
    plan=r.read(ROOT/STAGE/'plan.json')
    for a in arms():
        path=ROOT/STAGE/a/'config.json'
        config=r.read(path)
        assert config['llm']==cfg['llm'] and config['runtime']==cfg['runtime']
        config['experiment']['authored_bank_runtime']=r.read(ROOT/STAGE/a/'frozen_bank/imported_bank.json')['method_version']
        r.write_json(path,config);plan['files'][str(path)]=r.sha(path)
    locked=[Path(__file__),HERE/'offline_acceptance.json',HERE/'build_candidates.py',
        canonical,canonical.parent.parent/'manifest.json',r.DATASETS/'test.json',
        PARENT/'budget.json',ROOT/'selection_public.json',HERE/'check_candidates.py']
    plan['files'].update({str(p):r.sha(p) for p in locked})
    plan.update(selection_seed=20261020,canonical_task_order=True,automatic_promotion=False,
        hypotheses=['More concrete conditional experience improves EM without remembered answers.',
                    'Literal source-field Program adds benefit beyond identical guidance.'],
        cost_reference='Measured total tokens ratio to concurrent V7; target <=1.5; not used for per-question stopping.',
        reporting='Report all three arms and both gain/loss directions. No outcome-based edits during run.')
    r.write_json(ROOT/STAGE/'plan.json',plan);r.write_json(ROOT/STAGE/'plan_identity.json',{'sha256':r.digest(plan)})
    result=c.check(STAGE);r.write_json(ROOT/'preflight.json',{**result,'real_http':0,'at':r.utc(),'allocation':ALLOCATION})
    return result

def status():
    result=c.status(STAGE)
    for a in result['arms']:
        rows=c.completed(ROOT/STAGE,a)
        mappings=[q.get('node_response_mapping',{}) for row in rows.values() for q in row['requests']]
        result['arms'][a].update(
            fields_delivered=sum(m.get('source_fields_delivered',False) for m in mappings),
            matched_fields=sum(m.get('source_field_count',0) for m in mappings))
    return result

def check():
    result=c.check(STAGE)
    receipt=r.read(HERE/'offline_acceptance.json')
    assert receipt['passed'] and receipt['controller_sha256']==r.sha(__file__)
    return result

def start(resume=False):
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check()
        if (ROOT/'pid').exists():
            pid=int((ROOT/'pid').read_text());proc=Path('/proc')/str(pid)/'cmdline'
            assert not proc.exists() or str(Path(__file__)) not in proc.read_text(), 'Already active'
            assert resume, 'Explicit resume required'
        assert not (ROOT/STAGE/'result.json').exists() or resume
        assert not (ROOT/'STOP_AFTER_TASK').exists()
        with (ROOT/'run.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'run'],cwd=r.SOURCE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pid').write_text(str(child.pid)+'\n')
        launch={'pid':child.pid,'started_at':r.utc(),'root':str(ROOT),'stage':STAGE,'resume':resume}
        r.write_json(ROOT/'launch.json',launch)
        return launch

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','check','start','run','status'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':value=prepare()
    elif args.action=='check':value=check()
    elif args.action=='start':value=start(args.resume)
    elif args.action=='status':value=status()
    else:
        check()
        value=c.execute(STAGE)
        r.write_json(ROOT/'completed.json',{**status(),'at':r.utc()})
    print(json.dumps(value,ensure_ascii=False),flush=True)
