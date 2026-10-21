"""Fresh paired replay of the entire stopped V10 prefix, with truthful paused ancestry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v6 as paired

common=paired.common
PARENT='/home/yangchengyu/officeqa_humanstrong_v10_train120_test102_pairs_seed42_20261021/train120'
RESOURCES='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1'
OLD_SETTLED=paired.settled_parent


def settled_parent(base):
    if common.read(base/'launch_status.json')['status']=='completed':return OLD_SETTLED(base)
    with FileLock(str(base/'.run.lock'),timeout=0):
        if common.read(base/'launch_status.json')['status']!='paused' or not (base/'STOP_AFTER_EPISODE').exists():
            raise ValueError('An explicitly paused task boundary is required')
        plan=common.read(base/'plan.json');ledger=common.read(base/'budget.json')
        prefix=common.read(base/'budget_prefix.json')['attempts']
        if common.billing_blocked(ledger) or any(a['status']!='metered' for k,a in ledger['attempts'].items() if k not in prefix):
            raise ValueError('Paused parent has unsettled requests')
        ended=[];gap=False
        for episode in plan['episodes']:
            path=Path(episode['path'])/'summary.json'
            if not path.exists():gap=True;continue
            row=common.read(path)
            if gap or not row['complete'] or row['tasks']!=1:raise ValueError('Parent is not a complete contiguous prefix')
            ended.append(episode)
        if not ended or len(ended)%2 or any(ended[i]['task_id']!=ended[i+1]['task_id'] for i in range(0,len(ended),2)):
            raise ValueError('Paused prefix must contain complete pairs')
        locks=set(plan.get('locks',[]))|{str(base/'.run.lock')}
        return plan,ledger,sorted(locks)


def bind_runner():
    paired.PACKAGE=common.SOURCE/'resources/office_humanstrong_v11'
    paired.VERSION='office.humanstrong.v11.paused-prefix-pairs.20261021'
    paired.CUMULATIVE_CAP=262000000
    paired.AUTHORIZATION=('2026-10-21 user requested effective Skill/Program revisions and testing, under prior paid-experiment '
        'authorization and the fixed cumulative 262M ceiling. One fresh paired replay of all 62 complete V10 Train pairs, '
        'including all successes, failures and uncovered controls. Source-bound final calculators replace intermediate-only '
        'assets; Bank-off retains V10 settings. Preserve paused history and costs. No automatic Test, repetition or Ours Train partition.')
    paired.bind_runner();paired.settled_parent=settled_parent


def select_prefix(tasks):
    plan,_,_=settled_parent(Path(PARENT))
    ids={e['task_id'] for e in plan['episodes'] if (Path(e['path'])/'summary.json').exists()}
    selected=[t for t in tasks if t['task_id'] in ids]
    if len(selected)!=62:raise ValueError('Frozen stopped prefix changed')
    return selected,{'rule':'All 62 complete pairs in the previously stopped chronological Train prefix, canonical order; '
        'no filtering by correctness or applicability. A frozen human Bank regression test, not a partition of Ours learning.',
        'prior_scores_used_for_selection':False,'development_exposed':True,
        'interpretation':'Report paired wins/losses, total cost and real final-Program submission. Execution success alone is not efficacy.'}


def prepare(root,base=PARENT,resources=RESOURCES,config_run=PARENT):
    if str(Path(base).resolve())!=PARENT:raise ValueError('Fixed paused parent required')
    return paired.prepare(root,base,resources,config_run,split='train',selector=select_prefix,
        parent_reports=('billing_result.json','STOP_AFTER_EPISODE'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','check','launch','run','status'])
    ap.add_argument('--root',required=True)
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare':prepare(root)
    elif args.action=='check':common.verify(root);print('Identity and inherited billing verified; zero model HTTP.')
    elif args.action=='status':print(json.dumps(common.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(common.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid':child.pid,'root':str(root)}))
