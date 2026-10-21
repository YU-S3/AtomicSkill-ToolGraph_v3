"""Second bounded direction test: retain short rules, isolate precision and scalar submission."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v12 as previous
import run_office_humanstrong_v12_test as frozen

paired,common=previous.paired,previous.common
PARENT='/home/yangchengyu/officeqa_humanstrong_v12_directions_train12_seed42_20261022'
RESOURCES=previous.RESOURCES
ARMS=('off','rules','precise','display')
VERSION='office.humanstrong.v13.final-display-directions.20261022'
TEST_VERSION='office.humanstrong.v13.selected-profile-test102.20261022'


def bind_runner(test=False):
    paired.PACKAGE=common.SOURCE/'resources/office_humanstrong_v13'
    paired.VERSION=TEST_VERSION if test else VERSION
    paired.CUMULATIVE_CAP=262000000
    paired.AUTHORIZATION=('2026-10-22 user authorized the planned multi-direction short-rule/Program search and frozen '
        'Test comparison within the previously approved cumulative 262M. '
        +('One complete canonical Test102 per arm; no test-time edits or automatic repeats.' if test else
          'Second bounded 12-task/four-arm development diagnosis, with the same preregistered Train cases. '
          'Retain the preceding short-rule profile as a concurrent control. No Ours Train subdivision or Test in this plan.')+
        ' Preserve historical charges and unknown reservations.')
    paired.bind_runner()
    previous.ARMS=ARMS


def analysis(root):
    result=previous.analysis(root)
    result['vs_rules']={}
    for arm in ARMS[2:]:
        rows={uid:r for uid,r in result['cases'].items() if arm in r and 'rules' in r}
        wins=[uid for uid,r in rows.items() if r[arm]['correct'] and not r['rules']['correct']]
        losses=[uid for uid,r in rows.items() if r['rules']['correct'] and not r[arm]['correct']]
        result['vs_rules'][arm]={'pairs':len(rows),'wins':wins,'losses':losses,'net':len(wins)-len(losses)}
    return result


def choose(result):
    candidates=deepcopy(result)
    for arm,row in result['vs_rules'].items():
        candidates['vs_off'][arm]['losses']=list(set(candidates['vs_off'][arm]['losses'])|set(row['losses']))
    return frozen.choose(candidates)


def selection(tasks):
    chosen,receipt=previous.selection(tasks)
    receipt['decision']='Retain V12 short rules as one arm. Compare precision atom and optional scalar finalizer independently of the removed broad Program menu.'
    receipt['full_test_gate']['no_losses_vs_current_short_rules_for_new_profiles']=True
    return chosen,receipt


def prepare(root):
    return paired.prepare(root,PARENT,RESOURCES,PARENT,split='train',selector=selection,comparison_arms=ARMS)


def prepare_test(root,parent,profile):
    parent=Path(parent).resolve();result=analysis(parent);selected=choose(result)
    if profile!=selected:raise ValueError('Profile must match the preregistered nonregressing ranking')
    if common.read(parent/'plan.json')['package_identity']!=common.package_identity():
        raise ValueError('Bank changed after the direction test')
    decision={'profile':selected,'diagnostic_run':str(parent),'plan_sha256':common.sha(parent/'plan.json'),
              'metrics':result['metrics'],'vs_off':result['vs_off'],'vs_rules':result['vs_rules'],
              'rule':'Positive gain and no losses vs off; new directions must also preserve all short-rule successes. Rank by correct count, then tokens.'}
    def all_test(tasks):
        return tasks,{'rule':'All 102 canonical external-aligned Test tasks in unchanged order.',
            'prior_scores_used_for_selection':False,'development_exposed':True,'profile_selection':decision}
    return paired.prepare(root,parent,RESOURCES,parent,split='test',selector=all_test,comparison_arms=('off',selected))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','prepare-test','check','launch','run','status','analyze'])
    ap.add_argument('--root',required=True);ap.add_argument('--parent');ap.add_argument('--profile',choices=ARMS[1:])
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve()
    test=args.action=='prepare-test' or (root.exists() and common.read(root/'plan.json')['method_version']==TEST_VERSION)
    bind_runner(test)
    if args.action=='prepare':prepare(root)
    elif args.action=='prepare-test':
        if not args.parent or not args.profile:ap.error('prepare-test requires parent and reviewed profile')
        prepare_test(root,args.parent,args.profile)
    elif args.action=='check':common.verify(root);print('Frozen source, assets, inputs and inherited ledger verified; zero HTTP.')
    elif args.action in ('status','analyze'):
        result=common.status(root) if test else analysis(root)
        if args.action=='analyze':common.save_json(root/'direction_comparison.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('cases','results')},ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(common.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n');print(json.dumps({'pid':child.pid,'root':str(root)}))
