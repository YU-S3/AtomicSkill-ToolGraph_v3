"""Freeze a reviewed qualifying direction, then compare it with off on complete Test102."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v12 as search

paired,common=search.paired,search.common


def choose(result):
    if (result['status']['status']!='completed' or result['new_unsettled'] or
            any(r['completed']!=12 for r in result['metrics'].values())):
        raise ValueError('All four diagnostic arms must complete with settled billing')
    eligible=[a for a,r in result['vs_off'].items() if r['net']>0 and not r['losses']]
    if not eligible:raise ValueError('No direction has positive net gain without paired regressions; do not open Test')
    return min(eligible,key=lambda a:(-result['metrics'][a]['correct'],result['metrics'][a]['tokens'],a))


def prepare(root,parent,profile):
    parent=Path(parent).resolve()
    result=search.analysis(parent)
    chosen=choose(result)
    if profile!=chosen:raise ValueError('Explicit profile must match the preregistered ranking')
    plan=common.read(parent/'plan.json')
    if plan['package_identity']!=common.package_identity():raise ValueError('Bank changed after diagnosis; repeat bounded validation first')
    decision={'profile':chosen,'diagnostic_run':str(parent),'diagnostic_plan_sha256':common.sha(parent/'plan.json'),
              'metrics':result['metrics'],'vs_off':result['vs_off'],
              'rule':'Positive net gain and no losses against concurrent off; highest correct count then lowest tokens.',
              'caveat':'Development-selected direction, not an unbiased estimate from diagnostic accuracy.'}
    def all_test(tasks):
        return tasks,{'rule':'All 102 canonical external-aligned Test tasks, unchanged order and membership.',
                      'prior_scores_used_for_selection':False,'profile_selection':decision,'development_exposed':True}
    return paired.prepare(root,parent,search.RESOURCES,parent,split='test',selector=all_test,
                          comparison_arms=('off',chosen))


def bind_runner():
    search.bind_runner()
    paired.VERSION=common.VERSION='office.humanstrong.v12.selected-profile-test102.20261022'
    paired.AUTHORIZATION=('2026-10-22 user approved the planned rule/Program search followed by a frozen Test102 '
        'Bank/off comparison once the development checks pass. One Test102 execution per arm, same model/input/tools/budget; '
        'preserve all history and unknown reservations within cumulative 262M. No test-time edits or automatic repeats.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','check','launch','run','status'])
    ap.add_argument('--root',required=True);ap.add_argument('--parent');ap.add_argument('--profile',choices=search.ARMS[1:])
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare':
        if not args.parent or not args.profile:ap.error('prepare requires the reviewed parent and explicit profile')
        prepare(root,args.parent,args.profile)
    elif args.action=='check':common.verify(root);print('Frozen Test identity and inherited cost verified; zero model HTTP.')
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
            (root/'run.pid').write_text(str(child.pid)+'\n');print(json.dumps({'pid':child.pid,'root':str(root)}))
