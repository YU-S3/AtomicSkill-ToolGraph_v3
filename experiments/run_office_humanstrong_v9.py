"""Bounded second development comparison after V8's measured lack of benefit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v8 as previous

paired, common = previous.paired, previous.common
PARENT = '/home/yangchengyu/officeqa_humanstrong_v8_targeted_test15_pairs_seed42_20261021'


def bind_runner():
    paired.PACKAGE=common.SOURCE/'resources/office_humanstrong_v9'
    paired.VERSION='office.humanstrong.v9.targeted-test5-pairs.20261021'
    paired.AUTHORIZATION=(
        '2026-10-21 user requested iteratively testing applicable principles, keeping gains and removing harm. '
        'V8 completed without accuracy gain. One bounded follow-up tests the explicit normalized-ratio contract '
        'and removal of harmful/broad guidance, using five development-exposed tasks in two fresh arms. '
        'Carry all V8/ancestral costs within approved cumulative 190M; no full Test, training or automatic repetition.')
    paired.bind_runner()


def select_tasks(tasks):
    # Review-sample IDs stay in the experiment protocol, never the Bank or method.
    ids=['officeqa:UID0237','officeqa:UID0220','officeqa:UID0123','officeqa:UID0072','officeqa:UID0045']
    selected=[t for t in tasks if t['task_id'] in ids]
    if len(selected)!=5:raise ValueError('Missing frozen development task')
    return selected,{'rule':'One observed normalized-scale failure, percent-difference neighbor, removed-rule regression, '
        'previously correct share change and literal lookup; all fresh paired runs, canonical order.',
        'prior_scores_used_for_selection':True,
        'groups':[{'principle':'normalized_midpoint','task_ids':['officeqa:UID0237']}],
        'control_ids':[x for x in ids if x!='officeqa:UID0237'],
        'selected_principles':{t['task_id']:[s['title'] for s in common.frontier.matching(t['goal'])] for t in selected},
        'retention_rule':'Retain only as a development-supported example if the target is a fresh paired win '
            'and the four other tasks show no paired losses; otherwise do not promote.',
        'overall_gate':'Bank must exceed paired NoSkill; every regression control must have no paired loss. '
            'This single applicable target cannot establish broader Test accuracy.'}


def prepare(root,base=PARENT,resources=previous.RESOURCES,config_run=PARENT):
    return paired.prepare(root,base,resources,config_run,split='test',selector=select_tasks)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','check','launch','run','status']);ap.add_argument('--root',required=True)
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare':prepare(root)
    elif args.action=='check':common.verify(root);print('Identity/billing check passed; zero HTTP.')
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
