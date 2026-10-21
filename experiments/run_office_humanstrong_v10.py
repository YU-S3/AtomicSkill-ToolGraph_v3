"""Train-based Bank development and frozen full-split comparisons, with inherited billing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
from filelock import FileLock
import run_office_humanstrong_v8 as previous

paired, common = previous.paired, previous.common
PARENT = '/home/yangchengyu/officeqa_humanstrong_v9_targeted_test5_pairs_seed42_20261021'
RESOURCES = previous.RESOURCES
CUMULATIVE_CAP = 262000000


def bind_runner():
    paired.CUMULATIVE_CAP=CUMULATIVE_CAP
    paired.PACKAGE=common.SOURCE/'resources/office_humanstrong_v10'
    paired.VERSION='office.humanstrong.v10.train-to-full-splits.20261021'
    paired.AUTHORIZATION=(
        '2026-10-21 user explicitly approved cumulative 262M and fresh Bank/off comparisons on all '
        'Train120 and Test102: 444 episodes, new theoretical token cap 96,159,744. '
        'One frozen candidate across both splits; no extra paid Train12 or automatic repetition. '
        'Frozen human Bank evaluation only; no additional partition of Ours Train, no Test-time learning. '
        'Existing Train and Test exposure is disclosed; these are development observations.')
    paired.bind_runner()


def train_probe(tasks):
    # Public applicability only. Do not select failures or exploit private gold.
    groups = {'table_groups':2,'series_scope':3,'compound_growth':2,'price_basis':2,'uncovered':3}
    picked=set();receipt={}
    for name,count in groups.items():
        pool=[t for t in tasks if t['task_id'] not in picked and
              (not common.frontier.matching(t['goal'],t['inputs']) if name=='uncovered' else
               name in [s['title'] for s in common.frontier.matching(t['goal'],t['inputs'])])]
        pool.sort(key=lambda t:hashlib.sha256(('42:'+t['task_id']).encode()).hexdigest())
        if len(pool)<count:raise ValueError('Insufficient public Train examples for '+name)
        receipt[name]=[t['task_id'] for t in pool[:count]];picked.update(receipt[name])
    return [t for t in tasks if t['task_id'] in picked],{
        'rule':'12 public-applicability Train examples: 2 grouped tables, 3 scoped statistics, 2 CAGR, '
            '2 price basis and 3 uncovered controls, deterministic seed42 hash order within each group. '
            'Development tests only, not a partition or promotion stage of Ours learning.',
        'groups':receipt,'prior_scores_used_for_selection':False,
        'decision_rule':'Report all paired wins/losses and costs. Do not claim gain from tied families or discard '
            'them as proven harmful. Review observed regressions before freezing a full-split candidate.'}


def all_tasks(tasks):
    return tasks,{'rule':'Every member in canonical order; no performance-based filtering.',
        'prior_scores_used_for_selection':False}


def prepare(root,base=PARENT,resources=RESOURCES,config_run=PARENT,scope='train12'):
    split='test' if scope=='test102' else 'train'
    return paired.prepare(root,base,resources,config_run,split=split,
        selector=train_probe if scope=='train12' else all_tasks)


def coverage(resources,output):
    configuration=previous.paired.load('v10_coverage_config',paired.PACKAGE/'integration/configuration.py')
    root,_=configuration.verify_resources(Path(resources))
    result={'schema':'office.bank.public-applicability.v1','actual_llm_http':0,
        'bank_sha256':common.sha(paired.PACKAGE/'bank/bank.json'),
        'warning':'Applicability is not correctness. Train120 and Test102 contain prior development exposure.',
        'splits':{}}
    for split in ('train','test'):
        tasks=common.read(root/(split+'.json'))['tasks'];counts=Counter();rows=[]
        for t in tasks:
            cards=common.frontier.matching(t['goal'],t['inputs'])
            titles=[s['title'] for s in cards];counts.update(titles)
            rows.append({'task_id':t['task_id'],'principles':titles,
                'programs':[s['operation'] for s in cards if s.get('operation')],
                'guidance_characters':sum(len(s['guidance']) for s in cards)})
        result['splits'][split]={'total':len(rows),'covered':sum(bool(r['principles']) for r in rows),
            'per_principle':dict(counts),'rows':rows}
    common.save_json(output,result)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='rows'} for k,v in result['splits'].items()}))


def prepare_all(root):
    if root.exists():raise ValueError('Fresh campaign directory required')
    _,ledger,_=paired.settled_parent(Path(PARENT))
    used=sum(map(common.charge,ledger['attempts'].values()))
    cap=(120+102)*2*paired.TASK_CAP
    if used+cap>CUMULATIVE_CAP:raise ValueError('Insufficient complete-campaign authority')
    prepare(root/'train120',scope='train120')
    plan=common.read(root/'train120/plan.json')
    campaign={'created_at':common.utc(),'authorization':paired.AUTHORIZATION,
        'base_run':PARENT,'inherited_used_or_reserved':used,'new_token_cap':cap,
        'authorized_cumulative_ceiling':CUMULATIVE_CAP,'expected_episodes':444,
        'sequence':['train120','test102'],'code':plan['code'],'package_identity':plan['package_identity'],
        'train_plan_sha256':common.sha(root/'train120/plan.json'),
        'selection':'All 120 Train and all 102 Test in canonical order; fresh randomized arm order seed42.',
        'stop_rule':'Stop on infrastructure/billing/freeze failure; do not alter Bank or gate Test on accuracy.',
        'interpretation':'Human authored frozen evaluation. Both splits are development exposed. '
            'Train evaluation does not update Bank or split the Ours learning set. No automatic promotion.'}
    common.save_json(root/'campaign.json',campaign)
    common.save_json(root/'campaign_identity.json',{'sha256':common.sha(root/'campaign.json')})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()})
    print(json.dumps({'root':str(root),'episodes':444,'new_token_cap':cap,'cumulative_max':used+cap}))


def verify_campaign(root):
    assert common.sha(root/'campaign.json')==common.read(root/'campaign_identity.json')['sha256']
    plan=common.read(root/'campaign.json')
    assert common.code_identity()==plan['code'] and not plan['code']['tracked_dirty']
    assert common.package_identity()==plan['package_identity']
    assert common.sha(root/'train120/plan.json')==plan['train_plan_sha256']
    return plan


def campaign_status(root):
    plan=common.read(root/'campaign.json')
    runs={name:common.status(root/name) for name in plan['sequence'] if (root/name/'plan.json').exists()}
    return {'launch':common.read(root/'launch_status.json'),'expected_episodes':444,
        'completed_episodes':sum(s['completed_episodes'] for s in runs.values()),
        'new_tokens_used_or_reserved':sum(s['new_tokens_used_or_reserved'] for s in runs.values()),
        'new_http':sum(s['new_http'] for s in runs.values()),
        'new_unsettled':sum(s['new_unsettled'] for s in runs.values()),
        'cumulative_tokens_used_or_reserved':plan['inherited_used_or_reserved']+
            sum(s['new_tokens_used_or_reserved'] for s in runs.values()),
        'splits':runs}


def compact_status(root):
    status=campaign_status(root)
    status['splits']={name:{'completed':run['completed_episodes'],'expected':run['expected_episodes'],
        'launch':run['launch'],'arms':{arm:{
            'completed':sum(r['arm']==arm for r in run['results']),
            'correct':sum(r['correct'] for r in run['results'] if r['arm']==arm),
            'tokens':sum(r['tokens'] for r in run['results'] if r['arm']==arm),
            'program_calls':sum(r['program_calls'] for r in run['results'] if r['arm']==arm)}
            for arm in ('manual','off')}} for name,run in status['splits'].items()}
    return status


def run_all(root,env):
    plan=verify_campaign(root);started=common.utc()
    with FileLock(str(root/'.campaign.lock'),timeout=0):
        try:
            for name in plan['sequence']:
                verify_campaign(root)
                if not (root/name/'plan.json').exists():
                    prepare(root/name,base=str(root/'train120'),scope=name)
                common.save_json(root/'launch_status.json',{'status':'running','phase':name,
                    'pid':os.getpid(),'started_at':started})
                code=common.execute(root/name,env)
                if code:raise RuntimeError(name+' stopped with exit '+str(code))
        except Exception as error:
            common.save_json(root/'launch_status.json',{'status':'failed','started_at':started,
                'ended_at':common.utc(),'error':str(error)})
            raise
        common.save_json(root/'launch_status.json',{'status':'completed','started_at':started,'ended_at':common.utc()})
        common.save_json(root/'comparison.json',campaign_status(root))
        common.save_json(root/'billing_result.json',campaign_status(root))
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['coverage','prepare','check','launch','run','status',
        'prepare-all','launch-all','run-all','status-all']);ap.add_argument('--root',required=True)
    ap.add_argument('--scope',choices=['train120','test102'],default='train120')
    ap.add_argument('--base-run',default=PARENT);ap.add_argument('--config-run',default=PARENT)
    ap.add_argument('--resources',default=RESOURCES)
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare-all':prepare_all(root)
    elif args.action=='status-all':print(json.dumps(compact_status(root),ensure_ascii=False,indent=2))
    elif args.action=='run-all':sys.exit(run_all(root,args.env_file))
    elif args.action=='coverage':coverage(args.resources,root)
    elif args.action=='prepare':prepare(root,args.base_run,args.resources,args.config_run,args.scope)
    elif args.action=='check':common.verify(root);print('Identity/billing check passed; zero HTTP.')
    elif args.action=='status':print(json.dumps(common.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(common.execute(root,args.env_file))
    else:
        verify_campaign(root) if args.action=='launch-all' else common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    'run-all' if args.action=='launch-all' else 'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid':child.pid,'root':str(root)}))
