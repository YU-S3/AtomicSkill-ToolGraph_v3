"""One frozen V4/NoSkill Test102 observation under the approved 155M ceiling."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v1 as common
from run_office_humanstrong_v2 import load

PACKAGE=common.SOURCE/'resources/office_humanstrong_v4'
VERSION='office.humanstrong.v4.aligned-test102-pairs.20261020'
TASK_CAP=216576
CUMULATIVE_CAP=155000000


def bind_runner():
    common.PACKAGE=PACKAGE
    common.frontier=load('office_v4_paired_frontier',PACKAGE/'integration/frontier.py')
    common.VERSION=VERSION


def prepare(root,base,resources):
    root,base,resources=map(lambda p:Path(p).resolve(),(root,base,resources))
    if root.exists():raise ValueError('Use a fresh independent directory')
    code=common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit before preparing a paid run')
    if common.read(base/'launch_status.json')['status']!='completed':raise ValueError('Billing parent must be complete')
    parent=common.read(base/'plan.json');ledger=common.read(base/'budget.json')
    if common.billing_blocked(ledger):raise ValueError('Resolve billing before a new run')
    used=sum(map(common.charge,ledger['attempts'].values()))
    configuration=load('office_v4_paired_config',PACKAGE/'integration/configuration.py')
    resources,manifest=configuration.verify_resources(resources)
    tasks=common.read(resources/'test.json')['tasks']
    canonical_path=common.SOURCE/'data/main_experiment_v1/officeqa/test.json'
    canonical=common.read(canonical_path)['tasks']
    if len(tasks)!=102 or any(t['split']!='test' for t in tasks):
        raise ValueError('All 102 Test tasks required')
    if [t['task_id'] for t in tasks]!=[t['task_id'] for t in canonical]:
        raise ValueError('Test membership/order changed')
    if {t['task_id'] for t in tasks}&{t['task_id'] for t in common.read(base/'tasks.json')['tasks']}:
        raise ValueError('Test overlaps the parent Train observation')
    token_cap=len(tasks)*2*TASK_CAP
    if used+token_cap>CUMULATIVE_CAP:raise ValueError('Insufficient cumulative authority')
    cfg=configuration.configure(common.read(base/'base_config.json'),resources,'test')
    assert cfg['runtime']['global_action_budget']==48
    assert cfg['llm']['runtime']['max_total_tokens_per_task']==TASK_CAP
    decisions=max(16,cfg['runtime']['global_action_budget']*4+16)
    http_cap=len(tasks)*2*(decisions*2+1)*(cfg['llm']['max_retries']+1)
    cfg['experiment'].update(runtime_mode='frozen',seed=42,output_dir=str(root),observation_version=VERSION)
    cfg['data_dir']=str(root/'episodes');root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json',ledger)
    limits={**ledger['limits'],'token_limit':used+token_cap,'request_limit':len(ledger['attempts'])+http_cap,'finish_reserve':0}
    ledger['limits']=limits;common.save_json(root/'budget.json',ledger)
    cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'],
                         train_task_tokens=TASK_CAP,eval_task_tokens=TASK_CAP)
    common.write_json(root/'base_config.json',cfg);common.save_json(root/'tasks.json',{'tasks':tasks})
    bundle=common.read(PACKAGE/'bank/bank.json');episodes=[];rng=random.Random(42)
    for i,task in enumerate(tasks):
        arms=['off','manual'];rng.shuffle(arms)
        for arm in arms:
            out=root/'episodes'/f"{i+1:03}_{task['task_id'].split(':')[-1]}"/arm
            common.frontier.native_import(bundle,out/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'],42)
            actual=deepcopy(cfg);actual['data_dir']=str(out/'bank');actual['experiment']['output_dir']=str(out)
            common.write_json(out/'config.json',actual)
            episodes.append({'task_id':task['task_id'],'arm':arm,'path':str(out),'config_hash':common.sha(out/'config.json'),
                             'bank_identity':common.tree_identity(out/'bank')})
    protected=[base/'budget.json',base/'plan.json',base/'launch_status.json',base/'tasks.json',
        resources/'manifest.json',canonical_path,root/'tasks.json',root/'base_config.json',root/'budget_prefix.json']
    protected += [resources/name for name in manifest['files']]
    plan={'created_at':common.utc(),'code':code,'method_version':VERSION,'package_identity':common.package_identity(),
        'base_run':str(base),'resources':str(resources),
        'authorization':'2026-10-20 user explicitly approved cumulative 155M tokens and Bank plus NoSkill Test102 each. '
            'One frozen paired evaluation, per episode 216,576 tokens, HTTP bounded by existing loop and retry limits. '
            'Retain all historical charges and unknown reservations. No learning, Val, added Train or automatic repeat.',
        'selection':'all 102 canonical Test members in canonical order; seed42 fixes paired arm order; no result filtering',
        'development_exposed':True,'prior_scores_used_for_selection':False,
        'comparison_caveat':'Previously inspected Test set; human-authored reference Bank. Both fresh arms share external public input, scorer, tools, task budgets and final-output contract. Not a blind or learned-Ours result.',
        'seed':42,'episodes':episodes,'corpus_identity':parent['corpus_identity'],
        'protected_files':{str(p):common.sha(p) for p in protected},'ledger_limits':limits,
        'authorized_cumulative_ceiling':CUMULATIVE_CAP,'inherited_used_or_reserved':used,
        'new_token_cap':token_cap,'new_http_cap':http_cap,'fresh_paired_episodes':204,'test_runs_authorized_in_this_plan':2,
        'http_cap_basis':{'decisions_per_episode':decisions,'requests_per_decision_including_repair':2,
            'final_requests':1,'physical_attempts_per_request':cfg['llm']['max_retries']+1},
        'asset_origin':'human_authored_reference_not_learned_ours',
        'workflow_execution_mode':'native_dynamic_tool_loop_with_explicit_final_program_submission',
        'shared_answer_contract':common.frontier.ANSWER_CONTRACT,
        'locks':list(dict.fromkeys([str(base/'.run.lock'),*parent['locks']]))}
    common.save_json(root/'plan.json',plan);common.save_json(root/'plan_identity.json',{'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in
        ('base_run','authorization','inherited_used_or_reserved','authorized_cumulative_ceiling','new_token_cap','new_http_cap')})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()})
    print(json.dumps({'root':str(root),'episodes':204,'new_token_cap':token_cap,'cumulative_max':used+token_cap,'new_http_cap':http_cap}))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','check','launch','run','status']);ap.add_argument('--root',required=True)
    ap.add_argument('--base-run',default='/home/yangchengyu/officeqa_humanstrong_v3_train12_pairs_seed42_20261020')
    ap.add_argument('--resources',default='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare':prepare(root,args.base_run,args.resources)
    elif args.action=='check':common.verify(root);print('Identity and billing precheck passed; zero HTTP.')
    elif args.action=='status':print(json.dumps(common.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(common.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched; inspect first')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),'--env-file',args.env_file],
                    cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n');print(json.dumps({'pid':child.pid,'root':str(root)}))
