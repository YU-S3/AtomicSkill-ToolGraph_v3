"""One fresh, bounded V3/NoSkill Train12 observation; retain all prior billing."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import random
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v1 as common
from run_office_humanstrong_v2 import load

PACKAGE=common.SOURCE/'resources/office_humanstrong_v3'
VERSION='office.humanstrong.v3.aligned-train12-pairs.20261020'
TASK_CAP=216576
TOKEN_CAP,HTTP_CAP,CUMULATIVE_CAP=24*TASK_CAP,2400,116000000


def bind_runner():
    common.PACKAGE=PACKAGE
    common.frontier=load('office_v3_paired_frontier',PACKAGE/'integration/frontier.py')
    common.VERSION=VERSION


def prepare(root,base,comparison,resources):
    root,base,comparison,resources=map(lambda p:Path(p).resolve(),(root,base,comparison,resources))
    if root.exists():raise ValueError('Use a fresh independent directory')
    code=common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit before preparing a paid run')
    if common.read(base/'status.json')['status']!='completed':raise ValueError('Billing parent must be complete')
    if common.read(comparison/'launch_status.json')['status']!='completed':raise ValueError('Previous pairs must be complete')
    parent=common.read(comparison/'plan.json');ledger=common.read(base/'budget.json')
    if common.billing_blocked(ledger):raise ValueError('Resolve billing before a new run')
    old=common.read(comparison/'budget.json')['attempts']
    if any(ledger['attempts'].get(k)!=v for k,v in old.items()):raise ValueError('Billing ancestry mismatch')
    used=sum(map(common.charge,ledger['attempts'].values()))
    if used+TOKEN_CAP>CUMULATIVE_CAP:raise ValueError('Insufficient cumulative authority')
    configuration=load('office_v3_paired_config',PACKAGE/'integration/configuration.py')
    resources,manifest=configuration.verify_resources(resources)
    tasks=common.read(resources/'train.json')['tasks'][:12]
    canonical_path=common.SOURCE/'data/main_experiment_v1/officeqa/train.json'
    canonical=common.read(canonical_path)['tasks'][:12]
    assert len(tasks)==12 and all(t['split']=='train' for t in tasks)
    assert [t['task_id'] for t in tasks]==[t['task_id'] for t in canonical]
    assert tasks==common.read(comparison/'tasks.json')['tasks']
    cfg=configuration.configure(common.read(comparison/'base_config.json'),resources,'train')
    assert cfg['runtime']['global_action_budget']==48
    assert cfg['llm']['runtime']['max_total_tokens_per_task']==TASK_CAP
    cfg['experiment'].update(runtime_mode='frozen',seed=42,output_dir=str(root),observation_version=VERSION)
    cfg['data_dir']=str(root/'episodes');root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json',ledger)
    limits={**ledger['limits'],'token_limit':used+TOKEN_CAP,'request_limit':len(ledger['attempts'])+HTTP_CAP,'finish_reserve':0}
    ledger['limits']=limits;common.save_json(root/'budget.json',ledger)
    cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'],
                         train_task_tokens=TASK_CAP,eval_task_tokens=TASK_CAP)
    common.write_json(root/'base_config.json',cfg);common.save_json(root/'tasks.json',{'tasks':tasks})
    bundle=common.read(PACKAGE/'bank/bank.json');episodes=[];rng=random.Random(42)
    for i,task in enumerate(tasks):
        arms=['off','manual'];rng.shuffle(arms)
        for arm in arms:
            out=root/'episodes'/f"{i+1:02}_{task['task_id'].split(':')[-1]}"/arm
            common.frontier.native_import(bundle,out/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'],42)
            actual=deepcopy(cfg);actual['data_dir']=str(out/'bank');actual['experiment']['output_dir']=str(out)
            common.write_json(out/'config.json',actual)
            episodes.append({'task_id':task['task_id'],'arm':arm,'path':str(out),'config_hash':common.sha(out/'config.json'),
                             'bank_identity':common.tree_identity(out/'bank')})
    protected=[base/'budget.json',base/'plan.json',base/'status.json',comparison/'plan.json',comparison/'budget.json',
        comparison/'tasks.json',comparison/'launch_status.json',resources/'manifest.json',canonical_path,
        root/'tasks.json',root/'base_config.json',root/'budget_prefix.json']
    protected += [resources/name for name in manifest['files']]
    diagnostics_parent=Path(common.read(base/'plan.json')['base_run'])
    protected += [diagnostics_parent/'budget.json',diagnostics_parent/'plan.json',diagnostics_parent/'status.json']
    plan={'created_at':common.utc(),'code':code,'method_version':VERSION,'package_identity':common.package_identity(),
        'base_run':str(base),'comparison_run':str(comparison),'resources':str(resources),
        'authorization':'2026-10-20 user requested Bank revision based on observed failures and another experiment, '
            'with standing paid testing authorization. Freeze one fresh canonical Train12 paired observation at '
            '24 x 216,576 tokens and <=2,400 HTTP inside the existing 116M cumulative ceiling. '
            'Both arms share external public inputs and the expanded finish settings. No learning, Val/Test or automatic repeat.',
        'selection':'same first 12 canonical Train members; no result filtering; arm order fixed with seed42',
        'development_exposed':True,'prior_scores_used_for_selection':False,
        'comparison_caveat':'Development-exposed Train pairs, not a blind accuracy estimate. Compare V3 versus fresh off within this run; old V2 used different finish settings.',
        'seed':42,'episodes':episodes,'corpus_identity':parent['corpus_identity'],
        'protected_files':{str(p):common.sha(p) for p in protected},'ledger_limits':limits,
        'authorized_cumulative_ceiling':CUMULATIVE_CAP,'inherited_used_or_reserved':used,
        'new_token_cap':TOKEN_CAP,'new_http_cap':HTTP_CAP,'fresh_paired_episodes':24,'test_runs_authorized_in_this_plan':0,
        'asset_origin':'human_authored_reference_not_learned_ours',
        'workflow_execution_mode':'native_dynamic_tool_loop_no_automatic_graph_dispatch',
        'locks':list(dict.fromkeys([str(base/'.run.lock'),str(diagnostics_parent/'.run.lock'),str(comparison/'.run.lock'),*parent['locks']]))}
    common.save_json(root/'plan.json',plan);common.save_json(root/'plan_identity.json',{'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in
        ('base_run','comparison_run','authorization','inherited_used_or_reserved','authorized_cumulative_ceiling','new_token_cap','new_http_cap')})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()})
    print(json.dumps({'root':str(root),'episodes':24,'new_token_cap':TOKEN_CAP,'cumulative_max':used+TOKEN_CAP}))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','check','launch','run','status']);ap.add_argument('--root',required=True)
    ap.add_argument('--base-run',default='/home/yangchengyu/officeqa_humanstrong_v2_finish32k_body_diagnostic_20261020')
    ap.add_argument('--comparison-run',default='/home/yangchengyu/officeqa_humanstrong_v2_train12_pairs_seed42_20261020')
    ap.add_argument('--resources',default='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare':prepare(root,args.base_run,args.comparison_run,args.resources)
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
