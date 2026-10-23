"""One immutable V14 Test102 per model; carry every historical cost forward."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v13_crossmodel as previous
from run_office_humanstrong_v2 import load

common=previous.common
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v13_crossmodel_stream_resume_seed42_20261022')
DEEPSEEK=previous.PARENT
PACKAGE=common.SOURCE/'resources/office_humanstrong_v14'
MODELS=('deepseek','kimi_k26','glm53_flash')
TASK_CAP=216576
VERSION='office.humanstrong.submission-facts.v14.20261023'


def bind_runner():
    common.PACKAGE=PACKAGE
    common.frontier=load('office_v14_frontier',PACKAGE/'integration/frontier.py')
    common.VERSION=VERSION
    previous.MODELS=MODELS


def config_for(model,resources):
    config=load('office_v14_configuration',PACKAGE/'integration/configuration.py').configure(
        common.read(DEEPSEEK/'base_config.json'),resources,'test')
    if model!='deepseek':config=previous.model_config(config,model)
    config['llm']['purpose_overrides']['finish_only']['protocol']['thinking_type']='enabled'
    return common.validate_config(config)


def preview():
    parent,ledger,_=previous.fixed.paired.settled_parent(PARENT)
    used=sum(map(common.charge,ledger['attempts'].values()))
    cap=102*len(MODELS)*TASK_CAP
    return {'models':MODELS,'episodes':306,'inherited_used_or_reserved':used,
            'new_token_cap':cap,'cumulative_max':used+cap,'previous_ceiling':parent['authorized_cumulative_ceiling']}


def prepare(root,ceiling):
    if root.exists():raise ValueError('Use a fresh directory')
    if ceiling>270000000:raise ValueError('Above the explicitly authorized cumulative ceiling')
    code=common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit before preparing paid execution')
    parent,ledger,locks=previous.fixed.paired.settled_parent(PARENT)
    amount=preview()
    if amount['cumulative_max']>ceiling:raise ValueError('Insufficient explicitly authorized cumulative ceiling')
    resources=Path(parent['resources'])
    configuration=load('office_v14_config_verify',PACKAGE/'integration/configuration.py')
    configuration.verify_resources(resources)
    task_doc=common.read(resources/'test.json');tasks=task_doc['tasks']
    old=common.read(DEEPSEEK/'tasks.json')['tasks']
    if tasks!=old or len(tasks)!=102:raise ValueError('Canonical external Test102 changed')
    configs={model:config_for(model,resources) for model in MODELS}
    decisions=max(16,configs['deepseek']['runtime']['global_action_budget']*4+16)
    http_cap=306*(decisions*2+1)
    limits={**ledger['limits'],'token_limit':amount['cumulative_max'],
            'request_limit':len(ledger['attempts'])+http_cap,'finish_reserve':0}
    root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json',ledger)
    ledger['limits']=limits;common.save_json(root/'budget.json',ledger)
    common.save_json(root/'tasks.json',task_doc)
    episodes=[];bundle=common.read(PACKAGE/'bank/bank.json')
    for model,cfg in configs.items():
        cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'])
        cfg['experiment'].update(runtime_mode='frozen',seed=42,output_dir=str(root/model))
        cfg['data_dir']=str(root/model/'episodes')
        common.write_json(root/(model+'_config.json'),cfg)
    common.write_json(root/'base_config.json',configs['deepseek'])
    for i,task in enumerate(tasks):
        for model,cfg in configs.items():
            target=root/model/'episodes'/f"{i+1:03}_{task['task_id'].split(':')[-1]}"/'manual'
            common.frontier.native_import(bundle,target/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'],42)
            actual=deepcopy(cfg);actual['data_dir']=str(target/'bank');actual['experiment']['output_dir']=str(target)
            common.write_json(target/'config.json',actual)
            episodes.append({'task_id':task['task_id'],'model_key':model,'arm':'manual','path':str(target),
                'config_hash':common.sha(target/'config.json'),'bank_identity':common.tree_identity(target/'bank')})
    protected=dict(parent['protected_files'])
    for file in [PARENT/'budget.json',PARENT/'plan.json',PARENT/'launch_status.json',
                 root/'tasks.json',root/'base_config.json',root/'budget_prefix.json',Path(__file__).resolve(),
                 *[root/(m+'_config.json') for m in MODELS]]:
        protected[str(file)]=common.sha(file)
    plan={**parent,'created_at':common.utc(),'code':code,'method_version':VERSION,'package_identity':common.package_identity(),
        'base_run':str(PARENT),'inference_config_source':str(DEEPSEEK),'episodes':episodes,
        'protected_files':protected,'ledger_limits':limits,'inherited_used_or_reserved':amount['inherited_used_or_reserved'],
        'new_token_cap':amount['new_token_cap'],'new_http_cap':http_cap,'authorized_cumulative_ceiling':ceiling,
        'test_runs_authorized_in_this_plan':3,'models':list(MODELS),'comparison_arms':['manual'],
        'controller_sha256':common.sha(Path(__file__).resolve()),'observation_prefix_path':str(root/'budget_prefix.json'),
        'authorization':'User requested revising the native AtomicSkill human Bank, public fact reminders and reliable submission; '
            'evaluate DeepSeek/Kimi/GLM Test102 aiming >=73 correct, preferably >=77. '
            '2026-10-23 user explicitly approved cumulative 270M and one Test102 per model after offline checks. '
            'One frozen full run per model; no automatic repetition, score changes or test-time asset edits.',
        'selection':'All canonical external-aligned Test102, unchanged order, alternate models at task boundaries.',
        'selection_receipt':{'rule':'Complete Test102; no member filtering. Bank revised using prior exposed development traces.'},
        'development_exposed':True,'prior_scores_used_for_selection':False,'fresh_paired_episodes':0,
        'comparison_caveat':'Human-authored native Skill/Implementation assets with deterministic public-input selection. '
            'Not naturally learned or blind Test evidence. Legacy scores remain unchanged.',
        'shared_answer_contract':'Native runtime_step.finish, complete bare scalar stop responses, or structured finish_answer; '
            'optional scalar_display submits an acquired final quantity without model rewriting. '
            'Default number only; explicit task output templates take precedence.',
        'runtime_changes':['explicit scalar/text submission','structured final answer','rules retained during finalization',
                           'all finish thinking enabled','shared Docker disconnect raises infrastructure failure'],
        'http_cap_basis':{'decisions_per_episode':decisions,'requests_per_decision_including_repair':2,
                         'final_requests':1,'physical_attempts_per_request':1},
        'locks':locks,'repeat_index':None}
    for name in ('failed_start_recovery','recovery'):plan.pop(name,None)
    common.save_json(root/'plan.json',plan)
    common.save_json(root/'plan_identity.json',{'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in (
        'authorization','base_run','inherited_used_or_reserved','authorized_cumulative_ceiling','new_token_cap','new_http_cap')})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()})
    common.verify(root)
    print(json.dumps(amount))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preview','prepare','check','launch','run','status'))
    parser.add_argument('--root');parser.add_argument('--cumulative-token-limit',type=int,default=270000000)
    parser.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=parser.parse_args();bind_runner()
    if args.action=='preview':print(json.dumps(preview()));sys.exit(0)
    if not args.root:parser.error('--root required')
    root=Path(args.root).resolve()
    if args.action=='prepare':prepare(root,args.cumulative_token_limit)
    elif args.action=='check':common.verify(root);print('Immutable source, inputs, Bank and cumulative ledger checked; zero HTTP.')
    elif args.action=='status':print(json.dumps(previous.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(previous.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            image=common.read(root/'base_config.json')['program_environment']['image_digest']
            subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL,timeout=20)
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n');print(json.dumps({'pid':child.pid,'root':str(root)}))
