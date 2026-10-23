"""Bounded six-model V15 diagnosis, after the unchanged V14 campaign settles."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v14 as previous
from run_office_humanstrong_v2 import load

common=previous.common
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v14_three_model_test102_seed42_20261023')
PACKAGE=common.SOURCE/'resources/office_humanstrong_v15'
MODELS=('deepseek','kimi_k26','glm53_flash','minimax_m3','gemini38_flash','qwen38_flash')
VERSION='office.humanstrong.operation-bound.v15.20261024'
TASK_CAP=216576
TASK_CAPS={m:(400000 if m=='qwen38_flash' else TASK_CAP) for m in MODELS}
PROBE=('UID0091','UID0070','UID0232','UID0202','UID0014','UID0149','UID0237','UID0140','UID0123','UID0101','UID0124','UID0053')
TRANSPORTS={
    'kimi_k26':'searchqa_v7_crossmodel_test1400_20261021/kimi_k26_config.json',
    'glm53_flash':'searchqa_v7_crossmodel_test1400_20261021/glm53_flash_config.json',
    'minimax_m3':'searchqa_v7_qwen_minimax_test1400_20261022/minimax_m3_config.json',
    'qwen38_flash':'searchqa_v7_qwen_minimax_test1400_20261022/qwen38_flash_config.json',
    'gemini38_flash':'searchqa_v7_gemini38_test1400_20261023/base_config.json'}


def bind_runner():
    common.PACKAGE=PACKAGE;common.VERSION=VERSION
    common.frontier=load('office_v15_frontier',PACKAGE/'integration/frontier.py')
    previous.previous.MODELS=MODELS


def config_for(model,resources):
    cfg=load('office_v15_configuration',PACKAGE/'integration/configuration.py').configure(
        common.read(PARENT/'base_config.json'),resources,'test')
    if model!='deepseek':
        transport=common.read(common.SOURCE/'experiments'/TRANSPORTS[model])['llm']
        for key in ('provider','dialect','base_url','model','api_key_env','capability_profile','stream','token_limit_field','generation_seed_supported','thinking_budget'):
            if key in transport:cfg['llm'][key]=deepcopy(transport[key])
        cfg['llm']['display_name']=transport['model']
        if not transport['capability_profile'].get('reasoning_effort'):
            def remove_effort(value):
                if isinstance(value,dict):
                    value.pop('reasoning_effort',None)
                    for k,v in value.items():
                        if k!='capability_profile':remove_effort(v)
            remove_effort(cfg['llm'])
    cfg['llm']['max_retries']=0
    cfg['llm']['runtime']['request_timeout_seconds']=600
    cfg['llm']['runtime']['max_total_tokens_per_task']=TASK_CAPS[model]
    cfg['budget']['eval_task_tokens']=TASK_CAPS[model]
    return common.validate_config(cfg)


def prepare(root):
    if root.exists():raise ValueError('Use a fresh directory')
    code=common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit before paid execution')
    parent,ledger,locks=previous.previous.fixed.paired.settled_parent(PARENT)
    resources=Path(parent['resources']);tasks_doc=common.read(PARENT/'tasks.json')
    selected=[t for t in tasks_doc['tasks'] if t['task_id'].split(':')[-1] in PROBE]
    if len(selected)!=len(PROBE):raise ValueError('Incomplete diagnostic selection')
    amount=sum(map(common.charge,ledger['attempts'].values()));cap=len(selected)*sum(TASK_CAPS.values())
    if amount+cap>270000000:raise ValueError('Insufficient already-authorized cumulative 270M')
    configs={m:config_for(m,resources) for m in MODELS}
    decisions=max(16,configs['deepseek']['runtime']['global_action_budget']*4+16)
    http_cap=len(selected)*len(MODELS)*(decisions*2+1)
    limits={**ledger['limits'],'token_limit':amount+cap,'request_limit':len(ledger['attempts'])+http_cap,'finish_reserve':0}
    root.mkdir(parents=True);common.save_json(root/'budget_prefix.json',ledger)
    ledger['limits']=limits;common.save_json(root/'budget.json',ledger)
    common.save_json(root/'tasks.json',{**tasks_doc,'tasks':selected})
    bundle=common.read(PACKAGE/'bank/bank.json');episodes=[]
    for model,cfg in configs.items():
        cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'])
        cfg['experiment'].update(runtime_mode='frozen',seed=42,output_dir=str(root/model))
        cfg['data_dir']=str(root/model/'episodes');common.write_json(root/(model+'_config.json'),cfg)
    common.write_json(root/'base_config.json',configs['deepseek'])
    order={t['task_id']:i+1 for i,t in enumerate(tasks_doc['tasks'])}
    for task in selected:
        for model,cfg in configs.items():
            target=root/model/'episodes'/f"{order[task['task_id']]:03}_{task['task_id'].split(':')[-1]}"/'manual'
            common.frontier.native_import(bundle,target/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'],42)
            actual=deepcopy(cfg);actual['data_dir']=str(target/'bank');actual['experiment']['output_dir']=str(target)
            common.write_json(target/'config.json',actual)
            episodes.append({'task_id':task['task_id'],'model_key':model,'arm':'manual','path':str(target),
                'canonical_index':order[task['task_id']],'config_hash':common.sha(target/'config.json'),'bank_identity':common.tree_identity(target/'bank')})
    protected=dict(parent['protected_files'])
    for file in [PARENT/'budget.json',PARENT/'plan.json',PARENT/'launch_status.json',root/'tasks.json',root/'base_config.json',root/'budget_prefix.json',Path(__file__).resolve(),
                 *[root/(m+'_config.json') for m in MODELS],*[common.SOURCE/'experiments'/p for p in TRANSPORTS.values()]]:
        protected[str(file)]=common.sha(file)
    plan={**parent,'created_at':common.utc(),'code':code,'method_version':VERSION,'package_identity':common.package_identity(),
        'base_run':str(PARENT),'inference_config_source':str(PARENT),'episodes':episodes,'protected_files':protected,'ledger_limits':limits,
        'inherited_used_or_reserved':amount,'new_token_cap':cap,'new_http_cap':http_cap,'authorized_cumulative_ceiling':270000000,
        'controller_sha256':common.sha(Path(__file__).resolve()),'observation_prefix_path':str(root/'budget_prefix.json'),
        'models':list(MODELS),'task_caps':TASK_CAPS,'comparison_arms':['manual'],'test_runs_authorized_in_this_plan':0,
        'authorization':'2026-10-24 user requested effective native Bank improvements for all six existing model configurations and retained prior paid-test authorization. '
            'User additionally approved Qwen existing 400,000 per-task cap with separately metered thinking; other five remain 216,576. '
            'One bounded 12-task diagnostic per model within existing cumulative 270M. Current V14 must finish first. No full Test102, automatic repeats, scorer edits or increased ceiling.',
        'selection':'Twelve predeclared exposed development tasks: two successful controls, two list-submission failures, and eight calculation/extraction failures. Keep canonical relative order.',
        'selection_receipt':{'ids':list(PROBE),'based_on':'V14 exposed failure analysis and historical successful processes; selection before any V15 paid response'},
        'development_exposed':True,'prior_scores_used_for_selection':True,'fresh_paired_episodes':0,
        'comparison_caveat':'Targeted development diagnosis, not full-Test accuracy. V14 is a historical comparator for three models, not a fresh randomized control. Six models share the same frozen Bank and public inputs.',
        'shared_answer_contract':'Native runtime_step.finish, structured finish_answer, complete bare scalar or flat-list stop response; preserve the submitted text. Linked Programs may submit their computed final answer directly.',
        'runtime_changes':['opt-in whole flat-list final submission; V1 scalar protocol preserved',
            'admission and finish reservation include separately metered thinking from the existing transport'],
        'http_cap_basis':{'decisions_per_episode':decisions,'requests_per_decision_including_repair':2,'final_requests':1,'physical_attempts_per_request':1},
        'locks':locks,'repeat_index':None}
    common.save_json(root/'plan.json',plan);common.save_json(root/'plan_identity.json',{'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in ('authorization','base_run','inherited_used_or_reserved','authorized_cumulative_ceiling','new_token_cap','new_http_cap')})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()});common.verify(root)
    print(json.dumps({'episodes':len(episodes),'new_cap':cap,'cumulative_max':amount+cap}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','check','launch','run','status'));parser.add_argument('--root',required=True)
    parser.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=parser.parse_args();bind_runner();root=Path(args.root).resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='check':common.verify(root);print('Frozen source, native Bank, configured providers and cumulative ledger verified; zero HTTP.')
    elif args.action=='status':print(json.dumps(previous.previous.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(previous.previous.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            subprocess.run(['docker','image','inspect',common.read(root/'base_config.json')['program_environment']['image_digest']],check=True,stdout=subprocess.DEVNULL,timeout=20)
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),'--env-file',args.env_file],
                    cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()});(root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid':child.pid,'root':str(root)}))
