"""One frozen HumanStrong Test102 rerun, reusing the audited episode/billing runner."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v1 as common

VERSION = 'office.humanstrong.test102.evidence.v1.20261019'


def prepare(root, base, resources):
    root, base, resources = Path(root), Path(base).resolve(), Path(resources).resolve()
    if root.exists(): raise ValueError('Use a fresh independent Test directory')
    code = common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']: raise ValueError('Commit before preparing a paid run')
    parent = common.read(base/'plan.json')
    if common.read(base/'launch_status.json')['status']!='completed': raise ValueError('Parent run is not complete')
    ledger = common.read(base/'budget.json')
    if common.billing_blocked(ledger): raise ValueError('Resolve billing before a new run')
    used = sum(map(common.charge, ledger['attempts'].values()))
    tasks = common.read(resources/'test.json')['tasks']
    canonical = common.read(common.SOURCE/'data/main_experiment_v1/officeqa/test.json')['tasks']
    assert len(tasks)==102 and [t['task_id'] for t in tasks]==[t['task_id'] for t in canonical]
    assert all(t['split']=='test' for t in tasks)
    assert not {t['task_id'] for t in tasks}&{t['task_id'] for t in common.read(base/'tasks.json')['tasks']}
    asset_files = {name:h for name,h in parent['package_identity']['files_sha256'].items()
                   if name.startswith(('bank/','skills/','programs/','implementation/','workflows/')) or name=='SKILL.md'}
    assert all(common.sha(common.PACKAGE/name)==h for name,h in asset_files.items())
    cap = len(tasks)*192000
    if used+cap>116000000: raise ValueError('Insufficient authorized cumulative tokens')
    cfg = deepcopy(common.read(base/'base_config.json'))
    cfg['harness'].update(evaluator_records=str(resources/'evaluator_records_test.json'), split='test')
    cfg['manifest'] = str(resources/'test.json')
    cfg['experiment'].update(runtime_mode='frozen',seed=42,output_dir=str(root))
    cfg['data_dir']=str(root/'episodes')
    assert cfg['runtime']['global_action_budget']==48 and cfg['llm']['runtime']['max_total_tokens_per_task']==192000
    # Native Executor: 208 decisions, at most one repair, then one final answer;
    # each logical request permits the configured four retries (five HTTP attempts).
    decisions = max(16,cfg['runtime']['global_action_budget']*4+16)
    http_cap = len(tasks)*(decisions*2+1)*(cfg['llm']['max_retries']+1)
    root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json',ledger)
    limits={**ledger['limits'],'token_limit':used+cap,'request_limit':len(ledger['attempts'])+http_cap,'finish_reserve':0}
    ledger['limits']=limits;common.save_json(root/'budget.json',ledger)
    cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'],
                         train_task_tokens=192000,eval_task_tokens=192000)
    cfg=common.validate_config(cfg)
    common.write_json(root/'base_config.json',cfg)
    common.save_json(root/'tasks.json',{'tasks':tasks})
    bundle=common.read(common.PACKAGE/'bank/bank.json');episodes=[]
    for i,task in enumerate(tasks):
        out=root/'episodes'/f"{i+1:03}_{task['task_id'].split(':')[-1]}"/'manual'
        common.frontier.native_import(bundle,out/'bank',cfg['program_environment'],
            cfg['experiment'].get('officeqa_method_version'),42)
        actual=deepcopy(cfg);actual['data_dir']=str(out/'bank');actual['experiment']['output_dir']=str(out)
        common.write_json(out/'config.json',actual)
        episodes.append({'task_id':task['task_id'],'arm':'manual','path':str(out),
                         'config_hash':common.sha(out/'config.json'),'bank_identity':common.tree_identity(out/'bank')})
    protected=[base/'budget.json',base/'plan.json',base/'launch_status.json',resources/'test.json',
        resources/'evaluator_records_test.json',common.SOURCE/'data/main_experiment_v1/officeqa/test.json',
        root/'tasks.json',root/'base_config.json',root/'budget_prefix.json']
    plan={'created_at':common.utc(),'code':code,'method_version':VERSION,'package_identity':common.package_identity(),
        'authorization':'2026-10-19 user approved review recommendations and all corresponding paid experiments: '
            'one HumanStrong-only seed42 Test102 after offline acceptance; cumulative ceiling 116M tokens; '
            'preserve historic costs/reservations; no repeat Train12, no added Val or other arms.',
        'selection':'all 102 canonical Test members in manifest order; no score filtering',
        'seed':42,'episodes':episodes,'base_run':str(base),'resources':str(resources),
        'corpus_identity':parent['corpus_identity'],'protected_files':{str(p):common.sha(p) for p in protected},
        'original_cumulative_ceiling':98000000,'authorized_cumulative_ceiling':116000000,
        'inherited_used_or_reserved':used,'new_token_cap':cap,'new_http_cap':http_cap,'ledger_limits':limits,
        'http_cap_basis':{'decisions_per_episode':decisions,'requests_per_decision_including_repair':2,
                          'final_requests':1,'physical_attempts_per_request':cfg['llm']['max_retries']+1},
        'asset_origin':'external_humanstrong_reference_not_learned_ours','assets_unchanged_from_parent':asset_files,
        'workflow_execution_mode':'native_dynamic_tool_loop_no_automatic_graph_dispatch',
        'development_exposed':True,'comparison_caveat':'Previously inspected Test set; runtime evidence retention changed; not a blind evaluation or a pure Bank-content contrast to old NoSkill.',
        'test_runs_authorized_in_this_plan':1,'locks':[str(base/'.run.lock'),*parent['locks']]}
    common.save_json(root/'plan.json',plan)
    common.save_json(root/'plan_identity.json',{'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in ['base_run','authorization',
        'inherited_used_or_reserved','original_cumulative_ceiling','authorized_cumulative_ceiling','new_token_cap','new_http_cap']})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()})
    print({'root':str(root),'tasks':102,'new_tokens':cap,'cumulative_max':used+cap,'new_http':http_cap})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','check','launch','run','status'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--base-run',default='/home/yangchengyu/officeqa_humanstrong_v1_train12_pairs_seed42_20261019')
    parser.add_argument('--resources',default='/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/officeqa')
    parser.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=parser.parse_args();root=Path(args.root).resolve();common.VERSION=VERSION
    if args.action=='prepare':prepare(root,args.base_run,args.resources)
    elif args.action=='status':print(json.dumps(common.status(root),ensure_ascii=False,indent=2))
    elif args.action=='check':common.verify(root);print('Offline identity precheck passed; zero HTTP.')
    elif args.action=='run':sys.exit(common.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched; inspect before resuming')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print({'pid':child.pid,'root':str(root)})
