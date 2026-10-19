"""Frozen authored Bank/off comparison on a preregistered public Train prefix."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

from filelock import FileLock
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor, billing_blocked
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity, utc
from atomic_skillgraph.experiments.run_empirical import code_identity, load_env, run, write_json, validate_config

SOURCE=Path(__file__).resolve().parents[1]
PACKAGE=SOURCE/'resources/office_humanstrong_v1'
spec=importlib.util.spec_from_file_location('office_phase_frontier',PACKAGE/'integration/frontier.py')
frontier=importlib.util.module_from_spec(spec);spec.loader.exec_module(frontier)
VERSION=frontier.VERSION
read=frontier.read
charge=lambda a:a.get('accounted_tokens',a['reserved_tokens'])

def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def package_identity():
    files={p.relative_to(PACKAGE).as_posix():sha(p) for p in sorted(PACKAGE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    return {'sha256':digest(files),'files_sha256':files}

def prepare(root,base,resources,train_count=12,cumulative_token_limit=98000000):
    if train_count!=12:raise ValueError('This authorization fixes exactly 12 paired Train tasks')
    if cumulative_token_limit!=98000000:raise ValueError('This authorization fixes the cumulative ceiling at 98M')
    if root.exists():raise ValueError('Use a fresh independent directory')
    identity=code_identity()
    if identity['tracked_dirty'] or not identity['git_sha']:raise ValueError('Commit before preparing a paid run')
    base,resources=Path(base).resolve(),Path(resources).resolve()
    ledger=read(base/'budget.json');used=sum(map(charge,ledger['attempts'].values()))
    if billing_blocked(ledger):raise ValueError('Unresolved billing prevents a new paid attempt')
    assert read(base/'launch_status.json')['status']=='completed'
    task_doc=read(resources/'train.json')
    selected=task_doc['tasks'][:train_count]
    canonical=read(SOURCE/'data/main_experiment_v1/officeqa/train.json')['tasks']
    assert [x['task_id'] for x in selected]==[x['task_id'] for x in canonical[:train_count]]
    assert not {t['task_id'] for t in selected}&{x['task_id'] for x in read(resources/'test.json')['tasks']}
    cap=2*train_count*192000
    if used+cap>cumulative_token_limit:raise ValueError(f"Insufficient cumulative authority: need {used+cap}, allowed {cumulative_token_limit}")
    root.mkdir(parents=True)
    save_json(root/'budget_prefix.json',ledger)
    # Same cumulative charges; a stricter local observation cap, not new campaign authority.
    limits={**ledger['limits'],'token_limit':used+cap,'request_limit':len(ledger['attempts'])+200*train_count,'finish_reserve':0}
    ledger['limits']=limits;save_json(root/'budget.json',ledger)
    save_json(root/'tasks.json',{'tasks':selected})
    cfg=deepcopy(read(base/'base_config.json'))
    cfg['harness'].update(evaluator_records=str(resources/'evaluator_records_train.json'),split='train')
    cfg['manifest']=str(resources/'train.json')
    cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'],
                         train_task_tokens=192000,eval_task_tokens=192000)
    cfg['experiment']['runtime_mode']='frozen'
    cfg['llm']['runtime']['max_total_tokens_per_task']=192000
    cfg=validate_config(cfg)
    save_json(root/'base_config.json',cfg)
    bundle=read(PACKAGE/'bank/bank.json');rng=random.Random(42);episodes=[]
    for i,t in enumerate(selected):
        arms=['off','manual'];rng.shuffle(arms)
        for arm in arms:
            name=f"{i+1:02}_{t['task_id'].split(':')[-1]}/{arm}"
            out=root/'episodes'/name
            imported=frontier.native_import(bundle,out/'bank',cfg['program_environment'],
                cfg['experiment'].get('officeqa_method_version'),42)
            arm_cfg=deepcopy(cfg);arm_cfg['data_dir']=str(out/'bank')
            arm_cfg['experiment']['output_dir']=str(out)
            write_json(out/'config.json',arm_cfg)
            episodes.append({'task_id':t['task_id'],'arm':arm,'path':str(out),
                'config_hash':sha(out/'config.json'),'bank_identity':tree_identity(out/'bank')})
    files=[base/'budget.json',base/'plan.json',base/'launch_status.json',resources/'train.json',
           resources/'evaluator_records_train.json',root/'tasks.json',root/'base_config.json',
           root/'budget_prefix.json']
    plan={'created_at':utc(),'code':identity,'method_version':VERSION,'package_identity':package_identity(),
        'authorization':'2026-10-19 user explicitly approved fresh HumanStrong/NoSkill Train12 paired observation; cumulative 98M tokens; new <=4,608,000 tokens / 2,400 physical HTTP; preserve all historical charges and unknown reservations; no Val/Test.',
        'selection':f'first {train_count} canonical Train members, no score filtering; paired order randomized once with seed42',
        'seed':42,'episodes':episodes,'base_run':str(base),'resources':str(resources),
        'corpus_identity':read(base/'plan.json')['corpus_identity'],
        'protected_files':{str(p):sha(p) for p in files},
        'original_cumulative_ceiling':95000000,'authorized_cumulative_ceiling':cumulative_token_limit,'inherited_used_or_reserved':used,
        'new_token_cap':cap,'new_http_cap':200*train_count,'ledger_limits':limits,
        'asset_origin':'external_humanstrong_reference_not_learned_ours',
        'workflow_execution_mode':'native_dynamic_tool_loop_no_automatic_graph_dispatch',
        'fresh_paired_episodes':24,'prior_scores_used_for_selection':False,'test_runs_authorized_in_this_plan':0,
        'locks':[str(base/'.run.lock'),*read(base/'plan.json')['locks']]}
    save_json(root/'plan.json',plan);save_json(root/'plan_identity.json',{'sha256':sha(root/'plan.json')})
    save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in ['base_run','inherited_used_or_reserved',
        'original_cumulative_ceiling','authorized_cumulative_ceiling','new_token_cap','new_http_cap','authorization']})
    save_json(root/'launch_status.json',{'status':'prepared','created_at':utc()})
    print(json.dumps({'root':str(root),'episodes':[(e['task_id'],e['arm']) for e in episodes],
                      'new_token_cap':cap,'cumulative_max':used+cap},ensure_ascii=False))

def verify(root):
    plan=read(root/'plan.json')
    assert sha(root/'plan.json')==read(root/'plan_identity.json')['sha256']
    assert code_identity()==plan['code'] and not plan['code']['tracked_dirty']
    assert package_identity()==plan['package_identity']
    assert all(sha(path)==hash_ for path,hash_ in plan['protected_files'].items())
    assert read(root/'budget.json')['limits']==plan['ledger_limits']
    assert not billing_blocked(read(root/'budget.json'))
    for e in plan['episodes']:
        assert sha(Path(e['path'])/'config.json')==e['config_hash']
        assert tree_identity(Path(e['path'])/'bank')==e['bank_identity']
    return plan

class GuardedGovernor(BudgetGovernor):
    def __init__(self,root,plan):
        self.root,self.plan=root,plan
        super().__init__(root/'budget.json',**plan['ledger_limits'])
    def admit(self,*args,**kwargs):
        parent=Path(self.plan['base_run'])/'budget.json'
        if sha(parent)!=self.plan['protected_files'][str(parent)]:
            raise RuntimeError('Prior campaign ledger changed; refuse stale cumulative authority')
        return super().admit(*args,**kwargs)

def require_open_billing(governor):
    if governor.billing_blocked:
        raise RuntimeError('infrastructure_billing_unknown: preserve the reservation and stop before the next episode')

def status(root):
    plan=read(root/'plan.json');ledger=read(root/'budget.json');prefix=read(root/'budget_prefix.json')['attempts']
    new=[v for k,v in ledger['attempts'].items() if k not in prefix]
    observation_prefix=read(plan.get('observation_prefix_path',root/'budget_prefix.json'))['attempts']
    observation=[v for k,v in ledger['attempts'].items() if k not in observation_prefix]
    results=[]
    for episode in plan['episodes']:
        summary=Path(episode['path'])/'summary.json'
        if summary.exists():
            s=read(summary);results.append({'task_id':episode['task_id'],'arm':episode['arm'],
                'complete':s['complete'],'correct':s['successes'],'tokens':s['total_tokens'],
                'known_tokens':s['known_total_tokens'],'unknown_billing_attempts':s['unknown_billing_attempts'],
                'retained':bool(episode.get('retained_from')),'finish_recovery':bool(episode.get('recovery_source')),
                'program_calls':s['program_invocations'],
                'terminal_reasons':[x['execution']['reason'] for x in s['cases']]})
    return {'launch':read(root/'launch_status.json'),'completed_episodes':len(results),'expected_episodes':len(plan['episodes']),
        'results':results,'new_tokens_used_or_reserved':sum(map(charge,new)),'new_http':len(new),
        'new_unsettled':sum(v['status']!='metered' for v in new),
        'observation_tokens_used_or_reserved':sum(map(charge,observation)),'observation_http':len(observation),
        'cumulative_tokens_used_or_reserved':sum(map(charge,ledger['attempts'].values())),
        'new_token_cap':plan['new_token_cap'],'new_http_cap':plan['new_http_cap']}

def execute(root,env):
    plan=verify(root);started=utc();exit_code=1;state='failed'
    with ExitStack() as stack:
        for path in [str(root/'.run.lock'),*sorted(set(plan['locks']))]:
            stack.enter_context(FileLock(path,timeout=0))
        try:
            cfg=read(root/'base_config.json')
            assert tree_identity(cfg['harness']['corpus_root'])==plan['corpus_identity']
            load_env(env)
            governor=GuardedGovernor(root,plan)
            tasks={t['task_id']:PublicTask(**t) for t in read(root/'tasks.json')['tasks']}
            for i,e in enumerate(plan['episodes']):
                out=Path(e['path']);summary=out/'summary.json'
                if summary.exists() and read(summary)['complete']:continue
                require_open_billing(governor)
                if (root/'STOP_AFTER_EPISODE').exists():state='paused';exit_code=75;break
                save_json(root/'launch_status.json',{'status':'running','started_at':started,
                    'pid':os.getpid(),'episode_index':i,'task_id':e['task_id'],'arm':e['arm']})
                config=read(out/'config.json');bundle=read(out/'bank/imported_bank.json')
                formal=FormalLog(out/'formal',{'code':plan['code'],'method':VERSION,'arm':e['arm'],
                    'plan_sha256':sha(root/'plan.json'),'origin':plan['asset_origin']},config,
                    resume=(out/'formal/run_manifest.json').exists())
                try:
                    with frontier.bridge():
                        result=run(config,[tasks[e['task_id']]],out,readonly=True,
                            resume=(out/'execution_manifest.json').exists(),formal_log=formal,
                            bank_view_factory=lambda bank:frontier.HumanBankView(bank,bundle,e['arm']),
                            budget_governor=governor,freeze_on_complete=False)
                    require_open_billing(governor)
                    formal.finish('completed' if result['complete'] else 'stopped')
                    if not result['complete']:raise RuntimeError('Incomplete episode; inspect the original record')
                except BaseException as exc:
                    formal.task_error(exc);formal.finish('failed');raise
                finally:
                    after=tree_identity(out/'bank')
                    save_json(out/'freeze_record.json',{'before':e['bank_identity'],'after':after,
                        'unchanged':after==e['bank_identity'],'checked_at':utc()})
                    assert after==e['bank_identity']
                print(json.dumps(status(root),ensure_ascii=False),flush=True)
            else:state='completed';exit_code=0
        except Exception as exc:
            save_json(root/'last_error.json',{'time':utc(),'type':type(exc).__name__,'message':str(exc)})
            traceback.print_exc()
        finally:
            save_json(root/'launch_status.json',{'status':state,'started_at':started,'ended_at':utc(),
                'pid':os.getpid(),'exit_code':exit_code})
            receipt=status(root);save_json(root/'billing_result.json',receipt)
            if state=='completed':save_json(root/'comparison.json',receipt)
    return exit_code

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','check','launch','run','status'])
    ap.add_argument('--root',required=True)
    ap.add_argument('--base-run',default='/home/yangchengyu/officeqa_human_phase_v31_train12_pairs_seed42_recovery_20261018')
    ap.add_argument('--resources',default='/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/officeqa')
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    ap.add_argument('--train-count',type=int,choices=[12],default=12)
    ap.add_argument('--cumulative-token-limit',type=int,default=98000000)
    args=ap.parse_args();root=Path(args.root).resolve()
    if args.action=='prepare':prepare(root,args.base_run,args.resources,args.train_count,args.cumulative_token_limit)
    elif args.action=='status':print(json.dumps(status(root),ensure_ascii=False,indent=2))
    elif args.action=='check':verify(root);print('Offline identity precheck passed; zero HTTP.')
    elif args.action=='run':sys.exit(execute(root,args.env_file))
    else:
        verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched; inspect status before any resume')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'started_pid':child.pid,'root':str(root)}))
