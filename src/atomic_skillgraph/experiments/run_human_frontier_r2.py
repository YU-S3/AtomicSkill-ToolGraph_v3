"""Frozen manual Train12 three-arm comparison, followed by the preregistered Test100 arms."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile

from ..empirical import human_frontier_runtime as rt
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, billing_blocked, input_token_bound
from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, digest
from ..harness.benchmarks import AnswerAdapter
from .build_human_frontier_r2 import VERSION, PACKAGE, build
from .canonical_manifest import sha256
from .formal_choice_batch import BatchGovernor
from .formal_log import tree_identity, utc
from .run_empirical import code_identity, load_env

read=rt.read
ARMS=['NoSkill','upstream_skillopt','upstream_skillopt_lite']
TEST_ARMS=['benchmark_prior','upstream_skillopt']
LIMITS={'token_limit':45000000,'request_limit':9165,'finish_reserve':0}
PHASES={'development':{'token_limit':3000000,'request_limit':72},
        'test_benchmark_prior':{'token_limit':6300000,'request_limit':200},
        'test_upstream_skillopt':{'token_limit':6300000,'request_limit':200}}


def integration(package=PACKAGE):
    path=Path(package)/'integration/frontier.py'
    spec=importlib.util.spec_from_file_location('human_frontier_r2_integration',path)
    module=importlib.util.module_from_spec(spec)
    # Importing must not add a __pycache__ file to the frozen package.
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def task_rows(path):
    raw=read(path);raw=raw['tasks'] if isinstance(raw,dict) else raw
    fields={'task_id','physical_key','goal','inputs','split'}
    return [{k:v for k,v in row.items() if k in fields} for row in raw]


def selection(train):
    groups={True:[],False:[]}
    for t in train:groups[bool(rt.meta_labels(PublicTask(**t)))].append(t)
    rank=lambda t:digest(['human-frontier-r2-train12-seed42',t['physical_key']])
    chosen=[]
    for rows in groups.values():chosen+=sorted(rows,key=rank)[:6]
    if len(chosen)<12:
        chosen+=sorted([t for t in train if t not in chosen],key=rank)[:12-len(chosen)]
    return sorted(chosen,key=rank)


def schedule(train, test, test_arms=TEST_ARMS):
    result=[]
    for i,t in enumerate(train):
        for a in ARMS[i%3:]+ARMS[:i%3]:
            result.append({'phase':'development','arm':a,'task':t,'task_order_index':i})
    for i,t in enumerate(test):
        for a in test_arms[i%len(test_arms):]+test_arms[:i%len(test_arms)]:
            result.append({'phase':'test_'+a,'arm':a,'task':t,'task_order_index':i})
    for i,row in enumerate(result):row['episode_order_index']=i
    return result


def episode_dir(root,row):return Path(root)/row['phase']/row['arm']/row['task']['physical_key']


def prepare(root, base_config, datasets, ledger, history_run):
    root=Path(root).resolve();datasets=Path(datasets).resolve();code=code_identity()
    if root.exists():raise ValueError('New version needs a new output directory')
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit source before preparing')
    prefix=read(ledger)
    if prefix['limits']!=LIMITS or billing_blocked(prefix):raise ValueError('Cumulative ledger is not ready')
    occupied=sum(x.get('accounted_tokens',x['reserved_tokens']) for x in prefix['attempts'].values())
    if occupied+sum(x['token_limit'] for x in PHASES.values())>LIMITS['token_limit']:
        raise ValueError('Frozen complete arm plan exceeds actual cumulative tokens')
    if len(prefix['attempts'])+sum(x['request_limit'] for x in PHASES.values())>LIMITS['request_limit']:
        raise ValueError('Frozen complete arm plan exceeds actual cumulative HTTP')
    train=task_rows(datasets/'train.json');test=task_rows(datasets/'test.json');val=task_rows(datasets/'val.json')
    if [len(train),len(val),len(test)]!=[60,17,100]:raise ValueError('Canonical counts changed')
    if len({t['physical_key'] for t in train+val+test})!=177:raise ValueError('Split overlap')
    picked=selection(train);episodes=schedule(picked,test)
    # Distribution is recorded from all historical episodes, never selected by score.
    h=read(Path(history_run)/'budget.json')['attempts'];hp=read(Path(history_run)/'campaign_budget_prefix.json')['attempts']
    costs={}
    for key,row in h.items():
        if key not in hp:costs.setdefault(row['context']['parent_task_id'],[]).append(row.get('accounted_tokens',row['reserved_tokens']))
    values=sorted(sum(rows) for rows in costs.values())
    distribution={'n':len(values),'mean':statistics.mean(values),'p95':values[int(.95*len(values))-1],
        'maximum':max(values),'total':sum(values),'projected_test_arm':100*statistics.mean(values),
        'test_cap_multiple_of_historical_mean':6300000/(100*statistics.mean(values)),
        'not_worst_case_guarantee':True,'per_task_parent_cap_unchanged':128000}
    root.mkdir(parents=True);shutil.copytree(PACKAGE,root/'bundle',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    banks=root/'banks';variants=build(banks,root/'bundle')
    cfg=read(base_config);cfg=cfg.get('test',cfg)
    cfg={'llm':deepcopy(cfg['llm']),'runtime':deepcopy(rt.RUNTIME),'program_environment':cfg['program_environment']}
    cfg['llm'].update(max_retries=0)
    if cfg['llm']['model']!='deepseek-v4-flash':raise ValueError('Only the authorized model')
    for name,data in [('config.json',cfg),('tasks.json',{'train12':picked,'test':test}),('schedule.json',episodes),
        ('budget/prefix.json',prefix),('budget/ledger.json',prefix),('budget_projection.json',distribution)]:save_json(root/name,data)
    files={str(Path(p).resolve()):sha256(Path(p)) for p in [base_config,ledger,
        Path(history_run)/'budget.json',Path(history_run)/'campaign_budget_prefix.json',
        *[datasets/n for n in ['train.json','val.json','test.json','evaluator_records_train.json','evaluator_records_test.json']]]}
    plan={'schema':VERSION,'created_at':utc(),'code':code,'source_integration':str((root/'bundle/integration/frontier.py').resolve()),
        'data_root':str(datasets),'seed':42,'model':'deepseek-v4-flash','runtime':rt.RUNTIME,
        'variants':variants,'development_arms':ARMS,'test_arms':TEST_ARMS,'phase_limits':PHASES,
        'historical_tokens_or_reserved':occupied,'historical_http':len(prefix['attempts']),
        'cumulative_limits':LIMITS,'bank_tree':tree_identity(banks),'bundle_tree':tree_identity(root/'bundle'),
        'files':files,'prepared_hashes':{n:sha256(root/n) for n in ['config.json','tasks.json','schedule.json','budget/prefix.json','budget_projection.json']},
        'expected_episodes':236,'matched_baseline_missing':False,'test_noskill_missing':True,
        'test_development_history_disclosed':True,'external_reference_not_learned_ours':True,
        'arm_selection':'fixed before first development request; never score dependent',
        'paid_authorization':'User: 参考文档修改人工bank并单独启动test; document section 0 explicitly authorizes Train12 x3 then frozen Test100',
        'normal_observation_required':{'development':{a:12 for a in ARMS},'test':{a:20 for a in TEST_ARMS}}}
    plan['run_id']='human_frontier_r2_'+digest(plan)[:24]
    save_json(root/'plan.json',plan);save_json(root/'plan_identity.json',{'hash':digest(plan)})
    save_json(root/'paid_authorization.json',{'approved':True,'plan_hash':digest(plan),'basis':plan['paid_authorization'],
        'phase_limits':PHASES,'cumulative_limits':LIMITS})
    progress(root,status='prepared')
    return check(root)


def checked(root):
    root=Path(root);p=read(root/'plan.json')
    if p['schema']!=VERSION or p['code']!=code_identity() or digest(p)!=read(root/'plan_identity.json')['hash']:
        raise ValueError('Source or plan identity changed; use a new explicit revision')
    for name,h in p['files'].items():
        if sha256(Path(name))!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in p['prepared_hashes'].items():
        if sha256(root/name)!=h:raise ValueError('Prepared input changed: '+name)
    if tree_identity(root/'banks')!=p['bank_tree'] or tree_identity(root/'bundle')!=p['bundle_tree']:
        raise ValueError('Frozen manual bank or integration changed')
    prefix=read(root/'budget/prefix.json');ledger=read(root/'budget/ledger.json')
    if ledger['limits']!=LIMITS or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('Inherited physical billing changed')
    auth=read(root/'paid_authorization.json')
    if not auth['approved'] or auth['plan_hash']!=digest(p) or auth['phase_limits']!=p['phase_limits']:
        raise ValueError('Paid authorization differs from plan')
    return p


def open_view(root,arm):
    variant='benchmark_prior' if arm=='NoSkill' else arm
    directory=Path(root)/'banks/variants'/variant
    bundle=read(directory/'bank.json');bank=Bank(directory,readonly=True)
    try:view=integration(Path(root)/'bundle').HumanBankView(bank,bundle,'off' if arm=='NoSkill' else 'manual','off')
    except BaseException:bank.close();raise
    return bank,view


def check(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');payloads=[]
    with tempfile.TemporaryDirectory() as tmp:
        for index,row in enumerate(read(root/'schedule.json')):
            bank,view=open_view(root,row['arm']);t=PublicTask(**row['task'])
            try:policy=view.policy(t)
            finally:bank.close()
            g=BudgetGovernor(Path(tmp)/str(index)/'ledger.json',token_limit=128000,request_limit=2,finish_reserve=0)
            wire=rt.provider(cfg['llm'])._build_payload(rt.messages(t,policy),[])
            ctx={'parent_scope':str(index),'parent_token_limit':128000,'episode_scope':str(index),'episode_request_limit':2}
            g.admit('offline-main',wire,ctx)
            # Worst main reservation, then a bounded tail fitting the real second admission.
            g.complete({'request_id':'offline-main','raw_usage':{'total_tokens':g.state['attempts']['offline-main']['reserved_tokens']},'outcome':'offline_fixture'})
            end=rt.provider(cfg['llm'],True)._build_payload(rt.messages(t,policy,{'text':'','truncated':True}),[])
            g.admit('offline-finish',end,{**ctx,'purpose':'finish_only'})
            assert wire['max_tokens']==65536 and wire['thinking']['type']=='enabled' and not wire.get('tools')
            assert end['max_tokens']==2048 and end['thinking']['type']=='disabled' and not end.get('tools')
            assert not policy or wire['messages'][0]['content'].count(policy)==1
            payloads.append({'episode':index,'main_input_bound':input_token_bound(wire),
                'finish_min_input_bound':input_token_bound(end),'budget_admitted':True})
    result={'status':'passed','code':plan['code'],'integration_module':plan['source_integration'],
        'episodes':len(payloads),'production_provider_payloads':2*len(payloads),'new_model_http':0,
        'actual_admissions':payloads,'bank_hash':plan['bank_tree']['sha256']}
    save_json(root/'preflight.json',result)
    return {k:v for k,v in result.items() if k!='actual_admissions'}


def governor(root,phase,plan,owner):
    folder=Path(root)/'budget'/phase
    if not (folder/'campaign_budget_prefix.json').exists():save_json(folder/'campaign_budget_prefix.json',read(Path(root)/'budget/prefix.json'))
    class PhaseGovernor(BatchGovernor):
        def owns(self,context):return context.get('human_phase')==phase
    return PhaseGovernor(owner,{'new_limits':plan['phase_limits'][phase]},folder)


def progress(root,**change):
    root=Path(root);p=read(root/'plan.json');path=root/'progress.json'
    previous=read(path) if path.exists() else {};stats={};last=None;next_row=None
    for row in read(root/'schedule.json'):
        key=row['phase']+'/'+row['arm'];s=stats.setdefault(key,{'scored':0,'normal':0,'correct':0,'empty':0,'tokens':0,'http':0,'finalizations':0,'meta':{'n':0,'correct':0},'non_meta':{'n':0,'correct':0}})
        trace=episode_dir(root,row)/'trace.json'
        if not trace.exists():
            if next_row is None:next_row={k:v for k,v in row.items() if k!='task'}|{'task_id':row['task']['task_id']}
            continue
        t=read(trace);s['scored']+=1;s['normal']+=int(t['valid']);s['correct']+=int(t['score']['hard']);s['empty']+=int(not t['valid'])
        s['tokens']+=t['tokens'];s['http']+=t['http'];s['finalizations']+=int(t['finish_state']!='not_needed')
        group=s['meta' if t['meta_option_labels'] else 'non_meta'];group['n']+=1;group['correct']+=int(t['score']['hard'])
        last={'arm':row['arm'],'phase':row['phase'],'task_id':row['task']['task_id'],'trace_hash':t['trace_hash']}
    for s in stats.values():s['accuracy']=s['correct']/s['scored'] if s['scored'] else None
    ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    observation=(all(stats['development/'+a]['normal']==12 for a in ARMS) and
                 all(stats['test_'+a+'/'+a]['normal']>=20 for a in p['test_arms']))
    value={**previous,**change,'updated_at':utc(),'by_arm':stats,'last_completed':last,'next_episode':next_row,
        'new_metered_tokens':sum(v.get('accounted_tokens',0) for v in new.values()),'new_http':len(new),
        'new_pending_http':sum(v['status']!='metered' for v in new.values()),
        'cumulative_tokens_or_reserved':sum(v.get('accounted_tokens',v['reserved_tokens']) for v in ledger['attempts'].values()),
        'historical_unknown_reserved':sum(v['reserved_tokens'] for v in prefix['attempts'].values() if v['status']!='metered'),
        'observation_requirement_met':observation,'program_calls':0,'workflow_calls':0,'test_developmental_retest':True}
    save_json(path,value)
    return value


def event(root,value):
    with (Path(root)/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'at':utc(),**value},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def execute(root,env_file=None,*,solve_fn=None):
    import fcntl
    root=Path(root);lock=(root/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=checked(root);cfg=read(root/'config.json')
    if read(root/'preflight.json')['status']!='passed':raise ValueError('Offline preflight required')
    if env_file:load_env(env_file)
    owner=BudgetGovernor(root/'budget/ledger.json',**LIMITS)
    if owner.billing_blocked:raise RuntimeError('Unresolved new request; no blind replay')
    adapters={s:AnswerAdapter('livemath',read(Path(plan['data_root'])/f'evaluator_records_{s}.json')) for s in ('train','test')}
    print('Actual integration module:',plan['source_integration'],flush=True)
    print('Effective Runtime:',json.dumps(rt.RUNTIME),flush=True)
    progress(root,status='running',pid=os.getpid(),error=None,error_type=None);event(root,{'event':'execute','pid':os.getpid()})
    try:
        for row in read(root/'schedule.json'):
            if row['phase'].startswith('test_') and not (root/'final_frozen_manifest.json').exists():
                state=progress(root)
                if any(state['by_arm']['development/'+a]['scored']!=12 for a in ARMS):raise RuntimeError('Development comparison incomplete')
                save_json(root/'development_result.json',state['by_arm'])
                save_json(root/'final_frozen_manifest.json',{'plan_hash':digest(plan),'freeze_time':utc(),
                    'bank_tree':tree_identity(root/'banks'),'arms':plan['test_arms'],
                    'strategy_selected_before_development':True,'selection_by_development_score':False})
            directory=episode_dir(root,row);already=(directory/'trace.json').exists()
            task=PublicTask(**row['task']);bank,view=open_view(root,row['arm'])
            progress(root,status='running',active={'phase':row['phase'],'arm':row['arm'],'task_id':task.task_id})
            try:
                result=(solve_fn or rt.solve)(task,adapters[task.split],view,cfg['llm'],directory,
                    governor(root,row['phase'],plan,owner),phase=row['phase'],arm=row['arm'])
            finally:bank.close()
            value=progress(root)
            if not already:
                event(root,{'event':'task_completed','phase':row['phase'],'arm':row['arm'],'task_id':task.task_id,
                    'valid':result['valid'],'correct':result['score']['hard'],'trace_hash':result['trace_hash']})
                print(json.dumps({'completed':row['episode_order_index']+1,'total':plan['expected_episodes'],
                    'phase':row['phase'],'arm':row['arm'],'task':task.task_id,'valid':result['valid'],
                    'correct':result['score']['hard'],'tokens':result['tokens'],'new_metered_tokens':value['new_metered_tokens']},ensure_ascii=False),flush=True)
        if tree_identity(root/'banks')!=plan['bank_tree']:raise ValueError('Frozen Bank mutated')
        save_json(root/'freeze_audit.json',{'before':plan['bank_tree'],'after':tree_identity(root/'banks')})
        value=progress(root,status='completed_test',active=None);save_json(root/'result.json',value)
        event(root,{'event':'completed_test'});return value
    except BaseException as exc:
        value=progress(root,status='blocked',error_type=type(exc).__name__,error=str(exc))
        event(root,{'event':'blocked','error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        for a in adapters.values():a.close()
        lock.close()


def monitor(root):
    root=Path(root);value=read(root/'progress.json')
    ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    value.update(new_metered_tokens=sum(x.get('accounted_tokens',0) for x in new.values()),
        new_http=len(new),new_pending_http=sum(x['status']!='metered' for x in new.values()),
        cumulative_tokens_or_reserved=sum(x.get('accounted_tokens',x['reserved_tokens']) for x in ledger['attempts'].values()))
    try:os.kill(value.get('pid',-1),0);alive=value.get('pid',-1)>0
    except (OSError,TypeError):alive=False
    return {**value,'process_alive':alive}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','check','execute','monitor'])
    ap.add_argument('--root',required=True);ap.add_argument('--base-config');ap.add_argument('--datasets')
    ap.add_argument('--ledger');ap.add_argument('--history-run');ap.add_argument('--env-file')
    a=ap.parse_args()
    if a.command=='prepare':result=prepare(a.root,a.base_config,a.datasets,a.ledger,a.history_run)
    elif a.command=='execute':result=execute(a.root,a.env_file)
    else:result=globals()[a.command](a.root)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
