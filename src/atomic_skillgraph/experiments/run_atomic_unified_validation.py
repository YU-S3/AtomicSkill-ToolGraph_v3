"""One locked 12-Train / 48-Val integration batch, sharing one durable Governor."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

import yaml

from ..core.errors import BudgetExhausted
from ..empirical.bank import Bank
from ..empirical.bank_view import BankView
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.registry import create_simple_harness
from .canonical_manifest import ordered_train, sha256, verify
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, write_json
from .run_formal import resolved_config, model_settings, corpus_identity
from skillcompiler_bench_contracts import source_identity

BENCHMARKS = ('searchqa','livemath','officeqa','spreadsheetbench')
ARMS = ('NoSkill','Guidance-only','Atomic-full')
LIMITS = {'token_limit':6000000,'finish_reserve':0,'request_limit':600,'validation_limit':24}
PARENT_LIMITS = {'train_task_tokens':200000,'train_solve_tokens':120000,'train_learning_tokens':80000,'eval_task_tokens':96000}


def read(path): return json.loads(Path(path).read_text())


def fixed_schedule():
    schedule=[]
    for i in range(3):
        for b in BENCHMARKS[i:]+BENCHMARKS[:i]:schedule.append({'benchmark':b,'split':'train','index':i,'arm':'Train'})
    for i in range(4):
        for j,b in enumerate(BENCHMARKS[i:]+BENCHMARKS[:i]):
            rotation=(i+j)%3
            for arm in ARMS[rotation:]+ARMS[:rotation]:schedule.append({'benchmark':b,'split':'val','index':i,'arm':arm})
    return schedule


def prepare(config_path, datasets, authority, corpus, output):
    output,datasets,authority=map(lambda p:Path(p).resolve(),(output,datasets,authority))
    if (output/'manifest.json').exists(): raise ValueError('This fixed batch is already prepared; no replacement or second pilot')
    code=code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Commit the tested source before preparing a paid batch')
    verify(authority)
    materialized=read(datasets/'materialization.json')
    if materialized['authority_sha256'] != sha256(authority/'manifest.json'): raise ValueError('Public authority mismatch')
    for name,expected in materialized['files_sha256'].items():
        if sha256(datasets/name)!=expected: raise ValueError('Materialized input changed: '+name)
    for name,expected in materialized.get('external_files_sha256',{}).items():
        if sha256(name)!=expected: raise ValueError('Public input file changed: '+name)
    base=yaml.safe_load(Path(config_path).read_text())
    lock=read('models.lock.json'); base=model_settings(base,lock['current_test'])
    base['budget']={**LIMITS,**PARENT_LIMITS}
    base['llm']['planner']['max_completion_tokens']=8192
    base['llm']['extractor']['max_completion_tokens']=8192
    base['llm']['tool_builder']['max_completion_tokens']=16384
    base['llm']['runtime']['max_completion_tokens']=32768
    base['llm'].setdefault('purpose_overrides',{})['finish_only']={'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}
    profiles=read('benchmark_profiles.json')['profiles']
    selected,configs,exclusions={},{},[]
    for benchmark in BENCHMARKS:
        selected[benchmark]={};configs[benchmark]={}
        for split,size in (('train',3),('val',4)):
            cfg=resolved_config(base,profiles[benchmark if benchmark!='spreadsheetbench' else 'spreadsheet'],
                benchmark,42,split,output/benchmark,datasets,authority,corpus)
            cfg=validate_config(cfg)
            rows=[PublicTask(**r) for r in read(datasets/benchmark/(split+'.json'))['tasks']]
            canonical=read(authority/benchmark/(split+'.json'))['tasks']
            if [t.task_id for t in rows]!=[t['task_id'] for t in canonical]:raise ValueError('Canonical membership/order changed')
            rows=ordered_train(rows,42) if split=='train' else sorted(rows,key=lambda t:hashlib.sha256(
                ('atomic-unified-20261010|'+benchmark+'|'+t.physical_key).encode()).hexdigest())
            adapter=create_simple_harness(cfg); chosen=[]
            try:
                for task in rows:
                    try:
                        adapter.reset(task)
                        # Real public files must be readable before any model request.
                        for path in getattr(getattr(adapter,'workspace',None),'inputs',{}).values():
                            if not path.is_file():raise FileNotFoundError(str(path))
                    except (FileNotFoundError,OSError) as exc:
                        exclusions.append({'benchmark':benchmark,'split':split,'task_id':task.task_id,'reason':str(exc)});continue
                    chosen.append(asdict(task))
                    if len(chosen)==size:break
            finally:adapter.close()
            if len(chosen)!=size:raise ValueError('Insufficient publicly available inputs')
            selected[benchmark][split]=chosen; configs[benchmark][split]=cfg
    manifest={'schema':'atomic-unified.fixed-validation.v1','created_at':utc(),'output':str(output),'code':code,
        'configs':configs,'tasks':selected,'schedule':fixed_schedule(),'exclusions_before_paid':exclusions,
        'limits':LIMITS,'parent_limits':PARENT_LIMITS,'seed':42,'source_identity':source_identity(),
        'authority_sha256':sha256(authority/'manifest.json'),'materialization_sha256':sha256(datasets/'materialization.json'),
        'corpus_sha256':corpus_identity(corpus)['sha256'],'external_files_sha256':materialized.get('external_files_sha256',{}),
        'model_coverage':{'configured':lock['current_test'],'other_models':'unconfigured_user_requested',
            'docvqa':'unsupported text model; image fixture only','alfworld':'deferred paid validation'},
        'stop_rule':'one batch; no Test, replacement, extra seed, whole-task trial continuation or budget increase'}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'manifest.json',manifest)
    return check(output/'manifest.json')


def check(path):
    m=read(path)
    if m['schema']!='atomic-unified.fixed-validation.v1' or m['limits']!=LIMITS or m['parent_limits']!=PARENT_LIMITS:
        raise ValueError('Fixed integration budget changed')
    if m['code']!=code_identity() or m['source_identity']!=source_identity():raise ValueError('Fixed source identity changed')
    if m['schedule']!=fixed_schedule():raise ValueError('Fixed episode count or balanced order changed')
    for b in BENCHMARKS:
        for split,n in (('train',3),('val',4)):
            if len(m['tasks'][b][split])!=n:raise ValueError('Fixed task selection changed')
            validate_config(m['configs'][b][split])
    for path,expected in m['external_files_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Public input changed')
    return {'passed':True,'main_episodes':60,'train_episodes':12,'val_physical_tasks':16,'val_episodes':48,'manifest_sha256':digest(m)}


def summary(m,governor,status,error=None):
    output=Path(m['output']); rows=[];assets={};frozen={}
    for row in m['schedule']:
        path=episode_path(output,row)/'trace.json'
        task=m['tasks'][row['benchmark']][row['split']][row['index']]
        item={**row,'task_id':task['task_id'],'physical_key':task['physical_key'],'status':'missing'}
        if path.exists():
            t=read(path);ex=t['execution']; usage=t.get('usage',[])
            executions=[*ex.get('attempts',[]),*ex.get('temporary_executions',[])]
            item.update(status='scored',score=t['score'],answer=ex.get('prediction'),reason=ex.get('reason'),
                budget_censored=bool(ex.get('budget_censored') or t.get('learning_status')=='deferred_budget'),
                tokens=sum(u['total_tokens'] for u in usage),program_calls=len(ex.get('attempts',[])),
                consumed=sum(bool(a.get('outputs_consumed')) for a in ex.get('attempts',[])),
                temporary_calls=len(ex.get('temporary_executions',[])),learning=t.get('learning'),
                prompt_tokens=sum(u['prompt_tokens'] for u in usage),completion_tokens=sum(u['completion_tokens'] for u in usage),
                learning_tokens=sum(u['total_tokens'] for u in usage if u['bucket'] in {'extractor_e1','tool_builder_evolution'}),
                http_attempts=sum(len(q.get('http_attempts',[])) for q in t.get('requests',[])),
                planner_http=sum(len(q.get('http_attempts',[])) for q in t.get('requests',[]) if q['stage']=='planner'),
                native_calls=t.get('native_call_attempts',ex.get('native_calls',0)),
                discarded=sum(not a.get('outputs_consumed') for a in executions),
                replaced=sum(bool(a.get('replaced')) for a in executions),
                worker_cpu_seconds=sum(a.get('worker_cpu_seconds') or 0 for a in executions)+sum(
                    e.get('result',{}).get('worker_cpu_seconds') or 0 for e in t.get('tools',[])),
                injected_guidance_ids=t.get('injected_guidance_ids',[]),
                trace_sha256=sha256(path),trace_path=str(path))
        else:item['missing_reason']=error or status
        rows.append(item)
    comparisons={}
    for reference in ARMS[:2]:
        pairs=[]
        for b in BENCHMARKS:
            for i in range(4):
                group=[r for r in rows if r['benchmark']==b and r['split']=='val' and r['index']==i]
                if len(group)==3 and all(r['status']=='scored' for r in group):
                    full=next(r for r in group if r['arm']=='Atomic-full');ref=next(r for r in group if r['arm']==reference)
                    pairs.append({'benchmark':b,'index':i,'full_correct':full['score']['hard'],'reference_correct':ref['score']['hard'],
                        'token_delta':full['tokens']-ref['tokens'],'budget_censored':any(r['budget_censored'] for r in group)})
        def aggregate(ps):
            return {'n':len(ps),'wins':sum(p['full_correct'] and not p['reference_correct'] for p in ps),
                'losses':sum(p['reference_correct'] and not p['full_correct'] for p in ps),
                'ties':sum(p['reference_correct']==p['full_correct'] for p in ps),
                'total_token_delta':sum(p['token_delta'] for p in ps),
                'median_token_delta':statistics.median([p['token_delta'] for p in ps]) if ps else None}
        comparisons['Full-'+reference]={'normal':aggregate([p for p in pairs if not p['budget_censored']]),
            'censored':aggregate([p for p in pairs if p['budget_censored']]),'pairs':pairs}
    for b in BENCHMARKS:
        path=output/b/'train/bank'
        if path.exists():
            bank=Bank(path,readonly=True)
            try:assets[b]={'generated':len(bank.all('program')),'usable':sum(bank.program_eligible(p) for p in bank.all('program')),
                'skills':len(bank.all('skill')),'jobs':bank.jobs(),'programs':bank.all('program')}
            finally:bank.close()
        path=output/b/'train/frozen_bank'
        if path.exists():
            frozen[b]={'before':read(output/b/'freeze_identity.json'),'after':tree_identity(path)}
            if frozen[b]['before']!=frozen[b]['after']:raise RuntimeError('Frozen Bank changed during evaluation')
    attempts=governor.state['attempts']
    roles={}
    for attempt in attempts.values():
        stage=attempt['context'].get('stage','unknown');r=roles.setdefault(stage,{'http_attempts':0,'total_tokens':0,
            'prompt_tokens':0,'completion_tokens':0,'raw_usage':[]})
        r['http_attempts']+=1;r['total_tokens']+=attempt.get('accounted_tokens',0)
        u=attempt.get('actual_usage',{});r['raw_usage'].append(u)
        r['prompt_tokens']+=u.get('prompt_tokens',0);r['completion_tokens']+=u.get('completion_tokens',0)
    return {'status':status,'error':error,'manifest_sha256':digest(m),'source':m['code'],'rows':rows,'assets':assets,
        'comparisons':comparisons,'frozen_identity':frozen,'actual_http_attempts':len(attempts),
        'actual_tokens':sum(a.get('accounted_tokens',0) for a in attempts.values()),
        'unknown_billing':governor.state['unknown_billing'],'local_validation_executions':len(governor.state.get('worker_validations',[])),
        'role_usage':roles,'known_usd_cost':None,'usd_cost_reason':'No frozen price identity for this batch',
        'missing_episodes':sum(r['status']=='missing' for r in rows),
        'model_coverage':m['model_coverage'],'formal_resumed':False,'finished_at':utc()}


def episode_path(output,row):
    return output/row['benchmark']/row['split']/row['arm']/str(row['index'])


def run(path,env_file):
    check(path);m=read(path);output=Path(m['output'])
    binding=output/'batch_binding.json'
    if binding.exists() and read(binding)['manifest_sha256']!=digest(m):raise ValueError('Paid batch binding changed')
    if (output/'result.json').exists():return read(output/'result.json')
    write_json(binding,{'manifest_sha256':digest(m),'started_at':utc()}) if not binding.exists() else None
    load_env(env_file)
    governor=BudgetGovernor(output/'budget.json',**LIMITS)
    logs={b:FormalLog(output/b/'audit',{'method':'ours','benchmark':b,'source_commit':m['code']['git_sha'],
        'split_manifest_hash':m['authority_sha256'],'public_materialization_sha256':m['materialization_sha256'],
        'batch_manifest_sha256':digest(m)},m['configs'][b],resume=(output/b/'audit/run_manifest.json').exists()) for b in BENCHMARKS}
    status,error='completed',None
    try:
        for row in m['schedule']:
            phase=episode_path(output,row)
            if (phase/'trace.json').exists():continue
            if governor.state['unknown_billing'] or len(governor.state['attempts'])>=600:
                raise BudgetExhausted('batch_limit','Whole batch admission closed')
            cfg=deepcopy(m['configs'][row['benchmark']][row['split']]);cfg['experiment']['output_dir']=str(phase)
            frozen=output/row['benchmark']/'train/frozen_bank'
            if row['split']=='val' and tree_identity(frozen)!=read(output/row['benchmark']/'freeze_identity.json'):
                raise RuntimeError('Frozen identity changed before Val')
            factory=lambda cfg=cfg:create_simple_harness(cfg)
            mode={'NoSkill':'learned_assets_off','Guidance-only':'guidance_only'}.get(row['arm'])
            system=EmpiricalSystem(cfg,harness=factory(),adapter_factory=factory,readonly=row['split']=='val',
                bank_view_factory=(lambda bank,mode=mode:BankView(bank,mode)) if mode else None,budget_governor=governor)
            task=PublicTask(**m['tasks'][row['benchmark']][row['split']][row['index']])
            phase.mkdir(parents=True,exist_ok=True)
            write_json(phase/'config.json',system.config)
            write_json(phase/'execution_manifest.json',{'code':m['code'],'task':asdict(task),'arm':row['arm'],
                'parent_manifest_sha256':digest(m),'capabilities':asdict(system.adapter.capabilities)})
            system.checkpoint=TaskCheckpoint(phase/'checkpoint');system.audit_path=phase/'requests.json'
            log=logs[row['benchmark']];system.observer=log;system.bank.observer=log
            log.begin_task(task,row['index'],task.task_id+':'+row['arm'],{'arm':row['arm']})
            try:
                trace=system.run_task(task,learn=row['split']=='train',attempt_id=task.task_id+':'+row['arm'])
                log.end_task(trace)
                write_json(phase/'trace.json',trace)
                if row['split']=='train' and row['index']==2:
                    system.bank.freeze(frozen)
                    write_json(output/row['benchmark']/'freeze_identity.json',tree_identity(frozen))
            except BudgetExhausted:
                # Preserve a completed score if only downstream learning was censored.
                trace=system.checkpoint.state.get('trace')
                if trace and trace.get('solve_status')=='completed':
                    trace['execution']['budget_censored']=True
                    if (phase/'requests.json').exists():trace.update(read(phase/'requests.json'))
                    write_json(phase/'trace.json',trace)
                elif (phase/'requests.json').exists() and any(q.get('http_attempts') for q in read(phase/'requests.json')['requests']):
                    execution=deepcopy(getattr(system.executor,'partial_execution',{}))
                    execution.setdefault('prediction',None);execution.setdefault('attempts',[])
                    execution.update(reason='whole_batch_budget_exhausted',budget_censored=True)
                    trace={'task':asdict(task),'execution':execution,'score':system.adapter.evaluate(system.adapter.submit(execution['prediction'])),
                        **read(phase/'requests.json'),'learning_status':'deferred_budget'}
                    write_json(phase/'trace.json',trace)
                raise
            finally:system.close()
            write_json(output/'progress.json',{'completed':sum((episode_path(output,r)/'trace.json').exists() for r in m['schedule']),
                'last':row,'tokens':sum(a.get('accounted_tokens',0) for a in governor.state['attempts'].values()),'at':utc()})
            print(json.dumps({'event':'episode_completed',**row,'score':trace['score'],'tokens':sum(u['total_tokens'] for u in trace.get('usage',[]))}),flush=True)
    except BudgetExhausted as exc:status,error='budget_stopped',{'code':exc.code,'message':str(exc)}
    except Exception as exc:
        status,error='engineering_stopped',{'type':type(exc).__name__,'message':str(exc)}
        import traceback
        (output/'engineering_error.txt').write_text(traceback.format_exc())
    result=summary(m,governor,status,error);write_json(output/'result.json',result)
    for log in logs.values():log.finish(status)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--config',default='configs/default.yaml')
    for name in ('datasets','authority','corpus','output'):p.add_argument('--'+name,required=True)
    for command in ('check-offline','run'):
        p=sub.add_parser(command);p.add_argument('--manifest',required=True)
        if command=='run':p.add_argument('--env-file',required=True)
    a=parser.parse_args()
    result=prepare(a.config,a.datasets,a.authority,a.corpus,a.output) if a.command=='prepare' else check(a.manifest) if a.command=='check-offline' else run(a.manifest,a.env_file)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','assets','comparisons')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
