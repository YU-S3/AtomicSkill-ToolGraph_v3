"""Train-record derivation -> independent Frozen -> Val review. Never auto-start Test."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from ..core.errors import BudgetExhausted
from ..empirical import OFFICE_SOURCE_METHOD_VERSION as VERSION
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest
from ..empirical.source_catalog import verified_sources
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.registry import create_simple_harness
from .benchmark_config import benchmark_config
from .canonical_manifest import ordered_train, sha256, verify
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, run, write_json

PHASE_LIMITS={'recompile':3000000,'val':4608000,'test':19584000}
TOTAL_LIMITS={'token_limit':85000000,'request_limit':480000,'finish_reserve':0}


def read(path):return json.loads(Path(path).read_text())


class PhaseGovernor(BudgetGovernor):
    """One physical ledger, plus immutable phase caps; unused phases cannot subsidize others."""
    def __init__(self,path,phase):
        if phase not in PHASE_LIMITS:raise ValueError('Unknown Office phase')
        self.phase=phase
        super().__init__(path,**TOTAL_LIMITS)

    def headroom(self,context):
        room=super().headroom(context)
        charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in self.state['attempts'].values()
                    if a['context'].get('office_phase')==self.phase)
        room['tokens']=min(room['tokens'],PHASE_LIMITS[self.phase]-charged-context.get('parent_finish_reserve',0))
        return room

    def admit(self,audit_id,payload,context):
        with self.lock:
            context={**context,'office_phase':self.phase,'office_phase_token_limit':PHASE_LIMITS[self.phase]}
            reserved=input_token_bound(payload)+payload.get('max_tokens',payload.get('max_completion_tokens',0))
            if reserved>self.headroom(context)['tokens']:
                raise BudgetExhausted('diagnostic_budget_exhausted','Office phase allowance exhausted; no phase borrowing')
            super().admit(audit_id,payload,context)


def configuration(old,root,split):
    config=deepcopy(old)
    config['experiment'].update(officeqa_method_version=VERSION,output_dir=str(root/split),
        runtime_mode='online' if split=='recompile' else 'frozen')
    config=benchmark_config(config,'officeqa')
    config['data_dir']=str(root/'train'/('bank' if split=='recompile' else 'frozen_bank'))
    config['runtime']['global_action_budget']=64 if split=='recompile' else 48
    if split!='recompile':
        config['manifest']=str(Path(config['manifest']).with_name(split+'.json'))
        config['harness']['evaluator_records']=str(Path(config['harness']['evaluator_records']).with_name('evaluator_records_'+split+'.json'))
    return validate_config(config)


def prepare(source,output):
    source,root=Path(source).resolve(),Path(output).resolve()
    if root.exists() or source==root or source in root.parents:raise ValueError('Use an independent new output directory')
    code=code_identity()
    if not code['git_sha'] or code['tracked_dirty']:raise ValueError('Recompile requires committed clean source')
    old=read(source/'train/config.json');execution=read(source/'train/execution_manifest.json')
    if old['experiment'].get('benchmark')!='officeqa' or old['experiment']['seed']!=42:raise ValueError('Expected Office seed42 Train source')
    if (old['llm']['runtime'].get('max_completion_tokens')!=32768 or old['llm']['runtime'].get('reasoning_effort')!='high'
            or old['runtime'].get('finish_input_max_bytes')!=16384 or old['runtime'].get('dynamic_output_min_tokens')!=8192
            or old['llm']['purpose_overrides']['finish_only']!={'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}):
        raise ValueError('Source Runtime/finish settings differ from the frozen Office contract')
    if (source/'test/execution_manifest.json').exists():raise ValueError('Source must precede Test')
    ledger=read(source/'budget.json')
    if ledger['limits']!=TOTAL_LIMITS or ledger['unknown_billing'] or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('Only the frozen, fully metered 85M ledger is admissible')
    inherited=sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    if inherited+sum(PHASE_LIMITS.values())>TOTAL_LIMITS['token_limit']:raise ValueError('Phase allocation exceeds original allowance')
    authority=Path(old['manifest']).parent.parent
    verify(authority)
    canonical=read(old['manifest'])['tasks']
    tasks=[PublicTask(**t) for t in execution['tasks']]
    if len(tasks)!=120 or any(t.split!='train' for t in tasks) or {t.task_id for t in tasks}!={t['task_id'] for t in canonical}:
        raise ValueError('Source must contain exactly canonical Train120')
    if tasks!=ordered_train([next(t for t in tasks if t.task_id==r['task_id']) for r in canonical],42):
        raise ValueError('Source Train order differs from seed42')
    source_db=sqlite3.connect((source/'train/bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    cases={r['task']['physical_key']:r for r in (json.loads(x[0]) for x in source_db.execute('SELECT payload FROM train_cases'))}
    if set(cases)!={t.physical_key for t in tasks} or any(r['task']['split']!='train' for r in cases.values()):
        source_db.close();raise ValueError('Train Bank has missing or non-Train source cases')
    records=[]
    for index,task in enumerate(tasks):
        path=source/'train/traces'/(task.task_id+'.json');trace=read(path)
        if trace['task']!=asdict(task) or 'score' not in trace:raise ValueError('Incomplete or mismatched source task')
        case=cases[task.physical_key]['experience']
        available=bool(verified_sources([(task,case)]))
        records.append({'task':asdict(task),'path':str(path),'sha256':sha256(path),
            'source_record_hash':digest(trace),'canonical_run_order_index':index,
            'successful_python_source':available,'local_evidence_count':len(case.get('local_evidence',[]))})
    # Source availability, then the already frozen seed42 order; no score or held-out outcome sorting.
    records.sort(key=lambda r:(not r['successful_python_source'],not r['local_evidence_count'],r['canonical_run_order_index']))
    root.mkdir(parents=True)
    (root/'train/bank').mkdir(parents=True)
    target=sqlite3.connect(root/'train/bank/bank.sqlite3');source_db.backup(target);source_db.close()
    archived=[json.loads(r[0]) for r in target.execute('SELECT payload FROM realization_jobs')]
    with target:
        target.execute('DELETE FROM realization_jobs')
        target.execute('INSERT INTO metadata VALUES(?,?)',('officeqa_method_version',VERSION))
        target.execute('INSERT INTO metadata VALUES(?,?)',('office_train_derivation',str(source)))
    target.close()
    write_json(root/'inherited_jobs.json',archived)
    inputs={str(source/p):sha256(source/p) for p in ('budget.json','cost_carryforward_manifest.json',
        'train/config.json','train/execution_manifest.json','train/bank/bank.sqlite3')}
    configs={p:configuration(old,root,p) for p in PHASE_LIMITS}
    write_json(root/'resolved_config.json',configs)
    write_json(root/'budget.json',ledger)
    write_json(root/'inherited_budget.json',ledger)
    write_json(root/'cost_carryforward_manifest.json',{'parent':str(source),'parent_inputs':inputs,
        'inherited_tokens':inherited,'inherited_http':len(ledger['attempts']),
        'physical_requests':'exact keyed entries copied once; no synthetic usage',
        'phase_limits':PHASE_LIMITS,'worst_case_tokens':inherited+sum(PHASE_LIMITS.values()),
        'unallocated_tokens':TOTAL_LIMITS['token_limit']-inherited-sum(PHASE_LIMITS.values())})
    plan={'version':VERSION,'code':code,'created_at':utc(),'source':str(source),'source_code':execution['code'],
        'parent_inputs':inputs,'source_records':records,'config_hash':digest(configs),'phase_limits':PHASE_LIMITS,
        'initial_bank':tree_identity(root/'train/bank'),'inherited_jobs_sha256':sha256(root/'inherited_jobs.json'),
        'selection_rule':'successful Host Python, other local evidence, no evidence; original seed42 order within groups',
        'stage_order':['recompile','freeze','val','review_stop','test_after_explicit_approval']}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':sha256(root/'plan.json')})
    return plan


def checked(root):
    root=Path(root).resolve();plan=read(root/'plan.json');configs=read(root/'resolved_config.json')
    if sha256(root/'plan.json')!=read(root/'plan_identity.json')['sha256'] or plan['code']!=code_identity():
        raise ValueError('Prepared source identity changed; no cross-version resume')
    if digest(configs)!=plan['config_hash'] or plan['phase_limits']!=PHASE_LIMITS or plan['version']!=VERSION:
        raise ValueError('Prepared method/config/budget changed')
    for path,expected in plan['parent_inputs'].items():
        if sha256(path)!=expected:raise ValueError('Paused parent changed: '+path)
    for row in plan['source_records']:
        if sha256(row['path'])!=row['sha256']:raise ValueError('Train source changed: '+row['path'])
    return root,plan,configs


def recompile(output,*,max_new_records=None):
    if max_new_records is not None and max_new_records<1:raise ValueError('Record boundary must be positive')
    root,plan,configs=checked(output)
    if (root/'final_frozen_manifest.json').exists():
        if tree_identity(root/'train/frozen_bank')['sha256']!=read(root/'final_frozen_manifest.json')['final_artifact_hash']:
            raise ValueError('Final Frozen changed')
        return read(root/'recompile/summary.json')
    phase=root/'recompile';settings=configs['recompile']
    governor=PhaseGovernor(root/'budget.json','recompile')
    identity={'method':VERSION,'source_commit':plan['code']['git_sha'],'source_hash':digest(plan),
              'benchmark':'officeqa','run_seed':42,'phase':'train_recompile'}
    log=FormalLog(phase,identity,settings,resume=(phase/'run_manifest.json').exists())
    s=EmpiricalSystem(settings,budget_governor=governor)
    s.observer=log;s.bank.observer=log
    results=[];new=0;censored=False
    try:
        for index,row in enumerate(plan['source_records']):
            task=PublicTask(**row['task']);folder=phase/'units'/task.task_id
            if (root/'STOP_AFTER_TASK').exists():raise SystemExit(75)
            if (folder/'result.json').exists():results.append(read(folder/'result.json'));continue
            if max_new_records is not None and new>=max_new_records:break
            if not row['local_evidence_count']:
                result={'task_id':task.task_id,'checked':True,'decision':'no_change','reason':'no_local_operation_source','http':0}
            else:
                s.checkpoint=TaskCheckpoint(folder/'checkpoint');s.audit_path=folder/'requests.json'
                log.begin_task(task,index,task.task_id+':recompile',{'source_record_sha256':row['sha256']})
                source_id={'source_run_id':str(plan['source']),'source_record_hash':row['source_record_hash'],
                           'source_trace_sha256':row['sha256'],'recompile_method':VERSION}
                try:trace=s.recompile_tool_record(task,read(row['path']),source_id)
                except BudgetExhausted:
                    censored=True;break
                write_json(folder/'trace.json',trace)
                log.training(task.task_id+':recompile:source','completed_record_recompile',
                    log.current_task['task_start_time'],utc(),0,'completed',
                    source_record_hash=row['source_record_hash'],source_score_inherited=True)
                result={'task_id':task.task_id,'checked':True,'source_record_hash':row['source_record_hash'],
                    'learning_status':trace['learning_status'],**trace.get('learning',{}),
                    'http':sum(len(r.get('http_attempts',[])) for r in trace['requests']),
                    'builder_http':sum(len(r.get('http_attempts',[])) for r in trace['requests'] if r['stage']=='tool_builder'),
                    'tokens':sum(u['total_tokens'] for u in trace['usage'])}
            write_json(folder/'result.json',result);results.append(result);new+=1
            print(json.dumps({'unit':index+1,'task':task.task_id,'decision':result.get('decision'),
                'learning_status':result.get('learning_status'),'tokens':result.get('tokens',0),
                'builder_http':result.get('builder_http',0),'program':result.get('program')},ensure_ascii=False),flush=True)
        complete=len(results)==len(plan['source_records']) or censored
        stats=Counter(r.get('decision',r.get('learning_status','unknown')) for r in results)
        summary={'complete':complete,'censored':censored,'checked':len(results),'total_sources':120,
            'decisions':dict(stats),'builder_http':sum(r.get('builder_http',0) for r in results),
            'capacity_rejected':sum(r.get('learning_status')=='learning_capacity_rejected' for r in results),
            'programs':[{'id':p['id'],'state':p['state']} for p in s.bank.all('program')],
            'usable':sum(s.bank.program_eligible(p) for p in s.bank.all('program')),
            'source_trials':[a for p in s.bank.all('program') for a in s.bank.attempts(p['id'])],
            'phase_tokens':sum(a['accounted_tokens'] for a in governor.state['attempts'].values()
                if a['context'].get('office_phase')=='recompile' and a['status']=='metered')}
        write_json(phase/'summary.json',summary)
        if complete:
            frozen=root/'train/frozen_bank'
            manifest=s.bank.freeze(frozen) if not frozen.exists() else read(frozen/'freeze.json')
            if manifest['source_digest']!=s.bank.digest():raise ValueError('Existing Frozen differs from recompile Bank')
            frozen_record={'freeze_time':utc(),'freeze':manifest,'final_artifact_hash':tree_identity(frozen)['sha256']}
            if (root/'final_frozen_manifest.json').exists():
                if read(root/'final_frozen_manifest.json')['final_artifact_hash']!=frozen_record['final_artifact_hash']:
                    raise ValueError('Previously frozen identity changed')
            else:write_json(root/'final_frozen_manifest.json',frozen_record)
        log.finish('completed' if complete else 'paused')
        return summary
    except BaseException as exc:
        if isinstance(exc,Exception):log.task_error(exc)
        log.finish('interrupted')
        raise
    finally:s.close()


def evaluate(output,split='val',*,max_new_tasks=None,review=None):
    root,plan,configs=checked(output)
    if split not in {'val','test'}:raise ValueError('Unknown evaluation split')
    if split=='test':
        if not review:raise ValueError('Test requires a separately approved Val review file')
        approval=read(review)
        if approval.get('approved') is not True or approval.get('val_summary_sha256')!=sha256(root/'val/summary.json'):
            raise ValueError('Review must approve this exact completed Val result')
        if not read(root/'val/summary.json')['complete']:raise ValueError('Val24 is not complete')
    if not read(root/'recompile/summary.json')['complete']:raise ValueError('Recompile has not reached a completed or censored boundary')
    settings=configs[split];phase=root/split
    if settings['budget']['eval_task_tokens']!=192000 or settings['runtime']['global_action_budget']!=48:
        raise ValueError('Office evaluation contract changed')
    frozen=root/'train/frozen_bank';expected=read(root/'final_frozen_manifest.json')['final_artifact_hash']
    if tree_identity(frozen)['sha256']!=expected:raise ValueError('Frozen changed before evaluation')
    datasets=Path(settings['harness']['evaluator_records']).parent
    tasks=[PublicTask(**row) for row in read(datasets/(split+'.json'))['tasks']]
    public=read(settings['manifest'])['tasks']
    if [t.task_id for t in tasks]!=[r['task_id'] for r in public] or len(tasks)!=(24 if split=='val' else 102):
        raise ValueError('Evaluation must use the complete canonical split in manifest order')
    log=FormalLog(phase,{'method':VERSION,'benchmark':'officeqa','run_seed':42,'phase':split,
        'source_commit':plan['code']['git_sha'],'recompile_plan_sha256':digest(plan),'frozen_hash':expected},settings,
        resume=(phase/'run_manifest.json').exists())
    write_json(phase/'frozen_check.json',{'before':expected,'after':None})
    try:
        result=run(settings,tasks,phase,resume=(phase/'execution_manifest.json').exists(),readonly=True,
            formal_log=log,task_metadata={r['task_id']:r['task_metadata'] for r in public},
            budget_governor=PhaseGovernor(root/'budget.json',split),max_new_tasks=max_new_tasks)
        log.finish('completed_review_required' if result['complete'] and split=='val' else 'completed' if result['complete'] else 'paused')
        return {k:v for k,v in result.items() if k!='cases'}
    finally:
        actual=tree_identity(frozen)['sha256']
        write_json(phase/'frozen_check.json',{'before':expected,'after':actual,'unchanged':actual==expected})
        if actual!=expected:raise RuntimeError('Frozen changed during evaluation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','recompile','val','test'])
    parser.add_argument('--output',required=True);parser.add_argument('--source');parser.add_argument('--env-file')
    parser.add_argument('--max-new-records',type=int);parser.add_argument('--max-new-tasks',type=int);parser.add_argument('--review')
    args=parser.parse_args()
    if args.env_file:load_env(args.env_file)
    if args.action=='prepare':result=prepare(args.source,args.output)
    elif args.action=='recompile':result=recompile(args.output,max_new_records=args.max_new_records)
    else:result=evaluate(args.output,args.action,max_new_tasks=args.max_new_tasks,review=args.review)
    print(json.dumps({k:v for k,v in result.items() if k not in {'source_records','source_trials'}},ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
