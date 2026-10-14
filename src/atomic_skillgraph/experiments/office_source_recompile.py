"""Train-record derivation -> independent Frozen -> Val review. Never auto-start Test."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sqlite3

from ..core.errors import BudgetExhausted
from ..empirical import OFFICE_SOURCE_METHOD_VERSION as VERSION
from ..empirical import OFFICE_TRACE_METHOD_VERSION as TRACE_VERSION
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
        plan_path=Path(path).parent/'plan.json'
        self.trace=plan_path.exists() and read(plan_path)['version']==TRACE_VERSION
        limits=TOTAL_LIMITS
        if self.trace:
            revision=read(Path(path).parent/'budget_revision.json')
            if revision.get('approved') is not True or not revision.get('authorization'):raise ValueError('V3 budget revision is pending explicit authorization')
            if digest(revision)!=read(plan_path)['budget_revision_hash']:raise ValueError('V3 budget revision changed')
            limits=revision['total_limits']
        super().__init__(path,**limits)

    def owns(self,attempt):
        return attempt['context'].get('office_phase')==self.phase and (not self.trace or attempt['context'].get('office_method')==TRACE_VERSION)

    def headroom(self,context):
        room=super().headroom(context)
        charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in self.state['attempts'].values()
                    if self.owns(a))
        room['tokens']=min(room['tokens'],PHASE_LIMITS[self.phase]-charged-context.get('parent_finish_reserve',0))
        if self.trace and self.phase=='recompile':room['requests']=min(room['requests'],400-sum(self.owns(a) for a in self.state['attempts'].values()))
        return room

    def admit(self,audit_id,payload,context):
        with self.lock:
            context={**context,'office_phase':self.phase,'office_phase_token_limit':PHASE_LIMITS[self.phase]}
            if self.trace:context['office_method']=TRACE_VERSION
            reserved=input_token_bound(payload)+payload.get('max_tokens',payload.get('max_completion_tokens',0))
            room=self.headroom(context)
            if reserved>room['tokens'] or room['requests']<=context.get('finish_request_reserve',0):
                raise BudgetExhausted('diagnostic_budget_exhausted','Office phase allowance exhausted; no phase borrowing')
            super().admit(audit_id,payload,context)


def configuration(old,root,split,*,version=VERSION):
    config=deepcopy(old)
    config['experiment'].update(officeqa_method_version=version,output_dir=str(root/split),
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
    if digest(configs)!=plan['config_hash'] or plan['phase_limits']!=PHASE_LIMITS or plan['version'] not in {VERSION,TRACE_VERSION}:
        raise ValueError('Prepared method/config/budget changed')
    for path,expected in plan['parent_inputs'].items():
        if sha256(path)!=expected:raise ValueError('Paused parent changed: '+path)
    for row in plan['source_records']:
        if sha256(row['path'])!=row['sha256']:raise ValueError('Train source changed: '+row['path'])
    if plan['version']==TRACE_VERSION:
        if plan['method_contract_hash']!=trace_contract_hash():raise ValueError('V3 method contract changed')
        if digest(read(root/'budget_revision.json'))!=plan['budget_revision_hash']:raise ValueError('Budget revision changed')
        if corpus_identity(configs['recompile']['harness']['corpus_root'])!=plan['corpus_identity']:raise ValueError('Authorized corpus changed')
    return root,plan,configs


def corpus_identity(path):
    from skillcompiler_bench_contracts.office import select_paths
    return digest([(name,sha256(file)) for name,file in select_paths(path)])


def trace_contract_hash():
    from ..empirical import OFFICE_TRACE_POLICY
    from ..empirical.retrieval_chain import KIND
    from skillcompiler_bench_contracts.office import RETRIEVAL_VERSION,GREP_PAGE_SCHEMA,CONTEXT_SCHEMA
    return digest([TRACE_VERSION,OFFICE_TRACE_POLICY,KIND,RETRIEVAL_VERSION,GREP_PAGE_SCHEMA,CONTEXT_SCHEMA,
        {'evaluation_tokens':192000,'evaluation_calls':48,'runtime_output':32768,'dynamic_minimum':8192,
         'finish_wire_bytes':16384,'finish_output':2048,'workflow':'empirical.workflow.v2'}])


def prepare_trace(parent,output,budget_revision=None):
    """Explicit V2 -> V3 Train-only derivation, including a completed read-only V2 Val."""
    from ..empirical.retrieval_chain import recover, authentic
    parent,root=Path(parent).resolve(),Path(output).resolve()
    code=code_identity();old=read(parent/'plan.json');configs=read(parent/'resolved_config.json')
    if root.exists() or parent in root.parents or code['tracked_dirty'] or not code['git_sha']:
        raise ValueError('V3 requires a fresh independent output and committed source')
    if (sha256(parent/'plan.json')!=read(parent/'plan_identity.json')['sha256'] or digest(configs)!=old['config_hash']
            or old['version']!=VERSION):raise ValueError('Parent method identity changed')
    if (parent/'test/execution_manifest.json').exists():raise ValueError('Parent must precede Test')
    if (parent/'val/execution_manifest.json').exists() and (not (parent/'val/summary.json').exists()
            or not read(parent/'val/summary.json')['complete']):raise ValueError('Parent Val must be completed before V3 derivation')
    for path,expected in old['parent_inputs'].items():
        if sha256(path)!=expected:raise ValueError('Parent source changed: '+path)
    summary=read(parent/'recompile/summary.json')
    if not summary['complete'] or summary['checked']!=120:raise ValueError('V3 requires completed canonical Train120 derivation')
    if tree_identity(parent/'train/frozen_bank')['sha256']!=read(parent/'final_frozen_manifest.json')['final_artifact_hash']:
        raise ValueError('Parent Frozen changed')
    ledger=read(parent/'budget.json')
    if ledger['limits']!=TOTAL_LIMITS or ledger['unknown_billing'] or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('V3 requires the original fully metered 85M ledger')
    charged=sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    revision=read(budget_revision) if budget_revision else {'approved':False,'authorization':None,
        'total_limits':TOTAL_LIMITS,'phase_token_limits':PHASE_LIMITS,'recompile_http_limit':400}
    if revision.get('phase_token_limits')!=PHASE_LIMITS or revision.get('recompile_http_limit')!=400:
        raise ValueError('Phase allocation differs from V3 contract')
    if revision['total_limits'] not in [TOTAL_LIMITS,{**TOTAL_LIMITS,'token_limit':90000000}]:raise ValueError('Unsupported cumulative revision')
    if revision['total_limits']!=TOTAL_LIMITS and (revision.get('approved') is not True or not revision.get('authorization')):
        raise ValueError('90M is a proposal, not authorization')
    if revision.get('approved') and (not revision.get('authorization') or charged+sum(PHASE_LIMITS.values())>revision['total_limits']['token_limit']):
        raise ValueError('Approved allocation cannot fund all reserved phases')
    authority=Path(configs['recompile']['manifest']).parent.parent;verify(authority)
    canonical=read(configs['recompile']['manifest'])['tasks']
    original=sorted(old['source_records'],key=lambda r:r['canonical_run_order_index'])
    tasks=[PublicTask(**r['task']) for r in original]
    if len(tasks)!=120 or {t.task_id for t in tasks}!={r['task_id'] for r in canonical} or any(t.split!='train' for t in tasks):
        raise ValueError('V3 source must be exactly canonical Train120')
    if tasks!=ordered_train([next(t for t in tasks if t.task_id==r['task_id']) for r in canonical],42):raise ValueError('Source order changed')
    source_db=sqlite3.connect((parent/'train/bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    cases={r['task']['physical_key']:r for r in (json.loads(x[0]) for x in source_db.execute('SELECT payload FROM train_cases'))}
    if set(cases)!={t.physical_key for t in tasks}:source_db.close();raise ValueError('Noncanonical Bank cases')
    jobs={j['id']:j for j in (json.loads(r[0]) for r in source_db.execute('SELECT payload FROM realization_jobs'))}
    records=[];proofs={};reopened=[]
    for row,task in zip(original,tasks):
        trace=read(row['path'])
        if sha256(row['path'])!=row['sha256'] or trace['task']!=asdict(task):raise ValueError('Original Train trace identity changed')
        case=cases[task.physical_key]['experience']
        chains=recover(task,trace,configs['recompile']['program_environment'])
        if any(not authentic(e,case) for e in chains):raise ValueError('Recovered chain differs from original Bank events')
        case['local_evidence']+=chains
        fresh={k:deepcopy(v) for k,v in row.items() if k not in {'inherited_result','builder_recovery','capacity_recovery'}}
        fresh.update(work_type='retrieval_chain',work_id=task.task_id,local_evidence_count=len(chains))
        records.append(fresh)
        unit=parent/'recompile/units'/task.task_id;result=read(unit/'result.json')
        if result.get('learning_status')!='learning_capacity_rejected':continue
        audit=read(unit/'requests.json');refused=audit['requests'][-1]
        if refused['stage']!='tool_builder':continue  # New sources own separate new decisions; no old paid Extractor rerun.
        if not refused.get('material_refusal') or refused.get('http_attempts'):raise ValueError('Recovery must be a zero-HTTP refusal')
        job_id,epoch,generation=refused['job_key'];job=jobs[job_id]
        if generation!=1 or job['generation_count']!=2 or job['state']!='deferred' or job.get('last_error_kind')!='learning_capacity_rejected':
            raise ValueError('Only the proven unissued necessary repair may reopen')
        proof={'request_id':refused['id'],'request_file':str(unit/'requests.json'),'sha256':sha256(unit/'requests.json'),
            'job_id':job_id,'job_hash':digest(job),'epoch':epoch,'generation':generation}
        proofs[proof['request_file']]=proof['sha256']
        reopened.append({**fresh,'work_type':'unissued_builder_repair','work_id':task.task_id+':repair',
            'local_evidence_count':1,'builder_recovery':proof})
    root.mkdir(parents=True);(root/'train/bank').mkdir(parents=True)
    target=sqlite3.connect(root/'train/bank/bank.sqlite3');source_db.backup(target);source_db.close()
    with target:
        target.execute('UPDATE metadata SET value=? WHERE key=?',(TRACE_VERSION,'officeqa_method_version'))
        target.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('office_trace_parent',str(parent)))
        for key,row in cases.items():target.execute('UPDATE train_cases SET payload=? WHERE task_key=?',(json.dumps(row,ensure_ascii=False),key))
    target.close()
    configurations={p:configuration(configs['recompile'],root,p,version=TRACE_VERSION) for p in PHASE_LIMITS}
    for config in configurations.values():config['budget'].update(revision['total_limits'])
    ledger=deepcopy(ledger);ledger['limits']=revision['total_limits']
    write_json(root/'budget.json',ledger);write_json(root/'inherited_budget.json',read(parent/'budget.json'))
    write_json(root/'budget_revision.json',revision);write_json(root/'resolved_config.json',configurations)
    write_json(root/'inherited_jobs.json',list(jobs.values()))
    inputs={**old['parent_inputs'],**proofs,**{str(parent/name):sha256(parent/name) for name in
        ('plan.json','resolved_config.json','budget.json','recompile/summary.json','train/bank/bank.sqlite3','final_frozen_manifest.json')}}
    carry={'parent':str(parent),'parent_code':old['code'],'inherited_tokens':charged,'inherited_http':len(ledger['attempts']),
        'unknown_billing':False,'phase_limits':PHASE_LIMITS,'worst_case_tokens':charged+sum(PHASE_LIMITS.values()),
        'authorization_pending':revision.get('approved') is not True,'physical_requests':'exact keyed entries copied once'}
    write_json(root/'cost_carryforward_manifest.json',carry)
    plan={**old,'version':TRACE_VERSION,'code':code,'created_at':utc(),'parent_recompile':str(parent),
        'parent_recompile_code':old['code'],'parent_inputs':inputs,'source_records':reopened+records,
        'canonical_train_count':120,'chain_source_count':sum(bool(r['local_evidence_count']) for r in records),
        'config_hash':digest(configurations),'budget_revision_hash':digest(revision),
        'method_contract_hash':trace_contract_hash(),
        'corpus_identity':corpus_identity(configurations['recompile']['harness']['corpus_root']),
        'initial_bank':tree_identity(root/'train/bank'),'inherited_jobs_sha256':sha256(root/'inherited_jobs.json'),
        'selection_rule':'proven unissued Builder repair, then only new V3 retrieval sources in original canonical seed42 Train order',
        'derivation_reason':'Train-only V3 retrieval evidence; inherited assets and validation receipts remain original'}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':sha256(root/'plan.json')})
    return plan


def derive(parent,output):
    """Explicit code transition before Val; only certified unissued capacity slots reopen."""
    parent,root=Path(parent).resolve(),Path(output).resolve()
    code=code_identity();old=read(parent/'plan.json');configs=read(parent/'resolved_config.json')
    if root.exists() or parent in root.parents or code['tracked_dirty'] or not code['git_sha']:
        raise ValueError('Derivation requires a fresh independent path and committed source')
    if (sha256(parent/'plan.json')!=read(parent/'plan_identity.json')['sha256'] or digest(configs)!=old['config_hash']
            or old['phase_limits']!=PHASE_LIMITS or old['version']!=VERSION):raise ValueError('Parent plan changed')
    for path,expected in old['parent_inputs'].items():
        if sha256(path)!=expected:raise ValueError('Parent source changed: '+path)
    summary=read(parent/'recompile/summary.json')
    if not summary['complete'] or summary['checked']!=len(old['source_records']) or (parent/'val/execution_manifest.json').exists() or (parent/'test/execution_manifest.json').exists():
        raise ValueError('Only a completed pre-Val recompile can derive unissued slots')
    if tree_identity(parent/'train/frozen_bank')['sha256']!=read(parent/'final_frozen_manifest.json')['final_artifact_hash']:
        raise ValueError('Parent Frozen changed')
    ledger=read(parent/'budget.json')
    if ledger['limits']!=TOTAL_LIMITS or ledger['unknown_billing'] or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('Derivation requires a fully metered parent ledger')
    jobs_db=sqlite3.connect((parent/'train/bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    jobs={j['id']:j for j in (json.loads(r[0]) for r in jobs_db.execute('SELECT payload FROM realization_jobs'))}
    records=deepcopy(old['source_records']);reopened=[];proofs={}
    for row in records:
        task_id=row['task']['task_id'];unit=parent/'recompile/units'/task_id
        result=read(unit/'result.json');row.pop('builder_recovery',None)
        row['inherited_result']=str(unit/'result.json')
        proofs[str(unit/'result.json')]=sha256(unit/'result.json')
        if result.get('learning_status')!='learning_capacity_rejected':continue
        audit=read(unit/'requests.json');refused=audit['requests'][-1]
        if not refused.get('material_refusal') or refused.get('http_attempts'):raise ValueError('Capacity recovery needs a final unissued request')
        proof={'request_id':refused['id'],'request_file':str(unit/'requests.json'),'sha256':sha256(unit/'requests.json')}
        proofs[proof['request_file']]=proof['sha256']
        if refused['stage']=='extractor':
            if refused['purpose']!='extractor' or any(r.get('http_attempts') for r in audit['requests']):
                raise ValueError('Cannot reissue a paid Extractor or its spent repair')
        elif refused['stage']=='tool_builder':
            job_id,epoch,generation=refused['job_key'];job=jobs[job_id]
            if (job['generation_count']!=generation+1 or generation not in (0,1)
                    or job['state']!='deferred' or job.get('last_error_kind')!='learning_capacity_rejected'):
                raise ValueError('Not an unissued capacity-deferred Builder slot')
            row['builder_recovery']={**proof,'job_id':job_id,'job_hash':digest(job),'epoch':epoch,'generation':generation}
        else:raise ValueError('Unsupported capacity recovery stage')
        row.pop('inherited_result');row['capacity_recovery']=proof;reopened.append(task_id)
    if not reopened:raise ValueError('No proven unissued capacity requests to derive')
    root.mkdir(parents=True);(root/'train/bank').mkdir(parents=True)
    db=sqlite3.connect(root/'train/bank/bank.sqlite3');jobs_db.backup(db);db.close();jobs_db.close()
    configurations={phase:configuration(configs['recompile'],root,phase) for phase in PHASE_LIMITS}
    for filename in ('budget.json','inherited_jobs.json'):
        shutil.copy2(parent/filename,root/filename)
    write_json(root/'inherited_budget.json',ledger);write_json(root/'resolved_config.json',configurations)
    charged=sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    remaining={phase:limit-sum(a['accounted_tokens'] for a in ledger['attempts'].values()
        if a['context'].get('office_phase')==phase) for phase,limit in PHASE_LIMITS.items()}
    if min(remaining.values())<0 or charged+sum(remaining.values())>TOTAL_LIMITS['token_limit']:
        raise ValueError('Derivation exceeds the frozen cumulative allocation')
    carry={'parent':str(parent),'parent_code':old['code'],'inherited_tokens':charged,'inherited_http':len(ledger['attempts']),
           'phase_limits':PHASE_LIMITS,'remaining_phase_tokens':remaining,'worst_case_tokens':charged+sum(remaining.values()),
           'reopened_unissued_requests':reopened,'unknown_billing':False}
    write_json(root/'cost_carryforward_manifest.json',carry)
    inputs={**old['parent_inputs'],**proofs,**{str(parent/name):sha256(parent/name) for name in
        ('plan.json','budget.json','resolved_config.json','recompile/summary.json','train/bank/bank.sqlite3','final_frozen_manifest.json')}}
    plan={**old,'code':code,'created_at':utc(),'parent_recompile':str(parent),'parent_recompile_code':old['code'],
        'parent_inputs':inputs,'source_records':records,'config_hash':digest(configurations),'initial_bank':tree_identity(root/'train/bank'),
        'reopened_unissued_requests':reopened,'derivation_reason':'versioned full-Bank catalog and Program-only failure projection'}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':sha256(root/'plan.json')})
    return carry


def recompile(output,*,max_new_records=None):
    if max_new_records is not None and max_new_records<1:raise ValueError('Record boundary must be positive')
    root,plan,configs=checked(output)
    if (root/'final_frozen_manifest.json').exists():
        if tree_identity(root/'train/frozen_bank')['sha256']!=read(root/'final_frozen_manifest.json')['final_artifact_hash']:
            raise ValueError('Final Frozen changed')
        return read(root/'recompile/summary.json')
    phase=root/'recompile';settings=configs['recompile']
    governor=PhaseGovernor(root/'budget.json','recompile')
    identity={'method':plan['version'],'source_commit':plan['code']['git_sha'],'source_hash':digest(plan),
              'benchmark':'officeqa','run_seed':42,'phase':'train_recompile'}
    log=FormalLog(phase,identity,settings,resume=(phase/'run_manifest.json').exists())
    s=EmpiricalSystem(settings,budget_governor=governor)
    s.observer=log;s.bank.observer=log
    results=[];new=0;censored=False
    try:
        for index,row in enumerate(plan['source_records']):
            task=PublicTask(**row['task']);folder=phase/'units'/row.get('work_id',task.task_id)
            if (root/'STOP_AFTER_TASK').exists():raise SystemExit(75)
            if (folder/'result.json').exists():results.append(read(folder/'result.json'));continue
            if row.get('inherited_result'):
                result={**read(row['inherited_result']),'inherited_from':row['inherited_result']}
                write_json(folder/'result.json',result);results.append(result);continue
            if max_new_records is not None and new>=max_new_records:break
            if not row['local_evidence_count']:
                result={'task_id':task.task_id,'checked':True,'decision':'no_change',
                    'reason':'no_verified_retrieval_chain' if plan['version']==TRACE_VERSION else 'no_local_operation_source','http':0}
            else:
                s.checkpoint=TaskCheckpoint(folder/'checkpoint');s.audit_path=folder/'requests.json'
                log.begin_task(task,index,row.get('work_id',task.task_id)+':recompile',{'source_record_sha256':row['sha256']})
                source_id={'source_run_id':str(plan['source']),'source_record_hash':row['source_record_hash'],
                           'source_trace_sha256':row['sha256'],'recompile_method':plan['version'],
                           **({'work_type':row['work_type']} if row.get('work_type') else {})}
                try:trace=s.recompile_tool_record(task,read(row['path']),source_id,builder_recovery=row.get('builder_recovery'))
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
            'total_work_units':len(plan['source_records']),
            'decisions':dict(stats),'inherited_units':sum(bool(r.get('inherited_from')) for r in results),
            'builder_http':sum(governor.owns(a) and a['context'].get('purpose')=='builder'
                               for a in governor.state['attempts'].values()),
            'capacity_rejected':sum(r.get('learning_status')=='learning_capacity_rejected' for r in results),
            'programs':[{'id':p['id'],'state':p['state']} for p in s.bank.all('program')],
            'usable':sum(s.bank.program_eligible(p) for p in s.bank.all('program')),
            'source_trials':[a for p in s.bank.all('program') for a in s.bank.attempts(p['id'])],
            'phase_tokens':sum(a['accounted_tokens'] for a in governor.state['attempts'].values()
                if governor.owns(a) and a['status']=='metered')}
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
    log=FormalLog(phase,{'method':plan['version'],'benchmark':'officeqa','run_seed':42,'phase':split,
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
    parser.add_argument('action',choices=['prepare','derive','prepare-v3','recompile','val','test'])
    parser.add_argument('--output',required=True);parser.add_argument('--source');parser.add_argument('--env-file')
    parser.add_argument('--max-new-records',type=int);parser.add_argument('--max-new-tasks',type=int);parser.add_argument('--review')
    parser.add_argument('--budget-revision')
    args=parser.parse_args()
    if args.env_file:load_env(args.env_file)
    if args.action=='prepare':result=prepare(args.source,args.output)
    elif args.action=='prepare-v3':result=prepare_trace(args.source,args.output,args.budget_revision)
    elif args.action=='derive':result=derive(args.source,args.output)
    elif args.action=='recompile':result=recompile(args.output,max_new_records=args.max_new_records)
    else:result=evaluate(args.output,args.action,max_new_tasks=args.max_new_tasks,review=args.review)
    print(json.dumps({k:v for k,v in result.items() if k not in {'source_records','source_trials'}},ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
