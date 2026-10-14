"""One production entry: immutable Train derivation -> Worker/probe -> Frozen -> Val review."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from ..core.errors import BudgetExhausted
from ..empirical import POLICY_DEFAULTS
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.checkpoint import TaskCheckpoint, save_json
from ..empirical.contracts import PublicTask, digest
from ..empirical.searchqa_guidance import VERSION, MATERIAL, POLICY, RuntimeView, applicability
from ..empirical.searchqa_trace_learning import (verify_source, partition, public_case, learning_units, sha, read,
                                               propose_program_specs)
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from .benchmark_config import benchmark_config
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, run, write_json

STAGES = ('verify','prepare','learn','worker_probe','train_probe','freeze','val','test','report','execute')
AUTHORIZED_LIMITS = {'learn':{'tokens':800000,'http':60},'worker_probe':{'tokens':1200000,'http':80},
                     'train_probe':{'tokens':500000,'http':80},'val':{'tokens':600000,'http':144}}


def jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    os.replace(temporary,path)


def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def settings(path):
    value=yaml.safe_load(Path(path).read_text())
    if (value['method_version']!=VERSION or value['seed']!=42 or value['partition']!={'induction':240,'worker_probe':30,'utility_probe':30}
            or value['max_atomic_candidates']!=12 or value['max_program_specs']!=4 or value['builder_generations']!=2
            or value['wire_max_bytes']!=65536 or {k:value['phase_limits'][k] for k in AUTHORIZED_LIMITS}!=AUTHORIZED_LIMITS
            or value['authorized_stages']!=list(AUTHORIZED_LIMITS) or not value['authorization']
            or value['utility_rule']!='observe_only_user_decides_after_val'):
        raise ValueError('R2 authorized configuration changed')
    return value


def verify(args):
    if Path(args.output).resolve()==Path(args.source_run).resolve() or Path(args.source_run).resolve() in Path(args.output).resolve().parents:
        raise ValueError('R2 output must be independent of immutable parent')
    identity,cases,old=verify_source(args.source_run,args.datasets)
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'derived_run_manifest.json').exists(): raise ValueError('Existing derived run uses --resume, not a new verification')
    write_json(root/'source_identity.json',identity)
    jsonl(root/'train_source_catalog.jsonl',cases)
    write_json(root/'source_config.json',old)
    write_json(root/'train_partition.json',partition([PublicTask(**r['task']) for r in cases]))
    jsonl(root/'public_evidence_index.jsonl',[public_case(r) for r in cases])
    groups=read(root/'train_partition.json')
    write_json(root/'induction_units.json',learning_units(cases,groups))
    write_json(root/'verify.json',{'complete':True,'http':0,'source_identity_hash':digest(identity),
        'counts':identity['counts'],'records':len(cases),'partition_hash':digest(groups),
        'learning_isolation':'readonly bubblewrap mounts; Val/Test/evaluator paths denied before loading records'})
    return read(root/'verify.json')


def configuration(old,root,*,phase,arm='Atomic-full',bank=None):
    config=deepcopy(old)
    for section,values in POLICY_DEFAULTS.items():config.setdefault(section,{}).update(deepcopy(values))
    config['experiment'].update(searchqa_method_version=VERSION,output_dir=str(root/phase/arm),
                                runtime_mode='frozen' if bank else 'online')
    config['data_dir']=str(bank or root/'train/bank')
    config=benchmark_config(config,'searchqa')
    config['budget']['eval_task_tokens']=96000
    return validate_config(config)


def prepare(args):
    root=Path(args.output).resolve()
    if not (root/'verify.json').exists():verify(args)
    if (root/'derived_run_manifest.json').exists():
        if not args.resume:raise ValueError('Existing R2 run requires --resume')
        return checked(root)[0]
    cfg=settings(args.config)
    code=code_identity()
    if not code['git_sha'] or code['tracked_dirty']:raise ValueError('Prepare only committed, clean source before paid requests')
    source=read(root/'source_identity.json')
    for path,expected in source['source_files'].items():
        if sha(path)!=expected:raise ValueError('Original source changed: '+path)
    cases=rows(root/'train_source_catalog.jsonl');groups=read(root/'train_partition.json')
    public=read(root/'source_config.json')
    config=configuration(public,root,phase='learn')
    if (root/'train/bank').exists():raise ValueError('Fresh R2 must not inherit assets/checkpoints')
    bank=Bank(root/'train/bank',seed=42,environment=config['program_environment'])
    bank.bind_searchqa_method(VERSION)
    role={key:group for group,keys in groups.items() if isinstance(keys,list) for key in keys}
    for row in cases:bank.save_case(PublicTask(**row['task']),{'searchqa_role':role[row['physical_key']],
        'historical_operation_count':0,'public_fields_sha256':row['public_fields_sha256']})
    initial=bank.digest();bank.close()
    ledger=read(Path(args.source_run)/'budget.json')
    old_cost=source['old_total_tokens'];old_http=source['old_total_http']
    if (old_cost+3100000>ledger['limits']['token_limit'] or old_http+364>ledger['limits']['request_limit']):
        raise ValueError('Approved cumulative SearchQA allowance cannot cover R2 allocation')
    write_json(root/'budget.json',ledger)
    write_json(root/'inherited_budget.json',ledger)
    write_json(root/'resolved_config.json',config)
    locked=['source_identity.json','train_source_catalog.jsonl','source_config.json','train_partition.json',
            'public_evidence_index.jsonl','induction_units.json','resolved_config.json','inherited_budget.json']
    plan={'method_version':VERSION,'run_id':root.name,'created_at':utc(),'code':code,'source_run':str(Path(args.source_run).resolve()),
        'datasets':str(Path(args.datasets).resolve()),'config':cfg,'config_file_sha256':sha(args.config),
        'files':{name:sha(root/name) for name in locked},'source_hash':digest(source),'initial_bank_digest':initial,
        'source_trace_runtime':'historical Guidance-on direct answers; not new solve or new execution',
        'new_initial_assets':0,'new_http_authorized':364,'new_tokens_authorized':3100000,
        'old_train_tokens':source['old_train_tokens'],'old_train_http':source['old_train_http'],
        'inherited_cumulative_tokens':old_cost,'inherited_cumulative_http':old_http,
        'stage_order':['learn','worker_probe','train_probe','freeze','val','review_required','test_separately_authorized']}
    write_json(root/'derived_run_manifest.json',plan)
    write_json(root/'plan_identity.json',{'sha256':sha(root/'derived_run_manifest.json')})
    return plan


def checked(root):
    root=Path(root).resolve();plan=read(root/'derived_run_manifest.json')
    if sha(root/'derived_run_manifest.json')!=read(root/'plan_identity.json')['sha256'] or plan['code']!=code_identity():
        raise ValueError('Code/run identity changed; no cross-version resume')
    for name,expected in plan['files'].items():
        if sha(root/name)!=expected:raise ValueError('Derived identity changed: '+name)
    source=read(root/'source_identity.json')
    for path,expected in source['source_files'].items():
        if sha(path)!=expected:raise ValueError('Immutable parent changed: '+path)
    return plan,read(root/'resolved_config.json')


def check_arguments(args,plan):
    if (str(Path(args.source_run).resolve())!=plan['source_run'] or
            str(Path(args.datasets).resolve())!=plan['datasets'] or sha(args.config)!=plan['config_file_sha256']):
        raise ValueError('Resume source/datasets/config identity changed')


def stage_receipt(root,stage,result=None):
    """Completed stages are verified, never replayed after downstream Bank changes."""
    root=Path(root);path=root/'stages'/(stage+'.json')
    if path.exists():
        saved=read(path)
        for name,expected in saved['files'].items():
            if sha(root/name)!=expected:raise ValueError('Completed stage artifact changed: '+name)
        return saved['result']
    if result is None:return None
    extra={'prepare':['derived_run_manifest.json','plan_identity.json'],
        'learn':['atomic_candidates.jsonl','candidate_rejections.jsonl','source_ref_audit.json'],
        'train_probe':['deployment_policy.json','probe_frozen_bank'],
        'freeze':['final_frozen_manifest.json','train/frozen_bank']}
    paths=[root/stage,*[root/p for p in extra.get(stage,[])]];files={}
    for p in paths:
        for f in sorted(p.rglob('*')) if p.is_dir() else [p]:
            if f.is_file() and not f.name.endswith(('-shm','-wal')):files[f.relative_to(root).as_posix()]=sha(f)
    write_json(path,{'completed_at':utc(),'files':files,'result':result})
    return result


class PhaseGovernor(BudgetGovernor):
    def __init__(self,root,phase):
        self.root,self.phase=Path(root),phase
        self.plan=read(self.root/'derived_run_manifest.json')
        if phase not in self.plan['config']['authorized_stages']:
            review=read(self.root/'test_authorization.json') if (self.root/'test_authorization.json').exists() else {}
            if phase not in {'test_full','test_off'} or review.get('approved') is not True or review.get('val_report_sha256')!=sha(self.root/'val/report.json'):
                raise ValueError('This phase has no explicit paid authorization')
        self.cap=self.plan['config']['phase_limits'][phase]
        super().__init__(self.root/'budget.json',**read(self.root/'inherited_budget.json')['limits'])

    def owns(self,a):return a['context'].get('searchqa_run_id')==self.plan['run_id'] and a['context'].get('searchqa_phase')==self.phase

    def headroom(self,context):
        room=super().headroom(context)
        own=[a for a in self.state['attempts'].values() if self.owns(a)]
        room['tokens']=min(room['tokens'],self.cap['tokens']-sum(a.get('accounted_tokens',a['reserved_tokens']) for a in own)-context.get('parent_finish_reserve',0))
        room['requests']=min(room['requests'],self.cap['http']-len(own))
        return room

    def admit(self,key,payload,context):
        context={**context,'searchqa_run_id':self.plan['run_id'],'searchqa_phase':self.phase,'searchqa_method':VERSION}
        room=self.headroom(context)
        reserve=input_token_bound(payload)+payload.get('max_tokens',payload.get('max_completion_tokens',0))
        if reserve>room['tokens'] or room['requests']<=context.get('finish_request_reserve',0):
            raise BudgetExhausted('diagnostic_budget_exhausted','R2 phase cap reached; no borrowing or automatic rerun')
        return super().admit(key,payload,context)


@contextmanager
def learning_system(root,stage):
    plan,config=checked(root)
    system=EmpiricalSystem(config,harness=AnswerAdapter('searchqa',{}),budget_governor=PhaseGovernor(root,stage))
    try:yield system
    finally:system.close()


def unit(system,folder,identity):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    system.checkpoint=TaskCheckpoint(folder/'checkpoint')
    system.audit_path=folder/'requests.json';system.phase='train_recompile';system.budget_scope=identity
    system._task_start=len(system.usage.events);system._request_start=len(system.requests)
    system._learning_start=system._task_start
    system.prior_usage=read(system.audit_path)['usage'] if system.audit_path.exists() else []
    system.request_attribution={'parent_scope':identity,'training_unit_type':'train_trace_derivation',
                               'source_run_hash':digest(read(folder.parents[1]/'source_identity.json')) if (folder.parents[1]/'source_identity.json').exists() else None}


def model_failure(exc):
    # Refused model candidates are data; source, billing and infrastructure errors stop the run.
    return isinstance(exc,(SyntaxError,ValueError)) and not any(x in str(exc) for x in (
        'identity changed','across recovery','Cached learning','Duplicate HTTP','Paid legacy','Unknown HTTP','material_too_large'))


def learn(root):
    root=Path(root);plan,_=checked(root);phase=root/'learn'
    source={r['case_id']:r for r in rows(root/'public_evidence_index.jsonl')}
    units=read(root/'induction_units.json');results=[]
    with learning_system(root,'learn') as system:
        for u in units:
            if (root/'STOP_AFTER_TASK').exists():raise SystemExit(75)
            folder=phase/u['id'];path=folder/'result.json'
            if path.exists():results.append(read(path));continue
            unit(system,folder,'r2:induction:'+u['id'])
            started=utc()
            try:value=system.learner.learn_searchqa_trace(u,source)
            except (ValueError,SyntaxError) as exc:
                if not model_failure(exc):raise
                value={'decision':'rejected','reason':str(exc)}
            value.update(unit=u,start_time=started,end_time=utc())
            write_json(path,value);results.append(value)
            print(json.dumps({'stage':'learn','unit':u['id'],'decision':value['decision']},ensure_ascii=False),flush=True)
    jsonl(root/'atomic_candidates.jsonl',[r for r in results if r['decision']=='candidate'])
    jsonl(root/'candidate_rejections.jsonl',[r for r in results if r['decision']=='rejected'])
    summary={'complete':True,'units':len(results),'examples_consumed':len({k for r in results for k in r['unit']['case_ids']}),
             'induction_pool':240,'decisions':dict(Counter(r['decision'] for r in results))}
    write_json(phase/'summary.json',summary)
    write_json(root/'source_ref_audit.json',{'candidate_count':summary['decisions'].get('candidate',0),'validated_by':'exact Unicode offsets and quote SHA; at least two distinct induction sources',
        'rejections':[{'unit':r['unit']['id'],'reason':r['reason']} for r in results if r['decision']=='rejected']})
    return summary


def worker_probe(root):
    from ..empirical.searchqa_program_probe import build, execute, qualify, inputs
    root=Path(root);plan,_=checked(root)
    if not read(root/'learn/summary.json')['complete']:raise ValueError('Induction is incomplete')
    sources={r['case_id']:r for r in rows(root/'public_evidence_index.jsonl')}
    original={r['physical_key']:PublicTask(**r['task']) for r in rows(root/'train_source_catalog.jsonl')}
    groups=read(root/'train_partition.json');results=[]
    with learning_system(root,'worker_probe') as system:
        # Coverage uses public applicability only, never scores or held-out responses.
        ranked=sorted(system.bank.all('skill'),key=lambda a:(-sum(applicability(original[k],a['entry_constraints'])[0]
            for k in groups['induction']),a['id']))[:4]
        choice={'atomic_ids':[a['id'] for a in ranked],'rule':'public applicability coverage in induction240; ID tie-break'}
        lock=root/'worker_probe/spec_selection.json'
        if lock.exists() and read(lock)!=choice:raise ValueError('Pre-registered Program candidate order changed')
        write_json(lock,choice)
        for asset in ranked:
            if (root/'STOP_AFTER_TASK').exists():raise SystemExit(75)
            folder=root/'worker_probe'/asset['id'];out=folder/'result.json'
            if out.exists():results.append(read(out));continue
            unit(system,folder,'r2:program_spec:'+asset['id'])
            try:spec=propose_program_specs(system.learner,asset,sources)
            except (ValueError,SyntaxError) as exc:
                if not model_failure(exc):raise
                value={'skill_id':asset['id'],'decision':'rejected','reason':str(exc)}
                write_json(out,value);results.append(value);continue
            if spec is None:
                value={'skill_id':asset['id'],'decision':'no_change','reason':'no_deterministic_program_proposed'}
                write_json(out,value);results.append(value);continue
            write_json(folder/'spec.json',spec)
            attempts=[];previous=None;qualified=None;program=None
            for generation in range(2):
                build_folder=folder/('generation_'+str(generation));built=build_folder/'program.json'
                if built.exists():program=system.bank.get(read(built)['id'])
                else:
                    unit(system,build_folder,'r2:builder:'+asset['id']+':'+str(generation))
                    try:
                        proposal=build(system.learner,spec,asset,sources,generation=generation,previous=previous)
                        program=system.bank.put('program',proposal)
                        write_json(built,program)
                    except (ValueError,SyntaxError) as exc:
                        if not model_failure(exc):raise
                        previous={'reason':str(exc),'program_id':None,'generation':generation}
                        write_json(build_folder/'rejection.json',previous);continue
                source=original[spec['source_case_ids'][0]]
                receipt=execute(system.bank,system.worker,program,source,build_folder/'receipts',role='source',governor=system.budget_governor)
                attempts.append(receipt)
                if receipt['check']['passed']:break
                previous={'program_id':program['id'],'source':program['source'],'reason':receipt['check']['reason'],
                          'worker_result':receipt['result'],'generation':generation}
            if not program or not attempts or not attempts[-1]['check']['passed']:
                value={'skill_id':asset['id'],'decision':'candidate_rejected','reason':'source_worker_not_verified','previous':previous}
                write_json(out,value);results.append(value);continue
            records=[attempts[-1]];probe_rows=[]
            for key in groups['worker_probe']:
                task=original[key];okay,reason=applicability(task,spec['applicability'])
                if not okay:probe_rows.append({'physical_key':key,'applicable':False,'reason':reason});continue
                row=execute(system.bank,system.worker,program,task,folder/'transfer_receipts',role='transfer',governor=system.budget_governor)
                records.append(row);probe_rows.append({'physical_key':key,'applicable':True,'passed':row['check']['passed'],'receipt':row['id']})
            for name,arguments in [('empty_context',{**inputs(source),'context':''}),('empty_question',{**inputs(source),'question':''}),
                    ('unicode_parameter',{**inputs(source),'context':'[DOC] 中文前缀：独立参数变化。\n'+source.inputs['context']})]:
                records.append(execute(system.bank,system.worker,program,source,folder/'boundary_receipts',
                    role='boundary:'+name,arguments=arguments,governor=system.budget_governor))
            qualified=qualify(system.bank,program,records)
            value={'skill_id':asset['id'],'program_id':program['id'],'decision':'usable' if qualified['passed'] else 'candidate',
                   'qualification':qualified,'probe_coverage':probe_rows,'historical_operation_count':0,
                   'new_worker_invocations':len(records)}
            write_json(out,value);results.append(value)
            print(json.dumps({'stage':'worker_probe','program':program['id'],'decision':value['decision']},ensure_ascii=False),flush=True)
    summary={'complete':True,'spec_units':len(results),'decisions':dict(Counter(r['decision'] for r in results)),'results':results}
    write_json(root/'worker_probe/summary.json',summary)
    return summary


def freeze_snapshot(root,name):
    _,config=checked(root);path=Path(root)/name
    bank=Bank(Path(root)/'train/bank',seed=42,environment=config['program_environment'])
    bank.bind_searchqa_method(VERSION)
    try:
        if path.exists():
            old=read(path/'freeze.json')
            if old['source_digest']!=bank.digest():raise ValueError('Existing frozen snapshot differs from training Bank')
            return old
        return bank.freeze(path)
    finally:bank.close()


def eval_config(root,stage,arm,bank,split):
    plan,base=checked(root)
    config=configuration(base,Path(root),phase=stage,arm=arm,bank=bank)
    datasets=Path(plan['datasets'])
    if datasets.name!='searchqa':datasets=datasets/'searchqa'
    config['harness']['evaluator_records']=str(datasets/('evaluator_records_'+split+'.json'))
    config['manifest']=str(Path(read(Path(root)/'source_config.json')['manifest']).with_name(split+'.json'))
    return validate_config(config)


def episode_group(root,stage,arm,tasks,bank,split):
    root=Path(root);config=eval_config(root,stage,arm,bank,split);phase=root/stage/arm
    before=tree_identity(bank)['sha256']
    log=FormalLog(phase,{'method':VERSION,'benchmark':'searchqa','run_seed':42,'phase':stage,'arm':arm,
        'frozen_hash':before,'derived_run_manifest_sha256':sha(root/'derived_run_manifest.json')},config,
        resume=(phase/'run_manifest.json').exists())
    try:
        result=run(config,tasks,phase,resume=(phase/'execution_manifest.json').exists(),readonly=True,
            formal_log=log,bank_view_factory=lambda b:RuntimeView(b,arm),budget_governor=PhaseGovernor(root,stage),freeze_on_complete=False)
        log.finish('completed' if result['complete'] else 'paused')
        if not result['complete']:raise RuntimeError('Incomplete episode group; cannot advance to downstream stage')
        return result
    except BaseException:
        log.finish('interrupted');raise
    finally:
        after=tree_identity(bank)['sha256'];write_json(phase/'frozen_check.json',{'before':before,'after':after,'unchanged':before==after})
        if after!=before:raise RuntimeError('Frozen changed during evaluation')


def usage_totals(usage):
    result={k:sum(u.get(k,0) or 0 for u in usage) for k in ('total_tokens','prompt_tokens','completion_tokens')}
    result['reasoning_missing']=sum(u.get('reasoning_tokens') is None for u in usage)
    result['reasoning_tokens']=None if result['reasoning_missing'] else sum(u['reasoning_tokens'] for u in usage)
    return result


def pair_report(root,stage,arms,phases=None):
    root=Path(root);data={};by_arm={}
    for arm in arms:
        summary=read(root/(phases or {}).get(arm,stage)/arm/'summary.json')
        if not summary['complete']:raise ValueError('Incomplete split cannot be reported as complete')
        cases=summary['cases'];by_arm[arm]=cases
        requests=[r for t in cases for r in t['requests']]
        usage=[u for t in cases for u in t['usage']]
        data[arm]={'tasks':len(cases),'correct':sum(t['score']['hard'] for t in cases),
            'em':sum(t['score']['hard'] for t in cases)/len(cases),
            'f1':sum(t['score']['soft'] for t in cases)/len(cases),
            'tokens':sum(u['total_tokens'] for u in usage),'prompt_tokens':sum(u['prompt_tokens'] for u in usage),
            'completion_tokens':sum(u['completion_tokens'] for u in usage),
            'reasoning_tokens':usage_totals(usage)['reasoning_tokens'],
            'reasoning_missing':usage_totals(usage)['reasoning_missing'],
            'http':sum(len(r.get('http_attempts',[])) for r in requests),'llm_steps':len(requests),
            'guidance_covered':sum(bool(t.get('injected_guidance_ids')) for t in cases),
            'guidance_injections':sum(len(t.get('injected_guidance_ids',[])) for t in cases),
            'program_invocations':sum(len(t['execution'].get('attempts',[])) for t in cases),
            'program_consumed':sum(bool(a.get('outputs_consumed')) for t in cases for a in t['execution'].get('attempts',[])),
            'temporary_executions':sum(len(t['execution'].get('temporary_executions',[])) for t in cases),
            'worker_seconds':sum(a.get('worker_seconds') or 0 for t in cases for a in t['execution'].get('attempts',[])+t['execution'].get('temporary_executions',[])),
            'worker_cpu_seconds':sum(a.get('worker_cpu_seconds') or 0 for t in cases for a in t['execution'].get('attempts',[])+t['execution'].get('temporary_executions',[])),
            'empty':sum(bool(t['execution'].get('empty_answer')) for t in cases),
            'truncated':sum(bool(t['execution'].get('completion_truncated')) for t in cases),
            'budget_censored':sum(bool(t['execution'].get('budget_censored')) for t in cases)}
    pairs={}
    for i,left in enumerate(arms):
        for right in arms[i+1:]:
            a,b=by_arm[left],by_arm[right]
            if [t['task'] for t in a]!=[t['task'] for t in b]:raise ValueError('Paired public tasks/order mismatch')
            wins=sum(y['score']['hard'] and not x['score']['hard'] for x,y in zip(a,b))
            losses=sum(x['score']['hard'] and not y['score']['hard'] for x,y in zip(a,b))
            pairs[left+'->'+right]={'wins':wins,'losses':losses,'ties':len(a)-wins-losses,
                'token_difference':data[right]['tokens']-data[left]['tokens']}
    result={'method':VERSION,'stage':stage,'arms':data,'pairs':pairs,'generation_seed_supported':False,
        'interpretation':'Development observation; no significance claim from one small paired run'}
    write_json(root/stage/'report.json',result)
    return result


def train_probe(root):
    root=Path(root);plan,_=checked(root)
    if not read(root/'worker_probe/summary.json')['complete']:raise ValueError('Worker probes incomplete')
    freeze_snapshot(root,'probe_frozen_bank')
    cases={r['physical_key']:PublicTask(**r['task']) for r in rows(root/'train_source_catalog.jsonl')}
    tasks=[cases[k] for k in read(root/'train_partition.json')['utility_probe']]
    for arm in ('NoSkill','Atomic-full'):episode_group(root,'train_probe',arm,tasks,root/'probe_frozen_bank','train')
    report=pair_report(root,'train_probe',['NoSkill','Atomic-full'])
    decision={'rule':plan['config']['utility_rule'],'utility_passed':None,'report_sha256':sha(root/'train_probe/report.json'),
        'selector':POLICY,'observation_only':True,'retain_qualified_assets':True,
        'reason':'User requested the paired results before deciding utility; no score-based filtering or retuning'}
    path=root/'deployment_policy.json'
    if path.exists() and read(path)!=decision:raise ValueError('One-time utility decision cannot change')
    write_json(path,decision)
    return decision


def freeze(root):
    root=Path(root);checked(root);decision=read(root/'deployment_policy.json')
    if decision['report_sha256']!=sha(root/'train_probe/report.json'):raise ValueError('Train utility result changed')
    if not decision['observation_only'] or not decision['retain_qualified_assets']:raise ValueError('Observation policy changed')
    snapshot=freeze_snapshot(root,'train/frozen_bank')
    result={'freeze_time':utc(),'freeze':snapshot,'file_tree_sha256':tree_identity(root/'train/frozen_bank')['sha256'],
            'deployment_policy_sha256':sha(root/'deployment_policy.json'),'release_status':'observation_only_pending_user_review'}
    path=root/'final_frozen_manifest.json'
    if path.exists():
        old=read(path)
        if old['file_tree_sha256']!=result['file_tree_sha256']:raise ValueError('Frozen identity changed')
        return old
    write_json(path,result);return result


def evaluation_tasks(root,split):
    plan,_=checked(root);datasets=Path(plan['datasets'])
    if datasets.name!='searchqa':datasets=datasets/'searchqa'
    material=read(datasets/(split+'.json'))
    old=read(Path(root)/'source_config.json');path=Path(old['manifest']).with_name(split+'.json')
    authority=read(path.parent.parent/'manifest.json');canonical=read(path)
    if (material['authority_sha256']!=sha(path.parent.parent/'manifest.json') or
            material['split_sha256']!=sha(path) or sha(path)!=authority['benchmarks']['searchqa']['splits'][split]['sha256'] or
            [t['task_id'] for t in material['tasks']]!=[t['task_id'] for t in canonical['tasks']] or
            len(material['tasks'])!=(24 if split=='val' else 1400)):
        raise ValueError('Evaluation authority/order/count mismatch')
    return [PublicTask(**t) for t in material['tasks']]


def evaluate(root,split='val',review=None):
    root=Path(root);plan,_=checked(root)
    if split=='test':
        if not review:raise ValueError('Test requires separate human approval after Val review')
        approved=read(review)
        if (approved.get('approved') is not True or not approved.get('authorization') or
                approved.get('val_report_sha256')!=sha(root/'val/report.json') or
                approved.get('phase_limits')!={k:plan['config']['phase_limits'][k] for k in ('test_off','test_full')}):
            raise ValueError('Review must authorize this completed Val and both fixed Test stage budgets')
        write_json(root/'test_authorization.json',approved)
    frozen=root/'train/frozen_bank';identity=read(root/'final_frozen_manifest.json')
    if tree_identity(frozen)['sha256']!=identity['file_tree_sha256']:raise ValueError('Frozen changed before evaluation')
    tasks=evaluation_tasks(root,split)
    b=Bank(frozen,readonly=True,seed=42)
    try:skills=len(b.all('skill'));programs=len(b.all('program'))
    finally:b.close()
    arms=['NoSkill']+(['Guidance-only'] if skills else [])+(['Atomic-full'] if programs else [])
    if split=='test':arms=['NoSkill']+(['Atomic-full'] if skills or programs else [])
    phases={arm:split if split=='val' else 'test_off' if arm=='NoSkill' else 'test_full' for arm in arms}
    lock={'arms':arms,'tasks_sha256':digest([asdict(t) for t in tasks]),'code':plan['code'],
          'frozen_hash':identity['file_tree_sha256'],'skip_reason':None if programs else 'program_not_available',
          'input_contract':'identical full canonical question/context; intervention only in learned visibility'}
    lockpath=root/split/'evaluation_lock.json'
    if lockpath.exists() and read(lockpath)!=lock:raise ValueError('Evaluation arm identity changed')
    write_json(lockpath,lock)
    for arm in arms:episode_group(root,phases[arm],arm,tasks,frozen,split)
    return pair_report(root,split,arms,phases)


def report(root):
    root=Path(root);plan,_=checked(root);ledger=read(root/'budget.json')
    new={k:a for k,a in ledger['attempts'].items() if k not in read(root/'inherited_budget.json')['attempts']}
    phase={}
    for name in plan['config']['phase_limits']:
        entries=[a for a in new.values() if a['context'].get('searchqa_phase')==name]
        usage=[{**a.get('actual_usage',{}),'reasoning_tokens':a.get('actual_usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')}
               for a in entries if a['status']=='metered']
        phase[name]={'http':len(entries),'known_tokens':sum(a.get('accounted_tokens',0) for a in entries),
            'unknown_billing':sum(a['status']!='metered' for a in entries),
            'reserved_or_accounted_tokens':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in entries),
            'usage':usage_totals(usage)}
    local=[]
    for p in (root/'worker_probe').rglob('*.json'):
        row=read(p)
        if isinstance(row,dict) and row.get('policy')=='searchqa.worker-transfer-qualification.r2' and 'result' in row:local.append(row)
    audits=[read(p) for name in ('learn','worker_probe') for p in (root/name).rglob('requests.json')]
    learning_requests=[r for audit in audits for r in audit['requests']]
    local_cost={'invocations':len(local),'cpu_seconds':sum(r['result'].get('cpu_seconds') or 0 for r in local),
        'cpu_missing':sum(r['result'].get('cpu_seconds') is None for r in local),
        'wall_seconds':sum(r['end_time']-r['start_time'] for r in local)}
    outcome={'run_id':plan['run_id'],'code':plan['code'],'method':VERSION,'source_train_tokens':plan['old_train_tokens'],
        'source_train_http':plan['old_train_http'],'inherited_cumulative_tokens':plan['inherited_cumulative_tokens'],
        'incremental':phase,'incremental_known_tokens':sum(a.get('accounted_tokens',0) for a in new.values()),
        'incremental_http':len(new),'unknown_billing':ledger['unknown_billing'],'parent_immutable':True,
        'source_train_usage':usage_totals([u for t in read(Path(plan['source_run'])/'train/summary.json')['cases'] for u in t['usage']]),
        'end_to_end_tokens':plan['old_train_tokens']+sum(a.get('accounted_tokens',0) for a in new.values()),
        'worker_probe_cost':local_cost,
        'learning_logical_requests':len(learning_requests),
        'learning_http_retries':sum(max(0,len(r.get('http_attempts',[]))-1) for r in learning_requests),
        'learning':read(root/'learn/summary.json') if (root/'learn/summary.json').exists() else None,
        'worker':read(root/'worker_probe/summary.json') if (root/'worker_probe/summary.json').exists() else None,
        'utility':read(root/'train_probe/report.json') if (root/'train_probe/report.json').exists() else None,
        'val':read(root/'val/report.json') if (root/'val/report.json').exists() else None,
        'test_authorized':(root/'test_authorization.json').exists()}
    write_json(root/'report.json',outcome);return outcome


def isolation_command(args):
    """Linux mount permissions deny held-out/gold reads and all original writes."""
    root=Path(args.output).resolve()
    if root==Path(args.source_run).resolve() or Path(args.source_run).resolve() in root.parents:
        raise ValueError('R2 output must be independent of immutable parent')
    root.mkdir(parents=True,exist_ok=True)
    forbidden=root/'.denied_file';forbidden.touch(exist_ok=True);forbidden.chmod(0)
    blocked=root/'.denied_dir';blocked.mkdir(exist_ok=True);blocked.chmod(0)
    source=Path(args.source_run).resolve();datasets=Path(args.datasets).resolve()
    if datasets.name!='searchqa':datasets=datasets/'searchqa'
    old=read(source/'train/config.json');authority=Path(old['manifest']).parent
    dirs=[source/'val',source/'test',datasets.parent/'_resource_pool']
    files=[datasets/(split+'.json') for split in ('val','test','reserve')]
    files += [authority/(split+'.json') for split in ('val','test','reserve')]
    files += list(datasets.glob('evaluator_records_*.json'))
    if args.stage=='train_probe':files=[p for p in files if p.name!='evaluator_records_train.json']
    command=['bwrap','--die-with-parent','--ro-bind','/','/','--unshare-pid','--proc','/proc','--dev-bind','/dev','/dev',
        '--bind',str(root),str(root),'--bind','/tmp','/tmp','--setenv','GIT_OPTIONAL_LOCKS','0',
        '--setenv','SEARCHQA_R2_SANDBOX','1','--setenv','SEARCHQA_DENIED_PATHS',json.dumps([str(p) for p in files+dirs if p.exists()])]
    for path in dirs:
        if path.exists():command+=['--ro-bind',str(blocked),str(path)]
    for path in files:
        if path.exists():command+=['--ro-bind',str(forbidden),str(path)]
    return command+[sys.executable,'-m',__spec__.name,*sys.argv[1:]]


def validate_isolation():
    if os.environ.get('SEARCHQA_R2_SANDBOX')!='1':raise RuntimeError('Train derivation requires the permission sandbox')
    paths=json.loads(os.environ['SEARCHQA_DENIED_PATHS'])
    for path in paths:
        try:
            p=Path(path)
            if p.is_dir():list(p.iterdir())
            else:p.read_bytes()
        except PermissionError:continue
        raise RuntimeError('Held-out read denial failed: '+path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,default='verify')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--config',required=True);parser.add_argument('--source-run',required=True)
    parser.add_argument('--datasets',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--env-file');parser.add_argument('--review')
    args=parser.parse_args()
    if args.verify_only:args.stage='verify'
    if args.stage=='execute':
        for stage in ['prepare','learn','worker_probe','train_probe','freeze','val','report']:
            command=[sys.executable,'-m',__spec__.name,'--stage',stage,'--config',args.config,'--source-run',args.source_run,
                '--datasets',args.datasets,'--output',args.output]
            if args.resume:command+=['--resume']
            if args.env_file:command+=['--env-file',args.env_file]
            subprocess.run(command,check=True)
        return
    if args.stage in {'verify','prepare','learn','worker_probe','train_probe','freeze'}:
        if not os.environ.get('SEARCHQA_R2_SANDBOX'):
            raise SystemExit(subprocess.call(isolation_command(args)))
        validate_isolation()
    if args.env_file:load_env(args.env_file)
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another R2 stage owns this run; refusing duplicate launch')
        if (root/'STOP_AFTER_TASK').exists():raise SystemExit(75)
        write_json(root/'status.json',{'stage':args.stage,'state':'running','updated_at':utc(),'pid':os.getpid()})
        try:
            if (root/'derived_run_manifest.json').exists():check_arguments(args,checked(root)[0])
            previous=stage_receipt(root,args.stage) if args.stage not in {'verify','report'} else None
            if previous is not None:
                if not args.resume:raise ValueError('Completed stage requires --resume to verify its existing result')
                result=previous
            else:
                if args.stage in {'verify','prepare'}:result=globals()[args.stage](args)
                elif args.stage in {'val','test'}:result=evaluate(root,args.stage,args.review)
                else:result=globals()[args.stage](root)
                if args.stage not in {'verify','report'}:stage_receipt(root,args.stage,result)
        except BaseException as exc:
            write_json(root/'status.json',{'stage':args.stage,'state':'stopped','updated_at':utc(),'error':str(exc),'type':type(exc).__name__})
            raise
        write_json(root/'status.json',{'stage':args.stage,'state':'completed','updated_at':utc()})
        print(json.dumps({'stage':args.stage,'result':result},ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
