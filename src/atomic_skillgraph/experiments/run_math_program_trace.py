"""Fresh Train60 trace learning, real local probes, immutable candidate Bank, then Val17."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from ..empirical import math_program_induction as m
from ..empirical.math_program_trial import VERSION, verified_bundles
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.contracts import PublicTask,digest
from ..empirical.system import validate_config
from ..harness.registry import create_simple_harness
from . import run_math_program_p5 as pilot
from .canonical_manifest import sha256,verify
from .formal_log import FormalLog,tree_identity,utc
from .run_empirical import code_identity,run,write_json,load_env
from .run_math_p3 import StageGovernor

CONTRACT=Path(__file__).parents[3]/'configs/livemath_p5_trace_seed42.yaml'
read=pilot.read


def schedule(rows,opportunities,settings):
    if settings.get('formal_version')!=VERSION or settings.get('max_candidates_per_task')!=2:
        raise ValueError('Trace learning requires the predeclared two-candidate policy')
    selected=[];coverage=[]
    for row in rows:
        key=row['task']['physical_key']
        options=[o for o in opportunities if o['source_case_ids']==[key]]
        options=sorted(options,key=lambda o:digest([settings['seed'],o['opportunity_id']]))[:2]
        if len({o['requested_local_operation'] for o in options})!=len(options):raise ValueError('Duplicate operation for a source')
        selected.extend(options)
        coverage.append({'task_id':row['task']['task_id'],'physical_key':key,
            'opportunity_ids':[o['opportunity_id'] for o in options],
            'status':'scheduled' if options else 'no_supported_finite_demand',
            'basis':'Host public-demand catalog, not a model abstention or theorem judgement'})
    return selected,coverage


def allocation(settings,prefix):
    """Reserve parallel runs' full caps, without turning reservations into fictitious usage."""
    charged=sum(v.get('accounted_tokens',v['reserved_tokens']) for v in prefix['attempts'].values())
    tokens=charged;http=len(prefix['attempts']);external=[]
    for spec in settings['concurrent_runs']:
        root=Path(spec['path']);plan=read(root/'plan.json');old=read(root/'campaign_budget_prefix.json')
        if plan['test_limits']!={k:spec[k] for k in ('token_limit','request_limit')}:
            raise ValueError('Concurrent manual Test authority differs')
        if old['limits']!=prefix['limits'] or any(prefix['attempts'].get(k)!=v for k,v in old['attempts'].items()):
            raise ValueError('Parallel budget branches lack a common identical historical prefix')
        current=read(root/'budget.json')
        if any(current['attempts'].get(k)!=v for k,v in old['attempts'].items()):raise ValueError('Parallel historical charge changed')
        added={k:v for k,v in current['attempts'].items() if k not in old['attempts']}
        if any(k in prefix['attempts'] for k in added):raise ValueError('Parallel charges already inherited; explicit rebasing required')
        actual=sum(v.get('accounted_tokens',v['reserved_tokens']) for v in added.values())
        if actual>spec['token_limit'] or len(added)>spec['request_limit']:raise ValueError('Parallel stage exceeded its reserved allocation')
        tokens+=spec['token_limit'];http+=spec['request_limit']
        external.append({'path':str(root),'plan_hash':sha256(root/'plan.json'),'reserved_ceiling':{k:spec[k] for k in ('token_limit','request_limit')},
            'note':'separate live ledger; capacity reservation is not actual usage'})
    for stage in ('learning_budget_proposed','val_budget'):
        tokens+=settings[stage]['token_limit'];http+=settings[stage]['request_limit']
    if tokens>prefix['limits']['token_limit'] or http>prefix['limits']['request_limit']:
        raise ValueError('Joint allocations exceed cumulative authority')
    return {'schema':'live-campaign.allocation.v1','inherited_tokens':charged,'inherited_http':len(prefix['attempts']),
        'maximum_joint_tokens':tokens,'maximum_joint_http':http,'parallel_allocations':external,
        'limits':prefix['limits'],'created_at':utc()}


def val_config(base,root):
    cfg=deepcopy(base)
    for k in list(cfg['experiment']):
        if k.startswith('choice_') and k not in {'choice_projection_version','choice_seed'}:cfg['experiment'].pop(k)
    cfg['experiment'].update(math_program_method_version=m.VERSION,math_program_evaluation_version=VERSION,
        runtime_mode='frozen',output_dir=str(root/'val'),math_program_trial_manifest=str(root/'trial_manifest.json'))
    cfg['data_dir']=str(root/'frozen_bank')
    for section in ('learning','runtime'):cfg[section]['choice_guidance']={'enabled':False,'public_process':False}
    for k in ('capability_material_version','capability_payload_max_bytes','source_program_payload_max_bytes'):cfg['learning'].pop(k,None)
    cfg['llm']['purpose_overrides']={
        'math_program_spec':{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':4096,'max_retries':0},
        'math_program_build':{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':8192,'max_retries':0},
        'finish_only':{'protocol':{'thinking_type':'enabled'},'reasoning_effort':'low','max_completion_tokens':16384,'max_retries':0}}
    cfg['llm']['max_retries']=0;cfg['program_worker']['record_execution_receipt']=True
    return validate_config(cfg)


def prepare(source,output,config=CONTRACT):
    settings=yaml.safe_load(Path(config).read_text());root=Path(output).resolve()
    if settings!=yaml.safe_load(CONTRACT.read_text()):raise ValueError('Use the committed formal contract')
    review=read(Path(settings['revalidation'])/'pilot_review.json')
    if review['status']!='completed_train_trial' or not verified_bundles(settings['revalidation']):
        raise ValueError('Real P5-R1 generation and Docker revalidation must pass before formal preparation')
    pilot.prepare(source,root,config)
    plan=read(root/'pilot_manifest.json');prefix=read(root/'campaign_budget_prefix.json')
    basepath=Path(settings['source_p4'])/'plan.json';base=read(basepath);authority=Path(base['authority']);verify(authority)
    tasks=base['tasks']['val'];canonical=read(authority/'livemath/val.json')['tasks']
    if len(tasks)!=17 or [t['task_id'] for t in tasks]!=[t['task_id'] for t in canonical] or any(t['split']!='val' for t in tasks):
        raise ValueError('Canonical Val17 identity differs')
    if {t['physical_key'] for t in tasks}&{r['task']['physical_key'] for r in read(root/'source_catalog.json')}:
        raise ValueError('Val task overlaps Train sources')
    config=val_config(base['configs']['val'],root)
    extra={str(p):sha256(p) for p in (basepath,authority/'livemath/val.json',Path(config['harness']['evaluator_records']),
        Path(settings['revalidation'])/'pilot_review.json')}
    alloc=allocation(settings,prefix)
    write_json(root/'allocation_manifest.json',alloc);write_json(root/'val_config.json',config)
    write_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in ('inherited_tokens','inherited_http')})
    formal={'version':VERSION,'run_id':plan['run_id'],'code':plan['code'],'source_train':plan['source_train'],
        'pilot_plan_hash':digest(plan),'inputs':extra,'tasks':tasks,'settings':settings,
        'prepared_hashes':{n:sha256(root/n) for n in ('train_coverage.json','val_config.json','allocation_manifest.json')},
        'val_split_hash':sha256(authority/'livemath/val.json'),'val_policy':'readonly candidate evaluation; no automatic usable promotion',
        'initial_bank':'fresh empty; no manual Bank or P4/P5 published asset imports','test_enabled':False}
    write_json(root/'formal_plan.json',formal);write_json(root/'formal_identity.json',{'sha256':digest(formal)})
    result=check(root)
    write_json(root/'status.json',{'status':'prepared','stage':'trace_learning','at':utc(),'paid_http':0})
    return result


def check(output):
    root=Path(output);base=pilot.checked(root);plan=read(root/'formal_plan.json')
    if plan['version']!=VERSION or plan['code']!=code_identity() or digest(plan)!=read(root/'formal_identity.json')['sha256'] or plan['pilot_plan_hash']!=digest(base):
        raise ValueError('Formal trace identity changed')
    for p,h in plan['inputs'].items():
        if sha256(Path(p))!=h:raise ValueError('Formal input changed: '+p)
    for n,h in plan['prepared_hashes'].items():
        if sha256(root/n)!=h:raise ValueError('Formal prepared input changed: '+n)
    allocation(plan['settings'],read(root/'campaign_budget_prefix.json'))
    validate_config(read(root/'val_config.json'))
    result=pilot.check(root)
    result={**result,'formal_version':VERSION,'covered_train_tasks':len(read(root/'train_coverage.json')),
        'val_tasks':17,'val_enabled':True,'test_enabled':False}
    write_json(root/'formal_zero_http_check.json',result);return result


def freeze(root):
    root=Path(root);bundles=verified_bundles(root)
    target=root/'frozen_bank';manifest=root/'trial_manifest.json'
    if manifest.exists():
        value=read(manifest)
        if value['bundle_digest']!=digest(bundles) or value['tree_hash']!=tree_identity(target)['sha256']:
            raise ValueError('Frozen trial source or contents changed')
        return value
    bank=Bank(target,environment=read(root/'config.json')['program_environment'])
    try:
        for item in bundles:
            for kind,key in [('skill','atomic'),('program','program'),('implementation','implementation')]:
                bank.put(kind,item['bundle'][key])
        for kind,key in [('skill','atomic'),('program','program'),('implementation','implementation')]:
            if {p['id']:p for p in bank.all(kind)}!={r['bundle'][key]['id']:r['bundle'][key] for r in bundles}:
                raise ValueError('Unexpected asset in evaluation Bank')
        if bank.all('workflow') or bank.train_cases() or bank.jobs():raise ValueError('Unexpected evaluation Bank state')
        value={'version':VERSION,'bank_digest':bank.digest(),'program_hashes':{p['id']:digest(p) for p in bank.all('program')},
            'bundle_digest':digest(bundles),'qualification':'train_trial_only_not_usable','freeze_time':utc()}
    finally:bank.close()
    value['tree_hash']=tree_identity(target)['sha256'];write_json(manifest,value)
    return value


def campaign_snapshot(root):
    root=Path(root);plan=read(root/'formal_plan.json');own=read(root/'budget.json');merged=deepcopy(own['attempts'])
    for spec in plan['settings']['concurrent_runs']:
        for k,v in read(Path(spec['path'])/'budget.json')['attempts'].items():
            if k in merged and merged[k]!=v:raise ValueError('Conflicting physical attempt in campaign ledgers')
            merged[k]=v
    value={'as_of':utc(),'tokens_including_reservations':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in merged.values()),
        'known_metered_tokens':sum(a.get('accounted_tokens',0) for a in merged.values()),'http':len(merged),
        'source_ledgers':[str(root/'budget.json')]+[str(Path(s['path'])/'budget.json') for s in plan['settings']['concurrent_runs']],
        'unique_physical_attempts':sorted(merged),'note':'read-only union; old and parallel ledgers remain untouched'}
    write_json(root/'campaign_union_snapshot.json',value);return value


def execute(output,env_file):
    import fcntl
    root=Path(output)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gate=check(root)
        if gate['status']!='passed' or not gate['billing_ready']:raise ValueError('Formal preflight is not ready')
        if (root/'result.json').exists() and read(root/'result.json')['status']=='completed':
            freeze(root)
            return read(root/'result.json')
        load_env(env_file);plan=read(root/'formal_plan.json')
        try:
            review_path=root/'learning_completed.json'
            if review_path.exists():
                review=read(review_path)
            else:
                write_json(root/'status.json',{'status':'running','stage':'trace_learning','at':utc()})
                pilot.learn(root,approved=True)
                pilot.probe(root,approved=True)
                review=pilot.report(root)
            if review['status'] not in {'completed_train_trial','completed_no_program'} or review['billing_blocked']:
                result={'status':'stopped','stage':'trace_learning','learning':review}
                write_json(root/'result.json',result);write_json(root/'status.json',result);return result
            if not review_path.exists():write_json(review_path,review)
            frozen=freeze(root)
            write_json(root/'status.json',{'status':'running','stage':'val','at':utc(),'trial_programs':len(frozen['program_hashes'])})
            prefix=root/'val/campaign_budget_prefix.json'
            if not prefix.exists():write_json(prefix,read(root/'budget.json'))
            owner=BudgetGovernor(root/'budget.json',**plan['settings']['cumulative_limits'])
            governor=StageGovernor(owner,{'run_id':plan['run_id']+':val','new_limits':plan['settings']['val_budget']},root/'val')
            cfg=read(root/'val_config.json')
            log=FormalLog(root,{'method':VERSION,'benchmark':'livemath','run_seed':42,'split_seed':42,
                'source_commit':plan['code']['git_sha'],'source_sha256':plan['code']['source_sha256'],
                'model_name':cfg['llm']['model'],'provider_model_id':cfg['llm']['model'],
                'split_hashes':{'val':plan['val_split_hash']},'initial_artifact_hash':frozen['tree_hash']},
                {'learning':read(root/'config.json'),'val':cfg},resume=(root/'run_manifest.json').exists())
            adapter=create_simple_harness(cfg)
            try:
                summary=run(cfg,[PublicTask(**t) for t in plan['tasks']],root/'val',readonly=True,
                    resume=(root/'val/execution_manifest.json').exists(),adapter=adapter,
                    adapter_factory=lambda:create_simple_harness(cfg),formal_log=log,budget_governor=governor)
            finally:adapter.close()
            after=tree_identity(root/'frozen_bank')['sha256']
            if after!=frozen['tree_hash']:raise ValueError('Val changed the candidate Bank')
            write_json(root/'freeze_audit.json',{'before':frozen['tree_hash'],'after':after,'qualification':'trial only'})
            result={'status':'completed' if summary['complete'] else 'stopped','stage':'val','tasks':summary['tasks'],
                'successes':summary['successes'],'val_tokens':summary['total_tokens'],'trial_programs':len(frozen['program_hashes']),
                'usable_programs':0,'program_invocations':summary['program_invocations'],
                'consumed_program_results':sum(a.get('outputs_consumed',False) for t in summary['cases'] for a in t['execution'].get('attempts',[])),
                'learning_http':review['new_http'],'learning_tokens':review['new_tokens'],'test_executed':False}
            log.finish(result['status']);write_json(root/'result.json',result);write_json(root/'status.json',result)
            campaign_snapshot(root);return result
        except BaseException as exc:
            result={'status':'stopped','error_type':type(exc).__name__,'reason':str(exc),'at':utc()}
            write_json(root/'status.json',result);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--source',required=True);a.add_argument('--output',required=True)
    a.add_argument('--config',default=str(CONTRACT))
    for name in ('check','execute','status'):
        a=sub.add_parser(name);a.add_argument('--output',required=True)
        if name=='execute':a.add_argument('--env-file',required=True)
    args=vars(p.parse_args());name=args.pop('command')
    result=read(Path(args['output'])/'status.json') if name=='status' else globals()[name](**args)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
