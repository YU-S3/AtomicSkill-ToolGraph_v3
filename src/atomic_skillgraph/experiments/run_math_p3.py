"""LiveMath P3: immutable Train60 derivation, Val selection, separately reviewed Test."""
import argparse
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from ..core.errors import BudgetExhausted
from ..empirical import math_capability as math
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor,billing_blocked,reserved_recovery
from ..empirical.capability_batch import pair_fact,gate
from ..empirical.contracts import PublicTask,digest
from ..empirical.system import EmpiricalSystem,validate_config
from ..harness.benchmarks import AnswerAdapter
from ..harness.registry import create_simple_harness
from . import run_choice_capability_batch as batch
from .capability_schedule import math_schedule
from .canonical_manifest import sha256,verify
from .formal_choice_batch import BatchGovernor
from .formal_log import FormalLog,tree_identity,utc
from .run_empirical import code_identity,write_json,load_env,run
from .run_formal import resolved_config

CONTRACT=Path(__file__).parents[3]/'configs/livemath_p3_seed42.json'


def bounds():
    unit=3*(65536+4096)
    integration=2*(131072+8192)+8*(65536+4096)
    routing=(4*6+2*11)*(16384+2048)
    return {'learning_tokens':15*unit+integration,'learning_http':15*3+2+8,
        'unit_tokens':unit,'integration_tokens':integration,
        'val_tokens':routing+(4*6+2*11)*2*128000,'val_http':(4*6+2*11)*(1+2*9),
        'test_parent_tokens':100*128000,'test_http':100*(2+9)}


def prepare(source,ledger_run,datasets,output):
    source,ledger_run,data,root=map(lambda p:Path(p).resolve(),(source,ledger_run,datasets,output))
    if root.exists():raise ValueError('P3 requires a new derived directory, never an old --resume')
    code=code_identity()
    if not code['git_sha'] or code['tracked_dirty']:raise ValueError('P3 requires clean committed source')
    settings=batch.read(CONTRACT);original=batch.read(source/'train/config.json')
    old=batch.read(source/'run_manifest.json');summary=batch.read(source/'train/summary.json')
    execution=batch.read(source/'train/execution_manifest.json')
    if (not summary['complete'] or summary['tasks']!=60 or old['run_seed']!=42 or old['benchmark']!='livemath' or
        execution['config']!=original or execution['tasks']!=[t['task'] for t in summary['cases']]):
        raise ValueError('P3 requires complete, identically recorded LiveMath seed42 Train60')
    bank=Bank(source/'train/bank',readonly=True,environment=original['program_environment'])
    try:
        if bank.digest()!=summary['knowledge_digest']:raise ValueError('Historical Train Bank changed')
        cases=bank.train_cases()
    finally:bank.close()
    if [r['task'] for r in cases]!=execution['tasks']:raise ValueError('Historical Train cases differ from actual Runtime traces')
    records=math.records(cases,span_factory=math.math_spans)
    authority=Path(original['manifest']).resolve().parent.parent;verify(authority)
    material=batch.read(data/'materialization.json')
    if material['authority_sha256']!=sha256(authority/'manifest.json'):raise ValueError('Canonical authority changed')
    tasks={phase:batch.read(data/'livemath'/(phase+'.json'))['tasks'] for phase in ('train','val','test')}
    for phase,count in [('train',60),('val',17),('test',100)]:
        canonical=batch.read(authority/'livemath'/(phase+'.json'))['tasks']
        if len(tasks[phase])!=count or [r['task_id'] for r in tasks[phase]]!=[r['task_id'] for r in canonical]:
            raise ValueError('P3 canonical membership/order differs: '+phase)
        if any(r['split']!=phase for r in tasks[phase]):raise ValueError('Incorrect physical split')
    keys=[{t['physical_key'] for t in tasks[p]} for p in ('train','val','test')]
    if any(keys[i]&keys[j] for i in range(3) for j in range(i)):raise ValueError('Learning/evaluation overlap')
    if {t['physical_key'] for t in execution['tasks']}!=keys[0]:raise ValueError('Train60 source coverage mismatch')
    ledger=batch.read(ledger_run/'budget.json');prefix=deepcopy(ledger)
    reserved_recovery(prefix)
    if billing_blocked(prefix):raise ValueError('Unresolved billing without inherited reservation authority')
    prefix['limits']=settings['cumulative_limits']
    charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in prefix['attempts'].values())
    if charged+sum(v['token_limit'] for v in settings['stage_limits'].values())>prefix['limits']['token_limit']:
        raise ValueError('Stages exceed remaining cumulative tokens')
    if len(prefix['attempts'])+sum(v['request_limit'] for v in settings['stage_limits'].values())>prefix['limits']['request_limit']:
        raise ValueError('Stages exceed remaining cumulative HTTP')
    cfg=deepcopy(original);cfg['experiment'].update(choice_method_version=math.VERSION,output_dir=str(root/'learning'),runtime_mode='online')
    for key in ('choice_batch_contract','choice_batch_contract_sha256','choice_batch_version'):cfg['experiment'].pop(key,None)
    cfg['data_dir']=str(root/'learning/bank');cfg=validate_config(math.configure(cfg))
    profile=batch.read(Path(__file__).parents[3]/'benchmark_profiles.json')['profiles']['livemath']
    configs={phase:resolved_config(cfg,profile,'livemath',42,phase,root,data,authority,None) for phase in ('val','test')}
    for c in configs.values():
        c['data_dir']=str(root/'frozen_bank');c['llm']['max_retries']=0
    inputs={str(source/name):sha256(source/name) for name in ('run_manifest.json','train/config.json',
        'train/summary.json','train/execution_manifest.json','train/bank/bank.sqlite3')}
    inputs[str(ledger_run/'budget.json')]=sha256(ledger_run/'budget.json');inputs[str(CONTRACT)]=sha256(CONTRACT)
    for path in (source/'train/traces').glob('*.json'):inputs[str(path)]=sha256(path)
    for name,h in material['files_sha256'].items():
        if name.startswith('livemath/'):
            if sha256(data/name)!=h:raise ValueError('Materialized LiveMath resource changed')
            inputs[str(data/name)]=h
    for path in [authority/'manifest.json',data/'materialization.json',*[authority/'livemath'/(p+'.json') for p in tasks]]:
        inputs[str(path)]=sha256(path)
    run_id='p3_'+uuid4().hex
    initial=Bank(cfg['data_dir'],environment=cfg['program_environment']);initial.bind_choice_method(math.VERSION)
    try:initial_digest=initial.digest()
    finally:initial.close()
    initial_hash=tree_identity(Path(cfg['data_dir']))['sha256']
    identity={'method':'ours','benchmark':'livemath','run_seed':42,'split_seed':42,'provider_model_id':cfg['llm']['model'],
        'model_name':cfg['llm']['model'],'source_commit':code['git_sha'],'source_sha256':code['source_sha256'],
        'choice_method_version':math.VERSION,'source_train_run_id':old['run_id'],'source_train_commit':old['source_commit'],
        'split_hashes':{p:sha256(authority/'livemath'/(p+'.json')) for p in tasks},
        'initial_bank':'fresh empty P3 Bank; only original Train cases imported; no P2 assets or utility',
        'initial_bank_digest':initial_digest,'initial_artifact_hash':initial_hash,
        'learning_schedule':'Train60 replay -> 15 units -> consolidation -> Val6 -> Val11 -> Frozen -> reviewed Test100',
        'test_development_history_disclosed':True,'choice_projection_version':cfg['experiment'].get('choice_projection_version'),
        'choice_seed':cfg['experiment'].get('choice_seed'),'source_train_runtime_reexecuted':False}
    plan={'schema':math.VERSION,'run_id':run_id,'code':code,'prepared_at':utc(),'settings':settings,'identity':identity,
        'source_train':str(source),'ledger_source':str(ledger_run),'datasets':str(data),'authority':str(authority),
        'schedule':math_schedule(execution['tasks']),'tasks':tasks,'input_hashes':inputs,'configs':configs,
        'bounds':bounds(),'inherited_tokens':charged,'inherited_http':len(prefix['attempts']),
        'inherited_reservations':sorted(reserved_recovery(prefix)),'new_model_http':0}
    for name,value in [('config.json',cfg),('source_records.json',records),('source_cases.json',cases),
        ('campaign_budget_prefix.json',prefix),('budget.json',prefix)]:write_json(root/name,value)
    for stage in settings['stage_limits']:write_json(root/stage/'campaign_budget_prefix.json',prefix)
    plan['prepared_hashes']={n:sha256(root/n) for n in ('config.json','source_records.json','source_cases.json','campaign_budget_prefix.json')}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'plan_hash':digest(plan)})
    return check(root)


def check(output):
    root=Path(output);plan=batch.read(root/'plan.json')
    if plan['schema']!=math.VERSION or plan['code']!=code_identity() or batch.read(root/'plan_identity.json')['plan_hash']!=digest(plan):
        raise ValueError('P3 code or prepared plan identity changed')
    if plan['settings']!=batch.read(CONTRACT):raise ValueError('Frozen P3 method contract changed')
    if plan['settings']['selection_policy']!=math.POLICY:raise ValueError('P3 numerical gates changed')
    for name,h in plan['input_hashes'].items():
        if sha256(Path(name))!=h:raise ValueError('Historical source changed: '+name)
    for name,h in plan['prepared_hashes'].items():
        if sha256(root/name)!=h:raise ValueError('Prepared P3 input changed: '+name)
    cfg=validate_config(batch.read(root/'config.json'))
    if Path(cfg['experiment']['output_dir']).resolve()!=(root/'learning').resolve():raise ValueError('P3 output identity changed')
    if plan['schedule']!=math_schedule([r['task'] for r in batch.read(root/'source_cases.json')]):raise ValueError('P3 source schedule changed')
    ledger=batch.read(root/'budget.json');prefix=batch.read(root/'campaign_budget_prefix.json')
    if ledger['limits']!=plan['settings']['cumulative_limits'] or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('Inherited physical charges or campaign authority changed')
    if ledger.get('inherited_reservation_recovery')!=prefix.get('inherited_reservation_recovery'):raise ValueError('Reservation authority changed')
    for stage in plan['settings']['stage_limits']:
        if batch.read(root/stage/'campaign_budget_prefix.json')!=prefix:raise ValueError('Stage cannot reset its inherited prefix')
    if plan.get('continuation'):
        from .math_continuation import validate
        validate(root,plan)
    elif not (root/'learning/started.json').exists() and tree_identity(Path(cfg['data_dir']))['sha256']!=plan['identity']['initial_artifact_hash']:
        raise ValueError('Prepared empty Bank changed before P3 learning')
    for phase,key in [('learning','learning'),('val','val')]:
        if plan['bounds'][key+'_tokens']>plan['settings']['stage_limits'][phase]['token_limit'] or plan['bounds'][key+'_http']>plan['settings']['stage_limits'][phase]['request_limit']:
            raise ValueError('Frozen protocol bound exceeds its stage')
    if (root/'final_frozen_manifest.json').exists():
        frozen=batch.read(root/'final_frozen_manifest.json')
        if tree_identity(root/'frozen_bank')['sha256']!=frozen['final_artifact_hash']:raise ValueError('Published P3 Bank changed')
        if frozen['selection_result_hash']!=digest(batch.read(root/'val/result.json')):raise ValueError('Frozen Val evidence changed')
    return {'status':'passed','method_version':math.VERSION,'new_model_http':0,'billing_ready':not billing_blocked(ledger),
        'bounds':plan['bounds'],'test_requires_review':True}


class StageGovernor(BatchGovernor):
    def owns(self,context):
        return super().owns(context) or context.get('parent_scope','').startswith(str(self.root.resolve())+':')


def stage_governor(owner,root,plan,stage):
    return StageGovernor(owner,{'run_id':plan['run_id']+':'+stage,'new_limits':plan['settings']['stage_limits'][stage]},root/stage)


def ranked_candidates(candidates):
    # Round-robin categories, distinct declared operations, no use of Val outcomes.
    pending=sorted(candidates,key=lambda c:(-len(c['source_case_ids']),c['id']));result=[];seen=set()
    while pending:
        categories=set();remaining=[]
        for c in pending:
            if c['operation'] in seen:continue
            if c['category'] in categories:remaining.append(c);continue
            result.append(c);seen.add(c['operation']);categories.add(c['category'])
        pending=remaining
    return result


def outcome_rank(receipt,candidate_id):
    facts=receipt['paired']
    return (-sum(f['on']['correct']-f['off']['correct'] for f in facts),
        sum(f['off']['correct'] and not f['on']['correct'] for f in facts),
        sum(f[a]['usage']['total_tokens'] for f in facts for a in ('off','on')),candidate_id)


def learning(root,plan,cfg,governor,observer):
    if plan.get('continuation'):
        value=batch.read(root/'inherited/learning/result.json')
        if value['plan_hash']!=plan['continuation']['parent_plan_hash'] or any(not math.checked_candidate(c) for c in value['candidates']):
            raise ValueError('Inherited completed learning identity changed')
        return value
    path=root/'learning/result.json'
    if path.exists():
        value=batch.read(path)
        if value['plan_hash']!=digest(plan) or any(not math.checked_candidate(c) for c in value['candidates']):
            raise ValueError('Completed P3 learning identity changed')
        if value['candidate_hashes']!={c['id']:digest(c) for c in value['candidates']}:raise ValueError('Frozen candidate content changed')
        return value
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}),budget_governor=governor);s.observer=observer
    try:
        if not (root/'learning/started.json').exists():
            if s.bank.digest()!=plan['identity']['initial_bank_digest'] or s.bank.all('skill') or s.bank.all('program'):
                raise ValueError('P3 initial Bank is not the declared empty Bank')
            write_json(root/'learning/started.json',{'at':utc(),'plan_hash':digest(plan),'initial_bank_digest':s.bank.digest()})
        for row in batch.read(root/'source_cases.json'):s.bank.save_case(PublicTask(**row['task']),row['experience'])
        rows=batch.read(root/'source_records.json');by_key={r['task']['physical_key']:r for r in rows};candidates=[];unit_logs=[]
        for i,unit in enumerate(plan['schedule']['units']):
            folder=root/'learning/units'/str(i);batch.progress(root,'learning',unit=i,total=15)
            if observer:observer.begin_task(PublicTask(plan['run_id'],plan['run_id'],'P3 Train-only trace learning'),i,
                plan['run_id']+':learning:'+str(i),{'source_physical_keys':unit['source_keys']})
            batch.learning_boundary(s,folder,plan['run_id']+':learning:'+str(i),
                unit_limits={'token_limit':bounds()['unit_tokens'],'request_limit':3})
            started=utc();log=math.learn(s.learner,[by_key[k] for k in unit['source_keys']])
            write_json(folder/'learning.json',log);candidates.extend(log['candidates']);unit_logs.append(log)
            if observer:observer.requests(s.requests);observer.training(str(i),'p3_train_trace_unit',started,utc(),4,log['decision'],consumed_physical_keys=unit['source_keys'])
        batch.progress(root,'consolidation',natural_checked=len(candidates))
        consolidated=[]
        if candidates:
            if observer:observer.begin_task(PublicTask(plan['run_id'],plan['run_id'],'P3 Train-only consolidation'),15,
                plan['run_id']+':learning:consolidation',{'source_physical_keys':list(by_key)})
            folder=root/'learning/consolidation';batch.learning_boundary(s,folder,plan['run_id']+':learning:consolidation',
                unit_limits={'token_limit':bounds()['integration_tokens'],'request_limit':10})
            parents={c['id']:c for c in candidates};log=math.learn(s.learner,rows,parents=parents)
            write_json(folder/'learning.json',log);consolidated=ranked_candidates(log['candidates'])
            if observer:observer.requests(s.requests)
        order=consolidated[:plan['settings']['max_screen_candidates']]
        result={'method_version':math.VERSION,'status':'completed','natural_checked':len(candidates),
            'natural_proposed':sum(len(r.get('proposed_candidates',[])) for r in unit_logs),
            'unit_decisions':[r['decision'] for r in unit_logs],
            'consolidated':len(consolidated),'candidates':consolidated,'screen_ids':[c['id'] for c in order],
            'candidate_hashes':{c['id']:digest(c) for c in consolidated},'plan_hash':digest(plan),'completed_at':utc()}
        write_json(path,result);write_json(root/'candidate_order.json',{'ids':[c['id'] for c in consolidated],
            'screen_ids':result['screen_ids'],'candidate_hashes':result['candidate_hashes'],'frozen_before_val':True})
        return result
    finally:s.close()


def selection(root,plan,cfg,learned,governor,observer):
    result_path=root/'val/result.json'
    if result_path.exists():return batch.read(result_path)
    trial=root/'val/trial_bank'
    if not trial.exists():
        bank=Bank(root/'val/empty_bank');bank.bind_choice_method(math.VERSION)
        try:bank.freeze(trial)
        finally:bank.close()
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}),budget_governor=governor);s.observer=observer
    evaluator=batch.read(plan['configs']['val']['harness']['evaluator_records'])
    policy=plan['settings']['selection_policy'];receipts={};candidates={c['id']:c for c in learned['candidates']}
    def evaluate(candidate,stop):
        folder=root/'val/candidates'/candidate['id'];preflight=[];pairs={}
        for rank,t in enumerate(plan['tasks']['val'][:stop]):
            task=PublicTask(**t);batch.progress(root,'val_selection',candidate=candidate['id'],rank=rank,total=stop)
            if observer:observer.begin_task(task,rank,plan['run_id']+':val:routing:'+candidate['id']+':'+str(rank),
                {'purpose':'public_val_applicability','candidate_id':candidate['id']})
            inherited=plan.get('continuation',{}).get('routes',{}).get(candidate['id'],{}).get(str(rank))
            if inherited:
                route=inherited['match'];math.validate_route(route,candidate,task)
                if observer:observer.requests(batch.read(inherited['requests'])['requests'])
            else:
                route_folder=folder/'routing'/str(rank)
                batch.learning_boundary(s,route_folder,plan['run_id']+':val:route:'+candidate['id']+':'+str(rank),
                    unit_limits={'token_limit':18432,'request_limit':1})
                route=math.match(s.learner,candidate,task)
            preflight.append({'task':t,'rank':rank,'match':route})
            write_json(folder/'preflight.json',preflight)
            if observer and not inherited:observer.requests(s.requests)
            if route['applicable']:
                pairs[task.physical_key]=batch.evaluate_pair(folder/'selection',task,candidate,
                    {candidate['capability']['id']:route},plan['configs']['val'],governor,trial,evaluator,
                    rank,list(candidates).index(candidate['id']),run_id=plan['run_id']+':val',observer=observer,
                    view_factory=math.MathView,method_version=math.VERSION,
                    inherited_arms=plan.get('continuation',{}).get('arms',{}).get(candidate['id']))
        receipt=math.utility_receipt(candidate,pairs,preflight,policy,{'plan_hash':digest(plan),
            'val_manifest_sha256':plan['identity']['split_hashes']['val'],'public_val_hash':digest(plan['tasks']['val']),
            'method_version':math.VERSION,'source_commit':plan['code']['git_sha'],
            **({'continuation_version':plan['continuation']['version'],
                'inherited_evidence_commit':plan['continuation']['parent_code']['git_sha'],
                'parent_plan_hash':plan['continuation']['parent_plan_hash']} if plan.get('continuation') else {})})
        write_json(folder/('screen_receipt.json' if stop==6 else 'utility_receipt.json'),receipt)
        return receipt
    try:
        for key in learned['screen_ids']:receipts[key]=evaluate(candidates[key],6)
        advancing=sorted([key for key,r in receipts.items() if r['screen_passed']],key=lambda k:outcome_rank(receipts[k],k))[:2]
        selection_order={'candidate_hashes':learned['candidate_hashes'],'confirmation_ids':advancing,
            'screen_receipt_hashes':{k:r['receipt_hash'] for k,r in receipts.items()}}
        order_path=root/'val/confirmation_order.json'
        if order_path.exists() and batch.read(order_path)!=selection_order:raise ValueError('Frozen confirmation order changed')
        write_json(order_path,selection_order)
        for key in advancing:receipts[key]=evaluate(candidates[key],17)
        passed=sorted([key for key,r in receipts.items() if r['passed']],key=lambda k:outcome_rank(receipts[k],k))
        result={'status':'completed_selection' if learned['screen_ids'] else 'completed_no_candidate','passed_ids':passed,
            'receipts':receipts,'qualification':'Val development selection, not independent generalization','completed_at':utc()}
        write_json(result_path,result);return result
    finally:s.close()


def publish(root,plan,learned,selected):
    manifest=root/'final_frozen_manifest.json'
    if manifest.exists():return batch.read(manifest)
    bank=Bank(root/'publication_bank');bank.bind_choice_method(math.VERSION)
    try:
        by_id={c['id']:c for c in learned['candidates']}
        for rank,key in enumerate(selected['passed_ids']):
            c=by_id[key];asset={**c,'state':'usable','candidate':c,'utility_receipt':selected['receipts'][key],'publication_rank':rank}
            if not math.checked_asset(asset):raise ValueError('P3 publication receipt failed validation')
            bank.put('skill',asset)
        if (root/'frozen_bank').exists():
            existing=Bank(root/'frozen_bank',readonly=True)
            try:
                if existing.all('skill')!=bank.all('skill') or existing.all('program'):raise ValueError('Uncommitted Frozen differs from selection')
                frozen=batch.read(root/'frozen_bank/freeze.json')
            finally:existing.close()
        else:frozen=bank.freeze(root/'frozen_bank')
    finally:bank.close()
    result={'method_version':math.VERSION,'freeze_time':utc(),'plan_hash':digest(plan),
        'candidate_hashes':learned['candidate_hashes'],'selection_result_hash':digest(selected),
        'usable_skill_ids':selected['passed_ids'],'final_artifact_hash':tree_identity(root/'frozen_bank')['sha256'],'freeze':frozen}
    write_json(manifest,result);return result


def execute(output,env_file=None,*,test=False,reviewed=False):
    import fcntl
    with (Path(output)/'execution.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This P3 run already has an active executor') from None
        return _execute(output,env_file,test=test,reviewed=reviewed)


def _execute(output,env_file=None,*,test=False,reviewed=False):
    root=Path(output).resolve();check(root);plan=batch.read(root/'plan.json');cfg=batch.read(root/'config.json')
    if env_file:load_env(env_file)
    owner=BudgetGovernor(root/'budget.json',**plan['settings']['cumulative_limits'])
    if owner.billing_blocked:raise BudgetExhausted('unknown_billing','P3 billing must be reconciled before another request')
    observer=FormalLog(root,plan['identity'],{'learning':cfg,**plan['configs']},resume=(root/'run_manifest.json').exists())
    observer.manifest['initial_artifact_hash']=plan['identity']['initial_artifact_hash']
    write_json(observer.manifest_path,observer.manifest)
    try:
        if test:
            if not reviewed or not (root/'final_frozen_manifest.json').exists() or not (root/'val/result.json').exists():
                raise ValueError('Test requires completed P3 Val, Frozen and explicit review')
            before=tree_identity(root/'frozen_bank')['sha256'];tcfg=plan['configs']['test']
            write_json(root/'test_review.json',{'at':utc(),'plan_hash':digest(plan),'frozen_hash':before,
                'val_result_hash':sha256(root/'val/result.json'),'explicit_review':True,'development_history_disclosed':True})
            adapter=create_simple_harness(tcfg)
            try:
                summary=run(tcfg,[PublicTask(**t) for t in plan['tasks']['test']],root/'test',readonly=True,
                    resume=(root/'test/execution_manifest.json').exists(),adapter=adapter,adapter_factory=lambda:create_simple_harness(tcfg),
                    formal_log=observer,budget_governor=stage_governor(owner,root,plan,'test'),order_offset=77)
            finally:adapter.close()
            after=tree_identity(root/'frozen_bank')['sha256']
            if before!=after:raise ValueError('Readonly Test changed Frozen')
            write_json(root/'test_freeze_audit.json',{'hash_before_test':before,'hash_after_test':after})
            result={'status':'completed' if summary['complete'] else 'test_incomplete','test_summary':str(root/'test/summary.json')}
        else:
            learned=learning(root,plan,cfg,stage_governor(owner,root,plan,'learning'),observer)
            selected=selection(root,plan,cfg,learned,stage_governor(owner,root,plan,'val'),observer)
            frozen=publish(root,plan,learned,selected)
            result={'status':'awaiting_test_review','usable_skills':len(frozen['usable_skill_ids']),
                'learning':{k:learned[k] for k in ('natural_proposed','natural_checked','consolidated','screen_ids')},
                'val_passed_ids':selected['passed_ids'],'test_started':False}
        observer.finish(result['status']);write_json(root/'result.json',result);return result
    except BaseException as exc:
        status='budget_stopped' if isinstance(exc,BudgetExhausted) else 'execution_failed'
        observer.finish(status);write_json(root/'result.json',{'status':status,'error_type':type(exc).__name__,'reason':str(exc),'at':utc()})
        raise


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare')
    for key in ('source','ledger-run','datasets','output'):p.add_argument('--'+key,required=True)
    for stage in ('check','status','execute','test'):
        p=sub.add_parser(stage);p.add_argument('--output',required=True)
        if stage not in {'check','status'}:p.add_argument('--env-file',required=True)
        if stage=='test':p.add_argument('--reviewed-p3-val',action='store_true')
    p=sub.add_parser('continue-schema')
    p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    p.add_argument('--env-file')
    args=parser.parse_args()
    if args.stage=='continue-schema':
        from .math_continuation import prepare as continue_schema
        if args.env_file:load_env(args.env_file)
        print(json.dumps(continue_schema(args.parent,args.output),ensure_ascii=False));return
    result=prepare(args.source,args.ledger_run,args.datasets,args.output) if args.stage=='prepare' else status(args.output) if args.stage=='status' else check(args.output) if args.stage=='check' else execute(
        args.output,args.env_file,test=args.stage=='test',reviewed=getattr(args,'reviewed_p3_val',False))
    print(json.dumps(result,ensure_ascii=False))


def status(output):
    root=Path(output);result={name:batch.read(root/name) for name in ('progress.json','result.json') if (root/name).exists()}
    result['budget']={}
    for stage in ('learning','val','test','campaign'):
        path=root/'budget.json' if stage=='campaign' else root/stage/'budget.json'
        if not path.exists():continue
        ledger=batch.read(path);attempts=ledger['attempts'].values()
        result['budget'][stage]={'http':len(ledger['attempts']),
            'known_tokens':sum(a.get('accounted_tokens',0) for a in attempts),
            'charged_with_reservations':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in attempts),
            'unknown_billing':ledger['unknown_billing'],'billing_blocked':billing_blocked(ledger),'limits':ledger['limits']}
    return result


if __name__=='__main__':main()
