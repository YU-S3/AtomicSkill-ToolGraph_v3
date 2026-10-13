"""Explicit P3 schema correction after completed learning; no repeated paid evidence."""
from copy import deepcopy
from pathlib import Path
import fcntl
import shutil
from uuid import uuid4

from ..empirical import math_capability as math
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor,billing_blocked
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask,digest
from ..empirical.system import EmpiricalSystem,validate_config
from ..harness.benchmarks import AnswerAdapter
from . import run_choice_capability_batch as batch
from .canonical_manifest import sha256
from .formal_log import tree_identity,utc
from .run_empirical import write_json

VERSION='livemath.p3.schema-continuation.v1'


class ReplayGovernor:
    """Keep the original checkpoint identity; all physical admission is prohibited."""
    def __init__(self,owner,original_path):
        self.owner,self.path,self.limits=owner,Path(original_path),owner.limits

    def admit(self,*args,**kwargs):
        raise ValueError('Offline continuation needs an unsaved response; paid replay is forbidden')

    def complete(self,*args,**kwargs):
        raise ValueError('Offline continuation must not alter physical billing')

    def __getattr__(self,name):return getattr(self.owner,name)


def arm_config(base,trial,folder):
    cfg=deepcopy(base);cfg['data_dir']=str(trial);cfg['llm']['max_retries']=0
    cfg['experiment'].update(runtime_mode='frozen',output_dir=str(folder))
    return validate_config(cfg)


def recover_terminal(parent,root,old,candidate,task,rank,index,arm,match,owner):
    """Finish only a saved terminal response in a copy, with the original scorer."""
    relative=Path('val/candidates')/candidate['id']/'selection'/str(rank)/arm
    source=parent/relative;target=root/'inherited/recovered'/candidate['id']/str(rank)/arm
    shutil.copytree(source,target)
    cfg=arm_config(old['configs']['val'],parent/'val/trial_bank',source)
    checkpoint=TaskCheckpoint(target/'checkpoint',response_namespace=source/'checkpoint')
    responses=[batch.read(p) for p in (target/'checkpoint/responses').glob('*.json')]
    if not responses or not any(r['turn'].get('finish_reason')=='stop' and r['turn'].get('content') and
                               not r['turn'].get('tool_calls') for r in responses):
        raise ValueError('Partial P3 arm lacks a saved terminal response; requires a separate recovery')
    evaluator=batch.read(old['configs']['val']['harness']['evaluator_records'])
    system=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',evaluator),
        budget_governor=ReplayGovernor(owner,parent/'budget.json'),
        bank_view_factory=lambda b:math.MathView(b,{candidate['capability']['id']:match},off=arm=='off',candidate=candidate))
    before=deepcopy(owner.state)
    try:
        system.checkpoint=checkpoint;system.audit_path=str(target/'requests.json')
        attempt=old['run_id']+':val:'+math.VERSION+':'+str(index)+':'+str(rank)+':'+arm
        trace=system.run_task(task,learn=False,attempt_id=attempt)
        if trace.get('error') or trace['execution'].get('budget_censored') or owner.state!=before:
            raise ValueError('Offline terminal recovery did not complete without changing billing')
        write_json(target/'trace.json',trace)
        checkpoint.advance(checkpoint.state['stage'],paired_trace_hash=digest(trace))
        from . import run_math_p3 as p3
        write_json(target/'recovery_receipt.json',{'version':VERSION,'new_model_http':0,'at':utc(),
            'response_source_commit':old['code']['git_sha'],'recovery_code':p3.code_identity(),
            'source_checkpoint_sha256':sha256(source/'checkpoint/state.json'),'source':str(source),
            'original_method_identity':trace['method_identity'],'trace_hash':digest(trace),
            'response_hashes':{p.name:sha256(p) for p in (source/'checkpoint/responses').glob('*.json')}})
        return target/'trace.json'
    finally:system.close()


def prepare(parent,output):
    parent,root=Path(parent).resolve(),Path(output).resolve()
    if root.exists():raise ValueError('Schema continuation requires an independent new directory')
    with (parent/'execution.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Stop the parent executor before preparing continuation') from None
        return _prepare(parent,root)


def _prepare(parent,root):
    from . import run_math_p3 as p3
    code=p3.code_identity();old=batch.read(parent/'plan.json');settings=batch.read(p3.CONTRACT)
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Continuation requires committed clean source')
    if old['schema']!=math.VERSION or old.get('continuation') or digest(old)!=batch.read(parent/'plan_identity.json')['plan_hash']:
        raise ValueError('Original P3 plan identity changed')
    expected=deepcopy(old['settings']);expected['proposal_schema_revision']='phase_specific_parents.v2'
    if settings!=expected:raise ValueError('Only the declared proposal schema correction may change the contract')
    if batch.read(parent/'stop_receipt.json')['status']!='stopped_for_explicit_schema_continuation':
        raise ValueError('Missing explicit schema-continuation stop receipt')
    if any((parent/name).exists() for name in ('final_frozen_manifest.json','val/result.json','test/execution_manifest.json')):
        raise ValueError('Schema continuation requires unfinished, unpublished Val')
    for name,h in {**old['input_hashes'],**{str(parent/n):v for n,v in old['prepared_hashes'].items()}}.items():
        if sha256(Path(name))!=h:raise ValueError('Parent input changed: '+name)
    learned=batch.read(parent/'learning/result.json')
    if (learned['status']!='completed' or learned['plan_hash']!=digest(old) or len(learned['unit_decisions'])!=15 or
        learned['candidate_hashes']!={c['id']:digest(c) for c in learned['candidates']} or
        any(not math.checked_candidate(c) for c in learned['candidates'])):
        raise ValueError('Completed P3 learning or candidate identity changed')
    ledger=batch.read(parent/'budget.json');prefix=batch.read(parent/'campaign_budget_prefix.json')
    added={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    if (billing_blocked(ledger) or ledger['limits']!=settings['cumulative_limits'] or
        any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()) or
        any(v['status']!='metered' for v in added.values())):
        raise ValueError('P3 continuation requires the unchanged, fully metered paid prefix')
    phases={s:batch.read(parent/s/'budget.json')['attempts'] if (parent/s/'budget.json').exists() else {}
            for s in settings['stage_limits']}
    if (sum(map(len,phases.values()))!=len(added) or
        {k:v for phase in phases.values() for k,v in phase.items()}!=added):
        raise ValueError('Stage charges do not partition the physical campaign prefix')
    audited={}
    for path in parent.rglob('responses/*.json'):
        for attempt in batch.read(path).get('http_attempts',[]):
            key=attempt['request_id']
            if key not in added or added[key]['actual_usage']!=attempt['raw_usage']:
                raise ValueError('Saved physical response differs from its actual usage')
            if key in audited and audited[key]!=attempt:raise ValueError('Conflicting physical response audit')
            audited[key]=attempt
    if set(audited)!=set(added):raise ValueError('Missing durable paid response at continuation boundary')
    inputs={**old['input_hashes'],**{str(p):sha256(p) for p in parent.rglob('*') if p.is_file()}}
    for phase in ('learning','val'):shutil.copytree(parent/phase,root/'inherited'/phase)
    shutil.copytree(parent/'learning/bank',root/'learning/bank')
    for name in ('source_records.json','source_cases.json','candidate_order.json'):shutil.copy2(parent/name,root/name)
    for name in ('campaign_budget_prefix.json','budget.json'):write_json(root/name,ledger)
    for stage in settings['stage_limits']:
        write_json(root/stage/'campaign_budget_prefix.json',ledger)
        write_json(root/stage/'inherited_stage_attempts.json',phases[stage])
    owner=BudgetGovernor(root/'budget.json',**settings['cumulative_limits'])
    cfg=batch.read(parent/'config.json');cfg['data_dir']=str(root/'learning/bank');cfg['experiment']['output_dir']=str(root/'learning')
    configs=deepcopy(old['configs'])
    for value in configs.values():
        value['data_dir']=str(root/'frozen_bank')
        value['experiment']['output_dir']=str(root/Path(value['experiment']['output_dir']).relative_to(parent))
    imported={'version':VERSION,'at':utc(),'parent_root':str(parent),'parent_code':old['code'],
        'parent_plan_hash':digest(old),'parent_run_id':old['run_id'],'completed_learning_repeated':False,
        'candidate_hashes':learned['candidate_hashes'],'routes':{},'arms':{},
        'prior_stage_cost':{s:{'tokens':sum(v['accounted_tokens'] for v in values.values()),'http':len(values)} for s,values in phases.items()},
        'reason':'Correct unit/consolidation parent_ids schema; preserve all completed learning and rejection outcomes'}
    for index,candidate in enumerate(learned['candidates']):
        folder=parent/'val/candidates'/candidate['id'];routes={};arms={}
        for row in batch.read(folder/'preflight.json') if (folder/'preflight.json').exists() else []:
            rank=row['rank'];task=PublicTask(**old['tasks']['val'][rank])
            if row['task']!=task.__dict__:raise ValueError('Inherited public routing task changed')
            math.validate_route(row['match'],candidate,task)
            routes[str(rank)]={'match':row['match'],'requests':str(root/'inherited/val/candidates'/candidate['id']/'routing'/str(rank)/'requests.json')}
        for state_path in folder.glob('selection/*/*/checkpoint/state.json'):
            source=state_path.parent.parent;rank=int(source.parent.name);arm=source.name
            if arm not in {'off','on'} or not routes.get(str(rank),{}).get('match',{}).get('applicable'):
                raise ValueError('Inherited pair lacks its applicable public routing evidence')
            task=PublicTask(**old['tasks']['val'][rank]);trace_path=source/'trace.json'
            if trace_path.exists():
                if batch.read(state_path).get('paired_trace_hash')!=digest(batch.read(trace_path)):
                    raise ValueError('Parent paired trace is not committed')
                trace_path=root/'inherited'/trace_path.relative_to(parent)
            else:
                trace_path=recover_terminal(parent,root,old,candidate,task,rank,index,arm,routes[str(rank)]['match'],owner)
            trace=batch.read(trace_path);original_cfg=arm_config(old['configs']['val'],parent/'val/trial_bank',source)
            if trace['task']!=task.__dict__ or trace['method_identity']['config_hash']!=digest(original_cfg):
                raise ValueError('Inherited Runtime identity changed')
            from ..empirical.capability_batch import arm_fact
            arm_fact(trace,task,candidate,arm)
            arms[str(rank)+'/'+arm]={'path':str(trace_path),'trace_hash':digest(trace),'config_hash':digest(original_cfg),
                'source':str(source),'source_commit':old['code']['git_sha']}
        imported['routes'][candidate['id']]=routes;imported['arms'][candidate['id']]=arms
    covered={}
    for cid,rows in imported['routes'].items():
        for row in rows.values():
            for request in batch.read(row['requests'])['requests']:
                for attempt in request.get('http_attempts',[]):covered[attempt['request_id']]=attempt
    for rows in imported['arms'].values():
        for row in rows.values():
            for request in batch.read(row['path'])['requests']:
                for attempt in request.get('http_attempts',[]):covered[attempt['request_id']]=attempt
    if set(covered)!=set(phases['val']):raise ValueError('Not all paid Val evidence has an explicit continuation mapping')
    charged=sum(v.get('accounted_tokens',v['reserved_tokens']) for v in ledger['attempts'].values())
    bank=Bank(root/'learning/bank',readonly=True,environment=cfg['program_environment'])
    try:initial_digest=bank.digest()
    finally:bank.close()
    identity={**old['identity'],'source_commit':code['git_sha'],'source_sha256':code['source_sha256'],
        'continuation_version':VERSION,'parent_run_id':old['run_id'],'parent_run':str(parent),
        'initial_bank_digest':initial_digest,'initial_artifact_hash':tree_identity(root/'learning/bank')['sha256'],
        'parent_initial_artifact_hash':old['identity']['initial_artifact_hash'],
        'candidate_generation_source_commit':old['code']['git_sha'],'initial_bank':'inherited completed P3 learning; not a fresh training run'}
    plan={**old,'run_id':'p3_'+uuid4().hex,'code':code,'prepared_at':utc(),'settings':settings,'identity':identity,
        'configs':configs,'input_hashes':inputs,'continuation':imported,'inherited_tokens':charged,'inherited_http':len(ledger['attempts'])}
    write_json(root/'config.json',cfg);write_json(root/'continuation_manifest.json',imported)
    immutable=[root/n for n in ('config.json','source_records.json','source_cases.json','campaign_budget_prefix.json','candidate_order.json','continuation_manifest.json')]
    immutable.extend(p for p in (root/'inherited').rglob('*') if p.is_file())
    immutable.extend(root/s/'inherited_stage_attempts.json' for s in settings['stage_limits'])
    plan['prepared_hashes']={str(p.relative_to(root)):sha256(p) for p in immutable}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'plan_hash':digest(plan)})
    return p3.check(root)


def validate(root,plan):
    imported=plan['continuation']
    if imported['version']!=VERSION or imported!=batch.read(root/'continuation_manifest.json'):
        raise ValueError('P3 continuation provenance changed')
    original=batch.read(Path(imported['parent_root'])/'plan.json')
    if digest(original)!=imported['parent_plan_hash'] or original['code']!=imported['parent_code']:
        raise ValueError('P3 parent code identity changed')
    for stage,spent in imported['prior_stage_cost'].items():
        attempts=batch.read(root/stage/'inherited_stage_attempts.json')
        if spent!={'tokens':sum(a['accounted_tokens'] for a in attempts.values()),'http':len(attempts)}:
            raise ValueError('P3 inherited stage allowance changed')
