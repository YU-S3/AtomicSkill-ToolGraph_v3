"""Versioned Train-trace derivation: one P2 batch, optional Val, reviewed Test."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import yaml

from ..core.errors import BudgetExhausted
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.capability_batch import checked_asset
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import validate_config
from ..harness.registry import create_simple_harness
from . import run_choice_capability_batch as batch
from .canonical_manifest import verify, sha256
from .formal_choice_batch import BatchGovernor
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, run, write_json
from .run_formal import resolved_config

VERSION = 'choice-capability-batch.derived.v1'


def prepare(settings, source, output, datasets):
    root,source,data = map(lambda p:Path(p).resolve(),(output,source,datasets))
    if root.exists():raise ValueError('Derivation requires a fresh output, not ordinary --resume')
    plan = batch.prepare(settings,source,root/'p2',allow_unresolved_billing=True)
    original=batch.read(source/'train/config.json');old=batch.read(source/'run_manifest.json')
    if original['experiment'].get('choice_method_version')!='choice-capability-batch.v2':
        raise ValueError('This derivation preserves the completed P2 Train Bank method; incompatible legacy Bank needs a separate declared import')
    ledger=batch.read(source/'budget.json');prefix=deepcopy(ledger)
    unresolved=[key for key,row in ledger['attempts'].items() if row['status']!='metered']
    prefix['unknown_billing']=bool(prefix['unknown_billing'] or unresolved)
    cfg=batch.read(root/'p2/config.json')
    cfg['data_dir']=str(root/'train/bank')
    cfg['experiment'].update(choice_batch_version=VERSION,choice_batch_contract=plan['settings'],
                            choice_batch_contract_sha256=sha256(Path(settings)))
    cfg['learning']['source_program_payload_max_bytes']=plan['settings']['program_learning']['executed_source_payload_max_bytes']
    cfg=validate_config(cfg)
    from .capability_schedule import request_bound
    write_json(root/'request_bound.json',request_bound(cfg,plan['settings']))
    phase=batch.read(source/'p2/budget.json')['attempts'] if (source/'p2/budget.json').exists() else {}
    if (source/'p2/budget.json').exists():plan['input_hashes'][str(source/'p2/budget.json')]=sha256(source/'p2/budget.json')
    if any(prefix['attempts'].get(key)!=row for key,row in phase.items()):raise ValueError('Prior P2 billing differs from campaign ledger')
    for name,value in [('config.json',cfg),('campaign_budget_prefix.json',prefix),('inherited_stage_attempts.json',phase)]:
        write_json(root/'p2'/name,value)
        plan['prepared_hashes'][name]=sha256(root/'p2'/name)
    shutil.copytree(source/'train/bank',root/'train/bank')
    plan.update(derivation_mode=VERSION,initial_bank='explicit copy of completed Train Bank; original assets preserved',
                campaign_budget_prefix={'tokens':plan['inherited_tokens'],'http':plan['inherited_http']},
                prior_phase_tokens=sum(row.get('accounted_tokens',row['reserved_tokens']) for row in phase.values()),
                prior_phase_http=len(phase))
    write_json(root/'p2/plan.json',plan);write_json(root/'p2/plan_identity.json',{'plan_hash':digest(plan)})
    write_json(root/'budget.json',prefix)
    authority=Path(original['manifest']).resolve().parent.parent
    verify(authority)
    material=batch.read(data/'materialization.json')
    if material['authority_sha256']!=sha256(authority/'manifest.json'):raise ValueError('Different canonical materialization')
    for name,value in material['files_sha256'].items():
        if sha256(data/name)!=value:raise ValueError('Changed materialized input: '+name)
    for path in [data/'materialization.json',authority/'livemath/val.json',authority/'livemath/test.json',
                 *[data/name for name in material['files_sha256'] if name.startswith('livemath/')]]:
        plan['input_hashes'][str(path)]=sha256(path)
    profile=batch.read(Path(__file__).parents[3]/'benchmark_profiles.json')['profiles']['livemath']
    configs={phase:resolved_config(cfg,profile,'livemath',42,phase,root,data,authority,None) for phase in ('val','test')}
    identity={'method':'ours','benchmark':'livemath','run_seed':42,'split_seed':42,'model_name':old['provider_model_id'],
        'provider_model_id':old['provider_model_id'],'source_commit':plan['code']['git_sha'],
        'source_sha256':plan['code']['source_sha256'],'choice_batch_version':VERSION,
        'choice_method_version':cfg['experiment']['choice_method_version'],'initial_bank':'explicit copy of completed Train Bank',
        'source_train_run_id':old['run_id'],'source_train_commit':old['source_commit'],
        'source_train_run':str(source),'split_hashes':old['split_hashes'],
        'learning_schedule':'reuse Train60 -> P2 -> Frozen -> optional Val17 -> reviewed Test100',
        'test_development_history_disclosed':True,'materialization_sha256':sha256(data/'materialization.json')}
    write_json(root/'derivation_manifest.json',{'version':VERSION,'prepared_at':utc(),'code':code_identity(),
        'source_run':str(source),'source_run_id':old['run_id'],'source_commit':old['source_commit'],
        'source_hashes':plan['input_hashes'],'identity':identity,'configs':configs,'datasets':str(data),
        'authority':str(authority),'initial_bank_hash':tree_identity(root/'train/bank')['sha256'],
        'unresolved_billing_attempt_ids':unresolved,'new_model_http':0,'source_train_runtime_reexecuted':False})
    plan['prepared_hashes']['../derivation_manifest.json']=sha256(root/'derivation_manifest.json')
    plan['prepared_hashes']['../request_bound.json']=sha256(root/'request_bound.json')
    write_json(root/'p2/plan.json',plan);write_json(root/'p2/plan_identity.json',{'plan_hash':digest(plan)})
    return check(root)


def check(root):
    root=Path(root);record=batch.read(root/'derivation_manifest.json')
    if record['version']!=VERSION or record['code']!=code_identity():raise ValueError('Derivation source identity changed')
    result=batch.check(root/'p2/plan.json')
    if not (root/'p2/result.json').exists() and tree_identity(root/'train/bank')['sha256']!=record['initial_bank_hash']:
        raise ValueError('Imported Train Bank changed before P2')
    ledger=batch.read(root/'budget.json');prefix=batch.read(root/'p2/campaign_budget_prefix.json')
    if any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):raise ValueError('Inherited physical charges changed')
    frozen=root/'train/frozen_bank'
    if (root/'final_frozen_manifest.json').exists() and tree_identity(frozen)['sha256']!=batch.read(root/'final_frozen_manifest.json')['final_artifact_hash']:
        raise ValueError('Published Frozen changed')
    return {**result,'version':VERSION,'billing_ready':not ledger['unknown_billing'] and
        all(row['status']=='metered' for row in ledger['attempts'].values()),'test_requires_separate_review':True,
        'prior_phase_http':len(batch.read(root/'p2/inherited_stage_attempts.json'))}


def execute(root, env_file, *, stage='execute', reviewed=False):
    root=Path(root);check(root);record=batch.read(root/'derivation_manifest.json')
    if stage in {'val','test'}:
        if not (root/'p2/result.json').exists() or batch.read(root/'p2/result.json')['status'] not in {'completed_selection','completed_no_candidate'}:
            raise ValueError('Evaluation requires a completed P2 stage')
    if stage=='test' and (not reviewed or not (root/'val/summary.json').exists() or not batch.read(root/'val/summary.json')['complete']):
        raise ValueError('Test requires completed Val and explicit P2/Val review authorization')
    if env_file:load_env(env_file)
    ledger=batch.read(root/'budget.json')
    if ledger['unknown_billing'] or any(row['status']!='metered' for row in ledger['attempts'].values()):
        raise BudgetExhausted('unknown_billing','Inherited unfinished request needs an exact billing receipt; no paid request issued')
    governor=BudgetGovernor(root/'budget.json',**ledger['limits'])
    log=FormalLog(root,record['identity'],record['configs'],resume=(root/'run_manifest.json').exists())
    try:
        result=batch.execute(root/'p2/plan.json',None,
            governor=BatchGovernor(governor,batch.read(root/'p2/plan.json'),root/'p2'),observer=log,
            frozen_destination=root/'train/frozen_bank')
        if result['status'] not in {'completed_selection','completed_no_candidate'}:
            log.finish(result['status']);return result
        frozen=root/'train/frozen_bank';before=tree_identity(frozen)['sha256']
        bank=Bank(frozen,readonly=True)
        try:
            usable=sum(checked_asset(asset) for asset in bank.all('skill'))+sum(p.get('state')=='usable' and bank.program_eligible(p) for p in bank.all('program'))
        finally:bank.close()
        if not usable:
            log.finish('awaiting_p2_review_no_usable_assets')
            write_json(root/'completion.json',{'status':'awaiting_p2_review_no_usable_assets','p2':result,'val_started':False,'test_started':False})
            return {'status':'awaiting_p2_review_no_usable_assets','p2':result}
        phases=['val']
        if stage=='test':
            prior=root/'val/summary.json'
            write_json(root/'test_review.json',{'at':utc(),'source':record['code'],'frozen_hash':before,
                'p2_result_hash':sha256(root/'p2/result.json'),'val_summary_hash':sha256(prior),
                'explicit_reviewed_flag':True,'test_development_history_disclosed':True})
            phases=['test']
        summaries={}
        for phase in phases:
            cfg=record['configs'][phase];authority=Path(record['authority']);data=Path(record['datasets'])
            if sha256(data/'materialization.json')!=record['identity']['materialization_sha256']:raise ValueError('Materialization changed')
            public=batch.read(authority/'livemath'/(phase+'.json'))['tasks']
            tasks=[PublicTask(**row) for row in batch.read(data/'livemath'/(phase+'.json'))['tasks']]
            if [t.task_id for t in tasks]!=[r['task_id'] for r in public]:raise ValueError('Canonical evaluation membership/order changed')
            adapter=create_simple_harness(cfg)
            try:
                summary=run(cfg,tasks,root/phase,readonly=True,resume=(root/phase/'execution_manifest.json').exists(),
                    adapter=adapter,adapter_factory=lambda:create_simple_harness(cfg),formal_log=log,
                    task_metadata={r['task_id']:r['task_metadata'] for r in public},budget_governor=governor,order_offset=60 if phase=='val' else 77)
            finally:adapter.close()
            after=tree_identity(frozen)['sha256']
            if after!=before:raise ValueError('Frozen changed during readonly evaluation')
            write_json(root/(phase+'_freeze_audit.json'),{'hash_before':before,'hash_after':after,'complete':summary['complete']})
            summaries[phase]={k:summary[k] for k in ('complete','tasks','successes','known_total_tokens')}
        status='awaiting_test_review' if stage!='test' else 'completed'
        log.finish(status);write_json(root/'completion.json',{'status':status,'p2':result,'evaluation':summaries})
        return {'status':status,**summaries}
    except BaseException as exc:
        log.finish('budget_stopped' if isinstance(exc,BudgetExhausted) else 'execution_failed')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    item=sub.add_parser('prepare')
    for name in ('settings','source-run','output','datasets'):item.add_argument('--'+name,required=True)
    for stage in ('check','execute','val','test'):
        item=sub.add_parser(stage);item.add_argument('--output',required=True)
        if stage!='check':item.add_argument('--env-file',required=True)
        if stage=='test':item.add_argument('--reviewed-p2-val',action='store_true')
    args=parser.parse_args()
    result=prepare(args.settings,args.source_run,args.output,args.datasets) if args.stage=='prepare' else check(args.output) if args.stage=='check' else execute(
        args.output,args.env_file,stage=args.stage,reviewed=getattr(args,'reviewed_p2_val',False))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
