"""Registered Train-only predicate search, exact-response reuse, Val selection, one Test."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from ..empirical import math_predicate_evolution as e, math_effect_program_r3 as p
from ..empirical import math_effect_runtime_r3 as runtime
from ..empirical.contracts import PublicTask,digest
from ..empirical.budget_governor import billing_blocked
from ..harness.benchmarks import AnswerAdapter
from . import run_math_effect_r3 as base
from .run_empirical import write_json,load_env
from .canonical_manifest import sha256
from .run_math_effect_p7 import StageContext,phase_governor
from .formal_log import utc

read=base.read


def evidence_files(source_run,candidates,ledger):
    """Require the original paid response, submitted candidate and metered usage."""
    files=[]
    def contains(value,expected):
        if isinstance(value,dict):return value==expected or any(contains(x,expected) for x in value.values())
        if isinstance(value,list):return any(contains(x,expected) for x in value)
        return False
    responses=[(path,read(path)) for path in (Path(source_run)/'learning').glob('**/response.json')]
    for candidate in candidates:
        evidence=candidate['model_evidence']
        matched=[(path,value) for path,value in responses if value==evidence]
        if len(matched)!=1:raise ValueError('Candidate lacks one exact original model response')
        path,value=matched[0];records=read(path.parent/'requests.json')
        if (digest(value['turn'])!=value['response_hash'] or digest(records)!=value['request_hash'] or
            [x['request_id'] for x in records]!=value['request_ids'] or
            not contains(value['turn']['tool_calls'],candidate['candidate'])):
            raise ValueError('Candidate content/response/physical requests differ')
        for record in records:
            charge=ledger['attempts'].get(record['request_id'],{})
            if charge.get('status')!='metered' or charge.get('actual_usage')!=record['raw_usage']:
                raise ValueError('Candidate has no matching original metered charge')
        files.extend([path,path.parent/'request.json',path.parent/'requests.json'])
    return files


def prepare(root,config,source_run,datasets):
    import yaml
    root=Path(root);settings=yaml.safe_load(Path(config).read_text());spec=settings['derivation']
    if spec['version']!=e.VERSION or spec['max_neighbors_per_parent']!=6 or spec['max_val_policies']!=2:
        raise ValueError('Finite neighborhood contract changed')
    parent=Path(spec['parent_run']);progress=read(parent/'progress.json')
    if progress['stage']!='stopped' or not (parent/'controlled_pause.json').exists():raise ValueError('R3 must finish its current phase before derivation')
    ledger=read(parent/'budget/ledger.json');prefix=read(parent/'budget/prefix.json')['attempts']
    if billing_blocked(ledger) or any(a['status']!='metered' for k,a in ledger['attempts'].items() if k not in prefix):
        raise ValueError('Parent has an unfinished physical request')
    # Fresh preparation reuses original Train source and billing validators. No old
    # plan/checkpoint is modified; the derivation fields below are frozen before HTTP.
    base.prepare(root,config,source_run,datasets)
    plan=read(root/'plan.json');cfg=read(root/'config.json');sources=plan['learning_source_keys']
    learned=read(parent/'learning/0/learning.json');proposals=[x for x in learned['candidates'] if x['candidate']['public_choice_rule'] is not None]
    candidates=[child for x in proposals for child in e.neighborhood(x,sources,settings['tokenizer_path'])]
    policies=[];paths=[parent/'learning/0/learning.json',parent/'controlled_pause.json',parent/'plan.json']
    paths.extend(evidence_files(parent,proposals,ledger))
    for source_path in [Path(spec['text_source_run']),parent]:
        path=source_path/('feedback/result.json' if source_path!=parent else 'feedback/best.json')
        value=read(path);receipt=value['best'] if source_path!=parent else value['receipt']
        policy=[e.import_text(x,sources,settings['tokenizer_path']) for x in value['policy'] if x['candidate'].get('public_choice_rule') is None]
        old_policy=value['policy'];by_key={row['physical_key']:row['on'] for row in receipt['rows']}
        paths.extend(evidence_files(source_path,old_policy,ledger))
        if len(by_key)!=12 or any(t['task']['split']!='train' for t in by_key.values()):raise ValueError('Only real Train feedback policies enter selection')
        traces={}
        for key,t in by_key.items():
            runtime.verify(t)
            trace=source_path/'feedback/on'/digest(old_policy)/key/'trace.json'
            if read(trace)['trace_hash']!=t['trace_hash']:raise ValueError('Prior feedback path differs')
            traces[key]=str(trace);paths.append(trace)
            paths.extend(Path(x) for x in t['request_files'])
        policies.append({'source_run':str(source_path),'source_policy_hash':digest(old_policy),'policy':policy,
            'source_candidate_hashes':[digest(x) for x in old_policy],'train_traces':traces})
        paths.extend([path,source_path/'plan.json'])
    # Empty text is a transparent baseline, not an invented learned candidate.
    policies.append({'source_run':None,'source_policy_hash':None,'policy':[],
        'source_candidate_hashes':[],'train_traces':{k:v['path'] for k,v in read(root/'baseline.json')['feedback'].items()}})
    registration={'method_version':e.VERSION,'asset_contract_version':p.VERSION,'created_at':utc(),
        'parent_learning_hash':digest(learned),'mutation_operators':['delete_one_positive_if_nonempty','delete_one_exclusion','promote_one_exclusion'],
        'max_neighbors_per_parent':6,'candidate_hashes':[digest(x) for x in candidates],
        'text_policy_hashes':[digest(x) for x in policies],'max_train_selected_policies':2,'max_assets_per_policy':4,
        'selection_inputs':'Train60 feedback only; no Val/Test scores','additional_learning_http':0,
        'test_development_history_disclosed':True,'old_promotion_decisions_imported':False}
    for name,value in [('neighborhood_candidates.json',candidates),('text_policies.json',policies),('derivation_registration.json',registration)]:
        write_json(root/name,value);plan['prepared_hashes'][name]=sha256(root/name)
    for path in paths:plan['files'][str(path.resolve())]=sha256(path)
    plan.update(method_version=e.VERSION,asset_contract_version=p.VERSION,derivation=registration)
    plan['run_id']='predicate_evolution_'+digest(plan)[:24]
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'hash':digest(plan)})
    result=base.check(root);result['derivation']=registration;write_json(root/'zero_http_check.json',result);return result


def reuse_for(root,descriptor,phase,key,task,cfg,policy):
    from ..empirical import math_effect_runtime as original
    from ..empirical.math_effect_contract import card
    if phase=='feedback':return descriptor['train_traces'][key]
    selection=p.routing(policy,task,cfg['effect_p7']['tokenizer_path'])
    payload=original.provider(cfg['llm'],'runtime',cfg['effect_p7'])._build_payload(
        original.messages(task,[card(x['candidate']) for x in selection['selected']]),[])
    candidates=[]
    if descriptor['source_run']:
        candidates.append(Path(descriptor['source_run'])/'val/on'/descriptor['source_policy_hash']/key/'trace.json')
    candidates.append(Path(read(root/'baseline.json')['val'][key]['path']))
    for path in candidates:
        if runtime.compatible_response(path,task,cfg,payload):return str(path)
    return None


def evaluate(root,plan,descriptor,rule,q,phase):
    cfg=read(root/'config.json');split='train' if phase=='feedback' else 'val';policy=[rule,*descriptor['policy']]
    tasks={x['physical_key']:PublicTask(**x) for x in read(root/'tasks.json')[split]}
    baselines=read(root/'baseline.json')[phase];on=[];off=[];policy_id=digest(policy)
    if phase=='val':base.authorize(root,plan,'val')
    g=StageContext(phase_governor(root,phase,plan),phase)
    adapter=AnswerAdapter('livemath',read(Path(plan['datasets'])/'livemath'/('evaluator_records_'+split+'.json')))
    try:
        for key,b in baselines.items():
            task=tasks[key];off.append(read(b['path']))
            reuse=reuse_for(root,descriptor,phase,key,task,cfg,policy)
            if reuse:
                trace=read(reuse);ledger=read(root/'budget/ledger.json')
                for rid,usage in zip(trace['request_ids'],trace['raw_usage']):
                    charge=ledger['attempts'].get(rid,{})
                    if charge.get('status')!='metered' or charge.get('actual_usage')!=usage:raise ValueError('Reuse has no original physical charge')
            trace=runtime.solve(task,adapter,policy,{rule['id']:q},cfg,root/phase/'on'/policy_id/key,g,
                reuse_trace=reuse,require_reuse=phase=='feedback')
            on.append(trace)
            base.progress(root,phase,completed=len(on),total=len(baselines),policy_id=policy_id,
                correct=sum(x['score']['hard'] for x in on),program_answers=sum(x['answer_origin']=='learned_program' for x in on),
                reused_model_responses=sum(bool(x.get('reused_model_response')) for x in on))
    finally:adapter.close()
    receipt=runtime.pair_report(off,on,policy,split)
    write_json(root/phase/'receipts'/(policy_id+'.json'),receipt)
    return {'policy':policy,'qualifications':{rule['id']:q},'descriptor':descriptor,'receipt':receipt}


def execute(root):
    root=Path(root);plan=base.checked(root)
    if plan['method_version']!=e.VERSION:raise ValueError('Wrong derived method')
    if (root/'result.json').exists():return read(root/'result.json')
    if (root/'freeze_identity.json').exists():
        result=read(root/'selection.json');result['frozen']=base.frozen(root);write_json(root/'result.json',result);return result
    cfg=read(root/'config.json');candidates=read(root/'neighborhood_candidates.json')
    base.progress(root,'train_neighborhood',candidate_count=len(candidates))
    qs=base.qualify(root,plan,candidates);good=sorted([q for q in qs.values() if q['qualified_for_trial']],key=e.rule_rank)
    write_json(root/'neighborhood_result.json',{'candidates':candidates,'qualification_hashes':{k:digest(v) for k,v in qs.items()},
        'qualified_for_trial':[q['candidate_id'] for q in good],'new_model_http':0,'source':'Train only'})
    if not good:
        result={'status':'completed_no_candidate','method_version':e.VERSION};write_json(root/'result.json',result);return result
    chosen=good[0];rule=next(c for c in candidates if c['id']==chosen['candidate_id']);trials=[]
    for descriptor in read(root/'text_policies.json'):trials.append(evaluate(root,plan,descriptor,rule,chosen,'feedback'))
    ordered=sorted(trials,key=lambda x:base.rank(x['receipt']))[:2]
    choice={'search_version':e.VERSION,'selected':[digest(x['policy']) for x in ordered],
        'all_train_results':[{'policy_id':digest(x['policy']),'receipt_hash':x['receipt']['receipt_hash'],'rank':base.rank(x['receipt'])} for x in trials],
        'candidate_rule_id':rule['id'],'selection_split':'train','val_feedback_read':False}
    if (root/'train_selection.json').exists() and read(root/'train_selection.json')!=choice:raise ValueError('Train selection changed')
    write_json(root/'train_selection.json',choice)
    observed=[evaluate(root,plan,x['descriptor'],rule,chosen,'val') for x in ordered]
    accepted=[x for x in observed if x['receipt']['qualified']]
    val_key=lambda x:(-x['receipt']['groups']['all']['on_correct'],x['receipt']['groups']['all']['losses'],
        x['receipt']['groups']['all']['tokens_on'],digest(x['policy']))
    winner=min(accepted or observed,key=val_key)
    result={'method_version':e.VERSION,'status':'val_passed' if accepted else 'completed_negative_val',
        'policy':winner['policy'],'qualifications':winner['qualifications'],'val_receipt':winner['receipt'],
        'train_receipt':next(x['receipt'] for x in trials if digest(x['policy'])==digest(winner['policy'])),
        'val_selection_receipts':[x['receipt']['receipt_hash'] for x in observed],'val_is_selection_not_independent_test':True,
        'automatic_mutation_candidates':len(candidates),'train_qualified_candidates':len(good),'new_learning_model_http':0}
    write_json(root/'selection.json',result)
    if accepted:result['frozen']=base.freeze(root,plan,result)
    write_json(root/'result.json',result);base.progress(root,result['status']);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','execute','test','monitor'])
    parser.add_argument('--output',required=True);parser.add_argument('--config');parser.add_argument('--source-run');parser.add_argument('--datasets');parser.add_argument('--env-file')
    args=parser.parse_args();root=Path(args.output)
    if args.env_file:load_env(args.env_file)
    if args.command=='prepare':result=prepare(root,args.config,args.source_run,args.datasets)
    elif args.command=='check':result=base.check(root)
    elif args.command=='monitor':result=read(root/'progress.json') if (root/'progress.json').exists() else read(root/'zero_http_check.json')
    else:
        import fcntl
        with (root/'controller.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:result=execute(root) if args.command=='execute' else base.test(root)
            except Exception as exc:base.progress(root,'stopped',error_type=type(exc).__name__,error=str(exc));raise
    print(json.dumps({k:v for k,v in result.items() if k not in {'policy','qualifications','val_receipt','train_receipt','frozen'}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
