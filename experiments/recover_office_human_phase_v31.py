"""Explicit child run: retain metered prefix, finish from saved evidence, then untouched tasks."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import run_office_human_phase_v31 as runner
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint, save_json
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.budget_governor import reserved_recovery
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.prompts import finish_text_prompt
from atomic_skillgraph.experiments.formal_log import tree_identity,utc
from atomic_skillgraph.experiments.run_empirical import commit_trace, write_json

read=runner.read

def prepare(root,base):
    if root.exists():raise ValueError('Recovery requires a new directory')
    original=read(base/'plan.json');ledger=read(base/'budget.json')
    known=reserved_recovery(ledger)
    unknown={k:v for k,v in ledger['attempts'].items() if v['status']!='metered' and k not in known}
    if len(unknown)!=1:raise ValueError('Review each unresolved request before recovery')
    key,row=next(iter(unknown.items()))
    if row['status']!='unknown_billing' or row.get('actual_usage') is not None:raise ValueError('Not the reviewed missing-usage transport outcome')
    scope=row['context']['parent_scope'];source=Path(scope.rsplit(':',1)[0])
    if source.parent.parent!=base/'episodes':raise ValueError('Unexpected parent scope')
    prior=read(source/'summary.json');case=prior['cases'][0]
    if prior['unknown_billing_attempts']!=1 or case['execution']['prediction'] is not None:raise ValueError('Not an interrupted finalization')
    requests=read(next((source/'requests').glob('*.json')))
    failed=[h for q in requests['requests'] for h in q['http_attempts'] if h['request_id']==key]
    if len(failed)!=1 or failed[0]['error_code']!='provider_transport_error':raise ValueError('Expected the reviewed transport error')
    state=read(source/'checkpoints'/case['task']['task_id']/'1/state.json')
    if state.get('program_started') or any(t['state']!='finished' for t in case['tools']):raise ValueError('Unknown tool side effect')
    used=sum(runner.charge(v) for v in ledger['attempts'].values())
    remaining=ledger['limits']['token_limit']-used
    pending=[];carried=[]
    for e in original['episodes']:
        ep=Path(e['path']);s=read(ep/'summary.json')
        if ep==source:pending.append((e,'finish'))
        elif s['known_total_tokens']>0:
            if s['unknown_billing_attempts'] or s['cases'][0]['execution']['prediction'] is None:raise ValueError('Another incomplete paid episode')
            carried.append(e)
        else:
            q=read(next((ep/'requests').glob('*.json')))
            if any(x['http_attempts'] for x in q['requests']):raise ValueError('Unexpected HTTP in zero-cost terminal record')
            pending.append((e,'fresh'))
    owned=[v for v in ledger['attempts'].values() if v['context'].get('parent_scope')==scope]
    finish_room=192000-sum(runner.charge(v) for v in owned)
    if finish_room<=2048 or remaining<sum(192000 for _,mode in pending if mode=='fresh')+finish_room:
        raise ValueError('Insufficient original per-task/campaign authority')
    identity=runner.code_identity()
    if identity['tracked_dirty'] or not identity['git_sha']:raise ValueError('Commit before recovery preparation')
    if runner.package_identity()!=original['package_identity']:raise ValueError('Recovery cannot alter the Bank or bridge')
    root.mkdir(parents=True)
    save_json(root/'budget_prefix.json',ledger)
    recovery=deepcopy(ledger['inherited_reservation_recovery'])
    recovery['authorization']=('2026-10-18 user approved 12 paired tasks, cumulative95M and new4,608,000/2,400HTTP. '
        f'This child retains the exact additional {row["reserved_tokens"]}-token unknown upper bound within those unchanged limits; '
        'never reissues the unknown request; only uses the original parent finish reserve and untouched episodes.')
    recovery['attempts'][key]={'entry_hash':digest(row),'reserved_tokens':row['reserved_tokens']}
    ledger['inherited_reservation_recovery']=recovery;save_json(root/'budget.json',ledger)
    cfg=read(base/'base_config.json');save_json(root/'base_config.json',cfg)
    save_json(root/'tasks.json',read(base/'tasks.json'))
    bundle=read(runner.PACKAGE/'bank/bank.json');episodes=[]
    modes={e['path']:mode for e,mode in pending}
    for e in original['episodes']:
        old=Path(e['path']);mode=modes.get(e['path'])
        if mode is None:
            if read(old/'config.json')!=read(old/'execution_manifest.json')['config']:
                raise ValueError('Completed episode configuration differs from its execution identity')
            episodes.append({**e,'retained_from':str(base),'origin_code':original['code'],
                'prelaunch_config_hash':e['config_hash'],'config_hash':runner.sha(old/'config.json')});continue
        ep=root/'episodes'/old.relative_to(base/'episodes')
        runner.frontier.native_import(bundle,ep/'bank',cfg['program_environment'],cfg['experiment'].get('officeqa_method_version'),42)
        config=read(old/'config.json');config['data_dir']=str(ep/'bank');config['experiment']['output_dir']=str(ep)
        write_json(ep/'config.json',config)
        new={**e,'path':str(ep),'config_hash':runner.sha(ep/'config.json'),'bank_identity':tree_identity(ep/'bank')}
        if mode=='finish':new.update(recovery_source=str(old),original_parent_scope=scope,unknown_request_id=key,finish_reserve=finish_room)
        episodes.append(new)
    protected=[base/'budget.json',base/'plan.json',base/'launch_status.json',root/'budget_prefix.json',root/'tasks.json',root/'base_config.json']
    for e in carried:protected.extend([Path(e['path'])/'summary.json',Path(e['path'])/'freeze_record.json'])
    protected.extend(source.rglob('*.json'))
    plan={**original,'created_at':utc(),'code':identity,'base_run':str(base),'episodes':episodes,
        'protected_files':{**original['protected_files'],**{str(p):runner.sha(p) for p in protected}},'inherited_used_or_reserved':used,
        'new_token_cap':remaining,'new_http_cap':ledger['limits']['request_limit']-len(ledger['attempts']),
        'observation_prefix_path':str(base/'budget_prefix.json'),
        'locks':[str(base/'.run.lock'),*original['locks']], 'recovery':{
            'source_code':original['code'],'same_method_package':True,'retained_episodes':len(carried),
            'finish_from_checkpoint_only':str(source),'unknown_request_never_replayed':key,
            'unstarted_episodes':sum(mode=='fresh' for _,mode in pending),
            'zero_http_terminal_records_retained_in_parent_not_counted_as_answers':True}}
    save_json(root/'plan.json',plan);save_json(root/'plan_identity.json',{'sha256':runner.sha(root/'plan.json')})
    save_json(root/'continuation_manifest.json',plan['recovery'])
    save_json(root/'cost_carryforward_manifest.json',{'source':str(base),'used_or_reserved':used,'unknown_request_retained':key,
        'reserved_tokens':row['reserved_tokens'],'limits_unchanged':ledger['limits'],'authorization':recovery['authorization']})
    save_json(root/'launch_status.json',{'status':'prepared','created_at':utc()})
    print({'root':str(root),'carried':len(carried),'finish_reserve':finish_room,'remaining_tokens':remaining})


def finish_episode(config,task,out,episode,governor,formal,frontier,bundle):
    """No native calls or replay of the timed-out Runtime request; only reserved finalization."""
    source=Path(episode['recovery_source'])
    old=read(source/'summary.json');trace=deepcopy(old['cases'][0])
    saved=read(source/'checkpoints'/task.task_id/'1/state.json')
    checkpoint=TaskCheckpoint(out/'checkpoints'/task.task_id/'1')
    if not checkpoint.path.exists():
        checkpoint.advance('recovery_finish_prepared',source_checkpoint=str(source/'checkpoints'/task.task_id/'1/state.json'),
            executor_state=saved['executor_state'],human_office_evidence=saved.get('human_office_evidence',{}))
    identity={'schema':'empirical.finish-recovery.v1','config':config,'tasks':[asdict(task)],
              'code':runner.code_identity(),'readonly':True,'parent':str(source),
              'unknown_request_not_replayed':episode['unknown_request_id']}
    identity_path=out/'execution_manifest.json'
    if identity_path.exists() and read(identity_path)!=identity:raise ValueError('Finish recovery identity changed')
    save_json(identity_path,identity)
    audit=out/'requests'/f'{task.task_id}_attempt1.json';audit.parent.mkdir(parents=True,exist_ok=True)
    if not audit.exists():shutil.copy2(next((source/'requests').glob('*.json')),audit)
    formal.begin_task(task,0,task.task_id+':1',{'explicit_recovery':'finish_from_saved_evidence_only'})
    system=EmpiricalSystem(config,readonly=True,budget_governor=governor,
        bank_view_factory=lambda b:frontier.HumanBankView(b,bundle,episode['arm']))
    system.audit_path=audit;system.checkpoint=checkpoint;system.phase=task.split
    system.observer=formal
    system.budget_scope=task.task_id+':1';system.prior_usage=read(audit)['usage']
    system.request_attribution.update(parent_scope=episode['original_parent_scope'],parent_token_limit=192000,
        train_parent=True,parent_task_id=task.task_id)
    system.adapter.reset(task);system.task_context=TaskContext(config['runtime'])
    for k,v in saved['executor_state']['context'].items():setattr(system.task_context,k,deepcopy(v))
    try:
        prompt=finish_text_prompt(system.adapter.answer_contract())
        material={'original_goal':task.goal,'working_memory':system.task_context.model_memory(),
                  'public_notes':system.task_context.public_notes}
        if checkpoint.state['stage'] in {'task_execution_finished','task_committed'}:
            trace=deepcopy(checkpoint.state['trace'])
        else:
            prepared=system.prepare_finish(material,prompt)
            answer=system.agent('runtime',prompt,prepared['materials'],None,None,decision_purpose='finish_only')
            runner.require_open_billing(governor)
            trace['execution'].update(prediction=answer,reason='finish_only',finish_trigger='provider_transport_error_recovery',
                                      finish_evidence=prepared)
            trace['recovery_identity']={**identity,'original_parent_scope':episode['original_parent_scope']}
            trace['inherited_error']=trace.pop('error',None)
            broker=SimpleNamespace(context=system.task_context,events=trace['tools'],environment_steps=trace['environment_steps'])
            system.complete_execution(task,trace,broker,trace['execution'],learn=False)
        trace.update(read(audit));trace['knowledge_after']=system.bank.digest()
        checkpoint.native_events(trace['tools'])
        with sqlite3.connect(out/'run.sqlite3') as db:
            db.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,attempts INTEGER,status TEXT,result TEXT)')
            db.execute('INSERT OR IGNORE INTO tasks VALUES(?,?,?,?)',(task.task_id,1,'running',None))
            commit_trace(out,task.task_id,trace,db,checkpoint)
        known=sum(u['total_tokens'] for u in trace['usage'])
        summary={**old,'complete':True,'status':'completed_with_unknown_billing_reservation','successes':int(trace['score']['hard']),
            'total_tokens':None,'known_total_tokens':known,'completed_task_tokens':known,'cases':[trace],
            'recovery':trace['recovery_identity']}
        save_json(out/'summary.json',summary);formal.end_task(trace)
        return summary
    finally:system.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--base',required=True)
    args=ap.parse_args();prepare(Path(args.root).resolve(),Path(args.base).resolve())
