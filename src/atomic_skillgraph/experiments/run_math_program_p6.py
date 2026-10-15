"""Isolated P6 prepare/check, bounded Train learning/transfer and paired Train smoke."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from ..empirical import math_operation_contract_p6 as c, math_program_p6 as m
from ..empirical.math_program_induction import load_sources
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, billing_blocked, input_token_bound
from ..empirical.contracts import digest, PublicTask
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from . import run_math_program_p5 as p5
from .canonical_manifest import sha256
from .formal_choice_batch import BatchGovernor
from .formal_log import FormalLog, tree_identity, utc
from .run_choice_capability_batch import learning_boundary
from .run_empirical import code_identity, write_json, run, load_env

CONTRACT=Path(__file__).parents[3]/'configs/livemath_p6_seed42.yaml'
read=p5.read


def allocation(settings):
    """Reserve the parallel manual run's WHOLE authorized cap, not a changing cost estimate."""
    parallel=Path(settings['parallel_manual_run']);plan=read(parallel/'plan.json')
    prefix=read(parallel/'campaign_budget_prefix.json');current=read(parallel/'budget.json')
    limits=settings['cumulative_limits']
    if prefix['limits']!=limits or current['limits']!=limits:raise ValueError('P6 must retain the cumulative Governor')
    if any(current['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):raise ValueError('Parallel historical costs changed')
    added={k:v for k,v in current['attempts'].items() if k not in prefix['attempts']}
    reserve={k:sum(v[k] for v in plan['stage_limits'].values()) for k in ('token_limit','request_limit')}
    used=sum(v.get('accounted_tokens',v['reserved_tokens']) for v in added.values())
    if used>reserve['token_limit'] or len(added)>reserve['request_limit']:raise ValueError('Parallel allocation exceeded')
    proposed=settings['stage_limits_proposed_not_authorized']
    inherited=sum(v.get('accounted_tokens',v['reserved_tokens']) for v in prefix['attempts'].values())
    occupied={'token_limit':inherited+reserve['token_limit']+sum(v['token_limit'] for v in proposed.values()),
              'request_limit':len(prefix['attempts'])+reserve['request_limit']+sum(v['request_limit'] for v in proposed.values())}
    if any(occupied[k]>limits[k] for k in occupied):raise ValueError('Proposed P6 and parallel reservations exceed cumulative authority')
    if billing_blocked(prefix):raise ValueError('Historical billing is unresolved without reservation authority')
    return prefix,{'parallel_plan':str(parallel/'plan.json'),'parallel_plan_sha256':sha256(parallel/'plan.json'),
        'parallel_prefix_sha256':sha256(parallel/'campaign_budget_prefix.json'),'reserved_parallel_ceiling':reserve,
        'historical_tokens_or_reservation':inherited,'historical_http':len(prefix['attempts']),
        'occupied_if_approved':occupied,'proposed_not_authorized':proposed,
        'accounting':'physical IDs inherited once; parallel new usage covered by full reserved ceiling, never added twice'}


def configure(original, root, settings):
    cfg=p5.configure(original,root,{'cumulative_limits':settings['cumulative_limits']})
    cfg['experiment']['math_program_method_version']=c.VERSION
    return validate_config(cfg)


def prepare(output, config=CONTRACT):
    root=Path(output).resolve();settings=yaml.safe_load(Path(config).read_text());code=code_identity()
    if root.exists():raise ValueError('P6 requires a new directory')
    if settings!=yaml.safe_load(CONTRACT.read_text()) or settings['method_version']!=c.VERSION:raise ValueError('P6 frozen configuration changed')
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit P6 before preparing its identity')
    rows,atoms,files,original,parent=load_sources(settings['source_train'],settings['source_p4'])
    demands,units,transfer=m.catalog(rows,settings['max_units'])
    prefix,alloc=allocation(settings);cfg=configure(original,root,settings)
    # P5 read-only preservation evidence, not imported learning assets or utility.
    old=Path(settings['source_p5'])
    for name in ('budget.json','usage_ledger.json','pilot_manifest.json','trial_manifest.json'):
        if (old/name).exists():files[str(old/name)]=sha256(old/name)
    for path in (old/'frozen_bank').rglob('*'):
        if path.is_file():files[str(path)]=sha256(path)
    files[alloc['parallel_plan']]=alloc['parallel_plan_sha256']
    files[str(Path(settings['parallel_manual_run'])/'campaign_budget_prefix.json')]=alloc['parallel_prefix_sha256']
    initial=Bank(root/'bank',environment=cfg['program_environment']);initial_digest=initial.digest();initial.close()
    for name,value in [('config.json',cfg),('source_catalog.json',rows),('atomic_source_catalog.json',atoms),
        ('public_demands.json',demands),('units.json',units),('campaign_budget_prefix.json',prefix),('budget.json',prefix),
        ('allocation_manifest.json',alloc),('learning/campaign_budget_prefix.json',prefix)]:write_json(root/name,value)
    plan={'version':c.VERSION,'run_id':'p6_'+digest([code,files,settings])[:24],'code':code,'created_at':utc(),
        'settings':settings,'source_train_run_id':parent['run_id'],'files':files,'initial_bank_digest':initial_digest,
        'transfer_physical_keys':transfer,'smoke_physical_keys':transfer[:settings['smoke_tasks']],
        'initial_bank':'empty; no P4/P5/manually authored published assets','val_test_enabled':False,
        'prepared_hashes':{name:sha256(root/name) for name in ('config.json','source_catalog.json','atomic_source_catalog.json',
            'public_demands.json','units.json','campaign_budget_prefix.json','allocation_manifest.json')}}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    return check(root)


def checked(root):
    root=Path(root);plan=read(root/'plan.json')
    if plan['version']!=c.VERSION or plan['code']!=code_identity() or digest(plan)!=read(root/'plan_identity.json')['sha256']:
        raise ValueError('P6 source/plan changed; use an explicit new version')
    for path,h in plan['files'].items():
        if sha256(Path(path))!=h:raise ValueError('Historical input changed: '+path)
    for name,h in plan['prepared_hashes'].items():
        if sha256(root/name)!=h:raise ValueError('Prepared P6 input changed: '+name)
    prefix=read(root/'campaign_budget_prefix.json');ledger=read(root/'budget.json')
    if ledger['limits']!=prefix['limits'] or ledger.get('inherited_reservation_recovery')!=prefix.get('inherited_reservation_recovery') or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('P6 inherited costs or reservation identity changed')
    allocation(plan['settings']);validate_config(read(root/'config.json'));return plan


def check(root):
    from ..agents.provider import OpenAICompatibleProvider, OpenAICompatibleConfig
    from ..agents.protocol import NativeToolSpec
    from ..empirical.model_view import learning_payload
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');capacities=[]
    rows=read(root/'source_catalog.json');atoms=read(root/'atomic_source_catalog.json')
    for unit in read(root/'units.json'):
        materials=[('induction',c.PROMPT,m.material(unit,rows,atoms),'submit_math_program_spec',c.PROPOSAL,4096),
            ('builder_repair_capacity',c.BUILD_PROMPT,{'registry':c.interface(unit['operation_kind']),
                'previous_source':'x'*16000,'previous_failure':'x'*8192},'submit_program',c.BUILD,8192)]
        for stage,prompt,material,name,schema,cap in materials:
            provider=OpenAICompatibleProvider(OpenAICompatibleConfig(base_url=cfg['llm']['base_url'],model=cfg['llm']['model'],
                api_key_env=cfg['llm']['api_key_env'],dialect=cfg['llm']['dialect'],thinking_type='disabled',max_completion_tokens=cap,max_retries=0))
            wire=provider._build_payload([{'role':'system','content':prompt},{'role':'user','content':json.dumps(learning_payload(material),ensure_ascii=False)}],
                [NativeToolSpec(name,'Submit the requested result',schema)],{'type':'function','function':{'name':name}})
            size=input_token_bound(wire)
            if size>65536:raise ValueError('P6 required material exceeds frozen wire capacity')
            # Exercise the actual parent, stage and campaign admission without touching their ledgers.
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(prefix='p6_admission_') as scratch:
                scratch=Path(scratch);write_json(scratch/'budget.json',read(root/'budget.json'))
                write_json(scratch/'learning/campaign_budget_prefix.json',read(root/'learning/campaign_budget_prefix.json'))
                owner=BudgetGovernor(scratch/'budget.json',**plan['settings']['cumulative_limits'])
                governor=BatchGovernor(owner,{'run_id':plan['run_id'],'new_limits':plan['settings']['stage_limits_proposed_not_authorized']['learning']},scratch/'learning')
                context={'parent_scope':plan['run_id']+':'+unit['unit_id'],'parent_token_limit':1000000,
                    'episode_scope':unit['unit_id'],'episode_request_limit':3}
                governor.admit('offline_admission_only',wire,context)
            capacities.append({'unit_id':unit['unit_id'],'stage':stage,'wire_bytes':size,'reserved_tokens':size+cap,
                'actual_governor_admitted':True,'new_http':0})
    docker=p5.docker_ready()
    result={'status':'passed' if docker['ready'] else 'blocked','plan_hash':digest(plan),'new_model_http':0,'docker':docker,
        'train_sources':len(rows),'units':len(read(root/'units.json')),'transfer_sources':len(plan['transfer_physical_keys']),
        'capacities':capacities,'billing_ready':not billing_blocked(read(root/'budget.json')),
        'paid_authorized':(root/'paid_authorization.json').exists(),'val_test_enabled':False}
    write_json(root/'zero_http_check.json',result);return result


def receive(system, plan, unit, folder, stage, prompt, material, name, schema, purpose, validator):
    learning_boundary(system,folder,plan['run_id']+':'+unit['unit_id'],unit_limits={'token_limit':1000000,'request_limit':3})
    identity=digest([plan['code'],system.config['llm'],prompt,material,schema]);failure=folder/('rejected_'+identity+'.json')
    if failure.exists():
        old=read(failure);exc=c.ContractError(old['diagnostics']);exc.model_authored=True;raise exc
    try:
        return system.learner._receive('p6_'+identity,stage,prompt,material,name,schema,material_version=c.MATERIAL,
            decision_purpose=purpose,validator=validator,repair_limit=0)
    except ValueError as exc:
        if not p5.model_failure(exc):raise
        if str(exc).startswith('Recovered terminal'):
            raise RuntimeError('P6 rejection was saved without its diagnostic receipt; inspect the original response before recovery, no repeated HTTP') from exc
        diagnostics=getattr(exc,'diagnostics',None)
        turn=getattr(system,'last_response_turn',None)
        if diagnostics is None and turn and len(turn.tool_calls)==1:
            raw=turn.tool_calls[0].arguments
            diagnostics=c.schema_errors(raw,schema)
            if not diagnostics:
                try:validator(raw)
                except c.ContractError as detail:diagnostics=detail.diagnostics
        diagnostics=diagnostics or [c.error('response_json_invalid' if 'JSON' in str(exc) or 'json' in str(exc) else 'model_contract_invalid','$',str(exc),None)]
        write_json(failure,{'identity':identity,'diagnostics':diagnostics})
        wrapped=c.ContractError(diagnostics);wrapped.model_authored=True;raise wrapped from exc


def learn(root, provider=None):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');rows=read(root/'source_catalog.json');atoms=read(root/'atomic_source_catalog.json')
    owner=BudgetGovernor(root/'budget.json',**plan['settings']['cumulative_limits'])
    governor=BatchGovernor(owner,{'run_id':plan['run_id'],'new_limits':plan['settings']['stage_limits_proposed_not_authorized']['learning']},root/'learning')
    system=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}),budget_governor=governor,provider=provider)
    try:
        started=root/'learning/started.json'
        if not started.exists():
            if system.bank.digest()!=plan['initial_bank_digest']:raise ValueError('P6 must start from the declared empty Bank')
            write_json(started,{'plan_hash':digest(plan),'initial_bank_digest':system.bank.digest()})
        for unit in read(root/'units.json'):
            folder=root/'learning'/unit['unit_id'];path=folder/'learning.json'
            if path.exists():continue
            used_repair=False;value=None;diagnostics=[];material=m.material(unit,rows,atoms)
            while True:
                try:
                    value=receive(system,plan,unit,folder,'extractor',c.PROMPT,material,'submit_math_program_spec',c.PROPOSAL,
                        'math_program_spec',lambda v:c.validate_spec(v,unit,rows));value=c.validate_spec(value,unit,rows);break
                except ValueError as exc:
                    if not p5.model_failure(exc):raise
                    diagnostics+=exc.diagnostics
                    if used_repair:break
                    used_repair=True;material={**material,'previous_failure':exc.diagnostics}
            result={'unit':unit,'proposal':value,'diagnostics':diagnostics,'state':value['decision'] if value else 'model_invalid','usable':False}
            if value and value['decision']=='program_candidate':
                asset=system.bank.put('skill',m.atomic(value,unit));spec=value['program_spec']
                build_material={'atomic':value['atomic'],'program_spec':spec,'registry':c.interface(spec['operation_kind'])}
                while True:
                    try:
                        answer=receive(system,plan,unit,folder,'tool_builder',c.BUILD_PROMPT,build_material,'submit_program',c.BUILD,
                            'math_program_build',lambda v:m.program(v,spec,cfg['program_environment']))
                        program=system.bank.put('program',m.program(answer,spec,cfg['program_environment']))
                        qualification=m.qualify(system.worker,program,value,rows,plan['transfer_physical_keys'],folder)
                        result.update(state='local_algorithm_checked' if qualification['local_algorithm_checked'] else 'local_check_failed',
                            atomic=asset,program=program,qualification=qualification,autonomously_generated=True)
                        if qualification['local_algorithm_checked'] or used_repair:break
                        used_repair=True;build_material.update(previous_source=program['source'],previous_failure=qualification)
                    except ValueError as exc:
                        if not p5.model_failure(exc):raise
                        diagnostics+=exc.diagnostics;result['state']='model_invalid'
                        if used_repair:break
                        used_repair=True;build_material.update(previous_failure=exc.diagnostics)
            result['shared_repair_used']=used_repair;write_json(path,result)
    finally:system.close()
    return report(root)


def trial_bank(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');groups={};evidence=[]
    sources=read(root/'source_catalog.json')
    for unit in read(root/'units.json'):
        path=root/'learning'/unit['unit_id']/'learning.json'
        if not path.exists():raise ValueError('Finish all fixed learning units before trial export')
        row=read(path)
        if not row.get('qualification',{}).get('local_algorithm_checked'):continue
        verify_learning(row,unit,sources,plan,cfg,path.parent)
        kind=row['program']['entry_constraints']['operation_kind'];groups.setdefault(c.fingerprint(kind),[]).append(row)
        for p in path.parent.rglob('*.json'):
            if p.name.endswith('.started.json'):continue
            evidence.append({'path':str(p),'sha256':sha256(p)})
    path=root/'trial_manifest.json'
    if path.exists():
        from ..empirical.math_program_trial_p6 import TrialView
        old=read(path)
        if old['evidence_files']!=evidence or old['tree_hash']!=tree_identity(root/'trial_bank')['sha256']:
            raise ValueError('P6 trial Bank or evidence changed')
        bank=Bank(root/'trial_bank',readonly=True,environment=cfg['program_environment'])
        try:TrialView(bank,path)
        finally:bank.close()
        return old
    if (root/'trial_bank').exists():raise ValueError('Interrupted trial export without manifest; preserve it for explicit recovery')
    bank=Bank(root/'trial_bank',environment=cfg['program_environment']);links={};lineage={}
    try:
        for fp,items in groups.items():
            representative=items[0]['program'];bank.put('program',representative);links[representative['id']]=[]
            lineage[fp]={'representative_program_id':representative['id'],'members':[],
                'basis':'same registered deterministic operation and exact schemas; every member independently checked'}
            for row in items:
                asset=bank.put('skill',row['atomic']);links[representative['id']].append(asset['id'])
                bank.put('implementation',{'skill_id':asset['id'],'program_id':representative['id'],'state':'candidate','method_version':c.VERSION})
                lineage[fp]['members'].append({'program_id':row['program']['id'],'skill_id':asset['id'],'source_case_ids':asset['source_case_ids'],
                    'qualification':row['qualification'],'source_sha256':digest(row['program']['source'])})
        result={'version':c.VERSION,'bank_digest':bank.digest(),'program_hashes':{p['id']:digest(p) for p in bank.all('program')},
            'program_skills':links,'lineage':lineage,'evidence_files':evidence,'formal_usable':False,
            'qualification':'read-only experimental trial Bank, not formally published Frozen'}
    finally:bank.close()
    result['tree_hash']=tree_identity(root/'trial_bank')['sha256']
    write_json(path,result);return result


def verify_learning(row,unit,sources,plan,cfg,folder):
    import hashlib
    value=c.validate_spec(row['proposal'],unit,sources)
    if row['unit']!=unit or row['atomic']!=m.atomic(value,unit) or row.get('autonomously_generated') is not True:
        raise ValueError('P6 natural source lineage changed')
    p=row['program'];rebuilt=m.program({'source':p['source']},value['program_spec'],cfg['program_environment'])
    if p!=rebuilt:raise ValueError('P6 generated Program identity changed')
    requests=read(folder/'requests.json')['requests']
    for name,args in [('submit_math_program_spec',value),('submit_program',{'source':p['source']})]:
        if not any(r.get('response',{}).get('usage') and any(t['name']==name and t['arguments']==args for t in r['response'].get('tool_calls',[])) for r in requests):
            raise ValueError('P6 asset lacks its exact recorded model response')
    q=row['qualification'];expected=m.inputs(value['program_spec'],sources,value['source_case_ids'])
    if len(q['receipts'])!=len(expected):raise ValueError('Missing P6 local qualification')
    for receipt,(role,args) in zip(q['receipts'],expected):
        if receipt['role']!=role or receipt['arguments']!=args:raise ValueError('P6 local input changed')
    for receipt in q['receipts']+q['transfer_receipts']:
        if receipt!=read(folder/'worker'/(receipt['id']+'.json')) or receipt['receipt_hash']!=digest({k:v for k,v in receipt.items() if k!='receipt_hash'}):
            raise ValueError('P6 Worker evidence changed')
        result=receipt['result'];wire=result.get('execution_receipt',{})
        if (wire.get('image_digest')!=c.IMAGE
                or '--network=none' not in wire.get('command',[]) or wire.get('source_sha256')!=hashlib.sha256(p['source'].encode()).hexdigest()
                or wire.get('input_hash')!=digest(receipt['arguments']) or wire.get('result_hash')!=digest({k:v for k,v in result.items() if k!='execution_receipt'})):
            raise ValueError('P6 requires actual locked Worker output')
        expected=None if receipt['role']=='missing_premise' else c.oracle(p['entry_constraints']['operation_kind'],receipt['arguments'])
        passed=(result['status']=='needs_input' if expected is None else result['status']=='ok' and result['outputs']==expected)
        passed=passed and wire.get('exit_code')==0 and not receipt['rpc_calls']
        if receipt['passed']!=passed or receipt['expected_local_outputs']!=expected:
            raise ValueError('P6 independent oracle receipt changed')
        if receipt in q['receipts'] and not passed:raise ValueError('P6 local qualification failed')
        if receipt in q['transfer_receipts']:
            key=receipt['grounding']['physical_key']
            if key not in plan['transfer_physical_keys'] or key in value['source_case_ids']:raise ValueError('P6 transfer source is not independent')
            task=next(r['task'] for r in sources if r['task']['physical_key']==key)
            if receipt['grounding']!=c.ready_bindings(task,p['entry_constraints']['operation_kind']):raise ValueError('P6 transfer grounding changed')
    if q['task_transfer_checked']!=any(r['passed'] for r in q['transfer_receipts']):raise ValueError('P6 transfer qualification changed')


def smoke_config(root, arm):
    root=Path(root);cfg=read(root/'config.json');cfg['data_dir']=str(root/'trial_bank')
    cfg['experiment'].update(runtime_mode='frozen',math_program_evaluation_version=c.VERSION,
        math_program_trial_manifest=str(root/'trial_manifest.json'),math_program_trial_arm=arm,output_dir=str(root/'smoke'/arm))
    cfg['llm']['runtime'].update(max_completion_tokens=65536,request_timeout_seconds=600,max_total_tokens_per_task=128000,
        protocol={'thinking_type':'enabled'},reasoning_effort='high')
    cfg['llm']['purpose_overrides']['finish_only']={'protocol':{'thinking_type':'enabled'},'reasoning_effort':'low','max_completion_tokens':16384,'max_retries':0}
    cfg['budget']['eval_task_tokens']=128000
    cfg['runtime']['choice_bounded_finalization']={'enabled':True,'version':c.FINALIZATION}
    return validate_config(cfg)


def smoke(root):
    root=Path(root);plan=checked(root);frozen=trial_bank(root)
    if not frozen['program_hashes'] or not plan['smoke_physical_keys']:
        result={'status':'completed_no_bindable_program','new_http':0};write_json(root/'smoke/result.json',result);return result
    rows=read(root/'source_catalog.json');tasks=[PublicTask(**r['task']) for r in rows if r['task']['physical_key'] in plan['smoke_physical_keys']]
    bank=Bank(root/'trial_bank',readonly=True,environment=read(root/'config.json')['program_environment'])
    try:
        from dataclasses import asdict
        tasks=[t for t in tasks if any(c.ready_bindings(asdict(t),p['entry_constraints']['operation_kind']) for p in bank.all('program'))]
    finally:bank.close()
    if not tasks:
        result={'status':'completed_no_bindable_program','new_http':0};write_json(root/'smoke/result.json',result);return result
    if not (root/'smoke/campaign_budget_prefix.json').exists():write_json(root/'smoke/campaign_budget_prefix.json',read(root/'budget.json'))
    owner=BudgetGovernor(root/'budget.json',**plan['settings']['cumulative_limits']);summaries={}
    governor=BatchGovernor(owner,{'new_limits':plan['settings']['stage_limits_proposed_not_authorized']['smoke']},root/'smoke')
    for arm in ('off','program'):
        cfg=smoke_config(root,arm);write_json(root/'smoke'/arm/'config.json',cfg)
        logger=FormalLog(root/'smoke'/arm,{'method':c.VERSION,'run_seed':42,'benchmark':'livemath','phase':'train_consumer_smoke'},cfg,
            resume=(root/'smoke'/arm/'run_manifest.json').exists())
        summaries[arm]=run(cfg,tasks,root/'smoke'/arm,readonly=True,resume=(root/'smoke'/arm/'execution_manifest.json').exists(),formal_log=logger,budget_governor=governor)
        logger.finish('completed' if summaries[arm]['complete'] else 'stopped')
        if tree_identity(root/'trial_bank')['sha256']!=frozen['tree_hash']:raise ValueError('P6 read-only smoke changed Bank')
        if not summaries[arm]['complete']:break
    result={'status':'completed' if len(summaries)==2 and all(s['complete'] for s in summaries.values()) else 'stopped',
        'arms':{a:{k:s[k] for k in ('tasks','successes','total_tokens')} for a,s in summaries.items()},'promotion':False}
    write_json(root/'smoke/result.json',result);return result


def report(root):
    root=Path(root);plan=checked(root);units=read(root/'units.json');rows=[]
    for unit in units:
        p=root/'learning'/unit['unit_id']/'learning.json'
        row=read(p) if p.exists() else {'state':'pending'}
        rows.append({'unit_id':unit['unit_id'],'state':row['state'],**{k:row.get('qualification',{}).get(k,False) for k in
            ('local_algorithm_checked','task_binding_checked','task_transfer_checked')}})
    prefix=read(root/'campaign_budget_prefix.json');ledger=read(root/'budget.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    result={'version':c.VERSION,'status':'prepared' if any(r['state']=='pending' for r in rows) else
        'completed_trial_candidates' if any(r['local_algorithm_checked'] for r in rows) else 'completed_no_candidate',
        'units':rows,'usable_programs':0,'new_http':len(new),'new_tokens_or_reservation':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in new.values()),
        'billing_blocked':billing_blocked(ledger),'val_test_executed':False}
    write_json(root/'learning_report.json',result);return result


def execute(output, env_file, *, paid_approved=False, phase='learning'):
    import fcntl
    root=Path(output)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate=check(root);plan=checked(root)
        if gate['status']!='passed' or not gate['billing_ready']:raise ValueError('P6 preflight/billing not ready')
        authority={'plan_hash':digest(plan),'limits':plan['settings']['stage_limits_proposed_not_authorized'],'val_test':False}
        path=root/'paid_authorization.json'
        if paid_approved:
            if path.exists() and read(path)!=authority:raise ValueError('P6 authorization identity changed')
            write_json(path,authority)
        if not path.exists() or read(path)!=authority:raise ValueError('P6 budget is proposed, not authorized')
        load_env(env_file)
        try:
            if phase in {'learning','all'}:learn(root)
            if phase in {'smoke','all'}:smoke(root)
            result=report(root);write_json(root/'result.json',result);return result
        except BaseException as exc:
            write_json(root/'result.json',{'status':'stopped','error_type':type(exc).__name__,'reason':str(exc),'at':utc()});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--config',default=str(CONTRACT))
    for name in ('check','report','execute'):
        p=sub.add_parser(name);p.add_argument('--output',required=True)
        if name=='execute':
            p.add_argument('--env-file',required=True);p.add_argument('--paid-approved',action='store_true')
            p.add_argument('--phase',choices=['learning','smoke','all'],default='learning')
    args=vars(parser.parse_args());name=args.pop('command');print(json.dumps(globals()[name](**args),ensure_ascii=False))


if __name__=='__main__':main()
