"""Versioned manual LiveMath V2: development off/on, then an immutable Test100 retest."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from uuid import uuid4

from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, billing_blocked, reserved_recovery
from ..empirical.contracts import PublicTask, digest
from ..empirical.manual_math_v2 import VERSION, LIMITS, FINALIZATION
from ..empirical.manual_reference import ReferenceView
from ..empirical.program_worker import ProgramWorker
from ..empirical.task_context import TaskContext
from ..empirical.system import validate_config
from ..harness.benchmarks import AnswerAdapter
from ..harness.simple_protocol import Broker
from .canonical_manifest import verify, sha256
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, write_json, run, load_env
from .run_math_p3 import StageGovernor

STAGES={'development':{'token_limit':768000,'request_limit':66},'test':{'token_limit':13000000,'request_limit':5000}}


class ReferenceStageGovernor(StageGovernor):
    def headroom(self, context):
        room=super().headroom(context)
        charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in self.attempts().values())
        stage_room=self.plan['new_limits']['token_limit']-charged-context.get('parent_finish_reserve',0)
        return {**room,'tokens':min(room['tokens'],stage_room)}

    def admit(self, audit_id, payload, context):
        from ..core.errors import BudgetExhausted
        if self.headroom(context)['requests']<=context.get('finish_request_reserve',0):
            raise BudgetExhausted('manual_stage_budget_exhausted','Stage preserves the bounded finish HTTP slot')
        return super().admit(audit_id,payload,context)


def read(path):return json.loads(Path(path).read_text())


def carryforward(paths):
    attempts, sources, recovery = {}, [], {}
    for path in paths:
        path=Path(path).resolve();ledger=read(path)
        if billing_blocked(ledger):raise ValueError('Unresolved new billing: '+str(path))
        reserved_recovery(ledger)
        for key,row in ledger['attempts'].items():
            if key in attempts and attempts[key]!=row:raise ValueError('Conflicting physical cost: '+key)
            attempts[key]=row
        if ledger.get('inherited_reservation_recovery'):
            receipt=ledger['inherited_reservation_recovery']
            if recovery and recovery!=receipt:raise ValueError('Reservation authority differs')
            recovery=deepcopy(receipt)
        sources.append({'path':str(path),'sha256':sha256(path),'attempts':len(ledger['attempts'])})
    prefix={'limits':{'token_limit':45000000,'request_limit':9165,'finish_reserve':0},
        'attempts':attempts,'unknown_billing':any(a['status']!='metered' for a in attempts.values())}
    if recovery:prefix['inherited_reservation_recovery']=recovery
    if billing_blocked(prefix):raise ValueError('Merged cost ancestry has unresolved billing')
    charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in attempts.values())
    if charged+sum(s['token_limit'] for s in STAGES.values())>45000000 or len(attempts)+sum(s['request_limit'] for s in STAGES.values())>9165:
        raise ValueError('Stage reservations exceed original cumulative authority')
    return prefix,{'sources':sources,'inherited_tokens':charged,'inherited_http':len(attempts),'accounting':'unique physical attempts; inherited unknown reservation retained once'}


def configure(base, root, phase, enabled=True):
    cfg=deepcopy(base)
    for key in list(cfg['experiment']):
        if key.startswith('choice_') and key not in {'choice_projection_version','choice_seed'}:cfg['experiment'].pop(key)
    cfg['experiment'].update(runtime_mode='frozen',output_dir=str(root/phase),manual_reference_version=VERSION,
        manual_reference_limits=LIMITS,manual_reference_manifest=str(root/'reference_manifest.json'),manual_reference_enabled=enabled)
    cfg['data_dir']=str(root/'frozen_bank');cfg['llm']['max_retries']=0
    cfg['llm']['purpose_overrides']={k:v for k,v in cfg['llm']['purpose_overrides'].items() if k in {'finish_only','learning_structure_repair'}}
    for section in ('learning','runtime'):cfg[section]['choice_guidance']={'enabled':False,'public_process':False}
    for key in ('capability_material_version','capability_payload_max_bytes','source_program_payload_max_bytes'):cfg['learning'].pop(key,None)
    cfg['runtime']['choice_bounded_finalization']={'enabled':True,'version':FINALIZATION}
    cfg['program_worker']['record_execution_receipt']=True
    return validate_config(cfg)


def new_cases(name):
    if name=='exact_integer':
        return [({'operation':'gcd','a':84,'b':30},{'value':6,'scope':'exact_supplied_integers_only'}),
                ({'operation':'mod','a':-7,'b':5},{'value':3,'scope':'exact_supplied_integers_only'}),
                ({'operation':'mod','a':7,'b':0},None)]
    return [({'matrix':[['1','2'],['3','4']]},{'determinant':'-2','dimension':2,'scope':'exact_supplied_matrix_only'}),
            ({'matrix':[['1/2','1/3'],['2','4']]},{'determinant':'4/3','dimension':2,'scope':'exact_supplied_matrix_only'}),
            ({'matrix':[['1','2']]},None)]


def qualify(package, preflight, root, cfg):
    source=read(package/'banks/livemath/bank.json');old=read(preflight/'docker_preflight.json')
    if source['method_revision']!=VERSION or source['program_environment']!=cfg['program_environment']:raise ValueError('Wrong authored package/environment')
    bank=Bank(root/'frozen_bank',environment=cfg['program_environment'])
    manifest={'version':VERSION,'program_hashes':{},'program_receipts':{},
        'capability_index':read(package/'banks/livemath/capability_index.json'),
        'theorem_registry':[json.loads(line) for line in (package/'banks/livemath/reference_knowledge/theorem_registry.jsonl').read_text().splitlines()],
        'qualification':'human_authored_docker_checked_not_ours_usable'}
    for theorem in manifest['theorem_registry']:
        path=(package/'banks/livemath/reference_knowledge'/theorem['source_file']).resolve()
        if not path.is_relative_to(package) or sha256(path)!=theorem['source_sha256'] or not theorem['assumptions']:
            raise ValueError('Theorem source identity or premises missing')
    try:
        for kind,key in [('skill','skills'),('program','programs'),('implementation','implementations'),('workflow','workflows')]:
            for asset in source[key]:
                if bank.put(kind,asset)['id']!=asset['id']:raise ValueError('Authored import changed identity')
        worker=ProgramWorker(cfg['program_worker'])
        for p in bank.all('program'):
            cases=[read(r['path']) for r in old['records'] if r['program_id']==p['id']]
            if not cases:
                name=next(n for n in ('exact_integer','exact_matrix') if (package/'programs/livemath'/(n+'.py')).read_text()==p['source'])
                cases=[{'role':role,'arguments':a,'expected':v} for role,(a,v) in zip(('positive','variation','invalid'),new_cases(name))]
            manifest['program_hashes'][p['id']]=digest(p);refs=manifest['program_receipts'][p['id']]=[]
            for c in cases:
                task=PublicTask('qualification',digest(c['arguments']),'Local contract check',c['arguments'])
                adapter=AnswerAdapter('livemath',{task.task_id:{}});adapter.reset(task)
                broker=Broker(adapter,1,context=TaskContext(cfg['runtime']))
                try:result=worker.execute(p,c['arguments'],broker)
                finally:adapter.close()
                receipt=result.get('execution_receipt',{})
                passed=(result['status']==('needs_input' if c['expected'] is None else 'ok')
                    and (c['expected'] is None or result.get('outputs')==c['expected']) and receipt.get('exit_code')==0
                    and '--network=none' in receipt.get('command',[]) and not broker.events)
                path=root/'qualification'/(digest([p['id'],c['role']])+'.json')
                write_json(path,{'program_id':p['id'],'role':c['role'],'arguments':c['arguments'],'expected':c['expected'],
                    'result':result,'rpc_calls':broker.events,'passed':passed})
                if not passed:raise ValueError('Docker qualification failed: '+str(path))
                refs.append({'file':path.relative_to(root).as_posix(),'sha256':sha256(path),'role':c['role']})
        if any(bank.program_eligible(p) for p in bank.all('program')):raise ValueError('Imported assets must not forge Ours usability')
        manifest['bank_digest']=bank.digest()
    finally:bank.close()
    write_json(root/'reference_manifest.json',manifest)


def prepare(package, preflight, source_plan, ledgers, output):
    package,preflight,source_plan,root=map(lambda p:Path(p).resolve(),(package,preflight,source_plan,output))
    if root.exists():raise ValueError('Use a new immutable V2 run directory')
    code=code_identity()
    if not code['git_sha'] or code['tracked_dirty']:raise ValueError('Commit the implementation before prepare')
    base=read(source_plan);authority=Path(base['authority']);verify(authority)
    for phase,count in [('val',17),('test',100)]:
        if len(base['tasks'][phase])!=count or [t['task_id'] for t in base['tasks'][phase]]!=[t['task_id'] for t in read(authority/'livemath'/(phase+'.json'))['tasks']]:raise ValueError('Canonical task identity changed')
    sums=read(package/'SHA256SUMS.json')
    for name,h in sums.items():
        path=(package/name).resolve()
        if not path.is_relative_to(package) or sha256(path)!=h:raise ValueError('Authored input hash changed')
    prefix,ancestry=carryforward(ledgers)
    root.mkdir(parents=True);shutil.copytree(package,root/'input')
    configs={phase:configure(base['configs']['val' if phase.startswith('development') else 'test'],root,phase,enabled=phase!='development/off')
        for phase in ('development/off','development/on','test')}
    qualify(root/'input',preflight,root,configs['test'])
    for phase,cfg in configs.items():write_json(root/phase/'config.json',cfg)
    for path in ('budget.json','campaign_budget_prefix.json','development/campaign_budget_prefix.json','test/campaign_budget_prefix.json'):write_json(root/path,prefix)
    write_json(root/'cost_carryforward_manifest.json',ancestry)
    tasks={'development':base['tasks']['val'][:3],'test':base['tasks']['test']}
    write_json(root/'tasks.json',tasks)
    inputs={str(source_plan):sha256(source_plan),str(authority/'manifest.json'):sha256(authority/'manifest.json')}
    inputs.update({r['path']:r['sha256'] for r in ancestry['sources']})
    inputs.update({p:h for p,h in base['input_hashes'].items() if '/livemath/' in p and ('test' in p or 'val' in p)})
    plan={'version':VERSION,'code':code,'run_id':'manual_math_v2_'+uuid4().hex,'created_at':utc(),
        'files':inputs,'configs':{p:sha256(root/p/'config.json') for p in configs},'task_hash':sha256(root/'tasks.json'),
        'frozen':tree_identity(root/'frozen_bank'),'input_identity':tree_identity(root/'input'),
        'manifest_hash':sha256(root/'reference_manifest.json'),'prefix_hash':sha256(root/'campaign_budget_prefix.json'),
        'stage_limits':STAGES,'test_development_history_disclosed':True,'bank_provenance':'human_authored_external_reference_not_ours',
        'development_selection':'canonical Val first three, both arms; no performance-based editing or selection'}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    return check(root)


def check(output):
    root=Path(output);plan=read(root/'plan.json')
    if plan['version']!=VERSION or plan['code']!=code_identity() or digest(plan)!=read(root/'plan_identity.json')['sha256']:raise ValueError('Frozen V2 source/plan changed')
    for path,h in plan['files'].items():
        if sha256(Path(path))!=h:raise ValueError('Pinned input changed: '+path)
    for phase,h in plan['configs'].items():
        if sha256(root/phase/'config.json')!=h:raise ValueError('Runtime config changed')
        validate_config(read(root/phase/'config.json'))
    if (tree_identity(root/'frozen_bank')!=plan['frozen'] or tree_identity(root/'input')!=plan['input_identity']
            or sha256(root/'reference_manifest.json')!=plan['manifest_hash'] or sha256(root/'tasks.json')!=plan['task_hash']
            or sha256(root/'campaign_budget_prefix.json')!=plan['prefix_hash']):raise ValueError('Frozen V2 inputs changed')
    prefix=read(root/'campaign_budget_prefix.json');ledger=read(root/'budget.json')
    if ledger['limits']!=prefix['limits'] or ledger.get('inherited_reservation_recovery')!=prefix.get('inherited_reservation_recovery') or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):raise ValueError('Cost ancestry changed')
    cfg=read(root/'test/config.json');bank=Bank(root/'frozen_bank',readonly=True,environment=cfg['program_environment'])
    try:ReferenceView(bank,root/'reference_manifest.json')
    finally:bank.close()
    return {'status':'passed','version':VERSION,'new_model_http':0,'billing_ready':not billing_blocked(ledger)}


def evaluate(root,phase,owner,plan):
    cfg=read(root/phase/'config.json');stage=phase.split('/')[0]
    governor=ReferenceStageGovernor(owner,{'run_id':plan['run_id'],'new_limits':plan['stage_limits'][stage]},root/stage)
    identity={'method':VERSION,'benchmark':'livemath','run_seed':42,'source_commit':plan['code']['git_sha'],
        'source_sha256':plan['code']['source_sha256'],'manual_reference_enabled':cfg['experiment']['manual_reference_enabled'],
        'bank_provenance':plan['bank_provenance'],'test_development_history_disclosed':True}
    observer=FormalLog(root/phase,identity,cfg,resume=(root/phase/'run_manifest.json').exists())
    tasks=[PublicTask(**t) for t in read(root/'tasks.json')[stage]]
    result=run(cfg,tasks,root/phase,readonly=True,resume=(root/phase/'execution_manifest.json').exists(),formal_log=observer,budget_governor=governor)
    observer.finish('completed' if result['complete'] else 'stopped')
    if tree_identity(root/'frozen_bank')!=plan['frozen']:raise RuntimeError('Read-only phase changed the Bank')
    return result


def execute(output,env_file,phase='all'):
    import fcntl
    root=Path(output)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(root);plan=read(root/'plan.json');load_env(env_file)
        owner=BudgetGovernor(root/'budget.json',**read(root/'budget.json')['limits'])
        try:
            if phase in {'all','development'}:
                summaries={}
                for arm in ('off','on'):
                    summaries[arm]=evaluate(root,'development/'+arm,owner,plan)
                    if not summaries[arm]['complete']:
                        result={'status':'stopped','phase':'development/'+arm}
                        write_json(root/'development/result.json',result);return result
                pairs=[]
                for off,on in zip(summaries['off']['cases'],summaries['on']['cases']):
                    pairs.append({'task_id':off['task']['task_id'],'off':off['score'],'on':on['score'],
                        'off_execution':off['execution'],'on_execution':on['execution']})
                complete=all(s['complete'] for s in summaries.values())
                write_json(root/'development/result.json',{'status':'completed' if complete else 'stopped','pairs':pairs,
                    'tokens':{a:s['total_tokens'] for a,s in summaries.items()},'selection_or_tuning':False})
                if not complete:return {'status':'stopped','phase':'development'}
            if phase in {'all','test'}:
                if read(root/'development/result.json')['status']!='completed':raise ValueError('Complete the frozen development comparison first')
                summary=evaluate(root,'test',owner,plan)
                result={'status':'completed' if summary['complete'] else 'stopped',**{k:summary[k] for k in ('tasks','successes','total_tokens')}}
                write_json(root/'result.json',result);return result
            return {'status':'completed','phase':'development'}
        except BaseException as exc:
            write_json(root/'result.json',{'status':'stopped','error_type':type(exc).__name__,'reason':str(exc),'at':utc()});raise
        finally:write_json(root/'freeze_audit.json',{'before':plan['frozen'],'after':tree_identity(root/'frozen_bank')})


def status(output):
    root=Path(output);out={'output':str(root),'version':VERSION}
    for phase in ('development/off','development/on','test'):
        path=root/phase/'summary.json'
        out[phase]={k:v for k,v in read(path).items() if k in {'tasks','successes','total_tokens','complete'}} if path.exists() else {'tasks':0}
    if (root/'result.json').exists():out['result']=read(root/'result.json')
    ledger=read(root/'budget.json');prefix=read(root/'campaign_budget_prefix.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    out['new_usage']={'tokens_or_reservation':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in new.values()),'physical_http':len(new),'pending':sum(a['status']!='metered' for a in new.values())}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for key in ('package','preflight','source-plan','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--ledgers',nargs='+',required=True)
    for name in ('check','status','execute'):
        a=sub.add_parser(name);a.add_argument('--output',required=True)
        if name=='execute':
            a.add_argument('--env-file',required=True);a.add_argument('--phase',choices=['all','development','test'],default='all')
    args=vars(p.parse_args());command=args.pop('command');print(json.dumps(globals()[command](**args),ensure_ascii=False))


if __name__=='__main__':main()
