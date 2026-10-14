"""One independent, immutable human-authored Bank Test arm, with the official scorer."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from uuid import uuid4

from ..empirical.manual_reference import VERSION, LIMITS, ReferenceView
from ..empirical.bank import Bank
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import validate_config
from ..empirical.budget_governor import BudgetGovernor, billing_blocked
from ..harness.registry import create_simple_harness
from .canonical_manifest import sha256, verify
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, write_json, run, load_env
from .run_math_p3 import StageGovernor

STAGE_LIMITS = {'token_limit':13000000,'request_limit':5000}


def read(path): return json.loads(Path(path).read_text())


def prepare(package, preflight, source_plan, ledger, output):
    package,preflight,source_plan,ledger,root=map(lambda p:Path(p).resolve(),(package,preflight,source_plan,ledger,output))
    if root.exists():raise ValueError('Use a new manual-reference directory')
    code=code_identity()
    if not code['git_sha'] or code['tracked_dirty']:raise ValueError('Commit the evaluation implementation first')
    base=read(source_plan);source=read(package/'banks/livemath/bank.json')
    check_receipts=read(preflight/'docker_preflight.json')
    if (source['schema']!='manual_reference_bank.v2' or source['benchmark']!='livemath'
            or [len(source[k]) for k in ('skills','programs','implementations','workflows')]!=[12,6,7,5]
            or not check_receipts['passed'] or check_receipts['model_http']!=0):
        raise ValueError('Complete authored catalog and real Docker preflight required')
    authority=Path(base['authority']);verify(authority)
    tasks=base['tasks']['test'];canonical=read(authority/'livemath/test.json')['tasks']
    if len(tasks)!=100 or [t['task_id'] for t in tasks]!=[t['task_id'] for t in canonical] or any(t['split']!='test' for t in tasks):
        raise ValueError('Use the original canonical Test100 order')
    prefix=read(ledger)
    if prefix['limits']!={'token_limit':45000000,'request_limit':9165,'finish_reserve':0} or billing_blocked(prefix):
        raise ValueError('Incompatible or unresolved cumulative billing authority')
    charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in prefix['attempts'].values())
    if charged+STAGE_LIMITS['token_limit']>prefix['limits']['token_limit'] or len(prefix['attempts'])+STAGE_LIMITS['request_limit']>prefix['limits']['request_limit']:
        raise ValueError('Test stage exceeds remaining cumulative authority')
    root.mkdir(parents=True)
    cfg=deepcopy(base['configs']['test'])
    for k in list(cfg['experiment']):
        if k.startswith('choice_') and k not in {'choice_projection_version','choice_seed'}:cfg['experiment'].pop(k)
    cfg['experiment'].update(runtime_mode='frozen',output_dir=str(root/'test'),
        manual_reference_version=VERSION,manual_reference_limits=LIMITS,
        manual_reference_manifest=str(root/'reference_manifest.json'))
    cfg['data_dir']=str(root/'frozen_bank');cfg['llm']['max_retries']=0
    cfg['llm']['purpose_overrides']={k:v for k,v in cfg['llm']['purpose_overrides'].items() if k in {'finish_only','learning_structure_repair'}}
    for section in ('learning','runtime'):cfg[section]['choice_guidance']={'enabled':False,'public_process':False}
    for key in ('capability_material_version','capability_payload_max_bytes','source_program_payload_max_bytes'):
        cfg['learning'].pop(key,None)
    cfg['program_worker']['record_execution_receipt']=True
    cfg=validate_config(cfg)
    if cfg['program_environment']!=source['program_environment']:raise ValueError('Manual Worker environment differs')
    bank=Bank(root/'frozen_bank',environment=cfg['program_environment'])
    try:
        for kind,field in [('skill','skills'),('program','programs'),('implementation','implementations'),('workflow','workflows')]:
            for asset in source[field]:
                if bank.put(kind,asset)['id']!=asset['id']:raise ValueError('Manual asset identity changed on import')
        manifest={'version':VERSION,'bank_digest':bank.digest(),'qualification':'manual_reference_docker_checked_not_ours_usable',
            'program_hashes':{p['id']:digest(p) for p in bank.all('program')},'program_receipts':{p['id']:[] for p in bank.all('program')}}
    finally:bank.close()
    for ref in check_receipts['records']:
        path=Path(ref['path']);target=root/'qualification'/path.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        manifest['program_receipts'][ref['program_id']].append({'file':target.relative_to(root).as_posix(),'sha256':sha256(target),'role':ref['role']})
    write_json(root/'reference_manifest.json',manifest)
    frozen=tree_identity(root/'frozen_bank')
    original_inputs={str(p):sha256(p) for p in [source_plan,ledger,package/'banks/livemath/bank.json',package/'IMPORT_SOURCE.json',
        preflight/'docker_preflight.json',authority/'manifest.json',authority/'livemath/test.json']}
    original_inputs.update({p:h for p,h in base['input_hashes'].items() if '/livemath/' in p and 'train' not in p and 'val' not in p})
    for name,data in [('config.json',cfg),('campaign_budget_prefix.json',prefix),('budget.json',prefix),('test/campaign_budget_prefix.json',prefix)]:write_json(root/name,data)
    plan={'version':VERSION,'run_id':'manual_'+uuid4().hex,'code':code,'prepared_at':utc(),
        'source_package':str(package),'source_plan':str(source_plan),'input_hashes':original_inputs,
        'tasks':tasks,'test_limits':STAGE_LIMITS,'inherited_tokens':charged,'inherited_http':len(prefix['attempts']),
        'frozen_hash':frozen['sha256'],'reference_manifest_hash':sha256(root/'reference_manifest.json'),
        'config_hash':sha256(root/'config.json'),'test_development_history_disclosed':True,
        'bank_provenance':'human_authored_external_control_not_learned_ours',
        'initial_artifact_hash':frozen['sha256'],'split_hash':sha256(authority/'livemath/test.json')}
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    return check(root)


def check(output):
    root=Path(output);plan=read(root/'plan.json')
    if plan['version']!=VERSION or plan['code']!=code_identity() or digest(plan)!=read(root/'plan_identity.json')['sha256']:
        raise ValueError('Manual Test source/plan identity changed')
    for path,h in plan['input_hashes'].items():
        if sha256(Path(path))!=h:raise ValueError('Pinned input changed: '+path)
    if plan['config_hash']!=sha256(root/'config.json') or plan['reference_manifest_hash']!=sha256(root/'reference_manifest.json'):
        raise ValueError('Frozen manual evaluation configuration changed')
    if tree_identity(root/'frozen_bank')['sha256']!=plan['frozen_hash']:raise ValueError('Manual Bank changed')
    prefix=read(root/'campaign_budget_prefix.json');current=read(root/'budget.json')
    if (prefix!=read(root/'test/campaign_budget_prefix.json') or current['limits']!=prefix['limits']
            or current.get('inherited_reservation_recovery')!=prefix.get('inherited_reservation_recovery')
            or any(current['attempts'].get(k)!=v for k,v in prefix['attempts'].items())):
        raise ValueError('Cumulative inherited charges changed')
    cfg=validate_config(read(root/'config.json'))
    bank=Bank(root/'frozen_bank',readonly=True,environment=cfg['program_environment'])
    try:ReferenceView(bank,root/'reference_manifest.json')
    finally:bank.close()
    from .run_math_program_p5 import docker_ready
    if not docker_ready()['ready']:raise ValueError('Locked Linux Docker unavailable')
    return {'status':'passed','method':VERSION,'tasks':100,'assets':[12,6,7,5],
        'new_model_http':0,'billing_ready':not billing_blocked(current),'test_limits':plan['test_limits']}


def execute(output,env_file):
    import fcntl
    root=Path(output)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(root);plan=read(root/'plan.json');cfg=read(root/'config.json');load_env(env_file)
        owner=BudgetGovernor(root/'budget.json',**read(root/'campaign_budget_prefix.json')['limits'])
        governor=StageGovernor(owner,{'run_id':plan['run_id'],'new_limits':plan['test_limits']},root/'test')
        identity={'method':'manual_reference','benchmark':'livemath','run_seed':42,'split_seed':42,
            'source_commit':plan['code']['git_sha'],'source_sha256':plan['code']['source_sha256'],
            'model_name':cfg['llm']['model'],'provider_model_id':cfg['llm']['model'],
            'split_hashes':{'test':plan['split_hash']},'initial_artifact_hash':plan['frozen_hash'],
            'test_development_history_disclosed':True,'manual_reference_version':VERSION}
        observer=FormalLog(root,identity,{'test':cfg},resume=(root/'run_manifest.json').exists())
        adapter=create_simple_harness(cfg)
        try:
            summary=run(cfg,[PublicTask(**t) for t in plan['tasks']],root/'test',readonly=True,
                resume=(root/'test/execution_manifest.json').exists(),adapter=adapter,
                adapter_factory=lambda:create_simple_harness(cfg),formal_log=observer,budget_governor=governor)
            after=tree_identity(root/'frozen_bank')['sha256']
            if after!=plan['frozen_hash']:raise ValueError('Read-only manual Test changed Bank')
            write_json(root/'freeze_audit.json',{'before':plan['frozen_hash'],'after':after})
            result={'status':'completed' if summary['complete'] else 'stopped','tasks':summary['tasks'],
                'successes':summary['successes'],'tokens':summary['total_tokens'],'bank_provenance':plan['bank_provenance']}
            observer.finish(result['status']);write_json(root/'result.json',result);return result
        except BaseException as exc:
            observer.finish('stopped');write_json(root/'result.json',{'status':'stopped','error_type':type(exc).__name__,'reason':str(exc),'at':utc()});raise
        finally:adapter.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for key in ('package','preflight','source-plan','ledger','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('check');a.add_argument('--output',required=True)
    a=sub.add_parser('execute');a.add_argument('--output',required=True);a.add_argument('--env-file',required=True)
    args=vars(p.parse_args());command=args.pop('command');print(json.dumps(globals()[command](**args),ensure_ascii=False))


if __name__=='__main__':main()
