"""Independent authored SearchQA Bank: zero-HTTP qualification, then canonical Test1400."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.contracts import PublicTask, digest
from ..empirical.manual_reference import SEARCHQA_VERSION as VERSION, SEARCHQA_LIMITS as LIMITS, ReferenceView
from ..empirical.program_worker import ProgramWorker
from ..empirical.system import validate_config
from ..empirical.task_context import TaskContext
from ..harness.benchmarks import AnswerAdapter
from ..harness.simple_protocol import Broker
from .canonical_manifest import sha256, verify
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, run, write_json
from .searchqa_reference_oracles import cases

TEST_TOKENS = 1400 * 96000
# Eight outer decisions, at most one dynamic graph node and one budget finish;
# each physical request retains the existing four transport retries.
TEST_HTTP = 1400 * 10 * 5


def read(path): return json.loads(Path(path).read_text())


def configuration(base, root, datasets):
    cfg = deepcopy(base)
    for section in ('experiment','runtime','learning'):
        for key in list(cfg[section]):
            if key.startswith('searchqa_'): cfg[section].pop(key)
    cfg['experiment'].update(runtime_mode='frozen',output_dir=str(root/'test'),
        manual_reference_version=VERSION,manual_reference_limits=LIMITS,
        manual_reference_manifest=str(root/'reference_manifest.json'),
        asset_origin='human_authored_not_naturally_learned')
    cfg['experiment']['public_materialization_sha256'] = sha256(datasets/'test.json')
    cfg['manifest'] = str(Path(base['manifest']).with_name('test.json'))
    cfg['data_dir'] = str(root/'frozen_bank')
    cfg['harness']['evaluator_records'] = str(datasets/'evaluator_records_test.json')
    cfg['program_worker']['record_execution_receipt'] = True
    cfg['llm']['max_retries'] = 4
    return validate_config(cfg)


def qualify(package, root, cfg):
    data = read(package/'bank/bank.json')
    if data['schema'] != 'human-searchqa.reference-bank.v1' or data['program_environment'] != cfg['program_environment']:
        raise ValueError('Incompatible supplied Bank/environment')
    counts = {key:len(data[key]) for key in ('skills','programs','implementations','workflows')}
    if list(counts.values()) != [12,8,8,5]: raise ValueError('Incomplete supplied catalog')
    sources = {p.read_text():p.stem for p in (package/'programs').glob('*.py')}
    bank = Bank(root/'frozen_bank',seed=42,environment=cfg['program_environment'])
    manifest = {'version':VERSION,'program_hashes':{},'program_receipts':{},
        'qualification':'human_authored_synthetic_docker_checked_not_ours_usable'}
    try:
        for kind,key in [('skill','skills'),('program','programs'),('implementation','implementations'),('workflow','workflows')]:
            for asset in data[key]:
                if kind=='program' and (asset['state']!='candidate' or asset.get('local_validation') or asset['source'] not in sources):
                    raise ValueError('Manual import must not invent Ours qualification')
                if bank.put(kind,asset)['id']!=asset['id']: raise ValueError('Import changed supplied ID')
        worker = ProgramWorker(cfg['program_worker'])
        for program in bank.all('program'):
            name = sources[program['source']]
            manifest['program_hashes'][program['id']] = digest(program)
            refs = manifest['program_receipts'][program['id']] = []
            for case in [c for c in cases() if c['name']==name]:
                task = PublicTask('reference:'+name+':'+case['role'],digest(case),'Synthetic contract check',case['inputs'],'reference_probe')
                adapter = AnswerAdapter('searchqa',{task.task_id:{}});adapter.reset(task)
                broker = Broker(adapter,1,context=TaskContext(cfg['runtime']))
                try: result = worker.execute(program,case['inputs'],broker)
                finally: adapter.close()
                receipt = result.get('execution_receipt',{})
                passed = (result['status']==case['expected_status'] and result.get('outputs',{})==case['expected']
                    and receipt.get('exit_code')==0 and receipt.get('image_digest')==program['environment']['image_digest']
                    and '--network=none' in receipt.get('command',[]) and not broker.events)
                path = root/'qualification'/(name+'_'+case['role']+'.json')
                write_json(path,{**case,'program_id':program['id'],'result':result,'rpc_calls':broker.events,'passed':passed})
                if not passed: raise ValueError('Real Worker qualification failed: '+str(path))
                refs.append({'role':case['role'],'file':path.relative_to(root).as_posix(),'sha256':sha256(path)})
        if any(bank.program_eligible(p) for p in bank.all('program')): raise ValueError('Manual import contaminated Ours qualification')
        manifest['bank_digest'] = bank.digest()
    finally: bank.close()
    write_json(root/'reference_manifest.json',manifest)
    return counts


def prepare(package, parent, datasets, output):
    package,parent,datasets,root = [Path(p).resolve() for p in (package,parent,datasets,output)]
    if root.exists(): raise ValueError('Use a new independent output directory')
    code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Use a clean committed implementation')
    sums = read(package/'SHA256SUMS.json')
    for name,expected in sums.items():
        path = (package/name).resolve()
        if not path.is_relative_to(package) or sha256(path)!=expected: raise ValueError('Package hash mismatch: '+name)
    cfg = configuration(read(parent/'resolved_config.json'),root,datasets)
    canonical = Path(cfg['manifest']);verify(canonical.parent.parent)
    official = read(canonical)['tasks'];public = read(datasets/'test.json');tasks = public['tasks']
    if (len(tasks)!=1400 or [r['task_id'] for r in tasks]!=[r['task_id'] for r in official]
            or len({r['physical_key'] for r in tasks})!=1400 or public['split_sha256']!=sha256(canonical)
            or public['authority_sha256']!=sha256(canonical.parent.parent/'manifest.json')
            or any(r['split']!='test' for r in tasks)):
        raise ValueError('Canonical Test1400 identity/order changed')
    prefix = read(parent/'budget.json')
    if prefix['unknown_billing'] or any(a['status']!='metered' for a in prefix['attempts'].values()):
        raise ValueError('Resolve existing billing before Test')
    cost = sum(a['accounted_tokens'] for a in prefix['attempts'].values())
    limits = dict(token_limit=cost+TEST_TOKENS,request_limit=len(prefix['attempts'])+TEST_HTTP,finish_reserve=0)
    if any(limits[k]>prefix['limits'][k] for k in ('token_limit','request_limit')):
        raise ValueError('Independent Test exceeds existing cumulative authorization')
    root.mkdir(parents=True)
    shutil.copytree(package,root/'input',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    counts = qualify(root/'input',root,cfg)
    ledger = {**deepcopy(prefix),'limits':limits};cfg['budget'].update(limits)
    for name,value in [('config.json',cfg),('tasks.json',{'tasks':tasks}),('budget.json',ledger),
                       ('inherited_budget.json',prefix)]: write_json(root/name,value)
    write_json(root/'cost_carryforward_manifest.json',{'parent':str(parent),'parent_ledger_sha256':sha256(parent/'budget.json'),
        'historical_tokens':cost,'historical_http':len(prefix['attempts']),'test_tokens':TEST_TOKENS,'test_http':TEST_HTTP,
        'parent_limits':prefix['limits'],'child_limits':limits,'accounting':'Historical costs once plus this child delta; never sum inherited totals.'})
    paths = [root/name for name in ('config.json','tasks.json','reference_manifest.json','inherited_budget.json','cost_carryforward_manifest.json')]
    paths += [canonical,canonical.parent.parent/'manifest.json',datasets/'test.json',Path(cfg['harness']['evaluator_records']),parent/'budget.json']
    plan = dict(version=VERSION,code=code,created_at=utc(),files={str(p):sha256(p) for p in paths},
        frozen=tree_identity(root/'frozen_bank'),input_identity=tree_identity(root/'input'),counts=counts,
        test_tasks=1400,test_limits={'tokens':TEST_TOKENS,'http':TEST_HTTP},
        provenance='human_authored_external_reference_not_learned_ours',
        user_authorization='2026-10-15: 将这个bank接入再单独做一次test。',stop_after='canonical_test1400')
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    return check(root)


def check(output):
    root = Path(output);plan = read(root/'plan.json')
    if digest(plan)!=read(root/'plan_identity.json')['sha256'] or plan['code']!=code_identity():
        raise ValueError('Code/plan changed; no cross-version resume')
    for path,expected in plan['files'].items():
        if sha256(Path(path))!=expected: raise ValueError('Pinned input changed: '+path)
    if tree_identity(root/'input')!=plan['input_identity'] or tree_identity(root/'frozen_bank')!=plan['frozen']:
        raise ValueError('Manual Bank/input changed')
    ledger = read(root/'budget.json');prefix = read(root/'inherited_budget.json')
    if ledger['limits']!=read(root/'cost_carryforward_manifest.json')['child_limits'] or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('Cumulative billing identity changed')
    cfg=validate_config(read(root/'config.json'))
    bank=Bank(root/'frozen_bank',readonly=True,environment=cfg['program_environment'])
    try: ReferenceView(bank,root/'reference_manifest.json')
    finally: bank.close()
    from .run_math_program_p5 import docker_ready
    if not docker_ready()['ready']: raise ValueError('Locked Docker unavailable')
    return {'status':'passed','version':VERSION,'http':0,'tasks':1400,'counts':plan['counts'],'test_limits':plan['test_limits']}


def evaluate(output, env_file):
    import fcntl
    root=Path(output)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(root);load_env(env_file)
        plan=read(root/'plan.json');cfg=read(root/'config.json')
        if (root/'result.json').exists() and read(root/'result.json')['status']=='completed':
            return read(root/'result.json')
        tasks=[PublicTask(**row) for row in read(root/'tasks.json')['tasks']]
        log=FormalLog(root,{'method':VERSION,'benchmark':'searchqa','run_seed':42,
            'source_commit':plan['code']['git_sha'],'provenance':plan['provenance'],
            'frozen_hash':plan['frozen']['sha256'],'plan_hash':digest(plan)},cfg,
            resume=(root/'run_manifest.json').exists())
        governor=BudgetGovernor(root/'budget.json',**read(root/'budget.json')['limits'])
        try:
            summary=run(cfg,tasks,root/'test',readonly=True,resume=(root/'test/execution_manifest.json').exists(),
                formal_log=log,budget_governor=governor,
                task_metadata={t['task_id']:t['task_metadata'] for t in read(cfg['manifest'])['tasks']})
            result={'status':'completed' if summary['complete'] else 'stopped',
                **{k:v for k,v in summary.items() if k!='cases'},'provenance':plan['provenance']}
            write_json(root/'result.json',result);log.finish(result['status']);return result
        except BaseException as exc:
            log.finish('stopped');write_json(root/'result.json',{'status':'stopped','at':utc(),'type':type(exc).__name__,'reason':str(exc)});raise
        finally:
            after=tree_identity(root/'frozen_bank')
            write_json(root/'freeze_audit.json',{'before':plan['frozen']['sha256'],'after':after['sha256'],'unchanged':after==plan['frozen']})
            if after!=plan['frozen']: raise RuntimeError('Test modified frozen Bank')


def status(output):
    import sqlite3
    root=Path(output);rows=[]
    if (root/'test/run.sqlite3').exists():
        with sqlite3.connect('file:'+str(root/'test/run.sqlite3')+'?mode=ro',uri=True) as db:
            rows=db.execute('SELECT id,status,result FROM tasks').fetchall()
    finished=[json.loads(r[2]) for r in rows if r[1]=='completed']
    ledger=read(root/'budget.json');prefix=read(root/'inherited_budget.json')
    attempts=[a for key,a in ledger['attempts'].items() if key not in prefix['attempts']]
    correct=sum(bool(r['score']['hard']) for r in finished)
    return {'completed':len(finished),'total':1400,'correct':correct,
        'em':correct/len(finished) if finished else None,
        'f1':sum(r['score']['soft'] for r in finished)/len(finished) if finished else None,
        'active_tasks':[r[0] for r in rows if r[1]=='running'],
        'new_metered_tokens':sum(a.get('accounted_tokens',0) for a in attempts),
        'new_http':len(attempts),'billing_states':{s:sum(a['status']==s for a in attempts) for s in {a['status'] for a in attempts}},
        'outcome':read(root/'result.json').get('status') if (root/'result.json').exists() else 'in_progress',
        'last_log_update':(root/'run.log').stat().st_mtime if (root/'run.log').exists() else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('prepare')
    for key in ('package','parent','datasets','output'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('check');a.add_argument('--output',required=True)
    a=sub.add_parser('status');a.add_argument('--output',required=True)
    a=sub.add_parser('test');a.add_argument('--output',required=True);a.add_argument('--env-file',required=True)
    args=vars(p.parse_args());stage=args.pop('stage')
    print(json.dumps((evaluate if stage=='test' else globals()[stage])(**args),ensure_ascii=False))


if __name__=='__main__':main()
