"""Independent manual Programmatic Bank: fixed Train4 smoke then canonical Test100."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

from ..empirical import human_frontier_runtime as rt
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, billing_blocked, input_token_bound
from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, digest, program_digest, validate_workflow
from ..harness.benchmarks import AnswerAdapter
from .canonical_manifest import sha256
from .formal_log import tree_identity, utc
from .run_empirical import code_identity, load_env
from .run_human_frontier_r2 import task_rows, governor, event, LIMITS

VERSION = 'human.frontier.r31.campaign.v1'
PACKAGE = Path(__file__).resolve().parents[3] / 'resources/human_frontier_r31'
PHASES = {'development': {'token_limit': 512000, 'request_limit': 8},
          'test': {'token_limit': 8000000, 'request_limit': 200}}
read = rt.read


def bridge(package):
    """Import the exact frozen integration without writing into its directory."""
    root = Path(package).resolve()
    name = '_human_r31_' + digest(str(root))[:16]
    module = types.ModuleType(name); module.__path__ = [str(root/'integration')]
    sys.modules[name] = module
    for part in ('native_programmatic_runtime', 'episode_bridge'):
        path = root/'integration'/f'{part}.py'
        spec = importlib.util.spec_from_file_location(name+'.'+part, path)
        child = importlib.util.module_from_spec(spec); sys.modules[spec.name] = child
        exec(compile(path.read_bytes(), str(path), 'exec'), child.__dict__)
        setattr(module, part, child)
    return module.episode_bridge


def import_bank(package, destination):
    bundle = read(Path(package)/'bank.json')
    if Path(destination).exists():
        raise ValueError('Manual import requires a new directory')
    bank = Bank(destination, seed=42, environment=bundle['program_environment'])
    try:
        for kind in ('skill', 'program', 'implementation', 'workflow'):
            for asset in bundle[kind]:
                if kind == 'workflow':
                    validate_workflow(asset, bank=bank)
                saved = bank.put(kind, deepcopy(asset))
                if kind == 'program' and saved['id'] != 'program_' + program_digest(asset):
                    raise ValueError('Authored Program identity changed')
        if {k:len(bank.all(k)) for k in bundle['counts']} != bundle['counts']:
            raise ValueError('Missing authored assets')
        with bank.db:
            bank.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('manual_authority', bundle['method_version']))
        result = {'asset_counts': bundle['counts'], 'bank_digest': bank.digest(),
                  'source_bank_hash': sha256(Path(package)/'bank.json'), 'manual_permission': True,
                  'qualification': 'external_authored_not_learned_ours'}
    finally:
        bank.close()
    save_json(Path(destination)/'manual_import_receipt.json', result)
    return result


def selection(train):
    rank = lambda t: digest(['human-r31-train4-seed42', t['physical_key']])
    chosen = []
    for meta in (True, False):
        group = [t for t in train if (len(rt.meta_labels(PublicTask(**t))) == 1) == meta]
        if len(group) < 2:
            raise ValueError('Need two public tasks for each development branch')
        chosen += sorted(group, key=rank)[:2]
    return sorted(chosen, key=rank)


def episode_dir(root, row):
    return Path(root)/row['phase']/row['task']['physical_key']


def prepare(root, previous):
    root = Path(root).resolve(); previous = Path(previous).resolve()
    code = code_identity()
    if root.exists() or code['tracked_dirty'] or not code['git_sha']:
        raise ValueError('Prepare from a committed source into a new directory')
    old = read(previous/'plan.json'); old_result = read(previous/'result.json')
    if old_result['status'] != 'completed_test':
        raise ValueError('Predecessor must be completed before carrying its ledger')
    datasets = Path(old['data_root']); prefix = read(previous/'budget/ledger.json')
    if prefix['limits'] != LIMITS or billing_blocked(prefix):
        raise ValueError('Cumulative budget identity or unresolved billing')
    used = sum(v.get('accounted_tokens', v['reserved_tokens']) for v in prefix['attempts'].values())
    if used + sum(p['token_limit'] for p in PHASES.values()) > LIMITS['token_limit']:
        raise ValueError('Stage envelopes exceed cumulative token authorization')
    if len(prefix['attempts']) + sum(p['request_limit'] for p in PHASES.values()) > LIMITS['request_limit']:
        raise ValueError('Stage envelopes exceed cumulative HTTP authorization')
    splits = {s:task_rows(datasets/f'{s}.json') for s in ('train','val','test')}
    if [len(splits[s]) for s in ('train','val','test')] != [60,17,100]:
        raise ValueError('Canonical count changed')
    if len({t['physical_key'] for rows in splits.values() for t in rows}) != 177:
        raise ValueError('Canonical split overlap')
    picked = selection(splits['train'])
    schedule = [{'phase':phase, 'task':task, 'task_order_index':i} for phase,rows in
                [('development',picked),('test',splits['test'])] for i,task in enumerate(rows)]
    root.mkdir(parents=True)
    shutil.copytree(PACKAGE, root/'bundle', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assets = import_bank(root/'bundle', root/'bank')
    cfg = read(previous/'config.json')
    cfg['programmatic_runtime'] = {'variant':'benchmark_prior','graph':True,'manual_programs':True}
    for name,data in [('config.json',cfg),('schedule.json',schedule),('selection.json',picked),
                      ('budget/prefix.json',prefix),('budget/ledger.json',prefix)]:
        save_json(root/name, data)
    files = [previous/'plan.json',previous/'result.json',previous/'budget/ledger.json',
             *[datasets/n for n in ['train.json','val.json','test.json','evaluator_records_train.json','evaluator_records_test.json']]]
    plan = {'schema':VERSION,'created_at':utc(),'code':code,'previous_run':str(previous),
            'data_root':str(datasets),'model':cfg['llm']['model'],'seed':42,'variant':'benchmark_prior',
            'runtime':{**rt.RUNTIME,'programs':'author_trusted_docker','graph':True,
                       'workflow_execution':'programmatic'},
            'dynamic_subsolver_runtime':rt.RUNTIME,'phase_limits':PHASES,'cumulative_limits':LIMITS,
            'historical_tokens_or_reserved':used,'historical_http':len(prefix['attempts']),
            'source_integration':str(root/'bundle/integration/episode_bridge.py'),
            'asset_import':assets,'bank_tree':tree_identity(root/'bank'),'bundle_tree':tree_identity(root/'bundle'),
            'files':{str(p):sha256(p) for p in files},
            'prepared_hashes':{n:sha256(root/n) for n in ['config.json','schedule.json','selection.json','budget/prefix.json']},
            'expected_episodes':104,'development_tasks':4,'test_tasks':100,
            'normal_observation_required':{'development':4,'test':20},
            'baseline':'R2 historical reference; protocol differs, no new control arm',
            'test_developmental_retest':True,'external_authored_not_learned_ours':True,
            'paid_authorization':'User 2026-10-17: 接入新bank重新做实验，文档供参考; existing cumulative 45M/9165 limit retained',
            'cost_projection':'R2 Test 2.78M tokens; 8M fixed envelope, not 100x128K worst-case guarantee'}
    plan['run_id'] = 'human_frontier_r31_' + digest(plan)[:24]
    save_json(root/'plan.json',plan);save_json(root/'plan_identity.json',{'hash':digest(plan)})
    save_json(root/'paid_authorization.json',{'approved':True,'plan_hash':digest(plan),'phase_limits':PHASES,
              'cumulative_limits':LIMITS,'basis':plan['paid_authorization']})
    progress(root,status='prepared')
    return {'status':'prepared','run_id':plan['run_id'],'historical_tokens_or_reserved':used}


def checked(root):
    root = Path(root); plan = read(root/'plan.json')
    if plan['schema'] != VERSION or plan['code'] != code_identity() or digest(plan) != read(root/'plan_identity.json')['hash']:
        raise ValueError('Code or plan identity changed; use a new explicit revision')
    for filename,h in plan['files'].items():
        if sha256(Path(filename)) != h:raise ValueError('Frozen source changed: '+filename)
    for name,h in plan['prepared_hashes'].items():
        if sha256(root/name) != h:raise ValueError('Prepared input changed: '+name)
    if tree_identity(root/'bank') != plan['bank_tree'] or tree_identity(root/'bundle') != plan['bundle_tree']:
        raise ValueError('Frozen bank or integration changed')
    prefix=read(root/'budget/prefix.json'); ledger=read(root/'budget/ledger.json'); auth=read(root/'paid_authorization.json')
    if ledger['limits'] != LIMITS or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('Historical billing changed')
    if not auth['approved'] or auth['plan_hash'] != digest(plan) or auth['phase_limits'] != plan['phase_limits']:
        raise ValueError('Authorization differs from frozen plan')
    return plan


def check(root):
    root = Path(root); plan = checked(root); cfg = read(root/'config.json'); module = bridge(root/'bundle')
    bank = Bank(root/'bank',readonly=True); bundle=read(root/'bundle/bank.json'); receipts=[]
    try:
        # Actual Docker across every public task: routing has no scorer/gold access.
        for row in read(root/'schedule.json'):
            task=PublicTask(**row['task']); adapter=AnswerAdapter('livemath',{task.task_id:{}}); adapter.reset(task)
            stage_dir=root/'preflight_programs'/row['phase']/task.physical_key
            graph=module.run_programmatic_graph(task,bank,variant=plan['variant'],
                run_worker=module.docker_worker(adapter),stage_dir=stage_dir)
            if any(n['status']!='ok' or not n['result'].get('execution_receipt') or n['result']['execution_receipt']['exit_code']!=0
                   for n in graph['program_nodes'] if n['mode']=='program'):
                raise ValueError('Actual Docker preflight failed: '+task.task_id)
            policy=module.DynamicPolicyView(bank,bundle,graph.get('handoff') or {},plan['variant']).policy(task)
            with tempfile.TemporaryDirectory() as tmp:
                g=BudgetGovernor(Path(tmp)/'ledger.json',token_limit=128000,request_limit=2,finish_reserve=0)
                ctx={'parent_scope':tmp,'parent_token_limit':128000,'episode_scope':tmp,'episode_request_limit':2}
                wire=rt.provider(cfg['llm'])._build_payload(rt.messages(task,policy),[])
                g.admit('fixture_main',wire,ctx)
                g.complete({'request_id':'fixture_main','raw_usage':{'total_tokens':g.state['attempts']['fixture_main']['reserved_tokens']},'outcome':'offline_fixture'})
                end=rt.provider(cfg['llm'],True)._build_payload(rt.messages(task,policy,{'text':''}),[])
                g.admit('fixture_finish',end,{**ctx,'purpose':'finish_only'})
            receipts.append({'phase':row['phase'],'task_id':task.task_id,'branch':graph['branch'],
                             'programs':graph['program_invocations_total'],'provider_payloads':2,'budget_admitted':True})
            print(json.dumps({'preflight':len(receipts),'total':104,'branch':graph['branch']}),flush=True)
            adapter.close()
    finally:
        bank.close()
    result={'status':'passed','code':plan['code'],'new_model_http':0,'episodes':receipts,
            'bank_unchanged':tree_identity(root/'bank')==plan['bank_tree']}
    save_json(root/'preflight.json',result)
    return {'status':'passed','episodes':len(receipts),'docker_invocations':sum(r['programs'] for r in receipts),'new_model_http':0}


def progress(root, **change):
    root=Path(root);path=root/'progress.json'; old=read(path) if path.exists() else {}
    stats={}; last=None; following=None
    for row in read(root/'schedule.json'):
        s=stats.setdefault(row['phase'],{'scored':0,'normal':0,'correct':0,'empty':0,'tokens':0,'http':0,
            'program_invocations':0,'program_failures':0,'program_cpu_seconds':0,'program_wall_seconds':0,'branches':{}})
        path_=episode_dir(root,row)/'r31_result.json'
        if not path_.exists():
            if following is None:following={'phase':row['phase'],'index':row['task_order_index'],'task_id':row['task']['task_id']}
            continue
        t=read(path_);s['scored']+=1;s['normal']+=int(t['valid']);s['correct']+=int(t['correct']);s['empty']+=int(not t['valid'])
        for target,source in [('tokens','model_tokens'),('http','model_http'),('program_invocations','program_invocations_total'),
                ('program_failures','program_failures'),('program_cpu_seconds','program_cpu_seconds'),('program_wall_seconds','program_wall_seconds')]:
            s[target]+=t[source]
        branch=s['branches'].setdefault(t['graph']['branch'],{'n':0,'correct':0,'tokens':0,'http':0})
        branch['n']+=1;branch['correct']+=int(t['correct']);branch['tokens']+=t['model_tokens'];branch['http']+=t['model_http']
        last={'phase':row['phase'],'task_id':t['task_id'],'trace_hash':t['trace_hash']}
    prefix=read(root/'budget/prefix.json');ledger=read(root/'budget/ledger.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    value={**old,**change,'updated_at':utc(),'by_phase':stats,'last_completed':last,'next_episode':following,
        'new_metered_tokens':sum(v.get('accounted_tokens',0) for v in new.values()),'new_http':len(new),
        'new_pending_http':sum(v['status']!='metered' for v in new.values()),
        'cumulative_tokens_or_reserved':sum(v.get('accounted_tokens',v['reserved_tokens']) for v in ledger['attempts'].values()),
        'observation_requirement_met':stats['development']['normal']==4 and stats['test']['normal']>=20}
    save_json(path,value);return value


def execute(root, env_file=None, *, solve_fn=None):
    import fcntl
    root=Path(root);lock=(root/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=checked(root);cfg=read(root/'config.json')
    if read(root/'preflight.json')['status']!='passed':raise ValueError('Actual Docker and Provider preflight required')
    if env_file:load_env(env_file)
    owner=BudgetGovernor(root/'budget/ledger.json',**LIMITS)
    if owner.billing_blocked:raise RuntimeError('Unresolved new request; do not blindly replay')
    module=bridge(root/'bundle');bundle=read(root/'bundle/bank.json');bank=Bank(root/'bank',readonly=True)
    adapters={s:AnswerAdapter('livemath',read(Path(plan['data_root'])/f'evaluator_records_{s}.json')) for s in ('train','test')}
    progress(root,status='running',pid=os.getpid(),error=None);event(root,{'event':'execute','pid':os.getpid()})
    try:
        for row in read(root/'schedule.json'):
            if row['phase']=='test' and not (root/'final_frozen_manifest.json').exists():
                dev=progress(root)['by_phase']['development']
                if dev['normal']!=4 or dev['program_failures'] or set(dev['branches'])!={'meta_prior','theorem_reasoning'}:
                    raise RuntimeError('Development interface acceptance did not pass; no automatic paid repeats')
                save_json(root/'development_acceptance.json',{'status':'passed','statistics':dev,'no_score_threshold':True})
                save_json(root/'final_frozen_manifest.json',{'freeze_time':utc(),'bank_tree':tree_identity(root/'bank'),
                          'plan_hash':digest(plan),'content_fixed_before_development':True})
            task=PublicTask(**row['task']);directory=episode_dir(root,row);already=(directory/'r31_result.json').exists()
            progress(root,active={'phase':row['phase'],'task_id':task.task_id})
            result=(solve_fn or module.solve_one)(task,adapters[task.split],bank,bundle,cfg['llm'],directory,
                     governor(root,row['phase'],plan,owner),variant=plan['variant'],phase=row['phase'])
            state=progress(root)
            if not already:
                event(root,{'event':'task_completed','phase':row['phase'],'task_id':task.task_id,
                      'valid':result['valid'],'correct':result['correct'],'branch':result['graph']['branch']})
                print(json.dumps({'phase':row['phase'],'task':task.task_id,'branch':result['graph']['branch'],
                      'valid':result['valid'],'correct':result['correct'],'tokens':result['model_tokens'],
                      'completed':state['by_phase'][row['phase']]['scored']},ensure_ascii=False),flush=True)
        if tree_identity(root/'bank')!=plan['bank_tree']:raise ValueError('Frozen Bank changed')
        save_json(root/'freeze_audit.json',{'before':plan['bank_tree'],'after':tree_identity(root/'bank')})
        state=progress(root,status='completed_test',active=None);save_json(root/'result.json',state)
        event(root,{'event':'completed_test'});return state
    except BaseException as exc:
        progress(root,status='blocked',error=str(exc),error_type=type(exc).__name__)
        event(root,{'event':'blocked','error':str(exc)});raise
    finally:
        bank.close()
        for adapter in adapters.values():adapter.close()
        lock.close()


def monitor(root):
    root=Path(root);state=read(root/'progress.json')
    try:os.kill(state.get('pid',-1),0);alive=state.get('pid',-1)>0
    except (OSError,TypeError):alive=False
    ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    new=[v for k,v in ledger['attempts'].items() if k not in prefix['attempts']]
    return {**state,'process_alive':alive,'new_metered_tokens':sum(v.get('accounted_tokens',0) for v in new),
            'new_http':len(new),'new_pending_http':sum(v['status']!='metered' for v in new),
            'new_unknown_billing':sum(v['status']=='unknown_billing' for v in new)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','execute','monitor'])
    parser.add_argument('--root',required=True);parser.add_argument('--previous');parser.add_argument('--env-file')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.root,args.previous)
    elif args.command=='execute':result=execute(args.root,args.env_file)
    else:result=globals()[args.command](args.root)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
