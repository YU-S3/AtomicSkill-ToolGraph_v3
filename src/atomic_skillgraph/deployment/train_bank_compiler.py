"""Train-only deployment compilation; no generation, promotion or invented credit.

Raw Train state is immutable input. Existing lifecycle decisions are inherited;
unsafe executable routes are withheld in the deployment copy. All history stays.
Equivalence uses the production exact proof engines, never names or summaries.
"""
import json
from pathlib import Path
import shutil
import sqlite3

from ..core.serialization import atomic_write_json, to_primitive
from ..evolution.identity_matching import (raw_hash, match_tool, verify_tool_proof,
    match_implementation, verify_implementation_proof)
from ..knowledge.database import StateDatabase
from ..knowledge.artifact_store import ArtifactStore
from ..knowledge.skill_registry import SkillRegistry
from ..knowledge.tool_registry import ToolRegistry
from ..harness.scienceworld import ScienceWorldAdapter
from ..runtime.invocation_compiler import InvocationCompiler
from ..tooling.validator import ToolStaticValidator
from ..validation.tool_validator import ToolValidator
from .release_protocol import DDL, ReleaseError, sha, verify_deployments
from .workflow_entry_closure import audit_workflow

VERSION='scienceworld.train-bank-compiler.v1'


def bank_digest(bank):
    from ..system import AtomicSkillGraphSystem
    with StateDatabase(Path(bank)/'state.sqlite3',readonly=True,r103=True) as db:
        view=type('DigestView',(),{})()
        view.database=db
        view.artifacts=ArtifactStore(bank,db)
        view.artifacts.verify_all()
        return AtomicSkillGraphSystem.knowledge_digest(view)


def clone_bank(source,target):
    source,target=Path(source).resolve(),Path(target).resolve()
    if target.exists() or target.is_relative_to(source):
        raise ValueError('Compiler output must be a fresh directory outside raw Train bank')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('state.sqlite3','state.sqlite3-wal','state.sqlite3-shm'))
    with sqlite3.connect(f'file:{(source/"state.sqlite3").as_posix()}?mode=ro',uri=True) as src:
        with sqlite3.connect(target/'state.sqlite3') as dst:
            src.backup(dst)
            for table,key in [('artifact_index','artifact_ref'),('provisional_artifacts','provisional_ref'),
                              ('failure_experiences','experience_id')]:
                for ref,path in dst.execute(f'SELECT {key},file_path FROM {table}').fetchall():
                    relative=Path(path).resolve().relative_to(source)
                    dst.execute(f'UPDATE {table} SET file_path=? WHERE {key}=?',(str(target/relative),ref))


def _proof_aliases(skills,tools):
    ts=sorted(tools.list_refs(mode='frozen'),key=str)
    tool_aliases,impl_aliases,proofs={},{},[]
    for index,ref in enumerate(ts):
        for canonical in ts[:index]:
            if str(canonical) in tool_aliases:continue
            left,right=tools.get(ref),tools.get(canonical)
            proof=match_tool(left,right)
            if proof.status=='exact' and verify_tool_proof(left,right,proof.proof):
                tool_aliases[str(ref)]=str(canonical)
                proofs.append({'kind':'tool','source':str(ref),'target':str(canonical),'proof':proof.proof})
                break
    implementations=sorted(skills.implementations(mode='frozen'),key=lambda x:str(x.ref))
    for index,left in enumerate(implementations):
        for right in implementations[:index]:
            if left.abstract_ref!=right.abstract_ref or str(right.ref) in impl_aliases:continue
            args=dict(source_atomic=skills.get_atomic(left.abstract_ref),target_atomic=skills.get_atomic(right.abstract_ref),
                source_tools={str(b.tool_ref):tools.get(b.tool_ref) for b in left.tool_bindings},
                target_tools={str(b.tool_ref):tools.get(b.tool_ref) for b in right.tool_bindings})
            proof=match_implementation(left,right,**args)
            if proof.status=='exact' and verify_implementation_proof(left,right,proof.proof,**args):
                impl_aliases[str(left.ref)]=str(right.ref)
                proofs.append({'kind':'implementation','source':str(left.ref),'target':str(right.ref),'proof':proof.proof})
                break
    return tool_aliases,impl_aliases,proofs


def verify_compiler_preferences(skills,prefs):
    lock=skills.store.data_dir/'edit_plan.lock.json'
    if prefs.get('source_lock_hash')!=sha(lock):raise ReleaseError('Train compiler source lock changed')
    source=json.loads(lock.read_text())
    if source.get('compiler_protocol_version')!=VERSION or source.get('source_split')!='train':
        raise ReleaseError('Compiler requires Train-only source authority')
    tools=ToolRegistry(skills.store,skills.database)
    ta,ia,_=_proof_aliases(skills,tools)
    if prefs.get('canonical_tool_aliases')!=ta or prefs.get('canonical_implementation_aliases')!=ia:
        raise ReleaseError('Train compiler equivalence proof mismatch')
    if prefs.get('preferred_implementations') or prefs.get('preferred_composites'):
        raise ReleaseError('This compiler inherits ranking; it cannot invent a preference')


def compile_train(config_path,output):
    from ..system import load_config
    from experiments.protocol import hash_code, RunManifest, RunState
    from experiments.scienceworld_manifest import load
    from experiments.report import validate_formal_usage
    config=load_config(config_path)
    exp=config['experiment']
    source=Path(config['data_dir']).resolve()
    run=Path(exp['output_dir']).resolve()
    output=Path(output).resolve()
    if (exp.get('benchmark')!='scienceworld' or exp.get('phase')!='train' or
        not source.is_relative_to(run) or Path(config['trace_data_dir']).resolve()!=run):
        raise ValueError('Only the corresponding seed Train bank/trace root is allowed')
    train_manifest=load(Path(config['harness']['manifest']))
    if train_manifest['split']!='train':raise ValueError('Dev/Test may not be compiler authority')
    source_digest=bank_digest(source)
    if output.exists():raise FileExistsError(output)
    with StateDatabase(source/'state.sqlite3',readonly=True,r103=True) as db:
        manifest_path=(run.parent/exp['name']/'run_manifest.json').resolve()
        if not manifest_path.is_relative_to(run.parent):raise ValueError('Unsafe source run identity')
        manifest=RunManifest.from_dict(json.loads(manifest_path.read_text()))
        if manifest.run_id!=exp['name']:raise ValueError('Source run identity mismatch')
        run_state=db.execute('SELECT state FROM run_manifests WHERE run_id=?',(exp['name'],)).fetchone()
        if run_state is None or run_state['state']!=RunState.COMPLETED.value:
            raise ValueError('Train must complete final maintenance/accounting first')
        if manifest.metadata.get('reference_manifest_digest')!=train_manifest['digest']:
            raise ValueError('Train source manifest changed')
        completed=db.rows('SELECT * FROM run_tasks WHERE run_id=?',(exp['name'],))
        if not completed or any(r['state']!='completed' for r in completed):raise ValueError('Train contains uncompleted tasks')
        if (manifest.phase!='train' or manifest.metadata.get('seed')!=exp['seed'] or
                {r['task_id'] for r in completed}!={t.task_id for t in manifest.tasks}):
            raise ValueError('Train run/seed/task authority mismatch')
        actual=json.loads((run/'summary.json').read_text())
        if actual['knowledge_digest']!=source_digest:raise ValueError('Train summary/bank digest mismatch')
        last=next(r for r in completed if r['task_id']==manifest.tasks[-1].task_id)
        final=json.loads(last['result_json']).get('final_batch_maintenance',{})
        if (final.get('pending_count')!=0 or final.get('knowledge_digest_after')!=source_digest or
                not (run/'traces'/(final.get('maintenance_trace_id','')+'.json')).is_file()):
            raise ValueError('Missing final maintenance/queue/accounting boundary')
        if actual.get('completed')!=len(manifest.tasks) or actual.get('expected')!=len(manifest.tasks):
            raise ValueError('Incomplete Train report')
        rows=[dict(r) for r in db.rows('SELECT * FROM artifact_index ORDER BY artifact_ref')]
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='release_deployments'").fetchone():
            raise ValueError('Authored or previously compiled release is not raw Train authority')
        valid_ids={r['task_id'] for r in train_manifest['tasks']}
        if any(t.task_id not in valid_ids or t.split!='train' for t in manifest.tasks):
            raise ValueError('Manifest contains non-Train source')
        receipts=[]
        for path in sorted((run/'traces').glob('*.json')):
            payload=json.loads(path.read_text())
            source_identity=payload.get('metadata',{}).get('execution_source',{})
            if source_identity.get('split') not in (None,'train'):
                raise ValueError('Non-Train Trace in source root')
            validate_formal_usage([payload])
            receipts.append({'path':str(path),'sha256':sha(path),'trace_id':payload.get('trace_id')})
        if not receipts:raise ValueError('No Train traces')
    bank=output/'data_v3'
    clone_bank(source,bank)
    lock={'compiler_protocol_version':VERSION,'source_split':'train','seed':exp['seed'],
        'source_train_bank_digest':source_digest,'source_run_id':manifest.run_id,
        'source_code_hash':manifest.code_commit,'train_manifest_digest':train_manifest['digest'],
        'compiler_code_hash':hash_code(Path(__file__).resolve().parents[3]),
        'source_trace_receipts':receipts,'authored_reference_used':False}
    atomic_write_json(bank/'edit_plan.lock.json',lock)
    routes,workflows,checks=[],[],[]
    with StateDatabase(bank/'state.sqlite3',r103=True) as db:
        store=ArtifactStore(bank,db);skills=SkillRegistry(store,db);tools=ToolRegistry(store,db)
        harness=ScienceWorldAdapter()
        compiler=InvocationCompiler(skills,tools,harness,mode='frozen')
        for impl in skills.implementations(mode='frozen'):
            failures=[]
            try:
                atomic=skills.get_atomic(impl.abstract_ref)
                if atomic.status.value!='active':raise ValueError('Atomic not Active')
                used=[tools.get(b.tool_ref) for b in impl.tool_bindings]
                if any(t.status.value not in {'active','preferred'} for t in used):raise ValueError('Tool not Frozen-usable')
                compiler.compile(atomic,impl,used,{})
                for tool in used:
                    for report in (ToolStaticValidator().validate_tool_asset(tool,atomic,harness),ToolValidator().validate_asset(tool)):
                        if not report.passed:failures.extend(report.failure_codes)
            except (KeyError,TypeError,ValueError) as exc:failures.append(str(exc))
            if failures:db.execute("UPDATE artifact_index SET status='shadow' WHERE artifact_ref=?",(str(impl.ref),))
            routes.append({'implementation_ref':str(impl.ref),'atomic_ref':str(impl.abstract_ref),
                'frozen_executable':not failures,'failures':failures})
        db.connection.commit()
        for graph in skills.composites():
            if graph.status.value!='active':continue
            try:
                audit=audit_workflow(skills,graph,harness)
            except (KeyError,TypeError,ValueError) as exc:
                audit={'composite_ref':str(graph.ref),'program_static_closure':False,'error':str(exc)}
            if not audit['program_static_closure']:
                db.execute("UPDATE artifact_index SET status='shadow' WHERE artifact_ref=?",(str(graph.ref),))
            workflows.append(audit)
        db.connection.commit()
        ta,ia,proofs=_proof_aliases(skills,tools)
        prefs={'protocol_version':VERSION,'source_lock_hash':sha(bank/'edit_plan.lock.json'),
            'canonical_tool_aliases':ta,'canonical_implementation_aliases':ia,
            'preferred_implementations':[],'preferred_composites':[],
            'blocked_equivalent_groups':[],'ranking':'inherited_no_dev_selection'}
        atomic_write_json(bank/'deployment_preferences.json',prefs)
        verify_compiler_preferences(skills,prefs)
        db.execute(DDL)
        for original in rows:
            ref=original['artifact_ref']
            current=db.execute('SELECT * FROM artifact_index WHERE artifact_ref=?',(ref,)).fetchone()
            payload=store.get_payload(ref)
            if payload.get('metadata',{}).get('experiment_kind')=='authored_reference':raise ValueError('Authored asset in learned bank')
            source_hash=raw_hash(json.loads(Path(original['file_path']).read_text()))
            if source_hash!=raw_hash(payload):raise ValueError('Compiler changed an immutable Train artifact')
            check={'passed':True,'compiler_protocol_version':VERSION,'source_ref':ref,
                'source_payload_hash':source_hash,'source_status':original['status'],
                'effective_status':current['status'],'execution_credit_delta':0,
                'source_lock_hash':sha(bank/'edit_plan.lock.json')}
            path=Path('publication_checks')/(raw_hash(ref)+'.json')
            atomic_write_json(bank/path,check)
            db.execute('INSERT INTO release_deployments VALUES(?,?,?,?,?,?,?,?,?)',
                (ref,original['artifact_kind'],current['status'],'inherited_registry',ref,source_hash,source_hash,str(path),sha(bank/path)))
            checks.append(check)
        db.connection.commit()
        verify_deployments(db,bank)
        store.verify_all()
    for name,records in [('workflow_closure',workflows),('effective_routes',routes),('identity_proofs',proofs)]:
        with (output/(name+'.jsonl')).open('x') as handle:
            for row in records:handle.write(json.dumps(to_primitive(row),sort_keys=True)+'\n')
    atomic_write_json(output/'publication_checks.json',{'passed':True,'checks':checks})
    if bank_digest(source)!=source_digest:raise RuntimeError('Raw Train bank changed during compilation')
    compiled_digest=bank_digest(bank)
    atomic_write_json(bank/'freeze_manifest.json',{'schema_version':3,'knowledge_digest':compiled_digest,
        'source_data_dir':str(source),'provenance':lock})
    report={**lock,'compiled_bank_digest':compiled_digest,'compiled_bank':str(bank),
        'preserved_failure_history':True,'no_new_execution_credit':True,'completed':True,
        'routes':len(routes),'workflows':len(workflows)}
    atomic_write_json(output/'compiler_manifest.json',report)
    return report
