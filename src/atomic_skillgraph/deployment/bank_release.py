"""Fresh-schema import and checked publication of authored revisions."""
from __future__ import annotations
import copy
import hashlib
import json
import shutil
import sqlite3
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

from .release_protocol import *
from .asset_revision import revise,ref_map
from ..core.contracts import AbstractAtomicSkill,ToolAsset,ImplementationAtom,CompositeSkill
from ..core.refs import content_hash
from ..core.serialization import dataclass_from_dict,to_primitive,atomic_write_json
from ..core.status import RuntimeMode
from ..knowledge.database import StateDatabase,SCHEMA_VERSION
from ..knowledge.artifact_store import ArtifactStore
from ..knowledge.skill_registry import SkillRegistry
from ..knowledge.identity_index import prepare_index_row
from ..harness.alfworld import AlfWorldAdapter
from ..tooling.validator import ToolStaticValidator
from ..agents.skill_guidance import normalize_guideline
from ..runtime.input_authorization import validate_declarations
from ..planner.multiplicity import normalize_task_contract
from ..planner.compiler import PlanCompiler
from ..evolution.identity_matching import raw_hash
from .publication_audit import inventory, verify_preservation, identity_and_support, json_lines

CLASSES={'atomic':AbstractAtomicSkill,'tool':ToolAsset,'implementation':ImplementationAtom,'composite':CompositeSkill}
COUNTS={42:218,43:202,44:216}
IDENTITY_COUNTS={42:136,43:135,44:133}

def _json(path,value):
    atomic_write_json(path,to_primitive(value))


def _stage(root,name,inputs,outputs):
    """Immutable stage receipts; interrupted builds are archived, never replayed."""
    path=root/'stages'/f'{name}.json'
    payload={'stage':name,'status':'complete','inputs':inputs,'outputs':outputs}
    payload['input_hash']=raw_hash(inputs);payload['output_hash']=raw_hash(outputs)
    if path.exists():
        if json.loads(path.read_text())!=payload:raise ReleaseError('stage receipt changed')
    else:_json(path,payload)
    return sha(path)


def _validate_published_graph(graph,skills,db):
    from ..planner.validator import PlannerValidator
    from ..knowledge.graph_store import GraphStore
    from ..knowledge.query import complete_composite_contract_diagnosis
    from ..harness.protocol import HarnessTask
    checked=_graph_checks(graph,skills)
    if any(not o['implementation_candidates'] for o in checked['runtime_plan']['occurrences']):
        raise ReleaseError('graph has no frozen implementation')
    # Publisher fixtures only: no family-dependent Runtime routing is added.
    examples={'G01':('pick_and_place_simple','put an object in a destination'),
        'G02':('pick_clean_then_place_in_recep','put a clean object in a destination'),
        'G03':('pick_heat_then_place_in_recep','put a hot object in a destination'),
        'G04':('pick_cool_then_place_in_recep','put a cold object in a destination'),
        'G05':('look_at_obj_in_light','look at an object under a light'),
        'G06':('pick_two_obj_and_place','put two objects in a destination')}
    family,goal=examples[graph.metadata['catalog_id']]
    task=HarnessTask('publication-contract-check',goal,'alfworld',family)
    harness=AlfWorldAdapter(split='train');formal=harness.task_contract(task)
    diagnosis=complete_composite_contract_diagnosis(formal,graph.goal_contract)
    if not diagnosis.passed:raise ReleaseError(f'P0 shape mismatch: {diagnosis}')
    plan=PlanCompiler(skills).from_composite(task,formal,graph,mode=RuntimeMode.FROZEN,audit={})
    report=PlannerValidator(skills,GraphStore(db,skills)).validate(plan,mode=RuntimeMode.FROZEN,
        harness_profile=harness.profile_name,task_binding_roles={'object','destination','light_source'},literal_authorities={})
    if not report.passed:raise ReleaseError(f'graph runtime validation failed: {report}')
    checked['formal_plan_validation']=to_primitive(report)
    return checked


def _import_artifact(store,kind,obj,status):
    """Connection-owned fresh import; identity is built once after all A/T/I."""
    payload=to_primitive(obj);payload['schema_version']=SCHEMA_VERSION
    path=store.path_for(kind,obj.ref)
    _json(path,payload)
    ref=str(obj.ref);logical=obj.ref.tool_id if kind=='tool' else obj.ref.logical_id
    with store.database.transaction() as connection:
        connection.execute('INSERT INTO artifact_index VALUES(?,?,?,?,?,?,?,?)',
            (ref,kind,logical,obj.ref.version,content_hash(payload,exclude=('status','quality','statistics','evidence')),
             status,str(path.resolve()),SCHEMA_VERSION))
    return path

def _extract(source,destination):
    with zipfile.ZipFile(source) as z:
        seen=set()
        for member in z.infolist():
            name=member.filename
            if name in seen or '\\' in name or Path(name).is_absolute() or ':' in name or '..' in Path(name).parts:
                raise ReleaseError('unsafe or duplicate ZIP member')
            seen.add(name)
            if stat.S_ISLNK(member.external_attr>>16):raise ReleaseError('ZIP symlink refused')
            contained(destination,name)
        if z.testzip():raise ReleaseError('damaged ZIP')
        z.extractall(destination)

def _source_inventory(root):
    db=sqlite3.connect(f'file:{(root/"data_v3/state.sqlite3").as_posix()}?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT * FROM artifact_index ORDER BY artifact_ref'))
    assets=[]
    for row in rows:
        rel=Path('artifacts')/({'tool':'tools'}.get(row['artifact_kind'],row['artifact_kind']))/row['logical_id']
        rel=rel/row['version']/'tool.json' if row['artifact_kind']=='tool' else rel/(row['version']+'.json')
        path=contained(root/'data_v3',rel)
        payload=json.loads(path.read_text(encoding='utf-8'))
        if content_hash(payload,exclude=('status','quality','statistics','evidence'))!=row['content_hash']:
            raise ReleaseError('source artifact hash mismatch')
        obj=dataclass_from_dict(CLASSES[row['artifact_kind']],payload)
        if str(obj.ref)!=row['artifact_ref']:raise ReleaseError('source ref mismatch')
        assets.append((dict(row),obj,payload,path))
    return db,assets

def _graph_checks(graph,skills):
    """Prove effect coverage from actual ordered occurrences, not goal self-matching."""
    contract=normalize_task_contract(graph.goal_contract)
    graph.goal_contract=contract
    order=graph.control_sequence;occ={o.step_id:o for o in graph.occurrences}
    if len(order)!=len(set(order)) or set(order)!=set(occ):raise ReleaseError('invalid graph sequence')
    outputs={};observed=[];input_origins={};semantic_origins={}
    for step in order:
        o=occ[step];a=skills.get_atomic(o.node_ref)
        inputs={p.name:p for p in a.inputs};origins={}
        for role,p in inputs.items():
            b=o.binding_specs.get(role)
            if b is None:
                if not p.runtime_resolvable:raise ReleaseError('unbound non-runtime input')
                origins[role]=('runtime',step,role)
            elif b.kind.value=='skill_input':origins[role]=('task',b.source_role)
            elif b.kind.value=='data_flow':
                key=(b.source_step,b.source_role)
                if key not in outputs:raise ReleaseError('missing forward producer')
                origin,source_type=outputs[key]
                if source_type!=p.semantic_type:raise ReleaseError('dataflow type mismatch')
                edges=[e for e in graph.data_edges if e.target_step==step and e.target_role==role]
                if len(edges)!=1 or (edges[0].source_step,edges[0].source_role)!=key:raise ReleaseError('dataflow is not single explicit producer')
                origins[role]=origin
            else:raise ReleaseError('unsupported release graph binding')
        input_origins[step]=origins
        for p in a.outputs:
            derivation=a.validator_spec.get('output_derivations',{}).get(p.name,{})
            if derivation.get('kind')=='input_identity':origin=origins[derivation['input_role']]
            elif derivation.get('kind')=='effect_witness':
                origin=('fresh',step,p.name)
                constraint=a.validator_spec.get('output_semantic_constraints',{}).get(p.name,{})
                if constraint.get('compatible_with_input') in origins:
                    semantic_origins[origin]=origins[constraint['compatible_with_input']]
            else:raise ReleaseError('unproved graph output origin')
            outputs[(step,p.name)]=(origin,p.semantic_type)
        for e in a.effects:
            args={}
            for role,b in e.args.items():
                source=b.source_role if hasattr(b,'source_role') else b.get('source_role')
                args[role]=origins.get(source,outputs.get((step,source),(None,None))[0])
                if args[role] is None:raise ReleaseError('unbound graph effect')
            observed.append((e.predicate,e.effect_domain.value,args))
    witnesses=[]
    def covers(effect,goal):
        if effect[0]!=goal.predicate or effect[1]!=goal.effect_domain.value or set(effect[2])!=set(goal.args):return False
        for role,binding in goal.args.items():
            target=('task',binding.get('source_role') if isinstance(binding,dict) else binding.source_role)
            origin=effect[2][role]
            if semantic_origins.get(origin,origin)!=target:return False
        return True
    for goal in contract.target_effects:
        matching=[e for e in observed if covers(e,goal)]
        if len(matching)<goal.cardinality:raise ReleaseError('graph does not cover required effects/cardinality')
        witnesses.append(matching)
    for identity in contract.identity_constraints:
        if identity.relation.value=='same_as':
            shared={e[2].get(identity.left_role) for matches in witnesses for e in matches}
            if None in shared or len(shared)!=1:raise ReleaseError('processing and placement identity diverged')
    plan=PlanCompiler(skills).from_composite(SimpleNamespace(task_id='release_graph_check'),contract,graph,mode=RuntimeMode.FROZEN,audit={})
    if contract.cardinality_constraints:
        if not plan.repeat_constraints:raise ReleaseError('Repeat compilation missing')
        if not any(len(c.iteration_steps)==2 for c in plan.repeat_constraints):raise ReleaseError('Repeat not two iterations')
    return {'passed':True,'actual_effect_witnesses':to_primitive(witnesses),'runtime_plan':to_primitive(plan)}

def prepare_release(spec):
    from experiments.protocol import hash_code, hash_config
    if spec.seed not in INPUT_HASHES or sha(spec.input_zip)!=INPUT_HASHES[spec.seed]:
        raise ReleaseError('source ZIP does not match the frozen input identity')
    root=Path(spec.output_dir).resolve()
    identity={'seed':spec.seed,'source_zip_hash':sha(spec.input_zip),
              'config_hash':hash_config(spec.base_config),'code_hash':hash_code(Path(__file__).resolve().parents[3])}
    if root.exists():
        if not spec.resume:raise FileExistsError(root)
        recorded=json.loads((root/'build_identity.json').read_text())
        if recorded!=identity:raise ReleaseError('prepare resume identity mismatch')
        if (root/'prepare_manifest.json').exists():
            verify_release(PreparedRelease(root,spec.seed))
            return PreparedRelease(root,spec.seed)
        # No API is used by prepare. Preserve an interrupted build verbatim and
        # reconstruct a fresh database, never replay commits into a partial DB.
        import uuid
        root.rename(root.with_name(root.name+'.interrupted-'+uuid.uuid4().hex[:10]))
    root.mkdir(parents=True,exist_ok=False)
    _json(root/'build_identity.json',identity)
    source=root/'source';source.mkdir()
    shutil.copy2(spec.input_zip,source/'input.zip');_extract(source/'input.zip',source/'unpacked')
    src,assets=_source_inventory(source/'unpacked')
    if len(assets)!=COUNTS[spec.seed]:raise ReleaseError('source count mismatch')
    refs=ref_map([a for _,a,_,_ in assets])
    if len(refs)!=33:raise ReleaseError('expected exactly 33 authored source versions')
    source_inventory=inventory(assets,refs)
    _json(root/'source_inventory.json',source_inventory)
    imported_receipt=_stage(root,'01_import',identity,source_inventory)
    bank=root/'work/data_v3';db=StateDatabase(bank/'state.sqlite3',r103=True)
    db.connection.executescript(DDL);db.connection.commit()
    store=ArtifactStore(bank,db);skills=SkillRegistry(store,db)
    rewritten=[];checks={};mapping=[]
    try:
        for row,obj,payload,path in sorted(assets,key=lambda r:({'atomic':0,'tool':1,'implementation':2,'composite':3}[r[0]['artifact_kind']],r[0]['artifact_ref'])):
            kind=row['artifact_kind'];new=revise(kind,obj,refs);is_new=str(obj.ref) in refs
            if is_new and kind=='atomic':
                declarations=validate_declarations(new)
                guideline=normalize_guideline(new.guideline,formal_roles=[p.name for p in [*new.inputs,*new.outputs]])
                if set(new.validator_spec.get('output_derivations',{})) != {p.name for p in new.outputs}:
                    raise ReleaseError('Atomic output derivations are incomplete')
                checks[str(new.ref)]={'passed':True,'input_authorization':declarations,
                    'guideline':guideline,'output_derivations':new.validator_spec['output_derivations']}
            if is_new and kind=='tool':
                a=next(a for k,a in rewritten if k=='atomic' and a.metadata['catalog_id']==new.metadata['catalog_id'])
                report=ToolStaticValidator().validate_tool_asset(new,a,AlfWorldAdapter(split='train'))
                if not report.passed:raise ReleaseError(str(to_primitive(report)))
                # This is the completed static/safety review of authored IR,
                # not an empirical execution/admission success.
                new.safety.update(reviewed=True,review_basis='release_static_validation')
                from ..validation.tool_validator import ToolValidator
                local=ToolValidator().validate_asset(new)
                if not local.passed:raise ReleaseError(str(local))
                checks[str(new.ref)]={'passed':True,'static':to_primitive(report),'local_asset':to_primitive(local)}
            if is_new and kind=='implementation':
                from ..runtime.invocation_compiler import InvocationCompiler
                from ..knowledge.tool_registry import ToolRegistry
                from ..core.status import SkillStatus,ToolStatus
                compiler=InvocationCompiler(skills,ToolRegistry(store,db),AlfWorldAdapter(split='train'),mode=RuntimeMode.FROZEN)
                prospective=copy.deepcopy(new);prospective.status=SkillStatus.ACTIVE
                prospective_tools=[copy.deepcopy(compiler.tools.get(b.tool_ref)) for b in new.tool_bindings]
                for tool in prospective_tools:tool.status=ToolStatus.ACTIVE
                compiled=compiler.compile(skills.get_atomic(new.abstract_ref),prospective,prospective_tools,{})
                checks[str(new.ref)]={'passed':True,'check_scope':'proposed active deployment interface',
                                      'compiled_interface':to_primitive(compiled)}
            if is_new and kind=='composite':
                checked=_graph_checks(new,skills)
                new.validator_spec['task_contract_covered']=True
                new.metadata['completion_authority']={'kind':'complete_contract','basis':f'publication_checks/{new.ref.logical_id}.json'}
                checks[str(new.ref)]=checked
            target=_import_artifact(store,kind,new,'draft' if is_new else row['status'])
            if not is_new and target.read_bytes()!=path.read_bytes():
                # Dataclass serializers may normalize JSON formatting. Preserve
                # the source immutable bytes; the verified canonical hash agrees.
                target.write_bytes(path.read_bytes())
            if kind=='tool' and new.artifact.get('filename'):
                body=contained(path.parent,new.artifact['filename'])
                shutil.copy2(body,target.parent/new.artifact['filename'])
            if is_new:
                rewritten.append((kind,new))
                indexed=db.execute('SELECT content_hash FROM artifact_index WHERE artifact_ref=?',(str(new.ref),)).fetchone()[0]
                mapping.append({'seed':spec.seed,'kind':kind,'source_ref':str(obj.ref),'source_payload_hash':raw_hash(payload),
                    'released_ref':str(new.ref),'released_payload_hash':raw_hash(json.loads(target.read_text())),
                    'changed_fields':[k for k in to_primitive(new) if to_primitive(new).get(k)!=payload.get(k)],
                    'effective_status':'active','content_hash':indexed})
        # Import only exact supported historical shapes. Protocol/version and
        # source-only incompatible tables remain byte-preserved in the archive.
        excluded={'metadata','artifact_index','artifact_identity_index','graph_edges','runtime_support_observations',
            'run_manifests','run_tasks','release_deployments'}
        imported={}
        for (table,) in src.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            if table in excluded or not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():continue
            source_columns=[r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
            target_columns=[r[1] for r in db.execute(f'PRAGMA table_info("{table}")')]
            if source_columns!=target_columns:raise ReleaseError(f'unsupported historical table shape: {table}')
            values=src.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            relocated=[]
            for value in values:
                value=list(value)
                for locator in ('file_path','capsule_path','payload_path'):
                    if locator not in source_columns:continue
                    i=source_columns.index(locator);old=str(value[i]).replace('\\','/')
                    relative=old.split('/data_v3/',1)[-1] if '/data_v3/' in old else old
                    original=contained(source/'unpacked/data_v3',relative)
                    if not original.is_file():raise ReleaseError(f'missing historical payload: {table}:{relative}')
                    destination=contained(bank,relative);destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(original,destination)
                    value[i]=str(destination) if locator=='file_path' else relative
                relocated.append(tuple(value))
            if values:db.connection.executemany(f'INSERT INTO "{table}" VALUES({",".join("?" for _ in source_columns)})',relocated)
            imported[table]=len(values)
        db.connection.commit()
        rewrite_receipt=_stage(root,'02_rewrite',{'import':imported_receipt},mapping)
        # All original relation endpoints are preserved; authored source refs
        # are transformed only via the explicit map.
        for row in src.execute('SELECT * FROM graph_edges'):
            values=list(row)
            values[1]=str(refs.get(values[1],values[1]));values[2]=str(refs.get(values[2],values[2]))
            db.execute('INSERT INTO graph_edges VALUES(?,?,?,?,?)',tuple(values))
        db.connection.commit()
        db.execute('DELETE FROM artifact_identity_index');db.connection.commit()
        for kind in ('atomic','tool','implementation'):
            for row in db.rows("SELECT * FROM artifact_index WHERE artifact_kind=? ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'preferred' THEN 0 ELSE 1 END,artifact_ref",(kind,)):
                payload=json.loads(Path(row['file_path']).read_text(encoding='utf-8'));obj=dataclass_from_dict(CLASSES[kind],payload)
                identity=prepare_index_row(bank,kind,obj,payload,db)
                if identity is None:raise ReleaseError('identity row missing')
                db.execute('INSERT INTO artifact_identity_index VALUES(?,?,?,?,?,?,?,?,?,?)',identity);db.connection.commit()
        if db.execute('SELECT count(*) FROM artifact_identity_index').fetchone()[0]!=IDENTITY_COUNTS[spec.seed]:raise ReleaseError('identity index count mismatch')
        identity_receipt=_stage(root,'03_identity',{'rewrite':rewrite_receipt},
            [dict(r) for r in db.rows('SELECT * FROM artifact_identity_index ORDER BY artifact_ref')])
        identity_and_support(db,bank,root)
        merge_receipt=_stage(root,'04_merge',{'identity':identity_receipt},
            {name:sha(root/name) for name in ('identity_verification.json','identity_groups.jsonl','support_union_report.json')})
        for kind,obj in rewritten:
            checked=checks[str(obj.ref)]
            path=bank/'publication_checks'/f'{obj.ref.logical_id if kind!="tool" else obj.ref.tool_id}.json'
            _json(path,checked)
            checks[str(obj.ref)]=dict(path=str(path.relative_to(bank)),hash=sha(path))
        # Publication is one connection-owned transaction. No Registry commit
        # and no learned-success events or historical projection rewrites.
        with db.transaction() as conn:
            for item in mapping:
                ref=item['released_ref'];check=checks[ref]
                conn.execute('UPDATE artifact_index SET status=? WHERE artifact_ref=?',('active',ref))
                conn.execute('INSERT INTO release_deployments VALUES(?,?,?,?,?,?,?,?,?)',(
                    ref,item['kind'],'active','authored_revision',item['source_ref'],item['source_payload_hash'],item['released_payload_hash'],check['path'],check['hash']))
            # Check the real prospective Active graph before the single commit.
            # A failed dependency, mapping or P0 contract rolls back every status.
            for kind,obj in rewritten:
                if kind=='composite':
                    check=_validate_published_graph(obj,skills,db)
                    path=bank/checks[str(obj.ref)]['path'];_json(path,check)
                    checks[str(obj.ref)]['hash']=sha(path)
                    conn.execute('UPDATE release_deployments SET checks_hash=? WHERE artifact_ref=?',
                        (sha(path),str(obj.ref)))
            verify_deployments(db,bank)
        _stage(root,'05_publish',{'merge':merge_receipt},
            [dict(r) for r in db.rows('SELECT * FROM release_deployments ORDER BY artifact_ref')])
        verify_deployments(db,bank);store.verify_all()
        preserved=verify_preservation(db,bank,source_inventory)
        _json(root/'asset_rewrite_manifest.json',mapping)
        json_lines(root/'asset_rewrite_manifest.jsonl',mapping)
        json_lines(root/'contract_checks.jsonl',[{'ref':ref,**value} for ref,value in checks.items()])
        json_lines(root/'input_authorization_checks.jsonl',[{'ref':str(o.ref),'declarations':validate_declarations(o)} for k,o in rewritten if k=='atomic'])
        json_lines(root/'release_deployments.jsonl',[dict(r) for r in db.rows('SELECT * FROM release_deployments')])
        _json(root/'artifact_closure_report.json',{'preserved_assets':preserved,'source_versions_archive':'source/unpacked/data_v3','passed':True})
        _json(root/'prepare_manifest.json',{'protocol_version':PROTOCOL_VERSION,'seed':spec.seed,'source_zip_hash':sha(spec.input_zip),
            'imported_history_tables':imported,'config':spec.base_config,'status':'prepared'})
        _json(root/'publication_checks.json',{'passed':True,'count':len(mapping),'identity_count':IDENTITY_COUNTS[spec.seed]})
        for kind,obj in rewritten:_json(root/'readable_assets'/f'{obj.metadata["catalog_id"]}_{kind}.json',obj)
        _json(root/'catalog.json',[{'ref':r['artifact_ref'],'kind':r['artifact_kind'],'status':r['status']} for r in db.rows('SELECT * FROM artifact_index')])
    finally:
        src.close();db.close()
    return PreparedRelease(root,spec.seed)

def verify_release(prepared):
    from experiments.protocol import (active_composite_frozen_closure_audit,atomic_output_derivation_audit,
        active_repeat_identity_closure_audit,composite_deployment_evidence_audit)
    root=prepared.root;bank=root/'work/data_v3'
    with StateDatabase(bank/'state.sqlite3',readonly=True,r103=True) as db:
        store=ArtifactStore(bank,db);skills=SkillRegistry(store,db)
        store.verify_all();verify_deployments(db,bank)
        verify_preservation(db,bank,json.loads((root/'source_inventory.json').read_text()))
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or db.rows('PRAGMA foreign_key_check'):
            raise ReleaseError('SQLite integrity or foreign-key failure')
        audits={
            'children':active_composite_frozen_closure_audit(db,skills),
            'outputs':atomic_output_derivation_audit(skills),
            'repeat':active_repeat_identity_closure_audit(db,skills),
            'historical_deployment':composite_deployment_evidence_audit(db),
        }
        authored={r['artifact_ref'] for r in db.rows("SELECT artifact_ref FROM release_deployments WHERE basis='authored_revision'")}
        violations=[]
        for kind,audit in audits.items():
            for v in audit.get('violations',[]):
                if kind=='historical_deployment' and v.get('composite_ref',v.get('artifact_ref')) in authored:
                    continue  # This ref has separately verified authored publication, not historical deployment.
                violations.append({'audit':kind,**v})
        from ..knowledge.tool_registry import ToolRegistry
        from ..runtime.invocation_compiler import InvocationCompiler
        from ..planner.validator import PlannerValidator
        from ..knowledge.graph_store import GraphStore
        from ..knowledge.query import complete_composite_contract_diagnosis
        from ..harness.protocol import HarnessTask
        compiler=InvocationCompiler(skills,ToolRegistry(store,db),AlfWorldAdapter(split='train'),mode=RuntimeMode.FROZEN)
        graph_reports={};tool_reports={}
        for row in db.rows('SELECT * FROM release_deployments'):
            kind=row['artifact_kind'];payload=json.loads(Path(db.execute('SELECT file_path FROM artifact_index WHERE artifact_ref=?',(row['artifact_ref'],)).fetchone()[0]).read_text())
            obj=dataclass_from_dict(CLASSES[kind],payload)
            if kind=='composite':
                graph_reports[str(obj.ref)]=_graph_checks(obj,skills)
                plan=graph_reports[str(obj.ref)]['runtime_plan']
                if any(not o['implementation_candidates'] for o in plan['occurrences']):raise ReleaseError('graph has no frozen implementation')
                # Publication-only examples exercise the adapter's actual P0
                # contract shapes, not a runtime family decision policy.
                examples={'G01':('pick_and_place_simple','put an object in a destination'),
                    'G02':('pick_clean_then_place_in_recep','put a clean object in a destination'),
                    'G03':('pick_heat_then_place_in_recep','put a hot object in a destination'),
                    'G04':('pick_cool_then_place_in_recep','put a cold object in a destination'),
                    'G05':('look_at_obj_in_light','look at an object under a light'),
                    'G06':('pick_two_obj_and_place','put two objects in a destination')}
                family,goal=examples[obj.metadata['catalog_id']]
                task=HarnessTask('publication-contract-check',goal,'alfworld',family)
                formal=compiler.harness.task_contract(task)
                diagnosis=complete_composite_contract_diagnosis(formal,obj.goal_contract)
                if not diagnosis.passed:raise ReleaseError(f'P0 shape mismatch: {diagnosis}')
                runtime_plan=PlanCompiler(skills).from_composite(task,formal,obj,mode=RuntimeMode.FROZEN,audit={})
                report=PlannerValidator(skills,GraphStore(db,skills)).validate(runtime_plan,mode=RuntimeMode.FROZEN,
                    harness_profile=compiler.harness.profile_name,task_binding_roles={'object','destination','light_source'},literal_authorities={})
                graph_reports[str(obj.ref)]['formal_plan_validation']=to_primitive(report)
                if not report.passed:raise ReleaseError(f'graph runtime validation failed: {report}')
            if kind=='implementation':
                impl=skills.get_implementation(obj.ref);a=skills.get_atomic(impl.abstract_ref)
                tools=[compiler.tools.get(b.tool_ref) for b in impl.tool_bindings]
                compiler.compile(a,impl,tools,{})
                for tool in tools:
                    report=ToolStaticValidator().validate_tool_asset(tool,a,compiler.harness)
                    if not report.passed:raise ReleaseError(str(to_primitive(report)))
                    from ..validation.tool_validator import ToolValidator
                    local=ToolValidator().validate_asset(tool)
                    if not local.passed:raise ReleaseError(str(local))
                    tool_reports[str(tool.ref)]=to_primitive(report)
        check={'passed':not violations,'violations':violations,'audits':audits,'graphs':graph_reports,'tools':tool_reports}
        _json(root/'release_checks.json',check)
        if violations:raise ReleaseError(f'release structure audit failed: {violations}')
    return ReleaseChecks(True,sha(root/'release_checks.json'))

def _bank_files(bank):
    return {p.relative_to(bank).as_posix():sha(p) for p in sorted(bank.rglob('*'))
        if p.is_file() and p.name not in {'release_manifest.json','state.sqlite3-wal','state.sqlite3-shm'}}

def release_resources(config):
    resources = {k: config.get(k) for k in ('llm', 'runtime', 'cold_start', 'r103')}
    harness = config.get('harness', {})
    if harness.get('public_discovery_version'):
        import os
        from ..harness.public_discovery import resource_contract, VERSION
        if harness['public_discovery_version'] != VERSION:
            raise ReleaseError('unsupported public discovery resource version')
        data = harness.get('alfworld_data') or os.environ.get('ALFWORLD_DATA') or str(Path.home()/'.cache/alfworld')
        resources['public_discovery_contract'] = resource_contract(Path(data)/'logic/alfred.twl2')
    return resources


def freeze_release(prepared,checks):
    from experiments.protocol import hash_knowledge,hash_code,hash_config,code_file_manifest
    from ..knowledge.r103_protocol import METADATA
    from datetime import datetime,timezone
    if not checks.passed or sha(prepared.root/'release_checks.json')!=checks.checks_hash:
        raise ReleaseError('publication checks not verified')
    root=prepared.root;final=root/'frozen';temporary=root/'.freeze_build'
    if final.exists() or temporary.exists():raise FileExistsError(final)
    temporary.mkdir()
    srcbank=root/'work/data_v3'
    for p in srcbank.iterdir():
        if p.name.startswith('state.sqlite3'):continue
        if p.is_dir():shutil.copytree(p,temporary/p.name)
        else:shutil.copy2(p,temporary/p.name)
    with sqlite3.connect(srcbank/'state.sqlite3') as src,sqlite3.connect(temporary/'state.sqlite3') as dest:
        src.backup(dest)
        for rowid,old in dest.execute('SELECT rowid,file_path FROM artifact_index').fetchall():
            relative=Path(old).relative_to(srcbank)
            dest.execute('UPDATE artifact_index SET file_path=? WHERE rowid=?',(str(temporary/relative),rowid))
        for table in ('provisional_artifacts','failure_experiences'):
            for key,old in dest.execute(f'SELECT rowid,file_path FROM {table}').fetchall():
                dest.execute(f'UPDATE {table} SET file_path=? WHERE rowid=?',(str(temporary/Path(old).relative_to(srcbank)),key))
        dest.commit()
    with StateDatabase(temporary/'state.sqlite3',readonly=True,r103=True) as db:
        ArtifactStore(temporary,db).verify_all();verify_deployments(db,temporary)
    with sqlite3.connect(temporary/'state.sqlite3') as dest:
        for table in ('artifact_index','provisional_artifacts','failure_experiences'):
            for key,old in dest.execute(f'SELECT rowid,file_path FROM {table}').fetchall():
                dest.execute(f'UPDATE {table} SET file_path=? WHERE rowid=?',(str(final/Path(old).relative_to(temporary)),key))
        dest.commit()
        digest=hash_knowledge(temporary,database=dest)
    source=json.loads((root/'prepare_manifest.json').read_text())
    shutil.copy2(root/'release_checks.json',temporary/'release_checks.json')
    for name in ('source_inventory.json','asset_rewrite_manifest.jsonl','release_deployments.jsonl','identity_verification.json'):
        shutil.copy2(root/name,temporary/name)
    if source['protocol_version'] == OLDFIRST_PROTOCOL_VERSION:
        from .oldfirst_release import original_provenance
        provenance = original_provenance(root)
    else:
        provenance = json.loads((root/'source/unpacked/provenance/original_freeze_manifest.json').read_text())
    manifest={'protocol_version':source['protocol_version'],'seed':prepared.seed,'source_zip_hash':source['source_zip_hash'],
        'source_metadata':provenance,
        'knowledge_digest':digest,'code_hash':hash_code(Path(__file__).resolve().parents[3]),
        'execution_files':code_file_manifest(Path(__file__).resolve().parents[3]),
        'resource_hash':hash_config(release_resources(source['config'])),
        'protocol_metadata':METADATA,'checks_hash':checks.checks_hash,'input_authorization_version':'r103.caller-input.v1',
        'created_at':datetime.now(timezone.utc).isoformat(),'files':_bank_files(temporary)}
    _json(temporary/'release_manifest.json',manifest)
    temporary.rename(final)
    return FrozenRelease(final,prepared.seed,digest)

def load_release_source(path,config):
    from experiments.protocol import hash_knowledge,hash_code,hash_config
    from .release_protocol import _SOURCE_TOKEN
    path=Path(path).resolve();manifest=json.loads(path.read_text());bank=path.parent
    seed=config.get('bank_release',{}).get('source_seed')
    protocol = manifest.get('protocol_version')
    if config.get('bank_release', {}).get('protocol_version') != protocol:
        raise ReleaseError('configured release protocol differs from manifest')
    sources = OLDFIRST_SOURCE_ARCHIVES.get(seed, set()) if protocol == OLDFIRST_PROTOCOL_VERSION else {INPUT_HASHES.get(seed)}
    if protocol not in {PROTOCOL_VERSION, OLDFIRST_PROTOCOL_VERSION} or manifest.get('seed')!=seed or manifest.get('source_zip_hash') not in sources:
        raise ReleaseError('release source identity mismatch')
    if manifest['files']!=_bank_files(bank):raise ReleaseError('release files missing, modified or unexpected')
    if (bank/'native_call_contract.json').is_file():
        from ..agents.native_call_contract import NATIVE_CALL_CONTRACT_VERSION
        if json.loads((bank/'native_call_contract.json').read_text())['version'] != NATIVE_CALL_CONTRACT_VERSION:
            raise ReleaseError('native call contract version mismatch')
    if manifest['code_hash']!=hash_code(Path(__file__).resolve().parents[3]):raise ReleaseError('release evaluator code mismatch')
    resources=release_resources(config)
    if manifest['resource_hash']!=hash_config(resources):raise ReleaseError('release model/resource mismatch')
    with StateDatabase(bank/'state.sqlite3',readonly=True,r103=True) as db:
        ArtifactStore(bank,db).verify_all();verify_deployments(db,bank)
        if protocol == OLDFIRST_PROTOCOL_VERSION:
            from .preferences import verify_preferences
            verify_preferences(SkillRegistry(ArtifactStore(bank,db),db))
        actual=hash_knowledge(bank,database=db)
        if actual!=manifest['knowledge_digest'] or actual!=config['bank_release']['expected_bank_digest']:
            raise ReleaseError('release knowledge digest mismatch')
    return VerifiedFrozenSource(manifest,bank,_token=_SOURCE_TOKEN)

def make_configs(releases,profile,repeats=None):
    from experiments.protocol import hash_config
    repeats=repeats or {42:3,43:1,44:1}
    if profile not in {'current','lean'}:raise ReleaseError('unsupported release profile')
    output_root=releases[0].root.parent.parent
    comparison=output_root/'dev/comparison/summary.json'
    automatic=output_root/'dev/automation/acceptance.json'
    preparation = json.loads((releases[0].root.parent/'prepare_manifest.json').read_text())
    coverage_release = bool(preparation['config'].get('runtime', {}).get('preparation_coverage_version'))
    if coverage_release:
        if profile != 'current':
            raise ReleaseError('Release4 formal matrix must use its declared current profile')
        from experiments.release4_coverage import verify_coverage
        verify_coverage(output_root, write=True)
        comparison = output_root/'dev/coverage_acceptance.json'
        automatic = output_root/'dev/controlled_acceptance.json'
        paired = {}
    else:
        if not comparison.is_file() or not automatic.is_file():raise ReleaseError('paired dev and automatic acceptance must finish before formal configs')
        paired=json.loads(comparison.read_text())
    oldfirst = all(json.loads((r.root/'release_manifest.json').read_text())['protocol_version'] == OLDFIRST_PROTOCOL_VERSION for r in releases)
    expected_dev = 3 if oldfirst else 6
    if not coverage_release and (len(paired.get('pairs',[]))!=expected_dev or json.loads(automatic.read_text()).get('passed') is not True):
        raise ReleaseError('incomplete paired dev or automatic acceptance')
    if oldfirst and not coverage_release:
        from experiments.released_dev_checks import verify_dev
        verify_dev(output_root, write=True)
    plan=[]
    for release in releases:
        root=release.root.parent;output_root=root.parent
        base=json.loads((root/'prepare_manifest.json').read_text())['config']
        for rep in range(1,repeats[release.seed]+1):
            config=copy.deepcopy(base);config.pop('_config_path',None)
            output=output_root/'eval'/f'seed{release.seed}'/f'rep{rep:02}'
            config['data_dir']=str(release.root);config['trace_data_dir']=str(output)
            config['experiment'].update(name=output.name,experiment_kind='bank_release_eval',phase='test',
                runtime_mode='frozen',freeze_skills=True,allow_long_term_knowledge_writes=False,
                output_dir=str(output),task_manifest_path=str(output/'task_manifest.json'))
            config['experiment'].pop('source_train_run_dir',None)
            protocol = json.loads((release.root/'release_manifest.json').read_text())['protocol_version']
            config['bank_release']={'protocol_version':protocol,'source_seed':release.seed,
                'release_manifest':str(release.root/'release_manifest.json'),'expected_bank_digest':release.knowledge_digest}
            config['deployment']={'presentation_profile':profile,'presentation_version':'r103.frozen-expression.v2',
                'input_authorization_version':'r103.caller-input.v1','programs':True,'automatic_entry':True,'capture_final_request':True,'repeat_index':rep}
            path=output_root/'configs'/f'seed{release.seed}_rep{rep:02}.json'
            if path.exists():raise FileExistsError(path)
            # source guard is identical to the one used immediately before eval.
            load_release_source(release.root/'release_manifest.json',config)
            _json(path,config);plan.append({'seed':release.seed,'rep':rep,'config':str(path),'config_hash':hash_config(config),'output':str(output)})
    _json(output_root/'evaluation_plan.json',{'protocol_version':OLDFIRST_PROTOCOL_VERSION if oldfirst else PROTOCOL_VERSION,'profile':profile,'runs':plan,
        'paired_dev_hash':sha(comparison),'automatic_acceptance_hash':sha(automatic)})
    return plan
