"""Plan-locked publication from immutable original training archives."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from dataclasses import replace

from .release_protocol import (OLDFIRST_PROTOCOL_VERSION, OLDFIRST_INPUT_HASHES,
    OLDFIRST_SOURCE_ARCHIVES, PreparedRelease, ReleaseChecks, ReleaseError, DDL, sha, contained, verify_deployments)
from .bank_release import (_extract, _source_inventory, _import_artifact, _json, _stage, _graph_checks)
from .publication_audit import inventory, verify_preservation, identity_and_support, json_lines
from .oldfirst_revision import revise_atomic, author_program, author_implementation, revise_graph
from .preferences import static_closure, verify_preferences
from ..core.refs import SkillRef, ToolRef
from ..core.bindings import BindingExpression
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode, SkillStatus, ToolStatus
from ..knowledge.database import StateDatabase
from ..knowledge.artifact_store import ArtifactStore
from ..knowledge.skill_registry import SkillRegistry
from ..knowledge.tool_registry import ToolRegistry
from ..knowledge.identity_index import prepare_index_row
from ..harness.alfworld import AlfWorldAdapter
from ..evolution.identity_matching import (raw_hash, match_tool, verify_tool_proof,
    match_atomic, match_implementation, verify_implementation_proof)
from ..evolution.learning_interventions import identity_scope


def original_provenance(root):
    """Keep source chronology: a frozen DB can predate the final run-state update."""
    runs = json.loads((root/'source_run_manifests.json').read_text())
    trains = [r for r in runs if r['phase'] == 'train']
    original_freeze = root/'source/unpacked/data_v3/freeze_manifest.json'
    if original_freeze.is_file():
        manifest = json.loads(original_freeze.read_text())
        provenance = manifest['provenance']
        matched = [r for r in trains if r['run_id'] == provenance['source_run_id']
                   and r['code_commit'] == provenance['source_code_commit']]
        if (len(matched) != 1 or manifest['knowledge_digest'] != provenance['source_final_knowledge_digest']
                or provenance.get('r9_formal_freeze_audit_passed') is not True):
            raise ReleaseError('original frozen provenance disagrees with its source run')
        return {**manifest, 'original_run_manifest': matched[0],
                'source_freeze_manifest_hash': sha(original_freeze)}
    completed = [r for r in trains if r['state'] == 'completed']
    if len(completed) != 1:
        raise ReleaseError('original source lacks a verified freeze or completed training manifest')
    return {'provenance': {'source_run_id': completed[0]['run_id'],
                          'source_code_commit': completed[0]['code_commit']},
            'original_run_manifest': completed[0]}


def _copy_history(src, db, source, bank):
    archived = {'metadata', 'artifact_index', 'artifact_identity_index', 'run_manifests', 'run_tasks',
                'release_deployments', 'runtime_support_observations'}
    imported = {}
    for (table,) in src.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        if table in archived or table.startswith('sqlite_'):
            continue
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            raise ReleaseError(f'unknown historical table: {table}')
        columns = [r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
        if columns != [r[1] for r in db.execute(f'PRAGMA table_info("{table}")')]:
            raise ReleaseError(f'historical table columns differ: {table}')
        rows = src.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
        converted = []
        for row in rows:
            row = list(row)
            for locator in ('file_path', 'capsule_path', 'payload_path'):
                if locator not in columns:
                    continue
                index = columns.index(locator)
                old = str(row[index]).replace('\\', '/')
                relative = old.split('/data_v3/', 1)[-1] if '/data_v3/' in old else old
                original = contained(source, relative)
                if not original.is_file():
                    raise ReleaseError(f'missing historical payload: {table}:{relative}')
                target = contained(bank, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                row[index] = str(target) if locator == 'file_path' else relative
            converted.append(tuple(row))
        if converted:
            db.connection.executemany(f'INSERT INTO "{table}" VALUES({",".join("?" for _ in columns)})', converted)
        imported[table] = {'rows': len(rows), 'source_rows_hash': raw_hash([list(r) for r in rows]),
                           'imported_rows_hash': raw_hash([list(r) for r in converted])}
    db.connection.commit()
    return imported


def _rebindings(plan, assets, root):
    revised, proofs = [], []
    aliases, tool_aliases, blocked = {}, {}, []
    with identity_scope():
        for group in plan['merge_groups']:
            source, target = (assets[group[k]] for k in ('alias_tool_ref', 'canonical_tool_ref'))
            result = match_tool(source, target)
            if result.status != 'exact' or not verify_tool_proof(source, target, result.proof):
                raise ReleaseError(f'Tool identity does not verify: {source.ref}')
            proofs.append({'layer': 'tool', 'result': to_primitive(result)})
            tool_aliases[str(source.ref)] = str(target.ref)
            if group['action'] == 'blocked_equivalence_group':
                blocked.append([str(source.ref), str(target.ref)])
        for job in plan['identity_rebindings']:
            source = assets[job['source_implementation_ref']]
            target_tool = assets[job['canonical_tool_ref']]
            source_tool = assets[job['source_tool_ref']]
            proof = match_tool(source_tool, target_tool).proof
            if job['operation'] == 'alias_existing_implementation':
                target = assets[job['target_implementation_ref']]
                result = match_implementation(source, target,
                    source_atomic=assets[str(source.abstract_ref)], target_atomic=assets[str(target.abstract_ref)],
                    source_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in source.tool_bindings},
                    target_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in target.tool_bindings})
                verified = result.proof and verify_implementation_proof(source, target, result.proof,
                    source_atomic=assets[str(source.abstract_ref)], target_atomic=assets[str(target.abstract_ref)],
                    source_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in source.tool_bindings},
                    target_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in target.tool_bindings})
                if result.status != 'exact' or not verified:
                    raise ReleaseError(f'I identity does not verify: {source.ref}')
                if source.abstract_ref != target.abstract_ref:
                    raise ReleaseError(f'cross-Atomic alias requires explicit caller role adaptation: {source.ref}')
                aliases[str(source.ref)] = str(target.ref)
                proofs.append({'layer': 'implementation', 'result': to_primitive(result)})
                continue
            target = copy.deepcopy(source)
            target.ref = SkillRef.parse(job['target_implementation_ref'])
            target.quality = {}  # No empirical counters copied onto a new version.
            roles = {b.role for b in target.tool_bindings if b.tool_ref == source_tool.ref}
            target.tool_bindings = [replace(b, tool_ref=target_tool.ref,
                parameter_mapping={proof.input_role_map[r]: e for r, e in b.parameter_mapping.items()})
                if b.role in roles else b for b in target.tool_bindings]
            for out, raw in target.execution_policy.get('output_mapping', {}).items():
                expression = BindingExpression.from_dict(raw)
                if expression.kind.value == 'tool_output' and expression.source_step in roles:
                    target.execution_policy['output_mapping'][out] = replace(expression,
                        source_role=proof.output_role_map[expression.source_role])
            result = match_implementation(source, target, source_atomic=assets[str(source.abstract_ref)],
                target_atomic=assets[str(target.abstract_ref)],
                source_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in source.tool_bindings},
                target_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in target.tool_bindings})
            verified = result.proof and verify_implementation_proof(source, target, result.proof,
                source_atomic=assets[str(source.abstract_ref)], target_atomic=assets[str(target.abstract_ref)],
                source_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in source.tool_bindings},
                target_tools={str(b.tool_ref): assets[str(b.tool_ref)] for b in target.tool_bindings})
            if result.status != 'exact' or not verified:
                raise ReleaseError(f'Tool rebinding changed I semantics: {source.ref}')
            target.metadata = {'source_ref': str(source.ref), 'release_revision': OLDFIRST_PROTOCOL_VERSION,
                               'authoring_source': 'verified_tool_alpha_rebinding'}
            revised.append(('implementation', target, str(source.ref), source.status.value))
            aliases[str(source.ref)] = str(target.ref)
            proofs.append({'layer': 'implementation_rebinding', 'result': to_primitive(result)})
    for i, proof in enumerate(proofs):
        _json(root/'identity_proofs'/f'{i:02}.json', proof)
    return revised, tool_aliases, aliases, blocked


def prepare_oldfirst(spec, edit_plan):
    from experiments.protocol import hash_code, hash_config
    plan_path = Path(edit_plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['seed'] != spec.seed or plan['source_archive_sha256'] != OLDFIRST_INPUT_HASHES[spec.seed]:
        raise ReleaseError('edit plan source identity mismatch')
    actual_zip = sha(spec.input_zip)
    if actual_zip not in OLDFIRST_SOURCE_ARCHIVES[spec.seed]:
        raise ReleaseError('unknown original source archive')
    required = plan.get('required_protocols', {})
    for field, expected_version in required.items():
        section = 'harness' if field == 'public_discovery_version' else 'runtime'
        if spec.base_config.get(section, {}).get(field) != expected_version:
            raise ReleaseError(f'release config missing {section}.{field}')
    root = Path(spec.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = {'protocol_version': OLDFIRST_PROTOCOL_VERSION, 'seed': spec.seed,
        'source_zip_hash': actual_zip, 'declared_source_zip_hash': plan['source_archive_sha256'],
        'edit_plan_hash': sha(plan_path), 'code_hash': hash_code(Path(__file__).resolve().parents[3]),
        'config_hash': hash_config(spec.base_config)}
    _json(root/'build_identity.json', identity)
    shutil.copy2(plan_path, root/'edit_plan.lock.json')
    source = root/'source'
    source.mkdir()
    shutil.copy2(spec.input_zip, source/'input.zip')
    _extract(source/'input.zip', source/'unpacked')
    src, original = _source_inventory(source/'unpacked')
    expected = {a['ref']: a for a in plan['asset_dispositions']}
    if len(expected) != plan['source_asset_count'] or len(original) != len(expected):
        raise ReleaseError('source ref count differs from locked plan')
    for row, obj, payload, path in original:
        item = expected.get(str(obj.ref))
        if item is None or sha(path) != item['source_sha256'] or row['status'] != item['source_status']:
            raise ReleaseError(f'source ref/hash/status mismatch: {obj.ref}')
        obj.status = ToolStatus(row['status']) if row['artifact_kind'] == 'tool' else SkillStatus(row['status'])
    original_inventory = inventory(original, {})
    _json(root/'source_inventory.json', original_inventory)
    _stage(root, '01_import_identity', identity, original_inventory)
    bank = root/'work/data_v3'
    if required:
        from .bank_release import release_resources
        _json(bank/'public_discovery_contract.json', release_resources(spec.base_config)['public_discovery_contract'])
    db = StateDatabase(bank/'state.sqlite3', r103=True)
    db.connection.executescript(DDL); db.connection.commit()
    store = ArtifactStore(bank, db); skills = SkillRegistry(store, db)
    assets = {str(obj.ref): obj for _, obj, _, _ in original}
    source_hashes = {str(obj.ref): raw_hash(payload) for _, obj, payload, _ in original}
    changes = []
    try:
        for row, obj, payload, path in original:
            dest = _import_artifact(store, row['artifact_kind'], obj, row['status'])
            shutil.copy2(path, dest)  # Original payload bytes, not effective-status serialization.
            db.execute('UPDATE artifact_index SET content_hash=? WHERE artifact_ref=?', (row['content_hash'], row['artifact_ref']))
            db.connection.commit()
        db.connection.commit()
        history = _copy_history(src, db, source/'unpacked/data_v3', bank)
        _json(root/'historical_import.json', history)
        manifests = [dict(r) for r in src.execute('SELECT * FROM run_manifests')]
        _json(root/'source_run_manifests.json', manifests)
        additions, ta, ia, blocked = _rebindings(plan, assets, root)
        for job in [*plan['atomic_revision_jobs'], *plan['new_atomic_jobs']]:
            program = next(p for p in plan['program_realization_jobs'] if p['atomic_ref'] == job['target_ref'])
            atomic = revise_atomic(job, assets, program)
            assets[str(atomic.ref)] = atomic
            additions.append(('atomic', atomic, job.get('source_ref') or '', 'active'))
        for job in plan['program_realization_jobs']:
            atomic = assets[job['atomic_ref']]
            tool = author_program(job, atomic) if job.get('tool_ref') else None
            if tool:
                assets[str(tool.ref)] = tool
                additions.append(('tool', tool, job.get('source_tool_ref', ''), 'active'))
            impl = author_implementation(job, atomic, tool)
            additions.append(('implementation', impl, job.get('source_implementation_ref', ''), 'active'))
        for job in plan['workflow_targets']:
            if job['action'] == 'revise':
                graph = revise_graph(job, assets[job['source_ref']])
                assets[str(graph.ref)] = graph
                additions.append(('composite', graph, job['source_ref'], 'active'))
        from .oldfirst_plan import program_jobs, workflow_jobs, OPERATIONS
        resolved_refs = {}
        for job in plan.get('derived_revision_jobs', []):
            if OPERATIONS.get(job.get('operation')) != job['kind']:
                raise ReleaseError('unknown derived revision operation')
            if job['kind'] == 'atomic':
                atomic = revise_atomic(job, assets, {})
                assets[str(atomic.ref)] = atomic
                additions.append(('atomic', atomic, job['source_ref'], 'active'))
                resolved_refs[job['target_ref']] = str(atomic.ref)
            elif job['kind'] == 'program_realization':
                atomic = assets[resolved_refs.get(job['atomic_ref'], job['atomic_ref'])]
                tool = author_program(job, atomic)
                impl = author_implementation(job, atomic, tool)
                for kind, obj, source_key, target_key in (
                        ('tool', tool, 'source_tool_ref', 'tool_ref'),
                        ('implementation', impl, 'source_implementation_ref', 'implementation_ref')):
                    assets[str(obj.ref)] = obj
                    additions.append((kind, obj, job[source_key], 'active'))
                    resolved_refs[job[target_key]] = str(obj.ref)
            else:
                effective = copy.deepcopy(job)
                effective['action'] = 'revise'
                for node in effective['target_nodes']:
                    node['atomic_ref'] = resolved_refs.get(node['atomic_ref'], node['atomic_ref'])
                graph = revise_graph(effective, assets[job['source_ref']])
                assets[str(graph.ref)] = graph
                additions.append(('composite', graph, job['source_ref'], 'active'))
                resolved_refs[job['target_ref']] = str(graph.ref)
        _json(bank/'resolved_ref_map.json', resolved_refs)
        comparisons = []
        for kind, obj, source_ref, status in additions:
            if str(obj.ref) in expected:
                raise ReleaseError('new revision collides with immutable original')
            if kind == 'tool':
                from ..tooling.validator import ToolStaticValidator
                job = next(j for j in program_jobs(plan) if j.get('tool_ref') == str(obj.ref))
                report = ToolStaticValidator().validate_tool_asset(obj, assets[job['atomic_ref']], AlfWorldAdapter(split='train'))
                if not report.passed:
                    raise ReleaseError(f'{obj.ref}: {to_primitive(report)}')
                obj.safety.update(reviewed=True, review_basis='release_static_validation')
            if kind == 'composite':
                # Still Draft. Publication commits only after _check_selected
                # proves the prospective graph, including actual dataflow.
                obj.validator_spec['task_contract_covered'] = True
            _import_artifact(store, kind, obj, 'draft')
            assets[str(obj.ref)] = obj
            source_obj = assets.get(source_ref)
            if source_obj is not None and source_ref not in source_hashes:
                source_hashes[source_ref] = raw_hash(json.loads(store.path_for(kind, source_obj.ref).read_text()))
            if str(obj.ref) in resolved_refs.values() and source_obj is not None:
                before, after = to_primitive(source_obj), to_primitive(obj)
                matcher = (to_primitive(match_atomic(source_obj, obj)) if kind == 'atomic' else
                           to_primitive(match_tool(source_obj, obj)) if kind == 'tool' else None)
                comparisons.append({'kind': kind, 'source_ref': source_ref, 'target_ref': str(obj.ref),
                    'source_payload_hash': source_hashes[source_ref],
                    'changed_fields': {k: {'before': before.get(k), 'after': after.get(k)}
                        for k in before.keys() | after.keys() if before.get(k) != after.get(k)},
                    'identity_match': matcher, 'execution_credit_inherited': False,
                    'decision': 'version existing logical ID; changed guidance/program or declared workflow',
                    'required_protocols': required})
            changes.append({'kind': kind, 'released_ref': str(obj.ref), 'source_ref': source_ref,
                'source_payload_hash': source_hashes.get(source_ref, ''),
                'released_payload_hash': raw_hash(json.loads(store.path_for(kind, obj.ref).read_text())),
                'effective_status': status, 'basis': 'authored_revision'})
        _json(bank/'derived_revision_comparisons.json', comparisons)
        for kind, obj, _, _ in additions:
            destinations = ([('implements', str(obj.abstract_ref))] if kind == 'implementation' else
                            [('contains', str(o.node_ref)) for o in obj.occurrences] if kind == 'composite' else [])
            for relation, target in destinations:
                edge_id = raw_hash([str(obj.ref), relation, target])
                db.execute('INSERT OR IGNORE INTO graph_edges VALUES(?,?,?,?,?)',
                    (edge_id, str(obj.ref), target, relation, json.dumps({'source': OLDFIRST_PROTOCOL_VERSION})))
        db.connection.commit()
        preferences = {'protocol_version': OLDFIRST_PROTOCOL_VERSION, 'edit_plan_hash': sha(plan_path),
            'canonical_tool_aliases': ta, 'canonical_implementation_aliases': ia,
            'preferred_implementations': [], 'blocked_equivalent_groups': blocked,
            'preferred_composites': [{'composite_ref': j['target_ref'], 'priority': 100} for j in plan['workflow_targets']]}
        if plan.get('derived_revision_jobs'):
            patch = plan['deployment_preferences_patch']
            preferences['preferred_composites'].append({'composite_ref': resolved_refs.get(
                patch['prefer_workflow_ref'], patch['prefer_workflow_ref']), 'priority': 200})
        for job in program_jobs(plan):
            preferences['preferred_implementations'].append({'atomic_ref': job['atomic_ref'],
                'implementation_ref': job['implementation_ref'],
                'applicable_condition': 'ready' if job.get('route_preference') else 'any',
                'priority': 200 if job.get('route_preference') else 100})
        _json(bank/'deployment_preferences.json', preferences)
        shutil.copy2(plan_path, bank/'edit_plan.lock.json')
        for ref, permission in plan['manual_publications'].items():
            item = expected[ref]
            changes.append({'kind': item['kind'], 'released_ref': ref, 'source_ref': ref,
                'source_payload_hash': source_hashes[ref], 'released_payload_hash': source_hashes[ref],
                'effective_status': permission['target_effective_status'], 'basis': 'curated_existing'})
        _stage(root, '02_revisions', identity, changes)
        with db.transaction() as connection:
            for change in changes:
                connection.execute('UPDATE artifact_index SET status=? WHERE artifact_ref=?',
                    (change['effective_status'], change['released_ref']))
            # Real compiler / graph checks happen against this prospective view.
            checked = _check_selected(skills, db, plan)
            if plan.get('derived_revision_jobs'):
                from .preparation_coverage import audit_coverage
                coverage = audit_coverage(skills, required_composites=[j['target_ref'] for j in workflow_jobs(plan)
                    if j.get('operation') == 'version_existing_workflow'])
                _json(bank/'preparation_coverage.json', coverage)
                if not coverage['passed']:
                    raise ReleaseError(f'preparation coverage failed: {coverage["blocking_gaps"]}')
            for change in changes:
                ref = change['released_ref']
                report = {'passed': True, 'protocol_version': OLDFIRST_PROTOCOL_VERSION,
                    'source_ref': change['source_ref'], 'source_status': expected.get(ref, {}).get('source_status', ''),
                    'source_file_hash': expected.get(ref, {}).get('source_sha256', ''),
                    'target_status': change['effective_status'], 'edit_plan_hash': sha(plan_path),
                    'validation': checked.get(ref, {'identity_rebinding': ref in ia.values()})}
                path = bank/'publication_checks'/f'{raw_hash(ref)}.json'
                _json(path, report)
                connection.execute('INSERT INTO release_deployments VALUES(?,?,?,?,?,?,?,?,?)',
                    (ref, change['kind'], change['effective_status'], change['basis'], change['source_ref'],
                     change['source_payload_hash'], change['released_payload_hash'], str(path.relative_to(bank)), sha(path)))
            verify_deployments(db, bank)
        for kind in ('atomic', 'tool', 'implementation'):
            for row in db.rows("SELECT * FROM artifact_index WHERE artifact_kind=? ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END,artifact_ref", (kind,)):
                from .bank_release import CLASSES
                from ..core.serialization import dataclass_from_dict
                payload = json.loads(Path(row['file_path']).read_text())
                obj = dataclass_from_dict(CLASSES[kind], payload)
                rowdata = prepare_index_row(bank, kind, obj, payload, db)
                if rowdata is None:
                    raise ReleaseError(f'missing identity row: {obj.ref}')
                db.execute('INSERT INTO artifact_identity_index VALUES(?,?,?,?,?,?,?,?,?,?)', rowdata)
                db.connection.commit()
        identity_and_support(db, bank, root)
        preserved = verify_preservation(db, bank, original_inventory, plan['manual_publications'])
        store.verify_all()
        _json(root/'asset_rewrite_manifest.json', changes)
        json_lines(root/'asset_rewrite_manifest.jsonl', changes)
        json_lines(root/'asset_changes.jsonl', changes)
        json_lines(root/'unchanged_assets.jsonl', [p for p in preserved if not p['status_changed']])
        json_lines(root/'candidate_publications.jsonl', [c for c in changes if c['basis'] == 'curated_existing'])
        json_lines(root/'release_deployments.jsonl', [dict(r) for r in db.rows('SELECT * FROM release_deployments')])
        _json(root/'parameter_output_mapping_checks.json', checked)
        _json(root/'prepare_manifest.json', {**identity, 'config': spec.base_config, 'status': 'prepared',
            'imported_history_tables': history, 'original_asset_count': len(original)})
        _stage(root, '03_publication', {'changes': raw_hash(changes)}, {'checked': raw_hash(checked), 'preserved': preserved})
    finally:
        src.close(); db.close()
    prepared = PreparedRelease(root, spec.seed)
    verify_oldfirst(prepared)
    return prepared


def _check_selected(skills, db, plan):
    from .oldfirst_plan import program_jobs, workflow_jobs
    from ..runtime.invocation_compiler import InvocationCompiler
    from ..planner.compiler import PlanCompiler
    from ..planner.validator import PlannerValidator
    from ..knowledge.graph_store import GraphStore
    from ..knowledge.query import complete_composite_contract_diagnosis
    from ..harness.protocol import HarnessTask
    from ..tooling.validator import ToolStaticValidator
    from ..validation.tool_validator import ToolValidator
    harness = AlfWorldAdapter(split='train')
    compiler = InvocationCompiler(skills, ToolRegistry(skills.store, db), harness, mode=RuntimeMode.FROZEN)
    reports = {}
    for impl in skills.implementations():
        selected = str(impl.ref) in plan['manual_publications'] or any(
            j['implementation_ref'] == str(impl.ref) for j in program_jobs(plan))
        if not selected:
            continue
        atomic = skills.get_atomic(impl.abstract_ref)
        tools = [compiler.tools.get(b.tool_ref) for b in impl.tool_bindings]
        reports[str(impl.ref)] = {'passed': True, 'compiled_interface': to_primitive(compiler.compile(atomic, impl, tools, {}))}
        for tool in tools:
            # A serial Tool is checked against its own declared final effects;
            # only the completed Implementation has the aggregate Atomic effect.
            contracts = {str(atomic.ref): atomic}
            for owner in skills.implementations():
                if len(owner.tool_bindings) == 1 and owner.tool_bindings[0].tool_ref == tool.ref:
                    contracts[str(owner.abstract_ref)] = skills.get_atomic(owner.abstract_ref)
            static_reports = {ref: ToolStaticValidator().validate_tool_asset(tool, contract, harness)
                              for ref, contract in contracts.items()}
            if not any(r.passed for r in static_reports.values()):
                raise ReleaseError(f'{tool.ref}: {to_primitive(static_reports)}')
            local = ToolValidator().validate_asset(tool)
            if not local.passed:
                raise ReleaseError(f'{tool.ref}: {to_primitive(local)}')
            reports[str(tool.ref)] = {'passed': True, 'local_asset': to_primitive(local),
                'validated_tool_contracts': [ref for ref, result in static_reports.items() if result.passed]}
    # Fixed publication fixtures validate contract shape only, never Runtime dispatch.
    examples = {'simple': ('pick_and_place_simple', 'put an object in a destination'),
        'clean': ('pick_clean_then_place_in_recep', 'put a clean object in a destination'),
        'heat': ('pick_heat_then_place_in_recep', 'put a hot object in a destination'),
        'cool': ('pick_cool_then_place_in_recep', 'put a cold object in a destination'),
        'look': ('look_at_obj_in_light', 'look at an object under a light'),
        'two': ('pick_two_obj_and_place', 'put two objects in a destination')}
    for job in workflow_jobs(plan):
        graph = skills.get_composite(job['target_ref'])
        family, goal = examples[job['goal_group']]
        task = HarnessTask('publication-contract-check', goal, 'alfworld', family)
        formal = harness.task_contract(task)
        if not complete_composite_contract_diagnosis(formal, graph.goal_contract).passed:
            raise ReleaseError(f'goal shape changed: {graph.ref}')
        runtime_plan = PlanCompiler(skills).from_composite(task, formal, graph, mode=RuntimeMode.FROZEN, audit={})
        validation = PlannerValidator(skills, GraphStore(db, skills)).validate(runtime_plan, mode=RuntimeMode.FROZEN,
            harness_profile=harness.profile_name, task_binding_roles={'object', 'destination', 'light_source'}, literal_authorities={})
        if not validation.passed:
            raise ReleaseError(f'{graph.ref}: {to_primitive(validation)}')
        detached = copy.deepcopy(graph)
        # Declared formal role correspondence for this publication fixture.
        detached.goal_contract = copy.deepcopy(formal)
        from .oldfirst_revision import binding
        detached.goal_contract.target_effects = [replace(e, args={role: binding(
            {'location': 'destination', 'light': 'light_source'}.get(role, role)) for role in e.args})
            for e in formal.target_effects]
        checked = _graph_checks(detached, skills)
        closure = static_closure(skills, graph, harness, RuntimeMode.FROZEN)
        if not closure['program_static_closure']:
            raise ReleaseError(f'incomplete selected graph programs: {graph.ref}: {closure}')
        reports[str(graph.ref)] = {**checked, **closure, 'formal_validation': to_primitive(validation)}
    return reports


def verify_oldfirst(prepared):
    from experiments.protocol import (active_composite_frozen_closure_audit, atomic_output_derivation_audit,
        active_repeat_identity_closure_audit, composite_deployment_evidence_audit)
    root = prepared.root; bank = root/'work/data_v3'
    plan = json.loads((root/'edit_plan.lock.json').read_text())
    with StateDatabase(bank/'state.sqlite3', readonly=True, r103=True) as db:
        store = ArtifactStore(bank, db); skills = SkillRegistry(store, db)
        store.verify_all(); verify_deployments(db, bank)
        verify_preferences(skills)
        verify_preservation(db, bank, json.loads((root/'source_inventory.json').read_text()), plan['manual_publications'])
        audits = {'children': active_composite_frozen_closure_audit(db, skills),
            'outputs': atomic_output_derivation_audit(skills), 'repeat': active_repeat_identity_closure_audit(db, skills),
            'history': composite_deployment_evidence_audit(db)}
        published = {r['artifact_ref'] for r in db.rows('SELECT * FROM release_deployments')}
        violations = []
        for kind, audit in audits.items():
            for violation in audit.get('violations', []):
                if kind == 'history' and violation.get('composite_ref', violation.get('artifact_ref')) in published:
                    # Only the empirical qualification threshold can be replaced by
                    # authorship. Counts/projections are still independently checked.
                    if violation.get('reason') == 'active_composite_lacks_deployment_successes':
                        continue
                violations.append({'audit': kind, **violation})
        reports = _check_selected(skills, db, plan)
        if plan.get('derived_revision_jobs'):
            from .preparation_coverage import audit_coverage
            from .oldfirst_plan import workflow_jobs
            coverage = audit_coverage(skills, required_composites=[j['target_ref'] for j in workflow_jobs(plan)
                if j.get('operation') == 'version_existing_workflow'])
            if coverage != json.loads((bank/'preparation_coverage.json').read_text()) or not coverage['passed']:
                raise ReleaseError('preparation coverage changed since publication')
        _json(root/'release_checks.json', {'passed': not violations, 'violations': violations, 'audits': audits})
        json_lines(root/'workflow_closure.jsonl', [{'ref': ref, **r} for ref, r in reports.items() if 'nodes' in r])
        json_lines(root/'effective_routes.jsonl', [{'ref': ref, **r} for ref, r in reports.items() if 'compiled_interface' in r])
        if violations:
            raise ReleaseError(f'oldfirst release audit failed: {violations}')
    return ReleaseChecks(True, sha(root/'release_checks.json'))
