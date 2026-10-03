"""Train-only qualification, orthogonal to online reliability and lifecycle.

Evidence is produced by the SAME production replay/ToolRunner on an isolated
Harness instance. Two independent physical Train tasks and initial states are
mandatory; cached certificates, admission booleans and terminal prefixes are not
fresh qualification executions. Nothing here grants current grounding.
"""
import json
import time
from pathlib import Path
from ..core.refs import content_hash
from ..core.serialization import atomic_create_json, read_json, to_primitive
from ..core.status import RuntimeMode, skill_status_usable, tool_status_usable
from ..governance.ledger import EvidenceEvent, EvidenceEventType
from ..mechanism_profile import KEY, VERSION as PROFILE

VERSION = 'skillcompiler.train-qualification.v2'


def profile_enabled(database):
    row = database.execute('SELECT value FROM metadata WHERE key=?', (KEY,)).fetchone()
    return bool(row and json.loads(row[0]).get('version') == PROFILE)


def verify(event, database):
    metadata = event.metadata
    certificate = metadata.get('certificate', {})
    if (not profile_enabled(database) or metadata.get('version') != VERSION
            or metadata.get('certificate_hash') != content_hash(certificate)
            or event.artifact_ref not in certificate.get('artifact_hashes', {})):
        raise ValueError('invalid qualification profile/certificate identity')
    path = database.path.parent / 'artifacts' / 'deployment_qualification' / (metadata['certificate_hash'] + '.json')
    if read_json(path) != certificate:
        raise ValueError('qualification capsule mismatch')
    for ref, expected in certificate['artifact_hashes'].items():
        row = database.execute('SELECT content_hash FROM artifact_index WHERE artifact_ref=?', (ref,)).fetchone()
        if row is None or row[0] != expected:
            raise ValueError('qualification executable/interface changed')
        from ..knowledge.artifact_store import ArtifactStore
        ArtifactStore(database.path.parent, database).verify_ref(ref)
    if certificate.get('kind') == 'composition_closure':
        report = composite_closure(database, event.artifact_ref, certificate['harness_profile'])
        if not report['passed'] or report != certificate['closure']:
            raise ValueError('qualification graph no longer has full contract/dependency closure')
        return certificate
    if (certificate.get('version') != VERSION or certificate.get('accounting_class') != 'offline_train_qualification'
            or certificate.get('online_direct_success_credit') is not False or certificate.get('llm_requests') != 0):
        raise ValueError('qualification accounting/protocol invalid')
    cases = certificate.get('cases', [])
    if len(cases) != 2 or len({c['independent_task_key'] for c in cases}) != 2 or len({c['initial_state_digest'] for c in cases}) != 2:
        raise ValueError('qualification requires two distinct Train tasks AND initial scenes')
    for case in cases:
        source = database.execute('SELECT * FROM learning_source_index WHERE sample_key=?', (case['sample_key'],)).fetchone()
        if (source is None or source['independent_task_key'] != case['independent_task_key']
                or source['source_trace_hash'] != case['source_trace_hash']):
            raise ValueError('qualification source not committed Train authority')
        source_path = database.path.parent / source['capsule_path']
        capsule = read_json(source_path)
        from ..evolution.identity_matching import raw_hash
        from ..evolution.learning_interventions import training_source
        identity = capsule['source_identity']
        if raw_hash(capsule) != source['capsule_hash'] or identity.get('split') != 'train' or not training_source(identity):
            raise ValueError('qualification may not use Dev/Test or unverified sources')
        from ..knowledge.source_snapshots import read as read_parent
        read_parent(database.path.parent, source['source_trace_hash'], source['source_trace_id'])
        result = case['result']
        if (result.get('passed') is not True or result.get('started') is not True
                or result.get('completed') is not True or result.get('atomic_effect_passed') is not True
                or result.get('output_validation_passed') is not True or result.get('failure_code')
                or result.get('terminal_interrupted') or result.get('stage') != 'final_validation'
                or result.get('executed_action_count', 0) <= 0 or case.get('source_world_unchanged') is not True):
            raise ValueError('qualification lacks a complete restored production execution')
        if content_hash(case['production_trace']) != case['production_trace_hash']:
            raise ValueError('qualification execution Trace changed')
        raw = case['production_trace']
        if raw['task']['task_id'] != result['resolved_task_id'] or result['source_task_id'] != raw['task']['task_id']:
            raise ValueError('qualification execution task/source mismatch')
        executions = [e for e in raw.get('tool_executions', []) if e['tool_ref'] in certificate['artifact_hashes']]
        if len(executions) != 1:
            raise ValueError('qualification lacks unique production ToolRunner record')
        executed = executions[0]['result']
        # executed_action_count is a production Result property; the immutable
        # dataclass payload stores executed_step_count, not that alias.
        from ..core.results import ToolExecutionResult
        if ToolExecutionResult(**executed).executed_action_count != result['executed_action_count']:
            raise ValueError('qualification action count disagrees with ToolRunner Trace')
        for field in ('started', 'completed', 'atomic_effect_passed', 'terminal_interrupted', 'failure_code'):
            if executed.get(field) != result.get(field):
                raise ValueError('qualification result disagrees with ToolRunner Trace: ' + field)
        if raw.get('agent_turns') or raw.get('llm_calls'):
            raise ValueError('qualification cannot introduce model decisions')
        from ..knowledge.artifact_store import ArtifactStore
        from ..knowledge.skill_registry import SkillRegistry
        from ..runtime.input_authorization import validate_declarations
        registry = SkillRegistry(ArtifactStore(database.path.parent, database), database)
        atomic_refs = [ref for ref in certificate['artifact_hashes'] if database.execute(
            'SELECT artifact_kind FROM artifact_index WHERE artifact_ref=?', (ref,)).fetchone()[0] == 'atomic']
        if len(atomic_refs) != 1:
            raise ValueError('qualification must name one Atomic interface')
        atomic_ref = atomic_refs[0]
        controls = validate_declarations(registry.get_atomic(atomic_ref))
        checks = raw.get('metadata', {}).get('qualification_control_checks', [])
        if set(controls) != {c['role'] for c in checks} or len(checks) != len(controls):
            raise ValueError('qualification lacks declared program control authorization')
    return certificate


def qualified(database, ref, *, visiting=None):
    if not profile_enabled(database):
        return False
    visiting = set(visiting or ())
    if ref in visiting:
        return False
    visiting.add(ref)
    rows = database.rows('SELECT * FROM evidence_events WHERE artifact_ref=? AND event_type=? ORDER BY rowid DESC',
                         (ref, EvidenceEventType.DEPLOYMENT_QUALIFIED.value))
    for row in rows:
        try:
            certificate = verify(EvidenceEvent.from_row(row), database)
        except (KeyError, TypeError, ValueError):
            continue  # Changed interface/AST invalidates qualification, not lifecycle.
        if ref in certificate['artifact_hashes']:
            return True
    return False


def usable(database, ref, status, mode, kind):
    legacy = tool_status_usable if kind == 'tool' else skill_status_usable
    if legacy(status, mode):
        return True
    if RuntimeMode(mode) is not RuntimeMode.FROZEN or str(getattr(status, 'value', status)) != 'candidate':
        return False
    return qualified(database, str(ref))


def registry_usable(registry, ref, status, mode, kind='atomic'):
    """One deployment policy; legacy registry protocol remains supported."""
    database = getattr(registry, 'database', None)
    if database is None:
        return (tool_status_usable if kind == 'tool' else skill_status_usable)(status, mode)
    return usable(database, ref, status, mode, kind)


def qualify_routes(system, *, limit=1):
    if not system.mechanism_profile or system.readonly:
        return []
    offers = []
    for implementation in system.skills.implementations():
        if implementation.status.value != 'candidate' or len(implementation.tool_bindings) != 1:
            continue
        tool = system.tools.get_with_replay_evidence(implementation.tool_bindings[0].tool_ref)
        if qualified(system.database, str(implementation.ref)) or tool.status.value not in {'candidate', 'active', 'preferred'}:
            continue
        from ..evolution.replay_certificates import ReplayCertificates
        from ..governance.ledger import EvidenceLedger
        cases = {}
        candidates = [*tool.tests, *ReplayCertificates(EvidenceLedger(system.database)).cases_for(tool, system.harness.profile_name)]
        for case in candidates:
            row = system.database.execute('SELECT * FROM learning_source_index WHERE source_trace_id=? ORDER BY sample_key LIMIT 1',
                                          (case.get('trace_id', ''),)).fetchone()
            if row is not None:
                system.learning_source_store.verified(system, row)
                cases.setdefault(row['independent_task_key'], (case, dict(row)))
        if len(cases) >= 2:
            offers.append((str(tool.ref), implementation, tool, list(cases.values())[:2]))
    outcomes = []
    for _, implementation, tool, cases in sorted(offers, key=lambda x: x[0])[:limit]:
        from ..evolution.realization_queue import RealizationQueue
        identity = {'protocol': VERSION, 'implementation_ref': str(implementation.ref),
                    'tool_ref': str(tool.ref), 'case_hashes': [content_hash(c) for c, _ in cases]}
        queue = RealizationQueue(system.database)
        queue_key, claim, _ = queue.claim(identity)
        if not claim:
            continue
        start = time.monotonic()
        main_harness = system.harness
        before = main_harness.capture_runtime_checkpoint().state_digest
        from ..harness.registry import create_harness
        isolated = getattr(system, 'qualification_harness_factory', lambda: create_harness(system.config))()
        captures = []
        failure = None
        try:
            system.harness = isolated
            system._qualification_capture = True
            system._qualification_implementation = implementation
            for case, source in cases:
                # Source resolution is the existing immutable Trace/manifest authority.
                task = system._replay_source_authority().resolve(case, current_task=None, current_trace=None)
                result = system._replay_tool_candidate_result(task, tool, case, requested_task_id=task.task_id)
                captures.append({'sample_key': source['sample_key'], 'source_trace_hash': source['source_trace_hash'],
                    'independent_task_key': source['independent_task_key'],
                    'initial_state_digest': system._qualification_initial_digest,
                    'result': to_primitive(result), 'production_trace': to_primitive(system._qualification_trace),
                    'production_trace_hash': content_hash(system._qualification_trace),
                    'case': case, 'source_world_unchanged': True})
        except (KeyError, TypeError, ValueError) as exc:
            failure = str(exc)
        finally:
            system._qualification_capture = False
            system._qualification_implementation = None
            system.harness = main_harness
            close = getattr(isolated, '_close_backend', None) or getattr(isolated, 'close', None)
            if close:
                close()
        unchanged = main_harness.capture_runtime_checkpoint().state_digest == before
        if failure is not None:
            queue.finish(queue_key, status='qualification_rejected', failure_codes=[failure])
            outcomes.append({'status': 'rejected', 'reason': failure})
            continue
        for capture in captures:
            capture['source_world_unchanged'] = unchanged
        refs = [str(implementation.abstract_ref), str(implementation.ref), str(tool.ref)]
        hashes = {ref: system.database.execute('SELECT content_hash FROM artifact_index WHERE artifact_ref=?', (ref,)).fetchone()[0]
                  for ref in refs}
        certificate = {'version': VERSION, 'artifact_hashes': hashes, 'cases': captures,
            'harness_profile': system.harness.profile_name, 'applicability': 'original immutable entry/input/output contract; current preflight required',
            'accounting_class': 'offline_train_qualification', 'elapsed_seconds': time.monotonic() - start,
            'llm_requests': 0, 'online_direct_success_credit': False}
        digest = content_hash(certificate)
        atomic_create_json(system.data_dir / 'artifacts/deployment_qualification' / (digest + '.json'), certificate)
        events = [EvidenceEvent.create(task_id=cases[0][0]['source_task']['task_id'], trace_id='qualification_' + digest,
            occurrence_id='qualification', attempt_id='qualification_' + digest, sequence_no=i,
            artifact_ref=ref, artifact_kind=['atomic', 'implementation', 'tool'][i],
            event=EvidenceEventType.DEPLOYMENT_QUALIFIED,
            metadata={'version': VERSION, 'certificate_hash': digest, 'certificate': certificate}) for i, ref in enumerate(refs)]
        try:
            for event in events:
                verify(event, system.database)
        except (KeyError, TypeError, ValueError) as exc:
            queue.finish(queue_key, status='qualification_rejected', failure_codes=[str(exc)])
            outcomes.append({'status': 'rejected', 'certificate_hash': digest, 'reason': str(exc)})
            continue
        system._commit_evidence(events)
        queue.finish(queue_key, status='deployment_qualified', result_refs=refs)
        outcomes.append({'status': 'qualified', 'certificate_hash': digest, 'refs': refs})
    return outcomes


def composite_closure(database, ref, harness_profile):
    """A composition earns no execution credit: re-prove the complete static contract."""
    from types import SimpleNamespace
    from ..knowledge.artifact_store import ArtifactStore
    from ..knowledge.skill_registry import SkillRegistry
    from ..knowledge.graph_store import GraphStore
    from ..planner.compiler import PlanCompiler
    from ..planner.validator import PlannerValidator
    skills = SkillRegistry(ArtifactStore(database.path.parent, database), database)
    graph = skills.get_composite(ref)
    if graph.metadata.get('completion_authority', {}).get('kind', 'complete_contract') != 'complete_contract':
        return {'passed': False, 'reason': 'partial_completion_authority'}
    # Existing edge provenance is checked against committed Active graph evidence,
    # not against this not-yet-qualified graph itself (no circular certification).
    authority = GraphStore(database, skills)
    authority.skills = SimpleNamespace(composites=lambda **kw: [g for g in skills.composites()
        if g.status.value == 'active'], get_atomic=skills.get_atomic)
    try:
        plan = PlanCompiler(skills).from_composite(SimpleNamespace(task_id='qualification_closure'),
            graph.goal_contract, graph, mode='frozen', audit={})
        report = PlannerValidator(skills, authority).validate(plan, mode='frozen', harness_profile=harness_profile)
        dependencies = []
        for node in graph.occurrences:
            atomic = skills.get_atomic(node.node_ref)
            if not usable(database, str(atomic.ref), atomic.status, 'frozen', 'atomic'):
                return {'passed': False, 'reason': 'child_not_deployable', 'child': str(atomic.ref)}
            implementations = skills.implementations_for(node.node_ref, mode='frozen')
            routes = []
            for impl in implementations:
                if all(usable(database, str(b.tool_ref), database.execute(
                        'SELECT status FROM artifact_index WHERE artifact_ref=?', (str(b.tool_ref),)).fetchone()[0],
                        'frozen', 'tool') for b in impl.tool_bindings) and impl.tool_bindings:
                    routes.append(str(impl.ref))
            dependencies.append({'step_id': node.step_id, 'atomic_ref': str(node.node_ref), 'routes': sorted(routes)})
        return {'passed': report.passed and bool(graph.goal_contract.target_effects)
            and bool(dependencies) and all(d['routes'] for d in dependencies),
            'validation': to_primitive(report), 'dependencies': dependencies}
    except (KeyError, TypeError, ValueError) as exc:
        return {'passed': False, 'reason': str(exc)}


def qualify_composites(system):
    if not system.mechanism_profile or system.readonly:
        return []
    out = []
    for graph in system.skills.composites():
        if graph.status.value != 'candidate' or qualified(system.database, str(graph.ref)):
            continue
        closure = composite_closure(system.database, str(graph.ref), system.harness.profile_name)
        if not closure['passed']:
            out.append({'ref': str(graph.ref), 'status': 'blocked', 'closure': closure})
            continue
        # Dependencies carry production qualification/online reliability; this
        # only certifies composition structure, never "two whole-task successes".
        refs = {str(graph.ref)}
        for dependency in closure['dependencies']:
            refs.add(dependency['atomic_ref'])
            for ref in dependency['routes']:
                refs.add(ref)
                refs.update(str(b.tool_ref) for b in system.skills.get_implementation(ref).tool_bindings)
        certificate = {'version': VERSION, 'kind': 'composition_closure',
            'artifact_hashes': {ref: system.database.execute(
                'SELECT content_hash FROM artifact_index WHERE artifact_ref=?', (ref,)).fetchone()[0] for ref in sorted(refs)},
            'harness_profile': system.harness.profile_name, 'closure': closure,
            'llm_requests': 0, 'online_direct_success_credit': False}
        digest = content_hash(certificate)
        atomic_create_json(system.data_dir / 'artifacts/deployment_qualification' / (digest + '.json'), certificate)
        event = EvidenceEvent.create(task_id='qualification_closure', trace_id='qualification_' + digest,
            occurrence_id='qualification_closure', attempt_id='qualification_' + digest, sequence_no=0,
            artifact_ref=str(graph.ref), artifact_kind='composite', event=EvidenceEventType.DEPLOYMENT_QUALIFIED,
            metadata={'version': VERSION, 'certificate_hash': digest, 'certificate': certificate})
        system._commit_evidence([event])
        out.append({'ref': str(graph.ref), 'status': 'qualified', 'certificate_hash': digest})
    return out
