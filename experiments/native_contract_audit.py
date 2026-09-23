"""Read-only production constructors and request/validation/usage reconciliation."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from atomic_skillgraph.agents.native_call_contract import NativeCallContractView, NATIVE_CALL_CONTRACT_VERSION, canonical
from atomic_skillgraph.deployment.release_protocol import ReleaseError


def historical_regression(path):
    from atomic_skillgraph.agents.protocol import NativeToolSpec, NativeToolCall, AgentTurn
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents.usage import UsageLedger
    from atomic_skillgraph.core.errors import AgentProtocolError
    class NeverCalled:
        def complete(self, *args, **kwargs):
            raise AssertionError('offline protocol regression must not call provider')
        def snapshot(self):
            return {'offline': True}
    rows = []
    for original in json.loads(Path(path).read_text()):
        session = ReplayAgentSession(NeverCalled(), system_prompt='offline schema regression',
            usage_ledger=UsageLedger(), usage_bucket='runtime_seeded')
        specs = [NativeToolSpec(t['name'], t['description'], t['parameters']) for t in original['offered_matching_tool']]
        turn = AgentTurn('', [NativeToolCall(**c) for c in original['tool_calls']], 'tool_calls', 0, 0, 0, None, 0)
        try:
            session._validate_turn(turn, specs)
        except AgentProtocolError as exc:
            rows.append({'seed': original['seed'], 'task_id': original['task_id'], 'session_id': original['session_id'],
                'turn_index': original['turn_index'], 'error_code': exc.code,
                'passed': exc.code == original['code'], 'diagnostic': exc.diagnostics})
        else:
            rows.append({'passed': False, 'reason': 'historical rejection unexpectedly accepted'})
    if len(rows) != 35 or not all(r['passed'] for r in rows):
        raise ReleaseError('historical protocol regression failed')
    return {'passed': True, 'scope': 'offline production validator, not environment success',
            'count': len(rows), 'categories': dict(Counter(r['error_code'] for r in rows)), 'rows': rows}


def audit_bank(bank):
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
    from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
    from atomic_skillgraph.runtime.node_executor import NodeExecutor
    from atomic_skillgraph.runtime.runtime_step import automation_request_tool
    from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
    from atomic_skillgraph.core.status import skill_status_usable, tool_status_usable
    from atomic_skillgraph.core.bindings import BindingStatus
    from atomic_skillgraph.runtime.support_call_surface import SupportCallSurface, SupportCallOption, VERSION
    rows, unavailable = [], []
    def record(tool, constructor, **extra):
        rows.append({**NativeCallContractView.from_tool(tool).to_dict(), 'constructor': constructor,
            'dispatcher': 'NodeExecutor/RuntimeStep' if tool.scope != 'structured' else 'StructuredSubmission',
            'feedback_producer': 'ReplayAgentSession._validate_turn', 'audit_consumer': 'audit_trace', **extra})
    with StateDatabase(Path(bank)/'state.sqlite3', readonly=True, r103=True) as db:
        skills = SkillRegistry(ArtifactStore(bank, db), db)
        tools = ToolRegistry(skills.store, db)
        compiler = InvocationCompiler(skills, tools, AlfWorldAdapter(split='train'), mode='frozen')
        for impl in skills.implementations():
            atomic = skills.get_atomic(impl.abstract_ref)
            dependencies = [tools.get(b.tool_ref) for b in impl.tool_bindings]
            if not (skill_status_usable(impl.status, 'frozen') and skill_status_usable(atomic.status, 'frozen')
                    and all(tool_status_usable(t.status, 'frozen') for t in dependencies)):
                unavailable.append({'ref': str(impl.ref), 'status': 'not_deployable'})
                continue
            for bound in (False, True):
                bindings = {p.name: SimpleNamespace(status=BindingStatus.GROUNDED, resolution=p.required_resolution)
                            for p in atomic.inputs} if bound else {}
                spec = compiler.compile(atomic, impl, dependencies, bindings)
                record(NodeExecutor._invocation_tool(SimpleNamespace(spec=spec)), 'InvocationCompiler.compile',
                       implementation_ref=str(impl.ref), inputs_bound=bound)
            for scope in ('node', 'task'):
                option = SupportCallOption('audit_schema_option', canonical({'atomic_ref': str(atomic.ref), 'input_schema': spec.input_schema}))
                for options in ((), (option,)):
                    surface = SupportCallSurface(VERSION, scope, 'audit', str(atomic.ref), 'audit', 0, 'audit', options, '[]')
                    native = surface.native_tool()
                    if native:
                        record(native, 'SupportCallSurface.native_tool', selection_scope='schema_only_no_execution')
                    else:
                        unavailable.append({'scope': scope, 'status': 'not_applicable_no_support_options'})
            record(NodeExecutor._validate_current_atomic_tool(atomic), 'NodeExecutor._validate_current_atomic_tool')
            executor = NodeExecutor.__new__(NodeExecutor)
            for node in (False, True):
                record(executor._environment_tool(SimpleNamespace(action_catalog=[]), node_level=node,
                    atomic=atomic if node else None), 'NodeExecutor._environment_tool')
        for tool in (NodeExecutor._status_tool(allow_plan_conflict=True), NodeExecutor._status_tool(),
                     NodeExecutor._automation_tool(), automation_request_tool()):
            record(tool, 'NodeExecutor/RuntimeStep')
    from atomic_skillgraph.agents.structured_submission import (
        StructuredSubmissionClient, ATOMIC_EXTRACTION_SCHEMA, COMPOSITE_EXTRACTION_SCHEMA, TOOL_PROPOSAL_SCHEMA)
    from atomic_skillgraph.planner.requirement_agent import REQUIREMENT_SCHEMA
    from atomic_skillgraph.planner.workflow_agent import WORKFLOW_SCHEMA
    for name, schema in [('submit_planner_requirements', REQUIREMENT_SCHEMA), ('submit_planner_workflow', WORKFLOW_SCHEMA),
            ('submit_extractor_atomics', ATOMIC_EXTRACTION_SCHEMA), ('submit_extractor_composite', COMPOSITE_EXTRACTION_SCHEMA),
            ('create_tool', TOOL_PROPOSAL_SCHEMA)]:
        record(StructuredSubmissionClient.tool_spec(name, 'schema constructor audit', schema),
            'StructuredSubmissionClient.tool_spec', specialization='base_schema; per-request specialization audited at HTTP boundary')
    if not any(r['call_kind'] == 'implementation' for r in rows):
        raise ReleaseError('no deployable implementation constructor audited')
    return {'version': NATIVE_CALL_CONTRACT_VERSION, 'passed': True, 'constructors': rows,
            'unavailable': unavailable, 'execution_authority': False}


def audit_trace(trace):
    sessions = {s['session_id']: s.get('snapshot') or {} for s in trace.get('agent_sessions', [])}
    checks, diagnostics = [], []
    generated, rejected, repaired, repair_passed, repair_terminal, extras = (Counter() for _ in range(6))
    usage = trace.get('llm_usage', [])
    for request in trace.get('provider_requests', []):
        payload = request.get('final_payload_audit') or {}
        snapshot = sessions.get(request['session_id'], {})
        records = snapshot.get('native_call_contracts', [])
        matching = [r for r in records if r['turn_index'] == request.get('request_sequence')]
        actual = {t['function']['name']: canonical(t['function']['parameters']) for t in payload.get('tools', [])}
        sent = payload.get('native_call_contracts')
        expected = {c['tool_name']: c['input_schema_json'] for c in matching[0]['contracts']} if len(matching) == 1 else None
        passed = len(matching) == 1 and sent == matching[0]['contracts'] and actual == expected
        checks.append({'request_id': request['request_id'], 'session_id': request['session_id'],
            'repair': request.get('repair_in_progress'), 'passed': passed,
            'reason': 'matched' if passed else 'missing_or_inconsistent_request_contract'})
    for sid, snap in sessions.items():
        failures = {r['turn_index']: r for r in snap.get('protocol_failures', [])}
        for row in snap.get('native_call_contracts', []):
            turn = row['turn_index']
            kinds = {c['tool_name']: c['call_kind'] for c in row['contracts']}
            for call in row.get('generated_calls', []):
                generated[kinds.get(call['tool_name'], 'unknown_function')] += 1
            # A rejected call is not an executed native_tool_calls row.
            failure = failures.get(turn)
            if failure:
                d = failure.get('diagnostics', {})
                kind = d.get('call_kind', 'other_protocol')
                rejected[kind] += 1
                for issue in d.get('issues', []):
                    extras.update(issue.get('unexpected_properties', []))
                events = [e for e in usage if e['session_id'] == sid and e['turn_index'] == turn]
                links = [r for r in trace.get('provider_requests', []) if r['session_id'] == sid
                    and r.get('provider_request_id') == d.get('request_id') and d.get('request_id')]
                repair_rows = [r for r in snap.get('native_call_contracts', [])
                    if r['turn_index'] == turn + 1 and r.get('repair')]
                repair_events = [e for e in usage if e['session_id'] == sid
                    and e['turn_index'] == turn + 1] if repair_rows else []
                repair_links = [r for r in trace.get('provider_requests', [])
                    if r['session_id'] == sid and r.get('request_sequence') == turn + 1] if repair_rows else []
                diagnostics.append({**d, 'usage_event_ids': [e['event_id'] for e in events],
                    'physical_request_ids': [r['request_id'] for r in links],
                    'recorded_total_tokens': sum(e['prompt_tokens']+e['completion_tokens'] for e in events),
                    'repair_usage_event_ids': [e['event_id'] for e in repair_events],
                    'repair_physical_request_ids': [r['request_id'] for r in repair_links],
                    'repair_recorded_total_tokens': sum(e['prompt_tokens']+e['completion_tokens'] for e in repair_events),
                    'link_complete': bool(events and links) and (not repair_rows or bool(repair_events and repair_links))})
                if repair_rows:
                    repaired[kind] += 1
                    if turn + 1 in failures:
                        repair_terminal[kind] += 1
                    elif repair_rows[0].get('outcome') == 'accepted':
                        repair_passed[kind] += 1
    steps = {o['occurrence_id']: o['step_id'] for o in (trace.get('runtime_plan') or {}).get('occurrences', [])}
    return {'version': NATIVE_CALL_CONTRACT_VERSION,
        'passed': bool(checks) and all(c['passed'] for c in checks) and all(d['link_complete'] for d in diagnostics),
        'checks': checks, 'diagnostics': diagnostics, 'generated_by_kind': dict(generated),
        'rejected_by_kind': dict(rejected), 'repair_requests_by_kind': dict(repaired),
        'repair_passed_by_kind': dict(repair_passed), 'repair_terminal_by_kind': dict(repair_terminal),
        'unexpected_properties': dict(extras),
        'step_decisions': [{'step_id': steps.get(c['occurrence_id']), 'occurrence_id': c['occurrence_id'],
            'tool_name': c['tool_name'], 'call_id': c['call_id'], 'arguments': c['arguments']}
            for c in trace.get('native_tool_calls', [])],
        'accounting': 'diagnostic costs overlap all-attempt totals; never add again'}
