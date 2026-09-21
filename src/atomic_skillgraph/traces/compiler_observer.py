"""Trace-only associations. No provider, validator, reset or registry writes.

Absent evidence stays unknown. Raw attempts survive rollback; canonical action
membership is resolved only at immutable Trace finalization.
"""
from contextlib import contextmanager

from ..core.serialization import to_primitive, json_values_equal
from ..core.refs import content_hash
from .canonical import canonical_action_indices

VERSION = 'r103.compiler-observation.v1'
ROUTES = {'stored_composite': 'p0_module', 'atomic_composition': 'p2_composition',
          'cold_start': 'cold_start', 'full_dynamic': 'full_dynamic'}
ORIGINS = {'automatic_registered': 'graph_entry_auto', 'agent_selected_registered': 'agent_selected_registered',
           'task_agent_selected_registered': 'agent_selected_registered', 'runtime_trial': 'runtime_trial',
           'offline_replay': 'offline_replay'}


def field(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def initialize(builder, *, persistent_refs, request_snapshot, program_identities=None, program_sources=None):
    # These read-only task-start snapshots are NOT part of model context.
    builder.compiler_persistent_refs = frozenset(persistent_refs)
    builder.compiler_request_snapshot = request_snapshot
    builder.compiler_program_identities = program_identities or {}
    builder.compiler_program_sources = program_sources or {}
    builder.trace.metadata['compiler_observability'] = dict(version=VERSION, capture_status='partial',
        missing_reasons=[], initial_public_state_hash=None, initial_state_digest=None,
        compilation={}, node_entry_windows=[], program_invocation_links=[], dataflow_consumptions=[])


def admission_sources(tool):
    """Original immutable admission provenance, not later support credit.

    Legacy bool replays/fixtures are not proof. Require the already produced
    typed replay result to name the exact source case and resolved task.
    """
    from ..evolution.replay import replay_case_id
    results = tool.metadata.get('admission', {}).get('replay_results', [])
    sources = []
    if not tool.tests:
        return None
    for case in tool.tests:
        source = case.get('source_task', {})
        matches = [r for r in results if r.get('case_id') == replay_case_id(case)
                   and r.get('source_trace_id') == case.get('trace_id')
                   and r.get('source_task_id') == r.get('resolved_task_id') == source.get('task_id')
                   and r.get('passed') is True and r.get('started') is True
                   and r.get('completed') is True and r.get('output_validation_passed') is True]
        if not matches or not source.get('task_signature') or not source.get('benchmark'):
            return None
        sources.append(dict(case_id=replay_case_id(case), source_trace_id=case['trace_id'],
            task_signature=source['task_signature'], benchmark=source['benchmark']))
    return sources


def observation(ctx):
    trace = field(field(ctx, 'trace_builder'), 'trace')
    return field(trace, 'metadata', {}).get('compiler_observability')


def boundary(ctx):
    reader = getattr(ctx.trace_builder, 'compiler_request_snapshot', None)
    return reader() if reader is not None else None


def initial_state(ctx, reset):
    obs = observation(ctx)
    if obs is None:
        return
    obs['initial_public_state_hash'] = content_hash(to_primitive(dict(task_id=ctx.task_id,
        observation=reset.observation, catalog=reset.catalog, revision=reset.new_revision)))
    # A validator snapshot is not necessarily a complete hidden-world digest.
    obs['missing_reasons'].append('complete_initial_state_digest_unavailable')
    plan = to_primitive(ctx.plan)
    raw = plan.get('source')
    audit = ctx.trace_builder.trace.planner_audit
    obs['compilation'] = dict(selected_route=ROUTES.get(raw, 'unknown'), source_route_raw=raw,
        accepted_graph=raw in {'stored_composite', 'atomic_composition'}, graph_payload_hash=content_hash(plan),
        source_composite_ref=plan.get('source_composite_ref'), planner_request_refs=[],
        validation_refs=['planner_audit.' + k for k in audit if 'validation' in k] or None,
        repair_refs=['planner_audit.' + k for k in audit if 'repair' in k or k.endswith(('p1r', 'p2r'))] or None,
        structural_reuse_kind='existing_module' if raw == 'stored_composite' else 'unknown')


@contextmanager
def node_window(occurrence, ctx):
    obs = observation(ctx)
    if obs is None:
        yield {}
        return
    trace = ctx.trace_builder.trace
    row = dict(entry_id=f'entry_{len(obs["node_entry_windows"])}', occurrence_id=occurrence.occurrence_id,
        step_id=occurrence.step_id, atomic_ref=str(occurrence.node_ref), world_revision=ctx.world_revision,
        bootstrap=False, already_satisfied=False, terminal_skipped=False, compiled_invocation_refs=[],
        selected_invocation_attempt_id=None, entry_reason_raw=None,
        provider_request_start_index=None, provider_request_end_index=None,
        environment_action_start_index=len(trace.environment_actions), environment_action_end_index=None,
        outcome='agent_owned', capture_complete=False, provider_boundary_before=boundary(ctx))
    obs['node_entry_windows'].append(row)
    checks_start = len(trace.metadata.get('node_entry_checks', []))
    links_start = len(obs['program_invocation_links'])
    try:
        yield row
        row['capture_complete'] = True
    finally:
        row['provider_boundary_after'] = boundary(ctx)
        row['environment_action_end_index'] = len(trace.environment_actions)
        checks = trace.metadata.get('node_entry_checks', [])[checks_start:]
        if checks:
            row['entry_reason_raw'] = checks[0]['reason']
        automatic = [p for p in obs['program_invocation_links'][links_start:]
                     if p['origin'] == 'graph_entry_auto' and p['occurrence_id'] == occurrence.occurrence_id]
        if automatic:
            row['selected_invocation_attempt_id'] = automatic[0]['implementation_attempt_id']
            if all(p['complete_success'] for p in automatic):
                row['outcome'] = 'automatic_complete_success'
            elif any(p['terminal_interrupted'] and p['atomic_effect_passed'] for p in automatic):
                row['outcome'] = 'automatic_terminal_effect'
            else:
                row['outcome'] = 'automatic_failed'
        elif checks and not row['bootstrap']:
            reason = checks[0]['reason']
            row['outcome'] = 'ambiguous_route' if reason == 'ambiguous_implementations' else 'no_ready_route'


def _dataflow_inputs(ctx, occurrence, arguments, *, binding_updates=(), include_stored=True):
    """Record only actual compiled input reads, not output publication.

    A replaced or re-certified binding is conservatively not credited to the
    original DATA_FLOW edge. No value-only lineage inference.
    """
    bindings = ctx.binding_store.snapshot_for_node(occurrence) if include_stored else {}
    # Direct runner/trial callers do not commit to the store. Match the same
    # effective overlay used by ImplementationRunner, without changing it.
    bindings.update({b.role: b for b in binding_updates})
    trace = ctx.trace_builder.trace
    result = []
    for role, binding in bindings.items():
        b = to_primitive(binding)
        if b['source'] != 'data_flow' or role not in arguments or not json_values_equal(to_primitive(arguments[role]), b['value']):
            continue
        for edge in ctx.plan.data_edges:
            if edge.target_step != occurrence.step_id or edge.target_role != role or edge.edge_id not in b['evidence_refs']:
                continue
            producer = ctx.plan.occurrence(edge.source_step)
            changes = [(i, c) for i, c in enumerate(trace.binding_changes)
                       if field(c, 'occurrence_id') == occurrence.occurrence_id and field(c, 'role') == role
                       and field(c, 'current') == b]
            original = ctx.binding_store._outputs.get((producer.occurrence_id, edge.source_role))
            if not changes or original is None:
                continue
            result.append(dict(producer_occurrence_id=producer.occurrence_id, producer_invocation_id=None,
                producer_output_role=edge.source_role, edge_id=edge.edge_id,
                consumer_occurrence_id=occurrence.occurrence_id, consumer_input_role=role,
                binding_ref=f'binding_changes:{changes[-1][0]}', evidence_refs=b['evidence_refs'],
                producer_revision=original.world_revision, consumed_revision=ctx.world_revision,
                consumer_preflight_ref=None, consumer_invocation_id=None, consumed=True, consumer_outcome=None))
    return result


@contextmanager
def program_window(compiled, preflight, occurrence, ctx, execution_scope):
    obs = observation(ctx)
    if obs is None:
        yield
        return
    trace = ctx.trace_builder.trace
    impl_start, tool_start = len(trace.implementation_invocations), len(trace.tool_executions)
    before = boundary(ctx)
    # Read before runner commits/changes any further binding state.
    marker = getattr(ctx, '_compiler_invocation_marker', {})
    consumed = (marker['consumed'] if 'consumed' in marker else
                _dataflow_inputs(ctx, occurrence, preflight.normalized_arguments,
                                 binding_updates=preflight.binding_updates,
                                 include_stored=execution_scope != 'runtime_trial')) if preflight.passed else []
    origin = 'runtime_trial' if execution_scope == 'runtime_trial' else ORIGINS.get(marker.get('origin'), 'unknown')
    try:
        yield
    finally:
        after = boundary(ctx)
        spans = {field(s, 'span_id'): to_primitive(s) for s in trace.runtime_spans}
        tools = {str(t.ref): t for t in compiled.tools}
        persistent = getattr(ctx.trace_builder, 'compiler_persistent_refs', None)
        for raw in trace.implementation_invocations[impl_start:]:
            inv = to_primitive(raw)
            if inv['occurrence_id'] != occurrence.occurrence_id or inv['implementation_ref'] != str(compiled.implementation.ref):
                continue
            result = inv['result']
            related = [to_primitive(t) for t in trace.tool_executions[tool_start:]
                       if spans.get(field(t, 'span_id'), {}).get('parent_span_id') == inv['span_id']]
            # A started Impl with no Tool result still remains visible.
            for tool in related or [None]:
                tresult = tool['result'] if tool else {}
                span = spans.get(tool['span_id'] if tool else inv['span_id'], {})
                indices = list(range(span.get('action_start', 0), span.get('action_end', 0)))
                ref = tool['tool_ref'] if tool else None
                program = tools.get(ref)
                sources = getattr(ctx.trace_builder, 'compiler_program_sources', {}).get(ref)
                source_member = (any(s['benchmark'] == trace.task.benchmark and s['task_signature'] == trace.task.task_signature
                                     for s in sources) if sources else None)
                outputs_valid = bool(result.get('atomic_effect_passed')) and all(
                    not p.required or p.name in result.get('validated_outputs', {}) for p in compiled.atomic.outputs)
                complete = bool(result.get('started') and result.get('completed') and result.get('atomic_effect_passed')
                    and outputs_valid and not result.get('failure_code') and not result.get('terminal_interrupted')
                    and related and all(t['result'].get('completed') and not t['result'].get('terminal_interrupted')
                                       and not t['result'].get('failure_code') for t in related))
                obs['program_invocation_links'].append(dict(implementation_attempt_id=inv['attempt_id'],
                    tool_execution_id=tool['attempt_id'] if tool else None, occurrence_id=occurrence.occurrence_id,
                    span_id=tool['span_id'] if tool else inv['span_id'], consumer_scope=marker.get('consumer_scope', 'unknown'),
                    origin=origin, asset_origin=('unknown' if persistent is None or ref is None else
                        'persistent' if ref in persistent else 'task_local'), atomic_ref=str(compiled.atomic.ref),
                    implementation_ref=inv['implementation_ref'], tool_ref=ref,
                    program_raw_hash=content_hash(to_primitive(program.artifact)) if program else None,
                    program_equivalence_id=getattr(ctx.trace_builder, 'compiler_program_identities', {}).get(ref),
                    admission_source_refs=sources, source_task_member=source_member,
                    evidence_origin=('fixture' if compiled.atomic.metadata.get('acceptance_fixture') else
                                     'learned' if sources else 'unknown'),
                    authorizing_native_call_id=marker.get('native_call_id'), policy_action_indices=indices,
                    canonical_action_indices=None, provider_boundary_before=before, provider_boundary_after=after,
                    provider_request_refs=None, started=tresult.get('started', result.get('started')),
                    completed=tresult.get('completed', False), atomic_effect_passed=result.get('atomic_effect_passed'),
                    outputs_valid=outputs_valid, terminal_interrupted=bool(result.get('terminal_interrupted') or tresult.get('terminal_interrupted')),
                    rollback=None, intrinsic_failure=tresult.get('intrinsic_failure'), complete_success=complete))
            if result.get('started'):
                for c in consumed:
                    c.update(consumer_invocation_id=inv['attempt_id'], consumer_preflight_ref=inv['attempt_id'] + '.preflight',
                             consumer_outcome=result.get('node_status'))
                    producers = [i for i in trace.implementation_invocations[:impl_start]
                                 if field(i, 'occurrence_id') == c['producer_occurrence_id']
                                 and field(i, 'result', {}).get('atomic_effect_passed')]
                    if len(producers) == 1:
                        c['producer_invocation_id'] = field(producers[0], 'attempt_id')
                    obs['dataflow_consumptions'].append(c)


def finalize(trace):
    obs = trace.metadata.get('compiler_observability')
    if obs is None:
        return
    requests = [field(r, 'request_id') for r in trace.provider_requests]
    canonical = set(canonical_action_indices(trace))
    missing = set(obs['missing_reasons'])
    for row in obs['node_entry_windows'] + obs['program_invocation_links']:
        before, after = row.get('provider_boundary_before'), row.get('provider_boundary_after')
        if before is None or after is None:
            row['provider_request_refs'] = None
            missing.add('provider_boundary_unavailable')
        else:
            refs = [r for r in after if r not in set(before)]
            row['provider_request_refs'] = refs
            if any(r not in requests for r in refs):
                missing.add('provider_request_not_attached')
            positions = [requests.index(r) for r in refs if r in requests]
            if 'entry_id' in row:
                # Empty windows can occur between noncontiguous stage lists;
                # zero is proven by the snapshots, not guessed from len(trace).
                if positions and positions == list(range(min(positions), max(positions)+1)):
                    row['provider_request_start_index'], row['provider_request_end_index'] = min(positions), max(positions)+1
                elif not refs:
                    index = max((requests.index(r)+1 for r in before if r in requests), default=0)
                    row['provider_request_start_index'] = row['provider_request_end_index'] = index
        if 'policy_action_indices' in row:
            row['canonical_action_indices'] = sorted(set(row['policy_action_indices']) & canonical)
            row['rollback'] = bool(set(row['policy_action_indices']) - canonical)
            if row['rollback']:
                row['complete_success'] = False
    compilation = obs['compilation']
    if not compilation:
        missing.add('runtime_context_not_created')
    else:
        compilation['planner_request_refs'] = [field(r, 'request_id') for r in trace.provider_requests
                                               if field(r, 'stage', '').startswith('planner')]
    if any(not row['capture_complete'] for row in obs['node_entry_windows']):
        missing.add('node_entry_interrupted')
    obs['missing_reasons'] = sorted(missing)
    obs['capture_status'] = 'partial' if missing else 'complete'
