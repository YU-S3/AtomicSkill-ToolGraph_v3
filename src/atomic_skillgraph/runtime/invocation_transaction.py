"""One world/logic transaction boundary for a multi-action invocation."""
from __future__ import annotations

from ..core.errors import BudgetExhausted
from .checkpoint import restore, increment


class InvocationTransaction:
    def __init__(self, ctx, occurrence, *, origin, failure_code='invocation_rejected'):
        self.ctx, self.occurrence, self.origin = ctx, occurrence, origin
        self.failure_code = failure_code
        self.accepted = False
        self.checkpoint = None

    def __enter__(self):
        from .checkpoint import capture
        if getattr(self.ctx, 'runtime_config', {}).get('rollback_automatic_execution_failure'):
            self.checkpoint = capture(self.ctx, self.occurrence.occurrence_id)
        return self

    def __exit__(self, kind, exc, traceback):
        if self.checkpoint is None:
            return False
        # Infrastructure failures belong to attempt recovery, not fallback.
        if exc is not None and not isinstance(exc, BudgetExhausted):
            return False
        count = len(self.ctx.trace_builder.trace.environment_actions) - self.checkpoint.action_prefix_end
        if self.origin != 'runtime_trial':
            increment(self.ctx, 'llm_free_environment_action_count', count)
        if exc is not None or not self.accepted:
            before = len(self.ctx.trace_builder.trace.metadata.get('runtime_rollbacks', []))
            restore(self.ctx, self.checkpoint, exc.code if exc else self.failure_code)
            rows = self.ctx.trace_builder.trace.metadata.get('runtime_rollbacks', [])
            if len(rows) > before:
                rows[-1]['origin'] = self.origin
                values = self.ctx.trace_builder.trace.metadata.setdefault('r101_metrics', {})
                origins = values.setdefault('rollback_actions_by_origin', {})
                origins[self.origin] = origins.get(self.origin, 0) + count
                values['physical_restore_replay_actions'] = values.get('physical_restore_replay_actions', 0) + rows[-1]['restore_replay_action_count']
        return False


def execution_cache_key(compiled, arguments, consumer, ctx):
    from ..core.refs import content_hash
    from ..core.serialization import to_primitive
    from ..evolution.aligner import _tool_signature
    from ..evolution.contract_canonicalizer import canonical_atomic_contract
    from .negative_memory import state_signature, consumer_step_identity
    implementation = compiled.implementation
    # Ephemeral draft/Tool/Implementation IDs and source descriptions are not
    # executable identity. Preserve the actual program, parameter wiring,
    # validation contract and the consumer's authoritative obligation/state.
    return 'execution:' + content_hash({'tools': [_tool_signature(t) for t in compiled.tools],
        'contract': canonical_atomic_contract(compiled.atomic),
        'wiring': [{'role': b.role, 'order': b.order, 'parameters': to_primitive(b.parameter_mapping)}
                   for b in implementation.tool_bindings],
        'policy': to_primitive(implementation.execution_policy),
        'grounding': to_primitive(implementation.grounding_constraints),
        'arguments': arguments, 'consumer': str(consumer.node_ref),
        'step': consumer_step_identity(consumer), 'state': state_signature(ctx, consumer)})


def cache_lookup(ctx, key, occurrence, call_id=None):
    entry = ctx.rejected_runtime_candidates.get(key) if key else None
    if entry is not None:
        values = ctx.trace_builder.trace.metadata.setdefault('r101_metrics', {})
        values['exact_failure_cache_hits'] = values.get('exact_failure_cache_hits', 0) + 1
        ctx.trace_builder.trace.metadata.setdefault('runtime_execution_cache_hits', []).append({
            'occurrence_id': occurrence.occurrence_id, 'failure_code': entry['failure_code'], 'cached_rejection': True})
        if getattr(ctx, 'search_history', None) is not None:
            ctx.search_history.cache_hit(ctx, entry.get('search_observation_ids', []), call_id, occurrence.occurrence_id)
    return entry


def execute_invocation(runner, compiled, preflight, occurrence, ctx, *, agent_prepared,
                       execution_scope='registered', accept_result=None, origin=None, consumer=None,
                       authorizing_native_call_id=None):
    from ..core.results import ImplementationExecutionResult, NodeExecutionStatus
    consumer = consumer or occurrence
    cache_key = execution_cache_key(compiled, preflight.normalized_arguments, consumer, ctx)
    entry = cache_lookup(ctx, cache_key, occurrence, authorizing_native_call_id)
    if entry is not None:
        return ImplementationExecutionResult(str(compiled.implementation.ref), str(compiled.atomic.ref),
            True, False, False, False, failure_layer=entry['failure_layer'], failure_code=entry['failure_code'],
            node_status=NodeExecutionStatus.FAILED_NOT_STARTED, cached_rejection=True)
    origin = origin or ('agent_selected_registered' if agent_prepared else 'automatic_registered')
    from ..traces.compiler_observer import observation, _dataflow_inputs
    observed = observation(ctx)
    links = observed['program_invocation_links'] if observed is not None else []
    link_start = len(links)
    history_start = set(ctx.search_history.attempts) if getattr(ctx, 'search_history', None) is not None else set()
    with InvocationTransaction(ctx, occurrence, origin=origin) as transaction:
        ctx.binding_store.commit_grounded(occurrence.occurrence_id,
            {binding.role: binding for binding in preflight.binding_updates})
        consumed = _dataflow_inputs(ctx, occurrence, preflight.normalized_arguments,
            binding_updates=preflight.binding_updates,
            include_stored=execution_scope != 'runtime_trial') if observed is not None else []
        previous_marker = getattr(ctx, '_compiler_invocation_marker', None)
        ctx._compiler_invocation_marker = dict(origin=origin, native_call_id=authorizing_native_call_id,
            consumer_scope=getattr(consumer, 'consumer_scope', 'node'), consumed=consumed,
            parent_occurrence_id=consumer.occurrence_id)
        try:
            result = runner.run(compiled, preflight, occurrence, ctx,
                                agent_prepared=agent_prepared, execution_scope=execution_scope)
        finally:
            if previous_marker is None:
                del ctx._compiler_invocation_marker
            else:
                ctx._compiler_invocation_marker = previous_marker
        transaction.accepted = bool(result.completed and result.atomic_effect_passed and not result.failure_code)
        if transaction.accepted and accept_result is not None:
            report = accept_result(result)
            transaction.accepted = report.passed
            if not report.passed:
                result.atomic_effect_passed = False
                result.validated_outputs = {}
                result.validated_output_bindings = {}
                result.certified_input_bindings = {}
                result.failure_layer = 'runtime_binding'
                result.failure_code = report.failure_codes[0]
                for link in links[link_start:]:
                    if link['occurrence_id'] == occurrence.occurrence_id and link['implementation_ref'] == str(compiled.implementation.ref):
                        link.update(complete_success=False, outputs_valid=False)
        transaction.failure_code = result.failure_code or 'invocation_rejected'
    if cache_key and not result.atomic_effect_passed and result.failure_layer in {'tool', 'atomic', 'runtime_binding', 'implementation'}:
        ctx.rejected_runtime_candidates[cache_key] = {'failure_code': result.failure_code, 'failure_layer': result.failure_layer,
            'search_observation_ids': [i for i in ctx.search_history.attempts if i not in history_start]
                if getattr(ctx, 'search_history', None) is not None else []}
    return result
