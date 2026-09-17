"""Read-only status of declared values. No candidate search or assignment."""
from ..core.bindings import BindingStatus, resolution_satisfies
from ..core.serialization import to_primitive


class GivenInputReader:
    def __init__(self, invocation_compiler, validation):
        self.invocation_compiler = invocation_compiler
        self.validation = validation

    def refresh(self, occurrence, atomic, invocations, ctx):
        current = ctx.binding_store.snapshot_for_node(occurrence)
        known = {r: b.value for r, b in current.items() if b.status is BindingStatus.GROUNDED}
        confirmed = {p.name: known[p.name] for p in atomic.inputs
                     if p.name in known and resolution_satisfies(current[p.name].resolution, p.required_resolution)}
        missing = [p.name for p in atomic.inputs if p.required and p.name not in confirmed]
        statuses = []
        for predicate in atomic.preconditions:
            report = ctx.harness.validator_channel().validate_atomic_effect({
                'effects': [predicate], 'bindings': known, 'output_candidates': {}})
            statuses.append({'predicate': predicate.predicate,
                             'status': 'satisfied' if report.passed else 'missing'})
        entries = [self.invocation_compiler.autonomous_preflight(
            item, occurrence, ctx.binding_store, ctx.evidence_store, ctx.world_revision,
            task_contract=ctx.task_contract) for item in invocations]
        state = {
            'revision': ctx.world_revision, 'occurrence_id': occurrence.occurrence_id,
            'semantic_anchors': {p.name: b.value for p in atomic.inputs
                if (b := ctx.binding_store.semantic_anchor_for(occurrence, p.name)) is not None},
            'confirmed_bindings': confirmed, 'candidate_bindings': {},
            'missing_bindings': missing,
            'invalidated_bindings': {r: to_primitive(b) for r, b in current.items() if b.status is BindingStatus.INVALIDATED},
            'precondition_status': statuses,
            'effect_witness_status': {'passed': False, 'completion_requires_explicit_submission': True},
            'learned_invocation_ready': any(p.passed for p in entries),
            'blocking_reasons': [*(f'missing_{r}' for r in missing),
                *(p.failure_code for p in entries if not p.passed)],
        }
        record = getattr(ctx, 'record_grounding_state', None)
        if callable(record):
            record(occurrence.occurrence_id, state)
        return state
