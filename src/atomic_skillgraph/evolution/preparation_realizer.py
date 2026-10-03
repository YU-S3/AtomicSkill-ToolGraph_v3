"""Bounded preparation envelopes for existing E1/Builder, not answer routes."""
from ..core.refs import content_hash

VERSION = 'skillcompiler.preparation-realizer.v2'


def uncovered_intervals(normalized):
    actions = sorted(normalized.get('actions', ()), key=lambda event: event['event_index'])
    intervals, current = [], []
    covered = set(normalized.get('covered_program_event_indices', []))
    for event in actions:
        if event.get('accepted') and not event.get('canonical_discarded'):
            if event['event_index'] in covered or event.get('done') or event.get('won'):
                if len(current) >= 2:
                    intervals.append(current)
                current = []
            else:
                current.append(event)
    if len(current) >= 2:
        intervals.append(current)
    return sorted(intervals, key=lambda group: (-len(group), group[0]['event_index']))


def covered_program_events(trace):
    """Deduplicated canonical actions inside actually successful Tool spans."""
    from ..traces.canonical import canonical_action_indices
    from ..traces.compiler_observer import field
    canonical = set(canonical_action_indices(trace))
    spans = {field(s, 'span_id'): s for s in field(trace, 'runtime_spans', [])}
    result = set()
    for execution in field(trace, 'tool_executions', []):
        r = field(execution, 'result')
        if r.get('started') and r.get('completed') and r.get('atomic_effect_passed') and not r.get('failure_code'):
            span = spans.get(field(execution, 'span_id'))
            if span:
                result.update(range(field(span, 'action_start'), field(span, 'action_end')))
    return sorted(result & canonical)


def learning_view(view):
    """Stable covered spans need source attribution, not full-contract regeneration."""
    import copy
    result = copy.deepcopy(view)
    covered = set(view.get('covered_program_event_indices', []))
    for i, action in enumerate(result.get('actions', [])):
        if action['event_index'] in covered:
            result['actions'][i] = {k: action[k] for k in ('event_id', 'action_id', 'event_index', 'action_type', 'arguments',
                'accepted', 'before_revision', 'after_revision', 'extractor_event_start', 'extractor_event_end_exclusive') if k in action}
            result['actions'][i]['learning_responsibility'] = 'existing successful Program; use source attribution/reuse only'
    return result


def offer(system, trace, normalized, groups):
    intervals = uncovered_intervals(normalized)
    audit = trace.metadata.setdefault('preparation_realizer', {'version': VERSION,
        'candidate_limit': 1, 'independent_source_limit': 2,
        'initial_generation_limit': 1, 'content_repair_limit': 1, 'online_credit': False})
    if not groups or not intervals:
        audit.update(status='no_independent_uncovered_pair')
        return []
    ranked = [(g, uncovered_intervals(g['history'])) for g in groups]
    ranked = [item for item in ranked if item[1]]
    ranked.sort(key=lambda item: (-len(item[1][0]), item[0]['group_id']))
    group, history_intervals = (dict(ranked[0][0]), ranked[0][1]) if ranked else ({}, [])
    if not history_intervals:
        audit.update(status='history_has_no_preparation_interval')
        return []
    spans = {}
    for slot, actions in [('current', intervals[0]), ('history', history_intervals[0])]:
        # TraceNormalizer's production actions carry action_id. Legacy E1
        # views may carry event_id; use the same reference convention as the
        # Atomicizer, without rewriting either immutable source namespace.
        event_ids = [str(a.get('event_id', a.get('action_id', ''))) for a in actions]
        if not all(event_ids) or len(set(event_ids)) != len(event_ids):
            raise ValueError('preparation source interval requires distinct nonempty action references')
        spans[slot] = {'event_start': actions[0]['event_index'], 'event_end': actions[-1]['event_index'] + 1,
            'support_event_ids': event_ids, 'source_id': normalized['trace_id']
            if slot == 'current' else group['history']['trace_id'],
            'status': 'observed_candidate_interval_not_validated_contract'}
    from .realization_queue import RealizationQueue
    key, claimed, _ = RealizationQueue(system.database).claim({'protocol': VERSION,
        'source_pair': sorted([content_hash(normalized), group['history_reference']['canonical_snapshot_hash']]),
        'spans': spans})
    if not claimed:
        audit.update(status='already_scheduled', queue_key=key)
        return []
    group['preparation_spans'] = spans
    audit.update(status='offered', queue_key=key, spans=spans)
    return [group]


def finish(system, trace):
    audit = trace.metadata.get('preparation_realizer', {})
    if audit.get('status') != 'offered':
        return
    from .realization_queue import RealizationQueue
    outcome = trace.metadata.get('generalization', {})
    RealizationQueue(system.database).finish(audit['queue_key'],
        status=outcome.get('status', 'no_candidate_submitted'),
        failure_codes=[outcome['reason']] if outcome.get('reason') else [], result_refs=outcome.get('refs', []))
    audit['status'] = outcome.get('status', 'no_candidate_submitted')
