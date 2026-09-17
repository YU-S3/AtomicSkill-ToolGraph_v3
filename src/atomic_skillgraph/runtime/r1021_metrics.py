"""Observations of retired limits. Never used for execution decisions."""
from collections import defaultdict

from ..agents.usage import UsageBucket


ONLINE_BUCKETS = frozenset({
    UsageBucket.RUNTIME_PREPARATION.value, UsageBucket.RUNTIME_SEEDED.value,
    UsageBucket.RUNTIME_DYNAMIC.value, UsageBucket.RUNTIME_PROVISIONAL_SEEDED.value,
    UsageBucket.RUNTIME_DYNAMIC_COLD_START_CONTINUATION.value, UsageBucket.TOOL_BUILDER_RUNTIME.value,
})


def retired_limit_observations(trace):
    sessions = {s.session_id: s.occurrence_id for s in trace.agent_sessions}
    helper_owners = {row['occurrence_id']: row['consumer_occurrence_id']
        for row in trace.metadata.get('runtime_support_node_records', [])
        if row.get('consumer_occurrence_id')}
    def root_owner(owner):
        seen = set()
        while owner in helper_owners and owner not in seen:
            seen.add(owner)
            owner = helper_owners[owner]
        return owner
    node_tokens, node_actions = defaultdict(int), defaultdict(int)
    total, seen, crossings, unkeyed = 0, set(), {}, 0
    for event in trace.llm_usage:
        event_id = event.get('event_id')
        if not event_id:
            # Historical records cannot establish event-level reconciliation.
            unkeyed += 1
            continue
        if event_id in seen:
            raise ValueError('duplicate usage event in retired-limit audit')
        seen.add(event_id)
        if event.get('bucket') not in ONLINE_BUCKETS:
            continue
        value = int(event.get('usage', event).get('total_tokens') or 0)
        total += value
        owner = root_owner(sessions.get(event.get('session_id'), ''))
        if total > 300000:
            crossings.setdefault('task_300k', {'event_id': event_id, 'used_at_crossing': total})
        if owner and owner != '__task__':
            node_tokens[owner] += value
            if node_tokens[owner] > 100000:
                crossings.setdefault('node_100k:' + owner,
                    {'event_id': event_id, 'used_at_crossing': node_tokens[owner]})
    spans = {s.span_id: s for s in trace.runtime_spans}
    # Retired action gates charged physical attempts, including later rollbacks.
    for index, action in enumerate(trace.environment_actions):
        span = spans.get(action.span_id)
        if not span or not span.occurrence_id or span.occurrence_id == '__task__':
            continue
        owner = root_owner(span.occurrence_id)
        node_actions[owner] += 1
        if node_actions[owner] > 35:
            crossings.setdefault('node_35_actions:' + owner,
                {'action_index': index, 'action_id': action.action_id, 'used_at_crossing': node_actions[owner]})
    for boundary, record in crossings.items():
        if boundary == 'task_300k':
            record['subsequent_tokens'] = total - record['used_at_crossing']
        elif boundary.startswith('node_100k:'):
            record['subsequent_tokens'] = node_tokens[boundary.split(':', 1)[1]] - record['used_at_crossing']
        else:
            record['subsequent_actions'] = node_actions[boundary.split(':', 1)[1]] - record['used_at_crossing']
    return {'observational_only': True, 'online_total_tokens': total,
            'event_accounting_complete': unkeyed == 0, 'unkeyed_usage_records': unkeyed,
            'node_tokens': dict(node_tokens), 'node_actions': dict(node_actions),
            'crossings': crossings, 'official_won': bool(trace.benchmark_success)}
