"""Read-only request/call/usage linkage. Overlapping tags never add total cost."""
from collections import Counter


def trace_metrics(trace):
    metadata = trace.get('metadata', {})
    calls = [c for c in trace.get('native_tool_calls', []) if c['tool_name'] == 'invoke_support_atomic']
    resolutions = {(r['session_id'], r['call_id']): r for r in metadata.get('support_call_resolutions', [])}
    events = {e['event_id']: e for e in trace.get('llm_usage', [])}
    requests = {r['request_id']: r for r in trace.get('provider_requests', [])}
    links, rejected_events, cached_events, rejected_requests, cached_requests = [], set(), set(), set(), set()
    failures = Counter()
    for call in calls:
        result = call.get('preflight_result') or {}
        resolution = resolutions.get((call['session_id'], call['call_id']), {})
        # The accepted turn's returned call IDs, not merely a session, identify
        # its billable response. Repairs/retries retain their own event IDs.
        turns = [t for t in trace.get('agent_turns', []) if t['session_id'] == call['session_id']
                 and call['call_id'] in t['tool_call_ids']]
        event_rows = [e for e in events.values() if any(e['session_id'] == t['session_id']
            and t.get('provider_metadata', {}).get('request_id') and e.get('provider_metadata', {}).get('request_id')
            == t.get('provider_metadata', {}).get('request_id') for t in turns)]
        request_rows = [r for r in requests.values() if any(r['session_id'] == e['session_id']
            and r.get('provider_request_id') and r['provider_request_id'] == e.get('provider_metadata', {}).get('request_id')
            for e in event_rows)]
        rejected = result.get('accepted') is False
        cached = bool(result.get('deterministic_rejection_cache_hit'))
        reason = result.get('reason_code') or result.get('preflight_failure_code') or result.get('error')
        if rejected:
            failures[reason or 'unknown'] += 1
            rejected_events.update(e['event_id'] for e in event_rows)
            rejected_requests.update(r['request_id'] for r in request_rows)
        if cached:
            cached_events.update(e['event_id'] for e in event_rows)
            cached_requests.update(r['request_id'] for r in request_rows)
        links.append({'call_id': call['call_id'], 'session_id': call['session_id'],
            'support_call_id': call['arguments'].get('support_call_id'), 'resolution': resolution,
            'request_ids': [r['request_id'] for r in request_rows],
            'usage_event_ids': [e['event_id'] for e in event_rows],
            'link_complete': len(turns) == len(event_rows) == len(request_rows) == 1,
            'rejected': rejected, 'cache_hit': cached, 'reason_code': reason,
            'attempt_refs': result.get('attempt_refs', {})})

    def costs(ids):
        rows = [events[k] for k in ids]
        return {'usage_event_ids': sorted(ids), 'prompt_tokens': sum(e['prompt_tokens'] for e in rows),
            'completion_tokens': sum(e['completion_tokens'] for e in rows),
            'reasoning_tokens': sum(e['reasoning_tokens'] for e in rows) if all(e.get('reasoning_tokens') is not None for e in rows) else None,
            'recorded_total_tokens': sum(e['prompt_tokens'] + e['completion_tokens'] for e in rows)}

    frames = metadata.get('public_discovery_frames', [])
    frame_counts = Counter()
    for frame in frames:
        frame_counts.update({
            'public_observation_relations': sum(r['source_kind'] == 'public_observation_relation' for r in frame['records']),
            'conflicts': len(frame['conflicts'])})
        frame_counts.update(s['status'] for s in frame['inspected_scopes'])
    return {'version': 'r103.release4-metrics.v1',
        'funnel': metadata.get('runtime_support_funnel', {}), 'support_wire_selection_count': len(calls),
        'rejections_by_reason': dict(failures), 'call_links': links,
        'unlinked_call_ids': [r['call_id'] for r in links if not r['link_complete']],
        'support_rejected_decision_cost': costs(rejected_events),
        'support_cached_rejection_cost': costs(cached_events),
        'support_rejected_decision_requests': sorted(rejected_requests),
        'support_cached_rejection_requests': sorted(cached_requests),
        'all_recorded_usage': costs(set(events)),
        'unknown_usage_request_ids': [r['request_id'] for r in requests.values() if r.get('usage_status') != 'reported'],
        'public_discovery': dict(frame_counts),
        'accounting_note': 'Reasoning is included in completion. Rejected/cache/preparation are overlapping labels; never sum them into all-attempt totals. Missing links remain explicit.'}
