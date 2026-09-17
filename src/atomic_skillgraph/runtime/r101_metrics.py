"""R10.1 boundary diagnostics, separate from official benchmark success."""
COUNTERS = ('support_mapping_authority_rejects', 'support_parent_identity_rejects',
    'support_input_transfer_rejects',
    'exact_failure_cache_hits', 'replay_trace_only_candidate_events',
    'replay_registered_events', 'physical_restore_replay_actions', 'orphan_lifecycle_rows')


def finalize(trace):
    values = trace.metadata.setdefault('r101_metrics', {})
    for key in COUNTERS:
        values.setdefault(key, 0)
    values.setdefault('rollback_actions_by_origin', {})


def aggregate(rows):
    values = [row.get('r101_metrics', {}) for row in rows]
    result = {key: sum(int(value.get(key, 0) or 0) for value in values) for key in COUNTERS}
    origins = {}
    for value in values:
        for origin, count in value.get('rollback_actions_by_origin', {}).items():
            origins[origin] = origins.get(origin, 0) + count
    result['rollback_actions_by_origin'] = origins
    return result
