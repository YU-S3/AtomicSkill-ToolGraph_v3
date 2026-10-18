"""Host-owned references to actual search hits, never path/offset guesses."""
from copy import deepcopy

from .contracts import digest
from .office_operation_mining import at


def observed_hits(event, corpus_identity):
    result = event.get('result', {})
    if (event.get('state') != 'finished' or event.get('backend_invoked') is not True
            or not event.get('event_id') or not event.get('result_id') or result.get('accepted') is not True
            or event.get('name') not in {'grep', 'grep_page'}): return []
    data = result.get('data'); path = ['data'] if isinstance(data, list) else ['data', 'hits']
    hits = data if isinstance(data, list) else data.get('hits', []) if isinstance(data, dict) else []
    refs = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict) or not isinstance(hit.get('path'), str) or type(hit.get('offset')) is not int: continue
        ref = {'ref_kind': 'observed_hit', 'producer_event_id': event['event_id'],
               'producer_result_id': event['result_id'], 'result_path': [*path, index],
               'corpus_identity': corpus_identity, 'result_sha256': digest(result),
               'guarantees': ['path_and_offset_from_same_hit']}
        ref['id'] = 'hit_' + digest(ref); refs.append(ref)
    return refs


def resolve_hit(ref, events, corpus_identity):
    if not isinstance(ref, dict): raise ValueError('hit_ref must be a Host reference object')
    event = next((e for e in events if e.get('event_id') == ref.get('producer_event_id')), None)
    if event is None or ref not in observed_hits(event, corpus_identity):
        raise ValueError('Unknown/tampered hit_ref or changed corpus/result identity')
    return at(event['result'], ref['result_path'])


def bind_read(ref, controls, events, corpus_identity):
    """No missing before/after defaults, and no independent offset input."""
    if not isinstance(controls, dict) or set(controls) != {'before', 'after'}:
        return {'status': 'needs_input', 'missing': ['before', 'after'], 'worker_invoked': False}
    if any(type(v) is not int or v < 0 for v in controls.values()):
        raise ValueError('Read controls must be nonnegative integers')
    hit = resolve_hit(ref, events, corpus_identity)
    return {'status': 'bound', 'arguments': {'path': hit['path'], 'offset': hit['offset'], **controls},
            'input_ref': deepcopy(ref), 'worker_invoked': False}


def recover_handoffs(events, corpus_identity, *, binding_claimed=False):
    hits, result = [], []
    for event in events:
        if event.get('name') in {'read', 'read_context'}:
            args = event.get('arguments', {}); raw = event.get('result', {})
            matches = [r for r,h in hits if h['path'] == args.get('path') and h['offset'] == args.get('offset')]
            text = raw.get('data', {}).get('text') if isinstance(raw.get('data'), dict) else None
            occurred = event.get('state') == 'finished' and event.get('backend_invoked') is True and raw.get('accepted') is True
            valid = occurred and bool(text) and len(matches) == 1
            refs = event.get('input_refs', {})
            explicit = valid and all(refs.get(k) == {'result_id': matches[0]['producer_result_id'],
                'path': [*matches[0]['result_path'], k]} for k in ('path', 'offset'))
            same_file = any(h['path'] == args.get('path') for r,h in hits)
            result.append({'consumer_event_id': event.get('event_id'),
                'state': 'observed_handoff' if valid else 'unproven_handoff',
                'dataflow': 'explicit_result_binding' if explicit else 'unique_historical_observation' if valid else 'structural_match_unverified',
                'reason': None if valid else ('cross_file_handoff_invalid' if binding_claimed or refs else 'cross_file_handoff_unproven')
                    if hits and not same_file else 'missing_ambiguous_or_unread_hit',
                'hit_ref': matches[0] if valid else None, 'formal_qualification': False})
        hits.extend((ref, at(event['result'], ref['result_path'])) for ref in observed_hits(event, corpus_identity))
    return result
