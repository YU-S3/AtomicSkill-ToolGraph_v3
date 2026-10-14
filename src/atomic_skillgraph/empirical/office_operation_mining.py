"""Read-only Train operation opportunities. These are not publication certificates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .contracts import PublicTask, digest
from .public_results import public_result
from .retrieval_chain import recover

VERSION = 'office-trace-v4-pilot.v1'
QUALIFICATION = 'candidate_source_not_proven_reusable'
TOOLS = {'glob', 'grep', 'read', 'execute_python', 'grep_page', 'read_context'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def at(value, path):
    for key in path:
        if isinstance(value, dict) and isinstance(key, str) and key in value: value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value): value = value[key]
        else: raise ValueError('source_field_missing: use an advertised exact field path')
    return deepcopy(value)


def fields(value, path=()):
    """Include containers as bindable fields; no invented defaults or inferred numbers."""
    yield {'path': list(path), 'sha256': digest(value), 'type': type(value).__name__}
    if isinstance(value, dict):
        for key, child in value.items(): yield from fields(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value): yield from fields(child, (*path, index))


def mine(trace, trace_sha, environment):
    task = PublicTask(**trace['task'])
    if task.split != 'train': raise ValueError('Only original public Train sources are admissible')
    spans, opportunities = [], []
    for event in trace.get('tools', []):
        if event.get('name') not in TOOLS or not event.get('event_id'): continue
        raw = event.get('result', {})
        # Explicit envelope allowlist: never pass score, prediction, evaluator or execution state.
        public = public_result(raw, event.get('result_protocol'))
        result = {k: public[k] for k in ('accepted', 'data', 'error', 'error_code', 'status') if k in public}
        occurred = event.get('state') == 'finished' and event.get('backend_invoked') is True
        accepted = occurred and result.get('accepted') is True
        origin = ('observed_local_step' if accepted else 'observed_failure' if occurred or result.get('error')
                  else 'answer_or_task_hypothesis')
        span = {'id': 'span_' + digest([trace_sha, event['event_id']]), 'source_task': {
            'task_id': task.task_id, 'physical_key': task.physical_key, 'split': 'train',
            'goal': task.goal, 'inputs': deepcopy(task.inputs)}, 'source_trace_sha256': trace_sha,
            'event_id': event['event_id'], 'parent_event_id': event.get('parent_event_id'),
            'name': event['name'], 'arguments': deepcopy(event.get('arguments', {})),
            'result_id': event.get('result_id'), 'result': result, 'raw_result_sha256': digest(raw),
            'state': event.get('state'), 'backend_invoked': event.get('backend_invoked', False),
            'outputs_consumed': event.get('outputs_consumed'), 'origin': origin}
        span['available_fields'] = [{'section': section, **f} for section in ('arguments', 'result')
                                    for f in fields(span[section])]
        missing = ['local correctness and downstream consumption are unverified']
        data = result.get('data')
        if not accepted: missing.append('execution/preconditions failed or were not completed')
        if data in (None, [], {}, ''): missing.append('no acquired public data')
        if event['name'] == 'grep' and isinstance(data, list) and len(data) >= 40:
            missing.append('legacy search may be truncated; absence is not not_found')
        if isinstance(data, dict) and data.get('complete') is False: missing.append('pagination unfinished')
        if event['name'] in {'read', 'read_context'}: missing.append('row/header/unit alignment requires a new probe')
        previous = next((s for s in reversed(spans) if s['name'] in {'grep', 'grep_page'}
                         and s['origin'] == 'observed_local_step'), None) if event['name'] in {'read', 'read_context'} else None
        spans.append(span)
        # Context is an observed predecessor, explicitly NOT a certified handoff.
        opportunities.append(opportunity(task, trace_sha, origin, [previous, span] if previous else [span], missing))
    by_event = {s['event_id']: s for s in spans}
    for chain in recover(task, trace, environment):
        selected = [by_event[e] for e in chain['event_ids']]
        item = opportunity(task, trace_sha, 'verified_handoff', selected,
                           ['handoff occurred; numeric correctness and reuse are unverified'])
        item['old_chain_id'] = chain['id']
        opportunities.append(item)
    if not spans:
        opportunities.append(opportunity(task, trace_sha, 'answer_or_task_hypothesis', [],
                                         ['public goal only; no observed operation output']))
    return spans, opportunities


def opportunity(task, trace_sha, origin, spans, missing):
    return {'schema': 'operation_opportunity.v1', 'id': 'opp_' + digest([trace_sha, origin, [s['id'] for s in spans]]),
        'source_case_id': task.physical_key, 'source_task': {'task_id': task.task_id, 'goal': task.goal,
            'inputs': deepcopy(task.inputs), 'split': 'train'}, 'source_trace_sha256': trace_sha,
        'origin': origin, 'events': [{k: s[k] for k in ('id', 'event_id', 'name', 'result_id', 'state', 'backend_invoked')}
                                   for s in spans],
        'public_local_goal': 'Parameterize one observed local operation or explicitly unverified correction',
        'missing_preconditions': missing, 'qualification': QUALIFICATION}


def select(opportunities, spans, source_ids, maximum):
    """One actual read (else accepted operation, else failure) per preregistered task; then failures."""
    by_id = {s['id']: s for s in spans}
    chosen = []
    for task_id in source_ids:
        group = [o for o in opportunities if o['source_task']['task_id'] == task_id]
        if not group: raise ValueError('Preregistered Train trace is missing: ' + task_id)
        def rank(o):
            events = [by_id[e['id']] for e in o['events']]
            return (0 if o['origin'] == 'observed_local_step' and any(s['name'] == 'read' for s in events)
                    else 1 if o['origin'] == 'verified_handoff' else 2 if o['origin'] == 'observed_local_step' else 3)
        chosen.append(min(group, key=rank)['id'])
    for task_id in source_ids:
        failure = next((o for o in opportunities if o['source_task']['task_id'] == task_id
                        and o['origin'] == 'observed_failure' and o['id'] not in chosen), None)
        if failure and len(chosen) < maximum: chosen.append(failure['id'])
    return chosen[:maximum]


def material(opportunity, spans, window_chars=12000):
    """At most two exact public windows, with explicit truncation receipts and original hashes."""
    by_id = {s['id']: s for s in spans}
    def window(value, path=()):
        if isinstance(value, str) and len(value) > window_chars:
            return {'window': value[:window_chars], 'offset': 0, 'end': window_chars,
                    'full_size': len(value), 'full_sha256': digest(value), 'source_path': list(path)}
        if isinstance(value, list) and len(value) > 3:
            return {'items': [window(v, (*path, i)) for i, v in enumerate(value[:3])],
                    'full_size': len(value), 'full_sha256': digest(value), 'source_path': list(path)}
        if isinstance(value, dict): return {k: window(v, (*path, k)) for k, v in value.items()}
        if isinstance(value, list): return [window(v, (*path, i)) for i, v in enumerate(value)]
        return value
    selected = [by_id[e['id']] for e in opportunity['events'][-2:]]
    return {'opportunity': opportunity, 'spans': [{**{k: v for k, v in s.items() if k != 'available_fields'},
        'arguments': window(s['arguments']), 'result': window(s['result']),
        'available_fields': [f for f in s['available_fields'] if len(f['path']) <= 2
                             and all(type(p) is not int or p < 3 for p in f['path'])][:32]} for s in selected]}


def bind_inputs(spec, spans, source_ids):
    """Only exact recorded fields; a missing control is defaulted only when its observed value is unique."""
    eligible = {s['id']: s for s in spans if s['id'] in source_ids}
    bindings = deepcopy(spec.get('input_bindings', {}))
    properties = spec['input_schema'].get('properties', {})
    if set(bindings) - set(properties): raise ValueError('extra_schema_field: remove bindings absent from input_schema')
    values, provenance = {}, {}
    for name in properties:
        ref = bindings.get(name)
        if ref is None:
            observed = [(s, s['arguments'][name]) for s in eligible.values() if name in s['arguments']]
            if len({digest(v) for _, v in observed}) != 1:
                if name not in spec['input_schema'].get('required', []): continue
                raise ValueError('missing_observed_control: bind ' + name + ' to an advertised Host field; no invented default')
            s, _ = observed[0]; ref = {'span_id': s['id'], 'section': 'arguments', 'path': [name]}
        if (not isinstance(ref, dict) or set(ref) - {'span_id', 'section', 'path', 'slice'}
                or not {'span_id', 'section', 'path'} <= set(ref) or ref['span_id'] not in eligible
                or ref['section'] not in {'arguments', 'result', 'task'} or not isinstance(ref['path'], list)):
            raise ValueError('invalid_source_id: select an advertised span and exact public field')
        s = eligible[ref['span_id']]
        values[name] = at(s['source_task'] if ref['section'] == 'task' else s[ref['section']], ref['path'])
        if 'slice' in ref:
            window = ref['slice']
            if (not isinstance(values[name], str) or not isinstance(window, dict) or set(window) != {'start', 'end'}
                    or any(type(v) is not int for v in window.values())
                    or not 0 <= window['start'] < window['end'] <= len(values[name])):
                raise ValueError('invalid_source_window: bind an exact nonempty observed text substring')
            values[name] = values[name][window['start']:window['end']]
        provenance[name] = {**ref, 'value_sha256': digest(values[name]), 'source_trace_sha256': s['source_trace_sha256']}
    from ..agents.protocol import validate_schema_instance
    validate_schema_instance(values, spec['input_schema'])
    return values, provenance
