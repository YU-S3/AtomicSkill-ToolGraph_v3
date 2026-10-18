"""Outcome-blind partition and role-checked loading of immutable Train evidence."""
from collections import Counter
import hashlib
from pathlib import Path

from .contracts import digest
from .office_effect_contract import VERSION, MATERIAL
from .office_effect_handoff import recover_handoffs
from .office_operation_mining import read, sha
from .public_results import public_result


def partition(records):
    keys = [r['task']['physical_key'] for r in records]
    if len(keys) != 120 or len(set(keys)) != 120 or any(r['task']['split'] != 'train' for r in records):
        raise ValueError('V6 requires the original canonical Train120')
    ordered = sorted(keys, key=lambda k: hashlib.sha256((k + VERSION).encode()).hexdigest())
    return {'method_version': VERSION, 'rule': 'sha256(physical_key + method_version)',
            'induction': ordered[:80], 'edit_feedback': ordered[80:100], 'train_gate': ordered[100:],
            'pilot': ordered[80:84]}


def load_case(record, split, role):
    if role != 'induction' or record['task']['physical_key'] not in split['induction']:
        raise ValueError('Historical held-out traces cannot enter the candidate generator')
    if sha(record['path']) != record['sha256']: raise ValueError('Historical trace changed')
    trace = read(record['path'])
    if trace['task'] != record['task']: raise ValueError('Historical public task identity changed')
    return trace


def window(value, limit):
    if isinstance(value, str) and len(value) > limit:
        return {'text': value[:limit], 'start': 0, 'end': limit, 'full_length': len(value),
                'sha256': digest(value), 'truncated': True}
    if isinstance(value, list):
        return {'items': [window(v, limit) for v in value[:3]], 'full_count': len(value),
                'sha256': digest(value)} if len(value) > 3 else [window(v, limit) for v in value]
    if isinstance(value, dict): return {k: window(v, limit) for k,v in value.items()}
    return value


def material(trace, trace_sha, corpus_identity):
    events = trace.get('tools', []); gaps = []
    for e in events:
        if e.get('state') != 'finished' or not e.get('backend_invoked'): continue
        name, args, raw = e.get('name'), e.get('arguments', {}), e.get('result', {})
        data = raw.get('data')
        if name in {'grep', 'grep_page'} and raw.get('accepted'):
            if not args.get('paths'): gaps.append({'kind':'search_not_scoped','status':'observed','event_id':e['event_id']})
            if isinstance(data, dict) and data.get('complete') is False:
                gaps.append({'kind':'search_incomplete','status':'observed','event_id':e['event_id']})
        if name == 'grep' and isinstance(data, list) and len(data) >= 40:
            gaps.append({'kind':'search_incomplete','status':'hypothesis','event_id':e['event_id']})
    handoffs = recover_handoffs(events, corpus_identity)
    gaps += [{'kind':r['reason'],'status':'observed','event_id':r['consumer_event_id']} for r in handoffs
             if r['reason'] == 'cross_file_handoff_invalid']
    reason = trace.get('execution', {}).get('reason')
    if reason == 'runtime_finish_reserved': gaps.append({'kind':'finish_budget_reserved','status':'observed'})
    if not gaps: gaps = [{'kind':'unknown','status':'hypothesis','detail':'A wrong answer alone does not establish a failure cause'}]
    # Every event retains its identity/arguments/outcome. Text excerpts are explicitly bounded;
    # full raw files remain immutable and are not replaced by inferred row/column annotations.
    actual = []
    for event in events:
        raw = public_result(event.get('result', {}), event.get('result_protocol'))
        actual.append({'event_id':event.get('event_id'),'result_id':event.get('result_id'),
            'name':event.get('name'),'state':event.get('state'),'backend_invoked':event.get('backend_invoked'),
            'arguments':window(event.get('arguments', {}), 350),
            'result':window({k:raw[k] for k in ('accepted','data','error','error_code') if k in raw}, 450),
            'raw_result_sha256':digest(event.get('result', {}))})
    return {'material_version':MATERIAL, 'case_id':trace['task']['physical_key'],
        'task_public':trace['task'], 'trace_sha256':trace_sha,
        'official_train_outcome':trace['score'], 'observed_submission':trace.get('execution', {}).get('prediction'),
        'termination_reason':reason, 'actual_events':actual, 'handoffs':handoffs,
        'cost':{'tokens':sum(r['total_tokens'] for r in trace.get('usage', [])), 'tools':len(events)},
        'evidence_gaps':gaps, 'warning':'Accepted tool output is an observation, not whole-task correctness; omitted text is not absence'}


def build_catalog(records, split, corpus_identity):
    by_key = {r['task']['physical_key']: r for r in records}
    return [material(load_case(by_key[k], split, 'induction'), by_key[k]['sha256'], corpus_identity)
            for k in split['induction']]


def learning_projection(case):
    """Declared excerpts, sampled by tool kind and chronology, never by answer outcome."""
    events = case['actual_events']; groups = {}
    for index,event in enumerate(events): groups.setdefault(event['name'],[]).append(index)
    selected = set()
    for name in ('grep','grep_page','read','read_context','execute_python','glob'):
        indices = groups.get(name,[])
        if indices:
            selected.add(indices[0]); selected.add(indices[-1])
    # Six tools × first/last is bounded independently of the number of recorded calls.
    rows = []
    for i in sorted(selected):
        event = events[i]
        rows.append({'index':i,'event_id':event['event_id'],'result_id':event['result_id'],'name':event['name'],
            'arguments':window(event['arguments'],120),'result':window(event['result'],160),
            'state':event['state'],'backend_invoked':event['backend_invoked']})
    return {**{k:case[k] for k in ('case_id','task_public','trace_sha256','official_train_outcome','termination_reason','cost','warning')},
        'observed_submission':window(case['observed_submission'],400),
        'event_counts':dict(Counter(e['name'] for e in events)),
        'actual_event_excerpts':rows,'full_event_count':len(events),'shown_indices':sorted(selected),
        'projection_rule':'first and last of each actual tool kind; bounded exact text prefixes, no outcome-based selection',
        'evidence_gap_counts':dict(Counter((r['status']+':'+r['kind']) for r in case['evidence_gaps'])),
        'handoff_counts':dict(Counter(r['state'] for r in case['handoffs']))}
