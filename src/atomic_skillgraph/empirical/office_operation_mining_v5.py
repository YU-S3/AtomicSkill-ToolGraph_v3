"""Train-only operation inventory and deterministic, outcome-blind preregistration."""
from collections import Counter
from copy import deepcopy
import re

from .contracts import digest
from .office_operation_mining import fields, mine as mine_v4
from .office_program_contract_v5 import MATERIAL_VERSION


def requirements(opportunity, spans, existing):
    selected = [s for s in spans if s['id'] in {e['id'] for e in opportunity['events']}]
    names = {s['name'] for s in selected}
    texts = [s['result'].get('data', {}).get('text', '') for s in selected if isinstance(s['result'].get('data'), dict)]
    table = any(re.search(r'^\s*\|.*\|', t, re.M) for t in texts)
    kinds = []
    if names & {'glob', 'grep', 'grep_page'}: kinds.append('source_locator')
    if names & {'read', 'read_context'}:
        kinds += ['table_locator', 'evidence_check']
        if table: kinds += ['table_cell', 'series_extract']
    if 'execute_python' in names: kinds += ['numeric_compute', 'evidence_check']
    if not kinds: kinds = ['evidence_check']
    tools = sorted(names - {'execute_python'})
    return {'operation_kinds': kinds, 'known_inputs': [{ 'span_id': s['id'], 'fields': sorted(s['arguments'])} for s in selected],
        'missing_inputs': ['New execution must bind actual query/scope or text/row/column/unit prerequisites.',
                           'Observed failures do not provide successful outputs; incomplete series must remain pending.'],
        'observable_outputs': ['actual source locations/text' if tools else 'local deterministic result if inputs become available'],
        'forbidden_claims': ['unobserved values', 'verified whole-task answer', 'unexecuted producer-consumer handoff'],
        'table_text_observed': table, 'equivalence': 'Host cannot infer semantic equivalence from tool overlap',
        'existing_tool_overlap_ids': [p['id'] for p in existing if tools and set(tools) & set(p.get('allowed_tools', []))]}


def mine(trace, trace_sha, environment, existing):
    spans, opportunities = mine_v4(trace, trace_sha, environment)
    for op in opportunities:
        op['operation_requirements'] = requirements(op, spans, existing)
        selected = [s for s in spans if s['id'] in {e['id'] for e in op['events']}]
        op['operation_fingerprint'] = digest([(s['name'], s['arguments'], s['raw_result_sha256']) for s in selected])
    return spans, opportunities


def material(opportunity, spans, window_chars=12000):
    by_id = {s['id']: s for s in spans}
    selected = [by_id[e['id']] for e in opportunity['events'][-2:]]
    source_task = selected[-1]['source_task'] if selected else {**opportunity['source_task'], 'physical_key': opportunity['source_case_id']}
    projected = []
    for span in selected:
        catalog = []
        for section, source in [('arguments', span['arguments']), ('result', span['result']), ('task', source_task)]:
            # Scalar leaves and small arrays are explicit addressable values; large containers are not
            # advertised as fully visible. Text windows retain their original Unicode coordinates.
            for field in fields(source):
                path = field['path']
                if not path or len(path) > 5 or any(type(p) is int and p >= 3 for p in path): continue
                from .office_operation_mining import at
                value = at(source, path)
                if isinstance(value, dict): continue
                if isinstance(value, list) and (len(value) > 3 or any(isinstance(x, (dict, list)) for x in value)): continue
                item = {'section': section, **field}
                if section == 'task':
                    item['value_from_public_task'] = path
                elif isinstance(value, str) and len(value) > window_chars:
                    item['window'] = {'text': value[:window_chars], 'start': 0, 'end': window_chars, 'full_length': len(value)}
                else: item['value'] = value
                catalog.append(item)
        projected.append({k: span[k] for k in ('id', 'event_id', 'result_id', 'name', 'origin', 'state', 'backend_invoked')})
        projected[-1]['fields'] = catalog
    return {'material_version': MATERIAL_VERSION, 'source_task': deepcopy(source_task),
        'opportunity': {k: deepcopy(v) for k, v in opportunity.items() if k not in {'source_task', 'events'}},
        'spans': projected}


def inventory(opportunities, spans):
    return {'tasks': len({o['source_case_id'] for o in opportunities}), 'opportunities': len(opportunities),
        'nonduplicate_operations': len({o['operation_fingerprint'] for o in opportunities}),
        'origins': dict(Counter(o['origin'] for o in opportunities)),
        'operation_kinds': dict(Counter(k for o in opportunities for k in o['operation_requirements']['operation_kinds'])),
        'numeric_table_windows': sum(o['operation_requirements']['table_text_observed'] for o in opportunities),
        'truncated_legacy_grep': sum(s['name'] == 'grep' and isinstance(s['result'].get('data'), list)
            and len(s['result']['data']) >= 40 for s in spans),
        'selection_inputs': 'public Train task and actual operations only; final answer/score/gold excluded'}


def select(opportunities, *, focus_ids, transfer_tasks, task_count=8, maximum=10, paid_maximum=4):
    """Freeze diverse operations before requests. No success filtering or earliest-read rule."""
    eligible = [o for o in opportunities if o['source_task']['task_id'] not in transfer_tasks]
    by_id = {o['id']: o for o in eligible}
    if any(oid not in by_id for oid in focus_ids): raise ValueError('Focused V4 Train operation missing')
    chosen, seen, tasks, covered = [], set(), set(), set()
    priorities = ['table_cell', 'series_extract', 'table_locator', 'evidence_check', 'source_locator', 'numeric_compute']
    def add(op, kind):
        chosen.append({'opportunity_id': op['id'], 'requested_kind': kind, 'source_task_id': op['source_task']['task_id']})
        seen.add(op['operation_fingerprint']); tasks.add(op['source_case_id']); covered.add(kind)
    for oid in focus_ids:
        op = by_id[oid]; kinds = op['operation_requirements']['operation_kinds']
        add(op, next((k for k in priorities if k in kinds and k not in covered), kinds[0]))
    while len(chosen) < maximum:
        pool = [o for o in eligible if o['operation_fingerprint'] not in seen and
                (o['source_case_id'] not in tasks if len(tasks) < task_count else o['source_case_id'] in tasks)]
        if not pool: break
        options = [(o, k) for o in pool for k in o['operation_requirements']['operation_kinds']]
        op, kind = min(options, key=lambda pair: (pair[1] in covered, priorities.index(pair[1]),
                                                digest([pair[0]['operation_fingerprint'], pair[1]])))
        add(op, kind)
    if not 8 <= len(chosen) <= 12 or not 6 <= len(tasks) <= 10:
        raise ValueError('Insufficient distinct Train operations for V5 preregistration')
    available = {k for o in eligible for k in o['operation_requirements']['operation_kinds']}
    return {'entries': chosen, 'opportunities': [o['opportunity_id'] for o in chosen],
        'paid_opportunities': [o['opportunity_id'] for o in chosen[:paid_maximum]],
        'transfer_tasks': transfer_tasks, 'source_tasks': sorted({o['source_task_id'] for o in chosen}),
        'covered': sorted(covered), 'uncovered': sorted(available - covered),
        'absent_categories': sorted(set(priorities) - available),
        'rule': 'V4 op2/op3 needs fixed first; then uncovered operation kind, distinct physical task, public fingerprint hash; no score filtering'}
