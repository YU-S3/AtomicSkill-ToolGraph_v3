"""Independent candidate-only Docker probes and exact downstream consumption witnesses."""
from copy import deepcopy
import math
import re
import statistics

from ..agents.protocol import validate_schema_instance
from ..harness.benchmarks import OfficeAdapter
from ..harness.simple_protocol import Broker
from .contracts import PublicTask, digest
from .program_worker import ProgramWorker
from .task_context import TaskContext
from .office_operation_mining import at


def number(text):
    text = text.strip().replace(',', '')
    if re.fullmatch(r'\(?[-+]?\d+(?:\.\d+)?\)?', text) is None: raise ValueError('ambiguous_numeric_cell')
    value = float('-' + text[1:-1] if text.startswith('(') and text.endswith(')') else text)
    if not math.isfinite(value): raise ValueError('nonfinite_number')
    return value


def table_cells(text, column, unit):
    """Conservative independent oracle, not a general Office answer extractor."""
    if not isinstance(unit, str) or not unit.strip() or unit.casefold() not in text.casefold():
        raise ValueError('unit_missing_or_conflicting')
    unit_scales = set(re.findall(r'\bin (thousands|millions|billions)\b', text, re.I))
    if len({u.casefold() for u in unit_scales}) > 1: raise ValueError('unit_missing_or_conflicting')
    rows = [(i + 1, [p.strip() for p in line.strip().strip('|').split('|')])
            for i, line in enumerate(text.splitlines()) if line.strip().startswith('|')]
    headers = [(i, cells.index(column)) for i, (_, cells) in enumerate(rows) if cells.count(column) == 1]
    if len(headers) != 1: raise ValueError('missing_or_ambiguous_header')
    start, col = headers[0]; result = []
    for line, cells in rows[start + 1:]:
        if len(cells) <= col or not cells[0] or re.fullmatch(r'[-: ]+', cells[0]): continue
        try: value = number(cells[col])
        except ValueError: continue
        result.append({'row': cells[0], 'value': value, 'line': line})
    return result


def expected(contract, inputs):
    kind = contract.get('kind')
    if kind == 'numeric':
        values = inputs[contract['values_input']]
        if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('numeric_values_not_observed')
        op = contract['operation']
        if op == 'sum': value = sum(values)
        elif op == 'mean': value = statistics.mean(values)
        elif op == 'sample_std': value = statistics.stdev(values)
        elif op in {'difference', 'ratio', 'log_ratio'} and len(values) == 2:
            value = values[1] - values[0] if op == 'difference' else values[1] / values[0]
            if op == 'log_ratio': value = math.log(value)
        else: raise ValueError('unsupported_independent_formula')
        return {contract['output_field']: value}
    if kind in {'table_cell', 'series_extract'}:
        rows = table_cells(inputs[contract['text_input']], inputs[contract['column_input']], inputs[contract['unit_input']])
        requested = [inputs[contract['row_input']]] if kind == 'table_cell' else inputs[contract['rows_input']]
        if not isinstance(requested, list) or not requested: raise ValueError('missing_periods')
        values = []
        for label in requested:
            matches = [r for r in rows if r['row'] == label]
            if len(matches) != 1: raise ValueError('missing_or_ambiguous_row')
            values.append(matches[0]['value'])
        return {contract['output_field']: values[0] if kind == 'table_cell' else values}
    if kind == 'evidence_check':
        required, text = inputs[contract['required_input']], inputs[contract['text_input']]
        if (not isinstance(required, list) or not required or not isinstance(text, str)
                or any(not isinstance(v, str) or not v.strip() for v in required)):
            raise ValueError('missing_explicit_evidence_requirements')
        if any(v.casefold() not in text.casefold() for v in required): raise ValueError('missing_required_text')
        return {contract['output_field']: True}
    if kind == 'retrieval': return None
    raise ValueError('no_independent_oracle')


def probe_cases(bundle, transfer_spans=()):
    """Freeze probes before execution, without searching for a passing candidate."""
    candidate = bundle['candidate']; inputs = candidate['bound_inputs']; contract = candidate['program_spec']['probe_contract']
    base = {'label': 'positive', 'inputs': deepcopy(inputs), 'source_case_id': bundle['generation_basis']['source_case_id']}
    try: base['expected'] = expected(contract, inputs)
    except (ValueError, KeyError, ZeroDivisionError, statistics.StatisticsError) as exc:
        return {'status': 'coverage_insufficient', 'reason': str(exc), 'cases': []}
    variation = deepcopy(base); variation['label'] = 'variation'
    # First compatible pre-registered non-source operation, selected by public structure only.
    for span in transfer_spans:
        if span['origin'] != 'observed_local_step': continue
        values = {}
        for name, ref in candidate['input_provenance'].items():
            try:
                values[name] = at(span['source_task'] if ref['section']=='task' else span[ref['section']], ref['path'])
                if 'slice' in ref: values[name] = values[name][ref['slice']['start']:ref['slice']['end']]
            except (ValueError, KeyError): break
        if set(values) != set(inputs) or values == inputs: continue
        try:
            validate_schema_instance(values, bundle['program']['input_schema'])
            oracle = expected(contract, values)
        except (ValueError, KeyError, ZeroDivisionError, statistics.StatisticsError): continue
        variation.update(inputs=values, expected=oracle, source_case_id=span['source_task']['physical_key'],
                         transfer_span_id=span['id'], source_trace_sha256=span['source_trace_sha256'])
        break
    negative = deepcopy(base); negative['label'] = 'negative'; negative['expected'] = 'abstain_with_gap'
    if contract['kind'] == 'retrieval':
        key = next((k for k in ('pattern', 'query') if isinstance(inputs.get(k), str)), None)
        if key is None: return {'status': 'coverage_insufficient', 'reason': 'no_observed_search_control', 'cases': []}
        if 'transfer_span_id' not in variation: variation['inputs'][key] = inputs[key] + '|(?!)'
        negative['inputs'][key] = r'(?!)'
        negative['transformation'] = 'empty regex language; absence only after complete search'
    elif contract['kind'] == 'numeric':
        key = contract['values_input']; values = inputs[key]
        if 'transfer_span_id' not in variation:
            variation['inputs'][key] = (values[:-1] if contract['operation'] in {'sum', 'mean', 'sample_std'} else list(reversed(values)))
        negative['inputs'][key] = []
        negative['transformation'] = 'remove observed numeric inputs'
    elif contract['kind'] in {'series_extract', 'evidence_check'}:
        key = contract['text_input']
        if 'transfer_span_id' not in variation:
            if contract['kind'] == 'series_extract':
                labels = inputs[contract['rows_input']]
                if len(labels) < 2: return {'status':'coverage_insufficient','reason':'no_changed_observed_series','cases':[]}
                variation['inputs'][contract['rows_input']] = labels[:-1]
            else: variation['inputs'][key] = '\n' + inputs[key]
        negative['inputs'][key] = ''
        negative['transformation'] = 'remove required acquired evidence'
    else:
        text_key, row_key = contract['text_input'], contract['row_input']
        rows = table_cells(inputs[text_key], inputs[contract['column_input']], inputs[contract['unit_input']])
        other = next((r for r in rows if r['row'] != inputs[row_key]), None)
        if 'transfer_span_id' not in variation:
            if other is None: return {'status': 'coverage_insufficient', 'reason': 'no_other_observed_row', 'cases': []}
            variation['inputs'][row_key] = other['row']
            variation['observed_transform'] = {'input': text_key, 'line': other['line'], 'field': 'row'}
        negative['inputs'][text_key] = ''
        negative['transformation'] = 'remove acquired text/header/row'
    try:
        for case in (variation, negative): validate_schema_instance(case['inputs'], bundle['program']['input_schema'])
        variation['expected'] = expected(contract, variation['inputs'])
        if variation['inputs'] == inputs: raise ValueError('no_changed_legal_parameters')
    except (ValueError, KeyError, ZeroDivisionError, statistics.StatisticsError) as exc:
        return {'status': 'coverage_insufficient', 'reason': str(exc), 'cases': []}
    return {'status': 'ready', 'cases': [base, variation, negative], 'selection': 'fixed structure and first observed alternative, no outcome search'}


def verify_result(contract, case, result, events):
    if case['label'] == 'negative':
        if result.get('status') == 'not_found' and unfinished_search(events):
            return False, 'unfinished_pagination_is_not_not_found'
        passed = result.get('status') in {'not_found', 'needs_input', 'blocked'} and bool(result.get('detail'))
        return passed, 'explicit_abstention' if passed else 'negative_did_not_report_gap'
    if result.get('status') != 'ok': return False, 'positive_worker_status_' + result.get('status', 'missing')
    if contract['kind'] != 'retrieval':
        for field, value in case['expected'].items():
            actual = result.get('outputs', {}).get(field)
            if not equal_value(actual, value):
                return False, 'independent_formula_or_cell_mismatch'
        return True, 'independent_local_oracle'
    output = result.get('outputs', {})
    inputs = case.get('inputs', {})
    query_key = contract.get('query_input') or next((k for k in ('pattern','query') if k in inputs), None)
    if query_key is None: return False, 'no_observed_query_control'
    searches = [e for e in events if e['name'] in {'grep','grep_page'} and e.get('result',{}).get('accepted')
                and e.get('arguments',{}).get('pattern') == inputs[query_key]]
    if 'paths' in inputs:
        searches = [e for e in searches if 'paths' in e['arguments'] and sorted(e['arguments']['paths']) == sorted(inputs['paths'])]
    else:
        searches = [e for e in searches if 'paths' not in e['arguments']]
    if not searches: return False, 'query_or_scope_not_used_in_actual_rpc'
    hits = [h for e in searches for h in (e['result']['data'].get('hits',[]) if isinstance(e['result']['data'],dict) else e['result']['data'])]
    reads = [e for e in events if e['name'] in {'read', 'read_context'} and e.get('result', {}).get('accepted')]
    for event in reads:
        data = event['result']['data']; args = event['arguments']
        if not any(h.get('path')==args['path'] and h.get('offset')==args['offset'] for h in hits): continue
        offset = data.get('start_offset', args['offset'])
        if (output.get('text') and output['text'] == data['text'] and output.get('source_path') == args['path']
                and output.get('source_offset') == offset):
            incomplete = unfinished_search(events)
            if incomplete and output.get('complete') is not False: return False, 'unfinished_search_claimed_complete'
            if not isinstance(output.get('complete'), bool): return False, 'missing_search_completeness'
            return True, 'actual_worker_read_receipt'
    return False, 'no_matching_worker_read_receipt'


def unfinished_search(events):
    latest = {}
    for event in events:
        if event['name'] not in {'grep', 'grep_page'}: continue
        args = event.get('arguments', {})
        key = digest({k:args[k] for k in ('pattern','paths') if k in args})
        result = event.get('result', {}); data = result.get('data')
        latest[key] = (not result.get('accepted', True) or
            (data.get('complete') is not True if isinstance(data, dict) else not isinstance(data,list) or len(data)>=40))
    return any(latest.values())


def equal_value(actual, expected):
    if type(expected) in (int, float):
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal_value(a,e) for a,e in zip(actual,expected))
    return type(actual) is type(expected) and actual == expected


def execute_probe(bundle, case, config, task, worker_settings=None):
    adapter = OfficeAdapter({task['task_id']: {}}, config); adapter.reset(PublicTask(**task))
    broker = Broker(adapter, 32, context=TaskContext(config.get('runtime')))
    try:
        result = ProgramWorker({**(worker_settings or {}), 'record_execution_receipt': True}).execute(
            bundle['program'], case['inputs'], broker)
        passed, reason = verify_result(bundle['candidate']['program_spec']['probe_contract'], case, result, broker.events)
        return {'program_id': bundle['id'], 'case': case, 'result': result, 'events': broker.events,
                'passed': passed, 'reason': reason, 'input_sha256': digest(case['inputs']),
                'source_sha256': digest(bundle['program']['source']), 'http': 0}
    finally: adapter.close()


def consumed(producer, actions):
    """A claim in prose is not consumption. Only Host-resolved, accepted exact field uses count."""
    if producer.get('status') != 'ok': return []
    return [a for a in actions if a.get('accepted') is True and any(
        ref.get('result_id') == producer.get('result_id') and ref.get('path', [])[:1] == ['outputs']
        and len(ref['path']) >= 2 for ref in a.get('resolved_refs', {}).values())]


def partial_workflow(producers, actions, programs):
    """Observed two-program dataflow only; no invented final answer node."""
    from .contracts import validate_workflow
    edges, nodes, seen = [], [], {}
    for action in actions:
        if action.get('kind') != 'program' or action.get('status') != 'ok' or action.get('accepted') is not True: continue
        program = programs[action['program_id']]
        node = {'id': 'observed_' + str(len(nodes)), 'execution_mode': 'program', 'program_id': action['program_id'],
                'purpose': 'Observed intermediate operation', 'args': {}, 'after': []}
        for key, ref in action.get('resolved_refs', {}).items():
            parent = seen.get(ref['result_id'])
            if parent and ref['path'][:1] == ['outputs'] and len(ref['path']) == 2:
                node['args'][key] = {'from': parent['id'], 'field': ref['path'][1]}
                node['after'].append(parent['id'])
                edges.append({'producer': parent['program_id'], 'consumer': action['program_id'],
                              'input': key, 'result_ref': ref})
            else: node['args'][key] = {'unresolved': {'public_result_ref': ref}}
        validate_schema_instance(action['bound_inputs'], program['input_schema'])
        validate_schema_instance(action['result']['outputs'], program['output_schema'])
        node['after'] = sorted(set(node['after'])); nodes.append(node); seen[action['result_id']] = node
    if not edges: return None
    value = {'interface_version': 'empirical.workflow.v2', 'goal': 'Observed bounded Program dataflow',
             'nodes': nodes, 'outputs': {}, 'partial': True, 'observed_edges': edges, 'formal_publication': False}
    validate_workflow(value, known_programs=programs, known_skills={})
    return value


def runtime_view(context, programs):
    """V4-only compact current facts; canonical results remain accessible by exact references."""
    records = []
    for memory in list(context.memory.values())[-8:]:
        rid = memory['result_id']; raw = context.public_result(context.results[rid], origin=context.origin(rid))
        data = raw.get('data', raw.get('outputs', {}))
        preview = json_preview(data)
        records.append({**memory, 'public_preview': preview, 'full_result_ref': {'result_id': rid, 'path': ['data'] if 'data' in raw else ['outputs']},
                        'preview_is_not_complete': preview != data})
    return {'actual_results': records, 'programs': [{**b['atomic'], 'id': b['id'], 'atomic_id': b['atomic'].get('id'),
        'input_schema': b['program']['input_schema'], 'output_schema': b['program']['output_schema'],
        'missing_inputs': list(b['program']['input_schema'].get('required', [])),
        'binding': 'resolve every input from this episode public ResultRefs; no future values'} for b in programs]}


def json_preview(value):
    if isinstance(value, str): return value[:3000]
    if isinstance(value, list): return [json_preview(v) for v in value[:3]]
    if isinstance(value, dict): return {k: json_preview(v) for k, v in value.items()}
    return value
