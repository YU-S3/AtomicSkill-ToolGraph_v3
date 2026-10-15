"""Conservative independent V5 oracles. Unknown layouts remain coverage_insufficient."""
from copy import deepcopy
import re
import statistics

from ..agents.protocol import SchemaValidationError, validate_schema_instance
from ..harness.benchmarks import OfficeAdapter
from ..harness.simple_protocol import Broker
from .contracts import PublicTask, digest
from .office_operation_mining import at
from .office_program_contract_v5 import PROBE_VERSION
from .office_program_probe import number, equal_value, unfinished_search, expected as expected_v4, verify_result as verify_v4
from .program_worker import ProgramWorker
from .task_context import TaskContext

ORACLE_ERRORS = (ValueError, ZeroDivisionError, statistics.StatisticsError)


def table_cells(text, column, unit):
    if not all(isinstance(v, str) and v.strip() for v in (text, column, unit)):
        raise ValueError('source_incomplete: text/column/unit required')
    lines = text.splitlines()
    def cells(line):
        if line.strip().startswith('|'): return [p.strip() for p in line.strip().strip('|').split('|')]
        return re.split(r'\t+| {2,}', line.strip())
    headers = [(i, cells(line)) for i, line in enumerate(lines) if column in cells(line)]
    if len(headers) != 1 or headers[0][1].count(column) != 1:
        raise ValueError('missing_or_ambiguous_header')
    index, header = headers[0]
    if len(header) < 2: raise ValueError('oracle_unavailable: unsupported table layout')
    # Units must belong to this table's preceding context, not another table later in the window.
    begin = index - 1
    while begin >= 0 and not lines[begin].strip().startswith('|') and index - begin <= 12: begin -= 1
    context = '\n'.join(lines[begin + 1:index])
    if not re.search(r'(?<!\w)' + re.escape(unit) + r'(?!\w)', context, re.I): raise ValueError('unit_missing_or_conflicting')
    scales = {u.casefold() for u in re.findall(r'\b(thousands|millions|billions|percent)\b', context, re.I)}
    if len(scales) > 1: raise ValueError('unit_missing_or_conflicting')
    result = []; column_index = header.index(column); pipe = lines[index].strip().startswith('|')
    for i in range(index + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            if result: break
            continue
        if pipe and not line.strip().startswith('|'): break
        row = cells(line)
        if all(re.fullmatch(r'[-: ]*', value) for value in row): continue
        if row == header: raise ValueError('duplicate_header')
        if len(row) != len(header): raise ValueError('source_incomplete: missing column')
        if not row[0]: raise ValueError('source_incomplete: missing row label')
        try: value = number(row[column_index])
        except ValueError: value = None
        result.append({'row': row[0], 'value': value, 'line': i + 1})
    if not result: raise ValueError('source_incomplete: no data rows')
    return result


def expected(contract, inputs):
    kind = contract['kind']
    required = [v for k, v in contract.items() if k.endswith('_input_field')]
    if any(key not in inputs for key in required): raise ValueError('missing_runtime_precondition')
    if kind in {'table_cell', 'series_extract'}:
        rows = table_cells(inputs[contract['text_input_field']], inputs[contract['column_input_field']], inputs[contract['unit_input_field']])
        labels = [inputs[contract['row_input_field']]] if kind == 'table_cell' else inputs[contract['rows_input_field']]
        if not isinstance(labels, list) or not labels or any(not isinstance(s, str) or not s.strip() for s in labels):
            raise ValueError('missing_runtime_precondition: row labels')
        if len(set(labels)) != len(labels): raise ValueError('ambiguous_repeated_row_request')
        values = []
        for label in labels:
            matches = [r for r in rows if r['row'] == label]
            if len(matches) != 1 or matches[0]['value'] is None: raise ValueError('missing_or_ambiguous_row')
            values.append(matches[0]['value'])
        return {contract['output_field']: values[0] if kind == 'table_cell' else values}
    if kind in {'retrieval', 'source_locator'}: return None
    legacy = {k.removesuffix('_field') if k.endswith('_input_field') else k: v for k, v in contract.items()}
    return expected_v4(legacy, inputs)


def probe_cases(value, transfer_spans=()):
    candidate = value['candidate']; inputs = candidate['bound_inputs']; contract = candidate['program_spec']['probe_contract']
    if candidate['pending_inputs']:
        return {'status': 'pending_inputs', 'cases': [], 'pending_inputs': candidate['pending_inputs']}
    if contract['kind'] != 'retrieval' and {'source_path', 'source_offset'} & set(value['program']['output_schema']['properties']):
        return {'status': 'coverage_insufficient', 'reason': 'oracle_unavailable: no independent location output contract', 'cases': []}
    base = {'label': 'positive', 'inputs': deepcopy(inputs), 'source_case_id': candidate['generation_basis']['source_case_id']}
    try: base['expected'] = expected(contract, inputs)
    except ORACLE_ERRORS as exc: return {'status': 'coverage_insufficient', 'reason': str(exc), 'cases': []}
    variation = deepcopy(base); variation['label'] = 'variation'
    negative = deepcopy(base); negative.update(label='negative', expected='abstain_with_gap')
    kind = contract['kind']
    if kind in {'retrieval', 'source_locator'}:
        key = contract['query_input_field' if kind == 'retrieval' else 'pattern_input_field']
        variation['inputs'][key] = '(?:' + inputs[key] + ')' if kind == 'retrieval' else './' + inputs[key]
        negative['inputs'][key] = '(?!)' if kind == 'retrieval' else '__v5_missing_probe__/*'
    elif kind == 'table_cell':
        rows = table_cells(inputs[contract['text_input_field']], inputs[contract['column_input_field']], inputs[contract['unit_input_field']])
        other = next((r for r in rows if r['value'] is not None and r['row'] != inputs[contract['row_input_field']]), None)
        if other is None: return {'status': 'coverage_insufficient', 'reason': 'no_changed_observed_row', 'cases': []}
        variation['inputs'][contract['row_input_field']] = other['row']
        negative['inputs'][contract['text_input_field']] = ''
    elif kind == 'series_extract':
        key = contract['rows_input_field']
        if len(inputs[key]) < 2: return {'status': 'coverage_insufficient', 'reason': 'no_changed_observed_series', 'cases': []}
        variation['inputs'][key] = list(reversed(inputs[key]))
        negative['inputs'][contract['text_input_field']] = ''
    elif kind == 'evidence_check':
        key = contract['text_input_field']; variation['inputs'][key] = '\n' + inputs[key]; negative['inputs'][key] = ''
    else:
        key = contract['values_input_field']; variation['inputs'][key] = list(reversed(inputs[key])); negative['inputs'][key] = []
    try:
        for case in (variation, negative): validate_schema_instance(case['inputs'], value['program']['input_schema'])
        variation['expected'] = expected(contract, variation['inputs'])
        if variation['inputs'] == base['inputs']: raise ValueError('no_changed_legal_parameters')
    except ORACLE_ERRORS as exc: return {'status': 'coverage_insufficient', 'reason': str(exc), 'cases': []}
    cases = [base, variation, negative]
    # Each pre-registered transfer task uses its first structurally bindable event, not the first
    # successful Worker. Original string slices are never silently copied to another text field.
    transfers = []
    for task_id in dict.fromkeys(s['source_task']['task_id'] for s in transfer_spans):
        group = [s for s in transfer_spans if s['source_task']['task_id'] == task_id and s['origin'] == 'observed_local_step']
        bound = {}; refs = {}
        for name, ref in candidate['input_provenance'].items():
            if ref['kind'] == 'observed_text_slice': break
            for span in group:
                if span['name'] != ref['tool_name']: continue
                try:
                    v = at(span['source_task'] if ref['section'] == 'task' else span[ref['section']], ref['path'])
                    validate_schema_instance(v, value['program']['input_schema']['properties'][name])
                except ValueError: continue
                bound[name] = v; refs[name] = {'span_id': span['id'], 'section': ref['section'], 'path': ref['path'], 'value_sha256': digest(v)}
                break
        if set(bound) != set(inputs):
            transfers.append({'task_id': task_id, 'status': 'coverage_insufficient', 'reason': 'no_complete_observed_transfer_binding'}); continue
        case = {'label': 'transfer' + str(len(transfers)), 'inputs': bound, 'source_case_id': group[0]['source_task']['physical_key'],
                'transfer_task_id': task_id, 'input_provenance': refs}
        try:
            validate_schema_instance(bound, value['program']['input_schema']); case['expected'] = expected(contract, bound)
        except ORACLE_ERRORS as exc:
            transfers.append({'task_id': task_id, 'status': 'coverage_insufficient', 'reason': str(exc)}); continue
        cases.append(case); transfers.append({'task_id': task_id, 'status': 'ready', 'case_label': case['label']})
    return {'status': 'ready', 'probe_version': PROBE_VERSION, 'cases': cases, 'transfer_coverage': transfers}


def verify_result(contract, case, result, events):
    if contract['kind'] == 'source_locator':
        calls = [e for e in events if e['name'] == 'glob' and e.get('arguments', {}).get('pattern') == case['inputs'][contract['pattern_input_field']]
                 and e.get('result', {}).get('accepted') is True]
        if not calls: return False, 'no_matching_glob_receipt'
        paths = calls[-1]['result']['data']
        if case['label'] == 'negative':
            return not paths and result.get('status') in {'not_found', 'needs_input'} and bool(result.get('detail')), 'empty_glob_gap'
        return (bool(paths) and result.get('status') == 'ok' and result.get('outputs', {}).get(contract['output_field']) == paths), 'actual_glob_receipt'
    if contract['kind'] == 'retrieval':
        legacy = {'kind': 'retrieval', 'query_input': contract['query_input_field']}
        adapted = deepcopy(case)
        if 'paths_input_field' in contract:
            adapted['inputs']['paths'] = case['inputs'][contract['paths_input_field']]
        return verify_v4(legacy, adapted, result, events)
    if case['label'] == 'negative':
        return (result.get('status') in {'not_found', 'needs_input', 'blocked'} and bool(result.get('detail'))
                and not (result.get('status') == 'not_found' and unfinished_search(events))), 'explicit_gap_required'
    if result.get('status') != 'ok': return False, 'worker_not_ok'
    return all(equal_value(result.get('outputs', {}).get(k), v) for k, v in case['expected'].items()), 'independent_local_oracle'


def execute_probe(value, case, config, task):
    validate_schema_instance(case['inputs'], value['program']['input_schema'])
    adapter = OfficeAdapter({task['task_id']: {}}, config); adapter.reset(PublicTask(**task))
    broker = Broker(adapter, 32, context=TaskContext(config.get('runtime')))
    try:
        result = ProgramWorker({**config.get('program_worker', {}), 'record_execution_receipt': True}).execute(value['program'], case['inputs'], broker)
        passed, reason = verify_result(value['candidate']['program_spec']['probe_contract'], case, result, broker.events)
        if result.get('worker_status') == 'not_started': passed, reason = False, 'worker_contract_invalid'
        if result.get('status') == 'ok':
            try: validate_schema_instance(result.get('outputs'), value['program']['output_schema'])
            except SchemaValidationError: passed, reason = False, 'worker_contract_invalid'
        return {'probe_version': PROBE_VERSION, 'program_id': value['id'], 'case': case, 'result': result,
            'events': broker.events, 'passed': passed, 'reason': reason, 'http': 0,
            'input_sha256': digest(case['inputs']), 'source_sha256': digest(value['program']['source'])}
    finally: adapter.close()
