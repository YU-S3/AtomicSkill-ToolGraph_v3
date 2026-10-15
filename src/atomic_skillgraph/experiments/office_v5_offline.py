"""Read-only V4 replay and real-corpus engineering probes; never learned assets."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import re

from ..agents.protocol import parse_json_strict
from ..empirical.checkpoint import save_json
from ..empirical.contracts import digest, object_schema
from ..empirical.office_operation_mining import read, sha, at
from ..empirical.office_operation_mining_v5 import material
from ..empirical.office_program_contract_v5 import (ContractError, validate_candidate, check_schema, BINDING_KINDS)
from ..empirical.office_trace_program_induction_v5 import EXTRACTOR, repair_material
from ..empirical.office_program_probe_v5 import expected, execute_probe, table_negatives
from .office_v4_offline import SEARCH_FIXTURE, engineering_program
from .run_office_trace_v5_pilot import checked, rows, permission, provider_for
from .formal_log import utc


def replay_v4(root):
    plan = read(root / 'pilot_manifest.json'); prior = Path(plan['v4_run'])
    spans = rows(root / 'source_span_catalog.jsonl'); ops = {o['id']: o for o in rows(root / 'opportunity_catalog.jsonl')}
    selection = read(prior / 'selection.json')['opportunities']; config = read(root / 'runtime_config.json')
    tools = permission(config)['allowed_names']; existing = read(root / 'parent_programs.json')
    records = []
    for path in sorted((prior / 'requests').glob('op*/*.json')):
        record = read(path); index = int(path.parent.name[2:]); op = ops[selection[index]]; data = material(op, spans)
        evidence_hash = digest(data); diagnostics = []
        try: proposal = parse_json_strict(record.get('turn', {}).get('content', ''))
        except ValueError:
            proposal = None
        try: validate_candidate(proposal, op, spans, tools, existing, data)
        except ContractError as exc: diagnostics.append(exc.diagnostic)
        # Diagnostic-only inspection of legacy spelling; never normalize or admit it as a V5 candidate.
        spec = proposal.get('program_spec', {}) if isinstance(proposal, dict) else {}
        if isinstance(spec, dict):
            props = spec.get('input_schema', {}).get('properties', {})
            probe = spec.get('probe_contract', {})
            if isinstance(probe, dict) and isinstance(props, dict):
                for key, name in probe.items():
                    if key.endswith('_input') and (not isinstance(name, str) or name not in props):
                        diagnostics.append(ContractError('probe_input_field_missing', 'program_spec.probe_contract.' + key,
                            list(props), name, evidence_hash).diagnostic)
            refs = spec.get('input_bindings', {})
            refs = list(refs.values()) if isinstance(refs, dict) else refs if isinstance(refs, list) else []
            for ref in refs:
                if not isinstance(ref, dict): continue
                span = next((s for s in spans if s['id'] == ref.get('span_id') and s['source_task']['physical_key'] == op['source_case_id']), None)
                try:
                    if span is None: raise ValueError('unknown_source_span')
                    section = ref.get('section'); value = at(span['source_task'] if section == 'task' else span[section], ref['path'])
                    if 'slice' in ref:
                        sl = ref['slice']
                        if not isinstance(value, str) or not isinstance(sl, dict) or type(sl.get('start')) is not int or type(sl.get('end')) is not int or not 0 <= sl['start'] < sl['end'] <= len(value):
                            raise ValueError('text_slice_out_of_bounds')
                except (KeyError, ValueError, TypeError) as exc:
                    diagnostics.append(ContractError(str(exc) if str(exc) in {'unknown_source_span', 'text_slice_out_of_bounds'} else 'path_not_observed',
                        'program_spec.input_bindings', 'actual displayed field/window', ref, evidence_hash).diagnostic)
        codes = {d['code'] for d in diagnostics}
        categories = []
        if codes & {'binding_shape_invalid', 'proposal_shape_invalid', 'schema_invalid'}: categories.append('shape_invalid')
        if codes & {'probe_input_field_missing', 'probe_output_field_missing', 'field_ref_not_declared', 'value_type_mismatch'}: categories.append('semantic_ref_wrong')
        if codes & {'path_not_observed', 'unknown_source_span', 'text_slice_out_of_bounds', 'source_incomplete'}: categories.append('source_missing')
        if isinstance(proposal, dict) and proposal.get('decision') == 'no_change': categories.append('model_abstain')
        if not categories: categories.append('reuse_existing' if isinstance(proposal, dict) and proposal.get('decision') == 'reuse_existing' else 'other_rejection')
        records.append({'request_path': str(path), 'request_sha256': sha(path), 'physical_requests': len(record.get('requests', [])),
            'decision': proposal.get('decision') if isinstance(proposal, dict) else None, 'categories': categories,
            'diagnostics': diagnostics, 'new_generation': False, 'new_candidate_created': False})
    result = {'http': 0, 'records': records, 'physical_requests': sum(r['physical_requests'] for r in records),
        'categories_nonexclusive': dict(Counter(k for r in records for k in r['categories'])),
        'original_responses_unchanged': all(sha(r['request_path']) == r['request_sha256'] for r in records),
        'claim': 'Diagnostic-only legacy replay; no rewritten responses or V5 generation success'}
    save_json(root / 'zero_http/v4_response_replay.json', result)
    return result


CELL_FIXTURE = '''def run(ctx, inputs):
    text = inputs['text']
    if not inputs['unit'] or inputs['unit'].lower() not in text.lower():
        return {'status':'needs_input','outputs':{},'detail':'unit missing'}
    scales = [u for u in ('thousands','millions','billions') if 'in '+u in text.lower()]
    if len(scales) > 1: return {'status':'needs_input','outputs':{},'detail':'unit conflict'}
    lines = text.splitlines()
    matches = [i for i,line in enumerate(lines) if line.strip().startswith('|') and inputs['column'] in [c.strip() for c in line.strip().strip('|').split('|')]]
    if len(matches) != 1: return {'status':'needs_input','outputs':{},'detail':'ambiguous header'}
    h = matches[0]; header = [c.strip() for c in lines[h].strip().strip('|').split('|')]
    if header.count(inputs['column']) != 1: return {'status':'needs_input','outputs':{},'detail':'duplicate column'}
    col = header.index(inputs['column']); requested = inputs['rows']; found = {}
    for line in lines[h+1:]:
        if not line.strip().startswith('|'): break
        row = [c.strip() for c in line.strip().strip('|').split('|')]
        if row and row[0] in requested:
            if row[0] in found or len(row) != len(header): return {'status':'needs_input','outputs':{},'detail':'ambiguous row'}
            try: found[row[0]] = float(row[col].replace(',',''))
            except ValueError: return {'status':'needs_input','outputs':{},'detail':'invalid value'}
    if not requested or any(r not in found for r in requested): return {'status':'needs_input','outputs':{},'detail':'missing row'}
    return {'status':'ok','outputs':{'values':[found[r] for r in requested]}}
'''


def check(root):
    plan, config = checked(root, corpus=True); before = sha(root / 'budget/usage.json')
    replay = replay_v4(root); spans = rows(root / 'source_span_catalog.jsonl')
    ops = {o['id']: o for o in rows(root / 'opportunity_catalog.jsonl')}; selection = read(root / 'selection.json')
    permissions = permission(config); provider = provider_for(config, 'extractor', None)
    existing = [{k:p.get(k) for k in ('id', 'input_schema', 'output_schema', 'allowed_tools')} for p in read(root / 'parent_programs.json')]
    wires = []
    for oid in selection['opportunities']:
        data = {'source': material(ops[oid], spans), 'permissions': permissions, 'existing_programs': existing}
        payload = provider._build_payload([{'role': 'system', 'content': EXTRACTOR}, {'role': 'user', 'content': json.dumps(data, ensure_ascii=False, allow_nan=False)}], [])
        size = len(json.dumps(payload, allow_nan=False).encode())
        wires.append({'opportunity_id': oid, 'wire_bytes': size, 'within_cap': size <= plan['config']['wire_max_bytes']})
        if oid in selection['opportunities'][:2]:
            index = selection['opportunities'].index(oid) + 2
            old = read(Path(plan['v4_run']) / 'requests' / ('op' + str(index)) / 'proposal.json')
            diagnostic = {'code': 'binding_shape_invalid', 'field_path': 'program_spec.input_bindings',
                          'expected': 'object<string, InputBinding>', 'allowed_binding_kinds': BINDING_KINDS,
                          'repairable': True, 'evidence_hash': digest(data['source'])}
            repaired = repair_material(old, diagnostic, data)
            rp = provider._build_payload([{'role':'system','content':EXTRACTOR},
                {'role':'user','content':json.dumps(repaired,ensure_ascii=False,allow_nan=False)}], [])
            size = len(json.dumps(rp,allow_nan=False).encode())
            wires.append({'opportunity_id':oid,'stage':'repair_capacity_from_old_response_only','wire_bytes':size,
                          'within_cap':size <= plan['config']['wire_max_bytes']})
    sources = read(root / 'source_records.json'); receipts = []
    # The op3 trace contains this real table; these selectors are engineering fixtures only,
    # not an input to generation or any candidate Bank. The expected numbers are in the public text.
    op = ops[selection['opportunities'][1]]
    span = next(s for s in spans if s['id'] == op['events'][-1]['id'])
    task = next(s['task'] for s in sources if s['task']['physical_key'] == op['source_case_id'])
    corpus_text = (Path(plan['corpus_root']) / span['arguments']['path']).read_text(encoding='utf-8')
    observed = span['result']['data']['text']; offset = span['arguments']['offset']
    if corpus_text[offset:offset + len(observed)] != observed: raise AssertionError('Actual corpus/Train text identity differs')
    contract = {'kind': 'series_extract', 'text_input_field': 'text', 'rows_input_field': 'rows',
                'column_input_field': 'column', 'unit_input_field': 'unit', 'output_field': 'values'}
    inputs = {'text': observed, 'rows': ['1934', '1935'], 'column': "Veterans' Administration", 'unit': 'millions of dollars'}
    if expected(contract, inputs) != {'values': [507.0, 556.0]}: raise AssertionError('Independent real corpus oracle differs')
    props = {'text': {'type': 'string'}, 'rows': {'type': 'array', 'items': {'type': 'string'}},
             'column': {'type': 'string'}, 'unit': {'type': 'string'}}
    program = engineering_program(CELL_FIXTURE, object_schema(props, list(props)),
        object_schema({'values': {'type': 'array', 'items': {'type': 'number'}}}, ['values']), [], config)
    value = {'id': program['id'], 'program': program, 'candidate': {'program_spec': {'probe_contract': contract}}}
    cases = [('positive', inputs), ('variation', {**inputs, 'rows': ['1935', '1934']}), ('negative', {**inputs, 'rows': ['__absent_year__']})]
    def run(label, bundle, case):
        receipt = execute_probe(bundle, case, config, task)
        receipt['engineering_fixture_only'] = True
        save_json(root / 'zero_http' / (label + '.json'), receipt); receipts.append(receipt)
        if not receipt['passed']: raise AssertionError(label + ': ' + receipt['reason'])
    for label, values in cases:
        case = {'label': label, 'inputs': values, 'expected': expected(contract, values) if label != 'negative' else 'abstain_with_gap'}
        run('real_table_' + label, value, case)
    for case in table_negatives(contract, inputs): run('real_table_' + case['label'], value, case)
    run('missing_required', value, {'label':'host_missing_required','inputs':{k:v for k,v in inputs.items() if k!='text'},
                                   'expected':'missing_runtime_precondition','host_rejection':True})
    search_inputs = {'pattern': re.escape('Federal Expenditures - General'), 'paths': [span['arguments']['path']], 'legacy': False}
    search_props = {'pattern': {'type': 'string'}, 'paths': {'type': 'array', 'items': {'type': 'string'}}, 'legacy': {'type': 'boolean'}}
    outputs = {'text': {'type': 'string'}, 'source_path': {'type': 'string'}, 'source_offset': {'type': 'integer'}, 'complete': {'type': 'boolean'}}
    search = engineering_program(SEARCH_FIXTURE, object_schema(search_props, list(search_props)), object_schema(outputs, list(outputs)),
        ['grep', 'grep_page', 'read', 'read_context'], config)
    value = {'id': search['id'], 'program': search, 'candidate': {'program_spec': {'probe_contract':
        {'kind': 'retrieval', 'query_input_field': 'pattern', 'paths_input_field': 'paths'}}}}
    for label, changes in [('legacy', {'legacy': True}), ('current', {}), ('empty_search', {'pattern': '(?!)'})]:
        run(label, value, {'label': 'negative' if label == 'empty_search' else 'positive', 'inputs': {**search_inputs, **changes}, 'expected': None})
    # Illegal paths must fail at the production Broker; an unsuccessful query is not a valid oracle.
    bad = execute_probe(value, {'label': 'negative', 'inputs': {**search_inputs, 'paths': ['../outside.txt']}, 'expected': None}, config, task)
    if bad['result']['status'] != 'blocked' or not any(not e.get('result',{}).get('accepted') for e in bad['events']):
        raise AssertionError('Corpus path escape not refused')
    bad.update(passed=True, reason='actual_broker_scope_refusal', engineering_fixture_only=True)
    save_json(root / 'zero_http/path_escape.json', bad); receipts.append(bad)
    result = {'passed': all(w['within_cap'] for w in wires), 'time': utc(), 'http': 0,
        'manifest_sha256': sha(root / 'pilot_manifest.json'), 'real_train_sources': len(sources),
        'replay_physical_requests': replay['physical_requests'], 'replay_categories': replay['categories_nonexclusive'],
        'request_capacity': wires, 'docker_engineering_probes': sum(not r.get('host_validation_only') for r in receipts),
        'host_input_rejection_checks': sum(r.get('host_validation_only',False) for r in receipts), 'candidate_assets_created': 0,
        'ledger_unchanged': sha(root / 'budget/usage.json') == before, 'parent_frozen_unchanged': True,
        'claim': 'Engineering fixtures only; no naturally generated V5 candidate or accuracy claim'}
    checked(root, corpus=True)
    if not result['ledger_unchanged']: raise AssertionError('Zero HTTP gate changed budget')
    save_json(root / 'zero_http_check.json', result)
    return result
