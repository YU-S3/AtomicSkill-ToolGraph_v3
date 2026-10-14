"""Zero-model engineering fixtures. Never write these Programs to any learned Bank."""
from copy import deepcopy
import re

from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, object_schema, validate_program, digest
from ..empirical.office_operation_mining import read, sha
from ..empirical.office_program_probe import number, verify_result
from ..empirical.office_trace_program_induction import program_from_response
from ..empirical.program_worker import ProgramWorker
from ..empirical.task_context import TaskContext
from ..harness.benchmarks import OfficeAdapter
from ..harness.simple_protocol import Broker
from .run_office_trace_v4_pilot import checked, rows, permission, provider_for
from .formal_log import utc

SEARCH_FIXTURE = '''def run(ctx, inputs):
    if inputs['legacy']:
        found = ctx.call('grep', {'pattern': inputs['pattern'], 'paths': inputs['paths']})
        hits = found.get('data', []); complete = len(hits) < 40
    else:
        found = ctx.call('grep_page', {'pattern': inputs['pattern'], 'paths': inputs['paths'], 'limit': 40})
        hits = found.get('data', {}).get('hits', []); complete = found.get('data', {}).get('complete', False)
    if not found.get('accepted'): return {'status':'blocked','outputs':{},'detail':'search rejected'}
    if not hits: return {'status':'not_found' if complete else 'needs_input','outputs':{},'detail':'no hit; complete='+str(complete)}
    hit = hits[0]
    args = {'path':hit['path'],'offset':hit['offset']}
    if not inputs['legacy']: args.update(before=0,after=12000)
    page = ctx.call('read' if inputs['legacy'] else 'read_context', args)
    if not page.get('accepted'): return {'status':'blocked','outputs':{},'detail':'read rejected'}
    data = page['data']
    return {'status':'ok','outputs':{'text':data['text'],'source_path':hit['path'],
        'source_offset':data.get('start_offset',hit['offset']),'complete':complete}}
'''

CELL_FIXTURE = '''def run(ctx, inputs):
    page = ctx.call('read', {'path':inputs['path'],'offset':inputs['offset']})
    if not page.get('accepted'): return {'status':'blocked','outputs':{},'detail':'read rejected'}
    text = page['data']['text']; lines = text.splitlines()
    if inputs['unit'] and inputs['unit'].casefold() not in text.casefold():
        return {'status':'needs_input','outputs':{},'detail':'unit absent'}
    row, column = inputs['row'], inputs['column']
    if not 0 <= row < len(lines) or not lines[row].strip().startswith('|'):
        return {'status':'needs_input','outputs':{},'detail':'row absent'}
    cells = lines[row].strip().strip('|').split('|')
    if not 0 <= column < len(cells): return {'status':'needs_input','outputs':{},'detail':'column absent'}
    token = cells[column].strip().replace(',','')
    try: value = float('-'+token[1:-1] if token.startswith('(') and token.endswith(')') else token)
    except ValueError: return {'status':'needs_input','outputs':{},'detail':'cell nonnumeric'}
    return {'status':'ok','outputs':{'value':value}}
'''


def engineering_program(source, inputs, outputs, allowed_tools, config):
    value = {'source': source, 'entry': 'run', 'backend': 'sandbox_python_v1', 'input_schema': inputs,
             'output_schema': outputs, 'allowed_tools': allowed_tools, 'result_role': 'intermediate',
             'environment': config['program_environment']}
    value['id'] = 'engineering_fixture_' + validate_program(value)
    return value


def check(root):
    plan, config = checked(root, corpus=True); spans = rows(root / 'source_span_catalog.jsonl')
    selected = read(root / 'selection.json')['opportunities']; opportunities = rows(root / 'opportunity_catalog.jsonl')
    sources = read(root / 'source_records.json')
    fixtures = root / 'zero_http'; fixtures.mkdir(exist_ok=True)
    span = next(s for s in spans if s['name'] == 'read' and s['origin'] == 'observed_local_step'
                and s['source_task']['task_id'] == plan['config']['source_tasks'][0])
    task = next(r['task'] for r in sources if r['task']['task_id'] == span['source_task']['task_id'])
    text = span['result']['data']['text']
    lines = [line for line in text.splitlines() if len(line.strip()) >= 20]
    pattern = re.escape(lines[0].strip()[:80])
    inputs = {'pattern': pattern, 'paths': [span['arguments']['path']], 'legacy': True}
    search = engineering_program(SEARCH_FIXTURE,
        object_schema({'pattern': {'type':'string'}, 'paths': {'type':'array','items':{'type':'string'}}, 'legacy': {'type':'boolean'}}, list(inputs)),
        object_schema({'text':{'type':'string'}, 'source_path':{'type':'string'}, 'source_offset':{'type':'integer'}, 'complete':{'type':'boolean'}},
                      ['text','source_path','source_offset','complete']), ['grep','read','grep_page','read_context'], config)
    records = []
    def run_case(name, program, values):
        adapter = OfficeAdapter({task['task_id']: {}}, config); adapter.reset(PublicTask(**task))
        broker = Broker(adapter, 32, context=TaskContext(config['runtime']))
        try:
            result = ProgramWorker({'record_execution_receipt': True}).execute(program, values, broker)
            row = {'name': name, 'source': program, 'inputs': values, 'result': result, 'events': broker.events,
                   'engineering_fixture_only': True, 'model_http': 0, 'source_trace_sha256': span['source_trace_sha256']}
            save_json(fixtures / (name + '.json'), row); records.append(row)
            return result, broker.events
        finally: adapter.close()
    for legacy in (True, False):
        result, events = run_case('legacy_search' if legacy else 'current_search', search, {**inputs, 'legacy': legacy})
        passed, reason = verify_result({'kind':'retrieval'}, {'label':'positive','inputs':{**inputs,'legacy':legacy}}, result, events)
        if not passed: raise AssertionError('Real corpus search gate: ' + reason)
    result, events = run_case('no_match', search, {**inputs, 'pattern': '(?!)', 'legacy': False})
    assert verify_result({'kind':'retrieval'}, {'label':'negative'}, result, events)[0]
    result, _ = run_case('scope_escape', search, {**inputs, 'paths': ['../outside.txt']})
    assert result['status'] == 'blocked'
    cells = []
    for line, content in enumerate(text.splitlines()):
        if not content.strip().startswith('|'): continue
        for column, token in enumerate(content.strip().strip('|').split('|')):
            try: value = number(token)
            except ValueError: continue
            cells.append((line, column, value))
    if len(cells) < 2: raise AssertionError('Pre-registered actual source has insufficient numeric cells')
    cell_inputs = {'path': span['arguments']['path'], 'offset': span['arguments']['offset'],
                   'row': cells[0][0], 'column': cells[0][1], 'unit': ''}
    cell = engineering_program(CELL_FIXTURE, object_schema({k:{'type':'string' if k in {'path','unit'} else 'integer'}
        for k in cell_inputs}, list(cell_inputs)), object_schema({'value':{'type':'number'}}, ['value']), ['read'], config)
    for index in (0, 1):
        row, col, expected = cells[index]
        result, _ = run_case('numeric_cell_' + str(index), cell, {**cell_inputs, 'row':row, 'column':col})
        assert result['status'] == 'ok' and result['outputs']['value'] == expected
    for name, change in [('missing_row', {'row':-1}), ('unit_conflict', {'unit':'__absent_unit_probe__'})]:
        result, _ = run_case(name, cell, {**cell_inputs, **change})
        assert result['status'] == 'needs_input' and result.get('detail')
    # Check actual request assembly capacities without calling complete()/loading credentials.
    from ..empirical.office_operation_mining import material
    from ..empirical.office_trace_program_induction import EXTRACTOR
    provider = provider_for(config, 'extractor', None); permissions = permission(config)
    existing_view = [{k: p.get(k) for k in ('id','input_schema','output_schema','allowed_tools','entry_constraints')}
                     for p in read(root / 'parent_programs.json')]
    wire = []
    for oid in selected:
        op = next(o for o in opportunities if o['id'] == oid)
        data = {**material(op, spans), 'existing_programs':existing_view, 'permissions':permissions}
        payload = provider._build_payload([{'role':'system','content':EXTRACTOR},
            {'role':'user','content':__import__('json').dumps(data,ensure_ascii=False,allow_nan=False)}], [])
        size = len(__import__('json').dumps(payload,allow_nan=False).encode())
        wire.append({'opportunity_id':oid,'wire_bytes':size,'within_cap':size<=plan['config']['wire_max_bytes']})
    result = {'passed': all(r['within_cap'] for r in wire), 'http':0, 'manifest_sha256':sha(root/'pilot_manifest.json'),
        'time':utc(), 'real_train_sources':len(sources), 'selected_opportunities':len(selected),
        'candidate_assets_created':0, 'docker_fixtures':len(records), 'request_capacity':wire,
        'parent_frozen_unchanged':True, 'scientific_claim':'engineering fixtures, not naturally learned programs',
        'unit_regressions':'Run tests/test_office_trace_v4_pilot.py with intercepted HTTP; separate test report required'}
    checked(root, corpus=True)
    save_json(root/'zero_http_check.json',result)
    return result
