"""V5 production-path regression: saved failures, honest refusal and direct submission."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, digest, program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http, response
from test_office_human_phase_v3 import load
from test_office_humanstrong_v4 import LocalReads, polynomial_text, polynomial_args, PARENT, RESOURCES
import test_office_humanstrong_v2 as v2

ROOT = Path(__file__).parents[1]/'resources/office_humanstrong_v5'
REPEATS = Path('/home/yangchengyu/officeqa_humanstrong_v4_repeats_seed42_20261020')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs): raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request', refuse)


@pytest.fixture
def setup(tmp_path):
    if not PARENT.exists(): pytest.skip('Immutable public snapshot required')
    cfg = load(ROOT/'integration/configuration.py', 'v5_cfg').configure(
        json.loads((PARENT/'base_config.json').read_text()), RESOURCES, 'test')
    cfg['data_dir'] = str(tmp_path/'bank')
    cfg['experiment']['output_dir'] = str(tmp_path/'run')
    bridge = load(ROOT/'integration/frontier.py', 'v5_bridge')
    bundle = bridge.native_import(json.loads((ROOT/'bank/bank.json').read_text()),
        cfg['data_dir'], cfg['program_environment'], cfg['experiment']['officeqa_method_version'])
    return cfg, bridge, bundle


def display(style='number', decimals=2, scale='as_is', **extra):
    return dict(style=style, decimals=decimals, scale=scale, **extra)


def save_evidence(name, trace, sent):
    if not os.environ.get('V5_EVIDENCE_DIR'): return
    root = Path(os.environ['V5_EVIDENCE_DIR']); root.mkdir(parents=True, exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps({'actual_llm_http': 0,
        'fixture_model_decisions': True, 'real_docker': True, 'trace': trace, 'sent': sent},
        ensure_ascii=False, indent=2))


def test_bundle_reproducibility_and_native_links():
    bank = json.loads((ROOT/'bank/bank.json').read_text())
    old = json.loads((ROOT.parent/'office_humanstrong_v4/bank/bank.json').read_text())
    assert bank['parent_bank_digest'] == digest(old)
    assert len(bank['program']) == 9
    ids = {p['id'] for p in bank['program']}
    skills = {s['id'] for s in bank['skill']}
    for p in bank['program']:
        assert p['source'] == (ROOT/p['source_file']).read_text()
        assert p['id'] == 'program_'+program_digest(p)
    for link in bank['implementation']:
        assert link['program_id'] in ids and link['skill_id'] in skills
    for path in list((ROOT/'programs').glob('*.py'))+[ROOT/'guidance_changes.json']:
        assert 'UID0' not in path.read_text()


@pytest.mark.parametrize('value,contract,expected', [
    ('19.96', display('percent'), '19.96%'),
    ('1453', display(decimals=0), '1453'),
    ('44174', display(decimals=0), '44174'),
    ('-0.0001', display(decimals=1), '0.0'),
    ('115.61', display(), '115.61'),
    ('0.1996', display('percent', scale='fraction_to_percent'), '19.96%'),
    ('1.45', display('unit_suffix', suffix='million euros'), '1.45 million euros'),
])
def test_explicit_format_contract(value, contract, expected):
    formatter = load(ROOT/'programs/answer_format.py', 'v5_format')
    assert formatter.format_answer(value, contract) == expected
    with pytest.raises(ValueError): formatter.format_answer(value+' reasoning', contract)
    with pytest.raises(ValueError): formatter.format_answer(float('inf'), contract)


def table_args(ref):
    return {'submit_answer': True, 'format': display(decimals=0),
        'tables': [{'key': 'series', 'source_ref': ref, 'rows': ['June', 'July'], 'column': 'Value', 'unit': 'widgets'}],
        'bindings': {'values': ['series']}, 'calculation': {'method': 'series_metrics', 'operation': 'sum'}}


def test_rows_units_headers_and_recoverable_bounds():
    module = load(ROOT/'programs/table_calculate.py', 'v5_table')
    text = 'In widgets\n| Period | Value |\n| --- | --- |\n| June. | 123 |\n| July | 456 |\n'
    ctx = LocalReads(text); args = table_args(ctx.ref)
    out = module.run(ctx, args)
    assert out['status'] == 'ok', out
    assert out['outputs']['answer'] == '579'
    for cell in out['outputs']['evidence']:
        assert text[cell['start']:cell['end']] == cell['quote']
    args['format'] = display('unit_suffix', 0, suffix='nominal widgets')
    assert module.run(ctx, args)['outputs']['answer'] == '579 nominal widgets'
    duplicate = LocalReads(text+'| June | 999 |\n')
    out = module.run(duplicate, table_args(duplicate.ref))
    assert out['status'] == 'needs_input' and out['recovery']['code'] == 'row_missing_or_ambiguous'
    args['tables'][0]['offset'] = 10000
    out = module.run(ctx, args)
    assert out['recovery']['code'] == 'offset_out_of_bounds'
    assert out['recovery']['source_size'] == len(text)
    args = table_args(ctx.ref); args['tables'][0]['length'] = text.index('456')+1
    assert module.run(ctx, args)['status'] == 'needs_input'
    extractor = load(ROOT/'programs/table_extract.py', 'v5_extract')
    multi = 'In widgets\n| Period | Category | |\n| --- | --- | --- |\n| | Outside | Total |\n| June | 3 | 9 |\n'
    kwargs = dict(text=multi, rows=['June'], column='Total', unit='widgets')
    assert extractor.run(None, kwargs)['outputs']['values'] == ['9']
    kwargs['column'] = 'Outside'
    assert extractor.run(None, kwargs)['recovery']['code'] == 'header_path_required'
    kwargs['header_path'] = ['Category', 'Outside']
    assert extractor.run(None, kwargs)['outputs']['values'] == ['3']
    kwargs['scope_quote'] = 'unseen category'
    assert extractor.run(None, kwargs)['recovery']['code'] == 'scope_missing'
    double = 'In widgets\n| Period | A | B |\n| --- | --- | --- |\n| | Total | Total |\n| June | 3 | 9 |\n'
    kwargs = dict(text=double, rows=['June'], column='Total', unit='widgets')
    assert extractor.run(None, kwargs)['status'] == 'needs_input'
    kwargs['header_path'] = ['B', 'Total']
    assert extractor.run(None, kwargs)['outputs']['values'] == ['9']
    years = 'In widgets\n| Category | 2001 | 2002 |\n| --- | --- | --- |\n| A | 3 | 9 |\n'
    assert extractor.run(None, dict(text=years, rows=['A'], column='2002', unit='widgets'))['outputs']['values'] == ['9']


def test_schema_projection_is_lossless():
    bridge = load(ROOT/'integration/frontier.py','v5_projection')
    originals = [p['input_schema'] for p in json.loads((ROOT/'bank/bank.json').read_text())['program']]
    packed, definitions = bridge.schema_projection(originals)
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {'$ref'}: return expand(definitions[value['$ref'].rsplit('/',1)[-1]])
            return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list): return [expand(v) for v in value]
        return value
    assert expand(packed) == originals
    assert len(json.dumps([packed,definitions])) < len(json.dumps(originals))


@pytest.mark.parametrize('uid,repeat,expected', [('UID0003', 'repeat_2', '44463'), ('UID0192', 'repeat_3', None)])
def test_actual_saved_selectors(uid, repeat, expected):
    if not REPEATS.exists(): pytest.skip('Saved immutable traces required')
    path = next((REPEATS/repeat).glob(f'episodes/*{uid}/manual/traces/*.json'))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    trace = json.loads(path.read_text())
    args = deepcopy(trace['execution']['attempts'][-1]['arguments'])
    args.pop('answer_unit'); digits = args['calculation'].pop('decimals')
    args['format'] = display(decimals=digits)
    ctx = LocalReads('')
    for table, page in zip(args['tables'], trace['task']['inputs']['source_pages']):
        rid = ctx.context.register('page:'+page['page'], {'data': {'text': page['text']}})
        table['source_ref'] = {'result_id': rid, 'path': ['data', 'text']}
    if uid == 'UID0003': args['tables'][0]['rows'] = [r.rstrip('.') for r in args['tables'][0]['rows']]
    module = load(ROOT/'programs/table_calculate.py', 'v5_saved')
    out = module.run(ctx, args)
    if expected:
        assert out['status'] == 'ok', out
        assert out['outputs']['answer'] == expected
    else:
        assert out['status'] == 'needs_input', out
        assert out['recovery']['code'] == 'header_conflict'
        assert out['outputs'] == {}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_statistics_explicit_conventions_period_alignment_and_no_double_scaling():
    module = load(ROOT/'programs/table_calculate.py', 'v5_stats')
    text = 'In widgets\n| Period | Value | Other |\n| --- | --- | --- |\n'+''.join(
        f'| p{i} | {a} | {b} |\n' for i, (a,b) in enumerate(zip([10,12,16,18], [1,3,5,7])))
    ctx = LocalReads(text)
    args = table_args(ctx.ref); args['tables'][0]['rows'] = ['p0','p1','p2','p3']
    args['format'] = display(decimals=4)
    args['calculation'] = {'method': 'statistics', 'operation': 'z_score', 'transform': 'first_difference', 'ddof': 0, 'observation': 5}
    pop = module.run(ctx, args)
    args['calculation']['ddof'] = 1; sample = module.run(ctx, args)
    assert pop['outputs']['answer'] == '2.4749' and sample['outputs']['answer'] == '2.0207'
    del args['calculation']['ddof']
    assert module.run(ctx, args)['status'] == 'needs_input'
    args['calculation'] = {'method': 'statistics', 'operation': 'mean_spread', 'transform': 'none'}
    args['tables'].append(dict(deepcopy(args['tables'][0]), key='other', column='Other'))
    args['bindings']['other_values'] = ['other']
    assert module.run(ctx, args)['outputs']['answer'] == '10.0000'
    args['tables'][1]['rows'].reverse()
    assert module.run(ctx, args)['status'] == 'needs_input'
    ratio = table_args(ctx.ref)
    ratio['tables'][0]['rows'] = ['p0']
    ratio['tables'].append(dict(ratio['tables'][0], key='other', rows=['p1']))
    ratio['bindings'] = {'base_value': ['series'], 'new_value': ['other']}
    ratio['calculation'] = {'method': 'series_metrics', 'operation': 'percent_change', 'output_percent': True}
    ratio['format'] = display('percent', scale='fraction_to_percent')
    assert module.run(ctx, ratio)['status'] == 'needs_input'
    cv = table_args(ctx.ref)
    cv['tables'][0]['rows'] = ['p0','p1','p2','p3']
    cv['calculation'] = {'method':'statistics','operation':'coefficient_variation','transform':'none','ddof':0}
    cv['format'] = display('percent', scale='fraction_to_percent')
    assert module.run(ctx, cv)['outputs']['answer'] == '22.59%'
    poly_context = LocalReads(polynomial_text())
    poly = polynomial_args(poly_context.ref)
    poly.pop('answer_unit'); poly['format'] = display(decimals=3); poly['calculation'].pop('decimals')
    assert module.run(poly_context, poly)['outputs']['answer'] == '6.000'


@pytest.mark.parametrize('kind,expected', [('computed_scalar', '19.96%'), ('computed_list', '[19.96%, 0.749]')])
def test_docker_computed_result_to_exact_submission(setup, monkeypatch, kind, expected):
    cfg, bridge, bundle = setup; task = v2.fixture_task(bridge); seen = []
    program = next(p for p in bundle['program'] if p['source_file'].endswith(kind+'.py'))
    def post(*a, **kw):
        wire = kw['json']; seen.append(deepcopy(wire))
        material = json.loads(wire['messages'][1]['content'])
        if len(seen) == 1:
            actions = [{'action':'call_tool','name':'execute_python',
                        'arguments':{'source':'import json\nprint(json.dumps({"value":19.96,"other":0.749}))'}}]
        else:
            assert len(seen) == 2, 'No LLM rewrite after final Program'
            refs = material['answer_work']['numeric_result_refs']
            ref = next(r for r in refs if r['path'][-1] == 'value')
            args = dict(submit_answer=True, value_ref=ref, format=display('percent'))
            if kind == 'computed_list':
                other = next(r for r in refs if r['path'][-1] == 'other')
                args = dict(submit_answer=True, layout='brackets',
                    items=[{'value_ref':ref, 'format':display('percent')},
                           {'value_ref':other, 'format':display(decimals=3)}])
            actions = [{'action':'call_program','name':program['id'],'arguments':args}]
        return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda:response(actions))
    monkeypatch.setenv('MODEL_API_KEY', 'test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    bank_file = Path(cfg['data_dir'])/'bank.sqlite3'; before = hashlib.sha256(bank_file.read_bytes()).hexdigest()
    with bridge.bridge():
        system = EmpiricalSystem(cfg, harness=OfficeAdapter({task.task_id:{'answer':expected}}, cfg), readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try: trace = system.run_task(task, learn=False)
        finally: system.close()
    assert trace['score']['hard'], trace['execution']
    assert trace['execution']['reason'] == 'plan_submitted'
    assert trace['execution']['prediction'] == expected
    assert trace['execution']['attempts'][-1]['output_contract_status'] == 'valid'
    assert len(seen) == 2 and hashlib.sha256(bank_file.read_bytes()).hexdigest() == before
    save_evidence(kind, trace, seen)


@pytest.mark.parametrize('uid,repeat,expected', [('UID0003', 'repeat_2', '44463'), ('UID0192', 'repeat_3', None)])
def test_saved_source_to_docker_submission_or_refusal(setup, monkeypatch, uid, repeat, expected):
    cfg, bridge, bundle = setup
    saved = json.loads(next((REPEATS/repeat).glob(f'episodes/*{uid}/manual/traces/*.json')).read_text())
    task = PublicTask(**saved['task'])
    program = next(p for p in bundle['program'] if p['source_file'].endswith('/table_calculate.py'))
    seen = []
    def post(*a, **kw):
        wire = kw['json']; seen.append(deepcopy(wire)); material = json.loads(wire['messages'][1]['content'])
        if len(seen) == 1:
            args = deepcopy(saved['execution']['attempts'][-1]['arguments'])
            args.pop('answer_unit'); args['format'] = display(decimals=args['calculation'].pop('decimals'))
            for table, ref in zip(args['tables'], material['provided_source_page_refs']): table['source_ref'] = ref
            action = {'action':'call_program','name':program['id'],'arguments':args}
        else:
            assert expected is None and len(seen) == 2
            assert material['answer_work']['recent_program_refusals'][-1]['recovery']['code'] == 'header_conflict'
            action = {'action':'finish','answer':'offline refusal observed'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system = EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':expected or 'PRIVATE'}},cfg),
            readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try: trace=system.run_task(task,learn=False)
        finally: system.close()
    attempt = trace['execution']['attempts'][0]
    if expected:
        assert trace['execution']['reason'] == 'plan_submitted' and trace['execution']['prediction'] == expected
        assert attempt['status'] == 'ok' and attempt['output_contract_status'] == 'valid'
        assert len(seen) == 1
    else:
        assert attempt['status'] == 'needs_input' and not attempt['submission_by_program']
        assert trace['execution']['reason'] == 'agent_submitted'
    save_evidence(uid,trace,seen)


@pytest.mark.parametrize('arm', ['manual', 'off'])
def test_full_public_test_initial_admission_and_shared_finish(setup, monkeypatch, arm):
    cfg, bridge, bundle = setup
    from atomic_skillgraph.agents.provider import OpenAICompatibleProvider
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    original_payload = OpenAICompatibleProvider._build_payload
    sizes = []
    def measured(provider, *args, **kwargs):
        wire = original_payload(provider, *args, **kwargs)
        if wire.get('tools'):
            sizes.append(input_token_bound(wire))
        return wire
    monkeypatch.setattr(OpenAICompatibleProvider, '_build_payload', measured)
    rows = json.loads((RESOURCES/'test.json').read_text())['tasks']
    records = {row['task_id']: {'answer': 'PRIVATE_OFFLINE_FIXTURE'} for row in rows}
    seen = http(monkeypatch, [response([{'action':'finish','answer':'offline fixture'}]) for _ in rows])
    with bridge.bridge():
        for row in rows:
            system = EmpiricalSystem(cfg, harness=OfficeAdapter(records,cfg), readonly=True,
                bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
            try:
                trace = system.run_task(PublicTask(**row), learn=False)
                assert trace['execution']['reason'] == 'agent_submitted', (
                    row['task_id'], trace['execution']['reason'], trace['execution'].get('finish_trigger'), sizes[-4:])
            finally: system.close()
    assert len(seen) == 102
    for row, wire in zip(rows, seen):
        material = json.loads(wire['messages'][1]['content'])
        assert material['task']['inputs'] == row['inputs'] and wire.get('tools')
        assert len(material['calls']['programs']) == (9 if arm == 'manual' else 0)
        assert 'PRIVATE_OFFLINE_FIXTURE' not in json.dumps(wire)
    monkeypatch.setattr(Broker, 'remaining_calls', lambda b:0)
    sent = http(monkeypatch, [response(content='1.00')])
    with bridge.bridge():
        system = EmpiricalSystem(cfg, harness=OfficeAdapter(records,cfg), readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try: trace = system.run_task(PublicTask(**rows[0]), learn=False)
        finally: system.close()
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    assert sent[0]['messages'][0]['content'].count(bridge.ANSWER_CONTRACT) == 1
    assert input_token_bound(sent[0]) == trace['execution']['finish_evidence']['serialized_input_bytes']
    assert sent[0]['max_tokens'] == 32768


def test_frozen_evaluator_records_actual_bank_identity_and_ledger(setup,tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    cfg, bridge, bundle = setup
    evaluator = load(ROOT/'integration/evaluate.py', 'v5_evaluator')
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',
        lambda: {'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'})
    task = PublicTask(**json.loads((RESOURCES/'test.json').read_text())['tasks'][0])
    seen = http(monkeypatch, [response([{'action':'finish','answer':'offline_fixture'}])])
    governor = BudgetGovernor(tmp_path/'ledger.json', token_limit=216576, request_limit=2, finish_reserve=0)
    out = tmp_path/'evaluation'
    summary = evaluator.evaluate(cfg,[task],out,budget_governor=governor)
    assert summary['complete'] and len(seen) == len(governor.state['attempts']) == 1
    assert (out/'execution_manifest.json').is_file() and list((out/'requests').glob('*attempt1.json'))
    assert json.loads(seen[0]['messages'][1]['content'])['task']['inputs'] == task.inputs
    with pytest.raises(ValueError,match='Task differs'):
        evaluator.evaluate(cfg,[PublicTask(task.task_id,task.physical_key,'changed',task.inputs,task.split)],
                           tmp_path/'invalid',budget_governor=governor)
