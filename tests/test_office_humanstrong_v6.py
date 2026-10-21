"""V6: real saved source replay, native contract lookup and frozen paired checks."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, digest, program_digest
from atomic_skillgraph.empirical.public_results import FILE_COMPUTE, PROGRAM_EXECUTION
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as previous

ROOT = Path(__file__).parents[1]/'resources/office_humanstrong_v6'
SAVED = Path('/home/yangchengyu/officeqa_humanstrong_v5_test102_pairs_seed42_20261020')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs): raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request', refuse)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(previous, 'ROOT', ROOT)
    return previous.setup.__wrapped__(tmp_path)


def test_bundle_identity_and_consistent_skills():
    bank = json.loads((ROOT/'bank/bank.json').read_text())
    parent = json.loads((ROOT.parent/'office_humanstrong_v5/bank/bank.json').read_text())
    assert bank['parent_bank_digest'] == digest(parent)
    assert len(bank['skill']) == 14 and len(bank['program']) == 9
    specs = json.loads((ROOT/'skills.json').read_text())
    for skill in bank['skill']:
        assert skill['goal'] == specs[skill['title']]['goal']
        assert skill['guidance'] == specs[skill['title']]['guidance']
    for program in bank['program']:
        assert program['source'] == (ROOT/program['source_file']).read_text()
        assert program['id'] == 'program_'+program_digest(program)
    for link in bank['implementation']:
        assert link['skill_id'] in {s['id'] for s in bank['skill']}
        assert link['program_id'] in {p['id'] for p in bank['program']}
    from jsonschema import Draft202012Validator
    calc = next(p for p in bank['program'] if p['source_file'].endswith('table_calculate.py'))
    schema = calc['input_schema']['properties']['calculation']
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors({'method':'series_metrics', 'operation':'mean'}))
    assert list(validator.iter_errors({'method':'statistics', 'operation':'mean'}))
    assert list(validator.iter_errors({'method':'statistics', 'operation':'z_score', 'transform':'none'}))


def test_existing_parser_and_arithmetic_regressions(setup):
    previous.test_rows_units_headers_and_recoverable_bounds()
    previous.test_statistics_explicit_conventions_period_alignment_and_no_double_scaling()
    previous.test_actual_saved_selectors('UID0003', 'repeat_2', '44463')
    previous.test_actual_saved_selectors('UID0192', 'repeat_3', None)


def test_missing_value_rows_stop_header_without_changing_ambiguity():
    extract = load(ROOT/'programs/table_extract.py', 'v6_extract')
    for missing in ('-', '—', '...', 'n/a', ''):
        text = f'In dollars\n| Item | 2010 | 2011 |\n| --- | --- | --- |\n| Assets | | |\n| Missing | {missing} | {missing} |\n| Total | 4 | 5 |\n'
        args = dict(text=text, rows=['Total'], column='2011', unit='dollars')
        assert extract.run(None, args)['outputs']['values'] == ['5']
        assert extract.run(None, {**args, 'rows':['Missing']})['recovery']['code'] == 'nonnumeric_cell'
        assert extract.run(None, {**args, 'text':text+'| Total | 6 | 7 |\n'})['status'] == 'needs_input'
    text = 'In dollars\n| Item | Group | Group |\n| --- | --- | --- |\n| | 2010 | 2011 |\n| A | 4 | 5 |\n'
    args = dict(text=text, rows=['A'], column='2011', unit='dollars', header_path=['Group', '2011'])
    assert extract.run(None, args)['outputs']['values'] == ['5']


def test_real_missing_rows_and_exact_named_calculation():
    path = next(SAVED.glob('episodes/*UID0125/manual/traces/*.json'))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    trace = json.loads(path.read_text())
    args = deepcopy(trace['execution']['attempts'][0]['arguments'])
    ctx = previous.LocalReads('')
    for table, page in zip(args['tables'], trace['task']['inputs']['source_pages']):
        rid = ctx.context.register(page['page'], {'data': {'text': page['text']}})
        table['source_ref'] = {'result_id':rid, 'path':['data','text']}
    module = load(ROOT/'programs/table_calculate.py', 'v6_saved')
    # Original proposal also uses an invalid method; never silently reinterpret it.
    assert module.run(ctx, args)['recovery']['code'] == 'calculation_contract'
    args['calculation'] = {'method':'series_metrics', 'operation':'mean'}
    out = module.run(ctx, args)
    assert out['status'] == 'ok', out
    assert len(out['outputs']['evidence']) == 3
    for cell in out['outputs']['evidence']:
        source = ctx.context.resolve_public(cell['source_ref'])
        assert source[cell['start']:cell['end']] == cell['quote']
        assert cell['column_path'] == [cell['column']]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_conditional_selection_keeps_all_relevant_period_constraints(setup):
    cfg, bridge, bundle = setup
    from atomic_skillgraph.empirical.bank import Bank
    task = previous.v2.fixture_task(bridge)
    task = PublicTask(task.task_id, task.physical_key,
        'Correlation of monthly Treasury interest from published revised fiscal reports between 2001 and 2003',
        task.inputs, task.split)
    bank = Bank(cfg['data_dir'], readonly=True, environment=cfg['program_environment'])
    try:
        view = bridge.HumanBankView(bank, bundle)
        names = {s['title'] for s in view.selected(task)}
        assert {'OA-Correlation','OA-Periods','OA-Fiscal','OA-Revision','OA-Treasury'} <= names
        assert 'OA-Polynomial' not in names and 'OA-Commit' not in names and 'OA-Fallback' not in names
    finally: bank.close()


def test_work_status_does_not_promote_inputs_or_stale_failures(setup):
    _, bridge, _ = setup
    context = TaskContext()
    def put(event, value, producer='p', origin=PROGRAM_EXECUTION):
        return context.register(event, value, name=producer, origin=origin)
    put('a', {'status':'needs_input','recovery':{'code':'unit_missing'}})
    assert bridge.work_status(context)['recent_program_refusals']
    put('b', {'accepted':True,'data':{'computed':{'inputs':{'price':17},'count':12,'values':[4,5]}}},
        origin=FILE_COMPUTE)
    assert not bridge.work_status(context)['numeric_result_refs']
    assert bridge.work_status(context)['recent_program_refusals']
    put('c', {'status':'ok','outputs':{'value':20,'count':12}})
    state = bridge.work_status(context)
    assert not state['recent_program_refusals']
    assert state['numeric_result_refs'] == [{'result_id':context.sources['c'],'path':['outputs','value']}]
    assert state['task_completion'] == 'not_inferred'
    put('d', {'status':'needs_input'}, producer='another')
    assert bridge.work_status(context)['recent_program_refusals']
    put('e', {'accepted':True,'data':{'computed':{'mean':2.5,'inputs':{'a':8}}}}, origin=FILE_COMPUTE)
    assert not bridge.work_status(context)['recent_program_refusals']


@pytest.mark.parametrize('kind,expected', [('computed_scalar','19.96%'), ('computed_list','[19.96%, 0.749]')])
def test_real_docker_submission_preserved(setup, monkeypatch, kind, expected):
    previous.test_docker_computed_result_to_exact_submission(setup, monkeypatch, kind, expected)


@pytest.mark.parametrize('arm', ['off','manual'])
def test_full_public_input_admission_and_finish(setup, monkeypatch, arm):
    previous.test_full_public_test_initial_admission_and_shared_finish(setup, monkeypatch, arm)


def test_native_on_demand_schema_lookup_and_docker(setup, monkeypatch):
    cfg, bridge, bundle = setup
    task = previous.v2.fixture_task(bridge)
    program = next(p for p in bundle['program'] if p['source_file'].endswith('table_extract.py'))
    seen = []
    def post(*a, **kw):
        wire = kw['json']; seen.append(deepcopy(wire))
        body = json.loads(wire['messages'][1]['content'])
        cards = body['calls']['programs']
        card = next(c for c in cards if c['id'] == program['id'])
        assert len(cards) == 9
        assert 'input_schema' not in card
        if len(seen) == 1:
            action = {'action':'call_tool','name':'read_result','arguments':card['schema_ref']}
        elif len(seen) == 2:
            action = {'action':'call_program','name':program['id'], 'arguments':{
                'text':'In widgets\n| Period | Value |\n| --- | --- |\n| June | 42 |\n',
                'rows':['June'],'column':'Value','unit':'widgets'}}
        else:
            assert len(seen) == 3
            action = {'action':'finish','answer':'42'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system = EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'42'}},cfg),
            readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try: trace = system.run_task(task,learn=False)
        finally: system.close()
    assert trace['score']['hard'] and len(seen) == 3
    assert trace['execution']['attempts'][0]['status'] == 'ok'
    assert any(p['schema_ref'] for p in json.loads(seen[0]['messages'][1]['content'])['calls']['programs'])


def test_same_saved_public_input_wire_cost(tmp_path, monkeypatch):
    from test_cf2_contracts import http
    task = PublicTask(**json.loads(next(SAVED.glob('episodes/*UID0125/manual/traces/*.json')).read_text())['task'])
    wires = {}
    for version in ('v5','v6'):
        monkeypatch.setattr(previous,'ROOT',ROOT.parent/('office_humanstrong_'+version))
        cfg, bridge, bundle = previous.setup.__wrapped__(tmp_path/version)
        seen = http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}])])
        with bridge.bridge():
            system = EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'PRIVATE_OFFLINE'}},cfg),
                readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
            try: system.run_task(task,learn=False)
            finally: system.close()
        assert len(seen) == 1
        wire = seen[0]
        assert json.loads(wire['messages'][1]['content'])['task']['inputs'] == task.inputs
        assert 'PRIVATE_OFFLINE' not in json.dumps(wire)
        wires[version] = wire
    size = lambda wire: len(json.dumps(wire,ensure_ascii=False,separators=(',',':')).encode())
    assert size(wires['v6']) < size(wires['v5'])
    if os.environ.get('V6_EVIDENCE_DIR'):
        root = Path(os.environ['V6_EVIDENCE_DIR']); root.mkdir(parents=True,exist_ok=True)
        (root/'same_input_wire.json').write_text(json.dumps({
            'fixture_model_responses':True,'actual_model_http':0,'task_id':task.task_id,
            'wire_bytes':{k:size(v) for k,v in wires.items()},'wires':wires},ensure_ascii=False,indent=2))
