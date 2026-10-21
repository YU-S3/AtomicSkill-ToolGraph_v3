"""V7 regressions from immutable V6 records; intercepted HTTP, real native/Docker paths."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, digest, program_digest
from atomic_skillgraph.empirical.public_results import FILE_COMPUTE
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http, response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as previous

ROOT = Path(__file__).parents[1]/'resources/office_humanstrong_v7'
SAVED = Path('/home/yangchengyu/officeqa_humanstrong_v6_val24_pairs_seed42_20261021')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **kw): raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request', refuse)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(previous, 'ROOT', ROOT)
    return previous.setup.__wrapped__(tmp_path)


def saved(uid):
    return json.loads(next(SAVED.glob('episodes/*'+uid+'/manual/traces/*.json')).read_text())


def replay_context(trace):
    context = TaskContext()
    for index, event in enumerate(trace['tools']):
        context.register(str(index), deepcopy(event['result']), name=event['name'],
            arguments=event['arguments'], origin=event.get('result_protocol'))
    return context


def evidence(name, value):
    if os.environ.get('V7_EVIDENCE_DIR'):
        root = Path(os.environ['V7_EVIDENCE_DIR']); root.mkdir(parents=True, exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def test_identity_and_unmodified_parent():
    bank = json.loads((ROOT/'bank/bank.json').read_text())
    parent = json.loads((ROOT.parent/'office_humanstrong_v6/bank/bank.json').read_text())
    assert bank['parent_bank_digest'] == digest(parent)
    assert bank['measured_performance'] is None
    assert [len(bank[k]) for k in ('skill','program','implementation','workflow')] == [14,9,9,4]
    for program in bank['program']:
        assert program['source'] == (ROOT/program['source_file']).read_text()
        assert program['id'] == 'program_'+program_digest(program)
    skill_ids, program_ids = {s['id'] for s in bank['skill']}, {p['id'] for p in bank['program']}
    assert all(i['skill_id'] in skill_ids and i['program_id'] in program_ids for i in bank['implementation'])


@pytest.mark.parametrize('uid', ['UID0024','UID0233','UID0213'])
def test_saved_retrieval_is_not_a_computed_answer(uid):
    trace = saved(uid); context = replay_context(trace)
    bridge = load(ROOT/'integration/frontier.py', 'v7_status')
    old = load(ROOT.parent/'office_humanstrong_v6/integration/frontier.py', 'v6_status')
    state = bridge.work_status(context)
    assert not state['numeric_result_refs']
    assert not bridge.recent_scalar_result_refs(context)
    if uid != 'UID0213': assert old.work_status(context)['numeric_result_refs']
    else: assert state['recent_program_refusals'][0]['recovery']['code'] == 'python_execution_failed'
    evidence(uid+'_status', {'actual_model_http':0, 'before':old.work_status(context), 'after':state})


def test_actual_successful_computation_remains_candidate():
    bridge = load(ROOT/'integration/frontier.py', 'v7_success')
    context = replay_context(saved('UID0144'))
    refs = bridge.work_status(context)['numeric_result_refs']
    assert any(context.resolve_public(r) == 263 for r in refs)
    assert not any('values' in r['path'] or 'n' in r['path'] for r in refs)
    assert bridge.work_status(context)['task_completion'] == 'not_inferred'


def test_original_bad_bindings_refuse_before_reads_corrected_key_succeeds():
    trace = saved('UID0047')
    module = load(ROOT/'programs/table_calculate.py', 'v7_binding')
    original_hash = digest(trace)
    receipts = []
    for attempt in trace['execution']['attempts']:
        ctx = previous.LocalReads('')
        out = module.run(ctx, deepcopy(attempt['arguments']))
        assert out['recovery']['code'] == 'unknown_or_unused_table_key'
        assert out['recovery']['unknown_keys'] == ['1981-Aug.']
        assert out['recovery']['allowed_table_keys'] == ['ifs1']
        assert ctx.calls == []
        receipts.append(out)
    args = deepcopy(trace['execution']['attempts'][0]['arguments'])
    old_ref = args['tables'][0]['source_ref']
    event = next(e for e in trace['tools'] if e.get('result_id') == old_ref['result_id'])
    ctx = previous.LocalReads(event['result']['data']['text'])
    args['tables'][0]['source_ref'] = ctx.ref
    args['bindings'] = {'values':['ifs1']}
    out = module.run(ctx, args)
    assert out['status'] == 'ok', out
    assert out['outputs']['answer'] == trace['execution']['prediction'] == '29265'
    assert digest(trace) == original_hash
    evidence('binding_replay', {'actual_model_http':0, 'original_refusals':receipts,
        'corrected_argument_fixture_not_model_success':True, 'result':out})


def test_existing_scope_and_numeric_guards(setup, monkeypatch):
    # Parent test selects ROOT's generated calculator; retain its hard source guards.
    original = previous.load
    monkeypatch.setattr(previous, 'load', lambda path, name: original(
        ROOT.parent/'office_humanstrong_v6/programs/table_extract.py'
        if path == ROOT/'programs/table_extract.py' else path, name))
    previous.test_rows_units_headers_and_recoverable_bounds()
    previous.test_statistics_explicit_conventions_period_alignment_and_no_double_scaling()
    previous.test_actual_saved_selectors('UID0003', 'repeat_2', '44463')
    previous.test_actual_saved_selectors('UID0192', 'repeat_3', None)


@pytest.mark.parametrize('kind,expected', [('computed_scalar','19.96%'), ('computed_list','[19.96%, 0.749]')])
def test_docker_submission_preserved(setup, monkeypatch, kind, expected):
    previous.test_docker_computed_result_to_exact_submission(setup, monkeypatch, kind, expected)


def test_schema_lookup_remains_native(setup, monkeypatch):
    cfg, bridge, bundle=setup
    task=previous.v2.fixture_task(bridge)
    program=next(p for p in bundle['program'] if p['source_file'].endswith('table_extract.py'))
    seen=[]
    def post(*a, **kw):
        wire=kw['json']; seen.append(deepcopy(wire))
        body=json.loads(wire['messages'][1]['content'])
        card=next(c for c in body['calls']['programs'] if c['id']==program['id'])
        if len(seen)==1:
            action={'action':'call_tool','name':'read_result','arguments':card['schema_ref']}
        elif len(seen)==2:
            assert 'header_path' in json.dumps(body)
            action={'action':'call_program','name':program['id'],'arguments':{
                'text':'In widgets\n| Period | Value |\n| --- | --- |\n| June | 42 |\n',
                'rows':['June'],'column':'Value','unit':'widgets'}}
        else:
            assert len(seen)==3
            action={'action':'finish','answer':'42'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'42'}},cfg),
            readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    schema_read=next(e for e in trace['tools'] if e['name']=='read_result')['result']
    assert schema_read['accepted'] and schema_read['next_offset'] is None
    assert json.loads(schema_read['data']) == program['input_schema']
    assert trace['execution']['attempts'][0]['status']=='ok'
    assert trace['score']['hard'] and len(seen)==3
    evidence('native_schema_docker', {'actual_model_http':0,'fixture_model_decisions':True,
        'trace':trace,'sent':seen})


def test_every_advertised_contract_can_be_read_losslessly(setup):
    from atomic_skillgraph.empirical.bank import Bank
    cfg,bridge,bundle=setup
    bank=Bank(cfg['data_dir'],readonly=True,environment=cfg['program_environment'])
    try:
        view=bridge.HumanBankView(bank,bundle)
        view.bind_task(previous.v2.fixture_task(bridge))
        context=TaskContext();view.bind_context(context)
        for card in view.program_options():
            parts=[];offset=0
            while True:
                result=context.read(**card['schema_ref'],offset=offset,limit=500)
                assert result['accepted'] and isinstance(result['data'],str)
                parts.append(result['data'])
                if result['next_offset'] is None:break
                offset=result['next_offset']
            assert json.loads(''.join(parts))==bank.get(card['id'])['input_schema']
        assert len(view.program_options())==9
    finally:bank.close()


@pytest.mark.parametrize('arm', ['manual','off'])
def test_full_public_input_and_finish_admission(setup, monkeypatch, arm):
    previous.test_full_public_test_initial_admission_and_shared_finish(setup, monkeypatch, arm)


def test_evaluator_frozen_identity_and_ledger(setup, tmp_path, monkeypatch):
    previous.test_frozen_evaluator_records_actual_bank_identity_and_ledger(setup, tmp_path, monkeypatch)


def test_raw_text_contract_and_failed_python_recovery(setup, monkeypatch):
    cfg, bridge, bundle = setup
    trace = saved('UID0213'); task = PublicTask(**trace['task'])
    source = next(e['arguments']['source'] for e in trace['tools'] if e['name']=='execute_python')
    corrected = source.replace("r['data']['text']['content']", "r['data']['text']")
    seen = []
    def post(*a, **kw):
        wire=kw['json']; seen.append(deepcopy(wire))
        body=json.loads(wire['messages'][1]['content'])
        if len(seen)==1:
            assert 'plain string' in json.dumps(body['calls'])
            action={'action':'call_tool','name':'execute_python','arguments':{'source':source}}
        elif len(seen)==2:
            assert body['answer_work']['recent_program_refusals'][0]['recovery']['code']=='python_execution_failed'
            assert 'Python type error is not missing source evidence' in wire['messages'][0]['content']
            action={'action':'call_tool','name':'execute_python','arguments':{'source':corrected}}
        else:
            assert len(seen)==3
            action={'action':'finish','answer':'offline fixture'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'PRIVATE_OFFLINE'}},cfg),
            readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try: replay=system.run_task(task,learn=False)
        finally:system.close()
    events=[e for e in replay['tools'] if e['name']=='execute_python']
    assert len(events)==2
    assert not events[0]['result']['accepted']
    assert "string indices" in events[0]['result']['error']
    assert events[1]['result']['accepted'], events[1]['result']
    assert events[1]['result']['data']['computed']['answer_1dp'] == -550.3
    assert not bridge.work_status(replay_context(replay))['recent_program_refusals']
    evidence('raw_text_docker_replay', {'actual_model_http':0,'fixture_model_decisions':True,
        'corrected_accessor_only_not_verified_semantics':True,'trace':replay,'sent':seen})


def test_same_input_wire_reduction_and_off_is_empty(tmp_path, monkeypatch):
    task = PublicTask(**saved('UID0024')['task'])
    wires={}
    for version, arm in [('v6','manual'),('v7','manual'),('v7','off')]:
        monkeypatch.setattr(previous,'ROOT',ROOT.parent/('office_humanstrong_'+version))
        cfg, bridge, bundle=previous.setup.__wrapped__(tmp_path/(version+arm))
        seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}])])
        with bridge.bridge():
            system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'PRIVATE_OFFLINE'}},cfg),
                readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
            try:system.run_task(task,learn=False)
            finally:system.close()
        assert len(seen)==1
        body=json.loads(seen[0]['messages'][1]['content'])
        assert body['task']['inputs']==task.inputs
        assert 'PRIVATE_OFFLINE' not in json.dumps(seen)
        if arm=='off':
            assert not body['calls']['programs']
            assert not body.get('human_frontier_skill_ids')
            assert not body.get('answer_work')
        elif version=='v7':
            assert len(body['calls']['programs'])==9
            assert all(c.get('schema_ref') and 'input_schema' not in c for c in body['calls']['programs'])
        wires[version+'_'+arm]=seen[0]
    size=lambda w:len(json.dumps(w,ensure_ascii=False,separators=(',',':')).encode())
    assert size(wires['v7_manual']) < size(wires['v6_manual'])
    evidence('same_input_wire', {'actual_model_http':0, 'fixture_model_decisions':True,
        'wire_bytes':{k:size(v) for k,v in wires.items()},'wires':wires})
