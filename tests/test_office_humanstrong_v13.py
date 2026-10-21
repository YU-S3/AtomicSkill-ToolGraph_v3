"""Actual sandbox finalization and identical public inputs across display directions."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from atomic_skillgraph.empirical.contracts import PublicTask,program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http,response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as fixtures

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v13'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**kw:pytest.fail('Real HTTP forbidden'))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    return fixtures.setup.__wrapped__(tmp_path)


def test_bank_sources_and_precision_scope():
    bundle=json.loads((ROOT/'bank/bank.json').read_text())
    assert len(bundle['program'])==1 and not bundle['workflow']
    for program in bundle['program']:
        assert program['id']=='program_'+program_digest(program)
        assert program['source']==(ROOT/program['source_file']).read_text()
    bridge=load(ROOT/'integration/frontier.py','v13_scope')
    assert bridge.precision_needed('Report to the nearest thousandth')
    assert not bridge.precision_needed('Which country held the largest amount?')
    for path in [*ROOT.rglob('*.py'),ROOT/'precision.json']:assert 'UID0' not in path.read_text()


@pytest.mark.parametrize('arm',['off','rules','precise','display'])
def test_untyped_computation_to_exact_final_answer(tmp_path,monkeypatch,arm):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    old=fixtures.v2.fixture_task(bridge)
    task=PublicTask(**{**old.__dict__,'goal':'Report the computed amount to three decimal places.',
                      'inputs':{**old.inputs,'source_pages':[]}})
    seen=[]
    def post(*args,**kwargs):
        wire=kwargs['json'];seen.append(wire);body=json.loads(wire['messages'][1]['content'])
        if len(seen)==1:
            assert not body['calls']['programs']
            action={'action':'call_tool','name':'execute_python','arguments':{'source':'import json\nprint(json.dumps({"final_amount":1.6}))'}}
        else:
            assert len(seen)==2
            ref=body['acquired_result_refs']['numeric_refs'][0]['value_ref']
            assert ref['path']==['data','computed','final_amount']
            if arm=='display':
                program=bundle['program'][0]
                assert [p['id'] for p in body['calls']['programs']]==[program['id']]
                action={'action':'call_program','name':program['id'],'arguments':{'submit_answer':True,
                    'value_ref':ref,'source_scale':'quantity','unit':'widgets',
                    'format':{'output_scale':'quantity','style':'number','decimals':3}}}
            else:
                assert not body['calls']['programs']
                action={'action':'finish','answer':'1.600'}
        if arm=='off':assert 'human_frontier_skill_ids' not in body
        assert (bridge.PRECISION['guidance'] in wire['messages'][0]['content'])==(arm in ('precise','display'))
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','offline-key');monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'1.600'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(seen)==2
    assert trace['execution']['reason']==('plan_submitted' if arm=='display' else 'agent_submitted')


@pytest.mark.parametrize('arm',['off','rules','precise','display'])
def test_finish_precision_is_present_during_capacity_planning(tmp_path,monkeypatch,arm):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    old=fixtures.v2.fixture_task(bridge)
    task=PublicTask(**{**old.__dict__,'goal':'Round the amount to the nearest thousandth.',
                      'inputs':{**old.inputs,'source_pages':[]}})
    adapter=OfficeAdapter({},cfg);adapter.task=task
    seen=http(monkeypatch,[response(content='1.600',finish='stop')])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        system.task_context=TaskContext(cfg['runtime']);system.phase='test'
        system.task_context.reference('1.600','computed_final')
        try:
            prepared=system.prepare_finish({'original_goal':task.goal},'Return final answer.')
            system.agent('runtime','Return final answer.',prepared['materials'],None,None,decision_purpose='finish_only')
            wire=seen[0]
            assert wire['messages'][0]['content'].count(bridge.PRECISION['guidance'])==int(arm in ('precise','display'))
            assert len(json.dumps(wire,ensure_ascii=False,separators=(',',':')).encode())<=cfg['runtime']['finish_input_max_bytes']
            assert not prepared['required_evidence_missing']
        finally:system.close()


def test_new_direction_cannot_replace_a_better_short_rule_control():
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    from run_office_humanstrong_v13 import choose
    result={'status':{'status':'completed'},'new_unsettled':0,
        'metrics':{a:{'completed':12,'correct':n,'tokens':cost} for a,n,cost in
                   [('off',6,100),('rules',9,140),('precise',9,120),('display',10,200)]},
        'vs_off':{a:{'net':n,'losses':[]} for a,n in [('rules',3),('precise',3),('display',4)]},
        'vs_rules':{'precise':{'losses':[]},'display':{'losses':['regression']}}}
    assert choose(result)=='precise'
    result['vs_rules']['precise']['losses']=['regression']
    assert choose(result)=='rules'


def test_complete_four_arm_billing_and_freeze(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v13 as runner
    runner.bind_runner();common=runner.common
    # A settled historical billing parent exercises the same planner without touching the live parent.
    parent='/home/yangchengyu/officeqa_humanstrong_v11_train62_pairs_seed42_20261021'
    monkeypatch.setattr(runner,'PARENT',parent)
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda _:None)
    root=tmp_path/'run';before=common.sha(Path(parent)/'budget.json');runner.prepare(root)
    plan=common.verify(root)
    assert len(plan['episodes'])==48 and plan['new_token_cap']==10395648
    assert plan['comparison_arms']==['off','rules','precise','display']
    assert plan['test_runs_authorized_in_this_plan']==0
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in plan['episodes']])
    assert common.execute(root,'unused')==0
    for episode,wire in zip(plan['episodes'],seen):
        body=json.loads(wire['messages'][1]['content'])
        assert not body['calls']['programs']
        if episode['arm']=='off':assert 'human_frontier_skill_ids' not in body
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
    assert len(seen)==48 and common.sha(Path(parent)/'budget.json')==before
    assert runner.analysis(root)['new_unsettled']==0


@pytest.mark.parametrize('arm',['off','rules','precise','display'])
def test_complete_test_public_input_admission(tmp_path,monkeypatch,arm):
    import test_office_humanstrong_v12 as previous
    monkeypatch.setattr(previous,'ROOT',ROOT)
    previous.test_complete_test_public_input_admission(tmp_path,monkeypatch,arm)
