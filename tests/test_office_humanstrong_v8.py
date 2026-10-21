"""Sparse Bank boundaries, real Docker execution and paired-run isolation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from atomic_skillgraph.empirical.contracts import PublicTask, digest, program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http, response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as previous

SOURCE = Path(__file__).parents[1]
ROOT = SOURCE/'resources/office_humanstrong_v8'
PARENT = Path('/home/yangchengyu/officeqa_humanstrong_v7_val24_pairs_seed42_20261021')
RESOURCES = Path('/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **kw): raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request', refuse)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(previous, 'ROOT', ROOT)
    return previous.setup.__wrapped__(tmp_path)


def test_small_bank_and_public_applicability():
    b = json.loads((ROOT/'bank/bank.json').read_text())
    parent = json.loads((ROOT.parent/'office_humanstrong_v7/bank/bank.json').read_text())
    assert b['parent_bank_digest'] == digest(parent) and b['measured_performance'] is None
    assert [len(b[k]) for k in ('skill','program','implementation','workflow')] == [6,4,4,0]
    assert max(len(s['guidance']) for s in b['skill']) <= 200
    assert sum(len(s['guidance']) for s in b['skill']) < sum(len(s['guidance']) for s in parent['skill'])/3
    for p in b['program']:
        assert p['id'] == 'program_'+program_digest(p)
        assert p['source'] == (ROOT/p['source_file']).read_text()
    bridge = load(ROOT/'integration/frontier.py', 'v8_select')
    # Broad terms alone must not inject an irrelevant arithmetic convention.
    for goal in ['Total assets at the end of a calendar year.',
                 'Arc elasticity using midpoint percentage change.',
                 'A single percentage point difference and the percentage attributed to vessels.']:
        assert not bridge.matching(goal)
    assert [s['title'] for s in bridge.matching('Continuously compounded average annual growth.')] == ['annual_log']
    assert [s['title'] for s in bridge.matching('Change in share of the total.')] == ['share_change']
    assert all('UID0' not in p.read_text() for p in ROOT.rglob('*.py'))


@pytest.mark.parametrize('operation,arguments,goal,expected', [
    ('annual_log_growth',dict(base_value=100,new_value=200,periods=10,output_percent=False,decimals=3),
     'Continuously compounded annual growth from 100 to 200 over ten years.', '0.069'),
    ('cagr',dict(base_value=100,new_value=121,periods=2,output_percent=False,decimals=2),
     'Compute CAGR from 100 to 121 over two years.', '0.10'),
    ('share_difference',dict(values=[2,10,3,10],absolute=True,output_percent=True,decimals=2),
     'Absolute change in share from 2/10 to 3/10.', '10.00'),
    ('symmetric_difference',dict(values=[3,5],output_percent=False,decimals=2),
     'Absolute midpoint-normalized difference of 3 and 5.', '0.50'),
])
def test_real_docker_narrow_calculators(setup, monkeypatch, operation, arguments, goal, expected):
    cfg, bridge, bundle = setup
    task = previous.v2.fixture_task(bridge)
    task = PublicTask(**{**task.__dict__, 'goal':goal})
    program = next(p for p in bundle['program'] if p['operation'] == operation)
    seen=[]
    def post(*a, **kw):
        seen.append(deepcopy(kw['json']))
        body=json.loads(seen[-1]['messages'][1]['content'])
        assert not any(k in body for k in ['answer_work','recent_scalar_result_refs','human_office_progress'])
        if len(seen)==1:
            assert [c['id'] for c in body['calls']['programs']] == [program['id']]
            action={'action':'call_program','name':program['id'],
                    'arguments':dict(operation=operation,unit='as_requested',**arguments)}
        else:
            assert len(seen)==2
            action={'action':'finish','answer':expected}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':expected}},cfg),
            readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try: trace=system.run_task(task,learn=False)
        finally: system.close()
    attempt=trace['execution']['attempts'][0]
    assert attempt['status']=='ok',attempt
    assert trace['score']['hard'] and len(seen)==2


def test_full_paired_runner_and_no_guidance_on_controls(tmp_path,monkeypatch):
    sys.path.insert(0,str(SOURCE/'experiments'))
    import run_office_humanstrong_v8 as launcher
    launcher.bind_runner(); common=launcher.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda _:None)
    from filelock import FileLock
    lock_root=tmp_path/'locks';lock_root.mkdir()
    monkeypatch.setattr(common,'FileLock',lambda path,timeout=0:FileLock(str(lock_root/(digest(str(path))+'.lock')),timeout=timeout))
    root=tmp_path/'pairs'; parent_hash=common.sha(PARENT/'budget.json')
    launcher.prepare(root,PARENT,RESOURCES,PARENT)
    plan=common.verify(root)
    assert len(plan['episodes'])==30 and plan['new_token_cap']==6497280
    assert plan['inherited_used_or_reserved']==163443271
    assert plan['ledger_limits']['token_limit']==169940551
    assert plan['development_exposed'] and not plan['prior_scores_used_for_selection']
    assert plan['test_runs_authorized_in_this_plan']==2
    tasks={t['task_id']:t for t in common.read(root/'tasks.json')['tasks']}
    assert len(tasks)==15 and all(t['split']=='test' for t in tasks.values())
    cfg=common.read(root/'base_config.json'); before=common.read(PARENT/'base_config.json')
    for key in ['llm','runtime','program_environment']: assert cfg[key]==before[key]
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in plan['episodes']])
    assert common.execute(root,'unused')==0
    for episode,wire in zip(plan['episodes'],seen):
        task=tasks[episode['task_id']]; body=json.loads(wire['messages'][1]['content'])
        assert body['task']['inputs']==task['inputs']
        assert not any(k in body for k in ['answer_work','recent_scalar_result_refs','human_office_progress'])
        active=common.frontier.matching(task['goal']) if episode['arm']=='manual' else []
        assert len(body.get('human_frontier_skill_ids',[]))==len(active)
        assert len(body['calls']['programs'])==sum('operation' in s for s in active)
        assert len(active)<=2
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
    assert common.sha(PARENT/'budget.json')==parent_hash
    assert common.status(root)['new_http']==30
    common.verify(root)
