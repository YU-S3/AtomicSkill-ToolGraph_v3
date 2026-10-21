"""Scope errors from real development traces, production Docker and run isolation."""
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
import test_office_humanstrong_v5 as fixtures

SOURCE=Path(__file__).parents[1]
ROOT=SOURCE/'resources/office_humanstrong_v10'
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v9_targeted_test5_pairs_seed42_20261021')
RESOURCES=Path('/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


def test_full_public_scope_and_content_identity():
    bridge=load(ROOT/'integration/frontier.py','v10_scope')
    bank=json.loads((ROOT/'bank/bank.json').read_text())
    assert [len(bank[k]) for k in ('skill','program','implementation','workflow')]==[7,4,4,0]
    assert max(len(s['guidance']) for s in bank['skill'])<=200
    for p in bank['program']:
        assert p['id']=='program_'+program_digest(p)
        assert p['source']==(ROOT/p['source_file']).read_text()
    assert not bridge.matching('Fit a degree 2 polynomial OLS regression')
    assert not bridge.matching('Read a single currency balance')
    for goal in ('Nominal values, not inflation-adjusted.','Values not adjusted for inflation.',
                 'Return non-inflation-adjusted yields.'):
        assert not bridge.matching(goal)
    assert [s['title'] for s in bridge.matching('Read nominal values, not inflation-adjusted, then convert with CPI-U.')]==['price_basis']
    assert [s['title'] for s in bridge.matching('Continuously compounded annual growth')]==['annual_log']
    counts={}
    for split,count in [('train',120),('test',102)]:
        tasks=json.loads((RESOURCES/(split+'.json')).read_text())['tasks']
        assert len(tasks)==count
        cards=[bridge.matching(t['goal'],t['inputs']) for t in tasks]
        assert max(map(len,cards))<=2
        counts[split]=sum(bool(c) for c in cards)
        for t,c in zip(tasks,cards):
            renamed={**t,'task_id':'unrelated:123','gold':'ignored private field'}
            assert bridge.matching(renamed['goal'],renamed['inputs'])==c
    assert counts=={'train':51,'test':36}
    for path in list(ROOT.rglob('*.py'))+[ROOT/'principles.json']:
        assert 'UID0' not in path.read_text()


def arguments(statistic='mean',**extra):
    return dict(statistic=statistic,requested_periods=['Jan','Feb','Mar'],
        records=[{'period':p,'value':v,'x':x} for p,v,x in [('Jan',2,1),('Feb',4,2),('Mar',6,3),('Apr',999,4)]],
        decimals=3,unit='widgets',**extra)


@pytest.mark.parametrize('statistic,extra,expected',[
    ('mean',{},'4.000'),('variance_sample',{},'4.000'),('variance_population',{},'2.667'),
    ('correlation',{},'1.000'),('ols_slope',{},'2.000'),('ols_intercept',{},'0.000'),
    ('forecast_minus_actual',{'forecast_x':4,'actual':9},'-1.000'),
    ('actual_minus_forecast',{'forecast_x':4,'actual':9},'1.000')])
def test_named_scope_math(statistic,extra,expected):
    module=load(ROOT/'programs/scoped_statistics.py','scoped_math')
    out=module.run(None,arguments(statistic,**extra))
    assert out['status']=='ok',out
    assert out['outputs']['answer']==expected
    assert out['outputs']['selected_periods']==['Jan','Feb','Mar']
    assert out['outputs']['excluded_periods']==['Apr']


def test_missing_duplicate_coordinates_and_finiteness_refused():
    module=load(ROOT/'programs/scoped_statistics.py','scoped_refusals')
    cases=[]
    a=arguments();a['records'].pop(0);cases.append(a)
    a=arguments();a['records'].append(a['records'][0]);cases.append(a)
    a=arguments('ols_intercept');a['records'][0].pop('x');cases.append(a)
    a=arguments();a['records'][0]['value']='NaN';cases.append(a)
    cases.append(arguments('actual_minus_forecast',forecast_x=2,actual=9))
    for a in cases:assert module.run(None,a)['status']=='needs_input'
    # Translating x changes intercept, never slope; coordinates are not guessed.
    a=arguments('ols_intercept')
    for r in a['records']:r['x']+=2000
    assert module.run(None,a)['outputs']['answer']=='-4000.000'


def test_production_program_with_irrelevant_extra_period(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    cfg,bridge,bundle=fixtures.setup.__wrapped__(tmp_path)
    task=fixtures.v2.fixture_task(bridge)
    task=PublicTask(**{**task.__dict__,'goal':'Arithmetic mean for Jan, Feb and Mar; three decimals.',
        'inputs':{**task.inputs,'source_pages':[]}})
    p=next(p for p in bundle['program'] if p['operation']=='scoped_statistics')
    sent=[]
    def post(*a,**kw):
        sent.append(deepcopy(kw['json']))
        body=json.loads(sent[-1]['messages'][1]['content'])
        if len(sent)==1:
            assert [x['id'] for x in body['calls']['programs']]==[p['id']]
            action={'action':'call_program','name':p['id'],'arguments':arguments()}
        else:action={'action':'finish','answer':'4.000'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'4.000'}},cfg),
            readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['execution']['attempts'][0]['status']=='ok',trace['execution']['attempts']
    assert trace['score']['hard'] and len(sent)==2


@pytest.mark.parametrize('scope',['train12','full'])
def test_train_probe_runner_and_full_splits_budget_refusal(tmp_path,monkeypatch,scope):
    sys.path.insert(0,str(SOURCE/'experiments'))
    import run_office_humanstrong_v10 as launcher
    launcher.bind_runner();common=launcher.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda _:None)
    from filelock import FileLock
    locks=tmp_path/'locks';locks.mkdir()
    monkeypatch.setattr(common,'FileLock',lambda p,timeout=0:FileLock(str(locks/(digest(str(p))+'.lock')),timeout=timeout))
    monkeypatch.setattr(launcher.paired,'CUMULATIVE_CAP',190000000)
    for split_scope in ('train120','test102'):
        with pytest.raises(ValueError,match='Insufficient cumulative authority'):
            launcher.prepare(tmp_path/split_scope,scope=split_scope)
        assert not (tmp_path/split_scope).exists()
    monkeypatch.setattr(launcher.paired,'CUMULATIVE_CAP',262000000)
    root=tmp_path/'probe';parent_sha=common.sha(PARENT/'budget.json')
    if scope=='full':
        launcher.prepare_all(root)
        campaign=launcher.verify_campaign(root)
        assert campaign['new_token_cap']==96159744 and campaign['expected_episodes']==444
        assert campaign['inherited_used_or_reserved']+campaign['new_token_cap']==261255153
        seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in range(444)])
        assert launcher.run_all(root,'unused')==0
        status=launcher.campaign_status(root)
        assert status['completed_episodes']==444 and status['new_http']==444 and status['new_unsettled']==0
        assert status['new_tokens_used_or_reserved']==sum(s['new_tokens_used_or_reserved'] for s in status['splits'].values())
        assert common.read(root/'test102/plan.json')['base_run']==str(root/'train120')
        assert common.read(root/'test102/budget_prefix.json')==common.read(root/'train120/budget.json')
        assert common.sha(PARENT/'budget.json')==parent_sha
        for split,name in [('train','train120'),('test','test102')]:
            plan=common.verify(root/name)
            tasks=common.read(root/name/'tasks.json')['tasks']
            original=common.read(RESOURCES/(split+'.json'))['tasks']
            assert tasks==original
            for e in plan['episodes']:
                assert common.read(Path(e['path'])/'freeze_record.json')['unchanged']
        return
    launcher.prepare(root);plan=common.verify(root)
    assert len(plan['episodes'])==24 and plan['new_token_cap']==5197824
    assert plan['inherited_used_or_reserved']==165095409
    assert plan['test_runs_authorized_in_this_plan']==0
    assert not plan['prior_scores_used_for_selection']
    tasks={t['task_id']:t for t in common.read(root/'tasks.json')['tasks']}
    assert len(tasks)==12 and all(t['split']=='train' for t in tasks.values())
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in plan['episodes']])
    assert common.execute(root,'unused')==0
    for e,wire in zip(plan['episodes'],seen):
        body=json.loads(wire['messages'][1]['content']);task=tasks[e['task_id']]
        assert body['task']['inputs']==task['inputs']
        active=common.frontier.matching(task['goal'],task['inputs']) if e['arm']=='manual' else []
        assert len(body.get('human_frontier_skill_ids',[]))==len(active)
        assert common.read(Path(e['path'])/'freeze_record.json')['unchanged']
        if e['arm']=='off':assert not body['calls']['programs']
    assert common.sha(PARENT/'budget.json')==parent_sha
    assert common.status(root)['new_http']==24
