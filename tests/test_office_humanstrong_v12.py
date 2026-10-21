"""Production evidence references, typed units, multi-arm isolation and paid-plan freezing."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from atomic_skillgraph.empirical.contracts import PublicTask,program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http,response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as fixtures

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v12'
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v11_train62_pairs_seed42_20261021')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**k:pytest.fail('Real HTTP forbidden'))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    return fixtures.setup.__wrapped__(tmp_path)


def test_bundle_and_routing():
    b=json.loads((ROOT/'bank/bank.json').read_text());m=load(ROOT/'integration/frontier.py','v12_routes')
    assert len(b['program'])==9 and len(b['workflow'])==0
    for p in b['program']:
        assert p['source']==(ROOT/p['source_file']).read_text()
        assert p['id']=='program_'+program_digest(p)
    cases=[('How much more percentage of total is E vs D, as a decimal?','share_difference'),
           ('Continuous rate of change between two dates','annual_log'),
           ('Fit a quadratic polynomial','polynomial_fit')]
    for goal,title in cases:
        selected=m.matching(goal)
        assert title in [s['title'] for s in selected]
        assert 'difference_value' not in [s['title'] for s in selected]
    for path in list(ROOT.rglob('*.py'))+[ROOT/'principles.json']:
        assert 'UID0' not in path.read_text()


def test_scale_conversion_and_unit_refusals():
    m=load(ROOT/'programs/typed_scalar.py','v12_scale')
    assert m.render('52.8928046989721','percentage_points','fraction','number',3,'percentage points')=='0.529'
    assert m.render('0.13008796','fraction','percent','percent',3,'dimensionless')=='13.009%'
    assert m.render('0.05','percentage_points','percentage_points','number',2,'percentage points')=='0.05'
    for args in [('52.893','quantity','fraction','number',3,'dollars'),
                 ('0.529','fraction','fraction','percent',3,'dimensionless'),
                 ('NaN','quantity','quantity','number',3,'dollars')]:
        with pytest.raises(ValueError):m.render(*args)


def test_all_saved_numeric_reference_failures_have_exact_readable_refs():
    from atomic_skillgraph.empirical.task_context import TaskContext
    from atomic_skillgraph.empirical.public_results import FILE_COMPUTE
    bridge=load(ROOT/'integration/frontier.py','v12_saved_refs');checked=0
    for path in (PARENT/'episodes').glob('*/manual/summary.json'):
        c=json.loads(path.read_text())['cases'][0]
        failures=[a for a in c['execution']['attempts'] if a['status']=='needs_input' and 'value_ref' in a['arguments']]
        if not failures:continue
        for a in failures:
            context=TaskContext()
            for event in c['tools'][:a['native_event_start']]:
                if event['name']=='execute_python' and event['result'].get('accepted'):
                    context.results[event['result_id']]=event['result'];context.result_origins[event['result_id']]=FILE_COMPUTE
            exported=bridge.acquired_results(context)['numeric_refs']
            for row in exported:assert context.resolve_public(row['value_ref'])==row['value']
            bad=a['arguments']['value_ref'];leaf=bad['path'][-1]
            assert any(r['value_ref']['result_id']==bad['result_id'] and r['value_ref']['path'][-1]==leaf for r in exported)
            checked+=1
    assert checked==16


@pytest.mark.parametrize('arm',['off','rules','selective','guarded'])
def test_real_native_output_refs_and_typed_submission(tmp_path,monkeypatch,arm):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    old=json.loads(next((PARENT/'episodes').glob('*_UID0061/manual/summary.json')).read_text())['cases'][0]
    task=PublicTask(**old['task']);seen=[]
    def post(*a,**kw):
        wire=kw['json'];seen.append(wire);body=json.loads(wire['messages'][1]['content'])
        if len(seen)==1:
            if arm in ('off','rules'):assert not body['calls']['programs']
            if arm=='off':assert 'human_frontier_skill_ids' not in body
            action={'action':'call_tool','name':'execute_python','arguments':{'source':
                'import json\nprint(json.dumps({"answer":{"value":(2300-499)/3405*100,"scale":"percentage_points","unit":"percentage points"}}))'}}
        else:
            assert len(seen)==2
            refs=body['acquired_result_refs'];ref=refs['quantity_refs'][0]['quantity_ref']
            assert ref['path']==['data','computed','answer']
            assert refs['numeric_refs'][0]['value_ref']['path']==['data','computed','answer','value']
            if arm=='guarded':
                program=next(p for p in bundle['program'] if p['operation']=='typed_scalar')
                assert program['id'] in {p['id'] for p in body['calls']['programs']}
                action={'action':'call_program','name':program['id'],'arguments':{'submit_answer':True,
                    'quantity_ref':ref,'format':{'output_scale':'fraction','style':'number','decimals':3}}}
            else:
                assert not any(p['operation']=='typed_scalar' and p['id'] in {c['id'] for c in body['calls']['programs']} for p in bundle['program'])
                action={'action':'finish','answer':'0.529'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','offline-key');monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'0.529'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(seen)==2
    assert trace['execution']['reason']==('plan_submitted' if arm=='guarded' else 'agent_submitted')


def test_saved_share_cells_to_direct_docker_submission(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    old=json.loads(next((PARENT/'episodes').glob('*_UID0061/manual/summary.json')).read_text())['cases'][0]
    task=PublicTask(**old['task']);program=next(p for p in bundle['program'] if p['operation']=='share_difference');seen=[]
    def post(*a,**kw):
        seen.append(kw['json']);assert len(seen)==1
        body=json.loads(kw['json']['messages'][1]['content']);ref=body['provided_source_page_refs'][0]
        header=next(s for s in task.inputs['source_pages'][0]['text'].splitlines() if s.startswith('| Series |'))
        cell=lambda label:{'source_ref':ref,'header':header,'row':label,'column':2}
        args={'submit_answer':True,'first':cell('Series E.'),'second':cell('Series D.'),'total':cell('Total, Series A-C.'),
            'absolute':False,'output_scale':'fraction','format':{'style':'number','decimals':3,'scale':'as_is'}}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([{'action':'call_program','name':program['id'],'arguments':args}]))
    monkeypatch.setenv('MODEL_API_KEY','offline-key');monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'0.529'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,'selective'))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and trace['execution']['attempts'][0]['submission_by_program']


def test_complete_four_arm_plan_and_no_gold_leakage(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v12 as runner
    runner.bind_runner();common=runner.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda _:None)
    root=tmp_path/'run';before=common.sha(PARENT/'budget.json');runner.prepare(root)
    plan=common.verify(root)
    assert len(plan['episodes'])==48 and plan['new_token_cap']==10395648
    assert plan['inherited_used_or_reserved']==173708915 and plan['prior_scores_used_for_selection']
    assert plan['test_runs_authorized_in_this_plan']==0
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in plan['episodes']])
    assert common.execute(root,'unused')==0
    tasks={t['task_id']:t for t in common.read(root/'tasks.json')['tasks']}
    for e,wire in zip(plan['episodes'],seen):
        body=json.loads(wire['messages'][1]['content'])
        assert body['task']['inputs']==tasks[e['task_id']]['inputs']
        if e['arm'] in ('off','rules'):assert not body['calls']['programs']
        if e['arm']=='off':assert 'human_frontier_skill_ids' not in body
        assert common.read(Path(e['path'])/'freeze_record.json')['unchanged']
    assert len(seen)==48 and common.sha(PARENT/'budget.json')==before
    result=runner.analysis(root)
    assert result['new_http']==48 and all(a['completed']==12 for a in result['metrics'].values())
