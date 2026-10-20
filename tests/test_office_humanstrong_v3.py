"""V3 authored capability, production Docker route and bounded paired runner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from atomic_skillgraph.empirical.contracts import digest, program_digest, PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http,response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v2 as v2

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v3'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual HTTP forbidden in offline acceptance')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


@pytest.fixture
def setup(tmp_path):
    base=Path('/home/yangchengyu/officeqa_humanstrong_v2_train12_pairs_seed42_20261020/base_config.json')
    resources=Path('/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')
    if not base.exists():pytest.skip('Saved immutable paired run needed')
    cfg=load(ROOT/'integration/configuration.py','v3_cfg').configure(json.loads(base.read_text()),resources,'train')
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    bridge=load(ROOT/'integration/frontier.py','v3_frontier')
    bundle=bridge.native_import(json.loads((ROOT/'bank/bank.json').read_text()),cfg['data_dir'],
                                cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    return cfg,bridge,bundle,resources


def operands():
    ref={'result_id':'synthetic-public-source','path':['data','text']}
    return {'operation':'absolute_error','degree':2,'unit':'widgets','decimals':3,'prediction_x':6,
            'required_periods':[f'period_{i}' for i in range(1,6)],
            'observations':[{'period':f'period_{i}','x':i,'value':2+3*i+4*i*i,'unit':'widgets','source_ref':ref}
                            for i in range(1,6)],
            'actual':{'value':170,'unit':'widgets','source_ref':ref}}


def test_versioned_assets_and_parent_are_consistent():
    import jsonschema,sqlite3
    bank=json.loads((ROOT/'bank/bank.json').read_text())
    parent=json.loads((ROOT.parent/'office_humanstrong_v2/bank/bank.json').read_text())
    assert bank['parent_bank_digest']==digest(parent)
    assert {k:len(bank[k]) for k in ('skill','program','implementation','workflow')}==dict(skill=12,program=6,implementation=6,workflow=4)
    assert {p['id'] for p in parent['program']} < {p['id'] for p in bank['program']}
    members={a['id'] for k in ('skill','program','implementation','workflow') for a in bank[k]}
    for p in bank['program']:
        assert p['id']=='program_'+program_digest(p)
        assert p['source']==(ROOT/p['source_file']).read_text()
        jsonschema.Draft202012Validator.check_schema(p['input_schema'])
        jsonschema.Draft202012Validator.check_schema(p['output_schema'])
    for link in bank['implementation']:
        assert link['skill_id'] in members and link['program_id'] in members
    with sqlite3.connect(f'file:{ROOT / "bank/bank.sqlite3"}?mode=ro',uri=True) as db:
        assert db.execute('select count(*) from assets').fetchone()[0]==28
    text=(ROOT/'guidance_changes.json').read_text()+(ROOT/'programs/polynomial_forecast.py').read_text()
    assert 'UID0208' not in text and '0.162' not in text and '6.53' not in text


def test_polynomial_coverage_rank_precision_and_prediction_modes():
    p=load(ROOT/'programs/polynomial_forecast.py','v3_polynomial')
    data=operands();value=p.run(None,data)
    assert value['outputs']['answer']=='6.000' and float(value['outputs']['prediction'])==164
    for change in ('missing_period','missing_actual','unit','rank','nan'):
        bad=deepcopy(data)
        if change=='missing_period':bad['observations'].pop()
        elif change=='missing_actual':bad.pop('actual')
        elif change=='unit':bad['actual']['unit']='different'
        elif change=='nan':bad['observations'][0]['value']='NaN'
        else:
            for row in bad['observations']:row['x']=1
        assert p.run(None,bad)['status']=='needs_input',change
    prediction=deepcopy(data);prediction['operation']='predict';prediction.pop('actual')
    assert p.run(None,prediction)['outputs']['answer']=='164.000'
    assert p.run(None,prediction)['outputs']['actual'] is None
    rng=np.random.default_rng(42)
    for degree in range(1,6):
        x=np.arange(20,dtype=float);y=rng.normal(size=20)+x/7
        example=deepcopy(prediction);example.update(degree=degree,prediction_x=20,
            required_periods=[str(i) for i in range(20)])
        example['observations']=[dict(example['observations'][0],period=str(i),x=int(i),value=float(y[i])) for i in range(20)]
        result=p.run(None,example)
        expected=np.polynomial.polynomial.polyval(20,np.polynomial.polynomial.polyfit(x,y,degree))
        assert abs(float(result['outputs']['prediction'])-expected)<1e-8


def test_actual_native_program_route_and_refusal(setup,monkeypatch):
    cfg,bridge,bundle,_=setup;task=v2.fixture_task(bridge)
    program=next(p for p in bundle['program'] if p['source_file']=='programs/polynomial_forecast.py')
    good=operands();bad=deepcopy(good);bad.pop('actual')
    seen=http(monkeypatch,[response([{'action':'call_program','name':program['id'],'arguments':a}]) for a in (bad,good)]
              +[response([{'action':'finish','answer':'6.000'}])])
    before=hashlib.sha256(Path(cfg['data_dir'],'bank.sqlite3').read_bytes()).hexdigest()
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({'s':{'answer':'6.000'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(trace['execution']['attempts'])==2
    rejected,computed=trace['execution']['attempts']
    assert rejected['status']=='needs_input' and computed['status']=='ok'
    assert computed['output_contract_status']=='valid'
    assert len(json.loads(seen[0]['messages'][1]['content'])['calls']['programs'])==6
    assert '164' in json.dumps(seen[-1]['messages']) and '6.000' in json.dumps(seen[-1]['messages'])
    assert hashlib.sha256(Path(cfg['data_dir'],'bank.sqlite3').read_bytes()).hexdigest()==before


@pytest.mark.parametrize('arm',['off','manual'])
def test_largest_input_preserved_for_both_arms(setup,monkeypatch,arm):
    cfg,bridge,bundle,resources=setup
    row=max(json.loads((resources/'test.json').read_text())['tasks'],key=lambda t:sum(len(p['text']) for p in t['inputs']['source_pages']))
    task=PublicTask(**row);seen=http(monkeypatch,[response([{'action':'finish','answer':'offline'}])])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'PRIVATE_GOLD'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['execution']['reason']=='agent_submitted',trace['execution'].get('finish_trigger')
    body=json.loads(seen[0]['messages'][1]['content'])
    assert body['task']['inputs']==row['inputs'] and 'PRIVATE_GOLD' not in json.dumps(seen)
    assert len(body['calls']['programs'])==(6 if arm=='manual' else 0)


def test_same_pairs_expanded_finish_and_latest_billing_ancestry(setup,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]/'experiments'))
    launcher=load(ROOT.parents[1]/'experiments/run_office_humanstrong_v3.py','v3_launcher')
    common=launcher.common
    for name in ('PACKAGE','frontier','VERSION'):monkeypatch.setattr(common,name,getattr(common,name))
    launcher.bind_runner()
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda path:None)
    parent=Path('/home/yangchengyu/officeqa_humanstrong_v2_finish32k_body_diagnostic_20261020')
    comparison=Path('/home/yangchengyu/officeqa_humanstrong_v2_train12_pairs_seed42_20261020')
    before=common.sha(parent/'budget.json');root=tmp_path/'pairs'
    launcher.prepare(root,parent,comparison,setup[3]);plan=common.verify(root)
    assert plan['new_token_cap']==5197824 and plan['inherited_used_or_reserved']==109536722
    assert plan['ledger_limits']['token_limit']==114734546
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline_fixture'}]) for _ in range(24)])
    assert common.execute(root,'offline-unused-env')==0
    assert common.status(root)['completed_episodes']==24 and common.status(root)['new_http']==24
    assert common.sha(parent/'budget.json')==before
    common.verify(root)
    for episode,wire in zip(plan['episodes'],seen):
        cfg=common.read(Path(episode['path'])/'config.json')
        assert cfg['budget']['eval_task_tokens']==216576
        assert cfg['llm']['purpose_overrides']['finish_only']['protocol']['thinking_type']=='disabled'
        assert cfg['llm']['purpose_overrides']['finish_only']['max_completion_tokens']==32768
        assert len(json.loads(wire['messages'][1]['content'])['calls']['programs'])==(6 if episode['arm']=='manual' else 0)
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
