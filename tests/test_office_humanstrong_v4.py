"""Source-bound composition, real Docker submission and complete Test preflight."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask,digest,program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http,response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v2 as v2

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v4'
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v3_train12_pairs_seed42_20261020')
RESOURCES=Path('/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual HTTP forbidden in offline acceptance')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


@pytest.fixture
def setup(tmp_path):
    if not PARENT.exists():pytest.skip('Immutable parent and external resource snapshot required')
    cfg=load(ROOT/'integration/configuration.py','v4_cfg').configure(
        json.loads((PARENT/'base_config.json').read_text()),RESOURCES,'test')
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    bridge=load(ROOT/'integration/frontier.py','v4_frontier')
    bundle=bridge.native_import(json.loads((ROOT/'bank/bank.json').read_text()),cfg['data_dir'],
        cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    return cfg,bridge,bundle


def test_asset_identity_and_reproducible_source():
    bank=json.loads((ROOT/'bank/bank.json').read_text())
    old=json.loads((ROOT.parent/'office_humanstrong_v3/bank/bank.json').read_text())
    assert bank['parent_bank_digest']==digest(old)
    assert {p['id'] for p in old['program']} < {p['id'] for p in bank['program']}
    assert len(bank['program'])==7
    for p in bank['program']:
        assert p['source']==(ROOT/p['source_file']).read_text()
        assert p['id']=='program_'+program_digest(p)
    assert bank['program'][-1]['result_role']=='final_answer'
    assert bank['program'][-1]['allowed_tools']==['read_result']
    assets=(ROOT/'guidance_changes.json').read_text()+(ROOT/'programs/table_calculate.py').read_text()
    assert 'UID0' not in assets and '187551' not in assets and '0.162' not in assets


class LocalReads:
    def __init__(self,text):
        self.context=TaskContext();self.calls=[]
        rid=self.context.register('public-fixture',{'data':{'text':text}})
        self.ref={'result_id':rid,'path':['data','text']}
    def remaining_calls(self):return 48-len(self.calls)
    def call(self,name,args):
        assert name=='read_result'
        self.calls.append(args)
        return self.context.read(**args)


def polynomial_args(ref):
    common={'source_ref':ref,'column':'Value','unit':'widgets'}
    return {'submit_answer':True,'answer_unit':'widgets',
        'tables':[dict(common,key='fit',rows=['p1','p2','p3','p4','p5']),
                  dict(common,key='target',rows=['p6'])],
        'bindings':{'observations':['fit'],'actual':['target']},
        'calculation':{'method':'polynomial_forecast','operation':'absolute_error',
            'degree':2,'first_x':1,'x_step':1,'prediction_x':6,'decimals':3}}


def polynomial_text():
    return 'In widgets\n| Period | Value |\n| --- | --- |\n'+''.join(
        f'| p{i} | {2+3*i+4*i*i} |\n' for i in range(1,6))+'| p6 | 170 |\n'


def test_source_selection_missing_actual_ambiguity_and_complete_rows():
    program=load(ROOT/'programs/table_calculate.py','v4_program')
    ctx=LocalReads(polynomial_text());args=polynomial_args(ctx.ref)
    output=program.run(ctx,args)
    assert output['outputs']['answer']=='6.000'
    assert len(ctx.calls)==1  # Identical source windows are read once per invocation.
    for cell in output['outputs']['evidence']:
        assert polynomial_text()[cell['start']:cell['end']]==cell['quote']
        assert cell['source_ref']==ctx.ref
    missing=deepcopy(args);missing['tables'].pop();missing['bindings'].pop('actual')
    assert 'independently acquired actual' in program.run(ctx,missing)['detail']
    bad=deepcopy(args);bad['tables'][0]['rows'][0]='missing period'
    assert program.run(ctx,bad)['status']=='needs_input'
    duplicate=LocalReads(polynomial_text()+'| p1 | 9 |\n')
    assert program.run(duplicate,polynomial_args(duplicate.ref))['status']=='needs_input'
    short=deepcopy(args);short['tables'][1]['length']=polynomial_text().index('170')+1
    assert program.run(ctx,short)['status']=='needs_input'
    wrong=deepcopy(args);wrong['tables'][0]['source_ref']['result_id']='not-obtained'
    assert program.run(ctx,wrong)['status']=='needs_input'
    wrong=deepcopy(args);wrong['answer_unit']='other unit'
    assert program.run(ctx,wrong)['status']=='needs_input'
    wrong=deepcopy(args);wrong['submit_answer']=False
    assert program.run(ctx,wrong)['status']=='needs_input'
    ratio={'submit_answer':True,'answer_unit':'percent',
        'tables':[dict(args['tables'][0],key='base',rows=['p5']),
                  dict(args['tables'][0],key='new',rows=['p1'])],
        'bindings':{'base_value':['base'],'new_value':['new']},
        'calculation':{'method':'series_metrics','operation':'absolute_relative_change','output_percent':True,'decimals':2}}
    assert program.run(ctx,ratio)['outputs']['answer']=='92.31'
    ratio['calculation']['operation']='percent_change'
    ratio['calculation']['absolute']=True
    assert program.run(ctx,ratio)['status']=='needs_input'  # Never silently ignore an option.


@pytest.mark.parametrize('case',['saved_table','missing_then_polynomial'])
def test_source_to_real_docker_to_final_answer_without_llm_rewrite(setup,monkeypatch,case,tmp_path):
    cfg,bridge,bundle=setup
    if case=='saved_table':
        source=next(PARENT.glob('episodes/*UID0146/manual/traces/*.json'))
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        task=PublicTask(**json.loads(source.read_text())['task']);expected='3.26'
    else:
        task=v2.fixture_task(bridge);page=task.inputs['source_pages'][0]
        page.update(text=polynomial_text(),text_digest=digest(polynomial_text()))
        task=PublicTask(task.task_id,task.physical_key,'Fit quadratic observations p1..p5 and report absolute error at p6.',task.inputs,'train')
        expected='6.000'
    program=bundle['program'][-1];seen=[]
    def post(*a,**kw):
        wire=kw['json'];seen.append(deepcopy(wire))
        ref=json.loads(wire['messages'][1]['content'])['provided_source_page_refs'][0]
        if case=='saved_table':
            assert len(seen)==1,'Successful final Program must not invoke another model'
            common={'source_ref':ref,'column':'Total amount outstanding 1/','unit':'millions of dollars'}
            args={'submit_answer':True,'answer_unit':'ratio',
                'tables':[dict(common,key='a',rows=['Public marketable.................']),
                          dict(common,key='b',rows=['Public nonmarketable 5/.............'])],
                'bindings':{'numerator':['a'],'denominator':['b']},
                'calculation':{'method':'series_metrics','operation':'ratio','output_percent':False,'decimals':2}}
            assert '187551' not in json.dumps(args) and '57449' not in json.dumps(args)
        else:
            args=polynomial_args(ref)
            if len(seen)==1:
                args['tables'].pop();args['bindings'].pop('actual')
            else:assert len(seen)==2,'No extra model rewrite after successful calculation'
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([
            {'action':'call_program','name':program['id'],'arguments':args}]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    before_bank=hashlib.sha256(Path(cfg['data_dir'],'bank.sqlite3').read_bytes()).hexdigest()
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':expected}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'],trace['execution']
    assert trace['execution']['prediction']==expected and trace['execution']['reason']=='plan_submitted'
    attempts=trace['execution']['attempts']
    assert attempts[-1]['status']=='ok' and attempts[-1]['output_contract_status']=='valid'
    assert trace['execution']['submission_producer_attempt_id']==attempts[-1]['id']
    if case!='saved_table':assert attempts[0]['status']=='needs_input' and len(attempts)==2
    assert hashlib.sha256(Path(cfg['data_dir'],'bank.sqlite3').read_bytes()).hexdigest()==before_bank
    if case=='saved_table':assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    # Optional acceptance artifact contains public inputs and intercepted wire only.
    import os
    if os.environ.get('V4_EVIDENCE_DIR'):
        out=Path(os.environ['V4_EVIDENCE_DIR']);out.mkdir(parents=True,exist_ok=True)
        (out/(case+'.json')).write_text(json.dumps({'actual_llm_http':0,'fixture_decisions':True,
            'real_docker':True,'trace':trace,'sent':seen},ensure_ascii=False,indent=2))


@pytest.mark.parametrize('arm',['manual','off'])
def test_shared_answer_contract_matches_finish_admission(setup,monkeypatch,arm):
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    cfg,bridge,bundle=setup;task=v2.fixture_task(bridge)
    monkeypatch.setattr(Broker,'remaining_calls',lambda b:0)
    seen=http(monkeypatch,[response(content='0.15',finish='stop')])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({'s':{'answer':'0.15'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(seen)==1
    assert seen[0]['messages'][0]['content'].count(bridge.ANSWER_CONTRACT)==1
    assert input_token_bound(seen[0])==trace['execution']['finish_evidence']['serialized_input_bytes']
    assert seen[0]['max_tokens']==32768 and seen[0]['thinking']['type']=='disabled'


@pytest.mark.parametrize('arm',['manual','off'])
@pytest.mark.parametrize('measure',['pages','bytes'])
def test_most_evidence_keeps_initial_tools_and_exact_inputs(setup,monkeypatch,arm,measure):
    cfg,bridge,bundle=setup
    rows=json.loads((RESOURCES/'test.json').read_text())['tasks']
    row=max(rows,key=lambda t:len(t['inputs']['source_pages']) if measure=='pages' else
        sum(len(p['text'].encode()) for p in t['inputs']['source_pages']))
    task=PublicTask(**row);seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}])])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'PRIVATE_GOLD'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['execution']['reason']=='agent_submitted',trace['execution']['finish_trigger']
    assert json.loads(seen[0]['messages'][1]['content'])['task']['inputs']==task.inputs
    assert 'PRIVATE_GOLD' not in json.dumps(seen)
    assert seen[0].get('tools')


def test_all_test_pairs_public_input_wire_budget_and_frozen_billing(setup,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]/'experiments'))
    launcher=load(ROOT.parents[1]/'experiments/run_office_humanstrong_v4.py','v4_launcher')
    common=launcher.common
    for name in ('PACKAGE','frontier','VERSION'):monkeypatch.setattr(common,name,getattr(common,name))
    launcher.bind_runner()
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda p:None)
    before=common.sha(PARENT/'budget.json');root=tmp_path/'pairs'
    launcher.prepare(root,PARENT,RESOURCES);plan=common.verify(root)
    assert plan['new_token_cap']==44181504 and plan['inherited_used_or_reserved']==110804968
    assert plan['ledger_limits']['token_limit']==154986472 and plan['authorized_cumulative_ceiling']==155000000
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in range(204)])
    assert common.execute(root,'offline-unused-env')==0
    state=common.status(root)
    assert state['completed_episodes']==204 and state['new_http']==204
    assert common.sha(PARENT/'budget.json')==before
    rows={t['task_id']:t for t in common.read(root/'tasks.json')['tasks']}
    for episode,wire in zip(plan['episodes'],seen):
        body=json.loads(wire['messages'][1]['content'])
        assert 'task' in body,(episode['task_id'],episode['arm'],'initial request entered finish-only')
        assert body['task']['inputs']==rows[episode['task_id']]['inputs']
        assert 'answer' not in body['task']['inputs'] and 'gold' not in body['task']['inputs']
        assert common.frontier.ANSWER_CONTRACT in wire['messages'][0]['content']
        assert len(body['calls']['programs'])==(7 if episode['arm']=='manual' else 0)
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
    common.verify(root)
