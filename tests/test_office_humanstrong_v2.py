"""Authored V2: aligned public inputs, exact Provider wire and real sandbox calls."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http,response
from test_office_human_phase_v3 import load

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v2'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual HTTP forbidden in offline acceptance')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


@pytest.fixture
def setup(tmp_path):
    archive=os.environ.get('HUMANSTRONG_TEST102')
    resources=os.environ.get('OFFICE_EXTERNAL_RESOURCES')
    if not archive or not resources:pytest.skip('Saved config and external input snapshot needed')
    base=json.loads(next(Path(archive).glob('episodes/*/manual/config.json')).read_text())
    config=load(ROOT/'integration/configuration.py','strong2_cfg')
    cfg=config.configure(base,resources,'train')
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    bridge=load(ROOT/'integration/frontier.py','strong2_frontier')
    bundle=bridge.native_import(json.loads((ROOT/'bank/bank.json').read_text()),cfg['data_dir'],cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    return cfg,bridge,bundle,Path(resources)


def fixture_task(bridge):
    text='Millions of dollars\n| Year | Revenue |\n| --- | --- |\n| 1990 | 12 |\n| 1991 | 80 |\n'
    page={'document':'sample.txt','page':'1','source_url':'https://example.invalid/page=1','text':text,'text_digest':digest(text)}
    return PublicTask('s','s','Calculate the ratio of 1990 to 1991 revenue.',
        {'office_public_input_version':bridge.input_protocol.VERSION,'source_pages':[page],
         'candidate_files':['sample.txt'],'source_hints':[page['source_url']]},'train')


def test_bank_content_ids_visibility_and_new_formulas(setup):
    cfg,bridge,bundle,_=setup
    with closing(Bank(cfg['data_dir'],readonly=True,environment=cfg['program_environment'])) as bank:
        view=bridge.HumanBankView(bank,bundle);task=fixture_task(bridge);view.bind_task(task)
        for phase in ('locate','extract','compute','finalize'):
            view.observed['phase']=phase
            assert len(view.program_options())==5
            assert {'OA-Target','OA-Table','OA-Units','OA-Stats','OA-Commit'}<={a['title'] for a in view.selected(task)}
        off=bridge.HumanBankView(bank,bundle,'off');off.bind_task(task)
        assert not off.all('skill') and not off.program_options()
    text=(ROOT/'guidance.md').read_text()
    assert 'UID0' not in text and '820298' not in text
    metrics=load(ROOT/'programs/series_metrics.py','strong2_metrics')
    assert metrics.run(None,{'operation':'symmetric_difference','values':[10,20], 'unit':'percent','output_percent':True,'decimals':2})['outputs']['answer']=='66.67'
    assert metrics.run(None,{'operation':'share_difference','values':[1,4,3,5], 'unit':'percentage points','absolute':True,'output_percent':True,'decimals':2})['outputs']['answer']=='35.00'
    assert metrics.run(None,{'operation':'symmetric_difference','values':[0,0],'unit':'ratio'})['status']=='needs_input'


@pytest.mark.parametrize('arm',['off','manual'])
def test_real_wire_keeps_entire_largest_public_input_without_gold(setup,monkeypatch,arm):
    cfg,bridge,bundle,resources=setup
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    tasks=json.loads((resources/'test.json').read_text())['tasks']
    row=max(tasks,key=lambda t:sum(len(p['text']) for p in t['inputs']['source_pages']))
    task=PublicTask(**row);seen=http(monkeypatch,[response([{'action':'finish','answer':'offline-response'}])])
    adapter=OfficeAdapter({task.task_id:{'answer':'SECRET_EVALUATOR_ONLY'}},cfg)
    with bridge.bridge():
        s=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:
            trace=s.run_task(task,learn=False)
            assert len(seen)==1,trace['execution']
            body=json.loads(seen[0]['messages'][1]['content'])
            assert 'task' in body,trace['execution'].get('finish_trigger')
            assert body['task']['inputs']==task.inputs
            assert 'SECRET_EVALUATOR_ONLY' not in json.dumps(seen)
            assert len(body['provided_source_page_refs'])==len(task.inputs['source_pages'])
            assert not trace['execution']['attempts']
            assert len(body['calls']['programs'])==(5 if arm=='manual' else 0)
            for ref,page in zip(body['provided_source_page_refs'],task.inputs['source_pages']):
                assert s.task_context.resolve_public(ref)==page['text']
                assert s.task_context.results[ref['result_id']]['provenance']['kind']=='provided_public_input'
        finally:s.close()


@pytest.mark.parametrize('arm',['off','manual'])
def test_supplied_pages_survive_finish_with_no_tool_events(setup,monkeypatch,arm):
    cfg,bridge,bundle,_=setup;task=fixture_task(bridge)
    monkeypatch.setattr(Broker,'remaining_calls',lambda b:0)
    seen=http(monkeypatch,[response(content='0.15',finish='stop')])
    adapter=OfficeAdapter({'s':{'answer':'0.15'}},cfg)
    with bridge.bridge():
        s=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:trace=s.run_task(task,learn=False)
        finally:s.close()
    assert trace['score']['hard'] and trace['execution']['reason']=='finish_only'
    assert not seen[0].get('tools') and len(seen)==1
    body=json.loads(seen[0]['messages'][1]['content'])
    assert any(w['content']==task.inputs['source_pages'][0]['text'] for w in body['finish_evidence'])
    assert not trace['execution']['attempts']
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    assert input_token_bound(seen[0])==trace['execution']['finish_evidence']['serialized_input_bytes']


def test_five_programs_in_real_docker_and_no_implicit_workflow(setup,tmp_path,monkeypatch):
    cfg,bridge,bundle,_=setup;task=fixture_task(bridge)
    corpus=tmp_path/'corpus';corpus.mkdir();text=task.inputs['source_pages'][0]['text']
    (corpus/'sample.txt').write_text(text);cfg['harness']['corpus_root']=str(corpus)
    ps={Path(p['source_file']).stem:p for p in bundle['program']}
    cases=[('scoped_search',{'query':'Revenue','paths':['sample.txt']}),
        ('table_extract',{'text':text,'rows':['1990','1991'],'column':'Revenue','unit':'Millions of dollars'}),
        ('period_ledger',{'required_periods':['CY1990','CY1991'],'observations':[
            {'period':'CY1990','value':12,'unit':'millions','source_ref':{'result_id':'offline-synthetic','path':['data','text']}},
            {'period':'CY1991','value':80,'unit':'millions','source_ref':{'result_id':'offline-synthetic','path':['data','text']}}]}),
        ('series_metrics',{'operation':'symmetric_difference','values':[10,20],'output_percent':True,'unit':'percent','decimals':2}),
        ('series_metrics',{'operation':'share_difference','values':[1,4,3,5],'output_percent':True,'absolute':True,'unit':'percentage points','decimals':2}),
        ('treasury_transform',{'operation':'unit_scale','value':'12.5','source_unit':'millions_of_dollars','target_unit':'dollars','decimals':0})]
    seen=http(monkeypatch,[response([{'action':'call_program','name':ps[n]['id'],'arguments':a}]) for n,a in cases]+[response([{'action':'finish','answer':'12500000'}])])
    adapter=OfficeAdapter({'s':{'answer':'12500000'}},cfg)
    bank_before=hashlib.sha256(Path(cfg['data_dir'],'bank.sqlite3').read_bytes()).hexdigest()
    with bridge.bridge():
        s=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=s.run_task(task,learn=False)
        finally:s.close()
    assert trace['score']['hard'] and len(trace['execution']['attempts'])==6
    assert all(a['status']=='ok' and a['output_contract_status']=='valid' for a in trace['execution']['attempts'])
    assert len(seen)==7 and not trace['execution'].get('workflow_attempts')
    assert hashlib.sha256(Path(cfg['data_dir'],'bank.sqlite3').read_bytes()).hexdigest()==bank_before


def test_snapshot_split_isolation_and_old_input_refusal(setup):
    cfg,bridge,bundle,resources=setup
    manifest=json.loads((resources/'manifest.json').read_text())
    ids=[]
    for split,count in [('train',120),('val',24),('test',102)]:
        rows=json.loads((resources/(split+'.json')).read_text())['tasks'];assert len(rows)==count
        ids.extend(t['task_id'] for t in rows)
        for t in rows:
            assert set(t)=={'task_id','physical_key','goal','inputs','split'}
            assert not {'answer','gold','ground_truth','response','predicted_answer'}&set(t['inputs'])
            assert bool(t['inputs']['source_pages'])==(split!='val')
    assert len(ids)==len(set(ids)) and manifest['cross_seed_inputs_and_answers_identical']
    adapter=OfficeAdapter({'s':{'answer':'x'}},cfg);adapter.task=PublicTask('s','s','old',{'corpus':'full offline corpus'},'test')
    with bridge.bridge():
        s=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        s.task_context=TaskContext(cfg['runtime'])
        try:
            with pytest.raises(ValueError,match='External public inputs required'):
                bridge.input_protocol.seed_context(s)
        finally:s.close()


def test_frozen_evaluation_entry_logs_bill_and_checkpoint(setup,tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    cfg,bridge,bundle,resources=setup
    evaluate=load(ROOT/'integration/evaluate.py','strong2_eval')
    # Windows-managed worktree .git pointers are not Linux paths. This fixture
    # tests request/checkpoint/billing integration, not a paid source checkout.
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',
        lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'})
    task=PublicTask(**json.loads((resources/'train.json').read_text())['tasks'][0])
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline_fixture'}])])
    governor=BudgetGovernor(tmp_path/'fixture_budget.json',token_limit=192000,request_limit=2,finish_reserve=0)
    out=tmp_path/'evaluation'
    summary=evaluate.evaluate(cfg,[task],out,budget_governor=governor)
    assert summary['complete'] and len(seen)==1 and len(governor.state['attempts'])==1
    assert (out/'execution_manifest.json').is_file() and list((out/'requests').glob('*attempt1.json'))
    assert json.loads(seen[0]['messages'][1]['content'])['task']['inputs']==task.inputs
    with pytest.raises(ValueError,match='Task differs'):
        evaluate.evaluate(cfg,[PublicTask(task.task_id,task.physical_key,'changed',task.inputs,task.split)],tmp_path/'bad',budget_governor=governor)


@pytest.mark.parametrize('arm',['off','manual'])
def test_every_materialized_task_reaches_native_request_intact(setup,monkeypatch,arm):
    cfg,bridge,bundle,resources=setup
    rows=[t for split in ('train','val','test') for t in json.loads((resources/(split+'.json')).read_text())['tasks']]
    records={t['task_id']:{'answer':'PRIVATE_EVALUATOR_SENTINEL'} for t in rows}
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline_fixture'}]) for _ in rows])
    audit=[]
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter(records,cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
        try:
            for row in rows:
                trace=system.run_task(PublicTask(**row),learn=False)
                assert trace['execution']['reason']=='agent_submitted', (row['task_id'],trace['execution']['reason'])
                wire=seen[-1];body=json.loads(wire['messages'][1]['content'])
                assert body['task']['inputs']==row['inputs'],row['task_id']
                assert 'PRIVATE_EVALUATOR_SENTINEL' not in json.dumps(wire)
                assert len(body['calls']['programs'])==(5 if arm=='manual' else 0)
                audit.append({'task_id':row['task_id'],'split':row['split'],'public_input_digest':digest(row['inputs']),
                    'wire_bytes':len(json.dumps(wire).encode()),'completion_cap':wire['max_tokens'],
                    'page_count':len(row['inputs']['source_pages']),'normal_request':True})
            assert len(seen)==246
        finally:system.close()
    output=os.environ.get('OFFICE_V2_AUDIT_OUTPUT')
    if output:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        (out/f'wire_preflight_{arm}.json').write_text(json.dumps({'model_http':0,'responses':'offline_fixture_not_model_performance',
            'arm':arm,'count':len(audit),'tasks':audit},ensure_ascii=False,indent=2))
