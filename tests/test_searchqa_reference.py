"""Authored catalog through real Docker and intercepted production provider HTTP."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.manual_reference import ReferenceView, SEARCHQA_VERSION
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.public_results import MANUAL_GRAPH_EXECUTION, MANUAL_PROGRAM_EXECUTION
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments import run_searchqa_reference_test as runner
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http, response

PACKAGE=Path('/mnt/d/T3S_exp/resources/searchqa_manual_20261015/SearchQA_Manual_Ideal_SkillBank_20261015')
PARENT=Path('/home/yangchengyu/searchqa_trace_r2_seed42_20261014_a418866')
DATASETS=Path('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/searchqa')
pytestmark=pytest.mark.skipif(not PACKAGE.exists(),reason='External authored package unavailable')


@pytest.fixture(scope='module')
def qualified(tmp_path_factory):
    root=tmp_path_factory.mktemp('searchqa_reference')
    cfg=runner.configuration(runner.read(PARENT/'resolved_config.json'),root,DATASETS)
    assert runner.qualify(PACKAGE,root,cfg)=={'skills':12,'programs':8,'implementations':8,'workflows':5}
    runner.write_json(root/'config.json',cfg)
    return root


@pytest.fixture
def prepared(qualified,tmp_path):
    root=tmp_path/'run';shutil.copytree(qualified,root)
    cfg=runner.configuration(runner.read(PARENT/'resolved_config.json'),root,DATASETS)
    runner.write_json(root/'config.json',cfg)
    return root


def action(**kwargs):return response([kwargs],name='answer_step')


def system(root, task):
    cfg=runner.read(root/'config.json')
    s=EmpiricalSystem(cfg,readonly=True,harness=AnswerAdapter('searchqa',{task.task_id:{'answer':'fixture'}}))
    s.checkpoint=TaskCheckpoint(root/'checkpoint')
    return s


def test_catalog_remains_candidates_and_receipts_are_required(prepared):
    b=Bank(prepared/'frozen_bank',readonly=True)
    try:
        assert all(p['state']=='candidate' and not b.program_eligible(p) for p in b.all('program'))
        view=ReferenceView(b,prepared/'reference_manifest.json')
        assert len(view.program_options(''))==8 and len(view.workflow_options())==5
        ref=next(iter(runner.read(prepared/'reference_manifest.json')['program_receipts'].values()))[0]
        (prepared/ref['file']).write_text('{}')
        with pytest.raises(ValueError,match='receipt changed'):ReferenceView(b,prepared/'reference_manifest.json')
    finally:b.close()


@pytest.mark.parametrize('index,filename,extra,count',[
    (0,'evidence_search.json',{},2),(1,'explicit_relation.json',{},3),
    (2,'numeric_rows.json',{'subject':'Population','mode':'latest'},2),
    (3,None,{},1),(4,'evidence_search.json',{},2)])
def test_all_five_graphs_have_actual_dataflow(prepared,monkeypatch,index,filename,extra,count):
    data=runner.read(PACKAGE/'fixtures'/filename) if filename else {
        'question':'What does ERG stand for?','context':'Example Research Group (ERG)'}
    task=PublicTask('fixture','fixture',data['question'],{'context':data['context']},'test')
    wid=runner.read(PACKAGE/'bank/bank.json')['workflows'][index]['id']
    nodes=[response(finish='stop',content=json.dumps({'status':'ok','outputs':{'answer':'fixture'},
        'references':[{'task_path':['inputs','context'],'quote':data['context']}]}))] if index>=3 else []
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments=extra),*nodes,
        action(action='finish',answer='<answer>fixture</answer>')])
    before=tree_identity(prepared/'frozen_bank');s=system(prepared,task)
    try:trace=s.run_task(task,learn=False)
    finally:s.close()
    graph=trace['execution']['workflow_attempts'][0]
    assert graph['status']=='ok',graph
    assert graph['program_executions']==count and len(calls)==len(nodes)+2
    assert len(trace['injected_guidance_ids'])==12
    first=json.loads(calls[0]['messages'][1]['content'])
    assert len(first['programs'])==8 and len(first['workflows'])==5
    assert 'At most three local executions' in calls[0]['messages'][0]['content']
    for row in graph['nodes']:
        if row['mode']=='program':assert '--network=none' in row['result']['execution_receipt']['command']
    if index==1:assert graph['nodes'][2]['arguments']['verified'] is True
    for wire in calls:
        payload=json.loads(wire['messages'][1]['content'])
        assert payload['goal']==task.goal and payload['inputs']==task.inputs
        assert wire['max_tokens']==32768 and wire['thinking']=={'type':'enabled'} and wire['reasoning_effort']=='high'
        assert 'execution_receipt' not in json.dumps(payload) and 'evaluator_records' not in json.dumps(payload)
    assert tree_identity(prepared/'frozen_bank')==before


@pytest.mark.parametrize('argument',[{'context':'replacement'},{'question':'replacement'},{'inputs':{}},{'goal':'replacement'}])
def test_canonical_input_cannot_be_replaced(prepared,monkeypatch,argument):
    task=PublicTask('fixture','fixture','original',{'context':'original context'},'test')
    wid=runner.read(PACKAGE/'bank/bank.json')['workflows'][0]['id']
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments=argument),
        action(action='finish',answer='<answer>fixture</answer>')])
    s=system(prepared,task)
    try:trace=s.run_task(task,learn=False)
    finally:s.close()
    graph=trace['execution']['workflow_attempts'][0]
    assert graph['status']=='needs_input' and graph['program_executions']==0 and len(calls)==2


def test_host_receipts_unreadable():
    context=TaskContext(runner.read(PARENT/'resolved_config.json')['runtime'])
    for origin,hidden in [(MANUAL_GRAPH_EXECUTION,'nodes'),(MANUAL_PROGRAM_EXECUTION,'execution_receipt')]:
        rid=context.register(origin,{'status':'ok','outputs':{'answer':'public'},hidden:{'private':'receipt'}},origin=origin)
        assert hidden not in context.view(rid)
        with pytest.raises(ValueError,match='Host metadata'):context.resolve_public({'result_id':rid,'path':[hidden]})


def test_independent_config_cannot_claim_r2(prepared):
    cfg=runner.read(prepared/'config.json');cfg['experiment']['searchqa_method_version']='searchqa.train-trace-capability.r2'
    with pytest.raises(ValueError):runner.validate_config(cfg)
    assert runner.read(prepared/'config.json')['experiment']['manual_reference_version']==SEARCHQA_VERSION


def test_full_test_identity_budget_logging_and_resume(tmp_path,qualified,monkeypatch):
    # All 1400 canonical inputs use the production payload/scorer/logging path,
    # with HTTP intercepted and no real model calls. Qualification already ran above.
    def reuse(package,root,cfg):
        for name in ('frozen_bank','qualification'):shutil.copytree(qualified/name,root/name)
        shutil.copyfile(qualified/'reference_manifest.json',root/'reference_manifest.json')
        return {'skills':12,'programs':8,'implementations':8,'workflows':5}
    monkeypatch.setattr(runner,'qualify',reuse)
    monkeypatch.setattr(runner,'code_identity',lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'fixture'})
    monkeypatch.setattr(runner,'load_env',lambda p:None)
    root=tmp_path/'full';runner.prepare(PACKAGE,PARENT,DATASETS,root)
    calls=http(monkeypatch,[response(finish='stop',content='<answer>fixture</answer>') for _ in range(1400)])
    result=runner.evaluate(root,'unused')
    assert result['status']=='completed' and result['tasks']==1400 and len(calls)==1400
    assert runner.evaluate(root,'unused')==result and len(calls)==1400
    assert runner.read(root/'freeze_audit.json')['unchanged']
    tasks=runner.read(root/'tasks.json')['tasks']
    for task,wire in zip(tasks,calls):
        material=json.loads(wire['messages'][1]['content'])
        assert material['goal']==task['goal'] and material['inputs']==task['inputs']
    budget=runner.read(root/'budget.json');prefix=runner.read(root/'inherited_budget.json')
    assert len(budget['attempts'])-len(prefix['attempts'])==1400
    assert all(budget['attempts'][k]==v for k,v in prefix['attempts'].items())
    assert len(list((root/'test/traces').glob('*.json')))==1400
