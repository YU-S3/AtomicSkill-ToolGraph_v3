"""Authored Bank revision, exact spans, cost ancestry and unchanged public replay."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.manual_reference import ReferenceView
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments import run_searchqa_reference_test as runner
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http, response
from test_searchqa_reference import PACKAGE, PARENT, DATASETS, action, system

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_revision', REPO/'reference_banks/searchqa_v2/build.py')
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
pytestmark = pytest.mark.skipif(not PACKAGE.exists(), reason='Original reference package unavailable')


@pytest.fixture(scope='module')
def revised(tmp_path_factory):
    root = tmp_path_factory.mktemp('manual_revision')
    package = root/'package'; builder.build(PACKAGE, package)
    cfg = runner.configuration(runner.read(PARENT/'resolved_config.json'), root, DATASETS)
    assert runner.qualify(package, root, cfg) == {'skills':12,'programs':8,'implementations':8,'workflows':5}
    runner.write_json(root/'config.json', cfg)
    return root


def test_new_bank_identity_and_real_qualification(revised):
    old, new = runner.read(PACKAGE/'bank/bank.json'), runner.read(revised/'package/bank/bank.json')
    assert new['parent_bank_sha256'] == runner.sha256(PACKAGE/'bank/bank.json')
    assert {s['parent_id'] for s in new['skills']} == {s['id'] for s in old['skills']}
    assert not {s['id'] for s in new['skills']} & {s['id'] for s in old['skills']}
    assert len({p['id'] for p in new['programs']} - {p['id'] for p in old['programs']}) == 3
    bank = Bank(revised/'frozen_bank', readonly=True)
    try:
        assert len(ReferenceView(bank, revised/'reference_manifest.json').qualified) == 8
        assert all(not bank.program_eligible(p) for p in bank.all('program'))
    finally: bank.close()


def execute(revised, name, inputs):
    cfg = runner.read(revised/'config.json')
    source = (revised/'package/programs'/(name+'.py')).read_text()
    p = next(p for p in runner.read(revised/'package/bank/bank.json')['programs'] if p['source']==source)
    task = PublicTask('synthetic', digest(inputs), 'Synthetic span test', inputs, 'reference_probe')
    adapter = AnswerAdapter('searchqa', {'synthetic':{}}); adapter.reset(task)
    broker = Broker(adapter, 1, context=TaskContext(cfg['runtime']))
    try: result = ProgramWorker(cfg['program_worker']).execute(p, inputs, broker)
    finally: adapter.close()
    assert result['execution_receipt']['exit_code'] == 0 and not broker.events
    return result


def test_titles_linewraps_and_diverse_windows(revised):
    context = ''.join('[DOC] [TLE] Archive '+str(i)+' [PAR] Mira\nVale joined the Lyra ensemble. '+
        ('Mira Vale '*120 if i==1 else 'A documented relation.') for i in range(1,10))
    docs = execute(revised,'doc_catalog',{'context':context})['outputs']['documents']
    assert len(docs)==9 and docs[0]['title']=='Archive 1'
    assert all(context[d['start']:d['end']]==d['text'] for d in docs)
    result=execute(revised,'evidence_windows',{'question':'Mira Vale ensemble','documents':docs})
    windows=result['outputs']['windows']
    assert len(windows)==8 and len({w['doc_index'] for w in windows})==8
    assert all(context[w['start']:w['end']]==w['quote'] for w in windows)
    assert any('Mira\nVale joined the Lyra ensemble' in w['quote'] for w in windows)
    assert execute(revised,'explicit_relation',{'question':'Mira Vale, 1984','context':context})['status']=='not_found'


def test_ancestry_keeps_parent_delta_and_rejects_changed_costs(tmp_path):
    ancestor, parent = tmp_path/'ancestor', tmp_path/'parent'
    limits={'token_limit':1000,'request_limit':100,'finish_reserve':0}
    attempt={'status':'metered','accounted_tokens':10}
    runner.write_json(ancestor/'budget.json',{'attempts':{'a':attempt},'limits':limits})
    runner.write_json(parent/'budget.json',{'attempts':{'a':attempt,'b':attempt},'limits':{'token_limit':200}})
    runner.write_json(parent/'cost_carryforward_manifest.json',{'parent':str(ancestor)})
    runner.write_json(parent/'result.json',{'status':'completed'})
    assert runner.cumulative_authority(parent,ancestor)==(ancestor,limits)
    runner.write_json(parent/'result.json',{'status':'stopped'})
    with pytest.raises(ValueError,match='Finish the parent'):runner.cumulative_authority(parent,ancestor)
    runner.write_json(parent/'result.json',{'status':'completed'})
    runner.write_json(parent/'budget.json',{'attempts':{'b':attempt}})
    with pytest.raises(ValueError,match='lost or changed'):runner.cumulative_authority(parent,ancestor)


def test_frozen_failure_replay_uses_exact_public_inputs(revised,tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir()
    for name in ('budget.json','resolved_config.json'): shutil.copyfile(PARENT/name,parent/name)
    tasks=runner.read(DATASETS/'test.json')['tasks'][:2]
    runner.write_json(parent/'test/summary.json',{'cases':[{'task':t,'score':{'hard':False}} for t in tasks]})
    selected=tmp_path/'selection.json';runner.write_json(selected,{'task_ids':[t['task_id'] for t in tasks]})
    def reuse(package, root, cfg):
        for name in ('frozen_bank','qualification'):shutil.copytree(revised/name,root/name)
        shutil.copyfile(revised/'reference_manifest.json',root/'reference_manifest.json')
        return {'skills':12,'programs':8,'implementations':8,'workflows':5}
    monkeypatch.setattr(runner,'qualify',reuse)
    monkeypatch.setattr(runner,'code_identity',lambda:{'git_sha':'fixture','tracked_dirty':False})
    monkeypatch.setattr(runner,'load_env',lambda path:None)
    root=tmp_path/'replay'
    assert runner.prepare(revised/'package',parent,DATASETS,root,selection=selected)['tasks']==2
    calls=http(monkeypatch,[response(finish='stop',content='<answer>fixture</answer>') for _ in tasks])
    assert runner.evaluate(root,'unused')['status']=='completed'
    assert runner.status(root)['total']==2
    assert runner.read(root/'plan.json')['provenance']=='human_authored_test_feedback_revision_not_unseen_test'
    for t,call in zip(tasks,calls):
        material=json.loads(call['messages'][1]['content'])
        assert material['goal']==t['goal'] and material['inputs']==t['inputs']
        assert call['max_tokens']==32768 and call['reasoning_effort']=='high'
    assert runner.evaluate(root,'unused')['tasks']==2 and len(calls)==2
    assert runner.read(root/'freeze_audit.json')['unchanged']


def test_revised_workflow_roundtrip(revised,tmp_path,monkeypatch):
    root=tmp_path/'run';shutil.copytree(revised,root)
    cfg=runner.configuration(runner.read(PARENT/'resolved_config.json'),root,DATASETS)
    runner.write_json(root/'config.json',cfg)
    task=PublicTask('fixture','fixture','Who is the mother of Mira Vale?',
        {'context':'[DOC]The mother of Mira Vale is Ada Vale.'},'test')
    wid=runner.read(root/'package/bank/bank.json')['workflows'][1]['id']
    calls=http(monkeypatch,[action(action='call_workflow',name=wid,arguments={}),
        response(finish='stop',content='<answer>Ada Vale</answer>')])
    s=system(root,task)
    try: trace=s.run_task(task,learn=False)
    finally:s.close()
    assert trace['execution']['workflow_attempts'][0]['status']=='ok' and len(calls)==2
    assert trace['execution']['internal_execution_count']==3
