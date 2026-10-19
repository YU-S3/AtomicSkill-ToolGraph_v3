"""Native import, real Docker and exact Provider-wire acceptance; zero model HTTP."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.empirical.contracts import PublicTask, program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_office_human_phase_v3 import load
from test_cf2_contracts import http,response

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v1'

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual HTTP forbidden in offline acceptance')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)

@pytest.fixture
def setup(tmp_path):
    archive=os.environ.get('OFFICE_HUMAN_ARCHIVE')
    if not archive:pytest.skip('Set OFFICE_HUMAN_ARCHIVE to a locked real Office config')
    cfg=json.loads((Path(archive)/'config.json').read_text())
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    bridge=load(ROOT/'integration/frontier.py','humanstrong_test')
    bundle=json.loads((ROOT/'bank/bank.json').read_text())
    imported=bridge.native_import(bundle,tmp_path/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    return cfg,bridge,imported

def test_import_links_manual_permission_and_off_isolation(setup):
    cfg,bridge,imported=setup
    with closing(Bank(cfg['data_dir'],readonly=True,environment=cfg['program_environment'])) as bank:
        assert {k:len(bank.all(k)) for k in bridge.KINDS}=={'skill':11,'program':5,'implementation':5,'workflow':4}
        members={a['id'] for k in bridge.KINDS for a in bank.all(k)}
        for a in imported['implementation']:
            assert a['skill_id'] in members and a['program_id'] in members
        for wf in imported['workflow']:
            for node in wf['nodes']:
                assert all(node[k] in members for k in ['skill_id','program_id'] if node.get(k))
        assert all(p['id']=='program_'+program_digest(p) and not p.get('local_validation') for p in imported['program'])
        task=PublicTask('s','s','Calculate a currency exchange conversion.',{},'train')
        view=bridge.HumanBankView(bank,imported);view.bind_task(task)
        for phase in ['locate','extract','compute','finalize']:
            assert 1<=len(view.selected(task,phase))<=3
        off=bridge.HumanBankView(bank,imported,'off');off.bind_task(task)
        assert not off.selected(task) and not off.program_options() and not off.all('skill')
        assert all(view.program_eligible(p) and not off.program_eligible(p) for p in imported['program'])


def test_supplied_assets_unchanged_and_bridge_delta_audited():
    original=json.loads((ROOT/'FILE_SHA256S.json').read_text())['member_sha256']
    receipt=json.loads((ROOT/'integration_receipt.json').read_text())
    for name,expected in original.items():
        path=ROOT/name
        if name in receipt['local_changes']:
            assert hashlib.sha256((ROOT/'sources'/('original_'+Path(name).name+'.txt')).read_bytes()).hexdigest()==expected
            expected=receipt['local_changes'][name]['after_sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected


def test_all_five_programs_execute_native_docker_and_do_not_auto_run_workflow(setup,tmp_path,monkeypatch):
    cfg,bridge,bundle=setup
    corpus=tmp_path/'corpus';corpus.mkdir()
    text='Millions of dollars\n| Year | Revenue |\n| 1990 | 12 |\n| 1991 | 80 |\n'
    (corpus/'sample.txt').write_text(text);cfg['harness']['corpus_root']=str(corpus)
    ps={Path(p['source_file']).stem:p for p in bundle['program']}
    cases=[('scoped_search',{'query':'Revenue','paths':['sample.txt']}),
        ('table_extract',{'text':text,'rows':['1990','1991'],'column':'Revenue','unit':'Millions of dollars'}),
        ('period_ledger',{'required_periods':['CY1990','CY1991'],'observations':[
            {'period':'CY1990','value':12,'unit':'millions','source_ref':{'result_id':'synthetic-test-only','path':['data','text']}},
            {'period':'CY1991','value':80,'unit':'millions','source_ref':{'result_id':'synthetic-test-only','path':['data','text']}}]}),
        ('series_metrics',{'operation':'ratio','numerator':12,'denominator':80,'output_percent':False,'unit':'ratio'}),
        ('treasury_transform',{'operation':'unit_scale','value':'12.5','source_unit':'millions_of_dollars',
                              'target_unit':'dollars','decimals':0})]
    responses=[response([{'action':'call_program','name':ps[name]['id'],'arguments':args}]) for name,args in cases]
    responses.append(response([{'action':'finish','answer':'12500000'}]))
    seen=http(monkeypatch,responses);before=tree_identity(cfg['data_dir'])
    adapter=OfficeAdapter({'s':{'answer':'12500000'}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(PublicTask('s','s','Synthetic test: convert acquired revenue to dollars.',{},'train'),learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(trace['execution']['attempts'])==5
    assert all(a['status']=='ok' and a['output_contract_status']=='valid' for a in trace['execution']['attempts'])
    assert not trace['execution'].get('workflow_attempts')
    assert len(seen)==6 and tree_identity(cfg['data_dir'])==before
    assert all(len(json.loads(q['messages'][1]['content'])['human_frontier_skill_ids'])<=3 for q in seen)


def test_forced_finish_wire_has_no_native_finish_instruction(setup,tmp_path,monkeypatch):
    cfg,bridge,bundle=setup
    from atomic_skillgraph.harness.simple_protocol import Broker
    remaining=Broker.remaining_calls
    monkeypatch.setattr(Broker,'remaining_calls',lambda b:min(remaining(b),max(0,1-len(b.events))))
    corpus=tmp_path/'corpus';corpus.mkdir();(corpus/'sample.txt').write_text('Actual observed revenue: 12.5 millions.')
    cfg['harness']['corpus_root']=str(corpus)
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'read','arguments':{'path':'sample.txt','offset':0}}]),
                           response(content='12.5',finish='stop')])
    adapter=OfficeAdapter({'s':{'answer':'12.5'}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(PublicTask('s','s','Return the actual revenue in millions.',{},'train'),learn=False)
        finally:system.close()
    assert trace['score']['hard'] and trace['execution']['reason']=='finish_only'
    assert len(seen)==2 and not seen[-1].get('tools') and seen[-1]['thinking']['type']=='disabled'
    material=json.loads(seen[-1]['messages'][1]['content'])
    assert material['authored_commit_policy']==bridge.selection.FINISH_ONLY_RULE
    assert 'immediately call the native' not in json.dumps(seen[-1])
    assert input_token_bound(seen[-1])==trace['execution']['finish_evidence']['serialized_input_bytes']<=16384


def test_off_arm_has_no_manual_prompt_or_programs(setup,monkeypatch):
    cfg,bridge,bundle=setup
    seen=http(monkeypatch,[response([{'action':'finish','answer':'7'}])])
    adapter=OfficeAdapter({'s':{'answer':'7'}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,'off'))
        try:trace=system.run_task(PublicTask('s','s','Return seven.',{},'train'),learn=False)
        finally:system.close()
    assert trace['score']['hard'] and not trace['execution']['attempts']
    material=json.loads(seen[0]['messages'][1]['content'])
    assert 'human_frontier_skill_ids' not in material and not material['calls']['programs']
    assert bridge.selection.GLOBAL_RULE not in seen[0]['messages'][0]['content']


def test_preregistered_twelve_pairs_preserve_ledger_and_native_config_bytes(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
    import run_office_humanstrong_v1 as runner
    base=os.environ.get('OFFICE_STRONG_BASE')
    if not base:pytest.skip('Set OFFICE_STRONG_BASE to the completed V3.1 continuation')
    base=Path(base);root=tmp_path/'run';old=runner.read(base/'budget.json')
    resources=runner.read(base/'plan.json')['resources']
    monkeypatch.setattr(runner,'code_identity',lambda:{'git_sha':'offline-test','tracked_dirty':False,'source_sha256':'offline-test'})
    runner.prepare(root,base,resources)
    plan=runner.verify(root);new=runner.read(root/'budget.json')
    assert len(plan['episodes'])==24 and new['attempts']==old['attempts']
    assert plan['new_token_cap']==4608000 and plan['new_http_cap']==2400
    assert new['limits']['token_limit']==97635247 and plan['authorized_cumulative_ceiling']==98000000
    assert runner.read(root/'tasks.json')==runner.read(base/'tasks.json')
    assert [(e['task_id'],e['arm']) for e in plan['episodes']]==[(e['task_id'],e['arm']) for e in runner.read(base/'plan.json')['episodes']]
    # Production config validation/serialization cannot invalidate the frozen identity.
    for e in plan['episodes']:
        p=Path(e['path'])/'config.json';cfg=runner.validate_config(runner.read(p));runner.write_json(p,cfg)
    runner.verify(root)
