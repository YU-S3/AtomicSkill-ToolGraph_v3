"""Saved-source acceptance; model transport is intercepted, never paid."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.empirical.public_results import TEXT_CONTEXT, FILE_COMPUTE
from atomic_skillgraph.empirical.prompts import finish_text_prompt
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http, response
from test_office_human_phase_v3 import load

ROOT = Path(__file__).parents[1]/'resources/office_humanstrong_v1'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request', lambda *a, **kw: pytest.fail('No model HTTP in acceptance'))


def archived():
    path = os.environ.get('HUMANSTRONG_TRAIN12')
    if not path: pytest.skip('Set HUMANSTRONG_TRAIN12 to the completed fixed observation')
    return Path(path)


@pytest.mark.parametrize('arm', ['off', 'manual'])
@pytest.mark.parametrize('span_handoff', [False, True])
def test_saved_source_bodies_reach_tool_free_wire_in_both_paths(tmp_path, monkeypatch, arm, span_handoff):
    # Reuse the *same* public NoSkill state to compare retention, never its gold.
    ep = archived()/'episodes/02_UID0153/off'
    cfg = json.loads((ep/'config.json').read_text())
    if span_handoff:
        from test_evidence_handoff import configure
        configure(cfg)
    trace = json.loads((ep/'traces/officeqa:UID0153.json').read_text())
    saved = json.loads((ep/'checkpoints/officeqa:UID0153/1/state.json').read_text())
    old = next(q for q in trace['requests'] if q['purpose']=='finish_only')
    assert '42572912' not in json.dumps(old['http_attempts'][0]['final_payload_audit']['messages'])
    cfg['data_dir'] = str(tmp_path/'bank'); cfg['experiment']['output_dir'] = str(tmp_path/'run')
    bridge = load(ROOT/'integration/frontier.py', 'strong_finish_test')
    imported = bridge.native_import(json.loads((ROOT/'bank/bank.json').read_text()), cfg['data_dir'],
        cfg['program_environment'], cfg['experiment']['officeqa_method_version'])
    task = PublicTask(**trace['task']); adapter = OfficeAdapter({}, cfg); adapter.task = task
    context = TaskContext(cfg['runtime'])
    for k,v in saved['executor_state']['context'].items(): setattr(context, k, deepcopy(v))
    context.settings.update(cfg['runtime'])
    before = digest(context.results)
    seen = http(monkeypatch, [response(content='offline-fixture', finish='stop')])
    with bridge.bridge():
        system = EmpiricalSystem(cfg, harness=adapter, readonly=True,
            bank_view_factory=lambda b: bridge.HumanBankView(b, imported, arm))
        system.task_context = context; system.phase = 'test'
        try:
            material = deepcopy(saved['finish_prepared']['evidence']['materials'])
            material.pop('finish_evidence')
            prepared = system.prepare_finish(material, old['messages'][0]['content'])
            answer = system.agent('runtime', old['messages'][0]['content'], prepared['materials'],
                None, None, decision_purpose='finish_only')
            assert answer=='offline-fixture' and len(seen)==1
            wire = seen[0]; body = json.loads(wire['messages'][1]['content'])
            for value, date in [('42572912','June 30, 1995'),('42181150','June 30, 1996')]:
                w = next(w for w in body['finish_evidence'] if value in str(w['content']))
                assert date in w['content'] and 'thousands of dollars' in w['content']
                assert 'Assets, liabilities, and capital' in w['content'] and 'Total assets' in w['content']
                original = context.resolve_public(w['source_ref'])
                assert original[w['offset']:w['end']]==w['content']
            assert not wire.get('tools') and wire['thinking']['type']==('enabled' if span_handoff else 'disabled')
            assert wire['max_tokens']==(8192 if span_handoff else 2048)
            assert input_token_bound(wire)==prepared['serialized_input_bytes']<=(65536 if span_handoff else 16384)
            assert digest(context.results)==before
            output = os.environ.get('HUMANSTRONG_TEST_EVIDENCE')
            if output:
                root=Path(output);root.mkdir(parents=True,exist_ok=True)
                (root/f'UID0153_rebuilt_{arm}_{span_handoff}.json').write_text(json.dumps({'payload':wire,'receipt':prepared,
                    'source_context_hash':before,'model_http':0},ensure_ascii=False,indent=2))
        finally: system.close()


def test_recent_sources_displace_old_previews_but_keep_computation():
    c=TaskContext()
    old=c.register('old',{'accepted':True,'data':{'text':'obsolete body '*1200}},name='read_context',origin=TEXT_CONTEXT)
    c.sent_windows=c.displayed_windows(c.view(old))
    for name, text in [('a','Units: widgets\n| Date | Total |\n| 2040 | 123 |'),
                       ('b','Units: widgets\n| Date | Total |\n| 2041 | 456 |')]:
        c.register(name,{'accepted':True,'data':{'text':text}},name='read_context',arguments={'path':name},origin=TEXT_CONTEXT)
    result=c.register('calculation',{'accepted':True,'data':{'computed':{'operand_values':[123,456],'answer':333}},
                                   'worker_cpu_seconds':999},name='execute_python',origin=FILE_COMPUTE)
    evidence=build_finish_evidence(c,{'original_goal':'Compare totals','recent':[c.view(old)]},'finish',max_bytes=3500)
    windows=evidence['materials']['finish_evidence'];text=json.dumps(windows)
    assert '2040' in text and '2041' in text and '333' in text
    assert windows[0]['source_ref']['result_id']==result and 'worker_cpu_seconds' not in text
    restricted=build_finish_evidence(c,{'original_goal':'Compare totals'},'finish',max_bytes=3500,allowed_result_ids={old})
    assert all(w['source_ref']['result_id']==old for w in restricted['materials']['finish_evidence'])


def test_calculation_is_visible_after_real_read_without_optional_evidence_state(tmp_path, monkeypatch):
    cfg=json.loads((archived()/'episodes/01_UID0146/manual/config.json').read_text())
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'source.txt').write_text('Units: dollars\n| Year | Amount |\n| 2040 | 12 |\n| 2041 | 80 |')
    cfg['harness']['corpus_root']=str(corpus)
    bridge=load(ROOT/'integration/frontier.py','strong_visible_test')
    imported=bridge.native_import(json.loads((ROOT/'bank/bank.json').read_text()),cfg['data_dir'],
        cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    ps={Path(p['source_file']).stem:p for p in imported['program']}
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'read','arguments':{'path':'source.txt','offset':0}}]),
        response([{'action':'call_program','name':ps['series_metrics']['id'],
                   'arguments':{'operation':'ratio','numerator':12,'denominator':80,'output_percent':False,'unit':'ratio'}}]),
        response([{'action':'finish','answer':'0.15'}])])
    adapter=OfficeAdapter({'fixture':{'answer':'0.15'}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,imported))
        try:trace=system.run_task(PublicTask('fixture','fixture','Calculate the ratio of acquired amounts.',{},'test'),learn=False)
        finally:system.close()
    material=json.loads(seen[1]['messages'][1]['content'])
    assert material['human_office_progress']['phase']=='extract' and not material['human_office_evidence'].get('operands')
    assert ps['series_metrics']['id'] in [p['id'] for p in material['calls']['programs']]
    assert trace['execution']['attempts'][0]['status']=='ok' and trace['execution']['attempts'][0]['output_contract_status']=='valid'
    select=bridge.selection.callable_programs
    assert 'treasury_transform.py' in select('Convert yen currency amounts','extract',True,1)
    assert 'series_metrics.py' not in select('Calculate a ratio','locate',False,0)
    assert 'series_metrics.py' not in select('Calculate a ratio','extract',False,0)


def test_full_test_plan_uses_all_canonical_members_and_preserves_costs(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
    import run_office_humanstrong_test as runner
    monkeypatch.setattr(runner.common,'code_identity',lambda:{'git_sha':'offline','tracked_dirty':False,'source_sha256':'offline'})
    base=archived();root=tmp_path/'test';old=runner.common.read(base/'budget.json')
    runner.prepare(root,base,runner.common.read(base/'plan.json')['resources'])
    plan=runner.common.verify(root)
    assert len(plan['episodes'])==102 and all(e['arm']=='manual' for e in plan['episodes'])
    assert all(t['split']=='test' for t in runner.common.read(root/'tasks.json')['tasks'])
    assert runner.common.read(root/'budget.json')['attempts']==old['attempts']
    assert plan['new_token_cap']==19584000 and plan['authorized_cumulative_ceiling']==116000000
    assert plan['new_http_cap']==212670 and plan['ledger_limits']['token_limit']==115228535
    assert plan['development_exposed'] and plan['assets_unchanged_from_parent']
