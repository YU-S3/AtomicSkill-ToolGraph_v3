"""Evidence survives table position, wire capacity and real sandbox boundaries."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import evidence_handoff as handoff
from atomic_skillgraph.empirical.contracts import digest,PublicTask
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.public_results import TEXT_CONTEXT,FILE_COMPUTE
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import office,http,response
from test_office_human_phase_v3 import load


def configure(cfg):
    cfg['runtime'].update(evidence_handoff_version=handoff.VERSION,finish_input_max_bytes=65536,dynamic_output_min_tokens=8192)
    cfg['llm'].setdefault('purpose_overrides',{})['finish_only']={'protocol':{'thinking_type':'enabled'},'max_completion_tokens':8192}
    return cfg


def test_table_tail_is_whole_and_sources_remain_immutable():
    context=TaskContext({'evidence_handoff_version':handoff.VERSION})
    text='Unrelated background.\n\n'*180+'Ledger 2040 and 2041\nUnits: widgets\n| Year | Target total |\n| --- | --- |\n'+''.join(f'| {1900+i} | {i} |\n' for i in range(100))+'| 2040 | 123 |\n| 2041 | 456 |\n'
    rid=context.register('source',{'accepted':True,'data':{'text':text,'next_offset':len(text)}},name='read',arguments={'path':'ledger.txt'},origin=TEXT_CONTEXT)
    before=digest(context.results)
    finish=build_finish_evidence(context,{'original_goal':'Compare target total in 2040 and 2041'},'finish',max_bytes=3200)
    body=json.dumps(finish['materials'])
    for expected in ['| 2040 | 123 |','| 2041 | 456 |','| Year | Target total |','Units: widgets']:assert expected in body
    assert digest(context.results)==before and finish['serialized_input_bytes']<=3200
    for w in finish['materials']['finish_evidence']:
        assert context.resolve_public(w['source_ref'])[w['offset']:w['end']]==w['content']
        assert w['content'].endswith('\n')
    assert rid in {w['source_ref']['result_id'] for w in finish['materials']['finish_evidence']}


def test_required_source_never_silently_becomes_a_partial_row():
    c=TaskContext({'evidence_handoff_version':handoff.VERSION})
    text='Units: widgets\n| Year | Amount |\n| --- | --- |\n| 2040 | '+('1'*3000)+' |\n'
    rid=c.register('s',{'accepted':True,'data':{'text':text}},name='read',origin=TEXT_CONTEXT)
    quote=text.splitlines()[-1]
    finish=handoff.prepare(c,{'original_goal':'read amount'},max_bytes=1000,output_cap=100,
        payload_builder=lambda v,cap:{'messages':[{'role':'user','content':json.dumps(v)}]},
        priority_refs=[{'ref':{'result_id':rid,'path':['data','text']},'quote':quote}])
    assert finish['required_evidence_missing'] and any(x['required'] for x in finish['omitted_refs'])
    assert finish['serialized_input_bytes']<=1000


def test_explicit_tail_read_survives_and_access_scope_is_enforced():
    c=TaskContext({'evidence_handoff_version':handoff.VERSION})
    rid=c.register('s',{'data':{'text':'x'*5000+'tail-value'}},name='read')
    c.read(rid,path=['data','text'],offset=5000,limit=10)
    finish=build_finish_evidence(c,{'original_goal':'use the last explicitly read value'},'finish',max_bytes=1800)
    assert 'tail-value' in json.dumps(finish['materials'])
    denied=build_finish_evidence(c,{'original_goal':'use the last explicitly read value'},'finish',max_bytes=1800,allowed_result_ids=set())
    assert not denied['materials']['finish_evidence'] and denied['unresolved_refs']


def test_required_output_cannot_be_silently_cut_to_read_window():
    c=TaskContext({'evidence_handoff_version':handoff.VERSION,'result_read_window_chars':2000})
    rid=c.reference('x'*3000,'bound_output')
    finish=build_finish_evidence(c,{'original_goal':'use bound output','pending_outputs':{'result':rid}},'finish')
    assert finish['required_evidence_missing']
    assert any(x['reason']=='whole_window_read_limit' and x['required'] for x in finish['omitted_refs'])
    assert not finish['materials']['finish_evidence']


@pytest.mark.parametrize('value,content',[(None,'null'),(False,'false'),([],[]),('','')])
def test_required_empty_output_retains_its_actual_value(value,content):
    c=TaskContext({'evidence_handoff_version':handoff.VERSION})
    ref=c.reference(value,'bound_output')
    finish=build_finish_evidence(c,{'original_goal':'use bound output','pending_outputs':{'x':ref}},'finish')
    assert not finish['required_evidence_missing']
    assert finish['materials']['finish_evidence'][0]['content']==content


def test_unicode_output_reference_keeps_exact_bound_span():
    c=TaskContext({'evidence_handoff_version':handoff.VERSION})
    text='unneeded prefix'*300+'引用"\\\n'*40
    ref=c.reference(text,'bound_output')
    window=c.result_window(ref,4500,len(text)-4500)
    finish=build_finish_evidence(c,{'original_goal':'use explicit output','completed_results':{'x':window}},'finish',max_bytes=3000)
    assert not finish['required_evidence_missing']
    assert finish['materials']['finish_evidence'][0]['content']==text[4500:]
    assert finish['serialized_input_bytes']<=3000


def test_plain_stdout_retains_value_and_truncation_receipt():
    c=TaskContext({'evidence_handoff_version':handoff.VERSION})
    receipt={'total_bytes':3,'retained_bytes':3,'limit_bytes':65536,'truncated':False}
    c.register('calculation',{'accepted':True,'data':{'stdout':'42\n','stdout_receipt':receipt}},
        name='execute_python',origin=FILE_COMPUTE)
    finish=build_finish_evidence(c,{'original_goal':'use calculation'},'finish')
    window=finish['materials']['finish_evidence'][0]
    assert window['content']=='42\n' and window['stdout_receipt']==receipt


@pytest.mark.parametrize('uid,values',[('UID0091',['820298','1011262']),('UID0072',['14025','5656','18922','7711']),('UID0132',['92055','73979','55922'])])
@pytest.mark.parametrize('arm',['off','manual'])
def test_saved_failed_cases_reach_actual_wire(tmp_path,monkeypatch,uid,values,arm):
    archive=os.environ.get('HUMANSTRONG_TEST102')
    if not archive:pytest.skip('Saved immutable Test102 needed')
    ep=next(Path(archive).glob('episodes/*_'+uid+'/manual'))
    cfg=configure(json.loads((ep/'config.json').read_text()))
    saved=json.loads(next((ep/'checkpoints').glob('*/1/state.json')).read_text())
    trace=json.loads(next((ep/'traces').glob('*.json')).read_text())
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    package=Path(__file__).parents[1]/'resources/office_humanstrong_v1'
    bridge=load(package/'integration/frontier.py','handoff_wire_fixture')
    imported=bridge.native_import(json.loads((package/'bank/bank.json').read_text()),cfg['data_dir'],cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    task=PublicTask(**trace['task']);adapter=OfficeAdapter({},cfg);adapter.task=task
    context=TaskContext(cfg['runtime'])
    for k,v in saved['executor_state']['context'].items():setattr(context,k,deepcopy(v))
    context.settings.update(cfg['runtime'])
    before=digest(context.results);sent=http(monkeypatch,[response(content='offline-fixture',finish='stop')])
    old=trace['requests'][-1]
    with bridge.bridge():
        s=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,imported,arm))
        s.task_context=context;s.phase='test'
        try:
            p=s.prepare_finish({'original_goal':task.goal},old['messages'][0]['content'])
            assert not p['required_evidence_missing']
            s.agent('runtime',old['messages'][0]['content'],p['materials'],None,None,decision_purpose='finish_only')
            body=json.dumps(sent[0]['messages'])
            assert all(v in body for v in values)
            assert sent[0]['thinking']['type']=='enabled' and sent[0]['max_tokens']==8192
            assert p['serialized_input_bytes']<=65536 and digest(context.results)==before
            for w in p['materials']['finish_evidence']:
                raw=context.resolve_public(w['source_ref'])
                if isinstance(raw,str):assert raw[w['offset']:w['end']]==w['content']
            target=os.environ.get('HANDOFF_EVIDENCE_DIR')
            if target:
                out=Path(target);out.mkdir(parents=True,exist_ok=True)
                (out/f'{uid}_{arm}.json').write_text(json.dumps({'payload':sent[0],'receipt':p,'actual_model_http':0},ensure_ascii=False,indent=2))
        finally:s.close()


def test_real_python_business_result_exceeds_diagnostic_limit(tmp_path):
    s=office(tmp_path);configure(s.config);s.adapter.config=s.config
    archive=os.environ.get('HUMANSTRONG_TEST102')
    if not archive:s.close();pytest.skip('Saved sandbox identity needed')
    cfg=json.loads(next(Path(archive).glob('episodes/*/manual/config.json')).read_text())
    s.config['program_environment']=cfg['program_environment']
    broker=Broker(s.adapter,6)
    try:
        result=s.adapter.call_with_broker('execute_python',{'source':'import json\nprint(json.dumps({"values":list(range(2500)),"answer":42}))'},broker,deadline=None)
        assert result['accepted'],result
        assert result['data']['computed']['answer']==42 and len(result['data']['computed']['values'])==2500
        assert result['data']['stdout_receipt']['retained_bytes']>4096 and not result['data']['stdout_receipt']['truncated']
        context=TaskContext(s.config['runtime']);rid=context.register('python',result,name='execute_python',origin=s.adapter.result_protocol('execute_python'))
        assert context.read(rid,path=['data','computed','values'],offset=2490,limit=40)['data']==list(range(2490,2500))
        over=s.adapter.call_with_broker('execute_python',{'source':'print("x"*70000)'},broker,deadline=None)
        assert over['accepted'] and over['data']['stdout_receipt']['truncated']
        assert 'computed' not in over['data'] and len(over['data']['stdout'].encode())==65536
    finally:s.close()


def test_budget_boundary_uses_one_existing_runtime_turn_with_tools(tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    from atomic_skillgraph.empirical.prompts import STEP
    s=office(tmp_path);configure(s.config);s.task_context.settings.update(s.config['runtime'])
    s.config['llm']['runtime'].update(max_total_tokens_per_task=30000,max_completion_tokens=32768)
    s.budget_governor=BudgetGovernor(tmp_path/'budget.json',token_limit=30000,request_limit=3,finish_reserve=0)
    s.request_attribution={'parent_scope':'fixture','parent_token_limit':30000};s.phase='test'
    text='Units: dollars\n| Year | Total |\n| --- | --- |\n| 2040 | 123 |\n'
    rid=s.task_context.register('source',{'accepted':True,'data':{'text':text}},name='read',origin=TEXT_CONTEXT)
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'execute_python','arguments':{'source':'print(123*2)'}}])])
    try:
        material={'task':{'goal':'Double the 2040 total'},'calls':{'tools':s.adapter.available_tools()},
            'memory':{'operations':['redundant history '*10000]},'recovery':{}}
        s.agent('runtime','Solve with acquired evidence',material,'runtime_step',STEP,repair_limit=0,
            finish_request_material={'original_goal':'Double the 2040 total'})
        assert len(seen)==1 and seen[0]['tools']
        body=json.loads(seen[0]['messages'][1]['content'])
        assert body['recovery']['budget_phase']=='consolidate'
        assert '123' in json.dumps(body['memory']['acquired_evidence'])
        assert rid in json.dumps(body) and body['task']['goal']==material['task']['goal']
        assert len(s.budget_governor.state['attempts'])==1
        assert s.requests[-1]['parent_finish_reserve']<20000
    finally:s.close()
