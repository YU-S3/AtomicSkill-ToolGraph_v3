from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import office_human_policy_v31 as p
from atomic_skillgraph.empirical.contracts import PublicTask, validate_schema_instance
from atomic_skillgraph.empirical.public_results import TEXT_CONTEXT, PROGRAM_EXECUTION, FILE_COMPUTE
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.system import EmpiricalSystem
from test_office_human_phase_v3 import load, context, test_native_import_and_no_dangling_references

ROOT=Path(__file__).parents[1]/'resources/office_human_frontier_v31'

@pytest.fixture
def bundle():return json.loads((ROOT/'bank/bank.json').read_text())

@pytest.fixture
def bridge():return load(ROOT/'integration/frontier.py','office_frontier_v31_test')

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Real HTTP forbidden in offline acceptance')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)

def series(count=10):
    c=TaskContext();rows=[]
    for i in range(count):
        year=1952+i;value=100+i;text=f'FY{year}: {value} millions'
        rid=c.register(str(i),{'accepted':True,'data':{'text':text}},name='read_context',arguments={'year':year},origin=TEXT_CONTEXT)
        rows.append({'name':f'values.{i}','value':value,'period':f'FY{year}','unit':'millions',
            'source_ref':{'result_id':rid,'path':['data','text']},'quote':text})
    return c,rows

def test_incremental_years_dedup_and_partial_coverage():
    c,rows=series();task=PublicTask('synthetic','synthetic','Sum revenue for fiscal years 1952–1961.',{},'train')
    state=p.validate_state(c,{'operands':rows[:3]},task=task)
    assert p.period_coverage(task,state)['coverage_status']=='partial'
    assert p.period_coverage(task,state)['missing_periods']==[f'FY{y}' for y in range(1955,1962)]
    state=p.validate_state(c,{'operands':rows[3:]},state,task=task)
    state=p.validate_state(c,{'operands':rows},state,task=task)
    assert len(state['operands'])==10
    assert p.period_coverage(task,state)['coverage_status']=='complete'
    assert p.period_coverage(PublicTask('s','s','Compare 1952 and 1961 reports.',{},'train'),state)['coverage_status']=='unknown'
    task=PublicTask('synthetic','synthetic','Sum revenue for calendar years 1952–1961.',{},'train')
    assert len(p.period_coverage(task,state)['missing_periods'])==10

def conflicting(c,row):
    other=deepcopy(row);other['value']=999
    text='A different observation 999.'
    rid=c.register('conflict',{'accepted':True,'data':{'text':text}},name='read_context',arguments={'conflict':1},origin=TEXT_CONTEXT)
    other.update(source_ref={'result_id':rid,'path':['data','text']},quote=text)
    return other

def test_conflict_requires_explicit_correction_and_capacity_feedback():
    c,rows=series(41);first=rows[0];state=p.validate_state(c,{'operands':[first]})
    other=conflicting(c,first)
    state=p.validate_state(c,{'operands':[other]},state)
    assert [x['value'] for x in state['operands']]==[100,999] and state['operand_conflicts']
    correction={**other,'replaces_source_ref':first['source_ref']}
    state=p.validate_state(c,{'operands':[correction]},state)
    assert len(state['operands'])==1 and state['operands'][0]['value']==999 and not state['operand_conflicts']
    state=p.validate_state(c,{'operands':rows[:40]})
    over=p.validate_state(c,{'operands':rows[40:]},state)
    assert over['operands']==state['operands'] and 'capacity' in str(over['binding_feedback'])

def test_only_executed_operands_bind_and_unrelated_updates_preserve_candidate():
    c,operands,candidate=context()
    state=p.validate_state(c,{'operands':operands,'answer_candidate':candidate})
    unrelated={**operands[0],'name':'historical_other','period':'FY2000'}
    updated=p.validate_state(c,{'operands':[unrelated,*operands]},state)
    assert updated['answer_candidate']['ready'] and len(updated['operands'])==3
    bad=conflicting(c,operands[0])
    blocked=p.validate_state(c,{'operands':[bad]},updated)
    assert not blocked['answer_candidate']['ready']
    chosen={**candidate,'operand_bindings':{'numerator':operands[0]['source_ref']}}
    assert p.validate_state(c,{'answer_candidate':chosen},blocked)['answer_candidate']['ready']
    c.results.pop(operands[0]['source_ref']['result_id'])
    assert not p.validate_state(c,{},updated)['answer_candidate']['ready']

def test_partial_year_range_blocks_ready_without_certifying_metric_semantics():
    c,rows=series(3);rid=c.register('sum',{'status':'ok','outputs':{'answer':303}},name='sum',
        arguments={'operation':'sum','values':[100,101,102]},origin=PROGRAM_EXECUTION)
    candidate={'value':303,'output_unit':'millions','operation':'sum','covers_task':True,
               'result_ref':{'result_id':rid,'path':['outputs','answer']}}
    task=PublicTask('s','s','Sum for fiscal years from 1952 through 1961 inclusive.',{},'train')
    state=p.validate_state(c,{'operands':rows,'answer_candidate':candidate},task=task)
    assert not state['answer_candidate']['ready'] and 'incomplete_observation_periods' in state['binding_feedback']

def test_actual_scalar_paths_and_non_scalar_feedback():
    c,operands,candidate=context()
    rid=c.register('python',{'accepted':True,'data':{'computed':{'calculation':{
        'operand_values':{'numerator':12,'denominator':80},'answer':0.15}}}},name='execute_python',origin=FILE_COMPUTE)
    refs=p.recent_scalar_result_refs(c)
    assert any(r['path']==['data','computed','calculation','answer'] for r in refs)
    assert any(r['path']==['outputs','answer'] for r in refs)
    assert len(refs)<=6 and all(c.resolve_public({'result_id':r['result_id'],'path':r['path']})==r['display_value'] for r in refs)
    candidate.update(value=0.15,result_ref={'result_id':rid,'path':['data','computed']})
    bad=p.validate_state(c,{'operands':operands,'answer_candidate':candidate})
    assert 'non_scalar_result_reference' in str(bad['binding_feedback']) and bad['scalar_reference_choices']
    candidate['result_ref']['path']=['data','computed','calculation','answer']
    assert p.validate_state(c,{'answer_candidate':candidate},bad)['answer_candidate']['ready']
    candidate['result_ref']['path']=['outputs','answer']
    assert not p.validate_state(c,{'answer_candidate':candidate},bad)['answer_candidate']['ready']

def test_local_retrieval_ok_is_not_task_completeness():
    class Ctx:
        def available_tools(self):return [{'name':n} for n in ['grep_page','read_context']]
        def remaining_calls(self):return 8
        def call(self,n,a):return {'accepted':True,'data':{'hits':[{'path':'actual.txt','offset':100}],
            'complete':False,'next_cursor':'actual_cursor'} if n=='grep_page' else {'start_offset':90,'text':'A real window'}}
    result=load(ROOT/'programs/scoped_search.py','v31_search').run(Ctx(),{'query':'amount','max_pages':1})
    assert result['status']=='ok' and result['outputs']['scope']=='retrieval_window_only'
    assert not result['outputs']['search_exhausted'] and result['outputs']['target_periods_complete'] is None
    c=TaskContext();c.register('retrieval',result,name='retrieval_program',origin=PROGRAM_EXECUTION)
    assert p.progress(c,{})['phase']=='extract' and not p.progress(c,{})['candidate_ready']

def test_retrieval_read_failure_is_not_success():
    class Ctx:
        def available_tools(self):return [{'name':n} for n in ['grep_page','read_context']]
        def remaining_calls(self):return 8
        def call(self,n,a):return {'accepted':True,'data':{'hits':[{'path':'actual.txt','offset':0}],
            'complete':True,'next_cursor':None}} if n=='grep_page' else {'accepted':False,'error':'actual read failure'}
    result=load(ROOT/'programs/scoped_search.py','v31_failed_read').run(Ctx(),{'query':'amount'})
    assert result['status']=='blocked' and 'actual read failure' in result['detail']

@pytest.mark.parametrize('answer',['0.92','-12.30','[1952, 1953]','June 30, 1996'])
def test_native_finish_keeps_payload_without_scaling_or_tags(bundle,bridge,tmp_path,monkeypatch,answer):
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    from test_cf2_contracts import http,response
    archive=os.environ.get('OFFICE_HUMAN_ARCHIVE')
    if not archive:pytest.skip('Set OFFICE_HUMAN_ARCHIVE for locked production settings')
    cfg=json.loads((Path(archive)/'config.json').read_text());cfg['data_dir']=str(tmp_path/'bank')
    cfg['experiment']['output_dir']=str(tmp_path/'run')
    imported=bridge.native_import(bundle,tmp_path/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    seen=http(monkeypatch,[response([{'action':'finish','answer':answer}])])
    adapter=OfficeAdapter({'s':{'answer':answer}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,imported))
        try:trace=system.run_task(PublicTask('s','s','Return the requested value.',{},'train'),learn=False)
        finally:system.close()
    assert trace['execution']['prediction']==answer and trace['score']['hard']
    assert p.GLOBAL_SUBMISSION_RULE in seen[0]['messages'][0]['content']
    assert len(json.loads(seen[0]['messages'][1]['content'])['human_frontier_skill_ids'])<=2

def test_archived_computation_in_actual_runtime_and_fitted_finish(bundle,bridge,tmp_path,monkeypatch):
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    from atomic_skillgraph.empirical.prompts import finish_text_prompt
    from test_cf2_contracts import http,response
    archive=os.environ.get('OFFICE_PHASE_V3_ARCHIVE')
    if not archive:pytest.skip('Set OFFICE_PHASE_V3_ARCHIVE for real calculation replay')
    ep=Path(archive)/'episodes/02_UID0153/manual';cfg=json.loads((ep/'config.json').read_text())
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    trace=json.loads((ep/'summary.json').read_text())['cases'][0];task=PublicTask(**trace['task'])
    saved=json.loads((ep/'checkpoints'/task.task_id/'1/state.json').read_text())
    original_requests=json.loads((ep/'requests'/f'{task.task_id}_attempt1.json').read_text())['requests']
    q=original_requests[-1];material=json.loads(q['messages'][1]['content'])
    imported=bridge.native_import(bundle,tmp_path/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    adapter=OfficeAdapter({},cfg);adapter.task=task
    seen=http(monkeypatch,[response([{'action':'finish','answer':'0.92'}]),response(content='0.92',finish='stop')])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,imported))
        system.phase='train';system.task_context=TaskContext(cfg['runtime'])
        for k,v in saved['executor_state']['context'].items():setattr(system.task_context,k,deepcopy(v))
        try:
            system.agent('runtime',q['messages'][0]['content'],material,'runtime_step',
                q['tools'][0]['function']['parameters'],decision_purpose='runtime')
            sent=json.loads(seen[0]['messages'][1]['content'])
            assert p.GLOBAL_SUBMISSION_RULE in seen[0]['messages'][0]['content']
            assert any(r['path']==['data','computed','answer_pct_2dp'] for r in sent['recent_scalar_result_refs'])
            prompt=finish_text_prompt(adapter.answer_contract())
            prepared=system.prepare_finish({'original_goal':task.goal},prompt)
            system.agent('runtime',prompt,prepared['materials'],None,None,decision_purpose='finish_only')
            actual=json.loads(seen[-1]['messages'][1]['content'])
            assert p.GLOBAL_SUBMISSION_RULE in actual['authored_commit_policy']
            assert any(r['path']==['data','computed','answer_pct_2dp'] for r in actual['recent_scalar_result_refs'])
            assert input_token_bound(seen[-1])==prepared['serialized_input_bytes']<=16384
            assert actual['answer_candidate'] is None  # no fabricated historic ready candidate
        finally:system.close()


def test_native_tool_loop_docker_and_intercepted_http(bundle,bridge,tmp_path,monkeypatch):
    import os
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    from test_cf2_contracts import http,response
    archive=os.environ.get('OFFICE_HUMAN_ARCHIVE')
    if not archive:pytest.skip('Set OFFICE_HUMAN_ARCHIVE for the real locked Docker/config regression')
    cfg=json.loads((Path(archive)/'config.json').read_text())
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    corpus=tmp_path/'corpus';corpus.mkdir()
    text='Millions of dollars\n| Year | Revenue |\n| 1990 | 12 |\n| 1991 | 80 |\n'
    (corpus/'sample.txt').write_text(text)
    cfg['harness']['corpus_root']=str(corpus)
    imported=bridge.native_import(bundle,tmp_path/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    programs={Path(a['source_file']).stem:a for a in imported['program']}
    responses=[response([{'action':'call_program','name':programs['scoped_search']['id'],
        'arguments':{'query':'Revenue','paths':['sample.txt']}}]),
        response([{'action':'call_program','name':programs['table_extract']['id'],
            'arguments':{'text':text,'rows':['1990','1991'],'column':'Revenue','unit':'Millions of dollars'}}]),
        response([{'action':'call_program','name':programs['series_metrics']['id'],
            'arguments':{'operation':'ratio','numerator':12,'denominator':80,'output_percent':False,'unit':'ratio'}}]),
        response([{'action':'finish','answer':'0.15'}])]
    seen=http(monkeypatch,responses)
    adapter=OfficeAdapter({'synthetic':{'answer':'0.15'}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,imported))
        try:trace=system.run_task(PublicTask('synthetic','synthetic','Calculate the ratio of Revenue in 1990 to 1991.',{},'test'),learn=False)
        finally:system.close()
    assert trace['score']['hard'] and trace['execution']['reason']=='agent_submitted'
    assert len(trace['execution']['attempts'])==3 and len(seen)==4
    phases=[json.loads(q['messages'][1]['content'])['human_office_progress']['phase'] for q in seen]
    assert phases[:3]==['locate','extract','compute']
    assert trace['execution']['attempts'][0]['status']=='ok'
    assert all(a['status']=='ok' for a in trace['execution']['attempts'][1:])
    assert all(len(json.loads(q['messages'][1]['content'])['human_frontier_skill_ids'])<=2 for q in seen)


def test_native_forced_finish_matches_prepared_wire(bundle,bridge,tmp_path,monkeypatch):
    import os
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    from test_cf2_contracts import http,response
    archive=os.environ.get('OFFICE_HUMAN_ARCHIVE')
    if not archive:pytest.skip('Set OFFICE_HUMAN_ARCHIVE for the locked config regression')
    cfg=json.loads((Path(archive)/'config.json').read_text())
    cfg['data_dir']=str(tmp_path/'bank');cfg['experiment']['output_dir']=str(tmp_path/'run')
    from atomic_skillgraph.harness.simple_protocol import Broker
    remaining=Broker.remaining_calls
    # Emulate arriving at the last available native call; preserve the locked config.
    monkeypatch.setattr(Broker,'remaining_calls',lambda b:min(remaining(b),max(0,1-len(b.events))))
    corpus=tmp_path/'corpus';corpus.mkdir();(corpus/'sample.txt').write_text('Revenue in millions: 1990 12; 1991 80.')
    cfg['harness']['corpus_root']=str(corpus)
    imported=bridge.native_import(bundle,tmp_path/'bank',cfg['program_environment'],cfg['experiment']['officeqa_method_version'])
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'read','arguments':{'path':'sample.txt','offset':0}}]),
        response(content='0.15',finish='stop')])
    adapter=OfficeAdapter({'synthetic':{'answer':'0.15'}},cfg)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,imported))
        try:trace=system.run_task(PublicTask('synthetic','synthetic','Ratio of 1990 to 1991 revenue.',{},'test'),learn=False)
        finally:system.close()
    assert trace['execution']['reason']=='finish_only' and len(seen)==2
    evidence=trace['execution']['finish_evidence']
    assert input_token_bound(seen[-1])==evidence['serialized_input_bytes']<=16384
    assert evidence['projection_version']==p.VERSION
    assert not seen[-1].get('tools') and seen[-1]['thinking']['type']=='disabled'
    assert '1990 12' in json.loads(seen[-1]['messages'][1]['content'])['finish_evidence'][0]['content']
