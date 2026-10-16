from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical import office_human_policy as p
from atomic_skillgraph.empirical.contracts import PublicTask, validate_schema_instance, program_digest
from atomic_skillgraph.empirical.public_results import TEXT_CONTEXT, PROGRAM_EXECUTION
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.system import EmpiricalSystem

ROOT=Path(__file__).parents[1]/'resources/office_human_frontier_v3'

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture
def bundle():return json.loads((ROOT/'bank/bank.json').read_text())

@pytest.fixture
def bridge():return load(ROOT/'integration/frontier.py','office_frontier_v3_test')

def test_native_import_and_no_dangling_references(bundle,bridge,tmp_path):
    data=bridge.native_import(bundle,tmp_path/'bank',bundle['program'][0]['environment'])
    bank=Bank(tmp_path/'bank',readonly=True,environment=bundle['program'][0]['environment'])
    try:
        members={a['id'] for kind in bridge.KINDS for a in bank.all(kind)}
        for a in data['implementation']:
            assert a['skill_id'] in members and a['program_id'] in members
        for w in data['workflow']:
            for n in w['nodes']:
                assert all(n[k] in members for k in ['skill_id','program_id'] if n.get(k))
                assert all(s in members for s in n.get('reference_skill_ids',[]))
        task=PublicTask('synthetic','synthetic','Find the requested percentage.',{},'train')
        view=bridge.HumanBankView(bank,data);view.bind_task(task)
        for phase in ['locate','extract','compute','finalize']:
            assert 1<=len(view.selected(task,phase))<=2
            assert '<answer>' not in view.policy(task,phase)
        off=bridge.HumanBankView(bank,data,'off')
        assert not off.selected(task) and not off.program_options()
        assert all(a['id']=='program_'+program_digest(a) for a in data['program'])
    finally:bank.close()

@pytest.mark.parametrize('args,answer',[
    ({'operation':'ratio','numerator':12,'denominator':80,'output_percent':True},'15'),
    ({'operation':'ratio','numerator':80,'denominator':12,'output_percent':False,'decimals':3},'6.667'),
    ({'operation':'difference','from_value':20,'to_value':5,'absolute':False},'-15'),
    ({'operation':'difference','from_value':20,'to_value':5,'absolute':True},'15'),
    ({'operation':'percent_change','base_value':80,'new_value':100,'output_percent':True},'25'),
    ({'operation':'percent_change','base_value':80,'new_value':100,'output_percent':False},'0.25'),
    ({'operation':'cagr','base_value':100,'new_value':121,'periods':2,'output_percent':False,'decimals':2},'0.10'),
    ({'operation':'mean','values':[1,3,8]},'4'),
])
def test_named_arithmetic(bundle,args,answer):
    a=next(x for x in bundle['program'] if x['source_file'].endswith('series_metrics.py'))
    args={**args,'unit':'specified_by_caller'}
    validate_schema_instance(args,a['input_schema'])
    result=load(ROOT/a['source_file'],'metrics').run(None,args)
    assert result['status']=='ok' and result['outputs']['answer']==answer
    validate_schema_instance(result['outputs'],a['output_schema'])
    assert result['outputs']['proof_scope']=='local_computation_only'

def test_legacy_positional_operands_are_rejected(bundle):
    a=next(x for x in bundle['program'] if x['source_file'].endswith('series_metrics.py'))
    with pytest.raises(ValueError):
        validate_schema_instance({'operation':'ratio','values':[2,9],'output_scale':100},a['input_schema'])
    assert load(ROOT/a['source_file'],'metrics').run(None,{'operation':'ratio','values':[2,9]})['status']=='needs_input'

def test_table_missing_unit_and_ambiguous_header(bundle):
    mod=load(ROOT/'programs/table_extract.py','table')
    args={'text':'Millions of dollars\n| Year | Revenue |\n| 1990 | 12 |\n| 1991 | 15 |',
          'rows':['1990','1991'],'column':'Revenue'}
    assert mod.run(None,args)['status']=='needs_input'
    result=mod.run(None,{**args,'unit':'Millions of dollars'})
    assert result['outputs']['values']==['12','15']
    assert result['outputs']['target_periods_complete'] and result['outputs']['table_header_bound']
    assert mod.run(None,{**args,'unit':'Millions of dollars','text':args['text']+'\n| Year | Revenue |'})['status']=='needs_input'

def test_retrieval_exhaustion_does_not_claim_task_completion():
    calls=[]
    class Ctx:
        def available_tools(self):return [{'name':n} for n in ['grep_page','read_context']]
        def remaining_calls(self):return 8-len(calls)
        def call(self,n,a):
            calls.append((n,a))
            return {'accepted':True,'data':{'hits':[{'path':'real.txt','offset':120}], 'complete':True,'next_cursor':None}
                if n=='grep_page' else {'start_offset':100,'text':'An acquired source window'}}
    result=load(ROOT/'programs/scoped_search.py','search').run(Ctx(),{'query':'revenue','paths':['real.txt']})
    assert result['status']=='needs_input' and result['outputs']['search_exhausted']
    assert result['outputs']['target_periods_complete'] is None and not result['outputs']['table_header_bound']
    assert calls[1][1]['path']=='real.txt' and calls[1][1]['offset']==120

def context():
    c=TaskContext()
    r=c.register('page',{'accepted':True,'data':{'text':'Millions. 1990 A 12; 1991 B 80.'}},name='read_context',origin=TEXT_CONTEXT)
    out=c.register('calculation',{'status':'ok','outputs':{'answer':'0.15','unit':'ratio','formula':'numerator/denominator'}},
        name='actual_program',arguments={'operation':'ratio','numerator':12,'denominator':80},origin=PROGRAM_EXECUTION)
    operands=[{'name':n,'value':v,'period':period,'unit':'millions','source_ref':{'result_id':r,'path':['data','text']},
               'quote':quote} for n,v,period,quote in [('numerator',12,'1990','1990 A 12'),('denominator',80,'1991','1991 B 80')]]
    candidate={'value':'0.15','operation':'ratio','output_unit':'ratio','result_ref':{'result_id':out,'path':['outputs','answer']},'covers_task':True}
    return c,operands,candidate

def test_candidate_requires_real_complete_matching_operands():
    c,operands,candidate=context()
    good=p.validate_state(c,{'operands':operands,'answer_candidate':candidate})
    assert good['answer_candidate']['ready'] and good['answer_candidate']['proof_scope']=='reference_integrity_only'
    assert not p.validate_state(c,{'operands':operands[:1],'answer_candidate':candidate})['answer_candidate']['ready']
    changed=deepcopy(operands);changed[0]['value']=999
    assert not p.validate_state(c,{'operands':changed,'answer_candidate':candidate})['answer_candidate']['ready']
    changed=deepcopy(candidate);changed['value']='100'
    assert not p.validate_state(c,{'operands':operands,'answer_candidate':changed})['answer_candidate']['ready']
    changed=deepcopy(operands);changed[0]['name']='other'
    assert not p.validate_state(c,{'operands':changed,'answer_candidate':candidate})['answer_candidate']['ready']

def test_phase_uses_obtained_evidence_not_goal_words():
    assert p.progress(None,{})['phase']=='locate'
    c,operands,candidate=context()
    assert p.progress(c,{})['phase']=='extract'
    state=p.validate_state(c,{'operands':operands})
    assert p.progress(c,state)['phase']=='compute'
    state=p.validate_state(c,{'answer_candidate':candidate},state)
    assert p.progress(c,state)['phase']=='finalize'

def test_finish_keeps_candidate_and_quotes_before_directory_noise():
    c,operands,candidate=context();state=p.validate_state(c,{'operands':operands,'answer_candidate':candidate})
    for i in range(40):
        c.register(str(i),{'accepted':True,'data':['irrelevant_filename_'*80]*40},name='glob',arguments={'pattern':str(i)})
    wire=lambda v,cap:{'messages':[{'role':'system','content':'finish'}, {'role':'user','content':json.dumps(v)}],'max_tokens':cap}
    result=p.finish_evidence(c,{'original_goal':'Compute A/B'},'finish',max_bytes=6000,output_cap=2048,
        payload_builder=wire,state=state,policy='Commit only the actual target')
    assert result['serialized_input_bytes']<=6000
    assert result['materials']['answer_candidate']['ready']
    text=json.dumps(result['materials']['finish_evidence'])
    assert '1990 A 12' in text and '1991 B 80' in text
    assert result['included_refs'][0]['source_ref']==candidate['result_ref']
    assert result['omitted_refs']

def test_finish_bridge_does_not_append_after_fitting(bridge,monkeypatch):
    seen=[]
    monkeypatch.setattr(EmpiricalSystem,'agent',lambda *a,**kw:seen.append((a,kw)))
    v=bridge.HumanBankView.__new__(bridge.HumanBankView);v.arm='manual'
    s=SimpleNamespace(bank=v);material={'finish_evidence':[{'content':'bound'}]}
    with bridge.bridge():
        EmpiricalSystem.agent(s,'runtime','exact native prompt',material,None,None,decision_purpose='finish_only')
    assert seen[0][0][2]=='exact native prompt' and seen[0][0][3] is material


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
    assert trace['execution']['attempts'][0]['status']=='needs_input'
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
