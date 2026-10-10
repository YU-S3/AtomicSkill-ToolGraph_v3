"""Public input isolation, fixed scorer semantics and sealed spreadsheet variants."""
import json
import os
import sys
import pytest
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema
from atomic_skillgraph.harness.simple_protocol import Broker
from atomic_skillgraph.harness.benchmarks import AnswerAdapter, OfficeAdapter, SpreadsheetAdapter, truncate_context
from test_empirical import config_for, Provider, worker


def test_public_qa_does_not_contain_gold_and_scoring_is_independent():
    records={'searchqa:1':{'answers':['New York']}}
    adapter=AnswerAdapter('searchqa',records)
    task=PublicTask('searchqa:1','p','question',{'context':'public documents'})
    adapter.reset(task)
    assert 'New York' not in json.dumps(adapter.observe())
    assert not adapter.evaluate(adapter.submit('I succeeded'))['hard']
    assert adapter.evaluate(adapter.submit('<answer>New York</answer>'))['hard']
    assert truncate_context('a'*5900+'[DOC]'+'b'*300)=='a'*5900


def test_docvqa_threshold_and_livemath_choice_parsing():
    doc=AnswerAdapter('docvqa',{'d':{'answers':['abcd']}})
    doc.reset(PublicTask('d','p','question'))
    assert doc.evaluate(doc.submit('abxx'))['soft']==0
    assert not doc.evaluate(doc.submit('abce'))['hard']
    choices=[{'label':'A','text':'first'},{'label':'B','text':'second'}]
    math=AnswerAdapter('livemath',{'m':{'choices':choices,'correct_choice':choices[1]}})
    math.reset(PublicTask('m','p','question',{'choices':choices}))
    assert math.evaluate(math.submit('<answer>B.</answer>'))['hard']


def test_office_has_full_corpus_and_cannot_read_outside_it(tmp_path):
    corpus=tmp_path/'corpus'; corpus.mkdir()
    (corpus/'unrelated.txt').write_text('public unrelated material')
    (corpus/'source.txt').write_text('public answer evidence')
    gold=tmp_path/'gold.txt'; gold.write_text('private')
    config=config_for(tmp_path/'bank'); config['harness']['corpus_root']=str(corpus)
    adapter=OfficeAdapter({'o':{'answer':'hidden gold'}},config)
    adapter.reset(PublicTask('o','p','question'))
    assert adapter.call('glob',{'pattern':'*.txt'})['data']==['source.txt','unrelated.txt']
    refused = adapter.call('read',{'path':'../gold.txt','offset':0})
    assert not refused['accepted'] and refused['error_code'] == 'invalid_corpus_input'
    assert 'private' not in json.dumps(refused) and str(tmp_path) not in json.dumps(refused)
    assert 'hidden gold' not in json.dumps(adapter.observe())
    assert not adapter.evaluate(adapter.submit('I succeeded'))['hard']
    adapter.close()


def test_office_learner_builder_and_program_share_public_tools(tmp_path, worker, monkeypatch):
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    corpus=tmp_path/'corpus'; corpus.mkdir()
    (corpus/'source.txt').write_text('public evidence')
    config=config_for(tmp_path/'bank'); config['harness']['corpus_root']=str(corpus)
    adapter=OfficeAdapter({'o':{'answer':'private gold'}},config)
    task=PublicTask('o','physical','read public evidence',{'pattern':'*.txt'})
    adapter.reset(task)
    skill={'goal':task.goal,'input_schema':object_schema({'pattern':{'type':'string'}},['pattern']),
           'output_schema':object_schema({'text':{'type':'string'}},['text'])}
    source="""def run(ctx, inputs):
    paths = ctx.call('glob', {'pattern': inputs['pattern']})['data']
    text = ctx.call('read', {'path': paths[0], 'offset': 0})['data']['text']
    return {'status': 'ok', 'outputs': {'text': text}}
"""
    broker=Broker(adapter,2);broker.call('read',{'path':'source.txt','offset':0})
    from atomic_skillgraph.empirical.local_validation import evidence_from_trace
    config['program_environment']['adapter_abi']='simple.v2'
    event=evidence_from_trace(task,{'tools':broker.events},config['program_environment'])[0]
    binding={'case_id':task.physical_key,'inputs':{'pattern':'*.txt'},'start_mode':'reset','prefix':[],
        'local_evidence_ref':event['id'],'reference_fields':{'text':['data','text']}}
    provider=Provider([{'decision':'propose_skill_and_program_spec','skill':skill,
        'realization_request':{'skill_id':'$new','action':'build','case_bindings':[
            binding]}},
        {'source':source,'trial_inputs':[binding]}])
    system=EmpiricalSystem(config,harness=adapter,provider=provider)
    monkeypatch.setattr(system,'test_program',lambda *a,**kw:{'outcome':'inapplicable'})
    try:
        log=system.learner.learn(task,{'tools':broker.events,'score':{},'execution':{}})
        for index,messages in enumerate(provider.calls):
            material=json.loads(messages[1]['content'])
            tools=material['tools'] if index==0 else material['future_program_api']['runtime_tool_definitions']
            expected={'glob','read','grep','read_result'} | ({'execute_python'} if index==0 else set())
            assert {t['name'] for t in tools}==expected
        program=system.bank.get(log['program'])
        assert set(program['allowed_tools'])=={'glob','read','grep','read_result'}
        broker=Broker(adapter,2)
        result=worker.execute(program,{'pattern':'*.txt'},broker)
        assert result['status']=='ok' and result['outputs']=={'text':'public evidence'},result
        assert [e['name'] for e in broker.events]==['glob','read']
        assert program['state']=='candidate'
    finally:
        system.close()


def test_spreadsheet_seals_solution_and_replays_variants_without_gold(tmp_path):
    if sys.platform!='linux' or not os.environ.get('PROGRAM_IMAGE_DIGEST'):
        pytest.skip('requires locked container')
    import openpyxl
    cases=[]
    for index,value in enumerate([2,7]):
        input_path=tmp_path/f'{index}_input.xlsx'; gold=tmp_path/f'{index}_answer.xlsx'
        wb=openpyxl.Workbook(); wb.active['A1']=value; wb.save(input_path)
        wb.active['B1']=value*2; wb.save(gold); wb.close()
        cases.append({'input':str(input_path),'gold':str(gold)})
    records={'s':{'public_files':{'input.xlsx':cases[0]['input']},'cases':cases,
                  'instruction_type':'Cell-Level Manipulation','answer_position':'Sheet!B1'}}
    config=config_for(tmp_path/'bank'); config['program_worker'].update(wall_timeout_seconds=120,memory_limit_mb=2048)
    adapter=SpreadsheetAdapter(records,config)
    assert [t['name'] for t in adapter.tool_definitions()]==['execute_python']
    adapter.reset(PublicTask('s','p','double A1 into B1'))
    solution="import openpyxl\nw=openpyxl.load_workbook(INPUT_PATH)\nw.active['B1']='=A1*2'\nw.save(OUTPUT_PATH)\nw.close()\n"
    source="from pathlib import Path\nPath('solution.py').write_text("+repr(solution)+")\nexec("+repr(solution)+")"
    result=adapter.call('execute_python',{'source':source,'files':['solution.py','case1_result.xlsx']})
    assert result['accepted'],result
    score=adapter.evaluate(adapter.submit(None))
    assert score['hard'] and score['case_results']==[True,True],score
    adapter.close()
