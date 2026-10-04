"""Public input isolation, fixed scorer semantics and sealed spreadsheet variants."""
import json
import os
import sys
import pytest
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.harness.benchmarks import AnswerAdapter, OfficeAdapter, SpreadsheetAdapter, truncate_context
from test_empirical import config_for


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
    with pytest.raises(ValueError): adapter.call('read',{'path':'../gold.txt','offset':0})
    assert 'hidden gold' not in json.dumps(adapter.observe())
    assert not adapter.evaluate(adapter.submit('I succeeded'))['hard']
    adapter.close()


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
    adapter.reset(PublicTask('s','p','double A1 into B1'))
    solution="import openpyxl\nw=openpyxl.load_workbook(INPUT_PATH)\nw.active['B1']='=A1*2'\nw.save(OUTPUT_PATH)\nw.close()\n"
    source="from pathlib import Path\nPath('solution.py').write_text("+repr(solution)+")\nexec("+repr(solution)+")"
    result=adapter.call('execute_python',{'source':source,'files':['solution.py','case1_result.xlsx']})
    assert result['accepted'],result
    score=adapter.evaluate(adapter.submit(None))
    assert score['hard'] and score['case_results']==[True,True],score
    adapter.close()
