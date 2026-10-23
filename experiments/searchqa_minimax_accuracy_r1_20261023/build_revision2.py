"""Independent follow-up: restore the title-gap rule and foreground the actual clue."""
import json
import shutil
from pathlib import Path
import build_bank as b

HERE=b.HERE
SOURCE=HERE/'packages/minimax_r1'
DEST=HERE/'packages/minimax_r2'
VERSION='human.searchqa.minimax.r2.20261023'
GAP='''For a title or quotation with its beginning or ending already supplied, return EXACTLY the missing span, not the full work, its author, performer or plot. Substitute the span back into the given words once. Preserve necessary words inside the blank, including pronouns; do not add words outside it. If a complete title itself is requested, retain its necessary words instead. A title plus year with no blank remains a relation-inference task, not automatically a director question.'''


def build():
    assert not DEST.exists()
    shutil.copytree(SOURCE,DEST)
    # Re-emit assets under distinct IDs; preserve parent and all previous results.
    bundle=json.loads((SOURCE/'bank/bank.json').read_text().replace('minimax_r1','minimax_r2').replace('human.searchqa.minimax.r1.20261023',VERSION))
    bundle['parent_method_version']='human.searchqa.minimax.r1.20261023'
    bundle['development_exposure']+=' R2 restores an explicit title-gap rule from V7 and repeats the exact public clue after the sources; it does not add answer memory.'
    for asset in bundle['skill']:
        if asset['id']=='skill_hs_searchqa_solve_minimax_r2':
            asset['guidance']='These are Jeopardy-style short-answer clues; their original category may be omitted. Answer the clue as a contestant would, not as a general chat request.\n\n'+asset['guidance']+'\n\n'+GAP
            (DEST/'guidance/solve.txt').write_text(asset['guidance']+'\n')
    for kind,folder in [('skill','skills'),('program','program_assets'),('implementation','implementations'),('workflow','workflows')]:
        for old in (DEST/folder).glob('*.json'):old.unlink()
        for asset in bundle[kind]:b.write(DEST/folder/(asset['id']+'.json'),asset)
    b.write(DEST/'bank/bank.json',bundle)
    text=(SOURCE/'integration.py').read_text().replace('minimax_r1','minimax_r2').replace('human.searchqa.minimax.r1.20261023',VERSION)
    marker='                call_options = dict(kwargs)'
    assert text.count(marker)==1
    text=text.replace(marker,'''                text_material["final_question_to_answer"] = value["task"]["goal"]
                text_material["final_submission_instruction"] = "Fill only the unknown requested by this exact clue. Sources are evidence, not additional questions. Return one <answer>...</answer>."
'''+marker)
    (DEST/'integration.py').write_text(text)
    b.write(DEST/'candidate_provenance.json',{'version':VERSION,'parent_bundle_sha256':b.sha(SOURCE/'bank/bank.json'),
        'parent_integration_sha256':b.sha(SOURCE/'integration.py'),'restored_rule':GAP,
        'changes':['Explicit Jeopardy genre cue','Restore V7 missing-title-span constraint','Repeat unchanged clue after complete context'],
        'new_facts_added':False,'model_parameters_changed':False,'normal_model_calls':1,
        'author_trusted_not_learned':True,'post_test_development':True,'gold_runtime_input':False})
    (DEST/'README.md').write_text('# MiniMax R2 candidate\n\n独立后续候选，保留R1全部资产；恢复V7明确的标题/引文缺词规则，增加Jeopardy题型说明，并在完整公开context后重复原题。没有新增事实卡片、删改context或修改模型参数。尚需独立验收与真实测试，不替换R1。\n')
    b.write(DEST/'FILE_SHA256.json',{str(p.relative_to(DEST)):b.sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='FILE_SHA256.json'})
    print(json.dumps({'package':str(DEST),'bundle_sha256':b.sha(DEST/'bank/bank.json')}))


if __name__=='__main__':build()
