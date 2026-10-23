"""Reconstruct the effective 1400-task MiniMax baseline, excluding old Docker failures."""
import html
import json
import re
from pathlib import Path
import run_comparison as run
from atomic_skillgraph.harness.scorers.searchqa import extract_answer, normalize_answer

c,r,HERE=run.c,run.r,run.HERE
FIRST=Path('/home/yangchengyu/searchqa_v7_qwen_minimax_test1400_seed42_20261022/minimax_m3')
SECOND=Path('/home/yangchengyu/searchqa_v7_minimax_test1400_resume_20261023')
REPLACEMENT=run.PARENT
manifest_path=HERE.parent/'searchqa_v7_minimax_docker_replacement_20261023/replacement_manifest.json'


def audit():
    excluded={x['task_id'] for x in r.read(manifest_path)['excluded']}
    rows={};provenance={}
    stages=[FIRST/'fulltest1400',SECOND/'fulltest1400',REPLACEMENT/'replacement200',REPLACEMENT/'remaining303']
    for i,stage in enumerate(stages):
        for tid,row in c.completed(stage,'v7_text_compiler').items():
            if i<2 and tid in excluded:continue
            assert tid not in rows
            rows[tid]=row;provenance[tid]={'stage':str(stage),'result_sha256':r.digest(row)}
    assert len(rows)==1400 and sum(bool(v['score']['hard']) for v in rows.values())==1116
    gold=r.read(r.DATASETS/'evaluator_records_test.json')
    records=[]
    for i,t in enumerate(r.read(r.DATASETS/'test.json')['tasks']):
        tid=t['task_id'];row=rows[tid]
        records.append({'index':i+1,'task_id':tid,'question':row['task']['goal'],'context':row['task']['inputs']['context'],
            'prediction':extract_answer(row['execution']['prediction']),'gold':gold[tid]['answers'],
            'correct':row['score']['hard'],'f1':row['score']['soft'],'tokens':sum(x['total_tokens'] for x in row['usage'])})
    assert records==r.read(HERE/'test1400_host_only.json')
    groups={k:[] for k in ['substring_overlong','substring_too_short','encoding','contains_percent','numeric_words','zero_overlap']}
    for row in records:
        answer=normalize_answer(row['prediction']);gs=[normalize_answer(g) for g in row['gold']]
        if '%' in row['prediction']:groups['contains_percent'].append(row['index'])
        if row['correct']:continue
        if any(g and g!=answer and (' '+g+' ') in (' '+answer+' ') for g in gs):groups['substring_overlong'].append(row['index'])
        if any(answer and answer!=g and (' '+answer+' ') in (' '+g+' ') for g in gs):groups['substring_too_short'].append(row['index'])
        normalized=html.unescape(row['prediction']).translate(str.maketrans({'‘':"'",'’':"'",'“':'"','”':'"','–':'-','—':'-'}))
        if normalize_answer(normalized) in gs:groups['encoding'].append(row['index'])
        if any(re.fullmatch(r'\d+(?:\.\d+)?',g) for g in gs):groups['numeric_words'].append(row['index'])
        if row['f1']==0:groups['zero_overlap'].append(row['index'])
    assert groups==r.read(HERE/'error_categories_host_only.json')
    r.write_json(HERE/'baseline_provenance_host_only.json',{'tasks':provenance,'replacement_manifest_sha256':r.sha(manifest_path),
        'official_public_tasks_sha256':r.sha(r.DATASETS/'test.json'),'official_private_gold_sha256':r.sha(r.DATASETS/'evaluator_records_test.json'),
        'exclusion_by_infrastructure_not_correctness':True,'excluded_old_tokens':1659185,'excluded_old_http':439,
        'diagnostic_groups_overlap':True,'substring_is_not_semantic_equivalence':True})
    return {'passed':True,'correct':1116,'tasks':1400,'categories':{k:len(v) for k,v in groups.items()}}


if __name__=='__main__': print(json.dumps(audit()))
