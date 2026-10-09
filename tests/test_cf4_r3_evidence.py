"""Read the fixed review evidence and new public materialization without any model calls."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.model_view import pack_material,expand_material,canonical_bytes
from atomic_skillgraph.experiments.canonical_manifest import ordered_train, verify, sha256
from skillcompiler_bench_contracts.livemath import permute_livemath_choices
from test_cf2_contracts import office

REVIEW=Path('/mnt/d/T3S_exp/deliveries/outputs/ours_focused_review_20261009/package')
DATA=Path('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1')
AUTHORITY=Path('/home/yangchengyu/asg_cf4_r1_20261007/data/main_experiment_v1')
pytestmark=pytest.mark.skipif(not REVIEW.exists() or not DATA.exists(),reason='Requires the fixed local review/materialization')


def read(path): return json.loads(path.read_text())


def evidence(name,value):
    if os.environ.get('CF4_R3_EVIDENCE_DIR'):
        root=Path(os.environ['CF4_R3_EVIDENCE_DIR']); root.mkdir(parents=True,exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def test_all_177_public_choices_and_three_seeds_preserve_physical_splits():
    verify(AUTHORITY)
    universe={}; counts={}; public_bytes={}; records={}
    for split in ('train','val','test'):
        tasks=read(DATA/'livemath'/(split+'.json'))['tasks']
        frozen=read(AUTHORITY/'livemath'/(split+'.json'))['tasks']
        assert [t['task_id'] for t in tasks]==[t['task_id'] for t in frozen]
        previous=read(Path('/home/yangchengyu/main_experiment_v1_resources_cf4_20261006_v1')/'livemath'/(split+'.json'))['tasks']
        assert [(t['task_id'],t['physical_key']) for t in tasks]==[(t['task_id'],t['physical_key']) for t in previous]
        current=read(DATA/'livemath'/('evaluator_records_'+split+'.json'))
        records.update(current); universe.update({t['task_id']:t for t in tasks}); counts[split]=len(tasks)
        for task in tasks:
            record=current[task['task_id']]; original=deepcopy(record['choice_projection']['canonical'])
            projected=permute_livemath_choices(original,choice_seed=42,stable_item_id=task['task_id'].removeprefix('livemath:'))
            assert original==record['choice_projection']['canonical']
            assert projected==permute_livemath_choices(projected,choice_seed=42,stable_item_id=original['id'])
            assert task['inputs']['choices']==record['choices']==projected['choices']
            assert record['correct_choice']==projected['correct_choice']
            assert sorted(c['text'] for c in projected['choices'])==sorted(c['text'] for c in original['choices'])
            assert record['correct_choice']['text']==original['correct_choice']['text']
            assert not {'choice_projection','correct_choice','theorem','sketch'} & task['inputs'].keys()
            public_bytes[task['physical_key']]=digest(task['inputs'])
    assert counts=={'train':60,'val':17,'test':100} and len(universe)==177
    gold=Counter(r['correct_choice']['label'] for r in records.values())
    assert gold=={'A':33,'B':30,'C':45,'D':38,'E':31}
    train=read(DATA/'livemath/train.json')['tasks']
    orders=[ordered_train(train,seed) for seed in (42,43,44)]
    assert len({tuple(t['task_id'] for t in order) for order in orders})==3
    for order in orders:
        assert {t['physical_key']:digest(t['inputs']) for t in order}=={t['physical_key']:public_bytes[t['physical_key']] for t in train}
    evidence('M1',{'counts':counts,'gold_counts':dict(gold),'choice_seed':42,'three_run_seeds_same_projection':True,
                   'authority_sha256':sha256(AUTHORITY/'manifest.json'),'materialization_sha256':sha256(DATA/'materialization.json')})


def test_six_actual_mismatch_cases_use_their_own_results(tmp_path):
    rows=read(REVIEW/'case_source/examples.json'); assert len(rows)==6
    s=office(tmp_path); checks=[]
    try:
        for row in rows:
            current=row['current_task_event']; true=row['true_case_event']
            assert current['result']!=true['result']
            now=s.learner._view({'task':row['true_case_task'], 'events':[current]},source_id='current:'+row['trigger_task'])
            view=s.learner._view({'task':row['true_case_task'], 'events':[true]},source_id=row['case_id'])
            result_id=view['events'][0]['result']['result_id']
            assert s.task_context.results[result_id]==true['result']
            assert result_id!=now['events'][0]['result']['result_id']
            checks.append({'case_id':row['case_id'],'trigger_task':row['trigger_task'],'true_result_sha256':digest(true['result']),
                           'source_namespace':view['case_source'],'correct':True})
        evidence('M2',checks)
    finally:s.close()


def material(sample):
    return json.loads(next(m['content'] for m in sample['logical_request']['messages'] if m['role']=='user'))


def test_four_actual_duplicate_materials_are_reversible_and_smaller():
    checks=[]
    for name in ('05_officeqa_extractor.json','06_officeqa_extractor.json','13_spreadsheetbench_extractor.json','14_spreadsheetbench_extractor.json'):
        source=material(read(REVIEW/'request_samples'/name)); before=deepcopy(source)
        packed=pack_material(source)
        assert expand_material(packed)==source==before
        assert canonical_bytes(packed)<canonical_bytes(source)
        checks.append({'sample':name,'before_bytes':canonical_bytes(source),'after_bytes':canonical_bytes(packed),
                       'reversible':True,'candidate_ids':[c['id'] for c in source['related']]})
    evidence('M5_dedup',checks)


def test_ten_existing_repair_samples_and_maximum_builder_failure_remain_auditable(tmp_path):
    samples=[(p.name,read(p)) for p in sorted((REVIEW/'request_samples').glob('*.json'))]
    repairs=[(name,s) for name,s in samples if s['logical_request']['repair']>0]
    assert len(repairs)==10
    categories=[]
    for name,s in repairs:
        messages=s['logical_request']['messages']
        feedback=[m for m in messages[2:] if m['role'] in {'tool','user'}]
        assert feedback
        text=json.dumps(feedback,ensure_ascii=False)
        tags=[tag for tag,words in [('wrapper',('runtime_step','ToolCall')),('REF',('oneOf','reference','REF')),
            ('guidance_fields',('guidance_skill','existing_skill_id')),('schema',('schema','required','additional'))]
              if any(w in text for w in words)]
        categories.append({'sample':name,'repair_index':s['logical_request']['repair'],'feedback_sha256':digest(feedback),
                           'feedback_chars':len(text),'derived_tags':tags or ['other']})
    builders=[(name,material(s)) for name,s in samples if s['logical_request']['stage']=='tool_builder']
    name,source=max(builders,key=lambda row:canonical_bytes(row[1]))
    failure=source['previous_failure']; original=deepcopy(failure); s=office(tmp_path)
    try:
        view=s.learner._failure_view(failure)
        assert failure==original and view['source']==failure['source']
        assert canonical_bytes(view)<canonical_bytes(failure)
        assert json.dumps(view).count('"source":')==1
        evidence('M5_repair_failure',{'repair_count':len(repairs),'derived_feedback':categories,'largest_sample':name,
             'failure_before_bytes':canonical_bytes(failure),'failure_after_bytes':canonical_bytes(view),
             'original_preserved':True,'bounded_failure':view})
    finally:s.close()
