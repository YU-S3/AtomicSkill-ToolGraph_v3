"""Actual 92 -> 81 character rejection must get compression feedback, not another repair."""
import json
import os
from pathlib import Path

import pytest
import requests

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_capability_batch import fixture_source,fixture_transport


@pytest.mark.parametrize('final_size',[45,81])
def test_length_feedback_rewrites_with_headroom_and_preserves_single_repair(tmp_path,monkeypatch,final_size):
    calls=fixture_transport(monkeypatch);post=requests.post;proposals=[]
    def overlong(url,**kwargs):
        response=post(url,**kwargs);body=response.json();function=body['choices'][0]['message']['tool_calls'][0]['function']
        if function['name']=='submit_learning':
            value=json.loads(function['arguments']);value['applicability']=value['applicability'].ljust(92 if not proposals else final_size)
            proposals.append(value);function['arguments']=json.dumps(value)
        response.json=lambda:body
        return response
    monkeypatch.setattr(requests,'post',overlong)
    _,root,plan,_=fixture_source(tmp_path,monkeypatch);system=EmpiricalSystem(batch.read(root/'config.json'),harness=AnswerAdapter('livemath',{}))
    try:
        for row in batch.read(root/'source_cases.json'):system.bank.save_case(PublicTask(**row['task']),row['experience'])
        batch.learning_boundary(system,root/'learning/0',batch.learning_scope(plan,'candidate:0'))
        selected=[t['physical_key'] for rows in plan['selection_tasks'].values() for t in rows]
        log=system.learner.learn_capability_batch(plan['candidate_units'][0],selected)
        assert log['decision']==('candidate' if final_size==45 else 'rejected')
        assert len(proposals)==2 and not system.bank.all('skill')
        repair=next(p for p in calls if 'actual 92' in p['messages'][-1]['content'])
        assert 'Rewrite substantially shorter, not a one-word trim' in repair['messages'][-1]['content']
        assert 'aim <=50 characters' in repair['messages'][-1]['content']
        assert '3-5 word category' in repair['messages'][0]['content']
        material=json.loads(repair['messages'][1]['content'])
        assert 'train_cases' not in material and material['learning_material_version']==cb.MATERIAL_VERSION
        assert len(material['public_sources'])==2 and 'Shorten only' in material['instruction']
        assert len(json.dumps(repair).encode())<16384
        assert repair['tools'][0]['function']['parameters']['properties']['applicability']['maxLength']==80
        assert repair['tools'][0]['function']['parameters']['properties']['guidance']['maxLength']==180
        audit=batch.read(root/'learning/0/requests.json')['requests']
        assert [q['repair'] for q in audit if q['purpose']=='guidance_learning']==[0,1]
        count=len(calls);assert system.learner.learn_capability_batch(plan['candidate_units'][0],selected)==log and len(calls)==count
    finally:system.close()


def test_actual_repair_81_remains_rejected_under_unchanged_limits():
    value=os.environ.get('LIVEMATH_P2_R5_SELECTION_RUN')
    if not value:pytest.skip('Set the actual completed R5 learning prefix')
    root=Path(value);log=batch.read(root/'learning/1/learning.json')
    cases=batch.records(batch.read(root/'source_cases.json'));by_id={r['task']['physical_key']:r for r in cases}
    rows=[by_id[k] for k in log['source_case_ids']]
    proposals=[q['response']['tool_calls'][0]['arguments'] for q in batch.read(root/'learning/1/requests.json')['requests'] if q['purpose']=='guidance_learning']
    assert [len(p['applicability']) for p in proposals]==[92,81]
    for p in proposals:
        with pytest.raises(ValueError,match='expected <= 80'):
            cb.candidate_feedback(p,log['capability'],rows)
    assert cb.generation_schema(cb.CHECK,prompt_revision='procedural-actions.r6')['properties']['reason']['maxLength']==180
