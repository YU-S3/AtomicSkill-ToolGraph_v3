"""Real image bytes through production admission, learning and checkpoint paths."""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest

from atomic_skillgraph.agents.image_input import measure,wire_bytes
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments.run_formal import model_settings
from atomic_skillgraph.harness.benchmarks import AnswerAdapter,image_content_parts
from test_cf2_contracts import http,response
from test_empirical import config_for

LOCK=json.loads((Path(__file__).parents[1]/'configs/models_docvqa_deepseek_v41.lock.json').read_text())['formal_models'][0]


def picture(tmp_path):
    path=tmp_path/'public.png'
    Image.effect_noise((600,600),100).convert('RGB').save(path)
    return path


def test_visual_transport_and_token_bounds_are_separate(tmp_path):
    payload={'messages':[{'role':'user','content':[{'type':'text','text':'public'},*image_content_parts([picture(tmp_path)])]}]}
    before=deepcopy(payload);audit=measure(payload,LOCK['image_input'])
    assert audit['wire_bytes']>100000 and audit['text_wire_bytes']<1000
    assert 1024<audit['input_token_bound']<2000 and payload==before
    assert measure(payload,None)['input_token_bound']==wire_bytes(payload)
    for key,cap in [('image_bytes',1),('request_bytes',10),('max_dimension',10)]:
        with pytest.raises(ValueError):measure(payload,{**LOCK['image_input'],key:cap})
    payload['messages'][0]['role']='assistant'
    with pytest.raises(ValueError,match='user'):measure(payload,LOCK['image_input'])


def test_visual_solver_learning_metering_and_resume(tmp_path,monkeypatch):
    path=picture(tmp_path);config=model_settings(config_for(tmp_path/'bank'),LOCK)
    config['budget']={'train_task_tokens':200000,'train_solve_tokens':120000,'train_learning_tokens':80000,'eval_task_tokens':96000,
        'token_limit':496000,'request_limit':175,'finish_reserve':0}
    governor=BudgetGovernor(tmp_path/'budget.json',token_limit=496000,request_limit=175,finish_reserve=0)
    sent=http(monkeypatch,[response([],finish='stop',content='<answer>word</answer>'),
        response([{'decision':'upsert_guidance','guidance_skill':{'goal':'Read the requested document field',
            'guidance':'Locate the field label, then read the adjacent value exactly as printed.'}}],name='submit_learning')])
    s=EmpiricalSystem(config,harness=AnswerAdapter('docvqa',{'id':{'answers':['word']}}),budget_governor=governor)
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');s.audit_path=tmp_path/'requests.json'
    task=PublicTask('id','physical','Which word?',{'images':[str(path)]},split='train')
    trace=s.run_task(task)
    assert trace['score']['hard'] and trace['learning_status']=='completed',trace.get('learning')
    assert trace['learning']['persisted_skill_id'] and len(sent)==2
    assert all(p['messages'][1]['content'][1]['image_url']['url']==image_content_parts([path])[0]['image_url']['url'] for p in sent)
    assert all(a['input_estimate']<25000 for a in governor.state['attempts'].values())
    assert sum(a['accounted_tokens'] for a in governor.state['attempts'].values())==10
    s.run_task(task);assert len(sent)==2
    s.close()


def test_oversized_visual_learning_refusal_keeps_image_and_audit(tmp_path,monkeypatch):
    config=model_settings(config_for(tmp_path/'bank'),LOCK)
    s=EmpiricalSystem(config,harness=AnswerAdapter('docvqa',{'id':{'answers':['word']}}))
    sent=http(monkeypatch,[])
    with pytest.raises(ValueError,match='material_too_large'):
        s.agent('extractor','prompt',{'required':'x'*40000,'content_parts':image_content_parts([picture(tmp_path)])},
            'submit_learning',{'type':'object'},repair_limit=0,decision_purpose='guidance_learning')
    assert not sent
    audit=s.requests[-1]['material_refusal']
    assert audit['wire_bytes']>40000 and audit['payload']['messages'][1]['content'][1]['type']=='image_url'
    s.close()
