"""Invalid model evidence is terminal locally; paid prefixes remain immutable."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import requests

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.model_view import expand_material
from atomic_skillgraph.experiments import run_choice_capability_batch as batch, run_derived_choice as derived
from atomic_skillgraph.experiments import routing_continuation as recovery
from test_capability_batch import fixture_transport,candidate
from test_capability_schedule import prepare_fixture
from test_cf2_contracts import response
from test_public_process import task


def invalid_routes(monkeypatch, only=None, answers=('A','B')):
    calls=fixture_transport(monkeypatch,answers=answers);post=requests.post;routes=[]
    def wrapped(url,**kwargs):
        wire=kwargs['json'];name=wire.get('tools',[{'function':{'name':''}}])[0]['function']['name']
        if name=='submit_capability_match':
            routes.append(wire)
            if only is None or len(routes)==only:
                calls.append(deepcopy(wire));body=expand_material(json.loads(wire['messages'][1]['content']))
                value={'task_hash':body['task_hash'],'capability_id':body['capability']['id'],'applicable':True,
                    'span_ids':[body['spans'][0]['span_id']]}
                result=response([value],name=name)
                return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:result)
        return post(url,**kwargs)
    monkeypatch.setattr(requests,'post',wrapped)
    return calls,routes


def test_invalid_model_routes_finish_without_repairs_or_false_assets(tmp_path,monkeypatch):
    calls,routes=invalid_routes(monkeypatch)
    _,root,_,env,_=prepare_fixture(tmp_path,monkeypatch,full=True)
    result=derived.execute(root,env)
    assert result['status']=='awaiting_p2_review_no_usable_assets'
    assert len(routes)==24 and len(calls)==60 and result['p2']['paired_tasks']==0
    for path in (root/'p2/candidates').glob('*/preflight.json'):
        for row in batch.read(path):
            assert row['match']['applicable'] is False
            assert row['rejection']['reason']=='Applicability requires distinct public locations'
    derived.execute(root,env);assert len(calls)==60


def test_host_error_is_not_model_abstention():
    def receive(*a,**k):raise ValueError('Changed source identity')
    learner=SimpleNamespace(system=SimpleNamespace(checkpoint=None,config={'learning':{}}),_receive=receive)
    with pytest.raises(ValueError,match='Changed source identity'):cb.match_public(learner,candidate(),task('target'))


def test_rejected_route_is_durable_without_another_request(tmp_path):
    from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
    count=[]
    def receive(*a,**k):
        count.append(1);exc=ValueError('Applicability requires distinct public locations')
        exc.model_authored=True;exc.logical_decision_id='saved-rejected-response';raise exc
    system=SimpleNamespace(checkpoint=TaskCheckpoint(tmp_path),config={'learning':{}})
    learner=SimpleNamespace(system=system,_receive=receive);c=candidate();t=task('target')
    assert cb.match_public(learner,c,t)==cb.abstain_match(c['capability'],t)
    system.checkpoint=TaskCheckpoint(tmp_path)
    assert cb.match_public(learner,c,t)==cb.abstain_match(c['capability'],t) and len(count)==1


@pytest.mark.parametrize('promote',[False,True])
def test_versioned_child_reuses_all_paid_work_and_remaining_budget(tmp_path,monkeypatch,promote):
    calls,routes=invalid_routes(monkeypatch,only=17,answers=('B','A') if promote else ('A','B'))
    _,parent,_,env,_=prepare_fixture(tmp_path,monkeypatch,full=True,unresolved=True,recovery=True)
    match_public=cb.match_public
    def old_boundary(learner,candidate,task):
        result=match_public(learner,candidate,task)
        if any(k.endswith('_model_rejection') for k in learner.system.checkpoint.state):
            raise ValueError('Applicability requires distinct public locations')
        return result
    monkeypatch.setattr(cb,'match_public',old_boundary)
    with pytest.raises(ValueError,match='distinct public locations'):derived.execute(parent,env)
    monkeypatch.setattr(cb,'match_public',match_public)
    prior_http=77 if promote else 65
    new_http=29 if promote else 17
    assert len(calls)==prior_http and len(routes)==17
    prior=batch.read(parent/'p2/budget.json');before={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    old=batch.read(parent/'p2/plan.json')['code'];new={**old,'git_sha':'new-committed-routing-fix','source_sha256':'new-source'}
    for module in (batch,derived,recovery):monkeypatch.setattr(module,'code_identity',lambda:new)
    child=tmp_path/'child'
    assert recovery.prepare(parent,child)['billing_ready']
    assert len(calls)==prior_http
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    from atomic_skillgraph.experiments.formal_choice_batch import BatchGovernor
    state=batch.read(child/'budget.json');governor=BudgetGovernor(child/'budget.json',**state['limits'])
    stage=BatchGovernor(governor,batch.read(child/'p2/plan.json'),child/'p2')
    assert stage.headroom({})['requests']==600-prior_http
    assert stage.headroom({})['tokens']==10000000-sum(a['accounted_tokens'] for a in prior['attempts'].values())
    result=derived.execute(child,env)
    assert result['status']==('awaiting_test_review' if promote else 'awaiting_p2_review_no_usable_assets')
    p2=batch.read(child/'p2/result.json');assert p2['new_http']==new_http and p2['total_p2_http']==prior_http+new_http
    assert len(calls)==prior_http+new_http+(51 if promote else 0)
    assert p2['prior_p2_tokens']==prior_http*5 and p2['total_p2_tokens']==(prior_http+new_http)*5
    assert len(p2['candidates'])==2 and all(c['status']==('promoted' if promote else 'screen_rejected') for c in p2['candidates'])
    assert batch.read(child/'p2/budget.json')['attempts'].keys()>=prior['attempts'].keys()
    assert before=={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    count=len(calls);derived.execute(child,env);assert len(calls)==count
    assert (child/'val').exists()==promote and not (child/'test').exists()
    audit=next((parent/'p2/candidates').glob('*/routing/4/requests.json'))
    audit.write_text(audit.read_text()+' ')
    with pytest.raises(ValueError,match='Inherited source'):derived.check(child)
