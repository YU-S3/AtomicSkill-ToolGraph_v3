"""Exercise real provider serialization, compiler, Docker submit and resume without HTTP."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import run_campaign as x
from atomic_skillgraph.harness.scorers.searchqa import exact_match
from provider_retry import transient

c,r,HERE=x.c,x.r,x.HERE
BASE=Path(tempfile.mkdtemp(prefix='target_accuracy_offline_',dir='/home/yangchengyu'))
all_tasks=r.read(r.DATASETS/'test.json')['tasks']
cards=r.read(HERE/'packages/reference/fact_cards.json')
card_task=next(t for t in all_tasks if any(any(a.casefold() in t['goal'].casefold() for a in card['anchors']) for card in cards if not card['id'].startswith('parent_')))
tasks=[card_task,next(t for t in all_tasks if t!=card_task)]
checks=[]

def fixture(model,mode):
    root=BASE/(model+'_'+mode)
    state={'limits':{'token_limit':2000000,'request_limit':30,'finish_reserve':0},'attempts':{},'unknown_billing':False}
    for name in ('budget.json','inherited_budget.json'):r.write_json(root/name,state)
    r.write_json(root/'authorization.json',{'status':'approved','fixture_only':True})
    r.write_json(root/'cost_carryforward_manifest.json',{'fixture_only':True})
    r.write_json(root/'offline_acceptance.json',{'passed':True,'driver_sha256':r.sha(c.__file__)})
    r.write_json(root/'tasks.json',{'tasks':tasks})
    spec=r.read(HERE/model/'probe_spec.json');spec['tasks']=str(root/'tasks.json')
    if mode!='normal':spec['arms']={'reference':spec['arms']['reference']}
    r.write_json(root/'spec.json',spec)
    seen=[];cfg=r.read(HERE/model/'base_config.json')
    def post(*args,**kwargs):
        wire=deepcopy(kwargs['json']);current=r.read(root/'fixture/current.json')
        task=next(t for t in tasks if t['task_id']==current['task_id'])
        material=json.loads(wire['messages'][1]['content'])
        assert material['goal']==task['goal'] and material['inputs']==task['inputs']
        assert 'gold' not in material and 'answers' not in material
        assert wire['model']==cfg['llm']['model']
        if current['arm'] in ('scope','reference'):
            assert 'MISSING ROLE' in wire['messages'][0]['content']
            assert len(material['reference_card_ids'])<=4
            if current['arm']=='reference' and task==tasks[0]:assert any(not cid.startswith('parent_') for cid in material['reference_card_ids'])
        seen.append(wire)
        if mode=='rate_limit' and len(seen)==1:
            data={'error':{'type':'rate_limit_exceeded','code':'rate_limit_exceeded','message':'Too many requests'}}
            return SimpleNamespace(status_code=429,ok=False,headers={},content=json.dumps(data).encode(),text=json.dumps(data),json=lambda:data,close=lambda:None)
        chunk={'id':'offline','model':cfg['llm']['model'],'choices':[{'index':0,'delta':{'content':'<answer>J. K. Ch\u00e1vez</answer>'},'finish_reason':'stop'}],
               'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
        if mode=='incomplete_stream_unmetered' and len(seen)==1:chunk.pop('usage')
        lines=[b'data: '+json.dumps(chunk).encode()]
        if not (mode.startswith('incomplete_stream') and len(seen)==1):lines.append(b'data: [DONE]')
        return SimpleNamespace(status_code=200,ok=True,headers={},iter_lines=lambda **unused:iter(lines),close=lambda:None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c,'CAMPAIGN',root);c.prepare('fixture',root/'spec.json');patch.setattr(c,'HERE',root)
        patch.setattr(r,'load_env',lambda p:None);patch.setenv(cfg['llm']['api_key_env'],'offline')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
        patch.setattr(socket.socket,'connect',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Network forbidden')))
        patch.setattr('provider_retry.time.sleep',lambda seconds:None)
        with (root/'fixture.log').open('w') as log,redirect_stdout(log):c.execute('fixture');c.execute('fixture')
        for arm in spec['arms']:
            rows=c.completed(root/'fixture',arm)
            assert len(rows)==2 and all(v['execution']['all_authored_nodes_completed'] for v in rows.values())
            if arm!='baseline':assert all(v['execution']['prediction']=='<answer>J.K. Chavez</answer>' for v in rows.values())
        assert len(seen)==(6 if mode=='normal' else 3)
        assert all(r.read(root/'fixture/freeze_audit.json').values())
        if mode=='rate_limit' or mode.startswith('incomplete_stream'):
            assert seen[0]['messages']==seen[1]['messages']
            field=cfg['llm']['token_limit_field']
            assert 1024<=seen[1][field]<=seen[0][field]
            state=r.read(root/'budget.json')
            assert sum(v['status']=='unknown_billing' for v in state['attempts'].values())==(0 if mode=='incomplete_stream' else 1)
            assert sum(v.get('accounted_tokens',0) for v in state['attempts'].values())==(15 if mode=='incomplete_stream' else 10)
            assert not r.BudgetGovernor(root/'budget.json',**state['limits']).billing_blocked
    r.write_json(root/'intercepted_requests.json',seen)
    checks.append({'model':model,'mode':mode,'passed':True,'intercepted_http':len(seen),'fixture':str(root)})

for model in x.MODELS:
    for mode in ('normal','rate_limit'):fixture(model,mode)
    if model=='glm':
        fixture(model,'incomplete_stream');fixture(model,'incomplete_stream_unmetered')
for status in (400,401,403,422):assert not transient({'http_status':status},RuntimeError())
assert not transient({'http_status':429,'response_diagnostic':{'error':{'type':'insufficient_quota'}}},RuntimeError())

module_path=HERE/'packages/scope/programs/submit.py'
spec=importlib.util.spec_from_file_location('surface',module_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
replays={}
for model in ('minimax_r2','glm_v7'):
    changed=[]
    for row in r.read(HERE/'comparisons_host_only.json'):
        old=row['results'][model]
        if not old['prediction']:continue
        new,changes=mod.normalize(old['prediction'],row['question'])
        em=exact_match(new,row['gold'])
        if em!=old['em']:changed.append({'task_id':row['task_id'],'old':old['prediction'],'new':new,'old_em':old['em'],'new_em':em,'changes':changes})
    replays[model]={'changes':changed,'gains':sum(v['new_em']>v['old_em'] for v in changed),'losses':sum(v['new_em']<v['old_em'] for v in changed)}
r.write_json(HERE/'surface_replay_host_only.json',replays)
receipt={'passed':True,'real_http':0,'source':r.code_identity(),'driver_sha256':r.sha(c.__file__),
         'checks':checks,'files':{n:r.sha(HERE/n) for n in x.CODE_FILES},'at':r.utc()}
r.write_json(HERE/'offline_acceptance.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
