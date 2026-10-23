"""Verify R2 public-question framing in the real native execution path, zero HTTP."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import run_test as x

c,r,HERE=x.c,x.r,x.HERE
tasks=r.read(r.DATASETS/'test.json')['tasks']
tasks=[tasks[0],max(tasks,key=lambda t:len(t['inputs']['context']))]
root=Path(tempfile.mkdtemp(prefix='minimax_r2_offline_',dir='/home/yangchengyu'))
ledger={'limits':{'token_limit':1000000,'request_limit':10,'finish_reserve':0},'attempts':{},'unknown_billing':False}
for name in ['budget.json','inherited_budget.json']:r.write_json(root/name,ledger)
r.write_json(root/'authorization.json',{'status':'approved','fixture_only':True})
r.write_json(root/'cost_carryforward_manifest.json',{'fixture_only':True})
r.write_json(root/'tasks.json',{'tasks':tasks})
r.write_json(root/'spec.json',{'tasks':str(root/'tasks.json'),'base_config':str(HERE/'base_config.json'),
    'arms':{'r2':{'kind':'native','package':str(HERE/'packages/minimax_r2')}},'purpose':'Offline R2 payload/graph check'})
r.write_json(root/'offline_acceptance.json',{'passed':True,'driver_sha256':r.sha(c.__file__)})
seen=[]
def post(*args,**kwargs):
    wire=deepcopy(kwargs['json']);cur=r.read(root/'fixture/current.json')
    task=next(t for t in tasks if t['task_id']==cur['task_id'])
    material=json.loads(wire['messages'][1]['content'])
    assert material['goal']==material['final_question_to_answer']==task['goal']
    assert material['inputs']==task['inputs']
    assert list(material)[-2:]==['final_question_to_answer','final_submission_instruction']
    assert 'Jeopardy-style' in wire['messages'][0]['content']
    assert 'return EXACTLY the missing span' in wire['messages'][0]['content']
    assert wire['model']=='MiniMax-M3' and wire['max_completion_tokens']==65536 and wire['thinking']=={'type':'adaptive'}
    seen.append({'task_id':cur['task_id'],'payload':wire})
    chunk={'id':'offline','model':'MiniMax-M3','choices':[{'index':0,'delta':{'content':'<answer>A &amp; B</answer>'},'finish_reason':'stop'}],
           'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
    def lines(**unused):yield b'data: '+json.dumps(chunk).encode()
    return SimpleNamespace(status_code=200,ok=True,headers={},iter_lines=lines,close=lambda:None)
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(c,'CAMPAIGN',root);c.prepare('fixture',root/'spec.json');patch.setattr(c,'HERE',root)
    patch.setattr(r,'load_env',lambda p:None);patch.setenv('MINIMAX_API_KEY','offline')
    patch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    patch.setattr(socket.socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
    with (root/'fixture.log').open('w') as log,redirect_stdout(log):
        c.execute('fixture');c.execute('fixture')
    rows=c.completed(root/'fixture','r2');assert len(rows)==2 and len(seen)==2
    assert all(v['execution']['all_authored_nodes_completed'] for v in rows.values())
    assert all(v['execution']['prediction']=='<answer>A & B</answer>' for v in rows.values())
    assert all(len(v['requests'])==1 for v in rows.values())
    assert all(r.read(root/'fixture/freeze_audit.json').values())
r.write_json(root/'intercepted_requests.json',seen)
receipt={'passed':True,'real_http':0,'intercepted_http':2,'fixture':str(root),
    'package':r.tree_identity(HERE/'packages/minimax_r2'),'driver_sha256':r.sha(c.__file__),
    'test_sha256':r.sha(__file__),'same_validated_r2_package':r.tree_identity(HERE/'packages/minimax_r2')==r.tree_identity(x.PARENT/'followup96/r2/input')}
receipt['source']=r.code_identity()
receipt['controller_sha256']=r.sha(HERE/'run_test.py')
r.write_json(HERE/'offline_acceptance.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='package'}))
