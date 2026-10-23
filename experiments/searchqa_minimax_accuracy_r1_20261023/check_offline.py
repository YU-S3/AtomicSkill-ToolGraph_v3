"""Real native compiler/Docker/resume with intercepted MiniMax HTTP; no paid calls."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import requests
import run_comparison as run
import infrastructure_guard as guard
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.harness.scorers.searchqa import evaluate

c,r,HERE = run.c,run.r,run.HERE
spec = importlib.util.spec_from_file_location('surface_program', HERE / 'packages/minimax_r1/programs/submit.py')
surface = importlib.util.module_from_spec(spec); spec.loader.exec_module(surface)
cases = [
    ('<answer>R&amp;D</answer>', 'Name the group', 'R&D'),
    ('Writer’s Notebook', 'Name the work', "Writer's Notebook"),
    ('twenty-six', 'What age is required?', '26'),
    ('26 years old', 'What age is required?', '26'),
    ('12 percent', 'What percentage is this?', '12'),
    ('12%', 'What percentage is this?', '12'),
    ('Three', 'This many sides', '3'),
    ('The Three Musketeers', 'Name the book', 'The Three Musketeers'),
    ('12 million', 'How many people?', '12 million'),
    ('12%', 'Name the work', '12%'),
    ('Gabriel García Márquez', 'Name the author', 'Gabriel García Márquez'),
    ('42', 'Which highway?', '42'),
    ('-1.5 percent', 'What percentage?', '-1.5'),
    ('twenty zero', 'How many?', 'twenty zero'),
]
for answer,question,expected in cases:
    assert surface.normalize(answer,question)[0] == expected
for value in ['', '<answer></answer>', '<answer>x</answer><answer>y</answer>', '&lt;answer&gt;x&lt;/answer&gt;']:
    with pytest.raises(ValueError): surface.normalize(value,'Question')

records = r.read(HERE / 'test1400_host_only.json')
replay = []
for row in records:
    value, changes = surface.normalize(row['prediction'],row['question'])
    if changes:
        score = evaluate('<answer>'+value+'</answer>',row['gold'])
        replay.append({'task_id':row['task_id'],'old':row['prediction'],'new':value,'changes':changes,
                       'old_correct':row['correct'],'score':score})
r.write_json(HERE / 'surface_replay_host_only.json', replay)

tasks = r.read(HERE / 'tasks128_public.json')['tasks'][:2]
checks=[]
for mode in ['normal','docker_after_response','transport_failure']:
    root=Path(tempfile.mkdtemp(prefix='minimax_r1_offline_',dir='/home/yangchengyu'))
    ledger={'limits':{'token_limit':1000000,'request_limit':20,'finish_reserve':0},'attempts':{},'unknown_billing':False}
    for name in ['budget.json','inherited_budget.json']: r.write_json(root/name,ledger)
    r.write_json(root/'authorization.json',{'status':'approved','fixture_only':True})
    r.write_json(root/'cost_carryforward_manifest.json',{'fixture_only':True})
    r.write_json(root/'tasks.json',{'tasks':tasks})
    spec = {**r.read(HERE/'spec.json'),'tasks':str(root/'tasks.json')}
    if mode != 'normal': spec['arms']={'r1':run.arms()['r1']}
    r.write_json(root/'spec.json',spec)
    r.write_json(root/'offline_acceptance.json',{'passed':True,'driver_sha256':r.sha(c.__file__)})
    seen=[]
    def post(*args,**kwargs):
        wire=deepcopy(kwargs['json']);current=r.read(root/'fixture/current.json')
        task=next(t for t in tasks if t['task_id']==current['task_id'])
        public=json.loads(wire['messages'][1]['content'])
        assert public['goal']==task['goal'] and public['inputs']==task['inputs']
        assert not any(k in public for k in ['gold','answers','expected_answer'])
        assert wire['model']=='MiniMax-M3' and wire['max_completion_tokens']==65536
        assert wire['thinking']=={'type':'adaptive'} and wire['stream'] is True
        assert not wire.get('tools')
        if current['arm']=='r1':
            for name in ['solve','evidence','surface']:
                body=(HERE/'packages/minimax_r1/guidance'/f'{name}.txt').read_text().strip()
                assert wire['messages'][0]['content'].count(body)==1
        seen.append(current)
        if mode=='transport_failure': raise requests.ConnectionError('Synthetic transport failure')
        chunk={'id':'offline','model':'MiniMax-M3','choices':[{'index':0,'delta':{'content':'<answer>A &amp; B</answer>'},'finish_reason':'stop'}],
               'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
        def lines(**unused): yield b'data: '+json.dumps(chunk).encode()
        return SimpleNamespace(status_code=200,ok=True,headers={},iter_lines=lines,close=lambda:None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c,'CAMPAIGN',root);c.prepare('fixture',root/'spec.json')
        patch.setattr(c,'HERE',root)
        patch.setattr(r,'load_env',lambda p:None);patch.setenv('MINIMAX_API_KEY','offline')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
        patch.setattr(socket.socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No network')))
        with (root/'fixture.log').open('w') as log,redirect_stdout(log):
            if mode=='transport_failure':
                with pytest.raises(Exception): c.execute('fixture')
                assert len(seen)==1 and not c.completed(root/'fixture','r1')
                with pytest.raises(AssertionError,match='Unresolved billing'): c.execute('fixture')
                assert len(seen)==1
            else:
                if mode=='docker_after_response':
                    with pytest.MonkeyPatch.context() as failure:
                        failure.setattr(ProgramWorker,'execute',lambda *a,**k:{'status':'execution_error',
                            'diagnostic':'Cannot connect to the Docker daemon at unix:///var/run/docker.sock.',
                            'calls':0,'cpu_seconds':None,'execution_receipt':{'exit_code':125}})
                        with pytest.raises(guard.DockerUnavailable): c.execute('fixture')
                    assert len(seen)==1 and not c.completed(root/'fixture','r1')
                c.execute('fixture')
                for arm in spec['arms']:
                    rows=c.completed(root/'fixture',arm)
                    assert len(rows)==2
                    for row in rows.values():
                        assert row['execution']['all_authored_nodes_completed']
                        assert all(a['status']=='ok' and a['outputs_consumed'] for a in row['execution']['attempts'])
                        assert row['execution']['prediction']==('<answer>A & B</answer>' if arm=='r1' else '<answer>A &amp; B</answer>')
                        assert len(row['requests'])==1 and sum(x['total_tokens'] for x in row['usage'])==5
                count=len(seen); c.execute('fixture'); assert len(seen)==count==2*len(spec['arms'])
        assert all(r.read(root/'fixture/freeze_audit.json').values())
    checks.append({'mode':mode,'passed':True,'fixture':str(root),'intercepted_http':len(seen)})

bundle=r.read(HERE/'packages/minimax_r1/bank/bank.json')
runtime_text=json.dumps(bundle)
assert all(t['task_id'] not in runtime_text and t['task_id'].split(':')[-1] not in runtime_text for t in r.read(r.DATASETS/'test.json')['tasks'])
assert not any(k in runtime_text for k in ['evaluator_records','gold_answers','test1400_host_only'])
assert all(p['allowed_tools']==[] for p in bundle['program'])
receipt={'passed':True,'real_http':0,'surface_cases':len(cases)+4,'checks':checks,
         'controller_sha256':r.sha(run.__file__),'driver_sha256':r.sha(c.__file__),'test_sha256':r.sha(__file__),
         'source':r.code_identity(),'packages':{a:r.tree_identity(v['package']) for a,v in run.arms().items()},
         'surface_replay_changes':len(replay)}
r.write_json(HERE/'offline_acceptance.json',receipt)
print(json.dumps(receipt))
