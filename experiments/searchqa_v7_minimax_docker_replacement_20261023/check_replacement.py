"""Production-path offline tests for replacement, Docker stop and cached-response resume."""
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import pytest
import run_replacement as run
import infrastructure_guard as guard
from atomic_skillgraph.empirical.program_worker import ProgramWorker

r,c,HERE=run.r,run.c,run.HERE
manifest=r.read(HERE/'replacement_manifest.json')
parent=r.read(run.PARENT/'budget.json')
selected=r.read(HERE/'replacement200_tasks.json')['tasks'][:2]
old_rows=run.parent_rows()
checks=[]
for mode in ('normal','docker_after_response','docker_before_response'):
    root=Path(tempfile.mkdtemp(prefix='minimax_docker_replacement_'+mode+'_',dir='/home/yangchengyu'))
    r.write_json(root/'budget.json',run.recover_ledger(parent,manifest['failed_attempt']))
    r.write_json(root/'inherited_budget.json',parent)
    r.write_json(root/'authorization.json',{'status':'approved','fixture_only':True})
    r.write_json(root/'cost_carryforward_manifest.json',{'fixture_only':True})
    r.write_json(root/'tasks.json',{'tasks':selected})
    r.write_json(root/'spec.json',{**r.read(HERE/'replacement200_spec.json'),'tasks':str(root/'tasks.json')})
    r.write_json(root/'offline_acceptance.json',{'passed':True,'driver_sha256':r.sha(c.__file__)})
    seen=[]
    def post(*args,**kwargs):
        current=r.read(root/'fixture/current.json')['task_id']
        wire=deepcopy(kwargs['json'])
        assert wire['messages']==old_rows[current]['requests'][0]['messages']
        assert wire['model']=='MiniMax-M3' and wire['max_completion_tokens']==65536
        seen.append(current)
        chunk={'id':'offline','model':'MiniMax-M3','choices':[{'index':0,
            'delta':{'content':'<answer>fixture</answer>'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
        def lines(**unused):yield b'data: '+json.dumps(chunk).encode()
        return SimpleNamespace(status_code=200,ok=True,headers={},iter_lines=lines,close=lambda:None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c,'CAMPAIGN',root)
        c.prepare('fixture',root/'spec.json')
        patch.setattr(c,'HERE',root)
        patch.setattr(r,'load_env',lambda p:None)
        patch.setenv('MINIMAX_API_KEY','offline-fixture')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
        patch.setattr(socket.socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
        with (root/'fixture.log').open('w') as log,redirect_stdout(log):
            if mode=='normal':
                c.execute('fixture')
            else:
                with pytest.MonkeyPatch.context() as fail:
                    if mode=='docker_after_response':
                        fail.setattr(ProgramWorker,'execute',lambda *a,**k:{'status':'execution_error',
                            'diagnostic':'Cannot connect to the Docker daemon at unix:///var/run/docker.sock.',
                            'calls':0, 'cpu_seconds':None,
                            'execution_receipt':{'exit_code':125}})
                    else:
                        fail.setattr(guard,'docker_ready',lambda image:(_ for _ in ()).throw(guard.DockerUnavailable('offline Docker unavailable')))
                    with pytest.raises(guard.DockerUnavailable):c.execute('fixture')
                assert not c.completed(root/'fixture',run.ARM)
                assert len(seen)==(1 if mode=='docker_after_response' else 0)
                assert len(list((root/'fixture/infrastructure_events').glob('*.json')))==1
                if mode=='docker_after_response':
                    event=r.read(next((root/'fixture/infrastructure_events').glob('*.json')))
                    assert event['details']['no_execution_proven'] and event['checkpoint_before']['program_started']
                c.execute('fixture')
            rows=c.completed(root/'fixture',run.ARM)
            assert len(rows)==2 and len(seen)==2
            assert all(v['execution']['all_authored_nodes_completed'] for v in rows.values())
            assert all(a['status']=='ok' and a['outputs_consumed'] for v in rows.values() for a in v['execution']['attempts'])
            c.execute('fixture')
            assert len(seen)==2
        ledger=r.read(root/'budget.json')
        assert ledger['limits']==parent['limits']
        assert all(ledger['attempts'][k]==v for k,v in parent['attempts'].items())
        assert len(ledger['attempts'])==len(parent['attempts'])+2
        assert all(r.read(root/'fixture/freeze_audit.json').values())
    checks.append({'mode':mode,'passed':True,'intercepted_http':len(seen),'root':str(root)})

assert not guard.daemon_failure({'status':'execution_error','detail':'NameError: missing variable'})
assert not guard.daemon_failure({'status':'needs_input','diagnostic':'invalid answer'})
assert not guard.daemon_failure({'status':'ok','diagnostic':''})
excluded={x['task_id'] for x in manifest['excluded']}
kept={k:v for k,v in old_rows.items() if k not in excluded}
assert run.metrics(kept)['tokens']==3056694
assert manifest['old_excluded']['tokens']==1659185 and manifest['old_excluded']['http']==439
assert len(kept)==897 and len(excluded)==200
remaining={t['task_id'] for t in r.read(HERE/'remaining303_tasks.json')['tasks']}
assert len(remaining)==303 and not remaining & (set(kept)|excluded)
assert len(set(kept)|excluded|remaining)==1400
receipt={'passed':True,'real_http':0,'checks':checks,'source':r.code_identity(),
    'driver_sha256':r.sha(c.__file__),'launcher_sha256':r.sha(run.__file__),'guard_sha256':r.sha(guard.__file__),
    'test_sha256':r.sha(__file__),'exact200_replacement_and_cost_exclusion':True,
    'ordinary_model_errors_not_reclassified':True,'docker_failure_stops_before_repair_and_score':True,
    'cached_solve_resume_adds_no_duplicate_model_call':True}
r.write_json(HERE/'offline_acceptance.json',receipt)
print(json.dumps(receipt))
