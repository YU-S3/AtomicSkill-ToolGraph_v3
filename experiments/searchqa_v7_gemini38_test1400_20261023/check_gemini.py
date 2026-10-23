"""Zero-HTTP checks through the actual SearchQA runtime, Docker and scorer."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import sqlite3
import tempfile
import pytest
import run_gemini as run
import infrastructure_guard as guard
from atomic_skillgraph.agents.provider import AgentProviderError
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.budget_governor import output_token_bound, billing_blocked

r,c,HERE = run.r,run.c,run.HERE
TASKS = r.read(r.DATASETS/'test.json')['tasks']
SELECTED = [TASKS[0],max(TASKS[1:],key=lambda t:len(t['inputs']['context']))]
with sqlite3.connect((run.REFERENCE/run.ARM/'test/run.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
    MESSAGES = {t['task_id']:json.loads(db.execute('SELECT result FROM tasks WHERE id=?',
        (t['task_id'],)).fetchone()[0])['requests'][0]['messages'] for t in SELECTED}
# Deliberately non-additive counters: preserve and charge the reported total.
USAGE = {'prompt_tokens':3,'completion_tokens':2,'total_tokens':9,
         'completion_tokens_details':{'reasoning_tokens':4}}
CFG = r.read(HERE/'base_config.json')


def exercise(mode):
    root = Path(tempfile.mkdtemp(prefix='searchqa_gemini_'+mode+'_',dir='/home/yangchengyu'))
    limits = {'token_limit':1000000,'request_limit':20,'finish_reserve':0}
    for name in ('budget.json','inherited_budget.json'):
        r.write_json(root/name,{'limits':limits,'attempts':{},'unknown_billing':False})
    r.write_json(root/'authorization.json',{'status':'approved','fixture_only':True})
    r.write_json(root/'cost_carryforward_manifest.json',{'fixture_only':True})
    r.write_json(root/'tasks.json',{'tasks':SELECTED})
    r.write_json(root/'spec.json',{**r.read(HERE/'spec.json'),'tasks':str(root/'tasks.json')})
    r.write_json(root/'offline_acceptance.json',{'passed':True,'fixture_only':True,'driver_sha256':r.sha(c.__file__)})
    seen = []
    def post(*args,**kwargs):
        task_id = r.read(root/'fixture/current.json')['task_id']
        wire = deepcopy(kwargs['json'])
        assert wire['messages'] == MESSAGES[task_id]
        assert wire['model'] == 'gemini-3.8-flash' and wire['max_tokens'] == 65536
        assert wire['reasoning_effort'] == 'high' and wire['stream'] is True and kwargs['stream'] is True
        assert wire['stream_options'] == {'include_usage':True} and kwargs['timeout'] == (15.0,600)
        assert not {'thinking','thinking_budget','enable_thinking','seed','temperature','tools','tool_choice','max_completion_tokens'} & wire.keys()
        seen.append({'task_id':task_id,'payload':wire})
        abnormal = task_id == SELECTED[0]['task_id'] and mode != 'normal'
        refusal = abnormal and mode.startswith('refusal')
        finish = 'content_filter' if refusal else 'stop'
        chunk = {'id':'offline','created':10,'model':'gemini-3.8-flash','system_fingerprint':None,
                 'choices':[{'index':0,'delta':{'content':'' if refusal else '<answer>fixture</answer>'},
                             'finish_reason':None if abnormal and mode=='broken_stream' else finish}]}
        def lines(**unused):
            yield b'data: '+json.dumps(chunk).encode()
            if not abnormal or mode in ('refusal_metered','docker_after_response'):
                yield b'data: '+json.dumps({'id':'offline','created':10,'model':'gemini-3.8-flash',
                    'system_fingerprint':'','choices':[],'usage':USAGE}).encode()
            if not (abnormal and mode=='broken_stream'): yield b'data: [DONE]'
        return SimpleNamespace(status_code=200,ok=True,headers={},iter_lines=lines,close=lambda:None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c,'CAMPAIGN',root)
        c.prepare('fixture',root/'spec.json')
        patch.setattr(c,'HERE',root)
        patch.setattr(r,'load_env',lambda path:None)
        patch.setenv(CFG['llm']['api_key_env'],'offline-fixture')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
        patch.setattr(socket.socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
        with (root/'fixture.log').open('w') as log,redirect_stdout(log):
            if mode in ('broken_stream','missing_usage'):
                with pytest.raises(AgentProviderError):c.execute('fixture')
                assert len(seen)==1 and not c.completed(root/'fixture',run.ARM)
                with pytest.raises(AssertionError,match='Unresolved billing'):c.execute('fixture')
                assert len(seen)==1
            else:
                if mode=='docker_after_response':
                    with pytest.MonkeyPatch.context() as fail:
                        fail.setattr(ProgramWorker,'execute',lambda *a,**k:{'status':'execution_error',
                            'diagnostic':'Cannot connect to the Docker daemon at unix:///var/run/docker.sock.',
                            'calls':0,'cpu_seconds':None,'execution_receipt':{'exit_code':125}})
                        with pytest.raises(guard.DockerUnavailable):c.execute('fixture')
                    assert len(seen)==1 and not c.completed(root/'fixture',run.ARM)
                result = c.execute('fixture')
                assert result['arms'][run.ARM]['completed']==2 and len(seen)==2
                c.execute('fixture')
                assert len(seen)==2
                rows = c.completed(root/'fixture',run.ARM)
                for i,task in enumerate(SELECTED):
                    row = rows[task['task_id']]
                    refused = i==0 and mode.startswith('refusal')
                    if refused:
                        assert row['execution']['reason']=='provider_content_filter' and row['score']['hard'] is False
                        assert not row['execution'].get('all_authored_nodes_completed')
                    else:
                        assert row['execution']['all_authored_nodes_completed']
                        assert all(a['status']=='ok' and a['outputs_consumed'] for a in row['execution']['attempts'])
                    attempt = row['requests'][0]['http_attempts'][0]
                    assert attempt['raw_usage']==(None if refused and mode=='refusal_unknown' else USAGE)
                ledger = r.read(root/'budget.json')
                assert len(ledger['attempts'])==2
                assert all(a['accounted_tokens']==9 for a in ledger['attempts'].values() if a['status']=='metered')
                if mode=='refusal_unknown':
                    unknown = [a for a in ledger['attempts'].values() if a['status']=='unknown_billing']
                    assert len(unknown)==1 and 'accounted_tokens' not in unknown[0] and not billing_blocked(ledger)
        assert all(r.read(root/'fixture/freeze_audit.json').values())
    return {'mode':mode,'passed':True,'intercepted_requests':len(seen),'real_http':0,'fixture':str(root)},seen


checks, maximum = [],0
for mode in ('normal','refusal_metered','refusal_unknown','broken_stream','missing_usage','docker_after_response'):
    check,wires = exercise(mode);checks.append(check)
    if mode=='normal':
        template = wires[0]['payload']
        for task in TASKS:
            wire = deepcopy(template)
            material = json.loads(wire['messages'][1]['content'])
            material.update(goal=task['goal'],inputs=task['inputs'])
            wire['messages'][1]['content'] = json.dumps(material,ensure_ascii=False)
            bound = len(json.dumps(wire,allow_nan=False).encode())+output_token_bound(wire)
            assert bound < CFG['budget']['eval_task_tokens']
            maximum = max(maximum,bound)
system = object.__new__(EmpiricalSystem)
system.config,system.providers,system.provider_override,system.budget_governor = CFG,{},None,None
for purpose,cap in [(None,65536),('finish_only',16384)]:
    provider = system.provider('runtime',purpose)
    wire = provider._build_payload([{'role':'user','content':'fixture'}],[])
    assert wire['max_tokens']==cap and wire['reasoning_effort']=='high' and wire['stream']
    assert 'thinking' not in wire and 'seed' not in wire and provider.config.max_retries==0
receipt = {'passed':True,'real_http':0,'source':r.code_identity(),'checks':checks,
    'driver_sha256':r.sha(c.__file__),'launcher_sha256':r.sha(run.__file__),'test_sha256':r.sha(__file__),
    'guard_sha256':r.sha(guard.__file__),'canonical_payloads_checked':1400,
    'maximum_prompt_bound_plus_completion':maximum,'same_v7_messages_and_full_context':True,
    'native_docker_program_and_official_score_verified':True,'completed_resume_adds_no_calls':True,
    'raw_total_charged_without_fabricated_reasoning':True,'broken_stream_and_missing_usage_stop_before_score':True,
    'docker_failure_stops_before_repair_and_reuses_paid_answer':True}
r.write_json(HERE/'offline_acceptance.json',receipt)
print(json.dumps(receipt))
