"""Zero-real-HTTP checks using production graph, Docker, ledger, payload and resume."""
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
import run_experiment as run

c,r,HERE=run.c,run.r,run.HERE

def load(path):
    spec=importlib.util.spec_from_file_location('fixture_'+r.sha(path),path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

program=load(HERE/'packages/v8_programmatic/programs/source_fields.py')
bridge=load(HERE/'packages/v8_programmatic/integration.py')
checks=[]
def fields(q,s):
    x=program.run(None,{'question':q,'context':s})
    assert x['status']=='ok'
    rows=x['outputs']['source_fields']['fields']
    for v in rows:
        assert s[slice(*v['answer_span'])]==v['candidate']
        assert s[slice(*v['clue_span'])]==v['clue_text']
    return [v['candidate'] for v in rows]

assert fields('Who wrote the amber play?','[DOC][PAR]Clue | Who wrote the amber play? | Person A')==['Person A']
assert fields('Who wrote the amber play?','[DOC][PAR]Clue | Who wrote the amber play? | Person B')==['Person B']
assert fields('Who wrote the amber play?','[DOC][PAR]Clue | Who performed the amber play? | Person A')==[]
assert fields('Who did not write the amber play?','[DOC][PAR]Clue | Who wrote the amber play? | Person A')==[]
assert fields('Who wrote the amber play?','[DOC][PAR]Q: Who wrote the amber play?\n[DOC][PAR]Answer: Person A')==[]
assert fields('Who wrote the amber play?','[DOC][PAR]Q: Who wrote the amber play?\nAnswer: Person A')==['Person A']
assert fields('Who wrote the amber play?','[DOC][PAR]Who wrote the amber play?, Person A')==[]
assert fields('Who wrote the amber play?','[DOC][PAR]The amber play exists. No answer field is present.')==[]
checks.append({'name':'literal_extraction_8_cases','passed':True,'real_http':0})

# Dataset identifiers and known factual flashcards must not occur in executable assets.
all_tasks=r.read(r.DATASETS/'test.json')['tasks']
for a in ('v8_guidance','v8_programmatic'):
    bundle=r.read(HERE/'packages'/a/'bank/bank.json')
    content='\n'.join(x.get('guidance','')+x.get('source','') for k in ('skill','program') for x in bundle[k])
    assert not any(t['task_id'].split(':')[-1] in content for t in all_tasks)
    assert not any(v in content for v in ['Lenin','Hamburg','Timonium','gold_answers','evaluator_records','Domain Reference Facts'])
    for v in bundle['program']:assert v['allowed_tools']==[]
    assert bundle['trust']=='author_trusted'
assert [s['guidance'] for s in r.read(HERE/'packages/v8_guidance/bank/bank.json')['skill']]==[
    s['guidance'] for s in r.read(HERE/'packages/v8_programmatic/bank/bank.json')['skill']]
checks.append({'name':'no_answer_memory_and_same_guidance','passed':True,'real_http':0})

selected={}
for t in all_tasks:
    structure=bridge.has_source_fields(t['inputs']['context'],t['goal'])
    match=bool(fields(t['goal'],t['inputs']['context'])) if structure else False
    category='matched' if match else 'no_match' if structure else 'ordinary'
    selected.setdefault(category,t)
    if len(selected)==3:break
assert set(selected)=={'matched','no_match','ordinary'}

def exercise(failure=False):
    root=Path(tempfile.mkdtemp(prefix='searchqa_v8_fixture_',dir='/home/yangchengyu'))
    limits={'token_limit':1000000,'request_limit':40,'finish_reserve':0}
    for name in ('budget.json','inherited_budget.json'):r.write_json(root/name,{'limits':limits,'attempts':{},'unknown_billing':False})
    r.write_json(root/'authorization.json',{'status':'approved','fixture_only':True})
    r.write_json(root/'cost_carryforward_manifest.json',{'fixture_only':True})
    tasks=list(selected.values())
    r.write_json(root/'tasks.json',{'tasks':tasks})
    arms=run.arms()
    if failure:arms={'v8_programmatic':arms['v8_programmatic']}
    r.write_json(root/'spec.json',{'tasks':str(root/'tasks.json'),'arms':arms,'purpose':'offline fixture; production Docker; no real HTTP'})
    seen=[]
    def post(*args,**kwargs):
        payload=deepcopy(kwargs['json'])
        current=r.read(root/'fixture/current.json');arm=current['arm']
        task=next(t for t in tasks if t['task_id']==current['task_id'])
        material=json.loads(payload['messages'][1]['content'])
        assert material['goal']==task['goal'] and material['inputs']==task['inputs']
        assert payload['model']=='deepseek-v4-flash' and payload['max_tokens']==32768
        assert payload['thinking']['type']=='enabled' and payload['reasoning_effort']=='high'
        assert not payload.get('tools')
        should_fields=arm=='v8_programmatic' and bridge.has_source_fields(task['inputs']['context'],task['goal'])
        assert ('source_fields' in material)==should_fields
        if should_fields:
            expected=program.run(None,{'question':task['goal'],'context':task['inputs']['context']})['outputs']['source_fields']
            assert material['source_fields']==expected, 'Actual native producer output must reach solve unmodified'
        if arm.startswith('v8_'):
            for name in ('solve','evidence','conflict','span','surface','check'):
                card=(HERE/'guidance'/f'{name}.txt').read_text().strip()
                assert payload['messages'][0]['content'].count(card)==1
                assert card not in payload['messages'][1]['content']
        seen.append({'arm':arm,'task_id':task['task_id'],'payload':payload})
        if failure:raise requests.ConnectionError('Synthetic offline transport failure')
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:{
            'id':'offline','model':'fixture:deepseek-v4-flash',
            'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'<answer>fixture</answer>','reasoning_content':'synthetic'}}],
            'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c,'CAMPAIGN',root)
        c.prepare('fixture',root/'spec.json')
        plan=r.read(root/'fixture/plan.json')
        for a in arms:
            path=root/'fixture'/a/'config.json'
            cfg=r.read(path)
            cfg['experiment']['authored_bank_runtime']=r.read(root/'fixture'/a/'frozen_bank/imported_bank.json')['method_version']
            r.write_json(path,cfg);plan['files'][str(path)]=r.sha(path)
        r.write_json(root/'fixture/plan.json',plan)
        r.write_json(root/'fixture/plan_identity.json',{'sha256':r.digest(plan)})
        patch.setattr(r,'load_env',lambda p:None)
        patch.setenv('MODEL_API_KEY','offline-placeholder')
        patch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
        patch.setattr('atomic_skillgraph.agents.provider.OpenAICompatibleProvider._backoff',lambda *a,**k:None)
        patch.setattr(socket.socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
        with (root/'fixture.log').open('w') as f,redirect_stdout(f):
            if failure:
                from atomic_skillgraph.core.errors import BudgetExhausted
                with pytest.raises(BudgetExhausted):c.execute('fixture')
                assert not c.completed(root/'fixture','v8_programmatic')
                old=len(seen)
                with pytest.raises(AssertionError,match='Unresolved billing'):c.execute('fixture')
                assert len(seen)==old
            else:
                result=c.execute('fixture')
                assert result['status']=='completed' and len(seen)==9
                for arm in arms:
                    rows=c.completed(root/'fixture',arm);assert len(rows)==3
                    for task_id,row in rows.items():
                        assert row['execution']['all_authored_nodes_completed']
                        assert row['execution']['prediction']=='<answer>fixture</answer>'
                        assert sum(u['total_tokens'] for u in row['usage'])==5
                        attempts=row['execution'].get('attempts',[])
                        assert all(a['status']=='ok' for a in attempts)
                        if arm=='v8_programmatic':
                            q=next(t for t in tasks if t['task_id']==task_id)
                            expected=bridge.has_source_fields(q['inputs']['context'],q['goal'])
                            assert len(attempts)==(2 if expected else 1)
                            mapping=row['requests'][0]['node_response_mapping']
                            assert mapping['source_fields_delivered']==expected
                c.execute('fixture');assert len(seen)==9, 'Completed resume must not call HTTP'
        assert all(r.read(root/'fixture/freeze_audit.json').values())
    r.write_json(root/'intercepted_transport.json',seen)
    return {'name':'transport_stop' if failure else 'production_payload_graph_worker_resume','passed':True,
            'real_http':0,'intercepted_http':len(seen),'fixture':str(root)}

checks.extend([exercise(),exercise(True)])
receipt={'passed':True,'real_http':0,'checks':checks,'controller_sha256':r.sha(run.__file__),
         'driver_sha256':r.sha(c.__file__),'test_sha256':r.sha(__file__),
         'packages':{a:r.tree_identity(item['package']) for a,item in run.arms().items()}}
r.write_json(HERE/'offline_acceptance.json',receipt)
print(json.dumps(receipt))
