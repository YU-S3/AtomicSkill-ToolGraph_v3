"""CF2 state, resource, submission and HTTP regressions using production modules."""
from copy import deepcopy
import hashlib
import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.provider import OpenAICompatibleProvider, OpenAICompatibleConfig
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, RuntimeDecision, object_schema, validate_schema_instance
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.planner import Planner, dynamic
from atomic_skillgraph.empirical.prompts import BUILD, STEP
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.workspace import Workspace
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter, parse_alfworld_action
from atomic_skillgraph.harness.alfworld_simple import SimpleAlfWorld
from atomic_skillgraph.harness.benchmarks import OfficeAdapter, SpreadsheetAdapter
from atomic_skillgraph.harness.simple_protocol import Broker, HarnessActionResult, HarnessActionSpec, UnknownSideEffect
from test_empirical import config_for, program, worker


FEEDBACK = json.loads((Path(__file__).parent/'fixtures/cf2_public_feedback.json').read_text())['events']


class RecordedHarness:
    def __init__(self):
        self.rows = deepcopy(FEEDBACK)
        self.revision, self.calls = 0, []
        self.reject, self.inventory = False, None

    def primitive_action_schema(self): return AlfWorldAdapter.primitive_action_schema(self)
    def public_discovery_frame(self): return None
    def _close_backend(self): pass

    def reset(self, task):
        self.revision = 0
        return HarnessActionResult(True, 'Public reset', False, False, 0, [])

    def action_catalog(self):
        return [HarnessActionSpec(str(i), self.revision, row['name'], row['arguments'], '', '', {})
                for i, row in enumerate(self.rows.values())]

    def execute_action(self, action_id, revision):
        assert revision == self.revision
        row = list(self.rows.values())[int(action_id)]
        self.calls.append(deepcopy(row))
        self.revision += 1
        observation = self.inventory if row['name']=='INVENTORY' and self.inventory is not None else row['result']['observation']
        return HarnessActionResult(not (self.reject and row['name']=='MOVE'), observation, False, False, self.revision, [])


def alf():
    adapter = SimpleAlfWorld(RecordedHarness())
    adapter.reset(PublicTask('recorded', 'recorded-physical', 'obtain apple'))
    return adapter


def take(adapter): return adapter.call('TAKE', FEEDBACK['TAKE']['arguments'])
def move(adapter): return adapter.call('MOVE', FEEDBACK['MOVE']['arguments'])


def office(tmp_path, provider=None):
    corpus = tmp_path/'corpus'; corpus.mkdir(exist_ok=True)
    (corpus/'a.txt').write_bytes('first\r\n第二行 target\r\nlast'.encode())
    config = config_for(tmp_path/'bank')
    config['harness'].update(adapter='officeqa', corpus_root=str(corpus))
    config['runtime']['global_action_budget'] = 24
    adapter = OfficeAdapter({'office': {'answer': 'private'}}, config)
    adapter.reset(PublicTask('office','office-physical','find target'))
    system = EmpiricalSystem(config, harness=adapter, provider=provider)
    system.task_context = TaskContext(system.config['runtime'])
    return system


def http(monkeypatch, responses):
    monkeypatch.setenv('MODEL_API_KEY', 'test-only-key')
    seen, pending = [], iter(responses)
    def post(*args, **kwargs):
        seen.append(deepcopy(kwargs['json']))
        response = next(pending)
        if os.environ.get('CF2_EVIDENCE_DIR'):
            root=Path(os.environ['CF2_EVIDENCE_DIR']); root.mkdir(parents=True,exist_ok=True)
            test=os.environ.get('PYTEST_CURRENT_TEST','fixture')
            value={'test':test,'transport':'intercepted production requests.post; no model call','payload':seen[-1],
                   'finish_reason':response['choices'][0]['finish_reason'],'usage':response['usage']}
            (root/(hashlib.sha256(test.encode()).hexdigest()[:16]+'_'+str(len(seen))+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
        return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda: response)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    return seen


def response(actions=(), *, name='runtime_step', finish='tool_calls', content=''):
    return {'choices': [{'finish_reason': finish, 'message': {'role': 'assistant', 'content': content, 'reasoning_content': 'fixture-private',
        'tool_calls': [{'id': 'call_'+str(i), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(a)}}
                       for i,a in enumerate(actions)]}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


def test_t01_recorded_take_move_empty_inventory():
    a = alf(); take(a); assert a.observe()['held_objects']
    move(a); a.call('INVENTORY', {})
    assert a.observe()['held_objects']==[] and a.observe()['inventory_status']=='known'
    assert a.model_state()['held_objects']==[]


def test_t02_rejected_move_keeps_held_object():
    a = alf(); take(a); a.harness.reject=True
    assert not move(a)['accepted']
    assert a.observe()['held_objects']==[FEEDBACK['TAKE']['arguments']['object']]


def test_t03_unknown_inventory_cannot_assert_old_hold():
    a = alf(); take(a); a.harness.inventory='Inventory unavailable to parse'
    a.call('INVENTORY', {})
    assert a.model_state()['held_objects']=='unknown'
    assert 'inventory' not in a.model_state()['object_locations'].values()
    assert a.check_local({'target_query':'apple'}, {'object':'apple_1'}, [])=='unavailable'


def test_t04_old_take_is_not_a_current_local_success():
    a=alf(); broker=Broker(a,4)
    broker.call('TAKE', FEEDBACK['TAKE']['arguments']); broker.call('MOVE', FEEDBACK['MOVE']['arguments'])
    assert broker.check_local({'target_query':'apple'}, {'object':'apple_1'}, 0)!='passed'


def test_t05_program_and_agent_share_broker_and_state(worker,tmp_path):
    a=alf(); broker=Broker(a,4)
    broker.call('TAKE', FEEDBACK['TAKE']['arguments'])
    source='def run(ctx, inputs):\n    ctx.call("MOVE", '+repr(FEEDBACK['MOVE']['arguments'])+')\n    return {"status":"ok","outputs":{}}'
    p=program(source); p['allowed_tools']=['MOVE']
    bank=Bank(tmp_path/'bank'); p=bank.put('program',p)
    result=worker.execute(p,{},broker)
    assert result['status']=='ok', result
    assert a.observe()['held_objects']==[] and len(broker.events)==len(broker.context.results)==2
    bank.close()


def test_t06_parser_tool_spec_and_actual_http_preserve_semantics(monkeypatch,tmp_path):
    a=alf()
    name,args,_,_=parse_alfworld_action('heat apple 1 with microwave 1')
    assert name=='HEAT' and args=={'object':'apple_1','station':'microwave_1'}
    seen=http(monkeypatch,[response([{'source':'def run(ctx, inputs): return {}','trial_inputs':[]}],name='submit_program')])
    s=EmpiricalSystem(config_for(tmp_path/'bank'),harness=a)
    s.agent('tool_builder','Build',{'tools':a.tool_definitions(),'state':a.model_state()},'submit_program',BUILD)
    sent=json.loads(seen[0]['messages'][1]['content'])
    heat=next(t for t in sent['tools'] if t['name']=='HEAT')
    assert 'abstract heating' in heat['description'] and heat['input_schema']['required']==['object','station']
    assert {'MOVE','PUT'}.issubset({t['name'] for t in sent['tools']})
    s.close()


def test_t07_registered_handler_result_schemas(tmp_path,worker):
    s=office(tmp_path); specs={t['name']:t for t in s.adapter.tool_definitions()}
    for name,args in [('glob',{'pattern':'*.txt'}),('read',{'path':'a.txt','offset':0}),('grep',{'pattern':'target'}),('execute_python',{'source':'print(2+2)'})]:
        result=s.adapter.call(name,args)
        assert result['accepted'],result
        validate_schema_instance(result,specs[name]['result_schema'])
    a=alf()
    validate_schema_instance(take(a),next(t for t in a.tool_definitions() if t['name']=='TAKE')['result_schema'])
    s.close()


def test_t08_grep_read_unicode_and_crlf_share_offset(tmp_path):
    s=office(tmp_path)
    hit=s.adapter.call('grep',{'pattern':'target'})['data'][0]
    assert hit['line']==2 and hit['offset']==6
    data=s.adapter.call('read',{'path':hit['path'],'offset':hit['offset']})['data']
    assert data['text'].startswith('第二行 target\n')
    s.close()


def published(tmp_path):
    w=Workspace(tmp_path/'workspace'); stage=w.stage()
    (stage/'solution.py').write_text('print(1)'); (stage/'case1_result.xlsx').write_bytes(b'original')
    old=w.publish(stage,['solution.py','case1_result.xlsx'])
    return w,old


def test_t09_read_only_check_preserves_published_outputs(tmp_path):
    w,old=published(tmp_path); new=w.publish(w.stage(),[])
    assert old==new and set(new['outputs'])=={'solution.py','case1_result.xlsx'}


@pytest.mark.parametrize('change',['modify','delete'])
def test_t10_undeclared_changes_preserve_old_commit(tmp_path,change):
    w,old=published(tmp_path); stage=w.stage()
    if change=='modify': (stage/'solution.py').write_text('bad')
    else: (stage/'solution.py').unlink()
    with pytest.raises(ValueError,match='undeclared_output_change'): w.publish(stage,[])
    assert json.loads((w.root/'manifest.json').read_text())==old
    w.discard(stage)


def test_t11_explicit_delta_and_interrupted_publish(tmp_path,monkeypatch):
    import atomic_skillgraph.empirical.workspace as module
    w,old=published(tmp_path); stage=w.stage()
    (stage/'solution.py').write_text('print(2)'); (stage/'case1_result.xlsx').unlink()
    new=w.publish(stage,['solution.py'],['case1_result.xlsx'])
    assert new['outputs']==['solution.py'] and new['content_hash']!=old['content_hash']
    stage=w.stage(); (stage/'solution.py').write_text('print(3)')
    monkeypatch.setattr(module.os,'replace',lambda *a: (_ for _ in ()).throw(OSError('commit interrupted')))
    with pytest.raises(OSError): w.publish(stage,['solution.py'])
    assert json.loads((w.root/'manifest.json').read_text())==new


def run_steps(s,steps,plan=None,budget=24):
    pending=iter(steps); calls=[]
    def agent(*args,**kwargs): calls.append(args); return next(pending)
    executor=Executor(s.bank,agent,s.worker,Planner(s.bank,agent))
    result=executor.run(s.adapter.task,s.adapter,Broker(s.adapter,budget),plan or dynamic(s.adapter.task))
    return result,calls


def test_t12_explicit_plan_answer_submits_without_new_dynamic(tmp_path):
    s=office(tmp_path)
    plan=dynamic(s.adapter.task); plan['outputs']={'answer':{'from':'task','field':'answer'}}
    result,calls=run_steps(s,[{'action':'complete_node','outputs':{'answer':'final'}}],plan)
    assert result['prediction']=='final' and len(calls)==1 and result['reason']=='plan_submitted'
    s.close()


def test_t13_completed_file_plan_seals_registered_bundle(tmp_path):
    config=config_for(tmp_path/'bank')
    a=SpreadsheetAdapter({'s':{'public_files':{}}},config); a.reset(PublicTask('s','p','files'))
    stage=a.workspace.stage(); (stage/'solution.py').write_text('print(1)'); (stage/'case1_result.xlsx').write_bytes(b'file')
    a.workspace.publish(stage,['solution.py','case1_result.xlsx'])
    s=EmpiricalSystem(config,harness=a,provider=object())
    result,calls=run_steps(s,[{'action':'complete_node','outputs':{}}])
    assert result['reason']=='plan_submitted' and len(calls)==1
    assert a.submit(result['prediction'])['bundle']
    s.close()


def test_t14_intermediate_dictionary_is_not_an_answer(tmp_path):
    s=office(tmp_path)
    result,calls=run_steps(s,[{'action':'complete_node','outputs':{'found':'middle'}},{'action':'finish','answer':'final'}])
    assert len(calls)==2 and result['prediction']=='final'
    s.close()


def test_t15_last_read_gets_one_finish_only_without_more_retrieval(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':0}}]),
                           response([{'action':'finish','answer':'target'}],name='finish_answer')])
    s=office(tmp_path); broker=Broker(s.adapter,1,context=s.task_context)
    result=s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert len(broker.events)==1 and result['prediction']=='target' and len(seen)==2
    assert seen[-1]['tools'][0]['function']['parameters']['properties']['action']['enum']==['finish']
    assert len(s.requests)==2 and all(r['repair']==0 for r in s.requests)
    s.close()


def test_t16_terminal_does_not_trigger_finish_only(tmp_path):
    s=office(tmp_path); broker=Broker(s.adapter,1); broker.done=True
    s.executor.agent=lambda *a,**k: pytest.fail('terminal must not request another solve')
    result=s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert result['reason']=='environment_terminal'
    s.close()


@pytest.mark.parametrize('name,args',[('grep',{'pattern':'['}),('read',{'path':'missing','offset':0}),('read',{'path':'../secret','offset':0}),('read',{'path':'.','offset':0})])
def test_t17_corpus_input_errors_are_counted_normal_feedback(tmp_path,name,args):
    s=office(tmp_path); broker=Broker(s.adapter,24)
    result=broker.call(name,args)
    assert not result['accepted'] and broker.remaining_calls()==23 and not broker.unknown
    assert str(tmp_path) not in json.dumps(result)
    s.close()


def test_t18_missing_corpus_still_stops_as_infrastructure(tmp_path):
    s=office(tmp_path)
    (s.adapter.corpus/'a.txt').unlink(); s.adapter.corpus.rmdir()
    broker=Broker(s.adapter,24)
    with pytest.raises(UnknownSideEffect): broker.call('glob',{'pattern':'*'})
    assert broker.unknown and broker.events[-1]['state']=='unknown'
    s.close()


@pytest.mark.parametrize('count',[2,3])
def test_t19_actual_http_pure_read_batch_has_no_repair(tmp_path,monkeypatch,count):
    actions=[{'action':'call_tool','name':'glob','arguments':{'pattern':str(i)+'*'}} for i in range(count)]
    seen=http(monkeypatch,[response(actions),response([{'action':'finish','answer':'done'}])])
    s=office(tmp_path); broker=Broker(s.adapter,24,context=s.task_context)
    result=s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert len(broker.events)==count and len(seen)==2 and result['prediction']=='done'
    assert len({e['call_id'] for e in broker.events})==count and all(r['repair']==0 for r in s.requests)
    s.close()


@pytest.mark.parametrize('bad',[{'action':'finish','answer':'x'},{'action':'call_program','name':'p','arguments':{}},{'action':'call_tool','name':'execute_python','arguments':{'source':'print(1)'}}])
def test_t20_mixed_batch_is_repaired_before_any_dispatch(tmp_path,monkeypatch,bad):
    good={'action':'call_tool','name':'glob','arguments':{'pattern':'*'}}
    seen=http(monkeypatch,[response([good,bad]),response([{'action':'finish','answer':'done'}])])
    s=office(tmp_path); broker=Broker(s.adapter,24,context=s.task_context)
    s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert not broker.events and len(seen)==2 and s.requests[-1]['repair']==1
    s.close()


def test_t21_bad_regex_does_not_cancel_other_pure_read(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'grep','arguments':{'pattern':'['}},
                                   {'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':0}}]),
                           response([{'action':'finish','answer':'done'}])])
    s=office(tmp_path); broker=Broker(s.adapter,24,context=s.task_context)
    s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert [e['result']['accepted'] for e in broker.events]==[False,True] and len(seen)==2
    s.close()


def test_t22_saved_batch_resumes_without_repeat_or_usage(tmp_path,monkeypatch):
    actions=[{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':i}} for i in [0,6]]
    seen=http(monkeypatch,[response(actions),response([{'action':'finish','answer':'done'}])])
    s=office(tmp_path); checkpoint=TaskCheckpoint(tmp_path/'checkpoint')
    s.executor.checkpoint=checkpoint
    broker=Broker(s.adapter,24,context=s.task_context,journal=checkpoint.native_events)
    original=checkpoint.advance
    calls=0
    def interrupt(stage,**values):
        nonlocal calls
        original(stage,**values)
        if values.get('executor_state',{}).get('pending_step') and len(broker.events)==1:
            calls+=1
            if calls==1: raise KeyboardInterrupt('crash after first saved read')
    monkeypatch.setattr(checkpoint,'advance',interrupt)
    with pytest.raises(KeyboardInterrupt): s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    monkeypatch.setattr(checkpoint,'advance',original)
    resumed=Broker(s.adapter,24,context=TaskContext(s.config['runtime']),journal=checkpoint.native_events)
    resumed.events=json.loads((checkpoint.root/'native_events.json').read_text())
    result=s.executor.run(s.adapter.task,s.adapter,resumed,dynamic(s.adapter.task))
    assert result['prediction']=='done' and len(resumed.events)==2 and len(seen)==2
    assert sum(u.to_dict()['total_tokens'] for u in s.usage.events)==10
    s.close()


def test_t23_large_result_ref_crosses_actual_http_without_copy(tmp_path,monkeypatch):
    s=office(tmp_path); context=s.task_context
    large=['resource'+str(i) for i in range(1000)]
    rid=context.register('large',{'data':large})
    seen=http(monkeypatch,[response([{'action':'complete_node','outputs':{},'output_refs':{'matches':{'result_id':rid,'path':['data']}}}])])
    decision=s.agent('runtime','Use result references',{'completed_results':context.view(rid)},'runtime_step',STEP)
    assert isinstance(decision,RuntimeDecision) and decision.actions[0]['output_refs']['matches']['result_id']==rid
    assert 'resource999' not in json.dumps(seen[0]['messages'])
    assert context.resolve(decision.actions[0]['output_refs']['matches'])==large
    s.close()


@pytest.mark.parametrize('ref',[{'result_id':'other-episode','path':[]},{'result_id':'future','path':[]},{'result_id':'current','path':[-1]},{'result_id':'current','path':['*']}])
def test_t24_invalid_result_refs_do_not_fetch_or_forge(ref):
    context=TaskContext(); rid=context.register('event',{'data':[1]})
    if ref['result_id']=='current': ref={**ref,'result_id':rid}
    with pytest.raises(ValueError): context.resolve(ref)
    with pytest.raises(ValueError): context.bind({'field':1},{'field':{'result_id':rid,'path':['data']}})


def test_t25_result_read_is_local_counted_and_untruncated(tmp_path):
    s=office(tmp_path); broker=Broker(s.adapter,2,context=s.task_context)
    text='α'*3000; rid=s.task_context.register('original',{'data':text})
    assert s.task_context.view(rid)['data']['truncated']
    result=broker.call('read_result',{'result_id':rid,'path':['data'],'offset':0})
    assert result['data']==text and result['unit']=='unicode_codepoint_0_based'
    assert broker.remaining_calls()==1 and not broker.events[0]['backend_invoked'] and broker.environment_steps==0
    s.close()


@pytest.mark.parametrize('failure',['budget','infrastructure'])
def test_t22_batch_remaining_calls_are_explicitly_not_executed(tmp_path,monkeypatch,failure):
    actions=[{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':i}} for i in [0,6,8]]
    seen=http(monkeypatch,[response(actions),response([{'action':'finish','answer':'done'}],name='finish_answer')])
    s=office(tmp_path); checkpoint=TaskCheckpoint(tmp_path/'checkpoint'); s.executor.checkpoint=checkpoint
    broker=Broker(s.adapter,1 if failure=='budget' else 24,context=s.task_context,journal=checkpoint.native_events)
    if failure=='infrastructure':
        def broken(*a,**k): raise OSError('storage unavailable')
        s.adapter.call=broken
        with pytest.raises(UnknownSideEffect): s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
        history=checkpoint.state['executor_state']['history']
    else:
        history=s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))['history']
    assert len(broker.events)==1
    assert [h['call_id'] for h in history if h.get('status','').startswith('not_executed')]==['call_1','call_2']
    s.close()


@pytest.mark.parametrize('invalid',['over_limit','bad_arguments'])
def test_t20_full_batch_validation_precedes_dispatch(tmp_path,monkeypatch,invalid):
    good={'action':'call_tool','name':'glob','arguments':{'pattern':'*'}}
    actions=[good]*4 if invalid=='over_limit' else [good,{'action':'call_tool','name':'read','arguments':{'offset':0}}]
    seen=http(monkeypatch,[response(actions),response([{'action':'finish','answer':'done'}])])
    s=office(tmp_path); broker=Broker(s.adapter,24,context=s.task_context)
    s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert not broker.events and len(seen)==2 and s.requests[-1]['repair']==1
    s.close()


def test_t25_program_local_result_reads_share_native_and_invocation_limits(tmp_path,worker):
    s=office(tmp_path); broker=Broker(s.adapter,2)
    rid=broker.context.register('original',{'data':'α'*3000})
    asset=program("def run(ctx, inputs):\n    value=ctx.read_result(inputs['id'], path=['data'])\n    return {'status':'ok','outputs':{'text':value['data']}}",inputs=object_schema({'id':{'type':'string'}},['id']),
                  outputs=object_schema({'text':{'type':'string'}},['text']))
    asset['allowed_tools']=['read_result']; asset=s.bank.put('program',asset)
    result=worker.execute(asset,{'id':rid},broker)
    assert result['status']=='ok' and result['calls']==1 and len(result['outputs']['text'])==3000
    assert broker.remaining_calls()==1 and not broker.events[0]['backend_invoked'] and broker.environment_steps==0
    assert broker.context.preview(list(range(41)))['truncated']
    s.close()


@pytest.mark.parametrize('invalid',['input','traversal','symlink'])
def test_t11_publishing_rejects_unauthorized_output_paths(tmp_path,invalid):
    w,old=published(tmp_path); stage=w.stage()
    if invalid=='symlink':
        (stage/'outside').symlink_to(tmp_path/'workspace'/'manifest.json'); declared=['outside']
    elif invalid=='input': declared=['inputs/secret.txt']
    else: declared=['../manifest.json']
    with pytest.raises(ValueError): w.publish(stage,declared)
    assert json.loads((w.root/'manifest.json').read_text())==old
    w.discard(stage)


def test_t18_decoding_failure_is_not_an_input_error(tmp_path):
    s=office(tmp_path); (s.adapter.corpus/'bad.txt').write_bytes(b'\xff')
    broker=Broker(s.adapter,24)
    with pytest.raises(UnknownSideEffect): broker.call('read',{'path':'bad.txt','offset':0})
    assert broker.events[0]['state']=='unknown'
    s.close()



def test_t03_known_empty_inventory_invalidates_prior_locations():
    a=alf(); take(a); a.call('INVENTORY',{})
    assert a.model_state()['held_objects']==[] and 'inventory' not in a.model_state()['object_locations'].values()


def test_acceptance_preserves_actual_public_experience_without_history_duplication():
    from atomic_skillgraph.experiments.run_empirical_acceptance import public_experience
    source={'tools':[{'name':'read','arguments':{},'result':{'data':'raw'}}],
            'score':{'hard':False},'execution':{'prediction':None,'attempts':[],'history':[{'tool':'read'}]},
            'requests':['not training material']}
    experience=public_experience(source)
    assert experience=={key:source[key] for key in ('tools','score','execution')}
    assert experience['tools'] is not source['tools'] and 'requests' not in experience
