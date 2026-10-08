"""CF3 production control paths; HTTP is intercepted and Programs use Docker."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os

import pytest

from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema, output_view, ValueStore
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.prompts import STEP
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import office, alf
from test_cf2_realization import usable, skill
from test_empirical import program, worker


def transport(monkeypatch, responses):
    monkeypatch.setenv('MODEL_API_KEY', 'test-only-key')
    pending, sent = iter(responses), []
    def post(*args, **kwargs):
        sent.append(deepcopy(kwargs['json']))
        name, value = next(pending)
        value = value() if callable(value) else value
        message = {'role': 'assistant', 'content': '' if name else value, 'reasoning_content': 'fixture-private'}
        if name:
            actions = value if isinstance(value, list) else [value]
            message['tool_calls'] = [{'id': 'call_' + str(i), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(action)}} for i,action in enumerate(actions)]
        response = {'choices': [{'finish_reason': 'tool_calls' if name else 'stop', 'message': message}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
        if os.environ.get('CF3_EVIDENCE_DIR'):
            root = Path(os.environ['CF3_EVIDENCE_DIR']); root.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(os.environ.get('PYTEST_CURRENT_TEST', '').encode()).hexdigest()[:16]
            (root / (key + '_' + str(len(sent)) + '.json')).write_text(json.dumps({
                'test': os.environ.get('PYTEST_CURRENT_TEST'), 'transport': 'intercepted production HTTP; no model call',
                'payload': sent[-1], 'response': response}, ensure_ascii=False, indent=2))
        return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda: response)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    return sent


def workflow(nodes, outputs=None):
    return {'interface_version': 'empirical.workflow.v2', 'goal': 'public task',
            'nodes': nodes, 'outputs': outputs or {}}


def execute(system, plan):
    planned = system.planner.plan(system.adapter.task, system.adapter)
    broker = Broker(system.adapter, 24, context=system.task_context)
    result = system.executor.run(system.adapter.task, system.adapter, broker, planned)
    return result, broker


def answer_program(system, worker, inputs=None):
    system.worker = system.executor.worker = worker
    p = program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'answer':'private'}}",
                inputs=inputs, outputs=object_schema({'answer': {'type':'string'}}, ['answer']))
    p['result_role'] = 'final_answer'
    return usable(system.bank, p)


def test_t01_t07_reference_skill_does_not_take_over_or_impose_output(tmp_path, monkeypatch, worker):
    s = office(tmp_path)
    full = s.bank.put('skill', skill('whole task', outputs=object_schema({
        'completed': {'type':'boolean'}, 'action_sequence': {'type':'array'}}, ['completed','action_sequence'])))
    unwanted = usable(s.bank, program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{}}"))
    s.bank.put('implementation', {'skill_id':full['id'], 'program_id':unwanted['id']})
    p = answer_program(s, worker, object_schema({'resource_id': {'type':'string'}}, ['resource_id']))
    plan = workflow([{'id':'find', 'execution_mode':'dynamic', 'goal':'Find a readable resource',
        'reference_skill_ids':[full['id']], 'args':{}},
        {'id':'answer', 'execution_mode':'program', 'program_id':p['id'],
         'args':{'resource_id':{'from':'find','field':'resource_id'}}, 'after':['find']}],
        {'answer':{'from':'answer','field':'answer'}})
    sent = transport(monkeypatch, [('submit_plan', {'mode':'compose','workflow':plan}),
                                  ('runtime_step', {'action':'complete_node','outputs':{'resource_id':'a.txt'}})])
    result, _ = execute(s, plan)
    assert result['prediction'] == 'private' and len(sent) == 2
    assert [a['program_id'] for a in result['attempts']] == [p['id']]
    material = json.loads(sent[1]['messages'][1]['content'])
    assert material['handoff']['required_handoff_fields'] == ['resource_id']
    assert material['node']['interface']['output_schema'] is None
    assert s.bank.routes(plan['nodes'][0]) == []
    s.close()


@pytest.mark.parametrize('mode', ['skill','program'])
def test_t02_t03_bound_capability_ready_has_no_runtime_confirmation(tmp_path, monkeypatch, worker, mode):
    s = office(tmp_path)
    p = answer_program(s, worker)
    sk = s.bank.put('skill', skill('Complete the public answer', outputs=p['output_schema'], result_role='final_answer'))
    s.bank.put('implementation', {'skill_id':sk['id'],'program_id':p['id']})
    node = {'id':'answer','execution_mode':mode,'args':{},'purpose':'parent intent'}
    node['skill_id' if mode == 'skill' else 'program_id'] = sk['id'] if mode == 'skill' else p['id']
    plan = workflow([node], {'answer':{'from':'answer','field':'answer'}})
    sent = transport(monkeypatch, [('submit_plan', {'mode':'compose','workflow':plan})])
    result, _ = execute(s, plan)
    assert result['prediction'] == 'private' and len(sent) == 1 and len(result['attempts']) == 1
    assert result['values'][0]['origin'] == 'program'
    s.close()


@pytest.mark.parametrize('business_dict', [False, True])
def test_t04_t09_t10_new_program_graph_aliases_and_business_values(tmp_path, monkeypatch, worker, business_dict):
    s = office(tmp_path); s.worker = s.executor.worker = worker
    value = {'literal':'real business data', 'from':'ordinary key'} if business_dict else 'a.txt'
    schema = {'type':'object'} if business_dict else {'type':'string'}
    first = usable(s.bank, program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'actual':" + repr(value) + "}}",
        outputs=object_schema({'actual':schema}, ['actual'])))
    second = usable(s.bank, program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'value':inputs['value']}}",
        inputs=object_schema({'value':schema}, ['value']), outputs=object_schema({'value':schema}, ['value'])))
    plan = workflow([{'id':'produce','execution_mode':'program','program_id':first['id'],'args':{},'output_aliases':{'actual':'item'}},
        {'id':'consume','execution_mode':'program','program_id':second['id'],'args':{'value':{'from':'produce','field':'item'}}}],
        {'answer':{'from':'consume','field':'value'}})
    sent = transport(monkeypatch, [('submit_plan', {'mode':'compose','workflow':plan})])
    s.bank.put('skill', skill('public task'))
    result, _ = execute(s, plan)
    assert result['prediction'] == value and len(sent) == 1 and len(result['attempts']) == 2
    assert result['attempts'][0]['outputs_consumed']
    assert result['values'][0]['outputs'] == {'actual':value}
    assert result['values'][0]['handoff_view'] == {'actual':value,'item':value}
    for aliases in [{'missing':'x'}, {'actual':'other','other':'other'}]:
        with pytest.raises(ValueError): output_view({'actual':'one','other':'two'}, aliases)
    s.close()


def test_t08_t11_pending_handoff_is_patched_without_search_or_replan(tmp_path, monkeypatch, worker):
    s = office(tmp_path)
    p = answer_program(s, worker, object_schema({'resource_id':{'type':'string'}}, ['resource_id']))
    s.bank.put('skill', skill('find public document'))
    plan = workflow([{'id':'find','execution_mode':'dynamic','goal':'Find a document','args':{}},
        {'id':'answer','execution_mode':'program','program_id':p['id'],'args':{'resource_id':{'from':'find','field':'resource_id'}}}],
        {'answer':{'from':'answer','field':'answer'}})
    sent = transport(monkeypatch, [('submit_plan', {'mode':'compose','workflow':plan}),
        ('runtime_step', {'action':'call_tool','name':'glob','arguments':{'pattern':'*.txt'}}),
        ('runtime_step', lambda: {'action':'complete_node','output_refs':{
            'object_id':{'result_id':s.task_context.scope+':0','path':['data',0]}}}),
        ('runtime_step', {'action':'patch_node','patch':{'target_node':'find','kind':'handoff','reason':'Use actual output',
                                                      'output_aliases':{'object_id':'resource_id'}}})])
    result, broker = execute(s, plan)
    assert result['prediction'] == 'private' and result['plan_revisions'] == 0
    assert len(broker.events) == 1 and len(sent) == 4
    error = next(x for x in result['history'] if x.get('error') == 'handoff_error')
    assert error['feedback']['missing_fields'] == ['resource_id'] and result['pending_outputs'] == {}
    pending = json.loads(sent[3]['messages'][1]['content'])['handoff']['pending_outputs']
    assert pending['output_refs']['object_id']['path'] == ['data',0]
    assert result['attempts'][0]['outputs_consumed']
    s.close()


def test_t11_args_patch_readies_program_and_invalid_patch_keeps_plan(tmp_path, monkeypatch, worker):
    s = office(tmp_path)
    p = answer_program(s, worker, object_schema({'resource_id':{'type':'string'}}, ['resource_id']))
    s.bank.put('skill',skill())
    plan = workflow([{'id':'answer','execution_mode':'program','program_id':p['id'],
                     'args':{'resource_id':{'unresolved':'public resource'}}}], {'answer':{'from':'answer','field':'answer'}})
    sent = transport(monkeypatch, [('submit_plan', {'mode':'compose','workflow':plan}),
        ('runtime_step', {'action':'patch_node','patch':{'target_node':'unknown','kind':'args','reason':'invalid','args':{'resource_id':{'literal':'bad'}}}}),
        ('runtime_step', {'action':'patch_node','patch':{'target_node':'answer','kind':'args','reason':'public file','args':{'resource_id':{'literal':'a.txt'}}}})])
    result, broker = execute(s, plan)
    assert result['prediction'] == 'private' and result['plan_revisions'] == 0 and not broker.events
    patches = [x for x in result['history'] if 'patch_node' in x]
    assert len(patches) == 1 and patches[0]['before'] == plan and len(sent) == 3
    s.close()


def test_t12_repeated_handoff_failure_uses_one_replan_and_one_escape(tmp_path):
    s = office(tmp_path)
    plan = dynamic(s.adapter.task); plan['outputs'] = {'answer':{'from':'task','field':'answer'}}
    calls = []
    def bad(*args, **kwargs):
        calls.append(args[2])
        return {'action':'complete_node','outputs':{'wrong':'same'},'detail':str(len(calls))}
    class Replanner:
        def plan(self,*args,**kwargs): return deepcopy(plan)
    from atomic_skillgraph.empirical.executor import Executor
    result = Executor(s.bank,bad,s.worker,Replanner()).run(s.adapter.task,s.adapter,Broker(s.adapter,24),plan)
    assert result['reason'] == 'repeated_unchanged_failure'
    assert result['plan_revisions'] == result['dynamic_escapes'] == 1 and len(calls) <= 6
    assert not result['values'] and result['pending_outputs']
    s.close()


def test_t13_same_prompt_new_decisions_do_not_recover_old_response(tmp_path, monkeypatch):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path/'checkpoint')
    sent = transport(monkeypatch, [('runtime_step',{'action':'finish','answer':'first'}),('runtime_step',{'action':'finish','answer':'second'})])
    first = s.agent('runtime','same',{},'runtime_step',STEP)
    second = s.agent('runtime','same',{},'runtime_step',STEP)
    assert first.actions[0]['answer'] == 'first' and second.actions[0]['answer'] == 'second' and len(sent) == 2
    assert len({r['logical_decision_id'] for r in s.requests}) == 2
    assert all(d['status'] == 'applied' for d in s.checkpoint.state['decisions'].values())
    s.close()


def test_t14_same_unapplied_decision_recovers_response_once(tmp_path, monkeypatch):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path/'checkpoint'); s.audit_path=tmp_path/'requests.json'
    sent = transport(monkeypatch,[('runtime_step',{'action':'finish','answer':'private'})])
    first=s.agent('runtime','same',{},'runtime_step',STEP,owner_state_version=0)
    decision=s.last_decision_id; s.close()
    restored=office(tmp_path); restored.checkpoint=TaskCheckpoint(tmp_path/'checkpoint'); restored.audit_path=tmp_path/'requests.json'
    second=restored.agent('runtime','same',{},'runtime_step',STEP,owner_state_version=0)
    assert first==second and restored.last_decision_id==decision and len(sent)==1
    saved=json.loads(restored.audit_path.read_text())
    assert len(saved['usage'])==1 and saved['usage'][0]['total_tokens']==5
    restored.checkpoint.commit_decision(decision,'applied')
    restored.close()


def test_t15_rejected_response_and_feedback_survive_crash(tmp_path, monkeypatch):
    s=office(tmp_path); cp=TaskCheckpoint(tmp_path/'checkpoint'); s.checkpoint=s.executor.checkpoint=cp
    plan=dynamic(s.adapter.task); plan['outputs']={'answer':{'from':'task','field':'answer'}}
    sent=transport(monkeypatch,[('runtime_step',{'action':'complete_node','outputs':{'wrong':'value'}}),
                               ('runtime_step',{'action':'finish','answer':'private'})])
    original=cp.advance
    def interrupt(stage,**values):
        original(stage,**values)
        if values.get('executor_state',{}).get('owner_version')==1:
            raise KeyboardInterrupt('after atomic rejection')
    monkeypatch.setattr(cp,'advance',interrupt)
    with pytest.raises(KeyboardInterrupt): s.executor.run(s.adapter.task,s.adapter,Broker(s.adapter,24,context=s.task_context),plan)
    cp=TaskCheckpoint(cp.root); s.checkpoint=s.executor.checkpoint=cp
    result=s.executor.run(s.adapter.task,s.adapter,Broker(s.adapter,24,context=TaskContext()),plan)
    assert result['prediction']=='private' and len(sent)==2
    assert [d['status'] for d in cp.state['decisions'].values()]==['rejected','applied']
    assert len({r['logical_decision_id'] for r in s.requests})==2
    s.close()


def test_t16_three_read_batch_crash_after_second_keeps_usage_and_budget(tmp_path, monkeypatch):
    s=office(tmp_path); cp=TaskCheckpoint(tmp_path/'checkpoint'); s.checkpoint=s.executor.checkpoint=cp
    actions=[{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':i}} for i in [0,6,8]]
    sent=transport(monkeypatch,[('runtime_step',actions),('runtime_step',{'action':'finish','answer':'private'})])
    broker=Broker(s.adapter,24,context=s.task_context,journal=cp.native_events)
    original=cp.advance
    def interrupt(stage,**values):
        original(stage,**values)
        if values.get('executor_state',{}).get('pending_step') and len(broker.events)==2:
            raise KeyboardInterrupt('after second saved read')
    monkeypatch.setattr(cp,'advance',interrupt)
    with pytest.raises(KeyboardInterrupt): s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    cp=TaskCheckpoint(cp.root); s.checkpoint=s.executor.checkpoint=cp
    resumed=Broker(s.adapter,24,context=TaskContext(),journal=cp.native_events)
    resumed.events=json.loads((cp.root/'native_events.json').read_text())
    result=s.executor.run(s.adapter.task,s.adapter,resumed,dynamic(s.adapter.task))
    assert result['prediction']=='private' and len(sent)==2 and len(resumed.events)==3
    assert resumed.remaining_calls()==21 and sum(e.usage.total_tokens for e in s.usage.events)==10
    s.close()


def test_t17_t18_escape_keeps_state_and_explicit_program_takeover(tmp_path, monkeypatch, worker):
    s=office(tmp_path); p=answer_program(s,worker)
    revised=dynamic(s.adapter.task)
    sent=transport(monkeypatch,[('runtime_step',{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':0}}),
        ('runtime_step',{'action':'revise_plan','workflow':revised}),
        ('runtime_step',{'action':'revise_plan','workflow':revised}),
        ('runtime_step',{'action':'call_program','name':p['id'],'arguments':{}})])
    broker=Broker(s.adapter,24,context=s.task_context)
    result=s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    assert result['prediction']=='private' and result['plan_revisions']==result['dynamic_escapes']==1
    assert len(broker.events)==1 and broker.remaining_calls()==23 and len(result['attempts'])==1
    assert s.task_context.model_memory() and len(sent)==4
    s.close()


def test_t19_loop_detection_ignores_budget_revision_but_keeps_new_content():
    c=TaskContext(); c.active_node='same'
    for i in range(4):
        c.observe_progress({'name':'look','arguments':{},'progress_before':'state','progress_after':'state',
                            'result':{'observation':{'text':'same','revision':i,'remaining_calls':100-i},'data':{}}})
    assert c.loop_feedback['hits']>=2 and c.loop_feedback['pattern_length']==1
    exploratory=TaskContext(); exploratory.active_node='same'
    for i in range(12):
        exploratory.observe_progress({'name':'read','arguments':{'offset':i},'progress_before':'state','progress_after':'state',
                                      'result':{'data':{'text':'new page '+str(i)}}})
    assert exploratory.loop_feedback is None


@pytest.mark.parametrize('loop', [False, True])
def test_t19_production_accepted_loop_and_new_document_windows(tmp_path,monkeypatch,loop):
    s=office(tmp_path)
    actions=[('runtime_step',{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':i}})
             for i in ([0]*5 if loop else [0,6,8])]
    sent=transport(monkeypatch,[*actions,('runtime_step',{'action':'finish','answer':'private'})])
    broker=Broker(s.adapter,24,context=s.task_context)
    result=s.executor.run(s.adapter.task,s.adapter,broker,dynamic(s.adapter.task))
    if loop:
        assert result['reason']=='repeated_unchanged_failure'
        assert result['plan_revisions']==result['dynamic_escapes']==1 and len(broker.events)<=5
        assert json.loads(sent[2]['messages'][1]['content'])['recovery']['loop_feedback']
    else:
        assert result['prediction']=='private' and result['plan_revisions']==result['dynamic_escapes']==0
        assert len(broker.events)==3
    assert all(e['result']['accepted'] for e in broker.events)
    s.close()


def test_t11_t25_frozen_detach_is_only_a_task_projection(tmp_path,monkeypatch,worker):
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.experiments.formal_log import tree_identity
    s=office(tmp_path); p=answer_program(s,worker,object_schema({'missing':{'type':'string'}},['missing']))
    sk=s.bank.put('skill',skill('whole answer',inputs=p['input_schema'],outputs=p['output_schema']))
    s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    frozen_path=tmp_path/'frozen'; s.bank.freeze(frozen_path); s.bank.close()
    s.bank=s.executor.bank=s.planner.bank=Bank(frozen_path,readonly=True)
    s.executor.frozen=True; before=tree_identity(frozen_path)
    plan=workflow([{'id':'wrong','execution_mode':'skill','skill_id':sk['id'],'args':{}}])
    sent=transport(monkeypatch,[('submit_plan',{'mode':'compose','workflow':plan}),
        ('runtime_step',{'action':'patch_node','patch':{'target_node':'wrong','kind':'detach',
                                                     'dynamic_goal':'Answer from the current public state','reason':'Input is unavailable here'}}),
        ('runtime_step',{'action':'finish','answer':'private'})])
    result,_=execute(s,plan)
    assert result['prediction']=='private' and not result['attempts'] and result['plan_revisions']==0
    node=result['plan']['nodes'][0]
    assert node['execution_mode']=='dynamic' and node['reference_skill_ids']==[sk['id']]
    assert s.bank.get(p['id'])['state']=='usable' and tree_identity(frozen_path)==before and len(sent)==3
    s.close()


def test_t20_planner_http_includes_stable_unavailable_tool_semantics(tmp_path, monkeypatch):
    s=office(tmp_path); s.adapter=alf(); s.bank.put('skill',skill('public goal'))
    sent=transport(monkeypatch,[('submit_plan',{'mode':'compose','workflow':dynamic(PublicTask('t','p','public goal'))})])
    s.planner.plan(PublicTask('t','p','public goal'),s.adapter)
    material=json.loads(sent[0]['messages'][1]['content'])
    stable={t['name']:t for t in material['tool_definitions']}
    current={t['name']:t for t in material['current_tools']}
    assert 'HEAT' in stable and 'abstract heating' in stable['HEAT']['description']
    assert set(current).issubset(stable) and sent[0]['tools'][0]['function']['parameters']['type']=='object'
    s.close()


def test_t22_worker_requires_envelope_without_faking_success(tmp_path, worker):
    s=office(tmp_path)
    for source in ["def run(ctx, inputs): return {'answer':'private'}", "def run(ctx, inputs): return {'status':'ok'}"]:
        p=s.bank.put('program',program(source))
        result=worker.execute(p,{},Broker(s.adapter,24))
        assert result['status']=='execution_error' and not s.bank.attempts(p['id'])
    s.close()


@pytest.mark.parametrize('surface',['exact_catalog','named_tools'])
def test_t21_worker_uses_real_toolview_and_toolresult_shapes(tmp_path,worker,surface):
    from atomic_skillgraph.empirical.program_worker import public_program_abi
    s=office(tmp_path)
    if surface=='exact_catalog':
        s.adapter=alf()
        source="def run(ctx, inputs):\n    tool=next(t for t in ctx.available_tools() if t['name']=='TAKE')\n    r=ctx.call(tool['name'],tool['current_arguments'][0])\n    return {'status':'ok','outputs':{'accepted':r['accepted']}}"
        allowed=['TAKE']
    else:
        source="def run(ctx, inputs):\n    tool=next(t for t in ctx.available_tools() if t['name']=='glob')\n    r=ctx.call(tool['name'],{'pattern':'*.txt'})\n    return {'status':'ok','outputs':{'accepted':r['accepted']}}"
        allowed=['glob']
    tools=s.adapter.available_tools()
    abi=public_program_abi(s.adapter.tool_definitions(),current_tools=tools,tool_surface=surface)
    assert abi['current_tool_surface']==surface and abi['program_result_schema']['required']==['status','outputs']
    p=program(source,outputs=object_schema({'accepted':{'type':'boolean'}},['accepted'])); p['allowed_tools']=allowed
    p=s.bank.put('program',p)
    broker=Broker(s.adapter,24)
    result=worker.execute(p,{},broker)
    assert result['status']=='ok' and result['outputs']['accepted'] and len(broker.events)==1
    s.close()


def test_legacy_skill_selection_requires_explicit_mode_without_rewriting_bank(tmp_path,monkeypatch,worker):
    s=office(tmp_path); p=answer_program(s,worker)
    sk=s.bank.put('skill',skill('public answer',outputs=p['output_schema']))
    s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    old={'id':'workflow_legacy_fixture','goal':'public answer','nodes':[{
        'id':'old','goal':'ambiguous local wording','skill_id':sk['id'],'args':{}}],
        'outputs':{'answer':{'from':'old','field':'answer'}}}
    # Existing legacy bytes are inserted only as a compatibility fixture, never experiment assets.
    s.bank.db.execute('INSERT INTO assets VALUES(?,?,?)',('workflow',old['id'],json.dumps(old)));s.bank.db.commit()
    before=s.bank.digest()
    sent=transport(monkeypatch,[('submit_plan',{'mode':'select','workflow_id':old['id'],'node_args':{},'node_modes':{'old':'skill'}})])
    result,_=execute(s,old)
    assert result['prediction']=='private' and len(sent)==1 and s.bank.digest()==before
    assert s.bank.get(old['id'])==old and result['plan']['nodes'][0]['execution_mode']=='skill'
    assert 'goal' not in result['plan']['nodes'][0]
    s.close()


def test_t26_unknown_inflight_program_stops_without_dispatch_or_credit(tmp_path,monkeypatch):
    from atomic_skillgraph.harness.simple_protocol import UnknownSideEffect
    s=office(tmp_path); s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint')
    s.checkpoint.advance('task_started',program_started=True)
    before=s.bank.digest()
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',lambda *a,**kw:pytest.fail('unknown execution cannot request a model'))
    monkeypatch.setattr(s.worker,'execute',lambda *a,**kw:pytest.fail('unknown Program cannot be repeated'))
    with pytest.raises(UnknownSideEffect): s.run_task(s.adapter.task,learn=False,attempt_id='inflight')
    assert s.bank.digest()==before and s.checkpoint.state['program_started']
    assert s.bank.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0
    s.close()


def test_t27_single_solver_response_recovers_without_second_solve(tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    from atomic_skillgraph.harness.benchmarks import AnswerAdapter
    from test_empirical import config_for
    settings=config_for(tmp_path/'bank'); settings['harness']['adapter']='searchqa'
    adapter=AnswerAdapter('searchqa',{'q':{'answers':['private']}})
    s=EmpiricalSystem(settings,harness=adapter)
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');s.audit_path=tmp_path/'requests.json'
    sent=transport(monkeypatch,[(None,'private')])
    original=adapter.submit
    monkeypatch.setattr(adapter,'submit',lambda *a:(_ for _ in ()).throw(KeyboardInterrupt('received before result commit')))
    task=PublicTask('q','physical_q','question')
    with pytest.raises(KeyboardInterrupt): s.run_task(task,learn=False,attempt_id='single_solver')
    monkeypatch.setattr(adapter,'submit',original)
    s.checkpoint=TaskCheckpoint(s.checkpoint.root)
    result=s.run_task(task,learn=False,attempt_id='single_solver')
    assert result['score']['hard'] and len(sent)==1
    assert len(result['usage'])==1 and result['usage'][0]['total_tokens']==5
    assert len(s.checkpoint.state['decisions'])==1
    assert next(iter(s.checkpoint.state['decisions'].values()))['status']=='applied'
    s.close()


def test_t15_protocol_rejection_commits_feedback_with_owner_state(tmp_path,monkeypatch):
    s=office(tmp_path);cp=TaskCheckpoint(tmp_path/'checkpoint');s.checkpoint=s.executor.checkpoint=cp
    sent=transport(monkeypatch,[('runtime_step',{'action':'invented'}),('runtime_step',{'action':'invented'}),
                               ('runtime_step',{'action':'finish','answer':'private'})])
    original=cp.advance
    def interrupt(stage,**values):
        original(stage,**values)
        if values.get('executor_state',{}).get('owner_version')==1:
            raise KeyboardInterrupt('after atomic protocol rejection')
    monkeypatch.setattr(cp,'advance',interrupt)
    with pytest.raises(KeyboardInterrupt):
        s.executor.run(s.adapter.task,s.adapter,Broker(s.adapter,24,context=s.task_context),dynamic(s.adapter.task))
    cp=TaskCheckpoint(cp.root);s.checkpoint=s.executor.checkpoint=cp
    assert next(iter(cp.state['decisions'].values()))['status']=='rejected'
    assert cp.state['executor_state']['history'][-1]['error']=='runtime_protocol_error'
    result=s.executor.run(s.adapter.task,s.adapter,Broker(s.adapter,24,context=TaskContext()),dynamic(s.adapter.task))
    assert result['prediction']=='private' and len(sent)==3
    assert len({r['logical_decision_id'] for r in s.requests[:2]})==1 and s.requests[2]['logical_decision_id']!=s.requests[0]['logical_decision_id']
    assert sum(e.usage.total_tokens for e in s.usage.events)==15
    s.close()


@pytest.mark.parametrize('preparation',[False,True])
def test_t05_t06_planned_routes_and_preparation_continue_automatically(tmp_path,monkeypatch,worker,preparation):
    s=office(tmp_path);p=answer_program(s,worker,object_schema({'resource_id':{'type':'string'}},['resource_id']) if preparation else None)
    sk=s.bank.put('skill',skill('public answer',inputs=p['input_schema'],outputs=p['output_schema']))
    s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    node={'id':'answer','execution_mode':'skill','skill_id':sk['id'],
          'args':{'resource_id':{'unresolved':'Find a public resource'}} if preparation else {}}
    plan=workflow([node],{'answer':{'from':'answer','field':'answer'}})
    responses=[('submit_plan',{'mode':'compose','workflow':plan})]
    if preparation:
        prep=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'path':'a.txt'}}",
                                  outputs=object_schema({'path':{'type':'string'}},['path'])))
        responses.append(('runtime_step',{'action':'call_program','name':prep['id'],'arguments':{},'output_mapping':{'path':'resource_id'}}))
    else:
        first=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'answer':'unready'}}",
            inputs=object_schema({'missing':{'type':'string'}},['missing']),outputs=p['output_schema']),cost=0)
        s.bank.put('implementation',{'skill_id':sk['id'],'program_id':first['id']})
        assert s.bank.routes(node)[0]['id']==first['id']
    sent=transport(monkeypatch,responses)
    result,_=execute(s,plan)
    assert result['prediction']=='private' and len(sent)==(2 if preparation else 1)
    assert [a['program_id'] for a in result['attempts']]==([prep['id'],p['id']] if preparation else [p['id']])
    if preparation: assert result['attempts'][0]['outputs_consumed']
    s.close()


def test_t10_t11_completed_handoff_adds_view_without_rewriting_actual_result(tmp_path,monkeypatch,worker):
    s=office(tmp_path);s.executor.worker=worker
    first=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'actual':'a.txt'}}",
                               outputs=object_schema({'actual':{'type':'string'}},['actual'])))
    second=answer_program(s,worker,object_schema({'value':{'type':'string'},'resource_id':{'type':'string'}},['value','resource_id']))
    s.bank.put('skill',skill('public task'))
    plan=workflow([{'id':'produce','execution_mode':'program','program_id':first['id'],'args':{}},
        {'id':'consume','execution_mode':'program','program_id':second['id'],'args':{
            'value':{'from':'produce','field':'actual'},'resource_id':{'unresolved':'location'}}}],
        {'answer':{'from':'consume','field':'answer'}})
    sent=transport(monkeypatch,[('submit_plan',{'mode':'compose','workflow':plan}),
        ('runtime_step',{'action':'patch_node','patch':{'target_node':'produce','kind':'args','reason':'illegal past edit','args':{'x':{'literal':'bad'}}}}),
        ('runtime_step',{'action':'patch_node','patch':{'target_node':'produce','kind':'handoff','reason':'alias actual output',
            'output_aliases':{'actual':'item'},'consumer_refs':{'consume':{'value':{'from':'produce','field':'item'}}}}}),
        ('runtime_step',{'action':'patch_node','patch':{'target_node':'consume','kind':'args','reason':'actual location',
                                                     'args':{'resource_id':{'literal':'a.txt'}}}})])
    result,_=execute(s,plan)
    assert result['prediction']=='private' and len(sent)==4 and result['plan_revisions']==0
    original,view=result['values'][:2]
    assert original['outputs']==original['handoff_view']=={'actual':'a.txt'}
    assert view['origin']=='handoff' and s.task_context.results[view['view_result_id']]['outputs']=={'actual':'a.txt','item':'a.txt'}
    assert result['attempts'][0]['outputs_consumed'] and result['plan']['nodes'][0]['args']=={}
    assert next(h for h in result['history'] if h.get('program')==first['id'])['arguments']=={}
    s.close()


def test_t25_frozen_execution_failure_and_local_detach_preserve_qualification(tmp_path,monkeypatch,worker):
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.experiments.formal_log import tree_identity
    s=office(tmp_path);s.executor.worker=worker
    p=usable(s.bank,program("def run(ctx, inputs):\n    raise ValueError('ordinary execution failure')"))
    sk=s.bank.put('skill',skill('public task'));s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    frozen=tmp_path/'frozen';s.bank.freeze(frozen);s.bank.close()
    s.bank=s.executor.bank=s.planner.bank=Bank(frozen,readonly=True);s.executor.frozen=True
    node={'id':'bound','execution_mode':'program','program_id':p['id'],'args':{}}
    before=tree_identity(frozen);order=s.bank.routes(node)
    sent=transport(monkeypatch,[('submit_plan',{'mode':'compose','workflow':workflow([node])}),
        ('runtime_step',{'action':'patch_node','patch':{'target_node':'bound','kind':'detach','reason':'failed here','dynamic_goal':'Finish using public information'}}),
        ('runtime_step',{'action':'finish','answer':'private'})])
    result,_=execute(s,workflow([node]))
    assert len(result['attempts'])==1 and result['attempts'][0]['outcome']=='execution_failure'
    assert result['prediction']=='private' and len(sent)==3 and result['plan_revisions']==0
    assert tree_identity(frozen)==before and s.bank.routes(node)==order and s.bank.get(p['id'])['state']=='usable'
    s.close()


def test_t14_isolated_trial_replan_owns_trial_checkpoint_not_parent(tmp_path,monkeypatch,worker):
    from test_cf2_realization import trial_factory
    s=office(tmp_path);s.worker=worker;trial_factory(s)
    parent=TaskCheckpoint(tmp_path/'checkpoint');parent.advance('task_started',executor_state={'owner_version':99})
    s.checkpoint=s.planner.checkpoint=parent
    s.bank.put('skill',skill('find target'))
    asset=program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'value':'public'}}",
                  outputs=object_schema({'value':{'type':'string'}},['value']))
    asset['allowed_tools']=[];p=s.bank.put('program',asset)
    invalid={'action':'call_tool','name':'read','arguments':{'path':'missing','offset':0}}
    plan=workflow([{'id':'recover','execution_mode':'dynamic','goal':'Answer from acquired public information','args':{}}])
    sent=transport(monkeypatch,[('runtime_step',invalid),('runtime_step',invalid),
        ('submit_plan',{'mode':'compose','workflow':plan}),('runtime_step',{'action':'finish','answer':'private'})])
    record=s.test_program(p,{},s.adapter.task,trial_id='trial_scope')
    trial=TaskCheckpoint(parent.root/'trials/trial_scope')
    trial=TaskCheckpoint(trial.root/'executions'/trial.state['execution_id'])
    planner=next(d for d in trial.state['decisions'].values() if d['purpose']=='planner')
    assert planner['owner_state_version']==2 and 'trial_scope' in planner['scope']
    assert s.checkpoint is parent and s.planner.checkpoint is parent and not parent.state['decisions']
    assert parent.state['executor_state']['owner_version']==99 and len(sent)==4
    assert record['outcome']=='normal' and record['calls']==2
    s.close()
