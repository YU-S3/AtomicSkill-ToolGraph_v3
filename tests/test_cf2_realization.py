"""CF2 capability selection, durable Builder recovery and phase isolation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, digest, object_schema
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.planner import Planner, dynamic
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import AnswerAdapter, OfficeAdapter, SpreadsheetAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_empirical import config_for, program, worker
from test_cf2_contracts import http, office, response, run_steps


def usable(bank,asset,cost=1):
    p=bank.put('program',asset)
    for index in range(2):
        bank.record({'id':p['id']+str(index),'program_id':p['id'],'task_key':'fixture'+str(index),
            'origin':'train_test','outcome':'positive','basis':'local_check','calls':cost})
    return bank.get(p['id'])


def skill(goal='read target',inputs=None,outputs=None,**extra):
    return {'goal':goal,'guidance':'Use current public identities and character offsets.',
            'input_schema':inputs or object_schema(),'output_schema':outputs or object_schema(),**extra}


def requested_proposal(goal='prepare',**extra):
    return {'decision':'propose_skill_and_program_spec','skill':skill(goal),
            'realization_request':{'skill_id':'$new','action':'build','case_bindings':generated()['trial_inputs']},**extra}


def generated(source=None):
    return {'source':source or "def run(ctx, inputs):\n    return {'status':'needs_input','outputs':{}}",
            'trial_inputs':[{'case_id':'office-physical','inputs':{},'start_mode':'reset','prefix':[]}]}


def learning_trace(): return {'tools':[],'score':{'hard':False},'execution':{'prediction':None,'reason':'completed'}}


def trial_factory(s):
    s.adapter_factory=lambda: OfficeAdapter({'office':{'answer':'private'}},s.config)


def test_t26_current_guidance_and_program_goal_reach_http(tmp_path,monkeypatch,worker):
    seen=http(monkeypatch,[response([{'action':'finish','answer':'done'}])])
    s=office(tmp_path); s.worker=worker
    sk=s.bank.put('skill',skill(inputs=object_schema({'query':{'type':'string'}},['query'])))
    p=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'needs_input','outputs':{}}",inputs=sk['input_schema']))
    s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    plan={'nodes':[{'id':'current','execution_mode':'skill','skill_id':sk['id'],'args':{}}]}
    s.executor.run(s.adapter.task,s.adapter,Broker(s.adapter,24,context=s.task_context),plan)
    material=json.loads(seen[0]['messages'][1]['content'])
    assert material['memory']['guidance'][0]['guidance']==sk['guidance']
    assert material['calls']['programs'][0]['capabilities'][0]['goal']==sk['goal']
    assert 'source' not in material['calls']['programs'][0] and 'def run' not in json.dumps(material)
    s.close()


def test_t27_ninth_relevant_program_enters_menu_and_frozen_filters(tmp_path):
    bank=Bank(tmp_path/'bank')
    programs=[usable(bank,program("def run(ctx, inputs):\n    # variant "+str(i)+"\n    return {'status':'ok','outputs':{}}")) for i in range(9)]
    ninth=sorted(programs,key=lambda p:p['id'])[-1]
    sk=bank.put('skill',skill('target document search'))
    bank.put('implementation',{'skill_id':sk['id'],'program_id':ninth['id']})
    assert bank.program_options('target document search')[0]['id']==ninth['id']
    candidate=bank.put('program',program("def run(ctx, inputs):\n    return {'status':'blocked','outputs':{}}"))
    bank.freeze(tmp_path/'frozen'); frozen=Bank(tmp_path/'frozen',readonly=True)
    assert candidate['id'] not in {p['id'] for p in frozen.program_options('target',allow_candidate=True)}
    frozen.close(); bank.close()


def test_t28_second_authorized_route_ready_runs_without_runtime(tmp_path,worker):
    s=office(tmp_path); sk=s.bank.put('skill',skill('prepare answer',outputs=object_schema({'answer':{'type':'string'}},['answer'])))
    first=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'answer':'first'}}",inputs=object_schema({'missing':{'type':'string'}},['missing']),outputs=sk['output_schema']),cost=1)
    second=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'answer':'second'}}",outputs=sk['output_schema']),cost=2)
    for p in [first,second]: s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    node={'id':'answer','execution_mode':'skill','skill_id':sk['id'],'args':{}}
    assert [p['id'] for p in s.bank.routes(node)]==[first['id'],second['id']]
    executor=Executor(s.bank,lambda *a,**k: pytest.fail('ready route must bypass Runtime'),worker,object())
    result=executor.run(s.adapter.task,s.adapter,Broker(s.adapter,24),{'nodes':[node],'outputs':{'answer':{'from':'answer','field':'answer'}}})
    assert result['prediction']=='second' and len(result['attempts'])==1 and result['attempts'][0]['program_id']==second['id']
    s.close()


def test_t29_explicit_output_mapping_automatically_continues(tmp_path,worker):
    s=office(tmp_path); s.worker=worker
    sk=s.bank.put('skill',skill('consume object',inputs=object_schema({'object':{'type':'string'}},['object']),outputs=object_schema({'answer':{'type':'string'}},['answer'])))
    target=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'answer':inputs['object']}}",inputs=sk['input_schema'],outputs=sk['output_schema']))
    prep=usable(s.bank,program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'found_object':'public-object'}}",outputs=object_schema({'found_object':{'type':'string'}},['found_object'])))
    s.bank.put('implementation',{'skill_id':sk['id'],'program_id':target['id']})
    plan={'nodes':[{'id':'use','execution_mode':'skill','skill_id':sk['id'],'args':{'object':{'unresolved':'obtain'}}}],
          'outputs':{'answer':{'from':'use','field':'answer'}}}
    result,calls=run_steps(s,[{'action':'call_program','name':prep['id'],'arguments':{},'output_mapping':{'found_object':'object'}}],plan)
    assert result['prediction']=='public-object' and len(calls)==1 and len(result['attempts'])==2
    assert result['attempts'][0]['outputs_consumed']
    s.close()


def test_t30_revision_only_cannot_reset_rejected_call_limit(tmp_path):
    s=office(tmp_path); count=0
    original=s.adapter.observe
    def observe(): return {**original(),'revision':count}
    def call(name,args):
        nonlocal count
        count+=1
        return {'accepted':False,'observation':'same refusal','data':{},'error':'invalid','done':False}
    s.adapter.observe=observe; s.adapter.call=call
    agent=lambda *a,**k:{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':0}}
    class Replanner:
        def plan(self,*a,**k): return dynamic(s.adapter.task)
    result=Executor(s.bank,agent,s.worker,Replanner()).run(s.adapter.task,s.adapter,Broker(s.adapter,24),dynamic(s.adapter.task))
    assert count==2 and result['reason']=='repeated_unchanged_failure'
    s.close()


def test_t31_reuse_existing_can_build_a_missing_program(tmp_path,monkeypatch):
    s=office(tmp_path); trial_factory(s)
    sk=s.bank.put('skill',skill('prepare'))
    seen=http(monkeypatch,[response([{'decision':'reuse_existing','existing_skill_id':sk['id'],'generate_program':True,
        'realization_request':{'skill_id':sk['id'],'action':'build','case_bindings':generated()['trial_inputs']}}],name='submit_learning'),
                           response([generated()],name='submit_program')])
    log=s.learner.learn(s.adapter.task,learning_trace())
    assert log['program'] and len(seen)==2 and len(s.bank.jobs())==1
    s.close()


def test_t32_missing_light_binding_waits_without_builder(tmp_path,monkeypatch):
    s=office(tmp_path)
    sk=skill('find light',inputs=object_schema({'light_source':{'type':'string'}},['light_source']))
    proposal={'decision':'propose_skill_and_program_spec','skill':sk,'realization_request':{'skill_id':'$new','action':'build',
        'case_bindings':[{'case_id':'office-physical','inputs':{},'start_mode':'reset','prefix':[]}]}}
    deferred=deepcopy(proposal); deferred['realization_request'].update(action='defer',case_bindings=[])
    seen=http(monkeypatch,[response([proposal],name='submit_learning'),response([deferred],name='submit_learning')])
    log=s.learner.learn(s.adapter.task,learning_trace())
    assert len(seen)==2 and log['program'] is None and 'trial_binding_invalid' in json.dumps(seen[1]['messages'])
    assert s.bank.jobs()[0]['state']=='waiting_example'
    s.close()


def test_t33_new_case_fills_unused_trial_slot_and_duplicates_do_not(tmp_path,monkeypatch):
    s=office(tmp_path); trial_factory(s)
    sk=s.bank.put('skill',skill('find target'))
    p=s.bank.put('program',program("def run(ctx, inputs):\n    return {'status':'needs_input','outputs':{}}"))
    s.bank.put('implementation',{'skill_id':sk['id'],'program_id':p['id']})
    first={'case_id':'office-physical','inputs':{},'start_mode':'reset','prefix':[]}
    s.bank.save_job({'id':digest(['realization',sk['id']]),'skill_id':sk['id'],'skill_version':sk['id'],'kind':'trial','state':'ready','program_id':p['id'],
        'case_bindings':[first],'repair_used':False,'generation_count':1,'epoch':0,'trigger_task':'office-physical'})
    seen=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_learning')])
    s.learner.learn(s.adapter.task,learning_trace())
    assert len(s.bank.attempts(p['id']))==1
    next_task=PublicTask('office','new-physical','find target')
    request={'decision':'reuse_existing','existing_skill_id':sk['id'],'realization_request':{'skill_id':sk['id'],'action':'trial',
        'case_bindings':[{'case_id':'new-physical','inputs':{},'start_mode':'reset','prefix':[]}]}}
    http(monkeypatch,[response([request],name='submit_learning'),response([request],name='submit_learning')])
    s.learner.learn(next_task,learning_trace()); s.learner.learn(next_task,learning_trace())
    assert len(s.bank.attempts(p['id']))==2 and s.bank.get(p['id'])['state']=='candidate'
    s.close()


def test_t34_qa_guidance_only_uses_one_normal_solver(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response(content='public answer',finish='stop'),response([{'decision':'upsert_guidance',
        'guidance_skill':{'goal':'QA guidance', 'guidance':'Use public context.'}}],name='submit_learning')])
    config=config_for(tmp_path/'bank'); a=AnswerAdapter('searchqa',{'qa':{'answers':['public answer']}})
    s=EmpiricalSystem(config,harness=a)
    trace=s.run_task(PublicTask('qa','qa-physical','public question',{'context':'original public context'*1000,'options':['first','second']}))
    assert trace['score']['hard'] and [r['stage'] for r in s.requests]==['runtime','extractor']
    assert not s.bank.jobs() and not s.bank.all('program') and len(seen)==2
    material=json.loads(seen[0]['messages'][1]['content'])
    assert material['inputs']['context']=='original public context'*1000 and material['inputs']['options']==['first','second']
    s.close()


@pytest.mark.parametrize('malformed',[False,True])
def test_t35_both_length_paths_share_one_65536_recovery(tmp_path,monkeypatch,malformed):
    first=response(finish='length')
    if malformed:
        first=response([{'placeholder':0}],name='submit_program',finish='length')
        first['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{"source":'
    seen=http(monkeypatch,[response([requested_proposal()],name='submit_learning'),first,
                           response([generated()],name='submit_program')])
    s=office(tmp_path); trial_factory(s)
    s.learner.learn(s.adapter.task,learning_trace())
    assert [p['max_tokens'] for p in seen[1:]]==[32768,65536]
    abi=[json.loads(p['messages'][1]['content'])['future_program_api']['public_program_abi'] for p in seen[1:]]
    assert abi[0]==abi[1] and abi[0]['current_tool_surface']=='named_tools'
    assert abi[0]['methods']['call']=='(name, arguments)' and abi[0]['program_result_schema']['required']==['status','outputs']
    job=s.bank.jobs()[0]
    assert job['repair_used'] and job['generation_count']==2
    assert s.provider('runtime').config.max_completion_tokens==32768
    assert sum(u.to_dict()['total_tokens'] for u in s.usage.events)==15
    s.close()


def test_t36_recovery_then_syntax_failure_cannot_generate_third(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([requested_proposal()],name='submit_learning'),response(finish='length'),
                           response([generated('def run(')],name='submit_program')])
    s=office(tmp_path)
    log=s.learner.learn(s.adapter.task,learning_trace())
    assert len(seen)==3 and log['program'] is None and s.bank.jobs()[0]['state']=='deferred'
    s.close()


def test_t37_valid_length_submission_does_not_regenerate(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([requested_proposal()],name='submit_learning'),
                           response([generated()],name='submit_program',finish='length')])
    s=office(tmp_path); trial_factory(s)
    s.learner.learn(s.adapter.task,learning_trace())
    assert len(seen)==2 and not s.bank.jobs()[0]['repair_used']
    s.close()


def test_t38_saved_recovery_survives_crash_and_runtime_cap_is_unchanged(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([requested_proposal()],name='submit_learning'),response(finish='length'),
                           response([generated()],name='submit_program'),response(content='ordinary runtime',finish='stop')])
    s=office(tmp_path); trial_factory(s)
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint'); s.audit_path=tmp_path/'requests.json'
    original=s.checkpoint.advance
    def interrupt(stage,**kwargs):
        original(stage,**kwargs)
        if any(key.startswith('builder_') and not key.endswith(('_request_material','_semantics')) for key in kwargs):
            raise KeyboardInterrupt('after saved recovery')
    monkeypatch.setattr(s.checkpoint,'advance',interrupt)
    with pytest.raises(KeyboardInterrupt): s.learner.learn(s.adapter.task,learning_trace())
    assert s.bank.jobs()[0]['repair_used']
    monkeypatch.setattr(s.checkpoint,'advance',original)
    s.learner.learn(s.adapter.task,learning_trace())
    s.agent('runtime','Answer once',{'goal':'public'},None,None)
    assert len(seen)==4 and seen[-1]['max_tokens']==32768 and s.bank.jobs()[0]['generation_count']==2
    assert sum(u['total_tokens'] for u in json.loads(Path(s.audit_path).read_text())['usage'])==20
    s.close()


@pytest.mark.parametrize('values', [[2, 7], [2, 7, 11]])
def test_t39_file_trial_scores_produced_bundle_without_dynamic_solver(tmp_path,monkeypatch,values):
    import openpyxl
    if not __import__('os').environ.get('PROGRAM_IMAGE_DIGEST'): pytest.skip('requires real Docker')
    cases=[]
    for index,value in enumerate(values):
        input_path=tmp_path/f'{index}_input.xlsx'; gold=tmp_path/f'{index}_gold.xlsx'
        wb=openpyxl.Workbook(); wb.active['A1']=value; wb.save(input_path); wb.active['B1']=value*2; wb.save(gold); wb.close()
        cases.append({'input':str(input_path),'gold':str(gold)})
    records={'sheet':{'public_files':{'input.xlsx':cases[0]['input']},'cases':cases,
        'instruction_type':'Cell-Level Manipulation','answer_position':'Sheet!B1'}}
    config=config_for(tmp_path/'bank'); config['program_worker'].update(wall_timeout_seconds=120,memory_limit_mb=2048)
    a=SpreadsheetAdapter(records,config); task=PublicTask('sheet','sheet-physical','double A1 into B1')
    a.reset(task); s=EmpiricalSystem(config,harness=a,adapter_factory=lambda:SpreadsheetAdapter(records,config),provider=object())
    solution="import openpyxl\nw=openpyxl.load_workbook(INPUT_PATH)\nw.active['B1']=w.active['A1'].value*2\nw.save(OUTPUT_PATH)\nw.close()\n"
    source='def run(ctx, inputs):\n    from pathlib import Path\n    Path("solution.py").write_text('+repr(solution)+')\n    exec('+repr(solution)+', {"INPUT_PATH":"/workspace/inputs/input.xlsx","OUTPUT_PATH":"/workspace/case1_result.xlsx"})\n    return {"status":"ok","outputs":{"files":["solution.py","case1_result.xlsx"]}}'
    p=program(source,outputs=object_schema({'files':{'type':'array','items':{'type':'string'}}},['files']))
    p.update(allowed_tools=[],result_role='final_files'); p=s.bank.put('program',p)
    s.agent=lambda *a,**k: pytest.fail('final file trial must not solve a second time')
    trial=s.test_program(p,{},task,trial_id='actual-file-trial')
    assert trial['outcome']=='positive' and trial['result']['score']['case_results']==[True]*len(values)
    assert trial['result']['submission']=='direct_program_output'
    s.close()


def test_t40_episodes_seeds_frozen_and_real_image_http_are_isolated(tmp_path,monkeypatch):
    original=TaskContext(); rid=original.register('event',{'data':'only this episode'})
    with pytest.raises(ValueError): TaskContext().read(rid)
    for seed in [42,43,44]:
        bank=Bank(tmp_path/str(seed),seed=seed); bank.freeze(tmp_path/(str(seed)+'_frozen'))
        frozen=Bank(tmp_path/(str(seed)+'_frozen'),readonly=True,seed=seed)
        with pytest.raises(RuntimeError): frozen.save_job({'id':'bad'})
        assert frozen.digest()==json.loads((frozen.root/'freeze.json').read_text())['digest']
        bank.close(); frozen.close()
    image=tmp_path/'public.png'; image.write_bytes(b'\x89PNG\r\n\x1a\npublic-fixture-pixels')
    seen=http(monkeypatch,[response(content='word',finish='stop')])
    config=config_for(tmp_path/'visual-bank'); config['llm']['input_modalities']=['text','image']
    a=AnswerAdapter('docvqa',{'doc':{'answers':['word']}}); s=EmpiricalSystem(config,harness=a)
    trace=s.run_task(PublicTask('doc','doc-physical','Original image question',{'images':[str(image)]},split='val'),learn=False)
    content=seen[0]['messages'][1]['content']
    assert trace['score']['hard'] and any(p['type']=='image_url' and p['image_url']['url'].startswith('data:image/png;base64,') for p in content)
    assert 'Original image question' in content[0]['text'] and len(seen)==1
    authority=Path(__file__).resolve().parents[1]/'data/main_experiment_v1/manifest.json'
    assert hashlib.sha256(authority.read_bytes()).hexdigest()=='b85e0c2e442ca6b97ef98ece895719d4231f18db29682b0fcb6fdf3f91c37c95'
    s.close()



def test_t30_real_workspace_progress_allows_a_new_attempt(tmp_path):
    s=office(tmp_path); stage=s.adapter.workspace.stage(); (stage/'public.txt').write_text('one')
    s.adapter.workspace.publish(stage,[]); before=s.adapter.progress_key()
    stage=s.adapter.workspace.stage(); (stage/'public.txt').write_text('two')
    s.adapter.workspace.publish(stage,[])
    assert s.adapter.progress_key()!=before
    s.close()


def test_t39_empty_final_files_cannot_claim_an_old_bundle(tmp_path,worker):
    config=config_for(tmp_path/'bank')
    records={'sheet':{'public_files':{}}}
    def factory():
        adapter=SpreadsheetAdapter(records,config)
        original=adapter.reset
        def reset(task):
            result=original(task); stage=adapter.workspace.stage()
            (stage/'solution.py').write_text('old solution'); (stage/'case1_result.xlsx').write_bytes(b'old result')
            adapter.workspace.publish(stage,['solution.py','case1_result.xlsx'])
            return result
        adapter.reset=reset
        return adapter
    adapter=factory(); task=PublicTask('sheet','physical','files'); adapter.reset(task)
    s=EmpiricalSystem(config,harness=adapter,adapter_factory=factory,provider=object())
    source="def run(ctx, inputs):\n    return {'status':'ok','outputs':{'files':[]}}"
    asset=program(source,outputs=object_schema({'files':{'type':'array','items':{'type':'string'}}},['files']))
    asset.update(allowed_tools=[],result_role='final_files'); asset=s.bank.put('program',asset)
    s.agent=lambda *a,**k: pytest.fail('incomplete final bundle must not trigger another solve')
    trial=s.test_program(asset,{},task,trial_id='empty-output-trial')
    assert trial['outcome']=='execution_failure' and 'score' not in trial['result']
    assert trial['result']['error_code']=='program_publication_incomplete'
    assert s.bank.get(asset['id'])['state']=='candidate'
    s.close()



@pytest.mark.parametrize('explicit_intent',[True,False])
def test_t34_guidance_only_cannot_be_built_by_conflicting_trial_request(tmp_path,monkeypatch,explicit_intent):
    asset=skill('QA guidance',execution_intent='guidance_only') if explicit_intent else skill('QA guidance')
    proposal={'decision':'propose_skill_and_program_spec','skill':asset,'generate_program':False,
              'realization_request':{'skill_id':'$new','action':'trial','case_bindings':[]}}
    seen=http(monkeypatch,[response([proposal],name='submit_learning')] * 2)
    a=AnswerAdapter('searchqa',{'qa':{'answers':['answer']}}); a.reset(PublicTask('qa','qa-physical','question'))
    s=EmpiricalSystem(config_for(tmp_path/'bank'),harness=a)
    log=s.learner.learn(a.task,learning_trace())
    assert len(seen)==2 and not s.bank.jobs() and log['program'] is None
    assert log['decision']=='rejected' and s.bank.all('skill')==[]
    s.close()



def test_t40_stopped_train_smoke_does_not_start_val_or_another_benchmark(tmp_path,monkeypatch):
    from atomic_skillgraph.experiments import run_multibench
    root=tmp_path/'datasets'; (root/'searchqa').mkdir(parents=True)
    task=PublicTask('qa','physical','question')
    from dataclasses import asdict
    (root/'searchqa/train.json').write_text(json.dumps({'tasks':[asdict(task)]}))
    monkeypatch.setattr(run_multibench,'create_simple_harness',lambda *a:object())
    calls=[]
    def stopped(*a,**k):
        calls.append(a)
        return {'tasks':1,'successes':1,'total_tokens':5,'knowledge_digest':'unchanged','complete':False}
    monkeypatch.setattr(run_multibench,'run',stopped)
    report=run_multibench.run_smoke(config_for(tmp_path/'bank'),root,tmp_path/'out',['searchqa','livemath'])
    assert len(calls)==1 and report['benchmarks']['searchqa']['status']=='stopped'
    assert 'val' not in report['benchmarks']['searchqa']['runs'] and 'livemath' not in report['benchmarks']



def test_t26_planner_uses_an_object_root_at_actual_http_boundary(tmp_path,monkeypatch):
    s=office(tmp_path); s.bank.put('skill',skill('find target'))
    plan={'mode':'compose','workflow':dynamic(s.adapter.task)}
    seen=http(monkeypatch,[response([plan],name='submit_plan')])
    assert s.planner.plan(s.adapter.task,s.adapter)==plan['workflow']
    schema=seen[0]['tools'][0]['function']['parameters']
    assert schema['type']=='object' and schema['properties']['mode']['enum']==['select','compose']
    s.close()
