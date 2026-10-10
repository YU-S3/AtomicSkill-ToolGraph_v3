"""Unified qualification and answering checks, with intercepted HTTP and real Worker."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.bank_view import BankView
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema, digest
from atomic_skillgraph.empirical.local_validation import evidence_from_trace, validate_on_source
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_empirical import config_for, worker
from test_cf2_contracts import http, response


SCHEMA = object_schema({'n':{'type':'integer'}},['n'])
OUT = object_schema({'value':{'type':'integer'}},['value'])
SOURCE = "def run(ctx, inputs):\n    return {'status':'ok','outputs':{'value':inputs['n']*2}}"


def write_fixture(name,value):
    if os.environ.get('ATOMIC_UNIFIED_EVIDENCE_DIR'):
        root=Path(os.environ['ATOMIC_UNIFIED_EVIDENCE_DIR']);root.mkdir(parents=True,exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps({'kind':'zero-model engineering fixture',**value},ensure_ascii=False,indent=2))


def setup(tmp_path):
    task = PublicTask('source','source-physical','Double the given integer',{'n':3})
    config = config_for(tmp_path/'bank')
    config['experiment']['output_dir'] = str(tmp_path/'train')
    config['harness'] = {'adapter':'searchqa'}
    records = {task.task_id:{'answers':['6']}}
    adapter = AnswerAdapter('searchqa',records,config)
    system = EmpiricalSystem(config,harness=adapter,adapter_factory=lambda:AnswerAdapter('searchqa',records,config))
    return system,task


def candidate(system,source=SOURCE):
    return system.bank.put('program',{'source':source,'entry':'run','input_schema':SCHEMA,'output_schema':OUT,
        'allowed_tools':[],'environment':system.config['program_environment'],'result_role':'intermediate'})


def test_single_source_real_worker_freeze_and_consumption(tmp_path,monkeypatch,worker):
    s,t = setup(tmp_path)
    s.worker = worker
    seen = http(monkeypatch,[response([{'action':'execute_python','source':SOURCE,'arguments':{'n':3},
        'input_schema':SCHEMA,'output_schema':OUT}],name='answer_step'),
        response([{'action':'finish','answer':'6','used_results':[]}],name='answer_step')])
    try:
        trace = s.run_task(t,learn=False)
        assert trace['score']['hard'] and len(seen)==2 and not trace.get('initial_plan')
        assert trace['execution']['temporary_executions'][0]['status']=='ok'
        evidence = evidence_from_trace(t,trace,s.config['program_environment'])
        p = candidate(s)
        assert s.bank.program_options(t.goal)==[]
        binding = {'case_id':t.physical_key,'inputs':{'n':3},'start_mode':'reset','prefix':[],
            'local_evidence_ref':evidence[0]['id'],'reference_fields':{'value':['value']}}
        record = validate_on_source(s,p,binding,t,{'local_evidence':evidence},'trial')
        assert record['validation']['passed'] and record['validation']['variation']
        p = s.bank.get(p['id'])
        assert s.bank.program_eligible(p) and len(s.bank.attempts(p['id']))==1
        changed = deepcopy(p); changed['source']+='\n# new version'
        assert not s.bank.program_eligible(changed)
        frozen = tmp_path/'frozen'; s.bank.freeze(frozen)
        readonly = Bank(frozen,readonly=True)
        before = readonly.digest()
        for mode in ('learned_assets_off','guidance_only'):
            view=BankView(readonly,mode)
            assert view.get(p['id']) is None and not view.program_eligible(p)
            assert view.program_options(t.goal)==[]
        val = PublicTask('val','other-physical',t.goal,{'n':4},'val')
        cfg=deepcopy(s.config);cfg['data_dir']=str(frozen);cfg['experiment'].update(runtime_mode='frozen',output_dir=str(tmp_path/'val'))
        records={val.task_id:{'answers':['8']}}
        full=EmpiricalSystem(cfg,harness=AnswerAdapter('searchqa',records,cfg),readonly=True)
        full.worker=worker
        # The second fixed response binds the actual result ID returned by production.
        def post(*a,**kw):
            material=json.loads(kw['json']['messages'][-1]['content'].split('POLICY_CONTEXT_JSON\n')[-1])
            local=material.get('local_results',[])
            step={'action':'finish','answer':'8','used_results':[local[0]['result_id']]} if local else {
                'action':'call_program','name':p['id'],'arguments':{'n':4}}
            return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([step],name='answer_step'))
        monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
        try:
            result=full.run_task(val,learn=False)
            assert result['score']['hard'] and result['execution']['attempts'][0]['outputs_consumed']
            assert readonly.digest()==before==full.bank.digest()
            write_fixture('source_freeze_non_source_consumption',{'source_trace':trace,'local_validation':record,
                'frozen_sha256':before,'val_trace':result})
        finally:full.close();readonly.close()
    finally:s.close()


def test_docvqa_actual_pixels_learner_payload_and_readonly_worker(tmp_path,monkeypatch,worker):
    from PIL import Image
    path=tmp_path/'public.png';Image.new('RGB',(17,9),(12,34,56)).save(path)
    task=PublicTask('doc','doc-physical','What is shown?',{'images':[str(path)]})
    cfg=config_for(tmp_path/'bank');cfg['llm']['input_modalities']=['text','image']
    cfg['experiment']['output_dir']=str(tmp_path/'out')
    adapter=AnswerAdapter('docvqa',{'doc':{'answers':['image']}},cfg)
    s=EmpiricalSystem(cfg,harness=adapter);s.worker=worker
    seen=http(monkeypatch,[response(content='image',finish='stop'),response([{'decision':'no_change'}],name='submit_learning'),
        response([{'source':SOURCE,'trial_inputs':[]}],name='submit_program')])
    try:
        trace=s.run_task(task)
        assert trace['score']['hard'] and len(seen)==2
        for p in seen:
            content=p['messages'][1]['content']
            assert isinstance(content,list) and content[1]['image_url']['url'].startswith('data:image/png;base64,')
        source="def run(ctx, inputs):\n    from PIL import Image\n    return {'status':'ok','outputs':{'width':Image.open(inputs['path']).width}}"
        p={'id':'image-fixture','source':source,'entry':'run','input_schema':object_schema({'path':{'type':'string'}},['path']),
            'output_schema':object_schema({'width':{'type':'integer'}},['width']),'allowed_tools':[],
            'environment':s.config['program_environment'],'result_role':'intermediate'}
        result=worker.execute(p,{'path':adapter.worker_resources['image_0.png']},Broker(adapter,1))
        assert result['status']=='ok' and result['outputs']['width']==17
        from atomic_skillgraph.empirical.prompts import BUILD,BUILDER_PROMPT
        local={'public_task':{'inputs':task.inputs}}
        material=s.learner.builder_material({'goal':'read image width'},[],[{'local_evidence':local}],
            {'workspace_capabilities':{},'allowed_names':[]})
        s.agent('tool_builder',BUILDER_PROMPT,material,'submit_program',BUILD)
        assert len(seen)==3 and seen[-1]['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
        write_fixture('docvqa_pixels',{'payloads':seen,'worker':result,'actual_model_connected':False})
        cfg['data_dir']=str(tmp_path/'unsupported');cfg['llm']['input_modalities']=['text']
        with pytest.raises(ValueError,match='image'):EmpiricalSystem(cfg,harness=AnswerAdapter('docvqa',{},cfg))
    finally:s.close()


def test_constant_source_rejected_by_actual_parameter_variation(tmp_path,worker):
    s,t=setup(tmp_path);s.worker=worker;s.adapter.reset(t)
    try:
        operation={'id':'local:actual','kind':'executed_python','source_physical_key':t.physical_key,
            'source_trace_sha256':'fixture','prefix':[],'public_task':{'inputs':{'n':3}},
            'environment_identity':s.config['program_environment'],
            'operation':{'source':SOURCE,'inputs':{'n':3},'input_schema':SCHEMA,'output_schema':OUT},
            'reference':worker.execute(candidate(s),{'n':3},Broker(s.adapter,1))['outputs']}
        constant=candidate(s,"def run(ctx, inputs):\n    unused=inputs['n']\n    return {'status':'ok','outputs':{'value':6}}")
        binding={'inputs':{'n':3},'prefix':[],'local_evidence_ref':operation['id'],'reference_fields':{'value':['value']}}
        row=validate_on_source(s,constant,binding,t,{'local_evidence':[operation]},'constant')
        assert not row['validation']['passed'] and row['validation']['reason']=='parameterized_replay_mismatch'
        assert not s.bank.program_eligible(s.bank.get(constant['id']))
        with pytest.raises(ValueError,match='recorded host'):s.bank.record_validation(constant['id'],{'policy':'atomic.local-validation.v1','passed':True})
    finally:s.close()


def test_local_validation_uses_stable_public_permissions(tmp_path,worker):
    class FutureAdapter(AnswerAdapter):
        def tool_definitions(self):
            return [{'name':'future_action','input_schema':object_schema(),'output_schema':object_schema()}]
    s,t=setup(tmp_path);s.worker=worker
    records={t.task_id:{'answers':['6']}}
    s.adapter=FutureAdapter('searchqa',records,s.config)
    s.adapter_factory=lambda:FutureAdapter('searchqa',records,s.config)
    s.adapter.reset(t)
    try:
        original=worker.execute(candidate(s),{'n':3},Broker(s.adapter,1))
        e={'id':'source','kind':'executed_python','source_physical_key':t.physical_key,
            'source_trace_sha256':digest(original),'prefix':[],'public_task':{'inputs':t.inputs},
            'environment_identity':s.config['program_environment'],
            'operation':{'source':SOURCE,'inputs':{'n':3},'input_schema':SCHEMA,'output_schema':OUT},
            'reference':original['outputs']}
        binding={'inputs':{'n':3},'prefix':[],'local_evidence_ref':'source','reference_fields':{'value':['value']}}
        p=s.bank.put('program',{**{k:v for k,v in candidate(s).items() if k!='id'},'allowed_tools':['future_action']})
        result=validate_on_source(s,p,binding,t,{'local_evidence':[e]},'future-tools')
        assert result['validation']['passed'] and s.bank.program_eligible(s.bank.get(p['id']))
        assert not Broker(s.adapter,1).call('future_action',{})['accepted']
        p=s.bank.put('program',{**{k:v for k,v in p.items() if k!='id'},'allowed_tools':['private_action']})
        with pytest.raises(ValueError,match='public Adapter surface'):
            validate_on_source(s,p,binding,t,{'local_evidence':[e]},'private-tools')
    finally:s.close()


def test_local_validation_requires_actual_source_operation(tmp_path,worker):
    class NativeAdapter(AnswerAdapter):
        def tool_definitions(self):
            return [{'name':name,'input_schema':SCHEMA if name=='advance' else object_schema(),
                     'output_schema':object_schema({'accepted':{'type':'boolean'}},['accepted']),
                     'effect':effect} for name,effect in [('advance','stateful'),('peek','read_only')]]
        def available_tools(self):return self.tool_definitions()
        def call(self,name,arguments):return {'accepted':True,'done':False,'observation':'public native result'}
    s,t=setup(tmp_path);s.worker=worker
    records={t.task_id:{'answers':['6']}}
    s.adapter=NativeAdapter('searchqa',records,s.config)
    s.adapter_factory=lambda:NativeAdapter('searchqa',records,s.config)
    s.adapter.reset(t)
    try:
        broker=Broker(s.adapter,10);broker.call('advance',{'n':3})
        evidence=evidence_from_trace(t,{'tools':broker.events},s.config['program_environment'])
        binding={'inputs':{'n':3},'prefix':[],'local_evidence_ref':evidence[0]['id'],
                 'reference_fields':{'accepted':['accepted']}}
        sources=[
            ("ctx.call('advance',{'n':inputs['n']})",True),
            ("unused=inputs['n'];ctx.call('peek',{})",False),
            ("unused=inputs['n']",False),
            ("ctx.call('advance',{'n':inputs['n']});ctx.call('advance',{'n':4})",False)]
        for index,(action,passed) in enumerate(sources):
            p=s.bank.put('program',{'source':"def run(ctx, inputs):\n    "+action+
                "\n    return {'status':'ok','outputs':{'accepted':True}}",'entry':'run',
                'input_schema':SCHEMA,'output_schema':object_schema({'accepted':{'type':'boolean'}},['accepted']),
                'allowed_tools':['advance','peek'],'environment':s.config['program_environment']})
            row=validate_on_source(s,p,binding,t,{'local_evidence':evidence},'native:'+str(index))
            assert row['validation']['passed'] is passed
            assert s.bank.program_eligible(s.bank.get(p['id'])) is passed
            if not passed:assert row['validation']['reason']=='source_operation_mismatch'
            else:
                old=deepcopy(s.bank.get(p['id']));old['local_validation']['policy']='atomic.local-validation.v1'
                assert not s.bank.program_eligible(old)
    finally:s.close()


def test_literal_json_is_not_parameterized_local_evidence(tmp_path,worker):
    s,t=setup(tmp_path);s.worker=worker;s.adapter.reset(t)
    try:
        schema=object_schema({'text':{'type':'string'}},['text'])
        out=object_schema({'value':{'type':'string'}},['value'])
        source="def run(ctx, inputs):\n    return {'status':'ok','outputs':{'value':inputs['text'].upper()}}"
        p=s.bank.put('program',{'source':source,'entry':'run','input_schema':schema,'output_schema':out,
            'allowed_tools':[],'environment':s.config['program_environment']})
        reference=worker.execute(p,{'text':'x'},Broker(s.adapter,1))
        e={'id':'text-reference','kind':'executed_python','source_physical_key':t.physical_key,
            'source_trace_sha256':digest(reference),'prefix':[],'public_task':{'inputs':{'text':'x'}},
            'environment_identity':s.config['program_environment'],
            'operation':{'source':source,'inputs':{'text':'x'},'input_schema':schema,'output_schema':out},
            'reference':reference['outputs']}
        bad=s.bank.put('program',{**{k:v for k,v in p.items() if k!='id'},'source':
            "def run(ctx, inputs):\n    unused=inputs['text']\n    return {'status':'ok','outputs':{'value':'X'}}"})
        row=validate_on_source(s,bad,{'inputs':{'text':'x'},'prefix':[],'local_evidence_ref':e['id'],
            'reference_fields':{'value':['value']}},t,{'local_evidence':[e]},'literal-output')
        assert not row['validation']['passed'] and row['validation']['reason']=='constant_local_output'
    finally:s.close()


def test_parent_budget_is_shared_across_roles_retries_and_trial(tmp_path):
    gov=BudgetGovernor(tmp_path/'budget.json',token_limit=1000,finish_reserve=0,request_limit=8,validation_limit=2)
    ctx={'parent_scope':'one','parent_token_limit':130}
    for i,role in enumerate(('runtime','extractor','tool_builder')):
        gov.admit(str(i),{'max_tokens':10},{**ctx,'stage':role})
        gov.complete({'request_id':str(i),'raw_usage':{'total_tokens':40},'outcome':'success'})
    from atomic_skillgraph.core.errors import BudgetExhausted
    with pytest.raises(BudgetExhausted,match='Parent'):gov.admit('trial',{'max_tokens':10},{**ctx,'phase':'trial'})
    gov.admit_validation('one');gov.admit_validation('variation')
    with pytest.raises(BudgetExhausted):gov.admit_validation('extra')


def test_matrix_static_allocations_do_not_multiply_total_budget():
    from atomic_skillgraph.experiments.run_formal import allocate_campaign_budgets
    limits={'token_limit':101,'request_limit':17,'finish_reserve':8,'validation_limit':4,'train_task_tokens':50}
    rows=allocate_campaign_budgets(limits,['one','two','three'])
    assert rows==allocate_campaign_budgets(limits,['one','two','three'])
    for key in ('token_limit','request_limit','finish_reserve','validation_limit'):
        assert sum(r[key] for r in rows.values())==limits[key]
    assert all(r['train_task_tokens']==50 for r in rows.values())
    with pytest.raises(ValueError):allocate_campaign_budgets(None,['one'])
    with pytest.raises(ValueError):allocate_campaign_budgets(limits,['one','one'])
    with pytest.raises(ValueError):allocate_campaign_budgets({'token_limit':1,'request_limit':1},['one','two'])


def test_effective_settings_and_safe_model_label(tmp_path):
    from atomic_skillgraph.experiments.run_formal import model_settings,run_label
    cfg=config_for(tmp_path)
    model={'model_id':'vendor/model','endpoint':'https://example.test','api_key_env':'KEY','input_modalities':['text'],
        'dialect':'openai_chat','reasoning_effort':'medium','thinking_type':'disabled','token_limit_field':'max_completion_tokens',
        'capability_profile':{'tools':True,'thinking':False,'reasoning_effort':True}}
    cfg=model_settings(cfg,model)
    cfg['llm']['protocol']={'thinking_type':'enabled'}
    cfg['llm']['runtime']['protocol']={'thinking_type':'disabled'}
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('searchqa',{}))
    try:
        p=s.provider('runtime');payload=p._build_payload([{'role':'user','content':'public'}],[])
        assert payload['model']=='vendor/model' and payload['reasoning_effort']=='medium'
        assert 'thinking' not in payload and 'max_completion_tokens' in payload
        assert p.snapshot()['http_token_limit_field']=='max_completion_tokens'
        assert p.snapshot()['thinking_type'] is None and '/' not in run_label(model)
    finally:s.close()


@pytest.mark.parametrize('choice',[False,True])
def test_actual_trace_extractor_builder_one_binding_and_freeze(tmp_path,monkeypatch,worker,choice):
    s,t=setup(tmp_path);s.worker=worker;sent=[]
    if choice:
        s.config['learning'].setdefault('choice_guidance',{})['enabled']=True
        s.config['runtime'].setdefault('choice_guidance',{})['enabled']=True
        from dataclasses import replace
        t=replace(t,inputs={**t.inputs,'choices':[{'label':'A','text':'6'},{'label':'B','text':'8'}]})
    def post(*a,**kw):
        payload=kw['json'];sent.append(payload);name=payload['tools'][0]['function']['name']
        material=json.loads(payload['messages'][1]['content'])
        if name=='answer_step':
            value={'action':'finish','answer':'6'} if material['local_results'] else {
                'action':'execute_python','source':SOURCE,'arguments':{'n':3},'input_schema':SCHEMA,'output_schema':OUT}
        elif name=='submit_learning':
            e=material['experience']['local_evidence'][0]
            value={'decision':'propose_skill_and_program_spec','skill':{'goal':'double an integer','guidance':'Use explicit integer inputs.',
                'input_schema':SCHEMA,'output_schema':OUT,'execution_intent':'program_requested','result_role':'intermediate'},
                'realization_request':{'skill_id':'$new','action':'build','case_bindings':[{'case_id':t.physical_key,
                    'inputs':{'n':3},'prefix':[],'start_mode':'reset','local_evidence_ref':e['id'],'reference_fields':{'value':['value']}}]}}
        else:
            from atomic_skillgraph.empirical.model_view import expand_material
            material=expand_material(material)
            value={'source':SOURCE,'trial_inputs':material['build_request']['fixed_bindings']}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([value],name=name))
    monkeypatch.setenv('MODEL_API_KEY','intercepted-only')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    try:
        trace=s.run_task(t,learn=True)
        assert trace['learning_status']=='completed',trace['learning']
        assert len(sent)==4 and [q['tools'][0]['function']['name'] for q in sent]==['answer_step','answer_step','submit_learning','submit_program']
        for request,payload in zip(trace['requests'],sent):
            audit=request['http_attempts'][0]['final_payload_audit']
            assert audit['serialized_parameters']=={k:v for k,v in payload.items() if k not in {'messages','tools'}}
            assert audit['provider_snapshot']['http_token_limit_field']=='max_tokens'
        p=s.bank.get(trace['learning']['program'])
        assert s.bank.program_eligible(p) and len(s.bank.train_cases())==1 and len(s.bank.attempts(p['id']))==1
        assert s.bank.jobs()[0]['state']=='done' and s.bank.jobs()[0]['generation_count']==1
        if choice:
            from atomic_skillgraph.empirical.choice_guidance import STATS_KEY
            stats=json.loads(s.bank.db.execute('SELECT value FROM metadata WHERE key=?',(STATS_KEY,)).fetchone()[0])
            assert stats['N']==1
            asset=s.bank.all('skill')[0]
            assert asset['guidance']=='' and asset['id']=='skill_'+digest({k:v for k,v in asset.items() if k!='id'})
        s.bank.freeze(tmp_path/'frozen')
        b=Bank(tmp_path/'frozen',readonly=True)
        try:assert len(b.all('program'))==len(b.all('implementation'))==1 and b.get(p['id'])['local_validation']['passed']
        finally:b.close()
        write_fixture('full_learning_chain_choice' if choice else 'full_learning_chain',{'payloads':sent,'trace':trace,'program':p,'job':s.bank.jobs()[0]})
    finally:s.close()


def test_file_validation_rejects_wrong_effect_before_publication(tmp_path,worker):
    from atomic_skillgraph.harness.benchmarks import SpreadsheetAdapter
    from atomic_skillgraph.empirical.local_validation import artifact_identity
    cfg=config_for(tmp_path/'bank');cfg['harness']={'adapter':'spreadsheet'}
    task=PublicTask('sheet','sheet-source','Create a public file')
    adapter=SpreadsheetAdapter({'sheet':{}},cfg)
    s=EmpiricalSystem(cfg,harness=adapter,adapter_factory=lambda:SpreadsheetAdapter({'sheet':{}},cfg));s.worker=worker
    adapter.reset(task)
    source="def run(ctx, inputs):\n    from pathlib import Path\n    Path('/workspace/report.txt').write_text(inputs['text'])\n    return {'status':'ok','outputs':{'files':['report.txt']}}"
    schema=object_schema({'text':{'type':'string'}},['text'])
    out=object_schema({'files':{'type':'array','items':{'type':'string'}}},['files'])
    asset={'source':source,'entry':'run','input_schema':schema,'output_schema':out,'allowed_tools':[],
        'environment':s.config['program_environment'],'result_role':'intermediate'}
    try:
        original=worker.execute({**asset,'id':'original-effect'},{'text':'public source'},Broker(adapter,1))
        assert original['status']=='ok'
        published=adapter.workspace.root/original['workspace']['version']/'report.txt'
        before=published.read_bytes()
        ref={'outputs':original['outputs'],'local_artifact_identities':{
            'report.txt':artifact_identity(published)}}
        e={'id':'local:effect','kind':'native_operation','source_physical_key':task.physical_key,
            'source_trace_sha256':'actual-worker-fixture','prefix':[],
            'public_task':{'inputs':{'text':'public source'}},'operation':{'name':'execute_python','arguments':{'text':'public source'}},
            'reference':ref,'environment_identity':s.config['program_environment']}
        bad=s.bank.put('program',{**asset,'source':source.replace("write_text(inputs['text'])","write_text('wrong')")})
        binding={'inputs':{'text':'public source'},'prefix':[],'local_evidence_ref':e['id'],
            'reference_fields':{'files':['outputs','files']}}
        row=validate_on_source(s,bad,binding,task,{'local_evidence':[e]},'effect')
        assert not row['validation']['passed'] and not s.bank.program_eligible(s.bank.get(bad['id']))
        assert row['result']['status']=='execution_error'
        assert published.read_bytes()==before
        assert not s.bank.program_options(task.goal)
        write_fixture('rejected_file_effect',{'validation':row,'published_source_sha256':artifact_identity(published)})
    finally:s.close()


def test_fixed_runner_60_episodes_share_budget_and_resume_without_http(tmp_path,monkeypatch):
    from atomic_skillgraph.experiments import run_atomic_unified_validation as r
    from atomic_skillgraph.experiments.run_empirical import write_json
    code={'git_sha':'fixture','tracked_dirty':False,'source_sha256':'fixture'}
    monkeypatch.setattr(r,'code_identity',lambda:code)
    output=tmp_path/'batch';configs={};tasks={};records={}
    for b in r.BENCHMARKS:
        configs[b]={};tasks[b]={}
        for split,n in (('train',3),('val',4)):
            cfg=config_for(output/b/'train'/('bank' if split=='train' else 'frozen_bank'))
            cfg['budget']={**r.LIMITS,**r.PARENT_LIMITS}
            cfg['experiment'].update(benchmark=b,output_dir=str(output/b/split),runtime_mode='online' if split=='train' else 'frozen')
            configs[b][split]=cfg;tasks[b][split]=[]
            for i in range(n):
                task=PublicTask(b+':'+split+':'+str(i),b+':'+split+':physical:'+str(i),'Which word?',split=split)
                tasks[b][split].append(asdict(task));records[task.task_id]={'answers':['word']}
    m={'schema':'atomic-unified.fixed-validation.v1','output':str(output),'code':code,'configs':configs,'tasks':tasks,
        'schedule':r.fixed_schedule(),'limits':r.LIMITS,'parent_limits':r.PARENT_LIMITS,'source_identity':r.source_identity(),
        'external_files_sha256':{},'authority_sha256':'fixture','materialization_sha256':'fixture','model_coverage':{'fixture':True}}
    write_json(output/'manifest.json',m)
    monkeypatch.setattr(r,'create_simple_harness',lambda cfg:AnswerAdapter('searchqa',records,cfg))
    sent=[]
    def post(*a,**kw):
        payload=kw['json'];sent.append(payload);name=payload['tools'][0]['function']['name']
        body=response(content='word',finish='stop') if name=='answer_step' else response([{
            'decision':'upsert_guidance','guidance_skill':{'goal':'Which word?','guidance':'Use the public context.'}}],name=name)
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:body)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    env=tmp_path/'.env';env.write_text('MODEL_API_KEY=intercepted-only\n')
    result=r.run(output/'manifest.json',env)
    assert result['status']=='completed' and not result['missing_episodes'] and len(result['rows'])==60
    assert len(sent)==result['actual_http_attempts']==72 and result['actual_tokens']==360
    assert all(v['before']==v['after'] for v in result['frozen_identity'].values())
    assert all(v['normal']['n']==16 and v['censored']['n']==0 for v in result['comparisons'].values())
    for row in result['rows']:
        if row['arm']=='NoSkill':assert not row['injected_guidance_ids']
    assert r.run(output/'manifest.json',env)==result and len(sent)==72
    write_fixture('fixed_runner',{'result':result,'real_benchmark_performance':False})
