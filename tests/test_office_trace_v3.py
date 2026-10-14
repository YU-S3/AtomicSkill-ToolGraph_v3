"""V3 contracts and immutable real-source gates. All Provider transport is intercepted."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import time

import pytest

from atomic_skillgraph.empirical import OFFICE_TRACE_METHOD_VERSION as VERSION, OFFICE_TRACE_MATERIAL_VERSION as MATERIAL
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.source_catalog import catalog_material, bind_proposal, verified_sources
from atomic_skillgraph.empirical.model_view import learning_case, expand_material, executed_source_capacity
from atomic_skillgraph.empirical.retrieval_chain import recover, authentic, KIND
from atomic_skillgraph.empirical.local_validation import canonical_binding, validate_on_source
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from skillcompiler_bench_contracts.office import RetrievalSession, grep
from test_cf2_contracts import office, http, response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):raise AssertionError('No real model HTTP is authorized by this offline gate')
    monkeypatch.setattr('requests.sessions.Session.request', refuse)


def configure(config, root):
    config=deepcopy(config)
    config['experiment'].update(benchmark='officeqa',officeqa_method_version=VERSION,output_dir=str(root/'run'))
    config['data_dir']=str(root/'bank')
    config['llm']['api_key_env']='MODEL_API_KEY'
    config['budget']={'token_limit':85000000,'request_limit':480000,'finish_reserve':0,**config.get('budget',{})}
    config['runtime']['global_action_budget']=48 if config['experiment'].get('runtime_mode')=='frozen' else 64
    return validate_config(benchmark_config(config,'officeqa'))


def local_system(tmp_path):
    old=office(tmp_path);cfg=configure(old.config,tmp_path/'v3');task=old.adapter.task;old.close()
    adapter=OfficeAdapter({task.task_id:{'answer':'private'}},cfg);adapter.reset(task)
    s=EmpiricalSystem(cfg,harness=adapter);s.task_context=TaskContext(cfg['runtime'])
    s.adapter_factory=lambda:OfficeAdapter({task.task_id:{'answer':'private'}},cfg)
    return s,task


def source(s, task):
    broker=Broker(s.adapter,64,context=TaskContext(s.config['runtime']))
    hits=broker.call('grep',{'pattern':'target'})['data']
    broker.call('read',{'path':hits[0]['path'],'offset':hits[0]['offset']})
    return {'task':asdict(task),'tools':broker.events,'score':{'hard':False},
        'execution':{'attempts':[],'reason':'finish','prediction':'wrong'}}


def proposal_for(case, task):
    chain=next(e for e in case['local_evidence'] if e['kind']==KIND)
    material=catalog_material({'experience':learning_case({**case,'local_evidence':[chain]},task.physical_key)},trace=True)
    card=material['experience']['local_evidence'][0]
    proposal={'decision':'propose_skill_and_program_spec','generate_program':True,
        'skill':{'goal':'Retrieve the actual text at a selected search hit','execution_intent':'program_requested',
                 'result_role':'intermediate','entry_constraints':'Public regex and valid hit index'},
        'realization_request':{'skill_id':'$new','action':'build','case_bindings':[{'operation_id':card['operation_id'],
            'input_ids':{r['field'][0]:r['id'] for r in card['input_sources']},
            'output_ids':{'text':next(r['id'] for r in card['output_references'] if r.get('path')==['data','text'])}}]}}
    return proposal,material,chain


# This is an offline regression fixture, never a learned/published asset.
REPLAY_FIXTURE = '''def run(ctx, inputs):
    args = {'pattern': inputs['query']}
    if 'file_pattern' in inputs:
        found = ctx.call('glob', {'pattern': inputs['file_pattern']})
        if not found['accepted']: return {'status':'blocked','outputs':{}}
        args['paths'] = found['data']
    elif 'paths' in inputs: args['paths'] = inputs['paths']
    found = ctx.call('grep', args)
    if not found['accepted']: return {'status':'blocked','outputs':{}}
    hits = found['data']; index = inputs['hit_index']
    if not 0 <= index < len(hits): return {'status':'needs_input' if len(hits)==40 else 'not_found','outputs':{}}
    text = ctx.call('read', {'path':hits[index]['path'],'offset':hits[index]['offset']})
    if not text['accepted']: return {'status':'blocked','outputs':{}}
    return {'status':'ok','outputs':{'text':text['data']['text']}}
'''


def candidate(s, skill, source=REPLAY_FIXTURE):
    return s.bank.put('program',{'source':source,'entry':'run','backend':'sandbox_python_v1',
        'input_schema':skill['input_schema'],'output_schema':skill['output_schema'],'result_role':'intermediate',
        'allowed_tools':['glob','grep','read'],'environment':s.config['program_environment']})


def test_pagination_unicode_cursor_scope_and_invalidation(tmp_path):
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'a.txt').write_text('标题\n'+''.join('第%d行 target\n'%i for i in range(85)),encoding='utf-8')
    (corpus/'b.txt').write_text('other target',encoding='utf-8')
    session=RetrievalSession(corpus);first,audit=session.page('target')
    assert first['hits']==grep(corpus,'target')[0] and len(first['hits'])==40 and not first['complete']
    second,audit=session.page('target',cursor=first['next_cursor'])
    assert len(second['hits'])==40 and audit['continued'] and audit['scanned_paths']==['a.txt']
    third,_=session.page('target',cursor=second['next_cursor'])
    assert len(third['hits'])==6 and third['complete']
    hit=second['hits'][0];window=session.context(hit['path'],hit['offset'],0,100)
    assert window['text'].startswith(hit['text'])
    assert session.page('target')[1]['cache_hit']
    for args in [{'pattern':'other','cursor':first['next_cursor']}, {'pattern':'target','paths':['a.txt'],'cursor':first['next_cursor']},
                 {'pattern':'target','cursor':first['next_cursor']+'0'}, {'pattern':'target','paths':['../a.txt']}]:
        with pytest.raises(ValueError):session.page(**args)
    (corpus/'a.txt').write_text('changed target',encoding='utf-8')
    with pytest.raises(ValueError):session.page('target',cursor=first['next_cursor'])
    assert len(session.page('target')[0]['hits'])==2
    outside=tmp_path/'outside.txt';outside.write_text('secret')
    (corpus/'link.txt').symlink_to(outside)
    with pytest.raises(ValueError):session.page('target')


def test_adapter_resume_timeout_and_original_surface(tmp_path,monkeypatch):
    s,task=local_system(tmp_path)
    path=Path(s.config['harness']['corpus_root'])/'a.txt';path.write_text('target\n'*90)
    first=s.adapter.call('grep_page',{'pattern':'target'})
    s.adapter.reset(task)
    second=s.adapter.call('grep_page',{'pattern':'target','cursor':first['data']['next_cursor']})
    assert second['accepted'] and second['scope_audit']['continued']
    original=s.adapter.retrieval.page
    def slow(*a,**k):time.sleep(.03);return original(*a,**k)
    monkeypatch.setattr(s.adapter.retrieval,'page',slow)
    result=Broker(s.adapter,1,native_timeout=.001).call('grep_page',{'pattern':'target'})
    assert result['error_code']=='native_read_timeout'
    s.close()


def test_host_schema_unknown_ids_and_source_mismatch(tmp_path):
    s,task=local_system(tmp_path);trace=source(s,task);case=s.learner._experience(task,trace)
    proposal,material,chain=proposal_for(case,task)
    bound=bind_proposal(proposal,material,{task.physical_key:(task,case)},s.bank)
    assert bound['skill']['input_schema']['properties']['hit_index']=={'type':'integer'}
    assert bound['skill']['output_schema']['properties']=={'text':{'type':'string'}}
    assert authentic(chain,case)
    material['learning_material_version']=MATERIAL
    assert executed_source_capacity(material,s.config['learning'],16384,trusted=verified_sources([(task,case)]))==24576
    for change in ('id','control','role'):
        bad=deepcopy(proposal)
        if change=='id':bad['realization_request']['case_bindings'][0]['operation_id']='invented'
        if change=='control':bad['realization_request']['case_bindings'][0]['input_ids'].pop('query')
        if change=='role':bad['skill']['result_role']='final_answer'
        with pytest.raises(ValueError):bind_proposal(bad,material,{task.physical_key:(task,case)},s.bank)
    corrupted=deepcopy(case);corrupted['events'][0]['arguments']['pattern']='different'
    assert not authentic(chain,corrupted)
    s.close()


def test_nested_material_tables_and_composite_provenance():
    from types import SimpleNamespace
    from atomic_skillgraph.empirical.model_view import pack_trace_material
    from atomic_skillgraph.empirical.retrieval_chain import verify_workflow
    from atomic_skillgraph.empirical.contracts import object_schema
    long='actual source '*100
    contract={'input_schema':object_schema({'value':{'type':'string','description':long}},['value']),
              'output_schema':object_schema({'text':{'type':'string','description':long}},['text'])}
    original={'first':contract,'second':contract,'source':long,'again':long}
    assert expand_material(pack_trace_material(original))==original
    literal={'business_data':{'trace_value_ref':'not-a-host-reference'}}
    assert expand_material(literal)==literal
    programs={key:{'id':key,'source':'fixture','result_role':'intermediate','input_schema':object_schema({'x':{'type':'integer'}},['x']),
        'output_schema':object_schema({'y':{'type':'integer'}},['y'])} for key in ('p1','p2')}
    bank=SimpleNamespace(get=programs.get,program_eligible=lambda p:True,
        all=lambda kind:list(programs.values()) if kind=='program' else [])
    workflow={'goal':'Intermediate fixture','nodes':[
        {'id':'a','execution_mode':'program','program_id':'p1','args':{'x':{'literal':1}},'output_aliases':{'y':'value'}},
        {'id':'b','execution_mode':'program','program_id':'p2','args':{'x':{'from':'a','field':'value'}},'after':['a']}],
        'outputs':{'value':{'from':'b','field':'y'}}}
    case={'task':{'goal':'fixture','inputs':{}},'program_calls':[
        {'id':'one','program_id':'p1','status':'ok','result_id':'r1','arguments':{'x':1}},
        {'id':'two','program_id':'p2','status':'ok','result_id':'r2','arguments':{'x':2},
         'input_refs':{'x':{'result_id':'r1','path':['outputs','y']}}}],
        'result_store':{'program_results':{'r1':{'outputs':{'y':2}},'r2':{'outputs':{'y':3}}}}}
    assert verify_workflow(workflow,bank,case)['result_role']=='intermediate'
    case['program_calls'][1]['input_refs']={}
    with pytest.raises(ValueError,match='ResultRef'):verify_workflow(workflow,bank,case)


def test_production_learner_builder_one_repair_and_real_worker(tmp_path,monkeypatch):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'):pytest.skip('Locked Docker required')
    s,task=local_system(tmp_path);trace=source(s,task);case=s.learner._experience(task,trace)
    proposal,_,_=proposal_for(case,task)
    bad=deepcopy(proposal);bad['realization_request']['case_bindings'][0]['operation_id']='unknown'
    sent=http(monkeypatch,[response([bad],name='submit_learning'),response([proposal],name='submit_learning'),
                          response([{'source':REPLAY_FIXTURE}],name='submit_program')])
    result=s.learner.learn(task,trace)
    assert len(sent)==3,result
    assert result['tests'][0]['validation']['passed'],result
    assert s.bank.program_eligible(s.bank.get(result['program']))
    assert set(s.bank.get(result['program'])['allowed_tools'])=={'glob','grep','read'}
    assert s.requests[1]['repair']==1 and input_token_bound(sent[1])<=16384
    s.close()


@pytest.fixture
def actual_parent():
    path=os.environ.get('OFFICE_TRACE_PARENT')
    if not path:pytest.skip('Set immutable Office V2 parent for real gates')
    return Path(path)


def report(name,value):
    path=os.environ.get('OFFICE_TRACE_GATE_DIR')
    if path:
        Path(path).mkdir(parents=True,exist_ok=True)
        (Path(path)/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def test_actual_corpus_and_nested_chain_docker(actual_parent,tmp_path):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'):pytest.skip('Locked Docker required')
    settings=json.loads((actual_parent/'resolved_config.json').read_text())['recompile']
    cfg=configure(settings,tmp_path)
    original=json.loads((actual_parent/'plan.json').read_text())
    selected=None;counts=[]
    for row in sorted(original['source_records'],key=lambda r:r['canonical_run_order_index']):
        task=PublicTask(**row['task']);trace=json.loads(Path(row['path']).read_text())
        chains=recover(task,trace,cfg['program_environment'])
        assert all(authentic(e,{'events':trace['tools']}) for e in chains)
        counts.append({'task_id':task.task_id,'chains':len(chains),'original_trace_sha256':row['sha256']})
        if selected is None and any(e['operation']['source_context'] for e in chains):selected=(task,trace,chains)
    assert len(counts)==120 and selected
    task,trace,chains=selected
    adapter=OfficeAdapter({task.task_id:{'answer':'unused-private'}},cfg);adapter.reset(task)
    s=EmpiricalSystem(cfg,harness=adapter);s.task_context=TaskContext(cfg['runtime'])
    s.adapter_factory=lambda:OfficeAdapter({task.task_id:{'answer':'unused-private'}},cfg)
    case=s.learner._experience(task,trace);proposal,material,chain=proposal_for(case,task)
    bound=bind_proposal(proposal,material,{task.physical_key:(task,case)},s.bank)
    binding=canonical_binding(bound['realization_request']['case_bindings'][0],case,bound['skill'])
    program=candidate(s,bound['skill']);trial=validate_on_source(s,program,binding,task,case,'offline-real-chain')
    assert trial['validation']['passed'],trial['validation']
    constant=candidate(s,bound['skill'],"def run(ctx,inputs): return "+repr({'status':'ok','outputs':{'text':chain['reference']['data']['text']}}))
    assert not validate_on_source(s,constant,binding,task,case,'offline-constant')['validation']['passed']
    broker=Broker(adapter,48,context=TaskContext(cfg['runtime']))
    old=broker.call('grep',{'pattern':'Treasury'})
    first=broker.call('grep_page',{'pattern':'Treasury'})
    second=broker.call('grep_page',{'pattern':'Treasury','cursor':first['data']['next_cursor']})
    assert old['data']==first['data']['hits'] and second['data']['hits']!=first['data']['hits']
    hit=second['data']['hits'][0]
    window=broker.call('read_context',{'path':hit['path'],'offset':hit['offset'],'before':0,'after':1000})
    assert window['data']['text'].startswith(hit['text'])
    page_program={**program,'source':'''def run(ctx,inputs):
 a=ctx.call('grep_page',{'pattern':inputs['query']})
 b=ctx.call('grep_page',{'pattern':inputs['query'],'cursor':a['data']['next_cursor']})
 h=b['data']['hits'][0]
 c=ctx.call('read_context',{'path':h['path'],'offset':h['offset'],'before':0,'after':1000})
 return {'status':'ok','outputs':{'text':c['data']['text']}}
''','allowed_tools':['grep_page','read_context']}
    replay=s.worker.execute(page_program,{**binding['inputs'],'query':'Treasury'},broker)
    assert replay['status']=='ok',replay
    report('real_sources_and_docker.json',{'actual_model_http':0,'train_count':len(counts),
        'chain_source_count':sum(c['chains']>0 for c in counts),'sources':counts,
        'fixture_only_program':True,'source_trial':trial,'new_tool_worker':replay,
        'old_first_page_equal':True,'continuation_actual_hits':len(second['data']['hits'])})
    s.close()


def test_actual_uid0066_repair_and_all_source_payloads(actual_parent,tmp_path,monkeypatch):
    from itertools import repeat
    from atomic_skillgraph.empirical.model_view import builder_example
    from atomic_skillgraph.empirical.program_worker import program_permission_view
    from atomic_skillgraph.empirical.prompts import BUILD
    original=json.loads((actual_parent/'plan.json').read_text())
    settings=json.loads((actual_parent/'resolved_config.json').read_text())['recompile']
    cfg=configure(settings,tmp_path);s=EmpiricalSystem(cfg);s.phase='train';s.audit_path=tmp_path/'requests.json'
    sent=http(monkeypatch,repeat(response([{'decision':'no_change'}],name='submit_learning')))
    records=[];builder_materials=[]
    for row in sorted(original['source_records'],key=lambda r:r['canonical_run_order_index']):
        task=PublicTask(**row['task']);trace=json.loads(Path(row['path']).read_text())
        s.adapter.reset(task);s.task_context=TaskContext(cfg['runtime'])
        case=s.learner._experience(task,trace)
        if not any(e['kind']==KIND for e in case['local_evidence']):continue
        previous=len(s.requests);learned=s.learner.learn(task,trace)
        assert learned['decision']=='no_change',learned
        req=s.requests[previous];payload=sent[-1]
        assert req['payload_max_bytes']==24576 and input_token_bound(payload)<=24576
        proposal,material,chain=proposal_for(case,task)
        bound=bind_proposal(proposal,material,{task.physical_key:(task,case)},s.bank)
        binding=canonical_binding(bound['realization_request']['case_bindings'][0],case,bound['skill'])
        tools=[t for t in s.adapter.tool_definitions() if t['name'] in {'glob','grep','read'}]
        permissions=program_permission_view(tools,s.adapter.available_tools(),tool_surface=s.adapter.capabilities.tool_surface,
            workspace=s.adapter.workspace,environment=cfg['program_environment'])
        prepared=s.learner.builder_material(bound['skill'],[binding],[builder_example(binding,chain)],permissions)
        builder_materials.append((task.task_id,prepared))
        records.append({'task':task.task_id,'extractor_wire_bytes':input_token_bound(payload),
            'source_id':chain['id'],'no_model_response_reused':True})
    builder_sent=http(monkeypatch,repeat(response([{'source':REPLAY_FIXTURE}],name='submit_program')))
    for row,(_,material) in zip(records,builder_materials):
        material['learning_material_version']=MATERIAL
        try:
            s.agent('tool_builder',s.learner.builder_prompt(),material,'submit_program',BUILD,job_key=['offline-preflight',0,0])
        except ValueError as exc:
            assert getattr(exc,'code',None)=='material_too_large'
            row['builder_refused']=True;row['builder_wire_bytes']=s.requests[-1]['material_refusal']['wire_bytes']
        else:
            row['builder_refused']=False;row['builder_wire_bytes']=input_token_bound(builder_sent[-1])
        row['builder_limit']=s.requests[-1]['payload_max_bytes']
        assert row['builder_limit']==24576
    raw=json.loads((actual_parent/'recompile/units/officeqa:UID0066/requests.json').read_text())['requests'][-1]
    saved=expand_material(json.loads(raw['material_refusal']['payload']['messages'][1]['content']))
    saved['learning_material_version']=MATERIAL
    before=len(builder_sent)
    s.agent('tool_builder',s.learner.builder_prompt(),saved,'submit_program',BUILD,
        completion_override=raw['completion_cap'],repair_reason=raw['repair_reason'],job_key=raw['job_key'])
    assert len(builder_sent)==before+1 and s.requests[-1]['payload_max_bytes']==32768
    assert input_token_bound(builder_sent[-1])<=32768
    shown=expand_material(json.loads(builder_sent[-1]['messages'][1]['content']))
    for key in ('previous_failure','future_program_api','examples','build_request'):
        assert shown[key]==saved[key]
    report('all_source_capacity.json',{'actual_model_http':0,'sources':records,'new_source_count':len(records),
        'builder_capacity_refusals':sum(r['builder_refused'] for r in records),
        'uid0066':{'original_wire_bytes':raw['material_refusal']['wire_bytes'],
            'new_wire_bytes':input_token_bound(builder_sent[-1]),'limit':32768,'necessary_material_unchanged':True}})
    s.close()


def test_actual_prepare_derive_checkpoints_freeze_and_budget(actual_parent,tmp_path,monkeypatch):
    from itertools import repeat
    from atomic_skillgraph.experiments import office_source_recompile as entry
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.core.errors import BudgetExhausted
    monkeypatch.setattr(entry,'code_identity',lambda:{'git_sha':'offline-gate-only','tracked_dirty':False})
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',entry.code_identity)
    pending=tmp_path/'pending';plan=entry.prepare_trace(actual_parent,pending)
    assert plan['canonical_train_count']==120 and plan['chain_source_count']>0
    with pytest.raises(ValueError,match='pending'):entry.recompile(pending)
    revision={'approved':True,'authorization':'OFFLINE TEST FIXTURE ONLY; all HTTP intercepted',
        'total_limits':{**entry.TOTAL_LIMITS,'token_limit':90000000},'phase_token_limits':entry.PHASE_LIMITS,'recompile_http_limit':400}
    revision_file=tmp_path/'fixture_revision.json';revision_file.write_text(json.dumps(revision))
    root=tmp_path/'derived';plan=entry.prepare_trace(actual_parent,root,revision_file)
    parent_ledger=entry.read(actual_parent/'budget.json')
    assert entry.read(root/'budget.json')['attempts']==parent_ledger['attempts']
    before_assets=sqlite3.connect((actual_parent/'train/bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    original=list(before_assets.execute('SELECT * FROM assets ORDER BY id'));before_assets.close()
    bank=Bank(root/'train/bank');assert list(bank.db.execute('SELECT * FROM assets ORDER BY id'))==original
    assert sum(bank.program_eligible(p) for p in bank.all('program'))==8
    bank.close()
    # Stop before the actual unissued Builder: this gate verifies source decisions
    # without rerunning any historical paid candidate or injecting a learned asset.
    work=plan['source_records'];plan['source_records']=[r for r in work if r['work_type']=='retrieval_chain']
    entry.write_json(root/'plan.json',plan);entry.write_json(root/'plan_identity.json',{'sha256':entry.sha256(root/'plan.json')})
    sent=http(monkeypatch,repeat(response([{'decision':'no_change'}],name='submit_learning')))
    first=entry.recompile(root,max_new_records=2)
    assert not first['complete'];paid=len(sent)
    # Source code/config/corpus and checkpoint checks are production; cached units don't issue again.
    again=entry.recompile(root,max_new_records=1);assert again['checked']==first['checked']+1
    result=entry.recompile(root)
    assert result['complete'] and result['usable']==8 and result['checked']==120
    final=len(sent);assert entry.recompile(root)==result and len(sent)==final
    frozen=entry.read(root/'final_frozen_manifest.json')['final_artifact_hash']
    assert entry.tree_identity(root/'train/frozen_bank')['sha256']==frozen
    with pytest.raises(ValueError,match='approved Val review'):entry.evaluate(root,'test')
    g=entry.PhaseGovernor(root/'budget.json','recompile')
    assert g.headroom({})['requests']==400-final
    inherited=len(parent_ledger['attempts'])
    assert len(g.state['attempts'])==inherited+final
    # Exhaustion and unresolved billing have no Provider call.
    g.state['unknown_billing']=True
    with pytest.raises(BudgetExhausted):g.admit('blocked',{'max_tokens':1},{})
    g.state['unknown_billing']=False
    g.state['attempts']['fixture-exhaustion']={'reserved_tokens':3000000,'accounted_tokens':3000000,'status':'metered',
        'context':{'office_phase':'recompile','office_method':VERSION}}
    with pytest.raises(BudgetExhausted):g.admit('phase-exhausted',{'max_tokens':1},{})
    from types import SimpleNamespace
    eval_payloads=[]
    def transport(*a,**kwargs):
        wire=kwargs['json'];eval_payloads.append(wire)
        name=wire.get('tools',[{}])[0].get('function',{}).get('name') if wire.get('tools') else None
        if name=='submit_plan':body=response([{'mode':'compose','workflow':{'goal':'Answer the public question',
            'nodes':[{'id':'solve','execution_mode':'dynamic','goal':'Answer the public question','args':{}}],'outputs':{}}}],name=name)
        elif name=='runtime_step':body=response([{'action':'finish','answer':'offline fixture'}],name=name)
        elif name is None:body=response([],finish='stop',content='offline fixture')
        else:raise AssertionError('Unexpected evaluation or learning request: '+str(name))
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:body)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',transport)
    val=entry.evaluate(root,'val',max_new_tasks=1)
    assert not val['complete']
    partial_http=len(eval_payloads)
    val=entry.evaluate(root,'val');assert val['complete']
    review=tmp_path/'offline-review.json';entry.write_json(review,{'approved':True,'val_summary_sha256':entry.sha256(root/'val/summary.json')})
    test=entry.evaluate(root,'test',review=review);assert test['complete']
    assert entry.tree_identity(root/'train/frozen_bank')['sha256']==frozen
    assert all(entry.read(root/p/'frozen_check.json')['unchanged'] for p in ('val','test'))
    report('derive_checkpoint_budget.json',{'actual_model_http':0,'canonical_train':120,'chain_source_count':plan['chain_source_count'],
        'inherited_programs_unchanged':True,'inherited_usable':8,'intercepted_new_requests':final,
        'no_duplicate_resume_requests':True,'frozen_hash':frozen,'test_approval_guard':True,
        'unknown_billing_and_phase_exhaustion_guard':True,'val24_test102_intercepted':True,
        'eval_fixture_requests':len(eval_payloads),'partial_val_http':partial_http,
        'actual_budget_authorization':'still 85M; 90M used only by intercepted test fixture'})


def test_actual_unissued_builder_paid_checkpoint_is_reused(actual_parent,tmp_path,monkeypatch):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'):pytest.skip('Locked Docker required')
    from atomic_skillgraph.experiments import office_source_recompile as entry
    from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
    monkeypatch.setattr(entry,'code_identity',lambda:{'git_sha':'offline-recovery-gate','tracked_dirty':False})
    revision=tmp_path/'revision.json';entry.write_json(revision,{'approved':True,
        'authorization':'OFFLINE TEST ONLY; all transport intercepted','total_limits':{**entry.TOTAL_LIMITS,'token_limit':90000000},
        'phase_token_limits':entry.PHASE_LIMITS,'recompile_http_limit':400})
    root=tmp_path/'derived';plan=entry.prepare_trace(actual_parent,root,revision)
    repairs=[r for r in plan['source_records'] if r.get('builder_recovery')]
    assert len(repairs)==1
    plan['source_records']=repairs
    entry.write_json(root/'plan.json',plan);entry.write_json(root/'plan_identity.json',{'sha256':entry.sha256(root/'plan.json')})
    sent=http(monkeypatch,[response([{'source':"def run(ctx, inputs): return {'status':'blocked','outputs':{}}"}],name='submit_program')])
    original=TaskCheckpoint.commit_decision
    interrupted=[False]
    def interrupt(self,key,status,**kwargs):
        if not interrupted[0] and status=='applied' and any(k.startswith('builder_') for k in kwargs):
            interrupted[0]=True;raise RuntimeError('offline interruption after paid response checkpoint')
        return original(self,key,status,**kwargs)
    monkeypatch.setattr(TaskCheckpoint,'commit_decision',interrupt)
    with pytest.raises(RuntimeError,match='offline interruption'):entry.recompile(root)
    assert len(sent)==1
    ledger_before=entry.read(root/'budget.json')['attempts']
    result=entry.recompile(root)
    assert result['complete'] and len(sent)==1
    assert entry.read(root/'budget.json')['attempts']==ledger_before
    db=sqlite3.connect(root/'train/bank/bank.sqlite3')
    job=json.loads(db.execute('SELECT payload FROM realization_jobs WHERE id=?',(repairs[0]['builder_recovery']['job_id'],)).fetchone()[0]);db.close()
    assert job['generation_count']==2 and job['repair_used'] is True
    audit=entry.read(root/'recompile/units'/repairs[0]['work_id']/'requests.json')
    assert all(r['stage']=='tool_builder' for r in audit['requests'])
    assert any(r.get('recovered_response') for r in audit['requests'])
    report('unissued_builder_checkpoint.json',{'actual_model_http':0,'intercepted_http':1,
        'old_generation_count':2,'new_generation_count':job['generation_count'],'repair_used':True,
        'no_repeated_paid_response':True,'no_extractor_reissue':True,'usable_after_fixture':result['usable'],
        'wire_bytes':input_token_bound(sent[0]),'wire_limit':32768})
