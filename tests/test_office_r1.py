"""Fixed zero-model regressions: real protocol, recorded failures and ledger fixtures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import types

import pytest

from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor, input_token_bound
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.local_validation import evidence_from_trace, reference_value
from atomic_skillgraph.empirical.model_view import learning_case, structure_repair_material, expand_material
from atomic_skillgraph.empirical.public_results import SCOPED_SEARCH, FILE_COMPUTE
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from skillcompiler_bench_contracts.office import glob_paths, select_paths, grep
from test_cf2_contracts import office, http, response
from test_empirical import config_for


def record(name, value):
    root = os.environ.get('OFFICE_R1_EVIDENCE_DIR')
    if root:
        path = Path(root); path.mkdir(parents=True, exist_ok=True)
        (path / (name+'.json')).write_text(json.dumps({'fixture': 'zero-model engineering verification', **value},ensure_ascii=False,indent=2))


def test_glob_segment_scope_and_unicode(tmp_path):
    corpus=tmp_path/'corpus';corpus.mkdir()
    names=['a.txt','.hidden.txt','A.txt','d/b.txt','d/n/c.txt','other/e.txt']
    for name in names:
        path=corpus/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('一行\n第二行 target\nlast',encoding='utf-8')
    expected={'*.txt':['.hidden.txt','A.txt','a.txt'], '**/*.txt':sorted(names), '**/*':sorted(names),
              'd/*.txt':['d/b.txt'],'d/**/*.txt':['d/b.txt','d/n/c.txt'],'d\\**\\*.txt':['d/b.txt','d/n/c.txt'],
              'a?.txt':[], '[aA].txt':['A.txt','a.txt']}
    for pattern, selected in expected.items():assert glob_paths(corpus,pattern)==selected
    for pattern in ('/a.txt','../*.txt','C:\\*.txt','d/../../*'):
        with pytest.raises(ValueError):glob_paths(corpus,pattern)
    hits,audit=grep(corpus,'第二.*target',['d/b.txt'])
    assert hits[0]['offset']==3 and audit['execution_paths']==['d/b.txt']
    assert (corpus/'d/b.txt').read_text()[hits[0]['offset']:].startswith(hits[0]['text'])
    assert grep(corpus,'target',[])[0]==[]
    outside=tmp_path/'outside.txt';outside.write_text('secret');(corpus/'escape.txt').symlink_to(outside)
    with pytest.raises(ValueError):glob_paths(corpus,'**/*.txt')


def test_public_projection_binding_and_page_delivery(tmp_path):
    context=TaskContext()
    raw={'accepted':True,'data':[{'text':'命中'+str(i)} for i in range(43)],'scope_audit':{'execution_paths':[str(i) for i in range(697)]}}
    before=digest(raw);rid=context.register('grep',raw,name='grep',origin=SCOPED_SEARCH)
    assert 'scope_audit' not in context.view(rid) and digest(context.resolve({'result_id':rid,'path':[]}))==before
    assert 'scope_audit' not in context.bind({}, {'x':{'result_id':rid,'path':[]}})['x']
    e={'kind':'native_operation','operation':{'name':'grep'},'reference':raw,'result_protocol':SCOPED_SEARCH}
    for path in (['scope_audit'],['scope_audit','execution_paths']):
        with pytest.raises(ValueError):context.resolve_public({'result_id':rid,'path':path})
        with pytest.raises(ValueError):reference_value(e,{'kind':'json','path':path})
    assert 'scope_audit' not in reference_value(e,[])
    compute=context.register('compute',{'data':{'scope_audit':'business'},'scope_audit':'business'})
    assert context.bind({}, {'x':{'result_id':compute,'path':[]}})['x']['scope_audit']=='business'
    from atomic_skillgraph.empirical.public_results import PROGRAM_EXECUTION
    program=context.register('program',{'outputs':{'cpu_seconds':'business'},'cpu_seconds':1},origin=PROGRAM_EXECUTION)
    assert context.view(program)=={'result_id':program,'outputs':{'cpu_seconds':'business'}}
    with pytest.raises(ValueError):context.resolve_public({'result_id':program,'path':['cpu_seconds']})
    pages=[context.read(rid,path=['data'],offset=n) for n in (0,40)]
    assert [len(p['data']) for p in pages]==[40,3]
    assert [p['next_offset'] for p in pages]==[40,None]
    text='x'*11000+'后部标记'+'x'*1200
    tr=context.register('text',{'data':{'text':text}})
    page=context.read(tr,path=['data','text'])
    pr=context.register('page',page,name='read_result')
    assert context.view(pr)['data']['content']==text[:12000]
    assert page['end']==12000 and page['next_offset']==12000
    context.sent_windows=context.displayed_windows(context.view(pr))
    finish=build_finish_evidence(context,{'original_goal':'use obtained data','recent':[context.view(pr),context.view(rid)],
        'completed_results':{'n':{'text':{'result_id':tr,'path':['data','text']}}}},'finish')
    windows=finish['materials']['finish_evidence']
    assert any('后部标记' in w['content'] for w in windows if isinstance(w['content'],str))
    assert not any(w['source_ref']['path'][:1]==['scope_audit'] for w in windows)
    text_windows=[w for w in windows if w['source_ref']=={'result_id':tr,'path':['data','text']}]
    assert sum(w['end']-w['offset'] for w in text_windows)<=12000
    assert all(w['content_hash']==digest(w['content']) and w['end']-w['offset']==len(w['content']) for w in windows)
    # grep has acquired hit text, but has not acquired any un-read corpus body.
    assert 'unread-body-marker' not in json.dumps(finish['materials'])


def test_program_rpc_does_not_restore_host_scope(tmp_path):
    import time
    s=office(tmp_path)
    try:
        broker=Broker(s.adapter,3,context=s.task_context);broker.open_lease('fixture')
        value=broker.rpc('fixture',1,'call',{'name':'grep','arguments':{'pattern':'target'}},
            deadline=time.monotonic()+10,allowed_tools=['grep'])
        assert 'scope_audit' not in value
        assert broker.events[0]['result']['scope_audit']['execution_paths']==['a.txt']
    finally:s.close()


def test_wide_array_unit_is_not_half_json():
    context=TaskContext();rid=context.register('wide',{'data':[{'text':'z'*10000}, {'text':'small'}]})
    w=context.result_window({'result_id':rid,'path':['data']})
    fitted=context.fit_window(w,lambda w:len(json.dumps(w))<=600)
    assert fitted['content']==[] and fitted['next_offset']==0 and fitted['end']==0
    assert fitted['field_refs']==[{'result_id':rid,'path':['data',0,'text']}]


def test_learning_cards_keep_code_and_stdout_without_source(tmp_path):
    s=office(tmp_path);task=s.adapter.task
    try:
        rows=[]
        for i,(name,args,data,obs) in enumerate([
            ('glob',{'pattern':'**/*.txt'},['a.txt'],''),
            ('grep',{'pattern':'target'},[{'path':'a.txt','offset':0,'text':'target'}],''),
            ('read',{'path':'a.txt','offset':0},{'text':'target','next_offset':None},''),
            ('execute_python',{'source':'# '+'x'*990+'\nprint(6)'},{'files':[],'deleted_files':[]},'6\n')]):
            result={'accepted':True,'observation':obs,'data':data,'error':None,'done':False}
            if name=='execute_python':result.update(worker_seconds=1,worker_cpu_seconds=.1,local_artifact_identities={})
            if name=='grep':result['scope_audit']={'execution_paths':['a.txt']}
            rows.append({'state':'finished','backend_invoked':True,'event_id':str(i),'name':name,'arguments':args,
                         'result':result,'result_protocol':s.adapter.result_protocol(name)})
        experience={'task':{'goal':task.goal},'events':rows,'local_evidence':evidence_from_trace(task,{'tools':rows},s.config['program_environment'])}
        case=learning_case(experience,task.physical_key)
        assert len(case['local_evidence'])==4 and case['local_evidence'][0]['operation_name']=='execute_python'
        code=case['local_evidence'][0]
        assert any(r['path']==['observation'] and r['content_hint']=='6\n' for r in code['output_references'])
        assert 'worker_seconds' not in json.dumps(case) and '# xxx' not in json.dumps(case)
        code_e=experience['local_evidence'][-1]
        assert reference_value(code_e,{'kind':'json','path':['observation']})=='6\n'
        repaired=structure_repair_material({'realization_request':{'case_bindings':[{'local_evidence_ref':'unknown','inputs':{}}]}},[],[case['local_evidence']])
        assert len(repaired['legal_operations'])==4
        repaired=structure_repair_material({'realization_request':{'case_bindings':[{'local_evidence_ref':code['local_evidence_ref'],'inputs':{}}]}},[],[case['local_evidence']])
        assert len(repaired['legal_operations'])==1
        record('learning_directory',case)
    finally:s.close()


def test_disabled_proposal_and_one_bounded_repair(tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.prompts import LEARNING,LEARNER_PROMPT
    s=office(tmp_path);s.config=benchmark_config(s.config,'officeqa')
    sent=http(monkeypatch,[response([{'decision':'reuse_existing','existing_skill_id':'made_up'}],name='submit_learning'),
                           response([{'decision':'no_change'}],name='submit_learning')])
    try:
        material={'experience':{'task':{'goal':'fixed'},'local_evidence':[]}}
        assert s.agent('extractor',LEARNER_PROMPT,material,'submit_learning',LEARNING,
                       validator=s.learner.validate_learning_proposal,repair_limit=1)['decision']=='no_change'
        assert sent[0]['thinking']['type']=='disabled' and sent[0]['max_tokens']==8192
        assert sent[0]['tool_choice']['function']['name']=='submit_learning'
        assert sent[1]['max_tokens']==2048 and input_token_bound(sent[1])<=8192
        assert len(sent)==2
    finally:s.close()


@pytest.mark.parametrize('uid',['UID0091','UID0125','UID0164','UID0241'])
def test_recorded_admission_and_actual_finish(tmp_path,monkeypatch,uid):
    root=os.environ.get('OFFICE_R1_REPLAY_ROOT')
    if not root:pytest.skip('Set the recorded OfficeQA run for production admission playback')
    root=Path(root);phase=root/'test'
    db=sqlite3.connect((phase/'run.sqlite3').as_uri()+'?mode=ro',uri=True)
    trace=next(json.loads(r[0]) for r in db.execute('SELECT result FROM tasks') if json.loads(r[0])['task']['task_id']=='officeqa:'+uid);db.close()
    failed=next(r for r in trace['requests'] if r.get('error')=='Parent task HTTP admission refused')
    finish_request=next(r for r in trace['requests'] if r.get('purpose')=='finish_only')
    ledger=json.loads((root/'budget.json').read_text())
    prefix_ids={r['id'] for r in trace['requests'][:trace['requests'].index(failed)]}
    prefix={k:deepcopy(v) for k,v in ledger['attempts'].items() if v['context'].get('session_id') in prefix_ids}
    # Independent diagnostics; never write into the paid run.
    budget_path=tmp_path/'fixture_budget.json'
    budget_path.write_text(json.dumps({'limits':ledger['limits'],'attempts':prefix,'unknown_billing':False}))
    gov=BudgetGovernor(budget_path,**ledger['limits'])
    cfg=benchmark_config(config_for(tmp_path/'bank'),'officeqa')
    cfg['budget']={'token_limit':36096000,'request_limit':284070,'finish_reserve':0,'eval_task_tokens':96000}
    cfg['harness'].update(adapter='officeqa',corpus_root=json.loads((phase/'config.json').read_text())['harness']['corpus_root'])
    task=PublicTask(**trace['task']);adapter=OfficeAdapter({task.task_id:{'answer':'fixture'}},cfg)
    s=EmpiricalSystem(cfg,harness=adapter,budget_governor=gov);adapter.reset(task)
    s.phase='test';s.budget_scope='fixture'
    s.request_attribution={k:v for k,v in next(iter(prefix.values()))['context'].items() if k in {'parent_scope','parent_token_limit','parent_task_id','train_parent'}}
    context=TaskContext(s.config['runtime'])
    saved=json.loads((phase/'checkpoints'/task.task_id/'1/state.json').read_text())['executor_state']['context']
    for key,value in saved.items():setattr(context,key,deepcopy(value))
    s.task_context=context
    schema=failed['tools'][0]['function']['parameters']
    from atomic_skillgraph.agents.protocol import NativeToolSpec
    specs=[NativeToolSpec(t['function']['name'],t['function']['description'],t['function']['parameters']) for t in failed['tools']]
    old_module=types.ModuleType('old_finish_fixture');old_module.__package__='atomic_skillgraph.empirical'
    old_source=subprocess.check_output(['git','show','80762b4:src/atomic_skillgraph/empirical/finish_evidence.py'],text=True)
    exec(compile(old_source,'recorded_old_finish_fixture','exec'),old_module.__dict__)
    base=json.loads(finish_request['messages'][1]['content']);base.pop('finish_evidence',None)
    old_finish=old_module.build_finish_evidence(context,base,finish_request['messages'][0]['content'])
    provider=s.provider('runtime');finish_provider=s.provider('runtime','finish_only')
    old_wire=provider._build_payload(failed['messages'],specs);old_wire[provider.config.token_limit_field]=32768
    old_finish_wire=finish_provider._build_payload([{'role':'system','content':finish_request['messages'][0]['content']},
        {'role':'user','content':json.dumps(old_finish['materials'],ensure_ascii=False)}],[])
    old_reserve=input_token_bound(old_finish_wire)+2048
    with pytest.raises(BudgetExhausted):gov.admit('old_rejected_fixture',old_wire,{**s.request_attribution,'parent_finish_reserve':old_reserve})
    def project_recorded(v):
        if isinstance(v,dict):
            if isinstance(v.get('result_id'),str) and v['result_id'] in context.results and any(k in v for k in ('data','observation','scope_audit')):
                return context.view(v['result_id'])
            return {k:project_recorded(x) for k,x in v.items()}
        if isinstance(v,list):return [project_recorded(x) for x in v]
        return v
    material=project_recorded(json.loads(failed['messages'][1]['content']))
    sent=http(monkeypatch,[response([{'action':'finish','answer':'fixture'}]),response(content='fixture',finish='stop')])
    try:
        s.agent('runtime',failed['messages'][0]['content'],material,'runtime_step',schema,finish_request_material=base)
        next_entry=list(gov.state['attempts'].values())[-1]
        assert s.requests[-1]['parent_finish_reserve']==18432
        assert sent[0]['max_tokens']>=8192 and 'scope_audit' not in sent[0]['messages'][1]['content']
        # Simulate the complete declared input+output allowance, not just the tiny fixture response.
        next_entry['accounted_tokens']=next_entry['reserved_tokens']
        from atomic_skillgraph.empirical.checkpoint import save_json
        save_json(gov.path,gov.state)
        result=s.prepare_finish(base,finish_request['messages'][0]['content'])
        assert result['serialized_input_bytes']<=16384
        answer=s.agent('runtime',finish_request['messages'][0]['content'],result['materials'],None,None,decision_purpose='finish_only')
        assert answer=='fixture' and input_token_bound(sent[-1])<=16384
        assert sum(a['accounted_tokens'] for a in gov.state['attempts'].values())<=96000
        record(uid,{'old_input_bound':input_token_bound(old_wire),'old_finish_reserve':old_reserve,
            'new_input_bound':input_token_bound(sent[0]),'new_completion_cap':sent[0]['max_tokens'],
            'finish':{k:v for k,v in result.items() if k!='materials'},'actual_finalize_admitted':True})
    finally:s.close()


def test_real_authorized_glob_set():
    root=os.environ.get('OFFICE_R1_CORPUS')
    if not root:pytest.skip('Real authorized corpus not configured')
    expected=[n for n,_ in select_paths(root)]
    assert glob_paths(root,'**/*.txt')==expected==glob_paths(root,'**/*')
    record('real_corpus',{'authorized_count':len(expected),'authorized_names_hash':digest(expected)})


def test_wire_growth_unicode_finish_and_budget_end(tmp_path,monkeypatch):
    s=office(tmp_path);s.config=benchmark_config(s.config,'officeqa')
    s.budget_governor=BudgetGovernor(tmp_path/'fixture_budget.json',token_limit=96000,finish_reserve=0,request_limit=5)
    s.request_attribution={'parent_scope':'fixture','parent_token_limit':96000}
    s.phase='test'
    try:
        rid=s.task_context.register('growth',{'data':{'text':'引号"\\\n'*6000}})
        material={'original_goal':'Original complete question must survive','recent':[s.task_context.view(rid)]}
        finish=s.prepare_finish(material,'Finish with acquired evidence')
        assert finish['serialized_input_bytes']<=16384
        assert finish['materials']['original_goal']==material['original_goal']
        assert all(w['content_hash']==digest(w['content']) and w['next_offset']==w['end'] for w in finish['materials']['finish_evidence'])
        gov=s.budget_governor
        context={**s.request_attribution,'parent_finish_reserve':18432,'finish_request_reserve':1,'purpose':'runtime'}
        payload={'messages':[{'role':'user','content':'retry fixture'}],'max_tokens':8192}
        for i,purpose in enumerate(('runtime','runtime','planner')):
            gov.admit(str(i),payload,{**context,'purpose':purpose})
            gov.complete({'request_id':str(i),'raw_usage':{'total_tokens':100},'outcome':'fixture-metered'})
        assert gov.headroom({**context,'purpose':'runtime'})['requests']==2
        # A final decision must leave the one remaining HTTP slot for finish.
        gov.admit('last-runtime',payload,context)
        gov.complete({'request_id':'last-runtime','raw_usage':{'total_tokens':100},'outcome':'fixture-metered'})
        with pytest.raises(BudgetExhausted):gov.admit('retry-without-finish-slot',payload,context)
        sent=http(monkeypatch,[response(content='fixture',finish='stop')])
        finish=s.prepare_finish(material,'Finish with acquired evidence')
        assert s.agent('runtime','Finish with acquired evidence',finish['materials'],None,None,decision_purpose='finish_only')=='fixture'
        assert len(sent)==1 and len(gov.state['attempts'])==5
        # The base question itself cannot be cut to manufacture an available finish.
        s.budget_governor=None
        s.config['llm']['runtime']['max_total_tokens_per_task']=2600
        with pytest.raises(ValueError,match='finish_unavailable'):
            s.prepare_finish({'original_goal':'完整问题'*2000},'Finish')
    finally:s.close()


def test_learning_input_removes_history_before_code_card(tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.prompts import LEARNING,LEARNER_PROMPT
    s=office(tmp_path);s.config=benchmark_config(s.config,'officeqa')
    sent=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_learning')])
    card={'case_id':'case','local_evidence_ref':'recorded-code','operation_name':'execute_python','kind':'native_operation',
          'effect':'inspection_or_json','success':True,'published_files':[],'input_sources':[],
          'output_references':[{'kind':'json','path':['observation'],'content_hint':'6\n'}]}
    material={'experience':{'task':{'goal':'same goal'},'local_evidence':[card, {**card,'local_evidence_ref':'recorded-glob','operation_name':'glob'}]},
              'completed_train_cases':[{'task':{'goal':'optional'*10000}}], 'related':[{'id':'optional','guidance':'old'*10000}]}
    try:
        s.agent('extractor',LEARNER_PROMPT,material,'submit_learning',LEARNING)
        actual=expand_material(json.loads(sent[0]['messages'][1]['content']))
        assert input_token_bound(sent[0])<=16384
        assert not actual['completed_train_cases'] and not actual['related']
        assert actual['experience']['local_evidence'][0]['local_evidence_ref']=='recorded-code'
    finally:s.close()
