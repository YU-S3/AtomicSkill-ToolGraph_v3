"""P4 production parsing, source isolation, paid-response recovery and bounded flow."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import math_capability as m
from atomic_skillgraph.empirical.math_trace_learning import review_batch,review_checks,trace_records
from atomic_skillgraph.empirical.model_view import restore_math_items,expand_material
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.empirical.system import EmpiricalSystem,validate_config
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import run_math_p3 as runner
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from test_math_p3 import proposal,transport,prepared as prepare_p3,config as p3_config
from test_capability_batch import sources
from test_cf2_contracts import http,response
from test_public_process import task
from test_empirical import worker

REAL=Path('/home/yangchengyu/livemath_p3_schema_recovery_seed42_20261014')
TRAIN=Path('/home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42')


def config(path):
    cfg=p3_config(path);cfg['experiment']['choice_method_version']=m.P4
    return validate_config(m.configure(cfg))


def prepared(tmp_path,monkeypatch):
    source,_,env=prepare_p3(tmp_path,monkeypatch)
    bank=Bank(source/'train/bank',readonly=True)
    try:cases=bank.train_cases()
    finally:bank.close()
    folder=source/'train/traces';folder.mkdir(exist_ok=True)
    for old in folder.glob('*.json'):old.unlink() # only this test's newly constructed fixture
    for i,c in enumerate(cases):
        batch.write_json(folder/f'{i}.json',{'task':c['task'],'score':c['experience']['score'],
            'execution':{'prediction':c['experience'].get('submission')},'result_store':{}})
    root=tmp_path/'p4';runner.prepare(source,source,tmp_path/'data',root,version=m.P4)
    return source,root,env


def check_for(c,**changes):
    return {'candidate_hash':digest(c),'status':'procedural_only','proposal_quote':c['guidance'],
        'references':c['references'],'reason':'Conditional procedure with stop condition.',**changes}


def test_item_normalization_limits_and_strict_sources():
    rows=m.records(sources(),span_factory=m.math_spans);good=proposal(rows);bad=deepcopy(good)
    bad['references'][1]=bad['references'][0]
    good['parent_ids']=[{'p2_text_ref':'irrelevant-invalid'}]
    raw={'decision':'candidates','candidates':[good,bad]}
    normalized,audit=restore_math_items(raw,{})
    assert 'parent_ids' in raw['candidates'][0] and 'parent_ids' not in normalized['candidates'][0]
    valid,receipts=review_batch(normalized,rows);assert len(valid)==1
    assert [r['status'] for r in receipts]==['accepted','rejected']
    assert audit[0]['raw_hash']!=audit[0]['normalized_hash'] and audit[0]['discarded_fields']==['parent_ids']
    good=proposal(rows);good['guidance']='x'*640;good['prerequisites']=['x'*128];good['stop_condition']='x'*192
    c=m.validate_candidate(good,rows,version=m.P4)
    assert len(json.dumps(m.render(c),ensure_ascii=False))+2<=1536
    for field,n in [('guidance',641),('stop_condition',193),('applicability',129)]:
        changed={**good,field:'x'*n}
        with pytest.raises(ValueError):m.validate_candidate(changed,rows,version=m.P4)
    for change in [{'invented_metadata':True},{'references':[{'case_id':'invented','span_id':'fake'}]*2}]:
        assert not review_batch({'decision':'candidates','candidates':[{**good,**change}]},rows)[0]
    assert restore_math_items({'decision':'no_change','candidates':[{'guidance':{'p2_text_ref':'invented'}}]}, {})[0]=={'decision':'no_change'}
    for key in ['missing',{'bad':'id'}]:
        broken=proposal(rows);broken['references'][0]['span_id']={'p2_text_ref':key}
        normalized,_=restore_math_items({'decision':'candidates','candidates':[broken,proposal(rows,'negation_scope')]},{})
        assert len(review_batch(normalized,rows)[0])==1
    parent=proposal(rows);parent['parent_ids']=['invented']
    normalized,_=restore_math_items({'decision':'candidates','candidates':[parent]}, {},consolidation=True)
    assert normalized['candidates'][0]['parent_ids']==['invented']
    assert not review_batch(normalized,rows,parents={})[0]


def test_independent_checks_do_not_discard_valid_neighbors():
    rows=m.records(sources(),span_factory=m.math_spans)
    cs=[m.validate_candidate(proposal(rows,op),rows,version=m.P4) for op in ('quantifier_instantiation','negation_scope','implication_direction')]
    bad=check_for(cs[0]);bad['references']=deepcopy(bad['references']);bad['references'][0]['span_id']='invented'
    accepted,receipts=review_checks({'checks':[bad,check_for(cs[1],status='disputed'),check_for(cs[2],status='domain_fact')]},cs)
    assert set(accepted)=={cs[1]['id']}
    assert [r['status'] for r in receipts]==['rejected','accepted','rejected']
    assert not m.checked_asset({**cs[1],'state':'source_checked','source_check':accepted[cs[1]['id']]})
    accepted,_=review_checks({'checks':[check_for(cs[0]),check_for(cs[0]),check_for(cs[2])]},cs)
    assert set(accepted)=={cs[2]['id']}


def test_dedup_uses_steps_and_premises_then_operation_coverage():
    rows=m.records(sources(),span_factory=m.math_spans);a=proposal(rows);b=deepcopy(a)
    b['prerequisites']=['Verify that the domain is finite and fully enumerated.']
    values=[a,b,proposal(rows,'negation_scope')]
    cs,receipts=review_batch({'decision':'candidates','candidates':values},rows)
    assert len(cs)==3 and all(r['status']=='accepted' for r in receipts)
    order=runner.ranked_candidates(cs)
    assert len({c['operation'] for c in order[:2]})==2
    assert len(review_batch({'decision':'candidates','candidates':[a,a]},rows)[0])==1


def test_partial_repair_survives_failure_and_resume(tmp_path,monkeypatch):
    rows=m.records(sources(),span_factory=m.math_spans)
    good=proposal(rows);bad=proposal(rows,'negation_scope');bad['guidance']='x'*641
    mixed={'decision':'candidates','candidates':[good,bad]}
    c=m.validate_candidate(good,rows,version=m.P4)
    calls=http(monkeypatch,[response([mixed],name='submit_learning'),response([mixed],name='submit_learning'),
        response([{'checks':[check_for(c)]}],name='submit_math_checks')])
    cfg=config(tmp_path);s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    folder=tmp_path/'unit'
    try:
        batch.learning_boundary(s,folder,'test:learning',unit_limits={'token_limit':300000,'request_limit':3})
        result=m.learn(s.learner,rows)
        assert len(result['candidates'])==1 and len(calls)==3
        assert len([r for r in result['proposal_receipts'] if r.get('status')=='rejected'])==2
    finally:s.close()
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    try:
        batch.learning_boundary(s,folder,'test:learning',unit_limits={'token_limit':300000,'request_limit':3})
        assert m.learn(s.learner,rows)==result and len(calls)==3
    finally:s.close()


def test_interrupted_learning_after_paid_proposal(tmp_path,monkeypatch):
    from atomic_skillgraph.empirical.learner import Learner
    rows=m.records(sources(),span_factory=m.math_spans);value=proposal(rows)
    c=m.validate_candidate(value,rows,version=m.P4)
    calls=http(monkeypatch,[response([{'decision':'candidates','candidates':[value]}],name='submit_learning'),
        response([{'checks':[check_for(c)]}],name='submit_math_checks')])
    original=Learner._receive;interrupted=[False]
    def receive(self,key,*args,**kwargs):
        if '_check_' in key and not interrupted[0]:
            interrupted[0]=True;raise RuntimeError('simulated crash after paid proposal')
        return original(self,key,*args,**kwargs)
    monkeypatch.setattr(Learner,'_receive',receive)
    cfg=config(tmp_path)
    def attempt():
        s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
        try:
            batch.learning_boundary(s,tmp_path/'unit','test:learning',unit_limits={'token_limit':300000,'request_limit':3})
            return m.learn(s.learner,rows)
        finally:s.close()
    with pytest.raises(RuntimeError,match='simulated'):attempt()
    result=attempt();assert len(result['candidates'])==1 and len(calls)==2
    assert attempt()==result and len(calls)==2


@pytest.mark.parametrize('extra',[{'public_process':'x'*546},{'public_process':'short','public_steps':[{'kind':'inspect_quantifiers','references':[]}]}])
def test_answer_is_independent_from_optional_process(tmp_path,monkeypatch,extra):
    calls=http(monkeypatch,[response([{'action':'finish','answer':'<answer>A</answer>',**extra}],name='answer_step')])
    cfg=config(tmp_path);s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{'x':{'choices':task('x').inputs['choices'],'correct_choice':task('x').inputs['choices'][0]}}))
    try:trace=s.run_task(task('x',split='val'),learn=False)
    finally:s.close()
    assert trace['execution']['prediction']=='<answer>A</answer>' and trace['score']['hard']
    assert trace['execution']['public_process_evidence']['verification']=='rejected'
    assert not trace['execution']['choice_bounded_finalization']['attempted'] and len(calls)==1


@pytest.mark.parametrize('change',[{'answer':'<answer>Z</answer>'},{'invented_field':'ignore'},{'action':'execute_python','public_process':'x'*546}])
def test_required_answer_and_nonfinish_remain_strict(tmp_path,monkeypatch,change):
    http(monkeypatch,[response([{'action':'finish','answer':'A',**change}],name='answer_step')])
    s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{'x':{'choices':task('x').inputs['choices'],'correct_choice':task('x').inputs['choices'][0]}}))
    try:r=s.run_task(task('x',split='val'),learn=False)
    finally:s.close()
    assert r['execution']['prediction']=='' and r['execution']['reason']=='runtime_model_failure'


@pytest.mark.parametrize('no_candidate,answers,expected',[(True,('B','A'),0),(False,('A','B'),0),(False,('B','A'),2)])
def test_production_control_flow(tmp_path,monkeypatch,no_candidate,answers,expected):
    source,root,env=prepared(tmp_path,monkeypatch);calls=transport(monkeypatch,no_candidate=no_candidate,answers=answers)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=runner.execute(root,env);assert result['usable_skills']==expected
    plan=batch.read(root/'plan.json');assert plan['schema']==m.P4 and len(plan['schedule']['units'])==15
    assert len({k for u in plan['schedule']['units'] for k in u['source_keys']})==60
    count=len(calls);runner.execute(root,env);assert len(calls)==count
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not (root/'test/summary.json').exists()
    with pytest.raises(ValueError,match='Test requires'):runner.execute(root,env,test=True)
    if expected:
        bank=Bank(root/'frozen_bank',readonly=True)
        try:
            assets=bank.all('skill');assert len(assets)==expected and all(m.checked_asset(a) for a in assets)
            changed=deepcopy(assets[0]);changed['utility_receipt']['schema']=m.VERSION+'.val-utility.v1'
            assert not m.checked_asset(changed)
        finally:bank.close()
        assert runner.execute(root,env,test=True,reviewed=True)['status']=='completed'
        audit=batch.read(root/'test_freeze_audit.json');assert audit['hash_before_test']==audit['hash_after_test']
        assert batch.read(root/'test/summary.json')['tasks']==100


@pytest.mark.skipif(not REAL.exists(),reason='Saved P3 requests unavailable')
def test_all_saved_paid_unit_proposals(tmp_path):
    old_rows=batch.read(REAL/'source_records.json');by_key={r['task']['physical_key']:r for r in old_rows}
    results=[]
    for index in range(15):
        folder=REAL/'inherited/learning/units'/str(index);log=batch.read(folder/'learning.json')
        rows=[by_key[k] for k in log['source_case_ids']];attempts=[]
        for request in batch.read(folder/'requests.json')['requests']:
            calls=request.get('response',{}).get('tool_calls',[])
            if len(calls)!=1 or calls[0]['name']!='submit_learning':continue
            raw=calls[0]['arguments'];wire=json.loads(request['messages'][1]['content'])
            normalized,normalization=restore_math_items(raw,wire)
            try:accepted,receipts=review_batch(normalized,rows)
            except ValueError as exc:accepted=[];receipts=[{'status':'rejected','reason':str(exc)}]
            attempts.append({'request_id':request['id'],'accepted_source_count':len(accepted),
                'candidate_hashes':[digest(c) for c in accepted],'normalization':normalization,'receipts':receipts})
        assert attempts
        results.append({'unit':index,'old_decision':log['decision'],'attempts':attempts})
    assert any(a['accepted_source_count'] for r in results for a in r['attempts'])
    batch.write_json(tmp_path/'saved_toolcall_replay.json',{'new_model_http':0,'qualification':'offline admission only; not new model generation, content checks or utility','units':results})


@pytest.mark.skipif(not REAL.exists(),reason='Saved P3 Runtime response unavailable')
def test_saved_546_character_response(tmp_path,monkeypatch):
    key='skill_8e83b7b9536bee4fee819f244fb8e8a7b096c2542c4714f7bec8fff9bf67ecfe'
    folder=REAL/'val/candidates'/key/'selection/4/off'
    old=batch.read(folder/'trace.json');t=PublicTask(**old['task'])
    request=next(r for r in old['requests'] if r.get('response',{}).get('tool_calls'))
    call=request['response']['tool_calls'][0];assert len(call['arguments']['public_process'])==546
    payload=response([call['arguments']],name=call['name']);payload['usage']=request['http_attempts'][-1]['raw_usage']
    calls=http(monkeypatch,[payload])
    # A fixture scorer only lets the intercepted Runtime complete; it is not the
    # historical evaluator and its synthetic score is neither compared nor exported.
    s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{t.task_id:{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');s.audit_path=str(tmp_path/'requests.json')
    try:new=s.run_task(t,learn=False)
    finally:s.close()
    assert new['execution']['prediction']=='<answer>E</answer>'
    assert new['execution']['public_process_evidence']['verification']=='rejected' and len(calls)==1
    assert old['execution']['prediction']=='' and batch.read(folder/'trace.json')==old
    batch.write_json(tmp_path/'saved_answer_replay.json',{'new_model_http':0,'task_id':t.task_id,
        'prediction':new['execution']['prediction'],'process':new['execution']['public_process_evidence'],
        'old_score_unchanged':True,'qualification':'protocol replay; no new mathematical score'})


def test_partial_coverage_and_unknown_billing(tmp_path,monkeypatch):
    import test_math_p3 as original
    monkeypatch.setattr(original,'prepared',prepared)
    original.test_partial_applicability_and_frozen_single_skill_conflict(tmp_path/'coverage',monkeypatch)
    original.test_stage_limits_unknown_cost_and_no_borrowing(tmp_path/'billing',monkeypatch)


@pytest.mark.skipif(not TRAIN.exists(),reason='Original Train60 unavailable')
def test_real_trace_wire_capacity(tmp_path,monkeypatch):
    import requests
    from test_math_p3 import REAL_DATA
    cfg=batch.read(TRAIN/'train/config.json');cfg['experiment'].update(choice_method_version=m.P4,output_dir=str(tmp_path/'learning'))
    cfg['data_dir']=str(tmp_path/'bank');cfg['llm']['api_key_env']='MODEL_API_KEY';cfg=validate_config(m.configure(cfg))
    bank=Bank(TRAIN/'train/bank',readonly=True)
    try:cases=bank.train_cases()
    finally:bank.close()
    traces={r['task']['physical_key']:r for r in [batch.read(p) for p in (TRAIN/'train/traces').glob('*.json')]}
    rows,program_audit=trace_records(cases,traces)
    assert len(rows)==60 and len(program_audit['executions'])==3
    calls=transport(monkeypatch);post=requests.post
    def one_repair(url,**kwargs):
        reply=post(url,**kwargs);wire=kwargs['json'];body=expand_material(json.loads(wire['messages'][1]['content']))
        if wire.get('tools',[{'function':{'name':''}}])[0]['function']['name']=='submit_learning' and 'previous_proposal' not in body:
            payload=reply.json();args=json.loads(payload['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            args['candidates'][0]['guidance']='x'*641
            payload['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps(args)
            reply.json=lambda:payload
        return reply
    monkeypatch.setattr('requests.post',one_repair)
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}));candidates=[];audits=[]
    try:
        by_key={r['task']['physical_key']:r for r in rows}
        for i,unit in enumerate(runner.math_schedule([r['task'] for r in rows],version=m.P4)['units']):
            batch.learning_boundary(s,tmp_path/'learning'/str(i),'offline:'+str(i),unit_limits={'token_limit':runner.bounds()['unit_tokens'],'request_limit':3})
            log=m.learn(s.learner,[by_key[k] for k in unit['source_keys']]);assert log['candidates']
            candidates.extend(log['candidates']);audits.extend(deepcopy(s.requests))
        batch.learning_boundary(s,tmp_path/'integration','offline:integration',unit_limits={'token_limit':runner.bounds()['integration_tokens'],'request_limit':10})
        integrated=m.learn(s.learner,rows,parents={c['id']:c for c in candidates})
        assert integrated['candidates'];audits.extend(deepcopy(s.requests))
        largest=max(integrated['candidates'],key=lambda c:len(json.dumps(m.render(c))))
        public=[t for phase in ('val','test') for t in batch.read(REAL_DATA/'livemath'/(phase+'.json'))['tasks']]
        for i,t in enumerate(public):
            batch.learning_boundary(s,tmp_path/'routing'/str(i),'offline:route:'+str(i),unit_limits={'token_limit':18432,'request_limit':1})
            m.match(s.learner,largest,PublicTask(**t));audits.extend(deepcopy(s.requests))
    finally:s.close()
    for i,t in enumerate(public):
        t=PublicTask(**t);task_cfg=deepcopy(cfg);task_cfg['experiment']['output_dir']=str(tmp_path/'runtime'/str(i))
        target=EmpiricalSystem(task_cfg,harness=AnswerAdapter('livemath',{t.task_id:{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
        target.checkpoint=TaskCheckpoint(tmp_path/'runtime'/str(i)/'checkpoint');target.audit_path=str(tmp_path/'runtime'/str(i)/'requests.json')
        try:r=target.run_task(t,learn=False)
        finally:target.close()
        assert not r.get('error') and r['requests'][-1]['completion_cap']==65536
        audits.extend(r['requests'])
    assert all(a['final_payload_audit']['serialized_payload_utf8_bytes']<=r['payload_max_bytes'] for r in audits if r.get('payload_max_bytes') for a in r['http_attempts'])
    batch.write_json(tmp_path/'real_capacity.json',{'new_model_http':0,'status':'passed','qualification':'intercepted production Provider, synthetic replies, no effect estimate',
        'train_coverage':len(rows),'public_evaluation_wires':len(public),'intercepted_http':len(calls),
        'max_wire_bytes':max(a['final_payload_audit']['serialized_payload_utf8_bytes'] for r in audits for a in r['http_attempts']),
        'program_source_audit':program_audit})


@pytest.mark.skipif(not TRAIN.exists(),reason='Saved truncation records unavailable')
@pytest.mark.parametrize('key',['livemath:202602:32','livemath:202602:8'])
def test_real_truncation_recovery(tmp_path,monkeypatch,key):
    import test_math_p3 as original
    monkeypatch.setattr(original,'config',config)
    original.test_saved_real_truncations_and_idempotent_finalization(tmp_path,monkeypatch,key)


def test_actual_worker_and_optional_process(tmp_path,monkeypatch,worker):
    import test_public_process as original
    monkeypatch.setattr(original,'config',config)
    original.test_actual_worker_result_process_and_program_route(tmp_path,monkeypatch,worker)
