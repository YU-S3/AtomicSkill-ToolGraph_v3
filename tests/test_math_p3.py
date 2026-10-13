"""P3 contracts and complete intercepted production flow; no model transport escapes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical import math_capability as m
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.empirical.model_view import expand_material,restore_capability_response
from atomic_skillgraph.empirical.system import EmpiricalSystem,validate_config
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.experiments import run_math_p3 as p3
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.experiments.canonical_manifest import sha256
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_capability_schedule import prepare_fixture
from test_capability_batch import sources
from test_public_process import config as old_config,task
from test_cf2_contracts import http,response
from test_empirical import worker


def config(path):
    cfg=old_config(path);cfg['harness']['adapter']='livemath'
    cfg['experiment']['choice_method_version']=m.VERSION
    return validate_config(m.configure(cfg))


def proposal(rows,operation='quantifier_instantiation'):
    return {'decision':'candidate','category':next(k for k,v in m.OPERATIONS.items() if operation in v),'operation':operation,
        'guidance':'Instantiate each universal claim with an admissible object; compare its consequence with the existential claim.',
        'applicability':'Explicit quantified comparisons.','prerequisites':['Check object domain and hypotheses.'],
        'stop_condition':'Stop if admissibility cannot be verified; examples do not prove a universal statement.',
        'references':[{'case_id':r['task']['physical_key'],'span_id':r['spans'][0]['span_id']} for r in rows[:2]]}


def transport(monkeypatch,*,no_candidate=False,answers=('B','A'),unknown=False,eligible=lambda body:True):
    calls=[];ops=[v for values in m.OPERATIONS.values() for v in values];unit=[0]
    def post(url,**kwargs):
        wire=deepcopy(kwargs['json']);calls.append(wire)
        body=expand_material(json.loads(wire['messages'][1]['content']))
        name=wire.get('tools',[{'function':{'name':''}}])[0]['function']['name']
        if name=='submit_learning':
            if no_candidate:value={'decision':'no_change','guidance':{'p2_text_ref':'irrelevant-invalid'}}
            elif 'parent_candidates' in body:
                selected=[];seen=set()
                for c in body['parent_candidates']:
                    if c['operation'] in seen:continue
                    seen.add(c['operation']);selected.append({'decision':'candidate',**{k:v for k,v in c.items() if k!='id'},'parent_ids':[c['id']]})
                    if len(selected)==8:break
                value={'decision':'candidates','candidates':selected}
            else:
                value={'decision':'candidates','candidates':[proposal(body['train_cases'],ops[(unit[0]*4+i)%len(ops)]) for i in range(4)]}
                unit[0]+=1
        elif name=='submit_math_checks':
            value={'checks':[{'candidate_hash':c['candidate_hash'],'status':'procedural_only','proposal_quote':c['proposal']['guidance'],
                'references':c['proposal']['references'],'reason':'Conditional operations with explicit stopping conditions.'} for c in body['candidates']]}
        elif name=='submit_capability_match':
            available=body['spans'];applies=eligible(body)
            value={'task_hash':body['task_hash'],'capability_id':body['capability']['id'],'applicable':applies,
                'span_ids':[available[0]['span_id'],next(s['span_id'] for s in available if s['task_path'][0]=='inputs')] if applies else []}
        else:
            value=None
        result=response([value],name=name) if value is not None else response(content=answers[bool(body.get('guidance'))],finish='stop')
        if unknown:result['usage']={}
        return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:result)
    monkeypatch.setenv('MODEL_API_KEY','fixture-not-secret');monkeypatch.setattr('requests.post',post)
    return calls


def prepared(tmp_path,monkeypatch):
    source,old,_,env,_=prepare_fixture(tmp_path,monkeypatch,full=True)
    data=tmp_path/'data';summary=batch.read(source/'train/summary.json');summary['tasks']=60;batch.write_json(source/'train/summary.json',summary)
    manifest=batch.read(source/'run_manifest.json');manifest['benchmark']='livemath';batch.write_json(source/'run_manifest.json',manifest)
    cfg=batch.read(source/'train/config.json');cfg['harness']['adapter']='livemath';batch.write_json(source/'train/config.json',cfg)
    ex=batch.read(source/'train/execution_manifest.json');ex['config']=cfg;batch.write_json(source/'train/execution_manifest.json',ex)
    canonical=batch.read(Path(__file__).parents[1]/'data/main_experiment_v1/livemath/train.json')['tasks']
    by_id={r['task_id']:r for r in ex['tasks']};batch.write_json(data/'livemath/train.json',{'tasks':[by_id[r['task_id']] for r in canonical]})
    mat=batch.read(data/'materialization.json');mat['files_sha256']['livemath/train.json']=sha256(data/'livemath/train.json')
    batch.write_json(data/'materialization.json',mat)
    monkeypatch.setattr(p3,'code_identity',lambda:{'git_sha':'p3-fixture','tracked_dirty':False,'source_sha256':'p3-fixture-source'})
    root=tmp_path/'p3';p3.prepare(source,source,data,root)
    return source,root,env


def test_protocol_exact_refs_and_abstention():
    rows=m.records(sources(),span_factory=m.math_spans);value=proposal(rows)
    original=deepcopy(value);value['references'][0]['case_id']={'p2_text_ref':'c'};value['references'][0]['span_id']={'p2_text_ref':'s'}
    table={'p2_text_table':{'c':original['references'][0]['case_id'],'s':original['references'][0]['span_id']}}
    restored=restore_capability_response(value,table);assert restored==original
    assert m.validate_candidate(restored,rows)['source_case_ids']==sorted(r['task']['physical_key'] for r in rows)
    with pytest.raises(ValueError,match='Unknown'):restore_capability_response(value,{'p2_text_table':{}})
    with pytest.raises(ValueError,match='Ambiguous'):restore_capability_response(value,{'p2_text_table':{'a':'x','b':'x'}})
    assert restore_capability_response({'decision':'no_change','guidance':{'p2_text_ref':'unknown'}},table)=={'decision':'no_change'}
    value=deepcopy(original);value['references'][0]['span_id']='invented'
    with pytest.raises(ValueError):m.validate_candidate(value,rows)
    value=deepcopy(original);value['guidance']='Choose A.'
    with pytest.raises(ValueError):m.validate_candidate(value,rows)
    value=deepcopy(original);value['guidance']='x'*513;value['applicability']='x'*129
    with pytest.raises(ValueError) as error:m.validate_batch({'decision':'candidates','candidates':[value,deepcopy(value)]},rows)
    assert 'candidates[0]' in str(error.value) and 'candidates[1]' in str(error.value)


def test_truncation_tail_recovery_and_legacy_isolation(tmp_path,monkeypatch):
    first=response(content='',finish='length');first['choices'][0]['message']['reasoning_content']='界 mathematical reasoning. '*5000
    first['usage']={'prompt_tokens':3000,'completion_tokens':65536,'total_tokens':68536}
    calls=http(monkeypatch,[first,response(content='A',finish='stop')]);cfg=config(tmp_path)
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{'x':{'choices':task('x').inputs['choices'],'correct_choice':task('x').inputs['choices'][0]}}))
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');s.audit_path=str(tmp_path/'requests.json')
    try:trace=s.run_task(task('x',split='val'),learn=False)
    finally:s.close()
    assert trace['score']['hard'] is True and not trace['execution'].get('budget_censored')
    f=trace['execution']['choice_bounded_finalization'];assert f['attempted'] and f['valid_answer'] and not f['parent_budget_exhausted']
    assert f['reasoning_tail']['end_byte']-f['reasoning_tail']['start_byte']<=16384
    assert len(calls)==2
    assert calls[0]['max_tokens']==65536 and calls[1]['max_tokens']==16384
    assert not calls[1].get('tools') and calls[1]['reasoning_effort']=='low'
    body=json.loads(calls[1]['messages'][1]['content']);assert body['goal']==task('x').goal and body['inputs']==task('x').inputs
    assert len(body['previous_reasoning_tail']['text'].encode())<=16384
    legacy=old_config(tmp_path/'legacy');assert legacy['llm']['runtime']['max_completion_tokens']==32768
    cfg['harness']['adapter']='searchqa'
    with pytest.raises(ValueError,match='restricted'):validate_config(cfg)


@pytest.mark.parametrize('no_candidate,answers,expected',[(True,('B','A'),0),(False,('A','B'),0),(False,('B','A'),2)])
def test_complete_production_pipeline_and_resume(tmp_path,monkeypatch,no_candidate,answers,expected):
    source,root,env=prepared(tmp_path,monkeypatch);calls=transport(monkeypatch,no_candidate=no_candidate,answers=answers)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=p3.execute(root,env);assert result['status']=='awaiting_test_review' and result['usable_skills']==expected
    plan=batch.read(root/'plan.json');assert len(plan['schedule']['units'])==15
    assert len({k for u in plan['schedule']['units'] for k in u['source_keys']})==60
    count=len(calls);p3.execute(root,env);assert len(calls)==count
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    if no_candidate:assert len(calls)==15
    if expected:
        bank=Bank(root/'frozen_bank',readonly=True)
        try:
            assets=bank.all('skill');assert len(assets)==2 and all(m.checked_asset(a) for a in assets)
            changed=deepcopy(assets[0]);changed['utility_receipt']['schema']='choice-capability.utility.v1';assert not m.checked_asset(changed)
        finally:bank.close()
        with pytest.raises(ValueError,match='Test requires'):p3.execute(root,env,test=True)
        frozen=batch.read(root/'final_frozen_manifest.json')['final_artifact_hash']
        tested=p3.execute(root,env,test=True,reviewed=True);assert tested['status']=='completed'
        assert batch.read(root/'test/summary.json')['tasks']==100
        assert batch.read(root/'test_freeze_audit.json')=={'hash_before_test':frozen,'hash_after_test':frozen}
        ledger=batch.read(root/'budget.json')
        stages=[batch.read(root/s/'budget.json')['attempts'] for s in ('learning','val','test')]
        assert sum(len(a) for a in stages)==len(ledger['attempts'])==len(calls)


def test_stage_limits_unknown_cost_and_no_borrowing(tmp_path,monkeypatch):
    _,root,env=prepared(tmp_path,monkeypatch);calls=transport(monkeypatch,unknown=True)
    with pytest.raises(Exception):p3.execute(root,env)
    assert len(calls)==1 and batch.read(root/'budget.json')['unknown_billing']
    with pytest.raises(BudgetExhausted,match='billing'):p3.execute(root,env)
    assert len(calls)==1
    fresh=BudgetGovernor(tmp_path/'separate-budget.json',token_limit=10000000,request_limit=500,finish_reserve=0)
    folder=tmp_path/'stage';batch.write_json(folder/'campaign_budget_prefix.json',deepcopy(fresh.state))
    limited=p3.StageGovernor(fresh,{'run_id':'fixture:learning','new_limits':{'token_limit':20,'request_limit':2}},folder)
    with pytest.raises(BudgetExhausted):limited.admit('x',{'messages':[],'max_tokens':30},{'parent_scope':'fixture:learning:unit0'})
    assert not fresh.state['attempts']
    with pytest.raises(ValueError):limited.admit('x',{'messages':[],'max_tokens':1},{'parent_scope':'fixture:val:0'})


def test_p3_actual_python_worker(tmp_path,monkeypatch,worker):
    import test_public_process as original
    monkeypatch.setattr(original,'config',config)
    original.test_actual_worker_result_process_and_program_route(tmp_path,monkeypatch,worker)


def test_partial_applicability_and_frozen_single_skill_conflict(tmp_path,monkeypatch):
    _,root,env=prepared(tmp_path,monkeypatch)
    plan=batch.read(root/'plan.json');allowed={t['goal'] for t in plan['tasks']['val'][::2]}
    calls=transport(monkeypatch,eligible=lambda b:b['task']['goal'] in allowed)
    result=p3.execute(root,env);assert result['usable_skills']==2
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        assets=bank.all('skill');t=PublicTask(**plan['tasks']['val'][0]);spans=m.math_spans(t)
        matches={a['capability']['id']:{'task_hash':m.public_identity(t),'capability_id':a['capability']['id'],
            'applicable':True,'span_ids':[spans[0]['span_id'],next(s['span_id'] for s in spans if s['task_path'][0]=='inputs')]} for a in assets}
        selected=m.select(list(reversed(assets)),t,matches)
        assert selected['injected_ids']==[min(assets,key=lambda a:a['publication_rank'])['id']]
        for c in batch.read(root/'val/result.json')['receipts'].values():
            if c['passed']:assert len(c['paired'])==9
    finally:bank.close()


REAL_SOURCE=Path('/home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42')
REAL_RECOVERY=Path('/home/yangchengyu/livemath_p2_routing_recovery_seed42_20261013')
REAL_DATA=Path('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1')


@pytest.mark.skipif(not REAL_SOURCE.exists(),reason='Local immutable Train60 evidence unavailable')
def test_real_train_capacity_and_all_public_evaluation_wires(tmp_path,monkeypatch):
    import requests
    cfg=batch.read(REAL_SOURCE/'train/config.json');cfg['experiment'].update(choice_method_version=m.VERSION,output_dir=str(tmp_path/'learning'))
    cfg['data_dir']=str(tmp_path/'bank');cfg['llm']['api_key_env']='MODEL_API_KEY';cfg=validate_config(m.configure(cfg))
    bank=Bank(REAL_SOURCE/'train/bank',readonly=True)
    try:rows=m.records(bank.train_cases(),span_factory=m.math_spans)
    finally:bank.close()
    calls=transport(monkeypatch);post=requests.post
    def one_repair(url,**kwargs):
        reply=post(url,**kwargs);wire=kwargs['json'];body=expand_material(json.loads(wire['messages'][1]['content']))
        if wire.get('tools',[{'function':{'name':''}}])[0]['function']['name']=='submit_learning' and 'previous_proposal' not in body:
            payload=reply.json();args=json.loads(payload['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            args['candidates'][0]['guidance']='x'*513
            payload['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps(args)
            reply.json=lambda:payload
        return reply
    monkeypatch.setattr('requests.post',one_repair)
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}));candidates=[]
    try:
        by_key={r['task']['physical_key']:r for r in rows}
        for i,unit in enumerate(p3.math_schedule([r['task'] for r in rows])['units']):
            batch.learning_boundary(s,tmp_path/'learning'/str(i),'offline:'+str(i),unit_limits={'token_limit':p3.bounds()['unit_tokens'],'request_limit':3})
            log=m.learn(s.learner,[by_key[k] for k in unit['source_keys']]);assert len(log['candidates'])==4
            candidates.extend(log['candidates'])
        batch.learning_boundary(s,tmp_path/'integration','offline:integration',unit_limits={'token_limit':p3.bounds()['integration_tokens'],'request_limit':10})
        integrated=m.learn(s.learner,rows,parents={c['id']:c for c in candidates})
        assert len(integrated['candidates'])==8 and len(calls)==55
        learning_audits=deepcopy(s.requests)
        largest=max(integrated['candidates'],key=lambda c:len(json.dumps(m.render(c))))
        # Route on public Val/Test inputs only. No evaluator feedback enters learning.
        public=[t for phase in ('val','test') for t in batch.read(REAL_DATA/'livemath'/(phase+'.json'))['tasks']]
        for i,t in enumerate(public):
            batch.learning_boundary(s,tmp_path/'routing'/str(i),'offline:route:'+str(i),unit_limits={'token_limit':18432,'request_limit':1})
            assert m.match(s.learner,largest,PublicTask(**t))['applicable']
    finally:s.close()
    # Intercepted solving verifies actual Provider wire and budget admission, never model quality.
    for i,t in enumerate(public):
        task_cfg=deepcopy(cfg);task_cfg['experiment']['output_dir']=str(tmp_path/'runtime'/str(i))
        t=PublicTask(**t);target=EmpiricalSystem(task_cfg,harness=AnswerAdapter('livemath',{
            t.task_id:{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
        target.checkpoint=TaskCheckpoint(tmp_path/'runtime'/str(i)/'checkpoint');target.audit_path=str(tmp_path/'runtime'/str(i)/'requests.json')
        try:trace=target.run_task(t,learn=False)
        finally:target.close()
        assert not trace.get('error')
        assert trace['requests'][-1]['completion_cap']==65536
    output={'status':'passed','new_model_http':0,'transport':'intercepted fixture, not model results',
        'learning_requests_with_all_single_repairs':len(learning_audits),'train_count':60,'units':15,'public_eval_wires':117,
        'max_learning_wire_bytes':max(a['final_payload_audit']['serialized_payload_utf8_bytes'] for r in learning_audits for a in r['http_attempts']),
        'source_train_summary_sha256':sha256(REAL_SOURCE/'train/summary.json')}
    batch.write_json(tmp_path/'real_capacity.json',output)


@pytest.mark.skipif(not REAL_RECOVERY.exists(),reason='Local saved truncation responses unavailable')
@pytest.mark.parametrize('key',['livemath:202602:32','livemath:202602:8'])
def test_saved_real_truncations_and_idempotent_finalization(tmp_path,monkeypatch,key):
    trace=batch.read(REAL_RECOVERY/'val/traces'/(key+'.json'));t=PublicTask(**trace['task'])
    saved=[batch.read(p) for p in (REAL_RECOVERY/'val/checkpoints'/key/'1/responses').glob('*.json')]
    saved=next(r for r in saved if r['turn']['finish_reason']=='length');turn=saved['turn']
    first=response(content=turn['content'] or '',finish='length')
    first['choices'][0]['message']['reasoning_content']=turn['reasoning_content']
    first['usage']=saved['http_attempts'][-1]['raw_usage']
    calls=http(monkeypatch,[first,response(content=t.inputs['choices'][0]['label'],finish='stop')])
    cfg=config(tmp_path);cfg['experiment']['output_dir']=str(tmp_path/'out')
    original=TaskCheckpoint.commit_decision;crashed=[False]
    def interrupt(self,*args,**kwargs):
        if kwargs.get('answer_executor',{}).get('finished') and not crashed[0]:
            crashed[0]=True;raise RuntimeError('simulated interruption after metered finalization')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(TaskCheckpoint,'commit_decision',interrupt)
    def attempt():
        s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{t.task_id:{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}}))
        s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');s.audit_path=str(tmp_path/'requests.json')
        try:return s.run_task(t,learn=False,attempt_id='offline-saved-response')
        finally:s.close()
    with pytest.raises(RuntimeError,match='simulated'):attempt()
    completed=attempt();assert completed['execution']['answer_status']=='valid' and not completed['execution'].get('budget_censored')
    assert len(calls)==2 and calls[0]['max_tokens']==65536 and calls[1]['max_tokens']==16384
    f=completed['execution']['choice_bounded_finalization'];assert f['valid_answer'] and f['first_truncation']['no_valid_answer']
    assert len(batch.read(tmp_path/'out/budget.json')['attempts'])==2
