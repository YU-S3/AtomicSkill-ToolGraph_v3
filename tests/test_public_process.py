"""Fixed public-process/advisory contracts with intercepted production transport."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical import PUBLIC_CHOICE_POLICY_VERSION
from atomic_skillgraph.empirical.answer_executor import AnswerExecutor
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.choice_guidance import (build_verified_public_source,proposal_schema,
    validate_proposal,validate_check,checked_asset,select,retrieval_stats,render)
from atomic_skillgraph.empirical.contracts import PublicTask,digest,object_schema
from atomic_skillgraph.empirical.public_process import VERSION,FINALIZATION_VERSION,validate_process
from atomic_skillgraph.empirical.public_results import PROGRAM_EXECUTION
from atomic_skillgraph.empirical.system import EmpiricalSystem,validate_config
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http,response
from test_empirical import config_for,worker


def task(key='public-source',goal='Suppose x is an integer. Which is the strongest claim?',split='train'):
    return PublicTask(key,key+'-physical',goal,{'choices':[{'label':'A','text':'For every x, x equals itself.'},
        {'label':'B','text':'For some x, x equals itself.'}]},split)


def config(tmp_path,finalizer=False):
    raw=config_for(tmp_path/'bank');raw['experiment'].update(output_dir=str(tmp_path/'out'),
        choice_method_version=VERSION,choice_projection_version='choices.fixture.v1',choice_seed=42,
        public_materialization_sha256='fixture-public-hash')
    raw['llm'].update(max_retries=0,api_key_env='MODEL_API_KEY')
    raw['runtime']['choice_bounded_finalization']={'enabled':finalizer,'version':FINALIZATION_VERSION}
    return benchmark_config(raw,'livemath')


def process_step(answer='A'):
    return {'action':'finish','answer':answer,'public_process':'I compared the quantifiers in both public claims without treating examples as a proof.',
        'public_steps':[{'kind':'inspect_quantifiers','references':[
            {'task_path':['inputs','choices',0,'text'],'quote':'For every x'},
            {'task_path':['inputs','choices',1,'text'],'quote':'For some x'}]}]}


def evidence(t=None):
    t=t or task();return validate_process(process_step(),t,TaskContext(),[],[],0,'fixture-request')


def source(t=None):
    t=t or task()
    return build_verified_public_source(t,{'submission':'A','score':{'hard':True},'public_process_evidence':evidence(t)},
        {'method_version':VERSION,'projection_version':'choices.fixture.v1','public_materialization_sha256':'fixture-public-hash','choice_seed':42})


def proposal():
    return {'decision':'upsert_guidance','knowledge_type':'procedural_advisory','goal':'Inspect quantifiers',
        'guidance':'Compare universal and existential claims under the stated premises.',
        'applicability':'Only explicit quantifier comparisons; no theorem proof.',
        'operation_intent':'inspect_quantifiers','prerequisites':['multiple_public_claims','explicit_quantifiers'],
        'source_process_quote':process_step()['public_process']}


def check(p=None,s=None):
    p=p or proposal();s=s or source()
    return {'status':'supported','policy_version':PUBLIC_CHOICE_POLICY_VERSION,'proposal_hash':digest(p),
        'proposal_quote':p['guidance'],'source_quote':s['public_process_evidence']['public_process'],
        'reason':'Exact public self-report compatibility, not proof.'}


def asset():
    p=proposal();s=source()
    return {**p,'id':'skill_'+digest(p),'evidence_source':s,'grounded_proposal':p,
        'grounding_check':{**check(p,s),'source_hash':digest(s)},'execution_intent':'guidance_only',
        'guidance_policy_version':PUBLIC_CHOICE_POLICY_VERSION,'method_version':VERSION}


@pytest.mark.parametrize('structured',[False,True])
def test_direct_and_named_finish_capture_without_extra_reflection(tmp_path,monkeypatch,structured):
    calls=http(monkeypatch,[response([process_step()] if structured else [],name='answer_step',
        content='' if structured else 'A',finish='tool_calls' if structured else 'stop')])
    t=task();s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}))
    try:
        trace=s.run_task(t,learn=False);e=trace['execution']
        assert trace['score']['hard'] and len(calls)==1
        assert ('public_process_evidence' in e)==structured
        if structured:
            p=e['public_process_evidence'];assert p['verification']=='source_compatible' and p['kind']=='self_reported_process'
            assert p['observed_at'] and p['request_id']==trace['requests'][0]['id'] and not p['mathematically_proved']
            assert s.learner._experience(t,trace)['public_process_evidence']==p
    finally:s.close()


def test_actual_public_source_learning_and_frozen_selection(tmp_path,monkeypatch):
    calls=http(monkeypatch,[response([process_step()],name='answer_step'),response([proposal()],name='submit_learning'),
        response([check()],name='submit_guidance_check')])
    t=task();s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}))
    try:
        trace=s.run_task(t);assert len(calls)==3 and trace['learning']['decision']=='upsert_guidance'
        stored=s.bank.all('skill')[0];assert checked_asset(stored)
        assert stored['evidence_source']['projection_version']=='choices.fixture.v1'
        assert stored['evidence_source']['public_materialization_sha256']=='fixture-public-hash'
        assert stored['verification_strength']==['quote_matched','source_compatible']
        assert [r['purpose'] for r in trace['requests']]==['runtime','guidance_learning','guidance_grounding']
        frozen=tmp_path/'frozen';manifest=s.bank.freeze(frozen);assert manifest['choice_method_version']==VERSION
        before={p.name:p.read_bytes() for p in frozen.iterdir() if p.is_file()}
        b=Bank(frozen,readonly=True)
        try:
            b.bind_choice_method(VERSION)
            selected=b.select_guidance(task('other'),s.config['runtime']['choice_guidance'])
            assert selected['injected_ids']==[stored['id']] and selected['injected_chars']<=512
            with pytest.raises(RuntimeError):b.put('skill',stored)
        finally:b.close()
        assert before=={p.name:p.read_bytes() for p in frozen.iterdir() if p.is_file()}
        assert len(calls)==3
    finally:s.close()


def test_label_only_and_fake_public_source_cannot_create_procedure(tmp_path,monkeypatch):
    t=task();calls=http(monkeypatch,[response(content='A',finish='stop')])
    s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}))
    try:
        trace=s.run_task(t)
        assert trace['learning']['reason']=='public_operation_evidence_missing' and not s.bank.all('skill') and len(calls)==1
    finally:s.close()
    for change in ('quote','intent','proof','conditions'):
        p=proposal()
        if change=='quote':p['source_process_quote']='I solved a different equation.'
        if change=='intent':p['operation_intent']='deterministic_computation'
        if change=='proof':p['guidance']='This mathematically proves that the theorem is always true.'
        if change=='conditions':p['prerequisites']=['multiple_public_claims']
        with pytest.raises(ValueError):validate_proposal(p,source(),[],proposal_schema(VERSION))
    invalid=check();invalid['source_quote']='not the recorded process'
    with pytest.raises(ValueError):validate_check(invalid,proposal(),source(),__import__('atomic_skillgraph.empirical.prompts',fromlist=['GUIDANCE_CHECK']).GUIDANCE_CHECK)


@pytest.mark.parametrize('change',['unknown_path','wrong_quote','fake_result','fake_computation'])
def test_public_reference_rejection_preserves_answer_score(tmp_path,monkeypatch,change):
    step=process_step()
    if change=='unknown_path':step['public_steps'][0]['references'][0]['task_path']=['inputs','nonexistent']
    if change=='wrong_quote':step['public_steps'][0]['references'][0]['quote']='unseen premise'
    if change=='fake_result':step['public_steps'][0]['references'][0]={'result_id':'invented','path':['outputs'],'quote':'6'}
    if change=='fake_computation':step['public_steps'][0]['kind']='deterministic_computation'
    calls=http(monkeypatch,[response([step],name='answer_step')]);t=task()
    s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}))
    try:
        trace=s.run_task(t)
        assert trace['score']['hard'] and trace['execution']['public_process_evidence']['verification']=='rejected'
        assert trace['learning']['decision']=='no_change' and not s.bank.all('skill') and len(calls)==1
    finally:s.close()


def test_procedure_two_routes_and_negative_applicability():
    a=asset();assert checked_asset(a)
    stats=retrieval_stats([{'task':{'physical_key':task().physical_key,'goal':task().goal}}])
    policy={'max_items':2,'max_total_chars':2400,'public_process':True}
    assert len(select([a],stats,task('match'),policy)['selected'])==1
    unrelated=task('unrelated','Which city is the capital of France?')
    unrelated.inputs['choices']=[{'label':'A','text':'Paris'},{'label':'B','text':'London'}]
    assert not select([a],stats,unrelated,policy)['selected']
    assert not select([a],stats,task('proof','Prove that every group satisfies this theorem.'),policy)['selected']
    assert not select([a],stats,task(),{**policy,'public_process':False})['selected']
    old=deepcopy(a);old.pop('method_version');assert not checked_asset(old)
    false=deepcopy(a);false['operation_intent']='check_assumptions';assert not checked_asset(false)


@pytest.mark.parametrize('finalizer,answer,finish',[ (False,'','length'),(False,'unparseable','length'),
    (True,'','length'),(True,'A','length'),(True,'B','stop')])
def test_truncation_and_opt_in_one_bounded_final_answer(tmp_path,monkeypatch,finalizer,answer,finish):
    pending=[response(content=answer,finish=finish)]
    if finalizer and answer not in {'A','B'}:pending.append(response(content='A',finish='stop'))
    calls=http(monkeypatch,pending);t=task()
    gov=BudgetGovernor(tmp_path/'budget.json',token_limit=200000,request_limit=5,finish_reserve=0)
    s=EmpiricalSystem(config(tmp_path,finalizer),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}),budget_governor=gov)
    try:
        trace=s.run_task(t,learn=False);e=trace['execution']
        if finalizer and answer not in {'A','B'}:
            assert len(calls)==2 and not calls[1].get('tools') and calls[1]['thinking']['type']=='disabled'
            assert e['answer_origin']=='finish_only' and e['choice_bounded_finalization']['attempted']
            assert [r['purpose'] for r in trace['requests']]==['runtime','finish_only']
        else:
            assert len(calls)==1
            if answer not in {'A','B'}:assert e['reason']=='completion_truncated_no_answer' and e['answer_status']=='truncated_empty'
        assert len(gov.state['attempts'])==len(calls) and not gov.state['unknown_billing']
    finally:s.close()


def test_finalizer_spent_parent_does_not_send_http(tmp_path,monkeypatch):
    t=task();gov=BudgetGovernor(tmp_path/'budget.json',token_limit=200000,request_limit=5,finish_reserve=0)
    s=EmpiricalSystem(config(tmp_path,True),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}),budget_governor=gov)
    calls=http(monkeypatch,[])
    try:
        s.adapter.reset(t);s.task_context=TaskContext(s.config['runtime'])
        s.request_attribution={'parent_scope':'spent','parent_token_limit':100}
        result=AnswerExecutor(s).run(t,Broker(s.adapter,1,context=s.task_context),[],True)
        assert result['prediction']=='' and not calls and not gov.state['attempts']
    finally:s.close()


def test_optional_finalizer_failure_and_checkpoint_recovery_are_single_attempt(tmp_path,monkeypatch):
    t=task();calls=http(monkeypatch,[response(content='',finish='length'),response(content='A',finish='stop')])
    raw=config(tmp_path,True);raw['llm']['max_retries']=4
    gov=BudgetGovernor(tmp_path/'budget.json',token_limit=200000,request_limit=10,finish_reserve=0)
    s=EmpiricalSystem(raw,harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}),budget_governor=gov)
    from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
    s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');s.audit_path=str(tmp_path/'requests.json')
    original=TaskCheckpoint.commit_decision;interrupted=[]
    def commit(self,*args,**kwargs):
        if kwargs.get('answer_executor',{}).get('finished') and not interrupted:
            interrupted.append(True);raise OSError('fixture interruption after paid final answer')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(TaskCheckpoint,'commit_decision',commit)
    try:
        with pytest.raises(OSError):s.run_task(t,learn=False,attempt_id='fixed-attempt')
        assert len(calls)==2 and s.provider('runtime','finish_only').config.max_retries==0
        trace=s.run_task(t,learn=False,attempt_id='fixed-attempt')
        assert trace['score']['hard'] and len(calls)==len(gov.state['attempts'])==2
    finally:s.close()


def test_new_proposal_bad_leaf_types_fail_without_host_type_errors():
    for field in ('knowledge_type','operation_intent','prerequisites','source_process_quote'):
        for value in (123,True,None,{},[],['bad',None]):
            p={**proposal(),field:value}
            with pytest.raises(ValueError):validate_proposal(p,source(),[],proposal_schema(VERSION))


def test_two_quotes_of_one_location_do_not_create_a_comparison():
    step=process_step();step['public_steps'][0]['references'][1]={'task_path':['inputs','choices',0,'text'],'quote':'x equals itself'}
    with pytest.raises(ValueError,match='distinct references'):
        validate_process(step,task(),TaskContext(),[],[],0,'fixture-request')


def test_public_profile_retains_domain_fact_scope_and_limits():
    t=task(goal='Suppose x is an integer on manifold embeddings. Which is the strongest claim?');s=source(t)
    assets=[]
    for i in range(2):
        p={'decision':'upsert_guidance','knowledge_type':'domain_fact','goal':'Check manifold embeddings '+str(i),
            'guidance':'Keep all stated manifold embedding assumptions when comparing the conclusions.',
            'applicability':'Under the original manifold embedding premises.','scope_terms':['manifold','embeddings']}
        c=check(p,s);c['source_quote']=s['selected_choice_text']
        a={**p,'id':'skill_'+digest(p),'evidence_source':s,'grounded_proposal':p,
            'grounding_check':{**c,'source_hash':digest(s)},'execution_intent':'guidance_only',
            'guidance_policy_version':PUBLIC_CHOICE_POLICY_VERSION,'method_version':VERSION}
        assert checked_asset(a);assets.append(a)
    stats=retrieval_stats([{'task':{'physical_key':t.physical_key,'goal':t.goal}}])
    policy={'max_items':2,'max_total_chars':2400,'public_process':True}
    assert len(select(assets,stats,t,policy)['selected'])==2
    assert not select(assets,stats,task('one-term','Suppose the manifold is compact.'),policy)['selected']


def test_actual_worker_result_process_and_program_route(tmp_path,monkeypatch,worker):
    t=task(goal='Compute the numerical value of twice 3. Which claim is strongest?')
    s=EmpiricalSystem(config(tmp_path),harness=AnswerAdapter('livemath',{t.task_id:{'correct_choice':{'label':'A'},'choices':t.inputs['choices']}}));s.worker=worker
    monkeypatch.setenv('MODEL_API_KEY','test-only-key');calls=[]
    def post(*args,**kwargs):
        calls.append(deepcopy(kwargs['json']))
        if len(calls)==1:
            step={'action':'execute_python','source':"def run(ctx, inputs):\n    return {'status':'ok','outputs':{'value':inputs['n']*2}}",
                'arguments':{'n':3},'input_schema':object_schema({'n':{'type':'integer'}},['n']),
                'output_schema':object_schema({'value':{'type':'integer'}},['value'])}
        else:
            rid=next(iter(s.task_context.results));step=process_step()
            step.update(used_results=[rid],public_steps=[{'kind':'deterministic_computation','references':[
                {'task_path':['goal'],'quote':'twice 3'},{'result_id':rid,'path':['outputs','value'],'quote':'6'}]}])
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([step],name='answer_step'))
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    try:
        trace=s.run_task(t,learn=False)
        assert len(calls)==2 and trace['execution']['temporary_executions'][0]['outputs_consumed']
        assert trace['execution']['public_process_evidence']['verification']=='source_compatible'
        assert s.learner._experience(t,trace)['local_evidence']
    finally:s.close()


def test_old_bank_not_upgraded_and_projection_not_invented(tmp_path):
    b=Bank(tmp_path/'old');b.put('skill',{'goal':'old','guidance':'unchanged'})
    before=b.digest()
    try:
        with pytest.raises(ValueError,match='fresh Bank'):b.bind_choice_method(VERSION)
        assert b.digest()==before
    finally:b.close()
    s=build_verified_public_source(task(),{'submission':'A','score':{'hard':True},'public_process_evidence':evidence()},
        {'method_version':VERSION,'projection_version':None})
    assert not s['eligible'] and s['reason']=='public_projection_identity_missing'
    for benchmark in ('searchqa','officeqa','spreadsheetbench'):
        raw=benchmark_config(config_for(tmp_path/benchmark),benchmark)
        assert not raw['runtime'].get('choice_guidance',{}).get('public_process')
