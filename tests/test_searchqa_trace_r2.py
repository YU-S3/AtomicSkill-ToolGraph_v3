"""R2 zero-HTTP engineering checks; fixtures are never published as learned assets."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.searchqa_guidance import (VERSION, MATERIAL, POLICY, IMAGE, RuntimeView,
    documents, checked_asset, select, validate_wire, quote_hash)
from atomic_skillgraph.empirical.searchqa_trace_learning import (partition, public_case, validate_candidate, learning_units)
from atomic_skillgraph.empirical.searchqa_program_probe import (prepare, execute, qualify, check, inputs, qualification, QUALIFICATION)
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint, save_json
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.experiments.run_searchqa_trace_r2 import PhaseGovernor
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_empirical import config_for
from test_cf2_contracts import http, response


def task(index,split='train'):
    return PublicTask('fixture:'+str(index),'physical:'+str(index),'Which year was the station opened?',
        {'context':'[DOC] 中文 Station '+str(index)+' opened in '+str(2000+index)+'. [DOC] Other dates are uncertain.'},split)


def row(t,correct=True):
    return {'physical_key':t.physical_key,'task':asdict(t),'prediction':'<answer>'+str(2000+int(t.task_id.split(':')[-1]))+'</answer>',
            'correct':correct,'empty_answer':False,'completion_truncated':False}


def candidate(ts):
    public={t.physical_key:public_case(row(t)) for t in ts}
    refs=[]
    for t in ts:
        d=public[t.physical_key]['documents'][0]
        refs.append({'case_id':t.physical_key,**{k:d[k] for k in ('doc_index','start','end','quote_sha')},'basis':'posthoc_public_check'})
    c={'kind':'atomic_procedure','method_version':VERSION,'goal':'Locate dated public evidence',
       'applicability':{'question_pattern':'temporal_numeric','public_evidence_required':['documents','temporal_question']},
       'procedure':['Locate dated passages and compare their stated relation to the question.'],
       'stop_condition':'Use direct reasoning if dates conflict or evidence is absent.',
       'source_case_ids':[t.physical_key for t in ts],'evidence_refs':refs,'parent_ids':[],'execution_intent':'guidance_only'}
    return c,public


def cfg(root):
    c=config_for(root/'bank')
    c['experiment'].update(benchmark='searchqa',searchqa_method_version=VERSION,output_dir=str(root/'out'))
    c['harness']={'adapter':'searchqa'}
    c['program_environment'].update(adapter_abi='simple.v2',image_digest=IMAGE)
    c['budget']={'token_limit':3100000,'request_limit':364,'finish_reserve':0,'eval_task_tokens':96000}
    c['llm']['runtime'].update(max_completion_tokens=32768,reasoning_effort='high')
    c['llm']['protocol']={'thinking_type':'enabled'}
    return validate_config(benchmark_config(c,'searchqa'))


SOURCE = '''import re
def run(ctx, inputs):
    text = inputs['context']
    if not text.strip() or not inputs['question'].strip():
        return {'status':'needs_input','outputs':{}}
    markers = list(re.finditer(re.escape('[DOC]'), text))
    spans = [(m.end(), markers[i+1].start() if i+1<len(markers) else len(text)) for i,m in enumerate(markers)]
    if not markers: spans = [(0,len(text))]
    found=[]
    for i,(start,end) in enumerate(spans):
        quote=text[start:end]
        if re.search(r'\\d', quote) and len(quote)<=600:
            found.append({'doc_index':i+1,'start':start,'end':end,'quote':quote})
    return {'status':'ok','outputs':{'matches':found[:8]}} if found else {'status':'not_found','outputs':{}}
'''


def fixture_program(bank,environment):
    c,sources=candidate([task(1),task(2)])
    asset=bank.put('skill',validate_candidate(c,sources))
    spec={'goal':'Locate exact passages containing numeric surfaces','selection_basis':'date_or_number',
          'applicability':c['applicability'],'source_case_ids':c['source_case_ids'],
          'proposal_origin':'candidate_derived_from_text','result_role':'intermediate'}
    return bank.put('program',prepare(SOURCE,spec,asset,environment)),asset


def test_Z04_partition_is_score_independent_and_disjoint():
    ts=[task(i) for i in range(300)]
    a=partition(ts);b=partition(list(reversed(ts)))
    assert a==b
    assert [len(a[k]) for k in ('induction','worker_probe','utility_probe')]==[240,30,30]
    assert len(set(a['induction']+a['worker_probe']+a['utility_probe']))==300
    with pytest.raises(ValueError):partition(ts[:-1]+[ts[0]])


def test_Z02_Z05_allowlisted_projection_and_posthoc_honesty():
    raw={**row(task(1)), 'gold':'secret', 'score':{'hard':True}, 'answers':['secret']}
    projected=public_case(raw)
    assert not {'gold','answers','correct','score'} & set(projected)
    assert projected['historical_executed_evidence']==[] and projected['historical_operation_count']==0
    assert projected['posthoc_located_answer_surface'][0]['basis']=='posthoc_located_answer_surface'
    assert quote_hash('中文')==__import__('hashlib').sha256('中文'.encode()).hexdigest()


@pytest.mark.parametrize('bad',['offset','hash','source','field','answer','one_case','length'])
def test_Z06_Z07_Z08_candidate_rejection_is_local(bad):
    good,sources=candidate([task(1),task(2)]);broken=deepcopy(good)
    if bad=='offset':broken['evidence_refs'][0]['end']=99999
    if bad=='hash':broken['evidence_refs'][0]['quote_sha']='bad'
    if bad=='source':broken['evidence_refs'][0]['case_id']='unknown'
    if bad=='field':broken['gold']='forbidden'
    if bad=='answer':broken['procedure']=['Always answer 2001.']
    if bad=='one_case':broken['source_case_ids']=[task(1).physical_key]*2
    if bad=='length':broken['procedure']=['x'*300,'y'*300]
    with pytest.raises((ValueError,TypeError,KeyError)):validate_candidate(broken,sources)
    assert checked_asset(validate_candidate(good,sources))


def test_Z09_Z10_selection_empty_compact_and_legacy_unchanged(tmp_path):
    b=Bank(tmp_path/'bank');b.bind_searchqa_method(VERSION)
    c,sources=candidate([task(1),task(2)])
    a=b.put('skill',validate_candidate(c,sources))
    irrelevant=PublicTask('x','x','Who painted this?',{'context':'[DOC] A painting.'})
    assert select([a],irrelevant,POLICY)['selected']==[]
    assert b.retrieve_guidance(irrelevant.goal,3)==[a]  # old primitive remains unchanged
    selection=select([a],task(3),POLICY)
    assert selection['selected']==[a]
    validate_wire([{'skill_id':a['id'],'goal':a['goal'],'guidance':a['guidance']}])
    with pytest.raises(ValueError):validate_wire([{'skill_id':a['id'],'goal':a['goal'],'guidance':a['guidance']}]*3)
    b.close()


def test_Z11_Z13_new_candidate_cannot_be_marked_usable(tmp_path):
    c=cfg(tmp_path);b=Bank(tmp_path/'bank',environment=c['program_environment']);b.bind_searchqa_method(VERSION)
    p,a=fixture_program(b,c['program_environment'])
    assert not b.program_eligible(p)
    with pytest.raises(ValueError):b.put('program',{**p,'state':'usable'})
    unknown={**p,'entry_constraints':{**p['entry_constraints'],'atomic_skill_id':'skill_unknown'}}
    with pytest.raises(ValueError,match='unknown Atomic'):b.put('program',unknown)
    with pytest.raises(ValueError):b.record_validation(p['id'],{'policy':QUALIFICATION,'passed':True})
    assert not qualification(p,{'policy':QUALIFICATION,'passed':True,'receipts':[]})
    malicious={'status':'ok','outputs':{'matches':[{'doc_index':1,'start':6,'end':12,'quote':'secret'}]}}
    with pytest.raises(ValueError):check(p,inputs(task(1)),malicious)
    b.close()


def test_Z18_phase_cap_is_independent_and_parent_charge_preserved(tmp_path):
    caps={'learn':{'tokens':100,'http':2},'worker_probe':{'tokens':1000,'http':10}}
    save_json(tmp_path/'derived_run_manifest.json',{'run_id':'unit','config':{'authorized_stages':list(caps),'phase_limits':caps}})
    limits={'token_limit':1200,'finish_reserve':0,'request_limit':20}
    old={'limits':limits,'attempts':{'legacy':{'context':{},'status':'metered','accounted_tokens':1000,'reserved_tokens':1000}},'unknown_billing':False}
    save_json(tmp_path/'inherited_budget.json',old);save_json(tmp_path/'budget.json',old)
    g=PhaseGovernor(tmp_path,'learn')
    assert g.headroom({})['tokens']==100
    g.admit('new',{'max_tokens':1},{})
    g.complete({'request_id':'new','raw_usage':{'total_tokens':90},'outcome':'success'})
    assert g.headroom({})['tokens']==10
    from atomic_skillgraph.core.errors import BudgetExhausted
    with pytest.raises(BudgetExhausted):g.admit('over',{'max_tokens':20},{})
    assert g.state['attempts']['legacy']==old['attempts']['legacy']
    assert PhaseGovernor(tmp_path,'worker_probe').headroom({})['tokens']==110


def test_Z17_learner_cached_response_does_not_issue_again(tmp_path,monkeypatch):
    config=cfg(tmp_path);system=EmpiricalSystem(config,harness=AnswerAdapter('searchqa',{}))
    system.checkpoint=TaskCheckpoint(tmp_path/'checkpoint');system.audit_path=tmp_path/'requests.json'
    seen=http(monkeypatch,[response([{'decision':'no_change','reason':'no common procedure'}],name='submit_searchqa_candidate')])
    ts=[task(1),task(2)];sources={t.physical_key:public_case(row(t)) for t in ts}
    u={'id':'unit_01','family':'temporal_numeric','case_ids':list(sources)}
    try:
        a=system.learner.learn_searchqa_trace(u,sources)
        b=system.learner.learn_searchqa_trace(u,sources)
        assert a==b and len(seen)==1
        changed=deepcopy(sources);changed[ts[0].physical_key]['question']='different public question'
        with pytest.raises(ValueError,match='changed across recovery'):system.learner.learn_searchqa_trace(u,changed)
        assert len(seen)==1
    finally:system.close()


@pytest.mark.skipif(not os.environ.get('PROGRAM_IMAGE_DIGEST'),reason='real locked Docker required')
def test_Z12_Z14_Z16_real_worker_transfer_and_runtime_consumption(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY','offline-http-interception-only')
    config=cfg(tmp_path);bank=Bank(tmp_path/'bank',environment=config['program_environment']);bank.bind_searchqa_method(VERSION)
    p,a=fixture_program(bank,config['program_environment'])
    bank.save_case(task(1),{'searchqa_role':'induction'});bank.save_case(task(3),{'searchqa_role':'worker_probe'})
    worker=ProgramWorker({'record_execution_receipt':True,'wall_timeout_seconds':15})
    receipts=[execute(bank,worker,p,task(1),tmp_path/'receipts',role='source'),
              execute(bank,worker,p,task(3),tmp_path/'receipts',role='transfer')]
    for name,args in [('empty_context',{**inputs(task(1)),'context':''}),('empty_question',{**inputs(task(1)),'question':''}),
                      ('unicode_parameter',{**inputs(task(1)),'context':'[DOC] 新文档\n'+task(1).inputs['context']})]:
        receipts.append(execute(bank,worker,p,task(1),tmp_path/'receipts',role='boundary:'+name,arguments=args))
    assert qualify(bank,p,receipts)['passed']
    p=bank.get(p['id']);assert bank.program_eligible(p)
    altered=deepcopy(receipts);altered[1]['result']['outputs']['matches'][0]['quote']='forged'
    assert not qualification(p,{**p['local_validation'],'receipts':altered})
    frozen=tmp_path/'frozen';bank.freeze(frozen);bank.close()
    config['data_dir']=str(frozen);config['experiment']['runtime_mode']='frozen'
    val=task(4,'val')
    def post(*args,**kwargs):
        material=json.loads(kwargs['json']['messages'][-1]['content'])
        local=material.get('local_results',[])
        step={'action':'call_program','name':p['id'],'arguments':{}} if not local else {
            'action':'finish','answer':'<answer>2004</answer>','used_results':[local[0]['result_id']]}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([step],name='answer_step'))
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    system=EmpiricalSystem(config,harness=AnswerAdapter('searchqa',{val.task_id:{'answers':['2004']}}),
        readonly=True,bank_view_factory=lambda b:RuntimeView(b,'Atomic-full'))
    system.worker=worker;before=system.bank.digest()
    try:
        trace=system.run_task(val,learn=False)
        assert trace['score']['hard'] and trace['execution']['attempts'][0]['outputs_consumed']
        assert trace['execution']['attempts'][0]['arguments']==inputs(val)
        assert trace['knowledge_after']==before==system.bank.digest()
        assert trace['method_identity']['searchqa_method_version']==VERSION
    finally:system.close()
    readonly=Bank(frozen,readonly=True)
    try:
        for arm in ('NoSkill','Guidance-only'):
            view=RuntimeView(readonly,arm);view.retrieve_searchqa_guidance(val,POLICY)
            assert view.get(p['id']) is None and view.program_options(val.goal)==[]
    finally:readonly.close()


def test_Z19_real_parent_trace_assembly_when_available():
    root=os.environ.get('SEARCHQA_R2_SOURCE_AUDIT')
    if not root:pytest.skip('requires real immutable Train preflight, not a synthetic fixture')
    path=Path(root)
    cases=[json.loads(line) for line in (path/'train_source_catalog.jsonl').read_text().splitlines()]
    groups=json.loads((path/'train_partition.json').read_text())
    assert len(cases)==300 and sum(r['correct'] for r in cases)==251
    sources={r['physical_key']:r for r in cases}
    units=learning_units(cases,groups)
    assert 0<len(units)<=12
    for u in units:
        assert 2<=len(u['case_ids'])<=4 and set(u['case_ids'])<=set(groups['induction'])
        for k in u['case_ids']:
            public=public_case(sources[k])
            assert public['context']==sources[k]['task']['inputs']['context']
            assert public['historical_executed_evidence']==[] and 'correct' not in public


def test_Z17_completed_stage_and_resume_arguments_are_immutable(tmp_path):
    from atomic_skillgraph.experiments.run_searchqa_trace_r2 import stage_receipt,check_arguments
    from atomic_skillgraph.empirical.searchqa_trace_learning import sha
    save_json(tmp_path/'train_probe/summary.json',{'complete':True})
    save_json(tmp_path/'deployment_policy.json',{'passed':False})
    saved={'passed':False,'one_time_decision':True}
    stage_receipt(tmp_path,'train_probe',saved)
    assert stage_receipt(tmp_path,'train_probe')==saved
    config=tmp_path/'config.yaml';config.write_text('locked: true')
    args=SimpleNamespace(config=str(config),source_run=str(tmp_path/'old'),datasets=str(tmp_path/'data'))
    plan={'source_run':str(tmp_path/'old'),'datasets':str(tmp_path/'data'),'config_file_sha256':sha(config)}
    check_arguments(args,plan)
    args.source_run=str(tmp_path/'other')
    with pytest.raises(ValueError,match='Resume'):check_arguments(args,plan)
    save_json(tmp_path/'deployment_policy.json',{'passed':True})
    with pytest.raises(ValueError,match='artifact changed'):stage_receipt(tmp_path,'train_probe')


def test_Z11_real_train_builder_wire_capacity(tmp_path,monkeypatch):
    audit=os.environ.get('SEARCHQA_R2_SOURCE_AUDIT')
    if not audit:pytest.skip('requires real immutable Train preflight')
    from atomic_skillgraph.empirical.searchqa_program_probe import build
    from atomic_skillgraph.empirical.searchqa_trace_learning import read
    from atomic_skillgraph.experiments.run_searchqa_trace_r2 import rows,unit
    root=Path(audit);sources={r['case_id']:r for r in rows(root/'public_evidence_index.jsonl')}
    units=read(root/'induction_units.json')
    seen=http(monkeypatch,[response([{'source':SOURCE}],name='submit_searchqa_code') for u in units])
    system=EmpiricalSystem(cfg(tmp_path),harness=AnswerAdapter('searchqa',{}))
    try:
        for u in units:
            ids=u['case_ids'];refs=[]
            for key in ids:
                d=sources[key]['documents'][0]
                refs.append({'case_id':key,**{k:d[k] for k in ('doc_index','start','end','quote_sha')},'basis':'posthoc_public_check'})
            c={'kind':'atomic_procedure','method_version':VERSION,'goal':'Locate public evidence windows',
                'applicability':{'question_pattern':u['family'],'public_evidence_required':['documents']},
                'procedure':['Compare document windows against the question constraints.'],
                'stop_condition':'Return to direct reasoning if the window is insufficient.',
                'source_case_ids':ids,'evidence_refs':refs,'parent_ids':[],'execution_intent':'guidance_only'}
            asset=validate_candidate(c,{k:sources[k] for k in ids})
            spec={'goal':c['goal'],'selection_basis':'document_partition','applicability':c['applicability'],
                'source_case_ids':ids,'proposal_origin':'candidate_derived_from_text','result_role':'intermediate'}
            unit(system,tmp_path/'builder'/u['id'],u['id'])
            built=build(system.learner,spec,asset,sources)
            assert built['state']=='candidate' and not built.get('local_validation')
        assert len(seen)==12 and max(len(json.dumps(p).encode()) for p in seen)<=65536
    finally:system.close()


def test_Z18_unknown_admission_blocks_new_http(tmp_path):
    save_json(tmp_path/'derived_run_manifest.json',{'run_id':'unit','config':{'authorized_stages':['learn'],
        'phase_limits':{'learn':{'tokens':10000,'http':3}}}})
    limits={'token_limit':10000,'finish_reserve':0,'request_limit':3}
    state={'limits':limits,'attempts':{},'unknown_billing':False}
    save_json(tmp_path/'inherited_budget.json',state);save_json(tmp_path/'budget.json',state)
    PhaseGovernor(tmp_path,'learn').admit('unfinished',{'max_tokens':100},{})
    recovered=PhaseGovernor(tmp_path,'learn')
    assert recovered.billing_blocked and recovered.headroom({})['tokens']<9900
    from atomic_skillgraph.core.errors import BudgetExhausted
    with pytest.raises(BudgetExhausted):recovered.admit('new',{'max_tokens':1},{})
    assert len(recovered.state['attempts'])==1


def test_Z09_Z15_Z16_real_source_empty_bank_production_stages(tmp_path,monkeypatch):
    """Real source and scorer, intercepted HTTP only. Not evidence of learned utility."""
    from atomic_skillgraph.experiments import run_searchqa_trace_r2 as runner
    audit=os.environ.get('SEARCHQA_R2_SOURCE_AUDIT')
    if not audit:pytest.skip('requires real immutable Train preflight')
    import shutil
    for p in Path(audit).iterdir():
        if p.is_file() and p.name.endswith(('.json','.jsonl')):shutil.copy2(p,tmp_path/p.name)
    identity=json.loads((tmp_path/'source_identity.json').read_text())
    source_config=json.loads((tmp_path/'source_config.json').read_text())
    datasets=Path(next(p for p in identity['source_files'] if p.endswith('/searchqa/train.json') and 'resources_' in p)).parent.parent
    template=Path(__file__).resolve().parents[1]/'configs/searchqa_trace_r2_seed42.yaml'
    config=tmp_path/'plan.yaml'
    config.write_text(template.read_text())
    args=SimpleNamespace(source_run=identity['source_run'],datasets=str(datasets),output=str(tmp_path),config=str(config),resume=False)
    original_code=runner.code_identity()
    monkeypatch.setattr(runner,'code_identity',lambda:{**original_code,'tracked_dirty':False})
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',runner.code_identity)
    runner.prepare(args)
    monkeypatch.setenv('MODEL_API_KEY','offline-http-interception-only')
    sent=[]
    def post(*a,**kw):
        payload=kw['json'];sent.append(payload)
        material=json.loads(payload['messages'][-1]['content'])
        if 'candidate_contract' in material:
            value=response([{'decision':'no_change','reason':'intercepted protocol fixture'}],name='submit_searchqa_candidate')
        else:value=response([{'action':'finish','answer':'<answer>offline fixture</answer>'}],name='answer_step')
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:value)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    learning=runner.learn(tmp_path)
    assert learning['decisions']=={'no_change':12}
    assert max(len(json.dumps(p).encode()) for p in sent)<=65536
    assert all('gold' not in json.loads(p['messages'][-1]['content']) for p in sent)
    assert runner.worker_probe(tmp_path)['spec_units']==0
    assert runner.train_probe(tmp_path)['utility_passed'] is None
    runner.freeze(tmp_path)
    result=runner.evaluate(tmp_path)
    assert list(result['arms'])==['NoSkill'] and result['arms']['NoSkill']['tasks']==24
    assert result['arms']['NoSkill']['program_invocations']==0
    report=runner.report(tmp_path)
    assert report['incremental_http']==96 and report['source_train_tokens']==2517211
    assert report['unknown_billing'] is False and not report['test_authorized']
    with pytest.raises(ValueError,match='Test requires'):runner.evaluate(tmp_path,'test')
