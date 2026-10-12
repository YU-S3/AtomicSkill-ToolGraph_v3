"""P2 contracts and the production paired runner with intercepted HTTP only."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.empirical.model_view import expand_material
from atomic_skillgraph.empirical.system import EmpiricalSystem,validate_config
from atomic_skillgraph.experiments import run_choice_capability_batch as runner
from test_cf2_contracts import response
from test_public_process import config,task,asset as old_asset


def settings():return yaml.safe_load((Path(__file__).parents[1]/'configs/livemath_p2_selection_seed42.yaml').read_text())


def sources():
    return [{'task':task('source-'+str(i)).__dict__,'experience':{'submission':'B' if i==0 else '',
        'score':{'hard':False},'answer_status':'valid' if i==0 else 'truncated_empty'}} for i in range(2)]


def candidate():
    rows=cb.records(sources());refs=[{'case_id':r['task']['physical_key'],'span_id':r['spans'][0]['span_id']} for r in rows]
    cap=cb.validate_capability({'decision':'capability','required_conditions':'Compare explicit quantified claims.',
        'operations':['inspect_quantifiers'],'references':refs},rows)
    p={'decision':'candidate','guidance':'Separate universal from existential claims, then compare under each stated premise.',
        'applicability':'Use for explicit quantified comparisons.','references':refs}
    c=cb.validate_candidate(p,cap,rows)
    return {**c,'source_check':{'candidate_hash':digest(c),'status':'procedural_only','proposal_quote':c['guidance'],'references':refs}}


def match(c,t,eligible=True):
    spans=cb.spans(t)
    return {'task_hash':cb.public_identity(t),'capability_id':c['capability']['id'],'applicable':eligible,
        'span_ids':[spans[0]['span_id'],next(s['span_id'] for s in spans if s['task_path'][0]=='inputs')] if eligible else []}


def fixture_transport(monkeypatch,*,eligibility=True,answers=('B','A'),no_candidate=False,unknown=False,induction=None):
    calls=[]
    def post(url,**kwargs):
        wire=deepcopy(kwargs['json']);calls.append(wire);body=expand_material(json.loads(wire['messages'][1]['content']))
        name=wire.get('tools',[{'function':{'name':''}}])[0]['function']['name']
        if name=='submit_capability':
            rows=body['public_cases'];value={'decision':'no_change'} if no_candidate else {'decision':'capability',
                'required_conditions':'Compare explicit quantified claims.','operations':['inspect_quantifiers'],
                'references':[{'case_id':r['task']['physical_key'],'span_id':r['spans'][0]['span_id']} for r in rows]}
            if induction is not None:value=induction(rows)
        elif name=='submit_learning':
            value={'decision':'candidate','guidance':'Separate universal from existential claims and compare premises.',
                'applicability':'Explicit quantified comparisons.','references':body['capability']['definition']['references']}
        elif name=='submit_capability_check':
            c=body['candidate'];value={'candidate_hash':body['candidate_hash'],'status':'procedural_only','proposal_quote':c['guidance'],
                'references':c['proposal']['references']}
        elif name=='submit_capability_match':
            spans=body['spans'];value={'task_hash':body['task_hash'],'capability_id':body['capability']['id'],
                'applicable':eligibility(body) if callable(eligibility) else eligibility,'span_ids':[]}
            if value['applicable']:value['span_ids']=[spans[0]['span_id'],next(s['span_id'] for s in spans if s['task_path'][0]=='inputs')]
        else:
            result=response(content=answers(body) if callable(answers) else answers[bool(body.get('guidance'))],finish='stop')
            if unknown:result['usage']={}
            return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:result)
        result=response([value],name=name)
        return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:result)
    monkeypatch.setenv('MODEL_API_KEY','fixture-not-secret');monkeypatch.setattr('requests.post',post)
    return calls


def fixture_source(root,monkeypatch):
    spec=settings();base=root/'original';cfg=config(base)
    cfg['manifest']=str(Path(__file__).parents[1]/'data/main_experiment_v1/livemath/train.json')
    cfg['llm'].update(model='deepseek-v4-flash',generation_seed_supported=False,input_modalities=['text'],token_limit_field='max_tokens',
        capability_profile={'tools':True,'thinking':True,'reasoning_effort':True,'generation_seed':False})
    cfg['budget']={'token_limit':23232000,'request_limit':9165,'finish_reserve':0,
        'train_task_tokens':200000,'train_solve_tokens':120000,'train_learning_tokens':80000,'eval_task_tokens':96000}
    ids=spec['split']['candidate_source_task_ids']+spec['split']['selection_screen_ids']+spec['split']['selection_confirmation_ids']
    tasks=[task(k).__dict__ for k in ids]
    for t in tasks:
        t['physical_key']=spec['split']['selected_physical_keys'].get(t['task_id'],digest(t['task_id']))
        t['goal']+=' [Fixture public demand '+t['task_id']+']'
    b=Bank(base/'train/bank',environment=cfg['program_environment']);cases=[]
    for t in tasks:
        b.save_case(PublicTask(**t),{'submission':'B','score':{'hard':False},'answer_status':'valid'})
        cases.append({'task':t,'requests':[],'execution':{},'scoring_audit':{'raw_scorer_output':{'correct_label':'A','correct_text':t['inputs']['choices'][0]['text']}}})
    bank_hash=b.digest();b.close()
    runner.write_json(base/'train/config.json',cfg)
    runner.write_json(base/'train/execution_manifest.json',{'config':cfg,'tasks':tasks})
    runner.write_json(base/'train/summary.json',{'complete':True,'knowledge_digest':bank_hash,'cases':cases,'known_total_tokens':0})
    runner.write_json(base/'run_manifest.json',{'source_commit':spec['source']['historical_run_commit'],'run_seed':42,'split_seed':42,
        'provider_model_id':'deepseek-v4-flash','split_hashes':{'train':spec['source']['public_train_manifest_sha256']},'run_id':'explicit-simulated-fixture'})
    runner.write_json(base/'budget.json',{'unknown_billing':False,'attempts':{},'limits':{'token_limit':23232000,'request_limit':9165,'finish_reserve':0}})
    identity={'git_sha':'intercepted-fixture','tracked_dirty':False,'source_sha256':'simulated-fixture-source'}
    monkeypatch.setattr(runner,'code_identity',lambda:identity)
    spec_path=root/'contract.yaml';spec_path.write_text(yaml.safe_dump(spec));env=root/'empty.env';env.write_text('')
    out=root/'derived';plan=runner.prepare(spec_path,base,out)
    return base,out,plan,env


def test_exact_predeclared_split_and_readonly_prepare(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch);base,out,plan,env=fixture_source(tmp_path,monkeypatch)
    before={n:p.read_bytes() for n,p in [(str(p),p) for p in base.rglob('*') if p.is_file()]}
    assert runner.check(out/'plan.json')=={'status':'passed','new_model_http':0,'generation':48,'screen':6,'confirmation':6}
    selected={t['physical_key'] for rows in plan['selection_tasks'].values() for t in rows}
    assert not selected&{r['task']['physical_key'] for r in runner.read(out/'source_records.json')}
    assert not selected&{k for unit in plan['candidate_units'] for k in unit}
    assert not calls and before=={str(p):p.read_bytes() for p in base.rglob('*') if p.is_file()}
    changed=deepcopy(plan['settings']);changed['split']['selection_screen_ids'].reverse()
    with pytest.raises(ValueError):runner.partition(runner.read(base/'train/execution_manifest.json')['tasks'],changed)
    row=runner.read(out/'source_cases.json');row[0]['experience']['submission']='A';runner.write_json(out/'source_cases.json',row)
    with pytest.raises(ValueError,match='Prepared material'):runner.check(out/'plan.json')


@pytest.mark.parametrize('mutation',['duplicate','val','span','theorem','label'])
def test_source_and_procedural_fail_closed(mutation):
    raw=sources();c=candidate()
    if mutation=='duplicate':
        with pytest.raises(ValueError):cb.records([raw[0],raw[0]])
    elif mutation=='val':
        raw[0]['task']['split']='val'
        with pytest.raises(ValueError):cb.records(raw)
    elif mutation=='span':
        refs=deepcopy(c['proposal']['references']);refs[0]['span_id']='invented'
        with pytest.raises(ValueError):cb.source_refs(refs,c['source_records'])
    else:
        p=deepcopy(c['proposal']);p['guidance']='The answer is always correct: x = x.' if mutation=='theorem' else 'Choose A.'
        with pytest.raises(ValueError):cb.validate_candidate(p,c['capability'],c['source_records'])
    assert {r['observation']['outcome'] for r in c['source_records']}=={'incorrect','no_answer'}
    assert not c['mathematically_proved'] and c['state']=='candidate'


@pytest.mark.parametrize('eligibility,answers,no_candidate,status,promoted',[
    (True,('B','A'),False,'completed_selection',2),
    (True,('A','B'),False,'completed_selection',0),
    (False,('B','A'),False,'completed_selection',0),
    (True,('B','A'),True,'completed_no_candidate',0)])
def test_production_batch_selection_freeze_and_no_reexecution(tmp_path,monkeypatch,eligibility,answers,no_candidate,status,promoted):
    calls=fixture_transport(monkeypatch,eligibility=eligibility,answers=answers,no_candidate=no_candidate)
    base,out,plan,env=fixture_source(tmp_path,monkeypatch);before={str(p):p.read_bytes() for p in base.rglob('*') if p.is_file()}
    result=runner.execute(out/'plan.json',env)
    assert result['status']==status and result['new_http']==len(calls) and not result['source_train_runtime_reexecuted']
    b=Bank(out/'frozen_bank',readonly=True)
    try:
        assets=b.all('skill');assert len(assets)==promoted and all(cb.checked_asset(a) for a in assets)
    finally:b.close()
    runtime=[p for p in calls if p.get('tools',[{'function':{'name':''}}])[0]['function']['name']=='answer_step']
    if no_candidate or not eligibility:assert not runtime
    elif answers==('A','B'):assert len(runtime)==24  # screen only, both candidates stop before confirmation
    else:
        assert len(runtime)==48
        assert all(r['passed'] for r in [runner.read(p) for p in (out/'candidates').glob('*/utility_receipt.json')])
    assert before=={str(p):p.read_bytes() for p in base.rglob('*') if p.is_file()}
    previous=len(calls);assert runner.execute(out/'plan.json',env)==result and len(calls)==previous


def test_preserve_other_guidance_and_programs_in_targeted_view(tmp_path):
    c=candidate();t=task('other');b=Bank(tmp_path/'bank');old=b.put('skill',old_asset());b.update_choice_statistics();b.freeze(tmp_path/'frozen');b.close()
    b=Bank(tmp_path/'frozen',readonly=True)
    try:
        policy=validate_config(config(tmp_path/'cfg'))['runtime']['choice_guidance'];matches={c['capability']['id']:match(c,t)}
        off=cb.CapabilityView(b,matches,off=True,candidate=c);on=cb.CapabilityView(b,matches,candidate=c)
        assert off.select_guidance(t,policy)['injected_ids']==[old['id']]
        assert on.select_guidance(t,policy)['injected_ids']==[old['id'],c['id']]
        assert off.all('program')==on.all('program')==b.all('program')
        with pytest.raises(RuntimeError):off.put('skill',c)
    finally:b.close()


def test_budget_exhaustion_is_censored_without_refill(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch);base,out,plan,env=fixture_source(tmp_path,monkeypatch)
    plan['new_limits']['request_limit']=1;runner.write_json(out/'plan.json',plan);runner.write_json(out/'plan_identity.json',{'plan_hash':digest(plan)})
    result=runner.execute(out/'plan.json',env)
    assert result['status']=='stopped_budget_censored' and result['censored'] and result['new_http']==1
    assert not (out/'frozen_bank').exists()
    runner.execute(out/'plan.json',env);assert len(calls)==1


@pytest.mark.parametrize('count',[3,6])
def test_zero_screen_exposure_and_preflight_minimum(tmp_path,monkeypatch,count):
    ids=settings()['split']['selection_confirmation_ids'][:count]
    calls=fixture_transport(monkeypatch,eligibility=lambda b:any(k in b['task']['goal'] for k in ids))
    base,out,plan,env=fixture_source(tmp_path,monkeypatch);result=runner.execute(out/'plan.json',env)
    rows=[runner.read(p) for p in (out/'candidates').glob('*/utility_receipt.json')]
    assert all(r['screen_passed'] and r['passed']==(count>=4) for r in rows)
    assert all(len(r['paired'])==(count if count>=4 else 0) for r in rows)
    assert len([p for p in calls if p['tools'][0]['function']['name']=='answer_step'])==(4*count if count>=4 else 0)
    assert not any((out/'candidates').glob('*/selection/0'))


def test_exact_gate_boundaries_do_not_promote_tokens_only():
    rules=runner.policy(settings())
    def f(off,on,empty_off=0,empty_on=0):return {'off':{'correct':off,'empty':empty_off},'on':{'correct':on,'empty':empty_on},'exposed':True}
    assert cb.gate([f(0,0)],rules['screen'])
    assert cb.gate([f(0,1),f(0,1),f(0,1),f(1,0)],rules['promotion'])
    assert not cb.gate([f(0,1),f(0,1),f(0,1)],rules['promotion'])
    assert not cb.gate([f(0,1),f(0,1),f(1,0),f(1,0),f(0,1),f(0,1)],rules['promotion'])
    assert not cb.gate([f(0,1),f(0,1),f(0,0),f(0,0,empty_on=1)],rules['promotion'])
    assert not cb.gate([f(1,1)]*12,rules['promotion'])


def test_frozen_advisory_consumption_and_tampered_receipt_rejected(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch);base,out,plan,env=fixture_source(tmp_path,monkeypatch);runner.execute(out/'plan.json',env)
    b=Bank(out/'frozen_bank',readonly=True);asset=b.all('skill')[0];b.close()
    changed=deepcopy(asset);receipt=changed['utility_receipt'];receipt['paired'][0]['on']['correct']=0
    for pair in receipt['paired']:pair['on']['correct']=0
    receipt['receipt_hash']=digest({k:v for k,v in receipt.items() if k!='receipt_hash'})
    assert not cb.checked_asset(changed)
    fresh=task('separate-consumer','Compare universal and existential implications under explicit premises.',split='val')
    cfg=runner.read(out/'config.json');cfg['data_dir']=str(out/'frozen_bank');cfg['experiment'].update(runtime_mode='frozen',output_dir=str(out/'consumer'))
    matches={asset['capability']['id']:match(asset,fresh)}
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{fresh.task_id:{'correct_choice':fresh.inputs['choices'][0],'choices':fresh.inputs['choices']}}),
        bank_view_factory=lambda b:cb.CapabilityView(b,matches))
    try:
        trace=s.run_task(fresh,learn=False)
        assert trace['score']['hard'] and trace['injected_guidance_ids']==[asset['id']]
        assert trace['knowledge_before']==trace['knowledge_after'] and not trace.get('learning')
    finally:s.close()


def test_pair_checkpoint_replay_preserves_wire_usage_and_isolation(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch);base,out,plan,env=fixture_source(tmp_path,monkeypatch)
    cfg=runner.read(out/'config.json');s=EmpiricalSystem(cfg)
    try:s.bank.freeze(out/'trial')
    finally:s.close()
    c=candidate();t=PublicTask(**plan['selection_tasks']['screen'][0]);matches={c['capability']['id']:match(c,t)}
    governor=BudgetGovernor(out/'budget.json',**plan['new_limits'],finish_reserve=0)
    evaluate=lambda rank,index:runner.evaluate_pair(out/'pairs'/str(index),t,c,matches,cfg,governor,out/'trial',
        runner.evaluator_records(runner.read(base/'train/summary.json')),rank,index)
    first=evaluate(0,0);assert len(calls)==2
    saved=evaluate(0,0);assert first==saved and len(calls)==2
    second=evaluate(0,1);assert len(calls)==4
    assert bool(json.loads(calls[0]['messages'][1]['content'])['guidance']) is False
    assert bool(json.loads(calls[2]['messages'][1]['content'])['guidance']) is True
    assert first['off']['requests'][0]['id']!=second['off']['requests'][0]['id']
    bad=deepcopy(first);bad['on']['requests'][0]['effective_call_settings']['model']='other-model'
    with pytest.raises(ValueError,match='settings differ'):cb.pair_fact(c,t,bad)
    cfg['llm']['runtime']['max_completion_tokens']=4096
    with pytest.raises(ValueError,match='Checkpoint method identity changed'):evaluate(0,0)


def test_unknown_billing_blocks_following_arm_and_promotion(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch,unknown=True);base,out,plan,env=fixture_source(tmp_path,monkeypatch)
    result=runner.execute(out/'plan.json',env)
    assert result['status']=='stopped_unknown_billing' and result['unknown_billing'] and not (out/'frozen_bank').exists()
    before=len(calls);runner.execute(out/'plan.json',env);assert len(calls)==before


def test_real_train_public_sources_through_production_compiler(tmp_path,monkeypatch):
    source=os.environ.get('LIVEMATH_P1_SOURCE_RUN')
    if not source:pytest.skip('Set the immutable P1 Train60 source for actual-material interception')
    source=Path(source);spec=settings();summary=runner.read(source/'train/summary.json');original=runner.read(source/'train/config.json')
    generation,stages=runner.partition([t['task'] for t in summary['cases']],spec)
    files=[source/'train/bank/bank.sqlite3',source/'budget.json',source/'train/summary.json']
    before={str(p):p.read_bytes() for p in files}
    bank=Bank(source/'train/bank',readonly=True,environment=original['program_environment'])
    try:rows=bank.train_cases()
    finally:bank.close()
    by_key={r['task']['physical_key']:r for r in rows};cfg=runner.configuration(original,tmp_path/'derived');calls=fixture_transport(monkeypatch)
    s=EmpiricalSystem(cfg)
    try:
        for task_row in generation:s.bank.save_case(PublicTask(**task_row),by_key[task_row['physical_key']]['experience'])
        selected=[t['physical_key'] for rows in stages.values() for t in rows]
        for i in range(2):
            runner.learning_boundary(s,tmp_path/'learning'/str(i),'actual-fixture:'+str(i))
            log=s.learner.learn_capability_batch([t['physical_key'] for t in generation[2*i:2*i+2]],selected)
            assert log['decision']=='candidate',log
            candidate=log['candidate'];assert not set(candidate['source_case_ids'])&set(selected) and not s.bank.all('skill')
            assert len(json.dumps(cb.render(candidate),ensure_ascii=False))+2<=512
        assert len(calls)==6 and all(p['max_tokens']==2048 or p['max_tokens']==1536 for p in calls)
        assert all(p['thinking']['type']=='disabled' for p in calls)
        s.bank.put('skill',candidate);s.bank.freeze(tmp_path/'unqualified-frozen')
        frozen=Bank(tmp_path/'unqualified-frozen',readonly=True)
        try:assert not frozen.all('skill')
        finally:frozen.close()
        # Fixed intercepted answers validate the Host path on actual public Train input; these are not model scores.
        s.bank.db.execute('DELETE FROM assets');s.bank.db.commit();s.bank.freeze(tmp_path/'trial')
        evaluator=runner.evaluator_records(summary);selection=[PublicTask(**t) for rows in stages.values() for t in rows]
        labels={t.goal:evaluator[t.task_id]['correct_choice']['label'] for t in selection}
        def controlled_answer(body):
            correct=labels[body['goal']]
            return correct if body.get('guidance') else next(c['label'] for c in body['inputs']['choices'] if c['label']!=correct)
        paired_calls=fixture_transport(monkeypatch,answers=controlled_answer);governor=BudgetGovernor(tmp_path/'fixture_budget.json',token_limit=1800000,request_limit=80,finish_reserve=0)
        pairs={};preflight=[]
        for i,t in enumerate(selection):
            m=match(candidate,t);preflight.append({'task':t.__dict__,'stage':'screen' if i<6 else 'confirmation','rank':i,'match':m})
            pairs[t.physical_key]=runner.evaluate_pair(tmp_path/'pairs',t,candidate,{candidate['capability']['id']:m},cfg,
                governor,tmp_path/'trial',evaluator,i,0)
        receipt=cb.utility_receipt(candidate,pairs,runner.policy(spec),preflight,{'fixture_only':True})
        asset=cb.publish(s.bank,candidate,receipt);assert cb.checked_asset(asset) and len(paired_calls)==24
        s.bank.freeze(tmp_path/'qualified-frozen');frozen=Bank(tmp_path/'qualified-frozen',readonly=True)
        try:assert [a['id'] for a in frozen.all('skill')]==[asset['id']]
        finally:frozen.close()
    finally:s.close()
    assert before=={str(p):p.read_bytes() for p in files}


def test_exact_task_id_aliases_and_diagnostic_no_change():
    raw=sources()
    for i,row in enumerate(raw):row['task']['goal']+=' Distinct public source '+str(i)
    rows=cb.records(raw);refs=[{'case_id':r['task']['task_id'],'span_id':r['spans'][0]['span_id']} for r in rows]
    expected=sorted(r['task']['physical_key'] for r in rows)
    assert cb.source_refs(refs,rows)==expected
    cap=cb.validate_capability({'decision':'capability','required_conditions':'Compare explicit quantified claims.',
        'operations':['inspect_quantifiers'],'references':refs},rows)
    assert cap['source_case_ids']==expected
    value={'decision':'no_change','references':refs}
    assert cb.validate_capability(value,rows) is None
    assert cb.validate_candidate(value,cap,rows) is None
    for bad in ({**value,'operations':['inspect_quantifiers']},{**value,'guidance':'New procedure.'}):
        with pytest.raises(ValueError):cb.validate_capability(bad,rows)
    for case in ('invented','another-row','same-physical','ambiguous'):
        bad=deepcopy(refs);changed=deepcopy(rows)
        if case=='invented':bad[0]['case_id']='invented'
        elif case=='another-row':bad[0]['span_id']=rows[1]['spans'][0]['span_id']
        elif case=='same-physical':bad[1]={'case_id':rows[0]['task']['physical_key'],'span_id':rows[0]['spans'][0]['span_id']}
        else:changed[0]['task']['task_id']=changed[1]['task']['physical_key']
        with pytest.raises(ValueError):cb.validate_capability({'decision':'no_change','references':bad},changed)


def test_paid_induction_recovery_preserves_budget_sources_and_has_no_new_induction(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch)
    base,out,plan,env=fixture_source(tmp_path,monkeypatch);validate=cb.validate_capability
    monkeypatch.setattr(cb,'validate_capability',lambda *args:(_ for _ in ()).throw(ValueError('old interface rejection')))
    prior=runner.execute(out/'plan.json',env);assert prior['new_http']==2 and prior['new_tokens']==10
    monkeypatch.setattr(cb,'validate_capability',validate)
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    child=tmp_path/'recovered';new=runner.recover(out/'plan.json',child)
    assert runner.check(child/'plan.json')['status']=='passed' and len(calls)==2
    assert new['recovery']['remaining_tokens']==1799990 and new['recovery']['remaining_http']==78
    result=runner.execute(child/'plan.json',env)
    assert result['status']=='completed_selection' and result['new_http']==len(calls)-2
    assert sum(p['tools'][0]['function']['name']=='submit_capability' for p in calls)==2
    assert result['total_p2_http']==len(calls) and result['total_p2_tokens']==5*len(calls)
    assert result['inherited_http']==2 and result['cumulative_http']==len(calls)
    assert result['new_tokens']==5*(len(calls)-2) and result['inherited_tokens']==10
    ledger=runner.read(child/'budget.json');old=runner.read(out/'budget.json')
    assert all(ledger['attempts'][k]==v for k,v in old['attempts'].items())
    assert before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    assert all(runner.read(child/'learning'/str(i)/'learning.json')['recovered_demand']['original_code']==plan['code'] for i in range(2))
    with pytest.raises(ValueError,match='already recovered'):runner.recover(child/'plan.json',tmp_path/'third')
    ledger['attempts'][next(iter(old['attempts']))]['accounted_tokens']=0;runner.write_json(child/'budget.json',ledger)
    with pytest.raises(ValueError,match='Inherited paid attempts'):runner.check(child/'plan.json')


@pytest.mark.parametrize('damage',['unknown_billing','wire_source','response','cost','exhausted'])
def test_induction_recovery_rejects_changed_or_unfunded_paid_records(tmp_path,monkeypatch,damage):
    calls=fixture_transport(monkeypatch);base,out,plan,env=fixture_source(tmp_path,monkeypatch);validate=cb.validate_capability
    monkeypatch.setattr(cb,'validate_capability',lambda *args:(_ for _ in ()).throw(ValueError('old interface rejection')))
    runner.execute(out/'plan.json',env);monkeypatch.setattr(cb,'validate_capability',validate)
    if damage in {'unknown_billing','cost','exhausted'}:
        ledger=runner.read(out/'budget.json')
        if damage=='unknown_billing':ledger['unknown_billing']=True
        elif damage=='cost':next(iter(ledger['attempts'].values()))['accounted_tokens']=0
        else:
            plan['new_limits']['request_limit']=2;ledger['limits']['request_limit']=2
            runner.write_json(out/'plan.json',plan);runner.write_json(out/'plan_identity.json',{'plan_hash':digest(plan)})
            result=runner.read(out/'result.json');result['plan_hash']=digest(plan);runner.write_json(out/'result.json',result)
        runner.write_json(out/'budget.json',ledger)
    else:
        audit=runner.read(out/'learning/0/requests.json');request=audit['requests'][0]
        if damage=='wire_source':request['http_attempts'][0]['final_payload_audit']['messages'][1]['content']='{}'
        else:request['response']['tool_calls'][0]['arguments']={'decision':'no_change'}
        runner.write_json(out/'learning/0/requests.json',audit)
    with pytest.raises(ValueError):runner.recover(out/'plan.json',tmp_path/'child')
    assert len(calls)==2 and not (tmp_path/'child').exists()


def test_recovery_budget_admission_counts_paid_prefix_without_refill(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch);base,out,plan,env=fixture_source(tmp_path,monkeypatch);validate=cb.validate_capability
    monkeypatch.setattr(cb,'validate_capability',lambda *args:(_ for _ in ()).throw(ValueError('old interface rejection')))
    runner.execute(out/'plan.json',env);monkeypatch.setattr(cb,'validate_capability',validate)
    plan['new_limits']['request_limit']=3;runner.write_json(out/'plan.json',plan);runner.write_json(out/'plan_identity.json',{'plan_hash':digest(plan)})
    ledger=runner.read(out/'budget.json');ledger['limits']['request_limit']=3;runner.write_json(out/'budget.json',ledger)
    result=runner.read(out/'result.json');result['plan_hash']=digest(plan);runner.write_json(out/'result.json',result)
    child=tmp_path/'child';runner.recover(out/'plan.json',child);result=runner.execute(child/'plan.json',env)
    assert result['status']=='stopped_budget_censored' and result['total_p2_http']==3 and result['new_http']==1 and len(calls)==3
    assert not (child/'frozen_bank').exists()
    runner.execute(child/'plan.json',env);assert len(calls)==3


def test_actual_paid_induction_responses_through_fixed_compiler(tmp_path,monkeypatch):
    previous=os.environ.get('LIVEMATH_P2_REJECTED_RUN')
    if not previous:pytest.skip('Set the immutable real P2 induction-only run')
    previous=Path(previous);before={str(p):p.read_bytes() for p in previous.rglob('*') if p.is_file()}
    recovered,ledger,inputs=runner.recovery_inputs(previous/'plan.json')
    assert recovered['prior_p2_tokens']==9800 and recovered['prior_p2_http']==2
    assert recovered['remaining_tokens']==1790200 and recovered['remaining_http']==78
    config=runner.read(previous/'config.json');cfg=runner.configuration(config,tmp_path/'derived');calls=fixture_transport(monkeypatch)
    system=EmpiricalSystem(cfg)
    try:
        for row in runner.read(previous/'source_cases.json'):system.bank.save_case(PublicTask(**row['task']),row['experience'])
        plan=runner.read(previous/'plan.json');selected=[t['physical_key'] for stage in plan['selection_tasks'].values() for t in stage]
        logs=[]
        for i,unit in enumerate(plan['candidate_units']):
            runner.learning_boundary(system,tmp_path/'learning'/str(i),'candidate:'+str(i))
            logs.append(system.learner.learn_capability_batch(unit,selected,recovered_demand=recovered['demands'][i]))
        assert [r['decision'] for r in logs]==['candidate','no_change'],logs
        assert [p['tools'][0]['function']['name'] for p in calls]==['submit_learning','submit_capability_check']
        assert not system.bank.all('skill')  # Still unqualified: no automatic publication.
        assert logs[0]['candidate']['source_case_ids']==sorted(plan['candidate_units'][0])
    finally:system.close()
    assert before=={str(p):p.read_bytes() for p in previous.rglob('*') if p.is_file()}
