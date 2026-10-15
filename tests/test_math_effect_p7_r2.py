"""Production R2 contracts, real historical replay and intercepted paid control flow."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import yaml

from atomic_skillgraph.empirical import math_effect_contract as c, math_effect_selection as select
from atomic_skillgraph.empirical import math_effect_optimizer as opt, math_effect_learner as learner
from atomic_skillgraph.empirical import math_effect_runtime as runtime, math_effect_source as source
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.experiments import run_math_effect_p7 as run
from test_cf2_contracts import response,http
from test_math_effect_p7_contract import proposal

CONTRACT=Path('configs/livemath_effect_p7_r2_seed42.yaml')
H=Path('/home/yangchengyu')
PARENT=H/'livemath_effect_p7_scope_recovery_seed42_20261015'
SOURCE=H/'livemath_p2_formal_seed42_20261013/livemath/seed42'
DATA=H/'main_experiment_v1_resources_cf4_r3_20261009_v1'


def value(ids=('source_one',),mode='all_tasks'):
    return {'category':'arbitrary_mathematical_operation','goal':'Compare the assumptions and strength of every conclusion.',
        'applicability':{'mode':mode,'require_all':[] if mode=='all_tasks' else ['comparable_conclusions'],'exclude_any':[]},
        'steps':['Verify the assumptions before accepting a stronger claim.'],
        'stop_condition':'If unresolved, leave this local rule and continue normal reasoning to a legal label.',
        'rule_basis':'reasoning_procedure','source_case_ids':list(ids),'edit_rationale':'A hypothesis from actual Train feedback.'}


def candidate(v=None):
    v=v or value();return c.candidate(v,v['source_case_ids'],{'response_hash':'fixture'},version=c.R2)


def cfg():
    return {'effect_p7':yaml.safe_load(CONTRACT.read_text()),'llm':{'base_url':'https://fixture.invalid',
        'model':'deepseek-v4-flash','api_key_env':'MODEL_API_KEY','runtime':{'max_completion_tokens':32768}}}


def test_r2_migration_and_siblings():
    old=proposal('global');old['applicability']['required_public_indicators']=['comparable_conclusions']
    new=c.migrate_candidate_r2(old,['s1','s2'],all_have_choices=True)
    assert new['applicability']=={'mode':'all_tasks','require_all':[],'exclude_any':[]}
    assert all(new[k]==old[k] for k in ('goal','steps','source_case_ids','stop_condition'))
    old['applicability']['required_public_indicators']=['quantifier_syntax']
    with pytest.raises(c.ContractError,match='migration_needs_model'):c.migrate_candidate_r2(old,['s1','s2'],all_have_choices=True)
    local=proposal('local');saved=deepcopy(local)
    assert c.migrate_candidate_r2(local,['s1','s2'],all_have_choices=True)['applicability']['mode']=='when'
    assert local==saved


@pytest.mark.parametrize('bad',[None,{},[{}],[None],['quantifier_syntax',{}]])
def test_invalid_condition_container_has_field_path(bad):
    v=value(mode='when');v['applicability']['require_all']=bad
    with pytest.raises(c.ContractError) as exc:c.validate_candidate_r2(v,['source_one'])
    assert '$.applicability.require_all' in str(exc.value)


def test_complete_edit_and_benchmark_prior():
    original=candidate();v=value(mode='when');v.update(goal='Recognize the public stronger-result meta option.',
        rule_basis='benchmark_prior',steps=['Prefer that public meta statement as an empirical prior, then verify assumptions.'])
    v['applicability']['require_all']=['has_stronger_meta']
    child=opt.apply_edit_r2({'parent_id':original['id'],'replacement':v,'rationale':'Train-only hypothesis'},
        {original['id']:original},['source_one'],{'response_hash':'edited'},cfg()['effect_p7']['tokenizer_path'])
    assert child['parent_id']==original['id'] and child['candidate']==v and child['edit_delta']['diff']['applicability']
    assert c.checked_candidate(child) and not c.checked_effect_asset(c.asset(child))
    assert c.checked_experimental_asset(c.asset(child))
    for field in ['replacement','parent_id']:
        bad={'parent_id':original['id'],'replacement':v,'rationale':'x'};bad[field]=None
        with pytest.raises(c.ContractError):opt.apply_edit_r2(bad,{original['id']:original},['source_one'],{},cfg()['effect_p7']['tokenizer_path'])


@pytest.mark.parametrize('label',['A','C','Z'])
def test_meta_only_looks_at_actual_choices(label):
    text='One of the remaining options is correct, but a stronger result can be proven.'
    task=PublicTask('t','k','A stronger theorem can be proved.',{'choices':[{'label':label,'text':text},
        {'label':'Q','text':'None of the above.'},{'label':'R','text':'All of the above.'}]},'train')
    assert select.stronger_meta(task)=={'has_stronger_meta':True,'matches':[{'label':label,'text':text}]}
    task.inputs['choices']=task.inputs['choices'][1:]
    assert not select.stronger_meta(task)['has_stronger_meta']
    task.inputs['choices'].append({'label':'T','text':'A stronger bound holds.'})
    assert not select.stronger_meta(task)['has_stronger_meta']


def test_bad_candidate_does_not_discard_valid_sibling(tmp_path,monkeypatch):
    a=value();b=value(mode='when');b['applicability']['require_all']=[{}]
    fixed=value(mode='when')
    calls=http(monkeypatch,[response([{'decision':'propose','candidates':[a,b]}],name='submit_effect'),
        response([{'decision':'repair','repairs':[{'index':1,'replacement':fixed}]}],name='submit_effect')])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=1000000,request_limit=16,finish_reserve=0)
    result=learner.learn(tmp_path/'learn',cfg(),g,{'source_materials':[{'case_id':'source_one','public_task':{'goal':'g','choices':[]}}]},['source_one'])
    assert len(result['candidates'])==2 and result['structural_repairs']==1 and len(calls)==2
    assert result['candidates'][0]['candidate']==a
    assert learner.learn(tmp_path/'learn',cfg(),g,{'source_materials':[{'case_id':'source_one','public_task':{'goal':'g','choices':[]}}]},['source_one'])==result
    assert len(calls)==2


def test_no_change_ignores_irrelevant_fields(tmp_path,monkeypatch):
    calls=http(monkeypatch,[response([{'decision':'no_change','edits':None,'candidates':{'wrong':'shape'}}],name='submit_effect')])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=1000000,request_limit=2,finish_reserve=0)
    result=learner.learn(tmp_path/'learn',cfg(),g,{},['source_one'])
    assert result['state']=='no_change' and result['structural_repairs']==0 and len(calls)==1


@pytest.mark.parametrize('bad',[None,[],{'parent_id':None,'replacement':{},'rationale':'x'}])
def test_editor_repairs_complete_invalid_entry_without_type_error(tmp_path,monkeypatch,bad):
    parent=candidate();fixed=value();fixed['goal']='Compare the exact assumptions before any stronger conclusion.'
    item={'parent_id':parent['id'],'replacement':fixed,'rationale':'Repair the malformed entry only.'}
    calls=http(monkeypatch,[response([{'decision':'edit','edits':[bad]}],name='submit_effect'),
        response([{'decision':'repair','repairs':[{'index':0,'replacement':item}]}],name='submit_effect')])
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=1000000,request_limit=2,finish_reserve=0)
    result=learner.learn(tmp_path/'learn',cfg(),g,{},['source_one'],parents={parent['id']:parent})
    assert len(calls)==2 and len(result['candidates'])==1 and result['structural_repairs']==1
    assert result['candidates'][0]['parent_id']==parent['id']


def test_three_candidate_roster_routes_complementary_locals():
    global_=candidate();a=value(mode='when');a['applicability']['require_all']=['quantifier_syntax']
    b=value(mode='when');b['applicability']['require_all']=['bound_syntax']
    roster=run.policy_r2([global_,candidate(a),candidate(b)])
    task=PublicTask('t','k','Find the bound.',{'choices':[{'label':'A','text':'An inequality holds.'},{'label':'B','text':'No bound.'}]},'train')
    selected=select.select_r2(roster,task,cfg()['effect_p7']['tokenizer_path'])
    assert len(roster)==3 and [v['id'] for v in selected['selected']]==[roster[0]['id'],roster[2]['id']]


@pytest.mark.skipif(not PARENT.exists(),reason='Actual completed P7 evidence required')
def test_real_pair_attribution_leaves_original_scores_unchanged():
    for phase,wanted in [('feedback',(1,3)),('gate',(7,8))]:
        old=run.read(PARENT/phase/'result.json');old=old['receipt'] if phase=='gate' else old['best']
        before=digest(old)
        rows=opt.pairs_r2([r['off'] for r in old['rows']],[r['on'] for r in old['rows']]);stats=opt.statistics_r2(rows)
        assert (stats['all']['off_correct'],stats['all']['on_correct'])==wanted
        if phase=='gate':
            assert stats['intervened']['wins']==2 and stats['intervened']['losses']==0
            assert stats['not_intervened']['losses']==1 and stats['not_intervened']['tasks']==11
        else:assert stats['not_intervened']['wins']==1
        assert digest(old)==before


@pytest.fixture
def prepared_r2(tmp_path,monkeypatch):
    if not SOURCE.exists():pytest.skip('Actual immutable Train60 fixture required')
    settings=yaml.safe_load(CONTRACT.read_text())
    ledgers=[]
    for i,path in enumerate(settings['ledger_sources']):
        target=tmp_path/('ledger_'+str(i)+'.json');run.write_json(target,run.read(path));ledgers.append(str(target))
    settings['ledger_sources']=ledgers
    contract=tmp_path/'r2.yaml';contract.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(run,'code_identity',lambda:{'git_sha':'offline_fixture','source_sha256':'offline','tracked_dirty':False})
    root=tmp_path/'run';run.prepare(root,contract,SOURCE,DATA)
    plan=run.read(root/'plan.json')
    # Explicit test-only receipt, no real authorization or real HTTP is fabricated.
    run.write_json(root/'paid_authorization.json',{'plan_hash':digest(plan),'human_instruction':'OFFLINE INTERCEPTED FIXTURE',
        'phases':settings['phase_limits'],'test_authorized':False})
    return root,plan


def fake_transport(monkeypatch,root,*,edit=False):
    calls=[];tasks=run.read(root/'tasks.json');lookup={}
    for split,items in tasks.items():
        gold=run.read(DATA/'livemath'/('evaluator_records_'+split+'.json'))
        lookup.update({t['goal']:(t,gold[t['task_id']]) for t in items})
    edit_calls=[]
    def post(url,**kwargs):
        wire=deepcopy(kwargs['json']);calls.append(wire);body=json.loads(wire['messages'][1]['content'])
        if wire.get('tools'):
            shown={r['case_id'] for r in body.get('source_materials',[])}
            assert shown=={t['physical_key'] for t in tasks['train']}
            assert 'correct_label' not in json.dumps(body) and 'correct_text' not in json.dumps(body)
            if 'paired_feedback' in body:
                assert len(body['paired_feedback'])==12 and body['groups']
                edit_calls.append(body)
                if edit:
                    parent=body['parents'][0];v=deepcopy(parent['candidate'])
                    v['goal']='Compare all hypotheses before ranking conclusions, round '+str(len(edit_calls))
                    answer={'decision':'edit','edits':[{'parent_id':parent['id'],'replacement':v,'rationale':'Observed intervention hypothesis'}]}
                else:answer={'decision':'no_change'}
            else:answer={'decision':'propose','candidates':[value([next(iter(shown))])]}
            raw=response([answer],name='submit_effect')
        else:
            assert wire['max_tokens']==65536 and wire['thinking']=={'type':'enabled'} and wire['reasoning_effort']=='high'
            t,record=lookup[body['goal']];wrong=next(c['label'] for c in record['choices'] if c['label']!=record['correct_choice']['label'])
            raw=response(content=wrong,finish='stop')  # Low accuracy must still complete Val.
        return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:raw)
    monkeypatch.setenv('MODEL_API_KEY','fixture');monkeypatch.setattr('requests.post',post)
    return calls,edit_calls


@pytest.mark.parametrize('edit',[False,True])
def test_full_r2_flow_freeze_low_score_and_resume(prepared_r2,monkeypatch,edit):
    root,plan=prepared_r2;calls,edits=fake_transport(monkeypatch,root,edit=edit)
    result=run.execute_r2(root,plan)
    assert result['phase_complete']=={'feedback':True,'val':True} and result['paid_startup_observed']
    assert result['progress']['delivery_ready'] and not result['accuracy_improved']
    assert len(edits)==2 and run.read(root/'feedback/result.json')['versions_evaluated']==(3 if edit else 1)
    frozen=run.verify_frozen(root);assert frozen['evaluation_status']=='experimental_not_claimed_effective'
    b=Bank(root/'frozen_bank',readonly=True)
    try:
        assets=b.all('skill');assert assets and all(c.checked_experimental_asset(a) for a in assets)
        assert all(a['state']=='candidate' and not c.checked_effect_asset(a) for a in assets)
    finally:b.close()
    assert len(list((root/'feedback/off').rglob('trace.json')))==12
    assert len(list((root/'val/off').rglob('trace.json')))==17
    assert not (root/'gate').exists() and not (root/'test').exists()
    before=len(calls);run.execute_r2(root,plan);assert len(calls)==before
    assert run.verify_frozen(root)==frozen
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    conf=run.read(root/'config.json');conf['data_dir']=str(root/'frozen_bank')
    system=EmpiricalSystem(conf,harness=run.AnswerAdapter('livemath',{}))
    assert system.readonly;system.close()
    before=(root/'progress.json').read_bytes();run.monitor_r2(root);assert (root/'progress.json').read_bytes()==before
    with pytest.raises(ValueError,match='not authorized'):run.run_eval(root,plan,'test')
    assert all(opt.verify_receipt_r2(run.read(p)) for p in (root/'feedback/receipts').glob('*.json'))
    # All 60 full public tasks are present; old 28/12/20 membership is identity-only.
    rows=run.read(root/'source_materials.json');assert len(rows)==60
    assert all(r['public_task']['goal'] and r['public_task']['choices'] for r in rows)
    assert all(r['process_status']=='recorded' or r['observed_process']==[] for r in rows)


def test_r2_real_wire_and_crash_before_trace(tmp_path,monkeypatch):
    from test_math_effect_p7_runtime import setup
    task,adapter,g=setup(tmp_path)
    raw=response(content='',finish='length');raw['choices'][0]['message']['reasoning_content']='unverified '*5000
    calls=http(monkeypatch,[raw,response(content='A',finish='stop')])
    real_save=runtime.save_json
    def crash(path,value):
        if Path(path).name=='trace.json':raise RuntimeError('injected crash after response commit')
        return real_save(path,value)
    monkeypatch.setattr(runtime,'save_json',crash)
    with pytest.raises(RuntimeError,match='injected crash'):runtime.solve(task,adapter,[],cfg(),tmp_path/'task',g)
    monkeypatch.setattr(runtime,'save_json',real_save)
    result=runtime.solve(task,adapter,[],cfg(),tmp_path/'task',g)
    assert len(calls)==2 and result['valid'] and not result['budget_censored']
    assert calls[0]['max_tokens']==65536 and calls[1]['max_tokens']==16384 and calls[1]['reasoning_effort']=='low'
    assert runtime.provider(cfg()['llm'],'runtime',cfg()['effect_p7']).config.request_timeout_seconds==600
    assert all(x['context']['parent_token_limit']==128000 for x in g.state['attempts'].values())
    assert result['main_wire_hash']==digest(run.read(tmp_path/'task/main/request.json')['payload'])


def test_real_sources_preserved_across_byte_batches():
    rows=[{'case_id':str(i),'public_task':{'goal':'完整题干'*30,'choices':[{'label':'Q','text':'完整选项'*50}]},
           'observed_process':[]} for i in range(8)]
    batches=source.material_batches_r2(rows,6000)
    assert len(batches)>1 and [r for b in batches for r in b]==rows


def test_zero_candidate_is_an_honest_terminal_result(prepared_r2,monkeypatch):
    root,plan=prepared_r2
    legacy=run.read(root/'legacy_learning.json');legacy['proposal']={'decision':'no_change'}
    run.write_json(root/'legacy_learning.json',legacy)
    calls=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_effect')])
    result=run.execute_r2(root,plan)
    assert run.read(root/'learning/result.json')['status']=='no_candidate'
    assert not result['paid_startup_observed'] and len(calls)==1
    assert not (root/'frozen_bank').exists() and not (root/'val').exists()


def test_committed_received_response_survives_crash_before_request_file(tmp_path,monkeypatch):
    from test_math_effect_p7_runtime import setup
    task,adapter,g=setup(tmp_path);calls=http(monkeypatch,[response(content='A',finish='stop')])
    real_save=runtime.save_json
    def crash(path,value):
        if Path(path).name=='requests.json':raise RuntimeError('crash after received journal')
        return real_save(path,value)
    monkeypatch.setattr(runtime,'save_json',crash)
    with pytest.raises(RuntimeError,match='received journal'):runtime.solve(task,adapter,[],cfg(),tmp_path/'task',g)
    assert (tmp_path/'task/main/received.json').exists()
    monkeypatch.setattr(runtime,'save_json',real_save)
    result=runtime.solve(task,adapter,[],cfg(),tmp_path/'task',g)
    assert result['valid'] and len(calls)==1 and result['http']==1


def test_r2_phase_refusal_keeps_metered_main_and_never_resends(tmp_path,monkeypatch):
    from test_math_effect_p7_runtime import setup
    from atomic_skillgraph.core.errors import BudgetExhausted
    task,adapter,_=setup(tmp_path)
    raw=response(content='',finish='length');raw['usage']={'prompt_tokens':1000,'completion_tokens':59000,'total_tokens':60000}
    calls=http(monkeypatch,[raw]);g=BudgetGovernor(tmp_path/'bounded.json',token_limit=70000,request_limit=2,finish_reserve=0)
    for _ in range(2):
        with pytest.raises(BudgetExhausted,match='admission refused'):runtime.solve(task,adapter,[],cfg(),tmp_path/'task',g)
    assert len(calls)==1 and (tmp_path/'task/main/response.json').exists()
    assert len(g.state['attempts'])==1 and next(iter(g.state['attempts'].values()))['accounted_tokens']==60000


def test_r2_unknown_send_is_not_replayed(tmp_path,monkeypatch):
    from test_math_effect_p7_runtime import setup
    task,adapter,g=setup(tmp_path);calls=http(monkeypatch,[])
    run.write_json(tmp_path/'task/main/started.json',{'identity':'interrupted'})
    with pytest.raises(RuntimeError,match='no blind'):runtime.solve(task,adapter,[],cfg(),tmp_path/'task',g)
    assert not calls
