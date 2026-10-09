"""Focused CF4-R3 production regressions; intercepted HTTP, no model requests."""
from copy import deepcopy
import datetime
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from atomic_skillgraph.core.errors import AtomicSkillGraphError, BudgetExhausted
from atomic_skillgraph.empirical import LEARNING_MATERIAL_VERSION
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema, validate_schema_instance
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.model_view import pack_material, expand_material, canonical_bytes
from atomic_skillgraph.empirical.prompts import GUIDANCE_LEARNING, REF, STEP, PLAN
from atomic_skillgraph.empirical.trial_snapshot import seal_trial_workspace, validate_finished_execution
from atomic_skillgraph.harness.benchmarks import SpreadsheetAdapter
from atomic_skillgraph.harness.simple_protocol import UnknownSideEffect
from atomic_skillgraph.harness.scorers.spreadsheet import (
    EvaluatorContractError, _parse_range, _generate_cell_names, _compare_cell_value, compare_workbooks)
from skillcompiler_bench_contracts.livemath import normalize_livemath_item, permute_livemath_choices
from test_cf2_contracts import office, http, response
from test_empirical import config_for, program


def test_scoring_recovery_after_score_is_zero_llm_and_preserves_source(tmp_path,monkeypatch):
    from atomic_skillgraph.experiments.recover_finished_execution import recover
    from atomic_skillgraph.experiments.formal_log import tree_identity
    from atomic_skillgraph.experiments.run_empirical import write_json
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    source=tmp_path/'parent'; train=source/'train'; train.mkdir(parents=True)
    s=office(tmp_path); task=PublicTask('office','office-physical','find target')
    config=deepcopy(s.config); config['data_dir']=str(train/'bank')
    config['experiment']['output_dir']=str(train)
    config['harness']['evaluator_records']=str(tmp_path/'records.json')
    write_json(tmp_path/'records.json', {'office':{'answer':'42'}})
    cp=TaskCheckpoint(train/'checkpoints/office/1')
    receipt=seal_trial_workspace(s.adapter,cp.root/'executor_finished_workspace')
    cp.advance('task_started', executor_finished={'reason':'finish','prediction':'42','attempts':[]},
        executor_finished_workspace=receipt, executor_state={'pending_step':None,'pending_decision_id':None},
        trace={'task':asdict(task),'score':None,'execution':{},'learning_status':'not_started'})
    s.close()
    bank=Bank(train/'bank'); bank.close()
    write_json(train/'config.json',config)
    write_json(train/'execution_manifest.json',{'config':config,'code':{'git_sha':'parent'},'tasks':[asdict(task)]})
    write_json(source/'run_manifest.json',{'run_id':'parent','identity':{'method':'ours'}})
    audit={'requests':[{'id':'paid'}], 'usage':[{'event_id':'usage','total_tokens':123}]}
    write_json(train/'requests/office_attempt1.json',audit)
    db=sqlite3.connect(train/'run.sqlite3')
    db.execute('CREATE TABLE tasks(id TEXT, attempts INTEGER, status TEXT, result TEXT)')
    db.execute('INSERT INTO tasks VALUES(?,1,?,NULL)',('office','running')); db.commit(); db.close()
    before=tree_identity(source); scored=[]
    original=OfficeAdapter.evaluate
    def count(self,sealed):
        scored.append(True); result=original(self,sealed)
        self.score_audit['fixture_receipts']=str(self.evaluation_receipts)
        return result
    monkeypatch.setattr(OfficeAdapter,'evaluate',count)
    monkeypatch.setattr(EmpiricalSystem,'agent',lambda *a,**k:pytest.fail('Recovery must not call a model'))
    output=tmp_path/'child'
    with pytest.raises(InterruptedError): recover(source,output,'office',interrupt_after='score')
    result=recover(source,output,'office')
    assert len(scored)==1 and result['score']['hard'] and result['learning_status']=='deferred'
    assert result['new_model_calls']==0 and result['inherited_tokens']==123 and not result['task_committed']
    assert recover(source,output,'office')==result and tree_identity(source)==before
    child=TaskCheckpoint(output/'train/checkpoints/office/1')
    assert child.state['stage']=='task_execution_finished' and child.state['learning_workspace']
    assert json.loads((output/'train/requests/office_attempt1.json').read_text())==audit
    assert json.loads((child.root/'recovery_evaluation.json').read_text())['sealed']==child.state['trace']['episode_result']['sealed_prediction']
    assert result['scoring_audit']['fixture_receipts']==str(child.root/'evaluation_receipts')


def test_choices_are_identity_based_immutable_and_idempotent():
    source = normalize_livemath_item({'id': '202602:28', 'question': 'question',
        'choices': {'B': 'same', 'C': 'different'}, 'correct_choice': {'label': 'A', 'text': 'same'}})
    before = deepcopy(source)
    shuffled = permute_livemath_choices(source, choice_seed=42, stable_item_id=source['id'])
    assert source == before and sorted(c['text'] for c in shuffled['choices']) == ['different', 'same', 'same']
    assert shuffled == permute_livemath_choices(shuffled, choice_seed=42, stable_item_id=source['id'])
    assert permute_livemath_choices(shuffled, choice_seed=43, stable_item_id=source['id']) == permute_livemath_choices(source, choice_seed=43, stable_item_id=source['id'])
    changed = deepcopy(shuffled); changed['choices'][0]['text'] = 'changed'
    with pytest.raises(ValueError, match='Projected'): permute_livemath_choices(changed, choice_seed=42, stable_item_id=source['id'])
    with pytest.raises(ValueError, match='identity'): permute_livemath_choices(source, choice_seed=42, stable_item_id='other')


def experience(value, *, hard=False):
    return {'task': {'goal': 'find target', 'inputs': {}}, 'submission': 'B', 'score': {'hard': hard},
        'program_calls': [], 'events': [{'name': 'read', 'arguments': {'path': 'same'}, 'backend_invoked': True,
                                      'result': {'accepted': True, 'data': value}}]}


def test_case_result_sources_are_disjoint_and_conflicts_fail(tmp_path):
    s = office(tmp_path)
    try:
        current = s.task_context.register('native:0', {'data': 'current'})
        views = [s.learner._view(experience(text), source_id=case) for case, text in
                 [('first', 'one'), ('second', 'two'), ('first', 'new snapshot')]]
        ids = [v['events'][0]['result']['result_id'] for v in views]
        assert len(set([current, *ids])) == 4
        assert [s.task_context.results[r]['data'] for r in ids] == ['one', 'two', 'new snapshot']
        assert s.learner._view(experience('one'), source_id='first')['events'][0]['result']['result_id'] == ids[0]
        key = next(k for k,v in s.task_context.sources.items() if v == ids[0])
        with pytest.raises(AtomicSkillGraphError) as failure:
            s.task_context.register(key, {'data': 'conflicting'}, verify_source=True)
        assert failure.value.code == 'learner_case_source_conflict'
    finally: s.close()


def test_legacy_paid_learning_is_not_reused_under_new_material(tmp_path):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path/'cp')
    try:
        s.checkpoint.advance('task_execution_finished', learning_proposal={'decision': 'no_change'})
        s.agent = lambda *a,**k: pytest.fail('Paid legacy response must be deferred')
        with pytest.raises(ValueError, match='Legacy learning response'):
            s.learner._receive('learning_proposal', 'extractor', 'prompt', {}, 'submit_learning', object_schema())
    finally: s.close()


def test_unpaid_legacy_material_has_a_separate_new_cache_key(tmp_path):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path/'cp')
    s.checkpoint.advance('task_execution_finished', learning_proposal_request_material={'old': True})
    sent = []
    def answer(*args, **kwargs): sent.append(args[2]); return {'decision': 'no_change'}
    s.agent = answer
    try:
        s.learner._receive('learning_proposal', 'extractor', 'prompt', {'new': True}, 'submit_learning', object_schema())
        assert sent[0]['new'] and sent[0]['learning_material_version'] == LEARNING_MATERIAL_VERSION
        assert s.checkpoint.state['learning_proposal_request_material'] == {'old': True}
    finally: s.close()


@pytest.mark.parametrize('range_', ['', 'G:A', 'B3:A1', 'A:', ':G', 'A1:G', 'A0', 'XFE1', 'A1048577', 'Ａ:G', '1:4', 'A'])
def test_invalid_evaluator_ranges_are_host_errors(range_):
    with pytest.raises(EvaluatorContractError): _parse_range(range_)


def workbook(path, values, *, sheet='Sheet3'):
    wb = openpyxl.Workbook(); wb.active.title = sheet
    for cell,value in values.items(): wb.active[cell] = value
    wb.save(path); wb.close()


def test_whole_columns_cover_prediction_extra_rows_and_quoted_sheets(tmp_path):
    gold, pred = tmp_path/'gold.xlsx', tmp_path/'pred.xlsx'
    workbook(gold, {'A1': 1, 'G2': 2}); workbook(pred, {'A1': 1, 'G2': 2})
    assert compare_workbooks(str(gold), str(pred), "Sheet3'!$A:$G")[0]
    workbook(pred, {'A1': 1, 'G2': 2, 'B4': 'extra'})
    assert not compare_workbooks(str(gold), str(pred), "'Sheet3'!A:G")[0]
    assert _parse_range('$A$1:$G$2') == ((1,1),(7,2))
    assert list(_generate_cell_names('A:B', row_limit=2)) == ['A1','A2','B1','B2']
    with pytest.raises(EvaluatorContractError): compare_workbooks(str(gold), str(pred), 'Missing!A1')
    with pytest.raises(EvaluatorContractError): compare_workbooks(str(gold), str(pred), '')
    assert not compare_workbooks(str(gold), str(tmp_path/'missing'), 'A1')[0]


def test_official_value_semantics_are_retained():
    assert _compare_cell_value('1.234', 1.23) and _compare_cell_value('', None)
    assert not _compare_cell_value(1.235, 1.23)
    assert _compare_cell_value(datetime.datetime(2026,10,9), 46304)
    assert _compare_cell_value(datetime.time(1,2,3), datetime.time(1,2,3))


def test_case_scoring_receipts_are_idempotent_and_hash_checked(tmp_path, monkeypatch):
    gold, inp = tmp_path/'gold.xlsx', tmp_path/'input.xlsx'
    workbook(gold, {'A1': 1}); workbook(inp, {})
    config = config_for(tmp_path/'bank')
    adapter = SpreadsheetAdapter({'t': {'cases': [{'input': str(inp), 'gold': str(gold)}],
        'instruction_type': 'Cell-Level', 'answer_position': 'A:A', 'public_files': {'input.xlsx': str(inp)}}}, config)
    adapter.reset(PublicTask('t', 'physical', 'do it'))
    stage = adapter.workspace.stage()
    (stage/'solution.py').write_text('literal fixture; never execute')
    workbook(stage/'case1_result.xlsx', {'A1': 1})
    adapter.workspace.publish(stage, ['solution.py','case1_result.xlsx'])
    sealed = adapter.submit(None); calls = []
    def recalc(path): calls.append(str(path)); return SimpleNamespace(close=lambda: None), path
    monkeypatch.setattr(adapter, '_recalculate', recalc)
    adapter.evaluation_receipts = tmp_path/'receipts'
    try:
        first = adapter.evaluate(sealed)
        assert first['hard'] and adapter.evaluate(sealed) == first and len(calls) == 1
        predicted = next((tmp_path/'receipts').glob('*.xlsx')); predicted.write_bytes(b'corrupt')
        with pytest.raises(EvaluatorContractError, match='prediction changed'): adapter.evaluate(sealed)
        assert len(calls) == 1
    finally: adapter.close()


def test_sealed_completion_rejects_inflight_and_corruption(tmp_path):
    s = office(tmp_path); cp = TaskCheckpoint(tmp_path/'cp')
    try:
        stage=s.adapter.workspace.stage(); (stage/'a.txt').write_text('public')
        s.adapter.workspace.publish(stage, ['a.txt'])
        receipt=seal_trial_workspace(s.adapter, cp.root/'executor_finished_workspace')
        cp.advance('task_started', executor_finished={'reason':'done','attempts':[]}, executor_finished_workspace=receipt,
                   executor_state={'pending_step':None, 'pending_decision_id':None})
        assert validate_finished_execution(cp)[0]['reason'] == 'done'
        cp.native_events([{'state':'started'}])
        with pytest.raises(UnknownSideEffect): validate_finished_execution(cp)
        cp.native_events([])
        (Path(receipt['root'])/'manifest.json').write_text('{}')
        with pytest.raises(ValueError, match='hash'): validate_finished_execution(cp)
    finally: s.close()


@pytest.mark.parametrize('hard,learn', [(True,True),(False,True),(True,False),(False,False)])
def test_role_budget_end_scores_once_and_obeys_learning_boundary(tmp_path, hard, learn):
    s=office(tmp_path); s.checkpoint=TaskCheckpoint(tmp_path/'cp'); scores=[]; learning=[]
    def exhausted(*a): raise BudgetExhausted('empirical_token_budget_exhausted', 'role budget')
    s.planner.plan=lambda *a: {'goal':'find','nodes':[]}; s.executor.run=exhausted
    s.adapter.evaluate=lambda sealed: (scores.append(sealed) or {'hard':hard,'raw_score':float(hard)})
    s.learner.learn=lambda *a: (learning.append(True) or {'decision':'no_change'})
    try:
        result=s.run_task(s.adapter.task, learn=learn)
        assert len(scores)==1 and len(learning)==int(learn) and result['score']['hard']==hard
        assert result['execution']['reason']=='token_budget_exhausted'
        assert s.checkpoint.state['stage']==('learning_finished' if learn else 'task_execution_finished')
        if learn: assert s.checkpoint.state['learning_workspace']
    finally: s.close()


def test_global_diagnostic_budget_does_not_score_or_start_learning(tmp_path):
    s=office(tmp_path)
    def exhausted(*a): raise BudgetExhausted('diagnostic_budget_exhausted', 'global budget')
    s.planner.plan=exhausted
    s.adapter.evaluate=lambda *a: pytest.fail('Global limit cannot enter completion')
    s.learner.learn=lambda *a: pytest.fail('Global limit cannot trigger learning')
    try:
        with pytest.raises(BudgetExhausted): s.run_task(s.adapter.task)
    finally: s.close()


def test_material_tables_roundtrip_contracts_jobs_and_permissions():
    contract={'input_schema':object_schema({'q':{'type':'string'}},['q']), 'output_schema':object_schema(),
              'result_role':'intermediate', 'entry_constraints':'x'*512}
    job={'id':'j','skill_id':'s','case_bindings':[],'state':'waiting_example','generation_count':0}
    material={'related':[{'id':'s'+str(i), **deepcopy(contract), 'current_program':{'id':'p'+str(i), **deepcopy(contract),
               'allowed_tools':['read'], 'state':'candidate'}, 'current_job':deepcopy(job), 'pending':[deepcopy(job)]} for i in range(4)]}
    before=deepcopy(material); packed=pack_material(material)
    assert expand_material(packed)==material==before and canonical_bytes(packed)<canonical_bytes(material)


def test_role_examples_use_real_schemas_and_guidance_limits():
    for ref in ({'task':'x'}, {'from':'n','field':'x'}, {'literal':3}, {'unresolved':'unknown'}): validate_schema_instance(ref,REF)
    with pytest.raises(ValueError): validate_schema_instance('x',REF)
    validate_schema_instance({'action':'call_tool','name':'read_result','arguments':{'result_id':'actual','offset':0}},STEP)
    validate_schema_instance({'mode':'compose','workflow':{'goal':'x','nodes':[{'id':'n','execution_mode':'dynamic','goal':'x','args':{'q':{'task':'q'}}}]}},PLAN)
    for field, limit in [('goal',256),('guidance',1600),('rationale',512)]:
        good={'decision':'upsert_guidance','guidance_skill':{'goal':'goal','guidance':'guidance'}, 'rationale':'why'}
        if field=='rationale': good[field]='x'*(limit+1)
        else: good['guidance_skill'][field]='x'*(limit+1)
        with pytest.raises(ValueError): validate_schema_instance(good,GUIDANCE_LEARNING)


@pytest.mark.parametrize('hard,empty,truncated,eligible', [(False,False,False,False),(True,True,False,False),(True,False,True,False),(True,False,False,True)])
def test_guidance_host_gate_uses_evidence_not_score_truthiness(tmp_path,hard,empty,truncated,eligible):
    s=office(tmp_path)
    exp={**experience(''), 'events':[], 'empty_answer':empty, 'completion_truncated':truncated, 'score':{'hard':hard}}
    sent=[]
    s.learner._receive=lambda *a,**kw: (sent.append(kw) or {'decision':'no_change'})
    try:
        log=s.learner._learn_guidance(s.adapter.task,exp)
        assert len(sent)==int(eligible)
        assert log['evidence']['eligible']==eligible
        if not eligible: assert log['decision_origin']=='host' and log['reason']=='insufficient_reusable_evidence'
        else: assert sent[0]['decision_purpose']=='guidance_learning'
        assert s.learner._guidance_evidence(experience('real public output'))['eligible']
        failure={**experience(''), 'events':[{'backend_invoked':True,
            'result':{'accepted':False,'error':'Public missing-column check failed'}}]}
        assert s.learner._guidance_evidence(failure)['eligible']
    finally:s.close()


def test_guidance_purpose_reaches_actual_http_payload(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_learning')])
    s=office(tmp_path)
    try:
        result=s.learner._learn_guidance(s.adapter.task,{**experience(''), 'events':[], 'score':{'hard':True}})
        assert result['decision']=='no_change' and len(seen)==1
        payload=seen[0]
        assert payload['thinking']=={'type':'disabled'} and payload['max_tokens']==4096
        assert payload['reasoning_effort']=='high' and [t['function']['name'] for t in payload['tools']]==['submit_learning']
        assert s.requests[0]['purpose']=='guidance_learning' and s.usage.events[0].bucket=='extractor_e1'
    finally:s.close()


@pytest.mark.parametrize('count', [0,1,2])
def test_first_build_threshold_and_unbuilt_supplement_queue(tmp_path,count):
    s=office(tmp_path)
    skill=s.bank.put('skill',{'goal':'find target','guidance':'public','input_schema':object_schema(), 'output_schema':object_schema(),
        'result_role':'intermediate','execution_intent':'program_requested','entry_constraints':'public'})
    cases={str(i):(PublicTask(str(i),str(i),'find target'),experience('output')) for i in range(count)}
    bindings=[{'case_id':str(i),'inputs':{},'start_mode':'reset','prefix':[]} for i in range(count)]
    job={'id':'job','skill_id':skill['id'],'skill_version':skill['id'],'case_bindings':[], 'state':'waiting_example',
         'kind':'build','generation_count':0,'epoch':0,'program_id':None,'repair_used':False}
    request={'action':'build','case_bindings':bindings}
    try:
        s.learner._merge_request(job,request,skill,cases,experience('output'),True,{})
        assert job['state']==('ready' if count==2 else 'waiting_example')
        s.bank.save_job(job)
        if count==1:
            candidate=s.learner.related_candidates(PublicTask('new','new','find target'))[0]
            assert candidate['build_status']=='unbuilt_unverified' and candidate['current_program'] is None
        calls=[]
        def builder(*a,**k): calls.append(True); raise ValueError('fixture stops before execution')
        s.learner._receive=builder
        s.learner._realize(job,PublicTask('new','new','find target'),cases,[],{'errors':[]})
        assert bool(calls)==(count==2)
    finally:s.close()


def test_existing_candidate_trials_second_binding_without_rebuilding(tmp_path):
    s=office(tmp_path)
    skill=s.bank.put('skill',{'goal':'find','guidance':'g','input_schema':object_schema(),'output_schema':object_schema(),
        'result_role':'intermediate','execution_intent':'program_requested','entry_constraints':'public'})
    p=s.bank.put('program',{**program('def run(ctx, inputs):\n return {}'),'allowed_tools':[], 'entry_constraints':'public'})
    s.bank.put('implementation',{'skill_id':skill['id'],'program_id':p['id']})
    cases={str(i):(PublicTask(str(i),str(i),'find'),experience('output')) for i in range(2)}
    bindings=[{'case_id':str(i),'inputs':{},'start_mode':'reset','prefix':[]} for i in range(2)]
    job={'id':'j','skill_id':skill['id'],'skill_version':skill['id'],'case_bindings':bindings,'state':'ready','kind':'build',
         'generation_count':1,'epoch':0,'program_id':p['id'],'repair_used':False}
    s.learner._receive=lambda *a,**kw: pytest.fail('Existing candidate must be trialed first')
    calls=[]
    def trial(prog,inputs,task,**kwargs):
        calls.append(task.physical_key)
        record={'id':task.physical_key,'program_id':prog['id'],'task_key':task.physical_key,'origin':'train_test',
                'outcome':'positive','basis':'local_check','result':{'status':'ok'}}
        s.bank.record(record); return record
    s.test_program=trial
    try:
        s.learner._realize(job,cases['1'][0],cases,[],{'errors':[],'tests':[]})
        assert calls==['0','1'] and s.bank.get(p['id'])['state']=='usable'
    finally:s.close()
