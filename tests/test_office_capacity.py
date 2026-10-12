"""Capacity v6: saved bound proposal, real Host checks and engineering-only receipts."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3

import pytest

from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor,input_token_bound
from atomic_skillgraph.empirical.contracts import PublicTask,digest
from atomic_skillgraph.empirical.model_view import expand_material,learning_case,learning_capacity_audit
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.prompts import LEARNING,LEARNER_PROMPT
from atomic_skillgraph.empirical.system import validate_config
from atomic_skillgraph.experiments import audit_officeqa_prefix as gate
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.experiments.canonical_manifest import ordered_train,sha256
from atomic_skillgraph.experiments.run_empirical import write_json,capacity_statistics
from test_cf2_contracts import office,http,response
from test_cf2_realization import requested_proposal,learning_trace,model_bindings
from test_office_r1 import record


@pytest.fixture
def bound_archive():
    root=os.environ.get('OFFICE_CAPACITY_ARCHIVE')
    if not root:pytest.skip('Set the immutable v5 six-task prefix for bound replay')
    root=Path(root)
    traces=[json.loads(p.read_text()) for p in (root/'train/traces').glob('*.json')]
    trace=next(t for t in traces if any(r.get('material_refusal',{}).get('wire_bytes')==9415 for r in t['requests']))
    return root,trace


def test_real_9415_byte_binding_enters_original_host_repair(bound_archive,tmp_path,monkeypatch):
    root,trace=bound_archive
    initial=next(r for r in trace['requests'] if r['stage']=='extractor' and r['repair']==0)
    refused=next(r for r in trace['requests'] if r.get('material_refusal'))
    before=deepcopy(refused['material_refusal']['payload']);proposal=initial['response']['tool_calls'][0]['arguments']
    s=office(tmp_path);s.config=validate_config(benchmark_config(s.config,'officeqa'))
    task=PublicTask(**trace['task']);s.adapter._records[task.task_id]={'answer':'fixture'};s.adapter.reset(task)
    db=sqlite3.connect((root/'train/bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    cases={r['task']['physical_key']:(PublicTask(**r['task']),r['experience'])
        for r in (json.loads(x[0]) for x in db.execute('SELECT payload FROM train_cases'))};db.close()
    sent=http(monkeypatch,[response([proposal],name='submit_learning'),response([{'decision':'no_change'}],name='submit_learning')])
    try:
        with pytest.raises(ValueError):s.learner.validate_learning_proposal(proposal,cases)
        result=s.agent('extractor',LEARNER_PROMPT,expand_material(json.loads(initial['messages'][1]['content'])),
            'submit_learning',initial['tools'][0]['function']['parameters'],repair_limit=1,
            validator=lambda p:s.learner.validate_learning_proposal(p,cases))
        assert result=={'decision':'no_change'} and len(sent)==2
        assert {k:v for k,v in sent[1].items() if k!='messages'} == {k:v for k,v in before.items() if k!='messages'}
        current=json.loads(sent[1]['messages'][1]['content']);original=json.loads(before['messages'][1]['content'])
        assert {k:v for k,v in current.items() if k not in {'errors','legal_operations'}} == {k:v for k,v in original.items() if k not in {'errors','legal_operations'}}
        assert len(current['errors'])==len(original['errors'])
        for new,old in zip(current['errors'],original['errors']):
            assert {k:v for k,v in new.items() if k!='errors'} == {k:v for k,v in old.items() if k!='errors'}
            assert len(new['errors'])==len(old['errors'])
            assert all(all(error[k]==v for k,v in reference.items()) for error,reference in zip(new['errors'],old['errors']))
        assert len(current['legal_operations'])==len(original['legal_operations'])
        for new,old in zip(current['legal_operations'],original['legal_operations']):
            assert {k:v for k,v in new.items() if k!='input_sources'} == {k:v for k,v in old.items() if k!='input_sources'}
            assert all(ref in new['input_sources'] for ref in old['input_sources'])
        assert input_token_bound(before)==9415 and refused['material_refusal']['payload']==before
        assert input_token_bound(sent[1])<=s.config['learning']['structural_repair_payload_max_bytes']==16384
        assert sent[1]['max_tokens']==2048 and sent[1]['thinking']['type']=='disabled'
        assert s.requests[1]['repair']==1 and not s.bank.all('program')
        assert proposal==initial['response']['tool_calls'][0]['arguments']
        record('bound_9415_capacity_v6',{'wire_bytes':input_token_bound(sent[1]),'original_wire_bytes':9415,'limit_bytes':16384,'actual_http':0,
            'fixture_responses':2,'payload':sent[1],'host_rejects_original':True,'repair_result':result})
    finally:s.close()


@pytest.mark.parametrize('count',[0,1,2])
def test_zero_one_two_bindings_preserved_through_single_repair(tmp_path,monkeypatch,count):
    s=office(tmp_path);s.config=validate_config(benchmark_config(s.config,'officeqa'))
    task=s.adapter.task;tr=learning_trace(s);experience=s.learner._experience(task,tr)
    cases={task.physical_key:(task,experience)};cards=[learning_case(experience,task.physical_key)['local_evidence']]
    proposal=requested_proposal()
    proposal['skill'].pop('guidance')
    if count==0:proposal['realization_request'].update(action='defer',case_bindings=[])
    if count==2:
        other=PublicTask('other','other-physical',task.goal,task.inputs,'train')
        s.adapter._records[other.task_id]={'answer':'fixture'};s.adapter.reset(other)
        exp=s.learner._experience(other,learning_trace(s));cases[other.physical_key]=(other,exp)
        cards.append(learning_case(exp,other.physical_key)['local_evidence'])
        proposal['realization_request']['case_bindings']+=model_bindings(other.physical_key)
    proposal['workflow']={'goal':'Read the actual source','interface_version':'empirical.workflow.v2',
        'nodes':[{'id':'read','execution_mode':'skill','skill_id':'$new'}],
        'outputs':{'text':{'from':'read','field':'text'}}}
    valid=deepcopy(proposal);valid['workflow']['nodes'][0]['args']={}
    sent=http(monkeypatch,[response([proposal],name='submit_learning'),response([valid],name='submit_learning')])
    try:
        s.learner.validate_learning_proposal(valid,cases)
        s.agent('extractor',LEARNER_PROMPT,{'experience':learning_case(experience,task.physical_key),
            'completed_train_cases':[learning_case(e,k) for k,(t,e) in cases.items() if k!=task.physical_key]},
            'submit_learning',LEARNING,repair_limit=1,validator=lambda p:s.learner.validate_learning_proposal(p,cases))
        body=json.loads(sent[1]['messages'][1]['content'])
        assert len(sent)==2 and body['proposal']==proposal and len(body['proposal']['realization_request']['case_bindings'])==count
        assert input_token_bound(sent[1])<=16384 and sent[1]['max_tokens']==2048
        if count:assert all('output_references' in card and 'input_sources' in card for card in body['legal_operations'])
        bad=deepcopy(valid)
        if count:
            bad['realization_request']['case_bindings'][0]['reference_fields']['text']=['data','invented']
            with pytest.raises(ValueError):s.learner.validate_learning_proposal(bad,cases)
        record('binding_count_'+str(count),{'payload':sent[1],'wire_bytes':input_token_bound(sent[1]),
            'actual_http':0,'binding_count':count,'wrong_source_rejected':bool(count)})
    finally:s.close()


def test_capacity_does_not_override_parent_or_learning_pool(bound_archive,tmp_path,monkeypatch):
    _,trace=bound_archive;request=next(r for r in trace['requests'] if r.get('material_refusal'))
    s=office(tmp_path);s.config=validate_config(benchmark_config(s.config,'officeqa'))
    s.budget_governor=BudgetGovernor(tmp_path/'budget.json',token_limit=100000,request_limit=5,finish_reserve=0)
    s.config['budget']={'train_solve_tokens':120000,'train_learning_tokens':8000}
    s.phase='train';s._learning_start=0
    s.request_attribution={'parent_scope':'bounded-parent','parent_token_limit':8000,'train_parent':True}
    sent=http(monkeypatch,[])
    try:
        with pytest.raises(BudgetExhausted):s.agent('extractor',request['messages'][0]['content'],
            json.loads(request['messages'][1]['content']),'submit_learning',request['tools'][0]['function']['parameters'],
            repair_limit=0,decision_purpose='learning_structure_repair')
        assert not sent and not s.budget_governor.state['attempts']
    finally:s.close()


@pytest.fixture
def prefix(tmp_path,monkeypatch):
    root=tmp_path/'run';train=root/'train';train.mkdir(parents=True)
    s=office(train);s.config=validate_config(benchmark_config(s.config,'officeqa'))
    s.config['experiment']['output_dir']=str(train)
    s.config['budget']={'token_limit':1000000,'request_limit':100,'finish_reserve':0,
        'train_task_tokens':200000,'train_solve_tokens':120000,'train_learning_tokens':80000,'eval_task_tokens':96000}
    s.budget_governor=BudgetGovernor(root/'budget.json',token_limit=1000000,request_limit=100,finish_reserve=0)
    tasks=[PublicTask('officeqa:fixture'+str(i),'physical'+str(i),'Read the source.',{'corpus':'offline'},'train') for i in range(6)]
    s.adapter._records.update({t.task_id:{'answer':'999'} for t in tasks})
    authority=tmp_path/'train.json';write_json(authority,{'tasks':[asdict(t) for t in tasks]});s.config['manifest']=str(authority)
    s.planner.plan=lambda *a,**kw:dynamic(a[0])
    pending=[]
    for i in range(6):pending.extend([response([{'action':'call_tool','name':'read','arguments':{'path':'a.txt','offset':0}}]),
        response([{'action':'finish','answer':'0'}]),
        response([{'decision':'reuse_existing','existing_skill_id':'x'*20000} if i==0 else {'decision':'no_change'}],name='submit_learning')])
    http(monkeypatch,pending);rows=[]
    db=sqlite3.connect(train/'run.sqlite3');db.execute('CREATE TABLE tasks(id TEXT,attempts INTEGER,status TEXT,result TEXT)')
    for t in ordered_train(tasks,42):
        s.audit_path=str(train/'requests'/(t.task_id+'_attempt1.json'))
        tr=s.run_task(t,attempt_id=t.task_id+':1');rows.append(tr)
        db.execute('INSERT INTO tasks VALUES(?,?,?,?)',(t.task_id,1,'completed',json.dumps(tr)))
    db.commit();db.close()
    source={'git_sha':'fixture-commit','source_sha256':'fixture-source-hash','tracked_dirty':False}
    identity={'source_commit':source['git_sha'],'source_sha256':source['source_sha256'],'run_seed':42,'initial_bank':'empty independent Bank'}
    write_json(root/'run_manifest.json',{**identity,'identity':identity,'run_id':'fixture-run'})
    write_json(train/'config.json',s.config);write_json(train/'execution_manifest.json',{'config':s.config,'code':source,'readonly':False})
    write_json(root/'cost_carryforward_manifest.json',{'bank_inherited':False,'limits_unchanged':True,'inherited_attempt_ids':[]})
    s.close();monkeypatch.setattr(gate,'code_identity',lambda:source)
    assert all(r['score']['hard'] is False for r in rows)
    return root


def test_capacity_gate_passes_zero_scores_and_seals_real_receipts(prefix):
    report=gate.audit(prefix)
    assert report['status']=='passed' and not report['performance_gate']
    assert report['facts']['capacity_rejected_requests']==report['facts']['capacity_rejected_tasks']==1
    assert report['facts']['train_task_capacity_rejection_rate']==1/6
    assert gate.verify_receipt(prefix)==report
    audits=[json.loads(p.read_text()) for p in (prefix/'train/requests').glob('*.json')]
    db=sqlite3.connect(prefix/'train/run.sqlite3');cases=[json.loads(r[0]) for r in db.execute('SELECT result FROM tasks')];db.close()
    rates=capacity_statistics(audits,cases)
    assert rates['learning_request_capacity_rejection_rate']==1/7 and rates['train_task_capacity_rejection_rate']==1/6
    record('capacity_gate_fixture',{'fixture_only':True,'actual_http':0,'receipt':report})


def test_gate_accepts_explicit_budget_revision_and_rejects_changed_source(prefix):
    config_path=prefix/'train/config.json';config=json.loads(config_path.read_text())
    ledger_path=prefix/'budget.json';ledger=json.loads(ledger_path.read_text());original=deepcopy(ledger['limits'])
    source=prefix.parent/'old-costs';source.mkdir()
    write_json(source/'budget.json',{'limits':original,'attempts':{},'unknown_billing':False})
    requested={**original,'token_limit':2000000,'request_limit':200}
    ledger['limits']=requested;write_json(ledger_path,ledger)
    config['budget'].update(requested);write_json(config_path,config)
    execution_path=prefix/'train/execution_manifest.json';execution=json.loads(execution_path.read_text())
    execution['config']=config;write_json(execution_path,execution)
    db=sqlite3.connect(prefix/'train/run.sqlite3')
    for task_id,text in db.execute('SELECT id,result FROM tasks').fetchall():
        trace=json.loads(text);trace['method_identity']['config_hash']=digest(config)
        db.execute('UPDATE tasks SET result=? WHERE id=?',(json.dumps(trace),task_id))
    db.commit();db.close()
    carry=json.loads((prefix/'cost_carryforward_manifest.json').read_text())
    carry.update(limits_unchanged=False,source_run=str(source),source_budget_sha256=sha256(source/'budget.json'),
        budget_revision={'schema':'budget-revision.v1','authorization':'explicit increase fixture',
                         'old_limits':original,'new_limits':requested})
    write_json(prefix/'cost_carryforward_manifest.json',carry)
    report=gate.audit(prefix);assert report['status']=='passed' and gate.verify_receipt(prefix)==report
    changed=json.loads((source/'budget.json').read_text());changed['limits']['token_limit']+=1
    write_json(source/'budget.json',changed)
    with pytest.raises(AssertionError,match='Original cost ledger changed'):gate.verify_receipt(prefix)


def test_passed_gate_keeps_existing_checkpoint_resume_with_unchanged_bank(prefix):
    report=gate.audit(prefix)
    db=sqlite3.connect(prefix/'train/run.sqlite3');db.execute('INSERT INTO tasks VALUES(?,?,?,?)',('interrupted-task',1,'running',None));db.commit();db.close()
    checkpoint=prefix/'train/checkpoints/interrupted-task/1/state.json'
    with pytest.raises(FileNotFoundError):gate.verify_receipt(prefix)
    cfg=json.loads((prefix/'train/config.json').read_text())
    write_json(checkpoint,{'schema':'empirical.checkpoint.v2','stage':'task_started','decisions':{},'method_identity':{'config_hash':digest(cfg)}})
    db=sqlite3.connect(prefix/'train/run.sqlite3');trace=json.loads(db.execute("SELECT result FROM tasks WHERE status='completed' LIMIT 1").fetchone()[0]);db.close()
    attempt=deepcopy(next(a for r in trace['requests'] for a in r.get('http_attempts',[])))
    ledger=json.loads((prefix/'budget.json').read_text());billed=deepcopy(ledger['attempts'][attempt['request_id']])
    attempt['request_id']='resume-http-fixture';ledger['attempts'][attempt['request_id']]=billed
    write_json(prefix/'budget.json',ledger)
    write_json(prefix/'train/requests/interrupted-task_attempt1.json',{'requests':[{'http_attempts':[attempt]}]})
    assert gate.verify_receipt(prefix)==report


@pytest.mark.parametrize('change',['status','config','source','billing','ledger','bank','state','avoidable','repair_settings','missing_source'])
def test_gate_does_not_ignore_engineering_or_stale_receipt(prefix,monkeypatch,change):
    gate.audit(prefix)
    if change=='status':
        p=prefix/'prefix_engineering_audit.json';v=json.loads(p.read_text());v['status']='failed';write_json(p,v)
    elif change=='config':
        p=prefix/'train/config.json';v=json.loads(p.read_text());v['runtime']['dynamic_output_min_tokens']=0;write_json(p,v)
    elif change=='source':monkeypatch.setattr(gate,'code_identity',lambda:{'git_sha':'other','source_sha256':'other','tracked_dirty':False})
    elif change in {'billing','ledger'}:
        p=prefix/'budget.json';v=json.loads(p.read_text())
        if change=='billing':v['unknown_billing']=True
        else:next(iter(v['attempts'].values()))['accounted_tokens']+=1
        write_json(p,v)
        if change=='ledger':
            with pytest.raises(AssertionError,match='ledger cost differ'):gate.audit(prefix)
    elif change=='bank':
        db=sqlite3.connect(prefix/'train/bank/bank.sqlite3');db.execute("UPDATE metadata SET value='changed' WHERE key='program_qualification_policy'");db.commit();db.close()
    elif change=='state':(prefix/'train/execution_manifest.json').unlink()
    elif change in {'avoidable','repair_settings','missing_source'}:
        db=sqlite3.connect(prefix/'train/run.sqlite3');row=json.loads(db.execute('SELECT result FROM tasks LIMIT 1').fetchone()[0])
        r=next(r for r in row['requests'] if r.get('material_refusal'));payload=r['material_refusal']['payload']
        body=json.loads(payload['messages'][1]['content'])
        if change=='avoidable':body['proposal']={'decision':'no_change'}
        elif change=='repair_settings':payload['max_tokens']=4096
        else:body['legal_operations'][0]['local_evidence_ref']='local:missing-source'
        payload['messages'][1]['content']=json.dumps(body)+(' '*20000 if change=='avoidable' else '')
        r['messages']=deepcopy(payload['messages']);size=input_token_bound(payload)
        r['material_refusal'].update(wire_bytes=size,parts_bytes={'all':size},**learning_capacity_audit(payload,16384,r['stage'],r['purpose']))
        with pytest.raises(AssertionError):
            gate.capacity_refusal(r,json.loads((prefix/'train/config.json').read_text()))
            gate.verify_directory(body,{})
        db.close();return
    with pytest.raises((AssertionError,ValueError,KeyError,FileNotFoundError)):gate.verify_receipt(prefix)
