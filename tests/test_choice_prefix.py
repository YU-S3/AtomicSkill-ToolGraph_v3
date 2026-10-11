"""Fresh prefix gates inspect engineering receipts, never task accuracy."""
import json
import sqlite3

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.empirical.public_process import VERSION
from atomic_skillgraph.experiments import audit_choice_prefix as gate
from atomic_skillgraph.experiments.canonical_manifest import ordered_train


@pytest.fixture
def prefix(tmp_path,monkeypatch):
    root=tmp_path/'run';train=root/'train';train.mkdir(parents=True)
    write=lambda path,value:path.write_text(json.dumps(value))
    public=[{'task_id':'task-'+str(i)} for i in range(8)]
    authority=tmp_path/'train.json';write(authority,{'tasks':public})
    write(root/'run_manifest.json',{'source_commit':'fixture-commit','initial_bank':'empty independent Bank',
        'choice_method_version':VERSION,'run_seed':42,'choice_projection_version':'fixture-projection',
        'public_materialization_sha256':'fixture-public-hash'})
    limits={'token_limit':10000,'request_limit':20,'finish_reserve':0}
    write(train/'config.json',{'manifest':str(authority),'budget':limits,
        'runtime':{'choice_guidance':{'public_process':True},'choice_bounded_finalization':{'enabled':False}},
        'experiment':{'choice_projection_version':'fixture-projection','public_materialization_sha256':'fixture-public-hash'}})
    rows=[];attempts={'inherited':{'status':'metered','accounted_tokens':10}}
    for i,t in enumerate(ordered_train(public,42)[:6]):
        rid='http-'+str(i);attempts[rid]={'status':'metered','accounted_tokens':10}
        rows.append({'task':t,'score':{'hard':False},'execution':{},'learning':{'decision':'no_change'},
            'requests':[{'stage':'extractor','purpose':'guidance_learning','payload_max_bytes':16384,
                'http_attempts':[{'request_id':rid,'final_payload_audit':{'serialized_payload_utf8_bytes':1000,
                    'serialized_parameters':{'thinking':{'type':'disabled'}}}}]}]})
    db=sqlite3.connect(train/'run.sqlite3');db.execute('CREATE TABLE tasks(status TEXT,result TEXT)')
    db.executemany('INSERT INTO tasks VALUES(?,?)',[('completed',json.dumps(r)) for r in rows]);db.commit();db.close()
    bank=Bank(train/'bank');bank.bind_choice_method(VERSION);bank.close()
    write(root/'budget.json',{'limits':limits,'unknown_billing':False,'attempts':attempts})
    write(root/'cost_carryforward_manifest.json',{'inherited_attempt_ids':['inherited'],'limits_unchanged':True,'bank_inherited':False})
    monkeypatch.setattr(gate,'code_identity',lambda:{'tracked_dirty':False,'git_sha':'fixture-commit'})
    return root


def update(path,edit):
    value=json.loads(path.read_text());edit(value);path.write_text(json.dumps(value))


def test_zero_accuracy_does_not_block_same_bank_continuation(prefix):
    report=gate.audit(prefix)
    assert report['status']=='passed' and not report['performance_gate'] and report['extra_pilot_tasks']==0
    assert report['coverage']['counts']['hard_success']==0 and report['coverage']['counts']['public_process_present']==0


@pytest.mark.parametrize('change',['source','billing','duplicate','bank','limit','seed','oversize'])
def test_identity_budget_and_wire_errors_block_without_new_calls(prefix,change,monkeypatch):
    if change=='source':monkeypatch.setattr(gate,'code_identity',lambda:{'tracked_dirty':True,'git_sha':'fixture-commit'})
    elif change=='billing':update(prefix/'budget.json',lambda v:v.update(unknown_billing=True))
    elif change=='duplicate':update(prefix/'cost_carryforward_manifest.json',lambda v:v.update(inherited_attempt_ids=[]))
    elif change=='bank':update(prefix/'cost_carryforward_manifest.json',lambda v:v.update(bank_inherited=True))
    elif change=='limit':update(prefix/'budget.json',lambda v:v['limits'].update(token_limit=20000))
    elif change=='seed':update(prefix/'run_manifest.json',lambda v:v.update(run_seed=43))
    else:
        db=sqlite3.connect(prefix/'train/run.sqlite3');row=json.loads(db.execute('SELECT result FROM tasks LIMIT 1').fetchone()[0])
        row['requests'][0]['http_attempts'][0]['final_payload_audit']['serialized_payload_utf8_bytes']=16385
        db.execute('UPDATE tasks SET result=? WHERE rowid=1',(json.dumps(row),));db.commit();db.close()
    with pytest.raises(AssertionError):gate.audit(prefix)
    assert not (prefix/'prefix_engineering_audit.json').exists()


@pytest.mark.parametrize('complete',[True,False])
def test_required_oversize_needs_full_zero_http_receipt(prefix,complete):
    payload={'messages':[{'role':'user','content':'x'*9000}]};size=input_token_bound(payload)
    request={'stage':'extractor','purpose':'learning_structure_repair','error':'material_too_large: required fields',
        'http_attempts':[],'material_refusal':{'reason':'Required original proposal, errors and schema cannot be dropped.',
            'wire_bytes':size,'limit_bytes':8192,'required_fields':['proposal','errors','tools']}}
    if complete:request['material_refusal'].update(payload=payload,parts_bytes={'user':size})
    db=sqlite3.connect(prefix/'train/run.sqlite3');row=json.loads(db.execute('SELECT result FROM tasks LIMIT 1').fetchone()[0])
    row['requests'].append(request);db.execute('UPDATE tasks SET result=? WHERE rowid=1',(json.dumps(row),));db.commit();db.close()
    if complete:assert gate.audit(prefix)['ordinary_or_bounded_errors'][0]['purpose']=='learning_structure_repair'
    else:
        with pytest.raises((AssertionError,KeyError)):gate.audit(prefix)
