"""Read-only engineering gate for the six tasks counted in the fresh formal Train."""
import argparse
import json
from pathlib import Path
import sqlite3

from .canonical_manifest import ordered_train
from .run_empirical import code_identity, write_json
from ..empirical.contracts import digest
from ..empirical.model_view import expand_material, learning_case, learning_capacity_audit
from ..empirical.budget_governor import input_token_bound
from ..empirical.bank import Bank
from ..empirical import IMPLEMENTATION_REVISION, LEARNING_MATERIAL_VERSION

SCHEMA='office-engineering-gate.v2'


def evidence(root,*,continuation=False):
    root=Path(root);manifest=json.loads((root/'run_manifest.json').read_text());source=code_identity()
    assert manifest['source_commit']==source['git_sha'] and not source['tracked_dirty'], 'Pinned source mismatch'
    assert manifest['source_sha256']==source['source_sha256'] and manifest['run_seed']==42
    config=json.loads((root/'train/config.json').read_text())
    assert config['experiment']['implementation_revision']==IMPLEMENTATION_REVISION
    assert config['learning']['material_version']==LEARNING_MATERIAL_VERSION
    assert config['learning']['structural_repair_payload_max_bytes']==16384
    execution=json.loads((root/'train/execution_manifest.json').read_text())
    assert execution['config']==config and execution['code']==source and not execution['readonly']
    db=sqlite3.connect((root/'train/run.sqlite3').as_uri()+'?mode=ro',uri=True)
    pending=db.execute("SELECT id,attempts,status FROM tasks WHERE status!='completed'").fetchall()
    if pending:
        assert continuation and len(pending)==1, 'Initial gate requires a persisted task boundary'
        task_id,attempt,status=pending[0];assert status=='running', 'Unresolved task state'
        checkpoint=json.loads((root/'train/checkpoints'/task_id/str(attempt)/'state.json').read_text())
        assert checkpoint['schema']=='empirical.checkpoint.v2' and checkpoint['method_identity']['config_hash']==digest(config), 'Missing or mismatched resume checkpoint'
    rows=[json.loads(r[0]) for r in db.execute("SELECT result FROM tasks WHERE status='completed' ORDER BY rowid")];db.close()
    assert len(rows)>=6, 'Missing fixed prefix state'
    for trace in rows:
        assert trace['method_identity']['config_hash']==digest(config), 'Task configuration identity changed'
    for previous,current in zip(rows,rows[1:]):
        assert previous['knowledge_after']==current['knowledge_before'], 'Bank change outside committed learning'
    bank=Bank(root/'train/bank',readonly=True,seed=42,environment=config['program_environment'])
    try:
        assert bank.digest()==rows[-1]['knowledge_after'], 'Bank differs from the last committed task'
        empty=digest({'metadata':list(bank.db.execute('SELECT * FROM metadata ORDER BY rowid')),'assets':[],'attempts':[]})
        assert rows[0]['knowledge_before']==empty, 'Fresh run did not start from an empty Bank'
        cases={r['task']['physical_key']:r['task'] for r in bank.train_cases()}
        assert cases=={r['task']['physical_key']:r['task'] for r in rows}, 'Missing or foreign completed Train source'
    finally:bank.close()
    ledger=json.loads((root/'budget.json').read_text());carry=json.loads((root/'cost_carryforward_manifest.json').read_text())
    assert ledger['limits']=={k:config['budget'][k] for k in ('token_limit','finish_reserve','request_limit')}
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())<=ledger['limits']['token_limit'] and len(ledger['attempts'])<=ledger['limits']['request_limit']
    assert carry['bank_inherited'] is False and carry['limits_unchanged'] is True
    prefix=rows[:6];ids=[a['request_id'] for t in prefix for r in t['requests'] for a in r.get('http_attempts',[])]
    inherited=set(carry['inherited_attempt_ids'])
    assert len(ids)==len(set(ids)) and not inherited&set(ids), 'Repeated HTTP billing identity'
    assert inherited|set(ids)<=set(ledger['attempts']), 'HTTP receipt missing from ledger'
    if len(rows)==6:
        accounted=inherited|set(ids)
        if pending:
            audit_path=root/'train/requests'/(task_id+'_attempt'+str(attempt)+'.json')
            if audit_path.exists():
                audit=json.loads(audit_path.read_text())
                accounted|={a['request_id'] for r in audit['requests'] for a in r.get('http_attempts',[])}
        assert set(ledger['attempts'])==accounted, 'Unattributed prefix or interrupted-task cost'
    for trace in prefix:
        for request in trace['requests']:
            for attempt in request.get('http_attempts',[]):
                billed=ledger['attempts'][attempt['request_id']]
                assert billed['accounted_tokens']==attempt['usage']['total_tokens'] and billed['actual_usage']==attempt['raw_usage'], 'HTTP receipt and ledger cost differ'
    binding={'source_commit':source['git_sha'],'source_sha256':source['source_sha256'],
        'run_id':manifest['run_id'],'manifest_identity_sha256':digest(manifest['identity']),
        'config_sha256':digest(config),'execution_manifest_sha256':digest(execution),
        'carryforward_sha256':digest(carry),'prefix_traces_sha256':digest(prefix),
        'prefix_ledger_sha256':digest({key:ledger['attempts'][key] for key in sorted(inherited|set(ids))}),
        'ledger_limits_sha256':digest(ledger['limits']),'bank_digest_at_gate':prefix[-1]['knowledge_after']}
    return root,manifest,config,rows,ledger,binding


def capacity_refusal(request,config):
    receipt=request.get('material_refusal') or {};payload=receipt['payload']
    stage,purpose=request['stage'],request['purpose']
    assert stage in {'extractor','tool_builder'} and request.get('admission_status')=='local_refusal', 'Unresolved non-learning payload overflow: '+str(request.get('error'))
    assert request.get('failure_code')=='learning_capacity_rejected' and not request.get('http_attempts') and not request.get('response')
    key='structural_repair_payload_max_bytes' if purpose=='learning_structure_repair' else 'builder_payload_max_bytes' if stage=='tool_builder' else 'extractor_payload_max_bytes'
    assert receipt['limit_bytes']==request['payload_max_bytes']==config['learning'][key]
    assert receipt['wire_bytes']==input_token_bound(payload)>receipt['limit_bytes']
    assert sum(receipt['parts_bytes'].values())==receipt['wire_bytes'] and receipt['required_fields'] and receipt['reason']
    assert payload['messages']==request['messages'] and payload['tools']==request['tools']
    if purpose=='learning_structure_repair':
        assert payload['thinking']['type']=='disabled' and payload.get('max_tokens',payload.get('max_completion_tokens'))==2048
        assert payload['tool_choice']['function']['name']=='submit_learning' and request['repair']==1
    proof=learning_capacity_audit(payload,receipt['limit_bytes'],stage,purpose)
    assert proof['classification']=='learning_capacity_rejected' and all(receipt.get(k)==v for k,v in proof.items()), 'Unexplained or avoidable capacity refusal'
    return payload


def audit(root):
    root,manifest,config,rows,ledger,binding=evidence(root)
    assert manifest['initial_bank']=='empty independent Bank'
    assert config['runtime']['finish_input_max_bytes']==16384 and config['runtime']['dynamic_output_min_tokens']==8192
    assert config['llm']['extractor']['protocol']['thinking_type']=='disabled' and config['llm']['extractor']['max_completion_tokens']==8192
    assert len(rows)==6, 'The fixed prefix must contain exactly six completed formal tasks'
    public=json.loads(Path(config['manifest']).read_text())['tasks']
    assert [r['task']['task_id'] for r in rows]==[r['task_id'] for r in ordered_train(public,42)[:6]], 'Canonical Train order changed'
    counters={'tasks':6,'native_reads':0,'native_compute':0,'displayed_windows':0,'finish_requests':0,'ordinary_extractor':0,
        'learning_requests':0,'capacity_rejected_requests':0,'capacity_rejected_tasks':0}
    from skillcompiler_bench_contracts.office import select_paths
    authorized=[name for name,_ in select_paths(config['harness']['corpus_root'])]
    ordinary=[];capacity=[]
    sources={card['local_evidence_ref']:card for t in rows for card in saved_case_directory(t,config)}
    for trace in rows:
        assert not trace.get('error'), 'Infrastructure/execution wrapper failure: '+str(trace.get('error'))
        assert trace.get('learning_status')!='failed_engineering' and not trace.get('learning_error'), 'Unresolved learning infrastructure error'
        rejected=False
        for event in trace.get('tools',[]):
            if event.get('state')=='finished':
                counters['native_reads']+=event['name'] in {'read','read_result'}
                counters['native_compute']+=event['name']=='execute_python'
                if event['name']=='glob' and event['result'].get('accepted') and event['arguments']['pattern'] in {'**/*','**/*.txt'}:
                    assert event['result']['data']==authorized
        for request in trace['requests']:
            counters['learning_requests']+=request['stage'] in {'extractor','tool_builder'}
            if request.get('error'):
                assert not str(request['error']).startswith('finish_unavailable'), request['error']
                if str(request['error']).startswith('material_too_large'):
                    payload=capacity_refusal(request,config);rejected=True
                    capacity.append({'task_id':trace['task']['task_id'],'request_id':request['id'],'purpose':request['purpose'],
                        'classification':'learning_capacity_rejected','wire_bytes':input_token_bound(payload),'limit_bytes':request['payload_max_bytes']})
                    verify_directory(expand_material(json.loads(payload['messages'][1]['content'])),sources)
                else:ordinary.append({'task_id':trace['task']['task_id'],'purpose':request['purpose'],'reason':request['error']})
            counters['displayed_windows']+=len(request.get('displayed_windows',[]))
            for attempt in request.get('http_attempts',[]):
                wire=attempt['final_payload_audit'];settings=wire['serialized_parameters']
                body=expand_material(json.loads(wire['messages'][1]['content']))
                if request['stage'] in {'extractor','tool_builder'}:
                    key='structural_repair_payload_max_bytes' if request['purpose']=='learning_structure_repair' else 'builder_payload_max_bytes' if request['stage']=='tool_builder' else 'extractor_payload_max_bytes'
                    assert wire['serialized_payload_utf8_bytes']<=request['payload_max_bytes']==config['learning'][key]
                    verify_directory(body,sources)
                if request['stage']=='runtime':
                    # Business columns with this name are permitted; a Host root ref is not.
                    for window in request.get('displayed_windows',[]):
                        assert window['unit'] in {'array_item_0_based','unicode_codepoint_0_based'}
                        assert window['source_ref']['path'][:1]!=['scope_audit'], 'Host scope audit reached model'
                    if request['purpose']=='finish_only':
                        counters['finish_requests']+=1
                        assert wire['serialized_payload_utf8_bytes']<=16384 and settings.get('max_tokens',settings.get('max_completion_tokens'))==2048
                    else:
                        assert 8192<=settings.get('max_tokens',settings.get('max_completion_tokens'))<=32768
                        assert request['parent_finish_reserve']==18432
                if request['stage']=='extractor' and request['purpose']=='extractor':
                    counters['ordinary_extractor']+=1
                    assert settings['thinking']['type']=='disabled' and settings.get('max_tokens',settings.get('max_completion_tokens'))==8192
                    assert settings['tool_choice']['function']['name']=='submit_learning'
                    assert wire['serialized_payload_utf8_bytes']<=16384
                    cards=body.get('experience',{}).get('local_evidence',[]);assert len(cards)<=4
                    projected=saved_case_directory(trace,config)
                    code_ids={c['local_evidence_ref'] for c in projected if c['operation_name']=='execute_python'}
                    if code_ids:assert code_ids & {c['local_evidence_ref'] for c in cards}, 'Recorded code omitted from proposal'
                if request['purpose']=='learning_structure_repair':
                    assert wire['serialized_payload_utf8_bytes']<=16384 and settings['thinking']['type']=='disabled'
                    assert settings.get('max_tokens',settings.get('max_completion_tokens'))==2048
                    assert settings['tool_choice']['function']['name']=='submit_learning' and request['repair']==1
        counters['capacity_rejected_tasks']+=rejected
    counters['capacity_rejected_requests']=len(capacity)
    counters['learning_request_capacity_rejection_rate']=len(capacity)/counters['learning_requests'] if counters['learning_requests'] else 0
    counters['train_task_capacity_rejection_rate']=counters['capacity_rejected_tasks']/6
    report={'schema':SCHEMA,'status':'passed',**binding,'facts':counters,'ordinary_model_errors':ordinary,'capacity_rejections':capacity,
            'performance_gate':False,'extra_pilot_tasks':0,'full_continuation':'same commit and directory',
            'ledger_sha256':digest(ledger)}
    report['receipt_sha256']=digest(report)
    write_json(root/'prefix_engineering_audit.json',report)
    return report


def verify_directory(body,sources):
    cards=[*body.get('legal_operations',[]),*body.get('experience',{}).get('local_evidence',[])]
    for case in body.get('completed_train_cases',[]):cards.extend(case.get('local_evidence',[]))
    for card in cards:
        actual=sources.get(card['local_evidence_ref']);assert actual, 'Missing real source operation'
        for key,value in card.items():
            if key=='input_sources':assert all(v in actual[key] for v in value), 'Invented source binding'
            else:assert value==actual[key], 'Source card differs from the saved operation'
    if 'build_request' in body:assert body.get('examples'), 'Missing Builder source material'
    for example in body.get('examples',[]):
        actual=sources.get(example['local_evidence_ref']);assert actual, 'Missing Builder source operation'
        assert example['case_id']==actual['case_id'] and example['kind']==actual['kind'], 'Builder source identity changed'


def verify_receipt(root):
    root,_,_,_,_,binding=evidence(root,continuation=True)
    report=json.loads((root/'prefix_engineering_audit.json').read_text())
    assert report.get('schema')==SCHEMA and report.get('status')=='passed', 'No passed engineering receipt'
    assert report.get('receipt_sha256')==digest({k:v for k,v in report.items() if k!='receipt_sha256'}), 'Gate receipt changed'
    assert all(report.get(k)==v for k,v in binding.items()), 'Gate source/config/state/ledger binding changed'
    return report


def saved_case_directory(trace,config):
    from ..empirical.contracts import PublicTask
    from ..empirical.local_validation import evidence_from_trace
    task=PublicTask(**trace['task'])
    experience={'local_evidence':evidence_from_trace(task,trace,config['program_environment'])}
    return learning_case(experience,task.physical_key)['local_evidence']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();print(json.dumps(verify_receipt(args.root) if args.verify else audit(args.root),ensure_ascii=False),flush=True)
