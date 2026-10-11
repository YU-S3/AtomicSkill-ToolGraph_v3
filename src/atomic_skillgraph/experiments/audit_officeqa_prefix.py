"""Read-only engineering gate for the six tasks counted in the fresh formal Train."""
import argparse
import json
from pathlib import Path
import sqlite3

from .canonical_manifest import ordered_train
from .run_empirical import code_identity, write_json
from ..empirical.contracts import digest
from ..empirical.model_view import expand_material, learning_case


def audit(root):
    root=Path(root);manifest=json.loads((root/'run_manifest.json').read_text())
    source=code_identity()
    assert manifest['source_commit']==source['git_sha'] and not source['tracked_dirty'], 'Pinned source mismatch'
    config=json.loads((root/'train/config.json').read_text())
    assert manifest['initial_bank']=='empty independent Bank'
    assert config['runtime']['finish_input_max_bytes']==16384 and config['runtime']['dynamic_output_min_tokens']==8192
    assert config['llm']['extractor']['protocol']['thinking_type']=='disabled' and config['llm']['extractor']['max_completion_tokens']==8192
    db=sqlite3.connect((root/'train/run.sqlite3').as_uri()+'?mode=ro',uri=True)
    rows=[json.loads(r[0]) for r in db.execute("SELECT result FROM tasks WHERE status='completed' ORDER BY rowid")];db.close()
    assert len(rows)==6, 'The fixed prefix must contain exactly six completed formal tasks'
    public=json.loads(Path(config['manifest']).read_text())['tasks']
    assert [r['task']['task_id'] for r in rows]==[r['task_id'] for r in ordered_train(public,42)[:6]], 'Canonical Train order changed'
    counters={'tasks':6,'native_reads':0,'native_compute':0,'displayed_windows':0,'finish_requests':0,'ordinary_extractor':0}
    from skillcompiler_bench_contracts.office import select_paths
    authorized=[name for name,_ in select_paths(config['harness']['corpus_root'])]
    ordinary=[]
    for trace in rows:
        assert not trace.get('error'), 'Infrastructure/execution wrapper failure: '+str(trace.get('error'))
        for event in trace.get('tools',[]):
            if event.get('state')=='finished':
                counters['native_reads']+=event['name'] in {'read','read_result'}
                counters['native_compute']+=event['name']=='execute_python'
                if event['name']=='glob' and event['result'].get('accepted') and event['arguments']['pattern'] in {'**/*','**/*.txt'}:
                    assert event['result']['data']==authorized
        for request in trace['requests']:
            if request.get('error'):
                assert not str(request['error']).startswith(('material_too_large','finish_unavailable')), request['error']
                ordinary.append({'task_id':trace['task']['task_id'],'purpose':request['purpose'],'reason':request['error']})
            counters['displayed_windows']+=len(request.get('displayed_windows',[]))
            for attempt in request.get('http_attempts',[]):
                wire=attempt['final_payload_audit'];settings=wire['serialized_parameters']
                body=expand_material(json.loads(wire['messages'][1]['content']))
                if request['stage']=='runtime':
                    # Business columns with this name are permitted; a Host root ref is not.
                    text=wire['messages'][1]['content']
                    assert 'execution_paths' not in text and 'scope_omitted' not in text, 'Host scope audit reached model'
                    for window in request.get('displayed_windows',[]):
                        assert window['unit'] in {'array_item_0_based','unicode_codepoint_0_based'}
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
                    assert wire['serialized_payload_utf8_bytes']<=8192 and settings['thinking']['type']=='disabled'
                    assert settings.get('max_tokens',settings.get('max_completion_tokens'))==2048
    ledger=json.loads((root/'budget.json').read_text())
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    report={'status':'passed','source_commit':source['git_sha'],'facts':counters,'ordinary_model_errors':ordinary,
            'performance_gate':False,'extra_pilot_tasks':0,'full_continuation':'same commit and directory',
            'ledger_sha256':digest(ledger)}
    write_json(root/'prefix_engineering_audit.json',report)
    return report


def saved_case_directory(trace,config):
    from ..empirical.contracts import PublicTask
    from ..empirical.local_validation import evidence_from_trace
    task=PublicTask(**trace['task'])
    experience={'local_evidence':evidence_from_trace(task,trace,config['program_environment'])}
    return learning_case(experience,task.physical_key)['local_evidence']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root')
    print(json.dumps(audit(parser.parse_args().root),ensure_ascii=False),flush=True)
