import json
import sqlite3
from pathlib import Path
import pytest
from experiments.run_v3_bank_release import main as release_cli
from experiments.run_v3_released_frozen import main as eval_cli
from experiments.release_report import write_release_report,aggregate
from atomic_skillgraph.core.serialization import atomic_write_json


def test_code_inventory_prunes_only_previously_excluded_paths(tmp_path):
    import hashlib
    from experiments.protocol import code_file_manifest,_CODE_SUFFIXES,_CODE_EXCLUDED_DIRS
    for name in ('src/main.py','configs/test.yaml','runs/huge/ignored.py','.git/internal.py','build.egg-info/module.py','extra.txt'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('test')
    files=[p for p in tmp_path.rglob('*') if p.is_file() and p.suffix.casefold() in _CODE_SUFFIXES
           and not any(part in _CODE_EXCLUDED_DIRS or part.endswith('.egg-info') for part in p.relative_to(tmp_path).parts)]
    expected=[{'path':p.relative_to(tmp_path).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted(files,key=lambda p:p.as_posix())]
    assert code_file_manifest(tmp_path)==expected


@pytest.mark.parametrize('command',[release_cli,eval_cli])
def test_cli_help_and_missing_arguments(command):
    with pytest.raises(SystemExit) as exc:command(['--help'])
    assert exc.value.code==0
    with pytest.raises(SystemExit) as exc:command([])
    assert exc.value.code==2


def test_release_matrix_rejects_wrong_repeats(tmp_path):
    with pytest.raises(SystemExit) as exc:
        release_cli(['make-configs','--output-root',str(tmp_path),'--seeds','42','43','44','--profile','lean','--repeats','42:1','43:1','44:1'])
    assert exc.value.code==2


def test_costs_include_failed_attempts_unknown_usage_and_real_digest(tmp_path):
    config={'bank_release':{'expected_bank_digest':'d'},'deployment':{'presentation_profile':'current'}}
    atomic_write_json(tmp_path/'run_manifest.json',{'knowledge_digest':'d'})
    with sqlite3.connect(tmp_path/'run_state.sqlite3') as db:
        db.execute('CREATE TABLE run_tasks(task_id TEXT,trace_id TEXT,state TEXT)')
        db.execute("INSERT INTO run_tasks VALUES('task','done','completed')")
    traces=[{'trace_id':name,'task':{'task_id':'task'},'benchmark_success':success,
             'llm_usage':[{'event_id':name,'prompt_tokens':10,'completion_tokens':5,'reasoning_tokens':None,'bucket':'tool_builder_runtime'}],
             'provider_requests':[{'request_id':name,'session_id':'s','usage_status':'reported' if success else 'unknown'}]}
            for name,success in [('failed',False),('done',True)]]
    traces[-1]['metadata']={'runtime_context_projection_audits':[
        {'session_id':'s','release_expression':{'lean_payload_hash':'sent_hash'}}]}
    traces[-1]['provider_requests'][0]['final_payload_audit']={'policy_context_sha256':['sent_hash']}
    result=write_release_report(tmp_path,config,resource_traces=traces,digest_after='d')
    assert result['tasks']==1 and result['successes']==1
    assert result['total_recorded_tokens']==30 and result['total_tokens'] is None
    assert result['unknown_usage_requests']==1 and result['programmer_tokens']==30
    assert result['reasoning_tokens'] is None
    assert json.loads((tmp_path/'runtime_expression_coverage.json').read_text())['matched_lean_requests']==1
    with pytest.raises(ValueError,match='mutated'):
        write_release_report(tmp_path,config,resource_traces=traces,digest_after='changed')
