import copy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.deployment.oldfirst_plan import compile_plan
from atomic_skillgraph.deployment.release_protocol import ReleaseError
from experiments.release4_metrics import trace_metrics
from experiments.released_stream import run_state


def test_AC01_multiple_requests_same_session_never_duplicate_rejected_cost():
    events = [{'event_id': f'e{i}', 'session_id': 's', 'turn_index': i,
        'prompt_tokens': 10, 'completion_tokens': i + 1, 'reasoning_tokens': i,
        'provider_metadata': {'request_id': f'p{i}'}} for i in range(3)]
    trace = {'llm_usage': events, 'provider_requests': [
        {'request_id': f'r{i}', 'provider_request_id': f'p{i}', 'session_id': 's', 'usage_status': 'reported'}
        for i in range(3)], 'agent_turns': [{'session_id': 's', 'turn_index': i,
            'tool_call_ids': [f'c{i}'], 'provider_metadata': {'request_id': f'p{i}'}} for i in range(3)],
        'native_tool_calls': [{'call_id': f'c{i}', 'session_id': 's', 'tool_name': 'invoke_support_atomic',
            'arguments': {'support_call_id': 'stable'}, 'preflight_result': {'accepted': i == 2,
                'deterministic_rejection_cache_hit': i == 1}} for i in range(3)]}
    result = trace_metrics(trace)
    assert result['all_recorded_usage']['recorded_total_tokens'] == 36
    assert result['support_rejected_decision_cost']['recorded_total_tokens'] == 23
    assert result['support_cached_rejection_cost']['recorded_total_tokens'] == 12
    assert result['support_rejected_decision_requests'] == ['r0', 'r1']
    assert not result['unlinked_call_ids']


def test_AC03_fresh_completed_and_interrupted_are_distinct(tmp_path):
    row = {'output': str(tmp_path/'episode'), 'config_hash': 'fixed'}
    assert run_state(row) == 'fresh'
    out = Path(row['output']); out.mkdir()
    assert run_state(row) == 'resume_required'
    (out/'progress.json').write_text(json.dumps({'state': 'completed'}))
    (out/'summary.json').write_text(json.dumps({'tasks': 134, 'config_hash': 'fixed'}))
    assert run_state(row) == 'completed'
    (out/'summary.json').write_text(json.dumps({'tasks': 1, 'config_hash': 'fixed'}))
    with pytest.raises(ReleaseError):
        run_state(row)


def plan_inputs(tmp_path):
    base = {'seed': 42, 'asset_dispositions': [{'ref': 'skill://discover@1.0.0'}],
        'atomic_revision_jobs': [], 'new_atomic_jobs': [], 'workflow_targets': [],
        'program_realization_jobs': [], 'manual_publications': {'original': {}}}
    delta = {'schema': 'r103.oldfirst-preparation-delta.v1', 'seed': 42,
        'derived_revision_jobs': [{'job_id': 'revision', 'operation': 'version_existing_discovery_contract',
            'kind': 'atomic', 'source_ref': 'skill://discover@1.0.0', 'target_ref': 'skill://discover@1.1.0'}],
        'required_protocols': {}, 'deployment_preferences_patch': {}, 'baseline_commit': 'fixture',
        'preserve_base_publications': ['original']}
    def write(delta):
        a, b = tmp_path/'base.json', tmp_path/'delta.json'
        a.write_text(json.dumps(base)); b.write_text(json.dumps(delta))
        return a, b
    return base, delta, write


def test_BK01_compile_preserves_base_and_ref_dependencies(tmp_path):
    base, delta, write = plan_inputs(tmp_path)
    output = tmp_path/'compiled.json'
    result = compile_plan(*write(delta), output)
    assert all(result[k] == v for k, v in base.items())
    with pytest.raises(FileExistsError):
        compile_plan(*write(delta), output)
    for field, value in [('operation', 'invented'), ('source_ref', 'skill://missing@1.0.0'),
                         ('target_ref', 'skill://other_id@1.1.0')]:
        bad = copy.deepcopy(delta); bad['derived_revision_jobs'][0][field] = value
        with pytest.raises(ReleaseError):
            compile_plan(*write(bad), tmp_path/f'{field}.json')
        assert not (tmp_path/f'{field}.json').exists()
