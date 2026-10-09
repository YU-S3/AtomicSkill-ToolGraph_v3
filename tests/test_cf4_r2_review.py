"""Q1–Q5: eight inert review fixtures; no HTTP, Docker or generated execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.errors import AtomicSkillGraphError, BudgetExhausted
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, digest, object_schema
from atomic_skillgraph.empirical.learner import Learner
from atomic_skillgraph.empirical.model_view import canonical_bytes, project, project_related_candidates, expand_material
from atomic_skillgraph.empirical.program_submission import normalize_program_result
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.trial_snapshot import seal_trial_workspace
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity
from atomic_skillgraph.experiments.run_formal import model_settings, resolved_config
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import office
from test_empirical import program

CHILD = Path(os.environ.get('CF4_R2_CHILD', '/home/yangchengyu/cf4_r2_spreadsheet_recovery_seed42_20261008'))
ORIGINAL = Path('/home/yangchengyu/asg_cf4_r1_20261007')
DATASETS = Path('/home/yangchengyu/main_experiment_v1_resources_cf4_20261006_v1')


@pytest.fixture(autouse=True)
def inert(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('Review fixtures forbid HTTP, Docker and generated execution')
    monkeypatch.setattr('requests.sessions.Session.request', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.setattr(ProgramWorker, 'execute', forbidden)


def evidence(name, value):
    if os.environ.get('CF4_R2_REVIEW_EVIDENCE_DIR'):
        root = Path(os.environ['CF4_R2_REVIEW_EVIDENCE_DIR']); root.mkdir(parents=True, exist_ok=True)
        (root / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def trial(tmp_path):
    s = office(tmp_path)
    s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    s.adapter_factory = lambda: OfficeAdapter({'office': {'answer': '42'}}, s.config)
    calls = []
    def literal(*args):
        calls.append('literal_worker')
        return {'status': 'ok', 'outputs': {'answer': '42'}}
    s.worker = SimpleNamespace(execute=literal, settings={'max_rpc_message_bytes': 10000})
    p = s.bank.put('program', {**program('def run(ctx, inputs):\n    raise AssertionError("fixture source must never execute")', outputs=object_schema({'answer': {'type': 'string'}}, ['answer'])),
                              'result_role': 'final_answer', 'allowed_tools': []})
    trace = {'score': {'hard': True, 'raw_score': 1.0}, 'execution': {'attempts': []}}
    s.learner.learn = lambda task, trace: s.test_program(p, {}, task, trial_id='review', continuation=False)
    return s, p, trace, calls


def assert_host(s, p, trace, failure, operation):
    assert trace['score'] == {'hard': True, 'raw_score': 1.0}
    assert trace['learning_status'] == 'failed_engineering'
    error = trace['learning_error']
    assert error['code'] == 'infrastructure_failure' and error['repair_target'] == 'host'
    assert error['stage'] == 'worker_finished' and error['outer_stage'] == 'learning'
    assert error['operation'] == operation and error['cause_type'] == 'ValueError'
    assert isinstance(failure.value.__cause__, ValueError)
    assert not s.bank.attempts(p['id'])
    assert s.checkpoint.state['stage'] == 'task_execution_finished'
    state = json.loads((s.checkpoint.root / 'trials/review/executions' / error['trial_execution_id'] / 'state.json').read_text())
    assert state['stage'] == 'trial_exception' and state['exception']['operation'] == operation
    return error


def test_q1_scorer_valueerror_crosses_learning(tmp_path):
    s, p, trace, calls = trial(tmp_path)
    def factory():
        a = OfficeAdapter({'office': {'answer': '42'}}, s.config)
        def score(*args): raise ValueError('literal scorer fault')
        a.evaluate = score
        return a
    s.adapter_factory = factory
    try:
        with pytest.raises(AtomicSkillGraphError) as failure: s.learn_trace(s.adapter.task, trace)
        error = assert_host(s, p, trace, failure, 'evaluate')
        assert len(calls) == 1
        evidence('Q1', {'error': error, 'score': trace['score'], 'literal_worker_calls': len(calls), 'positive': 0})
    finally: s.close()


def test_q2_corrupt_saved_workspace_never_replays_worker(tmp_path):
    s, p, trace, calls = trial(tmp_path)
    logical = TaskCheckpoint(s.checkpoint.root / 'trials/review')
    logical.advance('trial_started', execution_id='saved')
    checkpoint = TaskCheckpoint(logical.root / 'executions/saved')
    workspace = s.adapter.workspace
    stage = workspace.stage(); (stage / 'note.txt').write_text('saved public artifact')
    workspace.publish(stage, ['note.txt'])
    receipt = seal_trial_workspace(s.adapter, checkpoint.root / 'artifacts')
    result = normalize_program_result(s.adapter, p, {'status': 'ok', 'outputs': {'answer': '42'}})
    result['terminal_by_program'] = False
    checkpoint.advance('worker_finished', worker_result=result, artifacts=receipt, native_event_start=0, workspace_before={})
    corrupt = Path(receipt['root']) / 'manifest.json'; corrupt.chmod(0o600); corrupt.write_text('{}')
    try:
        with pytest.raises(AtomicSkillGraphError) as failure: s.learn_trace(s.adapter.task, trace)
        error = assert_host(s, p, trace, failure, 'restore_workspace')
        assert not calls
        evidence('Q2', {'error': error, 'literal_worker_calls': 0, 'positive': 0, 'score': trace['score']})
    finally: s.close()


@pytest.mark.parametrize('kind', ['declaration', 'budget'])
def test_q3_known_learning_rejections_preserve_score(tmp_path, kind):
    s = office(tmp_path)
    trace = {'score': {'hard': True, 'raw_score': 1.0}, 'execution': {'attempts': []}}
    def reject(*args, **kwargs):
        if kind == 'declaration': raise ValueError('Skill reference must be an existing Skill ID')
        raise BudgetExhausted('extractor_token_budget_exhausted', 'Known learning budget exhausted')
    s.learner._receive = reject
    try:
        s.learn_trace(s.adapter.task, trace)
        assert trace['learning_status'] == ('deferred_budget' if kind == 'budget' else 'rejected') and 'learning_error' not in trace
        assert trace['score'] == {'hard': True, 'raw_score': 1.0}
        evidence('Q3_' + kind, {'learning_status': trace['learning_status'], 'score': trace['score']})
    finally: s.close()


@pytest.mark.parametrize('index,expected', [(57, 240493), (58, 227785)])
def test_q4_same_candidates_compact_view_readonly(tmp_path, index, expected):
    assert CHILD.exists(), 'Review requires the saved recovery Bank'
    before = tree_identity(CHILD / 'train/bank')
    bank = Bank(CHILD / 'train/bank', readonly=True)
    try:
        task = PublicTask(**json.loads((CHILD / 'train/execution_manifest.json').read_text())['tasks'][index])
        learner = Learner(SimpleNamespace(bank=bank))
        original = learner.related_candidates(task)
        full = deepcopy(original)
        projected = project('extractor', {'related': original})
        assert len(original) == 8 and original == full
        expected = canonical_bytes(original)
        assert projected['candidate_view_audit']['after_bytes'] < expected
        expanded = expand_material(projected)
        for old, new in zip(original, expanded['related']):
            assert old['id'] == new['id']
            for key in ('goal', 'guidance', 'input_schema', 'output_schema', 'entry_constraints', 'execution_intent', 'used_physical_tasks'):
                assert old.get(key) == new.get(key)
            assert old['current_job']['case_bindings'] == new['current_job']['case_bindings']
            for key in new['current_job']: assert old['current_job'][key] == new['current_job'][key]
            for key in new['current_program']: assert old['current_program'][key] == new['current_program'][key]
            for a, b in zip(old['independent_results'], new['independent_results']):
                for key in ('id', 'program_id', 'task_key', 'outcome', 'basis'): assert a.get(key) == b.get(key)
                assert not {'source', 'tools', 'result', 'workspace'} & b.keys()
            assert 'source' not in new['current_program']
            assert new['trial_slots']['limit'] == 2
            assert new['trial_slots']['fixed_case_count'] == len(old['current_job']['case_bindings'])
            assert new['trial_slots']['unfilled_slot_count'] == 2 - len(old['current_job']['case_bindings'])
            assert new['trial_slots']['untried_case_ids'] == [b['case_id'] for b in old['current_job']['case_bindings']
                                                           if b['case_id'] not in old['used_physical_tasks']]
        error_copy = deepcopy(original)
        error_copy[0]['independent_results'][0]['error'] = {'code': 'fixture', 'repair_target': 'host', 'message': 'x' * 4096}
        error = project_related_candidates(error_copy)[0]['independent_results'][0]['error']
        assert error == {'code': 'fixture', 'repair_target': 'host', 'message': 'x' * 512}
        from atomic_skillgraph.agents.protocol import AgentTurn
        provider = SimpleNamespace(complete=lambda *a, **k: AgentTurn(content='literal', tool_calls=[], finish_reason='stop',
            prompt_tokens=0, completion_tokens=0, total_tokens=0, reasoning_tokens=None, latency_ms=0))
        s = office(tmp_path, provider=provider)
        try:
            assert s.agent('extractor', 'fixture', projected, None, None) == 'literal'
            audit = s.requests[-1]
            assert audit['candidate_view_version'] == projected['candidate_view_version']
            assert audit['candidate_view_audit'] == projected['candidate_view_audit']
            assert json.loads(audit['messages'][1]['content']) == projected
        finally: s.close()
        evidence('Q4_' + str(index + 1), {'task_id': task.task_id, **projected['candidate_view_audit'],
            'reduction': 1 - projected['candidate_view_audit']['after_bytes'] / expected,
            'bank_digest': bank.digest(), 'bank_files_sha256': before['sha256'],
            'related': projected['related']})
    finally: bank.close()
    assert tree_identity(CHILD / 'train/bank') == before


def launch_configs(root, authority):
    base = json.loads((root / 'train/config.json').read_text())
    lock = json.loads((ORIGINAL / 'models.lock.json').read_text())
    base = model_settings(base, lock['current_test'])
    profile = json.loads((ORIGINAL / 'benchmark_profiles.json').read_text())['profiles']['spreadsheet']
    configs = {split: resolved_config(base, profile, 'spreadsheetbench', 42, split, root, DATASETS, authority, None)
               for split in ('train', 'val', 'test')}
    identity = json.loads((root / 'run_manifest.json').read_text())['identity']
    for config in configs.values():
        config['experiment'].update({key: identity[key] for key in ('benchmark_contracts_sha256', 'public_materialization_sha256')})
    return configs


@pytest.mark.parametrize('correct', [False, True])
def test_q5_launch_config_guard_before_http(tmp_path, correct):
    assert CHILD.exists(), 'Review requires the saved recovery configuration'
    authority = ORIGINAL / 'data/main_experiment_v1' if correct else Path.cwd() / 'data/main_experiment_v1'
    # A historical run cannot silently adopt the new learning/material identity.
    with pytest.raises(ValueError, match='Unsupported execution policy'):
        launch_configs(CHILD, authority)
    configs = json.loads((CHILD / 'resolved_config.json').read_text())
    manifest = json.loads((CHILD / 'run_manifest.json').read_text())
    if correct:
        assert digest(configs) == manifest['config_hash']
        assert configs == json.loads((CHILD / 'resolved_config.json').read_text())
        assert configs['train'] == json.loads((CHILD / 'train/config.json').read_text())
    else:
        configs['train']['manifest'] = str(authority / 'spreadsheetbench/train.json')
        assert digest(configs) != manifest['config_hash']
        copied = tmp_path / 'guard'; copied.mkdir()
        (copied / 'run_manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='Formal resolved config changed'):
            FormalLog(copied, manifest['identity'], configs, resume=True)
    evidence('Q5_' + str(correct), {'correct_absolute_authority': correct, 'config_hash': digest(configs),
        'expected_hash': manifest['config_hash'], 'split_hashes': {k: digest(v) for k, v in configs.items()},
        'http_calls': 0, 'test_bodies_read': False})
