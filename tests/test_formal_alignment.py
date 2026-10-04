"""Shared membership, passive observations and immutable evaluation state."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import requests
import yaml

from atomic_skillgraph.agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.experiments.canonical_manifest import BENCHMARKS, COUNTS, ordered_train, sha256, verify
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity
from atomic_skillgraph.experiments.run_empirical import run
from atomic_skillgraph.experiments.run_formal import resolved_config
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_empirical import Adapter, config_for, program
from test_provider_transport import Response


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / 'data/main_experiment_v1'


def test_all_canonical_members_fixed_and_run_seed_only_changes_train_order(tmp_path):
    manifest = verify(AUTHORITY)
    for benchmark in BENCHMARKS:
        train = json.loads((AUTHORITY / benchmark / 'train.json').read_text())['tasks']
        orders = [ordered_train(train, seed) for seed in (42, 43, 44)]
        assert len({tuple(r['source_id'] for r in order) for order in orders}) == 3
        assert all({r['source_id'] for r in order} == {r['source_id'] for r in train} for order in orders)
        assert [manifest['benchmarks'][benchmark]['splits'][split]['count'] for split in ('train', 'val', 'test', 'reserve')] == list(COUNTS[benchmark])
    copied = tmp_path / 'authority'
    import shutil
    shutil.copytree(AUTHORITY, copied)
    path = copied / 'docvqa/val.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='hash/path'):
        verify(copied)


def test_category_quotas_and_preserved_search_and_alf_evaluation_ids():
    for benchmark, field, expected in [
        ('officeqa', 'difficulty', {'easy': (55, 11, 47), 'hard': (65, 13, 55)}),
        ('spreadsheetbench', 'instruction_type', {'Cell-Level Manipulation': (138, 14, 123), 'Sheet-Level Manipulation': (62, 6, 57)})]:
        for i, split in enumerate(('train', 'val', 'test')):
            rows = json.loads((AUTHORITY / benchmark / (split + '.json')).read_text())['tasks']
            assert {label: sum(r['task_metadata'][field] == label for r in rows) for label in expected} == {label: sizes[i] for label, sizes in expected.items()}
    for split, filename in [('val', 'validation_24.json'), ('test', 'test_ood_full_134.json')]:
        old = json.loads((ROOT / 'data/baseline_manifests' / filename).read_text())['tasks']
        new = json.loads((AUTHORITY / 'alfworld' / (split + '.json')).read_text())['tasks']
        assert [(r['gamefile_rel'], r['gamefile_sha256']) for r in old] == [(r['gamefile_rel'], r['gamefile_sha256']) for r in new]
    rows = json.loads((AUTHORITY / 'alfworld/train.json').read_text())['tasks']
    assert all('/train/' in r['gamefile_rel'] for r in rows)
    assert set(sum(r['task_type'] == label for r in rows) for label in {r['task_type'] for r in rows}) == {20}


def test_formal_benchmark_config_keeps_method_caps_and_isolates_gold_and_state(tmp_path):
    base = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
    profiles = json.loads((ROOT / 'benchmark_profiles.json').read_text())['profiles']
    for benchmark in BENCHMARKS:
        profile = profiles['spreadsheet' if benchmark == 'spreadsheetbench' else benchmark]
        for seed in (42, 43, 44):
            root = tmp_path / benchmark / str(seed)
            configs = [resolved_config(base, profile, benchmark, seed, split, root, tmp_path / 'datasets', AUTHORITY, tmp_path / 'corpus') for split in ('train', 'val', 'test')]
            assert all(c['learning'] == base['learning'] and c['planning'] == base['planning'] and c['llm'] == base['llm'] for c in configs)
            assert configs[0]['data_dir'] != configs[1]['data_dir'] == configs[2]['data_dir']
            assert all(c['experiment']['seed'] == seed for c in configs)
            if benchmark != 'alfworld':
                assert len({c['harness']['evaluator_records'] for c in configs}) == 3


def test_actual_http_retry_and_raw_usage_recorded_without_private_reasoning(monkeypatch, tmp_path):
    monkeypatch.setenv('TEST_KEY', 'private-key')
    seen = []
    raw_usage = {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8, 'vendor_details': {'untouched': True}}
    def post(*args, **kwargs):
        seen.append(deepcopy(kwargs['json']))
        if len(seen) == 1:
            raise requests.Timeout('timeout')
        return Response({'model': 'actual-model', 'choices': [{'message': {'content': 'answer', 'reasoning_content': 'private-reasoning'}, 'finish_reason': 'stop'}], 'usage': raw_usage})
    monkeypatch.setattr(requests, 'post', post)
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig('https://example.invalid', 'actual-model', 'TEST_KEY', 32, max_retries=1, retry_backoff_seconds=0))
    provider.complete([{'role': 'user', 'content': 'question'}], tools=[])
    log = FormalLog(tmp_path, {'benchmark': 'fixture'}, {})
    task = PublicTask('task', 'physical', 'question')
    log.begin_task(task, 0, 'task:1', {})
    record = {'id': 'request', 'stage': 'runtime', 'phase': 'train', 'repair': 0, 'http_attempts': list(provider.request_records)}
    log.requests([record])
    log.requests([record])
    calls = log.rows('llm_calls')
    assert len(calls) == len(seen) == 2 and [r['retry_index'] for r in calls] == [0, 1]
    assert calls[0]['raw_usage'] is None and calls[1]['raw_usage'] == raw_usage
    assert calls[1]['reasoning_tokens'] is None and calls[1]['cached_tokens'] is None
    assert all(c['request_messages'] == sent['messages'] for c, sent in zip(calls, seen))
    assert 'private-reasoning' not in json.dumps(calls) and 'private-key' not in json.dumps(calls)
    supported = OpenAICompatibleProvider(OpenAICompatibleConfig('https://example.invalid', 'actual-model', 'TEST_KEY', 32, generation_seed=43))
    assert supported._build_payload([{'role': 'user', 'content': 'question'}], [])['seed'] == 43
    assert 'seed' not in seen[0]


def test_native_logging_preserves_public_feedback_exactly(tmp_path):
    log = FormalLog(tmp_path, {'benchmark': 'fixture'}, {})
    log.begin_task(PublicTask('task', 'physical', 'goal'), 0, 'task:1', {})
    plain = Broker(Adapter(), 2)
    observed = Broker(Adapter(), 2, observer=log.native_observer('task:1'))
    observed.context.scope = plain.context.scope
    assert plain.call('act', {'value': 'x'}, event_id='same-public-event') == observed.call('act', {'value': 'x'}, event_id='same-public-event')
    assert plain.events == observed.events
    assert [r['action_status'] for r in log.rows('interactions')] == ['intent', 'success']
    assert log.rows('interactions')[1]['timestamp'] and log.rows('interactions')[1]['end_time']


def test_formal_native_run_logs_no_extra_calls_and_frozen_hash_is_unchanged(monkeypatch, tmp_path):
    import atomic_skillgraph.experiments.run_empirical as runner
    monkeypatch.setattr(runner, 'code_identity', lambda: {'git_sha': 'fixture', 'tracked_dirty': False, 'source_sha256': 'fixture'})
    monkeypatch.setenv('MODEL_API_KEY', 'fixture-key')
    sent = []
    def post(*args, **kwargs):
        payload = kwargs['json']
        sent.append(payload)
        message = {'content': 'correct', 'reasoning_content': 'private'}
        reason = 'stop'
        if payload.get('tools'):
            name = payload['tools'][0]['function']['name']
            message.update(content='', tool_calls=[{'id': 'submit', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps({'decision': 'no_change'})}}])
            reason = 'tool_calls'
        return Response({'choices': [{'finish_reason': reason, 'message': message}], 'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}})
    monkeypatch.setattr(requests, 'post', post)
    train = PublicTask('qa:train', 'physical-train', 'question')
    records = {train.task_id: {'answers': ['correct']}, 'qa:test': {'answers': ['correct']}}
    config = config_for(tmp_path / 'train/bank')
    config['llm']['base_url'] = 'https://example.invalid'
    log = FormalLog(tmp_path, {'benchmark': 'fixture'}, config)
    result = run(config, [train], tmp_path / 'train', adapter=AnswerAdapter('searchqa', records), formal_log=log)
    assert result['tasks'] == 1 and len(sent) == 2
    assert 'raw_scorer_output' not in json.dumps(sent) and 'scorer_version' not in json.dumps(sent)
    frozen = tmp_path / 'train/frozen_bank'
    before = tree_identity(frozen)
    evaluation = deepcopy(config)
    evaluation['data_dir'] = str(frozen)
    evaluation['experiment']['runtime_mode'] = 'frozen'
    test = PublicTask('qa:test', 'physical-test', 'question', split='test')
    run(evaluation, [test], tmp_path / 'test', adapter=AnswerAdapter('searchqa', records), readonly=True, formal_log=log)
    assert len(sent) == len(log.rows('llm_calls')) == 3
    assert tree_identity(frozen) == before
    assert len(log.rows('episodes')) == 2 and len(log.rows('training_events')) == 2
    assert {r['training_unit_type'] for r in log.rows('training_events')} == {'trajectory', 'learning_update'}
    assert log.rows('scorer_outputs')[0]['raw_scorer_output']['em'] == 1
    assert log.rows('scorer_outputs')[0]['scorer_version']
    assert json.loads((tmp_path / 'final_frozen_manifest.json').read_text())['freeze_time']


def test_real_artifact_versions_keep_parents_and_candidates_are_not_accepted(tmp_path):
    log = FormalLog(tmp_path / 'logs', {'benchmark': 'fixture'}, {})
    log.begin_task(PublicTask('task', 'physical', 'goal'), 0, 'task:1', {})
    bank = Bank(tmp_path / 'bank')
    bank.observer = log
    first = bank.put('program', program("def run(ctx, inputs): return {'status':'ok','outputs':{}}"))
    log.requests_seen = [{'messages': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': json.dumps({'source': first['source']})}]}]
    second = bank.put('program', program("def run(ctx, inputs): return {'status':'not_found','outputs':{}}"))
    versions = log.rows('artifacts')
    assert versions[1]['parent_artifact_id'] == versions[0]['artifact_id']
    assert not any(r['accepted'] for r in versions)
    assert all(Path(r['artifact_path']).is_file() and sha256(r['artifact_path']) == r['file_sha256'] for r in versions)
    assert second['state'] == 'candidate'
    bank.close()
