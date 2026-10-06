"""CF4 differential public contracts and production HTTP fixtures; no live model."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from skillcompiler_bench_contracts.livemath import normalize_livemath_item
from skillcompiler_bench_contracts.office import GREP_SCHEMA, grep
from skillcompiler_bench_contracts import source_identity
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema, digest
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.task_context import TaskContext, progress_key
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_empirical import config_for, program, worker
from test_cf2_contracts import office, alf, http, response
from test_cf2_realization import skill, usable, learning_trace, trial_factory
from test_cf3_takeover import transport, workflow

ROOT = Path(__file__).resolve().parents[1]


def raw(choices=None, correct=None):
    return {'id': 'fixture', 'mcq': {'question': 'Which value?',
        'choices': choices or [{'label': 'B', 'text': '2'}],
        'correct_choice': correct or {'label': ' a.) ', 'text': '1'}}}


def test_t01_missing_original_choice_inserted_public_private_equal():
    item = normalize_livemath_item(raw())
    assert item['choices'] == [{'label': 'A', 'text': '1'}, {'label': 'B', 'text': '2'}]
    assert item['correct_choice'] == item['choices'][0]
    assert all(set(c) == {'label', 'text'} for c in item['choices'])


@pytest.mark.parametrize('choices', [dict(B='2', A='1'), ['1', '2'],
    [{'label': 'a:', 'text': '1'}, {'label': 'b)', 'content': '2'}]])
def test_t02_complete_and_idempotent(choices):
    item = normalize_livemath_item(raw(choices, 'a.'))
    assert normalize_livemath_item(item) == item
    assert [c['label'] for c in item['choices']] == ['A', 'B']


@pytest.mark.parametrize('changes', [
    {'choices': [{'label': 'A', 'text': '1'}, {'label': 'a)', 'text': '2'}]},
    {'correct_choice': {'label': 'A'}}, {'question': ' '}, {'choices': [None]},
    {'choices': [{'label': 'A', 'text': 'different'}]}])
def test_t03_invalid_source_is_not_guessed(changes):
    value = raw(); value['mcq'].update(changes)
    with pytest.raises(ValueError, match='LiveMath fixture:'): normalize_livemath_item(value)


def test_t04_all_177_materialized_same_ids_keys_orders():
    new = Path(os.environ['CF4_DATASETS'])
    old = Path('/home/yangchengyu/main_experiment_v1_resources_20261004')
    integrity = json.loads((new / 'livemath_integrity.json').read_text())
    assert integrity['pool_count'] == 177 and not integrity['unresolved_ids']
    assert integrity['public_choices_sha256'] == integrity['evaluator_choices_sha256']
    for split, count in [('train', 60), ('val', 17), ('test', 100)]:
        before = json.loads((old / 'livemath' / (split + '.json')).read_text())['tasks']
        after = json.loads((new / 'livemath' / (split + '.json')).read_text())['tasks']
        assert len(after) == count
        assert [(t['task_id'], t['physical_key']) for t in before] == [(t['task_id'], t['physical_key']) for t in after]
        assert (old / 'livemath' / (split + '.json')).read_bytes() != (new / 'livemath' / (split + '.json')).read_bytes()


def test_t05_actual_materialized_livemath_reaches_final_http(tmp_path, monkeypatch):
    root = Path(os.environ['CF4_DATASETS'])
    task = PublicTask(**json.loads((root / 'livemath/train.json').read_text())['tasks'][0])
    records = json.loads((root / 'livemath/evaluator_records_train.json').read_text())
    label = records[task.task_id]['correct_choice']['label']
    sent = transport(monkeypatch, [(None, '<answer>' + label + '</answer>')])
    s = EmpiricalSystem(config_for(tmp_path / 'bank'), harness=AnswerAdapter('livemath', records))
    s.run_task(task, learn=False)
    material = json.loads(sent[0]['messages'][1]['content'])
    assert material['inputs']['choices'] == records[task.task_id]['choices']
    assert not any(k in material for k in ('correct_choice', 'theorem', 'sketch'))
    s.close()


def test_t15_ours_scoped_grep_matches_public_package(tmp_path):
    s = office(tmp_path)
    pure, audit = grep(s.adapter.corpus, 'target', ['a.txt'])
    actual = s.adapter.call('grep', {'pattern': 'target', 'paths': ['a.txt']})
    assert actual['data'] == pure and actual['scope_audit']['execution_paths'] == audit['execution_paths']
    s.close()


def test_t32_single_answer_learning_keeps_full_context_without_unreadable_refs(tmp_path, monkeypatch):
    inputs = {'context': 'actual public context ' * 1000, 'version': 7, 'options': ['first', 'second']}
    sent = transport(monkeypatch, [(None, '<answer>word</answer>'), ('submit_learning', {'decision': 'no_change'})])
    s = EmpiricalSystem(config_for(tmp_path / 'bank'), harness=AnswerAdapter('searchqa', {'id': {'answers': ['word']}}))
    s.run_task(PublicTask('id', 'physical', 'Which word?', inputs), learn=True)
    material = json.loads(sent[1]['messages'][1]['content'])
    assert material['experience']['task']['inputs'] == inputs
    assert material['completed_train_cases'][0]['task']['inputs'] == inputs and material['tools'] == []
    assert len(sent) == 2 and s.bank.jobs() == []
    s.close()


@pytest.mark.parametrize('benchmark', ['livemath', 'searchqa', 'docvqa'])
def test_t05_t06_t07_actual_solver_http_contract_and_resume(tmp_path, monkeypatch, benchmark):
    item = normalize_livemath_item(raw())
    answer = '<answer>A</answer>' if benchmark == 'livemath' else '<answer>word</answer>'
    records = {'id': {'choices': item['choices'], 'correct_choice': item['correct_choice']} if benchmark == 'livemath' else {'answers': ['word']}}
    inputs = {'choices': item['choices']} if benchmark == 'livemath' else {'context': 'public full context'}
    config = config_for(tmp_path / 'bank')
    if benchmark == 'docvqa':
        image = tmp_path / 'pixels.png'; image.write_bytes(b'fixture pixels')
        inputs = {'images': [str(image)]}; config['llm']['input_modalities'] = ['text', 'image']
    sent = transport(monkeypatch, [(None, answer)])
    adapter = AnswerAdapter(benchmark, records)
    s = EmpiricalSystem(config, harness=adapter)
    s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    task = PublicTask('id', 'physical', 'Which word?', inputs, split='val')
    trace = s.run_task(task, learn=False)
    assert trace['score']['hard'] and len(sent) == 1 and not sent[0].get('tools')
    content = json.dumps(sent[0]['messages'])
    assert '<answer>' in content and 'correct_choice' not in content
    if benchmark == 'livemath': assert all(json.dumps(c) in sent[0]['messages'][1]['content'] for c in item['choices'])
    if benchmark == 'docvqa': assert 'data:image/png;base64,' in content
    s.run_task(task, learn=False)
    assert len(sent) == 1
    s.close()


def test_t08_t09_t10_t12_scope_offsets_limit_and_budget(tmp_path):
    s = office(tmp_path)
    (s.adapter.corpus / 'b.txt').write_bytes(('α target\r\n' * 50).encode())
    (s.adapter.corpus / 'z.txt').write_text('later target')
    unscoped = s.adapter.call('grep', {'pattern': 'target'})
    assert len(unscoped['data']) == 40 and unscoped['data'][0]['path'] == 'a.txt'
    late = s.adapter.call('grep', {'pattern': 'target', 'paths': ['z.txt']})
    assert late['data'] == [{'path': 'z.txt', 'line': 1, 'offset': 0, 'text': 'later target'}]
    broker = Broker(s.adapter, 24)
    result = broker.call('grep', {'pattern': 'target', 'paths': ['z.txt', './b.txt', 'b.txt']})
    assert len(broker.events) == 1 and broker.remaining_calls() == 23
    assert result['scope_audit']['execution_paths'] == ['b.txt', 'z.txt'] and len(result['data']) == 40
    for hit in result['data'][:3]:
        assert s.adapter.call('read', {'path': hit['path'], 'offset': hit['offset']})['data']['text'].startswith(hit['text'])
    assert s.adapter.call('grep', {'pattern': 'target', 'paths': []})['data'] == []
    assert s.adapter.failure_arguments('grep', {'pattern': 'x', 'paths': ['./z.txt', 'b.txt', 'b.txt']}) == s.adapter.failure_arguments('grep', {'pattern': 'x', 'paths': ['b.txt', 'z.txt']})
    assert s.adapter.failure_arguments('grep', {'pattern': 'x'}) != s.adapter.failure_arguments('grep', {'pattern': 'x', 'paths': []})
    s.close()


@pytest.mark.parametrize('paths', [None, 'a.txt', [''], ['../a.txt'], ['/tmp/a.txt'], ['missing.txt'], ['a.txt', 'bad.json'], ['.']])
def test_t11_scope_rejected_before_any_read(tmp_path, monkeypatch, paths):
    s = office(tmp_path)
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path.is_relative_to(s.adapter.corpus): pytest.fail('Invalid scope must be rejected before reading')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert not s.adapter.call('grep', {'pattern': 'target', 'paths': paths})['accepted']
    assert not s.adapter.call('grep', {'pattern': '[', 'paths': ['a.txt']})['accepted']
    s.close()


def test_t11_io_unicode_symlink_root_faults_remain_infrastructure(tmp_path, monkeypatch):
    s = office(tmp_path)
    outside = tmp_path / 'outside.txt'; outside.write_text('secret')
    (s.adapter.corpus / 'escape.txt').symlink_to(outside)
    assert not s.adapter.call('grep', {'pattern': 'secret', 'paths': ['escape.txt']})['accepted']
    (s.adapter.corpus / 'invalid.txt').write_bytes(b'\xff')
    with pytest.raises(UnicodeError): s.adapter.call('grep', {'pattern': 'x', 'paths': ['invalid.txt']})
    def unreadable(*a, **k): raise PermissionError('fixture I/O fault')
    monkeypatch.setattr(Path, 'read_text', unreadable)
    with pytest.raises(PermissionError): s.adapter.call('grep', {'pattern': 'x', 'paths': ['a.txt']})
    s.close()


def test_t13_scoped_read_batch_resume_no_repeat(tmp_path, monkeypatch):
    s = office(tmp_path); s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint'); s.executor.checkpoint = s.checkpoint
    calls = [dict(action='call_tool', name='grep', arguments={'pattern': 'target', 'paths': ['a.txt']}) for _ in range(3)]
    sent = transport(monkeypatch, [('runtime_step', calls), ('runtime_step', {'action': 'finish', 'answer': 'private'})])
    s.executor.run(s.adapter.task, s.adapter, Broker(s.adapter, 24, context=s.task_context, journal=s.checkpoint.native_events), dynamic(s.adapter.task))
    events = json.loads((s.checkpoint.root / 'native_events.json').read_text())
    assert len(events) == 3 and len(sent) == 2
    broker = Broker(s.adapter, 24)
    broker.events = events
    for event in events:
        assert broker.call(event['name'], event['arguments'], event_id=event['event_id']) == event['result']
    assert broker.remaining_calls() == 21
    s.close()


def test_t14_same_schema_in_planner_runtime_builder_http(tmp_path, monkeypatch):
    s = office(tmp_path); s.bank.put('skill', skill('find target'))
    sent = transport(monkeypatch, [('submit_plan', {'mode': 'compose', 'workflow': dynamic(s.adapter.task)}),
        ('runtime_step', {'action': 'finish', 'answer': 'private'}), ('submit_program', {'source': 'def run(ctx, inputs): pass', 'trial_inputs': []})])
    plan = s.planner.plan(s.adapter.task, s.adapter)
    s.executor.run(s.adapter.task, s.adapter, Broker(s.adapter, 24, context=s.task_context), plan)
    from atomic_skillgraph.empirical.prompts import BUILD
    s.agent('tool_builder', 'Build', {'tools': s.adapter.tool_definitions()}, 'submit_program', BUILD)
    for index, payload in enumerate(sent):
        material = json.loads(payload['messages'][1]['content'])
        tools = material['tool_definitions'] if index == 0 else material['calls']['tools'] if index == 1 else material['tools']
        assert next(t for t in tools if t['name'] == 'grep')['input_schema'] == GREP_SCHEMA
    s.close()


def binding(case_id, resource='a.txt'):
    return {'case_id': case_id, 'inputs': {'resource': resource}, 'start_mode': 'reset', 'prefix': []}


def seed_job(s, state='usable'):
    sk = s.bank.put('skill', skill('find target', inputs=object_schema({'resource': {'type': 'string'}}, ['resource']), execution_intent='program_requested'))
    asset = program("def run(ctx, inputs): return {'status':'needs_input','outputs':{}}", inputs=sk['input_schema'])
    p = usable(s.bank, asset) if state == 'usable' else s.bank.put('program', asset)
    s.bank.put('implementation', {'skill_id': sk['id'], 'program_id': p['id']})
    for key in ('k0', 'k1'): s.bank.save_case(PublicTask('office', key, 'find target'), learning_trace())
    job = {'id': digest(['realization', sk['id']]), 'skill_id': sk['id'], 'skill_version': sk['id'],
        'kind': 'trial', 'state': 'done', 'program_id': p['id'], 'case_bindings': [binding('k0'), binding('k1')],
        'repair_used': False, 'generation_count': 1, 'epoch': 0, 'last_error_kind': None, 'trigger_task': 'k0'}
    s.bank.save_job(job)
    return sk, p, job


def test_t16_t17_t19_usable_third_case_skips_but_saves_workflow_and_resume(tmp_path, monkeypatch):
    s = office(tmp_path); sk, p, job = seed_job(s)
    proposal = {'decision': 'reuse_existing', 'existing_skill_id': sk['id'],
        'realization_request': {'skill_id': sk['id'], 'action': 'trial', 'case_bindings': [binding('office-physical'), binding('unused-case'), binding('another-unused-case')]},
        'workflow': workflow([{'id': 'gap', 'execution_mode': 'dynamic', 'goal': 'find target', 'args': {}, 'reference_skill_ids': [sk['id']]}])}
    sent = transport(monkeypatch, [('submit_learning', proposal)])
    s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    before = s.bank.attempts(p['id'])
    for _ in range(2):
        result = s.learner.learn(s.adapter.task, learning_trace())
        assert result['realization_skipped'] == 'already_usable'
        assert s.bank.jobs()[0] == job and s.bank.attempts(p['id']) == before
    assert len(sent) == 1 and len(s.bank.all('workflow')) == 1 and len(s.bank.train_cases()) == 3
    s.close()


def test_t18_candidate_explicit_repair_new_version_not_skipped(tmp_path):
    s = office(tmp_path); sk, p, job = seed_job(s, 'candidate')
    proposal = {'decision': 'reuse_existing', 'existing_skill_id': sk['id'],
        'realization_request': {'skill_id': sk['id'], 'action': 'trial', 'case_bindings': [binding('k0')]}}
    assert not s.learner._resolve_realization_request(proposal)['already_usable']
    for i in range(2): s.bank.record({'id': 'positive' + str(i), 'program_id': p['id'], 'task_key': 'k' + str(i), 'origin': 'train_test', 'outcome': 'positive', 'basis': 'local_check'})
    proposal['realization_request']['action'] = 'repair'
    assert not s.learner._resolve_realization_request(proposal)['already_usable']
    proposal.update(skill=skill('new version', execution_intent='program_requested'))
    proposal['realization_request'].update(skill_id='$new', action='build')
    assert not s.learner._resolve_realization_request(proposal)['already_usable']
    s.close()


def test_t20_t22_missing_input_repaired_to_defer_before_asset_write(tmp_path, monkeypatch):
    s = office(tmp_path)
    proposal = {'decision': 'propose_skill_and_program_spec', 'skill': skill('find target', inputs=object_schema({'resource': {'type': 'string'}}, ['resource'])),
        'realization_request': {'skill_id': '$new', 'action': 'build', 'case_bindings': [binding('office-physical')]}}
    proposal['realization_request']['case_bindings'][0]['inputs'] = {}
    def repaired():
        assert s.bank.all('skill') == [] and s.bank.jobs() == []
        value = deepcopy(proposal); value['realization_request'].update(action='defer', case_bindings=[])
        return value
    sent = transport(monkeypatch, [('submit_learning', proposal), ('submit_learning', repaired)])
    log = s.learner.learn(s.adapter.task, learning_trace())
    assert len(sent) == 2 and log['program'] is None and s.bank.jobs()[0]['state'] == 'waiting_example'
    repair = json.loads(sent[1]['messages'][1]['content'])
    assert repair['error']['code'] == 'trial_binding_invalid' and repair['error']['missing_fields'] == ['resource']
    assert len(s.bank.all('skill')) == 1
    s.close()


def test_t21_t24_binding_repair_then_two_actual_train_trials(tmp_path, monkeypatch):
    s = office(tmp_path); trial_factory(s)
    sk = skill('find target', inputs=object_schema({'resource': {'type': 'string'}}, ['resource']),
        outputs=object_schema({'answer': {'type': 'string'}}, ['answer']), result_role='final_answer')
    valid = {'decision': 'propose_skill_and_program_spec', 'skill': sk,
        'realization_request': {'skill_id': '$new', 'action': 'build', 'case_bindings': [binding('office-physical')]}}
    invalid = deepcopy(valid); invalid['realization_request']['case_bindings'][0]['inputs'] = {}
    generated = {'source': "def run(ctx, inputs): return {'status':'ok','outputs':{'answer':'private'}}", 'trial_inputs': [binding('office-physical')]}
    sent = transport(monkeypatch, [('submit_learning', invalid), ('submit_learning', valid), ('submit_program', generated)])
    result = s.learner.learn(s.adapter.task, learning_trace()); p = result['program']
    assert len(sent) == 3 and s.bank.get(p)['state'] == 'candidate'
    skill_id = s.bank.all('skill')[0]['id']
    request = {'decision': 'reuse_existing', 'existing_skill_id': skill_id, 'realization_request': {'skill_id': skill_id, 'action': 'trial', 'case_bindings': [binding('second')]}}
    second = transport(monkeypatch, [('submit_learning', request)])
    s.learner.learn(PublicTask('office', 'second', 'find target'), learning_trace())
    assert len(second) == 1 and s.bank.get(p)['state'] == 'usable'
    assert len({a['task_key'] for a in s.bank.attempts(p) if a['outcome'] == 'positive'}) == 2
    s.close()


@pytest.mark.parametrize('mode', ['unknown', 'val', 'prefix', 'reset_prefix', 'type'])
def test_t23_invalid_binding_never_becomes_evidence(tmp_path, mode):
    s = office(tmp_path); sk, p, job = seed_job(s, 'candidate')
    case = binding('k0')
    if mode == 'unknown': case['case_id'] = 'missing'
    elif mode == 'val':
        cases = {'k0': (PublicTask('office', 'k0', 'find target', split='val'), learning_trace())}
    elif mode in ('prefix', 'reset_prefix'):
        case.update(start_mode='prefix_replay' if mode == 'prefix' else 'reset', prefix=[{'name': 'read', 'arguments': {'path': 'a.txt', 'offset': 0}}])
    else: case['inputs']['resource'] = 42
    proposal = {'decision': 'reuse_existing', 'existing_skill_id': sk['id'], 'realization_request': {'skill_id': sk['id'], 'action': 'trial', 'case_bindings': [case]}}
    with pytest.raises(ValueError, match='trial_binding_invalid'):
        s.learner.validate_learning_proposal(proposal, cases if mode == 'val' else None)
    assert s.bank.jobs()[0] == job and not s.bank.attempts(p['id'])
    s.close()


def test_t25_t27_dynamic_card_reports_binding_without_authorizing_route(tmp_path, monkeypatch):
    s = office(tmp_path); sk, p, _ = seed_job(s)
    old = s.bank.put('workflow', workflow([{'id': 'gap', 'execution_mode': 'dynamic', 'goal': 'find target', 'args': {}, 'reference_skill_ids': [sk['id']]}]))
    before = s.bank.digest()
    cards = s.bank.planning_cards('find target')
    card = next(c for c in cards if c['id'] == old['id'])
    assert card['execution_summary']['dynamic_nodes'] == 1
    assert sum(card['execution_summary'].values()) == 1 and s.bank.routes(old['nodes'][0]) == []
    sent = transport(monkeypatch, [('submit_plan', {'mode': 'compose', 'workflow': dynamic(s.adapter.task)})])
    s.planner.plan(s.adapter.task, s.adapter)
    material = json.loads(sent[0]['messages'][1]['content'])
    assert next(c for c in material['related_interfaces'] if c['id'] == old['id']) == card
    assert material['program_options'][0]['id'] == p['id'] and s.bank.digest() == before
    s.close()


def test_t29_t30_t31_projection_preserves_real_values_refs_and_checkpoint(tmp_path, monkeypatch):
    s = office(tmp_path)
    literal = {'literal': {'from': 'business', 'version': 7, 'state': 'public', 'location': 'here', 'text': 'x' * 10000}}
    task = PublicTask('office', 'office-physical', 'Original goal, two required items', {'resource': literal, 'constraint': 'two items'})
    s.adapter.reset(task)
    s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint'); s.executor.checkpoint = s.checkpoint
    def answer():
        assert s.checkpoint.state['model_context']['results']
        return {'action': 'finish', 'answer': 'private'}
    sent = transport(monkeypatch, [('runtime_step', answer)])
    plan = dynamic(task); plan['nodes'][0]['args'] = {'resource': {'task': 'resource'}}
    before = s.adapter.progress_key()
    s.executor.run(task, s.adapter, Broker(s.adapter, 24, context=s.task_context), plan)
    material = json.loads(sent[0]['messages'][1]['content'])
    ref = material['bindings']['inputs']['resource']
    assert s.task_context.resolve(ref) == literal
    assert material['task']['inputs']['constraint'] == 'two items' and material['task']['goal'] == task.goal
    assert material['calls']['tools'][:-1] == s.adapter.available_tools()
    assert s.adapter.progress_key() == before
    long = s.task_context.preview('long ' * 2000)
    assert long['ref'] and s.task_context.resolve(long['ref']) == 'long ' * 2000
    s.close()


def test_t33_alf_model_task_omits_only_backend_and_keeps_state_catalog():
    a = alf(); task = a.task
    task.inputs['environment_task'] = {'env_index': 8, 'task_type': 'pick', 'native_task_id': 'backend',
        'context': {'goal_roles': {'target': 'apple'}, 'gamefile': '/backend/path', 'task_signature': 'private plumbing'}}
    model = a.model_task()
    assert model['inputs']['environment_task']['goal_roles'] == {'target': 'apple'}
    assert not any(k in json.dumps(model) for k in ('env_index', 'gamefile', 'native_task_id', 'task_signature'))
    from atomic_skillgraph.empirical.model_view import model_task
    assert model_task(task, a) == model and task.inputs['environment_task']['env_index'] == 8
    assert progress_key(a) == a.progress_key()
    a.close()


def test_t38_public_wheel_contains_no_ours_and_installs_alone(tmp_path):
    subprocess.run([sys.executable, str(ROOT / 'tools/build_benchmark_contracts.py'), '--output', str(tmp_path / 'wheel')], check=True, capture_output=True)
    wheel = next((tmp_path / 'wheel').glob('*.whl'))
    manifest = json.loads((tmp_path / 'wheel/manifest.json').read_text())
    assert manifest['benchmark_contracts_sha256'] == source_identity()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(n.startswith(('skillcompiler_bench_contracts/', 'skillcompiler_bench_contracts-')) for n in names)
        assert not any('atomic_skillgraph' in archive.read(n).decode() for n in names if n.endswith('.py'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(tmp_path / 'install'), str(wheel)], check=True, capture_output=True)
    check = "import sys; sys.path.insert(0, sys.argv[1]); from skillcompiler_bench_contracts.livemath import normalize_livemath_item; assert 'atomic_skillgraph' not in sys.modules"
    subprocess.run([sys.executable, '-I', '-c', check, str(tmp_path / 'install')], check=True)
