"""CF4-R1 production paths, intercepted HTTP and locked Docker; no paid model calls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import yaml

from atomic_skillgraph.empirical import IMPLEMENTATION_REVISION
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.program_worker import program_permission_view
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments.formal_log import FormalLog
from atomic_skillgraph.harness.benchmarks import AnswerAdapter, SpreadsheetAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http, office, response
from test_cf2_realization import generated, learning_trace, requested_proposal, trial_factory
from test_cf3_takeover import transport
from test_empirical import config_for, program, worker

ROOT = Path(__file__).resolve().parents[1]


def finish_fixture(tmp_path, monkeypatch, final):
    sent = http(monkeypatch, [response([{'action': 'call_tool', 'name': 'read',
        'arguments': {'path': 'a.txt', 'offset': 0}}]), final])
    s = office(tmp_path)
    s.checkpoint = s.executor.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    s.audit_path = tmp_path / 'usage.json'
    broker = Broker(s.adapter, 1, context=s.task_context, journal=s.checkpoint.native_events)
    return s, broker, sent


@pytest.mark.parametrize('answer,finish', [('private', 'stop'), ('private', 'length'), ('wrong', 'stop')])
def test_r01_r03_finish_text_reaches_unchanged_scorer(tmp_path, monkeypatch, answer, finish):
    text = '<answer>' + answer + '</answer>'
    s, broker, sent = finish_fixture(tmp_path, monkeypatch, response(content=text, finish=finish))
    result = s.executor.run(s.adapter.task, s.adapter, broker, dynamic(s.adapter.task))
    assert result['prediction'] == text and result['reason'] == 'finish_only'
    assert s.adapter.evaluate(s.adapter.submit(result['prediction']))['hard'] == (answer == 'private')
    assert len(sent) == 2 and len(broker.events) == 1
    assert not sent[-1].get('tools') and 'tool_choice' not in sent[-1]
    assert s.requests[-1]['purpose'] == 'finish_only' and s.requests[-1]['response']['finish_reason'] == finish
    assert sum(e.to_dict()['total_tokens'] for e in s.usage.events) == 10
    assert s.checkpoint.state['decisions'][s.last_decision_id]['status'] == 'applied'
    s.close()


@pytest.mark.parametrize('final,reason', [
    (response(content=' ', finish='stop'), 'finish_only_empty_answer'),
    (response(content=None, finish='stop'), 'finish_only_empty_answer'),
    (response(content={'answer': 'private'}, finish='stop'), 'finish_only_invalid_text'),
    (response([{'pattern': 'target'}], name='grep'), 'finish_only_unexpected_tool_call')])
def test_r02_rejected_finish_does_not_execute_or_repair(tmp_path, monkeypatch, final, reason):
    s, broker, sent = finish_fixture(tmp_path, monkeypatch, final)
    result = s.executor.run(s.adapter.task, s.adapter, broker, dynamic(s.adapter.task))
    assert result['reason'] == reason and result['prediction'] == ''
    assert not s.adapter.evaluate(s.adapter.submit(result['prediction']))['hard']
    assert len(sent) == 2 and len(broker.events) == 1
    assert sum(e.to_dict()['total_tokens'] for e in s.usage.events) == 10
    assert s.checkpoint.state['decisions'][s.last_decision_id]['status'] == 'rejected'
    s.close()


@pytest.mark.parametrize('boundary', ['response_saved', 'answer_committed'])
def test_r04_finish_checkpoint_recovers_same_decision_once(tmp_path, monkeypatch, boundary):
    s, broker, sent = finish_fixture(tmp_path, monkeypatch, response(content='private', finish='stop'))
    checkpoint = s.checkpoint
    if boundary == 'response_saved':
        original = checkpoint.save_response
        def stop(key, value):
            original(key, value)
            if value['turn']['content'] == 'private':
                raise KeyboardInterrupt('after durable response')
        monkeypatch.setattr(checkpoint, 'save_response', stop)
    else:
        original = checkpoint.commit_decision
        def stop(key, status, **values):
            original(key, status, **values)
            if key and checkpoint.state['decisions'][key]['purpose'] == 'finish_only' and status == 'applied':
                raise KeyboardInterrupt('after atomic answer commit')
        monkeypatch.setattr(checkpoint, 'commit_decision', stop)
    with pytest.raises(KeyboardInterrupt):
        s.executor.run(s.adapter.task, s.adapter, broker, dynamic(s.adapter.task))
    decision_id = s.last_decision_id
    monkeypatch.undo()
    result = s.executor.run(s.adapter.task, s.adapter, broker, dynamic(s.adapter.task))
    assert result['prediction'] == 'private' and result['reason'] == 'finish_only'
    assert len(sent) == 2 and len(broker.events) == 1 and s.last_decision_id == decision_id
    assert sum(e.to_dict()['total_tokens'] for e in s.usage.events) == 10
    decisions = checkpoint.state['decisions']
    assert len(decisions) == 2 and decisions[decision_id]['status'] == 'applied'
    assert s.executor.run(s.adapter.task, s.adapter, broker, dynamic(s.adapter.task)) == result
    s.close()


def test_r07_same_permission_source_and_broker_denies_recursive_python(tmp_path):
    config = config_for(tmp_path / 'bank')
    a = SpreadsheetAdapter({'id': {'public_files': {}}}, config)
    a.reset(PublicTask('id', 'physical', 'write files'))
    context = TaskContext(config['runtime'])
    definitions = a.tool_definitions()
    before = deepcopy(definitions)
    permissions = program_permission_view(definitions, [*a.available_tools(), context.tool()],
        [context.tool()], tool_surface=a.capabilities.tool_surface, workspace=a.workspace)
    assert definitions == before and permissions['allowed_names'] == ['read_result']
    assert not any(t['name'] == 'execute_python' for t in permissions['public_program_abi']['tool_examples'])
    broker = Broker(a, 4, context=context)
    assert 'execute_python' in {t['name'] for t in broker.available_tools()}
    broker.open_lease('id')
    ready = broker.rpc('id', 0, 'available_tools', {}, deadline=time.monotonic() + 2,
        allowed_tools=['execute_python', 'read_result'])
    assert [t['name'] for t in ready] == ['read_result']
    rejected = broker.rpc('id', 1, 'call', {'name': 'execute_python', 'arguments': {}},
        deadline=time.monotonic() + 2, allowed_tools=['execute_python'])
    assert not rejected['accepted'] and not broker.events[-1]['backend_invoked']
    assert broker.remaining_calls() == 3
    a.close()


@pytest.mark.parametrize('first', ['wrong_read', 'wrong_grep', 'multiple', 'length'])
def test_r06_r08_builder_generations_have_one_submission_tool(tmp_path, monkeypatch, first):
    first_turn = response([{'path': 'a.txt', 'offset': 0}], name='read') if first == 'wrong_read' else (
        response([{'pattern': 'target'}], name='grep') if first == 'wrong_grep' else
        response([generated(), generated()], name='submit_program') if first == 'multiple' else response(finish='length'))
    sent = http(monkeypatch, [response([requested_proposal()], name='submit_learning'), first_turn,
                             response([generated()], name='submit_program')])
    s = office(tmp_path); trial_factory(s)
    log = s.learner.learn(s.adapter.task, learning_trace())
    job = s.bank.jobs()[0]
    assert job['generation_count'] == 2 and job['repair_used'] and len(sent) == 3
    materials = []
    for payload in sent[1:]:
        assert [t['function']['name'] for t in payload['tools']] == ['submit_program']
        materials.append(json.loads(payload['messages'][1]['content']))
    assert materials[0]['future_program_api'] == materials[1]['future_program_api']
    p = s.bank.all('program')[0]
    assert p['allowed_tools'] == materials[0]['future_program_api']['allowed_names']
    assert p['allowed_tools'] == ['glob', 'grep', 'read', 'read_result']
    assert materials[1]['previous_failure']['domain'] == 'builder_generation'
    assert materials[0]['examples'][0]['case_id'] == 'office-physical'
    if first != 'length': assert 'builder_submission_tool_mismatch' in log['errors'][0]
    assert sent[-1]['max_tokens'] == (65536 if first == 'length' else 32768)
    assert sum(e.to_dict()['total_tokens'] for e in s.usage.events) == 15
    s.close()


def test_r09_durable_spreadsheet_runs_local_python_and_seals_files(tmp_path, worker):
    config = config_for(tmp_path / 'bank')
    a = SpreadsheetAdapter({'id': {'public_files': {}}}, config)
    a.reset(PublicTask('id', 'physical', 'make a workbook'))
    context = TaskContext(config['runtime'])
    permissions = program_permission_view(a.tool_definitions(), a.available_tools(), [context.tool()], workspace=a.workspace)
    source = """def run(ctx, inputs):
    from pathlib import Path
    from openpyxl import Workbook
    assert 'INPUT_PATH' not in globals() and 'OUTPUT_PATH' not in globals()
    assert not any(t['name'] == 'execute_python' for t in ctx.available_tools())
    rejected = ctx.call('execute_python', {'source': 'raise AssertionError()', 'files': []})
    assert not rejected['accepted']
    wb = Workbook(); wb.active['A1'] = 7; wb.save('/workspace/case1_result.xlsx')
    Path('/workspace/solution.py').write_text("from openpyxl import Workbook\\nwb=Workbook()\\nwb.active['A1']=7\\nwb.save(OUTPUT_PATH)\\n")
    return {'status': 'ok', 'outputs': {'files': ['solution.py', 'case1_result.xlsx']}}
"""
    bank = Bank(tmp_path / 'programs')
    candidate = program(source, outputs=object_schema({'files': {'type': 'array', 'items': {'type': 'string'}}}, ['files']))
    candidate['allowed_tools'] = permissions['allowed_names']
    p = bank.put('program', candidate)
    broker = Broker(a, 4, context=context)
    result = worker.execute(p, {}, broker)
    assert result['status'] == 'ok' and result['outputs']['files'] == ['solution.py', 'case1_result.xlsx']
    assert len(broker.events) == 1 and not broker.events[0]['backend_invoked']
    sealed = a.submit(result['outputs'])
    assert sealed['bundle'] and Path(sealed['bundle'], 'solution.py').is_file()
    before = a.observe()['workspace']
    bad = deepcopy(candidate); bad['source'] = "def run(ctx, inputs):\n    return {'status':'ok','outputs':{'files':['/workspace/solution.py']}}"
    rejected = worker.execute(bank.put('program', bad), {}, Broker(a, 4, context=context))
    assert rejected['status'] == 'execution_error' and a.observe()['workspace'] == before
    bank.close(); a.close()


def guidance_proposal(text='Check qualifiers in the complete public question.', parent=None):
    p = {'decision': 'upsert_guidance', 'guidance_skill': {'goal': 'math question', 'guidance': text}}
    if parent: p['existing_skill_id'] = parent
    return p


def qa_system(tmp_path):
    return EmpiricalSystem(config_for(tmp_path / 'bank'), harness=AnswerAdapter('searchqa', {
        'train': {'answers': ['word']}, 'next': {'answers': ['word']}, 'val': {'answers': ['word']}}))


def test_r11_r14_guidance_actual_learn_revision_freeze_inject_and_replay(tmp_path, monkeypatch):
    sent = transport(monkeypatch, [(None, '<answer>word</answer>'), ('submit_learning', guidance_proposal())])
    s = qa_system(tmp_path)
    formal = FormalLog(tmp_path / 'formal', {'fixture': 'controlled'}, s.config)
    task = PublicTask('train', 'train-physical', 'math question', {'context': 'complete public context'})
    formal.begin_task(task, 0, 'first', {})
    s.observer = s.bank.observer = formal
    s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint')
    first = s.run_task(task, learn=True, attempt_id='first')
    old = s.bank.all('skill')[0]
    assert first['learning']['persisted_skill_id'] == old['id']
    assert not s.bank.jobs() and not s.bank.all('workflow') and not s.bank.all('implementation') and not s.worker.invocations
    assert [r['stage'] for r in s.requests] == ['runtime', 'extractor'] and len(sent) == 2
    assert s.run_task(task, learn=True, attempt_id='first')['learning'] == first['learning'] and len(sent) == 2
    # Interrupt after the revision asset is committed; checkpoint proposal must recreate the same content ID.
    task2 = PublicTask('next', 'next-physical', 'math question')
    formal.begin_task(task2, 1, 'second', {})
    second_sent = transport(monkeypatch, [(None, '<answer>word</answer>'),
        ('submit_learning', guidance_proposal('Compare every qualifier before selecting an answer.', old['id']))])
    s.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint2')
    put = s.bank.put
    def stop(kind, asset):
        saved = put(kind, asset)
        if kind == 'skill' and asset.get('parent_skill_id'):
            raise KeyboardInterrupt('after guidance persistence')
        return saved
    monkeypatch.setattr(s.bank, 'put', stop)
    with pytest.raises(KeyboardInterrupt): s.run_task(task2, learn=True, attempt_id='second')
    monkeypatch.setattr(s.bank, 'put', put)
    second = s.run_task(task2, learn=True, attempt_id='second')
    child = s.bank.get(second['learning']['persisted_skill_id'])
    assert child['id'] != old['id'] and child['parent_skill_id'] == old['id'] and s.bank.get(old['id']) == old
    assert len(second_sent) == 2 and len(s.bank.all('skill')) == 2
    assert [a['id'] for a in s.bank.retrieve_guidance('math question')] == [child['id']]
    events = formal.rows('artifacts')
    child_event = next(e for e in events if e['logical_asset_id'] == child['id'])
    assert child_event['parent_artifact_id'] == next(e['artifact_id'] for e in events if e['logical_asset_id'] == old['id'])
    updates = [e for e in formal.rows('training_events') if e['training_unit_type'] == 'learning_update']
    assert updates[-1]['persisted_skill_id'] == child['id'] and updates[-1]['parent_skill_id'] == old['id']
    frozen = tmp_path / 'frozen'; s.bank.freeze(frozen); s.close()
    config = config_for(frozen)
    val_sent = transport(monkeypatch, [(None, '<answer>word</answer>')])
    val = EmpiricalSystem(config, harness=AnswerAdapter('searchqa', {'val': {'answers': ['word']}}), readonly=True)
    trace = val.run_task(PublicTask('val', 'val-physical', 'math question', split='valid_seen'), learn=False)
    payload = json.loads(val_sent[0]['messages'][1]['content'])
    assert payload['guidance'] == [{'skill_id': child['id'], 'goal': child['goal'], 'guidance': child['guidance']}]
    assert trace['injected_guidance_ids'] == [child['id']] and trace['knowledge_before'] == trace['knowledge_after']
    assert len(val_sent) == 1 and trace['score']['hard']; val.close()


@pytest.mark.parametrize('decision', ['no_change', 'reuse', 'identical', 'workflow', 'job', 'invented', 'empty'])
def test_r12_r13_no_change_reuse_reject_have_no_partial_assets(tmp_path, monkeypatch, decision):
    s = qa_system(tmp_path)
    old = s.bank.put('skill', {**guidance_proposal()['guidance_skill'], 'execution_intent': 'guidance_only',
        'result_role': 'final_answer', 'input_schema': object_schema(), 'output_schema': object_schema()})
    proposals = {'no_change': {'decision': 'no_change'},
        'reuse': {'decision': 'reuse_existing', 'existing_skill_id': old['id']},
        'identical': guidance_proposal(parent=old['id']),
        'workflow': {'decision': 'no_change', 'workflow': {'nodes': []}},
        'job': {'decision': 'no_change', 'realization_request': {}},
        'invented': {'decision': 'reuse_existing', 'existing_skill_id': 'invented'},
        'empty': guidance_proposal(' ')}
    invalid = decision in {'workflow', 'job', 'invented', 'empty'}
    sent = transport(monkeypatch, [(None, '<answer>wrong</answer>'),
        *[('submit_learning', proposals[decision])] * (2 if invalid else 1)])
    task = PublicTask('train', 'train-physical', 'math question')
    formal = FormalLog(tmp_path / 'formal', {'fixture': decision}, s.config)
    formal.begin_task(task, 0, 'first', {})
    s.observer = s.bank.observer = formal
    trace = s.run_task(task, learn=True)
    assert not trace['score']['hard'] and len(sent) == (3 if invalid else 2)
    assert s.bank.all('skill') == [old] and not s.bank.all('workflow') and not s.bank.all('program') and not s.bank.jobs()
    log = trace['learning']
    assert (log['decision'] == 'rejected') == invalid
    if invalid:
        assert log['learning_rejected_reason'] and formal.rows('training_events')[-1]['training_event_status'] == 'rejected'
    if decision in {'reuse', 'identical'}: assert log['reused_skill_id'] == old['id']
    s.close()


def test_r15_guidance_filter_before_top_k_and_frozen_query_pure(tmp_path):
    bank = Bank(tmp_path / 'bank')
    for i in range(8): bank.put('workflow', {'goal': 'math question', 'nodes': [{'id': str(i), 'goal': 'math question', 'args': {}}]})
    for i in range(4): bank.put('skill', {'goal': 'math question', 'guidance': '', 'execution_intent': 'guidance_only', 'index': i})
    bank.put('skill', {'goal': 'math question', 'guidance': 'program', 'execution_intent': 'program_requested'})
    sk = bank.put('skill', {'goal': 'math question', 'guidance': 'Check assumptions.', 'execution_intent': 'guidance_only'})
    assert bank.retrieve_guidance('math question', limit=3) == [sk]
    bank.freeze(tmp_path / 'frozen'); bank.close()
    frozen = Bank(tmp_path / 'frozen', readonly=True)
    before = frozen.digest()
    assert frozen.retrieve_guidance('math question', limit=3) == [sk] and frozen.digest() == before
    frozen.close()


def test_r22_scientific_configuration_identical_to_cf4():
    for relative in ('configs/default.yaml', 'configs/alfworld_empirical_seed42.yaml',
                     'configs/main_experiment_v1.yaml', 'benchmark_profiles.json', 'models.lock.json'):
        raw = subprocess.check_output(['git', 'show', '65718eb:' + relative], cwd=ROOT, text=True)
        old, new = yaml.safe_load(raw), yaml.safe_load((ROOT / relative).read_text())
        if relative.startswith('configs/') and 'experiment' in old:
            assert old['experiment']['implementation_revision'] == 'empirical-v3.1-CF4'
            assert new['experiment']['implementation_revision'] == IMPLEMENTATION_REVISION
            old['experiment'].pop('implementation_revision'); new['experiment'].pop('implementation_revision')
        assert old == new
    assert subprocess.check_output(['git', 'diff', '65718eb', '--', 'data/main_experiment_v1'], cwd=ROOT) == b''


def test_r19_r20_launcher_four_isolated_parallel_cells_and_failure(tmp_path):
    repo = tmp_path / 'code'; repo.mkdir()
    shutil.copytree(ROOT / 'src', repo / 'src')
    shutil.copytree(ROOT / 'configs', repo / 'configs')
    shutil.copy(ROOT / 'models.lock.json', repo)
    for args in (['init', '-q'], ['add', '.'], ['-c', 'user.email=fixture@example.invalid', '-c', 'user.name=fixture', 'commit', '-qm', 'fixture']):
        subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    shim = tmp_path / 'python-shim'
    shim.write_text('#!' + sys.executable + '\n' + '''import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
if '--benchmark' not in args:
    raise SystemExit(subprocess.call([sys.executable, *args]))
b = args[args.index('--benchmark') + 1]
root = pathlib.Path(args[args.index('--output') + 1]); root.mkdir(parents=True)
assert '--stop-after-val' in args and '--resume' not in args
start = time.time(); time.sleep(0.8)
for name in ('bank', 'checkpoint'): (root / name).mkdir()
(root / 'completion.json').write_text(json.dumps({'status': 'awaiting_test', 'fixture': True}))
(root / 'invocation.json').write_text(json.dumps({'start': start, 'end': time.time(), 'tmp': os.environ['TMPDIR']}))
raise SystemExit(7 if b == 'officeqa' else 0)
''')
    shim.chmod(0o755)
    data = tmp_path / 'data'; data.mkdir(); (data / 'materialization.json').write_text('{}')
    corpus = tmp_path / 'corpus'; corpus.mkdir()
    env_file = tmp_path / '.env'; env_file.touch()
    output = tmp_path / 'output'
    env = {**os.environ, 'CODE': str(repo), 'PYTHON': str(shim), 'DATASETS': str(data),
        'CORPUS_ROOT': str(corpus), 'ENV_FILE': str(env_file), 'OUTPUT_ROOT': str(output), 'EXPECTED_SHA': sha}
    script = ROOT / 'scripts/run_non_alfworld_formal.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    dry = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True, check=True)
    assert len([line for line in dry.stdout.splitlines() if line.startswith('(cd ')]) == 4 and not output.exists()
    result = subprocess.run(['bash', str(script), '--execute'], env=env, text=True, capture_output=True)
    assert result.returncode == 1, result.stderr
    records = [json.loads(p.read_text()) for p in output.glob('*/seed42/invocation.json')]
    assert len(records) == 4 and max(r['start'] for r in records) < min(r['end'] for r in records)
    assert len({r['tmp'] for r in records}) == 4
    assert not (output / 'alfworld').exists() and not (output / 'docvqa').exists()
    for b, code in [('officeqa', 7), ('searchqa', 0), ('livemath', 0), ('spreadsheetbench', 0)]:
        assert (output / 'status' / (b + '.seed42.exit_code')).read_text().strip() == str(code)
        assert json.loads((output / b / 'seed42/completion.json').read_text())['status'] == 'awaiting_test'
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    again = subprocess.run(['bash', str(script), '--execute'], env=env, text=True, capture_output=True)
    assert again.returncode != 0 and {p: p.read_bytes() for p in before} == before
