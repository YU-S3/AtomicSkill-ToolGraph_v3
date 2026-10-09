"""Empirical behavior, resource isolation and independent result credit."""
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema, validate_workflow
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.harness.simple_protocol import Broker, Capabilities


def config_for(root):
    import yaml
    config = yaml.safe_load(Path(__file__).resolve().parents[1].joinpath('configs/alfworld_empirical_seed42.yaml').read_text())
    config['data_dir'] = str(root)
    config['program_environment'].update(adapter_abi='simple.v1', image_digest=os.environ.get('PROGRAM_IMAGE_DIGEST', 'sha256:' + 'a'*64))
    return config


def program(source, inputs=None, outputs=None):
    return {"source": source, "entry": "run", "input_schema": inputs or object_schema(),
            "output_schema": outputs or object_schema(), "allowed_tools": ["act"],
            "environment": {"python": "3.12", "dependencies": [], "adapter_abi": "simple.v1", "image_digest": os.environ.get("PROGRAM_IMAGE_DIGEST", "sha256:" + "a"*64)}}


class Adapter:
    capabilities = Capabilities()

    def __init__(self):
        self.steps = 0

    def observe(self):
        return {"steps": self.steps}

    def available_tools(self):
        return [{"name": "act", "input_schema": object_schema({"value": {"type": "string"}}, ["value"])}]

    def call(self, name, arguments):
        self.steps += 1
        return {"accepted": True, "outputs": {"value": arguments["value"]}}

    def check_local(self, inputs, outputs, events):
        return "passed" if events else "unavailable"


@pytest.fixture
def worker():
    if sys.platform != "linux" or not shutil.which("docker") or not os.environ.get("PROGRAM_IMAGE_DIGEST"):
        pytest.skip("container tests require Linux, Docker and PROGRAM_IMAGE_DIGEST")
    return ProgramWorker({"wall_timeout_seconds": 2})


def test_new_chain_does_not_construct_legacy_system(tmp_path, monkeypatch):
    from atomic_skillgraph.system import create_system
    assert 'atomic_skillgraph.knowledge.database' not in sys.modules
    system = create_system(config_for(tmp_path), harness=Adapter(), provider=object())
    assert system.bank.all("program") == []
    system.bank.close()


def test_workflow_can_trim_history_but_cannot_invent_result_edges():
    workflow = {"nodes": [{"id": "find", "goal": "find", "args": {}},
        {"id": "take", "goal": "take", "args": {"object": {"from": "find", "field": "object_id"}}}]}
    validate_workflow(workflow)
    bad = copy.deepcopy(workflow)
    bad["nodes"][0]["args"] = {"object": {"from": "take", "field": "object_id"}}
    with pytest.raises(ValueError):
        validate_workflow(bad)
    revised = {"nodes": [{"id": "deliver", "goal": "deliver", "args": {
        "object": {"from": "take", "field": "object_id"}}}]}
    validate_workflow(revised, completed={"take"})


def test_program_state_uses_independent_tasks_not_retries_and_frozen_is_readonly(tmp_path):
    bank = Bank(tmp_path / "bank")
    asset = bank.put("program", program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{}}"))
    for attempt_id, task in [("a", "physical1"), ("b", "physical1"), ("c", "physical2")]:
        row = {"id": attempt_id, "program_id": asset["id"], "task_key": task,
               "origin": "train_test", "outcome": "positive", "basis": "local_check"}
        bank.record(row)
        bank.record(row)
        assert bank.get(asset["id"])["state"] == ("usable" if attempt_id == "c" else "candidate")
    frozen = tmp_path / "frozen"
    bank.freeze(frozen)
    readonly = Bank(frozen, readonly=True)
    assert readonly.digest() == bank.digest()
    with pytest.raises(RuntimeError):
        readonly.record(row)
    readonly.close()
    bank.close()


def test_two_actual_failures_disable_but_normal_not_found_does_not(tmp_path):
    bank = Bank(tmp_path)
    asset = bank.put("program", program("def run(ctx, inputs):\n    return {'status':'not_found','outputs':{}}"))
    for i, outcome in enumerate(["normal", "execution_failure", "normal", "execution_failure"]):
        bank.record({"id": str(i), "program_id": asset["id"], "task_key": str(i), "origin": "online", "outcome": outcome})
    assert bank.get(asset["id"])["state"] == "candidate"
    bank.record({"id": "4", "program_id": asset["id"], "task_key": "4", "origin": "online", "outcome": "execution_failure"})
    assert bank.get(asset["id"])["state"] == "disabled"


def test_real_program_category_input_concrete_output_and_shared_calls(tmp_path, worker):
    bank = Bank(tmp_path)
    p = bank.put("program", program("""def run(ctx, inputs):
    result = ctx.call('act', {'value': inputs['query'] + '_1'})
    return {'status': 'ok', 'outputs': {'object_id': result['outputs']['value']}}
""", object_schema({"query": {"type": "string"}}, ["query"]),
        object_schema({"object_id": {"type": "string"}}, ["object_id"])))
    adapter = Adapter()
    broker = Broker(adapter, 1)
    result = worker.execute(p, {"query": "keychain"}, broker)
    assert result["status"] == "ok", result
    assert result["outputs"] == {"object_id": "keychain_1"}
    assert adapter.steps == 1 and broker.remaining_calls() == 0
    assert worker.execute(p, {"query": "keychain"}, broker)["status"] == "execution_error"


def test_real_sandbox_hides_keys_host_and_network_and_forbids_descendants(tmp_path, worker, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "private-test-secret")
    secret = tmp_path / "answer.json"
    secret.write_text("private answer")
    source = f"""import os, socket, subprocess
def run(ctx, inputs):
    assert 'MODEL_API_KEY' not in os.environ
    assert not os.path.exists({str(secret)!r})
    try:
        socket.create_connection(('1.1.1.1', 443), timeout=.2)
    except OSError:
        pass
    else:
        raise RuntimeError('network isolation failed')
    assert os.getuid() != 0
    try:
        open('/etc/worker-forbidden', 'w')
    except OSError:
        pass
    else:
        raise RuntimeError('root filesystem writable')
    return {{'status': 'ok', 'outputs': {{}}}}
"""
    bank = Bank(tmp_path / "bank")
    p = bank.put("program", program(source))
    assert worker.execute(p, {}, Broker(Adapter(), 3))["status"] == "ok"


def test_infinite_python_and_rpc_overflow_are_terminated(tmp_path, worker):
    bank = Bank(tmp_path)
    p = bank.put("program", program("def run(ctx, inputs):\n    while True: pass"))
    result = worker.execute(p, {}, Broker(Adapter(), 3))
    assert result["status"] == "execution_error" and result["elapsed_seconds"] < 4
    p = bank.put("program", program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'x':'a'*2000000}}"))
    assert worker.execute(p, {}, Broker(Adapter(), 3))["status"] == "execution_error"


def test_preparation_then_two_program_nodes_execute_without_model_calls(tmp_path, worker):
    bank = Bank(tmp_path)
    p = bank.put("program", program("""def run(ctx, inputs):
    ctx.call('act', {'value': inputs['value']})
    return {'status':'ok','outputs': {'value': inputs['value']}}
""", object_schema({"value": {"type": "string"}}, ["value"]),
        object_schema({"value": {"type": "string"}}, ["value"])))
    for key in ["physical1", "physical2"]:
        bank.record({"id": key, "program_id": p["id"], "task_key": key,
                     "origin": "train_test", "outcome": "positive", "basis": "local_check"})
    def no_agent(*a, **kw):
        pytest.fail("unnecessary model decision")
    adapter = Adapter()
    broker = Broker(adapter, 5)
    plan = {"nodes": [{"id": str(i), "goal": "pass value", "program_id": p["id"],
        "args": {"value": {"literal": "keychain_1"} if i == 0 else {"from": str(i-1), "field": "value"}}}
        for i in range(3)], "outputs": {"value": {"from": "2", "field": "value"}}}
    # Stop after the intended workflow; this test does not model a task rescue.
    adapter.capabilities = Capabilities(interaction="single_answer")
    result = Executor(bank, no_agent, worker, object()).run(PublicTask("t", "p", "goal"), adapter, broker, plan)
    assert result["prediction"] == {"value": "keychain_1"}
    assert adapter.steps == 3 and len(result["attempts"]) == 3
    assert all(a['outputs_consumed'] for a in result['attempts'])


def test_agent_completion_does_not_credit_unused_program_outputs(tmp_path, worker):
    bank = Bank(tmp_path)
    p = bank.put('program', program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{}}"))
    steps = iter([{'action': 'call_program', 'name': p['id'], 'arguments': {}},
                  {'action': 'complete_node', 'outputs': {'actual': 'agent result'}}])
    adapter = Adapter()
    adapter.capabilities = Capabilities(interaction='single_answer')
    result = Executor(bank, lambda *a, **k: next(steps), worker, object()).run(
        PublicTask('t', 'p', 'goal'), adapter, Broker(adapter, 5),
        {'nodes': [{'id': 'one', 'goal': 'goal', 'args': {}}]})
    assert result['attempts'][0]['status'] == 'ok'
    assert not result['attempts'][0]['outputs_consumed']


class AnswerAdapter(Adapter):
    capabilities = Capabilities(interaction='single_answer', tool_surface='none')

    def reset(self, task):
        self.task = task

    def submit(self, answer):
        return answer

    def evaluate(self, answer):
        return {'hard': answer == 'correct', 'raw_score': float(answer == 'correct')}

    def close(self):
        pass


class Provider:
    def __init__(self, answers):
        self.answers, self.calls = iter(answers), []

    def complete(self, messages, tools):
        from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
        answer = next(self.answers)
        self.calls.append(messages)
        calls = [NativeToolCall('call_1', tools[0].name, answer)] if tools else []
        return AgentTurn('' if tools else answer, calls, 'tool_calls' if tools else 'stop', 10, 20, 30, 5, 1)


@pytest.mark.parametrize('status,hard,positive', [
    ('ok', True, True), ('ok', False, False), ('not_found', True, False),
    ('execution_error', True, False)])
def test_program_terminal_success_requires_normal_return_and_independent_score(tmp_path, worker, status, hard, positive):
    from atomic_skillgraph.empirical.system import EmpiricalSystem

    class TerminalAdapter(AnswerAdapter):
        def reset(self, task):
            self.steps = 0

        def call(self, name, arguments):
            return dict(super().call(name, arguments), done=True)

        def check_local(self, inputs, outputs, events):
            return 'unavailable'

        def evaluate(self, answer):
            return {'hard': hard and self.steps == 1}

    adapter = TerminalAdapter()
    system = EmpiricalSystem(config_for(tmp_path), harness=adapter, provider=object(), adapter_factory=TerminalAdapter)
    system.worker = worker
    ending = "raise RuntimeError('failed after terminal')" if status == 'execution_error' else f"return {{'status': {status!r}, 'outputs': {{}}}}"
    p = system.bank.put('program', program("def run(ctx, inputs):\n    ctx.call('act', {'value': 'done'})\n    " + ending))
    for key in ['prior1', 'prior2']:
        system.bank.record({'id': key, 'program_id': p['id'], 'task_key': key,
                            'origin': 'train_test', 'outcome': 'positive', 'basis': 'local_check'})
    task = PublicTask('t', 'physical', 'finish task')
    def no_agent(*a, **kw):
        pytest.fail('terminal Program must not invoke Agent recovery')
    system.agent = no_agent
    record = system.test_program(p, {}, task, trial_id='trial')
    assert (record['outcome'] == 'positive') == positive
    assert record['result']['terminal_by_program']
    assert 'continuation' not in record['result']
    adapter.reset(task)
    execution = Executor(system.bank, no_agent, worker, object()).run(task, adapter, Broker(adapter, 1),
        {'nodes': [{'id': 'finish', 'goal': task.goal, 'program_id': p['id'], 'args': {}}]})
    attempt = execution['attempts'][0]
    assert attempt['terminal_by_program'] and not attempt['outputs_consumed']
    system.learner.learn = lambda *a: {}
    system.learn_trace(task, {'execution': execution, 'score': adapter.evaluate(adapter.submit(execution['prediction']))})
    assert (attempt['outcome'] == 'positive') == positive
    if positive:
        assert record['basis'] == attempt['basis'] == 'task_outcome'
    system.close()


def test_single_answer_uses_one_solve_and_independent_score_and_frozen(tmp_path):
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    config = config_for(tmp_path / 'bank')
    provider = Provider(['I declare success', {'decision': 'no_change'}])
    system = EmpiricalSystem(config, harness=AnswerAdapter(), provider=provider)
    trace = system.run_task(PublicTask('t', 'p', 'question'))
    assert not trace['score']['hard'] and len(provider.calls) == 1
    assert trace['learning_status'] == 'skipped_policy' and trace['learning']['decision_origin'] == 'host'
    assert [r['stage'] for r in trace['requests']] == ['runtime']
    system.bank.freeze(tmp_path / 'frozen')
    system.close()
    config['data_dir'] = str(tmp_path / 'frozen')
    provider = Provider(['correct'])
    system = EmpiricalSystem(config, harness=AnswerAdapter(), provider=provider, readonly=True)
    trace = system.run_task(PublicTask('v', 'q', 'question', split='valid_seen'))
    assert trace['score']['hard'] and len(provider.calls) == 1
    assert trace['knowledge_before'] == trace['knowledge_after']
    with pytest.raises(RuntimeError):
        system.run_task(PublicTask('v', 'q', 'question', split='valid_seen'), learn=True)
    system.close()


def test_resume_completed_task_does_not_repeat_solve_or_learning(tmp_path, monkeypatch):
    from experiments import run_empirical as runner
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    config = config_for(tmp_path / 'bank')
    provider = Provider(['correct', {'decision': 'no_change'}])
    monkeypatch.setattr(runner, 'EmpiricalSystem', lambda config, **kw: EmpiricalSystem(config, provider=provider, **kw))
    tasks = [PublicTask('t', 'p', 'question')]
    first = runner.run(config, tasks, tmp_path / 'run', adapter=AnswerAdapter())
    resumed = runner.run(config, tasks, tmp_path / 'run', adapter=AnswerAdapter(), resume=True)
    assert len(provider.calls) == 2 and first == resumed


def test_invalid_optional_workflow_preserves_tested_program(tmp_path, monkeypatch):
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    skill = {'goal': 'prepare', 'input_schema': object_schema(), 'output_schema': object_schema(),
             'execution_intent': 'program_requested'}
    provider = Provider([{'decision': 'propose_skill_and_program_spec', 'skill': skill,
                         'realization_request': {'skill_id': '$new', 'action': 'build', 'case_bindings': [
                             {'case_id': key, 'inputs': {}, 'start_mode': 'reset', 'prefix': []} for key in ['p1','p2']]},
                         'workflow': {'goal': 'bad', 'nodes': [{'id': 'one', 'execution_mode': 'dynamic', 'goal': 'bad',
                             'args': {'x': {'from': 'missing', 'field': 'x'}}}]}},
                         {'source': "def run(ctx, inputs):\n    return {'status':'ok','outputs':{}}", 'trial_inputs': [{'case_id': key, 'inputs': {}, 'start_mode': 'reset', 'prefix': []} for key in ['p1','p2']]}])
    config = config_for(tmp_path)
    adapter = AnswerAdapter()
    adapter.capabilities = Capabilities(interaction='tool_loop')
    system = EmpiricalSystem(config, harness=adapter, provider=provider)
    def trial(program, inputs, task, trial_id, **kwargs):
        row = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
               'origin': 'train_test', 'outcome': 'positive', 'basis': 'local_check'}
        system.bank.record(row)
        return row
    monkeypatch.setattr(system, 'test_program', trial)
    system.learner.cases.append((PublicTask('previous', 'p1', 'prepare'), {}))
    log = system.learner.learn(PublicTask('current', 'p2', 'prepare'), {'tools': [], 'score': {}, 'execution': {}})
    assert system.bank.get(log['program'])['state'] == 'usable'
    assert system.bank.all('workflow') == [] and 'Workflow rejected' in log['errors'][0]
    assert len(provider.calls) == 2 and len(log['tests']) == 2
    system.close()


def test_dynamic_continuation_keeps_completed_results(tmp_path):
    bank = Bank(tmp_path)
    steps = iter([{'action': 'complete_node', 'outputs': {'value': 'one'}},
                  {'action': 'complete_node', 'outputs': {'value': 'two'}}, {'action': 'finish', 'answer': 'done'}])
    adapter = Adapter()
    result = Executor(bank, lambda *a, **k: next(steps), object(), object()).run(
        PublicTask('t', 'p', 'goal'), adapter, Broker(adapter, 5),
        {'nodes': [{'id': 'task', 'goal': 'goal', 'args': {}}]})
    assert [v['outputs']['value'] for v in result['values']] == ['one']
    assert result['pending_outputs']['remaining_task']['outputs'] == {'value': 'two'}
    assert result['dynamic_escapes'] == 1 and result['prediction'] == 'done'


def test_interrupted_learning_resume_keeps_execution_and_all_billed_usage(tmp_path, monkeypatch):
    from experiments import run_empirical as runner
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    from atomic_skillgraph.empirical.learner import Learner
    config = config_for(tmp_path / 'bank')
    provider = Provider(['correct', {'decision': 'no_change'}])
    monkeypatch.setattr(runner, 'EmpiricalSystem', lambda config, **kw: EmpiricalSystem(config, provider=provider, **kw))
    original = Learner.learn
    def interrupted(learner, *a, **kw):
        learner.system.bank.put('skill', {'goal': 'uncommitted'})
        raise RuntimeError('interrupted')
    monkeypatch.setattr(Learner, 'learn', interrupted)
    tasks = [PublicTask('t', 'p', 'question')]
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.run(config, tasks, tmp_path / 'run', adapter=AnswerAdapter())
    monkeypatch.setattr(Learner, 'learn', original)
    summary = runner.run(config, tasks, tmp_path / 'run', adapter=AnswerAdapter(), resume=True)
    assert summary['total_tokens'] == 60 and summary['completed_task_tokens'] == 60
    assert len(provider.calls) == 2
    frozen = Bank(tmp_path / 'run' / 'frozen_bank', readonly=True)
    assert frozen.all('skill')[0]['goal'] == 'uncommitted'
    frozen.close()


def test_loader_continuation_preserves_completed_train_and_rejects_bank_changes(tmp_path):
    import json
    from atomic_skillgraph.experiments.run_empirical import code_identity, write_json
    from atomic_skillgraph.experiments.run_empirical_pilot import completed_train
    from atomic_skillgraph.empirical.contracts import digest

    output = tmp_path / 'pilot'
    config = config_for(output / 'train' / 'bank')
    entries = [{'gamefile_rel': 'original/game.tw-pddl', 'gamefile_sha256': 'original'}]
    bank = Bank(config['data_dir'])
    frozen = bank.freeze(output / 'train' / 'frozen_bank')
    summary = {'complete': True, 'tasks': 1, 'frozen': frozen, 'knowledge_digest': bank.digest(),
        'cases': [{'task': {'physical_key': digest({'path': entries[0]['gamefile_rel'], 'sha256': 'original'})}}]}
    code = dict(code_identity(), tracked_dirty=False)
    write_json(output / 'train' / 'execution_manifest.json', {'config': config, 'code': code})
    write_json(output / 'train' / 'summary.json', summary)
    original = (output / 'train' / 'summary.json').read_bytes()
    assert completed_train(config, entries, output) == summary
    assert (output / 'train' / 'summary.json').read_bytes() == original
    assert bank.digest() == summary['knowledge_digest']
    assert json.loads((output / 'continuation_manifest.json').read_text())['train_reused_without_execution']
    with pytest.raises(ValueError, match='physical task selection differs'):
        completed_train(config, [dict(entries[0], gamefile_sha256='other')], output)
    bank.put('skill', {'goal': 'unexpected mutation'})
    with pytest.raises(ValueError, match='Bank changed'):
        completed_train(config, entries, output)
    bank.close()
