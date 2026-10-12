"""Production regressions for public memory, field handoffs and bounded reads."""
import json
from copy import deepcopy

from atomic_skillgraph.empirical.contracts import PublicTask, RuntimeDecision, digest
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.prompts import RUNTIME_PROMPT, STEP
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import office, http, response


def test_alias_example_reaches_consumer_without_repeating_search(tmp_path):
    system = office(tmp_path)
    task = PublicTask('office', 'office-physical', 'extract then compute')
    plan = {'interface_version': 'empirical.workflow.v2', 'goal': task.goal, 'nodes': [
        {'id': 'extract', 'execution_mode': 'dynamic', 'goal': 'extract value', 'args': {},
         'output_aliases': {'value': 'acquired_value'}},
        {'id': 'compute', 'execution_mode': 'dynamic', 'goal': 'compute',
         'args': {'value': {'from': 'extract', 'field': 'acquired_value'}}}],
        'outputs': {'answer': {'from': 'compute', 'field': 'answer'}}}
    materials = []
    def agent(stage, prompt, material, *args, **kwargs):
        materials.append(deepcopy(material))
        if material['node']['id'] == 'extract':
            assert material['handoff']['return_example'] == {'value': '<actual ordinary value>'}
            return {'action': 'complete_node', 'outputs': {'value': 3}}
        assert material['bindings']['inputs']['value'] == 3
        return {'action': 'complete_node', 'outputs': {'answer': '6'}, 'used_inputs': ['value']}
    try:
        execution = Executor(system.bank, agent, system.worker, system.planner).run(
            task, system.adapter, Broker(system.adapter, 24, context=system.task_context), plan)
        assert execution['prediction'] == '6' and execution['reason'] == 'plan_submitted'
        assert len(materials) == 2 and not any('error' in h for h in execution['history'])
    finally: system.close()


def test_public_response_reaches_next_request_and_finish_without_private_reasoning(tmp_path, monkeypatch):
    system = office(tmp_path)
    text = 'Acquired table: March 3 rates 4.708/4.943; gap 0.235.'
    seen = http(monkeypatch, [response([{'action': 'call_tool', 'name': 'glob', 'arguments': {'pattern': '*.txt'}}], content=text),
                              response([{'action': 'finish', 'answer': '<answer>March 3</answer>'}])])
    try:
        materials = {'tools': system.adapter.available_tools()}
        system.agent('runtime', RUNTIME_PROMPT, materials, 'runtime_step', STEP)
        assert system.task_context.public_notes[0]['text'] == text
        finish = build_finish_evidence(system.task_context, {'original_goal': 'use acquired evidence',
            'public_notes': system.task_context.public_notes}, 'finish', max_bytes=16384)
        assert finish['materials']['public_notes'][0]['evidence_level'] == 'model_reported'
        system.agent('runtime', RUNTIME_PROMPT, {'tools': [], 'memory': {'public_notes': system.task_context.public_notes}},
                     'runtime_step', STEP)
        assert text in seen[1]['messages'][1]['content']
        assert 'fixture-private' not in json.dumps(finish) + json.dumps(seen)
    finally: system.close()


def test_independent_public_reads_use_production_batch_admission(tmp_path, monkeypatch):
    system = office(tmp_path)
    context = system.task_context
    one = context.register('one', {'data': ['alpha', 'beta']})
    two = context.register('two', {'data': {'text': 'second page'}})
    actions = [{'action': 'call_tool', 'name': 'read_result', 'arguments': {'result_id': one, 'path': ['data']}},
               {'action': 'call_tool', 'name': 'read_result', 'arguments': {'result_id': two, 'path': ['data', 'text']}}]
    seen = http(monkeypatch, [response(actions)])
    try:
        decision = system.agent('runtime', RUNTIME_PROMPT, {'tools': [context.tool()]}, 'runtime_step', STEP)
        assert isinstance(decision, RuntimeDecision) and len(decision.actions) == 2 and len(seen) == 1
        assert context.read(**actions[0]['arguments'])['data'] == ['alpha', 'beta']
        assert context.read(**actions[1]['arguments'])['data'] == 'second page'
    finally: system.close()


def test_notes_are_bounded_and_never_make_source_evidence():
    context = TaskContext()
    for i in range(8): context.record_public_note(str(i), str(i) + 'x' * 1200)
    assert len(context.public_notes) == 4
    assert all(len(r['text']) == 1024 and r['truncated'] for r in context.public_notes)
    assert context.results == {} and context.local_reads == []
    finish = build_finish_evidence(context, {'original_goal': 'answer', 'public_notes': context.public_notes}, 'finish', max_bytes=500)
    assert finish['serialized_input_bytes'] <= 500 and not finish['materials']['public_notes']


def test_finish_trigger_survives_tool_limit_and_keeps_public_note(tmp_path):
    system = office(tmp_path)
    context = system.task_context
    steps = iter([{'action': 'call_tool', 'name': 'glob', 'arguments': {'pattern': '*.txt'},
                   'detail': 'Acquired relative file a.txt; use its public text next.'}])
    def agent(stage, prompt, material, name, *args, **kwargs):
        if name: return next(steps)
        assert material['public_notes'][0]['text'].startswith('Acquired relative file')
        return '<answer>unavailable</answer>'
    try:
        result = Executor(system.bank, agent, system.worker, system.planner).run(
            system.adapter.task, system.adapter, Broker(system.adapter, 1, context=context), dynamic(system.adapter.task))
        assert result['finish_trigger'] == 'tool_budget_exhausted' and result['reason'] == 'finish_only'
    finally: system.close()
