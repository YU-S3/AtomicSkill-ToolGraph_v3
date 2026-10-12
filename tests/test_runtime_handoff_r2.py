"""Production regressions for public memory, field handoffs and bounded reads."""
import json
from copy import deepcopy
import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, RuntimeDecision, digest
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.prompts import RUNTIME_PROMPT, STEP
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.public_results import TEXT_PAGE, SCOPED_SEARCH
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


def test_native_acquired_page_reaches_actual_wire_and_records_exact_window(tmp_path, monkeypatch):
    system = office(tmp_path)
    text = '页' * 9000 + 'acquired late table row' + '尾' * 4000
    (system.adapter.corpus/'page.txt').write_text(text, encoding='utf-8')
    context = system.task_context
    arguments = {'path': 'page.txt', 'offset': 0}
    raw = system.adapter.call('read', arguments)
    rid = context.register('page', raw, name='read', arguments=arguments,
                           origin=system.adapter.result_protocol('read'))
    assert context.origin(rid) == TEXT_PAGE
    page = context.view(rid)['data']['text']
    assert page['content'] == text[:12000] and page['end'] == 12000
    assert 'acquired late table row' in page['content']
    seen = http(monkeypatch, [response([{'action': 'finish', 'answer': '<answer>acquired</answer>'}])])
    try:
        system.agent('runtime', RUNTIME_PROMPT, {'recent_results': [context.view(rid)], 'tools': system.adapter.available_tools()},
                     'runtime_step', STEP)
        assert 'acquired late table row' in seen[0]['messages'][1]['content']
        receipt = system.requests[-1]['displayed_windows'][0]
        assert receipt['end'] == 12000 and receipt['content_hash'] == digest(text[:12000])
        short = context.fit_window(page, lambda w: len(w['content']) <= 4000)
        assert short['end'] == len(short['content']) == 3000
        assert short['next_offset'] == 3000 and short['content_hash'] == digest(text[:3000])
    finally: system.close()


def test_read_contract_requires_path_and_keeps_host_root_unavailable():
    context = TaskContext()
    assert 'path' in context.tool()['input_schema']['required']
    rid = context.register('scoped', {'data': ['authorized.txt'], 'scope_audit': {'private_host_field': True}}, origin=SCOPED_SEARCH)
    with pytest.raises(ValueError, match='public field path') as error:
        context.read(rid)
    assert '["data"]' in str(error.value) and 'scope_audit' not in str(error.value)
    with pytest.raises(ValueError, match='Host metadata'):
        context.read(rid, path=['scope_audit'])
    assert context.local_reads == []
    assert context.read(rid, path=['data'])['data'] == ['authorized.txt']
