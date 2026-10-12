"""Real source binding, bounded learning repair and retained acquisition regressions."""
from copy import deepcopy
import ast
import json

import pytest

from atomic_skillgraph.empirical.contracts import object_schema
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.local_validation import source_literals, canonical_binding
from atomic_skillgraph.empirical.model_view import learning_case, expand_material
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.prompts import LEARNING, LEARNER_PROMPT
from atomic_skillgraph.harness.simple_protocol import Broker
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import office, http, response
from test_empirical import worker


def test_literal_series_is_bound_without_accepting_unexecuted_branch():
    rows = source_literals('values = [-2.5, 3, 7]\nif False:\n    hidden = [9, 11]\n')
    assert next(r for r in rows if r['path'] == ['body', 0, 'value'])['value'] == [-2.5, 3, 7]
    assert not any(r['value'] == [9, 11] for r in rows)
    assert not any(r['value'] == 9 for r in rows)


def test_real_series_and_computed_output_bind_but_fabricated_paths_fail(tmp_path, worker):
    s = office(tmp_path)
    s.adapter.worker = worker
    source = 'import json\nvalues = [-2.5, 3, 7]\nprint(json.dumps({"total": sum(values)}))'
    broker = Broker(s.adapter, 24)
    broker.call('execute_python', {'source': source})
    trace = {'tools': broker.events, 'execution': {}, 'score': {'hard': False}}
    experience = s.learner._experience(s.adapter.task, trace)
    evidence = experience['local_evidence'][0]
    skill = {'input_schema': object_schema({'values': {'type': 'array', 'items': {'type': 'number'}}}, ['values']),
             'output_schema': object_schema({'total': {'type': 'number'}}, ['total']), 'result_role': 'intermediate'}
    binding = {'case_id': s.adapter.task.physical_key, 'local_evidence_ref': evidence['id'],
        'inputs': {'values': [-2.5, 3, 7]},
        'input_refs': {'values': {'kind': 'source_literal', 'path': ['body', 1, 'value']}},
        'reference_fields': {'total': {'kind': 'json', 'path': ['data', 'computed', 'total']}}}
    try:
        canonical_binding(binding, experience, skill)
        card = learning_case(experience, s.adapter.task.physical_key)['local_evidence'][0]
        assert card['source_excerpt'] == ast.unparse(ast.parse(source)) and not card['source_truncated']
        assert any(r['value'] == [-2.5, 3, 7] for r in card['input_sources'])
        assert any(r['path'] == ['data', 'computed', 'total'] and r['type'] == 'float' for r in card['output_references'])
        bad = deepcopy(binding); bad['input_refs']['values']['path'] = ['observation']
        with pytest.raises(ValueError, match='unbound_input'): canonical_binding(bad, experience, skill)
        bad = deepcopy(binding); bad['reference_fields']['total']['path'] = ['observation']
        with pytest.raises(ValueError, match='output_type'): canonical_binding(bad, experience, skill)
    finally: s.close()


@pytest.mark.parametrize('stdout', ['log\n{"total": 3}', '{"total": 1}{"total": 2}',
                                   '{"total": NaN}', '{"total": 1e400}', '{"total": 1, "total": 2}', '3'])
def test_compute_never_guesses_json_from_arbitrary_stdout(tmp_path, worker, stdout):
    s = office(tmp_path); s.adapter.worker = worker
    try:
        result = s.adapter.call('execute_python', {'source': 'print(' + repr(stdout) + ')'})
        assert result['accepted'] and result['observation'] == stdout + '\n'
        assert 'computed' not in result['data']
    finally: s.close()


def test_declaration_repair_names_contract_and_exposes_real_series(tmp_path, monkeypatch):
    s = office(tmp_path)
    source = 'values = [2, 5, 9]\nprint(sum(values))'
    experience = {'task': {'goal': 'calculate total', 'inputs': {}}, 'local_evidence': [{
        'id': 'local:compute', 'kind': 'native_operation',
        'operation': {'name': 'execute_python', 'arguments': {'source': source}},
        'reference': {'accepted': True, 'observation': '16\n', 'data': {}}, 'public_task': {'goal': 'calculate total', 'inputs': {}}}]}
    proposal = {'decision': 'propose_skill_and_program_spec', 'skill': {
        'goal': 'calculate total', 'execution_intent': 'program_requested', 'result_role': 'final_answer',
        'input_schema': object_schema({'values': {'type': 'array', 'items': {'type': 'number'}}}, ['values']),
        'output_schema': object_schema({'total': {'type': 'number'}}, ['total'])},
        'realization_request': {'skill_id': '$new', 'action': 'defer', 'case_bindings': []}}
    seen = http(monkeypatch, [response([proposal], name='submit_learning'),
                              response([{'decision': 'no_change'}], name='submit_learning')])
    try:
        s.agent('extractor', LEARNER_PROMPT, {'experience': learning_case(experience, 'case')},
                'submit_learning', LEARNING, validator=s.learner.validate_learning_proposal, repair_limit=1)
        material = expand_material(json.loads(seen[1]['messages'][1]['content']))
        error = material['errors'][0]['errors'][0]
        assert error['code'] == 'program_answer_schema_invalid' and error['repair_target'] == 'declaration'
        assert 'Required string answer' in error['expected']
        assert 'Invalid schemas are not frozen' in seen[1]['messages'][0]['content']
        assert len(seen) == 2 and not s.bank.all('program')
    finally: s.close()


def test_read_page_survives_three_later_searches(tmp_path):
    s = office(tmp_path)
    actions = iter([{'action': 'call_tool', 'name': 'read', 'arguments': {'path': 'a.txt', 'offset': 0}},
        *[{'action': 'call_tool', 'name': 'glob', 'arguments': {'pattern': pattern}}
          for pattern in ('*.txt', '**/*.txt', 'a.*')], {'action': 'finish', 'answer': 'done'}])
    materials = []
    def agent(stage, prompt, material, *args, **kwargs):
        materials.append(expand_material(material)); return next(actions)
    try:
        Executor(s.bank, agent, s.worker, s.planner).run(s.adapter.task, s.adapter,
            Broker(s.adapter, 24, context=s.task_context), dynamic(s.adapter.task))
        recent = materials[-1]['memory']['recent_results']
        assert any(r.get('data', {}).get('text', {}).get('content') == 'first\n第二行 target\nlast'
                   for r in recent if isinstance(r.get('data'), dict))
    finally: s.close()


def test_learning_wire_keeps_acquired_page_before_discovery_cards(tmp_path, monkeypatch):
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    s = office(tmp_path)
    broker = Broker(s.adapter, 24)
    for name, arguments in [('glob', {'pattern': '*.txt'}),
                            ('grep', {'pattern': 'target'}),
                            ('read', {'path': 'a.txt', 'offset': 0})]:
        broker.call(name, arguments)
    experience = s.learner._experience(s.adapter.task, {'tools': broker.events, 'execution': {}, 'score': {'hard': False}})
    material = {'experience': learning_case(experience, s.adapter.task.physical_key)}
    cards = material['experience']['local_evidence']
    assert [c['operation_name'] for c in cards] == ['read', 'glob', 'grep']
    sent = http(monkeypatch, [response([{'decision': 'no_change'}], name='submit_learning')] * 2)
    try:
        s.config['learning']['extractor_payload_max_bytes'] = 65536
        s.agent('extractor', LEARNER_PROMPT, material, 'submit_learning', LEARNING)
        # Force one complete card out through the production full-wire admission.
        compact = deepcopy(sent[0])
        compact['messages'][1]['content'] = json.dumps(json.loads(compact['messages'][1]['content']),
                                                       ensure_ascii=False, separators=(',', ':'))
        s.config['learning']['extractor_payload_max_bytes'] = input_token_bound(compact) - 1
        s.agent('extractor', LEARNER_PROMPT, material, 'submit_learning', LEARNING)
        actual = expand_material(json.loads(sent[1]['messages'][1]['content']))
        assert len(actual['experience']['local_evidence']) < len(cards)
        assert actual['experience']['local_evidence'][0] == cards[0]
        assert actual['experience']['local_evidence'][0]['operation_name'] == 'read'
        assert input_token_bound(sent[1]) <= s.config['learning']['extractor_payload_max_bytes']
        assert any(row['reason'] == 'full_payload_input_limit' for row in s.requests[-1]['material_audit']['omitted'])
        assert not s.bank.all('program')
    finally: s.close()


def test_optional_source_preview_shrinks_before_capacity_refusal(tmp_path, monkeypatch):
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    from atomic_skillgraph.empirical.model_view import learning_capacity_audit
    s = office(tmp_path)
    source = 'values = [1, 2, 3]\nlabel = ' + repr('x' * 1500) + '\nprint(sum(values))'
    experience = {'task': {'goal': 'sum a series', 'inputs': {}}, 'local_evidence': [{
        'id': 'local:compute', 'kind': 'native_operation',
        'operation': {'name': 'execute_python', 'arguments': {'source': source}},
        'reference': {'accepted': True, 'observation': '6\n', 'data': {}},
        'public_task': {'goal': 'sum a series', 'inputs': {}}}]}
    material = {'experience': learning_case(experience, 'case')}
    original = material['experience']['local_evidence'][0]
    sent = http(monkeypatch, [response([{'decision': 'no_change'}], name='submit_learning')] * 2)
    try:
        s.config['learning']['extractor_payload_max_bytes'] = 65536
        s.agent('extractor', LEARNER_PROMPT, material, 'submit_learning', LEARNING)
        compact = deepcopy(sent[0]); compact['messages'][1]['content'] = json.dumps(
            json.loads(compact['messages'][1]['content']), ensure_ascii=False, separators=(',', ':'))
        limit = input_token_bound(compact) - 100
        assert learning_capacity_audit(compact, limit, 'extractor', 'extractor')['optional_material_remaining']
        s.config['learning']['extractor_payload_max_bytes'] = limit
        s.agent('extractor', LEARNER_PROMPT, material, 'submit_learning', LEARNING)
        actual = expand_material(json.loads(sent[1]['messages'][1]['content']))['experience']['local_evidence'][0]
        assert input_token_bound(sent[1]) <= limit
        assert actual['source_excerpt'] == original['source_excerpt'][:512] and actual['source_truncated']
        assert actual['input_sources'] == original['input_sources']
        assert actual['output_references'] == original['output_references']
        assert not s.bank.all('program') and len(sent) == 2
    finally: s.close()


def test_compute_learning_builds_usable_and_frozen_reuses_without_runtime(tmp_path, monkeypatch, worker):
    s = office(tmp_path); s.worker = worker; s.adapter.worker = worker
    s.adapter_factory = lambda: OfficeAdapter({'office': {'answer': 'private'}}, s.config)
    source = 'import json\nvalues = [-2.5, 3, 7]\nprint(json.dumps({"total": sum(values)}))'
    broker = Broker(s.adapter, 24); broker.call('execute_python', {'source': source})
    trace = {'tools': broker.events, 'execution': {'prediction': '7.5'}, 'score': {'hard': True}}
    experience = s.learner._experience(s.adapter.task, trace)
    proposal = {'decision': 'propose_skill_and_program_spec', 'skill': {
        'goal': 'Compute the sum of a numeric series', 'execution_intent': 'program_requested', 'result_role': 'intermediate',
        'input_schema': object_schema({'values': {'type': 'array', 'items': {'type': 'number'}}}, ['values']),
        'output_schema': object_schema({'total': {'type': 'number'}}, ['total'])},
        'realization_request': {'skill_id': '$new', 'action': 'build', 'case_bindings': [{
            'case_id': s.adapter.task.physical_key, 'local_evidence_ref': experience['local_evidence'][0]['id'],
            'inputs': {'values': [-2.5, 3, 7]},
            'input_refs': {'values': {'kind': 'source_literal', 'path': ['body', 1, 'value']}},
            'reference_fields': {'total': {'kind': 'json', 'path': ['data', 'computed', 'total']}}}]}}
    generated = {'source': "def run(ctx, inputs):\n    return {'status':'ok','outputs':{'total':sum(inputs['values'])}}"}
    seen = http(monkeypatch, [response([proposal], name='submit_learning'), response([generated], name='submit_program')])
    frozen = None
    try:
        learned = s.learner.learn(s.adapter.task, trace)
        assert not learned['errors'] and len(seen) == 2
        programs = s.bank.all('program'); assert len(programs) == 1
        program = programs[0]; assert s.bank.program_eligible(program)
        assert program['local_validation']['passed'] and program['local_validation']['reason'] == 'source_output_match'
        s.bank.freeze(tmp_path/'frozen'); frozen = Bank(tmp_path/'frozen', readonly=True); before = frozen.digest()
        final_calls = []
        def consume_total(*args, **kwargs):
            material = expand_material(args[2]); final_calls.append(material)
            assert material['node']['id'] == 'final' and material['bindings']['inputs']['total'] == 13
            return {'action': 'finish', 'answer': '13', 'used_inputs': ['total']}
        execution = Executor(frozen, consume_total, worker, s.planner, frozen=True).run(s.adapter.task, s.adapter,
            Broker(s.adapter, 24), {'nodes': [{'id': 'sum', 'execution_mode': 'program', 'program_id': program['id'],
                'args': {'values': {'literal': [1, 4, 8]}}}, {'id': 'final', 'execution_mode': 'dynamic',
                'goal': 'submit computed total', 'args': {'total': {'from': 'sum', 'field': 'total'}}}]})
        assert execution['prediction'] == '13' and execution['attempts'][0]['outputs_consumed']
        assert len(final_calls) == 1 and execution['attempts'][0]['calls'] == 0
        assert frozen.digest() == before and len(seen) == 2
    finally:
        if frozen: frozen.close()
        s.close()


def test_worker_keeps_stdout_when_done_arrives_before_diagnostic_pipe(tmp_path, monkeypatch, worker):
    import atomic_skillgraph.empirical.program_worker as module
    selector_type = module.selectors.DefaultSelector
    class RpcFirst:
        def __init__(self): self.selector = selector_type(); self.rpc_closed = False
        def register(self, *args): return self.selector.register(*args)
        def get_map(self): return self.selector.get_map()
        def close(self): self.selector.close()
        def unregister(self, stream):
            if self.selector.get_key(stream).data == 'rpc': self.rpc_closed = True
            return self.selector.unregister(stream)
        def select(self, timeout):
            ready = self.selector.select(timeout)
            return ready if self.rpc_closed else [row for row in ready if row[0].data == 'rpc']
    monkeypatch.setattr(module.selectors, 'DefaultSelector', RpcFirst)
    s = office(tmp_path); s.adapter.worker = worker
    try:
        result = s.adapter.call('execute_python', {'source': 'print("calculation evidence")'})
        assert result['accepted'] and result['observation'] == 'calculation evidence\n'
    finally: s.close()
