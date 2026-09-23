import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from atomic_skillgraph.agents.protocol import NativeToolSpec, NativeToolCall, AgentTurn, SchemaValidationError, validate_schema_instance
from atomic_skillgraph.agents.native_call_contract import NativeCallContractView, schema_diagnostics, rejection_diagnostic
from atomic_skillgraph.agents.session import ReplayAgentSession
from atomic_skillgraph.agents.usage import UsageLedger
from atomic_skillgraph.core.errors import AgentProtocolError

EXTRAS = {'intent': 'attempt_current_atomic', 'candidate_bindings': {}, 'candidate_outputs': {}}
SCHEMA = {'type': 'object', 'properties': {'object': {'type': 'string'}, 'source': {'type': 'string'}},
          'required': [], 'additionalProperties': False}
FIXTURES = Path(__file__).parent/'fixtures/release5'


class RawProvider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, *, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        n = len(self.requests)
        reply = next(self.replies)
        calls = reply if isinstance(reply, list) else [NativeToolCall(f'c{n}', tools[0].name, reply)]
        return AgentTurn('', calls, 'tool_calls', 7, 3, 10, 1, 1,
                         {'request_id': f'p{n}'})

    def snapshot(self):
        return {'fixture': True}


def session(replies):
    p = RawProvider(replies)
    return ReplayAgentSession(p, system_prompt='fixture', usage_ledger=UsageLedger(), usage_bucket='runtime_seeded'), p


def tool(schema=None):
    return NativeToolSpec('invoke_impl_fixture', 'fixture', copy.deepcopy(schema or SCHEMA),
                          call_kind='implementation', scope='node')


def test_NC01_03_first_feedback_complete_one_repair_no_mutation():
    invalid = {**EXTRAS, 'object': 'public_identity'}
    original = copy.deepcopy(invalid)
    s, p = session([invalid, {}])
    result = s.next_turn('call', tools=[tool()])
    d = s.snapshot()['native_protocol_diagnostics'][0]
    assert d['issues'][0]['unexpected_properties'] == sorted(EXTRAS)
    assert all(k in p.requests[1][0][-1]['content'] for k in EXTRAS)
    assert invalid == original and result.tool_calls[0].arguments == {}
    assert len(s.usage_ledger.events) == 2
    assert sum(e.usage.total_tokens for e in s.usage_ledger.events) == 20
    assert d['request_id'] == 'p1' and d['repair_attempted']
    s, p = session([invalid, invalid, {}])
    with pytest.raises(AgentProtocolError):
        s.next_turn('call', tools=[tool()])
    assert len(p.requests) == 2 and s.snapshot()['terminal_protocol_failure']


def test_NC04_07_actual_required_and_homonymous_business_inputs():
    validate_schema_instance({}, SCHEMA)
    schema = copy.deepcopy(SCHEMA); schema['required'] = ['object', 'source']
    s, p = session([EXTRAS, {'object': 'a', 'source': 'b'}])
    s.next_turn('call', tools=[tool(schema)])
    issue = s.snapshot()['native_protocol_diagnostics'][0]['issues'][0]
    assert issue['missing_required_properties'] == ['object', 'source']
    assert issue['unexpected_properties'] == sorted(EXTRAS)
    for name in EXTRAS:
        schema['properties'][name] = {}
    validate_schema_instance({**EXTRAS, 'object': 'a', 'source': 'b'}, schema)


def test_NC06_12_request_snapshots_not_aliased_or_stale():
    t = tool(); view = NativeCallContractView.from_tool(t)
    exported = t.to_openai(); exported['function']['parameters']['required'].append('source')
    assert t.input_schema['required'] == [] and json.loads(view.input_schema_json)['required'] == []
    s, _ = session([{}, {'object': 'a'}])
    first = s.next_turn('one', tools=[t]); s.acknowledge_tool_result(first.tool_calls[0].call_id, {})
    t.input_schema['required'].append('object')
    s.next_turn('two', tools=[t])
    contracts = s.snapshot()['native_call_contracts']
    assert contracts[0]['contracts'][0]['required_root_properties'] == []
    assert contracts[1]['contracts'][0]['required_root_properties'] == ['object']
    assert contracts[0]['contracts'][0]['schema_sha256'] != contracts[1]['contracts'][0]['schema_sha256']


@pytest.mark.parametrize('schema,value,path', [
    ({'type': 'object', 'properties': {'arguments': SCHEMA}}, {'arguments': EXTRAS}, '$.arguments'),
    ({'type': 'array', 'items': {'type': 'integer'}}, [True], '$[0]'),
    ({'type': 'object', 'properties': {'flag': {'type': 'boolean'}}}, {'flag': 1}, '$.flag'),
    ({'type': 'array', 'uniqueItems': True}, [1, 1], '$'),
    ({'type': 'array', 'minItems': 2}, [], '$'),
])
def test_NC08_nested_and_original_scalar_semantics(schema, value, path):
    with pytest.raises(SchemaValidationError) as exc:
        validate_schema_instance(value, schema)
    d = schema_diagnostics(value, schema, exc.value)
    assert any(r['failure_path'] == path or r['object_path'] == path for r in d['issues'])


@pytest.mark.parametrize('keyword', ['oneOf', 'anyOf'])
def test_NC08_branch_identity_no_required_union(keyword):
    schema = {'type': 'object', keyword: [dict(SCHEMA, required=['object']), dict(SCHEMA, required=['source'])]}
    with pytest.raises(SchemaValidationError) as exc:
        validate_schema_instance({}, schema)
    d = schema_diagnostics({}, schema, exc.value)
    branches = [r for r in d['issues'] if r.get('branch')]
    assert [r['missing_required_properties'] for r in branches] == [['object'], ['source']]
    assert NativeCallContractView.from_tool(tool(schema)).to_dict()['required_root_properties'] is None


@pytest.mark.parametrize('calls,code', [
    ([NativeToolCall('x', 'unknown', {})], 'runtime_agent_schema_error'),
    ([NativeToolCall('x', 'invoke_impl_fixture', {}), NativeToolCall('y', 'invoke_impl_fixture', {})], 'runtime_agent_multiple_tool_calls'),
    ([], 'agent_protocol_no_action'),
])
def test_NC09_non_schema_failures_keep_classification(calls, code):
    s, _ = session([calls, calls])
    with pytest.raises(AgentProtocolError) as exc:
        s.next_turn('call', tools=[tool()])
    assert exc.value.code == code and not exc.value.diagnostics


def test_NC14_all_35_historical_rejections_remain_invalid_with_complete_diagnostics():
    rows = json.loads((FIXTURES/'schema_failure_evidence.json').read_text())
    assert len(rows) == 35
    for row in rows:
        specs = [NativeToolSpec(t['name'], t['description'], t['parameters']) for t in row['offered_matching_tool']]
        s, _ = session([])
        turn = AgentTurn('', [NativeToolCall(**c) for c in row['tool_calls']], 'tool_calls', 0, 0, 0, None, 0)
        with pytest.raises(AgentProtocolError) as error:
            s._validate_turn(turn, specs)
        assert error.value.code == row['code']
        if len(row['tool_calls']) != 1:
            assert not error.value.diagnostics
            continue
        for raw in row['tool_calls']:
            offered = next(t for t in row['offered_matching_tool'] if t['name'] == raw['name'])
            spec = NativeToolSpec(offered['name'], offered['description'], offered['parameters'])
            call = NativeToolCall(**raw)
            with pytest.raises(SchemaValidationError) as exc:
                validate_schema_instance(call.arguments, spec.input_schema)
            d = rejection_diagnostic(spec, call, exc.value)
            expected = sorted(set(call.arguments) - set(spec.input_schema['properties']))
            assert d['issues'][0]['unexpected_properties'] == expected


def test_NC02_repair_enters_real_preflight_and_toolrunner_only_after_valid(tmp_path):
    from test_r10_runtime import setup, action
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    system, ctx, occ, invocations, p = setup(tmp_path, lambda r, n: action(r, 'GO_TO', destination='cabinet_1'))
    try:
        ex = system.orchestrator.node_executor
        run_runtime_step(ex, 'preparation', occ, ctx, invocations, [])
        p.choose = lambda r, n: action(r, 'OPEN', object='cabinet_1')
        run_runtime_step(ex, 'preparation', occ, ctx, invocations, [])
        before = len(ctx.trace_builder.trace.environment_actions)
        usage_before = len(system.usage.events)
        class Invoke(RawProvider):
            def complete(self, messages, *, tools):
                assert len(ctx.trace_builder.trace.environment_actions) == before
                selected = next(t for t in tools if t.call_kind == 'implementation')
                return super().complete(messages, tools=[selected])
        raw = Invoke([{**EXTRAS, 'object': 'egg_1', 'source': 'cabinet_1'}, {'object': 'egg_1', 'source': 'cabinet_1'}])
        p.complete = raw.complete
        invocations = system.invocation_compiler.compile_candidates(occ, ctx.binding_store, task_id=ctx.task_id)
        result = run_runtime_step(ex, 'preparation', occ, ctx, invocations, [])
        assert len(raw.requests) == 2 and len(system.usage.events) == usage_before + 2
        assert len(ctx.trace_builder.trace.environment_actions) > before
        assert result.result.atomic_effect_passed
    finally:
        system.close()


def test_NC12_13_final_http_normal_and_repair_match_session_and_usage(monkeypatch):
    from atomic_skillgraph.agents.provider import OpenAICompatibleProvider
    from test_deepseek_protocol import _config, _Response, _success
    from experiments.native_contract_audit import audit_trace
    monkeypatch.setenv('TEST_DEEPSEEK_KEY', 'test-only')
    sent = []
    def post(_url, *, headers, json, timeout):
        sent.append(copy.deepcopy(json))
        n = len(sent)
        result = _success(f'call{n}', 'invoke_impl_fixture')
        result['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = __import__('json').dumps(EXTRAS if n == 1 else {})
        return _Response(result, request_id=f'provider{n}')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    p = OpenAICompatibleProvider(_config())
    s = ReplayAgentSession(p, system_prompt='fixture', usage_ledger=UsageLedger(), usage_bucket='runtime_seeded')
    s.next_turn('call', tools=[tool()])
    trace = {'agent_sessions': [{'session_id': s.session_id, 'snapshot': s.snapshot()}],
        'provider_requests': list(p.request_records), 'llm_usage': [e.to_dict() for e in s.usage_ledger.events]}
    report = audit_trace(trace)
    assert report['passed'], report
    assert len(report['checks']) == 2 and len(report['diagnostics']) == 1
    assert report['diagnostics'][0]['link_complete']
    assert report['diagnostics'][0]['repair_usage_event_ids']
    assert report['diagnostics'][0]['repair_physical_request_ids']
    assert report['generated_by_kind'] == {'implementation': 2}
    assert report['repair_passed_by_kind'] == {'implementation': 1}
    assert sent[0]['tools'] == sent[1]['tools']
    assert sent[1]['messages'][-2]['role'] == 'tool'
    assert all(k in sent[1]['messages'][-1]['content'] for k in EXTRAS)
    trace['agent_sessions'] = []
    assert not audit_trace(trace)['passed']


def test_NC13_structured_submissions_count_without_runtime_execution_rows():
    from experiments.native_contract_audit import audit_trace
    s, _ = session([{}])
    spec = NativeToolSpec('submit_fixture', 'fixture', SCHEMA,
                          call_kind='structured_submission', scope='structured')
    s.next_turn('submit', tools=[spec])
    report = audit_trace({'agent_sessions': [{'session_id': s.session_id, 'snapshot': s.snapshot()}]})
    assert report['generated_by_kind'] == {'structured_submission': 1}


def test_MT01_accepted_turn_ordinal_after_repair_is_not_usage_ordinal():
    from experiments.release4_metrics import trace_metrics
    trace = {'llm_usage': [{'event_id': f'e{i}', 'session_id': 's', 'turn_index': i,
        'prompt_tokens': 7, 'completion_tokens': 3, 'reasoning_tokens': 1,
        'provider_metadata': {'request_id': f'p{i}'}} for i in (0, 1)],
        'provider_requests': [{'request_id': f'r{i}', 'provider_request_id': f'p{i}',
            'session_id': 's', 'usage_status': 'reported'} for i in (0, 1)],
        'agent_turns': [{'session_id': 's', 'turn_index': 0, 'tool_call_ids': ['accepted'],
            'provider_metadata': {'request_id': 'p1'}}],
        'native_tool_calls': [{'session_id': 's', 'call_id': 'accepted', 'tool_name': 'invoke_support_atomic',
            'arguments': {}, 'preflight_result': {'accepted': False}}]}
    report = trace_metrics(trace)
    assert report['all_recorded_usage']['recorded_total_tokens'] == 20
    assert report['support_rejected_decision_cost']['recorded_total_tokens'] == 10
    assert report['support_rejected_decision_requests'] == ['r1']
