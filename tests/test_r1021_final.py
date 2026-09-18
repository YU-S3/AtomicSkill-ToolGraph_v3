"""Final review: real production boundaries, controlled model choices only."""
import copy
from dataclasses import replace

import pytest

from atomic_skillgraph.agents.protocol import AgentTurn
from atomic_skillgraph.core.contracts import ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal
from test_r1021_boundaries import typed_preparation_example, typed_atomicizer
from test_r10_runtime import setup


def test_E01_declared_entry_not_first_sparse_support():
    p, n = typed_preparation_example()
    p.support_event_ids = ['e0']
    n['boundary_authorities']['inputs'][0]['available_revision'] = 1
    with pytest.raises(ValueError, match='entry'):
        typed_atomicizer().validate_and_canonicalize([p], n)


@pytest.mark.parametrize('owner', ['other', 'same'])
def test_E05_E06_history_cannot_veto_or_grant_new_contract(owner):
    p, n = typed_preparation_example()
    n['validations'] = [{'level': 'atomic', 'revision': 2, 'occurrence_id': owner,
                        'result': {'passed': False, 'witness_refs': ['irrelevant']}}]
    c, = typed_atomicizer().validate_and_canonicalize([p], n)
    assert 'irrelevant' not in c.validation_refs
    assert n['validations'][0]['result']['passed'] is False


def raw_proposal(p):
    from atomic_skillgraph.agents.structured_submission import ATOMIC_EXTRACTION_SCHEMA
    data = {k:v for k,v in to_primitive(p).items() if k in ATOMIC_EXTRACTION_SCHEMA['properties']}
    data['event_end'] += 1
    data['guideline'] = {'steps': ['Satisfy the declared capability with public evidence.'], 'notes': []}
    return data


def test_E08_invalid_sibling_range_is_subset_rejection():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.evolution.extractor_session import ExtractorSession
    from atomic_skillgraph.agents import UsageLedger
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    p, n = typed_preparation_example()
    bad = raw_proposal(p); bad.update(phase_id='bad', event_start=1, event_end=1)
    s = ReplayAgentSession(ScriptedAgentProvider([FakeReply.structured({'occurrences': [bad, raw_proposal(p)]})]),
        system_prompt='extractor', usage_ledger=UsageLedger(), usage_bucket='extractor_e1')
    proposed = ExtractorSession(s).propose_atomics(n)
    good, rejected = typed_atomicizer().validate_proposed_subset(proposed, n)
    assert len(good) == len(rejected) == 1
    assert good[0].phase_id == p.phase_id and rejected[0]['phase_id'] == 'bad'


def test_E10_real_runtime_steps_share_occurrence_not_span(tmp_path):
    from test_r102_execution import _opened
    from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
    from atomic_skillgraph.evolution.typed_boundary import public_value_authorities
    system, ctx, occurrence, _, _ = _opened(tmp_path)
    try:
        trace = ctx.trace_builder.trace
        trace.metadata['method_patch'] = '3.2'
        n = TraceNormalizer().build(trace)
        authorities = public_value_authorities(trace)
        n['boundary_authorities']['inputs'].extend(authorities)
        source = next(a for a in authorities if a['value'] == 'cabinet_1' and a['available_revision'] == 0)
        effects = n['actions'][1]['authoritative_positive_effects']
        fact = next(f for f in effects if f['predicate'] == 'container.open')
        n['boundary_authorities']['effects'] = [dict(fact, revision=2)]
        p = AtomicOccurrenceProposal('access', 'make_container_accessible', 0, 1,
            {'container': 'cabinet_1'}, {'accessible': 'cabinet_1'}, [],
            [SemanticPredicate('container.open', {'container': 'cabinet_1'})], 'real accepted preparation',
            support_event_ids=[a['action_id'] for a in n['actions']], effect_witness_refs=[fact['witness_ref']],
            input_provenance_refs={'container': {'authority_ref': source['authority_ref'], 'source_role': source['role']}},
            output_derivations={'accessible': {'kind': 'input_identity', 'input_role': 'container'}},
            input_provenance_contract='code_authority_v3_2', boundary_schema_version='2',
            input_specs=[ParameterSpec('container','entity',required_resolution='concrete')],
            output_specs=[ParameterSpec('accessible','entity',required_resolution='concrete')])
        assert len({a['span_id'] for a in n['actions']}) == 2
        c, = typed_atomicizer().validate_and_canonicalize([p], n)
        assert len(c.action_events) == 2 and c.prefix_events == []
    finally:
        system.close()


def test_E02_E03_E04_sparse_support_order_and_declared_entry():
    p, n = typed_preparation_example()
    original = copy.deepcopy(p)
    baseline, = typed_atomicizer().validate_and_canonicalize([p], n)
    p.support_event_ids.reverse()
    reversed_canonical, = typed_atomicizer().validate_and_canonicalize([p], n)
    assert reversed_canonical.action_events == baseline.action_events
    assert p.support_event_ids == list(reversed(original.support_event_ids))
    p.support_event_ids = ['e0']
    sparse, = typed_atomicizer().validate_and_canonicalize([p], n)
    assert sparse.prefix_events == [] and sparse.event_start == 0
    assert sparse.local_value_authority_refs == ['catalog:discovered']
    assert sparse.input_provenance_refs['target']['available_revision'] == 0
    n['boundary_authorities']['inputs'][1]['available_revision'] = 2
    with pytest.raises(ValueError, match='pre-use'):
        typed_atomicizer().validate_and_canonicalize([p], n)


@pytest.mark.parametrize('mutation', ['no_witness', 'wrong_relation', 'future_input', 'rollback'])
def test_E07_historical_pass_cannot_certify_invalid_proposal(mutation):
    p, n = typed_preparation_example()
    n['validations'] = [{'level':'atomic', 'revision':2, 'result':{'passed':True, 'witness_refs':['history']}}]
    if mutation == 'no_witness': p.effect_witness_refs = []
    if mutation == 'wrong_relation': p.effects = [SemanticPredicate('container.open', {'container':'apple_1'})]
    if mutation == 'future_input': n['boundary_authorities']['inputs'][0]['available_revision'] = 1
    if mutation == 'rollback': n['actions'][1]['canonical_discarded'] = True
    with pytest.raises(ValueError):
        typed_atomicizer().validate_and_canonicalize([p], n)


@pytest.mark.parametrize('mutation', ['other_owner', 'orphan', 'range', 'cycle', 'unlearnable', 'hidden_owner'])
def test_E11_sparse_envelope_cannot_hide_invalid_span(mutation):
    p, n = typed_preparation_example()
    original = n['runtime_spans'][0]
    left = dict(original, span_id='left', action_start=0, action_end=1, occurrence_id='owner')
    right = dict(original, span_id='right', action_start=1, action_end=2, occurrence_id='owner')
    n['runtime_spans'] = [left, right]
    n['actions'][0]['span_id'], n['actions'][1]['span_id'] = 'left', 'right'
    if mutation == 'other_owner': right['occurrence_id'] = 'another'
    if mutation == 'orphan': left['parent_span_id'] = 'missing'
    if mutation == 'range': right['action_start'] = 2
    if mutation == 'cycle': left['parent_span_id'] = 'left'
    if mutation == 'unlearnable': left['learnable'] = False
    if mutation == 'hidden_owner':
        p.support_event_ids = ['e0']
        left['occurrence_id'] = 'unselected_other_owner'
    with pytest.raises(ValueError, match='lineage|RuntimeSpan'):
        typed_atomicizer().validate_and_canonicalize([p], n)


def test_E12_same_owner_parent_chain_and_sparse_entry():
    p, n = typed_preparation_example()
    parent = dict(n['runtime_spans'][0], span_id='parent', occurrence_id='owner')
    n['runtime_spans'] = [parent] + [dict(parent, span_id=f'child{i}',
        parent_span_id='parent', action_start=i, action_end=i+1) for i in range(2)]
    for i, event in enumerate(n['actions']): event['span_id'] = f'child{i}'
    p.support_event_ids = ['e0']
    c, = typed_atomicizer().validate_and_canonicalize([p], n)
    assert c.event_start == 0 and c.prefix_events == []
    n['runtime_spans'][0]['action_end'] = 1
    with pytest.raises(ValueError, match='range'):
        typed_atomicizer().validate_and_canonicalize([p], n)


def test_X01_X02_X07_fresh_stage_messages_authority_and_identity():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from atomic_skillgraph.evolution.extractor_session import ExtractorSession
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    from test_extractor_e2_candidates import _chain, _e2_payload
    p, n = typed_preparation_example()
    n['source_marker'] = 'E1_TRACE_MUST_NOT_REPLAY'
    provider = ScriptedAgentProvider([FakeReply.structured({'occurrences':[raw_proposal(p)]}),
        FakeReply.structured(_e2_payload()), FakeReply.structured(_e2_payload())])
    ledger = UsageLedger()
    extractor = ExtractorSession(session_factory=lambda phase: ReplayAgentSession(provider,
        system_prompt='extractor', usage_ledger=ledger, usage_bucket='extractor_e1' if phase == 'e1' else 'extractor_e2'))
    assert extractor.propose_atomics(n)
    initial = provider.requests[0].policy_context
    assert 'boundary_authorities' not in initial
    assert initial['canonical_trace']['boundary_authorities'] == n['boundary_authorities']
    e2 = extractor.propose_composite(_chain(), [])
    extractor.repair_composite(e2, ValueError('fixed rejection'), _chain(), [])
    for request in provider.requests[1:]:
        assert [m['role'] for m in request.messages] == ['system', 'user']
        assert 'E1_TRACE_MUST_NOT_REPLAY' not in str(request.messages)
    for key in ('canonical_occurrences', 'canonical_control_sequence', 'new_edge_candidates', 'known_existing_edge_evidence'):
        assert provider.requests[1].policy_context[key] == provider.requests[2].policy_context[key]
    snapshots = [s.snapshot() for s in extractor._sessions]
    assert len({s['session_id'] for s in snapshots}) == 3
    assert [s['logical_stage']['phase'] for s in snapshots] == ['e1','e2','e2r']
    assert len({s['logical_stage']['logical_extractor_id'] for s in snapshots}) == 1
    assert len(ledger.events) == 3 and all(s['pending_tool_call'] is None for s in snapshots)
    with pytest.raises(RuntimeError): extractor.repair_composite(e2, ValueError('again'), _chain(), [])


def test_X08_fresh_sessions_do_not_multiply_protocol_repair():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from atomic_skillgraph.evolution.extractor_session import ExtractorSession, ExtractionContentError
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    from test_extractor_e2_candidates import _chain, _e2_payload
    class InvalidEnvelopeProvider(ScriptedAgentProvider):
        def complete(self, messages, *, tools):
            turn = super().complete(messages, tools=tools)
            if len(self.requests) in (1,3):
                return replace(turn, tool_calls=[], finish_reason='stop',
                    replay_assistant_message={'role':'assistant','content':'no native call'})
            return turn
    provider = InvalidEnvelopeProvider([FakeReply.structured({'occurrences':[]}),
        FakeReply.structured({'occurrences':[]}), FakeReply.structured(_e2_payload())])
    extractor = ExtractorSession(session_factory=lambda phase: ReplayAgentSession(provider,
        system_prompt='extractor', usage_ledger=UsageLedger(), usage_bucket='extractor_e1'))
    assert extractor.propose_atomics({'actions':[]}) == []
    with pytest.raises(ExtractionContentError): extractor.propose_composite(_chain(), [])
    assert len(provider.requests) == 3 and extractor._protocol_repair_budget['used'] == 1


def test_R03_R04_R07_feedback_is_public_but_preserves_typed_outputs():
    from atomic_skillgraph.runtime.runtime_step import public_step_feedback
    from atomic_skillgraph.agents.protocol import NativeToolCall
    call = NativeToolCall('call', 'invoke_support_atomic', {'arguments':{'target':'supplied'}})
    failure = {'failure_code':'unavailable', 'message':'not in catalog', 'rollback':True,
        'restored_revision':5, 'raw_tool_body':'SECRET', 'validator_snapshot':'SECRET',
        'failing_action':{'action_type':'OPEN', 'arguments':{'object':'supplied'}, 'program':'SECRET'}}
    payload = {'accepted':True, 'program':'SECRET', 'result':{'completed':False, 'failure_code':'unavailable', 'program':'SECRET'},
        'trial':{'r1_outputs':{}, 'failure_feedback':failure, 'promotion_bundle':'SECRET'}, 'trial_failure':failure}
    feedback = public_step_feedback(call,payload,before_revision=5,after_revision=5)
    assert 'SECRET' not in str(feedback)
    assert feedback['trial']['failure_feedback']['failing_action']['arguments'] == {'object':'supplied'}
    assert feedback['trial_failure']['rollback'] and feedback['helper_result']['completed'] is False
    success = public_step_feedback(call, {'accepted':True,'result':{'completed':True,'validated_outputs':{'found':'actual'}},
        'trial':{'r1_outputs':{'found':'actual'}}}, before_revision=5,after_revision=6)
    assert success['helper_result']['validated_outputs'] == success['trial']['r1_outputs'] == {'found':'actual'}
    assert 'failure_code' not in success['helper_result'] and 'trial_failure' not in success


@pytest.mark.parametrize('constraint', [{}, {'found':{'compatible_with_input':'target'}}])
def test_I01_I04_sparse_constraints_validate_own_evidence(constraint):
    p,n = typed_preparation_example()
    p.output_semantic_constraints = constraint
    assert typed_atomicizer().validate_and_canonicalize([p],n)


@pytest.mark.parametrize('role', ['missing', 'the input target category', 'category'])
def test_I03_constraint_requires_actual_formal_input_role(role):
    p,n = typed_preparation_example()
    p.output_semantic_constraints = {'found':{'compatible_with_input':role}}
    with pytest.raises(ValueError): typed_atomicizer().validate_and_canonicalize([p],n)


def test_X03_allocation_includes_intervening_builder(tmp_path):
    system, _, _, _, _ = setup(tmp_path, lambda *_: pytest.fail('no call'))
    try:
        for i,(bucket,tokens) in enumerate([('extractor_e1',70000),('tool_builder_evolution',120000)]):
            system.usage.record_turn(session_id='offline',turn_index=i,bucket=bucket,
                turn=AgentTurn('', [], 'stop', tokens, 0, tokens, 0, 0))
        session = system._extractor_session('task')
        assert session.snapshot()['budget']['max_total_tokens'] == 72144
    finally:
        system.close()


@pytest.mark.parametrize('mode', [{}, {'rescue':True}, {'cold_start_continuation':True}])
def test_R01_R06_dynamic_next_prompt_keeps_failed_call(tmp_path, mode):
    from experiments.r10_world_checks import install_fixture
    chosen = {}
    def choose(request, count):
        if count == 1:
            return 'invoke_support_atomic', {'support_atomic_ref': chosen['ref'], 'arguments': {'target': 'cabinet_1'}}
        feedback = request.policy_context['task_runtime_frame']['last_step']
        assert feedback['tool'] == 'invoke_support_atomic'
        assert feedback['arguments']['arguments']['target'] == 'cabinet_1'
        assert feedback['helper_result']['failure_code'] == 'tool_ir_action_unavailable'
        assert 'OPEN' in feedback['helper_result']['message']
        assert feedback['helper_result']['completed'] is False
        return 'report_runtime_status', {'status':'give_up','detail':'controlled boundary'}
    system, ctx, _, _, provider = setup(tmp_path, choose)
    atomic, impl = install_fixture(system,'unavailable_controlled',[],
        [SemanticPredicate('container.open',{'container':'$target'})],[('OPEN','object')])
    chosen['ref'] = str(atomic.ref)
    try:
        system.orchestrator.node_executor.run_dynamic(ctx, **mode)
        assert len(provider.requests) == 2
    finally:
        system.close()


def test_X04_X05_shared_request_recheck_overrun_and_task_reset(tmp_path):
    from atomic_skillgraph.agents.protocol import NativeToolSpec
    from atomic_skillgraph.core.errors import BudgetExhausted
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    system, _, _, _, _ = setup(tmp_path, lambda *_: pytest.fail('unused'))
    provider = ScriptedAgentProvider([FakeReply.structured({}, prompt_tokens=73000, completion_tokens=1)])
    system._provider_override = provider
    tool = NativeToolSpec('submit_probe','budget boundary',{'type':'object','properties':{},'additionalProperties':False})
    try:
        for i,(bucket,tokens) in enumerate([('extractor_e1',70000),('tool_builder_evolution',110000),('runtime_dynamic',200000)]):
            system.usage.record_turn(session_id='previous',turn_index=i,bucket=bucket,
                turn=AgentTurn('', [], 'stop', tokens, 0, tokens, 0, 0))
        e2 = system._extractor_session('task','e2')
        assert e2.snapshot()['budget']['max_total_tokens'] == 82144
        system.usage.record_turn(session_id='builder',turn_index=0,bucket='tool_builder_evolution',
            turn=AgentTurn('', [], 'stop', 10000, 0, 10000, 0, 0))
        with pytest.raises(BudgetExhausted): e2.next_turn('test',tools=[tool])
        assert len(provider.requests) == 1
        assert system.usage.events[-1].usage.total_tokens == 73001
        assert e2.snapshot()['shared_budget_requests'][0]['effective_remaining'] == 72144
        assert system._shared_tool_builder_tokens('evolution') == 0
        assert system._shared_tool_builder_tokens('runtime') == 400000
        for session in [system._extractor_session('task','e2r'),system._tool_builder_session('evolution','same')]:
            with pytest.raises(BudgetExhausted): session.next_turn('blocked',tools=[tool])
        assert len(provider.requests) == 1
        system._current_task_usage_start = len(system.usage.events)
        assert system._extractor_session('next','e1').snapshot()['budget']['max_total_tokens'] == 262144
        assert system._shared_tool_builder_tokens('runtime') == 600000
    finally:
        system.close()


def test_E09_global_schema_repair_all_rejected_and_empty_are_distinct():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from atomic_skillgraph.evolution.extractor_session import ExtractorSession, ExtractionContentError
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    class InvalidSchemaProvider(ScriptedAgentProvider):
        def complete(self, messages, *, tools):
            turn = super().complete(messages, tools=tools)
            call = replace(turn.tool_calls[0], arguments={'wrong_field':[]})
            return replace(turn, tool_calls=[call])
    provider = InvalidSchemaProvider([FakeReply.structured({'occurrences':[]})]*2)
    session = ReplayAgentSession(provider,system_prompt='extractor',usage_ledger=UsageLedger(),usage_bucket='extractor_e1')
    with pytest.raises(ExtractionContentError): ExtractorSession(session).propose_atomics({'actions':[]})
    assert len(provider.requests) == 2
    p,n = typed_preparation_example()
    p.event_end = -1
    from atomic_skillgraph.evolution.atomicizer import AtomicProposalBatchRejected
    with pytest.raises(AtomicProposalBatchRejected, match='no valid'):
        typed_atomicizer().validate_proposed_subset([p],n)
    assert typed_atomicizer().validate_proposed_subset([],n) == ([],[])


def test_R08_node_next_prompt_has_actual_action_and_no_automatic_work(tmp_path):
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    from test_r10_runtime import action
    def choose(request,count):
        if count == 2:
            frame = request.policy_context['execution_frame']
            assert frame['last_step']['action_type'] == 'GO_TO'
            assert frame['last_step']['action_arguments'] == {'destination':'cabinet_1'}
            assert frame['last_step']['before_revision'] == 0
            assert frame['last_step']['after_revision'] == 1
        return action(request, 'GO_TO', destination='cabinet_1') if count == 1 else action(request, 'OPEN')
    system,ctx,occurrence,invocations,provider = setup(tmp_path,choose)
    try:
        for _ in range(2): run_runtime_step(system.orchestrator.node_executor,'preparation',occurrence,ctx,invocations,[])
        assert len(provider.requests) == len(ctx.trace_builder.trace.environment_actions) == 2
        assert not ctx.trace_builder.trace.tool_executions
    finally:
        system.close()


def test_R02_preflight_feedback_survives_without_world_action(tmp_path):
    from experiments.r10_world_checks import install_fixture
    chosen = {}
    def choose(request,count):
        if count == 1:
            return 'invoke_support_atomic', {'support_atomic_ref':chosen['ref'],'arguments':{'target':'cabinet_1'}}
        feedback = request.policy_context['task_runtime_frame']['last_step']
        assert not feedback['accepted'] and feedback['error'] and feedback['message']
        assert feedback['arguments']['arguments'] == {'target':'cabinet_1'}
        assert feedback['before_revision'] == feedback['after_revision'] == 0
        return 'report_runtime_status', {'status':'give_up','detail':'preflight boundary observed'}
    system,ctx,_,_,provider = setup(tmp_path,choose)
    atomic,_ = install_fixture(system,'entry_guard',
        [SemanticPredicate('container.open',{'container':'$target'})],
        [SemanticPredicate('container.open',{'container':'$target'})],[('OPEN','object')])
    chosen['ref'] = str(atomic.ref)
    try:
        system.orchestrator.node_executor.run_dynamic(ctx)
        assert len(provider.requests)==2 and not ctx.trace_builder.trace.environment_actions
    finally:
        system.close()


def test_I05_isomorphic_boundary_roles_values_and_source_ids():
    import json
    p,n = typed_preparation_example()
    original, = typed_atomicizer().validate_and_canonicalize([p],n)
    replacements = {'target':'query_role','found':'answer_role','category':'public_query_port',
        'apple':'mug','apple_1':'mug_9',n['trace_id']:'isomorphic_source',p.phase_id:'another_phase'}
    def rename(value):
        if isinstance(value,str): return replacements.get(value,value)
        if isinstance(value,list): return [rename(v) for v in value]
        if isinstance(value,dict): return {replacements.get(k,k):rename(v) for k,v in value.items()}
        return value
    payload = rename(to_primitive(p))
    payload['input_specs'] = [ParameterSpec(**v) for v in payload['input_specs']]
    payload['output_specs'] = [ParameterSpec(**v) for v in payload['output_specs']]
    payload['effects'] = [SemanticPredicate(**v) for v in payload['effects']]
    payload['preconditions'] = [SemanticPredicate(**v) for v in payload['preconditions']]
    changed, = typed_atomicizer().validate_and_canonicalize([AtomicOccurrenceProposal(**payload)],rename(n))
    assert changed.input_bindings == {'query_role':'mug'}
    assert changed.output_bindings == {'answer_role':'mug_9'}
    assert len(changed.action_events)==len(original.action_events)
