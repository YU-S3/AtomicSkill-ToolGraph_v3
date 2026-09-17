"""R10.2.1 production-path positive/negative regression witnesses."""
from dataclasses import replace
import copy
import pytest

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind, GroundingConstraint, GroundingConstraintKind
from atomic_skillgraph.core.contracts import SemanticPredicate
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.evidence_store import GroundingEvidenceStore
from test_r10_runtime import setup
from experiments.r10_world_checks import install_fixture


def test_A02_only_task_action_limit():
    budget = RuntimeBudget(global_action_budget=100)
    budget.begin_node('one')
    for _ in range(100):
        budget.consume_action()
    assert budget.used_node_actions == budget.used_global_actions == 100
    with pytest.raises(BudgetExhausted, match='global') as error:
        budget.consume_action()
    assert error.value.code == 'episode_action_budget_exhausted'
    assert budget.used_global_actions == 100


def exists(store, values, revision=0):
    constraint = GroundingConstraint('exists', GroundingConstraintKind.ARGUMENT_EXISTS,
        argument_mapping={k: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=k) for k in values})
    return store.match_constraint(constraint, values, revision)


def test_E06_each_value_needs_its_own_evidence():
    store = GroundingEvidenceStore()
    store.add_task_evidence('first', 'a')
    store.add_task_evidence('again', 'a')
    assert not exists(store, {'left': 'a', 'right': 'b'})
    assert exists(store, {'left': 'a', 'right': 'a'})
    store.add_task_evidence('second', 'b')
    assert exists(store, {'left': 'a', 'right': 'b'})


@pytest.mark.parametrize('value,wrong', [(True, 1), ('1', 1), ([1, 2], [2, 1]), ({'a': True}, {'a': 1})])
def test_E06_json_values_do_not_coerce(value, wrong):
    store = GroundingEvidenceStore()
    store.add_task_evidence('supplied', value)
    assert exists(store, {'x': copy.deepcopy(value)})
    assert not exists(store, {'x': wrong})
    assert not exists(store, {})


def run_navigation(tmp_path, *, terminal=False, extra_action=False, false_effect=False):
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *_: pytest.fail('no provider'))
    if terminal:
        ctx.harness.case = replace(ctx.harness.case, terminal_action='GO_TO')
    effects = [SemanticPredicate('agent.at_location', {'location': '$target'})]
    if false_effect:
        effects.append(SemanticPredicate('agent.holds', {'object': '$target'}))
    atomic, impl = install_fixture(system, 'r1021_navigation',
        actions=[('GO_TO', 'destination')] * (2 if extra_action else 1),
        effects=effects, preconditions=[])
    tool = system.tools.get(str(impl.tool_bindings[0].tool_ref))
    try:
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(
            tool, {'target': 'cabinet_1'}, ctx, occurrence_id=occurrence.occurrence_id)
        return result, ctx
    finally:
        system.close()


def test_D01_terminal_runs_original_return(tmp_path):
    result, ctx = run_navigation(tmp_path, terminal=True)
    assert ctx.benchmark_terminal()
    assert result.completed and not result.terminal_interrupted
    assert result.executed_action_count == 1
    assert result.atomic_effect_passed


def test_D03_terminal_cannot_skip_next_action_to_return(tmp_path):
    result, ctx = run_navigation(tmp_path, terminal=True, extra_action=True)
    assert ctx.benchmark_terminal()
    assert not result.completed and result.terminal_interrupted
    assert not result.intrinsic_failure
    assert result.executed_action_count == 1


def test_E01_tool_must_fulfil_all_final_effects(tmp_path):
    result, _ = run_navigation(tmp_path, false_effect=True)
    assert not result.completed
    assert result.failure_code == 'tool_ir_final_effect_failed'


def test_D04_terminal_does_not_hide_bad_final_effect(tmp_path):
    result, ctx = run_navigation(tmp_path, terminal=True, false_effect=True)
    assert ctx.benchmark_terminal()
    assert not result.completed and not result.atomic_effect_passed
    assert result.failure_code == 'tool_ir_final_effect_failed'


def test_E02_second_tool_failure_cannot_be_covered_by_atomic_subset(tmp_path):
    from test_r102_execution import _opened, _prepare
    from atomic_skillgraph.core.refs import ToolRef
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        compiled = copy.deepcopy(invocations[0])
        extra = copy.deepcopy(compiled.tools[0])
        extra.ref = ToolRef('required_second_tool', '1.0.0')
        # The real first TAKE already satisfies Atomic agent.holds, but does
        # not erase the author's still-required second Tool invocation.
        extra.artifact['program'][0]['action_type'] = 'GO_TO'
        extra.artifact['program'][0]['argument_mapping'] = {
            'destination': {'kind': 'skill_input', 'source_role': 'object'}}
        compiled.tools.append(extra)
        compiled.implementation.tool_bindings.append(replace(
            compiled.implementation.tool_bindings[0], tool_ref=extra.ref, role='required_second', order=1))
        preflight = _prepare(system, ctx, occurrence, compiled)
        assert preflight.passed
        result = execute_invocation(system.orchestrator.node_executor.implementation_runner,
            compiled, preflight, occurrence, ctx, agent_prepared=True)
        assert result.tool_results[0].completed
        assert not result.completed and not result.atomic_effect_passed
        assert result.failure_code == 'tool_ir_action_unavailable'
        assert not result.validated_outputs and not result.validated_output_bindings
        assert not ctx.harness.held
        assert ctx.budget.used_global_actions == 3  # 2 preparation + TAKE, no refund
    finally:
        system.close()


def test_E03_ordinary_complete_tool_still_succeeds(tmp_path):
    result, _ = run_navigation(tmp_path)
    assert result.completed and result.atomic_effect_passed


@pytest.mark.parametrize('won', [True, False])
def test_D02_D05_last_action_keeps_lexical_return_without_world_work(tmp_path, won):
    from atomic_skillgraph.core.serialization import to_primitive
    from experiments.self_tooling_targeted import selector
    system, ctx, occurrence, _, provider = setup(tmp_path, lambda *_: pytest.fail('no LLM'))
    system.harness.case = replace(system.harness.case, terminal_action='GO_TO', terminal_won=won)
    _, impl = install_fixture(system, 'local_terminal', [],
        [SemanticPredicate('agent.at_location', {'location': '$target'})], [('GO_TO', 'destination')])
    tool = system.tools.get(str(impl.tool_bindings[0].tool_ref))
    # Declared interpreter fixture: original lexical RETURN is inside the loop.
    tool.interface['output_schema'] = {'type': 'object', 'properties': {'found': {'type': 'string'}}, 'required': ['found']}
    tool.artifact['program'] = [{'op': 'FOR_EACH', 'node_id': 'locations',
        'iteration_variable': 'cursor', 'max_iterations': 3,
        'collection_source': selector('GO_TO', 'destination'), 'body': [
            {'op': 'ACTION', 'node_id': 'move', 'action_type': 'GO_TO',
             'argument_mapping': {'destination': {'kind': 'local_variable', 'source_role': 'cursor'}}, 'expected_effects': []},
            {'op': 'RETURN', 'node_id': 'return_local', 'output_sources': {'found': {'source': 'local_variable', 'field': 'cursor'}}}]}]
    ctx.budget.global_action_budget = 1
    try:
        first = ctx.action_catalog[0].arguments['destination']
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(
            tool, {'target': first}, ctx, occurrence_id=occurrence.occurrence_id)
        assert ctx.execution_terminal() and ctx.benchmark_terminal() is won
        assert result.completed and result.output_candidates == {'found': first}
        assert result.executed_action_count == ctx.budget.used_global_actions == 1
        assert not provider.requests
    finally:
        system.close()


def test_G03_live_atomicizer_rejects_implicit_legacy_boundary():
    from atomic_skillgraph.evolution.atomicizer import Atomicizer
    proposal, normalized = typed_preparation_example()
    proposal.input_provenance_contract = 'legacy_action_argument_v1'
    with pytest.raises(ValueError, match='explicit source replay'):
        Atomicizer().validate_and_canonicalize([proposal], normalized)


def typed_preparation_example():
    from atomic_skillgraph.core.contracts import ParameterSpec
    from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal
    from test_v32_e1_input_authority import _normalized
    normalized = _normalized(None)
    final = normalized['actions'][0]
    final.update(event_index=1, before_revision=1, after_revision=2)
    normalized['actions'] = [dict(final, event_index=0, event_id='discover', action_id='discover',
        action_type='LOOK', arguments={}, before_revision=0, after_revision=1,
        authoritative_positive_effects=[]), final]
    normalized['runtime_spans'][0]['action_end'] = 2
    for effect in normalized['boundary_authorities']['effects']:
        effect.update(event_index=1, revision=2)
    normalized['boundary_authorities']['inputs'] = [
        dict(authority_ref='task:anchor', kind='public_binding', source_kind='public_binding',
             role='category', value='apple', semantic_type='entity', resolution='semantic',
             available_revision=0, trace_id=normalized['trace_id'], source_occurrence_id='__task__'),
        dict(authority_ref='catalog:discovered', kind='public_catalog', source_kind='public_catalog',
             role='item', value='apple_1', semantic_type='entity', resolution='concrete',
             available_revision=1, trace_id=normalized['trace_id'], source_occurrence_id=''),
    ]
    proposal = AtomicOccurrenceProposal('discover_and_acquire', 'acquire_requested_entity', 0, 1,
        {'target': 'apple'}, {'found': 'apple_1'}, [],
        [SemanticPredicate('agent.holds', {'object': 'apple_1'})], 'accepted causal preparation',
        support_event_ids=['discover', 'e0'], effect_witness_refs=['action:e0:revision:1'],
        input_provenance_refs={'target': {'authority_ref': 'task:anchor', 'source_role': 'category'}},
        output_derivations={'found': {'kind': 'effect_witness', 'predicate': 'agent.holds', 'argument_role': 'object'}},
        input_provenance_contract='code_authority_v3_2', boundary_schema_version='2',
        input_specs=[ParameterSpec('target', 'entity', required_resolution='semantic')],
        output_specs=[ParameterSpec('found', 'entity', required_resolution='concrete')],
        output_semantic_constraints={'found': {'compatible_with_input': 'target'}},
        local_value_authority_refs=['catalog:discovered'])
    return proposal, normalized


def typed_atomicizer():
    from atomic_skillgraph.evolution.atomicizer import Atomicizer
    from atomic_skillgraph.harness.alfworld import semantic_value_compatible
    return Atomicizer(semantic_value_compatible=semantic_value_compatible)


def test_B01_B02_B07_semantic_input_local_discovery_and_explicit_rename():
    proposal, normalized = typed_preparation_example()
    result = typed_atomicizer().validate_and_canonicalize([proposal], normalized)[0]
    assert result.input_specs[0].required_resolution == 'semantic'
    assert result.output_specs[0].required_resolution == 'concrete'
    assert result.local_value_authority_refs == ['catalog:discovered']
    assert result.input_provenance_refs['target']['role'] == 'category'
    assert len(result.action_events) == 2


@pytest.mark.parametrize('mutation,match', [
    ('future_local', 'pre-use'), ('wrong_trace', 'uniquely supplied'),
    ('missing_local', 'pre-use'), ('rename_without_mapping', 'role mismatch'),
    ('future_input', 'entry'), ('concrete_input', 'resolution mismatch'),
    ('wrong_constraint', 'compatibility'),
])
def test_B02_B06_B07_authority_negative_cases(mutation, match):
    proposal, normalized = typed_preparation_example()
    inputs = normalized['boundary_authorities']['inputs']
    if mutation == 'future_local': inputs[1]['available_revision'] = 2
    if mutation == 'wrong_trace': inputs[1]['trace_id'] = 'another_task'
    if mutation == 'missing_local': proposal.local_value_authority_refs = []
    if mutation == 'rename_without_mapping': proposal.input_provenance_refs['target']['source_role'] = 'target'
    if mutation == 'future_input': inputs[0]['available_revision'] = 1
    if mutation == 'concrete_input': proposal.input_specs[0].required_resolution = 'concrete'
    if mutation == 'wrong_constraint':
        proposal.input_roles['target'] = inputs[0]['value'] = 'pear'
    with pytest.raises(ValueError, match=match):
        typed_atomicizer().validate_and_canonicalize([proposal], normalized)


def test_C01_C03_partial_tool_span_does_not_cover_agent_preparation():
    from types import SimpleNamespace as NS
    from atomic_skillgraph.evolution.maintenance import ExtractionPolicy
    from atomic_skillgraph.traces.schema import RuntimeSpan, ToolExecutionRecord
    trace = NS(benchmark_success=True, learning_eligible=True, infrastructure_failure=False,
        runtime_plan={'source': 'stored_composite'}, task_rescue_required=False, strict_task_success=True,
        node_records=[], implementation_direct_success=True, metadata={},
        environment_actions=[NS(action_id='prepare', accepted=True, span_id='agent'),
                             NS(action_id='tool_action', accepted=True, span_id='tool')],
        runtime_spans=[RuntimeSpan('agent', 'runtime_preparation', 'one', 0, 2, None, True),
                       RuntimeSpan('tool', 'tool', 'one', 1, 2, 'agent', True)],
        tool_executions=[ToolExecutionRecord('invoke', 'one', 'tool://known@1.0.0',
                         {'started': True, 'completed': True}, 'tool')])
    decision = ExtractionPolicy().decide(trace)
    assert decision.reasons == ['uncovered_successful_runtime_work']
    assert decision.uncovered_event_ids == ['prepare']
    trace.metadata['extracted_event_ids'] = ['prepare']
    assert not ExtractionPolicy().decide(trace).should_extract
    trace.metadata = {}
    trace.benchmark_success = False
    assert not ExtractionPolicy().decide(trace).should_extract


def test_C02_empty_e1_is_legal_and_not_retried():
    from atomic_skillgraph.evolution.atomicizer import Atomicizer
    from atomic_skillgraph.evolution.extractor_session import E1_SCHEMA
    from atomic_skillgraph.agents.protocol import validate_schema_instance
    validate_schema_instance({'occurrences': []}, E1_SCHEMA)
    assert Atomicizer().validate_proposed_subset([], {'actions': []}) == ([], [])


@pytest.mark.parametrize('mode', ['initial', 'rescue', 'continuation'])
def test_F01_F03_dynamic_explicit_helper_uses_normal_execution(tmp_path, mode):
    chosen = {}
    def choose(request, count):
        assert 'request_runtime_automation' in {t.name for t in request.tools}
        if count == 1:
            return 'invoke_support_atomic', {'support_atomic_ref': chosen['ref'],
                                             'arguments': {'target': 'cabinet_1'}}
        return 'report_runtime_status', {'status': 'give_up', 'detail': 'fixture boundary'}
    system, ctx, _, _, provider = setup(tmp_path, choose)
    atomic, _ = install_fixture(system, 'task_navigation', actions=[('GO_TO', 'destination')],
        effects=[SemanticPredicate('agent.at_location', {'location': '$target'})], preconditions=[])
    chosen['ref'] = str(atomic.ref)
    before = ctx.budget.used_global_actions
    try:
        system.orchestrator.node_executor.run_dynamic(ctx, rescue=mode == 'rescue', cold_start_continuation=mode == 'continuation')
        assert ctx.budget.used_global_actions == before + 1
        records = ctx.trace_builder.trace.metadata['runtime_support_node_records']
        assert records[-1]['consumer_scope'] == 'task'
        assert records[-1]['parent_atomic_ref'] == ''
        assert records[-1]['direct_result']['atomic_effect_passed']
        assert len(provider.requests) == 2
        assert not ctx.trace_builder.trace.node_records  # no fake graph parent
        assert all([m['role'] for m in r.messages] == ['system', 'user'] for r in provider.requests)
    finally:
        system.close()


@pytest.mark.parametrize('readonly', [False, True])
def test_F02_F06_task_scoped_trial_uses_real_builder_and_runs_once(tmp_path, readonly):
    from experiments import self_tooling_targeted as route
    from experiments.fakes import knowledge_digest
    case = route.RouteCase('task_trial', terminal_action='TAKE')
    def choose(request, count):
        names = {t.name for t in request.tools}
        if 'create_tool' in names:
            atomic = {**request.policy_context['canonical_atomic'], 'ref': request.policy_context['atomic_ref']}
            return 'create_tool', route.scripted_proposal(atomic, case)
        if 'propose_runtime_automation_atomic' in names:
            draft = route.fixed_draft(request, case)
            assert draft['source_occurrence_id'] == '__task__'
            return 'propose_runtime_automation_atomic', draft
        if count == 1:
            return 'request_runtime_automation', {'reason': 'reusable discovery', 'intended_capability': 'find target'}
        target = next(a for a in route.newest_catalog(request) if a['action_type'] == 'TAKE')
        return 'environment_action', {'action_id': target['action_id']}
    system, ctx, _, _, provider = setup(tmp_path, choose)
    system.harness.case = case
    # Frozen policy is set before any dynamic request or trial, not after it.
    system.readonly = readonly
    if readonly:
        from atomic_skillgraph.core.status import RuntimeMode
        system.invocation_compiler.mode = RuntimeMode.FROZEN
    before = knowledge_digest(system.database)
    try:
        result = system.orchestrator.node_executor.run_dynamic(ctx)
        assert result['benchmark_won']
        assert len(provider.requests) == 4  # request, draft, Builder, take
        trial, = ctx.runtime_tool_trials.values()
        assert trial['r1']['admission_eligible'], trial
        assert trial['consumer_scope'] == 'task' and not trial['parent_atomic_ref']
        assert not trial.get('parent_completed_after_trial')
        assert [a.action_type for a in ctx.trace_builder.trace.environment_actions] == ['GO_TO'] * 3 + ['OPEN', 'TAKE']
        assert not ctx.trace_builder.trace.node_records
        assert knowledge_digest(system.database) == before  # only task-local at this boundary
        from atomic_skillgraph.runtime.orchestrator import apply_terminal_outcome
        from atomic_skillgraph.evolution.runtime_support_promotion import collect_observations
        system.orchestrator._persist_v32_task_local_assets(ctx)
        trace = ctx.trace_builder.trace
        channel = ctx.harness.validator_channel()
        apply_terminal_outcome(trace, system.validation.task.terminal(ctx.task_contract, channel, channel.won), channel)
        observations = collect_observations(system, trace)
        assert len(observations) == (0 if readonly else 1)
        if observations:
            assert observations[0]['trial']['consumer_scope'] == 'task'
            assert not observations[0]['trial'].get('parent_completed_after_trial')
        assert knowledge_digest(system.database) == before
    finally:
        system.close()


def test_A01_A03_A04_all_online_buckets_share_one_task_pool(tmp_path):
    from atomic_skillgraph.agents.protocol import AgentTurn
    from atomic_skillgraph.runtime.r1021_metrics import ONLINE_BUCKETS
    system, _, _, _, _ = setup(tmp_path, lambda *_: pytest.fail('no provider needed'))
    try:
        for index, bucket in enumerate(sorted(ONLINE_BUCKETS)):
            system.usage.record_turn(session_id='online', turn_index=index, bucket=bucket,
                turn=AgentTurn('', [], 'stop', 100001, 0, 100001, 0, 0))
        assert system.usage.total().total_tokens == 600006  # no truncation/refund
        assert system._runtime_remaining_tokens('fresh_node')['task_tokens'] == 0
        session = system._runtime_session('runtime_step_dynamic', '__task__')
        with pytest.raises(BudgetExhausted) as error:
            session.next_turn('No rescue allowance', tools=[])
        assert error.value.code == 'runtime_task_token_budget_exhausted'
        system._current_task_usage_start = len(system.usage.events)
        assert system._shared_tool_builder_tokens('runtime') == 600000
        system.usage.record_turn(session_id='offline', turn_index=0, bucket='tool_builder_evolution',
            turn=AgentTurn('', [], 'stop', 120001, 0, 120001, 0, 0))
        assert system._shared_tool_builder_tokens('runtime') == 600000
        assert system._shared_tool_builder_tokens('evolution') == 262144 - 120001
    finally:
        system.close()


@pytest.mark.parametrize('value,semantic_type', [(3, 'integer'), (True, 'boolean'), ([1, {'a': False}], 'array'), ({'b': 2, 'a': 1}, 'object_map')])
def test_E04_E05_typed_publication_and_explicit_flow_preserve_value(value, semantic_type):
    from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
    from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution
    store = RuntimeBindingStore()
    binding = RuntimeBinding('value', value, semantic_type, BindingSource.TOOL_OUTPUT,
        BindingStatus.GROUNDED, BindingResolution.SEMANTIC, ['witness'], 1)
    store.publish_validated_outputs('upstream', {'value': copy.deepcopy(value)}, ['witness'], 1,
        certified_bindings={'value': binding})
    resolved = store.resolve_expression('downstream', BindingExpression(BindingExprKind.DATA_FLOW,
        source_step='upstream', source_role='value'))
    assert resolved.semantic_type == semantic_type
    assert resolved.resolution is BindingResolution.SEMANTIC
    assert type(resolved.value) is type(value) and resolved.value == value


def test_E05_bad_later_value_does_not_partially_publish():
    from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
    from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution
    store = RuntimeBindingStore()
    good = RuntimeBinding('good', 7, 'integer', BindingSource.TOOL_OUTPUT, BindingStatus.GROUNDED,
        BindingResolution.CONCRETE, ['witness'], 1)
    bad = replace(good, role='bad', value=True)
    with pytest.raises(ValueError):
        store.publish_validated_outputs('owner', {'good': 7, 'bad': True}, ['witness'], 1,
            certified_bindings={'good': good, 'bad': bad})
    assert store.snapshot_for_node('owner') == {}
    assert store.validated_outputs('owner') == {}


def test_B03_output_evidence_keeps_its_owner():
    from atomic_skillgraph.evolution.typed_boundary import public_value_authorities
    from atomic_skillgraph.traces.schema import TraceRecord, TaskRecord, TraceBuilder
    from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution
    trace = TraceRecord.create(TaskRecord('scope', 'fake', 'test scope', 'generic', 'scope'), {}, {}, {})
    from atomic_skillgraph.core.serialization import to_primitive
    store = GroundingEvidenceStore(on_change=lambda operation, evidence, revision:
        trace.grounding_evidence_changes.append({'operation': operation, 'payload': to_primitive(evidence), 'revision': revision}))
    binding = RuntimeBinding('found', 'value', 'entity', BindingSource.TOOL_OUTPUT, BindingStatus.GROUNDED,
        BindingResolution.CONCRETE, ['witness'], 0)
    store.add_validated_tool_output('found', 'value', ['witness'], certified_binding=binding, occurrence_id='producer')
    authority, = public_value_authorities(trace)
    assert authority['source_occurrence_id'] == 'producer'
    assert authority['resolution'] == 'concrete'
    proposal, normalized = typed_preparation_example()
    normalized['boundary_authorities']['inputs'][1]['source_occurrence_id'] = 'unrelated_node'
    with pytest.raises(ValueError, match='uniquely supplied'):
        typed_atomicizer().validate_and_canonicalize([proposal], normalized)
