"""Frozen R10.1 boundary regressions; no benchmark policy decisions."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution
from atomic_skillgraph.core.contracts import ParameterSpec, SemanticPredicate, TaskContract
from atomic_skillgraph.core.results import RuntimeOccurrence, RuntimeRepeatConstraint
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.support_request import SupportRequest, prove_request, consumer_guard, transfer_inputs
from test_r92_support_and_public_memory import _atomic


def boundary(*, relation=True):
    parent = _atomic('parent', inputs=[ParameterSpec('object', 'entity'), ParameterSpec('station', 'entity')],
        preconditions=[SemanticPredicate('entity.discovered_at', {'entity': '$object', 'location': '$station'},
            effect_domain='evidence')] if relation else [])
    helper = _atomic('helper', inputs=[ParameterSpec('query', 'entity', required_resolution='semantic')],
        outputs=[ParameterSpec('entity', 'entity'), ParameterSpec('location', 'entity')],
        effects=[SemanticPredicate('entity.discovered_at', {'entity': '$entity', 'location': '$location'}, effect_domain='evidence')])
    helper.validator_spec = {'output_semantic_constraints': {'entity': {'compatible_with_input': 'query'}}}
    occurrence = RuntimeOccurrence('second', 'parent_occ', parent.ref, [], {}, [], [])
    store = RuntimeBindingStore()
    store.commit_grounded(occurrence.occurrence_id, {'object': RuntimeBinding('object', 'object_family', 'entity',
        BindingSource.TASK, BindingStatus.GROUNDED, BindingResolution.SEMANTIC)})
    ctx = SimpleNamespace(binding_store=store, task_contract=TaskContract(), world_revision=0,
        harness=SimpleNamespace(semantic_value_compatible=lambda **kw: kw['concrete_value'] in {'object_1', 'object_2'}),
        trace_builder=SimpleNamespace(trace=SimpleNamespace(metadata={})))
    request = SupportRequest('root', occurrence, helper, {'query': 'object'}, {'location': 'station'})
    return request, parent, ctx


@pytest.mark.parametrize('selected', [False, True])
def test_B01_B02_alias_does_not_authorize_unknown_station(selected):
    request, parent, ctx = boundary(relation=False)
    from atomic_skillgraph.core.bindings import GroundingConstraint, GroundingConstraintKind, BindingExpression, BindingExprKind
    # Several entry arguments alone must not invent a producer/consumer
    # relation. This was the task36/46 pre-execution boundary.
    request.grounding_constraints = [GroundingConstraint('entry', GroundingConstraintKind.HARNESS_AFFORDANCE,
        'PROCESS', {'item': BindingExpression(BindingExprKind.SKILL_INPUT, source_role='object'),
                    'station': BindingExpression(BindingExprKind.SKILL_INPUT, source_role='station')})]
    ctx.harness.public_catalog_relation_schema = lambda: []
    ctx.harness.semantic_predicate_schema = lambda: []
    report = prove_request(request, parent, ctx, agent_selected=selected)
    assert not report.passed
    assert report.failure_codes == ['support_mapping_authority_missing']
    assert ctx.binding_store.snapshot_for_node(request.consumer).keys() == {'object'}


@pytest.mark.parametrize('selected', [False, True])
def test_B03_B08_B11_complete_relation_transfers_inputs_only(selected):
    request, parent, ctx = boundary()
    assert prove_request(request, parent, ctx, agent_selected=selected).passed
    assert request.output_mapping == {'location': 'station', 'entity': 'object'}
    result = SimpleNamespace(validated_outputs={'entity': 'object_2', 'location': 'place_2'}, atomic_witness_refs=['joint_witness'])
    before = copy.deepcopy(ctx.binding_store.repeat_state)
    assert transfer_inputs(request, parent, result, ctx).passed
    assert ctx.binding_store.snapshot_for_node(request.consumer)['station'].value == 'place_2'
    assert ctx.binding_store.repeat_state == before
    assert not ctx.binding_store._outputs
    assert len(ctx.trace_builder.trace.metadata['support_input_transfers']) == 1


def test_B04_missing_correlated_output_never_partially_commits():
    request, parent, ctx = boundary()
    assert prove_request(request, parent, ctx).passed
    before = copy.deepcopy(vars(ctx.binding_store))
    result = SimpleNamespace(validated_outputs={'location': 'place_2'}, atomic_witness_refs=['partial'])
    assert not transfer_inputs(request, parent, result, ctx).passed
    assert vars(ctx.binding_store) == before


def test_B04_crossed_relation_values_cannot_borrow_separate_witnesses():
    from atomic_skillgraph.core.bindings import GroundingConstraint, GroundingConstraintKind, BindingExpression, BindingExprKind
    request, parent, ctx = boundary(relation=False)
    request.output_mapping['entity'] = 'object'
    constraint = GroundingConstraint('entry', GroundingConstraintKind.HARNESS_AFFORDANCE, 'TAKE',
        {'object': BindingExpression(BindingExprKind.SKILL_INPUT, source_role='object'),
         'source': BindingExpression(BindingExprKind.SKILL_INPUT, source_role='station')})
    request.grounding_constraints = [constraint]
    ctx.harness.public_catalog_relation_schema = lambda: [{'action_type': 'TAKE', 'predicates': [
        {'predicate': 'entity.discovered_at', 'argument_mapping': {'entity': 'object', 'location': 'source'}}]}]
    ctx.harness.semantic_predicate_schema = lambda: [SimpleNamespace(predicate='entity.discovered_at', effect_domain='evidence')]
    assert prove_request(request, parent, ctx).passed
    ctx.action_catalog = [SimpleNamespace(revision=0, action_type='TAKE', arguments={'object': obj, 'source': loc})
        for obj, loc in [('object_1', 'place_1'), ('object_2', 'place_2')]]
    ctx.evidence_store = SimpleNamespace(match_constraint=lambda *args: ['complete_action_witness'])
    before = copy.deepcopy(vars(ctx.binding_store))
    crossed = SimpleNamespace(validated_outputs={'entity': 'object_2', 'location': 'place_1'}, atomic_witness_refs=['different_facts'])
    assert not transfer_inputs(request, parent, crossed, ctx).passed
    assert vars(ctx.binding_store) == before
    matching = SimpleNamespace(validated_outputs={'entity': 'object_2', 'location': 'place_2'}, atomic_witness_refs=['same_fact'])
    assert transfer_inputs(request, parent, matching, ctx).passed


def test_B05_B06_B07_parent_repeat_known_values_and_fresh_boundary():
    request, parent, ctx = boundary()
    repeat = RuntimeRepeatConstraint('twice', 2, (('first',), ('second',)), ('item',), (),
        {'first': {'item': 'object'}, 'second': {'item': 'object'}})
    ctx.binding_store.configure_repeat_constraints([repeat])
    ctx.binding_store.commit_repeat_bindings('first', {'object': 'object_1'}, effect_passed=True)
    assert not prove_request(request, parent, ctx).passed  # no implicit exclude for fresh discovery
    request, parent, fresh_ctx = boundary()
    fresh_ctx.binding_store = ctx.binding_store
    assert prove_request(request, parent, fresh_ctx, agent_selected=True).passed
    assert not consumer_guard(request, parent, {'object': 'object_1'}, fresh_ctx).passed
    assert consumer_guard(request, parent, {'object': 'object_2'}, fresh_ctx).passed
    frame = ctx.binding_store.repeat_execution_frame('second')
    assert frame['iteration_index'] == 1 and frame['count'] == 2
    assert frame['committed_distinct_values'] == {'item': ['object_1']}


def test_D07_D08_execution_cache_ignores_uuid_but_not_program_arguments_or_world(tmp_path):
    from test_r10_runtime import setup
    from atomic_skillgraph.runtime.invocation_transaction import execution_cache_key
    system, ctx, occurrence, invocations, _ = setup(tmp_path, lambda *args: pytest.fail('no model'))
    try:
        occurrence = replace(occurrence, step_id='support::first')
        compiled = invocations[0]
        key = execution_cache_key(compiled, {'object': 'object_1'}, occurrence, ctx)
        renamed = replace(occurrence, step_id='support::second')
        assert execution_cache_key(compiled, {'object': 'object_1'}, renamed, ctx) == key
        assert execution_cache_key(compiled, {'object': 'object_2'}, occurrence, ctx) != key
        changed = copy.deepcopy(compiled)
        changed.tools[0].artifact['r101_program_change'] = True
        assert execution_cache_key(changed, {'object': 'object_1'}, occurrence, ctx) != key
        ctx.world_revision += 1
        assert execution_cache_key(compiled, {'object': 'object_1'}, occurrence, ctx) != key
    finally:
        system.close()


def test_D07_cache_hit_has_no_second_action_or_second_negative_attempt(tmp_path):
    from test_r10_runtime import setup
    from experiments.r10_world_checks import install_fixture, bind
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    system, ctx, _, _, _ = setup(tmp_path, lambda *args: pytest.fail('no model'))
    try:
        # The public world permits GO_TO but not CLOSE here. Fail after GO_TO.
        atomic, impl = install_fixture(system, 'cached_partial', [], [SemanticPredicate('container.open', {'container': '$target'})],
            [('GO_TO', 'destination'), ('CLOSE', 'object')], source_target='cabinet_1')
        occurrence = RuntimeOccurrence('partial', 'partial', atomic.ref, [], {}, [str(impl.ref)], atomic.effects)
        ctx.begin_occurrence(occurrence)
        ctx.budget.begin_node(occurrence.occurrence_id)
        bind(ctx, occurrence, 'cabinet_1')
        compiled = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=ctx.task_id)[0]
        preflight = system.invocation_compiler.autonomous_preflight(compiled, occurrence,
            ctx.binding_store, ctx.evidence_store, ctx.world_revision, task_contract=ctx.task_contract)
        assert preflight.passed
        invoke = lambda: execute_invocation(system.orchestrator.node_executor.implementation_runner,
            compiled, preflight, occurrence, ctx, agent_prepared=True)
        first = invoke()
        assert first.started and not first.atomic_effect_passed
        trace = ctx.trace_builder.trace
        counts = (len(trace.environment_actions), len(trace.implementation_invocations), len(trace.tool_executions))
        second = invoke()
        assert second.cached_rejection and not second.started
        assert counts == (len(trace.environment_actions), len(trace.implementation_invocations), len(trace.tool_executions))
        assert trace.metadata['r101_metrics']['exact_failure_cache_hits'] == 1
    finally:
        system.close()


@pytest.mark.parametrize('origin', ['agent_selected_registered', 'runtime_trial'])
@pytest.mark.parametrize('budget_kind', ['node', 'global'])
def test_D02_D04_budget_interrupt_is_audited_restored_not_refunded(tmp_path, origin, budget_kind):
    from test_r10_runtime import setup
    from experiments.r10_world_checks import install_fixture, bind
    from atomic_skillgraph.core.errors import BudgetExhausted
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    from atomic_skillgraph.traces.canonical import canonical_action_indices
    system, ctx, _, _, _ = setup(tmp_path, lambda *args: pytest.fail('no model'))
    try:
        atomic, impl = install_fixture(system, 'partial', [], [SemanticPredicate('container.open', {'container': '$target'})],
            [('GO_TO', 'destination'), ('OPEN', 'object')], source_target='cabinet_1')
        occurrence = RuntimeOccurrence('partial', 'partial', atomic.ref, [], {}, [str(impl.ref)], atomic.effects)
        ctx.begin_occurrence(occurrence)
        ctx.budget.begin_node(occurrence.occurrence_id)
        bind(ctx, occurrence, 'cabinet_1')
        compiled = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=ctx.task_id)[0]
        preflight = system.invocation_compiler.preflight(compiled, call_name=compiled.spec.name, call_id='boundary',
            arguments={'target': 'cabinet_1'}, occurrence=occurrence, binding_store=ctx.binding_store,
            evidence_store=ctx.evidence_store, revision=ctx.world_revision, arguments_are_agent_proposals=False,
            task_contract=ctx.task_contract)
        assert preflight.passed
        setattr(ctx.budget, 'node_action_budget' if budget_kind == 'node' else 'global_action_budget', 1)
        digest = ctx.harness._runtime_state_digest()
        with pytest.raises(BudgetExhausted):
            execute_invocation(system.orchestrator.node_executor.implementation_runner, compiled, preflight, occurrence, ctx,
                agent_prepared=True, origin=origin, execution_scope='runtime_trial' if origin == 'runtime_trial' else 'registered')
        trace = ctx.trace_builder.trace
        assert ctx.harness._runtime_state_digest() == digest
        assert ctx.budget.used_global_actions == 1
        assert len(trace.environment_actions) == 1 and canonical_action_indices(trace) == []
        assert trace.tool_executions[-1].result['interrupted_by_budget']
        assert trace.tool_executions[-1].result['intrinsic_failure'] is False
        assert trace.implementation_invocations[-1].result['interrupted_by_budget']
        assert trace.metadata['runtime_rollbacks'][-1]['origin'] == origin
        if origin == 'runtime_trial':
            excluded = trace.metadata['runtime_trial_credit_exclusions']
            assert trace.implementation_invocations[-1].attempt_id in excluded['implementation_attempt_ids']
            assert trace.tool_executions[-1].attempt_id in excluded['tool_execution_ids']
            events = system.credit.assign(trace)
            assert not any(event.artifact_ref in {str(impl.ref), str(compiled.tools[0].ref)} for event in events)
    finally:
        system.close()


def test_C01_satisfied_navigation_precedes_entry_affordance_and_has_no_tool_credit(tmp_path, monkeypatch):
    from test_r10_runtime import setup, action
    from experiments.r10_world_checks import install_fixture, bind
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    from atomic_skillgraph.runtime.support_closure import SupportClosure
    system, ctx, original, invocations, _ = setup(tmp_path,
        lambda request, count: action(request, 'GO_TO', destination='cabinet_1'))
    try:
        ex = system.orchestrator.node_executor
        run_runtime_step(ex, 'preparation', original, ctx, invocations, [])
        atomic, impl = install_fixture(system, 'already_at', [], [SemanticPredicate('agent.at_location', {'location': '$target'})],
            [('GO_TO', 'destination')], source_target='cabinet_1')
        occurrence = RuntimeOccurrence('already_at', 'already_at', atomic.ref, [], {}, [str(impl.ref)], atomic.effects)
        bind(ctx, occurrence, 'cabinet_1')
        compiled = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=ctx.task_id)
        before = len(ctx.trace_builder.trace.environment_actions)
        monkeypatch.setattr(system.invocation_compiler, 'autonomous_preflight', lambda *a, **kw: pytest.fail('effect must precede action preflight'))
        assert SupportClosure(ex).close(occurrence, ctx, compiled)
        assert len(ctx.trace_builder.trace.environment_actions) == before
        assert not ctx.trace_builder.trace.tool_executions
        assert not ctx.trace_builder.trace.implementation_invocations
    finally:
        system.close()


def test_C02_child_holds_allows_parent_completion_without_second_take(tmp_path):
    from test_r10_runtime import setup, action
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    from atomic_skillgraph.runtime.support_closure import SupportClosure
    def choose(request, count):
        return action(request, 'GO_TO', destination='cabinet_1') if count == 1 else action(request, 'OPEN') if count == 2 else action(request, 'TAKE', object='egg_1')
    system, ctx, occurrence, invocations, _ = setup(tmp_path, choose)
    try:
        ex = system.orchestrator.node_executor
        for _ in range(3):
            run_runtime_step(ex, 'preparation', occurrence, ctx, invocations, [])
        assert [a.action_type for a in ctx.trace_builder.trace.environment_actions] == ['GO_TO', 'OPEN', 'TAKE']
        assert SupportClosure(ex).close(occurrence, ctx, invocations)
        assert len(ctx.trace_builder.trace.environment_actions) == 3
        assert not ctx.trace_builder.trace.tool_executions
    finally:
        system.close()


def test_D10_inconsistent_restore_stops_as_infrastructure(tmp_path, monkeypatch):
    from test_r10_runtime import setup, action
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    from atomic_skillgraph.runtime.checkpoint import capture, restore
    from atomic_skillgraph.core.errors import AtomicSkillGraphError
    system, ctx, occurrence, invocations, _ = setup(tmp_path,
        lambda request, count: action(request, 'GO_TO', destination='cabinet_1'))
    try:
        checkpoint = capture(ctx, occurrence.occurrence_id)
        run_runtime_step(system.orchestrator.node_executor, 'preparation', occurrence, ctx, invocations, [])
        monkeypatch.setattr(ctx.harness, 'restore_runtime_checkpoint', lambda *args: SimpleNamespace(metadata={'restored_digest': 'wrong'}))
        with pytest.raises(AtomicSkillGraphError) as raised:
            restore(ctx, checkpoint, 'original_failure')
        assert raised.value.code == 'runtime_checkpoint_restore_failed'
        assert raised.value.layer.value == 'infrastructure'
        assert ctx.budget.used_global_actions == 1
        assert not ctx.trace_builder.trace.metadata.get('runtime_rollbacks')
    finally:
        system.close()


def test_D03_successful_support_with_rejected_parent_delivery_is_rolled_back(tmp_path):
    from test_r10_runtime import setup
    from experiments.r10_world_checks import install_fixture, bind
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    from atomic_skillgraph.traces.canonical import canonical_action_indices
    system, ctx, _, _, _ = setup(tmp_path, lambda *args: pytest.fail('no model'))
    try:
        atomic, impl = install_fixture(system, 'delivery_rejected', [],
            [SemanticPredicate('agent.at_location', {'location': '$target'})],
            [('GO_TO', 'destination')], source_target='cabinet_1')
        occurrence = RuntimeOccurrence('delivery', 'delivery', atomic.ref, [], {}, [str(impl.ref)], atomic.effects)
        ctx.begin_occurrence(occurrence)
        bind(ctx, occurrence, 'cabinet_1')
        compiled = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=ctx.task_id)[0]
        preflight = system.invocation_compiler.autonomous_preflight(compiled, occurrence,
            ctx.binding_store, ctx.evidence_store, ctx.world_revision, task_contract=ctx.task_contract)
        assert preflight.passed
        digest = ctx.harness._runtime_state_digest()
        bindings = copy.deepcopy(vars(ctx.binding_store))
        def reject_delivery(result):
            assert result.started and result.atomic_effect_passed
            return SimpleNamespace(passed=False, failure_codes=['runtime_repetition_distinctness_violation'])
        result = execute_invocation(system.orchestrator.node_executor.implementation_runner,
            compiled, preflight, occurrence, ctx, agent_prepared=True,
            origin='agent_selected_support', accept_result=reject_delivery)
        assert result.started and not result.atomic_effect_passed and not result.validated_outputs
        assert result.failure_code == 'runtime_repetition_distinctness_violation'
        assert ctx.harness._runtime_state_digest() == digest
        assert vars(ctx.binding_store) == bindings
        assert ctx.budget.used_global_actions == 1
        assert len(ctx.trace_builder.trace.environment_actions) == 1
        assert canonical_action_indices(ctx.trace_builder.trace) == []
        assert ctx.trace_builder.trace.metadata['runtime_rollbacks'][-1]['origin'] == 'agent_selected_support'
    finally:
        system.close()


def test_C05_C10_short_frame_keeps_concrete_last_action_and_shared_resource_query(tmp_path):
    from test_r10_runtime import setup, action
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    system, ctx, occurrence, invocations, provider = setup(tmp_path,
        lambda request, count: action(request, 'GO_TO', destination='cabinet_1'))
    try:
        ex = system.orchestrator.node_executor
        run_runtime_step(ex, 'preparation', occurrence, ctx, invocations, [])
        run_runtime_step(ex, 'seeded', occurrence, ctx, invocations, [])
        frame = provider.requests[-1].policy_context['execution_frame']
        assert frame['mode'] == 'seeded'
        assert frame['current_occurrence_id'] == occurrence.occurrence_id
        assert frame['remaining_resources']['task_actions'] == ctx.budget.global_action_budget - 1
        assert 'cabinet_1' in str(frame['last_step'])
        assert 'GO_TO' in str(frame['last_step'])
        assert all([m['role'] for m in request.messages] == ['system', 'user'] for request in provider.requests)
    finally:
        system.close()


def test_C04_C05_repeat_refusal_carries_completed_identity_and_actual_action(tmp_path):
    from test_r10_runtime import setup, action
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    def choose(request, count):
        if count < 3:
            return action(request, 'GO_TO', destination='cabinet_1') if count == 1 else action(request, 'OPEN')
        name, arguments = action(request, 'TAKE', object='egg_1')
        arguments['intent'] = 'attempt_current_atomic'
        return name, arguments
    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose)
    try:
        ex = system.orchestrator.node_executor
        for _ in range(2):
            run_runtime_step(ex, 'preparation', occurrence, ctx, invocations, [])
        repeat = RuntimeRepeatConstraint('twice', 2, (('previous',), (occurrence.step_id,)), ('item',), (),
            {'previous': {'item': 'object'}, occurrence.step_id: {'item': 'object'}})
        ctx.binding_store.configure_repeat_constraints([repeat])
        ctx.binding_store.commit_repeat_bindings('previous', {'object': 'egg_1'}, effect_passed=True)
        for _ in range(2):
            run_runtime_step(ex, 'seeded', occurrence, ctx, invocations, [])
        frame = provider.requests[-1].policy_context['execution_frame']
        assert frame['repeat']['iteration_index'] == 1
        assert frame['repeat']['committed_distinct_values'] == {'item': ['egg_1']}
        assert frame['last_step']['action_type'] == 'TAKE'
        assert frame['last_step']['action_arguments']['object'] == 'egg_1'
        assert frame['last_step']['error'] == 'runtime_repetition_distinctness_violation'
        assert provider.requests[-1].policy_context['rejected_candidates']
        assert len(ctx.trace_builder.trace.environment_actions) == 2
        assert len(provider.requests) == 4
    finally:
        system.close()


def test_B10_duplicate_relation_proofs_are_not_ambiguous():
    request, parent, ctx = boundary()
    parent.preconditions *= 2
    request.producer.effects *= 2
    assert prove_request(request, parent, ctx).passed
    assert request.output_mapping == {'location': 'station', 'entity': 'object'}
    assert len([p for p in request.mapping_evidence if p['kind'] == 'joint_relation']) == 1


def test_C06_support_refusal_is_exact_to_arguments_and_state(tmp_path):
    from test_r10_runtime import setup
    from atomic_skillgraph.agents.protocol import NativeToolCall
    from atomic_skillgraph.runtime.negative_memory import remember_rejection, cached_rejection, current_rejections
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *args: pytest.fail('no model'))
    try:
        call = NativeToolCall('refused', 'invoke_support_atomic', {'atomic_ref': 'skill://helper@1.0.0', 'arguments': {'object': 'egg_1'}})
        remember_rejection(ctx, occurrence, call, {'accepted': False, 'error': 'runtime_repetition_distinctness_violation'})
        assert cached_rejection(ctx, occurrence, call)
        assert 'egg_1' in str(current_rejections(ctx, occurrence))
        other = replace(call, arguments={**call.arguments, 'arguments': {'object': 'egg_2'}})
        assert not cached_rejection(ctx, occurrence, other)
        ctx.world_revision += 1
        assert not cached_rejection(ctx, occurrence, call)
    finally:
        system.close()
