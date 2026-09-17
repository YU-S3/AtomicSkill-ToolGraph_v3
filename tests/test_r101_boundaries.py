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
    finally:
        system.close()
