"""OF05-OF07: actual mapping proof core and provider-visible Support schemas."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest

from test_r101_boundaries import boundary
from atomic_skillgraph.runtime.support_request import preview_support_mapping_authority, prove_request
from atomic_skillgraph.runtime.support_interface import project_candidate, public_arguments_schema
from atomic_skillgraph.runtime.support_retriever import SupportCandidate, SupportRoleMapping
from atomic_skillgraph.runtime.node_executor import NodeExecutor
from atomic_skillgraph.agents.protocol import validate_schema_instance, SchemaValidationError


@pytest.mark.parametrize('relation', [True, False])
def test_preview_is_pure_and_agrees_with_current_execution_proof(relation):
    request, parent, ctx = boundary(relation=relation)
    before = copy.deepcopy(request)
    bindings = copy.deepcopy(vars(ctx.binding_store))
    preview = preview_support_mapping_authority(request, parent, ctx)
    assert request == before and vars(ctx.binding_store) == bindings
    assert ctx.trace_builder.trace.metadata == {}
    result = prove_request(request, parent, ctx, agent_selected=True)
    assert result.passed == (preview['status'] == 'proven')
    if result.passed:
        assert request.output_mapping == preview['output_mapping']
    else:
        assert ctx.trace_builder.trace.metadata['r101_metrics']['support_mapping_authority_rejects'] == 1


def schema(maximum=8):
    return {'type': 'object', 'required': ['locations', 'allow_open'], 'additionalProperties': False,
        'properties': {'locations': {'type': 'array', 'items': {'type': 'string'},
            'minItems': 1, 'maxItems': maximum, 'uniqueItems': True}, 'allow_open': {'type': 'boolean'}}}


def candidate():
    return SupportCandidate('skill://helper@1.0.0', 2, ('entity', 'location'), ('entity', 'location'),
        ('entity.discovered_at',), (), role_mappings=(
            SupportRoleMapping('entity', 'object', 'entity'), SupportRoleMapping('location', 'station', 'entity')),
        input_schema=schema(), execution_available=True)


def test_invalid_station_does_not_hide_legal_object_mapping():
    request, parent, ctx = boundary(relation=False)
    compiler = SimpleNamespace(skills=SimpleNamespace(implementations_for=lambda *a, **kw: []), mode='frozen')
    projected = project_candidate(candidate(), request.producer, parent, request.consumer, ctx, compiler,
                                  [SimpleNamespace(spec=SimpleNamespace(input_schema=schema()))])
    assert projected.allowed_output_mappings == ({'entity': 'object'},)
    assert projected.mapping_proven and not projected.input_ready
    assert projected.execution_available
    assert projected.score == 1
    assert any(p['status'] == 'needs_anchor' for p in projected.mapping_previews)
    assert ctx.trace_builder.trace.metadata == {}


def test_joint_relation_is_one_option_not_cartesian_role_choices():
    request, parent, ctx = boundary(relation=True)
    compiler = SimpleNamespace(skills=SimpleNamespace(implementations_for=lambda *a, **kw: []), mode='frozen')
    result = project_candidate(candidate(), request.producer, parent, request.consumer, ctx, compiler,
                               [SimpleNamespace(spec=SimpleNamespace(input_schema=schema()))])
    assert result.allowed_output_mappings == ({'entity': 'object', 'location': 'station'},)
    assert NodeExecutor._resolve_support_output_mapping(SimpleNamespace(arguments={}), result) == result.allowed_output_mappings[0]
    assert NodeExecutor._resolve_support_output_mapping(SimpleNamespace(arguments={
        'output_mapping': {'location': 'station'}}), result) is None
    # Preview is not authority for a later state where the parent anchor disappeared.
    ctx.binding_store = type(ctx.binding_store)()
    assert not prove_request(request, parent, ctx, agent_selected=True).passed


def test_native_schema_exposes_full_shared_constraints_and_error_length():
    tool = NodeExecutor._support_tool([candidate()])
    assert tool.input_schema['properties']['arguments'] == schema()
    arguments = {'locations': [f'place_{i}' for i in range(8)], 'allow_open': False}
    validate_schema_instance(arguments, schema())
    with pytest.raises(SchemaValidationError) as caught:
        validate_schema_instance({**arguments, 'locations': arguments['locations'] + ['extra']}, schema())
    assert caught.value.constraint == {'maxItems': 8}
    assert caught.value.actual == {'length': 9}
    with pytest.raises(SchemaValidationError):
        validate_schema_instance({**arguments, 'allow_open': 1}, schema())


def test_mixed_constraints_do_not_intersect_or_require_provider_conditionals():
    candidates = [candidate(), replace(candidate(), atomic_ref='skill://other@1.0.0', input_schema=schema(12))]
    interface = public_arguments_schema(candidates)
    assert 'maxItems' not in interface['properties']['locations']
    assert interface['properties']['allow_open']['type'] == 'boolean'
    assert all('oneOf' not in str(s) and 'if' not in s for s in [interface])
    validate_schema_instance({'locations': [str(i) for i in range(10)], 'allow_open': True}, candidates[1].input_schema)


def test_stable_constraint_reaches_next_request_and_deduplicates_across_sessions(tmp_path):
    from test_r10_runtime import setup, action
    from atomic_skillgraph.runtime.negative_memory import remember_rejection, current_rejections
    from atomic_skillgraph.runtime.runtime_step import run_runtime_step
    from atomic_skillgraph.agents.protocol import NativeToolCall
    system, ctx, occurrence, invocations, provider = setup(tmp_path,
        lambda request, count: action(request, 'GO_TO', destination='cabinet_1'))
    try:
        for revision in (0, 1):
            ctx.world_revision = revision
            call = NativeToolCall(f'bad_{revision}', 'invoke_support_atomic', {
                'support_atomic_ref': 'skill://helper@1.0.0',
                'arguments': {'locations': [f'place_{n}' for n in range(9)], 'allow_open': False}})
            with pytest.raises(SchemaValidationError) as error:
                validate_schema_instance(call.arguments['arguments'], schema())
            remember_rejection(ctx, occurrence, call, {'accepted': False,
                'error': 'support_atomic_input_schema_invalid', 'error_code': 'support_atomic_input_schema_invalid',
                'support_atomic_ref': 'skill://helper@1.0.0', 'constraint_scope': 'stable_schema',
                'argument_path': error.value.path, 'expected_constraint': error.value.constraint,
                'actual_summary': error.value.actual})
        ctx.world_revision = 0
        assert len(current_rejections(ctx, occurrence)) == 1
        run_runtime_step(system.orchestrator.node_executor, 'preparation', occurrence, ctx, invocations, [])
        feedback = provider.requests[-1].policy_context['rejected_candidates']
        assert len(feedback) == 1
        assert feedback[0]['constraint_feedback']['expected_constraint'] == {'maxItems': 8}
        assert feedback[0]['constraint_feedback']['actual_summary'] == {'length': 9}
        assert feedback[0]['candidate_group'] == []
    finally:
        system.close()


def test_last_step_projection_keeps_actionable_boundaries_not_tool_internals():
    from atomic_skillgraph.runtime.runtime_step import public_step_feedback
    from atomic_skillgraph.agents.protocol import NativeToolCall
    call = NativeToolCall('bad', 'invoke_support_atomic', {'arguments': {}})
    boundary = {'error_code': 'support_atomic_input_schema_invalid', 'argument_path': '$.arguments.locations',
        'expected_constraint': {'maxItems': 8}, 'actual_summary': {'length': 9},
        'allowed_output_mappings': [{'entity': 'object'}], 'required_anchor_or_relation': [], 'relevant_revision': 3}
    result = public_step_feedback(call, {'accepted': False, **boundary}, before_revision=3, after_revision=3)
    assert {key: result[key] for key in boundary} == boundary
    scope = {'outcome': 'scope_exhausted', 'checked_scope': ['place_a'], 'global_absence_claimed': False}
    result = public_step_feedback(call, {'accepted': True, 'result': {'failure_code': 'tool_ir_execution_error',
        'tool_results': [{'tool_path_evidence': {'final_effect_result': {'scope_diagnostic': scope}},
                          'program': 'never replicate program or full path'}]}}, before_revision=3, after_revision=3)
    assert result['helper_result']['scope_diagnostics'] == [scope]
    assert 'program' not in str(result)
