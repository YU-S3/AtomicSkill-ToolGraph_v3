"""OF08/09: authored search through real interpreter and Harness authority."""
import copy
import pytest
from atomic_skillgraph.deployment.oldfirst_revision import revise_atomic, author_program
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from test_r10_runtime import setup


def search():
    job = {'job_id': 'scope_test', 'target_ref': 'skill://scope_test@1.0.0',
           'source_ref': None, 'query_input_role': 'query', 'output_roles': ['entity', 'location'],
           'tool_ref': 'tool://scope_test@1.0.0'}
    atomic = revise_atomic(job, {}, job)
    tool = author_program(job, atomic)
    report = ToolStaticValidator().validate_tool_asset(tool, atomic, AlfWorldAdapter(split='train'))
    assert report.passed, report
    tool.safety.update(reviewed=True, review_basis='test_static_validation')
    return atomic, tool


@pytest.mark.parametrize('query,expected', [('egg', True), ('missing_kind', False)])
def test_bounded_search_continues_past_no_match_and_never_takes(tmp_path, query, expected):
    system, ctx, occurrence, _, provider = setup(tmp_path, lambda *a: pytest.fail('program called provider'))
    try:
        atomic, tool = search()
        report = ToolStaticValidator().validate_tool_asset(tool, atomic, AlfWorldAdapter(split='train'))
        assert report.passed, report
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(tool,
            {'query': query, 'locations': list(ctx.harness.case.locations), 'allow_open': True},
            ctx, occurrence_id=occurrence.occurrence_id)
        assert result.completed == expected, result
        assert not provider.requests
        assert 'TAKE' not in [a.action_type for a in ctx.trace_builder.trace.environment_actions]
        if expected:
            assert result.output_candidates == {'entity': 'egg_1', 'location': 'cabinet_1'}
            assert result.atomic_effect_passed
            assert not result.tool_path_evidence['final_effect_result'].get('scope_diagnostic')
        else:
            assert not result.output_candidates
            assert result.failure_code == 'tool_ir_execution_error'
            diagnostic = result.tool_path_evidence['final_effect_result']['scope_diagnostic']
            assert diagnostic['outcome'] == 'scope_exhausted'
            assert [r['scope_value'] for r in diagnostic['checked_scope']] == list(ctx.harness.case.locations)
            assert not diagnostic['global_absence_claimed']
    finally:
        system.close()


def test_unreached_scope_is_not_exhaustion(tmp_path):
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *a: pytest.fail('program called provider'))
    try:
        _, tool = search()
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(tool,
            {'query': 'egg', 'locations': ['unreachable'], 'allow_open': False}, ctx,
            occurrence_id=occurrence.occurrence_id)
        assert not result.completed and not result.output_candidates
        assert result.failure_code == 'tool_ir_selector_no_match'
        assert not result.tool_path_evidence['final_effect_result'].get('scope_diagnostic')
    finally:
        system.close()
