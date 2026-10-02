"""M-T06: real interpreter/validator, controlled public world observations."""
from dataclasses import replace

import pytest

from test_r10_runtime import setup
from test_oldfirst_program import search


def enable_public_receipts(harness, *, incomplete=False):
    """Expose the finite test world's visible objects through the real parser."""
    from atomic_skillgraph.harness.public_discovery import VERSION
    harness.public_discovery_version = VERSION
    harness._refresh_public_discovery(None, True)
    execute = harness.execute_action

    def step(action_id, revision):
        action = harness._catalog.get(action_id, revision)
        result = execute(action_id, revision)
        objects = ['a ' + a.arguments['object'].replace('_', ' ') for a in result.catalog
                   if a.action_type == 'TAKE']
        location = harness.location.replace('_', ' ')
        closed = (harness.location == harness.case.locations[-1]
                  and harness.case.openable and not harness.opened)
        observation = (f'The {location} is closed.' if closed else
                       f'On the {location}, you see ' + (', '.join(objects) if objects else 'nothing') + '.')
        if incomplete:
            observation = 'The inspection result could not be read.'
        harness._observation = observation
        harness._refresh_public_discovery(
            {'action_type': action.action_type, 'arguments': action.arguments}, True)
        return replace(result, observation=observation)

    harness.execute_action = step


@pytest.mark.parametrize('query,incomplete,outcome', [
    ('egg', False, 'completed'),
    ('absent', False, 'scope_no_match'),
    ('absent', True, 'scope_incomplete'),
])
def test_public_search_outcomes(tmp_path, query, incomplete, outcome):
    system, ctx, occ, _, provider = setup(tmp_path, lambda *a: pytest.fail('unexpected LLM'))
    try:
        enable_public_receipts(ctx.harness, incomplete=incomplete)
        _, tool = search(public_selector=True)
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(
            tool, {'query': query, 'locations': list(ctx.harness.case.locations), 'allow_open': True},
            ctx, occurrence_id=occ.occurrence_id)
        assert result.outcome == outcome, result
        assert result.completed == (outcome == 'completed')
        assert not result.intrinsic_failure
        assert bool(result.output_candidates) == (outcome == 'completed')
        assert not provider.requests
        assert list(ctx.search_history.attempts.values())[-1].execution_outcome == outcome
        if outcome in {'scope_no_match', 'scope_incomplete'}:
            from atomic_skillgraph.governance.credit import CreditAssigner
            events = CreditAssigner().assign(ctx.trace_builder.trace)
            assert not any(e.event.value in {'direct_failure', 'direct_success'}
                           for e in events if e.artifact_kind == 'tool')
            assert any(e.event.value == 'execution_started' for e in events if e.artifact_kind == 'tool')
        if outcome == 'scope_no_match':
            proof = result.tool_path_evidence['final_effect_result']['search_exit']
            assert all(c['source_refs'] and c['inspection_status'] == 'complete_listing'
                       for c in proof['checked_scope'])
    finally:
        system.close()


def test_matched_but_invalid_return_remains_program_error(tmp_path):
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('unexpected LLM'))
    try:
        enable_public_receipts(ctx.harness)
        _, tool = search(public_selector=True)
        tool.interface['output_schema']['required'].append('missing')
        tool.interface['output_schema']['properties']['missing'] = {'type': 'string'}
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(
            tool, {'query': 'egg', 'locations': list(ctx.harness.case.locations), 'allow_open': True},
            ctx, occurrence_id=occ.occurrence_id)
        assert result.outcome == 'program_error' and result.intrinsic_failure
        # Unvalidated candidates remain in the audit; they are not published.
        assert not result.completed
        assert not result.tool_path_evidence['final_effect_result'].get('search_exit')
    finally:
        system.close()


def test_negative_signature_ignores_only_clock_not_world_or_query_context(tmp_path):
    from atomic_skillgraph.runtime.negative_memory import state_signature
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('unexpected LLM'))
    try:
        stable = state_signature(ctx, occ, semantic_only=True)
        exact = state_signature(ctx, occ)
        ctx.world_revision += 1
        ctx.harness._validator.revision += 1
        assert state_signature(ctx, occ, semantic_only=True) == stable
        assert state_signature(ctx, occ) != exact
        ctx.observation = 'New public evidence'
        assert state_signature(ctx, occ, semantic_only=True) != stable
    finally:
        system.close()
