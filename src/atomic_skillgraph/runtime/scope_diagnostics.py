"""Conservative diagnostics and typed exits for bounded public searches.

This returns evidence for the runner's outcome; it never authorizes an output
or chooses a next scope.
It uses the executed IR, not an asset name or benchmark family.
"""
from ..tooling.ir import walk_program_nodes

SEARCH_OUTCOMES = frozenset({'scope_no_match', 'scope_incomplete'})
OUTCOME_VERSION = 'skillcompiler.search-outcome.v1'


def search_exit(program, state, signal, observation):
    """Classify only a recognized search that actually exhausted its input loop.

    Receipts come from the same production search observer used by history.
    A missing RETURN alone is never evidence of a successful search or absence.
    Runtime errors, matched selectors with invalid outputs and early interruption
    retain their original failure classification.
    """
    if signal or state.failure_code or state.outputs or observation is None:
        return None
    if len(program) != 1 or program[0].get('op') != 'FOR_EACH':
        return None
    loop = program[0]
    scope = list(observation.authorized_scope)
    outer = [r for r in state.collection_observations if r['node_id'] == loop['node_id']]
    if (not scope or len(scope) > loop['max_iterations'] or len(outer) != 1
        or not outer[0].get('completed') or outer[0]['values'] != scope):
        return None
    checks = observation.checks
    if len(checks) != len(scope) or any(c.outcome == 'matched_candidate' for c in checks):
        return None
    from .search_history import identity
    if any(identity(c.scope_value) != identity(v) for c, v in zip(checks, scope)):
        return None
    complete = all(c.reached and c.selector_evaluated
        and c.outcome == 'no_matching_candidate'
        and c.inspection_status in {'complete_listing', 'empty_listing'}
        and c.source_refs for c in checks)
    from dataclasses import asdict
    return {'version': OUTCOME_VERSION,
            'outcome': 'scope_no_match' if complete else 'scope_incomplete',
            'checked_scope': [asdict(c) for c in checks],
            'global_absence_claimed': False, 'outputs_authorized': False}


def scope_exhaustion(program, state, signal):
    if signal or state.failure_code or state.outputs:
        return None
    from .container_search_observer import checked_scopes
    container = checked_scopes(program,state)
    if container is not None:
        loop,_,scope,checks = container
        outer = [r for r in state.collection_observations if r['node_id'] == loop['node_id']]
        if (scope and len(outer) == 1 and outer[0].get('completed') and outer[0]['values'] == scope
            and all(c['outcome'] == 'no_matching_candidate' for c in checks)):
            return {'outcome':'scope_exhausted','checked_scope':checks,
                    'global_absence_claimed':False,'outputs_authorized':False}
        return None
    if len(program) != 1:
        return None
    loop = program[0]
    source = loop.get('collection_source', {})
    if loop.get('op') != 'FOR_EACH' or source.get('source') != 'tool_input':
        return None
    scope = state.bindings.get(source.get('field'))
    if not isinstance(scope, list) or not scope or len(scope) > loop['max_iterations']:
        return None
    outer = [r for r in state.collection_observations if r['node_id'] == loop['node_id']]
    if len(outer) != 1 or not outer[0].get('completed') or outer[0]['values'] != scope:
        return None
    variable = loop['iteration_variable']
    nodes = walk_program_nodes(loop['body'])
    guards = {n['node_id'] for n in nodes if n['op'] == 'FOR_EACH'
        and n.get('collection_source', {}).get('source') == 'semantic_evidence'
        and n['collection_source'].get('where', {}).get('semantic_compatible_with', {}) == {
            'source': 'local_variable', 'field': variable, 'semantic_type': 'entity'}}
    selectors = {n['node_id'] for n in nodes if n['op'] == 'IF'
        and n.get('condition', {}).get('op') == 'exists'
        and n['condition'].get('match', {}).get('source') in {'action_catalog', 'semantic_evidence'}
        and not n.get('else_branch')
        and any(child['op'] == 'RETURN' for child in walk_program_nodes(n.get('then_branch', [])))}
    if not guards or not selectors:
        return None
    checked = []
    for value in scope:
        reached = [r for r in state.collection_observations if r['node_id'] in guards
                   and r['locals'].get(variable) == value and r['values'] and r.get('completed')]
        searched = [r for r in state.condition_observations if r['node_id'] in selectors
                    and r['locals'].get(variable) == value]
        if not reached or not searched or any(r['result'] for r in searched):
            return None
        for row in searched:
            if row['condition'].get('match', {}).get('source') == 'semantic_evidence':
                scopes = [s for s in row.get('public_discovery', {}).get('inspected_scopes', []) if s['location'] == value]
                if not scopes or any(s['status'] != 'complete_listing' for s in scopes):
                    return None
        checked.append({'scope_value': value, 'guard_nodes': [r['node_id'] for r in reached],
                        'no_match_nodes': [r['node_id'] for r in searched],
                        'revision': searched[-1]['revision']})
    return {'outcome': 'scope_exhausted', 'checked_scope': checked,
            'global_absence_claimed': False, 'outputs_authorized': False}
