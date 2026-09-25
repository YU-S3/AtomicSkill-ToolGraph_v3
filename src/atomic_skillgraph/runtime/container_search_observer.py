"""Structural container search diagnostics; never supplies effect authority."""
import copy
from ..tooling.ir import walk_program_nodes


def recognize(program):
    loops = [n for n in program if n.get('op') == 'FOR_EACH'
             and n.get('collection_source',{}).get('source') == 'tool_input']
    if len(loops) != 1:
        return None
    loop = loops[0]
    targets = []
    for node in walk_program_nodes(loop.get('body',[])):
        selector = node.get('condition',{}).get('match',{})
        where = selector.get('where',{})
        if (node['op'] == 'IF' and node.get('condition',{}).get('op') == 'exists'
            and selector.get('source') == 'semantic_evidence'
            and where.get('predicate') == 'entity.in_container'
            and where.get('container') == {'source':'local_variable','field':loop['iteration_variable']}
            and where.get('semantic_compatible_with',{}).get('source') == 'tool_input'
            and any(n['op'] == 'RETURN' for n in walk_program_nodes(node.get('then_branch',[])))):
            targets.append(node)
    return (loop,targets[0]) if len(targets) == 1 else None


def checked_scopes(program, state, frames=()):
    shape = recognize(program)
    if shape is None:
        return None
    loop,target = shape
    scope = state.bindings.get(loop['collection_source']['field'])
    if not isinstance(scope,list):
        return None
    turns = iter(r for r in state.iteration_observations if r['node_id'] == loop['node_id'])
    checks = []
    for value in scope:
        turn = next(turns,None)
        conditions = [] if turn is None else state.condition_observations[turn['condition_start']:turn['condition_end']]
        matched = [r for r in conditions if r['node_id'] == target['node_id']]
        inspections = [row for r in conditions for row in r.get('public_container_inspection',{}).get('containers',[])
                       if row['container'] == value and row['revision'] == r['revision']]
        # A failed LOOK_IN may stop before the target IF; retain its failed
        # inspection as historical only, never turn it into checked-no-match.
        if (not inspections and turn and isinstance(turn.get('revision_start'),int)
            and isinstance(turn.get('revision_end'),int)):
            inspections = [r for r in frames if r['container'] == value and
                           turn['revision_start'] < r['revision'] <= turn['revision_end']]
        inspection = inspections[-1] if inspections else {}
        status = inspection.get('status','unknown')
        outcome = ('matched_candidate' if matched[-1]['result'] else
            'no_matching_candidate' if status in {'complete_listing','empty_listing'} else
            'incomplete_inspection') if matched else ('incomplete_inspection' if turn else 'unchecked')
        checks.append({'scope_value':copy.deepcopy(value),'outcome':outcome,
            'observed_revision':inspection.get('revision'), 'reached':bool(inspection),
            'selector_evaluated':bool(matched),'check_node_ids':tuple(r['node_id'] for r in matched),
            'action_indices':tuple(range(turn['action_start'],turn['action_end'])) if turn else (),
            'public_projection_version':'scienceworld.container-discovery.v1',
            'inspection_status':status,'source_refs':(inspection['source_ref'],) if inspection else ()})
    return loop,target,scope,checks
