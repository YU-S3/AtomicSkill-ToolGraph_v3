"""Episode-local observations, never world evidence or execution authority."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = 'r103.search-history.v1'
FEEDBACK_VERSION = 'r103.runtime-feedback.v2'
HISTORY_HELP = ('Search history records checks that actually occurred, including rolled-back branches. '
    'It is not current absence evidence. Consider earlier checks when choosing a new scope; a repeated '
    'check may be justified by a changed query, inspection method or relevant state. Scope exhaustion '
    'is a no-match diagnostic, not an input-schema error. You retain control over the next scope and '
    'may use an offered primitive or report a supported conflict.')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ScopeCheck:
    scope_value: Any
    outcome: str
    observed_revision: int | None
    reached: bool
    selector_evaluated: bool
    check_node_ids: tuple[str, ...]
    action_indices: tuple[int, ...]
    public_projection_version: str = ''
    inspection_status: str = 'unknown'
    source_refs: tuple[str, ...] = ()


@dataclass
class SearchAttemptObservation:
    observation_id: str
    episode_id: str
    tool_attempt_id: str
    occurrence_id: str
    parent_occurrence_id: str | None
    authorizing_call_id: str | None
    tool_ref: str
    program_identity: str
    query_value: Any
    query_role: str
    method_identity: str
    method: dict
    arguments: dict
    authorized_scope: tuple
    checks: tuple[ScopeCheck, ...]
    action_indices: tuple[int, ...]
    started: bool
    execution_outcome: str
    failure_code: str
    world_disposition: str
    observed_revision_start: int
    observed_revision_end: int
    rollback_id: str | None = None
    current_truth_authorized: bool = False
    output_authorized: bool = False


@dataclass
class TaskSearchHistory:
    attempts: dict[str, SearchAttemptObservation] = field(default_factory=dict)
    cache_hits: list[dict] = field(default_factory=list)
    projections: list[dict] = field(default_factory=list)

    def sync(self, ctx):
        ctx.trace_builder.trace.metadata['runtime_search_history'] = {
            'version': VERSION, 'attempts': [asdict(a) for a in self.attempts.values()],
            'cache_hits': copy.deepcopy(self.cache_hits), 'projections': copy.deepcopy(self.projections)}

    def record(self, ctx, observation):
        prior = self.attempts.get(observation.observation_id)
        if prior is not None and prior != observation:
            raise ValueError('conflicting search observation identity')
        self.attempts.setdefault(observation.observation_id, observation)
        self.sync(ctx)

    def disposition(self, ctx, attempt_ids, value, rollback_id=None):
        for row in self.attempts.values():
            if row.tool_attempt_id in attempt_ids:
                row.world_disposition, row.rollback_id = value, rollback_id
        self.sync(ctx)

    def cache_hit(self, ctx, observation_ids, call_id, occurrence_id):
        ids = [i for i in observation_ids if i in self.attempts]
        self.cache_hits.append({'observation_ids': ids, 'authorizing_call_id': call_id,
            'occurrence_id': occurrence_id, 'diagnostic_available': bool(ids)})
        self.sync(ctx)

    def policy_view(self):
        groups = {}
        for row in self.attempts.values():
            key = identity([row.query_value, row.method_identity])
            group = groups.setdefault(key, {'query': copy.deepcopy(row.query_value),
                'method_id': row.method_identity, 'method': copy.deepcopy(row.method),
                'checked_no_match': [], 'matched': [], 'incomplete': [], 'revisit_counts': {},
                'last_attempt_ref': row.tool_attempt_id, 'rolled_back': False,
                'current_truth_authorized': False})
            group['last_attempt_ref'] = row.tool_attempt_id
            group['rolled_back'] |= row.world_disposition == 'rolled_back'
            for check in row.checks:
                bucket = ('checked_no_match' if check.outcome == 'no_matching_candidate' else
                          'matched' if check.outcome == 'matched_candidate' else 'incomplete')
                # Typed identity: True is not the integer 1.
                if not any(identity(v) == identity(check.scope_value) for v in group[bucket]):
                    group[bucket].append(copy.deepcopy(check.scope_value))
                if check.selector_evaluated and check.reached:
                    name = json.dumps(check.scope_value, ensure_ascii=False)
                    group['revisit_counts'][name] = group['revisit_counts'].get(name, -1) + 1
        for group in groups.values():
            group['revisit_counts'] = {k: v for k, v in group['revisit_counts'].items() if v > 0}
        return {'version': VERSION, 'semantics': 'historical_checks_not_current_absence',
                'groups': list(groups.values())}

    def note_projection(self, ctx, session_id):
        self.projections.append({'session_id': session_id,
            'observation_ids': list(self.attempts), 'history_hash': identity(self.policy_view())})
        self.sync(ctx)


def observe_search(tool, bindings, state, ctx, *, attempt_id, occurrence_id,
                   action_start, before_revision, result=None, error=None):
    """Recognize the existing bounded input-scope/public-selector IR, not names.

    Only executed selectors under an actual reached-location guard can certify
    historical checks. Open authorization remains part of the method identity.
    """
    from ..tooling.ir import walk_program_nodes
    program = tool.artifact.get('program', [])
    from .container_search_observer import checked_scopes
    frame = getattr(ctx.harness, 'public_container_inspection_frame', lambda: {})()
    container = checked_scopes(program,state,frame.get('containers',[]))
    if container is not None:
        loop,target,scope,rows = container
        selector = target['condition']['match']
        query_role = selector['where']['semantic_compatible_with']['field']
        if query_role not in bindings:
            return None
        method = {'kind':'container-bounded-search','selector':copy.deepcopy(selector),
            'query_role':query_role,'meaning':'historical_container_inspection_not_global_absence'}
        program_id = identity([str(tool.ref),tool.artifact])
        marker = getattr(ctx,'_compiler_invocation_marker',{}) or {}
        return SearchAttemptObservation(identity([VERSION,attempt_id]),ctx.trace_builder.trace.trace_id,
            attempt_id,occurrence_id,marker.get('parent_occurrence_id'),marker.get('native_call_id'),
            str(tool.ref),program_id,copy.deepcopy(bindings[query_role]),query_role,
            identity([program_id,method]),method,copy.deepcopy(bindings),tuple(scope),
            tuple(ScopeCheck(**row) for row in rows),
            tuple(range(action_start,len(ctx.trace_builder.trace.environment_actions))),
            bool(result and result.started or error),
            'interrupted' if error else 'completed' if result and result.completed else 'failed',
            getattr(result,'failure_code','') or getattr(error,'code','') or state.failure_code,
            'terminal' if ctx.execution_terminal() else 'applied',before_revision,ctx.world_revision)
    if len(program) != 1:
        return None
    loop = program[0]
    source = loop.get('collection_source', {})
    if loop.get('op') != 'FOR_EACH' or source.get('source') != 'tool_input':
        return None
    scope_role = source.get('field')
    scope = bindings.get(scope_role)
    if not isinstance(scope, list) or not scope:
        return None
    variable = loop['iteration_variable']
    nodes = walk_program_nodes(loop['body'])
    guards = {n['node_id'] for n in nodes if n['op'] == 'FOR_EACH'
        and n.get('collection_source', {}).get('source') == 'semantic_evidence'
        and n['collection_source'].get('where', {}).get('predicate') == 'agent.at_location'
        and n['collection_source'].get('where', {}).get('argument_role') == 'location'
        and n['collection_source'].get('where', {}).get('semantic_compatible_with', {}) == {
            'source': 'local_variable', 'field': variable, 'semantic_type': 'entity'}}
    targets = [n for n in nodes if n['op'] == 'IF' and n.get('condition', {}).get('op') == 'exists'
        and (n['condition'].get('match', {}).get('source') == 'action_catalog' or (
            n['condition'].get('match', {}).get('source') == 'semantic_evidence'
            and n['condition']['match'].get('where', {}).get('predicate') == 'entity.discovered_at'))
        and not n.get('else_branch')
        and any(child['op'] == 'RETURN' for child in walk_program_nodes(n.get('then_branch', [])))]
    if not guards or len(targets) != 1:
        return None
    target = targets[0]
    guards = {n['node_id'] for n in nodes if n['node_id'] in guards
              and any(child is target for child in walk_program_nodes(n.get('body', [])))}
    if not guards:
        return None
    selector = target['condition']['match']
    query = selector.get('where', {}).get('semantic_compatible_with', {})
    if query.get('source') != 'tool_input' or query.get('field') not in bindings:
        return None
    query_role = query['field']
    method = {'selector': copy.deepcopy(selector), 'query_role': query_role,
        'controls': {k: copy.deepcopy(v) for k, v in bindings.items() if k not in {query_role, scope_role}},
        'meaning': 'public_candidate_check_not_entity_absence'}
    public_discovery = selector.get('source') == 'semantic_evidence'
    if public_discovery:
        method['public_projection_version'] = getattr(ctx.harness, 'public_discovery_version', '')
    program_id = identity([str(tool.ref), tool.artifact])
    checks = []
    turns = iter(r for r in state.iteration_observations if r['node_id'] == loop['node_id'])
    for value in scope:
        # Iterations are ordered occurrences, not a value-keyed set. A repeated
        # scope value must not multiply the same physical check.
        turn = next(turns, None)
        if turn is not None and identity(turn['value']) != identity(value):
            raise ValueError('search iteration/input order mismatch')
        reached = [] if turn is None else [r for r in state.collection_observations[
            turn['collection_start']:turn['collection_end']] if r['node_id'] in guards and r['values']]
        evaluated = [] if turn is None else [r for r in state.condition_observations[
            turn['condition_start']:turn['condition_end']] if r['node_id'] == target['node_id']]
        outcome = ('matched_candidate' if evaluated[-1]['result'] else 'no_matching_candidate') if reached and evaluated else (
            'interrupted' if turn is not None else 'unchecked')
        frame = evaluated[-1].get('public_discovery', {}) if evaluated else {}
        inspection = [s for s in frame.get('inspected_scopes', []) if s['location'] == value]
        status = 'complete_listing' if inspection and all(s['status'] == 'complete_listing' for s in inspection) else (
            inspection[-1]['status'] if inspection else 'unknown')
        if public_discovery and outcome == 'no_matching_candidate' and status != 'complete_listing':
            outcome = 'incomplete_inspection'
        checks.append(ScopeCheck(copy.deepcopy(value), outcome,
            evaluated[-1]['revision'] if evaluated else None, bool(reached), bool(evaluated),
            tuple(r['node_id'] for r in evaluated),
            tuple(range(turn['action_start'], turn['action_end'])) if turn else (),
            frame.get('version', ''), status, tuple(s['source_ref'] for s in inspection)))
    marker = getattr(ctx, '_compiler_invocation_marker', {}) or {}
    failure = getattr(result, 'failure_code', '') or getattr(error, 'code', '') or state.failure_code
    return SearchAttemptObservation(identity([VERSION, attempt_id]), ctx.trace_builder.trace.trace_id,
        attempt_id, occurrence_id, marker.get('parent_occurrence_id'), marker.get('native_call_id'),
        str(tool.ref), program_id, copy.deepcopy(bindings[query_role]), query_role,
        identity([program_id, method]), method, copy.deepcopy(bindings), tuple(copy.deepcopy(scope)), tuple(checks),
        tuple(range(action_start, len(ctx.trace_builder.trace.environment_actions))), True,
        'interrupted' if error else ('completed' if result and result.completed else 'failed'), failure,
        'terminal' if ctx.execution_terminal() else ('unknown' if error and not getattr(error, 'code', '') else 'applied'),
        before_revision, ctx.world_revision)
