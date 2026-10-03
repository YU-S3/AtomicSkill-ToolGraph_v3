"""Read-only Task/Node policy view. Execution authority stays outside prompts.

Unknown observations/qualifiers and the complete action catalogue survive.
Only expanded audit/proof material is replaced by its logged locator. Contract
and identity constraints are never inferred, summarized by a model or deleted.
"""
import copy
from .runtime_policy_projection import canonical_bytes, digest

VERSION = 'skillcompiler.decision-frame.v2'
HELP = ('DecisionFrame is the current decision surface, not an execution grant. '
        'goal.task_qualifiers is verbatim and remains binding together with node, identity, '
        'cardinality and downstream constraints. Use the complete current_action_catalog; '
        'a discovered identity is not proof of a currently available action. '
        'Repeated structures use {frame_ref: id}; resolve them in frame_values. '
        'frame_rows contains fields and rows: each row maps its ordered values to fields. '
        'frame_map uses the same rows plus keys to reconstruct a keyed object. '
        'Expanded proof locators are audit references, not new facts. Choose one call '
        'or report the specific unresolved dependency, without restating the full plan.')
_AUDIT_ONLY = {'proof_alternatives', 'identity_proof', 'source_proof',
               'retrieved_candidate_audit', 'mapping_evidence'}


def project_decision_frame(payload, *, scope, native_tool_specs=()):
    source = copy.deepcopy(payload)
    removed = []

    def public(value, path=''):
        if isinstance(value, list):
            return [public(v, f'{path}[{i}]') for i, v in enumerate(value)]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        out = {}
        for key, item in value.items():
            if key in _AUDIT_ONLY:
                locator = 'decision-proof:' + digest(item)
                removed.append({'path': f'{path}.{key}', 'locator': locator, 'original': item})
                out[key + '_locator'] = locator
            else:
                out[key] = public(item, f'{path}.{key}')
        return out

    raw = public(source)
    state = raw.pop('current_state_snapshot', {})
    task_frame = raw.pop('task_runtime_frame', {})
    execution = raw.pop('execution_frame', {})
    atomic = state.pop('current_atomic', {})
    support = raw.pop('support_atomic_candidates', None)
    if support is None:
        support = task_frame.pop('capability_candidates', [])
    last = execution.pop('last_step', None)
    if last is None:
        last = task_frame.pop('last_step', {})
    frame = {
        'version': VERSION, 'scope': scope,
        'goal': {'task_qualifiers': raw.pop('task_goal', ''), 'node': atomic,
                 'semantic_constraints': raw.pop('task_semantic_context', {}),
                 'task_contract': task_frame.pop('task_contract', {}),
                 'downstream': state.pop('downstream_obligations', {})},
        'state': {**state, 'observation': raw.pop('current_observation', ''),
                  'recent_accepted_actions': raw.pop('recent_accepted_actions', [])},
        'needs': {'blocked_support': raw.pop('blocked_support_candidates', []),
                  'rejected_candidates': raw.pop('rejected_candidates', [])},
        'calls': {'implementations': raw.pop('allowed_implementation_invocations', []),
                  'support': support},
        'feedback': {'last_call': last,
                     'last_rejected_invocation': raw.pop('recent_failed_learned_invocation', None)},
        'search': raw.pop('exploration_memory', {}),
        'execution': execution, 'task_runtime': task_frame,
    }
    # Lossless interning of repeated large structures, not heuristic filtering.
    seen, repeats = {}, {}
    def count(value):
        if isinstance(value, (dict, list)):
            key = canonical_bytes(value)
            if len(key) > 96:
                repeats[key] = repeats.get(key, 0) + 1
            for child in (value.values() if isinstance(value, dict) else value):
                count(child)
    count(frame)
    table = {}
    def pack(value):
        if isinstance(value, (dict, list)):
            key = canonical_bytes(value)
            if repeats.get(key, 0) > 1:
                if key not in seen:
                    ident = 'v' + str(len(seen) + 1)
                    seen[key] = ident
                    table[ident] = copy.deepcopy(value)
                return {'frame_ref': seen[key]}
            if isinstance(value, dict):
                return {k: pack(v) for k, v in value.items()}
            return [pack(v) for v in value]
        return value
    def rows(value):
        """Exact columnar serialization of repeated field names, not a filter."""
        if not isinstance(value, (dict, list)):
            return value
        rebuilt = ({k: rows(v) for k, v in value.items()} if isinstance(value, dict)
                   else [rows(v) for v in value])
        values = list(rebuilt.values()) if isinstance(rebuilt, dict) else rebuilt
        if len(values) < 2 or not all(isinstance(v, dict) for v in values):
            return rebuilt
        fields = list(values[0])
        if not fields or not all(set(v) == set(fields) for v in values):
            return rebuilt
        matrix = {'fields': fields, 'rows': [[v[k] for k in fields] for v in values]}
        if isinstance(rebuilt, dict):
            matrix['keys'] = list(rebuilt)
        compact = {'frame_map' if isinstance(rebuilt, dict) else 'frame_rows': matrix}
        return compact if len(canonical_bytes(compact)) < len(canonical_bytes(rebuilt)) else rebuilt
    packed = rows(pack(frame))
    out = {**raw, 'decision_frame': packed, 'frame_values': table}
    # Catalogue stays complete and revision-scoped; no hidden action filter.
    assert out.get('current_action_catalog') == source.get('current_action_catalog')
    audit = {'version': VERSION, 'scope': scope, 'source_payload': source,
             'source_hash': digest(source), 'projected_hash': digest(out),
             'removed_proof_expansions': removed, 'interned_structures': len(table),
             'task_qualifiers_hash': digest(frame['goal']['task_qualifiers']),
             'complete_action_catalog_preserved': True,
             'before_body_utf8_bytes': len(canonical_bytes(source)),
             'after_body_utf8_bytes': len(canonical_bytes(out)),
             'measurement': 'utf8_bytes_not_tokens'}
    return out, audit


def expand_frame(payload):
    """Exact structure expansion for conformance/receipt checks."""
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {'frame_ref'}:
                return visit(copy.deepcopy(payload['frame_values'][value['frame_ref']]))
            if set(value) in ({'frame_rows'}, {'frame_map'}):
                matrix = value.get('frame_rows', value.get('frame_map'))
                records = [{k: visit(v) for k, v in zip(matrix['fields'], row, strict=True)}
                           for row in matrix['rows']]
                return dict(zip(matrix['keys'], records, strict=True)) if 'frame_map' in value else records
            return {k: visit(v) for k, v in value.items()}
        return [visit(v) for v in value] if isinstance(value, list) else value
    return visit(payload['decision_frame'])
