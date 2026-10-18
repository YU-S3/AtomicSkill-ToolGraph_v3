"""Versioned short-policy contract. Generated text is never a qualification receipt."""
from copy import deepcopy

from ..agents.protocol import validate_schema_instance
from .contracts import digest

VERSION = 'officeqa.effect-evolution.v6'
MATERIAL = 'officeqa.effect-trace.v1.1'
SELECTION = 'officeqa.effect-selection.v1'
RUNTIME = 'officeqa.effect-runtime.v1'
FEATURES = ['all', 'time_series_or_comparison', 'calculation']
TEXT = {'type': 'string', 'minLength': 1}
SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['decision', 'reason', 'source_case_ids'], 'properties': {
        'decision': {'enum': ['propose', 'no_change']}, 'reason': TEXT,
        'source_case_ids': {'type': 'array', 'items': TEXT, 'uniqueItems': True},
        'goal': TEXT, 'applicability': {'enum': FEATURES},
        'procedure': {'type': 'array', 'items': TEXT, 'minItems': 2, 'maxItems': 4},
        'stop_condition': TEXT,
        'edit': {'type': 'object', 'additionalProperties': False,
                 'required': ['operation', 'index'], 'properties': {
                     'operation': {'enum': ['add', 'delete', 'replace']},
                     'index': {'type': 'integer', 'minimum': 0}}}}}


def schema_for(editing=False):
    schema=deepcopy(SCHEMA)
    if not editing:schema['properties'].pop('edit')
    schema['properties']['stop_condition']={**TEXT,'description':'Include the stop condition AND missing-evidence fallback in this single field. There is no separate fallback property.'}
    return schema


def guidance(candidate):
    return '\n'.join([candidate['goal'], *candidate['procedure'], candidate['stop_condition']])


def validate_candidate(value, sources, parent=None):
    validate_schema_instance(value, schema_for(parent is not None))
    if not set(value['source_case_ids']) <= set(sources):
        raise ValueError('source_case_ids must name only admitted Train induction/feedback cases')
    if value['decision'] == 'no_change': return None
    required = {'goal', 'applicability', 'procedure', 'stop_condition'}
    if not required <= value.keys() or not value['source_case_ids']:
        raise ValueError('A short policy requires complete content and real source cases')
    if len(guidance(value)) > 1000: raise ValueError('Total policy injection exceeds 1000 characters')
    if parent:
        if not value.get('edit'): raise ValueError('Feedback may change exactly one local principle')
        if any(value[k] != parent[k] for k in ('goal', 'applicability', 'stop_condition')):
            raise ValueError('A local edit must preserve goal, applicability and fallback')
        old, new, edit = parent['procedure'], value['procedure'], value['edit']
        index = edit['index']; op = edit['operation']
        legal = (op == 'replace' and index < len(old) and len(old) == len(new)
                 and old[:index] == new[:index] and old[index+1:] == new[index+1:] and old[index] != new[index])
        legal |= op == 'add' and index <= len(old) and len(new) == len(old)+1 and new[:index]+new[index+1:] == old
        legal |= op == 'delete' and index < len(old) and old[:index]+old[index+1:] == new
        if not legal: raise ValueError('Only the declared single add/delete/replace is allowed')
    elif 'edit' in value: raise ValueError('Initial induction has no parent to edit')
    candidate = {**deepcopy(value), 'method_version': VERSION, 'kind': 'guidance_only',
                 'state': 'source_checked', 'parent_hash': digest(parent) if parent else None}
    candidate['id'] = 'office_effect_' + digest(candidate)
    return candidate


def verify_candidate(candidate):
    raw = {k:v for k,v in candidate.items() if k != 'id'}
    if candidate.get('method_version') != VERSION or candidate.get('id') != 'office_effect_' + digest(raw):
        raise ValueError('Candidate content identity changed')
    if len(guidance(candidate)) > 1000: raise ValueError('Oversized policy')
