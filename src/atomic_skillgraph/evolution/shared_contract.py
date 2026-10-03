"""One generated contract; source witnesses remain independently scoped."""
import copy
from dataclasses import replace
from ..agents.structured_submission import ATOMIC_EXTRACTION_SCHEMA, PARAMETER_SPEC_SCHEMA
from ..core.contracts import ParameterSpec
from ..core.serialization import to_primitive, json_values_equal
from .extraction_view import CONTRACT_FIELDS

VERSION = 'skillcompiler.shared-contract.v2'
HELP = ('For generalization generate candidate_contract once and source_instances twice. '
        'Source evidence is independent: current uses current aliases; history uses original refs. '
        'Preserve the complete causal preparation envelope, not only the last action. '
        'A program_control is a new explicitly authorized invocation hypothesis, NEVER a historical '
        'Agent input authority. Use only public authenticated entry entities for ordered_entity_scope. '
        'Inputs to be discovered inside the program are outputs/locals, not concrete entry requirements. '
        'Do not rewrite stable single-step contracts; reuse their offered source mapping.')


def schema(groups):
    base = ATOMIC_EXTRACTION_SCHEMA
    def subset(fields):
        return {'type': 'object', 'required': [k for k in base['required'] if k in fields],
            'additionalProperties': False,
            'properties': {k: copy.deepcopy(v) for k, v in base['properties'].items() if k in fields}}
    contract = subset(CONTRACT_FIELDS)
    contract['properties']['program_controls'] = {'type': 'array', 'maxItems': 2, 'items': {
        'type': 'object', 'required': ['parameter', 'authorization'], 'additionalProperties': False,
        'properties': {'parameter': PARAMETER_SPEC_SCHEMA, 'authorization': {'type': 'object'}}}}
    instance = subset(set(base['properties']) - CONTRACT_FIELDS)
    instance['required'] += ['source_slot', 'source_id']
    instance['properties'].update(source_slot={'type': 'string', 'enum': ['current', 'history']},
        source_id={'type': 'string', 'minLength': 1}, caller_arguments={'type': 'object'})
    return {'type': 'array', 'maxItems': 1 if groups else 0, 'items': {
        'type': 'object', 'required': ['group_id', 'candidate_contract', 'source_instances', 'rationale'],
        'additionalProperties': False, 'properties': {
            'group_id': {'type': 'string', 'enum': [g['group_id'] for g in groups] or ['no_group_available']},
            'candidate_contract': contract,
            'source_instances': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': instance},
            'rationale': {'type': 'string', 'maxLength': 512}}}}


def instantiate(submission, source_ids):
    contract = copy.deepcopy(submission['candidate_contract'])
    contract.pop('program_controls', None)
    rows = submission['source_instances']
    if sorted(row['source_slot'] for row in rows) != ['current', 'history']:
        raise ValueError('shared contract requires exactly current/history')
    result = []
    for row in rows:
        if row['source_id'] != source_ids[row['source_slot']]:
            raise ValueError('shared source identity mismatch')
        evidence = {k: copy.deepcopy(v) for k, v in row.items()
                    if k not in {'source_slot', 'source_id', 'caller_arguments'}}
        if set(contract) & set(evidence):
            raise ValueError('source instance cannot override the shared contract')
        result.append({'source_slot': row['source_slot'], 'proposal': {**contract, **evidence}})
    return result


def add_program_controls(atomic, occurrence, declarations, arguments, normalized):
    """New call controls validated against PUBLIC entry identities, not history inputs."""
    if not declarations:
        if arguments:
            raise ValueError('undeclared caller arguments')
        return atomic, occurrence
    from ..runtime.input_authorization import validate_declarations
    params = [ParameterSpec(**value['parameter']) for value in declarations]
    auth = {value['parameter']['name']: copy.deepcopy(value['authorization']) for value in declarations}
    if len(auth) != len(params) or set(auth) & {p.name for p in atomic.inputs} or set(arguments) != set(auth):
        raise ValueError('program control roles/arguments must be new, declared and complete')
    candidate = replace(atomic, inputs=[*atomic.inputs, *params], validator_spec={**atomic.validator_spec,
        'control_input_protocol': VERSION, 'input_authorization': auth})
    validate_declarations(candidate)
    entry = next(a for a in normalized['actions'] if a['event_index'] == occurrence.event_start)
    public = [a for group in ('inputs', 'locals') for a in normalized['boundary_authorities'].get(group, ())
              if a.get('kind') in {'public_catalog', 'public_binding'}
              and type(a.get('available_revision')) is int and a['available_revision'] <= entry['before_revision']
              and a.get('resolution') in {'concrete', 'relation_verified'}]
    for role, declaration in auth.items():
        value = arguments[role]
        if declaration['kind'] == 'caller_boolean':
            if type(value) is not bool:
                raise ValueError('caller_boolean requires bool')
        elif (not isinstance(value, list) or not declaration['min_items'] <= len(value) <= declaration['max_items']
              or any(not isinstance(v, str) for v in value) or len(set(value)) != len(value)
              or any(not any(json_values_equal(v, a['value']) for a in public) for v in value)):
            raise ValueError('program scope lacks public pre-entry authorization')
    # This is a future invocation example, never staged as an ObservedOccurrence capsule.
    program_example = replace(occurrence, input_specs=candidate.inputs,
        input_bindings={**occurrence.input_bindings, **copy.deepcopy(arguments)})
    return candidate, program_example
