"""Harness-owned P1 vocabulary, shared native schema and Python conformance."""
import copy
from ..core.serialization import to_primitive
from ..agents.protocol import validate_schema_instance
from ..agents.structured_submission import BINDING_EXPRESSION_SCHEMA
from ..core.semantic_types import normalize_semantic_type


def _argument_schema(semantic_type):
    """Public literal type or a well-formed symbolic binding, not arbitrary JSON."""
    normalized = normalize_semantic_type(semantic_type)
    literal_type = {'boolean': 'boolean', 'number': 'number', 'array': 'array',
                    'object_map': 'object'}.get(normalized, 'string')
    alternatives = [{'type': literal_type}]
    if literal_type == 'string':
        alternatives[0]['minLength'] = 1
    for kind in BINDING_EXPRESSION_SCHEMA['properties']['kind']['enum']:
        binding = copy.deepcopy(BINDING_EXPRESSION_SCHEMA)
        binding['properties']['kind']['enum'] = [kind]
        if kind == 'constant':
            binding['required'].append('constant')
            binding['properties']['constant'] = alternatives[0]
        else:
            binding['required'].append('source_role')
            binding['properties']['source_role']['minLength'] = 1
            if kind in {'data_flow', 'tool_output'}:
                binding['required'].append('source_step')
                binding['properties']['source_step']['minLength'] = 1
            if kind == 'adapter_transform':
                binding['required'].append('transform_id')
                binding['properties']['transform_id']['minLength'] = 1
        alternatives.append(binding)
    return {'description': 'Harness semantic type: ' + semantic_type, 'anyOf': alternatives}


def requirement_schema(base, specs):
    schema = copy.deepcopy(base)
    predicates = []
    for spec in specs:
        predicates.append({
            'type': 'object', 'required': ['predicate', 'args', 'effect_domain'],
            'additionalProperties': False,
            'properties': {
                'predicate': {'type': 'string', 'enum': [spec.predicate]},
                'args': {'type': 'object', 'required': list(spec.argument_roles),
                         'additionalProperties': False,
                         'properties': {role: _argument_schema(spec.argument_semantic_types[role])
                                        for role in spec.argument_roles}},
                'effect_domain': {'type': 'string', 'enum': [spec.effect_domain]},
                'cardinality': {'type': 'integer', 'minimum': 1},
                'distinct_by': {'type': 'string'},
            },
        })
    if not predicates:
        raise ValueError('mechanism.v2 requires a nonempty Harness predicate vocabulary')
    item = schema['properties']['requirements']['items']
    for field in ('desired_effects', 'precondition_hints'):
        item['properties'][field]['items'] = {'anyOf': predicates}
    return schema


def check_bundle(bundle, schema):
    # Same schema as native submission: unknown predicates never reach retrieval.
    validate_schema_instance(to_primitive(bundle), schema)


def vocabulary_view(specs):
    return to_primitive(list(specs))
