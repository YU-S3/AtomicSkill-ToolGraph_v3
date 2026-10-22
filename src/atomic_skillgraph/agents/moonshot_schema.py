"""Moonshot wire schema syntax; callers still validate the original schema."""
from copy import deepcopy


def wire_schema(schema):
    value = deepcopy(schema)
    if not isinstance(value, dict): return value
    for key in ('properties', '$defs'):
        if key in value: value[key] = {k:wire_schema(v) for k,v in value[key].items()}
    for key in ('items', 'additionalProperties'):
        if isinstance(value.get(key), dict): value[key] = wire_schema(value[key])
    for key in ('oneOf', 'anyOf', 'allOf'):
        if key in value: value[key] = [wire_schema(v) for v in value[key]]
    if 'oneOf' in value:
        if 'anyOf' in value: raise ValueError('Combined oneOf/anyOf requires an explicit provider schema')
        value['anyOf'] = value.pop('oneOf')
    if isinstance(value.get('type'), list):
        kinds = value.pop('type')
        if 'anyOf' in value: raise ValueError('Combined union types require an explicit provider schema')
        value['anyOf'] = [{'type':kind} for kind in kinds]
    if 'enum' in value and 'type' not in value:
        kinds = {type(x) for x in value['enum']}
        types = {str:'string',int:'integer',float:'number'}
        if len(kinds)!=1 or not kinds <= types.keys(): raise ValueError('Unsupported Moonshot enum')
        value['type'] = types[next(iter(kinds))]
    return value
