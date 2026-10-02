"""Versioned exact-contract reuse transport; original E1 evidence gates remain.

Refs are taken from this request's offered contracts, not looked up from a
model-authored arbitrary path. Role mapping is the existing explicit
formal-role -> {authority_ref, source_role} mapping. No contract is inferred
from matching summaries or concrete values.
"""
import copy
from ..core.serialization import to_primitive

VERSION = 'skillcompiler.e1-reuse.v1'
REUSE_INSTRUCTION = '''The occurrences array proposes new contracts using the existing boundary schema.
For an exact offered existing contract, prefer reuse_existing with version skillcompiler.e1-reuse.v1.
Select its atomic_ref and submit only current source evidence. source.input_provenance_refs explicitly
maps each existing formal input to {authority_ref, source_role}; keep existing formal role names in
input_roles and output_roles. Do not regenerate or modify its contract. output_witness_refs contains
only current source witnesses for effect_witness outputs; never copy historical witness addresses.
Source availability, identity, resolution and effect validation still apply in full. Different
input responsibility, cardinality, distinctness or outputs require a new occurrence, not reuse.'''
CONTRACT_FIELDS = {'intent', 'input_specs', 'output_specs', 'preconditions', 'effects',
                   'output_derivations', 'output_semantic_constraints', 'guideline',
                   'boundary_schema_version'}


def source_reference_view(normalized):
    """Offer short, request-local IDs without changing immutable Trace coordinates.

    The full directory remains available for type/time/owner checks. The short
    choice is only transport sugar, never an additional source of authority.
    """
    view = copy.deepcopy(normalized)
    choices, expansion = [], {}
    directory = normalized.get('boundary_authorities', {})
    for group in ('inputs', 'locals', 'effects'):
        for item in directory.get(group, []):
            ref = item.get('authority_ref') if group != 'effects' else item.get('witness_ref')
            if not isinstance(ref, str) or not ref:
                continue
            alias = f'e1ref:{len(choices) + 1:04d}'
            expansion[alias] = ref
            choices.append({'reference_id': alias, 'kind': group, 'original_ref': ref,
                **{k: copy.deepcopy(item[k]) for k in (
                    'role', 'value', 'semantic_type', 'resolution', 'available_revision',
                    'event_index', 'predicate', 'args') if k in item}})
    # Reserved alias strings in original refs are ambiguous; retain the original
    # protocol instead of rewriting a source that could collide with a choice.
    if set(expansion) & set(expansion.values()):
        return view, {}
    view['source_reference_choices'] = {'version': VERSION, 'choices': choices}
    return view, expansion


def expand_source_references(payload, expansion):
    """Expand only reference-valued fields, never values, roles or coordinates."""
    scalar_fields = {'authority_ref'}
    array_fields = {'local_value_authority_refs', 'precondition_witness_refs',
                    'effect_witness_refs', 'witness_refs'}
    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            if key in scalar_fields and isinstance(item, str):
                out[key] = expansion.get(item, item)
            elif key in array_fields and isinstance(item, list):
                out[key] = [expansion.get(v, v) if isinstance(v, str) else v for v in item]
            elif key == 'output_witness_refs' and isinstance(item, dict):
                out[key] = {role: [expansion.get(v, v) if isinstance(v, str) else v for v in refs]
                            for role, refs in item.items()}
            else:
                out[key] = visit(item)
        return out
    return visit(copy.deepcopy(payload))


def reuse_contracts(known):
    result = {}
    for item in known:
        row = to_primitive(item)
        validator = row.get('validator_spec', {})
        supported = {'validator_id', 'identity_strict', 'output_identity',
                     'output_derivations', 'output_semantic_constraints'}
        if (validator.get('validator_id') != 'harness_atomic_effect'
            or validator.get('identity_strict') is not True
            or set(validator) - supported
            or set(validator.get('output_derivations', {})) != {o['name'] for o in row['outputs']}):
            continue
        result[row['atomic_ref']] = copy.deepcopy(row)
    return result


def reuse_schema(known, occurrence_schema):
    offered = reuse_contracts(known)
    source_properties = {k: copy.deepcopy(v) for k, v in occurrence_schema['properties'].items()
                         if k not in CONTRACT_FIELDS}
    source_required = [k for k in occurrence_schema['required'] if k not in CONTRACT_FIELDS]
    return {'type': 'array', **({} if offered else {'maxItems': 0}), 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['version', 'atomic_ref', 'source', 'output_witness_refs'],
        'properties': {
            'version': {'type': 'string', 'enum': [VERSION]},
            'atomic_ref': {'type': 'string', 'enum': sorted(offered) or ['no_reusable_contract']},
            'source': {'type': 'object', 'required': source_required,
                       'properties': source_properties, 'additionalProperties': False},
            'output_witness_refs': {'type': 'object', 'additionalProperties': {
                'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'uniqueItems': True}},
        }}}


def expand_reuse(item, known):
    if item.get('version') != VERSION:
        raise ValueError('unsupported E1 reuse version')
    offered = reuse_contracts(known)
    if item['atomic_ref'] not in offered:
        raise ValueError('reuse contract not offered in this request')
    contract = offered[item['atomic_ref']]
    source = copy.deepcopy(item['source'])
    if CONTRACT_FIELDS & set(source):
        raise ValueError('reuse source cannot override the existing contract')
    inputs = {p['name'] for p in contract['inputs']}
    outputs = {p['name'] for p in contract['outputs']}
    if set(source['input_provenance_refs']) != inputs or set(source['input_roles']) != inputs:
        raise ValueError('reuse must map every exact existing input role')
    if set(source['output_roles']) != outputs or set(item['output_witness_refs']) - outputs:
        raise ValueError('reuse output roles differ from existing contract')
    validator = contract['validator_spec']
    derivations = copy.deepcopy(validator['output_derivations'])
    for role, derivation in derivations.items():
        # Historical proof addresses never authorize the current occurrence.
        derivation.pop('witness_refs', None)
        refs = item['output_witness_refs'].get(role, [])
        if refs:
            if derivation['kind'] != 'effect_witness':
                raise ValueError('input identity cannot claim fresh output witnesses')
            derivation['witness_refs'] = list(refs)
    source.update(boundary_schema_version='2', intent=contract['canonical_intent'],
        input_specs=contract['inputs'], output_specs=contract['outputs'],
        preconditions=contract['preconditions'], effects=contract['effects'],
        output_derivations=derivations,
        output_semantic_constraints=validator.get('output_semantic_constraints', {}),
        guideline=contract['guideline'])
    return source


def exact_reuse_contract(candidate, existing):
    from .contract_canonicalizer import canonical_atomic_contract
    left, right = canonical_atomic_contract(candidate), canonical_atomic_contract(existing)
    return 'identity_search' not in left and 'identity_search' not in right and left == right
