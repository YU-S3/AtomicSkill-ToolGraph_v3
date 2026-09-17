"""Explicit entity-port declarations for controlled E1 protocol fixtures.

These helpers declare test inputs; they never infer production types, supply
validator outcomes, or change archived source traces.
"""
from atomic_skillgraph.core.contracts import ParameterSpec
from atomic_skillgraph.core.serialization import to_primitive


def entity_ports(inputs, outputs):
    return {
        'boundary_schema_version': '2',
        'input_specs': [to_primitive(ParameterSpec(role, 'entity', runtime_resolvable=True, required_resolution='concrete')) for role in inputs],
        'output_specs': [to_primitive(ParameterSpec(role, 'entity', required_resolution='concrete')) for role in outputs],
        'output_semantic_constraints': {},
        'local_value_authority_refs': [],
    }


def public_entities(normalized, values, *, revision=0, owner=''):
    """Declare the controlled fixture's explicitly known public input values."""
    authorities = [{
        'authority_ref': f'fixture_public:{role}', 'kind': 'public_catalog',
        'source_kind': 'entity_concrete', 'role': role, 'value': value,
        'semantic_type': 'entity', 'resolution': 'concrete',
        'available_revision': revision, 'trace_id': normalized['trace_id'],
        'source_occurrence_id': owner,
    } for role, value in values.items()]
    normalized['boundary_authorities']['inputs'].extend(authorities)
    return {a['role']: {'authority_ref': a['authority_ref'], 'source_role': a['role']} for a in authorities}


def declare_take_fixture(proposal, normalized):
    """One controlled pre-known entity, TAKE witness and identity return.

    Used only by the E2 unit fixtures; no production inference or validator
    result is replaced. Negative fixture inputs deliberately lack authority.
    """
    normalized.setdefault('trace_id', 'fixture_trace')
    normalized.setdefault('boundary_authorities', {}).setdefault('inputs', [])
    refs = {'item': {'authority_ref': 'fixture_public:item', 'source_role': 'item'}}
    if not any(item.get('authority_ref') == 'fixture_public:item' for item in normalized['boundary_authorities']['inputs']):
        public_entities(normalized, {'item': 'item_1'})
    proposal.input_provenance_contract = 'code_authority_v3_2'
    proposal.boundary_schema_version = '2'
    proposal.input_specs = [ParameterSpec(role, 'entity', runtime_resolvable=True, required_resolution='concrete') for role in proposal.input_roles]
    proposal.output_specs = [ParameterSpec(role, 'entity', required_resolution='concrete') for role in proposal.output_roles]
    proposal.input_provenance_refs = {role: refs.get(role, {'authority_ref': 'missing', 'source_role': role}) for role in proposal.input_roles}
    proposal.output_derivations = {'result': {'kind': 'input_identity', 'input_role': next(iter(proposal.input_roles))}}
    proposal.support_event_ids = [normalized['actions'][0].get('action_id', 'a0')]
    proposal.effect_witness_refs = ['fact:holds:item_1']
    for fact in normalized['actions'][0]['authoritative_positive_effects']:
        fact['effect_domain'] = 'world'
        fact['revision'] = normalized['actions'][0]['after_revision']
        fact['source_kind'] = 'semantic_snapshot_delta'
    normalized['boundary_authorities']['effects'] = normalized['actions'][0]['authoritative_positive_effects']
    return proposal
