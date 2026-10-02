import copy
import pytest

from atomic_skillgraph.agents.protocol import validate_schema_instance

from atomic_skillgraph.agents.structured_submission import ATOMIC_EXTRACTION_SCHEMA
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.extraction_view import (
    VERSION, expand_reuse, reuse_schema, exact_reuse_contract,
)
from atomic_skillgraph.evolution.extractor_session import parse_occurrence_payload
from atomic_skillgraph.evolution.portability import KnownAtomicContractView
from atomic_skillgraph.system import AtomicSkillGraphSystem
from test_extractor_contract_authority import _occurrence


def example():
    occurrence = _occurrence('local')
    occurrence.guideline = {'steps': ['Establish the requested state.'], 'notes': []}
    atomic = AtomicSkillGraphSystem._canonical_atomic_for_occurrence(None, occurrence)
    view = KnownAtomicContractView(str(atomic.ref), occurrence.intent,
        to_primitive(atomic.inputs), to_primitive(atomic.outputs), to_primitive(atomic.preconditions),
        to_primitive(atomic.effects), to_primitive(atomic.validator_spec), atomic.guideline)
    item = {'version': VERSION, 'atomic_ref': str(atomic.ref), 'output_witness_refs': {}, 'source': {
        'phase_id': 'current', 'event_start': 0, 'event_end': 1, 'support_event_ids': ['e0'],
        'input_roles': {'item': 'item_1'}, 'output_roles': {'result': 'item_1'},
        'input_provenance_refs': {'item': {'authority_ref': 'input_0', 'source_role': 'source_item'}},
        'precondition_witness_refs': [], 'effect_witness_refs': ['current_witness'],
        'local_value_authority_refs': [], 'rationale': 'The current source establishes this contract.'}}
    return atomic, [view], item


def test_reuse_expands_unchanged_contract_and_explicit_source_mapping():
    atomic, known, item = example()
    original = copy.deepcopy(item)
    validate_schema_instance([item], reuse_schema(known, ATOMIC_EXTRACTION_SCHEMA))
    expanded = expand_reuse(item, known)
    validate_schema_instance(expanded, ATOMIC_EXTRACTION_SCHEMA)
    proposal = parse_occurrence_payload(expanded)
    assert proposal.input_specs == atomic.inputs
    assert proposal.output_specs == atomic.outputs
    assert proposal.input_provenance_refs['item']['source_role'] == 'source_item'
    assert proposal.effect_witness_refs == ['current_witness']
    assert item == original


def test_no_contract_override_unoffered_ref_or_version_fallback():
    _, known, item = example()
    for bad in ({**item, 'version': 'legacy'}, {**item, 'atomic_ref': 'skill://unknown@1.0.0'},
                {**item, 'source': {**item['source'], 'input_specs': []}}):
        with pytest.raises(ValueError):
            expand_reuse(bad, known)


def test_contract_equivalence_preserves_resolution_responsibility_and_cardinality():
    from dataclasses import replace
    atomic, _, _ = example()
    assert exact_reuse_contract(atomic, copy.deepcopy(atomic))
    changed = copy.deepcopy(atomic)
    changed.inputs[0] = replace(changed.inputs[0], runtime_resolvable=False)
    assert not exact_reuse_contract(atomic, changed)
    changed = copy.deepcopy(atomic)
    changed.effects[0] = replace(changed.effects[0], cardinality=2)
    assert not exact_reuse_contract(atomic, changed)


def test_short_source_choices_restore_only_real_references_not_values_or_coordinates():
    from atomic_skillgraph.evolution.extraction_view import source_reference_view, expand_source_references
    normalized = {'actions': [{'event_index': 7, 'event_id': 'real-event'}],
        'boundary_authorities': {'inputs': [{'authority_ref': 'real-input', 'role': 'source',
            'value': 'e1ref:0001', 'resolution': 'semantic', 'available_revision': 4}],
            'effects': [{'witness_ref': 'real-witness', 'event_index': 8}]}}
    original = copy.deepcopy(normalized)
    view, aliases = source_reference_view(normalized)
    payload = {'event_start': 7, 'event_end': 9, 'input_roles': {'target': 'e1ref:0001'},
        'input_provenance_refs': {'target': {'authority_ref': 'e1ref:0001', 'source_role': 'source'}},
        'effect_witness_refs': ['e1ref:0002'], 'output_witness_refs': {'found': ['e1ref:0002']}}
    restored = expand_source_references(payload, aliases)
    assert normalized == original and view['actions'] == original['actions']
    assert restored['input_roles'] == payload['input_roles']
    assert restored['input_provenance_refs']['target'] == {'authority_ref': 'real-input', 'source_role': 'source'}
    assert restored['effect_witness_refs'] == ['real-witness']
    assert restored['output_witness_refs']['found'] == ['real-witness']
    assert (restored['event_start'], restored['event_end']) == (7, 9)
