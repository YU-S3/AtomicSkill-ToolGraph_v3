from __future__ import annotations

import hashlib
import json

import pytest

from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.agents.protocol import (
    SchemaValidationError,
    validate_schema_instance,
)
from atomic_skillgraph.agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
    BINDING_EXPRESSION_SCHEMA,
    TOOL_IR_COLLECTION_SOURCE_SCHEMA,
    TOOL_PROPOSAL_SCHEMA,
)
from atomic_skillgraph.tooling.proposal import ToolProvenance


_POLICY_SEPARATOR = "\n\nPOLICY_CONTEXT_JSON\n"
_R922_TOOL_BUILDER_INSTRUCTION_SHA256 = (
    "2b9d008b0563c4eea0c3bef0d70aed8d608fafe82fbace2d0b68d0c28529871b"
)


def _atomic_view() -> dict[str, object]:
    return {
        "summary": "navigate to the supplied destination",
        "inputs": [{
            "name": "destination",
            "semantic_type": "entity",
            "required": True,
            "runtime_resolvable": True,
            "required_resolution": "concrete",
            "description": "",
        }],
        "outputs": [{
            "name": "arrived_location",
            "semantic_type": "entity",
            "required": True,
            "runtime_resolvable": False,
            "required_resolution": "semantic",
            "description": "",
        }],
        "preconditions": [],
        "effects": [{
            "predicate": "agent.at_location",
            "args": {
                "location": {
                    "kind": "skill_input",
                    "source_role": "arrived_location",
                },
            },
            "cardinality": 1,
            "distinct_by": "",
            "effect_domain": "world",
        }],
        "validator_spec": {
            "output_derivations": {
                "arrived_location": {
                    "kind": "effect_witness",
                    "predicate": "agent.at_location",
                    "argument_role": "location",
                },
            },
        },
    }


def test_tool_builder_uses_r922_instruction_and_canonical_ref() -> None:
    provenance = ToolProvenance(
        source="success_evolution",
        atomic_ref="skill://example_navigation@1.0.0",
        source_trace_id="SECRET_TRACE_ID",
        occurrence_id="SECRET_OCCURRENCE_ID",
        task_id="SECRET_TASK_ID",
    )

    prompt = ContextBuilder().tool_builder(
        atomic=_atomic_view(),
        provenance=provenance,
        evidence_support=[],
        harness_interface={"primitive_actions": [], "predicates": []},
    )
    instruction, raw_payload = prompt.split(_POLICY_SEPARATOR, 1)
    payload = json.loads(raw_payload)

    assert "entry_contract" in instruction
    assert 'proposal_version="2"' in instruction
    assert 'Input presence and types are already checked' in instruction
    assert 'requires matching current grounding evidence' in instruction
    assert 'required_resolution="semantic" does not waive that evidence requirement' in instruction
    assert payload["atomic_ref"] == provenance.atomic_ref
    assert payload["canonical_atomic"]["effects"] == _atomic_view()["effects"]
    assert payload["source_kind"] == "success_evolution"
    assert (
        "an action_catalog loop local is an authorized primitive argument"
        in instruction
    )
    assert (
        "missing pre-trial atomic_evidence_support or semantic_delta is not by "
        "itself a reason for NO_TOOL"
        in instruction
    )
    assert "never substitute output_role or bare predicate/argument_role" in instruction
    assert "SECRET_TRACE_ID" not in prompt
    assert "SECRET_OCCURRENCE_ID" not in prompt
    assert "SECRET_TASK_ID" not in prompt


def test_tool_builder_schema_names_existing_action_catalog_selector_shape() -> None:
    properties = TOOL_IR_COLLECTION_SOURCE_SCHEMA["properties"]
    assert "action_catalog" in properties["source"]["enum"]
    assert set(properties["where"]["properties"]) >= {
        "action_type", "argument_role", "semantic_compatible_with",
    }
    assert properties["project"]["properties"]["kind"]["enum"] == [
        "field", "argument",
    ]
    description = TOOL_IR_COLLECTION_SOURCE_SCHEMA["description"]
    assert "ZERO matching entries aborts the ENTIRE Tool" in description
    assert "project.kind=argument" in description
    assert "candidates, not effect evidence" in description
    validate_schema_instance(
        {
            "source": "action_catalog",
            "where": {
                "action_type": "GO_TO",
                "argument_role": "destination",
                "semantic_compatible_with": {
                    "source": "tool_input",
                    "field": "target",
                    "semantic_type": "entity",
                },
            },
            "project": {"kind": "argument", "role": "destination"},
            "distinct": True,
        },
        TOOL_IR_COLLECTION_SOURCE_SCHEMA,
    )
    with pytest.raises(SchemaValidationError):
        validate_schema_instance(
            {"source": "invented_candidate_source", "field": "value"},
            TOOL_IR_COLLECTION_SOURCE_SCHEMA,
        )


def test_tool_builder_accepts_mapping_provenance_but_requires_atomic_ref() -> None:
    prompt = ContextBuilder().tool_builder(
        atomic=_atomic_view(),
        provenance={
            "source": "runtime_automation",
            "atomic_ref": "skill://mapping_fixture@1.0.0",
        },
    )
    payload = json.loads(prompt.split(_POLICY_SEPARATOR, 1)[1])
    assert payload["atomic_ref"] == "skill://mapping_fixture@1.0.0"
    assert payload["source_kind"] == "runtime_automation"

    for provenance in (
        {"source": "success_evolution"},
        {"source": "success_evolution", "atomic_ref": ""},
        {"source": "success_evolution", "atomic_ref": "   "},
        {"source": "success_evolution", "atomic_ref": None},
    ):
        with pytest.raises(
            ValueError,
            match="ToolBuilder provenance atomic_ref must be non-empty",
        ):
            ContextBuilder().tool_builder(
                atomic=_atomic_view(),
                provenance=provenance,
            )


def test_extractor_e1_contains_only_the_frozen_boundary_replacements() -> None:
    prompt = ContextBuilder().extractor_e1(canonical_trace={"actions": []})
    instruction = prompt.split(_POLICY_SEPARATOR, 1)[0]

    assert (
        "event_start is inclusive and event_end is exclusive in this submission. "
        "A single event at index i uses [i, i+1). Code performs the "
        "exclusive-to-inclusive conversion; do not subtract one yourself."
    ) in instruction
    assert (
        "The precondition boundary is exactly canonical_trace.actions[event_start]."
        "authoritative_before_state_facts."
    ) in instruction
    assert 'input_provenance_refs[formal_role]' in instruction
    assert '{authority_ref, source_role}' in instruction
    assert 'allowed only through that explicit mapping' in instruction
    assert (
        "precondition_witness_refs must name those exact entry-state certificates, "
        "with matching predicate, arguments, and effect_domain;"
    ) in instruction
    assert (
        "a fact may persist across revisions, but its certificate at a later "
        "revision is not interchangeable with the certificate at the entry boundary;"
    ) in instruction
    self_check = (
        "Before the one native submission, verify every proposed occurrence "
        "independently: [event_start,event_end) contains its support_event_ids;"
    )
    assert self_check in instruction
    assert instruction.index(self_check) < instruction.index(
        "Call the offered native submission tool exactly once."
    )

    # Existing non-boundary extraction semantics remain present.
    assert "support_event_ids, not envelope overlap" in instruction
    assert "shared_precondition_event_ids is not a general list" in instruction
    assert 'existing ToolBuilder' in instruction
    assert "incomplete Composite coverage must not force invented capabilities" in instruction
    assert "preconditions:\n- may be empty;" not in instruction


def test_r7_b1_extractor_explains_code_owned_semantic_aliases() -> None:
    prompt = ContextBuilder().extractor_e1(canonical_trace={"actions": []})
    instruction = prompt.split(_POLICY_SEPARATOR, 1)[0]

    assert 'public_binding/public_catalog' in instruction
    assert 'Do not invent authority references' in instruction
    assert 'entry time and occurrence/task scope must match' in instruction
    assert 'Formal-role renaming' in instruction


def test_r7_b2_extractor_requires_predicate_role_closure() -> None:
    prompt = ContextBuilder().extractor_e1(canonical_trace={"actions": []})
    instruction = prompt.split(_POLICY_SEPARATOR, 1)[0]

    assert (
        "every episode concrete identity referenced by a precondition"
    ) in instruction
    assert (
        "represented by one declared input role with a supplied input authority"
    ) in instruction
    assert (
        "every non-fresh episode concrete identity referenced by an Effect"
    ) in instruction


def test_r7_b3_extractor_forbids_reclassifying_input_as_fresh_output() -> None:
    prompt = ContextBuilder().extractor_e1(canonical_trace={"actions": []})
    instruction = prompt.split(_POLICY_SEPARATOR, 1)[0]

    assert (
        "do not reclassify that existing identity as a fresh output"
    ) in instruction
    assert (
        "use explicit formal-input/source-role mappings, with real authority refs."
    ) in instruction


def test_r102_schema_additions_preserve_formal_binding_boundaries() -> None:
    assert BINDING_EXPRESSION_SCHEMA["properties"]["kind"]["enum"] == [
        "skill_input",
        "constant",
        "data_flow",
        "tool_output",
        "adapter_transform",
    ]
    assert ATOMIC_EXTRACTION_SCHEMA["required"] == [
        "boundary_schema_version", "input_specs", "output_specs",
        "output_semantic_constraints", "local_value_authority_refs",
        "phase_id",
        "intent",
        "event_start",
        "event_end",
        "support_event_ids",
        "input_roles",
        "input_provenance_refs",
        "output_roles",
        "output_derivations",
        "preconditions",
        "precondition_witness_refs",
        "effects",
        "effect_witness_refs",
        "rationale",
        "guideline",
    ]
    assert TOOL_PROPOSAL_SCHEMA["required"] == [
        "proposal_version",
        "decision",
        "summary",
        "atomic_ref",
        "inputs",
        "outputs",
        "program",
        "max_actions",
        "final_effects",
        "evidence_outputs",
        "path_expectations",
        "rationale",
        "entry_contract",
    ]
    assert TOOL_PROPOSAL_SCHEMA["properties"]["max_actions"]["minimum"] == 1

    atomic_properties = ATOMIC_EXTRACTION_SCHEMA["properties"]
    assert "authoritative_before_state_facts" in (
        atomic_properties["event_start"]["description"]
    )
    assert "later revision is not interchangeable" in (
        atomic_properties["precondition_witness_refs"]["description"]
    )
    assert "role == submitted source_role" in atomic_properties["input_provenance_refs"]["description"]
    assert "NOT optional authority.source_role" in atomic_properties["input_provenance_refs"]["description"]
    assert "same value" in atomic_properties["input_provenance_refs"]["description"]

    tool_properties = TOOL_PROPOSAL_SCHEMA["properties"]
    program_properties = tool_properties["program"]["items"]["properties"]
    assert "Graph-only data_flow" in program_properties["argument_mapping"]["description"]
    assert "tool_output" in program_properties["expected_effects"]["description"]
    assert "copy canonical_atomic.effects exactly" in (
        tool_properties["final_effects"]["description"]
    )

    valid_no_tool = {
        "proposal_version": "2", "entry_contract": {"conditions": [], "grounding_constraints": []},
        "decision": "no_tool",
        "summary": "no safe reusable implementation",
        "atomic_ref": "skill://example_navigation@1.0.0",
        "inputs": _atomic_view()["inputs"],
        "outputs": _atomic_view()["outputs"],
        "program": [],
        "max_actions": 1,
        "final_effects": [],
        "evidence_outputs": [],
        "path_expectations": [],
        "rationale": "the supplied evidence is insufficient",
    }
    validate_schema_instance(valid_no_tool, TOOL_PROPOSAL_SCHEMA)

    invalid_no_tool = dict(valid_no_tool, max_actions=0)
    with pytest.raises(SchemaValidationError):
        validate_schema_instance(invalid_no_tool, TOOL_PROPOSAL_SCHEMA)
