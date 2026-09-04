from __future__ import annotations

from copy import deepcopy

import pytest

from atomic_skillgraph.agents import (
    ReplayAgentSession,
    SchemaValidationError,
    UsageLedger,
    validate_schema_instance,
)
from atomic_skillgraph.agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
)
from atomic_skillgraph.core.contracts import EffectDomain, SemanticPredicate
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.extractor_session import ExtractorSession
from experiments.fakes import FakeReply, ScriptedAgentProvider


def _normalized() -> dict[str, object]:
    return {
        "trace_id": "trace_e1_output_authority",
        "source_task": {"task_id": "task"},
        "actions": [
            {
                "event_index": 0,
                "event_id": "e0",
                "action_id": "e0",
                "action_type": "GO_TO",
                "arguments": {"destination": "desk_1"},
                "accepted": True,
                "before_revision": 0,
                "after_revision": 1,
                "span_id": "span",
                "authoritative_before_state_facts": [],
                "authoritative_positive_effects": [{
                    "predicate": "agent.at_location",
                    "args": {"location": "desk_1"},
                    "effect_domain": "world",
                    "witness_ref": "action:e0:revision:1",
                    "event_index": 0,
                    "revision": 1,
                    "source_kind": "semantic_snapshot_delta",
                }],
            },
            {
                "event_index": 1,
                "event_id": "e1",
                "action_id": "e1",
                "action_type": "TAKE",
                "arguments": {"item": "apple_1"},
                "accepted": True,
                "before_revision": 1,
                "after_revision": 2,
                "span_id": "span",
                "authoritative_before_state_facts": [{
                    "predicate": "agent.at_location",
                    "args": {"location": "desk_1"},
                    "effect_domain": "world",
                    "witness_ref": "action:e0:revision:1",
                    "revision": 1,
                }],
                "authoritative_positive_effects": [{
                    "predicate": "agent.holds",
                    "args": {"object": "apple_1"},
                    "effect_domain": "world",
                    "witness_ref": "action:e1:revision:2",
                    "event_index": 1,
                    "revision": 2,
                    "source_kind": "semantic_snapshot_delta",
                }],
            },
        ],
        "runtime_spans": [{
            "span_id": "span",
            "kind": "full_dynamic",
            "occurrence_id": "occ",
            "action_start": 0,
            "action_end": 2,
            "parent_span_id": None,
            "learnable": True,
        }],
        "validations": [],
        "boundary_authorities": {
            "inputs": [
                {
                    "authority_ref": "action_arg:e1:item",
                    "event_id": "e1",
                    "argument_role": "item",
                    "kind": "action_argument",
                    "source_kind": "action_argument",
                    "role": "item",
                    "value": "apple_1",
                },
                {
                    "authority_ref": "action_arg:e0:destination",
                    "event_id": "e0",
                    "argument_role": "destination",
                    "kind": "action_argument",
                    "source_kind": "action_argument",
                    "role": "destination",
                    "value": "desk_1",
                },
            ],
            "effects": [{
                "predicate": "agent.holds",
                "args": {"object": "apple_1"},
                "effect_domain": "world",
                "witness_ref": "action:e1:revision:2",
                "event_index": 1,
                "revision": 2,
                "source_kind": "semantic_snapshot_delta",
            }],
        },
    }


def _proposal() -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id="take",
        intent="take_item",
        event_start=1,
        event_end=1,
        input_roles={"item": "apple_1", "destination": "desk_1"},
        output_roles={"held_item": "apple_1"},
        preconditions=[SemanticPredicate(
            "agent.at_location", {"location": "desk_1"},
        )],
        effects=[SemanticPredicate(
            "agent.holds", {"object": "apple_1"},
        )],
        rationale="accepted transition",
        support_event_ids=["e1"],
        precondition_witness_refs=["action:e0:revision:1"],
        effect_witness_refs=["action:e1:revision:2"],
        input_provenance_refs={
            "item": "action_arg:e1:item",
            "destination": "action_arg:e0:destination",
        },
        output_derivations={
            "held_item": {
                "kind": "input_identity",
                "input_role": "item",
            },
        },
        input_provenance_contract="code_authority_v3_2",
    )


def _schema_occurrence() -> dict[str, object]:
    return {
        "phase_id": "take",
        "intent": "take_item",
        "event_start": 1,
        "event_end": 2,
        "support_event_ids": ["e1"],
        "input_roles": {"item": "apple_1"},
        "input_provenance_refs": {"item": "action_arg:e1:item"},
        "output_roles": {"held_item": "apple_1"},
        "output_derivations": {
            "held_item": {
                "kind": "input_identity",
                "input_role": "item",
            },
        },
        "preconditions": [],
        "precondition_witness_refs": [],
        "effects": [{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
            "effect_domain": "world",
        }],
        "effect_witness_refs": ["action:e1:revision:2"],
        "rationale": "accepted transition",
    }


def test_current_e1_schema_requires_all_evidence_authorities() -> None:
    occurrence = _schema_occurrence()
    validate_schema_instance(occurrence, ATOMIC_EXTRACTION_SCHEMA)

    for field in (
        "support_event_ids",
        "precondition_witness_refs",
        "effect_witness_refs",
        "output_derivations",
    ):
        invalid = deepcopy(occurrence)
        invalid.pop(field)
        with pytest.raises(SchemaValidationError, match="missing required"):
            validate_schema_instance(invalid, ATOMIC_EXTRACTION_SCHEMA)


def test_current_e1_schema_rejects_ambiguous_derivation_shape() -> None:
    occurrence = _schema_occurrence()
    occurrence["output_derivations"] = {
        "held_item": {
            "kind": "input_identity",
            "input_role": "item",
            "predicate": "agent.holds",
        }
    }
    with pytest.raises(SchemaValidationError, match="oneOf"):
        validate_schema_instance(occurrence, ATOMIC_EXTRACTION_SCHEMA)


def test_extractor_preserves_explicit_input_identity_derivation() -> None:
    provider = ScriptedAgentProvider([
        FakeReply.structured({"occurrences": [_schema_occurrence()]}),
    ])
    extractor = ExtractorSession(ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e1",
    ))

    proposal = extractor.propose_atomics(_normalized())[0]

    assert proposal.output_derivations == {
        "held_item": {
            "kind": "input_identity",
            "input_role": "item",
        }
    }
    assert proposal.input_provenance_contract == "code_authority_v3_2"


def test_current_e1_requires_exact_output_derivation_key_set() -> None:
    missing = _proposal()
    missing.output_derivations = {}
    with pytest.raises(ValueError, match="exactly match output roles"):
        Atomicizer().validate_and_canonicalize([missing], _normalized())

    extra = _proposal()
    extra.output_derivations["invented"] = {
        "kind": "input_identity", "input_role": "item",
    }
    with pytest.raises(ValueError, match="exactly match output roles"):
        Atomicizer().validate_and_canonicalize([extra], _normalized())


def test_current_e1_requires_explicit_support_and_witness_refs() -> None:
    no_support = _proposal()
    no_support.support_event_ids = []
    with pytest.raises(ValueError, match="support_event_ids must be explicit"):
        Atomicizer().validate_and_canonicalize([no_support], _normalized())

    numeric_alias = _proposal()
    numeric_alias.support_event_ids = ["1"]
    with pytest.raises(ValueError, match="outside evidence envelope"):
        Atomicizer().validate_and_canonicalize([numeric_alias], _normalized())

    no_precondition_ref = _proposal()
    no_precondition_ref.precondition_witness_refs = []
    with pytest.raises(ValueError, match="precondition witnesses must be explicit"):
        Atomicizer().validate_and_canonicalize(
            [no_precondition_ref], _normalized(),
        )

    no_effect_ref = _proposal()
    no_effect_ref.effect_witness_refs = []
    with pytest.raises(ValueError, match="effect witnesses must be explicit"):
        Atomicizer().validate_and_canonicalize([no_effect_ref], _normalized())


def test_current_e1_effect_domain_is_part_of_witness_authority() -> None:
    accepted = Atomicizer().validate_and_canonicalize(
        [_proposal()], _normalized(),
    )
    assert len(accepted) == 1

    wrong_domain = _proposal()
    wrong_domain.effects = [SemanticPredicate(
        "agent.holds",
        {"object": "apple_1"},
        effect_domain=EffectDomain.EVIDENCE,
    )]
    with pytest.raises(ValueError, match="effect lacks"):
        Atomicizer().validate_and_canonicalize(
            [wrong_domain], _normalized(),
        )


def test_legacy_internal_proposal_keeps_isolated_migration_behavior() -> None:
    legacy = _proposal()
    legacy.input_provenance_contract = "legacy_action_argument_v1"
    legacy.support_event_ids = []
    legacy.precondition_witness_refs = []
    legacy.effect_witness_refs = []
    legacy.output_derivations = {}
    legacy.input_provenance_refs = {}

    accepted = Atomicizer().validate_and_canonicalize(
        [legacy], _normalized(),
    )

    assert accepted[0].output_derivations == {
        "held_item": {
            "kind": "input_identity",
            "input_role": "item",
        }
    }
