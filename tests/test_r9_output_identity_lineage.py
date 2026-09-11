from __future__ import annotations

from copy import deepcopy

import pytest

from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    AtomicProposalBatchRejected,
    Atomicizer,
)
from atomic_skillgraph.evolution.contract_canonicalizer import (
    atomic_contract_signature,
)


def _trace(
    inputs: dict[str, object],
    predicate: str,
    effect_args: dict[str, object],
) -> dict[str, object]:
    witness_ref = f"semantic:r1:{predicate}"
    effect = {
        "predicate": predicate,
        "args": dict(effect_args),
        "effect_domain": "world",
        "witness_ref": witness_ref,
        "event_index": 0,
        "revision": 1,
        "source_kind": "semantic_snapshot_delta",
    }
    return {
        "trace_id": "trace_r9_identity_lineage",
        "source_task": {"task_id": "task_r9_identity_lineage"},
        "actions": [{
            "event_index": 0,
            "event_id": "e0",
            "action_id": "e0",
            "action_type": "TRANSITION",
            "arguments": dict(inputs),
            "accepted": True,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span_0",
            "authoritative_before_state_facts": [],
            "authoritative_positive_effects": [deepcopy(effect)],
        }],
        "runtime_spans": [{
            "span_id": "span_0",
            "kind": "full_dynamic",
            "occurrence_id": "occurrence_0",
            "action_start": 0,
            "action_end": 1,
            "parent_span_id": None,
            "learnable": True,
        }],
        "validations": [],
        "boundary_authorities": {
            "inputs": [{
                "authority_ref": f"action_arg:e0:{role}",
                "event_id": "e0",
                "argument_role": role,
                "kind": "action_argument",
                "source_kind": "action_argument",
                "role": role,
                "value": value,
            } for role, value in inputs.items()],
            "effects": [effect],
        },
    }


def _proposal(
    inputs: dict[str, object],
    outputs: dict[str, object],
    derivations: dict[str, object],
    predicate: str,
    effect_args: dict[str, object],
) -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id="transition",
        intent="perform_transition",
        event_start=0,
        event_end=0,
        input_roles=dict(inputs),
        output_roles=dict(outputs),
        preconditions=[],
        effects=[SemanticPredicate(predicate, dict(effect_args))],
        rationale="the accepted transition established the effect",
        support_event_ids=["e0"],
        precondition_witness_refs=[],
        effect_witness_refs=[f"semantic:r1:{predicate}"],
        input_provenance_refs={
            role: f"action_arg:e0:{role}" for role in inputs
        },
        output_derivations=deepcopy(derivations),
        input_provenance_contract="code_authority_v3_2",
    )


def _existing_identity_effect_witness(
    *, inputs: dict[str, object] | None = None,
) -> tuple[AtomicOccurrenceProposal, dict[str, object]]:
    resolved_inputs = inputs or {"object": "apple_1"}
    proposal = _proposal(
        resolved_inputs,
        {"held_object": "apple_1"},
        {"held_object": {
            "kind": "effect_witness",
            "predicate": "agent.holds",
            "argument_role": "object",
        }},
        "agent.holds",
        {"object": "apple_1"},
    )
    return proposal, _trace(
        resolved_inputs, "agent.holds", {"object": "apple_1"},
    )


def test_r9_b1_existing_entity_cannot_be_reclassified_as_effect_witness() -> None:
    proposal, trace = _existing_identity_effect_witness()

    with pytest.raises(AtomicProposalBatchRejected) as caught:
        Atomicizer().validate_proposed_subset([proposal], trace)

    rejection = caught.value.rejections[0]
    assert rejection["reason"] == (
        "extractor_output_existing_identity_reclassified"
    )
    assert rejection["detail"] == {
        "output_role": "held_object",
        "matching_input_roles": ["object"],
        "submitted_derivation_kind": "effect_witness",
    }
    assert rejection["reason"] in rejection["error"]


def test_r9_b2_existing_entity_with_explicit_input_identity_passes() -> None:
    inputs = {"object": "apple_1"}
    proposal = _proposal(
        inputs,
        {"held_object": "apple_1"},
        {"held_object": {
            "kind": "input_identity",
            "input_role": "object",
        }},
        "agent.holds",
        {"object": "apple_1"},
    )

    occurrence = Atomicizer().validate_and_canonicalize(
        [proposal], _trace(inputs, "agent.holds", {"object": "apple_1"}),
    )[0]

    assert occurrence.output_derivations == {
        "held_object": {
            "kind": "input_identity",
            "input_role": "object",
        },
    }


def test_r9_b3_genuinely_fresh_entity_may_use_effect_witness() -> None:
    inputs = {"source": "seed_1"}
    proposal = _proposal(
        inputs,
        {"created_object": "result_1"},
        {"created_object": {
            "kind": "effect_witness",
            "predicate": "entity.created",
            "argument_role": "object",
        }},
        "entity.created",
        {"object": "result_1"},
    )

    occurrence = Atomicizer().validate_and_canonicalize(
        [proposal],
        _trace(inputs, "entity.created", {"object": "result_1"}),
    )[0]

    assert occurrence.output_derivations["created_object"]["kind"] == (
        "effect_witness"
    )


def test_r9_b4_equal_identity_in_two_inputs_rejects_without_role_guess() -> None:
    inputs = {
        "object": "apple_1",
        "source_object": "apple_1",
    }
    proposal, trace = _existing_identity_effect_witness(inputs=inputs)

    with pytest.raises(AtomicProposalBatchRejected) as caught:
        Atomicizer().validate_proposed_subset([proposal], trace)

    assert caught.value.rejections[0]["detail"] == {
        "output_role": "held_object",
        "matching_input_roles": ["object", "source_object"],
        "submitted_derivation_kind": "effect_witness",
    }


def test_r9_b5_equal_identity_in_two_inputs_accepts_explicit_source_role() -> None:
    inputs = {
        "object": "apple_1",
        "source_object": "apple_1",
    }
    source = BindingExpression(
        BindingExprKind.SKILL_INPUT,
        source_role="source_object",
    )
    proposal = _proposal(
        inputs,
        {"held_object": "apple_1"},
        {"held_object": {
            "kind": "input_identity",
            "input_role": "source_object",
        }},
        "agent.holds",
        {"object": source},
    )

    occurrence = Atomicizer().validate_and_canonicalize(
        [proposal], _trace(inputs, "agent.holds", {"object": "apple_1"}),
    )[0]

    assert occurrence.output_derivations["held_object"] == {
        "kind": "input_identity",
        "input_role": "source_object",
    }
    effect_source = occurrence.effects[0].args["object"]
    assert isinstance(effect_source, BindingExpression)
    assert effect_source.source_role == "source_object"


def test_r9_b6_non_entity_scalar_equality_does_not_trigger_identity_gate() -> None:
    inputs = {"quantity": 1}
    proposal = _proposal(
        inputs,
        {"count": 1},
        {"count": {
            "kind": "effect_witness",
            "predicate": "counter.updated",
            "argument_role": "count",
        }},
        "counter.updated",
        {"count": 1},
    )

    occurrence = Atomicizer().validate_and_canonicalize(
        [proposal], _trace(inputs, "counter.updated", {"count": 1}),
    )[0]

    assert occurrence.output_derivations["count"]["kind"] == "effect_witness"


def test_r9_b7_e1_prompt_contains_entity_identity_lineage_self_check() -> None:
    instruction = ContextBuilder().extractor_e1(
        canonical_trace={"actions": []},
    ).split("\n\nPOLICY_CONTEXT_JSON\n", 1)[0]

    assert "For every entity output, perform this identity-lineage self-check:" in instruction
    assert "compare it against every declared input identity" in instruction
    assert "if it is an existing input identity, use input_identity" in instruction
    assert "effect_witness is only for an identity not already supplied" in instruction
    assert "never use effect_witness merely because a post-state predicate" in instruction


def _atomic_with_lineage(input_role: str) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef("atomic_identity_lineage", "1.0.0"),
        summary="hold an object",
        inputs=[
            ParameterSpec("object", "entity"),
            ParameterSpec("source_object", "entity"),
        ],
        outputs=[ParameterSpec("held_object", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="object",
            )},
        )],
        validator_spec={
            "validator_id": "harness_atomic_effect",
            "identity_strict": True,
            "output_derivations": {
                "held_object": {
                    "kind": "input_identity",
                    "input_role": input_role,
                },
            },
        },
        failure_modes=[],
        guideline={},
        metadata={},
    )


def test_r9_b8_canonical_signature_includes_explicit_identity_lineage() -> None:
    assert atomic_contract_signature(_atomic_with_lineage("object")) != (
        atomic_contract_signature(_atomic_with_lineage("source_object"))
    )
