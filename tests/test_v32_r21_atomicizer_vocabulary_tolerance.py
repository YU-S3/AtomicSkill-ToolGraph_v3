"""Deterministic tolerance tests for the E1 transport vocabulary.

The E1 context exposes the normalized trace's action ids verbatim and the
transport may spell output derivations without the canonical ``kind``/``type``
key.  The Atomicizer accepts the model's natural echo of code-authoritative
identity while every authority and derivation still resolves against real
code-side evidence; fabricated references remain rejected (Gate31).
"""

from __future__ import annotations

import pytest

from test_v32_r1_gates import _atomicizer_proposal, _atomicizer_trace


def _canonicalize(proposal, trace):
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    return Atomicizer().validate_and_canonicalize([proposal], trace)


def test_action_id_ref_vocabulary_accepted() -> None:
    """The model's ``action:<action_id>`` echo resolves to the event authority."""

    trace = _atomicizer_trace()
    trace["actions"][0]["action_id"] = "r039_a010"
    trace["actions"][0].pop("event_id", None)
    proposal = _atomicizer_proposal(
        "p", start=0, end=1, support=["r039_a010", "e1"], effect_refs=["effect:w1"],
    )
    proposal.input_roles = {"item": "apple_1"}
    proposal.input_provenance_refs = {"item": "action:r039_a010"}
    proposal.output_derivations = {
        "result": {"kind": "input_identity", "input_role": "item"},
    }
    canonical = _canonicalize(proposal, trace)
    assert len(canonical) == 1
    assert canonical[0].input_provenance_refs["item"]["authority_ref"] == (
        "action:r039_a010"
    )


def test_fabricated_action_ref_still_rejected() -> None:
    """Gate31: a reference to an action that is not part of the evidence
    slice stays a hard rejection."""

    trace = _atomicizer_trace()
    proposal = _atomicizer_proposal(
        "p", start=0, end=1, support=["e0", "e1"], effect_refs=["effect:w1"],
    )
    proposal.input_roles = {"item": "apple_1"}
    proposal.input_provenance_refs = {"item": "action:r999_a999"}
    proposal.output_derivations = {
        "result": {"kind": "input_identity", "input_role": "item"},
    }
    with pytest.raises(ValueError, match="input authority ref not found"):
        _canonicalize(proposal, trace)


def test_output_derivation_kind_omitted_infers_input_identity() -> None:
    trace = _atomicizer_trace()
    proposal = _atomicizer_proposal(
        "p", start=0, end=1, support=["e0", "e1"], effect_refs=["effect:w1"],
    )
    proposal.output_derivations = {"result": {"input_role": "item"}}
    canonical = _canonicalize(proposal, trace)
    assert len(canonical) == 1
    assert canonical[0].output_derivations["result"]["kind"] == "input_identity"


def test_output_derivation_unrecognized_kind_falls_back_to_binding_fields() -> None:
    trace = _atomicizer_trace()
    proposal = _atomicizer_proposal(
        "p", start=0, end=1, support=["e0", "e1"], effect_refs=["effect:w1"],
    )
    proposal.output_derivations = {
        "result": {"kind": "malformed_vocabulary", "input_role": "item"},
    }
    canonical = _canonicalize(proposal, trace)
    assert len(canonical) == 1
    assert canonical[0].output_derivations["result"]["kind"] == "input_identity"


def test_output_derivation_predicate_fields_infer_effect_witness() -> None:
    trace = _atomicizer_trace()
    proposal = _atomicizer_proposal(
        "p", start=0, end=1, support=["e0", "e1"], effect_refs=["effect:w1"],
    )
    proposal.output_derivations = {
        "result": {"predicate": "object.observed", "argument_role": "object"},
    }
    canonical = _canonicalize(proposal, trace)
    assert len(canonical) == 1
    assert canonical[0].output_derivations["result"]["kind"] == "effect_witness"


def test_output_derivation_without_any_binding_still_rejected() -> None:
    trace = _atomicizer_trace()
    proposal = _atomicizer_proposal(
        "p", start=0, end=1, support=["e0", "e1"], effect_refs=["effect:w1"],
    )
    proposal.output_derivations = {"result": {"kind": "malformed_vocabulary"}}
    with pytest.raises(ValueError, match="unsupported Atomic output derivation"):
        _canonicalize(proposal, trace)
