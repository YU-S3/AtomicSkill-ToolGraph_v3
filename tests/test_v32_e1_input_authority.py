from __future__ import annotations

from types import SimpleNamespace

import pytest

import atomic_skillgraph.system as system_module
from atomic_skillgraph.agents import (
    ReplayAgentSession,
    SchemaValidationError,
    UsageLedger,
    validate_schema_instance,
)
from atomic_skillgraph.agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
)
from atomic_skillgraph.core.contracts import SemanticPredicate, TaskContract
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.extractor_session import ExtractorSession
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from experiments.fakes import FakeReply, ScriptedAgentProvider


def _normalized(authority: dict[str, object] | None) -> dict[str, object]:
    inputs = [] if authority is None else [authority]
    return {
        "trace_id": "trace_input_authority",
        "source_task": {"task_id": "task"},
        "actions": [{
            "event_index": 0,
            "event_id": "e0",
            "action_id": "e0",
            "action_type": "TAKE",
            "arguments": {"item": "apple_1"},
            "accepted": True,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span",
            "authoritative_before_state_facts": [],
            "authoritative_positive_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "apple_1"},
                "effect_domain": "world",
                "witness_ref": "action:e0:revision:1",
                "event_index": 0,
                "revision": 1,
                "source_kind": "semantic_snapshot_delta",
            }],
        }],
        "runtime_spans": [{
            "span_id": "span",
            "kind": "full_dynamic",
            "occurrence_id": "occ",
            "action_start": 0,
            "action_end": 1,
            "parent_span_id": None,
            "learnable": True,
        }],
        "validations": [],
        "boundary_authorities": {
            "inputs": inputs,
            "effects": [{
                "predicate": "agent.holds",
                "args": {"object": "apple_1"},
                "effect_domain": "world",
                "witness_ref": "action:e0:revision:1",
                "event_index": 0,
                "revision": 1,
                "source_kind": "semantic_snapshot_delta",
            }],
        },
    }


def _current_proposal(refs: dict[str, str]) -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id="take",
        intent="take_item",
        event_start=0,
        event_end=0,
        input_roles={"item": "apple_1"},
        output_roles={"result": "apple_1"},
        preconditions=[],
        effects=[SemanticPredicate("agent.holds", {"object": "apple_1"})],
        rationale="accepted transition",
        support_event_ids=["e0"],
        precondition_witness_refs=[],
        effect_witness_refs=["action:e0:revision:1"],
        input_provenance_refs=refs,
        output_derivations={
            "result": {"kind": "input_identity", "input_role": "item"},
        },
        input_provenance_contract="code_authority_v3_2",
    )


def test_current_e1_accepts_exact_action_argument_authority() -> None:
    authority = {
        "authority_ref": "action_arg:e0:item",
        "event_id": "e0",
        "argument_role": "item",
        "kind": "action_argument",
        "source_kind": "action_argument",
        "role": "item",
        "value": "apple_1",
    }

    occurrence = Atomicizer().validate_and_canonicalize(
        [_current_proposal({"item": "action_arg:e0:item"})],
        _normalized(authority),
    )[0]

    assert occurrence.input_provenance_refs == {"item": authority}


@pytest.mark.parametrize(
    ("refs", "authority", "message"),
    [
        ({}, None, "exactly match input roles"),
        (
            {"item": "action_arg:unknown:item"},
            {
                "authority_ref": "action_arg:e0:item",
                "role": "item",
                "value": "apple_1",
            },
            "input authority ref not found",
        ),
        (
            {"item": "action_arg:e0:item"},
            {
                "authority_ref": "action_arg:e0:item",
                "role": "item",
                "value": "pear_1",
            },
            "input authority value mismatch",
        ),
        (
            {"item": "action_arg:e0:item"},
            {
                "authority_ref": "action_arg:e0:item",
                "role": "object",
                "value": "apple_1",
            },
            "input authority role mismatch",
        ),
    ],
)
def test_current_e1_input_authority_fails_closed(
    refs: dict[str, str],
    authority: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Atomicizer().validate_and_canonicalize(
            [_current_proposal(refs)], _normalized(authority),
        )


def test_current_e1_cannot_reference_future_action_argument() -> None:
    normalized = _normalized({
        "authority_ref": "action_arg:e1:item",
        "event_id": "e1",
        "argument_role": "item",
        "kind": "action_argument",
        "source_kind": "action_argument",
        "role": "item",
        "value": "apple_1",
    })
    normalized["actions"].append({
        "event_index": 1,
        "event_id": "e1",
        "action_id": "e1",
        "action_type": "EXAMINE",
        "arguments": {"item": "apple_1"},
        "accepted": True,
        "before_revision": 1,
        "after_revision": 2,
        "span_id": "span",
    })
    normalized["runtime_spans"][0]["action_end"] = 2

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_current_proposal({"item": "action_arg:e1:item"})],
            normalized,
        )


def _runtime_input_authority(
    *,
    source_occurrence_id: str = "occ",
    trial_event_start: int = 0,
    trial_event_end: int = 0,
) -> dict[str, object]:
    return {
        "authority_ref": "runtime_input:draft_1:item",
        "draft_id": "draft_1",
        "trial_event_start": trial_event_start,
        "trial_event_end": trial_event_end,
        "kind": "current_occurrence_anchor",
        "source_kind": "current_occurrence_anchor",
        "source_occurrence_id": source_occurrence_id,
        "source_role": "object",
        "role": "item",
        "value": "apple_1",
    }


def test_current_e1_accepts_runtime_input_from_selected_trial_lineage() -> None:
    authority = _runtime_input_authority()

    occurrence = Atomicizer().validate_and_canonicalize(
        [_current_proposal({"item": "runtime_input:draft_1:item"})],
        _normalized(authority),
    )[0]

    assert occurrence.input_provenance_refs == {"item": authority}


def test_current_e1_rejects_runtime_input_from_unrelated_occurrence() -> None:
    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_current_proposal({
                "item": "runtime_input:draft_1:item",
            })],
            _normalized(_runtime_input_authority(
                source_occurrence_id="unrelated_occurrence",
            )),
        )


def test_current_e1_rejects_runtime_input_from_future_trial() -> None:
    normalized = _normalized(_runtime_input_authority(
        trial_event_start=1,
        trial_event_end=1,
    ))
    normalized["actions"].append({
        "event_index": 1,
        "event_id": "e1",
        "action_id": "e1",
        "action_type": "EXAMINE",
        "arguments": {"item": "apple_1"},
        "accepted": True,
        "before_revision": 1,
        "after_revision": 2,
        "span_id": "span",
    })
    normalized["runtime_spans"][0]["action_end"] = 2

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_current_proposal({
                "item": "runtime_input:draft_1:item",
            })],
            normalized,
        )


def test_e1_schema_and_transport_require_and_preserve_input_refs() -> None:
    occurrence = {
        "phase_id": "take",
        "intent": "take_item",
        "event_start": 0,
        "event_end": 1,
        "support_event_ids": ["e0"],
        "input_roles": {"item": "apple_1"},
        "input_provenance_refs": {"item": "action_arg:e0:item"},
        "output_roles": {"result": "apple_1"},
        "output_derivations": {
            "result": {"kind": "input_identity", "input_role": "item"},
        },
        "preconditions": [],
        "precondition_witness_refs": [],
        "effect_witness_refs": ["action:e0:revision:1"],
        "effects": [{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
        }],
        "rationale": "accepted transition",
    }
    validate_schema_instance(occurrence, ATOMIC_EXTRACTION_SCHEMA)
    without_refs = dict(occurrence)
    without_refs.pop("input_provenance_refs")
    with pytest.raises(SchemaValidationError, match="required"):
        validate_schema_instance(without_refs, ATOMIC_EXTRACTION_SCHEMA)

    extractor = ExtractorSession(ReplayAgentSession(
        ScriptedAgentProvider([
            FakeReply.structured({"occurrences": [occurrence]}),
        ]),
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e1",
    ))
    proposal = extractor.propose_atomics({
        "actions": [],
        "boundary_authorities": {"inputs": [], "effects": []},
    })[0]
    assert proposal.input_provenance_refs == {
        "item": "action_arg:e0:item"
    }
    assert proposal.input_provenance_contract == "code_authority_v3_2"


def test_prepare_evolution_projects_only_accepted_action_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _CaptureExtractor:
        def __init__(self, _session: object) -> None:
            pass

        def propose_atomics(self, normalized: dict[str, object], *_args, **_kwargs):
            captured.update(normalized)
            raise RuntimeError("captured")

    monkeypatch.setattr(system_module, "ExtractorSession", _CaptureExtractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    system = system_module.AtomicSkillGraphSystem.__new__(
        system_module.AtomicSkillGraphSystem
    )
    system.normalizer = SimpleNamespace(build=lambda _trace: {
        "trace_id": "trace",
        "source_task": {},
        "actions": [
            {
                "event_id": "e0",
                "action_id": "e0",
                "accepted": True,
                "arguments": {"object": "apple_1", "source": "table_1"},
                "authoritative_positive_effects": [],
            },
            {
                "event_id": "e1",
                "action_id": "e1",
                "accepted": False,
                "arguments": {"object": "pear_1"},
                "authoritative_positive_effects": [],
            },
        ],
        "runtime_spans": [],
        "validations": [],
    })
    system._extractor_session = lambda _task_id: object()
    system.skills = object()
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(),
        contract_matcher=lambda: ExactContractMatcher(),
        semantic_predicate_schema=lambda: [],
    )
    trace = SimpleNamespace(metadata={}, runtime_plan={})

    with pytest.raises(RuntimeError, match="captured"):
        system._prepare_evolution(trace, SimpleNamespace(task_id="task"))

    assert captured["boundary_authorities"]["inputs"] == [
        {
            "authority_ref": "action_arg:e0:object",
            "event_id": "e0",
            "argument_role": "object",
            "kind": "action_argument",
            "source_kind": "action_argument",
            "role": "object",
            "value": "apple_1",
        },
        {
            "authority_ref": "action_arg:e0:source",
            "event_id": "e0",
            "argument_role": "source",
            "kind": "action_argument",
            "source_kind": "action_argument",
            "role": "source",
            "value": "table_1",
        },
    ]
