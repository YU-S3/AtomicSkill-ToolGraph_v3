"""R7 gates for code-owned E1 semantic-alias input authority."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import SemanticPredicate
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.traces.schema import (
    EnvironmentActionRecord,
    RuntimeSpan,
    TaskRecord,
    TraceRecord,
)


def _fact(
    revision: int,
    predicate: str,
    args: dict[str, str],
    *,
    effect_domain: str = "world",
) -> dict[str, Any]:
    arguments = ",".join(
        f"{role}={value}" for role, value in sorted(args.items())
    )
    return {
        "predicate": predicate,
        "args": dict(args),
        "effect_domain": effect_domain,
        "witness_ref": f"semantic:r{revision}:{predicate}:{arguments}",
    }


def _normalized(
    transitions: list[
        tuple[str, dict[str, str], list[dict[str, Any]]]
    ],
    *,
    runtime_tool_trials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = TaskRecord(
        "r7-semantic-alias",
        "fake",
        "exercise generic semantic alias authority",
        "semantic_alias",
        "r7-semantic-alias-signature",
    )
    trace = TraceRecord.create(task, {}, {}, {"source": "full_dynamic"})
    trace.metadata["method_patch"] = "3.2"
    if runtime_tool_trials is not None:
        trace.metadata["runtime_tool_trials"] = copy.deepcopy(
            runtime_tool_trials
        )

    snapshots: list[dict[str, Any]] = [{
        "sequence_index": 0,
        "revision": 0,
        "origin": "reset",
        "action_id": "",
        "occurrence_id": "",
        "accepted": True,
        "done": False,
        "won": False,
        "facts": [],
    }]
    state: list[dict[str, Any]] = []
    actions: list[EnvironmentActionRecord] = []
    for index, (action_type, arguments, added_facts) in enumerate(transitions):
        event_id = f"e{index}"
        revision = index + 1
        state.extend(copy.deepcopy(added_facts))
        snapshots.append({
            "sequence_index": revision,
            "revision": revision,
            "origin": "environment_action",
            "action_id": event_id,
            "occurrence_id": "occ",
            "accepted": True,
            "done": False,
            "won": False,
            "facts": copy.deepcopy(state),
        })
        actions.append(EnvironmentActionRecord(
            event_id,
            index,
            action_type,
            dict(arguments),
            True,
            "accepted",
            False,
            False,
            revision,
            "span",
        ))

    trace.metadata["semantic_state_snapshots"] = snapshots
    trace.environment_actions = actions
    trace.runtime_spans = [RuntimeSpan(
        "span",
        "full_dynamic",
        "occ",
        0,
        len(actions),
        None,
        True,
    )]
    normalized = TraceNormalizer().build(trace)
    normalized["boundary_authorities"]["effects"] = [
        copy.deepcopy(fact)
        for action in normalized["actions"]
        if action["accepted"] is True
        for fact in action["authoritative_positive_effects"]
    ]
    return normalized


def _aliases(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(authority)
        for authority in normalized["boundary_authorities"]["inputs"]
        if authority.get("kind") == "semantic_alias"
    ]


def _single_event_proposal(
    *,
    role: str = "device",
    value: str = "lamp_1",
    authority_ref: str = "semantic_alias:e0:raw_entity:device",
    predicate: str = "device.enabled",
    predicate_role: str = "device",
    witness_ref: str = "semantic:r1:device.enabled:device=lamp_1",
) -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id="enable",
        intent="enable a device",
        event_start=0,
        event_end=0,
        input_roles={role: value},
        output_roles={"enabled_device": value},
        preconditions=[],
        effects=[SemanticPredicate(predicate, {predicate_role: value})],
        rationale="accepted transition",
        support_event_ids=["e0"],
        precondition_witness_refs=[],
        effect_witness_refs=[witness_ref],
        input_provenance_refs={role: authority_ref},
        output_derivations={
            "enabled_device": {
                "kind": "input_identity",
                "input_role": role,
            },
        },
        input_provenance_contract="code_authority_v3_2",
    )


def test_r7_a1_same_action_semantic_alias_is_generated() -> None:
    normalized = _normalized([(
        "ENABLE",
        {"raw_entity": "lamp_1"},
        [_fact(1, "device.enabled", {"device": "lamp_1"})],
    )])

    assert _aliases(normalized) == [{
        "authority_ref": "semantic_alias:e0:raw_entity:device",
        "event_id": "e0",
        "event_index": 0,
        "kind": "semantic_alias",
        "source_kind": "semantic_snapshot_alias",
        "role": "device",
        "value": "lamp_1",
        "source_authority_ref": "action_arg:e0:raw_entity",
        "source_argument_role": "raw_entity",
        "predicate": "device.enabled",
        "predicate_argument_role": "device",
        "witness_ref": "semantic:r1:device.enabled:device=lamp_1",
        "effect_domain": "world",
    }]


def test_r7_a2_same_primitive_and_semantic_role_is_not_duplicated() -> None:
    normalized = _normalized([(
        "TAKE",
        {"object": "item_1"},
        [_fact(1, "entity.held", {"object": "item_1"})],
    )])

    assert _aliases(normalized) == []
    assert [
        authority["authority_ref"]
        for authority in normalized["boundary_authorities"]["inputs"]
    ] == ["action_arg:e0:object"]


def test_r7_a3_alias_join_requires_exact_identity() -> None:
    normalized = _normalized([(
        "ENABLE",
        {"raw_entity": "lamp_1"},
        [_fact(1, "device.enabled", {"device": "lamp_2"})],
    )])

    assert _aliases(normalized) == []


def test_r7_a4_fresh_effect_output_never_becomes_input_alias() -> None:
    normalized = _normalized([(
        "CREATE",
        {"trigger": "button_1"},
        [_fact(1, "result.created", {"result": "new_entity_1"})],
    )])

    assert _aliases(normalized) == []


def test_r7_a5_ambiguous_primitive_source_fails_closed() -> None:
    normalized = _normalized([(
        "COMPARE",
        {"left": "entity_1", "right": "entity_1"},
        [_fact(1, "selection.made", {"target": "entity_1"})],
    )])

    assert _aliases(normalized) == []


def test_r7_runtime_trial_actions_do_not_export_semantic_aliases() -> None:
    normalized = _normalized(
        [(
            "ENABLE",
            {"raw_entity": "lamp_1"},
            [_fact(1, "device.enabled", {"device": "lamp_1"})],
        )],
        runtime_tool_trials={
            "draft_1": {
                "trial_event_start": 0,
                "trial_event_end": 0,
            },
        },
    )

    assert _aliases(normalized) == []
    assert any(
        authority.get("authority_ref") == "action_arg:e0:raw_entity"
        for authority in normalized["boundary_authorities"]["inputs"]
    )


def test_r7_a6_future_semantic_alias_is_not_visible_to_earlier_occurrence() -> None:
    normalized = _normalized([
        (
            "PREPARE",
            {"trigger": "button_1"},
            [_fact(1, "device.pending", {"device": "lamp_1"})],
        ),
        (
            "ENABLE",
            {"raw_entity": "lamp_1"},
            [_fact(2, "device.enabled", {"device": "lamp_1"})],
        ),
    ])
    proposal = _single_event_proposal(
        authority_ref="semantic_alias:e1:raw_entity:device",
        predicate="device.pending",
        witness_ref="semantic:r1:device.pending:device=lamp_1",
    )

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize([proposal], normalized)


def test_r7_a7_tampered_alias_source_authority_is_rejected() -> None:
    normalized = _normalized([(
        "ENABLE",
        {"raw_entity": "lamp_1"},
        [_fact(1, "device.enabled", {"device": "lamp_1"})],
    )])
    alias = _aliases(normalized)[0]
    alias["source_authority_ref"] = "action_arg:e0:missing"
    normalized["boundary_authorities"]["inputs"] = [
        authority
        for authority in normalized["boundary_authorities"]["inputs"]
        if authority.get("kind") != "semantic_alias"
    ] + [alias]

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_single_event_proposal()], normalized,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("kind", "unvalidated_alias"),
        ("source_kind", "unvalidated_snapshot"),
    ],
)
def test_r7_alias_discriminator_tamper_is_rejected(
    field: str,
    tampered: str,
) -> None:
    normalized = _normalized([(
        "ENABLE",
        {"raw_entity": "lamp_1"},
        [_fact(1, "device.enabled", {"device": "lamp_1"})],
    )])
    alias = _aliases(normalized)[0]
    alias[field] = tampered
    normalized["boundary_authorities"]["inputs"] = [
        authority
        for authority in normalized["boundary_authorities"]["inputs"]
        if authority.get("kind") != "semantic_alias"
    ] + [alias]

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_single_event_proposal()], normalized,
        )


def test_r7_alias_rejects_tampered_source_record_discriminator() -> None:
    normalized = _normalized([(
        "ENABLE",
        {"raw_entity": "lamp_1"},
        [_fact(1, "device.enabled", {"device": "lamp_1"})],
    )])
    for authority in normalized["boundary_authorities"]["inputs"]:
        if authority.get("authority_ref") == "action_arg:e0:raw_entity":
            authority["source_kind"] = "semantic_snapshot_alias"

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_single_event_proposal()], normalized,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("predicate", "device.disabled"),
        ("predicate_argument_role", "target"),
        ("witness_ref", "semantic:r1:tampered"),
        ("effect_domain", "evidence"),
    ],
)
def test_r7_a8_tampered_semantic_witness_is_rejected(
    field: str,
    tampered: str,
) -> None:
    normalized = _normalized([(
        "ENABLE",
        {"raw_entity": "lamp_1"},
        [_fact(1, "device.enabled", {"device": "lamp_1"})],
    )])
    alias = _aliases(normalized)[0]
    alias[field] = tampered
    normalized["boundary_authorities"]["inputs"] = [
        authority
        for authority in normalized["boundary_authorities"]["inputs"]
        if authority.get("kind") != "semantic_alias"
    ] + [alias]

    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize(
            [_single_event_proposal()], normalized,
        )


def test_r7_a9_primitive_role_collision_canonicalizes_via_alias() -> None:
    held_ref = "semantic:r1:agent.holds:object=item_1"
    processed_ref = (
        "semantic:r2:item.processed_with:device=device_1,target=item_1"
    )
    normalized = _normalized([
        (
            "TAKE",
            {"object": "item_1"},
            [_fact(1, "agent.holds", {"object": "item_1"})],
        ),
        (
            "OPERATE",
            {"object": "device_1"},
            [_fact(2, "item.processed_with", {
                "target": "item_1",
                "device": "device_1",
            })],
        ),
    ])
    proposal = AtomicOccurrenceProposal(
        phase_id="process",
        intent="process an item with a device",
        event_start=1,
        event_end=1,
        input_roles={"object": "item_1", "device": "device_1"},
        output_roles={"processed_object": "item_1"},
        preconditions=[SemanticPredicate(
            "agent.holds", {"object": "item_1"},
        )],
        effects=[SemanticPredicate(
            "item.processed_with",
            {"target": "item_1", "device": "device_1"},
        )],
        rationale="the accepted operation established the semantic effect",
        support_event_ids=["e1"],
        precondition_witness_refs=[held_ref],
        effect_witness_refs=[processed_ref],
        input_provenance_refs={
            "object": "action_arg:e0:object",
            "device": "semantic_alias:e1:object:device",
        },
        output_derivations={
            "processed_object": {
                "kind": "input_identity",
                "input_role": "object",
            },
        },
        input_provenance_contract="code_authority_v3_2",
    )

    canonical = Atomicizer().validate_and_canonicalize(
        [proposal], normalized,
    )[0]

    target = canonical.effects[0].args["target"]
    device = canonical.effects[0].args["device"]
    assert isinstance(target, BindingExpression)
    assert target.kind is BindingExprKind.SKILL_INPUT
    assert target.source_role == "object"
    assert isinstance(device, BindingExpression)
    assert device.kind is BindingExprKind.SKILL_INPUT
    assert device.source_role == "device"


def test_r7_a10_alias_does_not_bypass_fresh_output_temporal_closure() -> None:
    source_ref = "semantic:r1:source.available:source=seed_1"
    result_ref = "semantic:r1:result.created:result=new_entity_1"
    normalized = _normalized([(
        "CREATE",
        {"raw_seed": "seed_1"},
        [
            _fact(1, "source.available", {"source": "seed_1"}),
            _fact(1, "result.created", {"result": "new_entity_1"}),
        ],
    )])
    proposal = AtomicOccurrenceProposal(
        phase_id="create",
        intent="create a result from a source",
        event_start=0,
        event_end=0,
        input_roles={"source": "seed_1"},
        output_roles={"result": "new_entity_1"},
        preconditions=[SemanticPredicate(
            "result.created",
            {"result": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="result",
            )},
        )],
        effects=[SemanticPredicate(
            "result.created", {"result": "new_entity_1"},
        )],
        rationale="accepted creation transition",
        support_event_ids=["e0"],
        precondition_witness_refs=[source_ref],
        effect_witness_refs=[result_ref],
        input_provenance_refs={
            "source": "semantic_alias:e0:raw_seed:source",
        },
        output_derivations={
            "result": {
                "kind": "effect_witness",
                "predicate": "result.created",
                "argument_role": "result",
            },
        },
        input_provenance_contract="code_authority_v3_2",
    )

    with pytest.raises(
        ValueError,
        match=r"Atomic precondition references unavailable input role: create\.result",
    ):
        Atomicizer().validate_and_canonicalize([proposal], normalized)

    proposal.preconditions = []
    proposal.precondition_witness_refs = []
    canonical = Atomicizer().validate_and_canonicalize(
        [proposal], normalized,
    )[0]
    result = canonical.effects[0].args["result"]
    assert isinstance(result, BindingExpression)
    assert result.source_role == "result"
