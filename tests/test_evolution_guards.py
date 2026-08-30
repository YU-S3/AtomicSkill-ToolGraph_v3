from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    ToolBinding,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    ContractSource,
    IdentityConstraint,
    IdentityRelation,
    ImplementationAtom,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
    ToolAsset,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal, Atomicizer
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
from atomic_skillgraph.evolution.failure_processor import FailureProcessor
from atomic_skillgraph.evolution.gap_diagnosis import GapDiagnoser
from atomic_skillgraph.evolution.maintenance import EvolutionMaintenance
from atomic_skillgraph.evolution.repair import RepairStore
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.traces.schema import (
    ImplementationInvocationRecord,
    TaskRecord,
    TraceRecord,
)
from atomic_skillgraph.validation.engine import ValidationEngine
from atomic_skillgraph.validation.failure_localizer import FailureLocalizer


def _action(
    index: int,
    action_type: str,
    arguments: dict,
    *,
    won: bool = False,
    span_id: str | None = None,
) -> dict:
    return {
        "event_index": index,
        "action_id": f"a{index}",
        "action_type": action_type,
        "arguments": arguments,
        "accepted": True,
        "observation": "",
        "before_revision": index,
        "after_revision": index + 1,
        "done": won,
        "won": won,
        "span_id": span_id or f"s{index}",
    }


def _normalized(
    actions: list[dict],
    *,
    target_effects: list[SemanticPredicate],
    validations: list[dict] | None = None,
    benchmark_success: bool = True,
) -> dict:
    facts: dict[tuple[str, tuple[tuple[str, object], ...]], dict] = {}
    certified_actions: list[dict] = []

    def identity(predicate: str, arguments: dict) -> tuple[str, tuple[tuple[str, object], ...]]:
        return predicate, tuple(sorted(arguments.items()))

    def add(predicate: str, arguments: dict, event_index: int) -> dict:
        raw = {
            "fact_ref": f"effect:{event_index}:{predicate}",
            "predicate": predicate,
            "args": dict(arguments),
            "cardinality": 1,
            "distinct_by": "",
        }
        facts[identity(predicate, arguments)] = raw
        return raw

    for action in actions:
        event_index = int(action["event_index"])
        before = [dict(item) for item in facts.values()]
        arguments = dict(action.get("arguments") or {})
        action_type = str(action.get("action_type", ""))
        obj = arguments.get("object", arguments.get("item"))
        positive: list[dict] = []
        negative: list[dict] = []
        required: list[dict] = []

        for raw in before:
            if raw["predicate"] == "agent.holds" and raw["args"].get("object") == obj:
                if action_type in {"PUT", "HEAT", "COOL", "CLEAN", "EXAMINE", "USE"}:
                    required.append({
                        **raw,
                        "fact_ref": f"required:{event_index}:{raw['predicate']}",
                    })
            elif action_type == "USE" and raw["predicate"] == "agent.holds":
                required.append({
                    **raw,
                    "fact_ref": f"required:{event_index}:{raw['predicate']}",
                })
        if action_type == "TAKE" and obj:
            positive.append(add("agent.holds", {"object": obj}, event_index))
        elif action_type == "PUT" and obj:
            held_id = identity("agent.holds", {"object": obj})
            if held_id in facts:
                negative.append(facts.pop(held_id))
            positive.append(add(
                "object.at_location",
                {"object": obj, "location": arguments["destination"]},
                event_index,
            ))
        elif action_type == "HEAT" and obj:
            positive.append(add("object.heated", {"object": obj}, event_index))
        elif action_type == "COOL" and obj:
            positive.append(add("object.cooled", {"object": obj}, event_index))
        elif action_type == "CLEAN" and obj:
            positive.append(add("object.cleaned", {"object": obj}, event_index))
        elif action_type == "EXAMINE" and obj:
            positive.append(add("object.observed", {"object": obj}, event_index))
        elif action_type == "GO_TO":
            positive.append(add(
                "agent.at_location", {"location": arguments["destination"]}, event_index,
            ))
        elif action_type == "USE" and action.get("won"):
            for target in target_effects:
                if target.predicate == "object.observed_with":
                    positive.append(add(target.predicate, dict(target.args), event_index))

        certificate = {
            "action_id": action["action_id"],
            "revision_before": action["before_revision"],
            "revision_after": action["after_revision"],
            "action_type": action_type,
            "arguments": arguments,
            "before_facts": before,
            "positive_effects": positive,
            "negative_effects": negative,
            "required_facts": required,
            "terminal_effects": positive if action.get("won") else [],
            "accepted": True,
            "state_changed": bool(positive or negative),
            "evidence_refs": [f"certificate:{event_index}"],
        }
        certified_actions.append({**action, "transition_certificate": certificate})
    return {
        "trace_id": "trace_guard",
        "source_task": {"task_id": "task", "task_signature": "sig", "task_type": "guard", "metadata": {}},
        "actions": certified_actions,
        "runtime_spans": [
            {
                "span_id": item["span_id"],
                "kind": "full_dynamic",
                "occurrence_id": "",
                "action_start": index,
                "action_end": index + 1,
                "parent_span_id": None,
                "learnable": True,
            }
            for index, item in enumerate(actions)
        ],
        "validations": list(validations or []),
        "task_contract": to_primitive(TaskContract(
            target_effects=target_effects,
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="guard",
        )),
        "benchmark_success": benchmark_success,
    }


def _proposal(
    phase: str,
    intent: str,
    index: int,
    inputs: dict,
    outputs: dict,
    effect: SemanticPredicate,
    *,
    preconditions: list[SemanticPredicate] | None = None,
) -> AtomicOccurrenceProposal:
    effect_ref = f"effect:{index}:{effect.predicate}"
    output_mapping = {}
    for output_role, value in outputs.items():
        argument = next(
            name for name, effect_value in effect.args.items()
            if effect_value == value
        )
        output_mapping[output_role] = f"fact:{effect_ref}:{argument}"
    return AtomicOccurrenceProposal(
        phase_id=phase,
        intent=intent,
        event_start=index,
        event_end_exclusive=index + 1,
        selected_effect_refs=[effect_ref],
        selected_precondition_refs=[
            f"required:{index}:{item.predicate}"
            for item in (preconditions or [])
        ],
        output_role_mapping=output_mapping,
        rationale="validated guard proposal",
    )


def test_atomicizer_uses_certificate_refs_and_rejects_unlisted_effects() -> None:
    heat = SemanticPredicate("object.heated", {"object": "apple_1"})
    actions = [_action(0, "HEAT", {"object": "apple_1", "station": "microwave_1"}, won=True)]
    proposal = _proposal(
        "heat", "heat object", 0,
        {"object": "apple_1", "station": "microwave_1"},
        {"heated_object": "apple_1"}, heat,
    )
    normalized = _normalized(
        actions,
        target_effects=[heat],
        validations=[{"occurrence_id": "", "level": "atomic", "result": {"passed": False}, "revision": 1}],
    )
    # Transition certificates, not a second validation projection, are E1's
    # sole semantic authority.
    assert Atomicizer().validate_and_canonicalize([proposal], normalized)

    # A terminal PUT may witness placement, but it cannot be used as a generic
    # witness for a different formal target effect achieved earlier.
    put = [_action(0, "PUT", {"object": "apple_1", "destination": "bowl_1"}, won=True)]
    forged = _proposal(
        "forged", "pretend to heat", 0,
        {"object": "apple_1", "destination": "bowl_1"},
        {"heated_object": "apple_1"}, heat,
    )
    with pytest.raises(ValueError, match="unknown/out-of-boundary effect"):
        Atomicizer().validate_and_canonicalize(
            [replace(forged, selected_effect_refs=["effect:0:object.heated"])],
            _normalized(put, target_effects=[heat, SemanticPredicate(
                "object.at_location", {"object": "apple_1", "location": "bowl_1"},
            )]),
        )

    with pytest.raises(ValueError, match="must be unique"):
        Atomicizer().validate_and_canonicalize(
            [replace(proposal, selected_effect_refs=[
                "effect:0:object.heated", "effect:0:object.heated",
            ])],
            _normalized(actions, target_effects=[heat]),
        )


def test_atomicizer_accepts_formal_contextual_observed_with_witness() -> None:
    actions = [
        _action(0, "TAKE", {"object": "alarmclock_1", "source": "desk_1"}),
        _action(1, "USE", {"object": "desklamp_1"}, won=True),
    ]
    effect = SemanticPredicate(
        "object.observed_with", {"object": "alarmclock_1", "light": "desklamp_1"},
    )
    proposal = _proposal(
        "observe", "observe object under light", 1,
        {"object": "alarmclock_1", "light": "desklamp_1"},
        {"observed_object": "alarmclock_1"}, effect,
        preconditions=[
            SemanticPredicate("agent.holds", {"object": "alarmclock_1"}),
        ],
    )
    result = Atomicizer().validate_and_canonicalize(
        [proposal], _normalized(actions, target_effects=[effect]),
    )
    assert result[0].effects[0].predicate == "object.observed_with"
    assert any(ref.startswith("effect:") for ref in result[0].validation_refs)
    compiled = ToolCompiler().compile(result)[0]
    assert set(compiled.tool.signature["required"]) == {"object", "light"}
    admitted_tool = Admission(ValidationEngine().tool).admit_tool(
        compiled.tool, replay=lambda _tool, _case: True,
    )
    admitted_impl = Admission(ValidationEngine().tool).admit_implementation(
        compiled.implementation,
        admitted_tool,
        atomic=compiled.atomic,
        harness=SimpleNamespace(
            profile_name="alfworld_v3",
            supports_constraint=lambda kind, verifier_id="": True,
        ),
    )
    assert admitted_impl.status is SkillStatus.CANDIDATE

    forged_actions = [
        _action(0, "GO_TO", {"destination": "desklamp_1"}),
        _action(1, "TAKE", {"object": "alarmclock_1", "source": "desk_1"}),
        _action(2, "EXAMINE", {"object": "alarmclock_1"}, won=True),
    ]
    with pytest.raises(ValueError, match="unknown/out-of-boundary effect"):
        Atomicizer().validate_and_canonicalize(
            [replace(proposal, event_start=2, event_end_exclusive=3)],
            _normalized(forged_actions, target_effects=[effect]),
        )


def test_atomicizer_action_state_reducer_invalidates_stale_witnesses() -> None:
    actions = [
        _action(0, "TAKE", {"object": "apple_1", "source": "desk_1"}, span_id="slice"),
        _action(1, "PUT", {"object": "apple_1", "destination": "bowl_1"}, span_id="slice"),
        _action(2, "EXAMINE", {"object": "apple_1"}, won=True),
    ]
    examine = _proposal(
        "examine",
        "examine a held object",
        2,
        {"object": "apple_1"},
        {"observed_object": "apple_1"},
        SemanticPredicate("object.observed", {"object": "apple_1"}),
        preconditions=[SemanticPredicate("agent.holds", {"object": "apple_1"})],
    )
    with pytest.raises(ValueError, match="precondition references"):
        Atomicizer().validate_and_canonicalize(
            [examine],
            _normalized(
                actions,
                target_effects=[SemanticPredicate(
                    "object.observed", {"object": "apple_1"},
                )],
            ),
        )

    forged_holds = replace(
        examine,
        phase_id="take_then_put",
        intent="pretend to retain object",
        event_start=0,
        event_end_exclusive=2,
        selected_effect_refs=["effect:0:agent.holds"],
        selected_precondition_refs=[],
        output_role_mapping={
            "held_object": "fact:effect:0:agent.holds:object",
        },
    )
    forged_trace = _normalized(actions, target_effects=[SemanticPredicate(
        "object.observed", {"object": "apple_1"},
    )])
    forged_trace["runtime_spans"] = [{
        "span_id": "slice",
        "kind": "full_dynamic",
        "occurrence_id": "",
        "action_start": 0,
        "action_end": 2,
        "parent_span_id": None,
        "learnable": True,
    }]
    with pytest.raises(ValueError, match="negated within"):
        Atomicizer().validate_and_canonicalize([forged_holds], forged_trace)

    two_holds = replace(
        examine,
        event_start=1,
        event_end_exclusive=2,
        selected_precondition_refs=[
            "required:1:agent.holds",
            "required:1:agent.holds",
        ],
    )
    with pytest.raises(ValueError, match="must be unique"):
        Atomicizer().validate_and_canonicalize(
            [two_holds],
            _normalized(
                [
                    _action(0, "TAKE", {"object": "apple_1", "source": "desk_1"}),
                    _action(1, "EXAMINE", {"object": "apple_1"}, won=True),
                ],
                target_effects=[SemanticPredicate(
                    "object.observed", {"object": "apple_1"},
                )],
            ),
        )


def _take_canonical():
    effect = SemanticPredicate("agent.holds", {"object": "apple_1"})
    return Atomicizer().validate_and_canonicalize(
        [_proposal(
            "take", "take item", 0,
            {"item": "apple_1"}, {"held_object": "apple_1"}, effect,
        )],
        _normalized([_action(0, "TAKE", {"item": "apple_1"}, won=True)], target_effects=[effect]),
    )[0]


def test_implementation_admission_is_fail_closed_on_mapping_and_output() -> None:
    compiled = ToolCompiler().compile([_take_canonical()])[0]
    admission = Admission(ValidationEngine().tool)
    tool = admission.admit_tool(compiled.tool, replay=lambda _tool, _case: True)
    harness = SimpleNamespace(
        profile_name="fake_v3",
        supports_constraint=lambda kind, verifier_id="": kind in {
            "argument_exists", "argument_concrete", "harness_affordance", "current_context",
        } or bool(verifier_id),
    )
    valid = admission.admit_implementation(
        compiled.implementation, tool, atomic=compiled.atomic, harness=harness,
    )
    assert valid.status is SkillStatus.CANDIDATE

    binding = compiled.implementation.tool_bindings[0]
    invalid_binding = replace(binding, parameter_mapping={
        "item": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="missing_role"),
    })
    invalid = admission.admit_implementation(
        replace(compiled.implementation, tool_bindings=[invalid_binding]),
        tool,
        atomic=compiled.atomic,
        harness=harness,
    )
    assert invalid.status is SkillStatus.SHADOW
    assert "implementation_mapping_not_closed" in invalid.quality["admission_failure"]

    invalid_output = replace(
        compiled.implementation,
        execution_policy={
            "mode": "serial",
            "output_mapping": {
                "held_object": BindingExpression(
                    BindingExprKind.TOOL_OUTPUT,
                    source_role="held_object",
                    source_step="forged_step",
                ),
            },
        },
    )
    rejected = admission.admit_implementation(
        invalid_output, tool, atomic=compiled.atomic, harness=harness,
    )
    assert rejected.status is SkillStatus.SHADOW
    assert "output_mapping_not_closed" in rejected.quality["admission_failure"]


def _atomic(ref: SkillRef, *, precondition: str = "", validator: str = "v", step: str = "TAKE") -> AbstractAtomicSkill:
    conditions = [SemanticPredicate(precondition, {"object": BindingExpression(
        BindingExprKind.SKILL_INPUT, source_role="item",
    )})] if precondition else []
    return AbstractAtomicSkill(
        ref,
        "take item",
        [ParameterSpec("item", "string", True, True, "concrete")],
        [ParameterSpec("held_object", "entity")],
        conditions,
        [SemanticPredicate("agent.holds", {"object": BindingExpression(
            BindingExprKind.SKILL_INPUT, source_role="item",
        )})],
        {"validator_id": validator},
        [],
        {"steps": [step]},
        {},
        SkillStatus.CANDIDATE,
    )


def test_aligner_does_not_merge_incompatible_atomic_contracts(tmp_path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    aligner = Aligner(skills, tools)
    first = aligner.align_atomic(_atomic(SkillRef("atomic_take", "1.0.0")))
    second = aligner.align_atomic(_atomic(
        SkillRef("atomic_take", "1.0.0"),
        precondition="container.open",
        validator="strict_v2",
        step="OPEN_THEN_TAKE",
    ))
    assert first != second
    assert first.logical_id != second.logical_id
    assert first.version == second.version == "1.0.0"
    shadow = replace(
        _atomic(
            SkillRef("atomic_shadow", "1.0.0"),
            validator="shadow_only",
            step="SHADOW_TAKE",
        ),
        status=SkillStatus.SHADOW,
    )
    skills.register_atomic(shadow)
    revived = aligner.align_atomic(replace(shadow, status=SkillStatus.DRAFT))
    assert revived.logical_id != shadow.ref.logical_id
    assert revived.version == "1.0.0"
    assert skills.get_atomic(revived).status is SkillStatus.CANDIDATE
    database.close()


def test_same_atomic_contract_different_tools_align_same_atomic(tmp_path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    aligner = Aligner(skills, tools)

    first_candidate = _atomic(SkillRef("wording_take", "1.0.0"))
    second_candidate = replace(
        first_candidate,
        ref=SkillRef("renamed_pickup", "1.0.0"),
        summary="pick up the requested target using another implementation",
        inputs=[ParameterSpec("target", "string", True, True, "concrete")],
        outputs=[ParameterSpec("acquired", "entity")],
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role="target",
            )},
        )],
        failure_modes=[{"wording_only": "different prose"}],
        guideline={"steps": ["NAVIGATE", "PICKUP_WITH_OTHER_TOOL"]},
    )
    atomic_ref = aligner.align_atomic(first_candidate)
    assert aligner.align_atomic(second_candidate) == atomic_ref

    def candidate_tool(tool_id: str, command: str) -> ToolAsset:
        return ToolAsset(
            ToolRef(tool_id, "1.0.0"),
            command,
            {"arguments": ["object"]},
            {"input_roles": ["object"], "output_roles": ["held_object"]},
            "harness_action",
            {"action_type": command},
            [],
            {"side_effects": ["world_state_change"]},
            {},
            {},
            ToolStatus.CANDIDATE,
        )

    tool_a = aligner.align_tool(candidate_tool("tool_take", "TAKE"))
    tool_b = aligner.align_tool(candidate_tool("tool_pickup", "PICKUP"))
    assert tool_a != tool_b

    def candidate_implementation(
        logical_id: str, tool_ref: ToolRef, source_role: str,
    ) -> ImplementationAtom:
        return ImplementationAtom(
            SkillRef(logical_id, "1.0.0"),
            SkillRef("trace_local_atomic", "1.0.0"),
            [ToolBinding(
                tool_ref,
                "execute",
                {"object": BindingExpression(
                    BindingExprKind.SKILL_INPUT, source_role=source_role,
                )},
            )],
            [],
            {"mode": "serial"},
            {"harness_profiles": ["alfworld_v3"]},
            {},
            SkillStatus.CANDIDATE,
        )

    implementation_a = aligner.align_implementation(
        candidate_implementation("impl_take", tool_a, "item"), atomic_ref, tool_a,
    )
    implementation_b = aligner.align_implementation(
        candidate_implementation("impl_pickup", tool_b, "target"), atomic_ref, tool_b,
    )
    assert implementation_a != implementation_b
    assert skills.get_implementation(implementation_a).abstract_ref == atomic_ref
    assert skills.get_implementation(implementation_b).abstract_ref == atomic_ref
    database.close()


def test_composite_requires_reused_identity_dataflow_and_identity_consistency() -> None:
    actions = [
        _action(0, "TAKE", {"item": "apple_1"}),
        _action(1, "EXAMINE", {"item": "apple_1"}, won=True),
    ]
    proposals = [
        _proposal(
            "take", "take item", 0,
            {"item": "apple_1"}, {"held_object": "apple_1"},
            SemanticPredicate("agent.holds", {"object": "apple_1"}),
        ),
        _proposal(
            "examine", "examine held item", 1,
            {"item": "apple_1"}, {"observed_object": "apple_1"},
            SemanticPredicate("object.observed", {"object": "apple_1"}),
            preconditions=[SemanticPredicate("agent.holds", {"object": "apple_1"})],
        ),
    ]
    canonical = Atomicizer().validate_and_canonicalize(
        proposals,
        _normalized(actions, target_effects=[SemanticPredicate(
            "object.observed", {"object": "apple_1"},
        )]),
    )
    sequence = [item.occurrence_id for item in canonical]
    disconnected = CompositeExtractionProposal(sequence, [], [], "take and examine", {}, {})
    with pytest.raises(ValueError, match="explicit DataFlow"):
        CompositeBuilder().validate_and_build(
            disconnected,
            canonical,
            TaskContract([SemanticPredicate("object.observed", {"object": "apple_1"})]),
        )

    edge = {
        "edge_id": "held_to_item",
        "edge_type": "data_flow",
        "source_step": sequence[0],
        "target_step": sequence[1],
        "source_role": "held_object",
        "target_role": "object",
    }
    composite = CompositeBuilder().validate_and_build(
        CompositeExtractionProposal(sequence, [], [edge], "take and examine", {}, {}),
        canonical,
        TaskContract([SemanticPredicate("object.observed", {"object": "apple_1"})]),
    )
    assert composite.occurrences[1].binding_specs["object"].kind is BindingExprKind.DATA_FLOW

    mismatched_actions = [
        _action(0, "TAKE", {"item": "apple_1"}),
        _action(1, "EXAMINE", {"item": "banana_1"}, won=True),
    ]
    mismatched = Atomicizer().validate_and_canonicalize(
        [
            proposals[0],
            replace(
                proposals[1],
                selected_precondition_refs=[],
            ),
        ],
        _normalized(mismatched_actions, target_effects=[SemanticPredicate(
            "object.observed", {"object": "banana_1"},
        )]),
    )
    mismatched_sequence = [item.occurrence_id for item in mismatched]
    mismatched_edge = {
        **edge,
        "source_step": mismatched_sequence[0],
        "target_step": mismatched_sequence[1],
    }
    with pytest.raises(ValueError, match="binding identity"):
        CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                mismatched_sequence, [], [mismatched_edge], "forged flow", {}, {},
            ),
            mismatched,
            TaskContract([SemanticPredicate("object.observed", {"object": "banana_1"})]),
        )

    mixed_actions = [
        _action(0, "HEAT", {"object": "apple_1", "station": "microwave_1"}),
        _action(1, "PUT", {"object": "banana_1", "destination": "bowl_1"}, won=True),
    ]
    mixed = Atomicizer().validate_and_canonicalize(
        [
            _proposal(
                "heat", "heat object", 0,
                {"object": "apple_1", "station": "microwave_1"},
                {"heated_object": "apple_1"},
                SemanticPredicate("object.heated", {"object": "apple_1"}),
            ),
            _proposal(
                "put", "place object", 1,
                {"object": "banana_1", "destination": "bowl_1"},
                {"placed_object": "banana_1"},
                SemanticPredicate("object.at_location", {"object": "banana_1", "location": "bowl_1"}),
            ),
        ],
        _normalized(mixed_actions, target_effects=[
            SemanticPredicate("object.heated", {"object": "apple_1"}),
            SemanticPredicate("object.at_location", {"object": "banana_1", "location": "bowl_1"}),
        ]),
    )
    control_only = CompositeBuilder().validate_and_build(
        CompositeExtractionProposal(
            [item.occurrence_id for item in mixed], [], [], "independent effects", {}, {},
        ),
        mixed,
        TaskContract([
            SemanticPredicate("object.heated", {"object": "apple_1"}),
            SemanticPredicate(
                "object.at_location", {"object": "banana_1", "location": "bowl_1"},
            ),
        ]),
    )
    assert control_only.data_edges == []
    assert control_only.dependency_edges == []

    contract = TaskContract(
        [
            SemanticPredicate("object.heated", {"object": "target"}),
            SemanticPredicate("object.at_location", {"object": "target", "location": "bowl"}),
        ],
        identity_constraints=[IdentityConstraint(
            "object", IdentityRelation.SAME_AS, "object", "task",
        )],
    )
    with pytest.raises(ValueError, match="TaskContract"):
        CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                [item.occurrence_id for item in mixed], [], [], "invalid mixed objects", {}, {},
            ),
            mixed,
            contract,
        )


def test_irrelevant_valid_exploration_occurrence_not_in_composite() -> None:
    target = SemanticPredicate(
        "object.at_location",
        {"object": "apple_1", "location": "fridge_1"},
    )
    actions = [
        _action(0, "GO_TO", {"destination": "cabinet_7"}),
        _action(1, "TAKE", {"object": "apple_1", "source": "desk_1"}),
        _action(
            2,
            "PUT",
            {"object": "apple_1", "destination": "fridge_1"},
            won=True,
        ),
    ]
    proposals = [
        _proposal(
            "explore",
            "visit an unrelated location",
            0,
            {"destination": "cabinet_7"},
            {"visited_location": "cabinet_7"},
            SemanticPredicate("agent.at_location", {"location": "cabinet_7"}),
        ),
        _proposal(
            "acquire",
            "acquire the target",
            1,
            {"object": "apple_1", "source": "desk_1"},
            {"held_object": "apple_1"},
            SemanticPredicate("agent.holds", {"object": "apple_1"}),
        ),
        _proposal(
            "place",
            "place the target",
            2,
            {"object": "apple_1", "destination": "fridge_1"},
            {"placed_object": "apple_1"},
            target,
            preconditions=[SemanticPredicate(
                "agent.holds", {"object": "apple_1"},
            )],
        ),
    ]
    canonical = Atomicizer().validate_and_canonicalize(
        proposals,
        _normalized(actions, target_effects=[target]),
    )
    by_phase = {item.phase_id: item for item in canonical}
    selected = [
        by_phase["acquire"].occurrence_id,
        by_phase["place"].occurrence_id,
    ]
    edge = {
        "edge_id": "held_to_placed",
        "edge_type": "data_flow",
        "source_step": selected[0],
        "target_step": selected[1],
        "source_role": "held_object",
        "target_role": "object",
    }
    with pytest.raises(ValueError, match="minimal task-causal closure"):
        CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                [item.occurrence_id for item in canonical],
                [],
                [edge],
                "exploration plus task actions",
                {},
                {},
            ),
            canonical,
            TaskContract([target]),
        )
    composite = CompositeBuilder().validate_and_build(
        CompositeExtractionProposal(
            selected, [], [edge], "acquire and place target", {}, {},
        ),
        canonical,
        TaskContract([target]),
    )

    assert len(canonical) == 3
    assert composite.control_sequence == selected
    assert by_phase["explore"].metadata["not_in_task_composite"] is True
    assert by_phase["acquire"].metadata["not_in_task_composite"] is False
    assert composite.metadata["excluded_atomic_occurrences"] == [
        by_phase["explore"].occurrence_id,
    ]
    compiled = ToolCompiler().compile(canonical)
    assert len(compiled) == 3
    assert next(
        item for item in compiled if item.occurrence.phase_id == "explore"
    ).atomic.metadata["not_in_task_composite"] is True


@pytest.mark.parametrize(
    ("actual_object", "actual_destination"),
    [("mug_1", "fridge_1"), ("apple_1", "cabinet_1")],
)
def test_composite_contract_rejects_wrong_object_or_destination_family(
    actual_object: str,
    actual_destination: str,
) -> None:
    class FamilyMatcher:
        @staticmethod
        def effect_covers_target(
            *, offered_predicate, offered_arguments, target_predicate,
        ) -> bool:
            def compatible(observed, expected) -> bool:
                if not isinstance(observed, str) or not isinstance(expected, str):
                    return observed == expected
                return observed == expected or observed.rsplit("_", 1)[0] == expected

            return (
                offered_predicate.predicate.casefold()
                == target_predicate.predicate.casefold()
                and set(offered_arguments) == set(target_predicate.args)
                and all(
                    compatible(offered_arguments[role], expected)
                    for role, expected in target_predicate.args.items()
                )
            )

    offered = SemanticPredicate(
        "object.at_location",
        {"object": actual_object, "location": actual_destination},
    )
    target = SemanticPredicate(
        "object.at_location", {"object": "apple", "location": "fridge"},
    )
    action = _action(
        0,
        "PUT",
        {"object": actual_object, "destination": actual_destination},
        won=True,
    )
    canonical = Atomicizer().validate_and_canonicalize(
        [_proposal(
            "place",
            "place an object",
            0,
            {"object": actual_object, "destination": actual_destination},
            {"placed_object": actual_object},
            offered,
        )],
        _normalized([action], target_effects=[target]),
    )

    with pytest.raises(ValueError, match="no Atomic effect covers"):
        CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                [canonical[0].occurrence_id], [], [], "place target", {}, {},
            ),
            canonical,
            TaskContract([target]),
            contract_matcher=FamilyMatcher(),
        )


def test_e1_occurrences_are_chronologically_canonicalized() -> None:
    held = SemanticPredicate("agent.holds", {"object": "apple_1"})
    heated = SemanticPredicate("object.heated", {"object": "apple_1"})
    actions = [
        _action(0, "TAKE", {"object": "apple_1", "source": "desk_1"}),
        _action(
            1,
            "HEAT",
            {"object": "apple_1", "station": "microwave_1"},
            won=True,
        ),
    ]
    proposals = [
        _proposal(
            "heat",
            "heat the held object",
            1,
            {"object": "apple_1", "station": "microwave_1"},
            {"heated_object": "apple_1"},
            heated,
            preconditions=[held],
        ),
        _proposal(
            "acquire",
            "acquire the object",
            0,
            {"object": "apple_1", "source": "desk_1"},
            {"held_object": "apple_1"},
            held,
        ),
    ]

    canonical = Atomicizer().validate_and_canonicalize(
        proposals,
        _normalized(actions, target_effects=[heated]),
    )

    assert [item.phase_id for item in canonical] == ["acquire", "heat"]
    assert [item.event_start for item in canonical] == [0, 1]


def test_e1_overlap_resolution_uses_declared_deterministic_priority() -> None:
    held = SemanticPredicate("agent.holds", {"object": "apple_1"})
    heated = SemanticPredicate("object.heated", {"object": "apple_1"})
    actions = [
        _action(
            0,
            "TAKE",
            {"object": "apple_1", "source": "desk_1"},
            span_id="slice",
        ),
        _action(
            1,
            "HEAT",
            {"object": "apple_1", "station": "microwave_1"},
            won=True,
            span_id="slice",
        ),
    ]
    normalized = _normalized(actions, target_effects=[heated])
    normalized["runtime_spans"] = [{
        "span_id": "slice",
        "kind": "full_dynamic",
        "occurrence_id": "",
        "action_start": 0,
        "action_end": 2,
        "parent_span_id": None,
        "learnable": True,
    }]
    short = _proposal(
        "short",
        "heat the held object",
        1,
        {"object": "apple_1", "station": "microwave_1"},
        {"heated_object": "apple_1"},
        heated,
        preconditions=[held],
    )
    rich = AtomicOccurrenceProposal(
        phase_id="z_rich",
        intent="acquire and heat",
        event_start=0,
        event_end_exclusive=2,
        selected_effect_refs=[
            "effect:0:agent.holds",
            "effect:1:object.heated",
        ],
        selected_precondition_refs=[],
        output_role_mapping={
            "heated_object": "fact:effect:1:object.heated:object",
        },
        rationale="two authoritative effects",
    )
    sparse = replace(
        rich,
        phase_id="a_sparse",
        selected_effect_refs=["effect:1:object.heated"],
        rationale="one authoritative effect",
    )

    accepted, rejected = Atomicizer().validate_proposed_subset(
        [rich, short], normalized,
    )
    assert [item.phase_id for item in accepted] == ["short"]
    assert rejected[0]["phase_id"] == "z_rich"

    accepted, rejected = Atomicizer().validate_proposed_subset(
        [sparse, rich], normalized,
    )
    assert [item.phase_id for item in accepted] == ["z_rich"]
    assert rejected[0]["phase_id"] == "a_sparse"


def test_failure_repair_queue_is_structured_and_checkpointed_in_sqlite(tmp_path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    trace = TraceRecord.create(
        TaskRecord("task", "fake", "goal", "guard", "sig"),
        {},
        {},
        {"source": "stored_composite", "source_composite_ref": "skill://composite_x@1.0.0"},
    )
    trace.implementation_invocations.append(ImplementationInvocationRecord(
        "attempt", "occ", "skill://impl_x@1.0.0", {},
        {
            "passed": False,
            "failure_layer": "implementation",
            "failure_code": "implementation_mapping_error",
            "matched_evidence_refs": [],
            "message": "",
        },
        {
            "started": False,
            "failure_layer": "implementation",
            "failure_code": "implementation_mapping_error",
        },
        "span",
    ))
    failures = FailureProcessor(FailureLocalizer()).localize(trace)
    maintenance = EvolutionMaintenance(RepairStore(database))
    proposals = maintenance.prepare_failure_repairs(failures)
    assert len(proposals) == 1
    assert proposals[0].operation == "revise_implementation_mapping"
    assert proposals[0].proposed_patch["requires_concrete_patch"] is True

    digest_view = SimpleNamespace(database=database, artifacts=artifacts)
    before = AtomicSkillGraphSystem.knowledge_digest(digest_view)
    maintenance.commit_repairs(proposals)
    after = AtomicSkillGraphSystem.knowledge_digest(digest_view)
    assert after != before
    assert RepairStore(database).list()[0].proposal_id == proposals[0].proposal_id

    canonical = [_take_canonical()]
    candidate = CompositeBuilder().validate_and_build(
        CompositeExtractionProposal(
            [canonical[0].occurrence_id], [], [], "repaired composite", {}, {},
        ),
        canonical,
        TaskContract([SemanticPredicate("agent.holds", {"object": "apple_1"})]),
    )
    concrete = maintenance.prepare_validated_composite_repair(
        "skill://composite_old@1.0.0", candidate, [failures[0].failure_id],
    )
    admitted = maintenance.admit_validated_composite_repair(
        concrete, admitted_ref=str(candidate.ref), validation_passed=True,
    )
    assert admitted.status == "admitted"
    assert admitted.replay_result["passed"] is True
    assert not (tmp_path / "evolution" / "repair_proposals.json").exists()
    database.close()


def test_gap_diagnosis_distinguishes_retrieval_miss_and_real_gap(tmp_path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    existing = _atomic(SkillRef("atomic_take", "1.0.0"))
    skills.register_atomic(existing)
    requirement = CapabilityRequirement(
        "r1", "take item", existing.effects, existing.inputs, existing.outputs,
        [], [], True, "required",
    )
    audit = {
        "atomic_search_p1r": [{
            "requirement": to_primitive(requirement),
            "candidates": [],
            "covered": False,
            "rejection_reasons": [],
        }],
    }
    trace = SimpleNamespace(
        runtime_plan={"source": "full_dynamic"},
        benchmark_success=True,
        planner_audit=audit,
    )
    diagnosis = GapDiagnoser(skills).diagnose(trace, [replace(existing, ref=SkillRef("atomic_take_other", "1.0.0"))])
    assert diagnosis["classification"] == "retrieval_miss"

    new_contract = replace(existing, ref=SkillRef("atomic_take_new", "1.0.0"), validator_spec={"validator_id": "new"})
    diagnosis = GapDiagnoser(skills).diagnose(trace, [new_contract])
    assert diagnosis["classification"] == "confirmed_capability_gap"
    assert diagnosis["skill_penalty_applied"] is False
    database.close()


def test_frozen_trace_output_cannot_be_inside_snapshot(tmp_path) -> None:
    snapshot = tmp_path / "frozen"
    trace_dir = snapshot / "eval_output"
    config = {
        "schema_version": 3,
        "data_dir": str(snapshot),
        "trace_data_dir": str(trace_dir),
        "condition": "full",
        "experiment": {
            "benchmark": "alfworld",
            "runtime_mode": "frozen",
            "freeze_skills": True,
            "allow_long_term_knowledge_writes": False,
        },
        "llm": {"provider": "openai_compatible", "api_key_env": "MODEL_API_KEY"},
        "planner": {"requirement_repair_limit": 1, "graph_repair_limit": 1},
    }
    with pytest.raises(ValueError, match="outside the immutable snapshot"):
        AtomicSkillGraphSystem(config, readonly=True)
    assert not trace_dir.exists()
