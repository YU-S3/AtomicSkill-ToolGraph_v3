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
from atomic_skillgraph.harness.alfworld import AlfWorldContractMatcher
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
    return {
        "trace_id": "trace_guard",
        "source_task": {"task_id": "task", "task_signature": "sig", "task_type": "guard", "metadata": {}},
        "actions": actions,
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
    return AtomicOccurrenceProposal(
        phase,
        intent,
        index,
        index,
        inputs,
        outputs,
        list(preconditions or []),
        [effect],
        "validated guard proposal",
    )


def test_atomicizer_rejects_false_validator_and_terminal_effect_leak() -> None:
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
    with pytest.raises(ValueError, match="validator rejected"):
        Atomicizer().validate_and_canonicalize([proposal], normalized)

    # A terminal PUT may witness placement, but it cannot be used as a generic
    # witness for a different formal target effect achieved earlier.
    put = [_action(0, "PUT", {"object": "apple_1", "destination": "bowl_1"}, won=True)]
    forged = _proposal(
        "forged", "pretend to heat", 0,
        {"object": "apple_1", "destination": "bowl_1"},
        {"heated_object": "apple_1"}, heat,
    )
    with pytest.raises(ValueError, match="state/validator witness"):
        Atomicizer().validate_and_canonicalize(
            [forged],
            _normalized(put, target_effects=[heat, SemanticPredicate(
                "object.at_location", {"object": "apple_1", "location": "bowl_1"},
            )]),
        )

    with pytest.raises(ValueError, match="state/validator witness"):
        Atomicizer().validate_and_canonicalize(
            [replace(proposal, effects=[SemanticPredicate(
                "object.heated", {"object": "apple_1"}, cardinality=2,
            )])],
            _normalized(actions, target_effects=[heat]),
        )


def test_atomicizer_accepts_state_derived_observed_with_witness() -> None:
    actions = [
        _action(0, "GO_TO", {"destination": "desk_1"}),
        _action(1, "TAKE", {"object": "alarmclock_1", "source": "desk_1"}),
        _action(2, "USE", {"object": "desklamp_1"}, won=True),
    ]
    effect = SemanticPredicate(
        "object.observed_with", {"object": "alarmclock_1", "light": "desklamp_1"},
    )
    proposal = _proposal(
        "observe", "observe object under light", 2,
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
    assert any(ref.startswith("action:a2:") for ref in result[0].validation_refs)
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
    with pytest.raises(ValueError, match="state/validator witness"):
        Atomicizer().validate_and_canonicalize(
            [replace(proposal, event_start=2, event_end=2)],
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
    with pytest.raises(ValueError, match="before-state witness"):
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
        event_end=1,
        input_roles={"object": "apple_1", "destination": "bowl_1"},
        output_roles={"held_object": "apple_1"},
        preconditions=[],
        effects=[SemanticPredicate("agent.holds", {"object": "apple_1"})],
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
    with pytest.raises(ValueError, match="state/validator witness"):
        Atomicizer().validate_and_canonicalize([forged_holds], forged_trace)

    two_holds = replace(
        examine,
        event_start=1,
        event_end=1,
        preconditions=[SemanticPredicate(
            "agent.holds", {"object": "apple_1"}, cardinality=2,
        )],
    )
    with pytest.raises(ValueError, match="before-state witness"):
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


def test_implementation_admission_is_fail_closed_on_mapping_and_output(
    tmp_path,
) -> None:
    compiled = ToolCompiler().compile([_take_canonical()])[0]
    assert compiled.atomic.validator_spec["output_identity"] == [{
        "output_role": "held_object",
        "input_role": "item",
    }]
    database = StateDatabase(tmp_path / "identity.sqlite3")
    registry = SkillRegistry(ArtifactStore(tmp_path, database), database)
    registry.register_atomic(compiled.atomic)
    assert registry.get_atomic(compiled.atomic.ref).validator_spec[
        "output_identity"
    ] == [{
        "output_role": "held_object",
        "input_role": "item",
    }]
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
            {"object": "apple_1"}, {"observed_object": "apple_1"},
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
                input_roles={"object": "banana_1"},
                output_roles={"observed_object": "banana_1"},
                preconditions=[],
                effects=[SemanticPredicate("object.observed", {"object": "banana_1"})],
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


def test_alfworld_contract_matcher_preserves_family_and_concrete_identity() -> None:
    matcher = AlfWorldContractMatcher()
    offered_heat = SemanticPredicate("object.heated", {"object": "apple_2"})
    assert matcher.covers(
        SemanticPredicate("object.heated", {"object": "apple"}),
        offered_heat,
        {"object": "apple_2"},
    )
    assert not matcher.covers(
        SemanticPredicate("object.heated", {"object": "apple_1"}),
        offered_heat,
        {"object": "apple_2"},
    )

    offered_place = SemanticPredicate(
        "object.at_location", {"object": "apple_2", "location": "fridge_1"},
    )
    family_target = SemanticPredicate(
        "object.at_location", {"object": "apple", "location": "fridge"},
    )
    assert matcher.covers(
        family_target,
        offered_place,
        {"object": "apple_2", "location": "fridge_1"},
    )
    assert not matcher.covers(
        family_target,
        offered_place,
        {"object": "mug_1", "location": "fridge_1"},
    )
    assert not matcher.covers(
        family_target,
        offered_place,
        {"object": "apple_2", "location": "cabinet_1"},
    )
    concrete_target = SemanticPredicate(
        "object.at_location", {"object": "apple_1", "location": "fridge_1"},
    )
    assert not matcher.covers(
        concrete_target,
        offered_place,
        {"object": "apple_2", "location": "fridge_1"},
    )
    assert not matcher.covers(
        concrete_target,
        offered_place,
        {"object": "apple_1", "location": "fridge_2"},
    )


def test_composite_rejects_cross_object_effect_coverage_with_alfworld_matcher() -> None:
    actions = [
        _action(0, "HEAT", {"object": "mug_1", "station": "microwave_1"}),
        _action(
            1,
            "PUT",
            {"object": "apple_1", "destination": "fridge_1"},
            won=True,
        ),
    ]
    canonical = Atomicizer().validate_and_canonicalize(
        [
            _proposal(
                "heat_mug", "heat mug", 0,
                {"object": "mug_1", "station": "microwave_1"},
                {"heated_object": "mug_1"},
                SemanticPredicate("object.heated", {"object": "mug_1"}),
            ),
            _proposal(
                "place_apple", "place apple", 1,
                {"object": "apple_1", "destination": "fridge_1"},
                {"placed_object": "apple_1"},
                SemanticPredicate(
                    "object.at_location",
                    {"object": "apple_1", "location": "fridge_1"},
                ),
            ),
        ],
        _normalized(
            actions,
            target_effects=[
                SemanticPredicate("object.heated", {"object": "mug_1"}),
                SemanticPredicate(
                    "object.at_location",
                    {"object": "apple_1", "location": "fridge_1"},
                ),
            ],
        ),
    )
    with pytest.raises(ValueError, match="authoritative TaskContract"):
        CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                [item.occurrence_id for item in canonical],
                [],
                [],
                "heat mug then place apple",
                {},
                {},
            ),
            canonical,
            TaskContract([
                SemanticPredicate("object.heated", {"object": "apple"}),
                SemanticPredicate(
                    "object.at_location", {"object": "apple", "location": "fridge"},
                ),
            ]),
            contract_matcher=AlfWorldContractMatcher(),
        )


def _reversed_take_heat_canonical():
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
    return Atomicizer().validate_and_canonicalize(
        [
            _proposal(
                "heat", "heat held object", 1,
                {"object": "apple_1", "station": "microwave_1"},
                {"heated_object": "apple_1"},
                heated,
                preconditions=[held],
            ),
            _proposal(
                "take", "take object", 0,
                {"object": "apple_1", "source": "desk_1"},
                {"held_object": "apple_1"},
                held,
            ),
        ],
        _normalized(actions, target_effects=[held, heated]),
    )


def test_e1_sorts_reversed_proposals_before_assigning_occurrence_ids() -> None:
    canonical = _reversed_take_heat_canonical()
    assert [
        (item.event_start, item.event_end, item.phase_id)
        for item in canonical
    ] == [(0, 0, "take"), (1, 1, "heat")]
    assert [item.occurrence_id for item in canonical] == [
        "occ_trace_guard_000",
        "occ_trace_guard_001",
    ]


def test_e2_rejects_reversed_or_incomplete_canonical_sequence() -> None:
    canonical = _reversed_take_heat_canonical()
    expected = [item.occurrence_id for item in canonical]
    contract = TaskContract([
        SemanticPredicate("agent.holds", {"object": "apple_1"}),
        SemanticPredicate("object.heated", {"object": "apple_1"}),
    ])
    for invalid in (list(reversed(expected)), expected[:-1]):
        with pytest.raises(ValueError, match="canonical chronological order"):
            CompositeBuilder().validate_and_build(
                CompositeExtractionProposal(
                    invalid, [], [], "invalid occurrence sequence", {}, {},
                ),
                canonical,
                contract,
                contract_matcher=AlfWorldContractMatcher(),
            )


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
