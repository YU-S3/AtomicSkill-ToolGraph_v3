from __future__ import annotations

from types import SimpleNamespace

import pytest

import atomic_skillgraph.system as system_module
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    AtomicEffectResolution,
    RuntimeOccurrence,
    ValidationResult,
)
from atomic_skillgraph.runtime.automation import (
    _trial_harness_effect_event_authorities,
)
from atomic_skillgraph.validation.atomic_validator import AtomicValidator
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from atomic_skillgraph.system import AtomicSkillGraphSystem


class _FreshOutputChannel:
    def resolve_atomic_effect(self, _request):
        return AtomicEffectResolution(
            True,
            resolved_bindings={"found_entity": "cup_3"},
            output_candidates={},
            witness_refs=[
                "validator:r1:entity.discovered_at:entity=cup_3"
            ],
        )

    def validate_atomic_effect(self, _request):
        return ValidationResult.ok("atomic", effect_witness=True)


def _fresh_output_atomic() -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef("atomic_find_entity", "1.0.0"),
        summary="find an entity",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("found_entity", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {
                "entity": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="found_entity",
                ),
                "location": "countertop",
            },
            effect_domain="evidence",
        )],
        validator_spec={
            "output_derivations": {
                "found_entity": {
                    "kind": "effect_witness",
                    "predicate": "entity.discovered_at",
                    "argument_role": "entity",
                }
            }
        },
        failure_modes=[],
        guideline={},
        metadata={},
    )


def test_effect_witness_validates_fresh_output_when_role_names_differ() -> None:
    atomic = _fresh_output_atomic()
    occurrence = RuntimeOccurrence(
        "step", "occ", atomic.ref, [], {}, [], list(atomic.effects),
    )
    validator = AtomicValidator()

    rejected = validator.validate_execution_result(
        atomic,
        occurrence,
        {"target": "cup"},
        {"found_entity": "plate_2"},
        _FreshOutputChannel(),
        current_revision=1,
    )
    accepted = validator.validate_execution_result(
        atomic,
        occurrence,
        {"target": "cup"},
        {"found_entity": "cup_3"},
        _FreshOutputChannel(),
        current_revision=1,
    )

    assert rejected.passed is False
    assert rejected.failure_codes == [
        "atomic_output_effect_witness_mismatch"
    ]
    assert accepted.passed is True
    assert accepted.witness_refs == [
        "validator:r1:entity.discovered_at:entity=cup_3"
    ]


def test_only_admission_eligible_runtime_trial_projects_real_r1_authority() -> None:
    trial = {
        "draft_id": "locate_1",
        "trial_bindings": {"target": "cup"},
        "r1_outputs": {
            "found_entity": "cup_3",
            "found_location": "countertop_2",
        },
        "r1_witness_refs": [
            "alfworld_action_fact:r4:entity.discovered_at:"
            "entity=cup_3,location=countertop_2"
        ],
        "tool_path_witness_refs": ["semantic_evidence:entity"],
        "declared_effects": [{
            "predicate": "entity.discovered_at",
            "args": {
                "entity": {
                    "kind": "skill_input",
                    "source_role": "found_entity",
                },
                "location": "$found_location",
            },
            "effect_domain": "evidence",
        }],
        "output_derivations": {
            "found_entity": {
                "kind": "effect_witness",
                "predicate": "entity.discovered_at",
                "argument_role": "entity",
            },
            "found_location": {
                "kind": "effect_witness",
                "predicate": "entity.discovered_at",
                "argument_role": "location",
            },
        },
        "after_revision": 4,
        "trial_event_start": 0,
        "trial_event_end": 0,
        "r1_effect_event_authorities": [{
            "predicate": "entity.discovered_at",
            "args": {
                "entity": "cup_3",
                "location": "countertop_2",
            },
            "effect_domain": "evidence",
            "event_index": 0,
            "revision": 4,
            "source_kind": "occurrence_action_delta",
            "source_occurrence_id": "occ",
        }],
        "r1": {"admission_eligible": True},
    }
    actions = [{
        "event_index": 0,
        "accepted": True,
        "after_revision": 4,
        "authoritative_positive_effects": [],
    }]

    facts = AtomicSkillGraphSystem._runtime_trial_effect_authorities(
        trial, actions,
    )

    assert facts == [{
        "predicate": "entity.discovered_at",
        "args": {
            "entity": "cup_3",
            "location": "countertop_2",
        },
        "cardinality": 1,
        "distinct_by": "",
        "effect_domain": "evidence",
        "witness_ref": (
            "alfworld_action_fact:r4:entity.discovered_at:"
            "entity=cup_3,location=countertop_2"
        ),
        "revision": 4,
        "source_kind": "runtime_trial_r1",
        "draft_id": "locate_1",
        "event_index": 0,
    }]
    assert all(
        fact["witness_ref"] != "semantic_evidence:entity"
        for fact in facts
    )

    rejected = {
        **trial,
        "r1": {"admission_eligible": False},
    }
    assert (
        AtomicSkillGraphSystem._runtime_trial_effect_authorities(
            rejected, actions,
        )
        == []
    )


def test_runtime_effect_owner_does_not_guess_last_trial_action() -> None:
    trial = {
        "trial_event_start": 0,
        "trial_event_end": 0,
        "after_revision": 4,
    }
    actions = [{
        "event_index": 0,
        "accepted": True,
        "after_revision": 4,
        "authoritative_positive_effects": [],
    }]
    fact = {
        "predicate": "entity.discovered_at",
        "args": {"entity": "cup_3", "location": "countertop_2"},
        "effect_domain": "evidence",
    }

    assert AtomicSkillGraphSystem._runtime_effect_event_index(
        trial, actions, fact,
    ) is None


def test_occurrence_local_harness_delta_supplies_exact_effect_owner() -> None:
    actions = [{
        "accepted": True,
        "new_revision": 4,
    }]
    snapshots = [{
        "occurrence_id": "occ_locate",
        "revision": 4,
        "active_facts": [{
            "predicate": "entity.discovered_at",
            "args": {
                "entity": "cup_3",
                "location": "countertop_2",
            },
        }],
    }]
    authorities = _trial_harness_effect_event_authorities(
        baseline_facts=[],
        evidence_snapshots=snapshots,
        environment_actions=actions,
        trial_event_start=0,
        trial_event_end=0,
        occurrence_id="occ_locate",
        predicate_domains={"entity.discovered_at": "evidence"},
    )
    assert authorities == [{
        "predicate": "entity.discovered_at",
        "args": {
            "entity": "cup_3",
            "location": "countertop_2",
        },
        "effect_domain": "evidence",
        "event_index": 0,
        "revision": 4,
        "source_kind": "occurrence_action_delta",
        "source_occurrence_id": "occ_locate",
    }]

    trial = {
        "trial_event_start": 0,
        "trial_event_end": 0,
        "after_revision": 4,
        "r1_effect_event_authorities": authorities,
    }
    normalized_actions = [{
        "event_index": 0,
        "accepted": True,
        "after_revision": 4,
        "authoritative_positive_effects": [],
    }]
    assert AtomicSkillGraphSystem._runtime_effect_event_index(
        trial,
        normalized_actions,
        {
            "predicate": "entity.discovered_at",
            "args": {
                "entity": "cup_3",
                "location": "countertop_2",
            },
            "effect_domain": "evidence",
        },
    ) == 0


def test_prepare_evolution_exposes_runtime_r1_fact_to_e1(
    monkeypatch,
) -> None:
    captured = {}

    class _CaptureExtractor:
        def __init__(self, _session):
            pass

        def propose_atomics(self, normalized, *_args, **_kwargs):
            captured.update(normalized)
            raise RuntimeError("captured")

    monkeypatch.setattr(system_module, "ExtractorSession", _CaptureExtractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.normalizer = SimpleNamespace(build=lambda _trace: {
        "trace_id": "trace",
        "source_task": {},
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [{
            "event_index": 0,
            "action_id": "a1",
            "action_type": "SEARCH",
            "arguments": {},
            "accepted": True,
            "before_revision": 3,
            "after_revision": 4,
            "span_id": "span",
            "authoritative_positive_effects": [],
        }],
        "runtime_spans": [],
        "validations": [],
        "boundary_authorities": {"inputs": [], "effects": []},
    })
    system._extractor_session = lambda _task_id: object()
    system.skills = object()
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(),
        contract_matcher=lambda: ExactContractMatcher(),
        semantic_predicate_schema=lambda: [],
    )
    trace = SimpleNamespace(
        metadata={"runtime_tool_trials": {"locate_1": {
            "draft_id": "locate_1",
            "trial_bindings": {"target": "cup"},
            "input_authorities": {"target": {
                "authority_ref": "runtime_input:locate_1:target",
                "kind": "current_occurrence_anchor",
                "source_occurrence_id": "occ",
                "source_role": "object",
                "value": "cup",
            }},
            "r1_outputs": {
                "entity": "cup_3",
                "location": "countertop_2",
            },
            "r1_witness_refs": [
                "alfworld_action_fact:r4:entity.discovered_at:"
                "entity=cup_3,location=countertop_2"
            ],
            "declared_effects": [{
                "predicate": "entity.discovered_at",
                "args": {"entity": "$entity", "location": "$location"},
                "effect_domain": "evidence",
            }],
            "output_derivations": {
                "entity": {
                    "kind": "effect_witness",
                    "predicate": "entity.discovered_at",
                    "argument_role": "entity",
                },
                "location": {
                    "kind": "effect_witness",
                    "predicate": "entity.discovered_at",
                    "argument_role": "location",
                },
            },
            "after_revision": 4,
            "trial_event_start": 0,
            "trial_event_end": 0,
            "r1_effect_event_authorities": [{
                "predicate": "entity.discovered_at",
                "args": {
                    "entity": "cup_3",
                    "location": "countertop_2",
                },
                "effect_domain": "evidence",
                "event_index": 0,
                "revision": 4,
                "source_kind": "occurrence_action_delta",
                "source_occurrence_id": "occ",
            }],
            "r1": {"admission_eligible": True},
        }}},
        runtime_plan={},
    )

    with pytest.raises(RuntimeError, match="captured"):
        system._prepare_evolution(trace, SimpleNamespace(task_id="task"))

    assert "after_state_facts" not in captured
    assert captured["boundary_authorities"]["inputs"][0] == {
        "authority_ref": "runtime_input:locate_1:target",
        "draft_id": "locate_1",
        "trial_event_start": 0,
        "trial_event_end": 0,
        "kind": "current_occurrence_anchor",
        "role": "target",
        "value": "cup",
        "source_kind": "current_occurrence_anchor",
        "source_occurrence_id": "occ",
        "source_role": "object",
    }
    assert captured["boundary_authorities"]["effects"][0] == {
        "witness_ref": (
            "alfworld_action_fact:r4:entity.discovered_at:"
            "entity=cup_3,location=countertop_2"
        ),
        "predicate": "entity.discovered_at",
        "args": {"entity": "cup_3", "location": "countertop_2"},
        "effect_domain": "evidence",
        "source_kind": "runtime_trial_r1",
        "draft_id": "locate_1",
        "event_index": 0,
        "revision": 4,
    }


def test_runtime_trial_effect_refs_are_not_cross_paired_between_effects() -> None:
    trial = {
        "draft_id": "multi_effect",
        "trial_bindings": {},
        "r1_outputs": {
            "entity": "cup_3",
            "location": "countertop_2",
            "destination": "kitchen_1",
        },
        "r1_witness_refs": [
            "alfworld_action_fact:r8:entity.discovered_at:"
            "entity=cup_3,location=countertop_2",
            "alfworld_action_fact:r8:agent.at_location:location=kitchen_1",
            # A final-validation ref is real but does not identify either
            # concrete predicate+binding fact and must not be projected.
            "alfworld_action_fact:r8:effect_0",
        ],
        "declared_effects": [
            {
                "predicate": "entity.discovered_at",
                "args": {"entity": "$entity", "location": "$location"},
                "effect_domain": "evidence",
            },
            {
                "predicate": "agent.at_location",
                "args": {"location": "$destination"},
                "effect_domain": "world",
            },
        ],
        "output_derivations": {
            "entity": {
                "kind": "effect_witness",
                "predicate": "entity.discovered_at",
                "argument_role": "entity",
            },
            "location": {
                "kind": "effect_witness",
                "predicate": "entity.discovered_at",
                "argument_role": "location",
            },
            "destination": {
                "kind": "effect_witness",
                "predicate": "agent.at_location",
                "argument_role": "location",
            },
        },
        "after_revision": 8,
        "trial_event_start": 0,
        "trial_event_end": 1,
        "r1_effect_event_authorities": [
            {
                "predicate": "entity.discovered_at",
                "args": {
                    "entity": "cup_3", "location": "countertop_2",
                },
                "effect_domain": "evidence",
                "event_index": 0,
                "revision": 7,
                "source_kind": "occurrence_action_delta",
                "source_occurrence_id": "occ",
            },
            {
                "predicate": "agent.at_location",
                "args": {"location": "kitchen_1"},
                "effect_domain": "world",
                "event_index": 1,
                "revision": 8,
                "source_kind": "occurrence_action_delta",
                "source_occurrence_id": "occ",
            },
        ],
        "r1": {"admission_eligible": True},
    }
    actions = [
        {
            "event_index": 0,
            "accepted": True,
            "after_revision": 7,
            "authoritative_positive_effects": [],
        },
        {
            "event_index": 1,
            "accepted": True,
            "after_revision": 8,
            "authoritative_positive_effects": [],
        },
    ]

    facts = AtomicSkillGraphSystem._runtime_trial_effect_authorities(
        trial, actions,
    )

    assert len(facts) == 2
    by_predicate = {fact["predicate"]: fact for fact in facts}
    assert by_predicate["entity.discovered_at"]["witness_ref"].endswith(
        "entity.discovered_at:entity=cup_3,location=countertop_2"
    )
    assert by_predicate["agent.at_location"]["witness_ref"].endswith(
        "agent.at_location:location=kitchen_1"
    )


def test_terminal_interrupted_runtime_prefix_remains_e1_effect_authority() -> None:
    trial = {
        "draft_id": "terminal_prefix",
        "trial_bindings": {},
        "r1_outputs": {"object": "apple_1"},
        "r1_witness_refs": [
            "alfworld_action_fact:r3:agent.holds:object=apple_1"
        ],
        "declared_effects": [{
            "predicate": "agent.holds",
            "args": {"object": "$object"},
            "effect_domain": "world",
        }],
        # Terminal-prefix world effects can be input-bound and need no fresh
        # output derivation; their exact validator ref remains the authority.
        "output_derivations": {},
        "after_revision": 3,
        "trial_event_start": 0,
        "trial_event_end": 0,
        "r1_effect_event_authorities": [{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
            "effect_domain": "world",
            "event_index": 0,
            "revision": 3,
            "source_kind": "occurrence_action_delta",
            "source_occurrence_id": "occ",
        }],
        "result": {"started": True, "tool_results": [{
            "intrinsic_failure": False,
        }]},
        "r1": {
            "started": True,
            "atomic_effect_passed": True,
            "executed_path_effects_passed": True,
            "tool_completed": False,
            "terminal_interrupted": True,
            "outputs_valid": True,
            "tool_intrinsic_failure": False,
            "admission_eligible": False,
        },
    }
    actions = [{
        "event_index": 0,
        "accepted": True,
        "after_revision": 3,
        "authoritative_positive_effects": [{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
            "effect_domain": "world",
        }],
    }]

    facts = AtomicSkillGraphSystem._runtime_trial_effect_authorities(
        trial, actions,
    )

    assert trial["r1"]["admission_eligible"] is False
    assert facts[0]["event_index"] == 0
    assert facts[0]["witness_ref"] == (
        "alfworld_action_fact:r3:agent.holds:object=apple_1"
    )
