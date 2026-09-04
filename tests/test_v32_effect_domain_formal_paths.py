from __future__ import annotations

from types import SimpleNamespace

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.evolution.contract_canonicalizer import (
    atomic_contract_signature,
    canonical_atomic_contract,
)
from atomic_skillgraph.evolution.failure_extraction_validator import (
    _atomic_occurrence_proposal,
)
from atomic_skillgraph.evolution.failure_extractor_session import (
    FailureAtomicProposal,
)
from atomic_skillgraph.evolution.gap_diagnosis import GapDiagnoser
from atomic_skillgraph.evolution.provisional_promotion import (
    ProvisionalPromotionCompiler,
)
from atomic_skillgraph.evolution.typed_repairs import TypedRepairEngine
from atomic_skillgraph.knowledge.query import (
    atomic_contract_compatible,
    complete_composite_contract_diagnosis,
)
from atomic_skillgraph.harness.alfworld import AlfWorldContractMatcher
from atomic_skillgraph.planner.cold_start_retriever import (
    provisional_contract_compatible,
    task_cluster_signature,
)
from atomic_skillgraph.planner.multiplicity import (
    _predicate_shape_matches,
    canonical_requirement_shape,
)
from atomic_skillgraph.planner.requirement_agent import requirement_bundle_from_dict
from atomic_skillgraph.planner.validator import _predicate_shape_compatible
from atomic_skillgraph.runtime.cold_start_executor import (
    ProvisionalTrialResult,
    provisional_atomic_view,
)
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher


def _predicate_payload(*, domain: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "predicate": "entity.discovered_at",
        "args": {"entity": "$target", "location": "$location"},
        "cardinality": 1,
        "distinct_by": "",
    }
    if domain is not None:
        payload["effect_domain"] = domain
    return payload


def _atomic(domain: EffectDomain | str = EffectDomain.WORLD) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef("locate_entity", "1.0.0"),
        summary="locate entity",
        inputs=[ParameterSpec("target", "object")],
        outputs=[ParameterSpec("location", "location")],
        preconditions=[],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": "$target", "location": "$location"},
            effect_domain=domain,
        )],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
    )


def _requirement(domain: EffectDomain | str) -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id="locate",
        intent="locate entity",
        desired_effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": "$target", "location": "$location"},
            effect_domain=domain,
        )],
        expected_inputs=[ParameterSpec("target", "object")],
        expected_outputs=[ParameterSpec("location", "location")],
        precondition_hints=[],
        semantic_variants=[],
        required=True,
        rationale="grounding is required",
    )


def test_mapping_parsers_preserve_explicit_domain_and_default_to_world(
    monkeypatch,
) -> None:
    raw_requirement = {
        "requirement_id": "locate",
        "intent": "locate entity",
        "desired_effects": [_predicate_payload(domain="evidence")],
        "expected_inputs": [{"name": "target", "semantic_type": "object"}],
        "expected_outputs": [{"name": "location", "semantic_type": "location"}],
        "precondition_hints": [_predicate_payload(domain=None)],
        "semantic_variants": [],
        "required": True,
        "rationale": "grounding is required",
    }
    bundle = requirement_bundle_from_dict({
        "requirements": [raw_requirement],
        "repeat_blocks": [],
    })
    assert bundle.requirements[0].desired_effects[0].effect_domain is EffectDomain.EVIDENCE
    assert bundle.requirements[0].precondition_hints[0].effect_domain is EffectDomain.WORLD

    contract = {
        "inputs": [{"name": "target", "semantic_type": "object"}],
        "outputs": [{"name": "location", "semantic_type": "location"}],
        "preconditions": [_predicate_payload(domain=None)],
        "effects": [_predicate_payload(domain="evidence")],
        "validator_spec": {},
    }
    provisional = SimpleNamespace(
        provisional_ref="provisional://locate",
        canonical_intent="locate_entity",
        atomic_contract=contract,
        seeded_guideline={},
        harness_profile="fake",
    )
    cold_view = provisional_atomic_view(provisional)
    assert cold_view.preconditions[0].effect_domain is EffectDomain.WORLD
    assert cold_view.effects[0].effect_domain is EffectDomain.EVIDENCE

    failure_proposal = _atomic_occurrence_proposal(FailureAtomicProposal(
        atomic_proposal={
            "phase_id": "locate",
            "intent": "locate entity",
            "event_start": 0,
            "event_end": 1,
            "input_roles": {"target": "cup_1"},
            "output_roles": {"location": "cabinet_1"},
            "preconditions": [_predicate_payload(domain=None)],
            "effects": [_predicate_payload(domain="evidence")],
            "rationale": "accepted observation",
        },
        aligned_plan_step_ids=["locate"],
        progress_relation="consumed_prerequisite",
    ))
    assert failure_proposal.preconditions[0].effect_domain is EffectDomain.WORLD
    assert failure_proposal.effects[0].effect_domain is EffectDomain.EVIDENCE

    observed_domains: list[EffectDomain] = []

    def compatible(requirement, _atomic_value):
        observed_domains.append(requirement.desired_effects[0].effect_domain)
        return True

    monkeypatch.setattr(
        "atomic_skillgraph.evolution.gap_diagnosis.atomic_contract_compatible",
        compatible,
    )
    diagnosis = GapDiagnoser(SimpleNamespace(atomics=lambda: [])).diagnose(
        SimpleNamespace(
            strict_task_success=True,
            runtime_plan={"source": "full_dynamic"},
            planner_audit={"atomic_search_p1r": [{
                "covered": False,
                "requirement": raw_requirement,
                "candidates": [],
            }]},
            trace_id="trace-1",
        ),
        [_atomic(EffectDomain.EVIDENCE)],
    )
    assert diagnosis["classification"] == "confirmed_capability_gap"
    assert observed_domains == [EffectDomain.EVIDENCE]


def test_promotion_and_typed_repair_rebuilds_preserve_domain() -> None:
    provisional = SimpleNamespace(
        canonical_intent="locate_entity",
        atomic_contract={
            "inputs": [{"name": "target", "semantic_type": "object"}],
            "outputs": [{"name": "location", "semantic_type": "location"}],
            "preconditions": [_predicate_payload(domain=None)],
            "effects": [_predicate_payload(domain="evidence")],
            "validator_spec": {},
        },
    )
    trial = ProvisionalTrialResult(
        provisional_ref="provisional://locate",
        step_id="locate",
        local_effect_passed=True,
        progress_before_digest="before",
        progress_after_digest="after",
        action_span=(0, 1),
        witness_refs=["witness:1"],
        failure_code="",
        resolved_bindings={"target": "cup_1", "location": "cabinet_1"},
    )
    promotion = ProvisionalPromotionCompiler._proposal(provisional, trial)
    assert promotion.preconditions[0].effect_domain is EffectDomain.WORLD
    assert promotion.effects[0].effect_domain is EffectDomain.EVIDENCE

    replacement = {
        "ref": {"logical_id": "locate_entity_revision", "version": "1.0.0"},
        "summary": "locate entity",
        "inputs": [{"name": "target", "semantic_type": "object"}],
        "outputs": [{"name": "location", "semantic_type": "location"}],
        "preconditions": [_predicate_payload(domain=None)],
        "effects": [_predicate_payload(domain="evidence")],
        "validator_spec": {},
        "failure_modes": [],
        "guideline": {},
        "metadata": {},
        "status": "draft",
    }
    repair = TypedRepairEngine.build_proposal(
        "revise_atomic_contract",
        [SkillRef("locate_entity", "1.0.0")],
        [replacement],
        [],
    )
    rebuilt = repair.proposed_patch["replacements"][0]
    assert rebuilt["preconditions"][0]["effect_domain"] == "world"
    assert rebuilt["effects"][0]["effect_domain"] == "evidence"


def test_domain_is_part_of_atomic_requirement_and_task_identity() -> None:
    world = _atomic()
    explicit_world = _atomic("world")
    evidence = _atomic("evidence")
    assert atomic_contract_signature(world) == atomic_contract_signature(explicit_world)
    assert atomic_contract_signature(world) != atomic_contract_signature(evidence)
    assert canonical_atomic_contract(evidence)["effects"][0]["effect_domain"] == "evidence"

    world_requirement = _requirement(EffectDomain.WORLD)
    evidence_requirement = _requirement(EffectDomain.EVIDENCE)
    assert canonical_requirement_shape(world_requirement) != canonical_requirement_shape(
        evidence_requirement
    )
    assert task_cluster_signature(
        TaskContract(target_effects=world_requirement.desired_effects), "fake"
    ) != task_cluster_signature(
        TaskContract(target_effects=evidence_requirement.desired_effects), "fake"
    )


def test_world_and_evidence_never_cross_cover_on_formal_paths() -> None:
    world_requirement = _requirement(EffectDomain.WORLD)
    evidence_requirement = _requirement(EffectDomain.EVIDENCE)
    world_atomic = _atomic(EffectDomain.WORLD)
    evidence_atomic = _atomic(EffectDomain.EVIDENCE)

    assert atomic_contract_compatible(world_requirement, world_atomic)
    assert atomic_contract_compatible(evidence_requirement, evidence_atomic)
    assert not atomic_contract_compatible(evidence_requirement, world_atomic)
    assert not atomic_contract_compatible(world_requirement, evidence_atomic)
    assert _predicate_shape_compatible(
        evidence_requirement.desired_effects[0], evidence_atomic.effects[0]
    )
    assert not _predicate_shape_compatible(
        evidence_requirement.desired_effects[0], world_atomic.effects[0]
    )
    assert _predicate_shape_matches(
        evidence_requirement.desired_effects[0], evidence_atomic.effects[0]
    )
    assert not _predicate_shape_matches(
        evidence_requirement.desired_effects[0], world_atomic.effects[0]
    )

    offered_arguments = {"entity": "cup_1", "location": "cabinet_1"}
    target = SemanticPredicate(
        "entity.discovered_at",
        offered_arguments,
        effect_domain=EffectDomain.EVIDENCE,
    )
    offered_evidence = SemanticPredicate(
        "entity.discovered_at", {}, effect_domain=EffectDomain.EVIDENCE,
    )
    offered_world = SemanticPredicate(
        "entity.discovered_at", {}, effect_domain=EffectDomain.WORLD,
    )
    for matcher in (ExactContractMatcher(), AlfWorldContractMatcher()):
        assert matcher.covers(target, offered_evidence, offered_arguments)
        assert not matcher.covers(target, offered_world, offered_arguments)

    evidence_contract = {
        "inputs": [{"name": "target", "semantic_type": "object"}],
        "effects": [_predicate_payload(domain="evidence")],
    }
    world_contract = {
        "inputs": [{"name": "target", "semantic_type": "object"}],
        "effects": [_predicate_payload(domain=None)],
    }
    assert provisional_contract_compatible(evidence_requirement, evidence_contract)
    assert not provisional_contract_compatible(evidence_requirement, world_contract)

    world_task = TaskContract(target_effects=world_atomic.effects)
    evidence_task = TaskContract(target_effects=evidence_atomic.effects)
    assert complete_composite_contract_diagnosis(world_task, world_task).passed
    diagnosis = complete_composite_contract_diagnosis(
        evidence_task, world_task
    )
    assert not diagnosis.passed
    assert diagnosis.target_effect_missing
    assert diagnosis.target_effect_extra
