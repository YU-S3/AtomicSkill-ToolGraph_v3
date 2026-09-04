"""Bounded P2/P2R policy projections must retain formal Planner authority."""

from __future__ import annotations

import json
from types import SimpleNamespace

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    AtomicCandidate,
    CapabilityRequirement,
    ContractSource,
    EffectDomain,
    ParameterSpec,
    PlannerWorkflowProposal,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import ValidationResult
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.planner.multiplicity import (
    RequirementExpansion,
    RequirementInstance,
)
from atomic_skillgraph.planner.support_retriever import (
    PlannerSupportCandidate,
    PlannerSupportRoleMapping,
)
from atomic_skillgraph.planner.workflow_agent import WorkflowAgent


class _BucketSession:
    def __init__(self) -> None:
        self.bucket = ""

    def set_usage_bucket(self, bucket: str) -> None:
        self.bucket = bucket


def _capture_prompt(agent: WorkflowAgent) -> dict[str, str]:
    captured: dict[str, str] = {}

    def capture(prompt: str, **_kwargs):
        captured["prompt"] = prompt
        return "captured"

    agent._request_workflow = capture  # type: ignore[method-assign]
    return captured


def _line_payload(prompt: str, label: str):
    line = next(item for item in prompt.splitlines() if item.startswith(label))
    return json.loads(line[len(label):])


def _requirement() -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id="locate",
        intent="locate one compatible entity",
        desired_effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "$location"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
        ],
        expected_inputs=[
            ParameterSpec(
                "target",
                "entity_type",
                required=True,
                runtime_resolvable=True,
                required_resolution="semantic",
                description="semantic target supplied by the task",
            ),
        ],
        expected_outputs=[
            ParameterSpec(
                "entity",
                "entity",
                required=True,
                runtime_resolvable=True,
                required_resolution="relation_verified",
                description="fresh concrete entity",
            ),
        ],
        precondition_hints=[],
        semantic_variants=["find target"],
        required=True,
        rationale="formal reusable evidence transition",
    )


def _expansion(count: int = 4) -> RequirementExpansion:
    requirement = _requirement()
    instances = tuple(
        RequirementInstance(
            instance_id=f"repeat::locate::{index}",
            template_requirement_id="locate",
            repeat_block_id="",
            repeat_index=index,
            requirement=requirement,
        )
        for index in range(count)
    )
    return RequirementExpansion(
        templates=(requirement,),
        repeat_blocks=(),
        instances=instances,
        instance_ids_by_template={
            "locate": tuple(item.instance_id for item in instances),
        },
    )


def _support_candidate() -> PlannerSupportCandidate:
    return PlannerSupportCandidate(
        atomic_ref="skill://atomic_support@1.0.0",
        consumer_requirement_instance_id="repeat::locate::0",
        score=2.0,
        role_mappings=(
            PlannerSupportRoleMapping(
                producer_role="entity",
                consumer_role="object",
                semantic_type="entity",
                producer_resolution="relation_verified",
                required_resolution="relation_verified",
                effect_domain="evidence",
                consumer_atomic_ref="skill://atomic_consumer@1.0.0",
            ),
        ),
        output_roles=("entity", "location"),
        effect_predicates=("entity.discovered_at",),
    )


def test_p2_projection_deduplicates_instances_without_changing_candidates() -> None:
    expansion = _expansion()
    required_candidates = {
        item.instance_id: [
            AtomicCandidate(
                SkillRef("atomic_consumer", "1.0.0"),
                0.75,
                reasons=["contract_compatible"],
                contract_match=True,
            ),
        ]
        for item in expansion.instances
    }
    support_candidates = [_support_candidate(), _support_candidate()]
    required_atomic = AbstractAtomicSkill(
        ref=SkillRef("atomic_consumer", "1.0.0"),
        summary="consume one located entity",
        inputs=[
            ParameterSpec(
                "object",
                "entity",
                required_resolution="relation_verified",
            ),
        ],
        outputs=[],
        preconditions=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$object", "location": "$source"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
        ],
        effects=[SemanticPredicate("object.heated", {"object": "$object"})],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )
    support_atomic = AbstractAtomicSkill(
        ref=SkillRef("atomic_support", "1.0.0"),
        summary="locate one entity",
        inputs=[ParameterSpec("target", "entity_type")],
        outputs=[
            ParameterSpec(
                "entity",
                "entity",
                required_resolution="relation_verified",
            ),
            ParameterSpec("location", "location"),
        ],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "$location"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
        ],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )
    session = _BucketSession()
    agent = WorkflowAgent(session)
    captured = _capture_prompt(agent)

    result = agent.propose(
        SimpleNamespace(goal="locate targets"),
        TaskContract(source=ContractSource.ADAPTER_DERIVED),
        expansion,
        required_candidates,
        [],
        [],
        support_candidates=support_candidates,
        authoritative_contracts=[required_atomic, support_atomic],
    )

    assert result == "captured"
    assert session.bucket == "planner_p2"
    prompt = captured["prompt"]
    requirements = _line_payload(prompt, "Requirements: ")
    assert requirements["templates"] == to_primitive(list(expansion.templates))
    assert len(requirements["instances"]) == len(expansion.instances)
    assert all("requirement" not in item for item in requirements["instances"])
    assert len(json.dumps(requirements)) < (
        len(json.dumps(to_primitive(expansion))) * 0.55
    )

    candidates = _line_payload(prompt, "Required candidates: ")
    assert sum(len(items) for items in candidates.values()) == len(
        expansion.instances
    )
    assert all(
        item["contract_match"] is True
        and item["reasons"] == ["contract_compatible"]
        for items in candidates.values()
        for item in items
    )

    support = _line_payload(
        prompt,
        "Support candidates and formal mappings: ",
    )
    assert len(support) == len(support_candidates)
    assert support == to_primitive(support_candidates)

    interfaces = _line_payload(
        prompt,
        "Authoritative candidate interfaces: ",
    )
    assert [item["ref"] for item in interfaces] == [
        str(required_atomic.ref),
        str(support_atomic.ref),
    ]
    assert interfaces == [
        {
            "ref": str(atomic.ref),
            "summary": atomic.summary,
            "inputs": [
                {
                    "name": item.name,
                    "semantic_type": item.semantic_type,
                    "required": item.required,
                    "runtime_resolvable": item.runtime_resolvable,
                    "required_resolution": item.required_resolution,
                }
                for item in atomic.inputs
            ],
            "outputs": [
                {
                    "name": item.name,
                    "semantic_type": item.semantic_type,
                    "required": item.required,
                    "runtime_resolvable": item.runtime_resolvable,
                    "required_resolution": item.required_resolution,
                }
                for item in atomic.outputs
            ],
            "preconditions": to_primitive(atomic.preconditions),
            "effects": to_primitive(atomic.effects),
        }
        for atomic in (required_atomic, support_atomic)
    ]
    # The authoritative interface projection supplements, but never mutates,
    # the ranked pools or their formal support mappings.
    assert _line_payload(prompt, "Required candidates: ") == candidates
    assert _line_payload(
        prompt,
        "Support candidates and formal mappings: ",
    ) == support


def test_p2r_projection_keeps_contract_and_drops_non_planner_bulk() -> None:
    marker = "NON_PLANNER_BULK_MARKER"
    atomic = AbstractAtomicSkill(
        ref=SkillRef("atomic_support", "1.0.0"),
        summary="locate a target",
        inputs=[
            ParameterSpec(
                "target",
                "entity_type",
                required_resolution="semantic",
                description=marker * 100,
            ),
        ],
        outputs=[
            ParameterSpec(
                "entity",
                "entity",
                required_resolution="relation_verified",
                description=marker * 100,
            ),
        ],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "$location"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
        ],
        validator_spec={"bulk": marker * 100},
        failure_modes=[{"bulk": marker * 100}],
        guideline={"bulk": marker * 100},
        metadata={"bulk": marker * 100},
        status=SkillStatus.ACTIVE,
    )
    validation = ValidationResult(
        level="planner_graph",
        passed=False,
        checks={"support_data_flow_mappings_valid": False},
        failure_codes=["planner_support_atomic_invalid"],
        messages=["support output must feed the required occurrence"],
        witness_refs=[marker * 100],
        before_ref=marker * 100,
        after_ref=marker * 100,
    )
    session = _BucketSession()
    agent = WorkflowAgent(session)
    captured = _capture_prompt(agent)

    result = agent.repair(
        PlannerWorkflowProposal([], [], [], [], {}),
        validation,
        [atomic],
        [],
        support_candidates=[_support_candidate()],
    )

    assert result == "captured"
    assert session.bucket == "planner_p2_repair"
    prompt = captured["prompt"]
    assert marker not in prompt
    contracts = _line_payload(prompt, "Authoritative contracts: ")
    assert contracts == [{
        "ref": str(atomic.ref),
        "summary": atomic.summary,
        "inputs": [{
            "name": "target",
            "semantic_type": "entity_type",
            "required": True,
            "runtime_resolvable": False,
            "required_resolution": "semantic",
        }],
        "outputs": [{
            "name": "entity",
            "semantic_type": "entity",
            "required": True,
            "runtime_resolvable": False,
            "required_resolution": "relation_verified",
        }],
        "preconditions": [],
        "effects": to_primitive(atomic.effects),
    }]
    findings = _line_payload(prompt, "Validation errors: ")
    assert findings == {
        "level": validation.level,
        "passed": False,
        "checks": validation.checks,
        "failure_codes": validation.failure_codes,
        "messages": validation.messages,
    }
    assert _line_payload(
        prompt,
        "Support candidates and formal mappings: ",
    ) == to_primitive([_support_candidate()])
