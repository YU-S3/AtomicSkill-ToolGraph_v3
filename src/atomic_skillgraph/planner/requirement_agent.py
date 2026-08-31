"""P1 and the single permitted P1R turn in one PlannerSession."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..agents.structured_submission import (
    CAPABILITY_REQUIREMENT_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.contracts import (
    CapabilityRequirement,
    ParameterSpec,
    PlannerRequirementBundle,
    RepeatBlock,
    SemanticPredicate,
    TaskContract,
)
from ..core.errors import AgentProtocolError, FailureLayer, PlannerProposalError
from ..core.serialization import to_primitive


REQUIREMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["requirements", "repeat_blocks"],
    "additionalProperties": False,
    "properties": {
        "requirements": {
            "type": "array",
            "minItems": 1,
            "items": CAPABILITY_REQUIREMENT_SCHEMA,
        },
        "repeat_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "block_id", "count", "ordered_requirement_ids",
                    "distinct_roles", "shared_roles", "basis_constraint_id",
                    "basis_role_map", "execution_policy", "rationale",
                ],
                "additionalProperties": False,
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "count": {"type": "integer", "minimum": 2, "maximum": 4},
                    "ordered_requirement_ids": {
                        "type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "distinct_roles": {
                        "type": "array", "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "shared_roles": {
                        "type": "array", "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "basis_constraint_id": {"type": "string", "minLength": 1},
                    "basis_role_map": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "minLength": 1},
                    },
                    "execution_policy": {"type": "string", "enum": ["serial"]},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def _predicate(value: dict[str, Any]) -> SemanticPredicate:
    return SemanticPredicate(
        predicate=str(value["predicate"]), args=dict(value.get("args", {})),
        cardinality=int(value.get("cardinality", 1)), distinct_by=str(value.get("distinct_by", "")),
    )


def _requirement(value: dict[str, Any]) -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id=str(value["requirement_id"]), intent=str(value["intent"]),
        desired_effects=[_predicate(item) for item in value.get("desired_effects", [])],
        expected_inputs=[ParameterSpec(**item) for item in value.get("expected_inputs", [])],
        expected_outputs=[ParameterSpec(**item) for item in value.get("expected_outputs", [])],
        precondition_hints=[_predicate(item) for item in value.get("precondition_hints", [])],
        semantic_variants=[str(item) for item in value.get("semantic_variants", [])],
        required=bool(value.get("required", True)), rationale=str(value.get("rationale", "")),
    )


def requirement_bundle_from_dict(value: dict[str, Any]) -> PlannerRequirementBundle:
    """Reconstruct the frozen P1 bundle from its structured Trace form."""

    return PlannerRequirementBundle(
        requirements=[
            _requirement(item) for item in value["requirements"]
        ],
        repeat_blocks=[
            RepeatBlock(
                block_id=str(item["block_id"]),
                count=int(item["count"]),
                ordered_requirement_ids=tuple(map(
                    str, item["ordered_requirement_ids"],
                )),
                distinct_roles=tuple(map(str, item["distinct_roles"])),
                shared_roles=tuple(map(str, item["shared_roles"])),
                basis_constraint_id=str(item["basis_constraint_id"]),
                basis_role_map={
                    str(key): str(raw)
                    for key, raw in item["basis_role_map"].items()
                },
                execution_policy=str(item["execution_policy"]),
                rationale=str(item.get("rationale", "")),
            )
            for item in value["repeat_blocks"]
        ],
    )


class RequirementAgent:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()

    def propose(
        self, task: Any, contract: TaskContract, observation: str,
        harness_profile: str,
    ) -> PlannerRequirementBundle:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p1")
        prompt = (
            "P1 CAPABILITY REQUIREMENTS. Decompose the task into complete, reusable state-transition "
            "capabilities. Do not prescribe environment actions or invent skills. Call only the "
            "offered submit tool.\n"
            "Represent reusable unit capabilities, not aggregate monolithic skills.\n\n"
            "When the TaskContract requires the same unit effect for multiple distinct\n"
            "identities, return one unit capability requirement and a RepeatBlock.\n"
            "A RepeatBlock may contain one requirement or an ordered small group of\n"
            "requirements. Do not duplicate the requirement definitions themselves.\n\n"
            "Repeat count, distinct roles, and shared roles must be grounded in the supplied\n"
            "TaskContract cardinality and identity constraints. Do not infer a repeat count\n"
            "from task wording when no formal contract constraint supports it.\n\n"
            "Every TaskContract cardinality constraint whose composition_mode is repeat_unit\n"
            "must be represented by exactly one RepeatBlock. Do not satisfy a repeat_unit\n"
            "constraint by giving one capability requirement an aggregate effect cardinality\n"
            "greater than one.\n\n"
            "Each member capability inside a RepeatBlock represents one reusable unit\n"
            "transition. Its desired effect for the repeat basis must have cardinality 1;\n"
            "the RepeatBlock count provides the task-level multiplicity.\n\n"
            "The block order describes one reusable iteration. P2 will instantiate the\n"
            "block multiple times and may reuse the same Atomic ref in different steps.\n"
            f"Task goal: {task.goal}\nPolicy observation: {observation}\n"
            f"TaskContract: {json.dumps(to_primitive(contract), ensure_ascii=False)}\n"
            f"Harness profile: {harness_profile}"
        )
        return self._request_requirements(
            prompt,
            error_code="planner_requirement_invalid",
            description="Submit the complete capability requirement proposal.",
        )

    def repair(
        self, task: Any, contract: TaskContract, requirements: PlannerRequirementBundle,
        search: list[Any], related_composites: list[dict[str, Any]],
        validation: Any | None = None,
    ) -> PlannerRequirementBundle:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p1_repair")
        prompt = (
            "P1R REQUIREMENT REPAIR. Coverage is incomplete. Return one complete replacement requirement "
            "bundle (requirements and repeat_blocks), not a patch. Composite material is only a hint and is not an oracle. "
            "Represent reusable unit capabilities and ground every repetition in the TaskContract. Call only "
            "the offered submit tool.\n"
            "The replacement bundle must materialize every formal repeat_unit constraint "
            "exactly once. Do not remove a required RepeatBlock merely to make aggregate "
            "coverage pass. Every member desired effect for the repeat basis must retain "
            "unit cardinality 1.\n"
            f"Task: {task.goal}\nTaskContract: {json.dumps(to_primitive(contract), ensure_ascii=False)}\n"
            f"requirements_v1: {json.dumps(to_primitive(requirements), ensure_ascii=False)}\n"
            f"search/rejections: {json.dumps(to_primitive(search), ensure_ascii=False)}\n"
            f"bundle validation: {json.dumps(to_primitive(validation), ensure_ascii=False)}\n"
            f"related hints: {json.dumps(related_composites, ensure_ascii=False)}"
        )
        return self._request_requirements(
            prompt,
            error_code="planner_requirement_repair_failed",
            description="Submit the complete replacement capability requirement proposal.",
        )

    def _request_requirements(
        self,
        prompt: str,
        *,
        error_code: str,
        description: str,
    ) -> PlannerRequirementBundle:
        try:
            submission = self.submissions.request(
                self.session,
                prompt=prompt,
                tool_name="submit_planner_requirements",
                description=description,
                schema=REQUIREMENT_SCHEMA,
            )
        except AgentProtocolError as exc:
            raise PlannerProposalError(
                error_code,
                f"Planner requirement proposal protocol validation failed: {exc}",
                layer=FailureLayer.PLANNER_REQUIREMENT,
            ) from exc
        try:
            return requirement_bundle_from_dict(submission.value)
        except (KeyError, TypeError, ValueError) as exc:
            raise PlannerProposalError(
                error_code,
                f"Planner requirement proposal semantic validation failed: {exc}",
                layer=FailureLayer.PLANNER_REQUIREMENT,
            ) from exc
