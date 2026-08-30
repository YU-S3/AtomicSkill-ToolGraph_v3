"""P1 and the single permitted P1R turn in one PlannerSession."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..agents.structured_submission import (
    CAPABILITY_REQUIREMENT_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.contracts import CapabilityRequirement, ParameterSpec, SemanticPredicate, TaskContract
from ..core.errors import AgentProtocolError, FailureLayer, PlannerProposalError
from ..core.serialization import to_primitive


REQUIREMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["requirements"],
    "additionalProperties": False,
    "properties": {
        "requirements": {
            "type": "array",
            "minItems": 1,
            "items": CAPABILITY_REQUIREMENT_SCHEMA,
        }
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


class RequirementAgent:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()

    def propose(
        self, task: Any, contract: TaskContract, observation: str,
        harness_profile: str,
    ) -> list[CapabilityRequirement]:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p1")
        prompt = (
            "P1 CAPABILITY REQUIREMENTS. Decompose the task into complete, reusable state-transition "
            "capabilities. Do not prescribe environment actions or invent skills. Call only the "
            "offered submit tool.\n"
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
        self, task: Any, contract: TaskContract, requirements: list[CapabilityRequirement],
        search: list[Any], related_composites: list[dict[str, Any]],
    ) -> list[CapabilityRequirement]:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p1_repair")
        prompt = (
            "P1R REQUIREMENT REPAIR. Coverage is incomplete. Return one complete replacement requirement "
            "list, not a patch. Composite material is only a hint and is not an oracle. Call only "
            "the offered submit tool.\n"
            f"Task: {task.goal}\nTaskContract: {json.dumps(to_primitive(contract), ensure_ascii=False)}\n"
            f"requirements_v1: {json.dumps(to_primitive(requirements), ensure_ascii=False)}\n"
            f"search/rejections: {json.dumps(to_primitive(search), ensure_ascii=False)}\n"
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
    ) -> list[CapabilityRequirement]:
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
            return [
                _requirement(item)
                for item in submission.value["requirements"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise PlannerProposalError(
                error_code,
                f"Planner requirement proposal semantic validation failed: {exc}",
                layer=FailureLayer.PLANNER_REQUIREMENT,
            ) from exc
