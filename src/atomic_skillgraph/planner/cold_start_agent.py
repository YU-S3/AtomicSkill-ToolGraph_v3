"""C1/C1R high-level cold-start planning over frozen requirements."""

from __future__ import annotations

import json
from typing import Any

from ..agents.structured_submission import (
    BINDING_EXPRESSION_SCHEMA,
    PROPOSED_EDGE_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.bindings import BindingExpression
from ..core.contracts import (
    ColdStartCandidateSource,
    ColdStartExecutionMode,
    ColdStartPlanProposal,
    ColdStartPlanStep,
    ProposedEdge,
)
from ..core.errors import AgentProtocolError, FailureLayer, PlannerProposalError
from ..core.serialization import to_primitive


_NONEMPTY = {"type": "string", "minLength": 1}

COLD_START_PLAN_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "step_id", "requirement_instance_ids", "candidate_source",
        "candidate_ref", "execution_mode", "binding_specs",
        "repeat_role_bindings",
    ],
    "additionalProperties": False,
    "properties": {
        "step_id": _NONEMPTY,
        "requirement_instance_ids": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": _NONEMPTY,
        },
        "candidate_source": {
            "type": "string",
            "enum": ["verified", "provisional", "unresolved"],
        },
        "candidate_ref": {"type": "string"},
        "execution_mode": {
            "type": "string",
            "enum": ["direct_or_seeded", "seeded_only", "dynamic"],
        },
        "binding_specs": {
            "type": "object",
            "additionalProperties": BINDING_EXPRESSION_SCHEMA,
        },
        "repeat_role_bindings": {
            "type": "object", "additionalProperties": _NONEMPTY,
        },
    },
}

COLD_START_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "plan_id", "steps", "control_sequence", "data_edges",
        "dependency_edges", "requirement_coverage",
        "referenced_failure_experience_ids",
    ],
    "additionalProperties": False,
    "properties": {
        "plan_id": _NONEMPTY,
        "steps": {
            "type": "array", "minItems": 1,
            "items": COLD_START_PLAN_STEP_SCHEMA,
        },
        "control_sequence": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": _NONEMPTY,
        },
        "data_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "dependency_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "requirement_coverage": {
            "type": "object",
            "additionalProperties": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": _NONEMPTY,
            },
        },
        "referenced_failure_experience_ids": {
            "type": "array", "uniqueItems": True, "items": _NONEMPTY,
        },
    },
}


COLD_START_PROMPT = """Construct one complete high-level cold-start plan over the supplied frozen
RequirementInstances.

This is not a verified Runtime Graph. Some steps may be unresolved.
Use only retrieved Verified or Provisional candidates and supplied negative
Failure Experiences.

Verified candidates may be marked direct_or_seeded.
Provisional candidates are locally validated but originate from failed tasks;
they must be marked seeded_only and must never be treated as successful plans
or executable learned tools.
Uncovered steps must be marked unresolved/dynamic.

Failure Experiences are negative examples. Do not copy their failed method;
use them only to avoid a previously observed unrecovered divergence.

Do not restate or invent candidate Effects. Candidate Atomic contracts are
code-authoritative and Runtime will derive executable expected effects directly
from the selected candidate.

Cover every RequirementInstance exactly once in the high-level plan. Preserve
RepeatBlock serial order, distinct-role requirements, and shared-role
requirements. The same candidate ref may appear in different unique steps.

Do not invent requirements, candidates, ToolCalls, concrete entity locations,
or existing edges. Call only the offered native submission tool."""


def cold_start_plan_from_dict(value: dict[str, Any]) -> ColdStartPlanProposal:
    """Reconstruct one validated C1 proposal from its structured Trace form."""

    steps = [
        ColdStartPlanStep(
            step_id=str(item["step_id"]),
            requirement_instance_ids=list(map(str, item["requirement_instance_ids"])),
            candidate_source=ColdStartCandidateSource(item["candidate_source"]),
            candidate_ref=str(item.get("candidate_ref", "")),
            execution_mode=ColdStartExecutionMode(item["execution_mode"]),
            binding_specs={
                str(key): BindingExpression.from_dict(raw)
                for key, raw in item.get("binding_specs", {}).items()
            },
            repeat_role_bindings={
                str(key): str(raw)
                for key, raw in item.get("repeat_role_bindings", {}).items()
            },
            # Internal compatibility projection only.  C1 cannot author
            # executable Effect authority; Runtime derives it from the
            # admitted candidate Atomic contract.
            expected_effects=[],
        )
        for item in value.get("steps", [])
    ]
    return ColdStartPlanProposal(
        plan_id=str(value["plan_id"]),
        steps=steps,
        control_sequence=list(map(str, value.get("control_sequence", []))),
        data_edges=[ProposedEdge(**item) for item in value.get("data_edges", [])],
        dependency_edges=[ProposedEdge(**item) for item in value.get("dependency_edges", [])],
        requirement_coverage={
            str(key): list(map(str, raw))
            for key, raw in value.get("requirement_coverage", {}).items()
        },
        referenced_failure_experience_ids=list(map(
            str, value.get("referenced_failure_experience_ids", []),
        )),
    )


class ColdStartPlanner:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()

    def propose(
        self,
        *,
        task: Any,
        task_contract: Any,
        requirement_expansion: Any,
        verified_candidates: Any,
        provisional_candidates: Any,
        failure_experiences: Any,
        observation: str,
    ) -> ColdStartPlanProposal:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("cold_start_c1")
        prompt = (
            COLD_START_PROMPT
            + "\n\n"
            + f"Task goal: {task.goal}\n"
            + f"Current observation: {observation}\n"
            + f"TaskContract: {json.dumps(to_primitive(task_contract), ensure_ascii=False)}\n"
            + f"RequirementExpansion: {json.dumps(to_primitive(requirement_expansion), ensure_ascii=False)}\n"
            + f"Verified candidates: {json.dumps(to_primitive(verified_candidates), ensure_ascii=False)}\n"
            + f"Provisional candidates: {json.dumps(to_primitive(provisional_candidates), ensure_ascii=False)}\n"
            + f"Failure Experiences (negative only): {json.dumps(to_primitive(failure_experiences), ensure_ascii=False)}"
        )
        return self._request(prompt)

    def repair(
        self,
        proposal: ColdStartPlanProposal,
        validation: Any,
        *,
        requirement_expansion: Any,
        verified_candidates: Any,
        provisional_candidates: Any,
        failure_experiences: Any,
    ) -> ColdStartPlanProposal:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("cold_start_c1_repair")
        prompt = (
            "C1R COLD-START PLAN REPAIR. Return one complete replacement plan, not a patch. "
            "Keep the frozen RequirementInstances and use only the supplied candidates and negative experiences.\n"
            f"Rejected plan: {json.dumps(to_primitive(proposal), ensure_ascii=False)}\n"
            f"Validation: {json.dumps(to_primitive(validation), ensure_ascii=False)}\n"
            f"RequirementExpansion: {json.dumps(to_primitive(requirement_expansion), ensure_ascii=False)}\n"
            f"Verified candidates: {json.dumps(to_primitive(verified_candidates), ensure_ascii=False)}\n"
            f"Provisional candidates: {json.dumps(to_primitive(provisional_candidates), ensure_ascii=False)}\n"
            f"Failure Experiences: {json.dumps(to_primitive(failure_experiences), ensure_ascii=False)}"
        )
        return self._request(prompt)

    def _request(self, prompt: str) -> ColdStartPlanProposal:
        try:
            submission = self.submissions.request(
                self.session,
                prompt=prompt,
                tool_name="submit_cold_start_plan",
                description="Submit one complete high-level cold-start plan.",
                schema=COLD_START_PLAN_SCHEMA,
            )
        except AgentProtocolError as exc:
            raise PlannerProposalError(
                "cold_start_plan_invalid",
                f"Cold-start plan protocol validation failed: {exc}",
                layer=FailureLayer.PLANNER_GRAPH,
            ) from exc
        try:
            return cold_start_plan_from_dict(submission.value)
        except (KeyError, TypeError, ValueError) as exc:
            raise PlannerProposalError(
                "cold_start_plan_invalid",
                f"Cold-start plan semantic validation failed: {exc}",
                layer=FailureLayer.PLANNER_GRAPH,
            ) from exc


__all__ = [
    "COLD_START_PLAN_SCHEMA",
    "COLD_START_PLAN_STEP_SCHEMA",
    "COLD_START_PROMPT",
    "ColdStartPlanner",
    "cold_start_plan_from_dict",
]
