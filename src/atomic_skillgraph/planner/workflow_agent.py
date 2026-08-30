"""P2 and the single permitted P2R turn in the same PlannerSession."""

from __future__ import annotations

import json
from typing import Any

from ..agents.structured_submission import (
    PROPOSED_EDGE_SCHEMA,
    PROPOSED_OCCURRENCE_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.bindings import BindingExpression
from ..core.contracts import PlannerWorkflowProposal, ProposedEdge, ProposedOccurrence
from ..core.refs import SkillRef
from ..core.serialization import to_primitive


WORKFLOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["steps", "control_sequence", "data_edges", "dependency_edges", "requirement_coverage"],
    "additionalProperties": False,
    "properties": {
        "steps": {"type": "array", "minItems": 1, "items": PROPOSED_OCCURRENCE_SCHEMA},
        "control_sequence": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "data_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "dependency_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "requirement_coverage": {
            "type": "object",
            "additionalProperties": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
    },
}


def _proposal(payload: dict[str, Any]) -> PlannerWorkflowProposal:
    steps = []
    for item in payload.get("steps", []):
        steps.append(ProposedOccurrence(
            step_id=str(item["step_id"]), occurrence_id=str(item.get("occurrence_id", item["step_id"])),
            node_ref=SkillRef.parse(item["node_ref"] if isinstance(item["node_ref"], str) else SkillRef.from_dict(item["node_ref"])),
            requirement_ids=[str(value) for value in item.get("requirement_ids", [])],
            binding_specs={name: BindingExpression.from_dict(value) for name, value in item.get("binding_specs", {}).items()},
            expected_effects=[],
        ))
    return PlannerWorkflowProposal(
        steps=steps, control_sequence=[str(item) for item in payload.get("control_sequence", [])],
        data_edges=[ProposedEdge(**item) for item in payload.get("data_edges", [])],
        dependency_edges=[ProposedEdge(**item) for item in payload.get("dependency_edges", [])],
        requirement_coverage={str(key): [str(item) for item in value] for key, value in payload.get("requirement_coverage", {}).items()},
        sequence_origin="planner_proposed_sequence",
    )


class WorkflowAgent:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()

    def propose(self, task: Any, contract: Any, requirements: Any, candidates: Any, existing_edges: Any, hints: Any) -> PlannerWorkflowProposal:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p2")
        prompt = (
            "P2 LINEAR WORKFLOW. Select only supplied Atomic candidates. Produce exactly one complete control "
            "sequence. Data/dependency edges may fan in/out but must point forward. Mark every new edge "
            "origin=planner_proposed and every reused edge origin=existing_active with its real id. "
            "Call only the offered submit tool.\n"
            f"Task: {task.goal}\nContract: {json.dumps(to_primitive(contract), ensure_ascii=False)}\n"
            f"Requirements: {json.dumps(to_primitive(requirements), ensure_ascii=False)}\n"
            f"Candidates: {json.dumps(to_primitive(candidates), ensure_ascii=False)}\n"
            f"Existing edges: {json.dumps(to_primitive(existing_edges), ensure_ascii=False)}\n"
            f"Related hints: {json.dumps(to_primitive(hints), ensure_ascii=False)}"
        )
        return _proposal(self.submissions.request(
            self.session,
            prompt=prompt,
            tool_name="submit_planner_workflow",
            description="Submit one complete linear Planner workflow proposal.",
            schema=WORKFLOW_SCHEMA,
        ).value)

    def repair(self, proposal: PlannerWorkflowProposal, validation: Any, authoritative_contracts: Any, existing_edges: Any) -> PlannerWorkflowProposal:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p2_repair")
        prompt = (
            "P2R GRAPH REPAIR. Return a complete replacement proposal. You may change occurrence selection/order, "
            "planner-proposed edges, delete bad edges, or repeat a supplied Atomic. Do not invent or rewrite skills, "
            "forge existing edges, or add another control path. Call only the offered submit tool.\n"
            f"Proposal: {json.dumps(to_primitive(proposal), ensure_ascii=False)}\n"
            f"Validation errors: {json.dumps(to_primitive(validation), ensure_ascii=False)}\n"
            f"Authoritative contracts: {json.dumps(to_primitive(authoritative_contracts), ensure_ascii=False)}\n"
            f"Existing edges: {json.dumps(to_primitive(existing_edges), ensure_ascii=False)}"
        )
        return _proposal(self.submissions.request(
            self.session,
            prompt=prompt,
            tool_name="submit_planner_workflow",
            description="Submit one complete replacement Planner workflow proposal.",
            schema=WORKFLOW_SCHEMA,
        ).value)
