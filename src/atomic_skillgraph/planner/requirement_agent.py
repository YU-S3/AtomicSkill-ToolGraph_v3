"""P1 and the single permitted P1R turn in one PlannerSession."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..core.contracts import CapabilityRequirement, ParameterSpec, SemanticPredicate, TaskContract
from ..core.serialization import to_primitive


REQUIREMENT_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["requirements"], "additionalProperties": False,
    "properties": {"requirements": {"type": "array", "items": {
        "type": "object", "required": ["requirement_id", "intent", "desired_effects", "required", "rationale"],
        "properties": {
            "requirement_id": {"type": "string"}, "intent": {"type": "string"},
            "desired_effects": {"type": "array", "items": {"type": "object"}},
            "expected_inputs": {"type": "array", "items": {"type": "object"}},
            "expected_outputs": {"type": "array", "items": {"type": "object"}},
            "precondition_hints": {"type": "array", "items": {"type": "object"}},
            "semantic_variants": {"type": "array", "items": {"type": "string"}},
            "required": {"type": "boolean"}, "rationale": {"type": "string"},
        },
    }}},
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


def _turn_json(turn: Any) -> dict[str, Any]:
    content = str(getattr(turn, "content", "") or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Planner structured output was not valid JSON") from exc


class RequirementAgent:
    def __init__(self, session: Any) -> None:
        self.session = session

    def propose(
        self, task: Any, contract: TaskContract, observation: str,
        harness_profile: str,
    ) -> list[CapabilityRequirement]:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p1")
        prompt = (
            "P1 CAPABILITY REQUIREMENTS. Decompose the task into complete, reusable state-transition "
            "capabilities. Do not prescribe environment actions or invent skills. Return JSON only.\n"
            f"Task goal: {task.goal}\nPolicy observation: {observation}\n"
            f"TaskContract: {json.dumps(to_primitive(contract), ensure_ascii=False)}\n"
            f"Harness profile: {harness_profile}"
        )
        turn = self.session.next_turn(prompt, structured_output_schema=REQUIREMENT_SCHEMA)
        return [_requirement(item) for item in _turn_json(turn).get("requirements", [])]

    def repair(
        self, task: Any, contract: TaskContract, requirements: list[CapabilityRequirement],
        search: list[Any], related_composites: list[dict[str, Any]],
    ) -> list[CapabilityRequirement]:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p1_repair")
        prompt = (
            "P1R REQUIREMENT REPAIR. Coverage is incomplete. Return one complete replacement requirement "
            "list, not a patch. Composite material is only a hint and is not an oracle. JSON only.\n"
            f"Task: {task.goal}\nTaskContract: {json.dumps(to_primitive(contract), ensure_ascii=False)}\n"
            f"requirements_v1: {json.dumps(to_primitive(requirements), ensure_ascii=False)}\n"
            f"search/rejections: {json.dumps(to_primitive(search), ensure_ascii=False)}\n"
            f"related hints: {json.dumps(related_composites, ensure_ascii=False)}"
        )
        turn = self.session.next_turn(prompt, structured_output_schema=REQUIREMENT_SCHEMA)
        return [_requirement(item) for item in _turn_json(turn).get("requirements", [])]
