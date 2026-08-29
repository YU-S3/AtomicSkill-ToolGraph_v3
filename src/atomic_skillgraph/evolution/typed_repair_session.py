"""One bounded structured-output turn for typed Atomic/Implementation reviews."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.refs import SkillRef, canonical_json
from ..core.serialization import to_primitive
from .repair import RepairProposal
from .typed_repairs import RepairEvidence, TypedRepairEngine


ATOMIC_OPERATIONS = (
    "revise_atomic_contract",
    "split_atomic",
    "merge_atomic",
)
IMPLEMENTATION_OPERATIONS = (
    "revise_implementation_mapping",
    "revise_grounding_constraint",
    "specialize_implementation",
)
TYPED_OPERATIONS = (*ATOMIC_OPERATIONS, *IMPLEMENTATION_OPERATIONS)


TYPED_REPAIR_DECISION_SCHEMA = {
    "type": "object",
    "required": ["decisions"],
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["review_id", "decision", "operation", "replacements", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "review_id": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "enum": ["no_change", "propose"]},
                    "operation": {"type": "string", "enum": ["no_change", *TYPED_OPERATIONS]},
                    "replacements": {"type": "array", "items": {"type": "object"}},
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class TypedRepairReview:
    """Code-owned review authority; evidence is never accepted from the LLM."""

    review_id: str
    target_layer: str
    target_refs: tuple[str, ...]
    eligible_operations: tuple[str, ...]
    context: dict[str, Any]
    evidence: tuple[RepairEvidence, ...]
    source_failure_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.review_id or self.target_layer not in {"atomic", "implementation"}:
            raise ValueError("typed review requires a review id and known target layer")
        object.__setattr__(
            self, "target_refs", tuple(str(SkillRef.parse(item)) for item in self.target_refs),
        )
        object.__setattr__(
            self, "evidence", tuple(RepairEvidence.from_value(item) for item in self.evidence),
        )
        allowed = set(ATOMIC_OPERATIONS if self.target_layer == "atomic" else IMPLEMENTATION_OPERATIONS)
        if not self.eligible_operations or not set(self.eligible_operations) <= allowed:
            raise ValueError("typed review contains an ineligible operation")
        if (
            any(item == "merge_atomic" for item in self.eligible_operations)
            and len(self.target_refs) < 2
        ):
            raise ValueError("Atomic merge review requires multiple targets")
        if any(item != "merge_atomic" for item in self.eligible_operations) and len(self.target_refs) != 1:
            raise ValueError("non-merge review requires exactly one target")

    def prompt_view(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "target_layer": self.target_layer,
            "target_refs": list(self.target_refs),
            "eligible_operations": list(self.eligible_operations),
            "context": self.context,
            "evidence_authority": [
                {
                    "evidence_id": item.evidence_id,
                    "task_id": item.task_id,
                    "trace_id": item.trace_id,
                    "cluster_key": item.cluster_key,
                    "failure_layer": item.failure_layer,
                    "failure_code": item.failure_code,
                    "candidate_keys": list(item.candidate_keys),
                }
                for item in self.evidence
            ],
        }


@dataclass(frozen=True)
class TypedRepairDecision:
    review_id: str
    decision: str
    operation: str
    replacements: tuple[dict[str, Any], ...]
    rationale: str


class TypedRepairProposalSession:
    """Ask once; code validates identity, scope, and evidence authority."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self._used = False

    def propose(self, reviews: Sequence[TypedRepairReview]) -> list[TypedRepairDecision]:
        if self._used:
            raise RuntimeError("TypedRepairProposalSession may run exactly once")
        self._validate_reviews(reviews)
        if not reviews:
            self._used = True
            return []
        turn = self.session.next_turn(
            "Review only the code-authoritative Atomic/Implementation contexts below. "
            "Return at most one decision per review; use no_change when the supplied "
            "evidence does not justify an eligible operation. You may propose replacement "
            "artifact payloads only. Never invent, select, edit, or return evidence, cluster "
            "membership, target refs, or failure ids: code owns those fields and will replay, "
            "validate, and admit every candidate.\n\nPOLICY_CONTEXT_JSON\n"
            + canonical_json({"reviews": [item.prompt_view() for item in reviews]}),
            structured_output_schema=TYPED_REPAIR_DECISION_SCHEMA,
        )
        payload = json.loads(turn.content)
        decisions = self.parse(payload, reviews)
        self._used = True
        return decisions

    @staticmethod
    def parse(
        payload: Mapping[str, Any], reviews: Sequence[TypedRepairReview],
    ) -> list[TypedRepairDecision]:
        TypedRepairProposalSession._validate_reviews(reviews)
        if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
            raise ValueError("typed repair response must contain only decisions[]")
        authority = {item.review_id: item for item in reviews}
        seen: set[str] = set()
        result: list[TypedRepairDecision] = []
        required_keys = {"review_id", "decision", "operation", "replacements", "rationale"}
        for raw in payload["decisions"]:
            if not isinstance(raw, Mapping) or set(raw) != required_keys:
                raise ValueError("typed repair decision schema mismatch")
            review_id = str(raw["review_id"])
            if review_id not in authority:
                raise ValueError(f"unknown typed repair review: {review_id}")
            if review_id in seen:
                raise ValueError(f"duplicate typed repair response: {review_id}")
            seen.add(review_id)
            decision = str(raw["decision"])
            operation = str(raw["operation"])
            replacements = raw["replacements"]
            rationale = str(raw["rationale"])
            if decision not in {"no_change", "propose"} or not rationale:
                raise ValueError("typed repair decision/rationale invalid")
            if not isinstance(replacements, list) or any(not isinstance(item, Mapping) for item in replacements):
                raise ValueError("typed repair replacements must be objects")
            if decision == "no_change":
                if operation != "no_change" or replacements:
                    raise ValueError("no_change cannot carry an operation or replacement")
            else:
                if operation not in authority[review_id].eligible_operations:
                    raise ValueError("typed repair operation is not authorized by the review")
                expected_multiple = operation == "split_atomic"
                if (expected_multiple and len(replacements) < 2) or (not expected_multiple and len(replacements) != 1):
                    raise ValueError("typed repair replacement cardinality invalid")
            result.append(TypedRepairDecision(
                review_id, decision, operation,
                tuple(dict(item) for item in replacements), rationale,
            ))
        return result

    @staticmethod
    def build_proposals(
        decisions: Sequence[TypedRepairDecision],
        reviews: Sequence[TypedRepairReview],
    ) -> list[RepairProposal]:
        TypedRepairProposalSession._validate_reviews(reviews)
        authority = {item.review_id: item for item in reviews}
        seen: set[str] = set()
        proposals: list[RepairProposal] = []
        for decision in decisions:
            if decision.review_id not in authority:
                raise ValueError(f"unknown typed repair review: {decision.review_id}")
            if decision.review_id in seen:
                raise ValueError(f"duplicate typed repair decision: {decision.review_id}")
            seen.add(decision.review_id)
            if decision.decision == "no_change":
                if decision.operation != "no_change" or decision.replacements:
                    raise ValueError("no_change cannot carry an operation or replacement")
                continue
            if decision.decision != "propose":
                raise ValueError("unknown typed repair decision")
            review = authority[decision.review_id]
            if decision.operation not in review.eligible_operations:
                raise ValueError("typed repair operation is not authorized by the review")
            if (
                decision.operation == "split_atomic" and len(decision.replacements) < 2
            ) or (
                decision.operation != "split_atomic" and len(decision.replacements) != 1
            ):
                raise ValueError("typed repair replacement cardinality invalid")
            proposals.append(TypedRepairEngine.build_proposal(
                decision.operation,
                review.target_refs,
                decision.replacements,
                review.evidence,
                review.source_failure_ids,
            ))
        return proposals

    @staticmethod
    def _validate_reviews(reviews: Sequence[TypedRepairReview]) -> None:
        ids = [item.review_id for item in reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("typed repair review ids must be unique")


__all__ = [
    "ATOMIC_OPERATIONS",
    "IMPLEMENTATION_OPERATIONS",
    "TYPED_REPAIR_DECISION_SCHEMA",
    "TypedRepairDecision",
    "TypedRepairProposalSession",
    "TypedRepairReview",
]
