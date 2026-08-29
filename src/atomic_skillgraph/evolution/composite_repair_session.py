"""Bounded structured producer for Composite sequence replacements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.refs import SkillRef, canonical_json
from .composite_repairs import CompositeSequenceRepairEngine
from .repair import RepairProposal
from .typed_repairs import RepairEvidence


COMPOSITE_SEQUENCE_DECISION_SCHEMA = {
    "type": "object",
    "required": ["decisions"],
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["review_id", "decision", "replacement", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "review_id": {"type": "string", "minLength": 1},
                    "decision": {"type": "string", "enum": ["no_change", "propose"]},
                    "replacement": {"type": "object"},
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class CompositeSequenceReview:
    """Code-authoritative target and structural replay evidence."""

    review_id: str
    target_ref: str
    source_composite: dict[str, Any]
    structural_context: dict[str, Any]
    evidence: tuple[RepairEvidence, ...]
    source_failure_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.review_id:
            raise ValueError("Composite sequence review requires review_id")
        object.__setattr__(self, "target_ref", str(SkillRef.parse(self.target_ref)))
        object.__setattr__(
            self,
            "evidence",
            tuple(RepairEvidence.from_value(item) for item in self.evidence),
        )
        if not isinstance(self.source_composite, dict) or not self.source_composite:
            raise ValueError("Composite sequence review requires the source artifact")
        if not isinstance(self.structural_context, dict):
            raise TypeError("Composite structural_context must be a dict")

    def prompt_view(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "target_ref": self.target_ref,
            "eligible_operation": "revise_composite_sequence",
            "source_composite": self.source_composite,
            "structural_context": self.structural_context,
            "evidence_authority": [
                {
                    "evidence_id": item.evidence_id,
                    "task_id": item.task_id,
                    "trace_id": item.trace_id,
                    "cluster_key": item.cluster_key,
                    "failure_layer": item.failure_layer,
                    "failure_code": item.failure_code,
                }
                for item in self.evidence
            ],
        }


@dataclass(frozen=True)
class CompositeSequenceDecision:
    review_id: str
    decision: str
    replacement: dict[str, Any]
    rationale: str


class CompositeSequenceProposalSession:
    """Run one proposal turn; code remains replay and admission authority."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self._used = False

    def propose(
        self, reviews: Sequence[CompositeSequenceReview],
    ) -> list[CompositeSequenceDecision]:
        if self._used:
            raise RuntimeError("CompositeSequenceProposalSession may run exactly once")
        self._validate_reviews(reviews)
        if not reviews:
            self._used = True
            return []
        turn = self.session.next_turn(
            "Review only the supplied code-authoritative Composite structural contexts. "
            "Return at most one decision per review. Use no_change unless the evidence "
            "supports a real control-sequence revision. For propose, return one complete "
            "replacement Composite artifact. Never return or invent target refs, operation "
            "names, evidence, clusters, trace ids, or failure ids. Do not insert/remove "
            "occurrences or change any non-sequence semantics. Code will reconstruct the "
            "original structured transitions and independently replay, PlannerValidate, "
            "and admit the replacement.\n\nPOLICY_CONTEXT_JSON\n"
            + canonical_json({"reviews": [item.prompt_view() for item in reviews]}),
            structured_output_schema=COMPOSITE_SEQUENCE_DECISION_SCHEMA,
        )
        payload = json.loads(turn.content)
        decisions = self.parse(payload, reviews)
        self._used = True
        return decisions

    @staticmethod
    def parse(
        payload: Mapping[str, Any],
        reviews: Sequence[CompositeSequenceReview],
    ) -> list[CompositeSequenceDecision]:
        CompositeSequenceProposalSession._validate_reviews(reviews)
        if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
            raise ValueError("Composite response must contain only decisions[]")
        authority = {item.review_id: item for item in reviews}
        seen: set[str] = set()
        result: list[CompositeSequenceDecision] = []
        expected = {"review_id", "decision", "replacement", "rationale"}
        for raw in payload["decisions"]:
            if not isinstance(raw, Mapping) or set(raw) != expected:
                raise ValueError("Composite decision schema mismatch")
            review_id = str(raw["review_id"])
            if review_id not in authority:
                raise ValueError(f"unknown Composite review: {review_id}")
            if review_id in seen:
                raise ValueError(f"duplicate Composite response: {review_id}")
            seen.add(review_id)
            decision = str(raw["decision"])
            replacement = raw["replacement"]
            rationale = str(raw["rationale"])
            if decision not in {"no_change", "propose"} or not rationale:
                raise ValueError("Composite decision/rationale invalid")
            if not isinstance(replacement, Mapping):
                raise ValueError("Composite replacement must be an object")
            if decision == "no_change" and replacement:
                raise ValueError("Composite no_change cannot carry a replacement")
            if decision == "propose" and not replacement:
                raise ValueError("Composite propose requires a complete replacement")
            result.append(CompositeSequenceDecision(
                review_id,
                decision,
                dict(replacement),
                rationale,
            ))
        return result

    @staticmethod
    def build_proposals(
        decisions: Sequence[CompositeSequenceDecision],
        reviews: Sequence[CompositeSequenceReview],
    ) -> list[RepairProposal]:
        CompositeSequenceProposalSession._validate_reviews(reviews)
        authority = {item.review_id: item for item in reviews}
        seen: set[str] = set()
        result: list[RepairProposal] = []
        for decision in decisions:
            if decision.review_id not in authority:
                raise ValueError(f"unknown Composite review: {decision.review_id}")
            if decision.review_id in seen:
                raise ValueError(f"duplicate Composite decision: {decision.review_id}")
            seen.add(decision.review_id)
            if decision.decision == "no_change":
                if decision.replacement:
                    raise ValueError("Composite no_change cannot carry a replacement")
                continue
            if decision.decision != "propose" or not decision.replacement:
                raise ValueError("Composite proposal decision is invalid")
            review = authority[decision.review_id]
            result.append(CompositeSequenceRepairEngine.build_proposal(
                review.target_ref,
                decision.replacement,
                review.evidence,
                review.source_failure_ids,
            ))
        return result

    @staticmethod
    def _validate_reviews(reviews: Sequence[CompositeSequenceReview]) -> None:
        ids = [item.review_id for item in reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("Composite review ids must be unique")


__all__ = [
    "COMPOSITE_SEQUENCE_DECISION_SCHEMA",
    "CompositeSequenceDecision",
    "CompositeSequenceProposalSession",
    "CompositeSequenceReview",
]
