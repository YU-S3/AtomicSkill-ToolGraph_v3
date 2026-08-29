"""Bounded, structured-output session for semantic batch evolution proposals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..core.refs import canonical_json


_OPERATIONS = ("generalize", "specialize", "update", "merge", "split")

EVOLUTION_REPAIR_SCHEMA = {
    "type": "object",
    "required": ["proposals"],
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "review_id",
                    "operation",
                    "target_refs",
                    "candidates",
                    "rationale",
                ],
                "additionalProperties": False,
                "properties": {
                    "review_id": {"type": "string", "minLength": 1},
                    "operation": {"type": "string", "enum": list(_OPERATIONS)},
                    "target_refs": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "candidates": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": [
                                "summary",
                                "signature",
                                "interface",
                                "artifact_kind",
                                "artifact",
                                "safety",
                                "source_cases",
                                "source_step_indexes",
                                "logical_id_suffix",
                            ],
                            "additionalProperties": False,
                            "properties": {
                                "summary": {"type": "string", "minLength": 1},
                                "signature": {"type": "object"},
                                "interface": {"type": "object"},
                                "artifact_kind": {
                                    "type": "string",
                                    "enum": ["primitive_ir"],
                                },
                                "artifact": {"type": "object"},
                                "safety": {"type": "object"},
                                "source_cases": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "required": ["case_id", "effect_indexes"],
                                        "additionalProperties": False,
                                        "properties": {
                                            "case_id": {
                                                "type": "string",
                                                "minLength": 1,
                                            },
                                            "effect_indexes": {
                                                "type": "array",
                                                "minItems": 1,
                                                "uniqueItems": True,
                                                "items": {
                                                    "type": "integer",
                                                    "minimum": 0,
                                                },
                                            },
                                        },
                                    },
                                },
                                "source_step_indexes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "integer",
                                        "minimum": 0,
                                    },
                                },
                                "logical_id_suffix": {"type": "string"},
                            },
                        },
                    },
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class EvolutionToolCandidateProposal:
    summary: str
    signature: dict[str, Any]
    interface: dict[str, Any]
    artifact_kind: str
    artifact: dict[str, Any]
    safety: dict[str, Any]
    source_cases: tuple[dict[str, Any], ...]
    source_step_indexes: tuple[int, ...]
    logical_id_suffix: str


@dataclass(frozen=True)
class EvolutionToolEditProposal:
    review_id: str
    operation: str
    target_refs: tuple[str, ...]
    candidates: tuple[EvolutionToolCandidateProposal, ...]
    rationale: str


class EvolutionRepairSession:
    """One bounded semantic-proposal turn; code remains admission authority."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self._used = False

    def propose(self, reviews: list[dict[str, Any]]) -> list[EvolutionToolEditProposal]:
        if self._used:
            raise RuntimeError("EvolutionRepairSession may run exactly once")
        if not reviews:
            self._used = True
            return []
        turn = self.session.next_turn(
            "Review only the supplied evidence-backed Tool cohorts. Return no proposal "
            "when evidence does not justify an edit. Never invent a source replay case, "
            "effect, action, or target reference. source_step_indexes declare the ordered "
            "source-step authority for each candidate. For split they must form disjoint, "
            "contiguous, exhaustive spans and source effects must form disjoint, exhaustive "
            "partitions. Code will independently replay, validate, and admit every proposed "
            "candidate."
            "\n\nPOLICY_CONTEXT_JSON\n"
            + canonical_json({"reviews": reviews}),
            structured_output_schema=EVOLUTION_REPAIR_SCHEMA,
        )
        payload = json.loads(turn.content)
        self._used = True
        return [
            EvolutionToolEditProposal(
                review_id=str(item["review_id"]),
                operation=str(item["operation"]),
                target_refs=tuple(map(str, item["target_refs"])),
                candidates=tuple(
                    EvolutionToolCandidateProposal(
                        summary=str(candidate["summary"]),
                        signature=dict(candidate["signature"]),
                        interface=dict(candidate["interface"]),
                        artifact_kind=str(candidate["artifact_kind"]),
                        artifact=dict(candidate["artifact"]),
                        safety=dict(candidate["safety"]),
                        source_cases=tuple(
                            {
                                "case_id": str(case["case_id"]),
                                "effect_indexes": [
                                    int(value) for value in case["effect_indexes"]
                                ],
                            }
                            for case in candidate["source_cases"]
                        ),
                        source_step_indexes=tuple(
                            int(value) for value in candidate["source_step_indexes"]
                        ),
                        logical_id_suffix=str(candidate["logical_id_suffix"]),
                    )
                    for candidate in item["candidates"]
                ),
                rationale=str(item["rationale"]),
            )
            for item in payload["proposals"]
        ]


__all__ = [
    "EVOLUTION_REPAIR_SCHEMA",
    "EvolutionRepairSession",
    "EvolutionToolCandidateProposal",
    "EvolutionToolEditProposal",
]
