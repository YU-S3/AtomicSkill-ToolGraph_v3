"""A real two-turn E1/E2 exchange in one AgentSession."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agents.context_builder import ContextBuilder
from ..core.contracts import SemanticPredicate
from ..core.serialization import to_primitive
from .atomicizer import AtomicOccurrenceProposal, CanonicalAtomicOccurrence


E1_SCHEMA = {
    "type": "object", "required": ["occurrences"], "additionalProperties": False,
    "properties": {"occurrences": {"type": "array", "items": {
        "type": "object", "required": ["phase_id", "intent", "event_start", "event_end", "input_roles", "output_roles", "preconditions", "effects", "rationale"],
        "properties": {
            "phase_id": {"type": "string"}, "intent": {"type": "string"},
            "event_start": {"type": "integer"}, "event_end": {"type": "integer"},
            "input_roles": {"type": "object"}, "output_roles": {"type": "object"},
            "preconditions": {"type": "array", "items": {"type": "object"}},
            "effects": {"type": "array", "items": {"type": "object"}},
            "rationale": {"type": "string"},
        },
    }}},
}

E2_SCHEMA = {
    "type": "object", "required": ["control_sequence", "existing_edges", "new_edges", "summary", "guideline", "insight"],
    "properties": {
        "control_sequence": {"type": "array", "items": {"type": "string"}},
        "existing_edges": {"type": "array", "items": {"type": "object"}},
        "new_edges": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "string"}, "guideline": {"type": "object"}, "insight": {"type": "object"},
    },
}


@dataclass
class CompositeExtractionProposal:
    control_sequence: list[str]
    existing_edges: list[dict[str, Any]]
    new_edges: list[dict[str, Any]]
    summary: str
    guideline: dict[str, Any]
    insight: dict[str, Any]


def _predicate(value: dict[str, Any]) -> SemanticPredicate:
    return SemanticPredicate(
        str(value["predicate"]), dict(value.get("args", {})),
        int(value.get("cardinality", 1)), str(value.get("distinct_by", "")),
    )


class ExtractorSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.context = ContextBuilder()
        self._e1_complete = False
        self._e2_complete = False

    def propose_atomics(self, normalized_trace: dict[str, Any]) -> list[AtomicOccurrenceProposal]:
        if self._e1_complete:
            raise RuntimeError("Extractor E1 may run exactly once")
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e1")
        turn = self.session.next_turn(
            self.context.extractor_e1(canonical_trace=normalized_trace),
            structured_output_schema=E1_SCHEMA,
        )
        payload = json.loads(turn.content)
        self._e1_complete = True
        return [AtomicOccurrenceProposal(
            phase_id=str(item["phase_id"]), intent=str(item["intent"]),
            event_start=int(item["event_start"]), event_end=int(item["event_end"]),
            input_roles=dict(item["input_roles"]), output_roles=dict(item["output_roles"]),
            preconditions=[_predicate(value) for value in item["preconditions"]],
            effects=[_predicate(value) for value in item["effects"]], rationale=str(item["rationale"]),
        ) for item in payload["occurrences"]]

    def propose_composite(
        self, authoritative_occurrences: list[CanonicalAtomicOccurrence],
        existing_edges: list[Any],
    ) -> CompositeExtractionProposal:
        if not self._e1_complete or self._e2_complete:
            raise RuntimeError("Extractor E2 requires one completed E1 and may run exactly once")
        authority = [
            {
                "occurrence_id": item.occurrence_id, "skill_ref": str(item.proposed_ref),
                "intent": item.intent, "inputs": to_primitive(item.input_specs),
                "outputs": to_primitive(item.output_specs), "effects": to_primitive(item.effects),
                "known_edge_evidence": [
                    to_primitive(edge)
                    for edge in existing_edges
                    if str(item.proposed_ref) in {
                        str(getattr(edge, "source_step_ref", "")),
                        str(getattr(edge, "target_step_ref", "")),
                    }
                ],
            } for item in authoritative_occurrences
        ]
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e2")
        turn = self.session.next_turn(
            self.context.extractor_e2(canonical_occurrences=authority),
            structured_output_schema=E2_SCHEMA,
        )
        payload = json.loads(turn.content)
        self._e2_complete = True
        return CompositeExtractionProposal(
            [str(item) for item in payload["control_sequence"]], list(payload["existing_edges"]),
            list(payload["new_edges"]), str(payload["summary"]), dict(payload["guideline"]), dict(payload["insight"]),
        )
