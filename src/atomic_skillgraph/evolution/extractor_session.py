"""A real two-turn E1/E2 exchange in one AgentSession."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.context_builder import ContextBuilder
from ..agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
    COMPOSITE_EXTRACTION_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.serialization import to_primitive
from .atomicizer import AtomicBoundaryProposal, CanonicalAtomicOccurrence


E1_SCHEMA = {
    "type": "object", "required": ["occurrences"], "additionalProperties": False,
    "properties": {"occurrences": {
        "type": "array", "minItems": 1, "items": ATOMIC_EXTRACTION_SCHEMA,
    }},
}

E2_SCHEMA = COMPOSITE_EXTRACTION_SCHEMA


@dataclass
class CompositeExtractionProposal:
    control_sequence: list[str]
    existing_edges: list[dict[str, Any]]
    new_edges: list[dict[str, Any]]
    summary: str
    guideline: dict[str, Any]
    insight: dict[str, Any]


class ExtractorSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.context = ContextBuilder()
        self.submissions = StructuredSubmissionClient()
        self._e1_complete = False
        self._e2_complete = False

    def propose_atomics(self, normalized_trace: dict[str, Any]) -> list[AtomicBoundaryProposal]:
        if self._e1_complete:
            raise RuntimeError("Extractor E1 may run exactly once")
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e1")
        payload = self.submissions.request(
            self.session,
            prompt=self.context.extractor_e1(canonical_trace=normalized_trace),
            tool_name="submit_extractor_atomics",
            description="Submit the complete Atomic occurrence extraction proposal.",
            schema=E1_SCHEMA,
        ).value
        self._e1_complete = True
        proposals: list[AtomicBoundaryProposal] = []
        for item in payload["occurrences"]:
            event_start = int(item["event_start"])
            event_end_exclusive = int(item["event_end_exclusive"])
            if event_end_exclusive <= event_start:
                raise ValueError(
                    "Extractor E1 event_end_exclusive must be greater than event_start"
                )
            proposals.append(AtomicBoundaryProposal(
                phase_id=str(item["phase_id"]), intent=str(item["intent"]),
                event_start=event_start,
                event_end_exclusive=event_end_exclusive,
                selected_effect_refs=[str(value) for value in item["selected_effect_refs"]],
                selected_precondition_refs=[
                    str(value) for value in item["selected_precondition_refs"]
                ],
                output_role_mapping={
                    str(role): str(source)
                    for role, source in dict(item["output_role_mapping"]).items()
                },
                rationale=str(item["rationale"]),
            ))
        return proposals

    def propose_composite(
        self, authoritative_occurrences: list[CanonicalAtomicOccurrence],
        existing_edges: list[Any],
        task_contract: Any,
    ) -> CompositeExtractionProposal:
        if not self._e1_complete or self._e2_complete:
            raise RuntimeError("Extractor E2 requires one completed E1 and may run exactly once")
        identity_by_value: dict[str, str] = {}

        def identity(value: Any) -> str:
            key = repr(value)
            if key not in identity_by_value:
                identity_by_value[key] = f"binding_{len(identity_by_value) + 1:03d}"
            return identity_by_value[key]

        authority = [
            {
                "occurrence_id": item.occurrence_id, "skill_ref": str(item.proposed_ref),
                "intent": item.intent, "inputs": to_primitive(item.input_specs),
                "outputs": to_primitive(item.output_specs), "effects": to_primitive(item.effects),
                "input_binding_identities": {
                    role: identity(value)
                    for role, value in item.input_bindings.items()
                },
                "output_binding_identities": {
                    role: identity(value)
                    for role, value in item.output_bindings.items()
                },
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
        payload = self.submissions.request(
            self.session,
            prompt=self.context.extractor_e2(
                canonical_occurrences=authority,
                task_contract=task_contract,
            ),
            tool_name="submit_extractor_composite",
            description="Submit the complete Composite extraction proposal.",
            schema=E2_SCHEMA,
        ).value
        self._e2_complete = True
        return CompositeExtractionProposal(
            [str(item) for item in payload["control_sequence"]], list(payload["existing_edges"]),
            list(payload["new_edges"]), str(payload["summary"]), dict(payload["guideline"]), dict(payload["insight"]),
        )
