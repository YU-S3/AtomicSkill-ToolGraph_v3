"""A real two-turn E1/E2 exchange in one AgentSession."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..agents.context_builder import ContextBuilder
from ..agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
    COMPOSITE_EXTRACTION_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.contracts import SemanticPredicate
from ..core.errors import AgentProtocolError
from ..core.serialization import to_primitive
from ..validation.contract_matcher import ContractMatcher, ExactContractMatcher
from .atomicizer import AtomicOccurrenceProposal, CanonicalAtomicOccurrence
from .composite_edge_candidates import CompositeEdgeCandidateBuilder


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


class ExtractionContentError(ValueError):
    """A staged Extractor submission/content rejection, never task failure."""

    def __init__(self, stage: str, error_code: str, message: str) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.error_code = str(error_code)


def _predicate(value: dict[str, Any]) -> SemanticPredicate:
    return SemanticPredicate(
        str(value["predicate"]), dict(value.get("args", {})),
        int(value.get("cardinality", 1)), str(value.get("distinct_by", "")),
        str(value.get("effect_domain", "world")),
    )


class ExtractorSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.context = ContextBuilder()
        self.submissions = StructuredSubmissionClient()
        self._e1_complete = False
        self._e2_complete = False

    def propose_atomics(
        self,
        normalized_trace: dict[str, Any],
        known_atomic_contracts: list[Any] | tuple[Any, ...] = (),
        required_task_contract_witnesses: Any = (),
        *,
        runtime_automation_drafts: list[Any] | tuple[Any, ...] = (),
        runtime_tool_trials: list[Any] | tuple[Any, ...] = (),
    ) -> list[AtomicOccurrenceProposal]:
        if self._e1_complete:
            raise RuntimeError("Extractor E1 may run exactly once")
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e1")
        try:
            payload = self.submissions.request(
                self.session,
                prompt=self.context.extractor_e1(
                    canonical_trace=normalized_trace,
                    known_atomic_contracts=known_atomic_contracts,
                    required_task_contract_witnesses=(
                        required_task_contract_witnesses
                    ),
                    runtime_automation_drafts=runtime_automation_drafts,
                    runtime_tool_trials=runtime_tool_trials,
                ),
                tool_name="submit_extractor_atomics",
                description=(
                    "Submit the complete Atomic occurrence extraction proposal."
                ),
                schema=E1_SCHEMA,
            ).value
        except AgentProtocolError as exc:
            raise ExtractionContentError(
                "e1",
                "extractor_e1_schema_rejected",
                str(exc),
            ) from exc
        self._e1_complete = True
        proposals: list[AtomicOccurrenceProposal] = []
        for item in payload["occurrences"]:
            event_start = int(item["event_start"])
            event_end_exclusive = int(item["event_end"])
            if event_end_exclusive <= event_start:
                raise ValueError(
                    "Extractor E1 event_end must be exclusive and greater than event_start"
                )
            proposals.append(AtomicOccurrenceProposal(
                phase_id=str(item["phase_id"]), intent=str(item["intent"]),
                event_start=event_start, event_end=event_end_exclusive - 1,
                input_roles=dict(item["input_roles"]), output_roles=dict(item["output_roles"]),
                preconditions=[_predicate(value) for value in item["preconditions"]],
                effects=[_predicate(value) for value in item["effects"]], rationale=str(item["rationale"]),
                support_event_ids=[str(value) for value in item.get("support_event_ids", [])],
                shared_precondition_event_ids=[
                    str(value)
                    for value in item.get(
                        "shared_precondition_event_ids", []
                    )
                ],
                precondition_witness_refs=[str(value) for value in item.get("precondition_witness_refs", [])],
                effect_witness_refs=[str(value) for value in item.get("effect_witness_refs", [])],
                ordering_constraints=[dict(value) for value in item.get("ordering_constraints", [])],
                input_provenance_refs={
                    str(role): str(authority_ref)
                    for role, authority_ref in dict(
                        item["input_provenance_refs"]
                    ).items()
                },
                input_provenance_contract="code_authority_v3_2",
                # Every output derivation is an explicit E1 authority claim.
                # Preserve it verbatim for deterministic code validation;
                # neither INPUT_IDENTITY nor EFFECT_WITNESS may be inferred
                # from coincident concrete values at this boundary.
                output_derivations={
                    str(role): dict(raw)
                    for role, raw in dict(
                        item["output_derivations"]
                    ).items()
                },
            ))
        return proposals

    def propose_composite(
        self, authoritative_occurrences: list[CanonicalAtomicOccurrence],
        existing_edges: list[Any],
        *,
        contract_matcher: ContractMatcher | None = None,
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
            } for item in authoritative_occurrences
        ]
        matcher = contract_matcher or ExactContractMatcher()
        edge_builder = CompositeEdgeCandidateBuilder()
        candidates = edge_builder.build(
            authoritative_occurrences,
            matcher=matcher,
        )
        candidate_by_id = {item.candidate_id: item for item in candidates}
        existing_views, existing_by_id = (
            edge_builder.existing_edge_materializations(
                authoritative_occurrences,
                existing_edges,
            )
        )
        control_sequence = [
            item.occurrence_id for item in authoritative_occurrences
        ]
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e2")
        try:
            payload = self.submissions.request(
                self.session,
                prompt=self.context.extractor_e2(
                    canonical_occurrences=authority,
                    canonical_control_sequence=control_sequence,
                    known_existing_edge_evidence=existing_views,
                    new_edge_candidates=to_primitive(candidates),
                ),
                tool_name="submit_extractor_composite",
                description=(
                    "Select admitted Composite edge evidence and candidates."
                ),
                schema=E2_SCHEMA,
            ).value
        except AgentProtocolError as exc:
            raise ExtractionContentError(
                "e2",
                "extractor_e2_schema_rejected",
                str(exc),
            ) from exc
        self._e2_complete = True
        selected_existing_ids = [
            str(item) for item in payload["selected_existing_edge_ids"]
        ]
        unknown_existing = sorted(
            set(selected_existing_ids) - set(existing_by_id)
        )
        if unknown_existing:
            raise ExtractionContentError(
                "e2",
                "extractor_e2_existing_edge_selection_invalid",
                "E2 selected unknown/inapplicable existing edge IDs: "
                + ", ".join(unknown_existing),
            )
        selected_candidate_ids = [
            str(item)
            for item in payload["selected_new_edge_candidate_ids"]
        ]
        unknown_candidates = sorted(
            set(selected_candidate_ids) - set(candidate_by_id)
        )
        if unknown_candidates:
            raise ExtractionContentError(
                "e2",
                "extractor_e2_new_edge_selection_invalid",
                "E2 selected unknown edge candidate IDs: "
                + ", ".join(unknown_candidates),
            )
        return CompositeExtractionProposal(
            control_sequence,
            [existing_by_id[item] for item in selected_existing_ids],
            [
                edge_builder.materialize_candidate(candidate_by_id[item])
                for item in selected_candidate_ids
            ],
            str(payload["summary"]),
            dict(payload["guideline"]),
            dict(payload["insight"]),
        )
