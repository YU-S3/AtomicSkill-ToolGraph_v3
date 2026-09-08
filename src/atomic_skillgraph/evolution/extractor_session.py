"""A real two-turn E1/E2 exchange in one AgentSession."""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class CompositeExtractionAuthority:
    """Deterministic code authority shared by initial E2 and its sole repair."""

    canonical_control_sequence: tuple[str, ...]
    canonical_occurrences: tuple[dict[str, Any], ...]
    new_edge_candidates: tuple[Any, ...]
    new_edge_candidate_ids: frozenset[str]
    existing_edge_views: tuple[dict[str, Any], ...]
    existing_edge_by_id: dict[str, dict[str, Any]]


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


def _composite_authority(
    authoritative_occurrences: list[CanonicalAtomicOccurrence],
    existing_edges: list[Any],
    *,
    contract_matcher: ContractMatcher | None = None,
) -> CompositeExtractionAuthority:
    """Project one immutable-in-practice E2 candidate view without LLM state."""

    identity_by_value: dict[str, str] = {}

    def identity(value: Any) -> str:
        key = repr(value)
        if key not in identity_by_value:
            identity_by_value[key] = (
                f"binding_{len(identity_by_value) + 1:03d}"
            )
        return identity_by_value[key]

    canonical_occurrences = tuple(
        {
            "occurrence_id": item.occurrence_id,
            "skill_ref": str(item.proposed_ref),
            "intent": item.intent,
            "inputs": to_primitive(item.input_specs),
            "outputs": to_primitive(item.output_specs),
            "effects": to_primitive(item.effects),
            "input_binding_identities": {
                role: identity(value)
                for role, value in item.input_bindings.items()
            },
            "output_binding_identities": {
                role: identity(value)
                for role, value in item.output_bindings.items()
            },
        }
        for item in authoritative_occurrences
    )
    matcher = contract_matcher or ExactContractMatcher()
    edge_builder = CompositeEdgeCandidateBuilder()
    candidates = edge_builder.build(
        authoritative_occurrences,
        matcher=matcher,
    )
    existing_views, existing_by_id = (
        edge_builder.existing_edge_materializations(
            authoritative_occurrences,
            existing_edges,
        )
    )
    return CompositeExtractionAuthority(
        canonical_control_sequence=tuple(
            item.occurrence_id for item in authoritative_occurrences
        ),
        canonical_occurrences=canonical_occurrences,
        new_edge_candidates=tuple(candidates),
        new_edge_candidate_ids=frozenset(
            item.candidate_id for item in candidates
        ),
        existing_edge_views=tuple(existing_views),
        existing_edge_by_id=dict(existing_by_id),
    )


def _proposal_from_payload(
    payload: Mapping[str, Any],
    authority: CompositeExtractionAuthority,
    *,
    stage: str,
) -> CompositeExtractionProposal:
    selected_existing_ids = [
        str(item) for item in payload["selected_existing_edge_ids"]
    ]
    unknown_existing = sorted(
        set(selected_existing_ids) - set(authority.existing_edge_by_id)
    )
    if unknown_existing:
        raise ExtractionContentError(
            stage,
            (
                "extractor_e2_existing_edge_selection_invalid"
                if stage == "e2"
                else "extractor_e2_repair_existing_edge_selection_invalid"
            ),
            "E2 selected unknown/inapplicable existing edge IDs: "
            + ", ".join(unknown_existing),
        )
    selected_candidate_ids = [
        str(item) for item in payload["selected_new_edge_candidate_ids"]
    ]
    unknown_candidates = sorted(
        set(selected_candidate_ids) - authority.new_edge_candidate_ids
    )
    if unknown_candidates:
        raise ExtractionContentError(
            stage,
            (
                "extractor_e2_new_edge_selection_invalid"
                if stage == "e2"
                else "extractor_e2_repair_new_edge_selection_invalid"
            ),
            "E2 selected unknown edge candidate IDs: "
            + ", ".join(unknown_candidates),
        )
    candidate_by_id = {
        item.candidate_id: item for item in authority.new_edge_candidates
    }
    edge_builder = CompositeEdgeCandidateBuilder()
    return CompositeExtractionProposal(
        list(authority.canonical_control_sequence),
        [
            authority.existing_edge_by_id[item]
            for item in selected_existing_ids
        ],
        [
            edge_builder.materialize_candidate(candidate_by_id[item])
            for item in selected_candidate_ids
        ],
        str(payload["summary"]),
        dict(payload["guideline"]),
        dict(payload["insight"]),
    )


def _e2_repair_prompt(
    rejected_proposal: CompositeExtractionProposal,
    rejection: Exception,
    authority: CompositeExtractionAuthority,
) -> str:
    rejection_detail = " ".join(str(rejection).split())[:1200]
    policy_context = {
        "deterministic_rejection": rejection_detail,
        "rejected_proposal": to_primitive(rejected_proposal),
        "canonical_control_sequence": list(
            authority.canonical_control_sequence
        ),
        "canonical_occurrences": to_primitive(
            authority.canonical_occurrences
        ),
        "known_existing_edge_evidence": to_primitive(
            authority.existing_edge_views
        ),
        "new_edge_candidates": to_primitive(
            authority.new_edge_candidates
        ),
    }
    instruction = (
        "E2R COMPOSITE REPAIR. The previous schema-valid Composite proposal "
        "was rejected by deterministic Composite validation. Return one "
        "complete replacement proposal using the unchanged E2 output schema. "
        "You may only re-select from the exact existing-edge IDs and new-edge "
        "candidate IDs supplied below. Do not modify the canonical control "
        "sequence, invent an edge, change an Atomic occurrence, add facts, or "
        "perform retrieval."
    )
    return instruction + "\n\nPOLICY_CONTEXT_JSON\n" + json.dumps(
        policy_context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ExtractorSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.context = ContextBuilder()
        self.submissions = StructuredSubmissionClient()
        self._e1_complete = False
        self._e2_complete = False
        self._e2_repair_complete = False
        self._e2_protocol_repairs_before: int | None = None

    def _protocol_repairs_used(self) -> int:
        snapshot = getattr(self.session, "snapshot", None)
        if not callable(snapshot):
            return 0
        value = snapshot().get("protocol_repairs_used", 0)
        if isinstance(value, bool):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @property
    def e2_protocol_repair_count(self) -> int:
        if self._e2_protocol_repairs_before is None:
            return 0
        return max(
            0,
            self._protocol_repairs_used()
            - self._e2_protocol_repairs_before,
        )

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
        authority = _composite_authority(
            authoritative_occurrences,
            existing_edges,
            contract_matcher=contract_matcher,
        )
        self._e2_protocol_repairs_before = self._protocol_repairs_used()
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e2")
        try:
            payload = self.submissions.request(
                self.session,
                prompt=self.context.extractor_e2(
                    canonical_occurrences=authority.canonical_occurrences,
                    canonical_control_sequence=(
                        authority.canonical_control_sequence
                    ),
                    known_existing_edge_evidence=(
                        authority.existing_edge_views
                    ),
                    new_edge_candidates=to_primitive(
                        authority.new_edge_candidates
                    ),
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
        proposal = _proposal_from_payload(payload, authority, stage="e2")
        self._e2_complete = True
        return proposal

    def repair_composite(
        self,
        rejected_proposal: CompositeExtractionProposal,
        rejection: Exception,
        authoritative_occurrences: list[CanonicalAtomicOccurrence],
        existing_edges: list[Any],
        *,
        contract_matcher: ContractMatcher | None = None,
    ) -> CompositeExtractionProposal:
        if not self._e1_complete:
            raise RuntimeError("Extractor E2R requires completed E1")
        if not self._e2_complete:
            raise RuntimeError(
                "Extractor E2R requires one schema-valid initial E2"
            )
        if self._e2_repair_complete:
            raise RuntimeError("Extractor E2R may run exactly once")
        self._e2_repair_complete = True
        authority = _composite_authority(
            authoritative_occurrences,
            existing_edges,
            contract_matcher=contract_matcher,
        )
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("extractor_e2")
        try:
            payload = self.submissions.request(
                self.session,
                prompt=_e2_repair_prompt(
                    rejected_proposal,
                    rejection,
                    authority,
                ),
                tool_name="submit_extractor_composite",
                description=(
                    "Replace the rejected Composite edge selection."
                ),
                schema=E2_SCHEMA,
            ).value
        except AgentProtocolError as exc:
            raise ExtractionContentError(
                "e2_repair",
                "extractor_e2_repair_schema_rejected",
                str(exc),
            ) from exc
        return _proposal_from_payload(
            payload,
            authority,
            stage="e2_repair",
        )
