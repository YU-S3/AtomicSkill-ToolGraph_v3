"""Two-turn F1/F2 extraction from a failed cold-start task trace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.errors import AgentProtocolError, FailureLayer, PlannerProposalError
from ..core.serialization import to_primitive


_NONEMPTY = {"type": "string", "minLength": 1}
_INDEX = {"type": ["integer", "null"], "minimum": 0}

PLAN_STEP_ALIGNMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "step_id", "status", "event_start", "event_end",
        "effect_witness_refs", "rationale",
    ],
    "additionalProperties": False,
    "properties": {
        "step_id": _NONEMPTY,
        "status": {
            "type": "string",
            "enum": [
                "achieved", "attempted_not_achieved", "not_attempted",
                "diverged_then_recovered",
            ],
        },
        "event_start": _INDEX,
        "event_end": _INDEX,
        "effect_witness_refs": {"type": "array", "items": _NONEMPTY},
        "rationale": _NONEMPTY,
    },
}

FAILURE_ALIGNMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "alignment_id", "step_alignments", "matched_prefix_step_ids",
        "first_unrecovered_divergence", "remaining_requirement_instance_ids",
        "candidate_progress_spans",
    ],
    "additionalProperties": False,
    "properties": {
        "alignment_id": _NONEMPTY,
        "step_alignments": {
            "type": "array", "minItems": 1,
            "items": PLAN_STEP_ALIGNMENT_SCHEMA,
        },
        "matched_prefix_step_ids": {
            "type": "array", "uniqueItems": True, "items": _NONEMPTY,
        },
        "first_unrecovered_divergence": {
            "type": "object",
            "required": ["kind", "step_id", "event_index", "summary"],
            "additionalProperties": False,
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "wrong_identity_reuse", "unsatisfied_precondition",
                        "repeated_no_progress", "binding_mismatch",
                        "plan_order_mismatch", "unresolved_requirement",
                        "budget_exhaustion", "other",
                    ],
                },
                "step_id": {"type": "string"},
                "event_index": _INDEX,
                "summary": _NONEMPTY,
            },
        },
        "remaining_requirement_instance_ids": {
            "type": "array", "uniqueItems": True, "items": _NONEMPTY,
        },
        "candidate_progress_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step_id", "event_start", "event_end", "effect_witness_refs"],
                "additionalProperties": False,
                "properties": {
                    "step_id": _NONEMPTY,
                    "event_start": {"type": "integer", "minimum": 0},
                    "event_end": {"type": "integer", "minimum": 1},
                    "effect_witness_refs": {"type": "array", "minItems": 1, "items": _NONEMPTY},
                },
            },
        },
    },
}

FAILURE_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "provisional_atomics", "validated_plan_prefix",
        "negative_method_suffix", "reusable_failure_summary",
    ],
    "additionalProperties": False,
    "properties": {
        "provisional_atomics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "atomic_proposal", "aligned_plan_step_ids",
                    "progress_relation",
                ],
                "additionalProperties": False,
                "properties": {
                    "atomic_proposal": ATOMIC_EXTRACTION_SCHEMA,
                    "aligned_plan_step_ids": {
                        "type": "array", "minItems": 1, "uniqueItems": True,
                        "items": _NONEMPTY,
                    },
                    "progress_relation": {
                        "type": "string",
                        "enum": [
                            "partial_target_effect", "consumed_prerequisite",
                            "no_progress",
                        ],
                    },
                },
            },
        },
        "validated_plan_prefix": {
            "type": "array", "uniqueItems": True, "items": _NONEMPTY,
        },
        "negative_method_suffix": {"type": "object"},
        "reusable_failure_summary": {"type": "object"},
    },
}


@dataclass
class PlanStepAlignment:
    step_id: str
    status: str
    event_start: int | None
    event_end: int | None
    effect_witness_refs: list[str]
    rationale: str


@dataclass
class FailurePlanAlignment:
    alignment_id: str
    step_alignments: list[PlanStepAlignment]
    matched_prefix_step_ids: list[str]
    first_unrecovered_divergence: dict[str, Any]
    remaining_requirement_instance_ids: list[str]
    candidate_progress_spans: list[dict[str, Any]]


@dataclass
class FailureAtomicProposal:
    atomic_proposal: dict[str, Any]
    aligned_plan_step_ids: list[str]
    progress_relation: str


@dataclass
class FailureExtractionProposal:
    provisional_atomics: list[FailureAtomicProposal]
    validated_plan_prefix: list[str]
    negative_method_suffix: dict[str, Any]
    reusable_failure_summary: dict[str, Any]


def _alignment(value: dict[str, Any]) -> FailurePlanAlignment:
    return FailurePlanAlignment(
        alignment_id=str(value["alignment_id"]),
        step_alignments=[PlanStepAlignment(
            step_id=str(item["step_id"]),
            status=str(item["status"]),
            event_start=(None if item["event_start"] is None else int(item["event_start"])),
            event_end=(None if item["event_end"] is None else int(item["event_end"])),
            effect_witness_refs=list(map(str, item["effect_witness_refs"])),
            rationale=str(item["rationale"]),
        ) for item in value["step_alignments"]],
        matched_prefix_step_ids=list(map(str, value["matched_prefix_step_ids"])),
        first_unrecovered_divergence=dict(value["first_unrecovered_divergence"]),
        remaining_requirement_instance_ids=list(map(str, value["remaining_requirement_instance_ids"])),
        candidate_progress_spans=[dict(item) for item in value["candidate_progress_spans"]],
    )


def _extraction(value: dict[str, Any]) -> FailureExtractionProposal:
    return FailureExtractionProposal(
        provisional_atomics=[FailureAtomicProposal(
            atomic_proposal=dict(item["atomic_proposal"]),
            aligned_plan_step_ids=list(map(str, item["aligned_plan_step_ids"])),
            progress_relation=str(item["progress_relation"]),
        ) for item in value["provisional_atomics"]],
        validated_plan_prefix=list(map(str, value["validated_plan_prefix"])),
        negative_method_suffix=dict(value["negative_method_suffix"]),
        reusable_failure_summary=dict(value["reusable_failure_summary"]),
    )


class FailureExtractorSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()
        self._f1_complete = False
        self._f2_complete = False

    def align(
        self,
        *,
        task_contract: Any,
        requirement_expansion: Any,
        cold_start_plan: Any,
        trace_events: Any,
        task_progress: Any,
        failures: Any,
        candidate_contracts: Any,
    ) -> FailurePlanAlignment:
        if self._f1_complete:
            raise RuntimeError("Failure Extractor F1 may run exactly once")
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("failure_extractor_f1")
        prompt = (
            "F1 PLAN-TRACE ALIGNMENT. Identify the first unrecovered divergence from the supplied plan, not the first\n"
            "low-level error. Later actions are not automatically invalid; a later span may\n"
            "still establish an independent reusable Effect. Call only the offered submission tool.\n"
            f"TaskContract: {json.dumps(to_primitive(task_contract), ensure_ascii=False)}\n"
            f"RequirementExpansion: {json.dumps(to_primitive(requirement_expansion), ensure_ascii=False)}\n"
            f"ColdStartPlan: {json.dumps(to_primitive(cold_start_plan), ensure_ascii=False)}\n"
            f"Structured events: {json.dumps(to_primitive(trace_events), ensure_ascii=False)}\n"
            f"TaskProgress: {json.dumps(to_primitive(task_progress), ensure_ascii=False)}\n"
            f"Failure envelopes: {json.dumps(to_primitive(failures), ensure_ascii=False)}\n"
            f"Candidate contract views: {json.dumps(to_primitive(candidate_contracts), ensure_ascii=False)}"
        )
        result = _alignment(self._request(
            prompt, "submit_failure_plan_alignment", FAILURE_ALIGNMENT_SCHEMA,
        ))
        self._f1_complete = True
        return result

    def extract(
        self,
        *,
        validated_alignment: FailurePlanAlignment,
        authoritative_trace: Any,
        task_contract: Any,
    ) -> FailureExtractionProposal:
        if not self._f1_complete or self._f2_complete:
            raise RuntimeError(
                "Failure Extractor F2 requires one completed F1 and may run exactly once"
            )
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("failure_extractor_f2")
        prompt = (
            "F2 FAILURE ASSET EXTRACTION. Propose only portable provisional Atomic contracts from independently validated "
            "real Effect spans and one non-executable negative method summary. Do not output a Composite, Implementation, "
            "Tool, source action script, or concrete source entity/location. Call only the offered submission tool.\n"
            f"Code-validated F1: {json.dumps(to_primitive(validated_alignment), ensure_ascii=False)}\n"
            f"Authoritative trace: {json.dumps(to_primitive(authoritative_trace), ensure_ascii=False)}\n"
            f"TaskContract: {json.dumps(to_primitive(task_contract), ensure_ascii=False)}"
        )
        result = _extraction(self._request(
            prompt, "submit_failure_assets", FAILURE_EXTRACTION_SCHEMA,
        ))
        self._f2_complete = True
        return result

    def _request(self, prompt: str, tool_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.submissions.request(
                self.session,
                prompt=prompt,
                tool_name=tool_name,
                description="Submit the complete failure-side analysis for this stage.",
                schema=schema,
            ).value
        except AgentProtocolError as exc:
            raise PlannerProposalError(
                "failure_extractor_protocol_rejected",
                f"Failure Extractor protocol validation failed: {exc}",
                layer=FailureLayer.RUNTIME_AGENT,
            ) from exc


__all__ = [
    "FAILURE_ALIGNMENT_SCHEMA", "FAILURE_EXTRACTION_SCHEMA",
    "FailureAtomicProposal", "FailureExtractionProposal",
    "FailureExtractorSession", "FailurePlanAlignment", "PlanStepAlignment",
]
