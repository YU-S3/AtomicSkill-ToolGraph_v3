"""Two-turn F1/F2 extraction from a failed cold-start task trace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.errors import (
    AgentProtocolError,
    BudgetExhausted,
    FailureLayer,
    PlannerProposalError,
)
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
        "negative_method_suffix": {
            "type": "object",
            "minProperties": 1,
        },
        "reusable_failure_summary": {
            "type": "object",
            "minProperties": 1,
        },
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


@dataclass(frozen=True)
class FailureExtractorSessionAllocation:
    """One fresh F2 conversation allocation derived from UsageLedger.

    This is deliberately not a token ledger.  ``remaining_tokens`` is a
    read-only allocation snapshot calculated by the System from authoritative
    provider usage already stored in the shared UsageLedger.
    """

    session: Any | None
    remaining_tokens: int

    def __post_init__(self) -> None:
        if self.remaining_tokens < 0:
            raise ValueError("Failure Extractor remaining_tokens must be non-negative")


class FailureExtractorBudgetUnavailable(BudgetExhausted):
    """No F2 provider call may start after F1 consumed the task budget."""

    def __init__(self) -> None:
        super().__init__(
            "extractor_token_budget_exhausted",
            "Failure Extractor F2 has no remaining task-level token budget",
            layer=FailureLayer.RUNTIME_AGENT,
        )
        self.failure_extractor_stage = "f2_not_started_no_remaining_budget"


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
    def __init__(
        self,
        f1_session: Any,
        f2_session_factory: Callable[[], FailureExtractorSessionAllocation],
    ) -> None:
        self.f1_session = f1_session
        self.f2_session_factory = f2_session_factory
        self.f2_session: Any | None = None
        self.submissions = StructuredSubmissionClient()
        self._f1_complete = False
        self._f2_complete = False
        self._diagnostics: dict[str, int] = {}

    @property
    def f1_session_id(self) -> str:
        return str(getattr(self.f1_session, "session_id", ""))

    @property
    def f2_session_id(self) -> str:
        return str(getattr(self.f2_session, "session_id", ""))

    @property
    def diagnostics(self) -> dict[str, int]:
        return dict(self._diagnostics)

    def align(
        self,
        *,
        alignment_view: Any,
    ) -> FailurePlanAlignment:
        if self._f1_complete:
            raise RuntimeError("Failure Extractor F1 may run exactly once")
        payload = to_primitive(alignment_view)
        prompt = (
            "F1 PLAN-TRACE ALIGNMENT. Identify the first unrecovered divergence from the supplied plan, not the first\n"
            "low-level error. Later actions are not automatically invalid; a later span may\n"
            "still establish an independent reusable Effect. Call only the offered submission tool.\n"
            f"FailureAlignmentView: {json.dumps(payload, ensure_ascii=False)}"
        )
        self._diagnostics.update({
            "failure_extractor_f1_input_event_count": len(
                dict(payload).get("execution_events", [])
            ),
            "failure_extractor_f1_prompt_chars": len(prompt),
            "failure_extractor_f1_prompt_bytes": len(prompt.encode("utf-8")),
        })
        result = _alignment(self._request(
            self.f1_session,
            prompt,
            "submit_failure_plan_alignment",
            FAILURE_ALIGNMENT_SCHEMA,
        ))
        self._f1_complete = True
        return result

    def extract(
        self,
        *,
        asset_view: Any,
    ) -> FailureExtractionProposal:
        if not self._f1_complete or self._f2_complete:
            raise RuntimeError(
                "Failure Extractor F2 requires one completed F1 and may run exactly once"
            )
        allocation = self.f2_session_factory()
        self._diagnostics[
            "failure_extractor_remaining_budget_before_f2"
        ] = int(allocation.remaining_tokens)
        if allocation.session is None or allocation.remaining_tokens == 0:
            self._diagnostics[
                "failure_extractor_skipped_after_budget_count"
            ] = 1
            raise FailureExtractorBudgetUnavailable()
        self.f2_session = allocation.session
        payload = to_primitive(asset_view)
        prompt = (
            "F2 FAILURE ASSET EXTRACTION. Propose only portable provisional Atomic contracts from independently validated "
            "real Effect spans and one non-executable negative method summary. Do not output a Composite, Implementation, "
            "Tool, source action script, or concrete source entity/location. Call only the offered submission tool.\n"
            "A failed task may contain no reusable local Atomic Effect. In that case submit provisional_atomics as an "
            "empty array and still provide the portable negative Failure Experience summary. Do not invent a local "
            "Atomic only to make the list non-empty.\n"
            f"FailureAssetExtractionView: {json.dumps(payload, ensure_ascii=False)}"
        )
        spans = list(dict(payload).get("candidate_progress_spans", []))
        self._diagnostics.update({
            "failure_extractor_f2_span_count": len(spans),
            "failure_extractor_f2_source_event_count": sum(
                len(dict(span).get("accepted_events", [])) for span in spans
            ),
            "failure_extractor_f2_prompt_chars": len(prompt),
            "failure_extractor_f2_prompt_bytes": len(prompt.encode("utf-8")),
        })
        result = _extraction(self._request(
            self.f2_session,
            prompt,
            "submit_failure_assets",
            FAILURE_EXTRACTION_SCHEMA,
        ))
        self._f2_complete = True
        return result

    def _request(
        self,
        session: Any,
        prompt: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.submissions.request(
                session,
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
    "FailureExtractorBudgetUnavailable", "FailureExtractorSession",
    "FailureExtractorSessionAllocation", "FailurePlanAlignment",
    "PlanStepAlignment",
]
