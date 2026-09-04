"""Deterministic Full-Dynamic post-success knowledge-gap diagnosis."""

from __future__ import annotations

from typing import Any

from ..core.bindings import BindingExpression
from ..core.contracts import CapabilityRequirement, ParameterSpec, SemanticPredicate
from ..knowledge.query import atomic_contract_compatible
from ..knowledge.skill_registry import SkillRegistry
from .aligner import _atomic_signature


def _predicate(raw: dict[str, Any]) -> SemanticPredicate:
    arguments = {
        name: BindingExpression.from_dict(value)
        if isinstance(value, dict) and "kind" in value else value
        for name, value in dict(raw.get("args") or {}).items()
    }
    return SemanticPredicate(
        str(raw.get("predicate", "")),
        arguments,
        int(raw.get("cardinality", 1)),
        str(raw.get("distinct_by", "")),
        raw.get("effect_domain", "world"),
    )


def _requirement(raw: dict[str, Any]) -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id=str(raw.get("requirement_id", "")),
        intent=str(raw.get("intent", "")),
        desired_effects=[_predicate(item) for item in raw.get("desired_effects", [])],
        expected_inputs=[ParameterSpec(**item) for item in raw.get("expected_inputs", [])],
        expected_outputs=[ParameterSpec(**item) for item in raw.get("expected_outputs", [])],
        precondition_hints=[_predicate(item) for item in raw.get("precondition_hints", [])],
        semantic_variants=list(raw.get("semantic_variants", [])),
        required=bool(raw.get("required", True)),
        rationale=str(raw.get("rationale", "")),
    )


class GapDiagnoser:
    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills

    def diagnose(self, trace: Any, extracted_atomics: list[Any]) -> dict[str, Any]:
        strict_success = bool(
            getattr(trace, "strict_task_success", False)
            or (
                getattr(trace, "benchmark_success", False)
                and getattr(trace, "learning_eligible", True)
            )
        )
        if trace.runtime_plan.get("source") != "full_dynamic" or not strict_success:
            return {}
        audit = dict(trace.planner_audit or {})
        search_rows = list(audit.get("atomic_search_p1r") or audit.get("atomic_search_p1") or [])
        uncovered_rows = [item for item in search_rows if item.get("covered") is not True]
        requirements = [
            (_requirement(dict(item.get("requirement") or {})), item)
            for item in uncovered_rows
            if item.get("requirement")
        ]
        existing = self.skills.atomics()
        details: list[dict[str, Any]] = []
        classifications: list[str] = []

        for atomic in extracted_atomics:
            equivalent = [item for item in existing if _atomic_signature(item) == _atomic_signature(atomic)]
            matched = [
                (requirement, row)
                for requirement, row in requirements
                if atomic_contract_compatible(requirement, atomic)
            ]
            retrieved_refs = {
                str(candidate.get("atomic_ref", ""))
                for _, row in matched
                for candidate in row.get("candidates", [])
            }
            if matched and equivalent:
                classification = (
                    "novel_workflow_only"
                    if any(str(item.ref) in retrieved_refs for item in equivalent)
                    else "retrieval_miss"
                )
            elif matched:
                classification = "confirmed_capability_gap"
            elif equivalent:
                classification = "novel_workflow_only"
            else:
                classification = "planner_requirement_error"
            classifications.append(classification)
            details.append({
                "extracted_atomic_ref": str(atomic.ref),
                "classification": classification,
                "matched_uncovered_requirement_ids": [item.requirement_id for item, _ in matched],
                "equivalent_existing_refs": [str(item.ref) for item in equivalent],
            })

        if not details:
            overall = "planner_requirement_error"
        elif "confirmed_capability_gap" in classifications:
            overall = "confirmed_capability_gap"
        elif "retrieval_miss" in classifications:
            overall = "retrieval_miss"
        elif all(item == "novel_workflow_only" for item in classifications):
            overall = "novel_workflow_only"
        else:
            overall = "planner_requirement_error"
        counts = {
            name: classifications.count(name)
            for name in (
                "confirmed_capability_gap",
                "planner_requirement_error",
                "retrieval_miss",
                "novel_workflow_only",
            )
        }
        return {
            "trace_id": str(getattr(trace, "trace_id", "")),
            "classification": overall,
            "counts": counts,
            "uncovered_requirement_ids": [item.requirement_id for item, _ in requirements],
            "details": details,
            "skill_penalty_applied": False,
        }


__all__ = ["GapDiagnoser"]
