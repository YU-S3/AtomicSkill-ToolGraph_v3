"""Deterministic Planner P1R repairability gate.

The gate never calls an LLM.  It only classifies the already-computed P1
bundle validation and sanitized Atomic retrieval diagnostics into:

* repairable interface mismatches, which may enter P1R with bounded hints; or
* hard capability gaps, where the required Effect family has no current Atomic
  candidate at all and P1R is skipped entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.contracts import (
    PlannerRequirementBundle,
)
from ..core.serialization import to_primitive


@dataclass(frozen=True)
class RepairabilityDecision:
    repairable: bool
    reason_code: str
    requirement_ids: tuple[str, ...]
    candidate_refs: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]


def _predicate(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("predicate", ""))
    return str(getattr(value, "predicate", ""))


def _effect_predicates(value: Any) -> set[str]:
    result: set[str] = set()
    for effect in value if isinstance(value, (list, tuple)) else ():
        name = _predicate(effect).casefold()
        if name:
            result.add(name)
    return result


def _failure_codes(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        value = value.get("compatibility", value)
    if isinstance(value, Mapping):
        return {str(item) for item in value.get("failure_codes", ())}
    return {str(item) for item in getattr(value, "failure_codes", ()) or ()}


_INTERFACE_MISMATCH_CODES = {
    "input_contract_mismatch",
    "output_contract_mismatch",
    "semantic_type_mismatch",
    "cardinality_mismatch",
    "role_interface_mismatch",
    "interface_mismatch",
    "missing_required_input_types",
}


def _near_effect_match(desired: set[str], hint: Mapping[str, Any]) -> bool:
    contract = dict(hint.get("contract_view") or {})
    offered = {
        _predicate(item).casefold()
        for item in contract.get("effects", ())
    }
    offered.update(
        str(item).casefold()
        for item in hint.get("effect_predicates", ())
    )
    for component in hint.get("components", ()):
        for item in dict(component).get("effects", ()):
            offered.add(_predicate(item).casefold())
    return bool(desired & offered)


def _composite_hint_interface_relevant(
    result: Any,
    hints: Iterable[Mapping[str, Any]],
) -> bool:
    """Deterministic effect-family / interface overlap, never existence-only."""

    if isinstance(result, Mapping):
        requirement = result.get("requirement")
    else:
        requirement = getattr(result, "requirement", None)
    if isinstance(requirement, Mapping):
        desired = _effect_predicates(requirement.get("desired_effects", ()))
        expected_inputs = {
            str(item.get("name", "")): str(item.get("semantic_type", ""))
            for item in requirement.get("expected_inputs", ())
            if isinstance(item, Mapping)
        }
    else:
        desired = _effect_predicates(
            getattr(requirement, "desired_effects", ())
        )
        expected_inputs = {
            str(item.name): str(item.semantic_type)
            for item in getattr(requirement, "expected_inputs", ()) or ()
        }
    for hint in hints:
        if _near_effect_match(desired, hint):
            return True
        for component in hint.get("components", ()):
            component = dict(component)
            component_outputs = {
                str(item.get("name", "")): str(item.get("semantic_type", ""))
                for item in component.get("outputs", ())
                if isinstance(item, Mapping)
            }
            if expected_inputs and any(
                value and value == component_outputs.get(key)
                for key, value in expected_inputs.items()
            ):
                return True
    return False


def _candidate_refs(results: Iterable[Any]) -> tuple[str, ...]:
    refs: list[str] = []
    for result in results:
        if isinstance(result, Mapping):
            candidates = result.get("candidates", ())
            hints = result.get("repair_hints", ())
        else:
            candidates = getattr(result, "candidates", ())
            hints = getattr(result, "repair_hints", ()) or ()
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                ref = candidate.get("atomic_ref")
            else:
                ref = getattr(candidate, "atomic_ref", "")
            if ref:
                refs.append(str(ref))
        for hint in hints:
            if isinstance(hint, Mapping):
                ref = hint.get("atomic_ref")
                if ref:
                    refs.append(str(ref))
    return tuple(dict.fromkeys(refs))


def _result_diagnostics(
    result: Any,
    *,
    repairable: bool,
    hard_gap: bool,
    reason_code: str,
) -> dict[str, Any]:
    if isinstance(result, Mapping):
        requirement = result.get("requirement")
        rejection_reasons = result.get("rejection_reasons", ())
        repair_hints = result.get("repair_hints", ())
        covered = bool(result.get("covered", False))
    else:
        requirement = getattr(result, "requirement", None)
        rejection_reasons = getattr(result, "rejection_reasons", ()) or ()
        repair_hints = getattr(result, "repair_hints", ()) or ()
        covered = bool(getattr(result, "covered", False))
    if isinstance(requirement, Mapping):
        requirement_id = str(requirement.get("requirement_id", ""))
        desired = _effect_predicates(requirement.get("desired_effects", ()))
    else:
        requirement_id = str(getattr(requirement, "requirement_id", ""))
        desired = _effect_predicates(
            getattr(requirement, "desired_effects", ())
        )
    return {
        "requirement_id": requirement_id,
        "covered": bool(covered),
        "desired_effect_predicates": sorted(desired),
        "repairable": bool(repairable),
        "hard_capability_gap": bool(hard_gap),
        "reason_code": reason_code,
        "rejection_reasons": to_primitive(list(rejection_reasons)),
        "repair_hints": to_primitive(list(repair_hints)),
    }


def _result_is_hard_gap(result: Any) -> bool:
    if isinstance(result, Mapping):
        requirement = result.get("requirement")
        desired = _effect_predicates(
            requirement.get("desired_effects", ())
            if isinstance(requirement, Mapping)
            else ()
        )
        hints = result.get("repair_hints", ())
        rejection_reasons = result.get("rejection_reasons", ())
    else:
        requirement = getattr(result, "requirement", None)
        desired = _effect_predicates(
            getattr(requirement, "desired_effects", ())
        )
        hints = getattr(result, "repair_hints", ()) or ()
        rejection_reasons = getattr(result, "rejection_reasons", ()) or ()
    near = any(_near_effect_match(desired, hint) for hint in hints if isinstance(hint, Mapping))
    if near:
        return False
    # A rejected candidate with the same predicate but only an input/output
    # mismatch is repairable even when the sanitized hint list is empty.
    for rejection in rejection_reasons:
        if not isinstance(rejection, Mapping):
            continue
        compatibility = dict(rejection.get("compatibility") or {})
        if bool(compatibility.get("effects_passed")):
            return False
        codes = _failure_codes(rejection)
        if codes & _INTERFACE_MISMATCH_CODES:
            return False
    return True


class RepairabilityGate:
    """Classify whether P1R can materially help one uncovered Requirement."""

    def decide(
        self,
        bundle: PlannerRequirementBundle,
        validation: Any,
        search_results: Iterable[Any],
        related_composite_hints: Iterable[Mapping[str, Any]] = (),
    ) -> RepairabilityDecision:
        results = list(search_results)
        required_uncovered = [
            result for result in results
            if not bool(
                result.get("covered", False)
                if isinstance(result, Mapping)
                else getattr(result, "covered", False)
            )
            and bool(
                result.get("requirement", {}).get("required", True)
                if isinstance(result, Mapping)
                else getattr(getattr(result, "requirement", None), "required", True)
            )
        ]
        if not required_uncovered:
            return RepairabilityDecision(
                False, "planner_repair_not_required", (), (), (),
            )

        validation_passed = bool(getattr(validation, "passed", True))
        diagnostics: list[dict[str, Any]] = []
        if not validation_passed:
            diagnostics.append({
                "stage": "bundle_validation",
                "passed": False,
                "detail": to_primitive(validation),
            })
            requirement_ids = [
                _result_diagnostics(
                    result,
                    repairable=True,
                    hard_gap=False,
                    reason_code="planner_requirement_bundle_invalid",
                )["requirement_id"]
                for result in required_uncovered
            ]
            return RepairabilityDecision(
                True,
                "planner_requirement_bundle_invalid",
                tuple(requirement_ids),
                _candidate_refs(results),
                tuple(diagnostics),
            )

        related_hints = list(related_composite_hints)
        per_result: list[tuple[Any, bool, str]] = []
        for result in required_uncovered:
            if _result_is_hard_gap(result):
                if _composite_hint_interface_relevant(result, related_hints):
                    per_result.append(
                        (result, True, "related_composite_interface_repairable")
                    )
                else:
                    per_result.append((result, False, "planner_hard_capability_gap"))
            else:
                per_result.append((result, True, "coverage_partial_effect_match"))

        repairable_count = sum(1 for _, ok, _ in per_result if ok)
        hard_count = len(per_result) - repairable_count
        if repairable_count and all(
            item_code == "related_composite_interface_repairable"
            for _result, _ok, item_code in per_result
            if _ok
        ):
            reason_code = "related_composite_interface_repairable"
        else:
            reason_code = (
                "planner_hard_capability_gap"
                if hard_count and not repairable_count
                else "coverage_partial_effect_match"
            )
        diagnostics = [
            _result_diagnostics(
                result,
                repairable=repairable,
                hard_gap=not repairable,
                reason_code=item_code,
            )
            for result, repairable, item_code in per_result
        ]
        if related_hints:
            diagnostics.append({
                "stage": "related_composite_hints",
                "count": len(related_hints),
                "hints": to_primitive(related_hints),
                "repairable_overlap_count": sum(
                    1 for result, repairable, _code in per_result if repairable
                ),
            })
        return RepairabilityDecision(
            bool(repairable_count),
            reason_code,
            tuple(item["requirement_id"] for item in diagnostics if "requirement_id" in item),
            _candidate_refs(results),
            tuple(diagnostics),
        )


__all__ = ["RepairabilityDecision", "RepairabilityGate"]
