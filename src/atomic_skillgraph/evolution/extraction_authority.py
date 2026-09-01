"""Code-authoritative TaskContract coverage for success extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import (
    IdentityRelation,
    SemanticPredicate,
    TaskContract,
)
from ..validation.contract_matcher import ContractMatcher

if TYPE_CHECKING:
    from .atomicizer import CanonicalAtomicOccurrence


@dataclass(frozen=True)
class TargetWitnessAuthority:
    """Accepted trace facts that code matched to one formal target."""

    target_index: int
    predicate: str
    required_argument_roles: tuple[str, ...]
    required_cardinality: int
    distinct_by: str
    witness_event_indexes: tuple[int, ...]
    witness_facts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExtractionCoverageAuthority:
    """Task targets witnessed by accepted, state-derived trace facts."""

    targets: tuple[TargetWitnessAuthority, ...]
    all_targets_witnessed: bool


@dataclass(frozen=True)
class ContractCoverageReport:
    """The sole deterministic TaskContract coverage report for E1 and E2."""

    passed: bool
    target_checks: tuple[dict[str, Any], ...]
    cardinality_checks: tuple[dict[str, Any], ...]
    identity_checks: tuple[dict[str, Any], ...]
    failure_codes: tuple[str, ...]


def extraction_coverage_authority(
    normalized_trace: dict[str, Any],
    contract: TaskContract,
    matcher: ContractMatcher,
) -> ExtractionCoverageAuthority:
    """Locate formal target witnesses without deriving facts from prose.

    ``authoritative_positive_effects`` is produced by the trace normalizer's
    accepted-action state reducer.  This function only matches and groups
    those facts; it never creates an Atomic occurrence or a semantic Effect.
    """

    targets: list[TargetWitnessAuthority] = []
    all_witnessed = bool(contract.target_effects)
    for target_index, target in enumerate(contract.target_effects):
        matched: list[tuple[int, dict[str, Any]]] = []
        for fallback_index, action in enumerate(normalized_trace.get("actions", ())):
            if not action.get("accepted"):
                continue
            event_index = int(action.get("event_index", fallback_index))
            for raw_fact in action.get("authoritative_positive_effects", ()):
                fact = dict(raw_fact)
                arguments = dict(fact.get("args") or {})
                offered = SemanticPredicate(
                    str(fact.get("predicate", "")),
                    arguments,
                    max(1, int(fact.get("cardinality", 1))),
                    str(fact.get("distinct_by", "")),
                )
                if matcher.covers(target, offered, arguments):
                    matched.append((event_index, fact))

        # A state fact may be projected more than once by a richer adapter.
        # Preserve one code fact per event/fact identity deterministically.
        unique: dict[tuple[int, str, str], tuple[int, dict[str, Any]]] = {}
        for event_index, fact in matched:
            key = (
                event_index,
                str(fact.get("predicate", "")),
                repr(sorted(dict(fact.get("args") or {}).items())),
            )
            unique.setdefault(key, (event_index, fact))
        ordered = [unique[key] for key in sorted(unique)]
        needed = max(1, int(target.cardinality))
        if target.distinct_by:
            distinct_values = {
                dict(fact.get("args") or {}).get(target.distinct_by)
                for _, fact in ordered
                if dict(fact.get("args") or {}).get(target.distinct_by)
                not in {None, ""}
            }
            witnessed = len(distinct_values) >= needed
        else:
            witnessed = len(ordered) >= needed
        all_witnessed = all_witnessed and witnessed
        targets.append(TargetWitnessAuthority(
            target_index=target_index,
            predicate=target.predicate,
            required_argument_roles=tuple(sorted(map(str, target.args))),
            required_cardinality=needed,
            distinct_by=str(target.distinct_by),
            witness_event_indexes=tuple(sorted({item[0] for item in ordered})),
            witness_facts=tuple(dict(item[1]) for item in ordered),
        ))
    return ExtractionCoverageAuthority(tuple(targets), all_witnessed)


def _resolved_predicate_args(
    predicate: SemanticPredicate,
    occurrence: CanonicalAtomicOccurrence,
) -> dict[str, Any]:
    bindings = {**occurrence.input_bindings, **occurrence.output_bindings}
    result: dict[str, Any] = {}
    for name, raw in predicate.args.items():
        if isinstance(raw, dict) and "kind" in raw:
            raw = BindingExpression.from_dict(raw)
        if isinstance(raw, BindingExpression):
            result[name] = (
                raw.constant
                if raw.kind is BindingExprKind.CONSTANT
                else bindings.get(raw.source_role)
            )
        else:
            result[name] = raw
    return result


def contract_coverage_report(
    contract: TaskContract,
    canonical_occurrences: list[CanonicalAtomicOccurrence],
    matcher: ContractMatcher,
) -> ContractCoverageReport:
    """Report whether validated E1 occurrences cover the formal contract.

    CompositeBuilder calls this same function, so E1 admission and E2 graph
    admission cannot diverge on target/cardinality/identity semantics.
    """

    failures: list[str] = []
    offered = [
        (effect, occurrence, _resolved_predicate_args(effect, occurrence))
        for occurrence in canonical_occurrences
        for effect in occurrence.effects
    ]
    matches_by_target: list[
        list[tuple[SemanticPredicate, CanonicalAtomicOccurrence, dict[str, Any]]]
    ] = []
    target_checks: list[dict[str, Any]] = []

    if not contract.target_effects:
        failures.append("extractor_contract_target_missing")

    for target_index, target in enumerate(contract.target_effects):
        compatible = [
            item
            for item in offered
            if matcher.covers(target, item[0], item[2])
        ]
        matches_by_target.append(compatible)
        required = max(1, int(target.cardinality))
        offered_cardinality = sum(
            max(1, int(effect.cardinality))
            for effect, _, _ in compatible
        )
        cardinality_passed = offered_cardinality >= required
        distinct_values: set[Any] = set()
        if target.distinct_by:
            for _, _, arguments in compatible:
                value = arguments.get(target.distinct_by)
                if value not in {None, ""}:
                    try:
                        distinct_values.add(value)
                    except TypeError:
                        distinct_values.add(repr(value))
        distinctness_passed = (
            not target.distinct_by or len(distinct_values) >= required
        )
        if not compatible:
            failures.append("extractor_contract_target_uncovered")
        if not cardinality_passed:
            failures.append(
                "extractor_contract_target_cardinality_insufficient"
            )
        if not distinctness_passed:
            failures.append(
                "extractor_contract_target_distinctness_insufficient"
            )
        target_checks.append({
            "target_index": target_index,
            "predicate": target.predicate,
            "required_argument_roles": sorted(map(str, target.args)),
            "required_cardinality": required,
            "matched_occurrence_ids": sorted({
                occurrence.occurrence_id
                for _, occurrence, _ in compatible
            }),
            "matched_effect_count": len(compatible),
            "offered_cardinality": offered_cardinality,
            "distinct_by": target.distinct_by,
            "distinct_value_count": len(distinct_values),
            "passed": bool(
                compatible and cardinality_passed and distinctness_passed
            ),
        })

    matched_offered = [
        item for matches in matches_by_target for item in matches
    ]
    cardinality_checks: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(contract.cardinality_constraints):
        predicate = str(rule.get("predicate", ""))
        count = max(1, int(rule.get("count", 1)))
        role = str(rule.get("distinct_by") or rule.get("role") or "")
        matching = [
            item
            for item in matched_offered
            if item[0].predicate.casefold() == predicate.casefold()
        ]
        offered_cardinality = sum(
            max(1, int(item[0].cardinality)) for item in matching
        )
        distinct_values: set[Any] = set()
        if role:
            for item in matching:
                value = item[2].get(role)
                if value not in {None, ""}:
                    try:
                        distinct_values.add(value)
                    except TypeError:
                        distinct_values.add(repr(value))
        passed = offered_cardinality >= count and (
            not role or len(distinct_values) >= count
        )
        if not passed:
            failures.append(
                "extractor_contract_cardinality_constraint_unsatisfied"
            )
        cardinality_checks.append({
            "constraint_index": rule_index,
            "predicate": predicate,
            "required_count": count,
            "distinct_by": role,
            "offered_cardinality": offered_cardinality,
            "distinct_value_count": len(distinct_values),
            "passed": passed,
        })

    identity_checks: list[dict[str, Any]] = []
    for constraint_index, constraint in enumerate(
        contract.identity_constraints
    ):
        passed = False
        if constraint.relation is IdentityRelation.SAME_AS:
            if constraint.left_role == constraint.right_role:
                witness_sets = [
                    {
                        item[2][constraint.left_role]
                        for item in matches
                        if constraint.left_role in item[2]
                    }
                    for matches in matches_by_target
                ]
                witness_sets = [values for values in witness_sets if values]
                passed = bool(witness_sets) and (
                    len(witness_sets) == 1
                    or bool(set.intersection(*witness_sets))
                )
            else:
                left = {
                    item[2][constraint.left_role]
                    for item in matched_offered
                    if constraint.left_role in item[2]
                }
                right = {
                    item[2][constraint.right_role]
                    for item in matched_offered
                    if constraint.right_role in item[2]
                }
                passed = bool(left and right and left.intersection(right))
        elif constraint.relation is IdentityRelation.DISTINCT_FROM:
            left = {
                item[2][constraint.left_role]
                for item in matched_offered
                if constraint.left_role in item[2]
            }
            right = {
                item[2][constraint.right_role]
                for item in matched_offered
                if constraint.right_role in item[2]
            }
            passed = bool(
                left
                and right
                and any(
                    left_value != right_value
                    for left_value in left
                    for right_value in right
                )
            )
        if not passed:
            failures.append(
                "extractor_contract_identity_constraint_unsatisfied"
            )
        identity_checks.append({
            "constraint_index": constraint_index,
            "left_role": constraint.left_role,
            "relation": constraint.relation.value,
            "right_role": constraint.right_role,
            "scope": constraint.scope,
            "passed": passed,
        })

    failure_codes = tuple(dict.fromkeys(failures))
    return ContractCoverageReport(
        passed=not failure_codes,
        target_checks=tuple(target_checks),
        cardinality_checks=tuple(cardinality_checks),
        identity_checks=tuple(identity_checks),
        failure_codes=failure_codes,
    )


__all__ = [
    "ContractCoverageReport",
    "ExtractionCoverageAuthority",
    "TargetWitnessAuthority",
    "contract_coverage_report",
    "extraction_coverage_authority",
]
