"""Contract-filtered lexical recall; embeddings may replace recall, never filtering."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..core.contracts import (
    AbstractAtomicSkill,
    AtomicContractCompatibilityReport,
    CapabilityRequirement,
    PredicateCompatibilityDetail,
    RequiredInputCompatibility,
    SemanticPredicate,
    TaskContract,
)
from ..core.semantic_types import (
    normalize_semantic_type,
    semantic_types_compatible,
)


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"[\w]+", text.casefold()))


def lexical_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    common = sum((a & b).values())
    return (2.0 * common) / (sum(a.values()) + sum(b.values()))


def predicate_compatible(required: SemanticPredicate, offered: SemanticPredicate) -> bool:
    if required.predicate.casefold() != offered.predicate.casefold():
        return False
    if required.effect_domain is not offered.effect_domain:
        return False
    if offered.cardinality < required.cardinality:
        return False
    required_keys = set(required.args)
    return required_keys.issubset(offered.args) or not required_keys


def effects_cover(required: Iterable[SemanticPredicate], offered: Iterable[SemanticPredicate]) -> bool:
    offered = list(offered)
    return all(any(predicate_compatible(item, candidate) for candidate in offered) for item in required)


def _diagnose_required_effect(
    required: SemanticPredicate,
    offered: Iterable[SemanticPredicate],
) -> PredicateCompatibilityDetail:
    required_roles = tuple(sorted(map(str, required.args)))
    required_role_set = set(required_roles)
    same_predicate = [
        candidate
        for candidate in offered
        if required.predicate.casefold() == candidate.predicate.casefold()
        and required.effect_domain is candidate.effect_domain
    ]
    if not same_predicate:
        return PredicateCompatibilityDetail(
            required_predicate=str(required.predicate),
            offered_predicate_found=False,
            required_argument_roles=required_roles,
            missing_argument_roles=required_roles,
            required_cardinality=int(required.cardinality),
            best_offered_cardinality=0,
            cardinality_sufficient=False,
        )

    # One offered Effect must satisfy both the role and cardinality contract.
    # Choose the closest deterministic witness for an explanatory report.
    best = sorted(
        same_predicate,
        key=lambda candidate: (
            -len(required_role_set.intersection(map(str, candidate.args))),
            -int(candidate.cardinality),
            tuple(sorted(map(str, candidate.args))),
        ),
    )[0]
    missing_roles = tuple(sorted(
        required_role_set.difference(map(str, best.args))
    ))
    return PredicateCompatibilityDetail(
        required_predicate=str(required.predicate),
        offered_predicate_found=True,
        required_argument_roles=required_roles,
        missing_argument_roles=missing_roles,
        required_cardinality=int(required.cardinality),
        best_offered_cardinality=int(best.cardinality),
        cardinality_sufficient=(
            int(best.cardinality) >= int(required.cardinality)
        ),
    )


def diagnose_atomic_contract_compatibility(
    requirement: CapabilityRequirement,
    atomic: AbstractAtomicSkill,
) -> AtomicContractCompatibilityReport:
    """Return the single deterministic Atomic compatibility decision/report."""

    effect_details = tuple(
        _diagnose_required_effect(required, atomic.effects)
        for required in requirement.desired_effects
    )
    input_details = tuple(
        RequiredInputCompatibility(
            required_name=str(required.name),
            required_semantic_type=normalize_semantic_type(
                required.semantic_type
            ),
            compatible_offered_roles=tuple(sorted(
                str(offered.name)
                for offered in atomic.inputs
                if semantic_types_compatible(
                    required.semantic_type,
                    offered.semantic_type,
                )
            )),
        )
        for required in requirement.expected_inputs
        if required.required
    )
    effects_passed = all(
        detail.offered_predicate_found
        and not detail.missing_argument_roles
        and detail.cardinality_sufficient
        for detail in effect_details
    )
    inputs_passed = all(
        bool(detail.compatible_offered_roles)
        for detail in input_details
    )
    failure_codes: list[str] = []
    if any(not detail.offered_predicate_found for detail in effect_details):
        failure_codes.append("atomic_effect_predicate_missing")
    if any(
        detail.offered_predicate_found and detail.missing_argument_roles
        for detail in effect_details
    ):
        failure_codes.append("atomic_effect_argument_role_missing")
    if any(
        detail.offered_predicate_found
        and not detail.cardinality_sufficient
        for detail in effect_details
    ):
        failure_codes.append("atomic_effect_cardinality_insufficient")
    if not inputs_passed:
        failure_codes.append("atomic_required_input_type_unavailable")
    return AtomicContractCompatibilityReport(
        passed=effects_passed and inputs_passed,
        effects_passed=effects_passed,
        inputs_passed=inputs_passed,
        effect_details=effect_details,
        input_details=input_details,
        missing_required_input_types=tuple(sorted({
            detail.required_semantic_type
            for detail in input_details
            if not detail.compatible_offered_roles
        })),
        failure_codes=tuple(failure_codes),
    )


def atomic_contract_compatible(
    requirement: CapabilityRequirement,
    atomic: AbstractAtomicSkill,
) -> bool:
    return diagnose_atomic_contract_compatibility(requirement, atomic).passed


def task_contract_compatible(required: TaskContract, offered: TaskContract) -> bool:
    if required.target_effects and not effects_cover(required.target_effects, offered.target_effects):
        return False
    for wanted in required.cardinality_constraints:
        wanted_predicate = str(wanted.get("predicate", ""))
        wanted_role = str(wanted.get("distinct_by") or wanted.get("role") or "")
        wanted_count = int(wanted.get("count", 1))
        if not any(
            str(candidate.get("predicate", "")) == wanted_predicate
            and str(candidate.get("distinct_by") or candidate.get("role") or "") == wanted_role
            and int(candidate.get("count", 1)) >= wanted_count
            for candidate in offered.cardinality_constraints
        ):
            return False
    required_identity = {
        (item.left_role, item.relation.value, item.right_role, item.scope)
        for item in required.identity_constraints
    }
    offered_identity = {
        (item.left_role, item.relation.value, item.right_role, item.scope)
        for item in offered.identity_constraints
    }
    return required_identity.issubset(offered_identity)


@dataclass(frozen=True)
class CompleteCompositeContractDiagnosis:
    passed: bool
    target_effect_missing: tuple[tuple[object, ...], ...]
    target_effect_extra: tuple[tuple[object, ...], ...]
    cardinality_missing: tuple[tuple[object, ...], ...]
    cardinality_extra: tuple[tuple[object, ...], ...]
    identity_missing: tuple[tuple[str, str, str, str], ...]
    identity_extra: tuple[tuple[str, str, str, str], ...]
    failure_codes: tuple[str, ...]


def complete_composite_contract_diagnosis(
    required: TaskContract,
    offered: TaskContract,
) -> CompleteCompositeContractDiagnosis:
    """Diagnose the one exact contract comparison used by the P0 route."""

    # Local import avoids making the generic query module own Planner IR.
    from ..planner.multiplicity import normalize_task_contract

    try:
        required = normalize_task_contract(required)
        offered = normalize_task_contract(offered)
    except (TypeError, ValueError):
        return CompleteCompositeContractDiagnosis(
            passed=False,
            target_effect_missing=(),
            target_effect_extra=(),
            cardinality_missing=(),
            cardinality_extra=(),
            identity_missing=(),
            identity_extra=(),
            failure_codes=("composite_contract_normalization_failed",),
        )

    def target_multiset(contract: TaskContract) -> Counter[tuple[object, ...]]:
        return Counter(
            (
                effect.predicate.casefold(),
                tuple(sorted(map(str, effect.args))),
                int(effect.cardinality),
                str(effect.distinct_by),
                effect.effect_domain.value,
            )
            for effect in contract.target_effects
        )

    def cardinality_multiset(contract: TaskContract) -> Counter[tuple[object, ...]]:
        return Counter(
            (
                str(value.get("predicate", "")).casefold(),
                int(value.get("count", 1)),
                str(value.get("distinct_by", "")),
                tuple(sorted(map(str, value.get("shared_roles", ())))),
                str(value.get("composition_mode", "atomic")),
            )
            for value in contract.cardinality_constraints
        )

    def identities(contract: TaskContract) -> Counter[tuple[str, str, str, str]]:
        return Counter(
            (
                item.left_role,
                item.relation.value,
                item.right_role,
                item.scope,
            )
            for item in contract.identity_constraints
        )

    required_targets = target_multiset(required)
    offered_targets = target_multiset(offered)
    required_cardinality = cardinality_multiset(required)
    offered_cardinality = cardinality_multiset(offered)
    required_identity = identities(required)
    offered_identity = identities(offered)
    target_effect_missing = tuple(sorted(
        (required_targets - offered_targets).elements()
    ))
    target_effect_extra = tuple(sorted(
        (offered_targets - required_targets).elements()
    ))
    cardinality_missing = tuple(sorted(
        (required_cardinality - offered_cardinality).elements()
    ))
    cardinality_extra = tuple(sorted(
        (offered_cardinality - required_cardinality).elements()
    ))
    identity_missing = tuple(sorted(
        (required_identity - offered_identity).elements()
    ))
    identity_extra = tuple(sorted(
        (offered_identity - required_identity).elements()
    ))
    differences = (
        (target_effect_missing, "composite_target_effect_missing"),
        (target_effect_extra, "composite_target_effect_extra"),
        (cardinality_missing, "composite_cardinality_missing"),
        (cardinality_extra, "composite_cardinality_extra"),
        (identity_missing, "composite_identity_missing"),
        (identity_extra, "composite_identity_extra"),
    )
    failure_codes = tuple(code for values, code in differences if values)
    return CompleteCompositeContractDiagnosis(
        passed=not failure_codes,
        target_effect_missing=target_effect_missing,
        target_effect_extra=target_effect_extra,
        cardinality_missing=cardinality_missing,
        cardinality_extra=cardinality_extra,
        identity_missing=identity_missing,
        identity_extra=identity_extra,
        failure_codes=failure_codes,
    )


def complete_composite_contract_compatible(
    required: TaskContract,
    offered: TaskContract,
) -> bool:
    """Return the P0 exact-match decision from its diagnostic authority."""

    return complete_composite_contract_diagnosis(required, offered).passed
