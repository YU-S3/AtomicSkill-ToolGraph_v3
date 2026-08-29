"""Contract-filtered lexical recall; embeddings may replace recall, never filtering."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from ..core.contracts import AbstractAtomicSkill, CapabilityRequirement, SemanticPredicate, TaskContract


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
    if offered.cardinality < required.cardinality:
        return False
    required_keys = set(required.args)
    return required_keys.issubset(offered.args) or not required_keys


def effects_cover(required: Iterable[SemanticPredicate], offered: Iterable[SemanticPredicate]) -> bool:
    offered = list(offered)
    return all(any(predicate_compatible(item, candidate) for candidate in offered) for item in required)


def atomic_contract_compatible(requirement: CapabilityRequirement, atomic: AbstractAtomicSkill) -> bool:
    if requirement.desired_effects and not effects_cover(requirement.desired_effects, atomic.effects):
        return False
    wanted_inputs = {item.semantic_type for item in requirement.expected_inputs if item.required}
    available_inputs = {item.semantic_type for item in atomic.inputs}
    return not wanted_inputs or wanted_inputs.issubset(available_inputs)


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
