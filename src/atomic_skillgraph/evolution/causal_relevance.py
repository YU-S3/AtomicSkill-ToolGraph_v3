"""Validate that an extracted Composite is the minimal task-causal closure."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import SemanticPredicate, TaskContract
from ..core.edges import GraphEdge, GraphEdgeType
from ..validation.contract_matcher import ContractMatcher
from .atomicizer import CanonicalAtomicOccurrence


def _value_key(value: Any) -> str:
    return repr(value)


def resolved_args(
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


def contract_covered(
    contract: TaskContract,
    occurrences: Iterable[CanonicalAtomicOccurrence],
    matcher: ContractMatcher,
) -> bool:
    canonical = list(occurrences)
    if not contract.target_effects:
        return False
    offered = [
        (effect, occurrence, resolved_args(effect, occurrence))
        for occurrence in canonical
        for effect in occurrence.effects
    ]
    matches_by_target: list[list[int]] = []
    matched_offered: set[int] = set()
    for target in contract.target_effects:
        compatible = [
            index for index, item in enumerate(offered)
            if matcher.effect_covers_target(
                offered_predicate=item[0],
                offered_arguments=item[2],
                target_predicate=target,
            )
        ]
        needed = max(1, int(target.cardinality))
        if sum(
            max(1, int(offered[index][0].cardinality))
            for index in compatible
        ) < needed:
            return False
        if target.distinct_by:
            values = {
                _value_key(offered[index][2].get(target.distinct_by))
                for index in compatible
                if offered[index][2].get(target.distinct_by) not in (None, "")
            }
            if len(values) < needed:
                return False
        matches_by_target.append(compatible)
        matched_offered.update(compatible)

    for rule in contract.cardinality_constraints:
        predicate = str(rule.get("predicate", ""))
        count = max(1, int(rule.get("count", 1)))
        role = str(rule.get("distinct_by") or rule.get("role") or "")
        matching_indexes = {
            offered_index
            for target_index, target in enumerate(contract.target_effects)
            if target.predicate.casefold() == predicate.casefold()
            for offered_index in matches_by_target[target_index]
        }
        matching = [offered[index] for index in sorted(matching_indexes)]
        if sum(max(1, int(item[0].cardinality)) for item in matching) < count:
            return False
        if role and len({
            _value_key(item[2].get(role)) for item in matching
            if item[2].get(role) not in (None, "")
        }) < count:
            return False

    # Identity constraints are evaluated only on the value-sensitive witnesses
    # that covered a target.  An unrelated same-predicate fact must never make
    # a contract pass.
    relevant = [offered[index] for index in sorted(matched_offered)]
    for constraint in contract.identity_constraints:
        left = {
            _value_key(arguments.get(constraint.left_role))
            for _, _, arguments in relevant
            if arguments.get(constraint.left_role) not in (None, "")
        }
        right = {
            _value_key(arguments.get(constraint.right_role))
            for _, _, arguments in relevant
            if arguments.get(constraint.right_role) not in (None, "")
        }
        relation = getattr(constraint.relation, "value", constraint.relation)
        if relation == "same_as":
            if constraint.left_role == constraint.right_role:
                witness_sets = [
                    {
                        _value_key(
                            offered[index][2].get(constraint.left_role)
                        )
                        for index in compatible
                        if offered[index][2].get(constraint.left_role)
                        not in (None, "")
                    }
                    for target_index, compatible in enumerate(matches_by_target)
                    if compatible
                    and constraint.left_role
                    in contract.target_effects[target_index].args
                ]
                if len(witness_sets) > 1 and not set.intersection(*witness_sets):
                    return False
            elif left and right and not left.intersection(right):
                return False
        if relation == "distinct_from" and left and right and not any(
            left_value != right_value for left_value in left for right_value in right
        ):
            return False
    return True


def _same_fact(
    left: SemanticPredicate,
    left_occurrence: CanonicalAtomicOccurrence,
    right: SemanticPredicate,
    right_occurrence: CanonicalAtomicOccurrence,
) -> bool:
    return (
        left.predicate.casefold() == right.predicate.casefold()
        and resolved_args(left, left_occurrence) == resolved_args(right, right_occurrence)
    )


class CausalRelevanceValidator:
    """Compute roots from the task contract, then close backwards over facts."""

    def validate(
        self,
        *,
        control_sequence: list[str],
        canonical: list[CanonicalAtomicOccurrence],
        contract: TaskContract,
        contract_matcher: ContractMatcher,
        edges: list[GraphEdge],
    ) -> set[str]:
        by_id = {item.occurrence_id: item for item in canonical}
        if not control_sequence:
            raise ValueError("E2 control sequence must contain at least one occurrence")
        if len(control_sequence) != len(set(control_sequence)):
            raise ValueError("E2 control sequence selects one occurrence more than once")
        unknown = sorted(set(control_sequence) - set(by_id))
        if unknown:
            raise ValueError(f"E2 control sequence references unknown occurrences: {unknown}")
        chronological = sorted(
            control_sequence,
            key=lambda item: (
                by_id[item].event_start,
                by_id[item].event_end_exclusive,
                by_id[item].phase_id,
            ),
        )
        if control_sequence != chronological:
            raise ValueError("E2 control sequence is not in real trace chronology")

        roots = self._task_roots(canonical, contract, contract_matcher)
        needed = set(roots)
        queue = deque(sorted(
            roots,
            key=lambda item: by_id[item].event_start,
            reverse=True,
        ))
        incoming = {
            target: [edge for edge in edges if edge.target_step == target]
            for target in by_id
        }
        while queue:
            consumer_id = queue.popleft()
            consumer = by_id[consumer_id]
            for precondition in consumer.preconditions:
                producers = [
                    candidate
                    for candidate in canonical
                    if candidate.event_end_exclusive <= consumer.event_start
                    and any(
                        _same_fact(effect, candidate, precondition, consumer)
                        for effect in candidate.effects
                    )
                ]
                if not producers:
                    # The required fact may be supplied by initial state rather
                    # than by a transition in this trace.
                    continue
                producer = max(
                    producers,
                    key=lambda item: (
                        item.event_end_exclusive,
                        item.event_start,
                        item.phase_id,
                    ),
                )
                if producer.occurrence_id not in needed:
                    needed.add(producer.occurrence_id)
                    queue.append(producer.occurrence_id)
            for edge in incoming.get(consumer_id, []):
                if edge.edge_type is not GraphEdgeType.DATA_FLOW:
                    continue
                if edge.source_step not in needed:
                    needed.add(edge.source_step)
                    queue.append(edge.source_step)

        selected = set(control_sequence)
        if selected != needed:
            extra = sorted(selected - needed)
            missing = sorted(needed - selected)
            raise ValueError(
                "E2 control sequence differs from minimal task-causal closure: "
                f"extra={extra}, missing={missing}"
            )
        if not contract_covered(contract, (by_id[item] for item in control_sequence), contract_matcher):
            raise ValueError("E2 selected occurrences do not cover the value-sensitive TaskContract")
        return needed

    @staticmethod
    def _task_roots(
        canonical: list[CanonicalAtomicOccurrence],
        contract: TaskContract,
        matcher: ContractMatcher,
    ) -> set[str]:
        roots: set[str] = set()
        for target in contract.target_effects:
            matches = [
                (occurrence, effect, resolved_args(effect, occurrence))
                for occurrence in canonical
                for effect in occurrence.effects
                if matcher.effect_covers_target(
                    offered_predicate=effect,
                    offered_arguments=resolved_args(effect, occurrence),
                    target_predicate=target,
                )
            ]
            if not matches:
                raise ValueError(
                    f"no Atomic effect covers TaskContract predicate {target.predicate!r}"
                )
            needed = max(1, int(target.cardinality))
            if target.distinct_by:
                latest_by_value: dict[
                    str,
                    tuple[
                        CanonicalAtomicOccurrence,
                        SemanticPredicate,
                        dict[str, Any],
                    ],
                ] = {}
                for item in matches:
                    value = item[2].get(target.distinct_by)
                    if value in (None, ""):
                        continue
                    value_key = _value_key(value)
                    previous = latest_by_value.get(value_key)
                    if (
                        previous is None
                        or item[0].event_end_exclusive
                        > previous[0].event_end_exclusive
                    ):
                        latest_by_value[value_key] = item
                chosen = sorted(
                    latest_by_value.values(),
                    key=lambda item: item[0].event_end_exclusive,
                    reverse=True,
                )[:needed]
                if len(chosen) < needed:
                    raise ValueError(
                        "TaskContract cardinality is not covered for "
                        f"{target.predicate!r}"
                    )
                roots.update(item[0].occurrence_id for item in chosen)
                continue
            count = 0
            for occurrence, effect, _ in sorted(
                matches,
                key=lambda item: item[0].event_end_exclusive,
                reverse=True,
            ):
                roots.add(occurrence.occurrence_id)
                count += max(1, int(effect.cardinality))
                if count >= needed:
                    break
            if count < needed:
                raise ValueError(
                    "TaskContract cardinality is not covered for "
                    f"{target.predicate!r}"
                )
        if not roots:
            raise ValueError("TaskContract produced no causal roots")
        return roots


__all__ = ["CausalRelevanceValidator", "contract_covered", "resolved_args"]
