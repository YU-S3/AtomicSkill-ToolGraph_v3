"""Public evidence assessment for declared Runtime DataFlow obligations.

This module only interprets verified contracts, current validated outputs, and
public semantic facts.  It never invents a graph edge or chooses a binding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import AbstractAtomicSkill, SemanticPredicate


@dataclass(frozen=True)
class OutputObligationAssessment:
    relation_predicate: str = ""
    effect_domain: str = ""
    relevant_anchor_roles: tuple[str, ...] = ()
    public_relation_status: str = "unknown"
    public_evidence_refs: tuple[str, ...] = ()


_FORMAL_ANCHOR_SOURCES = frozenset({
    "task",
    "runtime_plan",
    "data_flow",
    "repeat",
    "tool_output",
})
_PUBLIC_REVISION_PATTERN = re.compile(r":revision:(\d+)(?::|$)")


def _source_role(value: Any) -> str:
    if isinstance(value, BindingExpression):
        return (
            str(value.source_role or "")
            if value.kind is BindingExprKind.SKILL_INPUT
            else ""
        )
    if isinstance(value, Mapping) and value.get("kind"):
        try:
            return _source_role(BindingExpression.from_dict(dict(value)))
        except (KeyError, TypeError, ValueError):
            return ""
    return ""


def _relevant_predicates(
    atomic: AbstractAtomicSkill,
    consumer_input_role: str,
) -> list[SemanticPredicate]:
    return [
        predicate
        for predicate in [*atomic.preconditions, *atomic.effects]
        if any(
            _source_role(argument) == consumer_input_role
            for argument in predicate.args.values()
        )
    ]


def _public_fact_revision(fact: Mapping[str, Any]) -> int | None:
    raw_revision = fact.get("observed_at_revision", fact.get("revision"))
    if raw_revision is not None and not isinstance(raw_revision, bool):
        try:
            return int(raw_revision)
        except (TypeError, ValueError):
            return None
    public_ref = str(fact.get("public_evidence_ref", ""))
    match = _PUBLIC_REVISION_PATTERN.search(public_ref)
    return int(match.group(1)) if match is not None else None


def _predicate_key(predicate: SemanticPredicate) -> tuple[Any, ...]:
    return (
        str(predicate.predicate),
        str(predicate.effect_domain.value),
        tuple(sorted(
            (str(argument_role), _source_role(expression))
            for argument_role, expression in predicate.args.items()
        )),
        max(1, int(predicate.cardinality)),
        str(predicate.distinct_by),
    )


def _assess_predicate(
    *,
    predicate: SemanticPredicate,
    consumer_input_role: str,
    known_anchors: Mapping[str, Mapping[str, Any]],
    producer_output_value: Any,
    public_facts: tuple[Mapping[str, Any], ...],
    revision: int,
) -> OutputObligationAssessment:
    role_by_argument = {
        str(argument_role): _source_role(expression)
        for argument_role, expression in predicate.args.items()
    }
    relevant_anchor_roles = tuple(sorted({
        role
        for role in role_by_argument.values()
        if role and role != consumer_input_role and role in known_anchors
    }))

    current_public_facts: list[tuple[Mapping[str, Any], str]] = []
    for fact in public_facts:
        public_ref = str(fact.get("public_evidence_ref", "")).strip()
        if not public_ref:
            # Validator-only truth is never a policy-facing relation fact.
            continue
        if _public_fact_revision(fact) != revision:
            # A stale location is a historical clue, not current authority.
            continue
        if str(fact.get("predicate", "")) != predicate.predicate:
            continue
        current_public_facts.append((fact, public_ref))

    supported_refs: list[str] = []
    if producer_output_value not in (None, ""):
        for fact, public_ref in current_public_facts:
            arguments = dict(fact.get("args") or {})
            for argument_role, contract_role in role_by_argument.items():
                if contract_role == consumer_input_role:
                    expected = producer_output_value
                elif contract_role in known_anchors:
                    expected = known_anchors[contract_role].get("value")
                else:
                    continue
                if arguments.get(argument_role) != expected:
                    break
            else:
                supported_refs.append(public_ref)

    contradicted_refs: list[str] = []
    if (
        not supported_refs
        and producer_output_value not in (None, "")
        and str(predicate.predicate) == "object.at_location"
        and role_by_argument.get("location") == consumer_input_role
    ):
        object_contract_role = role_by_argument.get("object", "")
        object_anchor = known_anchors.get(object_contract_role, {})
        object_identity = object_anchor.get("value")
        anchor_source = str(object_anchor.get("source", ""))
        if (
            object_contract_role
            and object_identity not in (None, "")
            and anchor_source in _FORMAL_ANCHOR_SOURCES
        ):
            for fact, public_ref in current_public_facts:
                arguments = dict(fact.get("args") or {})
                public_object = arguments.get("object")
                public_location = arguments.get("location")
                # Literal equality is intentional: a semantic class such as
                # ``pencil`` is not the exact identity ``pencil_3``.
                if (
                    public_object == object_identity
                    and public_location not in (None, "")
                    and public_location != producer_output_value
                ):
                    contradicted_refs.append(public_ref)

    if supported_refs:
        status = "supported"
        evidence_refs = supported_refs
    elif contradicted_refs:
        status = "contradicted"
        evidence_refs = contradicted_refs
    else:
        status = "unknown"
        evidence_refs = []

    return OutputObligationAssessment(
        relation_predicate=str(predicate.predicate),
        effect_domain=str(predicate.effect_domain.value),
        relevant_anchor_roles=relevant_anchor_roles,
        public_relation_status=status,
        public_evidence_refs=tuple(sorted(set(evidence_refs))),
    )


def assess_output_obligation(
    *,
    consumer_atomic: AbstractAtomicSkill,
    consumer_input_role: str,
    known_anchors: Mapping[str, Mapping[str, Any]],
    producer_output_value: Any = None,
    public_facts: Iterable[Mapping[str, Any]] = (),
    revision: int = 0,
) -> OutputObligationAssessment:
    """Assess one real DataFlow edge without turning unknown into rejection."""

    predicates = _relevant_predicates(
        consumer_atomic,
        str(consumer_input_role),
    )
    if not predicates:
        return OutputObligationAssessment()
    try:
        current_revision = -1 if isinstance(revision, bool) else int(revision)
    except (TypeError, ValueError):
        current_revision = -1
    facts = tuple(public_facts)
    assessments = [
        (
            _assess_predicate(
                predicate=predicate,
                consumer_input_role=str(consumer_input_role),
                known_anchors=known_anchors,
                producer_output_value=producer_output_value,
                public_facts=facts,
                revision=current_revision,
            ),
            _predicate_key(predicate),
        )
        for predicate in predicates
    ]
    # Preserve the single-obligation projection while choosing the relation
    # with current public support first, then the most formal known anchors,
    # and finally a contract-derived stable key.
    assessments.sort(key=lambda item: (
        0 if item[0].public_relation_status == "supported" else 1,
        -len(item[0].relevant_anchor_roles),
        item[1],
    ))
    return assessments[0][0]


__all__ = ["OutputObligationAssessment", "assess_output_obligation"]
