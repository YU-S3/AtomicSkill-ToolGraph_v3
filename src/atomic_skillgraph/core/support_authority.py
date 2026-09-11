"""Deterministic semantic authority for Support-Atomic role mappings.

Support retrieval is allowed to close a formal value gap; it is not allowed
to infer workflow semantics merely because two boundaries share a broad type
such as ``entity``.  This module is the single code-owned authority used by
both Planner and Runtime retrieval and by Planner validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .bindings import BindingExprKind, BindingExpression, resolution_satisfies
from .contracts import AbstractAtomicSkill, EffectDomain
from .semantic_types import normalize_semantic_type, semantic_types_compatible


def _role_alias(value: Any) -> str:
    """Return the stable lexical form of one code-owned boundary role."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _referenced_role(raw: Any) -> str:
    expression: BindingExpression | None = None
    if isinstance(raw, BindingExpression):
        expression = raw
    elif isinstance(raw, Mapping) and "kind" in raw:
        try:
            expression = BindingExpression.from_dict(dict(raw))
        except (KeyError, TypeError, ValueError):
            return ""
    if expression is not None:
        if expression.kind is BindingExprKind.SKILL_INPUT:
            return str(expression.source_role)
        return ""
    if isinstance(raw, str) and raw.startswith("$"):
        return raw[1:]
    return ""


def input_identity_source_role(atomic: Any, output_role: str) -> str:
    """Resolve an output's explicit identity-preserving input role.

    A non-empty current ``output_derivations`` namespace is authoritative for
    every output.  Legacy ``output_identity`` is consulted only when that
    namespace is absent, matching Runtime Atomic validation semantics.
    """

    validator_spec = dict(getattr(atomic, "validator_spec", {}) or {})
    try:
        derivations = dict(validator_spec.get("output_derivations") or {})
    except (TypeError, ValueError):
        return ""
    if derivations:
        raw = derivations.get(str(output_role))
        if not isinstance(raw, Mapping):
            return ""
        if str(raw.get("kind", "")) != "input_identity":
            return ""
        input_role = str(raw.get("input_role", ""))
        return input_role if input_role else ""
    legacy = [
        item
        for item in list(validator_spec.get("output_identity") or ())
        if isinstance(item, Mapping)
        and str(item.get("output_role", "")) == str(output_role)
        and str(item.get("input_role", ""))
    ]
    return str(legacy[0]["input_role"]) if len(legacy) == 1 else ""


def _predicate_argument_aliases(
    predicates: Any,
    referenced_boundary_roles: set[str],
) -> set[str]:
    aliases: set[str] = set()
    for predicate in list(predicates or ()):
        for argument_role, raw in dict(
            getattr(predicate, "args", {}) or {},
        ).items():
            if _referenced_role(raw) in referenced_boundary_roles:
                alias = _role_alias(argument_role)
                if alias:
                    aliases.add(alias)
    return aliases


def producer_output_aliases(
    atomic: AbstractAtomicSkill,
    output_role: str,
) -> frozenset[str]:
    """Build code-owned aliases for one producer output boundary."""

    output_names = {str(item.name) for item in atomic.outputs}
    if str(output_role) not in output_names:
        return frozenset()
    identity_input = input_identity_source_role(atomic, output_role)
    referenced_roles = {str(output_role)}
    if identity_input:
        referenced_roles.add(identity_input)
    aliases = {
        alias
        for alias in (
            _role_alias(output_role),
            _role_alias(identity_input) if identity_input else "",
        )
        if alias
    }
    aliases.update(_predicate_argument_aliases(
        atomic.effects,
        referenced_roles,
    ))
    return frozenset(aliases)


def consumer_input_aliases(
    atomic: AbstractAtomicSkill,
    input_role: str,
) -> frozenset[str]:
    """Build code-owned aliases for one consumer input boundary."""

    input_names = {str(item.name) for item in atomic.inputs}
    if str(input_role) not in input_names:
        return frozenset()
    aliases = {_role_alias(input_role)}
    aliases.update(_predicate_argument_aliases(
        (*atomic.preconditions, *atomic.effects),
        {str(input_role)},
    ))
    return frozenset(item for item in aliases if item)


def output_resolution_authority(
    atomic: AbstractAtomicSkill,
    output_role: str,
) -> tuple[str, str]:
    """Return the strongest declared resolution and Effect domain."""

    output = next(
        (item for item in atomic.outputs if str(item.name) == str(output_role)),
        None,
    )
    if output is None:
        return "", ""
    identity_input = input_identity_source_role(atomic, output_role)
    referenced_roles = {str(output_role)}
    if identity_input:
        referenced_roles.add(identity_input)
    domains = {
        str(effect.effect_domain.value)
        for effect in atomic.effects
        if any(
            _referenced_role(raw) in referenced_roles
            for raw in dict(effect.args or {}).values()
        )
    }
    if EffectDomain.EVIDENCE.value in domains:
        return "relation_verified", EffectDomain.EVIDENCE.value
    if EffectDomain.WORLD.value in domains:
        return "concrete", EffectDomain.WORLD.value
    return str(output.required_resolution), ""


@dataclass(frozen=True)
class SupportRoleAuthority:
    authorized: bool
    reason: str
    semantic_type: str = ""
    producer_resolution: str = ""
    required_resolution: str = ""
    effect_domain: str = ""
    producer_aliases: tuple[str, ...] = ()
    consumer_aliases: tuple[str, ...] = ()
    semantic_alias_authorized: bool = False
    relation_verified_exception: bool = False


def support_role_authority(
    producer: AbstractAtomicSkill,
    producer_role: str,
    consumer: AbstractAtomicSkill,
    consumer_role: str,
) -> SupportRoleAuthority:
    """Prove one producer-output to consumer-input Support mapping.

    Ordinary concrete/world support requires a code-owned semantic alias.
    The only alias-free exception is evidence-domain support for an input
    whose declared resolution is ``relation_verified``.
    """

    output = next(
        (item for item in producer.outputs if str(item.name) == str(producer_role)),
        None,
    )
    required = next(
        (item for item in consumer.inputs if str(item.name) == str(consumer_role)),
        None,
    )
    if output is None or required is None:
        return SupportRoleAuthority(False, "boundary_role_missing")

    semantic_type = normalize_semantic_type(
        output.semantic_type or required.semantic_type,
    )
    producer_resolution, effect_domain = output_resolution_authority(
        producer,
        producer_role,
    )
    required_resolution = str(required.required_resolution)
    type_compatible = semantic_types_compatible(
        required.semantic_type,
        output.semantic_type,
    )
    try:
        resolution_compatible = resolution_satisfies(
            producer_resolution,
            required_resolution,
        )
    except (KeyError, TypeError, ValueError):
        resolution_compatible = False
    producer_alias_set = producer_output_aliases(producer, producer_role)
    consumer_alias_set = consumer_input_aliases(consumer, consumer_role)
    alias_authorized = bool(producer_alias_set & consumer_alias_set)
    relation_exception = bool(
        required_resolution == "relation_verified"
        and effect_domain == EffectDomain.EVIDENCE.value
    )

    if not type_compatible:
        reason = "semantic_type_incompatible"
    elif not resolution_compatible:
        reason = "resolution_incompatible"
    elif not alias_authorized and not relation_exception:
        reason = "semantic_role_authority_missing"
    else:
        reason = "authorized"
    return SupportRoleAuthority(
        authorized=reason == "authorized",
        reason=reason,
        semantic_type=semantic_type,
        producer_resolution=producer_resolution,
        required_resolution=required_resolution,
        effect_domain=effect_domain,
        producer_aliases=tuple(sorted(producer_alias_set)),
        consumer_aliases=tuple(sorted(consumer_alias_set)),
        semantic_alias_authorized=alias_authorized,
        relation_verified_exception=relation_exception,
    )


__all__ = [
    "SupportRoleAuthority",
    "consumer_input_aliases",
    "input_identity_source_role",
    "output_resolution_authority",
    "producer_output_aliases",
    "support_role_authority",
]
