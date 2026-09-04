"""Generic Runtime support-Atomic retrieval.

Only the blocked Atomic contract is used: missing input roles are matched
against other normal Atomic outputs/effects.  No task type, object family, or
benchmark workflow may enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.bindings import (
    BindingExprKind,
    BindingExpression,
    resolution_satisfies,
)
from ..core.contracts import AbstractAtomicSkill, EffectDomain
from ..core.serialization import to_primitive
from ..core.semantic_types import (
    normalize_semantic_type,
    semantic_types_compatible,
)


@dataclass(frozen=True)
class SupportRoleMapping:
    producer_role: str
    consumer_role: str
    semantic_type: str
    producer_resolution: str = "semantic"
    required_resolution: str = "semantic"
    effect_domain: str = ""


@dataclass(frozen=True)
class SupportCandidate:
    atomic_ref: str
    score: float
    supplied_roles: tuple[str, ...]
    output_roles: tuple[str, ...]
    effect_predicates: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]
    role_mappings: tuple[SupportRoleMapping, ...] = ()


def _predicate_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("predicate", ""))
    return str(getattr(value, "predicate", ""))


def _referenced_roles(value: Any) -> set[str]:
    roles: set[str] = set()
    for raw in dict(getattr(value, "args", {}) or {}).values():
        expression: BindingExpression | None = None
        if isinstance(raw, BindingExpression):
            expression = raw
        elif isinstance(raw, Mapping) and "kind" in raw:
            try:
                expression = BindingExpression.from_dict(dict(raw))
            except (KeyError, TypeError, ValueError):
                expression = None
        if expression is not None:
            if expression.kind is BindingExprKind.SKILL_INPUT:
                roles.add(str(expression.source_role))
            continue
        if isinstance(raw, str) and raw.startswith("$"):
            roles.add(raw[1:])
    return roles


def _output_authority(
    atomic: AbstractAtomicSkill,
    output_role: str,
    declared_resolution: str,
) -> tuple[str, str]:
    """Return the strongest contract-declared output authority."""

    domains = {
        str(effect.effect_domain.value)
        for effect in atomic.effects
        if output_role in _referenced_roles(effect)
    }
    if EffectDomain.EVIDENCE.value in domains:
        return "relation_verified", EffectDomain.EVIDENCE.value
    if EffectDomain.WORLD.value in domains:
        return "concrete", EffectDomain.WORLD.value
    return str(declared_resolution), ""


class SupportAtomicRetriever:
    """Return formal-compatible support candidates, never workflow choices."""

    def retrieve(
        self,
        *,
        blocked_atomic: AbstractAtomicSkill,
        missing_roles: Iterable[str],
        atomics: Iterable[AbstractAtomicSkill],
        top_k: int = 3,
    ) -> list[SupportCandidate]:
        missing = {str(role) for role in missing_roles}
        if not missing:
            return []
        blocked_inputs = {str(item.name): item for item in blocked_atomic.inputs}
        candidates: list[SupportCandidate] = []
        for atomic in atomics:
            if str(atomic.ref) == str(blocked_atomic.ref):
                continue
            mappings: list[SupportRoleMapping] = []
            supplied_roles: list[str] = []
            diagnostics: list[dict[str, Any]] = []
            for output in atomic.outputs:
                for consumer_role in sorted(missing):
                    required = blocked_inputs.get(consumer_role)
                    if required is None:
                        continue
                    producer_resolution, effect_domain = _output_authority(
                        atomic,
                        str(output.name),
                        str(output.required_resolution),
                    )
                    type_compatible = semantic_types_compatible(
                        required.semantic_type, output.semantic_type,
                    )
                    resolution_compatible = resolution_satisfies(
                        producer_resolution, required.required_resolution,
                    )
                    compatible = bool(
                        type_compatible and resolution_compatible
                    )
                    diagnostics.append({
                        "producer_role": output.name,
                        "consumer_role": consumer_role,
                        "compatible": bool(compatible),
                        "semantic_type_compatible": bool(type_compatible),
                        "resolution_compatible": bool(
                            resolution_compatible
                        ),
                        "required_type": required.semantic_type,
                        "offered_type": output.semantic_type,
                        "producer_resolution": producer_resolution,
                        "required_resolution": required.required_resolution,
                        "effect_domain": effect_domain,
                    })
                    if not compatible:
                        continue
                    mappings.append(SupportRoleMapping(
                        producer_role=str(output.name),
                        consumer_role=str(consumer_role),
                        semantic_type=normalize_semantic_type(
                            output.semantic_type or required.semantic_type,
                        ),
                        producer_resolution=producer_resolution,
                        required_resolution=str(
                            required.required_resolution
                        ),
                        effect_domain=effect_domain,
                    ))
                    if output.name not in supplied_roles:
                        supplied_roles.append(output.name)
            if not mappings:
                continue
            candidates.append(SupportCandidate(
                atomic_ref=str(atomic.ref),
                score=float(len(mappings)),
                supplied_roles=tuple(sorted(supplied_roles)),
                output_roles=tuple(sorted(str(item.name) for item in atomic.outputs)),
                effect_predicates=tuple(sorted({
                    _predicate_name(item) for item in atomic.effects
                })),
                diagnostics=tuple(to_primitive(diagnostics)),
                role_mappings=tuple(mappings),
            ))
        candidates.sort(key=lambda item: (-item.score, item.atomic_ref))
        return candidates[: max(0, int(top_k))]


__all__ = ["SupportAtomicRetriever", "SupportCandidate"]
