"""Generic Runtime support-Atomic retrieval.

Only the blocked Atomic contract is used: missing input roles are matched
against other normal Atomic outputs/effects.  No task type, object family, or
benchmark workflow may enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..core.contracts import AbstractAtomicSkill
from ..core.serialization import to_primitive
from ..core.support_authority import support_role_authority


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
                    authority = support_role_authority(
                        atomic,
                        str(output.name),
                        blocked_atomic,
                        consumer_role,
                    )
                    diagnostics.append({
                        "producer_role": output.name,
                        "consumer_role": consumer_role,
                        "compatible": authority.authorized,
                        "authority_reason": authority.reason,
                        "semantic_role_authorized": bool(
                            authority.semantic_alias_authorized
                            or authority.relation_verified_exception
                        ),
                        "required_type": required.semantic_type,
                        "offered_type": output.semantic_type,
                        "producer_resolution": (
                            authority.producer_resolution
                        ),
                        "required_resolution": (
                            authority.required_resolution
                        ),
                        "effect_domain": authority.effect_domain,
                        "producer_aliases": list(
                            authority.producer_aliases
                        ),
                        "consumer_aliases": list(
                            authority.consumer_aliases
                        ),
                        "relation_verified_exception": (
                            authority.relation_verified_exception
                        ),
                    })
                    if not authority.authorized:
                        continue
                    mappings.append(SupportRoleMapping(
                        producer_role=str(output.name),
                        consumer_role=str(consumer_role),
                        semantic_type=authority.semantic_type,
                        producer_resolution=authority.producer_resolution,
                        required_resolution=authority.required_resolution,
                        effect_domain=authority.effect_domain,
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
