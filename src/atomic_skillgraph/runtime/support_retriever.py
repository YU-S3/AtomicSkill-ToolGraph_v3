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
from ..core.semantic_types import semantic_types_compatible


@dataclass(frozen=True)
class SupportCandidate:
    atomic_ref: str
    score: float
    supplied_roles: tuple[str, ...]
    output_roles: tuple[str, ...]
    effect_predicates: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]


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
            supplied_roles: list[str] = []
            diagnostics: list[dict[str, Any]] = []
            for output in atomic.outputs:
                if output.name not in missing:
                    continue
                required = blocked_inputs[output.name]
                if not semantic_types_compatible(
                    required.semantic_type, output.semantic_type,
                ):
                    diagnostics.append({
                        "role": output.name,
                        "compatible": False,
                        "required_type": required.semantic_type,
                        "offered_type": output.semantic_type,
                    })
                    continue
                supplied_roles.append(output.name)
                diagnostics.append({
                    "role": output.name,
                    "compatible": True,
                    "offered_type": output.semantic_type,
                })
            if not supplied_roles:
                continue
            candidates.append(SupportCandidate(
                atomic_ref=str(atomic.ref),
                score=float(len(supplied_roles)),
                supplied_roles=tuple(sorted(supplied_roles)),
                output_roles=tuple(sorted(str(item.name) for item in atomic.outputs)),
                effect_predicates=tuple(sorted({
                    _predicate_name(item) for item in atomic.effects
                })),
                diagnostics=tuple(to_primitive(diagnostics)),
            ))
        candidates.sort(key=lambda item: (-item.score, item.atomic_ref))
        return candidates[: max(0, int(top_k))]


__all__ = ["SupportAtomicRetriever", "SupportCandidate"]
