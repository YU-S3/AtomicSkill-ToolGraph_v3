"""Deterministic P2 support-Atomic retrieval after required retrieval.

P1 remains bank-blind.  This module sees only the already-retrieved required
Atomic candidates and the normal Atomic bank, and emits formal producer to
consumer role mappings.  It contains no task-family or object-specific policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.bindings import BindingExprKind, BindingExpression, resolution_satisfies
from ..core.contracts import AbstractAtomicSkill, EffectDomain
from ..core.semantic_types import normalize_semantic_type, semantic_types_compatible
from ..core.status import RuntimeMode, SkillStatus
from ..knowledge.skill_registry import SkillRegistry


@dataclass(frozen=True)
class PlannerSupportRoleMapping:
    producer_role: str
    consumer_role: str
    semantic_type: str
    producer_resolution: str
    required_resolution: str
    effect_domain: str
    consumer_atomic_ref: str


@dataclass(frozen=True)
class PlannerSupportCandidate:
    atomic_ref: str
    consumer_requirement_instance_id: str
    score: float
    role_mappings: tuple[PlannerSupportRoleMapping, ...]
    output_roles: tuple[str, ...]
    effect_predicates: tuple[str, ...]


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
    """Return the strongest contract-declared output resolution/domain."""

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


class PlannerSupportAtomicRetriever:
    """Produce a bounded support pool for P2 without exposing the bank to P1."""

    def __init__(
        self,
        skills: SkillRegistry,
        *,
        top_k: int,
        candidate_policy: Any | None = None,
    ) -> None:
        if not 1 <= int(top_k) <= 5:
            raise ValueError("Planner support retrieval requires 1 <= top_k <= 5")
        self.skills = skills
        self.top_k = int(top_k)
        self.candidate_policy = candidate_policy

    def retrieve(
        self,
        *,
        required_instance_candidates: Mapping[str, Iterable[Any]],
        mode: RuntimeMode | str,
        harness_profile: str,
        task_id: str = "",
    ) -> list[PlannerSupportCandidate]:
        mode = RuntimeMode(mode)
        normal_atomics = self.skills.atomics(mode=mode)
        by_ref = {str(item.ref): item for item in normal_atomics}
        result: list[PlannerSupportCandidate] = []

        for instance_id, raw_required in sorted(
            required_instance_candidates.items(), key=lambda item: str(item[0]),
        ):
            consumer_atomics = [
                by_ref.get(str(getattr(candidate, "atomic_ref", candidate)))
                for candidate in raw_required
            ]
            consumer_atomics = [item for item in consumer_atomics if item is not None]
            required_refs = {str(item.ref) for item in consumer_atomics}
            ranked: list[tuple[float, str, AbstractAtomicSkill, tuple[PlannerSupportRoleMapping, ...]]] = []

            for producer in normal_atomics:
                producer_ref = str(producer.ref)
                profiles = producer.metadata.get("harness_profiles") or []
                if producer_ref in required_refs or (
                    profiles and harness_profile not in profiles
                ):
                    continue
                mappings: list[PlannerSupportRoleMapping] = []
                for output in producer.outputs:
                    produced_resolution, effect_domain = _output_authority(
                        producer,
                        str(output.name),
                        str(output.required_resolution),
                    )
                    for consumer in consumer_atomics:
                        for required in consumer.inputs:
                            if not required.required or not semantic_types_compatible(
                                required.semantic_type, output.semantic_type,
                            ):
                                continue
                            if not resolution_satisfies(
                                produced_resolution, required.required_resolution,
                            ):
                                continue
                            mappings.append(PlannerSupportRoleMapping(
                                producer_role=str(output.name),
                                consumer_role=str(required.name),
                                semantic_type=normalize_semantic_type(
                                    output.semantic_type or required.semantic_type,
                                ),
                                producer_resolution=produced_resolution,
                                required_resolution=str(required.required_resolution),
                                effect_domain=effect_domain,
                                consumer_atomic_ref=str(consumer.ref),
                            ))
                unique_mappings = tuple(dict.fromkeys(mappings))
                if unique_mappings:
                    ranked.append((
                        float(len(unique_mappings)),
                        producer_ref,
                        producer,
                        unique_mappings,
                    ))

            active_available = any(
                producer.status is SkillStatus.ACTIVE
                for _score, _ref, producer, _mappings in ranked
            )
            filtered: list[tuple[float, str, AbstractAtomicSkill, tuple[PlannerSupportRoleMapping, ...]]] = []
            for item in ranked:
                _score, producer_ref, producer, _mappings = item
                if self.candidate_policy is not None and not self.candidate_policy.allows(
                    artifact_ref=producer_ref,
                    artifact_kind="atomic",
                    status=producer.status,
                    mode=mode,
                    task_id=task_id or "unknown_task",
                    reliable_active_available=active_available,
                ):
                    continue
                filtered.append(item)
            filtered.sort(key=lambda item: (-item[0], item[1]))
            for score, producer_ref, producer, mappings in filtered[: self.top_k]:
                result.append(PlannerSupportCandidate(
                    atomic_ref=producer_ref,
                    consumer_requirement_instance_id=str(instance_id),
                    score=score,
                    role_mappings=mappings,
                    output_roles=tuple(sorted(str(item.name) for item in producer.outputs)),
                    effect_predicates=tuple(sorted({
                        str(item.predicate) for item in producer.effects
                    })),
                ))
        return result


__all__ = [
    "PlannerSupportAtomicRetriever",
    "PlannerSupportCandidate",
    "PlannerSupportRoleMapping",
]
