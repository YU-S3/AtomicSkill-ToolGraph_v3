"""Deterministic P2 support-Atomic retrieval after required retrieval.

P1 remains bank-blind.  This module sees only the already-retrieved required
Atomic candidates and the normal Atomic bank, and emits formal producer to
consumer role mappings.  It contains no task-family or object-specific policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.contracts import AbstractAtomicSkill
from ..core.status import RuntimeMode, SkillStatus
from ..core.support_authority import support_role_authority
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
        self.last_role_authority_rejection_count = 0
        self.last_runtime_resolvable_role_exclusion_count = 0
        self.last_diagnostics: tuple[dict[str, Any], ...] = ()

    def retrieve(
        self,
        *,
        required_instance_candidates: Mapping[str, Iterable[Any]],
        mode: RuntimeMode | str,
        harness_profile: str,
        task_id: str = "",
    ) -> list[PlannerSupportCandidate]:
        mode = RuntimeMode(mode)
        self.last_role_authority_rejection_count = 0
        self.last_runtime_resolvable_role_exclusion_count = 0
        self.last_diagnostics = ()
        normal_atomics = self.skills.atomics(mode=mode)
        by_ref = {str(item.ref): item for item in normal_atomics}
        result: list[PlannerSupportCandidate] = []
        runtime_exclusions: set[tuple[str, str, str]] = set()
        authority_rejections: set[
            tuple[str, str, str, str, str]
        ] = set()
        diagnostics: list[dict[str, Any]] = []

        for instance_id, raw_required in sorted(
            required_instance_candidates.items(), key=lambda item: str(item[0]),
        ):
            consumer_atomics = [
                by_ref.get(str(getattr(candidate, "atomic_ref", candidate)))
                for candidate in raw_required
            ]
            consumer_atomics = [item for item in consumer_atomics if item is not None]
            required_refs = {str(item.ref) for item in consumer_atomics}
            for consumer in consumer_atomics:
                for required in consumer.inputs:
                    if required.required and required.runtime_resolvable:
                        runtime_exclusions.add((
                            str(instance_id),
                            str(consumer.ref),
                            str(required.name),
                        ))
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
                    for consumer in consumer_atomics:
                        for required in consumer.inputs:
                            if not required.required:
                                continue
                            if required.runtime_resolvable:
                                runtime_exclusions.add((
                                    str(instance_id),
                                    str(consumer.ref),
                                    str(required.name),
                                ))
                                continue
                            authority = support_role_authority(
                                producer,
                                str(output.name),
                                consumer,
                                str(required.name),
                            )
                            diagnostics.append({
                                "producer_ref": producer_ref,
                                "producer_role": str(output.name),
                                "consumer_ref": str(consumer.ref),
                                "consumer_role": str(required.name),
                                "authorized": authority.authorized,
                                "reason": authority.reason,
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
                                if authority.reason == (
                                    "semantic_role_authority_missing"
                                ):
                                    authority_rejections.add((
                                        str(instance_id),
                                        producer_ref,
                                        str(output.name),
                                        str(consumer.ref),
                                        str(required.name),
                                    ))
                                continue
                            mappings.append(PlannerSupportRoleMapping(
                                producer_role=str(output.name),
                                consumer_role=str(required.name),
                                semantic_type=authority.semantic_type,
                                producer_resolution=(
                                    authority.producer_resolution
                                ),
                                required_resolution=(
                                    authority.required_resolution
                                ),
                                effect_domain=authority.effect_domain,
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
        self.last_role_authority_rejection_count = len(
            authority_rejections
        )
        self.last_runtime_resolvable_role_exclusion_count = len(
            runtime_exclusions
        )
        self.last_diagnostics = tuple(diagnostics)
        return result


__all__ = [
    "PlannerSupportAtomicRetriever",
    "PlannerSupportCandidate",
    "PlannerSupportRoleMapping",
]
