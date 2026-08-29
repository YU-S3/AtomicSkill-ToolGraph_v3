"""Semantic recall followed by non-negotiable contract filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.contracts import (
    AtomicCandidate, CapabilityRequirement, RequirementSearchResult,
)
from ..core.status import RuntimeMode, SkillStatus
from ..knowledge.query import atomic_contract_compatible, lexical_similarity
from ..knowledge.skill_registry import SkillRegistry


@dataclass
class AtomicSearchBatch:
    results: list[RequirementSearchResult]

    @property
    def full_coverage(self) -> bool:
        return all(item.covered for item in self.results if item.requirement.required)

    @property
    def candidates(self) -> dict[str, list[AtomicCandidate]]:
        return {item.requirement.requirement_id: item.candidates for item in self.results}

    @property
    def refs(self) -> list[str]:
        return sorted({str(candidate.atomic_ref) for result in self.results for candidate in result.candidates})


class AtomicRetriever:
    def __init__(
        self, skills: SkillRegistry, *, top_k: int = 3, max_top_k: int = 5,
        utility_lookup: Any | None = None, candidate_policy: Any | None = None,
    ) -> None:
        if not 1 <= top_k <= max_top_k <= 5:
            raise ValueError("Atomic retrieval requires 1 <= top_k <= max_top_k <= 5")
        self.skills, self.top_k, self.max_top_k = skills, top_k, max_top_k
        self.utility_lookup = utility_lookup or (lambda _ref: 0.0)
        self.candidate_policy = candidate_policy

    def retrieve(
        self, requirements: list[CapabilityRequirement], *, mode: RuntimeMode | str,
        harness_profile: str, task_id: str = "",
    ) -> AtomicSearchBatch:
        atomics = self.skills.atomics(mode=mode)
        results: list[RequirementSearchResult] = []
        for requirement in requirements:
            recalled: list[tuple[float, str, Any]] = []
            rejected: list[dict[str, Any]] = []
            contract_compatible = [
                atomic for atomic in atomics
                if atomic_contract_compatible(requirement, atomic)
                and not (atomic.metadata.get("harness_profiles") or [])
                or (
                    atomic_contract_compatible(requirement, atomic)
                    and harness_profile in (atomic.metadata.get("harness_profiles") or [])
                )
            ]
            active_available = any(item.status is SkillStatus.ACTIVE for item in contract_compatible)
            for atomic in atomics:
                text = " ".join([requirement.intent, *requirement.semantic_variants])
                score = lexical_similarity(text, f"{atomic.summary} {atomic.guideline}")
                profiles = atomic.metadata.get("harness_profiles") or []
                reasons: list[str] = []
                if not atomic_contract_compatible(requirement, atomic):
                    reasons.append("effect_or_io_contract_mismatch")
                if profiles and harness_profile not in profiles:
                    reasons.append("harness_incompatible")
                if reasons:
                    rejected.append({"atomic_ref": str(atomic.ref), "reasons": reasons, "recall_score": score})
                    continue
                if self.candidate_policy is not None and not self.candidate_policy.allows(
                    artifact_ref=str(atomic.ref), artifact_kind="atomic", status=atomic.status,
                    mode=mode, task_id=task_id or "unknown_task",
                    reliable_active_available=active_available,
                ):
                    rejected.append({
                        "atomic_ref": str(atomic.ref), "reasons": ["candidate_exploration_quota"],
                        "recall_score": score,
                    })
                    continue
                utility = float(self.utility_lookup(str(atomic.ref)))
                status_bonus = 0.05 if atomic.status is SkillStatus.ACTIVE else 0.0
                recalled.append((score + 0.1 * utility + status_bonus, str(atomic.ref), atomic))
            recalled.sort(key=lambda item: (-item[0], item[1]))
            candidates = [AtomicCandidate(item.ref, score, ["contract_compatible"], True) for score, _, item in recalled[: self.top_k]]
            results.append(RequirementSearchResult(requirement, candidates, bool(candidates) or not requirement.required, rejected))
        return AtomicSearchBatch(results)
