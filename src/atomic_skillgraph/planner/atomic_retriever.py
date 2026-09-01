"""Semantic recall followed by non-negotiable contract filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.contracts import (
    AbstractAtomicSkill,
    AtomicCandidate,
    AtomicContractCompatibilityReport,
    CapabilityRequirement,
    RequirementSearchResult,
)
from ..core.semantic_types import normalize_semantic_type
from ..core.status import RuntimeMode, SkillStatus
from ..knowledge.query import (
    diagnose_atomic_contract_compatibility,
    lexical_similarity,
)
from ..knowledge.skill_registry import SkillRegistry
from .multiplicity import RequirementExpansion


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


@dataclass
class MultiplicityAwareAtomicSearch:
    template_results: list[RequirementSearchResult]
    instance_candidates: dict[str, list[AtomicCandidate]]
    expansion: RequirementExpansion

    @property
    def full_coverage(self) -> bool:
        return all(
            bool(self.instance_candidates.get(instance.instance_id))
            for instance in self.expansion.instances
            if instance.requirement.required
        )

    @property
    def candidates(self) -> dict[str, list[AtomicCandidate]]:
        return dict(self.instance_candidates)

    @property
    def refs(self) -> list[str]:
        return sorted({
            str(candidate.atomic_ref)
            for values in self.instance_candidates.values()
            for candidate in values
        })

    @property
    def missing_instances(self) -> list[Any]:
        return [
            instance for instance in self.expansion.instances
            if instance.requirement.required
            and not self.instance_candidates.get(instance.instance_id)
        ]


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

    @staticmethod
    def _compatibility_view(
        report: AtomicContractCompatibilityReport,
    ) -> dict[str, Any]:
        return {
            "effects_passed": report.effects_passed,
            "inputs_passed": report.inputs_passed,
            "failure_codes": list(report.failure_codes),
            "missing_required_input_types": list(
                report.missing_required_input_types
            ),
        }

    @classmethod
    def _repair_hint(
        cls,
        atomic: AbstractAtomicSkill,
        report: AtomicContractCompatibilityReport,
    ) -> dict[str, Any]:
        """Expose only a sanitized Verified Atomic interface to P1R."""

        return {
            "atomic_ref": str(atomic.ref),
            "compatibility": cls._compatibility_view(report),
            "contract_view": {
                "inputs": [
                    {
                        "name": str(item.name),
                        "semantic_type": normalize_semantic_type(
                            item.semantic_type
                        ),
                        "required": bool(item.required),
                        "runtime_resolvable": bool(item.runtime_resolvable),
                        "required_resolution": str(item.required_resolution),
                    }
                    for item in atomic.inputs
                ],
                "outputs": [
                    {
                        "name": str(item.name),
                        "semantic_type": normalize_semantic_type(
                            item.semantic_type
                        ),
                    }
                    for item in atomic.outputs
                ],
                "effects": [
                    {
                        "predicate": str(item.predicate),
                        "args": {
                            str(role): "<role>"
                            for role in sorted(map(str, item.args))
                        },
                        "cardinality": int(item.cardinality),
                        "distinct_by": str(item.distinct_by),
                    }
                    for item in atomic.effects
                ],
            },
        }

    @staticmethod
    def _matched_required_effect_count(
        report: AtomicContractCompatibilityReport,
    ) -> int:
        return sum(
            detail.offered_predicate_found
            and not detail.missing_argument_roles
            and detail.cardinality_sufficient
            for detail in report.effect_details
        )

    def retrieve(
        self, requirements: list[CapabilityRequirement], *, mode: RuntimeMode | str,
        harness_profile: str, task_id: str = "",
    ) -> AtomicSearchBatch:
        atomics = self.skills.atomics(mode=mode)
        results: list[RequirementSearchResult] = []
        for requirement in requirements:
            recalled: list[tuple[float, str, Any]] = []
            rejected: list[dict[str, Any]] = []
            repair_candidates: list[
                tuple[int, int, float, str, dict[str, Any]]
            ] = []
            assessed = [
                (
                    atomic,
                    diagnose_atomic_contract_compatibility(
                        requirement,
                        atomic,
                    ),
                )
                for atomic in atomics
            ]
            contract_compatible = [
                atomic for atomic, diagnosis in assessed
                if diagnosis.passed
                and (
                    not (atomic.metadata.get("harness_profiles") or [])
                    or harness_profile in (
                        atomic.metadata.get("harness_profiles") or []
                    )
                )
            ]
            active_available = any(item.status is SkillStatus.ACTIVE for item in contract_compatible)
            for atomic, diagnosis in assessed:
                text = " ".join([requirement.intent, *requirement.semantic_variants])
                score = lexical_similarity(text, f"{atomic.summary} {atomic.guideline}")
                profiles = atomic.metadata.get("harness_profiles") or []
                reasons: list[str] = []
                if not diagnosis.passed:
                    reasons.append("effect_or_io_contract_mismatch")
                if profiles and harness_profile not in profiles:
                    reasons.append("harness_incompatible")
                if reasons:
                    rejection = {
                        "atomic_ref": str(atomic.ref),
                        "reasons": reasons,
                        "recall_score": score,
                    }
                    if not diagnosis.passed:
                        rejection["compatibility"] = self._compatibility_view(
                            diagnosis
                        )
                        repair_candidates.append((
                            -self._matched_required_effect_count(diagnosis),
                            len(diagnosis.missing_required_input_types),
                            -score,
                            str(atomic.ref),
                            self._repair_hint(atomic, diagnosis),
                        ))
                    rejected.append(rejection)
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
            repair_candidates.sort(key=lambda item: item[:4])
            candidates = [AtomicCandidate(item.ref, score, ["contract_compatible"], True) for score, _, item in recalled[: self.top_k]]
            results.append(RequirementSearchResult(
                requirement=requirement,
                candidates=candidates,
                covered=bool(candidates) or not requirement.required,
                rejection_reasons=rejected,
                repair_hints=[
                    item[4] for item in repair_candidates[: self.top_k]
                ],
            ))
        return AtomicSearchBatch(results)

    def retrieve_multiplicity(
        self,
        expansion: RequirementExpansion,
        *,
        mode: RuntimeMode | str,
        harness_profile: str,
        task_id: str = "",
    ) -> MultiplicityAwareAtomicSearch:
        batch = self.retrieve(
            list(expansion.templates),
            mode=mode,
            harness_profile=harness_profile,
            task_id=task_id,
        )
        by_template = {
            result.requirement.requirement_id: list(result.candidates)
            for result in batch.results
        }
        return MultiplicityAwareAtomicSearch(
            template_results=batch.results,
            instance_candidates={
                instance.instance_id: list(
                    by_template.get(instance.template_requirement_id, ())
                )
                for instance in expansion.instances
            },
            expansion=expansion,
        )
