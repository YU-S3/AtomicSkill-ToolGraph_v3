"""P0 complete-Composite retrieval; this path never calls an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import CompositeSkill, TaskContract
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode, SkillStatus
from ..knowledge.query import (
    complete_composite_contract_diagnosis,
    lexical_similarity,
)
from ..knowledge.skill_registry import SkillRegistry


@dataclass
class CompositeRetrieval:
    candidates: list[CompositeSkill] = field(default_factory=list)
    audit_candidates: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)


class CompositeRetriever:
    def __init__(self, skills: SkillRegistry, *, top_k: int = 5, candidate_policy: Any | None = None) -> None:
        self.skills = skills
        self.top_k = top_k
        self.candidate_policy = candidate_policy

    def retrieve_complete(
        self, task: Any, contract: TaskContract, *, mode: RuntimeMode | str,
        harness_profile: str,
    ) -> CompositeRetrieval:
        result = CompositeRetrieval()
        ranked: list[tuple[float, str, CompositeSkill]] = []
        composites = self.skills.composites(mode=mode)

        def structural_reasons(composite: CompositeSkill) -> list[str]:
            reasons: list[str] = []
            occurrence_ids = {item.step_id for item in composite.occurrences}
            if (
                len(composite.control_sequence) != len(occurrence_ids)
                or set(composite.control_sequence) != occurrence_ids
                or len(occurrence_ids) != len(composite.occurrences)
            ):
                reasons.append("canonical_sequence_incomplete")
            runtime_occurrence_ids = [
                item.occurrence_id for item in composite.occurrences
            ]
            if len(set(runtime_occurrence_ids)) != len(runtime_occurrence_ids):
                reasons.append("canonical_occurrence_ids_not_unique")
            position = {
                step_id: index for index, step_id in enumerate(composite.control_sequence)
            }
            if any(
                edge.source_step not in position
                or edge.target_step not in position
                or position[edge.source_step] >= position[edge.target_step]
                for edge in composite.data_edges + composite.dependency_edges
            ):
                reasons.append("canonical_edges_invalid")
            if any(
                edge.origin == "planner_proposed"
                for edge in composite.data_edges + composite.dependency_edges
            ):
                reasons.append("unvalidated_temporary_edge")
            return reasons

        def exact_compatible(composite: CompositeSkill) -> bool:
            profiles = composite.metadata.get("harness_profiles") or []
            return (
                complete_composite_contract_diagnosis(
                    contract, composite.goal_contract,
                ).passed
                and (not profiles or harness_profile in profiles)
                and not structural_reasons(composite)
            )

        compatible_active_available = any(
            item.status is SkillStatus.ACTIVE and exact_compatible(item)
            for item in composites
        )
        bootstrap_candidate_ref = ""
        if mode is RuntimeMode.ONLINE and not compatible_active_available:
            bootstrap_candidates = [
                (
                    lexical_similarity(
                        getattr(task, "goal", ""),
                        f"{item.summary} {item.guideline}",
                    ),
                    str(item.ref),
                )
                for item in composites
                if item.status is SkillStatus.CANDIDATE
                and exact_compatible(item)
            ]
            if bootstrap_candidates:
                bootstrap_candidates.sort(key=lambda item: (-item[0], item[1]))
                bootstrap_candidate_ref = bootstrap_candidates[0][1]
        for composite in composites:
            contract_diagnosis = complete_composite_contract_diagnosis(
                contract, composite.goal_contract,
            )
            contract_reasons: list[str] = []
            if not contract_diagnosis.passed:
                contract_reasons.append("goal_contract_exact_mismatch")
            structure_reasons = structural_reasons(composite)
            for occurrence in composite.occurrences:
                self.skills.get_atomic(occurrence.node_ref)
            profiles = composite.metadata.get("harness_profiles") or []
            if profiles and harness_profile not in profiles:
                contract_reasons.append("harness_incompatible")
            lifecycle_reasons: list[str] = []
            is_bootstrap_candidate = bool(
                composite.status is SkillStatus.CANDIDATE
                and mode is RuntimeMode.ONLINE
                and not compatible_active_available
            )
            if (
                is_bootstrap_candidate
                and str(composite.ref) != bootstrap_candidate_ref
            ):
                lifecycle_reasons.append("candidate_bootstrap_not_top1")
            elif self.candidate_policy is not None and not self.candidate_policy.allows(
                artifact_ref=str(composite.ref), artifact_kind="composite", status=composite.status,
                mode=mode, task_id=str(getattr(task, "task_id", "unknown_task")),
                reliable_active_available=compatible_active_available,
                explicit_exploration=is_bootstrap_candidate,
            ):
                lifecycle_reasons.append("candidate_exploration_quota")
            reasons = [
                *contract_reasons,
                *structure_reasons,
                *lifecycle_reasons,
            ]
            score = lexical_similarity(getattr(task, "goal", ""), f"{composite.summary} {composite.guideline}")
            item = {"composite_ref": str(composite.ref), "score": score}
            result.audit_candidates.append(item)
            if reasons:
                if contract_reasons:
                    stage = "retrieval_contract"
                elif structure_reasons:
                    stage = "retrieval_structure"
                else:
                    stage = "lifecycle_policy"
                rejection = {**item, "stage": stage, "reasons": reasons}
                if stage == "retrieval_contract":
                    rejection["contract_diagnosis"] = to_primitive(
                        contract_diagnosis
                    )
                result.rejections.append(rejection)
            else:
                ranked.append((score, str(composite.ref), composite))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        result.candidates = [item[2] for item in ranked[: self.top_k]]
        return result
