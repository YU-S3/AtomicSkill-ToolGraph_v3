"""P0 complete-Composite retrieval; this path never calls an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import CompositeSkill, TaskContract
from ..core.status import RuntimeMode
from ..knowledge.query import lexical_similarity, task_contract_compatible
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
        active_available = any(str(item.status.value) == "active" for item in composites)
        for composite in composites:
            reasons: list[str] = []
            occurrence_ids = {item.step_id for item in composite.occurrences}
            if len(composite.control_sequence) != len(occurrence_ids) or set(composite.control_sequence) != occurrence_ids:
                reasons.append("canonical_sequence_incomplete")
            if not task_contract_compatible(contract, composite.goal_contract):
                reasons.append("goal_contract_incompatible")
            for occurrence in composite.occurrences:
                try:
                    self.skills.get_atomic(occurrence.node_ref)
                except KeyError:
                    reasons.append(f"occurrence_ref_unavailable:{occurrence.node_ref}")
            profiles = composite.metadata.get("harness_profiles") or []
            if profiles and harness_profile not in profiles:
                reasons.append("harness_incompatible")
            if any(edge.origin == "planner_proposed" for edge in composite.data_edges + composite.dependency_edges):
                reasons.append("unvalidated_temporary_edge")
            if self.candidate_policy is not None and not self.candidate_policy.allows(
                artifact_ref=str(composite.ref), artifact_kind="composite", status=composite.status,
                mode=mode, task_id=str(getattr(task, "task_id", "unknown_task")),
                reliable_active_available=active_available,
            ):
                reasons.append("candidate_exploration_quota")
            score = lexical_similarity(getattr(task, "goal", ""), f"{composite.summary} {composite.guideline}")
            item = {"composite_ref": str(composite.ref), "score": score}
            result.audit_candidates.append(item)
            if reasons:
                result.rejections.append({**item, "reasons": reasons})
            else:
                ranked.append((score, str(composite.ref), composite))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        result.candidates = [item[2] for item in ranked[: self.top_k]]
        return result
