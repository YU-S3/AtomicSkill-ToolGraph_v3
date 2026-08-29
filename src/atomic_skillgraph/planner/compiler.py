"""Materialize exactly the sequence/edges proposed by P0 or P2; never synthesize semantics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..core.contracts import CompositeSkill, PlannerWorkflowProposal, TaskContract
from ..core.edges import GraphEdge, GraphEdgeType
from ..core.results import RuntimeLinearPlan, RuntimeOccurrence
from ..core.status import RuntimeMode
from ..knowledge.skill_registry import SkillRegistry


class PlanCompiler:
    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills

    def from_composite(
        self, task: Any, contract: TaskContract, composite: CompositeSkill,
        *, mode: RuntimeMode | str, audit: dict[str, Any],
    ) -> RuntimeLinearPlan:
        occurrences = []
        for item in composite.occurrences:
            atomic = self.skills.get_atomic(item.node_ref)
            implementations = self.skills.implementations_for(atomic.ref, mode=mode)
            occurrences.append(RuntimeOccurrence(
                item.step_id, item.occurrence_id, item.node_ref, [], dict(item.binding_specs),
                [candidate.ref for candidate in implementations], list(atomic.effects),
            ))
        details = dict(audit)
        details["sequence_origin"] = "existing_composite_sequence"
        return RuntimeLinearPlan(
            task.task_id, "stored_composite", str(composite.ref), occurrences,
            list(composite.control_sequence), list(composite.data_edges), list(composite.dependency_edges),
            contract, details,
        )

    def compile(
        self, proposal: PlannerWorkflowProposal, task: Any, contract: TaskContract,
        *, mode: RuntimeMode | str, audit: dict[str, Any],
    ) -> RuntimeLinearPlan:
        occurrences = []
        for item in proposal.steps:
            atomic = self.skills.get_atomic(item.node_ref)
            implementations = self.skills.implementations_for(atomic.ref, mode=mode)
            occurrences.append(RuntimeOccurrence(
                item.step_id, item.occurrence_id, item.node_ref, list(item.requirement_ids),
                dict(item.binding_specs), [candidate.ref for candidate in implementations],
                list(item.expected_effects or atomic.effects),
            ))
        def edge(item: Any) -> GraphEdge:
            return GraphEdge(
                item.edge_id, GraphEdgeType(item.edge_type), item.source_step, item.target_step,
                item.source_role, item.target_role, item.origin, item.existing_edge_id, (),
            )
        details = dict(audit)
        details["requirement_coverage"] = dict(proposal.requirement_coverage)
        details["sequence_origin"] = proposal.sequence_origin
        return RuntimeLinearPlan(
            task.task_id, "atomic_composition", None, occurrences, list(proposal.control_sequence),
            [edge(item) for item in proposal.data_edges], [edge(item) for item in proposal.dependency_edges],
            contract, details,
        )
