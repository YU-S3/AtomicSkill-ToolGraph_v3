"""Materialize exactly the sequence/edges proposed by P0 or P2; never synthesize semantics."""

from __future__ import annotations

from typing import Any

from ..core.contracts import CompositeSkill, PlannerWorkflowProposal, TaskContract
from ..core.edges import GraphEdge, GraphEdgeType
from ..core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
)
from ..core.status import RuntimeMode
from ..knowledge.skill_registry import SkillRegistry
from .multiplicity import RequirementExpansion
from .repeat_constraints import RuntimeRepeatConstraintCompiler


class PlanCompiler:
    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills
        self.repeat_compiler = RuntimeRepeatConstraintCompiler()

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
        repeat_constraints = self.repeat_compiler.from_complete_composite(
            composite, contract, self.skills,
        )
        return RuntimeLinearPlan(
            task.task_id, "stored_composite", str(composite.ref), occurrences,
            list(composite.control_sequence), list(composite.data_edges), list(composite.dependency_edges),
            contract, details, repeat_constraints,
        )

    def compile(
        self, proposal: PlannerWorkflowProposal, task: Any, contract: TaskContract,
        *, mode: RuntimeMode | str, audit: dict[str, Any],
        expansion: RequirementExpansion | None = None,
    ) -> RuntimeLinearPlan:
        occurrences = []
        for item in proposal.steps:
            atomic = self.skills.get_atomic(item.node_ref)
            implementations = self.skills.implementations_for(atomic.ref, mode=mode)
            instance_ids = list(
                item.requirement_instance_ids or item.requirement_ids
            )
            occurrences.append(RuntimeOccurrence(
                step_id=item.step_id,
                occurrence_id=item.occurrence_id,
                node_ref=item.node_ref,
                requirement_ids=list(instance_ids),
                binding_specs=dict(item.binding_specs),
                implementation_candidates=[
                    candidate.ref for candidate in implementations
                ],
                expected_effects=list(item.expected_effects or atomic.effects),
                requirement_instance_ids=list(instance_ids),
                repeat_role_bindings=dict(item.repeat_role_bindings),
            ))
        def edge(item: Any) -> GraphEdge:
            return GraphEdge(
                item.edge_id, GraphEdgeType(item.edge_type), item.source_step, item.target_step,
                item.source_role, item.target_role, item.origin, item.existing_edge_id, (),
            )
        details = dict(audit)
        details["requirement_coverage"] = dict(proposal.requirement_coverage)
        details["sequence_origin"] = proposal.sequence_origin
        repeat_constraints = self.repeat_compiler.from_requirement_expansion(
            proposal,
            expansion,
        )
        return RuntimeLinearPlan(
            task.task_id, "atomic_composition", None, occurrences, list(proposal.control_sequence),
            [edge(item) for item in proposal.data_edges], [edge(item) for item in proposal.dependency_edges],
            contract, details, repeat_constraints,
        )
