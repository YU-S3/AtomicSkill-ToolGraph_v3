"""Materialize exactly the sequence/edges proposed by P0 or P2; never synthesize semantics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..core.contracts import CompositeSkill, PlannerWorkflowProposal, TaskContract
from ..core.edges import GraphEdge, GraphEdgeType
from ..core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
    RuntimeRepeatConstraint,
)
from ..core.status import RuntimeMode
from ..knowledge.skill_registry import SkillRegistry
from .multiplicity import RequirementExpansion


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
        repeat_constraints = self._repeat_constraints(
            proposal,
            expansion,
        )
        return RuntimeLinearPlan(
            task.task_id, "atomic_composition", None, occurrences, list(proposal.control_sequence),
            [edge(item) for item in proposal.data_edges], [edge(item) for item in proposal.dependency_edges],
            contract, details, repeat_constraints,
        )

    @staticmethod
    def _repeat_constraints(
        proposal: PlannerWorkflowProposal,
        expansion: RequirementExpansion | None,
    ) -> list[RuntimeRepeatConstraint]:
        """Compile only declared RepeatBlock structure, never infer repeats.

        Invalid or incomplete model-authored coverage remains representable in
        the Runtime IR so :class:`PlannerValidator` can return a typed content
        rejection.  The compiler therefore preserves every covering step in
        control-sequence order instead of silently selecting one.
        """

        if expansion is None:
            return []
        position = {
            step_id: index
            for index, step_id in enumerate(proposal.control_sequence)
        }
        coverage: dict[str, list[str]] = {}
        by_step = {item.step_id: item for item in proposal.steps}
        for occurrence in proposal.steps:
            instance_ids = (
                occurrence.requirement_instance_ids
                or occurrence.requirement_ids
            )
            for instance_id in instance_ids:
                coverage.setdefault(instance_id, []).append(
                    occurrence.step_id
                )
        for instance_id, step_ids in coverage.items():
            coverage[instance_id] = sorted(
                step_ids,
                key=lambda value: (position.get(value, 10**9), value),
            )

        constraints: list[RuntimeRepeatConstraint] = []
        for block in expansion.repeat_blocks:
            iteration_steps: list[tuple[str, ...]] = []
            block_steps: list[str] = []
            for repeat_index in range(block.count):
                current: list[str] = []
                for requirement_id in block.ordered_requirement_ids:
                    instance_id = (
                        f"{block.block_id}::{repeat_index}::"
                        f"{requirement_id}"
                    )
                    current.extend(coverage.get(instance_id, ()))
                iteration_steps.append(tuple(current))
                block_steps.extend(current)
            constraints.append(RuntimeRepeatConstraint(
                block_id=block.block_id,
                count=block.count,
                iteration_steps=tuple(iteration_steps),
                distinct_roles=tuple(block.distinct_roles),
                shared_roles=tuple(block.shared_roles),
                step_role_bindings={
                    step_id: dict(by_step[step_id].repeat_role_bindings)
                    for step_id in dict.fromkeys(block_steps)
                    if step_id in by_step
                },
            ))
        return constraints
