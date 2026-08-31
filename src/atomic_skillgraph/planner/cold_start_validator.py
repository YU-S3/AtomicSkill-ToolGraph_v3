"""Deterministic admission and scaffold derivation for C1 plans."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import (
    ColdStartCandidateSource,
    ColdStartExecutionMode,
    ColdStartPlanProposal,
)
from ..core.results import ValidationResult
from .multiplicity import RequirementExpansion


@dataclass
class ColdStartScaffold:
    scaffold_id: str
    plan_id: str
    executable_step_ids: list[str]
    first_unresolved_step_id: str
    unresolved_requirement_instance_ids: list[str]
    referenced_failure_experience_ids: list[str]


def _looks_source_concrete(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return bool(
        re.search(r"(?:_|\s)\d+$", normalized)
        or re.search(r"\b(?:go to|take|put|open|close|heat|cool|clean)\b", normalized)
    )


class ColdStartPlanValidator:
    @staticmethod
    def _binding_path_closed(
        step: Any,
        *,
        incoming: Mapping[tuple[str, str], Any],
        position: Mapping[str, int],
        by_step: Mapping[str, Any],
        candidate_roles: Mapping[str, set[str]],
        candidate_required_inputs: Mapping[str, set[str]] | None = None,
        candidate_runtime_resolvable_roles: Mapping[str, set[str]] | None = None,
        candidate_output_roles: Mapping[str, set[str]] | None = None,
        task_roles: set[str] | None = None,
        executable_sources: set[str] | None = None,
    ) -> bool:
        target_roles = set(candidate_roles.get(step.candidate_ref, ()))
        required_inputs = set(
            (candidate_required_inputs or {}).get(step.candidate_ref, ())
        )
        runtime_resolvable = set(
            (candidate_runtime_resolvable_roles or {}).get(
                step.candidate_ref, (),
            )
        )
        task_roles = set(task_roles or ())
        output_roles = candidate_output_roles or candidate_roles
        for target_role, raw_expression in step.binding_specs.items():
            if target_roles and target_role not in target_roles:
                return False
            expression = BindingExpression.from_dict(raw_expression)
            if expression.kind is BindingExprKind.TOOL_OUTPUT:
                # Tool-local outputs are not a legal inter-node binding path.
                return False
            if expression.kind is not BindingExprKind.DATA_FLOW:
                # Task/constant/adapter expressions are explicitly resolved at
                # Runtime and do not claim an upstream cold-plan output.
                continue
            edge = incoming.get((step.step_id, target_role))
            source = by_step.get(expression.source_step)
            if not (
                edge
                and source is not None
                and edge.source_step == expression.source_step
                and edge.source_role == expression.source_role
                and position.get(edge.source_step, 10**9)
                < position.get(step.step_id, -1)
            ):
                return False
            if (
                executable_sources is not None
                and edge.source_step not in executable_sources
            ):
                return False
            source_roles = set(output_roles.get(source.candidate_ref, ()))
            if source_roles and expression.source_role not in source_roles:
                return False
        for role in required_inputs:
            raw_expression = step.binding_specs.get(role)
            if raw_expression is None:
                if role not in task_roles and role not in runtime_resolvable:
                    return False
                continue
            expression = BindingExpression.from_dict(raw_expression)
            if expression.kind is BindingExprKind.SKILL_INPUT:
                if expression.source_role not in task_roles:
                    return False
            elif expression.kind is BindingExprKind.ADAPTER_TRANSFORM:
                if expression.source_role not in task_roles:
                    return False
            elif expression.kind is BindingExprKind.DATA_FLOW:
                # The edge/source checks above are authoritative.
                continue
            elif expression.kind is BindingExprKind.CONSTANT:
                continue
            else:
                return False
        return True

    def validate(
        self,
        proposal: ColdStartPlanProposal,
        expansion: RequirementExpansion,
        *,
        verified_candidates: Mapping[str, set[str] | list[str]],
        provisional_candidates: Mapping[str, set[str] | list[str]],
        failure_experience_ids: set[str],
        candidate_roles: Mapping[str, set[str]] | None = None,
        candidate_required_inputs: Mapping[str, set[str]] | None = None,
        candidate_runtime_resolvable_roles: Mapping[str, set[str]] | None = None,
        candidate_output_roles: Mapping[str, set[str]] | None = None,
        task_roles: set[str] | None = None,
        scaffold_max_steps: int = 8,
    ) -> ValidationResult:
        checks: dict[str, bool] = {}
        errors: list[str] = []
        by_step = {item.step_id: item for item in proposal.steps}
        checks["step_and_control_unique"] = (
            bool(proposal.steps)
            and len(by_step) == len(proposal.steps)
            and len(proposal.control_sequence) == len(set(proposal.control_sequence))
            and set(proposal.control_sequence) == set(by_step)
        )
        if not checks["step_and_control_unique"]:
            errors.append("cold_start_plan_invalid")
        position = {
            step_id: index for index, step_id in enumerate(proposal.control_sequence)
        }

        required = {
            item.instance_id for item in expansion.instances if item.requirement.required
        }
        occurrences = Counter(
            instance_id
            for step in proposal.steps
            for instance_id in step.requirement_instance_ids
        )
        coverage = proposal.requirement_coverage
        checks["requirement_instances_exact_once"] = (
            set(occurrences) == required
            and all(occurrences[item] == 1 for item in required)
            and set(coverage) == required
            and all(
                isinstance(coverage[item], list)
                and len(coverage[item]) == 1
                and coverage[item][0] in by_step
                and item in by_step[coverage[item][0]].requirement_instance_ids
                for item in required
            )
        )
        if not checks["requirement_instances_exact_once"]:
            errors.append("planner_requirement_instance_uncovered")

        source_mode_valid = True
        candidates_valid = True
        provisional_isolated = True
        roles_valid = True
        concrete_free = True
        candidate_roles = candidate_roles or {}
        mapped_roles_by_iteration: dict[tuple[str, int], set[str]] = {}
        executable_members_by_iteration: dict[tuple[str, int], int] = {}
        total_members_by_iteration: dict[tuple[str, int], int] = {}
        block_by_id = {
            block.block_id: block for block in expansion.repeat_blocks
        }
        for instance in expansion.instances:
            if instance.repeat_block_id:
                key = (instance.repeat_block_id, instance.repeat_index)
                total_members_by_iteration[key] = (
                    total_members_by_iteration.get(key, 0) + 1
                )
        for step in proposal.steps:
            ids = step.requirement_instance_ids
            if step.candidate_source is ColdStartCandidateSource.VERIFIED:
                source_mode_valid &= step.execution_mode is ColdStartExecutionMode.DIRECT_OR_SEEDED
                candidates_valid &= bool(step.candidate_ref) and all(
                    step.candidate_ref in set(verified_candidates.get(item, ()))
                    for item in ids
                )
            elif step.candidate_source is ColdStartCandidateSource.PROVISIONAL:
                source_mode_valid &= step.execution_mode is ColdStartExecutionMode.SEEDED_ONLY
                candidates_valid &= bool(step.candidate_ref) and all(
                    step.candidate_ref in set(provisional_candidates.get(item, ()))
                    for item in ids
                )
                provisional_isolated &= (
                    step.candidate_ref.startswith("provisional://")
                    and not step.candidate_ref.startswith(("skill://implementation", "tool:"))
                )
            else:
                source_mode_valid &= (
                    step.execution_mode is ColdStartExecutionMode.DYNAMIC
                    and step.candidate_ref == ""
                )
            roles = set(candidate_roles.get(step.candidate_ref, ()))
            role_mapping = dict(step.repeat_role_bindings)
            repeat_instances = []
            for instance_id in ids:
                try:
                    instance = expansion.instance(instance_id)
                except KeyError:
                    continue
                if instance.repeat_block_id:
                    repeat_instances.append(instance)
            repeat_owners = {
                (item.repeat_block_id, item.repeat_index)
                for item in repeat_instances
            }
            if step.candidate_source is ColdStartCandidateSource.UNRESOLVED:
                # With no admitted candidate there is no authoritative Atomic
                # role namespace to map.  Runtime will handle the suffix.
                roles_valid &= not role_mapping
            elif not repeat_instances:
                roles_valid &= not role_mapping
            else:
                roles_valid &= len(repeat_owners) == 1
                owner = next(iter(repeat_owners), ("", -1))
                block = block_by_id.get(owner[0])
                allowed = set() if block is None else {
                    *block.distinct_roles,
                    *block.shared_roles,
                }
                roles_valid &= (
                    bool(role_mapping)
                    and set(role_mapping).issubset(allowed)
                    and all(role_mapping.values())
                    and len(set(role_mapping.values())) == len(role_mapping)
                    and all(value in roles for value in role_mapping.values())
                )
                mapped_roles_by_iteration.setdefault(owner, set()).update(
                    role_mapping
                )
                executable_members_by_iteration[owner] = (
                    executable_members_by_iteration.get(owner, 0)
                    + len(repeat_instances)
                )
            for expression in step.binding_specs.values():
                expression = BindingExpression.from_dict(expression)
                if expression.kind is BindingExprKind.CONSTANT:
                    concrete_free &= not _looks_source_concrete(expression.constant)
        # A fully executable repeat iteration must expose every formal
        # distinct/shared role somewhere across its member steps.  Individual
        # steps map only the roles their selected Atomic actually owns (for
        # example Acquire maps object while Place maps object+destination).
        for owner, member_count in executable_members_by_iteration.items():
            if member_count != total_members_by_iteration.get(owner, 0):
                continue
            block = block_by_id.get(owner[0])
            expected = set() if block is None else {
                *block.distinct_roles,
                *block.shared_roles,
            }
            roles_valid &= expected.issubset(
                mapped_roles_by_iteration.get(owner, set())
            )
        checks["candidate_source_mode_valid"] = source_mode_valid
        checks["candidate_refs_retrieved"] = candidates_valid
        checks["provisional_is_seeded_atomic_only"] = provisional_isolated
        checks["repeat_role_bindings_valid"] = roles_valid
        checks["no_source_concrete_values"] = concrete_free
        if not candidates_valid:
            errors.append("cold_start_candidate_not_retrieved")
        if not all((source_mode_valid, provisional_isolated, roles_valid, concrete_free)):
            errors.append("cold_start_plan_invalid")

        expected_order = [
            item.instance_id for item in expansion.instances if item.requirement.required
        ]
        actual_order = [
            instance_id
            for step_id in proposal.control_sequence
            for instance_id in by_step.get(step_id, ()).requirement_instance_ids
        ] if checks["step_and_control_unique"] else []
        checks["repeat_serial_order"] = actual_order == expected_order
        if not checks["repeat_serial_order"]:
            errors.append("planner_repeat_block_invalid")

        edge_forward = all(
            edge.source_step in position
            and edge.target_step in position
            and position[edge.source_step] < position[edge.target_step]
            for edge in proposal.data_edges + proposal.dependency_edges
        )
        checks["edges_forward_only"] = edge_forward
        if not edge_forward:
            errors.append("cold_start_plan_invalid")

        known_experiences = set(proposal.referenced_failure_experience_ids).issubset(
            failure_experience_ids
        )
        checks["failure_experiences_retrieved"] = known_experiences
        if not known_experiences:
            errors.append("cold_start_candidate_not_retrieved")

        incoming = {
            (edge.target_step, edge.target_role): edge
            for edge in proposal.data_edges
        }
        closure = all(
            self._binding_path_closed(
                step,
                incoming=incoming,
                position=position,
                by_step=by_step,
                candidate_roles=candidate_roles,
                candidate_required_inputs=candidate_required_inputs,
                candidate_runtime_resolvable_roles=(
                    candidate_runtime_resolvable_roles
                ),
                candidate_output_roles=candidate_output_roles,
                task_roles=task_roles,
            )
            for step in proposal.steps
            if step.candidate_source is not ColdStartCandidateSource.UNRESOLVED
        )
        checks["binding_paths_closed_or_runtime_resolvable"] = closure
        if not closure:
            errors.append("cold_start_plan_invalid")

        unresolved_count = sum(
            step.candidate_source is ColdStartCandidateSource.UNRESOLVED
            for step in proposal.steps
        )
        checks["step_limit"] = len(proposal.steps) <= int(scaffold_max_steps) + unresolved_count
        if not checks["step_limit"]:
            errors.append("cold_start_plan_invalid")

        errors = list(dict.fromkeys(errors))
        return ValidationResult(
            level="cold_start_plan",
            passed=not errors and all(checks.values()),
            checks=checks,
            failure_codes=errors,
        )

    def scaffold(
        self,
        proposal: ColdStartPlanProposal,
        *,
        statically_executable: Mapping[str, bool] | None = None,
        candidate_roles: Mapping[str, set[str]] | None = None,
        candidate_required_inputs: Mapping[str, set[str]] | None = None,
        candidate_runtime_resolvable_roles: Mapping[str, set[str]] | None = None,
        candidate_output_roles: Mapping[str, set[str]] | None = None,
        task_roles: set[str] | None = None,
    ) -> ColdStartScaffold:
        by_step = {item.step_id: item for item in proposal.steps}
        position = {
            step_id: index
            for index, step_id in enumerate(proposal.control_sequence)
        }
        incoming = {
            (edge.target_step, edge.target_role): edge
            for edge in proposal.data_edges
        }
        candidate_roles = candidate_roles or {}
        executable: list[str] = []
        first_unresolved = ""
        for step_id in proposal.control_sequence:
            step = by_step[step_id]
            allowed = (
                step.candidate_source is not ColdStartCandidateSource.UNRESOLVED
                and (statically_executable or {}).get(step_id, True)
                and self._binding_path_closed(
                    step,
                    incoming=incoming,
                    position=position,
                    by_step=by_step,
                    candidate_roles=candidate_roles,
                    candidate_required_inputs=candidate_required_inputs,
                    candidate_runtime_resolvable_roles=(
                        candidate_runtime_resolvable_roles
                    ),
                    candidate_output_roles=candidate_output_roles,
                    task_roles=task_roles,
                    executable_sources=set(executable),
                )
            )
            if not allowed:
                first_unresolved = step_id
                break
            executable.append(step_id)
        remaining = proposal.control_sequence[len(executable):]
        return ColdStartScaffold(
            scaffold_id=f"scaffold::{proposal.plan_id}",
            plan_id=proposal.plan_id,
            executable_step_ids=executable,
            first_unresolved_step_id=first_unresolved,
            unresolved_requirement_instance_ids=[
                instance_id
                for step_id in remaining
                for instance_id in by_step[step_id].requirement_instance_ids
            ],
            referenced_failure_experience_ids=list(
                proposal.referenced_failure_experience_ids
            ),
        )


__all__ = ["ColdStartPlanValidator", "ColdStartScaffold"]
