"""Deterministic Runtime repeat authority for P2 and stored Composites."""

from __future__ import annotations

from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import CompositeSkill, TaskContract
from ..core.results import RuntimeRepeatConstraint
from ..knowledge.skill_registry import SkillRegistry
from .multiplicity import RequirementExpansion, normalized_constraints


def _source_role(raw: Any) -> str:
    expression: BindingExpression | None = None
    if isinstance(raw, BindingExpression):
        expression = raw
    elif isinstance(raw, dict) and "kind" in raw:
        try:
            expression = BindingExpression.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return ""
    if expression is not None:
        if expression.kind is BindingExprKind.SKILL_INPUT:
            return str(expression.source_role)
        return ""
    if isinstance(raw, str) and raw.startswith("$"):
        return raw[1:]
    return ""


def _boundary_roles(atomic: Any) -> set[str]:
    return {
        str(spec.name)
        for spec in (
            *getattr(atomic, "inputs", ()),
            *getattr(atomic, "outputs", ()),
        )
        if str(spec.name)
    }


def unit_effect_role_mappings(
    atomic: Any,
    predicate: str,
    predicate_roles: set[str],
) -> tuple[list[dict[str, str]], bool]:
    """Return formal-role mappings for unit effects and aggregate presence.

    The boolean is true when the Atomic declares the basis predicate with a
    non-unit cardinality.  Such an Atomic cannot certify a ``repeat_unit``
    Stored Composite even if other steps happen to provide unit effects.
    """

    if not predicate_roles:
        return [], False
    boundary_roles = _boundary_roles(atomic)
    mappings: list[dict[str, str]] = []
    aggregate_present = False
    for effect in getattr(atomic, "effects", ()):
        if str(effect.predicate).casefold() != str(predicate).casefold():
            continue
        try:
            cardinality = int(effect.cardinality)
        except (TypeError, ValueError):
            cardinality = 0
        if cardinality != 1:
            aggregate_present = True
            continue
        if not predicate_roles.issubset(map(str, effect.args)):
            continue
        mapping = {
            role: _source_role(effect.args[role])
            for role in predicate_roles
        }
        if (
            all(mapping.values())
            and set(mapping.values()).issubset(boundary_roles)
            and len(set(mapping.values())) == len(mapping)
        ):
            mappings.append(mapping)
    return mappings, aggregate_present


def _constraint_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(value.get("predicate", "")).casefold(),
        int(value.get("count", 0)),
        str(value.get("distinct_by", "")),
        tuple(sorted(map(str, value.get("shared_roles", ())))),
        str(value.get("composition_mode", "")),
    )


def formal_repeat_role(block: Any, block_role: str) -> str:
    """Resolve one Planner-local RepeatBlock role to its formal basis role."""

    matches = [
        str(formal_role)
        for formal_role, mapped_role in dict(
            block.basis_role_map,
        ).items()
        if str(mapped_role) == str(block_role)
    ]
    return matches[0] if len(matches) == 1 else ""


class RuntimeRepeatConstraintCompiler:
    """Compile only explicit formal repeat authority; never infer workflows."""

    def from_requirement_expansion(
        self,
        proposal: Any,
        expansion: RequirementExpansion | None,
    ) -> list[RuntimeRepeatConstraint]:
        """Materialize the existing RepeatBlock expansion without repair."""

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
                getattr(occurrence, "requirement_instance_ids", ())
                or getattr(occurrence, "requirement_ids", ())
            )
            for instance_id in instance_ids:
                coverage.setdefault(instance_id, []).append(
                    occurrence.step_id,
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
                distinct_roles=tuple(
                    formal_repeat_role(block, role)
                    for role in block.distinct_roles
                ),
                shared_roles=tuple(
                    formal_repeat_role(block, role)
                    for role in block.shared_roles
                ),
                step_role_bindings={
                    step_id: {
                        formal_repeat_role(block, block_role): atomic_role
                        for block_role, atomic_role in dict(
                            by_step[step_id].repeat_role_bindings,
                        ).items()
                    }
                    for step_id in dict.fromkeys(block_steps)
                    if step_id in by_step
                },
                basis_constraint_id=block.basis_constraint_id,
            ))
        return constraints

    def from_complete_composite(
        self,
        composite: CompositeSkill,
        contract: TaskContract,
        skills: SkillRegistry,
    ) -> list[RuntimeRepeatConstraint]:
        """Derive Stored-Composite unit witnesses from formal contracts.

        An unprovable formal repeat is represented by an absent constraint.
        The PlannerValidator then emits ``planner_repeat_block_invalid`` so P0
        can audit the candidate and continue, rather than leaking an exception
        or guessing a workflow grouping.
        """

        try:
            task_constraints = normalized_constraints(contract)
            composite_constraints = normalized_constraints(
                composite.goal_contract,
            )
        except (TypeError, ValueError):
            return []

        composite_repeat = [
            value for value in composite_constraints.values()
            if value.get("composition_mode") == "repeat_unit"
        ]
        occurrences = {
            item.step_id: item for item in composite.occurrences
        }
        atomics: dict[str, Any] = {}
        try:
            atomics = {
                step_id: skills.get_atomic(occurrences[step_id].node_ref)
                for step_id in composite.control_sequence
                if step_id in occurrences
            }
        except (KeyError, TypeError, ValueError):
            return []

        compiled: list[RuntimeRepeatConstraint] = []
        for basis_id, basis in task_constraints.items():
            if basis.get("composition_mode") != "repeat_unit":
                continue
            matching_formal = [
                value for value in composite_repeat
                if _constraint_signature(value)
                == _constraint_signature(basis)
            ]
            if len(matching_formal) != 1:
                continue

            predicate = str(basis.get("predicate", ""))
            predicate_roles = {
                str(role)
                for effect in contract.target_effects
                if effect.predicate.casefold() == predicate.casefold()
                for role in effect.args
            }
            if not predicate_roles:
                continue

            candidates: list[tuple[str, dict[str, str]]] = []
            unprovable = False
            for step_id in composite.control_sequence:
                atomic = atomics.get(step_id)
                if atomic is None:
                    unprovable = True
                    break
                mappings, aggregate_present = unit_effect_role_mappings(
                    atomic, predicate, predicate_roles,
                )
                if aggregate_present or len(mappings) > 1:
                    unprovable = True
                    break
                if len(mappings) == 1:
                    candidates.append((step_id, mappings[0]))

            try:
                count = int(basis.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            if unprovable or count < 2 or len(candidates) != count:
                continue

            distinct_by = str(basis.get("distinct_by", ""))
            shared_roles = tuple(map(str, basis.get("shared_roles", ())))
            compiled.append(RuntimeRepeatConstraint(
                block_id=f"stored::{basis_id}",
                basis_constraint_id=basis_id,
                count=count,
                iteration_steps=tuple(
                    (step_id,) for step_id, _mapping in candidates
                ),
                distinct_roles=(distinct_by,) if distinct_by else (),
                shared_roles=shared_roles,
                step_role_bindings={
                    step_id: mapping
                    for step_id, mapping in candidates
                },
            ))
        return compiled


__all__ = [
    "RuntimeRepeatConstraintCompiler",
    "formal_repeat_role",
    "unit_effect_role_mappings",
]
