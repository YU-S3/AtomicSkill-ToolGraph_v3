"""Deterministic Runtime repeat authority for P2 and stored Composites."""

from __future__ import annotations

from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import CompositeSkill, TaskContract
from ..core.results import RuntimeRepeatConstraint
from ..core.semantic_types import semantic_types_compatible
from ..knowledge.skill_registry import SkillRegistry
from .multiplicity import RequirementExpansion, normalized_constraints


def _edge_type(raw: Any) -> str:
    value = getattr(raw, "value", raw)
    return str(value).casefold()


def _occurrence_instance_ids(occurrence: Any) -> tuple[str, ...]:
    return tuple(map(str, (
        getattr(occurrence, "requirement_instance_ids", ())
        or getattr(occurrence, "requirement_ids", ())
        or ()
    )))


def _proposal_steps(proposal: Any) -> list[Any]:
    return list(
        getattr(proposal, "steps", ())
        or getattr(proposal, "occurrences", ())
        or ()
    )


def _required_repeat_step_owners(
    proposal: Any,
    expansion: RequirementExpansion,
) -> dict[str, tuple[str, int]]:
    """Return the unique formal RepeatBlock owner of required steps.

    A malformed occurrence that claims instances from more than one repeat
    iteration intentionally receives no owner.  The PlannerValidator reports
    that malformed claim; the compiler must not guess which iteration won.
    """

    by_instance = {
        item.instance_id: item
        for item in expansion.instances
        if item.repeat_block_id
    }
    owners: dict[str, tuple[str, int]] = {}
    for occurrence in _proposal_steps(proposal):
        claimed = {
            (
                by_instance[instance_id].repeat_block_id,
                int(by_instance[instance_id].repeat_index),
            )
            for instance_id in _occurrence_instance_ids(occurrence)
            if instance_id in by_instance
        }
        if len(claimed) == 1:
            owners[str(occurrence.step_id)] = next(iter(claimed))
    return owners


def _repeat_scoped_support_owners(
    proposal: Any,
    expansion: RequirementExpansion,
) -> dict[str, tuple[str, int]]:
    """Resolve structurally scoped support steps without semantic repair.

    This helper supplies only ownership shape.  It deliberately does not
    certify output identity: Planner validation must additionally prove the
    producer Atomic's ``input_identity`` derivation and role/type closure.
    """

    steps = _proposal_steps(proposal)
    by_step = {str(item.step_id): item for item in steps}
    required_owners = _required_repeat_step_owners(proposal, expansion)
    blocks = {str(item.block_id): item for item in expansion.repeat_blocks}
    outgoing: dict[str, list[Any]] = {}
    for edge in getattr(proposal, "data_edges", ()):
        if _edge_type(getattr(edge, "edge_type", "")) != "data_flow":
            continue
        outgoing.setdefault(str(edge.source_step), []).append(edge)

    owners: dict[str, tuple[str, int]] = {}
    for occurrence in steps:
        step_id = str(occurrence.step_id)
        if (
            _occurrence_instance_ids(occurrence)
            or not dict(
                getattr(occurrence, "repeat_role_bindings", {}) or {},
            )
        ):
            continue
        target_owners: set[tuple[str, int]] = set()
        for edge in outgoing.get(step_id, ()):
            target_step = str(edge.target_step)
            owner = required_owners.get(target_step)
            target = by_step.get(target_step)
            if owner is None or target is None:
                continue
            block = blocks.get(owner[0])
            if block is None:
                continue
            allowed_roles = {
                *map(str, block.distinct_roles),
                *map(str, block.shared_roles),
            }
            target_bindings = dict(
                getattr(target, "repeat_role_bindings", {}) or {},
            )
            if any(
                str(block_role) in allowed_roles
                and str(target_role) == str(edge.target_role)
                for block_role, target_role in target_bindings.items()
            ):
                target_owners.add(owner)
        if len(target_owners) == 1:
            owners[step_id] = next(iter(target_owners))
    return owners


def _input_identity_source_role(atomic: Any, output_role: str) -> str:
    """Resolve an output's authoritative input identity, if explicit.

    ``output_identity`` is retained only as the repository's existing legacy
    compatibility projection.  Ambiguous legacy rows fail closed.
    """

    validator_spec = dict(getattr(atomic, "validator_spec", {}) or {})
    try:
        derivations = dict(validator_spec.get("output_derivations") or {})
    except (TypeError, ValueError):
        return ""
    if derivations:
        raw = derivations.get(str(output_role))
        if not isinstance(raw, dict):
            return ""
        if str(raw.get("kind", "")) != "input_identity":
            return ""
        input_role = str(raw.get("input_role", ""))
        return input_role if input_role else ""
    legacy = [
        item
        for item in list(validator_spec.get("output_identity") or ())
        if isinstance(item, dict)
        and str(item.get("output_role", "")) == str(output_role)
        and str(item.get("input_role", ""))
    ]
    return str(legacy[0]["input_role"]) if len(legacy) == 1 else ""


def _parameter_types(value: Any, attribute: str) -> dict[str, str]:
    return {
        str(item.name): str(item.semantic_type)
        for item in getattr(value, attribute, ())
        if str(item.name)
    }


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
        by_step = {item.step_id: item for item in proposal.steps}
        step_owners = _required_repeat_step_owners(proposal, expansion)
        support_owners = _repeat_scoped_support_owners(
            proposal, expansion,
        )
        for step_id, owner in support_owners.items():
            # Required ownership always wins structurally, although support
            # owners are defined only for occurrence-instance-free steps.
            step_owners.setdefault(step_id, owner)

        constraints: list[RuntimeRepeatConstraint] = []
        for block in expansion.repeat_blocks:
            iteration_steps: list[tuple[str, ...]] = []
            block_steps: list[str] = []
            for repeat_index in range(block.count):
                current = sorted(
                    (
                        step_id
                        for step_id, owner in step_owners.items()
                        if owner == (block.block_id, repeat_index)
                    ),
                    key=lambda value: (
                        position.get(value, 10**9), value,
                    ),
                )
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
            repeat_roles = {
                *(value for value in (distinct_by,) if value),
                *shared_roles,
            }
            position = {
                step_id: index
                for index, step_id in enumerate(
                    composite.control_sequence,
                )
            }
            incoming: dict[tuple[str, str], list[Any]] = {}
            for edge in composite.data_edges:
                if _edge_type(edge.edge_type) != "data_flow":
                    continue
                incoming.setdefault((
                    str(edge.target_step), str(edge.target_role),
                ), []).append(edge)

            iteration_members: list[set[str]] = [
                {step_id} for step_id, _mapping in candidates
            ]
            iteration_bindings: list[dict[str, dict[str, str]]] = [
                {step_id: dict(mapping)}
                for step_id, mapping in candidates
            ]
            owned_steps: dict[str, int] = {
                step_id: index
                for index, (step_id, _mapping) in enumerate(candidates)
            }

            for repeat_index, (target_step, mapping) in enumerate(
                candidates,
            ):
                target_atomic = atomics[target_step]
                target_inputs = _parameter_types(target_atomic, "inputs")
                for formal_role in repeat_roles:
                    target_role = str(mapping.get(formal_role, ""))
                    if not target_role:
                        unprovable = True
                        break
                    edges = incoming.get((target_step, target_role), ())
                    if not edges:
                        continue
                    if len(edges) != 1:
                        unprovable = True
                        break
                    edge = edges[0]
                    source_step = str(edge.source_step)
                    source_role = str(edge.source_role)
                    source_atomic = atomics.get(source_step)
                    if (
                        source_atomic is None
                        or position.get(source_step, 10**9)
                        >= position.get(target_step, -1)
                    ):
                        unprovable = True
                        break
                    source_outputs = _parameter_types(
                        source_atomic, "outputs",
                    )
                    source_inputs = _parameter_types(
                        source_atomic, "inputs",
                    )
                    source_input_role = _input_identity_source_role(
                        source_atomic, source_role,
                    )
                    if (
                        source_role not in source_outputs
                        or target_role not in target_inputs
                        or source_input_role not in source_inputs
                        or not semantic_types_compatible(
                            source_outputs[source_role],
                            target_inputs[target_role],
                        )
                        or not semantic_types_compatible(
                            source_inputs[source_input_role],
                            source_outputs[source_role],
                        )
                        or not semantic_types_compatible(
                            source_inputs[source_input_role],
                            target_inputs[target_role],
                        )
                    ):
                        unprovable = True
                        break
                    previous_owner = owned_steps.get(source_step)
                    if (
                        previous_owner is not None
                        and previous_owner != repeat_index
                    ):
                        unprovable = True
                        break
                    owned_steps[source_step] = repeat_index
                    iteration_members[repeat_index].add(source_step)
                    source_bindings = iteration_bindings[
                        repeat_index
                    ].setdefault(source_step, {})
                    existing_role = source_bindings.get(formal_role)
                    if (
                        existing_role is not None
                        and existing_role != source_input_role
                    ):
                        unprovable = True
                        break
                    source_bindings[formal_role] = source_input_role
                if unprovable:
                    break
            if unprovable:
                continue

            ordered_iterations = tuple(
                tuple(sorted(
                    members,
                    key=lambda step_id: (
                        position.get(step_id, 10**9), step_id,
                    ),
                ))
                for members in iteration_members
            )
            compiled_bindings = {
                step_id: role_map
                for per_iteration in iteration_bindings
                for step_id, role_map in per_iteration.items()
            }
            compiled.append(RuntimeRepeatConstraint(
                block_id=f"stored::{basis_id}",
                basis_constraint_id=basis_id,
                count=count,
                iteration_steps=ordered_iterations,
                distinct_roles=(distinct_by,) if distinct_by else (),
                shared_roles=shared_roles,
                step_role_bindings=compiled_bindings,
            ))
        return compiled


__all__ = [
    "RuntimeRepeatConstraintCompiler",
    "_input_identity_source_role",
    "_repeat_scoped_support_owners",
    "_required_repeat_step_owners",
    "formal_repeat_role",
    "unit_effect_role_mappings",
]
