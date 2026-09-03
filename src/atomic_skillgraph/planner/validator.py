"""Fail-closed RuntimeLinearPlan validation without semantic auto-repair."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.edges import GraphEdgeType
from ..core.contracts import IdentityRelation
from ..core.results import RuntimeLinearPlan, ValidationResult
from ..core.semantic_types import (
    normalize_semantic_type,
    semantic_types_compatible,
)
from ..core.status import RuntimeMode, skill_status_usable
from ..knowledge.graph_store import GraphStore
from ..knowledge.skill_registry import SkillRegistry
from .multiplicity import RequirementExpansion, normalized_constraints
from .repeat_constraints import formal_repeat_role, unit_effect_role_mappings


def _predicate_shape_compatible(required: Any, offered: Any) -> bool:
    return (
        required.predicate.casefold() == offered.predicate.casefold()
        and set(required.args).issubset(offered.args)
    )


def _cardinality(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return count if count > 0 else 0


def _project_plan_effects(
    plan: RuntimeLinearPlan, atomics: dict[str, Any],
) -> tuple[list[tuple[Any, str, dict[str, str | None]]], dict[str, set[str]]]:
    """Resolve effect arguments to stable symbolic plan sources.

    Planner validation cannot know runtime instance values, but it can prove
    whether two roles are forced through the same binding/DataFlow source or
    are explicitly separated into distinct task roles.
    """
    by_step = {item.step_id: item for item in plan.occurrences}
    incoming = {
        (edge.target_step, edge.target_role): edge
        for edge in plan.data_edges
    }
    input_memo: dict[tuple[str, str], str | None] = {}
    output_memo: dict[tuple[str, str], str] = {}
    resolving_inputs: set[tuple[str, str]] = set()
    task_role_usage: dict[str, set[str]] = {}
    repeat_role_source: dict[tuple[str, str], str] = {}
    ambiguous_repeat_sources: set[tuple[str, str]] = set()
    for constraint in plan.repeat_constraints:
        basis_id = str(constraint.basis_constraint_id)
        for repeat_index, iteration in enumerate(
            constraint.iteration_steps,
        ):
            for step_id in iteration:
                for block_role, atomic_role in constraint.step_role_bindings.get(
                    step_id, {},
                ).items():
                    if block_role in constraint.distinct_roles:
                        symbol = (
                            f"repeat:{basis_id}:{block_role}:"
                            f"{repeat_index}"
                        )
                    elif block_role in constraint.shared_roles:
                        symbol = f"repeat:{basis_id}:{block_role}:shared"
                    else:
                        continue
                    key = (step_id, atomic_role)
                    previous = repeat_role_source.get(key)
                    if previous is not None and previous != symbol:
                        ambiguous_repeat_sources.add(key)
                    else:
                        repeat_role_source[key] = symbol
    for key in ambiguous_repeat_sources:
        repeat_role_source.pop(key, None)

    def constant_symbol(value: Any) -> str:
        return f"constant:{type(value).__name__}:{value!r}"

    def expression_symbol(step_id: str, expression: BindingExpression) -> str | None:
        if expression.kind is BindingExprKind.CONSTANT:
            return constant_symbol(expression.constant)
        if expression.kind is BindingExprKind.SKILL_INPUT:
            symbol = f"task:{expression.source_role}"
            task_role_usage.setdefault(expression.source_role, set()).add(symbol)
            return symbol
        if expression.kind is BindingExprKind.DATA_FLOW:
            return output_symbol(expression.source_step, expression.source_role)
        if expression.kind is BindingExprKind.ADAPTER_TRANSFORM:
            source = input_symbol(step_id, expression.source_role)
            return None if source is None else f"transform:{expression.transform_id}:{source}"
        return None

    def input_symbol(step_id: str, role: str) -> str | None:
        key = (step_id, role)
        repeat_symbol = repeat_role_source.get(key)
        if repeat_symbol is not None:
            return repeat_symbol
        if key in input_memo:
            return input_memo[key]
        if key in resolving_inputs:
            return None
        resolving_inputs.add(key)
        occurrence = by_step.get(step_id)
        raw = None if occurrence is None else occurrence.binding_specs.get(role)
        symbol: str | None = None
        if raw is not None:
            try:
                symbol = expression_symbol(step_id, BindingExpression.from_dict(raw))
            except (KeyError, TypeError, ValueError):
                symbol = None
        elif key in incoming:
            edge = incoming[key]
            symbol = output_symbol(edge.source_step, edge.source_role)
        resolving_inputs.discard(key)
        input_memo[key] = symbol
        return symbol

    def effect_argument_symbol(step_id: str, raw: Any) -> str | None:
        if isinstance(raw, dict) and "kind" in raw:
            try:
                raw = BindingExpression.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                return None
        if isinstance(raw, BindingExpression):
            if raw.kind is BindingExprKind.SKILL_INPUT:
                atomic = atomics.get(step_id)
                input_names = {item.name for item in getattr(atomic, "inputs", [])}
                if raw.source_role in input_names:
                    return input_symbol(step_id, raw.source_role)
                return output_symbol(step_id, raw.source_role)
            return expression_symbol(step_id, raw)
        if isinstance(raw, str) and raw.startswith("$"):
            return input_symbol(step_id, raw[1:])
        return constant_symbol(raw)

    def output_symbol(step_id: str, role: str) -> str:
        key = (step_id, role)
        repeat_symbol = repeat_role_source.get(key)
        if repeat_symbol is not None:
            return repeat_symbol
        if key in output_memo:
            return output_memo[key]
        opaque = f"output:{step_id}:{role}"
        output_memo[key] = opaque
        atomic = atomics.get(step_id)
        if atomic is None:
            return opaque
        input_names = [item.name for item in atomic.inputs]
        normalized_role = re.sub(r"[^a-z0-9]+", "_", role.casefold()).strip("_")
        matching_inputs = [
            name for name in input_names
            if normalized_role == name.casefold()
            or normalized_role.endswith(f"_{name.casefold()}")
        ]
        if len(matching_inputs) == 1:
            output_memo[key] = input_symbol(step_id, matching_inputs[0]) or opaque
            return output_memo[key]
        effect_matches: list[str | None] = []
        for effect in atomic.effects:
            for argument_role, raw in effect.args.items():
                normalized_argument = re.sub(
                    r"[^a-z0-9]+", "_", argument_role.casefold()
                ).strip("_")
                if (
                    normalized_role == normalized_argument
                    or normalized_role.endswith(f"_{normalized_argument}")
                ):
                    effect_matches.append(effect_argument_symbol(step_id, raw))
        resolved = {item for item in effect_matches if item is not None}
        if len(resolved) == 1:
            output_memo[key] = next(iter(resolved))
        return output_memo[key]

    offered: list[tuple[Any, str, dict[str, str | None]]] = []
    for step_id, atomic in atomics.items():
        for effect in atomic.effects:
            offered.append((
                effect,
                step_id,
                {
                    role: effect_argument_symbol(step_id, raw)
                    for role, raw in effect.args.items()
                },
            ))
    return offered, task_role_usage


def _matching_effects(
    wanted: Any, offered: list[tuple[Any, str, dict[str, str | None]]],
) -> list[tuple[Any, str, dict[str, str | None]]]:
    return [item for item in offered if _predicate_shape_compatible(wanted, item[0])]


def _distinct_effect_capacity(
    matching: list[tuple[Any, str, dict[str, str | None]]], role: str,
) -> int:
    witnesses: set[str] = set()
    for effect, step_id, arguments in matching:
        count = _cardinality(effect.cardinality)
        symbol = arguments.get(role)
        if count > 1 and effect.distinct_by == role:
            base = symbol or f"declared:{step_id}:{effect.predicate}:{role}"
            witnesses.update(f"{base}:distinct:{index}" for index in range(count))
        elif symbol is not None:
            witnesses.add(symbol)
    return len(witnesses)


def _effects_cover_occurrences(
    required: list[Any], offered: list[tuple[Any, str, dict[str, str | None]]],
) -> bool:
    """Aggregate capacity across occurrences and prove requested distinctness."""
    for wanted in required:
        needed = _cardinality(wanted.cardinality)
        matching = _matching_effects(wanted, offered)
        if not needed or sum(_cardinality(item[0].cardinality) for item in matching) < needed:
            return False
        if wanted.distinct_by and _distinct_effect_capacity(matching, wanted.distinct_by) < needed:
            return False
    return True


def _predicate_role_aliases(predicate: Any, *, target: bool) -> set[str]:
    raw = str(predicate.predicate).strip().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    tail = normalized.rsplit("_", 1)[-1]
    aliases = {raw, normalized, tail}
    if target:
        aliases.update({f"requires_{tail}", f"requires_{normalized}"})
    return {item for item in aliases if item}


def _role_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value).casefold())
    result: set[str] = set()
    for token in tokens:
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if token not in {"agent", "object", "entity", "item", "state", "value"}:
            result.add(token)
    return result


def _role_matches_predicate(role: str, predicate: Any, *, target: bool) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(role).casefold()).strip("_")
    if normalized in _predicate_role_aliases(predicate, target=target):
        return True
    return bool(_role_tokens(normalized) & _role_tokens(predicate.predicate))


def _dependency_roles_compatible(edge: Any, source_atomic: Any, target_atomic: Any) -> bool:
    if bool(edge.source_role) != bool(edge.target_role):
        return False
    if not edge.source_role:
        return True
    source_effects = [
        effect for effect in source_atomic.effects
        if _role_matches_predicate(edge.source_role, effect, target=False)
    ]
    target_preconditions = [
        predicate for predicate in target_atomic.preconditions
        if _role_matches_predicate(edge.target_role, predicate, target=True)
    ]
    return any(
        _predicate_shape_compatible(precondition, effect)
        for effect in source_effects
        for precondition in target_preconditions
    )


def _contract_structures_well_formed(plan: RuntimeLinearPlan) -> bool:
    contract = plan.task_contract
    for effect in contract.target_effects:
        count = _cardinality(effect.cardinality)
        if not count:
            return False
        if count > 1 and not str(effect.distinct_by).strip():
            return False
    for constraint in contract.cardinality_constraints:
        try:
            count = int(constraint.get("count", 0))
        except (AttributeError, TypeError, ValueError):
            return False
        predicate = str(constraint.get("predicate", "")).strip()
        distinct_by = str(
            constraint.get("distinct_by") or constraint.get("role") or ""
        ).strip()
        composition_mode = str(
            constraint.get("composition_mode")
            or ("repeat_unit" if count > 1 else "atomic")
        )
        if not predicate or count <= 0:
            return False
        if count > 1 and not distinct_by:
            return False
        if composition_mode not in {"atomic", "repeat_unit"}:
            return False
        if composition_mode == "repeat_unit" and count < 2:
            return False
    return all(
        str(item.left_role).strip() and str(item.right_role).strip()
        for item in contract.identity_constraints
    )


def _occurrence_instance_ids(occurrence: Any) -> list[str]:
    return list(
        getattr(occurrence, "requirement_instance_ids", None)
        or getattr(occurrence, "requirement_ids", ())
    )


def _atomic_role_names(atomic: Any) -> set[str]:
    roles = {
        item.name
        for item in [*getattr(atomic, "inputs", ()), *getattr(atomic, "outputs", ())]
    }
    for predicate in [
        *getattr(atomic, "preconditions", ()),
        *getattr(atomic, "effects", ()),
    ]:
        roles.update(map(str, predicate.args))
    return roles


def _parameter_types(value: Any) -> dict[str, str]:
    return {
        item.name: item.semantic_type
        for item in [
            *getattr(value, "expected_inputs", ()),
            *getattr(value, "expected_outputs", ()),
        ]
    }


def _atomic_parameter_types(value: Any) -> dict[str, str]:
    return {
        item.name: item.semantic_type
        for item in [
            *getattr(value, "inputs", ()),
            *getattr(value, "outputs", ()),
        ]
    }


def validate_runtime_repeat_contract(
    plan: RuntimeLinearPlan,
    atomics: dict[str, Any],
) -> ValidationResult:
    """Prove Runtime repeat authority against the normalized TaskContract.

    This check applies to every plan source.  In particular, P0 has no
    RequirementExpansion, so the Runtime constraint itself is the only
    deterministic bridge between a Stored Composite and formal multiplicity.
    """

    checks: dict[str, bool] = {}
    try:
        formal = normalized_constraints(plan.task_contract)
        checks["runtime_repeat_task_contract_normalized"] = True
    except (TypeError, ValueError):
        formal = {}
        checks["runtime_repeat_task_contract_normalized"] = False
    formal_repeat = {
        constraint_id: value
        for constraint_id, value in formal.items()
        if value.get("composition_mode") == "repeat_unit"
    }
    runtime = list(plan.repeat_constraints)
    basis_ids = [str(item.basis_constraint_id) for item in runtime]
    block_ids = [str(item.block_id) for item in runtime]
    checks["runtime_repeat_basis_authority"] = (
        all(basis_ids)
        and len(basis_ids) == len(set(basis_ids))
        and len(runtime) == len(formal_repeat)
        and set(basis_ids) == set(formal_repeat)
    ) if runtime or formal_repeat else True
    checks["runtime_repeat_block_ids_unique"] = (
        all(block_ids) and len(block_ids) == len(set(block_ids))
    ) if runtime else True
    checks["nonrepeat_contract_has_no_runtime_repeat"] = bool(
        formal_repeat or not runtime
    )

    count_matches = True
    declared_roles_valid = True
    iteration_structure = True
    step_membership_unique = True
    step_bindings_complete = True
    mapped_roles_exist = True
    mapped_role_types_valid = True
    unit_effect_witnessed = True
    formal_role_authority = True
    formal_roles_declared = True
    globally_owned_steps: set[str] = set()

    for constraint in runtime:
        basis = formal_repeat.get(str(constraint.basis_constraint_id))
        if basis is None:
            count_matches = False
            unit_effect_witnessed = False
            formal_role_authority = False
            formal_roles_declared = False
            continue
        try:
            formal_count = int(basis.get("count", 0))
        except (TypeError, ValueError):
            formal_count = 0
        count_matches &= (
            int(constraint.count) == formal_count
            and formal_count >= 2
        )
        distinct_roles = set(map(str, constraint.distinct_roles))
        shared_roles = set(map(str, constraint.shared_roles))
        declared_roles_valid &= bool(
            distinct_roles.isdisjoint(shared_roles)
            and all(distinct_roles | shared_roles)
        )
        iteration_structure &= (
            len(constraint.iteration_steps) == int(constraint.count)
            and all(constraint.iteration_steps)
        )
        flattened = [
            step_id
            for iteration in constraint.iteration_steps
            for step_id in iteration
        ]
        step_membership_unique &= (
            len(flattened) == len(set(flattened))
            and not (set(flattened) & globally_owned_steps)
            and set(flattened).issubset(atomics)
        )
        globally_owned_steps.update(flattened)
        step_bindings_complete &= (
            set(constraint.step_role_bindings) == set(flattened)
        )

        for step_id, role_map in constraint.step_role_bindings.items():
            atomic = atomics.get(step_id)
            boundary_types = (
                _atomic_parameter_types(atomic) if atomic is not None else {}
            )
            declared_roles_valid &= set(role_map).issubset(
                distinct_roles | shared_roles,
            )
            for block_role, atomic_role in role_map.items():
                mapped_roles_exist &= bool(
                    block_role and atomic_role
                    and atomic_role in boundary_types
                )
                mapped_role_types_valid &= bool(
                    atomic_role in boundary_types
                    and normalize_semantic_type(
                        boundary_types.get(atomic_role, ""),
                    )
                )

        predicate = str(basis.get("predicate", ""))
        predicate_roles = {
            str(role)
            for effect in plan.task_contract.target_effects
            if effect.predicate.casefold() == predicate.casefold()
            for role in effect.args
        }
        formal_distinct = str(basis.get("distinct_by", ""))
        formal_shared = set(map(str, basis.get("shared_roles", ())))
        formal_roles_declared &= (
            (not formal_distinct or formal_distinct in distinct_roles)
            and formal_shared.issubset(shared_roles)
        )
        for iteration in constraint.iteration_steps:
            iteration_witness = False
            iteration_authority = False
            aggregate_present = False
            for step_id in iteration:
                atomic = atomics.get(step_id)
                if atomic is None:
                    continue
                mappings, has_aggregate = unit_effect_role_mappings(
                    atomic, predicate, predicate_roles,
                )
                aggregate_present |= has_aggregate
                if len(mappings) != 1:
                    continue
                iteration_witness = True
                formal_to_atomic = mappings[0]
                runtime_to_atomic = constraint.step_role_bindings.get(
                    step_id, {},
                )

                def runtime_roles_cover(
                    formal_role: str,
                    runtime_roles: set[str],
                ) -> bool:
                    expected_atomic = formal_to_atomic.get(formal_role, "")
                    return bool(expected_atomic) and any(
                        runtime_to_atomic.get(runtime_role)
                        == expected_atomic
                        for runtime_role in runtime_roles
                    )

                distinct_covered = (
                    not formal_distinct
                    or runtime_roles_cover(
                        formal_distinct, distinct_roles,
                    )
                )
                shared_covered = all(
                    runtime_roles_cover(role, shared_roles)
                    for role in formal_shared
                )
                if distinct_covered and shared_covered:
                    iteration_authority = True
            unit_effect_witnessed &= (
                iteration_witness and not aggregate_present
            )
            formal_role_authority &= iteration_authority

    checks.update({
        "runtime_repeat_counts_match_contract": count_matches,
        "runtime_repeat_declared_roles_valid": declared_roles_valid,
        "runtime_repeat_iteration_structure_valid": iteration_structure,
        "runtime_repeat_step_membership_unique": step_membership_unique,
        "runtime_repeat_step_bindings_complete": step_bindings_complete,
        "runtime_repeat_mapped_atomic_roles_exist": mapped_roles_exist,
        "runtime_repeat_mapped_role_types_valid": mapped_role_types_valid,
        "runtime_repeat_unit_effect_witnessed": unit_effect_witnessed,
        "runtime_repeat_formal_roles_declared": formal_roles_declared,
        "runtime_repeat_formal_role_authority": formal_role_authority,
    })
    passed = all(checks.values())
    return ValidationResult(
        level="planner_runtime_repeat",
        passed=passed,
        checks=checks,
        failure_codes=[] if passed else ["planner_repeat_block_invalid"],
        messages=[] if passed else [
            "Runtime repeat constraints do not prove the TaskContract "
            "repeat_unit authority",
        ],
    )


def _repeat_instance_validation(
    plan: RuntimeLinearPlan,
    expansion: RequirementExpansion,
    atomics: dict[str, Any],
    position: dict[str, int],
    instance_candidates: dict[str, set[str]] | None,
) -> tuple[dict[str, bool], list[str]]:
    """Validate instance authority and serial RepeatBlock structure.

    The function is deliberately a proof checker over supplied IR.  It never
    fills missing coverage, selects a candidate, or changes model-authored
    order/role mappings.
    """

    checks: dict[str, bool] = {}
    codes: list[str] = []
    by_instance = {
        item.instance_id: item
        for item in expansion.instances
    }
    by_block = {
        item.block_id: item
        for item in expansion.repeat_blocks
    }
    claims: dict[str, list[str]] = {
        instance_id: [] for instance_id in by_instance
    }
    instance_lists_unique = True
    known_instances_only = True
    every_occurrence_attributed = True
    candidate_authority = True
    repeat_role_maps = True
    repeat_role_types = True
    occurrence_one_iteration = True
    nonrepeat_role_maps_empty = True

    # Track which block roles are observable in each iteration.  Runtime can
    # enforce only roles that P2 maps onto real Atomic boundary/effect roles.
    mapped_roles_by_iteration: dict[tuple[str, int], set[str]] = {}

    for occurrence in plan.occurrences:
        instance_ids = _occurrence_instance_ids(occurrence)
        every_occurrence_attributed &= bool(instance_ids)
        instance_lists_unique &= len(instance_ids) == len(set(instance_ids))
        known = [
            by_instance[instance_id]
            for instance_id in instance_ids
            if instance_id in by_instance
        ]
        known_instances_only &= len(known) == len(instance_ids)
        for instance_id in instance_ids:
            if instance_id in claims:
                claims[instance_id].append(occurrence.step_id)
            if instance_candidates is not None:
                candidates = instance_candidates.get(instance_id, set())
                allowed = {
                    str(getattr(candidate, "atomic_ref", candidate))
                    for candidate in candidates
                }
                candidate_authority &= str(occurrence.node_ref) in allowed

        repeat_instances = [
            item for item in known if item.repeat_block_id
        ]
        repeat_owners = {
            (item.repeat_block_id, item.repeat_index)
            for item in repeat_instances
        }
        occurrence_one_iteration &= len(repeat_owners) <= 1
        role_bindings = dict(
            getattr(occurrence, "repeat_role_bindings", {}) or {}
        )
        if not repeat_instances:
            nonrepeat_role_maps_empty &= not role_bindings
            continue

        block_id, repeat_index = next(iter(repeat_owners))
        block = by_block.get(block_id)
        atomic = atomics.get(occurrence.step_id)
        if block is None or atomic is None:
            repeat_role_maps = False
            continue
        allowed_block_roles = {
            *block.distinct_roles,
            *block.shared_roles,
        }
        atomic_roles = _atomic_role_names(atomic)
        repeat_role_maps &= (
            set(role_bindings).issubset(allowed_block_roles)
            and all(
                bool(block_role)
                and bool(atomic_role)
                and atomic_role in atomic_roles
                for block_role, atomic_role in role_bindings.items()
            )
        )
        mapped_roles_by_iteration.setdefault(
            (block_id, repeat_index), set(),
        ).update(role_bindings)

        atomic_types = _atomic_parameter_types(atomic)
        for instance in repeat_instances:
            requirement_types = _parameter_types(instance.requirement)
            for block_role, atomic_role in role_bindings.items():
                required_type = requirement_types.get(block_role, "")
                offered_type = atomic_types.get(atomic_role, "")
                if required_type and offered_type:
                    repeat_role_types &= semantic_types_compatible(
                        required_type, offered_type,
                    )

    required_coverage = True
    repeat_exact_coverage = True
    for instance in expansion.instances:
        covered = claims.get(instance.instance_id, [])
        if instance.requirement.required:
            required_coverage &= bool(covered)
        if instance.repeat_block_id:
            repeat_exact_coverage &= len(covered) == 1

    audit = plan.planner_audit.get("requirement_coverage", {})
    audit_instance_coverage = isinstance(audit, dict)
    if audit_instance_coverage:
        audit_instance_coverage &= set(audit).issubset(by_instance)
        for instance in expansion.instances:
            if not instance.requirement.required:
                continue
            claimed = audit.get(instance.instance_id)
            audit_instance_coverage &= (
                isinstance(claimed, list)
                and len(claimed) == len(set(claimed))
                and set(claimed) == set(claims[instance.instance_id])
            )

    serial_order = True
    all_iteration_roles_mapped = True
    expected_iteration_steps: dict[str, tuple[tuple[str, ...], ...]] = {}
    for block in expansion.repeat_blocks:
        iterations: list[tuple[str, ...]] = []
        previous_last = -1
        expected_roles = {
            *block.distinct_roles,
            *block.shared_roles,
        }
        for repeat_index in range(block.count):
            steps: list[str] = []
            member_positions: list[int] = []
            for requirement_id in block.ordered_requirement_ids:
                instance_id = (
                    f"{block.block_id}::{repeat_index}::"
                    f"{requirement_id}"
                )
                covered = sorted(
                    claims.get(instance_id, ()),
                    key=lambda step_id: (
                        position.get(step_id, 10**9), step_id,
                    ),
                )
                steps.extend(covered)
                member_positions.extend(
                    position.get(step_id, 10**9)
                    for step_id in covered
                )
            iterations.append(tuple(steps))
            if len(member_positions) != len(
                block.ordered_requirement_ids
            ):
                serial_order = False
            elif any(
                left >= right
                for left, right in zip(
                    member_positions, member_positions[1:]
                )
            ):
                serial_order = False
            elif member_positions and member_positions[0] <= previous_last:
                serial_order = False
            if member_positions:
                previous_last = member_positions[-1]
            all_iteration_roles_mapped &= expected_roles.issubset(
                mapped_roles_by_iteration.get(
                    (block.block_id, repeat_index), set(),
                )
            )
        expected_iteration_steps[block.block_id] = tuple(iterations)

    runtime_constraints = {
        item.block_id: item for item in plan.repeat_constraints
    }
    repeat_constraint_integrity = (
        len(runtime_constraints) == len(plan.repeat_constraints)
        and set(runtime_constraints) == set(by_block)
    )
    by_step = {item.step_id: item for item in plan.occurrences}
    for block_id, block in by_block.items():
        constraint = runtime_constraints.get(block_id)
        if constraint is None:
            repeat_constraint_integrity = False
            continue
        expected_steps = expected_iteration_steps.get(block_id, ())
        expected_step_bindings = {
            step_id: {
                formal_repeat_role(block, block_role): atomic_role
                for block_role, atomic_role in dict(
                    by_step[step_id].repeat_role_bindings,
                ).items()
            }
            for iteration in expected_steps
            for step_id in iteration
            if step_id in by_step
        }
        repeat_constraint_integrity &= (
            constraint.count == block.count
            and constraint.basis_constraint_id
            == block.basis_constraint_id
            and tuple(constraint.iteration_steps) == expected_steps
            and tuple(constraint.distinct_roles) == tuple(
                formal_repeat_role(block, role)
                for role in block.distinct_roles
            )
            and tuple(constraint.shared_roles) == tuple(
                formal_repeat_role(block, role)
                for role in block.shared_roles
            )
            and dict(constraint.step_role_bindings)
            == expected_step_bindings
        )

    checks.update({
        "requirement_instance_lists_unique": instance_lists_unique,
        "requirement_instances_known": known_instances_only,
        "every_occurrence_has_requirement_instance": every_occurrence_attributed,
        "required_requirement_instances_covered": required_coverage,
        "repeat_requirement_instances_exactly_once": repeat_exact_coverage,
        "requirement_instance_candidate_authority": candidate_authority,
        "requirement_instance_audit_consistent": audit_instance_coverage,
        "repeat_occurrence_single_iteration": occurrence_one_iteration,
        "repeat_serial_order": serial_order,
        "repeat_role_bindings_valid": repeat_role_maps,
        "repeat_role_semantic_types_compatible": repeat_role_types,
        "repeat_iteration_roles_mapped": all_iteration_roles_mapped,
        "nonrepeat_role_bindings_empty": nonrepeat_role_maps_empty,
        "runtime_repeat_constraints_match": repeat_constraint_integrity,
    })
    if not all((
        instance_lists_unique,
        known_instances_only,
        every_occurrence_attributed,
        required_coverage,
        repeat_exact_coverage,
        candidate_authority,
        audit_instance_coverage,
    )):
        codes.append("planner_requirement_instance_uncovered")
    if not all((
        occurrence_one_iteration,
        serial_order,
        repeat_constraint_integrity,
    )):
        codes.append("planner_repeat_block_invalid")
    if not all((
        repeat_role_maps,
        repeat_role_types,
        all_iteration_roles_mapped,
        nonrepeat_role_maps_empty,
    )):
        codes.append("planner_repeat_role_invalid")
    return checks, list(dict.fromkeys(codes))


def _identity_cardinality_preserved(
    plan: RuntimeLinearPlan,
    offered: list[tuple[Any, str, dict[str, str | None]]],
    task_role_usage: dict[str, set[str]],
) -> bool:
    contract = plan.task_contract
    if not _contract_structures_well_formed(plan):
        return False
    if not _effects_cover_occurrences(contract.target_effects, offered):
        return False

    for constraint in contract.cardinality_constraints:
        predicate = str(constraint.get("predicate", "")).casefold()
        count = _cardinality(constraint.get("count", 0))
        role = str(constraint.get("distinct_by") or constraint.get("role") or "")
        matching = [
            item for item in offered
            if item[0].predicate.casefold() == predicate
        ]
        if (
            not count
            or sum(_cardinality(item[0].cardinality) for item in matching) < count
            or _distinct_effect_capacity(matching, role) < count
        ):
            return False

    relevant = [
        item for item in offered
        if any(_predicate_shape_compatible(target, item[0]) for target in contract.target_effects)
    ]

    def symbols_for(role: str) -> set[str]:
        effect_symbols = {
            arguments[role]
            for _, _, arguments in relevant
            if arguments.get(role) is not None
        }
        return effect_symbols or set(task_role_usage.get(role, set()))

    for constraint in contract.identity_constraints:
        left = symbols_for(constraint.left_role)
        right = symbols_for(constraint.right_role)
        if not left or not right:
            return False
        if constraint.scope == "occurrence" and constraint.left_role == constraint.right_role:
            if constraint.relation is IdentityRelation.DISTINCT_FROM:
                return False
        elif constraint.scope == "occurrence":
            pairs = [
                (arguments.get(constraint.left_role), arguments.get(constraint.right_role))
                for _, _, arguments in relevant
                if constraint.left_role in arguments or constraint.right_role in arguments
            ]
            if not pairs or any(left_value is None or right_value is None for left_value, right_value in pairs):
                return False
            if constraint.relation is IdentityRelation.SAME_AS:
                if any(left_value != right_value for left_value, right_value in pairs):
                    return False
            elif any(left_value == right_value for left_value, right_value in pairs):
                return False
        elif constraint.relation is IdentityRelation.SAME_AS:
            if len(left | right) != 1:
                return False
        elif not left.isdisjoint(right):
            return False
    return True


class PlannerValidator:
    def __init__(self, skills: SkillRegistry, graph: GraphStore, *, max_occurrences: int = 16) -> None:
        self.skills, self.graph, self.max_occurrences = skills, graph, max_occurrences

    def validate(
        self, plan: RuntimeLinearPlan, *, mode: RuntimeMode | str,
        required_requirement_ids: list[str] | None = None,
        requirement_candidates: dict[str, set[str]] | None = None,
        harness_profile: str = "",
        expansion: RequirementExpansion | None = None,
        instance_candidates: dict[str, set[str]] | None = None,
    ) -> ValidationResult:
        checks: dict[str, bool] = {}
        errors: list[str] = []
        messages: list[str] = []

        by_step = {item.step_id: item for item in plan.occurrences}
        checks["occurrence_limit"] = 0 < len(by_step) == len(plan.occurrences) <= self.max_occurrences
        occurrence_ids = [
            item.occurrence_id for item in plan.occurrences
        ]
        checks["occurrence_ids_unique"] = (
            bool(occurrence_ids)
            and len(occurrence_ids) == len(set(occurrence_ids))
        )
        checks["control_sequence_complete_unique"] = (
            len(plan.control_sequence) == len(by_step)
            and len(set(plan.control_sequence)) == len(plan.control_sequence)
            and set(plan.control_sequence) == set(by_step)
        )
        if (
            not checks["occurrence_limit"]
            or not checks["occurrence_ids_unique"]
            or not checks["control_sequence_complete_unique"]
        ):
            errors.append("planner_graph_invalid")
            messages.append(
                "step/occurrence ids must be unique and the control sequence "
                "must contain every occurrence exactly once"
            )
        position = {step: index for index, step in enumerate(plan.control_sequence)}

        refs_ok = True
        harness_ok = True
        atomics = {}
        for occurrence in plan.occurrences:
            atomic = self.skills.get_atomic(occurrence.node_ref)
            atomics[occurrence.step_id] = atomic
            refs_ok &= skill_status_usable(atomic.status, mode)
            profiles = atomic.metadata.get("harness_profiles") or []
            harness_ok &= not profiles or harness_profile in profiles
        checks["node_refs_exist_and_usable"] = refs_ok
        checks["harness_compatibility"] = harness_ok
        if not refs_ok or not harness_ok:
            errors.append("planner_graph_invalid")

        producer_count: Counter[tuple[str, str]] = Counter()
        forward = True
        edge_types = True
        edge_roles = True
        edge_semantic_types = True
        edge_origin = True
        existing_valid = True
        edge_ids = [edge.edge_id for edge in plan.data_edges + plan.dependency_edges]
        edge_ids_unique = bool(all(edge_ids)) and len(edge_ids) == len(set(edge_ids))
        edge_groups = [
            *((edge, GraphEdgeType.DATA_FLOW) for edge in plan.data_edges),
            *((edge, GraphEdgeType.REQUIRES_SKILL) for edge in plan.dependency_edges),
        ]
        allowed_origins = (
            {"extractor_validated", "existing_active"}
            if plan.source == "stored_composite"
            else {"planner_proposed", "existing_active"}
        )
        for edge, expected_type in edge_groups:
            if edge.edge_type is not expected_type:
                edge_types = False
            if edge.source_step not in by_step or edge.target_step not in by_step:
                forward = False
                edge_roles = False
                continue
            if position.get(edge.source_step, 10**9) >= position.get(edge.target_step, -1):
                forward = False
            source_atomic = atomics.get(edge.source_step)
            target_atomic = atomics.get(edge.target_step)
            source_type = target_type = ""
            if expected_type is GraphEdgeType.DATA_FLOW:
                producer_count[(edge.target_step, edge.target_role)] += 1
                source_outputs = {
                    item.name: item for item in getattr(source_atomic, "outputs", [])
                }
                target_inputs = {
                    item.name: item for item in getattr(target_atomic, "inputs", [])
                }
                source_spec = source_outputs.get(edge.source_role)
                target_spec = target_inputs.get(edge.target_role)
                if source_spec is None or target_spec is None:
                    edge_roles = False
                else:
                    source_type = source_spec.semantic_type
                    target_type = target_spec.semantic_type
                    if not semantic_types_compatible(source_type, target_type):
                        edge_semantic_types = False
            elif (
                source_atomic is None
                or target_atomic is None
                or not _dependency_roles_compatible(edge, source_atomic, target_atomic)
            ):
                edge_roles = False
            if edge.origin not in allowed_origins:
                edge_origin = False
            if edge.origin == "planner_proposed" and edge.existing_edge_id:
                edge_origin = False
            if edge.origin == "existing_active":
                known = self.graph.existing_edge_by_id(edge.existing_edge_id or edge.edge_id, mode=mode)
                if (
                    known is None
                    or known.source_step_ref != str(by_step[edge.source_step].node_ref)
                    or known.target_step_ref != str(by_step[edge.target_step].node_ref)
                    or known.edge_type != edge.edge_type.value
                    or known.source_role != edge.source_role
                    or known.target_role != edge.target_role
                ):
                    existing_valid = False
                elif expected_type is GraphEdgeType.DATA_FLOW and any(known.semantic_types):
                    known_source, known_target = known.semantic_types
                    if (
                        known_source and not semantic_types_compatible(known_source, source_type)
                    ) or (
                        known_target and not semantic_types_compatible(known_target, target_type)
                    ):
                        existing_valid = False
        checks["edges_forward_only"] = forward
        checks["edge_types_valid"] = edge_types
        checks["edge_ids_unique"] = edge_ids_unique or not edge_ids
        checks["edge_roles_valid"] = edge_roles
        checks["edge_semantic_types_compatible"] = edge_semantic_types
        checks["edge_origin_valid"] = edge_origin and existing_valid
        checks["one_authoritative_producer"] = all(count == 1 for count in producer_count.values())
        if not all((
            forward, edge_types, checks["edge_ids_unique"], edge_roles,
            edge_semantic_types, edge_origin, existing_valid,
            checks["one_authoritative_producer"],
        )):
            errors.append("planner_graph_invalid")

        coverage = plan.planner_audit.get("requirement_coverage", {})
        required_requirement_ids = required_requirement_ids or []
        requirement_coverage = True
        if requirement_candidates is not None:
            for occurrence in plan.occurrences:
                if not occurrence.requirement_ids:
                    requirement_coverage = False
                    continue
                for requirement_id in occurrence.requirement_ids:
                    if str(occurrence.node_ref) not in requirement_candidates.get(requirement_id, set()):
                        requirement_coverage = False
            for requirement_id, covered_steps in coverage.items():
                if requirement_id not in requirement_candidates or not isinstance(covered_steps, list):
                    requirement_coverage = False
                    continue
                for step_id in covered_steps:
                    occurrence = by_step.get(step_id)
                    if (
                        occurrence is None
                        or str(occurrence.node_ref) not in requirement_candidates[requirement_id]
                    ):
                        requirement_coverage = False
        for requirement_id in required_requirement_ids:
            covered_steps = coverage.get(requirement_id)
            if (
                not isinstance(covered_steps, list)
                or not covered_steps
                or len(covered_steps) != len(set(covered_steps))
            ):
                requirement_coverage = False
                continue
            for step_id in covered_steps:
                occurrence = by_step.get(step_id)
                if occurrence is None or requirement_id not in occurrence.requirement_ids:
                    requirement_coverage = False
        checks["requirement_coverage"] = requirement_coverage
        if required_requirement_ids and not checks["requirement_coverage"]:
            errors.append("planner_requirement_uncovered")

        if expansion is not None:
            repeat_checks, repeat_codes = _repeat_instance_validation(
                plan,
                expansion,
                atomics,
                position,
                instance_candidates,
            )
            checks.update(repeat_checks)
            errors.extend(repeat_codes)

        runtime_repeat = validate_runtime_repeat_contract(plan, atomics)
        checks.update(runtime_repeat.checks)
        errors.extend(runtime_repeat.failure_codes)

        offered_effects, task_role_usage = _project_plan_effects(plan, atomics)
        checks["task_contract_effect_coverage"] = _effects_cover_occurrences(
            plan.task_contract.target_effects, offered_effects
        )
        terminal_empirical = (
            str(
                plan.planner_audit.get(
                    "selected_composite_authority", {},
                ).get("kind", "")
            )
            == "terminal_empirical"
        )
        if (
            plan.task_contract.target_effects
            and not checks["task_contract_effect_coverage"]
            and not terminal_empirical
        ):
            errors.append("task_contract_mismatch")

        incoming = {(edge.target_step, edge.target_role) for edge in plan.data_edges}
        closure = True
        expressions_consistent = True
        for step_id, atomic in atomics.items():
            occurrence = by_step[step_id]
            input_names = {item.name for item in atomic.inputs}
            expression_sources: dict[str, bool] = {}
            for target_role, raw_expression in occurrence.binding_specs.items():
                if target_role not in input_names:
                    expressions_consistent = False
                    continue
                try:
                    expression = BindingExpression.from_dict(raw_expression)
                except (KeyError, TypeError, ValueError):
                    expressions_consistent = False
                    continue
                if expression.kind is BindingExprKind.DATA_FLOW:
                    matching = [
                        edge for edge in plan.data_edges
                        if edge.source_step == expression.source_step
                        and edge.target_step == step_id
                        and edge.source_role == expression.source_role
                        and edge.target_role == target_role
                    ]
                    expression_valid = (
                        len(matching) == 1
                        and expression.source_step in position
                        and position[expression.source_step] < position.get(step_id, -1)
                    )
                    expressions_consistent &= expression_valid
                    expression_sources[target_role] = expression_valid
                elif expression.kind is BindingExprKind.TOOL_OUTPUT:
                    # Cross-occurrence values must be published and carried by
                    # an explicit DATA_FLOW edge.  Runtime occurrence binding
                    # resolution has no live Tool output namespace.
                    expressions_consistent = False
                    expression_sources[target_role] = False
                else:
                    if (step_id, target_role) in incoming:
                        # A local/task/transform expression plus an explicit
                        # incoming edge would give this input two authorities.
                        expressions_consistent = False
                        checks["one_authoritative_producer"] = False
                    expression_sources[target_role] = expression.kind in {
                        BindingExprKind.SKILL_INPUT,
                        BindingExprKind.CONSTANT,
                        BindingExprKind.ADAPTER_TRANSFORM,
                    }
            for parameter in atomic.inputs:
                if not parameter.required:
                    continue
                sourced = (step_id, parameter.name) in incoming
                sourced = sourced or expression_sources.get(parameter.name, False)
                if not sourced and not parameter.runtime_resolvable:
                    closure = False
        checks["data_flow_expression_consistent"] = expressions_consistent
        checks["required_inputs_closed"] = closure
        checks["identity_cardinality_preserved"] = _identity_cardinality_preserved(
            plan, offered_effects, task_role_usage
        )
        if not closure or not expressions_consistent:
            errors.append("data_flow_error")
        if not checks["identity_cardinality_preserved"] and not terminal_empirical:
            errors.append("task_contract_mismatch")
        if terminal_empirical:
            checks["terminal_empirical_incomplete_coverage_nonblocking"] = True

        errors = list(dict.fromkeys(errors))
        diagnostic_when_terminal = {
            "task_contract_effect_coverage",
            "identity_cardinality_preserved",
        }
        blocking_checks = {
            name: value
            for name, value in checks.items()
            if not (
                terminal_empirical and name in diagnostic_when_terminal
            )
        }
        return ValidationResult(
            level="planner",
            passed=not errors and all(blocking_checks.values()),
            checks=checks,
            failure_codes=errors,
            messages=messages,
        )


def validate_runtime_plan(plan: RuntimeLinearPlan) -> ValidationResult:
    """Pure structural validator useful before registries are available."""
    by_id = {item.step_id: item for item in plan.occurrences}
    unique = (
        0 < len(by_id)
        and len(by_id) == len(plan.occurrences) == len(plan.control_sequence) == len(set(plan.control_sequence))
    )
    complete = set(plan.control_sequence) == set(by_id)
    position = {step: index for index, step in enumerate(plan.control_sequence)}
    forward = all(
        edge.source_step in position and edge.target_step in position
        and position[edge.source_step] < position[edge.target_step]
        for edge in plan.data_edges + plan.dependency_edges
    )
    producers = Counter((edge.target_step, edge.target_role) for edge in plan.data_edges)
    one_source = all(count == 1 for count in producers.values())
    edge_types = all(
        edge.edge_type is GraphEdgeType.DATA_FLOW for edge in plan.data_edges
    ) and all(
        edge.edge_type is GraphEdgeType.REQUIRES_SKILL for edge in plan.dependency_edges
    )
    expression_consistency = True
    for occurrence in plan.occurrences:
        for target_role, raw_expression in occurrence.binding_specs.items():
            try:
                expression = BindingExpression.from_dict(raw_expression)
            except (KeyError, TypeError, ValueError):
                expression_consistency = False
                continue
            if expression.kind is BindingExprKind.DATA_FLOW:
                matching = [
                    edge for edge in plan.data_edges
                    if edge.source_step == expression.source_step
                    and edge.target_step == occurrence.step_id
                    and edge.source_role == expression.source_role
                    and edge.target_role == target_role
                ]
                expression_consistency &= len(matching) == 1
            elif expression.kind is BindingExprKind.TOOL_OUTPUT:
                expression_consistency = False
            elif (occurrence.step_id, target_role) in producers:
                expression_consistency = False
                one_source = False
    checks = {
        "unique": unique, "complete": complete, "forward": forward,
        "one_source": one_source, "edge_types": edge_types,
        "data_flow_expression_consistent": expression_consistency,
    }
    return ValidationResult("planner", all(checks.values()), checks, [] if all(checks.values()) else ["planner_graph_invalid"])
