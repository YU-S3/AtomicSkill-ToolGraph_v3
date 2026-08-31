"""Contract-backed requirement repetition for the v3.1 method patch."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..core.bindings import BindingExpression
from ..core.contracts import (
    CapabilityRequirement,
    ParameterSpec,
    PlannerRequirementBundle,
    RepeatBlock,
    SemanticPredicate,
    TaskContract,
)
from ..core.refs import content_hash
from ..core.results import ValidationResult
from ..core.serialization import to_primitive


def _canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


def _stable_constraint_id(value: dict[str, Any]) -> str:
    canonical = {key: value[key] for key in sorted(value) if key != "constraint_id"}
    digest = hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()[:12]
    stem = re.sub(
        r"[^a-z0-9]+", "_",
        f"{canonical.get('predicate', 'predicate')}_{canonical.get('distinct_by', 'count')}".casefold(),
    ).strip("_")
    return f"cc_{stem[:48]}_{digest}"


def _predicate_roles(contract: TaskContract, predicate: str) -> set[str]:
    return {
        str(role)
        for effect in contract.target_effects
        if effect.predicate.casefold() == predicate.casefold()
        for role in effect.args
    }


class TaskContractNormalizer:
    """Normalize only formal cardinality shape; never infer a workflow."""

    def normalize(self, contract: TaskContract) -> TaskContract:
        if not isinstance(contract, TaskContract):
            raise TypeError("TaskContractNormalizer requires a TaskContract")
        raw_constraints = [dict(value) for value in contract.cardinality_constraints]
        declared_predicates = {
            str(value.get("predicate", "")).casefold() for value in raw_constraints
        }
        # A target predicate with cardinality > 1 is itself formal adapter
        # authority.  Materialize its constraint if the adapter omitted the
        # redundant list entry; no task wording or task type is consulted.
        for effect in contract.target_effects:
            if int(effect.cardinality) > 1 and effect.predicate.casefold() not in declared_predicates:
                raw_constraints.append({
                    "predicate": effect.predicate,
                    "count": int(effect.cardinality),
                    "distinct_by": str(effect.distinct_by),
                })

        normalized: list[dict[str, Any]] = []
        for raw in raw_constraints:
            predicate = str(raw.get("predicate", "")).strip()
            if not predicate:
                raise ValueError("cardinality constraint predicate must be non-empty")
            try:
                count = int(raw.get("count", 1))
            except (TypeError, ValueError) as exc:
                raise ValueError("cardinality constraint count must be an integer") from exc
            if count < 1:
                raise ValueError("cardinality constraint count must be >= 1")
            roles = _predicate_roles(contract, predicate)
            distinct_by = str(
                raw.get("distinct_by") or raw.get("role") or ""
            ).strip()
            if distinct_by and distinct_by not in roles:
                raise ValueError(
                    f"cardinality distinct_by {distinct_by!r} is not a {predicate!r} role"
                )
            shared_roles = tuple(dict.fromkeys(
                str(value).strip() for value in raw.get("shared_roles", ())
                if str(value).strip()
            ))
            if not shared_roles and count > 1:
                shared_roles = tuple(sorted(roles - ({distinct_by} if distinct_by else set())))
            if any(role not in roles or role == distinct_by for role in shared_roles):
                raise ValueError("cardinality shared_roles must be non-distinct predicate roles")
            composition_mode = str(
                raw.get("composition_mode") or ("repeat_unit" if count > 1 else "atomic")
            )
            if composition_mode not in {"atomic", "repeat_unit"}:
                raise ValueError("cardinality composition_mode must be atomic or repeat_unit")
            if composition_mode == "repeat_unit" and count < 2:
                raise ValueError("repeat_unit cardinality constraint requires count >= 2")
            item = {
                "predicate": predicate,
                "count": count,
                "distinct_by": distinct_by,
                "shared_roles": list(shared_roles),
                "composition_mode": composition_mode,
            }
            item["constraint_id"] = str(raw.get("constraint_id") or _stable_constraint_id(item))
            normalized.append(item)
        normalized.sort(key=lambda value: str(value["constraint_id"]))
        return TaskContract(
            target_effects=list(contract.target_effects),
            cardinality_constraints=normalized,
            identity_constraints=list(contract.identity_constraints),
            source=contract.source,
            confidence=contract.confidence,
            validator_id=contract.validator_id,
        )


def normalize_task_contract(contract: TaskContract) -> TaskContract:
    return TaskContractNormalizer().normalize(contract)


def normalized_constraints(contract: TaskContract) -> dict[str, dict[str, Any]]:
    normalized = normalize_task_contract(contract)
    return {
        str(value["constraint_id"]): dict(value)
        for value in normalized.cardinality_constraints
    }


def _effect_roles(effect: SemanticPredicate) -> set[str]:
    return {str(value) for value in effect.args}


def _requirement_roles(requirement: CapabilityRequirement) -> set[str]:
    roles = {item.name for item in requirement.expected_inputs}
    roles.update(item.name for item in requirement.expected_outputs)
    for predicate in (*requirement.desired_effects, *requirement.precondition_hints):
        roles.update(_effect_roles(predicate))
    return roles


def _predicate_shape_matches(required: SemanticPredicate, offered: SemanticPredicate) -> bool:
    return (
        required.predicate.casefold() == offered.predicate.casefold()
        and set(required.args).issubset(offered.args)
    )


@dataclass(frozen=True)
class RequirementInstance:
    instance_id: str
    template_requirement_id: str
    repeat_block_id: str
    repeat_index: int
    requirement: CapabilityRequirement


@dataclass(frozen=True)
class RequirementExpansion:
    templates: tuple[CapabilityRequirement, ...]
    repeat_blocks: tuple[RepeatBlock, ...]
    instances: tuple[RequirementInstance, ...]
    instance_ids_by_template: dict[str, tuple[str, ...]]

    def instance(self, instance_id: str) -> RequirementInstance:
        for value in self.instances:
            if value.instance_id == instance_id:
                return value
        raise KeyError(instance_id)


class RequirementBundleValidator:
    def validate(
        self,
        bundle: PlannerRequirementBundle,
        contract: TaskContract,
        *,
        max_repeat_count: int,
        max_runtime_occurrences: int,
    ) -> ValidationResult:
        checks: dict[str, bool] = {}
        codes: list[str] = []
        messages: list[str] = []
        requirements = list(bundle.requirements)
        blocks = list(bundle.repeat_blocks)
        requirement_ids = [item.requirement_id for item in requirements]
        block_ids = [item.block_id for item in blocks]
        checks["requirement_ids_unique"] = (
            bool(requirement_ids) and len(requirement_ids) == len(set(requirement_ids))
        )
        checks["repeat_block_ids_unique"] = len(block_ids) == len(set(block_ids))
        if not checks["requirement_ids_unique"]:
            codes.append("planner_requirement_multiplicity_invalid")
        if not checks["repeat_block_ids_unique"]:
            codes.append("planner_repeat_block_invalid")

        by_requirement = {item.requirement_id: item for item in requirements}
        membership = [value for block in blocks for value in block.ordered_requirement_ids]
        checks["one_repeat_block_per_requirement"] = len(membership) == len(set(membership))
        checks["block_requirements_exist_and_required"] = all(
            item in by_requirement and by_requirement[item].required
            for item in membership
        )
        if not checks["one_repeat_block_per_requirement"] or not checks["block_requirements_exist_and_required"]:
            codes.append("planner_repeat_block_invalid")

        try:
            constraints = normalized_constraints(contract)
            checks["task_contract_normalized"] = True
        except (TypeError, ValueError) as exc:
            constraints = {}
            checks["task_contract_normalized"] = False
            messages.append(str(exc))
            codes.append("planner_requirement_multiplicity_invalid")

        repeat_unit_constraint_ids = {
            constraint_id
            for constraint_id, constraint in constraints.items()
            if str(constraint.get("composition_mode", "")) == "repeat_unit"
        }
        block_basis_ids = [
            str(block.basis_constraint_id)
            for block in blocks
        ]
        basis_ids_unique = len(block_basis_ids) == len(set(block_basis_ids))
        repeat_unit_constraints_materialized = (
            checks["task_contract_normalized"]
            and basis_ids_unique
            and set(block_basis_ids) == repeat_unit_constraint_ids
        )
        checks["repeat_unit_constraints_materialized_once"] = (
            repeat_unit_constraints_materialized
        )
        if not repeat_unit_constraints_materialized:
            codes.append("planner_repeat_block_invalid")

        supported_counts = True
        modes_valid = True
        basis_effect_covered = True
        roles_valid = True
        repeat_range_valid = True
        for block in blocks:
            basis = constraints.get(block.basis_constraint_id)
            repeat_range_valid &= (
                2 <= int(block.count) <= int(max_repeat_count)
                and block.execution_policy == "serial"
                and bool(block.ordered_requirement_ids)
            )
            supported_counts &= bool(basis) and int(basis.get("count", -1)) == int(block.count)
            modes_valid &= bool(basis) and basis.get("composition_mode") == "repeat_unit"
            member_requirements = [
                by_requirement[item]
                for item in block.ordered_requirement_ids
                if item in by_requirement
            ]
            basis_effect_covered &= bool(basis) and any(
                effect.predicate.casefold() == str(basis.get("predicate", "")).casefold()
                and int(effect.cardinality) == 1
                for requirement in member_requirements
                for effect in requirement.desired_effects
            )
            distinct = set(block.distinct_roles)
            shared = set(block.shared_roles)
            role_map = dict(block.basis_role_map)
            available_roles = set().union(*(
                _requirement_roles(item) for item in member_requirements
            )) if member_requirements else set()
            basis_distinct = str((basis or {}).get("distinct_by", ""))
            basis_shared = set((basis or {}).get("shared_roles", ()))
            roles_valid &= (
                distinct.isdisjoint(shared)
                and all(value and value in available_roles for value in distinct | shared)
                and all(str(key) and str(value) for key, value in role_map.items())
                and (not basis_distinct or role_map.get(basis_distinct) in distinct)
                and all(role_map.get(value) in shared for value in basis_shared)
            )
        checks["repeat_counts_backed"] = supported_counts
        checks["repeat_basis_modes_valid"] = modes_valid
        checks["repeat_basis_unit_effect_covered"] = basis_effect_covered
        checks["repeat_roles_valid"] = roles_valid
        checks["repeat_range_and_policy_valid"] = repeat_range_valid
        if not supported_counts:
            codes.append("planner_repeat_count_unbacked")
        if not modes_valid or not repeat_range_valid or not basis_effect_covered:
            codes.append("planner_repeat_block_invalid")
        if not roles_valid:
            codes.append("planner_repeat_role_invalid")

        repeated = set(membership)
        expanded_count = sum(
            block.count * len(block.ordered_requirement_ids) for block in blocks
        ) + sum(
            item.requirement_id not in repeated for item in requirements
        )
        checks["expanded_occurrence_limit"] = 0 < expanded_count <= int(max_runtime_occurrences)
        if not checks["expanded_occurrence_limit"]:
            codes.append("planner_repeat_block_invalid")

        # Aggregate unit desired effects exactly as the compiler will expand
        # them.  This is a coverage proof only; it does not invent nodes.
        capacity: dict[tuple[str, tuple[str, ...]], int] = {}
        for requirement in requirements:
            multiplier = 1
            for block in blocks:
                if requirement.requirement_id in block.ordered_requirement_ids:
                    multiplier = block.count
                    break
            if not requirement.required:
                continue
            for effect in requirement.desired_effects:
                key = (effect.predicate.casefold(), tuple(sorted(map(str, effect.args))))
                capacity[key] = capacity.get(key, 0) + multiplier * max(1, int(effect.cardinality))
        target_covered = True
        for effect in contract.target_effects:
            wanted_roles = set(map(str, effect.args))
            offered = sum(
                count for (predicate, roles), count in capacity.items()
                if predicate == effect.predicate.casefold() and wanted_roles.issubset(roles)
            )
            target_covered &= offered >= max(1, int(effect.cardinality))
        checks["aggregate_task_contract_coverage"] = target_covered
        if not target_covered:
            codes.append("planner_requirement_multiplicity_invalid")

        codes = list(dict.fromkeys(codes))
        passed = all(checks.values()) and not codes
        return ValidationResult(
            level="planner_requirement_bundle",
            passed=passed,
            checks=checks,
            failure_codes=codes,
            messages=messages,
        )


class RequirementMultiplicityCompiler:
    def expand(
        self,
        bundle: PlannerRequirementBundle,
        contract: TaskContract | None = None,
    ) -> RequirementExpansion:
        if contract is not None:
            # Normalization is a deterministic integrity check at the IR
            # boundary; validation remains the caller's responsibility.
            normalize_task_contract(contract)
        by_requirement = {item.requirement_id: item for item in bundle.requirements}
        repeated = {
            requirement_id
            for block in bundle.repeat_blocks
            for requirement_id in block.ordered_requirement_ids
        }
        instances: list[RequirementInstance] = []
        ids_by_template: dict[str, list[str]] = {
            item.requirement_id: [] for item in bundle.requirements
        }
        for requirement in bundle.requirements:
            if requirement.requirement_id in repeated:
                continue
            instance_id = f"single::{requirement.requirement_id}"
            instances.append(RequirementInstance(
                instance_id, requirement.requirement_id, "", -1, requirement,
            ))
            ids_by_template[requirement.requirement_id].append(instance_id)
        for block in bundle.repeat_blocks:
            if block.execution_policy != "serial":
                raise ValueError("v3.1 supports only serial RepeatBlocks")
            for repeat_index in range(block.count):
                for requirement_id in block.ordered_requirement_ids:
                    requirement = by_requirement[requirement_id]
                    instance_id = f"{block.block_id}::{repeat_index}::{requirement_id}"
                    instances.append(RequirementInstance(
                        instance_id, requirement_id, block.block_id,
                        repeat_index, requirement,
                    ))
                    ids_by_template[requirement_id].append(instance_id)
        return RequirementExpansion(
            templates=tuple(bundle.requirements),
            repeat_blocks=tuple(bundle.repeat_blocks),
            instances=tuple(instances),
            instance_ids_by_template={
                key: tuple(value) for key, value in ids_by_template.items()
            },
        )


def repeat_block_for_instance(
    expansion: RequirementExpansion,
    instance: RequirementInstance,
) -> RepeatBlock | None:
    return next((
        block for block in expansion.repeat_blocks
        if block.block_id == instance.repeat_block_id
    ), None)


def _requirement_role_occurrences(
    requirement: CapabilityRequirement,
    role: str,
) -> tuple[tuple[str, str, str], ...]:
    """Describe a requirement role by its formal predicate wiring."""

    occurrences: list[tuple[str, str, str]] = []
    for boundary, predicates in (
        ("pre", requirement.precondition_hints),
        ("effect", requirement.desired_effects),
    ):
        for predicate in predicates:
            for argument_name, raw in predicate.args.items():
                expression: BindingExpression | None = None
                if isinstance(raw, BindingExpression):
                    expression = raw
                elif isinstance(raw, dict) and "kind" in raw:
                    try:
                        expression = BindingExpression.from_dict(raw)
                    except (KeyError, TypeError, ValueError):
                        expression = None
                source_role = expression.source_role if expression is not None else ""
                if source_role == role or raw == f"${role}":
                    occurrences.append((
                        boundary,
                        predicate.predicate.casefold(),
                        str(argument_name),
                    ))
    return tuple(sorted(occurrences))


def _requirement_parameter_descriptor(
    requirement: CapabilityRequirement,
    spec: ParameterSpec,
) -> tuple[Any, ...]:
    return (
        str(spec.semantic_type).casefold(),
        bool(spec.required),
        bool(spec.runtime_resolvable),
        str(spec.required_resolution).casefold(),
        _requirement_role_occurrences(requirement, spec.name),
    )


def _requirement_boundary_role_map(
    requirement: CapabilityRequirement,
    specs: list[ParameterSpec],
    prefix: str,
) -> dict[str, str]:
    described = [
        (_requirement_parameter_descriptor(requirement, spec), spec.name)
        for spec in specs
    ]
    described.sort(key=lambda item: (
        _canonical_json(item[0]),
        item[1],
    ))
    return {
        original: f"{prefix}_{index:03d}"
        for index, (_descriptor, original) in enumerate(described)
    }


def _requirement_source_role(raw: Any) -> str:
    expression: BindingExpression | None = None
    if isinstance(raw, BindingExpression):
        expression = raw
    elif isinstance(raw, dict) and "kind" in raw:
        try:
            expression = BindingExpression.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            expression = None
    if expression is not None:
        return str(expression.source_role)
    if isinstance(raw, str) and raw.startswith("$"):
        return raw[1:]
    return ""


def _canonical_requirement_predicate(
    predicate: SemanticPredicate,
    role_map: dict[str, str],
) -> dict[str, Any]:
    # Predicate argument names are formal semantics.  Values are represented
    # only through alpha-normalized boundary roles; concrete task instances
    # and Planner-local spellings never enter a cross-task shape.
    args = {
        str(argument_name): {
            "source_role": role_map.get(_requirement_source_role(raw), ""),
        }
        for argument_name, raw in sorted(
            predicate.args.items(), key=lambda item: str(item[0]),
        )
    }
    return {
        "predicate": predicate.predicate.casefold(),
        "args": args,
        "cardinality": int(predicate.cardinality),
        "distinct_by": str(predicate.distinct_by),
    }


def canonical_requirement_shape(
    requirement: CapabilityRequirement,
) -> dict[str, Any]:
    """Return an alpha-normalized semantic Requirement shape.

    Free-text intent/rationale, Planner IDs, and concrete source-task values
    are deliberately excluded so equivalent independent tasks share identity.
    """

    input_map = _requirement_boundary_role_map(
        requirement, requirement.expected_inputs, "input",
    )
    output_map = _requirement_boundary_role_map(
        requirement, requirement.expected_outputs, "output",
    )
    # Input roles take precedence if an old fixture reuses one spelling on
    # both boundaries; this remains deterministic and contains no raw alias.
    role_map = {**output_map, **input_map}

    def parameter_shape(spec: ParameterSpec, mapping: dict[str, str]) -> dict[str, Any]:
        return {
            "role": mapping[spec.name],
            "semantic_type": str(spec.semantic_type).casefold(),
            "required": bool(spec.required),
            "runtime_resolvable": bool(spec.runtime_resolvable),
            "required_resolution": str(spec.required_resolution).casefold(),
        }

    inputs = [parameter_shape(spec, input_map) for spec in requirement.expected_inputs]
    outputs = [parameter_shape(spec, output_map) for spec in requirement.expected_outputs]
    inputs.sort(key=_canonical_json)
    outputs.sort(key=_canonical_json)
    preconditions = [
        _canonical_requirement_predicate(predicate, role_map)
        for predicate in requirement.precondition_hints
    ]
    effects = [
        _canonical_requirement_predicate(predicate, role_map)
        for predicate in requirement.desired_effects
    ]
    preconditions.sort(key=_canonical_json)
    effects.sort(key=_canonical_json)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "preconditions": preconditions,
        "effects": effects,
        "required": bool(requirement.required),
    }


def requirement_instance_semantic_shape(
    instance: RequirementInstance,
    expansion: RequirementExpansion,
    contract: TaskContract,
) -> dict[str, Any]:
    block = repeat_block_for_instance(expansion, instance)
    if block is None:
        repeat_shape: dict[str, Any] = {"repeated": False}
    else:
        basis = normalized_constraints(contract)[block.basis_constraint_id]
        repeat_shape = {
            "repeated": True,
            "count": int(block.count),
            "repeat_index": int(instance.repeat_index),
            "position_in_iteration": list(
                block.ordered_requirement_ids
            ).index(instance.template_requirement_id),
            "basis": {
                "predicate": str(basis["predicate"]).casefold(),
                "count": int(basis["count"]),
                "distinct_by": str(basis.get("distinct_by", "")),
                "shared_roles": sorted(map(
                    str, basis.get("shared_roles", ()),
                )),
                "composition_mode": str(
                    basis.get("composition_mode", ""),
                ),
            },
        }
    return {
        "requirement": canonical_requirement_shape(instance.requirement),
        "repeat": repeat_shape,
    }


def requirement_instance_shape_id(
    instance: RequirementInstance,
    expansion: RequirementExpansion,
    contract: TaskContract,
) -> str:
    return content_hash(requirement_instance_semantic_shape(
        instance, expansion, contract,
    ))


__all__ = [
    "RequirementBundleValidator",
    "RequirementExpansion",
    "RequirementInstance",
    "RequirementMultiplicityCompiler",
    "TaskContractNormalizer",
    "canonical_requirement_shape",
    "normalize_task_contract",
    "normalized_constraints",
    "repeat_block_for_instance",
    "requirement_instance_semantic_shape",
    "requirement_instance_shape_id",
]
