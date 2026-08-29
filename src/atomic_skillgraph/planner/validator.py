"""Fail-closed RuntimeLinearPlan validation without semantic auto-repair."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.edges import GraphEdgeType
from ..core.contracts import IdentityRelation
from ..core.results import RuntimeLinearPlan, ValidationResult
from ..core.status import RuntimeMode, skill_status_usable
from ..knowledge.graph_store import GraphStore
from ..knowledge.skill_registry import SkillRegistry


_STRING_LIKE_TYPES = {"entity", "object", "string", "str"}
_NUMBER_LIKE_TYPES = {"integer", "int", "number", "float"}
_ARRAY_LIKE_TYPES = {"array", "list"}
_OBJECT_LIKE_TYPES = {"object_map", "dict", "mapping"}


def _type_family(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized in _STRING_LIKE_TYPES:
        return "string_like"
    if normalized in _NUMBER_LIKE_TYPES:
        return "number_like"
    if normalized in _ARRAY_LIKE_TYPES:
        return "array_like"
    if normalized in _OBJECT_LIKE_TYPES:
        return "object_like"
    return normalized


def _semantic_types_compatible(source: str, target: str) -> bool:
    return bool(source and target and _type_family(source) == _type_family(target))


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
        if not str(constraint.get("role", "")).strip() or count <= 0:
            return False
        if count > 1 and not str(constraint.get("distinct_by", "")).strip():
            return False
    return all(
        str(item.left_role).strip() and str(item.right_role).strip()
        for item in contract.identity_constraints
    )


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
    ) -> ValidationResult:
        checks: dict[str, bool] = {}
        errors: list[str] = []
        messages: list[str] = []

        by_step = {item.step_id: item for item in plan.occurrences}
        checks["occurrence_limit"] = 0 < len(by_step) == len(plan.occurrences) <= self.max_occurrences
        checks["control_sequence_complete_unique"] = (
            len(plan.control_sequence) == len(by_step)
            and len(set(plan.control_sequence)) == len(plan.control_sequence)
            and set(plan.control_sequence) == set(by_step)
        )
        if not checks["occurrence_limit"] or not checks["control_sequence_complete_unique"]:
            errors.append("planner_graph_invalid")
            messages.append("control sequence must contain every occurrence exactly once")
        position = {step: index for index, step in enumerate(plan.control_sequence)}

        refs_ok = True
        harness_ok = True
        atomics = {}
        for occurrence in plan.occurrences:
            try:
                atomic = self.skills.get_atomic(occurrence.node_ref)
                atomics[occurrence.step_id] = atomic
                refs_ok &= skill_status_usable(atomic.status, mode)
                profiles = atomic.metadata.get("harness_profiles") or []
                harness_ok &= not profiles or harness_profile in profiles
            except KeyError:
                refs_ok = False
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
                    if not _semantic_types_compatible(source_type, target_type):
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
                        known_source and not _semantic_types_compatible(known_source, source_type)
                    ) or (
                        known_target and not _semantic_types_compatible(known_target, target_type)
                    ):
                        existing_valid = False
        checks["edges_forward_only"] = forward
        incident_steps = {
            step
            for edge in plan.data_edges + plan.dependency_edges
            for step in (edge.source_step, edge.target_step)
            if step in by_step
        }
        checks["no_disconnected_occurrence"] = (
            len(by_step) == 1 or set(by_step) == incident_steps
        )
        checks["edge_types_valid"] = edge_types
        checks["edge_ids_unique"] = edge_ids_unique or not edge_ids
        checks["edge_roles_valid"] = edge_roles
        checks["edge_semantic_types_compatible"] = edge_semantic_types
        checks["edge_origin_valid"] = edge_origin and existing_valid
        checks["one_authoritative_producer"] = all(count == 1 for count in producer_count.values())
        if not all((
            forward, edge_types, checks["edge_ids_unique"], edge_roles,
            edge_semantic_types, edge_origin, existing_valid,
            checks["one_authoritative_producer"], checks["no_disconnected_occurrence"],
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

        offered_effects, task_role_usage = _project_plan_effects(plan, atomics)
        checks["task_contract_effect_coverage"] = _effects_cover_occurrences(
            plan.task_contract.target_effects, offered_effects
        )
        if plan.task_contract.target_effects and not checks["task_contract_effect_coverage"]:
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
        if not checks["identity_cardinality_preserved"]:
            errors.append("task_contract_mismatch")

        errors = list(dict.fromkeys(errors))
        return ValidationResult(
            level="planner", passed=not errors and all(checks.values()), checks=checks,
            failure_codes=errors, messages=messages,
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
    incident_steps = {
        step
        for edge in plan.data_edges + plan.dependency_edges
        for step in (edge.source_step, edge.target_step)
        if step in by_id
    }
    no_disconnected = len(by_id) == 1 or set(by_id) == incident_steps
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
            except Exception:
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
        "no_disconnected_occurrence": no_disconnected,
        "data_flow_expression_consistent": expression_consistency,
    }
    return ValidationResult("planner", all(checks.values()), checks, [] if all(checks.values()) else ["planner_graph_invalid"])
