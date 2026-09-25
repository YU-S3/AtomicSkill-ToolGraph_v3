"""Bounded v3.2 Tool IR: ACTION / IF / FOR_EACH / STOP_WHEN / RETURN.

The IR is deliberately declarative.  It contains no Python, shell, network, or
filesystem capability.  All conditions read only Tool inputs, local variables,
the current action catalog, semantic evidence, or binding evidence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .proposal import ToolProgramOp
from .value_reference import resolve_tool_value_reference, bounded_count_reference
from .ir_contract import collection_source_names

COLLECTION_SOURCES = frozenset(collection_source_names())


CONDITION_OPERATORS = frozenset({
    "exists", "not_exists", "equals", "not_equals",
    "contains", "empty", "non_empty",
})
CONDITION_SOURCES = (
    "tool_input", "local_variable", "action_catalog",
    "semantic_evidence", "binding_evidence",
)
ACTION_CATALOG_ENTRY_FIELDS = (
    "action_id", "revision", "action_type", "arguments",
)
TOOL_IR_MAX_NESTING_DEPTH = 4


@dataclass
class ToolExecutionState:
    bindings: dict[str, Any] = field(default_factory=dict)
    local: dict[str, Any] = field(default_factory=dict)
    catalog: list[dict[str, Any]] = field(default_factory=list)
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    binding_evidence: list[dict[str, Any]] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    executed_nodes: list[str] = field(default_factory=list)
    path_tokens: list[str] = field(default_factory=list)
    validated_paths: list[str] = field(default_factory=list)
    unvalidated_paths: list[str] = field(default_factory=list)
    loop_iteration_counts: dict[str, int] = field(default_factory=dict)
    stop_condition_witnesses: list[str] = field(default_factory=list)
    executed_action_count: int = 0
    max_actions: int = 0
    executed_control_step_count: int = 0
    max_control_steps: int = 0
    step_effect_results: list[dict[str, Any]] = field(default_factory=list)
    failure_code: str = ""
    failure_message: str = ""
    attempted_action: dict[str, Any] = field(default_factory=dict)
    program_node_id: str = ""
    catalog_revision: int | None = None
    collection_observations: list[dict[str, Any]] = field(default_factory=list)
    condition_observations: list[dict[str, Any]] = field(default_factory=list)
    iteration_observations: list[dict[str, Any]] = field(default_factory=list)
    value_reference_observations: list[dict[str, Any]] = field(default_factory=list)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def walk_program_nodes(
    program: Any,
    *,
    max_depth: int = TOOL_IR_MAX_NESTING_DEPTH,
    with_context: bool = False,
) -> list[Any]:
    """Single bounded recursive walker for every Tool IR consumer.

    A Tool proposal is untrusted structured output.  In particular, nested
    branch members may be scalars, mappings instead of lists, cyclic objects,
    or adversarially deep trees.  Report those shapes with stable Tool-IR
    errors instead of leaking Python ``TypeError``/``RecursionError`` out of a
    static validation gate.
    """

    result: list[Any] = []
    active_containers: set[int] = set()

    def visit(nodes: Any, depth: int, prefix: str) -> None:
        if depth > max_depth:
            raise ValueError("tool_ir_recursion_depth_exceeded")
        if not isinstance(nodes, (list, tuple)):
            raise ValueError("tool_ir_schema_invalid: nested program must be a list")
        container_id = id(nodes)
        if container_id in active_containers:
            raise ValueError("tool_ir_recursion_depth_exceeded")
        active_containers.add(container_id)
        try:
            for index, raw in enumerate(nodes):
                if not isinstance(raw, Mapping):
                    raise ValueError("tool_ir_schema_invalid")
                # Preserve ordinary dict identity so validators can apply
                # canonical structural normalization to the actual tree.
                node = raw if isinstance(raw, dict) else dict(raw)
                node_id = str(node.get("node_id", f"{prefix}[{index}]"))
                path_id = f"{prefix}/{node_id}"
                result.append(
                    (node, depth, path_id) if with_context else node
                )
                opcode = str(node.get("op", ""))
                if opcode == "IF":
                    then_branch = node.get("then_branch", [])
                    else_branch = node.get("else_branch", [])
                    visit(then_branch, depth + 1, f"{path_id}/then")
                    visit(else_branch, depth + 1, f"{path_id}/else")
                elif opcode == "FOR_EACH":
                    visit(
                        node.get("body", []), depth + 1,
                        f"{path_id}/body",
                    )
        finally:
            active_containers.remove(container_id)

    visit(program, 0, "program")
    return result


def normalize_tool_program(value: Any) -> list[dict[str, Any]]:
    """Return a validated flat list of top-level IR nodes.

    Nested branches are retained verbatim and are recursively checked by
    ``ToolStaticValidator``.  Unknown opcodes raise ``ValueError`` rather than
    being silently ignored.
    """

    if not isinstance(value, list) or not value:
        raise ValueError("tool_ir_schema_invalid: program must be a non-empty list")
    program: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool_ir_schema_invalid: program[{index}] is not an object")
        # The normalized program is its own canonical tree.  Static
        # normalization must never depend on mutating the Agent proposal's
        # shallow-shared nested branches.
        node = copy.deepcopy(dict(raw))
        opcode = str(node.get("op", ""))
        if opcode not in {item.value for item in ToolProgramOp}:
            raise ValueError(f"tool_ir_opcode_unsupported: {opcode}")
        if not str(node.get("node_id", "")).strip():
            raise ValueError("tool_ir_schema_invalid: missing node_id")
        program.append(node)
    return program


def _lookup(source: str, field_name: str, state: ToolExecutionState) -> Any:
    source = source.casefold()
    if source == "tool_input":
        return state.bindings.get(field_name)
    if source == "local_variable":
        return state.local.get(field_name)
    if source == "action_catalog":
        if field_name in {"length", "count", "size"}:
            return len(state.catalog)
        values = [
            item.get(field_name)
            for item in state.catalog
            if isinstance(item, Mapping) and field_name in item
        ]
        return values
    if source == "semantic_evidence":
        if field_name in {"length", "count", "size"}:
            return len(state.semantic_facts)
        values = [
            item.get(field_name)
            for item in state.semantic_facts
            if isinstance(item, Mapping) and field_name in item
        ]
        return values
    if source == "binding_evidence":
        if field_name in {"length", "count", "size"}:
            return len(state.binding_evidence)
        values = [
            item.get(field_name)
            for item in state.binding_evidence
            if isinstance(item, Mapping) and field_name in item
        ]
        return values
    return None


def validate_match_condition_shape(condition: Any) -> tuple[dict[str, Any], str]:
    """The narrow selector-condition shape shared by static and execution."""
    def invalid(message: str) -> None:
        raise ValueError(f"tool_ir_condition_match_invalid: {message}")

    if not isinstance(condition, Mapping) or set(condition) - {"match", "op"}:
        invalid("match cannot be mixed with legacy fields")
    operator = condition.get("op", "exists")
    if not isinstance(operator, str) or operator not in {"exists", "not_exists"}:
        invalid("match supports only exists/not_exists")
    selector = condition.get("match")
    if not isinstance(selector, Mapping) or set(selector) - {"source", "where", "project", "distinct"}:
        invalid("unexpected selector fields")
    if selector.get("source") not in {"action_catalog", "semantic_evidence"}:
        invalid("match requires a public structured source")
    if "distinct" in selector and not isinstance(selector["distinct"], bool):
        invalid("distinct must be boolean")
    where, project = selector.get("where"), selector.get("project")
    discriminator = 'action_type' if selector['source'] == 'action_catalog' else 'predicate'
    if not isinstance(where, Mapping) or not isinstance(where.get(discriminator), str) or not where[discriminator]:
        invalid(f"where.{discriminator} is required")
    if ('predicate' if discriminator == 'action_type' else 'action_type') in where:
        invalid('source discriminator mismatch')
    if (not isinstance(project, Mapping) or set(project) != {"kind", "role"}
            or project.get("kind") != "argument" or not isinstance(project.get("role"), str)
            or not project["role"]):
        invalid("project must name an argument role")
    semantic = where.get("semantic_compatible_with")
    if "semantic_compatible_with" in where:
        if (not isinstance(semantic, Mapping) or set(semantic) - {"source", "field", "semantic_type"}
                or not isinstance(semantic.get("source"), str)
                or semantic.get("source") not in {"tool_input", "local_variable"}
                or not isinstance(semantic.get("field"), str) or not semantic["field"]
                or not isinstance(where.get("argument_role"), str) or not where["argument_role"]
                or ("semantic_type" in semantic and not isinstance(semantic["semantic_type"], str))):
            invalid("semantic comparison must name an input/local and argument role")
    elif "argument_role" in where:
        invalid("argument_role requires semantic_compatible_with")
    for role, value in where.items():
        if role != "semantic_compatible_with" and isinstance(value, (Mapping, list, tuple)):
            if not is_bound_selector_value(value):
                invalid("direct filters must be scalar or an input/local bound value")
    return dict(selector), str(operator)


def evaluate_condition(condition: Any, state: ToolExecutionState, *, semantic_compatible: Any = None) -> bool:
    if isinstance(condition, Mapping) and "match" in condition:
        selector, operator = validate_match_condition_shape(condition)
        for role, reference in selector['where'].items():
            if role not in SELECTOR_META_FIELDS and isinstance(reference,Mapping):
                values = state.bindings if reference['source'] == 'tool_input' else state.local
                if reference['field'] not in values or values[reference['field']] in (None,''):
                    raise ValueError('tool_ir_condition_reference_unavailable')
        semantic = selector["where"].get("semantic_compatible_with")
        if semantic is not None:
            values = state.bindings if semantic["source"] == "tool_input" else state.local
            if semantic["field"] not in values or values[semantic["field"]] in (None, ""):
                raise ValueError("tool_ir_condition_reference_unavailable")
            if not callable(semantic_compatible):
                raise ValueError("tool_ir_condition_matcher_unavailable")
        # The runner refreshes this catalog after every accepted action. Never
        # interpret corrupt or stale entries as a negative target observation.
        for entry in state.catalog if selector['source'] == 'action_catalog' else []:
            if (not isinstance(entry, Mapping) or not isinstance(entry.get("arguments"), Mapping)
                    or not entry.get("action_id") or not isinstance(entry.get("revision"), int)
                    or isinstance(entry.get("revision"), bool)
                    or (state.catalog_revision is not None and entry["revision"] != state.catalog_revision)):
                raise ValueError("tool_ir_condition_catalog_invalid")
            if entry.get("action_type") == selector["where"]["action_type"]:
                roles = {selector["project"]["role"]}
                if semantic is not None:
                    roles.add(selector["where"]["argument_role"])
                if any(entry["arguments"].get(role) in (None, "") for role in roles):
                    raise ValueError("tool_ir_condition_projection_invalid")
        values = resolve_collection(selector, state, semantic_compatible=semantic_compatible)
        if any(value in (None, "") for value in values):
            raise ValueError("tool_ir_condition_projection_invalid")
        return bool(values) if operator == "exists" else not values
    condition = _as_mapping(condition)
    source = str(condition.get("source", "")).casefold()
    field_name = str(condition.get("field", ""))
    operator = str(condition.get("op", "exists")).casefold()
    expected = condition.get("value")
    if operator not in CONDITION_OPERATORS:
        raise ValueError("tool_ir_condition_operator_unsupported")

    if source not in CONDITION_SOURCES:
        raise ValueError("tool_ir_condition_source_unsupported")
    if not field_name:
        raise ValueError("tool_ir_condition_requires_field")

    actual = _lookup(source, field_name, state)
    if operator == "exists":
        return bool(actual) if not isinstance(actual, list) else bool(actual)
    if operator == "not_exists":
        return not (bool(actual) if not isinstance(actual, list) else bool(actual))
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "contains":
        if isinstance(actual, (list, tuple, set)):
            return expected in actual
        if isinstance(actual, str):
            return str(expected) in actual
        return False
    if operator == "empty":
        return not bool(actual)
    if operator == "non_empty":
        return bool(actual)
    raise ValueError("tool_ir_condition_operator_unsupported")


SELECTOR_META_FIELDS = frozenset({
    "action_type",
    "predicate",
    "argument_role",
    "semantic_compatible_with",
})
# Compatibility alias for existing imports.  New contracts use the public
# name so the interpreter and static validator share one vocabulary.
_SELECTOR_META_FIELDS = SELECTOR_META_FIELDS


def is_bound_selector_value(value):
    return (isinstance(value, Mapping) and set(value) == {'source', 'field'}
        and value.get('source') in {'tool_input', 'local_variable'}
        and isinstance(value.get('field'), str) and bool(value['field']))


def _selector_entries(
    source: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    state: ToolExecutionState,
    *,
    semantic_compatible: Any = None,
) -> list[Mapping[str, Any]]:
    """Apply one declarative selector without introducing a new opcode."""

    selected: list[Mapping[str, Any]] = []
    where = _as_mapping(source.get("where"))
    source_kind = str(source.get("source", "")).casefold()
    for entry in entries:
        item = dict(entry)
        if where.get("action_type") is not None and str(
            item.get("action_type", "")
        ) != str(where.get("action_type", "")):
            continue
        if where.get("predicate") is not None and str(
            item.get("predicate", "")
        ) != str(where.get("predicate", "")):
            continue
        arguments = _as_mapping(
            item.get("arguments")
            if source_kind == "action_catalog"
            else item.get("args")
        )
        ok = True
        for raw_role, expected in where.items():
            if raw_role in SELECTOR_META_FIELDS:
                continue
            # Primitive argument filters use their exact action-schema role as
            # the direct ``where`` key.  There is no wrapper or ``*_in``
            # dialect in Tool IR v1.
            actual = arguments.get(str(raw_role))
            if source_kind in {'semantic_evidence','action_catalog'} and isinstance(expected, Mapping):
                if not is_bound_selector_value(expected):
                    raise ValueError('tool_ir_selector_invalid_bound_value')
                values = state.bindings if expected['source'] == 'tool_input' else state.local
                if expected['field'] not in values or values[expected['field']] in (None, ''):
                    raise ValueError('tool_ir_condition_reference_unavailable')
                expected = values[expected['field']]
            if actual != expected:
                ok = False
                break
        semantic = _as_mapping(where.get("semantic_compatible_with"))
        if ok and semantic:
            anchor = _lookup(
                str(semantic.get("source", "")),
                str(semantic.get("field", "")),
                state,
            )
            role = str(semantic.get("argument_role") or where.get("argument_role") or "")
            value = arguments.get(role)
            if not callable(semantic_compatible):
                ok = False
            elif not bool(semantic_compatible(
                role=role,
                concrete_value=value,
                semantic_anchor=anchor,
                semantic_type=str(semantic.get("semantic_type", "entity")),
            )):
                ok = False
        if ok:
            selected.append(item)
    return selected


def _project(entry: Mapping[str, Any], source: Mapping[str, Any]) -> Any:
    project = _as_mapping(source.get("project"))
    source_kind = str(source.get("source", "")).casefold()
    if not project:
        field_name = str(source.get("field", ""))
        return entry.get(field_name)
    kind = str(project.get("kind", "field")).casefold()
    if kind == "argument":
        role = str(project.get("role", ""))
        if source_kind == "action_catalog":
            arguments = _as_mapping(entry.get("arguments"))
        else:
            arguments = _as_mapping(entry.get("args"))
        return arguments.get(role)
    field_name = str(project.get("field", ""))
    return entry.get(field_name)


def resolve_collection(
    collection_source: Any,
    state: ToolExecutionState,
    *,
    semantic_compatible: Any = None,
) -> list[Any]:
    source = _as_mapping(collection_source)
    kind = str(source.get("source", "")).casefold()
    if kind not in COLLECTION_SOURCES:
        raise ValueError('tool_ir_collection_source_unsupported')
    if kind == 'bounded_count':
        ref = bounded_count_reference(source)
        count = resolve_tool_value_reference(ref, state)
        if type(count) is not int or count < 0 or count > state.max_actions:
            raise ValueError('tool_ir_bounded_count_invalid: count outside finite Tool bound')
        values = list(range(count))
    elif kind in {"tool_input", "local_variable"}:
        value = (
            state.bindings.get(str(source.get("field", "")))
            if kind == "tool_input"
            else state.local.get(str(source.get("field", "")))
        )
        values = list(value) if isinstance(value, (list, tuple)) else []
    elif kind == "action_catalog":
        values = [
            _project(item, source)
            for item in _selector_entries(
                source, state.catalog, state,
                semantic_compatible=semantic_compatible,
            )
        ]
    elif kind == "semantic_evidence":
        values = [
            _project(item, source)
            for item in _selector_entries(
                source, state.semantic_facts, state,
                semantic_compatible=semantic_compatible,
            )
        ]
    elif kind == "binding_evidence":
        values = [
            _project(item, source)
            for item in _selector_entries(
                source, state.binding_evidence, state,
                semantic_compatible=semantic_compatible,
            )
        ]
    elif kind == "local_deterministic":
        raw_values = source.get("values", [])
        values = list(raw_values) if isinstance(raw_values, (list, tuple)) else []
        project = _as_mapping(source.get("project"))
        if project:
            values = [
                _project(item, source)
                for item in values
                if isinstance(item, Mapping)
            ]
    else:
        raise ValueError("tool_ir_collection_source_unsupported")
    if bool(source.get("distinct", False)):
        unique: list[Any] = []
        for value in values:
            try:
                duplicate = value in unique
            except TypeError:
                duplicate = any(repr(value) == repr(item) for item in unique)
            if not duplicate:
                unique.append(value)
        values = unique
    return values


def normalize_return_output_sources(
    node: Mapping[str, Any],
    output_roles: Any = (),
) -> dict[str, Any]:
    """Repair a common structural mistake where RETURN supplies a source spec
    instead of the required ``{output_role: source_spec}`` mapping."""

    raw = _as_mapping(node.get("output_sources"))
    if not raw:
        return {}
    roles = {str(role) for role in output_roles}
    source_spec_keys = {
        "source", "field", "where", "project", "kind", "constant",
        "value", "source_role", "distinct",
    }
    # An Atomic output role may itself be named ``source`` or ``kind``.  Only
    # treat the whole mapping as the legacy single selector when the reserved
    # discriminator has its scalar selector shape; a nested mapping is already
    # the canonical {output_role: source_spec} form.
    legacy_source = raw.get("source")
    legacy_kind = raw.get("kind")
    looks_like_source_spec = (
        isinstance(legacy_source, str) and bool(legacy_source.strip())
    ) or (
        isinstance(legacy_kind, str) and bool(legacy_kind.strip())
    )
    if set(raw) <= source_spec_keys and looks_like_source_spec:
        if len(roles) == 1:
            return {next(iter(roles)): dict(raw)}
    return dict(raw)


def resolve_return_sources(
    output_sources: Mapping[str, Any],
    state: ToolExecutionState,
    *,
    semantic_compatible: Any = None,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically resolve RETURN outputs and attach evidence refs."""

    outputs: dict[str, Any] = {}
    evidence_refs: list[str] = []
    for role, raw in _as_mapping(output_sources).items():
        spec = _as_mapping(raw) if isinstance(raw, Mapping) else {
            "source": "tool_input",
            "field": role,
        }
        if "source" not in spec and spec.get("kind") == "skill_input":
            spec = {**spec, "source": "tool_input", "field": str(spec.get("source_role", role))}
        elif "source" not in spec and spec.get("kind") == "local_variable":
            spec = {**spec, "source": "local_variable", "field": str(spec.get("source_role", role))}
        elif "source" not in spec and spec.get("kind") == "constant":
            spec = {**spec, "source": "constant", "value": spec.get("constant")}
        source = str(spec.get("source", "tool_input")).casefold()
        field_name = str(spec.get("field", spec.get("source_role", role)))
        if source in {"semantic_evidence", "binding_evidence", "action_catalog"} and (
            "where" in spec or "project" in spec
        ):
            values = resolve_collection(
                {
                    "source": source,
                    "field": field_name,
                    "where": dict(spec.get("where") or {}),
                    "project": dict(spec.get("project") or {}),
                    "distinct": bool(spec.get("distinct", True)),
                },
                state,
                semantic_compatible=semantic_compatible,
            )
            outputs[role] = values[-1] if values else None
            if values:
                evidence_refs.append(f"{source}:{field_name}")
            continue
        if source == "tool_input":
            outputs[role] = resolve_tool_value_reference(spec, state)
            evidence_refs.append(f"tool_input:{field_name}")
        elif source == "local_variable":
            outputs[role] = resolve_tool_value_reference(spec, state)
            evidence_refs.append(f"tool_local:{field_name}")
        elif source == "semantic_evidence":
            values = [
                item.get(field_name)
                for item in state.semantic_facts
                if isinstance(item, Mapping) and field_name in item
            ]
            outputs[role] = values[-1] if values else None
            if values:
                evidence_refs.append(f"semantic_evidence:{field_name}")
        elif source == "binding_evidence":
            values = [
                item.get(field_name)
                for item in state.binding_evidence
                if isinstance(item, Mapping) and field_name in item
            ]
            outputs[role] = values[-1] if values else None
            if values:
                evidence_refs.append(f"binding_evidence:{field_name}")
        elif source == "constant":
            outputs[role] = spec.get("value")
        else:
            raise ValueError("tool_ir_return_source_unsupported")
    return outputs, evidence_refs


def program_paths(program: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    visits = walk_program_nodes(program, with_context=True)
    paths = [str(path_id) for _node, _depth, path_id in visits]
    path_nodes = {
        str(path_id): [str(node.get("node_id", ""))]
        for node, _depth, path_id in visits
    }
    return {
        "path_ids": sorted(set(paths)),
        "paths": {path_id: path_nodes.get(path_id, []) for path_id in paths},
    }


__all__ = [
    "ACTION_CATALOG_ENTRY_FIELDS",
    "CONDITION_OPERATORS",
    "CONDITION_SOURCES",
    "SELECTOR_META_FIELDS",
    "TOOL_IR_MAX_NESTING_DEPTH",
    "ToolExecutionState",
    "evaluate_condition",
    "normalize_tool_program",
    "_SELECTOR_META_FIELDS",
    "normalize_return_output_sources",
    "walk_program_nodes",
    "program_paths",
    "resolve_collection",
    "resolve_return_sources",
]
