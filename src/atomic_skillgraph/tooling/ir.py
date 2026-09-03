"""Bounded v3.2 Tool IR: ACTION / IF / FOR_EACH / STOP_WHEN / RETURN.

The IR is deliberately declarative.  It contains no Python, shell, network, or
filesystem capability.  All conditions read only Tool inputs, local variables,
the current action catalog, semantic evidence, or binding evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .proposal import ToolProgramOp


CONDITION_OPERATORS = frozenset({
    "exists", "not_exists", "equals", "not_equals",
    "contains", "empty", "non_empty",
})


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
    validated_paths: list[str] = field(default_factory=list)
    unvalidated_paths: list[str] = field(default_factory=list)
    loop_iteration_counts: dict[str, int] = field(default_factory=dict)
    stop_condition_witnesses: list[str] = field(default_factory=list)
    executed_action_count: int = 0
    max_actions: int = 0
    step_effect_results: list[dict[str, Any]] = field(default_factory=list)
    failure_code: str = ""
    failure_message: str = ""
    program_node_id: str = ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def walk_program_nodes(program: Any) -> list[dict[str, Any]]:
    """Single recursive walker for every Tool IR consumer."""

    result: list[dict[str, Any]] = []
    for raw in program or ():
        if not isinstance(raw, Mapping):
            raise ValueError("tool_ir_schema_invalid")
        node = dict(raw)
        result.append(node)
        opcode = str(node.get("op", ""))
        if opcode == "IF":
            result.extend(walk_program_nodes(node.get("then_branch")))
            result.extend(walk_program_nodes(node.get("else_branch")))
        elif opcode == "FOR_EACH":
            result.extend(walk_program_nodes(node.get("body")))
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
        node = dict(raw)
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


def evaluate_condition(condition: Any, state: ToolExecutionState) -> bool:
    condition = _as_mapping(condition)
    source = str(condition.get("source", "")).casefold()
    field_name = str(condition.get("field", ""))
    operator = str(condition.get("op", "exists")).casefold()
    expected = condition.get("value")
    if operator not in CONDITION_OPERATORS:
        raise ValueError("tool_ir_condition_operator_unsupported")

    if source not in {
        "tool_input", "local_variable", "action_catalog",
        "semantic_evidence", "binding_evidence",
    }:
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
            if raw_role in {"action_type", "argument_role", "semantic_compatible_with"}:
                continue
            if raw_role.endswith("_in"):
                roles = [str(value) for value in expected] if isinstance(expected, (list, tuple)) else [str(expected)]
            else:
                roles = [str(raw_role)]
            for role in roles:
                actual = arguments.get(role)
                if expected is not None and actual != expected:
                    ok = False
                    break
            if not ok:
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
    if kind in {"tool_input", "local_variable"}:
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


def resolve_return_sources(
    output_sources: Mapping[str, Any],
    state: ToolExecutionState,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically resolve RETURN outputs and attach evidence refs."""

    outputs: dict[str, Any] = {}
    evidence_refs: list[str] = []
    for role, raw in _as_mapping(output_sources).items():
        spec = _as_mapping(raw) if isinstance(raw, Mapping) else {
            "source": "tool_input",
            "field": role,
        }
        source = str(spec.get("source", "tool_input")).casefold()
        field_name = str(spec.get("field", role))
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
            )
            outputs[role] = values[-1] if values else None
            if values:
                evidence_refs.append(f"{source}:{field_name}")
            continue
        if source == "tool_input":
            outputs[role] = state.bindings.get(field_name)
            evidence_refs.append(f"tool_input:{field_name}")
        elif source == "local_variable":
            outputs[role] = state.local.get(field_name)
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


def _walk_program(
    program: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    state: ToolExecutionState,
    paths: list[str],
    path_nodes: dict[str, list[str]],
) -> None:
    """Static path identity walk; branch coverage is recorded later at runtime."""

    for index, node in enumerate(program):
        node_id = str(node.get("node_id", f"{prefix}[{index}]"))
        path_id = f"{prefix}/{node_id}"
        path_nodes.setdefault(path_id, [node_id])
        opcode = str(node.get("op", ""))
        if opcode == ToolProgramOp.IF.value:
            _walk_program(
                _as_mapping(node).get("then_branch") or (),
                prefix=f"{path_id}/then",
                state=state,
                paths=paths,
                path_nodes=path_nodes,
            )
            _walk_program(
                _as_mapping(node).get("else_branch") or (),
                prefix=f"{path_id}/else",
                state=state,
                paths=paths,
                path_nodes=path_nodes,
            )
        elif opcode == ToolProgramOp.FOR_EACH.value:
            _walk_program(
                _as_mapping(node).get("body") or (),
                prefix=f"{path_id}/body",
                state=state,
                paths=paths,
                path_nodes=path_nodes,
            )
        paths.append(path_id)


def program_paths(program: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    state = ToolExecutionState()
    paths: list[str] = []
    path_nodes: dict[str, list[str]] = {}
    _walk_program(program, prefix="program", state=state, paths=paths, path_nodes=path_nodes)
    return {
        "path_ids": sorted(set(paths)),
        "paths": {path_id: path_nodes.get(path_id, []) for path_id in paths},
    }


__all__ = [
    "CONDITION_OPERATORS",
    "ToolExecutionState",
    "evaluate_condition",
    "normalize_tool_program",
    "walk_program_nodes",
    "program_paths",
    "resolve_collection",
    "resolve_return_sources",
]
