"""Bounded v3.2 Tool IR: ACTION / IF / FOR_EACH / STOP_WHEN / RETURN.

The IR is deliberately declarative.  It contains no Python, shell, network, or
filesystem capability.  All conditions read only Tool inputs, local variables,
the current action catalog, semantic evidence, or binding evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .proposal import ToolProgramOp


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


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def resolve_collection(collection_source: Any, state: ToolExecutionState) -> list[Any]:
    source = _as_mapping(collection_source)
    kind = str(source.get("source", "")).casefold()
    if kind == "tool_input":
        value = state.bindings.get(str(source.get("field", "")))
        return list(value) if isinstance(value, (list, tuple)) else []
    if kind == "local_variable":
        value = state.local.get(str(source.get("field", "")))
        return list(value) if isinstance(value, (list, tuple)) else []
    if kind == "action_catalog":
        field_name = str(source.get("field", ""))
        values: list[Any] = []
        for item in state.catalog:
            if field_name and field_name in item:
                values.append(item[field_name])
        return values
    if kind == "semantic_evidence":
        field_name = str(source.get("field", ""))
        values = []
        for item in state.semantic_facts:
            if field_name and field_name in item:
                values.append(item[field_name])
        return values
    if kind == "local_deterministic":
        values = source.get("values")
        return list(values) if isinstance(values, (list, tuple)) else []
    raise ValueError("tool_ir_collection_source_unsupported")


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
    "ToolExecutionState",
    "evaluate_condition",
    "normalize_tool_program",
    "program_paths",
    "resolve_collection",
    "resolve_return_sources",
]
