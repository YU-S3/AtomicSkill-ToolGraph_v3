"""Static and path validation for v3.2 ToolProposal / Tool IR.

This module is deliberately generic: it validates declarative IR against the
Atomic contract and Harness predicate/action interface.  It contains no task
family, object, or benchmark workflow knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..core.bindings import (
    BindingExprKind,
    BindingExpression,
    resolution_satisfies,
)
from ..core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from ..core.results import ValidationResult
from ..core.semantic_types import semantic_types_compatible
from ..core.serialization import to_primitive
from .ir import (
    CONDITION_OPERATORS,
    normalize_return_output_sources,
    normalize_tool_program,
    program_paths,
    walk_program_nodes,
)
from .proposal import RuntimeAutomationAtomicDraft, ToolProposal


_OPCODES = {"ACTION", "IF", "FOR_EACH", "STOP_WHEN", "RETURN"}
_CONDITION_SOURCES = {
    "tool_input", "local_variable", "action_catalog",
    "semantic_evidence", "binding_evidence",
}
_COLLECTION_SOURCES = {
    "tool_input", "local_variable", "action_catalog",
    "semantic_evidence", "binding_evidence", "local_deterministic",
}
_RETURN_SOURCES = {
    "tool_input", "local_variable", "semantic_evidence",
    "binding_evidence", "constant",
}
_INPUT_BINDING_KINDS = {
    "current_occurrence_anchor",
    "current_confirmed_binding",
    "current_candidate_binding",
    "data_flow",
    "constant",
}
_FORBIDDEN_CODE_MARKERS = (
    "python", "shell", "subprocess", "import ", "eval(", "exec(",
    "os.system", "__builtins__", "open(", "http://", "https://",
    "socket", "requests.", "pathlib", "/proc/", "C:\\",
)
_CONCRETE_ID_RE = re.compile(r"(?:^|[ _])(?:[a-z0-9]+[ _])?\d+$", re.IGNORECASE)


@dataclass
class ToolStaticReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failure_codes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    paths: dict[str, Any] = field(default_factory=dict)

    def as_validation(self) -> ValidationResult:
        return ValidationResult(
            "tool", self.passed, self.checks,
            self.failure_codes, self.messages,
        )


def _effect_name(value: Any) -> str:
    if isinstance(value, SemanticPredicate):
        return value.predicate.casefold()
    if isinstance(value, Mapping):
        return str(value.get("predicate", "")).casefold()
    return str(getattr(value, "predicate", "")).casefold()


def _argument_signature(value: Any) -> tuple[str, str]:
    if isinstance(value, BindingExpression):
        if value.kind is BindingExprKind.CONSTANT:
            return "constant", repr(value.constant)
        return "role", str(value.source_role)
    if isinstance(value, Mapping) and "kind" in value:
        try:
            expression = BindingExpression.from_dict(dict(value))
        except (KeyError, TypeError, ValueError):
            return "invalid", repr(to_primitive(value))
        return _argument_signature(expression)
    if isinstance(value, str) and value.startswith("$"):
        return "role", value[1:]
    return "constant", repr(value)


def _predicate_signature(value: Any) -> tuple[Any, ...]:
    predicate = _as_semantic(value)
    return (
        predicate.predicate.casefold(),
        tuple(sorted(
            (str(role), _argument_signature(argument))
            for role, argument in predicate.args.items()
        )),
        max(1, int(predicate.cardinality)),
        str(predicate.distinct_by),
        str(predicate.effect_domain.value),
    )


def _predicate_schema(harness: Any) -> list[Mapping[str, Any]]:
    method = getattr(harness, "semantic_predicate_schema", None)
    if callable(method):
        try:
            return [dict(to_primitive(item)) for item in method()]
        except Exception:
            return []
    return []


def _parameter_map(values: Iterable[ParameterSpec]) -> dict[str, ParameterSpec]:
    return {str(item.name): item for item in values}


def _selector_source(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _validate_program_node_shapes(nodes: Iterable[dict[str, Any]]) -> None:
    """Reject malformed nested node fields before semantic validation."""

    for node in nodes:
        node_id = str(node.get("node_id", "")).strip()
        if not node_id:
            raise ValueError("tool_ir_schema_invalid: nested node lacks node_id")
        opcode = str(node.get("op", ""))
        if opcode == "ACTION":
            if (
                "argument_mapping" in node
                and not isinstance(node.get("argument_mapping"), Mapping)
            ):
                raise ValueError(
                    f"tool_ir_schema_invalid: ACTION {node_id}.argument_mapping must be an object"
                )
            if (
                "expected_effects" in node
                and not isinstance(node.get("expected_effects"), (list, tuple))
            ):
                raise ValueError(
                    f"tool_ir_schema_invalid: ACTION {node_id}.expected_effects must be a list"
                )
        elif opcode in {"IF", "STOP_WHEN"}:
            if not isinstance(node.get("condition"), Mapping):
                raise ValueError(
                    f"tool_ir_schema_invalid: {opcode} {node_id}.condition must be an object"
                )
        elif opcode == "FOR_EACH":
            if not isinstance(node.get("collection_source"), Mapping):
                raise ValueError(
                    f"tool_ir_schema_invalid: FOR_EACH {node_id}.collection_source must be an object"
                )
            if (
                "max_iterations" in node
                and (
                    isinstance(node.get("max_iterations"), bool)
                    or not isinstance(node.get("max_iterations"), int)
                )
            ):
                raise ValueError(
                    f"tool_ir_schema_invalid: FOR_EACH {node_id}.max_iterations must be an integer"
                )
        elif opcode == "RETURN":
            if (
                "output_sources" in node
                and not isinstance(node.get("output_sources"), Mapping)
            ):
                raise ValueError(
                    f"tool_ir_schema_invalid: RETURN {node_id}.output_sources must be an object"
                )


def _validate_selector(
    source: dict[str, Any],
    node_id: str,
    *,
    fail: Any,
) -> None:
    kind = str(source.get("source", "")).casefold()
    if kind not in _COLLECTION_SOURCES:
        fail("tool_ir_selector_invalid", f"{node_id}: unknown collection source {kind}")
        return
    if kind == "local_deterministic":
        values = source.get("values")
        if not isinstance(values, (list, tuple)) or not values:
            fail("tool_ir_selector_invalid", f"{node_id}: local_deterministic.values must be non-empty")
        return
    project = _selector_source(source.get("project"))
    if project:
        project_kind = str(project.get("kind", "field")).casefold()
        if project_kind not in {"field", "argument"}:
            fail("tool_ir_selector_invalid", f"{node_id}: project.kind must be field or argument")
        if project_kind == "argument" and not str(project.get("role", "")):
            fail("tool_ir_selector_invalid", f"{node_id}: argument project requires role")
        if project_kind == "field" and not str(project.get("field", "")):
            fail("tool_ir_selector_invalid", f"{node_id}: field project requires field")
    elif not str(source.get("field", "")):
        fail("tool_ir_selector_invalid", f"{node_id}: selector requires field or project")
    where = _selector_source(source.get("where"))
    semantic = _selector_source(where.get("semantic_compatible_with"))
    if semantic:
        if str(semantic.get("source", "")).casefold() not in _CONDITION_SOURCES:
            fail("tool_ir_selector_invalid", f"{node_id}: semantic_compatible_with source invalid")
        if not str(semantic.get("field", "")):
            fail("tool_ir_selector_invalid", f"{node_id}: semantic_compatible_with field required")
        if not str(where.get("argument_role", "")):
            fail("tool_ir_selector_invalid", f"{node_id}: semantic_compatible_with requires where.argument_role")


def _condition_reference(condition: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(condition.get("source", "")).casefold(),
        str(condition.get("field", "")),
    )


def _check_scoped_reference(
    source: str,
    target: str,
    *,
    available_locals: set[str],
    atomic_inputs: set[str],
    fail: Any,
    node_id: str,
    context: str,
) -> None:
    if source == "tool_input":
        if target not in atomic_inputs:
            fail("tool_ir_local_scope_invalid", f"{node_id}.{context} references unknown tool input {target}")
    elif source == "local_variable":
        if target not in available_locals:
            fail("tool_ir_local_scope_invalid", f"{node_id}.{context} references {target} before it is definitely defined")


def _check_selector_scoped_references(
    selector: Mapping[str, Any],
    *,
    available_locals: set[str],
    atomic_inputs: set[str],
    fail: Any,
    node_id: str,
    context: str,
) -> None:
    """Close direct and nested selector references against lexical scope."""

    source = str(selector.get("source", "")).casefold()
    target = str(selector.get("field", ""))
    _check_scoped_reference(
        source,
        target,
        available_locals=available_locals,
        atomic_inputs=atomic_inputs,
        fail=fail,
        node_id=node_id,
        context=context,
    )
    where = selector.get("where")
    if not isinstance(where, Mapping):
        return
    semantic = where.get("semantic_compatible_with")
    if not isinstance(semantic, Mapping):
        return
    semantic_source = str(semantic.get("source", "")).casefold()
    semantic_target = str(semantic.get("field", ""))
    _check_scoped_reference(
        semantic_source,
        semantic_target,
        available_locals=available_locals,
        atomic_inputs=atomic_inputs,
        fail=fail,
        node_id=node_id,
        context=f"{context}.where.semantic_compatible_with",
    )


def _validate_expected_effect_references(
    effects: Any,
    *,
    node_id: str,
    available_locals: set[str],
    atomic_inputs: set[str],
    atomic_outputs: set[str],
    fail: Any,
) -> None:
    """Close every formal reference in an ACTION's expected Effects.

    ``$role`` and serialized/typed ``BindingExpression`` values are formal
    references, never free variables.  They may name a Tool input, a declared
    Tool output (including a not-yet-produced fresh output), or a local that
    is definitely in lexical scope at this ACTION.  Everything else fails
    static validation deterministically.
    """

    if effects is None:
        return
    if not isinstance(effects, (list, tuple)):
        fail(
            "tool_ir_effect_reference_invalid",
            f"ACTION {node_id}.expected_effects must be a list",
        )
        return
    available = set(atomic_inputs) | set(atomic_outputs) | set(available_locals)
    for effect_index, effect in enumerate(effects):
        if isinstance(effect, SemanticPredicate):
            arguments = effect.args
        elif isinstance(effect, Mapping):
            arguments = effect.get("args", {})
        else:
            fail(
                "tool_ir_effect_reference_invalid",
                f"ACTION {node_id}.expected_effects[{effect_index}] is not a predicate",
            )
            continue
        if not isinstance(arguments, Mapping):
            fail(
                "tool_ir_effect_reference_invalid",
                f"ACTION {node_id}.expected_effects[{effect_index}].args must be an object",
            )
            continue
        for argument_role, raw in arguments.items():
            reference = ""
            reference_kind = ""
            if isinstance(raw, str) and raw.startswith("$"):
                reference = raw[1:]
                reference_kind = "formal"
            elif isinstance(raw, BindingExpression):
                if raw.kind is BindingExprKind.CONSTANT:
                    continue
                if raw.kind is not BindingExprKind.SKILL_INPUT:
                    fail(
                        "tool_ir_effect_reference_invalid",
                        f"ACTION {node_id}.expected_effects[{effect_index}]"
                        f".args.{argument_role} uses unsupported "
                        f"BindingExpression kind {raw.kind.value}",
                    )
                    continue
                reference = str(raw.source_role)
                reference_kind = str(raw.kind.value)
            elif isinstance(raw, Mapping) and "kind" in raw:
                raw_kind = str(raw.get("kind", "")).casefold()
                if raw_kind == "constant":
                    continue
                if raw_kind == "local_variable":
                    reference = str(raw.get("source_role", ""))
                    reference_kind = raw_kind
                else:
                    try:
                        expression = BindingExpression.from_dict(dict(raw))
                    except (KeyError, TypeError, ValueError) as exc:
                        fail(
                            "tool_ir_effect_reference_invalid",
                            f"ACTION {node_id}.expected_effects[{effect_index}]"
                            f".args.{argument_role} has invalid BindingExpression: {exc}",
                        )
                        continue
                    if expression.kind is BindingExprKind.CONSTANT:
                        continue
                    if expression.kind is not BindingExprKind.SKILL_INPUT:
                        fail(
                            "tool_ir_effect_reference_invalid",
                            f"ACTION {node_id}.expected_effects[{effect_index}]"
                            f".args.{argument_role} uses unsupported "
                            f"BindingExpression kind {expression.kind.value}",
                        )
                        continue
                    reference = str(expression.source_role)
                    reference_kind = str(expression.kind.value)
            else:
                continue

            if reference_kind == "local_variable":
                valid = bool(reference and reference in available_locals)
            else:
                valid = bool(reference and reference in available)
            if not valid:
                fail(
                    "tool_ir_effect_reference_invalid",
                    f"ACTION {node_id}.expected_effects[{effect_index}]"
                    f".args.{argument_role} references out-of-scope role "
                    f"{reference or '<empty>'}",
                )


def _scope_pass(
    program: list[dict[str, Any]],
    *,
    atomic_inputs: set[str],
    atomic_outputs: set[str],
    fail: Any,
) -> None:
    """Fail-closed lexical scope without full SSA."""

    def visit(nodes: list[dict[str, Any]], available: set[str]) -> set[str]:
        current = set(available)
        for node in nodes:
            node_id = str(node.get("node_id", ""))
            opcode = str(node.get("op", ""))
            if opcode == "ACTION":
                for raw in dict(node.get("argument_mapping") or {}).values():
                    expression = _selector_source(raw)
                    if str(expression.get("kind", "")).casefold() == "local_variable":
                        _check_scoped_reference(
                            "local_variable",
                            str(expression.get("source_role", "")),
                            available_locals=current,
                            atomic_inputs=atomic_inputs,
                            fail=fail,
                            node_id=node_id,
                            context="argument_mapping",
                        )
                _validate_expected_effect_references(
                    node.get("expected_effects"),
                    node_id=node_id,
                    available_locals=current,
                    atomic_inputs=atomic_inputs,
                    atomic_outputs=atomic_outputs,
                    fail=fail,
                )
            elif opcode in {"IF", "STOP_WHEN"}:
                source, target = _condition_reference(_selector_source(node.get("condition")))
                _check_scoped_reference(
                    source, target,
                    available_locals=current,
                    atomic_inputs=atomic_inputs,
                    fail=fail, node_id=node_id, context="condition",
                )
                if opcode == "IF":
                    then_out = visit(list(node.get("then_branch") or []), current)
                    else_out = visit(list(node.get("else_branch") or []), current)
                    current = set(then_out) & set(else_out)
            elif opcode == "FOR_EACH":
                collection_source = _selector_source(
                    node.get("collection_source")
                )
                _validate_selector(
                    collection_source, node_id, fail=fail,
                )
                _check_selector_scoped_references(
                    collection_source,
                    available_locals=current,
                    atomic_inputs=atomic_inputs,
                    fail=fail,
                    node_id=node_id,
                    context="collection_source",
                )
                variable = str(node.get("iteration_variable", ""))
                if variable:
                    visit(list(node.get("body") or []), current | {variable})
                # Loop variables never leak outside their body.
            elif opcode == "RETURN":
                for raw in dict(node.get("output_sources") or {}).values():
                    spec = _selector_source(raw) if isinstance(raw, Mapping) else {
                        "source": "tool_input", "field": str(raw),
                    }
                    if "source" not in spec and spec.get("kind") == "skill_input":
                        spec = {
                            **spec,
                            "source": "tool_input",
                            "field": str(spec.get("source_role", "")),
                        }
                    elif "source" not in spec and spec.get("kind") == "local_variable":
                        spec = {
                            **spec,
                            "source": "local_variable",
                            "field": str(spec.get("source_role", "")),
                        }
                    _check_selector_scoped_references(
                        spec,
                        available_locals=current,
                        atomic_inputs=atomic_inputs,
                        fail=fail,
                        node_id=node_id,
                        context="output_sources.selector",
                    )
        return current

    visit(program, set())


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source", "kind", "op", "node_id"}:
                continue
            yield from _iter_string_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_string_values(item)


def _concrete_ids_from_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for node in walk_program_nodes(nodes):
        opcode = str(node.get("op", ""))
        if opcode == "ACTION":
            for raw in dict(node.get("argument_mapping") or {}).values():
                expression = _selector_source(raw)
                if str(expression.get("kind", "")).casefold() == "constant":
                    found.append(str(expression.get("constant", "")))
        elif opcode in {"IF", "STOP_WHEN"}:
            condition = _selector_source(node.get("condition"))
            if condition.get("value") is not None:
                found.append(str(condition.get("value")))
        elif opcode == "FOR_EACH":
            source = _selector_source(node.get("collection_source"))
            if source.get("source") == "local_deterministic":
                found.extend(_iter_string_values(source.get("values") or []))
            for item in _iter_string_values(source.get("where") or {}):
                found.append(item)
        elif opcode == "RETURN":
            for raw in dict(node.get("output_sources") or {}).values():
                spec = _selector_source(raw) if isinstance(raw, Mapping) else {}
                source = str(spec.get("source", "")).casefold()
                kind = str(spec.get("kind", "")).casefold()
                if source == "constant" or kind == "constant":
                    constant = (
                        spec.get("value")
                        if "value" in spec else spec.get("constant")
                    )
                    found.extend(_iter_string_values(constant))
        for raw in node.get("expected_effects") or ():
            found.extend(str(item) for item in _iter_string_values(raw) if item != "$")
        found.extend(str(item) for item in _iter_string_values(node.get("path_expectations") or []))
    return [
        value for value in found
        if value and _CONCRETE_ID_RE.search(value.casefold())
    ]


def _boundary_spec_signature(value: Any) -> tuple[str, str, bool, bool, str, str]:
    if isinstance(value, Mapping):
        value = ParameterSpec(
            name=str(value.get("name", "")),
            semantic_type=str(value.get("semantic_type", "entity")),
            required=bool(value.get("required", True)),
            runtime_resolvable=bool(value.get("runtime_resolvable", False)),
            required_resolution=str(value.get("required_resolution", "semantic")),
            description=str(value.get("description", "")),
        )
    return (
        str(value.name),
        str(value.semantic_type),
        bool(value.required),
        bool(value.runtime_resolvable),
        str(value.required_resolution),
    )


def _effect_formal_references(
    draft: RuntimeAutomationAtomicDraft,
) -> dict[str, list[tuple[str, str]]]:
    """Collect ``(predicate, argument_role)`` authorities per output role.

    A formal reference is either a ``$<role>`` string argument or a
    ``{"kind": "skill_input", "source_role": <role>}`` mapping argument of a
    declared Effect.  A declared input reference is a known Effect constraint
    and is therefore valid, but it is not evidence that derives a fresh
    output.  Declared output references are collected as fresh-output
    authorities.  A reference in neither boundary set is fail-closed invalid.
    """

    input_roles = {str(item.name) for item in draft.inputs}
    output_roles = {str(item.name) for item in draft.outputs}
    authorities: dict[str, list[tuple[str, str]]] = {}
    for predicate in draft.effects:
        predicate_name = str(predicate.predicate)
        for argument_role, raw in dict(predicate.args).items():
            source_role = ""
            if isinstance(raw, str) and raw.startswith("$"):
                source_role = raw[1:]
            elif isinstance(raw, BindingExpression):
                if raw.kind is BindingExprKind.SKILL_INPUT:
                    source_role = str(raw.source_role)
            elif isinstance(raw, Mapping) and "kind" in raw:
                expression = BindingExpression.from_dict(dict(raw))
                if expression.kind is BindingExprKind.SKILL_INPUT:
                    source_role = str(expression.source_role)
            if not source_role:
                continue
            if source_role in input_roles:
                # Input-bound Effect arguments constrain the witness but do
                # not constitute fresh-output derivation authority.  This
                # ordering also preserves INPUT_IDENTITY when a boundary role
                # is declared as both input and output.
                continue
            if source_role not in output_roles:
                raise ValueError(
                    "runtime_automation_r0_output_derivation_invalid: effect "
                    f"{predicate_name} references undeclared output role "
                    f"or input role {source_role}"
                )
            authorities.setdefault(source_role, []).append(
                (predicate_name, str(argument_role))
            )
    return authorities


def normalize_runtime_output_derivations(
    draft: RuntimeAutomationAtomicDraft,
) -> dict[str, dict[str, str]]:
    """One shared, fail-closed output-derivation authority for task-local
    Runtime Automation drafts.

    Both ``ToolStaticValidator.validate_automation_draft`` (R0) and
    ``RuntimeAutomationCoordinator._draft_atomic`` call exactly this function;
    no second derivation logic may exist.  For each required output:

    * INPUT_IDENTITY: a declared input role with the same name exists;
    * EFFECT_WITNESS: exactly one ``(predicate, argument_role)`` formal
      reference across the declared Effects;

    otherwise the draft is rejected with
    ``runtime_automation_r0_output_derivation_invalid``.
    """

    input_roles = {str(item.name) for item in draft.inputs}
    required_outputs = [
        str(item.name) for item in draft.outputs if bool(item.required)
    ]
    authorities = _effect_formal_references(draft)
    derivations: dict[str, dict[str, str]] = {}
    for output_role in required_outputs:
        if output_role in input_roles:
            derivations[output_role] = {
                "kind": "input_identity",
                "input_role": output_role,
            }
            continue
        candidates = authorities.get(output_role) or []
        unique = sorted({(predicate, role) for predicate, role in candidates})
        if not unique:
            raise ValueError(
                "runtime_automation_r0_output_derivation_invalid: required "
                f"output {output_role} has no legal derivation"
            )
        if len(unique) != 1:
            raise ValueError(
                "runtime_automation_r0_output_derivation_invalid: output "
                f"{output_role} has multiple Effect witness authorities "
                f"{unique!r}"
            )
        predicate, argument_role = unique[0]
        derivations[output_role] = {
            "kind": "effect_witness",
            "predicate": predicate,
            "argument_role": argument_role,
        }
    return derivations


def _boundary_exact(proposal: Any, atomic: AbstractAtomicSkill) -> bool:
    return (
        sorted(_boundary_spec_signature(item) for item in proposal.inputs)
        == sorted(_boundary_spec_signature(item) for item in atomic.inputs)
        and sorted(_boundary_spec_signature(item) for item in proposal.outputs)
        == sorted(_boundary_spec_signature(item) for item in atomic.outputs)
    )


def _historical_action(value: Any) -> dict[str, Any]:
    primitive = to_primitive(value)
    return dict(primitive) if isinstance(primitive, Mapping) else {}


def _action_structure(value: Any) -> tuple[str, tuple[str, ...]]:
    mapping = _historical_action(value)
    arguments = mapping.get("arguments", mapping.get("argument_mapping", {}))
    return (
        str(mapping.get("action_type", "")).casefold(),
        tuple(sorted(map(str, dict(arguments or {})))),
    )


def _action_instance(value: Any) -> tuple[tuple[str, str], ...]:
    mapping = _historical_action(value)
    arguments = mapping.get("arguments", mapping.get("argument_mapping", {}))
    return tuple(sorted(
        (str(role), repr(concrete))
        for role, concrete in dict(arguments or {}).items()
    ))


def _historical_loop_supported(
    loop: Mapping[str, Any],
    evidence_support: Iterable[Any],
) -> bool:
    body_actions = [
        node for node in walk_program_nodes(loop.get("body") or ())
        if str(node.get("op", "")) == "ACTION"
    ]
    wanted = [_action_structure(node) for node in body_actions]
    if not wanted or any(not action_type for action_type, _roles in wanted):
        return False
    evidence = [
        mapping for item in evidence_support
        for mapping in [_historical_action(item)]
        if mapping and bool(mapping.get("accepted", True))
    ]
    structures = [_action_structure(item) for item in evidence]
    matches: list[tuple[tuple[tuple[str, str], ...], ...]] = []
    index = 0
    while index <= len(structures) - len(wanted):
        if structures[index:index + len(wanted)] == wanted:
            matches.append(tuple(
                _action_instance(item)
                for item in evidence[index:index + len(wanted)]
            ))
            index += len(wanted)
        else:
            index += 1
    return len(set(matches)) >= 2


class ToolStaticValidator:
    """All checks are deterministic and code-authoritative."""

    def validate_proposal(
        self,
        proposal: ToolProposal,
        atomic: AbstractAtomicSkill,
        harness: Any,
        *,
        historical_evidence_support: Iterable[Any] | None = None,
    ) -> ToolStaticReport:
        checks: dict[str, bool] = {}
        codes: list[str] = []
        messages: list[str] = []

        def fail(code: str, message: str) -> None:
            checks.setdefault(code, False)
            codes.append(code)
            messages.append(message)

        if proposal.decision == "no_tool":
            checks["no_tool"] = True
            return ToolStaticReport(True, checks, [], ["NO_TOOL"], {})

        if not _boundary_exact(proposal, atomic):
            return ToolStaticReport(
                False,
                {"tool_builder_atomic_boundary": False},
                ["tool_builder_atomic_boundary_mismatch"],
                [
                    "ToolProposal must echo the immutable Atomic input/output boundary exactly",
                    f"atomic_inputs={[_boundary_spec_signature(item) for item in atomic.inputs]}",
                    f"proposal_inputs={[_boundary_spec_signature(item) for item in proposal.inputs]}",
                    f"atomic_outputs={[_boundary_spec_signature(item) for item in atomic.outputs]}",
                    f"proposal_outputs={[_boundary_spec_signature(item) for item in proposal.outputs]}",
                ],
                {},
            )

        try:
            program = normalize_tool_program(proposal.program)
            # Traverse with the frozen static nesting bound before any other
            # recursive consumer.  This turns malformed/cyclic/over-deep IR
            # into a deterministic report rather than a Python exception.
            walked_entries = walk_program_nodes(
                program, max_depth=4, with_context=True,
            )
            walked_nodes = [node for node, _depth, _path in walked_entries]
            _validate_program_node_shapes(walked_nodes)
            all_nodes = [
                (node, depth) for node, depth, _path in walked_entries
            ]
            output_roles_for_return = {
                str(item.name)
                for item in [*proposal.outputs, *atomic.outputs]
                if item.name
            }
            for node in walked_nodes:
                if node.get("op") == "RETURN":
                    node["output_sources"] = normalize_return_output_sources(
                        node, output_roles_for_return,
                    )
            paths = program_paths(program)
            checks["program_schema"] = True
            checks["program_paths_computable"] = bool(paths["path_ids"])
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            message = str(exc) or "tool_ir_schema_invalid"
            code = (
                "tool_ir_recursion_depth_exceeded"
                if isinstance(exc, RecursionError)
                else message.split(":", 1)[0]
            )
            return ToolStaticReport(
                False,
                {"program_schema": False, "program_paths_computable": False},
                [code],
                [message],
                {},
            )

        atomic_inputs = _parameter_map(atomic.inputs)
        atomic_outputs = _parameter_map(atomic.outputs)
        atomic_effects = {_effect_name(item) for item in atomic.effects}
        proposal_outputs = {str(item.name) for item in proposal.outputs}
        predicate_schema = _predicate_schema(harness)
        known_predicates = {
            str(item.get("predicate", "")).casefold(): item
            for item in predicate_schema
        }
        if not known_predicates:
            fail(
                "tool_ir_predicate_schema_unavailable",
                "Harness predicate schema is required for Tool static validation",
            )
        allowed_action_types = set()
        action_argument_roles: dict[str, set[str]] = {}
        action_schema = getattr(harness, "primitive_action_schema", None)
        if callable(action_schema):
            try:
                for item in action_schema():
                    action_type = str(item.get("action_type", ""))
                    if not action_type:
                        continue
                    allowed_action_types.add(action_type)
                    action_argument_roles[action_type] = {
                        str(role)
                        for role in item.get("argument_roles", [])
                    }
            except Exception:
                allowed_action_types = set()
                action_argument_roles = {}

        node_ids: set[str] = set()

        # 1. opcode whitelist / unique ids / recursion.
        checks["opcode_whitelist"] = all(
            str(node.get("op", "")) in _OPCODES
            for node, _ in all_nodes
        )
        if not checks["opcode_whitelist"]:
            fail("tool_ir_opcode_unsupported", "unknown Tool IR opcode")
        for node, depth in all_nodes:
            node_id = str(node.get("node_id", ""))
            if node_id in node_ids:
                fail("tool_ir_duplicate_node_id", f"duplicate node id {node_id}")
            node_ids.add(node_id)
            if depth > 4:
                fail("tool_ir_recursion_depth_exceeded", f"node {node_id} is nested too deeply")
        checks["tool_ir_node_ids_unique"] = "tool_ir_duplicate_node_id" not in codes
        checks["tool_ir_no_recursion"] = "tool_ir_recursion_depth_exceeded" not in codes

        # 2. max_actions and FOR_EACH bounds.
        checks["tool_ir_bounded_max_actions"] = bool(
            isinstance(proposal.max_actions, int) and proposal.max_actions > 0
        )
        if not checks["tool_ir_bounded_max_actions"]:
            fail("tool_ir_bounded_max_actions", "max_actions must be positive")
        has_return = any(
            str(node.get("op", "")) == "RETURN"
            for node, _depth in all_nodes
        )
        checks["tool_ir_return_present"] = has_return
        if not has_return:
            fail("tool_ir_return_missing", "Tool IR must contain at least one RETURN node")
        for node, _depth in all_nodes:
            if str(node.get("op", "")) == "FOR_EACH":
                max_iterations = int(node.get("max_iterations", 0) or 0)
                if max_iterations <= 0:
                    fail("tool_ir_for_each_unbounded", f"FOR_EACH {node.get('node_id')} lacks max_iterations")
                if proposal.max_actions and max_iterations > proposal.max_actions:
                    fail("tool_ir_for_each_unbounded", f"FOR_EACH {node.get('node_id')} exceeds max_actions")
                _validate_selector(
                    _selector_source(node.get("collection_source")),
                    str(node.get("node_id", "")), fail=fail,
                )
                variable = str(node.get("iteration_variable", ""))
                if not variable:
                    fail("tool_ir_for_each_variable_invalid", f"FOR_EACH {node.get('node_id')} lacks iteration_variable")
                if (
                    historical_evidence_support is not None
                    and not _historical_loop_supported(
                        node, historical_evidence_support,
                    )
                ):
                    fail(
                        "tool_ir_historical_loop_evidence_insufficient",
                        f"FOR_EACH {node.get('node_id')} requires at least "
                        "two structurally isomorphic distinct historical "
                        "repetitions",
                    )
        checks["tool_ir_for_each_bounded"] = not any(
            code in codes for code in {
                "tool_ir_for_each_unbounded",
                "tool_ir_selector_invalid",
                "tool_ir_for_each_variable_invalid",
            }
        )
        checks["tool_ir_historical_loop_evidence"] = (
            "tool_ir_historical_loop_evidence_insufficient" not in codes
        )

        # 3. Action nodes are harness primitives and argument mapping is closed
        #    by strict program-order/branch scope.
        for node, _depth in all_nodes:
            if str(node.get("op", "")) != "ACTION":
                continue
            action_type = str(node.get("action_type", ""))
            if not action_type:
                fail("tool_ir_action_schema_invalid", f"ACTION {node.get('node_id')} lacks action_type")
            elif allowed_action_types and action_type not in allowed_action_types:
                fail("tool_ir_action_schema_invalid", f"ACTION {node.get('node_id')} action_type not in Harness schema")
            elif action_type in action_argument_roles:
                actual_roles = {
                    str(role)
                    for role in dict(node.get("argument_mapping") or {})
                }
                expected_roles = action_argument_roles[action_type]
                if actual_roles != expected_roles:
                    missing_roles = sorted(expected_roles - actual_roles)
                    extra_roles = sorted(actual_roles - expected_roles)
                    fail(
                        "tool_ir_action_schema_invalid",
                        f"ACTION {node.get('node_id')} argument roles do not "
                        f"match Harness schema (missing={missing_roles}, "
                        f"extra={extra_roles})",
                    )
            for role, expression in dict(node.get("argument_mapping") or {}).items():
                expr = _selector_source(expression)
                kind = str(expr.get("kind", ""))
                if kind == "skill_input":
                    source_role = str(expr.get("source_role", ""))
                    if source_role not in atomic_inputs:
                        fail("tool_ir_input_closure_invalid", f"ACTION {node.get('node_id')} references unknown input role {source_role}")
                elif kind != "constant" and kind != "local_variable":
                    fail("tool_ir_argument_mapping_invalid", f"ACTION {node.get('node_id')}.{role} has unsupported mapping kind")
            if not list(node.get("expected_effects") or []):
                fail(
                    "tool_ir_step_effect_missing",
                    f"ACTION {node.get('node_id')} must declare expected_effects",
                )
        checks["tool_ir_input_closure"] = "tool_ir_input_closure_invalid" not in codes
        checks["tool_ir_action_schema"] = "tool_ir_action_schema_invalid" not in codes
        checks["tool_ir_argument_mapping"] = "tool_ir_argument_mapping_invalid" not in codes
        checks["tool_ir_step_effects_declared"] = (
            "tool_ir_step_effect_missing" not in codes
        )

        # 4. Conditions, RETURN closure and fail-closed lexical scope.
        for node, _depth in all_nodes:
            if str(node.get("op", "")) in {"IF", "STOP_WHEN"}:
                condition = _selector_source(node.get("condition"))
                if str(condition.get("source", "")).casefold() not in _CONDITION_SOURCES:
                    fail("tool_ir_condition_source_invalid", f"{node['op']} {node.get('node_id')} condition source invalid")
                if not str(condition.get("field", "")):
                    fail("tool_ir_condition_source_invalid", f"{node['op']} {node.get('node_id')} condition lacks field")
                operator = str(condition.get("op", "exists")).casefold()
                if operator not in CONDITION_OPERATORS:
                    fail("tool_ir_condition_operator_unsupported", f"{node['op']} {node.get('node_id')} condition operator invalid")
            if str(node.get("op", "")) == "RETURN":
                for role, raw in dict(node.get("output_sources") or {}).items():
                    spec = _selector_source(raw) if isinstance(raw, Mapping) else {"source": "tool_input", "field": role}
                    source = str(spec.get("source", "tool_input")).casefold()
                    if source not in _RETURN_SOURCES:
                        fail("tool_ir_return_closure_invalid", f"RETURN {node.get('node_id')}.{role} source invalid")
                    if role not in atomic_outputs and role not in proposal_outputs:
                        fail("tool_ir_return_closure_invalid", f"RETURN {node.get('node_id')}.{role} is not an Atomic/proposal output")
        return_roles: set[str] = set()
        for node in walk_program_nodes(program):
            if node.get("op") == "RETURN":
                return_roles.update(str(role) for role in dict(node.get("output_sources") or {}))
        required_output_roles = {
            str(name)
            for name, spec in atomic_outputs.items()
            if spec.required
        } or {
            str(item.name) for item in proposal.outputs if item.required
        }
        missing_return_outputs = sorted(required_output_roles - return_roles)
        if missing_return_outputs:
            fail(
                "tool_ir_return_closure_invalid",
                f"RETURN does not produce required outputs {missing_return_outputs}",
            )
        _scope_pass(
            program,
            atomic_inputs=set(atomic_inputs),
            atomic_outputs=set(atomic_outputs) | proposal_outputs,
            fail=fail,
        )
        checks["tool_ir_condition_source"] = "tool_ir_condition_source_invalid" not in codes
        checks["tool_ir_condition_operator"] = "tool_ir_condition_operator_unsupported" not in codes
        checks["tool_ir_return_closure"] = "tool_ir_return_closure_invalid" not in codes
        checks["tool_ir_local_scope"] = "tool_ir_local_scope_invalid" not in codes
        checks["tool_ir_effect_reference_closure"] = (
            "tool_ir_effect_reference_invalid" not in codes
        )

        # 5. Predicate vocabulary and effect domain.
        def validate_predicates(predicates: Iterable[Any], *, code: str) -> None:
            for predicate in predicates:
                name = _effect_name(predicate).casefold()
                if name not in known_predicates:
                    fail(code, f"unknown predicate {name}")
                    continue
                try:
                    normalized = _as_semantic(predicate)
                except (KeyError, TypeError, ValueError) as exc:
                    fail(code, f"malformed predicate {name}: {exc}")
                    continue
                domain = str(normalized.effect_domain.value)
                if domain not in {"world", "evidence"}:
                    fail("tool_ir_effect_domain_invalid", f"predicate {name} has invalid effect_domain")
                    continue
                offered_roles = set(map(str, normalized.args))
                expected_roles = set(map(
                    str, known_predicates[name].get("argument_roles", [])
                ))
                expected_domain = str(
                    known_predicates[name].get("effect_domain", "world")
                )
                if offered_roles != expected_roles or domain != expected_domain:
                    fail(
                        code,
                        f"predicate {name} does not match Harness argument/domain schema",
                    )

        for node, _depth in all_nodes:
            validate_predicates(
                node.get("expected_effects") or (),
                code="tool_ir_predicate_vocabulary",
            )
        validate_predicates(proposal.final_effects, code="tool_ir_predicate_vocabulary")
        checks["tool_ir_predicate_vocabulary"] = "tool_ir_predicate_vocabulary" not in codes
        checks["tool_ir_effect_domain"] = "tool_ir_effect_domain_invalid" not in codes

        # 6. Final effects compatible with the Atomic contract.
        final_names = {_effect_name(item) for item in proposal.final_effects}
        missing = sorted(atomic_effects - final_names)
        atomic_signatures = {_predicate_signature(item) for item in atomic.effects}
        try:
            final_signatures = {
                _predicate_signature(item) for item in proposal.final_effects
            }
        except (KeyError, TypeError, ValueError) as exc:
            final_signatures = set()
            fail(
                "tool_ir_final_effects_missing",
                f"malformed final effect: {exc}",
            )
        incompatible = sorted(
            repr(item) for item in atomic_signatures - final_signatures
        )
        checks["tool_ir_final_effects_compatible"] = bool(
            final_names and not missing and not incompatible
        )
        if missing or incompatible:
            fail(
                "tool_ir_final_effects_missing",
                "final effects do not structurally cover the Atomic contract "
                f"(missing_names={missing}, incompatible={incompatible})",
            )

        # 7. Portability / no arbitrary code / recursive episode-leakage scan.
        text_blob = str(to_primitive(proposal.program)).casefold()
        lowered = text_blob.casefold()
        arbitrary = [marker for marker in _FORBIDDEN_CODE_MARKERS if marker in lowered]
        checks["tool_ir_no_arbitrary_code"] = not arbitrary
        if arbitrary:
            fail("tool_ir_arbitrary_code", f"forbidden executable marker(s): {arbitrary}")

        concrete_ids = _concrete_ids_from_nodes(program)
        for raw in proposal.path_expectations:
            concrete_ids.extend(
                value for value in _iter_string_values(raw)
                if value and _CONCRETE_ID_RE.search(value.casefold())
            )
        for raw in proposal.evidence_outputs:
            concrete_ids.extend(
                value for value in _iter_string_values(raw)
                if value and _CONCRETE_ID_RE.search(value.casefold())
            )
        checks["tool_ir_no_episode_concrete_ids"] = not concrete_ids
        if concrete_ids:
            fail("tool_ir_episode_concrete_id", f"episode concrete constant(s): {list(dict.fromkeys(concrete_ids))[:5]}")

        # 8. Evidence outputs are deterministically verifiable.
        evidence_ok = True
        for item in proposal.evidence_outputs:
            source = str(item.get("source", "")).casefold()
            role = str(item.get("role", ""))
            if (
                source not in _RETURN_SOURCES
                or not role
                or role not in atomic_outputs
                or role not in proposal_outputs
            ):
                evidence_ok = False
                continue
            if source in {"semantic_evidence", "binding_evidence"}:
                before = len(codes)
                _validate_selector(
                    dict(item), f"evidence_output:{role}", fail=fail,
                )
                where = dict(item.get("where") or {})
                predicate = str(where.get("predicate", "")).casefold()
                if source == "semantic_evidence" and (
                    not predicate or predicate not in known_predicates
                ):
                    fail(
                        "tool_ir_evidence_output_invalid",
                        f"evidence_output:{role} predicate is not in Harness schema",
                    )
                evidence_ok = evidence_ok and len(codes) == before
        checks["tool_ir_evidence_outputs_verifiable"] = evidence_ok
        if not evidence_ok:
            fail("tool_ir_evidence_output_invalid", "evidence_outputs entries must be source-backed")

        checks["tool_ir_static_safe"] = not codes
        return ToolStaticReport(not codes, checks, codes, messages, paths)

    def validate_tool_asset(
        self,
        tool: Any,
        atomic: AbstractAtomicSkill,
        harness: Any,
    ) -> ToolStaticReport:
        """Revalidate a persisted ToolAsset with the same static authority."""

        artifact = dict(tool.artifact or {})
        program = [dict(item) for item in artifact.get("program", [])]
        input_schema = dict(tool.signature or {})
        output_schema = dict(
            dict(tool.interface or {}).get("output_schema") or {}
        )
        expected_input_roles = {str(item.name) for item in atomic.inputs}
        expected_required_inputs = {
            str(item.name) for item in atomic.inputs if bool(item.required)
        }
        expected_output_roles = {str(item.name) for item in atomic.outputs}
        expected_required_outputs = {
            str(item.name) for item in atomic.outputs if bool(item.required)
        }
        actual_input_roles = set(map(str, dict(
            input_schema.get("properties") or {}
        )))
        actual_required_inputs = set(map(str, list(
            input_schema.get("required") or []
        )))
        actual_output_roles = set(map(str, dict(
            output_schema.get("properties") or {}
        )))
        actual_required_outputs = set(map(str, list(
            output_schema.get("required") or []
        )))
        if (
            actual_input_roles != expected_input_roles
            or actual_required_inputs != expected_required_inputs
            or actual_output_roles != expected_output_roles
            or actual_required_outputs != expected_required_outputs
        ):
            return ToolStaticReport(
                False,
                {"tool_asset_atomic_boundary": False},
                ["tool_builder_atomic_boundary_mismatch"],
                [
                    "persisted Tool interface roles/required sets do not "
                    "match the immutable Atomic boundary"
                ],
                {},
            )
        proposal = ToolProposal(
            proposal_version="1",
            decision="create",
            summary=str(tool.summary),
            atomic_ref=str(atomic.ref),
            inputs=list(atomic.inputs),
            # Interface role equality was checked above; retain the complete
            # semantic types/resolution requirements from the Atomic rather
            # than reconstructing every output as a generic entity.
            outputs=list(atomic.outputs),
            program=program,
            max_actions=int(artifact.get("max_actions", 0) or 0),
            final_effects=[_as_semantic(item) for item in artifact.get("final_effects", [])],
            evidence_outputs=[dict(item) for item in artifact.get("evidence_outputs", [])],
            path_expectations=[dict(item) for item in artifact.get("path_expectations", [])],
            rationale=str(tool.metadata.get("tool_builder_rationale", "")),
        )
        return self.validate_proposal(proposal, atomic, harness)

    def validate_automation_draft(
        self,
        draft: RuntimeAutomationAtomicDraft,
        harness: Any,
        *,
        ctx: Any | None = None,
        occurrence: Any | None = None,
    ) -> ValidationResult:
        """R0: structure and task-local input binding authority."""

        checks: dict[str, bool] = {}
        codes: list[str] = []
        messages: list[str] = []

        def fail(code: str, message: str) -> None:
            codes.append(code)
            messages.append(message)

        checks["draft_schema"] = bool(draft.draft_id and draft.intent and draft.effects)
        checks["draft_roles"] = bool(draft.inputs and draft.outputs)
        if not checks["draft_roles"]:
            fail("runtime_automation_r0_role_closure", "draft must declare inputs and outputs")
        names = {str(item.name) for item in [*draft.inputs, *draft.outputs]}
        for predicate in [*draft.preconditions, *draft.effects]:
            for role, value in dict(predicate.args).items():
                if isinstance(value, str) and value.startswith("$") and value[1:] not in names:
                    fail("runtime_automation_r0_role_closure", f"predicate {predicate.predicate} references unknown role {value}")
        try:
            normalize_runtime_output_derivations(draft)
            checks["draft_output_derivations"] = True
        except ValueError as exc:
            checks["draft_output_derivations"] = False
            fail("runtime_automation_r0_output_derivation_invalid", str(exc))
        predicate_schema = _predicate_schema(harness)
        known_predicates = {str(item.get("predicate", "")).casefold() for item in predicate_schema}
        if known_predicates:
            unknown = {
                _effect_name(item)
                for item in [*draft.preconditions, *draft.effects]
                if _effect_name(item) not in known_predicates
            }
            checks["draft_predicate_vocabulary"] = not unknown
            if unknown:
                fail("runtime_automation_r0_predicate_vocabulary", f"unknown predicates {sorted(unknown)}")
        else:
            checks["draft_predicate_vocabulary"] = True

        domains = {str(item.effect_domain.value) for item in draft.effects}
        checks["draft_effect_domain"] = domains <= {"world", "evidence"}
        if not checks["draft_effect_domain"]:
            fail("runtime_automation_r0_effect_domain", f"invalid effect domains {sorted(domains)}")

        text_blob = str(to_primitive(draft)).casefold()
        arbitrary = [marker for marker in _FORBIDDEN_CODE_MARKERS if marker in text_blob]
        checks["draft_no_arbitrary_code"] = not arbitrary
        if arbitrary:
            fail("runtime_automation_r0_arbitrary_code", f"forbidden marker(s): {arbitrary}")

        specs = dict(getattr(draft, "input_binding_specs", None) or {})
        declared_inputs = {str(item.name) for item in draft.inputs}
        required_inputs = {
            str(item.name) for item in draft.inputs if bool(item.required)
        }
        spec_roles = {str(role) for role in specs}
        missing_specs = sorted(required_inputs - spec_roles)
        unexpected_specs = sorted(spec_roles - declared_inputs)
        checks["draft_input_binding_specs"] = not (
            missing_specs or unexpected_specs
        )
        if missing_specs:
            fail(
                "runtime_automation_input_binding_invalid",
                f"required input_binding_specs missing roles {missing_specs}",
            )
        if unexpected_specs:
            fail(
                "runtime_automation_input_binding_invalid",
                f"input_binding_specs contain undeclared roles "
                f"{unexpected_specs}",
            )
        for role, raw in specs.items():
            spec = _selector_source(raw)
            kind = str(spec.get("kind", "")).casefold()
            if role not in declared_inputs:
                checks["draft_input_binding_specs"] = False
                fail("runtime_automation_input_binding_invalid", f"input_binding_specs role {role} is not a draft input")
                continue
            if kind not in _INPUT_BINDING_KINDS:
                checks["draft_input_binding_specs"] = False
                fail("runtime_automation_input_binding_invalid", f"input_binding_specs.{role} has unsupported kind {kind}")
                continue
            if ctx is not None and occurrence is not None:
                binding_store = getattr(ctx, "binding_store", None)
                snapshot = binding_store.snapshot_for_node(occurrence) if binding_store is not None else {}
                resolved: Any = None
                source_binding: Any = None
                if kind == "current_occurrence_anchor":
                    source_role = str(spec.get("source_role", ""))
                    anchor = binding_store.semantic_anchor_for(occurrence, source_role) if binding_store is not None else None
                    source_binding = anchor
                    resolved = getattr(anchor, "value", None) if anchor is not None else None
                    if not source_role or resolved in (None, ""):
                        fail("runtime_automation_input_binding_invalid", f"{role}: current_occurrence_anchor.{source_role} unavailable")
                elif kind in {"current_confirmed_binding", "current_candidate_binding"}:
                    source_role = str(spec.get("source_role", ""))
                    binding = snapshot.get(source_role)
                    source_binding = binding
                    if binding is None:
                        fail("runtime_automation_input_binding_invalid", f"{role}: binding {source_role} unavailable")
                        continue
                    status = str(getattr(binding, "status", "")).casefold()
                    if kind == "current_confirmed_binding" and status != "grounded":
                        fail("runtime_automation_input_binding_invalid", f"{role}: binding {source_role} is not confirmed")
                    resolved = getattr(binding, "value", None)
                elif kind == "data_flow":
                    source_role = str(spec.get("source_role", ""))
                    output_binding = (
                        binding_store.validated_outputs(
                            occurrence.occurrence_id,
                        ).get(source_role)
                        if binding_store is not None else None
                    )
                    source_binding = output_binding
                    resolved = (
                        getattr(output_binding, "value", None)
                        if output_binding is not None else None
                    )
                    if resolved in (None, ""):
                        fail("runtime_automation_input_binding_invalid", f"{role}: data_flow.{source_role} unavailable")
                elif kind == "constant":
                    resolved = spec.get("value")
                    if resolved in (None, "") or (
                        isinstance(resolved, str) and _CONCRETE_ID_RE.search(resolved.casefold())
                    ):
                        fail("runtime_automation_input_binding_invalid", f"{role}: invalid episode concrete constant")
                if resolved not in (None, "") and kind != "constant":
                    input_spec = next(
                        (item for item in draft.inputs if str(item.name) == role),
                        None,
                    )
                    required_type = str(
                        getattr(input_spec, "semantic_type", "") or "entity"
                    )
                    offered_type = str(
                        getattr(source_binding, "semantic_type", "")
                    )
                    if not offered_type or not semantic_types_compatible(
                        required_type, offered_type,
                    ):
                        fail(
                            "runtime_automation_input_binding_invalid",
                            f"{role}: semantic type {offered_type or '<unknown>'} "
                            f"is incompatible with {required_type}",
                        )
                    actual_resolution = getattr(
                        source_binding, "resolution", "semantic",
                    )
                    required_resolution = str(
                        getattr(input_spec, "required_resolution", "semantic")
                    )
                    try:
                        resolution_ok = resolution_satisfies(
                            actual_resolution, required_resolution,
                        )
                    except (KeyError, TypeError, ValueError):
                        resolution_ok = False
                    if not resolution_ok:
                        fail(
                            "runtime_automation_input_binding_invalid",
                            f"{role}: resolution {actual_resolution} does not "
                            f"satisfy {required_resolution}",
                        )
                elif resolved not in (None, "") and kind == "constant":
                    input_spec = next(
                        (item for item in draft.inputs if str(item.name) == role),
                        None,
                    )
                    if str(getattr(
                        input_spec, "required_resolution", "semantic",
                    )) != "semantic":
                        fail(
                            "runtime_automation_input_binding_invalid",
                            f"{role}: semantic literal cannot satisfy "
                            f"{input_spec.required_resolution} resolution",
                        )

        checks["draft_no_episode_leakage"] = not bool(
            draft.source_occurrence_id
            and re.search(r"(?:_|\s)\d+$", draft.source_occurrence_id)
            and any(
                re.search(r"(?:_|\s)\d+$", str(value))
                for predicate in [*draft.preconditions, *draft.effects]
                for value in dict(predicate.args).values()
                if isinstance(value, str) and not value.startswith("$")
            )
        )
        passed = all(checks.values()) and not codes
        return ValidationResult("tool_r0", passed, checks, codes, messages)


def _as_semantic(value: Any) -> SemanticPredicate:
    if isinstance(value, SemanticPredicate):
        return value
    raw = dict(value)
    return SemanticPredicate(
        predicate=str(raw.get("predicate", "")),
        args=dict(raw.get("args") or {}),
        cardinality=int(raw.get("cardinality", 1)),
        distinct_by=str(raw.get("distinct_by", "")),
        effect_domain=str(raw.get("effect_domain", "world")),
    )


__all__ = [
    "ToolStaticReport",
    "ToolStaticValidator",
    "normalize_runtime_output_derivations",
]
