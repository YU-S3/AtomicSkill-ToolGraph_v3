"""Static and path validation for v3.2 ToolProposal / Tool IR.

This module is deliberately generic: it validates declarative IR against the
Atomic contract and Harness predicate/action interface.  It contains no task
family, object, or benchmark workflow knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from ..core.results import ValidationResult
from ..core.serialization import to_primitive
from .ir import normalize_tool_program, program_paths
from .proposal import RuntimeAutomationAtomicDraft, ToolProposal


_OPCODES = {"ACTION", "IF", "FOR_EACH", "STOP_WHEN", "RETURN"}
_CONDITION_SOURCES = {
    "tool_input", "local_variable", "action_catalog",
    "semantic_evidence", "binding_evidence",
}
_COLLECTION_SOURCES = {
    "tool_input", "local_variable", "action_catalog",
    "semantic_evidence", "local_deterministic",
}
_RETURN_SOURCES = {
    "tool_input", "local_variable", "semantic_evidence",
    "binding_evidence", "constant",
}
_FORBIDDEN_CODE_MARKERS = (
    "python", "shell", "subprocess", "import ", "eval(", "exec(",
    "os.system", "__builtins__", "open(", "http://", "https://",
    "socket", "requests.", "pathlib", "/proc/", "C:\\",
)


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


def _iter_nodes(nodes: Any, *, depth: int = 0, max_depth: int = 6) -> Iterable[tuple[dict[str, Any], int]]:
    if depth > max_depth:
        raise ValueError("tool_ir_recursion_depth_exceeded")
    for node in nodes or ():
        if not isinstance(node, Mapping):
            raise ValueError("tool_ir_schema_invalid")
        yield dict(node), depth
        op = str(node.get("op", ""))
        if op == "IF":
            yield from _iter_nodes(node.get("then_branch"), depth=depth + 1, max_depth=max_depth)
            yield from _iter_nodes(node.get("else_branch"), depth=depth + 1, max_depth=max_depth)
        elif op == "FOR_EACH":
            yield from _iter_nodes(node.get("body"), depth=depth + 1, max_depth=max_depth)


class ToolStaticValidator:
    """All checks are deterministic and code-authoritative."""

    def validate_proposal(
        self,
        proposal: ToolProposal,
        atomic: AbstractAtomicSkill,
        harness: Any,
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

        try:
            program = normalize_tool_program(proposal.program)
            paths = program_paths(program)
            checks["program_schema"] = True
            checks["program_paths_computable"] = bool(paths["path_ids"])
        except ValueError as exc:
            return ToolStaticReport(
                False,
                {"program_schema": False, "program_paths_computable": False},
                [str(exc).split(":")[0]],
                [str(exc)],
                {},
            )

        atomic_inputs = _parameter_map(atomic.inputs)
        atomic_outputs = _parameter_map(atomic.outputs)
        atomic_effects = {_effect_name(item) for item in atomic.effects}
        predicate_schema = _predicate_schema(harness)
        known_predicates = {
            str(item.get("predicate", "")).casefold(): item
            for item in predicate_schema
        }
        allowed_action_types = set()
        action_schema = getattr(harness, "primitive_action_schema", None)
        if callable(action_schema):
            try:
                allowed_action_types = {
                    str(item.get("action_type", ""))
                    for item in action_schema()
                }
            except Exception:
                allowed_action_types = set()

        node_ids: set[str] = set()
        defined_locals: set[str] = set()
        all_nodes = list(_iter_nodes(program))

        # 1. opcode whitelist / unique ids / recursion.
        checks["opcode_whitelist"] = all(node["op"] in _OPCODES for node, _ in all_nodes)
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
        has_return = any(node["op"] == "RETURN" for node, _depth in all_nodes)
        checks["tool_ir_return_present"] = has_return
        if not has_return:
            fail("tool_ir_return_missing", "Tool IR must contain at least one RETURN node")
        for node, _depth in all_nodes:
            if node["op"] == "FOR_EACH":
                max_iterations = int(node.get("max_iterations", 0) or 0)
                if max_iterations <= 0:
                    fail("tool_ir_for_each_unbounded", f"FOR_EACH {node.get('node_id')} lacks max_iterations")
                if proposal.max_actions and max_iterations > proposal.max_actions:
                    fail("tool_ir_for_each_unbounded", f"FOR_EACH {node.get('node_id')} exceeds max_actions")
                source = dict(node.get("collection_source") or {})
                if str(source.get("source", "")).casefold() not in _COLLECTION_SOURCES:
                    fail("tool_ir_collection_source_invalid", f"FOR_EACH {node.get('node_id')} collection source invalid")
                variable = str(node.get("iteration_variable", ""))
                if not variable:
                    fail("tool_ir_for_each_variable_invalid", f"FOR_EACH {node.get('node_id')} lacks iteration_variable")
                else:
                    defined_locals.add(variable)
        checks["tool_ir_for_each_bounded"] = not any(
            code in codes for code in {
                "tool_ir_for_each_unbounded",
                "tool_ir_collection_source_invalid",
                "tool_ir_for_each_variable_invalid",
            }
        )

        # 3. Action nodes are harness primitives and argument mapping is closed.
        for node, _depth in all_nodes:
            if node["op"] != "ACTION":
                continue
            action_type = str(node.get("action_type", ""))
            if not action_type:
                fail("tool_ir_action_schema_invalid", f"ACTION {node.get('node_id')} lacks action_type")
            elif allowed_action_types and action_type not in allowed_action_types:
                fail("tool_ir_action_schema_invalid", f"ACTION {node.get('node_id')} action_type not in Harness schema")
            for role, expression in dict(node.get("argument_mapping") or {}).items():
                expr = dict(expression) if isinstance(expression, Mapping) else {}
                kind = str(expr.get("kind", ""))
                if kind == "skill_input":
                    source_role = str(expr.get("source_role", ""))
                    if source_role not in atomic_inputs:
                        fail("tool_ir_input_closure_invalid", f"ACTION {node.get('node_id')} references unknown input role {source_role}")
                elif kind == "local_variable":
                    source_role = str(expr.get("source_role", ""))
                    if source_role not in defined_locals:
                        fail("tool_ir_local_closure_invalid", f"ACTION {node.get('node_id')} references unknown local {source_role}")
                elif kind != "constant":
                    fail("tool_ir_argument_mapping_invalid", f"ACTION {node.get('node_id')}.{role} has unsupported mapping kind")
        checks["tool_ir_input_closure"] = "tool_ir_input_closure_invalid" not in codes
        checks["tool_ir_local_closure"] = "tool_ir_local_closure_invalid" not in codes
        checks["tool_ir_action_schema"] = "tool_ir_action_schema_invalid" not in codes

        # 4. IF condition sources and STOP_WHEN / RETURN closure.
        for node, _depth in all_nodes:
            if node["op"] in {"IF", "STOP_WHEN"}:
                condition = dict(node.get("condition") or {})
                if str(condition.get("source", "")).casefold() not in _CONDITION_SOURCES:
                    fail("tool_ir_condition_source_invalid", f"{node['op']} {node.get('node_id')} condition source invalid")
                if not str(condition.get("field", "")):
                    fail("tool_ir_condition_source_invalid", f"{node['op']} {node.get('node_id')} condition lacks field")
            if node["op"] == "RETURN":
                for role, raw in dict(node.get("output_sources") or {}).items():
                    spec = dict(raw) if isinstance(raw, Mapping) else {"source": "tool_input", "field": role}
                    source = str(spec.get("source", "tool_input")).casefold()
                    if source not in _RETURN_SOURCES:
                        fail("tool_ir_return_closure_invalid", f"RETURN {node.get('node_id')}.{role} source invalid")
                    if source in {"tool_input", "local_variable"}:
                        target = str(spec.get("field", role))
                        if source == "tool_input" and target not in atomic_inputs:
                            fail("tool_ir_return_closure_invalid", f"RETURN {node.get('node_id')}.{role} unknown tool input")
                        if source == "local_variable" and target not in defined_locals:
                            fail("tool_ir_return_closure_invalid", f"RETURN {node.get('node_id')}.{role} unknown local")
                    if role not in atomic_outputs and role not in {item.name for item in proposal.outputs}:
                        fail("tool_ir_return_closure_invalid", f"RETURN {node.get('node_id')}.{role} is not an Atomic/proposal output")
        checks["tool_ir_condition_source"] = "tool_ir_condition_source_invalid" not in codes
        checks["tool_ir_return_closure"] = "tool_ir_return_closure_invalid" not in codes

        # 5. Predicate vocabulary and effect domain.
        def validate_predicates(predicates: Iterable[Any], *, code: str) -> None:
            for predicate in predicates:
                name = _effect_name(predicate).casefold()
                if known_predicates and name not in known_predicates:
                    fail(code, f"unknown predicate {name}")
                    continue
                domain = ""
                if isinstance(predicate, SemanticPredicate):
                    domain = str(predicate.effect_domain.value)
                elif isinstance(predicate, Mapping):
                    domain = str(predicate.get("effect_domain", "world"))
                if domain not in {"world", "evidence"}:
                    fail("tool_ir_effect_domain_invalid", f"predicate {name} has invalid effect_domain")

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
        checks["tool_ir_final_effects_compatible"] = not missing and bool(final_names)
        if missing:
            fail("tool_ir_final_effects_missing", f"final effects missing {missing}")

        # 7. Portability / no arbitrary code / no episode concrete IDs.
        text_blob = str(to_primitive(proposal.program)).casefold()
        lowered = text_blob.casefold()
        arbitrary = [marker for marker in _FORBIDDEN_CODE_MARKERS if marker in lowered]
        checks["tool_ir_no_arbitrary_code"] = not arbitrary
        if arbitrary:
            fail("tool_ir_arbitrary_code", f"forbidden executable marker(s): {arbitrary}")

        concrete_ids: list[str] = []
        for node, _depth in all_nodes:
            if node["op"] != "ACTION":
                continue
            for raw in dict(node.get("argument_mapping") or {}).values():
                expr = dict(raw) if isinstance(raw, Mapping) else {}
                if str(expr.get("kind", "")) == "constant":
                    value = str(expr.get("constant", ""))
                    if re.search(r"(?:^|[ _])(?:[a-z0-9]+[ _])?\d+$", value.casefold()):
                        concrete_ids.append(value)
        checks["tool_ir_no_episode_concrete_ids"] = not concrete_ids
        if concrete_ids:
            fail("tool_ir_episode_concrete_id", f"episode concrete constant(s): {concrete_ids[:5]}")

        # 8. Evidence outputs are deterministically verifiable.
        evidence_ok = True
        for item in proposal.evidence_outputs:
            source = str(item.get("source", "")).casefold()
            if source not in _RETURN_SOURCES or not item.get("role"):
                evidence_ok = False
        checks["tool_ir_evidence_outputs_verifiable"] = evidence_ok
        if not evidence_ok:
            fail("tool_ir_evidence_output_invalid", "evidence_outputs entries must be source-backed")

        checks["tool_ir_static_safe"] = not codes
        return ToolStaticReport(not codes, checks, codes, messages, paths)

    def validate_automation_draft(
        self,
        draft: RuntimeAutomationAtomicDraft,
        harness: Any,
    ) -> ValidationResult:
        """R0: structure only, before any Tool execution."""

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

        domains = {
            str(item.effect_domain.value)
            for item in draft.effects
        }
        checks["draft_effect_domain"] = domains <= {"world", "evidence"}
        if not checks["draft_effect_domain"]:
            fail("runtime_automation_r0_effect_domain", f"invalid effect domains {sorted(domains)}")

        text_blob = str(to_primitive(draft)).casefold()
        arbitrary = [marker for marker in _FORBIDDEN_CODE_MARKERS if marker in text_blob]
        checks["draft_no_arbitrary_code"] = not arbitrary
        if arbitrary:
            fail("runtime_automation_r0_arbitrary_code", f"forbidden marker(s): {arbitrary}")

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
        return ValidationResult(
            "tool_r0", passed, checks, codes, messages,
        )


__all__ = ["ToolStaticReport", "ToolStaticValidator"]
