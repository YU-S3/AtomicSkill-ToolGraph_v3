"""Execute serial Primitive/Tool IR against the Harness under one authority.

ToolRunner is the only interpreter for ``tool_ir_v1``.  Admission replay and
task-local runtime trials both execute the same bounded program semantics here.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import ToolAsset
from ..core.results import PrimitiveToolStep, ToolExecutionResult
from ..core.serialization import to_primitive
from ..traces.schema import EnvironmentActionRecord, ToolExecutionRecord
from ..validation.tool_validator import ToolValidator
from ..tooling.ir import (
    ToolExecutionState,
    evaluate_condition,
    normalize_return_output_sources,
    resolve_collection,
    resolve_return_sources,
    walk_program_nodes,
)


class ToolRunner:
    def __init__(self, validator: ToolValidator) -> None:
        self.validator = validator

    def _primitive(self, value: dict[str, Any]) -> PrimitiveToolStep:
        return PrimitiveToolStep(
            action_type=str(value["action_type"]),
            argument_mapping={
                role: BindingExpression.from_dict(expression) if isinstance(expression, dict) and "kind" in expression else expression
                for role, expression in value.get("argument_mapping", {}).items()
            },
        )

    def run(
        self, tool: ToolAsset, bindings: dict[str, Any], ctx: Any,
        *, occurrence_id: str, parent_span_id: str | None = None,
    ) -> ToolExecutionResult:
        before_revision = ctx.world_revision
        local = self.validator.validate_asset(tool)
        if not local.passed:
            return ToolExecutionResult(
                str(tool.ref), False, False, False, False, 0, None, [], {},
                before_revision, before_revision, "tool", "tool_preflight_rejected",
                "; ".join(local.messages),
            )
        span = ctx.trace_builder.start_span("tool", occurrence_id, parent_span_id=parent_span_id)
        if tool.artifact_kind == "tool_ir_v1":
            return self._run_ir_v1(tool, bindings, ctx, occurrence_id=occurrence_id, span_id=span.span_id)
        executed = 0
        started = False
        terminal_interrupted = False
        partial: list[dict[str, Any]] = []
        failure_index: int | None = None
        failure_code = failure_message = ""
        steps = [self._primitive(item) for item in tool.artifact.get("steps", [])]
        for index, primitive in enumerate(steps):
            try:
                spec = ctx.harness.compile_primitive(primitive, bindings)
            except (KeyError, TypeError, ValueError) as exc:
                failure_index, failure_code, failure_message = index, "tool_primitive_rejected", str(exc)
                break
            ctx.budget.consume_action()
            started = True
            result = ctx.harness.execute_action(spec.action_id, spec.revision)
            executed += 1
            record = EnvironmentActionRecord(
                spec.action_id, spec.revision, spec.action_type, dict(spec.arguments), result.accepted,
                result.observation, result.done, result.won, result.new_revision, span.span_id,
            )
            ctx.trace_builder.trace.environment_actions.append(record)
            ctx.update_after_action(result, {**to_primitive(record), "occurrence_id": occurrence_id, "origin": "tool"})
            if result.accepted:
                partial.append({"action_type": spec.action_type, "arguments": dict(spec.arguments), "revision": result.new_revision})
            else:
                failure_index, failure_code, failure_message = index, "tool_primitive_rejected", "Harness rejected primitive"
                break
            if result.won:
                terminal_interrupted = True
                break
            if result.done:
                failure_index, failure_code, failure_message = index, "tool_execution_error", "environment ended without success"
                break
        completed = failure_index is None and not terminal_interrupted and executed == len(steps)
        outputs: dict[str, Any] = {}
        output_mapping = tool.artifact.get("output_mapping", {})
        for role, expression in output_mapping.items():
            if isinstance(expression, dict) and "kind" in expression:
                expression = BindingExpression.from_dict(expression)
                if expression.kind is BindingExprKind.CONSTANT:
                    outputs[role] = expression.constant
                else:
                    outputs[role] = bindings.get(expression.source_role)
            elif isinstance(expression, str) and expression in bindings:
                outputs[role] = bindings[expression]
            else:
                outputs[role] = expression
        if completed and not outputs:
            for role in (tool.interface.get("output_schema", {}).get("properties", {}) or {}):
                if role in bindings:
                    outputs[role] = bindings[role]
        validation = self.validator.validate_output(tool, outputs)
        if completed and not validation.passed:
            completed, failure_code, failure_message = False, "tool_output_schema_error", "; ".join(validation.messages)
        ctx.trace_builder.finish_span(span.span_id)
        tool_result = ToolExecutionResult(
            str(tool.ref), True, started, completed, ctx.world_revision != before_revision,
            executed, failure_index, partial, outputs, before_revision, ctx.world_revision,
            "tool" if failure_code else "", failure_code, failure_message,
            terminal_interrupted=terminal_interrupted,
            intrinsic_failure=bool(failure_index is not None and not terminal_interrupted),
            executed_node_count=executed,
            remaining_node_count=max(0, len(steps) - executed),
            program_node_id=str(failure_index) if failure_index is not None else "",
            path_id=",".join(str(index) for index in range(executed)),
            tool_path_evidence={
                "program_path_id": ",".join(str(index) for index in range(executed)),
                "executed_node_ids": [str(index) for index in range(executed)],
                "validated_paths": [],
                "unvalidated_paths": [],
                "loop_iteration_counts": {},
                "stop_condition_witnesses": [],
                "step_effect_results": [],
                "final_effect_result": {
                    "passed": completed or terminal_interrupted,
                    "failure_code": failure_code,
                },
                "outputs": to_primitive(outputs),
                "evidence_refs": [],
                "terminal_interrupted": terminal_interrupted,
            },
        )
        ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
            f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref), to_primitive(tool_result), span.span_id,
        ))
        return tool_result

    def _ir_state(self, tool: ToolAsset, bindings: dict[str, Any], ctx: Any) -> ToolExecutionState:
        snapshot_method = getattr(ctx, "tool_evidence_snapshot", None)
        if callable(snapshot_method):
            snapshot = snapshot_method()
            facts = snapshot.get("semantic_facts", [])
            binding_evidence = snapshot.get("binding_evidence", [])
            catalog = snapshot.get("action_catalog", [])
        else:
            channel_snapshot = getattr(ctx.harness.validator_channel(), "snapshot", lambda: {})()
            if isinstance(channel_snapshot, dict):
                facts = channel_snapshot.get("facts", [])
            elif isinstance(channel_snapshot, list):
                facts = channel_snapshot
            else:
                facts = []
            binding_evidence = []
            catalog = [
                {
                    "action_id": str(item.action_id),
                    "revision": int(item.revision),
                    "action_type": str(item.action_type),
                    "arguments": dict(item.arguments),
                }
                for item in ctx.action_catalog
            ]
        return ToolExecutionState(
            bindings=dict(bindings),
            catalog=[dict(item) for item in catalog],
            semantic_facts=self._semantic_facts_with_domains(
                facts, ctx.harness,
            ),
            binding_evidence=[dict(item) for item in binding_evidence],
            max_actions=int(tool.artifact.get("max_actions", 0) or 0),
            max_control_steps=self._control_step_limit(tool, ctx),
        )

    @staticmethod
    def _semantic_facts_with_domains(
        facts: Any,
        harness: Any,
    ) -> list[dict[str, Any]]:
        """Attach the Harness-declared domain to validator-channel facts."""

        schema_method = getattr(harness, "semantic_predicate_schema", None)
        domains: dict[str, str] = {}
        if callable(schema_method):
            try:
                for raw in schema_method():
                    item = to_primitive(raw)
                    if not isinstance(item, dict):
                        continue
                    predicate = str(item.get("predicate", "")).casefold()
                    domain = str(item.get("effect_domain", ""))
                    if predicate and domain in {"world", "evidence"}:
                        domains[predicate] = domain
            except Exception:
                domains = {}
        result: list[dict[str, Any]] = []
        for raw in facts or ():
            if not isinstance(raw, dict):
                continue
            fact = dict(raw)
            predicate = str(fact.get("predicate", "")).casefold()
            declared_domain = domains.get(predicate)
            if declared_domain:
                fact["effect_domain"] = declared_domain
            elif str(fact.get("effect_domain", "")) not in {
                "world", "evidence",
            }:
                fact["effect_domain"] = ""
            result.append(fact)
        return result

    @staticmethod
    def _control_step_limit(tool: ToolAsset, ctx: Any) -> int:
        """IR interpreter safety bound, not a new experiment budget.

        Derived from the frozen global action budget and the flat program size:
        ``min(max_actions, global_action_budget) * flat_node_count`` bounds the
        total interpreter node visits of nested control flow.
        """

        program = tool.artifact.get("program", [])
        flat_node_count = max(
            1, ToolRunner._program_node_count(program),
        )
        global_cap = max(
            1, int(getattr(ctx, "global_action_budget", 100) or 100)
        )
        effective_action_cap = min(
            max(1, int(tool.artifact.get("max_actions", 0) or 0)),
            global_cap,
        )
        return max(1, effective_action_cap * flat_node_count)

    @staticmethod
    def _program_node_count(program: Any) -> int:
        """Shared-walker metric that remains total for malformed artifacts."""

        try:
            return len(walk_program_nodes(program))
        except (KeyError, TypeError, ValueError, RecursionError):
            return 0

    @staticmethod
    def _program_path_id(state: ToolExecutionState) -> str:
        return "/".join(["program", *state.path_tokens])

    @staticmethod
    def _selector_requires_match(source: Any) -> bool:
        if not isinstance(source, dict):
            return False
        return (
            str(source.get("source", "")).casefold()
            in {"action_catalog", "semantic_evidence", "binding_evidence"}
            and ("where" in source or "project" in source)
        )

    @staticmethod
    def _tool_ir_failure_detail_layer(failure_code: str) -> str:
        if failure_code == "tool_step_effect_violation":
            return "tool_effect"
        if failure_code in {"tool_primitive_rejected", "tool_execution_error"}:
            return "tool_step"
        return "tool_ir" if failure_code else ""

    def _tool_path_evidence(
        self,
        state: ToolExecutionState,
        *,
        outputs: dict[str, Any],
        terminal_interrupted: bool,
        final_effect_result: dict[str, Any],
    ) -> dict[str, Any]:
        failure_code = str(state.failure_code or "")
        return {
            "program_path_id": self._program_path_id(state),
            "executed_node_ids": list(state.executed_nodes),
            "program_node_id": str(state.program_node_id),
            # The global credit layer remains ``tool``.  This nested field is
            # the frozen Tool-IR diagnostic boundary.
            "failure_layer": self._tool_ir_failure_detail_layer(
                failure_code
            ),
            "failure_code": failure_code,
            "validated_paths": sorted(set(state.validated_paths)),
            "unvalidated_paths": sorted(set(state.unvalidated_paths)),
            "loop_iteration_counts": dict(state.loop_iteration_counts),
            "stop_condition_witnesses": list(
                state.stop_condition_witnesses
            ),
            "control_step_count": int(state.executed_control_step_count),
            "control_step_limit": int(state.max_control_steps),
            "step_effect_results": [
                dict(item) for item in state.step_effect_results
            ],
            "final_effect_result": dict(final_effect_result),
            "outputs": to_primitive(outputs),
            "evidence_refs": list(dict.fromkeys(state.evidence_refs)),
            "terminal_interrupted": bool(terminal_interrupted),
        }

    def _resolve_action_arguments(
        self, node: dict[str, Any], state: ToolExecutionState,
    ) -> PrimitiveToolStep:
        mapping: dict[str, Any] = {}
        for role, raw in dict(node.get("argument_mapping") or {}).items():
            expression = dict(raw) if isinstance(raw, dict) else raw
            kind = str(expression.get("kind", ""))
            if kind == "constant":
                mapping[role] = BindingExpression(BindingExprKind.CONSTANT, constant=expression.get("constant"))
            elif kind == "local_variable":
                mapping[role] = BindingExpression(
                    BindingExprKind.CONSTANT,
                    constant=state.local.get(str(expression.get("source_role", ""))),
                )
            else:
                mapping[role] = BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role=str(expression.get("source_role", role)),
                )
        return PrimitiveToolStep(str(node.get("action_type", "")), mapping)

    def _record_ir_action(
        self, node: dict[str, Any], primitive: PrimitiveToolStep,
        ctx: Any, state: ToolExecutionState, *, occurrence_id: str, span_id: str,
    ) -> dict[str, Any]:
        spec = ctx.harness.compile_primitive(primitive, state.bindings)
        ctx.budget.consume_action()
        result = ctx.harness.execute_action(spec.action_id, spec.revision)
        record = EnvironmentActionRecord(
            spec.action_id, spec.revision, spec.action_type, dict(spec.arguments),
            result.accepted, result.observation, result.done, result.won,
            result.new_revision, span_id,
        )
        ctx.trace_builder.trace.environment_actions.append(record)
        ctx.update_after_action(
            result,
            {
                **to_primitive(record),
                "occurrence_id": occurrence_id,
                "origin": "tool_ir",
                "program_node_id": str(node.get("node_id", "")),
            },
        )
        state.executed_action_count += 1
        snapshot_method = getattr(ctx, "tool_evidence_snapshot", None)
        if callable(snapshot_method):
            snapshot = snapshot_method()
            state.semantic_facts = self._semantic_facts_with_domains(
                snapshot.get("semantic_facts", []), ctx.harness,
            )
            state.binding_evidence = [dict(item) for item in snapshot.get("binding_evidence", [])]
            state.catalog = [dict(item) for item in snapshot.get("action_catalog", [])]
        else:
            state.catalog = [
                {
                    "action_id": str(item.action_id),
                    "revision": int(item.revision),
                    "action_type": str(item.action_type),
                    "arguments": dict(item.arguments),
                }
                for item in result.catalog
            ]
            channel_snapshot = getattr(ctx.harness.validator_channel(), "snapshot", lambda: {})()
            if isinstance(channel_snapshot, dict):
                state.semantic_facts = self._semantic_facts_with_domains(
                    channel_snapshot.get("facts", []), ctx.harness,
                )
            elif isinstance(channel_snapshot, list):
                state.semantic_facts = self._semantic_facts_with_domains(
                    channel_snapshot, ctx.harness,
                )
        return {
            "accepted": bool(result.accepted),
            "won": bool(result.won),
            "done": bool(result.done),
            "revision": int(result.new_revision),
            "action_type": spec.action_type,
            "arguments": dict(spec.arguments),
        }

    def _effect_bindings(self, state: ToolExecutionState) -> dict[str, Any]:
        merged = dict(state.bindings)
        for key, value in state.local.items():
            merged.setdefault(key, value)
        # Fresh RETURN outputs are the authoritative values for output-role
        # effect references in the post-program final-effect validation.
        for key, value in state.outputs.items():
            merged[key] = value
        return merged

    @staticmethod
    def _step_effect_resolution(
        effect: dict[str, Any],
        state: ToolExecutionState,
        *,
        input_roles: set[str],
        output_roles: set[str],
    ) -> tuple[dict[str, Any], set[str], list[str]]:
        """Resolve one step Effect and identify legal fresh-output slots.

        Only an argument whose formal reference names a declared output that
        has not yet been produced may be a wildcard.  A missing declared
        input, an unknown role, an out-of-scope local, or an unsupported
        BindingExpression is an error and can never be widened to a wildcard.
        """

        resolved = {
            "predicate": str(effect.get("predicate", "")),
            "args": {},
            "cardinality": int(effect.get("cardinality", 1) or 1),
            "distinct_by": str(effect.get("distinct_by", "")),
            "effect_domain": str(effect.get("effect_domain", "world")),
        }
        wildcard_arguments: set[str] = set()
        errors: list[str] = []
        for raw_argument_role, raw in dict(effect.get("args") or {}).items():
            argument_role = str(raw_argument_role)
            source_role = ""
            source_kind = ""
            if isinstance(raw, str) and raw.startswith("$"):
                source_role = raw[1:]
                source_kind = "formal"
            elif isinstance(raw, BindingExpression):
                if raw.kind is BindingExprKind.CONSTANT:
                    resolved["args"][argument_role] = raw.constant
                    continue
                if raw.kind is not BindingExprKind.SKILL_INPUT:
                    errors.append(
                        f"{argument_role}: unsupported BindingExpression "
                        f"kind {raw.kind.value}"
                    )
                    continue
                source_role = str(raw.source_role)
                source_kind = str(raw.kind.value)
            elif isinstance(raw, dict) and "kind" in raw:
                source_kind = str(raw.get("kind", "")).casefold()
                if source_kind == "constant":
                    resolved["args"][argument_role] = raw.get("constant")
                    continue
                if source_kind in {"skill_input", "local_variable"}:
                    source_role = str(raw.get("source_role", ""))
                else:
                    errors.append(
                        f"{argument_role}: unsupported BindingExpression "
                        f"kind {source_kind or '<empty>'}"
                    )
                    continue
            else:
                resolved["args"][argument_role] = raw
                continue

            if source_kind == "local_variable":
                if (
                    source_role not in state.local
                    or state.local.get(source_role) in (None, "")
                ):
                    errors.append(
                        f"{argument_role}: local {source_role or '<empty>'} "
                        "is unavailable"
                    )
                else:
                    resolved["args"][argument_role] = state.local[source_role]
                continue

            # A same-named input/output remains an input authority until a
            # fresh output is actually produced; absence must fail closed.
            if source_role in input_roles:
                if (
                    source_role not in state.bindings
                    or state.bindings.get(source_role) in (None, "")
                ):
                    errors.append(
                        f"{argument_role}: input {source_role} is unavailable"
                    )
                else:
                    resolved["args"][argument_role] = state.bindings[source_role]
            elif source_role in state.local:
                value = state.local.get(source_role)
                if value in (None, ""):
                    errors.append(
                        f"{argument_role}: local {source_role} is unavailable"
                    )
                else:
                    resolved["args"][argument_role] = value
            elif source_role in output_roles:
                if source_role in state.outputs:
                    value = state.outputs.get(source_role)
                    if value in (None, ""):
                        errors.append(
                            f"{argument_role}: output {source_role} is invalid"
                        )
                    else:
                        resolved["args"][argument_role] = value
                else:
                    resolved["args"][argument_role] = None
                    wildcard_arguments.add(argument_role)
            else:
                errors.append(
                    f"{argument_role}: unknown formal role "
                    f"{source_role or '<empty>'}"
                )
        return resolved, wildcard_arguments, errors

    def _resolved_effect(
        self, effect: dict[str, Any], state: ToolExecutionState,
    ) -> dict[str, Any]:
        """Resolve serialized BindingExpression effect args for Harness validation."""

        effect_bindings = self._effect_bindings(state)
        args: dict[str, Any] = {}
        for role, raw in dict(effect.get("args") or {}).items():
            if isinstance(raw, dict) and raw.get("kind") == "skill_input":
                args[role] = effect_bindings.get(str(raw.get("source_role", "")))
            elif isinstance(raw, dict) and raw.get("kind") == "constant":
                args[role] = raw.get("constant")
            elif isinstance(raw, str) and raw.startswith("$"):
                args[role] = effect_bindings.get(raw[1:])
            else:
                args[role] = raw
        return {
            "predicate": str(effect.get("predicate", "")),
            "args": args,
            "cardinality": int(effect.get("cardinality", 1)),
            "distinct_by": str(effect.get("distinct_by", "")),
            "effect_domain": str(effect.get("effect_domain", "world")),
        }

    def _validate_step_effects(
        self,
        node: dict[str, Any],
        ctx: Any,
        state: ToolExecutionState,
        *,
        tool: ToolAsset,
    ) -> dict[str, Any]:
        expected = [dict(item) if isinstance(item, dict) else to_primitive(item) for item in node.get("expected_effects", [])]
        report = {
            "program_node_id": str(node.get("node_id", "")),
            "expected_effects": expected,
            "observed_effects": [],
            "missing_effects": [],
            "step_effect_passed": True,
            "failure_code": "",
        }
        if not expected:
            state.step_effect_results.append(report)
            return report
        input_roles = set(map(str, (
            tool.signature.get("properties", {}) or {}
        )))
        output_roles = set(map(str, (
            tool.interface.get("output_schema", {}).get("properties", {})
            or {}
        )))
        resolutions = [
            self._step_effect_resolution(
                effect,
                state,
                input_roles=input_roles,
                output_roles=output_roles,
            )
            for effect in expected
        ]
        resolution_errors = [
            message
            for _effect, _wildcards, errors in resolutions
            for message in errors
        ]
        has_fresh_output = any(wildcards for _effect, wildcards, _errors in resolutions)
        observed: list[dict[str, Any]] = []
        passed = False
        validate_effect = getattr(ctx.harness.validator_channel(), "validate_atomic_effect", None)
        if not resolution_errors and not has_fresh_output and callable(validate_effect):
            try:
                passed = bool(validate_effect({
                    "effects": [effect for effect, _wildcards, _errors in resolutions],
                    "bindings": self._effect_bindings(state),
                }).passed)
            except Exception:
                passed = False
        if not passed and not resolution_errors and has_fresh_output:
            # A step may establish a declared fresh output whose concrete value
            # is published only by a later RETURN.  Match only that corresponding
            # predicate parameter as a wildcard; all other arguments and the
            # Effect domain remain exact.
            all_effects_observed = True
            for effect, wildcard_arguments, _errors in resolutions:
                matches = [
                    fact for fact in state.semantic_facts
                    if str(fact.get("predicate", "")).casefold()
                    == str(effect.get("predicate", "")).casefold()
                    and str(fact.get("effect_domain", "")).casefold()
                    == str(effect.get("effect_domain", "world")).casefold()
                    and all(
                        (
                            dict(fact.get("args") or {}).get(role)
                            not in (None, "")
                            if role in wildcard_arguments
                            else dict(fact.get("args") or {}).get(role)
                            == expected_value
                        )
                        for role, expected_value in dict(
                            effect.get("args") or {}
                        ).items()
                    )
                ]
                needed = max(1, int(effect.get("cardinality", 1) or 1))
                distinct_by = str(effect.get("distinct_by", ""))
                if distinct_by:
                    sufficient = len({
                        dict(fact.get("args") or {}).get(distinct_by)
                        for fact in matches
                        if dict(fact.get("args") or {}).get(distinct_by)
                        not in (None, "")
                    }) >= needed
                else:
                    sufficient = len(matches) >= needed
                all_effects_observed = (
                    all_effects_observed and sufficient
                )
                observed.extend(matches)
            if all_effects_observed:
                report.update({
                    "observed_effects": [dict(item) for item in observed],
                    "step_effect_passed": True,
                    "witness_refs": [
                        "semantic_fact:"
                        + str(item.get("predicate", ""))
                        + ":"
                        + repr(sorted(dict(item.get("args") or {}).items()))
                        for item in observed
                    ],
                })
                state.step_effect_results.append(report)
                return report
        if resolution_errors:
            report["resolution_errors"] = resolution_errors
        if not passed:
            report.update({
                "observed_effects": [dict(item) for item in observed],
                "missing_effects": expected,
                "step_effect_passed": False,
                "failure_code": "tool_step_effect_violation",
            })
        state.step_effect_results.append(report)
        return report

    def _execute_ir_nodes(
        self, nodes: list[dict[str, Any]], state: ToolExecutionState,
        ctx: Any, *, occurrence_id: str, span_id: str, tool: ToolAsset,
        terminal: list[dict[str, Any]],
    ) -> str:
        """Zero-LLM recursive IR control.

        Returns one of the frozen control signals.  ``terminal`` receives the
        winning action record.
        """

        for node in nodes:
            node_id = str(node.get("node_id", ""))
            state.program_node_id = node_id
            state.executed_control_step_count += 1
            if (
                state.max_control_steps
                and state.executed_control_step_count > state.max_control_steps
            ):
                state.failure_code = "tool_ir_control_step_exhausted"
                state.failure_message = (
                    f"Tool IR control-step bound {state.max_control_steps} "
                    f"exhausted at node {node.get('node_id')}"
                )
                return "FAIL_TOOL"
            if state.failure_code or terminal:
                return "BENCHMARK_TERMINAL" if terminal else "FAIL_TOOL"
            state.executed_nodes.append(node_id)
            state.path_tokens.append(node_id)
            opcode = str(node.get("op", ""))
            if opcode == "ACTION":
                if state.max_actions and state.executed_action_count >= state.max_actions:
                    state.failure_code = "tool_ir_max_actions_exhausted"
                    state.failure_message = (
                        f"Tool max_actions={state.max_actions} exhausted before "
                        f"node {node.get('node_id')}"
                    )
                    state.program_node_id = str(node.get("node_id", ""))
                    return "FAIL_TOOL"
                primitive = self._resolve_action_arguments(node, state)
                outcome = self._record_ir_action(
                    node, primitive, ctx, state,
                    occurrence_id=occurrence_id, span_id=span_id,
                )
                if not outcome["accepted"]:
                    state.failure_code = "tool_primitive_rejected"
                    state.failure_message = "Harness rejected Tool IR ACTION"
                    state.program_node_id = str(node.get("node_id", ""))
                    state.stop_condition_witnesses.append(f"rejected:{node.get('node_id')}")
                    return "FAIL_TOOL"
                effect_report = self._validate_step_effects(
                    node, ctx, state, tool=tool,
                )
                if not effect_report["step_effect_passed"]:
                    state.failure_code = "tool_step_effect_violation"
                    state.failure_message = (
                        f"required expected effect failed at {node.get('node_id')}"
                    )
                    state.program_node_id = str(node.get("node_id", ""))
                    return "FAIL_TOOL"
                if outcome["won"]:
                    terminal.append(outcome)
                    return "BENCHMARK_TERMINAL"
                if outcome["done"]:
                    state.failure_code = "tool_execution_error"
                    state.failure_message = "environment ended without success"
                    state.program_node_id = str(node.get("node_id", ""))
                    return "FAIL_TOOL"
            elif opcode == "IF":
                condition = dict(node.get("condition") or {})
                branch_taken = evaluate_condition(condition, state)
                branch = node.get("then_branch") if branch_taken else node.get("else_branch")
                state.path_tokens.append(
                    f"{node_id}:{'then' if branch_taken else 'else'}"
                )
                state.validated_paths.append(
                    f"{node.get('node_id')}:then" if branch_taken else f"{node.get('node_id')}:else"
                )
                if not branch_taken:
                    state.unvalidated_paths.append(f"{node.get('node_id')}:then")
                elif "else_branch" in node and node.get("else_branch") is not None:
                    state.unvalidated_paths.append(f"{node.get('node_id')}:else")
                signal = self._execute_ir_nodes(
                    list(branch or []), state, ctx, occurrence_id=occurrence_id,
                    span_id=span_id, tool=tool, terminal=terminal,
                )
                if signal:
                    return signal
            elif opcode == "FOR_EACH":
                collection_source = dict(
                    node.get("collection_source") or {}
                )
                values = resolve_collection(
                    collection_source, state,
                    semantic_compatible=getattr(ctx.harness, "semantic_value_compatible", None),
                )
                if (
                    not values
                    and self._selector_requires_match(collection_source)
                ):
                    state.failure_code = "tool_ir_selector_no_match"
                    state.failure_message = (
                        f"selector at {node_id} matched no collection values"
                    )
                    return "FAIL_TOOL"
                max_iterations = int(node.get("max_iterations", len(values)) or 0)
                variable = str(node.get("iteration_variable", ""))
                count = 0
                for value in values:
                    if count >= max_iterations or state.failure_code or terminal:
                        break
                    state.path_tokens.append(
                        f"{node_id}:iteration:{count + 1}"
                    )
                    state.local[variable] = value
                    signal = self._execute_ir_nodes(
                        list(node.get("body") or []), state, ctx,
                        occurrence_id=occurrence_id, span_id=span_id, tool=tool,
                        terminal=terminal,
                    )
                    count += 1
                    if signal in {"RETURN_PROGRAM", "BENCHMARK_TERMINAL", "FAIL_TOOL"}:
                        state.loop_iteration_counts[str(node.get("node_id", ""))] = count
                        return signal
                    if signal == "BREAK_LOOP":
                        state.loop_iteration_counts[str(node.get("node_id", ""))] = count
                        break
                state.loop_iteration_counts[str(node.get("node_id", ""))] = count
                if count > 1:
                    state.stop_condition_witnesses.append(
                        f"loop:{node.get('node_id')}:iterations:{count}"
                    )
            elif opcode == "STOP_WHEN":
                condition = dict(node.get("condition") or {})
                if evaluate_condition(condition, state):
                    state.stop_condition_witnesses.append(
                        f"stop:{node.get('node_id')}"
                    )
                    return "BREAK_LOOP"
            elif opcode == "RETURN":
                output_sources = normalize_return_output_sources(
                    node,
                    list(
                        tool.interface.get("output_schema", {})
                        .get("properties", {})
                    ),
                )
                outputs, refs = resolve_return_sources(
                    output_sources,
                    state,
                    semantic_compatible=getattr(
                        ctx.harness,
                        "semantic_value_compatible",
                        None,
                    ),
                )
                if any(value is None for value in outputs.values()):
                    selector_miss = any(
                        outputs.get(role) is None
                        and self._selector_requires_match(
                            output_sources.get(role)
                        )
                        for role in outputs
                    )
                    state.failure_code = (
                        "tool_ir_selector_no_match"
                        if selector_miss
                        else "tool_ir_return_output_unresolved"
                    )
                    state.failure_message = (
                        f"RETURN {node_id} selector matched no value"
                        if selector_miss
                        else f"RETURN {node_id} produced an unresolved output"
                    )
                    return "FAIL_TOOL"
                state.outputs.update(outputs)
                state.evidence_refs.extend(refs)
                return "RETURN_PROGRAM"
        return ""

    def _run_ir_v1(
        self, tool: ToolAsset, bindings: dict[str, Any], ctx: Any,
        *, occurrence_id: str, span_id: str,
    ) -> ToolExecutionResult:
        before_revision = ctx.world_revision
        state = self._ir_state(tool, bindings, ctx)
        program = [dict(node) for node in tool.artifact.get("program", [])]
        terminal: list[dict[str, Any]] = []
        control_signal = ""
        try:
            control_signal = self._execute_ir_nodes(
                program, state, ctx, occurrence_id=occurrence_id,
                span_id=span_id, tool=tool, terminal=terminal,
            )
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError) as exc:
            raw_code = str(exc).split(":", 1)[0]
            state.failure_code = (
                raw_code
                if raw_code.startswith("tool_ir_")
                or raw_code == "tool_step_effect_violation"
                else "tool_ir_execution_error"
            )
            state.failure_message = str(exc)
            total_nodes = self._program_node_count(program)
            path_id = self._program_path_id(state)
            final_effect_result = {
                "passed": False,
                "observed_effects": [],
                "missing_effects": [],
                "failure_code": state.failure_code,
            }
            ctx.trace_builder.finish_span(span_id)
            result = ToolExecutionResult(
                str(tool.ref), True, state.executed_action_count > 0, False,
                ctx.world_revision != before_revision, state.executed_action_count, None,
                [], {}, before_revision, ctx.world_revision,
                "tool", state.failure_code, state.failure_message,
                intrinsic_failure=True,
                executed_node_count=len(state.executed_nodes),
                remaining_node_count=max(
                    0, total_nodes - len(set(state.executed_nodes))
                ),
                path_id=path_id,
                program_node_id=state.program_node_id,
                tool_path_evidence=self._tool_path_evidence(
                    state,
                    outputs={},
                    terminal_interrupted=False,
                    final_effect_result=final_effect_result,
                ),
            )
            ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
                f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref),
                to_primitive(result), span_id,
            ))
            return result

        terminal_interrupted = bool(terminal and terminal[0].get("won"))
        outputs = dict(state.outputs)
        output_validation = self.validator.validate_output(tool, outputs)
        completed = bool(control_signal == "RETURN_PROGRAM")
        if completed and not output_validation.passed:
            completed = False
            state.failure_code = "tool_output_schema_error"
            state.failure_message = "; ".join(output_validation.messages)
        final_effects = [dict(item) if isinstance(item, dict) else to_primitive(item) for item in tool.artifact.get("final_effects", [])]
        observed_effects: list[dict[str, Any]] = []
        missing_effects: list[dict[str, Any]] = []
        for effect in final_effects:
            predicate = str(effect.get("predicate", ""))
            matches = [
                fact for fact in state.semantic_facts
                if str(fact.get("predicate", "")) == predicate
            ]
            if matches:
                observed_effects.append(dict(effect))
            else:
                missing_effects.append(dict(effect))
        atomic_effect_passed = False
        validate_effect = getattr(ctx.harness.validator_channel(), "validate_atomic_effect", None)
        if callable(validate_effect) and final_effects:
            atomic_effect_passed = bool(validate_effect({
                "effects": [
                    self._resolved_effect(effect, state)
                    for effect in final_effects
                ],
                "bindings": self._effect_bindings(state),
            }).passed)
        final_effect_result = {
            "passed": atomic_effect_passed,
            "observed_effects": [dict(item) for item in observed_effects],
            "missing_effects": [dict(item) for item in missing_effects],
            "failure_code": "" if atomic_effect_passed else "tool_ir_final_effect_failed",
        }
        failure_layer = ""
        failure_code = state.failure_code
        failure_message = state.failure_message
        if not completed and not terminal_interrupted and not failure_code:
            failure_layer = "tool"
            failure_code = "tool_ir_execution_error"
            failure_message = "Tool IR ended without RETURN or benchmark terminal"
        elif failure_code:
            failure_layer = "tool"
        state.failure_code = failure_code
        state.failure_message = failure_message
        total_nodes = self._program_node_count(program)
        path_id = self._program_path_id(state)
        result = ToolExecutionResult(
            str(tool.ref), True, state.executed_action_count > 0, completed,
            ctx.world_revision != before_revision,
            state.executed_action_count, None,
            state.bindings, outputs, before_revision, ctx.world_revision,
            failure_layer, failure_code, failure_message,
            terminal_interrupted=terminal_interrupted,
            intrinsic_failure=bool(failure_code and not terminal_interrupted),
            executed_node_count=len(state.executed_nodes),
            remaining_node_count=max(
                0, total_nodes - len(set(state.executed_nodes))
            ),
            path_id=path_id,
            program_node_id=state.program_node_id or (
                str(state.executed_nodes[-1]) if state.executed_nodes else ""
            ),
            expected_effects=final_effects,
            observed_effects=observed_effects,
            missing_effects=missing_effects,
            loop_iteration_counts=state.loop_iteration_counts,
            validated_paths=sorted(set(state.validated_paths)),
            unvalidated_paths=sorted(set(state.unvalidated_paths)),
            stop_condition_witnesses=list(state.stop_condition_witnesses),
            atomic_effect_passed=atomic_effect_passed,
            tool_path_evidence=self._tool_path_evidence(
                state,
                outputs=outputs,
                terminal_interrupted=terminal_interrupted,
                final_effect_result=final_effect_result,
            ),
        )
        ctx.trace_builder.finish_span(span_id)
        ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
            f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref),
            to_primitive(result), span_id,
        ))
        return result
