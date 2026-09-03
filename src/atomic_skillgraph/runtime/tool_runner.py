"""Execute serial Primitive IR against the Harness, preserving partial side effects."""

from __future__ import annotations

import uuid
from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import ToolAsset
from ..core.results import PrimitiveToolStep, ToolExecutionResult
from ..core.serialization import to_primitive
from ..traces.schema import EnvironmentActionRecord, ToolExecutionRecord
from ..validation.tool_validator import ToolValidator
from ..tooling.ir import ToolExecutionState, evaluate_condition, program_paths, resolve_collection, resolve_return_sources


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
            # Budget and Harness exceptions are not intrinsic Tool failures.
            # Let the orchestrator classify them (including infrastructure)
            # instead of manufacturing negative Tool evidence.
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
                # Benchmark terminal authority: never execute the remaining IR.
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
        )
        ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
            f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref), to_primitive(tool_result), span.span_id,
        ))
        return tool_result


    def _ir_state(self, tool: ToolAsset, bindings: dict[str, Any], ctx: Any) -> ToolExecutionState:
        snapshot = getattr(ctx.harness.validator_channel(), "snapshot", lambda: {})()
        if isinstance(snapshot, dict):
            facts = snapshot.get("facts", [])
        elif isinstance(snapshot, list):
            facts = snapshot
        else:
            facts = []
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
            catalog=catalog,
            semantic_facts=[dict(item) for item in facts],
            binding_evidence=[],
        )

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
        self, node: dict[str, Any], primitive: PrimitiveToolStep, result: Any,
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
        state.executed_nodes.append(str(node.get("node_id", "")))
        state.catalog = [
            {
                "action_id": str(item.action_id),
                "revision": int(item.revision),
                "action_type": str(item.action_type),
                "arguments": dict(item.arguments),
            }
            for item in result.catalog
        ]
        snapshot = getattr(ctx.harness.validator_channel(), "snapshot", lambda: {})()
        if isinstance(snapshot, dict):
            state.semantic_facts = [dict(item) for item in snapshot.get("facts", [])]
        elif isinstance(snapshot, list):
            state.semantic_facts = [dict(item) for item in snapshot]
        return {
            "accepted": bool(result.accepted),
            "won": bool(result.won),
            "done": bool(result.done),
            "revision": int(result.new_revision),
            "action_type": spec.action_type,
            "arguments": dict(spec.arguments),
        }

    def _execute_ir_nodes(
        self, nodes: list[dict[str, Any]], state: ToolExecutionState,
        ctx: Any, *, occurrence_id: str, span_id: str, tool: ToolAsset,
        stop: list[bool], returned: list[bool], terminal: list[dict[str, Any]],
    ) -> None:
        """Zero-LLM recursive IR control.  ``terminal`` receives the winning action record."""
        for node in nodes:
            if stop[0] or returned[0] or terminal:
                return
            opcode = str(node.get("op", ""))
            if opcode == "ACTION":
                primitive = self._resolve_action_arguments(node, state)
                outcome = self._record_ir_action(
                    node, primitive, None, ctx, state,
                    occurrence_id=occurrence_id, span_id=span_id,
                )
                if not outcome["accepted"]:
                    stop[0] = True
                    state.stop_condition_witnesses.append(
                        f"rejected:{node.get('node_id')}"
                    )
                    return
                if outcome["won"]:
                    terminal.append(outcome)
                    return
                if outcome["done"]:
                    stop[0] = True
                    return
            elif opcode == "IF":
                condition = dict(node.get("condition") or {})
                branch_taken = evaluate_condition(condition, state)
                branch = node.get("then_branch") if branch_taken else node.get("else_branch")
                state.validated_paths.append(
                    f"{node.get('node_id')}:then" if branch_taken else f"{node.get('node_id')}:else"
                )
                if not branch_taken:
                    state.unvalidated_paths.append(f"{node.get('node_id')}:then")
                elif "else_branch" in node and node.get("else_branch") is not None:
                    state.unvalidated_paths.append(f"{node.get('node_id')}:else")
                self._execute_ir_nodes(
                    list(branch or []), state, ctx, occurrence_id=occurrence_id,
                    span_id=span_id, tool=tool, stop=stop, returned=returned,
                    terminal=terminal,
                )
            elif opcode == "FOR_EACH":
                values = resolve_collection(
                    dict(node.get("collection_source") or {}), state,
                )
                max_iterations = int(node.get("max_iterations", len(values)) or 0)
                variable = str(node.get("iteration_variable", ""))
                count = 0
                for value in values:
                    if count >= max_iterations or stop[0] or returned[0] or terminal:
                        break
                    state.local[variable] = value
                    self._execute_ir_nodes(
                        list(node.get("body") or []), state, ctx,
                        occurrence_id=occurrence_id, span_id=span_id, tool=tool,
                        stop=stop, returned=returned, terminal=terminal,
                    )
                    count += 1
                state.loop_iteration_counts[str(node.get("node_id", ""))] = count
                if count > 1:
                    state.stop_condition_witnesses.append(
                        f"loop:{node.get('node_id')}:iterations:{count}"
                    )
            elif opcode == "STOP_WHEN":
                condition = dict(node.get("condition") or {})
                if evaluate_condition(condition, state):
                    stop[0] = True
                    state.stop_condition_witnesses.append(
                        f"stop:{node.get('node_id')}"
                    )
                    return
            elif opcode == "RETURN":
                outputs, refs = resolve_return_sources(
                    dict(node.get("output_sources") or {}), state,
                )
                state.outputs.update(outputs)
                state.evidence_refs.extend(refs)
                returned[0] = True
                return

    def _run_ir_v1(
        self, tool: ToolAsset, bindings: dict[str, Any], ctx: Any,
        *, occurrence_id: str, span_id: str,
    ) -> ToolExecutionResult:
        before_revision = ctx.world_revision
        state = self._ir_state(tool, bindings, ctx)
        program = [dict(node) for node in tool.artifact.get("program", [])]
        stop = [False]
        returned = [False]
        terminal: list[dict[str, Any]] = []
        try:
            self._execute_ir_nodes(
                program, state, ctx, occurrence_id=occurrence_id,
                span_id=span_id, tool=tool, stop=stop, returned=returned,
                terminal=terminal,
            )
        except (KeyError, TypeError, ValueError) as exc:
            ctx.trace_builder.finish_span(span_id)
            result = ToolExecutionResult(
                str(tool.ref), True, bool(state.executed_nodes), False,
                ctx.world_revision != before_revision, 0, None,
                [], {}, before_revision, ctx.world_revision,
                "tool", "tool_ir_execution_error", str(exc),
                intrinsic_failure=False,
                executed_node_count=len(state.executed_nodes),
                path_id="",
            )
            ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
                f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref),
                to_primitive(result), span_id,
            ))
            return result

        terminal_interrupted = bool(terminal and terminal[0].get("won"))
        outputs = dict(state.outputs)
        if not outputs:
            output_mapping = tool.artifact.get("output_mapping", {})
            for role, expression in output_mapping.items():
                if isinstance(expression, dict) and "kind" in expression:
                    expression = BindingExpression.from_dict(expression)
                    outputs[role] = (
                        expression.constant
                        if expression.kind is BindingExprKind.CONSTANT
                        else state.bindings.get(expression.source_role)
                    )
                else:
                    outputs[role] = state.bindings.get(role, expression)
        output_validation = self.validator.validate_output(tool, outputs)
        completed = bool(returned[0] or (stop[0] and not terminal_interrupted))
        if completed and not output_validation.passed:
            completed = False
        if terminal_interrupted and not outputs:
            for role in (tool.interface.get("output_schema", {}).get("properties", {}) or {}):
                if role in state.bindings:
                    outputs[role] = state.bindings[role]

        # Step/effect diagnostics: benchmark won never proves an unexecuted Tool
        # step, and a terminal interruption is explicitly not an intrinsic failure.
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
                "effects": final_effects,
                "bindings": state.bindings,
            }).passed)

        path_report = program_paths(program)
        total_nodes = sum(
            1 for _node in program for _node in self._walk_for_count(program)
        )
        result = ToolExecutionResult(
            str(tool.ref), True, bool(state.executed_nodes), completed,
            ctx.world_revision != before_revision,
            len(state.executed_nodes), None,
            state.bindings, outputs, before_revision, ctx.world_revision,
            "" if completed or terminal_interrupted else "tool",
            "" if completed or terminal_interrupted else "tool_ir_execution_error",
            "",
            terminal_interrupted=terminal_interrupted,
            intrinsic_failure=bool(stop[0] and not terminal_interrupted and not returned[0]),
            executed_node_count=len(state.executed_nodes),
            remaining_node_count=max(0, total_nodes - len(state.executed_nodes)),
            path_id=",".join(state.executed_nodes),
            program_node_id=(
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
        )
        ctx.trace_builder.finish_span(span_id)
        ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
            f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref),
            to_primitive(result), span_id,
        ))
        return result

    @staticmethod
    def _walk_for_count(program: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for node in program:
            result.append(node)
            if node.get("op") == "IF":
                result.extend(ToolRunner._walk_for_count(list(node.get("then_branch") or [])))
                result.extend(ToolRunner._walk_for_count(list(node.get("else_branch") or [])))
            elif node.get("op") == "FOR_EACH":
                result.extend(ToolRunner._walk_for_count(list(node.get("body") or [])))
        return result
