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
        executed = 0
        started = False
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
            if result.done and not result.won:
                failure_index, failure_code, failure_message = index, "tool_execution_error", "environment ended without success"
                break
        completed = failure_index is None and executed == len(steps)
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
        )
        ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
            f"tool_attempt_{uuid.uuid4().hex}", occurrence_id, str(tool.ref), to_primitive(tool_result), span.span_id,
        ))
        return tool_result
