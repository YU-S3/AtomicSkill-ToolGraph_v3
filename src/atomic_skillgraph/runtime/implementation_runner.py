"""Deterministic serial ToolBinding execution and post-effect validation."""

from __future__ import annotations

import uuid
from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression, RuntimeBinding
from ..core.results import (
    ImplementationExecutionResult, NodeExecutionStatus, ToolCallPreflightResult,
)
from ..core.serialization import to_primitive
from ..traces.schema import ImplementationInvocationRecord, ValidationRecord
from ..validation.engine import ValidationEngine
from .invocation_compiler import CompiledInvocation
from .tool_runner import ToolRunner


class ImplementationRunner:
    def __init__(self, validation: ValidationEngine) -> None:
        self.validation = validation
        self.tool_runner = ToolRunner(validation.tool)

    def _tool_arguments(
        self, mapping: dict[str, BindingExpression], atomic_values: dict[str, Any],
        tool_outputs: dict[tuple[str, str], Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for role, expression in mapping.items():
            expression = BindingExpression.from_dict(expression)
            if expression.kind is BindingExprKind.CONSTANT:
                result[role] = expression.constant
            elif expression.kind is BindingExprKind.SKILL_INPUT:
                if expression.source_role not in atomic_values:
                    raise KeyError(f"unresolved Atomic input {expression.source_role}")
                result[role] = atomic_values[expression.source_role]
            elif expression.kind is BindingExprKind.TOOL_OUTPUT:
                result[role] = tool_outputs[(expression.source_step, expression.source_role)]
            elif expression.kind in {BindingExprKind.DATA_FLOW, BindingExprKind.ADAPTER_TRANSFORM}:
                if expression.source_role not in atomic_values:
                    raise KeyError(f"unresolved mapped value {expression.source_role}")
                result[role] = atomic_values[expression.source_role]
            else:
                raise ValueError(f"unsupported BindingExpression {expression.kind}")
        return result

    def run(
        self, compiled: CompiledInvocation, preflight: ToolCallPreflightResult,
        occurrence: Any, ctx: Any, *, agent_prepared: bool,
    ) -> ImplementationExecutionResult:
        attempt_id = f"impl_attempt_{uuid.uuid4().hex}"
        if not preflight.passed:
            return ImplementationExecutionResult(
                str(compiled.implementation.ref), str(compiled.atomic.ref), False, False, False, False,
                failure_layer=preflight.failure_layer, failure_code=preflight.failure_code,
                node_status=NodeExecutionStatus.FAILED_NOT_STARTED,
            )
        span = ctx.trace_builder.start_span("implementation", occurrence.occurrence_id)
        atomic_values = dict(preflight.normalized_arguments)
        tool_outputs: dict[tuple[str, str], Any] = {}
        tool_results = []
        started = completed = False
        failure_layer = failure_code = ""
        bindings_by_ref = {str(item.tool_ref): item for item in compiled.implementation.tool_bindings}
        tools_by_ref = {str(item.ref): item for item in compiled.tools}
        for binding in sorted(compiled.implementation.tool_bindings, key=lambda item: item.order):
            tool = tools_by_ref[str(binding.tool_ref)]
            try:
                arguments = self._tool_arguments(binding.parameter_mapping, atomic_values, tool_outputs)
            except (KeyError, TypeError, ValueError):
                failure_layer, failure_code = "implementation", "implementation_mapping_error"
                break
            result = self.tool_runner.run(tool, arguments, ctx, occurrence_id=occurrence.occurrence_id, parent_span_id=span.span_id)
            tool_results.append(result)
            started = started or result.started
            if not result.completed:
                failure_layer = result.failure_layer or "tool"
                failure_code = result.failure_code or "tool_execution_error"
                break
            for role, value in result.output_candidates.items():
                tool_outputs[(binding.role, role)] = value
        else:
            completed = bool(tool_results) and all(item.completed for item in tool_results)

        output_candidates: dict[str, Any] = {}
        output_mapping = compiled.implementation.execution_policy.get("output_mapping", {})
        for role, raw_expression in output_mapping.items():
            expression = BindingExpression.from_dict(raw_expression) if isinstance(raw_expression, dict) else raw_expression
            if isinstance(expression, BindingExpression):
                if expression.kind is BindingExprKind.TOOL_OUTPUT:
                    output_candidates[role] = tool_outputs.get((expression.source_step, expression.source_role))
                elif expression.kind is BindingExprKind.CONSTANT:
                    output_candidates[role] = expression.constant
                else:
                    output_candidates[role] = atomic_values.get(expression.source_role)
            elif isinstance(expression, str):
                output_candidates[role] = atomic_values.get(expression, expression)
            else:
                output_candidates[role] = expression
        for output in compiled.atomic.outputs:
            if output.name not in output_candidates and output.name in atomic_values:
                output_candidates[output.name] = atomic_values[output.name]
        bindings = ctx.binding_store.snapshot_for_node(occurrence)
        bindings.update({item.role: item for item in preflight.binding_updates})
        atomic_validation = self.validation.atomic.validate(
            compiled.atomic, occurrence, bindings, ctx.harness.validator_channel(), output_candidates,
        )
        ctx.trace_builder.trace.validations.append(ValidationRecord(
            occurrence.occurrence_id, "atomic", to_primitive(atomic_validation), ctx.world_revision,
        ))
        atomic_passed = bool(started and atomic_validation.passed)
        if atomic_passed:
            repeat_values = {
                **dict(atomic_values),
                **{
                    item.role: item.value
                    for item in preflight.binding_updates
                },
                **dict(output_candidates),
            }
            repeat_commit = ctx.binding_store.commit_repeat_bindings(
                occurrence.step_id,
                repeat_values,
                effect_passed=True,
            )
            ctx.trace_builder.trace.validations.append(ValidationRecord(
                occurrence.occurrence_id,
                "runtime_repeat_commit",
                to_primitive(repeat_commit),
                ctx.world_revision,
            ))
            if not repeat_commit.passed:
                atomic_passed = False
                failure_layer = "runtime_binding"
                failure_code = (
                    repeat_commit.failure_codes[0]
                    if repeat_commit.failure_codes
                    else "runtime_repetition_distinctness_violation"
                )
        if completed and not atomic_passed and not failure_code:
            failure_layer, failure_code = "atomic", "atomic_effect_violation"
        validated_outputs = output_candidates if atomic_passed else {}
        ctx.trace_builder.finish_span(span.span_id)
        result = ImplementationExecutionResult(
            str(compiled.implementation.ref), str(compiled.atomic.ref), True, started,
            completed, atomic_passed, tool_results, {item.role: item for item in preflight.binding_updates},
            validated_outputs, f"revision:{tool_results[0].before_revision if tool_results else ctx.world_revision}",
            f"revision:{ctx.world_revision}", failure_layer, failure_code,
            NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS if atomic_passed and agent_prepared
            else NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS if atomic_passed
            else NodeExecutionStatus.DIRECT_FAILED if started else NodeExecutionStatus.FAILED_NOT_STARTED,
        )
        ctx.trace_builder.trace.implementation_invocations.append(ImplementationInvocationRecord(
            attempt_id, occurrence.occurrence_id, str(compiled.implementation.ref),
            dict(preflight.normalized_arguments), to_primitive(preflight), to_primitive(result), span.span_id,
        ))
        return result
