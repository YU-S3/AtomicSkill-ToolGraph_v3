"""Deterministic serial ToolBinding execution and post-effect validation."""

from __future__ import annotations

import uuid
from typing import Any

from ..core.bindings import BindingExprKind, BindingExpression, RuntimeBinding
from ..core.results import (
    ImplementationExecutionResult, NodeExecutionStatus, ToolCallPreflightResult,
)
from ..core.serialization import to_primitive
from ..core.errors import BudgetExhausted
from ..traces.schema import ImplementationInvocationRecord, ValidationRecord
from ..validation.engine import ValidationEngine
from .invocation_compiler import CompiledInvocation, _tool_arguments
from .tool_runner import ToolRunner


class ImplementationRunner:
    def __init__(self, validation: ValidationEngine) -> None:
        self.validation = validation
        self.tool_runner = ToolRunner(validation.tool)

    def run(
        self, compiled, preflight, occurrence, ctx, *, agent_prepared, execution_scope="registered",
    ):
        from ..traces.compiler_observer import program_window
        with program_window(compiled, preflight, occurrence, ctx, execution_scope):
            return self._run(compiled, preflight, occurrence, ctx,
                             agent_prepared=agent_prepared, execution_scope=execution_scope)

    def _run(
        self, compiled: CompiledInvocation, preflight: ToolCallPreflightResult,
        occurrence: Any, ctx: Any, *, agent_prepared: bool,
        execution_scope: str = "registered",
    ) -> ImplementationExecutionResult:
        if execution_scope not in {"registered", "runtime_trial"}:
            raise ValueError(
                f"unsupported Implementation execution_scope: {execution_scope!r}"
            )
        attempt_id = f"impl_attempt_{uuid.uuid4().hex}"
        if not preflight.passed:
            return ImplementationExecutionResult(
                str(compiled.implementation.ref), str(compiled.atomic.ref), False, False, False, False,
                failure_layer=preflight.failure_layer, failure_code=preflight.failure_code,
                node_status=NodeExecutionStatus.FAILED_NOT_STARTED,
            )
        span = ctx.trace_builder.start_span("implementation", occurrence.occurrence_id)
        tool_execution_start = len(
            getattr(ctx.trace_builder.trace, "tool_executions", ())
        )
        atomic_values = dict(preflight.normalized_arguments)
        tool_outputs: dict[tuple[str, str], Any] = {}
        tool_results = []
        started = completed = False
        failure_layer = failure_code = ""
        bindings_by_ref = {str(item.tool_ref): item for item in compiled.implementation.tool_bindings}
        tools_by_ref = {str(item.ref): item for item in compiled.tools}
        for binding in sorted(compiled.implementation.tool_bindings, key=lambda item: item.order):
            if ctx.execution_terminal():
                break
            tool = tools_by_ref[str(binding.tool_ref)]
            try:
                arguments = _tool_arguments(binding.parameter_mapping, atomic_values, tool_outputs, compiled.atomic, tool)
            except (KeyError, TypeError, ValueError) as exc:
                failure_layer, failure_code = "implementation", "implementation_mapping_error"
                ctx.trace_builder.trace.validations.append(ValidationRecord(
                    occurrence.occurrence_id, "implementation",
                    {"passed": False, "failure_codes": [failure_code], "messages": [str(exc)]},
                    ctx.world_revision))
                break
            try:
                result = self.tool_runner.run(
                    tool, arguments, ctx, occurrence_id=occurrence.occurrence_id,
                    parent_span_id=span.span_id, execution_scope=execution_scope,
                )
            except BudgetExhausted as exc:
                ctx.trace_builder.finish_span(span.span_id)
                actions = ctx.trace_builder.trace.environment_actions[span.action_start:]
                interrupted = ImplementationExecutionResult(str(compiled.implementation.ref),
                    str(compiled.atomic.ref), True, bool(actions), False, False,
                    failure_layer=exc.layer.value, failure_code=exc.code,
                    node_status=NodeExecutionStatus.DIRECT_FAILED if actions else NodeExecutionStatus.FAILED_NOT_STARTED)
                payload = to_primitive(interrupted)
                payload['interrupted_by_budget'] = True
                ctx.trace_builder.trace.implementation_invocations.append(ImplementationInvocationRecord(
                    attempt_id, occurrence.occurrence_id, str(compiled.implementation.ref),
                    dict(preflight.normalized_arguments), to_primitive(preflight), payload, span.span_id))
                if execution_scope == "runtime_trial":
                    self._exclude_trial_credit(ctx.trace_builder.trace, attempt_id, tool_execution_start)
                raise
            tool_results.append(result)
            started = started or result.started
            for role, value in result.output_candidates.items():
                tool_outputs[(binding.role, role)] = value
            if not result.completed:
                if result.failure_code or not result.terminal_interrupted:
                    failure_layer = result.failure_layer or "tool"
                    failure_code = result.failure_code or "tool_execution_error"
                break
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
        bindings = (
            {}
            if execution_scope == "runtime_trial"
            else ctx.binding_store.snapshot_for_node(occurrence)
        )
        bindings.update({item.role: item for item in preflight.binding_updates})
        try:
            authoritative_evidence_facts = ctx.atomic_evidence_for(occurrence).authoritative_facts()
        except (AttributeError, KeyError):
            authoritative_evidence_facts = []
        atomic_validation = self.validation.atomic.validate_execution_result(
            compiled.atomic, occurrence, bindings, output_candidates, ctx.harness.validator_channel(),
            current_revision=ctx.world_revision, authoritative_evidence_facts=authoritative_evidence_facts,
            semantic_compatible=getattr(ctx.harness, "semantic_value_compatible", None))
        ctx.trace_builder.trace.validations.append(ValidationRecord(
            occurrence.occurrence_id, "atomic", to_primitive(atomic_validation), ctx.world_revision,
        ))
        atomic_passed = bool(completed and not failure_code and atomic_validation.passed)
        if atomic_passed and execution_scope == "registered":
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
        validated_outputs = {role: item.value for role, item in atomic_validation.validated_output_bindings.items()} if atomic_passed else {}
        ctx.trace_builder.finish_span(span.span_id)
        result = ImplementationExecutionResult(
            str(compiled.implementation.ref), str(compiled.atomic.ref), True, started,
            completed, atomic_passed, tool_results, {item.role: item for item in preflight.binding_updates},
            validated_outputs, f"revision:{tool_results[0].before_revision if tool_results else ctx.world_revision}",
            f"revision:{ctx.world_revision}", failure_layer, failure_code,
            NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS if atomic_passed and agent_prepared
            else NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS if atomic_passed
            else NodeExecutionStatus.DIRECT_FAILED if started else NodeExecutionStatus.FAILED_NOT_STARTED,
            terminal_interrupted=bool(
                ctx.execution_terminal() and not completed
            ),
            atomic_witness_refs=list(dict.fromkeys(
                str(ref) for ref in atomic_validation.witness_refs
            )),
            official_terminal_observed=ctx.execution_terminal(),
            validated_output_bindings=dict(atomic_validation.validated_output_bindings) if atomic_passed else {},
            certified_input_bindings=dict(atomic_validation.certified_input_bindings) if atomic_passed else {},
        )
        if result.terminal_interrupted and result.started and not result.failure_code:
            result.node_status = NodeExecutionStatus.TERMINAL_PARTIAL
        invocation_record = ImplementationInvocationRecord(
            attempt_id, occurrence.occurrence_id, str(compiled.implementation.ref),
            dict(preflight.normalized_arguments), to_primitive(preflight), to_primitive(result), span.span_id,
        )
        ctx.trace_builder.trace.implementation_invocations.append(invocation_record)
        if execution_scope == "runtime_trial":
            self._exclude_trial_credit(ctx.trace_builder.trace, attempt_id, tool_execution_start)
        return result

    @staticmethod
    def _exclude_trial_credit(trace, attempt_id, tool_execution_start):
        """Task-local attempts remain raw audit even on budget interruption.

        They never name persistent deployment assets. This must happen before
        propagating an exception, not only after a normally returned trial.
        """
        exclusions = trace.metadata.setdefault("runtime_trial_credit_exclusions", {
            "implementation_attempt_ids": [], "tool_execution_ids": [],
        })
        exclusions["implementation_attempt_ids"] = list(dict.fromkeys([
            *exclusions.get("implementation_attempt_ids", []), attempt_id,
        ]))
        exclusions["tool_execution_ids"] = list(dict.fromkeys([
            *exclusions.get("tool_execution_ids", []),
            *[str(item.attempt_id) for item in trace.tool_executions[tool_execution_start:]],
        ]))
