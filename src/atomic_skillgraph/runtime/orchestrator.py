"""Strict occurrence-order task execution; Runtime never redesigns the plan."""

from __future__ import annotations

from typing import Any

from ..core.results import NodeExecutionStatus, RuntimeLinearPlan
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode
from ..harness.protocol import HarnessAdapter, HarnessTask
from ..planner.pipeline import PlannerPipeline
from ..traces.schema import NodeTraceRecord, TaskRecord, TraceBuilder, TraceRecord, ValidationRecord
from ..validation.engine import ValidationEngine
from .budget import RuntimeBudget
from .invocation_compiler import InvocationCompiler
from .node_executor import NodeExecutor
from .task_context import TaskRuntimeContext


class RuntimeOrchestrator:
    def __init__(
        self, planner: PlannerPipeline, harness: HarnessAdapter,
        invocation_compiler: InvocationCompiler, validation: ValidationEngine,
        session_factory: Any, *, runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.planner, self.harness = planner, harness
        self.invocation_compiler, self.validation = invocation_compiler, validation
        self.session_factory = session_factory
        self.runtime_config = runtime_config or {}
        self.node_executor = NodeExecutor(invocation_compiler, validation, session_factory)

    def _budget(self) -> RuntimeBudget:
        config = self.runtime_config
        return RuntimeBudget(
            global_action_budget=int(config.get("global_action_budget", 100)),
            node_action_budget=int(config.get("node_action_budget", 35)),
            token_limits=dict(config.get("token_limits", {})), turn_limits=dict(config.get("turn_limits", {})),
        )

    def _create_context(self, task: HarnessTask, plan: RuntimeLinearPlan) -> TaskRuntimeContext:
        task_record = TaskRecord(
            task.task_id, task.benchmark, task.goal, task.task_type,
            str(task.metadata.get("task_signature") or task.context.get("game_file") or task.task_id),
            {
                **dict(task.metadata),
                "env_index": task.context.get("env_index"),
                "game_file": task.context.get("game_file", ""),
                "split": getattr(self.harness, "split", ""),
            },
        )
        trace = TraceRecord.create(task_record, to_primitive(plan.task_contract), plan.planner_audit, to_primitive(plan))
        builder = TraceBuilder(trace)
        return TaskRuntimeContext.create(task, plan, self.harness, builder, self._budget())

    def run_task(
        self, task: HarnessTask, *, mode: RuntimeMode | str = RuntimeMode.ONLINE,
    ) -> TraceRecord:
        initial_observation = str(task.context.get("initial_observation", ""))
        plan = self.planner.build_plan(task, self.harness, mode=mode, initial_observation=initial_observation)
        ctx = self._create_context(task, plan)

        initial_terminal = self.validation.task.terminal(
            ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
        )
        if plan.source == "full_dynamic":
            result = self.node_executor.run_dynamic(ctx)
            ctx.trace_builder.trace.benchmark_success = bool(result["success"])
            ctx.trace_builder.trace.node_contract_success = False
            ctx.trace_builder.trace.graph_self_sufficient_success = False
            ctx.trace_builder.trace.graph_full_completion = False
            ctx.trace_builder.trace.learning_eligible = bool(result["success"])
            ctx.trace_builder.trace.metadata["dynamic_result"] = result
            ctx.trace_builder.trace.infrastructure_failure = result.get("failure_code") in {
                "infrastructure_failure", "llm_error",
            }
            return ctx.trace_builder.finish()

        if initial_terminal.passed:
            self._mark_remaining_terminal(ctx, 0)
        else:
            for index, step_id in enumerate(plan.control_sequence):
                ctx.current_step_index = index
                occurrence = plan.occurrence(step_id)
                ctx.budget.begin_node(occurrence.occurrence_id)
                node = ctx.trace_builder.start_node(occurrence.occurrence_id, occurrence.step_id, str(occurrence.node_ref))
                atomic = self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
                ctx.binding_store.apply_data_flow(plan, step_id, ctx.validated_outputs, revision=ctx.world_revision)
                ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
                bindings = ctx.binding_store.snapshot_for_node(occurrence)

                already = self.validation.atomic.already_satisfied(
                    atomic, occurrence, bindings, ctx.harness.validator_channel(),
                )
                ctx.trace_builder.trace.validations.append(ValidationRecord(
                    occurrence.occurrence_id, "already_satisfied", to_primitive(already), ctx.world_revision,
                ))
                if already.passed:
                    outputs = self.node_executor._validated_output_candidates(atomic, bindings, ctx)
                    node.status = NodeExecutionStatus.ALREADY_SATISFIED
                    node.validated_outputs = outputs
                    if outputs:
                        ctx.binding_store.publish_validated_outputs(occurrence, outputs, already.witness_refs, ctx.world_revision)
                        ctx.validated_outputs[occurrence.occurrence_id] = outputs
                    terminal = self.validation.task.terminal(
                        ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
                    )
                    if terminal.passed:
                        self._mark_remaining_terminal(ctx, index + 1)
                        break
                    continue

                invocations = self.invocation_compiler.compile_candidates(
                    occurrence, ctx.binding_store,
                    max_candidates=int(self.runtime_config.get("max_implementation_candidates", 3)),
                    task_id=task.task_id,
                )
                if not invocations:
                    direct = self.node_executor.not_started(occurrence, failure_code="no_compatible_implementation")
                else:
                    direct = self.node_executor.try_autonomous(occurrence, invocations, ctx)
                    if direct is None:
                        direct = self.node_executor.run_preparation_session(
                            occurrence, invocations, ctx,
                            learned_call_repair_limit=int(self.runtime_config.get("learned_toolcall_repair_limit", 2)),
                        )
                node.direct_result = to_primitive(direct)
                final = direct
                if not direct.atomic_effect_passed:
                    seeded = self.node_executor.run_seeded_fresh(occurrence, ctx)
                    node.seeded_result = to_primitive(seeded)
                    final = seeded
                    if not seeded.atomic_effect_passed:
                        node.status = NodeExecutionStatus.SEEDED_FAILED
                        node.failure = {
                            "failure_layer": seeded.failure_layer, "failure_code": seeded.failure_code,
                            "direct_started": direct.started,
                        }
                        ctx.plan_failed = True
                        break
                node.status = final.node_status
                node.validated_outputs = dict(final.validated_outputs)
                if final.validated_outputs:
                    validation_refs = self._latest_atomic_witnesses(ctx, occurrence.occurrence_id)
                    ctx.binding_store.publish_validated_outputs(
                        occurrence, final.validated_outputs, validation_refs, ctx.world_revision,
                    )
                    ctx.validated_outputs[occurrence.occurrence_id] = dict(final.validated_outputs)
                    for role, value in final.validated_outputs.items():
                        ctx.evidence_store.add_validated_tool_output(role, value, validation_refs)

                terminal = self.validation.task.terminal(
                    ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
                )
                ctx.trace_builder.trace.validations.append(ValidationRecord(
                    occurrence.occurrence_id, "task_terminal", to_primitive(terminal), ctx.world_revision,
                ))
                if terminal.passed:
                    self._mark_remaining_terminal(ctx, index + 1)
                    break
            else:
                ctx.current_step_index = len(plan.control_sequence)

        terminal = self.validation.task.terminal(
            ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
        )
        if ctx.plan_boundary_reached() and not terminal.passed:
            ctx.task_rescue_used = True
            # Task rescue is task-level Dynamic execution.  It keeps every
            # action already charged to the global episode budget, but it must
            # not inherit the final occurrence's node-local quota.
            ctx.budget.end_node()
            rescue = self.node_executor.run_dynamic(ctx, rescue=True)
            ctx.trace_builder.trace.task_rescue_required = True
            ctx.trace_builder.trace.metadata["task_rescue"] = rescue
            if rescue.get("failure_code") in {
                "infrastructure_failure", "llm_error",
            }:
                ctx.trace_builder.trace.infrastructure_failure = True
            terminal = self.validation.task.terminal(
                ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
            )

        trace = ctx.trace_builder.trace
        trace.benchmark_success = bool(getattr(ctx.harness.validator_channel(), "won", False))
        trace.node_contract_success = bool(trace.node_records) and all(
            node.status not in {NodeExecutionStatus.NOT_STARTED, NodeExecutionStatus.FAILED_NOT_STARTED,
                                NodeExecutionStatus.DIRECT_FAILED, NodeExecutionStatus.SEEDED_FAILED}
            for node in trace.node_records
        )
        trace.implementation_direct_success = any(
            node.status in {NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS, NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS}
            for node in trace.node_records
        )
        trace.graph_full_completion = (
            len(trace.node_records) == len(plan.occurrences)
            and all(node.status is not NodeExecutionStatus.SKIPPED_GOAL_TERMINAL for node in trace.node_records)
        )
        composite = self.validation.composite.validate_runtime(
            plan, trace.node_records, ctx.validated_outputs,
            task_rescue_required=ctx.task_rescue_used, task_contract_result=terminal,
        )
        trace.validations.append(ValidationRecord("", "composite", to_primitive(composite), ctx.world_revision))
        trace.graph_self_sufficient_success = composite.passed
        trace.learning_eligible = bool(trace.benchmark_success and terminal.passed)
        if trace.benchmark_success and not terminal.passed:
            trace.metadata["anomaly"] = "benchmark_goal_contract_mismatch"
        trace.metadata["invocation_compile_rejections"] = list(self.invocation_compiler.compile_rejections)
        return ctx.trace_builder.finish()

    def _latest_atomic_witnesses(self, ctx: TaskRuntimeContext, occurrence_id: str) -> list[str]:
        for record in reversed(ctx.trace_builder.trace.validations):
            if record.occurrence_id == occurrence_id and record.level == "atomic":
                refs = list(record.result.get("witness_refs", []))
                if refs:
                    return refs
        return [f"validator:occurrence:{occurrence_id}:revision:{ctx.world_revision}"]

    def _mark_remaining_terminal(self, ctx: TaskRuntimeContext, start_index: int) -> None:
        existing = {node.step_id for node in ctx.trace_builder.trace.node_records}
        for step_id in ctx.plan.control_sequence[start_index:]:
            if step_id in existing:
                continue
            occurrence = ctx.plan.occurrence(step_id)
            ctx.trace_builder.trace.node_records.append(NodeTraceRecord(
                occurrence.occurrence_id, occurrence.step_id, str(occurrence.node_ref),
                NodeExecutionStatus.SKIPPED_GOAL_TERMINAL,
            ))
        ctx.current_step_index = len(ctx.plan.control_sequence)
