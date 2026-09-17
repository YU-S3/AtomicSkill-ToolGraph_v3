"""Verified graph execution; the Runtime Agent is invoked only at breakpoints."""
from __future__ import annotations

from typing import Any

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.results import NodeExecutionStatus
from .checkpoint import increment
from .runtime_step import run_runtime_step
from .support_closure import SupportClosure


class VerifiedCompositeExecutor:
    def __init__(self, node_executor: Any) -> None:
        self.nodes = node_executor

    def run_occurrence(self, occurrence: Any, ctx: Any) -> Any:
        executor = self.nodes
        occurrence_id = occurrence.occurrence_id
        excluded = ctx.rejected_runtime_implementations.setdefault(occurrence_id, set())
        mode = ctx.runtime_step_modes.get(occurrence_id, "preparation")
        try:
            while True:
                for draft_id, trial in ctx.runtime_tool_trials.items():
                    if trial.get("source_occurrence_id") == occurrence_id and not trial.get("terminal_interrupted"):
                        executor._mark_runtime_trial_parent_resumed(ctx, {"draft_id": draft_id, "trial": trial})
                assisted_here = any(item.get("occurrence_id") == occurrence_id
                                    for item in ctx.trace_builder.trace.metadata.get("runtime_steps", []))
                effect = executor._complete_from_current_effect(
                    occurrence, ctx, mode=mode if assisted_here else "entry", preferred_values=[],
                )
                if effect is not None:
                    executor._mark_runtime_trial_parent_completed(ctx, occurrence)
                    return effect
                if ctx.benchmark_terminal():
                    return executor._runtime_automation_terminal_boundary(occurrence)
                compiled = executor.invocation_compiler.compile_candidates(
                    occurrence, ctx.binding_store,
                    max_candidates=int(ctx.runtime_config.get("max_implementation_candidates", 3)),
                    task_id=ctx.task_id,
                )
                compiled = [item for item in compiled if str(item.implementation.ref) not in excluded]
                if not compiled:
                    mode = "seeded"
                ctx.runtime_step_modes[occurrence_id] = mode
                bootstrap = not ctx.graph_bootstrap_completed
                result = None
                if not bootstrap:
                    increment(ctx, "composite_auto_gate_attempt_count")
                    if ctx.runtime_config.get("support_closure", False):
                        supported = SupportClosure(executor).close(occurrence, ctx, compiled)
                        if ctx.benchmark_terminal():
                            return executor._runtime_automation_terminal_boundary(occurrence)
                        if supported:
                            effect = executor._complete_from_current_effect(
                                occurrence, ctx, mode=mode if assisted_here else "entry", preferred_values=[],
                            )
                            if effect is not None:
                                executor._mark_runtime_trial_parent_completed(ctx, occurrence)
                                return effect
                    result = executor.try_autonomous(occurrence, compiled, ctx) if compiled else None
                if result is not None:
                    if result.atomic_effect_passed:
                        increment(ctx, "composite_auto_node_count")
                        assisted = any(item.get("occurrence_id") == occurrence_id
                                       for item in ctx.trace_builder.trace.metadata.get("runtime_steps", []))
                        if mode == "seeded":
                            result.node_status = NodeExecutionStatus.SEEDED_SUCCESS
                        elif assisted:
                            result.node_status = NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS
                        else:
                            increment(ctx, "post_bootstrap_llm_free_node_count")
                        executor._mark_runtime_trial_parent_completed(ctx, occurrence)
                        return result
                    if result.started:
                        mode = "seeded"
                        ctx.record_failed_invocation(
                            occurrence_id=occurrence_id, implementation_ref=result.implementation_ref,
                            failure_code=result.failure_code, message=result.failure_code,
                        )
                    if ctx.benchmark_terminal():
                        return result
                atomic = executor.invocation_compiler.skills.get_atomic(occurrence.node_ref)
                missing = ctx.binding_store.runtime_prompt_projection(
                    occurrence, atomic.inputs,
                )["missing_or_insufficient_bindings"]
                support_candidates = executor._retrieve_runtime_support_candidates(
                    blocked_atomic=atomic, missing_roles=missing, ctx=ctx,
                    obligations=tuple(SupportClosure(executor).obligations(occurrence, atomic, ctx, compiled)),
                )
                if not bootstrap:
                    increment(ctx, "composite_breakpoint_count")
                step = run_runtime_step(executor, mode, occurrence, ctx, compiled,
                                        support_candidates, bootstrap=bootstrap)
                ctx.graph_bootstrap_completed = True
                if step.automation_request:
                    step = run_runtime_step(executor, mode, occurrence, ctx, compiled,
                                            support_candidates, draft_request=step.automation_request)
                if step.failure_code:
                    return executor.not_started(occurrence, failure_code=step.failure_code)
                if step.result is not None:
                    if step.result.atomic_effect_passed:
                        executor._mark_runtime_trial_parent_completed(ctx, occurrence)
                        return step.result
                    if step.result.started:
                        mode = "seeded"
                    if ctx.benchmark_terminal():
                        return step.result
        except AtomicSkillGraphError as exc:
            if exc.layer == FailureLayer.INFRASTRUCTURE:
                raise
            result = executor.not_started(occurrence, failure_code=exc.code)
            result.failure_layer = exc.layer.value
            return result
