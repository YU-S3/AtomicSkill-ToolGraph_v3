"""One explicit-dataflow entry, then uninterrupted Agent node execution."""
from ..core.errors import AtomicSkillGraphError, FailureLayer
from .checkpoint import increment


class VerifiedCompositeExecutor:
    def __init__(self, node_executor):
        self.nodes = node_executor

    def run_occurrence(self, occurrence, ctx):
        ex = self.nodes
        try:
            completed = ex._complete_from_current_effect(occurrence, ctx, mode="entry", preferred_values=[])
            if completed is not None:
                return completed
            if ctx.execution_terminal():
                return ex._runtime_automation_terminal_boundary(occurrence)
            invocations = ex.invocation_compiler.compile_candidates(
                occurrence, ctx.binding_store,
                task_id=ctx.task_id)
            mode = "preparation" if invocations else "seeded"
            bootstrap = not ctx.graph_bootstrap_completed
            ctx.graph_bootstrap_completed = True
            if bootstrap:
                ctx.trace_builder.trace.metadata.setdefault("node_entry_checks", []).append({
                    "occurrence_id": occurrence.occurrence_id, "revision": ctx.world_revision,
                    "reason": "graph_bootstrap", "selected": None, "candidates": []})
            if not bootstrap:
                increment(ctx, "composite_auto_gate_attempt_count")
                result = ex.try_autonomous(occurrence, invocations, ctx)
                if result is not None:
                    if result.atomic_effect_passed:
                        increment(ctx, "composite_auto_node_count")
                        increment(ctx, "post_bootstrap_llm_free_node_count")
                        return result
                    if ctx.execution_terminal():
                        return result
                    ctx.record_failed_invocation(occurrence_id=occurrence.occurrence_id,
                        implementation_ref=result.implementation_ref,
                        failure_code=result.failure_code, message=result.failure_code)
                    mode = "seeded"
            increment(ctx, "composite_breakpoint_count")
            return ex.run_agent_node(occurrence, ctx, mode=mode, bootstrap=bootstrap)
        except AtomicSkillGraphError as exc:
            if exc.layer == FailureLayer.INFRASTRUCTURE:
                raise
            result = ex.not_started(occurrence, failure_code=exc.code)
            result.failure_layer = exc.layer.value
            return result
