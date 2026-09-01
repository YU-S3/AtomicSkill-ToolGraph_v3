"""Strict occurrence-order task execution; Runtime never redesigns the plan."""

from __future__ import annotations

from typing import Any

from ..core.contracts import ColdStartCandidateSource
from ..core.edges import GraphEdge, GraphEdgeType
from ..core.refs import SkillRef
from ..core.results import (
    NodeExecutionStatus,
    RuntimeLinearPlan,
    RuntimeOccurrence,
)
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode
from ..harness.protocol import HarnessAdapter, HarnessTask
from ..planner.pipeline import PlannerPipeline
from ..traces.schema import (
    ColdStartPlanRecord,
    ColdStartStepRecord,
    NodeTraceRecord,
    TaskRecord,
    TraceBuilder,
    TraceRecord,
    ValidationRecord,
)
from ..validation.engine import ValidationEngine
from .budget import RuntimeBudget
from .invocation_compiler import InvocationCompiler
from .node_executor import NodeExecutor
from .plan_context import RuntimePlanContextBuilder
from .task_context import TaskRuntimeContext
from .cold_start_executor import ProvisionalNodeExecutor, provisional_atomic_view


def refresh_learning_eligibility(trace: TraceRecord) -> None:
    trace.learning_eligible = bool(
        trace.strict_task_success
        and trace.resource_usage_complete
        and not trace.infrastructure_failure
    )


def apply_terminal_outcome(
    trace: TraceRecord,
    terminal_result: Any,
    validator_channel: Any,
) -> None:
    """Apply the one authoritative task-outcome definition to every route."""

    trace.benchmark_success = bool(getattr(validator_channel, "won", False))
    trace.task_contract_success = bool(
        dict(getattr(terminal_result, "checks", {}) or {}).get(
            "task_contract", False,
        )
    )
    trace.strict_task_success = bool(
        trace.benchmark_success and trace.task_contract_success
    )
    refresh_learning_eligibility(trace)


class RuntimeOrchestrator:
    def __init__(
        self, planner: PlannerPipeline, harness: HarnessAdapter,
        invocation_compiler: InvocationCompiler, validation: ValidationEngine,
        session_factory: Any, *, runtime_config: dict[str, Any] | None = None,
        failure_knowledge: Any | None = None,
    ) -> None:
        self.planner, self.harness = planner, harness
        self.invocation_compiler, self.validation = invocation_compiler, validation
        self.session_factory = session_factory
        self.runtime_config = runtime_config or {}
        self.node_executor = NodeExecutor(invocation_compiler, validation, session_factory)
        self.plan_context_builder = RuntimePlanContextBuilder(
            invocation_compiler.skills
        )
        # NodeExecutor owns prompt construction; both components share the
        # same narrow formal-plan interpreter and no benchmark/task policy.
        self.node_executor.plan_context_builder = self.plan_context_builder
        self.failure_knowledge = failure_knowledge
        self.provisional_node_executor = ProvisionalNodeExecutor(self.node_executor)

    def _budget(self) -> RuntimeBudget:
        config = self.runtime_config
        return RuntimeBudget(
            global_action_budget=int(config.get("global_action_budget", 100)),
            node_action_budget=int(config.get("node_action_budget", 35)),
            token_limits=dict(config.get("token_limits", {})), turn_limits=dict(config.get("turn_limits", {})),
        )

    def create_trace_builder(self, task: HarnessTask, *, attempt_id: str = "") -> TraceBuilder:
        """Create the immutable-at-finalization skeleton before Planner/API work."""

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
        trace = TraceRecord.create(
            task_record,
            {},
            {},
            {"source": "not_started", "failure_stage": ""},
        )
        if attempt_id:
            trace.metadata["attempt_id"] = attempt_id
        return TraceBuilder(trace)

    def _create_context(
        self,
        task: HarnessTask,
        plan: RuntimeLinearPlan,
        builder: TraceBuilder | None = None,
    ) -> TaskRuntimeContext:
        builder = builder or self.create_trace_builder(task)
        if builder.trace.task.task_id != task.task_id:
            raise ValueError("TraceBuilder task identity does not match Runtime task")
        builder.trace.task_contract = to_primitive(plan.task_contract)
        builder.trace.planner_audit = to_primitive(plan.planner_audit)
        builder.trace.runtime_plan = {
            **to_primitive(plan),
            "failure_stage": "runtime",
        }
        audit = dict(plan.planner_audit or {})
        builder.trace.requirement_bundle = dict(
            audit.get("requirements_p1r")
            or audit.get("requirements_p1")
            or {}
        )
        builder.trace.requirement_expansion = dict(
            audit.get("requirement_expansion") or {}
        )
        if plan.cold_start_plan is not None:
            validation = dict(
                audit.get("cold_start_repair_validation")
                or audit.get("cold_start_validation")
                or {}
            )
            scaffold = dict(plan.cold_start_scaffold or {})
            builder.trace.cold_start_plan = ColdStartPlanRecord(
                plan_id=plan.cold_start_plan.plan_id,
                proposal=to_primitive(plan.cold_start_plan),
                validation=validation,
                repair_used=bool(audit.get("cold_start_repair")),
                executable_step_ids=list(
                    scaffold.get("executable_step_ids", ())
                ),
                first_unresolved_step_id=str(
                    scaffold.get("first_unresolved_step_id", "")
                ),
            )
        return TaskRuntimeContext.create(task, plan, self.harness, builder, self._budget())

    def run_task(
        self,
        task: HarnessTask,
        *,
        mode: RuntimeMode | str = RuntimeMode.ONLINE,
        trace_builder: TraceBuilder | None = None,
        attempt_id: str = "",
    ) -> TraceRecord:
        builder = trace_builder or self.create_trace_builder(task, attempt_id=attempt_id)
        builder.trace.runtime_plan["failure_stage"] = "planner"
        initial_observation = str(task.context.get("initial_observation", ""))
        plan = self.planner.build_plan(task, self.harness, mode=mode, initial_observation=initial_observation)
        ctx = self._create_context(task, plan, builder)

        initial_terminal = self.validation.task.terminal(
            ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
        )
        if plan.source == "full_dynamic":
            cold_continuation = plan.cold_start_plan is not None
            result = self.node_executor.run_dynamic(
                ctx,
                cold_start_continuation=cold_continuation,
                continuation_context=(
                    self._cold_continuation_context(ctx, [])
                    if cold_continuation else None
                ),
            )
            trace = ctx.trace_builder.trace
            trace.node_contract_success = False
            trace.graph_self_sufficient_success = False
            trace.graph_full_completion = False
            trace.metadata["dynamic_result"] = result
            trace.infrastructure_failure = result.get("failure_code") in {
                "infrastructure_failure", "llm_error",
            }
            terminal = self.validation.task.terminal(
                ctx.task_contract,
                ctx.harness.validator_channel(),
                bool(getattr(ctx.harness.validator_channel(), "won", False)),
            )
            apply_terminal_outcome(
                trace, terminal, ctx.harness.validator_channel(),
            )
            trace.runtime_plan["failure_stage"] = ""
            ctx.task_progress.record("task_terminal")
            return ctx.trace_builder.finish()

        if plan.source == "cold_start":
            return self._run_cold_start(ctx, mode=mode)

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
                already = self.node_executor._complete_from_current_effect(
                    occurrence,
                    ctx,
                    mode="entry",
                    preferred_values=[],
                )
                if already is not None:
                    outputs = dict(already.validated_outputs)
                    node.status = NodeExecutionStatus.ALREADY_SATISFIED
                    node.validated_outputs = outputs
                    if outputs:
                        validation_refs = self._latest_atomic_witnesses(
                            ctx,
                            occurrence.occurrence_id,
                        )
                        ctx.binding_store.publish_validated_outputs(
                            occurrence,
                            outputs,
                            validation_refs,
                            ctx.world_revision,
                        )
                        ctx.validated_outputs[occurrence.occurrence_id] = outputs
                        for role, value in outputs.items():
                            ctx.evidence_store.add_validated_tool_output(
                                role,
                                value,
                                validation_refs,
                            )
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
                    if direct.failure_code == "runtime_plan_conflict":
                        node.status = direct.node_status
                        node.failure = {
                            "failure_layer": direct.failure_layer or "composite",
                            "failure_code": "runtime_plan_conflict",
                            "direct_started": direct.started,
                        }
                        ctx.plan_conflict_declared = True
                        ctx.plan_conflict_context = self._plan_conflict_context(
                            ctx, occurrence,
                        )
                        ctx.trace_builder.trace.metadata.setdefault(
                            "runtime_plan_conflicts", []
                        ).append(dict(ctx.plan_conflict_context))
                        break
                    seeded = self.node_executor.run_seeded_fresh(occurrence, ctx)
                    node.seeded_result = to_primitive(seeded)
                    final = seeded
                    if not seeded.atomic_effect_passed:
                        node.status = NodeExecutionStatus.SEEDED_FAILED
                        node.failure = {
                            "failure_layer": seeded.failure_layer, "failure_code": seeded.failure_code,
                            "direct_started": direct.started,
                        }
                        ctx.plan_execution_failed = True
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
        if ctx.rescue_allowed() and not terminal.passed:
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
            if ctx.plan_conflict_declared:
                conflicts = ctx.trace_builder.trace.metadata.get(
                    "runtime_plan_conflicts", []
                )
                if conflicts:
                    conflicts[-1]["rescue_attempted"] = True
                    conflicts[-1]["rescue_strict_success"] = bool(
                        getattr(ctx.harness.validator_channel(), "won", False)
                        and dict(getattr(terminal, "checks", {}) or {}).get(
                            "task_contract", False
                        )
                    )

        trace = ctx.trace_builder.trace
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
        apply_terminal_outcome(
            trace, terminal, ctx.harness.validator_channel(),
        )
        if trace.benchmark_success and not trace.task_contract_success:
            trace.metadata["anomaly"] = "benchmark_goal_contract_mismatch"
        trace.metadata["invocation_compile_rejections"] = list(self.invocation_compiler.compile_rejections)
        trace.runtime_plan["failure_stage"] = ""
        return ctx.trace_builder.finish()

    def _plan_conflict_context(
        self,
        ctx: TaskRuntimeContext,
        occurrence: RuntimeOccurrence,
        *,
        execution_plan: RuntimeLinearPlan | None = None,
    ) -> dict[str, Any]:
        """Record only formal/public diagnostics for an Agent declaration."""

        atomic = self.invocation_compiler.skills.get_atomic(
            occurrence.node_ref
        )
        projection = ctx.binding_store.runtime_prompt_projection(
            occurrence, atomic.inputs,
        )
        last_preflight_failure_code = ""
        detail = ""
        for call in reversed(ctx.trace_builder.trace.native_tool_calls):
            if call.occurrence_id != occurrence.occurrence_id:
                continue
            if (
                call.tool_name == "report_runtime_status"
                and call.arguments.get("status") == "plan_conflict"
            ):
                detail = str(call.arguments.get("detail") or "")
                continue
            result = dict(call.preflight_result or {})
            validation = dict(result.get("validation") or {})
            code = str(
                result.get("failure_code")
                or result.get("error")
                or validation.get("failure_code")
                or ""
            )
            if code:
                last_preflight_failure_code = code
                break
        if not last_preflight_failure_code:
            for validation_record in reversed(
                ctx.trace_builder.trace.validations
            ):
                if validation_record.occurrence_id != occurrence.occurrence_id:
                    continue
                result = dict(validation_record.result or {})
                if bool(result.get("passed", False)):
                    continue
                code = str(result.get("failure_code") or "")
                if code:
                    last_preflight_failure_code = code
                    break
        policy_context = self.plan_context_builder.build(
            execution_plan or ctx.plan,
            occurrence.step_id,
            ctx.binding_store,
        ).policy_view()
        progress = ctx.task_progress.snapshot()
        record = {
            "occurrence_id": occurrence.occurrence_id,
            "atomic_ref": str(occurrence.node_ref),
            "semantic_anchors": dict(
                projection["occurrence_semantic_anchors"]
            ),
            "last_preflight_failure_code": last_preflight_failure_code,
            "world_revision": ctx.world_revision,
            "task_progress_digest": progress.progress_digest,
            "conflict_code": "runtime_plan_conflict",
            "conflict_step_summary": atomic.summary,
            "remaining_method_outline": list(
                policy_context.get("remaining_method_outline", ())
            ),
        }
        if detail:
            record["detail"] = detail
        return record

    @staticmethod
    def _cold_edge(value: Any) -> GraphEdge:
        return GraphEdge(
            value.edge_id,
            GraphEdgeType(value.edge_type),
            value.source_step,
            value.target_step,
            value.source_role,
            value.target_role,
            value.origin,
            value.existing_edge_id,
            (),
        )

    def _materialize_cold_execution_plan(
        self,
        ctx: TaskRuntimeContext,
        *,
        mode: RuntimeMode | str,
    ) -> tuple[RuntimeLinearPlan, dict[str, Any]]:
        """Build the task-local executable view of an admitted C1 plan.

        This view is deliberately never registered as a Composite.  Verified
        steps retain their normal Skill/Implementation identities; a
        Provisional step is represented only by an ephemeral Atomic contract
        and therefore cannot acquire a learned invocation surface.
        """

        proposal = ctx.plan.cold_start_plan
        if proposal is None:
            raise RuntimeError("cold-start Runtime plan has no admitted C1 proposal")
        occurrences: list[RuntimeOccurrence] = []
        provisionals: dict[str, Any] = {}
        for step in proposal.steps:
            if step.candidate_source is ColdStartCandidateSource.UNRESOLVED:
                continue
            if step.candidate_source is ColdStartCandidateSource.VERIFIED:
                atomic_ref = SkillRef.parse(step.candidate_ref)
                atomic = self.invocation_compiler.skills.get_atomic(atomic_ref)
                implementations = self.invocation_compiler.skills.implementations_for(
                    atomic_ref,
                    mode=mode,
                )
                implementation_refs = [item.ref for item in implementations]
            else:
                if self.failure_knowledge is None:
                    raise RuntimeError(
                        "cold-start Provisional execution requires the isolated "
                        "failure-side store"
                    )
                provisional = self.failure_knowledge.get_provisional(
                    step.candidate_ref,
                )
                provisionals[step.step_id] = provisional
                atomic = provisional_atomic_view(provisional)
                atomic_ref = atomic.ref
                # The empty list is an executable isolation boundary: the
                # Provisional Atomic may never be compiled into a learned call.
                implementation_refs = []
            occurrences.append(RuntimeOccurrence(
                step_id=step.step_id,
                occurrence_id=f"cold::{step.step_id}",
                node_ref=atomic_ref,
                requirement_ids=list(step.requirement_instance_ids),
                binding_specs=dict(step.binding_specs),
                implementation_candidates=implementation_refs,
                expected_effects=list(atomic.effects),
                requirement_instance_ids=list(step.requirement_instance_ids),
                repeat_role_bindings=dict(step.repeat_role_bindings),
            ))
        known_steps = {item.step_id for item in occurrences}
        data_edges = [
            self._cold_edge(item)
            for item in proposal.data_edges
            if item.source_step in known_steps and item.target_step in known_steps
        ]
        dependency_edges = [
            self._cold_edge(item)
            for item in proposal.dependency_edges
            if item.source_step in known_steps and item.target_step in known_steps
        ]
        execution_plan = RuntimeLinearPlan(
            task_id=ctx.plan.task_id,
            source="cold_start_scaffold",
            source_composite_ref=None,
            occurrences=occurrences,
            control_sequence=[
                step_id for step_id in proposal.control_sequence
                if step_id in known_steps
            ],
            data_edges=data_edges,
            dependency_edges=dependency_edges,
            task_contract=ctx.task_contract,
            planner_audit=dict(ctx.plan.planner_audit),
            repeat_constraints=list(ctx.plan.repeat_constraints),
            cold_start_plan=proposal,
            cold_start_scaffold=dict(ctx.plan.cold_start_scaffold),
        )
        return execution_plan, provisionals

    def _publish_cold_outputs(
        self,
        ctx: TaskRuntimeContext,
        occurrence: RuntimeOccurrence,
        outputs: dict[str, Any],
    ) -> None:
        if not outputs:
            return
        witness_refs = self._latest_atomic_witnesses(
            ctx,
            occurrence.occurrence_id,
        )
        ctx.binding_store.publish_validated_outputs(
            occurrence,
            outputs,
            witness_refs,
            ctx.world_revision,
        )
        ctx.validated_outputs[occurrence.occurrence_id] = dict(outputs)
        for role, value in outputs.items():
            ctx.evidence_store.add_validated_tool_output(
                role,
                value,
                witness_refs,
            )

    def _run_verified_cold_step(
        self,
        ctx: TaskRuntimeContext,
        execution_plan: RuntimeLinearPlan,
        occurrence: RuntimeOccurrence,
    ) -> tuple[bool, str, str]:
        """Run one Verified scaffold step through the unchanged node ladder."""

        node = ctx.trace_builder.start_node(
            occurrence.occurrence_id,
            occurrence.step_id,
            str(occurrence.node_ref),
        )
        ctx.binding_store.apply_data_flow(
            execution_plan,
            occurrence.step_id,
            ctx.validated_outputs,
            revision=ctx.world_revision,
        )
        ctx.binding_store.resolve_occurrence_specs(
            occurrence,
            ctx.world_revision,
        )
        already = self.node_executor._complete_from_current_effect(
            occurrence,
            ctx,
            mode="entry",
            preferred_values=[],
        )
        if already is not None:
            node.status = NodeExecutionStatus.ALREADY_SATISFIED
            node.validated_outputs = dict(already.validated_outputs)
            self._publish_cold_outputs(
                ctx,
                occurrence,
                already.validated_outputs,
            )
            return True, "", "already_satisfied"

        invocations = self.invocation_compiler.compile_candidates(
            occurrence,
            ctx.binding_store,
            max_candidates=int(
                self.runtime_config.get("max_implementation_candidates", 3)
            ),
            task_id=ctx.task.task_id,
        )
        if not invocations:
            direct = self.node_executor.not_started(
                occurrence,
                failure_code="no_compatible_implementation",
            )
        else:
            direct = self.node_executor.try_autonomous(
                occurrence,
                invocations,
                ctx,
            )
            if direct is None:
                direct = self.node_executor.run_preparation_session(
                    occurrence,
                    invocations,
                    ctx,
                    learned_call_repair_limit=int(
                        self.runtime_config.get(
                            "learned_toolcall_repair_limit",
                            2,
                        )
                    ),
                    plan_context_plan=execution_plan,
                )
        node.direct_result = to_primitive(direct)
        final = direct
        if not direct.atomic_effect_passed:
            if direct.failure_code == "runtime_plan_conflict":
                node.status = direct.node_status
                node.failure = {
                    "failure_layer": direct.failure_layer or "composite",
                    "failure_code": "runtime_plan_conflict",
                    "direct_started": direct.started,
                }
                ctx.plan_conflict_declared = True
                ctx.plan_conflict_context = self._plan_conflict_context(
                    ctx,
                    occurrence,
                    execution_plan=execution_plan,
                )
                ctx.trace_builder.trace.metadata.setdefault(
                    "runtime_plan_conflicts", [],
                ).append(dict(ctx.plan_conflict_context))
                return False, "runtime_plan_conflict", "plan_conflict"
            seeded = self.node_executor.run_seeded_fresh(
                occurrence,
                ctx,
                plan_context_plan=execution_plan,
            )
            node.seeded_result = to_primitive(seeded)
            final = seeded
        if not final.atomic_effect_passed:
            node.status = (
                NodeExecutionStatus.SEEDED_FAILED
                if node.seeded_result
                else NodeExecutionStatus.DIRECT_FAILED
            )
            underlying_failure = (
                final.failure_code
                or "cold_start_verified_step_failed"
            )
            node.failure = {
                "failure_layer": final.failure_layer or "atomic",
                "failure_code": underlying_failure,
                "direct_started": direct.started,
            }
            return False, "cold_start_verified_step_failed", "failed"
        node.status = final.node_status
        node.validated_outputs = dict(final.validated_outputs)
        self._publish_cold_outputs(
            ctx,
            occurrence,
            final.validated_outputs,
        )
        return True, "", "success"

    def _cold_continuation_context(
        self,
        ctx: TaskRuntimeContext,
        completed_local_effects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        proposal = ctx.plan.cold_start_plan
        completed_step_ids = {
            str(item.get("step_id", "")) for item in completed_local_effects
        }
        remaining_instances: list[str] = []
        experience_ids: list[str] = []
        if proposal is not None:
            for step_id in proposal.control_sequence:
                step = next(
                    item for item in proposal.steps if item.step_id == step_id
                )
                if step_id not in completed_step_ids:
                    remaining_instances.extend(step.requirement_instance_ids)
            experience_ids = list(
                proposal.referenced_failure_experience_ids[:2]
            )

        failure_experiences: list[dict[str, Any]] = []
        if experience_ids:
            if self.failure_knowledge is None:
                raise RuntimeError(
                    "cold-start continuation references Failure Experience "
                    "without the isolated failure-side store"
                )
            for experience_id in experience_ids:
                view = self.failure_knowledge.failure_experience_view(
                    experience_id,
                )
                sanitized = dict(to_primitive(view))
                sanitized["warning"] = (
                    "FAILED HISTORICAL METHOD — NOT AN EXECUTABLE OR "
                    "SUCCESSFUL PLAN"
                )
                failure_experiences.append(sanitized)

        return {
            "completed_local_effects": to_primitive(completed_local_effects),
            "remaining_requirement_instance_ids": list(
                dict.fromkeys(remaining_instances)
            ),
            "failure_experiences": failure_experiences,
        }

    def _finish_cold_start(
        self,
        ctx: TaskRuntimeContext,
        terminal: Any,
        *,
        dynamic_result: dict[str, Any] | None,
    ) -> TraceRecord:
        trace = ctx.trace_builder.trace
        trace.node_contract_success = bool(trace.cold_start_steps) and all(
            item.local_effect_passed for item in trace.cold_start_steps
        )
        trace.implementation_direct_success = any(
            node.status in {
                NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS,
                NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS,
            }
            for node in trace.node_records
        )
        trace.graph_full_completion = False
        # A C1 scaffold is never a registered Composite or a P2 success,
        # regardless of whether its prefix or continuation completes the task.
        trace.graph_self_sufficient_success = False
        if dynamic_result is not None:
            metadata_key = (
                "task_rescue"
                if dynamic_result.get("rescue")
                else "cold_start_dynamic_continuation"
            )
            trace.metadata[metadata_key] = dynamic_result
            if dynamic_result.get("failure_code") in {
                "infrastructure_failure",
                "llm_error",
            }:
                trace.infrastructure_failure = True
        apply_terminal_outcome(
            trace,
            terminal,
            ctx.harness.validator_channel(),
        )
        trace.cold_start_assisted_success = bool(
            trace.strict_task_success
            and any(item.local_effect_passed for item in trace.cold_start_steps)
        )
        ctx.task_progress.record("task_terminal")
        trace.runtime_plan["failure_stage"] = ""
        return ctx.trace_builder.finish()

    def _run_cold_start(
        self,
        ctx: TaskRuntimeContext,
        *,
        mode: RuntimeMode | str,
    ) -> TraceRecord:
        proposal = ctx.plan.cold_start_plan
        if proposal is None:
            raise RuntimeError("cold-start Runtime source requires a C1 proposal")
        execution_plan, provisionals = self._materialize_cold_execution_plan(
            ctx,
            mode=mode,
        )
        by_step = {item.step_id: item for item in proposal.steps}
        executable = list(
            ctx.plan.cold_start_scaffold.get("executable_step_ids", ())
        )
        completed_local_effects: list[dict[str, Any]] = []
        terminal = self.validation.task.terminal(
            ctx.task_contract,
            ctx.harness.validator_channel(),
            bool(getattr(ctx.harness.validator_channel(), "won", False)),
        )
        if terminal.passed:
            return self._finish_cold_start(
                ctx,
                terminal,
                dynamic_result=None,
            )

        for index, step_id in enumerate(executable):
            ctx.current_step_index = index
            step = by_step[step_id]
            occurrence = execution_plan.occurrence(step_id)
            ctx.budget.begin_node(occurrence.occurrence_id)
            before = ctx.task_progress.record("cold_start_step_start")
            action_start = len(ctx.trace_builder.trace.environment_actions)
            failure_code = ""
            outcome = "failed"
            if step.candidate_source is ColdStartCandidateSource.VERIFIED:
                local_effect_passed, failure_code, outcome = (
                    self._run_verified_cold_step(
                        ctx,
                        execution_plan,
                        occurrence,
                    )
                )
            elif step.candidate_source is ColdStartCandidateSource.PROVISIONAL:
                ctx.binding_store.apply_data_flow(
                    execution_plan,
                    step_id,
                    ctx.validated_outputs,
                    revision=ctx.world_revision,
                )
                trial = self.provisional_node_executor.execute(
                    provisionals[step_id],
                    ctx,
                    step,
                    progress_tracker=ctx.task_progress,
                )
                local_effect_passed = trial.local_effect_passed
                failure_code = trial.failure_code
                outcome = "success" if local_effect_passed else "failed"
                ctx.trace_builder.trace.metadata.setdefault(
                    "provisional_trials",
                    [],
                ).append(to_primitive(trial))
            else:
                # The Scaffold is code-derived and cannot contain an
                # unresolved step.  Treat corrupted IR as a program error.
                raise RuntimeError(
                    "cold-start Scaffold contains an unresolved step"
                )

            after = ctx.task_progress.record("cold_start_step_complete")
            action_end = len(ctx.trace_builder.trace.environment_actions)
            ctx.trace_builder.trace.cold_start_steps.append(
                ColdStartStepRecord(
                    step_id=step.step_id,
                    candidate_source=step.candidate_source.value,
                    candidate_ref=step.candidate_ref,
                    execution_mode=step.execution_mode.value,
                    outcome=outcome,
                    local_effect_passed=local_effect_passed,
                    action_start=action_start,
                    action_end=action_end,
                    progress_before=before.progress_digest,
                    progress_after=after.progress_digest,
                    failure_code=failure_code,
                )
            )
            if not local_effect_passed:
                break
            completed_local_effects.append({
                "step_id": step.step_id,
                "candidate_source": step.candidate_source.value,
                "candidate_ref": step.candidate_ref,
                "requirement_instance_ids": list(
                    step.requirement_instance_ids
                ),
                "validated_effects": to_primitive(
                    occurrence.expected_effects
                ),
                "progress_after": after.progress_digest,
            })
            terminal = self.validation.task.terminal(
                ctx.task_contract,
                ctx.harness.validator_channel(),
                bool(getattr(ctx.harness.validator_channel(), "won", False)),
            )
            if terminal.passed:
                return self._finish_cold_start(
                    ctx,
                    terminal,
                    dynamic_result=None,
                )

        # No unresolved suffix is executed out of order.  The continuation is
        # a fresh task-level Agent Session over the current real environment.
        ctx.budget.end_node()
        if getattr(ctx, "plan_conflict_declared", False):
            ctx.task_rescue_used = True
            ctx.trace_builder.trace.task_rescue_required = True
            dynamic_result = self.node_executor.run_dynamic(ctx, rescue=True)
        else:
            dynamic_result = self.node_executor.run_dynamic(
                ctx,
                cold_start_continuation=True,
                continuation_context=self._cold_continuation_context(
                    ctx,
                    completed_local_effects,
                ),
            )
        terminal = self.validation.task.terminal(
            ctx.task_contract,
            ctx.harness.validator_channel(),
            bool(getattr(ctx.harness.validator_channel(), "won", False)),
        )
        if getattr(ctx, "plan_conflict_declared", False):
            conflicts = ctx.trace_builder.trace.metadata.get(
                "runtime_plan_conflicts", [],
            )
            if conflicts:
                conflicts[-1]["rescue_attempted"] = True
                conflicts[-1]["rescue_strict_success"] = bool(
                    getattr(ctx.harness.validator_channel(), "won", False)
                    and dict(getattr(terminal, "checks", {}) or {}).get(
                        "task_contract", False,
                    )
                )
        return self._finish_cold_start(
            ctx,
            terminal,
            dynamic_result=dynamic_result,
        )

    def _latest_atomic_witnesses(self, ctx: TaskRuntimeContext, occurrence_id: str) -> list[str]:
        for record in reversed(ctx.trace_builder.trace.validations):
            if (
                record.occurrence_id == occurrence_id
                and record.level in {"atomic", "already_satisfied"}
            ):
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
