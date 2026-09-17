"""One fresh Runtime decision, shared by Preparation and Seeded breakpoints.

The caller owns the execution loop. A semantic reply is always finalized here;
no provider conversation is reused to obtain the next decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.protocol import NativeToolSpec
from ..core.serialization import to_primitive
from ..tooling.runtime_interface import build_runtime_automation_interface
from .checkpoint import increment, metrics
from .loop_guard import ActionLoopGuard
from .negative_memory import cached_rejection, current_rejections, remember_rejection


@dataclass
class RuntimeStepResult:
    result: Any = None
    failure_code: str = ""
    automation_request: dict[str, Any] | None = None


def automation_request_tool() -> NativeToolSpec:
    return NativeToolSpec(
        "request_runtime_automation",
        "Request an independent Atomic draft step for systematic or repeated work. "
        "This request does not create a Tool or assert any effect.",
        {"type": "object", "properties": {
            "reason": {"type": "string"},
            "intended_capability": {"type": "string"},
        }, "required": ["reason", "intended_capability"], "additionalProperties": False},
    )


def run_runtime_step(executor: Any, mode: str, occurrence: Any, ctx: Any,
                     invocations: list[Any], support_candidates: list[Any],
                     *, bootstrap: bool = False, draft_request: dict | None = None) -> RuntimeStepResult:
    if mode not in {"preparation", "seeded"}:
        raise ValueError(f"unsupported RuntimeStep mode: {mode}")
    session = executor.session_factory(f"runtime_step_{mode}", occurrence.occurrence_id)
    record = executor._record_session_start(
        session, "RuntimePreparationSession" if mode == "preparation" else "SeededSession",
        occurrence.occurrence_id, ctx,
    )
    span = ctx.trace_builder.start_span(f"runtime_{mode}", occurrence.occurrence_id)
    atomic = executor.invocation_compiler.skills.get_atomic(occurrence.node_ref)
    state = executor._activate_occurrence_state(occurrence, atomic, invocations, ctx)
    state = {**state, "last_step_feedback": ctx.runtime_step_feedback.get(occurrence.occurrence_id, {})}
    bindings = ctx.binding_store.runtime_prompt_projection(occurrence, atomic.inputs)
    audit: dict[str, Any] = {}
    query = getattr(executor, "runtime_resources", None)
    resources = query(ctx.budget.current_occurrence_id or occurrence.occurrence_id) if query else {}
    frame = {"current_step_id": occurrence.step_id, "current_occurrence_id": occurrence.occurrence_id,
             "mode": mode, "current_atomic_ref": str(occurrence.node_ref),
             "repeat": ctx.binding_store.repeat_execution_frame(occurrence.step_id),
             "completed_step_ids": [node.step_id for node in ctx.trace_builder.trace.node_records
                 if node.status.value in {"direct_autonomous_success", "direct_agent_prepared_success", "seeded_success", "already_satisfied"}],
             "last_step": ctx.runtime_step_feedback.get(occurrence.occurrence_id, {}),
             "remaining_resources": {**resources, "node_actions": ctx.budget.remaining_node_actions,
                                     "task_actions": ctx.budget.remaining_global_actions}}
    prompt = executor.context_builder.runtime_node(
        task_goal=ctx.task_goal, atomic_contract=atomic,
        task_semantic_context=bindings["task_semantic_context"],
        current_occurrence_semantic_anchors=bindings["occurrence_semantic_anchors"],
        execution_ready_bindings=bindings["execution_ready_bindings"],
        missing_or_insufficient_bindings=bindings["missing_or_insufficient_bindings"],
        observation=ctx.observation, action_catalog=ctx.action_catalog,
        relevant_action_history=ctx.relevant_history(occurrence.occurrence_id),
        remaining_budget=ctx.budget.snapshot(),
        implementation_invocations=[item.spec for item in invocations],
        downstream_plan_context=state["downstream_obligations"],
        current_state_snapshot=state,
        exploration_memory=ctx.exploration_memory.policy_view(),
        support_atomic_candidates=support_candidates,
        runtime_automation_drafts=list(ctx.runtime_automation_drafts.values()),
        runtime_automation_interface=(build_runtime_automation_interface(
            ctx.harness, occurrence, ctx.binding_store) if draft_request else None),
        recent_failed_learned_invocation=ctx.last_failed_invocation,
        projection_audit=audit,
        runtime_step_mode=mode,
        rejected_candidates=[*sorted(ctx.rejected_runtime_implementations.get(occurrence.occurrence_id, set())),
                             *current_rejections(ctx, occurrence)],
        execution_frame=frame,
    )
    instruction = (
        "\nR10: This is a fresh one-decision RuntimeStep. Return exactly one native call. "
        "The graph executor will inspect authoritative state after this step. "
        "Use request_runtime_automation to request the separately loaded automation DSL."
    )
    if draft_request:
        import json
        instruction += "\nAUTOMATION_REQUEST\n" + json.dumps(draft_request, ensure_ascii=False)
        tools = [executor._automation_tool()]
    else:
        tools = [tool for tool in executor._node_tools(
            ctx, atomic, invocations=invocations, allow_plan_conflict=True,
            support_candidates=support_candidates,
        ) if tool.name != "propose_runtime_automation_atomic"]
        tools.append(automation_request_tool())
    prompt = instruction + "\n" + prompt
    increment(ctx, "runtime_step_count")
    increment(ctx, f"runtime_step_{mode}_count")
    if bootstrap:
        increment(ctx, "graph_bootstrap_agent_step_count")
    if draft_request:
        increment(ctx, "runtime_automation_draft_step_count")
        increment(ctx, "runtime_automation_interface_load_count")
    outcome = RuntimeStepResult()
    try:
        executor._record_runtime_context_projection(
            ctx, audit, session_id=session.session_id,
            occurrence_id=occurrence.occurrence_id, origin="runtime_step",
        )
        turn = session.next_turn(prompt, tools=tools)
        executor._record_turn(session, turn, ctx)
        call = turn.tool_calls[0]
        selected_action = next((item for item in ctx.action_catalog
            if call.name == 'environment_action' and item.action_id == call.arguments.get('action_id')
            and item.revision == ctx.world_revision), None)
        cached = cached_rejection(ctx, occurrence, call)
        if cached is not None:
            payload = cached
            executor._record_control_call(call, session, occurrence, ctx,
                                         call_kind="runtime_cached_rejection", result=payload)
        elif call.name == "request_runtime_automation":
            increment(ctx, "runtime_automation_request_count")
            outcome.automation_request = dict(call.arguments)
            payload = {"accepted": True, "next": "RuntimeAutomationDraftStep"}
            executor._record_control_call(call, session, occurrence, ctx,
                                         call_kind="runtime_automation_request", result=payload)
        elif call.name == "propose_runtime_automation_atomic":
            payload = executor._process_runtime_automation_call(call, ctx, occurrence)
            executor._record_control_call(call, session, occurrence, ctx,
                                         call_kind="runtime_automation_draft", result=payload)
        elif call.name == "report_runtime_status":
            status, payload = executor._status_result(call, session, occurrence, ctx)
            outcome.failure_code = ("runtime_plan_conflict" if status == "plan_conflict"
                                    else "runtime_binding_unresolved")
        elif call.name == "validate_current_atomic":
            outcome.result, payload = executor._validate_current_atomic_call(
                call, session, occurrence, ctx, mode=mode, atomic=atomic,
            )
        elif call.name == "environment_action":
            # The guard is occurrence-scoped across fresh sessions.
            guards = getattr(ctx, "_runtime_step_loop_guards", None)
            if guards is None:
                guards = ctx._runtime_step_loop_guards = {}
            guard = guards.setdefault(occurrence.occurrence_id, ActionLoopGuard())
            payload, spec = executor._execute_environment_call(
                call, session, occurrence, ctx, span_id=span.span_id,
                origin=f"runtime_{mode}", loop_guard=guard, atomic=atomic,
            )
            if call.arguments["intent"] == "attempt_current_atomic" and payload.get("accepted"):
                outcome.result = executor._complete_from_current_effect(
                    occurrence, ctx, mode=mode,
                    preferred_values=executor._environment_effect_preferences(occurrence, ctx, spec),
                )
            if payload.get("fallback_required"):
                outcome.failure_code = "runtime_action_loop_blocked"
        elif call.name == "invoke_support_atomic":
            increment(ctx, "support_agent_selected_count")
            payload = executor._invoke_support_atomic_call(
                call, session, occurrence, ctx, atomic, support_candidates, None,
            )
        else:
            compiled = next(item for item in invocations if item.spec.name == call.name)
            prepared = executor.invocation_compiler.prepare_arguments(
                compiled, call_name=call.name, call_id=call.call_id, arguments=call.arguments,
                occurrence=occurrence, binding_store=ctx.binding_store,
                evidence_store=ctx.evidence_store, revision=ctx.world_revision,
                task_contract=ctx.task_contract,
            )
            if prepared.passed:
                outcome.result = executor._complete_from_current_effect(
                    occurrence, ctx, mode=mode,
                    preferred_values=list(prepared.normalized_arguments.values()),
                    provisional_bindings=prepared.binding_updates,
                )
                preflight = executor.invocation_compiler.validate_execution_context(
                    compiled, prepared, occurrence=occurrence,
                    binding_store=ctx.binding_store, evidence_store=ctx.evidence_store,
                    revision=ctx.world_revision,
                ) if outcome.result is None else prepared
            else:
                preflight = prepared
            if outcome.result is None and preflight.passed:
                from .node_gate import precondition_gate
                gate = precondition_gate(executor, occurrence, compiled, ctx)
                if not gate.ready:
                    from ..core.results import ToolCallPreflightResult
                    preflight = ToolCallPreflightResult(False, str(compiled.implementation.ref),
                        failure_layer="runtime_binding", failure_code=gate.reason,
                        message="Atomic preconditions lack current authoritative evidence")
            if outcome.result is None and preflight.passed:
                from .invocation_transaction import execute_invocation
                outcome.result = execute_invocation(executor.implementation_runner,
                    compiled, preflight, occurrence, ctx, agent_prepared=True,
                )
                if mode == "seeded" and outcome.result.atomic_effect_passed:
                    from ..core.results import NodeExecutionStatus
                    outcome.result.node_status = NodeExecutionStatus.SEEDED_SUCCESS
            payload = to_primitive(outcome.result or preflight)
            executor._record_control_call(call, session, occurrence, ctx,
                                         call_kind="implementation_invocation", result=payload)
            if not preflight.passed:
                ctx.record_failed_invocation(
                    occurrence_id=occurrence.occurrence_id,
                    implementation_ref=str(compiled.implementation.ref),
                    failure_code=preflight.failure_code, message=preflight.message,
                )
        remember_rejection(ctx, occurrence, call, payload)
        # Carry finite public feedback, never an assistant conversation or
        # Tool body, to the next fresh decision. Rollback does not erase it.
        ctx.runtime_step_feedback[occurrence.occurrence_id] = {
            "tool": call.name,
            "arguments": to_primitive(call.arguments),
            **({'action_type': selected_action.action_type, 'action_arguments': to_primitive(selected_action.arguments)}
               if selected_action is not None else {}),
            **{key: payload[key] for key in (
                "accepted", "passed", "error", "message", "failure_code", "stage",
                "r0_passed", "static_passed", "r1_passed", "preflight_failure_code",
                "started", "failure_layer", "cached_rejection", "rollback", "trial_failure",
            ) if isinstance(payload, dict) and key in payload},
        }
        if isinstance(payload, dict) and isinstance(payload.get("validation"), dict):
            ctx.runtime_step_feedback[occurrence.occurrence_id].update({
                key: payload["validation"][key] for key in ("failure_code", "message")
                if payload["validation"].get(key)
            })
        executor._finalize_tool_result(session, call.call_id, payload, tools)
        return outcome
    finally:
        ctx.trace_builder.finish_span(span.span_id)
        executor._finish_session(record, session, ctx)
        turns = [item for item in ctx.trace_builder.trace.agent_turns if item.session_id == session.session_id]
        ctx.trace_builder.trace.metadata.setdefault("runtime_steps", []).append({
            "session_id": session.session_id, "occurrence_id": occurrence.occurrence_id,
            "mode": mode, "bootstrap": bootstrap, "draft": draft_request is not None,
            "accepted_semantic_turn_count": len(turns),
        })
        values = metrics(ctx)
        values["runtime_step_max_semantic_turns"] = max(
            int(values.get("runtime_step_max_semantic_turns", 0)),
            sum(1 for item in ctx.trace_builder.trace.agent_turns if item.session_id == session.session_id),
        )
