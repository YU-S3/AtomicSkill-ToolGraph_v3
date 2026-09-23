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
from ..core.errors import BudgetExhausted
from ..core.results import NodeExecutionStatus


def completed_step_ids_for_policy(node_records):
    complete = {NodeExecutionStatus.ALREADY_SATISFIED, NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS,
        NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS, NodeExecutionStatus.AGENT_COMPLETED_BEFORE_INVOCATION,
        NodeExecutionStatus.SEEDED_SUCCESS}
    return list(dict.fromkeys(n.step_id for n in node_records if n.status in complete))


def record_interrupted_call(executor, call, session, occurrence, ctx, error, starts):
    if call is None or any(r.call_id == call.call_id and r.session_id == session.session_id
                           for r in ctx.trace_builder.trace.native_tool_calls):
        return
    refs = {key: [r.attempt_id for r in getattr(ctx.trace_builder.trace, key)[start:]]
            for key, start in starts.items()}
    executor._record_control_call(call, session, occurrence, ctx, call_kind='runtime_budget_interrupted',
        result={'interrupted_by_budget': True, 'failure_code': error.code,
                'failure_layer': error.layer.value, 'attempt_refs': refs})


@dataclass
class RuntimeStepResult:
    result: Any = None
    failure_code: str = ""
    automation_request: dict[str, Any] | None = None


def public_step_feedback(call: Any, payload: dict, *, before_revision: int,
                         after_revision: int, selected_action: Any = None) -> dict:
    """One completed call's public feedback, shared by Node and task Runtime.

    This is a projection only: accepted is not a success certificate, and
    failed invocation internals/programs are never serialized wholesale.
    """
    keys = ("accepted", "passed", "error", "message", "failure_code", "stage",
            "r0_passed", "static_passed", "r1_passed", "preflight_failure_code",
            "started", "completed", "failure_layer", "cached_rejection", "rollback",
            "draft_id", "r1_outputs", "r1_witness_refs", "support_occurrence_id",
            "atomic_effect_passed", "validated_outputs", "atomic_witness_refs",
            "error_code", "reason_code", "diagnostics", "support_call_id", "deterministic_rejection_cache_hit",
            "argument_path", "expected_constraint", "actual_summary",
            "allowed_output_mappings", "required_anchor_or_relation", "relevant_revision")
    def project_result(value: dict) -> dict:
        projected = {k: to_primitive(value[k]) for k in keys if k in value}
        if value.get('failure_code') and not projected.get('message'):
            message = value.get('failure_message')
            if not message:
                # Implementation results retain the public error text on the
                # failed Tool result, not at their own root. Copy only that text.
                message = next((r.get('failure_message') for r in value.get('tool_results', [])
                    if isinstance(r, dict) and r.get('failure_code') == value['failure_code']
                    and r.get('failure_message')), None)
            if isinstance(message, str):
                projected['message'] = message
        scope_diagnostics = [r.get('tool_path_evidence', {}).get('final_effect_result', {}).get('scope_diagnostic')
                             for r in value.get('tool_results', []) if isinstance(r, dict)]
        if any(scope_diagnostics):
            projected['scope_diagnostics'] = [to_primitive(d) for d in scope_diagnostics if d]
        return projected

    feedback = {"version": 'r103.runtime-feedback.v2', "call_id": call.call_id, "tool": call.name, "arguments": to_primitive(call.arguments),
                "before_revision": before_revision, "after_revision": after_revision,
                **project_result(payload)}
    if selected_action is not None:
        feedback.update(action_type=selected_action.action_type,
                        action_arguments=to_primitive(selected_action.arguments))
    result = payload.get('result')
    if isinstance(result, dict):
        feedback['helper_result'] = project_result(result)
    trial = payload.get('trial')
    failure_keys = ('failure_code', 'failure_layer', 'message', 'started',
                    'not_committed', 'rollback', 'restored_revision', 'stage')

    def project_failure(value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        projected = {k: to_primitive(value[k]) for k in failure_keys if k in value}
        action = value.get('failing_action')
        if isinstance(action, dict):
            projected['failing_action'] = {k: to_primitive(action[k])
                for k in ('action_type', 'arguments') if k in action}
        return projected

    if isinstance(trial, dict):
        feedback['trial'] = {k: to_primitive(trial[k]) for k in
            ('draft_id', 'r1_outputs', 'terminal_interrupted') if k in trial}
        if 'failure_feedback' in trial:
            feedback['trial']['failure_feedback'] = project_failure(trial['failure_feedback'])
    if isinstance(payload.get('trial_failure'), dict):
        feedback['trial_failure'] = project_failure(payload['trial_failure'])
    validation = payload.get('validation')
    if isinstance(validation, dict):
        feedback['validation'] = {k: to_primitive(validation[k]) for k in keys if k in validation}
        feedback.update({k: validation[k] for k in ('failure_code','message') if validation.get(k)})
    return feedback


def automation_request_tool() -> NativeToolSpec:
    return NativeToolSpec(
        "request_runtime_automation",
        "Request an independent Atomic draft step for systematic or repeated work. "
        "This request does not create a Tool or assert any effect.",
        {"type": "object", "properties": {
            "reason": {"type": "string"},
            "intended_capability": {"type": "string"},
        }, "required": ["reason", "intended_capability"], "additionalProperties": False},
        call_kind='automation_request', scope='runtime', result_owner='draft_stage',
    )


def run_runtime_step(executor: Any, mode: str, occurrence: Any, ctx: Any,
                     invocations: list[Any], support_candidates: list[Any],
                     *, bootstrap: bool = False, draft_request: dict | None = None,
                     atomic_override: Any = None, plan_context_plan: Any = None) -> RuntimeStepResult:
    if mode not in {"preparation", "seeded"}:
        raise ValueError(f"unsupported RuntimeStep mode: {mode}")
    from .interventions import policy
    intervention = policy(ctx)
    if not intervention.programs:
        if draft_request:
            raise RuntimeError("disabled program channel cannot carry a draft request")
        invocations, support_candidates = [], []
    session_kind = "provisional_seeded" if atomic_override is not None else mode
    session = executor.session_factory(f"runtime_step_{session_kind}", occurrence.occurrence_id)
    record = executor._record_session_start(
        session, "RuntimePreparationSession" if mode == "preparation" else "SeededSession",
        occurrence.occurrence_id, ctx,
    )
    span = ctx.trace_builder.start_span(f"runtime_{mode}", occurrence.occurrence_id)
    atomic = atomic_override or executor.invocation_compiler.skills.get_atomic(occurrence.node_ref)
    state = executor._activate_occurrence_state(occurrence, atomic, invocations, ctx,
        plan_context_plan=plan_context_plan)
    state = {**state, "last_step_feedback": ctx.runtime_step_feedback.get(occurrence.occurrence_id, {})}
    bindings = ctx.binding_store.runtime_prompt_projection(occurrence, atomic.inputs)
    audit: dict[str, Any] = {}
    query = getattr(executor, "runtime_resources", None)
    resources = query(ctx.budget.current_occurrence_id or occurrence.occurrence_id) if query else {}
    frame = {"current_step_id": occurrence.step_id, "current_occurrence_id": occurrence.occurrence_id,
             "mode": mode, "current_atomic_ref": str(occurrence.node_ref),
             "repeat": ctx.binding_store.repeat_execution_frame(occurrence.step_id),
             "completed_step_ids": completed_step_ids_for_policy(ctx.trace_builder.trace.node_records),
             "terminal_completed_step_ids": list(dict.fromkeys(n.step_id for n in ctx.trace_builder.trace.node_records
                 if n.status == NodeExecutionStatus.DIRECT_TERMINAL_EFFECT_SUCCESS)),
             "last_step": ctx.runtime_step_feedback.get(occurrence.occurrence_id, {}),
             "remaining_resources": {**resources, "node_actions_used": ctx.budget.used_node_actions,
                                     "task_actions": ctx.budget.remaining_global_actions}}
    # Capture the actual allowed surface before projecting its public copy.
    # Spec construction cannot invoke tools or commit argument bindings.
    surface = executor._build_support_surface(support_candidates if not draft_request else [],
        atomic, occurrence, ctx, session.session_id)
    if draft_request:
        tools = [executor._automation_tool()]
    else:
        tools = [tool for tool in executor._node_tools(
            ctx, atomic, invocations=invocations, allow_plan_conflict=True,
            support_candidates=support_candidates,
            support_call_surface=surface,
        ) if tool.name != "propose_runtime_automation_atomic"]
        if intervention.programs:
            tools.append(automation_request_tool())
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
        exploration_memory=ctx.exploration_policy_view(),
        support_atomic_candidates=support_candidates,
        support_call_surface=surface,
        support_summary_lookup=executor.context_builder.selected_support_summaries(
            executor.invocation_compiler.skills, support_candidates),
        runtime_automation_drafts=list(ctx.runtime_automation_drafts.values()),
        runtime_automation_interface=(build_runtime_automation_interface(
            ctx.harness, occurrence, ctx.binding_store) if draft_request else None),
        recent_failed_learned_invocation=ctx.last_failed_invocation,
        projection_audit=audit,
        native_tool_specs=tools,
        runtime_step_mode=mode,
        rejected_candidates=current_rejections(ctx, occurrence),
        execution_frame=frame,
    )
    instruction = ""
    if draft_request:
        import json
        instruction += "\nAUTOMATION_REQUEST\n" + json.dumps(draft_request, ensure_ascii=False)
    prompt = instruction + "\n" + prompt
    increment(ctx, "runtime_step_count")
    increment(ctx, f"runtime_step_{mode}_count")
    if bootstrap:
        increment(ctx, "graph_bootstrap_agent_step_count")
    if draft_request:
        increment(ctx, "runtime_automation_draft_step_count")
        increment(ctx, "runtime_automation_interface_load_count")
    outcome = RuntimeStepResult()
    call = None
    starts = {key: len(getattr(ctx.trace_builder.trace, key))
              for key in ('tool_executions', 'implementation_invocations')}
    try:
        ctx.search_history.note_projection(ctx, session.session_id)
        executor._record_runtime_context_projection(
            ctx, audit, session_id=session.session_id,
            occurrence_id=occurrence.occurrence_id, origin="runtime_step",
        )
        turn = session.next_turn(prompt, tools=tools)
        if getattr(executor.invocation_compiler, "r103", False):
            executor.invocation_compiler.record_display(ctx, session.session_id, invocations, tools)
            executor._record_support_display(ctx, session.session_id, support_candidates, tools)
        executor._record_turn(session, turn, ctx)
        if not draft_request:
            executor._mark_prior_trials_parent_continuation(ctx, occurrence, session.session_id)
        call = turn.tool_calls[0]
        before_revision = ctx.world_revision
        selected_action = next((item for item in ctx.action_catalog
            if call.name == 'environment_action' and item.action_id == call.arguments.get('action_id')
            and item.revision == ctx.world_revision), None)
        # New Support calls must decode their request-local option before the
        # canonical real-call negative cache can be consulted.
        cached = (None if surface is not None and call.name == 'invoke_support_atomic'
                  else cached_rejection(ctx, occurrence, call))
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
                plan_context_plan=plan_context_plan,
            )
            if call.arguments["intent"] == "attempt_current_atomic" and payload.get("accepted"):
                resolutions = []
                outcome.result = executor._complete_from_current_effect(
                    occurrence, ctx, mode=mode,
                    preferred_values=[],
                    preferred_bindings=dict(call.arguments.get("candidate_bindings") or {}),
                    candidate_outputs=dict(call.arguments.get("candidate_outputs") or {}),
                    atomic_override=atomic, resolution_out=resolutions,
                )
                if resolutions:
                    payload["validation"] = to_primitive(resolutions[-1])
                    payload["atomic_effect_passed"] = outcome.result is not None
            if payload.get("fallback_required"):
                outcome.failure_code = "runtime_action_loop_blocked"
        elif call.name == "invoke_support_atomic":
            increment(ctx, "support_agent_selected_count")
            payload = executor._invoke_support_atomic_call(
                call, session, occurrence, ctx, atomic, support_candidates, None,
                surface=surface,
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
                from .invocation_transaction import execute_invocation
                outcome.result = execute_invocation(executor.implementation_runner,
                    compiled, preflight, occurrence, ctx, agent_prepared=True,
                    authorizing_native_call_id=call.call_id,
                )
                if mode == "seeded" and outcome.result.atomic_effect_passed:
                    outcome.result.node_status = NodeExecutionStatus.SEEDED_SUCCESS
            payload = to_primitive(outcome.result or preflight)
            executor._record_control_call(call, session, occurrence, ctx,
                                         call_kind="implementation_invocation", result=payload)
            if not preflight.passed:
                from ..traces.schema import ImplementationInvocationRecord
                rejected = executor.not_started(occurrence,
                    failure_code=preflight.failure_code, failure_layer=preflight.failure_layer)
                rejected.implementation_ref = str(compiled.implementation.ref)
                ctx.trace_builder.trace.implementation_invocations.append(
                    ImplementationInvocationRecord(
                        f"preflight_{call.call_id}", occurrence.occurrence_id,
                        str(compiled.implementation.ref), dict(call.arguments),
                        to_primitive(preflight), to_primitive(rejected), span.span_id,
                    ))
                ctx.record_failed_invocation(
                    occurrence_id=occurrence.occurrence_id,
                    implementation_ref=str(compiled.implementation.ref),
                    failure_code=preflight.failure_code, message=preflight.message,
                )
        if not (surface is not None and call.name == 'invoke_support_atomic'):
            remember_rejection(ctx, occurrence, call, payload)
        # Carry finite public feedback, never an assistant conversation or
        # Tool body, to the next fresh decision. Rollback does not erase it.
        ctx.runtime_step_feedback[occurrence.occurrence_id] = public_step_feedback(
            call, payload, before_revision=before_revision,
            after_revision=ctx.world_revision, selected_action=selected_action)
        executor._finalize_tool_result(session, call.call_id, payload, tools)
        return outcome
    except BudgetExhausted as exc:
        record_interrupted_call(executor, call, session, occurrence, ctx, exc, starts)
        raise
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
