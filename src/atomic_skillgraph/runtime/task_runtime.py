"""Task-scoped Runtime decisions using the normal tool and automation boundaries."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.results import RuntimeOccurrence
from ..core.refs import SkillRef
from ..core.serialization import to_primitive
from ..tooling.runtime_interface import build_runtime_automation_interface
from .invocation_transaction import execute_invocation
from .loop_guard import ActionLoopGuard
from .checkpoint import increment
from .runtime_step import automation_request_tool


@dataclass(frozen=True)
class TaskConsumer:
    """Audit/source owner, not a graph node or an invented Atomic contract."""
    occurrence_id: str = "__task__"
    step_id: str = "__task__"
    node_ref: str = ""
    consumer_scope: str = "task"
    binding_specs: dict = field(default_factory=dict)


def invoke_task_capability(executor, call, consumer, ctx, candidate):
    """Same compiler, preflight, interpreter and transaction; no parent transfer."""
    if call.arguments.get('output_mapping'):
        return {"accepted": False, "error": "task_consumer_has_no_parent_roles"}
    ref = SkillRef.parse(candidate.atomic_ref)
    atomic = executor.invocation_compiler.skills.get_atomic(ref)
    occurrence = RuntimeOccurrence(
        step_id='support::__task__', occurrence_id='support_task_' + uuid.uuid4().hex,
        node_ref=ref, requirement_ids=[], binding_specs={},
        implementation_candidates=[str(i.ref) for i in executor.invocation_compiler.skills.implementations_for(
            ref, mode=executor.invocation_compiler.mode)], expected_effects=list(atomic.effects))
    choices = executor.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=ctx.task_id)
    preferred = [c for c in choices if c.implementation.quality.get('preferred')]
    if not choices or (len(choices) > 1 and len(preferred) != 1):
        return {"accepted": False, "error": "runtime_support_no_unambiguous_executable"}
    compiled = preferred[0] if preferred else choices[0]
    previous, refresh = ctx.active_occurrence_id, ctx._after_action_refresh
    try:
        ctx.begin_occurrence(occurrence)
        prepared = executor.invocation_compiler.prepare_arguments(
            compiled, call_name=compiled.spec.name, call_id=call.call_id,
            arguments=dict(call.arguments.get('arguments') or {}), occurrence=occurrence,
            binding_store=ctx.binding_store, evidence_store=ctx.evidence_store,
            revision=ctx.world_revision)
        preflight = executor.invocation_compiler.validate_execution_context(
            compiled, prepared, occurrence=occurrence, binding_store=ctx.binding_store,
            evidence_store=ctx.evidence_store, revision=ctx.world_revision) if prepared.passed else prepared
        if not preflight.passed:
            return {"accepted": False, "error": preflight.failure_code, "message": preflight.message}
        result = execute_invocation(executor.implementation_runner, compiled, preflight,
            occurrence, ctx, agent_prepared=True, origin='task_agent_selected_registered', consumer=consumer)
        if result.completed and result.atomic_effect_passed and not result.failure_code:
            ctx.binding_store.publish_validated_outputs(occurrence.occurrence_id, result.validated_outputs,
                result.atomic_witness_refs, ctx.world_revision, certified_bindings=result.validated_output_bindings)
            for role, binding in result.validated_output_bindings.items():
                ctx.evidence_store.add_validated_tool_output(role, binding.value, result.atomic_witness_refs,
                    certified_binding=binding, occurrence_id=occurrence.occurrence_id)
        record = {"consumer_scope": "task", "parent_atomic_ref": "", "occurrence_id": occurrence.occurrence_id,
                  "step_id": occurrence.step_id, "atomic_ref": str(ref), "status": result.node_status.value,
                  "direct_result": to_primitive(result), "validated_outputs": to_primitive(result.validated_outputs)}
        ctx.trace_builder.trace.metadata.setdefault('runtime_support_node_records', []).append(record)
        return {"accepted": True, "passed": bool(result.completed and result.atomic_effect_passed and not result.failure_code),
                "consumer_scope": "task", "result": to_primitive(result), "new_revision": ctx.world_revision}
    finally:
        ctx.active_occurrence_id, ctx._after_action_refresh = previous, refresh


def run_dynamic(executor, ctx, *, rescue=False, cold_start_continuation=False, continuation_context=None):
    consumer = TaskConsumer()
    ctx.clear_active_occurrence()
    kind = 'cold_start_dynamic_continuation' if cold_start_continuation else 'task_rescue' if rescue else 'full_dynamic'
    session_kind = 'runtime_step_dynamic_cold_start_continuation' if cold_start_continuation else 'runtime_step_dynamic'
    span = ctx.trace_builder.start_span(kind, '', learnable=True)
    guard, failure_code, request, feedback = ActionLoopGuard(), '', None, {}
    try:
        while not ctx.execution_terminal():
            # New decision sessions do not create a resource pool. System's
            # UsageLedger provides the same task remainder to every session.
            pool = list(executor.invocation_compiler.skills.atomics(mode=executor.invocation_compiler.mode))
            candidates = executor.support_retriever.retrieve_for_task(
                query=ctx.task_goal, atomics=pool,
                execution_availability=executor._support_execution_availability(pool))
            tools = [executor._environment_tool(ctx, node_level=False), executor._status_tool(), automation_request_tool()]
            if candidates:
                tools.append(executor._support_tool(candidates))
            if request:
                tools = [executor._automation_tool()]
            frame = {"consumer_scope": "task", "source_occurrence_id": consumer.occurrence_id, "parent_atomic_ref": "",
                     "capability_candidates": to_primitive(candidates), "last_step": feedback,
                     "continuation_context": continuation_context or {},
                     "automation_request": request}
            resources = getattr(executor, 'runtime_resources', None)
            frame['remaining_resources'] = resources(consumer.occurrence_id) if resources else {}
            draft_step = request is not None
            if request:
                frame['runtime_automation_interface'] = build_runtime_automation_interface(ctx.harness, consumer, ctx.binding_store)
            audit = {}
            prompt = executor.context_builder.dynamic_task(
                task_goal=ctx.task_goal, observation=ctx.observation, action_catalog=ctx.action_catalog,
                relevant_action_history=ctx.relevant_history(''), remaining_budget=ctx.budget.snapshot(),
                task_progress=executor._task_progress_policy(ctx), exploration_memory=ctx.exploration_memory.policy_view(),
                recent_failed_learned_invocation=ctx.last_failed_invocation,
                rescue_method_guidance=executor._rescue_method_guidance(ctx) if rescue else None, projection_audit=audit, task_runtime_frame=frame)
            prompt = ('Choose one native call. You may invoke an offered capability using explicitly supplied arguments, '
                       'or request_runtime_automation for reusable multi-step work. Helper outputs are their own verified results, '
                       'not completion of the whole task. There is no parent Atomic and no automatic binding transfer. '
                       'A successful automation trial has already run; do not repeat it merely to commit it.\n') + prompt
            session = executor.session_factory(session_kind, '__task__')
            record = executor._record_session_start(session, 'DynamicTaskSession', '', ctx)
            increment(ctx, 'runtime_step_count')
            increment(ctx, 'runtime_step_dynamic_count')
            if draft_step:
                increment(ctx, 'runtime_automation_draft_step_count')
                increment(ctx, 'runtime_automation_interface_load_count')
            try:
                executor._record_runtime_context_projection(ctx, audit, session_id=session.session_id, occurrence_id='', origin='runtime_step')
                turn = session.next_turn(prompt, tools=tools)
                executor._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                request = None
                if call.name == 'environment_action':
                    payload, _ = executor._execute_environment_call(call, session, None, ctx,
                        span_id=span.span_id, origin=kind, loop_guard=guard)
                    if payload.get('fallback_required'):
                        failure_code = 'runtime_action_loop_blocked'
                elif call.name == 'request_runtime_automation':
                    increment(ctx, 'runtime_automation_request_count')
                    request = dict(call.arguments)
                    payload = {"accepted": True, "next": "RuntimeAutomationDraftStep"}
                elif call.name == 'propose_runtime_automation_atomic':
                    ctx.begin_occurrence(consumer)
                    payload = executor._process_runtime_automation_call(call, ctx, consumer)
                    ctx.clear_active_occurrence()
                elif call.name == 'invoke_support_atomic':
                    payload = executor._invoke_support_atomic_call(call, session, consumer, ctx, None, candidates)
                else:
                    payload = {"accepted": True, "status": call.arguments.get('status')}
                    failure_code = 'benchmark_failure'
                if call.name not in {'environment_action', 'invoke_support_atomic'}:
                    executor._record_control_call(call, session, consumer, ctx, call_kind='task_runtime', result=payload)
                feedback = {key: payload[key] for key in ('accepted', 'passed', 'error', 'message', 'failure_code', 'stage', 'r1_passed') if key in payload}
                result = payload.get('result') or payload.get('trial') or {}
                feedback['validated_outputs'] = result.get('validated_outputs', result.get('r1_outputs', {}))
                executor._finalize_tool_result(session, call.call_id, payload, tools)
                if failure_code:
                    break
            finally:
                executor._finish_session(record, session, ctx)
                ctx.trace_builder.trace.metadata.setdefault('runtime_steps', []).append({
                    'session_id': session.session_id, 'occurrence_id': consumer.occurrence_id,
                    'mode': 'dynamic', 'bootstrap': False, 'draft': draft_step,
                    'accepted_semantic_turn_count': sum(
                        item.session_id == session.session_id for item in ctx.trace_builder.trace.agent_turns),
                })
    except AtomicSkillGraphError as exc:
        if exc.layer == FailureLayer.INFRASTRUCTURE:
            raise
        failure_code = exc.code
    finally:
        ctx.trace_builder.finish_span(span.span_id)
    won = ctx.benchmark_terminal()
    terminal = executor.validation.task.terminal(ctx.task_contract, ctx.harness.validator_channel(), won)
    return {"benchmark_won": won, "success": won, "strict_success": won,
            "task_contract_success": bool(terminal.checks.get('task_contract', False)),
            "failure_code": failure_code, "rescue": rescue, "cold_start_continuation": cold_start_continuation}
