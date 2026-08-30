"""Per-occurrence Direct/Preparation/Fresh-Seeded state machine and Full Dynamic agent."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Callable

from ..agents.context_builder import ContextBuilder
from ..agents.protocol import AgentTurn, NativeToolSpec
from ..core.bindings import (
    BindingExpression, BindingExprKind, BindingResolution, BindingSource,
    BindingStatus, RuntimeBinding,
)
from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.results import ImplementationExecutionResult, NodeExecutionStatus
from ..core.serialization import to_primitive
from ..traces.schema import (
    AgentSessionRecord, AgentTurnRecord, EnvironmentActionRecord,
    ImplementationInvocationRecord, NativeToolCallRecord, ValidationRecord,
)
from ..validation.engine import ValidationEngine
from .implementation_runner import ImplementationRunner
from .invocation_compiler import CompiledInvocation, InvocationCompiler
from .loop_guard import ActionLoopGuard


SessionFactory = Callable[[str, str], Any]


class NodeExecutor:
    def __init__(
        self, invocation_compiler: InvocationCompiler, validation: ValidationEngine,
        session_factory: SessionFactory,
    ) -> None:
        self.invocation_compiler = invocation_compiler
        self.validation = validation
        self.session_factory = session_factory
        self.implementation_runner = ImplementationRunner(validation)
        self.context_builder = ContextBuilder()

    def not_started(self, occurrence: Any, *, failure_code: str) -> ImplementationExecutionResult:
        return ImplementationExecutionResult(
            "", str(occurrence.node_ref), False, False, False, False,
            failure_layer="implementation", failure_code=failure_code,
            node_status=NodeExecutionStatus.FAILED_NOT_STARTED,
        )

    def try_autonomous(
        self, occurrence: Any, invocations: list[CompiledInvocation], ctx: Any,
    ) -> ImplementationExecutionResult | None:
        if not invocations:
            return self.not_started(occurrence, failure_code="no_compatible_implementation")
        preferred = [item for item in invocations if item.implementation.quality.get("preferred")]
        if len(invocations) > 1 and len(preferred) != 1:
            return None
        compiled = preferred[0] if preferred else invocations[0]
        preflight = self.invocation_compiler.autonomous_preflight(
            compiled, occurrence, ctx.binding_store, ctx.evidence_store, ctx.world_revision,
            task_contract=ctx.task_contract,
        )
        if not preflight.passed:
            # Missing runtime-resolvable arguments are Preparation work, not an
            # attempted Direct failure and do not create long-term evidence.
            return None
        return self.implementation_runner.run(compiled, preflight, occurrence, ctx, agent_prepared=False)

    def _environment_tool(self, ctx: Any) -> NativeToolSpec:
        return NativeToolSpec(
            "environment_action", "Execute one currently available environment action.",
            {
                "type": "object", "required": ["action_id"], "additionalProperties": False,
                "properties": {"action_id": {"type": "string", "enum": [item.action_id for item in ctx.action_catalog]}},
            },
        )

    @staticmethod
    def _status_tool() -> NativeToolSpec:
        return NativeToolSpec(
            "report_runtime_status", "Explicitly report that the current mode cannot continue.",
            {
                "type": "object", "required": ["status"], "additionalProperties": False,
                "properties": {"status": {"type": "string", "enum": ["cannot_resolve", "give_up"]}},
            },
        )

    @staticmethod
    def _invocation_tool(item: CompiledInvocation) -> NativeToolSpec:
        return NativeToolSpec(item.spec.name, item.spec.description, item.spec.input_schema)

    def _record_session_start(self, session: Any, session_type: str, occurrence_id: str, ctx: Any) -> AgentSessionRecord:
        import time
        record = AgentSessionRecord(session.session_id, session_type, occurrence_id, time.time())
        ctx.trace_builder.trace.agent_sessions.append(record)
        return record

    def _record_turn(self, session: Any, turn: AgentTurn, ctx: Any) -> None:
        index = sum(1 for item in ctx.trace_builder.trace.agent_turns if item.session_id == session.session_id)
        usage = {
            "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens,
            "total_tokens": turn.total_tokens, "reasoning_tokens": turn.reasoning_tokens,
            "call_count": 1, "latency_ms": turn.latency_ms,
        }
        ctx.trace_builder.trace.agent_turns.append(AgentTurnRecord(
            session.session_id, index, turn.content, turn.finish_reason,
            [item.call_id for item in turn.tool_calls], usage, dict(turn.provider_metadata),
        ))
        ctx.trace_builder.trace.llm_usage.append({"session_id": session.session_id, **usage})

    def _finish_session(self, record: AgentSessionRecord, session: Any) -> None:
        import time
        record.ended_at = time.time()
        record.snapshot = session.snapshot()

    @staticmethod
    def _finalize_tool_result(session: Any, call_id: str, result: dict[str, Any], tools: list[NativeToolSpec]) -> None:
        finalize = getattr(session, "finalize_tool_result", None)
        if callable(finalize):
            finalize(call_id, result)
        else:
            # Deterministic test sessions may implement terminal submission
            # without issuing a provider call.
            session.submit_tool_result(call_id, result, tools=tools)

    def _execute_environment_call(
        self, call: Any, session: Any, occurrence: Any | None, ctx: Any,
        *, span_id: str, origin: str, loop_guard: ActionLoopGuard,
    ) -> tuple[dict[str, Any], Any]:
        action_id = str(call.arguments["action_id"])
        spec = next(
            (item for item in ctx.action_catalog if item.action_id == action_id),
            None,
        )
        if spec is None or int(spec.revision) != int(ctx.world_revision):
            raise AtomicSkillGraphError(
                "runtime_agent_schema_error",
                f"stale or unknown environment action_id: {action_id}",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        loop = loop_guard.inspect(
            action_type=spec.action_type,
            arguments=spec.arguments,
            observation=ctx.observation,
            catalog=ctx.action_catalog,
        )
        if loop.blocked:
            payload = {
                **loop.tool_result(),
                "observation": ctx.observation,
                "done": False,
                "won": False,
                "new_revision": ctx.world_revision,
                "remaining_budget": ctx.budget.snapshot(),
            }
            ctx.trace_builder.trace.native_tool_calls.append(NativeToolCallRecord(
                call.call_id, session.session_id,
                "" if occurrence is None else occurrence.occurrence_id,
                call.name, dict(call.arguments), "environment_action",
                {
                    "passed": False,
                    "failure_code": "loop_blocked",
                    "message": loop.reason,
                },
                f"revision:{ctx.world_revision}",
                sum(
                    1 for item in ctx.trace_builder.trace.agent_turns
                    if item.session_id == session.session_id
                ) - 1,
            ))
            return payload, spec
        ctx.budget.consume_action()
        result = ctx.harness.execute_action(action_id, spec.revision)
        record = EnvironmentActionRecord(
            spec.action_id, spec.revision, spec.action_type, dict(spec.arguments), result.accepted,
            result.observation, result.done, result.won, result.new_revision, span_id,
            to_primitive(result.transition_certificate) if result.transition_certificate else None,
        )
        ctx.trace_builder.trace.environment_actions.append(record)
        occurrence_id = "" if occurrence is None else occurrence.occurrence_id
        ctx.update_after_action(result, {**to_primitive(record), "occurrence_id": occurrence_id, "origin": origin})
        payload = {
            "accepted": result.accepted, "observation": result.observation, "done": result.done,
            "won": result.won, "new_revision": result.new_revision,
            # A revision changes the meaning of action ids.  The next Agent
            # turn receives the complete new policy-facing catalog.  Keep the
            # replay representation compact: action_id is revision-scoped,
            # new_revision is carried above, and display_text supplies the
            # only action semantics the Agent needs to select that opaque id.
            # Full typed arguments remain in TaskContext, GroundingEvidence,
            # and the canonical EnvironmentActionRecord.
            "action_catalog": [
                {
                    "action_id": item.action_id,
                    "display_text": item.display_text,
                }
                for item in result.catalog
            ],
            "remaining_budget": ctx.budget.snapshot(),
        }
        ctx.trace_builder.trace.native_tool_calls.append(NativeToolCallRecord(
            call.call_id, session.session_id, occurrence_id, call.name, dict(call.arguments),
            "environment_action", {"passed": True}, f"revision:{result.new_revision}",
            sum(1 for item in ctx.trace_builder.trace.agent_turns if item.session_id == session.session_id) - 1,
        ))
        return payload, spec

    def _effect_result(self, occurrence: Any, ctx: Any, *, mode: str) -> ImplementationExecutionResult | None:
        atomic = self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        bindings = ctx.binding_store.snapshot_for_node(occurrence)
        certificate = (
            ctx.trace_builder.trace.environment_actions[-1].transition_certificate
            if ctx.trace_builder.trace.environment_actions
            else None
        )
        validation_bindings = self._with_ephemeral_certificate_bindings(
            atomic, bindings, certificate, ctx.world_revision,
        )
        outputs = self._validated_output_candidates(
            atomic, validation_bindings, ctx, certificate=certificate,
        )
        validation = self.validation.atomic.validate(
            atomic, occurrence, validation_bindings,
            ctx.harness.validator_channel(), outputs,
        )
        ctx.trace_builder.trace.validations.append(ValidationRecord(
            occurrence.occurrence_id, "atomic", to_primitive(validation), ctx.world_revision,
        ))
        if not validation.passed:
            return None
        status = (
            NodeExecutionStatus.AGENT_COMPLETED_BEFORE_INVOCATION
            if mode == "preparation" else NodeExecutionStatus.SEEDED_SUCCESS
        )
        return ImplementationExecutionResult(
            "", str(atomic.ref), False, False, True, True, validated_outputs=outputs,
            before_state_ref="", after_state_ref=f"revision:{ctx.world_revision}", node_status=status,
        )

    @staticmethod
    def _with_ephemeral_certificate_bindings(
        atomic: Any,
        bindings: dict[str, RuntimeBinding],
        certificate: dict[str, Any] | None,
        revision: int,
    ) -> dict[str, RuntimeBinding]:
        """Project certified effect arguments for this validation only.

        Environment actions must not mutate the task-local BindingStore.  A
        transition certificate may nevertheless witness a previously missing
        input while validating the atomic occurrence that caused the action.
        The returned mapping is an isolated snapshot and is never committed.
        """

        projected = dict(bindings)
        if certificate is None:
            return projected
        parameters = {item.name: item for item in atomic.inputs}
        facts = [
            *list(certificate.get("positive_effects") or []),
            *list(certificate.get("terminal_effects") or []),
        ]
        for effect in atomic.effects:
            for fact in facts:
                if str(fact.get("predicate", "")).casefold() != effect.predicate.casefold():
                    continue
                arguments = dict(fact.get("args") or {})
                if set(arguments) != set(effect.args):
                    continue
                for argument_name, raw_expression in effect.args.items():
                    expression = (
                        BindingExpression.from_dict(raw_expression)
                        if isinstance(raw_expression, dict)
                        else raw_expression
                    )
                    if (
                        not isinstance(expression, BindingExpression)
                        or expression.kind is BindingExprKind.CONSTANT
                        or expression.source_role not in parameters
                    ):
                        continue
                    existing = projected.get(expression.source_role)
                    if existing is not None and existing.status is BindingStatus.GROUNDED:
                        continue
                    projected[expression.source_role] = RuntimeBinding(
                        role=expression.source_role,
                        value=arguments[argument_name],
                        semantic_type=parameters[expression.source_role].semantic_type,
                        source=BindingSource.HARNESS_EVIDENCE,
                        status=BindingStatus.GROUNDED,
                        resolution=BindingResolution.RELATION_VERIFIED,
                        evidence_refs=[str(
                            fact.get("fact_ref")
                            or f"transition:{certificate.get('action_id', '')}"
                        )],
                        world_revision=revision,
                    )
                break
        return projected

    def _validated_output_candidates(
        self,
        atomic: Any,
        bindings: dict[str, RuntimeBinding],
        ctx: Any,
        *,
        certificate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        plain = {role: value.value for role, value in bindings.items() if value.status is BindingStatus.GROUNDED}
        facts = list(ctx.harness.validator_channel().snapshot().get("facts", []))
        if certificate is not None:
            facts.extend(certificate.get("positive_effects") or [])
            facts.extend(certificate.get("terminal_effects") or [])
        witnesses = [fact for fact in facts if any(fact.get("predicate") == effect.predicate for effect in atomic.effects)]
        witness_args = witnesses[-1].get("args", {}) if witnesses else {}
        for output in atomic.outputs:
            if output.name in plain:
                result[output.name] = plain[output.name]
                continue
            if "object" in output.name or output.semantic_type in {"entity", "object"}:
                value = witness_args.get("object") or plain.get("object")
                if value is not None:
                    result[output.name] = value
                    continue
            if "location" in output.name:
                value = witness_args.get("location") or plain.get("destination") or plain.get("target_location")
                if value is not None:
                    result[output.name] = value
        return result

    def run_preparation_session(
        self, occurrence: Any, invocations: list[CompiledInvocation], ctx: Any,
        *, learned_call_repair_limit: int = 2,
    ) -> ImplementationExecutionResult:
        session = self.session_factory("runtime_preparation", occurrence.occurrence_id)
        record = self._record_session_start(session, "RuntimePreparationSession", occurrence.occurrence_id, ctx)
        span = ctx.trace_builder.start_span("runtime_preparation", occurrence.occurrence_id)
        atomic = self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        prompt_bindings = ctx.binding_store.runtime_prompt_projection(
            occurrence, atomic.inputs,
        )
        missing = prompt_bindings["missing_or_insufficient_bindings"]
        prompt = self.context_builder.runtime_node(
            task_goal=ctx.task_goal, atomic_contract=atomic,
            semantic_anchors=prompt_bindings["semantic_anchors"],
            execution_ready_bindings=prompt_bindings["execution_ready_bindings"],
            missing_or_insufficient_bindings=missing,
            observation=ctx.observation,
            action_catalog=ctx.action_catalog, relevant_action_history=ctx.relevant_history(occurrence.occurrence_id),
            remaining_budget=ctx.budget.snapshot(), implementation_invocations=[item.spec for item in invocations],
        )
        tools = [self._environment_tool(ctx), *[self._invocation_tool(item) for item in invocations], self._status_tool()]
        preflight_failures = 0
        loop_guard = ActionLoopGuard()
        try:
            turn = session.next_turn(prompt, tools=tools)
            while True:
                self._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                if call.name == "report_runtime_status":
                    self._finalize_tool_result(
                        session, call.call_id,
                        {"accepted": True, "status": call.arguments["status"]}, tools,
                    )
                    return self.not_started(occurrence, failure_code="runtime_binding_unresolved")
                if call.name == "environment_action":
                    payload, _ = self._execute_environment_call(
                        call, session, occurrence, ctx,
                        span_id=span.span_id, origin="runtime_preparation",
                        loop_guard=loop_guard,
                    )
                    if payload.get("loop_blocked"):
                        tools = [self._environment_tool(ctx), *[self._invocation_tool(item) for item in invocations], self._status_tool()]
                        if payload.get("fallback_required"):
                            self._finalize_tool_result(session, call.call_id, payload, tools)
                            return self.not_started(
                                occurrence, failure_code="runtime_action_loop_blocked",
                            )
                        turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                        continue
                    effect = self._effect_result(occurrence, ctx, mode="preparation")
                    tools = [self._environment_tool(ctx), *[self._invocation_tool(item) for item in invocations], self._status_tool()]
                    if effect is not None:
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        return effect
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                compiled = next(item for item in invocations if item.spec.name == call.name)
                preflight = self.invocation_compiler.preflight(
                    compiled, call_name=call.name, call_id=call.call_id, arguments=call.arguments,
                    occurrence=occurrence, binding_store=ctx.binding_store, evidence_store=ctx.evidence_store,
                    revision=ctx.world_revision,
                    task_contract=ctx.task_contract,
                )
                ctx.trace_builder.trace.native_tool_calls.append(NativeToolCallRecord(
                    call.call_id, session.session_id, occurrence.occurrence_id, call.name, dict(call.arguments),
                    "implementation_invocation", to_primitive(preflight), None,
                    sum(1 for item in ctx.trace_builder.trace.agent_turns if item.session_id == session.session_id) - 1,
                ))
                if not preflight.passed:
                    rejected = ImplementationExecutionResult(
                        str(compiled.implementation.ref), str(compiled.atomic.ref),
                        False, False, False, False,
                        failure_layer=preflight.failure_layer or "implementation",
                        failure_code=preflight.failure_code,
                        node_status=NodeExecutionStatus.FAILED_NOT_STARTED,
                    )
                    ctx.trace_builder.trace.implementation_invocations.append(
                        ImplementationInvocationRecord(
                            f"preflight_{call.call_id}", occurrence.occurrence_id,
                            str(compiled.implementation.ref), dict(call.arguments),
                            to_primitive(preflight), to_primitive(rejected), span.span_id,
                        )
                    )
                    preflight_failures += 1
                    payload = {"error": preflight.failure_code, "message": preflight.message, "repairable": preflight_failures <= learned_call_repair_limit}
                    if preflight_failures > learned_call_repair_limit:
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        return self.not_started(occurrence, failure_code=preflight.failure_code)
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                result = self.implementation_runner.run(compiled, preflight, occurrence, ctx, agent_prepared=True)
                self._finalize_tool_result(session, call.call_id, to_primitive(result), tools)
                return result
        except AtomicSkillGraphError as exc:
            if exc.layer == FailureLayer.INFRASTRUCTURE:
                raise
            failure = self.not_started(
                occurrence,
                failure_code=exc.code or "runtime_agent_schema_error",
            )
            failure.failure_layer = exc.layer.value
            return failure
        finally:
            ctx.trace_builder.finish_span(span.span_id)
            self._finish_session(record, session)

    def run_seeded_fresh(self, occurrence: Any, ctx: Any) -> ImplementationExecutionResult:
        session = self.session_factory("runtime_seeded", occurrence.occurrence_id)
        record = self._record_session_start(session, "SeededSession", occurrence.occurrence_id, ctx)
        span = ctx.trace_builder.start_span("runtime_seeded", occurrence.occurrence_id)
        atomic = self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        prompt_bindings = ctx.binding_store.runtime_prompt_projection(
            occurrence, atomic.inputs,
        )
        prompt = self.context_builder.seeded_node(
            task_goal=ctx.task_goal, atomic_contract=atomic,
            semantic_anchors=prompt_bindings["semantic_anchors"],
            execution_ready_bindings=prompt_bindings["execution_ready_bindings"],
            missing_or_insufficient_bindings=prompt_bindings[
                "missing_or_insufficient_bindings"
            ],
            observation=ctx.observation, action_catalog=ctx.action_catalog,
            relevant_action_history=ctx.relevant_history(occurrence.occurrence_id), remaining_budget=ctx.budget.snapshot(),
        )
        tools = [self._environment_tool(ctx), self._status_tool()]
        loop_guard = ActionLoopGuard()
        try:
            turn = session.next_turn(prompt, tools=tools)
            while True:
                self._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                if call.name == "report_runtime_status":
                    self._finalize_tool_result(session, call.call_id, {"accepted": True}, tools)
                    break
                payload, _ = self._execute_environment_call(
                    call, session, occurrence, ctx,
                    span_id=span.span_id, origin="runtime_seeded",
                    loop_guard=loop_guard,
                )
                if payload.get("loop_blocked"):
                    tools = [self._environment_tool(ctx), self._status_tool()]
                    if payload.get("fallback_required"):
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        result = self.not_started(
                            occurrence, failure_code="runtime_action_loop_blocked",
                        )
                        result.node_status = NodeExecutionStatus.SEEDED_FAILED
                        return result
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                effect = self._effect_result(occurrence, ctx, mode="seeded")
                tools = [self._environment_tool(ctx), self._status_tool()]
                if effect is not None:
                    self._finalize_tool_result(session, call.call_id, payload, tools)
                    return effect
                if payload["done"] and not payload["won"]:
                    self._finalize_tool_result(session, call.call_id, payload, tools)
                    break
                turn = session.submit_tool_result(call.call_id, payload, tools=tools)
        except AtomicSkillGraphError as exc:
            if exc.layer == FailureLayer.INFRASTRUCTURE:
                raise
            failure = self.not_started(
                occurrence,
                failure_code=exc.code or "runtime_node_token_budget_exhausted",
            )
            failure.failure_layer = exc.layer.value
            failure.node_status = NodeExecutionStatus.SEEDED_FAILED
            return failure
        finally:
            ctx.trace_builder.finish_span(span.span_id)
            self._finish_session(record, session)
        result = self.not_started(occurrence, failure_code="atomic_effect_violation")
        result.node_status = NodeExecutionStatus.SEEDED_FAILED
        return result

    def run_dynamic(self, ctx: Any, *, rescue: bool = False) -> dict[str, Any]:
        session = self.session_factory("runtime_dynamic", "__task__")
        session_record = self._record_session_start(session, "DynamicTaskSession", "", ctx)
        span = ctx.trace_builder.start_span("task_rescue" if rescue else "full_dynamic", "", learnable=True)
        prompt = self.context_builder.dynamic_task(
            task_goal=ctx.task_goal, observation=ctx.observation, action_catalog=ctx.action_catalog,
            relevant_action_history=ctx.action_history, remaining_budget=ctx.budget.snapshot(),
        )
        tools = [self._environment_tool(ctx), self._status_tool()]
        success = False
        failure_code = ""
        loop_guard = ActionLoopGuard()
        try:
            terminal = self.validation.task.terminal(
                ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
            )
            if terminal.passed:
                return {"success": True, "failure_code": "", "rescue": rescue}
            turn = session.next_turn(prompt, tools=tools)
            while True:
                self._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                if call.name == "report_runtime_status":
                    self._finalize_tool_result(session, call.call_id, {"accepted": True}, tools)
                    failure_code = "benchmark_failure"
                    break
                payload, _ = self._execute_environment_call(
                    call, session, None, ctx,
                    span_id=span.span_id,
                    origin="task_rescue" if rescue else "full_dynamic",
                    loop_guard=loop_guard,
                )
                if payload.get("loop_blocked"):
                    tools = [self._environment_tool(ctx), self._status_tool()]
                    if payload.get("fallback_required"):
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        failure_code = "runtime_action_loop_blocked"
                        break
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                terminal = self.validation.task.terminal(ctx.task_contract, ctx.harness.validator_channel(), payload["won"])
                tools = [self._environment_tool(ctx), self._status_tool()]
                if terminal.passed:
                    self._finalize_tool_result(session, call.call_id, payload, tools)
                    success = True
                    break
                if payload["done"]:
                    self._finalize_tool_result(session, call.call_id, payload, tools)
                    failure_code = terminal.failure_codes[0] if terminal.failure_codes else "benchmark_failure"
                    break
                turn = session.submit_tool_result(call.call_id, payload, tools=tools)
        except AtomicSkillGraphError as exc:
            if exc.layer == FailureLayer.INFRASTRUCTURE:
                raise
            failure_code = exc.code
        finally:
            ctx.trace_builder.finish_span(span.span_id)
            self._finish_session(session_record, session)
        return {"success": success, "failure_code": failure_code, "rescue": rescue}
