"""Explicit graph entry, one node Agent loop, and task-level Dynamic agent."""

from __future__ import annotations

import inspect
import uuid
from typing import Any, Callable

from ..agents.context_builder import ContextBuilder
from ..agents.runtime_policy_projection import project_runtime_payload
from ..agents.structured_submission import RUNTIME_AUTOMATION_ATOMIC_SCHEMA
from ..agents.protocol import AgentTurn, NativeToolSpec, SchemaValidationError, validate_schema_instance
from ..core.bindings import (
    BindingResolution, BindingStatus, RuntimeBinding,
)
from ..core.errors import AtomicSkillGraphError, BudgetExhausted, FailureLayer
from ..core.refs import SkillRef
from ..core.results import (
    AtomicEffectResolution, ImplementationExecutionResult, NodeExecutionStatus,
    RuntimeOccurrence, ToolCallPreflightResult,
)
from ..core.serialization import to_primitive
from ..tooling.proposal import runtime_automation_draft_from_dict
from ..tooling.runtime_interface import (
    build_runtime_automation_interface,
)
from ..traces.schema import (
    AgentSessionRecord, AgentTurnRecord, EnvironmentActionRecord,
    NativeToolCallRecord, ValidationRecord,
)
from ..validation.engine import ValidationEngine
from ..validation.atomic_validator import AtomicValidator
from .automation import safe_runtime_automation_outcome
from .implementation_runner import ImplementationRunner
from .grounding_state import GivenInputReader
from .invocation_compiler import CompiledInvocation, InvocationCompiler
from .loop_guard import ActionLoopGuard
from .support_retriever import SupportAtomicRetriever


SessionFactory = Callable[[str, str], Any]
from ..agents.protocol import ONE_NATIVE_CALL_PREFIX as _ONE_NATIVE_CALL


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
        self.support_retriever = SupportAtomicRetriever()
        self.automation_coordinator = None
        self.grounding_authority = GivenInputReader(
            invocation_compiler, validation,
        )

    def not_started(self, occurrence: Any, *, failure_code: str,
                    failure_layer: str = "implementation") -> ImplementationExecutionResult:
        return ImplementationExecutionResult(
            "", str(occurrence.node_ref), False, False, False, False,
            failure_layer=failure_layer, failure_code=failure_code,
            node_status=NodeExecutionStatus.FAILED_NOT_STARTED,
        )

    def try_autonomous(self, occurrence, invocations, ctx):
        """Purely inspect all declared candidates, then execute at most one."""
        from .interventions import policy
        if not policy(ctx).automatic_entry or not policy(ctx).programs:
            return None
        from .invocation_transaction import execute_invocation
        ready, reasons = [], []
        for compiled in invocations:
            statuses = [compiled.implementation.status, *(t.status for t in compiled.tools)]
            if any(getattr(status, "value", status) not in {"active", "preferred"} for status in statuses):
                reasons.append({"implementation_ref": str(compiled.implementation.ref), "reason": "candidate_not_automatic"})
                continue
            checked = self.invocation_compiler.autonomous_preflight(
                compiled, occurrence, ctx.binding_store, ctx.evidence_store,
                ctx.world_revision, task_contract=ctx.task_contract)
            reasons.append({"implementation_ref": str(compiled.implementation.ref),
                            "reason": "ready" if checked.passed else checked.failure_code})
            if checked.passed:
                ready.append((compiled, checked))
        preferred = [item for item in ready if item[0].implementation.quality.get("preferred")
                     or getattr(item[0].implementation.status, "value", item[0].implementation.status) == "preferred"]
        choice = preferred[0] if len(preferred) == 1 else ready[0] if len(ready) == 1 else None
        ctx.trace_builder.trace.metadata.setdefault("node_entry_checks", []).append({
            "occurrence_id": occurrence.occurrence_id, "revision": ctx.world_revision,
            "candidates": reasons, "selected": str(choice[0].implementation.ref) if choice else None,
            "reason": "selected" if choice else "no_implementation" if not invocations
                else "ambiguous_implementations" if len(ready) > 1 else "no_ready_active_implementation"})
        if choice is None:
            return None
        compiled, checked = choice
        before = len(ctx.trace_builder.trace.implementation_invocations)
        result = execute_invocation(self.implementation_runner, compiled, checked,
                                    occurrence, ctx, agent_prepared=False)
        for record in ctx.trace_builder.trace.implementation_invocations[before:]:
            ctx.trace_builder.trace.metadata.setdefault("graph_entry_invocations", []).append({
                "attempt_id": record.attempt_id, "occurrence_id": occurrence.occurrence_id,
                "implementation_ref": str(compiled.implementation.ref)})
        return result

    def _environment_tool(
        self, ctx: Any, *, node_level: bool = True, atomic: Any = None,
    ) -> NativeToolSpec:
        properties: dict[str, Any] = {
            "action_id": {
                "type": "string",
                "enum": [item.action_id for item in ctx.action_catalog],
            },
        }
        required = ["action_id"]
        description = (
            _ONE_NATIVE_CALL
            + "Execute one currently available environment action."
        )
        if node_level:
            properties["intent"] = {
                "type": "string",
                "enum": ["explore", "attempt_current_atomic"],
            }
            if atomic is not None:
                properties.update(self._completion_candidate_schema(atomic))
            required.append("intent")
            description += (
                " Set intent=explore for evidence gathering/preparation, or "
                "intent=attempt_current_atomic only when asking the Runtime "
                "to validate this action as completion of the current Atomic."
            )
        return NativeToolSpec(
            "environment_action", description,
            {
                "type": "object", "required": required,
                "additionalProperties": False, "properties": properties,
            },
            call_kind='environment_action', scope='node' if atomic is not None else 'task',
            result_owner='environment_and_validator',
        )

    @staticmethod
    def _status_tool(*, allow_plan_conflict: bool = False) -> NativeToolSpec:
        statuses = ["cannot_resolve", "give_up"]
        if allow_plan_conflict:
            statuses.insert(1, "plan_conflict")
        description = (
            _ONE_NATIVE_CALL
            + "Explicitly report why the current mode cannot continue. "
            "cannot_resolve means the current occurrence may still be valid, "
            "but public evidence is insufficient or search is incomplete; "
            "give_up terminates this route without asserting a formal plan conflict."
        )
        if allow_plan_conflict:
            description += (
                " plan_conflict means the current formal occurrence, hard semantic "
                "anchor, or downstream obligation conflicts with public evidence, so "
                "the same rigid graph cannot solve the task; only the Agent may declare it."
            )
        return NativeToolSpec(
            "report_runtime_status", description,
            {
                "type": "object", "required": ["status"], "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": statuses},
                    "detail": {
                        "type": "string",
                        "maxLength": 512,
                        "description": (
                            "Optional short diagnostic only; never formal binding "
                            "or validation authority."
                        ),
                    },
                },
            },
            call_kind='runtime_status', scope='node' if allow_plan_conflict else 'task',
            result_owner='runtime_dispatcher',
        )

    @staticmethod
    def _policy_catalog(catalog: Any, revision: Any) -> dict[str, Any]:
        """Return the frozen compact policy representation of one catalog."""

        return {
            "revision": revision,
            "actions": [
                {
                    "action_id": item.action_id,
                    "action_type": item.action_type,
                    "arguments": dict(item.arguments),
                }
                for item in catalog
            ],
        }

    @staticmethod
    def _policy_budget(ctx: Any) -> dict[str, int]:
        snapshot = dict(ctx.budget.snapshot())
        result = {
            "remaining_global_actions": max(
                0, int(snapshot.get("remaining_global_actions", 0))
            ),
        }
        result["node_actions_used"] = int(snapshot.get("used_node_actions", 0))
        return result

    @staticmethod
    def _task_progress_policy(ctx: Any) -> dict[str, Any]:
        tracker = getattr(ctx, "task_progress", None)
        policy_view = getattr(tracker, "policy_view", None)
        return dict(policy_view()) if callable(policy_view) else {}

    def _downstream_plan_context(
        self,
        ctx: Any,
        occurrence: Any,
        *,
        plan_context_plan: Any | None = None,
        producer_output_candidates: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        builder = getattr(self, "plan_context_builder", None)
        plan = plan_context_plan or getattr(ctx, "plan", None)
        if builder is None or plan is None:
            return {}
        try:
            public_relation_source = getattr(
                ctx.harness, "public_runtime_relation_facts", None,
            )
            public_relation_facts = (
                list(public_relation_source())
                if callable(public_relation_source)
                else []
            )
            build_kwargs: dict[str, Any] = {}
            try:
                parameters = inspect.signature(builder.build).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_keywords = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_keywords or "public_facts" in parameters:
                build_kwargs["public_facts"] = tuple(
                    dict(item)
                    for item in public_relation_facts
                    if isinstance(item, dict)
                    and str(item.get("public_evidence_ref", "")).strip()
                )
            if accepts_keywords or "public_revision" in parameters:
                build_kwargs["public_revision"] = int(
                    getattr(ctx, "world_revision", 0)
                )
            if accepts_keywords or "producer_output_candidates" in parameters:
                candidates = {}
                skills = getattr(builder, "skills", None)
                atomic = skills.get_atomic(occurrence.node_ref) if skills else None
                bindings = ctx.binding_store.snapshot_for_node(occurrence)
                derivations = AtomicValidator._output_derivations(atomic) if atomic else {}
                for role, derivation in derivations.items():
                    if derivation.get("kind") != "input_identity":
                        continue
                    binding = bindings.get(derivation.get("input_role"))
                    if binding is not None and binding.status is BindingStatus.GROUNDED:
                        candidates[role] = {
                            "value": binding.value,
                            "source": "input_identity",
                            "revision": binding.world_revision,
                            "resolution": binding.resolution.value,
                        }
                candidates.update(producer_output_candidates or {})
                build_kwargs["producer_output_candidates"] = candidates
            if accepts_keywords or "public_action_catalog" in parameters:
                build_kwargs["public_action_catalog"] = tuple(
                    to_primitive(action) for action in getattr(ctx, "action_catalog", ())
                )
            return dict(builder.build(
                plan,
                occurrence.step_id,
                ctx.binding_store,
                **build_kwargs,
            ).policy_view())
        except KeyError:
            # Provisional/test-only occurrences are not verified plan nodes.
            # Fail closed rather than manufacturing downstream intent.
            return {}

    @staticmethod
    def _atomic_policy_contract(atomic: Any) -> dict[str, Any]:
        from ..agents.portable_support_view import portable_support_view
        atomic = portable_support_view(atomic)
        primitive = dict(to_primitive(atomic))
        from ..agents.skill_guidance import guidance_view
        primitive["skill_guidance"] = guidance_view(atomic)
        return {
            key: primitive[key]
            for key in (
                "summary", "inputs", "outputs", "preconditions", "effects", "skill_guidance",
            )
            if key in primitive
        }

    def _current_state_snapshot(
        self,
        occurrence: Any,
        atomic: Any,
        ctx: Any,
        *,
        plan_context_plan: Any | None = None,
    ) -> dict[str, Any]:
        """Build the fixed-order policy projection from code-owned state."""

        state = dict(
            getattr(ctx, "grounding_state_by_occurrence", {}).get(
                occurrence.occurrence_id, {}
            )
        )
        downstream = self._downstream_plan_context(
            ctx, occurrence, plan_context_plan=plan_context_plan,
        )
        return {
            "current_atomic": self._atomic_policy_contract(atomic),
            "semantic_anchors": dict(state.get("semantic_anchors") or {}),
            "confirmed_bindings": dict(state.get("confirmed_bindings") or {}),
            "candidate_bindings": dict(state.get("candidate_bindings") or {}),
            "missing_bindings": list(state.get("missing_bindings") or []),
            "invalidated_bindings": dict(
                state.get("invalidated_bindings") or {}
            ),
            "preconditions": list(state.get("precondition_status") or []),
            "effect_witness_status": dict(
                state.get("effect_witness_status") or {}
            ),
            "learned_invocation_ready": bool(
                state.get("learned_invocation_ready", False)
            ),
            "blocking_reasons": list(state.get("blocking_reasons") or []),
            "downstream_obligations": downstream,
            "remaining_budget": self._policy_budget(ctx),
        }

    def _refresh_occurrence_state(
        self,
        occurrence: Any,
        atomic: Any,
        invocations: list[CompiledInvocation],
        ctx: Any,
        *,
        plan_context_plan: Any | None = None,
    ) -> dict[str, Any]:
        self.grounding_authority.refresh(
            occurrence,
            atomic,
            list(invocations),
            ctx,
        )
        return self._current_state_snapshot(
            occurrence,
            atomic,
            ctx,
            plan_context_plan=plan_context_plan,
        )

    def _activate_occurrence_state(
        self,
        occurrence: Any,
        atomic: Any,
        invocations: list[CompiledInvocation],
        ctx: Any,
        *,
        plan_context_plan: Any | None = None,
    ) -> dict[str, Any]:
        begin = getattr(ctx, "begin_occurrence", None)
        if callable(begin):
            begin(occurrence)

        def refresh() -> None:
            self._refresh_occurrence_state(
                occurrence,
                atomic,
                list(invocations),
                ctx,
                plan_context_plan=plan_context_plan,
            )

        install = getattr(ctx, "install_after_action_refresh", None)
        if callable(install):
            install(refresh)
        return self._refresh_occurrence_state(
            occurrence,
            atomic,
            list(invocations),
            ctx,
            plan_context_plan=plan_context_plan,
        )

    def _augment_runtime_payload(
        self,
        payload: dict[str, Any],
        ctx: Any,
        *,
        occurrence: Any | None = None,
        atomic: Any | None = None,
        plan_context_plan: Any | None = None,
        session_id: str = "",
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        audit_occurrence_id = (
            "" if occurrence is None else str(occurrence.occurrence_id)
        )
        if "draft_id" in payload:
            # A trial can advance several revisions before returning control.
            # Refresh even cached/rejected automation replies from the current
            # public surface; never replay their previous catalog snapshot.
            payload.update(
                new_revision=ctx.world_revision,
                observation=ctx.observation,
                action_catalog=self._policy_catalog(ctx.action_catalog, ctx.world_revision),
            )
        if occurrence is not None and atomic is not None:
            payload["current_state_snapshot"] = self._current_state_snapshot(
                occurrence,
                atomic,
                ctx,
                plan_context_plan=plan_context_plan,
            )
            payload["recent_failed_learned_invocation"] = (
                dict(ctx.last_failed_invocation)
                if getattr(ctx, "last_failed_invocation", None)
                and ctx.last_failed_invocation.get("occurrence_id")
                == occurrence.occurrence_id
                else None
            )
            occurrence_id = occurrence.occurrence_id
        else:
            occurrence_id = ""
            payload["current_state_snapshot"] = {
                "task_progress": self._task_progress_policy(ctx),
                "remaining_budget": self._policy_budget(ctx),
            }
            payload["recent_failed_learned_invocation"] = (
                dict(ctx.last_failed_invocation)
                if getattr(ctx, "last_failed_invocation", None)
                else None
            )
        memory = getattr(ctx, "exploration_memory", None)
        payload["exploration_memory"] = (
            memory.policy_view() if memory is not None else {}
        )
        payload["recent_accepted_actions"] = ctx.relevant_history(
            occurrence_id,
        ) if hasattr(ctx, "relevant_history") else []
        projected, audit = project_runtime_payload(payload)
        self._record_runtime_context_projection(
            ctx,
            audit,
            session_id=session_id,
            occurrence_id=audit_occurrence_id,
            origin="tool_result",
            tool_call_id=tool_call_id,
        )
        # Some callers intentionally ignore this method's return value.  Keep
        # the historical in-place augmentation contract while replacing only
        # the policy-facing message dictionary with its projected deep copy.
        payload.clear()
        payload.update(projected)
        return payload

    @staticmethod
    def _record_runtime_context_projection(
        ctx: Any,
        audit: dict[str, Any],
        *,
        session_id: str,
        occurrence_id: str,
        origin: str,
        tool_call_id: str = "",
    ) -> None:
        trace = ctx.trace_builder.trace
        metadata = getattr(trace, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(trace, "metadata", metadata)
        metadata["runtime_context_projection_version"] = "v3.2-r5"
        raw_revision = getattr(ctx, "world_revision", None)
        record = dict(to_primitive(audit))
        record.update({
            "projection_version": "v3.2-r5",
            "session_id": session_id or None,
            "occurrence_id": occurrence_id,
            "revision": None if raw_revision is None else int(raw_revision),
            "origin": origin,
            "tool_call_id": tool_call_id or None,
        })
        if "expression_audit" in record:
            record["expression_audit"].update(session_id=session_id or None,
                occurrence_id=occurrence_id, origin=origin, tool_call_id=tool_call_id or None)
        audits = metadata.setdefault("runtime_context_projection_audits", [])
        # Node-level environment results are augmented once on return from the
        # harness and may be augmented again after Atomic validation is added.
        # Retain the final view that can actually be submitted, not an internal
        # intermediate snapshot for the same native ToolCall.
        if session_id and tool_call_id:
            for index in range(len(audits) - 1, -1, -1):
                existing = audits[index]
                if (
                    existing.get("session_id") == session_id
                    and existing.get("tool_call_id") == tool_call_id
                    and existing.get("origin") == origin
                ):
                    audits[index] = record
                    break
            else:
                audits.append(record)
        else:
            audits.append(record)

    @staticmethod
    def _rescue_method_guidance(ctx: Any) -> dict[str, Any] | None:
        conflict = dict(getattr(ctx, "plan_conflict_context", {}) or {})
        if not conflict:
            return None
        guidance = {
            "conflict_step_summary": str(
                conflict.get("conflict_step_summary", "")
            ),
            "conflict_code": str(
                conflict.get("conflict_code", "runtime_plan_conflict")
            ),
            "remaining_method_outline": list(
                conflict.get("remaining_method_outline", ())
            ),
        }
        message = str(
            conflict.get("detail")
            or conflict.get("last_preflight_failure_code")
            or ""
        )
        if message:
            guidance["conflict_message"] = message
        return guidance

    @staticmethod
    def _completion_candidate_schema(atomic):
        from ..tooling.entry_contract import parameter_schema
        inputs = parameter_schema(atomic.inputs)
        outputs = parameter_schema(atomic.outputs)
        inputs["required"] = []
        outputs["required"] = []
        return {"candidate_bindings": inputs, "candidate_outputs": outputs}

    @staticmethod
    def _validate_current_atomic_tool(atomic: Any) -> NativeToolSpec:
        return NativeToolSpec(
            "validate_current_atomic",
            _ONE_NATIVE_CALL + "Submit explicit input/output candidates for current node completion. "
            "Candidates must match actual evidence and formal identities. Missing fresh outputs "
            "are not inferred. Skill guidance is optional; the structured contract is binding.",
            {"type": "object", "properties": NodeExecutor._completion_candidate_schema(atomic),
             "required": [], "additionalProperties": False},
            call_kind='atomic_validation', scope='node', result_owner='atomic_validator',
        )

    @staticmethod
    def _invocation_tool(item: CompiledInvocation) -> NativeToolSpec:
        return NativeToolSpec(
            item.spec.name,
            _ONE_NATIVE_CALL + item.spec.description,
            item.spec.input_schema,
            call_kind='implementation', scope='node', result_owner='implementation_and_validator',
        )

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
            [item.call_id for item in turn.tool_calls], usage,
            dict(turn.provider_metadata), turn.reasoning_content,
        ))
        ctx.trace_builder.trace.llm_usage.append({"session_id": session.session_id, **usage})

    def _finish_session(
        self, record: AgentSessionRecord, session: Any, ctx: Any,
    ) -> None:
        import time
        record.ended_at = time.time()
        record.snapshot = session.snapshot()
        events = record.snapshot.get("r3_events", [])
        if isinstance(events, list) and events:
            ctx.trace_builder.trace.metadata.setdefault("r3_events", []).extend(
                to_primitive(events)
            )

    @staticmethod
    def _finalize_tool_result(session: Any, call_id: str, result: dict[str, Any], tools: list[NativeToolSpec]) -> None:
        finalize = getattr(session, "finalize_tool_result", None)
        if callable(finalize):
            finalize(call_id, result)
        else:
            # Deterministic test sessions may implement terminal submission
            # without issuing a provider call.
            session.submit_tool_result(
                call_id,
                result,
                tools=tools,
                returned_action_executed=False,
            )

    def _execute_environment_call(
        self, call: Any, session: Any, occurrence: Any | None, ctx: Any,
        *, span_id: str, origin: str, loop_guard: ActionLoopGuard,
        atomic: Any | None = None,
        plan_context_plan: Any | None = None,
    ) -> tuple[dict[str, Any], Any]:
        if (occurrence is not None and call.arguments.get("intent") == "explore"
                and (call.arguments.get("candidate_bindings") or call.arguments.get("candidate_outputs"))):
            return {"accepted": False, "failure_code": "explore_cannot_submit_completion",
                    "message": "Completion candidates require attempt_current_atomic or validate_current_atomic"}, None
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
        intent = str(call.arguments.get("intent", ""))
        if occurrence is not None and intent not in {
            "explore", "attempt_current_atomic",
        }:
            raise AtomicSkillGraphError(
                "runtime_agent_schema_error",
                "node-level environment_action requires an explicit valid intent",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        if occurrence is not None and intent == "attempt_current_atomic":
            repeat_values = {
                role: binding.value
                for role, binding in ctx.binding_store.snapshot_for_node(
                    occurrence,
                ).items()
                if binding.status is BindingStatus.GROUNDED
            }
            repeat_values.update(dict(call.arguments.get("candidate_bindings") or {}))
            # RuntimeOccurrence always carries step_id.  A small number of
            # adapter-level callers use the historical occurrence-shaped
            # object that only exposes occurrence_id; retain that boundary
            # compatibility without weakening repeat preflight for real plans.
            repeat_step_id = str(
                getattr(
                    occurrence,
                    "step_id",
                    getattr(occurrence, "occurrence_id", ""),
                )
            )
            repeat_preflight = ctx.binding_store.preflight_repeat_bindings(
                repeat_step_id, repeat_values,
            )
            if not repeat_preflight.passed:
                code = (
                    repeat_preflight.failure_codes[0]
                    if repeat_preflight.failure_codes
                    else "runtime_repetition_distinctness_violation"
                )
                payload = {
                    "loop_blocked": True,
                    # A rejected Repeat identity is a typed preflight result,
                    # not a terminal loop-guard fallback.  The same Agent
                    # session must be allowed to choose another candidate.
                    "fallback_required": False,
                    "repeat_preflight_rejected": True,
                    "error": code,
                    "observation": ctx.observation,
                    "done": False,
                    "won": False,
                    "new_revision": ctx.world_revision,
                    "remaining_budget": self._policy_budget(ctx),
                }
                ctx.trace_builder.trace.native_tool_calls.append(
                    NativeToolCallRecord(
                        call.call_id,
                        session.session_id,
                        occurrence.occurrence_id,
                        call.name,
                        dict(call.arguments),
                        "environment_action",
                        to_primitive(repeat_preflight),
                        f"revision:{ctx.world_revision}",
                        sum(
                            1 for item in ctx.trace_builder.trace.agent_turns
                            if item.session_id == session.session_id
                        ) - 1,
                    )
                )
                return self._augment_runtime_payload(
                    payload,
                    ctx,
                    occurrence=occurrence,
                    atomic=atomic,
                    plan_context_plan=plan_context_plan,
                    session_id=session.session_id,
                    tool_call_id=call.call_id,
                ), spec
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
                "remaining_budget": self._policy_budget(ctx),
            }
            if occurrence is None:
                payload["task_progress"] = self._task_progress_policy(ctx)
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
            return self._augment_runtime_payload(
                payload,
                ctx,
                occurrence=occurrence,
                atomic=atomic,
                plan_context_plan=plan_context_plan,
                session_id=session.session_id,
                tool_call_id=call.call_id,
            ), spec
        ctx.budget.consume_action()
        result = ctx.harness.execute_action(action_id, spec.revision)
        record = EnvironmentActionRecord(
            spec.action_id, spec.revision, spec.action_type, dict(spec.arguments), result.accepted,
            result.observation, result.done, result.won, result.new_revision, span_id,
        )
        ctx.trace_builder.trace.environment_actions.append(record)
        occurrence_id = "" if occurrence is None else occurrence.occurrence_id
        ctx.update_after_action(result, {**to_primitive(record), "occurrence_id": occurrence_id, "origin": origin})
        payload = {
            "accepted": result.accepted, "observation": result.observation, "done": result.done,
            "won": result.won, "new_revision": result.new_revision,
            # A revision changes the meaning of action ids.  The next Agent
            # turn receives the complete new policy-facing catalog.  Keep the
            # replay representation compact while preserving the same public,
            # canonical affordance fields exposed in the initial policy
            # context.  Learned invocation arguments must remain copyable
            # after exploration advances the world revision.
            "action_catalog": self._policy_catalog(
                result.catalog, result.new_revision,
            ),
            "remaining_budget": self._policy_budget(ctx),
        }
        if occurrence is not None:
            payload["intent"] = intent
        else:
            payload["task_progress"] = self._task_progress_policy(ctx)
        ctx.trace_builder.trace.native_tool_calls.append(NativeToolCallRecord(
            call.call_id, session.session_id, occurrence_id, call.name, dict(call.arguments),
            "environment_action", {
                "passed": bool(result.accepted),
                "harness_accepted": bool(result.accepted),
                **({"intent": intent} if occurrence is not None else {}),
            }, f"revision:{result.new_revision}",
            sum(1 for item in ctx.trace_builder.trace.agent_turns if item.session_id == session.session_id) - 1,
        ))
        return self._augment_runtime_payload(
            payload,
            ctx,
            occurrence=occurrence,
            atomic=atomic,
            plan_context_plan=plan_context_plan,
            session_id=session.session_id,
            tool_call_id=call.call_id,
        ), spec

    def _complete_from_current_effect(
        self,
        occurrence: Any,
        ctx: Any,
        *,
        mode: str,
        preferred_values: list[Any],
        preferred_bindings: dict[str, Any] | None = None,
        candidate_outputs: dict[str, Any] | None = None,
        provisional_bindings: list[RuntimeBinding] = (),
        atomic_override: Any | None = None,
        resolution_out: list[AtomicEffectResolution] | None = None,
        effect_guard: Any = None,
    ) -> ImplementationExecutionResult | None:
        atomic = atomic_override or self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        bindings = ctx.binding_store.snapshot_for_node(occurrence)
        bindings.update({
            item.role: item
            for item in provisional_bindings
            if item.status is BindingStatus.GROUNDED
        })
        # Authenticate only Agent-supplied values; checking is pure.
        from ..agents.protocol import validate_schema_instance
        from ..tooling.entry_contract import parameter_schema
        from ..core.bindings import GroundingConstraint, GroundingConstraintKind, BindingExpression, BindingExprKind
        claims = dict(preferred_bindings or {})
        rejection = None
        try:
            schema = parameter_schema(atomic.inputs)
            schema["required"] = []
            validate_schema_instance(claims, schema)
        except (ValueError, TypeError) as exc:
            rejection = AtomicEffectResolution(False, failure_code="runtime_agent_schema_error", message=str(exc))
        for parameter in atomic.inputs:
            if rejection is not None or parameter.name not in claims:
                continue
            value = claims[parameter.name]
            anchor = ctx.binding_store.semantic_anchor_for(occurrence, parameter.name)
            if parameter.required_resolution == "semantic" and anchor is not None and anchor.value == value:
                continue
            constraint = GroundingConstraint("completion_" + parameter.name,
                GroundingConstraintKind.ARGUMENT_CONCRETE, argument_mapping={
                    parameter.name: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=parameter.name)})
            if not ctx.evidence_store.match_constraint(constraint, claims, ctx.world_revision):
                rejection = AtomicEffectResolution(False, failure_code="runtime_binding_not_concrete",
                    message=f"No certified identity for supplied input {parameter.name}")
        if rejection is not None:
            if resolution_out is not None:
                resolution_out.append(rejection)
            return None
        semantic_anchors = {
            parameter.name: anchor
            for parameter in atomic.inputs
            if (
                anchor := ctx.binding_store.semantic_anchor_for(
                    occurrence,
                    parameter.name,
                )
            ) is not None
        }
        try:
            authoritative_evidence_facts = ctx.atomic_evidence_for(
                occurrence
            ).authoritative_facts()
        except (AttributeError, KeyError):
            authoritative_evidence_facts = []
        resolution = self.validation.atomic.resolve_current_effect(
            atomic,
            occurrence,
            bindings,
            ctx.harness.validator_channel(),
            semantic_anchors=semantic_anchors,
            preferred_values=preferred_values,
            preferred_bindings=preferred_bindings,
            candidate_outputs=candidate_outputs,
            semantic_compatible=getattr(ctx.harness, "semantic_value_compatible", None),
            current_revision=ctx.world_revision,
            authoritative_evidence_facts=authoritative_evidence_facts,
        )
        # Assess pre-publication candidates without committing outputs, Repeat
        # identity or new evidence. Only current public values are projected.
        candidate_context = self._downstream_plan_context(
            ctx, occurrence,
            producer_output_candidates={
                parameter.name: {
                    "value": resolution.output_candidates[parameter.name],
                    "source": "effect_resolution",
                    "revision": ctx.world_revision,
                    # A contract's minimum resolution is not the evidence
                    # level of this actual witness. Public visibility is still
                    # checked independently by the downstream context builder.
                    "resolution": "concrete" if resolution.passed else "semantic",
                }
                for parameter in atomic.outputs
                if parameter.name in resolution.output_candidates
            },
        )
        if resolution.output_candidates and candidate_context.get("output_obligations"):
            ctx.trace_builder.trace.metadata.setdefault(
                "producer_output_candidate_assessments", [],
            ).append({
                "occurrence_id": occurrence.occurrence_id,
                "revision": ctx.world_revision,
                "before_output_publication": True,
                "context": candidate_context,
            })
        if not resolution.passed:
            if resolution_out is not None:
                resolution_out.append(resolution)
                ctx.trace_builder.trace.validations.append(ValidationRecord(
                    occurrence.occurrence_id,
                    "atomic",
                    to_primitive(resolution),
                    ctx.world_revision,
                ))
            return None
        repeat_effect_values = {
            **dict(resolution.resolved_bindings),
            **dict(resolution.output_candidates),
        }
        for constraint in getattr(getattr(ctx, "task_contract", None), "identity_constraints", ()):
            if constraint.scope != "occurrence" or not {
                constraint.left_role, constraint.right_role
            } <= repeat_effect_values.keys():
                continue
            equal = repeat_effect_values[constraint.left_role] == repeat_effect_values[constraint.right_role]
            if (constraint.relation.value == "same_as" and not equal) or (
                constraint.relation.value == "distinct_from" and equal
            ):
                if resolution_out is not None:
                    resolution_out.append(AtomicEffectResolution(False,
                        failure_code="runtime_identity_constraint_mismatch",
                        message="Completion candidates violate occurrence identity constraints"))
                return None
        repeat_preflight = ctx.binding_store.preflight_repeat_bindings(
            occurrence.step_id,
            repeat_effect_values,
        )
        ctx.trace_builder.trace.validations.append(ValidationRecord(
            occurrence.occurrence_id,
            "runtime_repeat_preflight",
            to_primitive(repeat_preflight),
            ctx.world_revision,
        ))
        if not repeat_preflight.passed:
            if resolution_out is not None:
                resolution_out.append(AtomicEffectResolution(
                    False,
                    resolved_bindings=dict(resolution.resolved_bindings),
                    output_candidates=dict(resolution.output_candidates),
                    witness_refs=list(resolution.witness_refs),
                    checks=dict(resolution.checks),
                    failure_code=(
                        repeat_preflight.failure_codes[0]
                        if repeat_preflight.failure_codes
                        else "runtime_repetition_distinctness_violation"
                    ),
                    message=(
                        repeat_preflight.messages[0]
                        if repeat_preflight.messages
                        else "RepeatBlock preflight rejected the Atomic witness"
                    ),
                ))
            return None
        if effect_guard is not None and not effect_guard(resolution).passed:
            return None
        committed = ctx.binding_store.commit_atomic_effect_witnesses(
            occurrence.occurrence_id,
            resolution.resolved_bindings,
            atomic.inputs,
            resolution.witness_refs,
            ctx.world_revision,
            certified_bindings=resolution.certified_input_bindings,
        )
        repeat_commit = ctx.binding_store.commit_repeat_bindings(
            occurrence.step_id,
            repeat_effect_values,
            effect_passed=True,
        )
        ctx.trace_builder.trace.validations.append(ValidationRecord(
            occurrence.occurrence_id,
            "runtime_repeat_commit",
            to_primitive(repeat_commit),
            ctx.world_revision,
        ))
        if resolution_out is not None:
            resolution_out.append(resolution)
        if not repeat_commit.passed:
            raise AtomicSkillGraphError(
                repeat_commit.failure_codes[0],
                "validated Atomic Effect could not commit RepeatBlock bindings",
                layer=FailureLayer.RUNTIME_BINDING,
            )
        ctx.trace_builder.trace.validations.append(ValidationRecord(
            occurrence.occurrence_id,
            "atomic",
            to_primitive(resolution),
            ctx.world_revision,
        ))
        if mode == "entry":
            # Preserve the explicit entry audit while retaining the ordinary
            # positive Atomic record consumed by evolution and credit logic.
            ctx.trace_builder.trace.validations.append(ValidationRecord(
                occurrence.occurrence_id,
                "already_satisfied",
                to_primitive(resolution),
                ctx.world_revision,
            ))
        if hasattr(ctx, "grounding_state_by_occurrence"):
            self._refresh_occurrence_state(
                occurrence,
                atomic,
                [],
                ctx,
                )
        clear_failed = getattr(ctx, "clear_failed_invocation", None)
        if callable(clear_failed):
            clear_failed(occurrence.occurrence_id)
        status = (
            NodeExecutionStatus.ALREADY_SATISFIED
            if mode == "entry"
            else NodeExecutionStatus.AGENT_COMPLETED_BEFORE_INVOCATION
            if mode == "preparation"
            else NodeExecutionStatus.SEEDED_SUCCESS
        )
        return ImplementationExecutionResult(
            "",
            str(atomic.ref),
            False,
            False,
            True,
            True,
            realized_bindings=committed,
            validated_outputs=dict(resolution.output_candidates),
            validated_output_bindings=dict(resolution.validated_output_bindings),
            certified_input_bindings=dict(resolution.certified_input_bindings),
            atomic_witness_refs=list(resolution.witness_refs),
            before_state_ref="",
            after_state_ref=f"revision:{ctx.world_revision}",
            node_status=status,
        )

    def _node_tools(
        self,
        ctx: Any,
        atomic: Any,
        *,
        invocations: list[CompiledInvocation] = (),
        allow_plan_conflict: bool = False,
        support_candidates: list[Any] = (),
        support_call_surface: Any = None,
    ) -> list[NativeToolSpec]:
        tools = [
            self._environment_tool(ctx, node_level=True, atomic=atomic),
            self._validate_current_atomic_tool(atomic),
            *[self._invocation_tool(item) for item in invocations],
            self._automation_tool(),
            self._status_tool(allow_plan_conflict=allow_plan_conflict),
        ]
        if support_call_surface is not None:
            tool = support_call_surface.native_tool()
            if tool is not None:
                tools.insert(2, tool)
        elif support_candidates:
            tools.insert(2, self._support_tool(list(support_candidates)))
        return tools

    @staticmethod
    def _increment_funnel(
        ctx: Any,
        funnel_name: str,
        field: str,
        amount: int = 1,
    ) -> None:
        trace = getattr(getattr(ctx, "trace_builder", None), "trace", None)
        metadata = getattr(trace, "metadata", None)
        if not isinstance(metadata, dict):
            return
        funnel = metadata.setdefault(funnel_name, {})
        funnel[field] = int(funnel.get(field, 0)) + int(amount)

    def _support_execution_availability(
        self,
        atomics: list[Any],
        ctx: Any = None,
    ) -> dict[str, bool]:
        """Check the same mode-compatible implementation/tool boundary used at call time."""

        skills = self.invocation_compiler.skills
        implementations_for = getattr(skills, "implementations_for", None)
        if not callable(implementations_for):
            # Narrow test registries predate this projection. Production
            # SkillRegistry always provides the executable authority method.
            return {str(item.ref): True for item in atomics}
        result: dict[str, bool] = {}
        self._r103_support_display_routes = {}
        for atomic in atomics:
            if ctx is not None and getattr(self.invocation_compiler, "r103", False):
                # Same empty helper occurrence and original compiler as the
                # explicit call. This is a read-only probe, not a new binding.
                occurrence = RuntimeOccurrence(
                    step_id="support_availability", occurrence_id="support_availability",
                    node_ref=atomic.ref, requirement_ids=[], binding_specs={},
                    implementation_candidates=[str(i.ref) for i in implementations_for(
                        atomic.ref, mode=self.invocation_compiler.mode)],
                    expected_effects=list(atomic.effects))
                routes = self.invocation_compiler.compile_candidates(occurrence, ctx.binding_store,
                    task_id=ctx.task_id, evidence_store=ctx.evidence_store,
                    revision=ctx.world_revision, task_contract=ctx.task_contract)
                preferred = [r for r in routes if r.implementation.quality.get("preferred")]
                selected = preferred if len(preferred) == 1 else routes if len(routes) == 1 else []
                result[str(atomic.ref)] = bool(selected)
                self._r103_support_display_routes[str(atomic.ref)] = selected
                continue
            available = False
            for implementation in implementations_for(
                atomic.ref,
                mode=self.invocation_compiler.mode,
            ):
                try:
                    tools = [
                        self.invocation_compiler.tools.get(binding.tool_ref)
                        for binding in sorted(
                            implementation.tool_bindings,
                            key=lambda item: item.order,
                        )
                    ]
                    self.invocation_compiler.compile(
                        atomic,
                        implementation,
                        tools,
                        {},
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                available = True
                break
            result[str(atomic.ref)] = available
        return result

    def _record_support_display(self, ctx, session_id, candidates, native_tools):
        if not getattr(self.invocation_compiler, "r103", False):
            return
        routes = getattr(self, "_r103_support_display_routes", {})
        rows = ctx.trace_builder.trace.metadata.get('support_call_surfaces', [])
        surface = next((row for row in reversed(rows) if row['session_id'] == session_id), None)
        selected = set(surface['selectable_atomic_refs']) if surface else None
        for candidate in candidates:
            if selected is not None and candidate.atomic_ref not in selected:
                continue
            self.invocation_compiler.record_display(ctx, session_id,
                routes.get(str(candidate.atomic_ref), []), native_tools,
                native_name="invoke_support_atomic")

    def _retrieve_runtime_support_candidates(
        self,
        *,
        blocked_atomic: Any,
        missing_roles: list[str],
        ctx: Any,
        obligations: tuple[Any, ...] = (),
        occurrence: Any = None,
    ) -> list[Any]:
        atomics_method = getattr(self.invocation_compiler.skills, "atomics", None)
        if not callable(atomics_method):
            return []
        mode_pool = list(atomics_method(mode=self.invocation_compiler.mode))
        try:
            all_pool = list(atomics_method())
        except TypeError:
            all_pool = list(mode_pool)
        all_compatible = self.support_retriever.retrieve(
            obligations=obligations,
            blocked_atomic=blocked_atomic,
            missing_roles=missing_roles,
            atomics=all_pool,
            top_k=None,
        )
        mode_compatible = self.support_retriever.retrieve(
            obligations=obligations,
            blocked_atomic=blocked_atomic,
            missing_roles=missing_roles,
            atomics=mode_pool,
            top_k=None,
        )
        availability = self._support_execution_availability(mode_pool, ctx)
        executable = self.support_retriever.retrieve(
            obligations=obligations,
            blocked_atomic=blocked_atomic,
            missing_roles=missing_roles,
            atomics=mode_pool,
            execution_availability=availability,
            top_k=None,
        )
        all_refs = {str(item.atomic_ref) for item in all_compatible}
        mode_refs = {str(item.atomic_ref) for item in mode_compatible}
        executable_refs = {str(item.atomic_ref) for item in executable}
        if getattr(self, 'support_interface_version', None):
            for name, value in {'support_related_candidates': len(all_refs),
                    'support_program_available_candidates': len(executable_refs),
                    'support_blocked_no_program': len(mode_refs - executable_refs)}.items():
                self._increment_funnel(ctx, 'runtime_support_funnel', name, value)
        self._increment_funnel(
            ctx,
            "runtime_support_funnel",
            "retrieved_count",
            len(all_refs),
        )
        self._increment_funnel(
            ctx,
            "runtime_support_funnel",
            "filtered_by_mode_count",
            len(all_refs - mode_refs),
        )
        self._increment_funnel(
            ctx,
            "runtime_support_funnel",
            "filtered_no_executable_count",
            len(mode_refs - executable_refs),
        )
        if occurrence is not None and getattr(self.invocation_compiler, "r103", False):
            from .support_interface import project_candidate
            routes = getattr(self, "_r103_support_display_routes", {})
            executable = [project_candidate(c, self.invocation_compiler.skills.get_atomic(c.atomic_ref),
                blocked_atomic, occurrence, ctx, self.invocation_compiler, routes.get(c.atomic_ref, []))
                for c in executable]
            executable.sort(key=lambda c: (-c.score, c.atomic_ref))
        displayed = executable if getattr(self, 'support_interface_version', None) else executable[:3]
        if displayed:
            metrics = ctx.trace_builder.trace.metadata.setdefault(
                "v32_metrics", {},
            )
            metrics["runtime_support_retrieval_count"] = int(
                metrics.get("runtime_support_retrieval_count", 0)
            ) + 1
            metrics["runtime_support_candidate_count"] = int(
                metrics.get("runtime_support_candidate_count", 0)
            ) + len(displayed)
        return displayed

    def _build_support_surface(self, candidates, atomic, occurrence, ctx, session_id):
        if not getattr(self, 'support_interface_version', None):
            return None
        from .support_call_surface import build_surface
        surface = build_surface(candidates, skills=self.invocation_compiler.skills,
            routes=getattr(self, '_r103_support_display_routes', {}), consumer=atomic,
            occurrence=occurrence, ctx=ctx, session_id=session_id)
        ctx.trace_builder.trace.metadata.setdefault('support_call_surfaces', []).append({
            'version': surface.version, 'session_id': session_id, 'scope': surface.scope,
            'consumer_occurrence_id': occurrence.occurrence_id, 'revision': surface.revision,
            'surface_context_fingerprint': surface.surface_context_fingerprint,
            'selectable_atomic_refs': list(dict.fromkeys(o.atomic_ref for o in surface.options)),
            'options': surface.public_candidates(), 'blocked': list(surface.blocked),
            'retrieved_candidate_audit': to_primitive(candidates)})
        for key, value in {'related': len(candidates),
                'selectable': len({o.atomic_ref for o in surface.options}),
                'options': len(surface.options), 'blocked': len(surface.blocked),
                'blocked_no_mapping': sum(not any(p['status'] == 'proven' for p in c.mapping_previews)
                    for c in candidates) if surface.scope == 'node' else 0}.items():
            self._increment_funnel(ctx, 'runtime_support_funnel', key + '_count', value)
        return surface

    @staticmethod
    def _support_tool(candidates: list[Any]) -> NativeToolSpec:
        from .support_interface import public_arguments_schema
        argument_properties: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            for parameter in getattr(candidate, "inputs", ()):
                name = str(parameter.get("name", ""))
                if not name:
                    continue
                semantic_type = str(
                    parameter.get("semantic_type", "")
                ).casefold()
                schema_type = {
                    "integer": "integer",
                    "int": "integer",
                    "number": "number",
                    "float": "number",
                    "boolean": "boolean",
                    "bool": "boolean",
                    "array": "array",
                    "list": "array",
                    "object_map": "object",
                    "dict": "object",
                }.get(semantic_type, "string")
                current = argument_properties.get(name)
                proposed = {
                    "type": schema_type,
                    "description": (
                        "Input role of the selected support Atomic; selected-"
                        "candidate required roles and grounding are validated "
                        "again before execution."
                    ),
                }
                if current is None:
                    argument_properties[name] = proposed
                elif current.get("type") != schema_type:
                    # The selected-candidate preflight remains authoritative.
                    argument_properties[name] = {
                        "description": proposed["description"],
                    }
        arguments_schema = public_arguments_schema(candidates) if all(
            getattr(c, "input_schema", {}) for c in candidates) else {
                "type": "object", "properties": argument_properties, "additionalProperties": False}
        return NativeToolSpec(
            "invoke_support_atomic",
            "Ask Runtime to execute a contract-compatible support Atomic whose "
            "output/evidence can resolve a missing binding of the current blocked "
            "Atomic. Only candidates returned by formal support retrieval are "
            "offered. Argument values are Agent-declared and still pass through "
            "deterministic preflight/validation.",
            {
                "type": "object",
                "required": ["support_atomic_ref", "arguments"],
                "additionalProperties": False,
                "properties": {
                    "support_atomic_ref": {
                        "type": "string",
                        "enum": [str(item.atomic_ref) for item in candidates],
                    },
                    "arguments": arguments_schema,
                    "output_mapping": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        )

    @staticmethod
    def _automation_tool() -> NativeToolSpec:
        return NativeToolSpec(
            "propose_runtime_automation_atomic",
            "Propose one bounded Automation Atomic draft when upcoming work is "
            "repetitive or systematic preparation expressible with the current "
            "structured action/evidence interface. A bounded check of public "
            "candidate sources for a missing required binding qualifies even "
            "when the target is semantic. This is an "
            "Atomic contract draft, never source code and never create_tool.",
            RUNTIME_AUTOMATION_ATOMIC_SCHEMA,
            call_kind='automation_draft', scope='draft', result_owner='draft_validator',
        )

    def _record_control_call(
        self,
        call: Any,
        session: Any,
        occurrence: Any,
        ctx: Any,
        *,
        call_kind: str,
        result: dict[str, Any],
    ) -> None:
        ctx.trace_builder.trace.native_tool_calls.append(
            NativeToolCallRecord(
                call.call_id,
                session.session_id,
                occurrence.occurrence_id,
                call.name,
                dict(call.arguments),
                call_kind,
                to_primitive(result),
                f"revision:{ctx.world_revision}",
                sum(
                    1 for item in ctx.trace_builder.trace.agent_turns
                    if item.session_id == session.session_id
                ) - 1,
            )
        )

    @staticmethod
    def _record_runtime_automation_outcome_metrics(
        ctx: Any,
        outcome: Any,
    ) -> None:
        """Apply the same R0/R1 counters on preparation and seeded paths."""

        metrics = ctx.trace_builder.trace.metadata.setdefault(
            "v32_metrics", {},
        )
        if outcome.r0_passed:
            metrics["runtime_automation_r0_pass_count"] = int(
                metrics.get("runtime_automation_r0_pass_count", 0)
            ) + 1
            NodeExecutor._increment_funnel(
                ctx, "runtime_automation_funnel", "r0_pass",
            )
        else:
            metrics["runtime_automation_r0_reject_count"] = int(
                metrics.get("runtime_automation_r0_reject_count", 0)
            ) + 1
            NodeExecutor._increment_funnel(
                ctx, "runtime_automation_funnel", "r0_reject",
            )
        if outcome.trial is None:
            return
        metrics["runtime_tool_trial_count"] = int(
            metrics.get("runtime_tool_trial_count", 0)
        ) + 1
        key = (
            "runtime_tool_trial_r1_pass_count"
            if outcome.r1_passed
            else "runtime_tool_trial_r1_reject_count"
        )
        metrics[key] = int(metrics.get(key, 0)) + 1

    def _process_runtime_automation_call(
        self,
        call: Any,
        ctx: Any,
        occurrence: Any,
    ) -> dict[str, Any]:
        """Run the identical draft/R0/R1 path for every Runtime session."""

        metrics = ctx.trace_builder.trace.metadata.setdefault(
            "v32_metrics", {},
        )
        metrics["runtime_automation_atomic_proposal_count"] = int(
            metrics.get("runtime_automation_atomic_proposal_count", 0)
        ) + 1
        self._increment_funnel(
            ctx, "runtime_automation_funnel", "proposal_count",
        )
        try:
            validate_schema_instance(
                call.arguments, RUNTIME_AUTOMATION_ATOMIC_SCHEMA,
            )
            draft = runtime_automation_draft_from_dict(call.arguments)
        except (SchemaValidationError, KeyError, TypeError) as exc:
            metrics["runtime_automation_r0_reject_count"] = int(
                metrics.get("runtime_automation_r0_reject_count", 0)
            ) + 1
            self._increment_funnel(
                ctx, "runtime_automation_funnel", "r0_reject",
            )
            return {
                "accepted": False,
                "error": "runtime_automation_r0_rejected",
                "message": str(exc),
            }

        consumer_scope = getattr(occurrence, "consumer_scope", "node")
        draft.metadata.update(consumer_scope=consumer_scope,
            parent_atomic_ref="" if consumer_scope == "task" else str(occurrence.node_ref))
        serialized_draft = to_primitive(draft)
        occurrence_id = str(occurrence.occurrence_id)
        if str(draft.source_occurrence_id) != occurrence_id:
            metrics["runtime_automation_r0_reject_count"] = int(
                metrics.get("runtime_automation_r0_reject_count", 0)
            ) + 1
            self._increment_funnel(
                ctx, "runtime_automation_funnel", "r0_reject",
            )
            return {
                "accepted": False,
                "r0_passed": False,
                "error": "runtime_automation_source_occurrence_mismatch",
                "failure_code": (
                    "runtime_automation_source_occurrence_mismatch"
                ),
                "stage": "r0_rejected",
                "draft_id": draft.draft_id,
                "message": (
                    "draft.source_occurrence_id does not identify the "
                    "active occurrence"
                ),
            }

        draft_cache = getattr(ctx, "runtime_automation_drafts", None)
        if not isinstance(draft_cache, dict):
            draft_cache = {}
            ctx.runtime_automation_drafts = draft_cache
        existing = draft_cache.get(draft.draft_id)
        if isinstance(existing, dict):
            if existing.get("draft") == serialized_draft:
                existing["duplicate_id_cache_hit"] = True
                self._increment_funnel(
                    ctx,
                    "runtime_automation_funnel",
                    "duplicate_id_cache_hit_count",
                )
                cached = dict(existing.get("safe_outcome") or {})
                cached.update({
                    "accepted": bool(cached.get("accepted", True)),
                    "draft_id": draft.draft_id,
                    "cache_hit": True,
                    "new_revision": getattr(ctx, "world_revision", 0),
                })
                return cached
            metrics["runtime_automation_r0_reject_count"] = int(
                metrics.get("runtime_automation_r0_reject_count", 0)
            ) + 1
            self._increment_funnel(
                ctx, "runtime_automation_funnel", "r0_reject",
            )
            return {
                "accepted": False,
                "r0_passed": False,
                "error": "runtime_automation_draft_id_conflict",
                "failure_code": "runtime_automation_draft_id_conflict",
                "stage": "r0_rejected",
                "draft_id": draft.draft_id,
                "message": (
                    "draft_id was already used with a different payload in "
                    "this task attempt"
                ),
            }

        draft_cache[draft.draft_id] = {
            "draft": to_primitive(draft),
            "stage": "r0",
        }
        ctx.record_r3_event(
            "runtime_automation_proposed",
            occurrence_id=occurrence.occurrence_id,
            details={"draft_id": draft.draft_id},
        )
        if self.automation_coordinator is None:
            payload = {
                "accepted": False,
                "r0_passed": False,
                "error": "runtime_automation_unavailable",
                "failure_code": "runtime_automation_unavailable",
                "stage": "unavailable",
                "draft_id": draft.draft_id,
                "message": "Runtime automation coordinator is unavailable",
            }
            draft_cache[draft.draft_id].update(payload)
            draft_cache[draft.draft_id]["safe_outcome"] = dict(payload)
            return payload

        outcome = self.automation_coordinator.process_draft(
            draft=draft,
            ctx=ctx,
            occurrence=occurrence,
        )
        if outcome.trial is not None:
            outcome.trial["authorizing_native_call_id"] = call.call_id
        safe_outcome = safe_runtime_automation_outcome(outcome)
        payload = {
            "accepted": True,
            "draft_id": draft.draft_id,
            "new_revision": ctx.world_revision,
            **safe_outcome,
        }
        draft_cache[draft.draft_id].update(to_primitive(outcome))
        draft_cache[draft.draft_id]["safe_outcome"] = dict(payload)
        self._record_runtime_automation_outcome_metrics(ctx, outcome)
        return payload

    def _mark_runtime_trial_parent_resumed(
        self,
        ctx: Any,
        payload: dict[str, Any],
    ) -> None:
        """Record one real trial returning control to its parent Runtime."""

        trial_view = payload.get("trial")
        if not isinstance(trial_view, dict):
            return
        draft_id = str(payload.get("draft_id", ""))
        trial = getattr(ctx, "runtime_tool_trials", {}).get(draft_id)
        if not isinstance(trial, dict) or trial.get(
            "parent_resumed_after_trial", False
        ):
            return
        trial["parent_resumed_after_trial"] = True
        self._increment_funnel(
            ctx,
            "runtime_automation_funnel",
            "parent_resumed_after_trial_count",
        )

    def _mark_runtime_trial_parent_completed(
        self,
        ctx: Any,
        occurrence: Any,
    ) -> None:
        """Credit only a later, separately validated parent completion."""

        occurrence_id = str(occurrence.occurrence_id)
        for trial in getattr(ctx, "runtime_tool_trials", {}).values():
            if not isinstance(trial, dict):
                continue
            if str(trial.get("source_occurrence_id", "")) != occurrence_id:
                continue
            if not trial.get("parent_resumed_after_trial", False):
                continue
            if trial.get("parent_completed_after_trial", False):
                continue
            trial["parent_completed_after_trial"] = True
            self._increment_funnel(
                ctx,
                "runtime_automation_funnel",
                "parent_completed_after_trial_count",
            )

    def _mark_prior_trials_parent_continuation(
        self, ctx: Any, occurrence: Any, session_id: str,
    ) -> None:
        """Only an accepted parent turn proves continuation after a trial."""

        for draft_id, trial in getattr(ctx, "runtime_tool_trials", {}).items():
            if not isinstance(trial, dict) or (
                str(trial.get("source_occurrence_id", ""))
                != str(occurrence.occurrence_id)
                or trial.get("terminal_interrupted", False)
                or trial.get("parent_resumed_after_trial", False)
            ):
                continue
            self._mark_runtime_trial_parent_resumed(
                ctx, {"draft_id": draft_id, "trial": trial},
            )
            trial["parent_continuation_session_id"] = session_id

    def _runtime_automation_terminal_boundary(
        self,
        occurrence: Any,
    ) -> ImplementationExecutionResult:
        """Return control without presenting the trial as parent success."""

        result = self.not_started(occurrence, failure_code="")
        # The Agent entered this node, but no parent Implementation completed.
        # Keep its execution attribution distinct from unentered future nodes.
        result.node_status = NodeExecutionStatus.TERMINAL_PARTIAL
        result.terminal_interrupted = True
        result.official_terminal_observed = True
        return result

    @staticmethod
    def _runtime_automation_reached_terminal(
        ctx: Any,
        payload: dict[str, Any],
    ) -> bool:
        checker = getattr(ctx, "execution_terminal", None)
        if callable(checker) and bool(checker()):
            return True
        if bool(getattr(ctx, "terminal_latched", False)):
            return True
        trial = payload.get("trial")
        return bool(
            isinstance(trial, dict)
            and trial.get("terminal_interrupted", False)
        )

    def _validate_current_atomic_call(
        self,
        call: Any,
        session: Any,
        occurrence: Any,
        ctx: Any,
        *,
        mode: str,
        atomic: Any,
    ) -> tuple[ImplementationExecutionResult | None, dict[str, Any]]:
        claims = dict(call.arguments.get("candidate_bindings") or {})
        input_roles = {item.name for item in atomic.inputs}
        unknown = sorted(set(claims) - input_roles)
        if unknown:
            resolution = AtomicEffectResolution(
                False,
                failure_code="atomic_preferred_binding_role_invalid",
                message=(
                    "validate_current_atomic candidate_bindings may reference "
                    f"only current Atomic inputs; unknown roles: {unknown!r}"
                ),
            )
            effect = None
        else:
            resolutions: list[AtomicEffectResolution] = []
            effect = self._complete_from_current_effect(
                occurrence,
                ctx,
                mode=mode,
                preferred_values=[],
                preferred_bindings=claims,
                candidate_outputs=dict(call.arguments.get("candidate_outputs") or {}),
                atomic_override=atomic,
                resolution_out=resolutions,
            )
            resolution = resolutions[-1]
        payload = {
            "accepted": True,
            "committed": effect is not None,
            "passed": effect is not None,
            "atomic_effect_passed": effect is not None,
            "validation": to_primitive(resolution),
            "new_revision": ctx.world_revision,
        }
        if effect is not None:
            self._refresh_occurrence_state(
                occurrence, atomic, [], ctx,
            )
        self._augment_runtime_payload(
            payload, ctx, occurrence=occurrence, atomic=atomic,
            session_id=session.session_id, tool_call_id=call.call_id,
        )
        self._record_control_call(
            call,
            session,
            occurrence,
            ctx,
            call_kind="atomic_validation",
            result=payload,
        )
        return effect, payload

    def _status_result(
        self,
        call: Any,
        session: Any,
        occurrence: Any,
        ctx: Any,
    ) -> tuple[str, dict[str, Any]]:
        status = str(call.arguments["status"])
        payload = {"accepted": True, "status": status}
        self._record_control_call(
            call,
            session,
            occurrence,
            ctx,
            call_kind="runtime_status",
            result=payload,
        )
        return status, payload

    @staticmethod
    def _resolve_support_output_mapping(
        call: Any, candidate: Any,
    ) -> dict[str, str] | None:
        if getattr(candidate, "mapping_previews", ()):
            options = candidate.allowed_output_mappings
            requested = call.arguments.get("output_mapping")
            if requested is None:
                return dict(options[0]) if len(options) == 1 else None
            return dict(requested) if requested in options else None
        role_mappings = tuple(
            getattr(candidate, "role_mappings", ())
        )
        if not role_mappings:
            if getattr(candidate, "predicate_obligations", ()) and not call.arguments.get("output_mapping"):
                return {}
            return None
        requested = dict(call.arguments.get("output_mapping") or {})
        if not requested:
            if len(role_mappings) == 1:
                only = role_mappings[0]
                return {
                    str(only.producer_role): str(only.consumer_role)
                }
            return None
        allowed: dict[str, set[str]] = {}
        for item in role_mappings:
            allowed.setdefault(str(item.producer_role), set()).add(
                str(item.consumer_role)
            )
        if (
            set(requested) - set(allowed)
            or len(set(map(str, requested.values()))) != len(requested)
        ):
            return None
        for producer_role, consumer_role in requested.items():
            if consumer_role not in allowed.get(producer_role, set()):
                return None
        return requested

    def _invoke_support_atomic_call(self, call, session, occurrence, ctx, atomic, candidates,
                                   plan_context_plan=None, *, surface=None):
        if surface is None:
            if getattr(self, 'support_interface_version', None):
                raise RuntimeError('versioned Support requires its current request surface')
            return self._execute_support_atomic_call(call, session, occurrence, ctx, atomic,
                candidates, plan_context_plan)
        from .support_call_surface import decode
        from .negative_memory import cached_rejection, remember_rejection
        from ..core.refs import content_hash
        selection, payload = decode(surface, call, session_id=session.session_id,
            occurrence=occurrence, consumer=atomic, ctx=ctx, skills=self.invocation_compiler.skills)
        audit = {'version': surface.version, 'session_id': session.session_id, 'call_id': call.call_id,
            'support_call_id': call.arguments.get('support_call_id'),
            'surface_context_fingerprint': surface.surface_context_fingerprint,
            'consumer_occurrence_id': occurrence.occurrence_id, 'revision': ctx.world_revision,
            'input_hash': content_hash(call.arguments.get('arguments')),
            'resolution_status': 'resolved' if payload is None else 'rejected'}
        ctx.trace_builder.trace.metadata.setdefault('support_call_resolutions', []).append(audit)
        if selection is not None:
            audit.update(producer_atomic_ref=selection.option.atomic_ref,
                         resolved_output_mapping=selection.output_mapping,
                         predicate_input_mapping=selection.input_mapping,
                         route_identity=selection.option.public()['route_identity'])
        if payload is not None:
            audit['reason_code'] = payload['reason_code']
            self._increment_funnel(ctx, 'runtime_support_funnel', 'decode_rejection_count')
            if payload['reason_code'] == 'support_atomic_input_schema_invalid':
                self._increment_funnel(ctx, 'runtime_support_funnel', 'input_schema_rejection_count')
            self._record_control_call(call, session, occurrence, ctx,
                call_kind='support_atomic_invocation', result=payload)
            if selection is not None:
                remember_rejection(ctx, occurrence, call, payload, selection=selection)
            return payload
        audit.update(producer_atomic_ref=selection.option.atomic_ref,
                     resolved_output_mapping=selection.output_mapping,
                     predicate_input_mapping=selection.input_mapping,
                     route_identity=selection.option.public()['route_identity'])
        self._increment_funnel(ctx, 'runtime_support_funnel', 'decode_passed_count')
        payload = cached_rejection(ctx, occurrence, call, selection=selection)
        if payload is not None:
            payload['support_call_id'] = selection.option.support_call_id
            self._increment_funnel(ctx, 'runtime_support_funnel', 'negative_cache_hit_count')
            self._record_control_call(call, session, occurrence, ctx,
                call_kind='runtime_cached_rejection', result=payload)
        else:
            payload = self._execute_support_atomic_call(call, session, occurrence, ctx, atomic,
                candidates, plan_context_plan, selection=selection)
            remember_rejection(ctx, occurrence, call, payload, selection=selection)
        audit['result'] = {k: payload[k] for k in ('accepted', 'passed', 'preflight_failure_code',
            'error', 'deterministic_rejection_cache_hit') if k in payload}
        return payload

    def _execute_support_atomic_call(
        self,
        call: Any,
        session: Any,
        occurrence: Any,
        ctx: Any,
        atomic: Any,
        candidates: list[Any],
        plan_context_plan: Any | None = None,
        *, selection: Any = None,
    ) -> dict[str, Any]:
        def finalize(payload: dict[str, Any]) -> dict[str, Any]:
            if selection is not None:
                payload.setdefault('support_call_id', selection.option.support_call_id)
            if payload.get("accepted") is False:
                candidate = next((c for c in candidates if str(c.atomic_ref) ==
                    (selection.option.atomic_ref if selection else str(call.arguments.get("support_atomic_ref", "")))), None)
                payload.setdefault("error_code", payload.get("error", payload.get("failure_code", "support_rejected")))
                payload.setdefault('reason_code', payload.get('preflight_failure_code') or payload['error_code'])
                diagnostics = payload.get('diagnostics') or {}
                if diagnostics.get('bindings'):
                    bindings = diagnostics['bindings']
                    payload.setdefault('argument_path', '$.arguments.' + bindings[0]['role'] if len(bindings) == 1 else '$.arguments')
                    payload.setdefault('expected_constraint', {b['role']: {'required_resolution': b['required_resolution'],
                        'semantic_anchor': b['semantic_anchor']} for b in bindings})
                    payload.setdefault('actual_summary', {b['role']: b['submitted_value'] for b in bindings})
                    payload.setdefault('required_anchor_or_relation', diagnostics.get('required_anchor_or_relation'))
                payload.setdefault("argument_path", "$.output_mapping" if "mapping" in payload["error_code"] else "$.arguments")
                mapping_error = payload['argument_path'] == '$.output_mapping'
                payload.setdefault("expected_constraint", list(getattr(candidate, 'allowed_output_mappings', ()))
                    if mapping_error else getattr(candidate, "input_schema", {}))
                raw_arguments = call.arguments.get("arguments")
                payload.setdefault("actual_summary", {'omitted': 'output_mapping' not in call.arguments,
                    'output_mapping': call.arguments.get('output_mapping')} if mapping_error else {"argument_roles": sorted(raw_arguments)}
                    if isinstance(raw_arguments, dict) else {"type": type(raw_arguments).__name__})
                payload.setdefault("allowed_output_mappings", list(getattr(candidate, "allowed_output_mappings", ())))
                payload.setdefault("required_anchor_or_relation", [p for p in getattr(candidate, "mapping_previews", ()) if p["status"] != "proven"])
                payload.setdefault("relevant_revision", ctx.world_revision)
            self._augment_runtime_payload(
                payload,
                ctx,
                occurrence=occurrence,
                atomic=atomic,
                plan_context_plan=plan_context_plan,
                session_id=session.session_id,
                tool_call_id=call.call_id,
            )
            self._record_control_call(
                call,
                session,
                occurrence,
                ctx,
                call_kind="support_atomic_invocation",
                result=payload,
            )
            return payload

        self._increment_funnel(
            ctx, "runtime_support_funnel", "raw_call_count",
        )
        try:
            support_ref = SkillRef.parse(
                selection.option.atomic_ref if selection else str(call.arguments["support_atomic_ref"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._increment_funnel(
                ctx,
                "runtime_support_funnel",
                "input_schema_rejection_count",
            )
            return finalize({
                "accepted": False,
                "error": "runtime_support_candidate_invalid",
                "message": str(exc),
            })
        candidate = next(
            (item for item in candidates if str(item.atomic_ref) == str(support_ref)),
            None,
        )
        if candidate is None:
            payload = {
                "accepted": False,
                "error": "runtime_support_candidate_invalid",
            }
            return finalize(payload)
        try:
            support_atomic = self.invocation_compiler.skills.get_atomic(support_ref)
        except KeyError:
            payload = {
                "accepted": False,
                "error": "runtime_support_atomic_unavailable",
            }
            return finalize(payload)
        arguments = selection.arguments if selection else call.arguments.get("arguments", {})
        if not isinstance(arguments, dict):
            return finalize({"accepted": False, "error": "support_atomic_input_schema_invalid",
                "argument_path": "$.arguments", "expected_constraint": {"type": "object"},
                "constraint_scope": "stable_schema", "support_atomic_ref": str(support_ref)})
        if getattr(candidate, "input_schema", {}):
            from ..agents.protocol import SchemaValidationError, validate_schema_instance
            try:
                validate_schema_instance(arguments, candidate.input_schema, path="$.arguments")
            except SchemaValidationError as exc:
                self._increment_funnel(ctx, "runtime_support_funnel", "input_schema_rejection_count")
                return finalize({"accepted": False, "error": "support_atomic_input_schema_invalid",
                    "message": str(exc), "argument_path": exc.path,
                    "expected_constraint": exc.constraint or candidate.input_schema,
                    "actual_summary": exc.actual or {"argument_roles": sorted(arguments)},
                    "constraint_scope": "stable_schema", "support_atomic_ref": str(support_ref)})
        if getattr(occurrence, "consumer_scope", "node") == "task":
            from .task_runtime import invoke_task_capability
            return finalize(invoke_task_capability(self, call, occurrence, ctx, candidate, selection=selection))
        output_mapping = (selection.output_mapping if selection else
                          self._resolve_support_output_mapping(call, candidate))
        if output_mapping is None:
            self._increment_funnel(
                ctx,
                "runtime_support_funnel",
                "output_mapping_rejection_count",
            )
            payload = {
                "accepted": False,
                "error": "support_atomic_output_mapping_invalid",
            }
            return finalize(payload)
        support_occurrence = RuntimeOccurrence(
            step_id=f"support_for_{occurrence.step_id}",
            occurrence_id=f"support_{occurrence.occurrence_id}_{uuid.uuid4().hex[:8]}",
            node_ref=support_ref,
            requirement_ids=[],
            # Agent values are proposals. They must pass the ordinary
            # invocation resolver and may not masquerade as Constants.
            binding_specs={},
            implementation_candidates=[
                str(item.ref)
                for item in self.invocation_compiler.skills.implementations_for(
                    support_ref, mode=self.invocation_compiler.mode,
                )
            ],
            expected_effects=list(support_atomic.effects),
        )
        invocations = self.invocation_compiler.compile_candidates(
            support_occurrence, ctx.binding_store,
            task_id=ctx.task_id, evidence_store=ctx.evidence_store, revision=ctx.world_revision, task_contract=ctx.task_contract,
        )
        if not invocations:
            return finalize({
                "accepted": False,
                "support_atomic_ref": str(support_ref),
                "error": "runtime_support_no_executable",
                "message": (
                    "No mode-compatible executable Implementation/Tool is "
                    "available for this support Atomic"
                ),
            })

        v32_metrics = ctx.trace_builder.trace.metadata.setdefault(
            "v32_metrics", {},
        )
        v32_metrics["runtime_support_selected_count"] = int(
            v32_metrics.get("runtime_support_selected_count", 0)
        ) + 1
        preferred = [item for item in invocations if item.implementation.quality.get("preferred")] if len(invocations) > 1 else []
        if len(invocations) > 1 and len(preferred) != 1:
            return finalize({"accepted": False, "error": "support_implementation_ambiguous"})
        compiled = preferred[0] if preferred else invocations[0]
        if selection and str(compiled.implementation.ref) != selection.option.public()['route_identity']:
            return finalize({'accepted': False, 'error': 'support_route_changed'})
        predicate_options = getattr(candidate, "predicate_obligations", ())
        from .support_request import SupportRequest, prove_request, consumer_guard, known_consumer_values, transfer_inputs, consumer_constraints, validate_transfer
        from ..core.support_authority import input_identity_source_role
        from .invocation_transaction import execute_invocation
        input_mapping = {}
        anchor_inputs = set()
        for output, consumer in output_mapping.items():
            source = input_identity_source_role(support_atomic, output)
            if not source:
                source = support_atomic.validator_spec.get("output_semantic_constraints", {}).get(output, {}).get("compatible_with_input")
                if source:
                    anchor_inputs.add(source)
            if source:
                input_mapping[source] = consumer
        if output_mapping:
            from dataclasses import replace
            from ..core.semantic_types import semantic_types_compatible
            constraints = support_atomic.validator_spec.get("output_semantic_constraints", {})
            specs = {item.name: item for item in support_atomic.inputs}
            anchors = {}
            for output, consumer in output_mapping.items():
                source = constraints.get(output, {}).get("compatible_with_input")
                anchor = ctx.binding_store.semantic_anchor_for(occurrence, consumer) if source else None
                if source not in specs or anchor is None:
                    continue
                if not semantic_types_compatible(specs[source].semantic_type, anchor.semantic_type):
                    continue
                if source in anchors and anchors[source].value != anchor.value:
                    return finalize({"accepted": False, "error": "support_semantic_anchor_ambiguous"})
                anchors[source] = replace(anchor, role=source)
            ctx.binding_store.commit_grounded(support_occurrence.occurrence_id, anchors)
        if predicate_options and not output_mapping and selection is None:
            from .support_request import binding_accepts_proposal
            from ..core.refs import canonical_json
            parent = ctx.binding_store.snapshot_for_node(occurrence)
            mappings = {canonical_json(option["input_mapping"]): option["input_mapping"]
                        for option in predicate_options}.values()
            valid = [mapping for mapping in mappings if all(
                consumer in parent and parent[consumer].status is BindingStatus.GROUNDED
                and (producer not in arguments or binding_accepts_proposal(
                    parent[consumer], consumer, arguments[producer], ctx))
                for producer, consumer in mapping.items()
                if producer in {item.name for item in support_atomic.inputs})]
            if len(valid) != 1:
                return finalize({"accepted": False, "error": "support_predicate_mapping_ambiguous"})
            from dataclasses import replace
            ctx.binding_store.commit_grounded(support_occurrence.occurrence_id, {
                producer: replace(parent[consumer], role=producer)
                for producer, consumer in valid[0].items()
                if producer in {item.name for item in support_atomic.inputs}
            })
            input_mapping.update({p: c for p, c in valid[0].items()
                                  if p in {item.name for item in support_atomic.inputs}})
        if selection is not None:
            input_mapping = selection.input_mapping
            anchor_inputs = set(selection.option.public()['anchor_inputs'])
        request = SupportRequest(ctx.budget.current_occurrence_id, occurrence, support_atomic,
                                 input_mapping, dict(output_mapping), anchor_inputs)
        request.grounding_constraints = consumer_constraints(self.invocation_compiler, atomic)
        request.producer_occurrence_id = support_occurrence.occurrence_id
        proof = prove_request(request, atomic, ctx, agent_selected=True)
        if not proof.passed:
            return finalize({"accepted": False, "passed": False, "error": proof.failure_codes[0],
                             "failure_code": proof.failure_codes[0], "message": proof.messages[0]})
        if selection and (request.input_mapping != selection.input_mapping or request.output_mapping != selection.output_mapping):
            return finalize({'accepted': False, 'error': 'support_selected_mapping_changed'})
        if selection:
            self._increment_funnel(ctx, 'runtime_support_funnel', 'mapping_reproof_passed_count')
        output_mapping = request.output_mapping
        from .support_request import mapped_support_bindings
        seeds = mapped_support_bindings(support_atomic, request.input_mapping,
            request.anchor_inputs, occurrence, ctx.binding_store)
        if seeds is not None:
            ctx.binding_store.commit_grounded(support_occurrence.occurrence_id, seeds)
        parent_guard = consumer_guard(request, atomic, known_consumer_values(request, arguments), ctx)
        if not parent_guard.passed:
            return finalize({"accepted": False, "passed": False, "error": parent_guard.failure_codes[0],
                             "failure_code": parent_guard.failure_codes[0], "message": parent_guard.messages[0]})
        parent_refresh = getattr(ctx, "_after_action_refresh", None)
        parent_failed_invocation = (
            dict(ctx.last_failed_invocation)
            if getattr(ctx, "last_failed_invocation", None) is not None
            else None
        )
        result = None
        preflight = None
        try:
            begin_occurrence = getattr(ctx, "begin_occurrence", None)
            if callable(begin_occurrence):
                begin_occurrence(support_occurrence)
            resolve_specs = getattr(
                getattr(ctx, "binding_store", None),
                "resolve_occurrence_specs",
                None,
            )
            if callable(resolve_specs):
                resolve_specs(support_occurrence, ctx.world_revision, input_specs=compiled.atomic.inputs)
            prepared = self.invocation_compiler.prepare_arguments(
                compiled,
                call_name=compiled.spec.name,
                call_id=call.call_id,
                arguments=arguments,
                occurrence=support_occurrence,
                binding_store=ctx.binding_store,
                evidence_store=ctx.evidence_store,
                revision=ctx.world_revision,
                task_contract=ctx.task_contract,
            )
            # Resolve effects before asking for a new entry affordance.
            # This allows an already-satisfied navigation/pick capability.
            result = self._complete_from_current_effect(
                support_occurrence, ctx, mode="entry", preferred_values=list(arguments.values()),
                preferred_bindings=arguments,
                provisional_bindings=prepared.binding_updates if prepared.passed else [],
                effect_guard=lambda resolution: validate_transfer(request, atomic, resolution.output_candidates, ctx),
            ) if prepared.passed else None
            if result is not None:
                transfer = transfer_inputs(request, atomic, result, ctx)
                if not transfer.passed:
                    result = None
            preflight = ToolCallPreflightResult(True, str(compiled.implementation.ref)) if result is not None else (
                self.invocation_compiler.validate_execution_context(
                    compiled,
                    prepared,
                    occurrence=support_occurrence,
                    binding_store=ctx.binding_store,
                    evidence_store=ctx.evidence_store,
                    revision=ctx.world_revision,
                )
                if prepared.passed
                else prepared
            )
            if result is None and preflight.passed:
                result = execute_invocation(self.implementation_runner,
                    compiled,
                    preflight,
                    support_occurrence,
                    ctx,
                    agent_prepared=True,
                    accept_result=lambda value: transfer_inputs(request, atomic, value, ctx),
                    consumer=occurrence,
                    authorizing_native_call_id=call.call_id,
                )
        finally:
            # A failed or exceptional support attempt may change the real
            # world, but it may never leave the helper occurrence active.
            begin_occurrence = getattr(ctx, "begin_occurrence", None)
            if callable(begin_occurrence):
                begin_occurrence(occurrence)
            if hasattr(ctx, "_after_action_refresh"):
                ctx._after_action_refresh = parent_refresh
            if callable(parent_refresh):
                parent_refresh()
            if (
                parent_failed_invocation is not None
                and getattr(ctx, "last_failed_invocation", None) is None
            ):
                ctx.last_failed_invocation = parent_failed_invocation

        if preflight is None or not preflight.passed:
            failure_code = str(
                getattr(preflight, "failure_code", "")
                or "support_not_execution_ready"
            )
            schema_rejection = failure_code == "runtime_agent_schema_error"
            self._increment_funnel(
                ctx,
                "runtime_support_funnel",
                (
                    "input_schema_rejection_count"
                    if schema_rejection
                    else "preflight_not_ready_count"
                ),
            )
            return finalize({
                "accepted": False,
                "support_atomic_ref": str(support_ref),
                "passed": False,
                "error": (
                    "support_atomic_input_schema_invalid"
                    if schema_rejection
                    else "support_not_execution_ready"
                ),
                "preflight_failure_code": failure_code,
                "diagnostics": getattr(preflight, 'diagnostics', {}),
                "message": str(getattr(preflight, "message", "")),
                "missing_required_inputs": sorted(
                    set(compiled.spec.input_schema.get("required", ()))
                    - set(arguments)
                ),
                "new_revision": ctx.world_revision,
            })

        atomic_effect_passed = bool(
            result is not None
            and getattr(result, "atomic_effect_passed", False)
        )
        if result is not None and bool(getattr(result, "started", False)):
            self._increment_funnel(
                ctx,
                "runtime_support_funnel",
                "execution_started_count",
            )
        blocked_input_roles = {
            str(item.name) for item in atomic.inputs
        }
        validated_outputs = dict(
            getattr(result, "validated_outputs", {}) or {}
        ) if result is not None else {}
        support_outputs = {
            consumer_role: validated_outputs[producer_role]
            for producer_role, consumer_role in output_mapping.items()
            if producer_role in validated_outputs
            and consumer_role in blocked_input_roles
            and validated_outputs[producer_role] not in (None, "")
        }
        output_mapping_complete = bool(
            (output_mapping or predicate_options)
            and len(support_outputs) == len(output_mapping)
        )
        passed = bool(atomic_effect_passed and output_mapping_complete)
        payload: dict[str, Any] = {
            "accepted": True,
            "support_atomic_ref": str(support_ref),
            "support_occurrence_id": support_occurrence.occurrence_id,
            "passed": passed,
            "result": to_primitive(result) if result is not None else None,
            "new_revision": ctx.world_revision,
        }
        if result is not None and result.failure_code:
            payload["error"] = result.failure_code
        if atomic_effect_passed and not output_mapping_complete:
            payload["error"] = "support_atomic_output_unresolved"
        if passed:
            v32_metrics["runtime_support_success_count"] = int(
                v32_metrics.get("runtime_support_success_count", 0)
            ) + 1
            self._increment_funnel(
                ctx,
                "runtime_support_funnel",
                "validated_output_published_count",
            )
            v32_metrics["runtime_graph_augmentation_count"] = int(
                v32_metrics.get("runtime_graph_augmentation_count", 0)
            ) + 1
            ctx.trace_builder.trace.metadata.setdefault(
                "runtime_graph_augmentation", []
            ).append({
                "support_atomic_ref": str(support_ref),
                "producer_occurrence_id": support_occurrence.occurrence_id,
                "consumer_occurrence_id": occurrence.occurrence_id,
                "data_flow_roles": sorted(output_mapping.items()),
                "output_mapping": dict(output_mapping),
                "reason": "missing_binding_support",
            })
            if getattr(result, "started", False):
                ctx.trace_builder.trace.metadata.setdefault("runtime_support_node_records", []).append({
                    "occurrence_id": support_occurrence.occurrence_id,
                    "consumer_occurrence_id": occurrence.occurrence_id,
                    "step_id": support_occurrence.step_id,
                    "atomic_ref": str(support_ref), "status": result.node_status.value,
                    "direct_result": to_primitive(result), "validated_outputs": support_outputs,
                })
        return finalize(payload)

    def run_agent_node(self, occurrence, ctx, *, mode="preparation", bootstrap=False,
                       atomic_override=None, plan_context_plan=None):
        """The single node Agent loop. No automatic entry or Support re-entry."""
        from .runtime_step import run_runtime_step
        while not ctx.execution_terminal():
            atomic = atomic_override or self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
            invocations = ([] if atomic_override else self.invocation_compiler.compile_candidates(
                occurrence, ctx.binding_store,
                task_id=ctx.task_id, evidence_store=ctx.evidence_store, revision=ctx.world_revision, task_contract=ctx.task_contract))
            if not invocations:
                mode = "seeded"
            missing = ctx.binding_store.runtime_prompt_projection(occurrence, atomic.inputs)["missing_or_insufficient_bindings"]
            candidates = self._retrieve_runtime_support_candidates(
                blocked_atomic=atomic, missing_roles=missing, ctx=ctx, occurrence=occurrence)
            step = run_runtime_step(self, mode, occurrence, ctx, invocations, candidates,
                                    bootstrap=bootstrap, atomic_override=atomic_override,
                                    plan_context_plan=plan_context_plan)
            bootstrap = False
            if ctx.execution_terminal():
                return step.result or self._runtime_automation_terminal_boundary(occurrence)
            if step.automation_request:
                step = run_runtime_step(self, mode, occurrence, ctx, invocations, candidates,
                    draft_request=step.automation_request, atomic_override=atomic_override,
                    plan_context_plan=plan_context_plan)
            if step.failure_code:
                layer = ("composite" if step.failure_code == "runtime_plan_conflict" else
                         "runtime_agent" if step.failure_code == "runtime_action_loop_blocked" else "runtime_binding")
                return self.not_started(occurrence, failure_code=step.failure_code, failure_layer=layer)
            if step.result is not None:
                if step.result.atomic_effect_passed:
                    self._mark_runtime_trial_parent_completed(ctx, occurrence)
                    return step.result
                if step.result.started:
                    mode = "seeded"
        return self._runtime_automation_terminal_boundary(occurrence)

    def run_dynamic(self, ctx: Any, *, rescue: bool = False,
                    cold_start_continuation: bool = False,
                    continuation_context: dict[str, Any] | None = None) -> dict[str, Any]:
        from .task_runtime import run_dynamic
        return run_dynamic(self, ctx, rescue=rescue, cold_start_continuation=cold_start_continuation,
                           continuation_context=continuation_context)
