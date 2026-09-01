"""Per-occurrence Direct/Preparation/Fresh-Seeded state machine and Full Dynamic agent."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Callable

from ..agents.context_builder import ContextBuilder
from ..agents.protocol import AgentTurn, NativeToolSpec
from ..core.bindings import (
    BindingResolution, BindingStatus, RuntimeBinding,
)
from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.results import (
    AtomicEffectResolution, ImplementationExecutionResult, NodeExecutionStatus,
)
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
        current = ctx.binding_store.snapshot_for_node(occurrence)
        effective = dict(current)
        effective.update({item.role: item for item in preflight.binding_updates})
        # Direct Autonomous is a deterministic fast path, not an entity or
        # location chooser.  Deterministic preflight may certify an already
        # formal argument from current validator evidence, but every required
        # runtime-resolvable role must end preflight as a role-specific
        # concrete or relation-verified binding before implementation begins.
        if any(
            parameter.required
            and parameter.runtime_resolvable
            and not self._runtime_role_is_deterministic(
                effective.get(parameter.name),
            )
            for parameter in compiled.atomic.inputs
        ):
            return None
        return self.implementation_runner.run(compiled, preflight, occurrence, ctx, agent_prepared=False)

    @staticmethod
    def _runtime_role_is_deterministic(
        binding: RuntimeBinding | None,
    ) -> bool:
        return bool(
            binding is not None
            and binding.status is BindingStatus.GROUNDED
            and binding.resolution in {
                BindingResolution.CONCRETE,
                BindingResolution.RELATION_VERIFIED,
            }
        )

    def _environment_tool(
        self, ctx: Any, *, node_level: bool = True,
    ) -> NativeToolSpec:
        properties: dict[str, Any] = {
            "action_id": {
                "type": "string",
                "enum": [item.action_id for item in ctx.action_catalog],
            },
        }
        required = ["action_id"]
        description = "Execute one currently available environment action."
        if node_level:
            properties["intent"] = {
                "type": "string",
                "enum": ["explore", "attempt_current_atomic"],
            }
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
        )

    @staticmethod
    def _status_tool(*, allow_plan_conflict: bool = False) -> NativeToolSpec:
        statuses = ["cannot_resolve", "give_up"]
        if allow_plan_conflict:
            statuses.insert(1, "plan_conflict")
        description = (
            "Explicitly report why the current mode cannot continue. "
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
        if bool(snapshot.get("node_budget_active")):
            result["remaining_node_actions"] = max(
                0, int(snapshot.get("remaining_node_actions", 0))
            )
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
    ) -> dict[str, Any]:
        builder = getattr(self, "plan_context_builder", None)
        plan = plan_context_plan or getattr(ctx, "plan", None)
        if builder is None or plan is None:
            return {}
        try:
            return dict(
                builder.build(
                    plan,
                    occurrence.step_id,
                    ctx.binding_store,
                ).policy_view()
            )
        except KeyError:
            # Provisional/test-only occurrences are not verified plan nodes.
            # Fail closed rather than manufacturing downstream intent.
            return {}

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
    def _validate_current_atomic_tool(atomic: Any) -> NativeToolSpec:
        return NativeToolSpec(
            "validate_current_atomic",
            (
                "Ask the Runtime to validate the current public environment "
                "state as completion of the current Atomic. Candidate bindings "
                "are Agent preferences only and cannot create facts or override "
                "formal Task/DataFlow anchors."
            ),
            {
                "type": "object",
                "required": ["candidate_bindings"],
                "additionalProperties": False,
                "properties": {
                    "candidate_bindings": {
                        "type": "object",
                        "properties": {
                            item.name: {
                                "type": [
                                    "string", "integer", "number", "boolean",
                                    "object", "array",
                                ],
                            }
                            for item in atomic.inputs
                        },
                        "additionalProperties": False,
                    },
                },
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
            repeat_values.update(dict(spec.arguments))
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
                return payload, spec
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
            return payload, spec
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
        return payload, spec

    def _complete_from_current_effect(
        self,
        occurrence: Any,
        ctx: Any,
        *,
        mode: str,
        preferred_values: list[Any],
        preferred_bindings: dict[str, Any] | None = None,
        provisional_bindings: list[RuntimeBinding] = (),
        atomic_override: Any | None = None,
        resolution_out: list[AtomicEffectResolution] | None = None,
    ) -> ImplementationExecutionResult | None:
        atomic = atomic_override or self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        bindings = ctx.binding_store.snapshot_for_node(occurrence)
        bindings.update({
            item.role: item
            for item in provisional_bindings
            if item.status is BindingStatus.GROUNDED
        })
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
        resolution = self.validation.atomic.resolve_current_effect(
            atomic,
            occurrence,
            bindings,
            ctx.harness.validator_channel(),
            semantic_anchors=semantic_anchors,
            preferred_values=preferred_values,
            preferred_bindings=preferred_bindings,
            current_revision=ctx.world_revision,
        )
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
        committed = ctx.binding_store.commit_atomic_effect_witnesses(
            occurrence.occurrence_id,
            resolution.resolved_bindings,
            atomic.inputs,
            resolution.witness_refs,
            ctx.world_revision,
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
            before_state_ref="",
            after_state_ref=f"revision:{ctx.world_revision}",
            node_status=status,
        )

    def _validated_output_candidates(self, atomic: Any, bindings: dict[str, RuntimeBinding], ctx: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        plain = {role: value.value for role, value in bindings.items() if value.status is BindingStatus.GROUNDED}
        mapped_outputs: set[str] = set()
        for item in atomic.validator_spec.get("output_identity") or []:
            output_role = str(item.get("output_role", ""))
            input_role = str(item.get("input_role", ""))
            if output_role and input_role in plain:
                result[output_role] = plain[input_role]
                mapped_outputs.add(output_role)
        facts = list(ctx.harness.validator_channel().snapshot().get("facts", []))
        witnesses = [fact for fact in facts if any(fact.get("predicate") == effect.predicate for effect in atomic.effects)]
        for output in atomic.outputs:
            if output.name in mapped_outputs:
                continue
            if output.name in plain:
                result[output.name] = plain[output.name]
                continue
            exact_values = {
                fact.get("args", {}).get(output.name)
                for fact in witnesses
                if fact.get("args", {}).get(output.name) is not None
            }
            if len(exact_values) == 1:
                result[output.name] = next(iter(exact_values))
        return result

    def _environment_effect_preferences(
        self,
        occurrence: Any,
        ctx: Any,
        spec: Any,
    ) -> list[Any]:
        """Treat the accepted node-session action as an explicit proposal.

        Arbitrary entry state is still rejected because the orchestrator calls
        effect resolution with no preferred values.  After an Agent chooses
        and the Harness accepts an action, its concrete arguments are the
        proposal needed to reconcile an otherwise runtime-resolvable role.
        """

        return list(spec.arguments.values())

    def _node_tools(
        self,
        ctx: Any,
        atomic: Any,
        *,
        invocations: list[CompiledInvocation] = (),
        allow_plan_conflict: bool = False,
    ) -> list[NativeToolSpec]:
        return [
            self._environment_tool(ctx, node_level=True),
            self._validate_current_atomic_tool(atomic),
            *[self._invocation_tool(item) for item in invocations],
            self._status_tool(allow_plan_conflict=allow_plan_conflict),
        ]

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

    def run_preparation_session(
        self, occurrence: Any, invocations: list[CompiledInvocation], ctx: Any,
        *, learned_call_repair_limit: int = 2,
        plan_context_plan: Any | None = None,
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
            task_semantic_context=prompt_bindings["task_semantic_context"],
            current_occurrence_semantic_anchors=prompt_bindings[
                "occurrence_semantic_anchors"
            ],
            execution_ready_bindings=prompt_bindings["execution_ready_bindings"],
            missing_or_insufficient_bindings=missing,
            observation=ctx.observation,
            action_catalog=ctx.action_catalog, relevant_action_history=ctx.relevant_history(occurrence.occurrence_id),
            remaining_budget=ctx.budget.snapshot(), implementation_invocations=[item.spec for item in invocations],
            downstream_plan_context=self._downstream_plan_context(
                ctx,
                occurrence,
                plan_context_plan=plan_context_plan,
            ),
        )
        tools = self._node_tools(
            ctx, atomic, invocations=invocations, allow_plan_conflict=True,
        )
        preflight_failures = 0
        loop_guard = ActionLoopGuard()
        try:
            turn = session.next_turn(prompt, tools=tools)
            while True:
                self._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                if call.name == "report_runtime_status":
                    status, payload = self._status_result(
                        call, session, occurrence, ctx,
                    )
                    self._finalize_tool_result(
                        session, call.call_id, payload, tools,
                    )
                    if status == "plan_conflict":
                        conflict = self.not_started(
                            occurrence, failure_code="runtime_plan_conflict",
                        )
                        conflict.failure_layer = "composite"
                        return conflict
                    return self.not_started(occurrence, failure_code="runtime_binding_unresolved")
                if call.name == "environment_action":
                    payload, action_spec = self._execute_environment_call(
                        call, session, occurrence, ctx,
                        span_id=span.span_id, origin="runtime_preparation",
                        loop_guard=loop_guard,
                    )
                    if payload.get("loop_blocked"):
                        tools = self._node_tools(
                            ctx, atomic, invocations=invocations,
                            allow_plan_conflict=True,
                        )
                        if payload.get("fallback_required"):
                            self._finalize_tool_result(session, call.call_id, payload, tools)
                            return self.not_started(
                                occurrence, failure_code="runtime_action_loop_blocked",
                            )
                        turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                        continue
                    effect = None
                    if (
                        call.arguments["intent"] == "attempt_current_atomic"
                        and payload.get("accepted")
                    ):
                        resolutions: list[AtomicEffectResolution] = []
                        effect = self._complete_from_current_effect(
                            occurrence,
                            ctx,
                            mode="preparation",
                            preferred_values=self._environment_effect_preferences(
                                occurrence,
                                ctx,
                                action_spec,
                            ),
                            resolution_out=resolutions,
                        )
                        payload["atomic_validation"] = to_primitive(
                            resolutions[-1]
                        )
                    elif call.arguments["intent"] == "attempt_current_atomic":
                        payload["atomic_validation"] = to_primitive(
                            AtomicEffectResolution(
                                False,
                                failure_code="environment_action_rejected",
                                message=(
                                    "Rejected environment action cannot commit "
                                    "the current Atomic"
                                ),
                            )
                        )
                    tools = self._node_tools(
                        ctx, atomic, invocations=invocations,
                        allow_plan_conflict=True,
                    )
                    if effect is not None:
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        return effect
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                if call.name == "validate_current_atomic":
                    effect, payload = self._validate_current_atomic_call(
                        call,
                        session,
                        occurrence,
                        ctx,
                        mode="preparation",
                        atomic=atomic,
                    )
                    tools = self._node_tools(
                        ctx, atomic, invocations=invocations,
                        allow_plan_conflict=True,
                    )
                    if effect is not None:
                        self._finalize_tool_result(
                            session, call.call_id, payload, tools,
                        )
                        return effect
                    turn = session.submit_tool_result(
                        call.call_id, payload, tools=tools,
                    )
                    continue
                compiled = next(item for item in invocations if item.spec.name == call.name)
                prepared = self.invocation_compiler.prepare_arguments(
                    compiled, call_name=call.name, call_id=call.call_id, arguments=call.arguments,
                    occurrence=occurrence, binding_store=ctx.binding_store, evidence_store=ctx.evidence_store,
                    revision=ctx.world_revision,
                    task_contract=ctx.task_contract,
                )
                if prepared.passed:
                    effect = self._complete_from_current_effect(
                        occurrence,
                        ctx,
                        mode="preparation",
                        preferred_values=list(
                            prepared.normalized_arguments.values()
                        ),
                        provisional_bindings=prepared.binding_updates,
                    )
                    if effect is not None:
                        witness_refs = next((
                            list(record.result.get("witness_refs", []))
                            for record in reversed(
                                ctx.trace_builder.trace.validations
                            )
                            if (
                                record.occurrence_id
                                == occurrence.occurrence_id
                                and record.level == "atomic"
                            )
                        ), [])
                        ctx.trace_builder.trace.native_tool_calls.append(
                            NativeToolCallRecord(
                                call.call_id,
                                session.session_id,
                                occurrence.occurrence_id,
                                call.name,
                                dict(call.arguments),
                                "implementation_invocation",
                                {
                                    "route": "implementation_skipped_effect_satisfied",
                                    "executed": False,
                                    "atomic_effect_passed": True,
                                    "witness_refs": witness_refs,
                                },
                                f"revision:{ctx.world_revision}",
                                sum(
                                    1
                                    for item in ctx.trace_builder.trace.agent_turns
                                    if item.session_id == session.session_id
                                ) - 1,
                            )
                        )
                        self._finalize_tool_result(
                            session,
                            call.call_id,
                            to_primitive(effect),
                            tools,
                        )
                        return effect
                    preflight = self.invocation_compiler.validate_execution_context(
                        compiled,
                        prepared,
                        occurrence=occurrence,
                        binding_store=ctx.binding_store,
                        evidence_store=ctx.evidence_store,
                        revision=ctx.world_revision,
                    )
                else:
                    preflight = prepared
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

    def run_seeded_fresh(
        self,
        occurrence: Any,
        ctx: Any,
        *,
        plan_context_plan: Any | None = None,
    ) -> ImplementationExecutionResult:
        session = self.session_factory("runtime_seeded", occurrence.occurrence_id)
        record = self._record_session_start(session, "SeededSession", occurrence.occurrence_id, ctx)
        span = ctx.trace_builder.start_span("runtime_seeded", occurrence.occurrence_id)
        atomic = self.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        prompt_bindings = ctx.binding_store.runtime_prompt_projection(
            occurrence, atomic.inputs,
        )
        prompt = self.context_builder.seeded_node(
            task_goal=ctx.task_goal, atomic_contract=atomic,
            task_semantic_context=prompt_bindings["task_semantic_context"],
            current_occurrence_semantic_anchors=prompt_bindings[
                "occurrence_semantic_anchors"
            ],
            execution_ready_bindings=prompt_bindings["execution_ready_bindings"],
            missing_or_insufficient_bindings=prompt_bindings[
                "missing_or_insufficient_bindings"
            ],
            observation=ctx.observation, action_catalog=ctx.action_catalog,
            relevant_action_history=ctx.relevant_history(occurrence.occurrence_id), remaining_budget=ctx.budget.snapshot(),
            downstream_plan_context=self._downstream_plan_context(
                ctx,
                occurrence,
                plan_context_plan=plan_context_plan,
            ),
        )
        tools = self._node_tools(ctx, atomic)
        loop_guard = ActionLoopGuard()
        try:
            turn = session.next_turn(prompt, tools=tools)
            while True:
                self._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                if call.name == "report_runtime_status":
                    _status, payload = self._status_result(
                        call, session, occurrence, ctx,
                    )
                    self._finalize_tool_result(
                        session, call.call_id, payload, tools,
                    )
                    break
                if call.name == "validate_current_atomic":
                    effect, payload = self._validate_current_atomic_call(
                        call,
                        session,
                        occurrence,
                        ctx,
                        mode="seeded",
                        atomic=atomic,
                    )
                    tools = self._node_tools(ctx, atomic)
                    if effect is not None:
                        self._finalize_tool_result(
                            session, call.call_id, payload, tools,
                        )
                        return effect
                    turn = session.submit_tool_result(
                        call.call_id, payload, tools=tools,
                    )
                    continue
                payload, action_spec = self._execute_environment_call(
                    call, session, occurrence, ctx,
                    span_id=span.span_id, origin="runtime_seeded",
                    loop_guard=loop_guard,
                )
                if payload.get("loop_blocked"):
                    tools = self._node_tools(ctx, atomic)
                    if payload.get("fallback_required"):
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        result = self.not_started(
                            occurrence, failure_code="runtime_action_loop_blocked",
                        )
                        result.node_status = NodeExecutionStatus.SEEDED_FAILED
                        return result
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                effect = None
                if (
                    call.arguments["intent"] == "attempt_current_atomic"
                    and payload.get("accepted")
                ):
                    resolutions: list[AtomicEffectResolution] = []
                    effect = self._complete_from_current_effect(
                        occurrence,
                        ctx,
                        mode="seeded",
                        preferred_values=self._environment_effect_preferences(
                            occurrence,
                            ctx,
                            action_spec,
                        ),
                        resolution_out=resolutions,
                    )
                    payload["atomic_validation"] = to_primitive(
                        resolutions[-1]
                    )
                elif call.arguments["intent"] == "attempt_current_atomic":
                    payload["atomic_validation"] = to_primitive(
                        AtomicEffectResolution(
                            False,
                            failure_code="environment_action_rejected",
                            message=(
                                "Rejected environment action cannot commit "
                                "the current Atomic"
                            ),
                        )
                    )
                tools = self._node_tools(ctx, atomic)
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

    def run_dynamic(
        self,
        ctx: Any,
        *,
        rescue: bool = False,
        cold_start_continuation: bool = False,
        continuation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_kind = (
            "runtime_dynamic_cold_start_continuation"
            if cold_start_continuation
            else "runtime_dynamic"
        )
        session = self.session_factory(session_kind, "__task__")
        session_record = self._record_session_start(
            session,
            "ColdStartDynamicContinuationSession"
            if cold_start_continuation
            else "DynamicTaskSession",
            "", ctx,
        )
        span_kind = (
            "cold_start_dynamic_continuation"
            if cold_start_continuation
            else "task_rescue" if rescue else "full_dynamic"
        )
        span = ctx.trace_builder.start_span(span_kind, "", learnable=True)
        prompt = self.context_builder.dynamic_task(
            task_goal=ctx.task_goal, observation=ctx.observation, action_catalog=ctx.action_catalog,
            relevant_action_history=ctx.action_history, remaining_budget=ctx.budget.snapshot(),
            task_progress=self._task_progress_policy(ctx),
            rescue_method_guidance=(
                self._rescue_method_guidance(ctx) if rescue else None
            ),
        )
        if cold_start_continuation:
            import json
            prompt = (
                "COLD_START_CONTINUATION_CONTEXT_JSON\n"
                + json.dumps(
                    to_primitive(continuation_context or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n\n"
                + prompt
            )
        tools = [
            self._environment_tool(ctx, node_level=False),
            self._status_tool(),
        ]
        success = False
        failure_code = ""
        loop_guard = ActionLoopGuard()

        def outcome(terminal: Any) -> dict[str, Any]:
            benchmark_won = bool(
                getattr(ctx.harness.validator_channel(), "won", False)
            )
            task_contract_success = bool(
                dict(getattr(terminal, "checks", {}) or {}).get(
                    "task_contract", False,
                )
            )
            strict_success = bool(
                benchmark_won and task_contract_success
            )
            return {
                "benchmark_won": benchmark_won,
                "task_contract_success": task_contract_success,
                "strict_success": strict_success,
                # Compatibility alias; unlike the old result, its exact
                # strict meaning is now explicit beside both components.
                "success": strict_success,
                "failure_code": failure_code,
                "rescue": rescue,
                "cold_start_continuation": cold_start_continuation,
            }

        try:
            terminal = self.validation.task.terminal(
                ctx.task_contract, ctx.harness.validator_channel(), getattr(ctx.harness.validator_channel(), "won", False),
            )
            if terminal.passed:
                return outcome(terminal)
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
                    origin=(
                        "cold_start_dynamic_continuation"
                        if cold_start_continuation
                        else "task_rescue" if rescue else "full_dynamic"
                    ),
                    loop_guard=loop_guard,
                )
                if payload.get("loop_blocked"):
                    tools = [
                        self._environment_tool(ctx, node_level=False),
                        self._status_tool(),
                    ]
                    if payload.get("fallback_required"):
                        self._finalize_tool_result(session, call.call_id, payload, tools)
                        failure_code = "runtime_action_loop_blocked"
                        break
                    turn = session.submit_tool_result(call.call_id, payload, tools=tools)
                    continue
                terminal = self.validation.task.terminal(ctx.task_contract, ctx.harness.validator_channel(), payload["won"])
                tools = [
                    self._environment_tool(ctx, node_level=False),
                    self._status_tool(),
                ]
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
        terminal = self.validation.task.terminal(
            ctx.task_contract,
            ctx.harness.validator_channel(),
            bool(getattr(ctx.harness.validator_channel(), "won", False)),
        )
        result = outcome(terminal)
        # Guard against a future validator implementation accidentally
        # diverging from the loop's strict completion flag.
        result["success"] = result["strict_success"]
        return result
