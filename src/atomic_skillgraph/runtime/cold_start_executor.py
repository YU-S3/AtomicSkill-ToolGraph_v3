"""Execution helpers for isolated v3.1 cold-start scaffolds."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..core.bindings import BindingExpression
from ..core.contracts import (
    AbstractAtomicSkill,
    ColdStartPlanStep,
    ParameterSpec,
    SemanticPredicate,
)
from ..core.errors import AgentProtocolError, AtomicSkillGraphError, FailureLayer
from ..core.refs import SkillRef
from ..core.results import NodeExecutionStatus, RuntimeOccurrence
from ..core.serialization import to_primitive
from ..core.status import SkillStatus
from .loop_guard import ActionLoopGuard


@dataclass
class ProvisionalTrialResult:
    provisional_ref: str
    step_id: str
    local_effect_passed: bool
    progress_before_digest: str
    progress_after_digest: str
    action_span: tuple[int, int]
    witness_refs: list[str]
    failure_code: str
    resolved_bindings: dict[str, Any] = field(default_factory=dict)


def _parameter(value: Any) -> ParameterSpec:
    return value if isinstance(value, ParameterSpec) else ParameterSpec(**dict(value))


def _predicate(value: Any) -> SemanticPredicate:
    if isinstance(value, SemanticPredicate):
        return value
    args = {
        str(key): (
            BindingExpression.from_dict(raw)
            if isinstance(raw, dict) and "kind" in raw
            else raw
        )
        for key, raw in dict(value.get("args") or {}).items()
    }
    return SemanticPredicate(
        str(value["predicate"]), args,
        int(value.get("cardinality", 1)), str(value.get("distinct_by", "")),
        value.get("effect_domain", "world"),
    )


def provisional_atomic_view(record: Any) -> AbstractAtomicSkill:
    contract = dict(record.atomic_contract)
    logical_hash = hashlib.sha256(
        str(record.provisional_ref).encode("utf-8")
    ).hexdigest()[:20]
    return AbstractAtomicSkill(
        ref=SkillRef(f"provisional_atomic_{logical_hash}", "1.0.0"),
        summary=str(record.canonical_intent),
        inputs=[_parameter(value) for value in contract.get("inputs", ())],
        outputs=[_parameter(value) for value in contract.get("outputs", ())],
        preconditions=[_predicate(value) for value in contract.get("preconditions", ())],
        effects=[_predicate(value) for value in contract.get("effects", ())],
        validator_spec=dict(contract.get("validator_spec") or {}),
        failure_modes=[],
        guideline=dict(record.seeded_guideline),
        metadata={
            "origin": "failure_side_provisional",
            "harness_profiles": [record.harness_profile],
        },
        status=SkillStatus.DRAFT,
    )


class ProvisionalNodeExecutor:
    """Fresh Seeded execution with no learned invocation surface."""

    def __init__(self, node_executor: Any) -> None:
        self.node_executor = node_executor

    def execute(
        self,
        provisional: Any,
        ctx: Any,
        step: ColdStartPlanStep,
        *,
        progress_tracker: Any,
    ) -> ProvisionalTrialResult:
        atomic = provisional_atomic_view(provisional)
        occurrence = RuntimeOccurrence(
            step_id=step.step_id,
            occurrence_id=f"cold::{step.step_id}",
            node_ref=atomic.ref,
            requirement_ids=list(step.requirement_instance_ids),
            binding_specs=dict(step.binding_specs),
            implementation_candidates=[],
            expected_effects=list(atomic.effects),
            requirement_instance_ids=list(step.requirement_instance_ids),
            repeat_role_bindings=dict(step.repeat_role_bindings),
        )
        ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
        before = progress_tracker.record("cold_start_step_start")
        action_start = len(ctx.trace_builder.trace.environment_actions)
        session = self.node_executor.session_factory(
            "runtime_provisional_seeded", occurrence.occurrence_id,
        )
        session_record = self.node_executor._record_session_start(
            session, "ProvisionalSeededSession", occurrence.occurrence_id, ctx,
        )
        span = ctx.trace_builder.start_span(
            "runtime_provisional_seeded", occurrence.occurrence_id,
        )
        prompt_bindings = ctx.binding_store.runtime_prompt_projection(
            occurrence, atomic.inputs,
        )
        prompt = self.node_executor.context_builder.seeded_node(
            task_goal=ctx.task_goal,
            atomic_contract=atomic,
            task_semantic_context=prompt_bindings["task_semantic_context"],
            current_occurrence_semantic_anchors=prompt_bindings[
                "occurrence_semantic_anchors"
            ],
            execution_ready_bindings=prompt_bindings["execution_ready_bindings"],
            missing_or_insufficient_bindings=prompt_bindings[
                "missing_or_insufficient_bindings"
            ],
            observation=ctx.observation,
            action_catalog=ctx.action_catalog,
            relevant_action_history=ctx.relevant_history(occurrence.occurrence_id),
            remaining_budget=ctx.budget.snapshot(),
        )
        tools = self.node_executor._node_tools(ctx, atomic)
        loop_guard = ActionLoopGuard()
        failure_code = "provisional_atomic_effect_failed"
        resolved: dict[str, Any] = {}
        witness_refs: list[str] = []
        try:
            turn = session.next_turn(prompt, tools=tools)
            while True:
                self.node_executor._record_turn(session, turn, ctx)
                call = turn.tool_calls[0]
                if call.name == "report_runtime_status":
                    self.node_executor._finalize_tool_result(
                        session, call.call_id, {"accepted": True}, tools,
                    )
                    break
                if call.name == "validate_current_atomic":
                    effect, payload = (
                        self.node_executor._validate_current_atomic_call(
                            call,
                            session,
                            occurrence,
                            ctx,
                            mode="seeded",
                            atomic=atomic,
                        )
                    )
                    tools = self.node_executor._node_tools(ctx, atomic)
                    if effect is not None:
                        self.node_executor._finalize_tool_result(
                            session, call.call_id, payload, tools,
                        )
                        resolved, witness_refs = self._record_success(
                            effect, occurrence, ctx,
                        )
                        failure_code = ""
                        break
                    turn = session.submit_tool_result(
                        call.call_id, payload, tools=tools,
                    )
                    continue
                payload, action_spec = self.node_executor._execute_environment_call(
                    call, session, occurrence, ctx,
                    span_id=span.span_id,
                    origin="runtime_provisional_seeded",
                    loop_guard=loop_guard,
                )
                progress_tracker.record("environment_action")
                tools = self.node_executor._node_tools(ctx, atomic)
                if payload.get("loop_blocked"):
                    if payload.get("fallback_required"):
                        failure_code = str(
                            payload.get("error")
                            or "provisional_atomic_effect_failed"
                        )
                        if failure_code.startswith("runtime_repetition_"):
                            failure_code = failure_code.replace(
                                "runtime_repetition_", "provisional_repetition_",
                                1,
                            )
                        elif "budget_exhausted" in failure_code:
                            failure_code = "provisional_seeded_budget_exhausted"
                        else:
                            failure_code = "provisional_atomic_effect_failed"
                        self.node_executor._finalize_tool_result(
                            session, call.call_id, payload, tools,
                        )
                        break
                    turn = session.submit_tool_result(
                        call.call_id, payload, tools=tools,
                    )
                    continue
                effect = None
                if (
                    call.arguments["intent"] == "attempt_current_atomic"
                    and payload.get("accepted")
                ):
                    resolutions = []
                    effect = self.node_executor._complete_from_current_effect(
                        occurrence,
                        ctx,
                        mode="seeded",
                        preferred_values=list(action_spec.arguments.values()),
                        atomic_override=atomic,
                        resolution_out=resolutions,
                    )
                    payload["atomic_validation"] = to_primitive(
                        resolutions[-1]
                    )
                elif call.arguments["intent"] == "attempt_current_atomic":
                    payload["atomic_validation"] = {
                        "passed": False,
                        "failure_code": "environment_action_rejected",
                        "message": (
                            "Rejected environment action cannot commit the "
                            "provisional Atomic"
                        ),
                    }
                if effect is not None:
                    self.node_executor._finalize_tool_result(
                        session, call.call_id, payload, tools,
                    )
                    resolved, witness_refs = self._record_success(
                        effect, occurrence, ctx,
                    )
                    failure_code = ""
                    break
                if payload.get("done"):
                    self.node_executor._finalize_tool_result(
                        session, call.call_id, payload, tools,
                    )
                    break
                turn = session.submit_tool_result(
                    call.call_id, payload, tools=tools,
                )
        except AtomicSkillGraphError as exc:
            if exc.layer is FailureLayer.INFRASTRUCTURE:
                raise
            if isinstance(exc, AgentProtocolError):
                # Protocol failures are neutral lifecycle evidence.  Preserve
                # that classification for the post-task failure-side commit;
                # do not count it toward consecutive local-effect failures.
                failure_code = "provisional_provider_or_protocol_failure"
            elif "budget_exhausted" in exc.code:
                failure_code = "provisional_seeded_budget_exhausted"
            elif exc.code.startswith("runtime_repetition_"):
                failure_code = exc.code.replace(
                    "runtime_repetition_",
                    "provisional_repetition_",
                    1,
                )
            else:
                failure_code = "provisional_atomic_effect_failed"
        finally:
            ctx.trace_builder.finish_span(span.span_id)
            self.node_executor._finish_session(session_record, session, ctx)

        after = progress_tracker.record("cold_start_step_complete")
        action_end = len(ctx.trace_builder.trace.environment_actions)
        return ProvisionalTrialResult(
            provisional_ref=str(provisional.provisional_ref),
            step_id=step.step_id,
            local_effect_passed=not failure_code,
            progress_before_digest=before.progress_digest,
            progress_after_digest=after.progress_digest,
            action_span=(action_start, action_end),
            witness_refs=witness_refs,
            failure_code=failure_code,
            resolved_bindings=resolved,
        )

    @staticmethod
    def _record_success(
        effect: Any,
        occurrence: RuntimeOccurrence,
        ctx: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        resolved = {
            key: value.value
            for key, value in ctx.binding_store.snapshot_for_node(
                occurrence,
            ).items()
        }
        witness_refs = next((
            list(item.result.get("witness_refs", ()))
            for item in reversed(ctx.trace_builder.trace.validations)
            if item.occurrence_id == occurrence.occurrence_id
            and item.level == "atomic"
        ), [])
        if effect.validated_outputs:
            if not witness_refs:
                witness_refs = [
                    f"validator:occurrence:{occurrence.occurrence_id}:"
                    f"revision:{ctx.world_revision}"
                ]
            ctx.binding_store.publish_validated_outputs(
                occurrence,
                effect.validated_outputs,
                witness_refs,
                ctx.world_revision,
            )
            ctx.validated_outputs[occurrence.occurrence_id] = dict(
                effect.validated_outputs
            )
            for role, value in effect.validated_outputs.items():
                ctx.evidence_store.add_validated_tool_output(
                    role,
                    value,
                    witness_refs,
                )
        return resolved, witness_refs


__all__ = [
    "ProvisionalNodeExecutor",
    "ProvisionalTrialResult",
    "provisional_atomic_view",
]
