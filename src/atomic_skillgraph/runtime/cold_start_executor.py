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
from ..tooling.runtime_interface import build_runtime_automation_interface
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
        failure_code = ""
        resolved, witness_refs = {}, []
        try:
            effect = self.node_executor.run_agent_node(occurrence, ctx, mode="seeded", atomic_override=atomic)
            if effect.atomic_effect_passed:
                resolved, witness_refs = self._record_success(effect, occurrence, ctx)
            else:
                failure_code = effect.failure_code or "provisional_atomic_effect_failed"
        except AtomicSkillGraphError as exc:
            if exc.layer is FailureLayer.INFRASTRUCTURE:
                raise
            failure_code = exc.code
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
        witness_refs = list(effect.atomic_witness_refs)
        if effect.validated_outputs:
            if not witness_refs:
                raise ValueError("validated provisional outputs require actual witnesses")
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
