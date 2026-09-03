"""Runtime self-tooling coordinator: R0 -> ToolBuilder -> static -> task-local R1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.contracts import AbstractAtomicSkill
from ..core.results import ToolCallPreflightResult
from ..core.serialization import to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..tooling.builder_session import ToolBuilderSession
from ..tooling.proposal import (
    RuntimeAutomationAtomicDraft,
    ToolProvenance,
)
from ..tooling.validator import ToolStaticValidator


@dataclass
class RuntimeAutomationOutcome:
    r0_passed: bool
    r0_report: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    static_passed: bool = False
    static_report: dict[str, Any] = field(default_factory=dict)
    trial: dict[str, Any] | None = None
    r1_passed: bool = False
    r1_report: dict[str, Any] = field(default_factory=dict)
    failure_code: str = ""
    message: str = ""


class RuntimeAutomationCoordinator:
    """Owns the only Runtime path from an Agent draft to a task-local Tool."""

    def __init__(
        self,
        *,
        tool_builder_factory: Callable[[str, str], Any],
        tool_compiler: Any,
        implementation_runner: Any,
        static_validator: ToolStaticValidator | None = None,
    ) -> None:
        self.tool_builder_factory = tool_builder_factory
        self.tool_compiler = tool_compiler
        self.implementation_runner = implementation_runner
        self.static_validator = static_validator or ToolStaticValidator()

    @staticmethod
    def _draft_atomic(
        draft: RuntimeAutomationAtomicDraft,
        *,
        occurrence_id: str,
        trace_id: str,
    ) -> AbstractAtomicSkill:
        from ..core.refs import SkillRef

        logical = "atomic_" + "".join(
            char if char.isalnum() else "_" for char in draft.intent.casefold()
        ).strip("_")[:40] or "runtime_automation"
        ref = SkillRef(f"{logical}_task_local", "1.0.0")
        return AbstractAtomicSkill(
            ref,
            draft.intent,
            draft.inputs,
            draft.outputs,
            draft.preconditions,
            draft.effects,
            {
                "validator_id": "runtime_automation_r1",
                "identity_strict": True,
                "task_local": True,
                "occurrence_id": occurrence_id,
                "trace_id": trace_id,
            },
            [],
            {"steps": [], "runtime_automation": True},
            {
                "task_local": True,
                "draft_id": draft.draft_id,
                "source_occurrence_id": draft.source_occurrence_id,
                "trace_id": trace_id,
            },
            SkillStatus.CANDIDATE,
        )

    def process_draft(
        self,
        *,
        draft: RuntimeAutomationAtomicDraft,
        ctx: Any,
        occurrence: Any,
    ) -> RuntimeAutomationOutcome:
        r0 = self.static_validator.validate_automation_draft(
            draft, ctx.harness,
        )
        if not r0.passed:
            return RuntimeAutomationOutcome(
                False, to_primitive(r0),
                failure_code=(r0.failure_codes[0] if r0.failure_codes else "runtime_automation_r0_rejected"),
                message="; ".join(r0.messages),
            )
        atomic = self._draft_atomic(
            draft,
            occurrence_id=str(occurrence.occurrence_id),
            trace_id=str(ctx.trace_builder.trace.trace_id),
        )
        provenance = ToolProvenance(
            source="runtime_automation",
            atomic_ref=str(atomic.ref),
            source_trace_id=str(ctx.trace_builder.trace.trace_id),
            occurrence_id=str(occurrence.occurrence_id),
            draft_id=draft.draft_id,
            task_id=str(ctx.task_id),
        )
        try:
            session = self.tool_builder_factory(
                "tool_builder_runtime", occurrence.occurrence_id,
            )
            builder = ToolBuilderSession(session)
            proposal = builder.build(
                atomic=atomic,
                provenance=provenance,
                evidence_support=[],
                semantic_delta={
                    "observation": ctx.observation,
                    "revision": int(ctx.world_revision),
                },
                harness_interface={
                    "profile": getattr(ctx.harness, "profile_name", ""),
                    "predicate_vocabulary": to_primitive(
                        ctx.harness.semantic_predicate_schema()
                    ),
                    "primitive_actions": self._primitive_actions(ctx),
                },
                bucket="tool_builder_runtime",
            )
        except Exception as exc:
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                failure_code="runtime_automation_tool_builder_failed",
                message=str(exc),
            )
        if proposal.decision == "no_tool":
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                proposal=to_primitive(proposal),
                static_passed=True,
                static_report={"decision": "no_tool"},
                failure_code="runtime_automation_no_tool",
                message=proposal.rationale,
            )
        static = self.static_validator.validate_proposal(
            proposal, atomic, ctx.harness,
        )
        if not static.passed:
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                proposal=to_primitive(proposal),
                static_passed=False,
                static_report=to_primitive(static),
                failure_code=(static.failure_codes[0] if static.failure_codes else "runtime_automation_static_rejected"),
                message="; ".join(static.messages),
            )
        try:
            compiled = self.tool_compiler.compile_proposal(
                self._synthetic_occurrence(draft, ctx, occurrence),
                atomic,
                proposal,
                provenance,
            )
            compiled.tool.status = ToolStatus.CANDIDATE
            compiled.implementation.status = SkillStatus.CANDIDATE
        except Exception as exc:
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                proposal=to_primitive(proposal),
                static_passed=True,
                static_report=to_primitive(static),
                failure_code="runtime_automation_compile_failed",
                message=str(exc),
            )

        trial_bindings = {
            role: binding.value
            for role, binding in ctx.binding_store.snapshot_for_node(
                occurrence,
            ).items()
            if binding.value is not None
        }
        preflight = ToolCallPreflightResult(
            True,
            str(compiled.implementation.ref),
            normalized_arguments=dict(trial_bindings),
        )
        result = self.implementation_runner.run(
            compiled, preflight, occurrence, ctx, agent_prepared=False,
        )
        r1_passed = bool(
            result.started
            and result.atomic_effect_passed
            and all(
                bool(tool.completed or tool.terminal_interrupted)
                for tool in result.tool_results
            )
        )
        trial = {
            "draft_id": draft.draft_id,
            "atomic_ref": str(atomic.ref),
            "tool_ref": str(compiled.tool.ref),
            "implementation_ref": str(compiled.implementation.ref),
            "result": to_primitive(result),
            "terminal_interrupted": bool(
                result.tool_results
                and result.tool_results[0].terminal_interrupted
            ),
        }
        ctx.runtime_tool_trials[draft.draft_id] = trial
        return RuntimeAutomationOutcome(
            True, to_primitive(r0),
            proposal=to_primitive(proposal),
            static_passed=True,
            static_report=to_primitive(static),
            trial=trial,
            r1_passed=bool(r1_passed),
            r1_report={"passed": bool(r1_passed)},
            failure_code="" if r1_passed else "runtime_automation_r1_rejected",
            message="" if r1_passed else "task-local trial did not pass R1",
        )

    def _synthetic_occurrence(self, draft: Any, ctx: Any, occurrence: Any):
        from ..evolution.atomicizer import CanonicalAtomicOccurrence
        from ..core.refs import SkillRef

        return CanonicalAtomicOccurrence(
            occurrence_id=f"runtime_auto::{draft.draft_id}",
            phase_id=draft.draft_id,
            intent=draft.intent,
            event_start=0,
            event_end=max(0, len(ctx.action_history) - 1),
            input_bindings={
                item.name: item.name for item in draft.inputs
            },
            output_bindings={
                item.name: item.name for item in draft.outputs
            },
            input_specs=list(draft.inputs),
            output_specs=list(draft.outputs),
            preconditions=list(draft.preconditions),
            effects=list(draft.effects),
            action_events=list(ctx.action_history),
            prefix_events=[],
            source_task={
                "task_id": ctx.task_id,
                "task_type": getattr(ctx.task, "task_type", ""),
            },
            source_trace_id=str(ctx.trace_builder.trace.trace_id),
            proposed_ref=SkillRef("atomic_runtime_draft", "1.0.0"),
        )

    @staticmethod
    def _primitive_actions(ctx: Any) -> list[dict[str, Any]]:
        schema_method = getattr(ctx.harness, "primitive_action_schema", None)
        if callable(schema_method):
            return [dict(item) for item in schema_method()]
        seen: dict[str, set[tuple[str, ...]]] = {}
        for item in ctx.action_history:
            action_type = str(item.get("action_type", ""))
            if action_type:
                seen.setdefault(action_type, set()).add(
                    tuple(sorted(dict(item.get("arguments") or {}).keys()))
                )
        return [
            {"action_type": action_type, "argument_roles": sorted(roles)}
            for action_type, roles in sorted(seen.items())
        ]


__all__ = ["RuntimeAutomationCoordinator", "RuntimeAutomationOutcome"]
