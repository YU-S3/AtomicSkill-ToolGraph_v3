"""Runtime self-tooling coordinator: R0 -> ToolBuilder -> static -> task-local R1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.contracts import AbstractAtomicSkill
from ..core.results import ToolCallPreflightResult
from ..core.serialization import to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..tooling.builder_session import ToolBuilderSession
from ..tooling.proposal import (
    RuntimeAutomationAtomicDraft,
    ToolProvenance,
)
from ..tooling.validator import (
    ToolStaticValidator,
    normalize_runtime_output_derivations,
)


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


def _fact_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(value.get("predicate", "")).casefold(),
        repr(sorted(dict(value.get("args") or {}).items())),
    )


def _trial_harness_effect_event_authorities(
    *,
    baseline_facts: list[dict[str, Any]],
    evidence_snapshots: list[dict[str, Any]],
    environment_actions: list[Any],
    trial_event_start: int,
    trial_event_end: int,
    occurrence_id: str,
    predicate_domains: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Attribute occurrence-local Harness fact deltas to real trial events.

    The evidence state is updated immediately after each accepted Harness
    transition. Comparing those code-owned snapshots gives the exact event
    that established a richer semantic fact (for example
    ``entity.discovered_at``), without asking the later E1 bridge to guess an
    owner from the last action in the trial.
    """

    if (
        trial_event_start < 0
        or trial_event_end < trial_event_start
        or trial_event_end >= len(environment_actions)
    ):
        return []

    revision_events: dict[int, int] = {}
    ambiguous_revisions: set[int] = set()
    for event_index in range(trial_event_start, trial_event_end + 1):
        raw = environment_actions[event_index]
        action = dict(raw) if isinstance(raw, Mapping) else dict(to_primitive(raw))
        if action.get("accepted") is not True:
            continue
        revision = action.get("new_revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            continue
        if revision in revision_events:
            ambiguous_revisions.add(revision)
            continue
        revision_events[revision] = event_index
    for revision in ambiguous_revisions:
        revision_events.pop(revision, None)

    previous = {
        _fact_identity(item): dict(item)
        for item in baseline_facts
        if isinstance(item, Mapping) and str(item.get("predicate", ""))
    }
    authorities: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for raw_snapshot in evidence_snapshots:
        if not isinstance(raw_snapshot, Mapping):
            continue
        if str(raw_snapshot.get("occurrence_id", "")) != str(occurrence_id):
            continue
        revision = raw_snapshot.get("revision")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision not in revision_events
        ):
            continue
        current = {
            _fact_identity(item): dict(item)
            for item in list(raw_snapshot.get("active_facts") or [])
            if isinstance(item, Mapping) and str(item.get("predicate", ""))
        }
        for identity in sorted(set(current) - set(previous)):
            fact = current[identity]
            predicate = str(fact.get("predicate", ""))
            effect_domain = str(predicate_domains.get(predicate, ""))
            if not effect_domain:
                # Predicate vocabulary/domain is Harness authority. A richer
                # fact absent from that interface cannot become E1 authority.
                continue
            event_index = revision_events[revision]
            key = (identity[0], identity[1], event_index, revision)
            if key in seen:
                continue
            seen.add(key)
            authorities.append({
                "predicate": predicate,
                "args": dict(fact.get("args") or {}),
                "effect_domain": effect_domain,
                "event_index": int(event_index),
                "revision": int(revision),
                "source_kind": "occurrence_action_delta",
                "source_occurrence_id": str(occurrence_id),
            })
        previous = current
    return authorities


class _TaskLocalInvocation:
    """Task-local CompiledKnowledge in the ImplementationRunner view.

    The runner consumes ``implementation``/``atomic``/``tools``; the trial
    Tool is task-local and never enters a Registry or an Agent tool schema.
    """

    def __init__(self, compiled: Any) -> None:
        self.atomic = compiled.atomic
        self.implementation = compiled.implementation
        self.tools = (
            [compiled.tool] if getattr(compiled, "tool", None) is not None else []
        )


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
                # The shared R0 authority decides fresh-output derivations;
                # production R1 then takes validate_execution_result().
                "output_derivations": normalize_runtime_output_derivations(
                    draft
                ),
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
            draft, ctx.harness, ctx=ctx, occurrence=occurrence,
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
                semantic_delta=ctx.tool_evidence_snapshot(),
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

        trial_bindings = self._resolve_trial_bindings(
            draft, ctx, occurrence,
        )
        preflight = ToolCallPreflightResult(
            True,
            str(compiled.implementation.ref),
            normalized_arguments=dict(trial_bindings),
        )
        trace_actions = getattr(
            getattr(ctx.trace_builder, "trace", None),
            "environment_actions",
            (),
        )
        trial_event_start = len(trace_actions)
        trace_metadata = getattr(ctx.trace_builder.trace, "metadata", None)
        evidence_snapshots = (
            trace_metadata.setdefault("atomic_evidence_snapshots", [])
            if isinstance(trace_metadata, dict)
            else []
        )
        evidence_snapshot_start = len(evidence_snapshots)
        try:
            baseline_occurrence_facts = (
                ctx.atomic_evidence_for(occurrence).authoritative_facts()
            )
        except (AttributeError, KeyError):
            baseline_occurrence_facts = []
        result = self.implementation_runner.run(
            _TaskLocalInvocation(compiled), preflight, occurrence, ctx,
            agent_prepared=False,
        )
        trial_event_end = len(getattr(
            getattr(ctx.trace_builder, "trace", None),
            "environment_actions",
            (),
        )) - 1
        predicate_domains = {
            str(item.predicate): str(item.effect_domain)
            for item in ctx.harness.semantic_predicate_schema()
        }
        r1_effect_event_authorities = (
            _trial_harness_effect_event_authorities(
                baseline_facts=list(baseline_occurrence_facts),
                evidence_snapshots=[
                    dict(item)
                    for item in evidence_snapshots[evidence_snapshot_start:]
                    if isinstance(item, Mapping)
                ],
                environment_actions=list(trace_actions),
                trial_event_start=int(trial_event_start),
                trial_event_end=int(trial_event_end),
                occurrence_id=str(occurrence.occurrence_id),
                predicate_domains=predicate_domains,
            )
        )
        tool_results = list(result.tool_results)
        tool_completed = bool(
            tool_results and all(bool(tool.completed) for tool in tool_results)
        )
        terminal_interrupted = bool(
            tool_results and any(tool.terminal_interrupted for tool in tool_results)
        )
        tool_intrinsic_failure = bool(
            tool_results and any(tool.intrinsic_failure for tool in tool_results)
        )
        outputs_valid = bool(
            result.validated_outputs
            and all(value not in (None, "") for value in result.validated_outputs.values())
        )
        atomic_effect_passed = bool(result.atomic_effect_passed)
        executed_path_effects_passed = bool(
            tool_results
            and all(
                int(getattr(tool, "executed_step_count", 0) or 0) == 0
                or (
                    len(list(dict(
                        getattr(tool, "tool_path_evidence", {}) or {}
                    ).get("step_effect_results", [])))
                    >= int(getattr(tool, "executed_step_count", 0) or 0)
                    and all(
                        isinstance(item, dict)
                        and item.get("step_effect_passed") is True
                        for item in list(dict(
                            getattr(tool, "tool_path_evidence", {}) or {}
                        ).get("step_effect_results", []))
                    )
                )
                for tool in tool_results
            )
        )
        r1_passed = bool(
            result.started
            and atomic_effect_passed
            and executed_path_effects_passed
            and tool_completed
            and outputs_valid
            and not tool_intrinsic_failure
            and not terminal_interrupted
        )
        admission_eligible = bool(
            result.started
            and atomic_effect_passed
            and executed_path_effects_passed
            and tool_completed
            and outputs_valid
            and not tool_intrinsic_failure
            and not terminal_interrupted
        )
        # A benchmark-terminal prefix cannot admit the original Tool, but its
        # executed prefix still supplies positive task-local Atomic evidence
        # to E1.  Downstream replay/admission remains responsible for any
        # shorter Tool that the Success Extractor proposes from that prefix.
        e1_effect_eligible = bool(
            result.started
            and atomic_effect_passed
            and executed_path_effects_passed
            and outputs_valid
            and not tool_intrinsic_failure
            and (admission_eligible or terminal_interrupted)
        )
        input_authorities: dict[str, dict[str, Any]] = {}
        for role, raw in (dict(getattr(draft, "input_binding_specs", None) or {})).items():
            if role not in trial_bindings:
                continue
            spec = dict(raw) if isinstance(raw, dict) else {}
            input_authorities[role] = {
                "kind": str(spec.get("kind", "")).casefold(),
                "source_occurrence_id": str(occurrence.occurrence_id),
                "source_role": str(spec.get("source_role", "")),
                "value": trial_bindings[role],
                "authority_ref": f"runtime_input:{draft.draft_id}:{role}",
            }
        tool_path_witness_refs: list[str] = []
        for tool_result in result.tool_results:
            evidence = dict(getattr(tool_result, "tool_path_evidence", {}) or {})
            tool_path_witness_refs.extend(
                str(item) for item in evidence.get("evidence_refs", [])
            )
            for step in evidence.get("step_effect_results", []):
                if isinstance(step, dict) and step.get("witness_refs"):
                    tool_path_witness_refs.extend(map(str, step.get("witness_refs", [])))
        r1_witness_refs = list(dict.fromkeys(
            str(ref) for ref in result.atomic_witness_refs
        ))
        trial = {
            "draft_id": draft.draft_id,
            "atomic_ref": str(atomic.ref),
            "tool_ref": str(compiled.tool.ref),
            "implementation_ref": str(compiled.implementation.ref),
            "trial_bindings": to_primitive(trial_bindings),
            "input_authorities": to_primitive(input_authorities),
            "r1_outputs": to_primitive(result.validated_outputs),
            "r1_witness_refs": r1_witness_refs,
            "tool_path_witness_refs": list(dict.fromkeys(tool_path_witness_refs)),
            "result": to_primitive(result),
            "r1": {
                "started": bool(result.started),
                "atomic_effect_passed": atomic_effect_passed,
                "executed_path_effects_passed": executed_path_effects_passed,
                "tool_completed": tool_completed,
                "terminal_interrupted": terminal_interrupted,
                "outputs_valid": outputs_valid,
                "tool_intrinsic_failure": tool_intrinsic_failure,
                "admission_eligible": admission_eligible,
                "e1_effect_eligible": e1_effect_eligible,
            },
            "terminal_interrupted": terminal_interrupted,
            "trial_event_start": int(trial_event_start),
            "trial_event_end": int(trial_event_end),
            "r1_effect_event_authorities": to_primitive(
                r1_effect_event_authorities
            ),
        }
        if e1_effect_eligible:
            trial.update({
                "declared_effects": to_primitive(list(compiled.atomic.effects)),
                "output_derivations": to_primitive(dict(
                    compiled.atomic.validator_spec.get("output_derivations") or {}
                )),
                "after_revision": int(tool_results[-1].after_revision),
            })
        ctx.runtime_tool_trials[draft.draft_id] = trial
        return RuntimeAutomationOutcome(
            True, to_primitive(r0),
            proposal=to_primitive(proposal),
            static_passed=True,
            static_report=to_primitive(static),
            trial=trial,
            r1_passed=bool(r1_passed),
            r1_report=trial["r1"],
            failure_code="" if r1_passed else "runtime_automation_r1_rejected",
            message="" if r1_passed else "task-local trial did not pass full R1",
        )

    @staticmethod
    def _resolve_trial_bindings(
        draft: RuntimeAutomationAtomicDraft,
        ctx: Any,
        occurrence: Any,
    ) -> dict[str, Any]:
        """Resolve task-local Automation inputs from their frozen binding specs."""

        bindings: dict[str, Any] = {}
        specs = dict(getattr(draft, "input_binding_specs", None) or {})
        snapshot = ctx.binding_store.snapshot_for_node(occurrence)
        validated_outputs = getattr(ctx, "validated_outputs", {}) or {}
        for role, raw in specs.items():
            spec = dict(raw) if isinstance(raw, dict) else {}
            kind = str(spec.get("kind", "")).casefold()
            source_role = str(spec.get("source_role", ""))
            if kind == "current_occurrence_anchor":
                anchor = ctx.binding_store.semantic_anchor_for(
                    occurrence, source_role,
                )
                value = getattr(anchor, "value", None)
            elif kind in {"current_confirmed_binding", "current_candidate_binding"}:
                binding = snapshot.get(source_role)
                value = getattr(binding, "value", None)
            elif kind == "data_flow":
                value = validated_outputs.get(
                    occurrence.occurrence_id, {},
                ).get(source_role)
                if value in (None, ""):
                    output_binding = ctx.binding_store.validated_outputs(
                        occurrence.occurrence_id,
                    ).get(source_role)
                    value = getattr(output_binding, "value", None)
            elif kind == "constant":
                value = spec.get("value")
            else:
                value = None
            if value is not None:
                bindings[role] = value
        return bindings

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
