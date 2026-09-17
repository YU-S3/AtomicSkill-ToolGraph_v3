"""Runtime self-tooling coordinator: R0 -> ToolBuilder -> static -> task-local R1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.errors import AgentProtocolError, BudgetExhausted
from ..core.contracts import AbstractAtomicSkill
from ..core.results import ToolCallPreflightResult
from ..core.serialization import to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..tooling.builder_session import ToolBuilderSession, ToolProposalParseError
from ..tooling.proposal import (
    RuntimeAutomationAtomicDraft,
    ToolProvenance,
)
from ..tooling.validator import (
    ToolStaticValidator,
    normalize_runtime_output_derivations,
)
from ..tooling.runtime_interface import (
    build_runtime_automation_interface,
    public_tool_ir_collection_sources,
    resolve_runtime_automation_inputs,
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
    stage: str = ""
    cache_hit: bool = False


def _increment_funnel(ctx: Any, field: str, amount: int = 1) -> None:
    trace = ctx.trace_builder.trace
    metadata = getattr(trace, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        trace.metadata = metadata
    funnel = metadata.setdefault("runtime_automation_funnel", {})
    funnel[field] = int(funnel.get(field, 0)) + int(amount)


def safe_runtime_automation_outcome(
    outcome: RuntimeAutomationOutcome,
) -> dict[str, Any]:
    """Return the Runtime-Agent view without executable or validator internals."""

    projected = {
        "r0_passed": bool(outcome.r0_passed),
        "static_passed": bool(outcome.static_passed),
        "r1_passed": bool(outcome.r1_passed),
        "failure_code": str(outcome.failure_code),
        "message": str(outcome.message),
        "stage": str(outcome.stage),
        "cache_hit": bool(outcome.cache_hit),
    }
    if outcome.trial is not None:
        projected["trial"] = {
            key: to_primitive(outcome.trial[key])
            for key in (
                "draft_id", "r1_outputs", "r1", "terminal_interrupted",
                "failure_feedback",
            )
            if key in outcome.trial
        }
        projected["trial_failure"] = to_primitive(outcome.trial.get("failure_feedback", {}))
    return projected


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
                "output_semantic_constraints": to_primitive(draft.output_semantic_constraints),
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
        input_resolution = resolve_runtime_automation_inputs(
            draft, ctx, occurrence,
        )
        r0 = self.static_validator.validate_automation_draft(
            draft,
            ctx.harness,
            ctx=ctx,
            occurrence=occurrence,
            input_resolution=input_resolution,
        )
        if not r0.passed:
            return RuntimeAutomationOutcome(
                False, to_primitive(r0),
                failure_code=(r0.failure_codes[0] if r0.failure_codes else "runtime_automation_r0_rejected"),
                message="; ".join(r0.messages),
                stage="r0_rejected",
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
            _increment_funnel(ctx, "builder_called")
            public_interface = build_runtime_automation_interface(
                ctx.harness, occurrence, ctx.binding_store,
            )
            proposal = builder.build(
                atomic=atomic,
                provenance=provenance,
                evidence_support=[],
                # A pre-execution Runtime draft has no public effect witness.
                # Validator facts remain code-only input to ToolRunner/R1 and
                # must never be copied into the ToolBuilder Agent prompt.
                semantic_delta={},
                harness_interface={
                    "profile": getattr(ctx.harness, "profile_name", ""),
                    "predicate_vocabulary": public_interface[
                        "predicate_vocabulary"
                    ],
                    "primitive_actions": public_interface["primitive_actions"],
                    "tool_ir_collection_sources": (
                        public_tool_ir_collection_sources()
                    ),
                    "runtime_entry": {
                        "revision": ctx.world_revision,
                        "remaining_resources": {
                            **(self.runtime_resources(ctx.budget.current_occurrence_id)
                               if callable(getattr(self, "runtime_resources", None)) else {}),
                            "node_actions": ctx.budget.remaining_node_actions,
                            "task_actions": ctx.budget.remaining_global_actions,
                        },
                        "consumer_obligation": {"atomic_ref": str(occurrence.node_ref),
                            "step_id": occurrence.step_id,
                            "repeat": ctx.binding_store.repeat_execution_frame(occurrence.step_id)},
                        "input_values": dict(input_resolution.values),
                        "action_catalog": [
                            {"action_type": action.action_type,
                             "arguments": dict(action.arguments)}
                            for action in ctx.action_catalog
                        ],
                    },
                    "public_catalog_relations": getattr(
                        ctx.harness, "public_catalog_relation_schema", lambda: [],
                    )(),
                },
                bucket="tool_builder_runtime",
            )
        except BudgetExhausted as exc:
            _increment_funnel(ctx, "budget_rejected")
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                failure_code=exc.code or "runtime_automation_builder_budget_exhausted",
                message=str(exc),
                stage="builder_budget_rejected",
            )
        except (AgentProtocolError, ToolProposalParseError) as exc:
            _increment_funnel(ctx, "content_rejected")
            return RuntimeAutomationOutcome(
                True,
                to_primitive(r0),
                failure_code="runtime_automation_builder_content_rejected",
                message=str(exc),
                stage="builder_content_rejected",
            )
        if proposal.decision == "no_tool":
            _increment_funnel(ctx, "no_tool")
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                proposal=to_primitive(proposal),
                static_passed=False,
                static_report={"decision": "no_tool"},
                failure_code="runtime_automation_no_tool",
                message=proposal.rationale,
                stage="builder_no_tool",
            )
        static = self.static_validator.validate_proposal(
            proposal, atomic, ctx.harness,
        )
        if not static.passed:
            _increment_funnel(ctx, "static_reject")
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                proposal=to_primitive(proposal),
                static_passed=False,
                static_report=to_primitive(static),
                failure_code=(static.failure_codes[0] if static.failure_codes else "runtime_automation_static_rejected"),
                message="; ".join(static.messages),
                stage="static_rejected",
            )
        _increment_funnel(ctx, "static_pass")
        try:
            compiled = self.tool_compiler.compile_proposal(
                self._synthetic_occurrence(draft, ctx, occurrence),
                atomic,
                proposal,
                provenance,
            )
            compiled.tool.status = ToolStatus.CANDIDATE
            compiled.implementation.status = SkillStatus.CANDIDATE
        except (KeyError, ValueError) as exc:
            return RuntimeAutomationOutcome(
                True, to_primitive(r0),
                proposal=to_primitive(proposal),
                static_passed=True,
                static_report=to_primitive(static),
                failure_code="runtime_automation_compile_failed",
                message=str(exc),
                stage="compile_rejected",
            )

        trial_bindings = dict(input_resolution.values)
        from .invocation_transaction import execution_cache_key, cache_lookup
        failure_key = execution_cache_key(_TaskLocalInvocation(compiled), trial_bindings, occurrence, ctx)
        cached = cache_lookup(ctx, failure_key, occurrence)
        if cached is not None:
            return RuntimeAutomationOutcome(True, to_primitive(r0), proposal=to_primitive(proposal),
                static_passed=True, static_report=to_primitive(static),
                failure_code=cached['failure_code'], message='Identical program, arguments and consumer state previously failed; no action was repeated',
                stage='deterministic_failure_cached', cache_hit=True)
        preflight = ToolCallPreflightResult(
            True,
            str(compiled.implementation.ref),
            normalized_arguments=dict(trial_bindings),
            binding_updates=list(input_resolution.binding_updates),
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
        implementation_record_start = len(
            getattr(ctx.trace_builder.trace, "implementation_invocations", ())
        )
        tool_record_start = len(
            getattr(ctx.trace_builder.trace, "tool_executions", ())
        )
        from .invocation_transaction import InvocationTransaction
        with InvocationTransaction(ctx, occurrence, origin="runtime_trial", failure_code="runtime_automation_r1_rejected") as transaction:
            result = self.implementation_runner.run(
                _TaskLocalInvocation(compiled), preflight, occurrence, ctx,
                agent_prepared=False,
                execution_scope="runtime_trial",
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
            input_authorities = dict(input_resolution.input_authorities)
            implementation_attempt_ids = [
                str(item.attempt_id)
                for item in list(
                    getattr(
                        ctx.trace_builder.trace,
                        "implementation_invocations",
                        (),
                    )
                )[implementation_record_start:]
            ]
            tool_execution_ids = [
                str(item.attempt_id)
                for item in list(
                    getattr(ctx.trace_builder.trace, "tool_executions", ())
                )[
                    tool_record_start:
                ]
            ]
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
                "source_occurrence_id": str(occurrence.occurrence_id),
                "atomic_ref": str(atomic.ref),
                "tool_ref": str(compiled.tool.ref),
                "implementation_ref": str(compiled.implementation.ref),
                "trial_bindings": to_primitive(trial_bindings),
                "input_authorities": to_primitive(input_authorities),
                "implementation_attempt_ids": implementation_attempt_ids,
                "tool_execution_ids": tool_execution_ids,
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
                "parent_resumed_after_trial": False,
                "parent_completed_after_trial": False,
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
            if r1_passed and getattr(ctx, "runtime_config", {}).get("persistent_runtime_support_promotion"):
                from dataclasses import replace
                from ..evolution.contract_canonicalizer import AtomicContractCanonicalizer
                from ..evolution.portability import contract_label
                from ..evolution.tool_compiler import build_occurrence_replay_case
                from ..traces.canonical import canonical_action_indices
                canonicalizer = AtomicContractCanonicalizer()
                source = replace(
                    compiled.occurrence, input_bindings=dict(trial_bindings),
                    output_bindings=dict(result.validated_outputs),
                    source_task={}, event_start=trial_event_start, event_end=trial_event_end,
                    action_events=[to_primitive(trace_actions[i]) for i in range(trial_event_start, trial_event_end + 1)],
                    prefix_events=[to_primitive(trace_actions[i]) for i in canonical_action_indices(ctx.trace_builder.trace)
                                   if i < trial_event_start],
                )
                # R1 provenance is task-local evidence, not the reusable contract.
                # Preserve all semantic/output constraints; move only source ids
                # out of the prospective persistent validator specification.
                persistent_spec = {key: value for key, value in compiled.atomic.validator_spec.items()
                                   if key not in {"task_local", "occurrence_id", "trace_id"}}
                persistent_spec["validator_id"] = "harness_atomic_effect"
                persistent_atomic = replace(compiled.atomic, validator_spec=persistent_spec,
                    summary=contract_label(compiled.atomic.effects, compiled.atomic.outputs),
                    inputs=[replace(p, description="") for p in compiled.atomic.inputs],
                    outputs=[replace(p, description="") for p in compiled.atomic.outputs],
                    guideline={"runtime_automation": True, "steps": []},
                    metadata={"runtime_support_promotion": True,
                              "source_runtime_trace_id": ctx.trace_builder.trace.trace_id})
                bundle = canonicalizer.canonicalize(persistent_atomic, compiled.tool, compiled.implementation)
                bundle.tool.summary = persistent_atomic.summary
                bundle.implementation.summary = persistent_atomic.summary
                source = canonicalizer.rewrite_canonical_occurrence(source, bundle, atomic_ref=bundle.atomic.ref)
                bundle.tool.tests = [build_occurrence_replay_case(
                    source, bundle.atomic, source_task=ctx.task, kind="tool_proposal_replay",
                )]
                trial["promotion_bundle"] = {
                    "atomic": to_primitive(bundle.atomic), "tool": to_primitive(bundle.tool),
                    "implementation": to_primitive(bundle.implementation),
                }
                trial["tool_proposal"] = to_primitive(proposal)
            transaction.accepted = r1_passed
        trial["failure_feedback"] = {
            "failure_code": result.failure_code,
            "failure_layer": result.failure_layer,
            "message": next((r.failure_message for r in result.tool_results if r.failure_code), ""),
            "started": result.started, "not_committed": not r1_passed,
            "rollback": bool(transaction.checkpoint and not r1_passed and not ctx.benchmark_terminal()),
            "restored_revision": ctx.world_revision,
            "failing_action": next((r.tool_path_evidence.get('attempted_action', {})
                for r in result.tool_results if r.failure_code), {}),
            "stage": "r1" if not r1_passed else "committed",
        }
        if failure_key and not r1_passed and result.failure_layer in {'tool', 'atomic', 'runtime_binding', 'implementation'}:
            ctx.rejected_runtime_candidates[failure_key] = {
                'failure_code': result.failure_code or 'runtime_automation_r1_rejected',
                'failure_layer': result.failure_layer}
        _increment_funnel(ctx, "trial_started", int(bool(result.started)))
        _increment_funnel(ctx, "trial_completed", int(bool(tool_completed)))
        _increment_funnel(
            ctx, "trial_terminal_interrupted", int(bool(terminal_interrupted)),
        )
        internal_actions = max(0, trial_event_end - trial_event_start + 1)
        _increment_funnel(ctx, "trial_internal_action_count", internal_actions)
        _increment_funnel(
            ctx, "trial_llm_bypassed_action_count", internal_actions,
        )
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
            stage="r1_passed" if r1_passed else "r1_rejected",
        )

    @staticmethod
    def _resolve_trial_bindings(
        draft: RuntimeAutomationAtomicDraft,
        ctx: Any,
        occurrence: Any,
    ) -> dict[str, Any]:
        """Compatibility view over the single R0/trial input resolver."""

        return dict(
            resolve_runtime_automation_inputs(draft, ctx, occurrence).values
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
