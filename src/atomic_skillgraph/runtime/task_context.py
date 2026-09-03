"""Finite code-side task state; not an ever-growing Agent conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.contracts import TaskContract
from ..core.results import RuntimeLinearPlan
from ..harness.protocol import HarnessActionSpec, HarnessAdapter, HarnessTask
from ..traces.schema import TraceBuilder
from .binding_store import RuntimeBindingStore
from .budget import RuntimeBudget
from .evidence_store import GroundingEvidenceStore
from .state import ExplorationMemory, OccurrenceAtomicEvidenceState, normalized_facts
from .task_progress import TaskProgressTracker


@dataclass
class TaskRuntimeContext:
    task_id: str
    task_goal: str
    task_contract: TaskContract
    plan: RuntimeLinearPlan
    current_step_index: int
    world_revision: int
    observation: str
    action_catalog: list[HarnessActionSpec]
    action_history: list[dict[str, Any]]
    binding_store: RuntimeBindingStore
    evidence_store: GroundingEvidenceStore
    validated_outputs: dict[str, dict[str, Any]]
    global_action_budget: int
    used_actions: int
    token_budget: dict[str, Any]
    trace_builder: TraceBuilder
    harness: HarnessAdapter
    task: HarnessTask
    budget: RuntimeBudget
    task_progress: TaskProgressTracker
    plan_execution_failed: bool = False
    plan_conflict_declared: bool = False
    plan_conflict_context: dict[str, Any] = field(default_factory=dict)
    task_rescue_used: bool = False
    occurrence_evidence: dict[str, OccurrenceAtomicEvidenceState] = field(
        default_factory=dict,
    )
    active_occurrence_id: str = ""
    exploration_memory: ExplorationMemory = field(default_factory=ExplorationMemory)
    grounding_state_by_occurrence: dict[str, dict[str, Any]] = field(
        default_factory=dict,
    )
    last_failed_invocation: dict[str, Any] | None = None
    terminal_latched: bool = False
    terminal_origin: str = ""
    terminal_revision: int = 0
    runtime_automation_drafts: dict[str, dict[str, Any]] = field(
        default_factory=dict,
    )
    runtime_tool_trials: dict[str, dict[str, Any]] = field(
        default_factory=dict,
    )
    _after_action_refresh: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def create(
        cls, task: HarnessTask, plan: RuntimeLinearPlan, harness: HarnessAdapter,
        trace_builder: TraceBuilder, budget: RuntimeBudget,
    ) -> "TaskRuntimeContext":
        reset = harness.reset(task)
        binding_store = RuntimeBindingStore(on_change=lambda change: trace_builder.trace.binding_changes.append(change.__dict__))
        evidence_store = GroundingEvidenceStore(
            on_change=lambda operation, evidence, revision: trace_builder.trace.grounding_evidence_changes.append({
                "evidence_id": evidence.evidence_id, "operation": operation, "revision": revision,
                "payload": evidence.__dict__,
            })
        )
        binding_store.seed_task_bindings(task, plan.task_contract, reset.new_revision)
        binding_store.configure_repeat_constraints(plan.repeat_constraints)
        evidence_store.replace_action_catalog(reset.catalog, reset.new_revision)
        progress = TaskProgressTracker(
            plan.task_contract,
            harness.validator_channel(),
            trace_builder=trace_builder,
        )
        context = cls(
            task.task_id, task.goal, plan.task_contract, plan, 0, reset.new_revision,
            reset.observation, reset.catalog, [], binding_store, evidence_store, {},
            budget.global_action_budget, 0, budget.token_limits, trace_builder,
            harness, task, budget, progress,
        )
        context.exploration_memory.observe_catalog(
            reset.catalog,
            revision=reset.new_revision,
            current_facts=normalized_facts(
                harness.validator_channel().snapshot()
            ).values(),
        )
        progress.record("task_reset")
        return context

    def begin_occurrence(self, occurrence: Any) -> OccurrenceAtomicEvidenceState:
        """Start (or resume) one occurrence without resetting its evidence."""

        occurrence_id = str(occurrence.occurrence_id)
        if self.active_occurrence_id and self.active_occurrence_id != occurrence_id:
            self.clear_failed_invocation(self.active_occurrence_id)
        self.active_occurrence_id = occurrence_id
        state = self.occurrence_evidence.get(occurrence_id)
        if state is None:
            state = OccurrenceAtomicEvidenceState.begin(
                occurrence_id,
                self.world_revision,
                self.harness.validator_channel().snapshot(),
            )
            self.occurrence_evidence[occurrence_id] = state
        return state

    def clear_active_occurrence(self) -> None:
        self.clear_failed_invocation(self.active_occurrence_id)
        self.active_occurrence_id = ""
        self._after_action_refresh = None

    def install_after_action_refresh(self, callback: Callable[[], None]) -> None:
        self._after_action_refresh = callback

    def atomic_evidence_for(
        self, occurrence: Any | str,
    ) -> OccurrenceAtomicEvidenceState:
        occurrence_id = (
            occurrence if isinstance(occurrence, str) else occurrence.occurrence_id
        )
        if occurrence_id not in self.occurrence_evidence:
            raise KeyError(f"occurrence evidence was not started: {occurrence_id}")
        return self.occurrence_evidence[str(occurrence_id)]

    def record_grounding_state(
        self,
        occurrence_id: str,
        state: dict[str, Any],
    ) -> None:
        import copy

        snapshot = copy.deepcopy(state)
        previous = self.grounding_state_by_occurrence.get(str(occurrence_id))
        self.grounding_state_by_occurrence[str(occurrence_id)] = snapshot
        signature = {
            key: snapshot.get(key)
            for key in (
                "confirmed_bindings",
                "candidate_bindings",
                "missing_bindings",
                "invalidated_bindings",
                "precondition_status",
                "effect_witness_status",
                "learned_invocation_ready",
                "blocking_reasons",
            )
        }
        self.exploration_memory.note_grounding_state(signature)
        self.trace_builder.trace.metadata.setdefault(
            "runtime_state_snapshots", []
        ).append(copy.deepcopy(snapshot))
        self.record_r3_event(
            "grounding_refresh",
            occurrence_id=str(occurrence_id),
            details={
                "learned_invocation_ready": bool(
                    snapshot.get("learned_invocation_ready", False)
                ),
                "effect_ready": bool(
                    dict(snapshot.get("effect_witness_status") or {}).get(
                        "passed", False,
                    )
                ),
            },
        )
        previous_ready = bool(
            (previous or {}).get("learned_invocation_ready", False)
        )
        current_ready = bool(snapshot.get("learned_invocation_ready", False))
        if current_ready and not previous_ready:
            self.record_r3_event(
                "invocation_ready_transition",
                occurrence_id=str(occurrence_id),
                details={"from": False, "to": True},
            )
        previous_effect = bool(
            dict((previous or {}).get("effect_witness_status") or {}).get(
                "passed", False,
            )
        )
        current_effect = bool(
            dict(snapshot.get("effect_witness_status") or {}).get(
                "passed", False,
            )
        )
        if current_effect and not previous_effect:
            self.record_r3_event(
                "effect_ready_transition",
                occurrence_id=str(occurrence_id),
                details={"from": False, "to": True},
            )

    def record_r3_event(
        self,
        event_type: str,
        *,
        occurrence_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        import copy

        self.trace_builder.trace.metadata.setdefault("r3_events", []).append({
            "revision": int(self.world_revision),
            "occurrence_id": str(occurrence_id),
            "event_type": str(event_type),
            "details": copy.deepcopy(details or {}),
        })

    def record_failed_invocation(
        self,
        *,
        occurrence_id: str,
        implementation_ref: str,
        failure_code: str,
        message: str,
    ) -> None:
        self.last_failed_invocation = {
            "occurrence_id": str(occurrence_id),
            "implementation_ref": str(implementation_ref),
            "failure_code": str(failure_code),
            "message": str(message)[:512],
            "revision": int(self.world_revision),
        }

    def clear_failed_invocation(self, occurrence_id: str = "") -> None:
        current = self.last_failed_invocation
        if current is None:
            return
        if occurrence_id and str(current.get("occurrence_id", "")) != str(
            occurrence_id
        ):
            return
        self.last_failed_invocation = None

    def update_after_action(self, result: Any, record: dict[str, Any]) -> None:
        self.observation = result.observation
        self.world_revision = result.new_revision
        self.action_catalog = list(result.catalog)
        self.action_history.append(record)
        self.used_actions = self.budget.used_global_actions
        self.binding_store.invalidate_revision(self.world_revision)
        self.evidence_store.replace_action_catalog(self.action_catalog, self.world_revision)
        self.task_progress.record("environment_action")
        validator_snapshot = self.harness.validator_channel().snapshot()
        facts = normalized_facts(validator_snapshot).values()
        occurrence_id = str(record.get("occurrence_id") or self.active_occurrence_id)
        state = self.occurrence_evidence.get(occurrence_id)
        if state is not None:
            state.reconcile(
                validator_snapshot,
                revision=self.world_revision,
                accepted=bool(result.accepted),
            )
            if bool(result.accepted):
                self.trace_builder.trace.metadata.setdefault(
                    "atomic_evidence_snapshots", []
                ).append({
                    "revision": int(self.world_revision),
                    **state.full_state(),
                })
        self.exploration_memory.record_action(
            record,
            metadata=getattr(result, "metadata", {}) or {},
            catalog=self.action_catalog,
            revision=self.world_revision,
            current_facts=facts,
        )
        if bool(result.accepted) and self._after_action_refresh is not None:
            self._after_action_refresh()
        if bool(getattr(result, "won", False)) and not self.terminal_latched:
            self.mark_terminal_latched(
                origin=str(record.get("origin") or "environment_action"),
            )

    def tool_evidence_snapshot(self) -> dict[str, Any]:
        """Unified code-authoritative projection for ToolRunner after actions.

        ToolRunner never reads Harness private state or Agent prose; it reads
        this revision-aware projection and the public action catalog.
        """

        channel_snapshot = self.harness.validator_channel().snapshot()
        if isinstance(channel_snapshot, dict):
            raw_facts = channel_snapshot.get("facts", [])
        elif isinstance(channel_snapshot, list):
            raw_facts = channel_snapshot
        else:
            raw_facts = []
        semantic_facts = list(
            normalized_facts({"facts": raw_facts}).values()
        )
        binding_evidence: list[dict[str, Any]] = []
        for evidence in self.evidence_store.active():
            payload = dict(getattr(evidence, "payload", {}) or {})
            if evidence.evidence_type not in {
                "entity_concrete", "validated_tool_output", "harness_affordance",
            }:
                continue
            binding_evidence.append({
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence.evidence_type,
                "role": str(payload.get("role", "")),
                "value": payload.get("value"),
                "action_type": str(payload.get("action_type", "")),
                "revision": int(evidence.observed_at_revision),
                "valid_from_revision": int(evidence.valid_from_revision),
            })
        return {
            "semantic_facts": semantic_facts,
            "binding_evidence": binding_evidence,
            "action_catalog": [
                {
                    "action_id": str(item.action_id),
                    "revision": int(item.revision),
                    "action_type": str(item.action_type),
                    "arguments": dict(item.arguments),
                }
                for item in self.action_catalog
            ],
            "revision": int(self.world_revision),
        }

    def relevant_history(
        self,
        occurrence_id: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the frozen recent window; the complete history stays in Trace."""

        values = [
            item
            for item in self.action_history
            if bool(item.get("accepted"))
            and (
                not occurrence_id
                or item.get("occurrence_id") in {"", occurrence_id}
            )
        ]
        selected = values[-max(0, int(limit)):] if limit else []
        projected: list[dict[str, Any]] = []
        for item in selected:
            value = {
                "action_type": str(item.get("action_type", "")),
                "arguments": dict(item.get("arguments") or {}),
                "observation": str(item.get("observation", "")),
                "revision": int(
                    item.get("new_revision", item.get("revision", 0))
                ),
                "done": bool(item.get("done", False)),
                "won": bool(item.get("won", False)),
                "origin": str(item.get("origin", "")),
            }
            if item.get("intent"):
                value["intent"] = str(item["intent"])
            projected.append(value)
        return projected

    def plan_boundary_reached(self) -> bool:
        return (
            not self.plan_execution_failed
            and not self.plan_conflict_declared
            and self.current_step_index >= len(self.plan.control_sequence)
        )

    def rescue_allowed(self) -> bool:
        """A completed graph or an Agent-declared conflict may enter Dynamic."""

        return self.plan_boundary_reached() or self.plan_conflict_declared

    def benchmark_terminal(self) -> bool:
        """Benchmark ``won`` is the sole task-terminal authority in v3.2."""

        return bool(getattr(self.harness.validator_channel(), "won", False))

    def mark_terminal_latched(
        self,
        *,
        origin: str = "",
        revision: int | None = None,
    ) -> None:
        self.terminal_latched = True
        self.terminal_origin = str(origin or "benchmark_won")
        self.terminal_revision = int(
            self.world_revision if revision is None else revision
        )
        self.record_r3_event(
            "task_terminal_latched",
            occurrence_id=self.active_occurrence_id,
            details={
                "origin": self.terminal_origin,
                "revision": self.terminal_revision,
            },
        )

    def task_complete(self) -> bool:
        return self.benchmark_terminal()
