"""Finite code-side task state; not an ever-growing Agent conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import TaskContract
from ..core.results import RuntimeLinearPlan
from ..harness.protocol import HarnessActionSpec, HarnessAdapter, HarnessTask
from ..traces.schema import TraceBuilder
from .binding_store import RuntimeBindingStore
from .budget import RuntimeBudget
from .evidence_store import GroundingEvidenceStore
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
        progress.record("task_reset")
        return context

    def update_after_action(self, result: Any, record: dict[str, Any]) -> None:
        old_revision = self.world_revision
        self.observation = result.observation
        self.world_revision = result.new_revision
        self.action_catalog = list(result.catalog)
        self.action_history.append(record)
        self.used_actions = self.budget.used_global_actions
        self.binding_store.invalidate_revision(self.world_revision)
        self.evidence_store.replace_action_catalog(self.action_catalog, self.world_revision)
        self.task_progress.record("environment_action")

    def relevant_history(self, occurrence_id: str) -> list[dict[str, Any]]:
        return [item for item in self.action_history if item.get("occurrence_id") in {"", occurrence_id}]

    def plan_boundary_reached(self) -> bool:
        return (
            not self.plan_execution_failed
            and not self.plan_conflict_declared
            and self.current_step_index >= len(self.plan.control_sequence)
        )

    def rescue_allowed(self) -> bool:
        """A completed graph or an Agent-declared conflict may enter Dynamic."""

        return self.plan_boundary_reached() or self.plan_conflict_declared

    def task_complete(self) -> bool:
        validation = self.harness.validator_channel().validate_task_contract(self.task_contract)
        return bool(validation.passed and getattr(self.harness.validator_channel(), "won", False))
