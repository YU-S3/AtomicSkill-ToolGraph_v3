"""Runtime IR and standardized result envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bindings import GroundingConstraint, RuntimeBinding
from .contracts import ColdStartPlanProposal, SemanticPredicate, TaskContract
from .edges import GraphEdge
from .refs import SkillRef, ToolRef


@dataclass
class ValidationResult:
    level: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failure_codes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    witness_refs: list[str] = field(default_factory=list)
    before_ref: str = ""
    after_ref: str = ""

    @classmethod
    def ok(cls, level: str, **checks: bool) -> "ValidationResult":
        return cls(level=level, passed=True, checks=checks)

    @classmethod
    def fail(cls, level: str, code: str, message: str, **checks: bool) -> "ValidationResult":
        return cls(level=level, passed=False, checks=checks, failure_codes=[code], messages=[message])


@dataclass
class AtomicEffectResolution:
    """Deterministic action-derived witness resolution for one Atomic effect."""

    passed: bool
    resolved_bindings: dict[str, Any] = field(default_factory=dict)
    output_candidates: dict[str, Any] = field(default_factory=dict)
    witness_refs: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    failure_code: str = ""
    message: str = ""


@dataclass
class RuntimeOccurrence:
    step_id: str
    occurrence_id: str
    node_ref: SkillRef
    requirement_ids: list[str]
    binding_specs: dict[str, Any]
    implementation_candidates: list[SkillRef]
    expected_effects: list[SemanticPredicate]
    status: str = "not_started"
    requirement_instance_ids: list[str] = field(default_factory=list)
    repeat_role_bindings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeRepeatConstraint:
    block_id: str
    count: int
    iteration_steps: tuple[tuple[str, ...], ...]
    distinct_roles: tuple[str, ...]
    shared_roles: tuple[str, ...]
    step_role_bindings: dict[str, dict[str, str]]
    basis_constraint_id: str = ""


@dataclass
class RuntimeLinearPlan:
    task_id: str
    source: str
    source_composite_ref: str | None
    occurrences: list[RuntimeOccurrence]
    control_sequence: list[str]
    data_edges: list[GraphEdge]
    dependency_edges: list[GraphEdge]
    task_contract: TaskContract
    planner_audit: dict[str, Any]
    repeat_constraints: list[RuntimeRepeatConstraint] = field(default_factory=list)
    cold_start_plan: ColdStartPlanProposal | None = None
    cold_start_scaffold: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def full_dynamic(
        cls, task_id: str, contract: TaskContract, *, reason: str,
        audit: dict[str, Any] | None = None,
    ) -> "RuntimeLinearPlan":
        details = dict(audit or {})
        details.setdefault("final_outcome", "full_dynamic")
        details.setdefault("fallback_reason", reason)
        return cls(task_id, "full_dynamic", None, [], [], [], [], contract, details)

    @classmethod
    def cold_start(
        cls,
        task_id: str,
        contract: TaskContract,
        *,
        proposal: ColdStartPlanProposal,
        scaffold: dict[str, Any],
        audit: dict[str, Any],
    ) -> "RuntimeLinearPlan":
        details = dict(audit)
        details["final_outcome"] = "cold_start"
        return cls(
            task_id, "cold_start", None, [], [], [], [], contract, details,
            [], proposal, dict(scaffold),
        )

    def occurrence(self, step_id: str) -> RuntimeOccurrence:
        for occurrence in self.occurrences:
            if occurrence.step_id == step_id:
                return occurrence
        raise KeyError(step_id)


class NodeExecutionStatus(str, Enum):
    NOT_STARTED = "not_started"
    ALREADY_SATISFIED = "already_satisfied"
    DIRECT_AUTONOMOUS_SUCCESS = "direct_autonomous_success"
    DIRECT_AGENT_PREPARED_SUCCESS = "direct_agent_prepared_success"
    AGENT_COMPLETED_BEFORE_INVOCATION = "agent_completed_before_invocation"
    SEEDED_SUCCESS = "seeded_success"
    FAILED_NOT_STARTED = "failed_not_started"
    DIRECT_FAILED = "direct_failed"
    SEEDED_FAILED = "seeded_failed"
    SKIPPED_GOAL_TERMINAL = "skipped_goal_terminal"


@dataclass
class ImplementationInvocationSpec:
    name: str
    implementation_ref: SkillRef
    atomic_ref: SkillRef
    description: str
    input_schema: dict[str, Any]
    grounding_constraints: list[GroundingConstraint]
    tool_refs: list[ToolRef]
    execution_policy: dict[str, Any]


@dataclass
class ToolCallPreflightResult:
    passed: bool
    implementation_ref: str
    normalized_arguments: dict[str, Any] = field(default_factory=dict)
    binding_updates: list[RuntimeBinding] = field(default_factory=list)
    matched_evidence_refs: list[str] = field(default_factory=list)
    failure_layer: str = ""
    failure_code: str = ""
    message: str = ""


@dataclass
class ToolExecutionResult:
    tool_ref: str
    preflight_passed: bool
    started: bool
    completed: bool
    state_changed: bool
    executed_step_count: int
    failure_step_index: int | None
    partial_effects: list[dict[str, Any]]
    output_candidates: dict[str, Any]
    before_revision: int
    after_revision: int
    failure_layer: str = ""
    failure_code: str = ""
    failure_message: str = ""
    terminal_interrupted: bool = False
    intrinsic_failure: bool = False
    executed_node_count: int = 0
    remaining_node_count: int = 0
    path_id: str = ""
    program_node_id: str = ""
    expected_effects: list[dict[str, Any]] = field(default_factory=list)
    observed_effects: list[dict[str, Any]] = field(default_factory=list)
    missing_effects: list[dict[str, Any]] = field(default_factory=list)
    loop_iteration_counts: dict[str, int] = field(default_factory=dict)
    validated_paths: list[str] = field(default_factory=list)
    unvalidated_paths: list[str] = field(default_factory=list)
    stop_condition_witnesses: list[str] = field(default_factory=list)
    atomic_effect_passed: bool = False


@dataclass
class ImplementationExecutionResult:
    implementation_ref: str
    atomic_ref: str
    preflight_passed: bool
    started: bool
    completed: bool
    atomic_effect_passed: bool
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    realized_bindings: dict[str, RuntimeBinding] = field(default_factory=dict)
    validated_outputs: dict[str, Any] = field(default_factory=dict)
    before_state_ref: str = ""
    after_state_ref: str = ""
    failure_layer: str = ""
    failure_code: str = ""
    node_status: NodeExecutionStatus = NodeExecutionStatus.NOT_STARTED
    terminal_interrupted: bool = False


@dataclass(frozen=True)
class PrimitiveToolStep:
    action_type: str
    argument_mapping: dict[str, Any]


@dataclass
class TaskOutcome:
    task_id: str
    trace_id: str
    benchmark_success: bool
    task_contract_success: bool
    strict_task_success: bool
    node_contract_success: bool
    implementation_direct_success: bool
    graph_self_sufficient_success: bool
    graph_full_completion: bool
    learning_eligible: bool
    failure_code: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
