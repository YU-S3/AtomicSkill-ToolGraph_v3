"""Every provenance boundary is represented explicitly; logs are never re-parsed."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import FailureEnvelope
from ..core.results import NodeExecutionStatus, RuntimeLinearPlan, ValidationResult


@dataclass
class TaskRecord:
    task_id: str
    benchmark: str
    goal: str
    task_type: str
    task_signature: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskProgressRecord:
    revision: int
    source: str
    snapshot: dict[str, Any]


@dataclass
class AgentSessionRecord:
    session_id: str
    session_type: str
    occurrence_id: str
    started_at: float
    ended_at: float = 0.0
    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurnRecord:
    session_id: str
    turn_index: int
    content: str
    finish_reason: str
    tool_call_ids: list[str]
    usage: dict[str, Any]
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderRequestRecord:
    """One sanitized HTTP attempt at the model-provider boundary."""

    request_id: str
    session_id: str
    stage: str
    started_at: float
    ended_at: float
    outcome: str
    http_status: int | None
    retry_count: int
    usage_status: str
    error_code: str
    sanitized_error: str
    payload_fingerprint: str


@dataclass
class NativeToolCallRecord:
    call_id: str
    session_id: str
    occurrence_id: str
    tool_name: str
    arguments: dict[str, Any]
    call_kind: str
    preflight_result: dict[str, Any]
    result_ref: str | None
    turn_index: int


@dataclass
class EnvironmentActionRecord:
    action_id: str
    revision: int
    action_type: str
    arguments: dict[str, Any]
    accepted: bool
    observation: str
    done: bool
    won: bool
    new_revision: int
    span_id: str


@dataclass
class ImplementationInvocationRecord:
    attempt_id: str
    occurrence_id: str
    implementation_ref: str
    arguments: dict[str, Any]
    preflight: dict[str, Any]
    result: dict[str, Any]
    span_id: str


@dataclass
class ToolExecutionRecord:
    attempt_id: str
    occurrence_id: str
    tool_ref: str
    result: dict[str, Any]
    span_id: str


@dataclass
class GroundingEvidenceChange:
    evidence_id: str
    operation: str
    revision: int
    payload: dict[str, Any]


@dataclass
class ValidationRecord:
    occurrence_id: str
    level: str
    result: dict[str, Any]
    revision: int


@dataclass
class RuntimeSpan:
    span_id: str
    kind: str
    occurrence_id: str
    action_start: int
    action_end: int
    parent_span_id: str | None
    learnable: bool


@dataclass
class NodeTraceRecord:
    occurrence_id: str
    step_id: str
    atomic_ref: str
    status: NodeExecutionStatus = NodeExecutionStatus.NOT_STARTED
    direct_result: dict[str, Any] = field(default_factory=dict)
    seeded_result: dict[str, Any] = field(default_factory=dict)
    validated_outputs: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = NodeExecutionStatus(self.status)


@dataclass
class ColdStartPlanRecord:
    plan_id: str
    proposal: dict[str, Any]
    validation: dict[str, Any]
    repair_used: bool
    executable_step_ids: list[str]
    first_unresolved_step_id: str


@dataclass
class ColdStartStepRecord:
    step_id: str
    candidate_source: str
    candidate_ref: str
    execution_mode: str
    outcome: str
    local_effect_passed: bool
    action_start: int
    action_end: int
    progress_before: str
    progress_after: str
    failure_code: str


@dataclass
class FailureExtractionRecord:
    f1_alignment: dict[str, Any]
    f1_validation: dict[str, Any]
    f2_proposal: dict[str, Any]
    provisional_refs: list[str]
    failure_experience_ids: list[str]
    rejection: dict[str, Any]


@dataclass
class TraceRecord:
    trace_id: str
    schema_version: int
    task: TaskRecord
    task_contract: dict[str, Any]
    planner_audit: dict[str, Any]
    runtime_plan: dict[str, Any]
    started_at: float
    ended_at: float = 0.0
    node_records: list[NodeTraceRecord] = field(default_factory=list)
    agent_sessions: list[AgentSessionRecord] = field(default_factory=list)
    agent_turns: list[AgentTurnRecord] = field(default_factory=list)
    native_tool_calls: list[NativeToolCallRecord] = field(default_factory=list)
    environment_actions: list[EnvironmentActionRecord] = field(default_factory=list)
    implementation_invocations: list[ImplementationInvocationRecord] = field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = field(default_factory=list)
    binding_changes: list[dict[str, Any]] = field(default_factory=list)
    grounding_evidence_changes: list[GroundingEvidenceChange] = field(default_factory=list)
    validations: list[ValidationRecord] = field(default_factory=list)
    failures: list[FailureEnvelope] = field(default_factory=list)
    evidence_event_refs: list[str] = field(default_factory=list)
    runtime_spans: list[RuntimeSpan] = field(default_factory=list)
    llm_usage: list[dict[str, Any]] = field(default_factory=list)
    provider_requests: list[ProviderRequestRecord] = field(default_factory=list)
    requirement_bundle: dict[str, Any] = field(default_factory=dict)
    requirement_expansion: dict[str, Any] = field(default_factory=dict)
    task_progress_records: list[TaskProgressRecord] = field(default_factory=list)
    cold_start_plan: ColdStartPlanRecord | None = None
    cold_start_steps: list[ColdStartStepRecord] = field(default_factory=list)
    failure_extraction: FailureExtractionRecord | None = None
    provisional_promotions: list[dict[str, Any]] = field(default_factory=list)
    cold_start_assisted_success: bool = False
    resource_usage_complete: bool = True
    benchmark_success: bool = False
    task_contract_success: bool = False
    strict_task_success: bool = False
    node_contract_success: bool = False
    implementation_direct_success: bool = False
    graph_self_sufficient_success: bool = False
    graph_full_completion: bool = False
    task_rescue_required: bool = False
    learning_eligible: bool = False
    infrastructure_failure: bool = False
    extraction_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls, task: TaskRecord, task_contract: dict[str, Any],
        planner_audit: dict[str, Any], runtime_plan: dict[str, Any],
    ) -> "TraceRecord":
        return cls(
            trace_id=f"trace_{uuid.uuid4().hex}", schema_version=3, task=task,
            task_contract=task_contract, planner_audit=planner_audit,
            runtime_plan=runtime_plan, started_at=time.time(),
            metadata={"method_patch": "3.1"},
        )

    def finish(self) -> "TraceRecord":
        self.ended_at = time.time()
        return self

    def node(self, occurrence_id: str) -> NodeTraceRecord:
        for node in self.node_records:
            if node.occurrence_id == occurrence_id:
                return node
        raise KeyError(occurrence_id)

    def attempt_started(self, artifact_ref: str, occurrence_id: str) -> bool:
        for invocation in self.implementation_invocations:
            if invocation.occurrence_id == occurrence_id and invocation.implementation_ref == artifact_ref:
                return bool(invocation.result.get("started"))
        for execution in self.tool_executions:
            if execution.occurrence_id == occurrence_id and execution.tool_ref == artifact_ref:
                return bool(execution.result.get("started"))
        return False


class TraceBuilder:
    def __init__(self, trace: TraceRecord) -> None:
        self.trace = trace
        self._open_spans: dict[str, RuntimeSpan] = {}

    def start_node(self, occurrence_id: str, step_id: str, atomic_ref: str) -> NodeTraceRecord:
        node = NodeTraceRecord(occurrence_id, step_id, atomic_ref)
        self.trace.node_records.append(node)
        return node

    def start_span(
        self, kind: str, occurrence_id: str, *, parent_span_id: str | None = None,
        learnable: bool = True,
    ) -> RuntimeSpan:
        span = RuntimeSpan(
            span_id=f"span_{uuid.uuid4().hex}", kind=kind, occurrence_id=occurrence_id,
            action_start=len(self.trace.environment_actions), action_end=-1,
            parent_span_id=parent_span_id, learnable=learnable,
        )
        self.trace.runtime_spans.append(span)
        self._open_spans[span.span_id] = span
        return span

    def finish_span(self, span_id: str) -> RuntimeSpan:
        span = self._open_spans.pop(span_id)
        span.action_end = len(self.trace.environment_actions)
        return span

    def finish(self) -> TraceRecord:
        for span_id in list(self._open_spans):
            self.finish_span(span_id)
        return self.trace.finish()
