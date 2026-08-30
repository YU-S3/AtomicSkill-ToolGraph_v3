"""Every provenance boundary is represented explicitly; logs are never re-parsed."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

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
    logical_call_id: str = ""
    attempt_index: int = 1
    is_terminal_attempt: bool = True


@dataclass(frozen=True)
class LogicalProviderCallRecord:
    """One logical model call and all of its audited transport attempts."""

    logical_call_id: str
    attempts: tuple[ProviderRequestRecord | Mapping[str, Any], ...]
    final_outcome: str
    final_usage_status: str
    unmetered_transport_attempt_count: int


def provider_request_accounting(
    records: Iterable[ProviderRequestRecord | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize HTTP attempts without treating retries as extra logical calls.

    Resource usage belongs to the terminal response of one logical provider
    call.  A transient transport/HTTP attempt may have no usage metadata; when
    its logical call later terminates with reported usage, accounting is still
    complete.  Missing attempts, multiple terminal attempts, or an unreported
    terminal attempt fail closed.
    """

    grouped: dict[str, list[ProviderRequestRecord | Mapping[str, Any]]] = {}
    for ordinal, record in enumerate(records, start=1):
        request_id = str(_provider_record_field(record, "request_id", ""))
        logical_call_id = str(
            _provider_record_field(record, "logical_call_id", "")
        )
        if not logical_call_id:
            # Backward-compatible traces had no logical id and therefore keep
            # their historical one-record-per-call interpretation.
            logical_call_id = request_id or f"legacy_provider_attempt_{ordinal}"
        grouped.setdefault(logical_call_id, []).append(record)

    incomplete: list[str] = []
    logical_calls: list[LogicalProviderCallRecord] = []
    transient_retry_count = 0
    unmetered_transport_attempt_count = 0
    all_logical_calls_succeeded = bool(grouped)
    for logical_call_id, attempts in grouped.items():
        ordered = sorted(attempts, key=_provider_attempt_index)
        indexes = [_provider_attempt_index(item) for item in ordered]
        terminal = [
            item for item in ordered
            if _provider_record_field(item, "is_terminal_attempt", True) is True
        ]
        nonterminal = [
            item for item in ordered
            if _provider_record_field(item, "is_terminal_attempt", True) is not True
        ]
        transient_retry_count += len(nonterminal)
        unmetered = sum(
            _provider_record_field(item, "http_status", None) != 200
            and str(_provider_record_field(item, "usage_status", "unavailable"))
            != "reported"
            for item in ordered
        )
        unmetered_transport_attempt_count += unmetered
        final = terminal[0] if len(terminal) == 1 else None
        final_outcome = str(
            _provider_record_field(final, "outcome", "") if final is not None else ""
        )
        final_usage_status = str(
            _provider_record_field(final, "usage_status", "unavailable")
            if final is not None else "unavailable"
        )
        logical_calls.append(LogicalProviderCallRecord(
            logical_call_id=logical_call_id,
            attempts=tuple(ordered),
            final_outcome=final_outcome,
            final_usage_status=final_usage_status,
            unmetered_transport_attempt_count=unmetered,
        ))
        structurally_valid = (
            indexes == list(range(1, len(ordered) + 1))
            and len(terminal) == 1
            and terminal[0] is ordered[-1]
        )
        every_200_is_metered = all(
            _provider_record_field(item, "http_status", None) != 200
            or str(_provider_record_field(item, "usage_status", "unavailable"))
            == "reported"
            for item in ordered
        )
        terminal_metered_200 = bool(
            final is not None
            and _provider_record_field(final, "http_status", None) == 200
            and final_usage_status == "reported"
        )
        if not (
            structurally_valid and every_200_is_metered and terminal_metered_200
        ):
            incomplete.append(logical_call_id)
        if not (
            structurally_valid
            and final is not None
            and _provider_record_field(final, "http_status", None) == 200
            and final_outcome == "success"
        ):
            all_logical_calls_succeeded = False

    return {
        "http_attempt_count": sum(len(items) for items in grouped.values()),
        "logical_call_count": len(grouped),
        "transient_retry_count": transient_retry_count,
        "unmetered_transport_attempt_count": unmetered_transport_attempt_count,
        "resource_usage_complete": not incomplete,
        "all_logical_calls_succeeded": all_logical_calls_succeeded,
        "incomplete_logical_call_ids": sorted(incomplete),
        "logical_calls": tuple(logical_calls),
    }


def _provider_record_field(record: Any, name: str, default: Any) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _provider_attempt_index(record: Any) -> int:
    value = _provider_record_field(record, "attempt_index", 1)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


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
    transition_certificate: dict[str, Any] | None = None


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
    resource_usage_complete: bool = True
    benchmark_success: bool = False
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
