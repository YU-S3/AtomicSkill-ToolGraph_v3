"""Deterministic, started-aware conversion of trace facts to ledger events.

Credit assignment consumes explicit structured facts.  It deliberately does
not inspect observations, action text, exception messages, or reasoning text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..core.errors import FailureLayer
from ..core.refs import SkillRef, ToolRef
from .ledger import EvidenceEvent, EvidenceEventType


class CreditAssignmentError(ValueError):
    """Structured trace facts are internally inconsistent."""


class CreditOutcome(str, Enum):
    DIRECT_SUCCESS = EvidenceEventType.DIRECT_SUCCESS.value
    DIRECT_FAILURE = EvidenceEventType.DIRECT_FAILURE.value
    AGENT_NODE_SUCCESS = EvidenceEventType.AGENT_NODE_SUCCESS.value
    SEEDED_SUCCESS = EvidenceEventType.SEEDED_SUCCESS.value
    SEEDED_FAILURE = EvidenceEventType.SEEDED_FAILURE.value
    SELF_SUFFICIENT_SUCCESS = EvidenceEventType.SELF_SUFFICIENT_SUCCESS.value
    TASK_RESCUE_REQUIRED = EvidenceEventType.TASK_RESCUE_REQUIRED.value
    GOAL_TERMINAL_SKIPPED = EvidenceEventType.GOAL_TERMINAL_SKIPPED.value
    CONTRACT_MISMATCH = EvidenceEventType.CONTRACT_MISMATCH.value
    SUPERSEDED = EvidenceEventType.SUPERSEDED.value


_DIRECT_KINDS = frozenset({"atomic", "implementation", "tool"})
_COMPOSITE_OUTCOMES = frozenset(
    {
        CreditOutcome.SELF_SUFFICIENT_SUCCESS,
        CreditOutcome.TASK_RESCUE_REQUIRED,
        CreditOutcome.GOAL_TERMINAL_SKIPPED,
        CreditOutcome.CONTRACT_MISMATCH,
    }
)
_FAILURE_OUTCOMES = frozenset(
    {
        CreditOutcome.DIRECT_FAILURE,
        CreditOutcome.SEEDED_FAILURE,
        CreditOutcome.TASK_RESCUE_REQUIRED,
        CreditOutcome.CONTRACT_MISMATCH,
    }
)


@dataclass(frozen=True)
class CreditAttempt:
    """The credit-relevant facts for one artifact in one runtime attempt."""

    artifact_ref: str
    artifact_kind: str
    occurrence_id: str
    attempt_id: str
    sequence_no: int
    proposed: bool = False
    validated: bool = False
    selected: bool = False
    preflight_rejected: bool = False
    started: bool = False
    outcome: CreditOutcome | None = None
    failure_layer: FailureLayer | str = ""
    intrinsic_failure: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.artifact_kind).strip().casefold()
        object.__setattr__(self, "artifact_kind", kind)
        object.__setattr__(self, "outcome", _optional_outcome(self.outcome))
        object.__setattr__(self, "failure_layer", _layer(self.failure_layer))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if kind not in {"atomic", "implementation", "tool", "composite"}:
            raise CreditAssignmentError(f"unsupported artifact kind: {kind!r}")
        for name in ("artifact_ref", "occurrence_id", "attempt_id"):
            if not str(getattr(self, name)).strip():
                raise CreditAssignmentError(f"CreditAttempt requires non-empty {name}")
        if isinstance(self.sequence_no, bool) or int(self.sequence_no) != self.sequence_no:
            raise CreditAssignmentError("CreditAttempt sequence_no must be an integer")
        if self.sequence_no < 0:
            raise CreditAssignmentError("CreditAttempt sequence_no must be non-negative")
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        if self.preflight_rejected and self.started:
            raise CreditAssignmentError("a preflight-rejected attempt cannot be started")
        if self.preflight_rejected and self.artifact_kind != "implementation":
            raise CreditAssignmentError(
                "preflight rejection is recorded only on the Implementation invocation boundary"
            )
        if self.outcome is None:
            return
        if self.outcome in _FAILURE_OUTCOMES and not self.failure_layer:
            raise CreditAssignmentError(f"{self.outcome.value} requires failure_layer")
        if self.outcome is CreditOutcome.DIRECT_SUCCESS:
            if self.artifact_kind not in _DIRECT_KINDS or not self.started:
                raise CreditAssignmentError(
                    "Direct success requires a started Atomic/Implementation/Tool attempt"
                )
        elif self.outcome is CreditOutcome.DIRECT_FAILURE:
            if self.artifact_kind not in {"implementation", "tool"} or not self.started:
                raise CreditAssignmentError(
                    "Direct failure evidence is only assigned to a started Implementation/Tool"
                )
            if self.artifact_kind == "tool" and self.failure_layer != FailureLayer.TOOL.value:
                raise CreditAssignmentError(
                    "Tool Direct failure requires started intrinsic Tool-layer failure"
                )
        elif self.outcome is CreditOutcome.AGENT_NODE_SUCCESS:
            if self.artifact_kind != "atomic" or self.started:
                raise CreditAssignmentError(
                    "agent-node success is Atomic evidence with no Learned Invocation start"
                )
        elif self.outcome in {CreditOutcome.SEEDED_SUCCESS, CreditOutcome.SEEDED_FAILURE}:
            if self.artifact_kind != "atomic":
                raise CreditAssignmentError("Seeded evidence belongs only to AbstractAtomicSkill")
            if self.outcome is CreditOutcome.SEEDED_FAILURE and not self.intrinsic_failure:
                raise CreditAssignmentError(
                    "Seeded failure is negative Atomic evidence only after inputs and lower layers are excluded"
                )
        elif self.outcome in _COMPOSITE_OUTCOMES and self.artifact_kind != "composite":
            raise CreditAssignmentError(f"{self.outcome.value} belongs only to CompositeSkill")


@dataclass(frozen=True)
class CreditTrace:
    task_id: str
    trace_id: str
    attempts: tuple[CreditAttempt, ...]
    infrastructure_failure: bool = False

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.trace_id.strip():
            raise CreditAssignmentError("CreditTrace requires task_id and trace_id")
        object.__setattr__(self, "attempts", tuple(self.attempts))


class CreditAssigner:
    """Convert standard trace facts into replay-stable EvidenceEvents."""

    def assign(self, trace: CreditTrace | Mapping[str, Any] | Any) -> list[EvidenceEvent]:
        if not isinstance(trace, CreditTrace) and bool(
            _field(trace, "infrastructure_failure", False)
        ):
            _trace_identity(trace)
            return []
        normalized = _coerce_trace(trace)
        if normalized.infrastructure_failure:
            # Infrastructure failures are globally neutral by design.
            return []

        events: list[EvidenceEvent] = []
        indexed = list(enumerate(normalized.attempts))
        indexed.sort(key=lambda item: (item[1].sequence_no, item[0]))
        for _, attempt in indexed:
            events.extend(self._assign_attempt(normalized, attempt))
        return events

    def _assign_attempt(
        self, trace: CreditTrace, attempt: CreditAttempt
    ) -> list[EvidenceEvent]:
        facts: list[tuple[EvidenceEventType, bool]] = []
        if attempt.proposed:
            facts.append((EvidenceEventType.PROPOSED, False))
        if attempt.validated:
            facts.append((EvidenceEventType.VALIDATED, False))
        if attempt.selected:
            facts.append((EvidenceEventType.SELECTED, False))
        if attempt.preflight_rejected:
            facts.append((EvidenceEventType.PREFLIGHT_REJECTED, True))
        if attempt.started and attempt.artifact_kind in {"implementation", "tool"}:
            facts.append((EvidenceEventType.EXECUTION_STARTED, False))
        if attempt.outcome is not None:
            facts.append((EvidenceEventType(attempt.outcome.value), True))

        assigned: list[EvidenceEvent] = []
        for ordinal, (event_type, terminal) in enumerate(facts):
            sequence_no = attempt.sequence_no * 100 + ordinal
            assigned.append(
                EvidenceEvent.create(
                    task_id=trace.task_id,
                    trace_id=trace.trace_id,
                    occurrence_id=attempt.occurrence_id,
                    attempt_id=attempt.attempt_id,
                    sequence_no=sequence_no,
                    artifact_ref=attempt.artifact_ref,
                    artifact_kind=attempt.artifact_kind,
                    event=event_type,
                    failure_layer=attempt.failure_layer if terminal else "",
                    confidence=attempt.confidence,
                    metadata=_event_metadata(attempt, terminal=terminal),
                )
            )
        return assigned

    def assign_evolution(
        self,
        trace: CreditTrace | Mapping[str, Any] | Any,
        atomic_refs: Iterable[SkillRef | str],
        implementation_refs: Iterable[SkillRef | str],
        tool_refs: Iterable[ToolRef | str],
        composite_ref: SkillRef | str | None,
    ) -> list[EvidenceEvent]:
        """Create proposal/admission facts for assets produced by successful extraction."""

        task_id, trace_id = _trace_identity(trace)
        assets: list[tuple[str, str]] = []
        assets.extend((str(ref), "atomic") for ref in atomic_refs)
        assets.extend((str(ref), "implementation") for ref in implementation_refs)
        assets.extend((str(ref), "tool") for ref in tool_refs)
        if composite_ref is not None:
            assets.append((str(composite_ref), "composite"))
        attempts = tuple(
            CreditAttempt(
                artifact_ref=artifact_ref,
                artifact_kind=kind,
                occurrence_id="evolution",
                attempt_id=f"evolution:{artifact_ref}",
                sequence_no=index,
                proposed=True,
                validated=True,
                metadata={"source": "extractor_admission"},
            )
            for index, (artifact_ref, kind) in enumerate(assets)
        )
        return self.assign(CreditTrace(task_id, trace_id, attempts))

    def assign_superseded(
        self,
        *,
        task_id: str,
        trace_id: str,
        old_ref: SkillRef | ToolRef | str,
        old_kind: str,
        replacement_ref: SkillRef | ToolRef | str,
        replacement_status: str,
        sequence_no: int = 0,
    ) -> list[EvidenceEvent]:
        attempt = CreditAttempt(
            artifact_ref=str(old_ref),
            artifact_kind=old_kind,
            occurrence_id="maintenance",
            attempt_id=f"supersede:{old_ref}:{replacement_ref}",
            sequence_no=sequence_no,
            outcome=CreditOutcome.SUPERSEDED,
            metadata={
                "replacement_ref": str(replacement_ref),
                "replacement_status": replacement_status,
                "reliable_replacement": replacement_status in {"active", "preferred"},
            },
        )
        return self.assign(CreditTrace(task_id, trace_id, (attempt,)))


def _coerce_trace(trace: CreditTrace | Mapping[str, Any] | Any) -> CreditTrace:
    if isinstance(trace, CreditTrace):
        return trace
    task_id, trace_id = _trace_identity(trace)
    raw_attempts = _field(trace, "credit_attempts", None)
    if raw_attempts is None:
        raw_attempts = _field(trace, "artifact_attempts", None)
    if raw_attempts is None:
        raw_attempts = _field(trace, "attempts", None)
    if raw_attempts is None:
        attempts = _derive_standard_trace_attempts(trace)
    else:
        attempts = tuple(_coerce_attempt(item) for item in raw_attempts)
    return CreditTrace(
        task_id=task_id,
        trace_id=trace_id,
        attempts=attempts,
        infrastructure_failure=bool(_field(trace, "infrastructure_failure", False)),
    )


def _coerce_attempt(value: CreditAttempt | Mapping[str, Any] | Any) -> CreditAttempt:
    if isinstance(value, CreditAttempt):
        return value
    names = {
        "artifact_ref",
        "artifact_kind",
        "occurrence_id",
        "attempt_id",
        "sequence_no",
        "proposed",
        "validated",
        "selected",
        "preflight_rejected",
        "started",
        "failure_layer",
        "intrinsic_failure",
        "confidence",
        "metadata",
    }
    payload = {name: _field(value, name) for name in names if _has_field(value, name)}
    outcome = _field(value, "outcome", _field(value, "result_event", None))
    payload["outcome"] = outcome
    try:
        return CreditAttempt(**payload)
    except TypeError as exc:
        raise CreditAssignmentError(f"invalid structured credit attempt: {exc}") from exc


def _trace_identity(trace: Mapping[str, Any] | Any) -> tuple[str, str]:
    task_id = str(_field(trace, "task_id", ""))
    if not task_id:
        task = _field(trace, "task", None)
        task_id = str(_field(task, "task_id", "")) if task is not None else ""
    trace_id = str(_field(trace, "trace_id", ""))
    if not task_id or not trace_id:
        raise CreditAssignmentError("trace must expose task_id and trace_id")
    return task_id, trace_id


def _derive_standard_trace_attempts(trace: Mapping[str, Any] | Any) -> tuple[CreditAttempt, ...]:
    """Derive credit facts from the v3 structured TraceRecord schema."""

    if not any(
        _has_field(trace, name)
        for name in ("node_records", "implementation_invocations", "tool_executions")
    ):
        raise CreditAssignmentError(
            "trace must expose structured credit_attempts or v3 TraceRecord fields; "
            "log/action text is intentionally unsupported"
        )

    attempts: list[CreditAttempt] = []
    sequence = 0
    invocations = list(_field(trace, "implementation_invocations", ()) or ())
    tool_executions = list(_field(trace, "tool_executions", ()) or ())

    for invocation in invocations:
        result = _field(invocation, "result", {}) or {}
        preflight = _field(invocation, "preflight", {}) or {}
        started = bool(_field(result, "started", False))
        preflight_passed = bool(
            _field(preflight, "passed", _field(result, "preflight_passed", False))
        )
        completed = bool(_field(result, "completed", False))
        atomic_passed = bool(_field(result, "atomic_effect_passed", False))
        failure_layer = _result_failure_layer(result, preflight)
        outcome: CreditOutcome | None = None
        if started:
            outcome = (
                CreditOutcome.DIRECT_SUCCESS
                if completed and atomic_passed
                else CreditOutcome.DIRECT_FAILURE
            )
            if outcome is CreditOutcome.DIRECT_FAILURE and not failure_layer:
                raise CreditAssignmentError(
                    f"started failed Implementation lacks failure_layer: "
                    f"{_field(invocation, 'attempt_id', '')}"
                )
        attempts.append(
            CreditAttempt(
                artifact_ref=str(_field(invocation, "implementation_ref", "")),
                artifact_kind="implementation",
                occurrence_id=str(_field(invocation, "occurrence_id", "")),
                attempt_id=str(_field(invocation, "attempt_id", "")),
                sequence_no=sequence,
                selected=True,
                preflight_rejected=not started and not preflight_passed,
                started=started,
                outcome=outcome,
                failure_layer=failure_layer,
                metadata={
                    "failure_code": str(
                        _field(result, "failure_code", _field(preflight, "failure_code", ""))
                    ),
                    "span_id": str(_field(invocation, "span_id", "")),
                },
            )
        )
        sequence += 1

    for execution in tool_executions:
        result = _field(execution, "result", {}) or {}
        started = bool(_field(result, "started", False))
        completed = bool(_field(result, "completed", False))
        failure_layer = _result_failure_layer(result)
        outcome = None
        if started and failure_layer == FailureLayer.TOOL.value:
            outcome = CreditOutcome.DIRECT_FAILURE
        elif started and completed:
            outcome = CreditOutcome.DIRECT_SUCCESS
        attempts.append(
            CreditAttempt(
                artifact_ref=str(_field(execution, "tool_ref", "")),
                artifact_kind="tool",
                occurrence_id=str(_field(execution, "occurrence_id", "")),
                attempt_id=str(_field(execution, "attempt_id", "")),
                sequence_no=sequence,
                selected=True,
                started=started,
                outcome=outcome,
                failure_layer=failure_layer,
                metadata={
                    "failure_code": str(_field(result, "failure_code", "")),
                    "span_id": str(_field(execution, "span_id", "")),
                },
            )
        )
        sequence += 1

    node_records = list(_field(trace, "node_records", ()) or ())
    for node in node_records:
        status = _enum_value(_field(node, "status", "not_started"))
        occurrence_id = str(_field(node, "occurrence_id", ""))
        atomic_ref = str(_field(node, "atomic_ref", ""))
        direct = _field(node, "direct_result", {}) or {}
        seeded = _field(node, "seeded_result", {}) or {}
        failure = _field(node, "failure", {}) or {}
        outcome: CreditOutcome | None = None
        started = bool(_field(direct, "started", False)) or any(
            str(_field(item, "occurrence_id", "")) == occurrence_id
            and bool(_field(_field(item, "result", {}) or {}, "started", False))
            for item in invocations
        )
        failure_layer = ""
        intrinsic = False
        if status in {"direct_autonomous_success", "direct_agent_prepared_success"}:
            outcome = CreditOutcome.DIRECT_SUCCESS
            if not started:
                raise CreditAssignmentError(
                    f"node {occurrence_id} is Direct success but its implementation never started"
                )
        elif status == "agent_completed_before_invocation":
            outcome = CreditOutcome.AGENT_NODE_SUCCESS
            started = False
        elif status == "seeded_success":
            outcome = CreditOutcome.SEEDED_SUCCESS
            started = False
        elif status == "seeded_failed":
            failure_layer = _result_failure_layer(failure, seeded)
            if failure_layer == FailureLayer.ATOMIC.value:
                outcome = CreditOutcome.SEEDED_FAILURE
                intrinsic = True
            started = False

        if outcome is not None:
            attempts.append(
                CreditAttempt(
                    artifact_ref=atomic_ref,
                    artifact_kind="atomic",
                    occurrence_id=occurrence_id,
                    attempt_id=f"node:{occurrence_id}:atomic",
                    sequence_no=sequence,
                    selected=True,
                    started=started,
                    outcome=outcome,
                    failure_layer=failure_layer,
                    intrinsic_failure=intrinsic,
                    metadata={"node_status": status},
                )
            )
            sequence += 1

    runtime_plan = _field(trace, "runtime_plan", {}) or {}
    composite_ref = _field(runtime_plan, "source_composite_ref", None)
    if composite_ref:
        composite_ref = str(composite_ref)
        for node in node_records:
            occurrence_id = str(_field(node, "occurrence_id", ""))
            status = _enum_value(_field(node, "status", "not_started"))
            outcome = (
                CreditOutcome.GOAL_TERMINAL_SKIPPED
                if status == "skipped_goal_terminal"
                else None
            )
            attempts.append(
                CreditAttempt(
                    artifact_ref=composite_ref,
                    artifact_kind="composite",
                    occurrence_id=occurrence_id,
                    attempt_id=f"composite:{occurrence_id}",
                    sequence_no=sequence,
                    selected=True,
                    outcome=outcome,
                    metadata={
                        "node_status": status,
                        "executed": status not in {"not_started", "skipped_goal_terminal"},
                    },
                )
            )
            sequence += 1

        composite_outcome: CreditOutcome | None = None
        composite_layer = ""
        if bool(_field(trace, "task_rescue_required", False)):
            composite_outcome = CreditOutcome.TASK_RESCUE_REQUIRED
            composite_layer = FailureLayer.COMPOSITE.value
        elif bool(_field(trace, "graph_self_sufficient_success", False)):
            composite_outcome = CreditOutcome.SELF_SUFFICIENT_SUCCESS
        elif bool(_field(trace, "benchmark_success", False)) and not bool(
            _field(trace, "learning_eligible", False)
        ):
            composite_outcome = CreditOutcome.CONTRACT_MISMATCH
            composite_layer = FailureLayer.TASK_CONTRACT.value
        if composite_outcome is not None:
            attempts.append(
                CreditAttempt(
                    artifact_ref=composite_ref,
                    artifact_kind="composite",
                    occurrence_id="graph",
                    attempt_id=f"composite:{composite_ref}:graph",
                    sequence_no=sequence,
                    outcome=composite_outcome,
                    failure_layer=composite_layer,
                )
            )

    return tuple(attempts)


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _has_field(value: Mapping[str, Any] | Any, name: str) -> bool:
    return name in value if isinstance(value, Mapping) else hasattr(value, name)


def _optional_outcome(value: CreditOutcome | EvidenceEventType | str | None) -> CreditOutcome | None:
    if value in (None, ""):
        return None
    if isinstance(value, EvidenceEventType):
        value = value.value
    return CreditOutcome(value)


def _layer(value: FailureLayer | str) -> str:
    if isinstance(value, FailureLayer):
        return value.value
    text = str(value or "")
    return FailureLayer(text).value if text else ""


def _event_metadata(attempt: CreditAttempt, *, terminal: bool) -> dict[str, Any]:
    metadata = dict(attempt.metadata)
    if not terminal:
        # Cost/latency belongs to one terminal fact, otherwise projection would
        # count the same invocation once for SELECTED and again for STARTED.
        for key in ("cost", "cost_usd", "latency_ms"):
            metadata.pop(key, None)
    metadata.update(
        {
            "started": attempt.started,
            "intrinsic_failure": _intrinsic_failure(attempt),
        }
    )
    if attempt.outcome is not None:
        metadata.setdefault("outcome", attempt.outcome.value)
    return metadata


def _result_failure_layer(*payloads: Any) -> str:
    for payload in payloads:
        layer = _field(payload, "failure_layer", "")
        if layer:
            return _layer(layer)
        failure = _field(payload, "failure", None)
        if failure is not None:
            layer = _field(failure, "layer", "")
            if layer:
                return _layer(layer)
        layer = _field(payload, "layer", "")
        if layer:
            return _layer(layer)
    return ""


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _intrinsic_failure(attempt: CreditAttempt) -> bool:
    if attempt.intrinsic_failure:
        return True
    if attempt.failure_layer == "implementation" and attempt.artifact_kind == "implementation":
        return attempt.preflight_rejected or attempt.outcome is CreditOutcome.DIRECT_FAILURE
    if attempt.failure_layer == "tool" and attempt.artifact_kind == "tool":
        return attempt.started and attempt.outcome is CreditOutcome.DIRECT_FAILURE
    if attempt.failure_layer == "composite" and attempt.artifact_kind == "composite":
        return attempt.outcome in {
            CreditOutcome.TASK_RESCUE_REQUIRED,
            CreditOutcome.CONTRACT_MISMATCH,
        }
    return False
