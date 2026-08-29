"""Append-only, exactly-once evidence ledger.

The ledger is the sole durable fact source for lifecycle statistics.  Replaying
the same event is a no-op; reusing an event id or natural id for different
content fails closed instead of silently changing history.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from ..core.errors import FailureLayer
from ..core.serialization import to_primitive
from ..knowledge.database import SCHEMA_VERSION, StateDatabase


class EvidenceEventType(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    SELECTED = "selected"
    PREFLIGHT_REJECTED = "preflight_rejected"
    EXECUTION_STARTED = "execution_started"
    DIRECT_SUCCESS = "direct_success"
    DIRECT_FAILURE = "direct_failure"
    AGENT_NODE_SUCCESS = "agent_node_success"
    SEEDED_SUCCESS = "seeded_success"
    SEEDED_FAILURE = "seeded_failure"
    SELF_SUFFICIENT_SUCCESS = "self_sufficient_success"
    TASK_RESCUE_REQUIRED = "task_rescue_required"
    GOAL_TERMINAL_SKIPPED = "goal_terminal_skipped"
    CONTRACT_MISMATCH = "contract_mismatch"
    SUPERSEDED = "superseded"


ARTIFACT_KINDS = frozenset({"atomic", "implementation", "tool", "composite"})


class EvidenceLedgerError(RuntimeError):
    """Base class for ledger contract violations."""


class EvidenceConflictError(EvidenceLedgerError):
    """An exactly-once identity was reused for different evidence."""


def deterministic_event_id(
    trace_id: str,
    attempt_id: str,
    artifact_ref: str,
    event: EvidenceEventType | str,
    sequence_no: int,
) -> str:
    """Return a replay-stable id derived from the ledger natural key."""

    event = EvidenceEventType(event)
    raw = "\x1f".join((trace_id, attempt_id, artifact_ref, event.value, str(sequence_no)))
    return f"evt_{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    schema_version: int
    task_id: str
    trace_id: str
    occurrence_id: str
    attempt_id: str
    sequence_no: int
    artifact_ref: str
    artifact_kind: str
    event: EvidenceEventType
    failure_layer: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", EvidenceEventType(self.event))
        object.__setattr__(self, "artifact_kind", str(self.artifact_kind).strip().casefold())
        object.__setattr__(self, "failure_layer", _failure_layer_value(self.failure_layer))
        object.__setattr__(self, "metadata", dict(to_primitive(self.metadata)))

        required = {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "occurrence_id": self.occurrence_id,
            "attempt_id": self.attempt_id,
            "artifact_ref": self.artifact_ref,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"EvidenceEvent requires non-empty {', '.join(missing)}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"EvidenceEvent schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.artifact_kind not in ARTIFACT_KINDS:
            raise ValueError(f"unsupported evidence artifact kind: {self.artifact_kind!r}")
        if isinstance(self.sequence_no, bool) or int(self.sequence_no) != self.sequence_no:
            raise ValueError("sequence_no must be an integer")
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be a finite value in [0, 1]")
        try:
            _metadata_json(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("EvidenceEvent metadata must be canonical-JSON serializable") from exc

    @property
    def natural_key(self) -> tuple[str, str, str, str, int]:
        return (
            self.trace_id,
            self.attempt_id,
            self.artifact_ref,
            self.event.value,
            self.sequence_no,
        )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        trace_id: str,
        occurrence_id: str,
        attempt_id: str,
        sequence_no: int,
        artifact_ref: str,
        artifact_kind: str,
        event: EvidenceEventType | str,
        failure_layer: FailureLayer | str = "",
        confidence: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> "EvidenceEvent":
        event_type = EvidenceEventType(event)
        return cls(
            event_id=event_id
            or deterministic_event_id(trace_id, attempt_id, artifact_ref, event_type, sequence_no),
            schema_version=SCHEMA_VERSION,
            task_id=task_id,
            trace_id=trace_id,
            occurrence_id=occurrence_id,
            attempt_id=attempt_id,
            sequence_no=sequence_no,
            artifact_ref=artifact_ref,
            artifact_kind=artifact_kind,
            event=event_type,
            failure_layer=_failure_layer_value(failure_layer),
            confidence=confidence,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "EvidenceEvent":
        return cls(
            event_id=str(row["event_id"]),
            schema_version=int(row["schema_version"]),
            task_id=str(row["task_id"]),
            trace_id=str(row["trace_id"]),
            occurrence_id=str(row["occurrence_id"]),
            attempt_id=str(row["attempt_id"]),
            sequence_no=int(row["sequence_no"]),
            artifact_ref=str(row["artifact_ref"]),
            artifact_kind=str(row["artifact_kind"]),
            event=EvidenceEventType(row["event_type"]),
            failure_layer=str(row["failure_layer"]),
            confidence=float(row["confidence"]),
            metadata=json.loads(str(row["metadata_json"])),
        )

    def database_values(self) -> tuple[Any, ...]:
        return (
            self.event_id,
            self.schema_version,
            self.task_id,
            self.trace_id,
            self.occurrence_id,
            self.attempt_id,
            self.sequence_no,
            self.artifact_ref,
            self.artifact_kind,
            self.event.value,
            self.failure_layer,
            float(self.confidence),
            _metadata_json(self.metadata),
        )


@dataclass(frozen=True)
class LedgerRecord:
    rowid: int
    event: EvidenceEvent


@dataclass(frozen=True)
class AppendResult:
    requested_count: int
    inserted_count: int
    duplicate_count: int
    last_event_rowid: int


_INSERT = """
INSERT INTO evidence_events(
    event_id, schema_version, task_id, trace_id, occurrence_id, attempt_id,
    sequence_no, artifact_ref, artifact_kind, event_type, failure_layer,
    confidence, metadata_json
) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


class EvidenceLedger:
    """Transactional append/read API over ``evidence_events``."""

    def __init__(self, database: StateDatabase) -> None:
        self.database = database

    def append_transaction(self, events: Iterable[EvidenceEvent]) -> AppendResult:
        if self.database.readonly:
            raise RuntimeError("frozen evidence ledger is read-only")
        requested = list(events)
        for event in requested:
            if not isinstance(event, EvidenceEvent):
                raise TypeError("EvidenceLedger only accepts EvidenceEvent instances")
        if not requested:
            return AppendResult(0, 0, 0, self.max_rowid())

        unique: list[EvidenceEvent] = []
        by_id: dict[str, EvidenceEvent] = {}
        by_natural: dict[tuple[str, str, str, str, int], EvidenceEvent] = {}
        in_batch_duplicates = 0
        for event in requested:
            id_match = by_id.get(event.event_id)
            natural_match = by_natural.get(event.natural_key)
            for match in (id_match, natural_match):
                if match is not None and not _same_event(match, event):
                    raise EvidenceConflictError(
                        f"conflicting evidence identity in append batch: {event.event_id}"
                    )
            if id_match is not None or natural_match is not None:
                in_batch_duplicates += 1
                continue
            by_id[event.event_id] = event
            by_natural[event.natural_key] = event
            unique.append(event)

        inserted = 0
        existing_duplicates = 0
        with self.database.transaction() as connection:
            for event in unique:
                id_row = connection.execute(
                    "SELECT rowid,* FROM evidence_events WHERE event_id=?", (event.event_id,)
                ).fetchone()
                natural_row = connection.execute(
                    "SELECT rowid,* FROM evidence_events WHERE trace_id=? AND attempt_id=? "
                    "AND artifact_ref=? AND event_type=? AND sequence_no=?",
                    event.natural_key,
                ).fetchone()
                rows = [row for row in (id_row, natural_row) if row is not None]
                if rows:
                    if any(not _same_event(EvidenceEvent.from_row(row), event) for row in rows):
                        raise EvidenceConflictError(
                            f"ledger identity already contains different evidence: {event.event_id}"
                        )
                    if len({int(row["rowid"]) for row in rows}) != 1:
                        raise EvidenceConflictError(
                            f"event id and natural id resolve to different ledger rows: {event.event_id}"
                        )
                    existing_duplicates += 1
                    continue
                connection.execute(_INSERT, event.database_values())
                inserted += 1
            row = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) AS last_rowid FROM evidence_events"
            ).fetchone()
            last_rowid = int(row["last_rowid"])

        return AppendResult(
            requested_count=len(requested),
            inserted_count=inserted,
            duplicate_count=in_batch_duplicates + existing_duplicates,
            last_event_rowid=last_rowid,
        )

    append = append_transaction

    def records_after(self, rowid: int = 0, *, limit: int | None = None) -> list[LedgerRecord]:
        if rowid < 0:
            raise ValueError("rowid must be non-negative")
        sql = "SELECT rowid,* FROM evidence_events WHERE rowid>? ORDER BY rowid"
        parameters: tuple[Any, ...] = (rowid,)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            sql += " LIMIT ?"
            parameters += (limit,)
        return [
            LedgerRecord(rowid=int(row["rowid"]), event=EvidenceEvent.from_row(row))
            for row in self.database.rows(sql, parameters)
        ]

    def get(self, event_id: str) -> EvidenceEvent | None:
        row = self.database.execute(
            "SELECT rowid,* FROM evidence_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return None if row is None else EvidenceEvent.from_row(row)

    def count(self) -> int:
        row = self.database.execute("SELECT COUNT(*) AS count FROM evidence_events").fetchone()
        return int(row["count"])

    def max_rowid(self) -> int:
        row = self.database.execute(
            "SELECT COALESCE(MAX(rowid), 0) AS last_rowid FROM evidence_events"
        ).fetchone()
        return int(row["last_rowid"])


def _failure_layer_value(value: FailureLayer | str) -> str:
    if isinstance(value, FailureLayer):
        return value.value
    text = str(value or "")
    if text:
        return FailureLayer(text).value
    return ""


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    return json.dumps(
        to_primitive(dict(metadata)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _same_event(left: EvidenceEvent, right: EvidenceEvent) -> bool:
    return left.database_values() == right.database_values()
