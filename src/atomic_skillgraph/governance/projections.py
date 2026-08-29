"""Checkpointed lifecycle projections derived only from EvidenceLedger facts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..knowledge.database import SCHEMA_VERSION, StateDatabase
from .ledger import EvidenceEvent, EvidenceEventType, EvidenceLedger


_SUCCESS_EVENTS = frozenset(
    {
        EvidenceEventType.DIRECT_SUCCESS,
        EvidenceEventType.AGENT_NODE_SUCCESS,
        EvidenceEventType.SEEDED_SUCCESS,
        EvidenceEventType.SELF_SUFFICIENT_SUCCESS,
    }
)
_FAILURE_EVENTS = frozenset(
    {
        EvidenceEventType.DIRECT_FAILURE,
        EvidenceEventType.SEEDED_FAILURE,
        EvidenceEventType.TASK_RESCUE_REQUIRED,
    }
)


class ProjectionError(RuntimeError):
    pass


class ProjectionCorruptionError(ProjectionError):
    pass


@dataclass
class ArtifactStats:
    artifact_ref: str
    artifact_kind: str
    schema_version: int = SCHEMA_VERSION
    event_counts: dict[str, int] = field(default_factory=dict)
    event_task_ids: dict[str, list[str]] = field(default_factory=dict)
    task_ids: list[str] = field(default_factory=list)
    success_task_ids: list[str] = field(default_factory=list)

    selected_count: int = 0
    started_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    intrinsic_failure_count: int = 0
    consecutive_intrinsic_failures: int = 0
    self_sufficient_success_count: int = 0
    task_rescue_count: int = 0
    goal_terminal_skip_count: int = 0
    preflight_rejected_count: int = 0
    contract_mismatch_count: int = 0
    proposed_count: int = 0
    validated_count: int = 0
    superseded_count: int = 0

    cost_sum: float = 0.0
    latency_sum: float = 0.0
    stable_replacement: bool = False
    preferred_utility_evidence_count: int = 0
    occurrence_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    first_event_rowid: int = 0
    last_event_rowid: int = 0
    last_event_id: str = ""

    def __post_init__(self) -> None:
        self.artifact_kind = str(self.artifact_kind).casefold()
        self.event_counts = {str(key): int(value) for key, value in self.event_counts.items()}
        self.event_task_ids = {
            str(key): sorted({str(item) for item in value})
            for key, value in self.event_task_ids.items()
        }
        self.task_ids = sorted({str(item) for item in self.task_ids})
        self.success_task_ids = sorted({str(item) for item in self.success_task_ids})
        self.occurrence_stats = {
            str(key): {str(name): int(value) for name, value in counts.items()}
            for key, counts in self.occurrence_stats.items()
        }
        if self.schema_version != SCHEMA_VERSION:
            raise ProjectionCorruptionError(
                f"projection schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )

    @property
    def independent_task_count(self) -> int:
        """Number of distinct tasks contributing positive reuse evidence."""

        return len(self.success_task_ids)

    @property
    def independent_direct_success_count(self) -> int:
        return len(self.event_task_ids.get(EvidenceEventType.DIRECT_SUCCESS.value, ()))

    @property
    def independent_self_sufficient_success_count(self) -> int:
        return len(
            self.event_task_ids.get(EvidenceEventType.SELF_SUFFICIENT_SUCCESS.value, ())
        )

    @property
    def direct_success_count(self) -> int:
        return self.event_count(EvidenceEventType.DIRECT_SUCCESS)

    @property
    def direct_failure_count(self) -> int:
        return self.event_count(EvidenceEventType.DIRECT_FAILURE)

    @property
    def reliability(self) -> float:
        if self.started_count <= 0:
            return 0.0
        return min(self.direct_success_count, self.started_count) / self.started_count

    def reliability_lower_bound(self, *, z: float = 1.96) -> float:
        """Wilson lower bound over real started executions."""

        n = self.started_count
        if n <= 0:
            return 0.0
        successes = min(self.direct_success_count, n)
        proportion = successes / n
        z2 = z * z
        denominator = 1.0 + z2 / n
        centre = proportion + z2 / (2.0 * n)
        margin = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * n)) / n)
        return max(0.0, (centre - margin) / denominator)

    def event_count(self, event: EvidenceEventType | str) -> int:
        value = event.value if isinstance(event, EvidenceEventType) else str(event)
        return self.event_counts.get(value, 0)

    def apply(self, event: EvidenceEvent, rowid: int) -> None:
        if event.artifact_ref != self.artifact_ref or event.artifact_kind != self.artifact_kind:
            raise ProjectionCorruptionError(
                f"projection/event identity mismatch for {event.artifact_ref}"
            )
        if rowid <= self.last_event_rowid:
            raise ProjectionCorruptionError(
                f"non-monotonic event rowid {rowid} for {self.artifact_ref}"
            )
        if self.first_event_rowid == 0:
            self.first_event_rowid = rowid
        self.last_event_rowid = rowid
        self.last_event_id = event.event_id

        name = event.event.value
        self.event_counts[name] = self.event_counts.get(name, 0) + 1
        _add_unique(self.task_ids, event.task_id)
        event_tasks = self.event_task_ids.setdefault(name, [])
        _add_unique(event_tasks, event.task_id)

        if event.event is EvidenceEventType.PROPOSED:
            self.proposed_count += 1
        elif event.event is EvidenceEventType.VALIDATED:
            self.validated_count += 1
        elif event.event is EvidenceEventType.SELECTED:
            self.selected_count += 1
        elif event.event is EvidenceEventType.EXECUTION_STARTED:
            self.started_count += 1
        elif event.event is EvidenceEventType.PREFLIGHT_REJECTED:
            self.preflight_rejected_count += 1
        elif event.event is EvidenceEventType.SELF_SUFFICIENT_SUCCESS:
            self.self_sufficient_success_count += 1
        elif event.event is EvidenceEventType.TASK_RESCUE_REQUIRED:
            self.task_rescue_count += 1
        elif event.event is EvidenceEventType.GOAL_TERMINAL_SKIPPED:
            self.goal_terminal_skip_count += 1
        elif event.event is EvidenceEventType.CONTRACT_MISMATCH:
            self.contract_mismatch_count += 1
        elif event.event is EvidenceEventType.SUPERSEDED:
            self.superseded_count += 1

        if event.event in _SUCCESS_EVENTS:
            self.success_count += 1
            _add_unique(self.success_task_ids, event.task_id)
            self.consecutive_intrinsic_failures = 0

        artifact_failure = _is_artifact_failure(event)
        intrinsic = _is_intrinsic_failure(event)
        if artifact_failure:
            self.failure_count += 1
        if intrinsic:
            self.intrinsic_failure_count += 1
            self.consecutive_intrinsic_failures += 1

        cost = event.metadata.get("cost_usd", event.metadata.get("cost"))
        if cost is not None:
            self.cost_sum += _nonnegative_number(cost, "cost")
        latency = event.metadata.get("latency_ms")
        if latency is not None:
            self.latency_sum += _nonnegative_number(latency, "latency_ms")

        replacement_status = str(event.metadata.get("replacement_status", ""))
        if event.event is EvidenceEventType.SUPERSEDED and (
            bool(event.metadata.get("reliable_replacement"))
            or replacement_status in {"active", "preferred"}
        ):
            self.stable_replacement = True
        if bool(event.metadata.get("preferred_utility")):
            self.preferred_utility_evidence_count += 1

        if self.artifact_kind == "composite":
            self._apply_occurrence(event, artifact_failure)

    def _apply_occurrence(self, event: EvidenceEvent, artifact_failure: bool) -> None:
        if not event.occurrence_id or event.occurrence_id in {"graph", "maintenance", "evolution"}:
            return
        counts = self.occurrence_stats.setdefault(event.occurrence_id, {})

        def increment(name: str) -> None:
            counts[name] = counts.get(name, 0) + 1

        if event.event is EvidenceEventType.SELECTED:
            increment("selected")
            if bool(event.metadata.get("executed")):
                increment("executed")
        if event.event is EvidenceEventType.EXECUTION_STARTED:
            increment("executed")
        if event.event is EvidenceEventType.GOAL_TERMINAL_SKIPPED:
            increment("skipped_goal_terminal")
        if event.event is EvidenceEventType.AGENT_NODE_SUCCESS:
            increment("agent_completed_before_invocation")
        if event.event is EvidenceEventType.SEEDED_SUCCESS:
            increment("seeded_success")
        if artifact_failure:
            increment("failure")

        status = str(event.metadata.get("node_status", ""))
        status_map = {
            "already_satisfied": "already_satisfied",
            "direct_autonomous_success": "direct_autonomous",
            "direct_agent_prepared_success": "direct_agent_prepared",
            "agent_completed_before_invocation": "agent_completed_before_invocation",
            "seeded_success": "seeded_success",
        }
        normalized = status_map.get(status)
        if normalized:
            increment(normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_ref": self.artifact_ref,
            "artifact_kind": self.artifact_kind,
            "schema_version": self.schema_version,
            "event_counts": dict(sorted(self.event_counts.items())),
            "event_task_ids": {
                key: sorted(value) for key, value in sorted(self.event_task_ids.items())
            },
            "task_ids": sorted(self.task_ids),
            "success_task_ids": sorted(self.success_task_ids),
            "independent_task_count": self.independent_task_count,
            "selected_count": self.selected_count,
            "started_count": self.started_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "intrinsic_failure_count": self.intrinsic_failure_count,
            "consecutive_intrinsic_failures": self.consecutive_intrinsic_failures,
            "self_sufficient_success_count": self.self_sufficient_success_count,
            "task_rescue_count": self.task_rescue_count,
            "goal_terminal_skip_count": self.goal_terminal_skip_count,
            "preflight_rejected_count": self.preflight_rejected_count,
            "contract_mismatch_count": self.contract_mismatch_count,
            "proposed_count": self.proposed_count,
            "validated_count": self.validated_count,
            "superseded_count": self.superseded_count,
            "cost_sum": self.cost_sum,
            "latency_sum": self.latency_sum,
            "stable_replacement": self.stable_replacement,
            "preferred_utility_evidence_count": self.preferred_utility_evidence_count,
            "occurrence_stats": {
                key: dict(sorted(value.items()))
                for key, value in sorted(self.occurrence_stats.items())
            },
            "first_event_rowid": self.first_event_rowid,
            "last_event_rowid": self.last_event_rowid,
            "last_event_id": self.last_event_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactStats":
        allowed = {
            field_name
            for field_name in cls.__dataclass_fields__
        }
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**values)


@dataclass(frozen=True)
class ProjectionResult:
    processed_count: int
    affected_artifacts: tuple[str, ...]
    checkpoint_before: int
    checkpoint_after: int


class LifecycleProjection:
    """Incremental and fully rebuildable projection over ledger row order."""

    DEFAULT_NAME = "lifecycle_v3"

    def __init__(
        self,
        database: StateDatabase,
        ledger: EvidenceLedger | None = None,
        *,
        projection_name: str = DEFAULT_NAME,
    ) -> None:
        if not projection_name.strip():
            raise ValueError("projection_name must be non-empty")
        self.database = database
        self.ledger = ledger or EvidenceLedger(database)
        self.projection_name = projection_name

    @property
    def checkpoint(self) -> int:
        row = self.database.execute(
            "SELECT last_event_rowid FROM projection_checkpoints WHERE projection_name=?",
            (self.projection_name,),
        ).fetchone()
        return 0 if row is None else int(row["last_event_rowid"])

    def consume(self, events: Iterable[EvidenceEvent] | None = None) -> ProjectionResult:
        if events is not None:
            for event in events:
                stored = self.ledger.get(event.event_id)
                if stored is None:
                    raise ProjectionError(
                        f"projection cannot consume uncommitted evidence: {event.event_id}"
                    )
        return self.consume_new_events()

    def consume_new_events(self, *, limit: int | None = None) -> ProjectionResult:
        if self.database.readonly:
            raise RuntimeError("frozen lifecycle projection is read-only")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        with self.database.transaction() as connection:
            checkpoint = self._checkpoint_for_update(connection)
            sql = "SELECT rowid,* FROM evidence_events WHERE rowid>? ORDER BY rowid"
            parameters: tuple[Any, ...] = (checkpoint,)
            if limit is not None:
                sql += " LIMIT ?"
                parameters += (limit,)
            rows = list(connection.execute(sql, parameters).fetchall())
            if not rows:
                return ProjectionResult(0, (), checkpoint, checkpoint)
            affected = self._project_rows(connection, rows)
            after = int(rows[-1]["rowid"])
            connection.execute(
                "INSERT INTO projection_checkpoints(projection_name,last_event_rowid) VALUES(?,?) "
                "ON CONFLICT(projection_name) DO UPDATE SET last_event_rowid=excluded.last_event_rowid",
                (self.projection_name, after),
            )
        return ProjectionResult(len(rows), tuple(sorted(affected)), checkpoint, after)

    def rebuild(self) -> ProjectionResult:
        """Atomically derive every statistic again from the immutable ledger."""

        if self.database.readonly:
            raise RuntimeError("frozen lifecycle projection is read-only")
        with self.database.transaction() as connection:
            checkpoint_before = self._checkpoint_for_update(connection, allow_orphan=True)
            connection.execute("DELETE FROM lifecycle_projection")
            connection.execute(
                "DELETE FROM projection_checkpoints WHERE projection_name=?",
                (self.projection_name,),
            )
            rows = list(connection.execute("SELECT rowid,* FROM evidence_events ORDER BY rowid"))
            affected = self._project_rows(connection, rows) if rows else set()
            checkpoint_after = int(rows[-1]["rowid"]) if rows else 0
            connection.execute(
                "INSERT INTO projection_checkpoints(projection_name,last_event_rowid) VALUES(?,?)",
                (self.projection_name, checkpoint_after),
            )
        return ProjectionResult(
            len(rows), tuple(sorted(affected)), checkpoint_before, checkpoint_after
        )

    def stats(self, artifact_ref: str, artifact_kind: str | None = None) -> ArtifactStats:
        row = self.database.execute(
            "SELECT projection_json FROM lifecycle_projection WHERE artifact_ref=?",
            (artifact_ref,),
        ).fetchone()
        if row is not None:
            return ArtifactStats.from_dict(json.loads(str(row["projection_json"])))
        if artifact_kind is None:
            artifact = self.database.execute(
                "SELECT artifact_kind FROM artifact_index WHERE artifact_ref=?", (artifact_ref,)
            ).fetchone()
            if artifact is None:
                raise KeyError(artifact_ref)
            artifact_kind = str(artifact["artifact_kind"])
        return ArtifactStats(artifact_ref=artifact_ref, artifact_kind=artifact_kind)

    def all_stats(self) -> list[ArtifactStats]:
        return [
            ArtifactStats.from_dict(json.loads(str(row["projection_json"])))
            for row in self.database.rows(
                "SELECT projection_json FROM lifecycle_projection ORDER BY artifact_ref"
            )
        ]

    def digest(self) -> str:
        payload = [stats.to_dict() for stats in self.all_stats()]
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _checkpoint_for_update(self, connection: Any, *, allow_orphan: bool = False) -> int:
        row = connection.execute(
            "SELECT last_event_rowid FROM projection_checkpoints WHERE projection_name=?",
            (self.projection_name,),
        ).fetchone()
        if row is None:
            projection_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM lifecycle_projection"
                ).fetchone()["count"]
            )
            if projection_count and not allow_orphan:
                raise ProjectionCorruptionError(
                    "lifecycle rows exist without a checkpoint; call rebuild()"
                )
            return 0
        checkpoint = int(row["last_event_rowid"])
        max_rowid = int(
            connection.execute(
                "SELECT COALESCE(MAX(rowid),0) AS rowid FROM evidence_events"
            ).fetchone()["rowid"]
        )
        if checkpoint > max_rowid and not allow_orphan:
            raise ProjectionCorruptionError(
                f"projection checkpoint {checkpoint} is ahead of ledger {max_rowid}"
            )
        return checkpoint

    def _project_rows(self, connection: Any, rows: list[Any]) -> set[str]:
        stats_by_ref: dict[str, ArtifactStats] = {}
        affected: set[str] = set()
        for row in rows:
            event = EvidenceEvent.from_row(row)
            stats = stats_by_ref.get(event.artifact_ref)
            if stats is None:
                existing = connection.execute(
                    "SELECT projection_json FROM lifecycle_projection WHERE artifact_ref=?",
                    (event.artifact_ref,),
                ).fetchone()
                stats = (
                    ArtifactStats.from_dict(json.loads(str(existing["projection_json"])))
                    if existing is not None
                    else ArtifactStats(event.artifact_ref, event.artifact_kind)
                )
                stats_by_ref[event.artifact_ref] = stats
            stats.apply(event, int(row["rowid"]))
            affected.add(event.artifact_ref)

        for artifact_ref, stats in stats_by_ref.items():
            encoded = json.dumps(
                stats.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            connection.execute(
                "INSERT INTO lifecycle_projection(artifact_ref,projection_json,last_event_rowid) "
                "VALUES(?,?,?) ON CONFLICT(artifact_ref) DO UPDATE SET "
                "projection_json=excluded.projection_json,last_event_rowid=excluded.last_event_rowid",
                (artifact_ref, encoded, stats.last_event_rowid),
            )
        return affected


def _add_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
        values.sort()


def _is_artifact_failure(event: EvidenceEvent) -> bool:
    if event.event in _FAILURE_EVENTS:
        return True
    if event.event is EvidenceEventType.PREFLIGHT_REJECTED:
        return bool(event.metadata.get("intrinsic_failure"))
    return bool(event.metadata.get("intrinsic_failure")) and event.event is EvidenceEventType.CONTRACT_MISMATCH


def _is_intrinsic_failure(event: EvidenceEvent) -> bool:
    if not bool(event.metadata.get("intrinsic_failure")):
        return False
    if event.artifact_kind == "tool":
        return (
            event.event is EvidenceEventType.DIRECT_FAILURE
            and bool(event.metadata.get("started"))
            and event.failure_layer == "tool"
        )
    if event.artifact_kind == "implementation":
        return event.failure_layer == "implementation" and event.event in {
            EvidenceEventType.PREFLIGHT_REJECTED,
            EvidenceEventType.DIRECT_FAILURE,
        }
    if event.artifact_kind == "atomic":
        return event.failure_layer == "atomic" and event.event in {
            EvidenceEventType.SEEDED_FAILURE,
            EvidenceEventType.DIRECT_FAILURE,
        }
    if event.artifact_kind == "composite":
        return event.failure_layer == "composite" and event.event in {
            EvidenceEventType.TASK_RESCUE_REQUIRED,
            EvidenceEventType.CONTRACT_MISMATCH,
        }
    return False


def _nonnegative_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectionCorruptionError(f"event {name} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ProjectionCorruptionError(f"event {name} must be finite and non-negative")
    return number
