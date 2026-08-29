"""SQLite fact ledger, indexes, projections, and run manifests."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 3

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "metadata": ("key", "value"),
    "artifact_index": (
        "artifact_ref", "artifact_kind", "logical_id", "version",
        "content_hash", "status", "file_path", "schema_version",
    ),
    "recommended_pointers": ("logical_id", "artifact_ref"),
    "evidence_events": (
        "event_id", "schema_version", "task_id", "trace_id",
        "occurrence_id", "attempt_id", "sequence_no", "artifact_ref",
        "artifact_kind", "event_type", "failure_layer", "confidence",
        "metadata_json",
    ),
    "lifecycle_projection": (
        "artifact_ref", "projection_json", "last_event_rowid",
    ),
    "projection_checkpoints": ("projection_name", "last_event_rowid"),
    "graph_edges": (
        "edge_id", "source_ref", "target_ref", "relation", "metadata_json",
    ),
    "run_manifests": (
        "run_id", "phase", "config_hash", "task_manifest_hash",
        "code_commit", "state",
    ),
    "run_tasks": (
        "run_id", "task_id", "task_signature", "config_hash",
        "code_commit", "knowledge_milestone", "state", "attempt_count",
        "trace_id", "result_json",
    ),
}

_REQUIRED_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "metadata": ("key",),
    "artifact_index": ("artifact_ref",),
    "recommended_pointers": ("logical_id",),
    "evidence_events": ("event_id",),
    "lifecycle_projection": ("artifact_ref",),
    "projection_checkpoints": ("projection_name",),
    "graph_edges": ("edge_id",),
    "run_manifests": ("run_id",),
    "run_tasks": ("run_id", "task_id"),
}

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_index (
    artifact_ref TEXT PRIMARY KEY,
    artifact_kind TEXT NOT NULL,
    logical_id TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    file_path TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS artifact_kind_status
    ON artifact_index(artifact_kind, status);

CREATE TABLE IF NOT EXISTS recommended_pointers (
    logical_id TEXT PRIMARY KEY,
    artifact_ref TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_events (
    event_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    occurrence_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    artifact_ref TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    event_type TEXT NOT NULL,
    failure_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE(trace_id, attempt_id, artifact_ref, event_type, sequence_no)
);

CREATE TABLE IF NOT EXISTS lifecycle_projection (
    artifact_ref TEXT PRIMARY KEY,
    projection_json TEXT NOT NULL,
    last_event_rowid INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS projection_checkpoints (
    projection_name TEXT PRIMARY KEY,
    last_event_rowid INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_ref TEXT NOT NULL,
    target_ref TEXT NOT NULL,
    relation TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_manifests (
    run_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    task_manifest_hash TEXT NOT NULL,
    code_commit TEXT NOT NULL,
    state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_tasks (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_signature TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_commit TEXT NOT NULL,
    knowledge_milestone TEXT NOT NULL,
    state TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    trace_id TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(run_id, task_id)
);
"""


class StateDatabase:
    def __init__(self, path: str | Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        self.readonly = readonly
        existed = self.path.is_file()
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = self._open()
        if not readonly and not existed:
            self.initialize()
        self.validate_integrity()

    def _open(self) -> sqlite3.Connection:
        if self.readonly:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if not self.readonly:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def initialize(self) -> None:
        self._connection.executescript(DDL)
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._connection.commit()

    def _validate_version(self) -> None:
        try:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise RuntimeError("state database is not an AtomicSkillGraph v3 database") from exc
        if row is None or int(row["value"]) != SCHEMA_VERSION:
            actual = None if row is None else row["value"]
            raise RuntimeError(f"unsupported state schema {actual!r}; v3 requires schema_version=3")

    def _validate_schema(self) -> None:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        available = {str(row["name"]) for row in rows}
        missing_tables = sorted(set(_REQUIRED_COLUMNS) - available)
        if missing_tables:
            raise RuntimeError(
                "state database is missing required v3 tables: "
                + ", ".join(missing_tables)
            )
        for table, required in _REQUIRED_COLUMNS.items():
            table_info = self._connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
            columns = {str(row["name"]) for row in table_info}
            missing_columns = sorted(set(required) - columns)
            if missing_columns:
                raise RuntimeError(
                    f"state database table {table} is missing required columns: "
                    + ", ".join(missing_columns)
                )
            primary_key = tuple(
                str(row["name"])
                for row in sorted(table_info, key=lambda item: int(item["pk"]))
                if int(row["pk"]) > 0
            )
            if primary_key != _REQUIRED_PRIMARY_KEYS[table]:
                raise RuntimeError(
                    f"state database table {table} has invalid primary key: "
                    f"expected {_REQUIRED_PRIMARY_KEYS[table]}, got {primary_key}"
                )

        index_rows = self._connection.execute(
            "PRAGMA index_list('evidence_events')"
        ).fetchall()
        unique_event_keys = {
            tuple(
                str(item["name"])
                for item in self._connection.execute(
                    f'PRAGMA index_info("{row["name"]}")'
                ).fetchall()
            )
            for row in index_rows
            if bool(row["unique"])
        }
        required_unique = (
            "trace_id", "attempt_id", "artifact_ref", "event_type", "sequence_no",
        )
        if required_unique not in unique_event_keys:
            raise RuntimeError("state database lacks the v3 EvidenceEvent uniqueness constraint")

    def validate_integrity(self) -> None:
        """Fail closed on corruption, version drift, or incomplete v3 schema."""
        try:
            rows = self._connection.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("state database integrity check failed") from exc
        messages = [str(row[0]) for row in rows]
        if messages != ["ok"]:
            raise RuntimeError(
                "state database integrity check failed: " + "; ".join(messages)
            )
        self._validate_version()
        self._validate_schema()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            raise RuntimeError("frozen database is read-only")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self._connection.execute(sql, parameters)

    def rows(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self._connection.execute(sql, parameters).fetchall())

    def set_metadata(self, key: str, value: Any) -> None:
        if self.readonly:
            raise RuntimeError("frozen database is read-only")
        encoded = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        self._connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, encoded),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "StateDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
