"""Immutable experiment manifests, stable digests, and fail-closed resume.

Manifest identity is immutable JSON written before the first task starts.
Mutable run/task state lives only in the v3 SQLite ``run_manifests`` and
``run_tasks`` tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atomic_skillgraph.core.serialization import atomic_create_json


SCHEMA_VERSION = 3
ALFWORLD_FORMAL_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
DEEPSEEK_FORMAL_DIALECT = "deepseek_v4_chat"
DEEPSEEK_FORMAL_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_FORMAL_MODEL = "deepseek-v4-flash"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CODE_SUFFIXES = frozenset(
    {".py", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".txt", ".lock"}
)
_CODE_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "data",
        "data_v3",
        "dist",
        "env",
        "outputs",
        "reports",
        "results",
        "runs",
        "runs_v3",
        "frozen",
        "snapshots",
        "traces",
        "venv",
    }
)


class ProtocolError(RuntimeError):
    pass


class ManifestExistsError(ProtocolError):
    pass


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)['\"]?[^\s,'\";]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def sanitize_error_text(value: Any) -> str:
    """Keep failure diagnostics useful without persisting provider secrets."""

    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            (lambda match: match.group(1) + "[REDACTED]")
            if pattern.groups
            else "[REDACTED]",
            text,
        )
    return text[:4000]


def write_failure_receipt(
    root: str | Path,
    *,
    attempt: "AttemptTraceRef",
    primary: BaseException,
    audit_errors: Sequence[Mapping[str, Any]],
) -> Path:
    """Create one immutable, sanitized receipt for a failed formal attempt."""

    target = Path(root).resolve() / f"{attempt.attempt_id}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "task_id": attempt.task_id,
        "task_signature": attempt.task_signature,
        "attempt_kind": attempt.attempt_kind,
        "sequence": attempt.sequence,
        "primary_error_type": type(primary).__name__,
        "primary_error_code": sanitize_error_text(getattr(primary, "code", "")),
        "primary_error_message": sanitize_error_text(primary),
        "audit_errors": [
            {
                "stage": sanitize_error_text(item.get("stage", "")),
                "error_type": sanitize_error_text(item.get("error_type", "")),
                "error": sanitize_error_text(item.get("error", "")),
            }
            for item in audit_errors
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    return atomic_create_json(target, payload)


def audit_failed_attempt(
    *,
    primary: BaseException,
    attempt: "AttemptTraceRef",
    attempt_ledger: "AttemptTraceLedger",
    receipt_root: str | Path,
    update_state: Any,
    capture_reason: str,
) -> list[dict[str, str]]:
    """Best-effort audit cleanup that can never replace ``primary``."""

    audit_errors: list[dict[str, str]] = []
    try:
        attempt_ledger.capture(attempt, reason=capture_reason)
    except Exception as exc:  # deliberately secondary to the caller's primary
        audit_errors.append({
            "stage": "capture", "error_type": type(exc).__name__,
            "error": sanitize_error_text(exc),
        })
    try:
        update_state()
    except Exception as exc:  # deliberately secondary to the caller's primary
        audit_errors.append({
            "stage": "state", "error_type": type(exc).__name__,
            "error": sanitize_error_text(exc),
        })
    try:
        write_failure_receipt(
            receipt_root,
            attempt=attempt,
            primary=primary,
            audit_errors=audit_errors,
        )
    except Exception as exc:  # receipt failure is reported, never raised
        audit_errors.append({
            "stage": "receipt", "error_type": type(exc).__name__,
            "error": sanitize_error_text(exc),
        })
    return audit_errors


def validate_deepseek_formal_llm(config: Mapping[str, Any]) -> None:
    """Fail closed unless a formal run uses the probed DeepSeek V4 dialect."""

    llm = dict(config.get("llm") or {})
    protocol = dict(llm.get("protocol") or {})
    expected = {
        "llm.provider": (llm.get("provider"), "openai_compatible"),
        "llm.dialect": (llm.get("dialect"), DEEPSEEK_FORMAL_DIALECT),
        "llm.base_url": (str(llm.get("base_url", "")).rstrip("/"), DEEPSEEK_FORMAL_BASE_URL),
        "llm.model": (llm.get("model"), DEEPSEEK_FORMAL_MODEL),
        "llm.protocol.endpoint_path": (protocol.get("endpoint_path"), "/chat/completions"),
        "llm.protocol.structured_output_transport": (
            protocol.get("structured_output_transport"), "native_submission_tool"
        ),
        "llm.protocol.token_limit_field": (protocol.get("token_limit_field"), "max_tokens"),
        "llm.protocol.thinking_type": (protocol.get("thinking_type"), "enabled"),
        "llm.protocol.send_reasoning_effort": (protocol.get("send_reasoning_effort"), True),
        "llm.protocol.replay_reasoning_content_with_tools": (
            protocol.get("replay_reasoning_content_with_tools"), True
        ),
        "llm.protocol.send_response_format": (protocol.get("send_response_format"), False),
        "llm.protocol.send_tool_choice": (protocol.get("send_tool_choice"), False),
        "llm.protocol.send_parallel_tool_calls": (
            protocol.get("send_parallel_tool_calls"), False
        ),
        "llm.protocol.send_temperature": (protocol.get("send_temperature"), False),
        "llm.protocol.strict_tools": (protocol.get("strict_tools"), False),
        "llm.protocol.require_usage": (protocol.get("require_usage"), True),
    }
    stage_expected = {
        "planner": {
            "reasoning_effort": "high", "max_completion_tokens": 32768,
            "request_timeout_seconds": 300, "max_turns": 4,
            "max_total_tokens_per_task": 120000,
        },
        "runtime": {
            "reasoning_effort": "low", "max_completion_tokens": 32768,
            "request_timeout_seconds": 180, "max_total_tokens_per_node": 100000,
            "max_total_tokens_per_task": 300000, "learned_toolcall_repair_limit": 2,
            "protocol_repair_limit": 1,
        },
        "extractor": {
            "reasoning_effort": "high", "max_completion_tokens": 131072,
            "request_timeout_seconds": 600, "max_turns": 2,
            "max_total_tokens_per_task": 262144,
        },
        "evolution_repair": {
            "reasoning_effort": "high", "max_completion_tokens": 32768,
            "request_timeout_seconds": 300, "max_turns": 1,
            "max_total_tokens_per_batch": 120000,
        },
    }
    for stage, fields_expected in stage_expected.items():
        configured = dict(llm.get(stage) or {})
        for name, wanted in fields_expected.items():
            expected[f"llm.{stage}.{name}"] = (configured.get(name), wanted)
    mismatches = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    forbidden_config_fields = sorted(
        f"llm.{stage}.{name}"
        for stage in ("planner", "runtime", "extractor", "evolution_repair")
        for name in ("max_visible_tokens", "max_turns_per_node", "max_turns_per_task")
        if name in dict(llm.get(stage) or {})
    )
    if forbidden_config_fields:
        mismatches.append(
            "removed hidden/visible token gates are configured: "
            + ", ".join(forbidden_config_fields)
        )
    runtime = dict(config.get("runtime") or {})
    for name, wanted in (
        ("global_action_budget", 100),
        ("node_action_budget", 35),
    ):
        actual = runtime.get(name)
        if actual != wanted:
            mismatches.append(
                f"runtime.{name}: expected {wanted!r}, got {actual!r}"
            )
    if mismatches:
        raise ProtocolError("formal DeepSeek protocol mismatch: " + "; ".join(mismatches))


@dataclass(frozen=True)
class FieldMismatch:
    field: str
    persisted: Any
    current: Any


class ManifestMismatchError(ProtocolError):
    def __init__(self, mismatches: Iterable[FieldMismatch]) -> None:
        self.mismatches = tuple(mismatches)
        details = "; ".join(
            f"{item.field}: persisted={item.persisted!r}, current={item.current!r}"
            for item in self.mismatches
        )
        super().__init__(f"resume manifest mismatch ({details})")


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    TASK_FAILED = "task_failed"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    TASK_FAILED = "task_failed"


@dataclass(frozen=True)
class AttemptTraceRef:
    """Identity of one append-only formal task/maintenance attempt."""

    attempt_id: str
    run_id: str
    task_id: str
    task_signature: str
    attempt_kind: str
    sequence: int
    expected_periodic_milestone: str = ""


@dataclass(frozen=True)
class TaskManifest:
    ordinal: int
    task_id: str
    task_signature: str
    knowledge_milestone: str
    benchmark: str = ""
    split: str = ""
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("TaskManifest ordinal must be a non-negative integer")
        for name in ("task_id", "task_signature", "knowledge_milestone"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"TaskManifest requires non-empty {name}")
        object.__setattr__(self, "metadata_json", _canonical_metadata(self.metadata_json))

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "task_id": self.task_id,
            "task_signature": self.task_signature,
            "knowledge_milestone": self.knowledge_milestone,
            "benchmark": self.benchmark,
            "split": self.split,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskManifest":
        return cls(
            ordinal=int(payload["ordinal"]),
            task_id=str(payload["task_id"]),
            task_signature=str(payload["task_signature"]),
            knowledge_milestone=str(payload["knowledge_milestone"]),
            benchmark=str(payload.get("benchmark", "")),
            split=str(payload.get("split", "")),
            metadata_json=_canonical_json(payload.get("metadata", {})),
        )

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, Any] | Any,
        *,
        ordinal: int,
        knowledge_milestone: str,
        split: str = "",
    ) -> "TaskManifest":
        return cls(
            ordinal=ordinal,
            task_id=str(_field(task, "task_id", "")),
            task_signature=task_signature(task),
            knowledge_milestone=knowledge_milestone,
            benchmark=str(_field(task, "benchmark", "")),
            split=split or str(_field(task, "split", "")),
            metadata_json=_canonical_json(_field(task, "metadata", {}) or {}),
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    phase: str
    created_at: str
    config_hash: str
    code_commit: str
    knowledge_digest: str
    task_manifest_hash: str
    tasks: tuple[TaskManifest, ...]
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "metadata_json", _canonical_metadata(self.metadata_json))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"RunManifest requires schema_version={SCHEMA_VERSION}")
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError(f"unsafe or invalid run_id: {self.run_id!r}")
        for name in (
            "phase",
            "created_at",
            "config_hash",
            "code_commit",
            "knowledge_digest",
            "task_manifest_hash",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"RunManifest requires non-empty {name}")
        ordinals = [task.ordinal for task in self.tasks]
        if ordinals != list(range(len(self.tasks))):
            raise ValueError("TaskManifest ordinals must be contiguous and in manifest order")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("RunManifest task_id values must be unique")
        task_signatures = [task.task_signature for task in self.tasks]
        if len(task_signatures) != len(set(task_signatures)):
            raise ValueError("RunManifest task_signature values must be unique")
        actual = hash_task_manifest(self.tasks)
        if actual != self.task_manifest_hash:
            raise ValueError(
                f"task_manifest_hash mismatch: declared={self.task_manifest_hash}, actual={actual}"
            )

    @property
    def code_hash(self) -> str:
        return self.code_commit

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    @property
    def manifest_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "phase": self.phase,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "knowledge_digest": self.knowledge_digest,
            "task_manifest_hash": self.task_manifest_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        phase: str,
        config_hash: str,
        code_commit: str,
        knowledge_digest: str,
        tasks: Sequence[TaskManifest],
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "RunManifest":
        frozen_tasks = tuple(tasks)
        return cls(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            phase=phase,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            code_commit=code_commit,
            knowledge_digest=knowledge_digest,
            task_manifest_hash=hash_task_manifest(frozen_tasks),
            tasks=frozen_tasks,
            metadata_json=_canonical_json(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        return cls(
            schema_version=int(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            phase=str(payload["phase"]),
            created_at=str(payload["created_at"]),
            config_hash=str(payload["config_hash"]),
            code_commit=str(payload["code_commit"]),
            knowledge_digest=str(payload["knowledge_digest"]),
            task_manifest_hash=str(payload["task_manifest_hash"]),
            tasks=tuple(TaskManifest.from_dict(item) for item in payload.get("tasks", [])),
            metadata_json=_canonical_json(payload.get("metadata", {})),
        )


def task_signature(task: Mapping[str, Any] | Any) -> str:
    existing = str(_field(task, "task_signature", "") or "")
    if not existing:
        metadata = _field(task, "metadata", {}) or {}
        existing = str(_field(metadata, "task_signature", "") or "")
    if existing:
        return existing
    return sha256_json(_to_primitive(task))


def validate_distinct_formal_tasks(
    tasks: Sequence[Mapping[str, Any] | Any], *, expected_total: int,
) -> None:
    """Fail closed unless a formal selection contains distinct physical tasks."""

    task_ids = [str(_field(task, "task_id", "")) for task in tasks]
    signatures = [task_signature(task) for task in tasks]
    if len(tasks) != expected_total:
        raise ProtocolError(
            f"formal selection returned {len(tasks)} tasks, expected {expected_total}"
        )
    if any(not item for item in task_ids) or len(set(task_ids)) != expected_total:
        raise ProtocolError("formal selection contains empty/duplicate task_id values")
    if any(not item for item in signatures) or len(set(signatures)) != expected_total:
        raise ProtocolError("formal selection contains duplicate task_signature values")


def hash_task_manifest(tasks: Sequence[TaskManifest]) -> str:
    return sha256_json([task.to_dict() for task in tasks])


def ensure_task_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Materialize/verify the external task manifest from immutable run identity."""
    target = Path(path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "task_manifest_hash": manifest.task_manifest_hash,
        "tasks": [item.to_dict() for item in manifest.tasks],
    }
    if target.is_file():
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"task manifest is unreadable: {target}") from exc
        if _canonical_json(current) != _canonical_json(expected):
            raise ProtocolError(f"task manifest differs from immutable run manifest: {target}")
        return target
    _atomic_write_json(target, expected)
    return target


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def hash_config(config: Mapping[str, Any] | Sequence[Any] | str | Path | Any) -> str:
    """Hash parsed config semantics, not YAML whitespace or key order."""

    if isinstance(config, Path) and not config.is_file():
        raise FileNotFoundError(config)
    path = _existing_path(config)
    if path is None:
        return sha256_json(_to_primitive(config))
    suffix = path.suffix.casefold()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        parsed = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project dependency
            raise ProtocolError("PyYAML is required to hash YAML configuration") from exc
        parsed = yaml.safe_load(text)
    else:
        parsed = {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return sha256_json(parsed)


def hash_code(root: str | Path) -> str:
    """Hash experiment-relevant source/config files in deterministic path order."""

    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in _CODE_SUFFIXES
        and not any(
            part in _CODE_EXCLUDED_DIRS or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
    ]
    records = []
    for path in sorted(files, key=lambda item: item.as_posix()):
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sha256_json(records)


def hash_knowledge(
    data_dir: str | Path,
    *,
    database: Any | None = None,
) -> str:
    """Hash immutable artifacts plus all mutable long-term knowledge tables.

    Run/task bookkeeping and eval traces are intentionally excluded: they are
    outputs, not executable knowledge.  Evidence and lifecycle tables are
    included so Frozen evaluation detects forbidden learning writes.
    """

    data_dir = Path(data_dir).resolve()
    file_records: list[dict[str, str]] = []
    roots = []
    artifact_root = data_dir if data_dir.name == "artifacts" else data_dir / "artifacts"
    if artifact_root.is_dir():
        roots.append(artifact_root)
    for name in ("indexes", "query_index", "failure_knowledge"):
        candidate = data_dir / name
        if candidate.is_dir():
            roots.append(candidate)
    for root in roots:
        for path in sorted((item for item in root.rglob("*") if item.is_file())):
            file_records.append(
                {
                    "path": path.relative_to(data_dir).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    table_records: dict[str, list[list[Any]]] = {}
    close_connection = False
    connection = None
    if database is not None:
        connection = getattr(database, "connection", database)
    else:
        candidates = (
            data_dir / "state" / "asg_v3.sqlite",
            data_dir / "asg_v3.sqlite",
            data_dir / "state.sqlite3",
            data_dir / "state.sqlite",
        )
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            raise ProtocolError(f"ambiguous knowledge database paths: {existing}")
        if existing:
            connection = sqlite3.connect(f"file:{existing[0].as_posix()}?mode=ro", uri=True)
            close_connection = True

    try:
        if connection is not None:
            table_records = _knowledge_table_rows(connection)
    finally:
        if close_connection and connection is not None:
            connection.close()
    return sha256_json({"files": file_records, "tables": table_records})


def artifact_audit_snapshot(database: Any) -> dict[str, Any]:
    """Capture authoritative registry status and lifecycle projection state.

    Absolute artifact paths are intentionally excluded.  The full immutable
    identity/status rows and projection payloads remain in the Trace so a
    historical per-task report never has to infer growth from mutable logs.
    """

    connection = getattr(database, "connection", database)
    artifact_records = [
        {
            "artifact_ref": str(row["artifact_ref"]),
            "artifact_kind": str(row["artifact_kind"]),
            "logical_id": str(row["logical_id"]),
            "version": str(row["version"]),
            "content_hash": str(row["content_hash"]),
            "status": str(row["status"]),
            "schema_version": int(row["schema_version"]),
        }
        for row in connection.execute(
            "SELECT artifact_ref,artifact_kind,logical_id,version,content_hash,status,"
            "schema_version FROM artifact_index ORDER BY artifact_ref"
        ).fetchall()
    ]
    by_ref = {item["artifact_ref"]: item for item in artifact_records}
    by_kind: dict[str, int] = {}
    by_kind_status: dict[str, dict[str, int]] = {}
    for item in artifact_records:
        kind, status = item["artifact_kind"], item["status"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
        statuses = by_kind_status.setdefault(kind, {})
        statuses[status] = statuses.get(status, 0) + 1

    projection_records: list[dict[str, Any]] = []
    projection_by_kind: dict[str, int] = {}
    projection_event_counts: dict[str, int] = {}
    for row in connection.execute(
        "SELECT artifact_ref,projection_json,last_event_rowid "
        "FROM lifecycle_projection ORDER BY artifact_ref"
    ).fetchall():
        artifact_ref = str(row["artifact_ref"])
        if artifact_ref not in by_ref:
            raise ProtocolError(
                f"lifecycle projection references unknown artifact: {artifact_ref}"
            )
        try:
            projection = json.loads(str(row["projection_json"]))
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                f"invalid lifecycle projection JSON for {artifact_ref}"
            ) from exc
        if not isinstance(projection, dict):
            raise ProtocolError(f"lifecycle projection must be an object: {artifact_ref}")
        if str(projection.get("artifact_ref", "")) != artifact_ref:
            raise ProtocolError(f"lifecycle projection identity mismatch: {artifact_ref}")
        kind = str(projection.get("artifact_kind", ""))
        if kind != by_ref[artifact_ref]["artifact_kind"]:
            raise ProtocolError(f"lifecycle projection kind mismatch: {artifact_ref}")
        last_event_rowid = int(row["last_event_rowid"])
        if int(projection.get("last_event_rowid", -1)) != last_event_rowid:
            raise ProtocolError(f"lifecycle projection rowid mismatch: {artifact_ref}")
        projection_by_kind[kind] = projection_by_kind.get(kind, 0) + 1
        event_counts = projection.get("event_counts", {})
        if not isinstance(event_counts, dict):
            raise ProtocolError(f"lifecycle event_counts must be an object: {artifact_ref}")
        for event_type, count in event_counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ProtocolError(
                    f"invalid lifecycle event count for {artifact_ref}: {event_type}={count!r}"
                )
            projection_event_counts[str(event_type)] = (
                projection_event_counts.get(str(event_type), 0) + count
            )
        projection_records.append({
            "artifact_ref": artifact_ref,
            "last_event_rowid": last_event_rowid,
            "projection": projection,
        })

    checkpoint_row = connection.execute(
        "SELECT last_event_rowid FROM projection_checkpoints "
        "WHERE projection_name='lifecycle_v3'"
    ).fetchone()
    checkpoint = 0 if checkpoint_row is None else int(checkpoint_row["last_event_rowid"])
    ledger_row = connection.execute(
        "SELECT COALESCE(MAX(rowid),0) AS rowid FROM evidence_events"
    ).fetchone()
    ledger_max_rowid = int(ledger_row["rowid"])
    if checkpoint != ledger_max_rowid:
        raise ProtocolError(
            "lifecycle projection is not current with EvidenceLedger: "
            f"checkpoint={checkpoint}, ledger_max_rowid={ledger_max_rowid}"
        )

    artifact_index = {
        "total": len(artifact_records),
        "by_kind": dict(sorted(by_kind.items())),
        "by_kind_status": {
            kind: dict(sorted(statuses.items()))
            for kind, statuses in sorted(by_kind_status.items())
        },
        "records": artifact_records,
        "digest": sha256_json(artifact_records),
    }
    lifecycle_projection = {
        "total": len(projection_records),
        "by_kind": dict(sorted(projection_by_kind.items())),
        "event_counts": dict(sorted(projection_event_counts.items())),
        "checkpoint": checkpoint,
        "ledger_max_rowid": ledger_max_rowid,
        "records": projection_records,
        "digest": sha256_json(projection_records),
    }
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "artifact_index": artifact_index,
        "lifecycle_projection": lifecycle_projection,
    }
    snapshot["snapshot_digest"] = sha256_json(snapshot)
    return snapshot


def artifact_growth_audit(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one compact, exact count delta while retaining snapshot digests."""

    before_compact = _compact_artifact_snapshot(before)
    after_compact = _compact_artifact_snapshot(after)
    return {
        "schema_version": SCHEMA_VERSION,
        "before": before_compact,
        "after": after_compact,
        "delta": {
            "artifact_total": (
                int(after_compact["artifact_index"]["total"])
                - int(before_compact["artifact_index"]["total"])
            ),
            "artifact_by_kind": _nested_count_delta(
                before_compact["artifact_index"]["by_kind"],
                after_compact["artifact_index"]["by_kind"],
            ),
            "artifact_by_kind_status": _nested_count_delta(
                before_compact["artifact_index"]["by_kind_status"],
                after_compact["artifact_index"]["by_kind_status"],
            ),
            "lifecycle_projection_total": (
                int(after_compact["lifecycle_projection"]["total"])
                - int(before_compact["lifecycle_projection"]["total"])
            ),
            "lifecycle_event_counts": _nested_count_delta(
                before_compact["lifecycle_projection"]["event_counts"],
                after_compact["lifecycle_projection"]["event_counts"],
            ),
        },
    }


def load_task_report_traces(
    trace_store: Any,
    database: Any,
    run_id: str,
) -> list[dict[str, Any]]:
    """Overlay durable per-task audit results without mutating immutable Traces."""

    connection = getattr(database, "connection", database)
    rows = connection.execute(
        "SELECT task_id,state,trace_id,result_json FROM run_tasks "
        "WHERE run_id=? ORDER BY rowid",
        (run_id,),
    ).fetchall()
    traces: list[dict[str, Any]] = []
    for row in rows:
        if str(row["state"]) != TaskState.COMPLETED.value:
            continue
        trace_id = str(row["trace_id"])
        if not trace_id:
            raise ProtocolError(f"completed task {row['task_id']!r} lacks trace_id")
        payload = trace_store.load_payload(trace_id)
        if str(payload.get("trace_id", "")) != trace_id:
            raise ProtocolError(f"persisted Trace identity mismatch: {trace_id}")
        task_payload = payload.get("task")
        if not isinstance(task_payload, dict) or str(task_payload.get("task_id", "")) != str(
            row["task_id"]
        ):
            raise ProtocolError(f"persisted Trace task mismatch: {trace_id}")
        try:
            result = json.loads(str(row["result_json"]))
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid task result JSON: {row['task_id']}") from exc
        if not isinstance(result, dict):
            raise ProtocolError(f"task result must be an object: {row['task_id']}")
        for key in ("artifact_growth", "artifact_lifecycle"):
            if not isinstance(result.get(key), dict) or not result[key]:
                raise ProtocolError(
                    f"completed task {row['task_id']!r} lacks authoritative {key} snapshot"
                )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata
        metadata["artifact_growth"] = result["artifact_growth"]
        metadata["artifact_lifecycle"] = result["artifact_lifecycle"]
        if isinstance(result.get("final_batch_maintenance"), dict):
            metadata["final_batch_maintenance"] = result["final_batch_maintenance"]
        traces.append(payload)
    return traces


def _compact_artifact_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = dict(snapshot.get("artifact_index") or {})
    projection = dict(snapshot.get("lifecycle_projection") or {})
    return {
        "snapshot_digest": str(snapshot.get("snapshot_digest", "")),
        "artifact_index": {
            "total": int(artifacts.get("total", 0)),
            "by_kind": dict(artifacts.get("by_kind") or {}),
            "by_kind_status": {
                str(kind): dict(statuses)
                for kind, statuses in dict(artifacts.get("by_kind_status") or {}).items()
            },
            "digest": str(artifacts.get("digest", "")),
        },
        "lifecycle_projection": {
            "total": int(projection.get("total", 0)),
            "by_kind": dict(projection.get("by_kind") or {}),
            "event_counts": dict(projection.get("event_counts") or {}),
            "checkpoint": int(projection.get("checkpoint", 0)),
            "ledger_max_rowid": int(projection.get("ledger_max_rowid", 0)),
            "digest": str(projection.get("digest", "")),
        },
    }


def _nested_count_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        left, right = before.get(key, 0), after.get(key, 0)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            result[str(key)] = _nested_count_delta(
                left if isinstance(left, Mapping) else {},
                right if isinstance(right, Mapping) else {},
            )
        else:
            result[str(key)] = int(right) - int(left)
    return result


config_digest = hash_config
code_digest = hash_code
knowledge_digest = hash_knowledge


def compare_manifests(
    persisted: RunManifest, current: RunManifest
) -> tuple[FieldMismatch, ...]:
    mismatches: list[FieldMismatch] = []
    scalar_fields = (
        "schema_version",
        "run_id",
        "phase",
        "created_at",
        "config_hash",
        "code_commit",
        "knowledge_digest",
        "task_manifest_hash",
        "metadata_json",
    )
    for name in scalar_fields:
        left, right = getattr(persisted, name), getattr(current, name)
        if left != right:
            mismatches.append(FieldMismatch(name, left, right))
    if len(persisted.tasks) != len(current.tasks):
        mismatches.append(FieldMismatch("tasks.length", len(persisted.tasks), len(current.tasks)))
    for index, (left, right) in enumerate(zip(persisted.tasks, current.tasks)):
        for item in fields(TaskManifest):
            old, new = getattr(left, item.name), getattr(right, item.name)
            if old != new:
                mismatches.append(FieldMismatch(f"tasks[{index}].{item.name}", old, new))
    return tuple(mismatches)


class ManifestStore:
    """Persist immutable manifests and manage only mutable SQLite run state."""

    def __init__(self, root: str | Path, database: Any) -> None:
        self.root = Path(root)
        self.database = database
        if getattr(database, "readonly", False):
            raise RuntimeError("run protocol requires a writable state database")
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"unsafe or invalid run_id: {run_id!r}")
        return self.root / run_id / "run_manifest.json"

    def load(self, run_id: str) -> RunManifest:
        path = self.path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def persist_before_run(self, manifest: RunManifest, *, resume: bool = False) -> Path:
        """Write/validate manifest and DB rows before any task can be marked running."""

        path = self.path_for(manifest.run_id)
        db_row = self.database.execute(
            "SELECT run_id FROM run_manifests WHERE run_id=?", (manifest.run_id,)
        ).fetchone()
        if path.exists():
            persisted = self.load(manifest.run_id)
            mismatches = compare_manifests(persisted, manifest)
            if mismatches:
                raise ManifestMismatchError(mismatches)
            if not resume:
                raise ManifestExistsError(
                    f"run {manifest.run_id!r} already has an immutable manifest; use resume"
                )
            self._validate_database(persisted)
            return path
        if resume:
            raise ProtocolError(f"cannot resume {manifest.run_id!r}: immutable manifest is missing")
        if db_row is not None:
            raise ProtocolError(
                f"run {manifest.run_id!r} exists in SQLite but its immutable manifest is missing"
            )

        _atomic_write_json(path, manifest.to_dict())
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO run_manifests(run_id,phase,config_hash,task_manifest_hash,"
                    "code_commit,state) VALUES(?,?,?,?,?,?)",
                    (
                        manifest.run_id,
                        manifest.phase,
                        manifest.config_hash,
                        manifest.task_manifest_hash,
                        manifest.code_commit,
                        RunState.PENDING.value,
                    ),
                )
                for task in manifest.tasks:
                    connection.execute(
                        "INSERT INTO run_tasks(run_id,task_id,task_signature,config_hash,"
                        "code_commit,knowledge_milestone,state) VALUES(?,?,?,?,?,?,?)",
                        (
                            manifest.run_id,
                            task.task_id,
                            task.task_signature,
                            manifest.config_hash,
                            manifest.code_commit,
                            task.knowledge_milestone,
                            TaskState.PENDING.value,
                        ),
                    )
        except Exception as exc:
            # The file intentionally remains as immutable crash evidence.  A
            # later resume will fail closed if DB registration is incomplete.
            raise ProtocolError(
                f"manifest file was written but SQLite registration failed for {manifest.run_id}"
            ) from exc
        return path

    prepare_run = persist_before_run
    create_before_run = persist_before_run

    def validate_resume(
        self,
        run_id: str,
        *,
        config_hash: str,
        code_commit: str,
        knowledge_digest: str,
        tasks: Sequence[TaskManifest] | None = None,
    ) -> RunManifest:
        persisted = self.load(run_id)
        mismatches: list[FieldMismatch] = []
        checks = {
            "config_hash": config_hash,
            "code_commit": code_commit,
            "knowledge_digest": knowledge_digest,
        }
        for name, current in checks.items():
            old = getattr(persisted, name)
            if old != current:
                mismatches.append(FieldMismatch(name, old, current))
        if tasks is not None:
            current_hash = hash_task_manifest(tasks)
            if current_hash != persisted.task_manifest_hash:
                mismatches.append(
                    FieldMismatch(
                        "task_manifest_hash", persisted.task_manifest_hash, current_hash
                    )
                )
            current_by_id = {task.task_id: task for task in tasks}
            for index, old in enumerate(persisted.tasks):
                current = current_by_id.get(old.task_id)
                if current is None:
                    mismatches.append(
                        FieldMismatch(f"tasks[{index}].task_id", old.task_id, "<missing>")
                    )
                    continue
                for item in fields(TaskManifest):
                    before, now = getattr(old, item.name), getattr(current, item.name)
                    if before != now:
                        mismatches.append(
                            FieldMismatch(f"tasks[{index}].{item.name}", before, now)
                        )
            extra = sorted(set(current_by_id) - {task.task_id for task in persisted.tasks})
            if extra:
                mismatches.append(FieldMismatch("tasks.extra", (), tuple(extra)))
        if mismatches:
            raise ManifestMismatchError(mismatches)
        self._validate_database(persisted)
        return persisted

    def tasks_to_run(self, manifest: RunManifest) -> tuple[TaskManifest, ...]:
        self._validate_database(manifest)
        state_rows = {
            str(row["task_id"]): str(row["state"])
            for row in self.database.rows(
                "SELECT task_id,state FROM run_tasks WHERE run_id=?", (manifest.run_id,)
            )
        }
        return tuple(
            task
            for task in manifest.tasks
            if state_rows[task.task_id] != TaskState.COMPLETED.value
        )

    def mark_run_state(self, run_id: str, state: RunState | str) -> None:
        state = RunState(state)
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE run_manifests SET state=? WHERE run_id=?", (state.value, run_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)

    def mark_task_running(
        self,
        run_id: str,
        task_id: str,
        *,
        max_attempts: int | None = None,
    ) -> int:
        if max_attempts is not None and (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer or None")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT state,attempt_count FROM run_tasks WHERE run_id=? AND task_id=?",
                (run_id, task_id),
            ).fetchone()
            if row is None:
                raise KeyError((run_id, task_id))
            if row["state"] not in {
                TaskState.PENDING.value,
                # A process can die after the durable running transition but
                # before a terminal task update.  Resume replays that task from
                # its initial environment state and records a new attempt.
                TaskState.RUNNING.value,
                TaskState.INFRASTRUCTURE_FAILED.value,
                TaskState.TASK_FAILED.value,
            }:
                raise ProtocolError(
                    f"task {task_id!r} cannot enter running from {row['state']!r}"
                )
            if max_attempts is not None and int(row["attempt_count"]) >= max_attempts:
                raise ProtocolError(
                    f"task {task_id!r} exhausted max_task_attempts={max_attempts}"
                )
            attempt_count = int(row["attempt_count"]) + 1
            connection.execute(
                "UPDATE run_tasks SET state=?,attempt_count=? WHERE run_id=? AND task_id=?",
                (TaskState.RUNNING.value, attempt_count, run_id, task_id),
            )
        return attempt_count

    def mark_task_completed(
        self,
        run_id: str,
        task_id: str,
        *,
        trace_id: str,
        result: Mapping[str, Any],
    ) -> None:
        if not trace_id:
            raise ValueError("completed task requires trace_id")
        self._finish_task(
            run_id,
            task_id,
            TaskState.COMPLETED,
            trace_id=trace_id,
            result=result,
        )

    def mark_task_failed(
        self,
        run_id: str,
        task_id: str,
        *,
        infrastructure: bool,
        trace_id: str = "",
        result: Mapping[str, Any] | None = None,
    ) -> None:
        self._finish_task(
            run_id,
            task_id,
            TaskState.INFRASTRUCTURE_FAILED if infrastructure else TaskState.TASK_FAILED,
            trace_id=trace_id,
            result=result or {},
        )

    def update_completed_task_knowledge_digest(
        self,
        run_id: str,
        task_id: str,
        *,
        expected_digest: str,
        new_digest: str,
        result_updates: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically extend the final completed-task milestone.

        Formal train performs one configured-batch maintenance pass after the
        last episode.  That pass is part of the last task boundary for the
        source-bank digest chain, even though it does not change the immutable
        task manifest.  This guarded update cannot be used to rewrite any
        other completed outcome or to conceal an unexpected knowledge change.
        """

        if not expected_digest or not new_digest:
            raise ValueError("knowledge digests must be non-empty")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT state,result_json FROM run_tasks WHERE run_id=? AND task_id=?",
                (run_id, task_id),
            ).fetchone()
            if row is None:
                raise KeyError((run_id, task_id))
            if str(row["state"]) != TaskState.COMPLETED.value:
                raise ProtocolError(
                    f"task {task_id!r} must be completed before final maintenance milestone"
                )
            try:
                result = json.loads(str(row["result_json"]))
            except json.JSONDecodeError as exc:
                raise ProtocolError(
                    f"completed task {task_id!r} has invalid result_json"
                ) from exc
            if not isinstance(result, dict):
                raise ProtocolError(f"completed task {task_id!r} result must be an object")
            persisted = str(result.get("knowledge_digest_after", ""))
            if persisted != expected_digest:
                raise ProtocolError(
                    f"completed task {task_id!r} digest changed before final maintenance: "
                    f"expected {expected_digest}, persisted {persisted}"
                )
            result["knowledge_digest_after"] = new_digest
            for key, value in dict(result_updates or {}).items():
                if key in {
                    "knowledge_digest_before",
                    "knowledge_digest_after",
                    "benchmark_success",
                    "task_contract_success",
                    "strict_task_success",
                    "learning_eligible",
                    "graph_self_sufficient_success",
                    "infrastructure_failure",
                }:
                    raise ValueError(f"final maintenance cannot rewrite task result field {key!r}")
                result[str(key)] = _to_primitive(value)
            cursor = connection.execute(
                "UPDATE run_tasks SET result_json=? "
                "WHERE run_id=? AND task_id=? AND state=?",
                (
                    _canonical_json(result),
                    run_id,
                    task_id,
                    TaskState.COMPLETED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ProtocolError(
                    f"task {task_id!r} completion changed during maintenance milestone update"
                )

    def _finish_task(
        self,
        run_id: str,
        task_id: str,
        state: TaskState,
        *,
        trace_id: str,
        result: Mapping[str, Any],
    ) -> None:
        encoded = _canonical_json(result)
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE run_tasks SET state=?,trace_id=?,result_json=? "
                "WHERE run_id=? AND task_id=? AND state=?",
                (
                    state.value,
                    trace_id,
                    encoded,
                    run_id,
                    task_id,
                    TaskState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ProtocolError(
                    f"task {task_id!r} must be running before terminal state {state.value!r}"
                )

    def _validate_database(self, manifest: RunManifest) -> None:
        mismatches: list[FieldMismatch] = []
        row = self.database.execute(
            "SELECT * FROM run_manifests WHERE run_id=?", (manifest.run_id,)
        ).fetchone()
        if row is None:
            raise ProtocolError(f"run {manifest.run_id!r} manifest is not registered in SQLite")
        run_fields = {
            "phase": manifest.phase,
            "config_hash": manifest.config_hash,
            "task_manifest_hash": manifest.task_manifest_hash,
            "code_commit": manifest.code_commit,
        }
        for name, expected in run_fields.items():
            if str(row[name]) != expected:
                mismatches.append(
                    FieldMismatch(f"sqlite.run.{name}", str(row[name]), expected)
                )
        if str(row["state"]) not in {state.value for state in RunState}:
            mismatches.append(
                FieldMismatch("sqlite.run.state", str(row["state"]), "<valid RunState>")
            )

        rows = self.database.rows(
            "SELECT * FROM run_tasks WHERE run_id=? ORDER BY task_id", (manifest.run_id,)
        )
        by_id = {str(item["task_id"]): item for item in rows}
        if len(rows) != len(manifest.tasks):
            mismatches.append(
                FieldMismatch("sqlite.tasks.length", len(rows), len(manifest.tasks))
            )
        for index, task in enumerate(manifest.tasks):
            stored = by_id.get(task.task_id)
            if stored is None:
                mismatches.append(
                    FieldMismatch(f"sqlite.tasks[{index}].task_id", "<missing>", task.task_id)
                )
                continue
            expected = {
                "task_signature": task.task_signature,
                "config_hash": manifest.config_hash,
                "code_commit": manifest.code_commit,
                "knowledge_milestone": task.knowledge_milestone,
            }
            for name, current in expected.items():
                if str(stored[name]) != current:
                    mismatches.append(
                        FieldMismatch(
                            f"sqlite.tasks[{index}].{name}", str(stored[name]), current
                        )
                    )
            if str(stored["state"]) not in {state.value for state in TaskState}:
                mismatches.append(
                    FieldMismatch(
                        f"sqlite.tasks[{index}].state",
                        str(stored["state"]),
                        "<valid TaskState>",
                    )
                )
        extras = sorted(set(by_id) - {task.task_id for task in manifest.tasks})
        if extras:
            mismatches.append(FieldMismatch("sqlite.tasks.extra", tuple(extras), ()))
        if mismatches:
            raise ManifestMismatchError(mismatches)


class AttemptTraceLedger:
    """Append-only usage ownership for every formal execution attempt.

    The ledger deliberately lives outside ``data_v3``.  A task checkpoint may
    roll back the knowledge database and artifact tree, but it must never erase
    the immutable Trace references needed to account for API work already paid
    for.  Each attempt writes one immutable start record before execution and
    one immutable capture record afterwards.  A later process can close a start
    record left pending by a hard crash by comparing its trace baseline with the
    still-persistent TraceStore.
    """

    _KINDS = frozenset({"task", "maintenance"})

    def __init__(self, root: str | Path, trace_root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.trace_root = Path(trace_root).resolve()
        if (
            self.root == self.trace_root
            or self.root in self.trace_root.parents
            or self.trace_root in self.root.parents
        ):
            raise ValueError("attempt ledger and TraceStore must be disjoint")

    @property
    def owner_path(self) -> Path:
        return self.trace_root / ".formal_run_owner.json"

    def begin(
        self,
        *,
        run_id: str,
        task_id: str,
        task_signature: str,
        attempt_kind: str,
        sequence: int,
        expected_periodic_milestone: str = "",
    ) -> AttemptTraceRef:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"unsafe or invalid run_id: {run_id!r}")
        if not task_id or not task_signature:
            raise ValueError("attempt requires task_id and task_signature")
        if attempt_kind not in self._KINDS:
            raise ValueError(f"invalid attempt kind: {attempt_kind!r}")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError("attempt sequence must be a positive integer")
        if expected_periodic_milestone and (
            attempt_kind != "task"
            or not re.fullmatch(r"online_success_[1-9][0-9]*", expected_periodic_milestone)
        ):
            raise ValueError("invalid expected periodic maintenance milestone")
        self._ensure_owner(run_id)
        pending = self.pending(run_id=run_id)
        if pending:
            raise ProtocolError(
                "cannot start an attempt while an earlier capture is pending: "
                + ", ".join(item.attempt_id for item in pending)
            )
        identity = {
            "run_id": run_id,
            "task_id": task_id,
            "task_signature": task_signature,
            "attempt_kind": attempt_kind,
            "sequence": sequence,
        }
        attempt_id = "attempt_" + sha256_json(identity)[:32]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt_id,
            **identity,
            "expected_periodic_milestone": expected_periodic_milestone,
            "trace_baseline": self._trace_hashes(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._start_path(attempt_id)
        if path.exists():
            existing = self._read(path)
            if any(existing.get(key) != value for key, value in payload.items() if key != "created_at"):
                raise ProtocolError(f"attempt start identity collision: {attempt_id}")
        else:
            _atomic_write_json(path, payload)
        return AttemptTraceRef(
            attempt_id=attempt_id,
            **identity,
            expected_periodic_milestone=expected_periodic_milestone,
        )

    def next_sequence(self, *, run_id: str, task_id: str, attempt_kind: str) -> int:
        self._ensure_owner(run_id)
        sequences = [
            item.sequence
            for item in self._starts(run_id=run_id)
            if item.task_id == task_id and item.attempt_kind == attempt_kind
        ]
        return max(sequences, default=0) + 1

    def capture(self, attempt: AttemptTraceRef, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("attempt capture requires a reason")
        start = self._read(self._start_path(attempt.attempt_id))
        self._validate_ref(attempt, start)
        capture_path = self._capture_path(attempt.attempt_id)
        if capture_path.exists():
            capture = self._read(capture_path)
            self._validate_capture(start, capture)
            self._validate_baseline_hashes(start)
            self._validate_captured_hashes(start, capture)
            self._validate_periodic_expectation(capture, capture["trace_ids"])
            self._claim_traces(start, capture["trace_ids"])
            return capture

        baseline_hashes = self._validate_baseline_hashes(start)
        current_hashes = self._trace_hashes()
        captured_ids = sorted(set(current_hashes) - set(baseline_hashes))
        if not captured_ids:
            raise ProtocolError(
                f"attempt {attempt.attempt_id} has no immutable Trace; "
                "formal usage cannot be proven"
            )
        task_trace_count = 0
        for trace_id in captured_ids:
            payload = self._validate_owned_trace(start, trace_id)
            if str(_field(payload.get("task", {}), "task_type", "")) != "maintenance":
                task_trace_count += 1
        if task_trace_count > 1:
            raise ProtocolError(
                f"attempt {attempt.attempt_id} captured multiple task Traces"
            )
        if attempt.attempt_kind == "task" and task_trace_count != 1:
            raise ProtocolError(
                f"task attempt {attempt.attempt_id} lacks one immutable task Trace"
            )
        self._validate_periodic_expectation(start, captured_ids)
        self._claim_traces(start, captured_ids)
        capture = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt.attempt_id,
            "run_id": attempt.run_id,
            "task_id": attempt.task_id,
            "task_signature": attempt.task_signature,
            "attempt_kind": attempt.attempt_kind,
            "sequence": attempt.sequence,
            "expected_periodic_milestone": attempt.expected_periodic_milestone,
            "trace_ids": captured_ids,
            "trace_hashes": {
                trace_id: current_hashes[trace_id] for trace_id in captured_ids
            },
            "reason": reason,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(capture_path, capture)
        return capture

    def recover_pending(self, *, run_id: str) -> list[dict[str, Any]]:
        self._ensure_owner(run_id)
        recovered = []
        for attempt in self.pending(run_id=run_id):
            start = self._read(self._start_path(attempt.attempt_id))
            baseline = self._validate_baseline_hashes(start)
            if set(self._trace_hashes()) == set(baseline):
                recovered.append(self._quarantine_untraced_attempt(start))
                continue
            recovered.append(self.capture(attempt, reason="resume_recovery"))
        return recovered

    def pending(self, *, run_id: str) -> tuple[AttemptTraceRef, ...]:
        self._ensure_owner(run_id)
        return tuple(
            attempt
            for attempt in self._starts(run_id=run_id)
            if not self._capture_path(attempt.attempt_id).is_file()
            and not self._unresolved_path(attempt.attempt_id).is_file()
        )

    def unresolved(self, *, run_id: str) -> tuple[dict[str, Any], ...]:
        """Return crash quarantines that deliberately block a formal report.

        A process death after ``begin`` but before a durable Trace cannot prove
        that no provider request occurred.  Such an attempt must not deadlock
        all later retries, but it also must not be represented as zero usage.
        """

        self._ensure_owner(run_id)
        rows: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return ()
        for path in sorted(self.root.glob("attempt_*.unresolved.json")):
            payload = self._read(path)
            if str(payload.get("run_id", "")) != run_id:
                continue
            attempt_id = str(payload.get("attempt_id", ""))
            if path != self._unresolved_path(attempt_id):
                raise ProtocolError(
                    f"attempt unresolved filename/identity mismatch: {path}"
                )
            start = self._read(self._start_path(attempt_id))
            self._validate_unresolved(start, payload)
            rows.append(payload)
        return tuple(rows)

    def auxiliary_traces(
        self,
        *,
        manifest: RunManifest,
        excluded_trace_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Return non-authoritative attempt Traces for resource accounting only."""

        self._ensure_owner(manifest.run_id)
        self._validate_run_files(manifest.run_id)
        excluded = {str(item) for item in excluded_trace_ids}
        manifest_tasks = {item.task_id: item for item in manifest.tasks}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        pending = self.pending(run_id=manifest.run_id)
        if pending:
            raise ProtocolError(
                "formal report has uncaptured attempts: "
                + ", ".join(item.attempt_id for item in pending)
            )
        unresolved = self.unresolved(run_id=manifest.run_id)
        if unresolved:
            raise ProtocolError(
                "formal report has unresolved attempts with no durable Trace; "
                "provider usage cannot be proven: "
                + ", ".join(str(item["attempt_id"]) for item in unresolved)
            )
        for attempt in self._starts(run_id=manifest.run_id):
            expected_task = manifest_tasks.get(attempt.task_id)
            if expected_task is None or expected_task.task_signature != attempt.task_signature:
                raise ProtocolError(
                    f"attempt {attempt.attempt_id} is not associated with the immutable run manifest"
                )
            capture = self._read(self._capture_path(attempt.attempt_id))
            start = self._read(self._start_path(attempt.attempt_id))
            self._validate_capture(start, capture)
            self._validate_baseline_hashes(start)
            self._validate_captured_hashes(start, capture)
            self._validate_periodic_expectation(capture, capture["trace_ids"])
            self._claim_traces(start, capture["trace_ids"])
            for trace_id in capture["trace_ids"]:
                if trace_id in seen:
                    raise ProtocolError(
                        f"attempt Trace is referenced more than once in run {manifest.run_id}: {trace_id}"
                    )
                seen.add(trace_id)
                payload = self._validate_owned_trace(capture, trace_id)
                if trace_id not in excluded:
                    selected.append(payload)
        return selected

    def _starts(self, *, run_id: str) -> tuple[AttemptTraceRef, ...]:
        if not self.root.is_dir():
            return ()
        starts: list[AttemptTraceRef] = []
        for path in sorted(self.root.glob("attempt_*.start.json")):
            payload = self._read(path)
            if str(payload.get("run_id", "")) != run_id:
                continue
            attempt = self._ref_from_payload(payload)
            if path != self._start_path(attempt.attempt_id):
                raise ProtocolError(f"attempt start filename/identity mismatch: {path}")
            starts.append(attempt)
        starts.sort(key=lambda item: (item.sequence, item.attempt_kind, item.task_id, item.attempt_id))
        return tuple(starts)

    def _ensure_owner(self, run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise ProtocolError(f"unsafe or invalid run_id: {run_id!r}")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
        }
        self.trace_root.mkdir(parents=True, exist_ok=True)
        if not self.owner_path.exists():
            try:
                atomic_create_json(self.owner_path, expected)
            except FileExistsError:
                pass
        actual = self._read(self.owner_path)
        if actual != expected:
            raise ProtocolError(
                f"TraceStore is owned by a different formal run: {self.trace_root}"
            )

    def _validate_run_files(self, run_id: str) -> None:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("attempt_*.capture.json")):
            capture = self._read(path)
            if str(capture.get("run_id", "")) != run_id:
                continue
            attempt_id = str(capture.get("attempt_id", ""))
            if path != self._capture_path(attempt_id):
                raise ProtocolError(f"attempt capture filename/identity mismatch: {path}")
            start_path = self._start_path(attempt_id)
            if not start_path.is_file():
                raise ProtocolError(f"attempt capture has no immutable start record: {path}")
            self._validate_capture(self._read(start_path), capture)
        self.unresolved(run_id=run_id)

    def _quarantine_untraced_attempt(
        self, start: Mapping[str, Any],
    ) -> dict[str, Any]:
        attempt_id = str(start["attempt_id"])
        target = self._unresolved_path(attempt_id)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "run_id": str(start["run_id"]),
            "task_id": str(start["task_id"]),
            "task_signature": str(start["task_signature"]),
            "attempt_kind": str(start["attempt_kind"]),
            "sequence": int(start["sequence"]),
            "status": "unresolved_no_durable_trace",
            "resource_usage_status": "unproven",
            "reason": "resume_recovery_found_no_new_trace",
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            atomic_create_json(target, payload)
        except FileExistsError:
            existing = self._read(target)
            self._validate_unresolved(start, existing)
            return existing
        return payload

    @staticmethod
    def _validate_unresolved(
        start: Mapping[str, Any], payload: Mapping[str, Any],
    ) -> None:
        for key in (
            "schema_version", "attempt_id", "run_id", "task_id",
            "task_signature", "attempt_kind", "sequence",
        ):
            if payload.get(key) != start.get(key):
                raise ProtocolError(
                    f"attempt unresolved identity mismatch for "
                    f"{start.get('attempt_id', '<unknown>')}"
                )
        if (
            payload.get("status") != "unresolved_no_durable_trace"
            or payload.get("resource_usage_status") != "unproven"
            or payload.get("reason") != "resume_recovery_found_no_new_trace"
        ):
            raise ProtocolError(
                f"invalid unresolved attempt record: {start.get('attempt_id')}"
            )

    def _claim_traces(
        self, owner: Mapping[str, Any], trace_ids: Sequence[str],
    ) -> None:
        for trace_id in trace_ids:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "trace_id": trace_id,
                "run_id": str(owner["run_id"]),
                "attempt_id": str(owner["attempt_id"]),
                "task_id": str(owner["task_id"]),
                "attempt_kind": str(owner["attempt_kind"]),
            }
            path = self._claim_path(trace_id)
            try:
                atomic_create_json(path, payload)
            except FileExistsError:
                existing = self._read(path)
                if existing != payload:
                    raise ProtocolError(
                        f"immutable Trace {trace_id} is already claimed by "
                        f"another formal attempt"
                    ) from None

    def _trace_hashes(self) -> dict[str, str]:
        return {
            path.stem: _sha256_file(path)
            for path in sorted(self.trace_root.glob("trace_*.json"))
            if path.is_file()
        }

    def _validate_baseline_hashes(
        self, start: Mapping[str, Any],
    ) -> dict[str, str | None]:
        raw = start.get("trace_baseline")
        if isinstance(raw, list):
            if any(not isinstance(item, str) or not item for item in raw):
                raise ProtocolError(
                    f"attempt {start.get('attempt_id')} has an invalid legacy trace baseline"
                )
            baseline: dict[str, str | None] = {item: None for item in raw}
        elif isinstance(raw, dict):
            if any(
                not isinstance(trace_id, str)
                or not trace_id
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for trace_id, digest in raw.items()
            ):
                raise ProtocolError(
                    f"attempt {start.get('attempt_id')} has an invalid trace baseline"
                )
            baseline = {str(key): str(value) for key, value in raw.items()}
        else:
            raise ProtocolError(
                f"attempt {start.get('attempt_id')} has an invalid trace baseline"
            )
        current = self._trace_hashes()
        missing = sorted(set(baseline) - set(current))
        if missing:
            raise ProtocolError(
                f"immutable Trace baseline disappeared for {start.get('attempt_id')}: "
                + ", ".join(missing[:10])
            )
        changed = sorted(
            trace_id for trace_id, digest in baseline.items()
            if digest is not None and current[trace_id] != digest
        )
        if changed:
            raise ProtocolError(
                f"immutable Trace baseline content changed for {start.get('attempt_id')}: "
                + ", ".join(changed[:10])
            )
        return baseline

    def _validate_captured_hashes(
        self, start: Mapping[str, Any], capture: Mapping[str, Any],
    ) -> None:
        raw = capture.get("trace_hashes")
        trace_ids = list(capture.get("trace_ids") or [])
        if raw is None and isinstance(start.get("trace_baseline"), list):
            return
        if (
            not isinstance(raw, dict)
            or set(raw) != set(trace_ids)
            or any(
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in raw.values()
            )
        ):
            raise ProtocolError(
                f"attempt {capture.get('attempt_id')} has invalid captured Trace hashes"
            )
        current = self._trace_hashes()
        changed = sorted(
            trace_id for trace_id in trace_ids
            if trace_id not in current or current[trace_id] != raw[trace_id]
        )
        if changed:
            raise ProtocolError(
                f"captured immutable Trace content changed for "
                f"{capture.get('attempt_id')}: " + ", ".join(changed[:10])
            )

    def _validate_owned_trace(
        self, owner: Mapping[str, Any], trace_id: str,
    ) -> dict[str, Any]:
        path = self.trace_root / f"{trace_id}.json"
        if not path.is_file():
            raise ProtocolError(f"attempt references a missing immutable Trace: {trace_id}")
        payload = self._read(path)
        if str(payload.get("trace_id", "")) != trace_id:
            raise ProtocolError(f"attempt Trace identity mismatch: {trace_id}")
        task = payload.get("task")
        metadata = payload.get("metadata")
        if not isinstance(task, dict) or not isinstance(metadata, dict):
            raise ProtocolError(f"attempt Trace lacks task/metadata identity: {trace_id}")
        is_maintenance = (
            task.get("task_type") == "maintenance"
            or metadata.get("trace_kind") == "maintenance"
        )
        if is_maintenance:
            if (
                task.get("task_type") != "maintenance"
                or metadata.get("trace_kind") != "maintenance"
                or str(metadata.get("triggering_task_id", "")) != str(owner["task_id"])
            ):
                raise ProtocolError(f"maintenance Trace ownership mismatch: {trace_id}")
        elif (
            str(task.get("task_id", "")) != str(owner["task_id"])
            or str(task.get("task_signature", "")) != str(owner["task_signature"])
        ):
            raise ProtocolError(f"task Trace ownership mismatch: {trace_id}")
        if str(owner["attempt_kind"]) == "maintenance" and not is_maintenance:
            raise ProtocolError(f"maintenance attempt captured a task Trace: {trace_id}")
        return payload

    def _validate_periodic_expectation(
        self, owner: Mapping[str, Any], trace_ids: Sequence[str],
    ) -> None:
        expected = str(owner.get("expected_periodic_milestone", ""))
        if str(owner.get("attempt_kind", "")) != "task" or not expected:
            return
        payloads = [
            self._read(self.trace_root / f"{trace_id}.json") for trace_id in trace_ids
        ]
        task_traces = [
            payload for payload in payloads
            if _field(payload.get("task", {}), "task_type", "") != "maintenance"
        ]
        if len(task_traces) != 1:
            raise ProtocolError(
                f"task attempt {owner.get('attempt_id')} lacks one immutable task Trace"
            )
        if not bool(task_traces[0].get(
            "strict_task_success",
            task_traces[0].get("benchmark_success", False),
        )):
            return
        matches = [
            payload for payload in payloads
            if _field(payload.get("task", {}), "task_type", "") == "maintenance"
            and str(_field(payload.get("metadata", {}), "triggering_task_id", ""))
            == str(owner.get("task_id", ""))
            and str(_field(payload.get("metadata", {}), "milestone", "")) == expected
        ]
        if len(matches) != 1:
            raise ProtocolError(
                f"successful task attempt {owner.get('attempt_id')} lacks expected "
                f"periodic maintenance Trace {expected}"
            )

    def _validate_capture(
        self, start: Mapping[str, Any], capture: Mapping[str, Any],
    ) -> None:
        for key in (
            "schema_version", "attempt_id", "run_id", "task_id",
            "task_signature", "attempt_kind", "sequence",
        ):
            if capture.get(key) != start.get(key):
                raise ProtocolError(
                    f"attempt capture identity mismatch for {start.get('attempt_id', '<unknown>')}"
                )
        if str(capture.get("expected_periodic_milestone", "")) != str(
            start.get("expected_periodic_milestone", "")
        ):
            raise ProtocolError(
                f"attempt capture periodic expectation mismatch for "
                f"{start.get('attempt_id', '<unknown>')}"
            )
        trace_ids = capture.get("trace_ids")
        if (
            not isinstance(trace_ids, list)
            or not trace_ids
            or any(not isinstance(item, str) or not item for item in trace_ids)
            or trace_ids != sorted(trace_ids)
            or len(trace_ids) != len(set(trace_ids))
        ):
            raise ProtocolError(f"attempt {start.get('attempt_id')} has invalid trace_ids")

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"invalid attempt ledger file: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError(f"invalid attempt ledger payload: {path}")
        return payload

    @staticmethod
    def _ref_from_payload(payload: Mapping[str, Any]) -> AttemptTraceRef:
        try:
            raw_sequence = payload["sequence"]
            if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, int):
                raise TypeError("attempt sequence is not an integer")
            attempt = AttemptTraceRef(
                attempt_id=str(payload["attempt_id"]),
                run_id=str(payload["run_id"]),
                task_id=str(payload["task_id"]),
                task_signature=str(payload["task_signature"]),
                attempt_kind=str(payload["attempt_kind"]),
                sequence=raw_sequence,
                expected_periodic_milestone=str(
                    payload.get("expected_periodic_milestone", "")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("invalid attempt start identity") from exc
        identity = {
            "run_id": attempt.run_id,
            "task_id": attempt.task_id,
            "task_signature": attempt.task_signature,
            "attempt_kind": attempt.attempt_kind,
            "sequence": attempt.sequence,
        }
        expected_id = "attempt_" + sha256_json(identity)[:32]
        if (
            not _RUN_ID.fullmatch(attempt.run_id)
            or not attempt.task_id
            or not attempt.task_signature
            or attempt.attempt_kind not in AttemptTraceLedger._KINDS
            or attempt.sequence <= 0
            or (
                bool(attempt.expected_periodic_milestone)
                and (
                    attempt.attempt_kind != "task"
                    or not re.fullmatch(
                        r"online_success_[1-9][0-9]*",
                        attempt.expected_periodic_milestone,
                    )
                )
            )
            or attempt.attempt_id != expected_id
        ):
            raise ProtocolError("invalid attempt start identity")
        return attempt

    @staticmethod
    def _validate_ref(attempt: AttemptTraceRef, payload: Mapping[str, Any]) -> None:
        expected = {
            "attempt_id": attempt.attempt_id,
            "run_id": attempt.run_id,
            "task_id": attempt.task_id,
            "task_signature": attempt.task_signature,
            "attempt_kind": attempt.attempt_kind,
            "sequence": attempt.sequence,
            "expected_periodic_milestone": attempt.expected_periodic_milestone,
        }
        if any(
            (payload.get(key, "") if key == "expected_periodic_milestone" else payload.get(key))
            != value
            for key, value in expected.items()
        ):
            raise ProtocolError(f"attempt start identity mismatch: {attempt.attempt_id}")

    def _start_path(self, attempt_id: str) -> Path:
        return self.root / f"{attempt_id}.start.json"

    def _capture_path(self, attempt_id: str) -> Path:
        return self.root / f"{attempt_id}.capture.json"

    def _unresolved_path(self, attempt_id: str) -> Path:
        return self.root / f"{attempt_id}.unresolved.json"

    def _claim_path(self, trace_id: str) -> Path:
        return self.trace_root / f".{trace_id}.attempt_claim.json"


class TaskCheckpointStore:
    """One durable knowledge-boundary rollback point for crash-safe train resume.

    The checkpoint is outside ``data_v3`` so it cannot affect the knowledge
    digest.  It is created after the task enters ``running`` and deleted only
    after the task result reaches ``completed``.  A later process can therefore
    distinguish a completed commit from a partial knowledge mutation.  The
    formal runner also uses a reserved non-task boundary id around the final
    configured-batch maintenance pass.
    """

    def __init__(self, root: str | Path, data_dir: str | Path) -> None:
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir).resolve()
        if (
            self.root == self.data_dir
            or self.root in self.data_dir.parents
            or self.data_dir in self.root.parents
        ):
            raise ValueError("task checkpoint must be outside the knowledge data_dir")
        self._cleanup_tombstones()

    @property
    def manifest_path(self) -> Path:
        return self.root / "checkpoint_manifest.json"

    def create(
        self,
        database: Any,
        *,
        run_id: str,
        task_id: str,
        before_digest: str,
        config_hash: str,
        code_commit: str,
    ) -> None:
        if self.root.exists():
            raise ProtocolError(f"unresolved task checkpoint already exists: {self.root}")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{self.root.name}.tmp-", dir=self.root.parent
        ))
        try:
            checkpoint_database = temporary / "state.sqlite3"
            target = sqlite3.connect(checkpoint_database)
            try:
                getattr(database, "connection", database).backup(target)
            finally:
                target.close()
            for name in ("artifacts", "failure_knowledge"):
                source_root = self.data_dir / name
                checkpoint_root = temporary / name
                if source_root.is_dir():
                    shutil.copytree(source_root, checkpoint_root)
                else:
                    checkpoint_root.mkdir()
            files = _file_hashes(temporary, exclude={"checkpoint_manifest.json"})
            _atomic_write_json(temporary / "checkpoint_manifest.json", {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "task_id": task_id,
                "before_digest": before_digest,
                "config_hash": config_hash,
                "code_commit": code_commit,
                "data_dir": str(self.data_dir),
                "files": files,
            })
            os.replace(temporary, self.root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def recover_if_present(
        self,
        *,
        run_id: str,
        config_hash: str,
        code_commit: str,
        resume: bool,
    ) -> str | None:
        if not self.root.exists():
            return None
        if not resume:
            raise ProtocolError(
                f"task checkpoint exists for an interrupted run; use --resume: {self.root}"
            )
        if not self.manifest_path.is_file():
            raise ProtocolError(f"task checkpoint manifest missing: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "config_hash": config_hash,
            "code_commit": code_commit,
            "data_dir": str(self.data_dir),
        }
        mismatches = [
            FieldMismatch(f"checkpoint.{key}", wanted, payload.get(key))
            for key, wanted in expected.items()
            if payload.get(key) != wanted
        ]
        if mismatches:
            raise ManifestMismatchError(mismatches)
        declared_files = payload.get("files")
        actual_files = _file_hashes(self.root, exclude={"checkpoint_manifest.json"})
        if declared_files != actual_files:
            raise ProtocolError("task checkpoint file hashes do not match its manifest")
        task_id = str(payload.get("task_id", ""))
        if not task_id or not str(payload.get("before_digest", "")):
            raise ProtocolError("task checkpoint lacks task identity or knowledge digest")

        live_database = self.data_dir / "state.sqlite3"
        if live_database.is_file() and _live_task_completed(live_database, run_id, task_id):
            self.clear()
            return None
        self._restore()
        return task_id

    def _restore(self) -> None:
        checkpoint_database = self.root / "state.sqlite3"
        checkpoint_artifacts = self.root / "artifacts"
        checkpoint_failure_knowledge = self.root / "failure_knowledge"
        if (
            not checkpoint_database.is_file()
            or not checkpoint_artifacts.is_dir()
            or not checkpoint_failure_knowledge.is_dir()
        ):
            raise ProtocolError("task checkpoint is incomplete")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        restore_roots: dict[str, Path] = {}
        for name, checkpoint_root in (
            ("artifacts", checkpoint_artifacts),
            ("failure_knowledge", checkpoint_failure_knowledge),
        ):
            restore_root = Path(tempfile.mkdtemp(
                prefix=f".{name}.restore-", dir=self.data_dir
            ))
            # mkdtemp creates the root; copy checkpoint contents into it.
            shutil.copytree(checkpoint_root, restore_root, dirs_exist_ok=True)
            restore_roots[name] = restore_root
        restore_database = self.data_dir / f".state.restore-{os.getpid()}.sqlite3"
        shutil.copy2(checkpoint_database, restore_database)
        displaced_roots: dict[str, Path] = {}
        try:
            for name in ("artifacts", "failure_knowledge"):
                live_root = self.data_dir / name
                displaced = self.data_dir / f".{name}.displaced-{os.getpid()}"
                if displaced.exists():
                    shutil.rmtree(displaced)
                if live_root.exists():
                    os.replace(live_root, displaced)
                    displaced_roots[name] = displaced
                os.replace(restore_roots[name], live_root)
            for suffix in ("-wal", "-shm"):
                (self.data_dir / f"state.sqlite3{suffix}").unlink(missing_ok=True)
            os.replace(restore_database, self.data_dir / "state.sqlite3")
            for displaced in displaced_roots.values():
                shutil.rmtree(displaced, ignore_errors=True)
            self.clear()
        except Exception:
            # Keep the immutable checkpoint.  Recovery is idempotent and can
            # be attempted again after the external filesystem issue is fixed.
            restore_database.unlink(missing_ok=True)
            for restore_root in restore_roots.values():
                shutil.rmtree(restore_root, ignore_errors=True)
            raise

    def clear(self) -> None:
        if self.root.exists():
            tombstone = self.root.with_name(
                f".{self.root.name}.cleared-{os.getpid()}-{time.time_ns()}"
            )
            os.replace(self.root, tombstone)
            shutil.rmtree(tombstone, ignore_errors=True)

    def _cleanup_tombstones(self) -> None:
        prefix = f".{self.root.name}.cleared-"
        if not self.root.parent.exists():
            return
        for item in self.root.parent.iterdir():
            if item.is_dir() and item.name.startswith(prefix):
                shutil.rmtree(item, ignore_errors=True)


RunProtocol = ManifestStore


def _knowledge_table_rows(connection: Any) -> dict[str, list[list[Any]]]:
    specs = {
        "metadata": ("key,value", "key"),
        "artifact_index": (
            "artifact_ref,artifact_kind,logical_id,version,content_hash,status,schema_version",
            "artifact_ref",
        ),
        "recommended_pointers": ("logical_id,artifact_ref", "logical_id"),
        "graph_edges": (
            "edge_id,source_ref,target_ref,relation,metadata_json",
            "edge_id",
        ),
        "evidence_events": (
            "event_id,schema_version,task_id,trace_id,occurrence_id,attempt_id,"
            "sequence_no,artifact_ref,artifact_kind,event_type,failure_layer,confidence,metadata_json",
            "event_id",
        ),
        "lifecycle_projection": (
            "artifact_ref,projection_json,last_event_rowid",
            "artifact_ref",
        ),
        "projection_checkpoints": (
            "projection_name,last_event_rowid",
            "projection_name",
        ),
        "provisional_artifacts": (
            "provisional_ref,contract_signature,canonical_intent,status,"
            "harness_profile,content_hash,source_trace_id,source_task_id,"
            "promoted_refs_json,schema_version,created_at,updated_at",
            "provisional_ref",
        ),
        "failure_experiences": (
            "experience_id,cluster_signature,divergence_signature,status,"
            "harness_profile,content_hash,support_count,resolved_count,"
            "schema_version,created_at,updated_at",
            "experience_id",
        ),
        "cold_start_evidence": (
            "event_id,task_id,trace_id,subject_ref,subject_kind,event_type,"
            "sequence_no,metadata_json",
            "event_id",
        ),
    }
    existing = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    result: dict[str, list[list[Any]]] = {}
    for table, (columns, order) in specs.items():
        if table not in existing:
            continue
        rows = connection.execute(f"SELECT {columns} FROM {table} ORDER BY {order}").fetchall()
        result[table] = [[_to_primitive(value) for value in row] for row in rows]
    return result


def _existing_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value if value.is_file() else None
    if isinstance(value, str):
        try:
            path = Path(value)
            return path if path.is_file() else None
        except OSError:
            return None
    return None


def _file_hashes(root: Path, *, exclude: set[str]) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in exclude
    ]


def _live_task_completed(database_path: Path, run_id: str, task_id: str) -> bool:
    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?", (run_id, task_id)
        ).fetchone()
        return row is not None and str(row[0]) == TaskState.COMPLETED.value
    finally:
        connection.close()


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _to_primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_to_primitive(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=_canonical_json)
        return items
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values cannot enter a manifest or digest")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_metadata(value: str | Mapping[str, Any]) -> str:
    parsed = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(parsed, dict):
        raise ValueError("manifest metadata must be a JSON object")
    return _canonical_json(parsed)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        _to_primitive(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))


__all__ = [
    "ALFWORLD_FORMAL_TASK_TYPES",
    "DEEPSEEK_FORMAL_BASE_URL",
    "DEEPSEEK_FORMAL_DIALECT",
    "DEEPSEEK_FORMAL_MODEL",
    "AttemptTraceLedger",
    "AttemptTraceRef",
    "audit_failed_attempt",
    "artifact_audit_snapshot",
    "artifact_growth_audit",
    "load_task_report_traces",
    "FieldMismatch",
    "ManifestExistsError",
    "ManifestMismatchError",
    "ManifestStore",
    "ProtocolError",
    "RunManifest",
    "RunProtocol",
    "RunState",
    "TaskManifest",
    "TaskCheckpointStore",
    "TaskState",
    "code_digest",
    "compare_manifests",
    "config_digest",
    "hash_code",
    "hash_config",
    "hash_knowledge",
    "hash_task_manifest",
    "ensure_task_manifest",
    "knowledge_digest",
    "sha256_json",
    "sanitize_error_text",
    "task_signature",
    "validate_deepseek_formal_llm",
    "validate_distinct_formal_tasks",
    "write_failure_receipt",
]
