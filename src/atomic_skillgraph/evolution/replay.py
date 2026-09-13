"""Replay-case diagnostics and fail-closed source-task authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.refs import content_hash
from ..harness.protocol import HarnessTask


def replay_case_id(case: Mapping[str, Any]) -> str:
    """Return a stable identity for one immutable replay evidence case."""

    declared = str(case.get("case_id", "")).strip()
    if declared:
        return declared
    payload = {str(key): value for key, value in case.items() if key != "case_id"}
    return f"replay_case_{content_hash(payload)[:24]}"


@dataclass(frozen=True)
class ReplayCaseResult:
    case_id: str
    source_trace_id: str
    source_task_id: str
    requested_task_id: str
    resolved_task_id: str
    stage: str
    passed: bool
    failure_code: str = ""
    message: str = ""
    started: bool = False
    executed_action_count: int = 0
    completed: bool = False
    terminal_interrupted: bool = False
    atomic_effect_passed: bool = False
    output_validation_passed: bool = False

    def __bool__(self) -> bool:
        raise TypeError("ReplayCaseResult must be checked through its .passed field")


class ReplaySourceAuthorityError(AtomicSkillGraphError):
    """A replay case cannot be tied to an allowed immutable source task."""

    def __init__(self, result: ReplayCaseResult) -> None:
        super().__init__(
            result.failure_code or "replay_source_authority_error",
            result.message or "replay source authority failed",
            layer=FailureLayer.INFRASTRUCTURE,
        )
        self.result = result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _record_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "task_id": str(getattr(value, "task_id", "")),
        "task_signature": str(getattr(value, "task_signature", "")),
        "goal": str(getattr(value, "goal", "")),
        "benchmark": str(getattr(value, "benchmark", "")),
        "task_type": str(getattr(value, "task_type", "")),
        "metadata": _mapping(getattr(value, "metadata", {})),
    }


class ReplaySourceAuthority:
    """Resolve each replay case only in its verified source task world."""

    def __init__(
        self,
        trace_store: Any,
        *,
        allowed_split: str = "",
        task_manifest_path: str | Path | None = None,
    ) -> None:
        self.trace_store = trace_store
        self.allowed_split = str(allowed_split or "")
        self.task_manifest_path = (
            Path(task_manifest_path) if task_manifest_path else None
        )
        self._manifest_tasks: dict[str, dict[str, Any]] | None = None

    def _failure(
        self,
        case: Mapping[str, Any],
        *,
        requested_task_id: str,
        code: str,
        message: str,
        resolved_task_id: str = "",
    ) -> ReplaySourceAuthorityError:
        source = _mapping(case.get("source_task"))
        return ReplaySourceAuthorityError(ReplayCaseResult(
            case_id=replay_case_id(case),
            source_trace_id=str(case.get("trace_id", "")),
            source_task_id=str(source.get("task_id", "")),
            requested_task_id=str(requested_task_id),
            resolved_task_id=str(resolved_task_id),
            stage="source_resolution",
            passed=False,
            failure_code=code,
            message=message,
        ))

    def _manifest(self, case: Mapping[str, Any], requested_task_id: str) -> dict[str, dict[str, Any]]:
        if self.task_manifest_path is None:
            return {}
        if self._manifest_tasks is not None:
            return self._manifest_tasks
        if not self.task_manifest_path.is_file():
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                code="replay_source_manifest_missing",
                message=f"replay task manifest is missing: {self.task_manifest_path}",
            )
        try:
            payload = json.loads(self.task_manifest_path.read_text(encoding="utf-8"))
            rows = payload.get("tasks")
            if not isinstance(rows, list):
                raise ValueError("tasks must be a list")
            by_id: dict[str, dict[str, Any]] = {}
            for raw in rows:
                if not isinstance(raw, Mapping):
                    raise ValueError("task entry must be a mapping")
                row = dict(raw)
                task_id = str(row.get("task_id", ""))
                if not task_id or task_id in by_id:
                    raise ValueError("task ids must be non-empty and unique")
                by_id[task_id] = row
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                code="replay_source_manifest_unreadable",
                message=f"replay task manifest is unreadable: {type(exc).__name__}",
            ) from exc
        self._manifest_tasks = by_id
        return by_id

    @staticmethod
    def _compare_source_to_record(
        source: Mapping[str, Any], record: Mapping[str, Any],
    ) -> str:
        for field in ("task_id", "task_signature", "goal", "benchmark", "task_type"):
            if str(source.get(field, "")) != str(record.get(field, "")):
                return field
        source_context = _mapping(source.get("context"))
        record_metadata = _mapping(record.get("metadata"))
        source_metadata = _mapping(source.get("metadata"))
        for field in ("env_index", "game_file"):
            expected = record_metadata.get(field)
            supplied = source_context.get(field)
            if expected not in {None, ""} and supplied != expected:
                return f"context.{field}"
        for field in ("task_signature", "env_index", "game_file", "split"):
            expected = record_metadata.get(field)
            supplied = source_metadata.get(field)
            if expected not in {None, ""} and supplied not in {None, "", expected}:
                return f"metadata.{field}"
        return ""

    def _check_manifest(
        self,
        case: Mapping[str, Any],
        record: Mapping[str, Any],
        *,
        requested_task_id: str,
    ) -> str:
        rows = self._manifest(case, requested_task_id)
        if not rows:
            return ""
        task_id = str(record.get("task_id", ""))
        manifest = rows.get(task_id)
        if manifest is None:
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                resolved_task_id=task_id,
                code="replay_source_task_not_in_manifest",
                message=f"replay source task {task_id!r} is not in the run manifest",
            )
        metadata = _mapping(record.get("metadata"))
        source = _mapping(case.get("source_task"))
        source_context = _mapping(source.get("context"))
        source_metadata = _mapping(source.get("metadata"))
        manifest_metadata = _mapping(manifest.get("metadata"))
        record_env_index = metadata.get(
            "env_index", source_context.get("env_index")
        )
        record_game_file = str(
            metadata.get("game_file")
            or source_context.get("game_file")
            or ""
        )
        record_split = str(
            metadata.get("split")
            or source_metadata.get("split")
            or ""
        )
        comparisons = {
            "task_signature": (
                str(record.get("task_signature", "")),
                str(manifest.get("task_signature", "")),
            ),
            "benchmark": (
                str(record.get("benchmark", "")),
                str(manifest.get("benchmark", "")),
            ),
            "task_type": (
                str(record.get("task_type", "")),
                str(manifest_metadata.get("task_type", "")),
            ),
            "env_index": (
                record_env_index,
                manifest_metadata.get("env_index"),
            ),
            "game_file": (
                record_game_file,
                str(manifest_metadata.get("game_file", "")),
            ),
        }
        conflict = next(
            (
                field
                for field, (left, right) in comparisons.items()
                if left not in {None, ""}
                and right not in {None, ""}
                and left != right
            ),
            "",
        )
        manifest_split = str(manifest.get("split", ""))
        if (
            not conflict
            and record_split
            and manifest_split
            and manifest_split != record_split
        ):
            conflict = "split"
        if conflict:
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                resolved_task_id=task_id,
                code="replay_source_manifest_conflict",
                message=f"replay source conflicts with manifest field {conflict}",
            )
        return manifest_split

    def resolve(
        self,
        case: Mapping[str, Any],
        *,
        current_task: HarnessTask | None = None,
        current_trace: Any | None = None,
    ) -> HarnessTask:
        requested_task_id = str(
            getattr(current_task, "task_id", "") if current_task is not None else ""
        )
        source = _mapping(case.get("source_task"))
        trace_id = str(case.get("trace_id", "")).strip()
        required = ("task_id", "task_signature", "goal", "benchmark", "task_type")
        missing = [field for field in required if not str(source.get(field, "")).strip()]
        if not trace_id:
            missing.append("trace_id")
        if missing:
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                code="replay_source_identity_missing",
                message="replay source identity is incomplete: " + ", ".join(sorted(set(missing))),
            )

        current_trace_id = str(getattr(current_trace, "trace_id", ""))
        source_task_id = str(source["task_id"])
        is_current_trace = bool(current_trace_id and trace_id == current_trace_id)
        if is_current_trace:
            if current_task is None or source_task_id != requested_task_id:
                raise self._failure(
                    case,
                    requested_task_id=requested_task_id,
                    code="replay_source_current_trace_conflict",
                    message="current replay Trace and source task identity conflict",
                )
            record = _record_payload(getattr(current_trace, "task", {}))
        else:
            if current_task is not None and source_task_id == requested_task_id:
                raise self._failure(
                    case,
                    requested_task_id=requested_task_id,
                    code="replay_source_current_trace_conflict",
                    message="current task replay case names a different source Trace",
                )
            try:
                exists = bool(self.trace_store.exists(trace_id))
            except (OSError, RuntimeError) as exc:
                raise self._failure(
                    case,
                    requested_task_id=requested_task_id,
                    code="replay_source_trace_unreadable",
                    message=f"cannot inspect replay source Trace: {type(exc).__name__}",
                ) from exc
            if not exists:
                raise self._failure(
                    case,
                    requested_task_id=requested_task_id,
                    code="replay_source_trace_missing",
                    message=f"immutable replay source Trace is missing: {trace_id}",
                )
            try:
                payload = self.trace_store.load_payload(trace_id)
                record = _record_payload(_mapping(payload).get("task"))
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise self._failure(
                    case,
                    requested_task_id=requested_task_id,
                    code="replay_source_trace_unreadable",
                    message=f"immutable replay source Trace is unreadable: {type(exc).__name__}",
                ) from exc

        conflict = self._compare_source_to_record(source, record)
        if conflict:
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                resolved_task_id=str(record.get("task_id", "")),
                code="replay_source_task_conflict",
                message=f"replay source conflicts with immutable Trace field {conflict}",
            )
        record_metadata = _mapping(record.get("metadata"))
        manifest_split = self._check_manifest(
            case, record, requested_task_id=requested_task_id,
        )
        source_split = str(
            record_metadata.get("split")
            or _mapping(source.get("metadata")).get("split")
            or manifest_split
            or ""
        )
        if self.allowed_split and source_split != self.allowed_split:
            raise self._failure(
                case,
                requested_task_id=requested_task_id,
                resolved_task_id=source_task_id,
                code="replay_source_split_disallowed",
                message=(
                    f"replay source split {source_split!r} is not allowed in "
                    f"the current {self.allowed_split!r} protocol"
                ),
            )
        context = _mapping(source.get("context"))
        for field in ("env_index", "game_file"):
            if record_metadata.get(field) not in {None, ""}:
                context[field] = record_metadata[field]
        metadata = {**_mapping(source.get("metadata")), **record_metadata}
        if source_split:
            metadata["split"] = source_split
        return HarnessTask(
            task_id=str(record["task_id"]),
            goal=str(record["goal"]),
            benchmark=str(record["benchmark"]),
            task_type=str(record["task_type"]),
            context=context,
            metadata=metadata,
        )


__all__ = [
    "ReplayCaseResult",
    "ReplaySourceAuthority",
    "ReplaySourceAuthorityError",
    "replay_case_id",
]
