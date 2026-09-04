"""Common Train / Validation / Test task manifests.

Schema per the baseline design document, section 5.5.  A manifest entry
deliberately carries no expert action, high-level plan, hidden object
position, or any reference material: only task identity and physical
gamefile location.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestTask:
    index: int
    task_id: str
    task_type: str
    source_split: str
    env_index: int
    gamefile_rel: str
    gamefile_sha256: str
    task_signature: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_type or not self.source_split:
            raise ValueError("ManifestTask requires non-empty identity fields")
        if not self.gamefile_rel or Path(self.gamefile_rel).is_absolute():
            raise ValueError("ManifestTask gamefile_rel must be a relative path")
        if self.index < 0 or self.env_index < 0:
            raise ValueError("ManifestTask indexes must be non-negative")
        if len(self.gamefile_sha256) != 64 or len(self.task_signature) != 64:
            raise ValueError("ManifestTask hashes must be 64-char SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManifestTask":
        if not isinstance(payload, dict):
            raise ValueError("ManifestTask payload must be a mapping")
        try:
            task = cls(
                index=int(payload["index"]),
                task_id=str(payload["task_id"]),
                task_type=str(payload["task_type"]),
                source_split=str(payload["source_split"]),
                env_index=int(payload["env_index"]),
                gamefile_rel=str(payload["gamefile_rel"]),
                gamefile_sha256=str(payload["gamefile_sha256"]),
                task_signature=str(payload["task_signature"]),
            )
        except KeyError as exc:
            raise ValueError(f"ManifestTask payload is missing {exc.args[0]}") from exc
        return task


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TaskManifestSet:
    manifest_id: str
    benchmark: str
    source_split: str
    seed: int
    tasks: tuple[ManifestTask, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "benchmark": self.benchmark,
            "source_split": self.source_split,
            "seed": self.seed,
            "digest": self.digest,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @staticmethod
    def digest_of(
        *, manifest_id: str, benchmark: str, source_split: str, seed: int,
        tasks: tuple[ManifestTask, ...],
    ) -> str:
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_id": manifest_id,
            "benchmark": benchmark,
            "source_split": source_split,
            "seed": seed,
            "tasks": [task.to_dict() for task in tasks],
        }
        return sha256_json(payload)

    @classmethod
    def create(
        cls, *, manifest_id: str, benchmark: str, source_split: str, seed: int,
        tasks: tuple[ManifestTask, ...],
    ) -> "TaskManifestSet":
        _validate_identity(tasks)
        return cls(
            manifest_id=manifest_id,
            benchmark=benchmark,
            source_split=source_split,
            seed=seed,
            tasks=tasks,
            digest=cls.digest_of(
                manifest_id=manifest_id, benchmark=benchmark,
                source_split=source_split, seed=seed, tasks=tasks,
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TaskManifestSet":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest is unreadable: {path}") from exc
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"manifest has invalid schema_version: {path}")
        try:
            tasks = tuple(ManifestTask.from_dict(item) for item in payload["tasks"])
        except KeyError as exc:
            raise ValueError(f"manifest is missing {exc.args[0]}: {path}") from exc
        _validate_identity(tasks)
        manifest = cls.create(
            manifest_id=str(payload["manifest_id"]),
            benchmark=str(payload["benchmark"]),
            source_split=str(payload["source_split"]),
            seed=int(payload["seed"]),
            tasks=tasks,
        )
        declared = str(payload.get("digest", ""))
        if declared != manifest.digest:
            raise ValueError(
                f"manifest digest mismatch: declared {declared}, computed {manifest.digest}"
            )
        return manifest

    def save(self, path: str | Path) -> Path:
        payload = self.to_dict()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target


def _validate_identity(tasks: tuple[ManifestTask, ...]) -> None:
    indexes = [task.index for task in tasks]
    if indexes != list(range(len(tasks))):
        raise ValueError("manifest task indexes must be contiguous from zero")
    if any(not task.task_id for task in tasks) or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("manifest task ids must be non-empty and unique")
    if len({task.task_signature for task in tasks}) != len(tasks):
        raise ValueError("manifest task signatures must be unique")
    if len({(task.source_split, task.gamefile_rel) for task in tasks}) != len(tasks):
        raise ValueError("manifest gamefiles must be unique")


def verify_disjoint(*sets: TaskManifestSet) -> None:
    """Fail closed if any two manifests share a physical gamefile or signature."""

    for left_index, left in enumerate(sets):
        for right in sets[left_index + 1:]:
            overlap = {task.gamefile_rel for task in left.tasks} & {
                task.gamefile_rel for task in right.tasks
            }
            if overlap:
                raise ValueError(
                    f"manifests {left.manifest_id} and {right.manifest_id} share "
                    f"gamefiles: {sorted(overlap)[:5]}"
                )
            signature_overlap = {task.task_signature for task in left.tasks} & {
                task.task_signature for task in right.tasks
            }
            if signature_overlap:
                raise ValueError(
                    f"manifests {left.manifest_id} and {right.manifest_id} share "
                    "task signatures"
                )


def verify_nesting(inner: TaskManifestSet, outer: TaskManifestSet) -> None:
    """Require ``inner ⊂ outer`` on physical gamefile identity."""

    inner_files = {task.gamefile_rel for task in inner.tasks}
    outer_files = {task.gamefile_rel for task in outer.tasks}
    if not inner_files <= outer_files:
        missing = sorted(inner_files - outer_files)[:5]
        raise ValueError(
            f"{inner.manifest_id} is not a subset of {outer.manifest_id}; "
            f"missing: {missing}"
        )
