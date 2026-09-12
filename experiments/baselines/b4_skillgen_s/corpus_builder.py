"""Deterministic SkillGen sampling jobs and extraction-barrier corpus."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet, sha256_json

from .progress_adapter import SkillGenLabel, labels_for_manifest


SAMPLE_SEED_ALGORITHM = "sha256-u32be(run_seed\\0task_id\\0sample_idx)-v1"
CHECKPOINT_INVENTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SamplingJob:
    job_index: int
    manifest_index: int
    task: ManifestTask
    sample_idx: int
    sample_seed: int
    label: SkillGenLabel

    def to_wire(self) -> dict[str, Any]:
        return {
            "job_index": self.job_index,
            "manifest_index": self.manifest_index,
            "task": self.task.to_dict(),
            "sample_idx": self.sample_idx,
            "sample_seed": self.sample_seed,
            "label": asdict(self.label),
        }


def stable_sample_seed(run_seed: int, task_id: str, sample_idx: int) -> int:
    if sample_idx < 0:
        raise ValueError("sample_idx must be non-negative")
    payload = f"{int(run_seed)}\0{task_id}\0{int(sample_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_sampling_jobs(
    manifest: TaskManifestSet,
    labels: dict[str, SkillGenLabel],
    *,
    run_seed: int,
    sampling_count: int = 6,
) -> tuple[SamplingJob, ...]:
    if manifest.source_split != "train":
        raise ValueError("SkillGen sampling accepts only a Train manifest")
    if sampling_count != 6:
        raise ValueError("SkillGen-S sampling_count is frozen at 6")
    bound = labels_for_manifest(manifest, labels)
    jobs: list[SamplingJob] = []
    for task in manifest.tasks:
        for sample_idx in range(sampling_count):
            jobs.append(SamplingJob(
                job_index=len(jobs),
                manifest_index=task.index,
                task=task,
                sample_idx=sample_idx,
                sample_seed=stable_sample_seed(run_seed, task.task_id, sample_idx),
                label=bound[task.task_id],
            ))
    return tuple(jobs)


def validate_completed_corpus(
    jobs: Iterable[SamplingJob],
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce the all-samples barrier before extraction."""

    expected = list(jobs)
    materialized = [dict(row) for row in rows]
    by_index: dict[int, dict[str, Any]] = {}
    for row in materialized:
        index = int(row.get("job_index", -1))
        if index in by_index:
            raise ValueError(f"duplicate SkillGen sampling job_index {index}")
        by_index[index] = row
    missing = [job.job_index for job in expected if job.job_index not in by_index]
    unexpected = sorted(set(by_index) - {job.job_index for job in expected})
    if missing or unexpected:
        raise ValueError(
            f"SkillGen corpus barrier incomplete: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    ordered: list[dict[str, Any]] = []
    for job in expected:
        row = by_index[job.job_index]
        identity = {
            "task_id": job.task.task_id,
            "manifest_index": job.manifest_index,
            "sample_idx": job.sample_idx,
            "sample_seed": job.sample_seed,
        }
        mismatches = [key for key, value in identity.items() if row.get(key) != value]
        if mismatches:
            raise ValueError(
                f"SkillGen sampling job {job.job_index} identity mismatch: {mismatches}"
            )
        if row.get("failure_kind") in {"infrastructure_failure", "protocol_failure"}:
            raise RuntimeError(
                f"SkillGen corpus cannot cross extraction barrier: job "
                f"{job.job_index} failed as {row.get('failure_kind')}"
            )
        required = {
            "goal", "trajectory", "grounding", "progress", "progress_rate",
            "official_success", "actions", "provider_calls",
        }
        absent = sorted(key for key in required if key not in row)
        if absent:
            raise ValueError(
                f"SkillGen sampling job {job.job_index} lacks evidence: {absent}"
            )
        ordered.append(row)
    return ordered


def corpus_metadata(
    *,
    manifest: TaskManifestSet,
    run_seed: int,
    rows: list[dict[str, Any]],
    label_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "b4_skillgen_s",
        "manifest_id": manifest.manifest_id,
        "manifest_digest": manifest.digest,
        "run_seed": int(run_seed),
        "sampling_count": 6,
        "sampling_episode_count": len(rows),
        "sample_seed_algorithm": SAMPLE_SEED_ALGORITHM,
        "label_sha256": label_sha256,
        "weak_supervision": "subgoal_progress",
        "extra_train_supervision": True,
        "corpus_digest": sha256_json(rows),
    }


def build_checkpoint_inventory(directory: str | Path) -> dict[str, Any]:
    """Bind every completed sampling-job file to immutable bytes."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"sampling checkpoint directory is missing: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sampling checkpoint must be a regular file: {path}")
        if path.suffix != ".json" or len(path.stem) != 4 or not path.stem.isdigit():
            raise ValueError(f"invalid sampling checkpoint filename: {path.name}")
        content = path.read_bytes()
        files.append({
            "name": path.name,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    core = {
        "schema_version": CHECKPOINT_INVENTORY_SCHEMA_VERSION,
        "files": files,
    }
    return {**core, "digest": sha256_json(core)}


def load_verified_checkpoint_bytes(
    directory: str | Path,
    inventory: Mapping[str, Any],
) -> dict[str, bytes]:
    """Verify an inventory and return the exact bytes that were verified."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"sampling checkpoint directory is missing: {root}")
    payload = dict(inventory)
    try:
        schema_version = payload["schema_version"]
        raw_files = payload["files"]
        declared_digest = str(payload["digest"])
    except KeyError as exc:
        raise ValueError("sampling checkpoint inventory has an invalid schema") from exc
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CHECKPOINT_INVENTORY_SCHEMA_VERSION
        or not isinstance(raw_files, list)
    ):
        raise ValueError("sampling checkpoint inventory has an invalid schema")
    core = {"schema_version": schema_version, "files": raw_files}
    if sha256_json(core) != declared_digest:
        raise ValueError("sampling checkpoint inventory digest mismatch")

    expected: dict[str, dict[str, Any]] = {}
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "name", "size_bytes", "sha256",
        }:
            raise ValueError("sampling checkpoint inventory entry is invalid")
        name = str(raw_entry["name"])
        digest = str(raw_entry["sha256"])
        size = raw_entry["size_bytes"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or Path(name).name != name
            or Path(name).suffix != ".json"
            or len(Path(name).stem) != 4
            or not Path(name).stem.isdigit()
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or size < 0
            or name in expected
        ):
            raise ValueError(f"sampling checkpoint inventory entry is invalid: {name}")
        expected[name] = {"size_bytes": size, "sha256": digest}
    if list(expected) != sorted(expected):
        raise ValueError("sampling checkpoint inventory is not canonically ordered")

    observed_paths = sorted(root.glob("*.json"), key=lambda item: item.name)
    observed_names = [path.name for path in observed_paths]
    if observed_names != list(expected):
        raise ValueError(
            "sampling checkpoint inventory membership changed: "
            f"expected={list(expected)}, observed={observed_names}"
        )
    verified: dict[str, bytes] = {}
    for path in observed_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sampling checkpoint must be a regular file: {path}")
        content = path.read_bytes()
        entry = expected[path.name]
        if len(content) != entry["size_bytes"] or hashlib.sha256(content).hexdigest() != entry[
            "sha256"
        ]:
            raise ValueError(f"sampling checkpoint digest changed: {path.name}")
        verified[path.name] = content
    return verified


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_job_result(path: str | Path, row: dict[str, Any]) -> Path:
    """Persist one durable sampling boundary without overwriting evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and fails if another writer created
        # the target after the preflight exists-check.  Both names are in the
        # same directory/filesystem, so no copy or overwrite can occur.
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target
