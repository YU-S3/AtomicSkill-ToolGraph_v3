"""Formal-safe epoch-boundary resume preparation for baseline workers.

The controller always creates a new attempt directory.  This module validates
the failed parent run, selects the last complete epoch, and materialises only
the state that pinned SkillOpt is allowed to consume.  ``resume.json`` is
written last and is the commit marker for the prepared checkpoint.

``expected_policy`` is a projection of top-level ``run_manifest.json`` fields.
Every named field must exist and its complete JSON value must match exactly.
This lets a campaign bind its concurrency and provider-retry policy without
making this common module depend on one scheduler's manifest layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifact_digest import digest_directory
from .manifest import sha256_json


RESUME_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_EXISTING_DESTINATION_FILES = frozenset({
    "environment_provenance.json",
    "provider_calls.jsonl",
    "worker_wire.json",
})
_CHECKPOINT_TARGETS = frozenset({
    "best_skill.md",
    "history.json",
    "meta_skill",
    "resume.json",
    "runtime_state.json",
    "skills",
    "slow_update",
    "steps",
})


class ResumeError(RuntimeError):
    """Base class for a rejected or uncommitted formal resume."""


class ResumeEligibilityError(ResumeError):
    """The parent run did not fail at a resumable infrastructure boundary."""


class ResumeIdentityError(ResumeError):
    """The parent run does not have the expected immutable identity/policy."""


class ResumeCheckpointError(ResumeError):
    """The parent SkillOpt checkpoint is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class PreparedEpochResume:
    """Identity and boundary of one committed resume preparation."""

    source_run: Path
    destination_train_dir: Path
    parent_run_id: str
    parent_run_digest: str
    source_last_completed_step: int
    checkpoint_step: int
    resume_from_step: int
    resume_from_epoch: int
    best_step: int
    copied_episode_cache_dirs: tuple[str, ...]
    resume_path: Path


def prepare_epoch_boundary_resume(
    *,
    source_run: str | Path,
    destination_train_dir: str | Path,
    expected_method: str,
    expected_seed: int,
    expected_identity: Mapping[str, Any],
    expected_policy: Mapping[str, Any],
    expected_initial_skill_sha256: str,
    steps_per_epoch: int,
    copy_episode_cache: bool = False,
) -> PreparedEpochResume:
    """Prepare a new SkillOpt train directory from a failed parent attempt.

    The parent must be a formal ``train`` run whose controller state and
    failure record both say ``infrastructure_failure``.  Identity is compared
    exactly; policy fields supplied by the caller are compared as exact
    top-level manifest values.  The source is never modified.

    The destination may be absent, empty, or contain only worker protocol
    bootstrap files.  Existing checkpoint state is never overwritten.
    """

    source_root = Path(source_run).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ResumeEligibilityError(f"resume source is not a directory: {source_root}")
    destination = Path(destination_train_dir).expanduser().resolve()
    if source_root == destination or _is_relative_to(destination, source_root):
        raise ResumeCheckpointError("resume destination must be outside the source run")

    method = str(expected_method).strip()
    if not method:
        raise ValueError("expected_method must be non-empty")
    if isinstance(expected_seed, bool):
        raise TypeError("expected_seed must be an integer")
    seed = int(expected_seed)
    if isinstance(steps_per_epoch, bool) or int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be a positive integer")
    steps_per_epoch = int(steps_per_epoch)
    identity = _json_mapping(expected_identity, "expected_identity")
    policy = _json_mapping(expected_policy, "expected_policy")
    if not identity:
        raise ValueError("expected_identity must be non-empty")
    if not policy:
        raise ValueError("expected_policy must be non-empty")
    initial_digest = _sha256(expected_initial_skill_sha256, "expected initial skill")

    run_manifest = _read_json_object(source_root / "run_manifest.json", "run manifest")
    run_state = _read_json_object(source_root / "run_state.json", "run state")
    failure = _read_json_object(source_root / "failure.json", "failure record")
    if (source_root / "completion.json").exists():
        raise ResumeEligibilityError("failed resume source must not contain completion.json")
    _validate_parent_eligibility(
        run_manifest=run_manifest,
        run_state=run_state,
        failure=failure,
        expected_method=method,
        expected_seed=seed,
    )
    _validate_parent_identity(
        run_manifest=run_manifest,
        expected_identity=identity,
        expected_policy=policy,
        expected_initial_skill_sha256=initial_digest,
    )

    source_train = source_root / "train"
    if not source_train.is_dir():
        raise ResumeCheckpointError(
            f"resume source has no SkillOpt train directory: {source_train}"
        )
    config = _load_and_validate_formal_config(
        source_root=source_root,
        expected_config_digest=str(identity.get("config_digest", "")),
        steps_per_epoch=steps_per_epoch,
    )
    runtime_path = source_train / "runtime_state.json"
    history_path = source_train / "history.json"
    runtime = (
        _read_json_object(runtime_path, "runtime state")
        if runtime_path.exists()
        else {}
    )
    history_value = (
        _read_json(history_path, "SkillOpt history")
        if history_path.exists()
        else []
    )
    if not isinstance(history_value, list):
        raise ResumeCheckpointError("SkillOpt history root must be a list")
    history = [_history_row(row, index) for index, row in enumerate(history_value)]

    if runtime:
        last_completed = _non_negative_int(
            runtime.get("last_completed_step"), "runtime_state.last_completed_step"
        )
    else:
        if history:
            raise ResumeCheckpointError(
                "SkillOpt history exists without runtime_state at checkpoint step 0"
            )
        last_completed = 0
    total_steps = int(config["train"]["num_epochs"]) * steps_per_epoch
    if last_completed > total_steps:
        raise ResumeCheckpointError(
            f"last_completed_step {last_completed} exceeds total steps {total_steps}"
        )
    _validate_history(history, last_completed=last_completed, steps_per_epoch=steps_per_epoch)
    checkpoint = _formal_checkpoint_step(
        source_train=source_train,
        last_completed_step=last_completed,
        steps_per_epoch=steps_per_epoch,
        optimizer=config["optimizer"],
    )
    completed_epochs = checkpoint // steps_per_epoch
    selected_history = [dict(row) for row in history if int(row["step"]) <= checkpoint]
    boundary_state = _boundary_runtime_state(
        source_train=source_train,
        source_runtime=runtime,
        selected_history=selected_history,
        checkpoint_step=checkpoint,
        source_last_completed_step=last_completed,
    )

    source_run_id = str(run_manifest.get("run_id", "")).strip()
    parent_digest = digest_directory(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=True)
    _validate_destination(destination)

    stage = Path(tempfile.mkdtemp(prefix=".resume_stage_", dir=str(destination.parent)))
    copied_cache_dirs: list[str] = []
    try:
        skills_inventory = _stage_skills(
            source_train=source_train,
            stage=stage,
            checkpoint_step=checkpoint,
            expected_initial_skill_sha256=initial_digest,
        )
        _write_json_exclusive(stage / "history.json", selected_history)

        meta_inventory = _stage_epoch_state(
            source_train=source_train,
            stage=stage,
            root_name="meta_skill",
            marker_name="meta_skill_result.json",
            completed_epochs=completed_epochs,
            required=bool(config["optimizer"].get("use_meta_skill", False)),
        )
        slow_inventory = _stage_epoch_state(
            source_train=source_train,
            stage=stage,
            root_name="slow_update",
            marker_name="slow_result.json",
            completed_epochs=completed_epochs,
            required=bool(config["optimizer"].get("use_slow_update", False)),
        )
        if copy_episode_cache:
            copied_cache_dirs = _stage_post_checkpoint_episode_caches(
                source_train=source_train,
                stage=stage,
                checkpoint_step=checkpoint,
                resume_epoch_last_step=min(checkpoint + steps_per_epoch, total_steps),
            )

        best_step = int(boundary_state["best_step"])
        best_source = stage / "skills" / f"skill_v{best_step:04d}.md"
        _copy_file_exclusive(best_source, stage / "best_skill.md")
        runtime_payload = {
            "last_completed_step": checkpoint,
            "current_skill_path": str(
                (destination / "skills" / f"skill_v{checkpoint:04d}.md").resolve()
            ),
            "current_score": boundary_state["current_score"],
            "current_origin": boundary_state["current_origin"],
            "best_skill_path": str((destination / "best_skill.md").resolve()),
            "best_score": boundary_state["best_score"],
            "best_step": best_step,
            "best_origin": boundary_state["best_origin"],
        }
        _write_json_exclusive(stage / "runtime_state.json", runtime_payload)

        cache_inventory = {
            relative: digest_directory(stage / Path(relative))
            for relative in copied_cache_dirs
        }
        resume_payload = {
            "schema_version": RESUME_SCHEMA_VERSION,
            "status": "prepared",
            "method": method,
            "seed": seed,
            "parent_run_id": source_run_id,
            "parent_run": str(source_root),
            "parent_run_digest": parent_digest,
            "source_last_completed_step": last_completed,
            "checkpoint_step": checkpoint,
            "checkpoint_epoch": completed_epochs,
            "resume_from_step": checkpoint + 1,
            "resume_from_epoch": completed_epochs + 1,
            "steps_per_epoch": steps_per_epoch,
            "expected_identity": identity,
            "expected_identity_sha256": sha256_json(identity),
            "expected_policy": policy,
            "expected_policy_sha256": sha256_json(policy),
            "inventory": {
                "skills": skills_inventory,
                "history.json": _sha256_file(stage / "history.json"),
                "best_skill.md": _sha256_file(stage / "best_skill.md"),
                "runtime_state.json": _sha256_file(stage / "runtime_state.json"),
                "meta_skill": meta_inventory,
                "slow_update": slow_inventory,
                "episode_cache_dirs": cache_inventory,
            },
        }
        _write_json_exclusive(stage / "resume.json", resume_payload)
        _commit_stage(stage=stage, destination=destination, resume_payload=resume_payload)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return PreparedEpochResume(
        source_run=source_root,
        destination_train_dir=destination,
        parent_run_id=source_run_id,
        parent_run_digest=parent_digest,
        source_last_completed_step=last_completed,
        checkpoint_step=checkpoint,
        resume_from_step=checkpoint + 1,
        resume_from_epoch=completed_epochs + 1,
        best_step=int(boundary_state["best_step"]),
        copied_episode_cache_dirs=tuple(copied_cache_dirs),
        resume_path=destination / "resume.json",
    )


def _validate_parent_eligibility(
    *,
    run_manifest: dict[str, Any],
    run_state: dict[str, Any],
    failure: dict[str, Any],
    expected_method: str,
    expected_seed: int,
) -> None:
    if run_manifest.get("formal") is not True:
        raise ResumeEligibilityError("resume source is not a formal run")
    if run_manifest.get("phase") != "train":
        raise ResumeEligibilityError("resume source run_manifest phase is not train")
    if run_state.get("phase") != "train" or failure.get("phase") != "train":
        raise ResumeEligibilityError("resume source failure phase is not train")
    if run_state.get("state") != "failed" or failure.get("passed") is not False:
        raise ResumeEligibilityError("resume source is not a failed run")
    kinds = {run_state.get("failure_kind"), failure.get("failure_kind")}
    if kinds != {"infrastructure_failure"}:
        raise ResumeEligibilityError(
            "resume source is not proven as an infrastructure_failure"
        )
    if run_manifest.get("method") != expected_method or failure.get("method") != expected_method:
        raise ResumeIdentityError("resume source method does not match")
    if run_manifest.get("run_seed") != expected_seed:
        raise ResumeIdentityError("resume source seed does not match")
    run_id = str(run_manifest.get("run_id", "")).strip()
    if not run_id or run_state.get("run_id") != run_id or failure.get("run_id") != run_id:
        raise ResumeIdentityError("resume source run_id is missing or inconsistent")
def _validate_parent_identity(
    *,
    run_manifest: dict[str, Any],
    expected_identity: dict[str, Any],
    expected_policy: dict[str, Any],
    expected_initial_skill_sha256: str,
) -> None:
    actual_identity = run_manifest.get("identity")
    if actual_identity != expected_identity:
        raise ResumeIdentityError("resume source immutable identity does not match")
    for field, expected in expected_policy.items():
        if field not in run_manifest or run_manifest[field] != expected:
            raise ResumeIdentityError(
                f"resume source policy field {field!r} does not match"
            )
    if run_manifest.get("initial_skill_sha256") != expected_initial_skill_sha256:
        raise ResumeIdentityError("resume source initial skill digest does not match")


def _load_and_validate_formal_config(
    *,
    source_root: Path,
    expected_config_digest: str,
    steps_per_epoch: int,
) -> dict[str, Any]:
    config = _read_json_object(source_root / "config_resolved.json", "resolved config")
    if not _SHA256_RE.fullmatch(expected_config_digest):
        raise ResumeIdentityError("expected identity has no valid config_digest")
    if sha256_json(config) != expected_config_digest:
        raise ResumeIdentityError("source resolved config digest does not match identity")
    if config.get("protocol_profile") != "formal_v2":
        raise ResumeEligibilityError("resume source config is not formal_v2")
    train = config.get("train")
    optimizer = config.get("optimizer")
    if not isinstance(train, dict) or not isinstance(optimizer, dict):
        raise ResumeCheckpointError("resolved config lacks train/optimizer mappings")
    train_size = _positive_int(train.get("train_size"), "train.train_size")
    batch_size = _positive_int(train.get("batch_size"), "train.batch_size")
    accumulation = _positive_int(train.get("accumulation"), "train.accumulation")
    num_epochs = _positive_int(train.get("num_epochs"), "train.num_epochs")
    inferred = math.ceil(train_size / (batch_size * accumulation))
    if inferred != steps_per_epoch:
        raise ResumeIdentityError(
            f"steps_per_epoch {steps_per_epoch} does not match resolved config {inferred}"
        )
    config["train"] = {**train, "num_epochs": num_epochs}
    config["optimizer"] = optimizer
    return config


def _validate_history(
    history: list[dict[str, Any]],
    *,
    last_completed: int,
    steps_per_epoch: int,
) -> None:
    steps = [_non_negative_int(row.get("step"), "history.step") for row in history]
    if steps != list(range(1, len(history) + 1)):
        raise ResumeCheckpointError("SkillOpt history steps are not contiguous from one")
    if len(history) < last_completed:
        raise ResumeCheckpointError("SkillOpt history ends before runtime_state")
    for row in history[:last_completed]:
        step = int(row["step"])
        expected_epoch = ((step - 1) // steps_per_epoch) + 1
        expected_in_epoch = (step - 1) % steps_per_epoch
        if row.get("epoch") != expected_epoch or row.get("step_in_epoch") != expected_in_epoch:
            raise ResumeCheckpointError(
                f"history step {step} has inconsistent epoch coordinates"
            )


def _formal_checkpoint_step(
    *,
    source_train: Path,
    last_completed_step: int,
    steps_per_epoch: int,
    optimizer: Mapping[str, Any],
) -> int:
    """Select a boundary whose epoch-final state is actually committed.

    SkillOpt advances ``last_completed_step`` before the slow-update and
    meta-skill epoch-finalizers have necessarily committed their markers.  A
    provider failure in either finalizer can therefore leave (for example)
    ``last_completed_step == 12`` while epoch 4 is not a formal-safe resume
    boundary.  Only that just-finished epoch can legitimately be in flight;
    earlier epoch markers are still validated strictly by
    :func:`_stage_epoch_state`.
    """

    checkpoint = (last_completed_step // steps_per_epoch) * steps_per_epoch
    if checkpoint == 0 or last_completed_step != checkpoint:
        return checkpoint

    epoch = checkpoint // steps_per_epoch
    required_markers = (
        ("meta_skill", "meta_skill_result.json", bool(optimizer.get("use_meta_skill"))),
        ("slow_update", "slow_result.json", bool(optimizer.get("use_slow_update"))),
    )
    for root_name, marker_name, required in required_markers:
        if not required:
            continue
        marker = source_train / root_name / f"epoch_{epoch:02d}" / marker_name
        try:
            payload = _read_json(marker, f"{root_name} epoch {epoch} marker")
        except ResumeCheckpointError:
            return checkpoint - steps_per_epoch
        if not isinstance(payload, dict):
            return checkpoint - steps_per_epoch
    return checkpoint


def _boundary_runtime_state(
    *,
    source_train: Path,
    source_runtime: dict[str, Any],
    selected_history: list[dict[str, Any]],
    checkpoint_step: int,
    source_last_completed_step: int,
) -> dict[str, Any]:
    if checkpoint_step == 0:
        if source_last_completed_step == 0:
            if source_runtime:
                current_score = _number(
                    source_runtime.get("current_score"), "current_score"
                )
                best_score = _number(
                    source_runtime.get("best_score"), "best_score"
                )
            else:
                # A provider outage during the initial selection rollout can
                # occur before pinned SkillOpt writes runtime_state/history.
                # Step 0 is a legal boundary; -1 asks the unchanged trainer to
                # rerun its baseline selection before entering optimizer step 1.
                current_score = -1.0
                best_score = -1.0
        else:
            current_score = _initial_selection_score(source_train)
            best_score = current_score
        return {
            "current_score": current_score,
            "best_score": best_score,
            "best_step": 0,
            "current_origin": "initial_skill",
            "best_origin": "initial_skill",
        }
    if not selected_history or int(selected_history[-1]["step"]) != checkpoint_step:
        raise ResumeCheckpointError("history does not contain the checkpoint boundary")
    row = selected_history[-1]
    current_score = _number(row.get("current_score"), "checkpoint current_score")
    best_score = _number(row.get("best_score"), "checkpoint best_score")
    best_step = _non_negative_int(row.get("best_step"), "checkpoint best_step")
    if best_step > checkpoint_step:
        raise ResumeCheckpointError("checkpoint best_step is in the discarded future")
    return {
        "current_score": current_score,
        "best_score": best_score,
        "best_step": best_step,
        "current_origin": str(row.get("current_origin") or f"step_{checkpoint_step:04d}"),
        "best_origin": str(
            row.get("best_origin")
            or ("initial_skill" if best_step == 0 else f"step_{best_step:04d}")
        ),
    }


def _initial_selection_score(source_train: Path) -> float:
    results = source_train / "selection_eval_baseline" / "results.jsonl"
    if not results.is_file():
        raise ResumeCheckpointError(
            "cannot reconstruct checkpoint step 0 without baseline selection results"
        )
    scores: list[float] = []
    for line_number, line in enumerate(results.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            value = float(row["hard"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResumeCheckpointError(
                f"invalid baseline selection result at line {line_number}"
            ) from exc
        if not 0.0 <= value <= 1.0:
            raise ResumeCheckpointError("baseline selection hard score is outside 0..1")
        scores.append(value)
    if not scores:
        raise ResumeCheckpointError("baseline selection results are empty")
    return sum(scores) / len(scores)


def _stage_skills(
    *,
    source_train: Path,
    stage: Path,
    checkpoint_step: int,
    expected_initial_skill_sha256: str,
) -> dict[str, str]:
    destination = stage / "skills"
    destination.mkdir()
    inventory: dict[str, str] = {}
    for step in range(checkpoint_step + 1):
        name = f"skill_v{step:04d}.md"
        source = source_train / "skills" / name
        if not source.is_file() or source.is_symlink():
            raise ResumeCheckpointError(f"checkpoint skill is missing or unsafe: {source}")
        if source.stat().st_size <= 0:
            raise ResumeCheckpointError(f"checkpoint skill is empty: {source}")
        target = destination / name
        _copy_file_exclusive(source, target)
        inventory[name] = _sha256_file(target)
    if inventory.get("skill_v0000.md") != expected_initial_skill_sha256:
        raise ResumeIdentityError("skill_v0000.md does not match the expected initial skill")
    return inventory


def _stage_epoch_state(
    *,
    source_train: Path,
    stage: Path,
    root_name: str,
    marker_name: str,
    completed_epochs: int,
    required: bool,
) -> dict[str, str]:
    inventory: dict[str, str] = {}
    if completed_epochs <= 0:
        return inventory
    source_root = source_train / root_name
    for epoch in range(1, completed_epochs + 1):
        relative = Path(root_name) / f"epoch_{epoch:02d}"
        source = source_train / relative
        marker = source / marker_name
        if not source.is_dir() or not marker.is_file():
            if required:
                raise ResumeCheckpointError(
                    f"completed epoch {epoch} lacks {root_name}/{marker_name}"
                )
            continue
        marker_payload = _read_json(marker, f"{root_name} epoch {epoch} marker")
        if not isinstance(marker_payload, dict):
            raise ResumeCheckpointError(f"{marker} must contain a JSON object")
        target = stage / relative
        _copy_tree_no_symlinks(source, target)
        inventory[relative.as_posix()] = digest_directory(target)
    return inventory


def _stage_post_checkpoint_episode_caches(
    *,
    source_train: Path,
    stage: Path,
    checkpoint_step: int,
    resume_epoch_last_step: int,
) -> list[str]:
    steps_root = source_train / "steps"
    if not steps_root.is_dir():
        return []
    copied: list[str] = []
    for step_dir in sorted(path for path in steps_root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"step_(\d{4,})", step_dir.name)
        step = int(match.group(1)) if match is not None else -1
        if (
            match is None
            or step <= checkpoint_step
            or step > resume_epoch_last_step
        ):
            continue
        episode_roots = sorted(
            path for path in step_dir.rglob("episodes") if path.is_dir()
        )
        for source in episode_roots:
            task_dirs = sorted(path for path in source.iterdir() if path.is_dir())
            committed_task_dirs: list[Path] = []
            for task_dir in task_dirs:
                receipt = task_dir / "receipt.json"
                # receipt.json is the sole transaction commit marker.  A task
                # directory without it is an interrupted/incomplete episode,
                # so omit it and let the resumed rollout execute that task.
                if not receipt.exists():
                    continue
                payload = _read_json(receipt, "episode cache receipt")
                if (
                    not isinstance(payload, dict)
                    or payload.get("status") != "completed"
                ):
                    raise ResumeCheckpointError(
                        f"episode cache has an invalid receipt.json: {task_dir}"
                    )
                committed_task_dirs.append(task_dir)
            if not committed_task_dirs:
                continue
            relative = source.relative_to(source_train)
            target = stage / relative
            target.mkdir(parents=True, exist_ok=False)
            for task_dir in committed_task_dirs:
                _copy_tree_no_symlinks(task_dir, target / task_dir.name)
            copied.append(relative.as_posix())
    return copied


def _validate_destination(destination: Path) -> None:
    entries = {path.name for path in destination.iterdir()}
    forbidden = sorted(entries & _CHECKPOINT_TARGETS)
    unexpected = sorted(entries - _ALLOWED_EXISTING_DESTINATION_FILES)
    if forbidden or unexpected:
        raise ResumeCheckpointError(
            "resume destination is not fresh; existing entries: "
            + ", ".join(sorted(set(forbidden + unexpected)))
        )


def _commit_stage(
    *,
    stage: Path,
    destination: Path,
    resume_payload: dict[str, Any],
) -> None:
    children = sorted(
        (path for path in stage.iterdir() if path.name != "resume.json"),
        key=lambda path: path.name,
    )
    for child in children:
        target = destination / child.name
        if target.exists():
            raise ResumeCheckpointError(f"refusing to overwrite resume target: {target}")
        os.replace(child, target)
    _verify_inventory(destination, dict(resume_payload["inventory"]))
    resume_source = stage / "resume.json"
    resume_target = destination / "resume.json"
    if resume_target.exists():
        raise ResumeCheckpointError(f"refusing to overwrite resume marker: {resume_target}")
    os.replace(resume_source, resume_target)


def _verify_inventory(root: Path, inventory: dict[str, Any]) -> None:
    for name in ("history.json", "best_skill.md", "runtime_state.json"):
        path = root / name
        if not path.is_file() or _sha256_file(path) != inventory[name]:
            raise ResumeCheckpointError(f"committed resume inventory mismatch: {path}")
    for name, digest in dict(inventory["skills"]).items():
        path = root / "skills" / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise ResumeCheckpointError(f"committed resume skill mismatch: {path}")
    for group in ("meta_skill", "slow_update", "episode_cache_dirs"):
        for relative, digest in dict(inventory[group]).items():
            path = root / Path(relative)
            if not path.is_dir() or digest_directory(path) != digest:
                raise ResumeCheckpointError(f"committed resume directory mismatch: {path}")


def _copy_tree_no_symlinks(source: Path, destination: Path) -> None:
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ResumeCheckpointError(f"resume source tree contains a symlink: {source}")
    if destination.exists():
        raise ResumeCheckpointError(f"resume staging target already exists: {destination}")
    shutil.copytree(source, destination)


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ResumeCheckpointError(f"resume source file is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ResumeCheckpointError(f"resume target already exists: {destination}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ResumeCheckpointError(f"{what} is missing or unsafe: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeCheckpointError(f"{what} is unreadable: {path}") from exc


def _read_json_object(path: Path, what: str) -> dict[str, Any]:
    value = _read_json(path, what)
    if not isinstance(value, dict):
        raise ResumeCheckpointError(f"{what} root must be a JSON object: {path}")
    return value


def _history_row(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResumeCheckpointError(f"history row {index} is not a JSON object")
    return dict(value)


def _json_mapping(value: Mapping[str, Any], what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    try:
        normalized = json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must contain JSON-serializable values") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{what} must normalize to a JSON object")
    return normalized


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResumeCheckpointError(f"{what} must be a positive integer")
    return value


def _non_negative_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResumeCheckpointError(f"{what} must be a non-negative integer")
    return value


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResumeCheckpointError(f"{what} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResumeCheckpointError(f"{what} must be finite")
    return result


def _sha256(value: str, what: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{what} must be a lowercase SHA-256 digest")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
