"""Fail-closed authority checks for formal ALFWorld baseline runs.

The immutable manifest digest proves that a manifest did not change after it
was written; it does *not* prove that its paths still name files inside the
configured ALFWorld data root or that those files still have the recorded
content.  Likewise, a list of episode sidecars is not a formal evaluation
result until it is a one-to-one rendering of the selected manifest.

This module deliberately contains no method-specific learning logic.  It is a
common controller boundary that can be reused by every baseline driver.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .manifest import ManifestTask, TaskManifestSet
from .schema import CommonEpisodeRecord


# Kept local so this minimal shared validation boundary remains importable in
# SkillGen's frozen Python 3.9 worker.  Importing experiments.protocol would
# initialize the Python >=3.10 AtomicSkillGraph package for an unrelated tuple.
ALFWORLD_FORMAL_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


@dataclass(frozen=True)
class FormalManifestSpec:
    """Frozen identity and balance requirements for one manifest role."""

    role: str
    manifest_id: str
    source_split: str
    task_count: int
    per_task_type: int | None
    final_evaluation_phase: str


_PILOT_MANIFEST_SPECS: dict[str, FormalManifestSpec] = {
    "train": FormalManifestSpec(
        role="train",
        manifest_id="train_30",
        source_split="train",
        task_count=30,
        per_task_type=5,
        final_evaluation_phase="train_eval",
    ),
    "validation": FormalManifestSpec(
        role="validation",
        manifest_id="validation_6",
        source_split="valid_seen",
        task_count=6,
        per_task_type=1,
        final_evaluation_phase="validation_eval",
    ),
    "test": FormalManifestSpec(
        role="test",
        manifest_id="test_ood_60",
        source_split="valid_unseen",
        task_count=60,
        per_task_type=10,
        final_evaluation_phase="test",
    ),
}

_FORMAL_V2_MANIFEST_SPECS: dict[str, FormalManifestSpec] = {
    "train": FormalManifestSpec(
        role="train",
        manifest_id="train_120",
        source_split="train",
        task_count=120,
        per_task_type=20,
        final_evaluation_phase="train_eval",
    ),
    "validation": FormalManifestSpec(
        role="validation",
        manifest_id="validation_24",
        source_split="valid_seen",
        task_count=24,
        per_task_type=4,
        final_evaluation_phase="validation_eval",
    ),
    "test": FormalManifestSpec(
        role="test",
        manifest_id="test_ood_full_134",
        source_split="valid_unseen",
        task_count=134,
        per_task_type=None,
        final_evaluation_phase="test",
    ),
}

_SMOKE_V1_MANIFEST_SPECS: dict[str, FormalManifestSpec] = {
    "train": FormalManifestSpec(
        role="train",
        manifest_id="train_6_smoke",
        source_split="train",
        task_count=6,
        per_task_type=1,
        final_evaluation_phase="smoke",
    ),
    "validation": FormalManifestSpec(
        role="validation",
        manifest_id="validation_6_smoke",
        source_split="valid_seen",
        task_count=6,
        per_task_type=1,
        final_evaluation_phase="validation_eval",
    ),
    "test": FormalManifestSpec(
        role="test",
        manifest_id="test_6_smoke",
        source_split="valid_unseen",
        task_count=6,
        per_task_type=1,
        final_evaluation_phase="smoke_test",
    ),
}

FORMAL_MANIFEST_PROFILES: dict[str, dict[str, FormalManifestSpec]] = {
    "pilot_v1": _PILOT_MANIFEST_SPECS,
    "smoke_v1": _SMOKE_V1_MANIFEST_SPECS,
    "formal_v2": _FORMAL_V2_MANIFEST_SPECS,
}
# Backward-compatible public name for existing pilot tests/importers.
FORMAL_MANIFEST_SPECS = _PILOT_MANIFEST_SPECS
_FORMAL_SELECTION_SEED = 42


def _spec(role: str, *, profile: str = "pilot_v1") -> FormalManifestSpec:
    try:
        specs = FORMAL_MANIFEST_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported manifest profile {profile!r}; expected one of "
            f"{sorted(FORMAL_MANIFEST_PROFILES)}"
        ) from exc
    try:
        return specs[str(role)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported formal manifest role {role!r}; expected one of "
            f"{sorted(specs)}"
        ) from exc


def _verify_manifest_structure(
    manifest: TaskManifestSet,
    *,
    role: str,
    profile: str = "pilot_v1",
) -> FormalManifestSpec:
    spec = _spec(role, profile=profile)
    if profile in {"formal_v2", "smoke_v1"} and manifest.seed != _FORMAL_SELECTION_SEED:
        raise ValueError(
            "formal manifest selection seed mismatch: expected "
            f"{_FORMAL_SELECTION_SEED}, got {manifest.seed}"
        )
    if manifest.manifest_id != spec.manifest_id:
        raise ValueError(
            f"formal {role} manifest_id mismatch: expected {spec.manifest_id!r}, "
            f"got {manifest.manifest_id!r}"
        )
    if manifest.benchmark != "alfworld":
        raise ValueError(
            f"formal {role} benchmark mismatch: expected 'alfworld', "
            f"got {manifest.benchmark!r}"
        )
    if manifest.source_split != spec.source_split:
        raise ValueError(
            f"formal {role} source_split mismatch: expected "
            f"{spec.source_split!r}, got {manifest.source_split!r}"
        )
    if len(manifest.tasks) != spec.task_count:
        raise ValueError(
            f"formal {role} task count mismatch: expected {spec.task_count}, "
            f"got {len(manifest.tasks)}"
        )

    indexes = [task.index for task in manifest.tasks]
    if indexes != list(range(spec.task_count)):
        raise ValueError(
            f"formal {role} manifest indexes must be contiguous and ordered "
            f"from 0 to {spec.task_count - 1}"
        )
    task_ids = [task.task_id for task in manifest.tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"formal {role} manifest has duplicate task_id values")
    signatures = [task.task_signature for task in manifest.tasks]
    if len(set(signatures)) != len(signatures):
        raise ValueError(f"formal {role} manifest has duplicate task signatures")
    gamefiles = [task.gamefile_rel for task in manifest.tasks]
    if len(set(gamefiles)) != len(gamefiles):
        raise ValueError(f"formal {role} manifest has duplicate gamefiles")
    if profile in {"formal_v2", "smoke_v1"} and gamefiles != sorted(gamefiles):
        raise ValueError(
            f"formal {role} manifest tasks must be stored in canonical "
            "gamefile path order"
        )

    for task in manifest.tasks:
        if task.source_split != spec.source_split:
            raise ValueError(
                f"formal {role} task {task.task_id!r} source_split mismatch: "
                f"expected {spec.source_split!r}, got {task.source_split!r}"
            )

    counts = Counter(task.task_type for task in manifest.tasks)
    if spec.per_task_type is None:
        unexpected_types = sorted(set(counts) - set(ALFWORLD_FORMAL_TASK_TYPES))
        missing_types = sorted(set(ALFWORLD_FORMAL_TASK_TYPES) - set(counts))
        if unexpected_types or missing_types:
            raise ValueError(
                f"formal {role} task families mismatch: missing={missing_types}, "
                f"unexpected={unexpected_types}"
            )
    else:
        expected_counts = {
            task_type: spec.per_task_type
            for task_type in ALFWORLD_FORMAL_TASK_TYPES
        }
        if dict(counts) != expected_counts:
            raise ValueError(
                f"formal {role} family balance mismatch: expected "
                f"{expected_counts}, got {dict(counts)}"
            )

    computed_digest = TaskManifestSet.digest_of(
        manifest_id=manifest.manifest_id,
        benchmark=manifest.benchmark,
        source_split=manifest.source_split,
        seed=manifest.seed,
        tasks=manifest.tasks,
    )
    if manifest.digest != computed_digest:
        raise ValueError(
            f"formal {role} manifest digest mismatch: declared "
            f"{manifest.digest}, computed {computed_digest}"
        )
    return spec


def _family_from_relative(relative: PurePosixPath) -> str:
    if len(relative.parts) < 4:
        return ""
    directory = relative.parts[2]
    matches = [
        task_type
        for task_type in ALFWORLD_FORMAL_TASK_TYPES
        if directory == task_type or directory.startswith(task_type + "-")
    ]
    return matches[0] if len(matches) == 1 else ""


def _verify_frozen_selection(
    manifest: TaskManifestSet,
    *,
    data_root: Path,
    spec: FormalManifestSpec,
) -> None:
    """Recompute the frozen seed-42 selection from physical gamefiles."""

    split_root = data_root / "json_2.1.1" / spec.source_split
    physical: dict[str, list[str]] = {
        task_type: [] for task_type in ALFWORLD_FORMAL_TASK_TYPES
    }
    unexpected: list[str] = []
    for gamefile in split_root.rglob("game.tw-pddl"):
        relative = PurePosixPath(gamefile.relative_to(data_root).as_posix())
        task_type = _family_from_relative(relative)
        if not task_type:
            unexpected.append(relative.as_posix())
            continue
        physical[task_type].append(relative.as_posix())
    if unexpected:
        raise ValueError(
            f"formal {spec.role} dataset contains unsupported task families: "
            f"{unexpected[:5]}"
        )
    if any(not paths for paths in physical.values()):
        missing = sorted(
            task_type for task_type, paths in physical.items() if not paths
        )
        raise ValueError(
            f"formal {spec.role} dataset is missing task families: {missing}"
        )

    if spec.per_task_type is None:
        expected = sorted(path for paths in physical.values() for path in paths)
    else:
        rng = random.Random(_FORMAL_SELECTION_SEED)
        selected: list[str] = []
        for task_type in ALFWORLD_FORMAL_TASK_TYPES:
            family = sorted(physical[task_type])
            rng.shuffle(family)
            if len(family) < spec.per_task_type:
                raise ValueError(
                    f"formal {spec.role} dataset has only {len(family)} "
                    f"{task_type} games; need {spec.per_task_type}"
                )
            selected.extend(family[:spec.per_task_type])
        expected = sorted(selected)

    observed = [task.gamefile_rel for task in manifest.tasks]
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected_paths = sorted(set(observed) - set(expected))
        raise ValueError(
            f"formal {spec.role} manifest does not match the frozen seed-42 "
            f"physical selection: missing={missing[:5]}, "
            f"unexpected={unexpected_paths[:5]}"
        )


def _resolve_gamefile(
    data_root: Path,
    task: ManifestTask,
    *,
    source_split: str,
) -> Path:
    raw = task.gamefile_rel
    # Formal manifests are serialized with portable POSIX paths.  Rejecting
    # backslashes also closes Windows drive/UNC spellings when validation is
    # executed on a POSIX worker.
    if "\\" in raw:
        raise ValueError(
            f"formal task {task.task_id!r} gamefile is not a POSIX relative path: {raw!r}"
        )
    relative = PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(
            f"formal task {task.task_id!r} gamefile path escapes or is not "
            f"normalized: {raw!r}"
        )
    expected_prefix = ("json_2.1.1", source_split)
    if relative.parts[:2] != expected_prefix or relative.name != "game.tw-pddl":
        raise ValueError(
            f"formal task {task.task_id!r} gamefile is outside the declared "
            f"{source_split!r} dataset partition: {raw!r}"
        )

    try:
        resolved = data_root.joinpath(*relative.parts).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"formal task {task.task_id!r} gamefile is missing: "
            f"{data_root.joinpath(*relative.parts)}"
        ) from exc
    try:
        resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"formal task {task.task_id!r} gamefile resolves outside "
            f"ALFWORLD_DATA: {raw!r}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"formal task {task.task_id!r} gamefile is not a regular file: {resolved}"
        )
    return resolved


def verify_formal_manifest(
    manifest: TaskManifestSet,
    *,
    alfworld_data: str | Path,
    role: str,
    profile: str = "pilot_v1",
) -> dict[str, Any]:
    """Verify formal identity, balance, containment, existence, and file SHA.

    ``role`` is intentionally explicit: a balanced ``valid_seen`` manifest
    must never be accepted in place of Train30 merely because it also has 30
    rows.  Every gamefile is resolved with symlinks followed and is required
    to remain under the resolved ALFWORLD data root.
    """

    spec = _verify_manifest_structure(manifest, role=role, profile=profile)
    try:
        data_root = Path(alfworld_data).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ALFWORLD_DATA root is missing: {Path(alfworld_data).expanduser()}"
        ) from exc
    if not data_root.is_dir():
        raise NotADirectoryError(f"ALFWORLD_DATA is not a directory: {data_root}")

    if profile in {"formal_v2", "smoke_v1"}:
        _verify_frozen_selection(manifest, data_root=data_root, spec=spec)

    for task in manifest.tasks:
        gamefile = _resolve_gamefile(
            data_root,
            task,
            source_split=spec.source_split,
        )
        actual = hashlib.sha256(gamefile.read_bytes()).hexdigest()
        if actual != task.gamefile_sha256:
            raise ValueError(
                f"formal task {task.task_id!r} gamefile SHA-256 mismatch: "
                f"declared {task.gamefile_sha256}, actual {actual}"
            )

    return {
        "passed": True,
        "role": spec.role,
        "manifest_id": manifest.manifest_id,
        "manifest_digest": manifest.digest,
        "source_split": manifest.source_split,
        "task_count": len(manifest.tasks),
        "family_counts": dict(Counter(task.task_type for task in manifest.tasks)),
        "physical_gamefiles_verified": len(manifest.tasks),
    }


def verify_observed_gamefile(
    task: ManifestTask,
    observed_gamefile: str | Path,
    *,
    alfworld_data: str | Path,
) -> Path:
    """Match the gamefile reported by an actual environment reset to a task.

    Upstream ALFWorld currently reports ``infos['extra.gamefile']`` as an
    absolute path.  Adapters must validate that value instead of replacing it
    with the manifest path before recording an episode.  Relative reported
    paths are also accepted, but are resolved only underneath ``alfworld_data``.
    """

    try:
        data_root = Path(alfworld_data).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ALFWORLD_DATA root is missing: {Path(alfworld_data).expanduser()}"
        ) from exc
    expected = _resolve_gamefile(
        data_root,
        task,
        source_split=task.source_split,
    )
    raw = os.path.expandvars(str(observed_gamefile)).strip()
    if not raw:
        raise ValueError(
            f"environment reset did not report a gamefile for task {task.task_id!r}"
        )
    reported = Path(raw).expanduser()
    if not reported.is_absolute():
        portable = PurePosixPath(raw.replace("\\", "/"))
        if portable.is_absolute() or any(
            part in {"", ".", ".."} for part in portable.parts
        ):
            raise ValueError(
                f"environment reset reported an unsafe gamefile for task "
                f"{task.task_id!r}: {raw!r}"
            )
        reported = data_root.joinpath(*portable.parts)
    try:
        reported = reported.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"environment reset reported a missing gamefile for task "
            f"{task.task_id!r}: {raw!r}"
        ) from exc
    try:
        reported.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"environment reset gamefile resolves outside ALFWORLD_DATA for "
            f"task {task.task_id!r}: {raw!r}"
        ) from exc
    if reported != expected:
        raise ValueError(
            f"environment reset selected the wrong gamefile for task "
            f"{task.task_id!r}: expected {expected}, got {reported}"
        )
    actual = hashlib.sha256(reported.read_bytes()).hexdigest()
    if actual != task.gamefile_sha256:
        raise ValueError(
            f"environment reset gamefile SHA-256 mismatch for task "
            f"{task.task_id!r}: declared {task.gamefile_sha256}, actual {actual}"
        )
    return reported


def verify_final_evaluation_bijection(
    episodes: Iterable[CommonEpisodeRecord],
    manifest: TaskManifestSet,
    *,
    role: str,
    expected_phase: str | None = None,
    expected_method: str | None = None,
    expected_run_seed: int | None = None,
    expected_artifact_digest: str | None = None,
    require_strict_outcomes: bool = True,
    profile: str = "pilot_v1",
) -> dict[str, Any]:
    """Require exactly one authoritative final-eval episode per manifest row.

    This is for a *final, frozen* evaluation, not the repeated Train or
    Validation rollouts inside an optimization loop.  Infrastructure failures
    invalidate the formal result instead of shrinking its denominator.
    """

    spec = _verify_manifest_structure(manifest, role=role, profile=profile)
    phase = expected_phase or spec.final_evaluation_phase
    if not phase:
        raise ValueError("final evaluation expected_phase must be non-empty")
    observed = list(episodes)
    expected_ids = [task.task_id for task in manifest.tasks]
    observed_ids = [episode.task_id for episode in observed]
    counts = Counter(observed_ids)
    duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
    missing = sorted(set(expected_ids) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(expected_ids))
    if len(observed) != len(manifest.tasks) or missing or duplicates or unexpected:
        raise ValueError(
            "final evaluation episode coverage is not bijective: "
            f"expected={len(manifest.tasks)}, observed={len(observed)}, "
            f"missing={missing[:5]}, duplicates={duplicates[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    if observed_ids != expected_ids:
        raise ValueError(
            "final evaluation episode order does not match manifest order"
        )

    entries = {task.task_id: task for task in manifest.tasks}
    infrastructure_failures: list[str] = []
    for episode in observed:
        task = entries[episode.task_id]
        mismatches: list[str] = []
        if episode.phase != phase:
            mismatches.append(f"phase={episode.phase!r} (expected {phase!r})")
        if expected_method is not None and episode.method != expected_method:
            mismatches.append(
                f"method={episode.method!r} (expected {expected_method!r})"
            )
        if expected_run_seed is not None and episode.run_seed != expected_run_seed:
            mismatches.append(
                f"run_seed={episode.run_seed!r} (expected {expected_run_seed!r})"
            )
        if episode.manifest_index != task.index:
            mismatches.append(
                f"manifest_index={episode.manifest_index!r} (expected {task.index!r})"
            )
        if episode.task_type != task.task_type:
            mismatches.append(
                f"task_type={episode.task_type!r} (expected {task.task_type!r})"
            )
        if episode.gamefile != task.gamefile_rel:
            mismatches.append(
                f"gamefile={episode.gamefile!r} (expected {task.gamefile_rel!r})"
            )
        if episode.gamefile_hash != task.gamefile_sha256:
            mismatches.append(
                "gamefile_hash does not match the manifest entry"
            )
        if not isinstance(episode.official_success, bool):
            mismatches.append("official_success is not boolean")
        if episode.contract_consistency != episode.task_contract_success:
            mismatches.append(
                "contract_consistency disagrees with task_contract_success"
            )
        if episode.common_strict_success != episode.strict_success:
            mismatches.append("common_strict_success disagrees with strict_success")
        if require_strict_outcomes and not isinstance(
            episode.contract_consistency, bool
        ):
            mismatches.append(
                "task_contract_success is not boolean (contract_consistency)"
            )
        if require_strict_outcomes and not isinstance(
            episode.common_strict_success, bool
        ):
            mismatches.append(
                "strict_success is not boolean (common_strict_success)"
            )
        if (
            isinstance(episode.contract_consistency, bool)
            and isinstance(episode.common_strict_success, bool)
            and episode.common_strict_success
            is not (episode.official_success and episode.contract_consistency)
        ):
            mismatches.append(
                "common_strict_success is not official_success && "
                "contract_consistency"
            )
        if expected_artifact_digest is not None and (
            episode.artifact_digest_before != expected_artifact_digest
            or episode.artifact_digest_after != expected_artifact_digest
        ):
            mismatches.append(
                "episode artifact digests do not match the frozen artifact"
            )
        if mismatches:
            raise ValueError(
                f"final evaluation episode {episode.task_id!r} disagrees with "
                f"its manifest/protocol identity: {'; '.join(mismatches)}"
            )
        if episode.infrastructure_failure:
            infrastructure_failures.append(episode.task_id)

    if infrastructure_failures:
        raise RuntimeError(
            "final evaluation contains infrastructure failures; no formal "
            f"accuracy may be reported: {infrastructure_failures[:5]}"
        )

    return {
        "passed": True,
        "role": spec.role,
        "phase": phase,
        "manifest_id": manifest.manifest_id,
        "manifest_digest": manifest.digest,
        "expected_tasks": len(manifest.tasks),
        "observed_tasks": len(observed),
        "missing_task_ids": [],
        "duplicate_task_ids": [],
        "unexpected_task_ids": [],
        "infrastructure_failed_episodes": 0,
        "exact_order": True,
    }


__all__ = [
    "FORMAL_MANIFEST_SPECS",
    "FORMAL_MANIFEST_PROFILES",
    "FormalManifestSpec",
    "verify_final_evaluation_bijection",
    "verify_formal_manifest",
    "verify_observed_gamefile",
]
