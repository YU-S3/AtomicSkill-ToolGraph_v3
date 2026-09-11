"""Formal manifest authority and frozen-evaluation coverage tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES
from experiments.baselines.common.formal_validation import (
    verify_final_evaluation_bijection,
    verify_formal_manifest,
    verify_observed_gamefile,
)
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet
from experiments.baselines.common.schema import CommonEpisodeRecord


def _build_manifest(
    data_root: Path,
    *,
    role: str,
) -> TaskManifestSet:
    manifest_id = "train_30" if role == "train" else "validation_6"
    split = "train" if role == "train" else "valid_seen"
    per_family = 5 if role == "train" else 1
    tasks: list[ManifestTask] = []
    index = 0
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        for family_index in range(per_family):
            relative = (
                Path("json_2.1.1")
                / split
                / task_type
                / f"trial_{family_index}"
                / "game.tw-pddl"
            )
            gamefile = data_root / relative
            gamefile.parent.mkdir(parents=True, exist_ok=True)
            content = f"{manifest_id}:{task_type}:{family_index}\n".encode()
            gamefile.write_bytes(content)
            tasks.append(ManifestTask(
                index=index,
                task_id=f"{manifest_id}_{index}",
                task_type=task_type,
                source_split=split,
                env_index=index,
                gamefile_rel=relative.as_posix(),
                gamefile_sha256=hashlib.sha256(content).hexdigest(),
                task_signature=hashlib.sha256(
                    f"signature:{manifest_id}:{index}".encode()
                ).hexdigest(),
            ))
            index += 1
    return TaskManifestSet.create(
        manifest_id=manifest_id,
        benchmark="alfworld",
        source_split=split,
        seed=42,
        tasks=tuple(tasks),
    )


def _build_formal_v2_manifest(
    data_root: Path,
    *,
    role: str,
    seed: int = 42,
) -> TaskManifestSet:
    specs = {
        "train": ("train_120", "train", {
            task_type: 20 for task_type in ALFWORLD_FORMAL_TASK_TYPES
        }),
        "validation": ("validation_24", "valid_seen", {
            task_type: 4 for task_type in ALFWORLD_FORMAL_TASK_TYPES
        }),
        "test": ("test_ood_full_134", "valid_unseen", {
            "pick_and_place_simple": 24,
            "look_at_obj_in_light": 18,
            "pick_clean_then_place_in_recep": 31,
            "pick_heat_then_place_in_recep": 23,
            "pick_cool_then_place_in_recep": 21,
            "pick_two_obj_and_place": 17,
        }),
    }
    manifest_id, split, counts = specs[role]
    pending: list[tuple[str, str, bytes]] = []
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        for family_index in range(counts[task_type]):
            relative = (
                Path("json_2.1.1")
                / split
                / f"{task_type}-Fixture-{family_index:03d}"
                / "trial_fixture"
                / "game.tw-pddl"
            )
            content = f"{manifest_id}:{task_type}:{family_index}\n".encode()
            gamefile = data_root / relative
            gamefile.parent.mkdir(parents=True, exist_ok=True)
            gamefile.write_bytes(content)
            pending.append((relative.as_posix(), task_type, content))
    tasks = tuple(
        ManifestTask(
            index=index,
            task_id=f"{manifest_id}_{index}",
            task_type=task_type,
            source_split=split,
            env_index=index,
            gamefile_rel=relative,
            gamefile_sha256=hashlib.sha256(content).hexdigest(),
            task_signature=hashlib.sha256(
                f"signature:{manifest_id}:{relative}".encode()
            ).hexdigest(),
        )
        for index, (relative, task_type, content) in enumerate(sorted(pending))
    )
    return TaskManifestSet.create(
        manifest_id=manifest_id,
        benchmark="alfworld",
        source_split=split,
        seed=seed,
        tasks=tasks,
    )


def _replace_task(
    manifest: TaskManifestSet,
    index: int,
    **changes,
) -> TaskManifestSet:
    tasks = list(manifest.tasks)
    tasks[index] = replace(tasks[index], **changes)
    return TaskManifestSet.create(
        manifest_id=manifest.manifest_id,
        benchmark=manifest.benchmark,
        source_split=manifest.source_split,
        seed=manifest.seed,
        tasks=tuple(tasks),
    )


def _episodes(
    manifest: TaskManifestSet,
    *,
    phase: str,
    digest: str = "f" * 64,
) -> list[CommonEpisodeRecord]:
    return [
        CommonEpisodeRecord(
            method="b3_skillopt",
            phase=phase,
            run_seed=42,
            task_id=task.task_id,
            task_type=task.task_type,
            manifest_index=task.index,
            gamefile=task.gamefile_rel,
            gamefile_hash=task.gamefile_sha256,
            official_success=bool(task.index % 2),
            task_contract_success=bool(task.index % 2),
            strict_success=bool(task.index % 2),
            target_llm_calls=1,
            artifact_digest_before=digest,
            artifact_digest_after=digest,
        )
        for task in manifest.tasks
    ]


@pytest.mark.parametrize(
    ("role", "split"),
    [("train", "train"), ("validation", "valid_seen")],
)
def test_formal_manifest_verifies_all_physical_files(
    tmp_path: Path,
    role: str,
    split: str,
) -> None:
    manifest = _build_manifest(tmp_path, role=role)
    report = verify_formal_manifest(
        manifest,
        alfworld_data=tmp_path,
        role=role,
    )
    assert report["passed"] is True
    assert report["source_split"] == split
    assert report["task_count"] == (30 if role == "train" else 6)
    assert report["physical_gamefiles_verified"] == (
        30 if role == "train" else 6
    )
    assert report["family_counts"] == {
        task_type: (5 if role == "train" else 1)
        for task_type in ALFWORLD_FORMAL_TASK_TYPES
    }


def test_formal_manifest_rejects_role_and_split_drift(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, role="validation")
    with pytest.raises(ValueError, match="manifest_id mismatch"):
        verify_formal_manifest(manifest, alfworld_data=tmp_path, role="train")

    train = _build_manifest(tmp_path, role="train")
    drifted = _replace_task(train, 0, source_split="valid_seen")
    with pytest.raises(ValueError, match="task .* source_split mismatch"):
        verify_formal_manifest(drifted, alfworld_data=tmp_path, role="train")


def test_formal_manifest_rejects_family_imbalance(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    drifted = _replace_task(
        manifest,
        0,
        task_type=ALFWORLD_FORMAL_TASK_TYPES[1],
    )
    with pytest.raises(ValueError, match="family balance mismatch"):
        verify_formal_manifest(drifted, alfworld_data=tmp_path, role="train")


@pytest.mark.parametrize(
    ("role", "expected_tasks"),
    [("train", 120), ("validation", 24), ("test", 134)],
)
def test_formal_v2_manifest_recomputes_frozen_physical_selection(
    tmp_path: Path,
    role: str,
    expected_tasks: int,
) -> None:
    manifest = _build_formal_v2_manifest(tmp_path, role=role)
    report = verify_formal_manifest(
        manifest,
        alfworld_data=tmp_path,
        role=role,
        profile="formal_v2",
    )
    assert report["passed"] is True
    assert report["task_count"] == expected_tasks


def test_formal_v2_rejects_selection_seed_drift(tmp_path: Path) -> None:
    manifest = _build_formal_v2_manifest(tmp_path, role="validation", seed=43)
    with pytest.raises(ValueError, match="selection seed mismatch"):
        verify_formal_manifest(
            manifest,
            alfworld_data=tmp_path,
            role="validation",
            profile="formal_v2",
        )


def test_formal_manifest_rejects_parent_path_escape(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    drifted = _replace_task(
        manifest,
        0,
        gamefile_rel="json_2.1.1/train/../../outside/game.tw-pddl",
    )
    with pytest.raises(ValueError, match="escapes|normalized"):
        verify_formal_manifest(drifted, alfworld_data=tmp_path, role="train")


def test_formal_manifest_rejects_missing_file_and_sha_drift(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    missing = tmp_path / manifest.tasks[0].gamefile_rel
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="gamefile is missing"):
        verify_formal_manifest(manifest, alfworld_data=tmp_path, role="train")

    manifest = _build_manifest(tmp_path, role="train")
    changed = tmp_path / manifest.tasks[0].gamefile_rel
    changed.write_text("changed after manifest creation", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_formal_manifest(manifest, alfworld_data=tmp_path, role="train")


def test_observed_reset_gamefile_must_match_manifest(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    task = manifest.tasks[0]
    expected = (tmp_path / task.gamefile_rel).resolve()
    assert verify_observed_gamefile(
        task, expected, alfworld_data=tmp_path,
    ) == expected
    assert verify_observed_gamefile(
        task, task.gamefile_rel, alfworld_data=tmp_path,
    ) == expected

    wrong = (tmp_path / manifest.tasks[1].gamefile_rel).resolve()
    with pytest.raises(ValueError, match="wrong gamefile"):
        verify_observed_gamefile(task, wrong, alfworld_data=tmp_path)

    outside = tmp_path.parent / "outside_game.tw-pddl"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="outside ALFWORLD_DATA"):
        verify_observed_gamefile(task, outside, alfworld_data=tmp_path)


@pytest.mark.parametrize(
    ("role", "phase"),
    [("train", "train_eval"), ("validation", "validation_eval")],
)
def test_final_evaluation_requires_phase_aware_bijection(
    tmp_path: Path,
    role: str,
    phase: str,
) -> None:
    manifest = _build_manifest(tmp_path, role=role)
    report = verify_final_evaluation_bijection(
        _episodes(manifest, phase=phase),
        manifest,
        role=role,
        expected_method="b3_skillopt",
        expected_run_seed=42,
        expected_artifact_digest="f" * 64,
    )
    assert report["passed"] is True
    assert report["expected_tasks"] == report["observed_tasks"] == (
        30 if role == "train" else 6
    )
    assert report["phase"] == phase


def test_final_evaluation_rejects_missing_duplicate_unexpected_and_order(
    tmp_path: Path,
) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    episodes = _episodes(manifest, phase="train_eval")

    with pytest.raises(ValueError, match="not bijective.*missing"):
        verify_final_evaluation_bijection(episodes[:-1], manifest, role="train")

    duplicate = list(episodes)
    duplicate[-1] = replace(duplicate[-1], task_id=duplicate[0].task_id)
    with pytest.raises(ValueError, match="duplicates"):
        verify_final_evaluation_bijection(duplicate, manifest, role="train")

    unexpected = list(episodes)
    unexpected[-1] = replace(unexpected[-1], task_id="not_in_manifest")
    with pytest.raises(ValueError, match="unexpected"):
        verify_final_evaluation_bijection(unexpected, manifest, role="train")

    with pytest.raises(ValueError, match="order"):
        verify_final_evaluation_bijection(
            list(reversed(episodes)), manifest, role="train",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("phase", "train", "phase="),
        ("manifest_index", 999, "manifest_index="),
        ("task_type", "wrong_family", "task_type="),
        ("gamefile", "json_2.1.1/train/wrong/game.tw-pddl", "gamefile="),
        ("gamefile_hash", "0" * 64, "gamefile_hash"),
        ("method", "wrong_method", "method="),
        ("run_seed", 7, "run_seed="),
        ("artifact_digest_after", "0" * 64, "artifact digests"),
    ],
)
def test_final_evaluation_rejects_identity_field_drift(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    episodes = _episodes(manifest, phase="train_eval")
    episodes[0] = replace(episodes[0], **{field: value})
    with pytest.raises(ValueError, match=message):
        verify_final_evaluation_bijection(
            episodes,
            manifest,
            role="train",
            expected_method="b3_skillopt",
            expected_run_seed=42,
            expected_artifact_digest="f" * 64,
        )


def test_final_evaluation_rejects_non_boolean_outcome_and_infrastructure(
    tmp_path: Path,
) -> None:
    manifest = _build_manifest(tmp_path, role="train")
    episodes = _episodes(manifest, phase="train_eval")
    episodes[0] = replace(episodes[0], strict_success=None)
    with pytest.raises(ValueError, match="strict_success is not boolean"):
        verify_final_evaluation_bijection(episodes, manifest, role="train")

    episodes = _episodes(manifest, phase="train_eval")
    episodes[0] = replace(
        episodes[0],
        infrastructure_failure=True,
        infrastructure_error="provider timeout",
    )
    with pytest.raises(RuntimeError, match="infrastructure failures"):
        verify_final_evaluation_bijection(episodes, manifest, role="train")
