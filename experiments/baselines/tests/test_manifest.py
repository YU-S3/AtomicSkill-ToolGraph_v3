"""Manifest schema / digest / nesting acceptance tests (§32 Dataset)."""

from __future__ import annotations

import json

import pytest

from experiments.baselines.common.manifest import (
    ManifestTask,
    TaskManifestSet,
    verify_disjoint,
    verify_nesting,
)


def _task(index: int, task_id: str, gamefile_rel: str) -> ManifestTask:
    import hashlib

    return ManifestTask(
        index=index,
        task_id=task_id,
        task_type="pick_and_place_simple",
        source_split="train",
        env_index=index,
        gamefile_rel=gamefile_rel,
        gamefile_sha256="a" * 64,
        task_signature=hashlib.sha256(f"{task_id}:{gamefile_rel}".encode()).hexdigest(),
    )


def _set(name: str, files: list[str]) -> TaskManifestSet:
    tasks = tuple(_task(index, f"task_{name}_{index}", file) for index, file in enumerate(files))
    return TaskManifestSet.create(
        manifest_id=name, benchmark="alfworld", source_split="train", seed=42, tasks=tasks,
    )


def test_manifest_roundtrip_and_digest(tmp_path) -> None:
    manifest = _set("train_30", [f"json_2.1.1/train/game_{i}.tw-pddl" for i in range(3)])
    path = manifest.save(tmp_path / "train_30.json")
    loaded = TaskManifestSet.load(path)
    assert loaded.digest == manifest.digest
    assert [task.task_id for task in loaded.tasks] == [
        task.task_id for task in manifest.tasks
    ]


def test_manifest_digest_tamper_fail_closed(tmp_path) -> None:
    manifest = _set("train_30", ["json_2.1.1/train/game_0.tw-pddl"])
    path = manifest.save(tmp_path / "train_30.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tasks"][0]["gamefile_rel"] = "json_2.1.1/train/other.tw-pddl"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    with pytest.raises(ValueError, match="digest mismatch"):
        TaskManifestSet.load(path)


def test_manifest_missing_field_fail_closed(tmp_path) -> None:
    manifest = _set("train_30", ["json_2.1.1/train/game_0.tw-pddl"])
    path = manifest.save(tmp_path / "train_30.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["tasks"][0]["task_signature"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    with pytest.raises(ValueError):
        TaskManifestSet.load(path)


def test_manifest_rejects_absolute_gamefile() -> None:
    with pytest.raises(ValueError, match="relative"):
        ManifestTask(
            index=0, task_id="t", task_type="pick_and_place_simple",
            source_split="train", env_index=0,
            gamefile_rel="/absolute/path/game.tw-pddl",
            gamefile_sha256="a" * 64, task_signature="b" * 64,
        )


def test_manifest_rejects_duplicate_signatures() -> None:
    first = _task(0, "a", "json_2.1.1/train/game_0.tw-pddl")
    second = ManifestTask(
        index=1, task_id="b", task_type="pick_and_place_simple",
        source_split="train", env_index=1,
        gamefile_rel="json_2.1.1/train/game_1.tw-pddl",
        gamefile_sha256="a" * 64, task_signature=first.task_signature,
    )
    with pytest.raises(ValueError, match="unique"):
        TaskManifestSet.create(
            manifest_id="m", benchmark="alfworld", source_split="train",
            seed=42, tasks=(first, second),
        )


def test_disjoint_and_nesting() -> None:
    inner = _set("train_30", [f"json_2.1.1/train/game_{i}.tw-pddl" for i in range(3)])
    outer = _set(
        "train_120",
        [f"json_2.1.1/train/game_{i}.tw-pddl" for i in range(6)],
    )
    other = _set("validation_30", ["json_2.1.1/valid_seen/game_0.tw-pddl"])
    verify_nesting(inner, outer)
    verify_disjoint(outer, other)
    with pytest.raises(ValueError, match="not a subset"):
        verify_nesting(outer, inner)
    with pytest.raises(ValueError, match="share"):
        verify_disjoint(inner, outer)
