"""Acceptance tests for the frozen baseline manifest selection algorithm."""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from pathlib import Path

import pytest

from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES
from experiments.baselines.common.manifest import (
    TaskManifestSet,
    verify_disjoint,
    verify_nesting,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMITTED_MANIFEST_DIR = REPO_ROOT / "data" / "baseline_manifests"


def _alfworld_data() -> Path:
    value = os.environ.get("ALFWORLD_DATA", "")
    if not value:
        return Path.home() / ".cache" / "alfworld"
    return Path(value).expanduser()


pytestmark = pytest.mark.skipif(
    not (_alfworld_data() / "json_2.1.1").is_dir(),
    reason="ALFWorld data is not available",
)


@pytest.fixture(scope="module")
def manifests() -> dict[str, TaskManifestSet]:
    if not _COMMITTED_MANIFEST_DIR.is_dir():
        pytest.skip("data/baseline_manifests has not been built yet")
    return {
        path.stem: TaskManifestSet.load(path)
        for path in sorted(_COMMITTED_MANIFEST_DIR.glob("*.json"))
    }


def _family(path: str) -> str:
    directory = path.split("/")[2]
    matches = [
        task_type
        for task_type in ALFWORLD_FORMAL_TASK_TYPES
        if directory == task_type or directory.startswith(task_type + "-")
    ]
    assert len(matches) == 1
    return matches[0]


def _physical_family_permutations(split: str) -> dict[str, list[str]]:
    root = _alfworld_data().resolve()
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "json_2.1.1" / split).rglob("game.tw-pddl")
    )
    rng = random.Random(42)
    result: dict[str, list[str]] = {}
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        family = [path for path in paths if _family(path) == task_type]
        rng.shuffle(family)
        result[task_type] = family
    return result


def _expected_balanced_paths(split: str, per_type: int) -> list[str]:
    permutations = _physical_family_permutations(split)
    return sorted(
        path
        for task_type in ALFWORLD_FORMAL_TASK_TYPES
        for path in permutations[task_type][:per_type]
    )


@pytest.mark.parametrize(
    ("name", "total", "per_type"),
    [
        ("train_6_smoke", 6, 1),
        ("train_30", 30, 5),
        ("train_120", 120, 20),
        ("train_300", 300, 50),
        ("validation_6_smoke", 6, 1),
        ("validation_6", 6, 1),
        ("validation_12", 12, 2),
        ("validation_24", 24, 4),
        ("validation_30", 30, 5),
        ("test_6_smoke", 6, 1),
        ("test_ood_60", 60, 10),
    ],
)
def test_balanced_manifest_counts(manifests, name, total, per_type) -> None:
    manifest = manifests[name]
    assert manifest.seed == 42
    assert len(manifest.tasks) == total
    assert Counter(task.task_type for task in manifest.tasks) == {
        task_type: per_type for task_type in ALFWORLD_FORMAL_TASK_TYPES
    }
    assert [task.gamefile_rel for task in manifest.tasks] == sorted(
        task.gamefile_rel for task in manifest.tasks
    )


def test_formal_train_and_validation_use_seeded_family_prefixes(manifests) -> None:
    assert [task.gamefile_rel for task in manifests["train_120"].tasks] == (
        _expected_balanced_paths("train", 20)
    )
    assert [task.gamefile_rel for task in manifests["validation_24"].tasks] == (
        _expected_balanced_paths("valid_seen", 4)
    )


def test_full_test_is_all_134_games_in_canonical_order(manifests) -> None:
    test = manifests["test_ood_full_134"]
    root = _alfworld_data().resolve()
    expected = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "json_2.1.1" / "valid_unseen").rglob(
            "game.tw-pddl"
        )
    )
    assert test.seed == 42
    assert len(expected) == len(test.tasks) == 134
    assert [task.gamefile_rel for task in test.tasks] == expected
    assert Counter(task.task_type for task in test.tasks) == {
        "pick_and_place_simple": 24,
        "look_at_obj_in_light": 18,
        "pick_clean_then_place_in_recep": 31,
        "pick_heat_then_place_in_recep": 23,
        "pick_cool_then_place_in_recep": 21,
        "pick_two_obj_and_place": 17,
    }


def test_smoke_pilot_and_formal_nesting(manifests) -> None:
    verify_nesting(manifests["train_6_smoke"], manifests["train_30"])
    verify_nesting(manifests["train_30"], manifests["train_120"])
    verify_nesting(manifests["train_120"], manifests["train_300"])
    verify_nesting(manifests["validation_6_smoke"], manifests["validation_12"])
    verify_nesting(manifests["validation_12"], manifests["validation_24"])
    verify_nesting(manifests["validation_24"], manifests["validation_30"])
    verify_nesting(manifests["test_6_smoke"], manifests["test_ood_60"])


def test_split_disjointness(manifests) -> None:
    verify_disjoint(
        manifests["train_300"],
        manifests["validation_30"],
        manifests["test_ood_full_134"],
    )


def test_gamefile_hashes_match_data(manifests) -> None:
    data = _alfworld_data()
    for name in ("train_120", "validation_24", "test_ood_full_134"):
        for task in manifests[name].tasks:
            assert task.gamefile_sha256 == hashlib.sha256(
                (data / task.gamefile_rel).read_bytes()
            ).hexdigest()
