"""Manifest builder acceptance test.

The critical acceptance is that ``train_30`` is the exact physical 30-task
batch of the main Full-30 experiment: same env indices, same task types, and
the same frozen task signatures (recorded from the main experiment's
immutable run manifest).  Selection drift in the ALFWorld data or the
harness ordering fails closed here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.baselines.common.manifest import (
    TaskManifestSet,
    verify_disjoint,
    verify_nesting,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMITTED_MANIFEST_DIR = REPO_ROOT / "data" / "baseline_manifests"

# Frozen from the main experiment's immutable Full-30 run manifest
# (runs/alfworld_train_full_30/task_manifest.json): (env_index, task_type,
# task_signature) in manifest order.
_FROZEN_MAIN_EXPERIMENT_TRAIN_30 = [
    (0, "pick_and_place_simple", "d1d0e3927843611f7f01d22b27789546bc3a73d40c1b3d6938af7a2f4701aa03"),
    (1, "pick_and_place_simple", "ff67ac0b6a129fbaad7165ce1541746d52daceabc0aa8d14e9d2c0f0cd56d962"),
    (2, "pick_two_obj_and_place", "6842c1fc755243cdb063ea36feb4a727ae47f2ffb49bfc6036fcf01bf98abb5e"),
    (3, "pick_two_obj_and_place", "8f7b587fceb75ea8922b37a800632fa3737b94bfc378de5df4f710dfa665d96a"),
    (4, "pick_clean_then_place_in_recep", "df1172765d85f08503150c3399fe1a6f4e629e974a72683964f822b4c3cc0278"),
    (5, "pick_two_obj_and_place", "d8658ce70da874cc561462ce8e54261b49363eaa124d34022bb1185bc651e5e0"),
    (6, "pick_cool_then_place_in_recep", "db463fd5f4d3342b50160766c9aa25e190c867ed70d6dc156e993dc5219e51d3"),
    (7, "pick_clean_then_place_in_recep", "a066609f682ee51de5fa43e835b08954cdae085ec95e80499df827591c01b737"),
    (9, "pick_two_obj_and_place", "ec201fa6eaa6c1a21bca6f1a9bf52b48586340c7b9d3fc8f30bd02a7c22f8c8a"),
    (10, "pick_clean_then_place_in_recep", "9abfe39357add082ec0e43c308f7d907bf36701b55a1acaea624ef89169d2adb"),
    (11, "pick_and_place_simple", "e007021557328106d5b100aa6779e2c79f6c5f04586eab4100fb810d3bc1a1d0"),
    (12, "pick_clean_then_place_in_recep", "ee530296e731e998fcf3d6ebe2ab8d6480462e180850732abbfc712a3765bce6"),
    (13, "pick_heat_then_place_in_recep", "1058826e421e4a10e0e1aef9595c51402fe6608603cffce94d56a31814433840"),
    (14, "pick_two_obj_and_place", "58d5d6380dad44558146e7e1a2885fc67d7ac9a3c3ecc14f77ebb2962838c219"),
    (18, "pick_and_place_simple", "1bff3495920c2d34d4b6fe56a6185a5fd376e027d95f1042bc8d833205c24492"),
    (21, "pick_clean_then_place_in_recep", "6b72fc4fb9752369845908e4df4391829ae2fa21d82c2ae0c6b3318a04e41f8c"),
    (22, "pick_and_place_simple", "4ae4602ef6d57c47c863f66663214b51eb407736bd52663d9b5448b1ccbd534d"),
    (24, "look_at_obj_in_light", "e3612ddf9837ee089577545a73dd36a5b824f6822e3a854c7cb6b19ecc6a97c6"),
    (25, "pick_heat_then_place_in_recep", "061847e758e3329602c4063a5b248dcedcd482db88acbdd7ac7ba3268758e9be"),
    (29, "pick_cool_then_place_in_recep", "243fdff529609b9495dd8a9e31df0c7bf2c2a531e59e63e048599344d85ed87f"),
    (30, "look_at_obj_in_light", "61b70a761d3e6703b494cba9a8339a448657918c848d7a4150c6bdf35bbe6b42"),
    (33, "look_at_obj_in_light", "0229a0bf1c42934b42b960f8077abc688dc12411da9e6070103bbcf5740b74ab"),
    (34, "look_at_obj_in_light", "bd8758bc3ea0a29a503aee9a52882d898edfe57a055029cb37ded3c431c995a2"),
    (36, "pick_heat_then_place_in_recep", "8c33238fc063fdc77d3853503e65f1f08f98b7a2cda83174ffbb395eca24286d"),
    (40, "pick_cool_then_place_in_recep", "3a0a82e7e80b5be17f4b28340dc133aa89ea05e30a2124e0113e7949ba72bd62"),
    (42, "look_at_obj_in_light", "73a839b74abb70d40fd5ef84f372d498ee1e13fb855f79ae273f162aa6f2f5d8"),
    (46, "pick_heat_then_place_in_recep", "88bd735b4e7a78a6bdd8cc512e45f78b6af514f53240312d63e2836d333f7aa5"),
    (47, "pick_cool_then_place_in_recep", "22c21b75c4d59a7ea5bb6d359ef9df459db295d05fafef68fcac2d38cd19e595"),
    (51, "pick_heat_then_place_in_recep", "33842b70a5a0e59cf0a744b4340239bc8e0929470d41f6c1c5bb53c968e6be94"),
    (59, "pick_cool_then_place_in_recep", "8889ce25aba799a613e1fc5fe7840c04d231c2d3bf70408558ddd49812083641"),
]


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
    """Load the committed common manifests (built by manifest_builder)."""

    if not _COMMITTED_MANIFEST_DIR.is_dir():
        pytest.skip("data/baseline_manifests has not been built yet")
    return {
        path.stem: TaskManifestSet.load(path)
        for path in sorted(_COMMITTED_MANIFEST_DIR.glob("*.json"))
    }


def test_six_class_balanced_counts(manifests) -> None:
    expected = {
        "train_30": 30, "train_120": 120, "train_300": 300,
        "validation_30": 30, "test_ood_60": 60, "test_ood_full_134": 134,
    }
    for name, total in expected.items():
        assert len(manifests[name].tasks) == total
        counts = {
            task.task_type: sum(item.task_type == task.task_type for item in manifests[name].tasks)
            for task in manifests[name].tasks
        }
        assert len(counts) == 6


def test_train_30_matches_main_experiment_batch(manifests) -> None:
    train_30 = manifests["train_30"]
    observed = [
        (task.env_index, task.task_type, task.task_signature)
        for task in train_30.tasks
    ]
    assert observed == _FROZEN_MAIN_EXPERIMENT_TRAIN_30


def test_train_nesting(manifests) -> None:
    verify_nesting(manifests["train_30"], manifests["train_120"])
    verify_nesting(manifests["train_120"], manifests["train_300"])


def test_split_disjointness(manifests) -> None:
    verify_disjoint(
        manifests["train_300"],
        manifests["validation_30"],
        manifests["test_ood_full_134"],
    )


def test_gamefile_hashes_match_data(manifests) -> None:
    data = _alfworld_data()
    for task in manifests["train_30"].tasks[:5]:
        assert task.gamefile_sha256 == _sha256(data / task.gamefile_rel)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
