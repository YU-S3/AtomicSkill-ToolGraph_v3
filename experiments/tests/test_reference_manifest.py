from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.protocol import ProtocolError, sha256_json
from experiments.reference_manifest import (
    FORMAL_REFERENCE_MANIFEST_SPECS,
    ReferenceManifest,
    load_formal_reference_manifest,
    load_reference_manifest,
    select_reference_tasks,
    validate_reference_disjoint,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "data" / "baseline_manifests"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_payload(
    *,
    index: int,
    env_index: int,
    source_split: str,
    task_type: str = "pick_and_place_simple",
    gamefile_rel: str | None = None,
    gamefile_sha256: str | None = None,
    task_signature_value: str | None = None,
) -> dict[str, Any]:
    relative = gamefile_rel or (
        f"json_2.1.1/{source_split}/{task_type}-fixture-{env_index}/"
        "trial_fixture/game.tw-pddl"
    )
    return {
        "index": index,
        "task_id": f"task-{source_split}-{env_index}",
        "task_type": task_type,
        "source_split": source_split,
        "env_index": env_index,
        "gamefile_rel": relative,
        "gamefile_sha256": gamefile_sha256 or sha256_json(f"file-{env_index}"),
        "task_signature": task_signature_value
        or sha256_json(f"signature-{source_split}-{env_index}"),
    }


def _manifest_payload(
    *,
    manifest_id: str,
    source_split: str,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "benchmark": "alfworld",
        "source_split": source_split,
        "seed": 42,
        "tasks": tasks,
    }
    payload["digest"] = sha256_json(payload)
    return payload


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / f"{payload['manifest_id']}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_frozen_reference_manifests_have_exact_identity() -> None:
    train = load_formal_reference_manifest(MANIFEST_ROOT / "train_120.json")
    test = load_formal_reference_manifest(MANIFEST_ROOT / "test_ood_full_134.json")

    assert train.digest == FORMAL_REFERENCE_MANIFEST_SPECS["train_120"].digest
    assert len(train.tasks) == 120
    assert test.digest == FORMAL_REFERENCE_MANIFEST_SPECS["test_ood_full_134"].digest
    assert len(test.tasks) == 134
    audit = validate_reference_disjoint(train, test)
    assert audit == {
        "passed": True,
        "train_manifest_id": "train_120",
        "train_digest": train.digest,
        "train_tasks": 120,
        "test_manifest_id": "test_ood_full_134",
        "test_digest": test.digest,
        "test_tasks": 134,
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda payload: payload.update(extra=True), "schema mismatch"),
        (lambda payload: payload.pop("seed"), "schema mismatch"),
        (
            lambda payload: payload["tasks"][0].update(index=2),
            "contiguous from zero",
        ),
        (
            lambda payload: payload["tasks"][1].update(
                env_index=payload["tasks"][0]["env_index"]
            ),
            "env_index values must be unique",
        ),
        (
            lambda payload: payload["tasks"][0].update(task_type="unsupported"),
            "not a formal ALFWorld family",
        ),
        (
            lambda payload: payload["tasks"][0].update(
                gamefile_rel="../game.tw-pddl"
            ),
            "normalized relative POSIX path",
        ),
    ),
)
def test_reference_loader_rejects_schema_and_identity_changes(
    tmp_path: Path, mutation: Any, error: str
) -> None:
    payload = _manifest_payload(
        manifest_id="fixture",
        source_split="train",
        tasks=[
            _task_payload(index=0, env_index=2, source_split="train"),
            _task_payload(index=1, env_index=7, source_split="train"),
        ],
    )
    mutation(payload)
    # Recompute the digest so the test reaches the intended structural gate.
    unsigned = {key: value for key, value in payload.items() if key != "digest"}
    payload["digest"] = sha256_json(unsigned)
    with pytest.raises(ProtocolError, match=error):
        load_reference_manifest(_write_manifest(tmp_path, payload))


def test_reference_loader_rejects_payload_tampering_by_digest(tmp_path: Path) -> None:
    payload = _manifest_payload(
        manifest_id="fixture",
        source_split="train",
        tasks=[_task_payload(index=0, env_index=2, source_split="train")],
    )
    payload["tasks"][0]["task_id"] = "tampered"
    with pytest.raises(ProtocolError, match="digest mismatch"):
        load_reference_manifest(_write_manifest(tmp_path, payload))


def test_formal_loader_pins_filename_id_and_frozen_digest(tmp_path: Path) -> None:
    payload = json.loads(
        (MANIFEST_ROOT / "train_120.json").read_text(encoding="utf-8")
    )
    payload["seed"] = 43
    unsigned = {key: value for key, value in payload.items() if key != "digest"}
    payload["digest"] = sha256_json(unsigned)
    path = _write_manifest(tmp_path, payload)

    with pytest.raises(ProtocolError, match="seed must be 42"):
        load_formal_reference_manifest(path, manifest_id="train_120")
    with pytest.raises(ProtocolError, match="formal reference manifest id mismatch"):
        load_formal_reference_manifest(
            MANIFEST_ROOT / "train_120.json", manifest_id="test_ood_full_134"
        )


@dataclass
class _FakeHarness:
    split: str
    alfworld_data: str
    tasks: list[Any]
    limits: list[int] = field(default_factory=list)

    def load_tasks(self, *, limit: int = 0) -> list[Any]:
        self.limits.append(limit)
        return list(self.tasks)


def _selection_fixture(
    tmp_path: Path,
) -> tuple[ReferenceManifest, _FakeHarness, dict[int, Path]]:
    files: dict[int, Path] = {}
    entries: list[dict[str, Any]] = []
    for manifest_index, env_index in enumerate((2, 7)):
        relative = (
            f"json_2.1.1/train/pick_and_place_simple-fixture-{env_index}/"
            "trial_fixture/game.tw-pddl"
        )
        gamefile = tmp_path / relative
        gamefile.parent.mkdir(parents=True, exist_ok=True)
        content = f"game-{env_index}".encode("utf-8")
        gamefile.write_bytes(content)
        files[env_index] = gamefile
        entries.append(
            _task_payload(
                index=manifest_index,
                env_index=env_index,
                source_split="train",
                gamefile_rel=relative,
                gamefile_sha256=_digest_bytes(content),
            )
        )
    manifest = load_reference_manifest(
        _write_manifest(
            tmp_path,
            _manifest_payload(
                manifest_id="selection_fixture",
                source_split="train",
                tasks=entries,
            ),
        )
    )
    actual_by_env = {
        item.env_index: SimpleNamespace(
            task_id=item.task_id,
            task_type=item.task_type,
            context={"env_index": item.env_index, "game_file": str(files[item.env_index])},
            metadata={"task_signature": item.task_signature},
        )
        for item in manifest.tasks
    }
    # Deliberately return a different order: selection authority is env_index.
    harness = _FakeHarness(
        split="train",
        alfworld_data=str(tmp_path),
        tasks=[actual_by_env[7], actual_by_env[2]],
    )
    return manifest, harness, files


def test_reference_selection_is_env_index_authoritative_and_manifest_ordered(
    tmp_path: Path,
) -> None:
    manifest, harness, _ = _selection_fixture(tmp_path)

    selected = select_reference_tasks(harness, manifest)

    assert harness.limits == [0]
    assert [task.context["env_index"] for task in selected] == [2, 7]
    assert [task.task_id for task in selected] == [
        task.task_id for task in manifest.tasks
    ]


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("task_id", "task_id mismatch"),
        ("task_type", "task_type mismatch"),
        ("task_signature", "task_signature mismatch"),
        ("gamefile_rel", "gamefile_rel mismatch"),
        ("gamefile_sha256", "gamefile SHA-256 mismatch"),
    ),
)
def test_reference_selection_fails_closed_on_observed_identity_mismatch(
    tmp_path: Path, field: str, error: str
) -> None:
    manifest, harness, files = _selection_fixture(tmp_path)
    actual = harness.tasks[1]  # env_index 2, the first manifest entry
    if field == "task_id":
        actual.task_id = "wrong-id"
    elif field == "task_type":
        actual.task_type = "look_at_obj_in_light"
    elif field == "task_signature":
        actual.metadata["task_signature"] = sha256_json("wrong-signature")
    elif field == "gamefile_rel":
        other = tmp_path / (
            "json_2.1.1/train/pick_and_place_simple-other/"
            "trial_fixture/game.tw-pddl"
        )
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_bytes(files[2].read_bytes())
        actual.context["game_file"] = str(other)
    elif field == "gamefile_sha256":
        files[2].write_bytes(b"changed-after-manifest")

    with pytest.raises(ProtocolError, match=error):
        select_reference_tasks(harness, manifest)


def test_reference_selection_rejects_wrong_split_and_duplicate_env_index(
    tmp_path: Path,
) -> None:
    manifest, harness, _ = _selection_fixture(tmp_path)
    wrong_split = copy.copy(harness)
    wrong_split.split = "eval_out_of_distribution"
    with pytest.raises(ProtocolError, match="harness split mismatch"):
        select_reference_tasks(wrong_split, manifest)

    harness.tasks.append(harness.tasks[0])
    with pytest.raises(ProtocolError, match="duplicate env_index"):
        select_reference_tasks(harness, manifest)


def test_reference_disjoint_rejects_shared_signature(tmp_path: Path) -> None:
    shared = sha256_json("shared")
    train_payload = _manifest_payload(
        manifest_id="train_fixture",
        source_split="train",
        tasks=[
            _task_payload(
                index=0,
                env_index=2,
                source_split="train",
                task_signature_value=shared,
            )
        ],
    )
    test_payload = _manifest_payload(
        manifest_id="test_fixture",
        source_split="valid_unseen",
        tasks=[
            _task_payload(
                index=0,
                env_index=2,
                source_split="valid_unseen",
                task_signature_value=shared,
            )
        ],
    )
    train = load_reference_manifest(_write_manifest(tmp_path, train_payload))
    # Use a second directory because both helpers otherwise use manifest_id safely,
    # but keeping inputs separate makes the physical fixture explicit.
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    test = load_reference_manifest(_write_manifest(test_dir, test_payload))

    with pytest.raises(ProtocolError, match="share task_signature"):
        validate_reference_disjoint(train, test)
