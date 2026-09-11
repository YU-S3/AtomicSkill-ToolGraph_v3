"""Fail-closed external-source verification tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from experiments.baselines.bootstrap_external import (
    compute_runtime_tree,
    ensure_skillopt_source,
    load_lock,
    verify_key_files,
    verify_runtime_tree,
)


def _materialize_runtime(root: Path) -> None:
    (root / "skillopt" / "engine").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "skillopt" / "engine" / "trainer.py").write_text(
        "VALUE = 1\n", encoding="utf-8",
    )
    (root / "skillopt" / "model.py").write_text(
        "MODEL = 'test'\n", encoding="utf-8",
    )
    (root / "configs" / "default.yaml").write_text(
        "seed: 42\n", encoding="utf-8",
    )
    for name in ("LICENSE", "README.md", "pyproject.toml", "requirements.txt"):
        (root / name).write_text(f"{name}\n", encoding="utf-8")


def _runtime_spec() -> dict:
    return {
        "algorithm": "sha256-path-content-v1",
        "roots": ["skillopt", "configs"],
        "root_files": [
            "LICENSE", "README.md", "pyproject.toml", "requirements.txt",
        ],
    }


def _lock_for(root: Path) -> dict:
    spec = _runtime_spec()
    spec.update(compute_runtime_tree(root, spec))
    trainer = root / "skillopt" / "engine" / "trainer.py"
    return {
        "schema_version": 1,
        "skillopt": {
            "repo": "https://github.com/microsoft/SkillOpt",
            "commit": "a" * 40,
            "version": "0.2.0",
            "runtime_tree": spec,
            "key_files": {
                "skillopt/engine/trainer.py": hashlib.sha256(
                    trainer.read_bytes()
                ).hexdigest(),
            },
        },
    }


def test_complete_runtime_tree_accepts_exact_source_and_ignores_caches(tmp_path) -> None:
    root = tmp_path / "source"
    _materialize_runtime(root)
    lock = _lock_for(root)

    cache = root / "skillopt" / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-312.pyc").write_bytes(b"generated")

    verified = verify_runtime_tree(root, "skillopt", lock)
    assert verified["sha256"] == lock["skillopt"]["runtime_tree"]["sha256"]
    assert verified["file_count"] == 7
    assert verify_key_files(root, "skillopt", lock) == lock["skillopt"]["key_files"]


@pytest.mark.parametrize("mutation", ["modify", "add", "remove"])
def test_complete_runtime_tree_rejects_any_unlisted_source_drift(
    tmp_path,
    mutation: str,
) -> None:
    root = tmp_path / "source"
    _materialize_runtime(root)
    lock = _lock_for(root)

    model = root / "skillopt" / "model.py"
    if mutation == "modify":
        model.write_text("MODEL = 'changed'\n", encoding="utf-8")
    elif mutation == "add":
        (root / "skillopt" / "extra.py").write_text("EXTRA = 1\n", encoding="utf-8")
    else:
        model.unlink()

    with pytest.raises(RuntimeError, match="runtime tree does not match"):
        verify_key_files(root, "skillopt", lock)


def test_runtime_tree_rejects_traversing_lock_path(tmp_path) -> None:
    root = tmp_path / "source"
    _materialize_runtime(root)
    spec = _runtime_spec()
    spec["roots"] = ["../outside"]

    with pytest.raises(ValueError, match="invalid runtime_tree.roots path"):
        compute_runtime_tree(root, spec)


def test_ensure_skillopt_source_reports_commit_version_and_tree(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "external" / "skillopt"
    _materialize_runtime(source)
    lock = _lock_for(source)
    lock_path = tmp_path / "baseline_lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")

    result = ensure_skillopt_source(
        local_source=source,
        destination=destination,
        lock_path=lock_path,
    )

    assert result["declared_commit"] == "a" * 40
    assert result["declared_version"] == "0.2.0"
    assert result["verification"] == "runtime_tree_sha256+key_file_sha256"
    assert result["runtime_tree"] == {
        "algorithm": "sha256-path-content-v1",
        "sha256": lock["skillopt"]["runtime_tree"]["sha256"],
        "file_count": 7,
    }
    assert load_lock(lock_path)["skillopt"]["runtime_tree"]["file_count"] == 7


def test_committed_lock_pins_latest_verified_runtime_tree() -> None:
    lock = load_lock()
    skillopt = lock["skillopt"]

    assert skillopt["commit"] == "79124b37e9a6371e13b753f8bcd7adb1e493ade1"
    assert skillopt["version"] == "0.2.0"
    assert skillopt["local_snapshot_verification"] == (
        "runtime_tree_sha256+key_file_sha256"
    )
    assert skillopt["runtime_tree"]["file_count"] == 155
    assert len(skillopt["runtime_tree"]["sha256"]) == 64
