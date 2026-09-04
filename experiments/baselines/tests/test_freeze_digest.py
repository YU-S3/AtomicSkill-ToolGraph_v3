"""Freeze protocol acceptance tests (§32 Frozen + §12 digest invariant)."""

from __future__ import annotations

import pytest

from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import (
    FrozenArtifact,
    assert_frozen_unchanged,
    freeze_files,
)


def _skill(tmp_path) -> None:
    (tmp_path / "best_skill.md").write_text("# Skill\nInitial content.\n", encoding="utf-8")


def test_freeze_create_load_verify(tmp_path) -> None:
    _skill(tmp_path)
    frozen = freeze_files(
        method_id="b3_skillopt",
        source_files={"best_skill.md": tmp_path / "best_skill.md"},
        destination=tmp_path / "frozen",
        source_train_manifest_hash="a" * 64,
        source_validation_manifest_hash="b" * 64,
    )
    assert frozen.root.is_dir()
    assert (frozen.root / "best_skill.md").is_file()
    loaded = FrozenArtifact.load(tmp_path / "frozen")
    assert loaded.digest == frozen.digest
    assert loaded.source_train_manifest_hash == "a" * 64
    assert_frozen_unchanged(loaded)


def test_frozen_tamper_detected(tmp_path) -> None:
    _skill(tmp_path)
    freeze_files(
        method_id="b3_skillopt",
        source_files={"best_skill.md": tmp_path / "best_skill.md"},
        destination=tmp_path / "frozen",
        source_train_manifest_hash="a" * 64,
        source_validation_manifest_hash=None,
    )
    (tmp_path / "frozen" / "artifact" / "best_skill.md").write_text(
        "mutated", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        FrozenArtifact.load(tmp_path / "frozen")


def test_frozen_addition_detected(tmp_path) -> None:
    _skill(tmp_path)
    frozen = freeze_files(
        method_id="b3_skillopt",
        source_files={"best_skill.md": tmp_path / "best_skill.md"},
        destination=tmp_path / "frozen",
        source_train_manifest_hash="a" * 64,
        source_validation_manifest_hash=None,
    )
    (frozen.root / "sneaky.md").write_text("new file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        assert_frozen_unchanged(frozen)


def test_directory_digest_is_content_based(tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.txt").write_text("one", encoding="utf-8")
    first = digest_directory(root)
    (root / "a.txt").write_text("two", encoding="utf-8")
    assert digest_directory(root) != first
