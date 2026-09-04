"""Freeze helper for the SkillOpt baseline: the persistent artifact is the
single evolved skill document ``best_skill.md`` (design doc §16.8/§26)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.baselines.common.freeze import FrozenArtifact, freeze_files


def freeze_best_skill(
    *,
    best_skill_path: str | Path,
    frozen_dir: str | Path,
    train_manifest_hash: str,
    validation_manifest_hash: str,
    metadata: dict[str, Any] | None = None,
) -> FrozenArtifact:
    best_skill = Path(best_skill_path)
    if not best_skill.is_file():
        raise FileNotFoundError(f"best_skill.md is missing: {best_skill}")
    return freeze_files(
        method_id="b3_skillopt",
        source_files={"best_skill.md": best_skill},
        destination=Path(frozen_dir),
        source_train_manifest_hash=train_manifest_hash,
        source_validation_manifest_hash=validation_manifest_hash,
        metadata=dict(metadata or {}),
    )
