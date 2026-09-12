"""Immutable B5 GEPA artifact construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.baselines.common.freeze import FrozenArtifact, freeze_files


_REQUIRED_FILES = (
    "best_skill.md",
    "gepa_result.json",
    "candidate_lineage.json",
    "pareto_metadata.json",
    "gepa_audit_summary.json",
)


def freeze_gepa_artifacts(
    *,
    source_files: dict[str, str | Path],
    frozen_dir: str | Path,
    train_manifest_hash: str,
    validation_manifest_hash: str,
    metadata: dict[str, Any] | None = None,
) -> FrozenArtifact:
    normalized = {name: Path(path) for name, path in source_files.items()}
    if set(normalized) != set(_REQUIRED_FILES):
        raise ValueError(
            "GEPA freeze files must be exactly: " + ", ".join(_REQUIRED_FILES)
        )
    for name, path in normalized.items():
        if not path.is_file():
            raise FileNotFoundError(f"GEPA freeze input {name} is missing: {path}")
    if not normalized["best_skill.md"].read_text(encoding="utf-8").strip():
        raise ValueError("GEPA best_skill.md is empty")
    return freeze_files(
        method_id="b5_gepa",
        source_files=normalized,
        destination=Path(frozen_dir),
        source_train_manifest_hash=train_manifest_hash,
        source_validation_manifest_hash=validation_manifest_hash,
        metadata=dict(metadata or {}),
    )
