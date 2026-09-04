"""Freeze protocol shared by all baseline methods.

A frozen artifact is a relocatable directory whose canonical digest is
computed at freeze time.  The held-out test phase must end with
``digest_before == digest_after``; any persistent-knowledge change
invalidates the run (§12 of the design document).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact_digest import digest_directory, verify_digest


@dataclass(frozen=True)
class FrozenArtifact:
    method_id: str
    root: Path
    digest: str
    source_train_manifest_hash: str
    source_validation_manifest_hash: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "digest": self.digest,
            "source_train_manifest_hash": self.source_train_manifest_hash,
            "source_validation_manifest_hash": self.source_validation_manifest_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, root: str | Path) -> "FrozenArtifact":
        root = Path(root)
        manifest_path = root / "digest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"frozen digest manifest is unreadable: {manifest_path}") from exc
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"frozen digest manifest has invalid schema: {manifest_path}")
        declared = str(payload.get("digest", ""))
        artifact_root = root / "artifact"
        if not artifact_root.is_dir():
            raise ValueError(f"frozen artifact directory is missing: {artifact_root}")
        if not verify_digest(artifact_root, declared):
            raise ValueError(
                f"frozen artifact digest mismatch in {artifact_root}; the snapshot "
                "is corrupt or was modified"
            )
        return cls(
            method_id=str(payload["method_id"]),
            root=artifact_root,
            digest=declared,
            source_train_manifest_hash=str(payload.get("source_train_manifest_hash", "")),
            source_validation_manifest_hash=payload.get("source_validation_manifest_hash"),
            metadata=dict(payload.get("metadata") or {}),
        )


def freeze_files(
    *,
    method_id: str,
    source_files: dict[str, Path],
    destination: Path,
    source_train_manifest_hash: str,
    source_validation_manifest_hash: str | None,
    metadata: dict[str, Any] | None = None,
) -> FrozenArtifact:
    """Copy the persistent knowledge files into an immutable frozen snapshot."""

    if destination.exists():
        raise FileExistsError(destination)
    artifact_root = destination / "artifact"
    artifact_root.mkdir(parents=True)
    for relative, source in source_files.items():
        target = artifact_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    digest = digest_directory(artifact_root)
    frozen_metadata = dict(metadata or {})
    frozen_metadata.update({
        "frozen_at": time.time(),
        "file_sha256": {
            relative: hashlib.sha256(source.read_bytes()).hexdigest()
            for relative, source in source_files.items()
        },
    })
    payload = {
        "schema_version": 1,
        "method_id": method_id,
        "digest": digest,
        "source_train_manifest_hash": source_train_manifest_hash,
        "source_validation_manifest_hash": source_validation_manifest_hash,
        "metadata": frozen_metadata,
    }
    (destination / "digest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return FrozenArtifact(
        method_id=method_id,
        root=artifact_root,
        digest=digest,
        source_train_manifest_hash=source_train_manifest_hash,
        source_validation_manifest_hash=source_validation_manifest_hash,
        metadata=frozen_metadata,
    )


def assert_frozen_unchanged(artifact: FrozenArtifact) -> None:
    """Fail closed when the frozen artifact diverged from its freeze-time digest."""

    if not verify_digest(artifact.root, artifact.digest):
        raise RuntimeError(
            f"frozen artifact {artifact.root} changed during the held-out phase; "
            "the run is invalid (persistent knowledge must not mutate on Test)"
        )
