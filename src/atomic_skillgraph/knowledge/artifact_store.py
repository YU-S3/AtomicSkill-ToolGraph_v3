"""Crash-safe immutable JSON/body artifact store registered in SQLite."""

from __future__ import annotations

import os
import shutil
import hashlib
from pathlib import Path
from typing import Any

from ..core.errors import ArtifactIntegrityError, FailureLayer
from ..core.refs import SkillRef, ToolRef, content_hash
from ..core.serialization import atomic_write_json, read_json, to_primitive
from .database import SCHEMA_VERSION, StateDatabase


_KIND_DIR = {
    "atomic": "atomic",
    "implementation": "implementation",
    "composite": "composite",
    "tool": "tools",
}


class ArtifactStore:
    def __init__(self, data_dir: str | Path, database: StateDatabase) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "artifacts"
        self.database = database
        if not database.readonly:
            for directory in _KIND_DIR.values():
                (self.root / directory).mkdir(parents=True, exist_ok=True)

    def _ref_parts(self, ref: SkillRef | ToolRef) -> tuple[str, str, str]:
        if isinstance(ref, SkillRef):
            return str(ref), ref.logical_id, ref.version
        return str(ref), ref.tool_id, ref.version

    def path_for(self, kind: str, ref: SkillRef | ToolRef) -> Path:
        if kind not in _KIND_DIR:
            raise ValueError(f"unsupported artifact kind: {kind}")
        _, logical_id, version = self._ref_parts(ref)
        if kind == "tool":
            return self.root / _KIND_DIR[kind] / logical_id / version / "tool.json"
        return self.root / _KIND_DIR[kind] / logical_id / f"{version}.json"

    def put(self, kind: str, artifact: Any, *, status: str | None = None) -> Path:
        ref = artifact.ref
        artifact_ref, logical_id, version = self._ref_parts(ref)
        payload = to_primitive(artifact)
        payload["schema_version"] = SCHEMA_VERSION
        digest = content_hash(payload, exclude=("status", "quality", "statistics", "evidence"))
        target = self.path_for(kind, ref)
        existing = self.database.execute(
            "SELECT content_hash,file_path,schema_version FROM artifact_index WHERE artifact_ref=?",
            (artifact_ref,),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != digest:
                raise ArtifactIntegrityError(
                    "immutable_artifact_conflict",
                    f"{artifact_ref} already exists with different content",
                    layer=FailureLayer.INFRASTRUCTURE,
                )
            self.verify_ref(artifact_ref)
            return Path(existing["file_path"])
        if self.database.readonly:
            raise RuntimeError("frozen artifact store is read-only")
        atomic_write_json(target, payload)
        current_status = status or str(getattr(getattr(artifact, "status", "draft"), "value", getattr(artifact, "status", "draft")))
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO artifact_index(artifact_ref,artifact_kind,logical_id,version,content_hash,status,file_path,schema_version) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (artifact_ref, kind, logical_id, version, digest, str(current_status), str(target.resolve()), SCHEMA_VERSION),
                )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def put_tool_body(self, ref: ToolRef, filename: str, body: str | bytes) -> Path:
        if self.database.readonly:
            raise RuntimeError("frozen artifact store is read-only")
        if Path(filename).name != filename:
            raise ValueError("tool artifact filename must be a basename")
        path = self.path_for("tool", ref).parent / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        if isinstance(body, str):
            body = body.encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path

    def get_payload(self, artifact_ref: str) -> dict[str, Any]:
        row = self.database.execute(
            "SELECT file_path,schema_version FROM artifact_index WHERE artifact_ref=?", (artifact_ref,)
        ).fetchone()
        if row is None:
            raise KeyError(artifact_ref)
        if int(row["schema_version"]) < SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                "reject_for_runtime", f"legacy artifact rejected: {artifact_ref}",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        self.verify_ref(artifact_ref)
        return read_json(row["file_path"])

    def verify_ref(self, artifact_ref: str) -> None:
        row = self.database.execute(
            "SELECT content_hash,file_path FROM artifact_index WHERE artifact_ref=?", (artifact_ref,)
        ).fetchone()
        if row is None:
            raise KeyError(artifact_ref)
        path = Path(row["file_path"])
        if not path.is_file():
            raise ArtifactIntegrityError(
                "artifact_file_missing", f"indexed artifact file missing: {path}",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        payload = read_json(path)
        actual = content_hash(payload, exclude=("status", "quality", "statistics", "evidence"))
        if actual != row["content_hash"]:
            raise ArtifactIntegrityError(
                "artifact_hash_mismatch", f"artifact hash mismatch: {artifact_ref}",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        if str(artifact_ref).startswith("tool:"):
            artifact = dict(payload.get("artifact") or {})
            filename = str(artifact.get("filename", ""))
            declared_body_hash = str(artifact.get("body_sha256", ""))
            if filename:
                if Path(filename).name != filename or not declared_body_hash:
                    raise ArtifactIntegrityError(
                        "artifact_hash_mismatch",
                        f"tool body metadata is incomplete: {artifact_ref}",
                        layer=FailureLayer.INFRASTRUCTURE,
                    )
                body_path = path.parent / filename
                if not body_path.is_file():
                    raise ArtifactIntegrityError(
                        "artifact_file_missing", f"indexed tool body missing: {body_path}",
                        layer=FailureLayer.INFRASTRUCTURE,
                    )
                actual_body_hash = hashlib.sha256(body_path.read_bytes()).hexdigest()
                if actual_body_hash != declared_body_hash:
                    raise ArtifactIntegrityError(
                        "artifact_hash_mismatch", f"tool body hash mismatch: {artifact_ref}/{filename}",
                        layer=FailureLayer.INFRASTRUCTURE,
                    )

    def verify_all(self) -> None:
        for row in self.database.rows("SELECT artifact_ref FROM artifact_index ORDER BY artifact_ref"):
            self.verify_ref(row["artifact_ref"])

    def copy_snapshot(self, destination: str | Path) -> Path:
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(self.root, destination / "artifacts")
        return destination
