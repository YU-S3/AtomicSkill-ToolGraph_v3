"""Executable ToolAsset registry."""

from __future__ import annotations

import hashlib
from typing import Any

from ..core.contracts import ToolAsset
from ..core.refs import ToolRef
from ..core.status import RuntimeMode, ToolStatus, tool_status_usable
from .artifact_store import ArtifactStore
from .database import StateDatabase


class ToolRegistry:
    def __init__(self, store: ArtifactStore, database: StateDatabase) -> None:
        self.store = store
        self.database = database

    def register(self, artifact: ToolAsset, *, body: str | bytes | None = None) -> None:
        filename = str(artifact.artifact.get("filename", ""))
        if filename and body is None:
            raise ValueError("a ToolAsset declaring artifact.filename requires its immutable body")
        if body is not None:
            filename = filename or "artifact.txt"
            if not artifact.artifact.get("filename"):
                artifact.artifact["filename"] = filename
            encoded = body if isinstance(body, bytes) else body.encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            declared = str(artifact.artifact.get("body_sha256", ""))
            if declared and declared != digest:
                raise ValueError("declared Tool body_sha256 differs from supplied body")
            artifact.artifact["body_sha256"] = digest
        self.store.put("tool", artifact)
        if body is not None:
            path = self.store.path_for("tool", artifact.ref).parent / filename
            if path.exists():
                existing = path.read_bytes()
                if existing != encoded:
                    raise RuntimeError(f"immutable tool body conflict: {artifact.ref}/{filename}")
            else:
                self.store.put_tool_body(artifact.ref, filename, body)

    def get(self, ref: ToolRef | str) -> ToolAsset:
        ref = ToolRef.parse(ref)
        row = self.database.execute(
            "SELECT status,artifact_kind FROM artifact_index WHERE artifact_ref=?", (str(ref),)
        ).fetchone()
        if row is None or row["artifact_kind"] != "tool":
            raise KeyError(str(ref))
        payload = self.store.get_payload(str(ref))
        return ToolAsset(
            ref=ref, summary=payload["summary"], signature=payload.get("signature", {}),
            interface=payload.get("interface", {}), artifact_kind=payload.get("artifact_kind", "primitive_ir"),
            artifact=payload.get("artifact", {}), tests=payload.get("tests", []), safety=payload.get("safety", {}),
            provenance=payload.get("provenance", {}), metadata=payload.get("metadata", {}), status=ToolStatus(row["status"]),
        )

    def list_refs(self, *, mode: RuntimeMode | str | None = None) -> list[ToolRef]:
        rows = self.database.rows(
            "SELECT logical_id,version,status FROM artifact_index WHERE artifact_kind='tool' ORDER BY logical_id,version"
        )
        return [
            ToolRef(row["logical_id"], row["version"]) for row in rows
            if mode is None or tool_status_usable(row["status"], mode)
        ]

    def tools(self, *, mode: RuntimeMode | str | None = None) -> list[ToolAsset]:
        return [self.get(ref) for ref in self.list_refs(mode=mode)]

    def update_status(self, ref: ToolRef | str, status: ToolStatus | str) -> None:
        if self.database.readonly:
            raise RuntimeError("frozen registry is read-only")
        ref, status = ToolRef.parse(ref), ToolStatus(status)
        cursor = self.database.execute(
            "UPDATE artifact_index SET status=? WHERE artifact_ref=? AND artifact_kind='tool'",
            (status.value, str(ref)),
        )
        if cursor.rowcount != 1:
            raise KeyError(str(ref))
        self.database.connection.commit()

    def body_path(self, ref: ToolRef | str) -> str | None:
        tool = self.get(ref)
        filename = tool.artifact.get("filename")
        if not filename:
            return None
        return str(self.store.path_for("tool", tool.ref).parent / str(filename))
