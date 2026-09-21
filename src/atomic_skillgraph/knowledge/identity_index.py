"""Versioned identity retrieval index over immutable artifacts, not authority.

The caller's existing artifact transaction owns the row. A bucket or hash match
must always be followed by the complete typed matcher before reuse or credit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.serialization import atomic_create_json, read_json, to_primitive, dataclass_from_dict
from ..core.contracts import AbstractAtomicSkill, ToolAsset, ImplementationAtom
from ..evolution.identity_matching import (IDENTITY_VERSION, _atomic_graph, raw_hash,
    typed_json, match_atomic, match_tool, match_implementation, MAX_SEARCH_STATES)

DDL = """
CREATE TABLE IF NOT EXISTS artifact_identity_index (
 artifact_ref TEXT NOT NULL, identity_version TEXT NOT NULL,
 artifact_kind TEXT NOT NULL, raw_payload_hash TEXT NOT NULL,
 harness_profile TEXT NOT NULL, bucket_key TEXT NOT NULL,
 refinement_fingerprint TEXT NOT NULL, equivalence_id TEXT NOT NULL,
 proof_path TEXT NOT NULL, proof_hash TEXT NOT NULL,
 PRIMARY KEY(artifact_ref, identity_version),
 FOREIGN KEY(artifact_ref) REFERENCES artifact_index(artifact_ref)
);
CREATE INDEX IF NOT EXISTS identity_bucket ON artifact_identity_index(identity_version,artifact_kind,bucket_key);
CREATE INDEX IF NOT EXISTS identity_equivalence ON artifact_identity_index(identity_version,equivalence_id);
"""


def bucket_key(kind: str, artifact: Any) -> str:
    if kind == "atomic":
        graph, _ = _atomic_graph(artifact)
        return raw_hash(sorted(graph.refinement().values()))
    if kind == "tool":
        return raw_hash([artifact.artifact_kind, len(artifact.signature.get("properties", {})),
            len(artifact.interface.get("output_schema", {}).get("properties", {}))])
    if kind == "implementation":
        return raw_hash([len(artifact.tool_bindings), sorted(b.order for b in artifact.tool_bindings)])
    return ""


def _read_artifact(database, ref, cls):
    row = database.execute("SELECT file_path FROM artifact_index WHERE artifact_ref=?", (str(ref),)).fetchone()
    if row is None:
        raise KeyError(str(ref))
    return dataclass_from_dict(cls, read_json(row[0]))


def _closure(database, impl):
    return {"atomic": _read_artifact(database, impl.abstract_ref, AbstractAtomicSkill),
            "tools": {str(b.tool_ref): _read_artifact(database, b.tool_ref, ToolAsset) for b in impl.tool_bindings}}


def _match(kind, source, target, source_dependencies, target_dependencies, remaining):
    if kind == "atomic":
        return match_atomic(source, target, max_states=remaining)
    if kind == "tool":
        return match_tool(source, target, max_states=remaining)
    return match_implementation(source, target, max_states=remaining,
        source_atomic=source_dependencies["atomic"], target_atomic=target_dependencies["atomic"],
        source_tools=source_dependencies["tools"], target_tools=target_dependencies["tools"])


def prepare_index_row(root: Path, kind: str, artifact: Any, immutable_payload: dict, database) -> tuple | None:
    if kind not in {"atomic", "tool", "implementation"}:
        return None
    proof, target, source_dependencies, target_dependencies = None, artifact, {}, {}
    equivalence = str(artifact.ref)
    bucket = bucket_key(kind, artifact)
    cls = {"atomic": AbstractAtomicSkill, "tool": ToolAsset, "implementation": ImplementationAtom}[kind]
    try:
        if kind == "implementation":
            source_dependencies = _closure(database, artifact)
        target_dependencies = source_dependencies
        remaining = MAX_SEARCH_STATES
        for row in database.rows("SELECT artifact_ref,equivalence_id FROM artifact_identity_index WHERE "
                "identity_version=? AND artifact_kind=? AND bucket_key=? ORDER BY rowid",
                (IDENTITY_VERSION, kind, bucket)):
            candidate = _read_artifact(database, row["artifact_ref"], cls)
            dependencies = _closure(database, candidate) if kind == "implementation" else {}
            result = _match(kind, artifact, candidate, source_dependencies, dependencies, remaining)
            remaining -= result.search_states
            if result.status == "exact":
                proof, target, target_dependencies = result.proof, candidate, dependencies
                equivalence = row["equivalence_id"]
                break
            if not remaining:
                break
        if proof is None and remaining:
            proof = _match(kind, artifact, artifact, source_dependencies, source_dependencies, remaining).proof
    except KeyError:
        # A disconnected fixture or pre-admission artifact is not an identity
        # proof. No missing dependency is filled or manufactured here.
        proof = None
    payload = {"identity_version": IDENTITY_VERSION, "artifact_kind": kind,
        "artifact_ref": str(artifact.ref), "immutable_payload_hash": raw_hash(immutable_payload),
        "artifact_snapshot": to_primitive(artifact), "proof": to_primitive(proof),
        "target_snapshot": to_primitive(target), "source_dependencies": to_primitive(source_dependencies),
        "target_dependencies": to_primitive(target_dependencies),
        "status": "exact" if proof else "unproven"}
    digest = raw_hash(payload)
    relative = Path("artifacts") / "identity" / "proofs" / (digest + ".json")
    path = root / relative
    if path.exists():
        if typed_json(read_json(path)) != typed_json(payload):
            raise RuntimeError("identity proof content-address collision")
    else:
        atomic_create_json(path, payload)
    return (str(artifact.ref), IDENTITY_VERSION, kind, raw_hash(immutable_payload),
        str(getattr(artifact, "metadata", {}).get("harness_profile", "")), bucket, bucket,
        equivalence, relative.as_posix(), digest)


class IdentityIndex:
    def __init__(self, database: Any, data_dir: str | Path):
        self.database, self.data_dir = database, Path(data_dir)

    def candidates(self, kind: str, artifact: Any) -> list[str] | None:
        present = self.database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_identity_index'").fetchone()
        if not present:
            return None  # Historical readonly reports; never create a table here.
        # Incomplete indexes are not negative authority. This also allows
        # explicit legacy fixture rows to be read without silently migrating.
        missing = self.database.execute("SELECT 1 FROM artifact_index a WHERE a.artifact_kind=? AND NOT EXISTS "
            "(SELECT 1 FROM artifact_identity_index i WHERE i.artifact_ref=a.artifact_ref AND i.identity_version=?) LIMIT 1",
            (kind, IDENTITY_VERSION)).fetchone()
        if missing:
            if self.database.r103:
                raise RuntimeError("R10.3 cannot query an incomplete identity index")
            return None
        rows = self.database.rows("SELECT artifact_ref FROM artifact_identity_index WHERE identity_version=? "
            "AND artifact_kind=? AND bucket_key=? ORDER BY artifact_ref",
            (IDENTITY_VERSION, kind, bucket_key(kind, artifact)))
        return [row["artifact_ref"] for row in rows]

    def verify(self, artifact_ref: str) -> None:
        row = self.database.execute("SELECT * FROM artifact_identity_index WHERE artifact_ref=? AND identity_version=?",
            (artifact_ref, IDENTITY_VERSION)).fetchone()
        if row is None:
            raise RuntimeError("missing immutable identity index row")
        path = (self.data_dir / row["proof_path"]).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise RuntimeError("identity proof path escapes knowledge root")
        proof = read_json(path)
        if raw_hash(proof) != row["proof_hash"]:
            raise RuntimeError("identity proof hash mismatch")
        artifact = self.database.execute("SELECT file_path FROM artifact_index WHERE artifact_ref=?", (artifact_ref,)).fetchone()
        if (raw_hash(read_json(artifact["file_path"])) != row["raw_payload_hash"]
                or proof.get("immutable_payload_hash") != row["raw_payload_hash"]
                or proof.get("artifact_ref") != artifact_ref):
            raise RuntimeError("identity index does not bind immutable artifact bytes")
        from ..core.serialization import dataclass_from_dict
        from ..core.contracts import AbstractAtomicSkill, ToolAsset
        from ..evolution.identity_matching import IdentityProof, verify_atomic_proof, verify_tool_proof
        snapshot = proof.get("artifact_snapshot")
        immutable = read_json(artifact["file_path"])
        if typed_json(snapshot) != typed_json({k: v for k, v in immutable.items() if k != "schema_version"}):
            raise RuntimeError("identity proof snapshot differs from immutable asset")
        saved = proof.get("proof")
        if (proof.get("identity_version") != IDENTITY_VERSION or proof.get("artifact_kind") != row["artifact_kind"]):
            raise RuntimeError("identity protocol/kind mismatch")
        if saved is not None:
            from ..evolution.identity_matching import verify_implementation_proof
            kind = row["artifact_kind"]
            cls = {"atomic": AbstractAtomicSkill, "tool": ToolAsset, "implementation": ImplementationAtom}[kind]
            value = dataclass_from_dict(cls, snapshot)
            target = dataclass_from_dict(cls, proof["target_snapshot"])
            target_group = str(value.ref) if target.ref == value.ref else self.database.execute(
                "SELECT equivalence_id FROM artifact_identity_index WHERE artifact_ref=? AND identity_version=?",
                (str(target.ref), IDENTITY_VERSION)).fetchone()[0]
            if row["equivalence_id"] != target_group:
                raise RuntimeError("identity equivalence key does not follow its proof")
            if target.ref != value.ref and typed_json(target) != typed_json(_read_artifact(self.database, target.ref, cls)):
                raise RuntimeError("identity target no longer binds immutable bytes")
            if kind == "implementation":
                source_closure, target_closure = _closure(self.database, value), _closure(self.database, target)
                if (typed_json(source_closure) != typed_json(proof["source_dependencies"])
                        or typed_json(target_closure) != typed_json(proof["target_dependencies"])):
                    raise RuntimeError("identity Implementation dependencies changed")
                verified = verify_implementation_proof(value, target, IdentityProof(**saved),
                    source_atomic=source_closure["atomic"], target_atomic=target_closure["atomic"],
                    source_tools=source_closure["tools"], target_tools=target_closure["tools"])
            else:
                checker = verify_atomic_proof if kind == "atomic" else verify_tool_proof
                verified = checker(value, target, IdentityProof(**saved))
            if not verified:
                raise RuntimeError("stored identity proof does not verify")

    def equivalence_key(self, artifact_ref):
        self.verify(artifact_ref)
        return self.database.execute("SELECT equivalence_id FROM artifact_identity_index WHERE artifact_ref=? AND identity_version=?",
                                     (artifact_ref, IDENTITY_VERSION)).fetchone()[0]
