"""Explicit, exact-release continuation after provider-only infrastructure repair.

Original manifests and task Traces remain immutable. A separate receipt identifies
the historical code and the reviewed replacement; arbitrary code drift still fails.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from atomic_skillgraph.core.serialization import atomic_create_json
from experiments.protocol import ProtocolError, _CODE_EXCLUDED_DIRS, _CODE_SUFFIXES, _file_hashes, sha256_json


RELEASE_FILE = "experiments/provider_recovery_release.json"
RECEIPT_FILE = "provider_recovery.json"


def release_inventory_hash(repo: Path) -> str:
    records = []
    for path in sorted(repo.rglob("*"), key=lambda p: p.as_posix()):
        relative = path.relative_to(repo)
        if (not path.is_file() or path.suffix.casefold() not in _CODE_SUFFIXES
                or relative.as_posix() == RELEASE_FILE
                or any(part in _CODE_EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts)):
            continue
        records.append({"path": relative.as_posix(), "sha256": _sha(path)})
    return sha256_json(records)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(repo: Path, original_hash: str) -> dict:
    declaration = json.loads((repo / RELEASE_FILE).read_text(encoding="utf-8"))
    if (declaration.get("original_code_hash") != original_hash
            or declaration.get("patched_inventory_hash") != release_inventory_hash(repo)):
        raise ProtocolError("provider recovery is only valid for the exact reviewed source and replacement release")
    return declaration


def read_recovery(repo: Path, output: Path, current_hash: str) -> dict | None:
    path = output / RECEIPT_FILE
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    _release(repo, manifest["code_commit"])
    expected = {
        "schema_version": 1, "kind": "provider_infrastructure_recovery",
        "run_id": manifest["run_id"], "original_code_hash": manifest["code_commit"],
        "execution_code_hash": current_hash, "config_hash": manifest["config_hash"],
        "original_manifest_sha256": _sha(output / "run_manifest.json"),
        "release_sha256": _sha(repo / RELEASE_FILE),
    }
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ProtocolError("provider recovery receipt does not match this run/code/config")
    return receipt


def prepare_recovery(repo: Path, output: Path, current_hash: str, config_hash: str, *, enabled: bool) -> dict | None:
    existing = read_recovery(repo, output, current_hash)
    if existing or not enabled:
        return existing
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _release(repo, manifest["code_commit"])
    if manifest["config_hash"] != config_hash:
        raise ProtocolError("provider recovery cannot change the experiment configuration")
    checkpoint = output / ".task_checkpoint"
    payload = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if (payload.get("code_commit") != manifest["code_commit"]
            or payload.get("config_hash") != config_hash
            or payload.get("run_id") != manifest["run_id"]
            or payload.get("files") != _file_hashes(checkpoint, exclude={"checkpoint_manifest.json"})):
        raise ProtocolError("provider recovery requires an intact checkpoint from the original run")
    history = output / "attempt_history"
    if list(history.glob("*.unresolved.json")) or any(
        not p.with_name(p.name.replace(".start.json", ".capture.json")).is_file()
        for p in history.glob("*.start.json")
    ):
        raise ProtocolError("provider recovery cannot waive missing immutable attempt Traces")
    for capture in history.glob("*.capture.json"):
        item = json.loads(capture.read_text(encoding="utf-8"))
        for trace_id, digest in item["trace_hashes"].items():
            if _sha(output / "traces" / (trace_id + ".json")) != digest:
                raise ProtocolError("provider recovery detected changed attempt evidence")
    receipt = {
        "schema_version": 1, "kind": "provider_infrastructure_recovery",
        "run_id": manifest["run_id"], "original_code_hash": manifest["code_commit"],
        "execution_code_hash": current_hash, "config_hash": config_hash,
        "original_manifest_sha256": _sha(manifest_path), "release_sha256": _sha(repo / RELEASE_FILE),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "restart_task_id": payload["task_id"],
        "unknown_provider_usage_policy": "retain failed HTTP attempts; known tokens are lower bounds",
    }
    atomic_create_json(output / RECEIPT_FILE, receipt)
    return read_recovery(repo, output, current_hash)


def checkpoint_identity(output: Path, current_hash: str, receipt: dict | None) -> str:
    path = output / ".task_checkpoint/checkpoint_manifest.json"
    if receipt and path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))["code_commit"]
        if recorded in {current_hash, receipt["original_code_hash"]}:
            return recorded
    return current_hash
