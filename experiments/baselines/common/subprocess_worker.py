"""Subprocess worker wire protocol (§28 of the design document).

The controller writes a wire JSON file (model identity + manifest paths +
output dir, never an API key), spawns the per-method venv worker, and reads
the worker's result JSON.  The worker inherits the controller's environment,
which is where ``MODEL_API_KEY`` / ``ALFWORLD_DATA`` live.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WIRE_SCHEMA_VERSION = 2
WORKER_RESULT_SCHEMA_VERSION = 1

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_IDENTITY_KEYS = (
    "config_digest",
    "train_manifest_digest",
    "validation_manifest_digest",
)


@dataclass(frozen=True)
class WorkerWire:
    method: str
    phase: str
    manifest_path: str | None
    validation_manifest_path: str | None
    test_manifest_path: str | None
    config_path: str
    output_dir: str
    run_seed: int
    model: dict[str, str]
    run_id: str
    result_path: str
    identity: dict[str, str]
    frozen_artifact_path: str | None = None
    external_skillopt_root: str | None = None
    skill_init_rel: str | None = None

    def __post_init__(self) -> None:
        if not self.method.strip() or not self.phase.strip():
            raise ValueError("worker wire method and phase must be non-empty")
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("worker wire run_id is empty or contains unsafe characters")
        if not self.output_dir.strip() or not self.result_path.strip():
            raise ValueError("worker wire output_dir and result_path must be non-empty")
        if not isinstance(self.model, dict) or not self.model:
            raise ValueError("worker wire model identity must be a non-empty mapping")
        if not isinstance(self.identity, dict):
            raise ValueError("worker wire identity must be a mapping")
        missing = [key for key in _REQUIRED_IDENTITY_KEYS if key not in self.identity]
        if missing:
            raise ValueError(
                "worker wire identity is missing required keys: " + ", ".join(missing)
            )
        for key, value in self.identity.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("worker wire identity keys must be non-empty strings")
            if not isinstance(value, str):
                raise ValueError(f"worker wire identity value for {key!r} must be a string")
            if key.endswith("_digest") and not _SHA256_RE.fullmatch(value):
                raise ValueError(
                    f"worker wire identity digest {key!r} must be lowercase SHA-256"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": WIRE_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerWire":
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != WIRE_SCHEMA_VERSION:
            raise ValueError("worker wire payload has invalid schema_version")
        try:
            return cls(
                method=str(payload["method"]),
                phase=str(payload["phase"]),
                manifest_path=payload.get("manifest_path"),
                validation_manifest_path=payload.get("validation_manifest_path"),
                test_manifest_path=payload.get("test_manifest_path"),
                config_path=str(payload["config_path"]),
                output_dir=str(payload["output_dir"]),
                run_seed=int(payload["run_seed"]),
                model={str(key): str(value) for key, value in dict(payload["model"]).items()},
                run_id=str(payload["run_id"]),
                result_path=str(payload["result_path"]),
                identity={
                    str(key): str(value)
                    for key, value in dict(payload["identity"]).items()
                },
                frozen_artifact_path=payload.get("frozen_artifact_path"),
                external_skillopt_root=payload.get("external_skillopt_root"),
                skill_init_rel=payload.get("skill_init_rel"),
            )
        except KeyError as exc:
            raise ValueError(f"worker wire payload is missing {exc.args[0]}") from exc


def run_worker(
    *,
    wire: WorkerWire,
    worker_module: str,
    python: str | Path,
    wire_dir: Path,
) -> dict[str, Any]:
    """Run a worker module in the per-method venv and return its result JSON."""

    _validate_wire_layout(wire, wire_dir)
    wire_path = wire_dir / "worker_wire.json"
    wire_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = Path(wire.result_path)
    if wire_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing worker wire: {wire_path}"
        )
    if result_path.exists():
        raise FileExistsError(
            f"refusing to reuse a stale worker result: {result_path}"
        )
    _write_json_atomic(wire_path, wire.to_dict(), overwrite=False)
    completed = subprocess.run(
        [str(python), "-m", worker_module, "--wire", str(wire_path)],
        check=False,
        env=dict(os.environ),
    )
    if not result_path.is_file():
        return {
            "passed": False,
            "worker_exit_code": completed.returncode,
            "error": "worker produced no worker_result.json",
        }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "worker_exit_code": completed.returncode,
            "error": f"worker_result.json is corrupt: {exc}",
        }
    contract_error = _worker_result_contract_error(result, wire)
    if contract_error is not None:
        return {
            "passed": False,
            "worker_exit_code": completed.returncode,
            "error": contract_error,
        }
    result["worker_exit_code"] = completed.returncode
    if completed.returncode != 0:
        result["passed"] = False
        result.setdefault(
            "error",
            f"worker exited with non-zero status {completed.returncode}",
        )
    return result


def write_worker_result(
    wire: WorkerWire,
    result: dict[str, Any],
    *,
    passed: bool,
) -> Path:
    """Write one identity-bound worker result atomically.

    The worker owns exactly the path declared by its wire.  Reserved protocol
    fields are derived from the wire rather than trusted from method-specific
    result data, and an existing result is never overwritten.
    """

    target = Path(wire.result_path)
    _validate_wire_layout(wire, target.parent)
    if not isinstance(result, dict):
        raise TypeError("worker result payload must be a mapping")
    expected = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "method": wire.method,
        "phase": wire.phase,
        "run_id": wire.run_id,
        "run_seed": wire.run_seed,
        "model": dict(wire.model),
        "identity": dict(wire.identity),
        "passed": bool(passed),
    }
    for key, value in expected.items():
        if key in result and result[key] != value:
            raise ValueError(f"worker result attempts to override reserved field {key!r}")
    payload = {**result, **expected}
    _write_json_atomic(target, payload, overwrite=False)
    return target


def _validate_wire_layout(wire: WorkerWire, wire_dir: Path) -> None:
    phase_dir = (Path(wire.output_dir) / wire.phase).resolve()
    actual_wire_dir = Path(wire_dir).resolve()
    result_path = Path(wire.result_path).resolve()
    expected_result = phase_dir / "worker_result.json"
    if actual_wire_dir != phase_dir:
        raise ValueError(
            f"worker wire directory must be the phase directory {phase_dir}, "
            f"got {actual_wire_dir}"
        )
    if result_path != expected_result:
        raise ValueError(
            f"worker result_path must be {expected_result}, got {result_path}"
        )


def _worker_result_contract_error(result: Any, wire: WorkerWire) -> str | None:
    if not isinstance(result, dict):
        return "worker_result.json root must be a mapping"
    expected = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "method": wire.method,
        "phase": wire.phase,
        "run_id": wire.run_id,
        "run_seed": wire.run_seed,
        "model": wire.model,
        "identity": wire.identity,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            return (
                f"worker_result.json identity mismatch for {key}: "
                f"expected {value!r}, got {result.get(key)!r}"
            )
    if not isinstance(result.get("passed"), bool):
        return "worker_result.json passed field must be boolean"
    return None


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
