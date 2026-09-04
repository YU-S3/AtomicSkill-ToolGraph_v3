"""Subprocess worker wire protocol (§28 of the design document).

The controller writes a wire JSON file (model identity + manifest paths +
output dir, never an API key), spawns the per-method venv worker, and reads
the worker's result JSON.  The worker inherits the controller's environment,
which is where ``MODEL_API_KEY`` / ``ALFWORLD_DATA`` live.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WIRE_SCHEMA_VERSION = 1


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
    frozen_artifact_path: str | None = None
    external_skillopt_root: str | None = None
    skill_init_rel: str | None = None

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

    wire_path = wire_dir / "worker_wire.json"
    wire_path.parent.mkdir(parents=True, exist_ok=True)
    wire_path.write_text(
        json.dumps(wire.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(python), "-m", worker_module, "--wire", str(wire_path)],
        check=False,
        env=dict(os.environ),
    )
    result_path = wire_dir / "worker_result.json"
    if not result_path.is_file():
        return {
            "passed": False,
            "worker_exit_code": completed.returncode,
            "error": "worker produced no worker_result.json",
        }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "worker_exit_code": completed.returncode,
            "error": f"worker_result.json is corrupt: {exc}",
        }
    result.setdefault("worker_exit_code", completed.returncode)
    return result
