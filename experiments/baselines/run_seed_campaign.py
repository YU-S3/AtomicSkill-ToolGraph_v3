"""Run one formal B3 SkillOpt campaign with three independent seed lanes.

The campaign controller owns only experiment protocol.  It does not import or
construct the SkillOpt trainer.  Every lane invokes :mod:`run_method` in a
subprocess and preserves the strict per-seed order::

    Train -> verify frozen artifact -> Test

Lanes may overlap with one another.  All real provider requests are bound to
the campaign gate named by the immutable ``campaign_lock.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import yaml

from .bootstrap_external import load_lock, verify_key_files
from .common.formal_validation import verify_formal_manifest
from .common.freeze import FrozenArtifact, assert_frozen_unchanged
from .common.manifest import TaskManifestSet, sha256_json, verify_disjoint
from .common.model_config import ModelConfig
from .common.runtime_python import resolve_formal_python
from .common.source_identity import hash_code, sanitize_error_text
from .common.trace import load_episodes


REPO_ROOT = Path(__file__).resolve().parents[2]
_METHOD = "b3_skillopt"
_FORMAL_SEEDS = (42, 43, 44)
_LOCK_SCHEMA_VERSION = 1
_REPORT_SCHEMA_VERSION = 1
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PROVIDER_CAP_FALLBACKS = (16, 12, 8)
_MAX_TRAIN_ATTEMPTS = 2


class CommandRunner(Protocol):
    """Injectable command boundary used by deterministic scheduler tests."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
    ) -> int: ...


SourceInspector = Callable[[Path], dict[str, Any]]
PreflightBuilder = Callable[
    ["CampaignSpec", Path, str, Mapping[str, Any]], dict[str, Any]
]


@dataclass(frozen=True)
class CampaignSpec:
    method: str
    seeds: tuple[int, ...]
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    config: Path
    output_dir: Path
    python: Path
    repo_root: Path = REPO_ROOT

    def __post_init__(self) -> None:
        if self.method != _METHOD:
            raise ValueError(f"unsupported campaign method: {self.method!r}")
        if tuple(self.seeds) != _FORMAL_SEEDS:
            raise ValueError(
                "formal B3 campaign seeds must be exactly 42, 43, 44 in order"
            )


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _select_phase_python(
    repo_root: Path,
    configured_value: str | Path,
    supplied_value: str | Path | None,
) -> Path:
    """Select the sole formal interpreter without dereferencing venv links."""

    configured = resolve_formal_python(repo_root, configured_value)
    if supplied_value is None:
        return configured
    supplied = resolve_formal_python(repo_root, supplied_value)
    if str(supplied) != str(configured):
        raise ValueError("--python must exactly match formal worker_python")
    return supplied


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a generated pre-campaign config without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(
                dict(payload),
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@contextmanager
def _method_campaign_lease(
    repo_root: Path,
    *,
    owner: str,
) -> Iterator[Path]:
    """Hold the repository-wide formal-method lease for the whole campaign.

    The frozen protocol permits concurrent seed lanes inside one method but
    forbids two method campaigns from competing for the provider.  A shared
    OS lock (rather than an existence sentinel) also recovers automatically
    after process death.  Formal execution is Linux; the Windows branch keeps
    local development and deterministic tests faithful to the same boundary.
    """

    lock_path = (
        Path(repo_root).resolve()
        / "runs"
        / "baselines"
        / ".formal_method_campaign.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        if os.name == "nt":  # pragma: no cover - formal environment is Linux
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "another formal method campaign already holds the "
                    f"repository campaign lease: {lock_path}"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise RuntimeError(
                    "another formal method campaign already holds the "
                    f"repository campaign lease: {lock_path}"
                ) from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "owner": str(owner),
                "pid": os.getpid(),
                "acquired_at_unix": time.time(),
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        if acquired:
            if os.name == "nt":  # pragma: no cover - formal environment is Linux
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return dict(payload)


def inspect_clean_source(repo_root: Path) -> dict[str, Any]:
    """Return formal controller identity, rejecting any source-tree drift."""

    repo_root = Path(repo_root).resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"controller git command failed: git {' '.join(arguments)}: {detail}"
            )
        return completed.stdout.strip()

    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top != repo_root:
        raise RuntimeError(
            f"controller repository root mismatch: expected {repo_root}, got {top}"
        )
    commit = git("rev-parse", "HEAD").lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("controller HEAD is not a full Git commit identity")
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "src",
        "experiments",
        "configs",
    )
    if status:
        raise RuntimeError(
            "formal campaign requires clean controller source under "
            "src/, experiments/, and configs/:\n" + status
        )
    return {
        "commit": commit,
        "dirty": False,
        "code_digest": hash_code(repo_root),
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"formal campaign {label} must be {expected!r}, got {actual!r}"
        )


def _load_merged_config(spec: CampaignSpec) -> dict[str, Any]:
    common_path = spec.repo_root / "configs" / "baselines" / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    method = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    if not isinstance(common, dict) or not isinstance(method, dict):
        raise ValueError("baseline configuration roots must be mappings")
    return {**common, **method}


def _formal_config_digest(config: Mapping[str, Any]) -> str:
    """Match the seed-independent identity enforced by ``run_method``."""

    normalized = json.loads(json.dumps(dict(config)))
    normalized.pop("protocol", None)
    normalized["run_seed"] = "<campaign-seed>"
    train = dict(normalized.get("train") or {})
    train["seed"] = "<campaign-seed>"
    normalized["train"] = train
    return sha256_json(normalized)


def _manifest_receipt(
    manifest: TaskManifestSet,
    *,
    path: Path,
    role: str,
    alfworld_data: Path,
) -> dict[str, Any]:
    preflight = verify_formal_manifest(
        manifest,
        alfworld_data=alfworld_data,
        role=role,
        profile="formal_v2",
    )
    return {
        "path": str(path),
        "manifest_id": manifest.manifest_id,
        "digest": manifest.digest,
        "tasks": len(manifest.tasks),
        "preflight": preflight,
    }


def build_campaign_lock(
    spec: CampaignSpec,
    campaign_root: Path,
    campaign_run_id: str,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform immutable formal preflight and construct the lock payload."""

    if source_state.get("dirty") is not False:
        raise RuntimeError("formal campaign preflight received dirty controller source")
    commit = str(source_state.get("commit", "")).lower()
    code_digest = str(source_state.get("code_digest", "")).lower()
    for value, label, length in (
        (commit, "controller commit", 40),
        (code_digest, "controller code digest", 64),
    ):
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"formal campaign {label} is invalid")

    config = _load_merged_config(spec)
    _require_equal(config.get("method"), _METHOD, "method")
    _require_equal(config.get("protocol_profile"), "formal_v2", "profile")
    campaign_id = str(config.get("campaign_id", "")).strip()
    if not _CAMPAIGN_ID.fullmatch(campaign_id):
        raise ValueError("formal campaign_id is empty or contains unsafe characters")

    model = ModelConfig.from_mapping(dict(config.get("model") or {}))
    model.validate_formal_identity()
    model.require_api_key()

    parallel = dict(config.get("parallel") or {})
    expected_parallel = {
        "seed_lanes": 3,
        "episode_workers_per_seed": 16,
        "test_workers_per_seed": 16,
        "skillopt_analyst_workers_per_seed": 16,
        "campaign_provider_max_inflight": 16,
    }
    for key, expected in expected_parallel.items():
        _require_equal(parallel.get(key), expected, f"parallel.{key}")
    _require_equal(dict(config.get("env") or {}).get("workers"), 16, "env.workers")
    _require_equal(
        dict(config.get("gradient") or {}).get("analyst_workers"),
        16,
        "gradient.analyst_workers",
    )

    transport = dict(config.get("provider_transport") or {})
    expected_transport = {
        "sdk_max_retries": 0,
        "application_retry_limit": 5,
        "retry_delays_seconds": [2, 5, 10, 20],
        "deterministic_jitter_ratio": 0.10,
    }
    for key, expected in expected_transport.items():
        _require_equal(transport.get(key), expected, f"provider_transport.{key}")

    probe = dict(config.get("provider_probe") or {})
    expected_probe = {
        "enabled": True,
        "concurrency": 16,
        "requests": 32,
        "max_completion_tokens": 256,
        "reasoning_effort": "high",
    }
    for key, expected in expected_probe.items():
        _require_equal(probe.get(key), expected, f"provider_probe.{key}")
    _require_equal(
        probe.get("reasoning_effort"),
        model.reasoning_effort,
        "provider probe reasoning effort",
    )

    resume = dict(config.get("resume") or {})
    expected_resume = {
        "enabled": True,
        "formal_boundary": "epoch",
        "reuse_verified_episode_cache": True,
    }
    for key, expected in expected_resume.items():
        _require_equal(resume.get(key), expected, f"resume.{key}")

    worker_python = resolve_formal_python(
        spec.repo_root, str(config.get("worker_python", "")),
    )
    phase_python = resolve_formal_python(spec.repo_root, spec.python)
    if str(phase_python) != str(worker_python):
        raise ValueError(
            "campaign phase Python must exactly match config worker_python"
        )

    alfworld_raw = os.environ.get("ALFWORLD_DATA", "").strip()
    if not alfworld_raw:
        raise RuntimeError("ALFWORLD_DATA is not set")
    alfworld_data = Path(alfworld_raw).expanduser().resolve(strict=True)
    if not alfworld_data.is_dir():
        raise NotADirectoryError(alfworld_data)

    train = TaskManifestSet.load(spec.train_manifest)
    validation = TaskManifestSet.load(spec.validation_manifest)
    test = TaskManifestSet.load(spec.test_manifest)
    verify_disjoint(train, validation, test)
    manifests = {
        "train": _manifest_receipt(
            train, path=spec.train_manifest, role="train", alfworld_data=alfworld_data
        ),
        "validation": _manifest_receipt(
            validation,
            path=spec.validation_manifest,
            role="validation",
            alfworld_data=alfworld_data,
        ),
        "test": _manifest_receipt(
            test, path=spec.test_manifest, role="test", alfworld_data=alfworld_data
        ),
    }

    external_lock_path = spec.repo_root / "experiments" / "baselines" / "baseline_lock.yaml"
    external_lock = load_lock(external_lock_path)
    external_root = spec.repo_root / ".external" / "skillopt"
    verify_key_files(external_root, "skillopt", external_lock)
    skillopt = dict(external_lock.get("skillopt") or {})
    runtime_tree = dict(skillopt.get("runtime_tree") or {})
    initial_skill_digest = str(
        dict(skillopt.get("key_files") or {}).get(
            "skillopt/envs/alfworld/skills/initial.md", ""
        )
    )

    return {
        "schema_version": _LOCK_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_run_id": campaign_run_id,
        "campaign_root": str(campaign_root),
        "method": spec.method,
        "seeds": list(spec.seeds),
        "train_manifest_digest": train.digest,
        "validation_manifest_digest": validation.digest,
        "test_manifest_digest": test.digest,
        "manifests": manifests,
        "controller_commit": commit,
        "controller_code_digest": code_digest,
        "controller_git": dict(source_state),
        "formal_config_digest": _formal_config_digest(config),
        "config_path": str(spec.config),
        "external_lock_digest": sha256_json(external_lock),
        "external_skillopt_commit": str(skillopt.get("commit", "")),
        "external_runtime_tree_digest": str(runtime_tree.get("sha256", "")),
        "initial_skill_sha256": initial_skill_digest,
        "model": model.model,
        "model_identity": model.to_wire(),
        "reasoning_effort": model.reasoning_effort,
        "seed_lanes": int(parallel["seed_lanes"]),
        "campaign_provider_max_inflight": int(
            parallel["campaign_provider_max_inflight"]
        ),
        "parallel": parallel,
        "retry_policy": {
            "sdk_max_retries": int(transport["sdk_max_retries"]),
            "attempts": int(transport["application_retry_limit"]),
            "delays": list(transport["retry_delays_seconds"]),
            "jitter_ratio": float(transport["deterministic_jitter_ratio"]),
        },
        "provider_probe": probe,
        "resume": resume,
        "provider_gate_dir": str((campaign_root / "provider_gate").resolve()),
        "phase_python": str(phase_python),
        "worker_python": str(worker_python),
        "provider_probe_python": str(worker_python),
        "created_at_unix": time.time(),
    }


def _default_command_runner(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            text=True,
        )
    return int(completed.returncode)


def _provider_probe_command(
    spec: CampaignSpec,
    lock_payload: Mapping[str, Any],
    probe_dir: Path,
) -> list[str]:
    model = dict(lock_payload["model_identity"])
    probe = dict(lock_payload["provider_probe"])
    retry = dict(lock_payload["retry_policy"])
    command = [
        str(lock_payload["provider_probe_python"]),
        "-m",
        "experiments.baselines.common.provider_load_probe",
        "--output-dir",
        str(probe_dir),
        "--gate-dir",
        str(lock_payload["provider_gate_dir"]),
        "--campaign-id",
        str(lock_payload["campaign_id"]),
        "--run-id",
        str(lock_payload["campaign_run_id"]) + "_probe",
        "--seed",
        str(spec.seeds[0]),
        "--base-url",
        str(model["base_url"]),
        "--model",
        str(model["model"]),
        "--api-key-env",
        str(model["api_key_env"]),
        "--reasoning-effort",
        str(model["reasoning_effort"]),
        "--concurrency",
        str(probe["concurrency"]),
        "--requests",
        str(probe["requests"]),
        "--max-completion-tokens",
        str(probe["max_completion_tokens"]),
        "--max-inflight",
        str(lock_payload["campaign_provider_max_inflight"]),
        "--sdk-max-retries",
        str(retry["sdk_max_retries"]),
        "--application-retry-limit",
        str(retry["attempts"]),
        "--retry-delays-seconds",
        ",".join(str(value) for value in retry["delays"]),
        "--deterministic-jitter-ratio",
        str(retry["jitter_ratio"]),
    ]
    if spec.method == "b5_gepa":
        command.extend([
            "--skillopt-root",
            str((spec.repo_root / ".external" / "skillopt").resolve()),
        ])
    return command


def _validate_probe_identity(
    report: Mapping[str, Any],
    *,
    lock_payload: Mapping[str, Any],
    report_path: Path,
) -> None:
    cap = int(lock_payload["campaign_provider_max_inflight"])
    probe = dict(lock_payload["provider_probe"])
    model = dict(lock_payload["model_identity"])
    requests = int(probe["requests"])
    expected = {
        "probe_kind": "campaign_provider_load",
        "campaign_id": str(lock_payload["campaign_id"]),
        "run_id": str(lock_payload["campaign_run_id"]) + "_probe",
        "model": str(model["model"]),
        "reasoning_effort": str(model["reasoning_effort"]),
        "concurrency": cap,
        "requests": requests,
        "max_completion_tokens": int(probe["max_completion_tokens"]),
        "campaign_provider_max_inflight": cap,
        "logical_calls_recorded": requests,
    }
    mismatches = [
        field for field, value in expected.items() if report.get(field) != value
    ]
    if mismatches:
        raise RuntimeError(
            "provider load probe identity is inconsistent: "
            + ", ".join(mismatches)
        )
    if report.get("provider_evidence_complete") is not True:
        raise RuntimeError("provider load probe evidence is incomplete")
    calls_path = Path(str(report.get("provider_calls_path", ""))).resolve()
    try:
        calls_path.relative_to(report_path.parent.resolve())
    except ValueError as exc:
        raise RuntimeError("provider load probe sidecar is outside its probe dir") from exc
    if (
        not calls_path.is_file()
        or str(report.get("provider_calls_sha256", "")) != _sha256_file(calls_path)
    ):
        raise RuntimeError("provider load probe sidecar hash is invalid")


def _validate_probe_report(
    path: Path,
    lock_payload: Mapping[str, Any],
) -> dict[str, Any]:
    report = _read_json(path, "campaign provider load-probe report")
    _validate_probe_identity(
        report,
        lock_payload=lock_payload,
        report_path=path,
    )
    if report.get("passed") is not True:
        raise RuntimeError("campaign provider load probe did not pass")
    completed = report.get("completed_logical_calls")
    exhausted = report.get("exhausted_provider_calls")
    permanent = report.get("permanent_provider_errors")
    expected_requests = int(dict(lock_payload["provider_probe"])["requests"])
    if completed is None or int(completed) != expected_requests:
        raise RuntimeError("provider load probe did not complete every logical call")
    if exhausted is None or permanent is None:
        raise RuntimeError("provider load probe omitted failure counters")
    if int(exhausted) != 0 or int(permanent) != 0:
        raise RuntimeError("provider load probe contains exhausted/permanent failures")
    return report


def _probe_failure_allows_lower_cap(
    report: Mapping[str, Any],
    *,
    lock_payload: Mapping[str, Any],
    report_path: Path,
) -> bool:
    """Return true only for a complete, transient-capacity probe failure."""

    _validate_probe_identity(
        report,
        lock_payload=lock_payload,
        report_path=report_path,
    )
    expected_requests = int(dict(lock_payload["provider_probe"])["requests"])
    exhausted = report.get("exhausted_provider_calls")
    permanent = report.get("permanent_provider_errors")
    completed = report.get("completed_logical_calls")
    failed = report.get("failed_provider_calls")
    recorded = report.get("logical_calls_recorded")
    if (
        exhausted is None
        or permanent is None
        or completed is None
        or failed is None
        or recorded is None
    ):
        raise RuntimeError("provider load probe omitted required failure counters")
    return bool(
        report.get("passed") is not True
        and report.get("observer_error_type") is None
        and int(recorded) == expected_requests
        and int(permanent) == 0
        and int(exhausted) > 0
        and int(exhausted) == int(failed)
        and int(completed) + int(failed) == expected_requests
        and 0 <= int(completed) < expected_requests
    )


def _payload_for_provider_cap(
    lock_payload: Mapping[str, Any],
    *,
    campaign_root: Path,
    cap: int,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(lock_payload)))
    parallel = dict(payload["parallel"])
    parallel["campaign_provider_max_inflight"] = int(cap)
    probe = dict(payload["provider_probe"])
    probe["concurrency"] = int(cap)
    payload.update(
        {
            "campaign_provider_max_inflight": int(cap),
            "parallel": parallel,
            "provider_probe": probe,
            "provider_gate_dir": str(
                (campaign_root / "provider_gate" / f"global_{cap:02d}").resolve()
            ),
        }
    )
    return payload


def _runtime_spec_for_provider_cap(
    spec: CampaignSpec,
    lock_payload: dict[str, Any],
    *,
    cap: int,
) -> tuple[CampaignSpec, dict[str, Any]]:
    """Materialize the selected pre-campaign cap as the phase config identity."""

    if cap == int(_PROVIDER_CAP_FALLBACKS[0]):
        return spec, lock_payload
    method_config = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    if not isinstance(method_config, dict):
        raise ValueError("method configuration root must be a mapping")
    method_config = dict(method_config)
    method_config["parallel"] = dict(lock_payload["parallel"])
    method_config["provider_probe"] = dict(lock_payload["provider_probe"])
    runtime_config = (
        spec.output_dir / f"campaign_config_global_{int(cap):02d}.yaml"
    ).resolve()
    _write_yaml_atomic(runtime_config, method_config)
    runtime_spec = replace(spec, config=runtime_config)
    payload = dict(lock_payload)
    payload["config_path"] = str(runtime_config)
    payload["formal_config_digest"] = _formal_config_digest(
        _load_merged_config(runtime_spec)
    )
    return runtime_spec, payload


def _run_provider_probe_with_fallback(
    spec: CampaignSpec,
    lock_payload: Mapping[str, Any],
    *,
    command_runner: CommandRunner,
) -> tuple[CampaignSpec, dict[str, Any], dict[str, Any]]:
    """Probe 16, then 12 and 8 before freezing exactly one campaign cap."""

    requested_cap = int(lock_payload["campaign_provider_max_inflight"])
    if requested_cap != _PROVIDER_CAP_FALLBACKS[0]:
        raise ValueError("formal provider probing must start at global cap 16")
    attempts: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] | None = None
    selected_report: dict[str, Any] | None = None

    for index, cap in enumerate(_PROVIDER_CAP_FALLBACKS):
        candidate = _payload_for_provider_cap(
            lock_payload,
            campaign_root=spec.output_dir,
            cap=cap,
        )
        probe_dir = spec.output_dir / "provider_probe" / f"global_{cap:02d}"
        report_path = probe_dir / "provider_load_probe.json"
        returncode = command_runner(
            _provider_probe_command(spec, candidate, probe_dir),
            cwd=spec.repo_root,
            log_path=spec.output_dir / f"provider_probe_global_{cap:02d}.log",
        )
        if not report_path.is_file():
            raise RuntimeError(
                f"global{cap} provider load probe produced no durable report "
                f"(exit status {returncode})"
            )
        report = _read_json(report_path, f"global{cap} provider load-probe report")
        receipt = {
            "cap": cap,
            "returncode": int(returncode),
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
            "report": report,
        }
        attempts.append(receipt)

        if returncode == 0:
            selected_report = _validate_probe_report(
                report_path,
                candidate,
            )
            selected_payload = candidate
            break
        can_fallback = _probe_failure_allows_lower_cap(
            report,
            lock_payload=candidate,
            report_path=report_path,
        )
        if not can_fallback or index == len(_PROVIDER_CAP_FALLBACKS) - 1:
            raise RuntimeError(
                f"global{cap} provider load probe failed and cannot be "
                "admitted as a lower-cap formal campaign"
            )

    if selected_payload is None or selected_report is None:
        raise RuntimeError("provider load probe exhausted every formal cap")
    selected_cap = int(selected_payload["campaign_provider_max_inflight"])
    runtime_spec, selected_payload = _runtime_spec_for_provider_cap(
        spec,
        selected_payload,
        cap=selected_cap,
    )
    locked_probe = dict(selected_payload["provider_probe"])
    final_receipt = attempts[-1]
    locked_probe.update(
        {
            "passed": True,
            "selected_cap": selected_cap,
            "report_path": str(final_receipt["path"]),
            "report_sha256": str(final_receipt["sha256"]),
        }
    )
    selected_payload["provider_probe"] = locked_probe
    selected_payload["provider_probe_attempts"] = attempts
    selected_payload["provider_probe_receipt"] = final_receipt
    return runtime_spec, selected_payload, selected_report


_HAPPY_TRANSITIONS = {
    "PENDING": "PREFLIGHT",
    "PREFLIGHT": "TRAINING",
    "TRAINING": "TRAIN_COMPLETED",
    "TRAIN_COMPLETED": "FROZEN",
    "FROZEN": "FROZEN_VERIFIED",
    "FROZEN_VERIFIED": "TESTING",
    "TESTING": "COMPLETED",
}


def _initialize_lane(
    lane_root: Path,
    seed: int,
    *,
    campaign_run_id: str,
    campaign_lock_digest: str,
) -> Path:
    lane_root.mkdir(parents=False, exist_ok=False)
    state_path = lane_root / "lane_state.json"
    now = time.time()
    _write_json_atomic(
        state_path,
        {
            "schema_version": 1,
            "campaign_run_id": campaign_run_id,
            "campaign_lock_digest": campaign_lock_digest,
            "seed": seed,
            "state": "PENDING",
            "transitions": [{"state": "PENDING", "at_unix": now}],
            "updated_at_unix": now,
        },
        overwrite=False,
    )
    return state_path


def _transition_lane(
    state_path: Path,
    state: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _read_json(state_path, "seed lane state")
    previous = str(payload.get("state", ""))
    expected = _HAPPY_TRANSITIONS.get(previous)
    if state != "FAILED" and expected != state:
        raise RuntimeError(
            f"invalid seed lane transition {previous!r} -> {state!r}; "
            f"expected {expected!r}"
        )
    if state == "FAILED" and previous in {"COMPLETED", "FAILED"}:
        raise RuntimeError(f"cannot fail terminal seed lane state {previous!r}")
    now = time.time()
    transitions = list(payload.get("transitions") or [])
    row: dict[str, Any] = {"state": state, "at_unix": now}
    if evidence:
        row["evidence"] = dict(evidence)
    transitions.append(row)
    payload.update(
        {
            "state": state,
            "transitions": transitions,
            "updated_at_unix": now,
        }
    )
    if evidence:
        payload["evidence"] = dict(evidence)
    _write_json_atomic(state_path, payload, overwrite=True)
    return payload


def _assert_lock_unchanged(lock_path: Path, expected_digest: str) -> None:
    if not lock_path.is_file() or _sha256_file(lock_path) != expected_digest:
        raise RuntimeError("campaign_lock.json changed after campaign start")


def _run_method_command(
    spec: CampaignSpec,
    *,
    phase: str,
    seed: int,
    output_dir: Path,
    campaign_lock: Path,
    source_run: Path | None = None,
    resume_source_run: Path | None = None,
) -> list[str]:
    command = [
        str(spec.python),
        "-m",
        "experiments.baselines.run_method",
        "--method",
        spec.method,
        "--phase",
        phase,
        "--seed",
        str(seed),
        "--train-manifest",
        str(spec.train_manifest),
        "--validation-manifest",
        str(spec.validation_manifest),
        "--test-manifest",
        str(spec.test_manifest),
        "--config",
        str(spec.config),
        "--output-dir",
        str(output_dir),
        "--campaign-lock",
        str(campaign_lock),
    ]
    if phase == "train":
        if resume_source_run is not None:
            command.extend(["--resume-source-run", str(resume_source_run)])
    elif phase == "test":
        if source_run is None:
            raise ValueError("test command requires its own seed train source")
        command.extend(["--source-run", str(source_run)])
    else:
        raise ValueError(f"unsupported campaign phase: {phase!r}")
    return command


def _validate_train_output(
    train_root: Path, seed: int
) -> tuple[FrozenArtifact, str, dict[str, Any]]:
    completion = _read_json(train_root / "completion.json", "train completion")
    report = _read_json(train_root / "report.json", "train report")
    if completion.get("passed") is not True or completion.get("phase") != "train":
        raise RuntimeError(f"seed {seed} train did not complete successfully")
    if report.get("passed") is not True or int(
        _read_json(train_root / "run_manifest.json", "train run manifest").get(
            "run_seed", -1
        )
    ) != seed:
        raise RuntimeError(f"seed {seed} train report/identity is invalid")
    frozen = FrozenArtifact.load(train_root / "frozen")
    assert_frozen_unchanged(frozen)
    if int(frozen.metadata.get("run_seed", -1)) != seed:
        raise RuntimeError(f"seed {seed} frozen artifact has wrong seed identity")
    if dict(report.get("frozen") or {}).get("digest") != frozen.digest:
        raise RuntimeError(f"seed {seed} train report names the wrong frozen digest")
    return frozen, str(completion.get("run_id", "")), report


def _validate_test_output(
    test_root: Path,
    train_root: Path,
    frozen: FrozenArtifact,
    seed: int,
) -> dict[str, Any]:
    completion = _read_json(test_root / "completion.json", "test completion")
    report = _read_json(test_root / "test_report.json", "test report")
    if completion.get("passed") is not True or completion.get("phase") != "test":
        raise RuntimeError(f"seed {seed} test did not complete successfully")
    run_manifest = _read_json(test_root / "run_manifest.json", "test run manifest")
    if int(run_manifest.get("run_seed", -1)) != seed or report.get("passed") is not True:
        raise RuntimeError(f"seed {seed} test report/identity is invalid")
    protocol = dict(report.get("protocol") or {})
    if Path(str(protocol.get("source_run", ""))).resolve() != train_root.resolve():
        raise RuntimeError(f"seed {seed} test did not read its own train lane")
    if dict(report.get("frozen") or {}).get("digest") != frozen.digest:
        raise RuntimeError(f"seed {seed} test names the wrong frozen digest")
    assert_frozen_unchanged(frozen)
    return report


def _failure_payload(output_root: Path, returncode: int) -> dict[str, Any]:
    failure_path = output_root / "failure.json"
    failure = _read_json(failure_path, "phase failure") if failure_path.is_file() else {}
    return {
        "returncode": int(returncode),
        "failure_kind": str(failure.get("failure_kind") or "protocol_failure"),
        "error_type": str(failure.get("error_type") or "SubprocessFailure"),
        "error": str(failure.get("error") or "phase command failed"),
        "failure_path": str(failure_path) if failure_path.is_file() else None,
    }


def _safe_error(exc: BaseException) -> str:
    text = sanitize_error_text(exc)
    live_key = os.environ.get("MODEL_API_KEY", "").strip()
    return text.replace(live_key, "[REDACTED]") if live_key else text


def _resource_usage(
    train_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> dict[str, Any]:
    training = dict(train_report.get("training_cost") or {})
    replay = dict(train_report.get("train_replay_cost") or {})
    testing = dict(test_report.get("test_cost") or {})
    train_usage = dict(training.get("usage") or {})
    replay_usage = dict(replay.get("usage") or {})
    test_usage = dict(testing.get("usage") or {})

    def role_total(usage: Mapping[str, Any], field: str) -> int:
        return sum(
            int(dict(usage.get(role) or {}).get(field, 0))
            for role in ("target", "evolution")
        )

    def episode_actions(cost: Mapping[str, Any]) -> int:
        return int(cost.get("environment_actions", 0))

    api_costs = [
        training.get("api_cost"),
        replay.get("api_cost", 0.0),
        testing.get("api_cost"),
    ]
    priced = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in api_costs
    )
    usage_segments = (train_usage, replay_usage, test_usage)
    logical_calls = sum(role_total(usage, "calls") for usage in usage_segments)
    evidence_segments = (
        dict(training.get("provider_evidence") or {}),
        dict(replay.get("provider_evidence") or {}),
        dict(testing.get("provider_evidence") or {}),
    )
    segment_attempts = tuple(
        int(evidence.get("application_attempts", role_total(usage, "calls")))
        for evidence, usage in zip(evidence_segments, usage_segments, strict=True)
    )
    segment_physical = tuple(
        int(evidence.get("physical_provider_calls", role_total(usage, "calls")))
        for evidence, usage in zip(evidence_segments, usage_segments, strict=True)
    )
    provider_attempts = sum(segment_attempts)
    physical_provider_calls = sum(segment_physical)
    provider_retries = sum(
        int(
            evidence.get(
                "provider_retries",
                max(0, attempts - physical),
            )
        )
        for evidence, attempts, physical in zip(
            evidence_segments,
            segment_attempts,
            segment_physical,
            strict=True,
        )
    )

    def provider_total(field: str) -> int:
        return sum(int(evidence.get(field, 0)) for evidence in evidence_segments)

    return {
        "api_calls": provider_attempts,
        "logical_api_calls": logical_calls,
        "provider_retries": provider_retries,
        "physical_provider_calls": physical_provider_calls,
        "cached_provider_calls": provider_total("cached_provider_calls"),
        "tokens": (
            role_total(train_usage, "prompt_tokens")
            + role_total(train_usage, "completion_tokens")
            + role_total(replay_usage, "prompt_tokens")
            + role_total(replay_usage, "completion_tokens")
            + role_total(test_usage, "prompt_tokens")
            + role_total(test_usage, "completion_tokens")
        ),
        "environment_actions": (
            episode_actions(dict(training.get("train_episodes") or {}))
            + episode_actions(dict(training.get("validation_episodes") or {}))
            + episode_actions(dict(replay.get("episodes") or {}))
            + episode_actions(dict(testing.get("episodes") or {}))
        ),
        "provider_queue_wait_ms": provider_total("provider_queue_wait_ms"),
        "provider_service_latency_ms": provider_total(
            "provider_service_latency_ms"
        ),
        "logical_call_latency_ms": provider_total("logical_call_latency_ms"),
        "retry_backoff_ms": provider_total("retry_backoff_ms"),
        "api_cost": sum(float(value) for value in api_costs) if priced else None,
        "api_cost_unpriced": not priced,
    }


_PROVIDER_TIME_FIELDS = (
    "provider_queue_wait_ms",
    "provider_service_latency_ms",
    "logical_call_latency_ms",
    "retry_backoff_ms",
)


def _provider_sidecar_summary(path: Path) -> dict[str, Any] | None:
    """Summarize actual HTTP evidence; cached imports are not physical calls."""

    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not a mapping")
            rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"provider sidecar is unreadable: {path}") from exc

    physical = [row for row in rows if not bool(row.get("cache_reused", False))]
    cached = [row for row in rows if bool(row.get("cache_reused", False))]

    def non_negative(row: Mapping[str, Any], field: str, default: int = 0) -> int:
        value = row.get(field, default)
        if isinstance(value, bool):
            raise RuntimeError(f"provider sidecar {field} is not numeric: {path}")
        number = int(value)
        if number < 0:
            raise RuntimeError(f"provider sidecar {field} is negative: {path}")
        return number

    application_attempts = sum(
        non_negative(row, "application_attempts", 1) for row in physical
    )
    summary = {
        "evidence_path": str(path.resolve()),
        "evidence_sha256": _sha256_file(path),
        "logical_api_calls": len(physical),
        "api_calls": application_attempts,
        "physical_provider_calls": len(physical),
        "provider_retries": max(0, application_attempts - len(physical)),
        "cached_provider_calls": len(cached),
        "cached_source_application_attempts": sum(
            non_negative(
                dict(row.get("source_provider_evidence") or {}),
                "application_attempts",
            )
            for row in cached
        ),
        "tokens": sum(
            non_negative(row, "prompt_tokens")
            + non_negative(row, "completion_tokens")
            for row in physical
        ),
    }
    for field in _PROVIDER_TIME_FIELDS:
        summary[field] = sum(non_negative(row, field) for row in physical)
    return summary


def _phase_provider_sidecar_summaries(phase_dir: Path) -> list[dict[str, Any]]:
    """Return every disjoint provider sidecar owned by one phase tree."""

    if not phase_dir.is_dir():
        return []
    return [
        summary
        for summary in (
            _provider_sidecar_summary(path)
            for path in sorted(phase_dir.rglob("provider_calls.jsonl"))
        )
        if summary is not None
    ]


def _phase_episode_action_summary(phase_dir: Path) -> dict[str, Any]:
    """Bind exact step journals when present, else the legacy episode lower bound."""

    if not phase_dir.is_dir():
        return {
            "episodes": 0,
            "environment_actions": 0,
            "evidence": [],
            "measurement_complete": False,
            "source": "missing_phase_directory",
        }
    paths = sorted({
        *phase_dir.rglob("common_episodes.jsonl"),
        *phase_dir.rglob("partial_common_episodes.jsonl"),
    })
    episodes = []
    evidence: list[dict[str, Any]] = []
    for path in paths:
        rows = load_episodes(path)
        episodes.extend(rows)
        evidence.append({
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "episodes": len(rows),
            "environment_actions": sum(
                int(episode.environment_actions) for episode in rows
            ),
        })
    if any(int(episode.environment_actions) < 0 for episode in episodes):
        raise RuntimeError(
            f"failed Train attempt has negative environment actions: {phase_dir}"
        )
    journal_paths = sorted(
        phase_dir.rglob("attempt_environment_actions/*.jsonl")
    )
    if journal_paths:
        action_ids: set[str] = set()
        journal_evidence: list[dict[str, Any]] = []
        for path in journal_paths:
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"failed Train action journal is unreadable: {path}"
                ) from exc
            if not rows or not all(isinstance(row, dict) for row in rows):
                raise RuntimeError(
                    f"failed Train action journal has no valid header: {path}"
                )
            header = rows[0]
            rollout_id = str(header.get("rollout_id", ""))
            task_id = str(header.get("episode_task_id", ""))
            if (
                header.get("schema_version") != 1
                or header.get("event") != "action_journal_started"
                or not rollout_id
                or not task_id
                or not str(header.get("actual_gamefile", "")).strip()
            ):
                raise RuntimeError(
                    f"failed Train action journal header is invalid: {path}"
                )
            for step_index, row in enumerate(rows[1:]):
                identity = {
                    "rollout_id": rollout_id,
                    "episode_task_id": task_id,
                    "step_index": step_index,
                }
                expected_id = hashlib.sha256(
                    json.dumps(
                        identity, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                action_id = str(row.get("action_id", ""))
                action = row.get("action")
                if (
                    row.get("schema_version") != 1
                    or row.get("event") != "environment_action"
                    or row.get("rollout_id") != rollout_id
                    or row.get("episode_task_id") != task_id
                    or int(row.get("step_index", -1)) != step_index
                    or action_id != expected_id
                    or action_id in action_ids
                    or not isinstance(action, str)
                    or not action.strip()
                ):
                    raise RuntimeError(
                        f"failed Train action journal row is invalid: {path}"
                    )
                action_ids.add(action_id)
            journal_evidence.append({
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "episodes": 1,
                "environment_actions": len(rows) - 1,
                "evidence_kind": "durable_environment_step_journal",
            })
        return {
            "episodes": len(journal_paths),
            "environment_actions": len(action_ids),
            "evidence": journal_evidence,
            "measurement_complete": True,
            "source": "durable_environment_step_journals",
        }
    return {
        "episodes": len(episodes),
        "environment_actions": sum(
            int(episode.environment_actions) for episode in episodes
        ),
        "evidence": evidence,
        "measurement_complete": False,
        "source": "persisted_episode_lower_bound",
    }


def _add_provider_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    fields = (
        "logical_api_calls",
        "api_calls",
        "physical_provider_calls",
        "provider_retries",
        "cached_provider_calls",
        "cached_source_application_attempts",
        "tokens",
        *_PROVIDER_TIME_FIELDS,
    )
    return {
        field: sum(int(summary.get(field, 0)) for summary in summaries)
        for field in fields
    }


def _cost_accounting(
    *,
    seed: int,
    resume_sources: Sequence[Path],
    failed_attempt_sources: Sequence[Path] | None = None,
    train_root: Path,
    test_root: Path,
    committed: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the four R1 §8.6 cost views without manufacturing precision."""

    current_sidecars = [
        *(_phase_provider_sidecar_summaries(train_root / "train")),
        *(_phase_provider_sidecar_summaries(train_root / "train_eval")),
        *(_phase_provider_sidecar_summaries(test_root / "test")),
    ]
    if current_sidecars:
        current_actual = _add_provider_summaries(current_sidecars)
        current_source = "provider_sidecars"
        expected_logical = (
            current_actual["physical_provider_calls"]
            + current_actual["cached_provider_calls"]
        )
        reconciliations = {
            "api_calls": current_actual["api_calls"],
            "logical_api_calls": expected_logical,
            "provider_retries": current_actual["provider_retries"],
            "physical_provider_calls": current_actual["physical_provider_calls"],
            "cached_provider_calls": current_actual["cached_provider_calls"],
            **{
                field: current_actual[field] for field in _PROVIDER_TIME_FIELDS
            },
        }
        for field, observed in reconciliations.items():
            if int(committed.get(field, -1)) != int(observed):
                raise RuntimeError(
                    f"seed {seed} provider sidecar does not reconcile with "
                    f"successful reports for {field}"
                )
    else:
        # Deterministic injected tests and legacy reports may have no sidecar.
        # The successful reports remain authoritative for committed logical cost.
        current_actual = {
            field: int(committed.get(field, 0))
            for field in (
                "logical_api_calls",
                "api_calls",
                "physical_provider_calls",
                "provider_retries",
                "cached_provider_calls",
                "tokens",
                *_PROVIDER_TIME_FIELDS,
            )
        }
        current_actual["cached_source_application_attempts"] = 0
        current_source = "successful_phase_reports"

    resume_roots = [Path(source).resolve() for source in resume_sources]
    failed_roots = (
        [Path(source).resolve() for source in failed_attempt_sources]
        if failed_attempt_sources is not None
        else list(resume_roots)
    )
    if len(set(failed_roots)) != len(failed_roots):
        raise RuntimeError("failed Train attempt sources contain duplicates")
    if not set(resume_roots).issubset(set(failed_roots)):
        raise RuntimeError("resume source is not a charged failed Train attempt")

    summaries_by_failed_root: dict[Path, list[dict[str, Any]]] = {}
    actions_by_failed_root: dict[Path, dict[str, Any]] = {}
    failed_attempt_summaries: list[dict[str, Any]] = []
    for failed_root in failed_roots:
        source_summaries = [
            *(_phase_provider_sidecar_summaries(failed_root / "train")),
            *(_phase_provider_sidecar_summaries(failed_root / "train_eval")),
        ]
        if not source_summaries:
            raise RuntimeError(
                "failed Train attempt has no train/train_eval provider sidecar: "
                f"{failed_root}"
            )
        summaries_by_failed_root[failed_root] = source_summaries
        failed_attempt_summaries.extend(source_summaries)
        actions_by_failed_root[failed_root] = _phase_episode_action_summary(
            failed_root / "train"
        )

    resume_parent_summaries = [
        summary
        for resume_root in resume_roots
        for summary in summaries_by_failed_root[resume_root]
    ]

    actual_provider = _add_provider_summaries(
        [current_actual, *failed_attempt_summaries]
    )
    failed_observed_actions = sum(
        int(summary["environment_actions"])
        for summary in actions_by_failed_root.values()
    )
    committed_actions = int(committed.get("environment_actions", 0))
    failed_actions_complete = all(
        summary.get("measurement_complete") is True
        for summary in actions_by_failed_root.values()
    )
    committed_method = dict(committed)
    committed_method["api_calls"] = int(committed_method["logical_api_calls"])
    committed_method.pop("physical_provider_calls", None)
    committed_method.pop("cached_provider_calls", None)
    committed_method["provider_retries"] = 0
    committed_method["retry_backoff_ms"] = 0
    if int(current_actual.get("provider_retries", 0)) > 0:
        # Per-call evidence aggregates base and retry attempt latency, so those
        # wall-time components cannot be assigned to algorithmic cost exactly.
        for field in (
            "provider_queue_wait_ms",
            "provider_service_latency_ms",
            "logical_call_latency_ms",
        ):
            committed_method[field] = None
    retry_provider = {
        "api_calls": actual_provider["provider_retries"],
        "provider_retries": actual_provider["provider_retries"],
        "retry_backoff_ms": actual_provider["retry_backoff_ms"],
    }
    no_retries = actual_provider["provider_retries"] == 0
    committed_complete = (
        not failed_roots and int(current_actual.get("provider_retries", 0)) == 0
    )
    zero_replay = {
        "api_calls": 0,
        "logical_api_calls": 0,
        "tokens": 0,
        "environment_actions": 0,
        "api_cost": 0.0,
    }
    if not resume_parent_summaries:
        replay: dict[str, Any] = {
            **zero_replay,
            "measurement_complete": True,
            "reason": "lane did not resume",
        }
    else:
        replay = {
            "api_calls": None,
            "logical_api_calls": None,
            "tokens": None,
            "environment_actions": None,
            "api_cost": None,
            "measurement_complete": False,
            "reason": (
                "provider sidecars do not identify which post-checkpoint current "
                "calls replay parent work; no replay amount is guessed"
            ),
            "parent_provider_costs": resume_parent_summaries,
            "current_cached_provider_calls": current_actual.get(
                "cached_provider_calls", 0
            ),
            "current_cached_source_application_attempts": current_actual.get(
                "cached_source_application_attempts", 0
            ),
        }

    actual = {
        **actual_provider,
        "environment_actions": (
            committed_actions + failed_observed_actions
            if not failed_attempt_summaries or failed_actions_complete
            else None
        ),
        "observed_environment_actions_lower_bound": (
            committed_actions + failed_observed_actions
        ),
        "failed_attempt_observed_environment_actions": failed_observed_actions,
        "api_cost": (
            committed.get("api_cost")
            if not failed_attempt_summaries
            and int(current_actual.get("provider_retries", 0)) == 0
            else None
        ),
        "measurement_complete": not failed_attempt_summaries
        and int(current_actual.get("provider_retries", 0)) == 0,
        "current_attempt_source": current_source,
        "includes_failed_attempts": bool(failed_attempt_summaries),
        "failed_attempt_count": len(failed_roots),
        "includes_resume_parent": bool(resume_parent_summaries),
        "resume_parent_count": len(resume_roots),
        "unmeasured_parent_environment_actions": bool(failed_attempt_summaries)
        and not failed_actions_complete,
        "unpriced": bool(committed.get("api_cost_unpriced", True))
        or bool(failed_attempt_summaries),
    }
    return {
        "schema_version": 1,
        "seed": int(seed),
        "committed_method_cost": {
            **committed_method,
            "measurement_complete": committed_complete,
            "semantics": (
                "successful Train/Freeze/Test logical method work"
                if not failed_roots
                else (
                    "continuation+Test logical work only; checkpoint-prefix usage "
                    "cannot be separated from the failed parent sidecar"
                    if resume_roots
                    else "successful fresh Train/Freeze/Test logical work only; "
                    "the discarded infrastructure-failed prefix is charged to "
                    "actual campaign cost"
                )
            ),
        },
        "infrastructure_retry_overhead": {
            **retry_provider,
            "tokens": 0 if no_retries else None,
            "environment_actions": 0 if no_retries else None,
            "api_cost": 0.0 if no_retries else None,
            "provider_queue_wait_ms": 0 if no_retries else None,
            "provider_service_latency_ms": 0 if no_retries else None,
            "logical_call_latency_ms": 0 if no_retries else None,
            "measurement_complete": no_retries,
            "semantics": (
                "zero exact overhead because no provider retry occurred"
                if no_retries
                else "proven extra provider attempts and backoff only; "
                "failed-attempt tokens, queue/service time and price are not "
                "separately observable"
            ),
        },
        "resume_replay_overhead": replay,
        "actual_campaign_cost": actual,
        "evidence": {
            "current_provider_sidecars": [
                {
                    "path": summary.get("evidence_path"),
                    "sha256": summary.get("evidence_sha256"),
                }
                for summary in current_sidecars
            ],
            "failed_attempt_provider_sidecars": [
                {
                    "path": failed_summary.get("evidence_path"),
                    "sha256": failed_summary.get("evidence_sha256"),
                }
                for failed_summary in failed_attempt_summaries
            ],
            "failed_attempt_episode_sidecars": [
                evidence
                for failed_root in failed_roots
                for evidence in actions_by_failed_root[failed_root]["evidence"]
            ],
            "failed_attempt_action_measurement": {
                str(failed_root): {
                    "source": actions_by_failed_root[failed_root]["source"],
                    "measurement_complete": actions_by_failed_root[failed_root][
                        "measurement_complete"
                    ],
                    "environment_actions": actions_by_failed_root[failed_root][
                        "environment_actions"
                    ],
                }
                for failed_root in failed_roots
            },
            "resume_parent_provider_sidecars": [
                {
                    "path": resume_summary.get("evidence_path"),
                    "sha256": resume_summary.get("evidence_sha256"),
                }
                for resume_summary in resume_parent_summaries
            ],
        },
    }


def _run_lane(
    spec: CampaignSpec,
    *,
    seed: int,
    lane_root: Path,
    state_path: Path,
    campaign_lock: Path,
    lock_digest: str,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_root = lane_root / "train"
    test_root = lane_root / "test"
    try:
        _transition_lane(state_path, "PREFLIGHT")
        _assert_lock_unchanged(campaign_lock, lock_digest)
        if train_root.exists() or test_root.exists():
            raise FileExistsError("seed lane phase output already exists")

        _transition_lane(state_path, "TRAINING")
        attempts_path = lane_root / "train_attempts.json"
        attempts: list[dict[str, Any]] = []
        resume_source: Path | None = None
        resume_lineage: list[Path] = []
        for attempt_number in range(1, _MAX_TRAIN_ATTEMPTS + 1):
            train_root = (
                lane_root / "train"
                if attempt_number == 1
                else lane_root / f"train_attempt_{attempt_number:03d}"
            )
            attempt_started = time.time()
            train_command = _run_method_command(
                spec,
                phase="train",
                seed=seed,
                output_dir=train_root,
                campaign_lock=campaign_lock,
                resume_source_run=resume_source,
            )
            train_code = command_runner(
                train_command,
                cwd=spec.repo_root,
                log_path=(
                    lane_root / "train.log"
                    if attempt_number == 1
                    else lane_root / f"train_attempt_{attempt_number:03d}.log"
                ),
            )
            row: dict[str, Any] = {
                "attempt": attempt_number,
                "output_dir": str(train_root),
                "resume_source_run": (
                    str(resume_source) if resume_source is not None else None
                ),
                "returncode": int(train_code),
                "started_at_unix": attempt_started,
                "completed_at_unix": time.time(),
            }
            if train_code == 0:
                row["status"] = "completed"
                attempts.append(row)
                _write_json_atomic(
                    attempts_path,
                    {"schema_version": 1, "seed": seed, "attempts": attempts},
                    overwrite=attempts_path.exists(),
                )
                break

            evidence = _failure_payload(train_root, train_code)
            row.update({"status": "failed", **evidence})
            attempts.append(row)
            _write_json_atomic(
                attempts_path,
                {"schema_version": 1, "seed": seed, "attempts": attempts},
                overwrite=attempts_path.exists(),
            )
            can_resume = (
                evidence["failure_kind"] == "infrastructure_failure"
                and attempt_number < _MAX_TRAIN_ATTEMPTS
            )
            if not can_resume:
                _transition_lane(state_path, "FAILED", evidence=evidence)
                return {
                    "seed": seed,
                    "passed": False,
                    "state": "FAILED",
                    "phase": "train",
                    "train_attempts": attempts,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    **evidence,
                }
            _assert_lock_unchanged(campaign_lock, lock_digest)
            resume_source = train_root
            resume_lineage.append(train_root)
        else:  # pragma: no cover - loop always breaks or returns
            raise AssertionError("train-attempt loop exhausted without a result")

        _transition_lane(state_path, "TRAIN_COMPLETED")
        _transition_lane(state_path, "FROZEN")
        frozen, train_run_id, train_report = _validate_train_output(train_root, seed)
        _assert_lock_unchanged(campaign_lock, lock_digest)
        _transition_lane(
            state_path,
            "FROZEN_VERIFIED",
            evidence={"frozen_digest": frozen.digest},
        )

        _transition_lane(state_path, "TESTING")
        test_command = _run_method_command(
            spec,
            phase="test",
            seed=seed,
            output_dir=test_root,
            campaign_lock=campaign_lock,
            source_run=train_root,
        )
        test_code = command_runner(
            test_command, cwd=spec.repo_root, log_path=lane_root / "test.log"
        )
        if test_code != 0:
            evidence = _failure_payload(test_root, test_code)
            _transition_lane(state_path, "FAILED", evidence=evidence)
            return {
                "seed": seed,
                "passed": False,
                "state": "FAILED",
                "phase": "test",
                "frozen_digest": frozen.digest,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                **evidence,
            }

        test_report = _validate_test_output(
            test_root, train_root, frozen, seed
        )
        _assert_lock_unchanged(campaign_lock, lock_digest)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        resource_usage = _resource_usage(train_report, test_report)
        cost_accounting = _cost_accounting(
            seed=seed,
            resume_sources=resume_lineage,
            train_root=train_root,
            test_root=test_root,
            committed=resource_usage,
        )
        _transition_lane(
            state_path,
            "COMPLETED",
            evidence={
                "frozen_digest": frozen.digest,
                "seed_elapsed_ms": elapsed_ms,
            },
        )
        return {
            "seed": seed,
            "passed": True,
            "state": "COMPLETED",
            "train_root": str(train_root),
            "test_root": str(test_root),
            "train_run_id": train_run_id,
            "train_attempts": attempts,
            "test_run_id": str(
                _read_json(test_root / "completion.json", "test completion").get(
                    "run_id", ""
                )
            ),
            "frozen_digest": frozen.digest,
            "seed_elapsed_ms": elapsed_ms,
            "resource_usage": resource_usage,
            "cost_accounting": cost_accounting,
            "test_report": test_report,
        }
    except Exception as exc:
        evidence = {
            "failure_kind": "protocol_failure",
            "error_type": type(exc).__name__,
            "error": _safe_error(exc),
        }
        try:
            current = _read_json(state_path, "seed lane state").get("state")
            if current not in {"COMPLETED", "FAILED"}:
                _transition_lane(state_path, "FAILED", evidence=evidence)
        except Exception:
            pass
        return {
            "seed": seed,
            "passed": False,
            "state": "FAILED",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            **evidence,
        }


def _numeric_test_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    candidates = dict(report.get("comparison_metrics") or {})
    if not candidates:
        candidates = dict(report.get("effectiveness") or {})
    metrics: dict[str, float] = {}
    for key, value in candidates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            metrics[str(key)] = number
    return metrics


def _mean_std(lanes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_seed = [
        _numeric_test_metrics(dict(lane.get("test_report") or {})) for lane in lanes
    ]
    if not by_seed:
        return {}
    keys = set(by_seed[0])
    for metrics in by_seed[1:]:
        keys.intersection_update(metrics)
    aggregate: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        values = [metrics[key] for metrics in by_seed]
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "std_ddof": 1 if len(values) > 1 else 0,
            "values_by_seed": {
                str(lane["seed"]): value for lane, value in zip(lanes, values, strict=True)
            },
        }
    return aggregate


def _resource_summary(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not lanes:
        return {"totals": {}, "mean_std": {}}
    rows = [dict(lane.get("resource_usage") or {}) for lane in lanes]
    numeric_keys = {
        key
        for key in set.intersection(*(set(row) for row in rows))
        if all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
            for row in rows
        )
    }
    totals: dict[str, Any] = {
        key: sum(float(row[key]) for row in rows) for key in sorted(numeric_keys)
    }
    for key in (
        "api_calls",
        "logical_api_calls",
        "provider_retries",
        "physical_provider_calls",
        "cached_provider_calls",
        "tokens",
        "environment_actions",
        *_PROVIDER_TIME_FIELDS,
    ):
        if key in totals:
            totals[key] = int(totals[key])
    totals["api_cost_unpriced"] = any(
        bool(row.get("api_cost_unpriced", True)) for row in rows
    )
    if totals["api_cost_unpriced"]:
        totals["api_cost"] = None

    mean_std: dict[str, Any] = {}
    for key in sorted(numeric_keys):
        values = [float(row[key]) for row in rows]
        mean_std[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "std_ddof": 1 if len(values) > 1 else 0,
            "values_by_seed": {
                str(lane["seed"]): value
                for lane, value in zip(lanes, values, strict=True)
            },
        }
    return {"totals": totals, "mean_std": mean_std}


def _campaign_cost_summary(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = (
        "committed_method_cost",
        "infrastructure_retry_overhead",
        "resume_replay_overhead",
        "actual_campaign_cost",
    )
    result: dict[str, Any] = {"schema_version": 1}
    for category in categories:
        rows = [
            dict(dict(lane.get("cost_accounting") or {}).get(category) or {})
            for lane in lanes
        ]
        numeric_keys: set[str] = set.intersection(
            *(set(row) for row in rows)
        ) if rows else set()
        numeric_keys = {
            key
            for key in numeric_keys
            if all(
                isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
                and math.isfinite(float(row[key]))
                for row in rows
            )
        }
        totals: dict[str, Any] = {}
        mean_std: dict[str, Any] = {}
        for key in sorted(numeric_keys):
            values = [float(row[key]) for row in rows]
            total: int | float = sum(values)
            if all(isinstance(row[key], int) for row in rows):
                total = int(total)
            totals[key] = total
            mean_std[key] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "std_ddof": 1 if len(values) > 1 else 0,
                "values_by_seed": {
                    str(lane["seed"]): value
                    for lane, value in zip(lanes, values, strict=True)
                },
            }
        result[category] = {
            "measurement_complete": bool(rows)
            and all(row.get("measurement_complete") is True for row in rows),
            "totals": totals,
            "mean_std": mean_std,
            "by_seed": {
                str(lane["seed"]): row
                for lane, row in zip(lanes, rows, strict=True)
            },
        }
    return result


def _new_campaign_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"b3_campaign_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _run_campaign_impl(
    spec: CampaignSpec,
    *,
    campaign_run_id: str,
    command_runner: CommandRunner | None = None,
    source_inspector: SourceInspector | None = None,
    preflight_builder: PreflightBuilder | None = None,
) -> dict[str, Any]:
    """Execute one formal campaign and return its durable aggregate report."""

    command_runner = command_runner or _default_command_runner
    source_inspector = source_inspector or inspect_clean_source
    preflight_builder = preflight_builder or build_campaign_lock
    if spec.output_dir.exists():
        raise FileExistsError(spec.output_dir)

    source_state = source_inspector(spec.repo_root)
    lock_payload = preflight_builder(
        spec, spec.output_dir, campaign_run_id, source_state
    )
    if lock_payload.get("method") != spec.method:
        raise ValueError("campaign preflight returned a different method")
    if tuple(int(seed) for seed in lock_payload.get("seeds", [])) != spec.seeds:
        raise ValueError("campaign preflight returned different seed lanes")

    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    spec.output_dir.mkdir(exist_ok=False)
    spec, lock_payload, probe_report = _run_provider_probe_with_fallback(
        spec,
        lock_payload,
        command_runner=command_runner,
    )

    source_after_probe = source_inspector(spec.repo_root)
    if source_after_probe != source_state:
        raise RuntimeError("controller source changed during campaign preflight")
    lock_payload = dict(lock_payload)
    campaign_lock = spec.output_dir / "campaign_lock.json"
    _write_json_atomic(campaign_lock, lock_payload, overwrite=False)
    lock_digest = _sha256_file(campaign_lock)

    lane_paths: dict[int, tuple[Path, Path]] = {}
    for seed in spec.seeds:
        lane_root = spec.output_dir / f"seed_{seed}"
        lane_paths[seed] = (
            lane_root,
            _initialize_lane(
                lane_root,
                seed,
                campaign_run_id=campaign_run_id,
                campaign_lock_digest=lock_digest,
            ),
        )

    campaign_started_at_unix = time.time()
    campaign_started = time.perf_counter()
    lanes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=len(spec.seeds), thread_name_prefix="b3-seed-lane"
    ) as executor:
        futures = {
            executor.submit(
                _run_lane,
                spec,
                seed=seed,
                lane_root=lane_paths[seed][0],
                state_path=lane_paths[seed][1],
                campaign_lock=campaign_lock,
                lock_digest=lock_digest,
                command_runner=command_runner,
            ): seed
            for seed in spec.seeds
        }
        for future in as_completed(futures):
            lanes.append(future.result())
    campaign_makespan_ms = int((time.perf_counter() - campaign_started) * 1000)
    lanes.sort(key=lambda lane: int(lane["seed"]))

    _assert_lock_unchanged(campaign_lock, lock_digest)
    if source_inspector(spec.repo_root) != source_state:
        raise RuntimeError("controller source changed while formal campaign ran")
    for lane in lanes:
        if lane.get("passed") is True:
            frozen = FrozenArtifact.load(
                Path(str(lane["train_root"])) / "frozen"
            )
            assert_frozen_unchanged(frozen)
            if frozen.digest != lane.get("frozen_digest"):
                raise RuntimeError(
                    f"seed {lane['seed']} final frozen digest verification failed"
                )

    passed = len(lanes) == len(spec.seeds) and all(
        lane.get("passed") is True and lane.get("state") == "COMPLETED"
        for lane in lanes
    )
    completed_lanes = [lane for lane in lanes if lane.get("passed") is True]
    resource_summary = _resource_summary(completed_lanes)
    cost_summary = _campaign_cost_summary(completed_lanes)
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "passed": passed,
        "method": spec.method,
        "campaign_id": lock_payload["campaign_id"],
        "campaign_run_id": campaign_run_id,
        "campaign_root": str(spec.output_dir),
        "campaign_lock": str(campaign_lock),
        "campaign_lock_digest": lock_digest,
        "seeds": list(spec.seeds),
        "campaign_started_at_unix": campaign_started_at_unix,
        "campaign_completed_at_unix": time.time(),
        "campaign_makespan_ms": campaign_makespan_ms,
        "provider_probe": probe_report,
        "lanes": lanes,
        "completed_seeds": [int(lane["seed"]) for lane in completed_lanes],
        "failed_seeds": [
            int(lane["seed"]) for lane in lanes if lane.get("passed") is not True
        ],
        "test_metrics_mean_std": (
            _mean_std(completed_lanes) if len(completed_lanes) == len(spec.seeds) else {}
        ),
        "resource_usage": resource_summary,
        "cost_accounting": cost_summary,
        "all_frozen_digests_verified": passed,
    }
    _write_json_atomic(
        spec.output_dir / "campaign_report.json", report, overwrite=False
    )
    terminal_name = "completion.json" if passed else "campaign_failure.json"
    _write_json_atomic(
        spec.output_dir / terminal_name,
        {
            "schema_version": 1,
            "passed": passed,
            "campaign_run_id": campaign_run_id,
            "campaign_lock_digest": lock_digest,
            "campaign_report": "campaign_report.json",
            "completed_at_unix": time.time(),
        },
        overwrite=False,
    )
    return report


def run_campaign(
    spec: CampaignSpec,
    *,
    command_runner: CommandRunner | None = None,
    source_inspector: SourceInspector | None = None,
    preflight_builder: PreflightBuilder | None = None,
) -> dict[str, Any]:
    """Execute one formal campaign under the cross-method process lease."""

    campaign_run_id = _new_campaign_run_id()
    with _method_campaign_lease(spec.repo_root, owner=campaign_run_id):
        return _run_campaign_impl(
            spec,
            campaign_run_id=campaign_run_id,
            command_runner=command_runner,
            source_inspector=source_inspector,
            preflight_builder=preflight_builder,
        )


def _default_output_dir(repo_root: Path, campaign_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo_root
        / "runs"
        / "baselines"
        / campaign_id
        / _METHOD
        / f"formal_3seed_{stamp}_{os.getpid()}"
    ).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=_METHOD, choices=[_METHOD])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(_FORMAL_SEEDS))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--config", default="configs/baselines/b3_skillopt.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "debug-only override; formal value must match config worker_python "
            "without resolving symlinks"
        ),
    )
    args = parser.parse_args(argv)

    repo_root = REPO_ROOT.resolve()
    config_path = _resolve(repo_root, args.config)
    try:
        merged = {
            **dict(
                yaml.safe_load(
                    (repo_root / "configs" / "baselines" / "common.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                or {}
            ),
            **dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}),
        }
        campaign_id = str(merged.get("campaign_id", "formal"))
        phase_python = _select_phase_python(
            repo_root,
            str(merged.get("worker_python", "")),
            args.python,
        )
        output_dir = (
            _resolve(repo_root, args.output_dir)
            if args.output_dir
            else _default_output_dir(repo_root, campaign_id)
        )
        spec = CampaignSpec(
            method=args.method,
            seeds=tuple(args.seeds),
            train_manifest=_resolve(repo_root, args.train_manifest),
            validation_manifest=_resolve(repo_root, args.validation_manifest),
            test_manifest=_resolve(repo_root, args.test_manifest),
            config=config_path,
            output_dir=output_dir,
            python=phase_python,
            repo_root=repo_root,
        )
        report = run_campaign(spec)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.get("passed") is True else 1
    except Exception as exc:
        failure = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": _safe_error(exc),
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
