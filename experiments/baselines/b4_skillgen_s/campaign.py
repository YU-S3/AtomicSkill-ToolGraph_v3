"""Run the formal B4 SkillGen-S campaign with serialized seed lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from experiments.baselines.bootstrap_external import (
    load_lock,
    verify_key_files,
    verify_runtime_tree,
    worker_expected_distributions,
)
from experiments.baselines.common.formal_validation import (
    ALFWORLD_FORMAL_TASK_TYPES,
    verify_formal_manifest,
)
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.manifest import (
    TaskManifestSet,
    sha256_json,
    verify_disjoint,
)
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.runtime_python import resolve_formal_python
from experiments.baselines.common.runtime_python import verify_runtime_python_executable
from experiments.baselines.common.source_identity import (
    hash_code,
    sanitize_error_text,
)

from .progress_adapter import require_complete_train_coverage


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "b4_skillgen_s"
FORMAL_SEEDS = (42, 43, 44)
PROVIDER_CAPS = (16, 12, 8)
MAX_TRAIN_ATTEMPTS = 2

CommandRunner = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class CampaignSpec:
    train_manifest: Path
    test_manifest: Path
    supervision: Path
    config: Path
    output_dir: Path
    python: Path
    seeds: tuple[int, ...] = FORMAL_SEEDS
    repo_root: Path = REPO_ROOT
    controller_python: Path | None = None

    def __post_init__(self) -> None:
        if self.seeds != FORMAL_SEEDS:
            raise ValueError("formal B4 seeds must be exactly 42, 43, 44")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_config(spec: CampaignSpec) -> dict[str, Any]:
    common = yaml.safe_load(
        (spec.repo_root / "configs" / "baselines" / "common.yaml").read_text(
            encoding="utf-8"
        )
    )
    method = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    if not isinstance(common, dict) or not isinstance(method, dict):
        raise ValueError("B4 campaign configuration must be mappings")
    return {**common, **method}


def _formal_config_digest(config: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(config))
    normalized["run_seed"] = "<campaign-seed>"
    train = dict(normalized.get("train") or {})
    train["seed"] = "<campaign-seed>"
    normalized["train"] = train
    return sha256_json(normalized)


def inspect_clean_source(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "git inspection failed")
        return completed.stdout.strip()

    status = git(
        "status", "--porcelain", "--untracked-files=all", "--",
        "src", "experiments", "configs",
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "code_digest": hash_code(repo_root),
    }


@contextmanager
def _campaign_lease(repo_root: Path, *, owner: str) -> Iterator[Path]:
    path = repo_root / "runs" / "baselines" / ".formal_method_campaign.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    locked = False
    try:
        try:
            if os.name == "nt":  # pragma: no cover - formal runs execute in WSL
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write("\n")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(
                "another formal method campaign holds the repository lease"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"owner": owner, "pid": os.getpid()}) + "\n")
        handle.flush()
        yield path
    finally:
        try:
            if locked:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                if os.name == "nt":  # pragma: no cover
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def build_campaign_lock(
    spec: CampaignSpec,
    *,
    campaign_run_id: str,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    if source_state.get("dirty") is not False:
        raise RuntimeError("formal B4 requires a clean experiments/configs source tree")
    config = _load_config(spec)
    if (
        config.get("method") != METHOD_ID
        or config.get("protocol_profile") != "formal_v2"
        or config.get("experiment_kind") != "formal"
    ):
        raise ValueError("B4 campaign requires the formal SkillGen-S configuration")
    parallel = dict(config.get("parallel") or {})
    if (
        int(parallel.get("seed_lanes", 0)) != 1
        or int(parallel.get("episode_workers_per_seed", 0)) != 16
        or int(parallel.get("test_workers_per_seed", 0)) != 16
        or int(parallel.get("campaign_provider_max_inflight", 0)) != 16
        or parallel.get("mp_start_method") != "spawn"
    ):
        raise ValueError(
            "formal B4 campaign must serialize seeds, use global16, and use spawn"
        )
    resume = dict(config.get("resume") or {})
    if resume != {
        "enabled": True,
        "formal_boundary": "sampling_job",
        "extraction_replay_from_complete_corpus": True,
    }:
        raise ValueError("formal B4 sampling-job resume settings differ from the frozen protocol")
    transport = dict(config.get("provider_transport") or {})
    if transport != {
        "sdk_max_retries": 0,
        "application_retry_limit": 5,
        "retry_delays_seconds": [2, 5, 10, 20],
        "deterministic_jitter_ratio": 0.10,
    }:
        raise ValueError("formal B4 retry policy differs from the frozen protocol")
    probe = dict(config.get("provider_probe") or {})
    expected_probe = {
        "enabled": True,
        "concurrency": 16,
        "requests": 32,
        "max_completion_tokens": 256,
        "reasoning_effort": "high",
    }
    if probe != expected_probe:
        raise ValueError("formal B4 provider probe differs from the frozen protocol")
    model = ModelConfig.from_mapping(dict(config.get("model") or {}))
    model.validate_formal_identity()
    model.require_api_key()
    configured_python = resolve_formal_python(
        spec.repo_root, str(config.get("worker_python", ""))
    )
    if str(resolve_formal_python(spec.repo_root, spec.python)) != str(configured_python):
        raise ValueError("B4 campaign Python must exactly match worker_python")
    python_runtime = verify_runtime_python_executable(
        repo_root=spec.repo_root,
        expected_python=configured_python,
        require_venv=True,
        method=METHOD_ID,
        expected_distributions=worker_expected_distributions(METHOD_ID),
        expected_python_major_minor="3.9",
    )

    data_raw = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_raw:
        raise RuntimeError("ALFWORLD_DATA is not set")
    data_root = Path(data_raw).expanduser().resolve(strict=True)
    train = TaskManifestSet.load(spec.train_manifest)
    test = TaskManifestSet.load(spec.test_manifest)
    verify_disjoint(train, test)
    train_receipt = verify_formal_manifest(
        train, alfworld_data=data_root, role="train", profile="formal_v2"
    )
    test_receipt = verify_formal_manifest(
        test, alfworld_data=data_root, role="test", profile="formal_v2"
    )
    coverage, _ = require_complete_train_coverage(train, spec.supervision)

    external_lock = load_lock(
        spec.repo_root / "experiments" / "baselines" / "baseline_lock.yaml"
    )
    external_root = spec.repo_root / ".external" / "skillgen"
    verify_key_files(external_root, "skillgen", external_lock)
    runtime = verify_runtime_tree(external_root, "skillgen", external_lock)
    skillgen = dict(external_lock["skillgen"])
    campaign_id = str(config.get("campaign_id", "")).strip()
    if not campaign_id:
        raise ValueError("formal B4 campaign_id is empty")
    campaign_root = spec.output_dir.resolve()
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_run_id": campaign_run_id,
        "campaign_root": str(campaign_root),
        "method": METHOD_ID,
        "seeds": list(spec.seeds),
        "train_manifest_digest": train.digest,
        "validation_manifest_digest": None,
        "test_manifest_digest": test.digest,
        "manifests": {
            "train": {"path": str(spec.train_manifest), "preflight": train_receipt},
            "validation": None,
            "test": {"path": str(spec.test_manifest), "preflight": test_receipt},
        },
        "supervision_path": str(spec.supervision),
        "supervision_digest": coverage.label_sha256,
        "supervision_coverage": coverage.to_dict(),
        "controller_commit": str(source_state["commit"]),
        "controller_code_digest": str(source_state["code_digest"]),
        "controller_git": dict(source_state),
        "formal_config_digest": _formal_config_digest(config),
        "config_path": str(spec.config),
        "external_lock_digest": sha256_json(external_lock),
        "external_method_commit": str(skillgen["commit"]),
        "external_runtime_tree_digest": str(runtime["sha256"]),
        "model": model.model,
        "model_identity": model.to_wire(),
        "reasoning_effort": model.reasoning_effort,
        "seed_lanes": 1,
        "parallel": parallel,
        "campaign_provider_max_inflight": 16,
        "provider_gate_dir": str(campaign_root / "provider_gate" / "global_16"),
        "provider_probe_python": str(configured_python),
        "worker_python": str(configured_python),
        "worker_python_runtime": python_runtime,
        "controller_python": str(
            spec.controller_python or Path(os.path.abspath(sys.executable))
        ),
        "provider_probe": probe,
        "retry_policy": {
            "sdk_max_retries": 0,
            "attempts": 5,
            "delays": [2, 5, 10, 20],
            "jitter_ratio": 0.10,
        },
        "resume": resume,
        "created_at_unix": time.time(),
    }


def _default_runner(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
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


def _probe_command(spec: CampaignSpec, payload: Mapping[str, Any], root: Path) -> list[str]:
    model = dict(payload["model_identity"])
    probe = dict(payload["provider_probe"])
    retry = dict(payload["retry_policy"])
    return [
        str(payload["provider_probe_python"]),
        "-m", "experiments.baselines.b4_skillgen_s.provider_probe",
        "--output-dir", str(root),
        "--gate-dir", str(payload["provider_gate_dir"]),
        "--campaign-id", str(payload["campaign_id"]),
        "--run-id", str(payload["campaign_run_id"]) + "_probe",
        "--seed", str(spec.seeds[0]),
        "--base-url", str(model["base_url"]),
        "--model", str(model["model"]),
        "--api-key-env", str(model["api_key_env"]),
        "--reasoning-effort", str(model["reasoning_effort"]),
        "--concurrency", str(probe["concurrency"]),
        "--requests", str(probe["requests"]),
        "--max-completion-tokens", str(probe["max_completion_tokens"]),
        "--max-inflight", str(payload["campaign_provider_max_inflight"]),
        "--retry-delays-seconds", ",".join(str(v) for v in retry["delays"]),
        "--deterministic-jitter-ratio", str(retry["jitter_ratio"]),
    ]


def _payload_for_cap(
    spec: CampaignSpec,
    payload: Mapping[str, Any],
    *,
    cap: int,
) -> tuple[CampaignSpec, dict[str, Any]]:
    candidate = json.loads(json.dumps(dict(payload)))
    parallel = dict(candidate["parallel"])
    parallel.update({
        "episode_workers_per_seed": cap,
        "test_workers_per_seed": cap,
        "campaign_provider_max_inflight": cap,
    })
    probe = dict(candidate["provider_probe"])
    probe["concurrency"] = cap
    candidate.update({
        "parallel": parallel,
        "provider_probe": probe,
        "campaign_provider_max_inflight": cap,
        "provider_gate_dir": str(spec.output_dir / "provider_gate" / f"global_{cap}"),
    })
    if cap == 16:
        return spec, candidate
    method_config = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    if not isinstance(method_config, dict):
        raise ValueError("B4 method config must be a mapping")
    method_config = dict(method_config)
    method_config["parallel"] = parallel
    method_config["provider_probe"] = probe
    generated = spec.output_dir / f"campaign_config_global_{cap}.yaml"
    _write_yaml(generated, method_config)
    runtime_spec = replace(spec, config=generated)
    candidate["config_path"] = str(generated)
    candidate["formal_config_digest"] = _formal_config_digest(
        _load_config(runtime_spec)
    )
    return runtime_spec, candidate


def _validate_probe_report(
    report: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    report_path: Path,
) -> None:
    probe = dict(payload["provider_probe"])
    calls_path = Path(str(report.get("provider_calls_path", ""))).resolve()
    expected = {
        "probe_kind": "campaign_provider_load",
        "passed": True,
        "campaign_id": payload["campaign_id"],
        "run_id": str(payload["campaign_run_id"]) + "_probe",
        "model": payload["model"],
        "reasoning_effort": payload["reasoning_effort"],
        "concurrency": payload["campaign_provider_max_inflight"],
        "requests": probe["requests"],
        "max_completion_tokens": probe["max_completion_tokens"],
        "campaign_provider_max_inflight": payload["campaign_provider_max_inflight"],
        "logical_calls_recorded": probe["requests"],
        "completed_logical_calls": probe["requests"],
        "failed_provider_calls": 0,
        "exhausted_provider_calls": 0,
        "permanent_provider_errors": 0,
        "provider_evidence_complete": True,
    }
    failed = [key for key, value in expected.items() if report.get(key) != value]
    try:
        calls_path.relative_to(report_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("B4 provider probe evidence escaped its probe directory") from exc
    if (
        failed
        or not calls_path.is_file()
        or _sha256_file(calls_path) != report.get("provider_calls_sha256")
    ):
        raise ValueError("B4 provider probe report is invalid: " + ", ".join(failed))


def _run_probe_with_fallback(
    spec: CampaignSpec,
    payload: Mapping[str, Any],
    *,
    runner: Callable[..., int],
) -> tuple[CampaignSpec, dict[str, Any], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for cap in PROVIDER_CAPS:
        candidate_spec, candidate = _payload_for_cap(spec, payload, cap=cap)
        root = spec.output_dir / "provider_probe" / f"global_{cap}"
        code = runner(
            _probe_command(candidate_spec, candidate, root),
            cwd=spec.repo_root,
            log_path=spec.output_dir / f"provider_probe_global_{cap}.log",
        )
        report_path = root / "provider_load_probe.json"
        if not report_path.is_file():
            raise RuntimeError(f"B4 global{cap} probe produced no durable report")
        report = _read_json(report_path, "B4 provider probe report")
        receipt = {
            "cap": cap,
            "returncode": code,
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
            "report": report,
        }
        attempts.append(receipt)
        if code == 0:
            _validate_probe_report(report, payload=candidate, report_path=report_path)
            probe = dict(candidate["provider_probe"])
            probe.update({
                "passed": True,
                "selected_cap": cap,
                "report_path": str(report_path),
                "report_sha256": receipt["sha256"],
            })
            candidate["provider_probe"] = probe
            candidate["provider_probe_attempts"] = attempts
            candidate["provider_probe_receipt"] = receipt
            return candidate_spec, candidate, report
        fallback_allowed = bool(
            report.get("observer_error_type") is None
            and int(report.get("permanent_provider_errors", -1)) == 0
            and int(report.get("exhausted_provider_calls", 0)) > 0
            and int(report.get("exhausted_provider_calls", -1))
            == int(report.get("failed_provider_calls", -2))
            and int(report.get("logical_calls_recorded", -1))
            == int(dict(candidate["provider_probe"])["requests"])
        )
        if not fallback_allowed:
            raise RuntimeError(f"B4 global{cap} provider probe failed permanently")
    raise RuntimeError("B4 provider probe failed at global16, global12, and global8")


def _phase_command(
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
        str(spec.controller_python or Path(os.path.abspath(sys.executable))),
        "-m", "experiments.baselines.run_method",
        "--method", METHOD_ID,
        "--phase", phase,
        "--seed", str(seed),
        "--train-manifest", str(spec.train_manifest),
        "--test-manifest", str(spec.test_manifest),
        "--config", str(spec.config),
        "--output-dir", str(output_dir),
        "--campaign-lock", str(campaign_lock),
    ]
    if phase == "train":
        command.extend(["--supervision", str(spec.supervision)])
        if resume_source_run is not None:
            command.extend(["--resume-source-run", str(resume_source_run)])
    elif phase == "test":
        if source_run is None:
            raise ValueError("B4 Test command requires its own seed Train source")
        command.extend(["--source-run", str(source_run)])
    else:
        raise ValueError(f"unsupported B4 campaign phase: {phase}")
    return command


def _failure(root: Path, returncode: int) -> dict[str, Any]:
    path = root / "failure.json"
    payload = _read_json(path, "B4 phase failure") if path.is_file() else {}
    state_path = root / "run_state.json"
    state = _read_json(state_path, "B4 run state") if state_path.is_file() else {}
    interrupted = bool(
        state.get("state") == "running"
        and (root / "train" / "sampling" / "jobs").is_dir()
    )
    return {
        "returncode": int(returncode),
        "failure_kind": (
            "infrastructure_failure"
            if interrupted else str(payload.get("failure_kind") or "protocol_failure")
        ),
        "error_type": str(payload.get("error_type") or "SubprocessFailure"),
        "error": str(payload.get("error") or "phase subprocess failed"),
        "interrupted_with_sampling_checkpoints": interrupted,
    }


def _cost_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"B4 checkpoint cost has invalid {label}")
    return value


def _checkpoint_cost_record(path: Path) -> dict[str, Any]:
    """Hash, parse, and validate cost evidence from the same immutable bytes."""

    try:
        content = path.read_bytes()
        row = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"B4 sampling checkpoint is unreadable: {path}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"B4 sampling checkpoint must be an object: {path}")
    if len(path.stem) != 4 or not path.stem.isdigit():
        raise ValueError(f"B4 sampling checkpoint filename is invalid: {path.name}")
    job_index = _cost_int(row.get("job_index"), "job_index")
    manifest_index = _cost_int(row.get("manifest_index"), "manifest_index")
    sample_idx = _cost_int(row.get("sample_idx"), "sample_idx")
    sample_seed = _cost_int(row.get("sample_seed"), "sample_seed")
    task_id = row.get("task_id")
    if (
        job_index != int(path.stem)
        or sample_idx > 5
        or job_index != manifest_index * 6 + sample_idx
        or not isinstance(task_id, str)
        or not task_id
    ):
        raise ValueError(f"B4 sampling checkpoint row identity differs: {path}")
    failure_kind = str(row.get("failure_kind", ""))
    if failure_kind not in {"", "infrastructure_failure", "protocol_failure"}:
        raise ValueError(f"B4 sampling checkpoint failure_kind is invalid: {path}")
    logical_task_id = f"{task_id}::sample_{sample_idx}"

    actions = row.get("actions")
    if not isinstance(actions, list):
        raise ValueError(f"B4 sampling checkpoint lacks action evidence: {path}")
    action_turns: set[int] = set()
    for step_index, event in enumerate(actions):
        if not isinstance(event, dict):
            raise ValueError(f"B4 sampling checkpoint action evidence is invalid: {path}")
        turn = _cost_int(event.get("command_turn_index"), "command_turn_index")
        if (
            event.get("episode_task_id") != logical_task_id
            or event.get("step_index") != step_index
            or turn in action_turns
            or not isinstance(event.get("action"), str)
            or not isinstance(event.get("env_feedback"), str)
            or not isinstance(event.get("reward"), (int, float))
            or isinstance(event.get("reward"), bool)
            or not isinstance(event.get("done"), bool)
            or not isinstance(event.get("won"), bool)
            or not isinstance(event.get("admissible_before"), bool)
        ):
            raise ValueError(f"B4 sampling checkpoint action evidence differs: {path}")
        action_turns.add(turn)
    if "environment_actions" in row and _cost_int(
        row["environment_actions"], "environment_actions"
    ) != len(actions):
        raise ValueError(f"B4 sampling checkpoint action count differs: {path}")

    provider_calls = row.get("provider_calls")
    if not isinstance(provider_calls, list):
        raise ValueError(f"B4 sampling checkpoint lacks provider evidence: {path}")
    provider_costs: list[dict[str, Any]] = []
    provider_run_ids: set[str] = set()
    provider_run_seeds: set[int] = set()
    for event in provider_calls:
        if not isinstance(event, dict):
            raise ValueError(f"B4 sampling checkpoint provider evidence is invalid: {path}")
        status = event.get("status")
        call_id = event.get("call_id")
        if (
            status not in {"succeeded", "failed"}
            or not isinstance(call_id, str)
            or not call_id
            or event.get("method") != METHOD_ID
            or event.get("phase") != "train"
            or event.get("episode_task_id") != logical_task_id
            or event.get("role") != "target"
            or event.get("stage") != "sampling"
            or event.get("sample_seed") != sample_seed
            or not isinstance(event.get("run_id"), str)
            or not event.get("run_id")
            or not isinstance(event.get("model"), str)
            or not event.get("model")
            or not isinstance(event.get("reasoning_effort"), str)
            or not event.get("reasoning_effort")
            or event.get("requested_retry_limit") != 5
        ):
            raise ValueError(f"B4 sampling checkpoint provider identity differs: {path}")
        run_seed = _cost_int(event.get("run_seed"), "run_seed")
        provider_run_ids.add(str(event["run_id"]))
        provider_run_seeds.add(run_seed)
        application_attempts = _cost_int(
            event.get("application_attempts"), "application_attempts", minimum=1,
        )
        sdk_attempts = _cost_int(
            event.get("sdk_boundary_attempts"), "sdk_boundary_attempts", minimum=1,
        )
        prompt_tokens = _cost_int(event.get("prompt_tokens"), "prompt_tokens")
        completion_tokens = _cost_int(
            event.get("completion_tokens"), "completion_tokens"
        )
        raw_reasoning = event.get("reasoning_tokens")
        reasoning_tokens = (
            0 if raw_reasoning is None
            else _cost_int(raw_reasoning, "reasoning_tokens")
        )
        total_tokens = _cost_int(event.get("total_tokens"), "total_tokens")
        if (
            sdk_attempts != application_attempts
            or application_attempts > 5
            or total_tokens != prompt_tokens + completion_tokens
            or (status == "succeeded" and raw_reasoning is None)
        ):
            raise ValueError(f"B4 sampling checkpoint provider usage differs: {path}")
        provider_costs.append({
            "call_id": call_id,
            "status": status,
            "application_attempts": application_attempts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        })
    if len(provider_run_ids) > 1 or len(provider_run_seeds) > 1:
        raise ValueError(f"B4 sampling checkpoint provider run identity differs: {path}")
    if not failure_kind and any(
        event.get("status") != "succeeded" for event in provider_calls
    ):
        raise ValueError(f"B4 successful sampling checkpoint contains a failed call: {path}")

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_digest": sha256_json(row),
        "job_index": job_index,
        "manifest_index": manifest_index,
        "sample_idx": sample_idx,
        "sample_seed": sample_seed,
        "task_id": task_id,
        "failure_kind": failure_kind,
        "environment_actions": len(actions),
        "wall_time_ms": _cost_int(row.get("wall_time_ms"), "wall_time_ms"),
        "provider_calls": provider_costs,
    }


def _summarize_checkpoint_cost_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    events = [
        dict(event)
        for record in records
        for event in list(record.get("provider_calls") or [])
    ]
    successful = [event for event in events if event.get("status") == "succeeded"]
    failed = [event for event in events if event.get("status") == "failed"]
    application_attempts = sum(int(event["application_attempts"]) for event in events)
    return {
        "charged_sampling_jobs": len(records),
        "failed_sampling_jobs": sum(bool(record.get("failure_kind")) for record in records),
        "successful_uncommitted_sampling_jobs": sum(
            not bool(record.get("failure_kind")) for record in records
        ),
        "environment_actions": sum(int(record["environment_actions"]) for record in records),
        "logical_calls": len(events),
        "successful_logical_calls": len(successful),
        "failed_logical_calls": len(failed),
        "application_attempts": application_attempts,
        "provider_retries": application_attempts - len(events),
        "prompt_tokens": sum(int(event["prompt_tokens"]) for event in events),
        "completion_tokens": sum(int(event["completion_tokens"]) for event in events),
        "reasoning_tokens": sum(int(event["reasoning_tokens"]) for event in events),
        "wall_time_ms_episode_sum": sum(int(record["wall_time_ms"]) for record in records),
    }


def _failed_checkpoint_overhead(root: Path) -> dict[str, Any]:
    jobs_dir = root / "train" / "sampling" / "jobs"
    records = (
        [_checkpoint_cost_record(path) for path in sorted(jobs_dir.glob("*.json"))]
        if jobs_dir.is_dir() else []
    )
    if len({int(record["job_index"]) for record in records}) != len(records):
        raise ValueError("B4 failed attempt has duplicate sampling job identities")
    call_ids = [
        str(event["call_id"])
        for record in records
        for event in list(record["provider_calls"])
    ]
    if len(set(call_ids)) != len(call_ids):
        raise ValueError("B4 failed attempt has duplicate provider call identities")
    discarded = [record for record in records if record.get("failure_kind")]
    totals = _summarize_checkpoint_cost_records(discarded)
    state_path = root / "run_state.json"
    state = _read_json(state_path, "B4 failed Train state") if state_path.is_file() else {}
    complete = state.get("state") in {"failed", "completed"}
    return {
        "schema_version": 2,
        **totals,
        "api_cost": None if totals["logical_calls"] else 0.0,
        "api_cost_unpriced": bool(totals["logical_calls"]),
        "measurement_complete": complete,
        "measurement_semantics": "exact" if complete else "observed_lower_bound",
        "checkpoint_cost_rows": records,
        "evidence": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "row_digest": record["row_digest"],
                "job_index": record["job_index"],
                "failure_kind": record["failure_kind"],
            }
            for record in records
        ],
    }


_FAILED_COST_FIELDS = (
    "charged_sampling_jobs",
    "failed_sampling_jobs",
    "successful_uncommitted_sampling_jobs",
    "environment_actions",
    "logical_calls",
    "successful_logical_calls",
    "failed_logical_calls",
    "application_attempts",
    "provider_retries",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "wall_time_ms_episode_sum",
)


def _aggregate_failed_attempt_paid_cost(
    attempts: Sequence[Mapping[str, Any]],
    *,
    include_successful_checkpoints: bool = False,
) -> dict[str, Any]:
    failed_attempts = [attempt for attempt in attempts if attempt.get("status") == "failed"]
    cost_attempts = list(attempts) if include_successful_checkpoints else failed_attempts
    fallback_totals = {field: 0 for field in _FAILED_COST_FIELDS}
    receipts: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    observed_row_digests: set[str] = set()
    observed_call_ids: set[str] = set()
    measurement_complete = True
    for attempt in cost_attempts:
        raw_cost = attempt.get("failed_job_overhead")
        cost = dict(raw_cost or {})
        records = cost.get("checkpoint_cost_rows")
        if isinstance(records, list):
            for raw_record in records:
                if not isinstance(raw_record, dict):
                    raise ValueError("B4 failed attempt checkpoint cost row is invalid")
                record = dict(raw_record)
                if not include_successful_checkpoints and not record.get("failure_kind"):
                    continue
                row_digest = str(record.get("row_digest", ""))
                if len(row_digest) != 64:
                    raise ValueError("B4 failed attempt checkpoint digest is invalid")
                if row_digest in observed_row_digests:
                    continue
                for event in list(record.get("provider_calls") or []):
                    call_id = str(dict(event).get("call_id", ""))
                    if not call_id or call_id in observed_call_ids:
                        raise ValueError("B4 cross-attempt provider cost is ambiguous")
                    observed_call_ids.add(call_id)
                observed_row_digests.add(row_digest)
                selected_records.append(record)
        else:
            # Test doubles and pre-v2 in-memory rows have no per-checkpoint
            # disposition.  They remain observable but cannot be deduplicated.
            for field in _FAILED_COST_FIELDS:
                fallback_totals[field] += int(cost.get(field, 0))
        complete = isinstance(raw_cost, Mapping) and cost.get(
            "measurement_complete", True
        ) is True
        measurement_complete = measurement_complete and complete
        receipts.append({
            "attempt": int(attempt.get("attempt", -1)),
            "output_dir": str(attempt.get("output_dir", "")),
            "measurement_complete": complete,
            "evidence": list(cost.get("evidence") or []),
        })
    totals = _summarize_checkpoint_cost_records(selected_records)
    for field in _FAILED_COST_FIELDS:
        totals[field] = int(totals.get(field, 0)) + fallback_totals[field]
    return {
        "schema_version": 1,
        "attempt_count": len(cost_attempts),
        **totals,
        "api_cost": None if totals["logical_calls"] else 0.0,
        "api_cost_unpriced": bool(totals["logical_calls"]),
        "measurement_complete": measurement_complete,
        "measurement_semantics": (
            "exact" if measurement_complete else "observed_lower_bound"
        ),
        "attempts": receipts,
        "semantics": (
            "all observed paid sampling checkpoints because no committed Train corpus exists"
            if include_successful_checkpoints
            else "paid failed sampling-job work excluded from the committed Train corpus"
        ),
    }


def _sum_failed_paid_costs(costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        field: sum(int(cost.get(field, 0)) for cost in costs)
        for field in _FAILED_COST_FIELDS
    }
    complete = all(cost.get("measurement_complete", False) is True for cost in costs)
    return {
        "schema_version": 1,
        "attempt_count": sum(int(cost.get("attempt_count", 0)) for cost in costs),
        **totals,
        "api_cost": None if totals["logical_calls"] else 0.0,
        "api_cost_unpriced": bool(totals["logical_calls"]),
        "measurement_complete": complete,
        "measurement_semantics": "exact" if complete else "observed_lower_bound",
        "lane_receipts": [
            {
                "measurement_complete": cost.get("measurement_complete", False) is True,
                "attempts": list(cost.get("attempts") or []),
            }
            for cost in costs
        ],
    }


def _add_role_usage(
    usage: dict[str, Any],
    *,
    calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
) -> None:
    for bucket_name in ("target",):
        bucket = dict(usage.get(bucket_name) or {})
        bucket["calls"] = int(bucket.get("calls", 0)) + calls
        bucket["prompt_tokens"] = int(bucket.get("prompt_tokens", 0)) + prompt_tokens
        bucket["completion_tokens"] = int(bucket.get("completion_tokens", 0)) + completion_tokens
        bucket["reasoning_tokens"] = int(bucket.get("reasoning_tokens", 0)) + reasoning_tokens
        usage[bucket_name] = bucket
    stages = dict(usage.get("per_stage") or {})
    sampling = dict(stages.get("sampling") or {})
    sampling["calls"] = int(sampling.get("calls", 0)) + calls
    sampling["prompt_tokens"] = int(sampling.get("prompt_tokens", 0)) + prompt_tokens
    sampling["completion_tokens"] = int(sampling.get("completion_tokens", 0)) + completion_tokens
    sampling["reasoning_tokens"] = int(sampling.get("reasoning_tokens", 0)) + reasoning_tokens
    stages["sampling"] = sampling
    usage["per_stage"] = stages


def _authoritative_training_cost(
    committed: Mapping[str, Any],
    failed_paid: Mapping[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(committed)))
    failed = dict(failed_paid)
    successful_calls = int(
        failed.get("successful_logical_calls", failed.get("logical_calls", 0))
    )
    prompt_tokens = int(failed.get("prompt_tokens", 0))
    completion_tokens = int(failed.get("completion_tokens", 0))
    reasoning_tokens = int(failed.get("reasoning_tokens", 0))

    sampling = dict(result.get("sampling_episodes") or {})
    sampling["environment_actions"] = int(sampling.get("environment_actions", 0)) + int(
        failed.get("environment_actions", 0)
    )
    sampling["target_llm_calls"] = int(sampling.get("target_llm_calls", 0)) + successful_calls
    sampling["target_prompt_tokens"] = int(sampling.get("target_prompt_tokens", 0)) + prompt_tokens
    sampling["target_completion_tokens"] = int(
        sampling.get("target_completion_tokens", 0)
    ) + completion_tokens
    sampling["target_reasoning_tokens"] = int(
        sampling.get("target_reasoning_tokens", 0)
    ) + reasoning_tokens
    sampling["wall_time_ms_episode_sum"] = int(
        sampling.get("wall_time_ms_episode_sum", 0)
    ) + int(failed.get("wall_time_ms_episode_sum", 0))
    sampling["failed_sampling_jobs"] = int(failed.get("failed_sampling_jobs", 0))
    sampling["successful_uncommitted_sampling_jobs"] = int(
        failed.get("successful_uncommitted_sampling_jobs", 0)
    )
    sampling["attempted_sampling_jobs"] = int(sampling.get("episodes", 0)) + int(
        failed.get("charged_sampling_jobs", failed.get("failed_sampling_jobs", 0))
    )
    result["sampling_episodes"] = sampling

    usage = dict(result.get("usage") or {})
    _add_role_usage(
        usage,
        calls=successful_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    result["usage"] = usage
    committed_provider = dict(result.get("provider_evidence") or {})
    committed_logical = int(
        committed_provider.get(
            "logical_calls", dict(usage.get("target") or {}).get("calls", 0) - successful_calls,
        )
    )
    committed_applications = int(
        committed_provider.get(
            "application_attempts",
            committed_logical + int(result.get("provider_retries", 0)),
        )
    )
    result["logical_provider_calls"] = committed_logical + int(failed.get("logical_calls", 0))
    result["application_provider_attempts"] = committed_applications + int(
        failed.get("application_attempts", 0)
    )
    result["provider_retries"] = int(result.get("provider_retries", 0)) + int(
        failed.get("provider_retries", 0)
    )
    if int(failed.get("logical_calls", 0)):
        result["api_cost"] = None
        result["api_cost_unpriced"] = True
    result.update({
        "schema_version": 1,
        "measurement_complete": failed.get("measurement_complete", True) is True,
        "includes_failed_attempt_paid_cost": bool(failed.get("attempt_count", 0)),
        "failed_attempt_paid_cost": failed,
        "semantics": (
            "authoritative paid Train total; committed corpus work plus discarded "
            "failed sampling-job work, with algorithm state remaining committed-only"
        ),
    })
    return result


def _lane_training_cost_fields(
    committed: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    committed_copy = json.loads(json.dumps(dict(committed)))
    failed_paid = _aggregate_failed_attempt_paid_cost(
        attempts,
        include_successful_checkpoints=not bool(committed_copy),
    )
    authoritative = _authoritative_training_cost(committed_copy, failed_paid)
    if not committed_copy:
        authoritative["measurement_complete"] = False
        authoritative["semantics"] += "; committed Train state is incomplete"
    return {
        "committed_training_cost": committed_copy,
        "failed_attempt_paid_cost": failed_paid,
        "authoritative_training_cost": authoritative,
        # Downstream consumers historically read training_cost.  It now names
        # the paid total; committed state cost remains explicitly separate.
        "training_cost": authoritative,
        "training_cost_accounting": {
            "schema_version": 1,
            "committed_algorithm_state": committed_copy,
            "failed_attempt_paid_cost": failed_paid,
            "authoritative_paid_training_total": authoritative,
        },
    }


def _sum_training_costs(costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sampling_fields = (
        "episodes", "attempted_sampling_jobs", "failed_sampling_jobs",
        "successful_uncommitted_sampling_jobs",
        "environment_actions", "target_llm_calls", "target_prompt_tokens",
        "target_completion_tokens", "target_reasoning_tokens",
        "wall_time_ms_episode_sum",
    )
    def sampling_value(cost: Mapping[str, Any], field: str) -> int:
        sampling = dict(cost.get("sampling_episodes") or {})
        if field == "attempted_sampling_jobs" and field not in sampling:
            return int(sampling.get("episodes", 0))
        return int(sampling.get(field, 0))

    def logical_calls(cost: Mapping[str, Any]) -> int:
        if "logical_provider_calls" in cost:
            return int(cost["logical_provider_calls"])
        evidence = dict(cost.get("provider_evidence") or {})
        if "logical_calls" in evidence:
            return int(evidence["logical_calls"])
        return int(dict(dict(cost.get("usage") or {}).get("target") or {}).get("calls", 0))

    def application_attempts(cost: Mapping[str, Any]) -> int:
        if "application_provider_attempts" in cost:
            return int(cost["application_provider_attempts"])
        evidence = dict(cost.get("provider_evidence") or {})
        if "application_attempts" in evidence:
            return int(evidence["application_attempts"])
        return logical_calls(cost) + int(cost.get("provider_retries", 0))

    totals: dict[str, Any] = {
        "seed_lanes": len(costs),
        "unique_train_tasks": sum(int(cost.get("unique_train_tasks", 0)) for cost in costs),
        "sampling_episodes": {
            field: sum(
                sampling_value(cost, field) for cost in costs
            )
            for field in sampling_fields
        },
        "logical_provider_calls": sum(logical_calls(cost) for cost in costs),
        "application_provider_attempts": sum(
            application_attempts(cost) for cost in costs
        ),
        "provider_retries": sum(int(cost.get("provider_retries", 0)) for cost in costs),
        "measurement_complete": all(cost.get("measurement_complete", True) is True for cost in costs),
    }
    usage: dict[str, Any] = {"target": {}, "evolution": {}, "per_stage": {}}
    for role in ("target", "evolution"):
        usage[role] = {
            field: sum(
                int(dict(dict(cost.get("usage") or {}).get(role) or {}).get(field, 0))
                for cost in costs
            )
            for field in ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens")
        }
    stages = sorted({
        stage
        for cost in costs
        for stage in dict(dict(cost.get("usage") or {}).get("per_stage") or {})
    })
    usage["per_stage"] = {
        stage: {
            field: sum(
                int(
                    dict(
                        dict(dict(cost.get("usage") or {}).get("per_stage") or {}).get(stage)
                        or {}
                    ).get(field, 0)
                )
                for cost in costs
            )
            for field in ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens")
        }
        for stage in stages
    }
    usage["embedding_calls"] = sum(
        int(dict(cost.get("usage") or {}).get("embedding_calls", 0)) for cost in costs
    )
    usage["wall_time_ms"] = sum(
        int(dict(cost.get("usage") or {}).get("wall_time_ms", 0)) for cost in costs
    )
    totals["usage"] = usage
    priced = [cost.get("api_cost") for cost in costs]
    totals["api_cost"] = (
        sum(float(value) for value in priced)
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in priced)
        else None
    )
    totals["api_cost_unpriced"] = totals["api_cost"] is None
    return totals


def _campaign_training_cost_accounting(
    lanes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    committed = [
        dict(lane.get("committed_training_cost") or {})
        for lane in lanes if lane.get("committed_training_cost")
    ]
    authoritative = [
        dict(lane.get("authoritative_training_cost") or {}) for lane in lanes
    ]
    failed = _sum_failed_paid_costs([
        dict(lane.get("failed_attempt_paid_cost") or {}) for lane in lanes
    ])
    return {
        "schema_version": 1,
        "committed_algorithm_state_totals": _sum_training_costs(committed),
        "failed_attempt_paid_cost_totals": failed,
        "authoritative_paid_training_totals": _sum_training_costs(authoritative),
    }


def _sampling_resume_source(root: Path) -> Path | None:
    """Return a resumable sampling boundary, or the fresh initial boundary."""

    jobs_dir = root / "train" / "sampling" / "jobs"
    return root if jobs_dir.is_dir() and any(jobs_dir.glob("*.json")) else None


def _validate_train(root: Path, seed: int) -> tuple[FrozenArtifact, dict[str, Any]]:
    completion = _read_json(root / "completion.json", "B4 Train completion")
    report = _read_json(root / "report.json", "B4 Train report")
    manifest = _read_json(root / "run_manifest.json", "B4 Train run manifest")
    if (
        completion.get("passed") is not True
        or completion.get("phase") != "train"
        or report.get("passed") is not True
        or report.get("method") != METHOD_ID
        or int(manifest.get("run_seed", -1)) != seed
    ):
        raise RuntimeError(f"B4 seed {seed} Train output is invalid")
    frozen = FrozenArtifact.load(root / "frozen")
    assert_frozen_unchanged(frozen)
    if (
        frozen.method_id != METHOD_ID
        or int(frozen.metadata.get("run_seed", -1)) != seed
        or dict(report.get("frozen") or {}).get("digest") != frozen.digest
        or manifest.get("frozen_artifact_digest") != frozen.digest
        or dict(manifest.get("identity") or {}).get("frozen_artifact_digest")
        != frozen.digest
        or dict(completion.get("identity") or {}).get("frozen_artifact_digest")
        != frozen.digest
    ):
        raise RuntimeError(f"B4 seed {seed} frozen artifact identity is invalid")
    return frozen, report


def _validate_test(
    root: Path,
    *,
    seed: int,
    source_run: Path,
    frozen: FrozenArtifact,
) -> dict[str, Any]:
    completion = _read_json(root / "completion.json", "B4 Test completion")
    report = _read_json(root / "test_report.json", "B4 Test report")
    manifest = _read_json(root / "run_manifest.json", "B4 Test run manifest")
    if (
        completion.get("passed") is not True
        or completion.get("phase") != "test"
        or report.get("passed") is not True
        or int(manifest.get("run_seed", -1)) != seed
        or Path(str(report.get("source_run", ""))).resolve() != source_run.resolve()
        or dict(report.get("frozen") or {}).get("digest") != frozen.digest
        or manifest.get("frozen_artifact_digest") != frozen.digest
        or dict(manifest.get("identity") or {}).get("frozen_artifact_digest")
        != frozen.digest
        or dict(completion.get("identity") or {}).get("frozen_artifact_digest")
        != frozen.digest
    ):
        raise RuntimeError(f"B4 seed {seed} Test output is invalid")
    assert_frozen_unchanged(frozen)
    return report


def _lane_state(path: Path, seed: int, state: str, **evidence: Any) -> None:
    previous = _read_json(path, "B4 lane state") if path.is_file() else {
        "schema_version": 1,
        "seed": seed,
        "transitions": [],
    }
    transitions = list(previous.get("transitions") or [])
    transitions.append({"state": state, "at_unix": time.time(), **evidence})
    previous.update({
        "state": state,
        "updated_at_unix": time.time(),
        "transitions": transitions,
        **evidence,
    })
    _write_json(path, previous, overwrite=path.exists())


def _assert_lock(path: Path, digest: str) -> None:
    if not path.is_file() or _sha256_file(path) != digest:
        raise RuntimeError("B4 campaign_lock.json changed after campaign start")


def _run_lane(
    spec: CampaignSpec,
    *,
    seed: int,
    lane_root: Path,
    campaign_lock: Path,
    lock_digest: str,
    runner: Callable[..., int],
) -> dict[str, Any]:
    lane_root.mkdir(parents=False, exist_ok=False)
    state_path = lane_root / "lane_state.json"
    _lane_state(state_path, seed, "PREFLIGHT")
    started = time.perf_counter()
    training_started = started
    attempts: list[dict[str, Any]] = []
    resume_source: Path | None = None
    train_root = lane_root / "train"
    committed_training_cost: dict[str, Any] = {}
    try:
        for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
            _assert_lock(campaign_lock, lock_digest)
            train_root = (
                lane_root / "train"
                if attempt == 1 else lane_root / f"train_attempt_{attempt:03d}"
            )
            _lane_state(state_path, seed, "TRAINING", attempt=attempt)
            code = runner(
                _phase_command(
                    spec,
                    phase="train",
                    seed=seed,
                    output_dir=train_root,
                    campaign_lock=campaign_lock,
                    resume_source_run=resume_source,
                ),
                cwd=spec.repo_root,
                log_path=lane_root / f"train_attempt_{attempt:03d}.log",
            )
            row: dict[str, Any] = {
                "attempt": attempt,
                "output_dir": str(train_root),
                "resume_source_run": str(resume_source) if resume_source else None,
                "returncode": code,
            }
            if code == 0:
                row["status"] = "completed"
                attempts.append(row)
                # The corpus is committed only after extraction/freeze validation.
                # Retain exact checkpoint cost evidence so a later terminal failure
                # still charges all sampling work that was paid for.
                row["failed_job_overhead"] = _failed_checkpoint_overhead(train_root)
                break
            failure = _failure(train_root, code)
            row.update({"status": "failed", **failure})
            attempts.append(row)
            row["failed_job_overhead"] = _failed_checkpoint_overhead(train_root)
            if (
                failure["failure_kind"] != "infrastructure_failure"
                or attempt == MAX_TRAIN_ATTEMPTS
            ):
                _lane_state(state_path, seed, "FAILED", failure=failure)
                return {
                    "seed": seed,
                    "passed": False,
                    "phase": "train",
                    "train_attempts": attempts,
                    **_lane_training_cost_fields({}, attempts),
                    **failure,
                }
            resume_source = _sampling_resume_source(train_root)
        frozen, train_report = _validate_train(train_root, seed)
        training_elapsed_ms = max(
            0, int((time.perf_counter() - training_started) * 1000)
        )
        committed_training_cost = dict(train_report.get("training_cost") or {})
        cost_fields = _lane_training_cost_fields(committed_training_cost, attempts)
        _lane_state(state_path, seed, "FROZEN_VERIFIED", frozen_digest=frozen.digest)
        _assert_lock(campaign_lock, lock_digest)
        test_root = lane_root / "test"
        _lane_state(state_path, seed, "TESTING")
        test_started = time.perf_counter()
        code = runner(
            _phase_command(
                spec,
                phase="test",
                seed=seed,
                output_dir=test_root,
                campaign_lock=campaign_lock,
                source_run=train_root,
            ),
            cwd=spec.repo_root,
            log_path=lane_root / "test.log",
        )
        if code:
            failure = _failure(test_root, code)
            _lane_state(state_path, seed, "FAILED", failure=failure)
            return {
                "seed": seed,
                "passed": False,
                "phase": "test",
                "train_attempts": attempts,
                "frozen_digest": frozen.digest,
                **cost_fields,
                **failure,
            }
        test_report = _validate_test(
            test_root,
            seed=seed,
            source_run=train_root,
            frozen=frozen,
        )
        test_elapsed_ms = max(0, int((time.perf_counter() - test_started) * 1000))
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        _lane_state(state_path, seed, "COMPLETED", elapsed_ms=elapsed_ms)
        return {
            "seed": seed,
            "passed": True,
            "phase": "completed",
            "train_root": str(train_root),
            "test_root": str(test_root),
            "train_attempts": attempts,
            "frozen_digest": frozen.digest,
            **cost_fields,
            "resume_replay_overhead": {
                "attempts": [
                    dict(attempt.get("failed_job_overhead") or {})
                    for attempt in attempts if attempt.get("status") == "failed"
                ],
            },
            "effectiveness": dict(test_report.get("effectiveness") or {}),
            "test_cost": dict(test_report.get("test_cost") or {}),
            "method_specific_metrics": _skillgen_method_metrics(
                committed_training_cost,
                dict(test_report.get("test_cost") or {}),
            ),
            "training_wall_time_ms": training_elapsed_ms,
            "test_wall_time_ms": test_elapsed_ms,
            "elapsed_ms": elapsed_ms,
        }
    except BaseException as exc:
        failure = {
            "failure_kind": "protocol_failure",
            "error_type": type(exc).__name__,
            "error": sanitize_error_text(exc),
        }
        _lane_state(state_path, seed, "FAILED", failure=failure)
        return {
            "seed": seed,
            "passed": False,
            "phase": "controller",
            "train_attempts": attempts,
            **_lane_training_cost_fields(committed_training_cost, attempts),
            **failure,
        }


def _seed_statistic(
    lanes: Sequence[Mapping[str, Any]],
    values: Sequence[int | float],
) -> dict[str, Any]:
    if len(lanes) != len(values) or not values:
        raise ValueError("B4 seed statistic has no aligned observations")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("B4 seed statistic contains a non-finite value")
    return {
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "std_ddof": 1 if len(numeric) > 1 else 0,
        "values_by_seed": {
            str(int(lane["seed"])): value
            for lane, value in zip(lanes, values, strict=True)
        },
    }


def _numeric_intersection(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not rows:
        return ()
    keys = set(rows[0])
    for row in rows[1:]:
        keys.intersection_update(row)
    return tuple(sorted(
        key
        for key in keys
        if all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
            for row in rows
        )
    ))


def _metric_mean_std(
    lanes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = [dict(lane.get("effectiveness") or {}) for lane in lanes]
    return {
        field: _seed_statistic(lanes, [row[field] for row in rows])
        for field in _numeric_intersection(rows)
    }


def _per_family_mean_std(
    lanes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    expected = tuple(ALFWORLD_FORMAL_TASK_TYPES)
    family_rows: list[dict[str, Any]] = []
    for lane in lanes:
        family = dict(dict(lane.get("effectiveness") or {}).get("family") or {})
        if set(family) != set(expected):
            raise ValueError("B4 Test effectiveness does not cover exactly six families")
        family_rows.append(family)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family in expected:
        rows = [dict(row[family]) for row in family_rows]
        if "official_rate" not in _numeric_intersection(rows):
            raise ValueError(f"B4 Test family {family} has no official success rate")
        result[family] = {
            field: _seed_statistic(lanes, [row[field] for row in rows])
            for field in _numeric_intersection(rows)
        }
    return result


_SKILLGEN_METRIC_FIELDS = (
    "sampling_episode_count",
    "valid_trajectory_count",
    "graph_nodes",
    "graph_edges",
    "skill_count",
    "golden_segment_count",
    "retrieval_count",
    "retrieved_skill_count",
)


def _non_negative_int(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or not float(value).is_integer()
    ):
        raise ValueError(f"B4 {label} must be a non-negative integer")
    return int(value)


def _skillgen_method_metrics(
    training_cost: Mapping[str, Any],
    test_cost: Mapping[str, Any],
) -> dict[str, int]:
    train = dict(training_cost.get("method_metrics") or {})
    combined = {
        **{field: train.get(field) for field in _SKILLGEN_METRIC_FIELDS[:6]},
        "retrieval_count": test_cost.get("retrieval_count"),
        "retrieved_skill_count": test_cost.get("retrieved_skill_count"),
    }
    return {
        field: _non_negative_int(combined.get(field), field)
        for field in _SKILLGEN_METRIC_FIELDS
    }


def _role_usage(usage: Mapping[str, Any], role: str) -> dict[str, int]:
    raw = dict(usage.get(role) or {})
    return {
        field: _non_negative_int(raw.get(field, 0), f"{role}.{field}")
        for field in (
            "calls", "prompt_tokens", "completion_tokens", "reasoning_tokens",
        )
    }


def _training_cost_row(lane: Mapping[str, Any]) -> dict[str, Any]:
    cost = dict(lane.get("authoritative_training_cost") or {})
    if not cost or cost.get("measurement_complete") is not True:
        raise ValueError("B4 authoritative paid Train cost is incomplete")
    sampling = dict(cost.get("sampling_episodes") or {})
    usage = dict(cost.get("usage") or {})
    target = _role_usage(usage, "target")
    evolution = _role_usage(usage, "evolution")
    committed_episodes = _non_negative_int(
        sampling.get("episodes"), "committed Train episode count"
    )
    train_episodes = _non_negative_int(
        sampling.get("attempted_sampling_jobs", committed_episodes),
        "paid Train episode count",
    )
    api_cost = cost.get("api_cost")
    if api_cost is not None and (
        isinstance(api_cost, bool)
        or not isinstance(api_cost, (int, float))
        or not math.isfinite(float(api_cost))
        or float(api_cost) < 0
    ):
        raise ValueError("B4 Train API cost is invalid")
    return {
        "unique_train_tasks": _non_negative_int(
            cost.get("unique_train_tasks"), "unique Train task count"
        ),
        "train_episode_count": train_episodes,
        "committed_train_episode_count": committed_episodes,
        "validation_episode_count": 0,
        "environment_actions_train": _non_negative_int(
            sampling.get("environment_actions", 0), "Train environment actions"
        ),
        "environment_actions_validation": 0,
        "target_llm_calls": target["calls"],
        "target_prompt_tokens": target["prompt_tokens"],
        "target_completion_tokens": target["completion_tokens"],
        "target_reasoning_tokens": target["reasoning_tokens"],
        "target_tokens": target["prompt_tokens"] + target["completion_tokens"],
        "evolution_llm_calls": evolution["calls"],
        "evolution_prompt_tokens": evolution["prompt_tokens"],
        "evolution_completion_tokens": evolution["completion_tokens"],
        "evolution_reasoning_tokens": evolution["reasoning_tokens"],
        "evolution_tokens": (
            evolution["prompt_tokens"] + evolution["completion_tokens"]
        ),
        "embedding_calls": _non_negative_int(
            usage.get("embedding_calls", 0), "Train embedding calls"
        ),
        "provider_retries": _non_negative_int(
            cost.get("provider_retries", 0), "Train provider retries"
        ),
        "logical_provider_calls": _non_negative_int(
            cost.get("logical_provider_calls", target["calls"]),
            "Train logical provider calls",
        ),
        "application_provider_attempts": _non_negative_int(
            cost.get("application_provider_attempts", target["calls"]),
            "Train provider attempts",
        ),
        "wall_time_ms": _non_negative_int(
            lane.get("training_wall_time_ms"), "Train wall time"
        ),
        "episode_wall_time_ms_sum": _non_negative_int(
            sampling.get("wall_time_ms_episode_sum", 0),
            "Train episode wall-time sum",
        ),
        "api_cost": float(api_cost) if api_cost is not None else None,
        "api_cost_unpriced": bool(cost.get("api_cost_unpriced", api_cost is None)),
        "measurement_complete": True,
    }


def _effect_number(
    effectiveness: Mapping[str, Any],
    *names: str,
    required: bool = True,
) -> int | float | None:
    for name in names:
        value = effectiveness.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return value
    if required:
        raise ValueError("B4 Test effectiveness is missing " + "/".join(names))
    return None


def _test_cost_row(lane: Mapping[str, Any]) -> dict[str, Any]:
    effectiveness = dict(lane.get("effectiveness") or {})
    cost = dict(lane.get("test_cost") or {})
    episodes = dict(cost.get("episodes") or {})
    usage = dict(cost.get("usage") or {})
    target = _role_usage(usage, "target")
    evolution = _role_usage(usage, "evolution")
    tasks = _non_negative_int(
        _effect_number(effectiveness, "tasks"), "Test task count"
    )
    successes = _non_negative_int(
        _effect_number(effectiveness, "official_success"),
        "Test official success count",
    )
    actions = _non_negative_int(
        episodes.get("environment_actions", effectiveness.get("environment_actions")),
        "Test environment actions",
    )
    api_cost = usage.get("api_cost", cost.get("api_cost"))
    if api_cost is not None and (
        isinstance(api_cost, bool)
        or not isinstance(api_cost, (int, float))
        or not math.isfinite(float(api_cost))
        or float(api_cost) < 0
    ):
        raise ValueError("B4 Test API cost is invalid")
    target_tokens = target["prompt_tokens"] + target["completion_tokens"]
    evolution_tokens = evolution["prompt_tokens"] + evolution["completion_tokens"]
    llm_calls = target["calls"] + evolution["calls"]
    llm_tokens = target_tokens + evolution_tokens
    return {
        "test_episode_count": tasks,
        "official_success": successes,
        "official_success_rate": _effect_number(
            effectiveness, "official_success_rate", "official_rate"
        ),
        "six_family_macro_success_rate": _effect_number(
            effectiveness,
            "six_family_macro_success_rate",
            "macro_family_official_rate",
        ),
        "contract_consistent_success_rate": _effect_number(
            effectiveness,
            "contract_consistent_success_rate",
            "task_contract_rate",
        ),
        "common_strict_success_rate": _effect_number(
            effectiveness, "common_strict_success_rate", "strict_rate"
        ),
        "environment_actions_test": actions,
        "actions_per_task": actions / tasks if tasks else None,
        "actions_per_solved": actions / successes if successes else None,
        "p50_actions": _effect_number(effectiveness, "p50_actions"),
        "p90_actions": _effect_number(effectiveness, "p90_actions"),
        "target_llm_calls": target["calls"],
        "evolution_llm_calls": evolution["calls"],
        "llm_calls": llm_calls,
        "calls_per_task": llm_calls / tasks if tasks else None,
        "p50_calls": _effect_number(effectiveness, "p50_calls"),
        "p90_calls": _effect_number(effectiveness, "p90_calls"),
        "target_prompt_tokens": target["prompt_tokens"],
        "target_completion_tokens": target["completion_tokens"],
        "target_reasoning_tokens": target["reasoning_tokens"],
        "target_tokens": target_tokens,
        "evolution_prompt_tokens": evolution["prompt_tokens"],
        "evolution_completion_tokens": evolution["completion_tokens"],
        "evolution_reasoning_tokens": evolution["reasoning_tokens"],
        "evolution_tokens": evolution_tokens,
        "llm_tokens": llm_tokens,
        "tokens_per_task": llm_tokens / tasks if tasks else None,
        "tokens_per_solved": llm_tokens / successes if successes else None,
        "p50_tokens": _effect_number(effectiveness, "p50_tokens"),
        "p90_tokens": _effect_number(effectiveness, "p90_tokens"),
        "embedding_calls": _non_negative_int(
            usage.get("embedding_calls", 0), "Test embedding calls"
        ),
        "latency_per_task_ms": _effect_number(
            effectiveness, "latency_per_task_ms"
        ),
        "p50_latency_ms": _effect_number(effectiveness, "p50_latency_ms"),
        "p90_latency_ms": _effect_number(effectiveness, "p90_latency_ms"),
        "wall_time_ms": _non_negative_int(
            lane.get("test_wall_time_ms"), "Test wall time"
        ),
        "episode_wall_time_ms_sum": _non_negative_int(
            episodes.get("wall_time_ms_episode_sum", 0),
            "Test episode wall-time sum",
        ),
        "provider_retries": _non_negative_int(
            cost.get("provider_retries", 0), "Test provider retries"
        ),
        "api_cost": float(api_cost) if api_cost is not None else None,
        "api_cost_unpriced": bool(
            usage.get("api_cost_unpriced", cost.get("api_cost_unpriced", api_cost is None))
        ),
        "cost_per_task": (
            float(api_cost) / tasks if api_cost is not None and tasks else None
        ),
        "cost_per_solved": (
            float(api_cost) / successes
            if api_cost is not None and successes else None
        ),
        "measurement_complete": True,
    }


def _aggregate_rows(
    lanes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    additive_fields: Sequence[str],
) -> dict[str, Any]:
    if len(lanes) != len(rows) or not rows:
        raise ValueError("B4 campaign aggregate has no aligned seed rows")
    common_fields = set(rows[0])
    for row in rows[1:]:
        common_fields.intersection_update(row)
    mean_std: dict[str, Any] = {}
    for field in sorted(common_fields):
        values = [row[field] for row in rows]
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            mean_std[field] = _seed_statistic(lanes, values)
        elif all(value is None for value in values):
            mean_std[field] = {
                "mean": None,
                "std": None,
                "std_ddof": 1,
                "values_by_seed": {
                    str(int(lane["seed"])): None for lane in lanes
                },
                "status": "unavailable_for_all_seeds",
            }
        elif all(
            value is None
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
            for value in values
        ):
            mean_std[field] = {
                "mean": None,
                "std": None,
                "std_ddof": 1,
                "values_by_seed": {
                    str(int(lane["seed"])): value
                    for lane, value in zip(lanes, values, strict=True)
                },
                "status": "undefined_for_one_or_more_seeds",
            }
    totals: dict[str, Any] = {}
    for field in additive_fields:
        values = [row.get(field) for row in rows]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            total = sum(values)
            totals[field] = (
                int(total) if all(isinstance(value, int) for value in values) else total
            )
        else:
            totals[field] = None
    return {
        "schema_version": 1,
        "measurement_complete": all(
            row.get("measurement_complete", True) is True for row in rows
        ),
        "api_cost_unpriced": any(
            row.get("api_cost_unpriced", False) is True for row in rows
        ),
        "totals": totals,
        "mean_std": mean_std,
        "by_seed": {
            str(int(lane["seed"])): dict(row)
            for lane, row in zip(lanes, rows, strict=True)
        },
    }


def _ordered_complete_lanes(
    lanes: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_seed: dict[int, Mapping[str, Any]] = {}
    for lane in lanes:
        seed = int(lane.get("seed", -1))
        if seed in by_seed or lane.get("passed") is not True:
            raise ValueError("B4 paper summary requires three distinct completed seed lanes")
        by_seed[seed] = lane
    if set(by_seed) != set(FORMAL_SEEDS):
        raise ValueError("B4 paper summary requires exactly seeds 42, 43, and 44")
    return [by_seed[seed] for seed in FORMAL_SEEDS]


def _campaign_summary(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = _ordered_complete_lanes(lanes)
    train_rows = [_training_cost_row(lane) for lane in ordered]
    test_rows = [_test_cost_row(lane) for lane in ordered]
    method_rows = [dict(lane["method_specific_metrics"]) for lane in ordered]
    method_aggregate = _aggregate_rows(
        ordered,
        method_rows,
        additive_fields=_SKILLGEN_METRIC_FIELDS,
    )
    effectiveness = {
        "mean_std": _metric_mean_std(ordered),
        "per_family_mean_std": _per_family_mean_std(ordered),
        "by_seed": {
            str(int(lane["seed"])): dict(lane["effectiveness"])
            for lane in ordered
        },
        "paired_transfer_status": "unavailable_without_matching_b0_result",
        "positive_transfer": None,
        "negative_transfer": None,
    }
    training = _aggregate_rows(
        ordered,
        train_rows,
        additive_fields=(
            "unique_train_tasks", "train_episode_count",
            "committed_train_episode_count", "validation_episode_count",
            "environment_actions_train", "environment_actions_validation",
            "target_llm_calls", "target_prompt_tokens",
            "target_completion_tokens", "target_reasoning_tokens",
            "target_tokens", "evolution_llm_calls", "evolution_prompt_tokens",
            "evolution_completion_tokens", "evolution_reasoning_tokens",
            "evolution_tokens", "embedding_calls", "provider_retries",
            "logical_provider_calls", "application_provider_attempts",
            "wall_time_ms", "episode_wall_time_ms_sum", "api_cost",
        ),
    )
    testing = _aggregate_rows(
        ordered,
        test_rows,
        additive_fields=(
            "test_episode_count", "official_success", "environment_actions_test",
            "target_llm_calls", "evolution_llm_calls", "llm_calls",
            "target_prompt_tokens", "target_completion_tokens",
            "target_reasoning_tokens", "target_tokens",
            "evolution_prompt_tokens", "evolution_completion_tokens",
            "evolution_reasoning_tokens", "evolution_tokens", "llm_tokens",
            "embedding_calls", "provider_retries", "wall_time_ms",
            "episode_wall_time_ms_sum", "api_cost",
        ),
    )
    return {
        "schema_version": 1,
        "passed": True,
        "paper_eligible": True,
        "method": METHOD_ID,
        "seeds": list(FORMAL_SEEDS),
        "statistical_definitions": {
            "seed_unit": "one complete method x seed run",
            "seed_aggregation": "arithmetic mean over seeds 42, 43, and 44",
            "standard_deviation": "sample standard deviation across seeds",
            "std_ddof": 1,
            "micro_success_rate": (
                "official successes divided by all exact Test-manifest tasks within each seed"
            ),
            "six_family_macro_success_rate": (
                "unweighted arithmetic mean of the six per-family official success rates"
            ),
            "per_family_success_rate": (
                "official successes divided by exact Test tasks in that family"
            ),
            "infrastructure_failure": (
                "invalidates the seed lane and is never entered as a task failure"
            ),
            "quantiles": (
                "nearest-rank P50/P90 within each seed, then mean and sample std across seeds"
            ),
            "tokens": (
                "prompt plus completion tokens; reasoning tokens are a reported subset "
                "of completion tokens and are not added twice"
            ),
            "training_cost": (
                "actual paid Train work, including discarded failed-attempt work exactly once"
            ),
            "test_cost": "one immutable frozen Test-134 evaluation per seed",
            "api_cost": "null when no frozen pricing authority is configured",
        },
        "effectiveness": effectiveness,
        "training_cost": training,
        "test_cost": testing,
        "method_specific_metrics": method_aggregate,
        "source_runs": {
            str(int(lane["seed"])): {
                "train_root": str(lane["train_root"]),
                "test_root": str(lane["test_root"]),
                "frozen_digest": str(lane["frozen_digest"]),
            }
            for lane in ordered
        },
    }


def _run_seed_lanes(
    spec: CampaignSpec,
    *,
    campaign_lock: Path,
    lock_digest: str,
    runner: CommandRunner,
) -> list[dict[str, Any]]:
    """Run one complete method x seed lane at a time.

    Episode workers within a lane retain their configured parallelism.  The
    outer campaign remains serialized so optimizer state and provider load
    from different seeds cannot overlap.
    """

    return [
        _run_lane(
            spec,
            seed=seed,
            lane_root=spec.output_dir / f"seed_{seed}",
            campaign_lock=campaign_lock,
            lock_digest=lock_digest,
            runner=runner,
        )
        for seed in spec.seeds
    ]


def run_campaign(
    spec: CampaignSpec,
    *,
    runner: Callable[..., int] | None = None,
    source_inspector: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runner = runner or _default_runner
    source_inspector = source_inspector or inspect_clean_source
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    campaign_run_id = f"b4_campaign_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    with _campaign_lease(spec.repo_root, owner=campaign_run_id):
        if spec.output_dir.exists():
            raise FileExistsError(spec.output_dir)
        source_before = source_inspector(spec.repo_root)
        lock_payload = build_campaign_lock(
            spec,
            campaign_run_id=campaign_run_id,
            source_state=source_before,
        )
        spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
        spec.output_dir.mkdir(exist_ok=False)
        runtime_spec, lock_payload, probe_report = _run_probe_with_fallback(
            spec, lock_payload, runner=runner
        )
        if source_inspector(spec.repo_root) != source_before:
            raise RuntimeError("B4 controller source changed during provider preflight")
        lock_path = spec.output_dir / "campaign_lock.json"
        _write_json(lock_path, lock_payload)
        lock_digest = _sha256_file(lock_path)
        started_at = time.time()
        started = time.perf_counter()
        lanes = _run_seed_lanes(
            runtime_spec,
            campaign_lock=lock_path,
            lock_digest=lock_digest,
            runner=runner,
        )
        _assert_lock(lock_path, lock_digest)
        if source_inspector(spec.repo_root) != source_before:
            raise RuntimeError("B4 controller source changed during the formal campaign")
        lanes_passed = len(lanes) == 3 and all(
            lane.get("passed") is True for lane in lanes
        )
        complete = [lane for lane in lanes if lane.get("passed") is True]
        reporting_error: dict[str, str] | None = None
        if lanes_passed:
            try:
                summary = _campaign_summary(complete)
            except BaseException as exc:
                reporting_error = {
                    "error_type": type(exc).__name__,
                    "error": sanitize_error_text(exc),
                }
                summary = {
                    "schema_version": 1,
                    "passed": False,
                    "paper_eligible": False,
                    "method": METHOD_ID,
                    "seeds": list(spec.seeds),
                    "failure_kind": "protocol_failure",
                    "reporting_error": reporting_error,
                }
        else:
            summary = {
                "schema_version": 1,
                "passed": False,
                "paper_eligible": False,
                "method": METHOD_ID,
                "seeds": list(spec.seeds),
                "failure_kind": "incomplete_seed_lanes",
            }
        passed = lanes_passed and reporting_error is None
        _write_json(spec.output_dir / "campaign_summary.json", summary)
        report = {
            "schema_version": 1,
            "passed": passed,
            "method": METHOD_ID,
            "campaign_id": lock_payload["campaign_id"],
            "campaign_run_id": campaign_run_id,
            "campaign_root": str(spec.output_dir),
            "campaign_lock": str(lock_path),
            "campaign_lock_digest": lock_digest,
            "seeds": list(spec.seeds),
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
            "campaign_makespan_ms": max(
                0, int((time.perf_counter() - started) * 1000)
            ),
            "provider_probe": probe_report,
            "lanes": lanes,
            "completed_seeds": [int(lane["seed"]) for lane in complete],
            "failed_seeds": [
                int(lane["seed"]) for lane in lanes if lane.get("passed") is not True
            ],
            "paper_eligible": passed,
            "campaign_summary": "campaign_summary.json",
            "statistical_definitions": dict(
                summary.get("statistical_definitions") or {}
            ),
            "test_metrics_mean_std": dict(
                dict(summary.get("effectiveness") or {}).get("mean_std") or {}
            ),
            "test_per_family_mean_std": dict(
                dict(summary.get("effectiveness") or {}).get(
                    "per_family_mean_std"
                ) or {}
            ),
            "training_cost_summary": dict(summary.get("training_cost") or {}),
            "test_cost_summary": dict(summary.get("test_cost") or {}),
            "method_specific_metrics": dict(
                summary.get("method_specific_metrics") or {}
            ),
            "training_cost_accounting": _campaign_training_cost_accounting(lanes),
            "all_frozen_digests_verified": lanes_passed,
            "reporting_error": reporting_error,
        }
        _write_json(spec.output_dir / "campaign_report.json", report)
        _write_json(
            spec.output_dir / ("completion.json" if passed else "campaign_failure.json"),
            {
                "schema_version": 1,
                "passed": passed,
                "campaign_run_id": campaign_run_id,
                "campaign_lock_digest": lock_digest,
                "campaign_report": "campaign_report.json",
                "campaign_summary": "campaign_summary.json",
                "completed_at_unix": time.time(),
            },
        )
        return report


def _default_output(repo_root: Path, campaign_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo_root
        / "runs"
        / "baselines"
        / campaign_id
        / METHOD_ID
        / f"formal_3seed_{stamp}_{os.getpid()}"
    )


def _safe_error(exc: BaseException) -> str:
    text = sanitize_error_text(exc)
    secret = os.environ.get("MODEL_API_KEY", "").strip()
    return text.replace(secret, "[REDACTED]") if secret else text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--config", default="configs/baselines/b4_skillgen_s.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default=None)
    args = parser.parse_args(argv)
    try:
        repo = REPO_ROOT.resolve()
        config = _resolve(repo, args.config)
        preview = {
            **dict(yaml.safe_load((repo / "configs/baselines/common.yaml").read_text(
                encoding="utf-8"
            )) or {}),
            **dict(yaml.safe_load(config.read_text(encoding="utf-8")) or {}),
        }
        configured_python = resolve_formal_python(
            repo, str(preview.get("worker_python", ""))
        )
        if args.python is not None:
            supplied = resolve_formal_python(repo, args.python)
            if str(supplied) != str(configured_python):
                raise ValueError("--python must exactly match B4 worker_python")
        output = (
            _resolve(repo, args.output_dir)
            if args.output_dir
            else _default_output(repo, str(preview.get("campaign_id", "formal")))
        )
        spec = CampaignSpec(
            train_manifest=_resolve(repo, args.train_manifest),
            test_manifest=_resolve(repo, args.test_manifest),
            supervision=_resolve(repo, args.supervision),
            config=config,
            output_dir=output,
            python=configured_python,
            repo_root=repo,
            controller_python=Path(os.path.abspath(sys.executable)),
        )
        report = run_campaign(spec)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    except BaseException as exc:
        print(json.dumps({
            "passed": False,
            "error_type": type(exc).__name__,
            "error": _safe_error(exc),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
