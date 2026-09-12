"""Run the formal three-seed B5 GEPA campaign.

The three method x seed lanes run serially.  Independent task evaluations
inside each lane retain their configured process parallelism.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from experiments.baselines.bootstrap_external import (
    load_lock,
    source_file_sha256,
    verify_key_files,
    verify_runtime_tree,
)
from experiments.baselines.common.formal_validation import verify_formal_manifest
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.manifest import (
    TaskManifestSet,
    sha256_json,
    verify_disjoint,
)
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.runtime_python import (
    resolve_formal_python,
    verify_runtime_python,
)
from experiments.baselines.common.usage import UsageSnapshot
from experiments.baselines.report_campaign import build_campaign_report
from experiments.baselines.run_seed_campaign import (
    _campaign_cost_summary,
    _cost_accounting,
    _default_command_runner,
    _formal_config_digest,
    _method_campaign_lease,
    _phase_episode_action_summary,
    _phase_provider_sidecar_summaries,
    _resource_summary,
    _resource_usage,
    _run_provider_probe_with_fallback as _shared_run_provider_probe_with_fallback,
    _sha256_file,
    _write_json_atomic,
    inspect_clean_source,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "b5_gepa"
FORMAL_SEEDS = (42, 43, 44)
_INITIAL_SKILL_REL = "skillopt/envs/alfworld/skills/initial.md"


CommandRunner = Callable[..., int]
SourceInspector = Callable[[Path], dict[str, Any]]


@dataclass(frozen=True)
class GEPACampaignSpec:
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
        if self.method != METHOD_ID:
            raise ValueError(f"B5 campaign method must be {METHOD_ID!r}")
        if self.seeds != FORMAL_SEEDS:
            raise ValueError("formal B5 seeds must be exactly 42, 43, 44 in order")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return dict(payload)


def _merged_config(spec: GEPACampaignSpec) -> dict[str, Any]:
    common = yaml.safe_load(
        (spec.repo_root / "configs/baselines/common.yaml").read_text(encoding="utf-8")
    )
    method = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    if not isinstance(common, dict) or not isinstance(method, dict):
        raise ValueError("B5/common config roots must be mappings")
    return {**common, **method}


def _replace_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a generated runtime config without exposing partial YAML."""

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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _provider_cap_mismatches(
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    cap: int,
) -> list[str]:
    parallel = dict(config.get("parallel") or {})
    locked_parallel = dict(payload.get("parallel") or {})
    env = dict(config.get("env") or {})
    config_probe = dict(config.get("provider_probe") or {})
    probe = dict(payload.get("provider_probe") or {})
    checks = {
        "config.parallel.seed_lanes": int(parallel.get("seed_lanes", 0)) == 1,
        "lock.parallel.seed_lanes": int(locked_parallel.get("seed_lanes", 0)) == 1,
        "config.parallel.episode_workers_per_seed": int(
            parallel.get("episode_workers_per_seed", 0)
        ) == cap,
        "config.parallel.test_workers_per_seed": int(
            parallel.get("test_workers_per_seed", 0)
        ) == cap,
        "config.parallel.campaign_provider_max_inflight": int(
            parallel.get("campaign_provider_max_inflight", 0)
        ) == cap,
        "config.parallel.mp_start_method": parallel.get("mp_start_method")
        == "spawn",
        "lock.parallel.episode_workers_per_seed": int(
            locked_parallel.get("episode_workers_per_seed", 0)
        ) == cap,
        "lock.parallel.test_workers_per_seed": int(
            locked_parallel.get("test_workers_per_seed", 0)
        ) == cap,
        "lock.parallel.campaign_provider_max_inflight": int(
            locked_parallel.get("campaign_provider_max_inflight", 0)
        ) == cap,
        "lock.parallel.mp_start_method": locked_parallel.get("mp_start_method")
        == "spawn",
        "lock.campaign_provider_max_inflight": int(
            payload.get("campaign_provider_max_inflight", 0)
        ) == cap,
        "config.env.workers": int(env.get("workers", 0)) == cap,
        "config.env.max_api_workers": int(env.get("max_api_workers", 0)) == cap,
        "config.provider_probe.concurrency": int(
            config_probe.get("concurrency", 0)
        ) == cap,
        "lock.provider_probe.concurrency": int(probe.get("concurrency", 0)) == cap,
    }
    return sorted(name for name, passed in checks.items() if not passed)


def _run_provider_probe_with_fallback(
    spec: GEPACampaignSpec,
    lock_payload: Mapping[str, Any],
    *,
    command_runner: CommandRunner,
) -> tuple[GEPACampaignSpec, dict[str, Any], dict[str, Any]]:
    """Bind B5 worker counts to the provider cap selected by the shared probe."""

    runtime_spec, selected_payload, probe_report = (
        _shared_run_provider_probe_with_fallback(
            spec,
            lock_payload,
            command_runner=command_runner,
        )
    )
    payload = dict(selected_payload)
    cap = int(payload.get("campaign_provider_max_inflight", 0))
    if cap not in (16, 12, 8):
        raise ValueError(f"unsupported B5 provider cap selected: {cap}")

    if cap == 16:
        if runtime_spec.config.resolve() != spec.config.resolve():
            raise ValueError("B5 cap 16 must retain the checked-in formal config")
        config = _merged_config(runtime_spec)
        mismatches = _provider_cap_mismatches(config, payload, cap=cap)
        if payload.get("formal_config_digest") != _formal_config_digest(config):
            mismatches.append("lock.formal_config_digest")
        if mismatches:
            raise ValueError(
                "B5 provider-cap authority mismatch: " + ", ".join(sorted(mismatches))
            )
        return runtime_spec, payload, probe_report

    runtime_config = runtime_spec.config.resolve()
    campaign_root = spec.output_dir.resolve()
    try:
        runtime_config.relative_to(campaign_root)
    except ValueError as exc:
        raise ValueError("B5 generated runtime config is outside campaign root") from exc
    if runtime_config == spec.config.resolve():
        raise ValueError("B5 fallback cannot overwrite the checked-in formal config")
    if not runtime_config.is_file():
        raise FileNotFoundError(f"B5 generated runtime config is missing: {runtime_config}")

    method_config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    if not isinstance(method_config, dict):
        raise ValueError("B5 generated runtime config root must be a mapping")
    parallel = dict(payload.get("parallel") or {})
    if int(parallel.get("seed_lanes", 0)) != 1:
        raise ValueError("B5 provider fallback requires exactly one seed lane")
    parallel.update({
        "episode_workers_per_seed": cap,
        "test_workers_per_seed": cap,
        "campaign_provider_max_inflight": cap,
        "mp_start_method": "spawn",
    })
    env = dict(method_config.get("env") or {})
    env.update({"workers": cap, "max_api_workers": cap})
    provider_probe = dict(payload.get("provider_probe") or {})
    provider_probe["concurrency"] = cap
    method_config.update({
        "parallel": parallel,
        "env": env,
        "provider_probe": {
            **dict(method_config.get("provider_probe") or {}),
            "concurrency": cap,
        },
    })
    _replace_yaml_atomic(runtime_config, method_config)

    payload.update({
        "parallel": parallel,
        "campaign_provider_max_inflight": cap,
        "provider_probe": provider_probe,
        "config_path": str(runtime_config),
    })
    config = _merged_config(runtime_spec)
    payload["formal_config_digest"] = _formal_config_digest(config)
    mismatches = _provider_cap_mismatches(config, payload, cap=cap)
    if mismatches:
        raise ValueError(
            "B5 provider-cap authority mismatch: " + ", ".join(mismatches)
        )
    return runtime_spec, payload, probe_report


def _expect(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    mismatches = [key for key, value in expected.items() if mapping.get(key) != value]
    if mismatches:
        raise ValueError(f"B5 {label} mismatch: " + ", ".join(mismatches))


def build_campaign_lock(
    spec: GEPACampaignSpec,
    campaign_root: Path,
    campaign_run_id: str,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen B5 protocol and return the pre-probe lock payload."""

    if source_state.get("dirty") is not False:
        raise RuntimeError("formal B5 campaign requires clean controller source")
    config = _merged_config(spec)
    if config.get("method") != METHOD_ID or config.get("protocol_profile") != "formal_v2":
        raise ValueError("B5 formal config method/profile mismatch")
    if config.get("experiment_kind") != "formal":
        raise ValueError("B5 campaign cannot use a smoke config")
    _expect(
        dict(config.get("train") or {}),
        {"train_size": 120, "validation_size": 24, "seed": 42},
        "Train/Validation setting",
    )
    _expect(
        dict(config.get("gepa") or {}),
        {
            "candidate_component": "skill_text",
            "candidate_selection_strategy": "pareto",
            "frontier_type": "instance",
            "skip_perfect_score": True,
            "batch_sampler": "epoch_shuffled",
            "reflection_minibatch_size": 3,
            "perfect_score": 1.0,
            "module_selector": "round_robin",
            "use_merge": False,
            "max_metric_calls": 720,
            "cache_evaluation": False,
            "val_evaluation_policy": "full_eval",
            "acceptance_criterion": "strict_improvement",
            "sampling_strategy": None,
            "selection_strategy": None,
            "reflection_max_completion_tokens": 16384,
        },
        "optimizer setting",
    )
    parallel = dict(config.get("parallel") or {})
    _expect(
        parallel,
        {
            "seed_lanes": 1,
            "episode_workers_per_seed": 16,
            "test_workers_per_seed": 16,
            "campaign_provider_max_inflight": 16,
            "mp_start_method": "spawn",
        },
        "parallel setting",
    )
    env = dict(config.get("env") or {})
    _expect(
        env,
        {
            "name": "alfworld",
            "max_steps": 100,
            "max_completion_tokens": 16384,
            "workers": 16,
            "max_api_workers": 16,
        },
        "environment setting",
    )
    transport = dict(config.get("provider_transport") or {})
    _expect(
        transport,
        {
            "sdk_max_retries": 0,
            "application_retry_limit": 5,
            "retry_delays_seconds": [2, 5, 10, 20],
            "deterministic_jitter_ratio": 0.10,
        },
        "provider transport",
    )
    probe = dict(config.get("provider_probe") or {})
    _expect(
        probe,
        {
            "enabled": True,
            "concurrency": 16,
            "requests": 32,
            "max_completion_tokens": 256,
            "reasoning_effort": "high",
        },
        "provider probe",
    )
    _expect(
        dict(config.get("resume") or {}),
        {"enabled": True, "authority": "gepa_run_dir"},
        "resume setting",
    )

    model = ModelConfig.from_mapping(dict(config.get("model") or {}))
    model.validate_formal_identity()
    model.require_api_key()
    configured_python = resolve_formal_python(
        spec.repo_root, str(config.get("worker_python", ""))
    )
    if str(configured_python) != str(resolve_formal_python(spec.repo_root, spec.python)):
        raise ValueError("B5 campaign Python differs from worker_python")
    source_lock_path = spec.repo_root / "experiments/baselines/baseline_lock.yaml"
    source_lock = load_lock(source_lock_path)
    verify_runtime_python(
        expected_python=configured_python,
        require_venv=True,
        method=METHOD_ID,
        expected_distributions={
            "alfworld": "0.4.2",
            "gepa": str(source_lock["gepa"]["version"]),
        },
        expected_python_major_minor="3.12",
    )

    data_raw = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_raw:
        raise RuntimeError("ALFWORLD_DATA is not set")
    data_root = Path(data_raw).expanduser().resolve(strict=True)
    train = TaskManifestSet.load(spec.train_manifest)
    validation = TaskManifestSet.load(spec.validation_manifest)
    test = TaskManifestSet.load(spec.test_manifest)
    verify_disjoint(train, validation, test)
    manifests: dict[str, Any] = {}
    for role, manifest, path in (
        ("train", train, spec.train_manifest),
        ("validation", validation, spec.validation_manifest),
        ("test", test, spec.test_manifest),
    ):
        manifests[role] = {
            "path": str(path),
            "manifest_id": manifest.manifest_id,
            "digest": manifest.digest,
            "tasks": len(manifest.tasks),
            "preflight": verify_formal_manifest(
                manifest,
                alfworld_data=data_root,
                role=role,
                profile="formal_v2",
            ),
        }

    gepa_root = spec.repo_root / ".external/gepa"
    skillopt_root = spec.repo_root / ".external/skillopt"
    verify_key_files(gepa_root, "gepa", source_lock)
    verify_key_files(skillopt_root, "skillopt", source_lock)
    gepa_tree = verify_runtime_tree(gepa_root, "gepa", source_lock)
    skillopt_tree = verify_runtime_tree(skillopt_root, "skillopt", source_lock)
    gepa_lock = dict(source_lock["gepa"])
    skillopt_lock = dict(source_lock["skillopt"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=gepa_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tags = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"],
        cwd=gepa_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if head != str(gepa_lock["commit"]) or str(gepa_lock["tag"]) not in tags:
        raise RuntimeError("B5 GEPA checkout does not match locked commit/tag")
    initial_digest = str(dict(skillopt_lock["key_files"])[_INITIAL_SKILL_REL])
    if source_file_sha256(
        skillopt_root / _INITIAL_SKILL_REL,
        algorithm=str(dict(skillopt_lock["runtime_tree"])["algorithm"]),
    ) != initial_digest:
        raise RuntimeError("B5 initial skill differs from B1/B3 authority")

    return {
        "schema_version": 1,
        "campaign_id": str(config["campaign_id"]),
        "campaign_run_id": campaign_run_id,
        "campaign_root": str(campaign_root),
        "method": METHOD_ID,
        "seeds": list(spec.seeds),
        "train_manifest_digest": train.digest,
        "validation_manifest_digest": validation.digest,
        "test_manifest_digest": test.digest,
        "manifests": manifests,
        "controller_commit": str(source_state["commit"]),
        "controller_code_digest": str(source_state["code_digest"]),
        "controller_git": dict(source_state),
        "formal_config_digest": _formal_config_digest(config),
        "config_path": str(spec.config),
        "external_lock_digest": sha256_json(source_lock),
        "external_gepa_commit": str(gepa_lock["commit"]),
        "external_runtime_tree_digest": str(gepa_tree["sha256"]),
        "external_skillopt_commit": str(skillopt_lock["commit"]),
        "skillopt_runtime_tree_digest": str(skillopt_tree["sha256"]),
        "initial_skill_sha256": initial_digest,
        "model": model.model,
        "model_identity": model.to_wire(),
        "reasoning_effort": model.reasoning_effort,
        "seed_lanes": 1,
        "campaign_provider_max_inflight": 16,
        "parallel": parallel,
        "retry_policy": {
            "sdk_max_retries": int(transport["sdk_max_retries"]),
            "attempts": int(transport["application_retry_limit"]),
            "delays": list(transport["retry_delays_seconds"]),
            "jitter_ratio": float(transport["deterministic_jitter_ratio"]),
        },
        "provider_probe": probe,
        "resume": dict(config["resume"]),
        "provider_gate_dir": str((campaign_root / "provider_gate/global_16").resolve()),
        "phase_python": str(configured_python),
        "worker_python": str(configured_python),
        "provider_probe_python": str(configured_python),
        "created_at_unix": time.time(),
    }


def phase_command(
    spec: GEPACampaignSpec,
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
        METHOD_ID,
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
        if source_run is not None:
            raise ValueError("B5 Train cannot receive --source-run")
        if resume_source_run is not None:
            command.extend(["--resume-source-run", str(resume_source_run)])
    elif phase == "test":
        if source_run is None or resume_source_run is not None:
            raise ValueError("B5 Test requires only its own frozen Train source")
        command.extend(["--source-run", str(source_run)])
    else:
        raise ValueError(f"unsupported B5 campaign phase: {phase}")
    return command


def _assert_manifest_frozen_binding(
    manifest: Mapping[str, Any],
    frozen: FrozenArtifact,
    *,
    phase: str,
) -> None:
    descriptor = dict(manifest.get("frozen_artifact") or {})
    try:
        descriptor_root = Path(str(descriptor.get("root", ""))).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("B5 run manifest has an invalid frozen artifact root") from exc
    if (
        manifest.get("method") != METHOD_ID
        or manifest.get("phase") != phase
        or manifest.get("frozen_artifact_digest") != frozen.digest
        or descriptor.get("digest") != frozen.digest
        or descriptor_root != frozen.root.resolve()
        or descriptor.get("source_train_manifest_hash")
        != frozen.source_train_manifest_hash
        or descriptor.get("source_validation_manifest_hash")
        != frozen.source_validation_manifest_hash
    ):
        raise RuntimeError(
            f"B5 {phase} run manifest is not bound to its frozen artifact"
        )


def _validate_train(root: Path, seed: int) -> FrozenArtifact:
    completion = _read_json(root / "completion.json", "B5 train completion")
    report = _read_json(root / "report.json", "B5 train report")
    manifest = _read_json(root / "run_manifest.json", "B5 train identity")
    if (
        completion.get("passed") is not True
        or completion.get("phase") != "train"
        or report.get("passed") is not True
        or report.get("method") != METHOD_ID
        or int(manifest.get("run_seed", -1)) != seed
    ):
        raise RuntimeError(f"seed {seed} B5 Train evidence is invalid")
    frozen = FrozenArtifact.load(root / "frozen")
    assert_frozen_unchanged(frozen)
    if (
        int(frozen.metadata.get("run_seed", -1)) != seed
        or dict(report.get("frozen") or {}).get("digest") != frozen.digest
    ):
        raise RuntimeError(f"seed {seed} B5 frozen artifact identity is invalid")
    _assert_manifest_frozen_binding(manifest, frozen, phase="train")
    return frozen


def _validate_test(
    root: Path, *, seed: int, train_root: Path, frozen: FrozenArtifact
) -> dict[str, Any]:
    completion = _read_json(root / "completion.json", "B5 test completion")
    report = _read_json(root / "test_report.json", "B5 test report")
    manifest = _read_json(root / "run_manifest.json", "B5 test identity")
    protocol = dict(report.get("protocol") or {})
    if (
        completion.get("passed") is not True
        or completion.get("phase") != "test"
        or report.get("passed") is not True
        or int(manifest.get("run_seed", -1)) != seed
        or Path(str(protocol.get("source_run", ""))).resolve() != train_root.resolve()
        or dict(report.get("frozen") or {}).get("digest") != frozen.digest
    ):
        raise RuntimeError(f"seed {seed} B5 frozen Test evidence is invalid")
    _assert_manifest_frozen_binding(manifest, frozen, phase="test")
    assert_frozen_unchanged(frozen)
    return report


def _failure_kind(root: Path) -> str:
    path = root / "failure.json"
    if not path.is_file():
        return "protocol_failure"
    return str(_read_json(path, "B5 phase failure").get("failure_kind", ""))


def _actual_usage_receipt(
    usage_paths: Sequence[Path], *, strict: bool
) -> dict[str, Any]:
    """Aggregate actual per-attempt role usage and bind every source file."""

    total = UsageSnapshot()
    evidence: list[dict[str, str]] = []
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    for path in usage_paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            usage = UsageSnapshot.load(path)
        except ValueError as exc:
            if strict:
                raise
            errors.append({"path": str(path), "error": str(exc)})
            continue
        total.add(usage)
        evidence.append({
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        })
    return {
        "schema_version": 1,
        "measurement_complete": not missing and not errors,
        "usage": total.to_dict(),
        "evidence": evidence,
        "missing": missing,
        "errors": errors,
    }


def _attempt_cost_evidence(attempt_root: Path) -> dict[str, Any]:
    phase_dir = attempt_root / "train"
    sidecars = _phase_provider_sidecar_summaries(phase_dir)
    usage = _actual_usage_receipt([phase_dir / "usage.json"], strict=False)
    actions = _phase_episode_action_summary(phase_dir)
    return {
        "usage": usage,
        "environment_actions": actions,
        "provider_sidecars": sidecars,
        "provider_sidecar_count": len(sidecars),
    }


def _campaign_actual_usage(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = UsageSnapshot()
    by_seed: dict[str, Any] = {}
    complete = True
    for lane in lanes:
        receipt = dict(lane.get("actual_attempt_usage") or {})
        usage = UsageSnapshot.from_dict(dict(receipt.get("usage") or {}))
        total.add(usage)
        complete = complete and receipt.get("measurement_complete") is True
        by_seed[str(lane["seed"])] = receipt
    return {
        "schema_version": 1,
        "measurement_complete": complete,
        "usage": total.to_dict(),
        "by_seed": by_seed,
        "semantics": (
            "sum of every persisted Train attempt and frozen Test usage receipt; "
            "includes failed-prefix and resume-replay calls"
        ),
    }


def _run_lane(
    spec: GEPACampaignSpec,
    *,
    seed: int,
    campaign_lock: Path,
    lock_digest: str,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    lane_root = spec.output_dir / f"seed_{seed}"
    lane_root.mkdir(exist_ok=False)
    train_attempts: list[dict[str, Any]] = []
    source_for_resume: Path | None = None
    resume_lineage: list[Path] = []
    failed_attempt_lineage: list[Path] = []
    train_root: Path | None = None
    test_root: Path | None = None
    try:
        for attempt in (1, 2):
            if _sha256_file(campaign_lock) != lock_digest:
                raise RuntimeError("B5 campaign lock changed before Train")
            attempt_root = lane_root / f"train_attempt_{attempt}"
            command = phase_command(
                spec,
                phase="train",
                seed=seed,
                output_dir=attempt_root,
                campaign_lock=campaign_lock,
                resume_source_run=source_for_resume,
            )
            returncode = command_runner(
                command,
                cwd=spec.repo_root,
                log_path=lane_root / f"train_attempt_{attempt}.log",
            )
            row = {
                "attempt": attempt,
                "root": str(attempt_root),
                "returncode": int(returncode),
                "resumed_from": str(source_for_resume) if source_for_resume else None,
                "cost_evidence": _attempt_cost_evidence(attempt_root),
            }
            train_attempts.append(row)
            if returncode == 0:
                row["status"] = "completed"
                train_root = attempt_root
                break
            kind = _failure_kind(attempt_root)
            row.update({"status": "failed", "failure_kind": kind})
            if attempt == 1 and kind == "infrastructure_failure":
                failed_attempt_lineage.append(attempt_root)
                if (attempt_root / "train/gepa_state/gepa_state.bin").is_file():
                    source_for_resume = attempt_root
                    resume_lineage.append(attempt_root)
                    row["retry_mode"] = "resume_checkpoint"
                else:
                    source_for_resume = None
                    row["retry_mode"] = "fresh_initial"
                continue
            raise RuntimeError(
                f"seed {seed} B5 Train attempt {attempt} failed as {kind}"
            )
        if train_root is None:
            raise RuntimeError(f"seed {seed} B5 Train did not produce a source run")
        frozen = _validate_train(train_root, seed)
        if _sha256_file(campaign_lock) != lock_digest:
            raise RuntimeError("B5 campaign lock changed before Test")
        test_root = lane_root / "test"
        test_rc = command_runner(
            phase_command(
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
        if test_rc != 0:
            raise RuntimeError(
                f"seed {seed} B5 Test failed as {_failure_kind(test_root)}"
            )
        test_report = _validate_test(
            test_root, seed=seed, train_root=train_root, frozen=frozen
        )
        train_report = _read_json(train_root / "report.json", "B5 train report")
        resource_usage = _resource_usage(train_report, test_report)
        cost_accounting = _cost_accounting(
            seed=seed,
            resume_sources=resume_lineage,
            failed_attempt_sources=failed_attempt_lineage,
            train_root=train_root,
            test_root=test_root,
            committed=resource_usage,
        )
        actual_attempt_usage = _actual_usage_receipt(
            [
                *(Path(str(row["root"])) / "train" / "usage.json" for row in train_attempts),
                test_root / "test" / "usage.json",
            ],
            strict=True,
        )
        lane = {
            "passed": True,
            "seed": seed,
            "train_root": str(train_root),
            "test_root": str(test_root),
            "frozen_digest": frozen.digest,
            "train_attempts": train_attempts,
            "training_cost": dict(train_report.get("training_cost") or {}),
            "test_cost": dict(test_report.get("test_cost") or {}),
            "effectiveness": dict(test_report.get("effectiveness") or {}),
            "resource_usage": resource_usage,
            "cost_accounting": cost_accounting,
            "actual_attempt_usage": actual_attempt_usage,
        }
    except Exception as exc:
        usage_paths = [
            Path(str(row["root"])) / "train" / "usage.json"
            for row in train_attempts
        ]
        if test_root is not None:
            usage_paths.append(test_root / "test" / "usage.json")
        lane = {
            "passed": False,
            "seed": seed,
            "train_attempts": train_attempts,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_attempt_usage": _actual_usage_receipt(
                usage_paths, strict=False
            ),
        }
    _write_json_atomic(lane_root / "lane_report.json", lane, overwrite=False)
    return lane


def _numeric_mapping_summary(
    rows_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply one sample-statistics definition to flat numeric seed records."""

    ordered_seeds = [str(seed) for seed in FORMAL_SEEDS if str(seed) in rows_by_seed]
    if not ordered_seeds:
        return {}
    rows = [dict(rows_by_seed[seed]) for seed in ordered_seeds]
    keys = set(rows[0])
    for row in rows[1:]:
        keys.intersection_update(row)
    numeric_keys = sorted(
        key
        for key in keys
        if all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
            for row in rows
        )
    )
    return {
        key: {
            "mean": statistics.fmean(float(row[key]) for row in rows),
            "std": (
                statistics.stdev(float(row[key]) for row in rows)
                if len(rows) > 1 else 0.0
            ),
            "std_ddof": 1 if len(rows) > 1 else 0,
            "values_by_seed": {
                seed: row[key]
                for seed, row in zip(ordered_seeds, rows, strict=True)
            },
        }
        for key in numeric_keys
    }


def _usage_cost_fields(usage_payload: Mapping[str, Any]) -> dict[str, int]:
    usage = UsageSnapshot.from_dict(dict(usage_payload))
    target = usage.target
    evolution = usage.evolution
    return {
        "target_llm_calls": target.calls,
        "evolution_llm_calls": evolution.calls,
        "llm_calls": target.calls + evolution.calls,
        "target_prompt_tokens": target.prompt_tokens,
        "target_completion_tokens": target.completion_tokens,
        "target_reasoning_tokens": target.reasoning_tokens,
        "evolution_prompt_tokens": evolution.prompt_tokens,
        "evolution_completion_tokens": evolution.completion_tokens,
        "evolution_reasoning_tokens": evolution.reasoning_tokens,
        # Completion tokens include reasoning tokens in the provider protocol.
        "llm_tokens": (
            target.prompt_tokens
            + target.completion_tokens
            + evolution.prompt_tokens
            + evolution.completion_tokens
        ),
        "embedding_calls": usage.embedding_calls,
        "wall_time_ms": usage.wall_time_ms,
    }


def _provider_cost_fields(evidence_payload: Mapping[str, Any]) -> dict[str, int]:
    evidence = dict(evidence_payload)
    return {
        "api_calls": int(evidence.get("application_attempts", 0)),
        "physical_provider_calls": int(evidence.get("physical_provider_calls", 0)),
        "provider_retries": int(evidence.get("provider_retries", 0)),
        "cached_provider_calls": int(evidence.get("cached_provider_calls", 0)),
        "provider_queue_wait_ms": int(evidence.get("provider_queue_wait_ms", 0)),
        "provider_service_latency_ms": int(
            evidence.get("provider_service_latency_ms", 0)
        ),
        "logical_call_latency_ms": int(evidence.get("logical_call_latency_ms", 0)),
        "retry_backoff_ms": int(evidence.get("retry_backoff_ms", 0)),
    }


def _training_cost_row(cost_payload: Mapping[str, Any]) -> dict[str, Any]:
    cost = dict(cost_payload)
    train = dict(cost.get("train_episodes") or {})
    validation = dict(cost.get("validation_episodes") or {})
    row: dict[str, Any] = {
        "train_pool_size": int(cost.get("train_pool_size", 0)),
        "validation_pool_size": int(cost.get("validation_pool_size", 0)),
        "unique_train_tasks_evaluated": int(
            cost.get("unique_train_tasks_evaluated", 0)
        ),
        "unique_validation_tasks_evaluated": int(
            cost.get("unique_validation_tasks_evaluated", 0)
        ),
        "train_episodes": int(train.get("episodes", 0)),
        "validation_episodes": int(validation.get("episodes", 0)),
        "episodes": int(train.get("episodes", 0))
        + int(validation.get("episodes", 0)),
        "train_environment_actions": int(train.get("environment_actions", 0)),
        "validation_environment_actions": int(
            validation.get("environment_actions", 0)
        ),
        "environment_actions": int(train.get("environment_actions", 0))
        + int(validation.get("environment_actions", 0)),
        **_usage_cost_fields(dict(cost.get("usage") or {})),
        **_provider_cost_fields(dict(cost.get("provider_evidence") or {})),
    }
    api_cost = cost.get("api_cost")
    if isinstance(api_cost, (int, float)) and not isinstance(api_cost, bool):
        row["api_cost"] = float(api_cost)
    row["api_cost_unpriced"] = bool(cost.get("api_cost_unpriced", True))
    return row


def _test_cost_row(cost_payload: Mapping[str, Any]) -> dict[str, Any]:
    cost = dict(cost_payload)
    episodes = dict(cost.get("episodes") or {})
    row: dict[str, Any] = {
        "episodes": int(episodes.get("episodes", 0)),
        "environment_actions": int(episodes.get("environment_actions", 0)),
        **_usage_cost_fields(dict(cost.get("usage") or {})),
        **_provider_cost_fields(dict(cost.get("provider_evidence") or {})),
    }
    api_cost = cost.get("api_cost")
    if isinstance(api_cost, (int, float)) and not isinstance(api_cost, bool):
        row["api_cost"] = float(api_cost)
    row["api_cost_unpriced"] = bool(cost.get("api_cost_unpriced", True))
    return row


def _phase_cost_summary(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    training_by_seed = {
        str(lane["seed"]): _training_cost_row(
            dict(lane.get("training_cost") or {})
        )
        for lane in lanes
    }
    test_by_seed = {
        str(lane["seed"]): _test_cost_row(dict(lane.get("test_cost") or {}))
        for lane in lanes
    }
    combined_by_seed: dict[str, dict[str, Any]] = {}
    additive = (
        "episodes",
        "environment_actions",
        "target_llm_calls",
        "evolution_llm_calls",
        "llm_calls",
        "target_prompt_tokens",
        "target_completion_tokens",
        "target_reasoning_tokens",
        "evolution_prompt_tokens",
        "evolution_completion_tokens",
        "evolution_reasoning_tokens",
        "llm_tokens",
        "embedding_calls",
        "wall_time_ms",
        "api_calls",
        "physical_provider_calls",
        "provider_retries",
        "cached_provider_calls",
        "provider_queue_wait_ms",
        "provider_service_latency_ms",
        "logical_call_latency_ms",
        "retry_backoff_ms",
    )
    for seed in training_by_seed:
        train = training_by_seed[seed]
        test = test_by_seed[seed]
        combined: dict[str, Any] = {
            key: int(train.get(key, 0)) + int(test.get(key, 0))
            for key in additive
        }
        unpriced = bool(train.get("api_cost_unpriced", True)) or bool(
            test.get("api_cost_unpriced", True)
        )
        combined["api_cost_unpriced"] = unpriced
        if not unpriced and "api_cost" in train and "api_cost" in test:
            combined["api_cost"] = float(train["api_cost"]) + float(test["api_cost"])
        combined_by_seed[seed] = combined
    return {
        "schema_version": 1,
        "statistics_definition": {
            "unit": "method_seed",
            "mean": "arithmetic_mean_over_seeds_42_43_44",
            "std": "sample_standard_deviation",
            "std_ddof": 1,
            "token_total": "prompt_plus_completion_reasoning_not_double_counted",
            "training": "successful_committed_train_attempt_including_validation",
            "test": "frozen_heldout_test",
            "combined": "training_plus_test_committed_cost",
            "retry_overhead": "reported_separately_in_cost_accounting",
        },
        "training": {
            "by_seed": training_by_seed,
            "mean_std": _numeric_mapping_summary(training_by_seed),
        },
        "test": {
            "by_seed": test_by_seed,
            "mean_std": _numeric_mapping_summary(test_by_seed),
        },
        "combined": {
            "by_seed": combined_by_seed,
            "mean_std": _numeric_mapping_summary(combined_by_seed),
        },
    }


_REQUIRED_GEPA_METRICS = (
    "configured_max_metric_calls",
    "actual_total_metric_calls",
    "budget_overshoot",
    "candidate_count",
    "accepted_candidate_count",
    "num_full_val_evals",
    "reflection_calls",
    "best_validation_score",
    "best_skill_tokens",
    "best_skill_sha256",
)


def _gepa_method_summary(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        seed = str(lane["seed"])
        metrics = dict(dict(lane.get("training_cost") or {}).get("gepa_metrics") or {})
        missing = sorted(key for key in _REQUIRED_GEPA_METRICS if key not in metrics)
        if missing:
            raise ValueError(
                f"seed {seed} GEPA report lacks method metrics: {', '.join(missing)}"
            )
        by_seed[seed] = metrics
    return {
        "schema_version": 1,
        "statistics_definition": {
            "unit": "method_seed",
            "mean": "arithmetic_mean_over_seeds_42_43_44",
            "std": "sample_standard_deviation",
            "std_ddof": 1,
            "actual_total_metric_calls": (
                "GEPA cumulative task-level metric evaluations at termination"
            ),
            "num_full_val_evals": "GEPA cumulative complete Validation24 evaluations",
            "candidate_count": "GEPA final candidate pool size including seed candidate",
            "reflection_calls": "GEPA cumulative reflection proposals",
        },
        "by_seed": by_seed,
        "mean_std": _numeric_mapping_summary(by_seed),
        "best_skill_sha256_by_seed": {
            seed: str(metrics["best_skill_sha256"])
            for seed, metrics in by_seed.items()
        },
    }


def run_campaign(
    spec: GEPACampaignSpec,
    *,
    command_runner: CommandRunner | None = None,
    source_inspector: SourceInspector | None = None,
    lock_builder: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    command_runner = command_runner or _default_command_runner
    source_inspector = source_inspector or inspect_clean_source
    lock_builder = lock_builder or build_campaign_lock
    campaign_run_id = (
        "b5_campaign_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    with _method_campaign_lease(spec.repo_root, owner=campaign_run_id):
        if spec.output_dir.exists():
            raise FileExistsError(spec.output_dir)
        source = source_inspector(spec.repo_root)
        payload = lock_builder(spec, spec.output_dir, campaign_run_id, source)
        spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
        spec.output_dir.mkdir(exist_ok=False)
        runtime_spec, payload, probe_report = _run_provider_probe_with_fallback(
            spec, payload, command_runner=command_runner
        )
        if source_inspector(spec.repo_root) != source:
            raise RuntimeError("controller source changed during B5 provider preflight")
        lock_path = spec.output_dir / "campaign_lock.json"
        _write_json_atomic(lock_path, payload, overwrite=False)
        lock_digest = _sha256_file(lock_path)
        campaign_started_at_unix = time.time()
        started = time.perf_counter()
        lanes = [
            _run_lane(
                runtime_spec,
                seed=seed,
                campaign_lock=lock_path,
                lock_digest=lock_digest,
                command_runner=command_runner,
            )
            for seed in FORMAL_SEEDS
        ]
        if _sha256_file(lock_path) != lock_digest:
            raise RuntimeError("B5 campaign lock changed during execution")
        if source_inspector(spec.repo_root) != source:
            raise RuntimeError("controller source changed during B5 campaign")
        passed = all(lane.get("passed") is True for lane in lanes)
        completed = [lane for lane in lanes if lane.get("passed") is True]
        paper_report: dict[str, Any] = {}
        paper_method_report: dict[str, Any] = {}
        paper_report_receipt: dict[str, Any] = {}
        if passed:
            paper_report = build_campaign_report(
                {
                    METHOD_ID: [
                        Path(str(lane["test_root"])) for lane in completed
                    ]
                },
                expected_seeds=FORMAL_SEEDS,
            )
            paper_method_report = dict(
                dict(paper_report.get("methods") or {}).get(METHOD_ID) or {}
            )
            if not paper_method_report:
                raise RuntimeError("common reporter omitted the B5 method summary")
            paper_report_path = spec.output_dir / "paper_report.json"
            _write_json_atomic(paper_report_path, paper_report, overwrite=False)
            paper_report_receipt = {
                "path": str(paper_report_path.resolve()),
                "sha256": _sha256_file(paper_report_path),
                "source": paper_report.get("source"),
                "paired_task_key": paper_report.get("paired_task_key"),
                "bootstrap": dict(paper_report.get("bootstrap") or {}),
            }
        report = {
            "schema_version": 1,
            "passed": passed,
            "method": METHOD_ID,
            "campaign_id": payload["campaign_id"],
            "campaign_run_id": campaign_run_id,
            "campaign_root": str(spec.output_dir),
            "campaign_lock": str(lock_path),
            "campaign_lock_digest": lock_digest,
            "seeds": list(FORMAL_SEEDS),
            "provider_probe": probe_report,
            "campaign_started_at_unix": campaign_started_at_unix,
            "campaign_completed_at_unix": time.time(),
            "campaign_makespan_ms": int((time.perf_counter() - started) * 1000),
            "lanes": lanes,
            "completed_seeds": [int(row["seed"]) for row in completed],
            "failed_seeds": [
                int(row["seed"]) for row in lanes if row.get("passed") is not True
            ],
            "test_metrics_mean_std": (
                dict(paper_method_report.get("mean_std") or {}) if passed else {}
            ),
            "test_per_family_mean_std": (
                dict(
                    paper_method_report.get(
                        "family_official_success_rate_mean_std"
                    ) or {}
                )
                if passed else {}
            ),
            "effectiveness": paper_method_report if passed else {},
            "paper_report": paper_report_receipt,
            "gepa_method_metrics": _gepa_method_summary(completed) if passed else {},
            "phase_costs": _phase_cost_summary(completed) if passed else {},
            "resource_usage": _resource_summary(completed),
            "cost_accounting": _campaign_cost_summary(completed),
            "actual_attempt_usage": _campaign_actual_usage(lanes),
            "all_frozen_digests_verified": passed,
        }
        _write_json_atomic(
            spec.output_dir / "campaign_report.json", report, overwrite=False
        )
        _write_json_atomic(
            spec.output_dir / ("completion.json" if passed else "campaign_failure.json"),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--config", default="configs/baselines/b5_gepa.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default=None)
    args = parser.parse_args(argv)
    try:
        repo_root = REPO_ROOT.resolve()
        config = _resolve(repo_root, args.config)
        merged = {
            **dict(
                yaml.safe_load(
                    (repo_root / "configs/baselines/common.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                or {}
            ),
            **dict(yaml.safe_load(config.read_text(encoding="utf-8")) or {}),
        }
        configured_python = resolve_formal_python(
            repo_root, str(merged.get("worker_python", ""))
        )
        if args.python is not None and str(
            resolve_formal_python(repo_root, args.python)
        ) != str(configured_python):
            raise ValueError("--python must exactly match B5 worker_python")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = _resolve(
            repo_root,
            args.output_dir
            or f"runs/baselines/{merged['campaign_id']}/{METHOD_ID}/formal_3seed_{stamp}",
        )
        spec = GEPACampaignSpec(
            method=METHOD_ID,
            seeds=tuple(args.seeds),
            train_manifest=_resolve(repo_root, args.train_manifest),
            validation_manifest=_resolve(repo_root, args.validation_manifest),
            test_manifest=_resolve(repo_root, args.test_manifest),
            config=config,
            output_dir=output,
            python=configured_python,
            repo_root=repo_root,
        )
        report = run_campaign(spec)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.get("passed") is True else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
