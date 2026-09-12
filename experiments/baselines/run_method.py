"""Run the B3 SkillOpt formal or pilot comparison protocol.

The formal ``train`` command performs one identity-bound workflow::

    upstream SkillOpt training on Train120
      -> upstream Validation24 selection gate
      -> immutable best_skill.md freeze
      -> training/selection cost report

The formal train worker never receives Test134. A separate ``test`` command
loads the matching train run's immutable frozen artifact and evaluates it
exactly once on complete held-out Test134 without constructing a trainer.
The retained pilot profile may additionally perform a Train30 resubstitution
diagnostic; that result is never a formal generalization claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    hash_code,
    sanitize_error_text,
    sha256_json,
)

from .b3_skillopt.driver import load_lock_and_driver
from .common.driver import RunContext
from .common.formal_validation import (
    verify_final_evaluation_bijection,
    verify_formal_manifest,
)
from .common.freeze import FrozenArtifact, assert_frozen_unchanged
from .common.integrity import assert_no_secrets_on_disk, validate_episode_usage
from .common.manifest import TaskManifestSet, verify_disjoint
from .common.model_config import ModelConfig
from .common.runtime_python import resolve_formal_python, verify_runtime_python
from .common.post_evaluator import TaskRow, summarize_rows, write_rows_jsonl
from .common.schema import CommonEpisodeRecord
from .common.usage import UsageSnapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
_METHOD = "b3_skillopt"


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_manifest(path: str | Path) -> TaskManifestSet:
    return TaskManifestSet.load(_path(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alfworld_data_signature(data_root: Path) -> dict[str, Any]:
    """Hash the logic/grammar and complete physical gamefile corpus."""

    data_root = data_root.resolve(strict=True)
    logic_root = data_root / "logic"
    required = {
        "logic_sha256": logic_root / "alfred.pddl",
        "grammar_sha256": logic_root / "alfred.twl2",
    }
    payload: dict[str, Any] = {"resolved_data_root": str(data_root)}
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"ALFWorld authority file is missing: {path}")
        payload[name] = _sha256_file(path)

    dataset_root = data_root / "json_2.1.1"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ALFWorld dataset root is missing: {dataset_root}")
    corpus = hashlib.sha256()
    count = 0
    for path in sorted(dataset_root.rglob("game.tw-pddl")):
        relative = path.relative_to(data_root).as_posix()
        corpus.update(relative.encode("utf-8"))
        corpus.update(b"\0")
        corpus.update(_sha256_file(path).encode("ascii"))
        corpus.update(b"\n")
        count += 1
    if count <= 0:
        raise RuntimeError(f"ALFWorld dataset has no game.tw-pddl files: {dataset_root}")
    payload.update({
        "dataset_gamefile_count": count,
        "dataset_content_merkle_sha256": corpus.hexdigest(),
    })
    return payload


def _new_run_id(*, phase: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"b3_{phase}_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _controller_git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "controller git inspection failed: "
                + (completed.stderr.strip() or "unknown git error")
            )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("controller git HEAD is not a full commit id")
    status = git(
        "status", "--porcelain", "--untracked-files=all", "--",
        "src", "experiments", "configs",
    )
    return {
        "commit": commit,
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_scope": ["src", "experiments", "configs"],
    }


def _formal_config_digest(config: dict[str, Any]) -> str:
    """Hash the seed-independent frozen method configuration."""

    normalized = json.loads(json.dumps(config))
    normalized.pop("protocol", None)
    normalized["run_seed"] = "<campaign-seed>"
    train = dict(normalized.get("train") or {})
    train["seed"] = "<campaign-seed>"
    normalized["train"] = train
    return sha256_json(normalized)


def _provider_retry_policy(config: dict[str, Any]) -> dict[str, Any]:
    transport = dict(config.get("provider_transport") or {})
    return {
        "sdk_max_retries": int(transport.get("sdk_max_retries", -1)),
        "attempts": int(transport.get("application_retry_limit", -1)),
        "delays": [
            float(value) for value in transport.get("retry_delays_seconds", [])
        ],
        "jitter_ratio": float(
            transport.get("deterministic_jitter_ratio", -1)
        ),
    }


def _validate_campaign_probe_receipt(
    payload: dict[str, Any],
    *,
    campaign_root: Path,
    model: ModelConfig,
) -> None:
    probe = dict(payload.get("provider_probe") or {})
    receipt = dict(payload.get("provider_probe_receipt") or {})
    report_path = Path(str(receipt.get("path", ""))).expanduser().resolve()
    try:
        report_path.relative_to(campaign_root.resolve())
    except ValueError as exc:
        raise ValueError("campaign provider-probe report is outside campaign root") from exc
    if not report_path.is_file():
        raise FileNotFoundError(
            f"campaign provider-probe report is missing: {report_path}"
        )
    report_hash = _sha256_file(report_path)
    if (
        int(receipt.get("returncode", -1)) != 0
        or str(receipt.get("sha256", "")) != report_hash
        or str(probe.get("report_sha256", "")) != report_hash
        or Path(str(probe.get("report_path", ""))).expanduser().resolve()
        != report_path
    ):
        raise ValueError("campaign provider-probe receipt/hash is invalid")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign provider-probe report is unreadable") from exc
    if not isinstance(report, dict) or dict(receipt.get("report") or {}) != report:
        raise ValueError("campaign provider-probe receipt does not bind its report")
    cap = int(payload.get("campaign_provider_max_inflight", 0))
    requests = int(probe.get("requests", 0))
    expected = {
        "probe_kind": "campaign_provider_load",
        "passed": True,
        "campaign_id": str(payload.get("campaign_id", "")),
        "run_id": str(payload.get("campaign_run_id", "")) + "_probe",
        "model": model.model,
        "reasoning_effort": model.reasoning_effort,
        "concurrency": cap,
        "requests": requests,
        "max_completion_tokens": int(probe.get("max_completion_tokens", 0)),
        "campaign_provider_max_inflight": cap,
        "logical_calls_recorded": requests,
        "completed_logical_calls": requests,
        "failed_provider_calls": 0,
        "exhausted_provider_calls": 0,
        "permanent_provider_errors": 0,
        "provider_evidence_complete": True,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if report.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "campaign provider-probe report identity is invalid: "
            + ", ".join(mismatches)
        )
    calls_path = Path(str(report.get("provider_calls_path", ""))).expanduser().resolve()
    try:
        calls_path.relative_to(report_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("provider-probe calls sidecar is outside its probe dir") from exc
    if (
        not calls_path.is_file()
        or _sha256_file(calls_path) != report.get("provider_calls_sha256")
    ):
        raise ValueError("provider-probe calls sidecar hash is invalid")


def _load_campaign_descriptor(
    path: str | Path,
    *,
    config: dict[str, Any],
    source_lock: dict[str, Any],
    model: ModelConfig,
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    test_manifest: TaskManifestSet,
    git_state: dict[str, Any],
    code_digest: str,
    seed: int,
) -> dict[str, Any]:
    lock_path = _path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"campaign lock is missing: {lock_path}")
    raw = lock_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"campaign lock is unreadable: {lock_path}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("campaign lock has an invalid schema version")
    parallel = dict(config.get("parallel") or {})
    transport = dict(config.get("provider_transport") or {})
    expected_retry = {
        "sdk_max_retries": int(transport.get("sdk_max_retries", -1)),
        "attempts": int(transport.get("application_retry_limit", -1)),
        "delays": [float(value) for value in transport.get("retry_delays_seconds", [])],
        "jitter_ratio": float(transport.get("deterministic_jitter_ratio", -1)),
    }
    runtime_tree = dict(source_lock["skillopt"]["runtime_tree"])
    configured_python = resolve_formal_python(
        REPO_ROOT, str(config.get("worker_python", "")),
    )
    locked_python = {
        field: resolve_formal_python(REPO_ROOT, str(payload.get(field, "")))
        for field in ("phase_python", "worker_python", "provider_probe_python")
    }
    checks = {
        "method": payload.get("method") == _METHOD,
        "seed": int(seed) in [int(value) for value in payload.get("seeds", [])],
        "train_manifest": payload.get("train_manifest_digest") == train_manifest.digest,
        "validation_manifest": payload.get("validation_manifest_digest")
        == validation_manifest.digest,
        "test_manifest": payload.get("test_manifest_digest") == test_manifest.digest,
        "controller_commit": payload.get("controller_commit") == git_state.get("commit"),
        "controller_code": payload.get("controller_code_digest") == code_digest,
        "external_commit": payload.get("external_skillopt_commit")
        == str(source_lock["skillopt"]["commit"]),
        "external_runtime": payload.get("external_runtime_tree_digest")
        == str(runtime_tree["sha256"]),
        "model": payload.get("model") == model.model,
        "reasoning_effort": payload.get("reasoning_effort") == model.reasoning_effort,
        "formal_config": payload.get("formal_config_digest")
        == _formal_config_digest(config),
        "seed_lanes": int(payload.get("seed_lanes", 0))
        == int(parallel.get("seed_lanes", 0)) == 3,
        "provider_cap": int(payload.get("campaign_provider_max_inflight", 0))
        == int(parallel.get("campaign_provider_max_inflight", 0)),
        "retry_policy": payload.get("retry_policy") == expected_retry,
        "provider_probe": dict(payload.get("provider_probe") or {}).get("passed") is True,
        "python_authority": all(
            str(value) == str(configured_python)
            for value in locked_python.values()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "campaign lock does not match this formal run: " + ", ".join(failed)
        )
    _validate_campaign_probe_receipt(
        payload,
        campaign_root=lock_path.parent,
        model=model,
    )
    gate_dir = Path(str(payload.get("provider_gate_dir", ""))).expanduser().resolve()
    try:
        gate_dir.relative_to(lock_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("campaign provider gate must be inside the campaign root") from exc
    if not str(payload.get("campaign_id", "")).strip():
        raise ValueError("campaign lock has no campaign_id")
    return {
        "campaign_id": str(payload["campaign_id"]),
        "campaign_lock_path": str(lock_path),
        "campaign_lock_digest": digest,
        "provider_gate_dir": str(gate_dir),
        "campaign_provider_max_inflight": int(
            payload["campaign_provider_max_inflight"]
        ),
        "phase_python": str(locked_python["phase_python"]),
        "worker_python": str(locked_python["worker_python"]),
        "provider_probe_python": str(locked_python["provider_probe_python"]),
    }


def _safe_error(exc: BaseException, *, api_key_env: str = "MODEL_API_KEY") -> str:
    text = sanitize_error_text(exc)
    live_key = os.environ.get(api_key_env, "").strip()
    if live_key:
        text = text.replace(live_key, "[REDACTED]")
    return text


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _validate_training_episode_membership(
    episodes: Iterable[CommonEpisodeRecord],
    manifest: TaskManifestSet,
    *,
    phase: str,
    run_seed: int,
) -> dict[str, Any]:
    observed = list(episodes)
    expected = {task.task_id: task for task in manifest.tasks}
    observed_ids: set[str] = set()
    for episode in observed:
        task = expected.get(episode.task_id)
        if task is None:
            raise ValueError(
                f"{phase} episode {episode.task_id!r} is outside its manifest"
            )
        mismatches: list[str] = []
        if episode.method != _METHOD:
            mismatches.append("method")
        if episode.phase != phase:
            mismatches.append("phase")
        if episode.run_seed != run_seed:
            mismatches.append("run_seed")
        if episode.manifest_index != task.index:
            mismatches.append("manifest_index")
        if episode.task_type != task.task_type:
            mismatches.append("task_type")
        if episode.gamefile != task.gamefile_rel:
            mismatches.append("gamefile")
        if episode.gamefile_hash != task.gamefile_sha256:
            mismatches.append("gamefile_hash")
        if episode.infrastructure_failure:
            mismatches.append("infrastructure_failure")
        if mismatches:
            raise RuntimeError(
                f"{phase} episode {episode.task_id!r} has invalid formal evidence: "
                + ", ".join(mismatches)
            )
        observed_ids.add(episode.task_id)
    missing = [task.task_id for task in manifest.tasks if task.task_id not in observed_ids]
    if not observed or missing:
        raise RuntimeError(
            f"{phase} did not cover every manifest task at least once; "
            f"episodes={len(observed)}, missing={missing[:5]}"
        )
    return {
        "episodes": len(observed),
        "unique_tasks": len(observed_ids),
        "environment_actions": sum(item.environment_actions for item in observed),
    }


def _episode_cost(episodes: list[CommonEpisodeRecord]) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "environment_actions": sum(int(item.environment_actions) for item in episodes),
        "target_llm_calls": sum(int(item.target_llm_calls) for item in episodes),
        "target_prompt_tokens": sum(int(item.target_prompt_tokens) for item in episodes),
        "target_completion_tokens": sum(
            int(item.target_completion_tokens) for item in episodes
        ),
        "target_reasoning_tokens": sum(
            int(item.target_reasoning_tokens) for item in episodes
        ),
        "wall_time_ms_episode_sum": sum(int(item.wall_time_ms) for item in episodes),
    }


def _require_phase_usage(usage: UsageSnapshot, *, phase: str, evolution: bool) -> None:
    if usage.target.calls <= 0:
        raise RuntimeError(f"{phase} completed without target provider calls")
    if evolution and usage.evolution.calls <= 0:
        raise RuntimeError(f"{phase} completed without SkillOpt evolution calls")


def _create_context(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    lock: dict[str, Any],
    model: ModelConfig,
    alfworld_data: Path,
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    test_manifest: TaskManifestSet | None,
    data_signature: dict[str, Any],
    output_dir: Path,
    run_id: str,
    campaign: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
) -> RunContext:
    code_digest = hash_code(REPO_ROOT)
    config_digest = sha256_json(config)
    identity = {
        "config_digest": config_digest,
        "train_manifest_digest": train_manifest.digest,
        "validation_manifest_digest": validation_manifest.digest,
        "controller_code_digest": code_digest,
        "skillopt_runtime_digest": str(lock["skillopt"]["runtime_tree"]["sha256"]),
        "model_identity_digest": sha256_json(model.to_wire()),
        "alfworld_data_digest": sha256_json(data_signature),
        "formal_config_digest": _formal_config_digest(config),
    }
    if test_manifest is not None:
        identity["test_manifest_digest"] = test_manifest.digest
    resolved_config_path = output_dir / "config_resolved.json"
    _write_json_atomic(resolved_config_path, config)
    return RunContext(
        campaign_id=str(config.get("campaign_id", "pilot")),
        method_id=_METHOD,
        run_seed=int(config.get("run_seed", 42)),
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        external_repo=REPO_ROOT / ".external" / "skillopt",
        external_commit=str(lock["skillopt"]["commit"]),
        model_config=model,
        max_environment_actions=int(config.get("max_environment_actions", 100)),
        alfworld_data=alfworld_data,
        config_hash=config_digest,
        code_hash=code_digest,
        run_id=run_id,
        resolved_config_path=resolved_config_path.resolve(),
        identity=identity,
        train_manifest_path=_path(args.train_manifest),
        validation_manifest_path=_path(args.validation_manifest),
        test_manifest_path=(
            _path(args.test_manifest) if getattr(args, "test_manifest", None) else None
        ),
        campaign=(dict(campaign) if campaign is not None else None),
        resume=(dict(resume) if resume is not None else None),
    )


def _write_identity_artifacts(
    *,
    ctx: RunContext,
    lock: dict[str, Any],
    config: dict[str, Any],
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    test_manifest: TaskManifestSet | None,
    train_preflight: dict[str, Any],
    validation_preflight: dict[str, Any],
    test_preflight: dict[str, Any] | None,
    data_signature: dict[str, Any],
    git_state: dict[str, Any],
    phase: str,
    python_runtime: dict[str, Any] | None,
) -> None:
    formal = str(config.get("protocol_profile")) == "formal_v2"
    workflow = {
        "train": (
            "train120_validation24_freeze" if formal
            else "train30_validation6_freeze_train30_read_only_replay"
        ),
        "smoke": "two_task_transport_and_environment_smoke",
        "test": (
            "prior_frozen_skill_test134_read_only_evaluation" if formal
            else "prior_frozen_skill_test60_read_only_evaluation"
        ),
    }[phase]
    provenance = {
        "schema_version": 2,
        "protocol": (
            "protocol-faithful-matched-train-v2"
            if formal
            else "pilot-v1"
        ),
        "formal": formal,
        "run_id": ctx.run_id,
        "method": _METHOD,
        "phase": phase,
        "workflow": workflow,
        "evaluation_scope": (
            ("training_and_selection_only" if formal else "train_resubstitution")
            if phase == "train" else (
                "held_out_test" if phase == "test" else "smoke_only"
            )
        ),
        "held_out": phase == "test",
        "generalization_claim": phase == "test",
        "external_repo": str(lock["skillopt"]["repo"]),
        "external_commit": str(lock["skillopt"]["commit"]),
        "external_version": str(lock["skillopt"].get("version") or ""),
        "external_runtime_tree": dict(lock["skillopt"]["runtime_tree"]),
        "controller_git": git_state,
        "controller_code_digest": ctx.code_hash,
        "python_runtime": dict(python_runtime) if python_runtime is not None else None,
        "identity": dict(ctx.identity),
        "campaign": dict(ctx.campaign) if ctx.campaign is not None else None,
        "resume": dict(ctx.resume) if ctx.resume is not None else None,
        "train_manifest_hash": train_manifest.digest,
        "validation_manifest_hash": validation_manifest.digest,
        "test_manifest_hash": test_manifest.digest if test_manifest else None,
        "alfworld_data": data_signature,
        "model": ctx.model_config.to_wire(),
        "reasoning_effort": ctx.model_config.reasoning_effort,
        "max_environment_actions": ctx.max_environment_actions,
        "run_seed": ctx.run_seed,
        "episode_workers": int(dict(config.get("env") or {}).get("workers", 1)),
        "test_workers": int(dict(config.get("env") or {}).get("workers", 1)),
        "provider_max_inflight": int(
            ctx.campaign["campaign_provider_max_inflight"]
            if ctx.campaign is not None
            else dict(config.get("env") or {}).get("max_api_workers", 1)
        ),
        "provider_retry_policy": _provider_retry_policy(config),
        "skillopt_analyst_workers": int(
            dict(config.get("gradient") or {}).get("analyst_workers", 1)
        ),
        "episode_executor": "thread_pool",
        "mp_start_method": None,
        "initial_skill_sha256": str(
            dict(lock["skillopt"].get("key_files") or {}).get(
                "skillopt/envs/alfworld/skills/initial.md", ""
            )
        ),
        "method_specific_decoding": {
            key: config.get(key)
            for key in ("train", "gradient", "optimizer", "evaluation", "env", "smoke")
            if key in config
        },
        "started_at_unix": time.time(),
    }
    _write_json_atomic(ctx.output_dir / "run_manifest.json", provenance)
    _write_json_atomic(ctx.output_dir / "source_lock.json", lock)
    _write_json_atomic(ctx.output_dir / "task_manifest.json", {
        "schema_version": 2,
        "train": {
            "path": str(ctx.train_manifest_path),
            "manifest_id": train_manifest.manifest_id,
            "digest": train_manifest.digest,
            "tasks": len(train_manifest.tasks),
            "preflight": train_preflight,
        },
        "validation": {
            "path": str(ctx.validation_manifest_path),
            "manifest_id": validation_manifest.manifest_id,
            "digest": validation_manifest.digest,
            "tasks": len(validation_manifest.tasks),
            "preflight": validation_preflight,
        },
        "test": ({
            "path": str(ctx.test_manifest_path),
            "manifest_id": test_manifest.manifest_id,
            "digest": test_manifest.digest,
            "tasks": len(test_manifest.tasks),
            "preflight": test_preflight,
        } if test_manifest is not None else None),
    })
    _write_json_atomic(ctx.output_dir / "run_state.json", {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "state": "running",
        "phase": phase,
        "updated_at_unix": time.time(),
    })


def _run_smoke(
    *,
    driver: Any,
    ctx: RunContext,
    train_manifest: TaskManifestSet,
    model: ModelConfig,
) -> dict[str, Any]:
    smoke = driver.smoke(ctx, train_manifest)
    episodes = list(smoke.episodes)
    resolved_config = json.loads(
        ctx.resolved_config_path.read_text(encoding="utf-8")
    )
    expected_tasks = int(resolved_config["smoke"]["task_count"])
    if len(episodes) != expected_tasks:
        raise RuntimeError(
            f"smoke must persist exactly {expected_tasks} episodes, "
            f"got {len(episodes)}"
        )
    failures = [
        episode for episode in episodes if episode.infrastructure_failure
    ]
    if failures:
        raise RuntimeError(
            f"smoke infrastructure failure: {failures[0].infrastructure_error}"
        )
    validate_episode_usage(episodes)
    _require_phase_usage(smoke.usage, phase="smoke", evolution=False)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=model.api_key_env)
    worker_result = json.loads(
        (ctx.output_dir / "smoke" / "worker_result.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "method": _METHOD,
        "phase": "smoke",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "task_ids": [episode.task_id for episode in episodes],
        "official_successes": sum(
            int(episode.official_success) for episode in episodes
        ),
        "task_success_required": False,
        "environment_actions": sum(
            episode.environment_actions for episode in episodes
        ),
        "usage": smoke.usage.to_dict(),
        "worker": worker_result,
    }
    _write_json_atomic(ctx.output_dir / "smoke_report.json", report)
    return report


def _run_train(
    *,
    driver: Any,
    ctx: RunContext,
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    model: ModelConfig,
    profile: str,
) -> dict[str, Any]:
    train_result = driver.train(ctx, train_manifest, validation_manifest)
    train_coverage = _validate_training_episode_membership(
        train_result.episodes, train_manifest, phase="train", run_seed=ctx.run_seed,
    )
    validation_coverage = _validate_training_episode_membership(
        train_result.validation_episodes,
        validation_manifest,
        phase="validation",
        run_seed=ctx.run_seed,
    )
    training_episodes = [*train_result.episodes, *train_result.validation_episodes]
    validate_episode_usage(training_episodes)
    _require_phase_usage(train_result.usage, phase="train", evolution=True)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=model.api_key_env)

    frozen = driver.freeze(ctx, train_result)
    if profile == "formal_v2":
        report = {
            "schema_version": 3,
            "passed": True,
            "method": _METHOD,
            "run_id": ctx.run_id,
            "output_dir": str(ctx.output_dir),
            "protocol": {
                "name": "protocol-faithful-matched-train-v2",
                "workflow": "train120_validation24_freeze",
                "evaluation_scope": "training_and_selection_only",
                "held_out": False,
                "generalization_claim": False,
                "formal": True,
            },
            "frozen": {
                "root": str(frozen.root),
                "digest": frozen.digest,
                "source_train_manifest_hash": frozen.source_train_manifest_hash,
                "source_validation_manifest_hash": frozen.source_validation_manifest_hash,
            },
            "training_cost": {
                "unique_train_tasks": len(train_manifest.tasks),
                "unique_validation_tasks": len(validation_manifest.tasks),
                "train_coverage": train_coverage,
                "validation_coverage": validation_coverage,
                "train_episodes": _episode_cost(train_result.episodes),
                "validation_episodes": _episode_cost(train_result.validation_episodes),
                "usage": train_result.usage.to_dict(),
                "api_cost": train_result.usage.api_cost,
                "api_cost_unpriced": train_result.usage.api_cost_unpriced,
                "skillopt_metrics": train_result.method_metrics,
                "provider_evidence": dict(
                    train_result.method_metrics.get("provider_evidence") or {}
                ),
            },
        }
        _write_json_atomic(ctx.output_dir / "report.json", report)
        assert_no_secrets_on_disk(ctx.output_dir, api_key_env=model.api_key_env)
        return report

    train_eval_episodes = driver.evaluate_train(ctx, frozen, train_manifest)
    bijection = verify_final_evaluation_bijection(
        train_eval_episodes,
        train_manifest,
        role="train",
        expected_phase="train_eval",
        expected_method=_METHOD,
        expected_run_seed=ctx.run_seed,
        expected_artifact_digest=frozen.digest,
        require_strict_outcomes=True,
        profile=profile,
    )
    validate_episode_usage(train_eval_episodes)
    train_eval_usage = UsageSnapshot.load(ctx.output_dir / "train_eval" / "usage.json")
    train_eval_worker_result = json.loads(
        (ctx.output_dir / "train_eval" / "worker_result.json").read_text(
            encoding="utf-8"
        )
    )
    train_eval_provider_evidence = dict(
        train_eval_worker_result.get("provider_evidence") or {}
    )
    _require_phase_usage(train_eval_usage, phase="train_eval", evolution=False)
    if train_eval_usage.evolution.calls != 0:
        raise RuntimeError("train_eval made forbidden SkillOpt evolution calls")

    rows = [TaskRow.from_episode(episode) for episode in train_eval_episodes]
    write_rows_jsonl(rows, ctx.output_dir / "train_eval" / "task_rows.jsonl")
    summary = summarize_rows(rows, task_types=list(ALFWORLD_FORMAL_TASK_TYPES))
    if summary["attempted_tasks"] != 30 or summary["tasks"] != 30:
        raise RuntimeError("Train30 replay summary denominator is not exactly 30")
    if any(values["tasks"] != 5 for values in summary["family"].values()):
        raise RuntimeError("Train30 replay summary is not balanced six families x five")
    _write_json_atomic(ctx.output_dir / "train_eval" / "summary.json", {
        "schema_version": 1,
        "method": _METHOD,
        "phase": "train_eval",
        "evaluation_scope": "train_resubstitution",
        "held_out": False,
        "generalization_claim": False,
        "frozen_digest": frozen.digest,
        "bijection": bijection,
        **summary,
    })

    total_usage = UsageSnapshot.from_dict(train_result.usage.to_dict())
    total_usage.add(train_eval_usage)
    _write_json_atomic(ctx.output_dir / "combined_usage.json", total_usage.to_dict())
    report = {
        "schema_version": 2,
        "passed": True,
        "method": _METHOD,
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "protocol": {
            "workflow": "train30_validation30_freeze_train30_read_only_replay",
            "evaluation_scope": "train_resubstitution",
            "held_out": False,
            "generalization_claim": False,
            "accuracy_denominator": 30,
        },
        "frozen": {
            "root": str(frozen.root),
            "digest": frozen.digest,
            "source_train_manifest_hash": frozen.source_train_manifest_hash,
            "source_validation_manifest_hash": frozen.source_validation_manifest_hash,
        },
        "effectiveness": {
            "train_resubstitution": summary,
            "validation_selection": train_result.method_metrics,
        },
        "training_cost": {
            "unique_train_tasks": len(train_manifest.tasks),
            "unique_validation_tasks": len(validation_manifest.tasks),
            "train_coverage": train_coverage,
            "validation_coverage": validation_coverage,
            "train_episodes": _episode_cost(train_result.episodes),
            "validation_episodes": _episode_cost(train_result.validation_episodes),
            "usage": train_result.usage.to_dict(),
            "api_cost": train_result.usage.api_cost,
            "api_cost_unpriced": train_result.usage.api_cost_unpriced,
            "skillopt_metrics": train_result.method_metrics,
            "provider_evidence": dict(
                train_result.method_metrics.get("provider_evidence") or {}
            ),
        },
        "train_replay_cost": {
            "episodes": _episode_cost(train_eval_episodes),
            "usage": train_eval_usage.to_dict(),
            "api_cost": train_eval_usage.api_cost,
            "api_cost_unpriced": train_eval_usage.api_cost_unpriced,
            "provider_evidence": train_eval_provider_evidence,
        },
        "combined_usage": total_usage.to_dict(),
        "comparison_metrics": {
            "official_success": summary["official_success"],
            "official_rate": summary["official_rate"],
            "micro_average_official_rate": summary[
                "micro_average_official_rate"
            ],
            "strict_rate": summary["strict_rate"],
            "macro_family_official_rate": summary["macro_family_official_rate"],
            "actions_per_task": summary["actions_per_task"],
            "actions_per_solved": summary["actions_per_solved"],
            "p50_actions": summary["p50_actions"],
            "p90_actions": summary["p90_actions"],
            "calls_per_task": summary["calls_per_task"],
            "tokens_per_task": summary["tokens_per_task"],
            "tokens_per_solved": summary["tokens_per_solved"],
            "latency_per_task_ms": summary["latency_per_task_ms"],
            "p50_latency_ms": summary["p50_latency_ms"],
            "p90_latency_ms": summary["p90_latency_ms"],
            "target_reasoning_tokens": summary["target_reasoning_tokens"],
            "reasoning_tokens_in_completion": summary[
                "reasoning_tokens_in_completion"
            ],
            "delta_vs_pure_dynamic": summary["delta_vs_pure_dynamic"],
            "positive_transfer": summary["positive_transfer"],
            "negative_transfer": summary["negative_transfer"],
            "paired_transfer_status": summary["paired_transfer_status"],
            "family": summary["family"],
        },
    }
    _write_json_atomic(ctx.output_dir / "report.json", report)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=model.api_key_env)
    return report


def _load_source_frozen_run(
    source_run: Path,
    *,
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    test_manifest: TaskManifestSet,
    expected_context: RunContext,
) -> tuple[FrozenArtifact, dict[str, Any]]:
    """Load a completed formal train run and bind every reusable identity."""

    completion_path = source_run / "completion.json"
    report_path = source_run / "report.json"
    run_manifest_path = source_run / "run_manifest.json"
    if (
        not completion_path.is_file()
        or not report_path.is_file()
        or not run_manifest_path.is_file()
    ):
        raise FileNotFoundError(
            "source run must contain completion.json, report.json, and "
            "run_manifest.json: " + str(source_run)
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if completion.get("passed") is not True or completion.get("phase") != "train":
        raise ValueError("source run is not a successfully completed train phase")
    if report.get("passed") is not True or report.get("method") != _METHOD:
        raise ValueError("source run report is not a successful B3 SkillOpt report")
    if (
        source_manifest.get("formal") is not True
        or source_manifest.get("protocol")
        != "protocol-faithful-matched-train-v2"
        or source_manifest.get("phase") != "train"
    ):
        raise ValueError("source run is not a formal-v2 SkillOpt train run")
    expected_campaign = expected_context.campaign
    source_campaign = source_manifest.get("campaign")
    if (
        not isinstance(expected_campaign, dict)
        or not isinstance(source_campaign, dict)
        or source_campaign != expected_campaign
    ):
        raise ValueError(
            "source train run does not belong to this exact campaign lane"
        )
    source_identity_seed = int(source_manifest["run_seed"])
    if source_identity_seed != expected_context.run_seed:
        raise ValueError(
            f"source run seed {source_identity_seed} does not match test seed "
            f"{expected_context.run_seed}"
        )
    expected_manifest_hashes = {
        "train_manifest_hash": train_manifest.digest,
        "validation_manifest_hash": validation_manifest.digest,
        "test_manifest_hash": test_manifest.digest,
    }
    for field, expected in expected_manifest_hashes.items():
        if source_manifest.get(field) != expected:
            raise ValueError(
                f"source run {field} does not match the requested formal campaign"
            )
    if source_manifest.get("external_commit") != expected_context.external_commit:
        raise ValueError("source run SkillOpt commit does not match the test runtime")
    source_runtime = dict(source_manifest.get("external_runtime_tree") or {})
    if source_runtime.get("sha256") != expected_context.identity.get(
        "skillopt_runtime_digest"
    ):
        raise ValueError("source run SkillOpt runtime digest does not match test")
    if source_manifest.get("model") != expected_context.model_config.to_wire():
        raise ValueError("source run model identity does not match test")

    source_identity = dict(source_manifest.get("identity") or {})
    completion_identity = dict(completion.get("identity") or {})
    if completion_identity != source_identity:
        raise ValueError("source completion identity differs from run_manifest identity")
    reusable_identity_fields = (
        "train_manifest_digest",
        "validation_manifest_digest",
        "test_manifest_digest",
        "controller_code_digest",
        "skillopt_runtime_digest",
        "model_identity_digest",
        "alfworld_data_digest",
    )
    for field in reusable_identity_fields:
        expected = expected_context.identity.get(field)
        if not expected or source_identity.get(field) != expected:
            raise ValueError(
                f"source run identity field {field!r} does not match test"
            )
    frozen_dir = source_run / "frozen"
    frozen = FrozenArtifact.load(frozen_dir)
    if frozen.source_train_manifest_hash != train_manifest.digest:
        raise ValueError("source frozen artifact does not match Train120")
    if frozen.source_validation_manifest_hash != validation_manifest.digest:
        raise ValueError("source frozen artifact does not match Validation24")
    reported_frozen = dict(report.get("frozen") or {})
    if reported_frozen.get("digest") != frozen.digest:
        raise ValueError("source report frozen digest does not match frozen artifact")
    if int(frozen.metadata.get("run_seed", -1)) != expected_context.run_seed:
        raise ValueError("frozen artifact metadata does not match the test run seed")
    assert_frozen_unchanged(frozen)
    return frozen, {
        "source_run_id": str(completion.get("run_id") or report.get("run_id") or ""),
        "source_run": str(source_run.resolve()),
        "source_completion_identity": completion_identity,
        "frozen_dir": str(frozen_dir.resolve()),
    }


def _run_test(
    *,
    driver: Any,
    ctx: RunContext,
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    test_manifest: TaskManifestSet,
    source_run: Path,
    model: ModelConfig,
    profile: str,
) -> dict[str, Any]:
    frozen, source = _load_source_frozen_run(
        source_run,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        expected_context=ctx,
    )
    episodes = driver.evaluate_test(
        ctx,
        frozen,
        train_manifest,
        validation_manifest,
        test_manifest,
        frozen_dir=Path(source["frozen_dir"]),
    )
    bijection = verify_final_evaluation_bijection(
        episodes,
        test_manifest,
        role="test",
        expected_phase="test",
        expected_method=_METHOD,
        expected_run_seed=ctx.run_seed,
        expected_artifact_digest=frozen.digest,
        require_strict_outcomes=True,
        profile=profile,
    )
    validate_episode_usage(episodes)
    usage = UsageSnapshot.load(ctx.output_dir / "test" / "usage.json")
    _require_phase_usage(usage, phase="test", evolution=False)
    if usage.evolution.calls != 0:
        raise RuntimeError("test made forbidden SkillOpt evolution calls")
    rows = [TaskRow.from_episode(episode) for episode in episodes]
    write_rows_jsonl(rows, ctx.output_dir / "test" / "task_rows.jsonl")
    summary = summarize_rows(rows, task_types=list(ALFWORLD_FORMAL_TASK_TYPES))
    expected_tasks = len(test_manifest.tasks)
    if (
        summary["attempted_tasks"] != expected_tasks
        or summary["tasks"] != expected_tasks
    ):
        raise RuntimeError(
            f"Test summary denominator is not exactly {expected_tasks}"
        )
    summary_payload = {
        "schema_version": 1,
        "method": _METHOD,
        "phase": "test",
        "evaluation_scope": "held_out_test",
        "held_out": True,
        "generalization_claim": True,
        "frozen_digest": frozen.digest,
        "bijection": bijection,
        **summary,
    }
    _write_json_atomic(ctx.output_dir / "test" / "summary.json", summary_payload)
    worker = json.loads(
        (ctx.output_dir / "test" / "worker_result.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "method": _METHOD,
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "protocol": {
            "workflow": "prior_frozen_skill_full_test_read_only_evaluation",
            "evaluation_scope": "held_out_test",
            "held_out": True,
            "generalization_claim": True,
            "accuracy_denominator": expected_tasks,
            **source,
        },
        "frozen": {
            "root": str(frozen.root),
            "digest": frozen.digest,
            "source_train_manifest_hash": frozen.source_train_manifest_hash,
            "source_validation_manifest_hash": frozen.source_validation_manifest_hash,
        },
        "effectiveness": summary,
        "test_cost": {
            "episodes": _episode_cost(episodes),
            "usage": usage.to_dict(),
            "api_cost": usage.api_cost,
            "api_cost_unpriced": usage.api_cost_unpriced,
            "provider_evidence": dict(worker.get("provider_evidence") or {}),
        },
        "comparison_metrics": {
            key: summary[key] for key in (
                "official_success", "official_rate", "micro_average_official_rate",
                "strict_rate", "macro_family_official_rate", "actions_per_task",
                "actions_per_solved", "p50_actions", "p90_actions", "calls_per_task",
                "tokens_per_task", "tokens_per_solved", "latency_per_task_ms",
                "p50_latency_ms", "p90_latency_ms", "target_reasoning_tokens",
                "reasoning_tokens_in_completion", "delta_vs_pure_dynamic",
                "positive_transfer", "negative_transfer", "paired_transfer_status",
                "family",
            )
        },
    }
    _write_json_atomic(ctx.output_dir / "test_report.json", report)
    assert_frozen_unchanged(frozen)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=model.api_key_env)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=_METHOD, choices=[_METHOD])
    parser.add_argument("--phase", required=True, choices=["smoke", "train", "test"])
    parser.add_argument(
        "--seed", type=int, choices=[42, 43, 44], default=None,
        help="formal run seed; overrides common.yaml before identity hashing",
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument(
        "--source-run", default=None,
        help="completed matching train run whose frozen artifact is used by --phase test",
    )
    parser.add_argument(
        "--resume-source-run", default=None,
        help="failed infrastructure train run used for epoch-boundary recovery",
    )
    parser.add_argument(
        "--campaign-lock", default=None,
        help="verified campaign_lock.json shared by all formal seed lanes",
    )
    parser.add_argument("--config", default="configs/baselines/b3_skillopt.yaml")
    parser.add_argument(
        "--output-dir", default=None,
        help="new output directory; an existing directory is always rejected",
    )
    args = parser.parse_args(argv)

    output_dir: Path | None = None
    output_owned = False
    model: ModelConfig | None = None
    run_id = _new_run_id(phase=args.phase)
    try:
        lock, driver = load_lock_and_driver(
            repo_root=REPO_ROOT, config_path=_path(args.config),
        )
        config = dict(driver.config)
        if args.seed is not None:
            config["run_seed"] = int(args.seed)
            driver.config["run_seed"] = int(args.seed)
            config["train"] = dict(config.get("train") or {})
            config["train"]["seed"] = int(args.seed)
            driver.config["train"] = dict(config["train"])
        profile = str(config.get("protocol_profile", "pilot_v1"))
        if profile not in {"pilot_v1", "formal_v2"}:
            raise ValueError(f"unsupported protocol_profile: {profile!r}")
        python_runtime: dict[str, Any] | None = None
        if profile == "formal_v2" and args.phase in {"train", "test"}:
            expected_python = resolve_formal_python(
                REPO_ROOT, str(config.get("worker_python", "")),
            )
            python_runtime = verify_runtime_python(
                expected_python=expected_python,
                require_venv=True,
            )
        if args.phase == "test":
            if not args.test_manifest or not args.source_run:
                raise ValueError("--phase test requires --test-manifest and --source-run")
            if args.resume_source_run:
                raise ValueError("--resume-source-run is accepted only by --phase train")
            config["protocol"] = {
                "workflow": "prior_frozen_skill_full_test_read_only_evaluation",
                "final_evaluation_scope": "held_out_test",
                "held_out": True,
                "generalization_claim": True,
                "test_manifest": str(_path(args.test_manifest)),
            }
        else:
            if args.source_run:
                raise ValueError("--source-run is accepted only by --phase test")
            if args.resume_source_run and args.phase != "train":
                raise ValueError("--resume-source-run is accepted only by --phase train")
            if args.phase == "smoke" and args.test_manifest:
                raise ValueError("smoke must not receive a Test manifest")
            if profile == "formal_v2" and args.phase == "train" and not args.test_manifest:
                raise ValueError("formal train requires --test-manifest for campaign identity")
            if profile == "pilot_v1" and args.test_manifest:
                raise ValueError("pilot train must not receive a Test manifest")
            config["protocol"] = {
                "workflow": (
                    "train120_validation24_freeze" if profile == "formal_v2"
                    else "train30_validation6_freeze_train30_read_only_replay"
                ),
                "final_evaluation_scope": (
                    "training_and_selection_only" if profile == "formal_v2"
                    else "train_resubstitution"
                ),
                "held_out": False,
                "generalization_claim": False,
                "test_manifest": (
                    str(_path(args.test_manifest)) if args.test_manifest else None
                ),
            }
        model = ModelConfig.from_mapping(config["model"])
        model.validate_formal_identity()
        model.require_api_key()

        alfworld_raw = os.environ.get("ALFWORLD_DATA", "").strip()
        if not alfworld_raw:
            raise RuntimeError("ALFWORLD_DATA is not set")
        alfworld_data = Path(alfworld_raw).expanduser().resolve(strict=True)
        if not alfworld_data.is_dir():
            raise NotADirectoryError(f"ALFWORLD_DATA is not a directory: {alfworld_data}")

        train_manifest = _load_manifest(args.train_manifest)
        validation_manifest = _load_manifest(args.validation_manifest)
        test_manifest = _load_manifest(args.test_manifest) if args.test_manifest else None
        verify_disjoint(*(
            [train_manifest, validation_manifest, test_manifest]
            if test_manifest is not None else [train_manifest, validation_manifest]
        ))
        train_preflight = verify_formal_manifest(
            train_manifest, alfworld_data=alfworld_data, role="train", profile=profile
        )
        validation_preflight = verify_formal_manifest(
            validation_manifest,
            alfworld_data=alfworld_data,
            role="validation",
            profile=profile,
        )
        test_preflight = (
            verify_formal_manifest(
                test_manifest, alfworld_data=alfworld_data, role="test", profile=profile
            )
            if test_manifest is not None else None
        )
        git_state = _controller_git_state()
        controller_code_digest = hash_code(REPO_ROOT)
        campaign: dict[str, Any] | None = None
        resume: dict[str, Any] | None = None
        if profile == "formal_v2" and args.phase in {"train", "test"}:
            if git_state.get("dirty"):
                raise RuntimeError(
                    "formal campaign requires clean tracked/untracked source under "
                    "src/, experiments/, and configs/"
                )
            if not args.campaign_lock:
                raise ValueError("formal train/test requires --campaign-lock")
            if test_manifest is None:
                raise AssertionError("formal campaign Test manifest unexpectedly missing")
            campaign = _load_campaign_descriptor(
                args.campaign_lock,
                config=config,
                source_lock=lock,
                model=model,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                test_manifest=test_manifest,
                git_state=git_state,
                code_digest=controller_code_digest,
                seed=int(config.get("run_seed", 42)),
            )
        elif args.campaign_lock:
            raise ValueError("--campaign-lock is reserved for formal train/test phases")
        if args.resume_source_run:
            if profile != "formal_v2" or args.phase != "train":
                raise ValueError(
                    "--resume-source-run requires a formal train phase"
                )
            source_resume = _path(args.resume_source_run)
            if not source_resume.is_dir():
                raise NotADirectoryError(
                    f"resume source run is not a directory: {source_resume}"
                )
            resume = {"source_run": str(source_resume)}
        max_actions = int(config.get("max_environment_actions", 100))
        env_max_steps = int(dict(config.get("env") or {}).get("max_steps", -1))
        if max_actions != 100 or env_max_steps != max_actions:
            raise ValueError(
                "formal action ceiling mismatch: common max_environment_actions "
                f"is {max_actions}, SkillOpt env.max_steps is {env_max_steps}"
            )

        data_signature = _alfworld_data_signature(alfworld_data)
        campaign_id = str(config.get("campaign_id", "pilot"))
        if args.output_dir:
            output_dir = _path(args.output_dir)
        else:
            bucket = args.phase if args.phase in {"smoke", "test"} else campaign_id
            output_dir = (
                REPO_ROOT / "runs" / "baselines" / bucket / _METHOD / run_id
            ).resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        output_owned = True

        ctx = _create_context(
            args=args,
            config=config,
            lock=lock,
            model=model,
            alfworld_data=alfworld_data,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            test_manifest=test_manifest,
            data_signature=data_signature,
            output_dir=output_dir,
            run_id=run_id,
            campaign=campaign,
            resume=resume,
        )
        _write_identity_artifacts(
            ctx=ctx,
            lock=lock,
            config=config,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            test_manifest=test_manifest,
            train_preflight=train_preflight,
            validation_preflight=validation_preflight,
            test_preflight=test_preflight,
            data_signature=data_signature,
            git_state=git_state,
            phase=args.phase,
            python_runtime=python_runtime,
        )
        driver.preflight(ctx)

        if args.phase == "smoke":
            report = _run_smoke(
                driver=driver, ctx=ctx, train_manifest=train_manifest, model=model,
            )
        elif args.phase == "train":
            report = _run_train(
                driver=driver,
                ctx=ctx,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                model=model,
                profile=profile,
            )
        else:
            if test_manifest is None:
                raise AssertionError("test manifest unexpectedly missing")
            report = _run_test(
                driver=driver,
                ctx=ctx,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                test_manifest=test_manifest,
                source_run=_path(args.source_run),
                model=model,
                profile=profile,
            )
        _write_json_atomic(ctx.output_dir / "completion.json", {
            "schema_version": 1,
            "passed": True,
            "run_id": ctx.run_id,
            "phase": args.phase,
            "identity": dict(ctx.identity),
            "report": ({
                "smoke": "smoke_report.json",
                "train": "report.json",
                "test": "test_report.json",
            }[args.phase]),
            "completed_at_unix": time.time(),
        })
        _write_json_atomic(ctx.output_dir / "run_state.json", {
            "schema_version": 1,
            "run_id": ctx.run_id,
            "state": "completed",
            "phase": args.phase,
            "updated_at_unix": time.time(),
        }, overwrite=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        error = _safe_error(
            exc,
            api_key_env=(model.api_key_env if model is not None else "MODEL_API_KEY"),
        )
        failure_kind = str(getattr(exc, "failure_kind", "protocol_failure"))
        if failure_kind not in {"infrastructure_failure", "protocol_failure"}:
            failure_kind = "protocol_failure"
        failure = {
            "schema_version": 1,
            "passed": False,
            "method": _METHOD,
            "phase": args.phase,
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": error,
            "failure_kind": failure_kind,
            "output_dir": str(output_dir) if output_dir is not None else None,
            "failed_at_unix": time.time(),
        }
        if output_owned and output_dir is not None and output_dir.is_dir():
            try:
                _write_json_atomic(output_dir / "failure.json", failure)
                _write_json_atomic(output_dir / "run_state.json", {
                    "schema_version": 1,
                    "run_id": run_id,
                    "state": "failed",
                    "phase": args.phase,
                    "error_type": type(exc).__name__,
                    "error": error,
                    "failure_kind": failure_kind,
                    "updated_at_unix": time.time(),
                }, overwrite=True)
            except Exception:
                pass
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
