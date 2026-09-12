"""Run B5 GEPA smoke, formal Train/Val optimization, or frozen Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.baselines.common.driver import RunContext
from experiments.baselines.common.formal_validation import (
    ALFWORLD_FORMAL_TASK_TYPES,
    verify_final_evaluation_bijection,
    verify_formal_manifest,
)
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.integrity import assert_no_secrets_on_disk, validate_episode_usage
from experiments.baselines.common.manifest import (
    TaskManifestSet,
    sha256_json,
    verify_disjoint,
)
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.post_evaluator import (
    TaskRow,
    summarize_rows,
    write_evaluated_episodes_jsonl,
    write_rows_jsonl,
)
from experiments.baselines.common.runtime_python import resolve_formal_python, verify_runtime_python
from experiments.baselines.common.source_identity import hash_code
from experiments.baselines.common.usage import UsageSnapshot
from experiments.baselines.run_method import (
    _alfworld_data_signature,
    _controller_git_state,
    _episode_cost,
    _formal_config_digest,
    _safe_error,
    _sha256_file,
    _validate_campaign_probe_receipt,
    _write_json_atomic,
)

from .driver import METHOD_ID, load_lock_and_driver


REPO_ROOT = Path(__file__).resolve().parents[3]
_INITIAL_SKILL_REL = "skillopt/envs/alfworld/skills/initial.md"


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _new_run_id(phase: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"b5_{phase}_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _load_campaign_descriptor(
    path: str | Path,
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    model: ModelConfig,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    test: TaskManifestSet,
    seed: int,
    git_state: dict[str, Any],
    code_digest: str,
) -> dict[str, Any]:
    lock_path = _path(path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"GEPA campaign lock is unreadable: {lock_path}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("GEPA campaign lock has an invalid schema")
    parallel = dict(config.get("parallel") or {})
    expected_external_commit = str(lock["gepa"]["commit"])
    observed_external_commit = str(
        payload.get("external_gepa_commit")
        or payload.get("external_method_commit")
        or ""
    )
    expected_runtime = str(lock["gepa"]["runtime_tree"]["sha256"])
    observed_runtime = str(
        payload.get("external_runtime_tree_digest")
        or payload.get("external_gepa_runtime_tree_digest")
        or ""
    )
    checks = {
        "method": payload.get("method") == METHOD_ID,
        "seed": seed in [int(value) for value in payload.get("seeds", [])],
        "train_manifest": payload.get("train_manifest_digest") == train.digest,
        "validation_manifest": payload.get("validation_manifest_digest") == validation.digest,
        "test_manifest": payload.get("test_manifest_digest") == test.digest,
        "controller_commit": payload.get("controller_commit") == git_state.get("commit"),
        "controller_code": payload.get("controller_code_digest") == code_digest,
        "external_commit": observed_external_commit == expected_external_commit,
        "external_runtime": observed_runtime == expected_runtime,
        "skillopt_commit": payload.get("external_skillopt_commit")
        == str(lock["skillopt"]["commit"]),
        "skillopt_runtime": payload.get("skillopt_runtime_tree_digest")
        == str(lock["skillopt"]["runtime_tree"]["sha256"]),
        "initial_skill": payload.get("initial_skill_sha256")
        == str(
            lock["skillopt"]["key_files"][_INITIAL_SKILL_REL]
        ),
        "model": payload.get("model") == model.model,
        "reasoning_effort": payload.get("reasoning_effort") == model.reasoning_effort,
        "formal_config": payload.get("formal_config_digest") == _formal_config_digest(config),
        "provider_cap": int(payload.get("campaign_provider_max_inflight", 0))
        == int(parallel.get("campaign_provider_max_inflight", 0)),
        "seed_lanes": int(payload.get("seed_lanes", 0))
        == int(parallel.get("seed_lanes", 0)) == 1,
        "mp_start_method": dict(payload.get("parallel") or {}).get(
            "mp_start_method"
        )
        == parallel.get("mp_start_method")
        == "spawn",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("GEPA campaign lock mismatch: " + ", ".join(failed))
    _validate_campaign_probe_receipt(
        payload, campaign_root=lock_path.parent, model=model
    )
    gate_dir = Path(str(payload.get("provider_gate_dir", ""))).expanduser().resolve()
    try:
        gate_dir.relative_to(lock_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("GEPA provider gate must be inside the campaign root") from exc
    return {
        "campaign_id": str(payload["campaign_id"]),
        "campaign_lock_path": str(lock_path),
        "campaign_lock_digest": _sha256_file(lock_path),
        "provider_gate_dir": str(gate_dir),
        "campaign_provider_max_inflight": int(payload["campaign_provider_max_inflight"]),
    }


def _create_context(
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    model: ModelConfig,
    output_dir: Path,
    run_id: str,
    train_path: Path,
    validation_path: Path,
    test_path: Path | None,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    test: TaskManifestSet | None,
    data_signature: dict[str, Any],
    campaign: dict[str, Any] | None,
    resume: dict[str, Any] | None = None,
) -> RunContext:
    config_digest = sha256_json(config)
    code_digest = hash_code(REPO_ROOT)
    identity = {
        "config_digest": config_digest,
        "train_manifest_digest": train.digest,
        "validation_manifest_digest": validation.digest,
        "controller_code_digest": code_digest,
        "external_source_digest": str(lock["gepa"]["runtime_tree"]["sha256"]),
        "skillopt_source_digest": str(lock["skillopt"]["runtime_tree"]["sha256"]),
        "initial_skill_digest": str(
            lock["skillopt"]["key_files"][_INITIAL_SKILL_REL]
        ),
        "model_identity_digest": sha256_json(model.to_wire()),
        "alfworld_data_digest": sha256_json(data_signature),
        "formal_config_digest": _formal_config_digest(config),
    }
    if test is not None:
        identity["test_manifest_digest"] = test.digest
    config_path = output_dir / "config_resolved.json"
    _write_json_atomic(config_path, config)
    return RunContext(
        campaign_id=str(config.get("campaign_id", "gepa")),
        method_id=METHOD_ID,
        run_seed=int(config["run_seed"]),
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        external_repo=REPO_ROOT / ".external" / "gepa",
        external_commit=str(lock["gepa"]["commit"]),
        model_config=model,
        max_environment_actions=int(config["max_environment_actions"]),
        alfworld_data=Path(str(data_signature["resolved_data_root"])),
        config_hash=config_digest,
        code_hash=code_digest,
        run_id=run_id,
        resolved_config_path=config_path.resolve(),
        identity=identity,
        train_manifest_path=train_path,
        validation_manifest_path=validation_path,
        test_manifest_path=test_path,
        campaign=dict(campaign) if campaign is not None else None,
        resume=dict(resume) if resume is not None else None,
    )


def _write_run_identity(
    *,
    ctx: RunContext,
    lock: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    test: TaskManifestSet | None,
    data_signature: dict[str, Any],
    git_state: dict[str, Any],
    receipts: dict[str, Any],
    python_runtime: dict[str, Any] | None,
) -> None:
    _write_json_atomic(ctx.output_dir / "source_lock.json", lock)
    _write_json_atomic(ctx.output_dir / "run_manifest.json", {
        "schema_version": 1,
        "method": METHOD_ID,
        "phase": phase,
        "run_id": ctx.run_id,
        "run_seed": ctx.run_seed,
        "formal": str(config["experiment_kind"]) == "formal",
        "protocol": (
            "protocol-faithful-matched-train-v2"
            if str(config["experiment_kind"]) == "formal"
            else "smoke-v1"
        ),
        "protocol_profile": str(config["protocol_profile"]),
        "workflow": {
            "smoke": "train6_val6_gepa_budget24_freeze_frozen_test6",
            "train": "train120_validation24_gepa_budget720_freeze",
            "test": "prior_frozen_best_skill_test134_read_only",
        }[phase],
        "external_repo": str(lock["gepa"]["repo"]),
        "external_tag": str(lock["gepa"].get("tag") or ""),
        "external_commit": str(lock["gepa"]["commit"]),
        "external_runtime_tree": dict(lock["gepa"]["runtime_tree"]),
        "shared_executor_repo": str(lock["skillopt"]["repo"]),
        "shared_executor_commit": str(lock["skillopt"]["commit"]),
        "initial_skill_sha256": str(ctx.identity["initial_skill_digest"]),
        "method_specific_prior": False,
        "controller_commit": str(git_state.get("commit", "")),
        "controller_git": git_state,
        "controller_code_digest": ctx.code_hash,
        "python_runtime": python_runtime,
        "train_manifest_path": str(ctx.train_manifest_path),
        "train_manifest_hash": train.digest,
        "validation_manifest_path": str(ctx.validation_manifest_path),
        "validation_manifest_hash": validation.digest,
        "test_manifest_path": (
            str(ctx.test_manifest_path) if ctx.test_manifest_path is not None else None
        ),
        "test_manifest_hash": test.digest if test is not None else None,
        "alfworld_package": {
            "distribution": "alfworld",
            "version": (
                python_runtime.get("alfworld_distribution_version")
                if python_runtime is not None else None
            ),
            "expected_version": "0.4.2",
        },
        "alfworld_package_version": (
            python_runtime.get("alfworld_distribution_version")
            if python_runtime is not None else None
        ),
        "alfworld_data_signature": dict(data_signature),
        "identity": dict(ctx.identity),
        "campaign": dict(ctx.campaign) if ctx.campaign is not None else None,
        "resume": dict(ctx.resume) if ctx.resume is not None else None,
        "model": ctx.model_config.model,
        "model_identity": ctx.model_config.to_wire(),
        "reasoning_effort": ctx.model_config.reasoning_effort,
        "max_environment_actions": ctx.max_environment_actions,
        "episode_workers": int(
            dict(config.get("parallel") or {}).get(
                "episode_workers_per_seed",
                dict(config.get("env") or {}).get("workers", 0),
            )
        ),
        "provider_max_inflight": int(
            dict(ctx.campaign or {}).get(
                "campaign_provider_max_inflight",
                dict(config.get("parallel") or {}).get(
                    "campaign_provider_max_inflight",
                    dict(config.get("env") or {}).get("max_api_workers", 0),
                ),
            )
        ),
        "optimizer": dict(config["gepa"]),
        "parallel": dict(config.get("parallel") or {}),
        "started_at_unix": time.time(),
    })
    _write_json_atomic(ctx.output_dir / "task_manifest.json", {
        "schema_version": 1,
        "train": {
            "path": str(ctx.train_manifest_path), "digest": train.digest,
            "tasks": len(train.tasks), "preflight": receipts["train"],
        },
        "validation": {
            "path": str(ctx.validation_manifest_path), "digest": validation.digest,
            "tasks": len(validation.tasks), "preflight": receipts["validation"],
        },
        "test": ({
            "path": str(ctx.test_manifest_path), "digest": test.digest,
            "tasks": len(test.tasks), "preflight": receipts["test"],
        } if test is not None else None),
    })
    _write_json_atomic(ctx.output_dir / "run_state.json", {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "state": "running",
        "phase": phase,
        "updated_at_unix": time.time(),
    })


def _bind_run_manifest_to_frozen(
    ctx: RunContext,
    frozen: FrozenArtifact,
) -> dict[str, Any]:
    """Atomically bind a completed freeze to the run's immutable authority."""

    manifest_path = ctx.output_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"GEPA run manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("GEPA run manifest must be a mapping")
    if (
        manifest.get("method") != METHOD_ID
        or manifest.get("run_id") != ctx.run_id
        or int(manifest.get("run_seed", -1)) != ctx.run_seed
    ):
        raise ValueError("GEPA frozen artifact cannot bind to a different run")
    if frozen.method_id != METHOD_ID:
        raise ValueError("GEPA run cannot bind another method's frozen artifact")
    if frozen.source_train_manifest_hash != manifest.get("train_manifest_hash"):
        raise ValueError("GEPA frozen artifact has the wrong Train manifest digest")
    if (
        frozen.source_validation_manifest_hash
        != manifest.get("validation_manifest_hash")
    ):
        raise ValueError("GEPA frozen artifact has the wrong Validation manifest digest")
    assert_frozen_unchanged(frozen)

    descriptor = {
        "root": str(frozen.root.resolve()),
        "digest": frozen.digest,
        "source_train_manifest_hash": frozen.source_train_manifest_hash,
        "source_validation_manifest_hash": frozen.source_validation_manifest_hash,
    }
    existing_digest = manifest.get("frozen_artifact_digest")
    existing_descriptor = manifest.get("frozen_artifact")
    if existing_digest is not None and existing_digest != frozen.digest:
        raise ValueError("GEPA run manifest is already bound to another frozen digest")
    if existing_descriptor is not None and existing_descriptor != descriptor:
        raise ValueError("GEPA run manifest frozen descriptor cannot be rebound")
    manifest["frozen_artifact_digest"] = frozen.digest
    manifest["frozen_artifact"] = descriptor
    _write_json_atomic(manifest_path, manifest, overwrite=True)
    return descriptor


def _validate_optimizer_episode(
    episode: Any,
    *,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    seed: int,
) -> None:
    manifest = train if episode.phase == "train" else validation
    tasks = {task.task_id: task for task in manifest.tasks}
    task = tasks.get(episode.task_id)
    if task is None:
        raise ValueError(f"GEPA optimizer episode is outside {episode.phase} manifest")
    if (
        episode.method != METHOD_ID
        or episode.run_seed != seed
        or episode.manifest_index != task.index
        or episode.gamefile != task.gamefile_rel
        or episode.gamefile_hash != task.gamefile_sha256
        or episode.infrastructure_failure
    ):
        raise ValueError(f"GEPA optimizer episode identity failed: {episode.task_id}")


def _run_smoke(
    driver: Any,
    ctx: RunContext,
    train: TaskManifestSet,
    test: TaskManifestSet,
) -> dict[str, Any]:
    result = driver.smoke(ctx, train)
    optimizer_episodes = [item for item in result.episodes if item.phase == "smoke"]
    heldout_episodes = [item for item in result.episodes if item.phase == "smoke_test"]
    if len(heldout_episodes) != len(test.tasks):
        raise RuntimeError("GEPA smoke did not complete frozen Test6")
    write_evaluated_episodes_jsonl(
        heldout_episodes,
        ctx.output_dir / "smoke_test" / "evaluated_common_episodes.jsonl",
    )
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=ctx.model_config.api_key_env)
    worker = json.loads(
        (ctx.output_dir / "smoke" / "worker_result.json").read_text(encoding="utf-8")
    )
    heldout_worker = json.loads(
        (ctx.output_dir / "smoke_test" / "worker_result.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = FrozenArtifact.load(ctx.output_dir / "smoke_frozen")
    _bind_run_manifest_to_frozen(ctx, frozen)
    report = {
        "schema_version": 1,
        "passed": True,
        "method": METHOD_ID,
        "phase": "smoke",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "optimizer_episodes": len(optimizer_episodes),
        "heldout_test_episodes": len(heldout_episodes),
        "official_test_successes": sum(
            int(item.official_success) for item in heldout_episodes
        ),
        "task_success_required": False,
        "frozen": {
            "root": str(frozen.root),
            "digest": frozen.digest,
            "digest_before_test": heldout_worker["frozen_digest_before"],
            "digest_after_test": heldout_worker["frozen_digest_after"],
        },
        "usage": result.usage.to_dict(),
        "gepa_metrics": dict(worker.get("train_summary") or {}),
        "provider_evidence": {
            "optimizer": dict(worker.get("provider_evidence") or {}),
            "heldout_test": dict(heldout_worker.get("provider_evidence") or {}),
        },
    }
    _write_json_atomic(ctx.output_dir / "smoke_report.json", report)
    return report


def _run_train(
    driver: Any,
    ctx: RunContext,
    train: TaskManifestSet,
    validation: TaskManifestSet,
) -> dict[str, Any]:
    result = driver.train(ctx, train, validation)
    all_episodes = [*result.episodes, *result.validation_episodes]
    for episode in all_episodes:
        _validate_optimizer_episode(
            episode, train=train, validation=validation, seed=ctx.run_seed
        )
    validate_episode_usage(all_episodes)
    if result.usage.target.calls <= 0 or result.usage.evolution.calls <= 0:
        raise RuntimeError("GEPA formal train lacks target/reflection provider usage")
    val_ids = {item.task_id for item in result.validation_episodes}
    if val_ids != {task.task_id for task in validation.tasks}:
        raise RuntimeError("GEPA did not evaluate the complete Validation24")
    frozen = driver.freeze(ctx, result)
    _bind_run_manifest_to_frozen(ctx, frozen)
    worker = json.loads(
        (ctx.output_dir / "train" / "worker_result.json").read_text(
            encoding="utf-8"
        )
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "method": METHOD_ID,
        "phase": "train",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "protocol": {
            "workflow": "train120_validation24_gepa_budget720_freeze",
            "evaluation_scope": "training_and_validation_only",
            "held_out": False,
            "generalization_claim": False,
        },
        "frozen": {
            "root": str(frozen.root),
            "digest": frozen.digest,
            "source_train_manifest_hash": frozen.source_train_manifest_hash,
            "source_validation_manifest_hash": frozen.source_validation_manifest_hash,
        },
        "training_cost": {
            "train_pool_size": len(train.tasks),
            "validation_pool_size": len(validation.tasks),
            "unique_train_tasks_evaluated": len({item.task_id for item in result.episodes}),
            "unique_validation_tasks_evaluated": len(val_ids),
            "train_episodes": _episode_cost(result.episodes),
            "validation_episodes": _episode_cost(result.validation_episodes),
            "usage": result.usage.to_dict(),
            "api_cost": result.usage.api_cost,
            "api_cost_unpriced": result.usage.api_cost_unpriced,
            "gepa_metrics": result.method_metrics,
            "provider_evidence": dict(worker.get("provider_evidence") or {}),
        },
    }
    _write_json_atomic(ctx.output_dir / "report.json", report)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=ctx.model_config.api_key_env)
    return report


def _load_frozen_source(
    source_run: Path,
    *,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    seed: int,
    expected_ctx: RunContext,
) -> tuple[FrozenArtifact, Path]:
    if not source_run.is_dir():
        raise NotADirectoryError(f"GEPA source run is missing: {source_run}")
    completion = json.loads((source_run / "completion.json").read_text(encoding="utf-8"))
    report = json.loads((source_run / "report.json").read_text(encoding="utf-8"))
    if completion.get("passed") is not True or completion.get("phase") != "train":
        raise ValueError("GEPA source run is not a completed train run")
    run_manifest = json.loads(
        (source_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    if report.get("method") != METHOD_ID or int(
        run_manifest["run_seed"]
    ) != seed:
        raise ValueError("GEPA source run method/seed mismatch")
    source_identity = dict(run_manifest.get("identity") or {})
    if (
        source_identity != dict(expected_ctx.identity)
        or dict(completion.get("identity") or {}) != source_identity
        or run_manifest.get("external_commit") != expected_ctx.external_commit
        or run_manifest.get("model_identity") != expected_ctx.model_config.to_wire()
        or dict(run_manifest.get("campaign") or {})
        != dict(expected_ctx.campaign or {})
    ):
        raise ValueError("GEPA source run identity differs from frozen Test authority")
    frozen_dir = source_run / "frozen"
    frozen = FrozenArtifact.load(frozen_dir)
    if frozen.source_train_manifest_hash != train.digest:
        raise ValueError("GEPA source frozen artifact has wrong Train digest")
    if frozen.source_validation_manifest_hash != validation.digest:
        raise ValueError("GEPA source frozen artifact has wrong Validation digest")
    if dict(report.get("frozen") or {}).get("digest") != frozen.digest:
        raise ValueError("GEPA source report names a different frozen artifact")
    frozen_descriptor = dict(run_manifest.get("frozen_artifact") or {})
    if (
        run_manifest.get("frozen_artifact_digest") != frozen.digest
        or frozen_descriptor.get("digest") != frozen.digest
        or frozen_descriptor.get("source_train_manifest_hash") != train.digest
        or frozen_descriptor.get("source_validation_manifest_hash")
        != validation.digest
    ):
        raise ValueError("GEPA source run manifest is not bound to its frozen artifact")
    assert_frozen_unchanged(frozen)
    return frozen, frozen_dir


def _run_test(
    driver: Any,
    ctx: RunContext,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    test: TaskManifestSet,
    source_run: Path,
) -> dict[str, Any]:
    frozen, frozen_dir = _load_frozen_source(
        source_run,
        train=train,
        validation=validation,
        seed=ctx.run_seed,
        expected_ctx=ctx,
    )
    _bind_run_manifest_to_frozen(ctx, frozen)
    episodes = driver.evaluate_test(
        ctx,
        frozen,
        train,
        validation,
        test,
        frozen_dir=frozen_dir,
    )
    verify_final_evaluation_bijection(
        episodes,
        test,
        role="test",
        expected_phase="test",
        expected_method=METHOD_ID,
        expected_run_seed=ctx.run_seed,
        expected_artifact_digest=frozen.digest,
        require_strict_outcomes=True,
        profile="formal_v2",
    )
    validate_episode_usage(episodes)
    rows = [TaskRow.from_episode(item) for item in episodes]
    write_evaluated_episodes_jsonl(
        episodes, ctx.output_dir / "test" / "evaluated_common_episodes.jsonl"
    )
    write_rows_jsonl(rows, ctx.output_dir / "test" / "task_rows.jsonl")
    usage = UsageSnapshot.load(ctx.output_dir / "test" / "usage.json")
    summary = summarize_rows(
        rows,
        task_types=list(ALFWORLD_FORMAL_TASK_TYPES),
        api_cost=usage.api_cost,
        api_cost_unpriced=usage.api_cost_unpriced,
    )
    if usage.target.calls <= 0 or usage.evolution.calls != 0:
        raise RuntimeError("GEPA frozen Test has invalid provider role usage")
    worker = json.loads((ctx.output_dir / "test" / "worker_result.json").read_text(encoding="utf-8"))
    if worker.get("optimizer_constructed") is not False:
        raise RuntimeError("GEPA Test constructed forbidden optimizer state")
    report = {
        "schema_version": 1,
        "passed": True,
        "method": METHOD_ID,
        "phase": "test",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "source_run": str(source_run),
        "protocol": {
            "source_run": str(source_run.resolve()),
            "workflow": "prior_frozen_best_skill_test134_read_only",
            "evaluation_scope": "held_out_test",
            "held_out": True,
            "generalization_claim": True,
        },
        "frozen": {
            "root": str(frozen.root), "digest": frozen.digest,
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
    }
    _write_json_atomic(ctx.output_dir / "test_report.json", report)
    assert_frozen_unchanged(frozen)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=ctx.model_config.api_key_env)
    return report


def _prepare_resume(
    *,
    source_run: Path,
    destination_run: Path,
    expected_ctx: RunContext,
) -> dict[str, Any]:
    """Import only GEPA's durable run_dir from a proven infrastructure failure."""

    source_run = source_run.resolve(strict=True)
    if not source_run.is_dir():
        raise NotADirectoryError(f"GEPA resume source is not a run directory: {source_run}")
    try:
        manifest = json.loads(
            (source_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        failure = json.loads(
            (source_run / "failure.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("GEPA resume source lacks readable run/failure evidence") from exc
    if (
        manifest.get("method") != METHOD_ID
        or manifest.get("phase") != "train"
        or int(manifest.get("run_seed", -1)) != expected_ctx.run_seed
        or failure.get("passed") is not False
        or failure.get("failure_kind") != "infrastructure_failure"
    ):
        raise ValueError(
            "GEPA resume source must be a matching failed Train infrastructure attempt"
        )
    if (source_run / "completion.json").exists():
        raise ValueError("GEPA resume source is already completed")
    source_identity = dict(manifest.get("identity") or {})
    if source_identity != dict(expected_ctx.identity):
        mismatches = sorted(
            key
            for key in set(source_identity) | set(expected_ctx.identity)
            if source_identity.get(key) != expected_ctx.identity.get(key)
        )
        raise ValueError("GEPA resume identity mismatch: " + ", ".join(mismatches))
    source_git = dict(manifest.get("controller_git") or {})
    current_git = _controller_git_state()
    if source_git.get("commit") != current_git.get("commit"):
        raise ValueError("GEPA resume controller commit mismatch")
    if manifest.get("external_commit") != expected_ctx.external_commit:
        raise ValueError("GEPA resume external commit mismatch")
    if manifest.get("model_identity") != expected_ctx.model_config.to_wire():
        raise ValueError("GEPA resume model identity mismatch")
    if dict(manifest.get("campaign") or {}) != dict(expected_ctx.campaign or {}):
        raise ValueError("GEPA resume campaign/parallel authority mismatch")

    source_state = source_run / "train" / "gepa_state"
    if not (source_state / "gepa_state.bin").is_file():
        raise FileNotFoundError("GEPA resume source has no durable gepa_state.bin")
    if (source_state / "gepa.stop").exists():
        raise ValueError("GEPA resume source contains an active gepa.stop marker")
    if any(path.is_symlink() for path in source_state.rglob("*")):
        raise ValueError("GEPA resume state must not contain symbolic links")
    state_digest = digest_directory(source_state)
    destination_state = destination_run / "train" / "gepa_state"
    if destination_state.exists():
        raise FileExistsError(destination_state)
    destination_state.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_state, destination_state)
    if digest_directory(destination_state) != state_digest:
        raise RuntimeError("copied GEPA resume state digest mismatch")
    return {
        "schema_version": 1,
        "authority": "gepa_run_dir",
        "source_run": str(source_run),
        "source_run_id": str(manifest.get("run_id", "")),
        "source_failure_kind": "infrastructure_failure",
        "source_state_digest": state_digest,
        "destination_state": str(destination_state),
        "resume_replay_policy": "count_as_provider_cost_not_algorithm_metric_calls",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=METHOD_ID, choices=[METHOD_ID])
    parser.add_argument("--phase", required=True, choices=["smoke", "train", "test"])
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--source-run", default=None)
    parser.add_argument("--resume-source-run", default=None)
    parser.add_argument("--campaign-lock", default=None)
    parser.add_argument("--config", default="configs/baselines/b5_gepa.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    run_id = _new_run_id(args.phase)
    output_dir: Path | None = None
    output_owned = False
    model: ModelConfig | None = None
    try:
        lock, driver = load_lock_and_driver(
            repo_root=REPO_ROOT, config_path=_path(args.config)
        )
        config = dict(driver.config)
        if args.seed is not None:
            config["run_seed"] = int(args.seed)
            config["train"] = dict(config.get("train") or {})
            config["train"]["seed"] = int(args.seed)
            driver.config = dict(config)
        else:
            config["run_seed"] = int(dict(config.get("train") or {}).get("seed", 42))
            driver.config = dict(config)
        kind = str(config.get("experiment_kind"))
        if args.phase == "smoke" and kind != "smoke":
            raise ValueError("GEPA smoke requires configs/baselines/b5_gepa_smoke.yaml")
        if args.phase != "smoke" and kind != "formal":
            raise ValueError("GEPA train/test require the formal config")
        if args.phase == "test":
            if not args.test_manifest or not args.source_run:
                raise ValueError("GEPA test requires --test-manifest and --source-run")
        elif args.source_run:
            raise ValueError("--source-run is accepted only for GEPA test")
        if args.phase == "smoke" and not args.test_manifest:
            raise ValueError("GEPA smoke requires the explicit Test6 manifest")
        if args.phase == "train" and not args.test_manifest:
            raise ValueError("formal GEPA train requires Test manifest for campaign identity")
        if args.resume_source_run and args.phase != "train":
            raise ValueError("GEPA resume is accepted only for the train phase")

        model = ModelConfig.from_mapping(config["model"])
        model.validate_formal_identity()
        model.require_api_key()
        data_raw = os.environ.get("ALFWORLD_DATA", "").strip()
        if not data_raw:
            raise RuntimeError("ALFWORLD_DATA is not set")
        alfworld_data = Path(data_raw).expanduser().resolve(strict=True)
        train_path = _path(args.train_manifest)
        validation_path = _path(args.validation_manifest)
        test_path = _path(args.test_manifest) if args.test_manifest else None
        train = TaskManifestSet.load(train_path)
        validation = TaskManifestSet.load(validation_path)
        test = TaskManifestSet.load(test_path) if test_path else None
        verify_disjoint(*([train, validation, test] if test else [train, validation]))
        profile = "smoke_v1" if args.phase == "smoke" else "formal_v2"
        receipts = {
            "train": verify_formal_manifest(
                train, alfworld_data=alfworld_data, role="train", profile=profile
            ),
            "validation": verify_formal_manifest(
                validation,
                alfworld_data=alfworld_data,
                role="validation",
                profile=profile,
            ),
        }
        if test is not None:
            receipts["test"] = verify_formal_manifest(
                test, alfworld_data=alfworld_data, role="test", profile=profile
            )
        git_state = _controller_git_state()
        code_digest = hash_code(REPO_ROOT)
        python_runtime = None
        campaign = None
        if args.phase in {"train", "test"}:
            if git_state["dirty"]:
                raise RuntimeError("formal GEPA requires a clean experiments/configs source tree")
            expected_python = resolve_formal_python(
                REPO_ROOT, str(config["worker_python"])
            )
            python_runtime = verify_runtime_python(
                expected_python=expected_python,
                require_venv=True,
                method=METHOD_ID,
                expected_distributions={
                    "alfworld": "0.4.2",
                    "gepa": str(lock["gepa"]["version"]),
                },
                expected_python_major_minor="3.12",
            )
            if not args.campaign_lock:
                raise ValueError("formal GEPA train/test requires --campaign-lock")
            if test is None:
                raise AssertionError("formal GEPA campaign Test manifest is missing")
            campaign = _load_campaign_descriptor(
                args.campaign_lock,
                config=config,
                lock=lock,
                model=model,
                train=train,
                validation=validation,
                test=test,
                seed=int(config["run_seed"]),
                git_state=git_state,
                code_digest=code_digest,
            )
        elif args.campaign_lock:
            raise ValueError("GEPA smoke does not accept a campaign lock")
        data_signature = _alfworld_data_signature(alfworld_data)
        if args.output_dir:
            output_dir = _path(args.output_dir)
        else:
            bucket = "smoke" if args.phase == "smoke" else (
                "test" if args.phase == "test" else str(config["campaign_id"])
            )
            output_dir = REPO_ROOT / "runs" / "baselines" / bucket / METHOD_ID / run_id
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        output_owned = True
        ctx = _create_context(
            config=config,
            lock=lock,
            model=model,
            output_dir=output_dir,
            run_id=run_id,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            train=train,
            validation=validation,
            test=test,
            data_signature=data_signature,
            campaign=campaign,
        )
        if args.resume_source_run:
            ctx = replace(
                ctx,
                resume=_prepare_resume(
                    source_run=_path(args.resume_source_run),
                    destination_run=output_dir,
                    expected_ctx=ctx,
                ),
            )
        _write_run_identity(
            ctx=ctx,
            lock=lock,
            config=config,
            phase=args.phase,
            train=train,
            validation=validation,
            test=test,
            data_signature=data_signature,
            git_state=git_state,
            receipts=receipts,
            python_runtime=python_runtime,
        )
        driver.preflight(ctx)
        if args.phase == "smoke":
            if test is None:
                raise AssertionError("GEPA smoke Test manifest is missing")
            report = _run_smoke(driver, ctx, train, test)
        elif args.phase == "train":
            report = _run_train(driver, ctx, train, validation)
        else:
            if test is None:
                raise AssertionError("GEPA Test manifest is missing")
            report = _run_test(
                driver, ctx, train, validation, test, _path(args.source_run)
            )
        report_name = {
            "smoke": "smoke_report.json", "train": "report.json", "test": "test_report.json"
        }[args.phase]
        _write_json_atomic(ctx.output_dir / "completion.json", {
            "schema_version": 1,
            "passed": True,
            "run_id": run_id,
            "phase": args.phase,
            "identity": dict(ctx.identity),
            "report": report_name,
            "completed_at_unix": time.time(),
        })
        _write_json_atomic(ctx.output_dir / "run_state.json", {
            "schema_version": 1,
            "run_id": run_id,
            "state": "completed",
            "phase": args.phase,
            "updated_at_unix": time.time(),
        }, overwrite=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        error = _safe_error(
            exc, api_key_env=model.api_key_env if model else "MODEL_API_KEY"
        )
        failure_kind = str(getattr(exc, "failure_kind", "protocol_failure"))
        if failure_kind not in {"infrastructure_failure", "protocol_failure"}:
            failure_kind = "protocol_failure"
        failure = {
            "schema_version": 1,
            "passed": False,
            "method": METHOD_ID,
            "phase": args.phase,
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": error,
            "failure_kind": failure_kind,
            "output_dir": str(output_dir) if output_dir else None,
            "failed_at_unix": time.time(),
        }
        if output_owned and output_dir is not None:
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
