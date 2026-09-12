"""Run B4 SkillGen-S smoke, formal extraction, or frozen held-out Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
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
from experiments.baselines.common.integrity import (
    assert_no_secrets_on_disk,
    validate_episode_usage,
)
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
from experiments.baselines.common.runtime_python import (
    resolve_formal_python,
    verify_runtime_python_executable,
)
from experiments.baselines.common.source_identity import hash_code
from experiments.baselines.bootstrap_external import worker_expected_distributions
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

from .corpus_builder import build_checkpoint_inventory
from .driver import SkillGenBaselineDriver, load_lock_and_driver
from .progress_adapter import analyze_label_coverage


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "b4_skillgen_s"


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _new_run_id(phase: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"b4_{phase}_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _load_campaign_descriptor(
    path: str | Path,
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    model: ModelConfig,
    train: TaskManifestSet,
    test: TaskManifestSet,
    supervision_digest: str | None,
    seed: int,
    git_state: dict[str, Any],
    code_digest: str,
) -> dict[str, Any]:
    lock_path = _path(path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"B4 campaign lock is unreadable: {lock_path}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("B4 campaign lock has an invalid schema")
    if payload.get("validation_manifest_digest") not in {None, ""}:
        raise ValueError("B4 campaign lock illegally binds Validation data")
    parallel = dict(config.get("parallel") or {})
    checks = {
        "method": payload.get("method") == METHOD_ID,
        "seed": seed in [int(value) for value in payload.get("seeds", [])],
        "train_manifest": payload.get("train_manifest_digest") == train.digest,
        "test_manifest": payload.get("test_manifest_digest") == test.digest,
        "controller_commit": payload.get("controller_commit") == git_state.get("commit"),
        "controller_code": payload.get("controller_code_digest") == code_digest,
        "external_commit": payload.get("external_method_commit")
        == str(lock["skillgen"]["commit"]),
        "external_runtime": payload.get("external_runtime_tree_digest")
        == str(lock["skillgen"]["runtime_tree"]["sha256"]),
        "supervision": (
            isinstance(payload.get("supervision_digest"), str)
            and len(str(payload.get("supervision_digest"))) == 64
            and (
                supervision_digest is None
                or payload.get("supervision_digest") == supervision_digest
            )
        ),
        "model": payload.get("model") == model.model,
        "reasoning_effort": payload.get("reasoning_effort") == model.reasoning_effort,
        "formal_config": payload.get("formal_config_digest")
        == _formal_config_digest(config),
        "seed_lanes": int(payload.get("seed_lanes", 0))
        == int(parallel.get("seed_lanes", 0)) == 1,
        "provider_cap": int(payload.get("campaign_provider_max_inflight", 0))
        == int(parallel.get("campaign_provider_max_inflight", 0)),
        "mp_start_method": dict(payload.get("parallel") or {}).get(
            "mp_start_method"
        )
        == parallel.get("mp_start_method")
        == "spawn",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("B4 campaign lock mismatch: " + ", ".join(failed))
    _validate_campaign_probe_receipt(payload, campaign_root=lock_path.parent, model=model)
    gate_dir = Path(str(payload.get("provider_gate_dir", ""))).expanduser().resolve()
    try:
        gate_dir.relative_to(lock_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("B4 provider gate must be inside the campaign root") from exc
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
    phase: str,
    train_path: Path,
    test_path: Path,
    train: TaskManifestSet,
    test: TaskManifestSet,
    supervision_path: Path,
    data_signature: dict[str, Any],
    campaign: dict[str, Any] | None,
    resume: dict[str, Any] | None = None,
) -> RunContext:
    config_digest = sha256_json(config)
    code_digest = hash_code(REPO_ROOT)
    identity = {
        "config_digest": config_digest,
        "train_manifest_digest": train.digest,
        "test_manifest_digest": test.digest,
        "external_source_digest": str(lock["skillgen"]["runtime_tree"]["sha256"]),
        "controller_code_digest": code_digest,
        "model_identity_digest": sha256_json(model.to_wire()),
        "alfworld_data_digest": sha256_json(data_signature),
        "formal_config_digest": _formal_config_digest(config),
        "execution_phase": phase,
    }
    if phase in {"train", "smoke"}:
        identity["supervision_digest"] = _sha256_file(supervision_path)
    config_path = output_dir / "config_resolved.json"
    _write_json_atomic(config_path, config)
    return RunContext(
        campaign_id=str(config.get("campaign_id", "skillgen")),
        method_id=METHOD_ID,
        run_seed=int(config["run_seed"]),
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        external_repo=REPO_ROOT / ".external" / "skillgen",
        external_commit=str(lock["skillgen"]["commit"]),
        model_config=model,
        max_environment_actions=int(config["max_environment_actions"]),
        alfworld_data=Path(str(data_signature["resolved_data_root"])),
        config_hash=config_digest,
        code_hash=code_digest,
        run_id=run_id,
        resolved_config_path=config_path.resolve(),
        identity=identity,
        train_manifest_path=train_path,
        validation_manifest_path=None,
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
    test: TaskManifestSet,
    receipts: dict[str, Any],
    git_state: dict[str, Any],
    data_signature: dict[str, Any],
    python_runtime: dict[str, Any],
) -> None:
    parallel = dict(config["parallel"])
    provider_max_inflight = int(
        ctx.campaign["campaign_provider_max_inflight"]
        if ctx.campaign is not None
        else parallel["campaign_provider_max_inflight"]
    )
    _write_json_atomic(ctx.output_dir / "source_lock.json", lock)
    _write_json_atomic(ctx.output_dir / "run_manifest.json", {
        "schema_version": 1,
        "protocol": "protocol-faithful-matched-train-v2",
        "method": METHOD_ID,
        "phase": phase,
        "run_id": ctx.run_id,
        "run_seed": ctx.run_seed,
        "formal": str(config.get("experiment_kind")) == "formal",
        "external_repo": str(lock["skillgen"]["repo"]),
        "external_commit": str(lock["skillgen"]["commit"]),
        "external_runtime_tree": dict(lock["skillgen"]["runtime_tree"]),
        "controller_git": git_state,
        "controller_commit": str(git_state["commit"]),
        "controller_code_digest": ctx.code_hash,
        "python_runtime": dict(python_runtime),
        "identity": dict(ctx.identity),
        "campaign": dict(ctx.campaign) if ctx.campaign is not None else None,
        "resume": dict(ctx.resume) if ctx.resume is not None else None,
        "model": ctx.model_config.model,
        "model_identity": ctx.model_config.to_wire(),
        "reasoning_effort": ctx.model_config.reasoning_effort,
        "max_environment_actions": ctx.max_environment_actions,
        "episode_workers": int(parallel["episode_workers_per_seed"]),
        "provider_max_inflight": provider_max_inflight,
        "alfworld_package_version": python_runtime.get(
            "alfworld_distribution_version"
        ),
        "method_specific_prior": False,
        "extra_train_supervision": True,
        "weak_supervision": "subgoal_progress",
        "sampling": dict(config["sampling"]),
        "extraction": dict(config["extraction"]),
        "inference": dict(config["inference"]),
        "parallel": parallel,
        "train_manifest_hash": train.digest,
        "validation_manifest_hash": None,
        "test_manifest_hash": test.digest,
        "alfworld_data_signature": data_signature,
        "started_at_unix": time.time(),
    })
    _write_json_atomic(ctx.output_dir / "task_manifest.json", {
        "schema_version": 1,
        "train": {
            "path": str(ctx.train_manifest_path), "digest": train.digest,
            "tasks": len(train.tasks), "preflight": receipts["train"],
        },
        "validation": None,
        "test": {
            "path": str(ctx.test_manifest_path), "digest": test.digest,
            "tasks": len(test.tasks), "preflight": receipts["test"],
        },
    })
    _write_json_atomic(ctx.output_dir / "run_state.json", {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "state": "running",
        "phase": phase,
        "updated_at_unix": time.time(),
    })


def _bind_frozen_digest(ctx: RunContext, frozen: FrozenArtifact) -> None:
    """Bind the post-freeze artifact identity into the durable run manifest."""

    manifest_path = ctx.output_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"B4 run manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("B4 run manifest must be a JSON object")
    if (
        manifest.get("method") != METHOD_ID
        or manifest.get("run_id") != ctx.run_id
        or int(manifest.get("run_seed", -1)) != ctx.run_seed
    ):
        raise ValueError("B4 run manifest identity changed before frozen binding")
    identity = dict(manifest.get("identity") or {})
    for declared in (
        manifest.get("frozen_artifact_digest"),
        identity.get("frozen_artifact_digest"),
        ctx.identity.get("frozen_artifact_digest"),
    ):
        if declared not in {None, "", frozen.digest}:
            raise ValueError("B4 run manifest already binds another frozen artifact")
    identity["frozen_artifact_digest"] = frozen.digest
    ctx.identity["frozen_artifact_digest"] = frozen.digest
    manifest.update({
        "identity": identity,
        "frozen_artifact_digest": frozen.digest,
        "frozen_source_train_manifest_hash": frozen.source_train_manifest_hash,
        "frozen_source_validation_manifest_hash": (
            frozen.source_validation_manifest_hash
        ),
    })
    _write_json_atomic(manifest_path, manifest, overwrite=True)


def _sampling_membership(episodes: list[Any], train: TaskManifestSet) -> None:
    expected = {
        (task.task_id, sample_idx)
        for task in train.tasks for sample_idx in range(6)
    }
    observed: set[tuple[str, int]] = set()
    tasks = {task.task_id: task for task in train.tasks}
    for episode in episodes:
        sample_idx = episode.method_metrics.get("sample_idx")
        if sample_idx is None:
            continue
        task = tasks.get(episode.task_id)
        key = (episode.task_id, int(sample_idx))
        if (
            task is None
            or episode.phase not in {"train", "smoke"}
            or episode.method != METHOD_ID
            or episode.manifest_index != task.index
            or episode.gamefile != task.gamefile_rel
            or episode.gamefile_hash != task.gamefile_sha256
            or episode.infrastructure_failure
            or key in observed
        ):
            raise ValueError(f"invalid B4 sampling episode identity: {episode.task_id}")
        observed.add(key)
    if observed != expected:
        raise RuntimeError(
            f"B4 sampling corpus is not exactly six samples/task: "
            f"expected={len(expected)}, observed={len(observed)}"
        )


def _provider_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"B4 provider evidence is missing: {path}")
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError(f"B4 provider evidence is empty or invalid: {path}")
    logical_ids = [str(event.get("call_id", "")) for event in events]
    if any(not value for value in logical_ids) or len(set(logical_ids)) != len(logical_ids):
        raise ValueError("B4 provider evidence has missing or duplicate logical call ids")
    attempts = sum(int(event.get("application_attempts", 0)) for event in events)
    sdk_attempts = sum(int(event.get("sdk_boundary_attempts", 0)) for event in events)
    if attempts < len(events) or sdk_attempts != attempts:
        raise ValueError("B4 provider evidence has inconsistent retry counters")
    return {
        "logical_calls": len(events),
        "application_attempts": attempts,
        "sdk_boundary_attempts": sdk_attempts,
        "provider_retries": attempts - len(events),
        "recovered_provider_calls": sum(bool(event.get("recovered")) for event in events),
        "failed_provider_calls": sum(event.get("status") == "failed" for event in events),
        "provider_queue_wait_ms": sum(
            int(event.get("provider_queue_wait_ms", 0)) for event in events
        ),
        "retry_backoff_ms": sum(
            int(event.get("retry_backoff_ms", 0)) for event in events
        ),
        "evidence_path": str(path),
        "evidence_sha256": _sha256_file(path),
    }


def _run_smoke(
    driver: SkillGenBaselineDriver,
    ctx: RunContext,
    train: TaskManifestSet,
    test: TaskManifestSet,
) -> dict[str, Any]:
    result = driver.smoke(ctx, train)
    _sampling_membership(result.episodes, train)
    inference = [
        episode for episode in result.episodes
        if episode.method_metrics.get("sample_idx") is None
    ]
    if [episode.task_id for episode in inference] != [task.task_id for task in test.tasks]:
        raise RuntimeError("B4 smoke inference is not an ordered Test-6 bijection")
    if any(episode.phase != "smoke_test" for episode in inference):
        raise RuntimeError("B4 smoke inference did not run in an isolated smoke-test worker")
    validate_episode_usage(result.episodes)
    train_worker = json.loads(
        (ctx.output_dir / "smoke" / "worker_result.json").read_text(encoding="utf-8")
    )
    test_worker = json.loads(
        (ctx.output_dir / "smoke_test" / "worker_result.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = FrozenArtifact.load(ctx.output_dir / "smoke_frozen")
    _bind_frozen_digest(ctx, frozen)
    if (
        int(train_worker.get("sampling_episodes", -1)) != 36
        or int(test_worker.get("rows", -1)) != 6
        or test_worker.get("frozen_digest_before") != frozen.digest
        or test_worker.get("frozen_digest_after") != frozen.digest
    ):
        raise RuntimeError(
            "B4 smoke did not complete Train6 extraction plus isolated frozen Test6"
        )
    write_evaluated_episodes_jsonl(
        inference,
        ctx.output_dir / "smoke_test" / "evaluated_common_episodes.jsonl",
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "method": METHOD_ID,
        "phase": "smoke",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "sampling_episodes": 36,
        "test_inference_episodes": 6,
        "official_test_successes": sum(int(item.official_success) for item in inference),
        "task_success_required": False,
        "frozen_digest": frozen.digest,
        "usage": result.usage.to_dict(),
        "method_metrics": dict(train_worker.get("train_summary") or {}),
        "provider_evidence": {
            "sampling": _provider_evidence(ctx.output_dir / "smoke" / "provider_calls.jsonl"),
            "inference": _provider_evidence(
                ctx.output_dir / "smoke_test" / "provider_calls.jsonl"
            ),
        },
    }
    _write_json_atomic(ctx.output_dir / "smoke_report.json", report)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=ctx.model_config.api_key_env)
    return report


def _run_train(
    driver: SkillGenBaselineDriver,
    ctx: RunContext,
    train: TaskManifestSet,
) -> dict[str, Any]:
    result = driver.train(ctx, train, None)
    _sampling_membership(result.episodes, train)
    validate_episode_usage(result.episodes)
    frozen = driver.freeze(ctx, result)
    _bind_frozen_digest(ctx, frozen)
    provider_evidence = _provider_evidence(
        ctx.output_dir / "train" / "provider_calls.jsonl"
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "method": METHOD_ID,
        "phase": "train",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "protocol": {
            "workflow": "train120_x6_sampling_extraction_freeze",
            "evaluation_scope": "training_only",
            "held_out": False,
            "generalization_claim": False,
        },
        "frozen": {
            "root": str(frozen.root),
            "digest": frozen.digest,
            "source_train_manifest_hash": frozen.source_train_manifest_hash,
            "source_validation_manifest_hash": None,
        },
        "training_cost": {
            "unique_train_tasks": len(train.tasks),
            "sampling_episodes": _episode_cost(result.episodes),
            "usage": result.usage.to_dict(),
            "api_cost": result.usage.api_cost,
            "api_cost_unpriced": result.usage.api_cost_unpriced,
            "method_metrics": result.method_metrics,
            "provider_retries": provider_evidence["provider_retries"],
            "provider_evidence": provider_evidence,
        },
    }
    _write_json_atomic(ctx.output_dir / "report.json", report)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=ctx.model_config.api_key_env)
    return report


def _load_frozen_source(
    source_run: Path,
    *,
    train: TaskManifestSet,
    seed: int,
    expected_identity: dict[str, str],
    expected_campaign: dict[str, Any] | None,
) -> tuple[FrozenArtifact, Path]:
    completion = json.loads((source_run / "completion.json").read_text(encoding="utf-8"))
    report = json.loads((source_run / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((source_run / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        completion.get("passed") is not True
        or completion.get("phase") != "train"
        or report.get("passed") is not True
        or report.get("method") != METHOD_ID
        or manifest.get("method") != METHOD_ID
        or int(manifest.get("run_seed", -1)) != seed
    ):
        raise ValueError("B4 source run is not a completed matching Train run")
    for key in (
        "train_manifest_digest", "test_manifest_digest", "external_source_digest",
        "controller_code_digest", "model_identity_digest", "alfworld_data_digest",
        "formal_config_digest",
    ):
        if dict(manifest.get("identity") or {}).get(key) != expected_identity.get(key):
            raise ValueError(f"B4 source run identity mismatch for {key}")
    if expected_campaign is not None:
        source_campaign = dict(manifest.get("campaign") or {})
        if source_campaign.get("campaign_lock_digest") != expected_campaign.get(
            "campaign_lock_digest"
        ):
            raise ValueError("B4 source Train run belongs to another campaign lock")
    frozen_dir = source_run / "frozen"
    frozen = FrozenArtifact.load(frozen_dir)
    if (
        frozen.method_id != METHOD_ID
        or frozen.source_train_manifest_hash != train.digest
        or frozen.source_validation_manifest_hash is not None
        or dict(report.get("frozen") or {}).get("digest") != frozen.digest
        or manifest.get("frozen_artifact_digest") != frozen.digest
        or dict(manifest.get("identity") or {}).get("frozen_artifact_digest")
        != frozen.digest
        or dict(completion.get("identity") or {}).get("frozen_artifact_digest")
        != frozen.digest
    ):
        raise ValueError("B4 source frozen artifact identity is invalid")
    assert_frozen_unchanged(frozen)
    return frozen, frozen_dir


def _load_resume_descriptor(
    source_run: Path,
    *,
    seed: int,
    expected_identity: dict[str, str],
    expected_campaign: dict[str, Any],
) -> dict[str, Any]:
    """Authorize a prior interrupted Train run as a sampling-job source."""

    source_run = source_run.expanduser().resolve(strict=True)
    manifest_path = source_run / "run_manifest.json"
    state_path = source_run / "run_state.json"
    jobs_dir = source_run / "train" / "sampling" / "jobs"
    if not manifest_path.is_file() or not state_path.is_file() or not jobs_dir.is_dir():
        raise ValueError(
            "B4 resume source lacks run identity, run state, or sampling checkpoints"
        )
    manifest_bytes = manifest_path.read_bytes()
    state_bytes = state_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        state = json.loads(state_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("B4 resume source metadata is unreadable") from exc
    if not isinstance(manifest, dict) or not isinstance(state, dict):
        raise ValueError("B4 resume source metadata must be JSON objects")
    if (
        manifest.get("method") != METHOD_ID
        or manifest.get("phase") != "train"
        or int(manifest.get("run_seed", -1)) != int(seed)
        or state.get("state") not in {"running", "failed"}
    ):
        raise ValueError("B4 resume source is not an interrupted matching Train run")
    if state.get("state") == "failed" and state.get("failure_kind") not in {
        None,
        "infrastructure_failure",
    }:
        raise ValueError("B4 protocol failures cannot be resumed")
    source_identity = dict(manifest.get("identity") or {})
    identity_keys = (
        "config_digest",
        "train_manifest_digest",
        "test_manifest_digest",
        "external_source_digest",
        "controller_code_digest",
        "model_identity_digest",
        "alfworld_data_digest",
        "formal_config_digest",
        "supervision_digest",
    )
    mismatches = [
        key for key in identity_keys
        if source_identity.get(key) != expected_identity.get(key)
    ]
    if mismatches:
        raise ValueError("B4 resume source identity mismatch: " + ", ".join(mismatches))
    source_campaign = dict(manifest.get("campaign") or {})
    if source_campaign.get("campaign_lock_digest") != expected_campaign.get(
        "campaign_lock_digest"
    ):
        raise ValueError("B4 resume source belongs to another campaign lock")
    checkpoint_inventory = build_checkpoint_inventory(jobs_dir)
    if not checkpoint_inventory["files"]:
        raise ValueError("B4 resume source contains no completed sampling-job boundary")
    return {
        "schema_version": 2,
        "boundary": "sampling_job",
        "source_run": str(source_run),
        "run_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "run_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "checkpoint_files_observed": len(checkpoint_inventory["files"]),
        "checkpoint_inventory": checkpoint_inventory,
    }


def _run_test(
    driver: SkillGenBaselineDriver,
    ctx: RunContext,
    train: TaskManifestSet,
    test: TaskManifestSet,
    source_run: Path,
) -> dict[str, Any]:
    frozen, frozen_dir = _load_frozen_source(
        source_run,
        train=train,
        seed=ctx.run_seed,
        expected_identity=ctx.identity,
        expected_campaign=ctx.campaign,
    )
    _bind_frozen_digest(ctx, frozen)
    episodes = driver.evaluate_test(
        ctx, frozen, train, None, test, frozen_dir=frozen_dir,
    )
    bijection = verify_final_evaluation_bijection(
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
    write_evaluated_episodes_jsonl(
        episodes, ctx.output_dir / "test" / "evaluated_common_episodes.jsonl"
    )
    rows = [TaskRow.from_episode(item) for item in episodes]
    write_rows_jsonl(rows, ctx.output_dir / "test" / "task_rows.jsonl")
    usage = UsageSnapshot.load(ctx.output_dir / "test" / "usage.json")
    if usage.target.calls <= 0 or usage.evolution.calls != 0:
        raise RuntimeError("B4 frozen Test has invalid provider role usage")
    summary = summarize_rows(
        rows,
        task_types=list(ALFWORLD_FORMAL_TASK_TYPES),
        api_cost=usage.api_cost,
        api_cost_unpriced=usage.api_cost_unpriced,
    )
    if summary["tasks"] != len(test.tasks) or summary["attempted_tasks"] != len(
        test.tasks
    ):
        raise RuntimeError("B4 Test summary denominator differs from Test manifest")
    _write_json_atomic(ctx.output_dir / "test" / "summary.json", {
        "schema_version": 1,
        "method": METHOD_ID,
        "phase": "test",
        "evaluation_scope": "held_out_test",
        "held_out": True,
        "generalization_claim": True,
        "frozen_digest": frozen.digest,
        "bijection": bijection,
        **summary,
    })
    worker = json.loads(
        (ctx.output_dir / "test" / "worker_result.json").read_text(encoding="utf-8")
    )
    provider_evidence = _provider_evidence(
        ctx.output_dir / "test" / "provider_calls.jsonl"
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "method": METHOD_ID,
        "phase": "test",
        "run_id": ctx.run_id,
        "output_dir": str(ctx.output_dir),
        "source_run": str(source_run.resolve()),
        "protocol": {
            "workflow": "frozen_skillgen_library_test134_read_only",
            "evaluation_scope": "held_out_test",
            "held_out": True,
            "generalization_claim": True,
        },
        "frozen": {"root": str(frozen.root), "digest": frozen.digest},
        "bijection": bijection,
        "effectiveness": summary,
        "test_cost": {
            "episodes": _episode_cost(episodes),
            "usage": usage.to_dict(),
            "api_cost": usage.api_cost,
            "api_cost_unpriced": usage.api_cost_unpriced,
            "retrieval_count": int(worker.get("retrieval_count", 0)),
            "retrieved_skill_count": int(worker.get("retrieved_skill_count", 0)),
            "provider_retries": provider_evidence["provider_retries"],
            "provider_evidence": provider_evidence,
        },
        "comparison_metrics": {
            key: summary[key] for key in (
                "official_success", "official_success_rate", "official_rate",
                "micro_average_official_rate", "contract_consistency_rate",
                "contract_consistent_success_rate", "common_strict_success_rate",
                "strict_rate", "macro_family_official_rate", "actions_per_task",
                "actions_per_solved", "p50_actions", "p90_actions",
                "calls_per_task", "tokens_per_task", "tokens_per_solved",
                "cost_per_task", "cost_per_solved", "cost_metrics_status",
                "latency_per_task_ms", "p50_latency_ms", "p90_latency_ms",
                "target_reasoning_tokens", "reasoning_tokens_in_completion",
                "delta_vs_pure_dynamic", "positive_transfer", "negative_transfer",
                "paired_transfer_status", "family",
            )
        },
    }
    _write_json_atomic(ctx.output_dir / "test_report.json", report)
    assert_frozen_unchanged(frozen)
    assert_no_secrets_on_disk(ctx.output_dir, api_key_env=ctx.model_config.api_key_env)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=METHOD_ID, choices=[METHOD_ID])
    parser.add_argument("--phase", required=True, choices=["smoke", "train", "test"])
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--source-run", default=None)
    parser.add_argument("--resume-source-run", default=None)
    parser.add_argument("--campaign-lock", default=None)
    parser.add_argument("--supervision", default=None)
    parser.add_argument("--config", default="configs/baselines/b4_skillgen_s.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    run_id = _new_run_id(args.phase)
    output_dir: Path | None = None
    output_owned = False
    model: ModelConfig | None = None
    try:
        if args.validation_manifest:
            raise ValueError("B4 SkillGen-S is forbidden from receiving Validation")
        if args.resume_source_run and args.phase != "train":
            raise ValueError("B4 --resume-source-run is accepted only for formal Train")
        if args.phase == "test" and not args.source_run:
            raise ValueError("B4 Test requires --source-run")
        if args.phase == "test" and args.supervision:
            raise ValueError("B4 Test must not receive a supervision path")
        if args.phase != "test" and args.source_run:
            raise ValueError("--source-run is accepted only for B4 Test")
        lock, driver = load_lock_and_driver(
            repo_root=REPO_ROOT,
            config_path=_path(args.config),
            supervision_path=args.supervision,
        )
        config = dict(driver.config)
        seed = int(args.seed if args.seed is not None else dict(config["train"])["seed"])
        config["run_seed"] = seed
        config["train"] = {**dict(config["train"]), "seed": seed}
        driver.config = dict(config)
        kind = str(config.get("experiment_kind"))
        if args.phase == "smoke" and kind != "smoke":
            raise ValueError("B4 smoke requires b4_skillgen_s_smoke.yaml")
        if args.phase != "smoke" and kind != "formal":
            raise ValueError("B4 train/test require b4_skillgen_s.yaml")

        model = ModelConfig.from_mapping(dict(config["model"]))
        model.validate_formal_identity()
        expected_python = resolve_formal_python(
            REPO_ROOT, str(config["worker_python"])
        )
        python_runtime = verify_runtime_python_executable(
            repo_root=REPO_ROOT,
            expected_python=expected_python,
            require_venv=True,
            method=METHOD_ID,
            expected_distributions=worker_expected_distributions(METHOD_ID),
            expected_python_major_minor="3.9",
        )
        data_raw = os.environ.get("ALFWORLD_DATA", "").strip()
        if not data_raw:
            raise RuntimeError("ALFWORLD_DATA is not set")
        alfworld_data = Path(data_raw).expanduser().resolve(strict=True)
        train_path = _path(args.train_manifest)
        test_path = _path(args.test_manifest)
        train = TaskManifestSet.load(train_path)
        test = TaskManifestSet.load(test_path)
        verify_disjoint(train, test)
        if args.phase == "smoke":
            receipts = {
                "train": verify_formal_manifest(
                    train, role="train", alfworld_data=alfworld_data,
                    profile="smoke_v1",
                ),
                "test": verify_formal_manifest(
                    test, role="test", alfworld_data=alfworld_data,
                    profile="smoke_v1",
                ),
            }
        else:
            receipts = {
                "train": verify_formal_manifest(
                    train, alfworld_data=alfworld_data, role="train", profile="formal_v2",
                ),
                "test": verify_formal_manifest(
                    test, alfworld_data=alfworld_data, role="test", profile="formal_v2",
                ),
            }

        # Label authority is checked before API credentials so an invalid
        # shipped valid_unseen label file fails closed without an API call.
        if args.phase in {"train", "smoke"}:
            coverage, _ = analyze_label_coverage(train, driver.supervision_path)
            if not coverage.passed:
                if args.output_dir:
                    output_dir = _path(args.output_dir)
                else:
                    output_dir = (
                        REPO_ROOT / "runs" / "baselines" / "preflight"
                        / METHOD_ID / run_id
                    )
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(exist_ok=False)
                output_owned = True
                _write_json_atomic(
                    output_dir / "preflight_label_coverage.json",
                    coverage.to_dict(),
                )
                raise ValueError(
                    "SkillGen-S Train label coverage failed before provider use: "
                    f"covered={coverage.covered_task_count}/{coverage.manifest_task_count}; "
                    f"report={json.dumps(coverage.to_dict(), ensure_ascii=False)}"
                )
        model.require_api_key()
        git_state = _controller_git_state()
        code_digest = hash_code(REPO_ROOT)
        campaign = None
        if args.phase in {"train", "test"}:
            if git_state["dirty"]:
                raise RuntimeError("formal B4 requires a clean experiments/configs source tree")
            if not args.campaign_lock:
                raise ValueError("formal B4 train/test requires --campaign-lock")
            campaign = _load_campaign_descriptor(
                args.campaign_lock,
                config=config,
                lock=lock,
                model=model,
                train=train,
                test=test,
                supervision_digest=(
                    _sha256_file(driver.supervision_path)
                    if args.phase == "train" else None
                ),
                seed=seed,
                git_state=git_state,
                code_digest=code_digest,
            )
        elif args.campaign_lock:
            raise ValueError("B4 smoke does not accept a campaign lock")

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
            phase=args.phase,
            train_path=train_path,
            test_path=test_path,
            train=train,
            test=test,
            supervision_path=driver.supervision_path,
            data_signature=data_signature,
            campaign=campaign,
        )
        if args.resume_source_run:
            if campaign is None:
                raise ValueError("B4 resume requires an immutable formal campaign lock")
            resume = _load_resume_descriptor(
                _path(args.resume_source_run),
                seed=seed,
                expected_identity=ctx.identity,
                expected_campaign=campaign,
            )
            ctx = RunContext(**{**ctx.__dict__, "resume": resume})
        _write_run_identity(
            ctx=ctx,
            lock=lock,
            config=config,
            phase=args.phase,
            train=train,
            test=test,
            receipts=receipts,
            git_state=git_state,
            data_signature=data_signature,
            python_runtime=python_runtime,
        )
        driver.preflight(ctx)
        if args.phase == "smoke":
            report = _run_smoke(driver, ctx, train, test)
        elif args.phase == "train":
            report = _run_train(driver, ctx, train)
        else:
            report = _run_test(driver, ctx, train, test, _path(args.source_run))
        report_name = {
            "smoke": "smoke_report.json", "train": "report.json", "test": "test_report.json",
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
            "schema_version": 1, "run_id": run_id, "state": "completed",
            "phase": args.phase, "updated_at_unix": time.time(),
        }, overwrite=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        error = _safe_error(
            exc, api_key_env=model.api_key_env if model else "MODEL_API_KEY",
        )
        failure_kind = str(getattr(exc, "failure_kind", "protocol_failure"))
        if failure_kind not in {"infrastructure_failure", "protocol_failure"}:
            failure_kind = "protocol_failure"
        failure = {
            "schema_version": 1, "passed": False, "method": METHOD_ID,
            "phase": args.phase, "run_id": run_id,
            "error_type": type(exc).__name__, "error": error,
            "failure_kind": failure_kind,
            "output_dir": str(output_dir) if output_dir else None,
            "failed_at_unix": time.time(),
        }
        if output_owned and output_dir is not None:
            try:
                _write_json_atomic(output_dir / "failure.json", failure)
                _write_json_atomic(output_dir / "run_state.json", {
                    "schema_version": 1, "run_id": run_id, "state": "failed",
                    "phase": args.phase, "error_type": type(exc).__name__,
                    "error": error, "failure_kind": failure_kind,
                    "updated_at_unix": time.time(),
                }, overwrite=True)
            except Exception:
                pass
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
