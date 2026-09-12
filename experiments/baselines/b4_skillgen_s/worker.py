"""Isolated B4 SkillGen-S worker for sampling, extraction, and frozen inference."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.baselines.bootstrap_external import (
    load_lock,
    verify_key_files,
    verify_runtime_tree,
    worker_expected_distributions,
)
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.formal_validation import verify_formal_manifest
from experiments.baselines.common.freeze import FrozenArtifact
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet, sha256_json, verify_disjoint
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.runtime_python import (
    resolve_formal_python,
    verify_runtime_python,
)
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.subprocess_worker import WorkerWire, write_worker_result
from experiments.baselines.common.usage import RoleUsage, UsageSnapshot

from .corpus_builder import (
    SamplingJob,
    build_sampling_jobs,
    corpus_metadata,
    load_verified_checkpoint_bytes,
    save_job_result,
    validate_completed_corpus,
)
from .extraction_adapter import extract_skill_library, load_sentence_encoder
from .inference_adapter import (
    EpisodeInfrastructureError,
    PinnedPromptCore,
    ProviderExecutionError,
    UpstreamRetrievalCore,
    run_skillgen_episode,
)
from .progress_adapter import (
    SkillGenLabel,
    SkillGenLabelAuthorityError,
    SubgoalProgressTracker,
    analyze_label_coverage,
    require_complete_train_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "b4_skillgen_s"
ALLOWED_PHASES = frozenset({"smoke", "smoke_test", "train", "train_eval", "test"})


class WorkerPhaseError(RuntimeError):
    def __init__(self, message: str, *, failure_kind: str = "protocol_failure", **evidence: Any) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.evidence = dict(evidence)


def _nested(config: dict[str, Any], *path: str) -> Any:
    current: Any = config
    for name in path:
        if not isinstance(current, dict) or name not in current:
            raise ValueError(f"resolved B4 config is missing {'.'.join(path)}")
        current = current[name]
    return current


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"resolved B4 config is missing: {target}")
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resolved B4 config root must be a mapping")
    return payload


def _reject_secret_or_initial(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if normalized != "api_key_env" and (
                normalized in {"api_key", "model_api_key", "secret", "password"}
                or normalized.endswith("_api_key")
            ):
                raise ValueError(f"B4 config embeds a secret at {'.'.join(path + (key,))}")
            if normalized in {"initial_skill", "initial_skill_path", "skill_init"}:
                raise ValueError("SkillGen-S has no initial.md/initial skill")
            _reject_secret_or_initial(child, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_or_initial(child, path + (str(index),))


def _validate_config(wire: WorkerWire, config: dict[str, Any]) -> ModelConfig:
    expected_path = (Path(wire.output_dir).resolve() / "config_resolved.json").resolve()
    if Path(wire.config_path).resolve() != expected_path:
        raise ValueError("B4 worker must consume its run-local config_resolved.json")
    if sha256_json(config) != wire.identity.get("config_digest"):
        raise ValueError("resolved B4 config digest does not match worker identity")
    if config.get("method") != METHOD_ID or wire.method != METHOD_ID:
        raise ValueError("worker/config method is not b4_skillgen_s")
    if int(config.get("run_seed", -1)) != wire.run_seed:
        raise ValueError("resolved B4 run_seed differs from the worker wire")
    model = ModelConfig.from_mapping(dict(config.get("model") or {}))
    model.validate_formal_identity()
    if model.to_wire() != wire.model:
        raise ValueError("resolved B4 model identity differs from the worker wire")
    expected: dict[tuple[str, ...], Any] = {
        ("sampling", "prompt_mode"): "1-shot",
        ("sampling", "sampling_count"): 6,
        ("sampling", "max_steps"): 10,
        ("sampling", "max_length"): 64,
        ("sampling", "do_sample"): True,
        ("sampling", "temperature"): 1.0,
        ("sampling", "top_p"): 0.95,
        ("sampling", "hist_size"): 20,
        ("extraction", "gamma"): 0.95,
        ("extraction", "lambda"): 0.9,
        ("extraction", "alpha"): 0.05,
        ("extraction", "q_iterations"): 500,
        ("extraction", "seed"): 42,
        ("extraction", "embedding_model"): "sentence-transformers/all-MiniLM-L6-v2",
        ("inference", "max_steps"): 20,
        ("inference", "max_length"): 64,
        ("inference", "temperature"): 0.0,
        ("inference", "top_p"): 0.95,
        ("inference", "hist_size"): 20,
        ("inference", "top_s"): 1,
        ("inference", "top_ac"): 1,
        ("supervision", "weak_supervision"): "subgoal_progress",
        ("supervision", "extra_train_supervision"): True,
        ("env", "max_steps"): 100,
        ("provider_transport", "sdk_max_retries"): 0,
        ("provider_transport", "application_retry_limit"): 5,
        ("provider_transport", "retry_delays_seconds"): [2, 5, 10, 20],
        ("provider_transport", "deterministic_jitter_ratio"): 0.10,
        ("parallel", "mp_start_method"): "spawn",
    }
    for path, wanted in expected.items():
        actual = _nested(config, *path)
        if actual != wanted:
            raise ValueError(
                f"frozen B4 setting {'.'.join(path)} must be {wanted!r}, got {actual!r}"
            )
    if int(config.get("max_environment_actions", -1)) != 100:
        raise ValueError("B4 common outer action ceiling must remain 100")
    allowed_workers = {1, 6} if config.get("experiment_kind") == "smoke" else {8, 12, 16}
    if int(_nested(config, "parallel", "episode_workers_per_seed")) not in allowed_workers:
        raise ValueError(
            f"B4 episode_workers_per_seed must be one of {sorted(allowed_workers)}"
        )
    if int(_nested(config, "parallel", "test_workers_per_seed")) not in allowed_workers:
        raise ValueError(
            f"B4 test_workers_per_seed must be one of {sorted(allowed_workers)}"
        )
    resume = dict(config.get("resume") or {})
    expected_resume = {
        "enabled": config.get("experiment_kind") == "formal",
        "formal_boundary": "sampling_job",
        "extraction_replay_from_complete_corpus": True,
    }
    for name, wanted in expected_resume.items():
        if resume.get(name) != wanted:
            raise ValueError(
                f"frozen B4 setting resume.{name} must be {wanted!r}, "
                f"got {resume.get(name)!r}"
            )
    _reject_secret_or_initial(config)
    return model


def _external_root(wire: WorkerWire) -> Path:
    root = Path(
        wire.external_method_root or REPO_ROOT / ".external" / "skillgen"
    ).resolve(strict=True)
    return root


def _verify_source(root: Path) -> dict[str, Any]:
    lock = load_lock(REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml")
    entry = dict(lock.get("skillgen") or {})
    expected = str(entry.get("commit", ""))
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode != 0 or actual != expected:
        raise RuntimeError(
            f"pinned SkillGen checkout mismatch: expected={expected}, actual={actual or '<unavailable>'}"
        )
    verified = verify_key_files(root, "skillgen", lock)
    runtime = verify_runtime_tree(root, "skillgen", lock)
    return {
        "repo": str(entry.get("repo", "")),
        "commit": expected,
        "root": str(root),
        "verified_key_files": len(verified),
        "runtime_tree": runtime,
    }


def _validate_manifest_physical(manifest: TaskManifestSet, data_root: Path) -> None:
    if manifest.benchmark != "alfworld":
        raise ValueError("B4 accepts only ALFWorld manifests")
    for task in manifest.tasks:
        path = (data_root / task.gamefile_rel).resolve(strict=True)
        try:
            path.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"manifest gamefile escapes ALFWORLD_DATA: {task.gamefile_rel}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != task.gamefile_sha256:
            raise ValueError(f"manifest gamefile hash mismatch for {task.task_id}")


def _load_manifests(
    wire: WorkerWire,
    config: dict[str, Any],
) -> tuple[TaskManifestSet | None, TaskManifestSet | None, Path]:
    if wire.validation_manifest_path is not None or "validation_manifest_digest" in wire.identity:
        raise ValueError("SkillGen-S must not receive Validation data")
    train: TaskManifestSet | None = None
    if wire.phase not in {"smoke_test", "test"}:
        if not wire.manifest_path:
            raise ValueError(f"B4 {wire.phase} requires an explicit Train manifest")
        train = TaskManifestSet.load(wire.manifest_path)
        if train.digest != wire.identity.get("train_manifest_digest"):
            raise ValueError("B4 Train manifest digest differs from worker identity")
        if train.source_split != "train":
            raise ValueError("B4 training authority must be the Train split")
    elif wire.manifest_path is not None:
        raise ValueError("B4 Test worker must not receive the Train manifest path")
    test: TaskManifestSet | None = None
    if wire.phase in {"smoke_test", "test"}:
        if not wire.test_manifest_path:
            raise ValueError(f"B4 {wire.phase} phase requires an explicit Test manifest")
        test = TaskManifestSet.load(wire.test_manifest_path)
        expected_test_digest = wire.identity.get("test_manifest_digest")
        if test.digest != expected_test_digest:
            raise ValueError(f"B4 {wire.phase} Test manifest digest differs from worker identity")
        if test.source_split != "valid_unseen":
            raise ValueError("B4 Test authority must be valid_unseen")
        if train is not None:
            verify_disjoint(train, test)
    elif wire.test_manifest_path is not None:
        raise ValueError("only B4 Test phase may receive the Test manifest as execution data")
    data_raw = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_raw:
        raise RuntimeError("ALFWORLD_DATA is missing from the worker environment")
    data_root = Path(data_raw).expanduser().resolve(strict=True)
    if train is not None:
        _validate_manifest_physical(train, data_root)
    if test is not None:
        _validate_manifest_physical(test, data_root)
    profile = str(config.get("protocol_profile", "pilot_v1"))
    if profile == "smoke_v1" and wire.phase == "smoke":
        if train is None or test is not None:
            raise AssertionError("B4 smoke Train authority is invalid")
        verify_formal_manifest(
            train, alfworld_data=data_root, role="train", profile=profile,
        )
    if profile == "smoke_v1" and wire.phase == "smoke_test":
        if test is None or train is not None:
            raise AssertionError("B4 smoke-test Test authority is invalid")
        verify_formal_manifest(
            test, alfworld_data=data_root, role="test", profile=profile,
        )
    if profile == "formal_v2" and wire.phase in {"train", "train_eval", "test"}:
        if train is not None:
            verify_formal_manifest(train, alfworld_data=data_root, role="train", profile=profile)
        if test is not None:
            verify_formal_manifest(test, alfworld_data=data_root, role="test", profile=profile)
    return train, test, data_root


def _label_from_wire(payload: dict[str, Any]) -> SkillGenLabel:
    return SkillGenLabel(
        task_key=str(payload["task_key"]),
        source_split=str(payload["source_split"]),
        goal=str(payload["goal"]),
        subgoals=tuple(str(value) for value in payload["subgoals"]),
        difficulty=str(payload["difficulty"]),
        source_line=int(payload["source_line"]),
    )


_PROCESS_PROMPTS: dict[str, PinnedPromptCore] = {}
_PROCESS_RETRIEVAL: dict[str, UpstreamRetrievalCore] = {}
_PROCESS_ENCODERS: dict[str, Any] = {}


def _prompt_core(root: str) -> PinnedPromptCore:
    return _PROCESS_PROMPTS.setdefault(root, PinnedPromptCore(root))


def _retrieval_core(root: str) -> UpstreamRetrievalCore:
    return _PROCESS_RETRIEVAL.setdefault(root, UpstreamRetrievalCore(root))


def _encoder(name: str) -> Any:
    return _PROCESS_ENCODERS.setdefault(name, load_sentence_encoder(name))


def _safe_error(exc: BaseException, api_key_env: str = "MODEL_API_KEY") -> str:
    text = f"{type(exc).__name__}: {exc}"
    secret = os.environ.get(api_key_env, "").strip()
    return text.replace(secret, "[REDACTED]") if secret else text


def _provider_events_from_error(exc: ProviderExecutionError) -> list[dict[str, Any]]:
    events = exc.event.get("provider_calls")
    if isinstance(events, list) and events:
        return [dict(event) for event in events if isinstance(event, dict)]
    event = dict(exc.event)
    event.pop("provider_calls", None)
    event.pop("action_events", None)
    return [event]


def _action_events_from_error(exc: ProviderExecutionError) -> list[dict[str, Any]]:
    events = exc.event.get("action_events")
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, dict)]


def _run_sampling_job(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    job = dict(payload["job"])
    task = ManifestTask.from_dict(dict(job["task"]))
    try:
        row = run_skillgen_episode(
            task=task,
            external_root=payload["external_root"],
            alfworld_data=payload["alfworld_data"],
            model=dict(payload["model"]),
            run_id=str(payload["run_id"]),
            run_seed=int(payload["run_seed"]),
            phase=str(payload["phase"]),
            max_steps=10,
            max_length=64,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            sample_seed=int(job["sample_seed"]),
            sample_idx=int(job["sample_idx"]),
            label=_label_from_wire(dict(job["label"])),
            prompt_core=_prompt_core(str(payload["external_root"])),
            retry_delays=tuple(float(v) for v in payload["retry_delays"]),
            jitter_ratio=float(payload["jitter_ratio"]),
            campaign=payload.get("campaign"),
        )
        row["job_index"] = int(job["job_index"])
        return row
    except ProviderExecutionError as exc:
        return {
            "job_index": int(job["job_index"]),
            "task_id": task.task_id,
            "manifest_index": task.index,
            "sample_idx": int(job["sample_idx"]),
            "sample_seed": int(job["sample_seed"]),
            "failure_kind": exc.failure_kind,
            "error": _safe_error(exc, str(payload["model"].get("api_key_env", "MODEL_API_KEY"))),
            "provider_calls": _provider_events_from_error(exc),
            "actions": _action_events_from_error(exc),
            "wall_time_ms": max(0, int((time.perf_counter() - started) * 1000)),
        }
    except BaseException as exc:
        kind = str(getattr(exc, "failure_kind", "infrastructure_failure"))
        if kind not in {"infrastructure_failure", "protocol_failure"}:
            kind = "infrastructure_failure"
        return {
            "job_index": int(job["job_index"]),
            "task_id": task.task_id,
            "manifest_index": task.index,
            "sample_idx": int(job["sample_idx"]),
            "sample_seed": int(job["sample_seed"]),
            "failure_kind": kind,
            "error": _safe_error(exc, str(payload["model"].get("api_key_env", "MODEL_API_KEY"))),
            "provider_calls": [
                dict(event) for event in getattr(exc, "provider_calls", [])
                if isinstance(event, dict)
            ],
            "actions": [
                dict(event) for event in getattr(exc, "action_events", [])
                if isinstance(event, dict)
            ],
            "wall_time_ms": max(0, int((time.perf_counter() - started) * 1000)),
        }


def _run_inference_job(payload: dict[str, Any]) -> dict[str, Any]:
    task = ManifestTask.from_dict(dict(payload["task"]))
    try:
        return run_skillgen_episode(
            task=task,
            external_root=payload["external_root"],
            alfworld_data=payload["alfworld_data"],
            model=dict(payload["model"]),
            run_id=str(payload["run_id"]),
            run_seed=int(payload["run_seed"]),
            phase=str(payload["phase"]),
            max_steps=20,
            max_length=64,
            temperature=0.0,
            top_p=float(payload["top_p"]),
            do_sample=False,
            sample_seed=int(payload["episode_seed"]),
            frozen_root=payload["frozen_root"],
            encoder=_encoder(str(payload["embedding_model"])),
            prompt_core=_prompt_core(str(payload["external_root"])),
            retrieval_core=_retrieval_core(str(payload["external_root"])),
            retry_delays=tuple(float(v) for v in payload["retry_delays"]),
            jitter_ratio=float(payload["jitter_ratio"]),
            campaign=payload.get("campaign"),
        )
    except ProviderExecutionError as exc:
        return {
            "task_id": task.task_id,
            "manifest_index": task.index,
            "failure_kind": exc.failure_kind,
            "error": _safe_error(exc, str(payload["model"].get("api_key_env", "MODEL_API_KEY"))),
            "provider_calls": _provider_events_from_error(exc),
            "actions": _action_events_from_error(exc),
        }
    except BaseException as exc:
        kind = str(getattr(exc, "failure_kind", "infrastructure_failure"))
        if kind not in {"infrastructure_failure", "protocol_failure"}:
            kind = "infrastructure_failure"
        return {
            "task_id": task.task_id,
            "manifest_index": task.index,
            "failure_kind": kind,
            "error": _safe_error(exc, str(payload["model"].get("api_key_env", "MODEL_API_KEY"))),
            "provider_calls": [
                dict(event) for event in getattr(exc, "provider_calls", [])
                if isinstance(event, dict)
            ],
            "actions": [
                dict(event) for event in getattr(exc, "action_events", [])
                if isinstance(event, dict)
            ],
        }


def _ordered_map(
    payloads: list[dict[str, Any]],
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    workers: int,
    index_field: str,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if workers == 1:
        rows = []
        for payload in payloads:
            row = runner(payload)
            if on_result is not None:
                on_result(row)
            rows.append(row)
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
        ) as pool:
            futures = [pool.submit(runner, payload) for payload in payloads]
            rows = []
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                if on_result is not None:
                    on_result(row)
                rows.append(row)
    return sorted(rows, key=lambda row: int(row[index_field]))


def _checkpoint_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"B4 resume checkpoint has invalid {label}")
    return value


def _validate_resumed_sampling_row(
    row: dict[str, Any],
    *,
    job: SamplingJob,
    data_root: Path,
    source_run_id: str,
    wire: WorkerWire,
    observed_call_ids: set[str],
    extract_action: Callable[[str], str],
) -> None:
    """Re-establish task, environment, action, and provider authority."""

    validate_completed_corpus((job,), (row,))
    logical_task_id = f"{job.task.task_id}::sample_{job.sample_idx}"
    expected_gamefile = (data_root / job.task.gamefile_rel).resolve(strict=True)
    actual_raw = row.get("actual_gamefile")
    if not isinstance(actual_raw, str) or not actual_raw or not Path(actual_raw).is_absolute():
        raise ValueError(f"B4 resume job {job.job_index} lacks an absolute actual_gamefile")
    try:
        actual_gamefile = Path(actual_raw).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"B4 resume job {job.job_index} actual_gamefile is unavailable"
        ) from exc
    fixed_row_identity = {
        "task_type": job.task.task_type,
        "task_key": job.label.task_key,
        "gamefile": job.task.gamefile_rel,
        "gamefile_hash": job.task.gamefile_sha256,
        "failure_kind": "",
    }
    mismatches = [
        name for name, expected in fixed_row_identity.items()
        if row.get(name) != expected
    ]
    if actual_gamefile != expected_gamefile:
        mismatches.append("actual_gamefile")
    if mismatches:
        raise ValueError(
            f"B4 resume job {job.job_index} environment identity mismatch: {mismatches}"
        )

    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"B4 resume job {job.job_index} trajectory is invalid")
    expected_tags = ["OBSERVATION"] + [
        tag for _ in range((len(trajectory) - 1) // 2) for tag in ("ACTION", "OBSERVATION")
    ]
    if (
        (len(trajectory) - 1) % 2
        or len(expected_tags) != len(trajectory)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or item[0] != expected_tags[index]
            or not isinstance(item[1], str)
            for index, item in enumerate(trajectory)
        )
    ):
        raise ValueError(f"B4 resume job {job.job_index} trajectory shape is invalid")
    command_turns = _checkpoint_int(row.get("command_turns"), "command_turns")
    if command_turns != (len(trajectory) - 1) // 2 or not 1 <= command_turns <= 10:
        raise ValueError(f"B4 resume job {job.job_index} command-turn evidence differs")

    actions = row.get("actions")
    if not isinstance(actions, list):
        raise ValueError(f"B4 resume job {job.job_index} action evidence is invalid")
    environment_actions = _checkpoint_int(
        row.get("environment_actions"), "environment_actions"
    )
    if environment_actions != len(actions) or environment_actions > command_turns:
        raise ValueError(f"B4 resume job {job.job_index} action count differs")
    action_turns: list[int] = []
    for step_index, event in enumerate(actions):
        if not isinstance(event, dict):
            raise ValueError(f"B4 resume job {job.job_index} action event is invalid")
        turn = _checkpoint_int(event.get("command_turn_index"), "command_turn_index")
        if (
            event.get("episode_task_id") != logical_task_id
            or event.get("step_index") != step_index
            or turn >= command_turns
            or not isinstance(event.get("action"), str)
            or not isinstance(event.get("env_feedback"), str)
            or not isinstance(event.get("reward"), (int, float))
            or isinstance(event.get("reward"), bool)
            or not isinstance(event.get("done"), bool)
            or not isinstance(event.get("won"), bool)
            or not isinstance(event.get("admissible_before"), bool)
            or event["action"] != trajectory[1 + 2 * turn][1]
            or event["env_feedback"] != trajectory[2 + 2 * turn][1]
        ):
            raise ValueError(f"B4 resume job {job.job_index} action evidence differs")
        action_turns.append(turn)
    if action_turns != sorted(set(action_turns)):
        raise ValueError(f"B4 resume job {job.job_index} action turns are ambiguous")

    official_success = row.get("official_success")
    environment_done = row.get("environment_done")
    termination_reason = row.get("termination_reason")
    if (
        not isinstance(official_success, bool)
        or not isinstance(environment_done, bool)
        or termination_reason not in {
            "official_won", "environment_done_without_win", "internal_step_limit",
        }
        or any(bool(event["won"]) for event in actions) != official_success
        or (official_success and termination_reason != "official_won")
        or (termination_reason == "environment_done_without_win" and not environment_done)
    ):
        raise ValueError(f"B4 resume job {job.job_index} terminal evidence differs")

    provider_calls = row.get("provider_calls")
    if not isinstance(provider_calls, list) or len(provider_calls) != command_turns:
        raise ValueError(f"B4 resume job {job.job_index} provider evidence is incomplete")
    action_by_turn = {int(event["command_turn_index"]): event for event in actions}
    parsed_actions: list[str] = []
    for turn in range(command_turns):
        provider_event = provider_calls[turn]
        response_text = provider_event.get("response_text") if isinstance(
            provider_event, dict
        ) else None
        if not isinstance(response_text, str) or not response_text:
            raise ValueError(f"B4 resume job {job.job_index} provider response is missing")
        parsed = str(extract_action(response_text))
        if parsed.endswith("."):
            parsed = parsed[:-1]
        if parsed != trajectory[1 + 2 * turn][1]:
            raise ValueError(f"B4 resume job {job.job_index} response/action evidence differs")
        parsed_actions.append(parsed)
        if parsed != "check valid actions" and turn not in action_by_turn:
            raise ValueError(f"B4 resume job {job.job_index} lacks an executed action event")

    grounding = row.get("grounding")
    if (
        not isinstance(grounding, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in grounding)
        or grounding != sorted(set(grounding))
        or any(value < 0 or value >= command_turns for value in grounding)
    ):
        raise ValueError(f"B4 resume job {job.job_index} grounding evidence is invalid")
    expected_grounding = [
        turn for turn, action in enumerate(parsed_actions)
        if action == "check valid actions"
        or bool(action_by_turn[turn].get("admissible_before"))
    ]
    if grounding != expected_grounding:
        raise ValueError(f"B4 resume job {job.job_index} grounding evidence differs")
    expected_grounding_rate = len(grounding) / command_turns
    try:
        grounding_rate = float(row.get("grounding_rate"))
        progress_rate = float(row.get("progress_rate"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"B4 resume job {job.job_index} progress evidence is invalid") from exc
    if abs(grounding_rate - expected_grounding_rate) > 1e-12 or not 0.0 <= progress_rate <= 1.0:
        raise ValueError(f"B4 resume job {job.job_index} progress evidence differs")
    progress = row.get("progress")
    if not isinstance(progress, list):
        raise ValueError(f"B4 resume job {job.job_index} progress evidence is invalid")
    previous_turn = -1
    previous_rate = 0.0
    for item in progress:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"B4 resume job {job.job_index} progress event is invalid")
        turn = _checkpoint_int(item[0], "progress turn")
        try:
            rate = float(item[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"B4 resume job {job.job_index} progress event is invalid") from exc
        if turn <= previous_turn or turn >= command_turns or not previous_rate < rate <= 1.0:
            raise ValueError(f"B4 resume job {job.job_index} progress event differs")
        previous_turn, previous_rate = turn, rate
    if progress and abs(previous_rate - progress_rate) > 1e-12:
        raise ValueError(f"B4 resume job {job.job_index} terminal progress differs")
    if not progress and progress_rate != 0.0:
        raise ValueError(f"B4 resume job {job.job_index} terminal progress differs")
    tracker = SubgoalProgressTracker(job.label)
    recomputed_progress: list[list[float | int]] = []
    recomputed_rate = 0.0
    for turn in range(command_turns):
        event = action_by_turn.get(turn)
        new_rate = float(
            tracker.observe(
                trajectory[2 + 2 * turn][1],
                done=bool(event.get("done")) if event is not None else False,
            )
        )
        if new_rate > recomputed_rate:
            recomputed_progress.append([turn, new_rate])
        recomputed_rate = new_rate
    if progress != recomputed_progress or abs(progress_rate - recomputed_rate) > 1e-12:
        raise ValueError(f"B4 resume job {job.job_index} supervised progress differs")

    if row.get("retrieval") != [] or row.get("embedding_calls") != 0:
        raise ValueError(f"B4 resume job {job.job_index} contains inference evidence")
    _checkpoint_int(row.get("wall_time_ms"), "wall_time_ms")
    expected_provider = {
        "schema_version": 1,
        "event": "provider_call",
        "method": METHOD_ID,
        "phase": "train",
        "run_id": source_run_id,
        "run_seed": wire.run_seed,
        "episode_task_id": logical_task_id,
        "role": "target",
        "stage": "sampling",
        "model": wire.model["model"],
        "reasoning_effort": wire.model["reasoning_effort"],
        "max_tokens": 64,
        "temperature": 1.0,
        "top_p": 0.95,
        "sample_seed": job.sample_seed,
        "do_sample": True,
        "requested_retry_limit": 5,
        "status": "succeeded",
    }
    summed_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }
    for turn, event in enumerate(provider_calls):
        if not isinstance(event, dict) or any(
            event.get(name) != expected for name, expected in expected_provider.items()
        ):
            raise ValueError(f"B4 resume job {job.job_index} provider identity differs")
        call_id = str(event.get("call_id", ""))
        response_text = event.get("response_text")
        messages_digest = str(event.get("messages_sha256", ""))
        expected_call_id = "skillgen_" + hashlib.sha256(
            f"{source_run_id}\0{logical_task_id}\0sampling\0{turn}\0{messages_digest}".encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        if (
            call_id != expected_call_id
            or call_id in observed_call_ids
            or not isinstance(response_text, str)
            or not response_text
            or hashlib.sha256(response_text.encode("utf-8")).hexdigest()
            != event.get("response_sha256")
        ):
            raise ValueError(f"B4 resume job {job.job_index} provider call evidence differs")
        observed_call_ids.add(call_id)
        attempts = _checkpoint_int(
            event.get("application_attempts"), "provider application_attempts", minimum=1,
        )
        sdk_attempts = _checkpoint_int(
            event.get("sdk_boundary_attempts"), "provider sdk_boundary_attempts", minimum=1,
        )
        prompt_tokens = _checkpoint_int(event.get("prompt_tokens"), "prompt_tokens")
        completion_tokens = _checkpoint_int(
            event.get("completion_tokens"), "completion_tokens"
        )
        reasoning_tokens = _checkpoint_int(
            event.get("reasoning_tokens"), "reasoning_tokens"
        )
        if (
            attempts != sdk_attempts
            or attempts > 5
            or event.get("total_tokens") != prompt_tokens + completion_tokens
            or len(messages_digest) != 64
            or any(character not in "0123456789abcdef" for character in messages_digest)
        ):
            raise ValueError(f"B4 resume job {job.job_index} provider usage differs")
        summed_usage["calls"] += 1
        summed_usage["prompt_tokens"] += prompt_tokens
        summed_usage["completion_tokens"] += completion_tokens
        summed_usage["reasoning_tokens"] += reasoning_tokens
    if row.get("target_usage") != summed_usage:
        raise ValueError(f"B4 resume job {job.job_index} target usage differs")


def _load_resume_rows(
    wire: WorkerWire,
    jobs: tuple[SamplingJob, ...],
    data_root: Path,
    *,
    extract_action: Callable[[str], str] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load only identity-bound successful sampling-job checkpoints."""

    if wire.resume is None:
        return {}, {"resumed": False, "reused_sampling_jobs": 0}
    if wire.phase != "train":
        raise ValueError("B4 resume is accepted only for formal Train")
    descriptor = dict(wire.resume)
    if extract_action is None:
        extract_action = _prompt_core(str(_external_root(wire))).extract_action
    if (
        int(descriptor.get("schema_version", 0)) != 2
        or descriptor.get("boundary") != "sampling_job"
    ):
        raise ValueError("B4 resume descriptor has an invalid schema or boundary")
    source_run = Path(str(descriptor.get("source_run", ""))).expanduser().resolve(
        strict=True
    )
    if source_run == Path(wire.output_dir).resolve():
        raise ValueError("B4 resume source must differ from the current run")
    manifest_path = source_run / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("B4 resume source has no run_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != descriptor.get(
        "run_manifest_sha256"
    ):
        raise ValueError("B4 resume source run manifest digest changed")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("B4 resume source run manifest is unreadable") from exc
    state_path = source_run / "run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("B4 resume source has no run_state.json")
    state_bytes = state_path.read_bytes()
    if hashlib.sha256(state_bytes).hexdigest() != descriptor.get("run_state_sha256"):
        raise ValueError("B4 resume source run state digest changed")
    try:
        state = json.loads(state_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("B4 resume source run state is unreadable") from exc
    if not isinstance(state, dict):
        raise ValueError("B4 resume source run state must be an object")
    source_identity = dict(manifest.get("identity") or {})
    identity_keys = (
        "config_digest",
        "train_manifest_digest",
        "external_source_digest",
        "controller_code_digest",
        "model_identity_digest",
        "alfworld_data_digest",
        "formal_config_digest",
        "supervision_digest",
    )
    mismatches = [
        name for name in identity_keys
        if source_identity.get(name) != wire.identity.get(name)
    ]
    if (
        manifest.get("method") != METHOD_ID
        or manifest.get("phase") != "train"
        or int(manifest.get("run_seed", -1)) != wire.run_seed
        or not isinstance(manifest.get("run_id"), str)
        or not manifest.get("run_id")
        or state.get("run_id") != manifest.get("run_id")
        or state.get("phase") != "train"
        or state.get("state") not in {"running", "failed"}
        or (
            state.get("state") == "failed"
            and state.get("failure_kind") not in {None, "infrastructure_failure"}
        )
        or mismatches
    ):
        raise ValueError(
            "B4 resume source identity mismatch: "
            + ", ".join(mismatches or ["method/phase/run_seed"])
        )
    if wire.campaign is not None:
        source_campaign = dict(manifest.get("campaign") or {})
        if source_campaign.get("campaign_lock_digest") != wire.campaign.get(
            "campaign_lock_digest"
        ):
            raise ValueError("B4 resume source belongs to a different campaign lock")

    jobs_dir = source_run / "train" / "sampling" / "jobs"
    if not jobs_dir.is_dir():
        raise FileNotFoundError("B4 resume source has no sampling-job checkpoints")
    inventory = descriptor.get("checkpoint_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("B4 resume descriptor lacks a checkpoint inventory")
    checkpoint_bytes = load_verified_checkpoint_bytes(jobs_dir, inventory)
    if int(descriptor.get("checkpoint_files_observed", -1)) != len(checkpoint_bytes):
        raise ValueError("B4 resume descriptor checkpoint count differs")
    by_index = {job.job_index: job for job in jobs}
    reused: dict[int, dict[str, Any]] = {}
    observed_call_ids: set[str] = set()
    for name, content in checkpoint_bytes.items():
        path = jobs_dir / name
        index = int(path.stem)
        job = by_index.get(index)
        if job is None or index in reused:
            raise ValueError(f"B4 resume contains an unexpected job index: {index}")
        try:
            row = json.loads(content.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"B4 resume job is unreadable: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"B4 resume job must be an object: {path}")
        if row.get("failure_kind"):
            continue
        _validate_resumed_sampling_row(
            row,
            job=job,
            data_root=data_root,
            source_run_id=str(manifest["run_id"]),
            wire=wire,
            observed_call_ids=observed_call_ids,
            extract_action=extract_action,
        )
        reused[index] = dict(row)
    return reused, {
        "resumed": True,
        "source_run": str(source_run),
        "source_run_manifest_sha256": descriptor["run_manifest_sha256"],
        "source_run_state_sha256": descriptor["run_state_sha256"],
        "checkpoint_inventory_digest": inventory["digest"],
        "reused_sampling_jobs": len(reused),
    }


def _write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    return _write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_bytes(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8"),
    )


def _episode_record(row: dict[str, Any], *, phase: str, run_seed: int, artifact_digest: str) -> CommonEpisodeRecord:
    usage = dict(row["target_usage"])
    retrieval = list(row.get("retrieval") or [])
    return CommonEpisodeRecord(
        method=METHOD_ID,
        phase=phase,
        run_seed=run_seed,
        task_id=str(row["task_id"]),
        task_type=str(row["task_type"]),
        manifest_index=int(row["manifest_index"]),
        gamefile=str(row["gamefile"]),
        gamefile_hash=str(row["gamefile_hash"]),
        official_success=bool(row["official_success"]),
        task_contract_success=None,
        strict_success=None,
        environment_actions=int(row["environment_actions"]),
        invalid_actions=None,
        command_turns=int(row["command_turns"]),
        timeout=str(row.get("termination_reason")) == "internal_step_limit",
        termination_reason=str(row.get("termination_reason", "")),
        target_llm_calls=int(usage["calls"]),
        target_prompt_tokens=int(usage["prompt_tokens"]),
        target_completion_tokens=int(usage["completion_tokens"]),
        target_reasoning_tokens=int(usage["reasoning_tokens"]),
        evolution_llm_calls=0,
        evolution_prompt_tokens=0,
        evolution_completion_tokens=0,
        embedding_calls=int(row.get("embedding_calls", 0)),
        wall_time_ms=int(row["wall_time_ms"]),
        artifact_digest_before=artifact_digest,
        artifact_digest_after=artifact_digest,
        method_metrics={
            "sample_idx": row.get("sample_idx"),
            "sample_seed": row.get("sample_seed"),
            "progress_rate": float(row.get("progress_rate", 0.0)),
            "grounding_rate": float(row.get("grounding_rate", 0.0)),
            "task_key": str(row.get("task_key", "")),
            "actual_gamefile": str(row.get("actual_gamefile", "")),
            "retrieval_count": len(retrieval),
            "retrieved_skill_count": sum(
                int(item.get("retrieved_skill_count", 0)) for item in retrieval
            ),
        },
    )


def _usage(rows: list[dict[str, Any]], *, embedding_calls: int = 0, wall_time_ms: int = 0) -> UsageSnapshot:
    usage = UsageSnapshot(embedding_calls=embedding_calls, wall_time_ms=wall_time_ms)
    for row in rows:
        for event in row.get("provider_calls", []):
            if event.get("status") != "succeeded":
                continue
            delta = RoleUsage(
                calls=1,
                prompt_tokens=int(event["prompt_tokens"]),
                completion_tokens=int(event["completion_tokens"]),
                reasoning_tokens=int(event.get("reasoning_tokens") or 0),
            )
            usage.target.add(delta)
            usage.per_stage.setdefault(str(event["stage"]), RoleUsage()).add(delta)
    return usage


def _persist_episode_evidence(
    phase_dir: Path,
    rows: list[dict[str, Any]],
    *,
    phase: str,
    run_seed: int,
    artifact_digest: str,
) -> list[CommonEpisodeRecord]:
    episodes = [
        _episode_record(row, phase=phase, run_seed=run_seed, artifact_digest=artifact_digest)
        for row in rows
    ]
    _write_jsonl(
        phase_dir / "rollout" / "common_episodes.jsonl",
        [episode.to_dict() for episode in episodes],
    )
    actions = [dict(event) for row in rows for event in row.get("actions", [])]
    _write_jsonl(
        phase_dir / "rollout" / "common_environment_actions.jsonl", actions,
    )
    calls = [dict(event) for row in rows for event in row.get("provider_calls", [])]
    _write_jsonl(phase_dir / "provider_calls.jsonl", calls)
    return episodes


def _sampling_payloads(
    jobs: tuple[SamplingJob, ...],
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    external_root: Path,
    data_root: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "job": job.to_wire(),
            "external_root": str(external_root),
            "alfworld_data": str(data_root),
            "model": dict(wire.model),
            "run_id": wire.run_id,
            "run_seed": wire.run_seed,
            "phase": wire.phase,
            "retry_delays": list(_nested(config, "provider_transport", "retry_delays_seconds")),
            "jitter_ratio": float(_nested(config, "provider_transport", "deterministic_jitter_ratio")),
            "campaign": wire.campaign,
        }
        for job in jobs
    ]


def _inference_payloads(
    tasks: tuple[ManifestTask, ...],
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    external_root: Path,
    data_root: Path,
    frozen_root: Path,
) -> list[dict[str, Any]]:
    from .corpus_builder import stable_sample_seed

    return [
        {
            "task": task.to_dict(),
            "external_root": str(external_root),
            "alfworld_data": str(data_root),
            "model": dict(wire.model),
            "run_id": wire.run_id,
            "run_seed": wire.run_seed,
            "phase": wire.phase,
            "episode_seed": stable_sample_seed(wire.run_seed, task.task_id, 0),
            "frozen_root": str(frozen_root),
            "embedding_model": str(_nested(config, "extraction", "embedding_model")),
            "top_p": float(_nested(config, "inference", "top_p")),
            "retry_delays": list(_nested(config, "provider_transport", "retry_delays_seconds")),
            "jitter_ratio": float(_nested(config, "provider_transport", "deterministic_jitter_ratio")),
            "campaign": wire.campaign,
        }
        for task in tasks
    ]


def _fail_if_episode_boundary_failed(rows: list[dict[str, Any]], *, stage: str) -> None:
    failures = [row for row in rows if row.get("failure_kind")]
    if failures:
        first = failures[0]
        raise WorkerPhaseError(
            f"B4 {stage} stopped before a legal durable boundary: {first.get('error', 'unknown')}",
            failure_kind=str(first["failure_kind"]),
            failed_job_count=len(failures),
            first_failed_task_id=first.get("task_id"),
            first_failed_job_index=first.get("job_index"),
        )


def _train_or_smoke(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    train: TaskManifestSet,
    data_root: Path,
    external_root: Path,
    phase_dir: Path,
    label_report: Any,
    labels: dict[str, SkillGenLabel],
) -> dict[str, Any]:
    report = label_report
    _write_json(phase_dir / "label_coverage.json", report.to_dict())
    if not report.passed:
        require_complete_train_coverage(train, wire.supervision_path)
    jobs = build_sampling_jobs(train, labels, run_seed=wire.run_seed, sampling_count=6)
    reused_rows, resume_evidence = _load_resume_rows(
        wire,
        jobs,
        data_root,
        extract_action=_prompt_core(str(external_root)).extract_action,
    )
    pending_jobs = tuple(job for job in jobs if job.job_index not in reused_rows)
    workers = int(_nested(config, "parallel", "episode_workers_per_seed"))
    started = time.perf_counter()
    jobs_dir = phase_dir / "sampling" / "jobs"
    for index, row in sorted(reused_rows.items()):
        save_job_result(jobs_dir / f"{index:04d}.json", row)
    fresh_rows = _ordered_map(
        _sampling_payloads(
            pending_jobs, wire=wire, config=config, external_root=external_root,
            data_root=data_root,
        ),
        _run_sampling_job,
        workers=workers,
        index_field="job_index",
        on_result=lambda row: save_job_result(
            jobs_dir / f"{int(row['job_index']):04d}.json", row
        ),
    )
    rows = sorted(
        [*reused_rows.values(), *fresh_rows],
        key=lambda row: int(row["job_index"]),
    )
    provider_calls = [dict(event) for row in rows for event in row.get("provider_calls", [])]
    if any(row.get("failure_kind") for row in rows):
        _write_jsonl(phase_dir / "provider_calls.jsonl", provider_calls)
    _fail_if_episode_boundary_failed(rows, stage="sampling")
    corpus = validate_completed_corpus(jobs, rows)
    _write_jsonl(phase_dir / "sampling" / "corpus.jsonl", corpus)
    metadata = corpus_metadata(
        manifest=train,
        run_seed=wire.run_seed,
        rows=corpus,
        label_sha256=report.label_sha256,
    )
    _write_json(phase_dir / "sampling" / "corpus_metadata.json", metadata)
    library = extract_skill_library(
        corpus,
        output_dir=phase_dir / "method_state" / "library",
        external_root=external_root,
        encoder_model=str(_nested(config, "extraction", "embedding_model")),
        extraction_seed=42,
        q_iterations=500,
        gamma=0.95,
        lambda_=0.9,
        alpha=0.05,
    )
    all_rows = list(corpus)
    episodes = _persist_episode_evidence(
        phase_dir, all_rows, phase=wire.phase, run_seed=wire.run_seed,
        artifact_digest="",
    )
    wall_time_ms = max(0, int((time.perf_counter() - started) * 1000))
    usage = _usage(
        all_rows,
        embedding_calls=int(library.metrics["embedding_calls"]),
        wall_time_ms=wall_time_ms,
    )
    usage.save(phase_dir / "usage.json")
    metrics = {
        **library.metrics,
        "sampling_episode_count": len(corpus),
        "expected_sampling_episode_count": len(train.tasks) * 6,
        "sample_seed_algorithm": metadata["sample_seed_algorithm"],
        "label_coverage": report.to_dict(),
        "resume": resume_evidence,
        "retrieval_count": sum(len(row.get("retrieval", [])) for row in all_rows),
        "retrieved_skill_count": sum(
            sum(int(item.get("retrieved_skill_count", 0)) for item in row.get("retrieval", []))
            for row in all_rows
        ),
    }
    return {
        "episodes": len(episodes),
        "sampling_episodes": len(corpus),
        "train_summary": metrics,
        "persistent_artifact_files": sorted(library.files),
        "corpus_digest": metadata["corpus_digest"],
        "label_sha256": report.label_sha256,
        "resumed_sampling_jobs": int(resume_evidence["reused_sampling_jobs"]),
        "executed_sampling_jobs": len(pending_jobs),
    }


def _evaluate(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    manifest: TaskManifestSet,
    data_root: Path,
    external_root: Path,
    phase_dir: Path,
) -> dict[str, Any]:
    if not wire.frozen_artifact_path:
        raise ValueError(f"B4 {wire.phase} requires a frozen artifact")
    frozen = FrozenArtifact.load(wire.frozen_artifact_path)
    if frozen.method_id != METHOD_ID:
        raise ValueError("frozen artifact belongs to a different baseline")
    if frozen.source_train_manifest_hash != wire.identity["train_manifest_digest"]:
        raise ValueError("frozen B4 artifact does not match the Train manifest")
    if frozen.source_validation_manifest_hash is not None:
        raise ValueError("B4 frozen artifact illegally binds Validation data")
    if frozen.digest != wire.identity.get("frozen_artifact_digest"):
        raise ValueError("frozen B4 artifact digest differs from worker identity")
    before = digest_directory(frozen.root)
    workers = int(
        _nested(config, "parallel", "test_workers_per_seed")
        if wire.phase in {"smoke_test", "test"}
        else _nested(config, "parallel", "episode_workers_per_seed")
    )
    started = time.perf_counter()
    rows = _ordered_map(
        _inference_payloads(
            manifest.tasks, wire=wire, config=config, external_root=external_root,
            data_root=data_root, frozen_root=frozen.root,
        ),
        _run_inference_job,
        workers=workers,
        index_field="manifest_index",
    )
    calls = [dict(event) for row in rows for event in row.get("provider_calls", [])]
    if any(row.get("failure_kind") for row in rows):
        _write_jsonl(phase_dir / "provider_calls.jsonl", calls)
    _fail_if_episode_boundary_failed(rows, stage=wire.phase)
    episodes = _persist_episode_evidence(
        phase_dir, rows, phase=wire.phase, run_seed=wire.run_seed,
        artifact_digest=frozen.digest,
    )
    after = digest_directory(frozen.root)
    if before != after or before != frozen.digest:
        raise WorkerPhaseError("B4 frozen retrieval artifact changed during evaluation")
    usage = _usage(
        rows,
        embedding_calls=sum(int(row.get("embedding_calls", 0)) for row in rows),
        wall_time_ms=max(0, int((time.perf_counter() - started) * 1000)),
    )
    usage.save(phase_dir / "usage.json")
    return {
        "episodes": len(episodes),
        "rows": len(rows),
        "official_successes": sum(int(row["official_success"]) for row in rows),
        "retrieval_count": sum(len(row.get("retrieval", [])) for row in rows),
        "retrieved_skill_count": sum(
            sum(int(item.get("retrieved_skill_count", 0)) for item in row.get("retrieval", []))
            for row in rows
        ),
        "frozen_unchanged": True,
        "frozen_digest_before": before,
        "frozen_digest_after": after,
    }


def _execute(wire: WorkerWire, wire_path: Path) -> dict[str, Any]:
    if wire.phase not in ALLOWED_PHASES:
        raise ValueError(f"unsupported B4 worker phase: {wire.phase}")
    phase_dir = Path(wire.output_dir).resolve() / wire.phase
    if wire_path.resolve() != (phase_dir / "worker_wire.json").resolve():
        raise ValueError("B4 worker wire is not phase-local")
    unexpected = sorted(path.name for path in phase_dir.iterdir() if path.name != "worker_wire.json")
    if unexpected:
        raise FileExistsError(f"B4 phase directory is not fresh: {unexpected}")
    config = _load_config(wire.config_path)
    expected_python = resolve_formal_python(
        REPO_ROOT, str(config.get("worker_python", ""))
    )
    python_runtime = verify_runtime_python(
        expected_python=expected_python,
        require_venv=True,
        method=METHOD_ID,
        expected_distributions=worker_expected_distributions(METHOD_ID),
        expected_python_major_minor="3.9",
    )
    model = _validate_config(wire, config)
    external_root = _external_root(wire)
    source = _verify_source(external_root)
    if source["runtime_tree"]["sha256"] != wire.identity.get("external_source_digest"):
        raise ValueError("verified SkillGen runtime tree differs from worker identity")
    train, test, data_root = _load_manifests(wire, config)
    label_report = None
    labels: dict[str, SkillGenLabel] = {}
    if wire.phase in {"train", "smoke"}:
        if not wire.supervision_path:
            raise SkillGenLabelAuthorityError(
                "B4 Train/smoke requires an explicit SkillGen Train supervision file"
            )
        supervision_path = Path(wire.supervision_path).expanduser().resolve(strict=True)
        supervision_digest = hashlib.sha256(supervision_path.read_bytes()).hexdigest()
        if supervision_digest != wire.identity.get("supervision_digest"):
            raise SkillGenLabelAuthorityError(
                "B4 supervision file digest differs from the immutable worker identity"
            )
        if train is None:
            raise AssertionError("B4 Train manifest unexpectedly missing")
        label_report, labels = analyze_label_coverage(train, supervision_path)
        if not label_report.passed:
            require_complete_train_coverage(train, supervision_path)
    model.require_api_key()
    if wire.phase in {"train", "smoke"}:
        if train is None:
            raise AssertionError("B4 Train manifest unexpectedly missing")
        result = _train_or_smoke(
            wire=wire, config=config, train=train, data_root=data_root,
            external_root=external_root, phase_dir=phase_dir,
            label_report=label_report, labels=labels,
        )
    elif wire.phase == "train_eval":
        if train is None:
            raise AssertionError("B4 Train-eval manifest unexpectedly missing")
        result = _evaluate(
            wire=wire, config=config, manifest=train, data_root=data_root,
            external_root=external_root, phase_dir=phase_dir,
        )
    else:
        if test is None:
            raise AssertionError("B4 Test manifest unexpectedly missing")
        result = _evaluate(
            wire=wire, config=config, manifest=test, data_root=data_root,
            external_root=external_root, phase_dir=phase_dir,
        )
    return {**result, "source": source, "python_runtime": python_runtime}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", required=True)
    args = parser.parse_args(argv)
    wire_path = Path(args.wire).resolve()
    wire: WorkerWire | None = None
    try:
        wire = WorkerWire.from_dict(json.loads(wire_path.read_text(encoding="utf-8")))
        result = _execute(wire, wire_path)
        write_worker_result(wire, result, passed=True)
        return 0
    except BaseException as exc:
        if wire is None:
            print(_safe_error(exc), file=sys.stderr)
            return 1
        evidence = dict(getattr(exc, "evidence", {}) or {})
        report = getattr(exc, "report", None)
        if report is not None:
            evidence["label_coverage"] = report.to_dict()
        payload = {
            **evidence,
            "error_type": type(exc).__name__,
            "error": _safe_error(exc, str(wire.model.get("api_key_env", "MODEL_API_KEY"))),
            "failure_kind": str(getattr(exc, "failure_kind", "protocol_failure")),
        }
        try:
            write_worker_result(wire, payload, passed=False)
        except FileExistsError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
