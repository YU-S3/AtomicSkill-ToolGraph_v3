"""Isolated SkillOpt worker for formal Train120 + Validation24 experiments.

The worker is deliberately a thin, fail-closed boundary around the pinned
upstream package. ``train`` constructs the upstream :class:`ReflACTTrainer`;
``smoke`` and ``train_eval`` only execute the common ALFWorld adapter and must
never construct training/evolution state.

Every successful phase proves the controller identity (config, manifests,
model, source tree and run id), records every provider call, and writes its
result exactly once to the path declared by WorkerWire v2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.baselines.bootstrap_external import (
    load_lock,
    verify_key_files,
    verify_runtime_tree,
)
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.formal_validation import (
    verify_final_evaluation_bijection,
    verify_formal_manifest,
)
from experiments.baselines.common.freeze import FrozenArtifact
from experiments.baselines.common.manifest import (
    TaskManifestSet,
    sha256_json,
    verify_disjoint,
)
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.subprocess_worker import (
    WorkerWire,
    write_worker_result,
)
from experiments.baselines.common.trace import load_episodes
from experiments.baselines.common.usage import RoleUsage, UsageSnapshot

from .provider_observer import (
    ProviderCallExhausted,
    ProviderCallObserver,
    install_provider_observer,
    uninstall_provider_observer,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ID = "b3_skillopt"
_SKILL_INIT_REL = "skillopt/envs/alfworld/skills/initial.md"
_ALLOWED_PHASES = frozenset({"train", "smoke", "train_eval", "test"})
_EXPECTED_METHOD_SETTINGS: dict[tuple[str, ...], Any] = {
    ("protocol_profile",): "formal_v2",
    ("train", "num_epochs"): 4,
    ("train", "train_size"): 120,
    ("train", "batch_size"): 40,
    ("train", "accumulation"): 1,
    ("gradient", "minibatch_size"): 8,
    ("gradient", "merge_batch_size"): 8,
    ("gradient", "analyst_workers"): 16,
    ("gradient", "failure_only"): False,
    ("optimizer", "learning_rate"): 4,
    ("optimizer", "min_learning_rate"): 2,
    ("optimizer", "lr_scheduler"): "cosine",
    ("optimizer", "lr_control_mode"): "fixed",
    ("optimizer", "skill_update_mode"): "patch",
    ("optimizer", "use_slow_update"): True,
    ("optimizer", "slow_update_samples"): 20,
    ("optimizer", "longitudinal_pair_policy"): "mixed",
    ("optimizer", "use_meta_skill"): True,
    ("optimizer", "use_skill_aware_reflection"): False,
    ("optimizer", "slow_update_gate_with_selection"): True,
    ("evaluation", "use_gate"): True,
    ("evaluation", "gate_metric"): "hard",
    ("evaluation", "sel_env_num"): 24,
    ("evaluation", "eval_test"): False,
    ("env", "name"): "alfworld",
    ("env", "max_steps"): 100,
    ("env", "max_completion_tokens"): 16384,
    ("env", "workers"): 16,
    ("env", "max_api_workers"): 16,
    ("provider_transport", "sdk_max_retries"): 0,
    ("provider_transport", "application_retry_limit"): 5,
    ("provider_transport", "retry_delays_seconds"): [2, 5, 10, 20],
    ("smoke", "task_count"): 2,
    ("smoke", "max_steps"): 2,
}


class WorkerPhaseError(RuntimeError):
    """Phase failure carrying non-secret evidence for worker_result.json."""

    def __init__(self, message: str, **evidence: Any) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _load_method_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"resolved B3 config is missing: {target}")
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resolved B3 config root must be a mapping")
    return payload


def _require_file(path: str | Path, what: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{what} is missing: {resolved}")
    return resolved


def _phase_dir(wire: WorkerWire) -> Path:
    return Path(wire.output_dir).resolve() / wire.phase


def _validate_wire_location(wire: WorkerWire, wire_path: Path) -> None:
    phase_dir = _phase_dir(wire)
    if wire_path.resolve() != (phase_dir / "worker_wire.json").resolve():
        raise ValueError(
            "worker wire must be phase-local: expected "
            f"{phase_dir / 'worker_wire.json'}, got {wire_path.resolve()}"
        )
    if Path(wire.result_path).resolve() != (phase_dir / "worker_result.json").resolve():
        raise ValueError("worker result path is not bound to its declared phase")


def _assert_fresh_phase(wire: WorkerWire) -> None:
    phase_dir = _phase_dir(wire)
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"worker phase directory is missing: {phase_dir}")
    unexpected = sorted(
        path.name for path in phase_dir.iterdir() if path.name != "worker_wire.json"
    )
    if unexpected:
        raise FileExistsError(
            f"worker phase directory is not fresh ({phase_dir}): {unexpected}"
        )


def _nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"resolved config is missing {'.'.join(path)}")
        current = current[key]
    return current


def _validate_config_identity(
    wire: WorkerWire,
    config: dict[str, Any],
    model: ModelConfig,
) -> str:
    expected_path = (Path(wire.output_dir).resolve() / "config_resolved.json").resolve()
    actual_path = Path(wire.config_path).resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"worker config_path must name {expected_path}, got {actual_path}"
        )
    actual_digest = sha256_json(config)
    if wire.identity.get("config_digest") != actual_digest:
        raise ValueError(
            "resolved config digest does not match worker identity: "
            f"expected {wire.identity.get('config_digest')}, got {actual_digest}"
        )
    if config.get("method") != METHOD_ID:
        raise ValueError("resolved config method is not b3_skillopt")
    if int(config.get("run_seed", -1)) != wire.run_seed:
        raise ValueError("resolved config run_seed does not match worker wire")
    if int(_nested(config, ("train", "seed"))) != wire.run_seed:
        raise ValueError("resolved train.seed does not match worker wire")
    configured_model = ModelConfig.from_mapping(dict(config.get("model") or {}))
    configured_model.validate_formal_identity()
    if configured_model != model:
        raise ValueError("resolved config model identity does not match worker wire")
    if int(config.get("max_environment_actions", -1)) != int(
        _nested(config, ("env", "max_steps"))
    ):
        raise ValueError("resolved max_environment_actions must equal env.max_steps")
    for path, expected in _EXPECTED_METHOD_SETTINGS.items():
        actual = _nested(config, path)
        if actual != expected:
            raise ValueError(
                f"formal B3 setting {'.'.join(path)} must be {expected!r}, "
                f"got {actual!r}"
            )
    _reject_embedded_secrets(config)
    return actual_digest


def _reject_embedded_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if normalized != "api_key_env" and (
                normalized in {"api_key", "model_api_key", "secret", "password"}
                or normalized.endswith("_api_key")
            ):
                raise ValueError(
                    f"resolved config must not embed credentials: {'.'.join(path + (key,))}"
                )
            _reject_embedded_secrets(child, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_embedded_secrets(child, path + (str(index),))


def _external_root(wire: WorkerWire) -> Path:
    raw = wire.external_skillopt_root or str(REPO_ROOT / ".external" / "skillopt")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pinned SkillOpt source is missing: {root}")
    return root


def _assert_module_origin(module_name: str, external_root: Path) -> str:
    module = importlib.import_module(module_name)
    raw_origin = getattr(module, "__file__", None)
    if not raw_origin:
        raise RuntimeError(f"imported module {module_name!r} has no file origin")
    origin = Path(raw_origin).resolve()
    try:
        origin.relative_to(external_root)
    except ValueError as exc:
        raise RuntimeError(
            f"module {module_name!r} was imported outside the pinned SkillOpt tree: "
            f"{origin}"
        ) from exc
    return str(origin)


def _verify_skillopt_source(
    wire: WorkerWire,
    lock: dict[str, Any],
) -> dict[str, Any]:
    root = _external_root(wire)
    verified_files = verify_key_files(root, "skillopt", lock)
    runtime_tree = verify_runtime_tree(root, "skillopt", lock)
    expected_version = str((lock.get("skillopt") or {}).get("version") or "")
    installed_version = importlib.metadata.version("skillopt")
    if not expected_version or installed_version != expected_version:
        raise RuntimeError(
            f"installed SkillOpt version mismatch: expected {expected_version!r}, "
            f"got {installed_version!r}"
        )
    origins = {
        name: _assert_module_origin(name, root)
        for name in (
            "skillopt",
            "skillopt.model",
            "skillopt.model.openai_compatible_backend",
        )
    }
    package = importlib.import_module("skillopt")
    package_paths = [Path(item).resolve() for item in getattr(package, "__path__", [])]
    expected_package = (root / "skillopt").resolve()
    if package_paths != [expected_package]:
        raise RuntimeError(
            "SkillOpt package search path is not isolated to the pinned source: "
            f"{package_paths}"
        )
    return {
        "repo": str((lock.get("skillopt") or {}).get("repo") or ""),
        "commit": str((lock.get("skillopt") or {}).get("commit") or ""),
        "declared_version": expected_version,
        "installed_version": installed_version,
        "root": str(root),
        "runtime_tree": runtime_tree,
        "verified_key_files": len(verified_files),
        "import_origins": origins,
    }


def _load_and_verify_manifests(
    wire: WorkerWire,
    config: dict[str, Any],
) -> tuple[TaskManifestSet, TaskManifestSet, TaskManifestSet | None, dict[str, Any]]:
    profile = str(config.get("protocol_profile", "pilot_v1"))
    train_path = _require_file(wire.manifest_path or "", "Train manifest")
    validation_path = _require_file(
        wire.validation_manifest_path or "", "Validation manifest"
    )
    if wire.phase == "test" and not wire.test_manifest_path:
        raise ValueError("B3 test requires the common Test manifest")
    if wire.phase != "test" and wire.test_manifest_path:
        raise ValueError("a Test manifest is accepted only by the B3 test phase")
    train = TaskManifestSet.load(train_path)
    validation = TaskManifestSet.load(validation_path)
    test = (
        TaskManifestSet.load(_require_file(wire.test_manifest_path, "Test manifest"))
        if wire.test_manifest_path else None
    )
    if wire.identity.get("train_manifest_digest") != train.digest:
        raise ValueError("Train manifest digest does not match worker identity")
    if wire.identity.get("validation_manifest_digest") != validation.digest:
        raise ValueError("Validation manifest digest does not match worker identity")
    if wire.phase == "train_eval" and (
        wire.identity.get("evaluation_manifest_digest") != train.digest
    ):
        raise ValueError("train_eval evaluation_manifest_digest must equal Train digest")
    if wire.phase == "test":
        if test is None:
            raise AssertionError("test manifest unexpectedly missing")
        if wire.identity.get("test_manifest_digest") != test.digest:
            raise ValueError("Test manifest digest does not match worker identity")
        if wire.identity.get("evaluation_manifest_digest") != test.digest:
            raise ValueError("test evaluation_manifest_digest must equal Test digest")
    verify_disjoint(*([train, validation, test] if test is not None else [train, validation]))
    data_root = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_root:
        raise RuntimeError("ALFWORLD_DATA is not set in the worker environment")
    receipts = {
        "train": verify_formal_manifest(
            train, alfworld_data=data_root, role="train", profile=profile
        ),
        "validation": verify_formal_manifest(
            validation, alfworld_data=data_root, role="validation", profile=profile
        ),
    }
    if test is not None:
        receipts["test"] = verify_formal_manifest(
            test, alfworld_data=data_root, role="test", profile=profile
        )
    return train, validation, test, receipts


def _verify_skill_init(
    *,
    external_root: Path,
    skill_init_rel: str | None,
    destination_dir: Path,
    lock: dict[str, Any],
) -> tuple[Path, str]:
    relative = str(skill_init_rel or _SKILL_INIT_REL)
    if relative != _SKILL_INIT_REL:
        raise ValueError(
            f"SkillOpt seed path must be {_SKILL_INIT_REL!r}, got {relative!r}"
        )
    source = _require_file(external_root / relative, "upstream SkillOpt initial.md")
    expected = str(
        (lock.get("skillopt") or {}).get("key_files", {}).get(relative, "")
    )
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise RuntimeError(
            f"upstream initial.md hash mismatch: expected {expected or '<missing>'}, "
            f"got {actual}"
        )
    destination_dir.mkdir(parents=True, exist_ok=False)
    target = destination_dir / "initial.md"
    shutil.copyfile(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != actual:
        raise RuntimeError("copied SkillOpt seed failed its post-copy digest check")
    return target, actual


def _flat_train_cfg(
    *,
    config: dict[str, Any],
    model: ModelConfig,
    run_seed: int,
    out_root: Path,
    skill_init_path: str,
    train_size: int,
    selection_size: int,
) -> dict[str, Any]:
    """Build the exact flat configuration consumed by upstream SkillOpt."""

    train = dict(config["train"])
    gradient = dict(config["gradient"])
    optimizer = dict(config["optimizer"])
    evaluation = dict(config["evaluation"])
    env = dict(config["env"])
    return {
        "out_root": str(out_root),
        "env": "alfworld",
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": model.model,
        "target_model": model.model,
        "reasoning_effort": model.reasoning_effort,
        "skill_init": skill_init_path,
        "batch_size": int(train["batch_size"]),
        "num_epochs": int(train["num_epochs"]),
        "accumulation": int(train["accumulation"]),
        "seed": int(run_seed),
        "merge_batch_size": int(gradient["merge_batch_size"]),
        "analyst_workers": int(gradient["analyst_workers"]),
        "failure_only": bool(gradient["failure_only"]),
        "minibatch_size": int(gradient["minibatch_size"]),
        "edit_budget": int(optimizer["learning_rate"]),
        "min_edit_budget": int(optimizer["min_learning_rate"]),
        "lr_scheduler": str(optimizer["lr_scheduler"]),
        "lr_control_mode": str(optimizer["lr_control_mode"]),
        "skill_update_mode": str(optimizer["skill_update_mode"]),
        "longitudinal_pair_policy": str(optimizer["longitudinal_pair_policy"]),
        "use_slow_update": bool(optimizer["use_slow_update"]),
        "slow_update_samples": int(optimizer["slow_update_samples"]),
        "slow_update_gate_with_selection": bool(
            optimizer["slow_update_gate_with_selection"]
        ),
        "use_meta_skill": bool(optimizer["use_meta_skill"]),
        "use_skill_aware_reflection": bool(
            optimizer["use_skill_aware_reflection"]
        ),
        "skill_aware_appendix_source": "both",
        "skill_aware_consolidate_threshold": 0,
        "use_gate": bool(evaluation["use_gate"]),
        "gate_metric": str(evaluation["gate_metric"]),
        "gate_mixed_weight": 0.5,
        "use_semantic_density": False,
        "semantic_density_weight": 0.05,
        "leading_words": None,
        "sel_env_num": int(selection_size),
        "test_env_num": 0,
        "eval_test": bool(evaluation["eval_test"]),
        "rewrite_reasoning_effort": model.reasoning_effort,
        "rewrite_max_completion_tokens": 64000,
        "train_size": int(train_size),
        "max_steps": int(env["max_steps"]),
        "workers": int(env["workers"]),
        "max_api_workers": int(env["max_api_workers"]),
        "max_completion_tokens": int(env["max_completion_tokens"]),
    }


def _configure_model(
    model: ModelConfig,
    *,
    sdk_max_retries: int,
) -> None:
    import skillopt.model as skillopt_model

    if int(sdk_max_retries) != 0:
        raise ValueError("formal B3 requires zero SDK-internal provider retries")
    # The adapter owns the retry loop so every attempt has a durable, safe
    # classification. Leaving the SDK default enabled creates an invisible
    # nested retry layer and was the reason the failed campaign could only
    # report one opaque RuntimeError after several minutes.
    skillopt_model.set_backend("openai_compatible")
    skillopt_model.configure_openai_compatible(
        base_url=model.base_url,
        api_key=model.require_api_key(),
        model=model.model,
        max_tokens=32768,
        timeout_seconds=300,
    )
    skillopt_model.set_reasoning_effort(model.reasoning_effort)


def _adapter_kwargs(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    phase: str,
    max_steps: int | None = None,
    artifact_digest_override: str | None = None,
) -> dict[str, Any]:
    env = dict(config["env"])
    gradient = dict(config["gradient"])
    optimizer = dict(config["optimizer"])
    kwargs: dict[str, Any] = {
        "train_manifest_path": wire.manifest_path,
        "validation_manifest_path": wire.validation_manifest_path,
        "test_manifest_path": wire.test_manifest_path,
        "alfworld_data": os.environ["ALFWORLD_DATA"],
        "max_steps": int(max_steps if max_steps is not None else env["max_steps"]),
        "workers": int(env["workers"]),
        "max_api_workers": int(env["max_api_workers"]),
        "analyst_workers": int(gradient["analyst_workers"]),
        "failure_only": False,
        "minibatch_size": int(gradient["minibatch_size"]),
        "edit_budget": int(optimizer["learning_rate"]),
        "max_completion_tokens": int(env["max_completion_tokens"]),
        "seed": int(wire.run_seed),
        "phase": phase,
    }
    from .common_alfworld_adapter import CommonALFWorldSkillOptAdapter

    parameters = inspect.signature(CommonALFWorldSkillOptAdapter).parameters
    optional = {
        "run_id": wire.run_id,
        "identity": dict(wire.identity),
    }
    if artifact_digest_override is not None:
        optional["artifact_digest_override"] = artifact_digest_override
    for name, value in optional.items():
        if name in parameters:
            kwargs[name] = value
        else:
            raise RuntimeError(
                f"common SkillOpt adapter lacks required production argument {name!r}"
            )
    return kwargs


def _build_adapter(**kwargs: Any) -> Any:
    from .common_alfworld_adapter import CommonALFWorldSkillOptAdapter

    return CommonALFWorldSkillOptAdapter(**kwargs)


def _guard_adapter_rollout(
    adapter: Any,
    observer: ProviderCallObserver,
) -> None:
    """Prevent infrastructure failures becoming ordinary hard=0 samples."""

    original = adapter.rollout

    def guarded(
        env_manager: Any,
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        before = len(observer.events())
        rows = original(env_manager, skill_content, out_dir, **kwargs)
        new_events = observer.events()[before:]
        failed = [event for event in new_events if event.get("status") == "failed"]
        if failed:
            raise WorkerPhaseError(
                "provider infrastructure failed during SkillOpt rollout",
                failed_provider_calls=len(failed),
            )
        sidecar = Path(out_dir) / "common_episodes.jsonl"
        if not sidecar.is_file():
            raise RuntimeError(f"SkillOpt rollout produced no Common sidecar: {sidecar}")
        episodes = load_episodes(sidecar)
        infra = [episode.task_id for episode in episodes if episode.infrastructure_failure]
        if infra:
            raise WorkerPhaseError(
                "SkillOpt rollout contained infrastructure failures",
                infrastructure_failed_task_ids=infra,
            )
        if len(rows) != len(episodes):
            raise RuntimeError(
                "SkillOpt rollout row count does not match Common episode evidence"
            )
        return rows

    adapter.rollout = guarded


def _collect_episodes(root: Path) -> list[Any]:
    episodes: list[Any] = []
    for sidecar in sorted(root.rglob("common_episodes.jsonl")):
        episodes.extend(load_episodes(sidecar))
    if not episodes:
        raise FileNotFoundError(f"no Common episode evidence exists under {root}")
    return episodes


def _validate_episode_action_evidence(root: Path) -> None:
    episode_sidecars = sorted(root.rglob("common_episodes.jsonl"))
    if not episode_sidecars:
        raise FileNotFoundError(f"no Common episode sidecar exists under {root}")
    for episode_sidecar in episode_sidecars:
        episodes = load_episodes(episode_sidecar)
        action_path = episode_sidecar.with_name("common_environment_actions.jsonl")
        if not action_path.is_file():
            raise FileNotFoundError(
                f"episode sidecar has no paired environment actions: {action_path}"
            )
        raw_rows = [
            json.loads(line)
            for line in action_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_task: dict[str, dict[int, str]] = {}
        for row in raw_rows:
            task_id = str(row.get("episode_task_id", ""))
            index = int(row.get("step_index", -1))
            action = row.get("action")
            if not task_id or index < 0 or not isinstance(action, str) or not action.strip():
                raise ValueError(f"invalid environment-action evidence in {action_path}")
            task_rows = by_task.setdefault(task_id, {})
            if index in task_rows:
                raise ValueError(
                    f"duplicate environment action {task_id!r} step {index}"
                )
            task_rows[index] = action
        for episode in episodes:
            steps = by_task.get(episode.task_id, {})
            if sorted(steps) != list(range(int(episode.environment_actions))):
                raise ValueError(
                    f"environment actions for {episode.task_id!r} are not contiguous "
                    "or disagree with the episode record"
                )


def _validate_training_episodes(
    episodes: list[Any],
    *,
    train: TaskManifestSet,
    validation: TaskManifestSet,
    wire: WorkerWire,
    max_actions: int,
) -> dict[str, int]:
    authorities = {
        "train": {task.task_id: task for task in train.tasks},
        "validation": {task.task_id: task for task in validation.tasks},
    }
    counts = Counter(episode.phase for episode in episodes)
    if not counts["train"] or not counts["validation"]:
        raise RuntimeError("SkillOpt did not persist both Train and Validation rollouts")
    for episode in episodes:
        if episode.phase not in authorities:
            raise ValueError(f"unexpected training episode phase: {episode.phase!r}")
        task = authorities[episode.phase].get(episode.task_id)
        if task is None:
            raise ValueError(
                f"episode {episode.task_id!r} is outside its {episode.phase} manifest"
            )
        if (
            episode.method != METHOD_ID
            or episode.run_seed != wire.run_seed
            or episode.manifest_index != task.index
            or episode.task_type != task.task_type
            or episode.gamefile != task.gamefile_rel
            or episode.gamefile_hash != task.gamefile_sha256
        ):
            raise ValueError(
                f"episode {episode.task_id!r} disagrees with manifest/run identity"
            )
        if episode.infrastructure_failure:
            raise WorkerPhaseError(
                "training episode contains an infrastructure failure",
                infrastructure_failed_task_ids=[episode.task_id],
            )
        if episode.environment_actions <= 0 or episode.environment_actions > max_actions:
            raise ValueError(
                f"episode {episode.task_id!r} has invalid environment action count "
                f"{episode.environment_actions}"
            )
        if episode.target_llm_calls <= 0:
            raise ValueError(
                f"episode {episode.task_id!r} has no target-provider usage evidence"
            )
    return {
        "total": len(episodes),
        "train": counts["train"],
        "validation": counts["validation"],
    }


def _provider_usage(
    observer: ProviderCallObserver,
    *,
    wire: WorkerWire,
    wall_time_ms: int,
    allow_failed: bool = False,
) -> UsageSnapshot:
    events = observer.events()
    if not events:
        raise RuntimeError("phase produced no provider-call evidence")
    seen_ids: set[str] = set()
    target = RoleUsage()
    evolution = RoleUsage()
    per_stage: dict[str, RoleUsage] = {}
    failures: list[dict[str, Any]] = []
    for event in events:
        if (
            event.get("event") != "provider_call"
            or event.get("method") != wire.method
            or event.get("phase") != wire.phase
            or event.get("run_id") != wire.run_id
            or event.get("model") != str(wire.model.get("model", ""))
            or event.get("reasoning_effort")
            != str(wire.model.get("reasoning_effort", ""))
        ):
            raise ValueError("provider-call evidence has the wrong run identity")
        call_id = str(event.get("call_id", ""))
        if not call_id or call_id in seen_ids:
            raise ValueError("provider-call evidence has an empty or duplicate call_id")
        seen_ids.add(call_id)
        retry_evidence_present = "application_attempts" in event
        application_attempts = int(event.get("application_attempts", 1))
        sdk_boundary_attempts = int(
            event.get("sdk_boundary_attempts", application_attempts)
        )
        requested_retry_limit = int(event.get("requested_retry_limit", 1))
        effective_retry_limit = int(
            event.get("retry_limit", requested_retry_limit)
        )
        raw_failure_counts = event.get("failure_code_counts", {})
        if not isinstance(raw_failure_counts, dict):
            raise ValueError("provider-call failure_code_counts is not a mapping")
        failure_counts = {
            str(code): int(count) for code, count in raw_failure_counts.items()
        }
        if (
            application_attempts <= 0
            or sdk_boundary_attempts < 0
            or requested_retry_limit <= 0
            or effective_retry_limit < application_attempts
            or any(not code or count <= 0 for code, count in failure_counts.items())
        ):
            raise ValueError("provider-call retry evidence is invalid")
        if event.get("status") == "failed":
            if retry_evidence_present and (
                not failure_counts or bool(event.get("recovered", False))
            ):
                raise ValueError("failed provider call has invalid failure evidence")
            failures.append(event)
            continue
        if event.get("status") != "succeeded":
            raise ValueError("provider-call evidence has an invalid status")
        if bool(event.get("recovered", False)) != bool(failure_counts):
            raise ValueError("provider-call recovered flag disagrees with retry evidence")
        role = str(event.get("role", ""))
        if role not in {"target", "optimizer"}:
            raise ValueError(f"provider-call evidence has an invalid role: {role!r}")
        stage = str(event.get("stage", "")).strip()
        if not stage:
            raise ValueError("provider-call evidence has no stage")
        reasoning_status = str(
            event.get("reasoning_tokens_status", "unavailable")
        )
        if reasoning_status not in {"reported", "unavailable"}:
            raise ValueError("provider-call evidence has an invalid reasoning status")
        raw_reasoning = event.get("reasoning_tokens")
        if reasoning_status == "reported":
            if raw_reasoning is None:
                raise ValueError("provider omitted reported reasoning-token usage")
            reasoning_tokens = int(raw_reasoning)
        else:
            if raw_reasoning not in {None, 0}:
                raise ValueError(
                    "provider supplied reasoning tokens while marking them unavailable"
                )
            reasoning_tokens = 0
        usage = RoleUsage(
            calls=1,
            prompt_tokens=int(event.get("prompt_tokens", 0)),
            completion_tokens=int(event.get("completion_tokens", 0)),
            reasoning_tokens=reasoning_tokens,
        )
        if usage.prompt_tokens < 0 or usage.completion_tokens <= 0:
            raise ValueError("provider-call evidence has invalid token usage")
        if usage.reasoning_tokens < 0 or (
            usage.reasoning_tokens > usage.completion_tokens
        ):
            raise ValueError("provider-call evidence has invalid reasoning-token usage")
        total = int(event.get("total_tokens", 0))
        if total != usage.prompt_tokens + usage.completion_tokens:
            raise ValueError(
                "provider total_tokens does not equal prompt_tokens + "
                "completion_tokens"
            )
        (target if role == "target" else evolution).add(usage)
        per_stage.setdefault(stage, RoleUsage()).add(usage)
    if failures and not allow_failed:
        raise WorkerPhaseError(
            "provider infrastructure failure is present in phase evidence",
            failed_provider_calls=len(failures),
            failed_provider_error_types=sorted(
                {str(event.get("error_type", "ProviderError")) for event in failures}
            ),
        )
    if target.calls + evolution.calls <= 0:
        raise RuntimeError("phase has no successful provider-call usage")
    return UsageSnapshot(
        target=target,
        evolution=evolution,
        per_stage=per_stage,
        wall_time_ms=max(0, int(wall_time_ms)),
    )


def _reconcile_upstream_usage(
    usage: UsageSnapshot,
    upstream: dict[str, Any],
) -> None:
    observed_stages = {
        stage: {
            "calls": bucket.calls,
            "prompt_tokens": bucket.prompt_tokens,
            "completion_tokens": bucket.completion_tokens,
        }
        for stage, bucket in usage.per_stage.items()
    }
    upstream_stages = {
        str(stage): {
            "calls": int(values.get("calls", 0)),
            "prompt_tokens": int(values.get("prompt_tokens", 0)),
            "completion_tokens": int(values.get("completion_tokens", 0)),
        }
        for stage, values in dict(upstream or {}).items()
        if stage != "_total" and isinstance(values, dict)
    }
    if observed_stages != upstream_stages:
        raise RuntimeError(
            "provider-call audit does not reconcile with SkillOpt token tracker"
        )


def _provider_evidence_summary(observer: ProviderCallObserver) -> dict[str, Any]:
    events = observer.events()
    application_attempts = sum(
        int(event.get("application_attempts", 1)) for event in events
    )
    reasoning_by_role: dict[str, Counter[str]] = {}
    reasoning_by_stage: dict[str, Counter[str]] = {}
    for event in events:
        status = str(event.get("reasoning_tokens_status", "unavailable"))
        reasoning_by_role.setdefault(
            str(event.get("role", "")), Counter()
        )[status] += 1
        reasoning_by_stage.setdefault(
            str(event.get("stage", "")), Counter()
        )[status] += 1
    failure_codes: Counter[str] = Counter()
    for event in events:
        raw = event.get("failure_code_counts", {})
        if isinstance(raw, dict):
            failure_codes.update({
                str(code): int(count) for code, count in raw.items()
            })
    return {
        "calls": len(events),
        "application_attempts": application_attempts,
        "provider_retries": application_attempts - len(events),
        "sdk_boundary_attempts": sum(
            int(event.get("sdk_boundary_attempts", 1)) for event in events
        ),
        "recovered_calls": sum(
            bool(event.get("recovered", False)) for event in events
        ),
        "retry_failure_code_counts": dict(sorted(failure_codes.items())),
        "status_counts": dict(sorted(Counter(
            str(event.get("status", "")) for event in events
        ).items())),
        "role_counts": dict(sorted(Counter(
            str(event.get("role", "")) for event in events
        ).items())),
        "reasoning_token_status_counts": dict(sorted(Counter(
            str(event.get("reasoning_tokens_status", "unavailable"))
            for event in events
        ).items())),
        "reasoning_token_status_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(reasoning_by_role.items())
        },
        "reasoning_token_status_by_stage": {
            stage: dict(sorted(counts.items()))
            for stage, counts in sorted(reasoning_by_stage.items())
        },
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _write_usage(
    phase_dir: Path,
    observer: ProviderCallObserver,
    *,
    wire: WorkerWire,
    started: float,
    upstream: dict[str, Any] | None = None,
    forbid_evolution: bool = False,
    allow_failed: bool = False,
) -> UsageSnapshot:
    usage = _provider_usage(
        observer,
        wire=wire,
        wall_time_ms=int((time.monotonic() - started) * 1000),
        allow_failed=allow_failed,
    )
    if upstream is not None:
        _reconcile_upstream_usage(usage, upstream)
    if forbid_evolution and usage.evolution.calls:
        raise RuntimeError(
            f"{wire.phase} unexpectedly made evolution/optimizer provider calls"
        )
    _write_json_exclusive(phase_dir / "usage.json", usage.to_dict())
    return usage


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{what} is unreadable: {path}") from exc


def _method_metrics(
    *,
    train_out: Path,
    upstream_summary: dict[str, Any],
    initial_skill: str,
    best_skill: str,
    model_name: str,
    selection_episodes: int,
    count_skill_tokens: Callable[[str, str | None], int],
) -> dict[str, Any]:
    history_payload = _read_json(train_out / "history.json", "SkillOpt history")
    if not isinstance(history_payload, list):
        raise ValueError("SkillOpt history root must be a list")
    history = [dict(row) for row in history_payload if isinstance(row, dict)]
    if len(history) != int(upstream_summary.get("total_steps", -1)):
        raise RuntimeError("SkillOpt history length disagrees with trainer summary")
    patch_proposals = sum(int(row.get("n_patches", 0) or 0) for row in history)
    failure_patches = sum(
        int(row.get("n_failure_patches", 0) or 0) for row in history
    )
    success_patches = sum(
        int(row.get("n_success_patches", 0) or 0) for row in history
    )
    actions = Counter(str(row.get("action", "")) for row in history)
    gate_decisions = sum(
        count
        for action, count in actions.items()
        if action in {"accept", "accept_new_best", "force_accept", "reject"}
    )
    gate_accepts = sum(
        count
        for action, count in actions.items()
        if action in {"accept", "accept_new_best", "force_accept"}
    )
    best_step = upstream_summary.get("best_step")
    best_epoch: int | None = 0 if best_step == 0 else None
    for row in history:
        if row.get("step") == best_step:
            best_epoch = int(row.get("epoch", 0))
            break
    initial_tokens = int(count_skill_tokens(initial_skill, model_name))
    best_tokens = int(count_skill_tokens(best_skill, model_name))
    slow_results: list[dict[str, Any]] = []
    for path in sorted(train_out.rglob("slow_result.json")):
        payload = _read_json(path, "SkillOpt slow-update result")
        if isinstance(payload, dict):
            slow_results.append(payload)
    slow_actions = Counter(str(row.get("action", "")) for row in slow_results)
    return {
        "upstream_version_label": upstream_summary.get("version"),
        "baseline_selection_hard": upstream_summary.get("baseline_selection_hard"),
        "best_selection_hard": upstream_summary.get("best_selection_hard"),
        "final_selection_hard": upstream_summary.get("final_selection_hard"),
        "final_selection_soft": upstream_summary.get("final_selection_soft"),
        "best_step": best_step,
        "best_epoch": best_epoch,
        "total_steps": len(history),
        "patch_proposals": patch_proposals,
        "failure_patch_proposals": failure_patches,
        "success_patch_proposals": success_patches,
        "candidate_proposals": sum(bool(row.get("candidate_hash")) for row in history),
        "accepted_candidates": gate_accepts,
        "rejected_candidates": actions.get("reject", 0),
        "skipped_steps": sum(
            count for action, count in actions.items() if action.startswith("skip_")
        ),
        "gate_decisions": gate_decisions,
        "gate_acceptance_rate": (
            gate_accepts / gate_decisions if gate_decisions else None
        ),
        "gate_action_counts": dict(sorted(actions.items())),
        "selection_episodes": int(selection_episodes),
        "skill_initial_chars": len(initial_skill),
        "skill_best_chars": len(best_skill),
        "skill_char_growth": len(best_skill) - len(initial_skill),
        "skill_initial_tokens": initial_tokens,
        "skill_best_tokens": best_tokens,
        "skill_token_growth": best_tokens - initial_tokens,
        "skill_token_count_method": "skillopt.openai_compatible.count_tokens",
        "slow_update_attempts": len(slow_results),
        "slow_update_action_counts": dict(sorted(slow_actions.items())),
        "epoch_stats": upstream_summary.get("epoch_stats"),
        "total_wall_time_s": upstream_summary.get("total_wall_time_s"),
        "upstream_summary": upstream_summary,
    }


def _run_train(
    *,
    wire: WorkerWire,
    model: ModelConfig,
    config: dict[str, Any],
    lock: dict[str, Any],
    source: dict[str, Any],
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    observer: ProviderCallObserver,
) -> dict[str, Any]:
    train_out = _phase_dir(wire)
    skill_init_path, initial_digest = _verify_skill_init(
        external_root=Path(source["root"]),
        skill_init_rel=wire.skill_init_rel,
        destination_dir=train_out / "skill_init",
        lock=lock,
    )
    initial_skill = skill_init_path.read_text(encoding="utf-8")
    adapter = _build_adapter(**_adapter_kwargs(
        wire=wire,
        config=config,
        phase="train",
    ))
    _guard_adapter_rollout(adapter, observer)
    upstream_cfg = _flat_train_cfg(
        config=config,
        model=model,
        run_seed=wire.run_seed,
        out_root=train_out,
        skill_init_path=str(skill_init_path),
        train_size=len(train_manifest.tasks),
        selection_size=len(validation_manifest.tasks),
    )
    from skillopt.engine.trainer import ReflACTTrainer

    trainer_origin = _assert_module_origin(
        "skillopt.engine.trainer", Path(source["root"])
    )
    started = time.monotonic()
    trainer = ReflACTTrainer(upstream_cfg, adapter)
    upstream_summary = trainer.train()
    if not isinstance(upstream_summary, dict):
        raise RuntimeError("upstream SkillOpt trainer returned no summary mapping")
    from skillopt.model import get_token_summary

    token_summary = get_token_summary()
    episodes = _collect_episodes(train_out)
    _validate_episode_action_evidence(train_out)
    episode_counts = _validate_training_episodes(
        episodes,
        train=train_manifest,
        validation=validation_manifest,
        wire=wire,
        max_actions=int(config["env"]["max_steps"]),
    )
    best_skill_path = _require_file(
        train_out / "best_skill.md", "trained best_skill.md"
    )
    best_skill = best_skill_path.read_text(encoding="utf-8")
    if not best_skill.strip():
        raise RuntimeError("trained best_skill.md is empty")
    usage = _write_usage(
        train_out,
        observer,
        wire=wire,
        started=started,
        upstream=token_summary,
    )
    from skillopt.model.openai_compatible_backend import count_tokens

    metrics = _method_metrics(
        train_out=train_out,
        upstream_summary=upstream_summary,
        initial_skill=initial_skill,
        best_skill=best_skill,
        model_name=model.model,
        selection_episodes=episode_counts["validation"],
        count_skill_tokens=lambda text, model_name: count_tokens(
            text, model=model_name
        ),
    )
    if str(config.get("protocol_profile", "pilot_v1")) == "formal_v2" and (
        metrics["total_steps"] != 12
    ):
        raise RuntimeError(
            "formal SkillOpt run must produce exactly 12 main optimizer steps; "
            f"got {metrics['total_steps']}"
        )
    return {
        "output_dir": str(Path(wire.output_dir).resolve()),
        "train_summary": metrics,
        "usage": usage.to_dict(),
        "episodes": episode_counts,
        "best_skill_path": str(best_skill_path.resolve()),
        "best_skill_sha256": hashlib.sha256(best_skill_path.read_bytes()).hexdigest(),
        "initial_skill_sha256": initial_digest,
        "trainer_import_origin": trainer_origin,
        "provider_calls": len(observer.events()),
        "provider_evidence": _provider_evidence_summary(observer),
    }


def _run_smoke(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    lock: dict[str, Any],
    source: dict[str, Any],
    train_manifest: TaskManifestSet,
    observer: ProviderCallObserver,
) -> dict[str, Any]:
    smoke_out = _phase_dir(wire)
    source_seed = Path(source["root"]) / _SKILL_INIT_REL
    _require_file(source_seed, "upstream SkillOpt initial.md")
    seed_bytes = source_seed.read_bytes()
    seed_digest = hashlib.sha256(seed_bytes).hexdigest()
    expected = str(lock["skillopt"]["key_files"][_SKILL_INIT_REL])
    if seed_digest != expected:
        raise RuntimeError("SkillOpt smoke seed does not match source lock")
    skill = seed_bytes.decode("utf-8")
    smoke_cfg = dict(config.get("smoke") or {})
    task_count = int(smoke_cfg.get("task_count", 1))
    max_steps = int(smoke_cfg.get("max_steps", 2))
    if task_count != 2:
        raise ValueError("formal B3 smoke.task_count must be exactly 2")
    if max_steps <= 0 or max_steps > int(config["env"]["max_steps"]):
        raise ValueError("smoke.max_steps must be in [1, env.max_steps]")
    adapter = _build_adapter(**_adapter_kwargs(
        wire=wire,
        config=config,
        phase="smoke",
        max_steps=max_steps,
    ))
    _guard_adapter_rollout(adapter, observer)
    tasks = list(train_manifest.tasks[:task_count])
    env = adapter.build_smoke_env(seed=wire.run_seed, task_count=task_count)
    before = hashlib.sha256(source_seed.read_bytes()).hexdigest()
    started = time.monotonic()
    rows = adapter.rollout(env, skill, str(smoke_out / "rollout"))
    after = hashlib.sha256(source_seed.read_bytes()).hexdigest()
    if before != after or after != seed_digest:
        raise WorkerPhaseError(
            "SkillOpt seed changed during smoke",
            skill_digest_before=before,
            skill_digest_after=after,
        )
    episodes = _collect_episodes(smoke_out)
    _validate_episode_action_evidence(smoke_out)
    if len(rows) != task_count or len(episodes) != task_count:
        raise RuntimeError(
            f"smoke must produce exactly {task_count} rows and episodes"
        )
    for episode, task in zip(episodes, tasks, strict=True):
        if (
            episode.phase != "smoke"
            or episode.method != METHOD_ID
            or episode.run_seed != wire.run_seed
            or episode.task_id != task.task_id
            or episode.manifest_index != task.index
            or episode.gamefile != task.gamefile_rel
            or episode.gamefile_hash != task.gamefile_sha256
            or episode.artifact_digest_before != seed_digest
            or episode.artifact_digest_after != seed_digest
        ):
            raise ValueError(
                "smoke episode disagrees with its run/manifest/skill identity"
            )
        if episode.infrastructure_failure:
            raise WorkerPhaseError(
                "smoke episode contains an infrastructure failure",
                infrastructure_failed_task_ids=[episode.task_id],
            )
        if episode.environment_actions <= 0 or episode.environment_actions > max_steps:
            raise ValueError("smoke episode has an invalid environment-action count")
    from skillopt.model import get_token_summary

    usage = _write_usage(
        smoke_out,
        observer,
        wire=wire,
        started=started,
        upstream=get_token_summary(),
        forbid_evolution=True,
    )
    if usage.target.calls <= 0:
        raise RuntimeError("smoke made no successful target-provider call")
    return {
        "output_dir": str(Path(wire.output_dir).resolve()),
        "episodes": task_count,
        "rows": task_count,
        "usage": usage.to_dict(),
        "task_ids": [task.task_id for task in tasks],
        "task_selection": f"Train manifest indexes 0..{task_count - 1}",
        "official_successes": sum(
            int(episode.official_success) for episode in episodes
        ),
        "success_required": False,
        "max_environment_actions": max_steps,
        "skill_unchanged": True,
        "skill_digest_before": before,
        "skill_digest_after": after,
        "provider_calls": len(observer.events()),
        "provider_evidence": _provider_evidence_summary(observer),
    }


def _run_frozen_evaluation(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet,
    test_manifest: TaskManifestSet | None,
    observer: ProviderCallObserver,
) -> dict[str, Any]:
    """Read-only replay of frozen best_skill (no Trainer import)."""

    if wire.phase == "train_eval":
        evaluation_manifest = train_manifest
        role = "train"
        build_env_name = "build_train_evaluation_env"
        cardinality_name = "Train"
    elif wire.phase == "test":
        if test_manifest is None:
            raise ValueError("test phase has no Test manifest")
        evaluation_manifest = test_manifest
        role = "test"
        build_env_name = "build_test_evaluation_env"
        cardinality_name = "Test"
    else:
        raise ValueError(f"unsupported frozen evaluation phase: {wire.phase!r}")

    frozen_root = Path(wire.frozen_artifact_path or "")
    frozen = FrozenArtifact.load(frozen_root)
    if frozen.method_id != METHOD_ID:
        raise ValueError("frozen artifact belongs to a different method")
    if frozen.source_train_manifest_hash != train_manifest.digest:
        raise ValueError("frozen artifact does not match Train manifest")
    if frozen.source_validation_manifest_hash != validation_manifest.digest:
        raise ValueError("frozen artifact does not match Validation manifest")
    declared = wire.identity.get("frozen_artifact_digest")
    if not declared or declared != frozen.digest:
        raise ValueError("frozen artifact digest does not match worker identity")
    best_skill_path = _require_file(frozen.root / "best_skill.md", "frozen best_skill.md")
    best_skill = best_skill_path.read_text(encoding="utf-8")
    if not best_skill.strip():
        raise RuntimeError("frozen best_skill.md is empty")
    before = digest_directory(frozen.root)
    evaluation_out = _phase_dir(wire)
    adapter = _build_adapter(**_adapter_kwargs(
        wire=wire,
        config=config,
        phase=wire.phase,
        artifact_digest_override=frozen.digest,
    ))
    _guard_adapter_rollout(adapter, observer)
    env = getattr(adapter, build_env_name)(seed=wire.run_seed)
    started = time.monotonic()
    try:
        rows = adapter.rollout(env, best_skill, str(evaluation_out / "rollout"))
    except Exception as exc:
        after_error = digest_directory(frozen.root)
        if after_error != before:
            raise WorkerPhaseError(
                f"frozen artifact changed while {wire.phase} was failing",
                frozen_digest_before=before,
                frozen_digest_after=after_error,
                frozen_unchanged=False,
            ) from exc
        raise
    after = digest_directory(frozen.root)
    if before != frozen.digest or after != before:
        raise WorkerPhaseError(
            f"frozen artifact changed during {wire.phase}",
            frozen_digest_before=before,
            frozen_digest_after=after,
            frozen_unchanged=False,
        )
    episodes = _collect_episodes(evaluation_out)
    _validate_episode_action_evidence(evaluation_out)
    if len(rows) != len(evaluation_manifest.tasks):
        raise RuntimeError(
            f"{wire.phase} row count does not equal {cardinality_name} cardinality"
        )
    verify_final_evaluation_bijection(
        episodes,
        evaluation_manifest,
        role=role,
        expected_phase=wire.phase,
        expected_method=METHOD_ID,
        expected_run_seed=wire.run_seed,
        expected_artifact_digest=frozen.digest,
        require_strict_outcomes=False,
        profile=str(config.get("protocol_profile", "pilot_v1")),
    )
    from skillopt.model import get_token_summary

    usage = _write_usage(
        evaluation_out,
        observer,
        wire=wire,
        started=started,
        upstream=get_token_summary(),
        forbid_evolution=True,
    )
    if usage.target.calls <= 0:
        raise RuntimeError(f"{wire.phase} made no successful target-provider calls")
    final_after = digest_directory(frozen.root)
    if final_after != before:
        raise WorkerPhaseError(
            f"frozen artifact changed while {wire.phase} evidence was finalized",
            frozen_digest_before=before,
            frozen_digest_after=final_after,
            frozen_unchanged=False,
        )
    return {
        "output_dir": str(Path(wire.output_dir).resolve()),
        "episodes": len(episodes),
        "rows": len(rows),
        "usage": usage.to_dict(),
        "frozen_digest_before": before,
        "frozen_digest_after": final_after,
        "frozen_unchanged": True,
        "provider_calls": len(observer.events()),
        "trainer_constructed": False,
        "evaluation_manifest_digest": evaluation_manifest.digest,
        "best_skill_sha256": hashlib.sha256(best_skill_path.read_bytes()).hexdigest(),
        "frozen_artifact": {
            "method_id": frozen.method_id,
            "digest": frozen.digest,
            "source_train_manifest_hash": frozen.source_train_manifest_hash,
            "source_validation_manifest_hash": frozen.source_validation_manifest_hash,
        },
        "provider_evidence": _provider_evidence_summary(observer),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("skillopt", "alfworld", "openai", "gymnasium", "omegaconf"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _provenance(
    *,
    wire: WorkerWire,
    config_digest: str,
    source: dict[str, Any],
    manifest_receipts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": wire.method,
        "phase": wire.phase,
        "run_id": wire.run_id,
        "run_seed": wire.run_seed,
        "model": dict(wire.model),
        "identity": dict(wire.identity),
        "resolved_config_path": str(Path(wire.config_path).resolve()),
        "resolved_config_digest": config_digest,
        "source": source,
        "manifests": manifest_receipts,
        "alfworld_data": str(Path(os.environ["ALFWORLD_DATA"]).resolve()),
        "versions": _package_versions(),
        "runtime": {
            "python": sys.version,
            # Preserve the invoked venv path; resolving its POSIX symlink
            # would misleadingly report only the base interpreter.
            "executable": os.path.abspath(sys.executable),
            "executable_resolved_target": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "os_name": os.name,
        },
    }


def _safe_error(exc: BaseException, model: ModelConfig | None) -> str:
    text = str(exc)
    candidates = [os.environ.get("MODEL_API_KEY", "")]
    if model is not None:
        candidates.append(os.environ.get(model.api_key_env, ""))
    for candidate in candidates:
        if candidate:
            text = text.replace(candidate, "<redacted>")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", required=True, help="phase-local WorkerWire v2 JSON")
    args = parser.parse_args(argv)
    wire: WorkerWire | None = None
    model: ModelConfig | None = None
    observer: ProviderCallObserver | None = None
    started = time.monotonic()
    try:
        wire_path = Path(args.wire)
        payload = json.loads(wire_path.read_text(encoding="utf-8"))
        wire = WorkerWire.from_dict(payload)
        if wire.method != METHOD_ID:
            raise ValueError(f"worker method must be {METHOD_ID!r}")
        if wire.phase not in _ALLOWED_PHASES:
            raise ValueError(
                f"unsupported worker phase {wire.phase!r}; expected {sorted(_ALLOWED_PHASES)}"
            )
        _validate_wire_location(wire, wire_path)
        _assert_fresh_phase(wire)
        config = _load_method_config(wire.config_path)
        model = ModelConfig.from_mapping(wire.model)
        model.validate_formal_identity()
        config_digest = _validate_config_identity(wire, config, model)
        lock = load_lock(
            REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml"
        )
        source = _verify_skillopt_source(wire, lock)
        train_manifest, validation_manifest, test_manifest, manifest_receipts = (
            _load_and_verify_manifests(wire, config)
        )
        phase_dir = _phase_dir(wire)
        _write_json_exclusive(
            phase_dir / "environment_provenance.json",
            _provenance(
                wire=wire,
                config_digest=config_digest,
                source=source,
                manifest_receipts=manifest_receipts,
            ),
        )
        provider_transport = dict(config["provider_transport"])
        _configure_model(
            model,
            sdk_max_retries=int(provider_transport["sdk_max_retries"]),
        )
        provider_path = phase_dir / "provider_calls.jsonl"
        if provider_path.exists():
            raise FileExistsError(provider_path)
        observer = install_provider_observer(
            output_path=provider_path,
            method=wire.method,
            phase=wire.phase,
            model=model.model,
            reasoning_effort=model.reasoning_effort,
            run_id=wire.run_id,
            application_retry_limit=int(
                provider_transport["application_retry_limit"]
            ),
            retry_delays_seconds=list(
                provider_transport["retry_delays_seconds"]
            ),
            expected_sdk_max_retries=int(
                provider_transport["sdk_max_retries"]
            ),
        )
        if wire.phase == "train":
            result = _run_train(
                wire=wire,
                model=model,
                config=config,
                lock=lock,
                source=source,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                observer=observer,
            )
        elif wire.phase == "smoke":
            result = _run_smoke(
                wire=wire,
                config=config,
                lock=lock,
                source=source,
                train_manifest=train_manifest,
                observer=observer,
            )
        else:
            result = _run_frozen_evaluation(
                wire=wire,
                config=config,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                test_manifest=test_manifest,
                observer=observer,
            )
        write_worker_result(wire, result, passed=True)
        print(json.dumps({
            "passed": True,
            "method": wire.method,
            "phase": wire.phase,
            "run_id": wire.run_id,
            "result_path": wire.result_path,
        }, ensure_ascii=False, indent=2))
        return 0
    except (Exception, ProviderCallExhausted) as exc:
        error = _safe_error(exc, model)
        evidence = dict(getattr(exc, "evidence", {}) or {})
        if observer is not None:
            evidence.setdefault("provider_calls", observer.event_cursor())
            usage_path = _phase_dir(wire) / "usage.json" if wire is not None else None
            try:
                persisted_events = observer.events()
            except Exception as audit_exc:
                persisted_events = []
                evidence["provider_evidence_error"] = _safe_error(audit_exc, model)
            if usage_path is not None and not usage_path.exists() and persisted_events:
                try:
                    partial = _provider_usage(
                        observer,
                        wire=wire,
                        wall_time_ms=int((time.monotonic() - started) * 1000),
                        allow_failed=True,
                    )
                    _write_json_exclusive(usage_path, partial.to_dict())
                    evidence["partial_usage"] = partial.to_dict()
                except Exception:
                    pass
        if wire is not None and not Path(wire.result_path).exists():
            try:
                write_worker_result(
                    wire,
                    {
                        "error_type": type(exc).__name__,
                        "error": error,
                        **evidence,
                    },
                    passed=False,
                )
            except Exception as write_exc:
                error = f"{error}; worker result write failed: {_safe_error(write_exc, model)}"
        print(json.dumps({
            "passed": False,
            "method": wire.method if wire else METHOD_ID,
            "phase": wire.phase if wire else "unknown",
            "run_id": wire.run_id if wire else "",
            "error_type": type(exc).__name__,
            "error": error,
        }, ensure_ascii=False, indent=2))
        return 1
    finally:
        if observer is not None:
            try:
                uninstall_provider_observer(observer)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
