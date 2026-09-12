"""Isolated B5 GEPA worker using pinned GEPA and SkillOpt sources."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from experiments.baselines.bootstrap_external import (
    load_lock,
    source_file_sha256,
    verify_key_files,
    verify_runtime_tree,
)
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.formal_validation import verify_formal_manifest
from experiments.baselines.common.freeze import FrozenArtifact
from experiments.baselines.common.manifest import (
    TaskManifestSet,
    sha256_json,
    verify_disjoint,
)
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.subprocess_worker import WorkerWire, write_worker_result
from experiments.baselines.common.trace import load_episodes
from experiments.baselines.common.usage import UsageSnapshot
from experiments.baselines.b3_skillopt.provider_observer import (
    ProviderCallExhausted,
    ProviderCallObserver,
    install_provider_observer,
    uninstall_provider_observer,
)
from experiments.baselines.b3_skillopt.worker import (
    _campaign_provider_gate,
    _configure_model,
    _failure_kind as _b3_failure_kind,
    _provider_evidence_summary,
    _provider_usage,
    _safe_error,
    _validate_episode_action_evidence,
)

from .adapter import (
    ALFWorldGEPAAdapter,
    METHOD_ID,
    _write_json_exclusive,
    manifest_examples,
    one_per_family,
)
from .callbacks import GEPAAuditCallback
from .reflection_lm import DeepSeekReflectionLM


REPO_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_PHASES = {"smoke", "train", "train_eval", "smoke_test", "test"}
_INITIAL_SKILL_REL = "skillopt/envs/alfworld/skills/initial.md"


class WorkerPhaseError(RuntimeError):
    def __init__(self, message: str, *, failure_kind: str = "protocol_failure", **evidence: Any) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.evidence = evidence


def _phase_dir(wire: WorkerWire) -> Path:
    return Path(wire.output_dir).resolve() / wire.phase


def _load_config(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resolved GEPA config is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("resolved GEPA config root must be a mapping")
    return dict(payload)


def _validate_config(
    wire: WorkerWire, config: dict[str, Any], model: ModelConfig
) -> tuple[str, int]:
    digest = sha256_json(config)
    if digest != wire.identity.get("config_digest"):
        raise ValueError("GEPA config digest does not match WorkerWire identity")
    if config.get("method") != METHOD_ID:
        raise ValueError(f"GEPA config method must be {METHOD_ID!r}")
    model_from_config = ModelConfig.from_mapping(dict(config.get("model") or {}))
    if model_from_config.to_wire() != model.to_wire():
        raise ValueError("GEPA worker model differs from resolved config")
    model.validate_formal_identity()

    experiment_kind = str(config.get("experiment_kind", "formal"))
    if experiment_kind not in {"formal", "smoke"}:
        raise ValueError("GEPA experiment_kind must be formal or smoke")
    gepa_cfg = dict(config.get("gepa") or {})
    expected = {
        "candidate_component": "skill_text",
        "candidate_selection_strategy": "pareto",
        "frontier_type": "instance",
        "skip_perfect_score": True,
        "batch_sampler": "epoch_shuffled",
        "reflection_minibatch_size": 3,
        "perfect_score": 1.0,
        "module_selector": "round_robin",
        "use_merge": False,
        "cache_evaluation": False,
        "val_evaluation_policy": "full_eval",
        "acceptance_criterion": "strict_improvement",
        "sampling_strategy": None,
        "selection_strategy": None,
    }
    mismatches = [
        key for key, wanted in expected.items() if gepa_cfg.get(key) != wanted
    ]
    if mismatches:
        raise ValueError(
            "GEPA frozen optimizer settings mismatch: " + ", ".join(mismatches)
        )
    max_metric_calls = int(gepa_cfg.get("max_metric_calls", 0))
    wanted_budget = 720 if experiment_kind == "formal" else 24
    if max_metric_calls != wanted_budget:
        raise ValueError(
            f"{experiment_kind} GEPA max_metric_calls must be {wanted_budget}"
        )
    expected_profile = "smoke_v1" if experiment_kind == "smoke" else "formal_v2"
    if str(config.get("protocol_profile")) != expected_profile:
        raise ValueError(
            f"{experiment_kind} GEPA requires protocol_profile={expected_profile!r}"
        )
    if wire.phase in {"smoke", "smoke_test"} and experiment_kind != "smoke":
        raise ValueError("GEPA smoke phases must use the dedicated smoke config")
    if wire.phase not in {"smoke", "smoke_test"} and experiment_kind != "formal":
        raise ValueError("GEPA formal phases cannot use the smoke config")

    env = dict(config.get("env") or {})
    if str(env.get("name")) != "alfworld":
        raise ValueError("GEPA environment must be ALFWorld")
    max_steps = int(env.get("max_steps", 0))
    if experiment_kind == "formal" and max_steps != 100:
        raise ValueError("formal GEPA must retain the 100-action ceiling")
    if max_steps <= 0 or max_steps > int(config.get("max_environment_actions", 0)):
        raise ValueError("GEPA env.max_steps is outside the common action ceiling")
    if int(env.get("workers", 0)) <= 0 or int(env.get("max_api_workers", 0)) <= 0:
        raise ValueError("GEPA episode/provider worker counts must be positive")
    transport = dict(config.get("provider_transport") or {})
    if int(transport.get("sdk_max_retries", -1)) != 0:
        raise ValueError("GEPA requires zero SDK-internal retries")
    if int(transport.get("application_retry_limit", 0)) <= 0:
        raise ValueError("GEPA application retry limit must be positive")
    return experiment_kind, max_metric_calls


def _source_root(wire: WorkerWire, name: str) -> Path:
    if name == "gepa":
        raw = wire.external_method_root or str(REPO_ROOT / ".external" / "gepa")
    else:
        raw = wire.external_skillopt_root or str(REPO_ROOT / ".external" / "skillopt")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pinned {name} source is missing: {root}")
    return root


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot inspect pinned GEPA git source: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _assert_import_origin(module_name: str, expected_root: Path) -> str:
    module = importlib.import_module(module_name)
    origin_raw = getattr(module, "__file__", None)
    if not origin_raw:
        raise RuntimeError(f"module {module_name!r} has no import origin")
    origin = Path(origin_raw).resolve()
    try:
        origin.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"module {module_name!r} loaded outside its pinned tree: {origin}"
        ) from exc
    return str(origin)


def _verify_sources(wire: WorkerWire, lock: dict[str, Any]) -> dict[str, Any]:
    gepa_root = _source_root(wire, "gepa")
    skillopt_root = _source_root(wire, "skillopt")
    verified_gepa = verify_key_files(gepa_root, "gepa", lock)
    verified_skillopt = verify_key_files(skillopt_root, "skillopt", lock)
    gepa_head = _git_output(gepa_root, "rev-parse", "HEAD")
    expected_commit = str(lock["gepa"]["commit"])
    if gepa_head != expected_commit:
        raise RuntimeError(
            f"GEPA checkout commit mismatch: expected {expected_commit}, got {gepa_head}"
        )
    expected_tag = str(lock["gepa"].get("tag") or "")
    tags = _git_output(gepa_root, "tag", "--points-at", "HEAD").splitlines()
    if expected_tag and expected_tag not in tags:
        raise RuntimeError(f"GEPA checkout HEAD is not tagged {expected_tag}")
    installed_version = importlib.metadata.version("gepa")
    expected_version = str(lock["gepa"].get("version") or "")
    if expected_version and installed_version != expected_version:
        raise RuntimeError(
            f"installed GEPA metadata mismatch: expected {expected_version}, "
            f"got {installed_version}"
        )
    expected_skillopt = str(lock["skillopt"].get("version") or "")
    result = {
        "gepa": {
            "root": str(gepa_root),
            "repo": str(lock["gepa"]["repo"]),
            "tag": expected_tag,
            "commit": gepa_head,
            "declared_version": expected_version,
            "installed_version": installed_version,
            "runtime_tree": verify_runtime_tree(gepa_root, "gepa", lock),
            "verified_key_files": len(verified_gepa),
            "import_origins": {
                name: _assert_import_origin(name, gepa_root)
                for name in ("gepa", "gepa.api", "gepa.core.engine")
            },
        },
        "skillopt": {
            "root": str(skillopt_root),
            "repo": str(lock["skillopt"]["repo"]),
            "commit": str(lock["skillopt"]["commit"]),
            "declared_version": expected_skillopt,
            # SkillOpt is intentionally imported from the pinned source tree;
            # no independently installed distribution is an authority here.
            "installed_version": None,
            "runtime_tree": verify_runtime_tree(skillopt_root, "skillopt", lock),
            "verified_key_files": len(verified_skillopt),
            "import_origins": {
                name: _assert_import_origin(name, skillopt_root)
                for name in (
                    "skillopt",
                    "skillopt.model",
                    "skillopt.envs.alfworld.rollout",
                )
            },
        },
    }
    if (
        wire.identity.get("external_source_digest")
        != result["gepa"]["runtime_tree"]["sha256"]
    ):
        raise RuntimeError("GEPA runtime digest differs from WorkerWire identity")
    if (
        wire.identity.get("skillopt_source_digest")
        != result["skillopt"]["runtime_tree"]["sha256"]
    ):
        raise RuntimeError("SkillOpt runtime digest differs from WorkerWire identity")
    return result


def _load_manifests(
    wire: WorkerWire,
    *,
    profile: str,
) -> tuple[
    TaskManifestSet | None,
    TaskManifestSet | None,
    TaskManifestSet | None,
    dict[str, Any],
]:
    alfworld_data = os.environ.get("ALFWORLD_DATA", "")
    if not alfworld_data:
        raise RuntimeError("ALFWORLD_DATA is not set in the GEPA worker")

    train: TaskManifestSet | None = None
    validation: TaskManifestSet | None = None
    test: TaskManifestSet | None = None
    receipts: dict[str, Any] = {}
    if wire.phase in {"train", "smoke"}:
        train = TaskManifestSet.load(Path(wire.manifest_path or ""))
        validation = TaskManifestSet.load(Path(wire.validation_manifest_path or ""))
        if wire.identity.get("train_manifest_digest") != train.digest:
            raise ValueError("GEPA Train manifest digest mismatch")
        if wire.identity.get("validation_manifest_digest") != validation.digest:
            raise ValueError("GEPA Validation manifest digest mismatch")
        verify_disjoint(train, validation)
        receipts["train"] = verify_formal_manifest(
            train, alfworld_data=alfworld_data, role="train", profile=profile
        )
        receipts["validation"] = verify_formal_manifest(
            validation,
            alfworld_data=alfworld_data,
            role="validation",
            profile=profile,
        )
    elif wire.phase == "train_eval":
        train = TaskManifestSet.load(Path(wire.manifest_path or ""))
        if (
            wire.identity.get("train_manifest_digest") != train.digest
            or wire.identity.get("evaluation_manifest_digest") != train.digest
        ):
            raise ValueError("GEPA train_eval manifest identity mismatch")
        receipts["train"] = verify_formal_manifest(
            train, alfworld_data=alfworld_data, role="train", profile=profile
        )
    elif wire.phase in {"smoke_test", "test"}:
        test = TaskManifestSet.load(Path(wire.test_manifest_path or ""))
        if (
            wire.identity.get("test_manifest_digest") != test.digest
            or wire.identity.get("evaluation_manifest_digest") != test.digest
        ):
            raise ValueError("GEPA Test manifest identity mismatch")
        receipts["test"] = verify_formal_manifest(
            test, alfworld_data=alfworld_data, role="test", profile=profile
        )
    else:
        raise ValueError(f"unsupported GEPA manifest phase: {wire.phase}")
    return train, validation, test, receipts


def _initial_skill(wire: WorkerWire, source: dict[str, Any], lock: dict[str, Any]) -> tuple[str, str, Path]:
    skillopt_root = Path(source["skillopt"]["root"])
    default = skillopt_root / _INITIAL_SKILL_REL
    path = Path(wire.initial_skill_path or default).resolve()
    if path != default.resolve():
        raise ValueError("GEPA seed must be the pinned SkillOpt initial.md")
    if not path.is_file():
        raise FileNotFoundError(f"GEPA initial skill is missing: {path}")
    runtime_tree = dict(lock["skillopt"]["runtime_tree"])
    digest = source_file_sha256(
        path,
        algorithm=str(runtime_tree["algorithm"]),
    )
    expected = str(lock["skillopt"]["key_files"][_INITIAL_SKILL_REL])
    if digest != expected:
        raise RuntimeError("GEPA initial skill digest differs from B1/B3 authority")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError("GEPA initial skill is empty")
    return text, digest, path


def _provider_setup(config: dict[str, Any], model: ModelConfig) -> None:
    transport = dict(config["provider_transport"])
    _configure_model(model, sdk_max_retries=int(transport["sdk_max_retries"]))


def _build_adapter(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    phase: str,
    output_root: Path,
    artifact_digest: str | None = None,
) -> ALFWorldGEPAAdapter:
    env = dict(config["env"])
    return ALFWorldGEPAAdapter(
        output_root=output_root,
        run_seed=wire.run_seed,
        max_actions=int(env["max_steps"]),
        max_completion_tokens=int(env["max_completion_tokens"]),
        workers=min(int(env["workers"]), int(env["max_api_workers"])),
        alfworld_data=os.environ["ALFWORLD_DATA"],
        execution_phase=phase,
        artifact_digest_override=artifact_digest,
        process_context={
            "model": dict(wire.model),
            "provider_transport": dict(config["provider_transport"]),
            "run_id": wire.run_id,
            "campaign": dict(wire.campaign) if wire.campaign is not None else None,
        },
    )


class _ProviderEventView:
    """Read-only observer facade for parent and spawned-child call evidence."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = [dict(event) for event in events]

    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]


def _provider_event_view(
    *,
    phase_dir: Path,
    observer: ProviderCallObserver,
) -> _ProviderEventView:
    events = observer.events()
    parent_path = observer.output_path.resolve()
    for path in sorted(phase_dir.rglob("provider_calls.jsonl")):
        if path.resolve() == parent_path:
            continue
        try:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"row {line_number} is not a mapping")
                events.append(dict(payload))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"GEPA provider sidecar is unreadable: {path}") from exc
    return _ProviderEventView(events)


_TRANSIENT_PROVIDER_CODES = frozenset({
    "connection",
    "empty_choices",
    "empty_message",
    "invalid_usage",
    "rate_limit",
    "server_error",
    "timeout",
})


def _verified_child_failure_receipt(wire: WorkerWire) -> dict[str, Any] | None:
    """Prove a spawned episode exhausted a transient provider call."""

    evaluations = _phase_dir(wire) / "evaluations"
    if not evaluations.is_dir():
        return None
    failed_events: list[dict[str, Any]] = []
    evidence_paths: list[dict[str, str]] = []
    seen_call_ids: set[str] = set()
    for path in sorted(evaluations.rglob("provider_calls.jsonl")):
        path_failed = False
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        for event in rows:
            if not isinstance(event, dict):
                return None
            try:
                call_id = str(event.get("call_id", ""))
                identity_valid = bool(
                    int(event.get("schema_version", 0)) >= 2
                    and event.get("event") == "provider_call"
                    and event.get("method") == METHOD_ID
                    and event.get("phase") == wire.phase
                    and event.get("run_id") == wire.run_id
                    and int(event.get("run_seed", -1)) == wire.run_seed
                    and event.get("model") == str(wire.model.get("model", ""))
                    and event.get("reasoning_effort")
                    == str(wire.model.get("reasoning_effort", ""))
                    and event.get("status") in {"succeeded", "failed"}
                    and call_id
                    and call_id not in seen_call_ids
                )
            except (TypeError, ValueError):
                return None
            if not identity_valid:
                return None
            seen_call_ids.add(call_id)
            if event.get("status") != "failed":
                continue
            try:
                attempts = int(event.get("application_attempts", 0))
                retry_limit = int(event.get("retry_limit", 0))
                sdk_attempts = int(event.get("sdk_boundary_attempts", -1))
                raw_counts = event.get("failure_code_counts")
                if not isinstance(raw_counts, dict):
                    return None
                failure_counts = {
                    str(code): int(count) for code, count in raw_counts.items()
                }
            except (TypeError, ValueError):
                return None
            code = str(event.get("last_failure_code", ""))
            if (
                code not in _TRANSIENT_PROVIDER_CODES
                or code not in failure_counts
                or any(not key or count <= 0 for key, count in failure_counts.items())
                or attempts <= 0
                or retry_limit <= 0
                or attempts < retry_limit
                or sdk_attempts != attempts
                or sum(int(value) for value in failure_counts.values()) != attempts
            ):
                return None
            failed_events.append(dict(event))
            path_failed = True
        if path_failed:
            evidence_paths.append({
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    if not failed_events:
        return None
    return {
        "schema_version": 1,
        "failed_calls": len(failed_events),
        "failure_codes": sorted({
            str(event["last_failure_code"]) for event in failed_events
        }),
        "sidecars": evidence_paths,
    }


def _gepa_failure_kind(
    exc: BaseException,
    observer: ProviderCallObserver | None,
    wire: WorkerWire | None,
) -> tuple[str, dict[str, Any] | None]:
    inherited = _b3_failure_kind(exc, observer)
    if inherited == "infrastructure_failure":
        return inherited, None
    if (
        wire is not None
        and str(getattr(exc, "failure_kind", "")) == "infrastructure_failure"
    ):
        receipt = _verified_child_failure_receipt(wire)
        if receipt is not None:
            return "infrastructure_failure", receipt
    return "protocol_failure", None


def _collect_episodes(root: Path) -> list[Any]:
    sidecars = sorted(root.rglob("common_episodes.jsonl"))
    if not sidecars:
        raise FileNotFoundError(f"GEPA produced no Common episode evidence under {root}")
    episodes: list[Any] = []
    for path in sidecars:
        episodes.extend(load_episodes(path))
    return episodes


def _validate_episodes(
    episodes: list[Any], *, wire: WorkerWire, allowed_phases: set[str]
) -> dict[str, int]:
    role_counts: dict[str, int] = {}
    for episode in episodes:
        if (
            episode.method != METHOD_ID
            or episode.phase not in allowed_phases
            or episode.run_seed != wire.run_seed
        ):
            raise ValueError("GEPA Common episode identity mismatch")
        if episode.infrastructure_failure:
            raise WorkerPhaseError(
                "GEPA episode contains infrastructure failure",
                failure_kind=(
                    str(episode.method_metrics.get("failure_kind"))
                    or "protocol_failure"
                ),
                task_id=episode.task_id,
            )
        if episode.environment_actions < 0:
            raise ValueError("GEPA episode has negative action count")
        if episode.target_llm_calls <= 0:
            raise ValueError("GEPA episode has no target provider usage")
        role = str(episode.method_metrics.get("dataset_role", episode.phase))
        role_counts[role] = role_counts.get(role, 0) + 1
    return role_counts


def _write_optimization_artifacts(
    *,
    phase_dir: Path,
    result: Any,
    callback: GEPAAuditCallback,
    reflection_lm: DeepSeekReflectionLM,
    adapter: ALFWorldGEPAAdapter,
    configured_budget: int,
    initial_skill_sha256: str,
    validation_size: int,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    result_payload = result.to_dict()
    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict) or set(best_candidate) != {"skill_text"}:
        raise RuntimeError("GEPA result best candidate violates skill_text-only contract")
    if any(set(candidate) != {"skill_text"} for candidate in result.candidates):
        raise RuntimeError("GEPA candidate pool contains an unauthorized component")
    actual_calls = int(result.total_metric_calls or 0)
    callback_summary = callback.summary()
    adapter_metrics = adapter.metrics()
    if actual_calls != int(adapter_metrics["metric_calls"]):
        raise RuntimeError(
            "GEPA result metric-call count disagrees with adapter episode count"
        )
    resumed_calls = int(adapter_metrics["resumed_metric_calls"])
    replay_calls = int(adapter_metrics["resume_replay_metric_calls"])
    cumulative_reflection_calls = int(adapter_metrics["reflective_dataset_calls"])
    resumed_reflection_calls = int(adapter_metrics["resumed_reflection_calls"])
    attempt_reflection_calls = (
        cumulative_reflection_calls - resumed_reflection_calls
    )
    is_resume = resumed_calls > 0
    if not is_resume and int(result.num_full_val_evals or 0) != int(
        callback_summary["num_full_val_evals"]
    ):
        raise RuntimeError("GEPA full-validation callback count disagrees with result")
    if int(callback_summary["unexpected_merge_events"]) != 0:
        raise RuntimeError("GEPA emitted merge events while use_merge=false")
    if reflection_lm.calls != int(callback_summary["reflection_calls"]):
        raise RuntimeError("GEPA reflection transport and callback counts disagree")
    if (
        attempt_reflection_calls < 0
        or attempt_reflection_calls != reflection_lm.calls
    ):
        raise RuntimeError(
            "GEPA cumulative reflection count disagrees with resumed/current attempts"
        )
    best_skill = str(best_candidate["skill_text"])
    if count_tokens_fn is None:
        from skillopt.model.openai_compatible_backend import (
            count_tokens as count_tokens_fn,
        )

    summary = {
        "configured_max_metric_calls": configured_budget,
        "actual_total_metric_calls": actual_calls,
        "budget_overshoot": max(0, actual_calls - configured_budget),
        "iteration_count": int(callback_summary["iteration_count"]),
        "candidate_count": len(result.candidates),
        "accepted_candidate_count": max(0, len(result.candidates) - 1),
        "attempt_accepted_candidate_count": int(
            callback_summary["accepted_candidate_count"]
        ),
        "num_full_val_evals": int(result.num_full_val_evals or 0),
        "attempt_full_val_evals_including_resume_replay": int(
            callback_summary["num_full_val_evals"]
        ),
        "reflection_calls": cumulative_reflection_calls,
        "attempt_reflection_calls": attempt_reflection_calls,
        "resumed_reflection_calls": resumed_reflection_calls,
        "resumed_metric_calls": resumed_calls,
        "resume_replay_metric_calls": replay_calls,
        "best_validation_score": float(result.val_aggregate_scores[result.best_idx]),
        "best_skill_tokens": int(count_tokens_fn(best_skill)),
        "best_skill_sha256": hashlib.sha256(best_skill.encode("utf-8")).hexdigest(),
        "initial_skill_sha256": initial_skill_sha256,
        "candidate_component": "skill_text",
        "candidate_selection_strategy": "pareto",
        "frontier_type": "instance",
        "batch_sampler": "epoch_shuffled",
        "reflection_minibatch_size": 3,
        "validation_policy": "full_eval",
        "validation_size": validation_size,
        "acceptance_criterion": "strict_improvement",
        "use_merge": False,
        "cache_evaluation": False,
        "sampling_strategy": None,
        "selection_strategy": None,
        "adapter": adapter_metrics,
    }
    lineage = {
        "schema_version": 1,
        "candidates": [
            {
                "candidate_index": index,
                "skill_text_sha256": hashlib.sha256(
                    candidate["skill_text"].encode("utf-8")
                ).hexdigest(),
                "parent_indices": list(result.parents[index]),
                "validation_score": float(result.val_aggregate_scores[index]),
                "metric_calls_at_discovery": int(result.discovery_eval_counts[index]),
            }
            for index, candidate in enumerate(result.candidates)
        ],
    }
    pareto = {
        "schema_version": 1,
        "frontier_type": "instance",
        "best_candidate_index": int(result.best_idx),
        "validation_aggregate_scores": list(result.val_aggregate_scores),
        "per_validation_instance_best_candidates": {
            str(key): sorted(int(index) for index in value)
            for key, value in result.per_val_instance_best_candidates.items()
        },
        "validation_subscores": [
            {str(key): float(value) for key, value in scores.items()}
            for scores in result.val_subscores
        ],
    }
    files = {
        "best_skill.md": phase_dir / "best_skill.md",
        "gepa_result.json": phase_dir / "gepa_result.json",
        "candidate_lineage.json": phase_dir / "candidate_lineage.json",
        "pareto_metadata.json": phase_dir / "pareto_metadata.json",
        "gepa_audit_summary.json": phase_dir / "gepa_audit_summary.json",
    }
    if files["best_skill.md"].exists():
        raise FileExistsError(files["best_skill.md"])
    files["best_skill.md"].write_text(best_skill, encoding="utf-8")
    _write_json_exclusive(files["gepa_result.json"], result_payload)
    _write_json_exclusive(files["candidate_lineage.json"], lineage)
    _write_json_exclusive(files["pareto_metadata.json"], pareto)
    _write_json_exclusive(
        files["gepa_audit_summary.json"],
        {"schema_version": 1, **callback_summary, **summary},
    )
    return files, summary


def _run_optimization(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    train: TaskManifestSet,
    validation: TaskManifestSet,
    initial_skill: str,
    initial_digest: str,
    observer: ProviderCallObserver,
    configured_budget: int,
) -> dict[str, Any]:
    from gepa import optimize

    phase_dir = _phase_dir(wire)
    experiment_kind = str(config["experiment_kind"])
    if experiment_kind == "smoke":
        train_examples = manifest_examples(train, dataset_role="train")
        val_examples = manifest_examples(validation, dataset_role="validation")
        expected = int(dict(config.get("smoke") or {}).get("task_count_per_split", 6))
        if len(train_examples) != expected or len(val_examples) != expected:
            raise ValueError("GEPA smoke did not select exactly one task per family")
    else:
        train_examples = manifest_examples(train, dataset_role="train")
        val_examples = manifest_examples(validation, dataset_role="validation")
        if len(train_examples) != 120 or len(val_examples) != 24:
            raise ValueError("formal GEPA requires Train120 and Validation24")
    evaluations = phase_dir / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=False)
    adapter = _build_adapter(
        wire=wire,
        config=config,
        phase=wire.phase,
        output_root=evaluations,
    )
    callback = GEPAAuditCallback(phase_dir / "gepa_events.jsonl")
    reflection_lm = DeepSeekReflectionLM(
        reasoning_effort=str(wire.model["reasoning_effort"]),
        max_completion_tokens=int(dict(config["gepa"])["reflection_max_completion_tokens"]),
        retries=int(dict(config["provider_transport"])["application_retry_limit"]),
    )
    gepa_cfg = dict(config["gepa"])
    started = time.monotonic()
    try:
        result = optimize(
            seed_candidate={"skill_text": initial_skill},
            trainset=train_examples,
            valset=val_examples,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            skip_perfect_score=True,
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=3,
            perfect_score=1.0,
            module_selector="round_robin",
            use_merge=False,
            max_metric_calls=configured_budget,
            cache_evaluation=False,
            seed=wire.run_seed,
            val_evaluation_policy="full_eval",
            acceptance_criterion="strict_improvement",
            sampling_strategy=None,
            selection_strategy=None,
            run_dir=str(phase_dir / "gepa_state"),
            callbacks=[callback],
            raise_on_exception=True,
            display_progress_bar=False,
        )
    finally:
        adapter.close()
    files, summary = _write_optimization_artifacts(
        phase_dir=phase_dir,
        result=result,
        callback=callback,
        reflection_lm=reflection_lm,
        adapter=adapter,
        configured_budget=configured_budget,
        initial_skill_sha256=initial_digest,
        validation_size=len(val_examples),
    )
    episodes = _collect_episodes(evaluations)
    _validate_episode_action_evidence(evaluations)
    role_counts = _validate_episodes(
        episodes,
        wire=wire,
        allowed_phases={"smoke"} if wire.phase == "smoke" else {"train", "validation"},
    )
    attempt_algorithm_calls = (
        int(summary["actual_total_metric_calls"])
        - int(summary["resumed_metric_calls"])
    )
    expected_attempt_episodes = (
        attempt_algorithm_calls + int(summary["resume_replay_metric_calls"])
    )
    if attempt_algorithm_calls < 0 or len(episodes) != expected_attempt_episodes:
        raise RuntimeError(
            "GEPA cumulative metric calls do not reconcile with current-attempt "
            "and resume-replay episodes"
        )
    if role_counts.get("validation", 0) != int(
        summary["attempt_full_val_evals_including_resume_replay"]
    ) * len(val_examples):
        raise RuntimeError("GEPA validation episodes do not prove full-eval policy")
    if role_counts.get("train", 0) <= 0 and int(summary["resumed_metric_calls"]) == 0:
        raise RuntimeError("GEPA optimization produced no Train minibatch episodes")
    if reflection_lm.calls <= 0 and int(summary["resumed_metric_calls"]) == 0:
        raise RuntimeError("GEPA optimization produced no reflection proposal")
    provider_events = _provider_event_view(phase_dir=phase_dir, observer=observer)
    usage = _provider_usage(
        provider_events,
        wire=wire,
        wall_time_ms=int((time.monotonic() - started) * 1000),
    )
    if usage.target.calls <= 0 or (
        usage.evolution.calls <= 0 and int(summary["resumed_metric_calls"]) == 0
    ):
        raise RuntimeError("GEPA optimization lacks target or reflection provider usage")
    usage.save(phase_dir / "usage.json")
    return {
        "output_dir": str(Path(wire.output_dir).resolve()),
        "episodes": {
            "total": len(episodes),
            "train": role_counts.get("train", 0),
            "validation": role_counts.get("validation", 0),
        },
        "rows": len(episodes),
        "usage": usage.to_dict(),
        "train_summary": summary,
        "persistent_artifacts": {name: str(path) for name, path in files.items()},
        "provider_calls": len(provider_events.events()),
        "provider_evidence": _provider_evidence_summary(provider_events),
        "optimizer_constructed": True,
    }


def _run_frozen_evaluation(
    *,
    wire: WorkerWire,
    config: dict[str, Any],
    train: TaskManifestSet | None,
    test: TaskManifestSet | None,
    observer: ProviderCallObserver,
) -> dict[str, Any]:
    if wire.phase in {"smoke_test", "test"}:
        if test is None:
            raise ValueError("GEPA test manifest is missing")
        manifest = test
        role = wire.phase
    elif wire.phase == "train_eval":
        if train is None:
            raise ValueError("GEPA train_eval manifest is missing")
        manifest = train
        role = "train_eval"
    else:
        raise ValueError(f"unsupported frozen phase: {wire.phase}")
    frozen = FrozenArtifact.load(Path(wire.frozen_artifact_path or ""))
    if frozen.method_id != METHOD_ID:
        raise ValueError("frozen artifact belongs to another method")
    if frozen.source_train_manifest_hash != wire.identity.get(
        "train_manifest_digest"
    ):
        raise ValueError("frozen GEPA artifact does not match Train")
    if frozen.source_validation_manifest_hash != wire.identity.get(
        "validation_manifest_digest"
    ):
        raise ValueError("frozen GEPA artifact does not match Validation")
    if wire.identity.get("frozen_artifact_digest") != frozen.digest:
        raise ValueError("frozen GEPA artifact digest differs from WorkerWire")
    best_path = frozen.root / "best_skill.md"
    best_skill = best_path.read_text(encoding="utf-8")
    if not best_skill.strip():
        raise RuntimeError("frozen GEPA best skill is empty")
    before = digest_directory(frozen.root)
    phase_dir = _phase_dir(wire)
    evaluations = phase_dir / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=False)
    adapter = _build_adapter(
        wire=wire,
        config=config,
        phase=wire.phase,
        output_root=evaluations,
        artifact_digest=frozen.digest,
    )
    started = time.monotonic()
    try:
        evaluated = adapter.evaluate(
            manifest_examples(manifest, dataset_role=role),
            {"skill_text": best_skill},
            capture_traces=False,
        )
    finally:
        adapter.close()
    after = digest_directory(frozen.root)
    if before != frozen.digest or after != before:
        raise WorkerPhaseError("frozen GEPA artifact changed during evaluation")
    episodes = _collect_episodes(evaluations)
    _validate_episode_action_evidence(evaluations)
    _validate_episodes(episodes, wire=wire, allowed_phases={wire.phase})
    if len(evaluated.scores) != len(manifest.tasks) or len(episodes) != len(manifest.tasks):
        raise RuntimeError("frozen GEPA evaluation is not manifest-complete")
    provider_events = _provider_event_view(phase_dir=phase_dir, observer=observer)
    usage = _provider_usage(
        provider_events,
        wire=wire,
        wall_time_ms=int((time.monotonic() - started) * 1000),
    )
    if usage.evolution.calls != 0:
        raise RuntimeError("frozen GEPA evaluation invoked the optimizer/reflection LM")
    usage.save(phase_dir / "usage.json")
    final_digest = digest_directory(frozen.root)
    if final_digest != before:
        raise WorkerPhaseError("frozen GEPA artifact changed while finalizing evidence")
    return {
        "output_dir": str(Path(wire.output_dir).resolve()),
        "episodes": len(episodes),
        "rows": len(episodes),
        "usage": usage.to_dict(),
        "official_successes": sum(int(item.official_success) for item in episodes),
        "frozen_digest_before": before,
        "frozen_digest_after": final_digest,
        "frozen_unchanged": True,
        "optimizer_constructed": False,
        "reflection_calls": 0,
        "provider_calls": len(provider_events.events()),
        "provider_evidence": _provider_evidence_summary(provider_events),
    }


def _provenance(
    *,
    wire: WorkerWire,
    config_digest: str,
    source: dict[str, Any],
    manifests: dict[str, Any],
    initial_skill_sha256: str | None,
) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("gepa", "skillopt", "alfworld", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "method": METHOD_ID,
        "phase": wire.phase,
        "run_id": wire.run_id,
        "run_seed": wire.run_seed,
        "model": dict(wire.model),
        "identity": dict(wire.identity),
        "resolved_config_path": str(Path(wire.config_path).resolve()),
        "resolved_config_digest": config_digest,
        "source": source,
        "manifests": manifests,
        "initial_skill_sha256": initial_skill_sha256,
        "alfworld_data": str(Path(os.environ["ALFWORLD_DATA"]).resolve()),
        "versions": versions,
        "runtime": {
            "python": sys.version,
            "executable": os.path.abspath(sys.executable),
            "executable_resolved_target": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", required=True)
    args = parser.parse_args(argv)
    wire: WorkerWire | None = None
    model: ModelConfig | None = None
    observer: ProviderCallObserver | None = None
    started = time.monotonic()
    try:
        wire_path = Path(args.wire).resolve()
        wire = WorkerWire.from_dict(json.loads(wire_path.read_text(encoding="utf-8")))
        if wire.method != METHOD_ID or wire.phase not in _ALLOWED_PHASES:
            raise ValueError("B5 worker received an invalid method/phase")
        expected_phase_dir = _phase_dir(wire)
        if wire_path.parent != expected_phase_dir:
            raise ValueError("B5 WorkerWire is outside its phase directory")
        config = _load_config(wire.config_path)
        model = ModelConfig.from_mapping(wire.model)
        experiment_kind, budget = _validate_config(wire, config, model)
        lock = load_lock(REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml")
        source = _verify_sources(wire, lock)
        profile = str(config["protocol_profile"])
        train, validation, test, manifest_receipts = _load_manifests(
            wire, profile=profile
        )
        initial_skill: str | None = None
        initial_digest: str | None = wire.identity.get("initial_skill_digest")
        if wire.phase in {"train", "smoke"}:
            initial_skill, initial_digest, _ = _initial_skill(wire, source, lock)
        config_digest = sha256_json(config)
        _write_json_exclusive(
            expected_phase_dir / "environment_provenance.json",
            _provenance(
                wire=wire,
                config_digest=config_digest,
                source=source,
                manifests=manifest_receipts,
                initial_skill_sha256=initial_digest,
            ),
        )
        campaign_gate = _campaign_provider_gate(wire, config)
        _provider_setup(config, model)
        provider_path = expected_phase_dir / "provider_calls.jsonl"
        transport = dict(config["provider_transport"])
        observer = install_provider_observer(
            output_path=provider_path,
            method=METHOD_ID,
            phase=wire.phase,
            model=model.model,
            reasoning_effort=model.reasoning_effort,
            run_id=wire.run_id,
            run_seed=wire.run_seed,
            application_retry_limit=int(transport["application_retry_limit"]),
            retry_delays_seconds=list(transport["retry_delays_seconds"]),
            deterministic_jitter_ratio=float(transport["deterministic_jitter_ratio"]),
            expected_sdk_max_retries=int(transport["sdk_max_retries"]),
            campaign_gate=campaign_gate,
        )
        if wire.phase in {"train", "smoke"}:
            if train is None or validation is None or initial_skill is None:
                raise AssertionError("GEPA optimization inputs are incomplete")
            result = _run_optimization(
                wire=wire,
                config=config,
                train=train,
                validation=validation,
                initial_skill=initial_skill,
                initial_digest=initial_digest,
                observer=observer,
                configured_budget=budget,
            )
        else:
            result = _run_frozen_evaluation(
                wire=wire,
                config=config,
                train=train,
                test=test,
                observer=observer,
            )
        result["experiment_kind"] = experiment_kind
        write_worker_result(wire, result, passed=True)
        print(json.dumps({
            "passed": True,
            "method": METHOD_ID,
            "phase": wire.phase,
            "run_id": wire.run_id,
            "result_path": wire.result_path,
        }, ensure_ascii=False, indent=2))
        return 0
    except (Exception, ProviderCallExhausted) as exc:
        error = _safe_error(exc, model)
        failure_kind, child_failure_receipt = _gepa_failure_kind(
            exc, observer, wire
        )
        evidence = dict(getattr(exc, "evidence", {}) or {})
        evidence["failure_kind"] = failure_kind
        if child_failure_receipt is not None:
            evidence["child_provider_failure"] = child_failure_receipt
        if observer is not None and wire is not None:
            try:
                event_view = _provider_event_view(
                    phase_dir=_phase_dir(wire), observer=observer
                )
                evidence.setdefault("provider_calls", len(event_view.events()))
                usage_path = _phase_dir(wire) / "usage.json"
                if not usage_path.exists() and event_view.events():
                    partial = _provider_usage(
                        event_view,
                        wire=wire,
                        wall_time_ms=int((time.monotonic() - started) * 1000),
                        allow_failed=True,
                    )
                    partial.save(usage_path)
                    evidence["partial_usage"] = partial.to_dict()
            except Exception:
                evidence.setdefault("provider_calls", observer.event_cursor())
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
            except Exception as result_exc:
                error += "; worker result write failed: " + _safe_error(result_exc, model)
        print(json.dumps({
            "passed": False,
            "method": wire.method if wire else METHOD_ID,
            "phase": wire.phase if wire else "unknown",
            "run_id": wire.run_id if wire else "",
            "error_type": type(exc).__name__,
            "error": error,
            "failure_kind": failure_kind,
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
