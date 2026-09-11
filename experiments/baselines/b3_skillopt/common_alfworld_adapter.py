"""Common-manifest ALFWorld adapter for the SkillOpt ReflACT trainer.

This adapter implements the upstream :class:`~skillopt.envs.base.EnvAdapter`
contract (section 16.3 of the baseline design document):

- Train batches come exclusively from the common Train manifest;
- selection/validation batches come exclusively from the common Validation
  manifest (SkillOpt's gate split);
- every episode runs on the exact manifest gamefile;
- target and optimizer LLMs share the frozen common base model (the worker
  configures the upstream ``openai_compatible`` backend before training);
- hidden expert reference material is disabled (``build_reference_text``
  returns ``""``, section 16.6);
- every episode is mirrored into a Common sidecar for unified accounting.

The upstream trainer, reflection, aggregation, optimizer, slow update, meta
skill, and gate are all reused unchanged; only the environment entry point
is adapted.
"""

from __future__ import annotations

import concurrent.futures
import base64
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.prompts import load_prompt

from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.trace import load_episodes

from .episode_runner import (
    EpisodeOutcome,
    SkillOptTextEpisodeRunner,
    _validate_episode_payload,
)
from .provider_observer import active_provider_observer


_ROLLOUT_RECEIPT_SCHEMA_VERSION = 1
_EPISODE_CACHE_SCHEMA_VERSION = 1
_RESULTS_NAME = "results.jsonl"
_EPISODES_NAME = "common_episodes.jsonl"
_ACTIONS_NAME = "common_environment_actions.jsonl"
_RECEIPT_NAME = "rollout_receipt.json"
_FAILURE_NAME = "rollout_failure.json"


class RolloutInfrastructureError(RuntimeError):
    """An infrastructure failure aborted a batch before SkillOpt could learn."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str = "protocol_failure",
    ) -> None:
        super().__init__(message)
        self.failure_kind = str(failure_kind)


def manifest_task_to_item(task: Any, *, phase_label: str) -> dict[str, Any]:
    """One manifest task as a SkillOpt batch item (id/gamefile identity only)."""

    return {
        "id": task.task_id,
        "task_id": task.task_id,
        "gamefile": task.gamefile_rel,
        "gamefile_sha256": task.gamefile_sha256,
        "task_signature": task.task_signature,
        "task_type": task.task_type,
        "source_split": task.source_split,
        "env_index": task.env_index,
        "manifest_index": task.index,
        "phase": phase_label,
    }


_SPLIT_PHASES = {
    "train": "train",
    "valid_seen": "validation",
    "selection": "validation",
    "val": "validation",
    "valid_unseen": "test",
    "test": "test",
}


class CommonSkillOptDataLoader(BaseDataLoader):
    """Manifest-backed batch planner for the SkillOpt trainer.

    The epoch shuffle uses the same deterministic scheme as the upstream
    ``SplitDataLoader`` (``random.Random(seed + epoch * 1000)``); batches are
    contiguous chunks of the shuffled train manifest.
    """

    def __init__(
        self,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet,
        test_manifest: TaskManifestSet | None = None,
    ) -> None:
        self.train_items = [
            manifest_task_to_item(task, phase_label="train")
            for task in train_manifest.tasks
        ]
        self.val_items = [
            manifest_task_to_item(task, phase_label="validation")
            for task in validation_manifest.tasks
        ]
        self.test_items = (
            [manifest_task_to_item(task, phase_label="test") for task in test_manifest.tasks]
            if test_manifest is not None
            else []
        )

    def _items_for_split(self, split: str) -> list[dict[str, Any]]:
        normalized = str(split).strip().lower()
        if normalized not in _SPLIT_PHASES:
            raise ValueError(f"unsupported split for SkillOpt baseline: {split}")
        phase = _SPLIT_PHASES[normalized]
        if phase == "train":
            return self.train_items
        if phase == "validation":
            return self.val_items
        if phase == "test":
            if not self.test_items:
                raise ValueError(
                    "no common Test manifest is available for this run phase"
                )
            return self.test_items
        raise ValueError(f"unsupported split for SkillOpt baseline: {split}")

    def get_train_size(self) -> int:
        return len(self.train_items)

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs: Any,
    ) -> list[BatchSpec]:
        import random

        epoch_rng = random.Random(seed + epoch * 1000)
        items = list(self.train_items)
        epoch_rng.shuffle(items)
        total_batches = steps_per_epoch * accumulation
        batches: list[BatchSpec] = []
        cursor = 0
        for batch_index in range(total_batches):
            batch_items = items[cursor: cursor + batch_size]
            cursor += len(batch_items)
            if not batch_items and items:
                refill_rng = random.Random(seed + epoch * 1000 + batch_index + 1)
                batch_items = list(items)
                refill_rng.shuffle(batch_items)
                batch_items = batch_items[:batch_size]
            batches.append(BatchSpec(
                phase="train",
                split="train",
                seed=seed + epoch * 1000 + batch_index + 1,
                batch_size=len(batch_items),
                payload=batch_items,
            ))
        return batches

    def build_train_batch(self, batch_size: int, seed: int, **kwargs: Any) -> BatchSpec:
        import random

        rng = random.Random(seed)
        items = list(self.train_items)
        rng.shuffle(items)
        items = items[:batch_size]
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs: Any,
    ) -> BatchSpec:
        items = self._items_for_split(split)
        if env_num and env_num < len(items):
            items = items[:env_num]
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )


class CommonSkillOptBatchRun:
    """Lazy batch description consumed by the trainer + slow-update paths."""

    def __init__(self, tasks: list[dict[str, Any]], *, seed: int) -> None:
        self.tasks = tasks
        self.seed = seed

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


class CommonALFWorldSkillOptAdapter(EnvAdapter):
    """SkillOpt EnvAdapter driven by the common manifests."""

    def __init__(
        self,
        *,
        train_manifest_path: str | Path,
        validation_manifest_path: str | Path,
        test_manifest_path: str | Path | None = None,
        alfworld_data: str = "",
        max_steps: int = 100,
        workers: int = 1,
        max_api_workers: int = 1,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        max_completion_tokens: int = 16384,
        steps_per_epoch: int = 3,
        seed: int = 42,
        phase: str = "train",
        episode_runner: Any | None = None,
        run_id: str | None = None,
        identity: dict[str, str] | None = None,
        campaign: dict[str, Any] | None = None,
        resume: dict[str, Any] | None = None,
        artifact_digest_override: str | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.workers = max(int(workers or 1), 1)
        self.max_api_workers = max_api_workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.steps_per_epoch = int(steps_per_epoch)
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.seed = seed
        self.phase = phase
        self._injected_episode_runner = episode_runner is not None
        self.alfworld_data = alfworld_data or os.environ.get(
            "ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"),
        )
        self.train_manifest = TaskManifestSet.load(Path(train_manifest_path))
        self.validation_manifest = TaskManifestSet.load(Path(validation_manifest_path))
        self.test_manifest = (
            TaskManifestSet.load(Path(test_manifest_path))
            if test_manifest_path
            else None
        )
        self.run_id = str(run_id or "").strip()
        if not self.run_id:
            if not self._injected_episode_runner:
                raise ValueError("production SkillOpt adapter requires an explicit run_id")
            self.run_id = "unit_test"
        self.identity = self._resolve_identity(identity)
        if campaign is not None and not isinstance(campaign, dict):
            raise TypeError("campaign must be a mapping when supplied")
        if resume is not None and not isinstance(resume, dict):
            raise TypeError("resume must be a mapping when supplied")
        self.campaign = dict(campaign) if campaign is not None else None
        self.resume = dict(resume) if resume is not None else None
        self.artifact_digest_override = _optional_sha256(
            artifact_digest_override,
            field="artifact_digest_override",
        )
        frozen_digest = self.identity.get("frozen_artifact_digest")
        if frozen_digest is not None:
            _optional_sha256(frozen_digest, field="identity.frozen_artifact_digest")
        if (
            self.artifact_digest_override is not None
            and frozen_digest is not None
            and self.phase in {"train_eval", "test"}
            and self.artifact_digest_override != frozen_digest
        ):
            raise ValueError(
                "artifact_digest_override does not match identity.frozen_artifact_digest"
            )
        # A new adapter process/session may never silently resume an older
        # rollout, even if a caller accidentally reuses the same run_id.
        self._session_id = uuid.uuid4().hex
        self.dataloader = CommonSkillOptDataLoader(
            self.train_manifest, self.validation_manifest, self.test_manifest,
        )
        self._episode_runner = episode_runner or SkillOptTextEpisodeRunner(
            max_actions=self.max_steps,
            max_completion_tokens=self.max_completion_tokens,
            seed=self.seed,
            alfworld_data=self.alfworld_data,
        )

    # ── EnvAdapter contract ───────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)

    def get_dataloader(self) -> BaseDataLoader:
        return self.dataloader

    def requires_ray(self) -> bool:
        return False

    def build_env_from_batch(self, batch: BatchSpec, **kwargs: Any):
        tasks = list(batch.payload or [])
        return CommonSkillOptBatchRun(tasks, seed=batch.seed)

    def build_train_env(self, batch_size: int, seed: int, **kwargs: Any):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size, seed=seed, **kwargs,
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs: Any):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs,
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_train_evaluation_env(self, *, seed: int) -> CommonSkillOptBatchRun:
        """Exact Train manifest once, labelled separately from optimizer rollouts."""

        tasks = [
            {**item, "phase": "train_eval"}
            for item in self.dataloader.train_items
        ]
        return CommonSkillOptBatchRun(tasks, seed=seed)

    def build_test_evaluation_env(self, *, seed: int) -> CommonSkillOptBatchRun:
        """Exact held-out Test manifest once, with no optimizer state."""

        if not self.dataloader.test_items:
            raise ValueError("test evaluation requires the common Test manifest")
        tasks = [{**item, "phase": "test"} for item in self.dataloader.test_items]
        return CommonSkillOptBatchRun(tasks, seed=seed)

    def build_smoke_env(
        self,
        *,
        seed: int,
        task_count: int = 1,
    ) -> CommonSkillOptBatchRun:
        """Small real-IO probe drawn only from Train, never Validation/Test."""

        if task_count <= 0 or task_count > len(self.dataloader.train_items):
            raise ValueError("smoke task_count must be within the Train manifest")
        tasks = [
            {**item, "phase": "smoke"}
            for item in self.dataloader.train_items[:task_count]
        ]
        return CommonSkillOptBatchRun(tasks, seed=seed)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run one identity-bound batch and commit its evidence transactionally."""

        tasks = [dict(task) for task in list(env_manager.tasks)]
        skill_digest = hashlib.sha256(skill_content.encode("utf-8")).hexdigest()
        phase = _single_phase(tasks)
        artifact_digest = self._artifact_digest_for_phase(
            phase=phase,
            skill_digest=skill_digest,
        )
        request = self._rollout_request(
            tasks=tasks,
            skill_digest=skill_digest,
            artifact_digest=artifact_digest,
            batch_seed=int(getattr(env_manager, "seed", self.seed)),
        )
        rollout_id = "rollout_" + sha256_json(request)[:32]
        request["rollout_id"] = rollout_id
        observer = active_provider_observer()
        out_path = Path(out_dir)
        if (out_path / _RECEIPT_NAME).is_file():
            cached_provider_events = (
                observer.events_since(0, rollout_id=rollout_id)
                if observer is not None else []
            )
            return _load_committed_rollout(
                out_path,
                request,
                provider_events=cached_provider_events,
                require_provider=not self._injected_episode_runner,
            )
        if (out_path / _FAILURE_NAME).exists():
            raise RuntimeError(
                f"refusing to reuse an infrastructure-failed rollout: "
                f"{out_path / _FAILURE_NAME}"
            )
        out_path.mkdir(parents=True, exist_ok=True)
        unexpected = sorted(
            path.name for path in out_path.iterdir() if path.name != "episodes"
        )
        if unexpected:
            raise RuntimeError(
                "rollout directory is partial or stale outside the verified episode "
                f"cache: {out_path}: {unexpected}"
            )

        provider_cursor = observer.event_cursor() if observer is not None else 0

        def execute_episode(task: dict[str, Any]) -> dict[str, Any]:
            cache_key = _episode_cache_key(
                request=request,
                task=task,
                rollout_path=out_path,
                steps_per_epoch=self.steps_per_epoch,
                allow_implicit_zero=self._injected_episode_runner,
            )
            try:
                cached = _load_episode_cache(
                    out_path=out_path,
                    task=task,
                    expected_cache_key=cache_key,
                )
                if cached is not None:
                    cached_row, cached_episode, cached_actions = (
                        _validate_cached_episode(
                            task=task,
                            request=request,
                            cached=cached,
                            require_provider=not self._injected_episode_runner,
                        )
                    )
                    if observer is None and not self._injected_episode_runner:
                        raise RuntimeError(
                            "provider observer is required to import episode cache usage"
                        )
                    _restore_cached_conversation(out_path, task, cached)
                    if observer is not None:
                        observer.import_cached_events(
                            list(cached["provider_events"]),
                            rollout_id=rollout_id,
                            task_id=str(task["task_id"]),
                        )
                    return {
                        "ok": True,
                        "row": cached_row,
                        "episode": cached_episode,
                        "actions": cached_actions,
                        "cached": True,
                    }

                outcome = self._episode_runner.run(
                    task, skill_content, str(out_path), rollout_id=rollout_id,
                )
                if outcome.infrastructure_failure:
                    return {"ok": False, "outcome": outcome}
                row = _normalize_result_row(task, outcome.skillopt_row)
                episode = _episode_record(
                    task=task,
                    outcome=outcome,
                    phase=str(task.get("phase", self.phase)),
                    run_seed=self.seed,
                    skill_digest=skill_digest,
                    artifact_digest=artifact_digest,
                )
                actions = _action_events(task, outcome.conversation)
                if len(actions) != episode.environment_actions:
                    raise RolloutInfrastructureError(
                        f"action evidence count mismatch for task {task.get('id')!r}"
                    )
                task_provider_events = (
                    [
                        event
                        for event in observer.events_since(0, rollout_id=rollout_id)
                        if str(event.get("episode_task_id", ""))
                        == str(task.get("task_id", task.get("id", "")))
                    ]
                    if observer is not None else []
                )
                provider_error = _provider_evidence_error(
                    task_provider_events,
                    tasks=[task],
                    required=not self._injected_episode_runner,
                )
                if provider_error is not None:
                    raise RolloutInfrastructureError(
                        f"invalid per-episode provider evidence: {provider_error}",
                        failure_kind="protocol_failure",
                    )
                _reconcile_episode_provider_usage(
                    [episode],
                    task_provider_events,
                    required=not self._injected_episode_runner,
                    validate_persisted_reasoning=False,
                )
                _persist_episode_cache(
                    out_path=out_path,
                    task=task,
                    cache_key=cache_key,
                    row=row,
                    episode=episode,
                    actions=actions,
                    conversation=outcome.conversation,
                    provider_events=task_provider_events,
                )
                return {
                    "ok": True,
                    "row": row,
                    "episode": episode,
                    "actions": actions,
                    "cached": False,
                }
            except Exception as exc:
                failure_kind = str(
                    getattr(exc, "failure_kind", "protocol_failure")
                )
                if failure_kind not in {
                    "infrastructure_failure", "protocol_failure",
                }:
                    failure_kind = "protocol_failure"
                outcome = EpisodeOutcome(
                    task=dict(task),
                    skillopt_row={
                        "id": str(task.get("id", "")), "hard": 0, "soft": 0.0,
                    },
                    conversation=[],
                    infrastructure_failure=True,
                    infrastructure_error=_redacted_error(exc),
                    failure_kind=failure_kind,
                )
                return {"ok": False, "outcome": outcome}

        parallelism = min(
            len(tasks),
            max(1, int(self.workers)),
            max(1, int(self.max_api_workers or 1)),
        )
        if parallelism == 1:
            results = [execute_episode(task) for task in tasks]
        else:
            # Each task owns an independent ALFWorld environment.  Futures are
            # consumed in manifest order so learning inputs and persisted rows
            # remain byte-order deterministic even when completion order varies.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parallelism,
                thread_name_prefix="asg-skillopt-episode",
            ) as executor:
                futures = [
                    executor.submit(execute_episode, task) for task in tasks
                ]
                results = [future.result() for future in futures]

        successful = [result for result in results if result.get("ok") is True]
        rows = [dict(result["row"]) for result in successful]
        episodes = [result["episode"] for result in successful]
        action_events = [
            dict(action)
            for result in successful
            for action in result["actions"]
        ]
        failures = [
            (task, result["outcome"])
            for task, result in zip(tasks, results, strict=True)
            if result.get("ok") is not True
        ]

        provider_events = (
            observer.events_since(provider_cursor, rollout_id=rollout_id)
            if observer is not None else []
        )
        if failures:
            failed_task, failed_outcome = failures[0]
            failed_episode = _episode_record(
                task=failed_task,
                outcome=failed_outcome,
                phase=str(failed_task.get("phase", self.phase)),
                run_seed=self.seed,
                skill_digest=skill_digest,
                artifact_digest=artifact_digest,
            )
            _persist_rollout_failure(
                out_path=out_path,
                request=request,
                completed_rows=rows,
                completed_episodes=episodes,
                completed_actions=action_events,
                failed_episode=failed_episode,
                provider_events=provider_events,
            )
            raise RolloutInfrastructureError(
                f"SkillOpt rollout {rollout_id} aborted on task "
                f"{failed_task.get('id')!r}: {failed_outcome.infrastructure_error}",
                failure_kind=(
                    failed_outcome.failure_kind or "protocol_failure"
                ),
            )

        # Cached and newly executed records are assembled strictly in manifest
        # order, independent of thread completion order.
        if [episode.task_id for episode in episodes] != [
            str(task["task_id"]) for task in tasks
        ]:
            raise RolloutInfrastructureError(
                "episode cache aggregation changed manifest order"
            )
        provider_error = _provider_evidence_error(
            provider_events,
            tasks=tasks,
            required=not self._injected_episode_runner,
        )
        if provider_error is None:
            try:
                _reconcile_episode_provider_usage(
                    episodes,
                    provider_events,
                    required=not self._injected_episode_runner,
                    validate_persisted_reasoning=False,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                provider_error = str(exc)
        if provider_error is not None:
            # This should normally have been caught by EpisodeOutcome.  Keep a
            # second boundary here so no swallowed target failure can reach the
            # trainer even if an upstream control-flow detail changes.
            failed_episode = _episode_record(
                task=tasks[-1],
                outcome=_provider_failure_outcome(tasks[-1]),
                phase=str(tasks[-1].get("phase", self.phase)),
                run_seed=self.seed,
                skill_digest=skill_digest,
                artifact_digest=artifact_digest,
            )
            _persist_rollout_failure(
                out_path=out_path,
                request=request,
                completed_rows=rows,
                completed_episodes=episodes,
                completed_actions=action_events,
                failed_episode=failed_episode,
                provider_events=provider_events,
            )
            raise RolloutInfrastructureError(
                f"SkillOpt rollout {rollout_id} has invalid provider evidence: "
                f"{provider_error}"
            )

        _commit_rollout(
            out_path=out_path,
            request=request,
            rows=rows,
            episodes=episodes,
            action_events=action_events,
            provider_events=provider_events,
        )
        return rows

    def _resolve_identity(self, identity: dict[str, str] | None) -> dict[str, str]:
        config_payload = {
            "max_steps": int(self.max_steps),
            "max_completion_tokens": int(self.max_completion_tokens),
            "workers": int(self.workers),
            "max_api_workers": int(self.max_api_workers),
            "analyst_workers": int(self.analyst_workers),
            "failure_only": bool(self.failure_only),
            "minibatch_size": int(self.minibatch_size),
            "edit_budget": int(self.edit_budget),
            "seed": int(self.seed),
        }
        derived = {
            "config_digest": sha256_json(config_payload),
            "train_manifest_digest": self.train_manifest.digest,
            "validation_manifest_digest": self.validation_manifest.digest,
        }
        if self.test_manifest is not None:
            derived["test_manifest_digest"] = self.test_manifest.digest
        if identity is None:
            if not self._injected_episode_runner:
                raise ValueError(
                    "production SkillOpt adapter requires immutable run identity"
                )
            return derived
        resolved = {str(key): str(value) for key, value in dict(identity).items()}
        for key in ("config_digest", "train_manifest_digest", "validation_manifest_digest"):
            if key not in resolved:
                raise ValueError(f"SkillOpt adapter identity is missing {key}")
        for key, value in resolved.items():
            if key.endswith("_digest"):
                _optional_sha256(value, field=f"identity.{key}")
        for key in ("train_manifest_digest", "validation_manifest_digest"):
            if resolved[key] != derived[key]:
                raise ValueError(
                    f"SkillOpt adapter identity {key} does not match loaded manifest"
                )
        if self.test_manifest is not None and (
            resolved.get("test_manifest_digest") != self.test_manifest.digest
        ):
            raise ValueError(
                "SkillOpt adapter identity test_manifest_digest does not match loaded manifest"
            )
        return resolved

    def _rollout_request(
        self,
        *,
        tasks: list[dict[str, Any]],
        skill_digest: str,
        artifact_digest: str,
        batch_seed: int,
    ) -> dict[str, Any]:
        if not tasks:
            raise ValueError("SkillOpt rollout batch must contain at least one task")
        phases = {str(task.get("phase", "")).strip() for task in tasks}
        if len(phases) != 1 or "" in phases:
            raise ValueError(f"SkillOpt rollout tasks have inconsistent phases: {phases}")
        phase = next(iter(phases))
        manifest = self._manifest_for_phase(phase)
        expected = {task.task_id: task for task in manifest.tasks}
        task_ids = [str(task.get("id", "")) for task in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("SkillOpt rollout batch contains duplicate task ids")
        task_identity: list[dict[str, Any]] = []
        for item in tasks:
            task_id = str(item.get("id", ""))
            entry = expected.get(task_id)
            if entry is None:
                raise ValueError(
                    f"SkillOpt rollout task {task_id!r} is outside its {phase} manifest"
                )
            observed = {
                "task_id": str(item.get("task_id", "")),
                "task_type": str(item.get("task_type", "")),
                "source_split": str(item.get("source_split", "")),
                "env_index": int(item.get("env_index", -1)),
                "manifest_index": int(item.get("manifest_index", -1)),
                "gamefile": str(item.get("gamefile", "")),
                "gamefile_sha256": str(item.get("gamefile_sha256", "")),
                "task_signature": str(item.get("task_signature", "")),
            }
            authoritative = {
                "task_id": entry.task_id,
                "task_type": entry.task_type,
                "source_split": entry.source_split,
                "env_index": entry.env_index,
                "manifest_index": entry.index,
                "gamefile": entry.gamefile_rel,
                "gamefile_sha256": entry.gamefile_sha256,
                "task_signature": entry.task_signature,
            }
            if observed != authoritative:
                raise ValueError(
                    f"SkillOpt rollout task {task_id!r} identity differs from manifest"
                )
            task_identity.append(authoritative)
        observer = active_provider_observer()
        provider_identity = (
            {
                "model": str(observer.model),
                "reasoning_effort": str(observer.reasoning_effort),
            }
            if observer is not None else None
        )
        return {
            "schema_version": _ROLLOUT_RECEIPT_SCHEMA_VERSION,
            "method": "b3_skillopt",
            "run_id": self.run_id,
            "run_seed": int(self.seed),
            "adapter_session_id": self._session_id,
            "phase": phase,
            "batch_seed": int(batch_seed),
            "identity": dict(self.identity),
            "provider_identity": provider_identity,
            "active_manifest_digest": manifest.digest,
            "skill_sha256": skill_digest,
            "artifact_digest": artifact_digest,
            "tasks": task_identity,
        }

    def _artifact_digest_for_phase(
        self,
        *,
        phase: str,
        skill_digest: str,
    ) -> str:
        """Resolve Common evidence identity without losing the skill byte hash.

        Optimizer/smoke episodes identify the exact skill bytes they executed.
        An immutable final replay instead identifies the complete
        frozen artifact directory, while retaining ``skill_sha256`` separately
        in the rollout request and episode method metrics.
        """

        if phase in {"train_eval", "test"}:
            frozen_digest = self.identity.get("frozen_artifact_digest")
            artifact_digest = self.artifact_digest_override or frozen_digest
            if artifact_digest is None:
                raise ValueError(
                    f"{phase} requires identity.frozen_artifact_digest or "
                    "artifact_digest_override"
                )
            if frozen_digest is not None and artifact_digest != frozen_digest:
                raise ValueError(
                    f"{phase} artifact digest differs from immutable run identity"
                )
            return artifact_digest
        if (
            self.artifact_digest_override is not None
            and self.artifact_digest_override != skill_digest
        ):
            raise ValueError(
                "artifact_digest_override is only distinct from skill_sha256 "
                f"during {phase}"
            )
        return skill_digest

    def _manifest_for_phase(self, phase: str) -> TaskManifestSet:
        if phase in {"train", "train_eval", "smoke"}:
            return self.train_manifest
        if phase in {"validation", "validation_eval"}:
            return self.validation_manifest
        if phase == "test" and self.test_manifest is not None:
            return self.test_manifest
        raise ValueError(f"unsupported SkillOpt rollout phase: {phase}")

    # ── Reference material guard (design doc §16.6) ───────────────────────

    def build_reference_text(self, item: dict) -> str:
        """Main comparison setting: no hidden expert plan/reference material."""
        return ""

    def get_reference_metadata(self, item: dict) -> dict:
        return {"fields": [], "preview": ""}

    # ── Prompts: always the upstream ALFWorld analyst prompts ─────────────

    def get_error_minibatch_prompt(self) -> str | None:
        return load_prompt("analyst_error", env="alfworld")

    def get_success_minibatch_prompt(self) -> str | None:
        return load_prompt("analyst_success", env="alfworld")

    def get_task_types(self) -> list[str]:
        from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES

        return list(ALFWORLD_FORMAL_TASK_TYPES)


def _load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SkillOpt results are corrupt at line {line_number}: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"SkillOpt result line {line_number} is not a mapping: {path}"
            )
        rows.append(dict(payload))
    return rows


def _normalize_result_row(
    task: dict[str, Any],
    raw_row: dict[str, Any],
) -> dict[str, Any]:
    row = dict(raw_row)
    task_id = str(task.get("id", ""))
    if str(row.get("id", "")) != task_id:
        raise RolloutInfrastructureError(
            f"SkillOpt result id mismatch for {task_id!r}: {row.get('id')!r}"
        )
    hard = row.get("hard")
    if isinstance(hard, bool):
        hard_value = int(hard)
    elif isinstance(hard, int) and hard in {0, 1}:
        hard_value = hard
    else:
        raise RolloutInfrastructureError(
            f"SkillOpt result for {task_id!r} has non-binary hard score"
        )
    try:
        soft_value = float(row["soft"])
        turns = int(row["n_turns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RolloutInfrastructureError(
            f"SkillOpt result for {task_id!r} lacks numeric soft/n_turns evidence"
        ) from exc
    if turns < 0:
        raise RolloutInfrastructureError(
            f"SkillOpt result for {task_id!r} has negative n_turns"
        )
    reported_task_type = str(row.get("task_type", ""))
    # Latest upstream SkillOpt retains its historical short label for the
    # simple placement family.  This is an explicit taxonomy alias only; all
    # other mismatches still fail closed after the reset gamefile was verified.
    canonical_task_type = {
        "pick_and_place": "pick_and_place_simple",
    }.get(reported_task_type, reported_task_type)
    if canonical_task_type != str(task.get("task_type", "")):
        raise RolloutInfrastructureError(
            f"SkillOpt result task_type mismatch for {task_id!r}"
        )
    if not str(row.get("gamefile", "")).strip():
        raise RolloutInfrastructureError(
            f"SkillOpt result for {task_id!r} has no reset gamefile"
        )
    if row.get("agent_ok") is not True:
        raise RolloutInfrastructureError(
            f"SkillOpt result for {task_id!r} reports agent_ok != true"
        )
    row.update({
        "id": task_id,
        "hard": hard_value,
        "soft": soft_value,
        "n_turns": turns,
        "fail_reason": str(row.get("fail_reason", "")),
        "task_description": str(row.get("task_description", "")),
        "task_type": canonical_task_type,
    })
    return row


def _single_phase(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        raise ValueError("SkillOpt rollout batch must contain at least one task")
    phases = {str(task.get("phase", "")).strip() for task in tasks}
    if len(phases) != 1 or "" in phases:
        raise ValueError(f"SkillOpt rollout tasks have inconsistent phases: {phases}")
    return next(iter(phases))


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _action_events(
    task: dict[str, Any],
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task_id = str(task.get("id", ""))
    return [
        {
            "episode_task_id": task_id,
            "step_index": step_index,
            "action": str(step["action"]),
            "env_feedback": str(step.get("env_feedback", "")),
            "reward": float(step.get("reward", 0.0)),
            "done": bool(step.get("done", False)),
        }
        for step_index, step in enumerate(conversation)
    ]


def _optimizer_step(
    rollout_path: Path,
    *,
    steps_per_epoch: int,
    allow_implicit_zero: bool,
) -> int:
    """Return the immutable optimizer step encoded by the upstream path."""

    parts = rollout_path.parts
    for index in range(len(parts) - 1):
        match = re.fullmatch(r"step_(\d{4})", parts[index + 1])
        if parts[index] == "steps" and match is not None:
            return int(match.group(1))
    for index in range(len(parts) - 1):
        match = re.fullmatch(r"epoch_(\d{2})", parts[index + 1])
        if parts[index] in {"slow_update", "meta_skill"} and match is not None:
            return int(match.group(1)) * int(steps_per_epoch)
    if any(part == "selection_eval_baseline" for part in parts):
        return 0
    if allow_implicit_zero:
        return 0
    raise RolloutInfrastructureError(
        f"cannot prove optimizer-step authority from rollout path: {rollout_path}"
    )


def _episode_cache_dir(out_path: Path, task: dict[str, Any]) -> Path:
    task_id = str(task.get("task_id", task.get("id", ""))).strip()
    if not task_id or re.fullmatch(r"[A-Za-z0-9_.-]+", task_id) is None:
        raise RolloutInfrastructureError(
            f"unsafe episode cache task id: {task_id!r}"
        )
    return out_path / "episodes" / task_id


def _episode_cache_key(
    *,
    request: dict[str, Any],
    task: dict[str, Any],
    rollout_path: Path,
    steps_per_epoch: int,
    allow_implicit_zero: bool,
) -> str:
    """Hash every authority that may affect one episode's exact result."""

    task_id = str(task.get("task_id", task.get("id", "")))
    model_digest = str(
        dict(request.get("identity") or {}).get("model_identity_digest", "")
    )
    if model_digest and _optional_sha256(
        model_digest, field="identity.model_identity_digest"
    ) is None:
        raise AssertionError("unreachable invalid model identity digest")
    payload = {
        "schema_version": _EPISODE_CACHE_SCHEMA_VERSION,
        "method": str(request.get("method", "")),
        # run_id and adapter_session_id are deliberately excluded: formal
        # resume creates a new attempt while preserving this immutable identity.
        "run_identity": dict(request.get("identity") or {}),
        "run_seed": int(request.get("run_seed", -1)),
        "optimizer_step": _optimizer_step(
            rollout_path,
            steps_per_epoch=steps_per_epoch,
            allow_implicit_zero=(
                allow_implicit_zero
                or str(request.get("phase", ""))
                in {"smoke", "train_eval", "test"}
            ),
        ),
        "phase": str(request.get("phase", "")),
        "batch_seed": int(request.get("batch_seed", -1)),
        "task_id": task_id,
        "skill_sha256": str(request.get("skill_sha256", "")),
        "gamefile_sha256": str(task.get("gamefile_sha256", "")),
        "model_identity_digest": model_digest,
        "provider_identity": request.get("provider_identity"),
        "active_manifest_digest": str(
            request.get("active_manifest_digest", "")
        ),
    }
    for field in (
        "skill_sha256", "gamefile_sha256", "active_manifest_digest",
    ):
        _optional_sha256(payload[field], field=f"episode_cache.{field}")
    return sha256_json(payload)


def _persist_episode_cache(
    *,
    out_path: Path,
    task: dict[str, Any],
    cache_key: str,
    row: dict[str, Any],
    episode: CommonEpisodeRecord,
    actions: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    provider_events: list[dict[str, Any]],
) -> None:
    """Commit one successful episode; ``receipt.json`` is the sole marker."""

    cache_dir = _episode_cache_dir(out_path, task)
    receipt_path = cache_dir / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    conversation_path = (
        out_path / "predictions" / str(task["task_id"]) / "conversation.json"
    )
    if not conversation_path.is_file():
        raise RolloutInfrastructureError(
            f"episode conversation is missing before cache commit: {conversation_path}"
        )
    conversation_bytes = conversation_path.read_bytes()
    try:
        persisted_conversation = json.loads(conversation_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutInfrastructureError(
            f"episode conversation is unreadable before cache commit: "
            f"{conversation_path}"
        ) from exc
    if persisted_conversation != conversation:
        raise RolloutInfrastructureError(
            "episode conversation bytes differ from the validated conversation"
        )

    payload = {
        "row": dict(row),
        "episode": episode.to_dict(),
        "actions": [dict(item) for item in actions],
        "provider_events": [dict(item) for item in provider_events],
    }
    payload_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt = {
        "schema_version": _EPISODE_CACHE_SCHEMA_VERSION,
        "status": "completed",
        "task_id": str(task["task_id"]),
        "cache_key": cache_key,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
        "conversation_sha256": hashlib.sha256(conversation_bytes).hexdigest(),
        "conversation_base64": base64.b64encode(conversation_bytes).decode("ascii"),
    }
    _write_json_atomic(receipt_path, receipt)


def _load_episode_cache(
    *,
    out_path: Path,
    task: dict[str, Any],
    expected_cache_key: str,
) -> dict[str, Any] | None:
    cache_dir = _episode_cache_dir(out_path, task)
    receipt_path = cache_dir / "receipt.json"
    if not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutInfrastructureError(
            f"episode cache receipt is unreadable: {receipt_path}"
        ) from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != _EPISODE_CACHE_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("task_id") != str(task["task_id"])
        or receipt.get("cache_key") != expected_cache_key
    ):
        raise RolloutInfrastructureError(
            f"episode cache receipt identity is invalid: {receipt_path}"
        )
    try:
        payload_bytes = base64.b64decode(
            str(receipt["payload_base64"]), validate=True
        )
        conversation_bytes = base64.b64decode(
            str(receipt["conversation_base64"]), validate=True
        )
    except (KeyError, ValueError) as exc:
        raise RolloutInfrastructureError(
            f"episode cache receipt bytes are invalid: {receipt_path}"
        ) from exc
    if (
        hashlib.sha256(payload_bytes).hexdigest()
        != receipt.get("payload_sha256")
        or hashlib.sha256(conversation_bytes).hexdigest()
        != receipt.get("conversation_sha256")
    ):
        raise RolloutInfrastructureError(
            f"episode cache receipt hash mismatch: {receipt_path}"
        )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
        conversation = json.loads(conversation_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutInfrastructureError(
            f"episode cache payload is unreadable: {receipt_path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "row", "episode", "actions", "provider_events",
    }:
        raise RolloutInfrastructureError(
            f"episode cache payload inventory is invalid: {receipt_path}"
        )
    if not isinstance(conversation, list):
        raise RolloutInfrastructureError(
            f"episode cache conversation is invalid: {receipt_path}"
        )
    for field in ("row", "episode"):
        if not isinstance(payload[field], dict):
            raise RolloutInfrastructureError(
                f"episode cache {field} is invalid: {receipt_path}"
            )
    for field in ("actions", "provider_events"):
        if not isinstance(payload[field], list) or any(
            not isinstance(item, dict) for item in payload[field]
        ):
            raise RolloutInfrastructureError(
                f"episode cache {field} is invalid: {receipt_path}"
            )
    return {
        **payload,
        "conversation": conversation,
        "conversation_bytes": conversation_bytes,
    }


def _validate_cached_episode(
    *,
    task: dict[str, Any],
    request: dict[str, Any],
    cached: dict[str, Any],
    require_provider: bool,
) -> tuple[dict[str, Any], CommonEpisodeRecord, list[dict[str, Any]]]:
    """Revalidate cached bytes against the current immutable authorities."""

    row = _normalize_result_row(task, dict(cached["row"]))
    try:
        episode = CommonEpisodeRecord(**dict(cached["episode"]))
    except (TypeError, ValueError) as exc:
        raise RolloutInfrastructureError(
            "episode cache contains an invalid Common episode"
        ) from exc
    task_id = str(task["task_id"])
    expected = {
        "method": "b3_skillopt",
        "phase": str(request["phase"]),
        "run_seed": int(request["run_seed"]),
        "task_id": task_id,
        "task_type": str(task["task_type"]),
        "manifest_index": int(task["manifest_index"]),
        "gamefile": str(task["gamefile"]),
        "gamefile_hash": str(task["gamefile_sha256"]),
        "artifact_digest_before": str(request["artifact_digest"]),
        "artifact_digest_after": str(request["artifact_digest"]),
    }
    mismatches = [
        field for field, value in expected.items()
        if getattr(episode, field) != value
    ]
    if (
        episode.infrastructure_failure
        or str(episode.method_metrics.get("skill_sha256", ""))
        != str(request["skill_sha256"])
        or str(episode.method_metrics.get("artifact_digest", ""))
        != str(request["artifact_digest"])
    ):
        mismatches.append("episode_status_or_skill")
    conversation = _validate_episode_payload(
        task=task,
        row=row,
        conversation=cached["conversation"],
        actual_gamefile=str(task["gamefile"]),
    )
    actions = [dict(item) for item in cached["actions"]]
    expected_actions = _action_events(task, conversation)
    if actions != expected_actions:
        mismatches.append("actions_or_conversation")
    if (
        int(row["n_turns"]) != len(conversation)
        or episode.environment_actions != len(conversation)
        or episode.command_turns != int(row["n_turns"])
        or episode.official_success != bool(row["hard"])
    ):
        mismatches.append("row_or_episode")
    if mismatches:
        raise RolloutInfrastructureError(
            "episode cache evidence does not match the current request: "
            + ", ".join(sorted(set(mismatches)))
        )
    provider_events = [dict(item) for item in cached["provider_events"]]
    provider_identity = request.get("provider_identity")
    if provider_identity is not None:
        expected_provider = dict(provider_identity)
        if any(
            int(event.get("schema_version", 0)) < 2
            or event.get("method") != "b3_skillopt"
            or int(event.get("run_seed", -1)) != int(request["run_seed"])
            or event.get("model") != expected_provider.get("model")
            or event.get("reasoning_effort")
            != expected_provider.get("reasoning_effort")
            for event in provider_events
        ):
            raise RolloutInfrastructureError(
                "episode cache provider identity is invalid"
            )
    provider_error = _provider_evidence_error(
        provider_events,
        tasks=[task],
        required=require_provider,
    )
    if provider_error is not None:
        raise RolloutInfrastructureError(
            f"episode cache provider evidence is invalid: {provider_error}"
        )
    _reconcile_episode_provider_usage(
        [episode],
        provider_events,
        required=require_provider,
        validate_persisted_reasoning=True,
    )
    _validate_action_coverage([row], [episode], actions)
    return row, episode, actions


def _restore_cached_conversation(
    out_path: Path,
    task: dict[str, Any],
    cached: dict[str, Any],
) -> None:
    target = out_path / "predictions" / str(task["task_id"]) / "conversation.json"
    content = bytes(cached["conversation_bytes"])
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise RolloutInfrastructureError(
                f"cached conversation target already differs: {target}"
            )
        return
    _write_bytes_atomic(target, content)


def _provider_failure_outcome(task: dict[str, Any]) -> EpisodeOutcome:
    return EpisodeOutcome(
        task=dict(task),
        skillopt_row={"id": str(task.get("id", "")), "hard": 0, "soft": 0.0},
        conversation=[],
        infrastructure_failure=True,
        infrastructure_error=(
            "provider observer evidence is missing or contains a failed call"
        ),
        failure_kind="protocol_failure",
    )


def _redacted_error(exc: Exception) -> str:
    message = str(exc)
    live_key = os.environ.get("MODEL_API_KEY", "").strip()
    if live_key:
        message = message.replace(live_key, "<redacted>")
    return f"{type(exc).__name__}: {message[:500]}"


def _provider_evidence_error(
    events: list[dict[str, Any]],
    *,
    tasks: list[dict[str, Any]],
    required: bool,
) -> str | None:
    failures = [event for event in events if event.get("status") != "succeeded"]
    if failures:
        return f"{len(failures)} provider call(s) did not succeed"
    if not required:
        return None
    if not events:
        return "no provider call events were recorded"
    expected_ids = {str(task.get("id", "")) for task in tasks}
    observed_ids = {
        str(event.get("episode_task_id", ""))
        for event in events
        if event.get("role") == "target"
    }
    if observed_ids != expected_ids:
        return (
            "target provider calls do not cover the rollout task set: "
            f"expected={sorted(expected_ids)}, observed={sorted(observed_ids)}"
        )
    if any(event.get("role") != "target" for event in events):
        return "rollout provider evidence contains non-target calls"
    if any(int(event.get("total_tokens", 0)) <= 0 for event in events):
        return "a provider call has no positive token usage"
    return None


def _reconcile_episode_provider_usage(
    episodes: list[CommonEpisodeRecord],
    events: list[dict[str, Any]],
    *,
    required: bool,
    validate_persisted_reasoning: bool,
) -> None:
    """Reconcile per-episode tracker deltas with provider-call evidence.

    OpenAI-compatible ``completion_tokens`` already includes reasoning tokens;
    the latter is retained as a disaggregated field and is never added to the
    completion count here.
    """

    if not events and not required:
        return
    by_task: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("status") != "succeeded" or event.get("role") != "target":
            raise RuntimeError(
                "episode usage reconciliation received failed or non-target evidence"
            )
        task_id = str(event.get("episode_task_id", "")).strip()
        if not task_id:
            raise RuntimeError("provider-call evidence has no episode_task_id")
        by_task.setdefault(task_id, []).append(event)

    for episode in episodes:
        task_events = by_task.pop(episode.task_id, [])
        if required and not task_events:
            raise RuntimeError(
                f"episode {episode.task_id!r} has no target-provider call evidence"
            )
        if not task_events:
            continue
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        reasoning_statuses: list[str] = []
        for event in task_events:
            try:
                prompt = int(event.get("prompt_tokens", -1))
                completion = int(event.get("completion_tokens", -1))
                total = int(event.get("total_tokens", -1))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"episode {episode.task_id!r} has non-numeric provider usage"
                ) from exc
            if prompt < 0 or completion <= 0 or total != prompt + completion:
                raise RuntimeError(
                    f"episode {episode.task_id!r} has invalid provider usage"
                )
            status = str(event.get("reasoning_tokens_status", "unavailable"))
            if status not in {"reported", "unavailable"}:
                raise RuntimeError(
                    f"episode {episode.task_id!r} has invalid reasoning-token status"
                )
            raw_reasoning = event.get("reasoning_tokens")
            if status == "reported":
                if raw_reasoning is None:
                    raise RuntimeError(
                        f"episode {episode.task_id!r} omits reported reasoning tokens"
                    )
                try:
                    reasoning = int(raw_reasoning)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"episode {episode.task_id!r} has non-numeric reasoning tokens"
                    ) from exc
                if reasoning < 0 or reasoning > completion:
                    raise RuntimeError(
                        f"episode {episode.task_id!r} has invalid reasoning tokens"
                    )
                reasoning_tokens += reasoning
            elif raw_reasoning not in {None, 0}:
                raise RuntimeError(
                    f"episode {episode.task_id!r} has reasoning tokens marked unavailable"
                )
            prompt_tokens += prompt
            completion_tokens += completion
            reasoning_statuses.append(status)

        calls = len(task_events)
        if (
            episode.target_llm_calls != calls
            or episode.target_prompt_tokens != prompt_tokens
            or episode.target_completion_tokens != completion_tokens
        ):
            raise RuntimeError(
                f"episode {episode.task_id!r} provider usage does not reconcile "
                "with SkillOpt's token tracker delta"
            )
        if episode.environment_actions != calls:
            raise RuntimeError(
                f"episode {episode.task_id!r} action count does not match target calls"
            )
        status = (
            "reported"
            if all(item == "reported" for item in reasoning_statuses)
            else "unavailable"
            if all(item == "unavailable" for item in reasoning_statuses)
            else "partial"
        )
        if validate_persisted_reasoning and (
            episode.target_reasoning_tokens != reasoning_tokens
            or episode.method_metrics.get("target_reasoning_tokens_status") != status
        ):
            raise RuntimeError(
                f"episode {episode.task_id!r} persisted reasoning usage does not "
                "match provider evidence"
            )
        episode.target_reasoning_tokens = reasoning_tokens
        episode.method_metrics["target_reasoning_tokens_status"] = status

    if by_task:
        raise RuntimeError(
            "provider-call evidence contains unexpected episode ids: "
            f"{sorted(by_task)[:5]}"
        )


def _commit_rollout(
    *,
    out_path: Path,
    request: dict[str, Any],
    rows: list[dict[str, Any]],
    episodes: list[CommonEpisodeRecord],
    action_events: list[dict[str, Any]],
    provider_events: list[dict[str, Any]],
) -> None:
    paths = {
        _RESULTS_NAME: out_path / _RESULTS_NAME,
        _EPISODES_NAME: out_path / _EPISODES_NAME,
        _ACTIONS_NAME: out_path / _ACTIONS_NAME,
    }
    for path in (*paths.values(), out_path / _RECEIPT_NAME, out_path / _FAILURE_NAME):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing rollout evidence: {path}"
            )
    _write_jsonl_atomic(paths[_RESULTS_NAME], rows)
    _write_jsonl_atomic(
        paths[_EPISODES_NAME], [episode.to_dict() for episode in episodes],
    )
    _write_jsonl_atomic(paths[_ACTIONS_NAME], action_events)
    conversations = _conversation_evidence(out_path, request["tasks"])
    receipt = {
        "schema_version": _ROLLOUT_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "request": request,
        "counts": {
            "results": len(rows),
            "episodes": len(episodes),
            "actions": len(action_events),
            "provider_calls": len(provider_events),
        },
        "files": {
            name: _sha256_file(path) for name, path in paths.items()
        },
        "conversations": conversations,
        "provider_event_ids": [
            str(event.get("call_id", "")) for event in provider_events
        ],
        "provider_events_sha256": sha256_json(provider_events),
    }
    # This is the commit marker.  Readers ignore no preceding file unless this
    # receipt exists and every declared byte/hash/identity validates.
    _write_json_atomic(out_path / _RECEIPT_NAME, receipt)


def _persist_rollout_failure(
    *,
    out_path: Path,
    request: dict[str, Any],
    completed_rows: list[dict[str, Any]],
    completed_episodes: list[CommonEpisodeRecord],
    completed_actions: list[dict[str, Any]],
    failed_episode: CommonEpisodeRecord,
    provider_events: list[dict[str, Any]],
) -> None:
    failure_kind = str(
        failed_episode.method_metrics.get("failure_kind", "protocol_failure")
    )
    if failure_kind not in {"infrastructure_failure", "protocol_failure"}:
        failure_kind = "protocol_failure"
    partial_paths: dict[str, Path] = {}
    if completed_rows:
        partial_paths["partial_results.jsonl"] = out_path / "partial_results.jsonl"
        _write_jsonl_atomic(partial_paths["partial_results.jsonl"], completed_rows)
    if completed_episodes:
        name = "partial_common_episodes.jsonl"
        partial_paths[name] = out_path / name
        _write_jsonl_atomic(
            partial_paths[name],
            [episode.to_dict() for episode in completed_episodes],
        )
    if completed_actions:
        name = "partial_common_environment_actions.jsonl"
        partial_paths[name] = out_path / name
        _write_jsonl_atomic(partial_paths[name], completed_actions)
    payload = {
        "schema_version": _ROLLOUT_RECEIPT_SCHEMA_VERSION,
        "status": failure_kind,
        "failure_kind": failure_kind,
        "request": request,
        "failed_episode": failed_episode.to_dict(),
        "completed_counts": {
            "results": len(completed_rows),
            "episodes": len(completed_episodes),
            "actions": len(completed_actions),
        },
        "partial_files": {
            name: _sha256_file(path) for name, path in partial_paths.items()
        },
        "provider_event_ids": [
            str(event.get("call_id", "")) for event in provider_events
        ],
        "provider_events_sha256": sha256_json(provider_events),
    }
    _write_json_atomic(out_path / _FAILURE_NAME, payload)


def _load_committed_rollout(
    out_path: Path,
    expected_request: dict[str, Any],
    *,
    provider_events: list[dict[str, Any]],
    require_provider: bool,
) -> list[dict[str, Any]]:
    failure_path = out_path / _FAILURE_NAME
    receipt_path = out_path / _RECEIPT_NAME
    if failure_path.exists():
        raise RuntimeError(
            f"refusing to reuse an infrastructure-failed rollout: {failure_path}"
        )
    if not receipt_path.is_file():
        raise RuntimeError(
            f"rollout directory is partial or stale and has no completion receipt: "
            f"{out_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"rollout completion receipt is unreadable: {receipt_path}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != _ROLLOUT_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "completed"
    ):
        raise RuntimeError(f"rollout completion receipt is invalid: {receipt_path}")
    if receipt.get("request") != expected_request:
        raise RuntimeError(
            f"rollout completion receipt identity does not match this request: {out_path}"
        )
    required_names = (_RESULTS_NAME, _EPISODES_NAME, _ACTIONS_NAME)
    file_hashes = receipt.get("files")
    if not isinstance(file_hashes, dict) or set(file_hashes) != set(required_names):
        raise RuntimeError("rollout completion receipt has an invalid file inventory")
    for name in required_names:
        path = out_path / name
        if not path.is_file() or _sha256_file(path) != file_hashes[name]:
            raise RuntimeError(f"rollout evidence hash mismatch: {path}")

    rows = _load_results(out_path / _RESULTS_NAME)
    episodes = load_episodes(out_path / _EPISODES_NAME)
    actions = _load_jsonl(out_path / _ACTIONS_NAME)
    expected_ids = [str(item["task_id"]) for item in expected_request["tasks"]]
    if [str(row.get("id", "")) for row in rows] != expected_ids:
        raise RuntimeError("cached SkillOpt result order does not match its receipt")
    if [episode.task_id for episode in episodes] != expected_ids:
        raise RuntimeError("cached SkillOpt episode order does not match its receipt")
    counts = receipt.get("counts")
    if not isinstance(counts, dict) or (
        int(counts.get("results", -1)) != len(rows)
        or int(counts.get("episodes", -1)) != len(episodes)
        or int(counts.get("actions", -1)) != len(actions)
        or int(counts.get("provider_calls", -1)) != len(provider_events)
    ):
        raise RuntimeError("cached SkillOpt evidence counts do not match its receipt")
    provider_error = _provider_evidence_error(
        provider_events,
        tasks=expected_request["tasks"],
        required=require_provider,
    )
    if provider_error is not None:
        raise RuntimeError(
            f"cached provider evidence is incomplete or failed: {provider_error}"
        )
    _reconcile_episode_provider_usage(
        episodes,
        provider_events,
        required=require_provider,
        validate_persisted_reasoning=True,
    )
    if (
        [str(event.get("call_id", "")) for event in provider_events]
        != receipt.get("provider_event_ids")
        or sha256_json(provider_events) != receipt.get("provider_events_sha256")
    ):
        raise RuntimeError("cached provider evidence does not match rollout receipt")
    _validate_action_coverage(rows, episodes, actions)
    if _conversation_evidence(out_path, expected_request["tasks"]) != receipt.get(
        "conversations"
    ):
        raise RuntimeError("cached SkillOpt conversation hashes do not match receipt")
    return rows


def _validate_action_coverage(
    rows: list[dict[str, Any]],
    episodes: list[CommonEpisodeRecord],
    actions: list[dict[str, Any]],
) -> None:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        by_task.setdefault(str(action.get("episode_task_id", "")), []).append(action)
    for row, episode in zip(rows, episodes):
        task_id = str(row.get("id", ""))
        task_actions = by_task.pop(task_id, [])
        indexes = [int(action.get("step_index", -1)) for action in task_actions]
        expected_count = int(row.get("n_turns", -1))
        if (
            indexes != list(range(expected_count))
            or episode.environment_actions != expected_count
        ):
            raise RuntimeError(
                f"cached action evidence is incomplete for task {task_id!r}"
            )
    if by_task:
        raise RuntimeError(
            f"cached action evidence contains unexpected task ids: {sorted(by_task)[:5]}"
        )


def _conversation_evidence(
    out_path: Path,
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        path = out_path / "predictions" / task_id / "conversation.json"
        if not path.is_file():
            raise RuntimeError(f"conversation evidence is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"conversation evidence is unreadable: {path}") from exc
        if not isinstance(payload, list):
            raise RuntimeError(f"conversation evidence is not a list: {path}")
        evidence.append({
            "task_id": task_id,
            "path": path.relative_to(out_path).as_posix(),
            "steps": len(payload),
            "sha256": _sha256_file(path),
        })
    return evidence


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSONL is corrupt at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"JSONL row is not a mapping at {path}:{line_number}")
        rows.append(dict(payload))
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _write_bytes_atomic(path, content.encode("utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n"
    _write_bytes_atomic(path, content.encode("utf-8"))


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "xb") as handle:
            handle.write(content)
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_record(
    *,
    task: dict[str, Any],
    outcome: Any,
    phase: str,
    run_seed: int,
    skill_digest: str,
    artifact_digest: str,
) -> CommonEpisodeRecord:
    row = dict(outcome.skillopt_row)
    fail_reason = str(row.get("fail_reason", ""))
    usage = outcome.target_usage
    return CommonEpisodeRecord(
        method="b3_skillopt",
        phase=str(phase),
        run_seed=int(run_seed),
        task_id=str(task.get("task_id", task.get("id", ""))),
        task_type=str(task.get("task_type", "")),
        manifest_index=int(task.get("manifest_index", 0)),
        gamefile=str(task.get("gamefile", "")),
        gamefile_hash=str(task.get("gamefile_sha256", "")),
        official_success=bool(row.get("hard")),
        task_contract_success=None,
        strict_success=None,
        environment_actions=len(outcome.conversation),
        invalid_actions=None,
        command_turns=int(row.get("n_turns", len(outcome.conversation))),
        timeout=bool(fail_reason and fail_reason.startswith("Timeout")),
        target_llm_calls=usage.calls,
        target_prompt_tokens=usage.prompt_tokens,
        target_completion_tokens=usage.completion_tokens,
        target_reasoning_tokens=usage.reasoning_tokens,
        evolution_llm_calls=0,
        evolution_prompt_tokens=0,
        evolution_completion_tokens=0,
        embedding_calls=0,
        wall_time_ms=outcome.wall_time_ms,
        artifact_digest_before=artifact_digest,
        artifact_digest_after=artifact_digest,
        method_metrics={
            "n_turns": int(row.get("n_turns", 0)),
            "fail_reason": fail_reason,
            "skill_sha256": skill_digest,
            "artifact_digest": artifact_digest,
            "observed_gamefile": str(outcome.actual_gamefile),
            "failure_kind": str(getattr(outcome, "failure_kind", "")),
        },
        infrastructure_failure=bool(outcome.infrastructure_failure),
        infrastructure_error=str(outcome.infrastructure_error),
    )
