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

import json
import os
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.prompts import load_prompt
from skillopt.utils import skill_hash

from experiments.baselines.common.manifest import TaskManifestSet
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.trace import CommonSidecarWriter

from .episode_runner import SkillOptTextEpisodeRunner


def manifest_task_to_item(task: Any, *, phase_label: str) -> dict[str, Any]:
    """One manifest task as a SkillOpt batch item (id/gamefile identity only)."""

    return {
        "id": task.task_id,
        "task_id": task.task_id,
        "gamefile": task.gamefile_rel,
        "gamefile_sha256": task.gamefile_sha256,
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
        phase = _SPLIT_PHASES.get(str(split).strip().lower(), "train")
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
        seed: int = 42,
        phase: str = "train",
        episode_runner: Any | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.workers = max(int(workers or 1), 1)
        self.max_api_workers = max_api_workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.seed = seed
        self.phase = phase
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

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run the batch with the upstream text-skill loop and mirror Common sidecars."""

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_path = out_path / "results.jsonl"
        # Resume semantics mirror the upstream adapter: an existing non-empty
        # results file means this exact rollout already completed.
        if results_path.exists():
            existing = _load_results(results_path)
            if existing:
                return existing

        rows: list[dict[str, Any]] = []
        episodes: list[CommonEpisodeRecord] = []
        action_events: list[dict[str, Any]] = []
        skill_digest = skill_hash(skill_content)
        for task in env_manager.tasks:
            outcome = self._episode_runner.run(
                dict(task), skill_content, str(out_path),
            )
            row = dict(outcome.skillopt_row)
            row.setdefault("id", str(task.get("id", "")))
            row.setdefault("hard", 0)
            row.setdefault("soft", 0.0)
            row.setdefault("n_turns", 0)
            row.setdefault("fail_reason", "")
            row.setdefault("agent_ok", True)
            row.setdefault("task_type", str(task.get("task_type", "")))
            row.setdefault("gamefile", str(task.get("gamefile", "")))
            row.setdefault("task_description", "")
            rows.append(row)
            episodes.append(_episode_record(
                task=task,
                outcome=outcome,
                phase=str(task.get("phase", self.phase)),
                run_seed=self.seed,
                skill_digest=skill_digest,
            ))
            for step_index, step in enumerate(outcome.conversation):
                action_events.append({
                    "episode_task_id": str(task.get("id", "")),
                    "step_index": step_index,
                    "action": str(step.get("action", "")),
                    "env_feedback": str(step.get("env_feedback", "")),
                    "reward": float(step.get("reward", 0.0)),
                    "done": bool(step.get("done", False)),
                })
        with open(results_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        writer = CommonSidecarWriter(out_path)
        writer.write_episodes(episodes)
        writer.write_environment_actions(action_events)
        return rows

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
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _episode_record(
    *,
    task: dict[str, Any],
    outcome: Any,
    phase: str,
    run_seed: int,
    skill_digest: str,
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
        command_turns=0,
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
        artifact_digest_before=skill_digest,
        artifact_digest_after=skill_digest,
        method_metrics={
            "n_turns": int(row.get("n_turns", 0)),
            "fail_reason": fail_reason,
            "skill_sha256_16": skill_digest,
        },
        infrastructure_failure=bool(outcome.infrastructure_failure),
        infrastructure_error=str(outcome.infrastructure_error),
    )
