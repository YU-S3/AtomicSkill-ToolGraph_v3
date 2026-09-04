"""B3 SkillOpt driver: controller-side orchestration of the worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.baselines.bootstrap_external import load_lock, verify_key_files
from experiments.baselines.common.driver import RunContext, TrainResult
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.manifest import TaskManifestSet
from experiments.baselines.common.subprocess_worker import WorkerWire, run_worker
from experiments.baselines.common.task_authority import StrictTaskEvaluator
from experiments.baselines.common.trace import load_episodes
from experiments.baselines.common.usage import UsageSnapshot

from .freeze import freeze_best_skill

_WORKER_MODULE = "experiments.baselines.b3_skillopt.worker"
_SKILL_INIT_REL = "skillopt/envs/alfworld/skills/initial.md"


class SkillOptBaselineDriver:
    method_id = "b3_skillopt"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        repo_root: Path,
        external_root: Path,
        lock: dict[str, Any],
        worker_python: Path,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.external_root = external_root
        self.lock = lock
        self.worker_python = worker_python

    # ── BaselineMethodDriver ──────────────────────────────────────────────

    def preflight(self, ctx: RunContext) -> None:
        if not self.worker_python.exists():
            raise FileNotFoundError(
                f"worker venv python is missing: {self.worker_python}; run "
                "`python -m experiments.baselines.bootstrap_external --setup-worker-venv`"
            )
        verify_key_files(self.external_root, "skillopt", self.lock)
        ctx.model_config.validate_formal_identity()
        ctx.model_config.require_api_key()
        if not os.environ.get("ALFWORLD_DATA", "").strip():
            raise RuntimeError("ALFWORLD_DATA is not set in the environment")
        if not ctx.alfworld_data.is_dir():
            raise FileNotFoundError(f"ALFWORLD_DATA directory does not exist: {ctx.alfworld_data}")

    def train(
        self,
        ctx: RunContext,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet | None,
    ) -> TrainResult:
        if validation_manifest is None:
            raise ValueError("SkillOpt requires the common Validation manifest (§5.4)")
        if ctx.train_manifest_path is None or ctx.validation_manifest_path is None:
            raise ValueError("SkillOpt train requires explicit manifest paths")
        wire = WorkerWire(
            method=self.method_id,
            phase="train",
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=str(ctx.test_manifest_path) if ctx.test_manifest_path else None,
            config_path=str(ctx.repo_root / "configs" / "baselines" / "b3_skillopt.yaml"),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            external_skillopt_root=str(self.external_root),
            skill_init_rel=_SKILL_INIT_REL,
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=ctx.output_dir / "train",
        )
        if not result.get("passed"):
            raise RuntimeError(
                "SkillOpt train worker failed: "
                + str(result.get("error") or result.get("worker_exit_code") or "unknown")
            )
        train_out = ctx.output_dir / "train"
        best_skill = train_out / "best_skill.md"
        if not best_skill.is_file():
            raise FileNotFoundError(f"train worker produced no best_skill.md: {best_skill}")
        usage = UsageSnapshot.load(train_out / "usage.json")
        episodes = _collect_episodes(train_out)
        return TrainResult(
            episodes=[episode for episode in episodes if episode.phase == "train"],
            validation_episodes=[
                episode for episode in episodes if episode.phase == "validation"
            ],
            usage=usage,
            persistent_artifact_files={"best_skill.md": best_skill},
            method_metrics=dict(result.get("train_summary") or {}),
        )

    def freeze(self, ctx: RunContext, train_result: TrainResult) -> FrozenArtifact:
        frozen_dir = ctx.output_dir / "frozen"
        if frozen_dir.exists():
            raise FileExistsError(frozen_dir)
        if ctx.train_manifest_path is None or ctx.validation_manifest_path is None:
            raise ValueError("SkillOpt freeze requires explicit manifest paths")
        train_manifest = TaskManifestSet.load(ctx.train_manifest_path)
        validation_manifest = TaskManifestSet.load(ctx.validation_manifest_path)
        return freeze_best_skill(
            best_skill_path=train_result.persistent_artifact_files["best_skill.md"],
            frozen_dir=frozen_dir,
            train_manifest_hash=train_manifest.digest,
            validation_manifest_hash=validation_manifest.digest,
            metadata={
                "campaign_id": ctx.campaign_id,
                "run_seed": ctx.run_seed,
                "method_metrics": train_result.method_metrics,
                "usage": train_result.usage.to_dict(),
            },
        )

    def evaluate(
        self,
        ctx: RunContext,
        frozen: FrozenArtifact,
        test_manifest: TaskManifestSet,
        *,
        frozen_dir: Path | None = None,
    ) -> list[Any]:
        assert_frozen_unchanged(frozen)
        frozen_dir = frozen_dir or (ctx.output_dir / "frozen")
        if (
            ctx.train_manifest_path is None
            or ctx.validation_manifest_path is None
            or ctx.test_manifest_path is None
        ):
            raise ValueError("SkillOpt evaluate requires all three manifest paths")
        wire = WorkerWire(
            method=self.method_id,
            phase="test",
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=str(ctx.test_manifest_path),
            config_path=str(ctx.repo_root / "configs" / "baselines" / "b3_skillopt.yaml"),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            frozen_artifact_path=str(frozen_dir),
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=ctx.output_dir / "test",
        )
        if not result.get("passed"):
            raise RuntimeError(
                "SkillOpt test worker failed: "
                + str(result.get("error") or result.get("worker_exit_code") or "unknown")
            )
        if not result.get("frozen_unchanged"):
            raise RuntimeError("frozen artifact changed during the held-out phase")
        assert_frozen_unchanged(frozen)
        test_out = ctx.output_dir / "test"
        episodes = _collect_episodes(test_out)
        return self._apply_strict_evaluation(episodes, test_manifest, ctx)

    # ── Strict post-evaluation (Ours harness boundary, controller-side) ───

    def _apply_strict_evaluation(
        self,
        episodes: list[Any],
        test_manifest: TaskManifestSet,
        ctx: RunContext,
    ) -> list[Any]:
        entries = {task.task_id: task for task in test_manifest.tasks}
        actions_by_task = _load_action_texts(ctx.output_dir / "test")
        evaluator = StrictTaskEvaluator(ctx.alfworld_data)
        for episode in episodes:
            entry = entries.get(episode.task_id)
            if entry is None:
                raise ValueError(
                    f"test episode {episode.task_id} is not part of the Test manifest"
                )
            if episode.infrastructure_failure:
                continue
            outcome = evaluator.evaluate(
                entry,
                actions_by_task.get(episode.task_id, []),
                official_success=episode.official_success,
            )
            episode.task_contract_success = outcome.task_contract_success
            episode.strict_success = outcome.strict_success
            episode.invalid_actions = outcome.invalid_actions
        return episodes


def _collect_episodes(root: Path) -> list[Any]:
    episodes: list[Any] = []
    for sidecar in sorted(root.rglob("common_episodes.jsonl")):
        episodes.extend(load_episodes(sidecar))
    return episodes


def _load_action_texts(root: Path) -> dict[str, list[str]]:
    """Per-episode action sequence in execution order, from the sidecar."""

    actions: dict[str, dict[int, str]] = {}
    for sidecar in sorted(root.rglob("common_environment_actions.jsonl")):
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("episode_task_id", ""))
            actions.setdefault(task_id, {})[int(row.get("step_index", 0))] = str(
                row.get("action", "")
            )
    return {
        task_id: [steps[index] for index in sorted(steps)]
        for task_id, steps in actions.items()
    }


def load_lock_and_driver(*, repo_root: Path) -> tuple[dict[str, Any], "SkillOptBaselineDriver"]:
    import yaml

    lock = load_lock(repo_root / "experiments" / "baselines" / "baseline_lock.yaml")
    method_config = yaml.safe_load(
        (repo_root / "configs" / "baselines" / "b3_skillopt.yaml").read_text(encoding="utf-8")
    )
    common_config = yaml.safe_load(
        (repo_root / "configs" / "baselines" / "common.yaml").read_text(encoding="utf-8")
    )
    config = {**dict(common_config or {}), **dict(method_config or {})}
    worker_python = Path(str(config.get("worker_python", ".venv_b3_skillopt/bin/python")))
    if not worker_python.is_absolute():
        worker_python = (repo_root / worker_python).resolve()
    return lock, SkillOptBaselineDriver(
        config=config,
        repo_root=repo_root,
        external_root=repo_root / ".external" / "skillopt",
        lock=lock,
        worker_python=worker_python,
    )
