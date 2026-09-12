"""Controller-side B5 GEPA method driver."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.baselines.bootstrap_external import load_lock, verify_key_files
from experiments.baselines.common.driver import RunContext, SmokeResult, TrainResult
from experiments.baselines.common.formal_validation import verify_final_evaluation_bijection
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.integrity import validate_episode_usage
from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.subprocess_worker import WorkerWire, run_worker
from experiments.baselines.common.task_authority import StrictTaskEvaluator
from experiments.baselines.common.trace import load_episodes
from experiments.baselines.common.usage import UsageSnapshot
from experiments.baselines.b3_skillopt.driver import _load_action_texts

from .freeze import freeze_gepa_artifacts


METHOD_ID = "b5_gepa"
_WORKER_MODULE = "experiments.baselines.b5_gepa.worker"
_INITIAL_SKILL_REL = "skillopt/envs/alfworld/skills/initial.md"
_ARTIFACT_NAMES = (
    "best_skill.md",
    "gepa_result.json",
    "candidate_lineage.json",
    "pareto_metadata.json",
    "gepa_audit_summary.json",
)


class WorkerExecutionFailure(RuntimeError):
    def __init__(self, phase: str, result: dict[str, Any]) -> None:
        failure_kind = str(result.get("failure_kind", "protocol_failure"))
        if failure_kind not in {"infrastructure_failure", "protocol_failure"}:
            failure_kind = "protocol_failure"
        self.failure_kind = failure_kind
        self.evidence = dict(result)
        detail = str(result.get("error") or result.get("worker_exit_code") or "unknown")
        super().__init__(f"GEPA {phase} worker failed: {detail}")


class GEPABaselineDriver:
    method_id = METHOD_ID

    def __init__(
        self,
        *,
        config: dict[str, Any],
        repo_root: Path,
        external_root: Path,
        skillopt_root: Path,
        lock: dict[str, Any],
        worker_python: Path,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.external_root = external_root
        self.skillopt_root = skillopt_root
        self.lock = lock
        self.worker_python = worker_python

    def preflight(self, ctx: RunContext) -> None:
        if not self.worker_python.is_file():
            raise FileNotFoundError(
                f"GEPA worker interpreter is missing: {self.worker_python}; run "
                "bootstrap_external --method gepa --setup-worker-venv"
            )
        verify_key_files(self.external_root, "gepa", self.lock)
        verify_key_files(self.skillopt_root, "skillopt", self.lock)
        initial_skill = self.skillopt_root / _INITIAL_SKILL_REL
        if not initial_skill.is_file():
            raise FileNotFoundError(f"pinned SkillOpt initial.md is missing: {initial_skill}")
        ctx.model_config.validate_formal_identity()
        ctx.model_config.require_api_key()
        raw_data = os.environ.get("ALFWORLD_DATA", "").strip()
        if not raw_data:
            raise RuntimeError("ALFWORLD_DATA is not set")
        if Path(raw_data).expanduser().resolve(strict=True) != ctx.alfworld_data.resolve(strict=True):
            raise ValueError("RunContext ALFWorld authority differs from worker environment")
        if not ctx.resolved_config_path.is_absolute() or not ctx.resolved_config_path.is_file():
            raise FileNotFoundError("GEPA resolved config path is not an absolute file")
        _validate_context_identity(ctx)

    def smoke(self, ctx: RunContext, train_manifest: TaskManifestSet) -> SmokeResult:
        if (
            ctx.train_manifest_path is None
            or ctx.validation_manifest_path is None
            or ctx.test_manifest_path is None
        ):
            raise ValueError("GEPA smoke requires explicit Train6/Val6/Test6 manifests")
        validation_manifest = TaskManifestSet.load(ctx.validation_manifest_path)
        test_manifest = TaskManifestSet.load(ctx.test_manifest_path)
        _validate_context_identity(
            ctx,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            test_manifest=test_manifest,
        )
        result = self._run_phase(ctx, phase="smoke")
        optimizer_episodes = _collect_episodes(ctx.output_dir / "smoke", {"smoke"})
        counts = dict(result.get("episodes") or {})
        if int(counts.get("total", -1)) != len(optimizer_episodes):
            raise RuntimeError("GEPA smoke worker count disagrees with evidence")
        if not optimizer_episodes:
            raise RuntimeError("GEPA smoke produced no episodes")
        if any(item.infrastructure_failure for item in optimizer_episodes):
            raise RuntimeError("GEPA smoke contains infrastructure failure")
        validate_episode_usage(optimizer_episodes)
        optimizer_usage = UsageSnapshot.load(ctx.output_dir / "smoke" / "usage.json")
        if optimizer_usage.target.calls <= 0 or optimizer_usage.evolution.calls <= 0:
            raise RuntimeError("GEPA smoke did not exercise target and reflection providers")
        persistent = {
            name: ctx.output_dir / "smoke" / name for name in _ARTIFACT_NAMES
        }
        smoke_frozen = freeze_gepa_artifacts(
            source_files=persistent,
            frozen_dir=ctx.output_dir / "smoke_frozen",
            train_manifest_hash=train_manifest.digest,
            validation_manifest_hash=validation_manifest.digest,
            metadata={
                "campaign_id": ctx.campaign_id,
                "run_seed": ctx.run_seed,
                "initial_skill_sha256": ctx.identity["initial_skill_digest"],
                "smoke_only": True,
            },
        )
        heldout_episodes = self._evaluate_frozen(
            ctx=ctx,
            frozen=smoke_frozen,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            evaluation_manifest=test_manifest,
            phase="smoke_test",
            frozen_dir=ctx.output_dir / "smoke_frozen",
        )
        heldout_usage = UsageSnapshot.load(
            ctx.output_dir / "smoke_test" / "usage.json"
        )
        if heldout_usage.target.calls <= 0 or heldout_usage.evolution.calls != 0:
            raise RuntimeError("GEPA frozen smoke Test6 has invalid provider usage")
        usage = UsageSnapshot()
        usage.add(optimizer_usage)
        usage.add(heldout_usage)
        return SmokeResult(
            episodes=[*optimizer_episodes, *heldout_episodes], usage=usage
        )

    def train(
        self,
        ctx: RunContext,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet | None,
    ) -> TrainResult:
        if validation_manifest is None:
            raise ValueError("GEPA requires Validation24")
        _validate_context_identity(
            ctx,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
        )
        result = self._run_phase(ctx, phase="train")
        episodes = _collect_episodes(ctx.output_dir / "train", {"train", "validation"})
        train_episodes = [item for item in episodes if item.phase == "train"]
        validation_episodes = [item for item in episodes if item.phase == "validation"]
        counts = dict(result.get("episodes") or {})
        expected = {
            "total": len(episodes),
            "train": len(train_episodes),
            "validation": len(validation_episodes),
        }
        if {key: counts.get(key) for key in expected} != expected:
            raise RuntimeError("GEPA worker episode counts disagree with sidecars")
        if not train_episodes or not validation_episodes:
            raise RuntimeError("GEPA training lacks Train or full-Validation evidence")
        validate_episode_usage(episodes)
        usage = UsageSnapshot.load(ctx.output_dir / "train" / "usage.json")
        persistent = {name: ctx.output_dir / "train" / name for name in _ARTIFACT_NAMES}
        for name, path in persistent.items():
            if not path.is_file():
                raise FileNotFoundError(f"GEPA worker omitted {name}: {path}")
        metrics = dict(result.get("train_summary") or {})
        metrics["provider_evidence"] = dict(result.get("provider_evidence") or {})
        return TrainResult(
            episodes=train_episodes,
            validation_episodes=validation_episodes,
            usage=usage,
            persistent_artifact_files=persistent,
            method_metrics=metrics,
        )

    def freeze(self, ctx: RunContext, train_result: TrainResult) -> FrozenArtifact:
        if ctx.train_manifest_path is None or ctx.validation_manifest_path is None:
            raise ValueError("GEPA freeze requires Train and Validation manifests")
        train = TaskManifestSet.load(ctx.train_manifest_path)
        validation = TaskManifestSet.load(ctx.validation_manifest_path)
        return freeze_gepa_artifacts(
            source_files=train_result.persistent_artifact_files,
            frozen_dir=ctx.output_dir / "frozen",
            train_manifest_hash=train.digest,
            validation_manifest_hash=validation.digest,
            metadata={
                "campaign_id": ctx.campaign_id,
                "run_seed": ctx.run_seed,
                "initial_skill_sha256": ctx.identity["initial_skill_digest"],
                "method_metrics": train_result.method_metrics,
                "usage": train_result.usage.to_dict(),
            },
        )

    def evaluate_train(
        self,
        ctx: RunContext,
        frozen: FrozenArtifact,
        train_manifest: TaskManifestSet,
        *,
        frozen_dir: Path | None = None,
    ) -> list[CommonEpisodeRecord]:
        return self._evaluate_frozen(
            ctx=ctx,
            frozen=frozen,
            train_manifest=train_manifest,
            validation_manifest=TaskManifestSet.load(ctx.validation_manifest_path),
            evaluation_manifest=train_manifest,
            phase="train_eval",
            frozen_dir=frozen_dir or ctx.output_dir / "frozen",
        )

    def evaluate_test(
        self,
        ctx: RunContext,
        frozen: FrozenArtifact,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet,
        test_manifest: TaskManifestSet,
        *,
        frozen_dir: Path,
    ) -> list[CommonEpisodeRecord]:
        return self._evaluate_frozen(
            ctx=ctx,
            frozen=frozen,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            evaluation_manifest=test_manifest,
            phase="test",
            frozen_dir=frozen_dir,
        )

    def _run_phase(self, ctx: RunContext, *, phase: str) -> dict[str, Any]:
        phase_dir = ctx.output_dir / phase
        wire = WorkerWire(
            method=METHOD_ID,
            phase=phase,
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=None,
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(phase_dir / "worker_result.json"),
            identity=dict(ctx.identity),
            external_method_root=str(self.external_root),
            external_skillopt_root=str(self.skillopt_root),
            initial_skill_path=str(self.skillopt_root / _INITIAL_SKILL_REL),
            campaign=dict(ctx.campaign) if ctx.campaign is not None else None,
            resume=dict(ctx.resume) if ctx.resume is not None else None,
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=phase_dir,
        )
        if not result.get("passed"):
            raise WorkerExecutionFailure(phase, result)
        return result

    def _evaluate_frozen(
        self,
        *,
        ctx: RunContext,
        frozen: FrozenArtifact,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet,
        evaluation_manifest: TaskManifestSet,
        phase: str,
        frozen_dir: Path,
    ) -> list[CommonEpisodeRecord]:
        assert_frozen_unchanged(frozen)
        if frozen.method_id != METHOD_ID:
            raise ValueError("frozen artifact belongs to another method")
        if frozen.root.resolve() != (Path(frozen_dir) / "artifact").resolve():
            raise ValueError("frozen_dir does not identify the supplied artifact")
        if frozen.source_train_manifest_hash != train_manifest.digest:
            raise ValueError("frozen artifact does not match Train")
        if frozen.source_validation_manifest_hash != validation_manifest.digest:
            raise ValueError("frozen artifact does not match Validation")
        if phase not in {"train_eval", "smoke_test", "test"}:
            raise ValueError(f"unsupported frozen GEPA phase: {phase}")
        _validate_context_identity(
            ctx,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            test_manifest=(
                evaluation_manifest if phase in {"smoke_test", "test"} else None
            ),
        )
        phase_dir = ctx.output_dir / phase
        base_identity = {
            key: value
            for key, value in ctx.identity.items()
            if key != "initial_skill_digest"
            and not (phase == "train_eval" and key == "test_manifest_digest")
        }
        identity = {
            **base_identity,
            "evaluation_manifest_digest": evaluation_manifest.digest,
            "frozen_artifact_digest": frozen.digest,
        }
        wire = WorkerWire(
            method=METHOD_ID,
            phase=phase,
            manifest_path=(
                str(ctx.train_manifest_path) if phase == "train_eval" else None
            ),
            validation_manifest_path=None,
            test_manifest_path=(
                str(ctx.test_manifest_path)
                if phase in {"smoke_test", "test"}
                else None
            ),
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(phase_dir / "worker_result.json"),
            identity=identity,
            frozen_artifact_path=str(frozen_dir),
            external_method_root=str(self.external_root),
            external_skillopt_root=str(self.skillopt_root),
            initial_skill_path=None,
            campaign=dict(ctx.campaign) if ctx.campaign is not None else None,
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=phase_dir,
        )
        if not result.get("passed"):
            raise WorkerExecutionFailure(phase, result)
        if result.get("optimizer_constructed") is not False:
            raise RuntimeError("frozen GEPA evaluation constructed an optimizer")
        if not result.get("frozen_unchanged"):
            raise RuntimeError("frozen GEPA artifact changed during evaluation")
        episodes = _collect_episodes(phase_dir, {phase})
        if result.get("episodes") != len(episodes):
            raise RuntimeError("GEPA frozen worker count disagrees with sidecars")
        verify_final_evaluation_bijection(
            episodes,
            evaluation_manifest,
            role="test" if phase in {"smoke_test", "test"} else "train",
            expected_phase=phase,
            expected_method=METHOD_ID,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=False,
            profile="smoke_v1" if phase == "smoke_test" else "formal_v2",
        )
        actions = _load_action_texts(phase_dir, episodes=episodes)
        entries = {task.task_id: task for task in evaluation_manifest.tasks}
        evaluator = StrictTaskEvaluator(ctx.alfworld_data)
        for episode in episodes:
            if episode.infrastructure_failure:
                continue
            outcome = evaluator.evaluate(
                entries[episode.task_id],
                actions[episode.task_id],
                official_success=episode.official_success,
            )
            episode.set_posthoc_outcome(
                contract_consistency=outcome.task_contract_success
            )
            if episode.common_strict_success is not outcome.strict_success:
                raise RuntimeError(
                    "GEPA strict evaluator disagrees with the common success contract"
                )
            episode.invalid_actions = outcome.invalid_actions
        verify_final_evaluation_bijection(
            episodes,
            evaluation_manifest,
            role="test" if phase in {"smoke_test", "test"} else "train",
            expected_phase=phase,
            expected_method=METHOD_ID,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=True,
            profile="smoke_v1" if phase == "smoke_test" else "formal_v2",
        )
        assert_frozen_unchanged(frozen)
        return episodes


def _collect_episodes(root: Path, phases: set[str]) -> list[CommonEpisodeRecord]:
    paths = sorted(root.rglob("common_episodes.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no Common episodes under {root}")
    episodes: list[CommonEpisodeRecord] = []
    for path in paths:
        episodes.extend(load_episodes(path))
    unexpected = sorted({item.phase for item in episodes} - phases)
    if unexpected:
        raise ValueError(f"GEPA evidence contains unexpected phases: {unexpected}")
    return episodes


def _validate_context_identity(
    ctx: RunContext,
    *,
    train_manifest: TaskManifestSet | None = None,
    validation_manifest: TaskManifestSet | None = None,
    test_manifest: TaskManifestSet | None = None,
) -> None:
    try:
        config = json.loads(ctx.resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("GEPA resolved config is unreadable") from exc
    digest = sha256_json(config)
    if digest != ctx.config_hash or digest != ctx.identity.get("config_digest"):
        raise ValueError("GEPA RunContext config identity mismatch")
    for role, supplied, path, key in (
        ("Train", train_manifest, ctx.train_manifest_path, "train_manifest_digest"),
        (
            "Validation",
            validation_manifest,
            ctx.validation_manifest_path,
            "validation_manifest_digest",
        ),
        ("Test", test_manifest, ctx.test_manifest_path, "test_manifest_digest"),
    ):
        if role == "Test" and supplied is None:
            continue
        if path is None:
            raise ValueError(f"GEPA RunContext has no {role} manifest path")
        persisted = TaskManifestSet.load(path)
        if ctx.identity.get(key) != persisted.digest:
            raise ValueError(f"GEPA RunContext {role} manifest identity mismatch")
        if supplied is not None and supplied.digest != persisted.digest:
            raise ValueError(f"supplied {role} manifest differs from RunContext")


def load_lock_and_driver(
    *,
    repo_root: Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], GEPABaselineDriver]:
    import yaml

    lock = load_lock(repo_root / "experiments" / "baselines" / "baseline_lock.yaml")
    method_path = Path(
        config_path or repo_root / "configs" / "baselines" / "b5_gepa.yaml"
    ).expanduser()
    if not method_path.is_absolute():
        method_path = (repo_root / method_path).resolve()
    common = yaml.safe_load(
        (repo_root / "configs" / "baselines" / "common.yaml").read_text(encoding="utf-8")
    )
    method = yaml.safe_load(method_path.read_text(encoding="utf-8"))
    if not isinstance(common, dict) or not isinstance(method, dict):
        raise ValueError("GEPA/common config root must be a mapping")
    config = {**common, **method}
    worker_python = Path(str(config.get("worker_python", ".venv_b5_gepa/bin/python")))
    if not worker_python.is_absolute():
        worker_python = repo_root / worker_python
    worker_python = Path(os.path.abspath(worker_python))
    return lock, GEPABaselineDriver(
        config=config,
        repo_root=repo_root,
        external_root=repo_root / ".external" / "gepa",
        skillopt_root=repo_root / ".external" / "skillopt",
        lock=lock,
        worker_python=worker_python,
    )
