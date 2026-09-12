"""Controller-side B4 SkillGen-S method driver."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

from experiments.baselines.bootstrap_external import load_lock, verify_key_files, verify_runtime_tree
from experiments.baselines.common.driver import RunContext, SmokeResult, TrainResult
from experiments.baselines.common.formal_validation import verify_final_evaluation_bijection
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged, freeze_files
from experiments.baselines.common.integrity import validate_episode_usage
from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.subprocess_worker import WorkerWire, run_worker
from experiments.baselines.common.task_authority import StrictTaskEvaluator
from experiments.baselines.common.trace import load_episodes
from experiments.baselines.common.usage import UsageSnapshot

from .progress_adapter import analyze_label_coverage, require_complete_train_coverage


WORKER_MODULE = "experiments.baselines.b4_skillgen_s.worker"


class WorkerExecutionFailure(RuntimeError):
    def __init__(self, phase: str, result: dict[str, Any]) -> None:
        self.failure_kind = str(result.get("failure_kind", "protocol_failure"))
        self.evidence = dict(result)
        super().__init__(
            f"SkillGen-S {phase} worker failed: "
            f"{result.get('error') or result.get('worker_exit_code') or 'unknown'}"
        )


class SkillGenBaselineDriver:
    method_id = "b4_skillgen_s"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        repo_root: Path,
        external_root: Path,
        lock: dict[str, Any],
        worker_python: Path,
        supervision_path: Path,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.external_root = external_root
        self.lock = lock
        self.worker_python = worker_python
        self.supervision_path = supervision_path

    def preflight(self, ctx: RunContext) -> None:
        if not self.worker_python.is_file():
            raise FileNotFoundError(
                f"SkillGen worker Python is missing: {self.worker_python}; run "
                "bootstrap_external --setup-worker-venv"
            )
        verify_key_files(self.external_root, "skillgen", self.lock)
        runtime = verify_runtime_tree(self.external_root, "skillgen", self.lock)
        if ctx.identity.get("external_source_digest") != runtime["sha256"]:
            raise ValueError("RunContext does not bind the verified SkillGen runtime tree")
        ctx.model_config.validate_formal_identity()
        if ctx.validation_manifest_path is not None:
            raise ValueError("SkillGen-S is forbidden from receiving a Validation manifest")
        if "validation_manifest_digest" in ctx.identity:
            raise ValueError("SkillGen-S identity must not bind Validation data")
        _validate_resolved_config(ctx)
        data_raw = os.environ.get("ALFWORLD_DATA", "").strip()
        if not data_raw:
            raise RuntimeError("ALFWORLD_DATA is missing")
        if Path(data_raw).expanduser().resolve(strict=True) != ctx.alfworld_data.resolve(strict=True):
            raise ValueError("RunContext ALFWorld root differs from worker ALFWORLD_DATA")
        phase = str(ctx.identity.get("execution_phase", "train"))
        if phase in {"train", "smoke"}:
            if ctx.train_manifest_path is None:
                raise ValueError("SkillGen Train/smoke preflight requires a Train manifest")
            manifest = TaskManifestSet.load(ctx.train_manifest_path)
            if ctx.identity.get("supervision_digest") != _sha256_file(self.supervision_path):
                raise ValueError("RunContext does not bind the SkillGen supervision file")
            report, _labels = analyze_label_coverage(manifest, self.supervision_path)
            report_path = ctx.output_dir / "preflight_label_coverage.json"
            report_path.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not report.passed:
                require_complete_train_coverage(manifest, self.supervision_path)
        ctx.model_config.require_api_key()

    def _wire(
        self,
        ctx: RunContext,
        *,
        phase: str,
        frozen_dir: Path | None = None,
        include_supervision: bool = False,
    ) -> WorkerWire:
        phase_dir = ctx.output_dir / phase
        identity = dict(ctx.identity)
        identity["external_source_digest"] = str(
            self.lock["skillgen"]["runtime_tree"]["sha256"]
        )
        if include_supervision:
            identity["supervision_digest"] = _sha256_file(self.supervision_path)
            identity.pop("test_manifest_digest", None)
            identity.pop("evaluation_manifest_digest", None)
            identity.pop("frozen_artifact_digest", None)
        else:
            identity.pop("supervision_digest", None)
            if phase == "train_eval":
                identity.pop("test_manifest_digest", None)
        return WorkerWire(
            method=self.method_id,
            phase=phase,
            manifest_path=(
                str(ctx.train_manifest_path)
                if phase not in {"smoke_test", "test"}
                and ctx.train_manifest_path is not None else None
            ),
            validation_manifest_path=None,
            test_manifest_path=(
                str(ctx.test_manifest_path)
                if phase in {"smoke_test", "test"}
                and ctx.test_manifest_path is not None else None
            ),
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(phase_dir / "worker_result.json"),
            identity=identity,
            frozen_artifact_path=str(frozen_dir) if frozen_dir is not None else None,
            external_method_root=str(self.external_root),
            supervision_path=str(self.supervision_path) if include_supervision else None,
            campaign=dict(ctx.campaign) if ctx.campaign is not None else None,
            resume=dict(ctx.resume) if ctx.resume is not None else None,
        )

    def smoke(self, ctx: RunContext, train_manifest: TaskManifestSet) -> SmokeResult:
        _validate_context_manifest(ctx, train_manifest)
        if ctx.test_manifest_path is None:
            raise ValueError("B4 smoke requires an independent Test-6 manifest")
        smoke_test = TaskManifestSet.load(ctx.test_manifest_path)
        if ctx.identity.get("test_manifest_digest") != smoke_test.digest:
            raise ValueError("B4 smoke Test manifest identity mismatch")
        wire = self._wire(ctx, phase="smoke", include_supervision=True)
        result = run_worker(
            wire=wire, worker_module=WORKER_MODULE, python=self.worker_python,
            wire_dir=ctx.output_dir / "smoke",
        )
        if not result.get("passed"):
            raise WorkerExecutionFailure("smoke", result)
        sampling_episodes = _collect_episodes(ctx.output_dir / "smoke", {"smoke"})
        if result.get("episodes") != len(sampling_episodes):
            raise RuntimeError("B4 smoke worker count differs from episode sidecars")
        expected = len(train_manifest.tasks) * 6
        if len(sampling_episodes) != expected:
            raise RuntimeError(
                f"B4 smoke requires {expected} Train sampling episodes, "
                f"got {len(sampling_episodes)}"
            )
        library = ctx.output_dir / "smoke" / "method_state" / "library"
        files = {
            path.relative_to(library).as_posix(): path
            for path in sorted(library.rglob("*")) if path.is_file()
        }
        declared = sorted(str(value) for value in result.get("persistent_artifact_files", []))
        if not files or declared != sorted(files):
            raise RuntimeError("B4 smoke artifact inventory is incomplete")
        frozen = freeze_files(
            method_id=self.method_id,
            source_files=files,
            destination=ctx.output_dir / "smoke_frozen",
            source_train_manifest_hash=train_manifest.digest,
            source_validation_manifest_hash=None,
            metadata={
                "run_seed": ctx.run_seed,
                "smoke_only": True,
                "extra_train_supervision": True,
                "weak_supervision": "subgoal_progress",
                "label_sha256": result.get("label_sha256"),
                "method_metrics": dict(result.get("train_summary") or {}),
            },
        )
        identity = {
            **ctx.identity,
            "evaluation_manifest_digest": smoke_test.digest,
            "frozen_artifact_digest": frozen.digest,
        }
        smoke_test_ctx = RunContext(**{**ctx.__dict__, "identity": identity})
        test_wire = self._wire(
            smoke_test_ctx,
            phase="smoke_test",
            frozen_dir=ctx.output_dir / "smoke_frozen",
        )
        test_result = run_worker(
            wire=test_wire,
            worker_module=WORKER_MODULE,
            python=self.worker_python,
            wire_dir=ctx.output_dir / "smoke_test",
        )
        if not test_result.get("passed"):
            raise WorkerExecutionFailure("smoke_test", test_result)
        if (
            not test_result.get("frozen_unchanged")
            or test_result.get("frozen_digest_before") != frozen.digest
            or test_result.get("frozen_digest_after") != frozen.digest
        ):
            raise RuntimeError("B4 smoke-test did not preserve its frozen artifact")
        inference_episodes = _collect_episodes(
            ctx.output_dir / "smoke_test", {"smoke_test"}
        )
        if len(inference_episodes) != len(smoke_test.tasks):
            raise RuntimeError("B4 smoke-test is not a complete Test-6 evaluation")
        verify_final_evaluation_bijection(
            inference_episodes,
            smoke_test,
            role="test",
            expected_phase="smoke_test",
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=False,
            profile="smoke_v1",
        )
        inference_episodes = _apply_strict_evaluation(
            inference_episodes,
            smoke_test,
            smoke_test_ctx,
            phase_root=ctx.output_dir / "smoke_test",
        )
        verify_final_evaluation_bijection(
            inference_episodes,
            smoke_test,
            role="test",
            expected_phase="smoke_test",
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=True,
            profile="smoke_v1",
        )
        episodes = [*sampling_episodes, *inference_episodes]
        validate_episode_usage(episodes)
        usage = UsageSnapshot.load(ctx.output_dir / "smoke" / "usage.json")
        usage.add(UsageSnapshot.load(ctx.output_dir / "smoke_test" / "usage.json"))
        assert_frozen_unchanged(frozen)
        return SmokeResult(episodes=episodes, usage=usage)

    def train(
        self,
        ctx: RunContext,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet | None,
    ) -> TrainResult:
        if validation_manifest is not None:
            raise ValueError("SkillGen-S does not use Validation")
        _validate_context_manifest(ctx, train_manifest)
        wire = self._wire(ctx, phase="train", include_supervision=True)
        result = run_worker(
            wire=wire, worker_module=WORKER_MODULE, python=self.worker_python,
            wire_dir=ctx.output_dir / "train",
        )
        if not result.get("passed"):
            raise WorkerExecutionFailure("train", result)
        train_dir = ctx.output_dir / "train"
        episodes = _collect_episodes(train_dir, {"train"})
        if result.get("sampling_episodes") != len(episodes):
            raise RuntimeError("B4 Train sidecar does not contain exactly all sampling jobs")
        expected = len(train_manifest.tasks) * 6
        if len(episodes) != expected:
            raise RuntimeError(f"B4 Train requires {expected} sampling episodes, got {len(episodes)}")
        validate_episode_usage(episodes)
        library = train_dir / "method_state" / "library"
        if not library.is_dir():
            raise FileNotFoundError("B4 Train produced no extracted skill library")
        files = {
            path.relative_to(library).as_posix(): path
            for path in sorted(library.rglob("*")) if path.is_file()
        }
        declared = sorted(str(value) for value in result.get("persistent_artifact_files", []))
        if declared != sorted(files):
            raise RuntimeError("B4 worker artifact inventory differs from the extracted library")
        return TrainResult(
            episodes=episodes,
            validation_episodes=[],
            usage=UsageSnapshot.load(train_dir / "usage.json"),
            persistent_artifact_files=files,
            method_metrics=dict(result.get("train_summary") or {}),
        )

    def freeze(self, ctx: RunContext, train_result: TrainResult) -> FrozenArtifact:
        if not train_result.persistent_artifact_files:
            raise ValueError("B4 has no extracted assets to freeze")
        if ctx.train_manifest_path is None:
            raise ValueError("B4 freeze requires its source Train manifest")
        manifest = TaskManifestSet.load(ctx.train_manifest_path)
        return freeze_files(
            method_id=self.method_id,
            source_files=train_result.persistent_artifact_files,
            destination=ctx.output_dir / "frozen",
            source_train_manifest_hash=manifest.digest,
            source_validation_manifest_hash=None,
            metadata={
                "run_seed": ctx.run_seed,
                "extra_train_supervision": True,
                "weak_supervision": "subgoal_progress",
                "label_sha256": train_result.method_metrics.get("label_coverage", {}).get("label_sha256"),
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
        return self._evaluate(
            ctx, frozen, train_manifest, phase="train_eval",
            frozen_dir=frozen_dir or ctx.output_dir / "frozen",
        )

    def evaluate_test(
        self,
        ctx: RunContext,
        frozen: FrozenArtifact,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet | None,
        test_manifest: TaskManifestSet,
        *,
        frozen_dir: Path,
    ) -> list[CommonEpisodeRecord]:
        if validation_manifest is not None:
            raise ValueError("SkillGen-S Test must not receive Validation data")
        if frozen.source_train_manifest_hash != train_manifest.digest:
            raise ValueError("B4 frozen artifact does not match its source Train manifest")
        if ctx.test_manifest_path is None or TaskManifestSet.load(ctx.test_manifest_path).digest != test_manifest.digest:
            raise ValueError("B4 Test manifest does not match RunContext")
        return self._evaluate(
            ctx, frozen, test_manifest, phase="test", frozen_dir=frozen_dir,
        )

    def _evaluate(
        self,
        ctx: RunContext,
        frozen: FrozenArtifact,
        manifest: TaskManifestSet,
        *,
        phase: str,
        frozen_dir: Path,
    ) -> list[CommonEpisodeRecord]:
        assert_frozen_unchanged(frozen)
        if frozen.method_id != self.method_id or frozen.source_validation_manifest_hash is not None:
            raise ValueError("invalid B4 frozen artifact identity")
        identity = dict(ctx.identity)
        identity.update({
            "evaluation_manifest_digest": manifest.digest,
            "frozen_artifact_digest": frozen.digest,
        })
        eval_ctx = RunContext(**{
            **ctx.__dict__,
            "identity": identity,
        })
        wire = self._wire(eval_ctx, phase=phase, frozen_dir=frozen_dir)
        result = run_worker(
            wire=wire, worker_module=WORKER_MODULE, python=self.worker_python,
            wire_dir=ctx.output_dir / phase,
        )
        if not result.get("passed"):
            raise WorkerExecutionFailure(phase, result)
        if (
            not result.get("frozen_unchanged")
            or result.get("frozen_digest_before") != frozen.digest
            or result.get("frozen_digest_after") != frozen.digest
        ):
            raise RuntimeError(f"B4 {phase} worker did not prove immutable retrieval assets")
        episodes = _collect_episodes(ctx.output_dir / phase, {phase})
        if len(episodes) != len(manifest.tasks):
            raise RuntimeError(f"B4 {phase} did not produce one episode per manifest task")
        validate_episode_usage(episodes)
        verify_final_evaluation_bijection(
            episodes,
            manifest,
            role="test" if phase == "test" else "train",
            expected_phase=phase,
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=False,
            profile=str(self.config.get("protocol_profile", "pilot_v1")),
        )
        episodes = _apply_strict_evaluation(
            episodes, manifest, ctx, phase_root=ctx.output_dir / phase,
        )
        verify_final_evaluation_bijection(
            episodes,
            manifest,
            role="test" if phase == "test" else "train",
            expected_phase=phase,
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=True,
            profile=str(self.config.get("protocol_profile", "pilot_v1")),
        )
        assert_frozen_unchanged(frozen)
        return episodes


def _collect_episodes(root: Path, phases: set[str]) -> list[CommonEpisodeRecord]:
    paths = sorted(root.rglob("common_episodes.jsonl"))
    if not paths:
        raise FileNotFoundError(f"B4 phase has no common episode sidecar: {root}")
    episodes = [episode for path in paths for episode in load_episodes(path)]
    unexpected = sorted({episode.phase for episode in episodes} - phases)
    if unexpected:
        raise ValueError(f"B4 phase sidecar contains unexpected phases: {unexpected}")
    return episodes


def _load_actions(root: Path, episodes: list[CommonEpisodeRecord]) -> dict[str, list[str]]:
    expected = {episode.task_id: episode.environment_actions for episode in episodes}
    if len(expected) != len(episodes):
        raise ValueError("B4 frozen evaluation contains duplicate task ids")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("common_environment_actions.jsonl")):
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    by_task: dict[str, dict[int, str]] = {}
    for row in rows:
        task_id = str(row.get("episode_task_id", ""))
        if task_id not in expected:
            raise ValueError(f"B4 evaluation action references unexpected task {task_id!r}")
        index = int(row.get("step_index", -1))
        if index in by_task.setdefault(task_id, {}):
            raise ValueError(f"B4 evaluation duplicates action {task_id}:{index}")
        by_task[task_id][index] = str(row.get("action", ""))
    result: dict[str, list[str]] = {}
    for task_id, count in expected.items():
        task_rows = by_task.get(task_id, {})
        if sorted(task_rows) != list(range(count)):
            raise ValueError(f"B4 evaluation action evidence is incomplete for {task_id}")
        result[task_id] = [task_rows[index] for index in range(count)]
    return result


def _apply_strict_evaluation(
    episodes: list[CommonEpisodeRecord],
    manifest: TaskManifestSet,
    ctx: RunContext,
    *,
    phase_root: Path,
) -> list[CommonEpisodeRecord]:
    actions = _load_actions(phase_root, episodes)
    entries = {task.task_id: task for task in manifest.tasks}
    evaluator = StrictTaskEvaluator(ctx.alfworld_data)
    for episode in episodes:
        outcome = evaluator.evaluate(
            entries[episode.task_id], actions[episode.task_id],
            official_success=episode.official_success,
        )
        episode.set_posthoc_outcome(
            contract_consistency=outcome.task_contract_success
        )
        if episode.common_strict_success is not outcome.strict_success:
            raise RuntimeError(
                "B4 strict evaluator disagrees with the common success contract"
            )
        episode.invalid_actions = outcome.invalid_actions
    return episodes


def _validate_resolved_config(ctx: RunContext) -> None:
    payload = json.loads(ctx.resolved_config_path.read_text(encoding="utf-8"))
    actual = sha256_json(payload)
    if actual != ctx.config_hash or actual != ctx.identity.get("config_digest"):
        raise ValueError("B4 RunContext config identity mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_context_manifest(ctx: RunContext, manifest: TaskManifestSet) -> None:
    _validate_resolved_config(ctx)
    if ctx.train_manifest_path is None:
        raise ValueError("B4 RunContext lacks a Train manifest")
    if TaskManifestSet.load(ctx.train_manifest_path).digest != manifest.digest:
        raise ValueError("B4 RunContext Train manifest path differs from supplied manifest")
    if ctx.identity.get("train_manifest_digest") != manifest.digest:
        raise ValueError("B4 RunContext Train identity differs from supplied manifest")


def load_lock_and_driver(
    *,
    repo_root: Path,
    config_path: str | Path | None = None,
    supervision_path: str | Path | None = None,
) -> tuple[dict[str, Any], SkillGenBaselineDriver]:
    import yaml

    lock = load_lock(repo_root / "experiments" / "baselines" / "baseline_lock.yaml")
    path = Path(config_path or repo_root / "configs" / "baselines" / "b4_skillgen_s.yaml")
    if not path.is_absolute():
        path = repo_root / path
    method = yaml.safe_load(path.read_text(encoding="utf-8"))
    common = yaml.safe_load(
        (repo_root / "configs" / "baselines" / "common.yaml").read_text(encoding="utf-8")
    )
    config = {**dict(common or {}), **dict(method or {})}
    worker_python = Path(str(config.get("worker_python", ".venv_b4_skillgen_s/bin/python")))
    if not worker_python.is_absolute():
        worker_python = repo_root / worker_python
    worker_python = Path(os.path.abspath(worker_python))
    if supervision_path is None:
        env_name = str(dict(config.get("supervision") or {}).get("label_path_env", ""))
        supplied = os.environ.get(env_name, "").strip() if env_name else ""
        if supplied:
            supervision_path = supplied
        else:
            relative = str(
                dict(config.get("supervision") or {}).get(
                    "diagnostic_upstream_label_path", "data/alfworld/all.jsonl"
                )
            )
            supervision_path = repo_root / ".external" / "skillgen" / relative
    label_path = Path(supervision_path).expanduser()
    if not label_path.is_absolute():
        label_path = repo_root / label_path
    label_path = label_path.resolve(strict=True)
    return lock, SkillGenBaselineDriver(
        config=config,
        repo_root=repo_root,
        external_root=repo_root / ".external" / "skillgen",
        lock=lock,
        worker_python=worker_python,
        supervision_path=label_path,
    )
