"""B3 SkillOpt driver: controller-side orchestration of the worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.protocol import sha256_json
from experiments.baselines.bootstrap_external import load_lock, verify_key_files
from experiments.baselines.common.driver import RunContext, SmokeResult, TrainResult
from experiments.baselines.common.formal_validation import (
    verify_final_evaluation_bijection,
)
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.integrity import validate_episode_usage
from experiments.baselines.common.manifest import TaskManifestSet
from experiments.baselines.common.schema import CommonEpisodeRecord
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
        alfworld_env = os.environ.get("ALFWORLD_DATA", "").strip()
        if not alfworld_env:
            raise RuntimeError("ALFWORLD_DATA is not set in the environment")
        if not ctx.alfworld_data.is_dir():
            raise FileNotFoundError(f"ALFWORLD_DATA directory does not exist: {ctx.alfworld_data}")
        worker_data = Path(alfworld_env).expanduser().resolve(strict=True)
        if worker_data != ctx.alfworld_data.resolve(strict=True):
            raise ValueError(
                "RunContext ALFWorld root does not match the ALFWORLD_DATA "
                "inherited by the worker"
            )
        if not ctx.run_id:
            raise ValueError("SkillOpt requires a non-empty immutable run_id")
        if not ctx.resolved_config_path.is_absolute():
            raise ValueError("SkillOpt resolved_config_path must be absolute")
        if not ctx.resolved_config_path.is_file():
            raise FileNotFoundError(
                f"resolved SkillOpt config does not exist: {ctx.resolved_config_path}"
            )
        _validate_resolved_config_identity(ctx)

    def smoke(
        self,
        ctx: RunContext,
        train_manifest: TaskManifestSet,
    ) -> SmokeResult:
        """Exercise the real provider and ALFWorld without constructing a trainer."""

        if ctx.train_manifest_path is None or ctx.validation_manifest_path is None:
            raise ValueError("SkillOpt smoke requires Train and Validation manifests")
        if ctx.test_manifest_path is not None:
            raise ValueError("SkillOpt smoke must not receive a Test manifest")
        _validate_context_manifest_identity(ctx, train_manifest)
        smoke_out = ctx.output_dir / "smoke"
        wire = WorkerWire(
            method=self.method_id,
            phase="smoke",
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=None,
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(smoke_out / "worker_result.json"),
            identity=dict(ctx.identity),
            external_skillopt_root=str(self.external_root),
            skill_init_rel=_SKILL_INIT_REL,
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=smoke_out,
        )
        if not result.get("passed"):
            raise RuntimeError(
                "SkillOpt smoke worker failed: "
                + str(result.get("error") or result.get("worker_exit_code") or "unknown")
            )
        usage = UsageSnapshot.load(smoke_out / "usage.json")
        episodes = _collect_episodes(smoke_out, expected_phases={"smoke"})
        reported_episodes = result.get("episodes")
        reported_rows = result.get("rows")
        if reported_episodes != len(episodes) or reported_rows != len(episodes):
            raise RuntimeError(
                "smoke worker result counts do not match persisted episode sidecars"
            )
        if not episodes:
            raise RuntimeError("SkillOpt smoke worker produced no episode evidence")
        allowed = {task.task_id for task in train_manifest.tasks}
        task_ids = [episode.task_id for episode in episodes]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("smoke has duplicate episode task_id values")
        unexpected = sorted(set(task_ids) - allowed)
        if unexpected:
            raise ValueError(f"smoke contains tasks outside Train manifest: {unexpected}")
        failures = sorted(
            episode.task_id for episode in episodes if episode.infrastructure_failure
        )
        if failures:
            raise RuntimeError(
                "SkillOpt smoke contains infrastructure failures: " + ", ".join(failures)
            )
        validate_episode_usage(episodes)
        # Smoke success itself is intentionally not required.  What this gate
        # proves is a real provider/environment trajectory with complete,
        # unambiguous action evidence.
        _load_action_texts(smoke_out, episodes=episodes)
        return SmokeResult(episodes=episodes, usage=usage)

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
        # A formal controller may bind the held-out manifest digest into the
        # campaign identity.  It is deliberately not passed to the train worker.
        _validate_context_manifest_identity(ctx, train_manifest, validation_manifest)
        train_out = ctx.output_dir / "train"
        wire = WorkerWire(
            method=self.method_id,
            phase="train",
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=None,
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(train_out / "worker_result.json"),
            identity=dict(ctx.identity),
            external_skillopt_root=str(self.external_root),
            skill_init_rel=_SKILL_INIT_REL,
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=train_out,
        )
        if not result.get("passed"):
            raise RuntimeError(
                "SkillOpt train worker failed: "
                + str(result.get("error") or result.get("worker_exit_code") or "unknown")
            )
        best_skill = train_out / "best_skill.md"
        if not best_skill.is_file():
            raise FileNotFoundError(f"train worker produced no best_skill.md: {best_skill}")
        usage = UsageSnapshot.load(train_out / "usage.json")
        episodes = _collect_episodes(
            train_out,
            expected_phases={"train", "validation"},
        )
        train_episodes = [episode for episode in episodes if episode.phase == "train"]
        validation_episodes = [
            episode for episode in episodes if episode.phase == "validation"
        ]
        if not train_episodes or not validation_episodes:
            raise RuntimeError(
                "SkillOpt train worker did not persist both Train and Validation episodes"
            )
        _validate_worker_episode_counts(
            result,
            train_count=len(train_episodes),
            validation_count=len(validation_episodes),
        )
        method_metrics = dict(result.get("train_summary") or {})
        provider_evidence = dict(result.get("provider_evidence") or {})
        if provider_evidence:
            method_metrics["provider_evidence"] = provider_evidence
        return TrainResult(
            episodes=train_episodes,
            validation_episodes=validation_episodes,
            usage=usage,
            persistent_artifact_files={"best_skill.md": best_skill},
            method_metrics=method_metrics,
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

    def evaluate_train(
        self,
        ctx: RunContext,
        frozen: FrozenArtifact,
        train_manifest: TaskManifestSet,
        *,
        frozen_dir: Path | None = None,
    ) -> list[CommonEpisodeRecord]:
        """Evaluate only the frozen best skill on the original Train manifest.

        This is a resubstitution diagnostic, never a held-out/Test result.  It
        executes in an independent worker phase so no optimizer, reflection,
        rejected buffer, or other mutable training state can cross the freeze
        boundary.
        """

        assert_frozen_unchanged(frozen)
        frozen_dir = frozen_dir or (ctx.output_dir / "frozen")
        if ctx.train_manifest_path is None or ctx.validation_manifest_path is None:
            raise ValueError("SkillOpt train_eval requires Train and Validation manifests")
        if ctx.test_manifest_path is not None:
            raise ValueError("SkillOpt train_eval must not receive a Test manifest")
        if frozen.method_id != self.method_id:
            raise ValueError(
                f"frozen artifact method mismatch: expected {self.method_id!r}, "
                f"got {frozen.method_id!r}"
            )
        if frozen.root.resolve() != (Path(frozen_dir) / "artifact").resolve():
            raise ValueError("frozen_dir does not identify the supplied FrozenArtifact")
        if frozen.source_train_manifest_hash != train_manifest.digest:
            raise ValueError("frozen artifact does not match the Train manifest")
        validation_manifest = TaskManifestSet.load(ctx.validation_manifest_path)
        _validate_context_manifest_identity(ctx, train_manifest, validation_manifest)
        if frozen.source_validation_manifest_hash != validation_manifest.digest:
            raise ValueError("frozen artifact does not match the Validation manifest")

        phase = "train_eval"
        phase_out = ctx.output_dir / phase
        phase_identity = {
            **dict(ctx.identity),
            "evaluation_manifest_digest": train_manifest.digest,
            "frozen_artifact_digest": frozen.digest,
        }
        wire = WorkerWire(
            method=self.method_id,
            phase=phase,
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=None,
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(phase_out / "worker_result.json"),
            identity=phase_identity,
            frozen_artifact_path=str(frozen_dir),
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=phase_out,
        )
        if not result.get("passed"):
            raise RuntimeError(
                "SkillOpt train_eval worker failed: "
                + str(result.get("error") or result.get("worker_exit_code") or "unknown")
            )
        if not result.get("frozen_unchanged"):
            raise RuntimeError("frozen artifact changed during train_eval")
        for field in ("frozen_digest_before", "frozen_digest_after"):
            if result.get(field) != frozen.digest:
                raise RuntimeError(
                    f"train_eval worker reported {field}={result.get(field)!r}; "
                    f"expected {frozen.digest!r}"
                )
        assert_frozen_unchanged(frozen)
        episodes = _collect_episodes(phase_out, expected_phases={phase})
        if result.get("episodes") != len(episodes) or result.get("rows") != len(episodes):
            raise RuntimeError(
                "train_eval worker result counts do not match persisted episode sidecars"
            )
        # Identity/bijection is checked once before replay (strict fields are
        # not populated yet), then again after replay with strict fields
        # required.  Infrastructure failures invalidate the result at the
        # first boundary and can never shrink the accuracy denominator.
        verify_final_evaluation_bijection(
            episodes,
            train_manifest,
            role="train",
            expected_phase=phase,
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=False,
            profile=str(self.config.get("protocol_profile", "pilot_v1")),
        )
        episodes = self._apply_strict_evaluation(
            episodes,
            train_manifest,
            ctx,
            phase_root=phase_out,
        )
        verify_final_evaluation_bijection(
            episodes,
            train_manifest,
            role="train",
            expected_phase=phase,
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=True,
            profile=str(self.config.get("protocol_profile", "pilot_v1")),
        )
        assert_frozen_unchanged(frozen)
        return episodes

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
        """Evaluate a prior run's frozen best skill on the held-out Test manifest."""

        assert_frozen_unchanged(frozen)
        if (
            ctx.train_manifest_path is None
            or ctx.validation_manifest_path is None
            or ctx.test_manifest_path is None
        ):
            raise ValueError("SkillOpt test requires Train, Validation, and Test manifests")
        if frozen.method_id != self.method_id:
            raise ValueError("frozen artifact belongs to a different method")
        if frozen.root.resolve() != (Path(frozen_dir) / "artifact").resolve():
            raise ValueError("frozen_dir does not identify the supplied FrozenArtifact")
        if frozen.source_train_manifest_hash != train_manifest.digest:
            raise ValueError("frozen artifact does not match Train manifest")
        if frozen.source_validation_manifest_hash != validation_manifest.digest:
            raise ValueError("frozen artifact does not match Validation manifest")
        _validate_context_manifest_identity(
            ctx, train_manifest, validation_manifest, test_manifest
        )

        phase = "test"
        phase_out = ctx.output_dir / phase
        phase_identity = {
            **dict(ctx.identity),
            "evaluation_manifest_digest": test_manifest.digest,
            "frozen_artifact_digest": frozen.digest,
        }
        wire = WorkerWire(
            method=self.method_id,
            phase=phase,
            manifest_path=str(ctx.train_manifest_path),
            validation_manifest_path=str(ctx.validation_manifest_path),
            test_manifest_path=str(ctx.test_manifest_path),
            config_path=str(ctx.resolved_config_path),
            output_dir=str(ctx.output_dir),
            run_seed=ctx.run_seed,
            model=ctx.model_config.to_wire(),
            run_id=ctx.run_id,
            result_path=str(phase_out / "worker_result.json"),
            identity=phase_identity,
            frozen_artifact_path=str(frozen_dir),
        )
        result = run_worker(
            wire=wire,
            worker_module=_WORKER_MODULE,
            python=self.worker_python,
            wire_dir=phase_out,
        )
        if not result.get("passed"):
            raise RuntimeError(
                "SkillOpt test worker failed: "
                + str(result.get("error") or result.get("worker_exit_code") or "unknown")
            )
        if not result.get("frozen_unchanged"):
            raise RuntimeError("frozen artifact changed during test")
        for field in ("frozen_digest_before", "frozen_digest_after"):
            if result.get(field) != frozen.digest:
                raise RuntimeError(f"test worker reported an invalid {field}")
        assert_frozen_unchanged(frozen)
        episodes = _collect_episodes(phase_out, expected_phases={phase})
        if result.get("episodes") != len(episodes) or result.get("rows") != len(episodes):
            raise RuntimeError("test worker counts do not match episode sidecars")
        verify_final_evaluation_bijection(
            episodes,
            test_manifest,
            role="test",
            expected_phase=phase,
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=False,
            profile=str(self.config.get("protocol_profile", "pilot_v1")),
        )
        episodes = self._apply_strict_evaluation(
            episodes, test_manifest, ctx, phase_root=phase_out
        )
        verify_final_evaluation_bijection(
            episodes,
            test_manifest,
            role="test",
            expected_phase=phase,
            expected_method=self.method_id,
            expected_run_seed=ctx.run_seed,
            expected_artifact_digest=frozen.digest,
            require_strict_outcomes=True,
            profile=str(self.config.get("protocol_profile", "pilot_v1")),
        )
        assert_frozen_unchanged(frozen)
        return episodes

    # ── Strict post-evaluation (Ours harness boundary, controller-side) ───

    def _apply_strict_evaluation(
        self,
        episodes: list[CommonEpisodeRecord],
        manifest: TaskManifestSet,
        ctx: RunContext,
        *,
        phase_root: Path,
    ) -> list[CommonEpisodeRecord]:
        entries = {task.task_id: task for task in manifest.tasks}
        actions_by_task = _load_action_texts(phase_root, episodes=episodes)
        evaluator = StrictTaskEvaluator(ctx.alfworld_data)
        for episode in episodes:
            entry = entries.get(episode.task_id)
            if entry is None:
                raise ValueError(
                    f"train_eval episode {episode.task_id} is not part of Train manifest"
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


def _collect_episodes(
    root: Path,
    *,
    expected_phases: set[str],
) -> list[CommonEpisodeRecord]:
    sidecars = sorted(root.rglob("common_episodes.jsonl"))
    if not sidecars:
        raise FileNotFoundError(
            f"no common episode sidecar exists inside phase directory {root}"
        )
    episodes: list[CommonEpisodeRecord] = []
    for sidecar in sidecars:
        episodes.extend(load_episodes(sidecar))
    unexpected = sorted({episode.phase for episode in episodes} - expected_phases)
    if unexpected:
        raise ValueError(
            f"phase directory {root.name!r} contains episodes from unexpected "
            f"phases: {unexpected}"
        )
    return episodes


def _load_action_texts(
    root: Path,
    *,
    episodes: list[CommonEpisodeRecord],
) -> dict[str, list[str]]:
    """Load exactly one contiguous action sequence for every episode.

    A repeated ``(task_id, step_index)`` in a second sidecar is not a resume;
    it is ambiguous evidence and therefore invalidates the formal replay.
    """

    episode_ids = [episode.task_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("train_eval has duplicate episode task_id values")
    expected_counts = {
        episode.task_id: int(episode.environment_actions)
        for episode in episodes
    }
    sidecars = sorted(root.rglob("common_environment_actions.jsonl"))
    if not sidecars:
        raise FileNotFoundError(
            f"no common environment-action sidecar exists inside {root}"
        )

    actions: dict[str, dict[int, str]] = {}
    for sidecar in sidecars:
        for line_number, line in enumerate(
            sidecar.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"environment-action sidecar {sidecar} is corrupt at "
                    f"line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"environment-action sidecar {sidecar} line {line_number} "
                    "must be a JSON object"
                )
            task_id = str(row.get("episode_task_id", ""))
            if task_id not in expected_counts:
                raise ValueError(
                    f"environment-action sidecar contains unexpected task_id "
                    f"{task_id!r}"
                )
            raw_index = row.get("step_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(
                    f"environment-action step_index for {task_id!r} must be an integer"
                )
            step_index = raw_index
            if step_index < 0:
                raise ValueError(
                    f"environment-action step_index for {task_id!r} is invalid: "
                    f"{raw_index!r}"
                )
            action = row.get("action")
            if not isinstance(action, str) or not action.strip():
                raise ValueError(
                    f"environment-action {task_id!r} step {step_index} has no action"
                )
            task_steps = actions.setdefault(task_id, {})
            if step_index in task_steps:
                raise ValueError(
                    f"environment-action evidence duplicates {task_id!r} "
                    f"step {step_index}"
                )
            task_steps[step_index] = action

    ordered: dict[str, list[str]] = {}
    for task_id in episode_ids:
        steps = actions.get(task_id)
        if steps is None:
            raise ValueError(
                f"train_eval task {task_id!r} has no environment-action evidence"
            )
        indexes = sorted(steps)
        expected = list(range(expected_counts[task_id]))
        if indexes != expected:
            raise ValueError(
                f"environment-action steps for {task_id!r} must be contiguous "
                f"from zero and match episode.environment_actions="
                f"{expected_counts[task_id]}; got {indexes}"
            )
        ordered[task_id] = [steps[index] for index in indexes]
    return ordered


def _validate_worker_episode_counts(
    result: dict[str, Any],
    *,
    train_count: int,
    validation_count: int,
) -> None:
    reported = result.get("episodes")
    if not isinstance(reported, dict):
        raise RuntimeError("SkillOpt train worker result has no episode-count mapping")
    expected = {
        "total": train_count + validation_count,
        "train": train_count,
        "validation": validation_count,
    }
    observed = {
        key: reported.get(key)
        for key in expected
    }
    if observed != expected:
        raise RuntimeError(
            f"SkillOpt train worker episode counts disagree with sidecars: "
            f"expected {expected}, got {observed}"
        )


def _validate_context_manifest_identity(
    ctx: RunContext,
    train_manifest: TaskManifestSet,
    validation_manifest: TaskManifestSet | None = None,
    test_manifest: TaskManifestSet | None = None,
) -> None:
    _validate_resolved_config_identity(ctx)
    expected_train = ctx.identity.get("train_manifest_digest")
    if expected_train != train_manifest.digest:
        raise ValueError(
            "RunContext Train manifest identity does not match the supplied manifest"
        )
    if ctx.train_manifest_path is None:
        raise ValueError("RunContext has no Train manifest path")
    persisted_train = TaskManifestSet.load(ctx.train_manifest_path)
    if persisted_train.digest != train_manifest.digest:
        raise ValueError("RunContext Train manifest path does not match Train identity")

    if ctx.validation_manifest_path is None:
        raise ValueError("RunContext has no Validation manifest path")
    persisted_validation = TaskManifestSet.load(ctx.validation_manifest_path)
    expected_validation = ctx.identity.get("validation_manifest_digest")
    if expected_validation != persisted_validation.digest:
        raise ValueError(
            "RunContext Validation manifest identity does not match its manifest path"
        )
    if (
        validation_manifest is not None
        and validation_manifest.digest != persisted_validation.digest
    ):
        raise ValueError(
            "supplied Validation manifest does not match RunContext identity"
        )
    if test_manifest is not None:
        if ctx.test_manifest_path is None:
            raise ValueError("RunContext has no Test manifest path")
        persisted_test = TaskManifestSet.load(ctx.test_manifest_path)
        if ctx.identity.get("test_manifest_digest") != persisted_test.digest:
            raise ValueError("RunContext Test manifest identity does not match its path")
        if test_manifest.digest != persisted_test.digest:
            raise ValueError("supplied Test manifest does not match RunContext identity")


def _validate_resolved_config_identity(ctx: RunContext) -> None:
    try:
        config_payload = json.loads(ctx.resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"resolved SkillOpt config is unreadable: {ctx.resolved_config_path}"
        ) from exc
    if not isinstance(config_payload, dict):
        raise ValueError("resolved SkillOpt config root must be a mapping")
    actual = sha256_json(config_payload)
    expected = ctx.identity.get("config_digest")
    if expected != actual or ctx.config_hash != actual:
        raise ValueError(
            "RunContext config identity does not match config_resolved.json"
        )


def load_lock_and_driver(
    *,
    repo_root: Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], "SkillOptBaselineDriver"]:
    import yaml

    lock = load_lock(repo_root / "experiments" / "baselines" / "baseline_lock.yaml")
    resolved_config = Path(
        config_path or repo_root / "configs" / "baselines" / "b3_skillopt.yaml"
    ).expanduser()
    if not resolved_config.is_absolute():
        resolved_config = (repo_root / resolved_config).resolve()
    else:
        resolved_config = resolved_config.resolve()
    method_config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    common_config = yaml.safe_load(
        (repo_root / "configs" / "baselines" / "common.yaml").read_text(encoding="utf-8")
    )
    config = {**dict(common_config or {}), **dict(method_config or {})}
    worker_python = Path(str(config.get("worker_python", ".venv_b3_skillopt/bin/python")))
    if not worker_python.is_absolute():
        worker_python = repo_root / worker_python
    # Do not Path.resolve() a venv interpreter: on POSIX ``bin/python`` is a
    # symlink to the base executable, and invoking that resolved target loses
    # the venv's sys.prefix/site-packages (including SkillOpt metadata).
    worker_python = Path(os.path.abspath(worker_python))
    return lock, SkillOptBaselineDriver(
        config=config,
        repo_root=repo_root,
        external_root=repo_root / ".external" / "skillopt",
        lock=lock,
        worker_python=worker_python,
    )
