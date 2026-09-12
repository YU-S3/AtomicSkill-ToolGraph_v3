"""GEPA adapter for the common ALFWorld text-skill executor.

Only ``skill_text`` is evolvable.  Target episodes are delegated to the exact
same :class:`SkillOptTextEpisodeRunner` used by B3; this module only maps GEPA
examples onto that executor and persists B5-labelled common evidence.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.baselines.common.schema import CommonEpisodeRecord


METHOD_ID = "b5_gepa"
_CANDIDATE_KEY = "skill_text"
_ERROR_HINT = re.compile(
    r"(?:nothing happens|invalid action|not a valid|cannot|can't|unable|not found)",
    re.IGNORECASE,
)


class GEPAEvaluationError(RuntimeError):
    """Systemic episode failure that must not be converted to task score zero."""

    def __init__(self, message: str, *, failure_kind: str = "protocol_failure") -> None:
        super().__init__(message)
        self.failure_kind = (
            failure_kind
            if failure_kind in {"infrastructure_failure", "protocol_failure"}
            else "protocol_failure"
        )


def _exception_failure_kind(exc: BaseException) -> str:
    """Normalize child failures without losing provider exhaustion semantics."""

    explicit = str(getattr(exc, "failure_kind", "") or "")
    if explicit in {"infrastructure_failure", "protocol_failure"}:
        return explicit
    infrastructure = getattr(exc, "infrastructure_failure", None)
    if isinstance(infrastructure, bool):
        return "infrastructure_failure" if infrastructure else "protocol_failure"
    return "protocol_failure"


@dataclass(frozen=True)
class _EpisodeResult:
    task: dict[str, Any]
    output: dict[str, Any]
    score: float
    trajectory: dict[str, Any]
    episode: CommonEpisodeRecord
    actions: list[dict[str, Any]]


@dataclass(frozen=True)
class _ProcessEpisodeJob:
    task: dict[str, Any]
    skill_text: str
    group_root: str
    evaluation_id: str
    task_index: int
    capture_traces: bool
    run_seed: int
    max_actions: int
    max_completion_tokens: int
    alfworld_data: str
    execution_phase: str
    model: dict[str, str]
    provider_transport: dict[str, Any]
    run_id: str
    campaign: dict[str, Any] | None


def _run_isolated_episode(job: _ProcessEpisodeJob) -> tuple[Any, str]:
    """Run one target episode in a fresh spawned process.

    GEPA owns optimizer ordering in the parent process.  This boundary owns
    only an independent ALFWorld episode and therefore may be process-parallel.
    Every child installs its own observer and writes a distinct provider-call
    sidecar; the campaign gate remains shared across processes.
    """

    from experiments.baselines.b3_skillopt.episode_runner import (
        SkillOptTextEpisodeRunner,
    )
    from experiments.baselines.b3_skillopt.provider_observer import (
        install_provider_observer,
        uninstall_provider_observer,
    )
    from experiments.baselines.b3_skillopt.worker import _configure_model
    from experiments.baselines.common.model_config import ModelConfig
    from experiments.baselines.common.provider_gate import CampaignProviderGate

    model = ModelConfig.from_mapping(dict(job.model))
    model.validate_formal_identity()
    transport = dict(job.provider_transport)
    _configure_model(model, sdk_max_retries=int(transport["sdk_max_retries"]))
    gate = None
    if job.campaign is not None:
        gate = CampaignProviderGate(
            gate_dir=Path(str(job.campaign["provider_gate_dir"])),
            campaign_id=str(job.campaign["campaign_id"]),
            max_inflight=int(job.campaign["campaign_provider_max_inflight"]),
        )
    evidence_root = Path(job.group_root) / f"process_{job.task_index:04d}"
    evidence_root.mkdir(parents=True, exist_ok=False)
    observer = install_provider_observer(
        output_path=evidence_root / "provider_calls.jsonl",
        method=METHOD_ID,
        phase=job.execution_phase,
        model=model.model,
        reasoning_effort=model.reasoning_effort,
        run_id=job.run_id,
        run_seed=job.run_seed,
        application_retry_limit=int(transport["application_retry_limit"]),
        retry_delays_seconds=list(transport["retry_delays_seconds"]),
        deterministic_jitter_ratio=float(transport["deterministic_jitter_ratio"]),
        expected_sdk_max_retries=int(transport["sdk_max_retries"]),
        campaign_gate=gate,
    )
    try:
        initial_observation = ""
        if job.capture_traces and str(job.task["dataset_role"]) == "train":
            initial_observation = _read_visible_initial_observation(
                job.task,
                run_seed=job.run_seed,
                alfworld_data=job.alfworld_data,
            )
        runner = SkillOptTextEpisodeRunner(
            max_actions=job.max_actions,
            max_completion_tokens=job.max_completion_tokens,
            seed=job.run_seed,
            alfworld_data=job.alfworld_data,
        )
        rollout_id = (
            f"{job.evaluation_id}_{job.task_index:04d}_"
            f"{hashlib.sha256(job.skill_text.encode('utf-8')).hexdigest()[:12]}"
        )
        outcome = runner.run(
            job.task,
            job.skill_text,
            job.group_root,
            rollout_id=rollout_id,
        )
        # ``events`` also checks that the child sidecar exactly matches the
        # observer's in-memory audit trail before it is returned to the parent.
        observer.events()
        return outcome, initial_observation
    finally:
        uninstall_provider_observer(observer)


def manifest_examples(manifest: Any, *, dataset_role: str) -> list[dict[str, Any]]:
    """Convert an immutable common manifest to GEPA's opaque examples."""

    if dataset_role not in {
        "train", "validation", "train_eval", "smoke_test", "test", "smoke"
    }:
        raise ValueError(f"unsupported GEPA dataset role: {dataset_role!r}")
    return [
        {
            "id": task.task_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "source_split": task.source_split,
            "env_index": task.env_index,
            "manifest_index": task.index,
            "gamefile": task.gamefile_rel,
            "gamefile_sha256": task.gamefile_sha256,
            "task_signature": task.task_signature,
            "dataset_role": dataset_role,
        }
        for task in manifest.tasks
    ]


def one_per_family(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the first manifest-ordered example from each ALFWorld family."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in examples:
        example = dict(raw)
        family = str(example.get("task_type", ""))
        if not family or family in seen:
            continue
        selected.append(example)
        seen.add(family)
    return selected


class ALFWorldGEPAAdapter:
    """Execute candidate skills and expose only policy-approved Train traces."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        output_root: str | Path,
        run_seed: int,
        max_actions: int = 100,
        max_completion_tokens: int = 16384,
        workers: int = 1,
        alfworld_data: str | Path = "",
        execution_phase: str = "train",
        artifact_digest_override: str | None = None,
        episode_runner: Any | None = None,
        initial_observation_fn: Callable[[dict[str, Any]], str] | None = None,
        process_context: dict[str, Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_seed = int(run_seed)
        self.max_actions = int(max_actions)
        self.max_completion_tokens = int(max_completion_tokens)
        self.workers = max(1, int(workers))
        self.alfworld_data = str(
            Path(alfworld_data or os.environ.get("ALFWORLD_DATA", "")).expanduser()
        )
        self.execution_phase = str(execution_phase)
        self.artifact_digest_override = _optional_sha256(
            artifact_digest_override, field="artifact_digest_override"
        )
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self._injected_runner = episode_runner is not None
        if self._injected_runner and self.workers != 1:
            raise ValueError("an injected GEPA episode runner requires workers=1")
        self._episode_runner = episode_runner
        self._process_context = dict(process_context or {})
        if not self._injected_runner:
            required = {"model", "provider_transport", "run_id"}
            missing = sorted(required - set(self._process_context))
            if missing:
                raise ValueError(
                    "process-isolated GEPA adapter context is missing: "
                    + ", ".join(missing)
                )
        self._initial_observation_fn = (
            initial_observation_fn or self._read_visible_initial_observation
        )
        self._state_lock = threading.Lock()
        self._initial_lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._process_pool: concurrent.futures.ProcessPoolExecutor | None = None
        self._closed = False
        self._initial_observations: dict[str, str] = {}
        self._evaluation_counter = self._discover_evaluation_counter()
        self._metric_calls = 0
        self._evaluation_batches = 0
        self._reflective_dataset_calls = 0
        self._resumed_metric_calls = 0
        self._resumed_reflection_calls = 0
        self._resume_replay_metric_calls = 0
        self._resume_replay_evaluation_ids: list[str] = []

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Evaluate one candidate/batch pair in input order."""

        return self._evaluate_items(
            [(candidate, batch)], capture_traces=capture_traces
        )[0]

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[dict[str, Any]]]],
    ) -> list[Any]:
        """Flatten independent episodes while preserving GEPA item/task order."""

        return self._evaluate_items(items, capture_traces=True)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build the frozen minimal reflection view from Train evidence only."""

        _validate_candidate(candidate)
        if components_to_update != [_CANDIDATE_KEY]:
            raise ValueError(
                "GEPA may update exactly one component: ['skill_text']"
            )
        trajectories = eval_batch.trajectories
        if trajectories is None or len(trajectories) != len(eval_batch.scores):
            raise ValueError("Train reflection requires aligned trajectories")
        records: list[dict[str, Any]] = []
        for trajectory in trajectories:
            if not isinstance(trajectory, dict):
                raise TypeError("GEPA trajectory must be a mapping")
            if trajectory.get("dataset_role") != "train":
                raise ValueError(
                    "Validation/Test trajectories cannot enter GEPA reflection"
                )
            record = {
                "Inputs": dict(trajectory["Inputs"]),
                "Generated Outputs": dict(trajectory["Generated Outputs"]),
                "Feedback": dict(trajectory["Feedback"]),
            }
            _assert_reflection_schema(record)
            records.append(record)
        with self._state_lock:
            self._reflective_dataset_calls += 1
        return {_CANDIDATE_KEY: records}

    def get_adapter_state(self) -> dict[str, Any]:
        """Persist only counters needed for collision-free GEPA resume."""

        with self._state_lock:
            return {
                "schema_version": 1,
                "evaluation_counter": self._evaluation_counter,
                "metric_calls": self._metric_calls,
                "evaluation_batches": self._evaluation_batches,
                "reflective_dataset_calls": self._reflective_dataset_calls,
            }

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("GEPA adapter state must be a mapping")
        # GEPA calls this hook with ``{}`` on a fresh run after evaluating the
        # seed candidate on Validation.  Empty means "no persisted state" and
        # must not erase those just-recorded metric calls.
        if not state:
            return
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("GEPA adapter state has an unsupported schema")
        with self._state_lock:
            # GEPA v0.1.4 evaluates the seed on full Validation before loading
            # its checkpoint.  That replay is real provider cost, but it is not
            # a new algorithm metric call in the restored GEPA state.
            replay_calls = self._metric_calls
            replay_batches = self._evaluation_batches
            replay_ids = [
                f"evaluation_{index:08d}"
                for index in range(1, replay_batches + 1)
            ]
            resumed_calls = int(state.get("metric_calls", 0))
            resumed_batches = int(state.get("evaluation_batches", 0))
            resumed_reflections = int(state.get("reflective_dataset_calls", 0))
            if min(resumed_calls, resumed_batches, resumed_reflections) < 0:
                raise ValueError("GEPA adapter state counters cannot be negative")
            self._resumed_metric_calls = resumed_calls
            self._resumed_reflection_calls = resumed_reflections
            self._resume_replay_metric_calls += replay_calls
            self._resume_replay_evaluation_ids.extend(replay_ids)
            self._evaluation_counter = max(
                self._evaluation_counter, int(state.get("evaluation_counter", 0))
            )
            self._metric_calls = resumed_calls
            self._evaluation_batches = resumed_batches
            self._reflective_dataset_calls = resumed_reflections

    def metrics(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "metric_calls": self._metric_calls,
                "evaluation_batches": self._evaluation_batches,
                "reflective_dataset_calls": self._reflective_dataset_calls,
                "resumed_metric_calls": self._resumed_metric_calls,
                "resumed_reflection_calls": self._resumed_reflection_calls,
                "resume_replay_metric_calls": self._resume_replay_metric_calls,
                "resume_replay_evaluation_ids": list(
                    self._resume_replay_evaluation_ids
                ),
            }

    def close(self) -> None:
        """Release the phase-scoped process pool after the final evaluation."""

        with self._pool_lock:
            pool = self._process_pool
            self._process_pool = None
            self._closed = True
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    def _pool(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._injected_runner:
            raise RuntimeError("an injected GEPA runner does not use a process pool")
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("GEPA adapter is already closed")
            if self._process_pool is None:
                self._process_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            return self._process_pool

    def _evaluate_items(
        self,
        items: list[tuple[dict[str, str], list[dict[str, Any]]]],
        *,
        capture_traces: bool,
    ) -> list[Any]:
        from gepa.core.adapter import EvaluationBatch

        if not items:
            return []
        groups: list[dict[str, Any]] = []
        jobs: list[tuple[int, int, dict[str, Any], str, Path, str]] = []
        for group_index, raw_item in enumerate(items):
            if not isinstance(raw_item, tuple) or len(raw_item) != 2:
                raise TypeError("batch_evaluate items must be (candidate, batch) pairs")
            candidate, raw_batch = raw_item
            skill_text = _validate_candidate(candidate)
            batch = [dict(example) for example in raw_batch]
            if not batch:
                raise ValueError("GEPA evaluation batch must not be empty")
            roles = {_validate_example(example) for example in batch}
            if len(roles) != 1:
                raise ValueError("a GEPA evaluation batch cannot mix dataset roles")
            dataset_role = next(iter(roles))
            evaluation_id = self._next_evaluation_id()
            group_root = self.output_root / evaluation_id
            group_root.mkdir(parents=True, exist_ok=False)
            candidate_sha = hashlib.sha256(skill_text.encode("utf-8")).hexdigest()
            groups.append(
                {
                    "candidate": dict(candidate),
                    "skill_text": skill_text,
                    "candidate_sha256": candidate_sha,
                    "batch": batch,
                    "dataset_role": dataset_role,
                    "evaluation_id": evaluation_id,
                    "root": group_root,
                }
            )
            for task_index, task in enumerate(batch):
                jobs.append(
                    (
                        group_index,
                        task_index,
                        task,
                        skill_text,
                        group_root,
                        evaluation_id,
                    )
                )

        def execute_local(job: tuple[int, int, dict[str, Any], str, Path, str]):
            group_index, task_index, task, skill_text, group_root, evaluation_id = job
            return group_index, task_index, self._run_one(
                task=task,
                skill_text=skill_text,
                group_root=group_root,
                evaluation_id=evaluation_id,
                task_index=task_index,
                capture_traces=capture_traces,
            )

        completed: list[tuple[int, int, _EpisodeResult]] = []
        failures: list[tuple[int, int, BaseException]] = []
        try:
            if self._injected_runner:
                for job in jobs:
                    group_index, task_index = job[0], job[1]
                    try:
                        completed.append(execute_local(job))
                    except BaseException as exc:
                        failures.append((group_index, task_index, exc))
            else:
                process_jobs = [
                    (
                        group_index,
                        task_index,
                        _ProcessEpisodeJob(
                            task=task,
                            skill_text=skill_text,
                            group_root=str(group_root),
                            evaluation_id=evaluation_id,
                            task_index=task_index,
                            capture_traces=capture_traces,
                            run_seed=self.run_seed,
                            max_actions=self.max_actions,
                            max_completion_tokens=self.max_completion_tokens,
                            alfworld_data=self.alfworld_data,
                            execution_phase=self.execution_phase,
                            model=dict(self._process_context["model"]),
                            provider_transport=dict(
                                self._process_context["provider_transport"]
                            ),
                            run_id=str(self._process_context["run_id"]),
                            campaign=(
                                dict(self._process_context["campaign"])
                                if self._process_context.get("campaign") is not None
                                else None
                            ),
                        ),
                    )
                    for (
                        group_index,
                        task_index,
                        task,
                        skill_text,
                        group_root,
                        evaluation_id,
                    ) in jobs
                ]
                pool = self._pool()
                futures = {
                    pool.submit(_run_isolated_episode, process_job): (
                        group_index,
                        task_index,
                    )
                    for group_index, task_index, process_job in process_jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    group_index, task_index = futures[future]
                    try:
                        outcome, initial_observation = future.result()
                        group = groups[group_index]
                        completed.append((
                            group_index,
                            task_index,
                            self._result_from_outcome(
                                task=group["batch"][task_index],
                                skill_text=group["skill_text"],
                                evaluation_id=group["evaluation_id"],
                                outcome=outcome,
                                initial_observation=initial_observation,
                                capture_traces=capture_traces,
                            ),
                        ))
                    except BaseException as exc:
                        # Drain every submitted future so successful paid calls
                        # can be normalized before the phase fails/resumes.
                        failures.append((group_index, task_index, exc))

            if failures:
                partial_by_group: list[list[_EpisodeResult | None]] = [
                    [None] * len(group["batch"]) for group in groups
                ]
                for group_index, task_index, result in completed:
                    partial_by_group[group_index][task_index] = result
                for group_index, (group, maybe_results) in enumerate(
                    zip(groups, partial_by_group, strict=True)
                ):
                    results = [result for result in maybe_results if result is not None]
                    if results:
                        _persist_group(
                            group,
                            results,
                            complete=len(results) == len(group["batch"]),
                            expected_task_count=len(group["batch"]),
                        )
                    group_failures = [
                        (task_index, exc)
                        for failed_group, task_index, exc in failures
                        if failed_group == group_index
                    ]
                    _write_json_exclusive(
                        group["root"] / "evaluation_failure.json",
                        {
                            "schema_version": 1,
                            "evaluation_id": group["evaluation_id"],
                            "dataset_role": group["dataset_role"],
                            "expected_task_count": len(group["batch"]),
                            "completed_task_count": len(results),
                            "failed_tasks": [
                                {
                                    "task_index": task_index,
                                    "task_id": str(group["batch"][task_index]["task_id"]),
                                    "error_type": type(exc).__name__,
                                    "error": _redacted_error(exc),
                                    "failure_kind": _exception_failure_kind(exc),
                                }
                                for task_index, exc in group_failures
                            ],
                        },
                    )
                # A protocol failure dominates a simultaneous transient outage;
                # otherwise preserve the infrastructure-tagged exception so
                # the worker can admit GEPA run_dir resume from durable proof.
                selected = next(
                    (
                        exc
                        for _, _, exc in failures
                        if _exception_failure_kind(exc) == "protocol_failure"
                    ),
                    failures[0][2],
                )
                raise selected
        except BaseException as exc:
            for group in groups:
                failure_path = group["root"] / "evaluation_failure.json"
                if not failure_path.exists():
                    _write_json_exclusive(
                        failure_path,
                        {
                            "schema_version": 1,
                            "evaluation_id": group["evaluation_id"],
                            "dataset_role": group["dataset_role"],
                            "error_type": type(exc).__name__,
                            "error": _redacted_error(exc),
                        },
                    )
            raise

        by_group: list[list[_EpisodeResult | None]] = [
            [None] * len(group["batch"]) for group in groups
        ]
        for group_index, task_index, result in completed:
            by_group[group_index][task_index] = result

        evaluation_batches: list[Any] = []
        for group, maybe_results in zip(groups, by_group, strict=True):
            if any(result is None for result in maybe_results):
                raise GEPAEvaluationError("GEPA parallel result aggregation is incomplete")
            results = [result for result in maybe_results if result is not None]
            _persist_group(
                group,
                results,
                complete=True,
                expected_task_count=len(group["batch"]),
            )
            trajectories = (
                [result.trajectory for result in results]
                if capture_traces else None
            )
            evaluation_batches.append(
                EvaluationBatch(
                    outputs=[result.output for result in results],
                    scores=[result.score for result in results],
                    trajectories=trajectories,
                    num_metric_calls=len(results),
                )
            )
        with self._state_lock:
            self._metric_calls += len(jobs)
            self._evaluation_batches += len(groups)
        return evaluation_batches

    def _run_one(
        self,
        *,
        task: dict[str, Any],
        skill_text: str,
        group_root: Path,
        evaluation_id: str,
        task_index: int,
        capture_traces: bool,
    ) -> _EpisodeResult:
        dataset_role = str(task["dataset_role"])
        initial_observation = ""
        if capture_traces and dataset_role == "train":
            initial_observation = self._cached_initial_observation(task)
        rollout_id = (
            f"{evaluation_id}_{task_index:04d}_"
            f"{hashlib.sha256(skill_text.encode('utf-8')).hexdigest()[:12]}"
        )
        if self._episode_runner is None:
            raise RuntimeError("local GEPA execution has no injected episode runner")
        outcome = self._episode_runner.run(
            task, skill_text, str(group_root), rollout_id=rollout_id
        )
        return self._result_from_outcome(
            task=task,
            skill_text=skill_text,
            evaluation_id=evaluation_id,
            outcome=outcome,
            initial_observation=initial_observation,
            capture_traces=capture_traces,
        )

    def _result_from_outcome(
        self,
        *,
        task: dict[str, Any],
        skill_text: str,
        evaluation_id: str,
        outcome: Any,
        initial_observation: str,
        capture_traces: bool,
    ) -> _EpisodeResult:
        if outcome.infrastructure_failure:
            raise GEPAEvaluationError(
                "target episode failed outside the ALFWorld task semantics: "
                + str(outcome.infrastructure_error),
                failure_kind=str(outcome.failure_kind or "protocol_failure"),
            )
        row = dict(outcome.skillopt_row)
        dataset_role = str(task["dataset_role"])
        official_success = bool(row.get("hard"))
        conversation = [dict(step) for step in outcome.conversation]
        environment_actions = len(conversation)
        if environment_actions > self.max_actions:
            raise GEPAEvaluationError("target episode exceeded the outer action ceiling")
        skill_sha = hashlib.sha256(skill_text.encode("utf-8")).hexdigest()
        artifact_digest = self._artifact_digest(dataset_role, skill_sha)
        fail_reason = str(row.get("fail_reason", ""))
        termination_reason = _termination_reason(
            official_success, fail_reason, environment_actions, self.max_actions
        )
        phase = self._evidence_phase(dataset_role)
        episode = CommonEpisodeRecord(
            method=METHOD_ID,
            phase=phase,
            run_seed=self.run_seed,
            task_id=str(task["task_id"]),
            task_type=str(task["task_type"]),
            manifest_index=int(task["manifest_index"]),
            gamefile=str(task["gamefile"]),
            gamefile_hash=str(task["gamefile_sha256"]),
            official_success=official_success,
            task_contract_success=None,
            strict_success=None,
            environment_actions=environment_actions,
            invalid_actions=None,
            command_turns=int(row.get("n_turns", environment_actions)),
            timeout=fail_reason.startswith("Timeout"),
            termination_reason=termination_reason,
            target_llm_calls=int(outcome.target_usage.calls),
            target_prompt_tokens=int(outcome.target_usage.prompt_tokens),
            target_completion_tokens=int(outcome.target_usage.completion_tokens),
            target_reasoning_tokens=int(outcome.target_usage.reasoning_tokens),
            evolution_llm_calls=0,
            evolution_prompt_tokens=0,
            evolution_completion_tokens=0,
            evolution_reasoning_tokens=0,
            embedding_calls=0,
            wall_time_ms=int(outcome.wall_time_ms),
            artifact_digest_before=artifact_digest,
            artifact_digest_after=artifact_digest,
            method_metrics={
                "dataset_role": dataset_role,
                "evaluation_id": evaluation_id,
                "skill_sha256": skill_sha,
                "observed_gamefile": str(outcome.actual_gamefile),
                "termination_reason": termination_reason,
            },
            infrastructure_failure=False,
            infrastructure_error="",
        )
        actions = [
            {
                "episode_task_id": str(task["task_id"]),
                "step_index": index,
                "action": str(step.get("action", "")),
                "env_feedback": str(step.get("env_feedback", "")),
                "reward": float(step.get("reward", 0.0)),
                "done": bool(step.get("done", False)),
            }
            for index, step in enumerate(conversation)
        ]
        output = {
            "task_id": str(task["task_id"]),
            "official_success": official_success,
            "termination_reason": episode.method_metrics["termination_reason"],
            "environment_actions": environment_actions,
        }
        if capture_traces and dataset_role == "train":
            trajectory = {
                "dataset_role": "train",
                "Inputs": {
                    "goal": str(row.get("task_description", "")),
                    "initial_observation": initial_observation,
                },
                "Generated Outputs": {
                    "steps": [
                        {
                            "raw_model_response": str(step.get("model_response", "")),
                            "parsed_action": str(step.get("action", "")),
                            "observation": str(step.get("env_feedback", "")),
                            "visible_error": _visible_error(step),
                        }
                        for step in conversation
                    ]
                },
                "Feedback": {
                    "official_success": official_success,
                    "termination_reason": episode.method_metrics["termination_reason"],
                    "environment_actions": environment_actions,
                },
            }
        else:
            # Validation/Test scoring output is intentionally non-reflective.
            trajectory = {"dataset_role": dataset_role}
        return _EpisodeResult(
            task=task,
            output=output,
            score=1.0 if official_success else 0.0,
            trajectory=trajectory,
            episode=episode,
            actions=actions,
        )

    def _artifact_digest(self, dataset_role: str, skill_sha: str) -> str:
        if dataset_role in {"train_eval", "smoke_test", "test"}:
            if self.artifact_digest_override is None:
                raise ValueError(
                    f"{dataset_role} requires an immutable frozen artifact digest"
                )
            return self.artifact_digest_override
        if self.artifact_digest_override not in {None, skill_sha}:
            raise ValueError("mutable GEPA evaluation cannot claim another artifact digest")
        return skill_sha

    def _evidence_phase(self, dataset_role: str) -> str:
        if self.execution_phase == "smoke":
            return "smoke"
        if dataset_role not in {
            "train", "validation", "train_eval", "smoke_test", "test"
        }:
            raise ValueError(f"invalid evidence role for {self.execution_phase}: {dataset_role}")
        return dataset_role

    def _cached_initial_observation(self, task: dict[str, Any]) -> str:
        task_id = str(task["task_id"])
        with self._initial_lock:
            if task_id in self._initial_observations:
                return self._initial_observations[task_id]
        observation = str(self._initial_observation_fn(task)).strip()
        if not observation:
            raise GEPAEvaluationError(
                f"Train task {task_id!r} has no visible initial observation"
            )
        with self._initial_lock:
            self._initial_observations.setdefault(task_id, observation)
            return self._initial_observations[task_id]

    def _read_visible_initial_observation(self, task: dict[str, Any]) -> str:
        """Read the public reset observation without exposing hidden simulator state."""
        return _read_visible_initial_observation(
            task,
            run_seed=self.run_seed,
            alfworld_data=self.alfworld_data,
        )

    def _discover_evaluation_counter(self) -> int:
        highest = 0
        for path in self.output_root.glob("evaluation_*"):
            match = re.fullmatch(r"evaluation_(\d{8})", path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest

    def _next_evaluation_id(self) -> str:
        with self._state_lock:
            self._evaluation_counter += 1
            return f"evaluation_{self._evaluation_counter:08d}"


def _validate_candidate(candidate: Any) -> str:
    if not isinstance(candidate, dict) or set(candidate) != {_CANDIDATE_KEY}:
        raise ValueError("GEPA candidate keys must be exactly {'skill_text'}")
    skill_text = candidate[_CANDIDATE_KEY]
    if not isinstance(skill_text, str) or not skill_text.strip():
        raise ValueError("GEPA candidate skill_text must be non-empty text")
    return skill_text


def _read_visible_initial_observation(
    task: dict[str, Any],
    *,
    run_seed: int,
    alfworld_data: str | Path,
) -> str:
    """Reset the exact manifest game once and return only its public text."""

    from skillopt.envs.alfworld.rollout import build_alfworld_env
    from experiments.baselines.b3_skillopt.episode_runner import (
        _ExactManifestEnvironment,
        _SPLIT_MODES,
    )

    split = str(task["source_split"])
    if split not in _SPLIT_MODES:
        raise ValueError(f"unsupported ALFWorld source split: {split!r}")
    eval_dataset, is_train = _SPLIT_MODES[split]
    base = build_alfworld_env(
        env_num=1,
        eval_dataset=eval_dataset,
        seed=int(run_seed) + int(task.get("env_index", 0)),
        is_train=is_train,
        specific_gamefiles=[str(task["gamefile"])],
    )
    env = _ExactManifestEnvironment(
        base,
        task=task,
        alfworld_data=str(alfworld_data),
        provider_observer=None,
    )
    try:
        observations, _ = env.reset({})
        anchor = observations.get("anchor") if isinstance(observations, dict) else None
        if not isinstance(anchor, list) or len(anchor) != 1:
            raise GEPAEvaluationError(
                "ALFWorld reset did not expose one visible anchor observation"
            )
        return str(anchor[0])
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _validate_example(example: dict[str, Any]) -> str:
    required = {
        "id", "task_id", "task_type", "source_split", "env_index",
        "manifest_index", "gamefile", "gamefile_sha256", "task_signature",
        "dataset_role",
    }
    missing = sorted(required - set(example))
    if missing:
        raise ValueError("GEPA example is missing: " + ", ".join(missing))
    if str(example["id"]) != str(example["task_id"]):
        raise ValueError("GEPA example id/task_id mismatch")
    role = str(example["dataset_role"])
    if role not in {
        "train", "validation", "train_eval", "smoke_test", "test", "smoke"
    }:
        raise ValueError(f"unsupported GEPA dataset role: {role!r}")
    return role


def _assert_reflection_schema(record: dict[str, Any]) -> None:
    if set(record) != {"Inputs", "Generated Outputs", "Feedback"}:
        raise ValueError("GEPA reflection record has unauthorized top-level fields")
    if set(record["Inputs"]) != {"goal", "initial_observation"}:
        raise ValueError("GEPA reflection Inputs schema changed")
    if set(record["Generated Outputs"]) != {"steps"}:
        raise ValueError("GEPA reflection Generated Outputs schema changed")
    if set(record["Feedback"]) != {
        "official_success", "termination_reason", "environment_actions"
    }:
        raise ValueError("GEPA reflection Feedback schema changed")
    for step in record["Generated Outputs"]["steps"]:
        if set(step) != {
            "raw_model_response", "parsed_action", "observation", "visible_error"
        }:
            raise ValueError("GEPA reflection step contains unauthorized fields")


def _termination_reason(
    success: bool, fail_reason: str, actions: int, max_actions: int
) -> str:
    if success:
        return "official_success"
    if actions >= max_actions or fail_reason.startswith("Timeout"):
        return "action_budget"
    return "episode_ended_without_success"


def _visible_error(step: dict[str, Any]) -> str:
    feedback = str(step.get("env_feedback", ""))
    return feedback if _ERROR_HINT.search(feedback) else ""


def _persist_group(
    group: dict[str, Any],
    results: list[_EpisodeResult],
    *,
    complete: bool = True,
    expected_task_count: int | None = None,
) -> None:
    root = Path(group["root"])
    _write_jsonl_exclusive(
        root / "common_episodes.jsonl",
        [result.episode.to_dict() for result in results],
    )
    _write_jsonl_exclusive(
        root / "common_environment_actions.jsonl",
        [action for result in results for action in result.actions],
    )
    _write_json_exclusive(
        root / "evaluation.json",
        {
            "schema_version": 1,
            "method": METHOD_ID,
            "evaluation_id": group["evaluation_id"],
            "dataset_role": group["dataset_role"],
            "candidate_sha256": group["candidate_sha256"],
            "task_ids": [result.episode.task_id for result in results],
            "scores": [result.score for result in results],
            "num_metric_calls": len(results),
            "expected_task_count": (
                int(expected_task_count)
                if expected_task_count is not None else len(results)
            ),
            "complete": bool(complete),
        },
    )


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return normalized


def _redacted_error(exc: BaseException) -> str:
    message = str(exc)
    key = os.environ.get("MODEL_API_KEY", "")
    if key:
        message = message.replace(key, "<redacted>")
    return f"{type(exc).__name__}: {message[:1000]}"
