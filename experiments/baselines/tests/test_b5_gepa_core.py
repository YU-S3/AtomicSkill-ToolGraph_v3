"""Focused B5 GEPA protocol and adapter tests."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from experiments.protocol import sha256_json
from experiments.baselines.b3_skillopt.episode_runner import EpisodeOutcome
from experiments.baselines.b3_skillopt.provider_observer import ProviderCallExhausted
from experiments.baselines.b5_gepa import adapter as adapter_module
from experiments.baselines.b5_gepa import controller as controller_module
from experiments.baselines.b5_gepa import run_seed_campaign as campaign_module
from experiments.baselines.b5_gepa.adapter import (
    ALFWorldGEPAAdapter,
    GEPAEvaluationError,
)
from experiments.baselines.b5_gepa.freeze import freeze_gepa_artifacts
from experiments.baselines.b5_gepa.run_seed_campaign import (
    FORMAL_SEEDS,
    GEPACampaignSpec,
    phase_command,
)
from experiments.baselines.b5_gepa import worker as worker_module
from experiments.baselines.bootstrap_external import source_file_sha256
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.subprocess_worker import WorkerWire
from experiments.baselines.common.usage import RoleUsage, UsageSnapshot
from experiments.baselines.run_seed_campaign import _provider_probe_command


REPO_ROOT = Path(__file__).resolve().parents[3]
_SHA = "a" * 64


def _example(index: int, *, role: str = "train") -> dict[str, object]:
    return {
        "id": f"task_{index}",
        "task_id": f"task_{index}",
        "task_type": f"family_{index}",
        "source_split": "valid_unseen" if "test" in role else "train",
        "env_index": index,
        "manifest_index": index,
        "gamefile": f"json_2.1.1/train/task_{index}/game.tw-pddl",
        "gamefile_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        "task_signature": hashlib.sha256(f"task:{index}".encode()).hexdigest(),
        "dataset_role": role,
    }


def _outcome(task: dict[str, object], *, success: bool = False) -> EpisodeOutcome:
    return EpisodeOutcome(
        task=dict(task),
        skillopt_row={
            "hard": success,
            "n_turns": 1,
            "fail_reason": "" if success else "Timeout: action limit",
            "task_description": "put the object on the receptacle",
        },
        conversation=[{
            "model_response": "<action>look</action>",
            "action": "look",
            "env_feedback": "Nothing happens.",
            "reward": 0.0,
            "done": False,
        }],
        target_usage=RoleUsage(
            calls=1, prompt_tokens=12, completion_tokens=4, reasoning_tokens=2
        ),
        wall_time_ms=7,
        actual_gamefile=str(task["gamefile"]),
    )


class _Runner:
    def run(self, task, skill_text, out_dir, *, rollout_id):
        del skill_text, out_dir, rollout_id
        return _outcome(task)


def _enable_gepa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / ".external/gepa/src"))


def test_candidate_reflection_schema_and_top_level_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gepa(monkeypatch)
    adapter = ALFWorldGEPAAdapter(
        output_root=tmp_path,
        run_seed=42,
        workers=1,
        episode_runner=_Runner(),
        initial_observation_fn=lambda task: f"visible:{task['task_id']}",
    )
    batch = adapter.evaluate([_example(0)], {"skill_text": "do the task"}, True)
    episode = json.loads(
        (tmp_path / "evaluation_00000001/common_episodes.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert episode["termination_reason"] == "action_budget"
    assert episode["target_reasoning_tokens"] == 2
    assert episode["evolution_reasoning_tokens"] == 0
    reflected = adapter.make_reflective_dataset(
        {"skill_text": "do the task"}, batch, ["skill_text"]
    )
    record = reflected["skill_text"][0]
    assert set(record) == {"Inputs", "Generated Outputs", "Feedback"}
    assert set(record["Inputs"]) == {"goal", "initial_observation"}
    assert set(record["Feedback"]) == {
        "official_success", "termination_reason", "environment_actions"
    }

    with pytest.raises(ValueError, match="candidate keys"):
        adapter.evaluate([_example(1)], {"skill_text": "x", "extra": "forbidden"})
    validation = SimpleNamespace(
        trajectories=[{"dataset_role": "validation"}], scores=[0.0]
    )
    with pytest.raises(ValueError, match="Validation/Test"):
        adapter.make_reflective_dataset(
            {"skill_text": "x"}, validation, ["skill_text"]
        )


def test_process_pool_uses_spawn_restores_order_and_persists_for_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gepa(monkeypatch)
    submitted: list[int] = []
    start_methods: list[str] = []
    shutdowns: list[tuple[bool, bool]] = []

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 3
            start_methods.append(mp_context.get_start_method())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, job):
            submitted.append(int(job.task_index))
            return Future(function(job))

        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdowns.append((bool(wait), bool(cancel_futures)))

    monkeypatch.setattr(adapter_module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        adapter_module.concurrent.futures,
        "as_completed",
        lambda futures: list(reversed(list(futures))),
    )
    monkeypatch.setattr(
        adapter_module,
        "_run_isolated_episode",
        lambda job: (_outcome(job.task, success=job.task_index == 1), ""),
    )
    adapter = ALFWorldGEPAAdapter(
        output_root=tmp_path,
        run_seed=42,
        workers=3,
        alfworld_data=tmp_path,
        execution_phase="test",
        artifact_digest_override=_SHA,
        process_context={
            "model": {"model": "deepseek-v4-flash"},
            "provider_transport": {},
            "run_id": "run",
            "campaign": None,
        },
    )
    result = adapter.evaluate(
        [_example(index, role="test") for index in range(3)],
        {"skill_text": "fixed"},
    )
    second = adapter.evaluate(
        [_example(3, role="test")],
        {"skill_text": "fixed"},
    )
    adapter.close()
    adapter.close()
    assert submitted == [0, 1, 2, 0]
    assert start_methods == ["spawn"]
    assert shutdowns == [(True, False)]
    assert result.scores == [0.0, 1.0, 0.0]
    assert second.scores == [0.0]
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "evaluation_00000001/common_episodes.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["task_id"] for row in rows] == ["task_0", "task_1", "task_2"]


def test_late_child_failure_persists_completed_paid_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gepa(monkeypatch)

    class Future:
        def __init__(self, function, job):
            self.function = function
            self.job = job

        def result(self):
            return self.function(self.job)

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            del max_workers, mp_context

        def submit(self, function, job):
            return Future(function, job)

        def shutdown(self, wait=True, *, cancel_futures=False):
            del wait, cancel_futures

    def isolated(job):
        if job.task_index == 1:
            raise GEPAEvaluationError(
                "provider retry budget exhausted",
                failure_kind="infrastructure_failure",
            )
        return _outcome(job.task, success=True), ""

    monkeypatch.setattr(adapter_module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        adapter_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(adapter_module, "_run_isolated_episode", isolated)
    adapter = ALFWorldGEPAAdapter(
        output_root=tmp_path,
        run_seed=42,
        workers=2,
        alfworld_data=tmp_path,
        execution_phase="train",
        process_context={
            "model": {"model": "deepseek-v4-flash"},
            "provider_transport": {},
            "run_id": "run",
            "campaign": None,
        },
    )

    with pytest.raises(GEPAEvaluationError, match="retry budget") as caught:
        adapter.evaluate(
            [_example(0), _example(1)],
            {"skill_text": "fixed"},
            capture_traces=True,
        )
    adapter.close()

    assert caught.value.failure_kind == "infrastructure_failure"
    evaluation = json.loads(
        (tmp_path / "evaluation_00000001/evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation["complete"] is False
    assert evaluation["expected_task_count"] == 2
    assert evaluation["task_ids"] == ["task_0"]
    episodes = [
        json.loads(line)
        for line in (
            tmp_path / "evaluation_00000001/common_episodes.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [episode["task_id"] for episode in episodes] == ["task_0"]
    failure = json.loads(
        (tmp_path / "evaluation_00000001/evaluation_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["completed_task_count"] == 1
    assert failure["failed_tasks"][0]["task_id"] == "task_1"
    assert failure["failed_tasks"][0]["failure_kind"] == "infrastructure_failure"


def test_protocol_failure_dominates_simultaneous_transient_provider_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gepa(monkeypatch)

    class Future:
        def __init__(self, function, job):
            self.function = function
            self.job = job

        def result(self):
            return self.function(self.job)

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            del max_workers, mp_context

        def submit(self, function, job):
            return Future(function, job)

        def shutdown(self, wait=True, *, cancel_futures=False):
            del wait, cancel_futures

    def isolated(job):
        if job.task_index == 0:
            raise ProviderCallExhausted(
                "provider retry budget exhausted",
                failure_code="timeout",
                application_attempts=5,
            )
        if job.task_index == 1:
            raise GEPAEvaluationError("manifest identity mismatch")
        return _outcome(job.task, success=True), ""

    monkeypatch.setattr(adapter_module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        adapter_module.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(adapter_module, "_run_isolated_episode", isolated)
    adapter = ALFWorldGEPAAdapter(
        output_root=tmp_path,
        run_seed=42,
        workers=3,
        alfworld_data=tmp_path,
        execution_phase="train",
        process_context={
            "model": {"model": "deepseek-v4-flash"},
            "provider_transport": {},
            "run_id": "run",
            "campaign": None,
        },
    )

    with pytest.raises(GEPAEvaluationError, match="manifest identity mismatch"):
        adapter.evaluate(
            [_example(0), _example(1), _example(2)],
            {"skill_text": "fixed"},
            capture_traces=True,
        )
    adapter.close()

    failure = json.loads(
        (tmp_path / "evaluation_00000001/evaluation_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["failure_kind"] for row in failure["failed_tasks"]] == [
        "infrastructure_failure",
        "protocol_failure",
    ]
    episodes = [
        json.loads(line)
        for line in (
            tmp_path / "evaluation_00000001/common_episodes.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [episode["task_id"] for episode in episodes] == ["task_2"]


def test_pinned_gepa_fresh_optimize_preserves_initial_validation_metric_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gepa(monkeypatch)
    from gepa import optimize

    adapter = ALFWorldGEPAAdapter(
        output_root=tmp_path / "evaluations",
        run_seed=42,
        workers=1,
        episode_runner=_Runner(),
        initial_observation_fn=lambda task: f"visible:{task['task_id']}",
    )
    result = optimize(
        seed_candidate={"skill_text": "initial"},
        trainset=[_example(0, role="train")],
        valset=[_example(1, role="validation")],
        adapter=adapter,
        reflection_lm=lambda prompt: "unused",
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        skip_perfect_score=True,
        batch_sampler="epoch_shuffled",
        reflection_minibatch_size=3,
        perfect_score=1.0,
        module_selector="round_robin",
        use_merge=False,
        max_metric_calls=1,
        cache_evaluation=False,
        seed=42,
        val_evaluation_policy="full_eval",
        acceptance_criterion="strict_improvement",
        sampling_strategy=None,
        selection_strategy=None,
        run_dir=str(tmp_path / "gepa_state"),
        raise_on_exception=True,
        display_progress_bar=False,
    )

    assert result.total_metric_calls == 1
    assert adapter.metrics()["metric_calls"] == 1
    assert adapter.metrics()["resume_replay_metric_calls"] == 0


def test_pinned_gepa_resume_keeps_cumulative_full_val_and_counts_seed_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_gepa(monkeypatch)
    from gepa import optimize

    class ImprovingRunner:
        def run(self, task, skill_text, out_dir, *, rollout_id):
            del out_dir, rollout_id
            return _outcome(task, success=skill_text == "better")

    trainset = [_example(index, role="train") for index in range(3)]
    valset = [_example(index + 10, role="validation") for index in range(2)]
    run_dir = tmp_path / "gepa_state"

    def adapter(root: str) -> ALFWorldGEPAAdapter:
        return ALFWorldGEPAAdapter(
            output_root=tmp_path / root,
            run_seed=42,
            workers=1,
            episode_runner=ImprovingRunner(),
            initial_observation_fn=lambda task: f"visible:{task['task_id']}",
        )

    def run(current: ALFWorldGEPAAdapter):
        return optimize(
            seed_candidate={"skill_text": "initial"},
            trainset=trainset,
            valset=valset,
            adapter=current,
            reflection_lm=lambda prompt: "better",
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            skip_perfect_score=True,
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=3,
            perfect_score=1.0,
            module_selector="round_robin",
            use_merge=False,
            max_metric_calls=10,
            cache_evaluation=False,
            seed=42,
            val_evaluation_policy="full_eval",
            acceptance_criterion="strict_improvement",
            sampling_strategy=None,
            selection_strategy=None,
            run_dir=str(run_dir),
            raise_on_exception=True,
            display_progress_bar=False,
        )

    first_adapter = adapter("first_attempt")
    first = run(first_adapter)
    assert first.total_metric_calls == 10
    assert first.num_full_val_evals == 2
    assert first_adapter.metrics()["metric_calls"] == 10

    resumed_adapter = adapter("resumed_attempt")
    resumed = run(resumed_adapter)
    metrics = resumed_adapter.metrics()
    assert resumed.total_metric_calls == 10
    assert resumed.num_full_val_evals == 2
    assert metrics["metric_calls"] == 10
    assert metrics["resumed_metric_calls"] == 10
    assert metrics["reflective_dataset_calls"] == 1
    assert metrics["resumed_reflection_calls"] == 1
    assert metrics["resume_replay_metric_calls"] == 2
    resumed_rows = list((tmp_path / "resumed_attempt").glob("evaluation_*"))
    assert len(resumed_rows) == 1


def test_resume_summary_reconciles_cumulative_and_attempt_reflection_calls(
    tmp_path: Path,
) -> None:
    result = SimpleNamespace(
        to_dict=lambda: {"schema_version": 1},
        best_candidate={"skill_text": "best"},
        candidates=[{"skill_text": "initial"}, {"skill_text": "best"}],
        total_metric_calls=54,
        num_full_val_evals=2,
        val_aggregate_scores=[0.0, 1.0],
        best_idx=1,
        parents=[[None], [0]],
        discovery_eval_counts=[0, 30],
        per_val_instance_best_candidates={"validation_0": {1}},
        val_subscores=[{"validation_0": 0.0}, {"validation_0": 1.0}],
    )
    callback = SimpleNamespace(summary=lambda: {
        "iteration_count": 7,
        "accepted_candidate_count": 1,
        "num_full_val_evals": 1,
        "reflection_calls": 2,
        "unexpected_merge_events": 0,
    })
    adapter_metrics = {
        "metric_calls": 54,
        "resumed_metric_calls": 42,
        "resume_replay_metric_calls": 24,
        "reflective_dataset_calls": 7,
        "resumed_reflection_calls": 5,
    }
    adapter = SimpleNamespace(metrics=lambda: dict(adapter_metrics))
    phase_dir = tmp_path / "train"
    phase_dir.mkdir()

    _, summary = worker_module._write_optimization_artifacts(
        phase_dir=phase_dir,
        result=result,
        callback=callback,
        reflection_lm=SimpleNamespace(calls=2),
        adapter=adapter,
        configured_budget=720,
        initial_skill_sha256=_SHA,
        validation_size=24,
        count_tokens_fn=lambda text: len(text),
    )

    assert summary["reflection_calls"] == 7
    assert summary["attempt_reflection_calls"] == 2
    assert summary["resumed_reflection_calls"] == 5

    adapter_metrics["resumed_reflection_calls"] = 6
    with pytest.raises(RuntimeError, match="cumulative reflection count"):
        worker_module._write_optimization_artifacts(
            phase_dir=tmp_path / "invalid",
            result=result,
            callback=callback,
            reflection_lm=SimpleNamespace(calls=2),
            adapter=adapter,
            configured_budget=720,
            initial_skill_sha256=_SHA,
            validation_size=24,
            count_tokens_fn=lambda text: len(text),
        )


def _wire(tmp_path: Path, *, phase: str, config: dict | None = None) -> WorkerWire:
    identity = {
        "config_digest": sha256_json(config or {}),
        "train_manifest_digest": "1" * 64,
        "validation_manifest_digest": "2" * 64,
        "external_source_digest": "3" * 64,
        "skillopt_source_digest": "4" * 64,
    }
    values: dict[str, object] = {
        "manifest_path": None,
        "validation_manifest_path": None,
        "test_manifest_path": None,
        "initial_skill_path": None,
        "frozen_artifact_path": None,
    }
    if phase in {"smoke", "train"}:
        identity["initial_skill_digest"] = "5" * 64
        values.update({
            "manifest_path": str(tmp_path / "train.json"),
            "validation_manifest_path": str(tmp_path / "validation.json"),
            "initial_skill_path": str(tmp_path / "initial.md"),
        })
    else:
        identity.update({
            "evaluation_manifest_digest": "6" * 64,
            "frozen_artifact_digest": "7" * 64,
        })
        values["frozen_artifact_path"] = str(tmp_path / "frozen")
        if phase == "train_eval":
            values["manifest_path"] = str(tmp_path / "train.json")
        else:
            identity["test_manifest_digest"] = "6" * 64
            values["test_manifest_path"] = str(tmp_path / "test.json")
    return WorkerWire(
        method="b5_gepa",
        phase=phase,
        config_path=str(tmp_path / "config.json"),
        output_dir=str(tmp_path / "run"),
        run_seed=42,
        model={"model": "deepseek-v4-flash", "reasoning_effort": "high"},
        run_id="b5_test",
        result_path=str(tmp_path / f"run/{phase}/worker_result.json"),
        identity=identity,
        external_method_root=str(tmp_path / "gepa"),
        external_skillopt_root=str(tmp_path / "skillopt"),
        **values,
    )


def _write_child_provider_event(
    wire: WorkerWire, *, failure_code: str, run_id: str | None = None
) -> Path:
    path = (
        Path(wire.output_dir)
        / wire.phase
        / "evaluations/evaluation_00000001/process_0000/provider_calls.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = 5
    event = {
        "schema_version": 2,
        "event": "provider_call",
        "method": "b5_gepa",
        "phase": wire.phase,
        "run_id": run_id or wire.run_id,
        "run_seed": wire.run_seed,
        "model": "deepseek-v4-flash",
        "reasoning_effort": "high",
        "call_id": "child_call_1",
        "status": "failed",
        "application_attempts": attempts,
        "retry_limit": attempts,
        "sdk_boundary_attempts": attempts,
        "failure_code_counts": {failure_code: attempts},
        "last_failure_code": failure_code,
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    return path


def test_child_transient_provider_sidecar_admits_infrastructure_failure(
    tmp_path: Path,
) -> None:
    wire = _wire(tmp_path, phase="train")
    path = _write_child_provider_event(wire, failure_code="timeout")
    error = GEPAEvaluationError("exhausted", failure_kind="infrastructure_failure")

    kind, receipt = worker_module._gepa_failure_kind(error, None, wire)

    assert kind == "infrastructure_failure"
    assert receipt is not None
    assert receipt["failed_calls"] == 1
    assert receipt["failure_codes"] == ["timeout"]
    assert receipt["sidecars"] == [{
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }]


@pytest.mark.parametrize(
    ("failure_code", "run_id"),
    [("authentication", None), ("timeout", "wrong_run")],
)
def test_child_provider_sidecar_fails_closed_for_permanent_or_wrong_identity(
    tmp_path: Path, failure_code: str, run_id: str | None
) -> None:
    wire = _wire(tmp_path, phase="train")
    _write_child_provider_event(wire, failure_code=failure_code, run_id=run_id)
    error = GEPAEvaluationError("failed", failure_kind="infrastructure_failure")

    kind, receipt = worker_module._gepa_failure_kind(error, None, wire)

    assert kind == "protocol_failure"
    assert receipt is None


def test_provider_usage_aggregates_reflection_reasoning_tokens(tmp_path: Path) -> None:
    wire = _wire(tmp_path, phase="smoke")

    def event(call_id: str, role: str, reasoning: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event": "provider_call",
            "method": "b5_gepa",
            "phase": "smoke",
            "run_id": "b5_test",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
            "call_id": call_id,
            "role": role,
            "stage": "gepa_reflection" if role == "optimizer" else "rollout",
            "status": "succeeded",
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
            "reasoning_tokens": reasoning,
            "reasoning_tokens_status": "reported",
            "application_attempts": 1,
            "sdk_boundary_attempts": 1,
            "requested_retry_limit": 5,
            "retry_limit": 5,
            "failure_code_counts": {},
            "last_failure_code": None,
            "recovered": False,
            "provider_slot_ids": [],
        }

    observer = SimpleNamespace(
        events=lambda: [event("target", "target", 2), event("reflect", "optimizer", 4)]
    )
    usage = worker_module._provider_usage(
        observer, wire=wire, wall_time_ms=9
    )
    assert usage.target.reasoning_tokens == 2
    assert usage.evolution.reasoning_tokens == 4
    assert usage.per_stage["gepa_reflection"].reasoning_tokens == 4


def _manifest(size: int, role: str) -> SimpleNamespace:
    tasks = [
        SimpleNamespace(
            task_id=f"{role}_{index}",
            task_type=f"family_{index % 6}",
            source_split="train" if role == "train" else "valid_seen",
            env_index=index,
            index=index,
            gamefile_rel=f"json_2.1.1/{role}/task_{index}/game.tw-pddl",
            gamefile_sha256=hashlib.sha256(f"g:{role}:{index}".encode()).hexdigest(),
            task_signature=hashlib.sha256(f"s:{role}:{index}".encode()).hexdigest(),
        )
        for index in range(size)
    ]
    return SimpleNamespace(tasks=tasks)


def test_optimize_call_uses_frozen_classic_arguments_and_smoke_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    fake_gepa = ModuleType("gepa")

    def optimize(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    fake_gepa.optimize = optimize
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa)

    class Adapter:
        def close(self):
            pass

    class Reflection:
        def __init__(self, **kwargs):
            self.calls = 1

    monkeypatch.setattr(worker_module, "_build_adapter", lambda **kwargs: Adapter())
    monkeypatch.setattr(worker_module, "DeepSeekReflectionLM", Reflection)
    monkeypatch.setattr(
        worker_module,
        "_write_optimization_artifacts",
        lambda **kwargs: ({}, {
            "actual_total_metric_calls": 1,
            "resumed_metric_calls": 0,
            "resume_replay_metric_calls": 0,
            "num_full_val_evals": 0,
            "attempt_full_val_evals_including_resume_replay": 0,
        }),
    )
    episode = SimpleNamespace(
        method="b5_gepa",
        phase="smoke",
        run_seed=42,
        infrastructure_failure=False,
        environment_actions=1,
        target_llm_calls=1,
        method_metrics={"dataset_role": "train"},
    )
    monkeypatch.setattr(worker_module, "_collect_episodes", lambda root: [episode])
    monkeypatch.setattr(worker_module, "_validate_episode_action_evidence", lambda root: None)
    events = SimpleNamespace(events=lambda: [{"event": "provider_call"}])
    monkeypatch.setattr(worker_module, "_provider_event_view", lambda **kwargs: events)
    monkeypatch.setattr(
        worker_module,
        "_provider_usage",
        lambda *args, **kwargs: UsageSnapshot(
            target=RoleUsage(calls=1, prompt_tokens=1, completion_tokens=1),
            evolution=RoleUsage(calls=1, prompt_tokens=1, completion_tokens=1),
        ),
    )
    monkeypatch.setattr(worker_module, "_provider_evidence_summary", lambda view: {})
    config = {
        "experiment_kind": "smoke",
        "env": {"max_steps": 2, "max_completion_tokens": 128, "workers": 6, "max_api_workers": 6},
        "gepa": {"reflection_max_completion_tokens": 128},
        "provider_transport": {"application_retry_limit": 5},
    }
    wire = _wire(tmp_path, phase="smoke", config=config)
    result = worker_module._run_optimization(
        wire=wire,
        config=config,
        train=_manifest(6, "train"),
        validation=_manifest(6, "validation"),
        initial_skill="initial",
        initial_digest=_SHA,
        observer=SimpleNamespace(),
        configured_budget=24,
    )
    assert result["optimizer_constructed"] is True
    assert captured["seed_candidate"] == {"skill_text": "initial"}
    assert captured["max_metric_calls"] == 24
    assert captured["candidate_selection_strategy"] == "pareto"
    assert captured["frontier_type"] == "instance"
    assert captured["batch_sampler"] == "epoch_shuffled"
    assert captured["reflection_minibatch_size"] == 3
    assert captured["acceptance_criterion"] == "strict_improvement"
    assert captured["sampling_strategy"] is None
    assert captured["selection_strategy"] is None
    assert captured["use_merge"] is False
    assert captured["cache_evaluation"] is False
    assert Path(str(captured["run_dir"])).name == "gepa_state"


def test_resume_reconciles_current_validation_sidecars_not_cumulative_full_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_gepa = ModuleType("gepa")
    fake_gepa.optimize = lambda **kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa)

    class Reflection:
        def __init__(self, **kwargs):
            self.calls = 0

    summary = {
        # The checkpoint already contains two historical full-Val evaluations.
        # This attempt performs only six new Train calls plus the unavoidable
        # 24-example seed-Val replay that precedes GEPA state restoration.
        "actual_total_metric_calls": 60,
        "resumed_metric_calls": 54,
        "resume_replay_metric_calls": 24,
        "num_full_val_evals": 2,
        "attempt_full_val_evals_including_resume_replay": 1,
    }
    adapter = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(worker_module, "_build_adapter", lambda **kwargs: adapter)
    monkeypatch.setattr(worker_module, "DeepSeekReflectionLM", Reflection)
    monkeypatch.setattr(
        worker_module,
        "_write_optimization_artifacts",
        lambda **kwargs: ({}, dict(summary)),
    )
    current_attempt_episodes = [SimpleNamespace()] * 30
    monkeypatch.setattr(
        worker_module, "_collect_episodes", lambda root: current_attempt_episodes
    )
    monkeypatch.setattr(
        worker_module, "_validate_episode_action_evidence", lambda root: None
    )
    monkeypatch.setattr(
        worker_module,
        "_validate_episodes",
        lambda episodes, **kwargs: {"train": 6, "validation": 24},
    )
    events = SimpleNamespace(events=lambda: [{"event": "provider_call"}])
    monkeypatch.setattr(worker_module, "_provider_event_view", lambda **kwargs: events)
    monkeypatch.setattr(
        worker_module,
        "_provider_usage",
        lambda *args, **kwargs: UsageSnapshot(
            target=RoleUsage(calls=30, prompt_tokens=30, completion_tokens=30)
        ),
    )
    monkeypatch.setattr(worker_module, "_provider_evidence_summary", lambda view: {})
    config = {
        "experiment_kind": "formal",
        "env": {
            "max_steps": 100,
            "max_completion_tokens": 16384,
            "workers": 16,
            "max_api_workers": 16,
        },
        "gepa": {"reflection_max_completion_tokens": 16384},
        "provider_transport": {"application_retry_limit": 5},
    }
    wire = _wire(tmp_path, phase="train", config=config)

    result = worker_module._run_optimization(
        wire=wire,
        config=config,
        train=_manifest(120, "train"),
        validation=_manifest(24, "validation"),
        initial_skill="initial",
        initial_digest=_SHA,
        observer=SimpleNamespace(),
        configured_budget=720,
    )

    assert result["episodes"] == {"total": 30, "train": 6, "validation": 24}
    assert result["train_summary"] == summary


def test_formal_config_freezes_720_metric_calls_and_same_initial_skill() -> None:
    common = yaml.safe_load(
        (REPO_ROOT / "configs/baselines/common.yaml").read_text(encoding="utf-8")
    )
    b5 = yaml.safe_load(
        (REPO_ROOT / "configs/baselines/b5_gepa.yaml").read_text(encoding="utf-8")
    )
    smoke = yaml.safe_load(
        (REPO_ROOT / "configs/baselines/b5_gepa_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    lock = yaml.safe_load(
        (REPO_ROOT / "experiments/baselines/baseline_lock.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert b5["gepa"]["max_metric_calls"] == 720
    assert b5["gepa"]["use_merge"] is False
    assert b5["gepa"]["sampling_strategy"] is None
    assert b5["gepa"]["selection_strategy"] is None
    assert b5["parallel"] == {
        "seed_lanes": 1,
        "episode_workers_per_seed": 16,
        "test_workers_per_seed": 16,
        "campaign_provider_max_inflight": 16,
        "mp_start_method": "spawn",
    }
    assert smoke["parallel"] == {
        "seed_lanes": 1,
        "episode_workers_per_seed": 6,
        "test_workers_per_seed": 6,
        "campaign_provider_max_inflight": 6,
        "mp_start_method": "spawn",
    }
    assert common["model"]["reasoning_effort"] == "high"
    initial = REPO_ROOT / ".external/skillopt/skillopt/envs/alfworld/skills/initial.md"
    assert source_file_sha256(
        initial,
        algorithm=lock["skillopt"]["runtime_tree"]["algorithm"],
    ) == lock["skillopt"]["key_files"][
        "skillopt/envs/alfworld/skills/initial.md"
    ]


def test_initial_skill_identity_uses_locked_line_ending_policy(tmp_path: Path) -> None:
    root = tmp_path / "skillopt"
    initial = root / "skillopt/envs/alfworld/skills/initial.md"
    initial.parent.mkdir(parents=True)
    initial.write_bytes(b"first line\r\nsecond line\r\n")
    algorithm = "sha256-path-canonical-content-v2"
    digest = source_file_sha256(initial, algorithm=algorithm)
    text, observed, path = worker_module._initial_skill(
        SimpleNamespace(initial_skill_path=str(initial)),
        {"skillopt": {"root": str(root)}},
        {
            "skillopt": {
                "runtime_tree": {"algorithm": algorithm},
                "key_files": {
                    "skillopt/envs/alfworld/skills/initial.md": digest,
                },
            }
        },
    )

    assert observed == digest
    assert path == initial.resolve()
    assert text == "first line\nsecond line\n"


def test_run_manifest_persists_frozen_protocol_authorities(tmp_path: Path) -> None:
    model = ModelConfig.from_mapping({
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "MODEL_API_KEY",
        "reasoning_effort": "high",
    })
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    test_path = tmp_path / "test.json"
    ctx = SimpleNamespace(
        output_dir=tmp_path / "run",
        run_id="b5_train_test",
        run_seed=42,
        identity={"initial_skill_digest": "a" * 64},
        code_hash="b" * 64,
        campaign={"campaign_provider_max_inflight": 16},
        resume=None,
        model_config=model,
        max_environment_actions=100,
        train_manifest_path=train_path,
        validation_manifest_path=validation_path,
        test_manifest_path=test_path,
    )
    train = SimpleNamespace(digest="1" * 64, tasks=[1] * 120)
    validation = SimpleNamespace(digest="2" * 64, tasks=[1] * 24)
    test = SimpleNamespace(digest="3" * 64, tasks=[1] * 134)
    data_signature = {"resolved_data_root": "/data/alfworld", "digest": "4" * 64}
    controller_module._write_run_identity(
        ctx=ctx,
        lock={
            "gepa": {
                "repo": "https://github.com/gepa-ai/gepa",
                "tag": "v0.1.4",
                "commit": "8" * 40,
                "runtime_tree": {"sha256": "5" * 64},
            },
            "skillopt": {"repo": "skillopt", "commit": "6" * 40},
        },
        config={
            "experiment_kind": "formal",
            "protocol_profile": "formal_v2",
            "gepa": {"max_metric_calls": 720},
            "parallel": {
                "seed_lanes": 1,
                "episode_workers_per_seed": 16,
                "campaign_provider_max_inflight": 16,
            },
            "env": {"workers": 16, "max_api_workers": 16},
        },
        phase="train",
        train=train,
        validation=validation,
        test=test,
        data_signature=data_signature,
        git_state={"commit": "7" * 40, "dirty": False},
        receipts={"train": {}, "validation": {}, "test": {}},
        python_runtime={"alfworld_distribution_version": "0.4.2"},
    )

    manifest = json.loads(
        (ctx.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"] == "protocol-faithful-matched-train-v2"
    assert manifest["protocol_profile"] == "formal_v2"
    assert manifest["controller_commit"] == "7" * 40
    assert manifest["train_manifest_path"] == str(train_path)
    assert manifest["train_manifest_hash"] == train.digest
    assert manifest["validation_manifest_path"] == str(validation_path)
    assert manifest["validation_manifest_hash"] == validation.digest
    assert manifest["test_manifest_path"] == str(test_path)
    assert manifest["test_manifest_hash"] == test.digest
    assert manifest["alfworld_package_version"] == "0.4.2"
    assert manifest["alfworld_data_signature"] == data_signature
    assert manifest["model"] == "deepseek-v4-flash"
    assert manifest["model_identity"] == model.to_wire()
    assert manifest["episode_workers"] == 16
    assert manifest["provider_max_inflight"] == 16
    assert manifest["method_specific_prior"] is False
    assert "frozen_artifact_digest" not in manifest

    files: dict[str, Path] = {}
    for name in (
        "best_skill.md",
        "gepa_result.json",
        "candidate_lineage.json",
        "pareto_metadata.json",
        "gepa_audit_summary.json",
    ):
        path = tmp_path / "freeze_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("skill" if name.endswith(".md") else "{}", encoding="utf-8")
        files[name] = path
    frozen = freeze_gepa_artifacts(
        source_files=files,
        frozen_dir=tmp_path / "frozen",
        train_manifest_hash=train.digest,
        validation_manifest_hash=validation.digest,
        metadata={"run_seed": 42},
    )
    descriptor = controller_module._bind_run_manifest_to_frozen(ctx, frozen)
    bound = json.loads(
        (ctx.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert bound["frozen_artifact_digest"] == frozen.digest
    assert bound["frozen_artifact"] == descriptor
    assert descriptor["source_train_manifest_hash"] == train.digest
    assert descriptor["source_validation_manifest_hash"] == validation.digest


def test_frozen_test_persists_posthoc_rows_and_binds_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    families = list(controller_module.ALFWORLD_FORMAL_TASK_TYPES)
    train = SimpleNamespace(digest="1" * 64)
    validation = SimpleNamespace(digest="2" * 64)
    test = SimpleNamespace(
        digest="3" * 64,
        tasks=[
            SimpleNamespace(task_id=f"task_{index}", task_type=family, index=index)
            for index, family in enumerate(families)
        ],
    )
    files: dict[str, Path] = {}
    for name in (
        "best_skill.md",
        "gepa_result.json",
        "candidate_lineage.json",
        "pareto_metadata.json",
        "gepa_audit_summary.json",
    ):
        path = tmp_path / "freeze_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("skill" if name.endswith(".md") else "{}", encoding="utf-8")
        files[name] = path
    frozen = freeze_gepa_artifacts(
        source_files=files,
        frozen_dir=tmp_path / "source_run" / "frozen",
        train_manifest_hash=train.digest,
        validation_manifest_hash=validation.digest,
        metadata={"run_seed": 42},
    )
    model = ModelConfig.from_mapping({
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "MODEL_API_KEY",
        "reasoning_effort": "high",
    })
    output = tmp_path / "test_run"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({
            "method": "b5_gepa",
            "phase": "test",
            "run_id": "b5_test_fixture",
            "run_seed": 42,
            "train_manifest_hash": train.digest,
            "validation_manifest_hash": validation.digest,
        }),
        encoding="utf-8",
    )
    test_dir = output / "test"
    test_dir.mkdir()
    UsageSnapshot(
        target=RoleUsage(calls=6, prompt_tokens=60, completion_tokens=12),
        api_cost_unpriced=True,
    ).save(test_dir / "usage.json")
    (test_dir / "worker_result.json").write_text(
        json.dumps({
            "optimizer_constructed": False,
            "provider_evidence": {"application_attempts": 6},
        }),
        encoding="utf-8",
    )
    episodes = []
    for index, family in enumerate(families):
        episode = CommonEpisodeRecord(
            method="b5_gepa",
            phase="test",
            run_seed=42,
            task_id=f"task_{index}",
            task_type=family,
            manifest_index=index,
            gamefile=f"task_{index}/game.tw-pddl",
            gamefile_hash=f"{index + 1:064x}",
            official_success=index % 2 == 0,
            environment_actions=index + 1,
            target_llm_calls=1,
            target_prompt_tokens=10,
            target_completion_tokens=2,
            artifact_digest_before=frozen.digest,
            artifact_digest_after=frozen.digest,
        )
        episode.set_posthoc_outcome(contract_consistency=True)
        episodes.append(episode)

    class Driver:
        @staticmethod
        def evaluate_test(*args, **kwargs):
            del args, kwargs
            return episodes

    ctx = SimpleNamespace(
        output_dir=output,
        run_id="b5_test_fixture",
        run_seed=42,
        model_config=model,
    )
    monkeypatch.setattr(
        controller_module,
        "_load_frozen_source",
        lambda *args, **kwargs: (frozen, frozen.root.parent),
    )
    monkeypatch.setattr(
        controller_module, "verify_final_evaluation_bijection", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        controller_module, "validate_episode_usage", lambda *args, **kwargs: None
    )
    report = controller_module._run_test(
        Driver(), ctx, train, validation, test, tmp_path / "source_run"
    )

    evaluated = test_dir / "evaluated_common_episodes.jsonl"
    assert evaluated.is_file()
    first = json.loads(evaluated.read_text(encoding="utf-8").splitlines()[0])
    assert first["contract_consistency"] is True
    assert first["common_strict_success"] is True
    assert report["effectiveness"]["official_success_rate"] == 0.5
    assert report["effectiveness"]["cost_metrics_status"] == (
        "unavailable_without_pricing_authority"
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_artifact_digest"] == frozen.digest


def test_campaign_descriptor_requires_one_serial_seed_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ModelConfig.from_mapping({
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "MODEL_API_KEY",
        "reasoning_effort": "high",
    })
    config = {
        "parallel": {
            "seed_lanes": 1,
            "campaign_provider_max_inflight": 16,
            "mp_start_method": "spawn",
        }
    }
    lock = {
        "gepa": {"commit": "a" * 40, "runtime_tree": {"sha256": "b" * 64}},
        "skillopt": {
            "commit": "c" * 40,
            "runtime_tree": {"sha256": "d" * 64},
            "key_files": {
                "skillopt/envs/alfworld/skills/initial.md": "e" * 64,
            },
        },
    }
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    path = campaign_root / "campaign_lock.json"
    payload = {
        "schema_version": 1,
        "method": "b5_gepa",
        "seeds": [42, 43, 44],
        "train_manifest_digest": "1" * 64,
        "validation_manifest_digest": "2" * 64,
        "test_manifest_digest": "3" * 64,
        "controller_commit": "4" * 40,
        "controller_code_digest": "5" * 64,
        "external_gepa_commit": "a" * 40,
        "external_runtime_tree_digest": "b" * 64,
        "external_skillopt_commit": "c" * 40,
        "skillopt_runtime_tree_digest": "d" * 64,
        "initial_skill_sha256": "e" * 64,
        "model": "deepseek-v4-flash",
        "reasoning_effort": "high",
        "formal_config_digest": controller_module._formal_config_digest(config),
        "campaign_provider_max_inflight": 16,
        "seed_lanes": 1,
        "parallel": {
            "seed_lanes": 1,
            "campaign_provider_max_inflight": 16,
            "mp_start_method": "spawn",
        },
        "provider_gate_dir": str(campaign_root / "provider_gate"),
        "campaign_id": "campaign",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        controller_module,
        "_validate_campaign_probe_receipt",
        lambda *args, **kwargs: None,
    )
    manifests = [
        SimpleNamespace(digest="1" * 64),
        SimpleNamespace(digest="2" * 64),
        SimpleNamespace(digest="3" * 64),
    ]

    descriptor = controller_module._load_campaign_descriptor(
        path,
        config=config,
        lock=lock,
        model=model,
        train=manifests[0],
        validation=manifests[1],
        test=manifests[2],
        seed=42,
        git_state={"commit": "4" * 40},
        code_digest="5" * 64,
    )
    assert descriptor["campaign_provider_max_inflight"] == 16

    payload["seed_lanes"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seed_lanes"):
        controller_module._load_campaign_descriptor(
            path,
            config=config,
            lock=lock,
            model=model,
            train=manifests[0],
            validation=manifests[1],
            test=manifests[2],
            seed=42,
            git_state={"commit": "4" * 40},
            code_digest="5" * 64,
        )

    payload["seed_lanes"] = 1
    payload["parallel"]["mp_start_method"] = "fork"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mp_start_method"):
        controller_module._load_campaign_descriptor(
            path,
            config=config,
            lock=lock,
            model=model,
            train=manifests[0],
            validation=manifests[1],
            test=manifests[2],
            seed=42,
            git_state={"commit": "4" * 40},
            code_digest="5" * 64,
        )


def test_b5_provider_probe_binds_pinned_skillopt_source(tmp_path: Path) -> None:
    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    payload = {
        "provider_probe_python": str(spec.python),
        "provider_gate_dir": str(tmp_path / "gate"),
        "campaign_id": "campaign",
        "campaign_run_id": "run",
        "model_identity": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "MODEL_API_KEY",
            "reasoning_effort": "high",
        },
        "provider_probe": {
            "concurrency": 16,
            "requests": 32,
            "max_completion_tokens": 256,
        },
        "campaign_provider_max_inflight": 16,
        "retry_policy": {
            "sdk_max_retries": 0,
            "attempts": 5,
            "delays": [2, 5, 10, 20],
            "jitter_ratio": 0.1,
        },
    }

    command = _provider_probe_command(spec, payload, tmp_path / "probe")

    assert command.count("--skillopt-root") == 1
    assert Path(command[command.index("--skillopt-root") + 1]) == (
        tmp_path / ".external/skillopt"
    ).resolve()


@pytest.mark.parametrize("selected_cap", [12, 8])
def test_b5_provider_fallback_binds_all_worker_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_cap: int,
) -> None:
    config_dir = tmp_path / "configs/baselines"
    config_dir.mkdir(parents=True)
    (config_dir / "common.yaml").write_text(
        yaml.safe_dump({"model": {"model": "deepseek-v4-flash"}}),
        encoding="utf-8",
    )
    source_config = config_dir / "b5_gepa.yaml"
    source_payload = {
        "parallel": {
            "seed_lanes": 1,
            "episode_workers_per_seed": 16,
            "test_workers_per_seed": 16,
            "campaign_provider_max_inflight": 16,
            "mp_start_method": "spawn",
        },
        "provider_probe": {"concurrency": 16, "requests": 32},
        "env": {"workers": 16, "max_api_workers": 16},
    }
    source_config.write_text(yaml.safe_dump(source_payload), encoding="utf-8")
    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=source_config,
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    spec.output_dir.mkdir()
    runtime_config = spec.output_dir / f"campaign_config_global_{selected_cap:02d}.yaml"
    partial_payload = json.loads(json.dumps(source_payload))
    partial_payload["parallel"]["campaign_provider_max_inflight"] = selected_cap
    partial_payload["provider_probe"]["concurrency"] = selected_cap
    runtime_config.write_text(yaml.safe_dump(partial_payload), encoding="utf-8")
    runtime_spec = GEPACampaignSpec(
        method=spec.method,
        seeds=spec.seeds,
        train_manifest=spec.train_manifest,
        validation_manifest=spec.validation_manifest,
        test_manifest=spec.test_manifest,
        config=runtime_config,
        output_dir=spec.output_dir,
        python=spec.python,
        repo_root=spec.repo_root,
    )
    selected_payload = {
        "parallel": dict(partial_payload["parallel"]),
        "provider_probe": {
            **dict(partial_payload["provider_probe"]),
            "passed": True,
            "selected_cap": selected_cap,
        },
        "campaign_provider_max_inflight": selected_cap,
        "config_path": str(runtime_config.resolve()),
        "formal_config_digest": "stale-partial-digest",
    }
    monkeypatch.setattr(
        campaign_module,
        "_shared_run_provider_probe_with_fallback",
        lambda *args, **kwargs: (
            runtime_spec,
            selected_payload,
            {"passed": True, "concurrency": selected_cap},
        ),
    )

    resolved_spec, payload, report = (
        campaign_module._run_provider_probe_with_fallback(
            spec,
            selected_payload,
            command_runner=lambda *args, **kwargs: 0,
        )
    )

    generated = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert source_config.read_text(encoding="utf-8") == yaml.safe_dump(source_payload)
    assert generated["parallel"] == {
        "seed_lanes": 1,
        "episode_workers_per_seed": selected_cap,
        "test_workers_per_seed": selected_cap,
        "campaign_provider_max_inflight": selected_cap,
        "mp_start_method": "spawn",
    }
    assert generated["env"]["workers"] == selected_cap
    assert generated["env"]["max_api_workers"] == selected_cap
    assert generated["provider_probe"]["concurrency"] == selected_cap
    assert payload["parallel"] == generated["parallel"]
    assert payload["campaign_provider_max_inflight"] == selected_cap
    assert payload["provider_probe"]["concurrency"] == selected_cap
    assert payload["config_path"] == str(runtime_config.resolve())
    assert payload["formal_config_digest"] == campaign_module._formal_config_digest(
        campaign_module._merged_config(resolved_spec)
    )
    assert report == {"passed": True, "concurrency": selected_cap}
    command = phase_command(
        resolved_spec,
        phase="train",
        seed=42,
        output_dir=spec.output_dir / "seed_42/train_attempt_1",
        campaign_lock=spec.output_dir / "campaign_lock.json",
    )
    assert Path(command[command.index("--config") + 1]) == runtime_config


def test_b5_provider_cap_16_validates_without_rewriting_source_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs/baselines"
    config_dir.mkdir(parents=True)
    (config_dir / "common.yaml").write_text("{}\n", encoding="utf-8")
    source_config = config_dir / "b5_gepa.yaml"
    source_payload = {
        "parallel": {
            "seed_lanes": 1,
            "episode_workers_per_seed": 16,
            "test_workers_per_seed": 16,
            "campaign_provider_max_inflight": 16,
            "mp_start_method": "spawn",
        },
        "provider_probe": {"concurrency": 16, "requests": 32},
        "env": {"workers": 16, "max_api_workers": 16},
    }
    source_bytes = yaml.safe_dump(source_payload).encode("utf-8")
    source_config.write_bytes(source_bytes)
    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=source_config,
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    selected_payload = {
        "parallel": dict(source_payload["parallel"]),
        "provider_probe": {
            **dict(source_payload["provider_probe"]),
            "passed": True,
            "selected_cap": 16,
        },
        "campaign_provider_max_inflight": 16,
        "config_path": str(source_config),
        "formal_config_digest": campaign_module._formal_config_digest(
            campaign_module._merged_config(spec)
        ),
    }
    monkeypatch.setattr(
        campaign_module,
        "_shared_run_provider_probe_with_fallback",
        lambda *args, **kwargs: (spec, selected_payload, {"passed": True}),
    )

    resolved_spec, payload, _ = campaign_module._run_provider_probe_with_fallback(
        spec,
        selected_payload,
        command_runner=lambda *args, **kwargs: 0,
    )

    assert resolved_spec.config == source_config
    assert payload["campaign_provider_max_inflight"] == 16
    assert source_config.read_bytes() == source_bytes


def test_freeze_and_three_seed_phase_commands(tmp_path: Path) -> None:
    files: dict[str, Path] = {}
    for name in (
        "best_skill.md", "gepa_result.json", "candidate_lineage.json",
        "pareto_metadata.json", "gepa_audit_summary.json",
    ):
        path = tmp_path / name
        path.write_text("skill" if name.endswith(".md") else "{}", encoding="utf-8")
        files[name] = path
    frozen = freeze_gepa_artifacts(
        source_files=files,
        frozen_dir=tmp_path / "frozen",
        train_manifest_hash="1" * 64,
        validation_manifest_hash="2" * 64,
    )
    assert_frozen_unchanged(frozen)

    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    lock = tmp_path / "campaign_lock.json"
    train = phase_command(
        spec,
        phase="train",
        seed=42,
        output_dir=tmp_path / "seed_42/train_attempt_2",
        campaign_lock=lock,
        resume_source_run=tmp_path / "seed_42/train_attempt_1",
    )
    test = phase_command(
        spec,
        phase="test",
        seed=42,
        output_dir=tmp_path / "seed_42/test",
        campaign_lock=lock,
        source_run=tmp_path / "seed_42/train_attempt_2",
    )
    assert train[train.index("--phase") + 1] == "train"
    assert "--resume-source-run" in train and "--source-run" not in train
    assert test[test.index("--phase") + 1] == "test"
    assert "--source-run" in test and "--resume-source-run" not in test
    with pytest.raises(ValueError, match="42, 43, 44"):
        GEPACampaignSpec(
            method="b5_gepa",
            seeds=(42,),
            train_manifest=spec.train_manifest,
            validation_manifest=spec.validation_manifest,
            test_manifest=spec.test_manifest,
            config=spec.config,
            output_dir=spec.output_dir,
            python=spec.python,
            repo_root=spec.repo_root,
        )


def test_b5_campaign_cost_and_method_statistics_use_one_seed_definition() -> None:
    lanes = []
    for seed in FORMAL_SEEDS:
        lanes.append({
            "seed": seed,
            "training_cost": {
                "train_pool_size": 120,
                "validation_pool_size": 24,
                "unique_train_tasks_evaluated": 120,
                "unique_validation_tasks_evaluated": 24,
                "train_episodes": {"episodes": 2, "environment_actions": 7},
                "validation_episodes": {"episodes": 1, "environment_actions": 3},
                "usage": {
                    "target": {
                        "calls": seed,
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "reasoning_tokens": 2,
                    },
                    "evolution": {
                        "calls": 1,
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "reasoning_tokens": 1,
                    },
                    "embedding_calls": 0,
                    "wall_time_ms": 5,
                },
                "api_cost": None,
                "api_cost_unpriced": True,
                "provider_evidence": {
                    "application_attempts": seed + 1,
                    "physical_provider_calls": seed + 1,
                },
                "gepa_metrics": {
                    "configured_max_metric_calls": 720,
                    "actual_total_metric_calls": seed,
                    "budget_overshoot": 0,
                    "candidate_count": seed - 40,
                    "accepted_candidate_count": seed - 41,
                    "num_full_val_evals": 2,
                    "reflection_calls": seed - 40,
                    "best_validation_score": (seed - 40) / 10,
                    "best_skill_tokens": 100 + seed,
                    "best_skill_sha256": f"{seed:064x}",
                },
            },
            "test_cost": {
                "episodes": {"episodes": 1, "environment_actions": 4},
                "usage": {
                    "target": {
                        "calls": 2,
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "reasoning_tokens": 1,
                    },
                    "evolution": {},
                    "embedding_calls": 0,
                    "wall_time_ms": 2,
                },
                "api_cost": None,
                "api_cost_unpriced": True,
                "provider_evidence": {
                    "application_attempts": 2,
                    "physical_provider_calls": 2,
                },
            },
        })

    phase = campaign_module._phase_cost_summary(lanes)
    assert phase["training"]["mean_std"]["target_llm_calls"] == {
        "mean": 43.0,
        "std": 1.0,
        "std_ddof": 1,
        "values_by_seed": {"42": 42, "43": 43, "44": 44},
    }
    assert phase["combined"]["by_seed"]["42"]["llm_tokens"] == 25
    assert phase["combined"]["by_seed"]["42"]["environment_actions"] == 14
    assert "api_cost" not in phase["combined"]["mean_std"]

    method = campaign_module._gepa_method_summary(lanes)
    assert method["mean_std"]["candidate_count"]["mean"] == 3.0
    assert method["mean_std"]["candidate_count"]["std"] == 1.0
    assert method["best_skill_sha256_by_seed"]["44"] == f"{44:064x}"


@pytest.mark.parametrize(
    ("has_checkpoint", "expected_mode", "expects_resume"),
    [
        (False, "fresh_initial", False),
        (True, "resume_checkpoint", True),
    ],
)
def test_infrastructure_retry_uses_gepa_checkpoint_only_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_checkpoint: bool,
    expected_mode: str,
    expects_resume: bool,
) -> None:
    commands: list[list[str]] = []

    def argument(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def runner(command, *, cwd, log_path):
        del cwd
        command = list(command)
        commands.append(command)
        output = Path(argument(command, "--output-dir"))
        output.mkdir(parents=True, exist_ok=False)
        log_path.write_text("attempt\n", encoding="utf-8")
        if argument(command, "--phase") == "train" and len(commands) == 1:
            (output / "failure.json").write_text(
                json.dumps({"failure_kind": "infrastructure_failure"}),
                encoding="utf-8",
            )
            if has_checkpoint:
                checkpoint = output / "train/gepa_state/gepa_state.bin"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
            return 1
        if argument(command, "--phase") == "train":
            (output / "report.json").write_text(
                json.dumps({"training_cost": {}}), encoding="utf-8"
            )
        return 0

    monkeypatch.setattr(
        campaign_module,
        "_validate_train",
        lambda root, seed: SimpleNamespace(digest="f" * 64),
    )
    monkeypatch.setattr(
        campaign_module,
        "_validate_test",
        lambda root, *, seed, train_root, frozen: {
            "test_cost": {},
            "effectiveness": {},
        },
    )
    monkeypatch.setattr(campaign_module, "_resource_usage", lambda *args: {})
    cost_calls: list[dict[str, object]] = []

    def cost_accounting(**kwargs):
        cost_calls.append(dict(kwargs))
        return {}

    monkeypatch.setattr(campaign_module, "_cost_accounting", cost_accounting)
    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    spec.output_dir.mkdir()
    lock = tmp_path / "campaign_lock.json"
    lock.write_text("{}", encoding="utf-8")
    report = campaign_module._run_lane(
        spec,
        seed=42,
        campaign_lock=lock,
        lock_digest=campaign_module._sha256_file(lock),
        command_runner=runner,
    )

    assert report["passed"] is True, report
    assert report["train_attempts"][0]["retry_mode"] == expected_mode
    second_train = commands[1]
    assert ("--resume-source-run" in second_train) is expects_resume
    if expects_resume:
        assert Path(argument(second_train, "--resume-source-run")) == (
            spec.output_dir / "seed_42/train_attempt_1"
        )
        assert cost_calls[0]["resume_sources"] == [
            spec.output_dir / "seed_42/train_attempt_1"
        ]
    else:
        assert cost_calls[0]["resume_sources"] == []


def test_three_seed_campaign_runs_train_freeze_then_test_per_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    commands_lock = threading.Lock()

    def argument(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def runner(command, *, cwd, log_path):
        del cwd
        command = list(command)
        phase = argument(command, "--phase")
        seed = int(argument(command, "--seed"))
        output = Path(argument(command, "--output-dir"))
        output.mkdir(parents=True, exist_ok=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{phase}:{seed}\n", encoding="utf-8")
        with commands_lock:
            commands.append(command)
        if phase == "train":
            phase_dir = output / "train"
            phase_dir.mkdir()
            files: dict[str, Path] = {}
            for name in (
                "best_skill.md",
                "gepa_result.json",
                "candidate_lineage.json",
                "pareto_metadata.json",
                "gepa_audit_summary.json",
            ):
                path = phase_dir / name
                path.write_text(
                    f"skill-{seed}" if name.endswith(".md") else "{}",
                    encoding="utf-8",
                )
                files[name] = path
            frozen = freeze_gepa_artifacts(
                source_files=files,
                frozen_dir=output / "frozen",
                train_manifest_hash="1" * 64,
                validation_manifest_hash="2" * 64,
                metadata={"run_seed": seed},
            )
            write_json(
                output / "completion.json",
                {"passed": True, "phase": "train"},
            )
            write_json(
                output / "report.json",
                {
                    "passed": True,
                    "method": "b5_gepa",
                    "frozen": {"digest": frozen.digest},
                    "training_cost": {
                        "seed": seed,
                        "usage": {},
                        "train_episodes": {},
                        "validation_episodes": {},
                        "gepa_metrics": {
                            "configured_max_metric_calls": 720,
                            "actual_total_metric_calls": seed,
                            "budget_overshoot": 0,
                            "candidate_count": 2,
                            "accepted_candidate_count": 1,
                            "num_full_val_evals": 1,
                            "reflection_calls": 1,
                            "best_validation_score": seed / 100.0,
                            "best_skill_tokens": seed,
                            "best_skill_sha256": hashlib.sha256(
                                f"skill-{seed}".encode("utf-8")
                            ).hexdigest(),
                        },
                    },
                },
            )
            write_json(output / "run_manifest.json", {
                "method": "b5_gepa",
                "phase": "train",
                "run_seed": seed,
                "frozen_artifact_digest": frozen.digest,
                "frozen_artifact": {
                    "root": str(frozen.root.resolve()),
                    "digest": frozen.digest,
                    "source_train_manifest_hash": frozen.source_train_manifest_hash,
                    "source_validation_manifest_hash": (
                        frozen.source_validation_manifest_hash
                    ),
                },
            })
        else:
            source_run = Path(argument(command, "--source-run"))
            frozen = FrozenArtifact.load(source_run / "frozen")
            write_json(
                output / "completion.json",
                {"passed": True, "phase": "test"},
            )
            write_json(
                output / "test_report.json",
                {
                    "passed": True,
                    "protocol": {"source_run": str(source_run.resolve())},
                    "frozen": {"digest": frozen.digest},
                    "test_cost": {"seed": seed},
                    "effectiveness": {"official_success_rate": seed / 100.0},
                },
            )
            write_json(output / "run_manifest.json", {
                "method": "b5_gepa",
                "phase": "test",
                "run_seed": seed,
                "frozen_artifact_digest": frozen.digest,
                "frozen_artifact": {
                    "root": str(frozen.root.resolve()),
                    "digest": frozen.digest,
                    "source_train_manifest_hash": frozen.source_train_manifest_hash,
                    "source_validation_manifest_hash": (
                        frozen.source_validation_manifest_hash
                    ),
                },
            })
        return 0

    monkeypatch.setattr(
        campaign_module,
        "_run_provider_probe_with_fallback",
        lambda spec, payload, *, command_runner: (
            spec,
            dict(payload),
            {"passed": True},
        ),
    )
    paper_inputs: list[dict[str, list[Path]]] = []

    def build_paper_report(method_runs, **kwargs):
        paper_inputs.append({
            method: [Path(path) for path in roots]
            for method, roots in method_runs.items()
        })
        assert kwargs["expected_seeds"] == FORMAL_SEEDS
        values = {"42": 0.42, "43": 0.43, "44": 0.44}
        return {
            "schema_version": 1,
            "source": "read_only_completed_test_episode_artifacts",
            "paired_task_key": ["task_id", "run_seed"],
            "bootstrap": {"samples": 10_000},
            "methods": {
                "b5_gepa": {
                    "mean_std": {
                        "official_success_rate": {
                            "status": "available",
                            "mean": 0.43,
                            "std": 0.01,
                            "std_ddof": 1,
                            "values_by_seed": values,
                        }
                    },
                    "family_official_success_rate_mean_std": {
                        "pick_and_place_simple": {
                            "status": "available",
                            "mean": 0.5,
                            "std": 0.0,
                            "std_ddof": 1,
                            "values_by_seed": {
                                "42": 0.5, "43": 0.5, "44": 0.5
                            },
                        }
                    },
                }
            },
            "paired_task_rows": [],
            "ours_pairwise": {},
        }

    monkeypatch.setattr(
        campaign_module, "build_campaign_report", build_paper_report
    )
    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    source = {"commit": "c" * 40, "dirty": False, "code_digest": "d" * 64}
    report = campaign_module.run_campaign(
        spec,
        command_runner=runner,
        source_inspector=lambda repo: dict(source),
        lock_builder=lambda *args: {"campaign_id": "test_campaign"},
    )

    assert report["passed"] is True, report
    assert report["completed_seeds"] == [42, 43, 44]
    assert report["failed_seeds"] == []
    assert report["test_metrics_mean_std"]["official_success_rate"]["values_by_seed"] == {
        "42": 0.42,
        "43": 0.43,
        "44": 0.44,
    }
    assert report["test_per_family_mean_std"]["pick_and_place_simple"]["mean"] == 0.5
    assert report["paper_report"]["sha256"] == campaign_module._sha256_file(
        spec.output_dir / "paper_report.json"
    )
    assert paper_inputs == [{
        "b5_gepa": [
            spec.output_dir / f"seed_{seed}" / "test" for seed in FORMAL_SEEDS
        ]
    }]
    assert report["test_metrics_mean_std"]["official_success_rate"]["std_ddof"] == 1
    assert report["gepa_method_metrics"]["mean_std"][
        "actual_total_metric_calls"
    ]["values_by_seed"] == {"42": 42, "43": 43, "44": 44}
    assert report["phase_costs"]["statistics_definition"]["std_ddof"] == 1
    assert report["phase_costs"]["combined"]["mean_std"]["llm_tokens"][
        "mean"
    ] == 0.0
    assert len(commands) == 6
    assert [
        (int(argument(command, "--seed")), argument(command, "--phase"))
        for command in commands
    ] == [
        (42, "train"),
        (42, "test"),
        (43, "train"),
        (43, "test"),
        (44, "train"),
        (44, "test"),
    ]
    for seed in FORMAL_SEEDS:
        seed_commands = [
            command
            for command in commands
            if int(argument(command, "--seed")) == seed
        ]
        assert [argument(command, "--phase") for command in seed_commands] == [
            "train",
            "test",
        ]
        assert "--source-run" not in seed_commands[0]
        assert Path(argument(seed_commands[1], "--source-run")) == (
            spec.output_dir / f"seed_{seed}" / "train_attempt_1"
        )


def test_b5_initial_validation_infrastructure_failure_retries_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = GEPACampaignSpec(
        method="b5_gepa",
        seeds=FORMAL_SEEDS,
        train_manifest=tmp_path / "train.json",
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "venv/bin/python",
        repo_root=tmp_path,
    )
    spec.output_dir.mkdir()
    lock = spec.output_dir / "campaign_lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    lock_digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    commands: list[list[str]] = []
    captured_cost: dict[str, object] = {}

    def runner(command, *, cwd, log_path):
        del cwd
        command = list(command)
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        phase = command[command.index("--phase") + 1]
        output.mkdir(parents=True)
        phase_dir = output / phase
        phase_dir.mkdir()
        log_path.write_text("fixture\n", encoding="utf-8")
        call_count = len(commands)
        UsageSnapshot(target=RoleUsage(calls=call_count)).save(
            phase_dir / "usage.json"
        )
        if phase == "train" and output.name == "train_attempt_1":
            (output / "failure.json").write_text(
                json.dumps({"failure_kind": "infrastructure_failure"}),
                encoding="utf-8",
            )
            return 1
        if phase == "train":
            (output / "report.json").write_text("{}\n", encoding="utf-8")
        return 0

    def cost_accounting(**kwargs):
        captured_cost.update(kwargs)
        return {"captured": True}

    monkeypatch.setattr(
        campaign_module, "_validate_train", lambda root, seed: SimpleNamespace(digest=_SHA)
    )
    monkeypatch.setattr(campaign_module, "_validate_test", lambda *args, **kwargs: {})
    monkeypatch.setattr(campaign_module, "_resource_usage", lambda *args: {})
    monkeypatch.setattr(campaign_module, "_cost_accounting", cost_accounting)

    lane = campaign_module._run_lane(
        spec,
        seed=42,
        campaign_lock=lock,
        lock_digest=lock_digest,
        command_runner=runner,
    )

    assert lane["passed"] is True
    assert lane["train_attempts"][0]["retry_mode"] == "fresh_initial"
    train_commands = [
        command for command in commands
        if command[command.index("--phase") + 1] == "train"
    ]
    assert len(train_commands) == 2
    assert all("--resume-source-run" not in command for command in train_commands)
    first = spec.output_dir / "seed_42/train_attempt_1"
    assert captured_cost["failed_attempt_sources"] == [first]
    assert captured_cost["resume_sources"] == []
    assert lane["actual_attempt_usage"]["usage"]["target"]["calls"] == 6


def test_failed_attempt_cost_is_charged_once_and_binds_partial_actions(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "failed"
    evidence_dir = failed / "train/evaluations/evaluation_00000001"
    evidence_dir.mkdir(parents=True)
    event = {
        "cache_reused": False,
        "application_attempts": 5,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "provider_queue_wait_ms": 2,
        "provider_service_latency_ms": 4,
        "logical_call_latency_ms": 9,
        "retry_backoff_ms": 3,
    }
    (evidence_dir / "provider_calls.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    episode = CommonEpisodeRecord(
        method="b5_gepa",
        phase="validation",
        run_seed=42,
        task_id="validation_1",
        task_type="pick_and_place_simple",
        manifest_index=0,
        gamefile="game.tw-pddl",
        gamefile_hash=_SHA,
        official_success=False,
        environment_actions=4,
    )
    (evidence_dir / "common_episodes.jsonl").write_text(
        json.dumps(episode.to_dict()) + "\n", encoding="utf-8"
    )
    committed = {
        "api_calls": 1,
        "logical_api_calls": 1,
        "provider_retries": 0,
        "physical_provider_calls": 1,
        "cached_provider_calls": 0,
        "tokens": 5,
        "environment_actions": 2,
        "provider_queue_wait_ms": 0,
        "provider_service_latency_ms": 0,
        "logical_call_latency_ms": 0,
        "retry_backoff_ms": 0,
        "api_cost": None,
        "api_cost_unpriced": True,
    }

    fresh = campaign_module._cost_accounting(
        seed=42,
        resume_sources=[],
        failed_attempt_sources=[failed],
        train_root=tmp_path / "current",
        test_root=tmp_path / "test",
        committed=committed,
    )
    resumed = campaign_module._cost_accounting(
        seed=42,
        resume_sources=[failed],
        failed_attempt_sources=[failed],
        train_root=tmp_path / "current",
        test_root=tmp_path / "test",
        committed=committed,
    )

    for costs in (fresh, resumed):
        actual = costs["actual_campaign_cost"]
        assert actual["api_calls"] == 6
        assert actual["logical_api_calls"] == 2
        assert actual["tokens"] == 15
        assert actual["environment_actions"] is None
        assert actual["observed_environment_actions_lower_bound"] == 6
        assert actual["failed_attempt_observed_environment_actions"] == 4
        assert actual["failed_attempt_count"] == 1
        assert len(costs["evidence"]["failed_attempt_provider_sidecars"]) == 1
        assert len(costs["evidence"]["failed_attempt_episode_sidecars"]) == 1
    assert fresh["actual_campaign_cost"]["resume_parent_count"] == 0
    assert fresh["resume_replay_overhead"]["measurement_complete"] is True
    assert resumed["actual_campaign_cost"]["resume_parent_count"] == 1
    assert resumed["resume_replay_overhead"]["measurement_complete"] is False


def test_failed_attempt_action_journal_makes_environment_cost_exact(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "failed"
    evidence_dir = failed / "train/evaluations/evaluation_00000001"
    process_dir = evidence_dir / "process_0000"
    process_dir.mkdir(parents=True)
    (process_dir / "provider_calls.jsonl").write_text(
        json.dumps({
            "cache_reused": False,
            "application_attempts": 5,
            "prompt_tokens": 7,
            "completion_tokens": 3,
        }) + "\n",
        encoding="utf-8",
    )
    rollout_id = "evaluation_00000001_0000_candidate"
    task_id = "train_1"
    header = {
        "schema_version": 1,
        "event": "action_journal_started",
        "rollout_id": rollout_id,
        "episode_task_id": task_id,
        "manifest_index": 0,
        "gamefile": "game.tw-pddl",
        "actual_gamefile": "/data/game.tw-pddl",
    }
    rows = [header]
    for index, action in enumerate(("look", "go to table 1")):
        identity = {
            "rollout_id": rollout_id,
            "episode_task_id": task_id,
            "step_index": index,
        }
        rows.append({
            "schema_version": 1,
            "event": "environment_action",
            "action_id": hashlib.sha256(
                json.dumps(
                    identity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            **identity,
            "manifest_index": 0,
            "action": action,
            "env_feedback": "visible",
            "reward": 0.0,
            "done": False,
        })
    journal = evidence_dir / "attempt_environment_actions/episode.jsonl"
    journal.parent.mkdir()
    journal.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    committed = {
        "api_calls": 1,
        "logical_api_calls": 1,
        "provider_retries": 0,
        "physical_provider_calls": 1,
        "cached_provider_calls": 0,
        "tokens": 5,
        "environment_actions": 3,
        "provider_queue_wait_ms": 0,
        "provider_service_latency_ms": 0,
        "logical_call_latency_ms": 0,
        "retry_backoff_ms": 0,
        "api_cost": None,
        "api_cost_unpriced": True,
    }

    costs = campaign_module._cost_accounting(
        seed=42,
        resume_sources=[],
        failed_attempt_sources=[failed],
        train_root=tmp_path / "current",
        test_root=tmp_path / "test",
        committed=committed,
    )

    actual = costs["actual_campaign_cost"]
    assert actual["environment_actions"] == 5
    assert actual["observed_environment_actions_lower_bound"] == 5
    assert actual["failed_attempt_observed_environment_actions"] == 2
    assert actual["unmeasured_parent_environment_actions"] is False
    measurements = costs["evidence"]["failed_attempt_action_measurement"]
    assert next(iter(measurements.values())) == {
        "source": "durable_environment_step_journals",
        "measurement_complete": True,
        "environment_actions": 2,
    }
