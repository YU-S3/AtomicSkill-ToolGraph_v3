from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

from experiments.baselines.common.freeze import freeze_files
from experiments.baselines.run_seed_campaign import (
    CampaignSpec,
    _cost_accounting,
    _formal_config_digest as campaign_formal_config_digest,
    _method_campaign_lease,
    _run_method_command,
    _select_phase_python,
    inspect_clean_source,
    run_campaign,
)
from experiments.baselines.run_method import (
    _formal_config_digest as run_method_formal_config_digest,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _value(command: Sequence[str], option: str) -> str:
    index = list(command).index(option)
    return str(command[index + 1])


def _source_state(_: Path) -> dict[str, Any]:
    return {
        "commit": "1" * 40,
        "dirty": False,
        "code_digest": "2" * 64,
    }


def _preflight(
    spec: CampaignSpec,
    campaign_root: Path,
    campaign_run_id: str,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": "protocol_faithful_matched_train_v2",
        "campaign_run_id": campaign_run_id,
        "campaign_root": str(campaign_root),
        "method": spec.method,
        "seeds": list(spec.seeds),
        "train_manifest_digest": "3" * 64,
        "validation_manifest_digest": "4" * 64,
        "test_manifest_digest": "5" * 64,
        "controller_commit": source_state["commit"],
        "controller_code_digest": source_state["code_digest"],
        "external_skillopt_commit": "6" * 40,
        "external_runtime_tree_digest": "7" * 64,
        "model": "deepseek-v4-flash",
        "model_identity": {
            "provider": "openai_compatible",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "MODEL_API_KEY",
            "reasoning_effort": "high",
        },
        "reasoning_effort": "high",
        "seed_lanes": 3,
        "campaign_provider_max_inflight": 16,
        "parallel": {
            "seed_lanes": 3,
            "episode_workers_per_seed": 16,
            "test_workers_per_seed": 16,
            "skillopt_analyst_workers_per_seed": 16,
            "campaign_provider_max_inflight": 16,
        },
        "retry_policy": {
            "sdk_max_retries": 0,
            "attempts": 5,
            "delays": [2, 5, 10, 20],
            "jitter_ratio": 0.10,
        },
        "provider_probe": {
            "enabled": True,
            "concurrency": 16,
            "requests": 32,
            "max_completion_tokens": 256,
            "reasoning_effort": "high",
        },
        "resume": {
            "enabled": True,
            "formal_boundary": "epoch",
            "reuse_verified_episode_cache": True,
        },
        "phase_python": str(spec.python),
        "worker_python": str(spec.python),
        "provider_probe_python": str(spec.python),
        "provider_gate_dir": str((campaign_root / "provider_gate").resolve()),
    }


class ScriptedCampaignRunner:
    def __init__(
        self,
        *,
        failed_train_seed: int | None = None,
        protocol_failed_train_seed: int | None = None,
        recoverable_train_seed: int | None = None,
        failed_probe_caps: set[int] | None = None,
    ) -> None:
        self.failed_train_seed = failed_train_seed
        self.protocol_failed_train_seed = protocol_failed_train_seed
        self.recoverable_train_seed = recoverable_train_seed
        self.failed_probe_caps = set(failed_probe_caps or set())
        self.train_barrier = threading.Barrier(3)
        self.release_slow_trains = threading.Event()
        self.test_42_overlapped_other_train = False
        self.commands: list[list[str]] = []
        self._commands_lock = threading.Lock()
        self._slow_train_finished: set[int] = set()
        self._train_attempts: dict[int, int] = {}

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
    ) -> int:
        del cwd
        command = list(command)
        with self._commands_lock:
            self.commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("scripted command\n", encoding="utf-8")

        module = command[2]
        if module.endswith("provider_load_probe"):
            cap = int(_value(command, "--max-inflight"))
            output = Path(_value(command, "--output-dir"))
            output.mkdir(parents=True, exist_ok=False)
            passed = cap not in self.failed_probe_caps
            calls_path = output / "provider_calls.jsonl"
            calls_path.write_text("{}\n", encoding="utf-8")
            _write_json(
                output / "provider_load_probe.json",
                {
                    "probe_kind": "campaign_provider_load",
                    "passed": passed,
                    "campaign_id": _value(command, "--campaign-id"),
                    "run_id": _value(command, "--run-id"),
                    "model": _value(command, "--model"),
                    "reasoning_effort": _value(command, "--reasoning-effort"),
                    "concurrency": cap,
                    "campaign_provider_max_inflight": cap,
                    "requests": 32,
                    "max_completion_tokens": 256,
                    "logical_calls_recorded": 32,
                    "completed_logical_calls": 32 if passed else 31,
                    "failed_provider_calls": 0 if passed else 1,
                    "exhausted_provider_calls": 0 if passed else 1,
                    "permanent_provider_errors": 0,
                    "provider_evidence_complete": True,
                    "observer_error_type": None,
                    "provider_calls_path": str(calls_path),
                    "provider_calls_sha256": hashlib.sha256(
                        calls_path.read_bytes()
                    ).hexdigest(),
                },
            )
            return 0 if passed else 1

        assert module.endswith("run_method")
        phase = _value(command, "--phase")
        seed = int(_value(command, "--seed"))
        output = Path(_value(command, "--output-dir"))

        if phase == "train":
            attempt = self._train_attempts.get(seed, 0) + 1
            self._train_attempts[seed] = attempt
            if attempt == 1:
                self.train_barrier.wait(timeout=5)
                if seed != 42:
                    assert self.release_slow_trains.wait(timeout=5)
                    self._slow_train_finished.add(seed)
            should_fail = (
                seed == self.failed_train_seed
                or seed == self.protocol_failed_train_seed
                or (seed == self.recoverable_train_seed and attempt == 1)
            )
            if should_fail:
                output.mkdir(parents=True, exist_ok=False)
                provider = output / "train" / "provider_calls.jsonl"
                provider.parent.mkdir(parents=True)
                provider.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "application_attempts": 5,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "provider_queue_wait_ms": 1,
                            "provider_service_latency_ms": 2,
                            "logical_call_latency_ms": 30,
                            "retry_backoff_ms": 27,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _write_json(
                    output / "failure.json",
                    {
                        "passed": False,
                        "failure_kind": (
                            "protocol_failure"
                            if seed == self.protocol_failed_train_seed
                            else "infrastructure_failure"
                        ),
                        "error_type": "ProviderCallExhausted",
                        "error": "scripted provider outage",
                    },
                )
                return 1
            self._write_train(output, seed)
            return 0

        assert phase == "test"
        source_run = Path(_value(command, "--source-run"))
        assert source_run.parent == output.parent
        assert source_run.name == "train" or source_run.name.startswith(
            "train_attempt_"
        )
        if seed == 42:
            self.test_42_overlapped_other_train = not self._slow_train_finished
            self.release_slow_trains.set()
        self._write_test(output, source_run, seed)
        return 0

    @staticmethod
    def _write_train(output: Path, seed: int) -> None:
        output.mkdir(parents=True, exist_ok=False)
        trained = output / "train" / "best_skill.md"
        trained.parent.mkdir(parents=True)
        trained.write_text(f"seed {seed} skill\n", encoding="utf-8")
        frozen = freeze_files(
            method_id="b3_skillopt",
            source_files={"best_skill.md": trained},
            destination=output / "frozen",
            source_train_manifest_hash="3" * 64,
            source_validation_manifest_hash="4" * 64,
            metadata={"run_seed": seed},
        )
        _write_json(
            output / "run_manifest.json",
            {"run_seed": seed, "phase": "train"},
        )
        _write_json(
            output / "report.json",
            {
                "passed": True,
                "method": "b3_skillopt",
                "frozen": {"digest": frozen.digest},
                "training_cost": {
                    "usage": {
                        "target": {
                            "calls": seed,
                            "prompt_tokens": seed * 10,
                            "completion_tokens": seed * 2,
                        },
                        "evolution": {
                            "calls": 2,
                            "prompt_tokens": 20,
                            "completion_tokens": 4,
                        },
                    },
                    "train_episodes": {"environment_actions": seed * 3},
                    "validation_episodes": {"environment_actions": seed},
                    "api_cost": None,
                    "provider_evidence": {
                        "application_attempts": seed + 2,
                        "physical_provider_calls": seed + 2,
                        "cached_provider_calls": 0,
                        "provider_queue_wait_ms": seed,
                        "provider_service_latency_ms": seed * 2,
                        "logical_call_latency_ms": seed * 3,
                        "retry_backoff_ms": 0,
                    },
                },
            },
        )
        _write_json(
            output / "completion.json",
            {
                "passed": True,
                "phase": "train",
                "run_id": f"train_{seed}",
            },
        )

    @staticmethod
    def _write_test(output: Path, source_run: Path, seed: int) -> None:
        output.mkdir(parents=True, exist_ok=False)
        frozen_digest = json.loads(
            (source_run / "frozen" / "digest.json").read_text(encoding="utf-8")
        )["digest"]
        _write_json(
            output / "run_manifest.json",
            {"run_seed": seed, "phase": "test"},
        )
        _write_json(
            output / "test_report.json",
            {
                "passed": True,
                "protocol": {"source_run": str(source_run.resolve())},
                "frozen": {"digest": frozen_digest},
                "comparison_metrics": {
                    "official_rate": 0.80 + (seed - 42) * 0.05,
                    "strict_rate": 0.70 + (seed - 42) * 0.05,
                    "family": {"ignored_nested_metric": 1.0},
                },
                "test_cost": {
                    "usage": {
                        "target": {
                            "calls": 10,
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                        },
                        "evolution": {
                            "calls": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        },
                    },
                    "episodes": {"environment_actions": seed * 2},
                    "api_cost": None,
                    "provider_evidence": {
                        "application_attempts": 10,
                        "physical_provider_calls": 10,
                        "cached_provider_calls": 0,
                        "provider_queue_wait_ms": 1,
                        "provider_service_latency_ms": 2,
                        "logical_call_latency_ms": 3,
                        "retry_backoff_ms": 0,
                    },
                },
            },
        )
        _write_json(
            output / "completion.json",
            {
                "passed": True,
                "phase": "test",
                "run_id": f"test_{seed}",
            },
        )


def _spec(tmp_path: Path) -> CampaignSpec:
    repo = tmp_path / "repo"
    repo.mkdir()
    common = repo / "configs" / "baselines" / "common.yaml"
    common.parent.mkdir(parents=True)
    common.write_text("{}\n", encoding="utf-8")
    for name in ("train.json", "validation.json", "test.json", "config.yaml"):
        (repo / name).write_text("{}\n", encoding="utf-8")
    return CampaignSpec(
        method="b3_skillopt",
        seeds=(42, 43, 44),
        train_manifest=repo / "train.json",
        validation_manifest=repo / "validation.json",
        test_manifest=repo / "test.json",
        config=repo / "config.yaml",
        output_dir=repo / "campaign",
        python=Path("/formal/python"),
        repo_root=repo,
    )


def test_three_seed_lanes_pipeline_test_without_waiting_for_all_train(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    runner = ScriptedCampaignRunner()

    report = run_campaign(
        spec,
        command_runner=runner,
        source_inspector=_source_state,
        preflight_builder=_preflight,
    )

    assert report["passed"] is True
    assert report["completed_seeds"] == [42, 43, 44]
    assert report["failed_seeds"] == []
    assert report["all_frozen_digests_verified"] is True
    assert report["campaign_makespan_ms"] >= 0
    assert runner.test_42_overlapped_other_train is True
    assert report["test_metrics_mean_std"]["official_rate"] == {
        "mean": pytest.approx(0.85),
        "std": pytest.approx(0.05),
        "std_ddof": 1,
        "values_by_seed": {
            "42": pytest.approx(0.80),
            "43": pytest.approx(0.85),
            "44": pytest.approx(0.90),
        },
    }
    assert report["resource_usage"]["totals"] == {
        "api_calls": (42 + 12) + (43 + 12) + (44 + 12),
        "logical_api_calls": (42 + 12) + (43 + 12) + (44 + 12),
        "provider_retries": 0,
        "physical_provider_calls": (42 + 12) + (43 + 12) + (44 + 12),
        "cached_provider_calls": 0.0,
        "tokens": (42 * 12 + 24 + 120)
        + (43 * 12 + 24 + 120)
        + (44 * 12 + 24 + 120),
        "environment_actions": 42 * 6 + 43 * 6 + 44 * 6,
        "provider_queue_wait_ms": 132.0,
        "provider_service_latency_ms": 264.0,
        "logical_call_latency_ms": 396.0,
        "retry_backoff_ms": 0.0,
        "api_cost": None,
        "api_cost_unpriced": True,
    }
    assert report["resource_usage"]["mean_std"]["provider_queue_wait_ms"] == {
        "mean": pytest.approx(44.0),
        "std": pytest.approx(1.0),
        "std_ddof": 1,
        "values_by_seed": {
            "42": pytest.approx(43.0),
            "43": pytest.approx(44.0),
            "44": pytest.approx(45.0),
        },
    }
    assert set(report["cost_accounting"]) == {
        "schema_version",
        "committed_method_cost",
        "infrastructure_retry_overhead",
        "resume_replay_overhead",
        "actual_campaign_cost",
    }
    assert report["cost_accounting"]["committed_method_cost"]["totals"][
        "provider_service_latency_ms"
    ] == 264
    assert report["cost_accounting"]["infrastructure_retry_overhead"][
        "measurement_complete"
    ] is True
    assert report["cost_accounting"]["resume_replay_overhead"]["totals"][
        "api_calls"
    ] == 0

    expected_states = [
        "PENDING",
        "PREFLIGHT",
        "TRAINING",
        "TRAIN_COMPLETED",
        "FROZEN",
        "FROZEN_VERIFIED",
        "TESTING",
        "COMPLETED",
    ]
    for seed in (42, 43, 44):
        lane_root = spec.output_dir / f"seed_{seed}"
        state = json.loads(
            (lane_root / "lane_state.json").read_text(encoding="utf-8")
        )
        assert [row["state"] for row in state["transitions"]] == expected_states
        assert (lane_root / "train").is_dir()
        assert (lane_root / "test").is_dir()

    lock = json.loads(
        (spec.output_dir / "campaign_lock.json").read_text(encoding="utf-8")
    )
    assert lock["seeds"] == [42, 43, 44]
    assert lock["campaign_provider_max_inflight"] == 16
    assert lock["retry_policy"]["delays"] == [2, 5, 10, 20]
    assert lock["provider_probe_receipt"]["report"]["passed"] is True


def test_one_infrastructure_lane_failure_does_not_cancel_other_lanes(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    runner = ScriptedCampaignRunner(failed_train_seed=43)

    report = run_campaign(
        spec,
        command_runner=runner,
        source_inspector=_source_state,
        preflight_builder=_preflight,
    )

    assert report["passed"] is False
    assert report["completed_seeds"] == [42, 44]
    assert report["failed_seeds"] == [43]
    failed = next(lane for lane in report["lanes"] if lane["seed"] == 43)
    assert failed["failure_kind"] == "infrastructure_failure"
    assert failed["error_type"] == "ProviderCallExhausted"
    assert not (spec.output_dir / "seed_43" / "test").exists()
    assert [row["attempt"] for row in failed["train_attempts"]] == [1, 2]
    assert failed["train_attempts"][1]["resume_source_run"] == str(
        spec.output_dir / "seed_43" / "train"
    )
    assert (spec.output_dir / "seed_42" / "test" / "completion.json").is_file()
    assert (spec.output_dir / "seed_44" / "test" / "completion.json").is_file()
    assert (spec.output_dir / "campaign_failure.json").is_file()
    assert report["test_metrics_mean_std"] == {}


def test_infrastructure_train_failure_resumes_inside_same_lane_then_tests(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    runner = ScriptedCampaignRunner(recoverable_train_seed=43)

    report = run_campaign(
        spec,
        command_runner=runner,
        source_inspector=_source_state,
        preflight_builder=_preflight,
    )

    assert report["passed"] is True
    lane = next(item for item in report["lanes"] if item["seed"] == 43)
    first = spec.output_dir / "seed_43" / "train"
    resumed = spec.output_dir / "seed_43" / "train_attempt_002"
    assert Path(lane["train_root"]) == resumed
    assert [row["status"] for row in lane["train_attempts"]] == [
        "failed",
        "completed",
    ]
    train_commands = [
        command
        for command in runner.commands
        if "--phase" in command
        and _value(command, "--phase") == "train"
        and int(_value(command, "--seed")) == 43
    ]
    assert len(train_commands) == 2
    assert Path(_value(train_commands[0], "--output-dir")) == first
    assert "--resume-source-run" not in train_commands[0]
    assert Path(_value(train_commands[1], "--output-dir")) == resumed
    assert Path(_value(train_commands[1], "--resume-source-run")) == first
    test_command = next(
        command
        for command in runner.commands
        if "--phase" in command
        and _value(command, "--phase") == "test"
        and int(_value(command, "--seed")) == 43
    )
    assert Path(_value(test_command, "--source-run")) == resumed
    assert lane["cost_accounting"]["actual_campaign_cost"][
        "includes_resume_parent"
    ] is True


def test_protocol_train_failure_is_not_resumed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runner = ScriptedCampaignRunner(protocol_failed_train_seed=43)

    report = run_campaign(
        spec,
        command_runner=runner,
        source_inspector=_source_state,
        preflight_builder=_preflight,
    )

    assert report["passed"] is False
    lane = next(item for item in report["lanes"] if item["seed"] == 43)
    assert lane["failure_kind"] == "protocol_failure"
    assert len(lane["train_attempts"]) == 1
    commands = [
        command
        for command in runner.commands
        if "--phase" in command
        and _value(command, "--phase") == "train"
        and int(_value(command, "--seed")) == 43
    ]
    assert len(commands) == 1


def test_transient_probe_failure_selects_lower_cap_before_lock(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    runner = ScriptedCampaignRunner(failed_probe_caps={16})

    report = run_campaign(
        spec,
        command_runner=runner,
        source_inspector=_source_state,
        preflight_builder=_preflight,
    )

    assert report["passed"] is True
    probes = [
        command
        for command in runner.commands
        if command[2].endswith("provider_load_probe")
    ]
    assert [int(_value(command, "--max-inflight")) for command in probes] == [
        16,
        12,
    ]
    assert Path(_value(probes[0], "--gate-dir")) != Path(
        _value(probes[1], "--gate-dir")
    )

    lock = json.loads(
        (spec.output_dir / "campaign_lock.json").read_text(encoding="utf-8")
    )
    assert lock["campaign_provider_max_inflight"] == 12
    assert lock["parallel"]["campaign_provider_max_inflight"] == 12
    assert lock["provider_probe"]["concurrency"] == 12
    assert lock["provider_probe"]["selected_cap"] == 12
    assert [row["cap"] for row in lock["provider_probe_attempts"]] == [16, 12]
    runtime_config = Path(lock["config_path"])
    assert runtime_config.parent == spec.output_dir
    resolved = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert resolved["parallel"]["campaign_provider_max_inflight"] == 12
    assert resolved["provider_probe"]["concurrency"] == 12
    phase_commands = [command for command in runner.commands if "--phase" in command]
    assert phase_commands
    assert {
        Path(_value(command, "--config")) for command in phase_commands
    } == {runtime_config}


def test_repository_campaign_lease_rejects_overlap(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with _method_campaign_lease(repo, owner="first"):
        with pytest.raises(RuntimeError, match="another formal method campaign"):
            with _method_campaign_lease(repo, owner="second"):
                pytest.fail("overlapping campaign lease was accepted")


def test_resume_cost_accounting_uses_parent_provider_sidecar_without_guessing(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "current"
    test_root = tmp_path / "test"
    parent = tmp_path / "parent"

    def write_event(path: Path, *, attempts: int, prompt: int, completion: int,
                    queue: int, service: int, logical: int, backoff: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "succeeded",
                    "application_attempts": attempts,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "provider_queue_wait_ms": queue,
                    "provider_service_latency_ms": service,
                    "logical_call_latency_ms": logical,
                    "retry_backoff_ms": backoff,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    write_event(
        train_root / "train" / "provider_calls.jsonl",
        attempts=2,
        prompt=10,
        completion=2,
        queue=3,
        service=4,
        logical=7,
        backoff=1,
    )
    write_event(
        parent / "train" / "provider_calls.jsonl",
        attempts=3,
        prompt=5,
        completion=1,
        queue=2,
        service=3,
        logical=9,
        backoff=4,
    )
    committed = {
        "api_calls": 2,
        "logical_api_calls": 1,
        "provider_retries": 1,
        "physical_provider_calls": 1,
        "cached_provider_calls": 0,
        "tokens": 12,
        "environment_actions": 10,
        "provider_queue_wait_ms": 3,
        "provider_service_latency_ms": 4,
        "logical_call_latency_ms": 7,
        "retry_backoff_ms": 1,
        "api_cost": None,
        "api_cost_unpriced": True,
    }

    costs = _cost_accounting(
        seed=42,
        resume_sources=[parent],
        train_root=train_root,
        test_root=test_root,
        committed=committed,
    )

    assert costs["infrastructure_retry_overhead"]["api_calls"] == 3
    assert costs["infrastructure_retry_overhead"]["retry_backoff_ms"] == 5
    assert costs["resume_replay_overhead"]["measurement_complete"] is False
    assert costs["resume_replay_overhead"]["api_calls"] is None
    assert costs["actual_campaign_cost"]["api_calls"] == 5
    assert costs["actual_campaign_cost"]["logical_api_calls"] == 2
    assert costs["actual_campaign_cost"]["tokens"] == 18
    assert costs["actual_campaign_cost"]["environment_actions"] is None
    assert costs["committed_method_cost"]["measurement_complete"] is False
    assert "continuation+Test" in costs["committed_method_cost"]["semantics"]


def test_test_commands_are_bound_to_same_seed_train_root(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runner = ScriptedCampaignRunner()
    run_campaign(
        spec,
        command_runner=runner,
        source_inspector=_source_state,
        preflight_builder=_preflight,
    )

    tests = [command for command in runner.commands if "--phase" in command and _value(command, "--phase") == "test"]
    assert len(tests) == 3
    for command in tests:
        seed = int(_value(command, "--seed"))
        source = Path(_value(command, "--source-run"))
        output = Path(_value(command, "--output-dir"))
        assert source == spec.output_dir / f"seed_{seed}" / "train"
        assert output == spec.output_dir / f"seed_{seed}" / "test"
        assert _value(command, "--campaign-lock") == str(
            spec.output_dir / "campaign_lock.json"
        )
        assert command[0] == str(spec.python)


def test_run_method_command_uses_frozen_lexical_phase_python(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    command = _run_method_command(
        spec,
        phase="train",
        seed=42,
        output_dir=spec.output_dir / "seed_42" / "train",
        campaign_lock=spec.output_dir / "campaign_lock.json",
    )
    assert command[0] == str(spec.python)


def test_explicit_python_override_must_match_configured_worker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    configured = repo / ".venv_b3_skillopt" / "bin" / "python"
    supplied = repo / "system" / "python"
    for executable in (configured, supplied):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    assert _select_phase_python(
        repo, ".venv_b3_skillopt/bin/python", None,
    ) == configured
    with pytest.raises(ValueError, match="exactly match formal worker_python"):
        _select_phase_python(repo, configured, supplied)


def test_formal_seed_identity_is_fixed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="exactly 42, 43, 44"):
        CampaignSpec(
            method=spec.method,
            seeds=(42, 43),
            train_manifest=spec.train_manifest,
            validation_manifest=spec.validation_manifest,
            test_manifest=spec.test_manifest,
            config=spec.config,
            output_dir=spec.output_dir,
            python=spec.python,
            repo_root=spec.repo_root,
        )


def test_campaign_and_per_phase_formal_config_identity_are_identical() -> None:
    config = {
        "campaign_id": "formal",
        "run_seed": 42,
        "train": {"seed": 42, "num_epochs": 4},
        "provider_transport": {"application_retry_limit": 5},
        "protocol": {"phase_specific": True},
    }
    assert campaign_formal_config_digest(config) == run_method_formal_config_digest(
        config
    )
    other_seed = json.loads(json.dumps(config))
    other_seed["run_seed"] = 44
    other_seed["train"]["seed"] = 44
    other_seed["protocol"] = {"different_phase": "test"}
    assert campaign_formal_config_digest(config) == campaign_formal_config_digest(
        other_seed
    )


def test_clean_source_preflight_rejects_tracked_and_untracked_source_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "controller"
    (repo / "src").mkdir(parents=True)
    (repo / "experiments").mkdir()
    (repo / "configs").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Baseline Tests"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    clean = inspect_clean_source(repo)
    assert clean["dirty"] is False
    assert len(clean["commit"]) == 40
    assert len(clean["code_digest"]) == 64

    (repo / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires clean controller source"):
        inspect_clean_source(repo)

    subprocess.run(
        ["git", "checkout", "--", "src/module.py"], cwd=repo, check=True
    )
    (repo / "experiments" / "untracked.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires clean controller source"):
        inspect_clean_source(repo)
