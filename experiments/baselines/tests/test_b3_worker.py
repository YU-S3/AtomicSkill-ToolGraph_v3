"""Offline contract tests for the isolated B3 SkillOpt worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.baselines.b3_skillopt import worker
from experiments.baselines.common.manifest import sha256_json
from experiments.baselines.common.model_config import FROZEN_BASELINE_MODEL, ModelConfig
from experiments.baselines.common.subprocess_worker import WorkerWire


def _resolved_config() -> dict:
    return {
        "schema_version": 1,
        "method": "b3_skillopt",
        "protocol_profile": "formal_v2",
        "campaign_id": "fixture",
        "run_seed": 42,
        "max_environment_actions": 100,
        "worker_python": ".venv_b3_skillopt/bin/python",
        "model": dict(FROZEN_BASELINE_MODEL),
        "provider_transport": {
            "sdk_max_retries": 0,
            "application_retry_limit": 5,
            "retry_delays_seconds": [2, 5, 10, 20],
        },
        "train": {
            "num_epochs": 4,
            "train_size": 120,
            "batch_size": 40,
            "accumulation": 1,
            "seed": 42,
        },
        "gradient": {
            "minibatch_size": 8,
            "merge_batch_size": 8,
            "analyst_workers": 16,
            "failure_only": False,
        },
        "optimizer": {
            "learning_rate": 4,
            "min_learning_rate": 2,
            "lr_scheduler": "cosine",
            "lr_control_mode": "fixed",
            "skill_update_mode": "patch",
            "use_slow_update": True,
            "slow_update_samples": 20,
            "longitudinal_pair_policy": "mixed",
            "use_meta_skill": True,
            "use_skill_aware_reflection": False,
            "slow_update_gate_with_selection": True,
        },
        "evaluation": {
            "use_gate": True,
            "gate_metric": "hard",
            "sel_env_num": 24,
            "eval_test": False,
        },
        "smoke": {"task_count": 2, "max_steps": 2},
        "env": {
            "name": "alfworld",
            "max_steps": 100,
            "max_completion_tokens": 16384,
            "workers": 16,
            "max_api_workers": 16,
        },
    }


def _wire(tmp_path: Path, *, phase: str = "smoke") -> WorkerWire:
    output = tmp_path / "run"
    phase_dir = output / phase
    phase_dir.mkdir(parents=True)
    config = _resolved_config()
    config_path = output / "config_resolved.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    identity = {
        "config_digest": sha256_json(config),
        "train_manifest_digest": "a" * 64,
        "validation_manifest_digest": "b" * 64,
    }
    if phase == "train_eval":
        identity.update({
            "evaluation_manifest_digest": "a" * 64,
            "frozen_artifact_digest": "c" * 64,
        })
    return WorkerWire(
        method="b3_skillopt",
        phase=phase,
        manifest_path=str(tmp_path / "train.json"),
        validation_manifest_path=str(tmp_path / "validation.json"),
        test_manifest_path=None,
        config_path=str(config_path),
        output_dir=str(output),
        run_seed=42,
        model=dict(FROZEN_BASELINE_MODEL),
        run_id="fixture_run",
        result_path=str(phase_dir / "worker_result.json"),
        identity=identity,
        external_skillopt_root=str(tmp_path / "missing-skillopt"),
    )


class _Observer:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    def events(self) -> list[dict]:
        return [dict(event) for event in self._events]


def _event(wire: WorkerWire, **updates) -> dict:
    event = {
        "schema_version": 1,
        "event": "provider_call",
        "method": wire.method,
        "phase": wire.phase,
        "run_id": wire.run_id,
        "model": wire.model["model"],
        "reasoning_effort": wire.model["reasoning_effort"],
        "call_id": "provider_1",
        "role": "target",
        "stage": "rollout",
        "status": "succeeded",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": 2,
        "reasoning_tokens_status": "reported",
        "total_tokens": 15,
        "requested_retry_limit": 5,
        "retry_limit": 5,
        "application_attempts": 1,
        "sdk_boundary_attempts": 1,
        "retry_backoff_ms": 0,
        "failure_code_counts": {},
        "last_failure_code": None,
        "recovered": False,
    }
    event.update(updates)
    return event


def test_config_identity_uses_canonical_merged_mapping(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    config = _resolved_config()
    model = ModelConfig.from_mapping(dict(FROZEN_BASELINE_MODEL))
    assert worker._validate_config_identity(wire, config, model) == sha256_json(config)
    config["optimizer"]["learning_rate"] = 5
    with pytest.raises(ValueError, match="digest"):
        worker._validate_config_identity(wire, config, model)


def test_provider_usage_preserves_roles_stages_and_reasoning(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    events = [
        _event(wire),
        _event(
            wire,
            call_id="provider_2",
            role="optimizer",
            stage="analyst",
            prompt_tokens=20,
            completion_tokens=7,
            reasoning_tokens=None,
            reasoning_tokens_status="unavailable",
            total_tokens=27,
        ),
    ]
    usage = worker._provider_usage(
        _Observer(events), wire=wire, wall_time_ms=123
    )
    assert usage.target.to_dict() == {
        "calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": 2,
    }
    assert usage.evolution.calls == 1
    assert usage.per_stage["analyst"].completion_tokens == 7
    assert usage.wall_time_ms == 123

    events[0]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="wrong run identity"):
        worker._provider_usage(
            _Observer(events), wire=wire, wall_time_ms=123
        )


def test_provider_evidence_summary_counts_recovered_retry_attempts(
    tmp_path: Path,
) -> None:
    wire = _wire(tmp_path)
    events = [
        _event(
            wire,
            application_attempts=3,
            sdk_boundary_attempts=3,
            failure_code_counts={"timeout": 2},
            last_failure_code="timeout",
            recovered=True,
        ),
        _event(
            wire,
            call_id="provider_2",
            role="optimizer",
            stage="analyst",
        ),
    ]

    summary = worker._provider_evidence_summary(_Observer(events))

    assert summary["calls"] == 2
    assert summary["application_attempts"] == 4
    assert summary["provider_retries"] == 2
    assert summary["sdk_boundary_attempts"] == 4
    assert summary["recovered_calls"] == 1
    assert summary["retry_failure_code_counts"] == {"timeout": 2}


def test_provider_failure_is_infrastructure_not_hard_zero(tmp_path: Path) -> None:
    wire = _wire(tmp_path)
    failed = _event(
        wire,
        status="failed",
        error_type="TimeoutError",
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=None,
        reasoning_tokens_status="unavailable",
        total_tokens=0,
        application_attempts=5,
        sdk_boundary_attempts=5,
        failure_code_counts={"timeout": 5},
        last_failure_code="timeout",
    )
    with pytest.raises(worker.WorkerPhaseError, match="infrastructure"):
        worker._provider_usage(
            _Observer([failed]), wire=wire, wall_time_ms=1
        )


def test_worker_failure_result_is_phase_bound_and_never_overwritten(
    tmp_path: Path,
) -> None:
    wire = _wire(tmp_path)
    wire_path = Path(wire.output_dir) / wire.phase / "worker_wire.json"
    wire_path.write_text(json.dumps(wire.to_dict()), encoding="utf-8")
    assert worker.main(["--wire", str(wire_path)]) == 1
    result_path = Path(wire.result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["phase"] == wire.phase
    assert result["run_id"] == wire.run_id
    assert result["identity"] == wire.identity
    first_bytes = result_path.read_bytes()
    assert worker.main(["--wire", str(wire_path)]) == 1
    assert result_path.read_bytes() == first_bytes


def test_method_metrics_export_patch_gate_and_skill_growth(
    tmp_path: Path,
) -> None:
    history = [
        {
            "step": 1,
            "epoch": 1,
            "n_patches": 3,
            "n_failure_patches": 2,
            "n_success_patches": 1,
            "candidate_hash": "a",
            "action": "accept_new_best",
        },
        {
            "step": 2,
            "epoch": 2,
            "n_patches": 0,
            "n_failure_patches": 0,
            "n_success_patches": 0,
            "action": "skip_no_patches",
        },
    ]
    (tmp_path / "history.json").write_text(json.dumps(history), encoding="utf-8")
    metrics = worker._method_metrics(
        train_out=tmp_path,
        upstream_summary={
            "version": "skillopt-0.1.0",
            "total_steps": 2,
            "best_step": 1,
            "best_selection_hard": 0.5,
            "epoch_stats": [],
        },
        initial_skill="short skill",
        best_skill="short skill with added rule",
        model_name="deepseek-v4-flash",
        selection_episodes=90,
        count_skill_tokens=lambda text, _model: max(1, len(text) // 4),
    )
    assert metrics["patch_proposals"] == 3
    assert metrics["accepted_candidates"] == 1
    assert metrics["skipped_steps"] == 1
    assert metrics["gate_acceptance_rate"] == 1.0
    assert metrics["best_epoch"] == 1
    assert metrics["selection_episodes"] == 90
    assert metrics["skill_char_growth"] > 0


def test_safe_error_does_not_replace_every_empty_boundary(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    message = worker._safe_error(RuntimeError("plain failure"), None)
    assert message == "plain failure"
