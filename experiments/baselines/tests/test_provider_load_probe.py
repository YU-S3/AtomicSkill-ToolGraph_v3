"""Offline tests for the campaign provider load probe and diagnostics."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl")
pytest.importorskip("skillopt")

import skillopt.model.openai_compatible_backend as backend

from experiments.baselines.common.provider_gate import CampaignProviderGate
from experiments.baselines.common.provider_load_probe import run_provider_load_probe


def _completion(*, request_id: str = "req_fixture") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="OK", tool_calls=[]),
        )],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=1,
            total_tokens=4,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        _request_id=request_id,
    )


def _install_fake_client(monkeypatch, completions) -> None:
    class Client:
        max_retries = 0
        chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr(backend, "_get_client", lambda role: Client())
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    monkeypatch.setattr(
        backend.TARGET_CONFIG, "deployment", "deepseek-v4-flash"
    )
    monkeypatch.setattr(
        backend.OPTIMIZER_CONFIG, "deployment", "deepseek-v4-flash"
    )


def test_exact_32_by_16_probe_writes_complete_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lock = threading.Lock()
    current = 0
    peak = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal current, peak
            assert kwargs["reasoning_effort"] == "high"
            assert kwargs["max_tokens"] == 256
            with lock:
                current += 1
                peak = max(peak, current)
            try:
                time.sleep(0.01)
                return _completion(request_id=f"req_{threading.get_ident()}")
            finally:
                with lock:
                    current -= 1

    _install_fake_client(monkeypatch, Completions())
    output = tmp_path / "probe"
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="probe_fixture",
        max_inflight=16,
    )
    report = run_provider_load_probe(
        output_dir=output,
        campaign_gate=gate,
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="load_probe_fixture",
        run_seed=42,
    )

    assert report["passed"] is True
    assert report["requests"] == 32
    assert report["completed_logical_calls"] == 32
    assert report["exhausted_provider_calls"] == 0
    assert report["permanent_provider_errors"] == 0
    assert report["provider_evidence_complete"] is True
    assert report["logical_calls_recorded"] == 32
    assert report["application_attempts"] == 32
    assert report["sdk_boundary_attempts"] == 32
    assert 1 <= peak <= 16

    persisted_report = json.loads(
        (output / "provider_load_probe.json").read_text(encoding="utf-8")
    )
    assert persisted_report == report
    events = [
        json.loads(line)
        for line in (output / "provider_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(events) == 32
    assert len({event["logical_call_id"] for event in events}) == 32
    assert {event["run_seed"] for event in events} == {42}
    assert {event["stage"] for event in events} == {"provider_load_probe"}
    assert all(event["provider_service_latency_ms"] >= 0 for event in events)
    assert all(event["logical_call_latency_ms"] >= 0 for event in events)


def test_d1_http_429_is_classified_and_recovery_is_accepted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class RateLimitError(RuntimeError):
        status_code = 429
        request_id = "req_rate_limited"

    outcomes = [RateLimitError("body must not be persisted"), _completion()]
    lock = threading.Lock()

    class Completions:
        def create(self, **kwargs):
            with lock:
                outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    _install_fake_client(monkeypatch, Completions())
    # Pinned SkillOpt sleeps once even when invoked with retries=1. This test is
    # about classification/evidence, so eliminate that fixture-only delay.
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    output = tmp_path / "probe"
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="rate_limit_fixture",
        max_inflight=1,
    )
    report = run_provider_load_probe(
        output_dir=output,
        campaign_gate=gate,
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="rate_limit_probe",
        run_seed=43,
        concurrency=1,
        requests=1,
        application_retry_limit=2,
        retry_delays_seconds=[0],
        deterministic_jitter_ratio=0,
    )

    assert report["passed"] is True
    assert report["completed_logical_calls"] == 1
    assert report["recovered_provider_calls"] == 1
    assert report["provider_failure_code_counts"] == {"rate_limit": 1}
    assert report["application_attempts"] == 2
    assert report["sdk_boundary_attempts"] == 2
    event = json.loads(
        (output / "provider_calls.jsonl").read_text(encoding="utf-8")
    )
    assert event["status"] == "succeeded"
    assert event["last_failure_code"] == "rate_limit"
    assert event["failure_code_counts"] == {"rate_limit": 1}
    assert event["application_attempts"] == 2
    assert event["recovered"] is True
    assert event["provider_request_ids"] == ["req_rate_limited", "req_fixture"]
    assert "body must not be persisted" not in json.dumps(event)


def test_permanent_provider_error_fails_probe_without_retrying(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class AuthenticationError(RuntimeError):
        status_code = 401
        request_id = "req_auth"

    calls = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise AuthenticationError("credential text must not be persisted")

    _install_fake_client(monkeypatch, Completions())
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    output = tmp_path / "probe"
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="auth_fixture",
        max_inflight=1,
    )
    report = run_provider_load_probe(
        output_dir=output,
        campaign_gate=gate,
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="auth_probe",
        run_seed=44,
        concurrency=1,
        requests=1,
        application_retry_limit=5,
        retry_delays_seconds=[0, 0, 0, 0],
        deterministic_jitter_ratio=0,
    )

    assert report["passed"] is False
    assert calls == 1
    assert report["completed_logical_calls"] == 0
    assert report["failed_provider_calls"] == 1
    assert report["exhausted_provider_calls"] == 0
    assert report["permanent_provider_errors"] == 1
    assert report["provider_evidence_complete"] is True
    assert report["provider_failure_code_counts"] == {"authentication": 1}
    event = json.loads(
        (output / "provider_calls.jsonl").read_text(encoding="utf-8")
    )
    assert event["last_failure_code"] == "authentication"
    assert "credential text" not in json.dumps(event)


def test_transient_exhaustion_is_complete_negative_probe_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class RateLimitError(RuntimeError):
        status_code = 429
        request_id = "req_rate_limit"

    calls = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RateLimitError("response body must not be persisted")

    _install_fake_client(monkeypatch, Completions())
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    output = tmp_path / "probe"
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="exhaustion_fixture",
        max_inflight=1,
    )
    report = run_provider_load_probe(
        output_dir=output,
        campaign_gate=gate,
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="exhaustion_probe",
        run_seed=42,
        concurrency=1,
        requests=1,
        application_retry_limit=5,
        retry_delays_seconds=[0, 0, 0, 0],
        deterministic_jitter_ratio=0,
    )

    assert report["passed"] is False
    assert calls == 5
    assert report["provider_evidence_complete"] is True
    assert report["logical_calls_recorded"] == 1
    assert report["completed_logical_calls"] == 0
    assert report["failed_provider_calls"] == 1
    assert report["exhausted_provider_calls"] == 1
    assert report["permanent_provider_errors"] == 0
    event = json.loads(
        (output / "provider_calls.jsonl").read_text(encoding="utf-8")
    )
    assert event["status"] == "failed"
    assert event["application_attempts"] == 5
    assert event["sdk_boundary_attempts"] == 5
    assert event["failure_code_counts"] == {"rate_limit": 5}
    assert event["last_failure_code"] == "rate_limit"
    assert event["recovered"] is False
    assert event["prompt_tokens"] == 0
    assert event["completion_tokens"] == 0
    assert event["total_tokens"] == 0
    assert "response body" not in json.dumps(event)
