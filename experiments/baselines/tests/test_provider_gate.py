"""Acceptance tests for the campaign-global provider gate (G1--G4)."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl")

from experiments.baselines.common.provider_gate import CampaignProviderGate


def _cross_process_gate_worker(
    gate_dir: str,
    start,
    current,
    peak,
    counter_lock,
    errors,
) -> None:
    try:
        gate = CampaignProviderGate(
            gate_dir=Path(gate_dir),
            campaign_id="three_seed_fixture",
            max_inflight=16,
        )

        def request(index: int) -> None:
            start.wait()
            with gate.acquire(
                run_id=f"seed_lane_{multiprocessing.current_process().pid}",
                seed=42,
                role="target",
                stage="rollout",
                logical_call_id=f"call_{index}",
            ):
                with counter_lock:
                    current.value += 1
                    peak.value = max(peak.value, current.value)
                try:
                    time.sleep(0.04)
                finally:
                    with counter_lock:
                        current.value -= 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(request, index) for index in range(16)]
            for future in futures:
                future.result(timeout=15)
    except BaseException as exc:  # child must report before exiting
        errors.put(type(exc).__name__)
        raise


def test_g1_three_processes_share_one_global_cap(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    current = context.Value("i", 0)
    peak = context.Value("i", 0)
    counter_lock = context.Lock()
    errors = context.Queue()
    gate_dir = tmp_path / "provider_gate"

    processes = [
        context.Process(
            target=_cross_process_gate_worker,
            args=(str(gate_dir), start, current, peak, counter_lock, errors),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("cross-process provider gate test timed out")

    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert errors.empty()
    assert current.value == 0
    assert 1 <= peak.value <= 16


def test_g2_exception_releases_slot_immediately(tmp_path: Path) -> None:
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="release_fixture",
        max_inflight=1,
    )
    with pytest.raises(RuntimeError, match="fixture"):
        with gate.acquire(
            run_id="run_a",
            seed=42,
            role="target",
            stage="rollout",
            logical_call_id="first",
        ):
            raise RuntimeError("fixture")

    started = time.perf_counter()
    with gate.acquire(
        run_id="run_b",
        seed=43,
        role="optimizer",
        stage="reflection",
        logical_call_id="second",
    ) as lease:
        assert lease.slot_id == 0
    assert time.perf_counter() - started < 0.5


def test_g3_backoff_outside_gate_allows_another_request(tmp_path: Path) -> None:
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="backoff_fixture",
        max_inflight=1,
    )
    first_released = threading.Event()
    second_acquired = threading.Event()

    def retrying_request() -> None:
        with gate.acquire(
            run_id="run_42",
            seed=42,
            role="target",
            stage="rollout",
            logical_call_id="retrying",
        ):
            pass
        first_released.set()
        assert second_acquired.wait(timeout=2), "slot remained held during backoff"
        with gate.acquire(
            run_id="run_42",
            seed=42,
            role="target",
            stage="rollout",
            logical_call_id="retrying",
        ):
            pass

    def other_request() -> None:
        assert first_released.wait(timeout=2)
        with gate.acquire(
            run_id="run_43",
            seed=43,
            role="target",
            stage="rollout",
            logical_call_id="other",
        ):
            second_acquired.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(retrying_request), executor.submit(other_request)]
        for future in futures:
            future.result(timeout=5)


def test_gate_directory_rejects_conflicting_campaign_policy(tmp_path: Path) -> None:
    gate_dir = tmp_path / "provider_gate"
    CampaignProviderGate(
        gate_dir=gate_dir,
        campaign_id="identity_fixture",
        max_inflight=16,
    )
    with pytest.raises(ValueError, match="different campaign policy"):
        CampaignProviderGate(
            gate_dir=gate_dir,
            campaign_id="identity_fixture",
            max_inflight=12,
        )


def test_g4_target_and_optimizer_share_the_same_sdk_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("skillopt")
    import skillopt.model.openai_compatible_backend as backend
    from experiments.baselines.b3_skillopt.provider_observer import ProviderCallObserver

    counter_lock = threading.Lock()
    current = 0
    peak = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal current, peak
            with counter_lock:
                current += 1
                peak = max(peak, current)
            try:
                time.sleep(0.02)
                usage = SimpleNamespace(
                    prompt_tokens=2,
                    completion_tokens=1,
                    total_tokens=3,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                )
                message = SimpleNamespace(content="ok", tool_calls=[])
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)],
                    usage=usage,
                    _request_id="req_fixture",
                )
            finally:
                with counter_lock:
                    current -= 1

    class Client:
        max_retries = 0
        chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(backend, "_get_client", lambda role: Client())
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)

    def direct_boundary_call(
        messages,
        max_completion_tokens,
        retries,
        stage,
        *,
        role,
        **kwargs,
    ):
        response = backend._get_client(role).chat.completions.create(
            model="fixture-model",
            messages=messages,
            max_tokens=max_completion_tokens,
        )
        return response.choices[0].message.content, {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
            "reasoning_tokens": 0,
            "reasoning_tokens_status": "reported",
        }

    monkeypatch.setattr(backend, "_chat_messages_impl", direct_boundary_call)
    gate = CampaignProviderGate(
        gate_dir=tmp_path / "provider_gate",
        campaign_id="role_fixture",
        max_inflight=1,
    )
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
        run_seed=42,
        application_retry_limit=1,
        retry_delays_seconds=[],
        expected_sdk_max_retries=0,
        campaign_gate=gate,
    )
    observer.install()
    try:
        def call(role: str) -> None:
            backend._chat_messages_impl(
                [{"role": "user", "content": "fixture"}],
                16,
                1,
                "rollout" if role == "target" else "reflection",
                role=role,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(call, "target" if index % 2 == 0 else "optimizer")
                for index in range(8)
            ]
            for future in futures:
                future.result(timeout=5)
    finally:
        observer.uninstall()

    assert peak == 1
    assert {event["role"] for event in observer.events()} == {"target", "optimizer"}
