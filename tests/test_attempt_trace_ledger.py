from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomic_skillgraph.traces.schema import TaskRecord, TraceRecord
from atomic_skillgraph.traces.store import TraceStore
from experiments.protocol import (
    AttemptTraceLedger,
    ProtocolError,
    RunManifest,
    TaskManifest,
    validate_distinct_formal_tasks,
)
from experiments.report import (
    summarize_traces,
    validate_formal_usage,
    validate_usage_event_persistence,
)


def _write_trace(
    root: Path,
    *,
    trace_id: str,
    task_id: str,
    task_signature: str,
    event_id: str,
    tokens: int,
    cost: float,
    success: bool = False,
    maintenance: bool = False,
    milestone: str = "periodic_success_5",
) -> dict:
    metadata = {
        "usage_reconciliation": {
            "episode_total_tokens": tokens,
            "bucket_total_tokens": tokens,
            "unattributed_total_tokens": 0,
            "token_mismatch": 0,
        },
    }
    actual_task_id = task_id
    task_type = "pick_and_place_simple"
    if maintenance:
        actual_task_id = f"maintenance_{trace_id}"
        task_type = "maintenance"
        metadata.update({
            "trace_kind": "maintenance",
            "triggering_task_id": task_id,
            "milestone": milestone,
        })
    payload = {
        "schema_version": 3,
        "trace_id": trace_id,
        "task": {
            "task_id": actual_task_id,
            "goal": "goal",
            "benchmark": "alfworld" if not maintenance else "maintenance",
            "task_type": task_type,
            "task_signature": task_signature,
            "context": {},
        },
        "runtime_plan": {},
        "planner_audit": {},
        "node_records": [],
        "agent_turns": [],
        "llm_usage": [{
            "event_id": event_id,
            "bucket": "evolution_repair" if maintenance else "runtime_dynamic",
            "prompt_tokens": tokens - 1,
            "completion_tokens": 1,
            "total_tokens": tokens,
            "reasoning_tokens": 0,
            "call_count": 1,
            "latency_ms": float(tokens),
            "provider_metadata": {"usage_status": "reported", "cost_usd": cost},
        }],
        "metadata": metadata,
        "benchmark_success": success,
        "graph_self_sufficient_success": False,
        "infrastructure_failure": not success,
        "started_at": 1.0,
        "ended_at": 2.0,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{trace_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8",
    )
    return payload


def _manifest(task_id: str, signature: str) -> RunManifest:
    task = TaskManifest(0, task_id, signature, "cold", "alfworld", "train")
    return RunManifest.create(
        run_id="formal_run",
        phase="train",
        config_hash="config",
        code_commit="code",
        knowledge_digest="knowledge",
        tasks=(task,),
    )


def test_crash_resume_preserves_failed_task_and_periodic_maintenance_usage(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    task_id, signature = "task-1", "signature-1"
    manifest = _manifest(task_id, signature)

    # Attempt 1 hard-crashes after both immutable Traces land, before capture.
    first = ledger.begin(
        run_id=manifest.run_id,
        task_id=task_id,
        task_signature=signature,
        attempt_kind="task",
        sequence=1,
    )
    failed = _write_trace(
        traces, trace_id="trace_failed", task_id=task_id,
        task_signature=signature, event_id="event_failed", tokens=10, cost=0.10,
    )
    periodic = _write_trace(
        traces, trace_id="trace_periodic", task_id=task_id,
        task_signature="maintenance-signature", event_id="event_periodic",
        tokens=4, cost=0.04, maintenance=True,
    )
    assert not (ledger.root / f"{first.attempt_id}.capture.json").exists()

    recovered = ledger.recover_pending(run_id=manifest.run_id)
    assert recovered[0]["trace_ids"] == ["trace_failed", "trace_periodic"]

    # Attempt 2 completes.  Its authoritative task row must not also be auxiliary.
    second = ledger.begin(
        run_id=manifest.run_id,
        task_id=task_id,
        task_signature=signature,
        attempt_kind="task",
        sequence=2,
    )
    completed = _write_trace(
        traces, trace_id="trace_completed", task_id=task_id,
        task_signature=signature, event_id="event_completed", tokens=20,
        cost=0.20, success=True,
    )
    ledger.capture(second, reason="run_task_returned")

    # A crashed final-maintenance pass is also paid usage, while the successful
    # replay is the authoritative maintenance Trace and must be counted once.
    orphan_final_attempt = ledger.begin(
        run_id=manifest.run_id,
        task_id=task_id,
        task_signature=signature,
        attempt_kind="maintenance",
        sequence=1,
    )
    orphan_final = _write_trace(
        traces, trace_id="trace_orphan_final", task_id=task_id,
        task_signature="maintenance-signature", event_id="event_orphan_final",
        tokens=6, cost=0.06, maintenance=True,
        milestone="formal_full_30_final_batch",
    )
    assert ledger.recover_pending(run_id=manifest.run_id)[0]["attempt_id"] == (
        orphan_final_attempt.attempt_id
    )
    official_final_attempt = ledger.begin(
        run_id=manifest.run_id,
        task_id=task_id,
        task_signature=signature,
        attempt_kind="maintenance",
        sequence=2,
    )
    official_final = _write_trace(
        traces, trace_id="trace_official_final", task_id=task_id,
        task_signature="maintenance-signature", event_id="event_official_final",
        tokens=8, cost=0.08, maintenance=True,
        milestone="formal_full_30_final_batch",
    )
    ledger.capture(official_final_attempt, reason="run_maintenance_returned")

    auxiliary = ledger.auxiliary_traces(
        manifest=manifest,
        excluded_trace_ids={"trace_completed", "trace_official_final"},
    )
    assert {item["trace_id"] for item in auxiliary} == {
        failed["trace_id"], periodic["trace_id"], orphan_final["trace_id"],
    }
    summary = summarize_traces(
        [completed], auxiliary_usage_traces=[official_final, *auxiliary],
    )
    assert summary["task_count"] == 1
    assert summary["solved_task_count"] == 1
    assert summary["benchmark_success_rate"] == 1.0
    assert summary["total_tokens"] == 48
    assert summary["cost_usd"] == pytest.approx(0.48)
    assert summary["cost_usd_per_solved_task"] == pytest.approx(0.48)
    validate_usage_event_persistence([], [completed, official_final, *auxiliary])


def test_attempt_usage_is_deduplicated_and_does_not_mix_historical_traces(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    historical = _write_trace(
        traces, trace_id="trace_other_run", task_id="other-task",
        task_signature="other-signature", event_id="other-event", tokens=3, cost=0.03,
    )
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    manifest = _manifest("task-1", "signature-1")
    attempt = ledger.begin(
        run_id=manifest.run_id,
        task_id="task-1",
        task_signature="signature-1",
        attempt_kind="task",
        sequence=1,
    )
    current = _write_trace(
        traces, trace_id="trace_current", task_id="task-1",
        task_signature="signature-1", event_id="shared-event", tokens=5, cost=0.05,
    )
    ledger.capture(attempt, reason="task_exception")
    auxiliary = ledger.auxiliary_traces(manifest=manifest)
    assert [item["trace_id"] for item in auxiliary] == [current["trace_id"]]
    assert historical["trace_id"] not in {item["trace_id"] for item in auxiliary}

    authoritative = _write_trace(
        traces, trace_id="trace_authoritative", task_id="task-1",
        task_signature="signature-1", event_id="shared-event", tokens=7,
        cost=0.07, success=True,
    )
    with pytest.raises(ValueError, match="multiple Traces"):
        validate_usage_event_persistence([], [authoritative, *auxiliary])


def test_trace_root_can_be_claimed_by_only_one_formal_run(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    first = AttemptTraceLedger(tmp_path / "attempt_history_a", traces)
    first.begin(
        run_id="formal_run_a", task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    second = AttemptTraceLedger(tmp_path / "attempt_history_b", traces)
    with pytest.raises(ProtocolError, match="owned by a different formal run"):
        second.begin(
            run_id="formal_run_b", task_id="task-2", task_signature="signature-2",
            attempt_kind="task", sequence=1,
        )


def test_trace_store_refuses_to_replace_existing_trace_id(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    task = TaskRecord(
        "task-1", "alfworld", "goal", "pick_and_place_simple", "signature-1", {},
    )
    trace = TraceRecord.create(task, {}, {}, {})
    store.save_atomic(trace)
    with pytest.raises(FileExistsError):
        store.save_atomic(trace)


def test_attempt_capture_detects_baseline_trace_content_rewrite(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    historical = _write_trace(
        traces, trace_id="trace_historical", task_id="task-1",
        task_signature="signature-1", event_id="event-historical", tokens=2,
        cost=0.02,
    )
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    attempt = ledger.begin(
        run_id="formal_run", task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    historical["metadata"]["tampered"] = True
    (traces / "trace_historical.json").write_text(
        json.dumps(historical), encoding="utf-8",
    )
    _write_trace(
        traces, trace_id="trace_current", task_id="task-1",
        task_signature="signature-1", event_id="event-current", tokens=3,
        cost=0.03,
    )
    with pytest.raises(ProtocolError, match="baseline content changed"):
        ledger.capture(attempt, reason="run_task_returned")


def test_attempt_report_detects_captured_trace_content_rewrite(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    manifest = _manifest("task-1", "signature-1")
    attempt = ledger.begin(
        run_id=manifest.run_id, task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    captured = _write_trace(
        traces, trace_id="trace_captured", task_id="task-1",
        task_signature="signature-1", event_id="event-captured", tokens=3,
        cost=0.03,
    )
    ledger.capture(attempt, reason="task_exception")
    captured["metadata"]["tampered"] = True
    (traces / "trace_captured.json").write_text(
        json.dumps(captured), encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="captured immutable Trace content changed"):
        ledger.auxiliary_traces(manifest=manifest)


def test_attempt_capture_fails_closed_on_cross_task_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    attempt = ledger.begin(
        run_id="formal_run", task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    _write_trace(
        traces, trace_id="trace_wrong", task_id="task-2",
        task_signature="signature-2", event_id="event-wrong", tokens=2, cost=0.02,
    )
    with pytest.raises(ProtocolError, match="ownership mismatch"):
        ledger.capture(attempt, reason="resume_recovery")


def test_legacy_run_without_attempt_history_remains_reportable(tmp_path: Path) -> None:
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", tmp_path / "traces")
    manifest = _manifest("task-1", "signature-1")
    assert ledger.recover_pending(run_id=manifest.run_id) == []
    assert ledger.auxiliary_traces(manifest=manifest) == []


def test_pending_attempt_without_new_trace_cannot_be_recovered(tmp_path: Path) -> None:
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", tmp_path / "traces")
    manifest = _manifest("task-1", "signature-1")
    attempt = ledger.begin(
        run_id=manifest.run_id, task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    capture_path = ledger.root / f"{attempt.attempt_id}.capture.json"
    with pytest.raises(ProtocolError, match="no immutable Trace"):
        ledger.recover_pending(run_id=manifest.run_id)
    assert not capture_path.exists()
    assert ledger.pending(run_id=manifest.run_id) == (attempt,)


def test_forged_empty_capture_is_rejected(tmp_path: Path) -> None:
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", tmp_path / "traces")
    manifest = _manifest("task-1", "signature-1")
    attempt = ledger.begin(
        run_id=manifest.run_id, task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    capture = {
        "schema_version": 3,
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "task_id": attempt.task_id,
        "task_signature": attempt.task_signature,
        "attempt_kind": attempt.attempt_kind,
        "sequence": attempt.sequence,
        "trace_ids": [],
        "reason": "forged",
    }
    (ledger.root / f"{attempt.attempt_id}.capture.json").write_text(
        json.dumps(capture), encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="invalid trace_ids"):
        ledger.auxiliary_traces(manifest=manifest)


def test_landed_zero_llm_task_and_maintenance_traces_are_valid(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    manifest = _manifest("task-1", "signature-1")
    attempt = ledger.begin(
        run_id=manifest.run_id, task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    task_trace = _write_trace(
        traces, trace_id="trace_zero_task", task_id="task-1",
        task_signature="signature-1", event_id="unused-task", tokens=1, cost=0.0,
    )
    maintenance_trace = _write_trace(
        traces, trace_id="trace_zero_maintenance", task_id="task-1",
        task_signature="maintenance", event_id="unused-maintenance", tokens=1,
        cost=0.0, maintenance=True,
    )
    for payload in (task_trace, maintenance_trace):
        payload["llm_usage"] = []
        payload["agent_turns"] = []
        payload["metadata"]["usage_reconciliation"] = {
            "episode_total_tokens": 0,
            "bucket_total_tokens": 0,
            "unattributed_total_tokens": 0,
            "token_mismatch": 0,
        }
        (traces / f"{payload['trace_id']}.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
    capture = ledger.capture(attempt, reason="run_task_returned")
    assert capture["trace_ids"] == ["trace_zero_maintenance", "trace_zero_task"]
    auxiliary = ledger.auxiliary_traces(manifest=manifest)
    assert validate_formal_usage(auxiliary)["total_tokens"] == 0


def test_successful_task_missing_expected_periodic_stays_pending_until_trace_lands(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    manifest = _manifest("task-1", "signature-1")
    attempt = ledger.begin(
        run_id=manifest.run_id, task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
        expected_periodic_milestone="online_success_5",
    )
    _write_trace(
        traces, trace_id="trace_success", task_id="task-1",
        task_signature="signature-1", event_id="event-success", tokens=5,
        cost=0.05, success=True,
    )
    capture_path = ledger.root / f"{attempt.attempt_id}.capture.json"
    with pytest.raises(ProtocolError, match="lacks expected periodic"):
        ledger.recover_pending(run_id=manifest.run_id)
    assert not capture_path.exists()
    assert ledger.pending(run_id=manifest.run_id) == (attempt,)

    _write_trace(
        traces, trace_id="trace_periodic_5", task_id="task-1",
        task_signature="maintenance", event_id="event-periodic-5", tokens=3,
        cost=0.03, maintenance=True, milestone="online_success_5",
    )
    recovered = ledger.recover_pending(run_id=manifest.run_id)
    assert recovered[0]["trace_ids"] == ["trace_periodic_5", "trace_success"]
    assert ledger.pending(run_id=manifest.run_id) == ()


def test_successful_task_with_matching_expected_periodic_captures(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    attempt = ledger.begin(
        run_id="formal_run", task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
        expected_periodic_milestone="online_success_10",
    )
    _write_trace(
        traces, trace_id="trace_success", task_id="task-1",
        task_signature="signature-1", event_id="event-success", tokens=5,
        cost=0.05, success=True,
    )
    _write_trace(
        traces, trace_id="trace_periodic", task_id="task-1",
        task_signature="maintenance", event_id="event-periodic", tokens=3,
        cost=0.03, maintenance=True, milestone="online_success_10",
    )
    assert ledger.capture(attempt, reason="run_task_returned")["trace_ids"] == [
        "trace_periodic", "trace_success",
    ]


def test_successful_task_before_periodic_threshold_needs_only_task_trace(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    attempt = ledger.begin(
        run_id="formal_run", task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
    )
    _write_trace(
        traces, trace_id="trace_success", task_id="task-1",
        task_signature="signature-1", event_id="event-success", tokens=5,
        cost=0.05, success=True,
    )
    assert ledger.capture(attempt, reason="run_task_returned")["trace_ids"] == [
        "trace_success",
    ]


def test_failed_task_does_not_require_conditional_periodic_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    ledger = AttemptTraceLedger(tmp_path / "attempt_history", traces)
    attempt = ledger.begin(
        run_id="formal_run", task_id="task-1", task_signature="signature-1",
        attempt_kind="task", sequence=1,
        expected_periodic_milestone="online_success_5",
    )
    _write_trace(
        traces, trace_id="trace_failure", task_id="task-1",
        task_signature="signature-1", event_id="event-failure", tokens=5,
        cost=0.05, success=False,
    )
    assert ledger.capture(attempt, reason="run_task_returned")["trace_ids"] == [
        "trace_failure",
    ]


def test_formal_selection_rejects_duplicate_task_signatures() -> None:
    tasks = [
        {"task_id": "task-1", "task_signature": "same"},
        {"task_id": "task-2", "task_signature": "same"},
    ]
    with pytest.raises(ProtocolError, match="duplicate task_signature"):
        validate_distinct_formal_tasks(tasks, expected_total=2)

    manifests = (
        TaskManifest(0, "task-1", "same", "cold"),
        TaskManifest(1, "task-2", "same", "cold"),
    )
    with pytest.raises(ValueError, match="task_signature values must be unique"):
        RunManifest.create(
            run_id="duplicate-signatures",
            phase="train",
            config_hash="config",
            code_commit="code",
            knowledge_digest="knowledge",
            tasks=manifests,
        )
