"""Fail-closed tests for the per-method worker wire/result protocol."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.baselines.common import subprocess_worker as worker_protocol
from experiments.baselines.common.subprocess_worker import (
    WorkerWire,
    run_worker,
    write_worker_result,
)


def _wire(tmp_path) -> WorkerWire:
    output_dir = tmp_path / "run"
    return WorkerWire(
        method="b3_skillopt",
        phase="train",
        manifest_path=str(tmp_path / "train.json"),
        validation_manifest_path=str(tmp_path / "validation.json"),
        test_manifest_path=None,
        config_path=str(tmp_path / "config_resolved.json"),
        output_dir=str(output_dir),
        run_seed=42,
        model={"provider": "test", "model": "test-model"},
        run_id="run_20260907T010203Z_abcd1234",
        result_path=str(output_dir / "train" / "worker_result.json"),
        identity={
            "config_digest": "a" * 64,
            "train_manifest_digest": "b" * 64,
            "validation_manifest_digest": "c" * 64,
        },
    )


def test_worker_roundtrip_reads_only_identity_bound_phase_result(
    tmp_path, monkeypatch,
) -> None:
    wire = _wire(tmp_path)

    def fake_run(*args, **kwargs):
        write_worker_result(wire, {"metric": 7}, passed=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_protocol.subprocess, "run", fake_run)
    result = run_worker(
        wire=wire,
        worker_module="fake.worker",
        python="python",
        wire_dir=tmp_path / "run" / "train",
    )

    assert result["passed"] is True
    assert result["metric"] == 7
    assert result["worker_exit_code"] == 0
    written_wire = json.loads(
        (tmp_path / "run" / "train" / "worker_wire.json").read_text(
            encoding="utf-8"
        )
    )
    assert written_wire["schema_version"] == 2
    assert written_wire["run_id"] == wire.run_id
    assert written_wire["identity"] == wire.identity


def test_nonzero_exit_cannot_be_overridden_by_passed_result(
    tmp_path, monkeypatch,
) -> None:
    wire = _wire(tmp_path)

    def fake_run(*args, **kwargs):
        write_worker_result(wire, {"metric": 7}, passed=True)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(worker_protocol.subprocess, "run", fake_run)
    result = run_worker(
        wire=wire,
        worker_module="fake.worker",
        python="python",
        wire_dir=tmp_path / "run" / "train",
    )

    assert result["passed"] is False
    assert result["worker_exit_code"] == 9
    assert "non-zero" in result["error"]


def test_worker_result_identity_mismatch_fails_closed(tmp_path, monkeypatch) -> None:
    wire = _wire(tmp_path)

    def fake_run(*args, **kwargs):
        result_path = tmp_path / "run" / "train" / "worker_result.json"
        result_path.write_text(
            json.dumps({
                "schema_version": 1,
                "method": wire.method,
                "phase": wire.phase,
                "run_id": "a_different_run",
                "identity": wire.identity,
                "passed": True,
            }),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_protocol.subprocess, "run", fake_run)
    result = run_worker(
        wire=wire,
        worker_module="fake.worker",
        python="python",
        wire_dir=tmp_path / "run" / "train",
    )

    assert result["passed"] is False
    assert "identity mismatch for run_id" in result["error"]


def test_stale_result_is_rejected_before_worker_launch(tmp_path, monkeypatch) -> None:
    wire = _wire(tmp_path)
    result_path = tmp_path / "run" / "train" / "worker_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{}", encoding="utf-8")
    launched = False

    def fake_run(*args, **kwargs):
        nonlocal launched
        launched = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_protocol.subprocess, "run", fake_run)
    with pytest.raises(FileExistsError, match="stale worker result"):
        run_worker(
            wire=wire,
            worker_module="fake.worker",
            python="python",
            wire_dir=tmp_path / "run" / "train",
        )
    assert launched is False


def test_atomic_result_writer_never_overwrites_or_changes_identity(tmp_path) -> None:
    wire = _wire(tmp_path)
    write_worker_result(wire, {"metric": 1}, passed=True)
    with pytest.raises(FileExistsError):
        write_worker_result(wire, {"metric": 2}, passed=True)

    other = _wire(tmp_path / "other")
    with pytest.raises(ValueError, match="reserved field 'run_id'"):
        write_worker_result(
            other,
            {"run_id": "forged_run"},
            passed=True,
        )


def test_wire_rejects_old_schema_and_invalid_digests(tmp_path) -> None:
    wire = _wire(tmp_path)
    payload = wire.to_dict()
    payload["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        WorkerWire.from_dict(payload)

    payload = wire.to_dict()
    payload["identity"]["config_digest"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        WorkerWire.from_dict(payload)
