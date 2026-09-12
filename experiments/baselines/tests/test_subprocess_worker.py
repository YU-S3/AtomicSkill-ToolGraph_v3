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
    assert written_wire["schema_version"] == 4
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


def _external_wire(tmp_path, *, method: str, phase: str) -> WorkerWire:
    output_dir = tmp_path / "run"
    identity = {
        "config_digest": "a" * 64,
        "train_manifest_digest": "b" * 64,
        "external_source_digest": "c" * 64,
    }
    values = {
        "manifest_path": None,
        "validation_manifest_path": None,
        "test_manifest_path": None,
        "frozen_artifact_path": None,
        "initial_skill_path": None,
        "supervision_path": None,
        "external_skillopt_root": None,
    }
    if method == "b4_skillgen_s":
        if phase in {"train", "smoke"}:
            identity["supervision_digest"] = "d" * 64
            values["manifest_path"] = str(tmp_path / "train.json")
            values["supervision_path"] = str(tmp_path / "labels.jsonl")
        else:
            identity.update({
                "evaluation_manifest_digest": "e" * 64,
                "frozen_artifact_digest": "f" * 64,
            })
            values["frozen_artifact_path"] = str(tmp_path / "frozen")
            if phase == "train_eval":
                values["manifest_path"] = str(tmp_path / "train.json")
            else:
                identity["test_manifest_digest"] = "e" * 64
                values["test_manifest_path"] = str(tmp_path / "test.json")
    else:
        identity.update({
            "validation_manifest_digest": "d" * 64,
            "skillopt_source_digest": "e" * 64,
        })
        values["external_skillopt_root"] = str(tmp_path / "skillopt")
        if phase in {"train", "smoke"}:
            identity["initial_skill_digest"] = "f" * 64
            values.update({
                "manifest_path": str(tmp_path / "train.json"),
                "validation_manifest_path": str(tmp_path / "validation.json"),
                "initial_skill_path": str(tmp_path / "initial.md"),
            })
        else:
            identity.update({
                "evaluation_manifest_digest": "1" * 64,
                "frozen_artifact_digest": "2" * 64,
            })
            values["frozen_artifact_path"] = str(tmp_path / "frozen")
            if phase == "train_eval":
                values["manifest_path"] = str(tmp_path / "train.json")
            else:
                identity["test_manifest_digest"] = "1" * 64
                values["test_manifest_path"] = str(tmp_path / "test.json")
    return WorkerWire(
        method=method,
        phase=phase,
        config_path=str(tmp_path / "config.json"),
        output_dir=str(output_dir),
        run_seed=42,
        model={"provider": "test", "model": "test-model"},
        run_id=f"{method}_{phase}",
        result_path=str(output_dir / phase / "worker_result.json"),
        identity=identity,
        external_method_root=str(tmp_path / method),
        **values,
    )


@pytest.mark.parametrize(
    "phase", ["smoke", "train", "train_eval", "smoke_test", "test"],
)
def test_b4_wire_accepts_only_phase_authorized_inputs(tmp_path, phase: str) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase=phase)
    assert WorkerWire.from_dict(wire.to_dict()) == wire


@pytest.mark.parametrize("phase", ["train_eval", "smoke_test", "test"])
def test_b4_frozen_phases_reject_supervision(
    tmp_path, phase: str,
) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase=phase)
    payload = wire.to_dict()
    payload["supervision_path"] = str(tmp_path / "labels.jsonl")
    with pytest.raises(ValueError, match="forbids: supervision_path"):
        WorkerWire.from_dict(payload)


@pytest.mark.parametrize("phase", ["smoke_test", "test"])
def test_b4_heldout_phases_reject_train_content(tmp_path, phase: str) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase=phase)
    payload = wire.to_dict()
    payload["manifest_path"] = str(tmp_path / "train.json")
    with pytest.raises(ValueError, match="forbids: manifest_path"):
        WorkerWire.from_dict(payload)


def test_b4_training_smoke_rejects_test_and_frozen_content(tmp_path) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase="smoke")
    payload = wire.to_dict()
    payload["test_manifest_path"] = str(tmp_path / "test.json")
    with pytest.raises(ValueError, match="forbids: test_manifest_path"):
        WorkerWire.from_dict(payload)

    payload = wire.to_dict()
    payload["frozen_artifact_path"] = str(tmp_path / "frozen")
    with pytest.raises(ValueError, match="forbids: frozen_artifact_path"):
        WorkerWire.from_dict(payload)

    payload = wire.to_dict()
    payload["identity"]["test_manifest_digest"] = "9" * 64
    with pytest.raises(ValueError, match="forbids identity keys: test_manifest_digest"):
        WorkerWire.from_dict(payload)


def test_b4_smoke_test_rejects_supervision_identity(tmp_path) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase="smoke_test")
    payload = wire.to_dict()
    payload["identity"]["supervision_digest"] = "9" * 64
    with pytest.raises(ValueError, match="forbids identity keys: supervision_digest"):
        WorkerWire.from_dict(payload)


def test_b4_wire_rejects_validation_even_when_only_digest_is_exposed(tmp_path) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase="train")
    payload = wire.to_dict()
    payload["identity"]["validation_manifest_digest"] = "9" * 64
    with pytest.raises(ValueError, match="must not bind Validation"):
        WorkerWire.from_dict(payload)


@pytest.mark.parametrize(
    "phase", ["smoke", "train", "train_eval", "smoke_test", "test"],
)
def test_b5_wire_accepts_only_phase_authorized_inputs(tmp_path, phase: str) -> None:
    wire = _external_wire(tmp_path, method="b5_gepa", phase=phase)
    assert WorkerWire.from_dict(wire.to_dict()) == wire


@pytest.mark.parametrize("phase", ["train_eval", "smoke_test", "test"])
def test_b5_frozen_phases_reject_validation_and_initial_skill_paths(
    tmp_path, phase: str,
) -> None:
    wire = _external_wire(tmp_path, method="b5_gepa", phase=phase)
    payload = wire.to_dict()
    payload["validation_manifest_path"] = str(tmp_path / "validation.json")
    with pytest.raises(ValueError, match="forbids: validation_manifest_path"):
        WorkerWire.from_dict(payload)

    payload = wire.to_dict()
    payload["initial_skill_path"] = str(tmp_path / "initial.md")
    with pytest.raises(ValueError, match="initial_skill_path"):
        WorkerWire.from_dict(payload)


def test_external_worker_gets_controlled_pythonpath(tmp_path, monkeypatch) -> None:
    wire = _external_wire(tmp_path, method="b4_skillgen_s", phase="test")
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        write_worker_result(wire, {}, passed=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_protocol.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted"))
    result = run_worker(
        wire=wire,
        worker_module="fake.worker",
        python="python",
        wire_dir=tmp_path / "run" / "test",
    )

    assert result["passed"] is True
    path_entries = captured["env"]["PYTHONPATH"].split(worker_protocol.os.pathsep)
    assert str(worker_protocol._REPO_ROOT) == path_entries[0]
    assert str(worker_protocol._REPO_ROOT / "src") == path_entries[1]
    assert str(tmp_path / "b4_skillgen_s") == path_entries[2]
    assert str(tmp_path / "untrusted") not in path_entries
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
    assert captured["cwd"] == worker_protocol._REPO_ROOT


def test_wire_rejects_unknown_method_or_phase(tmp_path) -> None:
    payload = _wire(tmp_path).to_dict()
    payload["method"] = "not_a_baseline"
    with pytest.raises(ValueError, match="unsupported worker wire method"):
        WorkerWire.from_dict(payload)

    payload = _wire(tmp_path).to_dict()
    payload["phase"] = "optimizer_magic"
    with pytest.raises(ValueError, match="unsupported worker wire phase"):
        WorkerWire.from_dict(payload)
