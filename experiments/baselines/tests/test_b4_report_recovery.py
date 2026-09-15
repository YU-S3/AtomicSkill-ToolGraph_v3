from pathlib import Path

import pytest

from experiments.baselines.b4_embodiskill.repair_reports import ReportOnlyController, protected_hashes
from experiments.baselines.b4_embodiskill.state import write_json


def test_report_only_controller_refuses_learning_and_worker_execution(tmp_path):
    controller = ReportOnlyController({"output": str(tmp_path), "config": {}}, 42, [])
    with pytest.raises(RuntimeError, match="cannot execute"):
        controller.operation("train_00_000", "train")
    with pytest.raises(RuntimeError, match="cannot start"):
        controller._operation_once("test_00_000", "test")


def test_report_recovery_fingerprints_inputs_not_derived_reports(tmp_path):
    write_json(tmp_path / "campaign_lock.json", {})
    for seed in (42,43,44):
        lane = tmp_path / f"seed_{seed}"
        for name in ("run_manifest.json", "selection.json", "checkpoints/train_00_000.json",
                     "attempts/train_00_000/attempt/provider_calls.jsonl", "frozen/digest.json"):
            write_json(lane / name, {})
    before = protected_hashes(tmp_path)
    write_json(tmp_path / "seed_42/summary.json", {"passed": True})
    assert protected_hashes(tmp_path) == before
    write_json(tmp_path / "seed_42/attempts/train_00_000/attempt/provider_calls.jsonl", {"changed": True})
    assert protected_hashes(tmp_path) != before
