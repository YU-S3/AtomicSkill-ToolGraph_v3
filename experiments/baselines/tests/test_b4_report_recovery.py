from pathlib import Path
from types import SimpleNamespace as NS

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


def test_report_only_entrypoint_publishes_canonical_strict_counts(tmp_path, monkeypatch):
    from experiments.baselines.b4_embodiskill import repair_reports as repair
    write_json(tmp_path / "campaign_summary.json", {"passed": False})
    (tmp_path / "REPORT.md").write_text("incomplete")
    lanes = [dict(seed=seed, passed=True, test=dict(tasks=134, official_success=count,
        common_strict_success=124)) for seed, count in ((42,124),(43,126),(44,126))]
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, root, recovery, seed):
            return NS(result=lambda: next(row for row in lanes if row["seed"] == seed))
    monkeypatch.setattr(repair.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(repair, "protected_hashes", lambda root: {"original": "hash"})
    monkeypatch.setattr(repair, "build_campaign_report", lambda roots: {"passed": True})
    monkeypatch.setattr(repair.subprocess, "check_output", lambda *a, **k: "a"*40)
    assert repair.main(["--campaign", str(tmp_path)]) == 0
    assert "| 42 | 124/134 | 124/134 |" in (tmp_path / "REPORT.md").read_text()
    assert repair.read_json(tmp_path / "campaign_summary.json")["passed"] is True
    receipt = next((tmp_path / "report_recovery").glob("*/recovery_report.json"))
    assert repair.read_json(receipt)["model_calls"] == 0
