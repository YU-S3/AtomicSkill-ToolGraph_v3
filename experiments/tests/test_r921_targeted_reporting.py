"""Independent targeted reporting cannot inflate autonomous/formal gates."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from experiments.run_v3_self_tooling_targeted import isolated_config, main, read_fixture_manifest
from experiments.self_tooling_targeted import (
    CandidateHarness, RouteCase, run_node_case, summarize_case, summarize_suite,
)


def test_empty_skipped_or_unreached_cases_never_pass():
    assert not summarize_suite([])["passed"]
    for incomplete in ({}, {"skipped": True}, {"case_passed": False, "route_last_stage": "not_reached"}):
        result = summarize_suite([incomplete])
        assert not result["passed"]
        assert result["positive_trial_passes"] == 0


def test_partial_suite_does_not_pass_before_the_second_required_case():
    cases = [{"case_passed": True}]
    assert not summarize_suite(cases, expected_case_count=2)["passed"]
    assert not summarize_suite(cases, expected_case_count=2)["complete"]
    cases.append({"case_passed": True})
    assert summarize_suite(cases, expected_case_count=2)["passed"]


def test_failed_live_case_retains_incurred_cost_without_a_completed_result(tmp_path, monkeypatch):
    import experiments.run_v3_self_tooling_targeted as runner
    monkeypatch.setenv("MODEL_API_KEY", "fixture_not_sent")
    monkeypatch.setattr(runner, "hash_code", lambda _root: "reporting_fixture")

    def interrupted(_system, _case, *, audit, **kwargs):
        audit("internal", 0, "usage", [{"provider": "fixture_external", "bucket": "tool_builder_runtime",
            "total_tokens": 190, "call_count": 1, "latency_ms": 23, "provider_metadata": {}}])
        raise RuntimeError("fixture transport interrupted")

    monkeypatch.setattr(runner, "run_node_case", interrupted)
    output = tmp_path / "interrupted"
    assert main(["--config", "configs/alfworld_train_full_120_r92_seed42.yaml", "--mode", "live-builder",
                 "--fixture-suite", "core-two", "--output-dir", str(output)]) == 3
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["passed"] and not summary["complete"]
    assert summary["targeted_external_tokens"] == 190
    assert summary["positive_trial_passes"] == 0


def test_positive_and_expected_rejection_counts_are_separate():
    summary = summarize_suite([
        {"case_passed": True, "expected_rejection": "r0_rejected", "fixture_tokens": 0},
        {"case_passed": True, "expected_rejection": None, "targeted_external_tokens": 190},
    ])
    assert summary["positive_trial_passes"] == summary["expected_rejection_passes"] == 1
    assert summary["targeted_external_tokens"] == 190
    assert summary["is_formal_experiment"] is False
    assert summary["phase"] == "targeted_smoke"


def test_native_no_tool_is_a_failed_positive_not_static_rejection(tmp_path):
    case = RouteCase("expected_no_tool", variant="no_tool")
    source = load_config("configs/alfworld_train_full_120_r92_seed42.yaml")
    config = isolated_config(source, tmp_path)
    with AtomicSkillGraphSystem(config, harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        negative = summarize_case(case, outcome)
        assert negative["case_passed"]
        positive_case = RouteCase(case.case_id)
        positive = summarize_case(positive_case, outcome)
        assert not positive["case_passed"]
        assert positive["route_last_stage"] == "builder_no_tool"
        assert not positive["trial_started"]
        assert positive["targeted_external_tokens"] == positive["fixture_tokens"] == 0
        assert all(not costs["external"] for costs in positive["costs_by_provider_and_bucket"].values())
        assert all("http_status" not in reply["provider_metadata"] for reply in outcome["builder"].replies)


def test_isolation_keeps_all_formal_resource_settings(tmp_path):
    source = load_config("configs/alfworld_train_full_120_r92_seed42.yaml")
    before = copy.deepcopy(source)
    effective = isolated_config(source, tmp_path)
    for key in ("llm", "runtime", "planner", "lifecycle", "cold_start", "extraction"):
        assert source[key] == effective[key]
    assert source == before
    assert Path(effective["data_dir"]).is_relative_to(tmp_path)


def test_no_key_never_counts_as_live_gate_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    output = tmp_path / "missing_key"
    assert main(["--config", "configs/alfworld_train_full_120_r92_seed42.yaml", "--mode", "live-builder",
                 "--fixture-suite", "core-two", "--output-dir", str(output)]) == 3
    result = json.loads((output / "summary.json").read_text())
    assert not result["passed"]
    assert result["positive_trial_passes"] == 0


def test_existing_output_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "keep.json"
    sentinel.write_text("unchanged")
    assert main(["--config", "unused", "--mode", "live-builder", "--fixture-suite", "core-two",
                 "--output-dir", str(tmp_path)]) == 2
    assert sentinel.read_text() == "unchanged"
    assert not (tmp_path / "summary.json").exists()


def test_frozen_test_manifest_is_not_accepted(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fixture_suite_version": "r921-core-two-v1", "cases": [{
        "case_id": "hidden_test", "task_manifest": {"ordinal": 0, "task_id": "test", "task_signature": "hash",
        "knowledge_milestone": "fixture", "benchmark": "alfworld", "split": "eval_out_of_distribution"}}]}))
    with pytest.raises(ValueError, match="train/dev only"):
        read_fixture_manifest(manifest)


def test_formal_entrypoint_never_imports_targeted_provider():
    result = subprocess.run([sys.executable, "-c", "import sys; import experiments.run_v3_train; "
        "assert 'experiments.self_tooling_targeted' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
