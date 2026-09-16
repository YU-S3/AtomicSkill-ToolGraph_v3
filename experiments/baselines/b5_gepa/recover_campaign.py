"""Audited GEPA continuation after the zero-output usage validation defect.

Keep the original campaign, failed attempt, and completed lanes immutable.
Only controller source identity may cross the audited repair boundary. Model,
configuration, data, parallelism, upstream code and GEPA state remain bound to
the original lock. This is not a general protocol-failure retry switch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.usage import UsageSnapshot
from experiments.baselines.report_campaign import build_campaign_report
from experiments.baselines.run_seed_campaign import (
    _default_command_runner, _method_campaign_lease, _sha256_file,
    _write_json_atomic, inspect_clean_source, _resource_summary, _campaign_cost_summary,
)
from . import run_seed_campaign as campaign
from .qualification import verify_load_receipt, verify_smoke_receipt, read

REPO = Path(__file__).resolve().parents[3]
POLICY = "b5-known-zero-output-usage-recovery-v1"
# These are software migration authorities, never benchmark task exceptions.
ORIGINAL_COMMIT = "c4c9766dfe1b791c1e3171a8d882ffe6efce0e47"
AUDITED_BASE = "9ad0870f410b83097000a377676d52504af261c7"
REPAIR_FILES = {
    "experiments/baselines/b3_skillopt/episode_runner.py",
    "experiments/baselines/b3_skillopt/provider_observer.py",
    "experiments/baselines/b3_skillopt/common_alfworld_adapter.py",
    "experiments/baselines/b3_skillopt/worker.py",
    "experiments/baselines/b5_gepa/controller.py",
    "experiments/baselines/b5_gepa/run_seed_campaign.py",
    "experiments/baselines/b5_gepa/recover_campaign.py",
    "experiments/baselines/tests/test_b5_usage_recovery.py",
    "experiments/baselines/launch_gepa.sh",
    "experiments/baselines/B5_RECOVERY.md",
}


def reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha256_file(path)}


def verify_reference(ref):
    if reference(ref["path"]) != ref:
        raise ValueError(f"Recovery evidence changed: {ref['path']}")


def audit_source(lock):
    source = inspect_clean_source(REPO)
    if lock["controller_commit"] != ORIGINAL_COMMIT:
        raise ValueError("This recovery policy does not cover the original controller version")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", AUDITED_BASE, source["commit"]],
        cwd=REPO, text=True).splitlines()
    if set(changed) - REPAIR_FILES:
        raise ValueError("Unaudited recovery source changes: " + str(sorted(set(changed)-REPAIR_FILES)))
    # c4c9766 -> 9ad0870 was separately audited: B0 addition and report replay
    # serialization. B5 optimizer/config/upstream and provider behavior are unchanged.
    return source


def load_failed_usage(root):
    """Reconstruct missing usage from disjoint immutable provider sidecars."""
    from experiments.baselines.b3_skillopt.worker import _provider_usage
    manifest, failure = read(root / "run_manifest.json"), read(root / "failure.json")
    if (failure.get("passed") is not False
            or failure.get("failure_kind") != "protocol_failure"
            or not str(failure.get("error", "")).endswith(
                "RuntimeError: episode provider evidence has invalid token usage")
            or (root / "completion.json").exists()):
        raise ValueError("Recovery requires the evidenced zero-output usage validation failure")
    paths = sorted((root / "train").rglob("provider_calls.jsonl"))
    events = [json.loads(line) for path in paths
              for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    zero = [e for e in events if e.get("status") == "succeeded"
            and e.get("role") == "target" and e.get("completion_tokens") == 0
            and e.get("prompt_tokens", 0) > 0 and e.get("finish_reason") == "stop"
            and e.get("content_present") is False and e.get("budget_exhaustion_count") == 0]
    if not zero:
        raise ValueError("No known zero-output target response proves this repair applies")
    if any(e.get("status") != "succeeded" or e.get("usage_complete") is not True
           or e.get("cache_reused") for e in events):
        raise ValueError("Failed-prefix usage cannot be reconstructed completely")
    wire = SimpleNamespace(method=manifest["method"], phase=manifest["phase"],
                           run_id=manifest["run_id"], run_seed=manifest["run_seed"],
                           model=manifest["model_identity"], campaign=manifest.get("campaign"))
    usage = _provider_usage(SimpleNamespace(events=lambda: events), wire=wire,
        wall_time_ms=round(1000*(failure["failed_at_unix"]-manifest["started_at_unix"])))
    return usage, [reference(p) for p in paths], [e["call_id"] for e in zero]


def current_smoke_identity(lock):
    spec = campaign.GEPACampaignSpec("b5_gepa", campaign.FORMAL_SEEDS,
        *(Path(lock["manifests"][role]["path"]) for role in ("train", "validation", "test")),
        REPO / "configs/baselines/b5_gepa.yaml", Path(lock["campaign_root"]),
        Path(lock["worker_python"]), REPO)
    current = campaign.build_campaign_lock(spec, spec.output_dir, "recovery_validation",
                                            audit_source(lock))
    for key in ("model_identity", "external_gepa_commit", "external_runtime_tree_digest",
                "external_skillopt_commit", "skillopt_runtime_tree_digest", "initial_skill_sha256",
                "train_manifest_digest", "validation_manifest_digest", "test_manifest_digest",
                "worker_python", "provider_completion_cap", "reasoning_effort", "budget_policy_version"):
        if current[key] != lock[key]:
            raise ValueError("Recovery changes method authority: " + key)
    if current["formal_config_digest"] != lock["source_formal_config_digest"]:
        raise ValueError("Recovery changes the original formal configuration")
    return current


def validate_recovery(path, lock_path):
    receipt, lock = read(path), read(lock_path)
    if (receipt.get("policy") != POLICY or receipt.get("passed") is not True
            or receipt["original_lock"] != reference(lock_path)
            or receipt["current_source"] != audit_source(lock)
            or receipt["original_commit"] != lock["controller_commit"]
            or receipt["original_code_digest"] != lock["controller_code_digest"]):
        raise ValueError("Recovery receipt source/lock/policy mismatch")
    if verify_smoke_receipt(receipt["smoke"]["path"], current_smoke_identity(lock)) != receipt["smoke"]:
        raise ValueError("Recovery smoke changed")
    for evidence in receipt["protected_evidence"]:
        verify_reference(evidence)
    source = Path(receipt["source_run"])
    if digest_directory(source / "train/gepa_state") != receipt["state_digest"]:
        raise ValueError("Recovery checkpoint changed")
    usage, _, zero_ids = load_failed_usage(source)
    if zero_ids != receipt["zero_output_calls"]:
        raise ValueError("Recovery defect evidence changed")
    verify_reference(receipt["reconstructed_usage"])
    if UsageSnapshot.load(receipt["reconstructed_usage"]["path"]).to_dict() != usage.to_dict():
        raise ValueError("Reconstructed failed-prefix usage differs from provider evidence")
    return receipt


def prepare(source, output, smoke):
    lock_path = source / "campaign_lock.json"
    lock, report = read(lock_path), read(source / "campaign_report.json")
    if report.get("passed") is not False or report["campaign_lock_digest"] != _sha256_file(lock_path):
        raise ValueError("Recovery needs an intact failed campaign")
    failed = [lane for lane in report["lanes"] if lane.get("passed") is not True]
    completed = [lane for lane in report["lanes"] if lane.get("passed") is True]
    if len(failed) != 1 or len(completed) != 2:
        raise ValueError("Recovery requires two completed lanes and exactly one failed lane")
    if sorted(lane["seed"] for lane in report["lanes"]) != list(campaign.FORMAL_SEEDS):
        raise ValueError("Original campaign seeds differ from formal protocol")
    # The last failed attempt owns the latest durable optimizer checkpoint.
    old_run = Path(failed[0]["train_attempts"][-1]["root"]).resolve(strict=True)
    if len(failed[0]["train_attempts"]) != 1:
        raise ValueError("This recovery requires a single failed-prefix cost lineage")
    old_run.relative_to(source)
    manifest = read(old_run / "run_manifest.json")
    if (manifest["run_seed"] != failed[0]["seed"] or manifest["method"] != "b5_gepa"
            or manifest["phase"] != "train" or manifest["controller_commit"] != lock["controller_commit"]
            or manifest["campaign"]["campaign_lock_digest"] != _sha256_file(lock_path)):
        raise ValueError("Failed lane is not bound to the original campaign")
    state = old_run / "train/gepa_state"
    if not (state / "gepa_state.bin").is_file() or (state / "gepa.stop").exists():
        raise ValueError("No resumable durable GEPA state")
    if any(p.is_symlink() for p in state.rglob("*")):
        raise ValueError("GEPA checkpoint contains symlinks")
    verify_load_receipt(lock, source)
    if verify_smoke_receipt(lock["smoke_qualification"]["path"], lock) != lock["smoke_qualification"]:
        raise ValueError("Original smoke evidence changed")
    current = current_smoke_identity(lock)
    resolved_spec = campaign.GEPACampaignSpec("b5_gepa", campaign.FORMAL_SEEDS,
        *(Path(lock["manifests"][role]["path"]) for role in ("train", "validation", "test")),
        Path(lock["config_path"]), output, Path(lock["worker_python"]), REPO)
    resolved_config = campaign._merged_config(resolved_spec)
    if (campaign._formal_config_digest(resolved_config) != lock["formal_config_digest"]
            or campaign._provider_cap_mismatches(resolved_config, lock,
                                               cap=lock["campaign_provider_max_inflight"])):
        raise ValueError("Original resolved campaign configuration changed")
    smoke_ref = verify_smoke_receipt(smoke, current)
    protected = [reference(lock_path), reference(source / "campaign_report.json"),
                 reference(old_run / "run_manifest.json"), reference(old_run / "failure.json")]
    for lane in completed:
        train, test = Path(lane["train_root"]), Path(lane["test_root"])
        train.resolve().relative_to(source)
        test.resolve().relative_to(source)
        frozen = campaign._validate_train(train, lane["seed"])
        campaign._validate_test(test, seed=lane["seed"], train_root=train, frozen=frozen)
        protected += [reference(p) for p in (train / "report.json", train / "completion.json",
            train / "run_manifest.json", test / "test_report.json", test / "completion.json",
            test / "run_manifest.json", test / "test/task_rows.jsonl")]
        protected += [reference(p) for p in sorted(frozen.root.rglob("*")) if p.is_file()]
    usage, provider_refs, zero_ids = load_failed_usage(old_run)
    output.mkdir(parents=True, exist_ok=False)
    usage_path = output / "failed_prefix_usage.json"
    _write_json_atomic(usage_path, usage.to_dict(), overwrite=False)
    receipt = dict(policy=POLICY, passed=True, seed=failed[0]["seed"], source_run=str(old_run),
        original_lock=reference(lock_path), original_commit=lock["controller_commit"],
        original_code_digest=lock["controller_code_digest"], current_source=audit_source(lock),
        protected_evidence=protected+provider_refs, zero_output_calls=zero_ids,
        reconstructed_usage=reference(usage_path), state_digest=digest_directory(state), smoke=smoke_ref,
        method_changed=False, original_failure_reclassified=False, created_at_unix=time.time())
    path = output / "recovery_receipt.json"
    _write_json_atomic(path, receipt, overwrite=False)
    return validate_recovery(path, lock_path)


def publish(output, lanes, receipt):
    if sorted(row["seed"] for row in lanes) != list(campaign.FORMAL_SEEDS):
        raise ValueError("Final report must contain each of the three seeds exactly once")
    completed = [row for row in lanes if row.get("passed") is True]
    passed = len(completed) == 3
    paper = {}
    if passed:
        paper = build_campaign_report({"b5_gepa": [Path(row["test_root"]) for row in lanes]})
    method = paper.get("methods", {}).get("b5_gepa", {})
    report = dict(schema_version=1, passed=passed, method="b5_gepa", lanes=lanes,
        completed_seeds=[r["seed"] for r in completed],
        failed_seeds=[r["seed"] for r in lanes if r.get("passed") is not True],
        recovery=receipt, effectiveness=method, test_metrics_mean_std=method.get("mean_std", {}),
        gepa_method_metrics=campaign._gepa_method_summary(completed) if passed else {},
        phase_costs=campaign._phase_cost_summary(completed) if passed else {},
        resource_usage=_resource_summary(completed), cost_accounting=_campaign_cost_summary(completed),
        actual_attempt_usage=campaign._campaign_actual_usage(lanes), completed_at_unix=time.time())
    lines = ["# B5 GEPA recovered three-seed result", "", f"Complete: {passed}", "",
             "| Seed | Official success | Strict success |", "|---|---:|---:|"]
    for lane in completed:
        e = lane["effectiveness"]
        lines.append(f"| {lane['seed']} | {e.get('official_success_rate')} | {e.get('common_strict_success_rate')} |")
    if passed:
        lines += ["", "Three-seed mean / sample standard deviation:", "", "```json",
                  json.dumps(method.get("mean_std", {}), indent=2), "```"]
    lines += ["", "Three-seed statistics: paper_report.json (published only after all seeds pass).",
        "Train/Val, frozen Test, method metrics and full attempt costs: campaign_report.json.",
        "Prior failed-prefix usage is reconstructed from hashed provider sidecars and charged once.",
        "Original runs remain immutable; source migration and checkpoint hashes: recovery_receipt.json.", ""]
    # Only derived reports in this recovery directory may be republished after
    # a report-writing failure; immutable Train/Test artifacts are never edited.
    if passed:
        _write_json_atomic(output / "paper_report.json", paper, overwrite=True)
    _write_json_atomic(output / "campaign_report.json", report, overwrite=True)
    with (output / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    _write_json_atomic(output / ("completion.json" if passed else "campaign_failure.json"),
        dict(passed=passed, report=reference(output / "campaign_report.json")), overwrite=True)
    return report


def run(source, output, smoke, *, prepare_only=False):
    with _method_campaign_lease(REPO, owner="b5_usage_recovery_"+output.name):
        lock_path = source / "campaign_lock.json"
        receipt_path = output / "recovery_receipt.json"
        if output.exists():
            receipt = validate_recovery(receipt_path, lock_path)
        else:
            receipt = prepare(source, output, smoke)
        if prepare_only:
            return dict(passed=True, formal_train_started=False, seed=receipt["seed"], receipt=str(receipt_path))
        old = read(source / "campaign_report.json")
        lock = read(lock_path)
        lane_path = output / f"seed_{receipt['seed']}/lane_report.json"
        if lane_path.exists():
            lane = read(lane_path)
            if lane.get("passed") is not True:
                raise ValueError("Recovery lane failed; preserve it and inspect before another attempt")
        else:
            spec = campaign.GEPACampaignSpec("b5_gepa", campaign.FORMAL_SEEDS,
                *(Path(lock["manifests"][r]["path"]) for r in ("train", "validation", "test")),
                Path(lock["config_path"]), output, Path(lock["worker_python"]), REPO)
            def runner(command, **kwargs):
                return _default_command_runner([*command, "--recovery-receipt", str(receipt_path)], **kwargs)
            lane = campaign._run_lane(spec, seed=receipt["seed"], campaign_lock=lock_path,
                lock_digest=_sha256_file(lock_path), command_runner=runner,
                initial_resume_source=Path(receipt["source_run"]),
                initial_usage_path=Path(receipt["reconstructed_usage"]["path"]))
        validate_recovery(receipt_path, lock_path)
        lanes = sorted([*(row for row in old["lanes"] if row["passed"] is True), lane], key=lambda r:r["seed"])
        for row in lanes:
            if row.get("passed") is True:
                frozen = campaign._validate_train(Path(row["train_root"]), row["seed"])
                campaign._validate_test(Path(row["test_root"]), seed=row["seed"],
                    train_root=Path(row["train_root"]), frozen=frozen)
        return publish(output, lanes, receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-campaign", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--smoke-receipt", required=True, type=Path)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    try:
        result = run(args.source_campaign.resolve(strict=True), args.output_dir.resolve(),
                     args.smoke_receipt.resolve(strict=True), prepare_only=args.prepare_only)
        print(json.dumps({k:v for k,v in result.items() if k in {
            "passed", "formal_train_started", "seed", "receipt", "completed_seeds", "failed_seeds"}}))
        return 0 if result["passed"] else 1
    except Exception as exc:
        from experiments.baselines.common.source_identity import sanitize_error_text
        print(json.dumps(dict(passed=False, error_type=type(exc).__name__, error=sanitize_error_text(exc))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
