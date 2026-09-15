"""Rebuild B4 reports from complete immutable checkpoints, without model calls."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
import os
import shutil
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .campaign import validate_config
from .controller import SeedController
from .manifest_adapter import train_chunks
from .state import read_json, write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.source_identity import hash_code, sanitize_error_text
from experiments.baselines.report_campaign import build_campaign_report


class ReportOnlyController(SeedController):
    def operation(self, *args, **kwargs):
        raise RuntimeError("Report-only recovery cannot execute an operation")

    def _operation_once(self, *args, **kwargs):
        raise RuntimeError("Report-only recovery cannot start a worker")


def load_complete_seed(root, seed, locked):
    """Validate the full planned task sequence before any report is published."""
    lane = root / f"seed_{seed}"
    frozen = FrozenArtifact.load(lane / "frozen")
    manifests = [TaskManifestSet.load(frozen.root / "provenance" / f"{phase}_manifest.json")
                 for phase in ("train", "validation", "test")]
    if [m.digest for m in manifests] != locked["identity"]["manifests"]:
        raise ValueError("Frozen task manifests differ from campaign lock")
    cfg = locked["config"]
    validate_config(cfg)
    if cfg["experiment_kind"] != "formal" or sha256_json(cfg) != locked["resolved_config_hash"]:
        raise ValueError("Report repair requires an intact formal campaign lock")
    data = Path(locked["alfworld_data"])
    for manifest in manifests:
        for task in manifest.tasks:
            if hashlib.sha256((data / task.gamefile_rel).read_bytes()).hexdigest() != task.gamefile_sha256:
                raise ValueError(f"Gamefile changed: {task.task_id}")
    expected = set()
    state_digests = {}
    def checkpoint(operation, phase, task=None):
        expected.add(operation)
        path = lane / "checkpoints" / f"{operation}.json"
        row = read_json(path)  # Missing evidence is fatal; never run a task here.
        result = row["result"]
        if row["operation"] != operation or result["operation"] != operation or result["phase"] != phase:
            raise ValueError(f"Checkpoint identity differs: {operation}")
        if task is not None and result["task"] != task.to_dict():
            raise ValueError(f"Checkpoint names the wrong manifest task: {operation}")
        attempt, state = Path(row["attempt"]).resolve(), Path(row["state"]).resolve()
        attempt.relative_to(lane.resolve())
        state.relative_to(lane.resolve())
        if read_json(attempt / "result.json") != result:
            raise ValueError(f"Checkpoint result differs from attempt evidence: {operation}")
        if state not in state_digests:
            state_digests[state] = digest_directory(state)
        if state_digests[state] != row["state_digest"]:
            raise ValueError(f"Checkpoint state changed: {operation}")
        return row
    train, val, test = manifests
    chunks = train_chunks(train.tasks, seed=seed, epochs=cfg["train"]["num_epochs"],
                          chunk_size=cfg["train"]["train_chunk_size"])
    trains, revisions, validations, history = [], [], [], []
    best, best_rate, best_epoch = None, -1.0, None
    for epoch, tasks in enumerate(chunks):
        trains.extend(checkpoint(f"train_{epoch:02d}_{i:03d}", "train", task) for i, task in enumerate(tasks))
        revision = checkpoint(f"revision_{epoch:02d}", "revision")
        revisions.append(revision)
        rows = [checkpoint(f"validation_{epoch:02d}_{i:03d}", "validation", task)
                for i, task in enumerate(val.tasks)]
        validations.extend(rows)
        rate = sum(row["result"]["official_success"] for row in rows) / len(rows)
        if rate > best_rate:
            best, best_rate, best_epoch = Path(revision["state"]), rate, epoch
        history.append(dict(epoch=epoch, validation_official_rate=rate, best_epoch=best_epoch,
            snapshot_digest=revision["state_digest"], manual_version=revision["result"]["manual_after"]["version"]))
    if history != read_json(lane / "selection.json") or frozen.metadata["best_epoch"] != best_epoch:
        raise ValueError("Saved selection differs from checkpoint validation scores")
    if frozen.metadata["state_digest"] != state_digests[best.resolve()]:
        raise ValueError("Frozen state differs from selected best checkpoint")
    tests = [checkpoint(f"test_{best_epoch:02d}_{i:03d}", "test", task) for i, task in enumerate(test.tasks)]
    if {p.stem for p in (lane / "checkpoints").glob("*.json")} != expected:
        raise ValueError("Unexpected or missing formal checkpoint operations")
    return manifests, (trains, revisions, validations, tests), dict(
        best=best, best_epoch=best_epoch, best_rate=best_rate, history=history)


def report_seed(root_text, recovery_text, seed):
    # This subprocess consumes persisted evidence only, with no API credentials.
    for key in ("MODEL_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        os.environ.pop(key, None)
    root, recovery = Path(root_text), Path(recovery_text)
    try:
        locked = read_json(root / "campaign_lock.json")
        manifests, receipts, selection = load_complete_seed(root, seed, locked)
        # Retain a fully completed lane byte-for-byte; the campaign reporter
        # independently checks its completion and persisted task records.
        lane = root / f"seed_{seed}"
        if (lane / "completion.json").exists() and read_json(lane / "completion.json").get("passed") is True:
            summary = read_json(lane / "summary.json")
            if not summary.get("passed") or summary.get("test_episodes") != len(manifests[2].tasks):
                raise ValueError("Completed lane summary is inconsistent")
            return summary
        spec = copy.deepcopy(locked)
        spec["output"] = str(root)
        spec["report_recovery_progress"] = str(recovery / f"seed_{seed}_progress.json")
        prior = read_json(recovery / "campaign_summary.before.json")
        if prior.get("transport_recovery"):
            spec["transport_recovery"] = prior["transport_recovery"]
        return ReportOnlyController(spec, seed, manifests).finalize_reports(*receipts, **selection)
    except Exception as exc:
        failure = dict(passed=False, seed=seed, error=sanitize_error_text(exc),
            error_type=type(exc).__name__, error_code=getattr(exc, "code", None),
            traceback=sanitize_error_text(traceback.format_exc()))
        write_json(recovery / f"seed_{seed}_failure.json", failure)
        return failure


def protected_hashes(root):
    paths = [root / "campaign_lock.json"]
    for seed in (42, 43, 44):
        lane = root / f"seed_{seed}"
        paths.extend(lane / name for name in ("run_manifest.json", "selection.json"))
        paths.extend((lane / "checkpoints").glob("*.json"))
        paths.extend(p for p in (lane / "frozen").rglob("*") if p.is_file())
        for name in ("result.json", "provider_calls.jsonl", "model_responses.jsonl",
                     "environment_actions.jsonl", "rollout_failure.json"):
            paths.extend((lane / "attempts").glob(f"*/*/{name}"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.campaign.resolve(strict=True)
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    recovery = root / "report_recovery" / stamp
    recovery.mkdir(parents=True)
    for name in ("campaign_summary.json", "REPORT.md"):
        shutil.copy2(root / name, recovery / (Path(name).stem + ".before" + Path(name).suffix))
    before = protected_hashes(root)
    write_json(recovery / "protected_hashes.json", before)
    code_root = Path(__file__).resolve().parents[3]
    code_hash = hash_code(code_root / "experiments/baselines")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(report_seed, str(root), str(recovery), seed) for seed in (42,43,44)]
        lanes = [future.result() for future in futures]
    unchanged = before == protected_hashes(root)
    passed = all(lane["passed"] for lane in lanes) and unchanged and code_hash == hash_code(code_root / "experiments/baselines")
    evidence = dict(passed=passed, mode="offline_checkpoint_replay", model_calls=0,
        original_evidence_unchanged=unchanged, protected_files=len(before),
        report_code_hash=code_hash, report_commit=subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip(),
        duration_seconds=time.monotonic()-started, lanes=[dict(seed=x["seed"], passed=x["passed"],
            error=x.get("error")) for x in lanes])
    if passed:
        paper = build_campaign_report({"b4_embodiskill": [root/f"seed_{seed}" for seed in (42,43,44)]})
        write_json(root / "paper_report.json", paper)
        result = read_json(recovery / "campaign_summary.before.json")
        result.update(passed=True, lanes=lanes, posthoc_report_recovery=evidence)
        rates = [lane["test"]["official_success"]/lane["test"]["tasks"] for lane in lanes]
        result["official_test_success_rate"] = dict(mean=statistics.mean(rates), std=statistics.stdev(rates),
            per_seed=dict(zip((42,43,44), rates)))
        write_json(root / "campaign_summary.json", result)
        report = ["# B4 EmbodiSkill Full120 / Test134", "", "Status: complete; offline report recovery passed.", "",
            "| Seed | Official Test | Strict Test |", "| --- | --- | --- |"]
        for lane in lanes:
            test = lane["test"]
            report.append(f'| {lane["seed"]} | {test["official_success"]}/{test["tasks"]} | {test["contract_consistent_success"]}/{test["tasks"]} |')
        report.extend(["", f"Mean official success: {statistics.mean(rates):.6%}.", "",
            "Reports use the original actions and official won signals. No model requests or learning reruns.",
            "Historical failed provider attempts retain unknown usage and known subtotals; exact total cost remains unavailable.",
            "Original failure report and recovery evidence: " + str(recovery.relative_to(root)), ""])
        (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(recovery / "recovery_report.json", evidence)
    print(json.dumps(evidence, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
