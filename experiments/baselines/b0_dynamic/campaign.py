"""B0 smoke / formal Test134 × three seeds, with verified task-boundary resume."""
import argparse
import concurrent.futures
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import yaml

from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl, write_json
from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.formal_validation import verify_formal_manifest
from experiments.baselines.common.freeze import assert_frozen_unchanged
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.source_identity import hash_code, sanitize_error_text
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.runtime_python import runtime_python_receipt
from experiments.baselines.common.post_evaluator import TaskRow, summarize_rows, write_rows_jsonl, write_evaluated_episodes_jsonl
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.integrity import assert_no_secrets_on_disk
from experiments.baselines.run_seed_campaign import _method_campaign_lease
from .driver import METHOD, PureDynamicDriver, file_hash, usage_totals

REPO = Path(__file__).resolve().parents[3]


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def lease_root():
    # All Git worktrees share the original repository's formal method lock.
    common = Path(git("rev-parse", "--git-common-dir"))
    return (REPO / common).resolve().parent


def validate_config(cfg):
    ModelConfig.from_mapping(cfg["model"]).validate_formal_identity()
    fixed = dict(method=METHOD, seeds=[42,43,44], max_environment_actions=100,
                 method_output_token_hint=16384)
    if any(cfg.get(k) != v for k,v in fixed.items()):
        raise ValueError("Frozen B0 identity/actions/seeds differ")
    if cfg["model"].get("provider_completion_cap") != 65536:
        raise ValueError("B0 reasoning-aware completion allowance must be 65536")
    if cfg["parallel"] != dict(seed_lanes=3, allowed_global_caps=[48,36,24], mp_start_method="spawn"):
        raise ValueError("B0 frozen concurrency policy differs")
    if cfg["provider_transport"] != dict(sdk_max_retries=0, application_retry_limit=5,
            retry_delays_seconds=[2,5,10,20], deterministic_jitter_ratio=.10):
        raise ValueError("B0 transport retry policy differs")
    if any(k in cfg for k in ("train", "validation", "skill_text", "initial_skill", "bank")):
        raise ValueError("B0 cannot configure training or persistent knowledge")


def source_identity(source):
    from experiments.baselines.bootstrap_external import verify_runtime_tree
    lock = yaml.safe_load((REPO/"experiments/baselines/baseline_lock.yaml").read_text())
    # Source hashing is provenance only; no document contents enter the agent.
    upstream = verify_runtime_tree(source, "skillopt", lock)
    runtime = runtime_python_receipt()
    if not runtime["venv_active"]:
        raise ValueError("B0 must use the configured virtual environment interpreter")
    if git("status", "--porcelain", "--", "src", "experiments", "configs"):
        raise ValueError("Commit source changes before B0 qualification/formal")
    return dict(commit=git("rev-parse", "HEAD"), code_hash=hash_code(REPO),
        upstream=upstream, upstream_commit=lock["skillopt"]["commit"],
        source=str(source), python=runtime,
        packages={p:importlib.metadata.version(p) for p in ("alfworld","textworld","openai")})


def smoke_tasks():
    # Engineering probes only, never part of B0's formal Train/Val costs or state.
    manifest = TaskManifestSet.load(REPO/"data/baseline_manifests/train_120.json")
    verify_formal_manifest(manifest, alfworld_data=os.environ["ALFWORLD_DATA"], role="train", profile="formal_v2")
    selected = {}
    for task in manifest.tasks:
        selected.setdefault(task.task_type, task)
    return list(selected.values())


def verify_receipt(path, identity, cfg):
    receipt = read_json(path)
    if receipt.get("passed") is not True or receipt.get("identity") != identity or receipt.get("config_hash") != sha256_json(cfg):
        raise ValueError("B0 smoke qualification missing/failed or source/config/runtime changed")
    if receipt.get("real_episodes") != 6 or receipt.get("seeds") != [42,43,44] or receipt.get("checks_passed") is not True:
        raise ValueError("B0 smoke lacks complete three-seed six-family checks")
    root = path.parent
    for name, digest in receipt["evidence_hashes"].items():
        target = (root/name).resolve()
        target.relative_to(root.resolve())
        if file_hash(target) != digest:
            raise ValueError("B0 smoke evidence changed")
    return receipt


def preflight(root, source, cfg, campaign_id):
    from .load_probe import loaded_workers
    from experiments.baselines.b3_skillopt.worker import _configure_model
    from experiments.baselines.common.provider_load_probe import run_provider_load_probe
    from experiments.baselines.common.provider_gate import CampaignProviderGate
    sys.path.insert(0, str(source))
    _configure_model(ModelConfig.from_mapping(cfg["model"]), sdk_max_retries=0)
    task = smoke_tasks()[0]
    for cap in cfg["parallel"]["allowed_global_caps"]:
        attempt = root / "preflight" / f"cap_{cap}_{time.time_ns()}"
        gate = CampaignProviderGate(gate_dir=attempt/"gate", campaign_id=campaign_id+"_probe", max_inflight=cap)
        try:
            with loaded_workers(str(source), str(Path(os.environ["ALFWORLD_DATA"])/task.gamefile_rel),
                                cap, attempt/"memory.json"):
                report = run_provider_load_probe(method=METHOD, output_dir=attempt/"provider",
                    campaign_gate=gate, model=cfg["model"]["model"], reasoning_effort="high",
                    run_id=campaign_id, run_seed=42, concurrency=cap, requests=cap*2,
                    max_completion_tokens=65536)
                if not report["passed"]:
                    raise RuntimeError("B0 provider load failed; not a memory fallback")
            return dict(passed=True, selected_cap=cap, workers_per_seed=cap//3,
                memory_report=str(attempt/"memory.json"), provider_report=str(attempt/"provider/provider_load_probe.json"),
                memory_sha256=file_hash(attempt/"memory.json"),
                provider_sha256=file_hash(attempt/"provider/provider_load_probe.json"))
        except MemoryError:
            continue
    raise RuntimeError("B0 cannot pass even global24 real memory load")


def load_record(path, task, seed, digest, phase):
    completion = read_json(path/"completion.json")
    if completion.get("passed") is not True or completion["record_hash"] != file_hash(path/"record.json") or completion["evidence_digest"] != digest_directory(path/"attempts"):
        raise ValueError("Episode record/evidence no longer matches completion")
    record = CommonEpisodeRecord(**read_json(path/"record.json"))
    if (record.task_id, record.manifest_index, record.run_seed, record.phase, record.gamefile_hash,
            record.artifact_digest_before, record.artifact_digest_after) != (
            task.task_id, task.index, seed, phase, task.gamefile_sha256, digest, digest):
        raise ValueError("Completed B0 task identity/freeze mismatch")
    return record


def lane_run(spec, seed, tasks, stop=None):
    lane = Path(spec["output"])/f"seed_{seed}"
    driver = PureDynamicDriver()
    frozen = driver.freeze(lane)
    write_json(lane/"run_manifest.json", dict(method=METHOD, seed=seed, identity=spec["identity"],
        model=spec["config"]["model"], train_manifest_hash=None, validation_manifest_hash=None,
        test_manifest_hash=spec.get("test_digest"), phase=spec["phase"],
        config_hash=sha256_json(spec["config"]), frozen_digest=frozen.digest))
    pending = []
    for task in tasks:
        path = lane/spec["phase"]/"episodes"/f"task_{task.index:04d}"
        if (path/"completion.json").exists():
            load_record(path, task, seed, frozen.digest, spec["phase"])
        else:
            pending.append(dict(episode_dir=str(path), task=task.to_dict(), frozen=str(lane/"frozen"),
                config=spec["config"], source=spec["source"], seed=seed, phase=spec["phase"],
                alfworld_data=spec["alfworld_data"], gate=spec["gate"], cap=spec["cap"], campaign_id=spec["campaign_id"]))
    failures = []
    # Each process executes one episode, then exits; a rolling window stops new
    # dispatch after an infrastructure error while draining already started work.
    with concurrent.futures.ProcessPoolExecutor(max_workers=spec["cap"]//3,
            mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        iterator = iter(pending)
        active = {}
        def dispatch():
            if stop is not None and stop.is_set():
                return
            job = next(iterator, None)
            if job is not None:
                active[pool.submit(driver.evaluate, job)] = job
        for _ in range(spec["cap"]//3):
            dispatch()
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(passed=False, error=sanitize_error_text(exc), task_id=job["task"]["task_id"])
                if not result["passed"]:
                    failures.append(result)
                    if stop is not None:
                        stop.set()
                print(f"B0 seed={seed} task={job['task']['task_id']} passed={result['passed']}", flush=True)
            if not failures:
                for _ in done:
                    dispatch()
    if failures:
        return dict(passed=False, seed=seed, failures=failures)
    if stop is not None and stop.is_set():
        return dict(passed=False, seed=seed, error="Campaign stopped at durable boundary after another seed failed")
    records = [load_record(lane/spec["phase"]/"episodes"/f"task_{task.index:04d}", task, seed,
                           frozen.digest, spec["phase"]) for task in tasks]
    rows = [TaskRow.from_episode(r) for r in records]
    target = lane/spec["phase"]
    evaluated = target/"evaluated_common_episodes.jsonl"
    if not evaluated.exists():
        write_evaluated_episodes_jsonl(records, evaluated)
    write_rows_jsonl(rows, target/"task_rows.jsonl")
    events = [e for path in sorted((target/"episodes").glob("*/attempts/*/provider_calls.jsonl")) for e in read_jsonl(path)]
    write_rows_jsonl(rows, target/"results.jsonl")
    # Rollup retains failed attempts too, with unique logical/physical ids.
    from experiments.baselines.common.post_evaluator import _write_jsonl_atomic
    _write_jsonl_atomic(target/"provider_calls.jsonl", events, overwrite=True)
    summary = summarize_rows(rows, task_types=list(dict.fromkeys(t.task_type for t in tasks)))
    assert_frozen_unchanged(frozen)
    report = dict(passed=True, method=METHOD, phase=spec["phase"], seed=seed, run_id=spec["campaign_id"],
        test=summary, training_cost=driver.train(), test_cost=dict(api_cost=None, api_cost_unpriced=True,
        usage=usage_totals(events),
        all_attempt_environment_actions=sum(r.method_metrics.get("all_attempt_environment_actions",r.environment_actions) for r in records),
        provider_service_latency_ms=sum(e.get("provider_service_latency_ms",0) for e in events),
        failed_logical_calls=sum(e.get("status")!="succeeded" for e in events)), frozen_unchanged=True)
    write_json(lane/"summary.json", report)
    write_json(lane/f"{spec['phase']}_report.json", report)
    write_json(lane/"completion.json", dict(passed=True, phase=spec["phase"], run_id=spec["campaign_id"],
                                           report=f"{spec['phase']}_report.json"))
    return report


def execute(spec, tasks):
    stop = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lane_run, spec, seed,
            tasks[i*2:(i+1)*2] if spec["phase"]=="smoke" else tasks, stop) for i,seed in enumerate((42,43,44))]
        results = []
        for seed,future in zip((42,43,44), futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(dict(passed=False, seed=seed, error=sanitize_error_text(exc)))
    return results


def smoke_checks(root):
    events = [e for p in root.glob("seed_*/smoke/provider_calls.jsonl") for e in read_jsonl(p)]
    physical = [p for e in events for p in e.get("physical_attempt_usage", [])]
    responses = [r for p in root.glob("seed_*/smoke/episodes/*/attempts/*/model_responses.jsonl") for r in read_jsonl(p)]
    records = [read_json(p) for p in root.glob("seed_*/smoke/episodes/*/record.json")]
    checks = dict(six_episodes=len(records)==6,
        six_families=len({r["task_type"] for r in records})==6,
        three_seeds={r["run_seed"] for r in records}=={42,43,44},
        target_only=bool(events) and all(e["role"]=="target" for e in events),
        no_skill_section=bool(responses) and all("## Skill Knowledge" not in str(r["messages"]) for r in responses),
        frozen_empty=all(r["method_metrics"]["skill_text"] is None and r["evolution_llm_calls"]==0 for r in records),
        reasoning_budget=bool(physical) and all(p.get("provider_completion_cap")==65536 and not p.get("budget_exhausted") for p in physical),
        usage_complete=bool(physical) and all(isinstance(p.get(k),int) for p in physical for k in ("prompt_tokens","completion_tokens")))
    write_json(root/"smoke_checks.json", checks)
    return all(checks.values())


def run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_config(cfg)
    source = Path(args.skillopt_root).resolve(strict=True)
    identity = source_identity(source)
    ModelConfig.from_mapping(cfg["model"]).require_api_key()
    data = str(Path(os.environ["ALFWORLD_DATA"]).resolve(strict=True))
    root = Path(args.output).absolute()
    is_smoke = args.mode == "smoke"
    if is_smoke:
        tasks = smoke_tasks()
        test_digest = None
    else:
        manifest = TaskManifestSet.load(REPO/"data/baseline_manifests/test_ood_full_134.json")
        verify_formal_manifest(manifest, alfworld_data=data, role="test", profile="formal_v2")
        tasks, test_digest = list(manifest.tasks), manifest.digest
    from contextlib import nullcontext
    lease = nullcontext() if is_smoke else _method_campaign_lease(lease_root(), owner="b0_"+root.name)
    with lease:
        if args.mode == "resume":
            spec = read_json(root/"campaign_lock.json")
            if spec["identity"]!=identity or spec["config"]!=cfg or spec["test_digest"]!=test_digest or spec["alfworld_data"]!=data:
                raise ValueError("B0 resume source/config/runtime/data identity changed")
            if spec["output"] != str(root):
                raise ValueError("Resume must use original campaign output path")
            verify_receipt(Path(spec["smoke_qualification"]), identity, cfg)
            load = spec["load_probe"]
            if spec["phase"]!="test" or not load["passed"] or spec["cap"]!=load["selected_cap"] or spec["cap"] not in (48,36,24):
                raise ValueError("B0 resume must retain qualified formal concurrency")
            for label in ("memory", "provider"):
                if file_hash(load[label+"_report"]) != load[label+"_sha256"]:
                    raise ValueError("B0 load qualification changed")
        else:
            if not is_smoke:
                if not args.qualification:
                    raise ValueError("B0 formal requires a passing --qualification")
                verify_receipt(Path(args.qualification).absolute(), identity, cfg)
            root.mkdir(parents=True, exist_ok=False)
            campaign_id = root.name
            load = None if is_smoke else preflight(root, source, cfg, campaign_id)
            spec = dict(identity=identity, config=cfg, output=str(root), source=str(source), alfworld_data=data,
                campaign_id=campaign_id, phase="smoke" if is_smoke else "test", test_digest=test_digest,
                cap=3 if is_smoke else load["selected_cap"], gate=str(root/"provider_gate"), load_probe=load,
                smoke_qualification=None if is_smoke else str(Path(args.qualification).absolute()))
            write_json(root/"campaign_lock.json", spec)
            write_json(root/"config_resolved.json", cfg)
            write_json(root/"task_manifest.json", dict(tasks=[t.to_dict() for t in tasks], digest=test_digest))
        started = time.monotonic()
        results = execute(spec, tasks)
        if source_identity(source)!=identity:
            raise ValueError("Source/runtime changed during B0")
        passed = all(r["passed"] for r in results)
        if passed and is_smoke:
            passed = smoke_checks(root)
        summary = dict(passed=passed, method=METHOD, lanes=results,
            duration_seconds=time.monotonic()-started, phase=spec["phase"], test_manifest_digest=test_digest)
        write_json(root/f"campaign_summary_{time.time_ns()}.json", summary)
        write_json(root/"campaign_summary.json", summary)
        assert_no_secrets_on_disk(root, api_key_env=cfg["model"]["api_key_env"])
        if passed and is_smoke:
            # Persist qualifications only when real six-family, three-seed
            # episodes, provider evidence and frozen replay all completed.
            evidence = {p.relative_to(root).as_posix():file_hash(p) for p in root.rglob("*.json")}
            evidence.update({p.relative_to(root).as_posix():file_hash(p) for p in root.rglob("*.jsonl")})
            write_json(root/"smoke_qualification.json", dict(passed=True, identity=identity,
                config_hash=sha256_json(cfg), families=[t.task_type for t in tasks], seeds=[42,43,44],
                real_episodes=6, skill_text=None, checks_passed=True, evidence_hashes=evidence))
        elif passed:
            from experiments.baselines.report_campaign import build_campaign_report
            paper = build_campaign_report({METHOD:[root/f"seed_{s}" for s in (42,43,44)]}, b0_method=METHOD)
            write_json(root/"paper_report.json", paper)
            lines = ["# B0 Pure Dynamic — Test134 × 3 seeds", "", "Status: complete", "",
                     "No training, validation selection, skill text or persistent experience.", "",
                     "| Seed | Official | Strict |", "| --- | --- | --- |"]
            for r in results:
                s=r["test"]
                lines.append(f'| {r["seed"]} | {s["official_success"]}/{s["tasks"]} | {s["common_strict_success"]}/{s["tasks"]} |')
            lines += ["", f'Mean official success: {statistics.mean(r["test"]["official_success"]/r["test"]["tasks"] for r in results):.4%}',
                      "", "API cost is unpriced. Unknown billed usage remains null with known subtotals."]
            (root/"REPORT.md").write_text("\n".join(lines)+"\n")
        print(f"B0 {args.mode}: passed={passed}; output={root}", flush=True)
        return 0 if passed else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke","formal","resume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--skillopt-root", required=True)
    parser.add_argument("--config", default=str(REPO/"configs/baselines/b0_dynamic.yaml"))
    args=parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        if Path(args.output).is_dir():
            write_json(Path(args.output)/f"controller_failure_{time.time_ns()}.json", dict(passed=False,
                error_type=type(exc).__name__, error=sanitize_error_text(exc)))
        print(f"B0 stopped: {type(exc).__name__}: {sanitize_error_text(exc)}", file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
