"""B4 v2.1 campaign: three independent seed lanes, serialized learning per lane."""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.baselines.bootstrap_external import (ensure_pinned_source, verify_worker_python,
    verify_worker_environment, worker_expected_distributions, compute_runtime_tree, load_lock)
from experiments.baselines.common.manifest import TaskManifestSet, verify_disjoint, sha256_json
from experiments.baselines.common.formal_validation import verify_formal_manifest
from experiments.baselines.common.source_identity import hash_code, sanitize_error_text
from experiments.baselines.run_seed_campaign import _method_campaign_lease
from experiments.baselines.b4_embodiskill.controller import SeedController, worker_environment, usage
from experiments.baselines.b4_embodiskill.state import write_json, read_json, read_jsonl
from experiments.baselines.common.reasoning_budget import policy_metadata, OUTPUT_HINTS, POLICY_VERSION

REPO = Path(__file__).resolve().parents[3]


def merged(left, right):
    out = copy.deepcopy(left)
    for k, v in right.items():
        out[k] = merged(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(smoke):
    base = yaml.safe_load((REPO / "configs/baselines/common.yaml").read_text())
    full = yaml.safe_load((REPO / "configs/baselines/b4_embodiskill.yaml").read_text())
    cfg = merged(base, full)
    if smoke:
        cfg = merged(cfg, yaml.safe_load((REPO / "configs/baselines/b4_embodiskill_smoke.yaml").read_text()))
    cfg["parallel"].pop("skillopt_analyst_workers_per_seed", None)
    return cfg


def validate_config(cfg):
    expected_transport = dict(sdk_max_retries=0, application_retry_limit=5,
        retry_delays_seconds=[2,5,10,20], completion_budget_authority=POLICY_VERSION)
    if cfg["provider_transport"] != expected_transport or cfg["max_environment_actions"] != 100:
        raise ValueError("Transport/action boundary differs from the frozen protocol")
    expected_method = dict(workflow="team",reasoning="io",successful_topk=1,failed_topk=0,
        skills_topk=5,threshold=0.0,hop=1,llm_concurrency=1,max_trials=30,static_few_shots=1,
        manual_reflection_max_tokens=512,manual_revision_max_tokens=2048,
        manual_sections_refactor_threshold=100,execution_notes_refactor_threshold=50)
    if cfg["embodiskill"] != expected_method or cfg["embedding"] != {"model":"sentence-transformers/all-MiniLM-L6-v2"}:
        raise ValueError("EmbodiSkill method parameters differ from the frozen protocol")
    if cfg["model"]["model"] != "deepseek-v4-flash" or cfg["model"]["reasoning_effort"] != "high":
        raise ValueError("Model authority mismatch")
    if cfg["model"].get("provider_completion_cap") != 65536 or cfg.get("protocol_revision") != "2.2":
        raise ValueError("v2.2 requires a fixed 65536 completion cap")
    if cfg.get("upstream_output_hints") != OUTPUT_HINTS or cfg.get("response_policy") != dict(
        consumable_payload="content_only", reasoning_content_used_by_method=False,
        count_reasoning_tokens_in_cost=True, fail_fast_on_budget_exhaustion=True):
        raise ValueError("Reasoning budget/response policy mismatch")
    if cfg["experiment_kind"] == "formal":
        expected = dict(train=dict(num_epochs=4,train_size=120,train_chunk_size=30),
            selection=dict(validation_size=24,metric="official_won_rate",acceptance="strict_improvement",tie_policy="keep_earlier_best"))
        for section in ("train", "selection"):
            if cfg[section] != expected[section]:
                raise ValueError(f"Formal {section} differs from frozen protocol")
        parallel = cfg["parallel"]
        w = parallel["episode_workers_per_seed"]
        if w != 8 or parallel["seed_lanes"] != 3 or parallel["test_workers_per_seed"] != w or parallel["campaign_provider_max_inflight"] != 24:
            raise ValueError("v2.2 formal parallelism must be 3 x 8, cap 24")


def memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values


def load_probe(spec, task, count):
    """Hold real worker dependencies and game instances, then exercise the shared gate.

    Ramp up without risking OOM. An unsafe projected capacity fails this probe,
    rather than reducing already-locked formal workers or changing the method.
    """
    root = Path(spec["output"]) / "load_probes" / f"workers_{count}"
    root.mkdir(parents=True)
    release = root / "release.json"
    processes, handles, samples, pending = [], [], [], []
    initial = memory()
    reserve = max(2*1024**3, int(initial["MemTotal"]*.15))
    safe, error = True, None
    try:
        for index in range(count):
            before = memory()
            if before["MemAvailable"] < reserve:
                safe, error = False, "available_memory_below_reserve"
                break
            out = root / f"worker_{index:02d}"
            (out / "state").mkdir(parents=True)
            job = {**spec, "output": str(out), "state": str(out / "state"),
                "phase": "load_probe", "epoch": 0, "operation": f"load_probe_{index}",
                "run_id": spec["campaign_id"], "run_seed": 42+index%3,
                "task": task.to_dict(), "release_file": str(release),
                "gate_dir": str(root / "provider_gate"),
                "campaign_id": spec["campaign_id"] + f"_load{count}"}
            write_json(out / "job.json", job)
            log = (out / "worker.log").open("w")
            handles.append(log)
            process = subprocess.Popen([spec["worker_python"], "-m", "experiments.baselines.b4_embodiskill.worker",
                "--job", str(out / "job.json")], cwd=REPO, env=worker_environment(REPO), stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            pending.append((process,out))
            # Measure one real worker first, then overlap imports in bounded
            # groups. Learning episodes are unaffected by this load-only path.
            if index > 0 and (index+1) % 4 and index+1 < count:
                continue
            deadline = time.monotonic()+600
            while any(not (path / "ready.json").exists() for _,path in pending):
                if any(p.poll() is not None for p,_ in pending):
                    raise RuntimeError(f"Method dependency/load probe failed: {root}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Method load timed out: {root}")
                if memory()["MemAvailable"] < reserve:
                    safe, error = False, "available_memory_below_reserve_during_load"
                    break
                time.sleep(.5)
            if not safe:
                break
            pending.clear()
            now = memory()
            samples.append(dict(active=index+1, mem_available=now["MemAvailable"]))
            per_worker = max(1, (initial["MemAvailable"] - now["MemAvailable"]) // (index+1))
            if now["MemAvailable"] - per_worker*(count-index-1) < reserve:
                safe, error = False, "projected_memory_below_reserve"
                break
        write_json(release, dict(release=True))
        for process in processes:
            if process.wait(timeout=600) != 0:
                safe, error = False, "provider_or_worker_failure"
        events = [e for out in root.glob("worker_*") for e in read_jsonl(out / "provider_calls.jsonl")]
        passed = safe and len(processes) == count and len(events) >= count and all(e["status"] == "succeeded" for e in events)
        report = dict(passed=passed, target_workers=count, loaded_workers=len(processes), samples=samples,
            reserve_bytes=reserve, memory_total=initial["MemTotal"], failure=error,
            memory_available_initial=initial["MemAvailable"],
            provider_attempts=len(events), real_alfworld=True, real_embedding=True, real_chroma=True)
        write_json(root / "report.json", report)
        return report
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


def run(args):
    cfg = load_config(args.smoke)
    validate_config(cfg)
    from experiments.baselines.run_method import _controller_git_state
    git_state = _controller_git_state()
    if not args.smoke and not args.load_probe_only and git_state["dirty"]:
        raise RuntimeError("Commit local experiment/config changes before formal execution")
    output = Path(args.output).resolve()
    data = Path(os.environ.get("ALFWORLD_DATA", str(Path.home()/".cache/alfworld"))).resolve()
    source = ensure_pinned_source("embodiskill")
    python = REPO / cfg["worker_python"]
    runtime = verify_worker_python(python, expected_version="3.12")
    deps = verify_worker_environment(python, expected_versions=worker_expected_distributions("embodiskill"),
        forbidden_distributions=("atomic-skillgraph", "skillopt"), forbidden_modules=("atomic_skillgraph", "skillopt"))
    from experiments.baselines.b4_embodiskill.embedding import ensure_embedding_model
    embedding_receipt = ensure_embedding_model(REPO, python)
    names = ("train_6_smoke", "validation_6_smoke", "test_6_smoke") if args.smoke else ("train_120", "validation_24", "test_ood_full_134")
    manifests = [TaskManifestSet.load(REPO / "data/baseline_manifests" / (name+".json")) for name in names]
    verify_disjoint(*manifests)
    receipts = [verify_formal_manifest(m, alfworld_data=data, role=r, profile=cfg["protocol_profile"])
        for m,r in zip(manifests,("train", "validation", "test"))]
    identity = dict(code_hash=hash_code(REPO / "experiments/baselines"),
        config_hash=sha256_json(cfg), manifests=[m.digest for m in manifests],
        upstream_tree=source["runtime_tree"], worker_runtime=runtime, dependencies=deps, embedding=embedding_receipt)
    spec = dict(repo=str(REPO), output=str(output), config=cfg,
        source=source["root"], source_receipt=source, worker_python=str(python), embedding_path=embedding_receipt["path"], git_state=git_state,
        alfworld_data=str(data), campaign_id=output.name, gate_dir=str(output / "provider_gate"),
        provider_cap=cfg["parallel"]["campaign_provider_max_inflight"], identity=identity,
        **policy_metadata(cfg["model"]), upstream_output_hints=cfg["upstream_output_hints"],
        b4_workers_per_seed=cfg["parallel"]["episode_workers_per_seed"],
        b4_seed_lanes=cfg["parallel"]["seed_lanes"],
        b4_campaign_provider_max_inflight=cfg["parallel"]["campaign_provider_max_inflight"])
    if not args.smoke and not args.load_probe_only and not args.resume:
        verify_smoke_qualification(args.smoke_receipt, identity, cfg)
    with _method_campaign_lease(REPO, owner="b4_"+output.name):
        if output.exists():
            if not args.resume:
                raise FileExistsError(f"Use a new output path, or --resume for {output}")
            locked = read_json(output / "campaign_lock.json")
            if locked["identity"] != identity or locked.get("resolved_config_hash") != sha256_json(locked["config"]):
                raise RuntimeError("Resume source/config/manifests/runtime identity changed")
            spec = locked
            cfg = spec["config"]
        else:
            output.mkdir(parents=True)
            write_json(output / "preflight.json", dict(manifests=receipts, identity=identity))
            if args.load_probe_only:
                passed = False
                for workers in (8,):
                    cfg["parallel"].update(episode_workers_per_seed=workers,test_workers_per_seed=workers,
                        campaign_provider_max_inflight=3*workers)
                    spec["provider_cap"] = 3*workers
                    report = load_probe(spec, manifests[1].tasks[0], 3*workers)
                    if report["passed"]:
                        passed=True
                        break
                if not passed:
                    write_json(output / "load_probe_summary.json", dict(passed=False,
                        formal_train_started=False, failure="no_allowed_concurrency_passed",
                        reports=[str(output/"load_probes"/"workers_24"/"report.json")]))
                    raise RuntimeError("No protocol-allowed concurrency passed real memory/provider load smoke; see load_probes")
            if args.load_probe_only:
                report = dict(passed=True, load_probe_only=True, formal_train_started=False,
                              parallel=cfg["parallel"])
                write_json(output / "load_probe_summary.json", report)
                print(json.dumps(report, indent=2))
                return 0
            spec["resolved_config_hash"] = sha256_json(cfg)
            write_json(output / "campaign_lock.json", spec)
        started = time.monotonic()
        seeds = (42,) if args.smoke else (42,43,44)
        def one(seed):
            try:
                return SeedController(spec, seed, manifests).run()
            except Exception as exc:
                seed_root = output / f"seed_{seed}"
                attempts = [e for path in (seed_root / "attempts").rglob("provider_calls.jsonl")
                            for e in read_jsonl(path)]
                failure = dict(passed=False, seed=seed, error=sanitize_error_text(exc),
                    failure_kind=getattr(exc, "failure_kind",
                        "infrastructure_failure" if isinstance(exc, (OSError, TimeoutError)) else "protocol_failure"),
                    all_attempts_usage=usage(attempts))
                write_json(output / f"seed_{seed}" / "campaign_failure.json", failure)
                return failure
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            lanes = list(pool.map(one,seeds))
        passed = all(lane["passed"] for lane in lanes)
        source_after = compute_runtime_tree(Path(spec["source"]), load_lock()["embodiskill"]["runtime_tree"])
        identity_unchanged = (
            hash_code(REPO / "experiments/baselines") == spec["identity"]["code_hash"]
            and source_after == spec["identity"]["upstream_tree"]
            and sha256_json(load_config(args.smoke)) == spec["identity"]["config_hash"]
            and [TaskManifestSet.load(REPO / "data/baseline_manifests" / (name+".json")).digest
                 for name in names] == spec["identity"]["manifests"])
        passed = passed and identity_unchanged
        result = dict(passed=passed, method="b4_embodiskill", experiment_kind=cfg["experiment_kind"],
            campaign_id=spec["campaign_id"], lanes=lanes, makespan_seconds=time.monotonic()-started,
            parallel=cfg["parallel"], formal_train_started=not args.smoke,
            source_config_unchanged=identity_unchanged)
        if passed:
            rates = [sum(row["official_success"] for row in read_jsonl(output/f"seed_{seed}/test/episodes.jsonl"))/len(manifests[2].tasks[:cfg["selection"].get("test_size",len(manifests[2].tasks))]) for seed in seeds]
            result["official_test_success_rate"] = dict(mean=statistics.mean(rates),
                std=statistics.stdev(rates) if len(rates)>1 else None, per_seed=dict(zip(seeds,rates)))
            from experiments.baselines.report_campaign import build_campaign_report
            paper = build_campaign_report({"b4_embodiskill": [output/f"seed_{seed}" for seed in seeds]},
                expected_seeds=seeds, task_types=sorted({t.task_type for t in manifests[2].tasks[:cfg["selection"].get("test_size",len(manifests[2].tasks))]}))
            write_json(output / "paper_report.json", paper)
        from experiments.baselines.common.integrity import assert_no_secrets_on_disk
        assert_no_secrets_on_disk(output, api_key_env="MODEL_API_KEY")
        write_json(output / "campaign_summary.json", result)
        if args.smoke and passed:
            write_json(output / "smoke_qualification.json", dict(passed=True, identity=identity,
                formal_config_hash=sha256_json(load_config(False)),
                **policy_metadata(cfg["model"]), upstream_output_hints=cfg["upstream_output_hints"],
                summary_sha256=sha256_json(result), output=str(output)))
        (output / "REPORT.md").write_text("# EmbodiSkill campaign\n\n"+
            f"Status: {'complete' if passed else 'failed/incomplete'}. Profile: {cfg['experiment_kind']}.\n\n"+
            "See campaign_summary.json, seed_*/{train,validation,test}/task_rows.jsonl and seed_*/attempts/ for auditable results.\n\n"+
            json.dumps(result.get("official_test_success_rate",{}),indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k != "lanes"},indent=2))
        return 0 if passed else 1


def verify_smoke_qualification(path, identity, cfg):
    if not path:
        raise ValueError("Formal requires --smoke-receipt from a passing v2.2 real smoke")
    receipt = read_json(path)
    summary = read_json(Path(path).parent / "campaign_summary.json")
    if not receipt.get("passed") or not summary.get("passed") or receipt.get("summary_sha256") != sha256_json(summary):
        raise ValueError("Smoke qualification is failed or inconsistent")
    for key in ("code_hash", "upstream_tree", "worker_runtime", "dependencies", "embedding"):
        if receipt["identity"].get(key) != identity.get(key):
            raise ValueError(f"Smoke qualification differs from formal {key}")
    if any(receipt.get(k) != v for k,v in policy_metadata(cfg["model"]).items()):
        raise ValueError("Smoke transport policy differs from formal")
    if receipt.get("formal_config_hash") != sha256_json(cfg):
        raise ValueError("Formal configuration changed since smoke qualification")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-receipt")
    p.add_argument("--load-probe-only", action="store_true",
                   help="Only verify formal memory/API concurrency; never start training")
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--method" in tokens:
        i = tokens.index("--method")
        del tokens[i:i+2]
    tokens = [t for t in tokens if not t.startswith("--method=")]
    args = p.parse_args(tokens)
    if args.load_probe_only and (args.smoke or args.resume):
        p.error("--load-probe-only cannot be combined with --smoke or --resume")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
