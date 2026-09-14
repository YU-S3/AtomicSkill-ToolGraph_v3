"""Fail-closed smoke and resident-load evidence for the frozen B5 campaign."""
import json
from pathlib import Path

from experiments.baselines.run_seed_campaign import _sha256_file, _write_json_atomic


IDENTITY_KEYS = (
    "controller_commit", "controller_code_digest", "external_lock_digest",
    "train_manifest_digest", "validation_manifest_digest", "test_manifest_digest",
    "model_identity", "provider_completion_cap", "budget_policy_version",
    "reasoning_effort", "worker_python",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_memory_report(report, cap, *, allow_rejected=False):
    if report.get("target_workers") != cap or report.get("workers_released") is not True:
        raise ValueError("B5 memory worker count/release authority mismatch")
    if allow_rejected and report.get("memory_rejected") is True and report.get("passed") is False:
        return
    if (report.get("passed") is not True or report.get("loaded_workers") != cap
            or report.get("exact_gamefile_reset") is not True
            or len(report.get("workers", [])) != cap
            or report.get("minimum_mem_available", 0) <= report.get("reserve_bytes", 0)):
        raise ValueError("B5 resident memory/exact reset gate failed")


def guarded_probe_runner(runner):
    def run(command, **kwargs):
        rc = runner(command, **kwargs)
        root = Path(command[command.index("--output-dir") + 1])
        report = read(root / "provider_load_probe.json")
        memory_path = Path(report.get("memory_load_report", "")).resolve()
        memory_path.relative_to(root.parent.resolve())
        memory = read(memory_path)
        cap = int(command[command.index("--concurrency") + 1])
        validate_memory_report(memory, cap, allow_rejected=rc != 0)
        if rc != 0 and memory.get("memory_rejected") is not True:
            raise RuntimeError("B5 provider/protocol failure is not a memory fallback")
        return rc
    return run


def write_load_receipt(spec, payload):
    provider_path = Path(payload["provider_probe"]["report_path"])
    provider = read(provider_path)
    memory_path = Path(provider["memory_load_report"])
    memory = read(memory_path)
    cap = payload["campaign_provider_max_inflight"]
    validate_memory_report(memory, cap)
    if provider.get("passed") is not True:
        raise ValueError("B5 provider gate did not pass")
    receipt = dict(passed=True, seed_lanes=3, selected_global_cap=cap,
                   selected_workers_per_seed=cap // 3,
                   identity={k: payload[k] for k in IDENTITY_KEYS},
                   source_commit=payload["controller_commit"],
                   config_hash=payload["formal_config_digest"],
                   source_config_hash=payload["source_formal_config_digest"],
                   provider_completion_cap=65536, reasoning_effort="high",
                   provider_probe_passed=True, exact_gamefile_reset=True,
                   memory_total=memory["memory_total"], reserve_bytes=memory["reserve_bytes"],
                   minimum_mem_available=memory["minimum_mem_available"],
                   provider_report={"path": str(provider_path), "sha256": _sha256_file(provider_path)},
                   memory_report={"path": str(memory_path), "sha256": _sha256_file(memory_path)})
    path = spec.output_dir / "load_probe_summary.json"
    _write_json_atomic(path, receipt, overwrite=False)
    payload.update(load_probe_receipt_path=str(path), load_probe_receipt_hash=_sha256_file(path))
    verify_load_receipt(payload, spec.output_dir)


def verify_load_receipt(payload, campaign_root):
    path = Path(payload.get("load_probe_receipt_path", "")).resolve()
    path.relative_to(Path(campaign_root).resolve())
    if not path.is_file() or _sha256_file(path) != payload.get("load_probe_receipt_hash"):
        raise ValueError("B5 missing or changed load receipt")
    receipt = read(path)
    cap = receipt.get("selected_global_cap")
    if (receipt.get("passed") is not True or cap not in (48, 36, 24)
            or receipt.get("identity") != {k: payload.get(k) for k in IDENTITY_KEYS}
            or receipt.get("config_hash") != payload.get("formal_config_digest")
            or receipt.get("source_config_hash") != payload.get("source_formal_config_digest")
            or receipt.get("selected_workers_per_seed") != cap // 3
            or receipt.get("seed_lanes") != 3
            or payload.get("campaign_provider_max_inflight") != cap
            or payload.get("seed_lanes") != 3):
        raise ValueError("B5 load receipt identity/config/cap mismatch")
    parallel = payload.get("parallel", {})
    if any(parallel.get(k) != v for k, v in {
        "seed_lanes": 3, "episode_workers_per_seed": cap // 3,
        "test_workers_per_seed": cap // 3, "campaign_provider_max_inflight": cap,
    }.items()):
        raise ValueError("B5 load receipt worker config mismatch")
    for key in ("provider_report", "memory_report"):
        reference = receipt[key]
        evidence = Path(reference["path"]).resolve()
        evidence.relative_to(Path(campaign_root).resolve())
        if _sha256_file(evidence) != reference["sha256"]:
            raise ValueError("B5 load receipt evidence changed")
    memory = read(receipt["memory_report"]["path"])
    validate_memory_report(memory, cap)
    provider = read(receipt["provider_report"]["path"])
    if provider.get("passed") is not True or provider.get("concurrency") != cap:
        raise ValueError("B5 load receipt provider mismatch")
    return receipt


def write_smoke_receipt(ctx):
    from .run_seed_campaign import GEPACampaignSpec, build_campaign_lock, inspect_clean_source
    repo = Path(__file__).resolve().parents[3]
    source = inspect_clean_source(repo)
    manifest = read(ctx.output_dir / "run_manifest.json")
    if manifest["controller_commit"] != source["commit"] or ctx.code_hash != source["code_digest"]:
        raise ValueError("B5 source changed during smoke")
    spec = GEPACampaignSpec("b5_gepa", (42,43,44),
        repo / "data/baseline_manifests/train_120.json",
        repo / "data/baseline_manifests/validation_24.json",
        repo / "data/baseline_manifests/test_ood_full_134.json",
        repo / "configs/baselines/b5_gepa.yaml", ctx.output_dir,
        repo / ".venv_b5_gepa/bin/python", repo)
    payload = build_campaign_lock(spec, ctx.output_dir, "smoke_qualification", source)
    report = read(ctx.output_dir / "smoke_report.json")
    if (report.get("passed") is not True or report["gepa_metrics"].get("reflection_calls", 0) < 1
            or report["gepa_metrics"].get("actual_total_metric_calls") != 24
            or report["frozen"]["digest_before_test"] != report["frozen"]["digest_after_test"]):
        raise ValueError("B5 smoke lacks real optimizer/reflection/frozen evidence")
    evidence = [ctx.output_dir / name for name in ("smoke_report.json", "completion.json", "run_manifest.json")]
    _write_json_atomic(ctx.output_dir / "smoke_qualification.json", dict(
        passed=True, identity={k: payload[k] for k in IDENTITY_KEYS},
        formal_config_digest=payload["formal_config_digest"],
        evidence=[dict(path=str(p), sha256=_sha256_file(p)) for p in evidence]), overwrite=False)


def verify_smoke_receipt(path, payload):
    if path is None:
        raise ValueError("B5 formal requires a current passing --smoke-receipt")
    receipt = read(path)
    if (receipt.get("passed") is not True
            or receipt.get("identity") != {k: payload.get(k) for k in IDENTITY_KEYS}
            or receipt.get("formal_config_digest") != payload.get("source_formal_config_digest", payload["formal_config_digest"])):
        raise ValueError("B5 smoke receipt source/config/model/manifests mismatch")
    for evidence in receipt.get("evidence", []):
        if _sha256_file(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("B5 smoke qualification evidence changed")
    if len(receipt.get("evidence", [])) != 3:
        raise ValueError("B5 smoke qualification evidence missing")
    return {"path": str(Path(path).resolve()), "sha256": _sha256_file(Path(path))}
