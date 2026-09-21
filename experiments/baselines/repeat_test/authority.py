"""Read-only provenance checks for historical seed42 evaluation sources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml

from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl
from experiments.baselines.common.freeze import FrozenArtifact
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.source_identity import hash_code

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/baselines/seed42_repeats.yaml"
METHODS = ("b0_dynamic", "b1_static_skill", "b3_skillopt", "b4_embodiskill", "b5_gepa")
SKILL_FILES = {"b0_dynamic": None, "b1_static_skill": "initial.md",
               "b3_skillopt": "best_skill.md", "b5_gepa": "best_skill.md"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "experiments", "configs", "src"],
        text=True).strip()
    if status:
        raise ValueError("Commit source changes before qualifying or running repeats")
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    return dict(commit=commit, code_hash=hash_code(REPO), source_config_sha256=file_hash(CONFIG))


def guard_environment():
    # Historical text backends leave temperature unset. Do not silently accept
    # an ambient experiment override in the new launch shell.
    for key in ("OPENAI_COMPATIBLE_TEMPERATURE", "TARGET_OPENAI_COMPATIBLE_TEMPERATURE"):
        if os.environ.get(key, "").strip():
            raise ValueError(f"Remove {key}; repeats must retain original provider-default decoding")


def runtime_receipt(python, source):
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE="1")
    script = ("import sys,json,importlib.metadata as m;"
              "print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,"
              "base_prefix=sys.base_prefix,version=list(sys.version_info[:3]),"
              "packages={p:m.version(p) for p in ['alfworld','textworld','openai']})))")
    result = json.loads(subprocess.check_output([str(python), "-c", script], env=env, text=True))
    if result["prefix"] == result["base_prefix"] or result["version"][:2] != [3, 12]:
        raise ValueError("Original Python3.12 virtual environment is required")
    if result["packages"]["alfworld"] != "0.4.2":
        raise ValueError("ALFWorld runtime differs")
    return result


def audit_source(method):
    if method not in METHODS:
        raise ValueError("Unknown repeat method")
    cfg = yaml.safe_load(CONFIG.read_text())
    if cfg["seed"] != 42 or cfg["repeats"] != [1, 2]:
        raise ValueError("This protocol repeats seed42 twice, not new training seeds")
    entry = cfg["sources"][method]
    test = (REPO / entry["test"]).resolve(strict=True)
    frozen_path = (REPO / entry["frozen"]).resolve(strict=True)
    frozen = FrozenArtifact.load(frozen_path)
    if frozen.method_id != method or frozen.digest != entry["digest"]:
        raise ValueError("Original frozen asset does not match pinned seed42 digest")
    manifest = TaskManifestSet.load(REPO / "data/baseline_manifests/test_ood_full_134.json")
    original = read_json(test / "run_manifest.json")
    report = read_json(test / "test_report.json")
    completion = read_json(test / "completion.json")
    rows = read_jsonl(test / "test/task_rows.jsonl")
    if not report.get("passed") or not completion.get("passed") or len(rows) != 134:
        raise ValueError("Original seed42 Test134 is not complete")
    if original["method"] != method or original.get("seed", original.get("run_seed")) != 42:
        raise ValueError("Original run identity differs")
    by_id = {r["task_id"]: r for r in rows}
    expected_digest = frozen.metadata["state_digest"] if method == "b4_embodiskill" else frozen.digest
    if len(by_id) != 134 or len(manifest.tasks) != 134:
        raise ValueError("Original Test134 has duplicate or missing tasks")
    for task in manifest.tasks:
        row = by_id[task.task_id]
        if (row["gamefile_hash"] != task.gamefile_sha256 or row["run_seed"] != 42
                or row["method"] != method or row["phase"] != "test"
                or row.get("infrastructure_failure")
                or row["artifact_digest_before"] != expected_digest
                or row["artifact_digest_after"] != expected_digest):
            raise ValueError("Original test task/freeze evidence differs")
    b4_spec = None
    if method in {"b0_dynamic", "b1_static_skill"}:
        lock_path = test.parent / "campaign_lock.json"
        lock = read_json(lock_path)
        resolved = lock["config"]
        python = Path(lock["identity"]["python"]["sys_executable"])
        source = Path(lock["source"])
        test_digest = lock["test_digest"]
        expected_packages = lock["identity"]["packages"]
    elif method == "b4_embodiskill":
        lock_path = frozen.root / "provenance/campaign_lock.json"
        b4_spec = read_json(lock_path)
        resolved = b4_spec["config"]
        python, source = Path(b4_spec["worker_python"]), Path(b4_spec["source"])
        test_digest = original["test_manifest_hash"]
        expected_packages = b4_spec["identity"]["dependencies"]["distributions"]
    else:
        lock_path = test / "config_resolved.json"
        resolved = read_json(lock_path)
        python = REPO / resolved["worker_python"]
        source = REPO / ".external/skillopt"
        test_digest = original["identity"]["test_manifest_digest"]
        expected_packages = None
        if report["frozen"]["digest"] != frozen.digest:
            raise ValueError("Original report names a different frozen asset")
    if test_digest != manifest.digest:
        raise ValueError("Repeated Test134 must equal original manifest")
    model = resolved["model"]
    ModelConfig.from_mapping(model).validate_formal_identity()
    cap = 16384 if method == "b3_skillopt" else 65536
    hint = resolved.get("method_output_token_hint", resolved.get("env", {}).get("max_completion_tokens", 16384))
    max_actions = resolved.get("max_environment_actions", resolved.get("env", {}).get("max_steps"))
    if max_actions != 100 or hint != 16384 or model.get("provider_completion_cap", cap) != cap:
        raise ValueError("Historical action/token limits differ")
    if method == "b0_dynamic" and read_json(frozen.root/"empty_descriptor.json").get("persistent_skill") is not None:
        raise ValueError("B0 must remain empty")
    from experiments.baselines.bootstrap_external import verify_runtime_tree
    source_lock = yaml.safe_load((REPO/"experiments/baselines/baseline_lock.yaml").read_text())
    upstream = verify_runtime_tree(source, "embodiskill" if method == "b4_embodiskill" else "skillopt", source_lock)
    runtime = runtime_receipt(python, source)
    if expected_packages and any(runtime["packages"][k] != expected_packages[k] for k in runtime["packages"] if k in expected_packages):
        raise ValueError("Original dependency versions differ")
    embedding = None
    if b4_spec:
        from experiments.baselines.b4_embodiskill.embedding import ensure_embedding_model
        embedding = ensure_embedding_model(REPO, python)
        if embedding != b4_spec["identity"]["embedding"]:
            raise ValueError("Original B4 embedding authority differs")
        if digest_directory(frozen.root/"state") != expected_digest:
            raise ValueError("Original B4 state differs")
    evidence_paths = [test/"run_manifest.json", test/"test_report.json", test/"completion.json",
                      test/"test/task_rows.jsonl", frozen_path/"digest.json", lock_path]
    return dict(method=method, seed=42, original_test=str(test), original_frozen=str(frozen_path),
        frozen_digest=frozen.digest, original_evidence={str(p):file_hash(p) for p in evidence_paths},
        source=str(source), python=str(python.absolute()), runtime=runtime, upstream=upstream,
        config=resolved, model=model, provider_completion_cap=cap, method_output_token_hint=hint,
        max_environment_actions=max_actions, test_manifest_digest=manifest.digest,
        frozen_policy="no_learning_per_episode_disposable_state" if b4_spec else "no_learning_readonly_skill",
        b4_spec=b4_spec, embedding=embedding,
        decoding_policy="upstream_temperature_0.1" if b4_spec else "provider_default_temperature_unset",
        provider_model_revision="not_pinned_by_provider; same model alias only",
        original_result=dict(tasks=134, official=sum(bool(r["official_success"]) for r in rows),
            strict=sum(bool(r.get("common_strict_success",r.get("strict_success"))) for r in rows)),
        scheduling=dict(workers_per_repeat=1, independent_rounds=True, shared_method_lease=False,
                        original_parallel=resolved.get("parallel"), change="execution scheduling only"))


def assert_source_unchanged(authority):
    if FrozenArtifact.load(authority["original_frozen"]).digest != authority["frozen_digest"]:
        raise ValueError("Original source frozen asset changed")
    for path, expected in authority["original_evidence"].items():
        if file_hash(path) != expected:
            raise ValueError("Original test/config evidence changed")


def qualification_identity(audits):
    return dict(code=code_identity(), sources={m:sha256_json(a) for m,a in audits.items()})
