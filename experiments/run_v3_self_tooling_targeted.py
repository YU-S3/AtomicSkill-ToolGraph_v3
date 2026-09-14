"""R9.2.1 isolated native automation route acceptance (never formal train)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from experiments.protocol import TaskManifest, hash_code, hash_config, task_signature, validate_deepseek_formal_llm
from experiments.self_tooling_targeted import (
    CORE_TWO, FIXTURE_VERSION, CandidateHarness, FixtureSetupError, RouteCase,
    fixed_draft, parent_atomic, run_node_case, summarize_case, summarize_suite, summarize_usage,
)

REPO = Path(__file__).resolve().parents[1]


def isolated_config(source, root):
    config = copy.deepcopy(source)
    config["data_dir"] = str(root / "bank")
    config["trace_data_dir"] = str(root / "trace_store")
    config["experiment"].update(output_dir=str(root), phase="targeted_smoke",
        task_manifest_path=str(root / "task_manifest.json"), frozen_snapshot_dir=str(root / "unused_frozen"),
        initialize_v3_bank="empty", runtime_mode="online", freeze_skills=False)
    return config


def read_fixture_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("fixture_suite_version") != FIXTURE_VERSION or not data.get("cases"):
        raise FixtureSetupError("missing/unsupported targeted fixture manifest version or cases")
    ids = []
    for entry in data["cases"]:
        case_id = str(entry.get("case_id", ""))
        if not case_id or not all(c.isalnum() or c in "_-" for c in case_id):
            raise FixtureSetupError("fixture case id must be a portable directory name")
        ids.append(case_id)
        manifest = TaskManifest.from_dict(entry["task_manifest"])
        if manifest.split not in {"train", "eval_in_distribution"} or manifest.benchmark != "alfworld":
            raise FixtureSetupError("targeted ALFWorld accepts train/dev only, never Frozen-134")
        if entry.get("prefix"):
            raise FixtureSetupError("this fixture version uses only initial public entries; prefix must be empty")
        if not isinstance(entry.get("env_index"), int) or isinstance(entry["env_index"], bool) or entry["env_index"] < 0:
            raise FixtureSetupError("a nonnegative fixed env_index is required")
        if not isinstance(entry.get("gamefile_sha256"), str) or len(entry["gamefile_sha256"]) != 64:
            raise FixtureSetupError("gamefile_sha256 is required")
    if len(ids) != len(set(ids)):
        raise FixtureSetupError("duplicate fixture case ids")
    return data


def load_alfworld_case(entry, source):
    manifest = TaskManifest.from_dict(entry["task_manifest"])
    harness_config = source.get("harness", {})
    harness = AlfWorldAdapter(split=manifest.split, max_steps=int(harness_config.get("max_steps", 100)),
        alfworld_data=os.environ.get(harness_config.get("alfworld_data_env", "ALFWORLD_DATA")))
    tasks = harness.load_tasks(limit=entry["env_index"] + 1)
    matches = [task for task in tasks if task.context["env_index"] == entry["env_index"]]
    if len(matches) != 1:
        raise FixtureSetupError("fixed source task is unavailable")
    task = matches[0]
    if task.task_id != manifest.task_id or task_signature(task) != manifest.task_signature:
        raise FixtureSetupError("TaskManifest identity differs from actual ALFWorld source task")
    game = Path(task.context["game_file"]).resolve(strict=True)
    data_root = Path(harness.alfworld_data).resolve(strict=True)
    relative = game.relative_to(data_root).as_posix()
    if relative != entry["gamefile_rel"] or hashlib.sha256(game.read_bytes()).hexdigest() != entry["gamefile_sha256"]:
        raise FixtureSetupError("ALFWorld source game path/hash mismatch")
    target = task.context.get("semantic_bindings", {}).get("object")
    if not target:
        raise FixtureSetupError("source task has no public object anchor")
    case = RouteCase(entry["case_id"], route="runtime_seeded", target=target)
    reset = harness.reset(task)
    candidates = [a for a in reset.catalog if a.action_type == "GO_TO"]
    visible_targets = [a for a in reset.catalog if a.action_type == "TAKE" and harness.semantic_value_compatible(
        role="object", concrete_value=a.arguments.get("object"), semantic_anchor=target, semantic_type="entity")]
    if len(candidates) < 2 or visible_targets:
        raise FixtureSetupError("fixed public entry must have at least two candidates and an unresolved target")
    return case, harness, task


def run(config_path, *, mode, output_dir, fixture_suite=None, fixture_manifest=None):
    output = Path(output_dir).expanduser().resolve()
    # Refuse overwrite before loading providers or touching any existing bank.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summaries = []
    expected_count = 0
    incomplete_usage = []
    try:
        source = load_config(config_path)
        validate_deepseek_formal_llm(source)
        key_env = source["llm"]["api_key_env"]
        if not os.environ.get(key_env):
            raise RuntimeError(f"required API environment variable {key_env} is missing")
        if mode == "live-builder":
            if fixture_suite != "core-two" or fixture_manifest:
                raise FixtureSetupError("live-builder requires --fixture-suite core-two")
            fixtures = {"fixture_suite_version": FIXTURE_VERSION, "cases": to_primitive(CORE_TWO)}
            pending = list(CORE_TWO)
        else:
            if not fixture_manifest or fixture_suite:
                raise FixtureSetupError("live-alfworld requires --fixture-manifest")
            fixtures = read_fixture_manifest(fixture_manifest)
            pending = fixtures["cases"]
        expected_count = len(pending)
        manifest = {"phase": "targeted_smoke", "is_formal_experiment": False,
            "fixture_suite_version": FIXTURE_VERSION, "runtime_selection_forced": True,
            "builder_submission_scripted": False, "builder_external_provider": True,
            "environment_kind": "controlled" if mode == "live-builder" else "alfworld",
            "base_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "working_code_fingerprint": hash_code(REPO), "source_config_hash": hash_config(source),
            "fixture_manifest_hash": hash_config(fixtures), "fixture_manifest": fixtures,
            "source_config": source, "mode": mode, "cases": []}
        atomic_write_json(output / "manifest.json", manifest)
        for fixed in pending:
            case_id = fixed.case_id if isinstance(fixed, RouteCase) else fixed["case_id"]
            root = output / case_id
            root.mkdir()
            config = isolated_config(source, root)
            if mode == "live-builder":
                case, harness, task = fixed, CandidateHarness(fixed), None
            else:
                case, harness, task = load_alfworld_case(fixed, source)
            case_manifest = {"case_id": case_id, "effective_config_hash": hash_config(config),
                "effective_config": config, "parent_contract_hash": hash_config(to_primitive(parent_atomic(case))),
                "draft_template_hash": hashlib.sha256(inspect.getsource(fixed_draft).encode()).hexdigest(),
                "scene_seed": 42, "scene_hash": hash_config(to_primitive(case) if task is None else fixed),
                "expected_route": case.route, "test_level": "node_integration"}
            manifest["cases"].append(case_manifest)
            atomic_write_json(output / "manifest.json", manifest)

            def audit(stage, index, kind, payload):
                nonlocal incomplete_usage
                if stage == "internal" and kind == "usage":
                    incomplete_usage = list(payload)
                atomic_write_json(root / f"{stage}_{index:03d}_{kind}.json", payload)

            print(json.dumps({"case_id": case_id, "status": "running", "mode": mode}), flush=True)
            with AtomicSkillGraphSystem(config, harness=harness) as system:
                outcome = run_node_case(system, case, live=True, task=task, audit=audit)
                result = summarize_case(case, outcome)
                atomic_write_json(root / "trace.json", to_primitive(outcome["trace"]))
                atomic_write_json(root / "timeline.json", outcome["timeline"])
                atomic_write_json(root / "case_result.json", result)
            summaries.append(result)
            incomplete_usage = []
            atomic_write_json(output / "summary.json", summarize_suite(summaries, expected_case_count=expected_count))
            print(json.dumps({"case_id": case_id, "passed": result["case_passed"],
                              "stage": result["route_last_stage"]}), flush=True)
        summary = summarize_suite(summaries, expected_case_count=expected_count)
        summary["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(output / "summary.json", summary)
        return 0 if summary["passed"] else 1
    except FixtureSetupError as exc:
        code = 2
        failure = "fixture_setup_failed"
        error = {"error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:
        # CLI infrastructure envelope only; not a negative model/trial result.
        code = 3
        failure = "infrastructure_failed"
        error = {"error_type": type(exc).__name__, "error": str(exc)}
        traceback.print_exc()
    summary = summarize_suite(summaries, expected_case_count=expected_count)
    incomplete_costs = summarize_usage(incomplete_usage)
    summary["incomplete_case_costs_by_provider_and_bucket"] = incomplete_costs
    summary["targeted_external_tokens"] += sum(item["tokens"] for item in incomplete_costs.values() if item["external"])
    summary["fixture_tokens"] += sum(item["tokens"] for item in incomplete_costs.values() if not item["external"])
    summary.update(passed=False, failure_code=failure, exit_code=code,
                   elapsed_seconds=time.monotonic() - started, **error)
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("live-builder", "live-alfworld"), required=True)
    parser.add_argument("--fixture-suite", choices=("core-two",))
    parser.add_argument("--fixture-manifest")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.config, mode=args.mode, output_dir=args.output_dir,
                   fixture_suite=args.fixture_suite, fixture_manifest=args.fixture_manifest)
    except FileExistsError:
        print("output directory already exists; choose a new isolated directory", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
