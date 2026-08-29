"""Static, deterministic, and real-ALFWorld v3 smoke gates."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config

from .protocol import RunManifest, TaskManifest, hash_task_manifest, task_signature
from .report import validate_formal_usage, write_reports


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_configured_task_manifest(
    config: dict[str, object], system: AtomicSkillGraphSystem,
) -> dict[str, object]:
    """Materialize the configured deterministic selection during preflight.

    A schema-only dummy manifest cannot establish that the installed ALFWorld
    dataset contains the requested balanced split.  Formal configurations
    therefore scan their configured selection and bind its concrete identities
    before a paid run is allowed to start.
    """

    harness = dict(config.get("harness") or {})
    selection = dict(harness.get("task_selection") or {})
    if not selection:
        task = TaskManifest(0, "preflight_task", "preflight_signature", "preflight")
        manifest_hash = hash_task_manifest((task,))
        return {
            "task_manifest_schema": bool(manifest_hash),
            "task_manifest_selection": "not_configured",
            "task_manifest_hash": manifest_hash,
        }
    if selection.get("policy") != "balanced_fixed_manifest":
        raise ValueError("preflight requires task_selection.policy=balanced_fixed_manifest")
    task_types = [str(item) for item in selection.get("task_types", [])]
    per_type = int(selection.get("tasks_per_type", 0))
    total = int(selection.get("total_tasks", 0))
    if not task_types or per_type <= 0 or total != len(task_types) * per_type:
        raise ValueError("configured balanced task count is inconsistent")
    if selection.get("require_exact_count") is not True:
        raise ValueError("configured task selection must require exact count")
    tasks = system.harness.load_balanced_tasks(task_types, per_type)
    counts = {label: sum(task.task_type == label for task in tasks) for label in task_types}
    task_ids = [task.task_id for task in tasks]
    signatures = [task_signature(task) for task in tasks]
    if (
        len(tasks) != total
        or any(count != per_type for count in counts.values())
        or len(set(task_ids)) != total
        or len(set(signatures)) != total
    ):
        raise ValueError(
            "configured task manifest is not exact, balanced, and identity-unique: "
            f"total={len(tasks)}, counts={counts}, unique_ids={len(set(task_ids))}, "
            f"unique_signatures={len(set(signatures))}"
        )
    items = tuple(
        TaskManifest.from_task(
            task,
            ordinal=index,
            knowledge_milestone="preflight",
            split=str(system.harness.split),
        )
        for index, task in enumerate(tasks)
    )
    return {
        "task_manifest_schema": True,
        "task_manifest_selection": True,
        "task_manifest_task_count": total,
        "task_manifest_counts": counts,
        "task_manifest_hash": hash_task_manifest(items),
    }


def run_preflight(config_path: str | Path) -> int:
    config = load_config(_path(config_path))
    with tempfile.TemporaryDirectory(prefix="asg_v3_preflight_") as temporary:
        isolated = Path(temporary)
        config["data_dir"] = str(isolated / "data_v3")
        config["trace_data_dir"] = str(isolated / "traces")
        experiment = dict(config.get("experiment") or {})
        experiment.update({
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "initialize_v3_bank": "empty",
        })
        config["experiment"] = experiment
        with AtomicSkillGraphSystem(config, readonly=False) as system:
            checks = system.preflight(require_api_key=True, initialize_harness=True)
            try:
                task_checks = _validate_configured_task_manifest(config, system)
                checks.update(task_checks)
                tasks = (
                    TaskManifest(0, "preflight_task", "preflight_signature", "preflight"),
                )
                RunManifest.create(
                    run_id="preflight", phase="preflight", config_hash="config",
                    code_commit="code", knowledge_digest=system.knowledge_digest(), tasks=tasks,
                )
            except Exception as exc:
                checks["task_manifest_schema"] = False
                checks["task_manifest_selection"] = False
                checks["task_manifest_error"] = str(exc)
            checks["passed"] = bool(
                checks.get("passed")
                and checks["task_manifest_schema"]
                and checks["task_manifest_selection"] in {True, "not_configured"}
            )
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return 0 if checks["passed"] else 1


def run_deterministic() -> int:
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_deterministic_fullchain.py", "tests/test_agent_finalization.py",
        "src/atomic_skillgraph/governance/tests/test_governance.py",
        "experiments/tests/test_protocol_report.py",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    print(json.dumps({
        "passed": True,
        "gate": "deterministic_no_api_fullchain",
        "episodes": 4,
        "coverage": [
            "dynamic_to_evolution", "candidate_direct", "preflight_to_fresh_seeded",
            "task_rescue", "ledger_exactly_once", "token_reconciliation", "frozen_digest",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def run_real_alfworld(config_path: str | Path) -> int:
    config = copy.deepcopy(load_config(_path(config_path)))
    base_output = _path(
        (config.get("experiment") or {}).get("output_dir", "runs/v3_real_smoke")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = base_output / f"run_{stamp}_{os.getpid()}"
    experiment = dict(config.get("experiment") or {})
    experiment.update({
        "name": f"v3_real_smoke_{stamp}",
        "phase": "smoke",
        "condition": "full",
        "runtime_mode": "online",
        "freeze_skills": False,
        "initialize_v3_bank": "empty",
        "output_dir": str(output),
    })
    config["experiment"] = experiment
    config["data_dir"] = str(output / "data_v3")
    with AtomicSkillGraphSystem(config, readonly=False) as system:
        preflight = system.preflight(require_api_key=True, initialize_harness=True)
        if not preflight["passed"]:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 1
        tasks = system.harness.load_balanced_tasks(["pick_and_place_simple"], 5)
        if len(tasks) != 5 or len({task_signature(task) for task in tasks}) != 5:
            raise RuntimeError("real smoke requires five distinct pick-and-place tasks")
        task_items = tuple(
            TaskManifest(
                index,
                task.task_id,
                task_signature(task),
                "cold_learning" if index < 3 else "warm_reuse",
                task.benchmark,
                str(system.harness.split),
                json.dumps({
                    "phase": "cold_learning" if index < 3 else "warm_reuse",
                    "task_type": task.task_type,
                    "env_index": task.context.get("env_index"),
                    "game_file": task.context.get("game_file", ""),
                }, ensure_ascii=False, sort_keys=True),
            )
            for index, task in enumerate(tasks)
        )
        atomic_write_json(output / "task_manifest.json", {
            "schema_version": 3,
            "task_manifest_hash": hash_task_manifest(task_items),
            "tasks": [item.to_dict() for item in task_items],
        })
        cold_traces = [system.run_task(task) for task in tasks[:3]]
        warm_traces = [system.run_task(task) for task in tasks[3:]]
        traces = [*cold_traces, *warm_traces]
        artifact_count = int(system.database.execute(
            "SELECT COUNT(*) AS count FROM artifact_index"
        ).fetchone()["count"])
        learned_preflight = any(
            call.call_kind == "implementation_invocation"
            and call.preflight_result.get("passed") is True
            for trace in traces
            for call in trace.native_tool_calls
        )
        direct_or_completed = any(
            node.status.value in {
                "direct_autonomous_success", "direct_agent_prepared_success",
                "agent_completed_before_invocation",
            }
            for trace in traces for node in trace.node_records
        )
        passed = (
            any(trace.benchmark_success for trace in traces)
            and artifact_count > 0 and learned_preflight and direct_or_completed
            and all(not trace.infrastructure_failure for trace in traces)
        )
        validate_formal_usage(traces)
        write_reports(traces, output / "reports", stem="real_alfworld_smoke")
        result = {
            "passed": passed,
            "output_dir": str(output),
            "tasks": len(traces),
            "cold_tasks": len(cold_traces),
            "warm_unseen_tasks": len(warm_traces),
            "successes": sum(trace.benchmark_success for trace in traces),
            "artifact_count": artifact_count,
            "learned_invocation_preflight": learned_preflight,
            "started_direct_or_agent_completed": direct_or_completed,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--deterministic", action="store_true")
    modes.add_argument("--real-alfworld", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    if args.preflight:
        return run_preflight(args.config)
    if args.deterministic:
        return run_deterministic()
    return run_real_alfworld(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
