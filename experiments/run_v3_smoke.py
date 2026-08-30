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

from atomic_skillgraph.agents.provider_probe import (
    ensure_provider_capability,
    run_provider_capability_probe,
)
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config

from .protocol import (
    RunManifest,
    TaskManifest,
    hash_code,
    hash_config,
    hash_task_manifest,
    task_signature,
    validate_deepseek_formal_llm,
)
from .report import (
    validate_formal_usage,
    validate_usage_event_persistence,
    write_reports,
)


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


def run_provider_probe(config_path: str | Path) -> int:
    config_path = _path(config_path)
    config = load_config(config_path)
    validate_deepseek_formal_llm(config)
    output = _path(
        (config.get("experiment") or {}).get(
            "output_dir", "runs/alfworld_train_full_30"
        )
    )
    try:
        manifest = run_provider_capability_probe(
            config,
            output_dir=output,
            config_hash=hash_config(config_path),
            code_hash=hash_code(REPO_ROOT),
        )
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "gate": "deepseek_provider_capability",
            "error_type": type(exc).__name__,
            "error_code": str(getattr(exc, "code", "")),
            "error": str(exc),
            "output_dir": str(output),
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "passed": True,
        "gate": "deepseek_provider_capability",
        "output_dir": str(output),
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))
    return 0


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


def _actual_started_direct(trace: object) -> bool:
    for node in getattr(trace, "node_records", ()):
        status = getattr(getattr(node, "status", ""), "value", getattr(node, "status", ""))
        if status not in {
            "direct_autonomous_success", "direct_agent_prepared_success",
        }:
            continue
        occurrence_id = str(getattr(node, "occurrence_id", ""))
        invocations = [
            item for item in getattr(trace, "implementation_invocations", ())
            if str(getattr(item, "occurrence_id", "")) == occurrence_id
            and dict(getattr(item, "preflight", {}) or {}).get("passed") is True
            and dict(getattr(item, "result", {}) or {}).get("started") is True
            and dict(getattr(item, "result", {}) or {}).get("completed") is True
        ]
        tools = [
            item for item in getattr(trace, "tool_executions", ())
            if str(getattr(item, "occurrence_id", "")) == occurrence_id
            and dict(getattr(item, "result", {}) or {}).get("started") is True
            and dict(getattr(item, "result", {}) or {}).get("completed") is True
        ]
        if invocations and tools:
            return True
    return False


def _validated_dataflow(trace: object) -> bool:
    plan = dict(getattr(trace, "runtime_plan", {}) or {})
    changes = [to_primitive(item) for item in getattr(trace, "binding_changes", ())]
    occurrences = {
        str(item.get("step_id", "")): str(item.get("occurrence_id", ""))
        for item in (plan.get("occurrences") or ())
        if isinstance(item, dict)
    }
    invocations = [
        to_primitive(item)
        for item in getattr(trace, "implementation_invocations", ())
    ]

    # A binding-store write alone is not consumption.  Match one declared
    # edge end-to-end: validator-backed source publication -> target DataFlow
    # binding -> passed and actually-started downstream Implementation whose
    # concrete argument contains that same value.
    consumed_edge = False
    for edge in plan.get("data_edges") or ():
        if not isinstance(edge, dict):
            continue
        source_occurrence = occurrences.get(str(edge.get("source_step", "")), "")
        target_occurrence = occurrences.get(str(edge.get("target_step", "")), "")
        source_role = str(edge.get("source_role", ""))
        target_role = str(edge.get("target_role", ""))
        publications = [
            dict(item.get("current") or {})
            for item in changes
            if item.get("reason") == "validated_output_published"
            and str(item.get("occurrence_id", "")) == source_occurrence
            and str(item.get("role", "")) == source_role
        ]
        for publication in publications:
            value = publication.get("value")
            flowed = any(
                item.get("reason") == "data_flow"
                and str(item.get("occurrence_id", "")) == target_occurrence
                and str(item.get("role", "")) == target_role
                and str(dict(item.get("current") or {}).get("source", "")) == "data_flow"
                and dict(item.get("current") or {}).get("value") == value
                for item in changes
            )
            downstream_started = any(
                str(item.get("occurrence_id", "")) == target_occurrence
                and dict(item.get("preflight") or {}).get("passed") is True
                and dict(item.get("arguments") or {}).get(target_role) == value
                and dict(item.get("result") or {}).get("started") is True
                and dict(item.get("result") or {}).get("completed") is True
                and dict(item.get("result") or {}).get("atomic_effect_passed") is True
                for item in invocations
            )
            if flowed and downstream_started:
                consumed_edge = True
                break
        if consumed_edge:
            break
    return bool(
        len(occurrences) >= 2
        and consumed_edge
        and getattr(trace, "graph_self_sufficient_success", False)
        and not getattr(trace, "task_rescue_required", False)
    )


def run_real_alfworld(config_path: str | Path) -> int:
    config_path = _path(config_path)
    config = copy.deepcopy(load_config(config_path))
    validate_deepseek_formal_llm(config)
    base_output = _path(
        (config.get("experiment") or {}).get("output_dir", "runs/v3_real_smoke")
    )
    capability = ensure_provider_capability(
        config,
        output_dir=base_output,
        config_hash=hash_config(config_path),
        code_hash=hash_code(REPO_ROOT),
        run_if_missing=False,
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
        pick_tasks = system.harness.load_balanced_tasks(["pick_and_place_simple"], 5)
        multi_tasks = system.harness.load_balanced_tasks(
            ["pick_heat_then_place_in_recep"], 1,
        )
        tasks = [*pick_tasks, *multi_tasks]
        if len(tasks) != 6 or len({task_signature(task) for task in tasks}) != 6:
            raise RuntimeError(
                "real smoke requires 3 cold + 2 unseen warm pick-and-place and 1 heat multi-node task"
            )
        task_items = tuple(
            TaskManifest(
                index,
                task.task_id,
                task_signature(task),
                (
                    "cold_learning" if index < 3
                    else "warm_reuse" if index < 5
                    else "multi_node_dataflow"
                ),
                task.benchmark,
                str(system.harness.split),
                json.dumps({
                    "phase": (
                        "cold_learning" if index < 3
                        else "warm_reuse" if index < 5
                        else "multi_node_dataflow"
                    ),
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
        warm_traces = [system.run_task(task) for task in tasks[3:5]]
        multi_traces = [system.run_task(tasks[5])]
        traces = [*cold_traces, *warm_traces, *multi_traces]
        artifact_counts = {
            str(row["artifact_kind"]): int(row["count"])
            for row in system.database.execute(
                "SELECT artifact_kind,COUNT(*) AS count FROM artifact_index GROUP BY artifact_kind"
            ).fetchall()
        }
        four_layer_assets = all(
            artifact_counts.get(kind, 0) > 0
            for kind in ("atomic", "implementation", "tool", "composite")
        )
        actual_started_direct = any(_actual_started_direct(trace) for trace in warm_traces)
        dataflow_proven = any(_validated_dataflow(trace) for trace in traces)
        cold_dynamic_success = any(
            trace.benchmark_success
            and trace.runtime_plan.get("source") == "full_dynamic"
            for trace in cold_traces
        )
        unknown_actions = sum(
            action.action_type == "UNKNOWN"
            for trace in traces for action in trace.environment_actions
        )
        contract_mismatches = sum(
            failure.code in {
                "task_contract_mismatch", "benchmark_goal_contract_mismatch",
            }
            for trace in traces for failure in trace.failures
        )
        passed = (
            capability.get("passed") is True
            and cold_dynamic_success
            and four_layer_assets
            and actual_started_direct
            and dataflow_proven
            and all(not trace.infrastructure_failure for trace in traces)
            and all(trace.resource_usage_complete for trace in traces)
            and unknown_actions == 0
            and contract_mismatches == 0
        )
        persisted_traces = list(system.traces.iter_payloads())
        validate_formal_usage(persisted_traces)
        validate_usage_event_persistence(system.usage.events, persisted_traces)
        write_reports(traces, output / "reports", stem="real_alfworld_smoke")
        result = {
            "passed": passed,
            "output_dir": str(output),
            "tasks": len(traces),
            "cold_tasks": len(cold_traces),
            "warm_unseen_tasks": len(warm_traces),
            "multi_node_tasks": len(multi_traces),
            "successes": sum(trace.benchmark_success for trace in traces),
            "artifact_counts": artifact_counts,
            "four_layer_assets": four_layer_assets,
            "cold_dynamic_success": cold_dynamic_success,
            "actual_started_direct": actual_started_direct,
            "validated_dataflow": dataflow_proven,
            "unknown_alfworld_actions": unknown_actions,
            "won_task_contract_mismatches": contract_mismatches,
            "provider_capability_passed": capability.get("passed") is True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--provider-probe", action="store_true")
    modes.add_argument("--deterministic", action="store_true")
    modes.add_argument("--real-alfworld", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    if args.preflight:
        return run_preflight(args.config)
    if args.provider_probe:
        return run_provider_probe(args.config)
    if args.deterministic:
        return run_deterministic()
    return run_real_alfworld(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
