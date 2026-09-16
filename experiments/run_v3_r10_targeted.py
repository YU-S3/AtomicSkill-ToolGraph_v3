"""Isolated R10 acceptance against a copied bank; never a formal launcher."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.harness.protocol import HarnessTask
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from experiments.protocol import hash_code, hash_config


def graph_cases(config_path, source_run, output, task_ids, *, online=False):
    start_hash = hash_code(Path(__file__).resolve().parents[1])
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_bank = source_run / "frozen" / "data_v3"
    shutil.copytree(source_bank, output / "bank")
    config = copy.deepcopy(load_config(config_path))
    config.update(data_dir=str(output / "bank"), trace_data_dir=str(output / "trace_store"))
    config["harness"]["split"] = "train"
    config["experiment"].update(output_dir=str(output), phase="r10_targeted",
        initialize_v3_bank="existing", freeze_skills=not online, runtime_mode="online" if online else "frozen",
        task_manifest_path=None)
    config["cold_start"] = {"enabled": False}
    tasks = {}
    for path in (source_run / "traces").glob("trace_*.json"):
        raw = json.loads(path.read_text())
        task = raw.get("task", {})
        if task.get("task_id") in task_ids:
            tasks[task["task_id"]] = HarnessTask(
                task["task_id"], task["goal"], task["benchmark"], task["task_type"],
                dict(task.get("metadata", {})), {})
    if set(tasks) != set(task_ids):
        raise ValueError("targeted tasks must name immutable source Traces")
    rows = []
    with AtomicSkillGraphSystem(config, readonly=not online) as system:
        before = system.knowledge_digest()
        loaded = {task.task_id: task for task in system.harness.load_tasks(
            limit=max(task.context["env_index"] for task in tasks.values()) + 1)}
        for task_id in task_ids:
            task = loaded[task_id]
            if task.goal != tasks[task_id].goal or task.context["game_file"] != tasks[task_id].context["game_file"]:
                raise ValueError("targeted source task identity changed")
            started = time.monotonic()
            trace = system.run_task(task)
            row = {"task_id": task_id, "trace_id": trace.trace_id,
                "benchmark_success": trace.benchmark_success,
                "task_contract_success": trace.task_contract_success,
                "source": trace.runtime_plan.get("source"),
                "selected_composite": trace.planner_audit.get("selected_composite"),
                "p0_metrics": trace.planner_audit.get("p0_metrics", {}),
                "node_statuses": [to_primitive(item.status) for item in trace.node_records],
                "duration_seconds": time.monotonic() - started,
                "r10_metrics": trace.metadata.get("r10_metrics", {})}
            rows.append(row)
            atomic_write_json(output / "progress.json", rows)
            print(json.dumps(row), flush=True)
        unchanged = before == system.knowledge_digest()
    result = {"cases": rows, "frozen_unchanged": unchanged,
              "code_hash": start_hash,
              "code_unchanged": start_hash == hash_code(Path(__file__).resolve().parents[1]),
              "config_hash": hash_config(config),
              "all_tasks_won": all(row["benchmark_success"] for row in rows)}
    atomic_write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alfworld_frozen_eval_134_r10_seed42.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--online", action="store_true", help="Allow ordinary Candidate bootstrap on the isolated copied bank")
    args = parser.parse_args()
    graph_cases(args.config, args.source_run, args.output, args.task_id, online=args.online)


if __name__ == "__main__":
    main()
