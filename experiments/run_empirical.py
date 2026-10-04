"""Fresh empirical Train or read-only evaluation with task-boundary resume."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import yaml

from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.registry import create_simple_harness


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_env(path):
    """Explicit opt-in loading; credentials never enter config or subprocess args."""
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid environment variable name")
        os.environ.setdefault(name, value.strip().strip("\"'"))


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("experiments").rglob("*.py")])
    content = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
    return {"git_sha": sha, "tracked_dirty": bool(dirty), "source_sha256": digest(content)}


def resolve_alfworld_tasks(adapter, entries):
    harness = adapter.harness
    root = Path(harness.alfworld_data).resolve(strict=True)
    discovered = {Path(t.context["game_file"]).resolve().relative_to(root).as_posix(): t
                  for t in harness.load_tasks(limit=max(e['env_index'] for e in entries) + 1)}
    selected, seen = [], set()
    for entry in entries:
        relative = entry["gamefile_rel"]
        task = discovered.get(relative)
        if task is None or relative in seen:
            raise ValueError("Missing or duplicate physical task: " + relative)
        seen.add(relative)
        file_hash = hashlib.sha256(Path(task.context["game_file"]).read_bytes()).hexdigest()
        if file_hash != entry["gamefile_sha256"] or task.task_type != entry["task_type"]:
            raise ValueError("Manifest physical identity mismatch")
        selected.append(PublicTask(task.task_id, digest({"path": relative, "sha256": file_hash}), task.goal,
            {"environment_task": {"task_type": task.task_type, "context": task.context, "metadata": task.metadata}},
            "train" if entry["source_split"] == "train" else entry["source_split"]))
    return selected


def run(config, tasks, output, *, resume=False, readonly=False, adapter=None, adapter_factory=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "empirical.run.v1", "config": config, "tasks": [asdict(t) for t in tasks],
                "code": code_identity(), "readonly": readonly}
    identity_path = output / "execution_manifest.json"
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        if old != identity or not resume:
            raise ValueError("Run identity differs, or existing run needs --resume")
    else:
        if resume:
            raise ValueError("Cannot resume a run without its execution manifest")
        write_json(identity_path, identity)
        write_json(output / "config.json", config)
    system = EmpiricalSystem(config, harness=adapter, readonly=readonly, adapter_factory=adapter_factory)
    state = sqlite3.connect(output / "run.sqlite3")
    state.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,attempts INTEGER,status TEXT,result TEXT)")
    checkpoint = output / "task_checkpoint.sqlite3"
    current = state.execute("SELECT id FROM tasks WHERE status='running'").fetchone()
    if current and not readonly:
        if not checkpoint.is_file():
            raise RuntimeError("Interrupted Train task has no durable Bank checkpoint")
        source = sqlite3.connect(checkpoint)
        source.backup(system.bank.db)
        source.close()
    cases = []
    try:
        for task in tasks:
            row = state.execute("SELECT attempts,status,result FROM tasks WHERE id=?", (task.task_id,)).fetchone()
            if row and row[1] == "completed":
                trace = json.loads(row[2])
                cases.append(trace)
                if not readonly:
                    system.learner.cases.append((task, {"task": {"goal": task.goal, "inputs": task.inputs},
                        "events": trace["tools"], "score": trace["score"], "result": trace["execution"]}))
                continue
            count = (row[0] if row else 0) + 1
            if count > config.get("experiment", {}).get("max_task_attempts", 3):
                raise RuntimeError("Task attempts exhausted")
            if not readonly:
                target = sqlite3.connect(checkpoint)
                system.bank.db.backup(target)
                target.close()
            with state:
                state.execute("INSERT INTO tasks VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET attempts=excluded.attempts,status=excluded.status",
                              (task.task_id, count, "running", None))
            trace_path = output / "traces" / (task.task_id + ".json")
            system.audit_path = output / "requests" / (task.task_id + "_attempt" + str(count) + ".json")
            trace = system.run_task(task, learn=not readonly, attempt_id=task.task_id + ":" + str(count))
            write_json(trace_path, trace)
            with state:
                state.execute("UPDATE tasks SET status='completed',result=? WHERE id=?", (json.dumps(trace), task.task_id))
            if checkpoint.exists():
                checkpoint.unlink()
            cases.append(trace)
            print(json.dumps({"task": task.task_id, "completed": len(cases), "score": trace["score"],
                              "tokens": sum(u["total_tokens"] for u in trace["usage"])}), flush=True)
        frozen = None
        if not readonly:
            destination = output / "frozen_bank"
            frozen = system.bank.freeze(destination) if not destination.exists() else json.loads((destination / "freeze.json").read_text())
            if frozen["digest"] != system.bank.digest():
                raise RuntimeError("Existing freeze differs from completed Train Bank")
        usage = [event for path in (output / 'requests').glob('*.json')
                 for event in json.loads(path.read_text())['usage']]
        summary = {"schema": "empirical.summary.v1", "complete": True, "tasks": len(cases),
                   "successes": sum(bool(t["score"]["hard"]) for t in cases),
                   "total_tokens": sum(u["total_tokens"] for u in usage),
                   "completed_task_tokens": sum(u['total_tokens'] for t in cases for u in t['usage']),
                   "program_invocations": sum(len(t["execution"].get("attempts", [])) for t in cases),
                   "frozen": frozen, "knowledge_digest": system.bank.digest(), "cases": cases}
        write_json(output / "summary.json", summary)
        return summary
    finally:
        state.close()
        system.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alfworld_empirical_seed42.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-bank")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)
    config = yaml.safe_load(Path(args.config).read_text())
    config["data_dir"] = str(Path(args.frozen_bank).resolve()) if args.frozen_bank else str(Path(args.output).resolve() / "bank")
    config["experiment"]["output_dir"] = str(Path(args.output).resolve())
    readonly = bool(args.frozen_bank)
    manifest = json.loads(Path(args.manifest).read_text())
    source_split = manifest["tasks"][0]["source_split"]
    config["harness"]["split"] = {"train": "train", "valid_seen": "eval_in_distribution",
                                    "valid_unseen": "eval_out_of_distribution"}[source_split]
    config["experiment"]["runtime_mode"] = "frozen" if readonly else "online"
    adapter = create_simple_harness(config)
    tasks = resolve_alfworld_tasks(adapter, manifest["tasks"])
    def factory():
        candidate = create_simple_harness(config)
        return candidate
    run(config, tasks, args.output, resume=args.resume, readonly=readonly, adapter=adapter, adapter_factory=factory)


if __name__ == "__main__":
    main()
