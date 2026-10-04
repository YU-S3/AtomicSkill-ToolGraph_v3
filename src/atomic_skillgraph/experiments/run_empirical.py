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
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.harness.simple_protocol import UnknownSideEffect
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
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
    root = next((p for p in Path(__file__).resolve().parents if (p / '.git').exists()), None)
    package = Path(__file__).resolve().parents[1]
    files = sorted([*root.joinpath('src').rglob('*.py'), *root.joinpath('experiments').rglob('*.py')]) if root else sorted(package.rglob('*.py'))
    base = root or package
    content = {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() if root else None
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip() if root else ''
    return {'git_sha':sha,'tracked_dirty':bool(dirty),'source_sha256':digest(content)}


def resolve_alfworld_tasks(adapter, entries, *, mapping_path=None, canonical_split=None):
    harness = adapter.harness
    root = Path(harness.alfworld_data).resolve(strict=True)
    files = set()
    for entry in entries:
        path = (root / entry['gamefile_rel']).resolve(strict=True)
        if not path.is_relative_to(root) or path.relative_to(root).as_posix() != entry['gamefile_rel']:
            raise ValueError('Manifest physical path must be canonical and within the dataset')
        if str(path) in files:
            raise ValueError('Duplicate physical task: ' + entry['gamefile_rel'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['gamefile_sha256']:
            raise ValueError('Manifest physical file hash mismatch: ' + entry['gamefile_rel'])
        files.add(str(path))
    discovered = {Path(t.context["game_file"]).resolve().relative_to(root).as_posix(): t
                  for t in harness.load_tasks(game_files=files)}
    selected, seen, mapping = [], set(), []
    for entry in entries:
        relative = entry["gamefile_rel"]
        task = discovered.get(relative)
        if task is None or relative in seen:
            raise ValueError("Missing or duplicate physical task: " + relative)
        seen.add(relative)
        file_hash = hashlib.sha256(Path(task.context["game_file"]).read_bytes()).hexdigest()
        if file_hash != entry["gamefile_sha256"] or task.task_type != entry["task_type"]:
            raise ValueError("Manifest physical identity mismatch")
        environment_task = {"task_type": task.task_type, "context": task.context, "metadata": task.metadata}
        if canonical_split:
            environment_task['native_task_id'] = task.task_id
        selected.append(PublicTask(entry['task_id'] if canonical_split else task.task_id,
            digest({"path": relative, "sha256": file_hash}), task.goal, {"environment_task": environment_task},
            canonical_split or ("train" if entry["source_split"] == "train" else entry["source_split"])))
        mapping.append({'source_task_id': entry['task_id'], 'source_env_index': entry['env_index'],
            'runtime_task_id': task.task_id, 'runtime_env_index': task.context['env_index'],
            'gamefile_rel': relative, 'gamefile_sha256': file_hash, 'physical_key': selected[-1].physical_key,
            'source_task_signature': entry.get('task_signature'),
            'runtime_task_signature': task.metadata.get('task_signature')})
    if mapping_path is not None:
        value = {'schema': 'empirical.physical-task-map.v1', 'source_entries_digest': digest(entries), 'tasks': mapping}
        if Path(mapping_path).exists() and json.loads(Path(mapping_path).read_text()) != value:
            raise ValueError('Existing physical task resolution changed')
        write_json(mapping_path, value)
    return selected


def run(config, tasks, output, *, resume=False, readonly=False, adapter=None, adapter_factory=None,
        formal_log=None, task_metadata=None, order_offset=0):
    config = validate_config(config)
    print(json.dumps({'event': 'resolved_config', 'config': config}, ensure_ascii=False), flush=True)
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
    system.observer = formal_log
    system.bank.observer = formal_log
    state = sqlite3.connect(output / "run.sqlite3")
    state.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,attempts INTEGER,status TEXT,result TEXT)")
    cases = []
    try:
        for task_index, task in enumerate(tasks):
            if (output / 'STOP_AFTER_TASK').exists():
                raise SystemExit(75)
            row = state.execute("SELECT attempts,status,result FROM tasks WHERE id=?", (task.task_id,)).fetchone()
            if row and row[1] == "completed":
                trace = json.loads(row[2])
                cases.append(trace)
                if not readonly:
                    system.learner.cases.append((task, {"task": {"goal": task.goal, "inputs": task.inputs},
                        "events": trace["tools"], "score": trace["score"], "result": trace["execution"]}))
                continue
            count = row[0] if row and row[1] == 'running' else (row[0] if row else 0) + 1
            if count > config.get('experiment', {}).get('max_task_attempts', 3):
                raise RuntimeError('Task attempts exhausted')
            checkpoint = TaskCheckpoint(output / 'checkpoints' / task.task_id / str(count))
            if checkpoint.state['stage'] == 'task_started' and (checkpoint.root / 'native_events.json').exists():
                native = json.loads((checkpoint.root/'native_events.json').read_text())
                surface = adapter or system.adapter
                specs = {t['name']: t for t in surface.available_tools()}
                if surface.capabilities.checkpoint_mode != 'workspace_copy' or any(
                    e['state'] != 'finished' or specs.get(e['name'], {}).get('effect') != 'read_only' for e in native):
                    raise UnknownSideEffect('Interrupted environment execution requires explicit new attempt; no blind replay')
            system.checkpoint = checkpoint
            with state:
                state.execute("INSERT INTO tasks VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET attempts=excluded.attempts,status=excluded.status",
                              (task.task_id, count, 'running', None))
            trace_path = output / "traces" / (task.task_id + ".json")
            system.audit_path = output / "requests" / (task.task_id + "_attempt" + str(count) + ".json")
            attempt_id = task.task_id + ":" + str(count)
            if formal_log:
                formal_log.begin_task(task, task_index + order_offset, attempt_id, (task_metadata or {}).get(task.task_id, {}))
            try:
                trace = system.run_task(task, learn=not readonly, attempt_id=attempt_id)
            except Exception as exc:
                if not formal_log or formal_log.task_error(exc):
                    raise
                # A model-authored failure is a result, never a retry-until-win.
                if getattr(exc, 'model_authored', False):
                    trace = {'schema': 'empirical.trace.v1', 'task': asdict(task), 'attempt_id': attempt_id,
                        'score': system.adapter.evaluate(system.adapter.submit(None)),
                        'execution': {'prediction': None, 'reason': 'execution_error', 'attempts': []},
                        'tools': json.loads((checkpoint.root / 'native_events.json').read_text())
                            if (checkpoint.root / 'native_events.json').exists() else [],
                        'usage': [], 'requests': [], 'error': {'code': type(exc).__name__, 'message': str(exc)}}
                    if Path(system.audit_path).exists():
                        trace.update(json.loads(Path(system.audit_path).read_text()))
                    trace['scoring_audit'] = getattr(system.adapter, 'score_audit', {})
                else:
                    raise
            if formal_log:
                formal_log.end_task(trace)
            write_json(trace_path, trace)
            with state:
                state.execute("UPDATE tasks SET status='completed',result=? WHERE id=?", (json.dumps(trace), task.task_id))
            checkpoint.advance('task_committed', trace=trace)
            cases.append(trace)
            print(json.dumps({"task": task.task_id, "completed": len(cases), "score": trace["score"],
                              "tokens": sum(u["total_tokens"] for u in trace["usage"])}), flush=True)
        frozen = None
        if not readonly:
            destination = output / "frozen_bank"
            newly_created = not destination.exists()
            frozen = system.bank.freeze(destination) if newly_created else json.loads((destination / "freeze.json").read_text())
            if formal_log:
                formal_log.freeze(destination, newly_created=newly_created)
            if frozen["source_digest"] != system.bank.digest():
                raise RuntimeError("Existing freeze differs from completed Train Bank")
        audits = [json.loads(path.read_text()) for path in (output / 'requests').glob('*.json')]
        usage = [event for audit in audits for event in audit['usage']]
        unknown_billing = sum(1 for audit in audits for request in audit['requests']
            for attempt in request.get('http_attempts', []) if attempt.get('usage') is None)
        unknown_billing += sum(1 for audit in audits for request in audit['requests']
            if not request.get('response', {}).get('usage') and not request.get('http_attempts'))
        known_tokens = sum(u['total_tokens'] for u in usage)
        summary = {"schema": "empirical.summary.v1", "complete": True, "tasks": len(cases),
                   "successes": sum(bool(t["score"]["hard"]) for t in cases),
                   "total_tokens": None if unknown_billing else known_tokens,
                   'known_total_tokens': known_tokens, 'unknown_billing_attempts': unknown_billing,
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
    config['manifest'] = str(Path(args.manifest).resolve())
    source_split = manifest["tasks"][0]["source_split"]
    config["harness"]["split"] = {"train": "train", "valid_seen": "eval_in_distribution",
                                    "valid_unseen": "eval_out_of_distribution"}[source_split]
    config["experiment"]["runtime_mode"] = "frozen" if readonly else "online"
    adapter = create_simple_harness(config)
    tasks = resolve_alfworld_tasks(adapter, manifest["tasks"], mapping_path=Path(args.output)/'task_identity_resolution.json')
    def factory():
        candidate = create_simple_harness(config)
        return candidate
    run(config, tasks, args.output, resume=args.resume, readonly=readonly, adapter=adapter, adapter_factory=factory)


if __name__ == "__main__":
    main()
