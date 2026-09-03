"""Formal read-only ALFWorld held-out 6×10=60 frozen evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config

from .protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    AttemptTraceLedger,
    ManifestStore,
    ProtocolError,
    RunManifest,
    RunState,
    TaskManifest,
    audit_failed_attempt,
    artifact_audit_snapshot,
    artifact_growth_audit,
    ensure_task_manifest,
    hash_code,
    hash_config,
    hash_knowledge,
    load_task_report_traces,
    task_signature,
    validate_deepseek_formal_llm,
    validate_distinct_formal_tasks,
)
from .report import (
    validate_frozen_v31_guards,
    validate_formal_usage,
    validate_usage_event_persistence,
    write_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _selection(config: dict[str, Any]) -> tuple[list[str], int, int]:
    selection = dict((config.get("harness") or {}).get("task_selection") or {})
    labels = [str(item) for item in selection.get("task_types", [])]
    per_type = int(selection.get("tasks_per_type", 0))
    total = int(selection.get("total_tasks", 0))
    if tuple(labels) != ALFWORLD_FORMAL_TASK_TYPES:
        raise ProtocolError(
            "formal frozen eval requires the six ALFWorld task types in frozen order"
        )
    if per_type != 10 or total != 60:
        raise ProtocolError("formal frozen eval requires six task types × ten = 60")
    if total != len(labels) * per_type or selection.get("require_exact_count") is not True:
        raise ProtocolError("formal frozen selection must require the exact balanced count")
    if selection.get("require_disjoint_from_train_manifest") is not True:
        raise ProtocolError("frozen held-out manifest must be disjoint from train")
    return labels, per_type, total


def _validate_formal_config(config: dict[str, Any], output_dir: Path) -> None:
    validate_deepseek_formal_llm(config)
    experiment = dict(config.get("experiment") or {})
    harness = dict(config.get("harness") or {})
    selection = dict(harness.get("task_selection") or {})
    planner = dict(config.get("planner") or {})
    cold_start = dict(config.get("cold_start") or {})
    expected = {
        "method_patch": (config.get("method_patch"), "3.2"),
        "planner.max_repeat_count": (planner.get("max_repeat_count"), 4),
        "planner.max_runtime_occurrences": (
            planner.get("max_runtime_occurrences"), 16
        ),
        "planner.cold_start_c1_repair_limit": (
            planner.get("cold_start_c1_repair_limit"), 1
        ),
        "cold_start": (cold_start, {"enabled": False}),
        "experiment.name": (experiment.get("name"), "alfworld_frozen_eval_60"),
        "experiment.condition": (experiment.get("condition"), "full"),
        "experiment.freeze_skills": (experiment.get("freeze_skills"), True),
        "experiment.seed": (experiment.get("seed"), 42),
        "experiment.require_knowledge_digest_unchanged": (
            experiment.get("require_knowledge_digest_unchanged"), True
        ),
        "experiment.allow_eval_traces_and_metrics": (
            experiment.get("allow_eval_traces_and_metrics"), True
        ),
        "experiment.allow_long_term_knowledge_writes": (
            experiment.get("allow_long_term_knowledge_writes"), False
        ),
        "experiment.resume_completed_task_boundary_only": (
            experiment.get("resume_completed_task_boundary_only"), True
        ),
        "harness.adapter": (harness.get("adapter"), "alfworld_v3"),
        "harness.alfworld_data_env": (harness.get("alfworld_data_env"), "ALFWORLD_DATA"),
        "harness.split": (harness.get("split"), "eval_out_of_distribution"),
        "harness.max_steps": (harness.get("max_steps"), 100),
        "harness.task_selection.policy": (
            selection.get("policy"), "balanced_fixed_manifest"
        ),
    }
    mismatches = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    frozen_dir = _path(experiment.get("source_frozen_snapshot_dir", ""))
    if _path(config.get("data_dir", "")) != frozen_dir:
        mismatches.append("data_dir must equal source_frozen_snapshot_dir")
    if _path(config.get("trace_data_dir", output_dir)) != output_dir:
        mismatches.append("trace_data_dir must equal experiment.output_dir")
    if output_dir.name != str(experiment.get("name", "")):
        mismatches.append("output_dir basename must equal experiment.name")
    if _path(experiment.get("task_manifest_path", "")) != output_dir / "task_manifest.json":
        mismatches.append("task_manifest_path must be <output_dir>/task_manifest.json")
    train_dir = _path(experiment.get("source_train_run_dir", ""))
    if frozen_dir != train_dir / "frozen" / "data_v3":
        mismatches.append("source_frozen_snapshot_dir must be <source_train_run_dir>/frozen/data_v3")
    if output_dir == frozen_dir or frozen_dir in output_dir.parents:
        mismatches.append("eval output_dir must be outside the read-only frozen snapshot")
    max_task_attempts = experiment.get("max_task_attempts")
    if (
        isinstance(max_task_attempts, bool)
        or not isinstance(max_task_attempts, int)
        or max_task_attempts <= 0
    ):
        mismatches.append("experiment.max_task_attempts must be a positive integer")
    if mismatches:
        raise ProtocolError("formal frozen config mismatch: " + "; ".join(mismatches))


def _verify_source_train(
    *,
    train_run_dir: Path,
    train_manifest: RunManifest,
    freeze_manifest: dict[str, Any],
    frozen_digest: str,
    current_code_digest: str,
    current_llm_hash: str,
) -> None:
    """Bind one frozen bank to the completed immutable full-30 source run."""
    if train_manifest.phase != "train":
        raise ProtocolError("source manifest is not a train run")
    if train_run_dir.name != train_manifest.run_id:
        raise ProtocolError("source_train_run_dir basename differs from source run_id")
    metadata = train_manifest.metadata
    if (
        metadata.get("condition") != "full"
        or int(metadata.get("tasks_per_type", 0)) != 5
        or int(metadata.get("total_tasks", 0)) != 30
        or len(train_manifest.tasks) != 30
    ):
        raise ProtocolError("source manifest is not the formal full 6×5 train run")
    if len({item.task_signature for item in train_manifest.tasks}) != 30:
        raise ProtocolError("source train manifest contains duplicate task signatures")
    expected_provenance = {
        "source_run_id": train_manifest.run_id,
        "source_run_manifest_hash": train_manifest.manifest_hash,
        "source_config_hash": train_manifest.config_hash,
        "source_code_commit": train_manifest.code_commit,
        "source_task_manifest_hash": train_manifest.task_manifest_hash,
        "source_initial_knowledge_digest": train_manifest.knowledge_digest,
        "source_final_knowledge_digest": frozen_digest,
        "source_llm_config_hash": str(metadata.get("llm_config_hash", "")),
    }
    if freeze_manifest.get("provenance") != expected_provenance:
        raise ProtocolError("frozen snapshot provenance does not match source train manifest")
    if train_manifest.code_commit != current_code_digest:
        raise ProtocolError("frozen evaluation code differs from the source train code")
    if str(metadata.get("llm_config_hash", "")) != current_llm_hash:
        raise ProtocolError("frozen evaluation LLM configuration differs from source train")

    source_data_dir = train_run_dir / "data_v3"
    source_database_path = source_data_dir / "state.sqlite3"
    if not source_database_path.is_file():
        raise ProtocolError(f"source train state database missing: {source_database_path}")
    with StateDatabase(source_database_path, readonly=True) as source_database:
        run_row = source_database.execute(
            "SELECT * FROM run_manifests WHERE run_id=?", (train_manifest.run_id,)
        ).fetchone()
        if run_row is None or str(run_row["state"]) != RunState.COMPLETED.value:
            raise ProtocolError("source train run is not durably completed")
        run_identity = {
            "phase": train_manifest.phase,
            "config_hash": train_manifest.config_hash,
            "task_manifest_hash": train_manifest.task_manifest_hash,
            "code_commit": train_manifest.code_commit,
        }
        if any(str(run_row[key]) != value for key, value in run_identity.items()):
            raise ProtocolError("source train run ledger identity differs from immutable manifest")
        rows = source_database.rows(
            "SELECT * FROM run_tasks "
            "WHERE run_id=? ORDER BY rowid",
            (train_manifest.run_id,),
        )
        if len(rows) != len(train_manifest.tasks) or any(
            str(row["state"]) != "completed" or not str(row["trace_id"])
            for row in rows
        ):
            raise ProtocolError("source train task ledger is incomplete")
        expected_before = train_manifest.knowledge_digest
        for task, row in zip(train_manifest.tasks, rows):
            identity = {
                "task_id": task.task_id,
                "task_signature": task.task_signature,
                "config_hash": train_manifest.config_hash,
                "code_commit": train_manifest.code_commit,
                "knowledge_milestone": task.knowledge_milestone,
            }
            if any(str(row[key]) != value for key, value in identity.items()):
                raise ProtocolError(
                    f"source train task ledger identity mismatch at {task.task_id}"
                )
            result = json.loads(str(row["result_json"]))
            if str(result.get("knowledge_digest_before", "")) != expected_before:
                raise ProtocolError(
                    f"source train digest chain is broken before {task.task_id}"
                )
            expected_before = str(result.get("knowledge_digest_after", ""))
            if not expected_before:
                raise ProtocolError(f"source train task {task.task_id} lacks final digest")
            trace_path = train_run_dir / "traces" / f"{row['trace_id']}.json"
            if not trace_path.is_file():
                raise ProtocolError(f"source train trace missing: {trace_path}")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_task = dict(trace.get("task") or {})
            if (
                str(trace.get("trace_id", "")) != str(row["trace_id"])
                or str(trace_task.get("task_id", "")) != task.task_id
                or str(trace_task.get("task_signature", "")) != task.task_signature
            ):
                raise ProtocolError(f"source train trace identity mismatch: {trace_path}")
        if expected_before != frozen_digest:
            raise ProtocolError("source train final task milestone differs from frozen bank")
        if hash_knowledge(source_data_dir, database=source_database) != frozen_digest:
            raise ProtocolError("source train live bank differs from frozen bank")


def run(config_path: str | Path, *, resume: bool = False) -> int:
    config_path = _path(config_path)
    config = load_config(config_path)
    experiment = dict(config.get("experiment") or {})
    if experiment.get("phase") != "frozen_eval" or experiment.get("runtime_mode") != "frozen":
        raise ProtocolError("frozen runner requires phase=frozen_eval/runtime_mode=frozen")
    labels, per_type, expected_total = _selection(config)
    output_dir = _path(experiment.get("output_dir", "runs/alfworld_frozen_eval_60"))
    _validate_formal_config(config, output_dir)
    max_task_attempts = int(experiment["max_task_attempts"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(experiment.get("name", "alfworld_frozen_eval_60"))
    attempt_ledger = AttemptTraceLedger(
        output_dir / "attempt_history", output_dir / "traces",
    )
    if not resume and attempt_ledger.root.exists():
        raise FileExistsError(attempt_ledger.root)
    recovered_attempts = attempt_ledger.recover_pending(run_id=run_id) if resume else []
    if recovered_attempts:
        print(json.dumps({
            "recovered_attempt_trace_captures": recovered_attempts,
        }, ensure_ascii=False), flush=True)
    unresolved_attempts = attempt_ledger.unresolved(run_id=run_id)
    if unresolved_attempts:
        raise ProtocolError(
            "resume found an attempt with no durable Trace; provider usage is unproven. "
            "Archive this eval run and start fresh without --resume: "
            + ", ".join(str(item["attempt_id"]) for item in unresolved_attempts)
        )

    with AtomicSkillGraphSystem(config) as system:
        if not system.readonly or system.database.readonly is not True:
            raise ProtocolError("frozen system did not open its knowledge database read-only")
        preflight = system.preflight(require_api_key=True, initialize_harness=True)
        if not preflight.get("passed"):
            raise ProtocolError(
                "formal frozen preflight failed: "
                + json.dumps(preflight, ensure_ascii=False, sort_keys=True)
            )
        digest_before = system.knowledge_digest()
        freeze_manifest_path = system.data_dir / "freeze_manifest.json"
        if not freeze_manifest_path.is_file():
            raise ProtocolError(f"frozen snapshot manifest missing: {freeze_manifest_path}")
        freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        if freeze_manifest.get("knowledge_digest") != digest_before:
            raise ProtocolError("frozen snapshot digest does not match its manifest")

        tasks = system.harness.load_balanced_tasks(labels, per_type)
        if len(tasks) != expected_total:
            raise ProtocolError(f"held-out loader returned {len(tasks)} tasks, expected 60")
        counts = {label: sum(task.task_type == label for task in tasks) for label in labels}
        if any(value != per_type for value in counts.values()):
            raise ProtocolError(f"balanced held-out task counts changed: {counts}")
        validate_distinct_formal_tasks(tasks, expected_total=expected_total)
        task_items = tuple(
            TaskManifest(
                index, task.task_id, task_signature(task), f"frozen:{digest_before}",
                task.benchmark, str(system.harness.split),
                json.dumps({
                    "task_type": task.task_type,
                    "env_index": task.context.get("env_index"),
                    "game_file": task.context.get("game_file", ""),
                }, ensure_ascii=False, sort_keys=True),
            )
            for index, task in enumerate(tasks)
        )

        train_run_dir = _path(experiment.get("source_train_run_dir", "runs/alfworld_train_full_30"))
        train_manifest_path = train_run_dir / "run_manifest.json"
        if not train_manifest_path.is_file():
            raise ProtocolError(f"source train manifest missing: {train_manifest_path}")
        train_manifest = RunManifest.from_dict(json.loads(train_manifest_path.read_text(encoding="utf-8")))
        code_digest = hash_code(REPO_ROOT)
        _verify_source_train(
            train_run_dir=train_run_dir,
            train_manifest=train_manifest,
            freeze_manifest=freeze_manifest,
            frozen_digest=digest_before,
            current_code_digest=code_digest,
            current_llm_hash=hash_config(config.get("llm") or {}),
        )
        train_signatures = {item.task_signature for item in train_manifest.tasks}
        overlap = train_signatures & {item.task_signature for item in task_items}
        if overlap:
            raise ProtocolError(f"held-out manifest overlaps train signatures ({len(overlap)})")

        state_db = StateDatabase(output_dir / "run_state.sqlite3")
        try:
            config_digest = hash_config(config_path)
            store = ManifestStore(output_dir.parent, state_db)
            if resume:
                manifest = store.validate_resume(
                    run_id,
                    config_hash=config_digest,
                    code_commit=code_digest,
                    knowledge_digest=digest_before,
                    tasks=task_items,
                )
            else:
                manifest = RunManifest.create(
                    run_id=run_id, phase="frozen_eval",
                    config_hash=config_digest, code_commit=code_digest,
                    knowledge_digest=digest_before, tasks=task_items,
                    metadata={
                        "condition": "full", "source_train_run": train_manifest.run_id,
                        "task_types": labels, "tasks_per_type": per_type,
                        "total_tasks": expected_total,
                    },
                )
                store.persist_before_run(manifest)
            ensure_task_manifest(
                _path(experiment.get("task_manifest_path", "")), manifest
            )
            store.mark_run_state(run_id, RunState.RUNNING)
            by_id = {task.task_id: task for task in tasks}
            for item in store.tasks_to_run(manifest):
                artifact_before = artifact_audit_snapshot(system.database)
                try:
                    attempt_sequence = store.mark_task_running(
                        run_id, item.task_id, max_attempts=max_task_attempts,
                    )
                except ProtocolError:
                    store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                    raise
                task_attempt = attempt_ledger.begin(
                    run_id=run_id,
                    task_id=item.task_id,
                    task_signature=item.task_signature,
                    attempt_kind="task",
                    sequence=attempt_sequence,
                )
                try:
                    trace = system.run_task(
                        by_id[item.task_id], attempt_id=task_attempt.attempt_id,
                    )
                    attempt_ledger.capture(task_attempt, reason="run_task_returned")
                    artifact_after = artifact_audit_snapshot(system.database)
                    artifact_growth = artifact_growth_audit(
                        artifact_before, artifact_after,
                    )
                    digest_after = system.knowledge_digest()
                    if digest_after != digest_before:
                        raise ProtocolError("frozen evaluation changed long-term knowledge")
                    result = {
                        "benchmark_success": trace.benchmark_success,
                        "task_contract_success": trace.task_contract_success,
                        "strict_task_success": trace.strict_task_success,
                        "learning_eligible": trace.learning_eligible,
                        "graph_self_sufficient_success": trace.graph_self_sufficient_success,
                        "infrastructure_failure": trace.infrastructure_failure,
                        "knowledge_digest_before": digest_before,
                        "knowledge_digest_after": digest_after,
                        "artifact_growth": artifact_growth,
                        "artifact_lifecycle": artifact_after,
                    }
                    if trace.infrastructure_failure:
                        store.mark_task_failed(
                            run_id, item.task_id, infrastructure=True,
                            trace_id=trace.trace_id, result=result,
                        )
                        store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                        raise RuntimeError(f"infrastructure failure at held-out task {item.task_id}")
                    store.mark_task_completed(
                        run_id, item.task_id, trace_id=trace.trace_id, result=result
                    )
                    print(json.dumps({
                        "task": item.task_id,
                        "official_alfworld_won": trace.benchmark_success,
                        "strict_task_success": trace.strict_task_success,
                        "learning_eligible": trace.learning_eligible,
                        "trace_id": trace.trace_id,
                    }, ensure_ascii=False), flush=True)
                except Exception as primary:
                    def update_failed_state() -> None:
                        row = state_db.execute(
                            "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?",
                            (run_id, item.task_id),
                        ).fetchone()
                        if row is not None and row["state"] == "running":
                            store.mark_task_failed(
                                run_id, item.task_id, infrastructure=True,
                                result={
                                    "error_type": type(primary).__name__,
                                    "error": str(primary),
                                },
                            )
                        store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)

                    audit_failed_attempt(
                        primary=primary,
                        attempt=task_attempt,
                        attempt_ledger=attempt_ledger,
                        receipt_root=output_dir / "failure_receipts",
                        update_state=update_failed_state,
                        capture_reason="task_exception",
                    )
                    raise

            traces = load_task_report_traces(system.traces, state_db, run_id)
            task_trace_ids = {str(trace.get("trace_id", "")) for trace in traces}
            if "" in task_trace_ids or len(task_trace_ids) != len(traces):
                raise ProtocolError("frozen report contains invalid/duplicate task trace_id")
            attempt_usage_traces = attempt_ledger.auxiliary_traces(
                manifest=manifest,
                excluded_trace_ids=task_trace_ids,
            )
            resource_traces = [*traces, *attempt_usage_traces]
            frozen_v31_guards = validate_frozen_v31_guards(traces)
            validate_formal_usage(resource_traces)
            usage_coverage = validate_usage_event_persistence(
                system.usage.events,
                resource_traces,
            )
            print(json.dumps({
                "usage_trace_coverage": usage_coverage,
                "frozen_v31_guards": frozen_v31_guards,
            }, ensure_ascii=False), flush=True)
            write_reports(
                traces, output_dir / "reports", stem="frozen_eval_60",
                title="AtomicSkillGraph v3 ALFWorld Frozen Held-out Eval",
                auxiliary_usage_traces=attempt_usage_traces,
            )
            if system.knowledge_digest() != digest_before:
                raise ProtocolError("knowledge digest guard failed after report generation")
            store.mark_run_state(run_id, RunState.COMPLETED)
            print(json.dumps({
                "run_id": run_id, "tasks": expected_total,
                "knowledge_digest_before": digest_before,
                "knowledge_digest_after": system.knowledge_digest(),
            }, ensure_ascii=False, indent=2))
        finally:
            state_db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/alfworld_frozen_eval.yaml",
        help="frozen held-out YAML configuration",
    )
    parser.add_argument("--resume", action="store_true", help="resume at completed-task boundaries")
    args = parser.parse_args(argv)
    return run(args.config, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
