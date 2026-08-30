"""Formal ALFWorld full-method 6×5=30 online training runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.agents.provider_probe import ensure_provider_capability

from .protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    AttemptTraceLedger,
    ManifestStore,
    ProtocolError,
    RunManifest,
    RunState,
    TaskCheckpointStore,
    TaskManifest,
    audit_failed_attempt,
    artifact_audit_snapshot,
    artifact_growth_audit,
    ensure_task_manifest,
    hash_code,
    hash_config,
    load_task_report_traces,
    task_signature,
    validate_deepseek_formal_llm,
    validate_distinct_formal_tasks,
)
from .report import (
    validate_formal_usage,
    validate_usage_event_persistence,
    write_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_FINAL_MAINTENANCE_CHECKPOINT_ID = "__final_batch_maintenance__"


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _selection(config: dict[str, Any]) -> tuple[list[str], int, int]:
    selection = dict((config.get("harness") or {}).get("task_selection") or {})
    task_types = [str(item) for item in selection.get("task_types", [])]
    per_type = int(selection.get("tasks_per_type", 0))
    total = int(selection.get("total_tasks", 0))
    if tuple(task_types) != ALFWORLD_FORMAL_TASK_TYPES:
        raise ProtocolError(
            "formal train protocol requires the six ALFWorld task types in frozen order"
        )
    if per_type != 5 or total != 30:
        raise ProtocolError("formal train protocol requires six task types × five = 30")
    if total != len(task_types) * per_type or selection.get("require_exact_count") is not True:
        raise ProtocolError("formal train selection must require the exact balanced count")
    return task_types, per_type, total


def _validate_formal_config(config: dict[str, Any], output_dir: Path) -> None:
    validate_deepseek_formal_llm(config)
    experiment = dict(config.get("experiment") or {})
    harness = dict(config.get("harness") or {})
    selection = dict(harness.get("task_selection") or {})
    expected = {
        "experiment.name": (experiment.get("name"), "alfworld_train_full_30"),
        "experiment.condition": (experiment.get("condition"), "full"),
        "experiment.freeze_skills": (experiment.get("freeze_skills"), False),
        "experiment.seed": (experiment.get("seed"), 42),
        "experiment.initialize_v3_bank": (experiment.get("initialize_v3_bank"), "empty"),
        "experiment.resume_completed_task_boundary_only": (
            experiment.get("resume_completed_task_boundary_only"), True
        ),
        "harness.adapter": (harness.get("adapter"), "alfworld_v3"),
        "harness.alfworld_data_env": (harness.get("alfworld_data_env"), "ALFWORLD_DATA"),
        "harness.split": (harness.get("split"), "train"),
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
    if _path(config.get("data_dir", "")) != output_dir / "data_v3":
        mismatches.append("data_dir must be <output_dir>/data_v3")
    if _path(config.get("trace_data_dir", output_dir)) != output_dir:
        mismatches.append("trace_data_dir must equal experiment.output_dir")
    if output_dir.name != str(experiment.get("name", "")):
        mismatches.append("output_dir basename must equal experiment.name")
    if _path(experiment.get("task_manifest_path", "")) != output_dir / "task_manifest.json":
        mismatches.append("task_manifest_path must be <output_dir>/task_manifest.json")
    if _path(experiment.get("frozen_snapshot_dir", "")) != output_dir / "frozen" / "data_v3":
        mismatches.append("frozen_snapshot_dir must be <output_dir>/frozen/data_v3")
    max_task_attempts = experiment.get("max_task_attempts")
    if (
        isinstance(max_task_attempts, bool)
        or not isinstance(max_task_attempts, int)
        or max_task_attempts <= 0
    ):
        mismatches.append("experiment.max_task_attempts must be a positive integer")
    if mismatches:
        raise ProtocolError("formal train config mismatch: " + "; ".join(mismatches))


def _task_manifests(tasks: list[Any], split: str, initial_digest: str) -> tuple[TaskManifest, ...]:
    result = []
    for index, task in enumerate(tasks):
        milestone = (
            f"initial:{initial_digest}" if index == 0
            else f"after_task:{tasks[index - 1].task_id}"
        )
        result.append(TaskManifest(
            index,
            task.task_id,
            task_signature(task),
            milestone,
            task.benchmark,
            split,
            json.dumps({
                "task_type": task.task_type,
                "env_index": task.context.get("env_index"),
                "game_file": task.context.get("game_file", ""),
            }, ensure_ascii=False, sort_keys=True),
        ))
    return tuple(result)


def _verify_resume_knowledge(
    store: ManifestStore, manifest: RunManifest, current_digest: str,
) -> None:
    rows = {
        str(row["task_id"]): row
        for row in store.database.rows(
            "SELECT task_id,state,result_json FROM run_tasks WHERE run_id=?",
            (manifest.run_id,),
        )
    }
    expected = manifest.knowledge_digest
    encountered_incomplete = False
    for task in manifest.tasks:
        row = rows[task.task_id]
        if str(row["state"]) == "completed":
            if encountered_incomplete:
                raise ProtocolError("completed train tasks are not a contiguous manifest prefix")
            result = json.loads(str(row["result_json"]))
            expected = str(result.get("knowledge_digest_after", ""))
            if not expected:
                raise ProtocolError(f"completed task {task.task_id} lacks knowledge digest")
        else:
            encountered_incomplete = True
    if current_digest != expected:
        raise ProtocolError(
            f"resume knowledge mismatch: expected {expected}, current {current_digest}"
        )


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "uncommitted"


def _maintenance_trace_payloads(system: AtomicSkillGraphSystem) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for payload in system.traces.iter_payloads():
        metadata = payload.get("metadata")
        task = payload.get("task")
        is_maintenance = (
            isinstance(metadata, dict) and metadata.get("trace_kind") == "maintenance"
        ) or (
            isinstance(task, dict) and task.get("task_type") == "maintenance"
        )
        if is_maintenance:
            trace_id = str(payload.get("trace_id", ""))
            if not trace_id or trace_id in result:
                raise ProtocolError("maintenance TraceStore contains invalid/duplicate trace_id")
            result[trace_id] = payload
    return result


def _select_run_maintenance_traces(
    payloads: dict[str, dict[str, Any]],
    *,
    referenced_trace_ids: set[str],
    manifest: RunManifest,
    required_final_trace_id: str,
    task_trace_ids: set[str],
) -> list[dict[str, Any]]:
    """Resolve maintenance explicitly referenced by this run's task ledger."""

    if not required_final_trace_id or required_final_trace_id not in referenced_trace_ids:
        raise ProtocolError("final batch maintenance lacks immutable maintenance Trace")
    manifest_task_ids = {item.task_id for item in manifest.tasks}
    selected: list[dict[str, Any]] = []
    seen = set(task_trace_ids)
    for trace_id in sorted(referenced_trace_ids):
        if trace_id in seen:
            raise ProtocolError(f"duplicate task/maintenance trace_id: {trace_id}")
        if trace_id not in payloads:
            raise ProtocolError(f"referenced maintenance Trace is missing: {trace_id}")
        payload = payloads[trace_id]
        metadata = payload.get("metadata")
        task = payload.get("task")
        if not isinstance(metadata, dict) or metadata.get("trace_kind") != "maintenance":
            raise ProtocolError(f"maintenance Trace {trace_id} lacks trace_kind metadata")
        if not isinstance(task, dict) or task.get("task_type") != "maintenance":
            raise ProtocolError(f"maintenance Trace {trace_id} lacks maintenance task identity")
        triggering_task_id = str(metadata.get("triggering_task_id", ""))
        if triggering_task_id not in manifest_task_ids:
            raise ProtocolError(
                f"maintenance Trace {trace_id} is not associated with run {manifest.run_id}"
            )
        if (
            trace_id == required_final_trace_id
            and metadata.get("milestone") != "formal_full_30_final_batch"
        ):
            raise ProtocolError("final batch maintenance Trace has the wrong milestone")
        seen.add(trace_id)
        selected.append(payload)
    return selected


def _referenced_run_maintenance_trace_ids(
    store: ManifestStore, manifest: RunManifest,
) -> set[str]:
    """Read the immutable per-task references spanning all resume processes."""

    referenced: set[str] = set()
    rows = store.database.rows(
        "SELECT task_id,state,result_json FROM run_tasks WHERE run_id=? ORDER BY rowid",
        (manifest.run_id,),
    )
    if len(rows) != len(manifest.tasks):
        raise ProtocolError("formal run task ledger is incomplete")
    for item, row in zip(manifest.tasks, rows):
        if str(row["task_id"]) != item.task_id or str(row["state"]) != "completed":
            raise ProtocolError("maintenance report requires all manifest tasks completed")
        try:
            result = json.loads(str(row["result_json"]))
        except json.JSONDecodeError as exc:
            raise ProtocolError("completed task result is invalid JSON") from exc
        raw_ids = result.get("maintenance_trace_ids") or []
        if not isinstance(raw_ids, list) or any(not isinstance(value, str) for value in raw_ids):
            raise ProtocolError("completed task has invalid maintenance_trace_ids")
        referenced.update(value for value in raw_ids if value)
        final_audit = result.get("final_batch_maintenance") or {}
        if not isinstance(final_audit, dict):
            raise ProtocolError("completed task has invalid final maintenance audit")
        raw_history = result.get("final_batch_maintenance_history") or []
        if not isinstance(raw_history, list) or any(
            not isinstance(audit, dict) for audit in raw_history
        ):
            raise ProtocolError("completed task has invalid final maintenance history")
        history_trace_ids = [
            str(audit.get("maintenance_trace_id", "")) for audit in raw_history
        ]
        if any(not trace_id for trace_id in history_trace_ids):
            raise ProtocolError("final maintenance history has an empty trace_id")
        if len(history_trace_ids) != len(set(history_trace_ids)):
            raise ProtocolError("final maintenance history has duplicate trace_id values")
        referenced.update(history_trace_ids)
        final_trace_id = str(final_audit.get("maintenance_trace_id", ""))
        if final_trace_id:
            referenced.add(final_trace_id)
    return referenced


def _run_final_batch_maintenance(
    system: AtomicSkillGraphSystem,
    store: ManifestStore,
    checkpoint: TaskCheckpointStore,
    manifest: RunManifest,
    *,
    attempt_ledger: AttemptTraceLedger | None = None,
    config_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    """Close the online maintenance queue before a frozen bank can exist.

    The pre-maintenance checkpoint makes the batch replayable after a crash in
    the same way as an interrupted task boundary.  The final knowledge digest
    belongs to the last completed task's milestone chain and is guarded by an
    atomic compare-and-update in ``ManifestStore``.
    """

    if not manifest.tasks:
        raise ProtocolError("formal train requires at least one task before maintenance")
    before = system.knowledge_digest()
    artifact_before = artifact_audit_snapshot(system.database)
    last_task = manifest.tasks[-1]
    maintenance_attempt = (
        attempt_ledger.begin(
            run_id=manifest.run_id,
            task_id=last_task.task_id,
            task_signature=last_task.task_signature,
            attempt_kind="maintenance",
            sequence=attempt_ledger.next_sequence(
                run_id=manifest.run_id,
                task_id=last_task.task_id,
                attempt_kind="maintenance",
            ),
        )
        if attempt_ledger is not None
        else None
    )
    try:
        checkpoint.create(
            system.database,
            run_id=manifest.run_id,
            task_id=_FINAL_MAINTENANCE_CHECKPOINT_ID,
            before_digest=before,
            config_hash=config_digest,
            code_commit=code_digest,
        )
        result = system.run_maintenance(
            triggering_task_id=manifest.tasks[-1].task_id,
            milestone="formal_full_30_final_batch",
            finalize_pending=True,
        )
        if attempt_ledger is not None and maintenance_attempt is not None:
            attempt_ledger.capture(maintenance_attempt, reason="run_maintenance_returned")
        pending_count = getattr(result, "pending_count", None)
        if (
            isinstance(pending_count, bool)
            or not isinstance(pending_count, int)
            or pending_count < 0
        ):
            raise ProtocolError(
                "final batch maintenance returned no valid pending_count audit"
            )
        if pending_count:
            raise ProtocolError(
                f"final batch maintenance left {pending_count} unresolved proposals"
            )
        after = system.knowledge_digest()
        artifact_after = artifact_audit_snapshot(system.database)
        last_task_id = manifest.tasks[-1].task_id
        result_row = store.database.execute(
            "SELECT result_json FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, last_task_id),
        ).fetchone()
        if result_row is None:
            raise ProtocolError("final completed task result is missing")
        try:
            last_result = json.loads(str(result_row["result_json"]))
        except json.JSONDecodeError as exc:
            raise ProtocolError("final completed task result is invalid JSON") from exc
        prior_growth = dict(last_result.get("artifact_growth") or {})
        original_before = prior_growth.get("before")
        if not isinstance(original_before, dict):
            raise ProtocolError("final completed task lacks pre-task artifact snapshot")
        final_task_growth = artifact_growth_audit(original_before, artifact_after)
        maintenance_growth = artifact_growth_audit(artifact_before, artifact_after)
        audit = {
            "knowledge_digest_before": before,
            "knowledge_digest_after": after,
            "pending_count": pending_count,
            "admitted_refs": list(getattr(result, "admitted_refs", ()) or ()),
            "rejected_proposal_ids": list(
                getattr(result, "rejected_proposal_ids", ()) or ()
            ),
            "maintenance_trace_id": str(
                getattr(result, "maintenance_trace_id", "") or ""
            ),
            "artifact_growth": maintenance_growth,
            "artifact_lifecycle": artifact_after,
        }
        raw_history = last_result.get("final_batch_maintenance_history") or []
        if not isinstance(raw_history, list) or any(
            not isinstance(item, dict) for item in raw_history
        ):
            raise ProtocolError("final batch maintenance history is invalid")
        maintenance_history = [dict(item) for item in raw_history]
        legacy_latest = last_result.get("final_batch_maintenance") or {}
        if not isinstance(legacy_latest, dict):
            raise ProtocolError("final batch maintenance audit is invalid")
        history_trace_ids = [
            str(item.get("maintenance_trace_id", ""))
            for item in maintenance_history
        ]
        if any(not trace_id for trace_id in history_trace_ids):
            raise ProtocolError("final batch maintenance history has an empty trace_id")
        if len(history_trace_ids) != len(set(history_trace_ids)):
            raise ProtocolError("final batch maintenance history has duplicate trace_id values")
        existing_trace_ids = set(history_trace_ids)
        if legacy_latest:
            legacy_trace_id = str(legacy_latest.get("maintenance_trace_id", ""))
            if not legacy_trace_id:
                raise ProtocolError("legacy final maintenance audit has no trace_id")
            if legacy_trace_id not in existing_trace_ids:
                maintenance_history.append(dict(legacy_latest))
                existing_trace_ids.add(legacy_trace_id)
        new_trace_id = str(audit.get("maintenance_trace_id", ""))
        if not new_trace_id:
            raise ProtocolError("final maintenance audit has no trace_id")
        if new_trace_id in existing_trace_ids:
            raise ProtocolError(
                f"final maintenance trace_id was already recorded: {new_trace_id}"
            )
        maintenance_history.append(audit)
        store.update_completed_task_knowledge_digest(
            manifest.run_id,
            last_task_id,
            expected_digest=before,
            new_digest=after,
            result_updates={
                "artifact_growth": final_task_growth,
                "artifact_lifecycle": artifact_after,
                "final_batch_maintenance": audit,
                "final_batch_maintenance_history": maintenance_history,
            },
        )
        checkpoint.clear()
        return audit
    except Exception as primary:
        # Keep the checkpoint intact.  A same-code --resume restores the exact
        # pre-maintenance bank and reruns this batch; no partial queue mutation
        # can reach freeze.
        if attempt_ledger is not None and maintenance_attempt is not None:
            audit_failed_attempt(
                primary=primary,
                attempt=maintenance_attempt,
                attempt_ledger=attempt_ledger,
                receipt_root=system.trace_data_dir / "failure_receipts",
                update_state=lambda: store.mark_run_state(
                    manifest.run_id, RunState.INFRASTRUCTURE_FAILED
                ),
                capture_reason="maintenance_exception",
            )
        else:
            try:
                store.mark_run_state(manifest.run_id, RunState.INFRASTRUCTURE_FAILED)
            except Exception:
                pass
        raise


def run(config_path: str | Path, *, resume: bool = False) -> int:
    config_path = _path(config_path)
    config = load_config(config_path)
    experiment = dict(config.get("experiment") or {})
    if experiment.get("phase") != "train" or experiment.get("runtime_mode") != "online":
        raise ProtocolError("train runner requires phase=train and runtime_mode=online")
    task_types, per_type, expected_total = _selection(config)
    output_dir = _path(experiment.get("output_dir", "runs/alfworld_train_full_30"))
    _validate_formal_config(config, output_dir)
    max_task_attempts = int(experiment["max_task_attempts"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(experiment.get("name", "alfworld_train_full_30"))
    config_digest = hash_config(config_path)
    code_digest = hash_code(REPO_ROOT)
    capability_manifest = ensure_provider_capability(
        config,
        output_dir=output_dir,
        config_hash=config_digest,
        code_hash=code_digest,
        run_if_missing=not resume,
    )
    print(json.dumps({
        "provider_capability_manifest": {
            "passed": capability_manifest.get("passed"),
            "provider_fingerprint": capability_manifest.get("provider_fingerprint"),
            "config_hash": capability_manifest.get("config_hash"),
            "code_hash": capability_manifest.get("code_hash"),
        }
    }, ensure_ascii=False), flush=True)
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
            "Archive this run and start a fresh run without --resume: "
            + ", ".join(str(item["attempt_id"]) for item in unresolved_attempts)
        )
    checkpoint = TaskCheckpointStore(
        output_dir / ".task_checkpoint", _path(config.get("data_dir", ""))
    )
    recovered_task = checkpoint.recover_if_present(
        run_id=run_id,
        config_hash=config_digest,
        code_commit=code_digest,
        resume=resume,
    )
    if recovered_task:
        print(json.dumps({
            "recovered_knowledge_checkpoint": recovered_task,
        }, ensure_ascii=False), flush=True)

    with AtomicSkillGraphSystem(config, readonly=False) as system:
        preflight = system.preflight(
            require_api_key=True,
            initialize_harness=True,
            require_empty_bank=not resume,
        )
        if not preflight.get("passed"):
            raise ProtocolError(
                "formal train preflight failed: "
                + json.dumps(preflight, ensure_ascii=False, sort_keys=True)
            )
        if not resume:
            if not system.is_empty_knowledge_bank():
                raise ProtocolError("fresh full-30 training requires an empty schema-v3 bank")
        tasks = system.harness.load_balanced_tasks(task_types, per_type)
        if len(tasks) != expected_total:
            raise ProtocolError(f"balanced loader returned {len(tasks)} tasks, expected 30")
        counts = {label: sum(task.task_type == label for task in tasks) for label in task_types}
        if any(value != per_type for value in counts.values()):
            raise ProtocolError(f"balanced task counts changed: {counts}")
        validate_distinct_formal_tasks(tasks, expected_total=expected_total)

        current_digest = system.knowledge_digest()
        maintenance_trace_ids_at_process_start = set(
            _maintenance_trace_payloads(system)
        )
        store = ManifestStore(output_dir.parent, system.database)

        if resume:
            persisted = store.load(run_id)
            # The immutable first-task milestone is anchored to the original
            # empty bank, not to the evolved bank that exists at resume time.
            task_items = _task_manifests(
                tasks, str(system.harness.split), persisted.knowledge_digest
            )
            manifest = store.validate_resume(
                run_id,
                config_hash=config_digest,
                code_commit=code_digest,
                knowledge_digest=persisted.knowledge_digest,
                tasks=task_items,
            )
            _verify_resume_knowledge(store, manifest, current_digest)
        else:
            initial_digest = current_digest
            task_items = _task_manifests(
                tasks, str(system.harness.split), initial_digest
            )
            manifest = RunManifest.create(
                run_id=run_id,
                phase="train",
                config_hash=config_digest,
                code_commit=code_digest,
                knowledge_digest=initial_digest,
                tasks=task_items,
                metadata={
                    "condition": "full",
                    "llm_config_hash": hash_config(config.get("llm") or {}),
                    "task_types": task_types,
                    "tasks_per_type": per_type,
                    "total_tasks": expected_total,
                    "git_revision": _git_revision(),
                },
            )
            store.persist_before_run(manifest)

        ensure_task_manifest(
            _path(experiment.get("task_manifest_path", "")), manifest
        )

        store.mark_run_state(run_id, RunState.RUNNING)
        by_id = {task.task_id: task for task in tasks}
        for item in store.tasks_to_run(manifest):
            before = system.knowledge_digest()
            artifact_before = artifact_audit_snapshot(system.database)
            expected_periodic_milestone = (
                system.expected_periodic_maintenance_milestone_after_success()
            )
            try:
                attempt_sequence = store.mark_task_running(
                    run_id,
                    item.task_id,
                    max_attempts=max_task_attempts,
                )
            except ProtocolError as exc:
                row = system.database.execute(
                    "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?",
                    (run_id, item.task_id),
                ).fetchone()
                if row is not None and row["state"] == "running":
                    store.mark_task_failed(
                        run_id,
                        item.task_id,
                        infrastructure=True,
                        result={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "max_task_attempts": max_task_attempts,
                        },
                    )
                store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                raise
            task_attempt = attempt_ledger.begin(
                run_id=run_id,
                task_id=item.task_id,
                task_signature=item.task_signature,
                attempt_kind="task",
                sequence=attempt_sequence,
                expected_periodic_milestone=expected_periodic_milestone,
            )
            try:
                checkpoint.create(
                    system.database,
                    run_id=run_id,
                    task_id=item.task_id,
                    before_digest=before,
                    config_hash=config_digest,
                    code_commit=code_digest,
                )
                maintenance_ids_before_task = set(
                    _maintenance_trace_payloads(system)
                )
                trace = system.run_task(
                    by_id[item.task_id], attempt_id=task_attempt.attempt_id,
                )
                attempt_ledger.capture(task_attempt, reason="run_task_returned")
                maintenance_ids_after_task = set(
                    _maintenance_trace_payloads(system)
                )
                task_maintenance_trace_ids = sorted(
                    maintenance_ids_after_task - maintenance_ids_before_task
                )
                artifact_after = artifact_audit_snapshot(system.database)
                artifact_growth = artifact_growth_audit(
                    artifact_before, artifact_after,
                )
                after = system.knowledge_digest()
                result = {
                    "benchmark_success": trace.benchmark_success,
                    "graph_self_sufficient_success": trace.graph_self_sufficient_success,
                    "infrastructure_failure": trace.infrastructure_failure,
                    "knowledge_digest_before": before,
                    "knowledge_digest_after": after,
                    "artifact_growth": artifact_growth,
                    "artifact_lifecycle": artifact_after,
                    "maintenance_trace_ids": task_maintenance_trace_ids,
                }
                if trace.infrastructure_failure:
                    store.mark_task_failed(
                        run_id, item.task_id, infrastructure=True,
                        trace_id=trace.trace_id, result=result,
                    )
                    store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                    raise RuntimeError(
                        f"infrastructure failure at {item.task_id}; resume after fixing it"
                    )
                # Benchmark failure is a completed experimental outcome, not a
                # resumable infrastructure failure.
                store.mark_task_completed(
                    run_id, item.task_id, trace_id=trace.trace_id, result=result
                )
                checkpoint.clear()
                print(json.dumps({
                    "task": item.task_id,
                    "success": trace.benchmark_success,
                    "trace_id": trace.trace_id,
                }, ensure_ascii=False), flush=True)
            except Exception as primary:
                def update_failed_state() -> None:
                    row = system.database.execute(
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

        maintenance_audit = _run_final_batch_maintenance(
            system,
            store,
            checkpoint,
            manifest,
            attempt_ledger=attempt_ledger,
            config_digest=config_digest,
            code_digest=code_digest,
        )
        print(json.dumps({
            "final_batch_maintenance": maintenance_audit,
        }, ensure_ascii=False), flush=True)

        task_traces = load_task_report_traces(system.traces, system.database, run_id)
        task_trace_ids = {
            str(trace.get("trace_id", "")) for trace in task_traces
        }
        if "" in task_trace_ids or len(task_trace_ids) != len(task_traces):
            raise ProtocolError("formal task report contains invalid/duplicate trace_id")
        maintenance_trace_id = str(maintenance_audit.get("maintenance_trace_id", ""))
        maintenance_payloads = _maintenance_trace_payloads(system)
        referenced_maintenance_ids = _referenced_run_maintenance_trace_ids(
            store, manifest,
        )
        maintenance_traces = _select_run_maintenance_traces(
            maintenance_payloads,
            referenced_trace_ids=referenced_maintenance_ids,
            manifest=manifest,
            required_final_trace_id=maintenance_trace_id,
            task_trace_ids=task_trace_ids,
        )
        authoritative_trace_ids = task_trace_ids | referenced_maintenance_ids
        attempt_usage_traces = attempt_ledger.auxiliary_traces(
            manifest=manifest,
            excluded_trace_ids=authoritative_trace_ids,
        )
        current_maintenance_ids = (
            set(maintenance_payloads) - maintenance_trace_ids_at_process_start
        )
        unreferenced_current = current_maintenance_ids - referenced_maintenance_ids
        if unreferenced_current:
            raise ProtocolError(
                "current process produced unledgered maintenance Traces: "
                + ", ".join(sorted(unreferenced_current))
            )
        resource_traces = [
            *task_traces, *maintenance_traces, *attempt_usage_traces,
        ]
        validate_formal_usage(resource_traces)
        usage_coverage = validate_usage_event_persistence(
            system.usage.events, resource_traces,
        )
        print(json.dumps({"usage_trace_coverage": usage_coverage}, ensure_ascii=False), flush=True)
        write_reports(
            task_traces, output_dir / "reports", stem="train_full_30",
            title="AtomicSkillGraph v3 ALFWorld Full-30 Train",
            auxiliary_usage_traces=[*maintenance_traces, *attempt_usage_traces],
        )
        frozen_dir = _path(
            experiment.get("frozen_snapshot_dir", "runs/alfworld_train_full_30/frozen/data_v3")
        )
        if frozen_dir.exists():
            manifest_path = frozen_dir / "freeze_manifest.json"
            if not resume or not manifest_path.is_file():
                raise FileExistsError(frozen_dir)
            frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if frozen_manifest.get("knowledge_digest") != system.knowledge_digest():
                raise ProtocolError("existing frozen snapshot does not match completed train knowledge")
            expected_provenance = {
                "source_run_id": manifest.run_id,
                "source_run_manifest_hash": manifest.manifest_hash,
                "source_config_hash": manifest.config_hash,
                "source_code_commit": manifest.code_commit,
                "source_task_manifest_hash": manifest.task_manifest_hash,
                "source_initial_knowledge_digest": manifest.knowledge_digest,
                "source_final_knowledge_digest": system.knowledge_digest(),
                "source_llm_config_hash": str(manifest.metadata.get("llm_config_hash", "")),
            }
            if frozen_manifest.get("provenance") != expected_provenance:
                raise ProtocolError("existing frozen snapshot provenance does not match train run")
        else:
            final_digest = system.knowledge_digest()
            system.freeze(frozen_dir, provenance={
                "source_run_id": manifest.run_id,
                "source_run_manifest_hash": manifest.manifest_hash,
                "source_config_hash": manifest.config_hash,
                "source_code_commit": manifest.code_commit,
                "source_task_manifest_hash": manifest.task_manifest_hash,
                "source_initial_knowledge_digest": manifest.knowledge_digest,
                "source_final_knowledge_digest": final_digest,
                "source_llm_config_hash": str(manifest.metadata.get("llm_config_hash", "")),
            })
        store.mark_run_state(run_id, RunState.COMPLETED)
        print(json.dumps({
            "run_id": run_id,
            "tasks": expected_total,
            "frozen_snapshot": str(frozen_dir),
            "knowledge_digest": system.knowledge_digest(),
        }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/alfworld_train_full_30.yaml",
        help="full-30 YAML configuration",
    )
    parser.add_argument("--resume", action="store_true", help="resume at completed-task boundaries")
    args = parser.parse_args(argv)
    return run(args.config, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
