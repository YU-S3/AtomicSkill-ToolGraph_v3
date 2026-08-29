from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.system import load_config
from experiments.protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    ManifestStore,
    ProtocolError,
    RunManifest,
    RunState,
    TaskCheckpointStore,
    TaskManifest,
    artifact_audit_snapshot,
    artifact_growth_audit,
    load_task_report_traces,
)
from experiments.run_v3_frozen_eval import _selection as frozen_selection
from experiments.run_v3_smoke import _validate_configured_task_manifest
from experiments.run_v3_train import (
    _referenced_run_maintenance_trace_ids,
    _run_final_batch_maintenance,
    _select_run_maintenance_traces,
    _selection as train_selection,
)
from experiments.report import summarize_traces, write_reports


ROOT = Path(__file__).resolve().parents[2]


def test_formal_configs_freeze_the_exact_six_task_types_and_counts() -> None:
    train = load_config(ROOT / "configs" / "alfworld_train_full_30.yaml")
    labels, per_type, total = train_selection(train)
    assert tuple(labels) == ALFWORLD_FORMAL_TASK_TYPES
    assert (per_type, total) == (5, 30)

    frozen = load_config(ROOT / "configs" / "alfworld_frozen_eval.yaml")
    labels, per_type, total = frozen_selection(frozen)
    assert tuple(labels) == ALFWORLD_FORMAL_TASK_TYPES
    assert (per_type, total) == (10, 60)

    reordered = copy.deepcopy(train)
    selection = reordered["harness"]["task_selection"]
    selection["task_types"] = list(reversed(selection["task_types"]))
    with pytest.raises(ProtocolError, match="frozen order"):
        train_selection(reordered)

    substituted = copy.deepcopy(train)
    substituted["harness"]["task_selection"]["task_types"][-1] = "unknown"
    with pytest.raises(ProtocolError, match="six ALFWorld task types"):
        train_selection(substituted)


@dataclass
class _Task:
    task_id: str
    task_type: str
    metadata: dict[str, str]
    benchmark: str = "alfworld"
    split: str = "train"
    goal: str = "test goal"
    context: dict[str, object] = field(default_factory=dict)


class _Harness:
    split = "train"

    def __init__(self, tasks: list[_Task]) -> None:
        self.tasks = tasks

    def load_balanced_tasks(self, task_types: list[str], per_type: int) -> list[_Task]:
        assert tuple(task_types) == ALFWORLD_FORMAL_TASK_TYPES
        assert per_type == 5
        return list(self.tasks)


def _configured_tasks() -> list[_Task]:
    return [
        _Task(
            task_id=f"task-{label}-{index}",
            task_type=label,
            metadata={"task_signature": f"signature-{label}-{index}"},
        )
        for label in ALFWORLD_FORMAL_TASK_TYPES
        for index in range(5)
    ]


def test_preflight_materializes_exact_balanced_unique_manifest() -> None:
    config = load_config(ROOT / "configs" / "alfworld_train_full_30.yaml")
    tasks = _configured_tasks()
    checks = _validate_configured_task_manifest(
        config, SimpleNamespace(harness=_Harness(tasks)),
    )
    assert checks["task_manifest_schema"] is True
    assert checks["task_manifest_selection"] is True
    assert checks["task_manifest_task_count"] == 30
    assert checks["task_manifest_counts"] == {
        label: 5 for label in ALFWORLD_FORMAL_TASK_TYPES
    }
    assert len(str(checks["task_manifest_hash"])) == 64

    duplicate = list(tasks)
    duplicate[-1] = _Task(
        task_id=tasks[-1].task_id,
        task_type=tasks[-1].task_type,
        metadata={"task_signature": tasks[0].metadata["task_signature"]},
    )
    with pytest.raises(ValueError, match="identity-unique"):
        _validate_configured_task_manifest(
            config, SimpleNamespace(harness=_Harness(duplicate)),
        )


@dataclass
class _BatchResult:
    pending_count: int
    admitted_refs: tuple[str, ...] = ("skill://atomic_probe@1.0.0",)
    rejected_proposal_ids: tuple[str, ...] = ()
    maintenance_trace_id: str = "maintenance-trace"


class _MaintenanceSystem:
    def __init__(self, database: StateDatabase, *, pending_count: int = 0) -> None:
        self.database = database
        self.pending_count = pending_count
        self.maintenance_calls: list[dict[str, object]] = []

    def knowledge_digest(self) -> str:
        row = self.database.execute(
            "SELECT value FROM metadata WHERE key='batch-maintained'"
        ).fetchone()
        if row is None:
            return "pre-maintenance"
        count = int(row["value"])
        return "post-maintenance" if count == 1 else f"post-maintenance-{count}"

    def run_maintenance(self, **kwargs: object) -> _BatchResult:
        self.maintenance_calls.append(dict(kwargs))
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='batch-maintained'"
            ).fetchone()
            count = (0 if row is None else int(row["value"])) + 1
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("batch-maintained", str(count)),
            )
        trace_id = "maintenance-trace" if count == 1 else f"maintenance-trace-{count}"
        return _BatchResult(self.pending_count, maintenance_trace_id=trace_id)


def _completed_single_task_run(
    tmp_path: Path,
) -> tuple[StateDatabase, ManifestStore, RunManifest, TaskCheckpointStore]:
    data_dir = tmp_path / "run" / "data_v3"
    database = StateDatabase(data_dir / "state.sqlite3")
    store = ManifestStore(tmp_path, database)
    task = TaskManifest(0, "task-1", "signature-1", "initial:empty")
    manifest = RunManifest.create(
        run_id="formal-run",
        phase="train",
        config_hash="config-hash",
        code_commit="code-hash",
        knowledge_digest="empty",
        tasks=(task,),
    )
    store.persist_before_run(manifest)
    store.mark_run_state(manifest.run_id, RunState.RUNNING)
    store.mark_task_running(manifest.run_id, task.task_id)
    artifact_snapshot = artifact_audit_snapshot(database)
    store.mark_task_completed(
        manifest.run_id,
        task.task_id,
        trace_id="trace-1",
        result={
            "knowledge_digest_before": "empty",
            "knowledge_digest_after": "pre-maintenance",
            "artifact_growth": artifact_growth_audit(
                artifact_snapshot, artifact_snapshot,
            ),
            "artifact_lifecycle": artifact_snapshot,
            "maintenance_trace_ids": ["periodic-maintenance"],
        },
    )
    checkpoint = TaskCheckpointStore(
        tmp_path / "run" / ".task_checkpoint", data_dir,
    )
    return database, store, manifest, checkpoint


def test_final_batch_maintenance_updates_digest_and_clears_checkpoint(
    tmp_path: Path,
) -> None:
    database, store, manifest, checkpoint = _completed_single_task_run(tmp_path)
    try:
        system = _MaintenanceSystem(database)
        audit = _run_final_batch_maintenance(
            system,  # type: ignore[arg-type]
            store,
            checkpoint,
            manifest,
            config_digest=manifest.config_hash,
            code_digest=manifest.code_commit,
        )
        assert audit["pending_count"] == 0
        assert audit["knowledge_digest_before"] == "pre-maintenance"
        assert audit["knowledge_digest_after"] == "post-maintenance"
        assert system.maintenance_calls == [{
            "triggering_task_id": "task-1",
            "milestone": "formal_full_30_final_batch",
            "finalize_pending": True,
        }]
        row = database.execute(
            "SELECT result_json FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, manifest.tasks[-1].task_id),
        ).fetchone()
        assert json.loads(row["result_json"])["knowledge_digest_after"] == "post-maintenance"
        persisted = json.loads(row["result_json"])
        assert persisted["final_batch_maintenance_history"] == [
            persisted["final_batch_maintenance"]
        ]
        assert _referenced_run_maintenance_trace_ids(store, manifest) == {
            "maintenance-trace",
            "periodic-maintenance",
        }
        assert not checkpoint.root.exists()
        source_payload = {
            "trace_id": "trace-1",
            "task": {"task_id": "task-1"},
            "metadata": {"immutable_trace_field": True},
        }
        trace_store = SimpleNamespace(
            load_payload=lambda _trace_id: copy.deepcopy(source_payload)
        )
        report_traces = load_task_report_traces(
            trace_store, database, manifest.run_id,
        )
        assert report_traces[0]["metadata"]["artifact_growth"]
        assert report_traces[0]["metadata"]["artifact_lifecycle"]
        assert report_traces[0]["metadata"]["final_batch_maintenance"]
        assert source_payload == {
            "trace_id": "trace-1",
            "task": {"task_id": "task-1"},
            "metadata": {"immutable_trace_field": True},
        }
    finally:
        database.close()


def _maintenance_payload(
    trace_id: str,
    triggering_task_id: str,
    milestone: str,
    *,
    total_tokens: int = 0,
) -> dict[str, object]:
    usage = []
    if total_tokens:
        usage.append({
            "event_id": f"usage-{trace_id}",
            "bucket": "evolution_repair",
            "prompt_tokens": total_tokens - 2,
            "completion_tokens": 2,
            "total_tokens": total_tokens,
            "reasoning_tokens": 0,
            "call_count": 1,
            "latency_ms": 1.0,
            "provider_metadata": {"usage_status": "reported"},
        })
    return {
        "trace_id": trace_id,
        "schema_version": 3,
        "task": {"task_id": f"task-{trace_id}", "task_type": "maintenance"},
        "runtime_plan": {"source": "batch_maintenance", "control_sequence": []},
        "llm_usage": usage,
        "metadata": {
            "trace_kind": "maintenance",
            "triggering_task_id": triggering_task_id,
            "milestone": milestone,
        },
    }


def test_crash_resume_appends_final_maintenance_history_and_reports_both(
    tmp_path: Path,
) -> None:
    database, store, manifest, checkpoint = _completed_single_task_run(tmp_path)
    try:
        system = _MaintenanceSystem(database)
        first = _run_final_batch_maintenance(
            system,  # type: ignore[arg-type]
            store,
            checkpoint,
            manifest,
            config_digest=manifest.config_hash,
            code_digest=manifest.code_commit,
        )
        # Simulate a crash after the first compare-and-update but before report/freeze.
        second = _run_final_batch_maintenance(
            system,  # type: ignore[arg-type]
            store,
            checkpoint,
            manifest,
            config_digest=manifest.config_hash,
            code_digest=manifest.code_commit,
        )
        row = database.execute(
            "SELECT result_json FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, manifest.tasks[-1].task_id),
        ).fetchone()
        persisted = json.loads(row["result_json"])
        history_ids = [
            item["maintenance_trace_id"]
            for item in persisted["final_batch_maintenance_history"]
        ]
        assert history_ids == ["maintenance-trace", "maintenance-trace-2"]
        assert persisted["final_batch_maintenance"] == second

        referenced = _referenced_run_maintenance_trace_ids(store, manifest)
        assert referenced == {
            "periodic-maintenance", "maintenance-trace", "maintenance-trace-2",
        }
        payloads = {
            "periodic-maintenance": _maintenance_payload(
                "periodic-maintenance", "task-1", "online_success_5",
            ),
            "maintenance-trace": _maintenance_payload(
                "maintenance-trace", "task-1", "formal_full_30_final_batch",
                total_tokens=5,
            ),
            "maintenance-trace-2": _maintenance_payload(
                "maintenance-trace-2", "task-1", "formal_full_30_final_batch",
                total_tokens=5,
            ),
            "other-run": _maintenance_payload(
                "other-run", "different-task", "formal_full_30_final_batch",
                total_tokens=99,
            ),
        }
        selected = _select_run_maintenance_traces(
            payloads,
            referenced_trace_ids=referenced,
            manifest=manifest,
            required_final_trace_id=str(second["maintenance_trace_id"]),
            task_trace_ids={"task-trace"},
        )
        assert [item["trace_id"] for item in selected] == [
            "maintenance-trace", "maintenance-trace-2", "periodic-maintenance",
        ]
        task_trace = {
            "trace_id": "task-trace",
            "schema_version": 3,
            "task": {"task_id": "task-1", "task_type": "train"},
            "runtime_plan": {"source": "full_dynamic", "control_sequence": []},
            "llm_usage": [],
            "metadata": {},
        }
        summary = summarize_traces(
            [task_trace], auxiliary_usage_traces=selected,
        )
        assert summary["task_count"] == 1
        assert summary["total_tokens"] == 10
        assert summary["usage_by_bucket"]["evolution_repair"]["call_count"] == 2
        paths = write_reports(
            [task_trace], tmp_path / "reports", stem="resume",
            auxiliary_usage_traces=selected,
        )
        assert len(paths.jsonl.read_text(encoding="utf-8").splitlines()) == 1
        assert "| evolution_repair | 2 | 6 | 4 | 10 |" in paths.markdown.read_text(
            encoding="utf-8"
        )
        assert first["maintenance_trace_id"] != second["maintenance_trace_id"]
    finally:
        database.close()


def test_run_maintenance_selection_uses_ledger_refs_without_history_or_duplicates() -> None:
    manifest = RunManifest.create(
        run_id="formal-run",
        phase="train",
        config_hash="config-hash",
        code_commit="code-hash",
        knowledge_digest="empty",
        tasks=(TaskManifest(0, "task-1", "signature-1", "initial:empty"),),
    )
    payloads = {
        "unreferenced-history": _maintenance_payload(
            "unreferenced-history", "task-1", "online_success_5",
        ),
        "periodic": _maintenance_payload(
            "periodic", "task-1", "online_success_10",
        ),
        "final": _maintenance_payload(
            "final", "task-1", "formal_full_30_final_batch",
        ),
    }
    selected = _select_run_maintenance_traces(
        payloads,
        referenced_trace_ids={"periodic", "final"},
        manifest=manifest,
        required_final_trace_id="final",
        task_trace_ids={"task-trace"},
    )
    assert [item["trace_id"] for item in selected] == ["final", "periodic"]

    payloads["other-run"] = _maintenance_payload(
        "other-run", "different-task", "online_success_5",
    )
    with pytest.raises(ProtocolError, match="not associated with run formal-run"):
        _select_run_maintenance_traces(
            payloads,
            referenced_trace_ids={"final", "other-run"},
            manifest=manifest,
            required_final_trace_id="final",
            task_trace_ids={"task-trace"},
        )

    with pytest.raises(ProtocolError, match="duplicate task/maintenance"):
        _select_run_maintenance_traces(
            {"final": payloads["final"]},
            referenced_trace_ids={"final"},
            manifest=manifest,
            required_final_trace_id="final",
            task_trace_ids={"final"},
        )


def test_unresolved_final_batch_is_fail_closed_and_checkpointed(
    tmp_path: Path,
) -> None:
    database, store, manifest, checkpoint = _completed_single_task_run(tmp_path)
    try:
        with pytest.raises(ProtocolError, match="left 1 unresolved"):
            _run_final_batch_maintenance(
                _MaintenanceSystem(database, pending_count=1),  # type: ignore[arg-type]
                store,
                checkpoint,
                manifest,
                config_digest=manifest.config_hash,
                code_digest=manifest.code_commit,
            )
        assert checkpoint.root.exists()
        assert database.execute(
            "SELECT state FROM run_manifests WHERE run_id=?", (manifest.run_id,)
        ).fetchone()["state"] == "infrastructure_failed"
    finally:
        database.close()
