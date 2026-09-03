from __future__ import annotations

import copy
import csv
import json
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


# Import the leaf database module without executing atomic_skillgraph.__init__;
# that public package intentionally loads optional runner/provider dependencies.
ROOT = Path(__file__).resolve().parents[2]
SRC_PACKAGE = ROOT / "src" / "atomic_skillgraph"
if "atomic_skillgraph" not in sys.modules:
    package = types.ModuleType("atomic_skillgraph")
    package.__path__ = [str(SRC_PACKAGE)]
    package.__package__ = "atomic_skillgraph"
    sys.modules["atomic_skillgraph"] = package

from atomic_skillgraph.knowledge.database import StateDatabase
from experiments.protocol import (
    ManifestExistsError,
    ManifestMismatchError,
    ManifestStore,
    ProtocolError,
    RunManifest,
    RunState,
    TaskCheckpointStore,
    TaskManifest,
    artifact_audit_snapshot,
    artifact_growth_audit,
    hash_code,
    hash_config,
    hash_knowledge,
)
from experiments.report import (
    REPORT_COLUMNS,
    summarize_traces,
    trace_to_row,
    validate_formal_usage,
    validate_usage_event_persistence,
    write_reports,
)


def _tasks(milestone: str = "knowledge_001") -> tuple[TaskManifest, ...]:
    return (
        TaskManifest(
            ordinal=0,
            task_id="task-1",
            task_signature="signature-1",
            knowledge_milestone=milestone,
            benchmark="alfworld",
            split="train",
            metadata_json='{"difficulty":"easy"}',
        ),
        TaskManifest(
            ordinal=1,
            task_id="task-2",
            task_signature="signature-2",
            knowledge_milestone=milestone,
            benchmark="alfworld",
            split="train",
        ),
    )


def _manifest(tasks: tuple[TaskManifest, ...] | None = None) -> RunManifest:
    return RunManifest.create(
        run_id="train-001",
        phase="train",
        config_hash="config-hash",
        code_commit="code-hash",
        knowledge_digest="knowledge-hash",
        tasks=tasks or _tasks(),
        metadata={"seed": 7},
        created_at="2026-08-29T00:00:00+00:00",
    )


def test_manifests_are_deeply_stable_and_hash_semantics_are_deterministic(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    before = manifest.manifest_hash

    with pytest.raises(FrozenInstanceError):
        manifest.run_id = "other"  # type: ignore[misc]
    copied_metadata = manifest.metadata
    copied_metadata["seed"] = 99
    assert manifest.metadata == {"seed": 7}
    assert manifest.manifest_hash == before
    assert RunManifest.from_dict(manifest.to_dict()) == manifest

    first_config = tmp_path / "a.json"
    second_config = tmp_path / "b.json"
    first_config.write_text('{"alpha":1,"nested":{"b":2,"a":1}}', encoding="utf-8")
    second_config.write_text(
        '{\n  "nested": {"a": 1, "b": 2},\n  "alpha": 1\n}', encoding="utf-8"
    )
    assert hash_config(first_config) == hash_config(second_config)
    with pytest.raises(FileNotFoundError):
        hash_config(tmp_path / "missing.json")

    code_root = tmp_path / "code"
    code_root.mkdir()
    source = code_root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    original_code_hash = hash_code(code_root)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert hash_code(code_root) != original_code_hash

    data_dir = tmp_path / "data"
    artifact = data_dir / "artifacts" / "atomic" / "a" / "v1.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"version":1}', encoding="utf-8")
    database = StateDatabase(data_dir / "state" / "asg_v3.sqlite")
    try:
        original_knowledge_hash = hash_knowledge(data_dir, database=database)
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO evidence_events(event_id,schema_version,task_id,trace_id,"
                "occurrence_id,attempt_id,sequence_no,artifact_ref,artifact_kind,event_type,"
                "failure_layer,confidence,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "event-1",
                    3,
                    "task-1",
                    "trace-1",
                    "occ-1",
                    "attempt-1",
                    0,
                    "atomic:a@1",
                    "atomic",
                    "selected",
                    "",
                    1.0,
                    "{}",
                ),
            )
        after_evidence_hash = hash_knowledge(data_dir, database=database)
        assert after_evidence_hash != original_knowledge_hash
        failure_payload = (
            data_dir / "failure_knowledge" / "provisional" / "probe.json"
        )
        failure_payload.parent.mkdir(parents=True)
        failure_payload.write_text('{"kind":"provisional_atomic"}', encoding="utf-8")
        after_failure_file_hash = hash_knowledge(data_dir, database=database)
        assert after_failure_file_hash != after_evidence_hash
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO cold_start_evidence VALUES(?,?,?,?,?,?,?,?)",
                (
                    "cold-event", "task-1", "trace-1", "provisional:probe",
                    "provisional_atomic", "observed", 0, "{}",
                ),
            )
        assert hash_knowledge(data_dir, database=database) != after_failure_file_hash
    finally:
        database.close()


def test_manifest_is_written_before_run_and_resume_is_fieldwise_fail_closed(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite")
    try:
        store = ManifestStore(tmp_path / "manifests", database)
        manifest = _manifest()
        path = store.persist_before_run(manifest)
        immutable_bytes = path.read_bytes()
        assert json.loads(immutable_bytes)["tasks"][0]["task_id"] == "task-1"
        assert database.execute(
            "SELECT state FROM run_manifests WHERE run_id=?", (manifest.run_id,)
        ).fetchone()["state"] == "pending"

        with pytest.raises(ManifestExistsError):
            store.persist_before_run(manifest)
        assert store.persist_before_run(manifest, resume=True) == path
        assert store.validate_resume(
            manifest.run_id,
            config_hash=manifest.config_hash,
            code_commit=manifest.code_commit,
            knowledge_digest=manifest.knowledge_digest,
            tasks=manifest.tasks,
        ) == manifest

        checks = (
            {"config_hash": "changed-config"},
            {"code_commit": "changed-code"},
            {"knowledge_digest": "changed-knowledge"},
        )
        for change in checks:
            parameters = {
                "config_hash": manifest.config_hash,
                "code_commit": manifest.code_commit,
                "knowledge_digest": manifest.knowledge_digest,
                "tasks": manifest.tasks,
            }
            parameters.update(change)
            with pytest.raises(ManifestMismatchError) as caught:
                store.validate_resume(manifest.run_id, **parameters)
            assert change.keys() <= {item.field for item in caught.value.mismatches}

        changed_tasks = (
            manifest.tasks[0],
            TaskManifest(
                ordinal=1,
                task_id="task-2",
                task_signature="changed-signature",
                knowledge_milestone="knowledge_001",
                benchmark="alfworld",
                split="train",
            ),
        )
        with pytest.raises(ManifestMismatchError) as caught:
            store.validate_resume(
                manifest.run_id,
                config_hash=manifest.config_hash,
                code_commit=manifest.code_commit,
                knowledge_digest=manifest.knowledge_digest,
                tasks=changed_tasks,
            )
        assert "task_manifest_hash" in {item.field for item in caught.value.mismatches}

        store.mark_run_state(manifest.run_id, RunState.RUNNING)
        assert store.mark_task_running(manifest.run_id, "task-1") == 1
        store.mark_task_completed(
            manifest.run_id,
            "task-1",
            trace_id="trace-1",
            result={"benchmark_success": True},
        )
        assert [task.task_id for task in store.tasks_to_run(manifest)] == ["task-2"]
        assert path.read_bytes() == immutable_bytes

        # A crash may leave a durable running row.  Resume must replay it as a
        # new attempt while still skipping only the completed task above.
        assert store.mark_task_running(manifest.run_id, "task-2") == 1
        assert store.mark_task_running(manifest.run_id, "task-2") == 2

        with database.transaction() as connection:
            connection.execute(
                "UPDATE run_tasks SET knowledge_milestone=? WHERE run_id=? AND task_id=?",
                ("tampered", manifest.run_id, "task-2"),
            )
        with pytest.raises(ManifestMismatchError) as caught:
            store.validate_resume(
                manifest.run_id,
                config_hash=manifest.config_hash,
                code_commit=manifest.code_commit,
                knowledge_digest=manifest.knowledge_digest,
                tasks=manifest.tasks,
            )
        assert any(
            item.field.endswith("knowledge_milestone") for item in caught.value.mismatches
        )
    finally:
        database.close()


def test_task_attempt_limit_is_positive_and_fail_closed_before_increment(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite")
    try:
        store = ManifestStore(tmp_path / "manifests", database)
        manifest = _manifest()
        store.persist_before_run(manifest)
        store.mark_run_state(manifest.run_id, RunState.RUNNING)

        assert store.mark_task_running(
            manifest.run_id, "task-1", max_attempts=2
        ) == 1
        store.mark_task_failed(
            manifest.run_id,
            "task-1",
            infrastructure=True,
            result={"attempt": 1},
        )
        assert store.mark_task_running(
            manifest.run_id, "task-1", max_attempts=2
        ) == 2
        store.mark_task_failed(
            manifest.run_id,
            "task-1",
            infrastructure=True,
            result={"attempt": 2},
        )

        with pytest.raises(ProtocolError, match="exhausted max_task_attempts=2"):
            store.mark_task_running(manifest.run_id, "task-1", max_attempts=2)
        row = database.execute(
            "SELECT state,attempt_count FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, "task-1"),
        ).fetchone()
        assert row["state"] == "infrastructure_failed"
        assert row["attempt_count"] == 2

        for invalid in (0, -1, True, 1.5, "3"):
            with pytest.raises(ValueError, match="positive integer"):
                store.mark_task_running(
                    manifest.run_id,
                    "task-2",
                    max_attempts=invalid,  # type: ignore[arg-type]
                )
    finally:
        database.close()


def test_final_maintenance_digest_extends_only_a_matching_completed_task(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite")
    try:
        store = ManifestStore(tmp_path / "manifests", database)
        manifest = _manifest()
        store.persist_before_run(manifest)
        store.mark_task_running(manifest.run_id, "task-1")
        store.mark_task_completed(
            manifest.run_id,
            "task-1",
            trace_id="trace-1",
            result={
                "benchmark_success": True,
                "knowledge_digest_before": "before",
                "knowledge_digest_after": "episode-digest",
            },
        )
        store.update_completed_task_knowledge_digest(
            manifest.run_id,
            "task-1",
            expected_digest="episode-digest",
            new_digest="post-maintenance-digest",
        )
        row = database.execute(
            "SELECT state,result_json FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, "task-1"),
        ).fetchone()
        assert row["state"] == "completed"
        result = json.loads(row["result_json"])
        assert result["knowledge_digest_before"] == "before"
        assert result["knowledge_digest_after"] == "post-maintenance-digest"

        with pytest.raises(ProtocolError, match="digest changed"):
            store.update_completed_task_knowledge_digest(
                manifest.run_id,
                "task-1",
                expected_digest="stale",
                new_digest="wrong",
            )
        with pytest.raises(ProtocolError, match="must be completed"):
            store.update_completed_task_knowledge_digest(
                manifest.run_id,
                "task-2",
                expected_digest="episode-digest",
                new_digest="wrong",
            )
    finally:
        database.close()


def test_task_checkpoint_rolls_back_partial_knowledge_before_resume(tmp_path: Path) -> None:
    data_dir = tmp_path / "run" / "data_v3"
    database = StateDatabase(data_dir / "state.sqlite3")
    store = ManifestStore(tmp_path / "run_manifests", database)
    manifest = _manifest()
    store.persist_before_run(manifest)
    store.mark_run_state(manifest.run_id, RunState.RUNNING)
    store.mark_task_running(manifest.run_id, "task-1")
    artifact = data_dir / "artifacts" / "probe.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"stage":"before"}', encoding="utf-8")
    failure_payload = data_dir / "failure_knowledge" / "provisional" / "probe.json"
    failure_payload.parent.mkdir(parents=True, exist_ok=True)
    failure_payload.write_text('{"stage":"before"}', encoding="utf-8")
    checkpoint = TaskCheckpointStore(tmp_path / "run" / ".task_checkpoint", data_dir)
    checkpoint.create(
        database,
        run_id=manifest.run_id,
        task_id="task-1",
        before_digest="before-digest",
        config_hash=manifest.config_hash,
        code_commit=manifest.code_commit,
    )
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?)", ("partial-write", "present")
        )
    artifact.write_text('{"stage":"partial"}', encoding="utf-8")
    failure_payload.write_text('{"stage":"partial"}', encoding="utf-8")
    database.close()

    assert checkpoint.recover_if_present(
        run_id=manifest.run_id,
        config_hash=manifest.config_hash,
        code_commit=manifest.code_commit,
        resume=True,
    ) == "task-1"
    with StateDatabase(data_dir / "state.sqlite3") as restored:
        assert restored.execute(
            "SELECT value FROM metadata WHERE key='partial-write'"
        ).fetchone() is None
        assert restored.execute(
            "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, "task-1"),
        ).fetchone()["state"] == "running"
    assert artifact.read_text(encoding="utf-8") == '{"stage":"before"}'
    assert failure_payload.read_text(encoding="utf-8") == '{"stage":"before"}'
    assert not checkpoint.root.exists()


def test_artifact_audit_snapshot_uses_index_status_and_current_projection(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite")
    try:
        empty = artifact_audit_snapshot(database)
        assert empty["artifact_index"]["total"] == 0
        assert empty["lifecycle_projection"]["checkpoint"] == 0

        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO artifact_index(artifact_ref,artifact_kind,logical_id,version,"
                "content_hash,status,file_path,schema_version) VALUES(?,?,?,?,?,?,?,?)",
                (
                    "skill://atomic_probe@1.0.0", "atomic", "atomic_probe", "1.0.0",
                    "content-hash", "candidate", str(tmp_path / "artifact.json"), 3,
                ),
            )
            connection.execute(
                "INSERT INTO evidence_events(event_id,schema_version,task_id,trace_id,"
                "occurrence_id,attempt_id,sequence_no,artifact_ref,artifact_kind,event_type,"
                "failure_layer,confidence,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "event-1", 3, "task-1", "trace-1", "occ-1", "attempt-1", 0,
                    "skill://atomic_probe@1.0.0", "atomic", "proposed", "", 1.0, "{}",
                ),
            )
            projection = {
                "artifact_ref": "skill://atomic_probe@1.0.0",
                "artifact_kind": "atomic",
                "event_counts": {"proposed": 1},
                "last_event_rowid": 1,
            }
            connection.execute(
                "INSERT INTO lifecycle_projection(artifact_ref,projection_json,last_event_rowid) "
                "VALUES(?,?,?)",
                (
                    "skill://atomic_probe@1.0.0",
                    json.dumps(projection, sort_keys=True),
                    1,
                ),
            )
            connection.execute(
                "INSERT INTO projection_checkpoints(projection_name,last_event_rowid) "
                "VALUES('lifecycle_v3',1)"
            )
        after = artifact_audit_snapshot(database)
        assert after["artifact_index"]["by_kind_status"] == {
            "atomic": {"candidate": 1}
        }
        assert after["lifecycle_projection"]["event_counts"] == {"proposed": 1}
        assert after["lifecycle_projection"]["records"][0]["projection"] == projection
        growth = artifact_growth_audit(empty, after)
        assert growth["delta"]["artifact_total"] == 1
        assert growth["delta"]["artifact_by_kind"] == {"atomic": 1}
        assert growth["delta"]["artifact_by_kind_status"] == {
            "atomic": {"candidate": 1}
        }
        assert growth["delta"]["lifecycle_event_counts"] == {"proposed": 1}
    finally:
        database.close()


def _trace_one() -> dict[str, object]:
    return {
        "trace_id": "trace-1",
        "schema_version": 3,
        "task": {
            "task_id": "task-1",
            "task_signature": "signature-1",
            "benchmark": "alfworld",
            "task_type": "pick_and_place",
        },
        "planner_audit": {
            "final_outcome": "atomic_composition",
            "requirements_p1r": [{"requirement_id": "r1"}],
            "workflow_p2r": {"steps": ["s1"]},
        },
        "runtime_plan": {
            "source": "atomic_composition",
            "source_composite_ref": None,
            "control_sequence": ["n1", "n2", "n3", "n4", "n5"],
        },
        "started_at": 10.0,
        "ended_at": 12.0,
        "node_records": [
            {"status": "direct_autonomous_success"},
            {"status": "direct_agent_prepared_success"},
            {"status": "agent_completed_before_invocation"},
            {"status": "seeded_success"},
            {"status": "already_satisfied"},
        ],
        "implementation_invocations": [
            {
                "preflight": {"passed": False},
                "result": {"started": False, "completed": False},
            },
            {
                "preflight": {"passed": True},
                "result": {"started": True, "completed": True},
            },
        ],
        "tool_executions": [
            {"result": {"started": True, "completed": True}}
        ],
        "environment_actions": [{"action_id": "a1"}],
        "llm_usage": {
            "events": [
                {
                    "bucket": "planner_p1",
                    "prompt_tokens": 6,
                    "completion_tokens": 4,
                    "total_tokens": 10,
                    "reasoning_tokens": 1,
                    "call_count": 1,
                    "latency_ms": 12.5,
                },
                {
                    "bucket": "runtime_preparation",
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "reasoning_tokens": None,
                    "call_count": 1,
                    "latency_ms": 20,
                },
            ],
            "reconciliation": {"episode_total_tokens": 30},
        },
        "benchmark_success": True,
        "node_contract_success": True,
        "implementation_direct_success": True,
        "graph_self_sufficient_success": True,
        "graph_full_completion": True,
        "task_rescue_required": True,
        "learning_eligible": True,
        "infrastructure_failure": False,
        "extraction_policy": {"classification": "confirmed_capability_gap"},
        "metadata": {
            "cost_usd": 0.3,
            "artifact_growth": {"atomic": 1, "implementation": 1},
            "artifact_lifecycle": {"atomic": {"candidate": 1}},
        },
    }


def _trace_two() -> dict[str, object]:
    return {
        "trace_id": "trace-2",
        "schema_version": 3,
        "task": {
            "task_id": "task-2",
            "task_signature": "signature-2",
            "benchmark": "alfworld",
            "task_type": "clean",
        },
        "planner_audit": {
            "final_outcome": "full_dynamic",
            "fallback_reason": "planner_requirement_uncovered",
        },
        "runtime_plan": {"source": "full_dynamic", "control_sequence": []},
        "started_at": 20.0,
        "ended_at": 21.0,
        "node_records": [],
        "agent_sessions": [
            {"session_id": "dynamic-1", "session_type": "DynamicTaskSession"}
        ],
        "llm_usage": [
            {
                "session_id": "dynamic-1",
                "prompt_tokens": 25,
                "completion_tokens": 15,
                "total_tokens": 40,
                "reasoning_tokens": 5,
                "call_count": 1,
                "latency_ms": 50,
            }
        ],
        "benchmark_success": False,
        "graph_self_sufficient_success": False,
        "task_rescue_required": False,
        "metadata": {"episode_total_tokens": 45},
    }


def test_report_rows_summary_and_all_three_output_formats(tmp_path: Path) -> None:
    traces = [_trace_one(), _trace_two()]
    first = trace_to_row(traces[0])
    second = trace_to_row(traces[1])

    assert tuple(first) == REPORT_COLUMNS
    assert first["completed_node_count"] == 5
    assert first["implementation_preflight_rejected_count"] == 1
    assert first["reasoning_tokens"] is None
    assert first["token_mismatch"] == 0
    assert first["confirmed_capability_gap"] is True
    assert second["runtime_dynamic_total_tokens"] == 40
    assert second["unattributed_total_tokens"] == 0
    assert second["token_mismatch"] == -5

    summary = summarize_traces([first, second])
    assert summary["benchmark_success_rate"] == 0.5
    assert summary["official_alfworld_won_rate"] == 0.5
    assert summary["strict_task_success_rate"] == 0.5
    assert summary["learning_eligible_success_rate"] == 0.5
    assert summary["graph_self_sufficient_success_rate"] == 0.5
    assert summary["direct_autonomous_rate"] == 0.2
    assert summary["direct_agent_prepared_rate"] == 0.2
    assert summary["agent_completed_before_invocation_rate"] == 0.2
    assert summary["seeded_success_rate"] == 0.2
    assert summary["full_dynamic_rate"] == 0.5
    assert summary["task_rescue_rate"] == 0.5
    assert summary["token_mismatch"] == -5
    assert summary["tokens_per_solved_task"] == 30.0
    assert summary["cost_usd"] is None
    assert summary["cost_usd_per_solved_task"] is None
    assert summary["artifact_growth"] == {}
    assert summary["artifact_lifecycle"] == {}

    paths = write_reports(
        traces,
        tmp_path,
        stem="heldout",
        run_artifact_growth={"delta": {"artifact_total": 2}},
        run_artifact_lifecycle={"artifact_index": {"total": 2}},
    )
    records = [
        json.loads(line)
        for line in paths.jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["task_id"] for record in records] == ["task-1", "task-2"]
    with paths.csv.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert tuple(csv_rows[0]) == REPORT_COLUMNS
    assert json.loads(csv_rows[0]["usage_by_bucket"])["planner_p1"]["total_tokens"] == 10
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "Official ALFWorld won" in markdown
    assert "Benchmark success (deprecated alias: strict_task_success)" in markdown
    assert "Learning-eligible success" in markdown
    assert "Per-agent usage buckets" in markdown
    assert "Artifact growth and lifecycle" in markdown
    assert "token mismatch" in markdown.casefold()


def test_unknown_usage_bucket_is_audited_as_unattributed() -> None:
    trace = _trace_two()
    trace["agent_sessions"] = []
    trace["metadata"] = {"episode_total_tokens": 40}
    row = trace_to_row(trace)
    assert row["unattributed_total_tokens"] == 40
    assert row["token_mismatch"] == 0


def test_r2_planner_extractor_and_runtime_diagnostics_are_structured(
    tmp_path: Path,
) -> None:
    trace = _trace_two()
    trace["planner_audit"] = {
        "composite_rejections": [
            {"stage": "retrieval_contract", "reasons": ["goal_contract_exact_mismatch"]},
            {"stage": "plan_validation", "reasons": ["planner_repeat_block_invalid"]},
        ],
        "atomic_search_p1": [{
            "rejection_reasons": [{
                "compatibility": {
                    "failure_codes": ["atomic_required_input_type_unavailable"],
                },
            }],
        }],
        "atomic_search_p1r": [{
            "rejection_reasons": [{
                "compatibility": {
                    "failure_codes": ["atomic_effect_predicate_missing"],
                },
            }],
        }],
    }
    trace["runtime_plan"] = {"source": "stored_composite", "control_sequence": ["step-1"]}
    trace["failures"] = [{"code": "runtime_node_token_budget_exhausted"}]
    trace["node_records"] = [{
        "occurrence_id": "occ-1",
        "step_id": "step-1",
        "atomic_ref": "skill://atomic_probe@1.0.0",
        "status": "seeded_failed",
        "direct_result": {"atomic_effect_passed": False},
        "seeded_result": {"unresolved_roles": ["destination"]},
        "failure": {"code": "runtime_node_token_budget_exhausted"},
    }]
    trace["agent_sessions"] = [{
        "session_id": "seeded-1",
        "session_type": "SeededNodeSession",
        "occurrence_id": "occ-1",
    }]
    trace["agent_turns"] = [{"session_id": "seeded-1"}]
    trace["runtime_spans"] = [{
        "kind": "seeded",
        "occurrence_id": "occ-1",
        "action_start": 2,
        "action_end": 4,
    }]
    trace["task_progress_records"] = [{
        "revision": 3,
        "snapshot": {"revision": 3, "progress_digest": "progress-3"},
    }]
    trace["environment_actions"] = [{
        "accepted": True,
        "new_revision": 4,
    }]
    trace["metadata"] = {
        "extraction": {
            "attempted": True,
            "stage": "e2",
            "prepared": False,
            "applied": False,
            "error_code": "extractor_e2_new_edge_selection_invalid",
            "e1_proposed": 2,
            "e1_validated": 2,
            "e1_rejected": 0,
            "e1_contract_coverage_passed": True,
            "e2_attempted": True,
            "e2_selected_existing_edges": 1,
            "e2_selected_new_edges": 0,
        },
    }

    row = trace_to_row(trace)
    assert row["p0_rejection_stage_distribution"] == {
        "plan_validation": 1,
        "retrieval_contract": 1,
    }
    assert row["atomic_contract_mismatch_reason_distribution_p1"] == {
        "atomic_required_input_type_unavailable": 1,
    }
    assert row["atomic_contract_mismatch_reason_distribution_p1r"] == {
        "atomic_effect_predicate_missing": 1,
    }
    assert row["extraction_stage"] == "e2"
    assert row["e1_contract_coverage_passed"] is True
    diagnostic = row["runtime_failure_diagnostic"]
    assert diagnostic["exhaustion_scope"] == "node"
    assert diagnostic["occurrence_id"] == "occ-1"
    assert diagnostic["seeded"]["turn_count"] == 1
    assert diagnostic["seeded"]["environment_action_count"] == 2
    assert diagnostic["binding"]["unresolved_roles"] == ["destination"]
    assert diagnostic["progress"]["actions_since_last_progress_change"] == 1

    summary = summarize_traces([row])
    assert summary["extractor_e2_content_rejection_count"] == 1
    assert summary["p0_rejection_stage_distribution"]["plan_validation"] == 1
    markdown = write_reports([trace], tmp_path, stem="diagnostic").markdown.read_text(
        encoding="utf-8"
    )
    assert "Extraction diagnostics" in markdown
    assert "Failed task diagnostics" in markdown
    assert "runtime_node_token_budget_exhausted" in markdown


def test_r22_failure_extractor_diagnostics_use_formal_rejection_and_usage(
    tmp_path: Path,
) -> None:
    exhausted_f2 = _trace_two()
    exhausted_f2["trace_id"] = "trace-r22-f2"
    exhausted_f2["task"]["task_id"] = "task-r22-f2"
    exhausted_f2["llm_usage"] = [
        {
            "bucket": "failure_extractor_f1",
            "prompt_tokens": 90000,
            "completion_tokens": 30000,
            "total_tokens": 120000,
            "reasoning_tokens": 20000,
            "call_count": 1,
            "latency_ms": 20.0,
        },
        {
            "bucket": "failure_extractor_f2",
            "prompt_tokens": 140000,
            "completion_tokens": 10000,
            "total_tokens": 150000,
            "reasoning_tokens": 8000,
            "call_count": 1,
            "latency_ms": 30.0,
        },
    ]
    exhausted_f2["metadata"] = {
        "episode_total_tokens": 270000,
        "failure_extractor_metrics": {
            "failure_extractor_f1_input_event_count": 12,
            "failure_extractor_f1_prompt_chars": 4000,
            "failure_extractor_f1_prompt_bytes": 4200,
            "failure_extractor_f1_tokens": 999999,
            "failure_extractor_f1_provider_call_count": 99,
            "failure_extractor_f2_span_count": 3,
            "failure_extractor_f2_source_event_count": 7,
            "failure_extractor_f2_prompt_chars": 2500,
            "failure_extractor_f2_prompt_bytes": 2700,
            "failure_extractor_f2_tokens": 999999,
            "failure_extractor_f2_provider_call_count": 99,
            "failure_extractor_remaining_budget_before_f2": 142144,
        },
    }
    exhausted_f2["failure_extraction"] = {
        "rejection": {
            "code": "failure_extractor_budget_exhausted",
            "stage": "f2",
            "source_code": "extractor_token_budget_exhausted",
        },
    }

    exhausted_row = trace_to_row(exhausted_f2)
    assert exhausted_row["failure_extractor_f1_input_event_count"] == 12
    assert exhausted_row["failure_extractor_f1_prompt_chars"] == 4000
    assert exhausted_row["failure_extractor_f1_prompt_bytes"] == 4200
    assert exhausted_row["failure_extractor_f1_tokens"] == 120000
    assert exhausted_row["failure_extractor_f2_span_count"] == 3
    assert exhausted_row["failure_extractor_f2_source_event_count"] == 7
    assert exhausted_row["failure_extractor_f2_prompt_chars"] == 2500
    assert exhausted_row["failure_extractor_f2_prompt_bytes"] == 2700
    assert exhausted_row["failure_extractor_f2_tokens"] == 150000
    assert exhausted_row["failure_extractor_f1_provider_call_count"] == 1
    assert exhausted_row["failure_extractor_f2_provider_call_count"] == 1
    assert exhausted_row["failure_extractor_budget_exhausted_count"] == 1
    assert exhausted_row["failure_extractor_skipped_after_budget_count"] == 0
    assert (
        exhausted_row[
            "failure_extractor_usage_persisted_after_rejection_count"
        ]
        == 1
    )
    assert (
        exhausted_row["failure_extractor_remaining_budget_before_f2"]
        == 142144
    )

    skipped_f2 = copy.deepcopy(exhausted_f2)
    skipped_f2["trace_id"] = "trace-r22-skipped"
    skipped_f2["task"]["task_id"] = "task-r22-skipped"
    skipped_f2["llm_usage"] = [exhausted_f2["llm_usage"][0]]
    skipped_f2["metadata"]["episode_total_tokens"] = 120000
    skipped_f2["metadata"]["failure_extractor_metrics"][
        "failure_extractor_remaining_budget_before_f2"
    ] = 0
    skipped_f2["failure_extraction"]["rejection"][
        "stage"
    ] = "f2_not_started_no_remaining_budget"
    skipped_row = trace_to_row(skipped_f2)
    assert skipped_row["failure_extractor_f2_tokens"] == 0
    assert skipped_row["failure_extractor_f2_provider_call_count"] == 0
    assert skipped_row["failure_extractor_budget_exhausted_count"] == 1
    assert skipped_row["failure_extractor_skipped_after_budget_count"] == 1
    assert (
        skipped_row[
            "failure_extractor_usage_persisted_after_rejection_count"
        ]
        == 1
    )

    exact_cap = copy.deepcopy(skipped_f2)
    exact_cap["trace_id"] = "trace-r22-exact-cap"
    exact_cap["task"]["task_id"] = "task-r22-exact-cap"
    exact_cap["llm_usage"][0] = {
        **exact_cap["llm_usage"][0],
        "prompt_tokens": 250000,
        "completion_tokens": 12144,
        "total_tokens": 262144,
    }
    exact_cap["metadata"]["episode_total_tokens"] = 262144
    exact_cap["failure_extraction"] = {"rejection": {}}
    exact_row = trace_to_row(exact_cap)
    assert exact_row["failure_extractor_f1_tokens"] == 262144
    assert exact_row["failure_extractor_budget_exhausted_count"] == 0
    assert exact_row["failure_extractor_skipped_after_budget_count"] == 0
    assert (
        exact_row[
            "failure_extractor_usage_persisted_after_rejection_count"
        ]
        == 0
    )

    summary = summarize_traces([exhausted_row, skipped_row, exact_row])
    assert summary["failure_extractor_f1_tokens"] == 502144
    assert summary["failure_extractor_f2_tokens"] == 150000
    assert summary["failure_extractor_f1_provider_call_count"] == 3
    assert summary["failure_extractor_f2_provider_call_count"] == 1
    assert summary["failure_extractor_budget_exhausted_count"] == 2
    assert summary["failure_extractor_skipped_after_budget_count"] == 1
    assert (
        summary["failure_extractor_usage_persisted_after_rejection_count"]
        == 2
    )
    markdown = write_reports(
        [exhausted_f2, skipped_f2, exact_cap], tmp_path, stem="r22",
    ).markdown.read_text(encoding="utf-8")
    assert "R2.2 Failure Extractor diagnostics" in markdown
    assert "failure_extractor_usage_persisted_after_rejection_count" in markdown


def test_failed_runtime_diagnostic_uses_temporal_trace_authority() -> None:
    trace = _trace_two()
    trace["runtime_plan"] = {
        "source": "stored_composite",
        "control_sequence": ["step-2"],
    }
    trace["failures"] = [
        {
            "code": "runtime_semantic_anchor_mismatch",
            "occurrence_id": "occ-1",
            "artifact_refs": ["skill://atomic_old@1.0.0"],
        },
        {
            "code": "runtime_relation_not_grounded",
            "occurrence_id": "occ-2",
            "artifact_refs": ["skill://atomic_target@1.0.0"],
        },
    ]
    # A recoverable failure may belong to a node that later completed.  The
    # last FailureEnvelope still supplies temporal occurrence authority when
    # there is no terminal failed NodeTraceRecord.
    trace["node_records"] = [{
        "occurrence_id": "occ-2",
        "step_id": "step-2",
        "atomic_ref": "skill://atomic_target@1.0.0",
        "status": "direct_autonomous_success",
        "direct_result": {"atomic_effect_passed": True},
        "seeded_result": {},
        "failure": {},
    }]
    trace["binding_changes"] = [
        {
            "occurrence_id": "occ-2",
            "role": "station",
            "current": {
                "status": "grounded",
                "value": "microwave_1",
            },
        },
        {
            "occurrence_id": "occ-2",
            "role": "destination",
            "current": {"status": "grounded", "value": "table_1"},
        },
        {
            "occurrence_id": "occ-2",
            "role": "station",
            "current": {
                "status": "invalidated",
                "value": "microwave_1",
            },
        },
    ]
    trace["task_progress_records"] = [
        {
            "revision": 1,
            "snapshot": {"revision": 2, "progress_digest": "progress-1"},
        },
        {
            "revision": 2,
            "snapshot": {"revision": 19, "progress_digest": "progress-2"},
        },
    ]
    trace["environment_actions"] = [
        {"accepted": True, "new_revision": 18},
        {"accepted": True, "new_revision": 19},
        {"accepted": True, "new_revision": 20},
    ]

    diagnostic = trace_to_row(trace)["runtime_failure_diagnostic"]
    assert diagnostic["occurrence_id"] == "occ-2"
    assert diagnostic["atomic_ref"] == "skill://atomic_target@1.0.0"
    assert diagnostic["last_runtime_failure_code"] == (
        "runtime_relation_not_grounded"
    )
    assert diagnostic["binding"] == {
        "resolved_roles": ["destination"],
        "unresolved_roles": ["station"],
    }
    assert diagnostic["progress"]["last_progress_digest"] == "progress-2"
    assert diagnostic["progress"]["actions_since_last_progress_change"] == 1


def test_strict_success_recoverable_failures_are_not_failed_diagnostics(
    tmp_path: Path,
) -> None:
    trace = _trace_one()
    trace["strict_task_success"] = True
    trace["failures"] = [{
        "code": "runtime_relation_not_grounded",
        "occurrence_id": "occ-recovered",
    }]

    row = trace_to_row(trace)
    assert row["failure_codes"] == ["runtime_relation_not_grounded"]
    assert row["runtime_failure_diagnostic"] == {}
    summary = summarize_traces([row])
    assert summary["failed_task_diagnostics"] == []
    markdown = write_reports(
        [trace], tmp_path, stem="strict_recovered",
    ).markdown.read_text(encoding="utf-8")
    assert "Failed task diagnostics" not in markdown


def _formal_usage_trace(trace_id: str, event_id: str) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "schema_version": 3,
        "task": {
            "task_id": f"task-{trace_id}",
            "task_signature": f"signature-{trace_id}",
            "benchmark": "alfworld",
            "task_type": "maintenance" if "maintenance" in trace_id else "train",
        },
        "planner_audit": {},
        "runtime_plan": {"source": "maintenance", "control_sequence": []},
        "node_records": [],
        "agent_turns": [],
        "llm_usage": [{
            "event_id": event_id,
            "session_id": f"session-{event_id}",
            "turn_index": 0,
            "bucket": "evolution_repair",
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "reasoning_tokens": 0,
            "call_count": 1,
            "latency_ms": 1.0,
            "provider_metadata": {"usage_status": "reported"},
        }],
        "metadata": {},
    }


def test_usage_coverage_accepts_task_and_maintenance_traces_and_rejects_hidden_call() -> None:
    traces = [
        _formal_usage_trace("trace-task", "usage-task"),
        _formal_usage_trace("trace-maintenance", "usage-maintenance"),
    ]
    events = [
        types.SimpleNamespace(event_id="usage-task"),
        types.SimpleNamespace(event_id="usage-maintenance"),
    ]
    assert validate_usage_event_persistence(events, traces) == {
        "in_process_event_count": 2,
        "persisted_event_count": 2,
        "trace_count": 2,
    }
    with pytest.raises(ValueError, match="absent from task/maintenance Traces"):
        validate_usage_event_persistence(
            [*events, types.SimpleNamespace(event_id="usage-hidden")], traces,
        )


def test_auxiliary_maintenance_usage_is_aggregated_without_adding_task_rows(
    tmp_path: Path,
) -> None:
    task = _formal_usage_trace("trace-task", "usage-task")
    task["llm_usage"][0]["bucket"] = "runtime_dynamic"  # type: ignore[index]
    task["llm_usage"][0]["latency_ms"] = 2.0  # type: ignore[index]
    task["benchmark_success"] = True
    task["metadata"] = {"cost_usd": 0.25}
    maintenance = _formal_usage_trace(
        "trace-maintenance", "usage-maintenance",
    )
    maintenance["llm_usage"][0]["latency_ms"] = 7.5  # type: ignore[index]
    maintenance["metadata"] = {"cost_usd": 0.10}

    summary = summarize_traces(
        [task], auxiliary_usage_traces=[maintenance],
    )
    assert summary["task_count"] == 1
    assert summary["solved_task_count"] == 1
    assert summary["total_tokens"] == 10
    assert summary["tokens_per_task"] == 10.0
    assert summary["tokens_per_solved_task"] == 10.0
    assert summary["llm_latency_ms"] == 9.5
    assert summary["llm_latency_ms_per_solved_task"] == 9.5
    assert summary["cost_usd"] == 0.35
    assert summary["cost_usd_per_solved_task"] == 0.35
    assert summary["usage_by_bucket"]["runtime_dynamic"]["total_tokens"] == 5
    assert summary["usage_by_bucket"]["evolution_repair"]["total_tokens"] == 5

    paths = write_reports(
        [task], tmp_path, stem="train",
        auxiliary_usage_traces=[maintenance],
    )
    rows = [
        json.loads(line)
        for line in paths.jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["trace_id"] for row in rows] == ["trace-task"]
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "| Tasks | 1 |" in markdown
    assert "| evolution_repair | 1 | 3 | 2 | 5 | 0 | 7.5 |" in markdown

    unknown_cost = copy.deepcopy(maintenance)
    unknown_cost["metadata"] = {}
    fail_closed = summarize_traces(
        [task], auxiliary_usage_traces=[unknown_cost],
    )
    assert fail_closed["cost_usd"] is None
    assert fail_closed["cost_usd_per_solved_task"] is None


def test_formal_usage_rejects_each_trace_mismatch_without_cross_trace_cancellation() -> None:
    positive = _formal_usage_trace("trace-positive", "usage-positive")
    negative = _formal_usage_trace("trace-negative", "usage-negative")
    positive["metadata"] = {"usage_reconciliation": {"episode_total_tokens": 4}}
    negative["metadata"] = {"usage_reconciliation": {"episode_total_tokens": 6}}
    with pytest.raises(ValueError, match="trace-positive.*non-zero token_mismatch"):
        validate_formal_usage([positive, negative])
