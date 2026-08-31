from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from atomic_skillgraph.knowledge import (
    FailureExperience,
    FailureExperienceStatus,
    FailureKnowledgeStore,
    ProvisionalAtomicRecord,
    ProvisionalStatus,
    StateDatabase,
    provisional_ref_for,
)
from atomic_skillgraph.knowledge.database import (
    STATE_PATCH_LEVEL,
    STATE_PATCH_MISMATCH,
)


def _provisional(
    *,
    signature: str = "contract123",
    task_id: str = "task-1",
    trace_id: str = "trace-1",
    status: ProvisionalStatus = ProvisionalStatus.TRIAL_READY,
) -> ProvisionalAtomicRecord:
    return ProvisionalAtomicRecord(
        provisional_ref=provisional_ref_for(signature),
        contract_signature=signature,
        canonical_intent="acquire_target_object",
        atomic_contract={
            "summary": "acquire target object",
            "inputs": [{"name": "object", "semantic_type": "entity"}],
            "outputs": [{"name": "held_object", "semantic_type": "entity"}],
            "preconditions": [],
            "effects": [{
                "predicate": "agent.holds",
                "args": {"object": "$object"},
                "cardinality": 1,
                "distinct_by": "",
            }],
            "validator_spec": {"validator_id": "harness_atomic_effect"},
        },
        seeded_guideline={
            "intent": "establish the declared local Effect",
            "parameter_flow": "preserve declared role identity",
        },
        harness_profile="alfworld_v3",
        source_trace_id=trace_id,
        source_task_id=task_id,
        source_span={"event_start": 1, "event_end": 2},
        source_replay={"passed": True, "witness_refs": ["witness-1"]},
        aligned_plan_step_ids=("step-acquire",),
        progress_relation="consumed_prerequisite",
        status=status,
        metadata={"source": "failure_extractor_f2"},
    )


def _experience(
    *,
    experience_id: str = "failure-exp-1",
    task_id: str = "task-1",
    trace_id: str = "trace-1",
) -> FailureExperience:
    return FailureExperience(
        experience_id=experience_id,
        cluster_signature="cluster-abc",
        divergence_signature="divergence-def",
        harness_profile="alfworld_v3",
        requirement_instance_ids=("repeat::0::acquire", "repeat::0::place"),
        validated_prefix_step_ids=("step-acquire",),
        first_unrecovered_divergence={
            "kind": "unsatisfied_precondition",
            "failed_plan_step_template": "place object at the shared destination",
        },
        remaining_requirement_instance_ids=("repeat::0::place",),
        negative_suffix_summary={
            "summary": "do not retry the same no-progress placement suffix",
        },
        avoid_pattern_codes=("repeated_no_progress",),
        provisional_atomic_refs=(provisional_ref_for("contract123"),),
        status=FailureExperienceStatus.OBSERVED,
        support_trace_ids=(trace_id,),
        metadata={"source_task_id": task_id},
    )


def test_fresh_database_has_v31_failure_tables_indexes_and_patch(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in database.rows("SELECT key,value FROM metadata")
        }
        assert metadata["schema_version"] == "3"
        assert metadata["state_patch_level"] == STATE_PATCH_LEVEL
        tables = {
            str(row["name"])
            for row in database.rows(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "provisional_artifacts",
            "failure_experiences",
            "cold_start_evidence",
        }.issubset(tables)
        assert database.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='provisional_contract_status'"
        ).fetchone()
        assert database.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='failure_experience_cluster_status'"
        ).fetchone()


def test_old_schema_v3_bank_fails_with_frozen_patch_message(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
        "INSERT INTO metadata(key,value) VALUES('schema_version','3');"
    )
    connection.close()

    with pytest.raises(RuntimeError, match=re.escape(STATE_PATCH_MISMATCH)):
        StateDatabase(path)


def test_provisional_alignment_integrity_views_and_read_counter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_v3"
    with StateDatabase(data_dir / "state.sqlite3") as database:
        store = FailureKnowledgeStore(data_dir, database)
        first = store.upsert_provisional(_provisional())
        assert first.status is ProvisionalStatus.TRIAL_READY
        assert store.failure_side_read_count == 0

        aligned = store.upsert_provisional(
            _provisional(task_id="task-2", trace_id="trace-2")
        )
        assert aligned.provisional_ref == first.provisional_ref
        assert database.execute(
            "SELECT COUNT(*) AS count FROM provisional_artifacts"
        ).fetchone()["count"] == 1
        assert database.execute(
            "SELECT COUNT(DISTINCT task_id) AS count FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type='source_replay_passed'",
            (first.provisional_ref,),
        ).fetchone()["count"] == 2

        rows = store.list_provisionals({ProvisionalStatus.TRIAL_READY})
        assert [item.provisional_ref for item in rows] == [first.provisional_ref]
        assert store.failure_side_read_count == 1
        candidate = store.provisional_candidate_view(rows[0])
        assert candidate.independent_source_replay_support == 2
        assert not hasattr(candidate, "source_span")
        assert not hasattr(candidate, "source_trace_id")
        assert "tool_body" not in json.dumps(candidate.seeded_guideline)

        store.verify_all()
        assert store.failure_side_read_count == 1
        row = database.execute(
            "SELECT file_path,content_hash FROM provisional_artifacts"
        ).fetchone()
        payload = json.loads(Path(row["file_path"]).read_text(encoding="utf-8"))
        assert payload["state_patch_level"] == "3.1"


def test_provisional_trial_support_suppression_and_promotion_are_isolated(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data_v3"
    with StateDatabase(data_dir / "state.sqlite3") as database:
        store = FailureKnowledgeStore(data_dir, database)

        supported = store.upsert_provisional(_provisional(signature="supported"))
        supported = store.record_provisional_trial(
            supported.provisional_ref,
            task_id="task-2",
            trace_id="trace-2",
            started=True,
            local_effect_passed=True,
            strict_task_success=False,
        )
        assert supported.status is ProvisionalStatus.TRIAL_SUPPORTED
        assert database.execute(
            "SELECT COUNT(*) AS count FROM artifact_index"
        ).fetchone()["count"] == 0

        promoted = store.promote_provisional(
            supported.provisional_ref,
            (
                "skill://atomic_acquire@1.0.0",
                "skill://impl_acquire@1.0.0",
                "tool://acquire@1.0.0",
            ),
            task_id="task-3",
            trace_id="trace-3",
        )
        assert promoted.status is ProvisionalStatus.PROMOTED
        assert len(promoted.promoted_verified_refs) == 3
        assert store.list_provisionals({ProvisionalStatus.TRIAL_READY}) == []

        suppressed = store.upsert_provisional(_provisional(signature="suppressed"))
        for index in range(3):
            suppressed = store.record_provisional_trial(
                suppressed.provisional_ref,
                task_id=f"failure-task-{index}",
                trace_id=f"failure-trace-{index}",
                started=True,
                local_effect_passed=False,
                suppress_after=3,
            )
        assert suppressed.status is ProvisionalStatus.SUPPRESSED

        neutral = store.upsert_provisional(_provisional(signature="neutral"))
        for index in range(3):
            neutral = store.record_provisional_trial(
                neutral.provisional_ref,
                task_id=f"infra-task-{index}",
                trace_id=f"infra-trace-{index}",
                started=True,
                local_effect_passed=False,
                infrastructure_failure=True,
                suppress_after=3,
            )
        assert neutral.status is ProvisionalStatus.TRIAL_READY


def test_failure_experience_independent_support_sanitized_view_and_resolution(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data_v3"
    with StateDatabase(data_dir / "state.sqlite3") as database:
        store = FailureKnowledgeStore(
            data_dir,
            database,
            experience_confirm_independent_tasks=2,
        )
        observed = store.upsert_failure_experience(_experience())
        assert observed.status is FailureExperienceStatus.OBSERVED
        assert store.upsert_failure_experience(_experience()) == observed

        confirmed = store.upsert_failure_experience(
            _experience(
                experience_id="equivalent-proposal-id",
                task_id="task-2",
                trace_id="trace-2",
            )
        )
        assert confirmed.experience_id == observed.experience_id
        assert confirmed.status is FailureExperienceStatus.CONFIRMED
        row = database.execute(
            "SELECT support_count FROM failure_experiences WHERE experience_id=?",
            (observed.experience_id,),
        ).fetchone()
        assert row["support_count"] == 2

        views = store.list_failure_experiences({FailureExperienceStatus.CONFIRMED})
        assert len(views) == 1
        view = store.failure_experience_view(views[0])
        encoded = json.dumps(view.__dict__, ensure_ascii=False)
        assert "source_actions" not in encoded
        assert view.executable is False
        assert view.warning.startswith("FAILED HISTORICAL METHOD")

        resolved = store.resolve_failure_experience(
            observed.experience_id,
            task_id="task-3",
            trace_id="trace-success",
        )
        assert resolved.status is FailureExperienceStatus.RESOLVED
        assert resolved.resolved_by_trace_ids == ("trace-success",)
        assert store.list_failure_experiences({
            FailureExperienceStatus.OBSERVED,
            FailureExperienceStatus.CONFIRMED,
        }) == []

        with pytest.raises(ValueError, match="concrete source terms"):
            replace(
                _experience(experience_id="unsafe"),
                negative_suffix_summary={
                    "source_actions": ["put apple 1 on countertop 1"],
                },
            )


def test_failure_payload_hash_tampering_fails_closed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_v3"
    with StateDatabase(data_dir / "state.sqlite3") as database:
        store = FailureKnowledgeStore(data_dir, database)
        record = store.upsert_provisional(_provisional())
        row = database.execute(
            "SELECT file_path FROM provisional_artifacts WHERE provisional_ref=?",
            (record.provisional_ref,),
        ).fetchone()
        path = Path(row["file_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["record"]["canonical_intent"] = "tampered_intent"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(RuntimeError, match="hash mismatch"):
            store.verify_all()
