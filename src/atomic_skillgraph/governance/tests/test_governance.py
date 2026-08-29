from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest


# The repository is assembled in vertical slices.  These governance-only tests
# avoid executing the top-level package facade while system.py is built by a
# different slice.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if "atomic_skillgraph" not in sys.modules:
    package = types.ModuleType("atomic_skillgraph")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["atomic_skillgraph"] = package

from atomic_skillgraph.core.errors import FailureLayer
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.governance.credit import (
    CreditAssigner,
    CreditAssignmentError,
    CreditAttempt,
    CreditOutcome,
    CreditTrace,
)
from atomic_skillgraph.governance.ledger import (
    EvidenceConflictError,
    EvidenceEvent,
    EvidenceEventType,
    EvidenceLedger,
)
from atomic_skillgraph.governance.lifecycle import (
    CandidateUsePolicy,
    LifecycleController,
    LifecyclePolicy,
)
from atomic_skillgraph.governance.projections import ArtifactStats, LifecycleProjection
from atomic_skillgraph.knowledge.database import StateDatabase


def _event(**overrides: object) -> EvidenceEvent:
    values = {
        "task_id": "task-1",
        "trace_id": "trace-1",
        "occurrence_id": "occ-1",
        "attempt_id": "attempt-1",
        "sequence_no": 0,
        "artifact_ref": "skill://atomic-a@1.0.0",
        "artifact_kind": "atomic",
        "event": EvidenceEventType.VALIDATED,
    }
    values.update(overrides)
    return EvidenceEvent.create(**values)


def _index_artifact(
    database: StateDatabase,
    *,
    ref: str,
    kind: str,
    logical_id: str,
    status: str,
    version: str = "1.0.0",
) -> None:
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO artifact_index(artifact_ref,artifact_kind,logical_id,version,"
            "content_hash,status,file_path,schema_version) VALUES(?,?,?,?,?,?,?,3)",
            (ref, kind, logical_id, version, "hash", status, "unused.json"),
        )


def test_ledger_append_is_exactly_once_and_conflicts_fail_closed(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        ledger = EvidenceLedger(database)
        event = _event(metadata={"witness": "v1"})

        first = ledger.append_transaction([event, event])
        assert first.inserted_count == 1
        assert first.duplicate_count == 1
        assert ledger.count() == 1

        replay = ledger.append_transaction([event])
        assert replay.inserted_count == 0
        assert replay.duplicate_count == 1
        assert ledger.count() == 1

        conflict = replace(event, metadata={"witness": "different"})
        new_event = _event(
            trace_id="trace-new",
            attempt_id="attempt-new",
            sequence_no=1,
        )
        with pytest.raises(EvidenceConflictError):
            ledger.append_transaction([new_event, conflict])
        assert ledger.count() == 1
        assert ledger.get(new_event.event_id) is None


def test_credit_assigner_enforces_started_and_failure_layer_boundaries() -> None:
    assigner = CreditAssigner()
    parameter_error = CreditAttempt(
        artifact_ref="skill://impl-a@1.0.0",
        artifact_kind="implementation",
        occurrence_id="occ-1",
        attempt_id="attempt-parameter",
        sequence_no=0,
        selected=True,
        preflight_rejected=True,
        failure_layer=FailureLayer.RUNTIME_BINDING,
    )
    mapping_error = CreditAttempt(
        artifact_ref="skill://impl-a@1.0.0",
        artifact_kind="implementation",
        occurrence_id="occ-1",
        attempt_id="attempt-mapping",
        sequence_no=1,
        selected=True,
        preflight_rejected=True,
        failure_layer=FailureLayer.IMPLEMENTATION,
    )
    tool_failure = CreditAttempt(
        artifact_ref="tool://tool-a@1.0.0",
        artifact_kind="tool",
        occurrence_id="occ-1",
        attempt_id="attempt-tool",
        sequence_no=2,
        selected=True,
        started=True,
        outcome=CreditOutcome.DIRECT_FAILURE,
        failure_layer=FailureLayer.TOOL,
    )
    events = assigner.assign(
        CreditTrace("task-1", "trace-1", (parameter_error, mapping_error, tool_failure))
    )

    parameter_events = [event for event in events if event.attempt_id == "attempt-parameter"]
    assert {event.event for event in parameter_events} == {
        EvidenceEventType.SELECTED,
        EvidenceEventType.PREFLIGHT_REJECTED,
    }
    assert not parameter_events[-1].metadata["intrinsic_failure"]

    mapping_events = [event for event in events if event.attempt_id == "attempt-mapping"]
    assert mapping_events[-1].metadata["intrinsic_failure"]

    tool_events = [event for event in events if event.attempt_id == "attempt-tool"]
    assert EvidenceEventType.EXECUTION_STARTED in {event.event for event in tool_events}
    assert EvidenceEventType.DIRECT_FAILURE in {event.event for event in tool_events}
    assert tool_events[-1].metadata["intrinsic_failure"]

    with pytest.raises(CreditAssignmentError):
        CreditAttempt(
            artifact_ref="tool://tool-a@1.0.0",
            artifact_kind="tool",
            occurrence_id="occ",
            attempt_id="not-started",
            sequence_no=0,
            outcome=CreditOutcome.DIRECT_SUCCESS,
        )

    assert assigner.assign(
        CreditTrace("task", "infra", (tool_failure,), infrastructure_failure=True)
    ) == []


def test_credit_assigner_consumes_standard_trace_record_shape() -> None:
    trace = {
        "trace_id": "trace-standard",
        "task": {"task_id": "task-standard"},
        "infrastructure_failure": False,
        "implementation_invocations": [
            {
                "attempt_id": "impl-attempt",
                "occurrence_id": "occ-1",
                "implementation_ref": "skill://impl@1.0.0",
                "preflight": {"passed": True},
                "result": {
                    "started": True,
                    "completed": True,
                    "atomic_effect_passed": True,
                    "failure_layer": "",
                },
                "span_id": "span-impl",
            }
        ],
        "tool_executions": [
            {
                "attempt_id": "tool-attempt",
                "occurrence_id": "occ-1",
                "tool_ref": "tool://tool@1.0.0",
                "result": {"started": True, "completed": True, "failure_layer": ""},
                "span_id": "span-tool",
            }
        ],
        "node_records": [
            {
                "occurrence_id": "occ-1",
                "atomic_ref": "skill://atomic@1.0.0",
                "status": "direct_agent_prepared_success",
                "direct_result": {"started": True},
            }
        ],
        "runtime_plan": {"source_composite_ref": "skill://composite@1.0.0"},
        "graph_self_sufficient_success": True,
        "task_rescue_required": False,
        "benchmark_success": True,
        "learning_eligible": True,
    }
    events = CreditAssigner().assign(trace)
    outcomes = {(event.artifact_kind, event.event) for event in events}
    assert ("implementation", EvidenceEventType.DIRECT_SUCCESS) in outcomes
    assert ("tool", EvidenceEventType.DIRECT_SUCCESS) in outcomes
    assert ("atomic", EvidenceEventType.DIRECT_SUCCESS) in outcomes
    assert ("composite", EvidenceEventType.SELF_SUFFICIENT_SUCCESS) in outcomes


def test_projection_checkpoint_rebuild_and_controller_promotion(tmp_path: Path) -> None:
    ref = "skill://impl-a@1.0.0"
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index_artifact(
            database,
            ref=ref,
            kind="implementation",
            logical_id="impl-a",
            status=SkillStatus.CANDIDATE.value,
        )
        ledger = EvidenceLedger(database)
        assigner = CreditAssigner()
        all_events = []
        for index in range(2):
            attempt = CreditAttempt(
                artifact_ref=ref,
                artifact_kind="implementation",
                occurrence_id=f"occ-{index}",
                attempt_id=f"attempt-{index}",
                sequence_no=0,
                validated=index == 0,
                selected=True,
                started=True,
                outcome=CreditOutcome.DIRECT_SUCCESS,
                metadata={"cost_usd": 0.25, "latency_ms": 10},
            )
            events = assigner.assign(
                CreditTrace(f"task-{index}", f"trace-{index}", (attempt,))
            )
            ledger.append_transaction(events)
            all_events.extend(events)

        projection = LifecycleProjection(database, ledger)
        consumed = projection.consume(all_events)
        assert consumed.processed_count == len(all_events)
        assert projection.checkpoint == ledger.max_rowid()
        stats = projection.stats(ref)
        assert stats.started_count == 2
        assert stats.direct_success_count == 2
        assert stats.independent_direct_success_count == 2
        assert stats.cost_sum == pytest.approx(0.5)
        assert stats.latency_sum == pytest.approx(20.0)

        digest_before = projection.digest()
        rebuilt = projection.rebuild()
        assert rebuilt.processed_count == ledger.count()
        assert projection.digest() == digest_before
        assert projection.consume_new_events().processed_count == 0

        controller = LifecycleController(database, projection)
        review = controller.review([ref])
        assert review.changed_count == 1
        row = database.execute(
            "SELECT status FROM artifact_index WHERE artifact_ref=?", (ref,)
        ).fetchone()
        assert row["status"] == SkillStatus.ACTIVE.value
        recommended = database.execute(
            "SELECT artifact_ref FROM recommended_pointers WHERE logical_id='impl-a'"
        ).fetchone()
        assert recommended["artifact_ref"] == ref


def test_four_lifecycle_policies_and_online_frozen_candidate_rules() -> None:
    policy = LifecyclePolicy()

    atomic = ArtifactStats(
        "skill://atomic@1.0.0",
        "atomic",
        validated_count=1,
        success_task_ids=["t1", "t2"],
    )
    assert policy.review_atomic(atomic.artifact_ref, SkillStatus.CANDIDATE, atomic).next_status == "active"

    implementation = ArtifactStats(
        "skill://impl@1.0.0",
        "implementation",
        validated_count=1,
        event_task_ids={"direct_success": ["t1", "t2"]},
    )
    assert (
        policy.review_implementation(
            implementation.artifact_ref, SkillStatus.CANDIDATE, implementation
        ).next_status
        == "active"
    )

    tool = ArtifactStats(
        "tool://tool@1.0.0",
        "tool",
        validated_count=1,
        started_count=5,
        event_counts={"direct_success": 5},
        event_task_ids={"direct_success": ["t1", "t2", "t3", "t4", "t5"]},
        preferred_utility_evidence_count=1,
    )
    assert policy.review_tool(tool.artifact_ref, ToolStatus.CANDIDATE, tool).next_status == "active"
    assert policy.review_tool(tool.artifact_ref, ToolStatus.ACTIVE, tool).next_status == "preferred"

    composite = ArtifactStats(
        "skill://composite@1.0.0",
        "composite",
        validated_count=1,
        event_task_ids={"self_sufficient_success": ["t1", "t2"]},
    )
    assert (
        policy.review_composite(composite.artifact_ref, SkillStatus.CANDIDATE, composite).next_status
        == "active"
    )

    no_random_exploration = CandidateUsePolicy(exploration_quota=0.0, seed=7)
    common = {
        "artifact_ref": atomic.artifact_ref,
        "artifact_kind": "atomic",
        "task_id": "task",
    }
    assert not no_random_exploration.allows(
        **common,
        status=SkillStatus.CANDIDATE,
        mode=RuntimeMode.ONLINE,
        reliable_active_available=True,
    )
    assert no_random_exploration.allows(
        **common,
        status=SkillStatus.CANDIDATE,
        mode=RuntimeMode.ONLINE,
        reliable_active_available=False,
    )
    assert not no_random_exploration.allows(
        **common,
        status=SkillStatus.CANDIDATE,
        mode=RuntimeMode.FROZEN,
    )
    assert no_random_exploration.allows(
        **common,
        status=SkillStatus.ACTIVE,
        mode=RuntimeMode.FROZEN,
    )
