from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.governance.ledger import (
    EvidenceEvent,
    EvidenceEventType,
    EvidenceLedger,
)
from atomic_skillgraph.governance.lifecycle import (
    LifecycleController,
    LifecyclePolicy,
    LifecycleThresholds,
)
from atomic_skillgraph.governance.projections import ArtifactStats, LifecycleProjection
from atomic_skillgraph.knowledge.database import StateDatabase


COMPOSITE_REF = "skill://composite-r8@1.0.0"
ATOMIC_A_REF = "skill://atomic-r8-a@1.0.0"
ATOMIC_B_REF = "skill://atomic-r8-b@1.0.0"


def _evidence(
    *,
    task_id: str,
    index: int,
    event: EvidenceEventType,
    artifact_ref: str = COMPOSITE_REF,
) -> EvidenceEvent:
    return EvidenceEvent.create(
        task_id=task_id,
        trace_id=f"trace-{task_id}-{index}",
        occurrence_id=f"occ-{index}",
        attempt_id=f"attempt-{task_id}-{index}",
        sequence_no=index,
        artifact_ref=artifact_ref,
        artifact_kind="composite",
        event=event,
    )


def _atomic_evidence(
    *,
    task_id: str,
    index: int,
    event: EvidenceEventType,
    intrinsic_failure: bool = False,
) -> EvidenceEvent:
    return EvidenceEvent.create(
        task_id=task_id,
        trace_id=f"atomic-trace-{task_id}-{index}",
        occurrence_id=f"atomic-occ-{index}",
        attempt_id=f"atomic-attempt-{task_id}-{index}",
        sequence_no=index,
        artifact_ref=ATOMIC_A_REF,
        artifact_kind="atomic",
        event=event,
        failure_layer="atomic" if intrinsic_failure else "",
        metadata={"intrinsic_failure": intrinsic_failure},
    )


def _index_artifact(
    database: StateDatabase,
    tmp_path: Path,
    *,
    ref: str,
    kind: str,
    status: SkillStatus,
    payload: dict[str, object] | None = None,
) -> None:
    logical_id = ref.removeprefix("skill://").rsplit("@", 1)[0]
    path = tmp_path / f"{logical_id}.json"
    path.write_text(json.dumps(payload or {}), encoding="utf-8")
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO artifact_index(artifact_ref,artifact_kind,logical_id,version,"
            "content_hash,status,file_path,schema_version) VALUES(?,?,?,?,?,?,?,3)",
            (
                ref,
                kind,
                logical_id,
                "1.0.0",
                "test-hash",
                status.value,
                str(path),
            ),
        )


def _composite_payload(*child_refs: str) -> dict[str, object]:
    return {
        "occurrences": [
            {
                "occurrence_id": f"occ-{index}",
                "step_id": f"step-{index}",
                "node_ref": {
                    "logical_id": ref.removeprefix("skill://").rsplit("@", 1)[0],
                    "version": ref.rsplit("@", 1)[1],
                },
            }
            for index, ref in enumerate(child_refs)
        ]
    }


def _add_contains(database: StateDatabase, child_ref: str, index: int = 0) -> None:
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO graph_edges(edge_id,source_ref,target_ref,relation,metadata_json) "
            "VALUES(?,?,?,?,?)",
            (f"edge-{index}", COMPOSITE_REF, child_ref, "contains", "{}"),
        )


def _promotion_projection(database: StateDatabase) -> LifecycleProjection:
    ledger = EvidenceLedger(database)
    ledger.append_transaction(
        [
            _evidence(task_id="success-a", index=0, event=EvidenceEventType.SELF_SUFFICIENT_SUCCESS),
            _evidence(task_id="success-b", index=1, event=EvidenceEventType.SELF_SUFFICIENT_SUCCESS),
            _evidence(task_id="success-a", index=0, event=EvidenceEventType.DEPLOYMENT_SUCCESS),
            _evidence(task_id="success-b", index=1, event=EvidenceEventType.DEPLOYMENT_SUCCESS),
        ]
    )
    projection = LifecycleProjection(database, ledger)
    projection.consume_new_events()
    return projection


def _setup_closure(
    database: StateDatabase,
    tmp_path: Path,
    child_statuses: tuple[SkillStatus, ...],
    *,
    relation_count: int | None = None,
) -> LifecycleProjection:
    refs = (ATOMIC_A_REF, ATOMIC_B_REF)[: len(child_statuses)]
    for ref, status in zip(refs, child_statuses):
        _index_artifact(
            database,
            tmp_path,
            ref=ref,
            kind="atomic",
            status=status,
        )
    _index_artifact(
        database,
        tmp_path,
        ref=COMPOSITE_REF,
        kind="composite",
        status=SkillStatus.CANDIDATE,
        payload=_composite_payload(*refs),
    )
    if relation_count is None:
        relation_count = len(refs)
    for index, ref in enumerate(refs[:relation_count]):
        _add_contains(database, ref, index)
    return _promotion_projection(database)


def test_r8_independent_selected_tasks_and_replay_are_deterministic(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        ledger = EvidenceLedger(database)
        events = [
            *(
                _evidence(task_id="task-a", index=index, event=EvidenceEventType.SELECTED)
                for index in range(3)
            ),
            *(
                _evidence(task_id="task-b", index=index + 3, event=EvidenceEventType.SELECTED)
                for index in range(2)
            ),
        ]
        ledger.append_transaction(events)
        projection = LifecycleProjection(database, ledger)
        projection.consume_new_events()

        stats = projection.stats(COMPOSITE_REF, "composite")
        assert stats.selected_count == 5
        assert stats.independent_selected_task_count == 2
        decision_before = LifecyclePolicy().review_composite(
            COMPOSITE_REF, SkillStatus.CANDIDATE, stats
        )

        projection.rebuild()
        replayed = projection.stats(COMPOSITE_REF, "composite")
        decision_after = LifecyclePolicy().review_composite(
            COMPOSITE_REF, SkillStatus.CANDIDATE, replayed
        )
        assert replayed.independent_selected_task_count == 2
        assert decision_after == decision_before


def test_r8_composite_zero_success_viability_gate_preserves_activation() -> None:
    policy = LifecyclePolicy()
    zero_success = ArtifactStats(
        COMPOSITE_REF,
        "composite",
        event_task_ids={
            "selected": ["task-a", "task-b", "task-c"],
            "deployment_unsuccessful": ["task-a", "task-b", "task-c"],
        },
    )
    suppressed = policy.review_composite(
        COMPOSITE_REF, SkillStatus.CANDIDATE, zero_success
    )
    assert suppressed.next_status == SkillStatus.SUPPRESSED.value
    assert suppressed.reason == "candidate_zero_success_after_independent_trials"

    one_success = ArtifactStats(
        COMPOSITE_REF,
        "composite",
        event_task_ids={
            "selected": [f"task-{index}" for index in range(4)],
            "deployment_success": ["task-0"],
            "deployment_unsuccessful": [f"task-{index}" for index in range(1, 4)],
        },
    )
    kept = policy.review_composite(COMPOSITE_REF, SkillStatus.CANDIDATE, one_success)
    assert kept.next_status == SkillStatus.CANDIDATE.value
    assert kept.reason == "needs_deployment_successes"

    two_successes = ArtifactStats(
        COMPOSITE_REF,
        "composite",
        event_task_ids={"deployment_success": ["task-a", "task-b"]},
    )
    promoted = policy.review_composite(
        COMPOSITE_REF, SkillStatus.CANDIDATE, two_successes
    )
    assert promoted.next_status == SkillStatus.ACTIVE.value
    assert promoted.reason == "independent_deployment_successes"

    with pytest.raises(ValueError, match="composite_candidate_zero_success_trial_limit"):
        LifecycleThresholds(composite_candidate_zero_success_trial_limit=0)


@pytest.mark.parametrize("invalid", [True, 1.0, 0, -1])
def test_r8_zero_success_trial_limit_is_a_strict_positive_integer(
    invalid: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="composite_candidate_zero_success_trial_limit",
    ):
        LifecycleThresholds(
            composite_candidate_zero_success_trial_limit=invalid,
        )


def test_r8_active_atomic_child_allows_composite_promotion(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        projection = _setup_closure(
            database, tmp_path, (SkillStatus.ACTIVE,)
        )
        decision = LifecycleController(database, projection).review(
            [COMPOSITE_REF]
        ).decisions[0]
        assert decision.next_status == SkillStatus.ACTIVE.value


def test_r8_candidate_child_blocks_then_later_allows_parent_promotion(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        projection = _setup_closure(
            database, tmp_path, (SkillStatus.CANDIDATE,)
        )
        controller = LifecycleController(database, projection)
        blocked = controller.review([COMPOSITE_REF]).decisions[0]
        assert blocked.next_status == SkillStatus.CANDIDATE.value
        assert blocked.reason == "awaiting_frozen_child_closure"

        with database.transaction() as connection:
            connection.execute(
                "UPDATE artifact_index SET status=? WHERE artifact_ref=?",
                (SkillStatus.ACTIVE.value, ATOMIC_A_REF),
            )
        promoted = controller.review([COMPOSITE_REF]).decisions[0]
        assert promoted.next_status == SkillStatus.ACTIVE.value


def test_r8_same_batch_child_promotion_waits_for_next_parent_review(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        projection = _setup_closure(
            database, tmp_path, (SkillStatus.CANDIDATE,)
        )
        projection.ledger.append_transaction([
            _atomic_evidence(
                task_id="atomic-success-a",
                index=0,
                event=EvidenceEventType.DIRECT_SUCCESS,
            ),
            _atomic_evidence(
                task_id="atomic-success-b",
                index=1,
                event=EvidenceEventType.DIRECT_SUCCESS,
            ),
        ])
        projection.consume_new_events()
        controller = LifecycleController(database, projection)

        first = controller.review([ATOMIC_A_REF, COMPOSITE_REF])
        by_ref = {decision.artifact_ref: decision for decision in first.decisions}
        assert by_ref[ATOMIC_A_REF].next_status == SkillStatus.ACTIVE.value
        assert by_ref[COMPOSITE_REF].next_status == SkillStatus.CANDIDATE.value
        assert by_ref[COMPOSITE_REF].reason == "awaiting_frozen_child_closure"

        second = controller.review([COMPOSITE_REF]).decisions[0]
        assert second.next_status == SkillStatus.ACTIVE.value


def test_r8_same_batch_child_demotion_blocks_parent_promotion(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        projection = _setup_closure(
            database, tmp_path, (SkillStatus.ACTIVE,)
        )
        projection.ledger.append_transaction([
            _atomic_evidence(
                task_id=f"atomic-failure-{index}",
                index=index,
                event=EvidenceEventType.DIRECT_FAILURE,
                intrinsic_failure=True,
            )
            for index in range(3)
        ])
        projection.consume_new_events()

        result = LifecycleController(database, projection).review(
            [ATOMIC_A_REF, COMPOSITE_REF]
        )
        by_ref = {decision.artifact_ref: decision for decision in result.decisions}
        assert by_ref[ATOMIC_A_REF].next_status == SkillStatus.SUPPRESSED.value
        assert by_ref[COMPOSITE_REF].next_status == SkillStatus.CANDIDATE.value
        assert by_ref[COMPOSITE_REF].reason == "awaiting_frozen_child_closure"

        statuses = {
            str(row["artifact_ref"]): str(row["status"])
            for row in database.rows(
                "SELECT artifact_ref,status FROM artifact_index "
                "WHERE artifact_ref IN (?,?)",
                (ATOMIC_A_REF, COMPOSITE_REF),
            )
        }
        assert statuses == {
            ATOMIC_A_REF: SkillStatus.SUPPRESSED.value,
            COMPOSITE_REF: SkillStatus.CANDIDATE.value,
        }


def test_r8_mixed_child_status_blocks_composite_promotion(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        projection = _setup_closure(
            database,
            tmp_path,
            (SkillStatus.ACTIVE, SkillStatus.CANDIDATE),
        )
        decision = LifecycleController(database, projection).review(
            [COMPOSITE_REF]
        ).decisions[0]
        assert decision.next_status == SkillStatus.CANDIDATE.value
        assert decision.reason == "awaiting_frozen_child_closure"


def test_r8_missing_contains_relation_fails_closed(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        projection = _setup_closure(
            database,
            tmp_path,
            (SkillStatus.ACTIVE, SkillStatus.ACTIVE),
            relation_count=1,
        )
        decision = LifecycleController(database, projection).review(
            [COMPOSITE_REF]
        ).decisions[0]
        assert decision.next_status == SkillStatus.CANDIDATE.value
        assert decision.reason == "awaiting_frozen_child_closure"
