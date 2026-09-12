from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.errors import ArtifactIntegrityError
from atomic_skillgraph.core.refs import SkillRef, content_hash
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.governance.ledger import (
    EvidenceEvent,
    EvidenceEventType,
    EvidenceLedger,
)
from atomic_skillgraph.governance.lifecycle import LifecycleController
from atomic_skillgraph.governance.projections import LifecycleProjection
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from experiments.protocol import active_composite_frozen_closure_audit


CHILD = "skill://r91-child@1.0.0"
CHILD_V2 = "skill://r91-child@2.0.0"
OTHER_CHILD = "skill://r91-other-child@1.0.0"
PARENT = "skill://r91-parent@1.0.0"
PARENT_V2 = "skill://r91-parent@2.0.0"
OTHER_PARENT = "skill://r91-other-parent@1.0.0"


def _payload(*child_refs: str) -> dict[str, object]:
    return {
        "occurrences": [
            {
                "occurrence_id": f"occ-{index}",
                "step_id": f"step-{index}",
                "node_ref": SkillRef.parse(ref).to_dict(),
            }
            for index, ref in enumerate(child_refs)
        ]
    }


def _index(
    database: StateDatabase,
    tmp_path: Path,
    ref: str,
    kind: str,
    status: SkillStatus,
    *,
    payload: dict[str, object] | None = None,
) -> Path:
    parsed = SkillRef.parse(ref)
    value = payload if payload is not None else {}
    path = tmp_path / f"{parsed.logical_id}-{parsed.version}.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO artifact_index(artifact_ref,artifact_kind,logical_id,version,"
            "content_hash,status,file_path,schema_version) VALUES(?,?,?,?,?,?,?,3)",
            (
                ref,
                kind,
                parsed.logical_id,
                parsed.version,
                content_hash(
                    value,
                    exclude=("status", "quality", "statistics", "evidence"),
                ),
                status.value,
                str(path),
            ),
        )
    return path


def _edge(
    database: StateDatabase,
    parent_ref: str,
    child_ref: str,
    *,
    suffix: str = "0",
) -> None:
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO graph_edges(edge_id,source_ref,target_ref,relation,metadata_json) "
            "VALUES(?,?,?,?,?)",
            (f"edge-{suffix}", parent_ref, child_ref, "contains", "{}"),
        )


def _event(
    ref: str,
    kind: str,
    event: EvidenceEventType,
    index: int,
    *,
    intrinsic: bool = False,
    metadata: dict[str, object] | None = None,
) -> EvidenceEvent:
    details = dict(metadata or {})
    if intrinsic:
        details["intrinsic_failure"] = True
    return EvidenceEvent.create(
        task_id=f"task-{ref}-{index}",
        trace_id=f"trace-{ref}-{index}",
        occurrence_id=f"occ-{index}",
        attempt_id=f"attempt-{ref}-{index}",
        sequence_no=index,
        artifact_ref=ref,
        artifact_kind=kind,
        event=event,
        failure_layer=kind if intrinsic else "",
        metadata=details,
    )


def _projection(
    database: StateDatabase,
    events: list[EvidenceEvent] | None = None,
) -> LifecycleProjection:
    ledger = EvidenceLedger(database)
    if events:
        ledger.append_transaction(events)
    projection = LifecycleProjection(database, ledger)
    projection.consume_new_events()
    return projection


def _failure_events(ref: str = CHILD) -> list[EvidenceEvent]:
    return [
        _event(ref, "atomic", EvidenceEventType.DIRECT_FAILURE, index, intrinsic=True)
        for index in range(3)
    ]


def _status(database: StateDatabase, ref: str) -> str:
    row = database.execute(
        "SELECT status FROM artifact_index WHERE artifact_ref=?", (ref,)
    ).fetchone()
    assert row is not None
    return str(row["status"])


def _pointer(database: StateDatabase, logical_id: str) -> str | None:
    row = database.execute(
        "SELECT artifact_ref FROM recommended_pointers WHERE logical_id=?",
        (logical_id,),
    ).fetchone()
    return None if row is None else str(row["artifact_ref"])


def _active_family(
    database: StateDatabase,
    tmp_path: Path,
    *,
    parents: tuple[str, ...] = (PARENT,),
    child: str = CHILD,
) -> None:
    _index(database, tmp_path, child, "atomic", SkillStatus.ACTIVE)
    for index, parent in enumerate(parents):
        _index(
            database,
            tmp_path,
            parent,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(child),
        )
        _edge(database, parent, child, suffix=str(index))
    with database.transaction() as connection:
        for ref in (child, *parents):
            logical_id = SkillRef.parse(ref).logical_id
            connection.execute(
                "INSERT OR REPLACE INTO recommended_pointers(logical_id,artifact_ref) "
                "VALUES(?,?)",
                (logical_id, ref),
            )


@pytest.mark.parametrize("restricted", [False, True])
def test_r91_b01_b02_child_and_active_parent_leave_active_in_one_review(
    tmp_path: Path,
    restricted: bool,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        controller = LifecycleController(
            database, _projection(database, _failure_events())
        )

        result = controller.review([CHILD] if restricted else None)
        decisions = {decision.artifact_ref: decision for decision in result.decisions}

        assert decisions[CHILD].next_status == SkillStatus.SUPPRESSED.value
        assert decisions[PARENT].next_status == SkillStatus.SUPPRESSED.value
        assert decisions[PARENT].reason == "frozen_dependency_unusable"
        assert _status(database, CHILD) == SkillStatus.SUPPRESSED.value
        assert _status(database, PARENT) == SkillStatus.SUPPRESSED.value
        assert _pointer(database, "r91-child") is None
        assert _pointer(database, "r91-parent") is None


def test_r91_b03_stable_replacement_does_not_rebind_exact_ref(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        _index(database, tmp_path, CHILD_V2, "atomic", SkillStatus.ACTIVE)
        event = _event(
            CHILD,
            "atomic",
            EvidenceEventType.SUPERSEDED,
            0,
            metadata={"replacement_status": "active", "replacement_ref": CHILD_V2},
        )

        result = LifecycleController(
            database, _projection(database, [event])
        ).review([CHILD])
        by_ref = {decision.artifact_ref: decision for decision in result.decisions}

        assert by_ref[CHILD].reason == "superseded"
        assert by_ref[PARENT].reason == "frozen_dependency_unusable"
        assert _status(database, PARENT) == SkillStatus.SUPPRESSED.value
        payload = json.loads(
            Path(
                database.execute(
                    "SELECT file_path FROM artifact_index WHERE artifact_ref=?", (PARENT,)
                ).fetchone()["file_path"]
            ).read_text("utf-8")
        )
        assert payload["occurrences"][0]["node_ref"] == SkillRef.parse(CHILD).to_dict()


def test_r91_b04_two_exact_ref_parents_each_get_one_final_decision(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path, parents=(PARENT, OTHER_PARENT))
        result = LifecycleController(
            database, _projection(database, _failure_events())
        ).review([CHILD])

        refs = [decision.artifact_ref for decision in result.decisions]
        assert refs.count(PARENT) == 1
        assert refs.count(OTHER_PARENT) == 1
        assert _status(database, PARENT) == SkillStatus.SUPPRESSED.value
        assert _status(database, OTHER_PARENT) == SkillStatus.SUPPRESSED.value


def test_r91_b05_same_logical_id_version_and_unrelated_parent_are_untouched(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        _index(database, tmp_path, CHILD_V2, "atomic", SkillStatus.ACTIVE)
        _index(
            database,
            tmp_path,
            OTHER_PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(CHILD_V2),
        )
        _edge(database, OTHER_PARENT, CHILD_V2, suffix="other")

        result = LifecycleController(
            database, _projection(database, _failure_events())
        ).review([CHILD])

        assert OTHER_PARENT not in {item.artifact_ref for item in result.decisions}
        assert _status(database, CHILD_V2) == SkillStatus.ACTIVE.value
        assert _status(database, OTHER_PARENT) == SkillStatus.ACTIVE.value


def test_r91_b06_candidate_promotion_guard_is_unchanged(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.CANDIDATE)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.CANDIDATE,
            payload=_payload(CHILD),
        )
        _edge(database, PARENT, CHILD)
        parent_events = [
            _event(PARENT, "composite", EvidenceEventType.DEPLOYMENT_SUCCESS, index)
            for index in range(2)
        ]

        result = LifecycleController(
            database, _projection(database, parent_events)
        ).review([PARENT])
        decision = result.decisions[0]

        assert decision.next_status == SkillStatus.CANDIDATE.value
        assert decision.reason == "awaiting_frozen_child_closure"


def test_r91_b07_same_batch_child_activation_does_not_rescue_parent(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.CANDIDATE)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.CANDIDATE,
            payload=_payload(CHILD),
        )
        _edge(database, PARENT, CHILD)
        events = [
            *[
                _event(CHILD, "atomic", EvidenceEventType.DIRECT_SUCCESS, index)
                for index in range(2)
            ],
            *[
                _event(PARENT, "composite", EvidenceEventType.DEPLOYMENT_SUCCESS, index)
                for index in range(2)
            ],
        ]
        controller = LifecycleController(database, _projection(database, events))

        first = {item.artifact_ref: item for item in controller.review([CHILD, PARENT]).decisions}
        assert first[CHILD].next_status == SkillStatus.ACTIVE.value
        assert first[PARENT].next_status == SkillStatus.CANDIDATE.value
        assert first[PARENT].reason == "awaiting_frozen_child_closure"
        assert controller.review([PARENT]).decisions[0].next_status == SkillStatus.ACTIVE.value


def test_r91_b08_parent_empirical_reason_takes_precedence(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        events = [
            *_failure_events(),
            *[
                _event(
                    PARENT,
                    "composite",
                    EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
                    index,
                )
                for index in range(3)
            ],
        ]
        result = LifecycleController(
            database, _projection(database, events)
        ).review([CHILD, PARENT])
        parent = {item.artifact_ref: item for item in result.decisions}[PARENT]

        assert parent.next_status == SkillStatus.SUPPRESSED.value
        assert parent.reason == "repeated_empirical_deployment_unsuccessful"


def test_r91_b09_already_invalid_child_closes_stale_parent_idempotently(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.SUPPRESSED)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(CHILD),
        )
        _edge(database, PARENT, CHILD)
        controller = LifecycleController(database, _projection(database))

        first = controller.review([CHILD])
        second = controller.review([CHILD])

        assert {item.artifact_ref for item in first.decisions} == {CHILD, PARENT}
        assert _status(database, PARENT) == SkillStatus.SUPPRESSED.value
        assert second.changed_count == 0
        assert {item.artifact_ref for item in second.decisions} == {CHILD}


def test_r91_active_parent_guard_closes_stale_dependency_without_child_request(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.SUPPRESSED)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(CHILD),
        )
        _edge(database, PARENT, CHILD)

        decision = LifecycleController(
            database, _projection(database)
        ).review([PARENT]).decisions[0]

        assert decision.next_status == SkillStatus.SUPPRESSED.value
        assert decision.reason == "frozen_dependency_unusable"
        assert _status(database, PARENT) == SkillStatus.SUPPRESSED.value


@pytest.mark.parametrize("damage", ["missing", "hash", "invalid_ref"])
def test_r91_b10_affected_parent_integrity_damage_aborts_batch(
    tmp_path: Path,
    damage: str,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        row = database.execute(
            "SELECT file_path FROM artifact_index WHERE artifact_ref=?", (PARENT,)
        ).fetchone()
        path = Path(str(row["file_path"]))
        if damage == "missing":
            path.unlink()
        elif damage == "hash":
            path.write_text(json.dumps(_payload(OTHER_CHILD)), encoding="utf-8")
        else:
            invalid = {"occurrences": [{"node_ref": "not-a-ref"}]}
            path.write_text(json.dumps(invalid), encoding="utf-8")
            database.execute(
                "UPDATE artifact_index SET content_hash=? WHERE artifact_ref=?",
                (content_hash(invalid), PARENT),
            )
            database.connection.commit()

        with pytest.raises(ArtifactIntegrityError) as captured:
            LifecycleController(
                database, _projection(database, _failure_events())
            ).review([CHILD])

        expected_code = {
            "missing": "artifact_file_missing",
            "hash": "artifact_hash_mismatch",
            "invalid_ref": "composite_dependency_integrity_error",
        }[damage]
        assert captured.value.code == expected_code
        assert _status(database, CHILD) == SkillStatus.ACTIVE.value
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value


@pytest.mark.parametrize("damage", ["missing_child", "wrong_kind"])
def test_r91_active_parent_invalid_child_registry_is_integrity_failure(
    tmp_path: Path,
    damage: str,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        if damage == "wrong_kind":
            _index(database, tmp_path, CHILD, "implementation", SkillStatus.ACTIVE)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(CHILD),
        )
        _edge(database, PARENT, CHILD)

        with pytest.raises(
            ArtifactIntegrityError, match="invalid Atomic dependencies"
        ) as captured:
            LifecycleController(database, _projection(database)).review([PARENT])

        assert captured.value.code == "composite_dependency_integrity_error"
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value


def test_r91_b11_missing_contains_is_integrity_failure_not_auto_repair(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.ACTIVE)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(CHILD),
        )

        with pytest.raises(ArtifactIntegrityError, match="lacks CONTAINS") as captured:
            LifecycleController(
                database, _projection(database, _failure_events())
            ).review([CHILD])

        assert captured.value.code == "composite_dependency_integrity_error"
        assert _status(database, CHILD) == SkillStatus.ACTIVE.value
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value
        assert database.execute(
            "SELECT COUNT(*) AS count FROM graph_edges"
        ).fetchone()["count"] == 0


def test_r91_b10_b11_exact_child_survives_malformed_sibling_without_edge(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.ACTIVE)
        damaged = _payload(CHILD)
        damaged["occurrences"].append({"node_ref": "not-a-ref"})
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=damaged,
        )

        with pytest.raises(ArtifactIntegrityError) as captured:
            LifecycleController(
                database, _projection(database, _failure_events())
            ).review([CHILD])

        assert captured.value.code == "composite_dependency_integrity_error"
        assert _status(database, CHILD) == SkillStatus.ACTIVE.value
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value


def test_r91_unlinked_malformed_unrelated_parent_is_not_invented_as_affected(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.ACTIVE)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload={"occurrences": [{"node_ref": "not-a-ref"}]},
        )

        result = LifecycleController(
            database, _projection(database, _failure_events())
        ).review([CHILD])

        assert {item.artifact_ref for item in result.decisions} == {CHILD}
        assert _status(database, CHILD) == SkillStatus.SUPPRESSED.value
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value


def test_r91_b12_extra_graph_edge_cannot_invent_parent_dependency(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _index(database, tmp_path, CHILD, "atomic", SkillStatus.ACTIVE)
        _index(database, tmp_path, OTHER_CHILD, "atomic", SkillStatus.ACTIVE)
        _index(
            database,
            tmp_path,
            PARENT,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(OTHER_CHILD),
        )
        _edge(database, PARENT, OTHER_CHILD, suffix="real")
        _edge(database, PARENT, CHILD, suffix="extra")

        result = LifecycleController(
            database, _projection(database, _failure_events())
        ).review([CHILD])

        assert PARENT not in {item.artifact_ref for item in result.decisions}
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value


def test_r91_b13_pointer_moves_to_other_active_parent_version(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        _index(database, tmp_path, OTHER_CHILD, "atomic", SkillStatus.ACTIVE)
        _index(
            database,
            tmp_path,
            PARENT_V2,
            "composite",
            SkillStatus.ACTIVE,
            payload=_payload(OTHER_CHILD),
        )
        _edge(database, PARENT_V2, OTHER_CHILD, suffix="v2")
        with database.transaction() as connection:
            connection.execute(
                "UPDATE recommended_pointers SET artifact_ref=? WHERE logical_id=?",
                (PARENT, "r91-parent"),
            )

        LifecycleController(
            database, _projection(database, _failure_events())
        ).review([CHILD])

        assert _status(database, PARENT) == SkillStatus.SUPPRESSED.value
        assert _status(database, PARENT_V2) == SkillStatus.ACTIVE.value
        assert _pointer(database, "r91-parent") == PARENT_V2


@pytest.mark.parametrize("failure_point", ["parent_update", "pointer"])
def test_r91_b14_status_and_pointer_failures_roll_back_whole_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    database_path = tmp_path / "state.sqlite"
    with StateDatabase(database_path) as database:
        _active_family(database, tmp_path)
        controller = LifecycleController(
            database, _projection(database, _failure_events())
        )
        if failure_point == "parent_update":
            database.execute(
                "CREATE TRIGGER fail_parent_update BEFORE UPDATE ON artifact_index "
                f"WHEN OLD.artifact_ref='{PARENT}' "
                "BEGIN SELECT RAISE(ABORT, 'injected parent update failure'); END"
            )
            database.connection.commit()
        else:
            original_refresh = controller._refresh_recommended

            def fail_pointer(connection: object, logical_id: str) -> None:
                if logical_id == "r91-parent":
                    raise sqlite3.OperationalError("injected pointer failure")
                original_refresh(connection, logical_id)

            monkeypatch.setattr(controller, "_refresh_recommended", fail_pointer)

        with pytest.raises(sqlite3.DatabaseError):
            controller.review([CHILD])

        assert _status(database, CHILD) == SkillStatus.ACTIVE.value
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value
        assert _pointer(database, "r91-child") == CHILD
        assert _pointer(database, "r91-parent") == PARENT

    with StateDatabase(database_path) as reopened:
        registry = SkillRegistry(ArtifactStore(tmp_path, reopened), reopened)
        projection = LifecycleProjection(reopened, EvidenceLedger(reopened))
        projection.consume_new_events()
        recovered_controller = LifecycleController(reopened, projection)

        assert CHILD in {str(ref) for ref in registry.list_refs("atomic")}
        assert PARENT in {str(ref) for ref in registry.list_refs("composite")}
        assert _status(reopened, CHILD) == SkillStatus.ACTIVE.value
        assert _status(reopened, PARENT) == SkillStatus.ACTIVE.value
        assert _pointer(reopened, "r91-child") == CHILD
        assert _pointer(reopened, "r91-parent") == PARENT
        assert projection.stats(
            CHILD, "atomic"
        ).consecutive_intrinsic_failures == 3

        if failure_point == "parent_update":
            reopened.execute("DROP TRIGGER fail_parent_update")
            reopened.connection.commit()
        recovered = {
            decision.artifact_ref: decision
            for decision in recovered_controller.review([CHILD]).decisions
        }
        assert recovered[CHILD].next_status == SkillStatus.SUPPRESSED.value
        assert recovered[PARENT].next_status == SkillStatus.SUPPRESSED.value
        assert _status(reopened, CHILD) == SkillStatus.SUPPRESSED.value
        assert _status(reopened, PARENT) == SkillStatus.SUPPRESSED.value
        assert _pointer(reopened, "r91-child") is None
        assert _pointer(reopened, "r91-parent") is None


def test_r91_b15_formal_closure_passes_after_dependency_invalidation(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        LifecycleController(
            database, _projection(database, _failure_events())
        ).review([CHILD])

        class Registry:
            @staticmethod
            def get_composite(ref: str) -> object:
                return SimpleNamespace(
                    occurrences=(SimpleNamespace(node_ref=SkillRef.parse(CHILD)),)
                )

        audit = active_composite_frozen_closure_audit(database, Registry())
        assert audit["active_composite_frozen_closure_passed"] is True
        assert audit["active_composite_count"] == 0


def test_r91_explicit_empty_review_never_scans_or_mutates_registry(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite") as database:
        _active_family(database, tmp_path)
        result = LifecycleController(database, _projection(database)).review([])

        assert result.reviewed_count == 0
        assert result.decisions == ()
        assert _status(database, CHILD) == SkillStatus.ACTIVE.value
        assert _status(database, PARENT) == SkillStatus.ACTIVE.value
