from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.governance.ledger import (
    EvidenceEvent,
    EvidenceEventType,
    EvidenceLedger,
)
from atomic_skillgraph.governance.projections import LifecycleProjection
from atomic_skillgraph.knowledge.database import StateDatabase

from experiments.protocol import (
    ProtocolError,
    composite_deployment_evidence_audit,
    require_r9_formal_freeze_audit,
    r9_formal_freeze_audit,
)


REF = "skill://r9-composite@1.0.0"


def _event(task_id: str, event: EvidenceEventType, index: int) -> EvidenceEvent:
    return EvidenceEvent.create(
        task_id=task_id,
        trace_id=f"trace-{task_id}",
        occurrence_id="graph",
        attempt_id=f"attempt-{task_id}-{event.value}",
        sequence_no=index,
        artifact_ref=REF,
        artifact_kind="composite",
        event=event,
    )


def _index_active(database: StateDatabase) -> None:
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO artifact_index(artifact_ref,artifact_kind,logical_id,version,"
            "content_hash,status,file_path,schema_version) VALUES(?,?,?,?,?,?,?,3)",
            (REF, "composite", "r9-composite", "1.0.0", "hash", "active", "unused"),
        )


def test_r9_deployment_freeze_audit_requires_complete_disjoint_outcomes(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        _index_active(database)
        ledger = EvidenceLedger(database)
        ledger.append_transaction([
            _event("task-a", EvidenceEventType.SELECTED, 0),
            _event("task-a", EvidenceEventType.DEPLOYMENT_SUCCESS, 1),
            _event("task-b", EvidenceEventType.SELECTED, 2),
            _event("task-b", EvidenceEventType.DEPLOYMENT_SUCCESS, 3),
        ])
        LifecycleProjection(database, ledger).consume_new_events()
        audit = composite_deployment_evidence_audit(database)
        assert audit["composite_deployment_evidence_passed"] is True
        assert audit["composite_deployment_success_count"] == 2
        assert audit["composite_deployment_trial_count"] == 2


def test_r9_deployment_freeze_audit_fails_missing_or_ambiguous_terminal_outcome(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        _index_active(database)
        ledger = EvidenceLedger(database)
        ledger.append_transaction([
            _event("task-a", EvidenceEventType.SELECTED, 0),
            _event("task-a", EvidenceEventType.DEPLOYMENT_SUCCESS, 1),
            _event("task-a", EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL, 2),
            _event("task-b", EvidenceEventType.SELECTED, 3),
        ])
        LifecycleProjection(database, ledger).consume_new_events()
        audit = composite_deployment_evidence_audit(database)
        assert audit["composite_deployment_evidence_passed"] is False
        reasons = {item["reason"] for item in audit["violations"]}
        assert "deployment_outcomes_not_mutually_exclusive" in reasons
        assert "selected_deployment_outcome_incomplete" in reasons
        assert "selected_trace_deployment_outcome_incomplete" in reasons


def test_r9_combined_freeze_audit_passes_empty_fresh_bank(tmp_path: Path) -> None:
    registry = SimpleNamespace(atomics=lambda: [])
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        audit = require_r9_formal_freeze_audit(database, registry)
    assert audit["r9_formal_freeze_audit_passed"] is True
    assert all(audit["component_passes"].values())


def test_r9_combined_freeze_audit_fails_closed_on_incomplete_deployment(
    tmp_path: Path,
) -> None:
    registry = SimpleNamespace(atomics=lambda: [])
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        _index_active(database)
        EvidenceLedger(database).append_transaction([
            _event("task-a", EvidenceEventType.SELECTED, 0),
        ])
        audit = r9_formal_freeze_audit(database, registry)
        assert audit["r9_formal_freeze_audit_passed"] is False
        with pytest.raises(ProtocolError, match="R9 formal freeze authority failed"):
            require_r9_formal_freeze_audit(database, registry)
