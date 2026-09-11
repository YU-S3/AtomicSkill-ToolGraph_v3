from __future__ import annotations

from pathlib import Path

from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceEventType, EvidenceLedger
from atomic_skillgraph.governance.projections import LifecycleProjection
from atomic_skillgraph.knowledge.database import StateDatabase


COMPOSITE_REF = "skill://r9-composite@1.0.0"


def _trace(**overrides: object) -> dict[str, object]:
    trace: dict[str, object] = {
        "trace_id": "trace-r9",
        "task": {"task_id": "task-r9"},
        "infrastructure_failure": False,
        "implementation_invocations": [],
        "tool_executions": [],
        "runtime_spans": [
            {"occurrence_id": "occ-1", "action_start": 0, "action_end": 1}
        ],
        "node_records": [
            {
                "occurrence_id": "occ-1",
                "atomic_ref": "skill://r9-atomic@1.0.0",
                "status": "agent_completed_before_invocation",
            }
        ],
        "runtime_plan": {"source_composite_ref": COMPOSITE_REF},
        "benchmark_success": True,
        "task_contract_success": True,
        "graph_self_sufficient_success": True,
        "graph_full_completion": True,
        "task_rescue_required": False,
    }
    trace.update(overrides)
    return trace


def _deployment_events(trace: dict[str, object]) -> list[object]:
    return [
        event
        for event in CreditAssigner().assign(trace)
        if event.event
        in {
            EvidenceEventType.DEPLOYMENT_SUCCESS,
            EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
        }
    ]


def test_r9_full_executed_graph_gets_one_deployment_success() -> None:
    events = _deployment_events(_trace())
    assert len(events) == 1
    assert events[0].event is EvidenceEventType.DEPLOYMENT_SUCCESS
    assert events[0].artifact_kind == "composite"
    assert events[0].metadata["intrinsic_failure"] is False


def test_r9_terminal_tail_exhaustion_and_preterminal_are_unsuccessful() -> None:
    terminal_tail = _trace(
        graph_full_completion=False,
        node_records=[
            {
                "occurrence_id": "occ-1",
                "atomic_ref": "skill://r9-atomic@1.0.0",
                "status": "skipped_goal_terminal",
            }
        ],
    )
    exhausted = _trace(
        benchmark_success=False,
        task_contract_success=False,
        graph_self_sufficient_success=False,
        graph_full_completion=False,
    )
    preterminal = _trace(runtime_spans=[], node_records=[])
    for trace in (terminal_tail, exhausted, preterminal):
        events = _deployment_events(trace)
        assert len(events) == 1
        assert events[0].event is EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL
        assert events[0].metadata["intrinsic_failure"] is False


def test_r9_rescue_retains_rescue_fact_and_terminal_deployment_outcome() -> None:
    events = CreditAssigner().assign(_trace(task_rescue_required=True))
    composite_events = {
        event.event for event in events if event.artifact_kind == "composite"
    }
    assert EvidenceEventType.TASK_RESCUE_REQUIRED in composite_events
    assert EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL in composite_events
    assert EvidenceEventType.DEPLOYMENT_SUCCESS not in composite_events


def test_r9_selected_noninfra_is_exactly_one_terminal_trial_and_infra_is_neutral() -> None:
    assert len(_deployment_events(_trace())) == 1
    assert CreditAssigner().assign(_trace(infrastructure_failure=True)) == []


def test_r9_deployment_unsuccessful_does_not_increment_intrinsic_failure(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        ledger = EvidenceLedger(database)
        ledger.append_transaction(
            CreditAssigner().assign(_trace(benchmark_success=False))
        )
        projection = LifecycleProjection(database, ledger)
        projection.consume_new_events()
        stats = projection.stats(COMPOSITE_REF, "composite")
        assert stats.deployment_unsuccessful_count == 1
        assert stats.independent_deployment_trial_count == 1
        assert stats.intrinsic_failure_count == 0
