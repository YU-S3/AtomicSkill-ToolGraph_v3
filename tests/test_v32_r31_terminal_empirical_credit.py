"""v3.2-R3.1 gates for Terminal-Empirical Composite credit provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import (
    CompositeOccurrence,
    CompositeSkill,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import ValidationResult
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.maintenance import EvolutionMaintenance
from atomic_skillgraph.evolution.repair import RepairStore
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceEventType, EvidenceLedger
from atomic_skillgraph.governance.lifecycle import LifecycleController
from atomic_skillgraph.governance.projections import LifecycleProjection
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.graph_store import GraphStore
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.system import AtomicSkillGraphSystem


def _terminal_empirical_candidate() -> CompositeSkill:
    occurrence = CompositeOccurrence(
        step_id="step_terminal",
        occurrence_id="occ_terminal",
        node_ref=SkillRef("atomic_terminal_fixture", "1.0.0"),
        binding_specs={},
    )
    return CompositeSkill(
        ref=SkillRef("terminal_empirical_fixture", "1.0.0"),
        summary="terminal empirical fixture",
        occurrences=[occurrence],
        control_sequence=[occurrence.step_id],
        data_edges=[],
        dependency_edges=[],
        goal_contract=TaskContract(),
        guideline={},
        insight={},
        validator_spec={},
        metadata={
            "completion_authority": {
                "kind": "terminal_empirical",
                "source_trace_id": "trace_creation",
            },
        },
        status=SkillStatus.CANDIDATE,
    )


def _credit_system(tmp_path) -> tuple[AtomicSkillGraphSystem, StateDatabase]:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)

    system = object.__new__(AtomicSkillGraphSystem)
    system.database = database
    system.skills = skills
    system.tools = tools
    system.graph = GraphStore(database, skills)
    system.aligner = Aligner(skills, tools)
    system.credit = CreditAssigner()
    system.ledger = EvidenceLedger(database)
    system.projection = LifecycleProjection(database, system.ledger)
    system.lifecycle = LifecycleController(database, system.projection)
    return system, database


def _apply_terminal_creation(
    system: AtomicSkillGraphSystem,
) -> tuple[str, SimpleNamespace]:
    trace = SimpleNamespace(
        trace_id="trace_creation",
        task=SimpleNamespace(task_id="task_creation"),
        runtime_plan={"source_composite_ref": None},
        metadata={},
        benchmark_success=True,
        task_rescue_required=False,
    )
    prepared = SimpleNamespace(
        compiled=[],
        composite=_terminal_empirical_candidate(),
        source_composite_ref="",
    )
    applied = system._apply_evolution(
        prepared,
        trace,
        SimpleNamespace(task_id="task_creation", context={}),
    )
    return str(applied["composite_ref"]), trace


def _terminal_runtime_trace(
    composite_ref: str,
    *,
    task_id: str,
    trace_id: str,
    candidate_executed: bool,
) -> dict:
    node_status = (
        "direct_autonomous_success"
        if candidate_executed
        else "already_satisfied"
    )
    return {
        "trace_id": trace_id,
        "task": {
            "task_id": task_id,
            "goal": "satisfy the terminal fixture",
            "benchmark": "fake",
            "task_type": "terminal_empirical",
            "metadata": {},
        },
        "infrastructure_failure": False,
        "runtime_plan": {
            "source_composite_ref": composite_ref,
            "control_sequence": ["step_terminal"],
            "planner_audit": {
                "selected_composite_authority": {
                    "kind": "terminal_empirical",
                },
            },
        },
        "metadata": {
            "terminal_empirical_execution": {
                "candidate_executed": candidate_executed,
            },
        },
        "benchmark_success": True,
        "task_contract_success": True,
        "graph_self_sufficient_success": True,
        "task_rescue_required": False,
        "node_records": [{
            "step_id": "step_terminal",
            "occurrence_id": "occ_terminal",
            "atomic_ref": "skill://atomic_terminal_fixture@1.0.0",
            "status": node_status,
            "direct_result": {"started": candidate_executed},
        }],
        "implementation_invocations": (
            [{
                "attempt_id": f"attempt_{trace_id}",
                "occurrence_id": "occ_terminal",
                "implementation_ref": "skill://impl_terminal_fixture@1.0.0",
                "preflight": {"passed": True},
                "arguments": {"item": "fixture_1"},
                "result": {
                    "started": True,
                    "completed": True,
                    "atomic_effect_passed": True,
                },
            }]
            if candidate_executed
            else []
        ),
        "tool_executions": [],
        "binding_changes": [],
        "failures": [],
    }


def _composite_successes(events, composite_ref: str):
    return [
        event
        for event in events
        if event.artifact_kind == "composite"
        and event.artifact_ref == composite_ref
        and event.event is EvidenceEventType.SELF_SUFFICIENT_SUCCESS
    ]


def test_gate43_terminal_empirical_creation_has_no_self_sufficient_credit(
    tmp_path,
) -> None:
    system, database = _credit_system(tmp_path)
    composite_ref, creation_trace = _apply_terminal_creation(system)

    rows = database.rows(
        "SELECT event_type,metadata_json FROM evidence_events "
        "WHERE artifact_ref=? ORDER BY event_type",
        (composite_ref,),
    )
    assert {str(row["event_type"]) for row in rows} == {
        "proposed",
        "validated",
    }
    assert not any(
        json.loads(str(row["metadata_json"])).get("source")
        == "terminal_empirical_candidate_creation"
        for row in rows
    )
    assert creation_trace.runtime_plan["source_composite_ref"] is None
    database.close()


def test_gate44_terminal_empirical_real_execution_gets_runtime_success() -> None:
    composite_ref = "skill://terminal_runtime_fixture@1.0.0"
    trace = _terminal_runtime_trace(
        composite_ref,
        task_id="task_runtime",
        trace_id="trace_runtime",
        candidate_executed=True,
    )

    successes = _composite_successes(
        CreditAssigner().assign(trace),
        composite_ref,
    )
    assert len(successes) == 1
    assert successes[0].trace_id == trace["trace_id"]
    assert successes[0].task_id == trace["task"]["task_id"]
    assert trace["runtime_plan"]["source_composite_ref"] == composite_ref


def test_gate44_presatisfied_terminal_candidate_gets_no_runtime_success() -> None:
    composite_ref = "skill://terminal_presatisfied_fixture@1.0.0"
    trace = _terminal_runtime_trace(
        composite_ref,
        task_id="task_presatisfied",
        trace_id="trace_presatisfied",
        candidate_executed=False,
    )

    assert _composite_successes(
        CreditAssigner().assign(trace),
        composite_ref,
    ) == []


def test_gate45_terminal_empirical_credit_survives_periodic_maintenance(
    tmp_path,
) -> None:
    system, database = _credit_system(tmp_path)
    composite_ref, _creation_trace = _apply_terminal_creation(system)

    payloads = {
        trace_id: _terminal_runtime_trace(
            composite_ref,
            task_id=task_id,
            trace_id=trace_id,
            candidate_executed=True,
        )
        for task_id, trace_id in (
            ("task_runtime_2", "trace_runtime_2"),
            ("task_runtime_3", "trace_runtime_3"),
        )
    }
    for payload in payloads.values():
        system._commit_evidence(system.credit.assign(payload))

    stats = system.projection.stats(composite_ref, "composite")
    assert stats.independent_self_sufficient_success_count == 2
    lifecycle = system.lifecycle.review([composite_ref])
    assert lifecycle.changed_count == 1
    assert system.skills.get_composite(composite_ref).status is SkillStatus.ACTIVE

    loaded_trace_ids: list[str] = []

    class Traces:
        def load_payload(self, trace_id):
            loaded_trace_ids.append(str(trace_id))
            return payloads[str(trace_id)]

    class EmptyTools:
        def tools(self):
            return []

    maintenance = EvolutionMaintenance(RepairStore(database))
    batch = maintenance.run_batch(
        maintenance_trace_id="trace_periodic_maintenance",
        reviews=[],
        agent_proposals=[],
        tools=EmptyTools(),
        skills=system.skills,
        admission=SimpleNamespace(),
        projection=system.projection,
        traces=Traces(),
        planner_validator=SimpleNamespace(
            validate=lambda *_args, **_kwargs: ValidationResult.ok("planner"),
        ),
        harness_profile="fake_v3",
        replay_tool=lambda _tool, _case: True,
        replay_composite=lambda _candidate, _case: True,
    )
    assert batch.pending_count == 0
    assert set(loaded_trace_ids) == set(payloads)
    assert "trace_creation" not in loaded_trace_ids

    outcome_rows = database.rows(
        "SELECT task_id,trace_id,artifact_ref FROM evidence_events "
        "WHERE artifact_kind='composite' "
        "AND event_type IN "
        "('self_sufficient_success','task_rescue_required',"
        "'goal_terminal_skipped','contract_mismatch')",
    )
    assert len(outcome_rows) == 2
    for row in outcome_rows:
        payload = payloads[str(row["trace_id"])]
        assert payload["task"]["task_id"] == str(row["task_id"])
        assert (
            payload["runtime_plan"]["source_composite_ref"]
            == str(row["artifact_ref"])
        )

    # Maintenance must remain fail-closed for a genuine stored-runtime
    # provenance mismatch; only the creation-side false producer was removed.
    payloads["trace_runtime_2"] = {
        **payloads["trace_runtime_2"],
        "runtime_plan": {
            **payloads["trace_runtime_2"]["runtime_plan"],
            "source_composite_ref": "skill://different_composite@1.0.0",
        },
    }
    with pytest.raises(
        ValueError,
        match="Composite insight Trace does not name its ledger artifact",
    ):
        maintenance._review_composite_insight(
            maintenance_trace_id="trace_invalid_maintenance",
            skills=system.skills,
            traces=Traces(),
            planner_validator=SimpleNamespace(
                validate=lambda *_args, **_kwargs: ValidationResult.ok(
                    "planner"
                ),
            ),
            harness_profile="fake_v3",
            replay_composite=lambda _candidate, _case: True,
        )
    database.close()
