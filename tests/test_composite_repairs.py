from __future__ import annotations

from dataclasses import replace

import pytest

from atomic_skillgraph.core.contracts import CompositeOccurrence, CompositeSkill, TaskContract
from atomic_skillgraph.core.edges import GlobalRelationType, GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.evolution.composite_repairs import CompositeSequenceRepairEngine
from atomic_skillgraph.evolution.repair import RepairStore
from atomic_skillgraph.evolution.typed_repairs import RepairEvidence
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry


def _edge(source: str, target: str, index: int) -> GraphEdge:
    return GraphEdge(
        f"next_{index}", GraphEdgeType.NEXT, source, target,
        origin="reviewed_sequence",
    )


def _composite() -> CompositeSkill:
    steps = ["step_a", "step_b", "step_c"]
    return CompositeSkill(
        SkillRef("composite_prepare", "1.0.0"),
        "prepare workflow",
        [
            CompositeOccurrence(step, f"occ_{step}", SkillRef(f"atomic_{step}", "1.0.0"), {})
            for step in steps
        ],
        steps,
        [],
        [_edge("step_a", "step_b", 1), _edge("step_b", "step_c", 2)],
        TaskContract(),
        {}, {}, {"validator_id": "planner"}, {}, SkillStatus.CANDIDATE,
    )


def _replacement(source: CompositeSkill) -> CompositeSkill:
    return replace(
        source,
        control_sequence=["step_a", "step_c", "step_b"],
        dependency_edges=[_edge("step_a", "step_c", 3), _edge("step_c", "step_b", 4)],
    )


def _evidence(*, agent_error: bool = False):
    return [
        RepairEvidence(
            f"e{i}", f"task{i}", f"trace{i}", "sequence_cluster", {"case": i},
            failure_layer="runtime_agent" if agent_error else "composite",
            agent_parameter_error=agent_error,
        )
        for i in (1, 2)
    ]


@pytest.fixture
def context(tmp_path):
    database = StateDatabase(tmp_path / "state.sqlite3")
    registry = SkillRegistry(ArtifactStore(tmp_path, database), database)
    engine = CompositeSequenceRepairEngine(RepairStore(database), registry)
    yield database, registry, engine
    database.close()


def _execute(engine, proposal):
    return engine.execute(
        proposal,
        replay=lambda _candidate, case: case["case"] in {1, 2},
        validate=lambda _candidate: {"passed": True},
        admit=lambda candidate: candidate,
    )


def test_composite_sequence_revision_registers_new_candidate_and_lineage(context) -> None:
    _database, registry, engine = context
    source = _composite()
    registry.register_composite(source)
    result = _execute(
        engine,
        engine.propose(source.ref, _replacement(source), _evidence()),
    )
    assert result.proposal.status == "admitted"
    assert result.admitted_ref == "skill://composite_prepare@1.0.1"
    assert result.lineage[0].relation is GlobalRelationType.DERIVED_FROM
    assert registry.get_composite(source.ref).control_sequence == ["step_a", "step_b", "step_c"]
    assert registry.get_composite(result.admitted_ref).status is SkillStatus.CANDIDATE


@pytest.mark.parametrize(
    ("candidate_factory", "code"),
    [
        (lambda source: replace(source, control_sequence=["step_a", "step_c"]), "missing_or_unknown_sequence_step"),
        (lambda source: replace(source, control_sequence=["step_a", "step_c", "step_c"]), "duplicate_sequence_step"),
        (lambda source: source, "semantic_edit_empty"),
        (
            lambda source: replace(
                _replacement(source),
                summary="illegally changed summary",
            ),
            "sequence_revision_scope_invalid",
        ),
        (
            lambda source: replace(
                _replacement(source),
                dependency_edges=[_edge("step_b", "step_c", 5)],
            ),
            "edge_order_invalid",
        ),
    ],
)
def test_composite_sequence_rejects_invalid_or_out_of_scope_edits(context, candidate_factory, code) -> None:
    _database, registry, engine = context
    source = _composite()
    registry.register_composite(source)
    outcome = _execute(
        engine,
        engine.propose(source.ref, candidate_factory(source), _evidence()),
    )
    assert outcome.proposal.status == "rejected"
    assert outcome.proposal.replay_result["failure_code"] == code
    assert registry.list_refs("composite") == [source.ref]


@pytest.mark.parametrize(
    ("evidence", "code"),
    [
        (
            [RepairEvidence("one", "task", "trace", "sequence_cluster", {"case": 1})],
            "independent_support_insufficient",
        ),
        (_evidence(agent_error=True), "agent_parameter_error_not_composite_evidence"),
    ],
)
def test_composite_sequence_rejects_single_or_agent_error_evidence(context, evidence, code) -> None:
    _database, registry, engine = context
    source = _composite()
    registry.register_composite(source)
    outcome = _execute(
        engine,
        engine.propose(source.ref, _replacement(source), evidence),
    )
    assert outcome.proposal.status == "rejected"
    assert outcome.proposal.replay_result["failure_code"] == code
