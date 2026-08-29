from __future__ import annotations

from dataclasses import replace

import pytest

from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    GroundingConstraint,
    GroundingConstraintKind,
    ToolBinding,
)
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ImplementationAtom, ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.edges import GlobalRelationType
from atomic_skillgraph.core.errors import AtomicSkillGraphError, FailureLayer
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.evolution.repair import RepairStore
from atomic_skillgraph.evolution.typed_repairs import RepairEvidence, TypedRepairEngine
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry


def _predicate(name: str) -> SemanticPredicate:
    return SemanticPredicate(
        name,
        {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="item")},
    )


def _atomic(
    logical_id: str,
    version: str = "1.0.0",
    *,
    effects: tuple[str, ...] = ("object.ready",),
    summary: str = "prepare item",
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef(logical_id, version),
        summary,
        [ParameterSpec("item", "entity", required_resolution="concrete")],
        [ParameterSpec("result", "entity")],
        [],
        [_predicate(item) for item in effects],
        {"validator_id": "atomic_test"},
        [],
        {},
        {},
        SkillStatus.CANDIDATE,
    )


def _implementation(
    logical_id: str,
    *,
    mapping: BindingExpression | None = None,
    constraints: tuple[GroundingConstraint, ...] = (),
    compatibility: dict | None = None,
) -> ImplementationAtom:
    expression = mapping or BindingExpression(BindingExprKind.SKILL_INPUT, source_role="item")
    return ImplementationAtom(
        SkillRef(logical_id, "1.0.0"),
        SkillRef("atomic_prepare", "1.0.0"),
        [ToolBinding(ToolRef("tool_prepare", "1.0.0"), "prepare", {"item": expression}, 0)],
        list(constraints),
        {
            "mode": "serial",
            "output_mapping": {
                "result": BindingExpression(
                    BindingExprKind.TOOL_OUTPUT,
                    source_role="result",
                    source_step="prepare",
                ),
            },
        },
        compatibility or {"harness_profiles": ["alfworld"]},
        {},
        SkillStatus.CANDIDATE,
    )


def _evidence(cluster: str = "stable", *, candidate_keys: tuple[str, ...] = ()) -> list[RepairEvidence]:
    return [
        RepairEvidence(
            f"evidence_{index}", f"task_{index}", f"trace_{index}", cluster,
            {"case": index}, candidate_keys=candidate_keys,
        )
        for index in (1, 2)
    ]


@pytest.fixture
def repair_context(tmp_path):
    database = StateDatabase(tmp_path / "state.sqlite3")
    registry = SkillRegistry(ArtifactStore(tmp_path, database), database)
    engine = TypedRepairEngine(RepairStore(database), registry)
    yield database, registry, engine
    database.close()


def _execute(engine: TypedRepairEngine, proposal):
    return engine.execute(
        proposal,
        replay=lambda _candidate, case: case["case"] in {1, 2},
        validate=lambda _candidate: {"passed": True, "validator": "test"},
        admit=lambda candidate: candidate,
    )


def test_atomic_revise_split_and_merge_create_new_candidates_with_lineage(repair_context) -> None:
    _database, registry, engine = repair_context
    original = _atomic("atomic_prepare")
    registry.register_atomic(original)

    revision = replace(original, guideline={"stable": "prepare item safely"})
    revised = _execute(
        engine,
        engine.propose_atomic_revision(original.ref, revision, _evidence("revise")),
    )
    assert revised.proposal.status == "admitted"
    assert revised.admitted_refs == ("skill://atomic_prepare@1.0.1",)
    assert revised.lineage[0].relation is GlobalRelationType.DERIVED_FROM
    assert registry.get_atomic(original.ref).summary == "prepare item"
    assert registry.get_atomic(revised.admitted_refs[0]).status is SkillStatus.CANDIDATE

    compound = _atomic(
        "atomic_compound", effects=("object.ready", "object.checked"),
    )
    registry.register_atomic(compound)
    left = _atomic("atomic_ready", effects=("object.ready",))
    right = _atomic("atomic_checked", effects=("object.checked",))
    split = _execute(
        engine,
        engine.propose_atomic_split(
            compound.ref,
            [left, right],
            _evidence("split", candidate_keys=("atomic_ready", "atomic_checked")),
        ),
    )
    assert split.proposal.status == "admitted"
    assert set(split.admitted_refs) == {
        "skill://atomic_ready@1.0.0", "skill://atomic_checked@1.0.0",
    }
    assert {item.relation for item in split.lineage} == {GlobalRelationType.SPLIT_FROM}

    duplicate = _atomic("atomic_prepare_duplicate")
    registry.register_atomic(duplicate)
    merged_candidate = _atomic("atomic_prepare_merged")
    merged = _execute(
        engine,
        engine.propose_atomic_merge(
            [original.ref, duplicate.ref], merged_candidate, _evidence("merge"),
        ),
    )
    assert merged.proposal.status == "admitted"
    assert merged.admitted_refs == ("skill://atomic_prepare_merged@1.0.0",)
    assert len(merged.lineage) == 2
    assert {item.relation for item in merged.lineage} == {GlobalRelationType.MERGED_FROM}


def test_atomic_revision_rejects_summary_or_provenance_only_change(repair_context) -> None:
    _database, registry, engine = repair_context
    original = _atomic("atomic_prepare")
    registry.register_atomic(original)
    summary_only = _execute(
        engine,
        engine.propose_atomic_revision(
            original.ref, replace(original, summary="different words"), _evidence("summary"),
        ),
    )
    assert summary_only.proposal.replay_result["failure_code"] == "semantic_edit_empty"

    metadata_only = _execute(
        engine,
        engine.propose_atomic_revision(
            original.ref,
            replace(original, guideline={"real": "semantic"}, metadata={"forged": True}),
            _evidence("metadata"),
        ),
    )
    assert metadata_only.proposal.replay_result["failure_code"] == "atomic_revision_scope_invalid"


def test_implementation_mapping_constraint_and_specialize_are_typed_candidates(repair_context) -> None:
    _database, registry, engine = repair_context
    registry.register_atomic(_atomic("atomic_prepare"))
    original = _implementation("implementation_prepare")
    registry.register_implementation(original)

    mapping_candidate = replace(
        original,
        tool_bindings=[replace(
            original.tool_bindings[0],
            parameter_mapping={"item": BindingExpression(BindingExprKind.CONSTANT, constant="apple_1")},
        )],
    )
    mapping = _execute(
        engine,
        engine.propose_implementation_mapping_revision(
            original.ref, mapping_candidate, _evidence("mapping"),
        ),
    )
    assert mapping.proposal.status == "admitted"
    assert mapping.admitted_refs == ("skill://implementation_prepare@1.0.1",)

    constraint = GroundingConstraint(
        "item_exists",
        GroundingConstraintKind.ARGUMENT_EXISTS,
        argument_mapping={
            "item": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="item"),
        },
        verifier_id="alfworld.argument_exists",
    )
    constraint_candidate = replace(original, grounding_constraints=[constraint])
    constrained = _execute(
        engine,
        engine.propose_implementation_constraint_revision(
            original.ref, constraint_candidate, _evidence("constraint"),
        ),
    )
    assert constrained.proposal.status == "admitted"
    assert constrained.admitted_refs == ("skill://implementation_prepare@1.0.2",)

    specialized_candidate = replace(
        original,
        ref=SkillRef("implementation_prepare_kitchen", "1.0.0"),
        compatibility={"harness_profiles": ["alfworld"], "room_type": "kitchen"},
    )
    specialized = _execute(
        engine,
        engine.propose_implementation_specialization(
            original.ref, specialized_candidate, _evidence("kitchen"),
        ),
    )
    assert specialized.proposal.status == "admitted"
    assert specialized.admitted_refs == ("skill://implementation_prepare_kitchen@1.0.0",)
    assert specialized.lineage[0].relation is GlobalRelationType.DERIVED_FROM


def test_mapping_cannot_replace_tool_or_policy_and_specialize_cannot_broaden_domain(repair_context) -> None:
    _database, registry, engine = repair_context
    registry.register_atomic(_atomic("atomic_prepare"))
    original = _implementation("implementation_prepare")
    registry.register_implementation(original)

    changed_tool = replace(
        original,
        tool_bindings=[replace(
            original.tool_bindings[0],
            tool_ref=ToolRef("different_tool", "1.0.0"),
            parameter_mapping={"item": BindingExpression(BindingExprKind.CONSTANT, constant="apple_1")},
        )],
    )
    rejected_tool = _execute(
        engine,
        engine.propose_implementation_mapping_revision(
            original.ref, changed_tool, _evidence("mapping_tool"),
        ),
    )
    assert rejected_tool.proposal.replay_result["failure_code"] == "mapping_revision_scope_invalid"

    changed_policy = replace(
        original,
        execution_policy={
            **original.execution_policy,
            "mode": "parallel",
            "output_mapping": {},
        },
    )
    rejected_policy = _execute(
        engine,
        engine.propose_implementation_mapping_revision(
            original.ref, changed_policy, _evidence("mapping_policy"),
        ),
    )
    assert rejected_policy.proposal.replay_result["failure_code"] == "mapping_revision_scope_invalid"

    broadened = replace(
        original,
        ref=SkillRef("implementation_prepare_broad", "1.0.0"),
        compatibility={"harness_profiles": ["alfworld", "other"]},
    )
    rejected_specialization = _execute(
        engine,
        engine.propose_implementation_specialization(
            original.ref, broadened, _evidence("broad"),
        ),
    )
    assert rejected_specialization.proposal.replay_result["failure_code"] == "specialization_domain_broadened"


@pytest.mark.parametrize(
    ("evidence", "failure_code"),
    [
        (
            [RepairEvidence(
                "only", "task", "trace", "stable", {"case": 1},
                failure_layer="runtime_agent", agent_parameter_error=True,
            )],
            "agent_parameter_error_not_artifact_evidence",
        ),
        (
            [*_evidence("first")[:1], *_evidence("second")[1:]],
            "heterogeneous_failure_cluster",
        ),
    ],
)
def test_specialization_rejects_single_agent_error_and_heterogeneous_cluster(
    repair_context, evidence, failure_code,
) -> None:
    _database, registry, engine = repair_context
    registry.register_atomic(_atomic("atomic_prepare"))
    original = _implementation("implementation_prepare")
    registry.register_implementation(original)
    candidate = replace(
        original,
        ref=SkillRef("implementation_prepare_special", "1.0.0"),
        compatibility={"harness_profiles": ["alfworld"], "domain": "special"},
    )
    outcome = _execute(
        engine,
        engine.propose_implementation_specialization(original.ref, candidate, evidence),
    )
    assert outcome.proposal.status == "rejected"
    assert outcome.proposal.replay_result["failure_code"] == failure_code
    assert "skill://implementation_prepare_special@1.0.0" not in {
        str(item) for item in registry.list_refs("implementation")
    }


def test_replay_rejection_closes_proposal_but_unexpected_and_infrastructure_propagate(repair_context) -> None:
    _database, registry, engine = repair_context
    original = _atomic("atomic_prepare")
    registry.register_atomic(original)
    candidate = replace(original, guideline={"stable": "revised"})

    replay_rejected = engine.propose_atomic_revision(original.ref, candidate, _evidence("replay"))
    outcome = engine.execute(
        replay_rejected,
        replay=lambda _candidate, case: case["case"] == 1,
        validate=lambda _candidate: True,
        admit=lambda admitted: admitted,
    )
    assert outcome.proposal.status == "rejected"
    assert outcome.proposal.replay_result["failure_code"] == "source_replay_failed"

    unexpected = engine.propose_atomic_revision(original.ref, candidate, _evidence("unexpected"))
    with pytest.raises(RuntimeError, match="callback bug"):
        engine.execute(
            unexpected,
            replay=lambda _candidate, _case: (_ for _ in ()).throw(RuntimeError("callback bug")),
            validate=lambda _candidate: True,
            admit=lambda admitted: admitted,
        )
    assert RepairStore(_database).pending()[0].status == "replaying"

    infrastructure = engine.propose_atomic_revision(original.ref, candidate, _evidence("infra"))
    with pytest.raises(AtomicSkillGraphError, match="provider unavailable"):
        engine.execute(
            infrastructure,
            replay=lambda _candidate, _case: (_ for _ in ()).throw(AtomicSkillGraphError(
                "infrastructure_failure", "provider unavailable", layer=FailureLayer.INFRASTRUCTURE,
            )),
            validate=lambda _candidate: True,
            admit=lambda admitted: admitted,
        )
    assert any(
        item.proposal_id == infrastructure.proposal_id and item.status == "replaying"
        for item in RepairStore(_database).pending()
    )
