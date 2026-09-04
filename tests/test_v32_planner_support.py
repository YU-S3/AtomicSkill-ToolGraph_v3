"""Regression gates for the v3.2-R1 Planner Support-Atomic closure."""

from __future__ import annotations

from dataclasses import replace

from atomic_skillgraph.agents.protocol import validate_schema_instance
from atomic_skillgraph.agents.structured_submission import PROPOSED_OCCURRENCE_SCHEMA
from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    AtomicCandidate,
    ContractSource,
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeLinearPlan, RuntimeOccurrence
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.planner.support_retriever import (
    PlannerSupportAtomicRetriever,
    PlannerSupportCandidate,
    PlannerSupportRoleMapping,
)
from atomic_skillgraph.planner.validator import PlannerValidator


def _atomic(
    name: str,
    *,
    inputs: list[ParameterSpec] | None = None,
    outputs: list[ParameterSpec] | None = None,
    effects: list[SemanticPredicate] | None = None,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"),
        summary=name,
        inputs=list(inputs or ()),
        outputs=list(outputs or ()),
        preconditions=[],
        effects=list(effects or ()),
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )


class _Skills:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self._atomics = list(atomics)
        self._by_ref = {str(item.ref): item for item in atomics}

    def atomics(self, *, mode: RuntimeMode | str | None = None):
        del mode
        return list(self._atomics)

    def get_atomic(self, ref: SkillRef | str) -> AbstractAtomicSkill:
        return self._by_ref[str(ref)]


class _NoExistingEdges:
    def existing_edge_by_id(self, *_args, **_kwargs):
        return None


def test_planner_schema_allows_empty_instance_ids_for_support_occurrence() -> None:
    validate_schema_instance(
        {
            "step_id": "support",
            "occurrence_id": "occ_support",
            "node_ref": {
                "logical_id": "atomic_locate",
                "version": "1.0.0",
            },
            "requirement_instance_ids": [],
            "repeat_role_bindings": {},
            "binding_specs": {},
        },
        PROPOSED_OCCURRENCE_SCHEMA,
    )


def _fixture_atomics() -> tuple[
    AbstractAtomicSkill, AbstractAtomicSkill, AbstractAtomicSkill,
]:
    required = _atomic(
        "atomic_heat",
        inputs=[
            ParameterSpec(
                "object",
                "entity",
                runtime_resolvable=False,
                required_resolution="relation_verified",
            ),
        ],
        effects=[SemanticPredicate("object.heated", {"object": "$object"})],
    )
    support = _atomic(
        "atomic_locate",
        outputs=[ParameterSpec("entity", "entity")],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "source_location"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
        ],
    )
    world_only = _atomic(
        "atomic_world_only",
        outputs=[ParameterSpec("found", "entity")],
        effects=[SemanticPredicate("object.visible", {"object": "$found"})],
    )
    return required, support, world_only


def test_planner_support_retrieval_uses_type_resolution_and_evidence_domain() -> None:
    required, support, world_only = _fixture_atomics()
    candidates = PlannerSupportAtomicRetriever(
        _Skills(required, support, world_only),
        top_k=3,
    ).retrieve(
        required_instance_candidates={
            "req_heat::0": [AtomicCandidate(required.ref, 1.0)],
        },
        mode=RuntimeMode.ONLINE,
        harness_profile="alfworld_v3",
        task_id="task",
    )

    assert [item.atomic_ref for item in candidates] == [str(support.ref)]
    assert candidates[0].role_mappings == (
        PlannerSupportRoleMapping(
            producer_role="entity",
            consumer_role="object",
            semantic_type="entity",
            producer_resolution="relation_verified",
            required_resolution="relation_verified",
            effect_domain="evidence",
            consumer_atomic_ref=str(required.ref),
        ),
    )


def _candidate(
    support: AbstractAtomicSkill,
    required: AbstractAtomicSkill,
) -> PlannerSupportCandidate:
    return PlannerSupportCandidate(
        atomic_ref=str(support.ref),
        consumer_requirement_instance_id="req_heat::0",
        score=1.0,
        role_mappings=(
            PlannerSupportRoleMapping(
                producer_role="entity",
                consumer_role="object",
                semantic_type="entity",
                producer_resolution="relation_verified",
                required_resolution="relation_verified",
                effect_domain="evidence",
                consumer_atomic_ref=str(required.ref),
            ),
        ),
        output_roles=("entity",),
        effect_predicates=("entity.discovered_at",),
    )


def _plan(
    required: AbstractAtomicSkill,
    support: AbstractAtomicSkill,
    *,
    consume_support: bool,
) -> RuntimeLinearPlan:
    required_binding = (
        BindingExpression(
            BindingExprKind.DATA_FLOW,
            source_role="entity",
            source_step="support",
        )
        if consume_support
        else BindingExpression(BindingExprKind.CONSTANT, constant="apple_1")
    )
    data_edges = (
        [
            GraphEdge(
                "support_to_required",
                GraphEdgeType.DATA_FLOW,
                "support",
                "required",
                "entity",
                "object",
                "planner_proposed",
            ),
        ]
        if consume_support
        else []
    )
    return RuntimeLinearPlan(
        task_id="task",
        source="atomic_composition",
        source_composite_ref=None,
        occurrences=[
            RuntimeOccurrence(
                "support",
                "occ_support",
                support.ref,
                [],
                {},
                [],
                support.effects,
                requirement_instance_ids=[],
            ),
            RuntimeOccurrence(
                "required",
                "occ_required",
                required.ref,
                ["req_heat::0"],
                {"object": required_binding},
                [],
                required.effects,
                requirement_instance_ids=["req_heat::0"],
            ),
        ],
        control_sequence=["support", "required"],
        data_edges=data_edges,
        dependency_edges=[],
        task_contract=TaskContract(
            target_effects=list(required.effects),
            source=ContractSource.ADAPTER_DERIVED,
        ),
        planner_audit={"requirement_coverage": {"req_heat::0": ["required"]}},
    )


def test_planner_accepts_support_only_when_formal_output_is_consumed() -> None:
    required, support, _world_only = _fixture_atomics()
    validator = PlannerValidator(_Skills(required, support), _NoExistingEdges())
    candidate = _candidate(support, required)

    valid = validator.validate(
        _plan(required, support, consume_support=True),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["req_heat::0"],
        harness_profile="alfworld_v3",
        support_candidates=[candidate],
    )
    assert valid.passed is True
    assert valid.checks["support_occurrences_authorized"] is True
    assert valid.checks[
        "support_outputs_consumed_by_required_occurrence"
    ] is True
    assert valid.checks["support_data_flow_mappings_valid"] is True

    unused = validator.validate(
        _plan(required, support, consume_support=False),
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["req_heat::0"],
        harness_profile="alfworld_v3",
        support_candidates=[candidate],
    )
    assert unused.passed is False
    assert "planner_support_atomic_invalid" in unused.failure_codes
    assert unused.checks[
        "support_outputs_consumed_by_required_occurrence"
    ] is False


def test_planner_rejects_invented_support_role_mapping() -> None:
    required, support, _world_only = _fixture_atomics()
    required_with_alternative_role = replace(
        required,
        inputs=[
            *required.inputs,
            ParameterSpec(
                "destination",
                "entity",
                runtime_resolvable=False,
                required_resolution="relation_verified",
            ),
        ],
    )
    plan = _plan(required_with_alternative_role, support, consume_support=True)
    plan.data_edges[0] = replace(plan.data_edges[0], target_role="destination")
    plan.occurrences[1].binding_specs = {
        "object": BindingExpression(BindingExprKind.CONSTANT, constant="apple_1"),
        "destination": BindingExpression(
            BindingExprKind.DATA_FLOW,
            source_role="entity",
            source_step="support",
        ),
    }

    result = PlannerValidator(
        _Skills(required_with_alternative_role, support),
        _NoExistingEdges(),
    ).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["req_heat::0"],
        harness_profile="alfworld_v3",
        support_candidates=[_candidate(support, required_with_alternative_role)],
    )

    assert result.passed is False
    assert "planner_support_atomic_invalid" in result.failure_codes
    assert result.checks["edge_roles_valid"] is True
    assert result.checks["edge_semantic_types_compatible"] is True
    assert result.checks["support_data_flow_mappings_valid"] is False
