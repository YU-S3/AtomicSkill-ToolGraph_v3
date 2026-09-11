"""R9 gates for shared Planner/Runtime Support role authority."""

from __future__ import annotations

from typing import Any

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    AtomicCandidate,
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeLinearPlan, RuntimeOccurrence
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.core.support_authority import support_role_authority
from atomic_skillgraph.planner.support_retriever import (
    PlannerSupportAtomicRetriever,
    PlannerSupportCandidate,
    PlannerSupportRoleMapping,
)
from atomic_skillgraph.planner.validator import PlannerValidator
from atomic_skillgraph.runtime.support_retriever import SupportAtomicRetriever


def _input(role: str) -> BindingExpression:
    return BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)


def _atomic(
    name: str,
    *,
    inputs: list[ParameterSpec] | None = None,
    outputs: list[ParameterSpec] | None = None,
    preconditions: list[SemanticPredicate] | None = None,
    effects: list[SemanticPredicate] | None = None,
    output_derivations: dict[str, dict[str, str]] | None = None,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(name, "1.0.0"),
        summary=name,
        inputs=list(inputs or ()),
        outputs=list(outputs or ()),
        preconditions=list(preconditions or ()),
        effects=list(effects or ()),
        validator_spec={
            "output_derivations": dict(output_derivations or {}),
        },
        failure_modes=[],
        guideline={},
        metadata={"harness_profiles": ["r9_support"]},
        status=SkillStatus.ACTIVE,
    )


class _Skills:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self.atomics_by_ref = {str(item.ref): item for item in atomics}

    def atomics(
        self, *, mode: RuntimeMode | str,
    ) -> list[AbstractAtomicSkill]:
        RuntimeMode(mode)
        return list(self.atomics_by_ref.values())

    def get_atomic(self, ref: SkillRef | str) -> AbstractAtomicSkill:
        return self.atomics_by_ref[str(ref)]


class _Graph:
    @staticmethod
    def existing_edge_by_id(*_args: Any, **_kwargs: Any) -> None:
        return None


def _retrieve(
    consumer: AbstractAtomicSkill,
    *producers: AbstractAtomicSkill,
):
    retriever = PlannerSupportAtomicRetriever(
        _Skills(consumer, *producers),
        top_k=5,
    )
    result = retriever.retrieve(
        required_instance_candidates={
            "required::0": [AtomicCandidate(consumer.ref, 1.0)],
        },
        mode=RuntimeMode.ONLINE,
        harness_profile="r9_support",
        task_id="r9_support",
    )
    return retriever, result


def test_planner_support_excludes_runtime_resolvable_inputs() -> None:
    consumer = _atomic(
        "heat",
        inputs=[
            ParameterSpec("object", "entity", runtime_resolvable=True),
            ParameterSpec("station", "entity", runtime_resolvable=True),
        ],
        effects=[SemanticPredicate(
            "object.heated", {"object": _input("object")},
        )],
    )
    producer = _atomic(
        "take_under_light",
        outputs=[
            ParameterSpec("object", "entity"),
            ParameterSpec("station", "entity"),
        ],
        effects=[
            SemanticPredicate("agent.holds", {"object": _input("object")}),
            SemanticPredicate(
                "station.available", {"station": _input("station")},
            ),
        ],
    )

    retriever, candidates = _retrieve(consumer, producer)

    assert candidates == []
    assert retriever.last_runtime_resolvable_role_exclusion_count == 2


def test_relation_verified_evidence_exception_remains_authorized() -> None:
    consumer = _atomic(
        "relation_consumer",
        inputs=[ParameterSpec(
            "object",
            "entity",
            runtime_resolvable=False,
            required_resolution="relation_verified",
        )],
        effects=[SemanticPredicate(
            "object.ready", {"object": _input("object")},
        )],
    )
    producer = _atomic(
        "discover_entity",
        outputs=[ParameterSpec("entity", "entity")],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": _input("entity"), "location": "known_location"},
            effect_domain=EffectDomain.EVIDENCE,
        )],
    )

    authority = support_role_authority(
        producer, "entity", consumer, "object",
    )
    _retriever, candidates = _retrieve(consumer, producer)

    assert authority.authorized is True
    assert authority.relation_verified_exception is True
    assert [item.atomic_ref for item in candidates] == [str(producer.ref)]


def test_type_compatible_light_to_station_has_no_semantic_authority() -> None:
    consumer = _atomic(
        "heat_station_consumer",
        inputs=[ParameterSpec(
            "station", "entity", runtime_resolvable=False,
        )],
        effects=[SemanticPredicate(
            "object.heated", {"station": _input("station")},
        )],
    )
    producer = _atomic(
        "light_producer",
        inputs=[ParameterSpec("light_input", "entity")],
        outputs=[ParameterSpec("light", "entity")],
        effects=[SemanticPredicate(
            "light.active", {"light": _input("light")},
        )],
    )

    authority = support_role_authority(
        producer, "light", consumer, "station",
    )
    retriever, candidates = _retrieve(consumer, producer)

    assert authority.authorized is False
    assert authority.reason == "semantic_role_authority_missing"
    assert candidates == []
    assert retriever.last_role_authority_rejection_count == 1


def test_input_identity_alias_authorizes_placed_object_to_object() -> None:
    producer = _atomic(
        "place",
        inputs=[ParameterSpec("object", "entity")],
        outputs=[ParameterSpec("placed_object", "entity")],
        effects=[SemanticPredicate(
            "object.at_location",
            {"object": _input("placed_object"), "location": "destination"},
        )],
        output_derivations={
            "placed_object": {
                "kind": "input_identity",
                "input_role": "object",
            },
        },
    )
    consumer = _atomic(
        "consume_object",
        inputs=[ParameterSpec(
            "object", "entity", runtime_resolvable=False,
        )],
        preconditions=[SemanticPredicate(
            "object.at_location", {"object": _input("object")},
        )],
        effects=[SemanticPredicate(
            "object.consumed", {"object": _input("object")},
        )],
    )

    authority = support_role_authority(
        producer, "placed_object", consumer, "object",
    )

    assert authority.authorized is True
    assert authority.semantic_alias_authorized is True
    assert "object" in authority.producer_aliases


def test_runtime_support_uses_same_identity_alias_authority() -> None:
    producer = _atomic(
        "take",
        inputs=[ParameterSpec("object", "entity")],
        outputs=[ParameterSpec("held_object", "entity")],
        effects=[SemanticPredicate(
            "agent.holds", {"object": _input("held_object")},
        )],
        output_derivations={
            "held_object": {
                "kind": "input_identity",
                "input_role": "object",
            },
        },
    )
    consumer = _atomic(
        "runtime_consumer",
        inputs=[ParameterSpec(
            "object", "entity", runtime_resolvable=True,
        )],
        preconditions=[SemanticPredicate(
            "agent.holds", {"object": _input("object")},
        )],
        effects=[SemanticPredicate(
            "object.ready", {"object": _input("object")},
        )],
    )

    candidates = SupportAtomicRetriever().retrieve(
        blocked_atomic=consumer,
        missing_roles=["object"],
        atomics=[producer, consumer],
    )

    assert len(candidates) == 1
    assert candidates[0].role_mappings[0].producer_role == "held_object"
    assert candidates[0].role_mappings[0].consumer_role == "object"


def test_validator_rechecks_invented_support_mapping_fail_closed() -> None:
    producer = _atomic(
        "bad_light_support",
        outputs=[ParameterSpec("light", "entity")],
        effects=[SemanticPredicate(
            "light.active", {"light": _input("light")},
        )],
    )
    consumer = _atomic(
        "station_consumer",
        inputs=[ParameterSpec(
            "station", "entity", runtime_resolvable=False,
        )],
        effects=[SemanticPredicate(
            "station.used", {"station": _input("station")},
        )],
    )
    candidate = PlannerSupportCandidate(
        atomic_ref=str(producer.ref),
        consumer_requirement_instance_id="required::0",
        score=1.0,
        role_mappings=(PlannerSupportRoleMapping(
            producer_role="light",
            consumer_role="station",
            semantic_type="entity",
            producer_resolution="concrete",
            required_resolution="semantic",
            effect_domain="world",
            consumer_atomic_ref=str(consumer.ref),
        ),),
        output_roles=("light",),
        effect_predicates=("light.active",),
    )
    plan = RuntimeLinearPlan(
        task_id="invented_support",
        source="atomic_composition",
        source_composite_ref=None,
        occurrences=[
            RuntimeOccurrence(
                "support", "occ_support", producer.ref, [], {}, [],
                producer.effects, requirement_instance_ids=[],
            ),
            RuntimeOccurrence(
                "required",
                "occ_required",
                consumer.ref,
                ["required::0"],
                {"station": BindingExpression(
                    BindingExprKind.DATA_FLOW,
                    source_step="support",
                    source_role="light",
                )},
                [],
                consumer.effects,
                requirement_instance_ids=["required::0"],
            ),
        ],
        control_sequence=["support", "required"],
        data_edges=[GraphEdge(
            "invented_light_station",
            GraphEdgeType.DATA_FLOW,
            "support",
            "required",
            "light",
            "station",
            "planner_proposed",
        )],
        dependency_edges=[],
        task_contract=TaskContract(target_effects=list(consumer.effects)),
        planner_audit={
            "requirement_coverage": {"required::0": ["required"]},
        },
    )

    result = PlannerValidator(
        _Skills(producer, consumer), _Graph(),
    ).validate(
        plan,
        mode=RuntimeMode.ONLINE,
        required_requirement_ids=["required::0"],
        harness_profile="r9_support",
        support_candidates=[candidate],
    )

    assert result.passed is False
    assert "planner_support_atomic_invalid" in result.failure_codes
    assert result.checks["support_role_mappings_authorized"] is False
