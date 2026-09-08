from __future__ import annotations

import pytest

from atomic_skillgraph.agents import (
    ReplayAgentSession,
    SchemaValidationError,
    UsageLedger,
    validate_schema_instance,
)
from atomic_skillgraph.agents.structured_submission import (
    COMPOSITE_EXTRACTION_SCHEMA,
)
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.edges import ExistingEdgeEvidence
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.composite_edge_candidates import (
    CompositeEdgeCandidateBuilder,
)
from atomic_skillgraph.evolution.extractor_session import (
    ExtractionContentError,
    ExtractorSession,
)
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from experiments.fakes import FakeReply, ScriptedAgentProvider


def _expr(role: str) -> BindingExpression:
    return BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)


def _occurrence(
    occurrence_id: str,
    *,
    ref: str,
    inputs: dict[str, tuple[object, str]],
    outputs: dict[str, tuple[object, str]],
    preconditions: list[SemanticPredicate] | None = None,
    effects: list[SemanticPredicate] | None = None,
    index: int = 0,
) -> CanonicalAtomicOccurrence:
    return CanonicalAtomicOccurrence(
        occurrence_id=occurrence_id,
        phase_id=occurrence_id,
        intent=f"capability_{occurrence_id}",
        event_start=index,
        event_end=index,
        input_bindings={key: value[0] for key, value in inputs.items()},
        output_bindings={key: value[0] for key, value in outputs.items()},
        input_specs=[
            ParameterSpec(key, value[1], True, True, "concrete")
            for key, value in sorted(inputs.items())
        ],
        output_specs=[
            ParameterSpec(key, value[1])
            for key, value in sorted(outputs.items())
        ],
        preconditions=list(preconditions or []),
        effects=list(effects or []),
        action_events=[],
        prefix_events=[],
        source_task={},
        source_trace_id="trace",
        proposed_ref=SkillRef(ref, "1.0.0"),
    )


def _chain() -> list[CanonicalAtomicOccurrence]:
    source = _occurrence(
        "source",
        ref="atomic_source",
        inputs={"object": ("item_1", "entity")},
        outputs={"held_object": ("item_1", "entity")},
        effects=[SemanticPredicate(
            "state.ready", {"object": _expr("object")},
        )],
    )
    target = _occurrence(
        "target",
        ref="atomic_target",
        inputs={"target_object": ("item_1", "entity")},
        outputs={"result": ("item_1", "entity")},
        preconditions=[SemanticPredicate(
            "state.ready", {"object": _expr("target_object")},
        )],
        effects=[SemanticPredicate(
            "state.done", {"object": _expr("target_object")},
        )],
        index=1,
    )
    return [source, target]


def _session(reply) -> ExtractorSession:
    provider = ScriptedAgentProvider([FakeReply.structured(reply)])
    extractor = ExtractorSession(ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e2",
    ))
    extractor._e1_complete = True
    return extractor


def _repair_session(*replies) -> tuple[
    ExtractorSession, ScriptedAgentProvider,
]:
    provider = ScriptedAgentProvider([
        FakeReply.structured(reply) for reply in replies
    ])
    extractor = ExtractorSession(ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e2",
    ))
    extractor._e1_complete = True
    return extractor, provider


def _e2_payload(*candidate_ids: str) -> dict[str, object]:
    return {
        "selected_existing_edge_ids": [],
        "selected_new_edge_candidate_ids": list(candidate_ids),
        "summary": "compose reusable capabilities",
        "guideline": {},
        "insight": {},
    }


def _chain_contract() -> TaskContract:
    return TaskContract(target_effects=[SemanticPredicate(
        "state.done", {"object": "item_1"},
    )])


def test_dataflow_is_identity_typed_nearest_and_dependency_is_effect_based() -> None:
    first = _occurrence(
        "first",
        ref="atomic_first",
        inputs={"object": ("item_1", "entity")},
        outputs={"result": ("item_1", "entity")},
        effects=[SemanticPredicate(
            "state.ready", {"object": _expr("object")},
        )],
    )
    nearer, target = _chain()
    nearer.occurrence_id = "nearer"
    nearer.event_start = nearer.event_end = 1
    target.event_start = target.event_end = 2
    candidates = CompositeEdgeCandidateBuilder().build(
        [first, nearer, target]
    )
    flows = [item for item in candidates if item.edge_type == "data_flow"]
    target_flow = next(
        item for item in flows
        if item.target_step == "target"
        and item.target_role == "target_object"
    )
    assert target_flow.source_step == "nearer"
    assert target_flow.authority == "binding_identity_match"
    dependencies = [
        item for item in candidates
        if item.edge_type == "requires_skill"
        and item.target_step == "target"
    ]
    assert dependencies
    assert all(
        item.authority == "effect_precondition_compatibility"
        for item in dependencies
    )

    incompatible_target = _occurrence(
        "scalar_target",
        ref="atomic_scalar_target",
        inputs={"target_object": ("item_1", "string")},
        outputs={"result": ("item_1", "string")},
        index=2,
    )
    incompatible = CompositeEdgeCandidateBuilder().build(
        [first, incompatible_target]
    )
    assert not any(item.edge_type == "data_flow" for item in incompatible)


def test_e2_schema_is_minimal_and_model_cannot_author_graph_facts() -> None:
    properties = COMPOSITE_EXTRACTION_SCHEMA["properties"]
    assert "control_sequence" not in properties
    assert "existing_edges" not in properties
    assert "new_edges" not in properties
    valid = {
        "selected_existing_edge_ids": [],
        "selected_new_edge_candidate_ids": [],
        "summary": "compose reusable capabilities",
        "guideline": {},
        "insight": {},
    }
    validate_schema_instance(valid, COMPOSITE_EXTRACTION_SCHEMA)
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_schema_instance(
            {**valid, "control_sequence": ["forged"]},
            COMPOSITE_EXTRACTION_SCHEMA,
        )
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_schema_instance(
            {**valid, "origin": "extractor_validated"},
            COMPOSITE_EXTRACTION_SCHEMA,
        )


def test_selected_candidate_is_code_materialized_as_standard_graph_edge() -> None:
    chain = _chain()

    def reply(request):
        candidate = next(
            item for item in request.policy_context["new_edge_candidates"]
            if item["edge_type"] == "data_flow"
        )
        return {
            "selected_existing_edge_ids": [],
            "selected_new_edge_candidate_ids": [candidate["candidate_id"]],
            "summary": "establish then consume reusable state",
            "guideline": {},
            "insight": {},
        }

    proposal = _session(reply).propose_composite(chain, [])
    assert proposal.control_sequence == ["source", "target"]
    assert proposal.new_edges[0]["edge_id"].startswith("edge_")
    assert "origin" not in proposal.new_edges[0]
    composite = CompositeBuilder().validate_and_build(
        proposal,
        chain,
        TaskContract(target_effects=[SemanticPredicate(
            "state.done", {"object": "item_1"},
        )]),
        contract_matcher=ExactContractMatcher(),
        task_bindings={"object": "item_1"},
    )
    assert len(composite.data_edges) == 1
    assert composite.data_edges[0].origin == "extractor_validated"


@pytest.mark.parametrize(
    ("payload", "error_code"),
    [
        (
            {
                "selected_existing_edge_ids": ["unknown_edge"],
                "selected_new_edge_candidate_ids": [],
                "summary": "compose reusable capabilities",
                "guideline": {},
                "insight": {},
            },
            "extractor_e2_existing_edge_selection_invalid",
        ),
        (
            {
                "selected_existing_edge_ids": [],
                "selected_new_edge_candidate_ids": ["unknown_candidate"],
                "summary": "compose reusable capabilities",
                "guideline": {},
                "insight": {},
            },
            "extractor_e2_new_edge_selection_invalid",
        ),
    ],
)
def test_unknown_e2_ids_are_typed_semantic_rejections(
    payload,
    error_code,
) -> None:
    extractor = _session(payload)
    with pytest.raises(ExtractionContentError) as caught:
        extractor.propose_composite(_chain(), [])
    assert caught.value.stage == "e2"
    assert caught.value.error_code == error_code
    assert extractor._e2_complete is False
    with pytest.raises(
        RuntimeError,
        match="requires one schema-valid initial E2",
    ):
        extractor.repair_composite(
            _e2_payload(),
            caught.value,
            _chain(),
            [],
        )


def test_existing_edge_is_selected_only_by_known_id_and_code_maps_endpoints() -> None:
    chain = _chain()
    evidence = ExistingEdgeEvidence(
        edge_id="known_edge",
        source_composite_ref="skill://old@1.0.0",
        source_step_ref=str(chain[0].proposed_ref),
        target_step_ref=str(chain[1].proposed_ref),
        edge_type="data_flow",
        source_role="held_object",
        target_role="target_object",
        semantic_types=("entity", "entity"),
        support_trace_ids=("old_trace",),
    )

    def reply(request):
        known = request.policy_context["known_existing_edge_evidence"]
        assert known == [{
            "edge_id": "known_edge",
            "edge_type": "data_flow",
            "source_step": "source",
            "target_step": "target",
            "source_role": "held_object",
            "target_role": "target_object",
            "authority": "existing_active",
        }]
        return {
            "selected_existing_edge_ids": [known[0]["edge_id"]],
            "selected_new_edge_candidate_ids": [],
            "summary": "reuse verified data flow",
            "guideline": {},
            "insight": {},
        }

    proposal = _session(reply).propose_composite(chain, [evidence])
    assert proposal.existing_edges[0] == {
        "edge_id": "known_edge",
        "existing_edge_id": "known_edge",
        "edge_type": "data_flow",
        "source_step": "source",
        "target_step": "target",
        "source_role": "held_object",
        "target_role": "target_object",
    }


def test_e2_repair_selects_missing_dataflow_from_frozen_candidates() -> None:
    chain = _chain()
    initial_context: dict[str, object] = {}

    def initial_reply(request):
        initial_context.update(request.policy_context)
        return _e2_payload()

    def repaired_reply(request):
        context = request.policy_context
        assert context["canonical_control_sequence"] == [
            "source", "target",
        ]
        assert context["canonical_occurrences"] == initial_context[
            "canonical_occurrences"
        ]
        assert context["known_existing_edge_evidence"] == initial_context[
            "known_existing_edge_evidence"
        ]
        assert context["new_edge_candidates"] == initial_context[
            "new_edge_candidates"
        ]
        candidate = next(
            item for item in context["new_edge_candidates"]
            if item["edge_type"] == "data_flow"
        )
        return _e2_payload(candidate["candidate_id"])

    extractor, provider = _repair_session(
        initial_reply,
        repaired_reply,
    )
    proposal = extractor.propose_composite(chain, [])
    builder = CompositeBuilder()
    with pytest.raises(
        ValueError,
        match="reused required input must have explicit DataFlow",
    ) as rejected:
        builder.validate_and_build(
            proposal,
            chain,
            _chain_contract(),
            contract_matcher=ExactContractMatcher(),
            task_bindings={"object": "item_1"},
        )

    repaired = extractor.repair_composite(
        proposal,
        rejected.value,
        chain,
        [],
        contract_matcher=ExactContractMatcher(),
    )
    composite = builder.validate_and_build(
        repaired,
        chain,
        _chain_contract(),
        contract_matcher=ExactContractMatcher(),
        task_bindings={"object": "item_1"},
    )

    assert repaired.control_sequence == ["source", "target"]
    assert len(repaired.new_edges) == 1
    assert len(composite.data_edges) == 1
    assert len(provider.requests) == 2
    repair_context = provider.requests[1].policy_context
    assert "reused required input" in repair_context[
        "deterministic_rejection"
    ]
    assert repair_context["rejected_proposal"]["new_edges"] == []


def test_e2_repair_restores_code_authoritative_control_sequence() -> None:
    chain = _chain()
    extractor, _provider = _repair_session(
        _e2_payload(),
        _e2_payload(),
    )
    proposal = extractor.propose_composite(chain, [])
    proposal.control_sequence[:] = ["forged_by_rejected_proposal"]

    repaired = extractor.repair_composite(
        proposal,
        ValueError("deterministic Composite rejection"),
        chain,
        [],
    )

    assert repaired.control_sequence == ["source", "target"]


def test_e2_repair_rejects_unknown_candidate_and_is_consumed() -> None:
    chain = _chain()
    extractor, provider = _repair_session(
        _e2_payload(),
        _e2_payload("candidate_unknown"),
    )
    proposal = extractor.propose_composite(chain, [])

    with pytest.raises(ExtractionContentError) as rejected:
        extractor.repair_composite(
            proposal,
            ValueError("deterministic Composite rejection"),
            chain,
            [],
        )

    assert rejected.value.stage == "e2_repair"
    assert rejected.value.error_code == (
        "extractor_e2_repair_new_edge_selection_invalid"
    )
    with pytest.raises(RuntimeError, match="may run exactly once"):
        extractor.repair_composite(
            proposal,
            ValueError("second repair forbidden"),
            chain,
            [],
        )
    assert len(provider.requests) == 2


def test_schema_valid_e2_uses_no_semantic_repair_turn() -> None:
    chain = _chain()

    def initial_reply(request):
        candidate = next(
            item for item in request.policy_context["new_edge_candidates"]
            if item["edge_type"] == "data_flow"
        )
        return _e2_payload(candidate["candidate_id"])

    extractor, provider = _repair_session(initial_reply)
    proposal = extractor.propose_composite(chain, [])
    composite = CompositeBuilder().validate_and_build(
        proposal,
        chain,
        _chain_contract(),
        contract_matcher=ExactContractMatcher(),
        task_bindings={"object": "item_1"},
    )

    assert len(composite.data_edges) == 1
    assert len(provider.requests) == 1
    assert extractor._e2_repair_complete is False
