from __future__ import annotations

import pytest

from atomic_skillgraph.agents import (
    ReplayAgentSession,
    SchemaValidationError,
    UsageLedger,
    validate_schema_instance,
)
from atomic_skillgraph.agents.structured_submission import (
    ATOMIC_EXTRACTION_SCHEMA,
    PROPOSED_EDGE_SCHEMA,
)
from atomic_skillgraph.core.contracts import (
    ContractSource,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import ExtractorSession
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.harness.alfworld import AlfWorldContractMatcher
from atomic_skillgraph.traces.schema import (
    EnvironmentActionRecord,
    RuntimeSpan,
    TaskRecord,
    TraceRecord,
)
from experiments.fakes import FakeReply, ScriptedAgentProvider


def _observed_with_trace() -> tuple[TraceRecord, TaskContract]:
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "object.observed_with",
            {"object": "alarmclock", "light": "desklamp"},
        )],
        source=ContractSource.ADAPTER_DERIVED,
        confidence=1.0,
        validator_id="alfworld_v3_goal",
    )
    trace = TraceRecord.create(
        TaskRecord(
            "look-task",
            "alfworld",
            "look at an alarmclock under a desklamp",
            "look_at_obj_in_light",
            "look-signature",
        ),
        to_primitive(contract),
        {},
        {"source": "full_dynamic"},
    )
    trace.benchmark_success = True
    trace.environment_actions = [
        EnvironmentActionRecord(
            "r000_a001", 0, "TAKE",
            {"object": "alarmclock_1", "source": "desk_1"},
            True, "observation text is not extractor authority", False, False, 1, "span",
            {
                "action_id": "r000_a001",
                "revision_before": 0,
                "revision_after": 1,
                "action_type": "TAKE",
                "arguments": {"object": "alarmclock_1", "source": "desk_1"},
                "before_facts": [],
                "positive_effects": [{
                    "fact_ref": "effect:take:holds",
                    "predicate": "agent.holds",
                    "args": {"object": "alarmclock_1"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "negative_effects": [],
                "required_facts": [],
                "terminal_effects": [],
                "accepted": True,
                "state_changed": True,
                "evidence_refs": ["cert:take"],
            },
        ),
        EnvironmentActionRecord(
            "r001_a001", 1, "USE", {"object": "desklamp_1"},
            True, "terminal prose is not extractor authority", True, True, 2, "span",
            {
                "action_id": "r001_a001",
                "revision_before": 1,
                "revision_after": 2,
                "action_type": "USE",
                "arguments": {"object": "desklamp_1"},
                "before_facts": [{
                    "fact_ref": "before:use:holds",
                    "predicate": "agent.holds",
                    "args": {"object": "alarmclock_1"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "positive_effects": [{
                    "fact_ref": "effect:use:observed_with",
                    "predicate": "object.observed_with",
                    "args": {"object": "alarmclock_1", "light": "desklamp_1"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "negative_effects": [],
                "required_facts": [{
                    "fact_ref": "required:use:holds",
                    "predicate": "agent.holds",
                    "args": {"object": "alarmclock_1"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "terminal_effects": [{
                    "fact_ref": "effect:use:observed_with",
                    "predicate": "object.observed_with",
                    "args": {"object": "alarmclock_1", "light": "desklamp_1"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "accepted": True,
                "state_changed": True,
                "evidence_refs": ["cert:use"],
            },
        ),
    ]
    trace.runtime_spans = [RuntimeSpan(
        "span", "full_dynamic", "", 0, 2, None, True,
    )]
    return trace, contract


def _e1_occurrences() -> list[dict]:
    return [
        {
            "phase_id": "take_target",
            "intent": "take target object",
            "event_start": 0,
            "event_end_exclusive": 1,
            "selected_effect_refs": ["effect:take:holds"],
            "selected_precondition_refs": [],
            "output_role_mapping": {
                "held_object": "fact:effect:take:holds:object",
            },
            "rationale": "Accepted TAKE establishes the held object.",
        },
        {
            "phase_id": "observe_under_light",
            "intent": "observe held object under the light",
            "event_start": 1,
            "event_end_exclusive": 2,
            "selected_effect_refs": ["effect:use:observed_with"],
            "selected_precondition_refs": ["required:use:holds"],
            "output_role_mapping": {
                "observed_object": "fact:effect:use:observed_with:object",
            },
            "rationale": "Terminal USE has the narrow TaskContract certificate.",
        },
    ]


def test_real_trace_authority_half_open_e1_and_terminal_certificate() -> None:
    trace, contract = _observed_with_trace()
    normalized = TraceNormalizer().build(trace)

    take, use = normalized["actions"]
    assert "observation" not in take
    assert (take["extractor_event_start"], take["extractor_event_end_exclusive"]) == (0, 1)
    assert [
        item["predicate"]
        for item in take["transition_certificate"]["positive_effects"]
    ] == [
        "agent.holds"
    ]
    assert any(
        item["predicate"] == "agent.holds"
        and item["args"] == {"object": "alarmclock_1"}
        for item in use["transition_certificate"]["before_facts"]
    )
    assert use["transition_certificate"]["terminal_effects"] == [{
        "fact_ref": "effect:use:observed_with",
        "predicate": "object.observed_with",
        "args": {"object": "alarmclock_1", "light": "desklamp_1"},
        "cardinality": 1,
        "distinct_by": "",
    }]

    provider = ScriptedAgentProvider([
        FakeReply.structured({"occurrences": _e1_occurrences()}),
    ])
    extractor = ExtractorSession(ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e1",
    ))
    proposals = extractor.propose_atomics(normalized)
    assert [
        (item.event_start, item.event_end_exclusive) for item in proposals
    ] == [(0, 1), (1, 2)]
    canonical = Atomicizer().validate_and_canonicalize(proposals, normalized)
    assert canonical[-1].effects[0].predicate == "object.observed_with"
    assert to_primitive(contract.target_effects) == normalized["task_contract"]["target_effects"]


def test_e2_authority_exposes_binding_identity_and_accepts_dataflow() -> None:
    trace, contract = _observed_with_trace()
    normalized = TraceNormalizer().build(trace)

    def e2_reply(request):
        authority = request.policy_context["canonical_occurrences"]
        return {
            "control_sequence": [item["occurrence_id"] for item in authority],
            "existing_edges": [],
            "new_edges": [{
                "edge_id": "held_to_observed_target",
                "edge_type": "data_flow",
                "source_step": authority[0]["occurrence_id"],
                "target_step": authority[1]["occurrence_id"],
                "source_role": "held_object",
                "target_role": "object",
            }],
            "summary": "take then observe target under light",
            "guideline": {},
            "insight": {},
        }

    provider = ScriptedAgentProvider([
        FakeReply.structured({"occurrences": _e1_occurrences()}),
        FakeReply.structured(e2_reply),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e1",
    )
    extractor = ExtractorSession(session)
    canonical = Atomicizer().validate_and_canonicalize(
        extractor.propose_atomics(normalized), normalized,
    )
    proposal = extractor.propose_composite(canonical, [], contract)
    authority = provider.requests[1].policy_context["canonical_occurrences"]
    assert provider.requests[1].policy_context["task_contract"] == to_primitive(
        contract
    )
    assert (
        authority[0]["output_binding_identities"]["held_object"]
        == authority[1]["input_binding_identities"]["object"]
    )
    composite = CompositeBuilder().validate_and_build(
        proposal,
        canonical,
        contract,
        contract_matcher=AlfWorldContractMatcher(),
    )
    assert len(composite.data_edges) == 1


def test_invalid_e1_extra_is_rejected_without_discarding_valid_causal_occurrences() -> None:
    trace, _contract = _observed_with_trace()
    normalized = TraceNormalizer().build(trace)
    provider = ScriptedAgentProvider([
        FakeReply.structured({"occurrences": _e1_occurrences()}),
    ])
    extractor = ExtractorSession(ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e1",
    ))
    valid = extractor.propose_atomics(normalized)
    invented = AtomicOccurrenceProposal(
        phase_id="invented",
        intent="invent state from prose",
        event_start=0,
        event_end_exclusive=1,
        selected_effect_refs=["not-a-certificate-ref"],
        selected_precondition_refs=[],
        output_role_mapping={"result": "argument:object"},
        rationale="invalid extra",
    )
    canonical, rejections = Atomicizer().validate_proposed_subset(
        [invented, *valid], normalized,
    )
    assert len(canonical) == 2
    assert rejections == [{
        "phase_id": "invented",
        "error_type": "ValueError",
        "error": "unknown/out-of-boundary effect references: ['not-a-certificate-ref']",
    }]


def test_extractor_schema_rejects_empty_refs_and_legacy_dependency_wire() -> None:
    occurrence = _e1_occurrences()[0]
    validate_schema_instance(occurrence, ATOMIC_EXTRACTION_SCHEMA)
    with pytest.raises(SchemaValidationError, match="minItems"):
        validate_schema_instance(
            {**occurrence, "selected_effect_refs": []}, ATOMIC_EXTRACTION_SCHEMA,
        )
    with pytest.raises(SchemaValidationError, match="minProperties"):
        validate_schema_instance(
            {**occurrence, "output_role_mapping": {}}, ATOMIC_EXTRACTION_SCHEMA,
        )

    edge = {
        "edge_id": "edge",
        "edge_type": "requires_skill",
        "source_step": "a",
        "target_step": "b",
        "source_role": "",
        "target_role": "",
    }
    validate_schema_instance(edge, PROPOSED_EDGE_SCHEMA)
    with pytest.raises(SchemaValidationError, match="enum"):
        validate_schema_instance(
            {**edge, "edge_type": "dependency"}, PROPOSED_EDGE_SCHEMA,
        )
