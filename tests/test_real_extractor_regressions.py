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
            "r000_a001", 0, "GO_TO", {"destination": "desk_1"},
            True, "location evidence", False, False, 1, "span",
        ),
        EnvironmentActionRecord(
            "r001_a001", 1, "TAKE",
            {"object": "alarmclock_1", "source": "desk_1"},
            True, "observation text is not extractor authority", False, False, 2, "span",
        ),
        EnvironmentActionRecord(
            "r002_a001", 2, "USE", {"object": "desklamp_1"},
            True, "terminal prose is not extractor authority", True, True, 3, "span",
        ),
    ]
    trace.runtime_spans = [RuntimeSpan(
        "span", "full_dynamic", "", 0, 3, None, True,
    )]
    return trace, contract


def _e1_occurrences() -> list[dict]:
    return [
        {
            "phase_id": "take_target",
            "intent": "take target object",
            "event_start": 1,
            "event_end": 2,
            "input_roles": {
                "object": "alarmclock_1",
                "source": "desk_1",
            },
            "output_roles": {"held_object": "alarmclock_1"},
            "preconditions": [],
            "effects": [{
                "predicate": "agent.holds",
                "args": {"object": "alarmclock_1"},
            }],
            "rationale": "Accepted TAKE establishes the held object.",
        },
        {
            "phase_id": "observe_under_light",
            "intent": "observe held object under the light",
            "event_start": 2,
            "event_end": 3,
            "input_roles": {
                "target_object": "alarmclock_1",
                "light": "desklamp_1",
            },
            "output_roles": {"observed_object": "alarmclock_1"},
            "preconditions": [{
                "predicate": "agent.holds",
                "args": {"object": "alarmclock_1"},
            }],
            "effects": [{
                "predicate": "object.observed_with",
                "args": {
                    "object": "alarmclock_1",
                    "light": "desklamp_1",
                },
            }],
            "rationale": "Current held/light/location state establishes the relation.",
        },
    ]


def test_real_trace_authority_half_open_e1_and_state_derived_effect() -> None:
    trace, contract = _observed_with_trace()
    normalized = TraceNormalizer().build(trace)

    _go_to, take, use = normalized["actions"]
    assert "observation" not in take
    assert (take["extractor_event_start"], take["extractor_event_end_exclusive"]) == (1, 2)
    assert take["input_role_candidates"] == {
        "object": "alarmclock_1",
        "source": "desk_1",
    }
    assert [item["predicate"] for item in take["authoritative_positive_effects"]] == [
        "agent.holds"
    ]
    assert any(
        item["predicate"] == "agent.holds"
        and item["args"] == {"object": "alarmclock_1"}
        for item in use["authoritative_before_state_facts"]
    )
    assert use["authoritative_terminal_effect_certificates"] == []
    assert any(
        item["predicate"] == "object.observed_with"
        and item["args"] == {
            "object": "alarmclock_1",
            "light": "desklamp_1",
        }
        for item in use["authoritative_positive_effects"]
    )

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
    assert [(item.event_start, item.event_end) for item in proposals] == [(1, 1), (2, 2)]
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
                "target_role": "target_object",
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
    proposal = extractor.propose_composite(canonical, [])
    authority = provider.requests[1].policy_context["canonical_occurrences"]
    assert (
        authority[0]["output_binding_identities"]["held_object"]
        == authority[1]["input_binding_identities"]["target_object"]
    )
    composite = CompositeBuilder().validate_and_build(
        proposal, canonical, contract,
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
        "inventory_search",
        "invent state from prose",
        0,
        0,
        {},
        {},
        [],
        [SemanticPredicate("agent.inventory_empty", {})],
        "invalid extra",
    )
    canonical, rejections = Atomicizer().validate_proposed_subset(
        [invented, *valid], normalized,
    )
    assert len(canonical) == 2
    assert rejections == [{
        "phase_id": "inventory_search",
        "error_type": "ValueError",
        "error": "Atomic occurrence requires explicit input roles",
    }]


def test_extractor_schema_rejects_empty_roles_and_legacy_dependency_wire() -> None:
    occurrence = _e1_occurrences()[0]
    validate_schema_instance(occurrence, ATOMIC_EXTRACTION_SCHEMA)
    with pytest.raises(SchemaValidationError, match="minProperties"):
        validate_schema_instance(
            {**occurrence, "input_roles": {}}, ATOMIC_EXTRACTION_SCHEMA,
        )
    with pytest.raises(SchemaValidationError, match="minProperties"):
        validate_schema_instance(
            {**occurrence, "output_roles": {}}, ATOMIC_EXTRACTION_SCHEMA,
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
