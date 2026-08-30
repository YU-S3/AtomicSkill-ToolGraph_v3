from __future__ import annotations

from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents import NativeToolCall
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.composite_repair_session import (
    CompositeSequenceProposalSession,
    CompositeSequenceReview,
)
from atomic_skillgraph.evolution.typed_repairs import RepairEvidence

from test_composite_repairs import _composite, _replacement


def _review() -> CompositeSequenceReview:
    source = _composite()
    return CompositeSequenceReview(
        "review_sequence",
        str(source.ref),
        to_primitive(source),
        {
            "authoritative_occurrence_order": source.control_sequence,
            "transition_trace_ids": ["trace1", "trace2"],
        },
        tuple(
            RepairEvidence(
                f"e{i}", f"task{i}", f"trace{i}", "stable_sequence",
                {"trace_id": f"trace{i}"},
            )
            for i in (1, 2)
        ),
        ("failure_sequence",),
    )


def test_composite_session_builds_proposal_with_code_owned_authority() -> None:
    review = _review()
    replacement = to_primitive(_replacement(_composite()))
    payload = {"decisions": [{
        "review_id": review.review_id,
        "decision": "propose",
        "replacement": replacement,
        "rationale": "stable structural evidence supports reordering",
    }]}
    decisions = CompositeSequenceProposalSession.parse(payload, [review])
    proposals = CompositeSequenceProposalSession.build_proposals(decisions, [review])
    assert len(proposals) == 1
    assert proposals[0].target_ref == review.target_ref
    assert proposals[0].source_failure_ids == ["failure_sequence"]
    assert {
        item["evidence_id"]
        for item in proposals[0].proposed_patch["source_cases"]
    } == {"e1", "e2"}
    assert set(payload["decisions"][0]) == {
        "review_id", "decision", "replacement", "rationale",
    }


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"decisions": [{
            "review_id": "unknown", "decision": "no_change",
            "replacement": {}, "rationale": "none",
        }]}, "unknown"),
        ({"decisions": [
            {
                "review_id": "review_sequence", "decision": "no_change",
                "replacement": {}, "rationale": "none",
            },
            {
                "review_id": "review_sequence", "decision": "no_change",
                "replacement": {}, "rationale": "none",
            },
        ]}, "duplicate"),
        ({"decisions": [{
            "review_id": "review_sequence", "decision": "no_change",
            "replacement": {"forged": True}, "rationale": "invalid",
        }]}, "cannot carry"),
    ],
)
def test_composite_session_rejects_unknown_duplicate_and_no_change_payload(payload, match) -> None:
    with pytest.raises(ValueError, match=match):
        CompositeSequenceProposalSession.parse(payload, [_review()])


def test_composite_session_is_single_turn_and_output_has_no_authority_fields() -> None:
    class FakeSession:
        def __init__(self):
            self.schema = None

        def next_turn(self, _prompt, *, tools):
            self.schema = tools[0].input_schema
            return SimpleNamespace(tool_calls=[
                NativeToolCall("call_composite", tools[0].name, {"decisions": []})
            ])

        def acknowledge_tool_result(self, call_id, result):
            assert call_id == "call_composite"
            assert result["accepted"] is True

    fake = FakeSession()
    session = CompositeSequenceProposalSession(fake)
    assert session.propose([_review()]) == []
    properties = fake.schema["properties"]["decisions"]["items"]["properties"]
    assert set(properties) == {"review_id", "decision", "replacement", "rationale"}
    with pytest.raises(RuntimeError, match="exactly once"):
        session.propose([_review()])
