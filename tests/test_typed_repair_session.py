from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents import NativeToolCall
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.evolution.typed_repair_session import (
    TypedRepairProposalSession,
    TypedRepairReview,
)
from atomic_skillgraph.evolution.typed_repairs import RepairEvidence


def _atomic() -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef("atomic_prepare", "1.0.0"), "prepare",
        [ParameterSpec("item", "entity")], [ParameterSpec("result", "entity")],
        [], [SemanticPredicate("object.ready", {"object": "item"})],
        {"validator_id": "v"}, [], {}, {}, SkillStatus.CANDIDATE,
    )


def _review() -> TypedRepairReview:
    evidence = tuple(
        RepairEvidence(f"e{i}", f"task{i}", f"trace{i}", "stable", {"case": i})
        for i in (1, 2)
    )
    return TypedRepairReview(
        "review_atomic", "atomic", ("skill://atomic_prepare@1.0.0",),
        ("revise_atomic_contract",), {"source": to_primitive(_atomic())}, evidence,
        ("failure_1",),
    )


def test_typed_session_parses_and_builds_with_code_owned_evidence() -> None:
    review = _review()
    replacement = replace(_atomic(), guideline={"stable": "advice"})
    payload = {
        "decisions": [{
            "review_id": review.review_id,
            "decision": "propose",
            "operation": "revise_atomic_contract",
            "replacements": [to_primitive(replacement)],
            "rationale": "two independent cases support the guideline",
        }],
    }
    decisions = TypedRepairProposalSession.parse(payload, [review])
    proposals = TypedRepairProposalSession.build_proposals(decisions, [review])
    assert len(proposals) == 1
    assert proposals[0].source_failure_ids == ["failure_1"]
    assert {
        item["evidence_id"] for item in proposals[0].proposed_patch["source_cases"]
    } == {"e1", "e2"}
    assert "source_cases" not in payload["decisions"][0]


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"decisions": [{
            "review_id": "unknown", "decision": "no_change", "operation": "no_change",
            "replacements": [], "rationale": "none",
        }]}, "unknown"),
        ({"decisions": [{
            "review_id": "review_atomic", "decision": "propose", "operation": "merge_atomic",
            "replacements": [to_primitive(_atomic())], "rationale": "invalid",
        }]}, "not authorized"),
        ({"decisions": [
            {
                "review_id": "review_atomic", "decision": "no_change", "operation": "no_change",
                "replacements": [], "rationale": "none",
            },
            {
                "review_id": "review_atomic", "decision": "no_change", "operation": "no_change",
                "replacements": [], "rationale": "none",
            },
        ]}, "duplicate"),
    ],
)
def test_typed_session_rejects_unknown_operation_and_duplicate(payload, match) -> None:
    with pytest.raises(ValueError, match=match):
        TypedRepairProposalSession.parse(payload, [_review()])


def test_typed_session_is_one_bounded_turn_and_schema_has_no_evidence_output() -> None:
    review = _review()

    class FakeSession:
        def __init__(self):
            self.schema = None

        def next_turn(self, _prompt, *, tools):
            self.schema = tools[0].input_schema
            return SimpleNamespace(tool_calls=[
                NativeToolCall("call_typed", tools[0].name, {"decisions": []})
            ])

        def acknowledge_tool_result(self, call_id, result):
            assert call_id == "call_typed"
            assert result["accepted"] is True

    fake = FakeSession()
    session = TypedRepairProposalSession(fake)
    assert session.propose([review]) == []
    decision_properties = fake.schema["properties"]["decisions"]["items"]["properties"]
    assert "evidence" not in decision_properties
    assert "cluster_key" not in decision_properties
    with pytest.raises(RuntimeError, match="exactly once"):
        session.propose([review])
