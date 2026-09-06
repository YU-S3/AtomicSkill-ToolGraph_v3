"""R4 content-boundary and full E1 rejection diagnostic regressions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    AtomicProposalBatchRejected,
    Atomicizer,
)
from atomic_skillgraph.tooling.builder_session import (
    ToolBuilderSession,
    ToolProposalParseError,
)


class _Session:
    def __init__(self) -> None:
        self.bucket = ""

    def set_usage_bucket(self, bucket: str) -> None:
        self.bucket = bucket


def _builder(monkeypatch: pytest.MonkeyPatch) -> ToolBuilderSession:
    builder = ToolBuilderSession(_Session())
    monkeypatch.setattr(
        builder.context,
        "tool_builder",
        lambda **_kwargs: "bounded ToolBuilder prompt",
    )
    return builder


@pytest.mark.parametrize(
    ("payload", "cause_type"),
    [
        ({"max_actions": "not-an-integer"}, ValueError),
        ({"max_actions": 1, "final_effects": [{}]}, KeyError),
        ({"max_actions": None}, TypeError),
    ],
)
def test_tool_proposal_parse_error_wraps_only_payload_conversion(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    cause_type: type[Exception],
) -> None:
    builder = _builder(monkeypatch)
    monkeypatch.setattr(
        builder.submissions,
        "request",
        lambda *_args, **_kwargs: SimpleNamespace(value=payload),
    )

    with pytest.raises(ToolProposalParseError) as caught:
        builder.build(atomic=object(), provenance=object())  # type: ignore[arg-type]

    assert isinstance(caught.value.__cause__, cause_type)


def test_tool_proposal_parse_error_does_not_reclassify_request_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder(monkeypatch)
    original = ValueError("submission transport boundary failed")

    def reject_request(*_args: Any, **_kwargs: Any) -> Any:
        raise original

    monkeypatch.setattr(builder.submissions, "request", reject_request)

    with pytest.raises(ValueError) as caught:
        builder.build(atomic=object(), provenance=object())  # type: ignore[arg-type]

    assert caught.value is original
    assert not isinstance(caught.value, ToolProposalParseError)


def _invalid_proposal(
    phase_id: str,
    *,
    input_roles: dict[str, Any],
) -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id=phase_id,
        intent="invalid proposal used to exercise fail-closed validation",
        event_start=0,
        event_end=0,
        input_roles=input_roles,
        output_roles={},
        preconditions=[],
        effects=[],
        rationale="invalid by construction",
    )


def _single_event_trace() -> dict[str, Any]:
    return {
        "actions": [{
            "accepted": True,
            "event_id": "event_0",
            "event_index": 0,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span_0",
        }],
        "runtime_spans": [{
            "span_id": "span_0",
            "occurrence_id": "",
            "action_start": 0,
            "action_end": 1,
            "parent_span_id": None,
        }],
    }


def test_all_e1_rejections_preserve_every_reason() -> None:
    proposals = [
        _invalid_proposal("missing_inputs", input_roles={}),
        _invalid_proposal(
            "ambiguous_inputs",
            input_roles={"object": "apple_1", "source": "apple_1"},
        ),
    ]

    with pytest.raises(AtomicProposalBatchRejected) as caught:
        Atomicizer().validate_proposed_subset(proposals, _single_event_trace())

    assert caught.value.rejections == [
        {
            "phase_id": "missing_inputs",
            "error_type": "ValueError",
            "error": "Atomic occurrence requires explicit input roles",
        },
        {
            "phase_id": "ambiguous_inputs",
            "error_type": "ValueError",
            "error": "Atomic input identity is ambiguous: ambiguous_inputs",
        },
    ]
    assert str(caught.value).endswith(
        "Atomic occurrence requires explicit input roles"
    )


def test_atomic_batch_rejection_deep_copies_diagnostics() -> None:
    source = [{
        "phase_id": "phase",
        "error_type": "ValueError",
        "error": "original",
    }]

    error = AtomicProposalBatchRejected("summary", rejections=source)
    source[0]["error"] = "mutated"
    source.append({
        "phase_id": "other",
        "error_type": "ValueError",
        "error": "later",
    })

    assert error.rejections == [{
        "phase_id": "phase",
        "error_type": "ValueError",
        "error": "original",
    }]


def test_no_e1_proposals_has_an_empty_rejection_list() -> None:
    with pytest.raises(AtomicProposalBatchRejected) as caught:
        Atomicizer().validate_proposed_subset([], {})

    assert caught.value.rejections == []
    assert str(caught.value).endswith("no proposals")
