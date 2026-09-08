from __future__ import annotations

import copy
import json
from collections import deque
from typing import Any

import pytest

from atomic_skillgraph.agents import (
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    ReplayAgentSession,
    UsageLedger,
)
from atomic_skillgraph.agents.structured_submission import (
    StructuredSubmissionClient,
)
from atomic_skillgraph.core.errors import AgentProtocolError


SCHEMA = {
    "type": "object",
    "required": ["guideline", "insight"],
    "additionalProperties": False,
    "properties": {
        "guideline": {"type": "object"},
        "insight": {"type": "object"},
    },
}


class _SchemaViolatingProvider:
    """Return raw native calls so the client-side validator owns the test."""

    def __init__(self, *arguments: dict[str, Any]) -> None:
        self._arguments = deque(copy.deepcopy(arguments))
        self.requests: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        self.requests.append(copy.deepcopy(messages))
        if not self._arguments:
            raise AssertionError("no scripted R6.1 reply remains")
        offered = list(tools or ())
        if len(offered) != 1:
            raise AssertionError(f"expected one submit tool, got {len(offered)}")
        call_index = len(self.requests) - 1
        return AgentTurn(
            content="",
            tool_calls=[NativeToolCall(
                f"call_r61_protocol_{call_index:06d}",
                offered[0].name,
                self._arguments.popleft(),
            )],
            finish_reason="tool_calls",
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
            reasoning_tokens=1,
            latency_ms=1.0,
            provider_metadata={"provider": "r61_test"},
            reasoning_content="deterministic reasoning",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": "r61_test",
            "call_count": len(self.requests),
            "remaining_replies": len(self._arguments),
        }


def _session(
    *arguments: dict[str, Any],
) -> tuple[ReplayAgentSession, _SchemaViolatingProvider]:
    provider = _SchemaViolatingProvider(*arguments)
    session = ReplayAgentSession(
        provider,
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e2",
    )
    return session, provider


def test_protocol_repair_includes_exact_schema_failure_detail() -> None:
    session, provider = _session(
        {"guideline": {}, "insight": "wrong string"},
        {"guideline": {}, "insight": {}},
    )

    result = StructuredSubmissionClient().request(
        session,
        prompt="submit E2",
        tool_name="submit_extractor_composite",
        description="submit E2",
        schema=SCHEMA,
    )

    assert result.value == {"guideline": {}, "insight": {}}
    assert len(provider.requests) == 2
    repair_prompt = provider.requests[1][-1]["content"]
    assert "$.insight" in repair_prompt
    assert "expected object" in repair_prompt
    assert "got string" in repair_prompt
    assert "Correct only the rejected protocol/schema issue" in repair_prompt
    snapshot = session.snapshot()
    assert snapshot["protocol_repairs_used"] == 1
    rejected = [
        json.loads(message["content"])
        for message in snapshot["messages"]
        if message.get("role") == "tool"
        and json.loads(message["content"]).get("accepted") is False
    ]
    assert rejected == [{
        "accepted": False,
        "executed": False,
        "error": "runtime_agent_schema_error",
    }]


def test_protocol_repair_limit_remains_one_after_two_invalid_turns() -> None:
    session, provider = _session(
        {"guideline": {}, "insight": "wrong"},
        {"guideline": "wrong", "insight": {}},
    )

    with pytest.raises(AgentProtocolError, match=r"\$.guideline"):
        StructuredSubmissionClient().request(
            session,
            prompt="submit E2",
            tool_name="submit_extractor_composite",
            description="submit E2",
            schema=SCHEMA,
        )

    assert len(provider.requests) == 2
    snapshot = session.snapshot()
    assert snapshot["protocol_repairs_used"] == 1
    assert snapshot["terminal_protocol_failure"]["code"] == (
        "runtime_agent_schema_error"
    )
