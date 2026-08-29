from __future__ import annotations

import pytest

from atomic_skillgraph.agents import (
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    ReplayAgentSession,
    UsageLedger,
)
from atomic_skillgraph.core.errors import AgentProtocolError


class _OneCallProvider:
    def __init__(self) -> None:
        self.calls = 0

    def snapshot(self) -> dict:
        return {"provider": "fake", "model": "fake"}

    def complete(self, messages, *, tools=None, structured_output_schema=None):
        self.calls += 1
        return AgentTurn(
            content="",
            tool_calls=[NativeToolCall("call_1", "environment_action", {"action_id": "a1"})],
            finish_reason="tool_calls",
            prompt_tokens=2,
            completion_tokens=1,
            total_tokens=3,
            reasoning_tokens=0,
            latency_ms=0.0,
            provider_metadata=self.snapshot(),
        )


def test_terminal_tool_result_does_not_purchase_or_leave_an_extra_turn() -> None:
    provider = _OneCallProvider()
    session = ReplayAgentSession(
        provider,
        system_prompt="test",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
    )
    tool = NativeToolSpec(
        "environment_action", "act",
        {
            "type": "object",
            "required": ["action_id"],
            "additionalProperties": False,
            "properties": {"action_id": {"type": "string"}},
        },
    )
    turn = session.next_turn("go", tools=[tool])
    session.finalize_tool_result(turn.tool_calls[0].call_id, {"won": True})

    assert provider.calls == 1
    assert session.pending_tool_call is None
    assert session.snapshot()["finalized"] is True
    assert session.snapshot()["messages"][-1]["role"] == "tool"
    with pytest.raises(AgentProtocolError):
        session.next_turn("extra", tools=[tool])
