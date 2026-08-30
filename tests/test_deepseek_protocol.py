from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atomic_skillgraph.agents import (
    ContextBuilder,
    NativeToolSpec,
    ReplayAgentSession,
    StructuredSubmissionClient,
    UsageLedger,
)
from atomic_skillgraph.agents.provider import (
    AgentProviderError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)
from atomic_skillgraph.agents.provider_probe import (
    ProviderCapabilityError,
    ensure_provider_capability,
    run_provider_capability_probe,
)


class _Response:
    def __init__(
        self,
        payload: dict[str, Any] | None,
        *,
        status: int = 200,
        text: str = "",
        request_id: str = "req_fixture",
    ) -> None:
        self._payload = payload
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = text
        self.headers = {"x-request-id": request_id}

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("invalid json")
        return self._payload


def _config(**changes: Any) -> OpenAICompatibleConfig:
    values = {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "TEST_DEEPSEEK_KEY",
        "max_completion_tokens": 32768,
        "max_retries": 0,
    }
    values.update(changes)
    return OpenAICompatibleConfig(**values)


def _capability_config() -> dict[str, Any]:
    return {
        "llm": {
            "provider": "openai_compatible",
            "dialect": "deepseek_v4_chat",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "TEST_DEEPSEEK_KEY",
            "protocol": {
                "endpoint_path": "/chat/completions",
                "thinking_type": "enabled",
                "token_limit_field": "max_tokens",
            },
            "planner": {
                "max_completion_tokens": 32768,
                "reasoning_effort": "high",
            },
            "extractor": {
                "max_completion_tokens": 131072,
                "reasoning_effort": "high",
            },
        },
    }


def _tool(name: str = "submit_probe") -> NativeToolSpec:
    return NativeToolSpec(
        name,
        "Submit one probe value.",
        {
            "type": "object",
            "required": ["ok"],
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
        },
    )


def _success(
    call_id: str,
    tool_name: str,
    *,
    reasoning: str = "provider-private-reasoning",
) -> dict[str, Any]:
    return {
        "id": f"response_{call_id}",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "reasoning_content": reasoning,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{\"ok\":true}"},
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "total_tokens": 17,
            "completion_tokens_details": {"reasoning_tokens": 5},
            "prompt_cache_hit_tokens": 2,
        },
    }


def test_deepseek_payload_uses_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    posted: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        posted.append(json)
        assert headers["Authorization"].endswith("secret-fixture-key")
        assert timeout == (15.0, 120.0)
        return _Response(_success("call_1", "submit_probe"))

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    provider = OpenAICompatibleProvider(_config())
    turn = provider.complete(
        [{"role": "system", "content": "probe"}], tools=[_tool()],
    )

    payload = posted[0]
    assert payload["max_tokens"] == 32768
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["tools"][0]["function"]["parameters"] == _tool().input_schema
    forbidden = {
        "max_completion_tokens",
        "response_format",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
    }
    assert forbidden.isdisjoint(payload)
    assert turn.reasoning_content == "provider-private-reasoning"
    assert turn.reasoning_tokens == 5
    record = provider.request_records[0]
    assert record["outcome"] == "success"
    assert record["usage_status"] == "reported"
    assert record["reasoning_content_chars"] == len(turn.reasoning_content)
    assert "provider-private-reasoning" not in json.dumps(record)
    assert "secret-fixture-key" not in json.dumps(record)


def test_reasoning_content_is_replayed_across_tool_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    responses = iter([
        _Response(_success("call_1", "probe_step_one", reasoning="exact replay value")),
        _Response(_success("call_2", "probe_step_two", reasoning="second private value")),
    ])
    posted: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        posted.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    session = ReplayAgentSession(
        OpenAICompatibleProvider(_config()),
        system_prompt="probe",
        usage_ledger=UsageLedger(),
        usage_bucket="planner_p1",
        session_id="reasoning_replay",
    )
    first = session.next_turn("step one", tools=[_tool("probe_step_one")])
    session.acknowledge_tool_result(first.tool_calls[0].call_id, {"accepted": True})
    second = session.next_turn("step two", tools=[_tool("probe_step_two")])
    session.acknowledge_tool_result(second.tool_calls[0].call_id, {"accepted": True})

    replay = posted[1]["messages"]
    assistant = next(message for message in replay if message["role"] == "assistant")
    tool_result = next(message for message in replay if message["role"] == "tool")
    assert assistant["reasoning_content"] == "exact replay value"
    assert set(tool_result) == {"role", "tool_call_id", "content"}
    snapshot_json = json.dumps(session.snapshot(), ensure_ascii=False)
    assert "exact replay value" not in snapshot_json
    assert "second private value" not in snapshot_json
    assert '"reasoning_content_present": true' in snapshot_json


def test_structured_submission_uses_native_tool_and_one_format_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    no_call = {
        "id": "response_no_call",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "I will answer in prose",
                "reasoning_content": "format repair reasoning",
            },
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    responses = iter([
        _Response(no_call, request_id="req_no_call"),
        _Response(_success("call_fixed", "submit_probe"), request_id="req_fixed"),
    ])
    posted: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        posted.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    session = ReplayAgentSession(
        OpenAICompatibleProvider(_config()),
        system_prompt="probe",
        usage_ledger=UsageLedger(),
        usage_bucket="planner_p1",
    )
    submission = StructuredSubmissionClient().request(
        session,
        prompt="submit ok",
        tool_name="submit_probe",
        description="Submit probe.",
        schema=_tool().input_schema,
    )
    assert submission.value == {"ok": True}
    assert len(posted) == 2
    assert posted[0]["tools"][0]["function"]["name"] == "submit_probe"
    assert "PROTOCOL REPAIR REQUIRED" in posted[1]["messages"][-1]["content"]
    assert session.pending_tool_call is None


def test_provider_missing_reasoning_and_usage_are_classified_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    missing_reasoning = _success("call_1", "submit_probe")
    del missing_reasoning["choices"][0]["message"]["reasoning_content"]
    responses = iter([
        _Response(missing_reasoning, request_id="req_reasoning"),
        _Response({
            "id": "no_usage",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "reasoning_content": "private",
                    "tool_calls": [],
                },
            }],
        }, request_id="req_usage"),
    ])
    monkeypatch.setattr(
        "atomic_skillgraph.agents.provider.requests.post",
        lambda *_args, **_kwargs: next(responses),
    )
    provider = OpenAICompatibleProvider(_config())
    with pytest.raises(AgentProviderError) as reasoning_error:
        provider.complete([{"role": "system", "content": "probe"}], tools=[_tool()])
    assert reasoning_error.value.code == "provider_reasoning_content_missing"
    with pytest.raises(AgentProviderError) as usage_error:
        provider.complete([{"role": "system", "content": "probe"}], tools=[_tool()])
    assert usage_error.value.code == "provider_usage_missing"
    assert provider.request_records[0]["usage_status"] == "reported"
    assert provider.request_records[1]["usage_status"] == "unavailable"


def test_provider_error_body_cannot_leak_replayed_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    leaked = "private-reasoning-that-a-proxy-must-not-echo-to-trace"
    monkeypatch.setattr(
        "atomic_skillgraph.agents.provider.requests.post",
        lambda *_args, **_kwargs: _Response(
            None,
            status=400,
            text=f'proxy echoed {{"reasoning_content":"{leaked}"}}',
            request_id="req_echo",
        ),
    )
    provider = OpenAICompatibleProvider(_config())
    messages = [
        {"role": "system", "content": "probe"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": leaked,
            "tool_calls": [{
                "id": "prior_call",
                "type": "function",
                "function": {
                    "name": "submit_probe",
                    "arguments": "{\"ok\":true}",
                },
            }],
        },
        {"role": "tool", "tool_call_id": "prior_call", "content": "{\"accepted\":true}"},
        {"role": "user", "content": "continue"},
    ]
    with pytest.raises(AgentProviderError) as error:
        provider.complete(messages, tools=[_tool()])
    audit = json.dumps(provider.request_records, ensure_ascii=False)
    assert leaked not in str(error.value)
    assert leaked not in audit
    assert "proxy echoed" not in audit
    assert provider.request_records[0]["sanitized_error"] == (
        "HTTP 400 from LLM provider"
    )


def test_context_builder_separates_grounding_authorities() -> None:
    payload_text = ContextBuilder().runtime_node(
        task_goal="put apple in fridge",
        atomic_contract={"summary": "place", "inputs": [], "outputs": []},
        semantic_anchors={"object": "apple", "destination": "fridge"},
        execution_ready_bindings={"held_object": "apple_2"},
        missing_or_insufficient_bindings=["destination"],
        observation="holding apple_2",
        action_catalog=[],
        relevant_action_history=[],
        remaining_budget={"actions": 10},
        implementation_invocations=[],
    )
    payload = json.loads(payload_text.split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1])
    assert payload["semantic_anchors"] == {"object": "apple", "destination": "fridge"}
    assert payload["execution_ready_bindings"] == {"held_object": "apple_2"}
    assert payload["missing_or_insufficient_bindings"] == ["destination"]
    assert "certified_bindings" not in payload


def test_deepseek_payload_has_thinking_and_reasoning_effort() -> None:
    payload = OpenAICompatibleProvider(_config())._build_payload(
        [{"role": "system", "content": "probe"}], [_tool()],
    )
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"


def test_deepseek_payload_omits_response_format_tool_choice_parallel_temperature() -> None:
    payload = OpenAICompatibleProvider(_config())._build_payload(
        [{"role": "system", "content": "probe"}], [_tool()],
    )
    assert {
        "response_format", "tool_choice", "parallel_tool_calls", "temperature",
        "max_completion_tokens",
    }.isdisjoint(payload)


def test_reasoning_content_is_not_exposed_to_runtime_or_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    monkeypatch.setattr(
        "atomic_skillgraph.agents.provider.requests.post",
        lambda *_args, **_kwargs: _Response(
            _success("call_private", "submit_probe", reasoning="never persist this text")
        ),
    )
    ledger = UsageLedger()
    session = ReplayAgentSession(
        OpenAICompatibleProvider(_config()),
        system_prompt="probe",
        usage_ledger=ledger,
        usage_bucket="planner_p1",
    )
    StructuredSubmissionClient().request(
        session,
        prompt="submit",
        tool_name="submit_probe",
        description="submit",
        schema=_tool().input_schema,
    )
    formal_audit = json.dumps(
        {"session": session.snapshot(), "usage": ledger.snapshot()},
        ensure_ascii=False,
    )
    assert "never persist this text" not in formal_audit
    assert "reasoning_content_sha256" in formal_audit


def test_structured_submission_uses_native_tool() -> None:
    from experiments.fakes import FakeReply, ScriptedAgentProvider

    provider = ScriptedAgentProvider([FakeReply.structured({"ok": True})])
    session = ReplayAgentSession(
        provider,
        system_prompt="probe",
        usage_ledger=UsageLedger(),
        usage_bucket="planner_p1",
    )
    result = StructuredSubmissionClient().request(
        session,
        prompt="submit",
        tool_name="submit_probe",
        description="submit",
        schema=_tool().input_schema,
    )
    assert result.value == {"ok": True}
    assert provider.requests[0].tools[0].name == "submit_probe"
    assert session.pending_tool_call is None


def test_same_session_supports_p1_p1r_p2_after_acknowledge() -> None:
    from experiments.fakes import FakeReply, ScriptedAgentProvider

    provider = ScriptedAgentProvider([
        FakeReply.structured({"ok": True}),
        FakeReply.structured({"ok": True}),
        FakeReply.structured({"ok": True}),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="planner",
        usage_ledger=UsageLedger(),
        usage_bucket="planner_p1",
        session_id="planner_same_session",
    )
    client = StructuredSubmissionClient()
    for index, stage in enumerate(("planner_p1", "planner_p1_repair", "planner_p2"), start=1):
        session.set_usage_bucket(stage)
        result = client.request(
            session,
            prompt=f"stage {index}",
            tool_name=f"submit_stage_{index}",
            description="submit planner stage",
            schema=_tool().input_schema,
        )
        assert result.value == {"ok": True}
    assert len(provider.requests) == 3
    assert [message["role"] for message in provider.requests[-1].messages] == [
        "system", "user", "assistant", "tool", "user", "assistant", "tool", "user",
    ]


def test_provider_capability_probe_is_exact_and_stored_gate_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    responses = iter([
        _Response(_success("probe_a", "submit_probe"), request_id="req_a"),
        _Response(_success("probe_b1", "probe_step_one"), request_id="req_b1"),
        _Response(_success("probe_b2", "probe_step_two"), request_id="req_b2"),
        _Response(
            _success("probe_c", "submit_extractor_probe"), request_id="req_c",
        ),
    ])
    payloads = []

    def post(_url, *, headers, json, timeout):
        payloads.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    config = _capability_config()
    manifest = run_provider_capability_probe(
        config,
        output_dir=tmp_path,
        config_hash="config_fixture",
        code_hash="code_fixture",
    )
    assert manifest["passed"] is True
    assert manifest["http_statuses"] == [200, 200, 200, 200]
    assert manifest["http_outcomes"] == ["success"] * 4
    assert len(payloads) == 4
    assert payloads[3]["max_tokens"] == 131072
    assert ensure_provider_capability(
        config,
        output_dir=tmp_path,
        config_hash="config_fixture",
        code_hash="code_fixture",
        run_if_missing=False,
    )["passed"] is True

    manifest_path = tmp_path / "provider_capability_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["probe_pass"]["structured_submission"] = False
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ProviderCapabilityError, match="does not match"):
        ensure_provider_capability(
            config,
            output_dir=tmp_path,
            config_hash="config_fixture",
            code_hash="code_fixture",
            run_if_missing=False,
        )


def test_provider_capability_probe_audits_repaired_b_turn_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    no_tool_call = {
        "id": "response_b2_rejected",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "prose is not a native tool submission",
                "reasoning_content": "metered rejected reasoning",
            },
        }],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 11,
            "total_tokens": 24,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }
    responses = iter([
        _Response(_success("probe_a", "submit_probe"), request_id="req_a"),
        _Response(_success("probe_b1", "probe_step_one"), request_id="req_b1"),
        _Response(no_tool_call, request_id="req_b2_rejected"),
        _Response(_success("probe_b2", "probe_step_two"), request_id="req_b2"),
        _Response(
            _success("probe_c", "submit_extractor_probe"), request_id="req_c",
        ),
    ])
    payloads: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        payloads.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    config = _capability_config()
    manifest = run_provider_capability_probe(
        config,
        output_dir=tmp_path,
        config_hash="config_fixture",
        code_hash="code_fixture",
    )

    assert manifest["passed"] is True
    assert manifest["provider_request_count"] == 5
    assert manifest["http_statuses"] == [200] * 5
    assert manifest["http_outcomes"] == ["success"] * 5
    assert len(payloads) == 5
    assert "PROTOCOL REPAIR REQUIRED" in payloads[3]["messages"][-1]["content"]
    assert payloads[4]["max_tokens"] == 131072

    trace = json.loads(
        (tmp_path / "provider_probe_trace.json").read_text(encoding="utf-8")
    )
    replay = trace["probes"]["reasoning_replay"]
    assert trace["probe_request_shape_valid"] is True
    assert replay["accepted_turn_count"] == 2
    assert replay["request_count"] == 3
    assert replay["turn_count"] == 3
    assert replay["metered_turn_count"] == 3
    assert replay["protocol_repair_count"] == 1
    assert replay["protocol_failure_codes"] == ["agent_protocol_no_action"]
    assert replay["reasoning_content_replayed_verbatim"] is True
    assert ensure_provider_capability(
        config,
        output_dir=tmp_path,
        config_hash="config_fixture",
        code_hash="code_fixture",
        run_if_missing=False,
    )["passed"] is True
