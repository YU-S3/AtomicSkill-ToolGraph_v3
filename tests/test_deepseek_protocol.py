from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atomic_skillgraph.agents import (
    AgentBudget,
    ContextBuilder,
    NativeToolSpec,
    ReplayAgentSession,
    StructuredSubmissionClient,
    UsageLedger,
    structured_provider_turn_cap,
)
from atomic_skillgraph.core.errors import (
    AgentProtocolError,
    BudgetExhausted,
    FailureEnvelope,
    FailureLayer,
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
from atomic_skillgraph.core.contracts import (
    CapabilityRequirement,
    ColdStartCandidateSource,
    ColdStartExecutionMode,
    ColdStartPlanProposal,
    ColdStartPlanStep,
    ParameterSpec,
    PlannerRequirementBundle,
)
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.traces.schema import ColdStartPlanRecord
from experiments.fakes import FakeHarness, fake_task
from experiments.report import trace_to_row


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


def test_http200_usage_persists_when_extractor_budget_rejects_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    payload = _success("call_over_cap", "submit_probe")
    payload["usage"] = {
        "prompt_tokens": 200000,
        "completion_tokens": 70000,
        "total_tokens": 270000,
        "completion_tokens_details": {"reasoning_tokens": 60000},
    }

    def post(_url, *, headers, json, timeout):
        return _Response(payload, request_id="req_over_cap")

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    provider = OpenAICompatibleProvider(_config(max_completion_tokens=131072))
    ledger = UsageLedger()
    session = ReplayAgentSession(
        provider,
        system_prompt="failure extractor",
        usage_ledger=ledger,
        usage_bucket="failure_extractor_f1",
        budget=AgentBudget(
            structured_provider_turn_cap(1),
            262144,
            "extractor_token_budget_exhausted",
        ),
        semantic_max_turns=1,
        session_id="failure-f1-over-cap",
    )

    with pytest.raises(BudgetExhausted) as exhausted:
        StructuredSubmissionClient().request(
            session,
            prompt="bounded failure alignment input",
            tool_name="submit_probe",
            description="submit",
            schema=_tool().input_schema,
        )

    assert exhausted.value.code == "extractor_token_budget_exhausted"
    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event.session_id == "failure-f1-over-cap"
    assert event.usage.prompt_tokens == 200000
    assert event.usage.completion_tokens == 70000
    assert event.usage.reasoning_tokens == 60000
    assert event.usage.total_tokens == 270000
    assert event.usage.latency_ms >= 0
    assert provider.request_records[0]["outcome"] == "success"
    assert provider.request_records[0]["usage_status"] == "reported"
    assert provider.request_records[0]["usage"] == {
        "prompt_tokens": 200000,
        "completion_tokens": 70000,
        "total_tokens": 270000,
        "reasoning_tokens": 60000,
    }
    budget = session.snapshot()["budget"]
    assert budget["used_total_tokens"] == 270000
    assert budget["remaining_total_tokens"] == 0


def test_failed_task_preserves_http200_failure_extractor_overcap_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate E: learning rejection cannot roll back outcome or provider usage."""

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    payload = _success("call_failure_f1_over_cap", "submit_failure_alignment")
    payload["usage"] = {
        "prompt_tokens": 200000,
        "completion_tokens": 70000,
        "total_tokens": 270000,
        "completion_tokens_details": {"reasoning_tokens": 60000},
    }

    def post(_url, *, headers, json, timeout):
        return _Response(payload, request_id="req_failure_f1_over_cap")

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    provider = OpenAICompatibleProvider(
        _config(max_completion_tokens=131072)
    )
    harness = FakeHarness()
    task = fake_task("failed-over-cap", "apple_1")
    config = {
        "schema_version": 3,
        "data_dir": str(tmp_path / "data_v3"),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "TEST_DEEPSEEK_KEY",
            "extractor": {
                "max_completion_tokens": 131072,
                "max_total_tokens_per_task": 262144,
            },
        },
        "experiment": {
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "output_dir": str(tmp_path / "run"),
        },
        "cold_start": {
            "enabled": True,
            "failure_extractor_enabled": True,
        },
    }
    with AtomicSkillGraphSystem(
        config,
        harness=harness,
        provider={"extractor": provider, "default": provider},
    ) as system:
        contract = harness.task_contract(task)
        requirement = CapabilityRequirement(
            "req_hold",
            "hold the target item",
            list(contract.target_effects),
            [ParameterSpec("item", "entity", runtime_resolvable=True)],
            [],
            [],
            [],
            True,
            "the task requires the target item to be held",
        )
        bundle = PlannerRequirementBundle([requirement], [])
        expansion = system.planner.multiplicity_compiler.expand(bundle, contract)
        instance_id = expansion.instances[0].instance_id
        proposal = ColdStartPlanProposal(
            "cold-failure-over-cap",
            [ColdStartPlanStep(
                "unresolved-hold",
                [instance_id],
                ColdStartCandidateSource.UNRESOLVED,
                "",
                ColdStartExecutionMode.DYNAMIC,
                {},
                {},
                [],
            )],
            ["unresolved-hold"],
            [],
            [],
            {instance_id: ["unresolved-hold"]},
            [],
        )

        def failed_runtime(
            _task, *, mode, trace_builder, attempt_id="",
        ):
            trace = trace_builder.trace
            trace.task_contract = to_primitive(contract)
            trace.requirement_bundle = to_primitive(bundle)
            trace.requirement_expansion = to_primitive(expansion)
            trace.runtime_plan = {
                "source": "full_dynamic",
                "failure_stage": "runtime",
            }
            trace.cold_start_plan = ColdStartPlanRecord(
                proposal.plan_id,
                to_primitive(proposal),
                {"passed": True},
                False,
                [],
                "unresolved-hold",
            )
            trace.benchmark_success = False
            trace.task_contract_success = False
            trace.strict_task_success = False
            trace.learning_eligible = False
            trace.infrastructure_failure = False
            trace.failures = [FailureEnvelope(
                "failure-runtime-budget",
                FailureLayer.RUNTIME_AGENT,
                "runtime_task_token_budget_exhausted",
                task.task_id,
                trace.trace_id,
                "",
                "runtime",
                True,
                message="runtime task budget exhausted",
            )]
            return trace_builder.finish()

        system.orchestrator.run_task = failed_runtime
        system.failure_processor.localize = lambda _trace: []
        system.extraction_policy.decide = lambda _trace: type(
            "Decision", (), {"should_extract": False, "reasons": ["task_failed"]}
        )()
        assert system.evolution_maintenance is not None
        system.evolution_maintenance.prepare_failure_repairs = (
            lambda *_args, **_kwargs: []
        )

        result = system.run_task(task)

        assert result.benchmark_success is False
        assert result.strict_task_success is False
        assert result.infrastructure_failure is False
        assert result.failure_extraction is not None
        assert result.failure_extraction.rejection == {
            "code": "failure_extractor_budget_exhausted",
            "stage": "f1",
            "source_code": "extractor_token_budget_exhausted",
        }
        f1_events = [
            event for event in result.llm_usage
            if event["bucket"] == "failure_extractor_f1"
        ]
        assert len(f1_events) == 1
        assert f1_events[0]["prompt_tokens"] == 200000
        assert f1_events[0]["completion_tokens"] == 70000
        assert f1_events[0]["reasoning_tokens"] == 60000
        assert f1_events[0]["total_tokens"] == 270000
        assert f1_events[0]["latency_ms"] >= 0
        assert len(result.provider_requests) == 1
        assert result.provider_requests[0].request_id == "req_failure_f1_over_cap"
        assert result.provider_requests[0].usage_status == "reported"
        sessions = [
            item for item in result.agent_sessions
            if item.session_type == "FailureExtractorF1Session"
        ]
        assert len(sessions) == 1
        exhausted = sessions[0].snapshot["budget"]
        assert exhausted["used_total_tokens"] == 270000
        assert exhausted["remaining_total_tokens"] == 0
        assert provider.request_records[0]["usage"] == {
            "prompt_tokens": 200000,
            "completion_tokens": 70000,
            "total_tokens": 270000,
            "reasoning_tokens": 60000,
        }
        metrics = result.metadata["failure_extractor_metrics"]
        assert metrics["failure_extractor_f1_tokens"] == 270000
        assert metrics["failure_extractor_f1_provider_call_count"] == 1
        assert (
            metrics[
                "failure_extractor_usage_persisted_after_rejection_count"
            ]
            == 1
        )
        row = trace_to_row(result)
        assert row["failure_extractor_usage_persisted_after_rejection_count"] == 1
        persisted = list(system.traces.iter_payloads())
        assert len(persisted) == 1
        assert persisted[0]["failure_extraction"]["rejection"] == (
            result.failure_extraction.rejection
        )


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


def test_malformed_native_tool_arguments_use_session_protocol_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    malformed = _success("call_bad", "submit_probe")
    malformed["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = '{"ok":true}{"extra":1}'
    responses = iter([
        _Response(malformed, request_id="req_bad_arguments"),
        _Response(_success("call_fixed", "submit_probe"), request_id="req_fixed"),
    ])
    posted: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        posted.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    provider = OpenAICompatibleProvider(_config())
    session = ReplayAgentSession(
        provider,
        system_prompt="probe",
        usage_ledger=UsageLedger(),
        usage_bucket="planner_p1",
    )

    turn = session.next_turn("submit ok", tools=[_tool()])

    assert turn.tool_calls[0].name == "submit_probe"
    assert turn.tool_calls[0].arguments == {"ok": True}
    assert len(posted) == 2
    assert "PROTOCOL REPAIR REQUIRED" in posted[1]["messages"][-1]["content"]
    snapshot = session.snapshot()
    assert snapshot["protocol_repairs_used"] == 1
    assert snapshot["protocol_failures"][0]["code"] == "runtime_agent_schema_error"
    assert snapshot["terminal_protocol_failure"] is None
    assert [item["outcome"] for item in provider.request_records] == ["error", "success"]
    assert provider.request_records[0]["usage_status"] == "reported"


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
        task_semantic_context={"object": "apple", "destination": "fridge"},
        current_occurrence_semantic_anchors={"object": "apple"},
        execution_ready_bindings={"held_object": "apple_2"},
        missing_or_insufficient_bindings=["destination"],
        observation="holding apple_2",
        action_catalog=[{
            "action_id": "r000_a001",
            "action_type": "GO_TO",
            "arguments": {"destination": "coffeetable_1"},
            "display_text": "go to coffeetable 1",
            "revision": 0,
        }],
        relevant_action_history=[],
        remaining_budget={"actions": 10},
        implementation_invocations=[{
            "name": "invoke_impl_0123456789abcdef",
            "description": "navigate to source",
            "input_schema": {"type": "object", "properties": {}},
            "implementation_ref": {
                "logical_id": "impl_navigate_to_the_source_location_of_the_target",
                "version": "1.0.0",
            },
            "atomic_ref": {"logical_id": "atomic_navigate", "version": "1.0.0"},
        }],
    )
    payload = json.loads(payload_text.split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1])
    assert payload["task_semantic_context"] == {
        "object": "apple", "destination": "fridge",
    }
    assert payload["current_occurrence_semantic_anchors"] == {"object": "apple"}
    assert "semantic_anchors" not in payload
    assert payload["current_action_catalog"]["actions"][0]["arguments"] == {
        "destination": "coffeetable_1",
    }
    assert "copy canonical values exactly" in payload_text
    assert "action_catalog.actions[].arguments in environment tool results" in payload_text
    assert "portable semantic guidance" in payload_text
    assert "never current bindings or evidence" in payload_text
    assert "the task's final destination" in payload_text
    assert "applies only to roles absent" in payload_text
    assert "When a role is explicitly anchored there" in payload_text
    assert "call the exact native-tool name" in payload_text
    assert payload["allowed_implementation_invocations"] == [{
        "name": "invoke_impl_0123456789abcdef",
        "description": "navigate to source",
        "input_schema": {"type": "object", "properties": {}},
    }]
    assert payload["execution_ready_bindings"] == {"held_object": "apple_2"}
    assert payload["missing_or_insufficient_bindings"] == ["destination"]
    assert "certified_bindings" not in payload

    seeded_text = ContextBuilder().seeded_node(
        task_goal="put apple in fridge",
        atomic_contract={"summary": "move to old desk", "inputs": [], "outputs": []},
        task_semantic_context={"object": "apple", "destination": "fridge"},
        current_occurrence_semantic_anchors={},
        execution_ready_bindings={},
        missing_or_insufficient_bindings=["destination"],
        observation="in a new room",
        action_catalog=[],
        relevant_action_history=[],
        remaining_budget={"actions": 10},
    )
    assert "portable semantic guidance" in seeded_text
    assert "never current bindings or evidence" in seeded_text
    assert "When a role is explicitly anchored there" in seeded_text


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


def test_structured_semantic_budget_keeps_one_session_wide_format_repair_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    no_call = {
        "id": "response_no_call",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "prose instead of the submit call",
                "reasoning_content": "repairable private reasoning",
            },
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    responses = iter([
        _Response(no_call, request_id="req_bad"),
        *[
            _Response(
                _success(f"call_{index}", f"submit_stage_{index}"),
                request_id=f"req_{index}",
            )
            for index in range(1, 5)
        ],
    ])
    posted: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        posted.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    semantic_turns = 4
    session = ReplayAgentSession(
        OpenAICompatibleProvider(_config()),
        system_prompt="planner",
        usage_ledger=UsageLedger(),
        usage_bucket="planner_p1",
        budget=AgentBudget(
            structured_provider_turn_cap(semantic_turns),
            1000,
            "planner_token_budget_exhausted",
        ),
        semantic_max_turns=semantic_turns,
    )
    client = StructuredSubmissionClient()
    for index in range(1, 5):
        result = client.request(
            session,
            prompt=f"semantic stage {index}",
            tool_name=f"submit_stage_{index}",
            description="submit stage",
            schema=_tool().input_schema,
        )
        assert result.value == {"ok": True}

    snapshot = session.snapshot()
    assert snapshot["provider_call_count"] == 5
    assert snapshot["accepted_turn_count"] == 4
    assert snapshot["protocol_repairs_used"] == 1
    assert snapshot["budget"]["max_turns"] == 5
    assert snapshot["semantic_budget"] == {
        "max_turns": 4,
        "used_turns": 4,
        "remaining_turns": 0,
    }
    with pytest.raises(BudgetExhausted, match="semantic turn budget"):
        client.request(
            session,
            prompt="forbidden fifth semantic stage",
            tool_name="submit_stage_5",
            description="submit stage",
            schema=_tool().input_schema,
        )
    assert len(posted) == 5
    assert structured_provider_turn_cap(2) == 3
    assert structured_provider_turn_cap(1) == 2


def test_protocol_format_repair_quota_is_global_to_replay_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    no_call = {
        "id": "response_no_call",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "prose",
                "reasoning_content": "private reasoning",
            },
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    responses = iter([
        _Response(no_call, request_id="req_bad_1"),
        _Response(_success("call_fixed", "submit_first"), request_id="req_fixed"),
        _Response(no_call, request_id="req_bad_2"),
    ])
    posted: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        posted.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    session = ReplayAgentSession(
        OpenAICompatibleProvider(_config()),
        system_prompt="extractor",
        usage_ledger=UsageLedger(),
        usage_bucket="extractor_e1",
        budget=AgentBudget(3, 1000, "extractor_token_budget_exhausted"),
        semantic_max_turns=2,
    )
    client = StructuredSubmissionClient()
    assert client.request(
        session,
        prompt="E1",
        tool_name="submit_first",
        description="submit E1",
        schema=_tool().input_schema,
    ).value == {"ok": True}
    with pytest.raises(AgentProtocolError, match="no native tool call"):
        client.request(
            session,
            prompt="E2",
            tool_name="submit_second",
            description="submit E2",
            schema=_tool().input_schema,
        )
    assert len(posted) == 3
    snapshot = session.snapshot()
    assert snapshot["accepted_turn_count"] == 1
    assert snapshot["protocol_repairs_used"] == 1
    assert snapshot["terminal_protocol_failure"] is not None


def test_replay_compacts_only_superseded_action_catalogs_and_keeps_reasoning() -> None:
    from experiments.fakes import FakeReply, ScriptedAgentProvider

    def action_tool(action_id: str) -> NativeToolSpec:
        return NativeToolSpec(
            "environment_action",
            "Execute one current action.",
            {
                "type": "object",
                "required": ["action_id"],
                "additionalProperties": False,
                "properties": {
                    "action_id": {"type": "string", "enum": [action_id]},
                },
            },
        )

    provider = ScriptedAgentProvider([
        FakeReply.tool("environment_action", {"action_id": "r000_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r001_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r002_a001"}),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_dynamic",
    )
    first = session.next_turn("start", tools=[action_tool("r000_a001")])
    second = session.submit_tool_result(
        first.tool_calls[0].call_id,
        {
            "accepted": True,
            "observation": "first observation",
            "new_revision": 1,
            "action_catalog": [{
                "action_id": "r001_a001",
                "display_text": "go to desk 1",
            }],
        },
        tools=[action_tool("r001_a001")],
    )
    third = session.submit_tool_result(
        second.tool_calls[0].call_id,
        {
            "accepted": True,
            "observation": "second observation",
            "new_revision": 2,
            "action_catalog": [{
                "action_id": "r002_a001",
                "display_text": "take apple 1",
            }],
        },
        tools=[action_tool("r002_a001")],
    )
    session.acknowledge_tool_result(
        third.tool_calls[0].call_id, {"accepted": True, "complete": True},
    )

    replay_messages = provider.requests[2].messages
    tool_payloads = [
        json.loads(message["content"])
        for message in replay_messages
        if message["role"] == "tool"
    ]
    assert tool_payloads[0]["observation"] == "first observation"
    assert tool_payloads[0]["action_catalog"] == {
        "status": "superseded",
        "entry_count": 1,
        "superseded_by_revision": 2,
    }
    assert tool_payloads[1]["action_catalog"] == [{
        "action_id": "r002_a001",
        "display_text": "take apple 1",
    }]
    first_reasoning = next(
        message["reasoning_content"]
        for message in provider.requests[1].messages
        if message["role"] == "assistant"
    )
    assert next(
        message["reasoning_content"]
        for message in replay_messages
        if message["role"] == "assistant"
    ) == first_reasoning
    assert session.snapshot()["context_compaction_count"] == 1


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


def test_provider_probe_accepts_empty_terminal_reasoning_after_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret-fixture-key")
    responses = iter([
        _Response(_success("probe_a", "submit_probe"), request_id="req_a"),
        _Response(_success("probe_b1", "probe_step_one"), request_id="req_b1"),
        _Response(
            _success("probe_b2", "probe_step_two", reasoning=""),
            request_id="req_b2",
        ),
        _Response(
            _success("probe_c", "submit_extractor_probe"), request_id="req_c",
        ),
    ])
    payloads: list[dict[str, Any]] = []

    def post(_url, *, headers, json, timeout):
        payloads.append(json)
        return next(responses)

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    manifest = run_provider_capability_probe(
        _capability_config(),
        output_dir=tmp_path,
        config_hash="config_fixture",
        code_hash="code_fixture",
    )

    assert manifest["passed"] is True
    trace = json.loads(
        (tmp_path / "provider_probe_trace.json").read_text(encoding="utf-8")
    )
    replay = trace["probes"]["reasoning_replay"]
    assert replay["reasoning_content_replayed_verbatim"] is True
    assert replay["second_reasoning_content_present"] is False
    first_reasoning = next(
        message["reasoning_content"]
        for message in payloads[2]["messages"]
        if message["role"] == "assistant"
    )
    assert first_reasoning == "provider-private-reasoning"


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
