from __future__ import annotations

import copy
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import atomic_skillgraph.system as system_module
from atomic_skillgraph.agents.session import ReplayAgentSession
from atomic_skillgraph.agents.usage import AgentBudget, UsageBucket, UsageLedger
from atomic_skillgraph.agents.provider import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)
from atomic_skillgraph.core.contracts import SemanticPredicate, TaskContract
from atomic_skillgraph.core.errors import (
    AgentProtocolError,
    ArtifactIntegrityError,
    BudgetExhausted,
    FailureLayer,
)
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.extractor_session import ExtractionContentError
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from experiments.fakes import FakeAgentFactory, FakeReply, ScriptedAgentProvider
from experiments.fakes import FakeHarness, fake_task
from experiments.protocol import (
    AttemptTraceLedger,
    ManifestStore,
    RunManifest,
    RunState,
    TaskCheckpointStore,
    TaskManifest,
)
from tests.test_r91_task_deployment_review import (
    COMPOSITE_REF,
    _config as _task_review_config,
    _install_runtime_outcome,
    _register_composite,
)
from tests.test_v32_r4_learning_retention import (
    _OneOccurrenceExtractor,
    _RaisingSession,
    _create_take_tool_payload,
    _evidence_reason,
    _minimal_system,
    _no_tool_payload,
    _normalized_take,
    _take_proposal,
)


_BUDGET_CODE = "tool_builder_token_budget_exhausted"


def _install_real_usage(system: Any, *, cap: int) -> UsageLedger:
    ledger = UsageLedger()
    system.usage = ledger
    system.config = {
        "method_patch": "3.2",
        "llm": {"extractor": {"max_total_tokens_per_task": int(cap)}},
    }
    system._current_task_usage_start = 0
    return ledger


def _real_budgeted_session(
    ledger: UsageLedger,
    reply: FakeReply,
    *,
    max_tokens: int,
    session_id: str,
) -> tuple[ReplayAgentSession, ScriptedAgentProvider]:
    provider = ScriptedAgentProvider([reply], provider_id=session_id)
    session = ReplayAgentSession(
        provider,
        system_prompt="Submit one bounded Tool proposal.",
        usage_ledger=ledger,
        usage_bucket=UsageBucket.TOOL_BUILDER_EVOLUTION,
        budget=AgentBudget(2, max_tokens, _BUDGET_CODE),
        session_id=session_id,
    )
    return session, provider


class _HTTPResponse:
    def __init__(self, payload: dict[str, Any], *, request_id: str) -> None:
        self._payload = copy.deepcopy(payload)
        self.status_code = 200
        self.ok = True
        self.text = ""
        self.headers = {"x-request-id": request_id}

    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)


def _tool_builder_http_payload(
    tool_payload: dict[str, Any],
    *,
    prompt_tokens: int = 6,
    completion_tokens: int = 4,
) -> dict[str, Any]:
    return {
        "id": "response-r91-tool-budget",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "reasoning_content": "bounded fixture reasoning",
                "tool_calls": [{
                    "id": "call-r91-tool-budget",
                    "type": "function",
                    "function": {
                        "name": "create_tool",
                        "arguments": json.dumps(
                            tool_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


def _audited_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(OpenAICompatibleConfig(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="TEST_R91_DEEPSEEK_KEY",
        max_completion_tokens=131072,
        max_retries=0,
    ))


def _real_http_budgeted_session(
    ledger: UsageLedger,
    provider: OpenAICompatibleProvider,
    *,
    max_tokens: int,
    session_id: str,
) -> ReplayAgentSession:
    return ReplayAgentSession(
        provider,
        system_prompt="Submit one bounded Tool proposal.",
        usage_ledger=ledger,
        usage_bucket=UsageBucket.TOOL_BUILDER_EVOLUTION,
        budget=AgentBudget(2, max_tokens, _BUDGET_CODE),
        session_id=session_id,
    )


def _canonical_take(system: Any):
    occurrence = Atomicizer().validate_and_canonicalize(
        [_take_proposal()], _normalized_take(),
    )[0]
    atomic = system._canonical_atomic_for_occurrence(occurrence)
    assert atomic is not None
    return occurrence, atomic


def test_c01_zero_balance_skips_session_and_retains_atomic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(
            AssertionError("zero budget must not enter ToolBuilder"),
        ),
        static_validator=ToolStaticValidator(),
    )
    _install_real_usage(system, cap=0)
    session_calls: list[tuple[Any, ...]] = []

    def forbidden_session(*args: Any, **kwargs: Any) -> Any:
        session_calls.append((args, kwargs))
        raise AssertionError("zero budget must not allocate a session")

    system._tool_builder_session = forbidden_session
    trace.trace_id = "trace_r91_zero_budget"

    prepared = system._prepare_evolution(trace, task)
    assert len(prepared.compiled) == 1
    assert prepared.compiled[0].tool is None
    applied = system._apply_evolution(prepared, trace, task)

    assert session_calls == []
    assert len(applied["atomic_refs"]) == 1
    assert applied["tool_refs"] == []
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "budget_exhausted"
    assert record["builder_entered"] is False
    assert record["session_id"] == ""
    assert record["proposal_received"] is False
    assert record["static_checked"] is False
    assert record["atomic_only_prepared"] is True
    assert record["atomic_registered"] is True
    assert record["budget_boundary"] == "before_session"
    assert record["shared_remaining_before"] == 0
    assert record["shared_remaining_after"] == 0
    assert record["budget_provider_call_count"] == 0
    assert record["error_code"] == _BUDGET_CODE
    assert record["failure_codes"] == [_BUDGET_CODE]
    metrics = trace.metadata["v32_metrics"]
    assert metrics["tool_builder_budget_exhausted_count"] == 1
    assert metrics["tool_builder_budget_skipped_before_session_count"] == 1
    assert metrics["atomic_only_prepared_after_tool_budget_count"] == 1
    assert metrics["atomic_only_retained_after_tool_budget_count"] == 1
    assert metrics["tool_builder_call_count"] == 0
    assert metrics["tool_builder_no_tool_count"] == 0
    assert _evidence_reason(database, trace.trace_id) == (
        "tool_builder_budget_exhausted_atomic_only"
    )
    database.close()


def test_c02_exact_reuse_precedes_zero_balance_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(
            AssertionError("exact reuse must not enter ToolBuilder"),
        ),
        static_validator=ToolStaticValidator(),
    )
    _install_real_usage(system, cap=0)
    occurrence, atomic = _canonical_take(system)
    legacy = ToolCompiler().compile([occurrence])[0]
    assert legacy.tool is not None and legacy.implementation is not None
    staged = system.aligner.stage_atomic(
        atomic, legacy.tool, legacy.implementation,
    )
    system.skills.register_atomic(replace(staged.atomic, status=SkillStatus.CANDIDATE))
    system.tools.register(replace(staged.tool, status=ToolStatus.CANDIDATE))
    system.skills.register_implementation(
        replace(staged.implementation, status=SkillStatus.CANDIDATE)
    )
    session_calls: list[tuple[Any, ...]] = []

    def forbidden_session(*args: Any, **kwargs: Any) -> Any:
        session_calls.append((args, kwargs))
        raise AssertionError("exact reuse must not allocate a session")

    system._tool_builder_session = forbidden_session

    prepared = system._prepare_evolution(trace, task)

    assert len(prepared.compiled) == 1
    assert prepared.compiled[0].tool is not None
    assert session_calls == []
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "exact_reuse"
    assert record["shared_remaining_before"] is None
    assert record["budget_boundary"] == ""
    assert trace.metadata["v32_metrics"]["tool_builder_budget_exhausted_count"] == 0
    database.close()


def test_c03_provider_usage_overrun_is_retained_with_atomic_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(AssertionError("replaced below")),
        static_validator=ToolStaticValidator(),
    )
    ledger = _install_real_usage(system, cap=5)
    monkeypatch.setenv("TEST_R91_DEEPSEEK_KEY", "r91-secret-fixture")
    posted: list[dict[str, Any]] = []

    def post(_url: str, *, headers, json, timeout):
        assert headers["Authorization"] == "Bearer r91-secret-fixture"
        posted.append(copy.deepcopy(json))
        return _HTTPResponse(
            _tool_builder_http_payload(_create_take_tool_payload()),
            request_id="req-r91-tool-budget-c03",
        )

    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    provider = _audited_provider()
    session = _real_http_budgeted_session(
        ledger,
        provider,
        max_tokens=5,
        session_id="r91_tool_builder_overrun",
    )
    system._tool_builder_session = lambda *_args: session
    trace.trace_id = "trace_r91_provider_overrun"

    prepared = system._prepare_evolution(trace, task)
    assert len(prepared.compiled) == 1
    assert prepared.compiled[0].tool is None
    applied = system._apply_evolution(prepared, trace, task)

    assert len(ledger.events) == 1
    usage = ledger.events[0]
    assert usage.session_id == session.session_id
    assert usage.bucket is UsageBucket.TOOL_BUILDER_EVOLUTION
    assert usage.usage.call_count == 1
    assert usage.usage.total_tokens == 10
    assert len(posted) == 1
    assert posted[0]["tools"][0]["function"]["name"] == "create_tool"
    assert session.snapshot()["provider_call_count"] == 1
    assert provider.request_record_count == 1
    request = provider.request_records[0]
    assert request["provider_request_id"] == "req-r91-tool-budget-c03"
    assert request["session_id"] == session.session_id
    assert request["stage"] == "tool_builder_evolution"
    assert request["outcome"] == "success"
    assert request["usage_status"] == "reported"
    assert request["usage"] == {
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "total_tokens": 10,
        "reasoning_tokens": 2,
    }
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "budget_exhausted"
    assert record["budget_boundary"] == "builder_request"
    assert record["shared_remaining_before"] == 5
    assert record["shared_remaining_after"] == 0
    assert record["budget_provider_call_count"] == 1
    assert record["builder_entered"] is True
    assert record["session_id"] == session.session_id
    assert record["proposal_received"] is False
    assert record["static_checked"] is False
    assert record["atomic_only_prepared"] is True
    assert record["atomic_registered"] is True
    assert applied["tool_refs"] == []
    assert trace.metadata["tool_build_rejections"] == []
    metrics = trace.metadata["v32_metrics"]
    assert metrics["tool_builder_call_count"] == 1
    assert metrics["tool_builder_no_tool_count"] == 0
    assert metrics["tool_builder_aborted_count"] == 0
    assert metrics["tool_builder_budget_exhausted_count"] == 1
    assert metrics["atomic_only_retained_after_tool_budget_count"] == 1
    assert _evidence_reason(database, trace.trace_id) == (
        "tool_builder_budget_exhausted_atomic_only"
    )
    database.close()


def test_session_allocation_budget_exhaustion_is_retained_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(AssertionError("replaced below")),
        static_validator=ToolStaticValidator(),
    )
    ledger = _install_real_usage(system, cap=100)
    allocation_error = BudgetExhausted(
        _BUDGET_CODE,
        "allocation budget exhausted",
        layer=FailureLayer.RUNTIME_AGENT,
    )

    def exhausted_allocation(*_args: Any) -> Any:
        raise allocation_error

    system._tool_builder_session = exhausted_allocation
    prepared = system._prepare_evolution(trace, task)
    system._apply_evolution(prepared, trace, task)

    assert ledger.events == ()
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "budget_exhausted"
    assert record["budget_boundary"] == "session_allocation"
    assert record["shared_remaining_before"] == 100
    assert record["shared_remaining_after"] == 100
    assert record["budget_provider_call_count"] == 0
    assert record["builder_entered"] is False
    assert record["atomic_only_prepared"] is True
    assert record["atomic_registered"] is True
    database.close()


@pytest.mark.parametrize(
    "allocation_error",
    [
        BudgetExhausted(
            "extractor_token_budget_exhausted",
            "wrong allocation code",
            layer=FailureLayer.RUNTIME_AGENT,
        ),
        BudgetExhausted(
            _BUDGET_CODE,
            "wrong allocation layer",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
    ],
)
def test_session_allocation_only_neutralizes_exact_budget_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    allocation_error: BudgetExhausted,
) -> None:
    system, trace, _task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(AssertionError("replaced below")),
        static_validator=ToolStaticValidator(),
    )
    _install_real_usage(system, cap=100)
    occurrence, atomic = _canonical_take(system)
    system._initialize_r4_learning_diagnostics(trace)

    def rejected_allocation(*_args: Any) -> Any:
        raise allocation_error

    system._tool_builder_session = rejected_allocation
    with pytest.raises(BudgetExhausted) as caught:
        system._build_tool_for_occurrence(
            occurrence, atomic, _normalized_take(), trace,
        )

    assert caught.value is allocation_error
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "aborted"
    assert record["budget_boundary"] == ""
    assert trace.metadata["v32_metrics"]["tool_builder_budget_exhausted_count"] == 0
    database.close()


def test_c06_atomic_only_stage_rejection_is_local_and_not_fabricated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(AssertionError("must not be called")),
        static_validator=ToolStaticValidator(),
    )
    _install_real_usage(system, cap=0)
    system._stage_atomic_only_occurrence = lambda *_args: (_ for _ in ()).throw(
        ValueError("atomic stage rejected")
    )

    with pytest.raises(ExtractionContentError):
        system._prepare_evolution(trace, task)

    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "budget_exhausted"
    assert record["atomic_only_prepared"] is False
    assert record["atomic_registered"] is False
    assert trace.metadata["tool_build_rejections"] == []
    assert trace.metadata["knowledge_preparation_rejections"] == [{
        "occurrence_id": record["occurrence_id"],
        "phase_id": "take",
        "stage": "atomic_only_stage_after_tool_budget",
        "error_type": "ValueError",
        "error_code": "knowledge_preparation_failed",
        "failure_codes": [],
        "messages": ["atomic stage rejected"],
    }]
    metrics = trace.metadata["v32_metrics"]
    assert metrics["atomic_only_prepared_after_tool_budget_count"] == 0
    assert metrics["atomic_only_retained_after_tool_budget_count"] == 0
    assert system.skills.list_refs("atomic") == []
    database.close()


def test_c07_no_tool_remains_distinct_from_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    factory = FakeAgentFactory()
    session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _no_tool_payload())],
    )
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=session,
        static_validator=ToolStaticValidator(),
    )
    system.usage = factory.usage_ledger
    system.config = {
        "method_patch": "3.2",
        "llm": {"extractor": {"max_total_tokens_per_task": 100}},
    }
    system._current_task_usage_start = 0

    prepared = system._prepare_evolution(trace, task)
    system._apply_evolution(prepared, trace, task)

    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "no_tool"
    assert record["budget_boundary"] == ""
    metrics = trace.metadata["v32_metrics"]
    assert metrics["tool_builder_no_tool_count"] == 1
    assert metrics["tool_builder_budget_exhausted_count"] == 0
    assert metrics["atomic_only_prepared_after_tool_budget_count"] == 0
    assert _evidence_reason(database, trace.trace_id) == (
        "tool_builder_no_tool_atomic_only"
    )
    factory.assert_exhausted()
    database.close()


@pytest.mark.parametrize(
    "error",
    [
        AgentProtocolError(
            "provider_auth_error",
            "authentication failed",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
        AgentProtocolError(
            "provider_timeout",
            "provider request timed out",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
        AgentProtocolError(
            "provider_usage_missing",
            "provider response omitted required usage",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
        sqlite3.OperationalError("injected ToolBuilder database failure"),
        ArtifactIntegrityError(
            "artifact_integrity_error",
            "artifact is corrupt",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
        RuntimeError("builder implementation crashed"),
        BudgetExhausted(
            "extractor_token_budget_exhausted",
            "wrong budget code",
            layer=FailureLayer.RUNTIME_AGENT,
        ),
        BudgetExhausted(
            _BUDGET_CODE,
            "right code, wrong layer",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
    ],
)
def test_c10_c11_unrelated_failures_are_not_neutralized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    error: BaseException,
) -> None:
    system, trace, _task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(error),
        static_validator=ToolStaticValidator(),
    )
    _install_real_usage(system, cap=100)
    occurrence, atomic = _canonical_take(system)
    system._initialize_r4_learning_diagnostics(trace)

    with pytest.raises(type(error)) as caught:
        system._build_tool_for_occurrence(
            occurrence, atomic, _normalized_take(), trace,
        )

    assert caught.value is error
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "aborted"
    assert record["atomic_only_prepared"] is False
    assert record["budget_boundary"] == ""
    assert trace.metadata["v32_metrics"]["tool_builder_budget_exhausted_count"] == 0
    database.close()


def test_c12_budget_metrics_are_recomputed_idempotently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(AssertionError("must not be called")),
        static_validator=ToolStaticValidator(),
    )
    _install_real_usage(system, cap=0)
    prepared = system._prepare_evolution(trace, task)
    system._apply_evolution(prepared, trace, task)
    expected = copy.deepcopy(trace.metadata["v32_metrics"])

    system._finalize_r4_learning_metrics(trace)
    system._finalize_r4_learning_metrics(trace)

    assert trace.metadata["v32_metrics"] == expected
    assert expected["tool_builder_budget_exhausted_count"] == 1
    assert expected["atomic_only_retained_after_tool_budget_count"] == 1
    database.close()


def test_c13_shared_balance_resets_at_next_task_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, _task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(AssertionError("replaced below")),
        static_validator=ToolStaticValidator(),
    )
    ledger = _install_real_usage(system, cap=5)
    first, _ = _real_budgeted_session(
        ledger,
        FakeReply.tool(
            "create_tool", _create_take_tool_payload(),
            prompt_tokens=6, completion_tokens=4,
        ),
        max_tokens=5,
        session_id="r91_previous_task",
    )
    system._tool_builder_session = lambda *_args: first
    occurrence, atomic = _canonical_take(system)
    system._initialize_r4_learning_diagnostics(trace)

    with pytest.raises(system_module._ToolBuildBudgetExhausted):
        system._build_tool_for_occurrence(
            occurrence, atomic, _normalized_take(), trace,
        )
    assert system._shared_tool_builder_tokens("tool_builder_evolution") == 0

    system._current_task_usage_start = len(ledger.events)
    second, provider = _real_budgeted_session(
        ledger,
        FakeReply.tool(
            "create_tool", _no_tool_payload(),
            prompt_tokens=2, completion_tokens=2,
        ),
        max_tokens=5,
        session_id="r91_next_task",
    )
    system._tool_builder_session = lambda *_args: second
    next_trace = SimpleNamespace(
        metadata={}, trace_id="trace_r91_next_task",
        task=SimpleNamespace(task_id="task_next"),
    )
    system._initialize_r4_learning_diagnostics(next_trace)

    item, _ = system._build_tool_for_occurrence(
        occurrence, atomic, _normalized_take(), next_trace,
    )

    assert item is None
    assert provider.snapshot()["call_count"] == 1
    assert next_trace.metadata["evolution_tool_builds"][0]["outcome"] == "no_tool"
    assert next_trace.metadata["evolution_tool_builds"][0][
        "shared_remaining_before"
    ] == 5
    database.close()


def _three_take_normalized() -> dict[str, Any]:
    normalized = _normalized_take("apple_1")
    second = copy.deepcopy(normalized["actions"][0])
    second.update({
        "event_index": 1,
        "event_id": "e1",
        "action_id": "e1",
        "arguments": {"item": "mug_1"},
        "before_revision": 1,
        "after_revision": 2,
        "span_id": "span_second",
    })
    second_effect = copy.deepcopy(second["authoritative_positive_effects"][0])
    second_effect.update({
        "args": {"object": "mug_1"},
        "witness_ref": "action:e1:revision:2",
        "event_index": 1,
        "revision": 2,
    })
    second["authoritative_positive_effects"] = [second_effect]
    normalized["actions"].append(second)
    normalized["runtime_spans"].append({
        "span_id": "span_second",
        "kind": "full_dynamic",
        "occurrence_id": "occ_second",
        "action_start": 1,
        "action_end": 2,
        "parent_span_id": None,
        "learnable": True,
    })
    normalized["boundary_authorities"]["inputs"].append({
        "authority_ref": "action_arg:e1:item",
        "event_id": "e1",
        "argument_role": "item",
        "kind": "action_argument",
        "source_kind": "action_argument",
        "role": "item",
        "value": "mug_1",
    })
    normalized["boundary_authorities"]["effects"].append(second_effect)
    third = copy.deepcopy(second)
    third.update({
        "event_index": 2,
        "event_id": "e2",
        "action_id": "e2",
        "arguments": {"item": "book_1"},
        "before_revision": 2,
        "after_revision": 3,
        "span_id": "span_third",
    })
    third_effect = copy.deepcopy(third["authoritative_positive_effects"][0])
    third_effect.update({
        "args": {"object": "book_1"},
        "witness_ref": "action:e2:revision:3",
        "event_index": 2,
        "revision": 3,
    })
    third["authoritative_positive_effects"] = [third_effect]
    normalized["actions"].append(third)
    normalized["runtime_spans"].append({
        "span_id": "span_third",
        "kind": "full_dynamic",
        "occurrence_id": "occ_third",
        "action_start": 2,
        "action_end": 3,
        "parent_span_id": None,
        "learnable": True,
    })
    normalized["boundary_authorities"]["inputs"].append({
        "authority_ref": "action_arg:e2:item",
        "event_id": "e2",
        "argument_role": "item",
        "kind": "action_argument",
        "source_kind": "action_argument",
        "role": "item",
        "value": "book_1",
    })
    normalized["boundary_authorities"]["effects"].append(third_effect)
    return normalized


def _canonical_take_tool_payload() -> dict[str, Any]:
    payload = _create_take_tool_payload()
    payload["inputs"][0].update({
        "semantic_type": "string",
        "required_resolution": "semantic",
    })
    payload["outputs"][0]["semantic_type"] = "string"
    payload["final_effects"][0]["args"]["object"]["source_role"] = "item"
    return payload


class _ThreeOccurrenceExtractor:
    def __init__(self, _session: object) -> None:
        pass

    def propose_atomics(self, *_args: Any, **_kwargs: Any):
        first = _take_proposal("apple_1")
        second = _take_proposal("mug_1")
        second.phase_id = "take_second"
        second.event_start = 1
        second.event_end = 1
        second.support_event_ids = ["e1"]
        second.effect_witness_refs = ["action:e1:revision:2"]
        second.input_provenance_refs = {"item": "action_arg:e1:item"}
        third = _take_proposal("book_1")
        third.phase_id = "take_third"
        third.event_start = 2
        third.event_end = 2
        third.support_event_ids = ["e2"]
        third.effect_witness_refs = ["action:e2:revision:3"]
        third.input_provenance_refs = {"item": "action_arg:e2:item"}
        return [first, second, third]

    def propose_composite(self, *_args: Any, **_kwargs: Any):
        raise AssertionError("incomplete fixture coverage must not invoke E2")


class _BudgetAtomicOnlyExtractor(_OneOccurrenceExtractor):
    def propose_composite(self, *_args: Any, **_kwargs: Any):
        raise BudgetExhausted(
            "extractor_token_budget_exhausted",
            "fixture keeps the already staged Atomic",
            layer=FailureLayer.RUNTIME_AGENT,
        )


class _BudgetE2RepairExtractor(_OneOccurrenceExtractor):
    e2_proposal = SimpleNamespace(existing_edges=[], new_edges=[])

    def propose_composite(self, *_args: Any, **_kwargs: Any):
        return type(self).e2_proposal

    def repair_composite(
        self,
        proposal: Any,
        rejection: Any,
        *_args: Any,
        **_kwargs: Any,
    ):
        assert proposal is type(self).e2_proposal
        assert str(rejection) == "initial E2 validation rejection"
        raise BudgetExhausted(
            "extractor_token_budget_exhausted",
            "E2R exhausted after the Atomic was staged",
            layer=FailureLayer.RUNTIME_AGENT,
        )


class _RejectInitialE2:
    def validate_and_build(self, *_args: Any, **_kwargs: Any):
        raise ValueError("initial E2 validation rejection")


def test_c09_e2r_budget_rejection_retains_staged_atomic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    factory = FakeAgentFactory()
    builder_session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _no_tool_payload())],
    )
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=builder_session,
        static_validator=ToolStaticValidator(),
    )
    monkeypatch.setattr(
        system_module, "ExtractorSession", _BudgetE2RepairExtractor,
    )
    system.usage = factory.usage_ledger
    system.config = {
        "method_patch": "3.2",
        "llm": {"extractor": {"max_total_tokens_per_task": 100}},
    }
    system._current_task_usage_start = 0
    system.harness.task_contract = lambda _task: TaskContract(target_effects=[
        SemanticPredicate("agent.holds", {"object": "apple_1"}),
    ])
    system.composite_builder = _RejectInitialE2()

    prepared = system._prepare_evolution(trace, task)
    applied = system._apply_evolution(prepared, trace, task)

    assert len(prepared.compiled) == 1
    assert prepared.composite is None
    assert prepared.composite_rejection["error_code"] == (
        "extractor_token_budget_exhausted"
    )
    assert len(applied["atomic_refs"]) == 1
    registered_atomic_ref = str(applied["atomic_refs"][0])
    assert registered_atomic_ref in {
        str(ref) for ref in system.skills.list_refs("atomic")
    }
    system.skills.store.verify_ref(registered_atomic_ref)
    quality = trace.metadata["extractor_quality"]
    assert quality["extractor_e2_repair_attempt_count"] == 1
    assert quality["extractor_e2_repair_success_count"] == 0
    assert quality["extractor_e2_repair_failure_count"] == 1
    extraction = trace.metadata["extraction"]
    assert extraction["e2_repair_attempted"] is True
    assert extraction["e2_repair_applied"] is False
    assert "E2R exhausted" in extraction["e2_repair_error"]
    factory.assert_exhausted()
    database.close()


def test_c04_c05_budget_exhaustion_preserves_prepared_sibling_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    factory = FakeAgentFactory()
    first_session = factory.new_session(
        "tool_builder",
        [FakeReply.tool(
            "create_tool", _canonical_take_tool_payload(),
            prompt_tokens=6, completion_tokens=4,
        )],
    )
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=first_session,
        static_validator=ToolStaticValidator(),
    )
    monkeypatch.setattr(system_module, "ExtractorSession", _ThreeOccurrenceExtractor)
    normalized = _three_take_normalized()
    system.normalizer = SimpleNamespace(build=lambda _trace: copy.deepcopy(normalized))
    system.usage = factory.usage_ledger
    system.config = {
        "method_patch": "3.2",
        "llm": {"extractor": {"max_total_tokens_per_task": 10}},
    }
    system._current_task_usage_start = 0
    session_calls: list[str] = []

    def one_session(_kind: str, occurrence_id: str) -> Any:
        session_calls.append(str(occurrence_id))
        if len(session_calls) > 1:
            raise AssertionError("empty shared budget must not allocate again")
        return first_session

    system._tool_builder_session = one_session

    prepared = system._prepare_evolution(trace, task)
    applied = system._apply_evolution(prepared, trace, task)

    assert len(prepared.compiled) == 3
    assert trace.metadata["evolution_tool_builds"][0]["outcome"] == "created", (
        trace.metadata["evolution_tool_builds"][0]
    )
    assert prepared.compiled[0].tool is not None, trace.metadata[
        "evolution_tool_builds"
    ]
    assert prepared.compiled[1].tool is None
    assert prepared.compiled[2].tool is None
    assert len(session_calls) == 1
    records = trace.metadata["evolution_tool_builds"]
    assert [item["outcome"] for item in records] == [
        "created", "budget_exhausted", "budget_exhausted",
    ]
    assert records[1]["budget_boundary"] == "before_session"
    assert records[1]["atomic_only_prepared"] is True
    assert records[2]["budget_boundary"] == "before_session"
    assert records[2]["atomic_only_prepared"] is True
    assert len(applied["atomic_refs"]) == 3
    assert len(set(map(str, applied["atomic_refs"]))) == 1
    assert len(applied["tool_refs"]) == 1
    assert all(item["atomic_registered"] is True for item in records)
    admitted_atomic_rows = database.execute(
        "SELECT COUNT(*) AS count FROM evidence_events "
        "WHERE trace_id=? AND artifact_kind='atomic' AND event_type='validated'",
        (trace.trace_id,),
    ).fetchone()
    assert admitted_atomic_rows is not None
    assert int(admitted_atomic_rows["count"]) == 1
    metrics = trace.metadata["v32_metrics"]
    assert metrics["tool_builder_call_count"] == 1
    assert metrics["tool_builder_budget_exhausted_count"] == 2
    assert metrics["atomic_staged_occurrence_count"] == 3
    database.close()


def test_c14_d05_real_run_task_retains_budget_atomic_and_deployment_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    harness = FakeHarness()
    harness.task_contract = lambda _task: TaskContract(target_effects=[
        SemanticPredicate("state.uncovered", {"item": "apple_1"}),
    ])
    config = _task_review_config(tmp_path / "data_v3")
    config["method_patch"] = "3.2"
    config["llm"]["extractor"] = {
        "max_turns": 2,
        "max_total_tokens_per_task": 5,
    }
    monkeypatch.setattr(
        system_module, "ExtractorSession", _BudgetAtomicOnlyExtractor,
    )
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )

    provider = ScriptedAgentProvider([
        FakeReply.tool(
            "create_tool",
            _create_take_tool_payload(),
            prompt_tokens=6,
            completion_tokens=4,
        ),
    ], provider_id="r91-c14-provider")
    with system_module.AtomicSkillGraphSystem(
        config,
        harness=harness,
        provider=provider,
    ) as system:
        _register_composite(system, harness)
        _install_runtime_outcome(
            system,
            successful=True,
            task_rescue_required=True,
        )
        system.normalizer = SimpleNamespace(
            build=lambda _trace: copy.deepcopy(_normalized_take())
        )
        system.extraction_policy.decide = lambda _trace: SimpleNamespace(
            should_extract=True,
            reasons=["r91_budget_deployment_integration"],
        )
        assert system.lifecycle is not None
        real_review = system.lifecycle.review
        reviewed_refs: list[tuple[str, ...]] = []

        def observed_review(*, artifact_refs=None):
            reviewed_refs.append(tuple(artifact_refs or ()))
            return real_review(artifact_refs=artifact_refs)

        system.lifecycle.review = observed_review
        task = fake_task("r91-budget-and-deployment", "apple_1")
        task_manifest = TaskManifest.from_task(
            task,
            ordinal=0,
            knowledge_milestone="initial:r91-c14",
        )
        manifest = RunManifest.create(
            run_id="r91-c14-run",
            phase="train",
            config_hash="r91-c14-config",
            code_commit="r91-c14-code",
            knowledge_digest=system.knowledge_digest(),
            tasks=(task_manifest,),
        )
        store = ManifestStore(tmp_path / "run_manifest", system.database)
        store.persist_before_run(manifest)
        store.mark_run_state(manifest.run_id, RunState.RUNNING)
        sequence = store.mark_task_running(
            manifest.run_id, task.task_id, max_attempts=3,
        )
        attempts = AttemptTraceLedger(
            tmp_path / "attempt_history", system.traces.root,
        )
        attempt = attempts.begin(
            run_id=manifest.run_id,
            task_id=task.task_id,
            task_signature=task_manifest.task_signature,
            attempt_kind="task",
            sequence=sequence,
        )
        checkpoint = TaskCheckpointStore(
            tmp_path / ".task_checkpoint", system.data_dir,
        )
        checkpoint.create(
            system.database,
            run_id=manifest.run_id,
            task_id=task.task_id,
            before_digest=system.knowledge_digest(),
            config_hash=manifest.config_hash,
            code_commit=manifest.code_commit,
        )
        trace = system.run_task(task, attempt_id=attempt.attempt_id)
        attempts.capture(attempt, reason="run_task_returned")
        store.mark_task_completed(
            manifest.run_id,
            task.task_id,
            trace_id=trace.trace_id,
            result={"knowledge_digest_after": system.knowledge_digest()},
        )
        checkpoint.clear()

        assert trace.infrastructure_failure is False
        assert trace.benchmark_success is True
        assert trace.strict_task_success is True
        assert trace.learning_eligible is True
        record = trace.metadata["evolution_tool_builds"][0]
        assert record["outcome"] == "budget_exhausted"
        assert record["budget_boundary"] == "builder_request"
        assert record["budget_provider_call_count"] == 1
        assert record["atomic_only_prepared"] is True
        assert record["atomic_registered"] is True
        registered_atomic_ref = record["registered_atomic_ref"]
        assert registered_atomic_ref
        assert registered_atomic_ref in {
            str(ref) for ref in system.skills.list_refs("atomic")
        }
        system.artifacts.verify_ref(registered_atomic_ref)
        assert trace.metadata["v32_metrics"][
            "atomic_only_retained_after_tool_budget_count"
        ] == 1
        assert reviewed_refs == [(COMPOSITE_REF,)]
        assert provider.snapshot()["call_count"] == 1
        assert len(trace.llm_usage) == 1
        assert trace.llm_usage[0]["bucket"] == "tool_builder_evolution"
        assert trace.llm_usage[0]["total_tokens"] == 10
        builder_sessions = [
            item for item in trace.agent_sessions
            if item.session_type == "ToolBuilderSession"
        ]
        assert len(builder_sessions) == 1
        assert builder_sessions[0].session_id == record["session_id"]
        assert builder_sessions[0].snapshot["provider_call_count"] == 1
        composite_rejection = trace.metadata["extraction"][
            "composite_rejection"
        ]
        assert composite_rejection["error_code"] == (
            "extractor_token_budget_exhausted"
        )
        assert trace.metadata["evolution_applied"]["atomic_refs"]
        state = system.database.execute(
            "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?",
            (manifest.run_id, task.task_id),
        ).fetchone()
        assert state is not None and str(state["state"]) == "completed"
        assert not checkpoint.root.exists()
        assert system.projection is not None
        deployment_stats = system.projection.stats(COMPOSITE_REF, "composite")
        assert deployment_stats.independent_deployment_trial_count == 1
        assert deployment_stats.independent_deployment_success_count == 0
        assert deployment_stats.deployment_unsuccessful_count == 1
        rows = system.database.execute(
            "SELECT artifact_kind, artifact_ref, event_type, metadata_json "
            "FROM evidence_events WHERE trace_id=? ORDER BY rowid",
            (trace.trace_id,),
        ).fetchall()
        assert any(
            str(row["artifact_kind"]) == "composite"
            and str(row["artifact_ref"]) == COMPOSITE_REF
            and str(row["event_type"]) == "deployment_unsuccessful"
            for row in rows
        )
        assert any(
            str(row["artifact_kind"]) == "atomic"
            and "tool_builder_budget_exhausted_atomic_only"
            in str(row["metadata_json"])
            for row in rows
        )
