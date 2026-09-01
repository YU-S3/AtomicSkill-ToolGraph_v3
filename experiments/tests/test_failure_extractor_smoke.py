from __future__ import annotations

from types import SimpleNamespace

from atomic_skillgraph.agents import NativeToolSpec, validate_schema_instance
from atomic_skillgraph.planner.cold_start_agent import COLD_START_PLAN_SCHEMA
from experiments.run_v3_smoke import (
    _AllUnresolvedC1Provider,
    _failure_extractor_smoke_audit,
)


def _trace() -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_success=False,
        strict_task_success=False,
        infrastructure_failure=False,
        resource_usage_complete=True,
        failures=[{"code": "runtime_task_token_budget_exhausted"}],
        failure_extraction={
            "f1_alignment": {},
            "f1_validation": {},
            "f2_proposal": {},
            "provisional_refs": [],
            "failure_experience_ids": [],
            "rejection": {},
        },
        metadata={
            "failure_extractor_metrics": {
                "failure_extractor_f1_input_event_count": 1,
                "failure_extractor_f1_prompt_chars": 12000,
                "failure_extractor_f1_prompt_bytes": 14000,
                "failure_extractor_f2_span_count": 0,
                "failure_extractor_f2_source_event_count": 0,
                "failure_extractor_f2_prompt_chars": 2000,
                "failure_extractor_f2_prompt_bytes": 2100,
            },
        },
        cold_start_plan={
            "validation": {"passed": True},
            "proposal": {
                "steps": [{
                    "candidate_source": "unresolved",
                    "candidate_ref": "",
                    "execution_mode": "dynamic",
                }],
            },
        },
        llm_usage=[
            {
                "event_id": "usage-p1",
                "session_id": "session-p1",
                "bucket": "planner_p1",
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "total_tokens": 100,
                "call_count": 1,
            },
            {
                "event_id": "usage-c1",
                "session_id": "session-c1",
                "bucket": "cold_start_c1",
                "provider": "deterministic_c1_fixture",
                "model": "all-unresolved-v1",
                "total_tokens": 0,
                "call_count": 1,
            },
            {
                "event_id": "usage-runtime",
                "session_id": "session-runtime",
                "bucket": "runtime_dynamic_cold_start_continuation",
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "total_tokens": 101,
                "call_count": 1,
            },
            {
                "event_id": "usage-f1",
                "session_id": "session-f1",
                "bucket": "failure_extractor_f1",
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "reasoning_tokens": 10,
                "call_count": 1,
                "latency_ms": 10.0,
            },
            {
                "event_id": "usage-f2",
                "session_id": "session-f2",
                "bucket": "failure_extractor_f2",
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "total_tokens": 80,
                "call_count": 1,
            },
        ],
        provider_requests=[
            {
                "request_id": "request-c1",
                "session_id": "session-c1",
                "stage": "cold_start_c1",
                "usage_status": "reported",
                "http_status": None,
            },
            {
                "request_id": "request-f1",
                "session_id": "session-f1",
                "stage": "failure_extractor_f1",
                "usage_status": "reported",
            },
            {
                "request_id": "request-f2",
                "session_id": "session-f2",
                "stage": "failure_extractor_f2",
                "usage_status": "reported",
            },
        ],
    )


def test_failure_extractor_smoke_audit_accepts_bounded_audited_failure() -> None:
    audit = _failure_extractor_smoke_audit(_trace(), harness_max_steps=100)

    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["failure_codes"] == ["runtime_task_token_budget_exhausted"]


def test_failure_extractor_smoke_audit_rejects_unbounded_or_budget_failed_f1() -> None:
    trace = _trace()
    trace.metadata["failure_extractor_metrics"].update({
        "failure_extractor_f1_input_event_count": 101,
        "failure_extractor_f1_prompt_chars": 700000,
        "failure_extractor_f1_prompt_bytes": 2500000,
    })
    trace.failure_extraction["rejection"] = {
        "code": "failure_extractor_budget_exhausted",
        "stage": "f1",
    }
    trace.provider_requests[1]["usage_status"] = "unavailable"

    audit = _failure_extractor_smoke_audit(trace, harness_max_steps=100)

    assert audit["passed"] is False
    assert audit["checks"]["failure_extractor_budget_not_exhausted"] is False
    assert audit["checks"]["f1_input_event_count_bounded"] is False
    assert audit["checks"]["f1_prompt_chars_bounded"] is False
    assert audit["checks"]["f1_prompt_bytes_bounded"] is False
    assert audit["checks"]["f1_provider_requests_audited"] is False


def test_all_unresolved_c1_fixture_projects_only_required_instance_ids() -> None:
    provider = _AllUnresolvedC1Provider()
    provider.set_request_context(session_id="session-c1", stage="cold_start_c1")
    prompt = (
        "Task goal: ignored\n"
        "RequirementExpansion: {\"instances\":["
        "{\"instance_id\":\"required-a\",\"requirement\":{\"required\":true}},"
        "{\"instance_id\":\"optional-b\",\"requirement\":{\"required\":false}},"
        "{\"instance_id\":\"required-c\",\"requirement\":{\"required\":true}}"
        "]}\nVerified candidates: {}"
    )
    tool = NativeToolSpec(
        "submit_cold_start_plan",
        "test",
        COLD_START_PLAN_SCHEMA,
    )

    turn = provider.complete(
        [{"role": "user", "content": prompt}],
        tools=[tool],
    )

    assert turn.total_tokens == 0
    assert turn.provider_metadata["fixture"] is True
    assert len(turn.tool_calls) == 1
    proposal = turn.tool_calls[0].arguments
    validate_schema_instance(proposal, COLD_START_PLAN_SCHEMA)
    assert [
        item["requirement_instance_ids"] for item in proposal["steps"]
    ] == [["required-a"], ["required-c"]]
    assert all(item["candidate_ref"] == "" for item in proposal["steps"])
    assert provider.request_record_count == 1
    record = provider.request_records_since(0)[0]
    assert record["session_id"] == "session-c1"
    assert record["stage"] == "cold_start_c1"
    assert record["usage_status"] == "reported"
    assert record["http_status"] is None


def test_failure_extractor_smoke_audit_rejects_fixture_stage_pollution() -> None:
    trace = _trace()
    trace.llm_usage[-1]["provider"] = "deterministic_c1_fixture"

    audit = _failure_extractor_smoke_audit(trace, harness_max_steps=100)

    assert audit["passed"] is False
    assert audit["checks"]["real_provider_stages_uncontaminated"] is False
