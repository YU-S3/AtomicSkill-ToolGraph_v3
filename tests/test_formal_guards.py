from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.provider import (
    AgentProviderError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    _parse_usage,
)
from atomic_skillgraph.agents.provider_probe import (
    ProviderCapabilityError,
    ensure_provider_capability,
)
from atomic_skillgraph.agents.usage import AgentBudget, BudgetTracker, LLMUsage
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.errors import (
    ArtifactIntegrityError,
    AtomicSkillGraphError,
    BudgetExhausted,
    FailureLayer,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.runtime.budget import RuntimeBudget
from experiments.fakes import FakeHarness, fake_task
from experiments.protocol import (
    AttemptTraceRef,
    ProtocolError,
    audit_failed_attempt,
    hash_code,
)
from experiments.report import validate_formal_usage
from experiments.run_v3_frozen_eval import _validate_formal_config as _validate_frozen_config
from experiments.run_v3_train import _validate_formal_config
from experiments.run_v3_smoke import _validated_dataflow


def _config() -> dict:
    return {
        "schema_version": 3,
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "model",
            "api_key_env": "MODEL_API_KEY",
        },
        "experiment": {"condition": "full"},
    }


def _system_config(data_dir: Path) -> dict:
    config = _config()
    config["data_dir"] = str(data_dir)
    config["experiment"].update({
        "runtime_mode": "online",
        "freeze_skills": False,
        "output_dir": str(data_dir.parent),
    })
    return config


@pytest.mark.parametrize(
    "name",
    ["x-api-key", "subscription-key", "access-token", "Authorization", "cookie"],
)
def test_config_rejects_every_inline_auth_alias(name: str) -> None:
    config = _config()
    config["llm"][name] = "must-not-be-stored"
    with pytest.raises(ValueError, match="api_key_env only"):
        load_config(config)


def test_provider_usage_is_required_but_reasoning_may_be_unavailable() -> None:
    with pytest.raises(ValueError, match="missing required provider usage"):
        _parse_usage({"prompt_tokens": 2, "completion_tokens": 1})
    usage, metadata = _parse_usage({
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    })
    assert usage["reasoning_tokens"] is None
    assert metadata["usage_status"] == "reported"
    assert metadata["reasoning_tokens_status"] == "unavailable"


def test_reasoning_metering_never_adds_a_visible_token_budget_gate() -> None:
    budget = AgentBudget(2, 100, "runtime_node_token_budget_exhausted")
    first = BudgetTracker(budget)
    second = BudgetTracker(budget)
    first.consume(LLMUsage(1, 5, 6, reasoning_tokens=0, call_count=1))
    second.consume(LLMUsage(1, 5, 6, reasoning_tokens=5, call_count=1))
    assert first.snapshot() == second.snapshot()
    assert "max_visible_tokens_per_turn" not in first.snapshot()


def test_task_level_dynamic_keeps_global_usage_but_leaves_node_quota() -> None:
    budget = RuntimeBudget(global_action_budget=3, node_action_budget=1)
    budget.begin_node("occ_last")
    budget.consume_action()
    with pytest.raises(BudgetExhausted) as node_failure:
        budget.consume_action()
    assert getattr(node_failure.value, "code", "") == "runtime_node_action_budget_exhausted"

    budget.end_node()
    assert budget.snapshot()["node_budget_active"] is False
    budget.consume_action()
    budget.consume_action()
    with pytest.raises(BudgetExhausted) as global_failure:
        budget.consume_action()
    assert getattr(global_failure.value, "code", "") == "episode_action_budget_exhausted"

    tracker = BudgetTracker(AgentBudget(0, 1, "runtime_node_token_budget_exhausted"))
    with pytest.raises(BudgetExhausted) as token_failure:
        tracker.check_before_call()
    assert token_failure.value.code == "runtime_node_token_budget_exhausted"
    assert token_failure.value.layer is FailureLayer.RUNTIME_AGENT


@pytest.mark.parametrize(
    ("online", "last_maintenance", "interval", "expected"),
    [
        (3, 0, 5, ""),
        (4, 0, 5, "online_success_5"),
        (9, 5, 5, "online_success_10"),
        (8, 5, 5, ""),
    ],
)
def test_periodic_expectation_uses_pre_task_authoritative_counters(
    online: int, last_maintenance: int, interval: int, expected: str,
) -> None:
    system = object.__new__(AtomicSkillGraphSystem)
    system._online_successes = online
    system._last_maintenance_success_count = last_maintenance
    system.maintenance_interval = interval
    assert system.expected_periodic_maintenance_milestone_after_success() == expected


def test_code_hash_ignores_mutable_run_outputs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = hash_code(tmp_path)
    (tmp_path / "runs" / "formal" / "reports").mkdir(parents=True)
    (tmp_path / "runs" / "formal" / "run_manifest.json").write_text(
        '{"mutable":true}', encoding="utf-8"
    )
    (tmp_path / "runs" / "formal" / "reports" / "result.json").write_text(
        '{"tasks":30}', encoding="utf-8"
    )
    (tmp_path / "src" / "atomic_skillgraph.egg-info").mkdir()
    (tmp_path / "src" / "atomic_skillgraph.egg-info" / "SOURCES.txt").write_text(
        "generated packaging metadata\n", encoding="utf-8"
    )
    (tmp_path / "build" / "lib").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "module.py").write_text(
        "GENERATED = True\n", encoding="utf-8"
    )
    assert hash_code(tmp_path) == before


@pytest.mark.parametrize(
    ("sql", "parameters"),
    [
        (
            "INSERT INTO artifact_index VALUES(?,?,?,?,?,?,?,?)",
            ("atomic:probe@1", "atomic", "probe", "1", "hash", "candidate", "missing", 3),
        ),
        (
            "INSERT INTO recommended_pointers VALUES(?,?)",
            ("probe", "atomic:probe@1"),
        ),
        (
            "INSERT INTO graph_edges VALUES(?,?,?,?,?)",
            ("edge", "atomic:a@1", "atomic:b@1", "contains", "{}"),
        ),
        (
            "INSERT INTO evidence_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "event", 3, "task", "trace", "occ", "attempt", 0,
                "atomic:probe@1", "atomic", "selected", "", 1.0, "{}",
            ),
        ),
        (
            "INSERT INTO lifecycle_projection VALUES(?,?,?)",
            ("atomic:probe@1", "{}", 0),
        ),
        (
            "INSERT INTO projection_checkpoints VALUES(?,?)",
            ("lifecycle", 0),
        ),
        (
            "INSERT INTO metadata VALUES(?,?)",
            ("online_success_count", "0"),
        ),
    ],
)
def test_fresh_bank_gate_covers_every_long_term_table(
    tmp_path: Path,
    sql: str,
    parameters: tuple[object, ...],
) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        assert system.is_empty_knowledge_bank() is True
        system.database.execute(sql, parameters)
        system.database.connection.commit()
        assert system.is_empty_knowledge_bank() is False
        checks = system.preflight(
            require_api_key=False,
            initialize_harness=False,
            require_empty_bank=True,
        )
        assert checks["empty_bank"] is False
        assert checks["bank_protocol"] is False
        assert checks["passed"] is False


def test_fresh_bank_gate_rejects_unindexed_artifact_file(tmp_path: Path) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        assert system.is_empty_knowledge_bank() is True
        orphan = system.artifacts.root / "atomic" / "orphan.json"
        orphan.write_text("{}", encoding="utf-8")
        assert system.is_empty_knowledge_bank() is False


def test_tool_body_is_covered_by_immutable_artifact_verification(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_v3"
    with AtomicSkillGraphSystem(_system_config(data_dir)) as system:
        tool = ToolAsset(
            ToolRef("body_probe", "1.0.0"), "probe", {}, {}, "python",
            {"filename": "tool.py"}, [], {}, {}, {}, ToolStatus.DRAFT,
        )
        system.tools.register(tool, body="VALUE = 1\n")
        system.artifacts.verify_all()
        body_path = Path(system.tools.body_path(tool.ref) or "")
        body_path.write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="body hash mismatch"):
            system.artifacts.verify_all()
    with pytest.raises(ArtifactIntegrityError, match="body hash mismatch"):
        AtomicSkillGraphSystem(_system_config(data_dir))


def test_existing_incomplete_database_schema_fails_closed_at_startup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_v3"
    with AtomicSkillGraphSystem(_system_config(data_dir)) as system:
        system.database.execute("DROP TABLE graph_edges")
        system.database.connection.commit()
        checks = system.preflight(
            require_api_key=False,
            initialize_harness=False,
            require_empty_bank=False,
        )
        assert checks["database_schema"] is False
        assert checks["passed"] is False

    with pytest.raises(RuntimeError, match="missing required v3 tables"):
        AtomicSkillGraphSystem(_system_config(data_dir))


def test_formal_train_max_task_attempts_config_is_strict_positive_integer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "alfworld_train_full_30.yaml"
    output_dir = repo_root / "runs" / "alfworld_train_full_30"
    config = load_config(config_path)
    assert config["experiment"]["max_task_attempts"] == 3
    _validate_formal_config(config, output_dir)

    for invalid in (None, 0, -1, True, 1.5, "3"):
        changed = load_config(config_path)
        changed["experiment"]["max_task_attempts"] = invalid
        with pytest.raises(ProtocolError, match="max_task_attempts"):
            _validate_formal_config(changed, output_dir)

    changed = load_config(config_path)
    changed["trace_data_dir"] = str(output_dir / "different-trace-root")
    with pytest.raises(ProtocolError, match="trace_data_dir"):
        _validate_formal_config(changed, output_dir)


def test_formal_frozen_max_task_attempts_config_is_strict_positive_integer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "alfworld_frozen_eval.yaml"
    output_dir = repo_root / "runs" / "alfworld_frozen_eval_60"
    config = load_config(config_path)
    assert config["experiment"]["max_task_attempts"] == 3
    _validate_frozen_config(config, output_dir)

    for invalid in (None, 0, -1, True, 1.5, "3"):
        changed = load_config(config_path)
        changed["experiment"]["max_task_attempts"] = invalid
        with pytest.raises(ProtocolError, match="max_task_attempts"):
            _validate_frozen_config(changed, output_dir)

    changed = load_config(config_path)
    changed["trace_data_dir"] = str(output_dir / "different-trace-root")
    with pytest.raises(ProtocolError, match="trace_data_dir"):
        _validate_frozen_config(changed, output_dir)


def test_formal_usage_allows_true_zero_llm_and_rejects_partial_metering() -> None:
    zero_llm = {
        "trace_id": "zero",
        "agent_turns": [],
        "llm_usage": [],
        "metadata": {"usage_reconciliation": {"episode_total_tokens": 0}},
    }
    assert validate_formal_usage([zero_llm])["token_mismatch"] == 0

    partial = {
        "trace_id": "partial",
        "agent_turns": [{"turn_index": 0}],
        "llm_usage": [{
            "bucket": "planner_p1",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "call_count": 1,
            "latency_ms": 0,
            "provider_metadata": {"usage_status": "partial"},
        }],
    }
    with pytest.raises(ValueError, match="unavailable/partial"):
        validate_formal_usage([partial])


def test_provider_probe_fails_before_formal_attempt(tmp_path: Path) -> None:
    output_dir = tmp_path / "formal_output"
    attempt_root = output_dir / "attempt_history"
    with pytest.raises(ProviderCapabilityError, match="manifest is missing"):
        ensure_provider_capability(
            _config(),
            output_dir=output_dir,
            config_hash="config_fixture",
            code_hash="code_fixture",
            run_if_missing=False,
        )
    assert not attempt_root.exists()
    assert not list(output_dir.glob("attempt_*.json"))


def test_planner_provider_error_persists_failure_trace(tmp_path: Path) -> None:
    class AuditedFailingProvider:
        def __init__(self) -> None:
            self.records = []
            self.context = {"session_id": "", "stage": ""}

        @property
        def request_record_count(self):
            return len(self.records)

        def request_records_since(self, start_index):
            return tuple(dict(item) for item in self.records[start_index:])

        def set_request_context(self, *, session_id, stage):
            self.context = {"session_id": session_id, "stage": stage}

        def snapshot(self):
            return {
                "provider": "fixture",
                "model": "failing-provider",
                "request_record_count": len(self.records),
            }

        def complete(self, messages, *, tools=None):
            started = time.time()
            self.records.append({
                "request_id": "audit_req_planner_failure",
                "provider_request_id": "provider_req_planner_failure",
                **self.context,
                "started_at": started,
                "ended_at": time.time(),
                "outcome": "error",
                "http_status": 400,
                "retry_count": 0,
                "usage_status": "unavailable",
                "error_code": "provider_invalid_request",
                "sanitized_error": "HTTP 400 from LLM provider",
                "payload_fingerprint": "fixture-payload-sha256",
                "payload_field_names": ["max_tokens", "messages", "model"],
            })
            raise AgentProviderError(
                "provider_invalid_request",
                "HTTP 400 from LLM provider",
                http_status=400,
            )

    config = _system_config(tmp_path / "data_v3")
    config["trace_data_dir"] = str(tmp_path / "formal_trace_output")
    provider = AuditedFailingProvider()
    with AtomicSkillGraphSystem(
        config, harness=FakeHarness(), provider=provider,
    ) as system:
        # The formal empty-bank protocol intentionally skips Planner.  Seed an
        # unrelated online-usable Atomic so this test reaches the provider
        # failure boundary it is specifically intended to audit.
        system.skills.register_atomic(AbstractAtomicSkill(
            SkillRef("unrelated_active_atomic", "1.0.0"),
            "unrelated active atomic",
            [],
            [],
            [],
            [SemanticPredicate("unrelated.effect", {"value": "x"})],
            {},
            [],
            {},
            {},
            SkillStatus.ACTIVE,
        ))
        with pytest.raises(AgentProviderError, match="HTTP 400"):
            system.run_task(
                fake_task("planner-provider-failure", "apple_1"),
                attempt_id="attempt_planner_fixture",
            )
        payloads = list(system.traces.iter_payloads())

    assert len(payloads) == 1
    trace = payloads[0]
    assert trace["runtime_plan"]["failure_stage"] == "planner"
    assert trace["metadata"]["attempt_id"] == "attempt_planner_fixture"
    assert trace["metadata"]["failure"]["error_code"] == "provider_invalid_request"
    assert trace["infrastructure_failure"] is True
    assert trace["learning_eligible"] is False
    assert trace["ended_at"] >= trace["started_at"]
    assert trace["failures"][0]["attempt_id"] == "attempt_planner_fixture"
    assert trace["provider_requests"][0]["request_id"] == "provider_req_planner_failure"
    assert trace["provider_requests"][0]["http_status"] == 400
    assert trace["provider_requests"][0]["usage_status"] == "unavailable"
    assert trace["resource_usage_complete"] is False


def test_missing_composite_atomic_ref_persists_planner_failure_and_rethrows(
    tmp_path: Path,
) -> None:
    config = _system_config(tmp_path / "data_v3")
    config["trace_data_dir"] = str(tmp_path / "formal_trace_output")
    harness = FakeHarness()
    task = fake_task("planner-missing-atomic", "apple_1")
    missing_ref = SkillRef("missing_atomic", "1.0.0")

    with AtomicSkillGraphSystem(
        config, harness=harness, provider=object(),
    ) as system:
        system.skills.register_composite(CompositeSkill(
            SkillRef("broken_active_composite", "1.0.0"),
            "active Composite with a missing Atomic registry reference",
            [CompositeOccurrence("s1", "occ1", missing_ref, {})],
            ["s1"],
            [],
            [],
            harness.task_contract(task),
            {},
            {},
            {},
            {"harness_profiles": [harness.profile_name]},
            SkillStatus.ACTIVE,
        ))

        with pytest.raises(KeyError, match="missing_atomic@1.0.0"):
            system.run_task(
                task,
                attempt_id="attempt_missing_composite_atomic",
            )
        payloads = list(system.traces.iter_payloads())

    assert len(payloads) == 1
    trace = payloads[0]
    assert trace["runtime_plan"]["failure_stage"] == "planner"
    assert trace["metadata"]["failure"]["error_type"] == "KeyError"
    assert trace["metadata"]["attempt_id"] == "attempt_missing_composite_atomic"
    assert trace["infrastructure_failure"] is True
    assert trace["learning_eligible"] is False


def test_attempt_capture_does_not_mask_primary_error(tmp_path: Path) -> None:
    attempt = AttemptTraceRef(
        "attempt_primary_fixture",
        "run_fixture",
        "task_fixture",
        "signature_fixture",
        "task",
        1,
    )

    class BrokenAttemptLedger:
        def capture(self, _attempt, *, reason):
            raise RuntimeError(f"capture audit failed: {reason}")

    primary = ValueError("primary execution failure")
    reraised = None
    try:
        try:
            raise primary
        except Exception as caught:
            audit_errors = audit_failed_attempt(
                primary=caught,
                attempt=attempt,
                attempt_ledger=BrokenAttemptLedger(),
                receipt_root=tmp_path / "receipts",
                update_state=lambda: (_ for _ in ()).throw(
                    RuntimeError("state audit failed")
                ),
                capture_reason="primary_error",
            )
            raise
    except Exception as caught_again:
        reraised = caught_again

    assert reraised is primary
    assert [item["stage"] for item in audit_errors] == ["capture", "state"]
    receipt = json.loads(
        (tmp_path / "receipts" / "attempt_primary_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["primary_error_type"] == "ValueError"
    assert receipt["primary_error_message"] == "primary execution failure"
    assert [item["stage"] for item in receipt["audit_errors"]] == [
        "capture", "state",
    ]


def test_unavailable_retry_usage_aborts_before_evolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRY_USAGE_KEY", "fixture-key")

    class Response:
        def __init__(self, status, payload, text=""):
            self.status_code = status
            self.ok = 200 <= status < 300
            self._payload = payload
            self.text = text
            self.headers = {"x-request-id": f"req_{status}"}

        def json(self):
            return self._payload

    success = {
        "id": "response_success",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "ok", "reasoning_content": "private"},
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    responses = iter([
        Response(429, {}, "rate limited"),
        Response(200, success),
    ])
    monkeypatch.setattr(
        "atomic_skillgraph.agents.provider.requests.post",
        lambda *_args, **_kwargs: next(responses),
    )
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="RETRY_USAGE_KEY",
        max_completion_tokens=32768,
        max_retries=1,
        retry_backoff_seconds=0,
        max_retry_after_seconds=0,
    ))
    config = _system_config(tmp_path / "data_v3")
    config["trace_data_dir"] = str(tmp_path / "trace_output")
    with AtomicSkillGraphSystem(
        config, harness=FakeHarness(), provider=provider,
    ) as system:
        def provider_only_runtime(task, *, mode, trace_builder, attempt_id=""):
            provider.complete([{"role": "system", "content": "fixture"}])
            trace_builder.trace.benchmark_success = True
            trace_builder.trace.learning_eligible = True
            return trace_builder.finish()

        evolution_called = False

        def forbidden_evolution(*_args, **_kwargs):
            nonlocal evolution_called
            evolution_called = True
            raise AssertionError("evolution must not run with incomplete usage")

        system.orchestrator.run_task = provider_only_runtime
        system._prepare_evolution = forbidden_evolution
        with pytest.raises(AtomicSkillGraphError) as error:
            system.run_task(
                fake_task("incomplete-usage", "apple_1"),
                attempt_id="attempt_incomplete_usage",
            )
        payloads = list(system.traces.iter_payloads())

    assert getattr(error.value, "code", "") == "provider_usage_missing"
    assert evolution_called is False
    assert len(payloads) == 1
    assert payloads[0]["resource_usage_complete"] is False
    assert [item["usage_status"] for item in payloads[0]["provider_requests"]] == [
        "unavailable", "reported",
    ]


def test_real_smoke_dataflow_requires_started_downstream_consumption() -> None:
    trace = SimpleNamespace(
        runtime_plan={
            "occurrences": [
                {"step_id": "s1", "occurrence_id": "occ_source"},
                {"step_id": "s2", "occurrence_id": "occ_target"},
            ],
            "data_edges": [{
                "source_step": "s1",
                "target_step": "s2",
                "source_role": "held_object",
                "target_role": "object",
            }],
        },
        binding_changes=[
            {
                "occurrence_id": "occ_source",
                "role": "held_object",
                "reason": "validated_output_published",
                "current": {"source": "tool_output", "value": "apple_2"},
            },
            {
                "occurrence_id": "occ_target",
                "role": "object",
                "reason": "data_flow",
                "current": {"source": "data_flow", "value": "apple_2"},
            },
        ],
        implementation_invocations=[],
        graph_self_sufficient_success=True,
        task_rescue_required=False,
    )
    assert _validated_dataflow(trace) is False
    trace.implementation_invocations = [{
        "occurrence_id": "occ_target",
        "arguments": {"object": "apple_2"},
        "preflight": {"passed": True},
        "result": {
            "started": True,
            "completed": True,
            "atomic_effect_passed": True,
        },
    }]
    assert _validated_dataflow(trace) is True


def test_evolution_error_finalizes_original_skeleton_as_failure_trace(
    tmp_path: Path,
) -> None:
    config = _system_config(tmp_path / "data_v3")
    config["trace_data_dir"] = str(tmp_path / "trace_output")
    with AtomicSkillGraphSystem(
        config, harness=FakeHarness(), provider=object(),
    ) as system:
        def successful_runtime(task, *, mode, trace_builder, attempt_id=""):
            trace_builder.trace.runtime_plan = {
                "source": "full_dynamic", "failure_stage": "runtime",
            }
            trace_builder.trace.benchmark_success = True
            trace_builder.trace.learning_eligible = True
            return trace_builder.finish()

        system.orchestrator.run_task = successful_runtime
        system.extraction_policy.decide = lambda _trace: SimpleNamespace(
            should_extract=True, reasons=["fixture"],
        )
        system._prepare_evolution = lambda _trace, _task: SimpleNamespace(
            compiled=[], gap_diagnosis={}, source_composite_ref="",
        )
        system._apply_evolution = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("evolution admission failed")
        )
        with pytest.raises(RuntimeError, match="evolution admission failed"):
            system.run_task(
                fake_task("evolution-failure", "apple_1"),
                attempt_id="attempt_evolution_failure",
            )
        payloads = list(system.traces.iter_payloads())

    assert len(payloads) == 1
    trace = payloads[0]
    assert trace["metadata"]["attempt_id"] == "attempt_evolution_failure"
    assert trace["metadata"]["failure"]["error_type"] == "RuntimeError"
    assert trace["runtime_plan"]["failure_stage"] == "evolution"
    assert trace["infrastructure_failure"] is True
