"""R9.2 task-local automation trial boundary regressions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import atomic_skillgraph.runtime.automation as automation_module
from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
    ToolBinding,
)
from atomic_skillgraph.core.contracts import (
    ImplementationAtom,
    ParameterSpec,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
    ToolCallPreflightResult,
    ToolExecutionResult,
    ValidationResult,
)
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.runtime.automation import (
    RuntimeAutomationCoordinator,
    RuntimeAutomationOutcome,
)
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.implementation_runner import ImplementationRunner
from atomic_skillgraph.runtime.support_retriever import SupportCandidate
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.tooling.ir import ToolExecutionState
from atomic_skillgraph.tooling.proposal import (
    ToolProposal,
    runtime_automation_draft_from_dict,
)
from atomic_skillgraph.traces.schema import (
    TaskRecord,
    ToolExecutionRecord,
    TraceBuilder,
    TraceRecord,
)
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import (
    FakeAgentFactory,
    FakeHarness,
    FakeReply,
    ScriptedAgentProvider,
    fake_task,
)


def _fixtures():
    path = Path(__file__).with_name(
        "test_stored_composite_binding_authority.py"
    )
    spec = importlib.util.spec_from_file_location("_r92_trial_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(factory: FakeAgentFactory):
    fixtures = _fixtures()
    runtime, ctx, occurrence, invocations = fixtures._single_nav_context(
        fixtures._PickPlaceHarness(), factory,
    )
    schema_harness = FakeHarness()
    ctx.harness.semantic_predicate_schema = (
        schema_harness.semantic_predicate_schema
    )
    ctx.harness.primitive_action_schema = schema_harness.primitive_action_schema
    return runtime, ctx, occurrence, invocations


def _draft_payload(occurrence_id: str, *, draft_id: str = "r92-draft") -> dict:
    return {
        "draft_id": draft_id,
        "intent": "establish one bounded location witness",
        "inputs": [{"name": "target", "semantic_type": "entity"}],
        "outputs": [{"name": "object", "semantic_type": "entity"}],
        "preconditions": [],
        "effects": [{
            "predicate": "agent.holds",
            "args": {"object": "$object"},
            "effect_domain": "world",
        }],
        "rationale": "bounded task-local automation",
        "source_occurrence_id": occurrence_id,
        "input_binding_specs": {
            "target": {
                "kind": "constant",
                "value": "desk",
            },
        },
    }


def _frozen_config(data_dir: Path, trace_dir: Path, *, frozen: bool) -> dict:
    return {
        "schema_version": 3,
        "repair_revision": "R10.2.1",
        "method_patch": "3.2",
        "data_dir": str(data_dir),
        "trace_data_dir": str(trace_dir),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "deterministic-test",
            "api_key_env": "MODEL_API_KEY",
        },
        "planner": {
            "max_repeat_count": 4,
            "max_runtime_occurrences": 16,
            "cold_start_c1_repair_limit": 1,
        },
        "cold_start": {"enabled": not frozen},
        "experiment": {
            "benchmark": "alfworld",
            "condition": "full",
            "runtime_mode": "frozen" if frozen else "online",
            "freeze_skills": frozen,
            "allow_long_term_knowledge_writes": not frozen,
            "output_dir": str(trace_dir),
        },
    }


def _successful_trial_fixture(system: AtomicSkillGraphSystem, task_id: str):
    task = fake_task(task_id, "apple_1", requires_rescue=True)
    task.context["binding_types"] = {"item": "entity"}
    occurrence = RuntimeOccurrence(
        "parent_take",
        "parent_take",
        SkillRef("parent_take", "1.0.0"),
        [],
        {
            "item": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="item",
            ),
        },
        [],
        [SemanticPredicate("agent.holds", {"object": "$item"})],
    )
    contract = system.harness.task_contract(task)
    plan = RuntimeLinearPlan(
        task.task_id,
        "stored_composite",
        "skill://r92_parent@1.0.0",
        [occurrence],
        [occurrence.step_id],
        [],
        [],
        contract,
        {"final_outcome": "stored_composite"},
    )
    builder = system.orchestrator.create_trace_builder(task)
    ctx = TaskRuntimeContext.create(
        task,
        plan,
        system.harness,
        builder,
        RuntimeBudget(global_action_budget=4),
    )
    ctx.budget.begin_node(occurrence.occurrence_id)
    ctx.binding_store.resolve_occurrence_specs(
        occurrence,
        ctx.world_revision,
    )
    ctx.begin_occurrence(occurrence)
    draft = runtime_automation_draft_from_dict({
        "draft_id": "frozen-local-take",
        "intent": "take target for local use",
        "inputs": [{"name": "item", "semantic_type": "entity"}],
        "outputs": [{"name": "item", "semantic_type": "entity"}],
        "preconditions": [],
        "effects": [{
            "predicate": "agent.holds",
            "args": {"object": "$item"},
            "effect_domain": "world",
        }],
        "rationale": "bounded task-local Frozen trial",
        "source_occurrence_id": occurrence.occurrence_id,
        "input_binding_specs": {
            "item": {
                "kind": "current_occurrence_anchor",
                "source_role": "item",
            },
        },
    })
    atomic = RuntimeAutomationCoordinator._draft_atomic(
        draft,
        occurrence_id=occurrence.occurrence_id,
        trace_id=ctx.trace_builder.trace.trace_id,
    )
    parameter = {
        "name": "item",
        "semantic_type": "entity",
        "required": True,
        "runtime_resolvable": False,
        "required_resolution": "semantic",
        "description": "",
    }
    proposal = {
        "proposal_version": "2", "entry_contract": {"conditions": [], "grounding_constraints": []},
        "decision": "create",
        "summary": "take the target once",
        "atomic_ref": str(atomic.ref),
        "inputs": [parameter],
        "outputs": [dict(parameter)],
        "program": [
            {
                "node_id": "take",
                "op": "ACTION",
                "action_type": "TAKE",
                "argument_mapping": {
                    "item": {
                        "kind": "skill_input",
                        "source_role": "item",
                    },
                },
                "expected_effects": [{
                    "predicate": "agent.holds",
                    "args": {"object": "$item"},
                    "effect_domain": "world",
                }],
            },
            {
                "node_id": "return",
                "op": "RETURN",
                "output_sources": {
                    "item": {
                        "source": "tool_input",
                        "field": "item",
                    },
                },
            },
        ],
        "max_actions": 1,
        "final_effects": [{
            "predicate": "agent.holds",
            "args": {"object": "$item"},
            "effect_domain": "world",
        }],
        "evidence_outputs": [],
        "path_expectations": [],
        "rationale": "One bounded task-local action.",
    }
    return ctx, occurrence, draft, proposal


class _TrackingFactory(FakeAgentFactory):
    def __call__(self, first, second):
        session = super().__call__(first, second)
        session.submit_count = 0
        session.finalize_count = 0
        submit = session.submit_tool_result
        finalize = session.finalize_tool_result

        def tracked_submit(*args, **kwargs):
            session.submit_count += 1
            return submit(*args, **kwargs)

        def tracked_finalize(*args, **kwargs):
            session.finalize_count += 1
            return finalize(*args, **kwargs)

        session.submit_tool_result = tracked_submit
        session.finalize_tool_result = tracked_finalize
        return session


def _install_real_trial_coordinator(
    runtime,
    monkeypatch: pytest.MonkeyPatch,
    program: list[dict],
) -> dict[str, int]:
    """Keep the real coordinator/runners while scripting one candidate asset."""

    tool = ToolAsset(
        ToolRef("r92_runtime_trial", "1.0.0"),
        "execute the bounded task-local trial",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        {"entry_contract": {"conditions": [], "grounding_constraints": []},
            "output_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        "tool_ir_v1",
        {
            "schema_version": 1,
            "max_actions": 2,
            "program": list(program),
            # ToolValidator requires a declared final effect. The trial tests
            # exercise control/budget authority, not admission of this fixture.
            "final_effects": [{
                "predicate": "agent.at_location",
                "args": {"location": "desk"},
                "effect_domain": "world",
            }],
            "evidence_outputs": [],
        },
        [],
        {"reviewed": True, "zero_llm": True},
        {},
        {},
        ToolStatus.CANDIDATE,
    )
    proposal = ToolProposal(
        "1",
        "create",
        "bounded task-local trial",
        str(tool.ref),
        [],
        [],
        [{"node_id": "return", "op": "RETURN", "output_sources": {}}],
        1,
        [],
        [],
        [],
        "scripted candidate for the control-boundary regression",
    )
    calls = {"builder": 0, "compiler": 0}

    class Builder:
        def __init__(self, _session):
            pass

        def build(self, **_kwargs):
            calls["builder"] += 1
            return proposal

    class StaticValidator:
        def validate_automation_draft(self, *_args, **_kwargs):
            return ValidationResult.ok("runtime_automation_r0")

        def validate_proposal(self, *_args, **_kwargs):
            return ValidationResult.ok("tool_static")

    class Compiler:
        def compile_proposal(self, _occurrence, atomic, _proposal, _provenance):
            calls["compiler"] += 1
            implementation = ImplementationAtom(
                SkillRef("impl_r92_runtime_trial", "1.0.0"),
                atomic.ref,
                [ToolBinding(tool.ref, "primary", {})],
                [],
                {},
                {},
                {},
                SkillStatus.CANDIDATE,
            )
            return SimpleNamespace(
                atomic=atomic,
                tool=tool,
                implementation=implementation,
            )

    monkeypatch.setattr(automation_module, "ToolBuilderSession", Builder)
    runtime.node_executor.automation_coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=lambda *_args: object(),
        tool_compiler=Compiler(),
        implementation_runner=ImplementationRunner(ValidationEngine()),
        static_validator=StaticValidator(),
    )
    return calls


def test_draft_idempotency_conflict_and_safe_agent_projection() -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    calls = []

    def process_draft(**kwargs):
        calls.append(kwargs)
        return RuntimeAutomationOutcome(
            r0_passed=True,
            r0_report={"validator_snapshot": "trace-only"},
            proposal={"program": {"secret": "must-not-return"}},
            static_passed=True,
            static_report={"implementation_mapping": "trace-only"},
            trial={
                "draft_id": kwargs["draft"].draft_id,
                "tool_ref": "tool://task-local@1.0.0",
                "result": {"program": "private"},
                "r1_outputs": {"location": "desk_1"},
                "r1": {"outputs_valid": True},
                "terminal_interrupted": False,
            },
            r1_passed=True,
            r1_report={"private": "trace-only"},
            stage="r1_passed",
        )

    runtime.node_executor.automation_coordinator = SimpleNamespace(
        process_draft=process_draft,
    )
    arguments = _draft_payload(occurrence.occurrence_id)
    first = runtime.node_executor._process_runtime_automation_call(
        SimpleNamespace(arguments=arguments), ctx, occurrence,
    )
    duplicate = runtime.node_executor._process_runtime_automation_call(
        SimpleNamespace(arguments=dict(arguments)), ctx, occurrence,
    )
    conflicting = runtime.node_executor._process_runtime_automation_call(
        SimpleNamespace(arguments={**arguments, "intent": "different"}),
        ctx,
        occurrence,
    )

    assert len(calls) == 1
    assert duplicate["cache_hit"] is True
    assert conflicting["failure_code"] == "runtime_automation_draft_id_conflict"
    encoded = json.dumps(first, sort_keys=True).casefold()
    assert "program" not in encoded
    assert "implementation_mapping" not in encoded
    assert "validator_snapshot" not in encoded
    assert first["trial"] == {
        "draft_id": "r92-draft",
        "r1_outputs": {"location": "desk_1"},
        "r1": {"outputs_valid": True},
        "terminal_interrupted": False,
    }
    funnel = ctx.trace_builder.trace.metadata["runtime_automation_funnel"]
    assert funnel["duplicate_id_cache_hit_count"] == 1


def test_terminal_trial_finalizes_pending_call_without_parent_success() -> None:
    class TrackingFactory(FakeAgentFactory):
        def __call__(self, first, second):
            session = super().__call__(first, second)
            session.submit_count = 0
            session.finalize_count = 0
            submit = session.submit_tool_result
            finalize = session.finalize_tool_result

            def tracked_submit(*args, **kwargs):
                session.submit_count += 1
                return submit(*args, **kwargs)

            def tracked_finalize(*args, **kwargs):
                session.finalize_count += 1
                return finalize(*args, **kwargs)

            session.submit_tool_result = tracked_submit
            session.finalize_tool_result = tracked_finalize
            return session

    factory = TrackingFactory()
    runtime, ctx, occurrence, invocations = _context(factory)
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("request_runtime_automation", {"reason": "bounded search", "intended_capability": "locate"}),
        FakeReply.tool(
            "propose_runtime_automation_atomic",
            _draft_payload(occurrence.occurrence_id),
        ),
    ])

    def process_draft(**_kwargs):
        ctx.terminal_latched = True
        ctx.harness.validator_channel().done = True
        ctx.harness.validator_channel().won = True
        return RuntimeAutomationOutcome(
            r0_passed=True,
            static_passed=True,
            trial={
                "draft_id": "r92-draft",
                "r1_outputs": {},
                "r1": {"terminal_interrupted": True},
                "terminal_interrupted": True,
            },
            r1_passed=False,
            stage="r1_rejected",
        )

    runtime.node_executor.automation_coordinator = SimpleNamespace(
        process_draft=process_draft,
    )
    result = runtime.node_executor.run_agent_node(occurrence, ctx, mode='preparation')

    assert result.atomic_effect_passed is False
    assert result.failure_code == ""
    assert result.node_status.value == "terminal_partial"
    session = next(
        item for item in ctx.trace_builder.trace.agent_sessions
        if item.session_type == "RuntimePreparationSession"
    )
    assert session.snapshot["finalized"] is True
    assert session.snapshot["turn_count"] == 1
    assert session.snapshot["pending_tool_call"] is None
    runtime_session = factory.sessions_of("runtime_preparation")[0]
    assert runtime_session.finalize_count == 1
    assert runtime_session.submit_count == 0
    assert runtime_session.snapshot()["turn_count"] == 1
    factory.assert_exhausted()


def test_real_runtime_trial_won_stops_program_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _TrackingFactory()
    runtime, ctx, occurrence, invocations = _context(factory)
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("request_runtime_automation", {"reason": "bounded search", "intended_capability": "locate"}),
        FakeReply.tool(
            "propose_runtime_automation_atomic",
            _draft_payload(occurrence.occurrence_id),
        ),
    ])
    calls = _install_real_trial_coordinator(
        runtime,
        monkeypatch,
        [
            {
                "node_id": "winning_action",
                "op": "ACTION",
                "action_type": "LOOK",
                "argument_mapping": {},
                "expected_effects": [],
            },
            {
                "node_id": "forbidden_after_terminal",
                "op": "ACTION",
                "action_type": "LOOK",
                "argument_mapping": {},
                "expected_effects": [],
            },
            {"node_id": "forbidden_return", "op": "RETURN", "output_sources": {}},
        ],
    )
    execute_calls: list[str] = []
    execute_action = ctx.harness.execute_action

    def execute_and_win(action_id: str, revision: int):
        if execute_calls:
            raise AssertionError("environment action executed after benchmark won")
        execute_calls.append(action_id)
        result = execute_action(action_id, revision)
        result.done = True
        result.won = True
        channel = ctx.harness.validator_channel()
        channel.done = True
        channel.won = True
        return result

    monkeypatch.setattr(ctx.harness, "execute_action", execute_and_win)

    result = runtime.node_executor.run_agent_node(occurrence, ctx, mode='preparation')

    trial = ctx.runtime_tool_trials["r92-draft"]
    tool_result = trial["result"]["tool_results"][0]
    assert result.failure_code == ""
    assert result.node_status.value == "terminal_partial"
    assert result.started is False
    assert ctx.terminal_latched is True
    assert ctx.benchmark_terminal() is True
    assert tool_result["started"] is True
    assert tool_result["completed"] is False
    assert tool_result["terminal_interrupted"] is True
    assert tool_result["executed_step_count"] == 1
    assert tool_result["tool_path_evidence"]["executed_node_ids"] == [
        "winning_action",
    ]
    assert trial["r1"]["tool_completed"] is False
    assert trial["r1"]["admission_eligible"] is False
    assert len(execute_calls) == 1
    assert len(ctx.trace_builder.trace.environment_actions) == 1
    assert len(ctx.trace_builder.trace.tool_executions) == 1
    assert len(ctx.trace_builder.trace.implementation_invocations) == 1
    assert ctx.budget.used_global_actions == 1
    assert calls == {"builder": 1, "compiler": 1}
    funnel = ctx.trace_builder.trace.metadata["runtime_automation_funnel"]
    assert funnel["trial_started"] == 1
    assert funnel["trial_completed"] == 0
    assert funnel["trial_terminal_interrupted"] == 1
    runtime_session = factory.sessions_of("runtime_preparation")[0]
    assert runtime_session.snapshot()["turn_count"] == 1
    assert runtime_session.finalize_count == 1
    assert runtime_session.submit_count == 0
    factory.assert_exhausted()


@pytest.mark.parametrize(
    ("boundary", "failure_code"),
    [
        ("global", "episode_action_budget_exhausted"),
    ],
)
def test_real_runtime_trial_budget_exhaustion_preserves_usage_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    failure_code: str,
) -> None:
    factory = _TrackingFactory()
    runtime, ctx, occurrence, invocations = _context(factory)
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("request_runtime_automation", {"reason": "bounded search", "intended_capability": "locate"}),
        FakeReply.tool(
            "propose_runtime_automation_atomic",
            _draft_payload(occurrence.occurrence_id),
        ),
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    calls = _install_real_trial_coordinator(
        runtime,
        monkeypatch,
        [
            {
                "node_id": "last_budgeted_action",
                "op": "ACTION",
                "action_type": "LOOK",
                "argument_mapping": {},
                "expected_effects": [],
            },
            {
                "node_id": "over_budget_action",
                "op": "ACTION",
                "action_type": "LOOK",
                "argument_mapping": {},
                "expected_effects": [],
            },
            {"node_id": "unreached_return", "op": "RETURN", "output_sources": {}},
        ],
    )
    ctx.budget.used_global_actions = ctx.budget.global_action_budget - 1
    execute_calls: list[str] = []
    execute_action = ctx.harness.execute_action

    def tracked_execute(action_id: str, revision: int):
        execute_calls.append(action_id)
        return execute_action(action_id, revision)

    monkeypatch.setattr(ctx.harness, "execute_action", tracked_execute)

    from atomic_skillgraph.core.errors import BudgetExhausted
    with pytest.raises(BudgetExhausted) as raised:
        runtime.node_executor.run_agent_node(occurrence, ctx, mode='preparation')
    assert raised.value.code == failure_code
    trace = ctx.trace_builder.trace
    assert len(execute_calls) == len(trace.environment_actions) == 1
    assert len(trace.tool_executions) == len(trace.implementation_invocations) == 1
    assert trace.tool_executions[0].result["interrupted_by_budget"]
    assert trace.implementation_invocations[0].result["failure_code"] == failure_code
    assert calls == {"builder": 1, "compiler": 1}
    assert ctx.budget.used_global_actions == ctx.budget.global_action_budget
    assert ctx.budget.used_node_actions == 1
    assert len(factory.sessions_of("runtime_preparation")) == 2
    assert not ctx.runtime_tool_trials  # No full R1 report after the hard interruption.
    assert len(factory.usage_ledger.events) == 2


def test_frozen_successful_task_local_trial_is_trace_only_and_not_reusable(
    tmp_path: Path,
) -> None:
    source_data = tmp_path / "source" / "data_v3"
    snapshot_data = tmp_path / "frozen" / "data_v3"
    with AtomicSkillGraphSystem(
        _frozen_config(
            source_data,
            tmp_path / "source-traces",
            frozen=False,
        ),
        harness=FakeHarness(),
    ) as source:
        source.freeze(snapshot_data)

    provider = ScriptedAgentProvider(provider_id="r92-frozen-builder")
    trace_dir = tmp_path / "frozen-traces"
    with AtomicSkillGraphSystem(
        _frozen_config(snapshot_data, trace_dir, frozen=True),
        harness=FakeHarness(),
        provider={"tool_builder": provider},
    ) as frozen:
        ctx, occurrence, draft, proposal = _successful_trial_fixture(
            frozen,
            "frozen-trial-task",
        )
        provider.enqueue(FakeReply.tool("create_tool", proposal))
        frozen._current_task_id = ctx.task_id
        frozen._current_task_usage_start = len(frozen.usage.events)

        refs_before = {
            "atomic": frozen.skills.list_refs("atomic"),
            "composite": frozen.skills.list_refs("composite"),
            "implementation": frozen.skills.list_refs("implementation"),
            "tool": frozen.tools.list_refs(),
        }
        evidence_before = frozen.database.execute(
            "SELECT COUNT(*) AS count FROM evidence_events"
        ).fetchone()["count"]
        recommended_before = [
            tuple(row)
            for row in frozen.database.execute(
                "SELECT logical_id,artifact_ref FROM recommended_pointers "
                "ORDER BY logical_id"
            ).fetchall()
        ]
        digest_before = frozen.knowledge_digest()

        outcome = frozen.orchestrator.node_executor.automation_coordinator.process_draft(
            draft=draft,
            ctx=ctx,
            occurrence=occurrence,
        )
        frozen.orchestrator._persist_v32_task_local_assets(ctx)
        trace = ctx.trace_builder.finish()
        trace_path = frozen.traces.save_atomic(trace)

        assert outcome.r1_passed is True, outcome.r1_report
        assert outcome.trial is not None
        assert outcome.trial["r1"]["admission_eligible"] is True
        assert outcome.trial["r1"]["tool_completed"] is True
        assert outcome.trial["terminal_interrupted"] is False
        assert trace_path.is_file()
        persisted = frozen.traces.load_payload(trace.trace_id)
        assert "frozen-local-take" in persisted["metadata"][
            "runtime_tool_trials"
        ]

        refs_after = {
            "atomic": frozen.skills.list_refs("atomic"),
            "composite": frozen.skills.list_refs("composite"),
            "implementation": frozen.skills.list_refs("implementation"),
            "tool": frozen.tools.list_refs(),
        }
        assert refs_after == refs_before
        assert frozen.database.execute(
            "SELECT COUNT(*) AS count FROM evidence_events"
        ).fetchone()["count"] == evidence_before
        assert [
            tuple(row)
            for row in frozen.database.execute(
                "SELECT logical_id,artifact_ref FROM recommended_pointers "
                "ORDER BY logical_id"
            ).fetchall()
        ] == recommended_before
        assert frozen.knowledge_digest() == digest_before

        trial_tool_ref = str(outcome.trial["tool_ref"])
        assert trial_tool_ref not in {
            str(ref) for ref in frozen.tools.list_refs()
        }
        next_ctx, _next_occurrence, _next_draft, _next_proposal = (
            _successful_trial_fixture(frozen, "next-frozen-task")
        )
        assert next_ctx.runtime_tool_trials == {}
        assert next_ctx.runtime_automation_drafts == {}
        assert trial_tool_ref not in {
            str(ref) for ref in frozen.tools.list_refs()
        }
        assert len(provider.requests) == 1


def test_loop_blocked_preparation_keeps_unchanged_support_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeAgentFactory()
    runtime, ctx, occurrence, invocations = _context(factory)
    action_id = ctx.action_catalog[0].action_id
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("environment_action", {
            "action_id": action_id,
            "intent": "attempt_current_atomic",
        }),
        FakeReply.tool(
            "report_runtime_status", {"status": "cannot_resolve"},
        ),
    ])
    candidate = SupportCandidate(
        atomic_ref="skill://r92_support@1.0.0",
        score=1.0,
        supplied_roles=(),
        output_roles=(),
        effect_predicates=(),
        diagnostics=(),
        execution_available=True,
    )
    executor = runtime.node_executor
    original_node_tools = executor._node_tools
    projected_candidates: list[tuple] = []

    def recording_node_tools(*args, **kwargs):
        projected_candidates.append(tuple(kwargs.get("support_candidates", ())))
        return original_node_tools(*args, **kwargs)

    monkeypatch.setattr(
        executor,
        "_retrieve_runtime_support_candidates",
        lambda **_kwargs: [candidate],
    )
    monkeypatch.setattr(executor, "_node_tools", recording_node_tools)
    monkeypatch.setattr(
        ctx.binding_store,
        "preflight_repeat_bindings",
        lambda *_args, **_kwargs: ValidationResult.fail(
            "repeat_preflight",
            "runtime_repetition_distinctness_violation",
            "the proposed Repeat identity was already consumed",
        ),
    )

    result = executor.run_agent_node(occurrence, ctx, mode='preparation')

    assert result.failure_code == "runtime_binding_unresolved"
    assert projected_candidates[0] == (candidate,)
    assert projected_candidates[1] == (candidate,)
    factory.assert_exhausted()


def test_builder_infrastructure_failure_propagates() -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = runtime_automation_draft_from_dict(
        _draft_payload(occurrence.occurrence_id)
    )

    def unavailable_builder(*_args, **_kwargs):
        raise ConnectionError("provider transport failed")

    coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=unavailable_builder,
        tool_compiler=SimpleNamespace(),
        implementation_runner=SimpleNamespace(),
    )
    with pytest.raises(ConnectionError, match="provider transport failed"):
        coordinator.process_draft(
            draft=draft, ctx=ctx, occurrence=occurrence,
        )


def test_runtime_builder_does_not_read_validator_only_semantic_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = runtime_automation_draft_from_dict(
        _draft_payload(occurrence.occurrence_id)
    )
    captured: dict = {}

    def forbidden_snapshot():
        raise AssertionError("validator-only snapshot crossed Agent boundary")

    class SafeNoToolBuilder:
        def __init__(self, _session):
            pass

        def build(self, **kwargs):
            captured.update(kwargs)
            return ToolProposal.no_tool(
                atomic_ref=str(kwargs["atomic"].ref),
                reason_code="public_interface_insufficient",
            )

    ctx.tool_evidence_snapshot = forbidden_snapshot
    monkeypatch.setattr(
        automation_module, "ToolBuilderSession", SafeNoToolBuilder,
    )
    coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=lambda *_args: object(),
        tool_compiler=SimpleNamespace(),
        implementation_runner=SimpleNamespace(),
    )

    outcome = coordinator.process_draft(
        draft=draft, ctx=ctx, occurrence=occurrence,
    )

    assert outcome.r0_passed is True
    assert outcome.failure_code == "runtime_automation_no_tool"
    assert outcome.static_passed is False
    assert captured["semantic_delta"] == {}
    assert captured["harness_interface"]["predicate_vocabulary"]
    assert captured["harness_interface"]["primitive_actions"]
    collection_sources = captured["harness_interface"][
        "tool_ir_collection_sources"
    ]
    assert [item["source"] for item in collection_sources] == [
        "action_catalog"
    ]
    selector = collection_sources[0]
    assert selector["entry_fields"] == [
        "action_id", "revision", "action_type", "arguments",
    ]
    assert selector["where"]["semantic_compatible_with"]["source"] == [
        "tool_input", "local_variable", "action_catalog",
        "semantic_evidence", "binding_evidence",
    ]
    assert selector["projection"]["project"]["kind"] == [
        "field", "argument",
    ]


def test_runtime_trial_inputs_are_isolated_and_repeat_credit_is_excluded() -> None:
    captured: dict = {}

    class AtomicValidation:
        def validate_execution_result(self, _atomic, _occurrence, bindings, *_args, **_kwargs):
            captured["bindings"] = dict(bindings)
            return ValidationResult.ok("atomic", effect=True)

    validation = SimpleNamespace(
        tool=SimpleNamespace(),
        atomic=AtomicValidation(),
    )
    runner = ImplementationRunner(validation)
    trace = TraceRecord.create(
        TaskRecord(
            "runtime-trial-task",
            "fake",
            "exercise a task-local automation",
            "generic",
            "runtime-trial-signature",
            {},
        ),
        {},
        {},
        {},
    )

    commit_calls = []
    parent_binding = RuntimeBinding(
        "parent_only", "parent-value", "entity", BindingSource.TASK,
        BindingStatus.GROUNDED, BindingResolution.CONCRETE,
    )
    trial_binding = RuntimeBinding(
        "helper_target", "trial-value", "entity", BindingSource.TASK,
        BindingStatus.GROUNDED, BindingResolution.CONCRETE,
    )
    binding_store = SimpleNamespace(
        snapshot_for_node=lambda _occurrence: {
            "parent_only": parent_binding,
            "helper_target": RuntimeBinding(
                "helper_target", "wrong-parent-value", "entity",
                BindingSource.TASK, BindingStatus.GROUNDED,
                BindingResolution.CONCRETE,
            ),
        },
        commit_repeat_bindings=lambda *args, **kwargs: (
            commit_calls.append((args, kwargs))
            or ValidationResult.ok("repeat", committed=True)
        ),
    )
    ctx = SimpleNamespace(
        trace_builder=TraceBuilder(trace),
        execution_terminal=lambda: False,
        binding_store=binding_store,
        harness=SimpleNamespace(
            validator_channel=lambda: SimpleNamespace(),
        ),
        world_revision=1,
    )
    tool = SimpleNamespace(ref="tool://trial@1.0.0", signature={
        "properties": {"target": {"type": "string"}}, "required": ["target"]})
    tool_binding = SimpleNamespace(
        tool_ref=tool.ref,
        role="helper",
        order=0,
        parameter_mapping={
            "target": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="helper_target",
            ),
        },
    )
    compiled = SimpleNamespace(
        atomic=SimpleNamespace(
            ref="skill://trial_atomic@1.0.0",
            inputs=[SimpleNamespace(name="helper_target", required=True)],
            outputs=[],
            validator_spec={},
        ),
        implementation=SimpleNamespace(
            ref="impl://trial@1.0.0",
            tool_bindings=[tool_binding],
            execution_policy={},
        ),
        tools=[tool],
    )

    def run_tool(_tool, arguments, _ctx, **kwargs):
        captured["tool_arguments"] = dict(arguments)
        captured["execution_scope"] = kwargs["execution_scope"]
        tool_span = _ctx.trace_builder.start_span(
            "tool",
            "parent_occurrence",
            parent_span_id=kwargs["parent_span_id"],
        )
        trace.tool_executions.append(ToolExecutionRecord(
            "trial_tool_attempt", "parent_occurrence", str(tool.ref), {},
            tool_span.span_id,
        ))
        _ctx.trace_builder.finish_span(tool_span.span_id)
        return ToolExecutionResult(
            str(tool.ref), True, True, True, True, 1, None, [], {}, 0, 1,
        )

    runner.tool_runner = SimpleNamespace(run=run_tool)
    preflight = ToolCallPreflightResult(
        True,
        str(compiled.implementation.ref),
        normalized_arguments={"helper_target": "trial-value"},
        binding_updates=[trial_binding],
    )
    result = runner.run(
        compiled,
        preflight,
        SimpleNamespace(
            occurrence_id="parent_occurrence", step_id="repeat_parent",
        ),
        ctx,
        agent_prepared=False,
        execution_scope="runtime_trial",
    )

    assert result.atomic_effect_passed is True
    assert captured["tool_arguments"] == {"target": "trial-value"}
    assert captured["execution_scope"] == "runtime_trial"
    assert set(captured["bindings"]) == {"helper_target"}
    assert captured["bindings"]["helper_target"].value == "trial-value"
    assert commit_calls == []
    exclusions = trace.metadata["runtime_trial_credit_exclusions"]
    assert len(exclusions["implementation_attempt_ids"]) == 1
    assert exclusions["tool_execution_ids"] == ["trial_tool_attempt"]
    implementation_span = next(
        item for item in trace.runtime_spans if item.kind == "implementation"
    )
    tool_span = next(item for item in trace.runtime_spans if item.kind == "tool")
    assert tool_span.parent_span_id == implementation_span.span_id
    assert trace.tool_executions[0].span_id == tool_span.span_id


def test_unknown_implementation_execution_scope_fails_closed() -> None:
    runner = ImplementationRunner(SimpleNamespace(tool=SimpleNamespace()))
    with pytest.raises(ValueError, match="unsupported Implementation execution_scope"):
        runner.run(
            SimpleNamespace(),
            ToolCallPreflightResult(True, "impl"),
            SimpleNamespace(),
            SimpleNamespace(),
            agent_prepared=False,
            execution_scope="unknown",
        )


def _runner_boundary_context(ctx):
    ctx.global_action_budget = ctx.budget.global_action_budget
    return ctx


def _runner_boundary_tool(program: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        ref="tool://r92_runtime_boundary@1.0.0",
        artifact={
            "program": program,
            "max_actions": 1,
            "final_effects": [],
        },
        interface={"output_schema": {"properties": {}}},
    )


def test_runtime_trial_programming_type_error_reaches_attempt_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    runner = ToolRunner(SimpleNamespace(
        validate_output=lambda *_args: ValidationResult.ok("tool_output"),
    ))
    tool = _runner_boundary_tool([])
    span = ctx.trace_builder.start_span("tool", occurrence.occurrence_id)

    def programming_defect(*_args, **_kwargs):
        raise TypeError("unexpected interpreter programming defect")

    monkeypatch.setattr(runner, "_execute_ir_nodes", programming_defect)

    with pytest.raises(TypeError, match="programming defect"):
        runner._run_ir_v1(
            tool,
            {},
            _runner_boundary_context(ctx),
            occurrence_id=occurrence.occurrence_id,
            span_id=span.span_id,
            execution_scope="runtime_trial",
        )

    assert ctx.trace_builder.trace.tool_executions == []
    assert next(
        item for item in ctx.trace_builder.trace.runtime_spans
        if item.span_id == span.span_id
    ).action_end >= 0


def test_runtime_trial_catalog_miss_is_a_typed_content_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    runner = ToolRunner(SimpleNamespace(
        validate_output=lambda *_args: ValidationResult.ok("tool_output"),
    ))
    tool = _runner_boundary_tool([{
        "node_id": "missing_action",
        "op": "ACTION",
        "action_type": "SEARCH",
        "argument_mapping": {},
    }])
    span = ctx.trace_builder.start_span("tool", occurrence.occurrence_id)
    before_actions = ctx.budget.used_global_actions

    monkeypatch.setattr(
        ctx.harness,
        "compile_primitive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyError("no current SEARCH affordance")
        ),
    )

    result = runner._run_ir_v1(
        tool,
        {},
        _runner_boundary_context(ctx),
        occurrence_id=occurrence.occurrence_id,
        span_id=span.span_id,
        execution_scope="runtime_trial",
    )

    assert result.failure_code == "tool_ir_action_unavailable"
    assert result.intrinsic_failure is True
    assert result.executed_step_count == 0
    assert ctx.budget.used_global_actions == before_actions


def test_runtime_trial_unexpected_action_key_error_reaches_attempt_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    runner = ToolRunner(SimpleNamespace(
        validate_output=lambda *_args: ValidationResult.ok("tool_output"),
    ))
    tool = _runner_boundary_tool([{
        "node_id": "programming_defect",
        "op": "ACTION",
        "action_type": "SEARCH",
        "argument_mapping": {},
    }])
    span = ctx.trace_builder.start_span("tool", occurrence.occurrence_id)

    def programming_defect(*_args, **_kwargs):
        raise KeyError("unexpected action execution defect")

    monkeypatch.setattr(runner, "_record_ir_action", programming_defect)

    with pytest.raises(KeyError, match="action execution defect"):
        runner._run_ir_v1(
            tool,
            {},
            _runner_boundary_context(ctx),
            occurrence_id=occurrence.occurrence_id,
            span_id=span.span_id,
            execution_scope="runtime_trial",
        )

    assert ctx.trace_builder.trace.tool_executions == []


def test_runtime_trial_predicate_schema_error_reaches_attempt_boundary() -> None:
    harness = SimpleNamespace(
        semantic_predicate_schema=lambda: (_ for _ in ()).throw(
            TypeError("unexpected predicate schema defect")
        ),
    )

    with pytest.raises(TypeError, match="predicate schema defect"):
        ToolRunner._semantic_facts_with_domains([], harness)


def test_runtime_trial_step_validator_error_reaches_attempt_boundary() -> None:
    validator_channel = SimpleNamespace(
        validate_atomic_effect=lambda _payload: (_ for _ in ()).throw(
            AttributeError("unexpected step validator defect")
        ),
    )
    ctx = SimpleNamespace(
        harness=SimpleNamespace(
            validator_channel=lambda: validator_channel,
        ),
    )
    state = ToolExecutionState(bindings={"object": "apple_1"})
    tool = SimpleNamespace(
        signature={"properties": {"object": {}}},
        interface={"output_schema": {"properties": {}}},
    )
    node = {
        "node_id": "take",
        "expected_effects": [{
            "predicate": "agent.holds",
            "args": {"object": "$object"},
            "effect_domain": "world",
        }],
    }

    with pytest.raises(AttributeError, match="step validator defect"):
        ToolRunner(SimpleNamespace())._validate_step_effects(
            node, ctx, state, tool=tool,
        )
