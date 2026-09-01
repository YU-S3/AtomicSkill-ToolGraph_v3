from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.fakes import FakeAgentFactory, FakeReply
from atomic_skillgraph.core.bindings import (
    BindingExpression, BindingExprKind,
    BindingResolution, BindingSource, BindingStatus, RuntimeBinding,
)
from atomic_skillgraph.core.contracts import (
    ColdStartCandidateSource, ColdStartExecutionMode, ColdStartPlanProposal,
    ColdStartPlanStep, ParameterSpec, SemanticPredicate,
)
from atomic_skillgraph.core.results import (
    ToolCallPreflightResult, ValidationResult,
)
from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
from atomic_skillgraph.harness.protocol import HarnessActionSpec


def _binding_fixtures():
    """Load the established real Runtime/Harness fixture without copying it."""

    path = Path(__file__).with_name("test_stored_composite_binding_authority.py")
    spec = importlib.util.spec_from_file_location("_r21_binding_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(factory: FakeAgentFactory):
    fixtures = _binding_fixtures()
    return fixtures._single_nav_context(
        fixtures._PickPlaceHarness(), factory,
    )


def test_node_environment_action_requires_intent_but_dynamic_does_not() -> None:
    runtime, ctx, occurrence, _ = _context(FakeAgentFactory())
    atomic = runtime.invocation_compiler.skills.get_atomic(occurrence.node_ref)

    node_tool = runtime.node_executor._node_tools(ctx, atomic)[0]
    dynamic_tool = runtime.node_executor._environment_tool(
        ctx, node_level=False,
    )

    assert node_tool.input_schema["required"] == ["action_id", "intent"]
    assert node_tool.input_schema["properties"]["intent"]["enum"] == [
        "explore", "attempt_current_atomic",
    ]
    assert dynamic_tool.input_schema["required"] == ["action_id"]
    assert "intent" not in dynamic_tool.input_schema["properties"]


def test_direct_autonomous_rejects_runtime_resolvable_semantic_choice(
    monkeypatch,
) -> None:
    runtime, ctx, occurrence, invocations = _context(FakeAgentFactory())
    ctx.binding_store._set(
        occurrence.occurrence_id,
        RuntimeBinding(
            "destination", "drawer", "entity", BindingSource.TASK,
            BindingStatus.GROUNDED, BindingResolution.SEMANTIC,
            ["task:semantic-destination"], ctx.world_revision,
        ),
        "test_semantic_choice",
    )

    calls = 0

    def reject_semantic_choice(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return ToolCallPreflightResult(
            True,
            str(invocations[0].implementation.ref),
            normalized_arguments={"destination": "drawer"},
            matched_evidence_refs=["current_context:destination"],
        )

    monkeypatch.setattr(
        runtime.invocation_compiler,
        "autonomous_preflight",
        reject_semantic_choice,
    )

    assert runtime.node_executor.try_autonomous(
        occurrence, invocations, ctx,
    ) is None
    assert calls == 1


def test_direct_autonomous_accepts_validator_backed_concrete_binding(
    monkeypatch,
) -> None:
    runtime, ctx, occurrence, invocations = _context(FakeAgentFactory())
    ctx.binding_store._set(
        occurrence.occurrence_id,
        RuntimeBinding(
            "destination", "drawer_2", "entity",
            BindingSource.HARNESS_EVIDENCE, BindingStatus.GROUNDED,
            BindingResolution.CONCRETE, ["validator:drawer_2"],
            ctx.world_revision,
        ),
        "test_concrete_binding",
    )
    preflight = ToolCallPreflightResult(
        True, str(invocations[0].implementation.ref),
    )
    sentinel = object()
    monkeypatch.setattr(
        runtime.invocation_compiler,
        "autonomous_preflight",
        lambda *_args, **_kwargs: preflight,
    )
    monkeypatch.setattr(
        runtime.node_executor.implementation_runner,
        "run",
        lambda *_args, **_kwargs: sentinel,
    )

    assert runtime.node_executor.try_autonomous(
        occurrence, invocations, ctx,
    ) is sentinel


def test_direct_autonomous_accepts_semantic_value_only_after_role_constraint_certifies_it(
    monkeypatch,
) -> None:
    runtime, ctx, occurrence, invocations = _context(FakeAgentFactory())
    # The value is exact and publicly present in the current GO_TO catalog,
    # but the formal Task-style binding is still only SEMANTIC before
    # execution-context validation.
    ctx.binding_store._set(
        occurrence.occurrence_id,
        RuntimeBinding(
            "destination", "drawer_2", "entity", BindingSource.TASK,
            BindingStatus.GROUNDED, BindingResolution.SEMANTIC,
            ["task:drawer"], ctx.world_revision,
        ),
        "test_role_specific_constraint",
    )
    sentinel = object()
    monkeypatch.setattr(
        runtime.node_executor.implementation_runner,
        "run",
        lambda *_args, **_kwargs: sentinel,
    )

    assert runtime.node_executor.try_autonomous(
        occurrence, invocations, ctx,
    ) is sentinel
    certified = ctx.binding_store.snapshot_for_node(occurrence)["destination"]
    assert certified.resolution is BindingResolution.CONCRETE
    assert certified.source is BindingSource.HARNESS_EVIDENCE
    assert any(
        ref.startswith("affordance:") for ref in certified.evidence_refs
    )


def test_exploratory_navigation_does_not_commit_atomic_output() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001", "intent": "explore"},
        ),
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.atomic_effect_passed is False
    assert result.failure_code == "runtime_binding_unresolved"
    assert ctx.harness.validator_channel().snapshot()["facts"]
    assert ctx.validated_outputs == {}
    assert not any(
        item.level == "atomic" and item.result.get("passed")
        for item in ctx.trace_builder.trace.validations
    )
    factory.assert_exhausted()


def test_attempt_navigation_commits_only_after_effect_validation() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {
                "action_id": "r000_a001",
                "intent": "attempt_current_atomic",
            },
        ),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.atomic_effect_passed is True
    assert result.validated_outputs == {"reached_location": "drawer_2"}
    assert any(
        item.level == "atomic" and item.result.get("passed")
        for item in ctx.trace_builder.trace.validations
    )
    factory.assert_exhausted()


def test_validate_current_atomic_retroactively_commits_current_witness() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001", "intent": "explore"},
        ),
        FakeReply.tool(
            "validate_current_atomic",
            {"candidate_bindings": {"destination": "drawer_2"}},
        ),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.atomic_effect_passed is True
    assert result.validated_outputs == {"reached_location": "drawer_2"}
    validation_call = next(
        item for item in ctx.trace_builder.trace.native_tool_calls
        if item.tool_name == "validate_current_atomic"
    )
    assert validation_call.preflight_result["committed"] is True
    factory.assert_exhausted()


def test_validate_current_atomic_cannot_invent_binding() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001", "intent": "explore"},
        ),
        FakeReply.tool(
            "validate_current_atomic",
            {"candidate_bindings": {"destination": "cabinet_9"}},
        ),
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.atomic_effect_passed is False
    validation_call = next(
        item for item in ctx.trace_builder.trace.native_tool_calls
        if item.tool_name == "validate_current_atomic"
    )
    assert validation_call.preflight_result["committed"] is False
    assert (
        validation_call.preflight_result["validation"]["failure_code"]
        == "atomic_effect_violation"
    )
    assert ctx.validated_outputs == {}
    factory.assert_exhausted()


def test_validate_current_atomic_cannot_override_dataflow_anchor() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001", "intent": "explore"},
        ),
        FakeReply.tool(
            "validate_current_atomic",
            {"candidate_bindings": {"destination": "drawer_2"}},
        ),
        FakeReply.tool("report_runtime_status", {"status": "plan_conflict"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)
    ctx.binding_store._set(
        occurrence.occurrence_id,
        RuntimeBinding(
            "destination", "cabinet_1", "entity", BindingSource.DATA_FLOW,
            BindingStatus.GROUNDED, BindingResolution.CONCRETE,
            ["edge:hard-destination"], ctx.world_revision,
        ),
        "test_dataflow_anchor",
    )

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.failure_code == "runtime_plan_conflict"
    validation_call = next(
        item for item in ctx.trace_builder.trace.native_tool_calls
        if item.tool_name == "validate_current_atomic"
    )
    assert validation_call.preflight_result["committed"] is False
    assert ctx.binding_store.snapshot_for_node(occurrence)[
        "destination"
    ].value == "cabinet_1"
    factory.assert_exhausted()


def test_exploration_does_not_consume_repeat_preflight(monkeypatch) -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {"action_id": "r000_a001", "intent": "explore"},
        ),
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)
    calls = 0

    def reject_if_called(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("exploration must not run Repeat preflight")

    monkeypatch.setattr(
        ctx.binding_store, "preflight_repeat_bindings", reject_if_called,
    )

    runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert calls == 0
    factory.assert_exhausted()


def test_repeat_preflight_rejection_is_typed_and_keeps_preparation_session(
    monkeypatch,
) -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "environment_action",
            {
                "action_id": "r000_a001",
                "intent": "attempt_current_atomic",
            },
        ),
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)
    monkeypatch.setattr(
        ctx.binding_store,
        "preflight_repeat_bindings",
        lambda *_args, **_kwargs: ValidationResult.fail(
            "repeat",
            "runtime_repetition_distinctness_violation",
            "choose a different identity",
        ),
    )

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.failure_code == "runtime_binding_unresolved"
    assert len(ctx.trace_builder.trace.environment_actions) == 0
    repeat_call = next(
        item for item in ctx.trace_builder.trace.native_tool_calls
        if item.tool_name == "environment_action"
    )
    assert repeat_call.preflight_result["passed"] is False
    sessions = [
        item for item in ctx.trace_builder.trace.agent_sessions
        if item.session_type == "RuntimePreparationSession"
    ]
    assert len(sessions) == 1
    assert len(sessions[0].snapshot["messages"]) >= 4
    factory.assert_exhausted()


def test_status_surface_and_node_prompt_define_three_distinct_meanings() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    status_tool = runtime.node_executor._status_tool(
        allow_plan_conflict=True,
    )
    assert "evidence is insufficient or search is incomplete" in status_tool.description
    assert "same rigid graph cannot solve the task" in status_tool.description
    assert "without asserting a formal plan conflict" in status_tool.description
    session = next(
        item for item in ctx.trace_builder.trace.agent_sessions
        if item.session_type == "RuntimePreparationSession"
    )
    prompt = next(
        item["content"] for item in session.snapshot["messages"]
        if item["role"] == "user"
    )
    assert "Use cannot_resolve only" in prompt
    assert "Use plan_conflict only" in prompt
    assert "Use give_up" in prompt
    factory.assert_exhausted()


def test_cold_scaffold_plan_is_used_for_downstream_data_edge_prompt() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)
    cold_execution_plan = SimpleNamespace(source="cold_execution_plan")
    seen: list[object] = []

    def build(plan, step_id, binding_store):
        seen.append(plan)
        assert step_id == occurrence.step_id
        assert binding_store is ctx.binding_store
        return SimpleNamespace(policy_view=lambda: {
            "current_step": occurrence.step_id,
            "output_obligations": [{
                "edge_id": "cold-edge",
                "producer_output_role": "reached_location",
                "consumer_step": "take",
                "consumer_input_role": "source",
            }],
            "remaining_method_outline": [],
        })

    runtime.node_executor.plan_context_builder = SimpleNamespace(build=build)

    runtime.node_executor.run_preparation_session(
        occurrence,
        invocations,
        ctx,
        plan_context_plan=cold_execution_plan,
    )

    assert seen == [cold_execution_plan]
    session = next(
        item for item in ctx.trace_builder.trace.agent_sessions
        if item.session_type == "RuntimePreparationSession"
    )
    prompt = next(
        item["content"] for item in session.snapshot["messages"]
        if item["role"] == "user"
    )
    policy = json.loads(prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1])
    obligations = policy["downstream_plan_context"]["output_obligations"]
    assert obligations[0]["edge_id"] == "cold-edge"
    assert obligations[0]["consumer_input_role"] == "source"
    factory.assert_exhausted()


def test_multi_effect_preferred_claim_is_checked_on_joint_assignment() -> None:
    channel = AlfWorldValidatorChannel()
    channel.record(
        HarnessActionSpec(
            "go", 0, "GO_TO", {"destination": "drawer_2"}, "go", {}, {},
        ),
        accepted=True, revision=1, done=False, won=False,
    )
    channel.record(
        HarnessActionSpec(
            "open", 1, "OPEN", {"object": "cabinet_1"}, "open", {}, {},
        ),
        accepted=True, revision=2, done=False, won=False,
    )
    destination = BindingExpression(
        BindingExprKind.SKILL_INPUT, source_role="destination",
    )
    container = BindingExpression(
        BindingExprKind.SKILL_INPUT, source_role="container",
    )

    result = channel.resolve_atomic_effect({
        "effects": [
            SemanticPredicate(
                "agent.at_location", {"location": destination},
            ),
            SemanticPredicate(
                "container.open", {"container": container},
            ),
        ],
        "known_bindings": {"container": "cabinet_1"},
        "semantic_anchors": {},
        "preferred_values": [],
        "preferred_bindings": {"destination": "drawer_2"},
        "input_specs": [
            ParameterSpec("destination", "entity"),
            ParameterSpec("container", "entity"),
        ],
        "output_specs": [],
        "output_identity": [],
        "current_revision": 2,
    })

    assert result.passed is True
    assert result.resolved_bindings == {
        "container": "cabinet_1", "destination": "drawer_2",
    }


def test_cold_verified_plan_conflict_skips_same_occurrence_seeded(
    monkeypatch,
) -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    monkeypatch.setattr(
        runtime.node_executor,
        "_complete_from_current_effect",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "try_autonomous",
        lambda *_args, **_kwargs: None,
    )

    def declare_conflict(*_args, **_kwargs):
        result = runtime.node_executor.not_started(
            occurrence, failure_code="runtime_plan_conflict",
        )
        result.failure_layer = "composite"
        return result

    monkeypatch.setattr(
        runtime.node_executor, "run_preparation_session", declare_conflict,
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "run_seeded_fresh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plan conflict must skip same-occurrence Seeded"),
        ),
    )

    passed, failure_code, outcome = runtime._run_verified_cold_step(
        ctx, ctx.plan, occurrence,
    )

    assert passed is False
    assert failure_code == "runtime_plan_conflict"
    assert outcome == "plan_conflict"
    assert ctx.plan_conflict_declared is True
    assert ctx.rescue_allowed() is True
    assert ctx.trace_builder.trace.node_records[-1].failure == {
        "failure_layer": "composite",
        "failure_code": "runtime_plan_conflict",
        "direct_started": False,
    }


def test_cold_plan_conflict_enters_task_rescue_not_cold_continuation(
    monkeypatch,
) -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    step = ColdStartPlanStep(
        occurrence.step_id,
        ["requirement::0"],
        ColdStartCandidateSource.VERIFIED,
        str(occurrence.node_ref),
        ColdStartExecutionMode.DIRECT_OR_SEEDED,
        {},
        {},
    )
    ctx.plan.cold_start_plan = ColdStartPlanProposal(
        "cold-conflict", [step], [step.step_id], [], [],
        {"requirement::0": [step.step_id]}, [],
    )
    ctx.plan.cold_start_scaffold = {
        "executable_step_ids": [step.step_id],
    }
    monkeypatch.setattr(
        runtime,
        "_materialize_cold_execution_plan",
        lambda *_args, **_kwargs: (ctx.plan, {}),
    )
    failed_terminal = ValidationResult.fail(
        "task", "task_contract_unsatisfied", "continue", task_contract=False,
    )
    monkeypatch.setattr(
        runtime.validation.task,
        "terminal",
        lambda *_args, **_kwargs: failed_terminal,
    )

    def conflict_step(*_args, **_kwargs):
        ctx.plan_conflict_declared = True
        ctx.plan_conflict_context = {
            "conflict_code": "runtime_plan_conflict",
        }
        ctx.trace_builder.trace.metadata["runtime_plan_conflicts"] = [
            dict(ctx.plan_conflict_context),
        ]
        return False, "runtime_plan_conflict", "plan_conflict"

    monkeypatch.setattr(runtime, "_run_verified_cold_step", conflict_step)
    dynamic_calls: list[dict] = []

    def run_dynamic(*_args, **kwargs):
        dynamic_calls.append(dict(kwargs))
        return {
            "success": False,
            "strict_success": False,
            "failure_code": "benchmark_failure",
            "rescue": bool(kwargs.get("rescue")),
            "cold_start_continuation": bool(
                kwargs.get("cold_start_continuation"),
            ),
        }

    monkeypatch.setattr(runtime.node_executor, "run_dynamic", run_dynamic)

    trace = runtime._run_cold_start(ctx, mode="online")

    assert dynamic_calls == [{"rescue": True}]
    assert trace.task_rescue_required is True
    assert trace.graph_self_sufficient_success is False
    assert "task_rescue" in trace.metadata
    assert "cold_start_dynamic_continuation" not in trace.metadata


def test_preparation_plan_conflict_surfaces_typed_result() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("report_runtime_status", {"status": "plan_conflict"}),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    result = runtime.node_executor.run_preparation_session(
        occurrence, invocations, ctx,
    )

    assert result.failure_code == "runtime_plan_conflict"
    assert result.failure_layer == "composite"
    status_call = next(
        item for item in ctx.trace_builder.trace.native_tool_calls
        if item.tool_name == "report_runtime_status"
    )
    assert status_call.preflight_result == {
        "accepted": True, "status": "plan_conflict",
    }
    factory.assert_exhausted()


def test_plan_conflict_skips_seeded_and_dynamic_rescue_stays_graph_failure() -> None:
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool(
            "report_runtime_status",
            {
                "status": "plan_conflict",
                "detail": "current rigid occurrence conflicts with public evidence",
            },
        ),
    ])
    factory.enqueue("runtime_dynamic", [
        FakeReply.tool("environment_action", {"action_id": "r000_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r001_a003"}),
        FakeReply.tool("environment_action", {"action_id": "r002_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r003_a003"}),
    ])
    runtime, ctx, _occurrence, _invocations = _context(factory)

    trace = runtime.run_task(ctx.task)

    assert trace.strict_task_success is True
    assert trace.task_rescue_required is True
    assert trace.graph_self_sufficient_success is False
    assert sum(
        item.session_type == "RuntimePreparationSession"
        for item in trace.agent_sessions
    ) == 1
    assert not any(
        item.session_type == "SeededSession"
        for item in trace.agent_sessions
    )
    conflict = trace.metadata["runtime_plan_conflicts"][0]
    assert conflict["rescue_attempted"] is True
    assert conflict["rescue_strict_success"] is True
    rescue_session = next(
        item for item in trace.agent_sessions
        if item.session_type == "DynamicTaskSession"
    )
    rescue_prompt = next(
        item["content"] for item in rescue_session.snapshot["messages"]
        if item["role"] == "user"
    )
    policy = json.loads(
        rescue_prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1]
    )
    guidance = policy["rescue_method_guidance"]
    assert guidance["conflict_code"] == "runtime_plan_conflict"
    assert "semantic_anchors" not in guidance
    assert "world_revision" not in guidance
    factory.assert_exhausted()
