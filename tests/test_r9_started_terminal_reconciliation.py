from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
)
from atomic_skillgraph.core.results import (
    ImplementationExecutionResult,
    NodeExecutionStatus,
    ToolExecutionResult,
    ValidationResult,
)
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceEventType
from atomic_skillgraph.traces.schema import (
    ImplementationInvocationRecord,
    ToolExecutionRecord,
)


def _binding_fixtures():
    path = Path(__file__).with_name(
        "test_stored_composite_binding_authority.py"
    )
    spec = importlib.util.spec_from_file_location("_r9_binding_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terminal_navigation_runtime():
    fixtures = _binding_fixtures()

    class TerminalActionHarness(fixtures._PickPlaceHarness):
        def __init__(self) -> None:
            super().__init__()
            self.execution_count = 0

        def execute_action(self, action_id: str, revision: int):
            self.execution_count += 1
            self._won = True
            return super().execute_action(action_id, revision)

    harness = TerminalActionHarness()
    runtime, context, occurrence, invocations = fixtures._single_nav_context(
        harness,
        fixtures.FakeAgentFactory(),
    )
    occurrence.binding_specs = {
        "destination": BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role="destination",
        ),
    }
    return runtime, context, occurrence, invocations, harness


def _execute_terminal_action(ctx) -> None:
    spec = next(
        item
        for item in ctx.action_catalog
        if item.action_type == "GO_TO"
        and item.arguments == {"destination": "desk_1"}
    )
    ctx.budget.consume_action()
    result = ctx.harness.execute_action(spec.action_id, spec.revision)
    ctx.update_after_action(
        result,
        {
            "action_id": spec.action_id,
            "revision": spec.revision,
            "action_type": spec.action_type,
            "arguments": dict(spec.arguments),
            "accepted": result.accepted,
            "observation": result.observation,
            "done": result.done,
            "won": result.won,
            "new_revision": result.new_revision,
            "occurrence_id": ctx.active_occurrence_id,
            "origin": "r9_started_terminal_test",
        },
    )


def _grounded_destination(value: str, revision: int) -> RuntimeBinding:
    return RuntimeBinding(
        "destination",
        value,
        "entity",
        BindingSource.HARNESS_EVIDENCE,
        BindingStatus.GROUNDED,
        BindingResolution.CONCRETE,
        [f"validator:test:{value}"],
        revision,
    )


def _started_terminal_result(
    occurrence,
    ctx,
    *,
    destination: str = "desk_1",
    failure_layer: str = "atomic",
    failure_code: str = "terminal_interrupted",
    tool_intrinsic_failure: bool = False,
    terminal_interrupted: bool = True,
) -> ImplementationExecutionResult:
    tool_result = ToolExecutionResult(
        tool_ref="tool://started_terminal@1.0.0",
        preflight_passed=True,
        started=True,
        completed=False,
        state_changed=True,
        executed_step_count=1,
        failure_step_index=None,
        partial_effects=[],
        output_candidates={},
        before_revision=0,
        after_revision=ctx.world_revision,
        failure_layer="tool" if tool_intrinsic_failure else "",
        failure_code="tool_execution_error" if tool_intrinsic_failure else "",
        terminal_interrupted=terminal_interrupted,
        intrinsic_failure=tool_intrinsic_failure,
    )
    direct = ImplementationExecutionResult(
        implementation_ref="skill://started_impl@1.0.0",
        atomic_ref=str(occurrence.node_ref),
        preflight_passed=True,
        started=True,
        completed=False,
        atomic_effect_passed=False,
        tool_results=[tool_result],
        realized_bindings={
            "destination": _grounded_destination(
                destination,
                ctx.world_revision,
            ),
        },
        before_state_ref="revision:0",
        after_state_ref=f"revision:{ctx.world_revision}",
        failure_layer=failure_layer,
        failure_code=failure_code,
        node_status=NodeExecutionStatus.DIRECT_FAILED,
        terminal_interrupted=terminal_interrupted,
    )
    ctx.trace_builder.trace.tool_executions.append(ToolExecutionRecord(
        "tool_attempt_started_terminal",
        occurrence.occurrence_id,
        tool_result.tool_ref,
        to_primitive(tool_result),
        "tool_span_started_terminal",
    ))
    ctx.trace_builder.trace.implementation_invocations.append(
        ImplementationInvocationRecord(
            "impl_attempt_started_terminal",
            occurrence.occurrence_id,
            direct.implementation_ref,
            {"destination": destination},
            {"passed": True},
            to_primitive(direct),
            "impl_span_started_terminal",
        )
    )
    return direct


def test_started_terminal_effect_reconciles_exact_identity_without_new_work(
    monkeypatch,
) -> None:
    runtime, context, occurrence, _invocations, harness = (
        _terminal_navigation_runtime()
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "try_autonomous",
        lambda *_args, **_kwargs: None,
    )
    observed: dict[str, object] = {}

    def preparation(_occurrence, _invocations, ctx, **_kwargs):
        _execute_terminal_action(ctx)
        observed["ctx"] = ctx
        return _started_terminal_result(occurrence, ctx)

    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        preparation,
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "run_seeded_fresh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("successful reconciliation must avoid Seeded")
        ),
    )
    original_complete = runtime.node_executor._complete_from_current_effect
    preferred_bindings: list[dict[str, object]] = []

    def inspect_complete(*args, **kwargs):
        if kwargs.get("resolution_out") is not None:
            preferred_bindings.append(dict(kwargs["preferred_bindings"]))
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(
        runtime.node_executor,
        "_complete_from_current_effect",
        inspect_complete,
    )

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.DIRECT_TERMINAL_EFFECT_SUCCESS
    assert node.direct_result["started"] is True
    assert node.direct_result["completed"] is False
    assert node.direct_result["terminal_interrupted"] is True
    assert node.direct_result["terminal_effect_reconciled"] is True
    assert node.direct_result["atomic_effect_passed"] is True
    assert node.direct_result["validated_outputs"] == {
        "reached_location": "desk_1",
    }
    assert node.direct_result["atomic_witness_refs"]
    assert node.validated_outputs == {"reached_location": "desk_1"}
    assert preferred_bindings == [{"destination": "desk_1"}]
    assert harness.execution_count == 1
    assert trace.agent_sessions == []
    assert len(trace.implementation_invocations) == 1
    assert len(trace.tool_executions) == 1

    invocation = trace.implementation_invocations[0].result
    assert invocation["terminal_effect_reconciled"] is True
    assert invocation["atomic_effect_passed"] is True
    assert invocation["completed"] is False
    tool = trace.tool_executions[0].result
    assert tool["terminal_interrupted"] is True
    assert tool["completed"] is False

    levels = [item.level for item in trace.validations]
    assert "runtime_repeat_preflight" in levels
    assert "runtime_repeat_commit" in levels
    metrics = trace.metadata["v32_metrics"]
    assert metrics["runtime_terminal_started_reconciliation_attempt_count"] == 1
    assert metrics["runtime_terminal_started_reconciliation_success_count"] == 1
    assert metrics.get(
        "runtime_terminal_started_reconciliation_failure_count", 0,
    ) == 0

    events = CreditAssigner().assign(trace)
    atomic_events = [
        event
        for event in events
        if event.artifact_ref == str(occurrence.node_ref)
        and event.event is EvidenceEventType.DIRECT_SUCCESS
    ]
    assert len(atomic_events) == 1
    assert atomic_events[0].metadata["node_status"] == (
        "direct_terminal_effect_success"
    )
    assert not any(
        event.artifact_ref
        in {
            "skill://started_impl@1.0.0",
            "tool://started_terminal@1.0.0",
        }
        and event.event in {
            EvidenceEventType.DIRECT_SUCCESS,
            EvidenceEventType.DIRECT_FAILURE,
        }
        for event in events
    )
    live_ctx = observed["ctx"]
    assert live_ctx.validated_outputs[occurrence.occurrence_id] == {
        "reached_location": "desk_1",
    }


def test_started_terminal_effect_for_different_identity_is_rejected(
    monkeypatch,
) -> None:
    runtime, context, occurrence, _invocations, harness = (
        _terminal_navigation_runtime()
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "try_autonomous",
        lambda *_args, **_kwargs: None,
    )

    def preparation(_occurrence, _invocations, ctx, **_kwargs):
        _execute_terminal_action(ctx)
        return _started_terminal_result(
            occurrence,
            ctx,
            destination="drawer_2",
        )

    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        preparation,
    )
    original_complete = runtime.node_executor._complete_from_current_effect
    preferred_bindings: list[dict[str, object]] = []

    def inspect_complete(*args, **kwargs):
        if kwargs.get("resolution_out") is not None:
            preferred_bindings.append(dict(kwargs["preferred_bindings"]))
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(
        runtime.node_executor,
        "_complete_from_current_effect",
        inspect_complete,
    )

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.SKIPPED_GOAL_TERMINAL
    assert node.direct_result["atomic_effect_passed"] is False
    assert node.direct_result["terminal_effect_reconciled"] is False
    assert preferred_bindings == [{"destination": "drawer_2"}]
    assert harness.execution_count == 1
    metrics = trace.metadata["v32_metrics"]
    assert metrics["runtime_terminal_started_reconciliation_attempt_count"] == 1
    assert metrics["runtime_terminal_started_reconciliation_failure_count"] == 1
    assert metrics.get(
        "runtime_terminal_started_reconciliation_success_count", 0,
    ) == 0


def test_started_terminal_reconciliation_preserves_repeat_rejection(
    monkeypatch,
) -> None:
    runtime, context, occurrence, _invocations, harness = (
        _terminal_navigation_runtime()
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "try_autonomous",
        lambda *_args, **_kwargs: None,
    )
    commit_calls: list[object] = []

    def preparation(_occurrence, _invocations, ctx, **_kwargs):
        _execute_terminal_action(ctx)
        monkeypatch.setattr(
            ctx.binding_store,
            "preflight_repeat_bindings",
            lambda *_args, **_kwargs: ValidationResult.fail(
                "runtime_repeat",
                "runtime_repetition_distinctness_violation",
                "the started identity reused a distinct RepeatBlock value",
            ),
        )
        monkeypatch.setattr(
            ctx.binding_store,
            "commit_repeat_bindings",
            lambda *_args, **_kwargs: commit_calls.append(object()),
        )
        return _started_terminal_result(occurrence, ctx)

    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        preparation,
    )

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.SKIPPED_GOAL_TERMINAL
    assert node.direct_result["terminal_effect_reconciled"] is False
    assert commit_calls == []
    repeat = next(
        item
        for item in reversed(trace.validations)
        if item.level == "runtime_repeat_preflight"
    )
    assert repeat.result["passed"] is False
    assert repeat.result["failure_codes"] == [
        "runtime_repetition_distinctness_violation",
    ]
    assert not any(
        item.level == "atomic" and item.result.get("passed")
        for item in trace.validations
    )
    assert harness.execution_count == 1
    metrics = trace.metadata["v32_metrics"]
    assert metrics["runtime_terminal_started_reconciliation_attempt_count"] == 1
    assert metrics["runtime_terminal_started_reconciliation_failure_count"] == 1


@pytest.mark.parametrize(
    ("failure_layer", "failure_code", "tool_intrinsic_failure"),
    [
        (
            "runtime_binding",
            "runtime_repetition_distinctness_violation",
            False,
        ),
        ("implementation", "implementation_mapping_error", False),
        ("tool", "tool_execution_error", False),
        ("atomic", "terminal_interrupted", True),
    ],
)
def test_intrinsic_started_failure_cannot_use_terminal_shortcut(
    monkeypatch,
    failure_layer: str,
    failure_code: str,
    tool_intrinsic_failure: bool,
) -> None:
    runtime, context, occurrence, _invocations, harness = (
        _terminal_navigation_runtime()
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "try_autonomous",
        lambda *_args, **_kwargs: None,
    )

    def preparation(_occurrence, _invocations, ctx, **_kwargs):
        _execute_terminal_action(ctx)
        return _started_terminal_result(
            occurrence,
            ctx,
            failure_layer=failure_layer,
            failure_code=failure_code,
            tool_intrinsic_failure=tool_intrinsic_failure,
        )

    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        preparation,
    )

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.SKIPPED_GOAL_TERMINAL
    assert node.direct_result["atomic_effect_passed"] is False
    assert node.direct_result["terminal_effect_reconciled"] is False
    assert harness.execution_count == 1
    metrics = trace.metadata.get("v32_metrics", {})
    assert metrics.get(
        "runtime_terminal_started_reconciliation_attempt_count", 0,
    ) == 0
    assert not any(
        item.level == "atomic" and item.result.get("passed")
        for item in trace.validations
    )
