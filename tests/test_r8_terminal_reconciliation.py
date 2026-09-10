from __future__ import annotations

import importlib.util
from pathlib import Path

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.results import (
    ImplementationExecutionResult,
    NodeExecutionStatus,
    ValidationResult,
)
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceEventType


def _binding_fixtures():
    path = Path(__file__).with_name(
        "test_stored_composite_binding_authority.py"
    )
    spec = importlib.util.spec_from_file_location("_r8_binding_fixtures", path)
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
            # The selected environment action is the benchmark terminal action
            # in this fixture.  The base harness still owns all semantic facts.
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


def _execute_terminal_action(ctx, *, action_type: str) -> None:
    spec = next(
        item
        for item in ctx.action_catalog
        if item.action_type == action_type
        and (
            action_type != "GO_TO"
            or item.arguments == {"destination": "desk_1"}
        )
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
            "origin": "r8_terminal_test",
        },
    )


def test_terminal_preparation_effect_is_credited_and_outputs_are_published(
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
        observed["ctx"] = ctx
        _execute_terminal_action(ctx, action_type="GO_TO")
        return runtime.node_executor.not_started(
            occurrence,
            failure_code="runtime_binding_unresolved",
        )

    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        preparation,
    )
    monkeypatch.setattr(
        runtime.node_executor,
        "run_seeded_fresh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal reconciliation must avoid Seeded")
        ),
    )

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.AGENT_COMPLETED_BEFORE_INVOCATION
    assert node.direct_result["started"] is False
    assert node.direct_result["atomic_effect_passed"] is True
    assert node.validated_outputs == {"reached_location": "desk_1"}
    live_ctx = observed["ctx"]
    assert live_ctx.validated_outputs[occurrence.occurrence_id] == {
        "reached_location": "desk_1",
    }
    assert live_ctx.binding_store.validated_outputs(
        occurrence.occurrence_id
    )["reached_location"].value == "desk_1"
    levels = [item.level for item in trace.validations]
    assert "runtime_repeat_preflight" in levels
    assert "runtime_repeat_commit" in levels
    assert any(
        item.level == "atomic" and item.result["passed"]
        for item in trace.validations
    )
    positive_events = [
        event
        for event in CreditAssigner().assign(trace)
        if event.event is EvidenceEventType.AGENT_NODE_SUCCESS
    ]
    assert len(positive_events) == 1
    assert positive_events[0].artifact_ref == str(occurrence.node_ref)
    assert positive_events[0].occurrence_id == occurrence.occurrence_id
    assert harness.execution_count == 1
    assert not trace.agent_sessions
    assert not trace.implementation_invocations
    assert not trace.tool_executions


def test_terminal_without_current_atomic_effect_remains_skipped(
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
        _execute_terminal_action(ctx, action_type="LOOK")
        return runtime.node_executor.not_started(
            occurrence,
            failure_code="runtime_binding_unresolved",
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
    assert not any(
        item.level == "atomic" and item.result["passed"]
        for item in trace.validations
    )
    assert harness.execution_count == 1


def test_seeded_terminal_effect_reconciles_as_seeded_success(
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
    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        lambda *_args, **_kwargs: runtime.node_executor.not_started(
            occurrence,
            failure_code="runtime_binding_unresolved",
        ),
    )

    def seeded(_occurrence, ctx, **_kwargs):
        _execute_terminal_action(ctx, action_type="GO_TO")
        result = runtime.node_executor.not_started(
            occurrence,
            failure_code="atomic_effect_violation",
        )
        result.node_status = NodeExecutionStatus.SEEDED_FAILED
        return result

    monkeypatch.setattr(runtime.node_executor, "run_seeded_fresh", seeded)

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.SEEDED_SUCCESS
    assert node.seeded_result["started"] is False
    assert node.seeded_result["atomic_effect_passed"] is True
    assert node.validated_outputs == {"reached_location": "desk_1"}
    assert harness.execution_count == 1
    assert any(
        item.level == "atomic" and item.result["passed"]
        for item in trace.validations
    )


def test_started_learned_invocation_is_not_reclassified_before_invocation(
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
        _execute_terminal_action(ctx, action_type="GO_TO")
        return ImplementationExecutionResult(
            "skill://started_impl@1.0.0",
            str(occurrence.node_ref),
            True,
            True,
            False,
            False,
            failure_layer="atomic",
            failure_code="terminal_interrupted",
            node_status=NodeExecutionStatus.DIRECT_FAILED,
            terminal_interrupted=True,
        )

    monkeypatch.setattr(
        runtime.node_executor,
        "run_preparation_session",
        preparation,
    )
    monkeypatch.setattr(
        runtime,
        "_reconcile_terminal_current_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("started invocation must retain terminal semantics")
        ),
    )

    trace = runtime.run_task(context.task)

    node = trace.node_records[0]
    assert node.status is NodeExecutionStatus.SKIPPED_GOAL_TERMINAL
    assert node.direct_result["started"] is True
    assert node.direct_result["terminal_interrupted"] is True
    assert harness.execution_count == 1


def test_terminal_reconciliation_preserves_repeat_preflight_authority(
    monkeypatch,
) -> None:
    runtime, ctx, occurrence, _invocations, harness = (
        _terminal_navigation_runtime()
    )
    ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
    ctx.begin_occurrence(occurrence)
    _execute_terminal_action(ctx, action_type="GO_TO")
    commit_calls: list[object] = []
    monkeypatch.setattr(
        ctx.binding_store,
        "preflight_repeat_bindings",
        lambda *_args, **_kwargs: ValidationResult.fail(
            "runtime_repeat",
            "runtime_repetition_distinctness_violation",
            "the terminal witness reused a distinct RepeatBlock value",
        ),
    )
    monkeypatch.setattr(
        ctx.binding_store,
        "commit_repeat_bindings",
        lambda *_args, **_kwargs: commit_calls.append(object()),
    )

    reconciled = runtime._reconcile_terminal_current_atomic(
        occurrence,
        ctx,
        mode="seeded",
    )

    assert reconciled is None
    repeat = ctx.trace_builder.trace.validations[-1]
    assert repeat.level == "runtime_repeat_preflight"
    assert repeat.result["passed"] is False
    assert repeat.result["failure_codes"] == [
        "runtime_repetition_distinctness_violation"
    ]
    assert commit_calls == []
    assert not any(
        item.level == "atomic" and item.result["passed"]
        for item in ctx.trace_builder.trace.validations
    )
    assert harness.execution_count == 1
