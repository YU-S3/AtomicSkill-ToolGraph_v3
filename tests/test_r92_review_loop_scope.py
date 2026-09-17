"""Interpreter regressions for the 1fd8ab1 review's lexical-scope defect."""

from types import SimpleNamespace

import pytest

from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.ir import ToolExecutionState
from atomic_skillgraph.tooling.validator import _scope_pass


def _loop(name, variable, body):
    return {
        "op": "FOR_EACH", "node_id": name, "iteration_variable": variable,
        "collection_source": {"source": "tool_input", "field": name},
        "max_iterations": 2, "body": body,
    }


def _return(variable="cursor"):
    return {
        "op": "RETURN", "node_id": "return",
        "output_sources": {
            "result": {"source": "local_variable", "field": variable},
            "input": {"source": "tool_input", "field": "cursor"},
        },
    }


def _execute(program, state, runner=None, terminal=None):
    runner = runner or ToolRunner.__new__(ToolRunner)
    return runner._execute_ir_nodes(
        program, state, SimpleNamespace(harness=SimpleNamespace()),
        occurrence_id="occ", span_id="span",
        tool=SimpleNamespace(interface={"output_schema": {"properties": {
            "result": {}, "input": {},
        }}}), terminal=terminal if terminal is not None else [],
    )


@pytest.mark.parametrize("variable", ["cursor", "other"])
@pytest.mark.parametrize("stop", [False, True])
def test_nested_scope_matches_static_scope_and_keeps_input_namespace(variable, stop):
    inner_body = [{
        "op": "STOP_WHEN", "node_id": "stop",
        "condition": {"source": "local_variable", "field": variable,
                      "op": "equals", "value": "inner_1"},
    }] if stop else []
    program = [_loop("outer", "cursor", [
        _loop("inner", variable, inner_body), _return(),
    ])]
    failures = []
    _scope_pass(
        program, atomic_inputs={"outer", "inner", "cursor"},
        atomic_outputs={"result", "input"},
        fail=lambda *args: failures.append(args),
    )
    assert failures == []
    state = ToolExecutionState(bindings={
        "outer": ["outer_1"], "inner": ["inner_1", "inner_2"],
        "cursor": "input_identity",
    })
    assert _execute(program, state) == "RETURN_PROGRAM"
    assert state.outputs == {"result": "outer_1", "input": "input_identity"}
    assert state.local == {}
    assert state.bindings["cursor"] == "input_identity"
    assert state.loop_iteration_counts == {"outer": 1, "inner": 1 if stop else 2}
    assert state.executed_nodes == (["outer", "inner", "stop", "return"]
                                    if stop else ["outer", "inner", "return"])
    assert state.executed_control_step_count == len(state.executed_nodes)


@pytest.mark.parametrize("exit_kind", ["return", "failure", "terminal", "exception"])
@pytest.mark.parametrize("previous", [None, "surrounding_value"])
def test_loop_restores_locals_on_every_nonlocal_exit(exit_kind, previous):
    state = ToolExecutionState(bindings={"outer": ["outer_1"], "inner": ["inner_1"],
                                         "cursor": "input_identity"})
    if previous is not None:
        state.local["cursor"] = previous
    runner = ToolRunner.__new__(ToolRunner)
    recorded = []

    def action(node, primitive, ctx, state, **kwargs):
        assert state.local["cursor"] == "inner_1"
        recorded.append(node["node_id"])
        state.executed_action_count += 1
        if exit_kind == "exception":
            raise RuntimeError("primitive failure")
        return {"accepted": exit_kind != "failure", "won": exit_kind == "terminal",
                "done": exit_kind == "terminal"}

    runner._resolve_action_arguments = lambda node, state: {}
    runner._record_ir_action = action
    runner._validate_step_effects = lambda *args, **kwargs: {"step_effect_passed": True}
    body = [_return()] if exit_kind == "return" else [{"op": "ACTION", "node_id": "act"}]
    program = [_loop("outer", "cursor", [_loop("inner", "cursor", body)])]
    terminal = []
    if exit_kind == "exception":
        with pytest.raises(RuntimeError, match="primitive failure"):
            _execute(program, state, runner, terminal)
    else:
        assert _execute(program, state, runner, terminal) == {
            "return": "RETURN_PROGRAM", "failure": "FAIL_TOOL",
            "terminal": "",  # Original local scopes may unwind; no further ACTION exists.
        }[exit_kind]
        assert state.loop_iteration_counts == {"inner": 1, "outer": 1}
    assert state.local == ({} if previous is None else {"cursor": previous})
    assert state.bindings["cursor"] == "input_identity"
    assert state.executed_action_count == (0 if exit_kind == "return" else 1)
    assert len(recorded) == state.executed_action_count
    assert bool(terminal) == (exit_kind == "terminal")
    assert state.executed_control_step_count == 3
    assert state.path_tokens[:4] == ["outer", "outer:iteration:1", "inner", "inner:iteration:1"]
    if exit_kind == "return":
        assert state.outputs["result"] == "inner_1"


def test_loop_cleanup_does_not_reset_action_budget_or_execution_history():
    state = ToolExecutionState(
        bindings={"outer": ["outer_1", "outer_2"], "inner": ["inner_1"]},
        max_actions=2, max_control_steps=20,
    )
    runner = ToolRunner.__new__(ToolRunner)
    actions = []
    runner._resolve_action_arguments = lambda node, state: {"value": state.local["cursor"]}

    def action(node, primitive, ctx, state, **kwargs):
        actions.append(primitive["value"])
        state.executed_action_count += 1
        return {"accepted": True, "won": False, "done": False}

    runner._record_ir_action = action
    runner._validate_step_effects = lambda *args, **kwargs: {"step_effect_passed": True}
    program = [_loop("outer", "cursor", [
        _loop("inner", "cursor", [{"op": "ACTION", "node_id": "inside"}]),
        {"op": "ACTION", "node_id": "outside"},
    ])]
    assert _execute(program, state, runner) == "FAIL_TOOL"
    assert state.failure_code == "tool_ir_max_actions_exhausted"
    assert actions == ["inner_1", "outer_1"]
    assert state.local == {}
    assert state.executed_action_count == state.max_actions == 2
    assert state.max_control_steps == 20
    assert state.executed_control_step_count == 6
    assert state.executed_nodes == ["outer", "inner", "inside", "outside", "inner", "inside"]
    assert state.loop_iteration_counts == {"inner": 1, "outer": 2}


def test_control_budget_failure_restores_nested_scope_and_keeps_counter():
    state = ToolExecutionState(
        bindings={"outer": ["outer_1"], "inner": ["inner_1"]},
        local={"cursor": "surrounding_value"}, max_control_steps=2,
    )
    assert _execute([_loop("outer", "cursor", [
        _loop("inner", "cursor", [_return()]),
    ])], state) == "FAIL_TOOL"
    assert state.local == {"cursor": "surrounding_value"}
    assert state.failure_code == "tool_ir_control_step_exhausted"
    assert state.executed_control_step_count == 3
    assert state.max_control_steps == 2
    assert state.executed_nodes == ["outer", "inner"]
    assert state.loop_iteration_counts == {"inner": 1, "outer": 1}
