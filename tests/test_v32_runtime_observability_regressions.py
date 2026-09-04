"""Focused closure tests for remaining v3.2 runtime evidence boundaries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from atomic_skillgraph.core.contracts import (
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.results import RuntimeLinearPlan
from atomic_skillgraph.core.status import ToolStatus
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceEventType
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeHarness, fake_task


def _event(
    index: int,
    action_type: str,
    arguments: dict[str, str],
    *,
    shared: bool = False,
) -> dict[str, object]:
    return {
        "accepted": True,
        "action_id": f"e{index}",
        "event_id": f"e{index}",
        "event_index": index,
        "action_type": action_type,
        "arguments": arguments,
        "before_revision": index,
        "after_revision": index + 1,
        "span_id": "span",
        "shared_precondition_evidence": shared,
    }


def _normalized(actions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "trace_id": "trace_atomicizer_closure",
        "source_task": {"task_id": "task"},
        "actions": actions,
        "runtime_spans": [{
            "span_id": "span",
            "kind": "runtime_seeded",
            "occurrence_id": "occ",
            "action_start": 0,
            "action_end": len(actions),
            "parent_span_id": None,
            "learnable": True,
        }],
        "before_state_facts": [],
        "after_state_facts": [],
        "validations": [],
    }


def _proposal(
    phase: str,
    *,
    start: int,
    end: int,
    support: list[str],
    inputs: dict[str, str],
    effect: SemanticPredicate,
    shared: list[str] | None = None,
) -> AtomicOccurrenceProposal:
    first_value = next(iter(inputs.values()))
    return AtomicOccurrenceProposal(
        phase_id=phase,
        intent=phase,
        event_start=start,
        event_end=end,
        input_roles=inputs,
        output_roles={"result": first_value},
        preconditions=[],
        effects=[effect],
        rationale="code-authoritative transition",
        support_event_ids=support,
        shared_precondition_event_ids=list(shared or ()),
    )


def test_atomicizer_rejects_a_single_orphan_runtime_span() -> None:
    normalized = _normalized([
        _event(0, "EXAMINE", {"item": "apple_1"}),
    ])
    normalized["runtime_spans"] = []
    proposal = _proposal(
        "observe",
        start=0,
        end=0,
        support=["e0"],
        inputs={"item": "apple_1"},
        effect=SemanticPredicate(
            "object.observed", {"object": "apple_1"},
        ),
    )

    with pytest.raises(ValueError, match="orphan RuntimeSpan"):
        Atomicizer().validate_and_canonicalize([proposal], normalized)


def test_atomicizer_allows_only_non_effect_shared_precondition_overlap() -> None:
    actions = [
        _event(
            0,
            "EXAMINE",
            {"item": "apple_1"},
            shared=True,
        ),
        _event(1, "HEAT", {"object": "mug_1"}),
    ]
    normalized = _normalized(actions)
    observe = _proposal(
        "observe",
        start=0,
        end=0,
        support=["e0"],
        inputs={"item": "apple_1"},
        effect=SemanticPredicate(
            "object.observed", {"object": "apple_1"},
        ),
    )
    heat = _proposal(
        "heat",
        start=0,
        end=1,
        support=["e0", "e1"],
        shared=["e0"],
        inputs={"prerequisite": "apple_1", "object": "mug_1"},
        effect=SemanticPredicate(
            "object.heated", {"object": "mug_1"},
        ),
    )

    canonical = Atomicizer().validate_and_canonicalize(
        [observe, heat], normalized,
    )
    assert len(canonical) == 2
    assert canonical[1].shared_precondition_event_ids == ["e0"]

    duplicate_effect = replace(
        heat,
        effects=[SemanticPredicate(
            "object.observed", {"object": "apple_1"},
        )],
    )
    with pytest.raises(ValueError, match="duplicate Effect ownership"):
        Atomicizer().validate_and_canonicalize(
            [observe, duplicate_effect], normalized,
        )


def test_runtime_r1_effect_witness_has_single_atomic_owner() -> None:
    actions = [
        _event(0, "EXAMINE", {"item": "apple_1"}, shared=True),
        _event(1, "EXAMINE", {"item": "apple_1"}, shared=True),
    ]
    actions[0]["after_revision"] = 2
    normalized = _normalized(actions)
    runtime_ref = (
        "alfworld_action_fact:r2:object.observed:object=apple_1"
    )
    normalized["after_state_facts"] = [
        {
            "predicate": "object.observed",
            "args": {"object": "apple_1"},
            "revision": 2,
            "witness_ref": runtime_ref,
            "event_index": 0,
            "source_kind": "runtime_trial_r1",
            "draft_id": "draft_observe",
        },
        {
            "predicate": "object.observed",
            "args": {"object": "apple_1"},
            "revision": 2,
            "witness_ref": runtime_ref,
            "event_index": 1,
            "source_kind": "runtime_trial_r1",
            "draft_id": "draft_observe",
        },
    ]
    first = _proposal(
        "first",
        start=0,
        end=1,
        support=["e0"],
        shared=["e0"],
        inputs={"item": "apple_1"},
        effect=SemanticPredicate(
            "object.observed", {"object": "apple_1"},
        ),
    )
    second = _proposal(
        "second",
        start=0,
        end=1,
        support=["e1"],
        shared=["e1"],
        inputs={"item": "apple_1"},
        effect=SemanticPredicate(
            "object.observed", {"object": "apple_1"},
        ),
    )
    first.effect_witness_refs = [runtime_ref]
    second.effect_witness_refs = [runtime_ref]

    with pytest.raises(ValueError, match="Effect witness ownership"):
        Atomicizer().validate_and_canonicalize(
            [first, second], normalized,
        )


def test_terminal_empirical_credit_requires_candidate_execution() -> None:
    base = {
        "trace_id": "trace-terminal-credit",
        "task": {"task_id": "task-terminal-credit"},
        "infrastructure_failure": False,
        "implementation_invocations": [],
        "tool_executions": [],
        "node_records": [{
            "occurrence_id": "occ-1",
            "atomic_ref": "skill://atomic@1.0.0",
            "status": "already_satisfied",
            "direct_result": {"started": False},
        }],
        "runtime_plan": {
            "source_composite_ref": "skill://terminal@1.0.0",
            "planner_audit": {
                "selected_composite_authority": {
                    "kind": "terminal_empirical",
                },
            },
        },
        "graph_self_sufficient_success": True,
        "task_rescue_required": False,
        "benchmark_success": True,
        "task_contract_success": False,
        "metadata": {
            "terminal_empirical_execution": {
                "candidate_executed": False,
            },
        },
    }

    unexecuted = CreditAssigner().assign(base)
    assert not any(
        item.artifact_kind == "composite"
        and item.event is EvidenceEventType.SELF_SUFFICIENT_SUCCESS
        for item in unexecuted
    )

    executed = CreditAssigner().assign({
        **base,
        "metadata": {
            "terminal_empirical_execution": {
                "candidate_executed": True,
            },
        },
    })
    assert any(
        item.artifact_kind == "composite"
        and item.event is EvidenceEventType.SELF_SUFFICIENT_SUCCESS
        for item in executed
    )


def _tool(program: list[dict[str, object]]) -> ToolAsset:
    return ToolAsset(
        ref="tool://selector_probe@1.0.0",
        summary="selector probe",
        signature={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        interface={
            "output_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
        artifact_kind="tool_ir_v1",
        artifact={
            "schema_version": 1,
            "max_actions": 2,
            "program": program,
            "final_effects": [{
                "predicate": "object.observed",
                "args": {"object": "$result"},
                "effect_domain": "world",
            }],
            "evidence_outputs": [],
            "path_expectations": [],
        },
        tests=[],
        safety={"reviewed": True, "allowed_action_types": [], "zero_llm": True},
        provenance={"source": "success_evolution"},
        metadata={},
        status=ToolStatus.ADMISSION_PENDING,
    )


def _run_tool(tool: ToolAsset):
    harness = FakeHarness()
    task = fake_task("selector-probe", "apple_1")
    harness.reset(task)
    plan = RuntimeLinearPlan.full_dynamic(
        task.task_id, harness.task_contract(task), reason="selector_probe",
    )
    trace = TraceRecord.create(
        TaskRecord(
            task.task_id,
            task.benchmark,
            task.goal,
            task.task_type,
            "selector-probe",
        ),
        {},
        {},
        {"source": "full_dynamic"},
    )
    context = TaskRuntimeContext.create(
        task,
        plan,
        harness,
        TraceBuilder(trace),
        RuntimeBudget(global_action_budget=10, node_action_budget=5),
    )
    return ToolRunner(ValidationEngine().tool).run(
        tool,
        {"target": "apple_1"},
        context,
        occurrence_id="occ-selector",
    )


def test_tool_runner_localizes_selector_no_match_and_control_path() -> None:
    tool = _tool([
        {
            "node_id": "select",
            "op": "FOR_EACH",
            "collection_source": {
                "source": "action_catalog",
                "where": {"action_type": "NOT_PRESENT"},
                "project": {"kind": "argument", "role": "object"},
                "distinct": True,
            },
            "iteration_variable": "candidate",
            "max_iterations": 1,
            "body": [{
                "node_id": "stop",
                "op": "STOP_WHEN",
                "condition": {
                    "source": "local_variable",
                    "field": "candidate",
                    "op": "exists",
                },
            }],
        },
        {
            "node_id": "return",
            "op": "RETURN",
            "output_sources": {
                "result": {"source": "tool_input", "field": "target"},
            },
        },
    ])

    result = _run_tool(tool)
    evidence = result.tool_path_evidence
    assert result.failure_layer == "tool"
    assert result.failure_code == "tool_ir_selector_no_match"
    assert result.program_node_id == "select"
    assert result.started is False
    assert evidence["failure_layer"] == "tool_ir"
    assert evidence["failure_code"] == "tool_ir_selector_no_match"
    assert evidence["program_node_id"] == "select"
    assert evidence["executed_node_ids"] == ["select"]
    assert evidence["program_path_id"] == "program/select"
