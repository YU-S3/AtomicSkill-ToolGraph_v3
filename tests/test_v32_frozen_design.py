"""Deterministic gates for the v3.2 frozen design."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    EffectDomain,
    ParameterSpec,
    PlannerRequirementBundle,
    RequirementSearchResult,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.errors import AtomicSkillGraphError
from atomic_skillgraph.core.results import RuntimeLinearPlan
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.planner.repairability import RepairabilityGate
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.support_retriever import SupportAtomicRetriever
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.ir import program_paths
from atomic_skillgraph.tooling.proposal import ToolProposal
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeHarness, fake_task


def _requirement() -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id="req_1",
        intent="look at object with light",
        desired_effects=[
            SemanticPredicate(
                "object.observed_with",
                {"object": "target", "light": "light"},
            )
        ],
        expected_inputs=[
            ParameterSpec("target", "entity"),
            ParameterSpec("light", "entity"),
        ],
        expected_outputs=[],
        precondition_hints=[],
        semantic_variants=[],
        required=True,
        rationale="contract requires observed_with",
    )


def _bundle() -> PlannerRequirementBundle:
    return PlannerRequirementBundle(requirements=[_requirement()])


def test_effect_domain_defaults_to_world_and_serializes() -> None:
    predicate = SemanticPredicate("object.at_location", {"object": "cup"})
    assert predicate.effect_domain is EffectDomain.WORLD
    evidence = SemanticPredicate(
        "entity.discovered_at",
        {"entity": "cup", "location": "cabinet"},
        effect_domain="evidence",
    )
    assert evidence.effect_domain is EffectDomain.EVIDENCE


def test_repairability_hard_gap_skips_p1r() -> None:
    decision = RepairabilityGate().decide(
        _bundle(),
        SimpleNamespace(passed=True),
        [
            RequirementSearchResult(
                requirement=_requirement(),
                candidates=[],
                covered=False,
                rejection_reasons=[],
                repair_hints=[],
            )
        ],
    )
    assert decision.repairable is False
    assert decision.reason_code == "planner_hard_capability_gap"
    assert decision.requirement_ids == ("req_1",)


def test_repairability_near_match_allows_p1r() -> None:
    decision = RepairabilityGate().decide(
        _bundle(),
        SimpleNamespace(passed=True),
        [
            RequirementSearchResult(
                requirement=_requirement(),
                candidates=[],
                covered=False,
                rejection_reasons=[
                    {
                        "atomic_ref": "atomic_near",
                        "compatibility": {
                            "effects_passed": False,
                            "failure_codes": ["input_contract_mismatch"],
                        },
                    }
                ],
                repair_hints=[
                    {
                        "atomic_ref": "atomic_near",
                        "contract_view": {
                            "effects": [{"predicate": "object.observed_with"}]
                        },
                    }
                ],
            )
        ],
    )
    assert decision.repairable is True
    assert decision.reason_code == "coverage_partial_effect_match"


def test_terminal_latch_blocks_second_env_step() -> None:
    harness = FakeHarness()
    harness.reset(fake_task("task-latch", "apple_1"))
    spec = next(
        item for item in harness.action_catalog()
        if item.action_type == "TAKE"
    )
    first = harness.execute_action(spec.action_id, spec.revision)
    assert first.won is True
    with pytest.raises(AtomicSkillGraphError) as error:
        harness.execute_action(spec.action_id, spec.revision)
    assert error.value.code == "harness_terminal_latched"


def _ir_tool() -> ToolAsset:
    return ToolAsset(
        ref="tool://tool_ir_take@1.0.0",
        summary="take target",
        signature={
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
        interface={
            "output_schema": {
                "type": "object",
                "properties": {"held_object": {"type": "string"}},
                "required": ["held_object"],
                "additionalProperties": False,
            }
        },
        artifact_kind="tool_ir_v1",
        artifact={
            "schema_version": 1,
            "max_actions": 5,
            "program": [
                {
                    "node_id": "take",
                    "op": "ACTION",
                    "action_type": "TAKE",
                    "argument_mapping": {
                        "item": {"kind": "skill_input", "source_role": "item"}
                    },
                },
                {
                    "node_id": "return_held",
                    "op": "RETURN",
                    "output_sources": {
                        "held_object": {
                            "source": "tool_input",
                            "field": "item",
                        }
                    },
                },
            ],
            "final_effects": [
                {
                    "predicate": "agent.holds",
                    "args": {"object": "$item"},
                    "effect_domain": "world",
                }
            ],
            "evidence_outputs": [
                {"role": "held_object", "source": "tool_input", "field": "item"}
            ],
            "output_mapping": {
                "held_object": {
                    "kind": "skill_input",
                    "source_role": "item",
                }
            },
        },
        tests=[],
        safety={"reviewed": True, "zero_llm": True},
        provenance={"source": "test_v32"},
        metadata={},
        status=ToolStatus.CANDIDATE,
    )


def test_tool_ir_executes_zero_llm_and_stops_at_return() -> None:
    harness = FakeHarness()
    task = fake_task("task-ir", "apple_1")
    plan = RuntimeLinearPlan.full_dynamic(
        task.task_id, harness.task_contract(task), reason="test",
    )
    trace = TraceRecord.create(
        TaskRecord(task.task_id, "fake", task.goal, task.task_type, "sig"),
        {},
        {},
        {"source": "full_dynamic"},
    )
    ctx = TaskRuntimeContext.create(
        task, plan, harness, TraceBuilder(trace),
        RuntimeBudget(global_action_budget=10),
    )
    runner = ToolRunner(ValidationEngine().tool)
    result = runner.run(
        _ir_tool(), {"item": "apple_1"}, ctx, occurrence_id="occ",
    )
    assert result.started is True
    assert result.completed is False
    assert result.terminal_interrupted is True
    assert result.intrinsic_failure is False
    assert result.output_candidates == {}  # terminal interruption is not a RETURN authority
    assert result.executed_node_count == 1
    assert ctx.used_actions == 1
    assert ctx.benchmark_terminal() is True


def test_tool_static_validator_rejects_unknown_opcode_and_code() -> None:
    atomic = AbstractAtomicSkill(
        ref="skill://atomic_take@1.0.0",
        summary="take",
        inputs=[ParameterSpec("item", "entity")],
        outputs=[ParameterSpec("held_object", "entity")],
        preconditions=[],
        effects=[SemanticPredicate("agent.holds", {"object": "$item"})],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.DRAFT,
    )
    proposal = ToolProposal(
        proposal_version="1",
        decision="create",
        summary="bad",
        atomic_ref="skill://atomic_take@1.0.0",
        inputs=atomic.inputs,
        outputs=atomic.outputs,
        program=[
            {
                "node_id": "bad",
                "op": "PYTHON",
                "action_type": "import os",
            }
        ],
        max_actions=1,
        final_effects=atomic.effects,
        evidence_outputs=[],
        path_expectations=[],
        rationale="bad",
    )
    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )
    assert report.passed is False
    assert report.failure_codes == ["tool_ir_opcode_unsupported"]


def test_support_retriever_is_contract_only() -> None:
    blocked = AbstractAtomicSkill(
        ref="skill://atomic_heat@1.0.0",
        summary="heat object",
        inputs=[ParameterSpec("object", "entity")],
        outputs=[],
        preconditions=[],
        effects=[SemanticPredicate("object.heated", {"object": "$object"})],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
    )
    locator = AbstractAtomicSkill(
        ref="skill://atomic_locate@1.0.0",
        summary="locate entity",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("object", "entity")],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$object", "location": "location"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
    )
    candidates = SupportAtomicRetriever().retrieve(
        blocked_atomic=blocked,
        missing_roles=["object"],
        atomics=[locator],
    )
    assert [item.atomic_ref for item in candidates] == [
        "skill://atomic_locate@1.0.0"
    ]
    assert candidates[0].supplied_roles == ("object",)


def test_runtime_agent_has_no_create_tool_and_has_automation_draft_tool() -> None:
    from atomic_skillgraph.runtime.node_executor import NodeExecutor

    atomic = AbstractAtomicSkill(
        ref="skill://atomic_take@1.0.0",
        summary="take",
        inputs=[ParameterSpec("item", "entity")],
        outputs=[],
        preconditions=[],
        effects=[SemanticPredicate("agent.holds", {"object": "$item"})],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
    )
    invocation_compiler = SimpleNamespace(skills=SimpleNamespace(atomics=lambda mode: []))
    executor = NodeExecutor(
        invocation_compiler, ValidationEngine(), lambda *_args: None,
    )
    names = {
        tool.name
        for tool in executor._node_tools(
            SimpleNamespace(action_catalog=[]), atomic,
        )
    }
    assert "propose_runtime_automation_atomic" in names
    assert "create_tool" not in names


def test_program_paths_cover_branches() -> None:
    report = program_paths([
        {
            "node_id": "choice",
            "op": "IF",
            "condition": {"source": "tool_input", "field": "x", "op": "exists"},
            "then_branch": [
                {"node_id": "then_action", "op": "ACTION", "action_type": "TAKE"},
            ],
            "else_branch": [],
        }
    ])
    assert report["path_ids"]
    assert any("then" in path_id for path_id in report["path_ids"])


def test_success_evolution_tool_builder_compiles_ir_tool(tmp_path: Any) -> None:
    from experiments.fakes import FakeAgentFactory, FakeReply
    from atomic_skillgraph.system import AtomicSkillGraphSystem

    from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal

    harness = FakeHarness()
    normalized = {
        "actions": [{
            "accepted": True,
            "action_id": "a0",
            "action_type": "TAKE",
            "arguments": {"item": "apple_1"},
            "before_revision": 0,
            "after_revision": 1,
            "event_id": "a0",
            "event_index": 0,
            "span_id": "span_1",
        }],
        "runtime_spans": [],
        "before_state_facts": [],
        "after_state_facts": [{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
            "revision": 1,
            "witness_ref": "fact:holds",
            "event_index": 0,
        }],
        "source_task": {"task_id": "task"},
        "trace_id": "trace_builder_test",
    }
    proposal = Atomicizer().validate_and_canonicalize(
        [
            AtomicOccurrenceProposal(
                phase_id="take",
                intent="take item",
                event_start=0,
                event_end=0,
                input_roles={"item": "apple_1"},
                output_roles={"held_object": "apple_1"},
                preconditions=[],
                effects=[
                    SemanticPredicate(
                        "agent.holds", {"object": "apple_1"},
                    )
                ],
                rationale="take",
            )
        ],
        normalized,
    )[0]
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    factory = FakeAgentFactory()
    session = factory.new_session(
        "tool_builder",
        [
            FakeReply.tool("create_tool", {
                "proposal_version": "1",
                "decision": "create",
                "summary": "take item",
                "atomic_ref": str(proposal.proposed_ref),
                "inputs": [{
                    "name": "item",
                    "semantic_type": "string",
                    "required": True,
                    "runtime_resolvable": True,
                    "required_resolution": "semantic",
                    "description": "",
                }],
                "outputs": [{
                    "name": "held_object",
                    "semantic_type": "entity",
                    "required": True,
                    "runtime_resolvable": False,
                    "required_resolution": "semantic",
                    "description": "",
                }],
                "program": [
                    {
                        "node_id": "take",
                        "op": "ACTION",
                        "action_type": "TAKE",
                        "argument_mapping": {
                            "item": {"kind": "skill_input", "source_role": "item"}
                        },
                    },
                    {
                        "node_id": "return_held",
                        "op": "RETURN",
                        "output_sources": {
                            "held_object": {"source": "tool_input", "field": "item"}
                        },
                    },
                ],
                "max_actions": 5,
                "final_effects": [{
                    "predicate": "agent.holds",
                    "args": {"object": "$item"},
                    "effect_domain": "world",
                }],
                "evidence_outputs": [{
                    "role": "held_object",
                    "source": "tool_input",
                    "field": "item",
                }],
                "path_expectations": [],
                "rationale": "single bounded take",
            })
        ],
    )
    system = object.__new__(AtomicSkillGraphSystem)
    system.config = {"llm": {}}
    system.usage = factory.usage_ledger
    system.tool_compiler = ToolCompiler()
    system.tool_static_validator = ToolStaticValidator()
    system.aligner = Aligner(skills, tools)
    system.skills = skills
    system.tools = tools
    system.harness = harness
    system._tool_builder_session = lambda _kind, _occurrence: session
    system._current_task_id = "task"
    trace = SimpleNamespace(trace_id="trace_builder_test", task=SimpleNamespace(task_id="task"))
    item, metrics = system._build_tool_for_occurrence(
        proposal, system._canonical_atomic_for_occurrence(proposal),
        normalized, trace,
    )
    assert metrics["call_count"] == 1
    assert metrics["static_pass_count"] == 1
    assert item is not None
    assert item.tool.artifact_kind == "tool_ir_v1"
    assert item.tool.artifact["program"][0]["op"] == "ACTION"
    assert item.implementation.tool_bindings[0].tool_ref == item.tool.ref
