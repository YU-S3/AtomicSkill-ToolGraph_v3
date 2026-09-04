"""Deterministic gates for the v3.2-R1 freeze document."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    ContractSource,
    EffectDomain,
    ParameterSpec,
    PlannerRequirementBundle,
    RequirementSearchResult,
    SemanticPredicate,
    TaskContract,
    ToolAsset,
)
from atomic_skillgraph.core.results import (
    PrimitiveToolStep,
    RuntimeLinearPlan,
    ToolExecutionResult,
)
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal
from atomic_skillgraph.harness.alfworld import (
    AlfWorldValidatorChannel,
    parse_alfworld_action,
)
from atomic_skillgraph.harness.protocol import (
    HarnessActionSpec,
    HarnessActionResult,
    HarnessTask,
)
from atomic_skillgraph.planner.repairability import RepairabilityGate
from atomic_skillgraph.runtime.automation import RuntimeAutomationCoordinator
from atomic_skillgraph.runtime.support_retriever import SupportAtomicRetriever
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.ir import (
    ToolExecutionState,
    resolve_collection,
)
from atomic_skillgraph.tooling.proposal import (
    RuntimeAutomationAtomicDraft,
    ToolProposal,
    ToolProvenance,
)
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.validation.engine import ValidationEngine
from atomic_skillgraph.validation.task_validator import TaskValidator
from experiments.fakes import (
    FakeAgentFactory,
    FakeHarness,
    FakeReply,
    fake_task,
)


def _atomic(name: str = "atomic_locate") -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=f"skill://{name}@1.0.0",
        summary=name,
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("found", "entity")],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$found", "location": "location"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.CANDIDATE,
    )


def test_gate1_won_only_terminal_authority() -> None:
    channel = SimpleNamespace(
        validate_task_contract=lambda _contract: SimpleNamespace(
            passed=False, witness_refs=[],
        ),
        won=True,
    )
    terminal = TaskValidator().terminal(
        TaskContract(
            [SemanticPredicate("object.heated", {"object": "cup"})]
        ),
        channel,
        benchmark_won=True,
    )
    assert terminal.passed is True
    assert terminal.checks == {"benchmark_won": True, "task_contract": False}
    assert terminal.failure_codes == []
    assert terminal.messages == ["benchmark_goal_contract_mismatch"]


def test_gate11_use_actual_result_does_not_toggle() -> None:
    channel = AlfWorldValidatorChannel()
    spec = HarnessActionSpec("use", 0, "USE", {"object": "lamp_1"}, "use lamp 1", "use lamp 1", {})
    for revision in (1, 2):
        channel.record(
            spec, accepted=True, revision=revision, done=False, won=False,
            observation="You turn on the lamp.",
        )
    facts = channel.snapshot()["facts"]
    assert {"predicate": "light.on", "args": {"light": "lamp_1"}} in facts
    assert {"predicate": "light.off", "args": {"light": "lamp_1"}} not in facts


def test_gate10_entity_discovered_at_from_catalog() -> None:
    channel = AlfWorldValidatorChannel()
    channel.set_catalog([
        HarnessActionSpec("take", 0, "TAKE", {"object": "cup_3", "source": "countertop_2"}, "", "", {}),
    ])
    facts = channel.snapshot()["facts"]
    assert {
        "predicate": "entity.discovered_at",
        "args": {"entity": "cup_3", "location": "countertop_2"},
    } in facts


def test_gate8_and_gate9_catalog_semantic_selectors() -> None:
    state = ToolExecutionState(catalog=[
        {"action_type": "GO_TO", "arguments": {"destination": "a"}},
        {"action_type": "GO_TO", "arguments": {"destination": "b"}},
        {"action_type": "TAKE", "arguments": {"object": "cup_3", "source": "b"}},
    ])
    values = resolve_collection(
        {
            "source": "action_catalog",
            "where": {"action_type": "GO_TO"},
            "project": {"kind": "argument", "role": "destination"},
            "distinct": True,
        },
        state,
        semantic_compatible=lambda **kwargs: True,
    )
    assert values == ["a", "b"]

    def compatible(**kwargs: Any) -> bool:
        assert kwargs["concrete_value"] == "cup_3"
        assert kwargs["semantic_anchor"] == "cup"
        return True

    values = resolve_collection(
        {
            "source": "action_catalog",
            "where": {
                "action_type": "TAKE",
                "argument_role": "object",
                "semantic_compatible_with": {
                    "source": "tool_input",
                    "field": "target",
                    "semantic_type": "entity",
                },
            },
            "project": {"kind": "argument", "role": "object"},
            "distinct": True,
        },
        replace_state := ToolExecutionState(
            bindings={"target": "cup"},
            catalog=state.catalog,
        ),
        semantic_compatible=compatible,
    )
    assert values == ["cup_3"]


class _MiniTrace:
    def __init__(self) -> None:
        self.environment_actions = []
        self.tool_executions = []


class _MiniBuilder:
    def __init__(self) -> None:
        self.trace = _MiniTrace()

    def start_span(self, _kind: str, _occurrence_id: str, *, parent_span_id: str | None = None) -> Any:
        return SimpleNamespace(span_id="span_mini")

    def finish_span(self, _span_id: str) -> None:
        pass


class _MiniBudget:
    def __init__(self) -> None:
        self.used = 0

    def consume_action(self) -> None:
        self.used += 1


class _MiniHarness:
    def __init__(self, *, accepted: bool = True, won_on: str = "") -> None:
        self.accepted = accepted
        self.won_on = won_on
        self.revision = 0
        self._catalog: list[HarnessActionSpec] = []

    def set_catalog(self, actions: list[dict[str, Any]]) -> None:
        self._catalog = [
            HarnessActionSpec(
                action_id=f"a{index}", revision=0, action_type=item["action_type"],
                arguments=item.get("arguments", {}), display_text="", raw_action="", metadata={},
            )
            for index, item in enumerate(actions)
        ]

    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog

    def compile_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> HarnessActionSpec:
        expected: dict[str, Any] = {}
        for role, expression in primitive.argument_mapping.items():
            expected[role] = (
                expression.constant
                if expression.kind is BindingExprKind.CONSTANT
                else bindings.get(expression.source_role)
            )
        for spec in self._catalog:
            if spec.action_type == primitive.action_type and all(
                spec.arguments.get(role) == value for role, value in expected.items()
            ):
                return spec
        raise KeyError(expected)

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        spec = next(item for item in self._catalog if item.action_id == action_id)
        self.revision += 1
        won = self.won_on == spec.action_type
        return HarnessActionResult(
            self.accepted, "ok", won, won, self.revision,
            [replace(item, revision=self.revision) for item in self._catalog],
            {"action_type": spec.action_type},
        )


class _MiniContext:
    def __init__(self, harness: _MiniHarness) -> None:
        self.harness = harness
        self.world_revision = 0
        self.action_catalog = harness.action_catalog()
        self.trace_builder = _MiniBuilder()
        self.budget = _MiniBudget()
        self.used_actions = 0
        self._facts: list[dict[str, Any]] = []

    def update_after_action(self, result: Any, record: dict[str, Any]) -> None:
        self.world_revision = result.new_revision
        self.action_catalog = list(result.catalog)
        self.used_actions = self.budget.used
        self._facts.append({
            "predicate": "agent.at_location",
            "args": {"location": record.get("arguments", {}).get("destination", "")},
        })

    def tool_evidence_snapshot(self) -> dict[str, Any]:
        return {
            "semantic_facts": [dict(item) for item in self._facts],
            "binding_evidence": [],
            "action_catalog": [
                {
                    "action_id": item.action_id,
                    "revision": item.revision,
                    "action_type": item.action_type,
                    "arguments": dict(item.arguments),
                }
                for item in self.action_catalog
            ],
            "revision": self.world_revision,
        }


def _mini_runner_context(actions: list[dict[str, Any]], **kwargs: Any) -> tuple[_MiniHarness, _MiniContext]:
    harness = _MiniHarness(**kwargs)
    harness.set_catalog(actions)
    return harness, _MiniContext(harness)


def _ir_tool(
    *,
    program: list[dict[str, Any]],
    max_actions: int,
    final_effects: list[dict[str, Any]],
    name: str = "tool",
) -> ToolAsset:
    return ToolAsset(
        ref=f"tool://{name}@1.0.0",
        summary=name,
        signature={
            "type": "object",
            "properties": {"loc": {"type": "string"}},
            "required": ["loc"],
        },
        interface={
            "output_schema": {
                "type": "object",
                "properties": {"found": {"type": "string"}},
                "required": ["found"],
                "additionalProperties": False,
            }
        },
        artifact_kind="tool_ir_v1",
        artifact={
            "schema_version": 1,
            "max_actions": max_actions,
            "program": program,
            "final_effects": final_effects,
            "evidence_outputs": [],
            "output_mapping": {
                "found": {"kind": "skill_input", "source_role": "loc"}
            },
        },
        tests=[],
        safety={"reviewed": True, "allowed_action_types": ["GO_TO"], "zero_llm": True},
        provenance={},
        metadata={},
        status=ToolStatus.ADMISSION_PENDING,
    )


class _MiniValidator:
    def __init__(self, pass_effects: bool = True) -> None:
        self.pass_effects = pass_effects

    def validate_atomic_effect(self, _request: dict[str, Any]) -> Any:
        return SimpleNamespace(passed=self.pass_effects)


def _mini_harness_channel(harness: _MiniHarness, *, pass_effects: bool = True) -> None:
    harness.validator = _MiniValidator(pass_effects)
    harness.validator_channel = lambda: harness.validator


def test_gate6_max_actions_is_runtime_authority() -> None:
    harness, ctx = _mini_runner_context([
        {"action_type": "GO_TO", "arguments": {"destination": "a"}},
        {"action_type": "GO_TO", "arguments": {"destination": "b"}},
    ])
    _mini_harness_channel(harness)
    tool = _ir_tool(
        program=[
            {
                "node_id": "go_a",
                "op": "ACTION",
                "action_type": "GO_TO",
                "argument_mapping": {
                    "destination": {"kind": "skill_input", "source_role": "loc"}
                },
            },
            {
                "node_id": "go_b",
                "op": "ACTION",
                "action_type": "GO_TO",
                "argument_mapping": {
                    "destination": {"kind": "skill_input", "source_role": "loc"}
                },
            },
            {
                "node_id": "ret",
                "op": "RETURN",
                "output_sources": {
                    "found": {"source": "local_variable", "field": "loc"}
                },
            },
        ],
        max_actions=1,
        final_effects=[{"predicate": "agent.at_location", "args": {}}],
    )
    result = ToolRunner(ValidationEngine().tool).run(
        tool, {"loc": "a"}, ctx, occurrence_id="occ",
    )
    assert result.failure_code == "tool_ir_max_actions_exhausted"
    assert result.executed_action_count == 1


def test_gate7_stop_when_breaks_loop_then_return_runs() -> None:
    harness, ctx = _mini_runner_context([
        {"action_type": "GO_TO", "arguments": {"destination": "a"}},
        {"action_type": "GO_TO", "arguments": {"destination": "b"}},
    ])
    _mini_harness_channel(harness)
    tool = _ir_tool(
        program=[
            {
                "node_id": "loop",
                "op": "FOR_EACH",
                "collection_source": {
                    "source": "local_deterministic",
                    "values": ["a", "b"],
                },
                "iteration_variable": "loc",
                "max_iterations": 5,
                "body": [
                    {
                        "node_id": "go",
                        "op": "ACTION",
                        "action_type": "GO_TO",
                        "argument_mapping": {
                            "destination": {"kind": "local_variable", "source_role": "loc"}
                        },
                    },
                    {
                        "node_id": "stop_a",
                        "op": "STOP_WHEN",
                        "condition": {
                            "source": "local_variable",
                            "field": "loc",
                            "op": "equals",
                            "value": "a",
                        },
                    },
                ],
            },
            {
                "node_id": "ret",
                "op": "RETURN",
                "output_sources": {
                    "found": {"source": "local_variable", "field": "loc"}
                },
            },
        ],
        max_actions=5,
        final_effects=[{"predicate": "agent.at_location", "args": {}}],
    )
    result = ToolRunner(ValidationEngine().tool).run(
        tool, {"loc": ""}, ctx, occurrence_id="occ",
    )
    assert result.completed is True
    assert result.output_candidates == {"found": "a"}
    assert result.executed_action_count == 1
    assert result.loop_iteration_counts.get("loop") == 1


def test_gate17_step_effect_violation_localized() -> None:
    harness, ctx = _mini_runner_context([
        {"action_type": "GO_TO", "arguments": {"destination": "a"}},
    ])
    _mini_harness_channel(harness, pass_effects=False)
    tool = _ir_tool(
        program=[
            {
                "node_id": "go",
                "op": "ACTION",
                "action_type": "GO_TO",
                "argument_mapping": {
                    "destination": {"kind": "constant", "constant": "a"}
                },
                "expected_effects": [
                    {"predicate": "agent.at_location", "args": {"location": "a"}}
                ],
            },
            {
                "node_id": "ret",
                "op": "RETURN",
                "output_sources": {
                    "found": {"source": "tool_input", "field": "loc"}
                },
            },
        ],
        max_actions=2,
        final_effects=[{"predicate": "agent.at_location", "args": {}}],
    )
    result = ToolRunner(ValidationEngine().tool).run(
        tool, {"loc": "a"}, ctx, occurrence_id="occ",
    )
    assert result.failure_code == "tool_step_effect_violation"
    assert result.program_node_id == "go"


def test_gate3_tool_ir_admission_and_gate4_replay_executes() -> None:
    harness = FakeHarness()
    task = fake_task("task-ir-admission", "apple_1")
    atomic = _atomic("atomic_take")
    atomic.effects = [SemanticPredicate("agent.holds", {"object": "$item"})]
    atomic.inputs = [ParameterSpec("item", "entity")]
    atomic.outputs = [ParameterSpec("held_object", "entity")]
    tool = ToolAsset(
        ref="tool://admission_ir@1.0.0",
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
                    "node_id": "ret",
                    "op": "RETURN",
                    "output_sources": {
                        "held_object": {"source": "tool_input", "field": "item"}
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
            "evidence_outputs": [],
            "output_mapping": {
                "held_object": {
                    "kind": "skill_input",
                    "source_role": "item",
                }
            },
        },
        tests=[{
            "kind": "tool_proposal_replay",
            "trace_id": "trace_ir",
            "occurrence_id": "occ",
            "bindings": {"item": "apple_1"},
            "source_task": {"task_id": task.task_id},
            "prefix": [],
            "effects": [],
        }],
        safety={"reviewed": True, "allowed_action_types": ["TAKE"], "zero_llm": True},
        provenance={"source": "success_evolution"},
        metadata={},
        status=ToolStatus.ADMISSION_PENDING,
    )
    executed = []

    def replay(candidate: ToolAsset, case: dict[str, Any]) -> bool:
        plan = RuntimeLinearPlan.full_dynamic(
            task.task_id, harness.task_contract(task), reason="replay_gate",
        )
        from atomic_skillgraph.runtime.budget import RuntimeBudget
        from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
        from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord
        trace = TraceRecord.create(
            TaskRecord(task.task_id, "fake", task.goal, task.task_type, "sig"),
            {}, {}, {"source": "full_dynamic"},
        )
        ctx = TaskRuntimeContext.create(
            task, plan, harness, TraceBuilder(trace),
            RuntimeBudget(global_action_budget=10),
        )
        result = ToolRunner(ValidationEngine().tool).run(
            candidate, dict(case.get("bindings") or {}), ctx,
            occurrence_id="replay",
        )
        executed.append(result.executed_action_count)
        return result.atomic_effect_passed and result.executed_action_count > 0

    admitted = Admission(ValidationEngine().tool).admit_tool(
        tool, replay=replay, atomic=atomic, harness=harness,
    )
    assert admitted.status is ToolStatus.CANDIDATE
    assert executed == [1]
    assert admitted.metadata["admission"]["kind"] == "tool_ir_v1"


def test_gate5_no_tool_compiles_atomic_only() -> None:
    from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
    from atomic_skillgraph.core.refs import SkillRef

    atomic = _atomic("atomic_no_tool")
    occurrence = CanonicalAtomicOccurrence(
        occurrence_id="occ", phase_id="p", intent="locate",
        event_start=0, event_end=0,
        input_bindings={"target": "cup"}, output_bindings={"found": "cup"},
        input_specs=atomic.inputs, output_specs=atomic.outputs,
        preconditions=[], effects=atomic.effects, action_events=[],
        prefix_events=[], source_task={"task_id": "t"}, source_trace_id="tr",
        proposed_ref=SkillRef("atomic_no_tool", "1.0.0"),
    )
    proposal = ToolProposal.no_tool(atomic_ref=str(atomic.ref), reason_code="no_reusable_tool")
    compiled = ToolCompiler().compile_proposal(
        occurrence, atomic, proposal,
        ToolProvenance("success_evolution", str(atomic.ref), "tr", "occ", task_id="t"),
    )
    assert compiled.tool is None
    assert compiled.implementation is None


def test_gate18_support_role_mapping_is_explicit() -> None:
    blocked = AbstractAtomicSkill(
        ref="skill://atomic_heat@1.0.0",
        summary="heat",
        inputs=[ParameterSpec("object", "entity")],
        outputs=[],
        preconditions=[],
        effects=[SemanticPredicate("object.heated", {"object": "$object"})],
        validator_spec={}, failure_modes=[], guideline={}, metadata={},
    )
    locator = AbstractAtomicSkill(
        ref="skill://atomic_locate@1.0.0",
        summary="locate",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("entity", "entity")],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "location"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        validator_spec={}, failure_modes=[], guideline={}, metadata={},
    )
    candidates = SupportAtomicRetriever().retrieve(
        blocked_atomic=blocked, missing_roles=["object"], atomics=[locator],
    )
    assert len(candidates) == 1
    assert candidates[0].role_mappings[0].producer_role == "entity"
    assert candidates[0].role_mappings[0].consumer_role == "object"


def test_gate19_repairability_uses_composite_hint() -> None:
    requirement = CapabilityRequirement(
        requirement_id="req_1", intent="find object",
        desired_effects=[SemanticPredicate("entity.discovered_at", {})],
        expected_inputs=[ParameterSpec("target", "entity")],
        expected_outputs=[],
        precondition_hints=[], semantic_variants=[], required=True,
        rationale="",
    )
    bundle = PlannerRequirementBundle([requirement])
    search = [
        RequirementSearchResult(
            requirement=requirement, candidates=[], covered=False,
            rejection_reasons=[], repair_hints=[],
        )
    ]
    hints = [{
        "composite_ref": "skill://composite_locate@1.0.0",
        "components": [{
            "atomic_ref": "skill://atomic_locate@1.0.0",
            "effects": [{"predicate": "entity.discovered_at"}],
            "inputs": [], "outputs": [],
        }],
        "effect_predicates": ["entity.discovered_at"],
    }]
    decision = RepairabilityGate().decide(
        bundle, SimpleNamespace(passed=True), search, hints,
    )
    assert decision.repairable is True
    assert decision.reason_code == "related_composite_interface_repairable"


def test_gate20_tool_builder_shares_remaining_runtime_budget() -> None:
    from atomic_skillgraph.agents.usage import (
        LLMUsage, UsageBucket, UsageEvent, UsageLedger,
    )
    from atomic_skillgraph.system import AtomicSkillGraphSystem

    system = object.__new__(AtomicSkillGraphSystem)
    system.config = {
        "llm": {
            "runtime": {"max_total_tokens_per_task": 300000},
            "extractor": {"max_total_tokens_per_task": 262144},
            "tool_builder": {},
        }
    }
    system.usage = UsageLedger()
    system.usage.append(UsageEvent(
        event_id="evt_1", session_id="s", turn_index=0,
        bucket=UsageBucket.RUNTIME_DYNAMIC,
        usage=LLMUsage(total_tokens=290000, call_count=1),
    ))
    assert system._shared_tool_builder_tokens("tool_builder_runtime") == 10000
    system._current_task_usage_start = 1
    assert system._shared_tool_builder_tokens("tool_builder_runtime") == 300000


def test_gate21_tool_builder_context_has_no_provenance_leakage() -> None:
    from atomic_skillgraph.agents.context_builder import ContextBuilder

    atomic = _atomic()
    provenance = SimpleNamespace(
        source="success_evolution", task_id="task_secret", trace_id="trace_secret",
        occurrence_id="occ_secret", draft_id="",
    )
    prompt = ContextBuilder().tool_builder(
        atomic=atomic,
        provenance=provenance,
        evidence_support=[],
        semantic_delta={"semantic_facts": [], "revision": 1},
        harness_interface={"primitive_actions": []},
    )
    assert "task_secret" not in prompt
    assert "trace_secret" not in prompt
    assert "occ_secret" not in prompt
    assert "source_kind" in prompt


def test_gate12_runtime_automation_input_binding_specs_resolve() -> None:
    class Bindings:
        def snapshot_for_node(self, _occurrence: Any) -> dict[str, Any]:
            return {}

        def semantic_anchor_for(self, occurrence: Any, role: str) -> Any:
            return SimpleNamespace(value="cup")

        def validated_outputs(self, _occurrence_id: str) -> dict[str, Any]:
            return {}

    ctx = SimpleNamespace(
        binding_store=Bindings(), validated_outputs={},
    )
    draft = RuntimeAutomationAtomicDraft(
        draft_id="d", intent="locate target",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("found", "entity")],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at", {"entity": "$found", "location": "location"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        rationale="", source_occurrence_id="occ",
        input_binding_specs={
            "target": {
                "kind": "current_occurrence_anchor",
                "source_role": "object",
            }
        },
    )
    bindings = RuntimeAutomationCoordinator._resolve_trial_bindings(
        draft, ctx, SimpleNamespace(occurrence_id="occ"),
    )
    assert bindings == {"target": "cup"}


def _atomicizer_trace(
    *,
    child_occurrence: str = "occ_1",
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if actions is None:
        actions = [
            {
                "accepted": True, "action_id": "e0", "action_type": "TAKE",
                "arguments": {"item": "apple_1"}, "before_revision": 0,
                "after_revision": 1, "event_id": "e0", "event_index": 0,
                "span_id": "parent",
            },
            {
                "accepted": True, "action_id": "e1", "action_type": "EXAMINE",
                "arguments": {"item": "apple_1"}, "before_revision": 1,
                "after_revision": 2, "event_id": "e1", "event_index": 1,
                "span_id": "child",
            },
        ]
    return {
        "actions": actions,
        "runtime_spans": [
            {
                "span_id": "parent", "kind": "runtime", "occurrence_id": "occ_1",
                "action_start": 0, "action_end": len(actions), "parent_span_id": None,
                "learnable": True,
            },
            {
                "span_id": "child", "kind": "tool", "occurrence_id": child_occurrence,
                "action_start": 1, "action_end": len(actions), "parent_span_id": "parent",
                "learnable": True,
            },
        ],
        "before_state_facts": [],
        "after_state_facts": [
            {
                "predicate": "object.observed", "args": {"object": "apple_1"},
                "revision": 2, "witness_ref": "effect:w1",
            }
        ],
        "source_task": {"task_id": "task_atomicizer"},
        "trace_id": "trace_atomicizer",
    }


def _atomicizer_proposal(
    phase: str,
    *,
    start: int,
    end: int,
    support: list[str],
    obj: str = "apple_1",
    effect_refs: list[str] | None = None,
) -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id=phase, intent=f"observe {obj}",
        event_start=start, event_end=end,
        input_roles={"item": obj},
        output_roles={"result": obj},
        preconditions=[],
        effects=[SemanticPredicate("object.observed", {"object": obj})],
        rationale="observed",
        support_event_ids=support,
        precondition_witness_refs=[],
        effect_witness_refs=effect_refs or [],
    )


def test_gate14_cross_nested_span_same_occurrence_passes() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    canonical = Atomicizer().validate_and_canonicalize(
        [_atomicizer_proposal("p", start=0, end=1, support=["e0", "e1"])],
        _atomicizer_trace(),
    )
    assert len(canonical) == 1


def test_gate14_cross_occurrence_lineage_rejected() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    with pytest.raises(ValueError, match="lineage|crosses incompatible"):
        Atomicizer().validate_and_canonicalize(
            [_atomicizer_proposal("p", start=0, end=1, support=["e0", "e1"])],
            _atomicizer_trace(child_occurrence="occ_2"),
        )


def test_gate15_effect_witness_ref_must_exist_and_match() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    Atomicizer().validate_and_canonicalize(
        [_atomicizer_proposal("p", start=0, end=1, support=["e1"], effect_refs=["effect:w1"])],
        _atomicizer_trace(),
    )
    with pytest.raises(ValueError, match="evidence_witness_ref_invalid"):
        Atomicizer().validate_and_canonicalize(
            [_atomicizer_proposal("p", start=0, end=1, support=["e1"], effect_refs=["effect:missing"])],
            _atomicizer_trace(),
        )


def test_gate16_envelope_overlap_allowed_when_support_events_disjoint() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    actions = [
        {
            "accepted": True, "action_id": "e0", "action_type": "EXAMINE",
            "arguments": {"item": "apple_1"}, "before_revision": 0,
            "after_revision": 1, "event_id": "e0", "event_index": 0,
            "span_id": "parent",
        },
        {
            "accepted": True, "action_id": "e1", "action_type": "EXAMINE",
            "arguments": {"item": "mug_1"}, "before_revision": 1,
            "after_revision": 2, "event_id": "e1", "event_index": 1,
            "span_id": "parent",
        },
        {
            "accepted": True, "action_id": "e2", "action_type": "EXAMINE",
            "arguments": {"item": "mug_1"}, "before_revision": 2,
            "after_revision": 3, "event_id": "e2", "event_index": 2,
            "span_id": "parent",
        },
    ]
    normalized = _atomicizer_trace(actions=actions)
    normalized["after_state_facts"] = [
        {
            "predicate": "object.observed", "args": {"object": "apple_1"},
            "revision": 1, "witness_ref": "effect:a",
        },
        {
            "predicate": "object.observed", "args": {"object": "mug_1"},
            "revision": 3, "witness_ref": "effect:b",
        },
    ]
    canonical = Atomicizer().validate_and_canonicalize(
        [
            _atomicizer_proposal("a", start=0, end=1, support=["e0"], obj="apple_1"),
            _atomicizer_proposal("b", start=1, end=2, support=["e2"], obj="mug_1"),
        ],
        normalized,
    )
    assert len(canonical) == 2
    with pytest.raises(ValueError, match="already owned"):
        Atomicizer().validate_and_canonicalize(
            [
                _atomicizer_proposal("a", start=0, end=1, support=["e1"], obj="mug_1"),
                _atomicizer_proposal("b", start=1, end=2, support=["e1"], obj="mug_1"),
            ],
            normalized,
        )


def _terminal_canonical_occurrence() -> Any:
    from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
    from atomic_skillgraph.core.refs import SkillRef

    return CanonicalAtomicOccurrence(
        occurrence_id="occ_1",
        phase_id="p1",
        intent="observe target",
        event_start=0,
        event_end=0,
        input_bindings={"item": "cup_1"},
        output_bindings={"result": "cup_1"},
        input_specs=[ParameterSpec("item", "entity")],
        output_specs=[ParameterSpec("result", "entity")],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "object.observed",
                {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="item")},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        action_events=[],
        prefix_events=[],
        source_task={"task_id": "task_terminal"},
        source_trace_id="trace_terminal",
        proposed_ref=SkillRef("atomic_observe_terminal", "1.0.0"),
    )


def test_gate23_terminal_empirical_composite_candidate_created_without_forgery() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal

    canonical = [_terminal_canonical_occurrence()]
    contract = TaskContract(
        [
            SemanticPredicate(
                "object.observed", {"object": "cup_1"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
            SemanticPredicate("object.at_location", {"object": "cup_1"}),
        ]
    )
    composite = CompositeBuilder().validate_and_build(
        CompositeExtractionProposal(
            control_sequence=["occ_1"],
            existing_edges=[],
            new_edges=[],
            summary="observe target prefix",
            guideline={},
            insight={},
        ),
        canonical,
        contract,
        task_bindings={"item": "cup_1"},
        terminal_certificate={
            "benchmark_won": True,
            "source_trace_id": "trace_terminal",
            "terminal_revision": 3,
            "executed_occurrence_ids": ["occ_1"],
            "skipped_planned_occurrence_ids": ["occ_2"],
            "observed_task_contract_coverage": {
                "covered_effects": ["object.observed"],
                "uncovered_effects": ["object.at_location"],
            },
        },
        source_composite_ref="skill://composite_longer@1.0.0",
    )
    assert composite.status is SkillStatus.CANDIDATE
    assert composite.metadata["completion_authority"]["kind"] == "terminal_empirical"
    assert composite.metadata["terminal_certificate"]["benchmark_won"] is True
    assert composite.metadata["terminal_certificate"]["skipped_planned_occurrence_ids"] == ["occ_2"]
    assert composite.metadata["observed_task_contract_coverage"]["uncovered_effects"] == [
        "object.at_location"
    ]
    assert all(
        effect.predicate != "object.at_location"
        for occurrence in canonical
        for effect in occurrence.effects
    )


def test_gate24_and_gate25_terminal_candidate_separate_retrieval_channel() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.planner.composite_retriever import CompositeRetriever

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        database = StateDatabase(_Path(tmp) / "state.sqlite3")
        artifacts = ArtifactStore(_Path(tmp), database)
        skills = SkillRegistry(artifacts, database)
        canonical = [_terminal_canonical_occurrence()]
        atomic_artifact = _atomic("atomic_observe_terminal")
        skills.register_atomic(replace(atomic_artifact, ref=canonical[0].proposed_ref))
        contract = TaskContract(
            [
                SemanticPredicate(
                    "object.observed", {"object": "cup_1"},
                    effect_domain=EffectDomain.EVIDENCE,
                ),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]
        )
        terminal_candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"],
                existing_edges=[], new_edges=[],
                summary="observe target prefix", guideline={}, insight={},
            ),
            canonical,
            contract,
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True,
                "source_trace_id": "trace_terminal",
                "terminal_revision": 3,
                "executed_occurrence_ids": ["occ_1"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        skills.register_composite(terminal_candidate)
        # A terminal candidate is never returned as a complete-contract P0 hit.
        complete = CompositeRetriever(skills).retrieve_complete(
            SimpleNamespace(task_id="task_terminal", goal="observe"),
            contract, mode="online", harness_profile="fake_v3",
        )
        assert complete.candidates == []
        terminal = CompositeRetriever(skills).retrieve_terminal(
            SimpleNamespace(task_id="task_terminal", goal="observe"),
            contract, mode="online", harness_profile="fake_v3",
        )
        assert [str(item.ref) for item in terminal.terminal_empirical_candidates] == [
            str(terminal_candidate.ref)
        ]
        assert terminal.terminal_empirical_audit[0]["completion_authority"] == "terminal_empirical"


def test_gate30_fresh_output_effect_witness_mismatch_fail_closed() -> None:
    from atomic_skillgraph.validation.atomic_validator import AtomicValidator
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from atomic_skillgraph.core.refs import SkillRef

    atomic = AbstractAtomicSkill(
        ref=SkillRef("atomic_locate_gate30", "1.0.0"),
        summary="locate fresh entity",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[
            ParameterSpec("entity", "entity"),
            ParameterSpec("location", "entity"),
        ],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {
                    "entity": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="entity"),
                    "location": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="location"),
                },
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        validator_spec={
            "output_derivations": {
                "entity": {
                    "kind": "effect_witness",
                    "predicate": "entity.discovered_at",
                    "argument_role": "entity",
                },
                "location": {
                    "kind": "effect_witness",
                    "predicate": "entity.discovered_at",
                    "argument_role": "location",
                },
            }
        },
        failure_modes=[], guideline={}, metadata={}, status=SkillStatus.CANDIDATE,
    )
    channel = AlfWorldValidatorChannel()
    channel.set_catalog([
        HarnessActionSpec(
            "take", 0, "TAKE",
            {"object": "cup_3", "source": "countertop_2"}, "", "", {},
        )
    ])
    occurrence = RuntimeOccurrence(
        "s1", "occ1", atomic.ref, [], {}, [], list(atomic.effects),
    )
    authority_facts = [{
        "predicate": "entity.discovered_at",
        "args": {"entity": "cup_3", "location": "countertop_2"},
    }]
    wrong = AtomicValidator().validate_execution_result(
        atomic, occurrence, {"target": "cup"},
        {"entity": "mug_2", "location": "countertop_2"},
        channel, current_revision=0,
        authoritative_evidence_facts=authority_facts,
    )
    assert wrong.passed is False
    assert "atomic_output_effect_witness_mismatch" in wrong.failure_codes
    right = AtomicValidator().validate_execution_result(
        atomic, occurrence, {"target": "cup"},
        {"entity": "cup_3", "location": "countertop_2"},
        channel, current_revision=0,
        authoritative_evidence_facts=authority_facts,
    )
    assert right.passed is True


def test_gate31_fake_input_authority_rejected() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    normalized = _atomicizer_trace()
    normalized["boundary_authorities"] = {"inputs": [], "effects": []}
    proposal = _atomicizer_proposal("p", start=0, end=1, support=["e1"])
    proposal.input_roles = {"target": "cup"}
    proposal.output_roles = {"result": "cup"}
    proposal.output_derivations = {
        "result": {"kind": "input_identity", "input_role": "target"}
    }
    proposal.input_provenance_refs = {
        "target": "runtime_input:fake:target"
    }
    with pytest.raises(ValueError, match="input authority ref not found"):
        Atomicizer().validate_and_canonicalize([proposal], normalized)


def test_gate32_raw_observation_cannot_create_fresh_output() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    normalized = _atomicizer_trace()
    normalized["actions"][1]["observation_text"] = "You see cup 3 on countertop 2"
    normalized["boundary_authorities"] = {
        "inputs": [{
            "authority_ref": "runtime_input:d:target",
            "role": "target", "value": "cup",
            "source_kind": "current_occurrence_anchor",
        }],
        "effects": [],
    }
    proposal = _atomicizer_proposal("p", start=0, end=1, support=["e1"])
    proposal.input_roles = {"target": "cup"}
    proposal.output_roles = {"entity": "cup_3"}
    proposal.input_provenance_refs = {
        "target": "runtime_input:d:target"
    }
    proposal.output_derivations = {
        "entity": {
            "kind": "effect_witness",
            "predicate": "entity.discovered_at",
            "argument_role": "entity",
        }
    }
    proposal.effects = [
        SemanticPredicate(
            "entity.discovered_at",
            {"entity": "$entity", "location": "countertop_2"},
            effect_domain=EffectDomain.EVIDENCE,
        )
    ]
    with pytest.raises(ValueError):
        Atomicizer().validate_and_canonicalize([proposal], normalized)


def test_gate33_recursive_concrete_id_leakage_rejected() -> None:
    atomic = _atomic("atomic_nested_leak")
    proposal = ToolProposal(
        proposal_version="1", decision="create", summary="nested leak",
        atomic_ref=str(atomic.ref), inputs=atomic.inputs, outputs=atomic.outputs,
        program=[
            {
                "node_id": "loop", "op": "FOR_EACH",
                "collection_source": {
                    "source": "local_deterministic",
                    "values": ["cabinet_7"],
                },
                "iteration_variable": "loc", "max_iterations": 1,
                "body": [{
                    "node_id": "go", "op": "ACTION", "action_type": "GO_TO",
                    "argument_mapping": {
                        "destination": {
                            "kind": "constant", "constant": "cabinet_7",
                        }
                    },
                }],
            },
            {
                "node_id": "ret", "op": "RETURN",
                "output_sources": {
                    "found": {"source": "tool_input", "field": "target"}
                },
            },
        ],
        max_actions=2,
        final_effects=atomic.effects,
        evidence_outputs=[], path_expectations=[], rationale="",
    )
    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )
    assert report.passed is False
    assert "tool_ir_episode_concrete_id" in report.failure_codes


def test_gate34_recursive_safety_action_collection() -> None:
    from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
    from atomic_skillgraph.core.refs import SkillRef

    atomic = _atomic("atomic_nested_safety")
    atomic.ref = SkillRef.parse(str(atomic.ref))
    atomic.effects = [
        SemanticPredicate("entity.discovered_at", {"entity": "$found"})
    ]
    occurrence = CanonicalAtomicOccurrence(
        occurrence_id="occ", phase_id="p", intent="nested search",
        event_start=0, event_end=0,
        input_bindings={"target": "cup"}, output_bindings={"found": "cup_3"},
        input_specs=atomic.inputs, output_specs=atomic.outputs,
        preconditions=[], effects=atomic.effects, action_events=[],
        prefix_events=[], source_task={"task_id": "t"}, source_trace_id="tr",
        proposed_ref=SkillRef("atomic_nested_safety", "1.0.0"),
    )
    proposal = ToolProposal(
        proposal_version="1", decision="create", summary="nested search",
        atomic_ref=str(atomic.ref), inputs=atomic.inputs, outputs=atomic.outputs,
        program=[
            {
                "node_id": "loop", "op": "FOR_EACH",
                "collection_source": {
                    "source": "local_deterministic", "values": ["a"],
                },
                "iteration_variable": "loc", "max_iterations": 1,
                "body": [
                    {
                        "node_id": "go", "op": "ACTION", "action_type": "GO_TO",
                        "argument_mapping": {
                            "destination": {
                                "kind": "local_variable", "source_role": "loc",
                            }
                        },
                    },
                    {
                        "node_id": "choice", "op": "IF",
                        "condition": {
                            "source": "local_variable", "field": "loc",
                            "op": "exists",
                        },
                        "then_branch": [{
                            "node_id": "open", "op": "ACTION",
                            "action_type": "OPEN",
                            "argument_mapping": {
                                "object": {
                                    "kind": "local_variable", "source_role": "loc",
                                }
                            },
                        }],
                        "else_branch": [],
                    },
                ],
            },
            {
                "node_id": "ret", "op": "RETURN",
                "output_sources": {
                    "found": {"source": "tool_input", "field": "target"}
                },
            },
        ],
        max_actions=2,
        final_effects=atomic.effects,
        evidence_outputs=[], path_expectations=[], rationale="",
    )
    compiled = ToolCompiler().compile_proposal(
        occurrence, atomic, proposal,
        ToolProvenance("success_evolution", str(atomic.ref), "tr", "occ"),
    )
    assert compiled.tool is not None
    assert set(compiled.tool.safety.get("allowed_action_types") or []) == {
        "GO_TO", "OPEN",
    }


def test_gate36_terminal_empirical_current_contract_subset_gate() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.planner.composite_retriever import CompositeRetriever

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        database = StateDatabase(_Path(tmp) / "state.sqlite3")
        artifacts = ArtifactStore(_Path(tmp), database)
        skills = SkillRegistry(artifacts, database)
        canonical = [_terminal_canonical_occurrence()]
        skills.register_atomic(replace(
            _atomic("atomic_observe_terminal"),
            ref=canonical[0].proposed_ref,
        ))
        candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"],
                existing_edges=[], new_edges=[],
                summary="observe prefix", guideline={}, insight={},
            ),
            canonical,
            TaskContract([
                SemanticPredicate(
                    "object.observed", {"object": "cup_1"},
                    effect_domain=EffectDomain.EVIDENCE,
                ),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]),
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True,
                "source_trace_id": "tr",
                "terminal_revision": 1,
                "executed_occurrence_ids": ["occ_1"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        skills.register_composite(candidate)
        retriever = CompositeRetriever(skills)
        eligible = retriever.retrieve_terminal(
            SimpleNamespace(task_id="t_clean", goal="clean"),
            TaskContract([
                SemanticPredicate(
                    "object.observed", {"object": "cup_1"},
                    effect_domain=EffectDomain.EVIDENCE,
                ),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]),
            mode="online", harness_profile="fake_v3",
        )
        assert len(eligible.terminal_empirical_candidates) == 1
        ineligible = retriever.retrieve_terminal(
            SimpleNamespace(task_id="t_heat", goal="heat"),
            TaskContract([
                SemanticPredicate("object.heated", {"object": "cup_1"}),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]),
            mode="online", harness_profile="fake_v3",
        )
        assert ineligible.terminal_empirical_candidates == []


def test_r2_2_semantic_evidence_selector_and_return() -> None:
    from atomic_skillgraph.tooling.ir import resolve_collection, resolve_return_sources

    state = ToolExecutionState(semantic_facts=[
        {
            "predicate": "entity.discovered_at",
            "args": {"entity": "cup_3", "location": "countertop_2"},
        }
    ])
    values = resolve_collection({
        "source": "semantic_evidence",
        "where": {"predicate": "entity.discovered_at"},
        "project": {"kind": "argument", "role": "entity"},
    }, state)
    assert values == ["cup_3"]
    outputs, refs = resolve_return_sources({
        "entity": {
            "source": "semantic_evidence",
            "where": {"predicate": "entity.discovered_at"},
            "project": {"kind": "argument", "role": "entity"},
        }
    }, state)
    assert outputs == {"entity": "cup_3"}
    assert refs


def test_r2_3_terminal_empirical_all_signatures_subset() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.planner.composite_retriever import CompositeRetriever

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        database = StateDatabase(_Path(tmp) / "state.sqlite3")
        artifacts = ArtifactStore(_Path(tmp), database)
        skills = SkillRegistry(artifacts, database)
        canonical = [_terminal_canonical_occurrence()]
        skills.register_atomic(replace(
            _atomic("atomic_observe_terminal"), ref=canonical[0].proposed_ref,
        ))
        candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"], existing_edges=[], new_edges=[],
                summary="observe prefix", guideline={}, insight={},
            ),
            canonical,
            TaskContract([
                SemanticPredicate("object.observed", {"object": "cup_1"}),
            ]),
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True, "source_trace_id": "tr",
                "terminal_revision": 1,
                "executed_occurrence_ids": ["occ_1"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        # Inject a second historical signature that is incompatible with a
        # clean+at_location current task.
        metadata = dict(candidate.metadata)
        certificate = dict(metadata.get("terminal_certificate") or {})
        certificate["covered_effect_signatures"] = [
            {
                "predicate": "object.observed", "effect_domain": "evidence",
                "argument_roles": ["object"], "cardinality": 1, "distinct_by": "",
            },
            {
                "predicate": "object.observed", "effect_domain": "world",
                "argument_roles": ["object", "location"],
                "cardinality": 1, "distinct_by": "",
            },
        ]
        metadata["terminal_certificate"] = certificate
        candidate = replace(candidate, metadata=metadata)
        skills.register_composite(candidate)
        retriever = CompositeRetriever(skills)
        ineligible = retriever.retrieve_terminal(
            SimpleNamespace(task_id="t_clean", goal="clean"),
            TaskContract([
                SemanticPredicate("object.cleaned", {"object": "cup_1"}),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]),
            mode="online", harness_profile="fake_v3",
        )
        assert ineligible.terminal_empirical_candidates == []


def test_r2_4_tool_builder_boundary_exactness() -> None:
    from atomic_skillgraph.core.refs import SkillRef

    atomic = _atomic("atomic_boundary")
    atomic.ref = SkillRef.parse(str(atomic.ref))
    atomic.effects = [
        SemanticPredicate("object.observed", {"object": "$found"})
    ]
    base = ToolProposal(
        proposal_version="1", decision="create", summary="locate",
        atomic_ref=str(atomic.ref),
        inputs=list(atomic.inputs),
        outputs=list(atomic.outputs),
        program=[
            {
                "node_id": "ret", "op": "RETURN",
                "output_sources": {
                    "found": {"source": "tool_input", "field": "target"}
                },
            }
        ],
        max_actions=1,
        final_effects=atomic.effects,
        evidence_outputs=[], path_expectations=[], rationale="",
    )
    assert ToolStaticValidator().validate_proposal(
        base, atomic, FakeHarness(),
    ).passed
    changed = replace(base, inputs=[ParameterSpec("semantic_target", "entity")])
    report = ToolStaticValidator().validate_proposal(
        changed, atomic, FakeHarness(),
    )
    assert report.passed is False
    assert "tool_builder_atomic_boundary_mismatch" in report.failure_codes
    extra = replace(base, inputs=[*base.inputs, ParameterSpec("current_location", "entity")])
    assert ToolStaticValidator().validate_proposal(
        extra, atomic, FakeHarness(),
    ).passed is False


def test_r21_e1_output_derivation_type_vocabulary_tolerated() -> None:
    from atomic_skillgraph.evolution.atomicizer import Atomicizer

    proposal = _atomicizer_proposal("p", start=0, end=1, support=["e0", "e1"])
    # The E1 transport may spell the same derivation vocabulary as "type".
    proposal.output_derivations = {
        "result": {"type": "INPUT_IDENTITY", "input_role": "item"}
    }
    canonical = Atomicizer().validate_and_canonicalize(
        [proposal], _atomicizer_trace(),
    )
    assert len(canonical) == 1
    assert canonical[0].output_derivations["result"]["kind"] == "input_identity"


def _locate_draft() -> RuntimeAutomationAtomicDraft:
    return RuntimeAutomationAtomicDraft(
        draft_id="draft_locate", intent="locate target",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[
            ParameterSpec("entity", "entity"),
            ParameterSpec("location", "location"),
        ],
        preconditions=[],
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "$location"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
        rationale="", source_occurrence_id="occ",
        input_binding_specs={
            "target": {"kind": "current_occurrence_anchor", "source_role": "object"}
        },
    )


def test_r21_runtime_automation_output_derivations_enter_validate_execution_result() -> None:
    from atomic_skillgraph.tooling.validator import (
        normalize_runtime_output_derivations,
    )
    from atomic_skillgraph.validation.atomic_validator import AtomicValidator
    from atomic_skillgraph.core.results import RuntimeOccurrence

    derivations = normalize_runtime_output_derivations(_locate_draft())
    assert derivations == {
        "entity": {
            "kind": "effect_witness",
            "predicate": "entity.discovered_at",
            "argument_role": "entity",
        },
        "location": {
            "kind": "effect_witness",
            "predicate": "entity.discovered_at",
            "argument_role": "location",
        },
    }
    atomic = RuntimeAutomationCoordinator._draft_atomic(
        _locate_draft(), occurrence_id="occ", trace_id="tr",
    )
    assert atomic.validator_spec["output_derivations"] == derivations
    # The production branch switch is exactly this condition; prove it is on
    # and that the task-local R1 authority resolves through effect witnesses.
    assert bool(atomic.validator_spec.get("output_derivations"))

    channel = AlfWorldValidatorChannel()
    channel.set_catalog([
        HarnessActionSpec(
            "take", 0, "TAKE", {"object": "cup_3", "source": "countertop_2"},
            "", "", {},
        )
    ])
    occurrence = RuntimeOccurrence(
        "s1", "occ1", atomic.ref, [], {}, [], list(atomic.effects),
    )
    result = AtomicValidator().validate_execution_result(
        atomic,
        occurrence,
        {"target": "cup"},
        {"entity": "cup_3", "location": "countertop_2"},
        channel,
        current_revision=0,
        authoritative_evidence_facts=[{
            "predicate": "entity.discovered_at",
            "args": {"entity": "cup_3", "location": "countertop_2"},
        }],
    )
    assert result.passed is True


def test_r21_runtime_automation_output_derivation_fail_closed() -> None:
    from atomic_skillgraph.tooling.validator import (
        normalize_runtime_output_derivations,
    )

    no_derivation = replace(
        _locate_draft(),
        effects=[
            SemanticPredicate(
                "object.observed", {"object": "cup"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
    )
    with pytest.raises(ValueError, match="runtime_automation_r0_output_derivation_invalid"):
        normalize_runtime_output_derivations(no_derivation)
    multiple = replace(
        _locate_draft(),
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "$location"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
            SemanticPredicate(
                "entity.observed_at",
                {"entity": "$entity", "location": "$location"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
        ],
    )
    with pytest.raises(ValueError, match="multiple Effect witness authorities"):
        normalize_runtime_output_derivations(multiple)
    undeclared = replace(
        _locate_draft(),
        effects=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": "$entity", "location": "$missing_role"},
                effect_domain=EffectDomain.EVIDENCE,
            )
        ],
    )
    with pytest.raises(ValueError, match="undeclared output role"):
        normalize_runtime_output_derivations(undeclared)
    r0 = ToolStaticValidator().validate_automation_draft(
        undeclared, FakeHarness(),
    )
    assert r0.passed is False
    assert "runtime_automation_r0_output_derivation_invalid" in r0.failure_codes


def _control_step_tool(
    *, program: list[dict[str, Any]], max_actions: int,
) -> Any:
    from atomic_skillgraph.core.contracts import ToolAsset

    return ToolAsset(
        ref="tool://tool_control_step@1.0.0",
        summary="control-step probe",
        signature={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        interface={
            "output_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        },
        artifact_kind="tool_ir_v1",
        artifact={
            "schema_version": 1,
            "max_actions": max_actions,
            "program": program,
            "final_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "$x"},
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


def _run_control_step_tool(tool: Any) -> ToolExecutionResult:
    from atomic_skillgraph.runtime.budget import RuntimeBudget
    from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
    from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord

    harness = FakeHarness()
    task = fake_task("task-control-step", "apple_1")
    harness.reset(task)
    plan = RuntimeLinearPlan.full_dynamic(
        task.task_id, harness.task_contract(task), reason="control_step",
    )
    trace = TraceRecord.create(
        TaskRecord(task.task_id, task.benchmark, task.goal, task.task_type, "sig"),
        {}, {}, {"source": "full_dynamic"},
    )
    ctx = TaskRuntimeContext.create(
        task, plan, harness, TraceBuilder(trace),
        RuntimeBudget(global_action_budget=100, node_action_budget=35),
    )
    return ToolRunner(ValidationEngine().tool).run(
        tool, {"target": "apple_1"}, ctx, occurrence_id="control_step",
    )


def test_r21_tool_ir_control_step_exhaustion_bound() -> None:
    nested = _control_step_tool(
        program=[
            {
                "node_id": "outer", "op": "FOR_EACH",
                "collection_source": {
                    "source": "local_deterministic",
                    "values": ["a", "b", "c", "d", "e", "f", "g", "h"],
                },
                "iteration_variable": "outer_v", "max_iterations": 8,
                "body": [{
                    "node_id": "inner", "op": "FOR_EACH",
                    "collection_source": {
                        "source": "local_deterministic",
                        "values": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    },
                    "iteration_variable": "inner_v", "max_iterations": 8,
                    "body": [{
                        "node_id": "check", "op": "IF",
                        "condition": {
                            "source": "local_variable",
                            "field": "inner_v",
                            "op": "exists",
                        },
                        "then_branch": [], "else_branch": [],
                    }],
                }],
            },
            {
                "node_id": "ret", "op": "RETURN",
                "output_sources": {"x": {"source": "tool_input", "field": "target"}},
            },
        ],
        max_actions=8,
    )
    result = _run_control_step_tool(nested)
    assert result.completed is False
    assert result.failure_code == "tool_ir_control_step_exhausted"
    evidence = dict(result.tool_path_evidence or {})
    assert evidence.get("control_step_limit", 0) > 0
    assert (
        evidence.get("control_step_count", 0)
        > evidence.get("control_step_limit", 0)
    )

    bounded = _control_step_tool(
        program=[
            {
                "node_id": "search", "op": "FOR_EACH",
                "collection_source": {
                    "source": "local_deterministic",
                    "values": ["room_a", "room_b"],
                },
                "iteration_variable": "loc", "max_iterations": 2,
                "body": [{
                    "node_id": "stop", "op": "STOP_WHEN",
                    "condition": {
                        "source": "local_variable", "field": "loc", "op": "exists",
                    },
                }],
            },
            {
                "node_id": "ret", "op": "RETURN",
                "output_sources": {"x": {"source": "tool_input", "field": "target"}},
            },
        ],
        max_actions=2,
    )
    result = _run_control_step_tool(bounded)
    assert result.completed is True
    assert result.failure_code == ""


def test_r21_terminal_empirical_support_effects_excluded_from_signatures() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.planner.composite_retriever import CompositeRetriever
    from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
    from atomic_skillgraph.core.refs import SkillRef

    import tempfile
    from pathlib import Path as _Path

    def occurrence(
        occurrence_id: str, input_role: str, output_role: str, predicate: str,
        effect_domain: EffectDomain, value: str,
    ) -> Any:
        return CanonicalAtomicOccurrence(
            occurrence_id=occurrence_id, phase_id=occurrence_id,
            intent=f"{predicate} {occurrence_id}",
            event_start=0, event_end=0,
            input_bindings={input_role: value},
            output_bindings={output_role: value},
            input_specs=[ParameterSpec(input_role, "entity")],
            output_specs=[ParameterSpec(output_role, "entity")],
            preconditions=[],
            effects=[
                SemanticPredicate(
                    predicate,
                    {
                        "object": BindingExpression(
                            BindingExprKind.SKILL_INPUT,
                            source_role=input_role,
                        )
                    },
                    effect_domain=effect_domain,
                )
            ],
            action_events=[], prefix_events=[],
            source_task={"task_id": "t_r21"}, source_trace_id="tr_r21",
            proposed_ref=SkillRef(f"atomic_{occurrence_id}", "1.0.0"),
        )

    with tempfile.TemporaryDirectory() as tmp:
        database = StateDatabase(_Path(tmp) / "state.sqlite3")
        artifacts = ArtifactStore(_Path(tmp), database)
        skills = SkillRegistry(artifacts, database)
        cleaned = occurrence(
            "clean", "item", "result", "object.cleaned", EffectDomain.WORLD,
            "cup_1",
        )
        located = occurrence(
            "locate", "target", "found", "entity.discovered_at",
            EffectDomain.EVIDENCE, "cup_1",
        )
        canonical = [cleaned, located]
        contract = TaskContract([
            SemanticPredicate("object.cleaned", {"object": "cup_1"}),
            SemanticPredicate("object.at_location", {"object": "cup_1"}),
        ])
        candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["clean", "locate"],
                existing_edges=[],
                new_edges=[{
                    "edge_id": "e_r21",
                    "edge_type": "data_flow",
                    "source_step": "clean",
                    "target_step": "locate",
                    "source_role": "result",
                    "target_role": "target",
                }],
                summary="clean then locate", guideline={}, insight={},
            ),
            canonical,
            contract,
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True, "source_trace_id": "tr_r21",
                "terminal_revision": 1,
                "executed_occurrence_ids": ["clean", "locate"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        certificate = dict(candidate.metadata["terminal_certificate"])
        assert certificate["covered_effect_signatures"] == [
            {
                "predicate": "object.cleaned",
                "effect_domain": "world",
                "argument_roles": ["object"],
                "cardinality": 1,
                "distinct_by": "",
            }
        ]
        for occurrence in canonical:
            skills.register_atomic(replace(
                _atomic("atomic_r21_support"), ref=occurrence.proposed_ref,
            ))
        skills.register_composite(candidate)
        retriever = CompositeRetriever(skills)
        eligible = retriever.retrieve_terminal(
            SimpleNamespace(task_id="t_clean2", goal="clean"),
            TaskContract([
                SemanticPredicate("object.cleaned", {"object": "cup_2"}),
                SemanticPredicate("object.at_location", {"object": "cup_2"}),
            ]),
            mode="online", harness_profile="fake_v3",
        )
        assert len(eligible.terminal_empirical_candidates) == 1

        wrong_candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["clean", "locate"],
                existing_edges=[],
                new_edges=[{
                    "edge_id": "e_r21b",
                    "edge_type": "data_flow",
                    "source_step": "clean",
                    "target_step": "locate",
                    "source_role": "result",
                    "target_role": "target",
                }],
                summary="clean then locate", guideline={}, insight={},
            ),
            canonical,
            TaskContract([
                SemanticPredicate("object.cleaned", {"object": "cup_1"}),
                SemanticPredicate("object.at_location", {"object": "cup_9"}),
            ]),
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True, "source_trace_id": "tr_r21b",
                "terminal_revision": 1,
                "executed_occurrence_ids": ["clean", "locate"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        metadata = dict(wrong_candidate.metadata)
        wrong_certificate = dict(metadata["terminal_certificate"])
        wrong_certificate["covered_effect_signatures"] = [
            {
                "predicate": "object.cleaned",
                "effect_domain": "world",
                "argument_roles": ["object"],
                "cardinality": 1,
                "distinct_by": "",
            },
            {
                "predicate": "object.observed",
                "effect_domain": "evidence",
                "argument_roles": ["object"],
                "cardinality": 1,
                "distinct_by": "",
            },
        ]
        metadata["terminal_certificate"] = wrong_certificate
        skills.register_composite(replace(wrong_candidate, metadata=metadata))
        ineligible = retriever.retrieve_terminal(
            SimpleNamespace(task_id="t_clean3", goal="clean"),
            TaskContract([
                SemanticPredicate("object.cleaned", {"object": "cup_3"}),
                SemanticPredicate("object.at_location", {"object": "cup_3"}),
            ]),
            mode="online", harness_profile="fake_v3",
        )
        # The wrong cross-target candidate must be rejected; the correctly
        # covered candidate remains eligible.
        returned = {
            str(item.ref)
            for item in ineligible.terminal_empirical_candidates
        }
        assert str(wrong_candidate.ref) not in returned
        assert str(candidate.ref) in returned


def test_gate26_terminal_empirical_promotion_requires_distinct_tasks() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.governance.credit import (
        CreditAssigner,
        CreditAttempt,
        CreditOutcome,
        CreditTrace,
    )
    from atomic_skillgraph.governance.lifecycle import LifecycleController
    from atomic_skillgraph.governance.ledger import EvidenceLedger
    from atomic_skillgraph.governance.projections import LifecycleProjection
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        database = StateDatabase(_Path(tmp) / "state.sqlite3")
        artifacts = ArtifactStore(_Path(tmp), database)
        skills = SkillRegistry(artifacts, database)
        ledger = EvidenceLedger(database)
        projection = LifecycleProjection(database, ledger)
        canonical = [_terminal_canonical_occurrence()]
        skills.register_atomic(replace(
            _atomic("atomic_gate26"), ref=canonical[0].proposed_ref,
        ))
        candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"], existing_edges=[], new_edges=[],
                summary="observe prefix", guideline={}, insight={},
            ),
            canonical,
            TaskContract([
                SemanticPredicate(
                    "object.observed", {"object": "cup_1"},
                    effect_domain=EffectDomain.EVIDENCE,
                ),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]),
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True, "source_trace_id": "tr26",
                "terminal_revision": 1,
                "executed_occurrence_ids": ["occ_1"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        ref = str(candidate.ref)
        skills.register_composite(candidate)
        assert skills.get_composite(ref).status is SkillStatus.CANDIDATE
        assigner = CreditAssigner()

        def self_sufficient(task_id: str, trace_id: str, sequence: int):
            return assigner.assign(CreditTrace(
                task_id, trace_id,
                (CreditAttempt(
                    artifact_ref=ref, artifact_kind="composite",
                    occurrence_id="graph",
                    attempt_id=f"composite:{ref}:graph:{task_id}",
                    sequence_no=sequence,
                    outcome=CreditOutcome.SELF_SUFFICIENT_SUCCESS,
                ),),
            ))

        ledger.append_transaction(self_sufficient("task_a", "trace_a1", 0))
        ledger.append_transaction(self_sufficient("task_a", "trace_a2", 1))
        projection.consume_new_events()
        stats = projection.stats(ref, "composite")
        assert stats.independent_self_sufficient_success_count == 1
        LifecycleController(database, projection).review([ref])
        assert skills.get_composite(ref).status is SkillStatus.CANDIDATE

        ledger.append_transaction(self_sufficient("task_b", "trace_b1", 2))
        projection.consume_new_events()
        stats = projection.stats(ref, "composite")
        assert stats.independent_self_sufficient_success_count == 2
        LifecycleController(database, projection).review([ref])
        assert skills.get_composite(ref).status is SkillStatus.ACTIVE


def test_gate27_dynamic_rescue_never_credits_empirical_candidate() -> None:
    from atomic_skillgraph.governance.credit import CreditAssigner
    from atomic_skillgraph.governance.ledger import EvidenceEventType

    ref = "skill://composite_gate27@1.0.0"
    events = CreditAssigner().assign({
        "task_id": "task_rescue", "trace_id": "trace_rescue",
        "runtime_plan": {"source_composite_ref": ref},
        "task_rescue_required": True,
        "graph_self_sufficient_success": False,
        "benchmark_success": True,
        "node_records": [{
            "occurrence_id": "n1",
            "atomic_ref": "skill://atomic_gate27@1.0.0",
            "status": "direct_autonomous_success",
            "direct_result": {"started": True},
        }],
        "implementation_invocations": [{
            "attempt_id": "a1",
            "occurrence_id": "n1",
            "implementation_ref": "skill://impl_gate27@1.0.0",
            "preflight": {"passed": True},
            "result": {
                "started": True, "completed": True,
                "atomic_effect_passed": True,
            },
        }],
        "tool_executions": [],
    })
    composite_events = [
        item for item in events
        if item.artifact_ref == ref and item.artifact_kind == "composite"
    ]
    assert composite_events
    assert all(
        item.event is not EvidenceEventType.SELF_SUFFICIENT_SUCCESS
        for item in composite_events
    )
    assert any(
        item.event is EvidenceEventType.TASK_RESCUE_REQUIRED
        for item in composite_events
    )


def test_gate28_shorter_candidate_cannot_suppress_old_composite_prematurely() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.governance.credit import CreditAssigner
    from atomic_skillgraph.governance.lifecycle import LifecycleController
    from atomic_skillgraph.governance.ledger import EvidenceLedger
    from atomic_skillgraph.governance.projections import LifecycleProjection
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        database = StateDatabase(_Path(tmp) / "state.sqlite3")
        artifacts = ArtifactStore(_Path(tmp), database)
        skills = SkillRegistry(artifacts, database)
        ledger = EvidenceLedger(database)
        projection = LifecycleProjection(database, ledger)
        canonical = [_terminal_canonical_occurrence()]
        skills.register_atomic(replace(
            _atomic("atomic_gate28"), ref=canonical[0].proposed_ref,
        ))
        contract = TaskContract([
            SemanticPredicate(
                "object.observed", {"object": "cup_1"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
            SemanticPredicate("object.at_location", {"object": "cup_1"}),
        ])
        shorter_contract = TaskContract([
            SemanticPredicate(
                "object.observed", {"object": "cup_1"},
                effect_domain=EffectDomain.EVIDENCE,
            ),
            SemanticPredicate("object.at_location", {"object": "cup_2"}),
        ])
        terminal_certificate = {
            "benchmark_won": True, "source_trace_id": "tr28",
            "terminal_revision": 1,
            "executed_occurrence_ids": ["occ_1"],
            "skipped_planned_occurrence_ids": [],
            "observed_task_contract_coverage": {},
        }
        old = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"], existing_edges=[], new_edges=[],
                summary="longer original", guideline={}, insight={},
            ),
            canonical, contract, task_bindings={"item": "cup_1"},
            terminal_certificate=terminal_certificate,
        )
        shorter = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"], existing_edges=[], new_edges=[],
                summary="shorter replacement", guideline={}, insight={},
            ),
            canonical, shorter_contract, task_bindings={"item": "cup_1"},
            terminal_certificate=terminal_certificate,
        )
        old_ref = str(old.ref)
        shorter_ref = str(shorter.ref)
        skills.register_composite(old)
        skills.register_composite(shorter)
        skills.update_status(old_ref, SkillStatus.ACTIVE)
        assert skills.get_composite(old_ref).status is SkillStatus.ACTIVE
        assert skills.get_composite(shorter_ref).status is SkillStatus.CANDIDATE

        LifecycleController(database, projection).review([old_ref])
        assert skills.get_composite(old_ref).status is SkillStatus.ACTIVE

        # Only stable replacement evidence may suppress the original.
        assigner = CreditAssigner()
        ledger.append_transaction(assigner.assign_superseded(
            task_id="task_maintenance", trace_id="trace_maintenance",
            old_ref=old_ref, old_kind="composite",
            replacement_ref=shorter_ref, replacement_status="candidate",
        ))
        projection.consume_new_events()
        LifecycleController(database, projection).review([old_ref])
        assert skills.get_composite(old_ref).status is SkillStatus.ACTIVE

        ledger.append_transaction(assigner.assign_superseded(
            task_id="task_maintenance", trace_id="trace_maintenance2",
            old_ref=old_ref, old_kind="composite",
            replacement_ref=shorter_ref, replacement_status="active",
        ))
        projection.consume_new_events()
        LifecycleController(database, projection).review([old_ref])
        assert skills.get_composite(old_ref).status is SkillStatus.SUPPRESSED
        LifecycleController(database, projection).review([old_ref])
        assert skills.get_composite(old_ref).status is SkillStatus.RETIRED


def test_gate35_terminal_empirical_enters_planner_and_runtime() -> None:
    from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
    from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
    from atomic_skillgraph.governance.ledger import EvidenceLedger
    from atomic_skillgraph.governance.projections import LifecycleProjection
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.graph_store import GraphStore
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
    from atomic_skillgraph.planner.pipeline import PlannerPipeline
    from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
    from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
    from atomic_skillgraph.validation.engine import ValidationEngine
    from experiments.fakes import FakeAgentFactory, FakeHarness

    import tempfile
    from pathlib import Path as _Path

    class Gate35Harness(FakeHarness):
        """FakeHarness plus a second TaskContract target the prefix cannot cover."""

        def task_contract(self, task):
            return TaskContract(
                target_effects=[
                    SemanticPredicate(
                        "agent.holds",
                        {
                            "object": BindingExpression(
                                BindingExprKind.SKILL_INPUT,
                                source_role="item",
                            )
                        },
                    ),
                    SemanticPredicate(
                        "object.at_location",
                        {
                            "object": BindingExpression(
                                BindingExprKind.SKILL_INPUT,
                                source_role="item",
                            )
                        },
                    ),
                ],
                source=ContractSource.ADAPTER_DERIVED,
                confidence=1.0,
                validator_id="fake_v3_goal",
            )

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = _Path(tmp) / "bank"
        database = StateDatabase(data_dir / "state.sqlite3")
        artifacts = ArtifactStore(data_dir, database)
        skills = SkillRegistry(artifacts, database)
        graph = GraphStore(database, skills)
        ledger = EvidenceLedger(database)
        projection = LifecycleProjection(database, ledger)
        validation = ValidationEngine()
        harness = Gate35Harness()
        factory = FakeAgentFactory()
        from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
        from atomic_skillgraph.core.refs import SkillRef

        canonical = [CanonicalAtomicOccurrence(
            occurrence_id="occ_1", phase_id="p1", intent="take target",
            event_start=0, event_end=0,
            input_bindings={"item": "cup_1"},
            output_bindings={"result": "cup_1"},
            input_specs=[ParameterSpec("item", "entity")],
            output_specs=[ParameterSpec("result", "entity")],
            preconditions=[],
            effects=[SemanticPredicate(
                "agent.holds",
                {
                    "object": BindingExpression(
                        BindingExprKind.SKILL_INPUT, source_role="item",
                    )
                },
                effect_domain=EffectDomain.WORLD,
            )],
            action_events=[], prefix_events=[],
            source_task={"task_id": "task_terminal"},
            source_trace_id="trace_terminal",
            proposed_ref=SkillRef("atomic_take_gate35", "1.0.0"),
        )]
        take_atomic = replace(
            _atomic("atomic_take_gate35"),
            ref=canonical[0].proposed_ref,
        )
        take_atomic.inputs = [ParameterSpec("item", "entity")]
        take_atomic.outputs = [ParameterSpec("result", "entity")]
        take_atomic.validator_spec = {
            "validator_id": "harness_atomic_effect",
            "identity_strict": True,
            "output_identity": [{
                "output_role": "result",
                "input_role": "item",
            }],
        }
        take_atomic.effects = [
            SemanticPredicate(
                "agent.holds",
                {
                    "object": BindingExpression(
                        BindingExprKind.SKILL_INPUT, source_role="item",
                    )
                },
                effect_domain=EffectDomain.WORLD,
            )
        ]
        skills.register_atomic(take_atomic)
        candidate = CompositeBuilder().validate_and_build(
            CompositeExtractionProposal(
                control_sequence=["occ_1"], existing_edges=[], new_edges=[],
                summary="take prefix", guideline={}, insight={},
            ),
            canonical,
            TaskContract([
                SemanticPredicate("agent.holds", {"object": "cup_1"}),
                SemanticPredicate("object.at_location", {"object": "cup_1"}),
            ]),
            task_bindings={"item": "cup_1"},
            terminal_certificate={
                "benchmark_won": True, "source_trace_id": "tr35",
                "terminal_revision": 1,
                "executed_occurrence_ids": ["occ_1"],
                "skipped_planned_occurrence_ids": [],
                "observed_task_contract_coverage": {},
            },
        )
        skills.register_composite(candidate)

        planner = PlannerPipeline(skills, graph, factory)
        invocation_compiler = InvocationCompiler(
            skills,
            ToolRegistry(artifacts, database),
            harness,
            mode=RuntimeMode.ONLINE,
        )
        runtime = RuntimeOrchestrator(
            planner,
            harness,
            invocation_compiler,
            validation,
            factory,
            runtime_config={
                "global_action_budget": 20,
                "node_action_budget": 5,
                "learned_toolcall_repair_limit": 2,
            },
        )
        task = fake_task("gate35", "mug_1")
        plan = planner.build_plan(task, harness, mode=RuntimeMode.ONLINE)
        assert plan.source == "stored_composite"
        assert plan.planner_audit["selected_composite_authority"]["kind"] == "terminal_empirical"
        report = planner.validator.validate(
            plan, mode=RuntimeMode.ONLINE, harness_profile="fake_v3",
        )
        assert report.passed is True
        assert report.checks["task_contract_effect_coverage"] is False
        assert report.checks["terminal_empirical_incomplete_coverage_nonblocking"] is True

        factory.enqueue(
            "runtime_seeded",
            [
                FakeReply.tool("environment_action", {
                    "action_id": "r000_a001", "intent": "attempt_current_atomic",
                }),
            ],
        )
        trace = runtime.run_task(task)
        assert trace.benchmark_success is True
        assert [item.action_type for item in trace.environment_actions] == ["TAKE"]
        from atomic_skillgraph.core.results import NodeExecutionStatus

        assert trace.node_records[0].status not in {
            NodeExecutionStatus.NOT_STARTED,
            NodeExecutionStatus.FAILED_NOT_STARTED,
        }
