from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from experiments.fakes import FakeHarness

from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.results import (
    ImplementationExecutionResult,
    ToolExecutionResult,
    ValidationResult,
)
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.runtime.automation import RuntimeAutomationCoordinator
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.ir import ToolExecutionState
from atomic_skillgraph.tooling.proposal import (
    RuntimeAutomationAtomicDraft,
    ToolProposal,
)
from atomic_skillgraph.tooling.validator import (
    ToolStaticValidator,
    normalize_runtime_output_derivations,
)
from atomic_skillgraph.validation.tool_validator import ToolValidator


def _atomic_and_proposal() -> tuple[AbstractAtomicSkill, ToolProposal]:
    effect = SemanticPredicate("agent.holds", {"object": "$item"})
    atomic = AbstractAtomicSkill(
        SkillRef("atomic_take", "1.0.0"),
        "take item",
        [ParameterSpec("item", "entity")],
        [ParameterSpec("held_object", "entity")],
        [],
        [effect],
        {},
        [],
        {},
        {},
    )
    proposal = ToolProposal(
        "1",
        "create",
        "take item",
        str(atomic.ref),
        list(atomic.inputs),
        list(atomic.outputs),
        [{
            "node_id": "take",
            "op": "ACTION",
            "action_type": "TAKE",
            "argument_mapping": {
                "item": {"kind": "skill_input", "source_role": "item"},
            },
            "expected_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "$item"},
                "effect_domain": "world",
            }],
        }, {
            "node_id": "return",
            "op": "RETURN",
            "output_sources": {
                "held_object": {"source": "tool_input", "field": "item"},
            },
        }],
        1,
        [effect],
        [],
        [],
        "bounded",
    )
    return atomic, proposal


@pytest.mark.parametrize(
    "input_reference",
    [
        "$target",
        {"kind": "skill_input", "source_role": "target"},
        BindingExpression(BindingExprKind.SKILL_INPUT, source_role="target"),
    ],
)
def test_runtime_output_derivation_allows_input_effect_constraints(
    input_reference: object,
) -> None:
    draft = RuntimeAutomationAtomicDraft(
        draft_id="locate",
        intent="locate target",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("found", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "entity.discovered_for_query",
            {"entity": "$found", "query": input_reference},
            effect_domain=EffectDomain.EVIDENCE,
        )],
        rationale="",
        source_occurrence_id="occ",
    )

    assert normalize_runtime_output_derivations(draft) == {
        "found": {
            "kind": "effect_witness",
            "predicate": "entity.discovered_for_query",
            "argument_role": "entity",
        },
    }


def test_runtime_output_derivation_rejects_role_outside_boundary() -> None:
    draft = RuntimeAutomationAtomicDraft(
        draft_id="locate",
        intent="locate target",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("found", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "entity.discovered_for_query",
            {"entity": "$found", "query": "$unknown"},
            effect_domain=EffectDomain.EVIDENCE,
        )],
        rationale="",
        source_occurrence_id="occ",
    )

    with pytest.raises(
        ValueError,
        match="undeclared output role or input role unknown",
    ):
        normalize_runtime_output_derivations(draft)


def test_malformed_nested_ir_returns_deterministic_static_report() -> None:
    atomic, proposal = _atomic_and_proposal()
    proposal.program = [{
        "node_id": "branch",
        "op": "IF",
        "condition": {"source": "tool_input", "field": "item", "op": "exists"},
        "then_branch": {"not": "a-list"},
        "else_branch": [],
    }]

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )

    assert report.passed is False
    assert report.failure_codes == ["tool_ir_schema_invalid"]


def test_overdeep_ir_returns_deterministic_static_report() -> None:
    atomic, proposal = _atomic_and_proposal()
    nested: dict[str, object] = {
        "node_id": "return",
        "op": "RETURN",
        "output_sources": {
            "held_object": {"source": "tool_input", "field": "item"},
        },
    }
    for index in range(5):
        nested = {
            "node_id": f"branch_{index}",
            "op": "IF",
            "condition": {
                "source": "tool_input", "field": "item", "op": "exists",
            },
            "then_branch": [nested],
            "else_branch": [],
        }
    proposal.program = [nested]

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )

    assert report.passed is False
    assert report.failure_codes == ["tool_ir_recursion_depth_exceeded"]


def test_admission_closure_uses_same_fail_closed_walker() -> None:
    tool = _effect_tool()
    tool.artifact["program"] = [{
        "node_id": "branch",
        "op": "IF",
        "condition": {"source": "tool_input", "field": "target", "op": "exists"},
        "then_branch": {"not": "a-list"},
        "else_branch": [],
    }]

    reasons = Admission._tool_ir_closure_failures(tool)

    assert "tool_ir_schema_invalid" in reasons


@pytest.mark.parametrize(
    "reference",
    [
        "$unknown_role",
        {"kind": "skill_input", "source_role": "unknown_role"},
        {"kind": "local_variable", "source_role": "unknown_local"},
        {"kind": "data_flow", "source_role": "item", "source_step": "prior"},
    ],
)
def test_expected_effect_formals_are_lexically_closed(reference: object) -> None:
    atomic, proposal = _atomic_and_proposal()
    proposal.program[0]["expected_effects"][0]["args"]["object"] = reference

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )

    assert "tool_ir_effect_reference_invalid" in report.failure_codes


def test_for_each_collection_and_nested_semantic_selector_are_scoped() -> None:
    atomic, proposal = _atomic_and_proposal()
    action = copy.deepcopy(proposal.program[0])
    proposal.program = [{
        "node_id": "loop",
        "op": "FOR_EACH",
        "collection_source": {
            "source": "action_catalog",
            "field": "action_id",
            "where": {
                "argument_role": "item",
                "semantic_compatible_with": {
                    "source": "tool_input",
                    "field": "unknown_input",
                },
            },
        },
        "iteration_variable": "candidate",
        "max_iterations": 1,
        "body": [action],
    }, copy.deepcopy(proposal.program[1])]

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )

    assert "tool_ir_local_scope_invalid" in report.failure_codes


@pytest.mark.parametrize(
    ("source", "field"),
    [
        ("tool_input", "unknown_input"),
        ("local_variable", "unknown_local"),
    ],
)
def test_for_each_direct_collection_authority_is_scoped(
    source: str,
    field: str,
) -> None:
    atomic, proposal = _atomic_and_proposal()
    action = copy.deepcopy(proposal.program[0])
    proposal.program = [{
        "node_id": "loop",
        "op": "FOR_EACH",
        "collection_source": {"source": source, "field": field},
        "iteration_variable": "candidate",
        "max_iterations": 1,
        "body": [action],
    }, copy.deepcopy(proposal.program[1])]

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )

    assert "tool_ir_local_scope_invalid" in report.failure_codes


def test_return_nested_selector_local_authority_is_scoped() -> None:
    atomic, proposal = _atomic_and_proposal()
    proposal.program[1]["output_sources"] = {
        "held_object": {
            "source": "semantic_evidence",
            "where": {
                "predicate": "agent.holds",
                "argument_role": "object",
                "semantic_compatible_with": {
                    "source": "local_variable",
                    "field": "unknown_local",
                },
            },
            "project": {"kind": "argument", "role": "object"},
        },
    }

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, FakeHarness(),
    )

    assert "tool_ir_local_scope_invalid" in report.failure_codes


def _effect_tool() -> ToolAsset:
    return ToolAsset(
        ToolRef("tool_effect", "1.0.0"),
        "effect validation",
        {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        {"output_schema": {
            "type": "object",
            "properties": {"found": {"type": "string"}},
            "required": ["found"],
        }},
        "tool_ir_v1",
        {"program": [], "max_actions": 1, "final_effects": []},
        [],
        {"reviewed": True},
        {},
        {},
    )


def test_step_effect_wildcard_is_only_a_declared_unproduced_output() -> None:
    runner = ToolRunner(ToolValidator())
    effect = {
        "predicate": "entity.discovered_at",
        "args": {"entity": "$found", "query": "$target"},
        "effect_domain": "evidence",
    }

    resolved, wildcards, errors = runner._step_effect_resolution(
        effect,
        ToolExecutionState(bindings={"target": "cup"}),
        input_roles={"target"},
        output_roles={"found"},
    )
    assert errors == []
    assert wildcards == {"entity"}
    assert resolved["args"] == {"entity": None, "query": "cup"}

    _resolved, missing_wildcards, missing_errors = runner._step_effect_resolution(
        effect,
        ToolExecutionState(bindings={}),
        input_roles={"target"},
        output_roles={"found"},
    )
    assert missing_wildcards == {"entity"}
    assert any("input target is unavailable" in item for item in missing_errors)

    _resolved, unknown_wildcards, unknown_errors = runner._step_effect_resolution(
        {**effect, "args": {"entity": "$unknown"}},
        ToolExecutionState(bindings={"target": "cup"}),
        input_roles={"target"},
        output_roles={"found"},
    )
    assert unknown_wildcards == set()
    assert any("unknown formal role" in item for item in unknown_errors)


def test_fresh_output_fallback_matches_effect_domain() -> None:
    channel = SimpleNamespace(
        validate_atomic_effect=lambda _payload: SimpleNamespace(passed=False),
    )
    ctx = SimpleNamespace(
        harness=SimpleNamespace(validator_channel=lambda: channel),
    )
    state = ToolExecutionState(
        bindings={"target": "cup"},
        semantic_facts=[{
            "predicate": "entity.discovered_at",
            "args": {"entity": "cup_3", "query": "cup"},
            "effect_domain": "world",
        }],
    )
    node = {
        "node_id": "search",
        "expected_effects": [{
            "predicate": "entity.discovered_at",
            "args": {"entity": "$found", "query": "$target"},
            "effect_domain": "evidence",
        }],
    }

    mismatch = ToolRunner(ToolValidator())._validate_step_effects(
        node, ctx, state, tool=_effect_tool(),
    )
    assert mismatch["step_effect_passed"] is False

    state.step_effect_results.clear()
    state.semantic_facts[0]["effect_domain"] = "evidence"
    matched = ToolRunner(ToolValidator())._validate_step_effects(
        node, ctx, state, tool=_effect_tool(),
    )
    assert matched["step_effect_passed"] is True


def test_runtime_automation_admission_requires_executed_path_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = RuntimeAutomationAtomicDraft(
        draft_id="draft",
        intent="locate target",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("found", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "object.observed", {"object": "$found"},
            effect_domain=EffectDomain.EVIDENCE,
        )],
        rationale="",
        source_occurrence_id="occ",
        input_binding_specs={
            "target": {"kind": "constant", "value": "cup"},
        },
    )
    proposal = ToolProposal(
        "1", "create", "locate", "atomic://draft@1.0.0",
        list(draft.inputs), list(draft.outputs),
        [{"node_id": "return", "op": "RETURN", "output_sources": {}}],
        1, list(draft.effects), [], [], "",
    )

    class Builder:
        def __init__(self, _session: object) -> None:
            pass

        def build(self, **_kwargs: object) -> ToolProposal:
            return proposal

    class Compiler:
        def compile_proposal(self, _occ, atomic, _proposal, _provenance):
            return SimpleNamespace(
                atomic=atomic,
                tool=SimpleNamespace(ref="tool://draft@1.0.0", status="draft"),
                implementation=SimpleNamespace(
                    ref="implementation://draft@1.0.0", status="draft",
                ),
            )

    tool_result = ToolExecutionResult(
        "tool://draft@1.0.0", True, True, True, True, 1, None,
        [], {"found": "cup_3"}, 0, 1,
        tool_path_evidence={
            "step_effect_results": [{"step_effect_passed": False}],
        },
    )

    class Runner:
        def run(self, *_args: object, **_kwargs: object):
            return ImplementationExecutionResult(
                "implementation://draft@1.0.0", "atomic://draft@1.0.0",
                True, True, True, True,
                tool_results=[tool_result],
                validated_outputs={"found": "cup_3"},
                atomic_witness_refs=["effect:witness"],
            )

    class Static:
        def validate_automation_draft(self, *_args, **_kwargs):
            return ValidationResult.ok("r0")

        def validate_proposal(self, *_args, **_kwargs):
            return ValidationResult.ok("static")

    monkeypatch.setattr(
        "atomic_skillgraph.runtime.automation.ToolBuilderSession", Builder,
    )
    coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=lambda *_args: object(),
        tool_compiler=Compiler(),
        implementation_runner=Runner(),
        static_validator=Static(),
    )
    ctx = SimpleNamespace(
        harness=SimpleNamespace(
            profile_name="alfworld",
            semantic_predicate_schema=lambda: [],
            primitive_action_schema=lambda: [],
        ),
        tool_evidence_snapshot=lambda: {},
        trace_builder=SimpleNamespace(trace=SimpleNamespace(trace_id="trace")),
        binding_store=SimpleNamespace(snapshot_for_node=lambda _occ: {}),
        validated_outputs={},
        runtime_tool_trials={},
        action_history=[],
        task_id="task",
        task=SimpleNamespace(task_type="generic"),
    )

    outcome = coordinator.process_draft(
        draft=draft,
        ctx=ctx,
        occurrence=SimpleNamespace(occurrence_id="occ"),
    )

    assert outcome.r1_passed is False
    assert outcome.trial["r1"]["executed_path_effects_passed"] is False
    assert outcome.trial["r1"]["admission_eligible"] is False
    assert "declared_effects" not in outcome.trial
