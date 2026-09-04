from __future__ import annotations

from dataclasses import replace

from experiments.fakes import FakeHarness

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.tooling.ir import (
    normalize_return_output_sources,
    normalize_tool_program,
    walk_program_nodes,
)
from atomic_skillgraph.tooling.proposal import ToolProposal
from atomic_skillgraph.tooling.validator import (
    ToolStaticValidator,
    _concrete_ids_from_nodes,
)


def test_recursive_leakage_scan_covers_structured_locals_and_return_constant() -> None:
    program = [{
        "node_id": "loop",
        "op": "FOR_EACH",
        "collection_source": {
            "source": "local_deterministic",
            "values": [{"nested": ["cabinet_7"]}],
        },
        "iteration_variable": "destination",
        "max_iterations": 1,
        "body": [{
            "node_id": "ret",
            "op": "RETURN",
            "output_sources": {
                "result": {
                    "kind": "constant",
                    "constant": {"nested": ["cup_3"]},
                }
            },
        }],
    }]

    assert _concrete_ids_from_nodes(program) == ["cabinet_7", "cup_3"]


def test_single_output_return_normalization_updates_nested_program_tree() -> None:
    program = normalize_tool_program([{
        "node_id": "branch",
        "op": "IF",
        "condition": {
            "source": "tool_input",
            "field": "ready",
            "op": "exists",
        },
        "then_branch": [{
            "node_id": "ret",
            "op": "RETURN",
            "output_sources": {
                "kind": "constant",
                "constant": "portable_literal",
            },
        }],
        "else_branch": [],
    }])

    for node in walk_program_nodes(program):
        if node.get("op") == "RETURN":
            node["output_sources"] = normalize_return_output_sources(
                node, {"result"},
            )

    nested_return = program[0]["then_branch"][0]
    assert nested_return["output_sources"] == {
        "result": {
            "kind": "constant",
            "constant": "portable_literal",
        }
    }


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
                "item": {"kind": "skill_input", "source_role": "item"}
            },
            "expected_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "$item"},
                "effect_domain": "world",
            }],
        }, {
            "node_id": "ret",
            "op": "RETURN",
            "output_sources": {
                "held_object": {"source": "tool_input", "field": "item"}
            },
        }],
        1,
        [effect],
        [],
        [],
        "bounded",
    )
    return atomic, proposal


def test_static_gate_requires_step_effect_and_structural_final_effect() -> None:
    atomic, proposal = _atomic_and_proposal()
    missing_step_effect = replace(
        proposal,
        program=[
            {key: value for key, value in proposal.program[0].items()
             if key != "expected_effects"},
            proposal.program[1],
        ],
    )
    wrong_final = replace(
        proposal,
        final_effects=[SemanticPredicate(
            "agent.holds",
            {"object": "$held_object"},
        )],
    )

    missing_report = ToolStaticValidator().validate_proposal(
        missing_step_effect, atomic, FakeHarness(),
    )
    wrong_report = ToolStaticValidator().validate_proposal(
        wrong_final, atomic, FakeHarness(),
    )

    assert "tool_ir_step_effect_missing" in missing_report.failure_codes
    assert "tool_ir_final_effects_missing" in wrong_report.failure_codes


def test_static_gate_fails_closed_for_schema_and_evidence_output() -> None:
    atomic, proposal = _atomic_and_proposal()
    invalid_evidence = replace(
        proposal,
        evidence_outputs=[{
            "role": "undeclared",
            "source": "tool_input",
            "field": "item",
        }],
    )
    missing_schema_harness = object()

    evidence_report = ToolStaticValidator().validate_proposal(
        invalid_evidence, atomic, FakeHarness(),
    )
    schema_report = ToolStaticValidator().validate_proposal(
        proposal, atomic, missing_schema_harness,
    )

    assert "tool_ir_evidence_output_invalid" in evidence_report.failure_codes
    assert (
        "tool_ir_predicate_schema_unavailable"
        in schema_report.failure_codes
    )
