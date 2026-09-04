from __future__ import annotations

from experiments.fakes import FakeHarness

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.tooling.proposal import ToolProposal, ToolProvenance
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.validation.engine import ValidationEngine


def test_single_output_return_is_canonical_from_static_through_admission() -> None:
    effect = SemanticPredicate("agent.holds", {"object": "$item"})
    atomic = AbstractAtomicSkill(
        SkillRef("atomic_take_canonical_return", "1.0.0"),
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
    legacy_return_source = {
        "source": "tool_input",
        "field": "item",
    }
    proposal = ToolProposal(
        proposal_version="1",
        decision="create",
        summary="take item",
        atomic_ref=str(atomic.ref),
        inputs=list(atomic.inputs),
        outputs=list(atomic.outputs),
        program=[
            {
                "node_id": "take",
                "op": "ACTION",
                "action_type": "TAKE",
                "argument_mapping": {
                    "item": {
                        "kind": "skill_input",
                        "source_role": "item",
                    }
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
                # Compatibility input accepted by static validation: a lone
                # source spec for the Atomic's single output.
                "output_sources": dict(legacy_return_source),
            },
        ],
        max_actions=1,
        final_effects=[effect],
        evidence_outputs=[],
        path_expectations=[],
        rationale="bounded",
    )
    occurrence = CanonicalAtomicOccurrence(
        occurrence_id="occ_take",
        phase_id="phase_take",
        intent="take item",
        event_start=0,
        event_end=0,
        input_bindings={"item": "apple_1"},
        output_bindings={"held_object": "apple_1"},
        input_specs=list(atomic.inputs),
        output_specs=list(atomic.outputs),
        preconditions=[],
        effects=[effect],
        action_events=[],
        prefix_events=[],
        source_task={"task_id": "task_take"},
        source_trace_id="trace_take",
        proposed_ref=atomic.ref,
    )
    harness = FakeHarness()

    static = ToolStaticValidator().validate_proposal(
        proposal, atomic, harness,
    )
    assert static.passed, static.failure_codes
    # Validation must not be the persistence mechanism for canonical IR.
    assert proposal.program[1]["output_sources"] == legacy_return_source

    compiled = ToolCompiler().compile_proposal(
        occurrence,
        atomic,
        proposal,
        ToolProvenance(
            "success_evolution",
            str(atomic.ref),
            "trace_take",
            "occ_take",
            task_id="task_take",
        ),
    )
    assert compiled.tool is not None
    assert compiled.tool.artifact["program"][1]["output_sources"] == {
        "held_object": legacy_return_source,
    }
    assert proposal.program[1]["output_sources"] == legacy_return_source

    admitted = Admission(ValidationEngine().tool).admit_tool(
        compiled.tool,
        replay=lambda _tool, _case: True,
        atomic=atomic,
        harness=harness,
    )
    assert admitted.status is ToolStatus.CANDIDATE, admitted.metadata.get(
        "admission_failure"
    )
