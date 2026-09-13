"""R9.2 fail-closed Runtime draft and Tool IR public-contract checks."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.agents.structured_submission import (
    TOOL_IR_CONDITION_SCHEMA,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.ir import ToolExecutionState, resolve_collection
from atomic_skillgraph.tooling.proposal import (
    RuntimeAutomationAtomicDraft,
    ToolProposal,
    ToolProvenance,
)
from atomic_skillgraph.tooling.runtime_interface import (
    public_tool_ir_condition_contract,
)
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.validation.tool_validator import ToolValidator
from experiments.fakes import FakeAgentFactory, FakeHarness


def _fixture_module():
    path = Path(__file__).with_name(
        "test_stored_composite_binding_authority.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_r92_tool_ir_public_contract_fixtures", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_context():
    fixtures = _fixture_module()
    harness = fixtures._PickPlaceHarness()
    runtime, ctx, occurrence, _invocations = fixtures._single_nav_context(
        harness, FakeAgentFactory(),
    )
    public_fake = FakeHarness()
    harness.primitive_action_schema = public_fake.primitive_action_schema
    predicate_schema = [*public_fake.semantic_predicate_schema(), {
        "predicate": "agent.at_location",
        "effect_domain": "world",
        "argument_roles": ["location"],
        "argument_semantic_types": {"location": "entity"},
    }]
    harness.semantic_predicate_schema = lambda: list(predicate_schema)
    return runtime, ctx, occurrence


def _identity_draft(
    occurrence_id: str,
    *,
    input_type: str = "entity",
    output_type: str = "entity",
    input_resolution: str = "semantic",
    output_resolution: str = "semantic",
) -> RuntimeAutomationAtomicDraft:
    return RuntimeAutomationAtomicDraft(
        draft_id="r92_identity_boundary",
        intent="preserve one validated identity",
        inputs=[ParameterSpec(
            "object", input_type, required_resolution=input_resolution,
        )],
        outputs=[ParameterSpec(
            "object", output_type, required_resolution=output_resolution,
        )],
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": "$object"},
            effect_domain=EffectDomain.WORLD,
        )],
        rationale="The output is the exact declared input identity.",
        source_occurrence_id=occurrence_id,
        input_binding_specs={
            "object": {"kind": "constant", "value": "portable object"},
        },
    )


@pytest.mark.parametrize(
    "draft_kwargs",
    [
        {"input_type": "location", "output_type": "entity"},
        {"input_resolution": "semantic", "output_resolution": "concrete"},
    ],
)
def test_r0_rejects_same_name_identity_boundary_mismatch(
    draft_kwargs: dict[str, str],
) -> None:
    _runtime, ctx, occurrence = _runtime_context()

    report = ToolStaticValidator().validate_automation_draft(
        _identity_draft(occurrence.occurrence_id, **draft_kwargs),
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is False
    assert report.checks["draft_output_derivations"] is False
    assert "runtime_automation_r0_output_derivation_invalid" in (
        report.failure_codes
    )
    assert any(
        "identical semantic_type and required_resolution" in message
        for message in report.messages
    )


def test_r0_preserves_exact_same_name_identity_boundary() -> None:
    _runtime, ctx, occurrence = _runtime_context()

    report = ToolStaticValidator().validate_automation_draft(
        _identity_draft(occurrence.occurrence_id),
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is True, report
    assert report.checks["draft_output_derivations"] is True


def _catalog_atomic_and_proposal() -> tuple[
    AbstractAtomicSkill, ToolProposal,
]:
    boundary = ParameterSpec(
        "destination", "entity", required_resolution="concrete",
    )
    final_effect = SemanticPredicate(
        "agent.at_location",
        {"location": "$destination"},
        effect_domain=EffectDomain.WORLD,
    )
    atomic = AbstractAtomicSkill(
        ref=SkillRef("atomic_catalog_navigation", "1.0.0"),
        summary="visit the current public destinations",
        inputs=[boundary],
        outputs=[copy.deepcopy(boundary)],
        preconditions=[],
        effects=[final_effect],
        validator_spec={
            "output_derivations": {
                "destination": {
                    "kind": "input_identity",
                    "input_role": "destination",
                },
            },
        },
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.CANDIDATE,
    )
    program = [{
        "node_id": "visit_public_destinations",
        "op": "FOR_EACH",
        "collection_source": {
            "source": "action_catalog",
            "where": {"action_type": "GO_TO"},
            "project": {"kind": "argument", "role": "destination"},
            "distinct": True,
        },
        "iteration_variable": "candidate_destination",
        "max_iterations": 2,
        "body": [{
            "node_id": "visit_one_destination",
            "op": "ACTION",
            "action_type": "GO_TO",
            "argument_mapping": {
                "destination": {
                    "kind": "local_variable",
                    "source_role": "candidate_destination",
                },
            },
            "expected_effects": [{
                "predicate": "agent.at_location",
                "args": {
                    "location": {
                        "kind": "local_variable",
                        "source_role": "candidate_destination",
                    },
                },
                "effect_domain": "world",
            }],
        }],
    }, {
        "node_id": "return_destination",
        "op": "RETURN",
        "output_sources": {
            "destination": {
                "source": "tool_input",
                "field": "destination",
            },
        },
    }]
    proposal = ToolProposal(
        proposal_version="1",
        decision="create",
        summary="bounded current-catalog navigation",
        atomic_ref=str(atomic.ref),
        inputs=list(atomic.inputs),
        outputs=list(atomic.outputs),
        program=program,
        max_actions=2,
        final_effects=list(atomic.effects),
        evidence_outputs=[],
        path_expectations=[],
        rationale="Enumerate only current public GO_TO destinations.",
    )
    return atomic, proposal


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_action_type",
        "unknown_argument_role",
        "unknown_project_role",
        "unknown_direct_filter_role",
        "legacy_filter_wrapper",
    ],
)
def test_action_catalog_selector_roles_close_against_harness_schema(
    mutation: str,
) -> None:
    _runtime, ctx, _occurrence = _runtime_context()
    atomic, proposal = _catalog_atomic_and_proposal()
    selector = proposal.program[0]["collection_source"]
    if mutation == "unknown_action_type":
        selector["where"]["action_type"] = "INVENTED_ACTION"
    elif mutation == "unknown_argument_role":
        selector["where"].update({
            "argument_role": "invented_role",
            "semantic_compatible_with": {
                "source": "tool_input",
                "field": "destination",
            },
        })
    elif mutation == "unknown_project_role":
        selector["project"]["role"] = "invented_role"
    elif mutation == "unknown_direct_filter_role":
        selector["where"]["invented_role"] = "portable value"
    else:
        selector["where"]["primitive_argument_filters"] = {
            "destination": "portable value",
        }

    report = ToolStaticValidator().validate_proposal(
        proposal, atomic, ctx.harness,
    )

    assert report.passed is False
    assert "tool_ir_selector_invalid" in report.failure_codes


def test_action_catalog_interpreter_uses_direct_argument_filter_keys() -> None:
    state = ToolExecutionState(catalog=[
        {
            "action_id": "go-drawer",
            "revision": 0,
            "action_type": "GO_TO",
            "arguments": {"destination": "drawer_2"},
        },
        {
            "action_id": "go-desk",
            "revision": 0,
            "action_type": "GO_TO",
            "arguments": {"destination": "desk_1"},
        },
    ])

    values = resolve_collection(
        {
            "source": "action_catalog",
            "where": {
                "action_type": "GO_TO",
                "destination": "desk_1",
            },
            "project": {"kind": "argument", "role": "destination"},
            "distinct": True,
        },
        state,
    )

    assert values == ["desk_1"]


def test_public_condition_contract_matches_native_submission_schema() -> None:
    contract = public_tool_ir_condition_contract()
    properties = TOOL_IR_CONDITION_SCHEMA["properties"]

    assert set(contract["sources"]) == set(properties["source"]["enum"])
    assert set(contract["operators"]) == set(properties["op"]["enum"])
    assert "action_catalog" in contract["source_field_contracts"]
    assert set(contract["operators"]) == {
        "exists", "not_exists", "equals", "not_equals",
        "contains", "empty", "non_empty",
    }

    atomic, _proposal = _catalog_atomic_and_proposal()
    prompt = ContextBuilder().tool_builder(
        atomic=atomic,
        provenance=ToolProvenance(
            source="runtime_automation",
            atomic_ref=str(atomic.ref),
            source_trace_id="trace",
            occurrence_id="occurrence",
            draft_id="draft",
            task_id="task",
        ),
    )
    payload = json.loads(prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1])
    assert payload["tool_ir_schema"]["condition_contract"] == contract


def test_action_catalog_projection_static_validation_and_real_execution() -> None:
    _runtime, ctx, occurrence = _runtime_context()
    atomic, proposal = _catalog_atomic_and_proposal()

    static = ToolStaticValidator().validate_proposal(
        proposal, atomic, ctx.harness,
    )
    assert static.passed is True, static

    tool = ToolAsset(
        ref=ToolRef("tool_catalog_navigation", "1.0.0"),
        summary=proposal.summary,
        signature={
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"],
            "additionalProperties": False,
        },
        interface={
            "output_schema": {
                "type": "object",
                "properties": {"destination": {"type": "string"}},
                "required": ["destination"],
                "additionalProperties": False,
            },
        },
        artifact_kind="tool_ir_v1",
        artifact={
            "schema_version": 1,
            "max_actions": proposal.max_actions,
            "program": copy.deepcopy(proposal.program),
            "final_effects": [to_primitive(item) for item in atomic.effects],
            "evidence_outputs": [],
        },
        tests=[],
        safety={
            "reviewed": True,
            "allowed_action_types": ["GO_TO"],
            "zero_llm": True,
        },
        provenance={},
        metadata={},
        status=ToolStatus.ADMISSION_PENDING,
    )

    result = ToolRunner(ToolValidator()).run(
        tool,
        {"destination": "desk_1"},
        ctx,
        occurrence_id=occurrence.occurrence_id,
        execution_scope="runtime_trial",
    )

    assert result.completed is True, result
    assert result.atomic_effect_passed is True
    assert result.executed_action_count == 2
    assert result.loop_iteration_counts == {"visit_public_destinations": 2}
    assert result.output_candidates == {"destination": "desk_1"}
    assert [
        action.arguments["destination"]
        for action in ctx.trace_builder.trace.environment_actions
    ] == ["drawer_2", "desk_1"]
