"""R9.2 public Runtime self-tooling interface regressions."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.runtime_prompt_texts import (
    PREPARATION_PROMPT,
    SEEDED_PROMPT,
)
from atomic_skillgraph.agents.session import ReplayAgentSession
from atomic_skillgraph.agents.structured_submission import (
    RUNTIME_AUTOMATION_ATOMIC_SCHEMA,
)
from atomic_skillgraph.agents.usage import UsageLedger
from atomic_skillgraph.core.contracts import (
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.tooling.proposal import RuntimeAutomationAtomicDraft
from atomic_skillgraph.tooling.runtime_interface import (
    RUNTIME_INPUT_BINDING_KINDS,
    build_runtime_automation_interface,
    build_runtime_automation_interface_update,
    resolve_runtime_automation_inputs,
)
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.runtime.automation import RuntimeAutomationCoordinator
from experiments.fakes import (
    FakeAgentFactory,
    FakeHarness,
    FakeReply,
    ScriptedAgentProvider,
)


def _fixtures():
    path = Path(__file__).with_name(
        "test_stored_composite_binding_authority.py"
    )
    spec = importlib.util.spec_from_file_location("_r92_binding_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(factory: FakeAgentFactory):
    fixtures = _fixtures()
    runtime, ctx, occurrence, invocations = fixtures._single_nav_context(
        fixtures._PickPlaceHarness(), factory,
    )
    schema_harness = FakeHarness()
    ctx.harness.semantic_predicate_schema = (
        schema_harness.semantic_predicate_schema
    )
    ctx.harness.primitive_action_schema = schema_harness.primitive_action_schema
    return runtime, ctx, occurrence, invocations


def _first_user_payload(ctx, session_type: str) -> dict:
    record = next(
        item
        for item in ctx.trace_builder.trace.agent_sessions
        if item.session_type == session_type
    )
    message = next(
        item for item in record.snapshot["messages"]
        if item.get("role") == "user"
    )
    _instruction, encoded = message["content"].split(
        "\n\nPOLICY_CONTEXT_JSON\n", 1,
    )
    return json.loads(encoded)


def test_runtime_prompts_do_not_misclassify_bounded_semantic_search() -> None:
    clarification = (
        "A bounded systematic check of public candidate sources for a missing "
        "required binding can be mechanical even though the target is semantic."
    )
    for prompt in (PREPARATION_PROMPT, SEEDED_PROMPT):
        assert clarification in prompt
        assert "preconditions may reference declared inputs only" in prompt
        assert "effects may reference declared inputs and outputs" in prompt
        assert "Use $role references" in prompt
        assert "never copied runtime values" in prompt
        assert "do not automate merely to use automation" in prompt


def test_runtime_draft_schema_explains_role_reference_derivation() -> None:
    properties = RUNTIME_AUTOMATION_ATOMIC_SCHEMA["properties"]
    precondition_help = properties["preconditions"]["description"]
    effect_help = properties["effects"]["description"]

    assert "$<input_role>" in precondition_help
    assert "$<role>" in effect_help
    assert "Each fresh required output" in effect_help
    assert "exactly one distinct (predicate, argument_role)" in effect_help
    assert "angle-bracket placeholder" in effect_help


@pytest.mark.parametrize(
    ("session_kind", "session_type"),
    [
        ("runtime_preparation", "RuntimePreparationSession"),
        ("runtime_seeded", "SeededSession"),
    ],
)
def test_first_runtime_request_projects_complete_public_interface(
    session_kind: str,
    session_type: str,
) -> None:
    factory = FakeAgentFactory()
    factory.enqueue(session_kind, [
        FakeReply.tool(
            "report_runtime_status", {"status": "cannot_resolve"},
        ),
    ])
    runtime, ctx, occurrence, invocations = _context(factory)

    if session_kind == "runtime_preparation":
        runtime.node_executor.run_preparation_session(
            occurrence, invocations, ctx,
        )
    else:
        runtime.node_executor.run_seeded_fresh(occurrence, ctx)

    payload = _first_user_payload(ctx, session_type)
    interface = payload["runtime_automation_interface"]
    assert interface["schema_version"] == "runtime_automation_interface_v1"
    assert interface["source_occurrence_id"] == occurrence.occurrence_id
    assert interface["primitive_actions"]
    assert interface["predicate_vocabulary"]
    assert {
        item["kind"] for item in interface["input_binding_kinds"]
    } == set(RUNTIME_INPUT_BINDING_KINDS)
    from atomic_skillgraph.tooling.runtime_interface import RUNTIME_OUTPUT_DERIVATION_RULES
    assert interface["fresh_output_rules"] == {
        "derivation_contract": RUNTIME_OUTPUT_DERIVATION_RULES,
        "future_effect_witness_allowed": True,
        "existing_output_witness_required_at_r0": False,
        "outputs_must_be_validated_after_trial": True,
    }
    assert interface["trial_scope"] == "task_local"
    assert len(ctx.trace_builder.trace.environment_actions) == 0
    factory.assert_exhausted()


def _future_output_draft(occurrence_id: str) -> RuntimeAutomationAtomicDraft:
    return RuntimeAutomationAtomicDraft(
        draft_id="r92_future_output",
        intent="establish a bounded future location witness",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("object", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": "$object"},
            effect_domain=EffectDomain.WORLD,
        )],
        rationale="The output will be established by the future trial.",
        source_occurrence_id=occurrence_id,
        input_binding_specs={
            "target": {
                "kind": "constant",
                "value": "desk",
            },
        },
    )


def test_r0_allows_future_output_but_rejects_false_contract_authority() -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = _future_output_draft(occurrence.occurrence_id)
    validator = ToolStaticValidator()

    accepted = validator.validate_automation_draft(
        draft,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )
    assert accepted.passed is True
    assert not any(
        item.get("predicate") == "agent.holds"
        for item in ctx.harness.validator_channel().snapshot().get("facts", [])
    )

    wrong_occurrence = validator.validate_automation_draft(
        replace(draft, source_occurrence_id="forged_occurrence"),
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )
    assert "runtime_automation_source_occurrence_mismatch" in (
        wrong_occurrence.failure_codes
    )

    wrong_signature = replace(draft, effects=[SemanticPredicate(
        "agent.holds",
        {"wrong_role": "$object"},
        effect_domain=EffectDomain.WORLD,
    )])
    report = validator.validate_automation_draft(
        wrong_signature,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )
    assert "runtime_automation_r0_predicate_signature" in report.failure_codes


@pytest.mark.parametrize(
    "structured_reference",
    [
        {"kind": "skill_input", "source_role": "object"},
        {
            "kind": "skill_input",
            "source_role": "object",
            "source_step": "",
            "constant": None,
            "transform_id": "",
        },
    ],
)
def test_r0_accepts_exact_structured_skill_input_role_reference(
    structured_reference: dict[str, object],
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = replace(
        _future_output_draft(occurrence.occurrence_id),
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": structured_reference},
            effect_domain=EffectDomain.WORLD,
        )],
    )

    report = ToolStaticValidator().validate_automation_draft(
        draft,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is True, report
    assert report.checks["draft_formal_predicate_references"] is True


@pytest.mark.parametrize(
    "invalid_reference",
    [
        "desk",
        "<discovered_candidate_location>",
        {"kind": "constant", "constant": "desk"},
        {
            "kind": "data_flow",
            "source_step": "prior",
            "source_role": "object",
        },
        {
            "kind": "skill_input",
            "source_role": "object",
            "source_step": "forged",
        },
        {
            "kind": "skill_input",
            "source_role": "object",
            "constant": "desk",
        },
        {
            "kind": "skill_input",
            "source_role": "object",
            "transform_id": "forged",
        },
        {
            "kind": "skill_input",
            "source_role": "object",
            "note": "not part of the formal reference",
        },
        {"kind": "skill_input", "source_role": "undeclared"},
    ],
)
def test_r0_rejects_nonformal_or_undeclared_predicate_arguments(
    invalid_reference: object,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = replace(
        _future_output_draft(occurrence.occurrence_id),
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": invalid_reference},
            effect_domain=EffectDomain.WORLD,
        )],
    )

    report = ToolStaticValidator().validate_automation_draft(
        draft,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is False
    assert report.checks["draft_formal_predicate_references"] is False
    assert "runtime_automation_r0_role_closure" in report.failure_codes


@pytest.mark.parametrize("boundary", ["inputs", "outputs"])
def test_r0_rejects_duplicate_roles_within_each_boundary(
    boundary: str,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = _future_output_draft(occurrence.occurrence_id)
    repeated = list(getattr(draft, boundary))
    draft = replace(draft, **{boundary: [*repeated, repeated[0]]})

    report = ToolStaticValidator().validate_automation_draft(
        draft,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is False
    assert report.checks["draft_role_names_unique"] is False
    assert "runtime_automation_r0_role_closure" in report.failure_codes


def test_r0_preserves_cross_boundary_input_identity_role() -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = RuntimeAutomationAtomicDraft(
        draft_id="r92_input_identity",
        intent="retain a formally bound object identity",
        inputs=[ParameterSpec("object", "entity")],
        outputs=[ParameterSpec("object", "entity")],
        preconditions=[SemanticPredicate(
            "agent.holds",
            {"object": "$object"},
            effect_domain=EffectDomain.WORLD,
        )],
        effects=[SemanticPredicate(
            "agent.holds",
            {"object": "$object"},
            effect_domain=EffectDomain.WORLD,
        )],
        rationale="The output preserves the declared input identity.",
        source_occurrence_id=occurrence.occurrence_id,
        input_binding_specs={
            "object": {"kind": "constant", "value": "desk"},
        },
    )

    report = ToolStaticValidator().validate_automation_draft(
        draft,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is True, report
    assert report.checks["draft_role_names_unique"] is True


@pytest.mark.parametrize(
    "output_reference",
    [
        "$object",
        {"kind": "skill_input", "source_role": "object"},
    ],
)
def test_r0_preconditions_cannot_reference_output_only_roles(
    output_reference: object,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = replace(
        _future_output_draft(occurrence.occurrence_id),
        preconditions=[SemanticPredicate(
            "agent.holds",
            {"object": output_reference},
            effect_domain=EffectDomain.WORLD,
        )],
    )

    report = ToolStaticValidator().validate_automation_draft(
        draft,
        ctx.harness,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert report.passed is False
    assert report.checks["draft_formal_predicate_references"] is False
    assert "runtime_automation_r0_role_closure" in report.failure_codes


@pytest.mark.parametrize("leak_source", ["input", "predicate"])
def test_r0_rejects_episode_literals_before_tool_builder(
    leak_source: str,
) -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    draft = replace(
        _future_output_draft(occurrence.occurrence_id),
        draft_id=f"r92_episode_literal_{leak_source}",
    )
    if leak_source == "input":
        draft = replace(draft, input_binding_specs={
            "target": {"kind": "constant", "value": "mug_1"},
        })
    else:
        draft = replace(draft, effects=[SemanticPredicate(
            "agent.holds",
            {"object": "mug_1"},
            effect_domain=EffectDomain.WORLD,
        )])

    builder_calls: list[str] = []
    coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=lambda *_args: builder_calls.append("called"),
        tool_compiler=SimpleNamespace(),
        implementation_runner=SimpleNamespace(),
    )
    before_actions = len(ctx.trace_builder.trace.environment_actions)

    outcome = coordinator.process_draft(
        draft=draft,
        ctx=ctx,
        occurrence=occurrence,
    )

    assert outcome.stage == "r0_rejected"
    assert "runtime_automation_r0_episode_concrete_id" in (
        outcome.r0_report["failure_codes"]
    )
    assert builder_calls == []
    assert len(ctx.trace_builder.trace.environment_actions) == before_actions


def test_public_interface_excludes_private_or_executable_state() -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    interface = build_runtime_automation_interface(
        ctx.harness, occurrence, ctx.binding_store,
    )
    encoded = json.dumps(interface, sort_keys=True).casefold()
    assert "validator_snapshot" not in encoded
    assert "hidden_state" not in encoded
    assert "primitive_ir" not in encoded
    assert "tool_bindings" not in encoded
    assert "source_task" not in encoded


def test_lightweight_store_projects_empty_sources_without_weakening_r0() -> None:
    _runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    empty_store = object()

    interface = build_runtime_automation_interface(
        ctx.harness, occurrence, empty_store,
    )

    sources = interface["current_input_sources"]
    assert sources["current_occurrence_anchor"] == []
    assert sources["current_confirmed_binding"] == []
    assert sources["current_candidate_binding"] == []
    assert sources["data_flow"] == []

    draft = replace(
        _future_output_draft(occurrence.occurrence_id),
        input_binding_specs={
            "target": {
                "kind": "current_confirmed_binding",
                "source_role": "target",
            },
        },
    )
    resolution = resolve_runtime_automation_inputs(
        draft,
        SimpleNamespace(binding_store=empty_store, world_revision=0),
        occurrence,
    )
    assert resolution.passed is False
    assert resolution.values == {}
    assert resolution.failure_codes == [
        "runtime_automation_input_binding_invalid",
    ]


def test_a07_multiturn_runtime_uses_one_static_interface_and_latest_sources() -> None:
    factory = FakeAgentFactory()
    runtime, ctx, occurrence, invocations = _context(factory)
    action_id = ctx.action_catalog[0].action_id
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("environment_action", {
            "action_id": action_id,
            "intent": "explore",
        }),
        FakeReply.tool(
            "report_runtime_status", {"status": "cannot_resolve"},
        ),
    ])

    runtime.node_executor.run_preparation_session(occurrence, invocations, ctx)

    record = next(
        item for item in ctx.trace_builder.trace.agent_sessions
        if item.session_type == "RuntimePreparationSession"
    )
    messages = record.snapshot["messages"]
    user_payload = json.loads(
        next(item["content"] for item in messages if item["role"] == "user")
        .split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1]
    )
    tool_payloads = [
        json.loads(item["content"])
        for item in messages
        if item["role"] == "tool"
    ]
    dynamic = next(
        item for item in tool_payloads
        if "runtime_automation_interface_update" in item
    )

    assert user_payload["runtime_automation_interface"]["predicate_vocabulary"]
    assert "runtime_automation_interface" not in dynamic
    assert dynamic["runtime_automation_interface_update"] == (
        build_runtime_automation_interface_update(
            occurrence, ctx.binding_store,
        )
    )
    assert dynamic["action_catalog"]
    assert dynamic["action_catalog"]["revision"] == dynamic["new_revision"]
    assert dynamic["action_catalog"]["actions"]
    assert sum(
        "predicate_vocabulary" in str(item.get("content", ""))
        for item in messages
    ) == 1
    encoded_payloads = json.dumps(
        [user_payload, *tool_payloads], sort_keys=True,
    ).casefold()
    assert '"program"' not in encoded_payloads
    factory.assert_exhausted()


def test_a07_request_audit_combines_static_interface_with_latest_update() -> None:
    runtime, ctx, occurrence, _invocations = _context(FakeAgentFactory())
    interface = build_runtime_automation_interface(
        ctx.harness, occurrence, ctx.binding_store,
    )
    initial = (
        "runtime\n\nPOLICY_CONTEXT_JSON\n"
        + json.dumps({"runtime_automation_interface": interface})
    )
    update = {
        "current_state_snapshot": {"revision": 1},
        "exploration_memory": {},
        "recent_accepted_actions": [],
        "runtime_automation_interface_update": {
            "source_occurrence_id": occurrence.occurrence_id,
            "current_input_sources": interface["current_input_sources"],
        },
    }
    provider = ScriptedAgentProvider([
        FakeReply.tool(
            "report_runtime_status", {"status": "cannot_resolve"},
        ),
        FakeReply.tool(
            "report_runtime_status", {"status": "cannot_resolve"},
        ),
    ])
    session = ReplayAgentSession(
        provider,
        system_prompt="runtime",
        usage_ledger=UsageLedger(),
        usage_bucket="runtime_preparation",
        session_id="r92-a07-audit",
    )
    tools = [
        runtime.node_executor._automation_tool(),
        runtime.node_executor._status_tool(),
    ]

    first = session.next_turn(initial, tools=tools)
    session.submit_tool_result(first.tool_calls[0].call_id, update, tools=tools)

    audits = session.snapshot()["runtime_request_context_audits"]
    assert [
        item["runtime_automation_interface_projected"] for item in audits
    ] == [True, True]
    assert [
        item["runtime_automation_interface_update_projected"]
        for item in audits
    ] == [False, True]
    assert sum(
        "predicate_vocabulary" in str(item.get("content", ""))
        for item in audits[-1]["messages"]
    ) == 1
