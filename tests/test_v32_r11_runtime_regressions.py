from __future__ import annotations

import json
from types import SimpleNamespace

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    EffectDomain,
    ParameterSpec,
    PlannerRequirementBundle,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    ImplementationExecutionResult,
    ToolCallPreflightResult,
    ToolExecutionResult,
    ValidationResult,
)
from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.runtime.automation import RuntimeAutomationCoordinator
from atomic_skillgraph.runtime.implementation_runner import ImplementationRunner
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.planner.pipeline import PlannerPipeline
from atomic_skillgraph.planner.repairability import (
    RepairabilityDecision,
    RepairabilityGate,
)
from atomic_skillgraph.tooling.ir import ToolExecutionState
from atomic_skillgraph.tooling.proposal import RuntimeAutomationAtomicDraft, ToolProposal
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.validation.tool_validator import ToolValidator


def test_resolved_effect_uses_fresh_outputs_for_aliased_roles() -> None:
    state = ToolExecutionState(
        bindings={"target": "cup"},
        local={"found_location": "countertop_2"},
        outputs={"found_entity": "cup_3"},
    )
    effect = {
        "predicate": "entity.discovered_at",
        "args": {
            "entity": {
                "kind": "skill_input",
                "source_role": "found_entity",
            },
            "location": "$found_location",
        },
        "effect_domain": "world",
    }

    resolved = ToolRunner(ToolValidator())._resolved_effect(effect, state)

    assert resolved["args"] == {
        "entity": "cup_3",
        "location": "countertop_2",
    }


def test_extractor_prompt_allows_shared_prerequisite_envelope_overlap() -> None:
    prompt = ContextBuilder().extractor_e1(canonical_trace={"actions": []})
    instruction, _payload = prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)

    assert "Temporal evidence envelopes may overlap" in instruction
    assert "support_event_ids, not envelope overlap" in instruction
    assert "same effect-producing support event" in instruction
    assert "shared_precondition_event_ids" in instruction
    assert "not a general list of prerequisite" in instruction
    assert "must also be selected in support_event_ids" in instruction
    assert "prerequisites belong in the temporal envelope" in instruction
    assert "Ranges must be ordered and non-overlapping." not in instruction


class _TraceBuilder:
    def __init__(self) -> None:
        self.trace = SimpleNamespace(
            trace_id="trace_authority",
            validations=[],
            implementation_invocations=[],
        )

    def start_span(self, _kind: str, _occurrence_id: str) -> SimpleNamespace:
        return SimpleNamespace(span_id="span_authority")

    def finish_span(self, _span_id: str) -> None:
        pass


class _BindingStore:
    def snapshot_for_node(self, _occurrence: object) -> dict[str, object]:
        return {}

    def commit_repeat_bindings(self, *_args: object, **_kwargs: object) -> ValidationResult:
        return ValidationResult.ok("runtime_repeat")


class _AtomicValidation:
    def validate_execution_result(self, *_args: object, **_kwargs: object) -> ValidationResult:
        return ValidationResult(
            "atomic",
            True,
            witness_refs=[
                "alfworld_action_fact:r7:entity.discovered_at:entity=cup_3,location=countertop_2"
            ],
        )


def _successful_tool_result() -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_ref="tool://locate@1.0.0",
        preflight_passed=True,
        started=True,
        completed=True,
        state_changed=True,
        executed_step_count=1,
        failure_step_index=None,
        partial_effects=[],
        output_candidates={"result": "cup_3"},
        before_revision=6,
        after_revision=7,
        tool_path_evidence={
            "evidence_refs": ["semantic_evidence:entity"],
            "step_effect_results": [{
                "program_node_id": "locate",
                "step_effect_passed": True,
            }],
        },
    )


def test_implementation_result_carries_atomic_validator_witnesses() -> None:
    validation = SimpleNamespace(tool=ToolValidator(), atomic=_AtomicValidation())
    runner = ImplementationRunner(validation)
    runner.tool_runner.run = lambda *_args, **_kwargs: _successful_tool_result()
    atomic = SimpleNamespace(
        ref="atomic://locate@1.0.0",
        outputs=[ParameterSpec("result", "entity")],
        validator_spec={"output_derivations": {"result": {"kind": "effect_witness"}}},
    )
    binding = SimpleNamespace(
        order=0,
        tool_ref="tool://locate@1.0.0",
        role="locate_step",
        parameter_mapping={},
    )
    compiled = SimpleNamespace(
        atomic=atomic,
        implementation=SimpleNamespace(
            ref="implementation://locate@1.0.0",
            tool_bindings=[binding],
            execution_policy={
                "output_mapping": {
                    "result": {"kind": "constant", "constant": "cup_3"},
                }
            },
        ),
        tools=[SimpleNamespace(ref="tool://locate@1.0.0")],
    )
    ctx = SimpleNamespace(
        world_revision=7,
        trace_builder=_TraceBuilder(),
        binding_store=_BindingStore(),
        atomic_evidence_for=lambda _occurrence: SimpleNamespace(
            authoritative_facts=lambda: []
        ),
        harness=SimpleNamespace(validator_channel=lambda: object()),
    )

    result = runner.run(
        compiled,
        ToolCallPreflightResult(
            True,
            "implementation://locate@1.0.0",
            normalized_arguments={},
        ),
        SimpleNamespace(occurrence_id="occ_locate", step_id="step_locate"),
        ctx,
        agent_prepared=False,
    )

    assert result.atomic_witness_refs == [
        "alfworld_action_fact:r7:entity.discovered_at:entity=cup_3,location=countertop_2"
    ]
    assert ctx.trace_builder.trace.implementation_invocations[0].result[
        "atomic_witness_refs"
    ] == result.atomic_witness_refs


class _StaticValidator:
    def validate_automation_draft(self, *_args: object, **_kwargs: object) -> ValidationResult:
        return ValidationResult.ok("runtime_automation_r0")

    def validate_proposal(self, *_args: object, **_kwargs: object) -> ValidationResult:
        return ValidationResult.ok("tool_static")


class _Compiler:
    def compile_proposal(
        self,
        _occurrence: object,
        atomic: object,
        _proposal: object,
        _provenance: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            atomic=atomic,
            tool=SimpleNamespace(ref="tool://locate@1.0.0", status="draft"),
            implementation=SimpleNamespace(
                ref="implementation://locate@1.0.0", status="draft"
            ),
        )


class _TrialRunner:
    def run(self, *_args: object, **_kwargs: object) -> ImplementationExecutionResult:
        return ImplementationExecutionResult(
            implementation_ref="implementation://locate@1.0.0",
            atomic_ref="atomic://locate@1.0.0",
            preflight_passed=True,
            started=True,
            completed=True,
            atomic_effect_passed=True,
            tool_results=[_successful_tool_result()],
            validated_outputs={
                "entity": "cup_3",
                "location": "countertop_2",
            },
            atomic_witness_refs=[
                "alfworld_action_fact:r7:entity.discovered_at:entity=cup_3,location=countertop_2"
            ],
        )


def test_runtime_trial_separates_atomic_authority_from_tool_path_refs(monkeypatch) -> None:
    draft = RuntimeAutomationAtomicDraft(
        draft_id="draft_locate",
        intent="locate target",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[
            ParameterSpec("entity", "entity"),
            ParameterSpec("location", "location"),
        ],
        preconditions=[],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": "$entity", "location": "$location"},
            effect_domain=EffectDomain.EVIDENCE,
        )],
        rationale="",
        source_occurrence_id="occ_locate",
        input_binding_specs={
            "target": {"kind": "constant", "value": "cup"},
        },
    )
    proposal = ToolProposal(
        proposal_version="1",
        decision="create",
        summary="locate target",
        atomic_ref="atomic://locate@1.0.0",
        inputs=list(draft.inputs),
        outputs=list(draft.outputs),
        program=[{"op": "RETURN", "node_id": "return"}],
        max_actions=1,
        final_effects=list(draft.effects),
        evidence_outputs=[],
        path_expectations=[],
        rationale="",
    )

    class _Builder:
        def __init__(self, _session: object) -> None:
            pass

        def build(self, **_kwargs: object) -> ToolProposal:
            return proposal

    monkeypatch.setattr("atomic_skillgraph.runtime.automation.ToolBuilderSession", _Builder)
    coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=lambda *_args: object(),
        tool_compiler=_Compiler(),
        implementation_runner=_TrialRunner(),
        static_validator=_StaticValidator(),
    )
    ctx = SimpleNamespace(
        harness=SimpleNamespace(
            profile_name="alfworld",
            semantic_predicate_schema=lambda: [],
            primitive_action_schema=lambda: [],
        ),
        tool_evidence_snapshot=lambda: {},
        trace_builder=_TraceBuilder(),
        binding_store=_BindingStore(),
        validated_outputs={},
        runtime_tool_trials={},
        action_history=[],
        task_id="task_locate",
        task=SimpleNamespace(task_type="pick_and_place_simple"),
    )

    outcome = coordinator.process_draft(
        draft=draft,
        ctx=ctx,
        occurrence=SimpleNamespace(occurrence_id="occ_locate"),
    )

    assert outcome.r1_passed is True
    assert outcome.trial is not None
    assert outcome.trial["r1_witness_refs"] == [
        "alfworld_action_fact:r7:entity.discovered_at:entity=cup_3,location=countertop_2"
    ]
    assert outcome.trial["tool_path_witness_refs"] == [
        "semantic_evidence:entity"
    ]
    assert all(
        not ref.startswith("semantic_evidence:")
        for ref in outcome.trial["r1_witness_refs"]
    )
    assert outcome.trial["after_revision"] == 7
    assert outcome.trial["declared_effects"][0]["predicate"] == (
        "entity.discovered_at"
    )
    assert outcome.trial["output_derivations"] == {
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


def test_terminal_runtime_prefix_is_e1_evidence_but_not_tool_admission(
    monkeypatch,
) -> None:
    draft = RuntimeAutomationAtomicDraft(
        draft_id="terminal_locate",
        intent="locate target before task terminal",
        inputs=[ParameterSpec("target", "entity")],
        outputs=[ParameterSpec("entity", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": "$entity"},
            effect_domain=EffectDomain.EVIDENCE,
        )],
        rationale="",
        source_occurrence_id="occ_locate",
        input_binding_specs={
            "target": {"kind": "constant", "value": "cup"},
        },
    )
    proposal = ToolProposal(
        proposal_version="1",
        decision="create",
        summary="locate target",
        atomic_ref="atomic://locate@1.0.0",
        inputs=list(draft.inputs),
        outputs=list(draft.outputs),
        program=[{"op": "RETURN", "node_id": "return"}],
        max_actions=1,
        final_effects=list(draft.effects),
        evidence_outputs=[],
        path_expectations=[],
        rationale="",
    )

    class _Builder:
        def __init__(self, _session: object) -> None:
            pass

        def build(self, **_kwargs: object) -> ToolProposal:
            return proposal

    class _TerminalTrialRunner:
        def run(self, *_args: object, **_kwargs: object) -> ImplementationExecutionResult:
            tool = _successful_tool_result()
            tool.completed = False
            tool.terminal_interrupted = True
            return ImplementationExecutionResult(
                implementation_ref="implementation://locate@1.0.0",
                atomic_ref="atomic://locate@1.0.0",
                preflight_passed=True,
                started=True,
                completed=False,
                atomic_effect_passed=True,
                tool_results=[tool],
                validated_outputs={"entity": "cup_3"},
                terminal_interrupted=True,
                atomic_witness_refs=[
                    "alfworld_action_fact:r7:entity.discovered_at:entity=cup_3"
                ],
            )

    monkeypatch.setattr(
        "atomic_skillgraph.runtime.automation.ToolBuilderSession", _Builder,
    )
    coordinator = RuntimeAutomationCoordinator(
        tool_builder_factory=lambda *_args: object(),
        tool_compiler=_Compiler(),
        implementation_runner=_TerminalTrialRunner(),
        static_validator=_StaticValidator(),
    )
    ctx = SimpleNamespace(
        harness=SimpleNamespace(
            profile_name="alfworld",
            semantic_predicate_schema=lambda: [],
            primitive_action_schema=lambda: [],
        ),
        tool_evidence_snapshot=lambda: {},
        trace_builder=_TraceBuilder(),
        binding_store=_BindingStore(),
        validated_outputs={},
        runtime_tool_trials={},
        action_history=[],
        task_id="task_locate",
        task=SimpleNamespace(task_type="pick_and_place_simple"),
    )

    outcome = coordinator.process_draft(
        draft=draft,
        ctx=ctx,
        occurrence=SimpleNamespace(occurrence_id="occ_locate"),
    )

    assert outcome.r1_passed is False
    assert outcome.trial is not None
    assert outcome.trial["r1"]["admission_eligible"] is False
    assert outcome.trial["r1"]["e1_effect_eligible"] is True
    assert outcome.trial["declared_effects"][0]["predicate"] == (
        "entity.discovered_at"
    )
    assert outcome.trial["after_revision"] == 7


def _builder_atomic() -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef("atomic_navigate", "1.0.0"),
        "navigate to each destination",
        [ParameterSpec("destination", "entity")],
        [ParameterSpec("result", "entity")],
        [],
        [SemanticPredicate(
            "agent.at_location",
            {"location": "$destination"},
        )],
        {
            "output_derivations": {
                "result": {
                    "kind": "input_identity",
                    "input_role": "destination",
                }
            }
        },
        [],
        {},
        {},
    )


def test_tool_builder_context_exposes_only_bounded_occurrence_authority() -> None:
    witness = "alfworld_action_fact:r2:agent.at_location:location=cabinet_1"
    prompt = ContextBuilder().tool_builder(
        atomic=_builder_atomic(),
        provenance=SimpleNamespace(source="success_evolution"),
        evidence_support=[{
            "event_id": "event_1",
            "action_type": "GO_TO",
            "arguments": {"destination": "cabinet_1"},
            "accepted": True,
            "before_revision": 1,
            "after_revision": 2,
            "observation": "SECRET_RAW_OBSERVATION",
            "task_goal": "SECRET_FULL_TASK_GOAL",
            "authoritative_positive_effects": [{
                "predicate": "agent.at_location",
                "args": {"location": "cabinet_1"},
                "effect_domain": "world",
                "witness_ref": witness,
                "revision": 2,
            }],
        }],
        semantic_delta={},
        harness_interface={},
    )
    _instruction, raw_payload = prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)
    payload = json.loads(raw_payload)

    assert payload["atomic_output_derivations"] == {
        "result": {
            "kind": "input_identity",
            "input_role": "destination",
        }
    }
    assert payload["atomic_effect_witness_refs"] == [witness]
    assert payload["historical_loop_evidence_required"] is True
    assert payload["atomic_evidence_support"] == [{
        "event_id": "event_1",
        "action_type": "GO_TO",
        "arguments": {"destination": "cabinet_1"},
        "accepted": True,
        "before_revision": 1,
        "after_revision": 2,
        "authoritative_positive_effects": [{
            "predicate": "agent.at_location",
            "args": {"location": "cabinet_1"},
            "effect_domain": "world",
            "witness_ref": witness,
            "revision": 2,
        }],
    }]
    assert "SECRET_RAW_OBSERVATION" not in prompt
    assert "SECRET_FULL_TASK_GOAL" not in prompt


def _loop_proposal() -> ToolProposal:
    atomic = _builder_atomic()
    return ToolProposal(
        proposal_version="1",
        decision="create",
        summary="navigate over destinations",
        atomic_ref=str(atomic.ref),
        inputs=list(atomic.inputs),
        outputs=list(atomic.outputs),
        program=[
            {
                "node_id": "visit_destinations",
                "op": "FOR_EACH",
                "collection_source": {
                    "source": "local_deterministic",
                    "values": ["candidate_a", "candidate_b"],
                },
                "iteration_variable": "candidate",
                "max_iterations": 2,
                "body": [{
                    "node_id": "visit_one",
                    "op": "ACTION",
                    "action_type": "GO_TO",
                    "argument_mapping": {
                        "destination": {
                            "kind": "local_variable",
                            "source_role": "candidate",
                        }
                    },
                    "expected_effects": [{
                        "predicate": "agent.at_location",
                        "args": {"location": "$destination"},
                        "effect_domain": "world",
                    }],
                }],
            },
            {
                "node_id": "return_result",
                "op": "RETURN",
                "output_sources": {
                    "result": {
                        "source": "tool_input",
                        "field": "destination",
                    }
                },
            },
        ],
        max_actions=2,
        final_effects=list(atomic.effects),
        evidence_outputs=[],
        path_expectations=[],
        rationale="",
    )


def test_historical_tool_builder_loop_requires_two_distinct_repetitions() -> None:
    harness = SimpleNamespace(
        semantic_predicate_schema=lambda: [{
            "predicate": "agent.at_location",
            "argument_roles": ["location"],
            "effect_domain": "world",
        }],
        primitive_action_schema=lambda: [{
            "action_type": "GO_TO",
            "argument_roles": ["destination"],
        }],
    )
    one = [{
        "action_type": "GO_TO",
        "arguments": {"destination": "cabinet_1"},
        "accepted": True,
    }]
    two = [
        *one,
        {
            "action_type": "GO_TO",
            "arguments": {"destination": "drawer_2"},
            "accepted": True,
        },
    ]
    validator = ToolStaticValidator()

    runtime_report = validator.validate_proposal(
        _loop_proposal(), _builder_atomic(), harness,
    )
    one_history_report = validator.validate_proposal(
        _loop_proposal(), _builder_atomic(), harness,
        historical_evidence_support=one,
    )
    two_history_report = validator.validate_proposal(
        _loop_proposal(), _builder_atomic(), harness,
        historical_evidence_support=two,
    )

    assert runtime_report.passed is True
    assert one_history_report.passed is False
    assert "tool_ir_historical_loop_evidence_insufficient" in (
        one_history_report.failure_codes
    )
    assert two_history_report.passed is True


def _invalid_bundle() -> PlannerRequirementBundle:
    return PlannerRequirementBundle([CapabilityRequirement(
        requirement_id="req_invalid",
        intent="invalid requirement",
        desired_effects=[SemanticPredicate("object.heated", {"object": "target"})],
        expected_inputs=[ParameterSpec("target", "entity")],
        expected_outputs=[],
        precondition_hints=[],
        semantic_variants=[],
        required=True,
        rationale="",
    )])


def test_invalid_p1_bundle_is_repairable_without_search_results() -> None:
    decision = RepairabilityGate().decide(
        _invalid_bundle(),
        ValidationResult.fail(
            "planner_requirement_bundle",
            "planner_requirement_multiplicity_invalid",
            "invalid",
        ),
        (),
    )

    assert decision.repairable is True
    assert decision.reason_code == "planner_requirement_bundle_invalid"
    assert decision.requirement_ids == ("req_invalid",)


def test_pipeline_routes_invalid_p1_bundle_through_repairability_gate(monkeypatch) -> None:
    bundle = _invalid_bundle()
    calls = {"gate": 0, "repair": 0}

    class _RequirementAgent:
        def __init__(self, _session: object) -> None:
            pass

        def propose(self, *_args: object, **_kwargs: object) -> PlannerRequirementBundle:
            return bundle

        def repair(self, *_args: object, **_kwargs: object) -> PlannerRequirementBundle:
            calls["repair"] += 1
            return bundle

    class _RepairabilityGate:
        def decide(self, _bundle, validation, search, hints=()):
            calls["gate"] += 1
            assert validation.passed is False
            assert list(search) == []
            assert list(hints) == []
            return RepairabilityDecision(
                True,
                "planner_requirement_bundle_invalid",
                ("req_invalid",),
                (),
                (),
            )

    class _CompositeRetriever:
        def retrieve_complete(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(candidates=[], audit_candidates=[], rejections=[])

        def retrieve_terminal(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                terminal_empirical_audit=[],
                terminal_empirical_candidates=[],
            )

    monkeypatch.setattr(
        "atomic_skillgraph.planner.pipeline.RequirementAgent",
        _RequirementAgent,
    )
    pipeline = object.__new__(PlannerPipeline)
    pipeline.skills = SimpleNamespace(
        list_refs=lambda kind, **_kwargs: ["atomic"] if kind == "atomic" else []
    )
    pipeline.graph = object()
    pipeline.session_factory = lambda *_args: object()
    pipeline.composite_retriever = _CompositeRetriever()
    pipeline.requirement_validator = SimpleNamespace(
        validate=lambda *_args, **_kwargs: ValidationResult.fail(
            "planner_requirement_bundle",
            "planner_requirement_multiplicity_invalid",
            "invalid",
        )
    )
    pipeline.repairability_gate = _RepairabilityGate()
    pipeline.max_repeat_count = 4
    pipeline.max_occurrences = 16
    pipeline.cold_start_enabled = False
    task = SimpleNamespace(task_id="task_invalid", goal="invalid")
    harness = SimpleNamespace(
        profile_name="alfworld",
        task_contract=lambda _task: TaskContract(),
    )

    plan = pipeline.build_plan(task, harness)

    assert calls == {"gate": 1, "repair": 1}
    assert plan.planner_audit["repairability"]["reason_code"] == (
        "planner_requirement_bundle_invalid"
    )
