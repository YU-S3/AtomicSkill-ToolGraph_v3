from __future__ import annotations

from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.protocol import (
    AgentTurn,
    NativeToolCall,
    SchemaValidationError,
    validate_schema_instance,
)
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    CapabilityRequirement,
    ColdStartCandidateSource,
    ColdStartExecutionMode,
    ColdStartPlanProposal,
    ColdStartPlanStep,
    ParameterSpec,
    RepeatBlock,
    SemanticPredicate,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import ValidationResult
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import RuntimeMode
from atomic_skillgraph.knowledge.failure_knowledge_store import ProvisionalStatus
from atomic_skillgraph.planner.cold_start_agent import (
    COLD_START_PLAN_SCHEMA,
    cold_start_plan_from_dict,
)
from atomic_skillgraph.planner.cold_start_validator import ColdStartPlanValidator
from atomic_skillgraph.planner.multiplicity import (
    RequirementExpansion,
    RequirementInstance,
)
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.cold_start_executor import ProvisionalNodeExecutor
from atomic_skillgraph.runtime.node_executor import NodeExecutor
from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
from atomic_skillgraph.traces.schema import (
    ColdStartStepRecord,
    TaskRecord,
    TraceBuilder,
    TraceRecord,
)


def _requirement() -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id="move",
        intent="move one object to its destination",
        desired_effects=[SemanticPredicate(
            "object_at_location",
            {
                "object": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="object",
                ),
                "location": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="location",
                ),
            },
        )],
        expected_inputs=[
            ParameterSpec("object", "entity", True, "semantic"),
            ParameterSpec("location", "entity", True, "semantic"),
        ],
        expected_outputs=[],
        precondition_hints=[],
        semantic_variants=[],
        required=True,
        rationale="test",
    )


def _expansion() -> RequirementExpansion:
    requirement = _requirement()
    instances = tuple(
        RequirementInstance(
            f"single::{index}",
            requirement.requirement_id,
            "",
            -1,
            requirement,
        )
        for index in range(3)
    )
    return RequirementExpansion(
        templates=(requirement,),
        repeat_blocks=(),
        instances=instances,
        instance_ids_by_template={
            requirement.requirement_id: tuple(
                item.instance_id for item in instances
            )
        },
    )


def _step(
    index: int,
    source: ColdStartCandidateSource,
    ref: str,
    mode: ColdStartExecutionMode,
) -> ColdStartPlanStep:
    return ColdStartPlanStep(
        step_id=f"step-{index}",
        requirement_instance_ids=[f"single::{index}"],
        candidate_source=source,
        candidate_ref=ref,
        execution_mode=mode,
        binding_specs={},
        repeat_role_bindings={},
    )


def _c1_payload() -> dict:
    return {
        "plan_id": "cold-plan",
        "steps": [{
            "step_id": "step-0",
            "requirement_instance_ids": ["single::0"],
            "candidate_source": "verified",
            "candidate_ref": "skill://verified@1.0.0",
            "execution_mode": "direct_or_seeded",
            "binding_specs": {},
            "repeat_role_bindings": {},
        }],
        "control_sequence": ["step-0"],
        "data_edges": [],
        "dependency_edges": [],
        "requirement_coverage": {"single::0": ["step-0"]},
        "referenced_failure_experience_ids": [],
    }


def test_c1_schema_and_parser_reject_model_effect_authority() -> None:
    payload = _c1_payload()
    validate_schema_instance(payload, COLD_START_PLAN_SCHEMA)
    assert cold_start_plan_from_dict(payload).steps[0].expected_effects == []

    forged = {
        **payload,
        "steps": [{
            **payload["steps"][0],
            "expected_effects": [{
                "predicate": "object_cleaned",
                "args": {"object": "forged-model-value"},
            }],
        }],
    }
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_schema_instance(forged, COLD_START_PLAN_SCHEMA)

    # Old Trace payloads may still carry the compatibility projection.  The
    # typed reconstruction must ignore it rather than restore model authority.
    assert cold_start_plan_from_dict(forged).steps[0].expected_effects == []


def test_cold_start_runtime_effects_are_candidate_authoritative() -> None:
    verified_effect = SemanticPredicate("agent_holds", {"object": "apple"})
    provisional_effect = SemanticPredicate(
        "object_at_location",
        {"object": "apple", "location": "table"},
    )
    forged_effect = SemanticPredicate("object_cleaned", {"object": "apple"})
    verified_ref = "skill://verified_effect@1.0.0"
    provisional_ref = "provisional://effect@1.0.0"
    steps = [
        ColdStartPlanStep(
            "verified-step",
            ["single::0"],
            ColdStartCandidateSource.VERIFIED,
            verified_ref,
            ColdStartExecutionMode.DIRECT_OR_SEEDED,
            {},
            {},
            [forged_effect],
        ),
        ColdStartPlanStep(
            "provisional-step",
            ["single::1"],
            ColdStartCandidateSource.PROVISIONAL,
            provisional_ref,
            ColdStartExecutionMode.SEEDED_ONLY,
            {},
            {},
            [forged_effect],
        ),
    ]
    proposal = ColdStartPlanProposal(
        "authority-plan",
        steps,
        [item.step_id for item in steps],
        [],
        [],
        {"single::0": ["verified-step"], "single::1": ["provisional-step"]},
        [],
    )
    verified_atomic = SimpleNamespace(
        ref=SkillRef.parse(verified_ref),
        effects=[verified_effect],
    )
    provisional = SimpleNamespace(
        provisional_ref=provisional_ref,
        canonical_intent="place an object",
        atomic_contract={
            "inputs": [],
            "outputs": [],
            "preconditions": [],
            "effects": [provisional_effect],
            "validator_spec": {},
        },
        seeded_guideline={},
        harness_profile="fake",
    )
    skills = SimpleNamespace(
        get_atomic=lambda _ref: verified_atomic,
        implementations_for=lambda *_args, **_kwargs: [],
    )
    orchestrator = RuntimeOrchestrator.__new__(RuntimeOrchestrator)
    orchestrator.invocation_compiler = SimpleNamespace(skills=skills)
    orchestrator.failure_knowledge = SimpleNamespace(
        get_provisional=lambda _ref: provisional,
    )
    ctx = SimpleNamespace(
        task_contract=SimpleNamespace(),
        task_progress=SimpleNamespace(
            policy_view=lambda: {
                "targets": [],
                "unsatisfied_identity_constraint_count": 0,
            },
        ),
        plan=SimpleNamespace(
            task_id="task",
            cold_start_plan=proposal,
            planner_audit={},
            repeat_constraints=[],
            cold_start_scaffold={},
        ),
    )

    runtime_plan, _ = orchestrator._materialize_cold_execution_plan(
        ctx,
        mode=RuntimeMode.ONLINE,
    )
    by_step = {item.step_id: item for item in runtime_plan.occurrences}
    assert by_step["verified-step"].expected_effects == [verified_effect]
    assert by_step["provisional-step"].expected_effects == [
        provisional_effect
    ]
    assert all(
        forged_effect not in item.expected_effects
        for item in runtime_plan.occurrences
    )

    completed_local_effects = [{
        "step_id": item.step_id,
        "validated_effects": to_primitive(item.expected_effects),
    } for item in runtime_plan.occurrences]
    continuation = orchestrator._cold_continuation_context(
        ctx,
        completed_local_effects,
    )
    # ContextBuilder.dynamic_task already carries the current policy-safe
    # TaskProgress.  The cold-continuation side context must not duplicate it.
    assert "task_progress" not in continuation
    assert continuation["completed_local_effects"] == completed_local_effects
    assert continuation["completed_local_effects"][0]["validated_effects"] == (
        to_primitive([verified_effect])
    )
    assert continuation["completed_local_effects"][1]["validated_effects"] == (
        to_primitive([provisional_effect])
    )

    captured: dict = {}

    def run_dynamic(*_args, **kwargs) -> dict:
        captured.update(kwargs["continuation_context"])
        return {"success": False, "failure_code": ""}

    failed_terminal = ValidationResult.fail(
        "task",
        "task_contract_unsatisfied",
        "continue",
        task_contract=False,
    )
    orchestrator.validation = SimpleNamespace(
        task=SimpleNamespace(terminal=lambda *_args, **_kwargs: failed_terminal),
    )
    orchestrator._run_verified_cold_step = (
        lambda *_args, **_kwargs: (True, "", "success")
    )
    orchestrator.provisional_node_executor = SimpleNamespace(
        execute=lambda *_args, **_kwargs: SimpleNamespace(
            local_effect_passed=True,
            failure_code="",
        ),
    )
    orchestrator.node_executor = SimpleNamespace(run_dynamic=run_dynamic)
    orchestrator._finish_cold_start = (
        lambda _ctx, _terminal, *, dynamic_result: dynamic_result
    )
    ctx.plan.cold_start_scaffold = {
        "executable_step_ids": [item.step_id for item in steps],
    }
    ctx.trace_builder = _trace_builder()
    ctx.world_revision = 0
    ctx.validated_outputs = {}
    ctx.binding_store = SimpleNamespace(
        apply_data_flow=lambda *_args, **_kwargs: None,
    )
    ctx.budget = SimpleNamespace(
        begin_node=lambda *_args, **_kwargs: None,
        end_node=lambda *_args, **_kwargs: None,
    )
    ctx.task_progress = SimpleNamespace(
        record=lambda source: SimpleNamespace(
            progress_digest=f"progress::{source}",
        ),
        snapshot=lambda: {"digest": "progress"},
        policy_view=lambda: {
            "targets": [],
            "unsatisfied_identity_constraint_count": 0,
        },
    )
    ctx.harness = SimpleNamespace(
        validator_channel=lambda: SimpleNamespace(won=False),
    )

    orchestrator._run_cold_start(ctx, mode=RuntimeMode.ONLINE)
    assert captured["completed_local_effects"][0]["validated_effects"] == (
        to_primitive([verified_effect])
    )
    assert captured["completed_local_effects"][1]["validated_effects"] == (
        to_primitive([provisional_effect])
    )


def test_c1_candidate_authority_seeded_only_and_longest_prefix() -> None:
    expansion = _expansion()
    proposal = ColdStartPlanProposal(
        plan_id="cold-plan",
        steps=[
            _step(
                0,
                ColdStartCandidateSource.VERIFIED,
                "skill://verified@1.0.0",
                ColdStartExecutionMode.DIRECT_OR_SEEDED,
            ),
            _step(
                1,
                ColdStartCandidateSource.PROVISIONAL,
                "provisional://atomic_x@1.0.0",
                ColdStartExecutionMode.SEEDED_ONLY,
            ),
            _step(
                2,
                ColdStartCandidateSource.UNRESOLVED,
                "",
                ColdStartExecutionMode.DYNAMIC,
            ),
        ],
        control_sequence=["step-0", "step-1", "step-2"],
        data_edges=[],
        dependency_edges=[],
        requirement_coverage={
            f"single::{index}": [f"step-{index}"] for index in range(3)
        },
        referenced_failure_experience_ids=[],
    )
    validator = ColdStartPlanValidator()
    report = validator.validate(
        proposal,
        expansion,
        verified_candidates={
            "single::0": {"skill://verified@1.0.0"},
        },
        provisional_candidates={
            "single::1": {"provisional://atomic_x@1.0.0"},
        },
        failure_experience_ids=set(),
    )
    assert report.passed
    scaffold = validator.scaffold(proposal)
    assert scaffold.executable_step_ids == ["step-0", "step-1"]
    assert scaffold.first_unresolved_step_id == "step-2"

    proposal.steps[1].execution_mode = ColdStartExecutionMode.DIRECT_OR_SEEDED
    rejected = validator.validate(
        proposal,
        expansion,
        verified_candidates={
            "single::0": {"skill://verified@1.0.0"},
        },
        provisional_candidates={
            "single::1": {"provisional://atomic_x@1.0.0"},
        },
        failure_experience_ids=set(),
    )
    assert not rejected.passed
    assert rejected.checks["candidate_source_mode_valid"] is False

    proposal.steps[1].execution_mode = ColdStartExecutionMode.SEEDED_ONLY
    proposal.steps[1].binding_specs = {
        "object": BindingExpression(
            BindingExprKind.DATA_FLOW,
            source_role="object",
            source_step="step-0",
        )
    }
    statically_blocked = validator.scaffold(proposal)
    assert statically_blocked.executable_step_ids == ["step-0"]
    assert statically_blocked.first_unresolved_step_id == "step-1"


def test_c1_repeat_role_mapping_is_complete_and_candidate_authoritative() -> None:
    requirement = _requirement()
    block = RepeatBlock(
        block_id="repeat",
        count=2,
        ordered_requirement_ids=("move",),
        distinct_roles=("object",),
        shared_roles=("location",),
        basis_constraint_id="constraint",
        basis_role_map={"object": "object", "location": "location"},
    )
    instances = tuple(
        RequirementInstance(
            f"repeat::{index}::move",
            "move",
            "repeat",
            index,
            requirement,
        )
        for index in range(2)
    )
    expansion = RequirementExpansion(
        templates=(requirement,),
        repeat_blocks=(block,),
        instances=instances,
        instance_ids_by_template={
            "move": tuple(item.instance_id for item in instances),
        },
    )
    steps = [
        ColdStartPlanStep(
            step_id=f"repeat-step-{index}",
            requirement_instance_ids=[instance.instance_id],
            candidate_source=ColdStartCandidateSource.VERIFIED,
            candidate_ref="skill://verified@1.0.0",
            execution_mode=ColdStartExecutionMode.DIRECT_OR_SEEDED,
            binding_specs={},
            repeat_role_bindings={
                "object": "object",
                "location": "location",
            },
        )
        for index, instance in enumerate(instances)
    ]
    proposal = ColdStartPlanProposal(
        "repeat-plan",
        steps,
        [item.step_id for item in steps],
        [],
        [],
        {
            instance.instance_id: [steps[index].step_id]
            for index, instance in enumerate(instances)
        },
        [],
    )
    verified = {
        instance.instance_id: {"skill://verified@1.0.0"}
        for instance in instances
    }
    validator = ColdStartPlanValidator()
    accepted = validator.validate(
        proposal,
        expansion,
        verified_candidates=verified,
        provisional_candidates={},
        failure_experience_ids=set(),
        candidate_roles={
            "skill://verified@1.0.0": {"object", "location"},
        },
    )
    assert accepted.passed

    steps[1].repeat_role_bindings = {"object": "object"}
    rejected = validator.validate(
        proposal,
        expansion,
        verified_candidates=verified,
        provisional_candidates={},
        failure_experience_ids=set(),
        candidate_roles={
            "skill://verified@1.0.0": {"object", "location"},
        },
    )
    assert not rejected.passed
    assert rejected.checks["repeat_role_bindings_valid"] is False


class _StatusSession:
    def __init__(self, session_id: str, offered: list[list[str]]) -> None:
        self.session_id = session_id
        self.offered = offered

    def next_turn(self, prompt: str, *, tools: list[object]) -> AgentTurn:
        self.offered.append([item.name for item in tools])
        return AgentTurn(
            "",
            [NativeToolCall("call-1", "report_runtime_status", {"status": "give_up"})],
            "tool_calls",
            1,
            1,
            2,
            0,
            1.0,
        )

    def finalize_tool_result(self, call_id: str, result: dict) -> None:
        return None

    def snapshot(self) -> dict:
        return {"session_id": self.session_id}


class _Progress:
    def record(self, source: str) -> SimpleNamespace:
        return SimpleNamespace(progress_digest=f"progress::{source}")


class _Budget:
    def snapshot(self) -> dict:
        return {"remaining_global_actions": 10}


def _trace_builder() -> TraceBuilder:
    return TraceBuilder(TraceRecord.create(
        TaskRecord("task", "fake", "goal", "type", "sig", {}),
        {},
        {},
        {"source": "cold_start", "failure_stage": "runtime"},
    ))


def test_provisional_scaffold_session_has_no_learned_invocation_tool() -> None:
    offered: list[list[str]] = []
    node_executor = NodeExecutor(
        SimpleNamespace(),
        SimpleNamespace(tool=SimpleNamespace()),
        lambda kind, occurrence_id: _StatusSession(
            f"{kind}::{occurrence_id}",
            offered,
        ),
    )
    binding_store = RuntimeBindingStore()
    ctx = SimpleNamespace(
        binding_store=binding_store,
        world_revision=0,
        trace_builder=_trace_builder(),
        task_goal="goal",
        observation="observation",
        action_catalog=[],
        budget=_Budget(),
        validated_outputs={},
        evidence_store=SimpleNamespace(),
        relevant_history=lambda occurrence_id: [],
    )
    provisional = SimpleNamespace(
        provisional_ref="provisional://atomic_x@1.0.0",
        contract_signature="x",
        canonical_intent="move an object",
        atomic_contract={
            "inputs": [],
            "outputs": [],
            "preconditions": [],
            "effects": [],
            "validator_spec": {},
        },
        seeded_guideline={"strategy": "use current affordances"},
        harness_profile="fake",
        status=ProvisionalStatus.TRIAL_READY,
    )
    step = _step(
        1,
        ColdStartCandidateSource.PROVISIONAL,
        provisional.provisional_ref,
        ColdStartExecutionMode.SEEDED_ONLY,
    )
    ProvisionalNodeExecutor(node_executor).execute(
        provisional,
        ctx,
        step,
        progress_tracker=_Progress(),
    )
    assert len(offered) == 1
    assert "environment_action" in offered[0]
    assert all(not name.startswith("invoke_") for name in offered[0])


def test_cold_continuation_session_is_fresh_and_never_graph_credit() -> None:
    created: list[_StatusSession] = []

    def session_factory(kind: str, occurrence_id: str) -> _StatusSession:
        session = _StatusSession(f"session-{len(created)}", [])
        created.append(session)
        assert kind == "runtime_dynamic_cold_start_continuation"
        return session

    terminal_failed = ValidationResult.fail(
        "task",
        "task_contract_unsatisfied",
        "not yet",
        task_contract=False,
    )
    validation = SimpleNamespace(
        task=SimpleNamespace(terminal=lambda *args, **kwargs: terminal_failed),
        tool=SimpleNamespace(),
    )
    executor = NodeExecutor(SimpleNamespace(), validation, session_factory)
    ctx = SimpleNamespace(
        task_goal="goal",
        observation="observation",
        action_catalog=[],
        action_history=[],
        budget=_Budget(),
        trace_builder=_trace_builder(),
        harness=SimpleNamespace(
            validator_channel=lambda: SimpleNamespace(won=False),
        ),
        task_contract=SimpleNamespace(),
    )
    first = executor.run_dynamic(
        ctx,
        cold_start_continuation=True,
        continuation_context={"task_progress": {}},
    )
    second = executor.run_dynamic(
        ctx,
        cold_start_continuation=True,
        continuation_context={"task_progress": {}},
    )
    assert first["cold_start_continuation"] is True
    assert second["cold_start_continuation"] is True
    assert [item.session_id for item in created] == ["session-0", "session-1"]

    orchestrator = RuntimeOrchestrator.__new__(RuntimeOrchestrator)
    orchestrator.validation = SimpleNamespace()
    channel = SimpleNamespace(won=True)
    finish_ctx = SimpleNamespace(
        trace_builder=_trace_builder(),
        harness=SimpleNamespace(validator_channel=lambda: channel),
        task_progress=SimpleNamespace(record=lambda source: None),
    )
    finish_ctx.trace_builder.trace.cold_start_steps.append(
        ColdStartStepRecord(
            "step-0",
            "verified",
            "skill://verified@1.0.0",
            "direct_or_seeded",
            "success",
            True,
            0,
            1,
            "before",
            "after",
            "",
        )
    )
    trace = orchestrator._finish_cold_start(
        finish_ctx,
        SimpleNamespace(checks={"task_contract": True}),
        dynamic_result={"success": True, "failure_code": ""},
    )
    assert trace.strict_task_success is True
    assert trace.cold_start_assisted_success is True
    assert trace.graph_self_sufficient_success is False
