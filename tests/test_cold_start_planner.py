from __future__ import annotations

from types import SimpleNamespace

from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
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
from atomic_skillgraph.core.results import ValidationResult
from atomic_skillgraph.knowledge.failure_knowledge_store import ProvisionalStatus
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
        expected_effects=[],
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
            expected_effects=[],
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
