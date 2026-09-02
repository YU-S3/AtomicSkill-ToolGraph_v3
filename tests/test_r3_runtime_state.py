from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any, Iterable

from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeLinearPlan, RuntimeOccurrence
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.harness.alfworld import (
    AlfWorldValidatorChannel,
    semantic_value_compatible,
)
from atomic_skillgraph.harness.protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
)
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.evidence_store import GroundingEvidenceStore
from atomic_skillgraph.runtime.grounding_state import IncrementalGroundingAuthority
from atomic_skillgraph.runtime.loop_guard import ActionLoopGuard
from atomic_skillgraph.runtime.node_executor import NodeExecutor
from atomic_skillgraph.runtime.state import OccurrenceAtomicEvidenceState
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord
from atomic_skillgraph.validation.engine import ValidationEngine


_GROUNDING_FIELDS = {
    "revision",
    "occurrence_id",
    "semantic_anchors",
    "confirmed_bindings",
    "candidate_bindings",
    "missing_bindings",
    "invalidated_bindings",
    "precondition_status",
    "effect_witness_status",
    "learned_invocation_ready",
    "blocking_reasons",
}


def _action(
    action_id: str,
    revision: int,
    action_type: str,
    arguments: dict[str, Any],
) -> HarnessActionSpec:
    return HarnessActionSpec(
        action_id=action_id,
        revision=revision,
        action_type=action_type,
        arguments=dict(arguments),
        display_text=action_type.casefold(),
        raw_action=action_type.casefold(),
        metadata={},
    )


def _occurrence(
    *,
    binding_specs: dict[str, BindingExpression] | None = None,
) -> RuntimeOccurrence:
    return RuntimeOccurrence(
        step_id="effect_step",
        occurrence_id="occ_effect",
        node_ref=SkillRef("r3_runtime_state_fixture", "1.0.0"),
        requirement_ids=[],
        binding_specs=dict(binding_specs or {}),
        implementation_candidates=[],
        expected_effects=[],
    )


def _atomic(
    effects: Iterable[SemanticPredicate],
    *,
    required_resolution: str = "concrete",
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef("r3_runtime_state_fixture", "1.0.0"),
        summary="establish the declared generic effects",
        inputs=[ParameterSpec(
            "object",
            "entity",
            runtime_resolvable=True,
            required_resolution=required_resolution,
        )],
        outputs=[],
        preconditions=[],
        effects=list(effects),
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )


def _effect(predicate: str) -> SemanticPredicate:
    return SemanticPredicate(
        predicate,
        {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="object",
            ),
        },
    )


def _run_two_effect_actions(
    action_types: tuple[str, str],
) -> tuple[
    AlfWorldValidatorChannel,
    OccurrenceAtomicEvidenceState,
    Any,
]:
    channel = AlfWorldValidatorChannel()
    channel.reset()
    state = OccurrenceAtomicEvidenceState.begin(
        "occ_effect", 0, channel.snapshot(),
    )
    for revision, action_type in enumerate(action_types, start=1):
        channel.record(
            _action(
                f"a{revision}",
                revision - 1,
                action_type,
                {"object": "pen_1"},
            ),
            accepted=True,
            revision=revision,
            done=False,
            won=False,
        )
        state.reconcile(
            channel.snapshot(),
            revision=revision,
            accepted=True,
        )

    resolution = ValidationEngine().atomic.resolve_current_effect(
        _atomic((_effect("agent.holds"), _effect("object.heated"))),
        _occurrence(),
        {},
        channel,
        semantic_anchors={"object": "pen"},
        preferred_values=[],
        current_revision=2,
        authoritative_evidence_facts=state.authoritative_facts(),
    )
    return channel, state, resolution


def _fact_id(fact: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        str(fact["predicate"]),
        tuple(sorted(
            (str(role), str(value))
            for role, value in dict(fact["args"]).items()
        )),
    )


def test_occurrence_evidence_merges_across_actions_is_order_invariant_and_invalidates() -> None:
    first_channel, first_state, first_resolution = _run_two_effect_actions(
        ("TAKE", "HEAT"),
    )
    _second_channel, second_state, second_resolution = _run_two_effect_actions(
        ("HEAT", "TAKE"),
    )

    expected = {
        ("agent.holds", (("object", "pen_1"),)),
        ("object.heated", (("object", "pen_1"),)),
    }
    assert {_fact_id(item) for item in first_state.authoritative_facts()} == expected
    assert {
        _fact_id(item) for item in second_state.authoritative_facts()
    } == expected
    assert first_resolution.passed is True
    assert second_resolution.passed is True
    assert any("agent.holds" in item for item in first_resolution.witness_refs)
    assert any("object.heated" in item for item in first_resolution.witness_refs)
    assert any("agent.holds" in item for item in second_resolution.witness_refs)
    assert any("object.heated" in item for item in second_resolution.witness_refs)

    # PUT is an accepted real transition that makes agent.holds false while
    # leaving the independent heated property true.
    first_channel.record(
        _action(
            "a3",
            2,
            "PUT",
            {"object": "pen_1", "destination": "countertop_1"},
        ),
        accepted=True,
        revision=3,
        done=False,
        won=False,
    )
    first_state.reconcile(
        first_channel.snapshot(),
        revision=3,
        accepted=True,
    )
    active = {_fact_id(item) for item in first_state.authoritative_facts()}
    invalidated = {
        _fact_id(item) for item in first_state.invalidated_facts()
    }
    assert ("agent.holds", (("object", "pen_1"),)) not in active
    assert ("agent.holds", (("object", "pen_1"),)) in invalidated
    assert ("object.heated", (("object", "pen_1"),)) in active
    assert first_state.invalidated_facts()[0]["invalidated_at_revision"] == 3

    stale_resolution = ValidationEngine().atomic.resolve_current_effect(
        _atomic((_effect("agent.holds"), _effect("object.heated"))),
        _occurrence(),
        {},
        first_channel,
        semantic_anchors={"object": "pen"},
        preferred_values=[],
        current_revision=3,
        authoritative_evidence_facts=first_state.authoritative_facts(),
    )
    assert stale_resolution.passed is False


def _candidate_catalog(
    revision: int,
    values: Iterable[str],
) -> list[HarnessActionSpec]:
    return [
        _action(
            f"r{revision:03d}_candidate_{index}",
            revision,
            "TAKE",
            {"object": value},
        )
        for index, value in enumerate(values)
    ]


def _grounding_fixture(
    values: Iterable[str],
) -> tuple[
    IncrementalGroundingAuthority,
    AbstractAtomicSkill,
    RuntimeOccurrence,
    Any,
    list[dict[str, Any]],
]:
    channel = AlfWorldValidatorChannel()
    channel.reset()
    catalog = _candidate_catalog(0, values)
    bindings = RuntimeBindingStore()
    bindings.bind_task_value("object", "pen", "entity", 0)
    occurrence = _occurrence(binding_specs={
        "object": BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role="object",
        ),
    })
    bindings.resolve_occurrence_specs(occurrence, 0)
    evidence = GroundingEvidenceStore()
    evidence.replace_action_catalog(catalog, 0)
    snapshots: list[dict[str, Any]] = []
    harness = SimpleNamespace(
        validator_channel=lambda: channel,
        semantic_value_compatible=semantic_value_compatible,
    )
    ctx = SimpleNamespace(
        world_revision=0,
        action_catalog=catalog,
        binding_store=bindings,
        evidence_store=evidence,
        task_contract=TaskContract(),
        harness=harness,
        record_grounding_state=lambda _occurrence_id, state: snapshots.append(
            copy.deepcopy(state)
        ),
    )
    authority = IncrementalGroundingAuthority(
        SimpleNamespace(), ValidationEngine(),
    )
    return (
        authority,
        _atomic((_effect("agent.holds"),)),
        occurrence,
        ctx,
        snapshots,
    )


def test_grounding_refresh_records_every_field_and_confirms_only_unique_candidate() -> None:
    authority, atomic, occurrence, ctx, snapshots = _grounding_fixture(
        ("pen_1", "mug_1"),
    )

    first = authority.refresh(occurrence, atomic, [], ctx)
    assert _GROUNDING_FIELDS.issubset(first)
    assert first["semantic_anchors"] == {"object": "pen"}
    assert first["confirmed_bindings"] == {"object": "pen_1"}
    assert first["candidate_bindings"] == {}
    assert first["missing_bindings"] == []
    committed = ctx.binding_store.snapshot_for_node(occurrence)["object"]
    assert committed.source is BindingSource.HARNESS_EVIDENCE
    assert committed.status is BindingStatus.GROUNDED
    assert committed.resolution is BindingResolution.CONCRETE

    # A later world step invalidates concrete proof.  The next refresh must
    # write another complete snapshot and re-confirm only from that revision's
    # authoritative catalog.
    ctx.world_revision = 1
    ctx.binding_store.invalidate_revision(1)
    ctx.action_catalog = _candidate_catalog(1, ("pen_2",))
    ctx.evidence_store.replace_action_catalog(ctx.action_catalog, 1)
    ctx.harness.validator_channel().revision = 1
    second = authority.refresh(occurrence, atomic, [], ctx)

    assert len(snapshots) == 2
    assert [item["revision"] for item in snapshots] == [0, 1]
    assert all(_GROUNDING_FIELDS.issubset(item) for item in snapshots)
    assert second["confirmed_bindings"] == {"object": "pen_2"}
    assert second["invalidated_bindings"]["object"]["value"] == "pen_1"


def test_grounding_refresh_projects_two_valid_candidates_without_selecting() -> None:
    authority, atomic, occurrence, ctx, snapshots = _grounding_fixture(
        ("pen_1", "pen_2"),
    )

    state = authority.refresh(occurrence, atomic, [], ctx)

    assert len(snapshots) == 1
    assert _GROUNDING_FIELDS.issubset(snapshots[0])
    assert state["confirmed_bindings"] == {}
    assert state["candidate_bindings"] == {"object": ["pen_1", "pen_2"]}
    assert state["missing_bindings"] == ["object"]
    assert "multiple_valid_object_candidates" in state["blocking_reasons"]
    current = ctx.binding_store.snapshot_for_node(occurrence)["object"]
    assert current.source is BindingSource.TASK
    assert current.resolution is BindingResolution.SEMANTIC


def test_grounding_does_not_confirm_below_required_resolution_without_invocation() -> None:
    authority, atomic, occurrence, ctx, _snapshots = _grounding_fixture(
        ("pen_1",),
    )
    atomic.inputs[0] = ParameterSpec(
        "object",
        "entity",
        runtime_resolvable=True,
        required_resolution="relation_verified",
    )

    state = authority.refresh(occurrence, atomic, [], ctx)

    assert state["confirmed_bindings"] == {}
    assert state["missing_bindings"] == ["object"]
    current = ctx.binding_store.snapshot_for_node(occurrence)["object"]
    assert current.source is BindingSource.TASK
    assert current.resolution is BindingResolution.SEMANTIC


def test_learned_invocation_ready_includes_current_preconditions() -> None:
    authority, atomic, occurrence, ctx, _snapshots = _grounding_fixture(())
    atomic.preconditions = [_effect("agent.holds")]
    ctx.binding_store.commit_grounded(
        occurrence.occurrence_id,
        {
            "object": RuntimeBinding(
                role="object",
                value="pen_1",
                semantic_type="entity",
                source=BindingSource.HARNESS_EVIDENCE,
                status=BindingStatus.GROUNDED,
                resolution=BindingResolution.CONCRETE,
                evidence_refs=["entity:0:object:pen_1"],
                world_revision=0,
            ),
        },
    )
    invocation = SimpleNamespace(
        spec=SimpleNamespace(grounding_constraints=[]),
    )

    state = authority.refresh(occurrence, atomic, [invocation], ctx)

    assert state["confirmed_bindings"] == {"object": "pen_1"}
    assert state["precondition_status"] == [{
        "predicate": "agent.holds",
        "status": "missing",
    }]
    assert state["learned_invocation_ready"] is False
    assert "preconditions_not_satisfied" in state["blocking_reasons"]


class _LongHistoryHarness:
    """Generic revisioned actions for exercising history policy boundaries."""

    profile_name = "r3_runtime_state_test"
    semantic_value_compatible = staticmethod(semantic_value_compatible)

    def __init__(self) -> None:
        self._revision = 0
        self._validator = AlfWorldValidatorChannel()

    @staticmethod
    def _arguments(revision: int) -> tuple[str, dict[str, str]]:
        phase = revision % 3
        if phase == 0:
            return "GO_TO", {"destination": f"location_{revision}"}
        if phase == 1:
            return "OPEN", {"object": f"container_{revision}"}
        return "EXAMINE", {"object": f"item_{revision}"}

    def _catalog(self) -> list[HarnessActionSpec]:
        action_type, arguments = self._arguments(self._revision)
        return [_action(
            f"r{self._revision:03d}_action",
            self._revision,
            action_type,
            arguments,
        )]

    def reset(self, _task: HarnessTask) -> HarnessActionResult:
        self._revision = 0
        self._validator.reset()
        return HarnessActionResult(
            accepted=True,
            observation="fresh generic world",
            done=False,
            won=False,
            new_revision=0,
            catalog=self._catalog(),
            metadata={"reset": True},
        )

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        spec = self._catalog()[0]
        if action_id != spec.action_id or revision != self._revision:
            raise KeyError("stale generic action")
        self._revision += 1
        self._validator.record(
            spec,
            accepted=True,
            revision=self._revision,
            done=False,
            won=False,
        )
        metadata: dict[str, Any] = {"action_type": spec.action_type}
        if spec.revision == 0:
            metadata["negative_observations"] = {
                "initial_check": "no additional current candidate",
            }
        return HarnessActionResult(
            accepted=True,
            observation=f"accepted {spec.action_id}",
            done=False,
            won=False,
            new_revision=self._revision,
            catalog=self._catalog(),
            metadata=metadata,
        )

    def validator_channel(self) -> AlfWorldValidatorChannel:
        return self._validator


def _trace_builder(task: HarnessTask) -> TraceBuilder:
    task_record = TaskRecord(
        task_id=task.task_id,
        benchmark=task.benchmark,
        goal=task.goal,
        task_type=task.task_type,
        task_signature=str(task.metadata["task_signature"]),
    )
    return TraceBuilder(TraceRecord.create(
        task_record,
        task_contract={},
        planner_audit={},
        runtime_plan={},
    ))


def _policy_json(prompt: str) -> dict[str, Any]:
    marker = "\n\nPOLICY_CONTEXT_JSON\n"
    return json.loads(prompt.split(marker, 1)[1])


def test_long_history_keeps_full_trace_but_projects_memory_and_five_recent_actions() -> None:
    task = HarnessTask(
        task_id="long_history",
        goal="exercise generic state transitions",
        benchmark="generic",
        context={},
        metadata={"task_signature": "generic:long-history"},
    )
    contract = TaskContract()
    plan = RuntimeLinearPlan.full_dynamic(
        task.task_id,
        contract,
        reason="deterministic history fixture",
    )
    harness = _LongHistoryHarness()
    trace_builder = _trace_builder(task)
    ctx = TaskRuntimeContext.create(
        task,
        plan,
        harness,
        trace_builder,
        RuntimeBudget(global_action_budget=30, node_action_budget=25),
    )
    executor = NodeExecutor(
        SimpleNamespace(),
        ValidationEngine(),
        lambda *_args: None,
    )
    session = SimpleNamespace(session_id="long_history_session")
    loop_guard = ActionLoopGuard()
    last_payload: dict[str, Any] = {}

    for index in range(22):
        spec = ctx.action_catalog[0]
        last_payload, _ = executor._execute_environment_call(
            SimpleNamespace(
                call_id=f"call_{index}",
                name="environment_action",
                arguments={"action_id": spec.action_id},
            ),
            session,
            None,
            ctx,
            span_id="span_long_history",
            origin="runtime_dynamic",
            loop_guard=loop_guard,
        )

    assert len(ctx.action_history) == 22
    assert len(trace_builder.trace.environment_actions) == 22
    assert [
        item.new_revision for item in trace_builder.trace.environment_actions
    ] == list(range(1, 23))
    assert len(trace_builder.trace.native_tool_calls) == 22

    memory = ctx.exploration_memory.policy_view()
    assert "location_0" in memory["visited"]
    assert "container_1" in memory["opened"]
    assert "item_2" in memory["inspected"]
    assert memory["negative_observations"] == {
        "initial_check": "no additional current candidate",
    }
    assert memory["discovered"]["location_0"]["evidence_status"] == "historical"
    assert memory["discovered"]["container_22"]["evidence_status"] == "observed"
    assert memory["progress_since_last_grounding_change"] == {
        "accepted_actions": 22,
    }

    recent = ctx.relevant_history("", limit=5)
    assert len(recent) == 5
    assert [item["arguments"] for item in recent] == [
        item.arguments for item in trace_builder.trace.environment_actions[-5:]
    ]
    assert last_payload["recent_accepted_actions"] == recent
    assert len(last_payload["recent_accepted_actions"]) == 5

    prompt = ContextBuilder().dynamic_task(
        task_goal=ctx.task_goal,
        observation=ctx.observation,
        action_catalog=ctx.action_catalog,
        # Supply full history: ContextBuilder owns the final five-action
        # projection and must not rely on callers having pre-truncated it.
        relevant_action_history=ctx.action_history,
        remaining_budget=ctx.budget.snapshot(),
        task_progress=ctx.task_progress.policy_view(),
        exploration_memory=memory,
    )
    policy = _policy_json(prompt)
    assert len(policy["recent_accepted_actions"]) == 5
    assert [item["arguments"] for item in policy["recent_accepted_actions"]] == [
        item.arguments for item in trace_builder.trace.environment_actions[-5:]
    ]
    assert policy["current_action_catalog"] == {
        "revision": 22,
        "actions": [{
            "action_id": "r022_action",
            "action_type": "OPEN",
            "arguments": {"object": "container_22"},
        }],
    }
    assert policy["exploration_memory"]["discovered"]["location_0"][
        "evidence_status"
    ] == "historical"
    assert "location_0" not in json.dumps(
        policy["current_action_catalog"], sort_keys=True,
    )
    # Policy projection must not destructively compact canonical history.
    assert len(ctx.action_history) == 22
    assert len(trace_builder.trace.environment_actions) == 22
