"""One deterministic no-API smoke spanning Runtime, Evolution, and Governance.

This intentionally follows design document section 40.2 as four sequential
episodes in one bank.  It is not four isolated helper tests: later episodes
must consume knowledge produced by earlier episodes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from experiments.fakes import (
    FakeAgentFactory,
    FakeHarness,
    FakeProviderSet,
    FakeReply,
    ScriptedAgentProvider,
    fake_task,
    knowledge_digest,
    planner_gap_replies,
)
from atomic_skillgraph.agents import (
    AgentProvider,
    NativeToolSpec,
    ReplayAgentSession,
    UsageBucket,
    UsageLedger,
)
from atomic_skillgraph.core.results import NodeExecutionStatus
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import ExtractorSession
from atomic_skillgraph.evolution.maintenance import ExtractionPolicy
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceEventType, EvidenceLedger
from atomic_skillgraph.governance.projections import LifecycleProjection
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.graph_store import GraphStore
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.planner.pipeline import PlannerPipeline
from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.traces.store import TraceStore
from atomic_skillgraph.validation.engine import ValidationEngine


def test_scripted_provider_matches_replay_session_protocol() -> None:
    providers = FakeProviderSet()
    assert isinstance(providers, Mapping)
    assert set(providers) == {
        "planner",
        "runtime_preparation",
        "runtime_seeded",
        "runtime_dynamic",
        "extractor",
    }
    assert isinstance(providers["planner"], ScriptedAgentProvider)
    assert isinstance(providers["planner"], AgentProvider)

    structured_schema = {
        "type": "object",
        "required": ["answer"],
        "additionalProperties": False,
        "properties": {"answer": {"type": "string"}},
    }
    providers.enqueue(
        "planner",
        [FakeReply.structured(lambda request: {"answer": request.policy_context["seed"]})],
    )
    usage = UsageLedger()
    structured_session = ReplayAgentSession(
        providers["planner"],
        system_prompt="provider contract test",
        usage_ledger=usage,
        usage_bucket=UsageBucket.PLANNER_P1,
        session_id="provider_structured",
    )
    structured_turn = structured_session.next_turn(
        "Return the seed.\n\nPOLICY_CONTEXT_JSON\n{\"seed\":\"deterministic\"}",
        structured_output_schema=structured_schema,
    )
    assert json.loads(structured_turn.content) == {"answer": "deterministic"}
    structured_request = providers["planner"].requests[0]
    assert [item["role"] for item in structured_request.messages] == ["system", "user"]
    assert structured_request.tools == ()
    assert structured_request.structured_output_schema == structured_schema

    learned_tool = NativeToolSpec(
        "invoke_impl_fixture",
        "Invoke one learned implementation.",
        {
            "type": "object",
            "required": ["item"],
            "additionalProperties": False,
            "properties": {"item": {"type": "string"}},
        },
    )
    status_tool = NativeToolSpec(
        "report_runtime_status",
        "Report a terminal runtime status.",
        {
            "type": "object",
            "required": ["status"],
            "additionalProperties": False,
            "properties": {"status": {"enum": ["cannot_resolve"]}},
        },
    )
    offered_tools = [learned_tool, status_tool]
    providers.enqueue(
        "runtime_preparation",
        [
            FakeReply.tool("$learned", {"item": "ghost_9"}),
            FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
        ],
    )
    tool_session = ReplayAgentSession(
        providers["runtime_preparation"],
        system_prompt="provider native tool test",
        usage_ledger=usage,
        usage_bucket=UsageBucket.RUNTIME_PREPARATION,
        session_id="provider_tools",
    )
    learned_turn = tool_session.next_turn("Prepare one node.", tools=offered_tools)
    learned_call = learned_turn.tool_calls[0]
    assert learned_call.name == "invoke_impl_fixture"
    status_turn = tool_session.submit_tool_result(
        learned_call.call_id,
        {"error": "runtime_binding_not_concrete", "repairable": True},
        tools=offered_tools,
    )
    status_call = status_turn.tool_calls[0]
    assert status_call.name == "report_runtime_status"
    tool_session.finalize_tool_result(status_call.call_id, {"accepted": True})

    second_request = providers["runtime_preparation"].requests[1]
    assert [item["role"] for item in second_request.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_request.messages[2]["tool_calls"][0]["id"] == learned_call.call_id
    assert json.loads(second_request.messages[3]["content"]) == {
        "error": "runtime_binding_not_concrete",
        "repairable": True,
    }
    assert [item.name for item in second_request.tools] == [
        "invoke_impl_fixture",
        "report_runtime_status",
    ]
    assert second_request.structured_output_schema is None
    assert tool_session.snapshot()["finalized"] is True
    assert providers.snapshot()["runtime_preparation"]["call_count"] == 2
    assert len(usage.events) == 3
    assert usage.total().total_tokens == 30
    providers.assert_exhausted()


def _e1_take(item: str, *, start: int = 0) -> dict:
    return {
        "phase_id": f"take_{start}",
        "intent": "take target item",
        "event_start": start,
        "event_end": start,
        "input_roles": {"item": item},
        "output_roles": {"held_object": item},
        "preconditions": [],
        "effects": [{"predicate": "agent.holds", "args": {"object": item}}],
        "rationale": "The accepted TAKE transition establishes possession.",
    }


def _e1_examine(item: str, *, start: int) -> dict:
    return {
        "phase_id": f"examine_{start}",
        "intent": "observe held item",
        "event_start": start,
        "event_end": start,
        "input_roles": {"item": item},
        "output_roles": {"observed_object": item},
        "preconditions": [],
        "effects": [{"predicate": "object.observed", "args": {"object": item}}],
        "rationale": "The accepted EXAMINE transition establishes observation.",
    }


def _extract_and_register(
    *,
    trace,
    task,
    task_contract,
    e1_occurrences: list[dict],
    factory: FakeAgentFactory,
    skills: SkillRegistry,
    tools: ToolRegistry,
    validation: ValidationEngine,
    harness: FakeHarness,
    ledger: EvidenceLedger,
    projection: LifecycleProjection,
    data_edge: bool = False,
) -> dict:
    normalized = TraceNormalizer().build(trace)
    session = factory.new_session(
        "extractor",
        [FakeReply.structured({"occurrences": e1_occurrences})],
    )
    extractor = ExtractorSession(session)
    proposals = extractor.propose_atomics(normalized)
    canonical = Atomicizer().validate_and_canonicalize(proposals, normalized)

    new_edges: list[dict] = []
    if data_edge:
        new_edges.append(
            {
                "edge_id": f"edge_{trace.trace_id}_held_to_observe",
                "edge_type": "data_flow",
                "source_step": canonical[0].occurrence_id,
                "target_step": canonical[1].occurrence_id,
                "source_role": "held_object",
                "target_role": "item",
            }
        )
    session.enqueue(
        FakeReply.structured(
            {
                "control_sequence": [item.occurrence_id for item in canonical],
                "existing_edges": [],
                "new_edges": new_edges,
                "summary": (
                    "take then observe target item"
                    if len(canonical) > 1
                    else "take target item"
                ),
                "guideline": {"canonical": True},
                "insight": {"source": "deterministic_fullchain"},
            }
        )
    )
    e2 = extractor.propose_composite(canonical, [])
    assert session.remaining_replies == 0

    compiled = ToolCompiler().compile(canonical)
    admission = Admission(validation.tool)
    aligner = Aligner(skills, tools)
    atomic_refs = []
    implementation_refs = []
    tool_refs = []
    occurrence_to_atomic = {}
    for item in compiled:
        atomic_ref = aligner.align_atomic(item.atomic)
        admitted_tool = admission.admit_tool(
            item.tool,
            replay=lambda candidate, case: harness.replay_tool(task, candidate, case),
        )
        assert admitted_tool.status is ToolStatus.CANDIDATE
        tool_ref = aligner.align_tool(admitted_tool)
        admitted_implementation = admission.admit_implementation(
            item.implementation,
            admitted_tool,
            atomic=item.atomic,
            harness=harness,
        )
        assert admitted_implementation.status is SkillStatus.CANDIDATE
        implementation_ref = aligner.align_implementation(
            admitted_implementation,
            atomic_ref,
            tool_ref,
        )
        atomic_refs.append(atomic_ref)
        implementation_refs.append(implementation_ref)
        tool_refs.append(tool_ref)
        occurrence_to_atomic[item.occurrence.occurrence_id] = atomic_ref

    candidate_composite = CompositeBuilder().validate_and_build(
        e2,
        canonical,
        task_contract,
    )
    composite_ref = aligner.align_composite(candidate_composite, occurrence_to_atomic)
    evolution_events = CreditAssigner().assign_evolution(
        trace,
        atomic_refs,
        implementation_refs,
        tool_refs,
        composite_ref,
    )
    append = ledger.append_transaction(evolution_events)
    assert append.inserted_count == len(evolution_events)
    projection.consume_new_events()
    return {
        "canonical": canonical,
        "atomic_refs": atomic_refs,
        "implementation_refs": implementation_refs,
        "tool_refs": tool_refs,
        "composite_ref": composite_ref,
        "new_edges": new_edges,
    }

def _persist_and_credit(
    trace,
    trace_store: TraceStore,
    assigner: CreditAssigner,
    ledger: EvidenceLedger,
    projection: LifecycleProjection,
):
    path = trace_store.save_atomic(trace)
    assert path.is_file()
    events = assigner.assign(trace)
    result = ledger.append_transaction(events)
    assert result.inserted_count == len(events)
    projection.consume_new_events()
    return events


def test_deterministic_no_api_fullchain_four_episode_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "bank"
    database = StateDatabase(data_dir / "state.sqlite")
    artifact_store = ArtifactStore(data_dir, database)
    skills = SkillRegistry(artifact_store, database)
    tools = ToolRegistry(artifact_store, database)
    graph = GraphStore(database, skills)
    ledger = EvidenceLedger(database)
    projection = LifecycleProjection(database, ledger)
    trace_store = TraceStore(data_dir)
    validation = ValidationEngine()
    harness = FakeHarness()
    factory = FakeAgentFactory()
    planner = PlannerPipeline(skills, graph, factory)
    invocation_compiler = InvocationCompiler(
        skills,
        tools,
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
    assigner = CreditAssigner()

    # Episode 1: empty-bank Full Dynamic succeeds, then the real E1/E2 pipeline
    # admits all four kinds as online-usable Candidates.
    factory.enqueue("planner", planner_gap_replies())
    factory.enqueue(
        "runtime_dynamic",
        [FakeReply.tool("environment_action", {"action_id": "a001"})],
    )
    task1 = fake_task("episode-1", "apple_1")
    episode1 = runtime.run_task(task1)
    assert episode1.runtime_plan["source"] == "full_dynamic"
    assert episode1.benchmark_success is True
    assert episode1.learning_eligible is True
    assert len(episode1.environment_actions) == 1
    assert episode1.environment_actions[0].accepted is True
    assert episode1.environment_actions[0].action_type == "TAKE"
    assert ExtractionPolicy().should_extract(episode1) is True
    trace_store.save_atomic(episode1)

    learned = _extract_and_register(
        trace=episode1,
        task=task1,
        task_contract=harness.task_contract(task1),
        e1_occurrences=[_e1_take("apple_1")],
        factory=factory,
        skills=skills,
        tools=tools,
        validation=validation,
        harness=harness,
        ledger=ledger,
        projection=projection,
    )
    atomic_ref = learned["atomic_refs"][0]
    implementation_ref = learned["implementation_refs"][0]
    tool_ref = learned["tool_refs"][0]
    composite_ref = learned["composite_ref"]
    assert skills.get_atomic(atomic_ref).status is SkillStatus.CANDIDATE
    assert skills.get_implementation(implementation_ref).status is SkillStatus.CANDIDATE
    assert tools.get(tool_ref).status is ToolStatus.CANDIDATE
    assert skills.get_composite(composite_ref).status is SkillStatus.CANDIDATE

    # Episode 2: a new concrete instance uses that Candidate through P0 and an
    # autonomous Direct.  No Planner or Runtime Agent turn is allowed here.
    usage_before_direct = factory.usage_ledger.total().total_tokens
    episode2 = runtime.run_task(fake_task("episode-2", "apple_2"))
    usage_after_direct = factory.usage_ledger.total().total_tokens
    assert episode2.runtime_plan["source"] == "stored_composite"
    assert episode2.runtime_plan["source_composite_ref"] == str(composite_ref)
    assert episode2.node_records[0].status is NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS
    assert episode2.node_records[0].validated_outputs == {"held_object": "apple_2"}
    assert episode2.graph_self_sufficient_success is True
    assert episode2.implementation_direct_success is True
    assert episode2.agent_turns == []
    assert usage_after_direct == usage_before_direct
    direct_events = _persist_and_credit(
        episode2,
        trace_store,
        assigner,
        ledger,
        projection,
    )
    assert any(
        item.artifact_ref == str(tool_ref)
        and item.event is EvidenceEventType.DIRECT_SUCCESS
        for item in direct_events
    )

    # Exactly-once is exercised on the same actual Trace, including its stable
    # attempt ids—not on a separately constructed ledger helper event.
    count_before_replay = ledger.count()
    replay = ledger.append_transaction(assigner.assign(episode2))
    assert replay.inserted_count == 0
    assert replay.duplicate_count == len(direct_events)
    assert ledger.count() == count_before_replay

    # Episode 3: the task withholds a task binding.  Preparation proposes a
    # schema-valid but ungrounded instance, explicitly stops, and a separately
    # created fresh Seeded session solves the Atomic through environment_action.
    factory.enqueue(
        "runtime_preparation",
        [
            FakeReply.tool("$learned", {"item": "ghost_9"}),
            FakeReply.tool("report_runtime_status", {"status": "cannot_resolve"}),
        ],
    )
    factory.enqueue(
        "runtime_seeded",
        [FakeReply.tool("environment_action", {"action_id": "a001"})],
    )
    tool_started_before = projection.stats(str(tool_ref), "tool").started_count
    implementation_failures_before = projection.stats(
        str(implementation_ref), "implementation"
    ).failure_count
    episode3 = runtime.run_task(
        fake_task("episode-3", "banana_1", expose_binding=False)
    )
    assert episode3.node_records[0].status is NodeExecutionStatus.SEEDED_SUCCESS
    assert episode3.node_records[0].direct_result["started"] is False
    assert episode3.node_records[0].seeded_result["started"] is False
    assert episode3.node_records[0].seeded_result["atomic_effect_passed"] is True
    assert episode3.tool_executions == []
    rejected_calls = [
        call
        for call in episode3.native_tool_calls
        if call.call_kind == "implementation_invocation"
        and call.preflight_result.get("passed") is False
    ]
    assert len(rejected_calls) == 1
    assert rejected_calls[0].preflight_result["failure_code"] == "runtime_binding_not_concrete"
    preparation = [item for item in episode3.agent_sessions if item.session_type == "RuntimePreparationSession"]
    seeded = [item for item in episode3.agent_sessions if item.session_type == "SeededSession"]
    assert len(preparation) == len(seeded) == 1
    assert preparation[0].session_id != seeded[0].session_id
    assert preparation[0].snapshot["session_kind"] == "runtime_preparation"
    assert seeded[0].snapshot["session_kind"] == "runtime_seeded"
    assert all("ToolAsset" not in str(message) for message in seeded[0].snapshot["messages"])

    episode3_events = _persist_and_credit(
        episode3,
        trace_store,
        assigner,
        ledger,
        projection,
    )
    assert any(
        item.artifact_ref == str(implementation_ref)
        and item.event is EvidenceEventType.PREFLIGHT_REJECTED
        and item.metadata["started"] is False
        for item in episode3_events
    )
    assert not any(item.artifact_kind == "tool" for item in episode3_events)
    implementation_stats = projection.stats(str(implementation_ref), "implementation")
    assert implementation_stats.preflight_rejected_count >= 1
    assert implementation_stats.failure_count == implementation_failures_before
    assert projection.stats(str(tool_ref), "tool").started_count == tool_started_before
    assert any(
        item.artifact_ref == str(atomic_ref)
        and item.event is EvidenceEventType.SEEDED_SUCCESS
        for item in episode3_events
    )

    # Episode 4: the learned graph's sole node succeeds, but benchmark win is
    # intentionally withheld until EXAMINE.  Rescue must therefore penalize
    # only Composite structure while leaving started node layers positive.
    factory.enqueue(
        "runtime_dynamic",
        [FakeReply.tool("environment_action", {"action_id": "a001"})],
    )
    task4 = fake_task("episode-4", "mug_1", requires_rescue=True)
    episode4 = runtime.run_task(task4)
    assert episode4.node_records[0].status is NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS
    assert episode4.node_contract_success is True
    assert episode4.graph_full_completion is True
    assert episode4.task_rescue_required is True
    assert episode4.benchmark_success is True
    assert episode4.graph_self_sufficient_success is False
    assert [item.action_type for item in episode4.environment_actions] == ["TAKE", "EXAMINE"]
    assert ExtractionPolicy().should_extract(episode4) is True
    episode4_events = _persist_and_credit(
        episode4,
        trace_store,
        assigner,
        ledger,
        projection,
    )
    assert any(
        item.artifact_ref == str(composite_ref)
        and item.event is EvidenceEventType.TASK_RESCUE_REQUIRED
        and item.failure_layer == "composite"
        for item in episode4_events
    )
    assert any(
        item.artifact_ref == str(atomic_ref)
        and item.event is EvidenceEventType.DIRECT_SUCCESS
        for item in episode4_events
    )
    assert any(
        item.artifact_ref == str(tool_ref)
        and item.event is EvidenceEventType.DIRECT_SUCCESS
        for item in episode4_events
    )
    composite_stats = projection.stats(str(composite_ref), "composite")
    assert composite_stats.task_rescue_count == 1
    assert composite_stats.failure_count == 1

    rescued = _extract_and_register(
        trace=episode4,
        task=task4,
        task_contract=harness.task_contract(task4),
        e1_occurrences=[_e1_take("mug_1", start=0), _e1_examine("mug_1", start=1)],
        factory=factory,
        skills=skills,
        tools=tools,
        validation=validation,
        harness=harness,
        ledger=ledger,
        projection=projection,
        data_edge=True,
    )
    assert rescued["atomic_refs"][0] == atomic_ref
    assert rescued["atomic_refs"][1] != atomic_ref
    assert skills.get_atomic(rescued["atomic_refs"][1]).status is SkillStatus.CANDIDATE
    rescue_composite = skills.get_composite(rescued["composite_ref"])
    assert rescue_composite.status is SkillStatus.CANDIDATE
    assert rescue_composite.control_sequence == [
        item.occurrence_id for item in rescued["canonical"]
    ]
    assert len(rescue_composite.data_edges) == 1
    assert rescue_composite.data_edges[0].source_role == "held_object"
    assert rescue_composite.data_edges[0].target_role == "item"

    # Every scripted provider call belongs to a real bucket and provider totals
    # reconcile without adding reasoning tokens a second time.
    usage = factory.usage_ledger.snapshot()
    expected_provider_total = 10 * len(factory.usage_ledger.events)
    reconciliation = factory.usage_ledger.reconcile(expected_provider_total)
    assert reconciliation["token_mismatch"] == 0
    assert reconciliation["unattributed_total_tokens"] == 0
    assert usage["episode_total"]["total_tokens"] == expected_provider_total
    assert usage["episode_total"]["total_tokens"] == sum(
        event.usage.total_tokens for event in factory.usage_ledger.events
    )
    assert usage["episode_total"]["reasoning_tokens"] == sum(
        int(event.usage.reasoning_tokens or 0) for event in factory.usage_ledger.events
    )
    assert all(
        event.usage.total_tokens
        == event.usage.prompt_tokens + event.usage.completion_tokens
        for event in factory.usage_ledger.events
    )

    # Frozen evaluation may emit a task-local Trace but cannot alter artifacts,
    # Ledger facts, graph state, lifecycle projection, or their digest.
    digest_view = SimpleNamespace(database=database, artifacts=artifact_store)
    digest_before_frozen = AtomicSkillGraphSystem.knowledge_digest(digest_view)
    assert knowledge_digest(database) == digest_before_frozen
    ledger_count_before_frozen = ledger.count()
    factory.enqueue("planner", planner_gap_replies())
    factory.enqueue(
        "runtime_dynamic",
        [FakeReply.tool("environment_action", {"action_id": "a001"})],
    )
    frozen_trace = runtime.run_task(
        fake_task("frozen-check", "plate_1"),
        mode=RuntimeMode.FROZEN,
    )
    trace_store.save_atomic(frozen_trace)
    assert frozen_trace.runtime_plan["source"] == "full_dynamic"
    assert frozen_trace.benchmark_success is True
    assert ledger.count() == ledger_count_before_frozen
    assert AtomicSkillGraphSystem.knowledge_digest(digest_view) == digest_before_frozen
    assert knowledge_digest(database) == digest_before_frozen

    for trace in (episode1, episode2, episode3, episode4, frozen_trace):
        assert trace.ended_at >= trace.started_at
        assert trace_store.exists(trace.trace_id)
        payload = trace_store.load_payload(trace.trace_id)
        assert payload["trace_id"] == trace.trace_id
        assert payload["runtime_plan"]["task_id"] == trace.task.task_id
        assert all(item.ended_at >= item.started_at for item in trace.agent_sessions)

    artifact_store.verify_all()
    factory.assert_exhausted()
    database.close()
