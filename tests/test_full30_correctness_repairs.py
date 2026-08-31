from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.core.contracts import (
    ContractSource,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.errors import BudgetExhausted, FailureLayer
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.evidence import EvolutionEvidenceAccumulator
from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
from atomic_skillgraph.evolution.failure_processor import FailureProcessor
from atomic_skillgraph.evolution.portability import (
    episode_specific_terms,
    relevant_known_atomic_contracts,
    resolve_capability_label,
    resolve_capability_label_group,
    source_forbidden_terms,
    validate_portability,
)
from atomic_skillgraph.evolution.tool_compiler import (
    CompiledKnowledge,
    ToolCompiler,
    rewrite_capability_labels,
)
from atomic_skillgraph.governance.credit import (
    CreditAssigner,
    CreditAttempt,
    CreditTrace,
)
from atomic_skillgraph.governance.ledger import EvidenceEventType
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.orchestrator import apply_terminal_outcome
from atomic_skillgraph.traces.schema import TaskRecord, TraceRecord
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from atomic_skillgraph.validation.failure_localizer import FailureLocalizer
from atomic_skillgraph.core.results import ValidationResult
from experiments.report import summarize_traces, trace_to_row


def _take_occurrence(
    object_name: str,
    source_name: str,
    intent: str,
):
    action = {
        "event_index": 0,
        "action_id": "a0",
        "action_type": "TAKE",
        "arguments": {"object": object_name, "source": source_name},
        "accepted": True,
        "before_revision": 0,
        "after_revision": 1,
        "done": True,
        "won": True,
        "span_id": "span",
    }
    normalized = {
        "trace_id": f"trace_{object_name}_{source_name}",
        "source_task": {
            "task_id": f"task_{object_name}_{source_name}",
            "task_signature": "signature",
            "benchmark": "alfworld",
            "task_type": "pick_and_place_simple",
            "metadata": {},
        },
        "actions": [action],
        "runtime_spans": [{
            "span_id": "span",
            "kind": "full_dynamic",
            "occurrence_id": "",
            "action_start": 0,
            "action_end": 1,
            "parent_span_id": None,
            "learnable": True,
        }],
        "validations": [],
        "task_contract": to_primitive(TaskContract(
            target_effects=[SemanticPredicate(
                "agent.holds", {"object": object_name},
            )],
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="test",
        )),
        "benchmark_success": True,
    }
    proposal = AtomicOccurrenceProposal(
        "acquire",
        intent,
        0,
        0,
        {"object": object_name, "source": source_name},
        {"held_object": object_name},
        [],
        [SemanticPredicate("agent.holds", {"object": object_name})],
        "verified transition",
    )
    return Atomicizer().validate_and_canonicalize(
        [proposal], normalized,
    )[0]


def _stage_with_label(
    occurrence,
    skills: SkillRegistry,
    tools: ToolRegistry,
):
    aligner = Aligner(skills, tools)
    compiled = ToolCompiler().compile([occurrence])[0]
    bundle = aligner.stage_atomic(
        compiled.atomic,
        compiled.tool,
        compiled.implementation,
    )
    assert bundle.tool is not None and bundle.implementation is not None
    staged_occurrence = aligner.atomic_canonicalizer.rewrite_canonical_occurrence(
        occurrence,
        bundle,
        atomic_ref=bundle.atomic.ref,
    )
    staged = CompiledKnowledge(
        staged_occurrence,
        bundle.atomic,
        bundle.tool,
        bundle.implementation,
    )
    try:
        existing = skills.get_atomic(bundle.atomic.ref)
    except KeyError:
        existing = None
    label = resolve_capability_label(
        staged_occurrence,
        bundle.atomic,
        existing_atomic=existing,
    )
    return aligner, rewrite_capability_labels(staged, label), label


def test_portability_rejects_episode_terms_but_keeps_contract_transition() -> None:
    terms = episode_specific_terms(
        {"object": "cellphone_1", "source": "bed_1"},
        {"held_object": "cellphone_1"},
        ["object", "source", "held_object"],
        ["entity"],
    )
    assert {"cellphone_1", "cellphone", "bed_1", "bed"} <= terms
    assert not validate_portability(
        "take_the_cellphone_from_the_bed",
        episode_terms=terms,
        require_intent=True,
    ).passed
    assert not validate_portability(
        "take_cellphone_1",
        episode_terms=terms,
        require_intent=True,
    ).passed
    assert validate_portability(
        "acquire_target_object",
        episode_terms=terms,
        require_intent=True,
    ).passed
    assert validate_portability(
        "establish_agent_holds",
        episode_terms=terms,
        require_intent=True,
    ).passed

    occurrence = _take_occurrence(
        "cellphone_1",
        "bed_1",
        "take_the_cellphone_from_the_bed",
    )
    compiled = ToolCompiler().compile([occurrence])[0]
    label = resolve_capability_label(occurrence, compiled.atomic)
    assert label.canonical_intent == "establish_agent_holds"
    assert label.source == "contract_fallback"


def test_concrete_role_name_cannot_self_authorize_entity_family() -> None:
    terms = episode_specific_terms(
        {"cellphone": "cellphone_1"},
        {"cellphone": "cellphone_1"},
        ["cellphone"],
        ["string"],
    )
    assert "cellphone" in terms
    assert not validate_portability(
        "take_cellphone",
        episode_terms=terms,
        require_intent=True,
    ).passed
    assert not validate_portability(
        "take_cellphone1",
        episode_terms=terms,
        require_intent=True,
    ).passed
    assert not validate_portability(
        "acquire_and_place",
        require_intent=True,
    ).passed


def test_source_task_text_is_not_a_portable_label_authority() -> None:
    occurrence = _take_occurrence(
        "cellphone_1", "bed_1", "acquire_target_object",
    )
    occurrence.source_task["goal"] = "source episode narration"
    occurrence.source_task["metadata"] = {
        "observation": "literal source observation",
    }
    forbidden = source_forbidden_terms(occurrence)
    assert "source episode narration" in forbidden
    assert "literal source observation" in forbidden
    assert not validate_portability(
        "literal_source_observation",
        additional_forbidden_terms=forbidden,
        require_intent=True,
    ).passed
    occurrence.source_task["task_type"] = (
        "pick_cool_then_place_in_recep"
    )
    forbidden = source_forbidden_terms(occurrence)
    assert "pick_cool" in forbidden
    assert not validate_portability(
        "pick_cool",
        additional_forbidden_terms=forbidden,
        require_intent=True,
    ).passed


def test_aligned_batch_prefers_portable_intent_independent_of_order() -> None:
    concrete = _take_occurrence(
        "cellphone_1", "bed_1", "take_the_cellphone_from_the_bed",
    )
    portable = _take_occurrence(
        "ladle_2", "cabinet_3", "acquire_target_object",
    )
    concrete_atomic = ToolCompiler().compile([concrete])[0].atomic
    portable_atomic = ToolCompiler().compile([portable])[0].atomic
    forward = resolve_capability_label_group([
        (concrete, concrete_atomic),
        (portable, portable_atomic),
    ])
    reverse = resolve_capability_label_group([
        (portable, portable_atomic),
        (concrete, concrete_atomic),
    ])
    assert forward == reverse
    assert forward.canonical_intent == "acquire_target_object"
    assert forward.source == "llm_portable"


def test_aligned_batch_rejects_other_occurrence_entity_family() -> None:
    cellphone = _take_occurrence(
        "cellphone_1", "bed_1", "acquire_ladle",
    )
    ladle = _take_occurrence(
        "ladle_2", "cabinet_3", "take_ladle_2",
    )
    cellphone_atomic = ToolCompiler().compile([cellphone])[0].atomic
    ladle_atomic = ToolCompiler().compile([ladle])[0].atomic
    forward = resolve_capability_label_group([
        (cellphone, cellphone_atomic),
        (ladle, ladle_atomic),
    ])
    reverse = resolve_capability_label_group([
        (ladle, ladle_atomic),
        (cellphone, cellphone_atomic),
    ])
    assert forward == reverse
    assert forward.canonical_intent == "establish_agent_holds"
    assert forward.source == "contract_fallback"


def test_equivalent_contract_reuses_ref_and_label_across_entities(tmp_path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    store = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(store, database)
    tools = ToolRegistry(store, database)

    first = _take_occurrence(
        "cellphone_1", "bed_1", "take_the_cellphone_from_the_bed",
    )
    aligner, staged_first, first_label = _stage_with_label(
        first, skills, tools,
    )
    first_atomic_ref = aligner.align_atomic(staged_first.atomic)
    first_tool_ref = aligner.align_tool(staged_first.tool)
    first_impl_ref = aligner.align_implementation(
        staged_first.implementation,
        first_atomic_ref,
        first_tool_ref,
    )

    second = _take_occurrence(
        "ladle_2", "cabinet_3", "take_ladle_2_from_cabinet_3",
    )
    aligner, staged_second, second_label = _stage_with_label(
        second, skills, tools,
    )
    second_atomic_ref = aligner.align_atomic(staged_second.atomic)
    second_tool_ref = aligner.align_tool(staged_second.tool)
    second_impl_ref = aligner.align_implementation(
        staged_second.implementation,
        second_atomic_ref,
        second_tool_ref,
    )

    assert first_atomic_ref == second_atomic_ref
    assert first_tool_ref == second_tool_ref
    assert first_impl_ref == second_impl_ref
    assert first_label.canonical_intent == second_label.canonical_intent
    assert second_label.source == "existing_contract"
    assert first_atomic_ref.logical_id.startswith("atomic_")
    assert first_tool_ref.tool_id.startswith("tool_")
    assert first_impl_ref.logical_id.startswith("impl_")

    persisted_atomic = skills.get_atomic(first_atomic_ref)
    persisted_tool = tools.get(first_tool_ref)
    persisted_impl = skills.get_implementation(first_impl_ref)
    semantic_text = json.dumps({
        "atomic": persisted_atomic.summary,
        "tool": [persisted_tool.summary, persisted_tool.metadata],
        "implementation": persisted_impl.metadata,
    }).casefold()
    for concrete in ("cellphone", "bed", "ladle", "cabinet"):
        assert concrete not in semantic_text
    assert persisted_impl.metadata["canonical_intent"] == (
        persisted_atomic.metadata["canonical_intent"]
    )

    known = relevant_known_atomic_contracts({
        "actions": [{
            "authoritative_positive_effects": [{
                "predicate": "agent.holds",
            }],
            "authoritative_terminal_effect_certificates": [],
        }],
    }, skills)
    assert len(known) == 1
    assert known[0].canonical_intent == "establish_agent_holds"


def test_e1_prompt_is_general_and_carries_compact_known_contracts() -> None:
    known = [{
        "atomic_ref": "atomic://stable@1.0.0",
        "canonical_intent": "establish_agent_holds",
        "inputs": [],
        "outputs": [],
        "preconditions": [],
        "effects": [{"predicate": "agent.holds", "args": {}}],
    }]
    prompt = ContextBuilder().extractor_e1(
        canonical_trace={"actions": []},
        known_atomic_contracts=known,
    )
    instruction, payload_text = prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)
    for forbidden in (
        "LOOK",
        "INVENTORY",
        "lamp",
        "microwave",
        "fridge",
        "sink",
        "pick_and_place",
        "pick_two",
        "ALFWorld",
    ):
        assert forbidden.casefold() not in instruction.casefold()
    assert json.loads(payload_text)["known_atomic_contracts"] == known


def test_composite_unsafe_text_falls_back_without_rejecting_graph() -> None:
    occurrence = _take_occurrence(
        "cellphone_1", "bed_1", "take_the_cellphone_from_the_bed",
    )
    compiled = ToolCompiler().compile([occurrence])[0]
    label = resolve_capability_label(occurrence, compiled.atomic)
    rewritten = rewrite_capability_labels(compiled, label)
    proposal = CompositeExtractionProposal(
        [rewritten.occurrence.occurrence_id],
        [],
        [],
        "canonical control sequence for pick cool cellphone_1",
        {"source_location": "bed_1"},
        {},
    )
    contract = TaskContract(target_effects=[SemanticPredicate(
        "agent.holds", {"object": "cellphone_1"},
    )])
    composite = CompositeBuilder().validate_and_build(
        proposal,
        [rewritten.occurrence],
        contract,
        contract_matcher=ExactContractMatcher(),
        task_bindings={"object": "cellphone_1", "source": "bed_1"},
    )
    assert composite.summary == "compose_establish_agent_holds"
    assert composite.metadata["summary_portability_fallback"] is True
    assert composite.metadata["guideline_portability_fallback"] is True
    assert composite.metadata[
        "artifact_label_concrete_term_violation_count"
    ] == 0
    assert composite.ref.logical_id.startswith("composite_")


def test_composite_identity_uses_contract_shape_across_entities(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    store = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(store, database)
    tools = ToolRegistry(store, database)
    refs = []
    for object_name, source_name in (
        ("cellphone_1", "bed_1"),
        ("ladle_2", "cabinet_3"),
    ):
        occurrence = _take_occurrence(
            object_name,
            source_name,
            "acquire_target_object",
        )
        aligner, staged, _label = _stage_with_label(
            occurrence, skills, tools,
        )
        atomic_ref = aligner.align_atomic(staged.atomic)
        proposal = CompositeExtractionProposal(
            [staged.occurrence.occurrence_id],
            [],
            [],
            "compose acquire target object",
            {"ordered_capabilities": ["acquire_target_object"]},
            {},
        )
        contract = TaskContract(
            target_effects=[SemanticPredicate(
                "agent.holds", {"object": object_name},
            )],
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="test",
        )
        composite = CompositeBuilder().validate_and_build(
            proposal,
            [staged.occurrence],
            contract,
            contract_matcher=ExactContractMatcher(),
            task_bindings={
                "object": object_name,
                "source": source_name,
            },
        )
        refs.append(aligner.align_composite(
            composite,
            {staged.occurrence.occurrence_id: atomic_ref},
        ))
    assert refs[0] == refs[1]


def test_composite_digest_fallback_includes_binding_structure() -> None:
    intent = "perform_" + ("x" * 130)
    summaries = []
    refs = []
    for object_name, source_name, task_role in (
        ("cellphone_1", "bed_1", "object"),
        ("ladle_2", "cabinet_3", "target"),
    ):
        occurrence = _take_occurrence(
            object_name, source_name, intent,
        )
        proposal = CompositeExtractionProposal(
            [occurrence.occurrence_id],
            [],
            [],
            f"source composite for {object_name}",
            {"ordered_capabilities": [intent]},
            {},
        )
        contract = TaskContract(
            target_effects=[SemanticPredicate(
                "agent.holds", {"object": object_name},
            )],
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="test",
        )
        composite = CompositeBuilder().validate_and_build(
            proposal,
            [occurrence],
            contract,
            contract_matcher=ExactContractMatcher(),
            task_bindings={task_role: object_name},
        )
        summaries.append(composite.summary)
        refs.append(composite.ref)
    assert all(item.startswith("compose_") for item in summaries)
    assert summaries[0] != summaries[1]
    assert refs[0] != refs[1]


def test_evolution_evidence_aggregates_or_and_emits_one_event_pair() -> None:
    accumulator = EvolutionEvidenceAccumulator()
    accumulator.record(
        "tool://shared@1.0.0",
        "tool",
        occurrence_id="occ_b",
        passed=False,
        reason="replay_failed",
    )
    accumulator.record(
        "tool://shared@1.0.0",
        "tool",
        occurrence_id="occ_a",
        passed=True,
        reason="replay_passed",
    )
    assets = accumulator.assets()
    assert len(assets) == 1
    assert assets[0].validated_any is True
    assert assets[0].metadata()["occurrence_ids"] == ["occ_a", "occ_b"]
    assert len(assets[0].metadata()["validation_outcomes"]) == 2

    attempts = tuple(
        CreditAttempt(
            artifact_ref=item.artifact_ref,
            artifact_kind=item.artifact_kind,
            occurrence_id="evolution",
            attempt_id=f"evolution:{item.artifact_kind}:{item.artifact_ref}",
            sequence_no=index,
            proposed=True,
            validated=item.validated_any,
            metadata=item.metadata(),
        )
        for index, item in enumerate(assets)
    )
    events = CreditAssigner().assign(CreditTrace(
        "task", "trace", attempts,
    ))
    assert [item.event for item in events].count(
        EvidenceEventType.PROPOSED
    ) == 1
    assert [item.event for item in events].count(
        EvidenceEventType.VALIDATED
    ) == 1

    reversed_accumulator = EvolutionEvidenceAccumulator()
    reversed_accumulator.record(
        "tool://shared@1.0.0",
        "tool",
        occurrence_id="occ_a",
        passed=True,
        reason="replay_passed",
    )
    reversed_accumulator.record(
        "tool://shared@1.0.0",
        "tool",
        occurrence_id="occ_b",
        passed=False,
        reason="replay_failed",
    )
    assert reversed_accumulator.assets()[0].metadata() == assets[0].metadata()


def test_legacy_evolution_credit_api_deduplicates_kind_and_ref() -> None:
    events = CreditAssigner().assign_evolution(
        CreditTrace("task", "trace", ()),
        ["skill://atomic@1.0.0", "skill://atomic@1.0.0"],
        [],
        ["tool://shared@1.0.0", "tool://shared@1.0.0"],
        None,
    )
    assert [item.event for item in events].count(
        EvidenceEventType.PROPOSED
    ) == 2
    assert [item.event for item in events].count(
        EvidenceEventType.VALIDATED
    ) == 2
    assert len({
        (item.artifact_kind, item.artifact_ref, item.event)
        for item in events
    }) == len(events)


def _trace() -> TraceRecord:
    return TraceRecord.create(
        TaskRecord("task", "alfworld", "goal", "type", "signature"),
        {},
        {},
        {"source": "full_dynamic"},
    )


@pytest.mark.parametrize(
    ("won", "contract", "resource", "infrastructure", "strict", "learning"),
    [
        (True, True, True, False, True, True),
        (True, False, True, False, False, False),
        (False, True, True, False, False, False),
        (True, True, False, False, True, False),
        (True, True, True, True, True, False),
    ],
)
def test_terminal_outcome_truth_table(
    won: bool,
    contract: bool,
    resource: bool,
    infrastructure: bool,
    strict: bool,
    learning: bool,
) -> None:
    trace = _trace()
    trace.resource_usage_complete = resource
    trace.infrastructure_failure = infrastructure
    terminal = ValidationResult(
        "task_terminal",
        won and contract,
        {"benchmark_won": won, "task_contract": contract},
    )
    apply_terminal_outcome(trace, terminal, SimpleNamespace(won=won))
    assert trace.benchmark_success is won
    assert trace.task_contract_success is contract
    assert trace.strict_task_success is strict
    assert trace.learning_eligible is learning


def test_token_exhaustion_codes_are_task_and_node_scoped() -> None:
    task_budget = RuntimeBudget(
        token_limits={"runtime_dynamic": 1},
    )
    with pytest.raises(BudgetExhausted) as task_error:
        task_budget.consume_llm("runtime_dynamic", 2)
    assert task_error.value.code == "runtime_task_token_budget_exhausted"
    task_failure = FailureLocalizer().localize(
        code=task_error.value.code,
        task_id="task",
        trace_id="trace",
        occurrence_id="",
        attempt_id="dynamic",
        started=True,
    )
    assert task_failure.layer is FailureLayer.RUNTIME_AGENT

    node_budget = RuntimeBudget(
        token_limits={"runtime_seeded": 1},
    )
    with pytest.raises(BudgetExhausted) as node_error:
        node_budget.consume_llm("runtime_seeded", 2)
    assert node_error.value.code == "runtime_node_token_budget_exhausted"
    node_failure = FailureLocalizer().localize(
        code=node_error.value.code,
        task_id="task",
        trace_id="trace",
        occurrence_id="occ",
        attempt_id="seeded",
        started=True,
    )
    assert node_failure.layer is FailureLayer.RUNTIME_AGENT

    task_trace = _trace()
    task_trace.failures.append(task_failure)
    node_trace = _trace()
    node_trace.task.task_id = "task_2"
    node_trace.trace_id = "trace_2"
    node_trace.failures.append(node_failure)
    summary = summarize_traces([
        trace_to_row(task_trace),
        trace_to_row(node_trace),
    ])
    assert summary["task_token_budget_exhausted_count"] == 1
    assert summary["node_token_budget_exhausted_count"] == 1


def test_task_rescue_token_exhaustion_is_localized_and_reported() -> None:
    trace = _trace()
    trace.runtime_plan["source_composite_ref"] = (
        "skill://composite@1.0.0"
    )
    trace.task_rescue_required = True
    trace.environment_actions.append(SimpleNamespace())
    trace.metadata["task_rescue"] = {
        "failure_code": "runtime_task_token_budget_exhausted",
    }
    FailureProcessor(FailureLocalizer()).localize(trace)
    failures = [
        item for item in trace.failures
        if item.code == "runtime_task_token_budget_exhausted"
    ]
    assert len(failures) == 1
    assert failures[0].attempt_id == "task_rescue"
    assert failures[0].layer is FailureLayer.RUNTIME_AGENT
    row = trace_to_row(trace)
    assert row["task_token_budget_exhausted_count"] == 1
    assert row["node_token_budget_exhausted_count"] == 0


def test_report_separates_official_strict_and_learning_success() -> None:
    trace = _trace()
    terminal = ValidationResult(
        "task_terminal",
        False,
        {"benchmark_won": True, "task_contract": False},
    )
    apply_terminal_outcome(trace, terminal, SimpleNamespace(won=True))
    summary = summarize_traces([trace_to_row(trace)])
    assert summary["official_alfworld_won_rate"] == 1.0
    assert summary["strict_task_success_rate"] == 0.0
    assert summary["learning_eligible_success_rate"] == 0.0


def test_planner_p2_metric_counts_attempted_provider_turn() -> None:
    trace = _trace()
    trace.llm_usage = [{
        "bucket": "planner_p2",
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
        "reasoning_tokens": 0,
        "call_count": 1,
        "latency_ms": 1.0,
    }]
    # A content-invalid P2 proposal never reaches workflow_p2, but the P2
    # stage was still activated and must be visible diagnostically.
    trace.planner_audit["workflow_p2"] = {}
    row = trace_to_row(trace)
    assert row["planner_p2_used"] is True
    assert summarize_traces([row])["planner_p2_count"] == 1
