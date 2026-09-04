from __future__ import annotations

import json
from types import SimpleNamespace

import atomic_skillgraph.system as system_module
from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
    CanonicalAtomicOccurrence,
)
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.extraction_authority import (
    contract_coverage_report,
    extraction_coverage_authority,
)
from atomic_skillgraph.evolution.extractor_session import ExtractionContentError
from atomic_skillgraph.evolution.repair import RepairStore
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.governance import (
    CreditAssigner,
    EvidenceLedger,
    LifecycleController,
    LifecyclePolicy,
    LifecycleProjection,
)
from atomic_skillgraph.knowledge import (
    ArtifactStore,
    GraphStore,
    SkillRegistry,
    StateDatabase,
    ToolRegistry,
)
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from atomic_skillgraph.validation.engine import ValidationEngine


def _occurrence(
    occurrence_id: str,
    *,
    predicate: str = "state.p",
    value: str = "item_1",
) -> CanonicalAtomicOccurrence:
    expression = BindingExpression(
        BindingExprKind.SKILL_INPUT,
        source_role="item",
    )
    return CanonicalAtomicOccurrence(
        occurrence_id=occurrence_id,
        phase_id=occurrence_id,
        intent="establish_state",
        event_start=0,
        event_end=0,
        input_bindings={"item": value},
        output_bindings={"result": value},
        input_specs=[ParameterSpec("item", "entity", True, True, "concrete")],
        output_specs=[ParameterSpec("result", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(predicate, {"item": expression})],
        action_events=[],
        prefix_events=[],
        source_task={},
        source_trace_id="trace",
        proposed_ref=SkillRef(f"atomic_{occurrence_id}", "1.0.0"),
    )


def _normalized() -> dict:
    return {
        "trace_id": "trace",
        "source_task": {},
        "runtime_spans": [{
            "span_id": "span",
            "kind": "full_dynamic",
            "occurrence_id": "phase",
            "action_start": 0,
            "action_end": 1,
            "parent_span_id": None,
            "closed": True,
        }],
        "validations": [],
        "actions": [{
            "event_index": 0,
            "action_id": "a0",
            "action_type": "GENERIC",
            "arguments": {"item": "item_1"},
            "accepted": True,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span",
            "authoritative_positive_effects": [{
                "predicate": "state.p",
                "args": {"item": "item_1"},
                "witness_ref": "fact:p:item_1",
                "event_index": 0,
            }],
        }],
    }


def test_state_derived_target_witness_authority_and_missing_target() -> None:
    matcher = ExactContractMatcher()
    witnessed = extraction_coverage_authority(
        _normalized(),
        TaskContract(target_effects=[SemanticPredicate(
            "state.p", {"item": "item_1"},
        )]),
        matcher,
    )
    assert witnessed.all_targets_witnessed is True
    assert witnessed.targets[0].witness_event_indexes == (0,)
    assert witnessed.targets[0].witness_facts[0]["witness_ref"] == (
        "fact:p:item_1"
    )

    invented = extraction_coverage_authority(
        _normalized(),
        TaskContract(target_effects=[SemanticPredicate(
            "state.invented", {"item": "item_1"},
        )]),
        matcher,
    )
    assert invented.all_targets_witnessed is False
    assert invented.targets[0].witness_facts == ()


def test_e1_policy_context_carries_only_code_authoritative_target_witnesses() -> None:
    authority = extraction_coverage_authority(
        _normalized(),
        TaskContract(target_effects=[SemanticPredicate(
            "state.p", {"item": "item_1"},
        )]),
        ExactContractMatcher(),
    )
    prompt = ContextBuilder().extractor_e1(
        canonical_trace={"actions": []},
        required_task_contract_witnesses=authority,
    )
    instruction, payload = prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)
    assert "collectively cover every supplied" in instruction
    supplied = json.loads(payload)["required_task_contract_witnesses"]
    assert supplied["all_targets_witnessed"] is True
    assert supplied["targets"][0]["witness_facts"][0]["predicate"] == (
        "state.p"
    )


def test_one_contract_coverage_report_is_used_for_pass_and_failure() -> None:
    matcher = ExactContractMatcher()
    canonical = [_occurrence("one")]
    passed = contract_coverage_report(
        TaskContract(target_effects=[SemanticPredicate(
            "state.p", {"item": "item_1"},
        )]),
        canonical,
        matcher,
    )
    assert passed.passed is True
    assert passed.target_checks[0]["matched_occurrence_ids"] == ["one"]

    incomplete = contract_coverage_report(
        TaskContract(target_effects=[SemanticPredicate(
            "state.q", {"item": "item_1"},
        )]),
        canonical,
        matcher,
    )
    assert incomplete.passed is False
    assert "extractor_contract_target_uncovered" in (
        incomplete.failure_codes
    )


def test_e1_incomplete_coverage_prepares_atomic_but_skips_e2(
    monkeypatch, tmp_path,
) -> None:
    normalized = _normalized()
    normalized["actions"][0]["action_type"] = "TAKE"
    normalized["actions"][0]["authoritative_positive_effects"] = [{
        "predicate": "agent.holds",
        "args": {"object": "item_1"},
        "witness_ref": "fact:holds:item_1",
        "event_index": 0,
    }]
    proposal = AtomicOccurrenceProposal(
        phase_id="phase",
        intent="establish_state",
        event_start=0,
        event_end=0,
        input_roles={"item": "item_1"},
        output_roles={"result": "item_1"},
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds", {"object": "item_1"},
        )],
        rationale="accepted transition",
    )
    rejected_proposal = AtomicOccurrenceProposal(
        phase_id="rejected_phase",
        intent="invent_unbound_transition",
        event_start=0,
        event_end=0,
        input_roles={"other": "missing_1"},
        output_roles={"result": "missing_1"},
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds", {"object": "item_1"},
        )],
        rationale="lacks action/binding provenance",
    )

    class FakeExtractor:
        e2_called = False

        def __init__(self, _session) -> None:
            pass

        def propose_atomics(self, *_args, **_kwargs):
            return [proposal, rejected_proposal]

        def propose_composite(self, *_args, **_kwargs):
            type(self).e2_called = True
            raise AssertionError("E2 must not run after incomplete E1 coverage")

    monkeypatch.setattr(system_module, "ExtractorSession", FakeExtractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    system = object.__new__(AtomicSkillGraphSystem)
    system.normalizer = SimpleNamespace(build=lambda _trace: normalized)
    system.atomicizer = Atomicizer()
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    system.skills = SkillRegistry(artifacts, database)
    system.tools = ToolRegistry(artifacts, database)
    system.graph = GraphStore(database, system.skills)
    system.aligner = Aligner(system.skills, system.tools)
    system._extractor_session = lambda _task_id: object()
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(target_effects=[
            SemanticPredicate("state.q", {"item": "item_1"}),
        ]),
        contract_matcher=lambda: ExactContractMatcher(),
        profile_name="fake_v3",
        replay_tool=lambda *_args, **_kwargs: True,
        supports_constraint=lambda *_args, **_kwargs: True,
    )
    system.tool_compiler = ToolCompiler()
    system.admission = Admission(ValidationEngine().tool)
    system.credit = CreditAssigner()
    system.ledger = EvidenceLedger(database)
    system.projection = LifecycleProjection(database, system.ledger)
    system.lifecycle = LifecycleController(
        database, system.projection, LifecyclePolicy(),
    )
    system.repair_store = RepairStore(database)
    system.gap_diagnoser = SimpleNamespace(
        diagnose=lambda *_args, **_kwargs: {},
    )
    trace = SimpleNamespace(
        metadata={},
        runtime_plan={},
        trace_id="trace_partial",
        task=SimpleNamespace(task_id="task"),
    )
    task = SimpleNamespace(task_id="task", context={})

    prepared = system._prepare_evolution(trace, task)

    assert len(prepared.compiled) == 1
    assert prepared.composite is None
    assert prepared.composite_rejection["error_code"] == (
        "extractor_e1_task_contract_coverage_incomplete"
    )
    assert FakeExtractor.e2_called is False
    assert trace.metadata["extractor_contract_coverage"]["passed"] is False
    assert trace.metadata["extraction"]["e2_attempted"] is False
    assert trace.metadata["extraction"]["e1_proposed"] == 2
    assert trace.metadata["extraction"]["e1_validated"] == 1
    assert trace.metadata["extraction"]["e1_rejected"] == 1
    assert [
        item["phase_id"]
        for item in trace.metadata["extraction_occurrence_rejections"]
    ] == ["rejected_phase"]
    applied = system._apply_evolution(prepared, trace, task)
    assert len(applied["atomic_refs"]) == 1
    assert len(applied["implementation_refs"]) == 1
    assert len(applied["tool_refs"]) == 1
    assert applied["composite_ref"] is None
    assert applied["composite_validated"] is False
    assert len(system.skills.list_refs("atomic")) == 1
    assert len(system.skills.list_refs("implementation")) == 1
    assert system.skills.list_refs("composite") == []
    assert len(system.tools.list_refs()) == 1
    assert {
        row["artifact_kind"]
        for row in database.rows(
            "SELECT artifact_kind FROM evidence_events"
        )
    } == {"atomic", "implementation", "tool"}

    # A second independent Trace with the exact same canonical contract must
    # reuse the Atomic while adding auditable evidence/lifecycle support.
    normalized["trace_id"] = "trace_partial_2"
    trace_2 = SimpleNamespace(
        metadata={},
        runtime_plan={},
        trace_id="trace_partial_2",
        task=SimpleNamespace(task_id="task_2"),
    )
    task_2 = SimpleNamespace(task_id="task_2", context={})
    prepared_2 = system._prepare_evolution(trace_2, task_2)
    applied_2 = system._apply_evolution(prepared_2, trace_2, task_2)
    assert applied_2["atomic_refs"] == applied["atomic_refs"]
    assert len(system.skills.list_refs("atomic")) == 1
    assert trace_2.metadata["extractor_quality"][
        "partial_atomic_alignment_reuse_count"
    ] == 1
    atomic_stats = system.projection.stats(
        str(applied["atomic_refs"][0]), "atomic",
    )
    assert atomic_stats.validated_count == 2
    assert {
        (row["task_id"], row["trace_id"])
        for row in database.rows(
            "SELECT task_id,trace_id FROM evidence_events "
            "WHERE artifact_ref=? AND event_type='validated'",
            (str(applied["atomic_refs"][0]),),
        )
    } == {
        ("task", "trace_partial"),
        ("task_2", "trace_partial_2"),
    }
    database.close()


def test_e2_rejection_does_not_discard_prepared_atomic(
    monkeypatch, tmp_path,
) -> None:
    normalized = _normalized()
    normalized["actions"][0]["action_type"] = "TAKE"
    normalized["actions"][0]["authoritative_positive_effects"] = [{
        "predicate": "agent.holds",
        "args": {"object": "item_1"},
        "witness_ref": "fact:holds:item_1",
        "event_index": 0,
    }]
    proposal = AtomicOccurrenceProposal(
        phase_id="phase",
        intent="establish_state",
        event_start=0,
        event_end=0,
        input_roles={"item": "item_1"},
        output_roles={"result": "item_1"},
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds", {"object": "item_1"},
        )],
        rationale="accepted transition",
    )

    class RejectingE2Extractor:
        e2_called = False

        def __init__(self, _session) -> None:
            pass

        def propose_atomics(self, *_args, **_kwargs):
            return [proposal]

        def propose_composite(self, *_args, **_kwargs):
            type(self).e2_called = True
            raise ExtractionContentError(
                "e2",
                "extractor_e2_schema_rejected",
                "invalid Composite submission",
            )

    monkeypatch.setattr(
        system_module, "ExtractorSession", RejectingE2Extractor,
    )
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    system = object.__new__(AtomicSkillGraphSystem)
    system.config = {}
    system.normalizer = SimpleNamespace(build=lambda _trace: normalized)
    system.atomicizer = Atomicizer()
    system.skills = SkillRegistry(artifacts, database)
    system.tools = ToolRegistry(artifacts, database)
    system.graph = GraphStore(database, system.skills)
    system.aligner = Aligner(system.skills, system.tools)
    system.tool_compiler = ToolCompiler()
    system._extractor_session = lambda _task_id: object()
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(target_effects=[
            SemanticPredicate("agent.holds", {"object": "item_1"}),
        ]),
        contract_matcher=lambda: ExactContractMatcher(),
    )
    system.gap_diagnoser = SimpleNamespace(
        diagnose=lambda *_args, **_kwargs: {},
    )
    trace = SimpleNamespace(metadata={}, runtime_plan={})
    task = SimpleNamespace(task_id="task", context={})

    prepared = system._prepare_evolution(trace, task)

    assert len(prepared.compiled) == 1
    assert prepared.composite is None
    assert prepared.composite_rejection == {
        "stage": "e2",
        "error_type": "ExtractionContentError",
        "error_code": "extractor_e2_schema_rejected",
        "error": "invalid Composite submission",
    }
    assert RejectingE2Extractor.e2_called is True
    assert trace.metadata["extractor_contract_coverage"]["passed"] is True
    assert trace.metadata["extractor_quality"]["extractor_e2_attempted"] is True
    database.close()
