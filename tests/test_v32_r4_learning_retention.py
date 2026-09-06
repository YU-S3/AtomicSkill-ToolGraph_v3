from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import atomic_skillgraph.system as system_module
from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
from atomic_skillgraph.agents.structured_submission import (
    TOOL_PROPOSAL_SCHEMA,
)
from atomic_skillgraph.agents.protocol import validate_schema_instance
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.errors import (
    AgentProtocolError,
    BudgetExhausted,
    FailureLayer,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import PrimitiveToolStep, RuntimeLinearPlan
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
    CanonicalAtomicOccurrence,
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
from atomic_skillgraph.harness.action_catalog import HarnessActionCatalog
from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
from atomic_skillgraph.harness.protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
    PredicateSpec,
)
from atomic_skillgraph.knowledge import (
    ArtifactStore,
    GraphStore,
    SkillRegistry,
    StateDatabase,
    ToolRegistry,
)
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.proposal import (
    ToolProvenance,
    tool_proposal_from_dict,
)
from atomic_skillgraph.tooling.validator import (
    ToolStaticReport,
    ToolStaticValidator,
)
from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeAgentFactory, FakeReply


def _normalized_take(value: str = "apple_1") -> dict[str, Any]:
    effect = {
        "predicate": "agent.holds",
        "args": {"object": value},
        "effect_domain": "world",
        "witness_ref": "action:e0:revision:1",
        "event_index": 0,
        "revision": 1,
        "source_kind": "semantic_snapshot_delta",
    }
    return {
        "trace_id": f"trace_{value}",
        "source_task": {"task_id": f"task_{value}"},
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [{
            "event_index": 0,
            "event_id": "e0",
            "action_id": "e0",
            "action_type": "TAKE",
            "arguments": {"item": value},
            "accepted": True,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span",
            "authoritative_before_state_facts": [],
            "authoritative_positive_effects": [effect],
        }],
        "runtime_spans": [{
            "span_id": "span",
            "kind": "full_dynamic",
            "occurrence_id": "occ",
            "action_start": 0,
            "action_end": 1,
            "parent_span_id": None,
            "learnable": True,
        }],
        "validations": [],
        "boundary_authorities": {
            "inputs": [{
                "authority_ref": "action_arg:e0:item",
                "event_id": "e0",
                "argument_role": "item",
                "kind": "action_argument",
                "source_kind": "action_argument",
                "role": "item",
                "value": value,
            }],
            "effects": [effect],
        },
    }


def _take_proposal(value: str = "apple_1") -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
        phase_id="take",
        intent="take item",
        event_start=0,
        event_end=0,
        input_roles={"item": value},
        output_roles={"result": value},
        preconditions=[],
        effects=[SemanticPredicate(
            "agent.holds", {"object": value}, effect_domain="world",
        )],
        rationale="accepted TAKE establishes possession",
        support_event_ids=["e0"],
        precondition_witness_refs=[],
        effect_witness_refs=["action:e0:revision:1"],
        input_provenance_refs={"item": "action_arg:e0:item"},
        output_derivations={
            "result": {"kind": "input_identity", "input_role": "item"},
        },
        input_provenance_contract="code_authority_v3_2",
    )


class _OneOccurrenceExtractor:
    proposal = _take_proposal()

    def __init__(self, _session: object) -> None:
        pass

    def propose_atomics(self, *_args: Any, **_kwargs: Any):
        return [copy.deepcopy(type(self).proposal)]

    def propose_composite(self, *_args: Any, **_kwargs: Any):
        raise AssertionError("incomplete E1 coverage must not invoke E2")


class _RaisingSession:
    session_id = "raising_tool_builder"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def set_usage_bucket(self, _bucket: str) -> None:
        pass

    def next_turn(self, *_args: Any, **_kwargs: Any) -> AgentTurn:
        raise self.error


class _RejectedStaticValidator:
    def __init__(self, *failure_codes: str) -> None:
        self.failure_codes = list(failure_codes)

    def validate_proposal(self, *_args: Any, **_kwargs: Any) -> ToolStaticReport:
        return ToolStaticReport(
            False,
            {code: False for code in self.failure_codes},
            list(self.failure_codes),
            [f"message for {code}" for code in self.failure_codes],
            {"path_ids": ["program/n1/n2"]},
        )


class _ExplodingStaticValidator:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def validate_proposal(self, *_args: Any, **_kwargs: Any) -> ToolStaticReport:
        raise self.error


def _wire_parameter(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "semantic_type": "entity",
        "required": True,
        "runtime_resolvable": name == "item",
        "required_resolution": "concrete" if name == "item" else "semantic",
        "description": "",
    }


def _create_take_tool_payload(atomic_ref: str = "skill://placeholder@1.0.0") -> dict[str, Any]:
    return {
        "proposal_version": "1",
        "decision": "create",
        "summary": "take the supplied item",
        "atomic_ref": atomic_ref,
        "inputs": [_wire_parameter("item")],
        "outputs": [_wire_parameter("result")],
        "program": [
            {
                "node_id": "n1",
                "op": "ACTION",
                "action_type": "TAKE",
                "argument_mapping": {
                    "item": {"kind": "skill_input", "source_role": "item"},
                },
                "expected_effects": [{
                    "predicate": "agent.holds",
                    "args": {
                        "object": {"kind": "skill_input", "source_role": "item"},
                    },
                    "cardinality": 1,
                    "distinct_by": "",
                    "effect_domain": "world",
                }],
            },
            {
                "node_id": "n2",
                "op": "RETURN",
                "output_sources": {
                    "result": {"source": "tool_input", "field": "item"},
                },
            },
        ],
        "max_actions": 1,
        "final_effects": [{
            "predicate": "agent.holds",
            "args": {
                "object": {"kind": "skill_input", "source_role": "result"},
            },
            "cardinality": 1,
            "distinct_by": "",
            "effect_domain": "world",
        }],
        "evidence_outputs": [],
        "path_expectations": [],
        "rationale": "bounded declarative fixture",
    }


def _no_tool_payload(atomic_ref: str = "skill://placeholder@1.0.0") -> dict[str, Any]:
    payload = _create_take_tool_payload(atomic_ref)
    payload.update({
        "decision": "no_tool",
        "summary": "no reusable tool",
        "program": [],
        "max_actions": 1,
        "final_effects": [],
        "rationale": "the supplied interface is insufficient",
    })
    return payload


def _minimal_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    builder_session: Any,
    static_validator: Any,
) -> tuple[system_module.AtomicSkillGraphSystem, Any, Any, StateDatabase]:
    monkeypatch.setattr(system_module, "ExtractorSession", _OneOccurrenceExtractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    normalized = _normalized_take()
    system = object.__new__(system_module.AtomicSkillGraphSystem)
    system.config = {"method_patch": "3.2"}
    system.usage = object()
    system.mode = "online"
    system.readonly = False
    system.normalizer = SimpleNamespace(build=lambda _trace: copy.deepcopy(normalized))
    system.atomicizer = Atomicizer()
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    system.skills = SkillRegistry(artifacts, database)
    system.tools = ToolRegistry(artifacts, database)
    system.graph = GraphStore(database, system.skills)
    system.aligner = Aligner(system.skills, system.tools)
    system._extractor_session = lambda _task_id: object()
    system._tool_builder_session = lambda *_args: builder_session
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(target_effects=[
            SemanticPredicate("state.uncovered", {"item": "apple_1"}),
        ]),
        contract_matcher=lambda: ExactContractMatcher(),
        profile_name="fake_v3",
        primitive_action_schema=lambda: [{
            "action_type": "TAKE", "argument_roles": ["item"],
        }],
        semantic_predicate_schema=lambda: [
            PredicateSpec(
                "agent.holds", "world", ("object",),
                {"object": "entity"}, "fixture",
            ),
        ],
        supports_constraint=lambda *_args, **_kwargs: True,
        replay_tool=lambda *_args, **_kwargs: True,
    )
    system.tool_compiler = ToolCompiler()
    system.tool_static_validator = static_validator
    system.admission = Admission(ValidationEngine().tool)
    system.credit = CreditAssigner()
    system.ledger = EvidenceLedger(database)
    system.projection = LifecycleProjection(database, system.ledger)
    system.lifecycle = LifecycleController(
        database, system.projection, LifecyclePolicy(),
    )
    system.repair_store = RepairStore(database)
    system.gap_diagnoser = SimpleNamespace(diagnose=lambda *_args, **_kwargs: {})
    trace = SimpleNamespace(
        metadata={},
        runtime_plan={},
        trace_id="trace_static_or_no_tool",
        benchmark_success=False,
        environment_actions=[],
        task=SimpleNamespace(task_id="task_a"),
    )
    task = SimpleNamespace(task_id="task_a", context={})
    return system, trace, task, database


def _evidence_reason(database: StateDatabase, trace_id: str) -> str:
    row = database.execute(
        "SELECT metadata_json FROM evidence_events "
        "WHERE trace_id=? AND artifact_kind='atomic' AND event_type='validated'",
        (trace_id,),
    ).fetchone()
    assert row is not None
    metadata = json.loads(str(row["metadata_json"]))
    outcomes = list(metadata["validation_outcomes"])
    assert len(outcomes) == 1
    return str(outcomes[0]["reason"])


def test_invalid_atomic_never_reaches_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    invalid = _take_proposal()
    # The v3.2 Atomicizer requires one authoritative provenance entry for
    # every input role.  This is an Extractor-content rejection, before any
    # executable is proposed.
    invalid.input_provenance_refs = {}
    monkeypatch.setattr(_OneOccurrenceExtractor, "proposal", invalid)
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(
            AssertionError("ToolBuilder must not see an invalid Atomic"),
        ),
        static_validator=ToolStaticValidator(),
    )
    builder_factory_calls: list[tuple[Any, ...]] = []

    def forbidden_builder_factory(*args: Any, **_kwargs: Any) -> Any:
        builder_factory_calls.append(args)
        raise AssertionError("ToolBuilder must not see an invalid Atomic")

    system._tool_builder_session = forbidden_builder_factory

    with pytest.raises(
        ExtractionContentError,
        match="no valid Atomic occurrences",
    ):
        system._prepare_evolution(trace, task)

    assert builder_factory_calls == []
    assert trace.metadata["evolution_tool_builds"] == []
    assert trace.metadata["tool_build_rejections"] == []
    assert len(trace.metadata["extraction_occurrence_rejections"]) == 1
    rejection = trace.metadata["extraction_occurrence_rejections"][0]
    assert rejection["stage"] == "atomicizer"
    assert rejection["error_code"] == "extractor_e1_occurrence_rejected"
    assert "provenance roles" in rejection["messages"][0]
    metrics = trace.metadata["v32_metrics"]
    assert metrics["extractor_e1_proposal_count"] == 1
    assert metrics["extractor_e1_validated_occurrence_count"] == 0
    assert metrics["extractor_e1_rejection_count"] == 1
    assert metrics["tool_builder_call_count"] == 0
    assert system.skills.list_refs("atomic") == []
    assert system.skills.list_refs("implementation") == []
    assert system.tools.list_refs() == []
    database.close()


def test_existing_atomic_and_tools_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(
            AssertionError("exact reuse must not invoke ToolBuilder"),
        ),
        static_validator=ToolStaticValidator(),
    )
    occurrence = Atomicizer().validate_and_canonicalize(
        [_take_proposal()], _normalized_take(),
    )[0]
    atomic = system._canonical_atomic_for_occurrence(occurrence)
    assert atomic is not None
    legacy = ToolCompiler().compile([occurrence])[0]
    assert legacy.tool is not None
    assert legacy.implementation is not None
    staged = system.aligner.stage_atomic(
        atomic, legacy.tool, legacy.implementation,
    )
    seeded_atomic = replace(staged.atomic, status=SkillStatus.CANDIDATE)
    seeded_tool = replace(staged.tool, status=ToolStatus.CANDIDATE)
    seeded_implementation = replace(
        staged.implementation, status=SkillStatus.CANDIDATE,
    )
    system.skills.register_atomic(seeded_atomic)
    system.tools.register(seeded_tool)
    system.skills.register_implementation(seeded_implementation)

    atomic_refs_before = system.skills.list_refs("atomic")
    implementation_refs_before = system.skills.list_refs("implementation")
    tool_refs_before = system.tools.list_refs()
    atomic_before = to_primitive(system.skills.get_atomic(seeded_atomic.ref))
    implementation_before = to_primitive(
        system.skills.get_implementation(seeded_implementation.ref)
    )
    tool_before = to_primitive(system.tools.get(seeded_tool.ref))

    prepared = system._prepare_evolution(trace, task)
    assert len(prepared.compiled) == 1
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "exact_reuse"
    assert record["builder_entered"] is False
    assert record["session_id"] == ""
    applied = system._apply_evolution(prepared, trace, task)

    assert applied["atomic_refs"] == [seeded_atomic.ref]
    assert applied["tool_refs"] == [seeded_tool.ref]
    assert applied["implementation_refs"] == [seeded_implementation.ref]
    assert system.skills.list_refs("atomic") == atomic_refs_before
    assert system.skills.list_refs("implementation") == implementation_refs_before
    assert system.tools.list_refs() == tool_refs_before
    assert to_primitive(system.skills.get_atomic(seeded_atomic.ref)) == atomic_before
    assert to_primitive(
        system.skills.get_implementation(seeded_implementation.ref)
    ) == implementation_before
    assert to_primitive(system.tools.get(seeded_tool.ref)) == tool_before
    assert trace.metadata["tool_build_rejections"] == []
    assert trace.metadata["knowledge_preparation_rejections"] == []
    assert trace.metadata["v32_metrics"]["tool_builder_call_count"] == 0
    database.close()


def test_valid_atomic_survives_tool_static_rejection_and_counts_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    factory = FakeAgentFactory()
    session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _create_take_tool_payload())],
    )
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=session,
        static_validator=_RejectedStaticValidator(
            "tool_ir_final_effects_missing",
            "tool_ir_effect_reference_invalid",
        ),
    )

    prepared = system._prepare_evolution(trace, task)
    assert len(prepared.compiled) == 1
    assert prepared.compiled[0].tool is None
    assert prepared.compiled[0].implementation is None
    applied = system._apply_evolution(prepared, trace, task)

    assert len(applied["atomic_refs"]) == 1
    assert applied["tool_refs"] == []
    assert applied["implementation_refs"] == []
    assert len(system.skills.list_refs("atomic")) == 1
    assert system.skills.list_refs("implementation") == []
    assert system.tools.list_refs() == []
    assert trace.metadata["extraction_occurrence_rejections"] == []
    assert len(trace.metadata["tool_build_rejections"]) == 1
    assert trace.metadata["tool_build_rejections"][0]["failure_codes"] == [
        "tool_ir_final_effects_missing",
        "tool_ir_effect_reference_invalid",
    ]
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "static_rejected"
    assert record["atomic_only_prepared"] is True
    assert record["atomic_registered"] is True
    metrics = trace.metadata["v32_metrics"]
    assert metrics["extractor_e1_proposal_count"] == 1
    assert metrics["extractor_e1_validated_occurrence_count"] == 1
    assert metrics["extractor_e1_rejection_count"] == 0
    assert metrics["tool_builder_call_count"] == 1
    assert metrics["tool_builder_proposal_count"] == 1
    assert metrics["tool_builder_static_rejection_count"] == 1
    assert metrics["atomic_only_prepared_after_tool_rejection_count"] == 1
    assert metrics["atomic_only_retained_after_tool_rejection_count"] == 1
    assert metrics["atomic_staged_occurrence_count"] == 1
    assert _evidence_reason(database, trace.trace_id) == (
        "tool_builder_rejected_atomic_only"
    )
    factory.assert_exhausted()
    database.close()


def test_valid_atomic_survives_tool_submission_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    session_error = AgentProtocolError(
        "runtime_agent_schema_error",
        "invalid native ToolBuilder submission",
        layer=FailureLayer.RUNTIME_AGENT,
    )
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(session_error),
        static_validator=ToolStaticValidator(),
    )
    trace.trace_id = "trace_submission_rejection"

    prepared = system._prepare_evolution(trace, task)
    applied = system._apply_evolution(prepared, trace, task)

    assert len(applied["atomic_refs"]) == 1
    assert applied["tool_refs"] == []
    assert applied["implementation_refs"] == []
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "submission_rejected"
    assert record["failure_codes"] == ["runtime_agent_schema_error"]
    assert record["atomic_registered"] is True
    metrics = trace.metadata["v32_metrics"]
    assert metrics["tool_builder_submission_rejection_count"] == 1
    assert metrics["tool_builder_no_tool_count"] == 0
    assert metrics["atomic_only_retained_after_tool_rejection_count"] == 1
    assert _evidence_reason(database, trace.trace_id) == (
        "tool_builder_rejected_atomic_only"
    )
    database.close()


def test_no_tool_and_static_rejection_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    factory = FakeAgentFactory()
    no_tool_session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _no_tool_payload())],
    )
    no_tool, trace_no_tool, task, no_tool_db = _minimal_system(
        monkeypatch,
        tmp_path / "no_tool",
        builder_session=no_tool_session,
        static_validator=ToolStaticValidator(),
    )
    trace_no_tool.trace_id = "trace_no_tool"
    prepared = no_tool._prepare_evolution(trace_no_tool, task)
    no_tool._apply_evolution(prepared, trace_no_tool, task)

    record = trace_no_tool.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "no_tool"
    assert record["proposal_received"] is False
    assert record["static_checked"] is False
    assert record["static_passed"] is None
    metrics = trace_no_tool.metadata["v32_metrics"]
    assert metrics["tool_builder_no_tool_count"] == 1
    assert metrics["tool_builder_submission_rejection_count"] == 0
    assert metrics["tool_builder_static_rejection_count"] == 0
    assert metrics["atomic_only_prepared_after_tool_rejection_count"] == 0
    assert metrics["atomic_only_retained_after_tool_rejection_count"] == 0
    assert _evidence_reason(no_tool_db, trace_no_tool.trace_id) == (
        "tool_builder_no_tool_atomic_only"
    )
    factory.assert_exhausted()
    no_tool_db.close()


@pytest.mark.parametrize(
    "error",
    [
        AgentProtocolError(
            "provider_auth_error", "authentication failed",
            layer=FailureLayer.INFRASTRUCTURE,
        ),
        BudgetExhausted(
            "extractor_token_budget_exhausted", "budget exhausted",
            layer=FailureLayer.RUNTIME_AGENT,
        ),
    ],
)
def test_unexpected_provider_and_budget_errors_do_not_become_content_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    error: BaseException,
) -> None:
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=_RaisingSession(error),
        static_validator=ToolStaticValidator(),
    )
    occurrence = Atomicizer().validate_and_canonicalize(
        [_take_proposal()], _normalized_take(),
    )[0]
    atomic = system._canonical_atomic_for_occurrence(occurrence)
    assert atomic is not None
    system._initialize_r4_learning_diagnostics(trace)

    with pytest.raises(type(error)) as caught:
        system._build_tool_for_occurrence(
            occurrence, atomic, _normalized_take(), trace,
        )

    if hasattr(error, "code"):
        assert getattr(caught.value, "code") == getattr(error, "code")
    assert trace.metadata["tool_build_rejections"] == []
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "aborted"
    assert record["atomic_only_prepared"] is False
    assert trace.metadata["v32_metrics"]["tool_builder_aborted_count"] == 1
    database.close()


def test_static_validator_internal_error_does_not_become_content_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    error = RuntimeError("static validator crashed")
    factory = FakeAgentFactory()
    session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _create_take_tool_payload())],
    )
    system, trace, _task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=session,
        static_validator=_ExplodingStaticValidator(error),
    )
    occurrence = Atomicizer().validate_and_canonicalize(
        [_take_proposal()], _normalized_take(),
    )[0]
    atomic = system._canonical_atomic_for_occurrence(occurrence)
    assert atomic is not None
    system._initialize_r4_learning_diagnostics(trace)

    with pytest.raises(RuntimeError, match="static validator crashed"):
        system._build_tool_for_occurrence(
            occurrence, atomic, _normalized_take(), trace,
        )

    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "aborted"
    assert record["failure_stage"] == "tool_static"
    assert record["static_checked"] is False
    assert record["atomic_only_prepared"] is False
    assert trace.metadata["tool_build_rejections"] == []
    assert trace.metadata["v32_metrics"]["tool_builder_aborted_count"] == 1
    factory.assert_exhausted()
    database.close()


def test_static_validator_value_error_is_recorded_at_real_stage_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    factory = FakeAgentFactory()
    session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _create_take_tool_payload())],
    )
    system, trace, task, database = _minimal_system(
        monkeypatch,
        tmp_path,
        builder_session=session,
        static_validator=_ExplodingStaticValidator(
            ValueError("static validator crashed"),
        ),
    )

    with pytest.raises(ExtractionContentError):
        system._prepare_evolution(trace, task)

    assert trace.metadata["tool_build_rejections"] == []
    assert len(trace.metadata["knowledge_preparation_rejections"]) == 1
    rejection = trace.metadata["knowledge_preparation_rejections"][0]
    assert rejection["stage"] == "tool_static"
    assert rejection["error_type"] == "ValueError"
    assert rejection["error_code"] == "knowledge_preparation_failed"
    assert rejection["failure_codes"] == []
    assert rejection["messages"] == ["static validator crashed"]
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "aborted"
    assert record["failure_stage"] == "tool_static"
    assert record["atomic_only_prepared"] is False
    metrics = trace.metadata["v32_metrics"]
    assert metrics["tool_builder_aborted_count"] == 1
    assert metrics["atomic_only_prepared_after_tool_rejection_count"] == 0
    assert metrics["atomic_staged_occurrence_count"] == 0
    assert system.skills.list_refs("atomic") == []
    assert system.skills.list_refs("implementation") == []
    assert system.tools.list_refs() == []
    factory.assert_exhausted()
    database.close()


def _navigation_atomic() -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef("example_navigation", "1.0.0"),
        "navigate to supplied destination",
        [ParameterSpec(
            "destination", "entity", True, True, "concrete", "",
        )],
        [ParameterSpec(
            "arrived_location", "entity", True, False, "semantic", "",
        )],
        [],
        [SemanticPredicate(
            "agent.at_location",
            {"location": BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role="arrived_location",
            )},
            effect_domain="world",
        )],
        {
            "validator_id": "harness_atomic_effect",
            "identity_strict": True,
            "output_derivations": {
                "arrived_location": {
                    "kind": "effect_witness",
                    "predicate": "agent.at_location",
                    "argument_role": "location",
                },
            },
        },
        [],
        {},
        {},
        SkillStatus.DRAFT,
    )


def _navigation_payload() -> dict[str, Any]:
    return {
        "proposal_version": "1",
        "decision": "create",
        "summary": "navigate to supplied destination",
        "atomic_ref": "skill://example_navigation@1.0.0",
        "inputs": [{
            "name": "destination",
            "semantic_type": "entity",
            "required": True,
            "runtime_resolvable": True,
            "required_resolution": "concrete",
            "description": "",
        }],
        "outputs": [{
            "name": "arrived_location",
            "semantic_type": "entity",
            "required": True,
            "runtime_resolvable": False,
            "required_resolution": "semantic",
            "description": "",
        }],
        "program": [
            {
                "node_id": "n1",
                "op": "ACTION",
                "action_type": "GO_TO",
                "argument_mapping": {
                    "destination": {
                        "kind": "skill_input", "source_role": "destination",
                    },
                },
                "expected_effects": [{
                    "predicate": "agent.at_location",
                    "args": {
                        "location": {
                            "kind": "skill_input", "source_role": "destination",
                        },
                    },
                    "cardinality": 1,
                    "distinct_by": "",
                    "effect_domain": "world",
                }],
            },
            {
                "node_id": "n2",
                "op": "RETURN",
                "output_sources": {
                    "arrived_location": {
                        "source": "tool_input", "field": "destination",
                    },
                },
            },
        ],
        "max_actions": 1,
        "final_effects": [{
            "predicate": "agent.at_location",
            "args": {
                "location": {
                    "kind": "skill_input", "source_role": "arrived_location",
                },
            },
            "cardinality": 1,
            "distinct_by": "",
            "effect_domain": "world",
        }],
        "evidence_outputs": [],
        "path_expectations": [],
        "rationale": (
            "The action establishes arrival and RETURN publishes that same "
            "witnessed location under the declared output role."
        ),
    }


class _NavigationHarness:
    profile_name = "fake_v3"

    def __init__(self) -> None:
        self._catalog = HarnessActionCatalog(self._parse_action)
        self._validator = AlfWorldValidatorChannel()
        self._task: HarnessTask | None = None
        self._revision = 0

    @staticmethod
    def _parse_action(raw: Mapping[str, Any]):
        return (
            str(raw["action_type"]),
            dict(raw.get("arguments") or {}),
            str(raw.get("display_text", raw["action_type"])),
            {},
        )

    def _replace_catalog(self) -> list[HarnessActionSpec]:
        destination = str(self._task.context["destination"]) if self._task else ""
        return self._catalog.replace([{
            "action_type": "GO_TO",
            "arguments": {"destination": destination},
            "display_text": f"go to {destination}",
        }], self._revision)

    def reset(self, task: HarnessTask) -> HarnessActionResult:
        self._task = task
        self._revision = 0
        self._validator.reset()
        catalog = self._replace_catalog()
        self._validator.set_catalog(catalog)
        return HarnessActionResult(
            True, "ready", False, False, 0, catalog, {"reset": True},
        )

    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog.items()

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        spec = self._catalog.get(action_id, revision)
        self._revision += 1
        catalog = self._replace_catalog()
        self._validator.record(
            spec,
            accepted=True,
            revision=self._revision,
            done=False,
            won=False,
            observation="arrived",
            catalog=catalog,
        )
        return HarnessActionResult(
            True, "arrived", False, False, self._revision, catalog,
        )

    def compile_primitive(
        self, primitive: PrimitiveToolStep, bindings: dict[str, Any],
    ) -> HarnessActionSpec:
        expected = {
            role: (
                expression.constant
                if expression.kind is BindingExprKind.CONSTANT
                else bindings.get(expression.source_role)
            )
            for role, expression in primitive.argument_mapping.items()
        }
        return next(
            spec for spec in self.action_catalog()
            if spec.action_type == primitive.action_type
            and spec.arguments == expected
        )

    def execute_primitive(
        self, primitive: PrimitiveToolStep, bindings: dict[str, Any],
    ) -> HarnessActionResult:
        spec = self.compile_primitive(primitive, bindings)
        return self.execute_action(spec.action_id, spec.revision)

    def validator_channel(self) -> AlfWorldValidatorChannel:
        return self._validator

    def task_contract(self, task: HarnessTask) -> TaskContract:
        return TaskContract(target_effects=[SemanticPredicate(
            "agent.at_location", {"location": task.context["destination"]},
        )])

    def contract_matcher(self) -> ExactContractMatcher:
        return ExactContractMatcher()

    def semantic_predicate_schema(self) -> list[PredicateSpec]:
        return [PredicateSpec(
            "agent.at_location", "world", ("location",),
            {"location": "entity"}, "fixture",
        )]

    def primitive_action_schema(self) -> list[dict[str, Any]]:
        return [{"action_type": "GO_TO", "argument_roles": ["destination"]}]

    def supports_constraint(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


def _navigation_occurrence() -> CanonicalAtomicOccurrence:
    return CanonicalAtomicOccurrence(
        occurrence_id="navigate_occurrence",
        phase_id="navigate",
        intent="navigate to supplied destination",
        event_start=0,
        event_end=0,
        input_bindings={"destination": "desk_1"},
        output_bindings={"arrived_location": "desk_1"},
        input_specs=[ParameterSpec(
            "destination", "entity", True, True, "concrete", "",
        )],
        output_specs=[ParameterSpec(
            "arrived_location", "entity", True, False, "semantic", "",
        )],
        preconditions=[],
        effects=copy.deepcopy(_navigation_atomic().effects),
        action_events=[{
            "action_type": "GO_TO",
            "arguments": {"destination": "desk_1"},
            "accepted": True,
        }],
        prefix_events=[],
        source_task={"task_id": "navigation_task"},
        source_trace_id="navigation_trace",
        proposed_ref=SkillRef("example_navigation", "1.0.0"),
        output_derivations={
            "arrived_location": {
                "kind": "effect_witness",
                "predicate": "agent.at_location",
                "argument_role": "location",
            },
        },
    )


def test_final_effect_output_role_example_passes_schema_static_and_tool_runner() -> None:
    payload = _navigation_payload()
    validate_schema_instance(payload, TOOL_PROPOSAL_SCHEMA)
    proposal = tool_proposal_from_dict(payload)
    atomic = _navigation_atomic()
    harness = _NavigationHarness()
    static = ToolStaticValidator().validate_proposal(proposal, atomic, harness)
    assert static.passed, static.failure_codes

    compiled = ToolCompiler().compile_proposal(
        _navigation_occurrence(),
        atomic,
        proposal,
        ToolProvenance(
            "success_evolution",
            str(atomic.ref),
            "navigation_trace",
            "navigate_occurrence",
            task_id="navigation_task",
        ),
    )
    assert compiled.tool is not None
    task = HarnessTask(
        "navigation_task", "arrive at desk_1", "fake",
        context={"destination": "desk_1"},
    )
    plan = RuntimeLinearPlan.full_dynamic(
        task.task_id, harness.task_contract(task), reason="fixture",
    )
    trace = TraceRecord.create(
        TaskRecord(task.task_id, "fake", task.goal, "navigation", "sig", {}),
        to_primitive(plan.task_contract),
        {},
        {"source": "fixture"},
    )
    ctx = TaskRuntimeContext.create(
        task,
        plan,
        harness,
        TraceBuilder(trace),
        RuntimeBudget(global_action_budget=5, node_action_budget=5),
    )
    result = ToolRunner(ValidationEngine().tool).run(
        compiled.tool,
        {"destination": "desk_1"},
        ctx,
        occurrence_id="navigate_occurrence",
    )

    assert result.completed is True
    assert result.atomic_effect_passed is True
    assert result.output_candidates == {"arrived_location": "desk_1"}
    assert result.tool_path_evidence["step_effect_results"][0][
        "step_effect_passed"
    ] is True
    assert result.tool_path_evidence["final_effect_result"]["passed"] is True


def test_final_effect_input_role_counterexample_remains_rejected() -> None:
    payload = _navigation_payload()
    payload["final_effects"][0]["args"]["location"]["source_role"] = (
        "destination"
    )
    validate_schema_instance(payload, TOOL_PROPOSAL_SCHEMA)

    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload),
        _navigation_atomic(),
        _NavigationHarness(),
    )

    assert report.passed is False
    assert "tool_ir_final_effects_missing" in report.failure_codes


def test_action_expected_effect_tool_output_is_rejected_even_with_source_step() -> None:
    payload = _navigation_payload()
    payload["program"][0]["expected_effects"][0]["args"]["location"] = {
        "kind": "tool_output",
        "source_role": "arrived_location",
        "source_step": "n1",
    }
    validate_schema_instance(payload, TOOL_PROPOSAL_SCHEMA)

    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload),
        _navigation_atomic(),
        _NavigationHarness(),
    )

    assert report.passed is False
    assert "tool_ir_effect_reference_invalid" in report.failure_codes
