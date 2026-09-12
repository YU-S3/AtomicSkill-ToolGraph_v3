from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import atomic_skillgraph.system as system_module
from atomic_skillgraph.agents import UsageLedger as AgentUsageLedger
from atomic_skillgraph.agents.protocol import validate_schema_instance
from atomic_skillgraph.agents.structured_submission import TOOL_PROPOSAL_SCHEMA
from atomic_skillgraph.core.contracts import SemanticPredicate, TaskContract
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.extractor_session import E1_SCHEMA
from atomic_skillgraph.evolution.repair import RepairStore
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.governance import (
    CreditAssigner,
    EvidenceLedger,
    LifecycleController,
    LifecyclePolicy,
    LifecycleProjection,
)
from atomic_skillgraph.harness.protocol import PredicateSpec
from atomic_skillgraph.knowledge import (
    ArtifactStore,
    GraphStore,
    SkillRegistry,
    StateDatabase,
    ToolRegistry,
)
from atomic_skillgraph.tooling.proposal import tool_proposal_from_dict
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeAgentFactory, FakeReply


_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "v32_r4"
    / "trace_44c43bd92d274dffb964b252a440db51_minimal.json"
)


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _predicate(raw: Mapping[str, Any]) -> SemanticPredicate:
    return SemanticPredicate(
        str(raw["predicate"]),
        copy.deepcopy(dict(raw.get("args") or {})),
        int(raw.get("cardinality", 1)),
        str(raw.get("distinct_by", "")),
        str(raw.get("effect_domain", "world")),
    )


def _replay_proposal(fixture: Mapping[str, Any]) -> AtomicOccurrenceProposal:
    original = copy.deepcopy(
        dict(fixture["e1_native_submission_original"])["occurrences"][0]
    )
    rebase = dict(fixture["replay_rebase"])
    assert [original["event_start"], original["event_end"]] == [
        rebase["source_event_start"],
        rebase["source_event_end"],
    ]
    original["event_start"] = int(rebase["replay_event_start"])
    original["event_end"] = int(rebase["replay_event_end"])
    # The frozen R4 fixture intentionally remains a verbatim historical
    # submission.  Its placed object is nevertheless the declared ``object``
    # input, so replay the unrelated ToolBuilder regression through the R9
    # identity-lineage contract instead of relying on the old misclassification.
    original["output_derivations"]["placed_object"] = {
        "kind": "input_identity",
        "input_role": "object",
    }
    return AtomicOccurrenceProposal(
        phase_id=str(original["phase_id"]),
        intent=str(original["intent"]),
        event_start=int(original["event_start"]),
        event_end=int(original["event_end"]) - 1,
        input_roles=copy.deepcopy(dict(original["input_roles"])),
        output_roles=copy.deepcopy(dict(original["output_roles"])),
        preconditions=[_predicate(item) for item in original["preconditions"]],
        effects=[_predicate(item) for item in original["effects"]],
        rationale=str(original["rationale"]),
        support_event_ids=[
            str(item) for item in original.get("support_event_ids", [])
        ],
        shared_precondition_event_ids=[
            str(item)
            for item in original.get("shared_precondition_event_ids", [])
        ],
        precondition_witness_refs=[
            str(item)
            for item in original.get("precondition_witness_refs", [])
        ],
        effect_witness_refs=[
            str(item) for item in original.get("effect_witness_refs", [])
        ],
        ordering_constraints=copy.deepcopy(
            list(original.get("ordering_constraints", []))
        ),
        input_provenance_refs={
            str(role): str(authority)
            for role, authority in dict(
                original["input_provenance_refs"]
            ).items()
        },
        output_derivations={
            str(role): copy.deepcopy(dict(derivation))
            for role, derivation in dict(
                original["output_derivations"]
            ).items()
        },
        input_provenance_contract="code_authority_v3_2",
    )


class _FixtureHarness:
    profile_name = "alfworld_v3_fixture"

    @staticmethod
    def primitive_action_schema() -> list[dict[str, Any]]:
        return [{
            "action_type": "MOVE",
            "argument_roles": ["object", "destination"],
        }]

    @staticmethod
    def semantic_predicate_schema() -> list[PredicateSpec]:
        return [
            PredicateSpec(
                "agent.at_location",
                "world",
                ("location",),
                {"location": "entity"},
                "fixture",
            ),
            PredicateSpec(
                "agent.holds",
                "world",
                ("object",),
                {"object": "entity"},
                "fixture",
            ),
            PredicateSpec(
                "object.at_location",
                "world",
                ("object", "location"),
                {"object": "entity", "location": "entity"},
                "fixture",
            ),
        ]

    @staticmethod
    def task_contract(_task: Any) -> TaskContract:
        # Keep E2 outside this focused replay.  The preserved Atomic remains
        # independently valid even though this synthetic task contract is not
        # covered by it.
        return TaskContract(target_effects=[
            SemanticPredicate("state.uncovered", {"value": "fixture"}),
        ])

    @staticmethod
    def contract_matcher() -> ExactContractMatcher:
        return ExactContractMatcher()

    @staticmethod
    def supports_constraint(*_args: Any, **_kwargs: Any) -> bool:
        return True

    @staticmethod
    def replay_tool(*_args: Any, **_kwargs: Any) -> bool:
        return True


def _canonical_occurrence_and_atomic(
    fixture: Mapping[str, Any],
) -> tuple[Any, Any]:
    normalized = copy.deepcopy(dict(fixture["normalized_trace_minimal"]))
    canonical, rejections = Atomicizer().validate_proposed_subset(
        [_replay_proposal(fixture)], normalized,
    )
    assert rejections == []
    system = object.__new__(system_module.AtomicSkillGraphSystem)
    atomic = system._canonical_atomic_for_occurrence(canonical[0])
    assert atomic is not None
    return canonical[0], atomic


def _historical_r4_control(
    fixture: Mapping[str, Any], *, atomic_ref: str,
) -> dict[str, Any]:
    original = dict(fixture["tool_builder_native_submission_original"])
    payload = copy.deepcopy(dict(original["value"]))
    control = dict(fixture["human_authored_legal_control"])
    assert control["classification"] == (
        "human_authored_legal_control_not_model_output"
    )
    assert control["base"] == "tool_builder_native_submission_original.value"
    operations = list(control["operations"])
    assert operations == [
        {
            "op": "replace",
            "path": "/atomic_ref",
            "value": "$CANONICAL_ATOMIC_REF",
        },
        {
            "op": "replace",
            "path": "/final_effects/0/args/object/source_role",
            "value": "placed_object",
        },
    ]
    payload["atomic_ref"] = atomic_ref
    payload["final_effects"][0]["args"]["object"]["source_role"] = (
        "placed_object"
    )
    return payload


def _r9_legal_control(
    fixture: Mapping[str, Any], *, atomic_ref: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(
        fixture["tool_builder_native_submission_original"]["value"]
    ))
    payload["atomic_ref"] = atomic_ref
    return payload


def _system_for_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: Mapping[str, Any],
    builder_session: Any,
) -> tuple[Any, Any, Any, StateDatabase]:
    proposal = _replay_proposal(fixture)

    class _FixtureExtractor:
        def __init__(self, _session: Any) -> None:
            pass

        def propose_atomics(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return [copy.deepcopy(proposal)]

        def propose_composite(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("uncovered synthetic contract must not invoke E2")

    monkeypatch.setattr(system_module, "ExtractorSession", _FixtureExtractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )

    normalized = copy.deepcopy(dict(fixture["normalized_trace_minimal"]))
    system = object.__new__(system_module.AtomicSkillGraphSystem)
    system.config = {"method_patch": "3.2"}
    system.usage = AgentUsageLedger()
    system._current_task_usage_start = 0
    system.mode = "online"
    system.readonly = False
    system.normalizer = SimpleNamespace(
        build=lambda _trace: copy.deepcopy(normalized),
    )
    system.atomicizer = Atomicizer()
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    system.skills = SkillRegistry(artifacts, database)
    system.tools = ToolRegistry(artifacts, database)
    system.graph = GraphStore(database, system.skills)
    system.aligner = Aligner(system.skills, system.tools)
    system._extractor_session = lambda _task_id: object()
    system._tool_builder_session = lambda *_args: builder_session
    system.harness = _FixtureHarness()
    system.tool_compiler = ToolCompiler()
    system.tool_static_validator = ToolStaticValidator()
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
        trace_id=str(dict(fixture["source"])["trace_id"]),
        benchmark_success=False,
        environment_actions=[],
        task=SimpleNamespace(
            task_id=str(dict(fixture["source"])["task_id"]),
        ),
    )
    task = SimpleNamespace(
        task_id=str(dict(fixture["source"])["task_id"]),
        context={},
    )
    return system, trace, task, database


def test_real_trace_historical_tool_payload_passes_after_r9_lineage_fix() -> None:
    fixture = _load_fixture()
    source = dict(fixture["source"])
    assert source["trace_sha256"] == (
        "2d4cb747e89c04c31e1e8489e358fa72544aa7662d7a9ef472b9f90659d2dfd2"
    )
    assert source["classification"] == (
        "verbatim_native_submissions_from_historical_trace"
    )
    validate_schema_instance(fixture["e1_native_submission_original"], E1_SCHEMA)

    _occurrence, atomic = _canonical_occurrence_and_atomic(fixture)
    original = dict(fixture["tool_builder_native_submission_original"])
    assert original["classification"] == (
        "verbatim_historical_model_output_invalid"
    )
    payload = copy.deepcopy(dict(original["value"]))
    validate_schema_instance(payload, TOOL_PROPOSAL_SCHEMA)
    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload), atomic, _FixtureHarness(),
    )

    assert report.passed is True
    assert report.failure_codes == []
    assert payload == original["value"]


def test_real_trace_static_rejection_retains_valid_atomic_with_r4_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _load_fixture()
    original = dict(fixture["tool_builder_native_submission_original"])
    payload = _historical_r4_control(
        fixture,
        atomic_ref=str(original["value"]["atomic_ref"]),
    )
    factory = FakeAgentFactory()
    builder_session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", payload)],
    )
    system, trace, task, database = _system_for_replay(
        monkeypatch, tmp_path, fixture, builder_session,
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
    rejection = trace.metadata["tool_build_rejections"][0]
    assert rejection["stage"] == "tool_static"
    assert rejection["error_code"] == "tool_builder_static_rejected"
    assert rejection["failure_codes"] == original[
        "expected_static_failure_codes"
    ]
    record = trace.metadata["evolution_tool_builds"][0]
    assert record["outcome"] == "static_rejected"
    assert record["proposal_received"] is True
    assert record["static_checked"] is True
    assert record["static_passed"] is False
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
    assert original["value"]["final_effects"][0]["args"]["object"][
        "source_role"
    ] == "object"
    assert payload["final_effects"][0]["args"]["object"][
        "source_role"
    ] == "placed_object"
    factory.assert_exhausted()
    database.close()


def test_r9_input_identity_control_is_labeled_and_passes_static() -> None:
    fixture = _load_fixture()
    _occurrence, atomic = _canonical_occurrence_and_atomic(fixture)
    payload = _r9_legal_control(fixture, atomic_ref=str(atomic.ref))

    validate_schema_instance(payload, TOOL_PROPOSAL_SCHEMA)
    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload), atomic, _FixtureHarness(),
    )

    assert report.passed is True, report.failure_codes
    assert report.failure_codes == []
    assert fixture["tool_builder_native_submission_original"]["value"][
        "final_effects"
    ][0]["args"]["object"]["source_role"] == "object"
    assert payload["final_effects"][0]["args"]["object"][
        "source_role"
    ] == "object"
