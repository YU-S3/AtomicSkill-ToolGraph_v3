from __future__ import annotations

from types import SimpleNamespace

import atomic_skillgraph.system as system_module

from atomic_skillgraph.core.contracts import SemanticPredicate, TaskContract
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.knowledge import (
    ArtifactStore,
    GraphStore,
    SkillRegistry,
    StateDatabase,
    ToolRegistry,
)
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher


def _normalized_trace() -> dict[str, object]:
    return {
        "trace_id": "trace_r61",
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
            "action_type": "TAKE",
            "arguments": {"item": "item_1"},
            "accepted": True,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span",
            "authoritative_positive_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "item_1"},
                "witness_ref": "fact:holds:item_1",
                "event_index": 0,
            }],
        }],
    }


def _atomic_proposal() -> AtomicOccurrenceProposal:
    return AtomicOccurrenceProposal(
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


def _system(tmp_path, extractor_type, composite_builder):
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    system = object.__new__(AtomicSkillGraphSystem)
    system.config = {}
    system.normalizer = SimpleNamespace(
        build=lambda _trace: _normalized_trace(),
    )
    system.atomicizer = Atomicizer()
    system.skills = SkillRegistry(artifacts, database)
    system.tools = ToolRegistry(artifacts, database)
    system.graph = GraphStore(database, system.skills)
    system.aligner = Aligner(system.skills, system.tools)
    system.tool_compiler = ToolCompiler()
    system.composite_builder = composite_builder
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
    return system, database


def _trace_and_task():
    return (
        SimpleNamespace(metadata={}, runtime_plan={}),
        SimpleNamespace(task_id="task", context={}),
    )


def test_prepare_evolution_does_not_repair_valid_initial_e2(
    monkeypatch, tmp_path,
) -> None:
    proposal = SimpleNamespace(existing_edges=[], new_edges=[])

    class Extractor:
        repair_calls = 0
        e2_protocol_repair_count = 0

        def __init__(self, _session) -> None:
            pass

        def propose_atomics(self, *_args, **_kwargs):
            return [_atomic_proposal()]

        def propose_composite(self, *_args, **_kwargs):
            return proposal

        def repair_composite(self, *_args, **_kwargs):
            type(self).repair_calls += 1
            raise AssertionError("valid E2 must not trigger E2R")

    class AcceptingBuilder:
        calls = 0

        def validate_and_build(self, *_args, **_kwargs):
            type(self).calls += 1
            return SimpleNamespace(metadata={}, ref="skill://composite@1.0.0")

    monkeypatch.setattr(system_module, "ExtractorSession", Extractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    system, database = _system(tmp_path, Extractor, AcceptingBuilder())
    trace, task = _trace_and_task()

    prepared = system._prepare_evolution(trace, task)

    assert prepared.composite is not None
    assert Extractor.repair_calls == 0
    assert AcceptingBuilder.calls == 1
    quality = trace.metadata["extractor_quality"]
    assert quality["extractor_e2_repair_attempt_count"] == 0
    assert quality["extractor_e2_repair_success_count"] == 0
    assert quality["extractor_e2_repair_failure_count"] == 0
    database.close()


def test_failed_e2_repair_preserves_prepared_atomics(
    monkeypatch, tmp_path,
) -> None:
    initial = SimpleNamespace(existing_edges=[], new_edges=[])
    repaired = SimpleNamespace(existing_edges=[], new_edges=[])

    class Extractor:
        repair_calls = 0
        e2_protocol_repair_count = 0

        def __init__(self, _session) -> None:
            pass

        def propose_atomics(self, *_args, **_kwargs):
            return [_atomic_proposal()]

        def propose_composite(self, *_args, **_kwargs):
            return initial

        def repair_composite(self, proposal, rejection, *_args, **_kwargs):
            assert proposal is initial
            assert str(rejection) == "initial deterministic rejection"
            type(self).repair_calls += 1
            return repaired

    class RejectingBuilder:
        calls = 0

        def validate_and_build(self, proposal, *_args, **_kwargs):
            type(self).calls += 1
            if proposal is initial:
                raise ValueError("initial deterministic rejection")
            assert proposal is repaired
            raise ValueError("repair deterministic rejection")

    monkeypatch.setattr(system_module, "ExtractorSession", Extractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    system, database = _system(tmp_path, Extractor, RejectingBuilder())
    trace, task = _trace_and_task()

    prepared = system._prepare_evolution(trace, task)

    assert len(prepared.compiled) == 1
    assert prepared.composite is None
    assert Extractor.repair_calls == 1
    assert RejectingBuilder.calls == 2
    assert prepared.composite_rejection["error"] == (
        "repair deterministic rejection"
    )
    quality = trace.metadata["extractor_quality"]
    assert quality["extractor_e2_repair_attempt_count"] == 1
    assert quality["extractor_e2_repair_success_count"] == 0
    assert quality["extractor_e2_repair_failure_count"] == 1
    extraction = trace.metadata["extraction"]
    assert extraction["e2_repair_attempted"] is True
    assert extraction["e2_repair_applied"] is False
    assert extraction["e2_repair_error"] == (
        "repair deterministic rejection"
    )
    database.close()
