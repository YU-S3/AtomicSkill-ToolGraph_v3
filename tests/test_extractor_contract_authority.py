from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
from atomic_skillgraph.evolution.extraction_authority import (
    contract_coverage_report,
    extraction_coverage_authority,
)
from atomic_skillgraph.evolution.extractor_session import ExtractionContentError
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher


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
        "runtime_spans": [],
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


def test_e1_incomplete_coverage_stops_before_e2_and_compilation(
    monkeypatch,
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

    class FakeExtractor:
        e2_called = False

        def __init__(self, _session) -> None:
            pass

        def propose_atomics(self, *_args, **_kwargs):
            return [proposal]

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
    system.skills = object()
    system._extractor_session = lambda _task_id: object()
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(target_effects=[
            SemanticPredicate("state.q", {"item": "item_1"}),
        ]),
        contract_matcher=lambda: ExactContractMatcher(),
    )
    system.tool_compiler = SimpleNamespace(
        compile=lambda *_args: (_ for _ in ()).throw(
            AssertionError("Tool compilation must not run")
        )
    )
    trace = SimpleNamespace(metadata={})
    task = SimpleNamespace(task_id="task", context={})

    with pytest.raises(ExtractionContentError) as caught:
        system._prepare_evolution(trace, task)

    assert caught.value.error_code == (
        "extractor_e1_task_contract_coverage_incomplete"
    )
    assert FakeExtractor.e2_called is False
    assert trace.metadata["extractor_contract_coverage"]["passed"] is False
