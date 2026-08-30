from __future__ import annotations

import inspect
import re

import pytest

from atomic_skillgraph.agents.protocol import (
    SchemaValidationError,
    validate_schema_instance,
)
from atomic_skillgraph.agents.structured_submission import ATOMIC_EXTRACTION_SCHEMA
from atomic_skillgraph.core.contracts import SemanticPredicate, TaskContract
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution import atomicizer as atomicizer_module
from atomic_skillgraph.evolution.atomicizer import AtomicBoundaryProposal, Atomicizer
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.harness.protocol import build_transition_certificate
from atomic_skillgraph.traces.schema import EnvironmentActionRecord, TaskRecord, TraceRecord


def test_harness_transition_certificate_is_atomicizer_only_authority() -> None:
    before = {
        "facts": [{
            "predicate": "resource.available",
            "args": {"resource": "sample_1"},
        }],
    }
    after = {
        "facts": [
            *before["facts"],
            {
                "predicate": "resource.ready",
                "args": {"resource": "sample_1"},
                "cardinality": 2,
                "distinct_by": "resource",
            },
        ],
    }
    required_identity = {
        ("resource.available", (("resource", "sample_1"),)),
    }
    certificate = build_transition_certificate(
        action_id="r000_a001",
        revision_before=0,
        revision_after=1,
        action_type="CUSTOM_TRANSITION",
        arguments={"resource": "sample_1"},
        before_snapshot=before,
        after_snapshot=after,
        accepted=True,
        required_fact_identities=required_identity,
        evidence_refs=("validator:transition:1",),
    )
    trace = TraceRecord.create(
        TaskRecord(
            "task-certificate",
            "generic",
            "make the selected resource ready",
            "generic_transition",
            "task-signature",
        ),
        to_primitive(TaskContract(target_effects=[SemanticPredicate(
            "resource.ready", {"resource": "sample_1"},
        )])),
        {},
        {},
    )
    trace.benchmark_success = True
    trace.environment_actions = [EnvironmentActionRecord(
        "r000_a001",
        0,
        "CUSTOM_TRANSITION",
        {"resource": "sample_1"},
        True,
        "Untrusted observation prose claims an unrelated result.",
        True,
        True,
        1,
        "span-1",
        to_primitive(certificate),
    )]

    normalized = TraceNormalizer().build(trace)
    projected = normalized["actions"][0]["transition_certificate"]
    effect_ref = projected["positive_effects"][0]["fact_ref"]
    required_ref = projected["required_facts"][0]["fact_ref"]
    occurrences = Atomicizer().validate_and_canonicalize([
        AtomicBoundaryProposal(
            phase_id="make_ready",
            intent="establish readiness",
            event_start=0,
            event_end_exclusive=1,
            selected_effect_refs=[effect_ref],
            selected_precondition_refs=[required_ref],
            output_role_mapping={
                "ready_resource": f"fact:{effect_ref}:resource",
            },
            rationale="The referenced validator effect establishes the result.",
        ),
    ], normalized)

    assert [effect.predicate for effect in occurrences[0].effects] == ["resource.ready"]
    assert occurrences[0].effects[0].cardinality == 2
    assert occurrences[0].effects[0].distinct_by == "resource"
    assert [item.predicate for item in occurrences[0].preconditions] == [
        "resource.available",
    ]
    assert "unrelated" not in repr(to_primitive(occurrences[0])).casefold()


def test_generic_atomicizer_contains_no_alfworld_action_names() -> None:
    source = inspect.getsource(atomicizer_module)
    forbidden_actions = (
        "TAKE",
        "PUT",
        "HEAT",
        "COOL",
        "CLEAN",
        "USE",
        "LOOK",
        "INVENTORY",
    )
    for action_name in forbidden_actions:
        assert re.search(rf"\b{action_name}\b", source) is None
    for reducer_name in (
        "_ACTION_EFFECTS",
        "reduce_action_state",
        "_TERMINAL_CONTEXT_EFFECT_ACTIONS",
        "terminal_context_effect_certificates",
    ):
        assert reducer_name not in source


def test_e1_proposal_references_effect_ids_instead_of_copying_facts() -> None:
    properties = set(ATOMIC_EXTRACTION_SCHEMA["properties"])
    assert properties == {
        "phase_id",
        "intent",
        "event_start",
        "event_end_exclusive",
        "selected_effect_refs",
        "selected_precondition_refs",
        "output_role_mapping",
        "rationale",
    }
    reference_only = {
        "phase_id": "phase-1",
        "intent": "establish the certified result",
        "event_start": 0,
        "event_end_exclusive": 1,
        "selected_effect_refs": ["transition:a:effect:1"],
        "selected_precondition_refs": [],
        "output_role_mapping": {
            "result": "fact:transition:a:effect:1:resource",
        },
        "rationale": "The boundary cites the validator-issued fact.",
    }
    validate_schema_instance(reference_only, ATOMIC_EXTRACTION_SCHEMA)

    copied_fact = {
        **reference_only,
        "effects": [{
            "predicate": "resource.ready",
            "args": {"resource": "sample_1"},
        }],
    }
    with pytest.raises(SchemaValidationError):
        validate_schema_instance(copied_fact, ATOMIC_EXTRACTION_SCHEMA)


def test_admissible_action_does_not_depend_on_nothing_happens_text() -> None:
    class _StepEnvironment:
        def step(self, actions: list[str]):
            assert actions == ["look"]
            return (
                ["Nothing happens."],
                [0.0],
                [False],
                {"won": [False], "admissible_commands": [["look"]]},
            )

    adapter = AlfWorldAdapter()
    adapter._env = _StepEnvironment()
    admitted = adapter._catalog.replace(["look"], adapter._revision)[0]

    result = adapter.execute_action(admitted.action_id, admitted.revision)

    assert result.accepted is True
    assert result.transition_certificate is not None
    assert result.transition_certificate.accepted is True
    assert result.transition_certificate.state_changed is False
