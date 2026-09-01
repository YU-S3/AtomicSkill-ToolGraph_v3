from __future__ import annotations

import json
from types import SimpleNamespace

from atomic_skillgraph.core.errors import FailureEnvelope, FailureLayer
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.failure_extraction_view import (
    FailureExtractionViewBuilder,
)
from atomic_skillgraph.traces.schema import (
    EnvironmentActionRecord,
    RuntimeSpan,
    TaskProgressRecord,
    TaskRecord,
    TraceRecord,
    ValidationRecord,
)


def _trace(action_count: int = 1) -> TraceRecord:
    trace = TraceRecord.create(
        TaskRecord(
            "failed-task", "alfworld", "inspect an object", "opaque", "sig",
        ),
        {"target_effects": []},
        {},
        {"source": "full_dynamic"},
    )
    trace.environment_actions = [
        EnvironmentActionRecord(
            f"action-{index}", index, "OPEN", {"object": f"drawer_{index}"},
            True, f"public-observation-{index}", False, False, index + 1,
            "span-runtime",
        )
        for index in range(action_count)
    ]
    trace.runtime_spans = [RuntimeSpan(
        "span-runtime", "cold_start_dynamic_continuation", "occurrence-a",
        0, action_count, None, True,
    )]
    trace.node_records = [SimpleNamespace(
        occurrence_id="occurrence-a", step_id="step-a",
    )]
    return trace


def _plan(expected_effects: list[dict] | None = None) -> dict:
    return {
        "plan_id": "cold-plan",
        "steps": [{
            "step_id": "step-a",
            "requirement_instance_ids": ["single::a"],
            "candidate_source": "unresolved",
            "candidate_ref": "",
            "execution_mode": "dynamic",
            "binding_specs": {},
            "repeat_role_bindings": {},
            "expected_effects": list(expected_effects or []),
        }],
        "control_sequence": ["step-a"],
        "data_edges": [],
        "dependency_edges": [],
        "requirement_coverage": {"single::a": ["step-a"]},
        "referenced_failure_experience_ids": [],
    }


def test_alignment_view_is_bounded_projection_not_interpretation() -> None:
    trace = _trace()
    raw_observation = "You open drawer 1. In it, you see nothing."
    trace.environment_actions[0].observation = raw_observation
    trace.agent_sessions = [SimpleNamespace(
        replay_messages="forbidden-session-payload" * 1000,
    )]
    trace.agent_turns = [SimpleNamespace(
        reasoning="forbidden-reasoning-payload" * 1000,
    )]
    trace.provider_requests = [SimpleNamespace(
        transport="forbidden-provider-payload" * 1000,
    )]
    trace.llm_usage = [{"forbidden_usage_payload": "x" * 10000}]
    trace.grounding_evidence_changes = [SimpleNamespace(
        raw_action_catalog="forbidden-action-catalog" * 1000,
    )]
    trace.failures = [FailureEnvelope(
        "failure-1", FailureLayer.RUNTIME_AGENT,
        "runtime_task_token_budget_exhausted", trace.task.task_id,
        trace.trace_id, "", "attempt", True,
    )]

    view = FailureExtractionViewBuilder(
        public_observation_char_limit=256,
    ).build_alignment(
        trace=trace,
        task_contract={"target_effects": []},
        requirement_expansion={"instances": []},
        cold_start_plan=_plan(),
        candidate_contract_views=[{
            "candidate_source": "unresolved",
            "candidate_ref": "",
            "contract": {"unresolved": True},
        }],
    )

    assert len(view.execution_events) == 1
    assert view.execution_events[0].bounded_public_observation == raw_observation
    encoded = json.dumps(to_primitive(view), ensure_ascii=False)
    for forbidden in (
        "agent_sessions", "agent_turns", "provider_requests", "llm_usage",
        "grounding_evidence_changes", "forbidden-session-payload",
        "forbidden-reasoning-payload", "forbidden-provider-payload",
        "forbidden_usage_payload", "forbidden-action-catalog",
        "target is not in drawer 1", "search another drawer",
        "this is a failed source location",
    ):
        assert forbidden not in encoded


def test_alignment_view_projects_every_event_and_bounds_public_observation() -> None:
    trace = _trace(action_count=100)
    trace.environment_actions[0].observation = (
        "You open drawer 1. In it, you see nothing.\r\n" + "z" * 100
    )
    trace.task_progress_records = [TaskProgressRecord(
        revision=1,
        source="task_reset",
        snapshot={
            "revision": 0,
            "progress_digest": "digest-0",
            "targets": [{
                "constraint_id": "target::0",
                "predicate": "object.at_location",
                "required_count": 1,
                "satisfied_count": 0,
                "remaining_count": 1,
                "distinct_by": "",
                "satisfied_witnesses": [{"object": "hidden_object_1"}],
                "shared_values": {"location": ["hidden_location_1"]},
                "used_distinct_values": ["hidden_object_1"],
            }],
            "unsatisfied_identity_constraints": [],
        },
    )]

    view = FailureExtractionViewBuilder(
        public_observation_char_limit=64,
    ).build_alignment(
        trace=trace,
        task_contract={"target_effects": []},
        requirement_expansion={},
        cold_start_plan=_plan(),
        candidate_contract_views=[],
    )

    assert len(view.execution_events) == 100
    observation = view.execution_events[0].bounded_public_observation
    assert len(observation) <= 64
    assert "\r" not in observation
    assert observation.startswith("You open drawer 1. In it, you see nothing.\n")
    encoded_progress = json.dumps(to_primitive(view.task_progress_deltas))
    assert "hidden_object_1" not in encoded_progress
    assert "hidden_location_1" not in encoded_progress
    assert view.execution_events[0].task_progress_before_digest == "digest-0"


def test_asset_view_contains_only_validated_candidate_span_events_and_effects() -> None:
    trace = _trace(action_count=100)
    expected_effect = {
        "predicate": "container.open",
        "args": {"container": {"kind": "skill_input", "source_role": "object"}},
        "cardinality": 1,
        "distinct_by": "",
    }
    trace.validations = [ValidationRecord(
        "occurrence-a",
        "atomic",
        {"passed": True, "witness_refs": ["effect:w1"]},
        43,
    )]
    builder = FailureExtractionViewBuilder()
    alignment_view = builder.build_alignment(
        trace=trace,
        task_contract={"target_effects": []},
        requirement_expansion={},
        cold_start_plan=_plan([expected_effect]),
        candidate_contract_views=[],
    )
    validated_alignment = {
        "alignment_id": "alignment",
        "step_alignments": [],
        "matched_prefix_step_ids": [],
        "first_unrecovered_divergence": {
            "kind": "budget_exhaustion",
            "step_id": "step-a",
            "event_index": 43,
            "summary": "the validated plan remains incomplete",
        },
        "remaining_requirement_instance_ids": ["single::a"],
        "candidate_progress_spans": [{
            "step_id": "step-a",
            "event_start": 40,
            "event_end": 43,
            "effect_witness_refs": ["effect:w1"],
        }],
    }

    view = builder.build_assets(
        trace=trace,
        alignment_view=alignment_view,
        validated_alignment=validated_alignment,
        task_contract={"target_effects": []},
    )

    assert len(view.candidate_progress_spans) == 1
    span = view.candidate_progress_spans[0]
    assert [event.event_index for event in span.accepted_events] == [40, 41, 42]
    assert list(span.authoritative_positive_effects) == [expected_effect]
    assert span.witness_refs == ("effect:w1",)
    encoded = json.dumps(to_primitive(view), ensure_ascii=False)
    assert "public-observation-39" not in encoded
    assert "public-observation-43" not in encoded
    for forbidden in (
        "agent_sessions", "provider_requests", "llm_usage", "reasoning",
        "raw_action_catalog",
    ):
        assert forbidden not in encoded


def test_asset_view_does_not_invent_effects_from_action_or_observation() -> None:
    trace = _trace()
    trace.environment_actions[0].action_type = "USE"
    trace.environment_actions[0].arguments = {"object": "desklamp_1"}
    trace.environment_actions[0].observation = "You turn on the desklamp 1."
    builder = FailureExtractionViewBuilder()
    alignment_view = builder.build_alignment(
        trace=trace,
        task_contract={"target_effects": []},
        requirement_expansion={},
        cold_start_plan=_plan(),
        candidate_contract_views=[],
    )
    view = builder.build_assets(
        trace=trace,
        alignment_view=alignment_view,
        validated_alignment={
            "matched_prefix_step_ids": [],
            "first_unrecovered_divergence": {},
            "remaining_requirement_instance_ids": [],
            "candidate_progress_spans": [{
                "step_id": "step-a", "event_start": 0, "event_end": 1,
                "effect_witness_refs": [],
            }],
        },
        task_contract={"target_effects": []},
    )

    assert view.candidate_progress_spans[0].authoritative_positive_effects == ()
    assert (
        view.candidate_progress_spans[0]
        .accepted_events[0].bounded_public_observation
        == "You turn on the desklamp 1."
    )


def test_asset_view_maps_same_predicate_progress_by_exact_target_ordinal() -> None:
    trace = _trace()
    trace.task_progress_records = [
        TaskProgressRecord(0, "before", {
            "revision": 0,
            "progress_digest": "before",
            "targets": [
                {
                    "constraint_id": "target::0::object.at_location",
                    "predicate": "object.at_location",
                    "required_count": 1,
                    "satisfied_count": 0,
                    "remaining_count": 1,
                    "distinct_by": "",
                },
                {
                    "constraint_id": "target::1::object.at_location",
                    "predicate": "object.at_location",
                    "required_count": 1,
                    "satisfied_count": 0,
                    "remaining_count": 1,
                    "distinct_by": "",
                },
            ],
            "unsatisfied_identity_constraints": [],
        }),
        TaskProgressRecord(1, "after", {
            "revision": 1,
            "progress_digest": "after",
            "targets": [
                {
                    "constraint_id": "target::0::object.at_location",
                    "predicate": "object.at_location",
                    "required_count": 1,
                    "satisfied_count": 1,
                    "remaining_count": 0,
                    "distinct_by": "",
                },
                {
                    "constraint_id": "target::1::object.at_location",
                    "predicate": "object.at_location",
                    "required_count": 1,
                    "satisfied_count": 0,
                    "remaining_count": 1,
                    "distinct_by": "",
                },
            ],
            "unsatisfied_identity_constraints": [],
        }),
    ]
    first = {
        "predicate": "object.at_location",
        "args": {"object": "apple", "location": "bowl"},
        "cardinality": 1,
        "distinct_by": "",
    }
    second = {
        "predicate": "object.at_location",
        "args": {"object": "mug", "location": "table"},
        "cardinality": 1,
        "distinct_by": "",
    }
    contract = {"target_effects": [first, second]}
    builder = FailureExtractionViewBuilder()
    alignment_view = builder.build_alignment(
        trace=trace,
        task_contract=contract,
        requirement_expansion={},
        cold_start_plan=_plan(),
        candidate_contract_views=[],
    )

    view = builder.build_assets(
        trace=trace,
        alignment_view=alignment_view,
        validated_alignment={
            "matched_prefix_step_ids": [],
            "first_unrecovered_divergence": {},
            "remaining_requirement_instance_ids": [],
            "candidate_progress_spans": [{
                "step_id": "step-a",
                "event_start": 0,
                "event_end": 1,
                "effect_witness_refs": [],
            }],
        },
        task_contract=contract,
    )

    assert list(
        view.candidate_progress_spans[0].authoritative_positive_effects
    ) == [first]
