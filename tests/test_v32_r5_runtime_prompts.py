from __future__ import annotations

import copy
import json

from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.agents.runtime_policy_projection import (
    unpack_downstream_context,
)
from atomic_skillgraph.agents.runtime_prompt_texts import (
    ATOMIC_AUTHORITY,
    DYNAMIC_ONLY,
    DYNAMIC_PROMPT,
    PREPARATION_ONLY,
    PREPARATION_PROMPT,
    SEARCH_POLICY,
    SEEDED_ONLY,
    SEEDED_PROMPT,
)


SEPARATOR = "\n\nPOLICY_CONTEXT_JSON\n"


def _split(rendered: str) -> tuple[str, dict]:
    instruction, payload = rendered.split(SEPARATOR, 1)
    return instruction, json.loads(payload)


def _atomic() -> dict:
    return {
        "summary": "place an object",
        "inputs": [{"name": "object"}, {"name": "destination"}],
        "outputs": [{"name": "placed_object"}],
        "preconditions": [],
        "effects": [{"predicate": "object.at_location"}],
        "guideline": {"ordering": ["take", "move"]},
    }


def _downstream() -> dict:
    shared = {
        "consumer_summary": (
            "Inspect the placed object using the required downstream source "
            "without changing its task identity."
        ),
        "consumer_preconditions": [{
            "predicate": "object.at_location",
            "args": {"object": "object", "location": "source"},
        }],
        "consumer_effects": [{
            "predicate": "object.observed",
            "args": {"object": "object"},
        }],
        "consumer_known_semantic_anchors": {
            "object": {"value": "apple", "source": "task"}
        },
    }
    obligations = []
    for edge_id, producer_role, input_role in (
        ("e-object", "placed_object", "object"),
        ("e-source", "place_location", "source"),
    ):
        obligations.append({
            "producer_step": "place",
            "producer_output_role": producer_role,
            "edge_id": edge_id,
            "consumer_step": "observe",
            "consumer_input_role": input_role,
            "consumer_input_contract": {
                "name": input_role,
                "semantic_type": "entity",
                "description": "Required downstream entity input.",
            },
            **copy.deepcopy(shared),
        })
    return {
        "current_step": "place",
        "output_obligations": obligations,
        "remaining_method_outline": [
            {"step_id": "observe", "summary": "inspect placed object"}
        ],
    }


def test_prompt_constants_are_single_source_compositions_with_shared_search_policy() -> None:
    assert PREPARATION_PROMPT == "\n\n".join(
        (ATOMIC_AUTHORITY, PREPARATION_ONLY, SEARCH_POLICY)
    )
    assert SEEDED_PROMPT == "\n\n".join(
        (ATOMIC_AUTHORITY, SEEDED_ONLY, SEARCH_POLICY)
    )
    assert DYNAMIC_PROMPT == "\n\n".join((DYNAMIC_ONLY, SEARCH_POLICY))
    for prompt in (PREPARATION_PROMPT, SEEDED_PROMPT, DYNAMIC_PROMPT):
        assert prompt.count(SEARCH_POLICY) == 1
        assert "Never invent or reuse a superseded action_id." in prompt
        assert "not an automatic action policy or a new stopping rule" in prompt
        assert "Never invent an absence fact" in prompt


def test_prompt_permissions_and_soft_priorities_remain_entry_specific() -> None:
    assert "prefer validate_current_atomic over redoing" in PREPARATION_PROMPT
    assert "learned_invocation_ready is true" in PREPARATION_PROMPT
    assert "Use invoke_support_atomic only if offered" in PREPARATION_PROMPT
    assert "propose_runtime_automation_atomic only if offered" in PREPARATION_PROMPT
    assert "This preference never bypasses preflight or validation" in (
        PREPARATION_PROMPT
    )

    assert "fresh Seeded session without a failed Tool body" in SEEDED_PROMPT
    assert "absence of an implementation is not task failure" in SEEDED_PROMPT
    assert "Do not invent a learned-tool name" in SEEDED_PROMPT
    assert "invoke_support_atomic" not in SEEDED_PROMPT

    assert "Solve the whole task using the native tools actually offered" in (
        DYNAMIC_PROMPT
    )
    assert "Do not invent validate_current_atomic or learned invocations" in (
        DYNAMIC_PROMPT
    )
    assert "ColdStart continuation context" in DYNAMIC_PROMPT
    assert "invoke_support_atomic" not in DYNAMIC_PROMPT
    assert ATOMIC_AUTHORITY not in DYNAMIC_PROMPT


def test_preparation_prompt_projects_initial_payload_and_returns_full_audit() -> None:
    downstream = _downstream()
    state = {
        "current_atomic": _atomic(),
        "semantic_anchors": {"object": {"value": "apple"}},
        "confirmed_bindings": {"object": "apple_1"},
        "candidate_bindings": {"destination": "countertop_1"},
        "missing_bindings": [],
        "invalidated_bindings": {},
        "preconditions": [],
        "effect_witness_status": {},
        "learned_invocation_ready": True,
        "blocking_reasons": [],
        "downstream_obligations": downstream,
        "remaining_budget": {"remaining_node_actions": 35},
    }
    support = [{
        "atomic_ref": "atomic:support",
        "score": 0.75,
        "supplied_roles": ["entity"],
        "output_roles": ["entity"],
        "effect_predicates": ["object.visible"],
        "role_mappings": [{
            "producer_role": "entity",
            "consumer_role": "object",
        }],
        "diagnostics": {"rank_reason": "semantic match"},
    }]
    original_state = copy.deepcopy(state)
    original_support = copy.deepcopy(support)
    audit: dict = {}

    rendered = ContextBuilder().runtime_node(
        task_goal="place apple then observe it",
        atomic_contract=_atomic(),
        task_semantic_context={"object": "apple"},
        current_occurrence_semantic_anchors={"object": "apple"},
        execution_ready_bindings={"object": "apple_1"},
        missing_or_insufficient_bindings=[],
        observation="holding apple_1",
        action_catalog=[{
            "action_id": "r004_a001",
            "action_type": "PUT",
            "arguments": {
                "object": "apple_1",
                "destination": "countertop_1",
            },
            "revision": 4,
        }],
        relevant_action_history=[{
            "action_type": "TAKE",
            "arguments": {"object": "apple_1"},
            "accepted": True,
        }],
        remaining_budget={"remaining_node_actions": 35},
        implementation_invocations=[{
            "name": "invoke_impl_example",
            "description": "place an object",
            "input_schema": {"type": "object", "properties": {}},
        }],
        downstream_plan_context=downstream,
        current_state_snapshot=state,
        support_atomic_candidates=support,
        projection_audit=audit,
    )
    instruction, payload = _split(rendered)

    assert instruction == PREPARATION_PROMPT
    assert state == original_state
    assert support == original_support
    assert "diagnostics" not in payload["support_atomic_candidates"][0]
    assert payload["support_atomic_candidates"][0]["role_mappings"] == (
        support[0]["role_mappings"]
    )
    assert unpack_downstream_context(
        payload["current_state_snapshot"]["downstream_obligations"]
    ) == downstream
    assert payload["current_action_catalog"]["revision"] == 4
    assert payload["current_action_catalog"]["actions"][0] == {
        "action_id": "r004_a001",
        "action_type": "PUT",
        "arguments": {
            "object": "apple_1",
            "destination": "countertop_1",
        },
    }
    assert len(payload["recent_accepted_actions"]) == 1
    assert audit["projection_version"] == "v3.2-r5"
    assert audit["removed_fields"] == [
        "support_atomic_candidates[0].diagnostics"
    ]
    assert audit["raw_support_candidates"] == support
    assert audit["raw_downstream_obligations"] == downstream
    assert audit["downstream_projection"] == "deduplicated"


def test_seeded_and_dynamic_use_exact_replacement_prefixes_and_emit_audits() -> None:
    seeded_audit: dict = {}
    seeded = ContextBuilder().seeded_node(
        task_goal="place apple",
        atomic_contract=_atomic(),
        observation="apple is visible",
        action_catalog=[],
        relevant_action_history=[],
        remaining_budget={"remaining_node_actions": 35},
        projection_audit=seeded_audit,
    )
    seeded_instruction, seeded_payload = _split(seeded)
    assert seeded_instruction == SEEDED_PROMPT
    assert seeded_payload["current_state_snapshot"]["current_atomic"][
        "guideline"
    ] == {"ordering": ["take", "move"]}
    assert seeded_audit["projection_version"] == "v3.2-r5"

    dynamic_audit: dict = {}
    dynamic = ContextBuilder().dynamic_task(
        task_goal="place apple",
        observation="apple is visible",
        action_catalog=[{
            "action_id": "r000_a001",
            "action_type": "TAKE",
            "arguments": {"object": "apple_1"},
            "revision": 0,
        }],
        relevant_action_history=[],
        remaining_budget={"remaining_global_actions": 100},
        task_progress={"remaining_requirements": ["place"]},
        rescue_method_guidance={"conflict_code": "public_conflict"},
        projection_audit=dynamic_audit,
    )
    dynamic_instruction, dynamic_payload = _split(dynamic)
    assert dynamic_instruction == DYNAMIC_PROMPT
    assert dynamic_payload["rescue_method_guidance"] == {
        "conflict_code": "public_conflict"
    }
    assert dynamic_payload["current_action_catalog"]["actions"][0][
        "action_id"
    ] == "r000_a001"
    assert dynamic_audit["projection_version"] == "v3.2-r5"
    assert dynamic_audit["downstream_projection"] == "not_present"
    assert dynamic_audit["before_payload_sha256"] == (
        dynamic_audit["after_payload_sha256"]
    )
