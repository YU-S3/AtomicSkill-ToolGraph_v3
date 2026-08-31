"""Strict structured submissions transported only as native function calls.

DeepSeek Chat Completions does not provide the JSON-Schema response-format
transport used by the original implementation.  These definitions keep JSON
Schema as a local/native-tool contract while making the submitted ToolCall the
only machine-readable control channel.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..core.errors import AgentProtocolError, FailureLayer
from .protocol import (
    AgentSession,
    AgentTurn,
    NativeToolSpec,
    SchemaValidationError,
    validate_schema_instance,
)


NONEMPTY_STRING_SCHEMA: dict[str, Any] = {"type": "string", "minLength": 1}
SKILL_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["logical_id", "version"],
    "additionalProperties": False,
    "properties": {
        "logical_id": NONEMPTY_STRING_SCHEMA,
        "version": NONEMPTY_STRING_SCHEMA,
    },
}

BINDING_EXPRESSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kind"],
    "additionalProperties": False,
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "skill_input",
                "constant",
                "data_flow",
                "tool_output",
                "adapter_transform",
            ],
        },
        "source_role": {"type": "string"},
        "source_step": {"type": "string"},
        "constant": {},
        "transform_id": {"type": "string"},
    },
}

PREDICATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["predicate", "args"],
    "additionalProperties": False,
    "properties": {
        "predicate": NONEMPTY_STRING_SCHEMA,
        # Predicate role names are intentionally dynamic.  Their values remain
        # subject to deterministic semantic validation after schema admission.
        "args": {"type": "object"},
        "cardinality": {"type": "integer", "minimum": 1},
        "distinct_by": {"type": "string"},
    },
}

PARAMETER_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "semantic_type"],
    "additionalProperties": False,
    "properties": {
        "name": NONEMPTY_STRING_SCHEMA,
        "semantic_type": NONEMPTY_STRING_SCHEMA,
        "required": {"type": "boolean"},
        "runtime_resolvable": {"type": "boolean"},
        "required_resolution": {
            "type": "string",
            "enum": ["semantic", "concrete", "relation_verified"],
        },
        "description": {"type": "string"},
    },
}

CAPABILITY_REQUIREMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "requirement_id",
        "intent",
        "desired_effects",
        "expected_inputs",
        "expected_outputs",
        "precondition_hints",
        "semantic_variants",
        "required",
        "rationale",
    ],
    "additionalProperties": False,
    "properties": {
        "requirement_id": NONEMPTY_STRING_SCHEMA,
        "intent": NONEMPTY_STRING_SCHEMA,
        "desired_effects": {
            "type": "array",
            "minItems": 1,
            "items": PREDICATE_SCHEMA,
        },
        "expected_inputs": {"type": "array", "items": PARAMETER_SPEC_SCHEMA},
        "expected_outputs": {"type": "array", "items": PARAMETER_SPEC_SCHEMA},
        "precondition_hints": {"type": "array", "items": PREDICATE_SCHEMA},
        "semantic_variants": {
            "type": "array",
            "items": NONEMPTY_STRING_SCHEMA,
        },
        "required": {"type": "boolean"},
        "rationale": NONEMPTY_STRING_SCHEMA,
    },
}

PROPOSED_OCCURRENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "step_id",
        "occurrence_id",
        "node_ref",
        "requirement_instance_ids",
        "repeat_role_bindings",
        "binding_specs",
    ],
    "additionalProperties": False,
    "properties": {
        "step_id": NONEMPTY_STRING_SCHEMA,
        "occurrence_id": NONEMPTY_STRING_SCHEMA,
        "node_ref": {
            "oneOf": [NONEMPTY_STRING_SCHEMA, SKILL_REF_SCHEMA],
        },
        "requirement_instance_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": NONEMPTY_STRING_SCHEMA,
        },
        "repeat_role_bindings": {
            "type": "object",
            "additionalProperties": NONEMPTY_STRING_SCHEMA,
        },
        "binding_specs": {
            "type": "object",
            "additionalProperties": BINDING_EXPRESSION_SCHEMA,
        },
    },
}

PROPOSED_EDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "edge_id",
        "edge_type",
        "source_step",
        "target_step",
        "source_role",
        "target_role",
    ],
    "additionalProperties": False,
    "properties": {
        "edge_id": NONEMPTY_STRING_SCHEMA,
        "edge_type": {
            "type": "string",
            "enum": ["data_flow", "requires_skill"],
        },
        "source_step": NONEMPTY_STRING_SCHEMA,
        "target_step": NONEMPTY_STRING_SCHEMA,
        "source_role": {"type": "string"},
        "target_role": {"type": "string"},
        "origin": {
            "type": "string",
            "enum": ["planner_proposed", "existing_active", "extractor_proposed"],
        },
        "existing_edge_id": {"type": "string"},
    },
}

ATOMIC_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "phase_id",
        "intent",
        "event_start",
        "event_end",
        "input_roles",
        "output_roles",
        "preconditions",
        "effects",
        "rationale",
    ],
    "additionalProperties": False,
    "properties": {
        "phase_id": NONEMPTY_STRING_SCHEMA,
        "intent": NONEMPTY_STRING_SCHEMA,
        "event_start": {
            "type": "integer",
            "minimum": 0,
            "description": "Inclusive index of the first selected canonical action event.",
        },
        "event_end": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Exclusive index after the last selected event; a single event i uses [i,i+1)."
            ),
        },
        "input_roles": {
            "type": "object",
            "minProperties": 1,
            "description": (
                "Non-empty role-to-concrete-value bindings copied exactly from selected action arguments."
            ),
        },
        "output_roles": {
            "type": "object",
            "minProperties": 1,
            "description": (
                "Published role identities; every value must exactly repeat an input_roles value."
            ),
        },
        "preconditions": {
            "type": "array",
            "items": PREDICATE_SCHEMA,
            "description": "Only facts present in authoritative_before_state_facts.",
        },
        "effects": {
            "type": "array",
            "minItems": 1,
            "items": PREDICATE_SCHEMA,
            "description": (
                "Only authoritative positive effects or explicitly listed narrow terminal certificates "
                "of the selected accepted events."
            ),
        },
        "rationale": NONEMPTY_STRING_SCHEMA,
    },
}

COMPOSITE_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "control_sequence",
        "existing_edges",
        "new_edges",
        "summary",
        "guideline",
        "insight",
    ],
    "additionalProperties": False,
    "properties": {
        "control_sequence": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": NONEMPTY_STRING_SCHEMA,
        },
        "existing_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "new_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "summary": NONEMPTY_STRING_SCHEMA,
        "guideline": {"type": "object"},
        "insight": {"type": "object"},
    },
}

REPAIR_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["review_id", "decision", "rationale"],
    "properties": {
        "review_id": NONEMPTY_STRING_SCHEMA,
        "decision": {"type": "string", "enum": ["no_change", "propose"]},
        "rationale": NONEMPTY_STRING_SCHEMA,
    },
}


@dataclass(frozen=True)
class StructuredSubmission:
    value: dict[str, Any]
    call_id: str
    tool_name: str
    turn: AgentTurn


class StructuredSubmissionClient:
    """Request and acknowledge exactly one schema-validated submit ToolCall."""

    def request(
        self,
        session: AgentSession,
        *,
        prompt: str,
        tool_name: str,
        description: str,
        schema: dict[str, Any],
    ) -> StructuredSubmission:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("structured submission prompt must be non-empty")
        tool = NativeToolSpec(tool_name, description, copy.deepcopy(schema))
        turn = session.next_turn(prompt, tools=[tool])
        if len(turn.tool_calls) != 1:
            raise AgentProtocolError(
                "runtime_agent_schema_error",
                f"structured submission requires exactly one {tool_name!r} ToolCall",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        call = turn.tool_calls[0]
        if call.name != tool_name:
            raise AgentProtocolError(
                "runtime_agent_schema_error",
                f"structured submission called {call.name!r}, expected {tool_name!r}",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        try:
            validate_schema_instance(call.arguments, schema)
        except SchemaValidationError as exc:
            raise AgentProtocolError(
                "runtime_agent_schema_error",
                f"structured submission failed local schema validation: {exc}",
                layer=FailureLayer.RUNTIME_AGENT,
            ) from exc
        session.acknowledge_tool_result(
            call.call_id,
            {"accepted": True, "submission": tool_name},
        )
        return StructuredSubmission(
            value=copy.deepcopy(call.arguments),
            call_id=call.call_id,
            tool_name=call.name,
            turn=turn,
        )


__all__ = [
    "ATOMIC_EXTRACTION_SCHEMA",
    "BINDING_EXPRESSION_SCHEMA",
    "CAPABILITY_REQUIREMENT_SCHEMA",
    "COMPOSITE_EXTRACTION_SCHEMA",
    "PARAMETER_SPEC_SCHEMA",
    "PREDICATE_SCHEMA",
    "PROPOSED_EDGE_SCHEMA",
    "PROPOSED_OCCURRENCE_SCHEMA",
    "REPAIR_PROPOSAL_SCHEMA",
    "SKILL_REF_SCHEMA",
    "StructuredSubmission",
    "StructuredSubmissionClient",
]
