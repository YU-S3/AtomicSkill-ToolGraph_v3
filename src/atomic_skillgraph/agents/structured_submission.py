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
        "effect_domain": {"type": "string", "enum": ["world", "evidence"]},
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
        "support_event_ids",
        "input_roles",
        "input_provenance_refs",
        "output_roles",
        "output_derivations",
        "preconditions",
        "precondition_witness_refs",
        "effects",
        "effect_witness_refs",
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
        "support_event_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
            "description": (
                "Explicit accepted events that actually support this Atomic. "
                "They must lie within the event_start..event_end evidence envelope and "
                "may be non-contiguous."
            ),
        },
        "shared_precondition_event_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
            "description": (
                "Selected support events used only as shared prerequisite "
                "context. Code permits overlap only when no two independent "
                "Atomic Effects claim the same event."
            ),
        },
        "precondition_witness_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "effect_witness_refs": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "ordering_constraints": {
            "type": "array",
            "items": {"type": "object"},
        },
        "input_provenance_refs": {
            "type": "object",
            "minProperties": 1,
            "propertyNames": {"type": "string", "minLength": 1},
            "additionalProperties": NONEMPTY_STRING_SCHEMA,
            "description": (
                "One supplied code-authoritative boundary input reference for "
                "every input_roles key. Code requires the key sets to match."
            ),
        },
        "output_derivations": {
            "type": "object",
            "minProperties": 1,
            "propertyNames": {"type": "string", "minLength": 1},
            "additionalProperties": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["kind", "input_role"],
                        "additionalProperties": False,
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["input_identity"],
                            },
                            "input_role": NONEMPTY_STRING_SCHEMA,
                        },
                    },
                    {
                        "type": "object",
                        "required": [
                            "kind", "predicate", "argument_role",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["effect_witness"],
                            },
                            "predicate": NONEMPTY_STRING_SCHEMA,
                            "argument_role": NONEMPTY_STRING_SCHEMA,
                        },
                    },
                ],
            },
            "description": (
                "Exactly one explicit input_identity or effect_witness "
                "derivation for every output_roles key. Code requires the "
                "key sets to match."
            ),
        },
        "input_roles": {
            "type": "object",
            "minProperties": 1,
            "description": (
                "Non-empty role-to-concrete-value bindings backed exactly by "
                "the supplied boundary_authorities.inputs references."
            ),
        },
        "output_roles": {
            "type": "object",
            "minProperties": 1,
            "description": (
                "Published role identities; every value must have exactly one "
                "input_identity or effect_witness derivation."
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
        "selected_existing_edge_ids",
        "selected_new_edge_candidate_ids",
        "summary",
        "guideline",
        "insight",
    ],
    "additionalProperties": False,
    "properties": {
        "selected_existing_edge_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": NONEMPTY_STRING_SCHEMA,
        },
        "selected_new_edge_candidate_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": NONEMPTY_STRING_SCHEMA,
        },
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


TOOL_IR_PROGRAM_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["node_id", "op"],
    "additionalProperties": False,
    "properties": {
        "node_id": NONEMPTY_STRING_SCHEMA,
        "op": {
            "type": "string",
            "enum": ["ACTION", "IF", "FOR_EACH", "STOP_WHEN", "RETURN"],
        },
        "action_type": {"type": "string"},
        "argument_mapping": {
            "type": "object",
            "additionalProperties": BINDING_EXPRESSION_SCHEMA,
        },
        "condition": {"type": "object"},
        "then_branch": {"type": "array", "items": {"type": "object"}},
        "else_branch": {"type": "array", "items": {"type": "object"}},
        "collection_source": {"type": "object"},
        "iteration_variable": {"type": "string"},
        "body": {"type": "array", "items": {"type": "object"}},
        "max_iterations": {"type": "integer", "minimum": 1},
        "output_sources": {"type": "object"},
        "expected_effects": {"type": "array", "items": PREDICATE_SCHEMA},
    },
}

TOOL_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "proposal_version", "decision", "summary", "atomic_ref", "inputs",
        "outputs", "program", "max_actions", "final_effects",
        "evidence_outputs", "path_expectations", "rationale",
    ],
    "additionalProperties": False,
    "properties": {
        "proposal_version": NONEMPTY_STRING_SCHEMA,
        "decision": {"type": "string", "enum": ["create", "no_tool"]},
        "summary": NONEMPTY_STRING_SCHEMA,
        "atomic_ref": NONEMPTY_STRING_SCHEMA,
        "inputs": {"type": "array", "items": PARAMETER_SPEC_SCHEMA},
        "outputs": {"type": "array", "items": PARAMETER_SPEC_SCHEMA},
        "program": {"type": "array", "items": TOOL_IR_PROGRAM_NODE_SCHEMA},
        "max_actions": {"type": "integer", "minimum": 1},
        "final_effects": {"type": "array", "items": PREDICATE_SCHEMA},
        "evidence_outputs": {"type": "array", "items": {"type": "object"}},
        "path_expectations": {"type": "array", "items": {"type": "object"}},
        "rationale": NONEMPTY_STRING_SCHEMA,
    },
}

RUNTIME_AUTOMATION_ATOMIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "draft_id", "intent", "inputs", "outputs", "preconditions",
        "effects", "rationale", "source_occurrence_id",
        "input_binding_specs",
    ],
    "additionalProperties": False,
    "properties": {
        "draft_id": NONEMPTY_STRING_SCHEMA,
        "intent": NONEMPTY_STRING_SCHEMA,
        "inputs": {"type": "array", "items": PARAMETER_SPEC_SCHEMA},
        "outputs": {"type": "array", "items": PARAMETER_SPEC_SCHEMA},
        "preconditions": {"type": "array", "items": PREDICATE_SCHEMA},
        "effects": {
            "type": "array",
            "minItems": 1,
            "items": PREDICATE_SCHEMA,
        },
        "rationale": NONEMPTY_STRING_SCHEMA,
        "source_occurrence_id": NONEMPTY_STRING_SCHEMA,
        "input_binding_specs": {
            "type": "object",
            "additionalProperties": {"type": "object"},
        },
    },
}


__all__ = [
    "ATOMIC_EXTRACTION_SCHEMA",
    "RUNTIME_AUTOMATION_ATOMIC_SCHEMA",
    "TOOL_IR_PROGRAM_NODE_SCHEMA",
    "TOOL_PROPOSAL_SCHEMA",
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
