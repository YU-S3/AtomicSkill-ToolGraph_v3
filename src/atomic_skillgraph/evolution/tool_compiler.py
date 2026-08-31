"""Compile necessary accepted action slices into parameterized Primitive IR assets."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from ..core.bindings import (
    BindingExpression, BindingExprKind, GroundingConstraint, GroundingConstraintKind,
    ToolBinding,
)
from ..core.contracts import AbstractAtomicSkill, ImplementationAtom, ToolAsset
from ..core.refs import SkillRef, ToolRef
from ..core.status import SkillStatus, ToolStatus
from .atomicizer import CanonicalAtomicOccurrence
from .portability import CanonicalCapabilityLabel


@dataclass
class CompiledKnowledge:
    occurrence: CanonicalAtomicOccurrence
    atomic: AbstractAtomicSkill
    tool: ToolAsset
    implementation: ImplementationAtom


def rewrite_capability_labels(
    compiled: CompiledKnowledge,
    label: CanonicalCapabilityLabel,
) -> CompiledKnowledge:
    """Return a pre-registration bundle with one portable semantic label."""

    atomic = replace(
        compiled.atomic,
        summary=label.display_summary,
        guideline={
            **dict(compiled.atomic.guideline or {}),
            "canonical_intent": label.canonical_intent,
        },
        metadata={
            **dict(compiled.atomic.metadata or {}),
            "canonical_intent": label.canonical_intent,
            "canonical_label_source": label.source,
        },
    )
    tool = replace(
        compiled.tool,
        summary=f"Primitive executable for {label.display_summary}",
        metadata={
            **dict(compiled.tool.metadata or {}),
            "canonical_intent": label.canonical_intent,
            "semantic_description": label.display_summary,
            "canonical_label_source": label.source,
        },
    )
    implementation = replace(
        compiled.implementation,
        metadata={
            **dict(compiled.implementation.metadata or {}),
            "canonical_intent": label.canonical_intent,
            "semantic_description": label.display_summary,
            "canonical_label_source": label.source,
        },
    )
    return CompiledKnowledge(
        replace(compiled.occurrence, intent=label.canonical_intent),
        atomic,
        tool,
        implementation,
    )


def _role_for_value(value: Any, bindings: dict[str, Any]) -> str | None:
    matches = [role for role, bound in bindings.items() if bound == value]
    return matches[0] if len(matches) == 1 else None


class ToolCompiler:
    def compile(self, occurrences: list[CanonicalAtomicOccurrence]) -> list[CompiledKnowledge]:
        result: list[CompiledKnowledge] = []
        for occurrence in occurrences:
            output_identity: list[dict[str, str]] = []
            for output_role, value in sorted(occurrence.output_bindings.items()):
                input_role = _role_for_value(value, occurrence.input_bindings)
                if input_role is None:
                    raise ValueError(
                        f"Atomic output {output_role} cannot be grounded in a reusable input role"
                    )
                output_identity.append({
                    "output_role": output_role,
                    "input_role": input_role,
                })
            atomic = AbstractAtomicSkill(
                occurrence.proposed_ref, occurrence.intent, occurrence.input_specs, occurrence.output_specs,
                occurrence.preconditions, occurrence.effects,
                {
                    "validator_id": "harness_atomic_effect",
                    "identity_strict": True,
                    "output_identity": output_identity,
                }, [],
                {"steps": [item["action_type"] for item in occurrence.action_events]},
                {"source_trace_ids": [occurrence.source_trace_id]}, SkillStatus.DRAFT,
            )
            primitive_steps = []
            # The Tool signature carries the whole Atomic input context, not
            # only values syntactically present in the terminal primitive.
            # ALFWorld USE-lamp, for example, also requires the held target
            # object to publish/validate ``object.observed_with``.
            tool_properties: dict[str, Any] = {
                role: {"type": "string"}
                for role in occurrence.input_bindings
            }
            for event in occurrence.action_events:
                mapping: dict[str, BindingExpression] = {}
                for argument, value in event.get("arguments", {}).items():
                    role = _role_for_value(value, occurrence.input_bindings)
                    if role is None:
                        # Stable context constants such as LOOK have no args;
                        # concrete episode entities are never embedded.
                        if isinstance(value, str) and re.search(r"(?:_|\s)\d+$", value):
                            raise ValueError(f"cannot parameterize concrete action argument {argument}={value}")
                        mapping[argument] = BindingExpression(BindingExprKind.CONSTANT, constant=value)
                    else:
                        mapping[argument] = BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)
                        tool_properties[role] = {"type": "string"}
                primitive_steps.append({"action_type": event["action_type"], "argument_mapping": mapping})
            tool_id = f"tool_{occurrence.proposed_ref.logical_id.removeprefix('atomic_')}"
            tool_ref = ToolRef(tool_id, "1.0.0")
            tool_output_mapping: dict[str, BindingExpression] = {}
            implementation_output_mapping: dict[str, BindingExpression] = {}
            for output_role, value in occurrence.output_bindings.items():
                input_role = _role_for_value(value, occurrence.input_bindings)
                # The same relation was validated before constructing the
                # Atomic validator contract above.
                assert input_role is not None
                tool_output_mapping[output_role] = BindingExpression(BindingExprKind.SKILL_INPUT, source_role=input_role)
                implementation_output_mapping[output_role] = BindingExpression(
                    BindingExprKind.TOOL_OUTPUT, source_role=output_role, source_step="primary",
                )
            tool = ToolAsset(
                tool_ref, f"Primitive implementation of {occurrence.intent}",
                {"type": "object", "properties": tool_properties, "required": sorted(tool_properties)},
                {"output_schema": {
                    "type": "object", "properties": {role: {"type": "string"} for role in tool_output_mapping},
                    "required": sorted(tool_output_mapping), "additionalProperties": False,
                }},
                "primitive_ir", {"steps": primitive_steps, "output_mapping": tool_output_mapping},
                [{"kind": "source_replay", "trace_id": occurrence.source_trace_id,
                  "event_range": [occurrence.event_start, occurrence.event_end],
                  "bindings": dict(occurrence.input_bindings),
                  "source_task": dict(occurrence.source_task),
                  "prefix": [
                      {"action_type": event["action_type"], "arguments": dict(event.get("arguments", {}))}
                      for event in occurrence.prefix_events if event.get("accepted")
                  ],
                  "effects": list(occurrence.effects)}],
                {"reviewed": True, "allowed_action_types": [item["action_type"] for item in primitive_steps]},
                {"source_trace_id": occurrence.source_trace_id, "occurrence_id": occurrence.occurrence_id},
                {}, ToolStatus.ADMISSION_PENDING,
            )
            tool_binding_mapping = {
                role: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)
                for role in tool_properties
            }
            first = primitive_steps[0]
            constraints = []
            if first["argument_mapping"]:
                constraints.append(GroundingConstraint(
                    "entry_affordance", GroundingConstraintKind.HARNESS_AFFORDANCE,
                    action_type=first["action_type"], argument_mapping=dict(first["argument_mapping"]),
                    required_resolution="relation_verified" if len(first["argument_mapping"]) > 1 else "concrete",
                ))
            implementation = ImplementationAtom(
                SkillRef(f"impl_{occurrence.proposed_ref.logical_id.removeprefix('atomic_')}", "1.0.0"),
                atomic.ref, [ToolBinding(tool.ref, "primary", tool_binding_mapping, 0)], constraints,
                {"mode": "serial", "output_mapping": implementation_output_mapping},
                {"harness_profiles": ["alfworld_v3", "fake_v3"]}, {}, SkillStatus.DRAFT,
            )
            result.append(CompiledKnowledge(occurrence, atomic, tool, implementation))
        return result
