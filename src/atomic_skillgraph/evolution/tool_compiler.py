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
from ..tooling.proposal import ToolProposal, ToolProvenance
from .atomicizer import CanonicalAtomicOccurrence
from .portability import CanonicalCapabilityLabel


@dataclass
class CompiledKnowledge:
    occurrence: CanonicalAtomicOccurrence
    atomic: AbstractAtomicSkill
    tool: ToolAsset | None
    implementation: ImplementationAtom | None


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
    tool = None
    if compiled.tool is not None:
        tool = replace(
            compiled.tool,
            summary=f"Tool executable for {label.display_summary}",
            metadata={
                **dict(compiled.tool.metadata or {}),
                "canonical_intent": label.canonical_intent,
                "semantic_description": label.display_summary,
                "canonical_label_source": label.source,
            },
        )
    implementation = None
    if compiled.implementation is not None:
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

    def compile_proposal(
        self,
        occurrence: CanonicalAtomicOccurrence,
        atomic: AbstractAtomicSkill,
        proposal: ToolProposal,
        provenance: ToolProvenance,
    ) -> CompiledKnowledge:
        """Compile an Agent-authored ToolProposal into ToolAsset/ImplementationAtom.

        The compiler never chooses which actions enter the Tool; that authority
        belongs to ToolBuilder.  It only normalizes the already-validated IR into
        the persistent artifact envelope.
        """

        if proposal.decision == "no_tool":
            return CompiledKnowledge(occurrence, atomic, None, None)
        program = proposal.program
        if not program:
            raise ValueError("ToolProposal cannot compile an empty Tool IR program")
        action_nodes = [
            node for node in program
            if str(node.get("op", "")) == "ACTION"
        ]
        tool_properties = {
            str(item.name): {"type": "string"}
            for item in proposal.inputs or atomic.inputs
        }
        output_properties = {
            str(item.name): {"type": "string"}
            for item in proposal.outputs or atomic.outputs
        }
        output_mapping: dict[str, Any] = {}
        implementation_output_mapping: dict[str, Any] = {}
        for output in proposal.outputs or atomic.outputs:
            role = str(output.name)
            source = next(
                (
                    item.get("source", "tool_input")
                    for item in proposal.evidence_outputs
                    if str(item.get("role", "")) == role
                ),
                "tool_input",
            )
            if source == "tool_input":
                output_mapping[role] = BindingExpression(
                    BindingExprKind.SKILL_INPUT, source_role=role,
                )
            else:
                output_mapping[role] = BindingExpression(
                    BindingExprKind.TOOL_OUTPUT, source_role=role, source_step="primary",
                )
            implementation_output_mapping[role] = BindingExpression(
                BindingExprKind.TOOL_OUTPUT, source_role=role, source_step="primary",
            )
        tool_id = f"tool_{atomic.ref.logical_id.removeprefix('atomic_')}"
        tool_ref = ToolRef(tool_id, "1.0.0")
        tool = ToolAsset(
            tool_ref,
            f"IR implementation of {atomic.summary}",
            {"type": "object", "properties": tool_properties, "required": sorted(tool_properties)},
            {"output_schema": {
                "type": "object",
                "properties": output_properties,
                "required": sorted(output_properties),
                "additionalProperties": False,
            }},
            "tool_ir_v1",
            {
                "schema_version": 1,
                "max_actions": int(proposal.max_actions),
                "program": program,
                "final_effects": proposal.final_effects,
                "evidence_outputs": proposal.evidence_outputs,
                "output_mapping": output_mapping,
                "path_expectations": proposal.path_expectations,
            },
            [{
                "kind": "tool_proposal_replay",
                "trace_id": provenance.source_trace_id,
                "occurrence_id": provenance.occurrence_id,
                "draft_id": provenance.draft_id,
                "bindings": dict(occurrence.input_bindings),
                "source_task": dict(occurrence.source_task),
                "prefix": [
                    {
                        "action_type": event["action_type"],
                        "arguments": dict(event.get("arguments", {})),
                    }
                    for event in occurrence.prefix_events
                    if event.get("accepted")
                ],
                "effects": [dict(
                    predicate=item.predicate,
                    args=dict(item.args),
                    cardinality=int(item.cardinality),
                    distinct_by=str(item.distinct_by),
                    effect_domain=str(item.effect_domain.value),
                ) for item in proposal.final_effects],
            }],
            {
                "reviewed": True,
                "allowed_action_types": sorted({
                    str(node.get("action_type", ""))
                    for node in action_nodes
                    if node.get("action_type")
                }),
                "zero_llm": True,
                "terminal_interruptible": True,
            },
            {
                "source": provenance.source,
                "source_trace_id": provenance.source_trace_id,
                "occurrence_id": provenance.occurrence_id,
                "draft_id": provenance.draft_id,
            },
            {
                "tool_builder_summary": proposal.summary,
                "tool_builder_rationale": proposal.rationale,
                "schema_version": 1,
            },
            ToolStatus.ADMISSION_PENDING,
        )
        tool_binding_mapping = {
            str(item.name): BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role=str(item.name),
            )
            for item in proposal.inputs or atomic.inputs
        }
        constraints = []
        if action_nodes:
            first = action_nodes[0]
            constraints.append(GroundingConstraint(
                "entry_affordance", GroundingConstraintKind.HARNESS_AFFORDANCE,
                action_type=str(first.get("action_type", "")),
                argument_mapping=dict(first.get("argument_mapping", {})),
                required_resolution="concrete",
            ))
        implementation = ImplementationAtom(
            SkillRef(f"impl_{atomic.ref.logical_id.removeprefix('atomic_')}", "1.0.0"),
            atomic.ref,
            [ToolBinding(tool.ref, "primary", tool_binding_mapping, 0)],
            constraints,
            {"mode": "serial", "output_mapping": implementation_output_mapping},
            {"harness_profiles": ["alfworld_v3", "fake_v3"]},
            {},
            SkillStatus.DRAFT,
        )
        return CompiledKnowledge(occurrence, atomic, tool, implementation)
