"""Compile necessary accepted action slices into parameterized Primitive IR assets."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ..core.bindings import (
    BindingExpression, BindingExprKind, GroundingConstraint, GroundingConstraintKind,
    ToolBinding,
)
from ..core.contracts import AbstractAtomicSkill, ImplementationAtom, ToolAsset
from ..core.refs import SkillRef, ToolRef, content_hash
from ..core.serialization import to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..harness.protocol import HarnessTask
from ..tooling.ir import (
    normalize_return_output_sources,
    normalize_tool_program,
    walk_program_nodes,
)
from ..tooling.entry_contract import normalize_entry_contract, parameter_schema
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
        guideline=dict(compiled.atomic.guideline or {}),
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


def _task_source_payload(source_task: HarnessTask | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source_task, HarnessTask):
        metadata = dict(source_task.metadata)
        return {
            "task_id": str(source_task.task_id),
            "task_signature": str(metadata.get("task_signature", "")),
            "goal": str(source_task.goal),
            "benchmark": str(source_task.benchmark),
            "task_type": str(source_task.task_type),
            "context": copy.deepcopy(dict(source_task.context)),
            "metadata": copy.deepcopy(metadata),
        }
    return copy.deepcopy(dict(source_task))


def build_occurrence_replay_case(
    canonical_occurrence: CanonicalAtomicOccurrence,
    canonical_atomic: AbstractAtomicSkill,
    *,
    source_task: HarnessTask | Mapping[str, Any],
    kind: str = "source_replay",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one current-occurrence replay case after canonical role rewrite."""

    def present(value: Any) -> bool:
        return value is not None and value != ""

    supplied = copy.deepcopy(dict(canonical_occurrence.source_task or {}))
    authority = _task_source_payload(source_task)
    for field in ("task_id", "task_signature", "goal", "benchmark", "task_type"):
        left, right = supplied.get(field), authority.get(field)
        if present(left) and present(right) and left != right:
            raise ValueError(f"occurrence source task conflicts with current task field {field}")
    for group in ("context", "metadata"):
        supplied_group = dict(supplied.get(group) or {})
        authority_group = dict(authority.get(group) or {})
        for field in set(supplied_group) & set(authority_group):
            left, right = supplied_group[field], authority_group[field]
            if present(left) and present(right) and left != right:
                raise ValueError(
                    f"occurrence source task conflicts with current task field {group}.{field}"
                )
        authority[group] = {**supplied_group, **authority_group}
    normalized_source = {
        field: authority.get(field) or supplied.get(field, "")
        for field in ("task_id", "task_signature", "goal", "benchmark", "task_type")
    }
    normalized_source["context"] = copy.deepcopy(dict(authority.get("context") or {}))
    normalized_source["metadata"] = copy.deepcopy(dict(authority.get("metadata") or {}))

    declared_roles = {str(item.name) for item in canonical_atomic.inputs}
    required_roles = {
        str(item.name) for item in canonical_atomic.inputs if bool(item.required)
    }
    bindings = copy.deepcopy(dict(canonical_occurrence.input_bindings))
    unknown = sorted(set(bindings) - declared_roles)
    missing = sorted(required_roles - set(bindings))
    if unknown or missing:
        raise ValueError(
            "canonical replay bindings do not match Atomic inputs: "
            f"unknown={unknown}, missing={missing}"
        )
    trace_id = str(canonical_occurrence.source_trace_id).strip()
    if not trace_id:
        raise ValueError("current occurrence replay requires source_trace_id")
    case: dict[str, Any] = {
        "kind": str(kind),
        "trace_id": trace_id,
        "occurrence_id": str(canonical_occurrence.occurrence_id),
        "event_range": [
            int(canonical_occurrence.event_start),
            int(canonical_occurrence.event_end),
        ],
        "support_event_ids": [
            str(event.get("event_id", event.get("action_id", index)))
            for index, event in enumerate(canonical_occurrence.action_events)
        ],
        "bindings": bindings,
        "source_task": normalized_source,
        "prefix": [
            {
                "action_type": str(event["action_type"]),
                "arguments": copy.deepcopy(dict(event.get("arguments") or {})),
            }
            for event in canonical_occurrence.prefix_events
            if event.get("accepted")
        ],
        "effects": to_primitive(canonical_atomic.effects),
    }
    if extra:
        case.update(copy.deepcopy(dict(extra)))
    case["case_id"] = f"replay_case_{content_hash(case)[:24]}"
    return case


class ToolCompiler:
    def compile(self, occurrences: list[CanonicalAtomicOccurrence], *, entry_contracts=None) -> list[CompiledKnowledge]:
        """Compile source-replay artifacts; deployment requires an authored entry contract."""
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
                dict(occurrence.guideline),
                {"source_trace_ids": [occurrence.source_trace_id]}, SkillStatus.DRAFT,
            )
            primitive_steps = []
            # The Tool signature carries the whole Atomic input context, not
            # only values syntactically present in the terminal primitive.
            # ALFWorld USE-lamp, for example, also requires the held target
            # object to publish/validate ``object.observed_with``.
            input_schema = parameter_schema(occurrence.input_specs)
            tool_properties = input_schema["properties"]
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
                input_schema,
                {"output_schema": parameter_schema(occurrence.output_specs)},
                "primitive_ir", {"steps": primitive_steps, "output_mapping": tool_output_mapping},
                [build_occurrence_replay_case(
                    occurrence,
                    atomic,
                    source_task=occurrence.source_task,
                    kind="source_replay",
                )],
                {"reviewed": True, "allowed_action_types": [item["action_type"] for item in primitive_steps]},
                {"source_trace_id": occurrence.source_trace_id, "occurrence_id": occurrence.occurrence_id,
                 "source_replay_only": occurrence.phase_id not in (entry_contracts or {})},
                {}, ToolStatus.ADMISSION_PENDING,
            )
            tool_binding_mapping = {
                role: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)
                for role in tool_properties
            }
            constraints = []
            if occurrence.phase_id in (entry_contracts or {}):
                tool.interface["entry_contract"] = normalize_entry_contract(
                    entry_contracts[occurrence.phase_id], (p.name for p in occurrence.input_specs))
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
        *,
        source_task: HarnessTask | Mapping[str, Any] | None = None,
    ) -> CompiledKnowledge:
        """Compile an Agent-authored ToolProposal into ToolAsset/ImplementationAtom.

        The compiler never chooses which actions enter the Tool; that authority
        belongs to ToolBuilder.  It only normalizes the already-validated IR into
        the persistent artifact envelope.
        """

        if proposal.decision == "no_tool":
            return CompiledKnowledge(occurrence, atomic, None, None)
        if not proposal.program:
            raise ValueError("ToolProposal cannot compile an empty Tool IR program")
        # Static validation works on a normalized projection and must not be
        # relied upon to mutate the submitted proposal.  Persist a detached,
        # canonical program explicitly so a legacy single-output RETURN has
        # the same shape at compile, admission, and runtime boundaries.
        program = normalize_tool_program(copy.deepcopy(proposal.program))
        output_roles = {str(item.name) for item in atomic.outputs}
        for node in walk_program_nodes(program):
            if str(node.get("op", "")) == "RETURN":
                node["output_sources"] = normalize_return_output_sources(
                    node, output_roles,
                )
        action_nodes = [
            node for node in walk_program_nodes(program)
            if str(node.get("op", "")) == "ACTION"
        ]
        if proposal.proposal_version != "2":
            raise ValueError("R10.2 requires ToolProposal version 2")
        entry = normalize_entry_contract(proposal.entry_contract, (p.name for p in atomic.inputs))
        output_mapping: dict[str, Any] = {}
        implementation_output_mapping: dict[str, Any] = {}
        for output in atomic.outputs:
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
            parameter_schema(atomic.inputs),
            {"output_schema": parameter_schema(atomic.outputs), "entry_contract": entry},
            "tool_ir_v1",
            {
                "schema_version": 1,
                "max_actions": int(proposal.max_actions),
                "program": program,
                "final_effects": proposal.final_effects,
                "evidence_outputs": proposal.evidence_outputs,
                "path_expectations": proposal.path_expectations,
            },
            [build_occurrence_replay_case(
                occurrence,
                atomic,
                source_task=source_task or occurrence.source_task,
                kind="tool_proposal_replay",
                extra={
                    "trace_id": provenance.source_trace_id,
                    "occurrence_id": provenance.occurrence_id,
                    "draft_id": provenance.draft_id,
                },
            )],
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
            for item in atomic.inputs
        }
        implementation = ImplementationAtom(
            SkillRef(f"impl_{atomic.ref.logical_id.removeprefix('atomic_')}", "1.0.0"),
            atomic.ref,
            [ToolBinding(tool.ref, "primary", tool_binding_mapping, 0)],
            [],  # Tool entry requirements stay in its authored interface.
            {"mode": "serial", "output_mapping": implementation_output_mapping},
            {"harness_profiles": ["alfworld_v3", "fake_v3"]},
            {},
            SkillStatus.DRAFT,
        )
        return CompiledKnowledge(occurrence, atomic, tool, implementation)
