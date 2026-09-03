"""Neutral v3.2 tooling contracts shared by Runtime and Success Evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..core.contracts import ParameterSpec, SemanticPredicate


class ToolProgramOp(str, Enum):
    ACTION = "ACTION"
    IF = "IF"
    FOR_EACH = "FOR_EACH"
    STOP_WHEN = "STOP_WHEN"
    RETURN = "RETURN"


@dataclass
class RuntimeAutomationAtomicDraft:
    draft_id: str
    intent: str
    inputs: list[ParameterSpec]
    outputs: list[ParameterSpec]
    preconditions: list[SemanticPredicate]
    effects: list[SemanticPredicate]
    rationale: str
    source_occurrence_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolProposal:
    proposal_version: str
    decision: str
    summary: str
    atomic_ref: str
    inputs: list[ParameterSpec]
    outputs: list[ParameterSpec]
    program: list[dict[str, Any]]
    max_actions: int
    final_effects: list[SemanticPredicate]
    evidence_outputs: list[dict[str, Any]]
    path_expectations: list[dict[str, Any]]
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def creates_tool(self) -> bool:
        return self.decision == "create"

    @classmethod
    def no_tool(
        cls,
        *,
        atomic_ref: str,
        reason_code: str,
        summary: str = "NO_TOOL",
    ) -> "ToolProposal":
        return cls(
            proposal_version="1",
            decision="no_tool",
            summary=summary,
            atomic_ref=str(atomic_ref),
            inputs=[],
            outputs=[],
            program=[],
            max_actions=0,
            final_effects=[],
            evidence_outputs=[],
            path_expectations=[],
            rationale=reason_code,
            metadata={"reason_code": reason_code},
        )


@dataclass(frozen=True)
class ToolProvenance:
    source: str
    atomic_ref: str
    source_trace_id: str
    occurrence_id: str
    draft_id: str = ""
    task_id: str = ""

    @property
    def runtime_automation(self) -> bool:
        return self.source == "runtime_automation"


@dataclass
class CompiledToolBundle:
    atomic: Any
    tool: Any
    implementation: Any


def _predicate(value: Mapping[str, Any]) -> SemanticPredicate:
    return SemanticPredicate(
        predicate=str(value["predicate"]),
        args=dict(value.get("args", {})),
        cardinality=int(value.get("cardinality", 1)),
        distinct_by=str(value.get("distinct_by", "")),
        effect_domain=str(value.get("effect_domain", "world")),
    )


def _parameter(value: Mapping[str, Any]) -> ParameterSpec:
    return ParameterSpec(**{
        key: value.get(key, default)
        for key, default in {
            "name": None,
            "semantic_type": None,
            "required": True,
            "runtime_resolvable": False,
            "required_resolution": "semantic",
            "description": "",
        }.items()
        if value.get(key, default) is not None
    })


def runtime_automation_draft_from_dict(value: Mapping[str, Any]) -> RuntimeAutomationAtomicDraft:
    return RuntimeAutomationAtomicDraft(
        draft_id=str(value["draft_id"]),
        intent=str(value["intent"]),
        inputs=[_parameter(item) for item in value.get("inputs", [])],
        outputs=[_parameter(item) for item in value.get("outputs", [])],
        preconditions=[_predicate(item) for item in value.get("preconditions", [])],
        effects=[_predicate(item) for item in value.get("effects", [])],
        rationale=str(value.get("rationale", "")),
        source_occurrence_id=str(value.get("source_occurrence_id", "")),
        metadata=dict(value.get("metadata", {})),
    )


def tool_proposal_from_dict(value: Mapping[str, Any]) -> ToolProposal:
    return ToolProposal(
        proposal_version=str(value.get("proposal_version", "1")),
        decision=str(value.get("decision", "create")),
        summary=str(value.get("summary", "")),
        atomic_ref=str(value.get("atomic_ref", "")),
        inputs=[_parameter(item) for item in value.get("inputs", [])],
        outputs=[_parameter(item) for item in value.get("outputs", [])],
        program=[dict(item) for item in value.get("program", [])],
        max_actions=int(value.get("max_actions", 0)),
        final_effects=[_predicate(item) for item in value.get("final_effects", [])],
        evidence_outputs=[dict(item) for item in value.get("evidence_outputs", [])],
        path_expectations=[dict(item) for item in value.get("path_expectations", [])],
        rationale=str(value.get("rationale", "")),
        metadata=dict(value.get("metadata", {})),
    )


__all__ = [
    "CompiledToolBundle",
    "RuntimeAutomationAtomicDraft",
    "ToolProgramOp",
    "ToolProposal",
    "ToolProvenance",
    "runtime_automation_draft_from_dict",
    "tool_proposal_from_dict",
]
