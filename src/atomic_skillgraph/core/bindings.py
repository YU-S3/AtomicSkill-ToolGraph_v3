"""Typed mapping AST, task-local bindings, and grounding evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .refs import ToolRef


class BindingExprKind(str, Enum):
    SKILL_INPUT = "skill_input"
    CONSTANT = "constant"
    DATA_FLOW = "data_flow"
    TOOL_OUTPUT = "tool_output"
    ADAPTER_TRANSFORM = "adapter_transform"


@dataclass(frozen=True)
class BindingExpression:
    kind: BindingExprKind
    source_role: str = ""
    source_step: str = ""
    constant: Any = None
    transform_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BindingExprKind(self.kind))
        if self.kind is BindingExprKind.CONSTANT:
            return
        if self.kind is BindingExprKind.ADAPTER_TRANSFORM and not self.transform_id:
            raise ValueError("adapter_transform requires transform_id")
        if not self.source_role:
            raise ValueError(f"{self.kind.value} requires source_role")
        if self.kind in {BindingExprKind.DATA_FLOW, BindingExprKind.TOOL_OUTPUT} and not self.source_step:
            raise ValueError(f"{self.kind.value} requires source_step")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | "BindingExpression") -> "BindingExpression":
        if isinstance(value, cls):
            return value
        return cls(
            kind=BindingExprKind(value["kind"]),
            source_role=str(value.get("source_role", "")),
            source_step=str(value.get("source_step", "")),
            constant=value.get("constant"),
            transform_id=str(value.get("transform_id", "")),
        )


class GroundingConstraintKind(str, Enum):
    ARGUMENT_EXISTS = "argument_exists"
    ARGUMENT_CONCRETE = "argument_concrete"
    HARNESS_AFFORDANCE = "harness_affordance"
    CURRENT_CONTEXT = "current_context"
    CUSTOM_ADAPTER = "custom_adapter"


@dataclass
class GroundingConstraint:
    constraint_id: str
    kind: GroundingConstraintKind
    action_type: str = ""
    argument_mapping: dict[str, BindingExpression] = field(default_factory=dict)
    required_resolution: str = "concrete"
    verifier_id: str = ""

    def __post_init__(self) -> None:
        self.kind = GroundingConstraintKind(self.kind)
        self.argument_mapping = {
            name: BindingExpression.from_dict(expr)
            for name, expr in self.argument_mapping.items()
        }


@dataclass
class ToolBinding:
    tool_ref: ToolRef
    role: str
    parameter_mapping: dict[str, BindingExpression]
    order: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tool_ref, ToolRef):
            self.tool_ref = ToolRef.from_dict(self.tool_ref) if isinstance(self.tool_ref, dict) else ToolRef.parse(self.tool_ref)
        self.parameter_mapping = {
            name: BindingExpression.from_dict(expr)
            for name, expr in self.parameter_mapping.items()
        }


class BindingSource(str, Enum):
    TASK = "task"
    DATA_FLOW = "data_flow"
    TOOL_OUTPUT = "tool_output"
    HARNESS_EVIDENCE = "harness_evidence"
    AGENT_PROPOSED = "agent_proposed"
    UNRESOLVED = "unresolved"


class BindingStatus(str, Enum):
    GROUNDED = "grounded"
    PROPOSED = "proposed"
    UNRESOLVED = "unresolved"
    INVALIDATED = "invalidated"


class BindingResolution(str, Enum):
    SEMANTIC = "semantic"
    CONCRETE = "concrete"
    RELATION_VERIFIED = "relation_verified"


_RESOLUTION_RANK = {
    BindingResolution.SEMANTIC: 0,
    BindingResolution.CONCRETE: 1,
    BindingResolution.RELATION_VERIFIED: 2,
}


def resolution_satisfies(actual: BindingResolution | str, required: BindingResolution | str) -> bool:
    return _RESOLUTION_RANK[BindingResolution(actual)] >= _RESOLUTION_RANK[BindingResolution(required)]


@dataclass
class RuntimeBinding:
    role: str
    value: Any
    semantic_type: str
    source: BindingSource
    status: BindingStatus
    resolution: BindingResolution
    evidence_refs: list[str] = field(default_factory=list)
    world_revision: int = 0

    def __post_init__(self) -> None:
        self.source = BindingSource(self.source)
        self.status = BindingStatus(self.status)
        self.resolution = BindingResolution(self.resolution)


class EvidenceStability(str, Enum):
    REVISION_SCOPED = "revision_scoped"
    STATE_SCOPED = "state_scoped"
    PERSISTENT = "persistent"


@dataclass
class GroundingEvidence:
    evidence_id: str
    evidence_type: str
    payload: dict[str, Any]
    source: str
    observed_at_revision: int
    valid_from_revision: int
    invalidated_at_revision: int | None = None
    stability: EvidenceStability = EvidenceStability.REVISION_SCOPED
    action_id: str | None = None

    def __post_init__(self) -> None:
        self.stability = EvidenceStability(self.stability)

    def valid_at(self, revision: int) -> bool:
        if revision < self.valid_from_revision:
            return False
        if self.invalidated_at_revision is not None and revision >= self.invalidated_at_revision:
            return False
        if self.stability is EvidenceStability.REVISION_SCOPED:
            return revision == self.observed_at_revision
        return True


@dataclass(frozen=True)
class RuntimeBindingChange:
    occurrence_id: str
    role: str
    previous: dict[str, Any] | None
    current: dict[str, Any] | None
    reason: str
    revision: int
