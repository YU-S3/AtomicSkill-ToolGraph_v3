"""Task, atomic, implementation, tool, and composite semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bindings import BindingExpression, GroundingConstraint, ToolBinding
from .edges import GraphEdge
from .refs import SkillRef, ToolRef
from .status import SkillStatus, ToolStatus


@dataclass(frozen=True)
class SemanticPredicate:
    predicate: str
    args: dict[str, BindingExpression | Any]
    cardinality: int = 1
    distinct_by: str = ""


@dataclass
class ParameterSpec:
    name: str
    semantic_type: str
    required: bool = True
    runtime_resolvable: bool = False
    required_resolution: str = "semantic"
    description: str = ""

    def __post_init__(self) -> None:
        if self.required_resolution not in {"semantic", "concrete", "relation_verified"}:
            raise ValueError(f"unsupported resolution: {self.required_resolution}")


class ContractSource(str, Enum):
    BENCHMARK_FORMAL = "benchmark_formal"
    ADAPTER_DERIVED = "adapter_derived"
    PLANNER_PROPOSED = "planner_proposed"


class IdentityRelation(str, Enum):
    SAME_AS = "same_as"
    DISTINCT_FROM = "distinct_from"


@dataclass(frozen=True)
class IdentityConstraint:
    left_role: str
    relation: IdentityRelation
    right_role: str
    scope: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", IdentityRelation(self.relation))
        if self.scope not in {"occurrence", "task"}:
            raise ValueError("identity constraint scope must be occurrence or task")


@dataclass
class TaskContract:
    target_effects: list[SemanticPredicate] = field(default_factory=list)
    cardinality_constraints: list[dict[str, Any]] = field(default_factory=list)
    identity_constraints: list[IdentityConstraint | dict[str, Any]] = field(default_factory=list)
    source: ContractSource = ContractSource.PLANNER_PROPOSED
    confidence: float = 0.0
    validator_id: str = ""

    def __post_init__(self) -> None:
        self.source = ContractSource(self.source)
        self.identity_constraints = [
            value if isinstance(value, IdentityConstraint) else IdentityConstraint(**value)
            for value in self.identity_constraints
        ]


@dataclass(frozen=True)
class CompositeOccurrence:
    step_id: str
    occurrence_id: str
    node_ref: SkillRef
    binding_specs: dict[str, BindingExpression]


@dataclass
class AbstractAtomicSkill:
    ref: SkillRef
    summary: str
    inputs: list[ParameterSpec]
    outputs: list[ParameterSpec]
    preconditions: list[SemanticPredicate]
    effects: list[SemanticPredicate]
    validator_spec: dict[str, Any]
    failure_modes: list[dict[str, Any]]
    guideline: dict[str, Any]
    metadata: dict[str, Any]
    status: SkillStatus = SkillStatus.DRAFT

    def __post_init__(self) -> None:
        self.status = SkillStatus(self.status)


@dataclass
class ImplementationAtom:
    ref: SkillRef
    abstract_ref: SkillRef
    tool_bindings: list[ToolBinding]
    grounding_constraints: list[GroundingConstraint]
    execution_policy: dict[str, Any]
    compatibility: dict[str, Any]
    quality: dict[str, Any]
    status: SkillStatus = SkillStatus.DRAFT

    def __post_init__(self) -> None:
        self.status = SkillStatus(self.status)
        self.tool_bindings = [item if isinstance(item, ToolBinding) else ToolBinding(**item) for item in self.tool_bindings]
        self.grounding_constraints = [
            item if isinstance(item, GroundingConstraint) else GroundingConstraint(**item)
            for item in self.grounding_constraints
        ]


@dataclass
class ToolAsset:
    ref: ToolRef
    summary: str
    signature: dict[str, Any]
    interface: dict[str, Any]
    artifact_kind: str
    artifact: dict[str, Any]
    tests: list[dict[str, Any]]
    safety: dict[str, Any]
    provenance: dict[str, Any]
    metadata: dict[str, Any]
    status: ToolStatus = ToolStatus.DRAFT

    def __post_init__(self) -> None:
        self.status = ToolStatus(self.status)


@dataclass
class CompositeSkill:
    ref: SkillRef
    summary: str
    occurrences: list[CompositeOccurrence]
    control_sequence: list[str]
    data_edges: list[GraphEdge]
    dependency_edges: list[GraphEdge]
    goal_contract: TaskContract
    guideline: dict[str, Any]
    insight: dict[str, Any]
    validator_spec: dict[str, Any]
    metadata: dict[str, Any]
    status: SkillStatus = SkillStatus.DRAFT

    def __post_init__(self) -> None:
        self.status = SkillStatus(self.status)


@dataclass
class CapabilityRequirement:
    requirement_id: str
    intent: str
    desired_effects: list[SemanticPredicate]
    expected_inputs: list[ParameterSpec]
    expected_outputs: list[ParameterSpec]
    precondition_hints: list[SemanticPredicate]
    semantic_variants: list[str]
    required: bool
    rationale: str


@dataclass
class AtomicCandidate:
    atomic_ref: SkillRef
    score: float
    reasons: list[str] = field(default_factory=list)
    contract_match: bool = True


@dataclass
class RequirementSearchResult:
    requirement: CapabilityRequirement
    candidates: list[AtomicCandidate]
    covered: bool
    rejection_reasons: list[dict[str, Any]]


@dataclass
class ProposedOccurrence:
    step_id: str
    occurrence_id: str
    node_ref: SkillRef
    requirement_ids: list[str]
    binding_specs: dict[str, BindingExpression]
    expected_effects: list[SemanticPredicate] = field(default_factory=list)


@dataclass
class ProposedEdge:
    edge_id: str
    edge_type: str
    source_step: str
    target_step: str
    source_role: str = ""
    target_role: str = ""
    origin: str = "planner_proposed"
    existing_edge_id: str = ""


@dataclass
class PlannerWorkflowProposal:
    steps: list[ProposedOccurrence]
    control_sequence: list[str]
    data_edges: list[ProposedEdge]
    dependency_edges: list[ProposedEdge]
    requirement_coverage: dict[str, list[str]]
    sequence_origin: str = "planner_proposed_sequence"


@dataclass
class PlannerAudit:
    composite_candidates: list[dict[str, Any]] = field(default_factory=list)
    composite_rejections: list[dict[str, Any]] = field(default_factory=list)
    selected_composite: str | None = None
    requirements_p1: list[dict[str, Any]] = field(default_factory=list)
    atomic_search_p1: dict[str, Any] = field(default_factory=dict)
    related_composite_hints: list[dict[str, Any]] = field(default_factory=list)
    requirements_p1r: list[dict[str, Any]] = field(default_factory=list)
    atomic_search_p1r: dict[str, Any] = field(default_factory=dict)
    workflow_p2: dict[str, Any] = field(default_factory=dict)
    validation_p2: dict[str, Any] = field(default_factory=dict)
    workflow_p2r: dict[str, Any] = field(default_factory=dict)
    validation_p2r: dict[str, Any] = field(default_factory=dict)
    final_outcome: str = ""
    fallback_reason: str = ""
