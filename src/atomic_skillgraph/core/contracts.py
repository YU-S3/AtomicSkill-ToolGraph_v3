"""Task, atomic, implementation, tool, and composite semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bindings import BindingExpression, GroundingConstraint, ToolBinding
from .edges import GraphEdge
from .refs import SkillRef, ToolRef
from .status import SkillStatus, ToolStatus


class EffectDomain(str, Enum):
    WORLD = "world"
    EVIDENCE = "evidence"


@dataclass(frozen=True)
class SemanticPredicate:
    predicate: str
    args: dict[str, BindingExpression | Any]
    cardinality: int = 1
    distinct_by: str = ""
    effect_domain: EffectDomain = EffectDomain.WORLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_domain", EffectDomain(self.effect_domain))


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
    metadata: dict[str, Any] = field(default_factory=dict)

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


@dataclass(frozen=True)
class RepeatBlock:
    """A contract-backed serial repetition of one reusable requirement unit."""

    block_id: str
    count: int
    ordered_requirement_ids: tuple[str, ...]
    distinct_roles: tuple[str, ...]
    shared_roles: tuple[str, ...]
    basis_constraint_id: str
    basis_role_map: dict[str, str]
    execution_policy: str = "serial"
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_requirement_ids", tuple(self.ordered_requirement_ids))
        object.__setattr__(self, "distinct_roles", tuple(self.distinct_roles))
        object.__setattr__(self, "shared_roles", tuple(self.shared_roles))
        object.__setattr__(self, "basis_role_map", dict(self.basis_role_map))


@dataclass
class PlannerRequirementBundle:
    requirements: list[CapabilityRequirement]
    repeat_blocks: list[RepeatBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.requirements = list(self.requirements)
        self.repeat_blocks = [
            value if isinstance(value, RepeatBlock) else RepeatBlock(**value)
            for value in self.repeat_blocks
        ]


@dataclass
class AtomicCandidate:
    atomic_ref: SkillRef
    score: float
    reasons: list[str] = field(default_factory=list)
    contract_match: bool = True


@dataclass(frozen=True)
class PredicateCompatibilityDetail:
    required_predicate: str
    offered_predicate_found: bool
    required_argument_roles: tuple[str, ...]
    missing_argument_roles: tuple[str, ...]
    required_cardinality: int
    best_offered_cardinality: int
    cardinality_sufficient: bool


@dataclass(frozen=True)
class RequiredInputCompatibility:
    required_name: str
    required_semantic_type: str
    compatible_offered_roles: tuple[str, ...]


@dataclass(frozen=True)
class AtomicContractCompatibilityReport:
    passed: bool
    effects_passed: bool
    inputs_passed: bool
    effect_details: tuple[PredicateCompatibilityDetail, ...]
    input_details: tuple[RequiredInputCompatibility, ...]
    missing_required_input_types: tuple[str, ...]
    failure_codes: tuple[str, ...]


@dataclass
class RequirementSearchResult:
    requirement: CapabilityRequirement
    candidates: list[AtomicCandidate]
    covered: bool
    rejection_reasons: list[dict[str, Any]]
    repair_hints: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProposedOccurrence:
    step_id: str
    occurrence_id: str
    node_ref: SkillRef
    requirement_ids: list[str]
    binding_specs: dict[str, BindingExpression]
    expected_effects: list[SemanticPredicate] = field(default_factory=list)
    # ``requirement_ids`` is retained as a read-only compatibility projection
    # for v3 deterministic fixtures.  New Planner submissions use the two
    # fields below and the compiler copies instance ids into that projection.
    requirement_instance_ids: list[str] = field(default_factory=list)
    repeat_role_bindings: dict[str, str] = field(default_factory=dict)


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


class ColdStartCandidateSource(str, Enum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    UNRESOLVED = "unresolved"


class ColdStartExecutionMode(str, Enum):
    DIRECT_OR_SEEDED = "direct_or_seeded"
    SEEDED_ONLY = "seeded_only"
    DYNAMIC = "dynamic"


@dataclass
class ColdStartPlanStep:
    step_id: str
    requirement_instance_ids: list[str]
    candidate_source: ColdStartCandidateSource
    candidate_ref: str
    execution_mode: ColdStartExecutionMode
    binding_specs: dict[str, BindingExpression]
    repeat_role_bindings: dict[str, str]
    expected_effects: list[SemanticPredicate] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.candidate_source = ColdStartCandidateSource(self.candidate_source)
        self.execution_mode = ColdStartExecutionMode(self.execution_mode)


@dataclass
class ColdStartPlanProposal:
    plan_id: str
    steps: list[ColdStartPlanStep]
    control_sequence: list[str]
    data_edges: list[ProposedEdge]
    dependency_edges: list[ProposedEdge]
    requirement_coverage: dict[str, list[str]]
    referenced_failure_experience_ids: list[str]


@dataclass
class PlannerAudit:
    composite_candidates: list[dict[str, Any]] = field(default_factory=list)
    composite_rejections: list[dict[str, Any]] = field(default_factory=list)
    terminal_empirical_candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_composite_authority: dict[str, Any] = field(default_factory=dict)
    selected_composite: str | None = None
    requirements_p1: dict[str, Any] = field(default_factory=dict)
    atomic_search_p1: dict[str, Any] = field(default_factory=dict)
    related_composite_hints: list[dict[str, Any]] = field(default_factory=list)
    repairability: dict[str, Any] = field(default_factory=dict)
    requirements_p1r: dict[str, Any] = field(default_factory=dict)
    atomic_search_p1r: dict[str, Any] = field(default_factory=dict)
    workflow_p2: dict[str, Any] = field(default_factory=dict)
    validation_p2: dict[str, Any] = field(default_factory=dict)
    workflow_p2r: dict[str, Any] = field(default_factory=dict)
    validation_p2r: dict[str, Any] = field(default_factory=dict)
    final_outcome: str = ""
    fallback_reason: str = ""
    requirement_validation_p1: dict[str, Any] = field(default_factory=dict)
    requirement_validation_final: dict[str, Any] = field(default_factory=dict)
    requirement_expansion: dict[str, Any] = field(default_factory=dict)
    cold_start_retrieval: dict[str, Any] = field(default_factory=dict)
    cold_start_plan: dict[str, Any] = field(default_factory=dict)
    cold_start_validation: dict[str, Any] = field(default_factory=dict)
    cold_start_repair: dict[str, Any] = field(default_factory=dict)
    cold_start_repair_validation: dict[str, Any] = field(default_factory=dict)
