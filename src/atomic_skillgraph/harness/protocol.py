"""Policy-facing and validator-only benchmark boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

from ..core.contracts import TaskContract
from ..core.results import AtomicEffectResolution, PrimitiveToolStep, ValidationResult
from ..validation.contract_matcher import ContractMatcher


@dataclass
class HarnessTask:
    task_id: str
    goal: str
    benchmark: str
    task_type: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessActionSpec:
    action_id: str
    revision: int
    action_type: str
    arguments: dict[str, Any]
    display_text: str
    raw_action: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PredicateSpec:
    predicate: str
    effect_domain: str
    argument_roles: tuple[str, ...]
    argument_semantic_types: dict[str, str]
    validation_source: str


@dataclass
class HarnessActionResult:
    accepted: bool
    observation: str
    done: bool
    won: bool
    new_revision: int
    catalog: list[HarnessActionSpec]
    metadata: dict[str, Any] = field(default_factory=dict)


class AtomicEffectResolutionRequest(TypedDict, total=False):
    """Validator request for action-derived Atomic witness resolution.

    ``preferred_bindings`` is an Agent-declared role-to-value preference.  It
    only filters factual candidate assignments; it is not factual authority
    and cannot create facts or override known bindings/semantic anchors.

    ``authoritative_evidence_facts`` contains only still-current facts created
    or re-established by accepted actions in this occurrence.  It is code
    authority and is independent of AgentSession/ToolCall boundaries.
    """

    atomic_ref: str
    occurrence_id: str
    effects: list[Any]
    known_bindings: dict[str, Any]
    semantic_anchors: dict[str, Any]
    input_specs: list[Any]
    output_specs: list[Any]
    output_identity: list[dict[str, Any]]
    preferred_values: list[Any]
    preferred_bindings: dict[str, Any]
    authoritative_evidence_facts: list[dict[str, Any]]
    current_revision: int


@runtime_checkable
class ValidatorChannel(Protocol):
    validation_strength: str

    def snapshot(self) -> dict[str, Any]: ...
    def resolve_atomic_effect(
        self, request: AtomicEffectResolutionRequest,
    ) -> AtomicEffectResolution: ...
    def validate_atomic_effect(self, request: dict[str, Any]) -> ValidationResult: ...
    def validate_task_contract(self, contract: TaskContract) -> ValidationResult: ...


@runtime_checkable
class HarnessAdapter(Protocol):
    profile_name: str

    def reset(self, task: HarnessTask) -> HarnessActionResult: ...
    def action_catalog(self) -> list[HarnessActionSpec]: ...
    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult: ...
    def semantic_value_compatible(
        self, *, role: str, concrete_value: Any,
        semantic_anchor: Any, semantic_type: str,
    ) -> bool: ...
    def task_contract(self, task: HarnessTask) -> TaskContract: ...
    def contract_matcher(self) -> ContractMatcher: ...
    def validator_channel(self) -> ValidatorChannel: ...
    def compile_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> Any: ...
    def execute_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> HarnessActionResult: ...
    def semantic_predicate_schema(self) -> list[PredicateSpec]: ...
    def primitive_action_schema(self) -> list[dict[str, Any]]: ...
    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool: ...
