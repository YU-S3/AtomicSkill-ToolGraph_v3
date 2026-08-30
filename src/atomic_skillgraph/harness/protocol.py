"""Policy-facing and validator-only benchmark boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.contracts import TaskContract
from ..core.refs import content_hash
from ..core.results import PrimitiveToolStep, ValidationResult
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
class SemanticFact:
    """One validator-owned semantic witness exposed through an opaque reference."""

    fact_ref: str
    predicate: str
    args: dict[str, Any]
    cardinality: int = 1
    distinct_by: str = ""


@dataclass(frozen=True)
class ActionTransitionCertificate:
    """The sole semantic authority for one executed environment action."""

    action_id: str
    revision_before: int
    revision_after: int
    action_type: str
    arguments: dict[str, Any]
    before_facts: tuple[SemanticFact, ...]
    positive_effects: tuple[SemanticFact, ...]
    negative_effects: tuple[SemanticFact, ...]
    required_facts: tuple[SemanticFact, ...]
    terminal_effects: tuple[SemanticFact, ...]
    accepted: bool
    state_changed: bool
    evidence_refs: tuple[str, ...]


def build_transition_certificate(
    *,
    action_id: str,
    revision_before: int,
    revision_after: int,
    action_type: str,
    arguments: dict[str, Any],
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    accepted: bool,
    required_fact_identities: set[tuple[str, tuple[tuple[str, Any], ...]]] | None = None,
    terminal_fact_identities: set[tuple[str, tuple[tuple[str, Any], ...]]] | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> ActionTransitionCertificate:
    """Create an immutable certificate from validator snapshots.

    This helper knows nothing about benchmark actions or predicates.  Adapters
    decide which snapshot facts are required or terminal and pass identities.
    """

    def identity(raw: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
        return (
            str(raw.get("predicate", "")),
            tuple(sorted(dict(raw.get("args") or {}).items())),
        )

    before_by_id = {identity(item): item for item in before_snapshot.get("facts", [])}
    after_by_id = {identity(item): item for item in after_snapshot.get("facts", [])}
    positive_ids = sorted(set(after_by_id) - set(before_by_id), key=repr)
    negative_ids = sorted(set(before_by_id) - set(after_by_id), key=repr)

    def fact(
        raw_identity: tuple[str, tuple[tuple[str, Any], ...]],
        boundary: str,
        source: dict[str, Any],
    ) -> SemanticFact:
        predicate, items = raw_identity
        arguments_value = dict(items)
        digest = content_hash({
            "action_id": action_id,
            "revision_before": revision_before,
            "revision_after": revision_after,
            "predicate": predicate,
            "args": arguments_value,
        })[:20]
        return SemanticFact(
            f"transition:{action_id}:{boundary}:{digest}",
            predicate,
            arguments_value,
            max(1, int(source.get("cardinality", 1))),
            str(source.get("distinct_by", "")),
        )

    before_facts = tuple(
        fact(item, "before", before_by_id[item])
        for item in sorted(before_by_id, key=repr)
    )
    positive_effects = tuple(
        fact(item, "effect", after_by_id[item]) for item in positive_ids
    )
    negative_effects = tuple(
        fact(item, "negative", before_by_id[item]) for item in negative_ids
    )
    before_fact_by_identity = {
        identity_value: value
        for identity_value, value in zip(sorted(before_by_id, key=repr), before_facts)
    }
    positive_fact_by_identity = {
        identity_value: value
        for identity_value, value in zip(positive_ids, positive_effects)
    }
    required_facts = tuple(
        before_fact_by_identity[item]
        for item in sorted(required_fact_identities or set(), key=repr)
        if item in before_fact_by_identity
    )
    terminal_effects = tuple(
        positive_fact_by_identity[item]
        for item in sorted(terminal_fact_identities or set(), key=repr)
        if item in positive_fact_by_identity
    )
    all_evidence = tuple(dict.fromkeys((
        *evidence_refs,
        *(item.fact_ref for item in positive_effects),
        *(item.fact_ref for item in negative_effects),
        *(item.fact_ref for item in terminal_effects),
    )))
    return ActionTransitionCertificate(
        action_id=action_id,
        revision_before=int(revision_before),
        revision_after=int(revision_after),
        action_type=action_type,
        arguments=dict(arguments),
        before_facts=before_facts,
        positive_effects=positive_effects,
        negative_effects=negative_effects,
        required_facts=required_facts,
        terminal_effects=terminal_effects,
        accepted=bool(accepted),
        state_changed=bool(positive_effects or negative_effects),
        evidence_refs=all_evidence,
    )


@dataclass
class HarnessActionResult:
    accepted: bool
    observation: str
    done: bool
    won: bool
    new_revision: int
    catalog: list[HarnessActionSpec]
    metadata: dict[str, Any] = field(default_factory=dict)
    transition_certificate: ActionTransitionCertificate | None = None


@runtime_checkable
class ValidatorChannel(Protocol):
    validation_strength: str

    def snapshot(self) -> dict[str, Any]: ...
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
    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool: ...
