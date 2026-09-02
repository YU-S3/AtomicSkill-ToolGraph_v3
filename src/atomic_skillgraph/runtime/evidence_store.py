"""Revision-aware executable grounding evidence."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Callable

from ..core.bindings import (
    BindingExprKind, BindingExpression, EvidenceStability, GroundingConstraint,
    GroundingConstraintKind, GroundingEvidence,
)
from ..harness.protocol import HarnessActionSpec


class GroundingEvidenceStore:
    def __init__(self, *, on_change: Callable[[str, GroundingEvidence, int], None] | None = None) -> None:
        self._evidence: dict[str, GroundingEvidence] = {}
        self._custom_verifiers: dict[str, Callable[[GroundingConstraint, dict[str, Any], int], bool]] = {}
        self.revision = 0
        self._on_change = on_change

    def register_verifier(self, verifier_id: str, callback: Callable[[GroundingConstraint, dict[str, Any], int], bool]) -> None:
        self._custom_verifiers[verifier_id] = callback

    def _add(self, evidence: GroundingEvidence) -> GroundingEvidence:
        self._evidence[evidence.evidence_id] = evidence
        if self._on_change:
            self._on_change("add", evidence, self.revision)
        return evidence

    def replace_action_catalog(self, catalog: list[HarnessActionSpec], revision: int) -> None:
        old_revision = self.revision
        self.invalidate_after_transition(old_revision, revision)
        self.revision = int(revision)
        for spec in catalog:
            for role, value in spec.arguments.items():
                self._add(GroundingEvidence(
                    evidence_id=f"entity:{revision}:{role}:{value}:{spec.action_id}", evidence_type="entity_concrete",
                    payload={"role": role, "value": value, "action_type": spec.action_type}, source="action_catalog",
                    observed_at_revision=revision, valid_from_revision=revision,
                    stability=EvidenceStability.REVISION_SCOPED, action_id=spec.action_id,
                ))
            self._add(GroundingEvidence(
                evidence_id=f"affordance:{revision}:{spec.action_id}", evidence_type="harness_affordance",
                payload={"action_type": spec.action_type, "arguments": dict(spec.arguments)}, source="action_catalog",
                observed_at_revision=revision, valid_from_revision=revision,
                stability=EvidenceStability.REVISION_SCOPED, action_id=spec.action_id,
            ))

    def add_task_evidence(self, role: str, value: Any, *, semantic_type: str = "entity") -> GroundingEvidence:
        return self._add(GroundingEvidence(
            evidence_id=f"task:{role}:{uuid.uuid4().hex}", evidence_type="task_binding",
            payload={"role": role, "value": value, "semantic_type": semantic_type}, source="task",
            observed_at_revision=self.revision, valid_from_revision=self.revision,
            stability=EvidenceStability.PERSISTENT,
        ))

    def add_validated_tool_output(self, role: str, value: Any, validation_refs: list[str]) -> GroundingEvidence:
        return self._add(GroundingEvidence(
            evidence_id=f"tool_output:{role}:{uuid.uuid4().hex}", evidence_type="validated_tool_output",
            payload={"role": role, "value": value, "validation_refs": list(validation_refs)}, source="tool_output",
            observed_at_revision=self.revision, valid_from_revision=self.revision,
            stability=EvidenceStability.STATE_SCOPED,
        ))

    def invalidate_after_transition(self, old_revision: int, new_revision: int) -> None:
        for evidence in self._evidence.values():
            if evidence.invalidated_at_revision is not None:
                continue
            if (
                evidence.stability in {
                    EvidenceStability.REVISION_SCOPED,
                    EvidenceStability.STATE_SCOPED,
                }
                and evidence.observed_at_revision <= old_revision
            ):
                evidence.invalidated_at_revision = new_revision
                if self._on_change:
                    self._on_change("invalidate", evidence, new_revision)

    def _constraint_values(self, constraint: GroundingConstraint, bindings: dict[str, Any]) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        for argument, expression in constraint.argument_mapping.items():
            expression = BindingExpression.from_dict(expression)
            if expression.kind is BindingExprKind.CONSTANT:
                value = expression.constant
            elif expression.kind is BindingExprKind.SKILL_INPUT:
                value = bindings.get(expression.source_role)
            else:
                value = bindings.get(expression.source_role)
            if value is None:
                return None
            values[argument] = value
        return values

    def match_constraint(
        self, constraint: GroundingConstraint, bindings: dict[str, Any], revision: int,
    ) -> list[GroundingEvidence]:
        constraint = constraint if isinstance(constraint, GroundingConstraint) else GroundingConstraint(**constraint)
        values = self._constraint_values(constraint, bindings)
        valid = [item for item in self._evidence.values() if item.valid_at(revision)]
        if constraint.kind is GroundingConstraintKind.ARGUMENT_EXISTS:
            if values is None:
                return []
            matched = [item for item in valid if item.evidence_type in {"entity_concrete", "task_binding", "validated_tool_output"} and item.payload.get("value") in values.values()]
            return matched if len(matched) >= len(set(values.values())) else []
        if constraint.kind is GroundingConstraintKind.ARGUMENT_CONCRETE:
            if values is None:
                return []
            matched = [
                item for item in valid
                if item.evidence_type in {"entity_concrete", "validated_tool_output"}
                and item.payload.get("value") in values.values()
            ]
            return matched if len({item.payload.get("value") for item in matched}) >= len(set(values.values())) else []
        if constraint.kind is GroundingConstraintKind.HARNESS_AFFORDANCE:
            if values is None:
                return []
            return [
                item for item in valid
                if item.evidence_type == "harness_affordance"
                and item.payload.get("action_type") == constraint.action_type
                and all(item.payload.get("arguments", {}).get(role) == value for role, value in values.items())
            ]
        if constraint.kind in {GroundingConstraintKind.CURRENT_CONTEXT, GroundingConstraintKind.CUSTOM_ADAPTER}:
            verifier = self._custom_verifiers.get(constraint.verifier_id)
            if verifier and verifier(constraint, values or bindings, revision):
                synthetic = GroundingEvidence(
                    evidence_id=f"custom:{constraint.constraint_id}:{revision}", evidence_type="custom_verifier",
                    payload={"constraint_id": constraint.constraint_id}, source=constraint.verifier_id,
                    observed_at_revision=revision, valid_from_revision=revision,
                    stability=EvidenceStability.REVISION_SCOPED,
                )
                return [synthetic]
        return []

    def active(self, revision: int | None = None) -> list[GroundingEvidence]:
        revision = self.revision if revision is None else revision
        return [item for item in self._evidence.values() if item.valid_at(revision)]

    def get(self, evidence_id: str) -> GroundingEvidence:
        return self._evidence[evidence_id]
