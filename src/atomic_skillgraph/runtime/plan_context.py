"""Policy-safe downstream context derived from the formal Runtime plan.

This module is deliberately an interpreter of already-validated plan and
Atomic contracts.  It does not inspect a task adapter, a benchmark task type,
an implementation body, or validator-private state, and it never proposes a
concrete Runtime binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..core.bindings import (
    BindingExprKind,
    BindingExpression,
    BindingSource,
    BindingStatus,
)
from ..core.contracts import AbstractAtomicSkill, ParameterSpec
from ..core.results import RuntimeLinearPlan, RuntimeOccurrence
from ..core.serialization import to_primitive
from ..core.status import SkillStatus
from .binding_store import RuntimeBindingStore


class AtomicContractResolver(Protocol):
    """The narrow verified-contract surface required by this builder."""

    def get_atomic(self, ref: Any) -> AbstractAtomicSkill:
        """Return the registered Atomic contract identified by ``ref``."""


@dataclass(frozen=True)
class RuntimeConsumerObligation:
    producer_step: str
    producer_output_role: str
    edge_id: str
    consumer_step: str
    consumer_input_role: str
    consumer_summary: str
    consumer_input_contract: dict[str, Any]
    consumer_preconditions: tuple[dict[str, Any], ...]
    consumer_effects: tuple[dict[str, Any], ...]
    consumer_known_semantic_anchors: dict[str, dict[str, Any]]

    def policy_view(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class RuntimePlanPolicyContext:
    current_step: str
    output_obligations: tuple[RuntimeConsumerObligation, ...]
    remaining_method_outline: tuple[dict[str, str], ...]

    def policy_view(self) -> dict[str, Any]:
        return to_primitive(self)


class RuntimePlanContextBuilder:
    """Build downstream intent without becoming a Runtime decision maker."""

    _VERIFIED_STATUSES = frozenset({SkillStatus.CANDIDATE, SkillStatus.ACTIVE})

    def __init__(self, skills: AtomicContractResolver) -> None:
        self.skills = skills

    def _verified_atomic(
        self,
        occurrence: RuntimeOccurrence,
    ) -> AbstractAtomicSkill | None:
        try:
            atomic = self.skills.get_atomic(occurrence.node_ref)
        except KeyError:
            # Ephemeral cold-start contracts are not registered verified
            # assets.  Fail closed instead of consulting their provisional
            # payload or any implementation body.
            return None
        if atomic.status not in self._VERIFIED_STATUSES:
            return None
        return atomic

    @staticmethod
    def _parameter_contract(parameter: ParameterSpec) -> dict[str, Any]:
        return to_primitive(parameter)

    @staticmethod
    def _formal_anchor(
        occurrence: RuntimeOccurrence,
        role: str,
        binding_store: RuntimeBindingStore,
    ) -> dict[str, Any] | None:
        """Resolve only formal Task/DataFlow binding expressions.

        The returned view intentionally omits evidence identities and world
        revisions.  Agent-proposed or incidental Harness bindings are never
        promoted into downstream semantic intent here.
        """

        existing = binding_store.semantic_anchor_for(occurrence, role)
        if existing is None:
            candidate = binding_store.snapshot_for_node(occurrence).get(role)
            if (
                candidate is not None
                and candidate.status is BindingStatus.GROUNDED
                and candidate.source in {
                    BindingSource.HARNESS_EVIDENCE,
                    BindingSource.TOOL_OUTPUT,
                }
            ):
                existing = candidate
        if existing is not None:
            return {
                "value": to_primitive(existing.value),
                "semantic_type": existing.semantic_type,
                "source": existing.source.value,
            }

        raw_expression = occurrence.binding_specs.get(role)
        if raw_expression is None:
            return None
        try:
            expression = BindingExpression.from_dict(raw_expression)
        except (KeyError, TypeError, ValueError):
            return None
        if expression.kind not in {
            BindingExprKind.SKILL_INPUT,
            BindingExprKind.CONSTANT,
            BindingExprKind.DATA_FLOW,
            BindingExprKind.ADAPTER_TRANSFORM,
        }:
            return None
        binding = binding_store.resolve_expression(
            occurrence.occurrence_id,
            expression,
        )
        if binding is None or binding.source not in {
            BindingSource.TASK,
            BindingSource.DATA_FLOW,
            BindingSource.TOOL_OUTPUT,
        }:
            return None
        # A TOOL_OUTPUT value is admissible only through an explicit formal
        # DATA_FLOW expression, in which case the output is validator-backed.
        if (
            binding.source is BindingSource.TOOL_OUTPUT
            and expression.kind is not BindingExprKind.DATA_FLOW
        ):
            return None
        source = (
            BindingSource.DATA_FLOW.value
            if expression.kind is BindingExprKind.DATA_FLOW
            else binding.source.value
        )
        return {
            "value": to_primitive(binding.value),
            "semantic_type": binding.semantic_type,
            "source": source,
        }

    def _known_anchors(
        self,
        occurrence: RuntimeOccurrence,
        atomic: AbstractAtomicSkill,
        binding_store: RuntimeBindingStore,
    ) -> dict[str, dict[str, Any]]:
        anchors: dict[str, dict[str, Any]] = {}
        for parameter in atomic.inputs:
            value = self._formal_anchor(
                occurrence,
                parameter.name,
                binding_store,
            )
            if value is not None:
                anchors[parameter.name] = value
        return anchors

    def build(
        self,
        plan: RuntimeLinearPlan,
        current_step: str,
        binding_store: RuntimeBindingStore,
    ) -> RuntimePlanPolicyContext:
        if current_step not in plan.control_sequence:
            raise KeyError(current_step)

        current_index = plan.control_sequence.index(current_step)
        occurrences = {item.step_id: item for item in plan.occurrences}
        producer_occurrence = occurrences.get(current_step)
        producer_atomic = (
            self._verified_atomic(producer_occurrence)
            if producer_occurrence is not None
            else None
        )
        producer_outputs = {
            item.name for item in producer_atomic.outputs
        } if producer_atomic is not None else set()

        obligations: list[RuntimeConsumerObligation] = []
        for edge in plan.data_edges:
            if edge.source_step != current_step:
                continue
            consumer_occurrence = occurrences.get(edge.target_step)
            if (
                consumer_occurrence is None
                or edge.source_role not in producer_outputs
                or edge.target_step not in plan.control_sequence
                or plan.control_sequence.index(edge.target_step) <= current_index
            ):
                continue
            consumer_atomic = self._verified_atomic(consumer_occurrence)
            if consumer_atomic is None:
                continue
            input_by_role = {
                parameter.name: parameter for parameter in consumer_atomic.inputs
            }
            consumer_input = input_by_role.get(edge.target_role)
            if consumer_input is None:
                continue
            obligations.append(RuntimeConsumerObligation(
                producer_step=current_step,
                producer_output_role=edge.source_role,
                edge_id=edge.edge_id,
                consumer_step=edge.target_step,
                consumer_input_role=edge.target_role,
                consumer_summary=consumer_atomic.summary,
                consumer_input_contract=self._parameter_contract(consumer_input),
                consumer_preconditions=tuple(
                    to_primitive(item) for item in consumer_atomic.preconditions
                ),
                consumer_effects=tuple(
                    to_primitive(item) for item in consumer_atomic.effects
                ),
                consumer_known_semantic_anchors=self._known_anchors(
                    consumer_occurrence,
                    consumer_atomic,
                    binding_store,
                ),
            ))

        # Preserve formal plan order.  The outline contains portable Atomic
        # summaries only; refs, implementation bodies, and concrete guesses
        # are intentionally absent.
        outline: list[dict[str, str]] = []
        for step_id in plan.control_sequence[current_index + 1:]:
            occurrence = occurrences.get(step_id)
            atomic = (
                self._verified_atomic(occurrence)
                if occurrence is not None
                else None
            )
            if atomic is not None:
                outline.append({"step_id": step_id, "summary": atomic.summary})

        return RuntimePlanPolicyContext(
            current_step=current_step,
            output_obligations=tuple(obligations),
            remaining_method_outline=tuple(outline),
        )


__all__ = [
    "AtomicContractResolver",
    "RuntimeConsumerObligation",
    "RuntimePlanContextBuilder",
    "RuntimePlanPolicyContext",
]
