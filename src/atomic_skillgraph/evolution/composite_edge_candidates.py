"""Deterministic, minimal E2 edge-candidate authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import SemanticPredicate
from ..core.edges import ExistingEdgeEvidence, GraphEdgeType
from ..core.refs import content_hash
from ..core.semantic_types import semantic_types_compatible
from ..validation.contract_matcher import ContractMatcher, ExactContractMatcher

if TYPE_CHECKING:
    from .atomicizer import CanonicalAtomicOccurrence


@dataclass(frozen=True)
class CompositeEdgeCandidate:
    candidate_id: str
    edge_type: str
    source_step: str
    target_step: str
    source_role: str
    target_role: str
    authority: str


def _resolved_args(
    predicate: SemanticPredicate,
    occurrence: CanonicalAtomicOccurrence,
) -> dict[str, Any]:
    bindings = {**occurrence.input_bindings, **occurrence.output_bindings}
    result: dict[str, Any] = {}
    for role, raw in predicate.args.items():
        if isinstance(raw, dict) and "kind" in raw:
            raw = BindingExpression.from_dict(raw)
        if isinstance(raw, BindingExpression):
            result[role] = (
                raw.constant
                if raw.kind is BindingExprKind.CONSTANT
                else bindings.get(raw.source_role)
            )
        else:
            result[role] = raw
    return result


def _candidate(
    edge_type: str,
    source_step: str,
    target_step: str,
    source_role: str,
    target_role: str,
    authority: str,
) -> CompositeEdgeCandidate:
    identity = {
        "edge_type": edge_type,
        "source_step": source_step,
        "target_step": target_step,
        "source_role": source_role,
        "target_role": target_role,
    }
    return CompositeEdgeCandidate(
        candidate_id="candidate_" + content_hash(identity)[:24],
        edge_type=edge_type,
        source_step=source_step,
        target_step=target_step,
        source_role=source_role,
        target_role=target_role,
        authority=authority,
    )


class CompositeEdgeCandidateBuilder:
    """Build only edges whose endpoints and semantic authority code proves."""

    def build(
        self,
        occurrences: list[CanonicalAtomicOccurrence],
        *,
        matcher: ContractMatcher | None = None,
    ) -> tuple[CompositeEdgeCandidate, ...]:
        matcher = matcher or ExactContractMatcher()
        result: dict[str, CompositeEdgeCandidate] = {}

        # A target input may have only its nearest preceding identity producer
        # offered as a DataFlow choice.
        for target_index, target in enumerate(occurrences):
            target_inputs = {item.name: item for item in target.input_specs}
            for target_role in sorted(target_inputs):
                target_value = target.input_bindings.get(target_role)
                compatible: list[tuple[int, str]] = []
                for source_index, source in enumerate(
                    occurrences[:target_index]
                ):
                    for source_spec in sorted(
                        source.output_specs, key=lambda item: item.name
                    ):
                        if (
                            source.output_bindings.get(source_spec.name)
                            != target_value
                        ):
                            continue
                        if not semantic_types_compatible(
                            target_inputs[target_role].semantic_type,
                            source_spec.semantic_type,
                        ):
                            continue
                        compatible.append((source_index, source_spec.name))
                if compatible:
                    source_index, source_role = sorted(
                        compatible,
                        key=lambda item: (item[0], item[1]),
                    )[-1]
                    item = _candidate(
                        GraphEdgeType.DATA_FLOW.value,
                        occurrences[source_index].occurrence_id,
                        target.occurrence_id,
                        source_role,
                        target_role,
                        "binding_identity_match",
                    )
                    result[item.candidate_id] = item

        # Dependency candidates are admitted only when an earlier Effect can
        # deterministically establish a later Precondition.
        for target_index, target in enumerate(occurrences):
            for source in occurrences[:target_index]:
                compatible = any(
                    max(1, int(effect.cardinality))
                    >= max(1, int(precondition.cardinality))
                    and matcher.covers(
                        SemanticPredicate(
                            precondition.predicate,
                            _resolved_args(precondition, target),
                            precondition.cardinality,
                            precondition.distinct_by,
                        ),
                        effect,
                        _resolved_args(effect, source),
                    )
                    for effect in source.effects
                    for precondition in target.preconditions
                )
                if compatible:
                    item = _candidate(
                        GraphEdgeType.REQUIRES_SKILL.value,
                        source.occurrence_id,
                        target.occurrence_id,
                        "",
                        "",
                        "effect_precondition_compatibility",
                    )
                    result[item.candidate_id] = item

        return tuple(
            result[key]
            for key in sorted(
                result,
                key=lambda key: (
                    next(
                        index
                        for index, occurrence in enumerate(occurrences)
                        if occurrence.occurrence_id
                        == result[key].target_step
                    ),
                    next(
                        index
                        for index, occurrence in enumerate(occurrences)
                        if occurrence.occurrence_id
                        == result[key].source_step
                    ),
                    result[key].edge_type,
                    result[key].target_role,
                    result[key].source_role,
                    key,
                ),
            )
        )

    @staticmethod
    def materialize_candidate(
        candidate: CompositeEdgeCandidate,
    ) -> dict[str, Any]:
        """Create the standard Builder payload without model provenance."""

        structure = {
            "edge_type": candidate.edge_type,
            "source_step": candidate.source_step,
            "target_step": candidate.target_step,
            "source_role": candidate.source_role,
            "target_role": candidate.target_role,
        }
        return {
            "edge_id": "edge_" + content_hash({
                **structure,
                "origin": "extractor_validated",
            })[:24],
            **structure,
        }

    def existing_edge_materializations(
        self,
        occurrences: list[CanonicalAtomicOccurrence],
        evidence: list[ExistingEdgeEvidence],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Map reusable edge evidence to an unambiguous occurrence pair.

        The minimal E2 submission carries only an existing edge ID, so an
        evidence item with multiple possible occurrence mappings is withheld
        instead of asking the model to author endpoints.
        """

        positions = {
            item.occurrence_id: index for index, item in enumerate(occurrences)
        }
        views: list[dict[str, Any]] = []
        materialized: dict[str, dict[str, Any]] = {}
        for authority in sorted(evidence, key=lambda item: item.edge_id):
            mappings: list[tuple[CanonicalAtomicOccurrence, CanonicalAtomicOccurrence]] = []
            for source in occurrences:
                if str(source.proposed_ref) != authority.source_step_ref:
                    continue
                for target in occurrences:
                    if (
                        str(target.proposed_ref) != authority.target_step_ref
                        or positions[source.occurrence_id]
                        >= positions[target.occurrence_id]
                    ):
                        continue
                    if not self._existing_roles_valid(
                        authority, source, target
                    ):
                        continue
                    mappings.append((source, target))
            if len(mappings) != 1:
                continue
            source, target = mappings[0]
            raw = {
                "edge_id": authority.edge_id,
                "existing_edge_id": authority.edge_id,
                "edge_type": authority.edge_type,
                "source_step": source.occurrence_id,
                "target_step": target.occurrence_id,
                "source_role": authority.source_role,
                "target_role": authority.target_role,
            }
            materialized[authority.edge_id] = raw
            views.append({
                "edge_id": authority.edge_id,
                "edge_type": authority.edge_type,
                "source_step": source.occurrence_id,
                "target_step": target.occurrence_id,
                "source_role": authority.source_role,
                "target_role": authority.target_role,
                "authority": "existing_active",
            })
        return views, materialized

    @staticmethod
    def _existing_roles_valid(
        authority: ExistingEdgeEvidence,
        source: CanonicalAtomicOccurrence,
        target: CanonicalAtomicOccurrence,
    ) -> bool:
        if authority.edge_type == GraphEdgeType.DATA_FLOW.value:
            source_specs = {
                item.name: item for item in source.output_specs
            }
            target_specs = {
                item.name: item for item in target.input_specs
            }
            if (
                authority.source_role not in source_specs
                or authority.target_role not in target_specs
                or source.output_bindings.get(authority.source_role)
                != target.input_bindings.get(authority.target_role)
            ):
                return False
            return semantic_types_compatible(
                target_specs[authority.target_role].semantic_type,
                source_specs[authority.source_role].semantic_type,
            )
        if authority.edge_type != GraphEdgeType.REQUIRES_SKILL.value:
            return False
        if bool(authority.source_role) != bool(authority.target_role):
            return False
        if not authority.source_role:
            return True
        return (
            authority.source_role
            in {item.name for item in source.output_specs}
            and authority.target_role
            in {item.name for item in target.input_specs}
        )


__all__ = [
    "CompositeEdgeCandidate",
    "CompositeEdgeCandidateBuilder",
]
