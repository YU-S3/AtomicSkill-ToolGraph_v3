"""Validate E2 authority and construct one canonical Composite version."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import (
    CompositeOccurrence, CompositeSkill, IdentityRelation, SemanticPredicate, TaskContract,
)
from ..core.edges import ExistingEdgeEvidence, GraphEdge, GraphEdgeType
from ..core.refs import SkillRef, content_hash
from ..core.status import SkillStatus
from ..validation.contract_matcher import ContractMatcher, ExactContractMatcher
from .atomicizer import CanonicalAtomicOccurrence
from .extractor_session import CompositeExtractionProposal


def _types_compatible(
    source: str, target: str, *, source_role: str = "", target_role: str = "",
) -> bool:
    if source == target:
        return True
    entity_types = {"entity", "object"}
    if source in entity_types and target in entity_types:
        return True
    entity_roles = ("object", "item", "entity", "location", "source", "destination", "station", "light", "tool")
    return {source, target} == {"entity", "string"} and any(
        token in role.casefold()
        for role in (source_role, target_role)
        for token in entity_roles
    )


def _predicate_resolved_args(
    predicate: SemanticPredicate, occurrence: CanonicalAtomicOccurrence,
) -> dict[str, Any]:
    bindings = {**occurrence.input_bindings, **occurrence.output_bindings}
    result: dict[str, Any] = {}
    for name, raw in predicate.args.items():
        if isinstance(raw, dict) and "kind" in raw:
            raw = BindingExpression.from_dict(raw)
        if isinstance(raw, BindingExpression):
            result[name] = raw.constant if raw.kind is BindingExprKind.CONSTANT else bindings.get(raw.source_role)
        else:
            result[name] = raw
    return result


def _contract_covered(
    contract: TaskContract, canonical: list[CanonicalAtomicOccurrence],
    matcher: ContractMatcher,
) -> bool:
    if not contract.target_effects:
        return False
    offered = [
        (effect, occurrence, _predicate_resolved_args(effect, occurrence))
        for occurrence in canonical
        for effect in occurrence.effects
    ]
    matches_by_target: list[list[tuple[SemanticPredicate, CanonicalAtomicOccurrence, dict[str, Any]]]] = []
    for target in contract.target_effects:
        compatible = [
            (effect, occurrence, arguments)
            for effect, occurrence, arguments in offered
            if matcher.covers(target, effect, arguments)
        ]
        if sum(max(1, int(effect.cardinality)) for effect, _, _ in compatible) < max(1, int(target.cardinality)):
            return False
        if target.distinct_by:
            values = {
                arguments.get(target.distinct_by)
                for _, _, arguments in compatible
                if arguments.get(target.distinct_by) is not None
            }
            if len(values) < max(1, int(target.cardinality)):
                return False
        matches_by_target.append(compatible)
    matched_offered = [
        item
        for matches in matches_by_target
        for item in matches
    ]
    for rule in contract.cardinality_constraints:
        predicate = str(rule.get("predicate", ""))
        count = int(rule.get("count", 1))
        role = str(rule.get("distinct_by") or rule.get("role") or "")
        matching = [
            item for item in matched_offered
            if item[0].predicate.casefold() == predicate.casefold()
        ]
        if sum(max(1, int(item[0].cardinality)) for item in matching) < count:
            return False
        if role and len({item[2].get(role) for item in matching if item[2].get(role) is not None}) < count:
            return False
    for constraint in contract.identity_constraints:
        if constraint.relation is IdentityRelation.SAME_AS:
            if constraint.left_role == constraint.right_role:
                witness_sets = [
                    {item[2][constraint.left_role] for item in matches if constraint.left_role in item[2]}
                    for matches in matches_by_target
                ]
                witness_sets = [values for values in witness_sets if values]
                if len(witness_sets) > 1 and not set.intersection(*witness_sets):
                    return False
            else:
                left = {item[2][constraint.left_role] for item in matched_offered if constraint.left_role in item[2]}
                right = {item[2][constraint.right_role] for item in matched_offered if constraint.right_role in item[2]}
                if not left or not right or not left.intersection(right):
                    return False
        elif constraint.relation is IdentityRelation.DISTINCT_FROM:
            left = {item[2][constraint.left_role] for item in matched_offered if constraint.left_role in item[2]}
            right = {item[2][constraint.right_role] for item in matched_offered if constraint.right_role in item[2]}
            if left and right:
                if not any(left_value != right_value for left_value in left for right_value in right):
                    return False
            else:
                distinct_values = {
                    item[2].get(target.distinct_by)
                    for target, matches in zip(contract.target_effects, matches_by_target)
                    if target.distinct_by
                    for item in matches
                    if item[2].get(target.distinct_by) is not None
                }
                if len(distinct_values) < 2:
                    return False
    return True


class CompositeBuilder:
    def validate_and_build(
        self, proposal: CompositeExtractionProposal,
        canonical: list[CanonicalAtomicOccurrence], contract: TaskContract,
        *,
        existing_edge_evidence: list[ExistingEdgeEvidence] | None = None,
        known_edge_ids: set[str] | None = None,
        contract_matcher: ContractMatcher | None = None,
        task_bindings: Mapping[str, Any] | None = None,
    ) -> CompositeSkill:
        evidence_by_id = {
            item.edge_id: item for item in (existing_edge_evidence or [])
        }
        known_edge_ids = set(known_edge_ids or set()) | set(evidence_by_id)
        by_id = {item.occurrence_id: item for item in canonical}
        if len(by_id) != len(canonical):
            raise ValueError("canonical occurrence ids must be unique")
        expected_sequence = [item.occurrence_id for item in canonical]
        if proposal.control_sequence != expected_sequence:
            raise ValueError(
                "E2 control sequence must equal the canonical chronological order"
            )
        matcher = contract_matcher or ExactContractMatcher()
        if not _contract_covered(contract, canonical, matcher):
            raise ValueError("E2 canonical occurrences do not cover the authoritative TaskContract")

        edges: list[GraphEdge] = []
        for raw in proposal.existing_edges:
            edge_id = str(raw.get("existing_edge_id") or raw.get("edge_id") or "")
            if edge_id not in known_edge_ids:
                raise ValueError(f"E2 forged/unknown existing edge: {edge_id}")
            authority = evidence_by_id.get(edge_id)
            if authority is None:
                raise ValueError(f"E2 existing edge lacks authoritative evidence: {edge_id}")
            self._validate_existing_edge(raw, authority, by_id)
            edges.append(self._edge(
                raw,
                origin="existing_active",
                existing_edge_id=edge_id,
                evidence_refs=authority.support_trace_ids,
            ))
        for raw in proposal.new_edges:
            if raw.get("source_step") not in by_id or raw.get("target_step") not in by_id:
                raise ValueError("E2 edge references non-authoritative occurrence")
            source = by_id[str(raw["source_step"])]
            target = by_id[str(raw["target_step"])]
            claimed = (
                str(source.proposed_ref),
                str(target.proposed_ref),
                str(raw.get("edge_type", "")),
                str(raw.get("source_role", "")),
                str(raw.get("target_role", "")),
            )
            if any(
                claimed == (
                    authority.source_step_ref,
                    authority.target_step_ref,
                    authority.edge_type,
                    authority.source_role,
                    authority.target_role,
                )
                for authority in evidence_by_id.values()
            ):
                raise ValueError("E2 re-proposed an authoritative existing edge as new")
            edges.append(self._edge(raw, origin="extractor_validated"))
        position = {item: index for index, item in enumerate(proposal.control_sequence)}
        if any(position[edge.source_step] >= position[edge.target_step] for edge in edges):
            raise ValueError("E2 edge must point forward")
        if len({edge.edge_id for edge in edges}) != len(edges):
            raise ValueError("E2 edge ids must be unique")
        semantics = {
            (edge.edge_type, edge.source_step, edge.target_step, edge.source_role, edge.target_role)
            for edge in edges
        }
        if len(semantics) != len(edges):
            raise ValueError("E2 contains duplicate semantic edges")
        incoming_roles: set[tuple[str, str]] = set()
        for edge in edges:
            self._validate_roles(edge, by_id)
            if edge.edge_type is GraphEdgeType.DATA_FLOW:
                target = (edge.target_step, edge.target_role)
                if target in incoming_roles:
                    raise ValueError(f"E2 DataFlow role has multiple producers: {target}")
                incoming_roles.add(target)

        task_values = dict(task_bindings or {})
        bindings_by_occurrence: dict[str, dict[str, BindingExpression]] = {}
        for occurrence_id, occurrence in by_id.items():
            task_aliases: dict[str, str] = {}
            for target in contract.target_effects:
                for effect in occurrence.effects:
                    arguments = _predicate_resolved_args(effect, occurrence)
                    if not matcher.covers(target, effect, arguments):
                        continue
                    for argument_name, target_value in target.args.items():
                        offered_value = effect.args.get(argument_name)
                        if isinstance(target_value, dict) and "kind" in target_value:
                            target_value = BindingExpression.from_dict(target_value)
                        if isinstance(offered_value, dict) and "kind" in offered_value:
                            offered_value = BindingExpression.from_dict(offered_value)
                        if not (
                            isinstance(offered_value, BindingExpression)
                            and offered_value.kind is BindingExprKind.SKILL_INPUT
                        ):
                            continue
                        if (
                            isinstance(target_value, BindingExpression)
                            and target_value.kind is BindingExprKind.SKILL_INPUT
                        ):
                            task_aliases[offered_value.source_role] = (
                                target_value.source_role
                            )
                            continue
                        matching_task_roles = [
                            role for role, value in task_values.items()
                            if value == target_value
                        ]
                        if argument_name in matching_task_roles:
                            task_aliases[offered_value.source_role] = argument_name
                        elif len(matching_task_roles) == 1:
                            task_aliases[offered_value.source_role] = (
                                matching_task_roles[0]
                            )
            bindings_by_occurrence[occurrence_id] = {
                role: BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role=task_aliases.get(role, role),
                )
                for role in occurrence.input_bindings
            }
        for edge in edges:
            if edge.edge_type is GraphEdgeType.DATA_FLOW:
                bindings_by_occurrence[edge.target_step][edge.target_role] = BindingExpression(
                    BindingExprKind.DATA_FLOW,
                    source_role=edge.source_role,
                    source_step=edge.source_step,
                )
        for target_id in proposal.control_sequence:
            target_position = position[target_id]
            target = by_id[target_id]
            earlier_outputs = {
                (source_id, role): value
                for source_id in proposal.control_sequence[:target_position]
                for role, value in by_id[source_id].output_bindings.items()
            }
            for parameter in target.input_specs:
                if not parameter.required:
                    continue
                expression = bindings_by_occurrence[target_id].get(parameter.name)
                if expression is None:
                    raise ValueError(f"E2 required input has no binding: {target_id}.{parameter.name}")
                matching_sources = [
                    (source_id, role)
                    for (source_id, role), value in earlier_outputs.items()
                    if value == target.input_bindings.get(parameter.name)
                ]
                if matching_sources and expression.kind is not BindingExprKind.DATA_FLOW:
                    raise ValueError(
                        f"E2 reused required input must have explicit DataFlow: {target_id}.{parameter.name}"
                    )
        occurrences = [
            CompositeOccurrence(
                step_id=occurrence_id,
                occurrence_id=occurrence_id,
                node_ref=by_id[occurrence_id].proposed_ref,
                binding_specs=bindings_by_occurrence[occurrence_id],
            )
            for occurrence_id in proposal.control_sequence
        ]
        logical = re.sub(r"[^a-z0-9]+", "_", proposal.summary.casefold()).strip("_")[:40]
        signature = content_hash({
            "sequence": proposal.control_sequence,
            "refs": [str(by_id[item].proposed_ref) for item in proposal.control_sequence],
            "edges": [
                (edge.edge_type.value, edge.source_step, edge.target_step, edge.source_role, edge.target_role)
                for edge in edges
            ],
            "contract": contract,
        })[:12]
        return CompositeSkill(
            SkillRef(f"composite_{logical}_{signature}", "1.0.0"), proposal.summary,
            occurrences, list(proposal.control_sequence),
            [edge for edge in edges if edge.edge_type is GraphEdgeType.DATA_FLOW],
            [edge for edge in edges if edge.edge_type is GraphEdgeType.REQUIRES_SKILL],
            contract, proposal.guideline, proposal.insight,
            {
                "canonical_sequence": True,
                "self_sufficiency_required": True,
                "task_contract_covered": True,
            },
            {"source_trace_ids": sorted({item.source_trace_id for item in canonical})}, SkillStatus.CANDIDATE,
        )

    def _edge(
        self,
        raw: dict[str, Any],
        *,
        origin: str,
        existing_edge_id: str = "",
        evidence_refs: tuple[str, ...] = (),
    ) -> GraphEdge:
        edge_type = GraphEdgeType(str(raw.get("edge_type", "data_flow")))
        if edge_type not in {GraphEdgeType.DATA_FLOW, GraphEdgeType.REQUIRES_SKILL}:
            raise ValueError("E2 edge must be data_flow or requires_skill; control order is canonical")
        if not raw.get("source_step") or not raw.get("target_step"):
            raise ValueError("E2 edge endpoints are required")
        edge_id = str(raw.get("edge_id") or content_hash(raw)[:20])
        return GraphEdge(
            edge_id, edge_type, str(raw["source_step"]), str(raw["target_step"]),
            str(raw.get("source_role", "")), str(raw.get("target_role", "")),
            origin, existing_edge_id, tuple(evidence_refs),
        )

    @staticmethod
    def _validate_existing_edge(
        raw: dict[str, Any],
        authority: ExistingEdgeEvidence,
        by_id: dict[str, CanonicalAtomicOccurrence],
    ) -> None:
        source_step = str(raw.get("source_step", ""))
        target_step = str(raw.get("target_step", ""))
        if source_step not in by_id or target_step not in by_id:
            raise ValueError("E2 existing edge references non-authoritative occurrence")
        claimed = (
            str(raw.get("edge_type", "")),
            str(raw.get("source_role", "")),
            str(raw.get("target_role", "")),
        )
        expected = (authority.edge_type, authority.source_role, authority.target_role)
        if claimed != expected:
            raise ValueError(f"E2 existing edge semantics differ from authority: {authority.edge_id}")
        if (
            str(by_id[source_step].proposed_ref) != authority.source_step_ref
            or str(by_id[target_step].proposed_ref) != authority.target_step_ref
        ):
            raise ValueError(f"E2 existing edge endpoints differ from authority: {authority.edge_id}")

    @staticmethod
    def _validate_roles(
        edge: GraphEdge, by_id: dict[str, CanonicalAtomicOccurrence],
    ) -> None:
        source = by_id[edge.source_step]
        target = by_id[edge.target_step]
        source_outputs = {item.name: item for item in source.output_specs}
        target_inputs = {item.name: item for item in target.input_specs}
        if edge.edge_type is GraphEdgeType.DATA_FLOW:
            if edge.source_role not in source_outputs or edge.target_role not in target_inputs:
                raise ValueError("E2 DataFlow roles do not close over canonical I/O")
            if (
                source.output_bindings.get(edge.source_role)
                != target.input_bindings.get(edge.target_role)
            ):
                raise ValueError("E2 DataFlow violates concrete binding identity")
            if not _types_compatible(
                source_outputs[edge.source_role].semantic_type,
                target_inputs[edge.target_role].semantic_type,
                source_role=edge.source_role,
                target_role=edge.target_role,
            ):
                raise ValueError("E2 DataFlow semantic types are incompatible")
        elif bool(edge.source_role) != bool(edge.target_role):
            raise ValueError("E2 dependency roles must be both present or both absent")
        elif edge.source_role:
            if edge.source_role not in source_outputs or edge.target_role not in target_inputs:
                raise ValueError("E2 dependency roles do not close over canonical I/O")
