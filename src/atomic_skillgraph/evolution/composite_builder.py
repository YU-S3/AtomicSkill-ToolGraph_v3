"""Validate E2 authority and construct one canonical Composite version."""

from __future__ import annotations

from typing import Any, Mapping

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import (
    CompositeOccurrence, CompositeSkill, SemanticPredicate, TaskContract,
)
from ..core.edges import ExistingEdgeEvidence, GraphEdge, GraphEdgeType
from ..core.refs import SkillRef, content_hash
from ..core.semantic_types import semantic_types_compatible
from ..core.status import SkillStatus
from ..validation.contract_matcher import ContractMatcher, ExactContractMatcher
from .atomicizer import CanonicalAtomicOccurrence
from .contract_canonicalizer import composite_structure_payload
from .extraction_authority import contract_coverage_report
from .extractor_session import CompositeExtractionProposal
from .portability import (
    composite_fallback_summary,
    occurrence_terms,
    portable_guideline_fallback,
    source_forbidden_terms,
    validate_portability,
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


def _identity_input_for_output(
    occurrence: CanonicalAtomicOccurrence,
    output_role: str,
) -> str | None:
    """Return the unique input that carries an output's concrete identity."""

    output_value = occurrence.output_bindings.get(output_role)
    matches = [
        input_role
        for input_role, input_value in occurrence.input_bindings.items()
        if input_value == output_value
    ]
    return matches[0] if len(matches) == 1 else None


class CompositeBuilder:
    def validate_and_build(
        self, proposal: CompositeExtractionProposal,
        canonical: list[CanonicalAtomicOccurrence], contract: TaskContract,
        *,
        existing_edge_evidence: list[ExistingEdgeEvidence] | None = None,
        known_edge_ids: set[str] | None = None,
        contract_matcher: ContractMatcher | None = None,
        task_bindings: Mapping[str, Any] | None = None,
        terminal_certificate: Mapping[str, Any] | None = None,
        source_composite_ref: str = "",
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
        coverage = contract_coverage_report(contract, canonical, matcher)
        terminal_authority = bool(
            terminal_certificate
            and dict(terminal_certificate).get("benchmark_won") is True
        )
        if not coverage.passed and not terminal_authority:
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
        task_anchor_by_input: dict[tuple[str, str], str] = {}
        for occurrence_id, occurrence in by_id.items():
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
                        input_role = offered_value.source_role
                        if input_role not in occurrence.input_bindings:
                            input_role = _identity_input_for_output(
                                occurrence, offered_value.source_role,
                            )
                        if input_role is None:
                            continue
                        task_role: str | None = None
                        if (
                            isinstance(target_value, BindingExpression)
                            and target_value.kind is BindingExprKind.SKILL_INPUT
                        ):
                            task_role = target_value.source_role
                        else:
                            matching_task_roles = [
                                role for role, value in task_values.items()
                                if value == target_value
                            ]
                            if argument_name in matching_task_roles:
                                task_role = argument_name
                            elif len(matching_task_roles) == 1:
                                task_role = matching_task_roles[0]
                        if task_role is None:
                            continue
                        key = (occurrence_id, input_role)
                        previous = task_anchor_by_input.get(key)
                        if previous is not None and previous != task_role:
                            raise ValueError(
                                "conflicting Task anchors cover one occurrence input"
                            )
                        task_anchor_by_input[key] = task_role

        data_edges = [
            edge for edge in edges if edge.edge_type is GraphEdgeType.DATA_FLOW
        ]
        changed = True
        while changed:
            changed = False
            for edge in reversed(data_edges):
                target_task_role = task_anchor_by_input.get(
                    (edge.target_step, edge.target_role)
                )
                if target_task_role is None:
                    continue
                source_input_role = _identity_input_for_output(
                    by_id[edge.source_step], edge.source_role,
                )
                if source_input_role is None:
                    continue
                key = (edge.source_step, source_input_role)
                previous = task_anchor_by_input.get(key)
                if previous is not None and previous != target_task_role:
                    raise ValueError(
                        "conflicting Task anchors propagated through DataFlow"
                    )
                if previous is None:
                    task_anchor_by_input[key] = target_task_role
                    changed = True

        bindings_by_occurrence: dict[str, dict[str, BindingExpression]] = {
            occurrence_id: {} for occurrence_id in by_id
        }
        for edge in data_edges:
            bindings_by_occurrence[edge.target_step][edge.target_role] = BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_role=edge.source_role,
                source_step=edge.source_step,
            )
        for (occurrence_id, input_role), task_role in task_anchor_by_input.items():
            if input_role in bindings_by_occurrence[occurrence_id]:
                continue
            bindings_by_occurrence[occurrence_id][input_role] = BindingExpression(
                BindingExprKind.SKILL_INPUT,
                source_role=task_role,
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
                matching_sources = [
                    (source_id, role)
                    for (source_id, role), value in earlier_outputs.items()
                    if value == target.input_bindings.get(parameter.name)
                ]
                if matching_sources and (
                    expression is None
                    or expression.kind is not BindingExprKind.DATA_FLOW
                ):
                    raise ValueError(
                        f"E2 reused required input must have explicit DataFlow: {target_id}.{parameter.name}"
                    )
                if expression is None:
                    if parameter.runtime_resolvable:
                        continue
                    raise ValueError(
                        "required non-runtime input has no authority: "
                        f"{target_id}.{parameter.name}"
                    )
        binding_origins: dict[str, dict[str, dict[str, str]]] = {}
        for occurrence_id, occurrence in by_id.items():
            origins: dict[str, dict[str, str]] = {}
            specs = {item.name: item for item in occurrence.input_specs}
            for role, parameter in specs.items():
                expression = bindings_by_occurrence[occurrence_id].get(role)
                if expression is not None and expression.kind is BindingExprKind.DATA_FLOW:
                    origins[role] = {
                        "kind": "data_flow",
                        "source_step": expression.source_step,
                        "source_role": expression.source_role,
                    }
                elif expression is not None and expression.kind is BindingExprKind.SKILL_INPUT:
                    origins[role] = {
                        "kind": "task",
                        "source_role": expression.source_role,
                    }
                elif parameter.runtime_resolvable:
                    origins[role] = {"kind": "runtime"}
            binding_origins[occurrence_id] = origins
        occurrences = [
            CompositeOccurrence(
                step_id=occurrence_id,
                occurrence_id=occurrence_id,
                node_ref=by_id[occurrence_id].proposed_ref,
                binding_specs=bindings_by_occurrence[occurrence_id],
            )
            for occurrence_id in proposal.control_sequence
        ]
        canonical_intents = [
            by_id[item].intent for item in proposal.control_sequence
        ]
        concrete_terms = set().union(*(
            occurrence_terms(by_id[item])
            for item in proposal.control_sequence
        ))
        source_terms = set().union(*(
            source_forbidden_terms(by_id[item])
            for item in proposal.control_sequence
        ))
        validator_spec = {
            "canonical_sequence": True,
            "self_sufficiency_required": True,
            "task_contract_covered": bool(coverage.passed),
            "completion_authority": (
                "complete_contract" if coverage.passed else "terminal_empirical"
            ),
        }
        structure = composite_structure_payload(
            occurrences=occurrences,
            control_sequence=list(proposal.control_sequence),
            data_edges=[
                edge for edge in edges
                if edge.edge_type is GraphEdgeType.DATA_FLOW
            ],
            dependency_edges=[
                edge for edge in edges
                if edge.edge_type is GraphEdgeType.REQUIRES_SKILL
            ],
            goal_contract=contract,
            validator_spec=validator_spec,
        )
        signature = content_hash(structure)
        summary_result = validate_portability(
            proposal.summary,
            episode_terms=concrete_terms,
            additional_forbidden_terms=source_terms,
        )
        guideline_result = validate_portability(
            proposal.guideline,
            episode_terms=concrete_terms,
            additional_forbidden_terms=source_terms,
        )
        summary = (
            proposal.summary
            if summary_result.passed
            else composite_fallback_summary(
                canonical_intents,
                structure_digest=signature,
            )
        )
        guideline = (
            proposal.guideline
            if guideline_result.passed
            else portable_guideline_fallback(canonical_intents)
        )
        final_label_violations = sum((
            not validate_portability(
                summary,
                episode_terms=concrete_terms,
                additional_forbidden_terms=source_terms,
            ).passed,
            not validate_portability(
                guideline,
                episode_terms=concrete_terms,
                additional_forbidden_terms=source_terms,
            ).passed,
        ))
        terminal_metadata = dict(terminal_certificate or {}) if terminal_authority else {}
        if terminal_authority:
            # covered_effect_signatures must come exclusively from TaskContract
            # targets this empirical path actually covered.  Internal Support /
            # Evidence Effects stay in the graph but never become a
            # Terminal-Empirical compatibility obligation.
            covered_signatures: list[dict[str, Any]] = []
            for check in coverage.target_checks:
                if not bool(check.get("passed")):
                    continue
                target = contract.target_effects[int(check["target_index"])]
                covered_signatures.append({
                    "predicate": target.predicate,
                    "effect_domain": str(target.effect_domain.value),
                    "argument_roles": sorted(map(str, target.args)),
                    "cardinality": max(1, int(target.cardinality)),
                    "distinct_by": str(target.distinct_by),
                })
            terminal_metadata["covered_effect_signatures"] = covered_signatures
        return CompositeSkill(
            SkillRef(f"composite_{signature[:24]}", "1.0.0"), summary,
            occurrences, list(proposal.control_sequence),
            [edge for edge in edges if edge.edge_type is GraphEdgeType.DATA_FLOW],
            [edge for edge in edges if edge.edge_type is GraphEdgeType.REQUIRES_SKILL],
            contract, guideline, proposal.insight,
            validator_spec,
            {
                "source_trace_ids": sorted({item.source_trace_id for item in canonical}),
                "binding_origins": binding_origins,
                "ordered_canonical_intents": canonical_intents,
                "summary_portability_fallback": not summary_result.passed,
                "guideline_portability_fallback": not guideline_result.passed,
                "artifact_label_concrete_term_violation_count": (
                    final_label_violations
                ),
                "completion_authority": {
                    "kind": (
                        "complete_contract"
                        if coverage.passed
                        else "terminal_empirical"
                    ),
                    "source_composite_ref": str(source_composite_ref),
                },
                "terminal_certificate": terminal_metadata,
                "observed_task_contract_coverage": {
                    "covered_effects": [
                        item.get("predicate")
                        for item in coverage.target_checks
                        if bool(item.get("passed"))
                    ],
                    "uncovered_effects": [
                        item.get("predicate")
                        for item in coverage.target_checks
                        if not bool(item.get("passed"))
                    ],
                } if terminal_authority else {},
            }, SkillStatus.CANDIDATE,
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
            if not semantic_types_compatible(
                target_inputs[edge.target_role].semantic_type,
                source_outputs[edge.source_role].semantic_type,
            ):
                raise ValueError("E2 DataFlow semantic types are incompatible")
        elif bool(edge.source_role) != bool(edge.target_role):
            raise ValueError("E2 dependency roles must be both present or both absent")
        elif edge.source_role:
            if edge.source_role not in source_outputs or edge.target_role not in target_inputs:
                raise ValueError("E2 dependency roles do not close over canonical I/O")
