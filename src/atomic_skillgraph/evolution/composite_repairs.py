"""Concrete replay/admission closure for Composite ``revise_sequence``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from ..core.bindings import BindingExpression
from ..core.contracts import (
    CompositeOccurrence,
    CompositeSkill,
    IdentityConstraint,
    SemanticPredicate,
    TaskContract,
)
from ..core.edges import GlobalRelationType, GraphEdge, GraphEdgeType
from ..core.refs import SkillRef, bump_version, content_hash
from ..core.serialization import to_primitive
from ..core.status import SkillStatus
from ..knowledge.skill_registry import SkillRegistry
from .repair import RepairProposal, RepairStore
from .typed_repairs import LineageRecord, RepairEvidence, SemanticRepairRejection


@dataclass(frozen=True)
class CompositeRepairResult:
    proposal: RepairProposal
    admitted_ref: str = ""
    lineage: tuple[LineageRecord, ...] = ()


def _predicate(value: SemanticPredicate | Mapping[str, Any]) -> SemanticPredicate:
    if isinstance(value, SemanticPredicate):
        return value
    args = {
        str(key): BindingExpression.from_dict(item)
        if isinstance(item, dict) and "kind" in item else item
        for key, item in dict(value.get("args") or {}).items()
    }
    return SemanticPredicate(
        str(value["predicate"]), args,
        int(value.get("cardinality", 1)), str(value.get("distinct_by", "")),
        str(value.get("effect_domain", "world")),
    )


def _composite(value: CompositeSkill | Mapping[str, Any]) -> CompositeSkill:
    if isinstance(value, CompositeSkill):
        return value
    ref = SkillRef.from_dict(dict(value["ref"])) if isinstance(value["ref"], dict) else SkillRef.parse(value["ref"])
    contract = dict(value.get("goal_contract") or {})
    return CompositeSkill(
        ref=ref,
        summary=str(value["summary"]),
        occurrences=[
            CompositeOccurrence(
                step_id=str(item["step_id"]),
                occurrence_id=str(item["occurrence_id"]),
                node_ref=SkillRef.from_dict(dict(item["node_ref"]))
                if isinstance(item["node_ref"], dict) else SkillRef.parse(item["node_ref"]),
                binding_specs={
                    str(key): BindingExpression.from_dict(raw)
                    for key, raw in dict(item.get("binding_specs") or {}).items()
                },
            )
            for item in value.get("occurrences", ())
        ],
        control_sequence=[str(item) for item in value.get("control_sequence", ())],
        data_edges=[GraphEdge(**dict(item)) for item in value.get("data_edges", ())],
        dependency_edges=[GraphEdge(**dict(item)) for item in value.get("dependency_edges", ())],
        goal_contract=TaskContract(
            target_effects=[_predicate(item) for item in contract.get("target_effects", ())],
            cardinality_constraints=[dict(item) for item in contract.get("cardinality_constraints", ())],
            identity_constraints=[IdentityConstraint(**dict(item)) for item in contract.get("identity_constraints", ())],
            source=contract.get("source", "planner_proposed"),
            confidence=float(contract.get("confidence", 0.0)),
            validator_id=str(contract.get("validator_id", "")),
        ),
        guideline=dict(value.get("guideline") or {}),
        insight=dict(value.get("insight") or {}),
        validator_spec=dict(value.get("validator_spec") or {}),
        metadata=dict(value.get("metadata") or {}),
        status=SkillStatus(value.get("status", SkillStatus.DRAFT)),
    )


def _passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return bool(value.get("passed"))
    return bool(getattr(value, "passed", False))


def _details(value: Any) -> Any:
    try:
        return to_primitive(value)
    except (TypeError, ValueError):
        return {"type": type(value).__name__}


def _edge_semantics(edge: GraphEdge) -> dict[str, Any]:
    return {
        "edge_type": edge.edge_type.value,
        "source_step": edge.source_step,
        "target_step": edge.target_step,
        "source_role": edge.source_role,
        "target_role": edge.target_role,
        "origin": edge.origin,
        "existing_edge_id": edge.existing_edge_id,
        "evidence_refs": edge.evidence_refs,
    }


def _sorted_edge_semantics(edges: Sequence[GraphEdge]) -> list[dict[str, Any]]:
    return sorted(
        (_edge_semantics(item) for item in edges),
        key=lambda item: content_hash(item),
    )


def _non_sequence_semantics(value: CompositeSkill) -> str:
    return content_hash({
        "summary": value.summary,
        "occurrences": sorted(
            (to_primitive(item) for item in value.occurrences),
            key=lambda item: (item["step_id"], item["occurrence_id"]),
        ),
        "data_edges": _sorted_edge_semantics(value.data_edges),
        "dependency_edges": _sorted_edge_semantics([
            item for item in value.dependency_edges
            if item.edge_type is not GraphEdgeType.NEXT
        ]),
        "goal_contract": value.goal_contract,
        "guideline": value.guideline,
        "insight": value.insight,
        "validator_spec": value.validator_spec,
        "metadata": value.metadata,
    })


def _sequence_semantics(value: CompositeSkill) -> str:
    return content_hash({
        "control_sequence": value.control_sequence,
        "next_edges": _sorted_edge_semantics([
            item for item in value.dependency_edges
            if item.edge_type is GraphEdgeType.NEXT
        ]),
    })


class CompositeSequenceRepairEngine:
    """Admit one immutable Composite sequence revision."""

    _SCHEMA = "composite.revise_sequence.v1"

    def __init__(self, store: RepairStore, skills: SkillRegistry) -> None:
        self.store = store
        self.skills = skills

    @classmethod
    def build_proposal(
        cls,
        source_ref: SkillRef | str,
        replacement: CompositeSkill | Mapping[str, Any],
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        """Parse an agent replacement under code-owned target/evidence authority."""
        return cls.propose(
            source_ref,
            _composite(replacement),
            evidence,
            source_failure_ids,
        )

    @classmethod
    def propose(
        cls,
        source_ref: SkillRef | str,
        replacement: CompositeSkill,
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        return RepairProposal.create(
            str(SkillRef.parse(source_ref)),
            "composite",
            "revise_composite_sequence",
            {
                "typed_schema": cls._SCHEMA,
                "replacement": to_primitive(replacement),
                "source_cases": [
                    to_primitive(RepairEvidence.from_value(item)) for item in evidence
                ],
                "requires_concrete_patch": False,
            },
            list(source_failure_ids),
        )

    def execute(
        self,
        proposal: RepairProposal,
        *,
        replay: Callable[[CompositeSkill, dict[str, Any]], bool],
        validate: Callable[[CompositeSkill], Any],
        admit: Callable[[CompositeSkill], Any],
    ) -> CompositeRepairResult:
        if proposal.status not in {"proposed", "replaying"}:
            raise ValueError("only proposed/replaying Composite repairs can execute")
        if proposal.status == "proposed":
            self.store.save(proposal)
        proposal.status = "replaying"
        self.store.save(proposal)
        try:
            source, candidate, evidence = self._materialize_and_gate(proposal)
            replays = []
            for item in evidence:
                passed = bool(replay(candidate, dict(item.replay_case)))
                detail = {
                    "evidence_id": item.evidence_id,
                    "task_id": item.task_id,
                    "trace_id": item.trace_id,
                    "passed": passed,
                }
                replays.append(detail)
                if not passed:
                    raise SemanticRepairRejection(
                        "source_replay_failed", "Composite source replay failed",
                        details=detail,
                    )
            validation = validate(candidate)
            if not _passed(validation):
                raise SemanticRepairRejection(
                    "planner_validator_rejected", "PlannerValidator rejected Composite sequence",
                    details={"validation": _details(validation)},
                )
            decision = admit(candidate)
            admitted = decision if isinstance(decision, CompositeSkill) else candidate
            if not isinstance(decision, CompositeSkill) and not _passed(decision):
                raise SemanticRepairRejection(
                    "admission_rejected", "Composite admission rejected",
                    details={"admission": _details(decision)},
                )
            if admitted.ref != candidate.ref or _non_sequence_semantics(admitted) != _non_sequence_semantics(candidate) or _sequence_semantics(admitted) != _sequence_semantics(candidate):
                raise SemanticRepairRejection(
                    "admission_semantic_mutation", "admission changed immutable Composite semantics",
                )
            if admitted.status is not SkillStatus.CANDIDATE:
                raise SemanticRepairRejection(
                    "admission_rejected", "admission did not return Candidate",
                )
            self.skills.register_composite(admitted)
            lineage = LineageRecord(
                str(admitted.ref), str(source.ref), GlobalRelationType.DERIVED_FROM,
                proposal.operation, proposal.proposal_id,
            )
            proposal.status = "admitted"
            proposal.replay_result = {
                "passed": True,
                "source_replays": replays,
                "validation": _details(validation),
                "admitted_ref": str(admitted.ref),
                "lineage": [to_primitive(lineage)],
            }
            self.store.save(proposal)
            return CompositeRepairResult(proposal, str(admitted.ref), (lineage,))
        except SemanticRepairRejection as exc:
            proposal.status = "rejected"
            proposal.replay_result = {
                "passed": False,
                "failure_code": exc.code,
                "message": str(exc),
                "details": exc.details,
            }
            self.store.save(proposal)
            return CompositeRepairResult(proposal)

    def _materialize_and_gate(
        self, proposal: RepairProposal,
    ) -> tuple[CompositeSkill, CompositeSkill, list[RepairEvidence]]:
        if (
            proposal.target_layer != "composite"
            or proposal.operation != "revise_composite_sequence"
            or proposal.proposed_patch.get("typed_schema") != self._SCHEMA
        ):
            raise SemanticRepairRejection("typed_proposal_invalid", "Composite operation/layer/schema mismatch")
        try:
            source = self.skills.get_composite(proposal.target_ref)
        except KeyError as exc:
            raise SemanticRepairRejection("repair_target_missing", str(exc)) from exc
        try:
            candidate = _composite(proposal.proposed_patch["replacement"])
            evidence = [
                RepairEvidence.from_value(item)
                for item in proposal.proposed_patch.get("source_cases", ())
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise SemanticRepairRejection("typed_proposal_invalid", str(exc)) from exc
        candidate = replace(
            candidate,
            ref=self._next_ref(source.ref.logical_id),
            status=SkillStatus.CANDIDATE,
        )
        self._sequence_gate(source, candidate)
        self._evidence_gate(evidence)
        return source, candidate, evidence

    def _next_ref(self, logical_id: str) -> SkillRef:
        versions = [
            ref.version for ref in self.skills.list_refs("composite")
            if ref.logical_id == logical_id
        ]
        if not versions:
            return SkillRef(logical_id, "1.0.0")
        try:
            latest = max(versions, key=lambda version: tuple(int(item) for item in version.split(".")))
            return SkillRef(logical_id, bump_version(latest))
        except ValueError as exc:
            raise SemanticRepairRejection("semantic_version_required", str(exc)) from exc

    @staticmethod
    def _sequence_gate(source: CompositeSkill, candidate: CompositeSkill) -> None:
        source_steps = [item.step_id for item in source.occurrences]
        candidate_steps = list(candidate.control_sequence)
        if len(source_steps) != len(set(source_steps)):
            raise SemanticRepairRejection("source_composite_invalid", "source occurrence step ids are duplicated")
        if len(candidate_steps) != len(set(candidate_steps)):
            raise SemanticRepairRejection("duplicate_sequence_step", "replacement sequence has duplicate steps")
        if set(candidate_steps) != set(source_steps) or len(candidate_steps) != len(source_steps):
            raise SemanticRepairRejection("missing_or_unknown_sequence_step", "replacement sequence must contain every source step exactly once")
        if _non_sequence_semantics(candidate) != _non_sequence_semantics(source):
            raise SemanticRepairRejection(
                "sequence_revision_scope_invalid",
                "Composite revise_sequence changed non-sequence semantics",
            )
        if _sequence_semantics(candidate) == _sequence_semantics(source):
            raise SemanticRepairRejection("semantic_edit_empty", "Composite sequence did not change")
        positions = {step: index for index, step in enumerate(candidate_steps)}
        for edge in [*candidate.data_edges, *candidate.dependency_edges]:
            if edge.source_step not in positions or edge.target_step not in positions:
                raise SemanticRepairRejection("edge_endpoint_invalid", "edge references an unknown step")
            if positions[edge.source_step] >= positions[edge.target_step]:
                raise SemanticRepairRejection("edge_order_invalid", "edge violates replacement control order")
        next_pairs: set[tuple[str, str]] = set()
        for edge in candidate.dependency_edges:
            if edge.edge_type is not GraphEdgeType.NEXT:
                continue
            pair = (edge.source_step, edge.target_step)
            if pair in next_pairs:
                raise SemanticRepairRejection("duplicate_next_edge", "replacement has duplicate NEXT edges")
            next_pairs.add(pair)
            if positions[edge.target_step] != positions[edge.source_step] + 1:
                raise SemanticRepairRejection("edge_order_invalid", "NEXT edge endpoints are not adjacent")

    @staticmethod
    def _evidence_gate(evidence: Sequence[RepairEvidence]) -> None:
        if any(
            item.agent_parameter_error
            or item.failure_layer == "runtime_agent"
            or item.failure_code.startswith("runtime_agent_")
            for item in evidence
        ):
            raise SemanticRepairRejection("agent_parameter_error_not_composite_evidence", "Agent error cannot justify Composite evolution")
        if len({item.evidence_id for item in evidence}) < 2:
            raise SemanticRepairRejection("independent_support_insufficient", "Composite revision requires two evidence records")
        if len({item.task_id for item in evidence}) < 2 or len({item.trace_id for item in evidence}) < 2:
            raise SemanticRepairRejection("independent_support_insufficient", "Composite evidence must come from independent tasks/traces")
        clusters = {item.cluster_key for item in evidence}
        if "" in clusters or len(clusters) != 1:
            raise SemanticRepairRejection("heterogeneous_structural_evidence", "Composite evidence must form one structural cluster")


__all__ = ["CompositeRepairResult", "CompositeSequenceRepairEngine"]
