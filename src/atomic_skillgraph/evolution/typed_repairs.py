"""Replay-gated, immutable Atomic and Implementation evolution operations.

This module deliberately owns only the typed semantic edit.  Trace/Ledger,
lifecycle promotion, and graph persistence remain maintenance concerns; the
engine returns explicit lineage records for those callers to commit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence, TypeAlias

from ..core.bindings import BindingExpression, GroundingConstraint, ToolBinding
from ..core.contracts import AbstractAtomicSkill, ImplementationAtom, ParameterSpec, SemanticPredicate
from ..core.edges import GlobalRelationType
from ..core.refs import SkillRef, bump_version, content_hash
from ..core.serialization import to_primitive
from ..core.status import SkillStatus
from ..knowledge.skill_registry import SkillRegistry
from .repair import RepairProposal, RepairStore


Artifact: TypeAlias = AbstractAtomicSkill | ImplementationAtom
ReplayCallback: TypeAlias = Callable[[Artifact, dict[str, Any]], bool]
ValidationCallback: TypeAlias = Callable[[Artifact], Any]
AdmissionCallback: TypeAlias = Callable[[Artifact], Any]


_SCHEMAS = {
    ("atomic", "revise_atomic_contract"): "atomic.revise.v1",
    ("atomic", "split_atomic"): "atomic.split.v1",
    ("atomic", "merge_atomic"): "atomic.merge.v1",
    ("implementation", "revise_implementation_mapping"): "implementation.revise_mapping.v1",
    ("implementation", "revise_grounding_constraint"): "implementation.revise_constraint.v1",
    ("implementation", "specialize_implementation"): "implementation.specialize.v1",
}


class SemanticRepairRejection(Exception):
    """Expected proposal rejection; never used for infrastructure failures."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class RepairEvidence:
    """One independently identifiable source replay/evidence case."""

    evidence_id: str
    task_id: str
    trace_id: str
    cluster_key: str
    replay_case: dict[str, Any]
    failure_layer: str = ""
    failure_code: str = ""
    agent_parameter_error: bool = False
    candidate_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.task_id or not self.trace_id:
            raise ValueError("repair evidence requires evidence_id, task_id, and trace_id")
        if not isinstance(self.replay_case, dict):
            raise TypeError("repair evidence replay_case must be a dict")
        object.__setattr__(self, "candidate_keys", tuple(self.candidate_keys))

    @classmethod
    def from_value(cls, value: "RepairEvidence | Mapping[str, Any]") -> "RepairEvidence":
        if isinstance(value, cls):
            return value
        return cls(
            evidence_id=str(value.get("evidence_id", "")),
            task_id=str(value.get("task_id", "")),
            trace_id=str(value.get("trace_id", "")),
            cluster_key=str(value.get("cluster_key", "")),
            replay_case=dict(value.get("replay_case") or {}),
            failure_layer=str(value.get("failure_layer", "")),
            failure_code=str(value.get("failure_code", "")),
            agent_parameter_error=bool(value.get("agent_parameter_error", False)),
            candidate_keys=tuple(str(item) for item in value.get("candidate_keys", ())),
        )


@dataclass(frozen=True)
class LineageRecord:
    source_ref: str
    target_ref: str
    relation: GlobalRelationType
    operation: str
    proposal_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", GlobalRelationType(self.relation))


@dataclass(frozen=True)
class TypedRepairResult:
    proposal: RepairProposal
    admitted_refs: tuple[str, ...] = ()
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
        predicate=str(value["predicate"]),
        args=args,
        cardinality=int(value.get("cardinality", 1)),
        distinct_by=str(value.get("distinct_by", "")),
    )


def _parameter(value: ParameterSpec | Mapping[str, Any]) -> ParameterSpec:
    return value if isinstance(value, ParameterSpec) else ParameterSpec(**dict(value))


def _atomic(value: AbstractAtomicSkill | Mapping[str, Any]) -> AbstractAtomicSkill:
    if isinstance(value, AbstractAtomicSkill):
        return value
    ref = SkillRef.from_dict(dict(value["ref"])) if isinstance(value["ref"], dict) else SkillRef.parse(value["ref"])
    return AbstractAtomicSkill(
        ref=ref,
        summary=str(value["summary"]),
        inputs=[_parameter(item) for item in value.get("inputs", ())],
        outputs=[_parameter(item) for item in value.get("outputs", ())],
        preconditions=[_predicate(item) for item in value.get("preconditions", ())],
        effects=[_predicate(item) for item in value.get("effects", ())],
        validator_spec=dict(value.get("validator_spec") or {}),
        failure_modes=[dict(item) for item in value.get("failure_modes", ())],
        guideline=dict(value.get("guideline") or {}),
        metadata=dict(value.get("metadata") or {}),
        status=SkillStatus(value.get("status", SkillStatus.DRAFT)),
    )


def _implementation(value: ImplementationAtom | Mapping[str, Any]) -> ImplementationAtom:
    if isinstance(value, ImplementationAtom):
        return value
    ref = SkillRef.from_dict(dict(value["ref"])) if isinstance(value["ref"], dict) else SkillRef.parse(value["ref"])
    abstract = value["abstract_ref"]
    abstract_ref = SkillRef.from_dict(dict(abstract)) if isinstance(abstract, dict) else SkillRef.parse(abstract)
    return ImplementationAtom(
        ref=ref,
        abstract_ref=abstract_ref,
        tool_bindings=[ToolBinding(**dict(item)) for item in value.get("tool_bindings", ())],
        grounding_constraints=[GroundingConstraint(**dict(item)) for item in value.get("grounding_constraints", ())],
        execution_policy=dict(value.get("execution_policy") or {}),
        compatibility=dict(value.get("compatibility") or {}),
        quality=dict(value.get("quality") or {}),
        status=SkillStatus(value.get("status", SkillStatus.DRAFT)),
    )


def _semantic(value: Artifact) -> str:
    if isinstance(value, AbstractAtomicSkill):
        body = {
            "summary": value.summary,
            "inputs": value.inputs,
            "outputs": value.outputs,
            "preconditions": value.preconditions,
            "effects": value.effects,
            "validator_spec": value.validator_spec,
            "failure_modes": value.failure_modes,
            "guideline": value.guideline,
            "metadata": value.metadata,
        }
    else:
        body = {
            "abstract_ref": str(value.abstract_ref),
            "tool_bindings": value.tool_bindings,
            "grounding_constraints": value.grounding_constraints,
            "execution_policy": value.execution_policy,
            "compatibility": value.compatibility,
        }
    return content_hash(body)


def _atomic_contract(value: AbstractAtomicSkill) -> str:
    return content_hash({
        "inputs": value.inputs,
        "outputs": value.outputs,
        "preconditions": value.preconditions,
        "effects": value.effects,
        "validator_spec": value.validator_spec,
        "failure_modes": value.failure_modes,
    })


def _effect_key(value: SemanticPredicate) -> str:
    return content_hash(value)


def _compatibility_narrows(source: Any, candidate: Any) -> tuple[bool, bool]:
    """Return ``(valid_narrowing, strictly_narrower)`` for compatibility data."""
    if not isinstance(source, Mapping) or not isinstance(candidate, Mapping):
        return source == candidate, False
    if set(source) - set(candidate):
        return False, False
    added_keys = set(candidate) - set(source)
    if any(candidate[key] in (None, "", [], {}, ()) for key in added_keys):
        return False, False
    strict = bool(added_keys)
    for key, source_value in source.items():
        candidate_value = candidate[key]
        if isinstance(source_value, Mapping):
            valid, nested_strict = _compatibility_narrows(source_value, candidate_value)
            if not valid:
                return False, False
            strict = strict or nested_strict
        elif isinstance(source_value, (list, tuple, set)):
            source_set = {content_hash(item) for item in source_value}
            candidate_values = candidate_value if isinstance(candidate_value, (list, tuple, set)) else ()
            candidate_set = {content_hash(item) for item in candidate_values}
            if not candidate_set or not candidate_set <= source_set:
                return False, False
            strict = strict or candidate_set != source_set
        elif candidate_value != source_value:
            return False, False
    return True, strict


def _constraints_strengthen(
    source: Sequence[GroundingConstraint], candidate: Sequence[GroundingConstraint],
) -> tuple[bool, bool]:
    source_by_id = {item.constraint_id: item for item in source}
    candidate_by_id = {item.constraint_id: item for item in candidate}
    if len(source_by_id) != len(source) or len(candidate_by_id) != len(candidate):
        return False, False
    if set(source_by_id) - set(candidate_by_id):
        return False, False
    resolution_rank = {"semantic": 0, "concrete": 1, "relation_verified": 2}
    strict = bool(set(candidate_by_id) - set(source_by_id))
    for constraint_id, before in source_by_id.items():
        after = candidate_by_id[constraint_id]
        if (
            after.kind != before.kind
            or after.action_type != before.action_type
            or content_hash(after.argument_mapping) != content_hash(before.argument_mapping)
            or after.verifier_id != before.verifier_id
        ):
            return False, False
        before_rank = resolution_rank.get(before.required_resolution)
        after_rank = resolution_rank.get(after.required_resolution)
        if before_rank is None or after_rank is None or after_rank < before_rank:
            return False, False
        strict = strict or after_rank > before_rank
    return True, strict


def _result_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return bool(value.get("passed"))
    return bool(getattr(value, "passed", False))


def _result_details(value: Any) -> Any:
    try:
        return to_primitive(value)
    except (TypeError, ValueError):
        return {"type": type(value).__name__}


def _version_key(version: str) -> tuple[int, int, int]:
    try:
        return tuple(int(item) for item in version.split("."))  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise SemanticRepairRejection("semantic_version_required", str(exc)) from exc


class TypedRepairEngine:
    """Execute the §25.6 Atomic/Implementation operations under hard gates."""

    def __init__(self, store: RepairStore, skills: SkillRegistry) -> None:
        self.store = store
        self.skills = skills

    @staticmethod
    def build_proposal(
        operation: str,
        target_refs: Sequence[SkillRef | str],
        replacements: Sequence[Artifact | Mapping[str, Any]],
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        """Build one typed proposal from code-authoritative review inputs."""
        refs = [SkillRef.parse(item) for item in target_refs]
        if not refs:
            raise ValueError("typed repair requires target refs")
        if operation in {"revise_atomic_contract", "split_atomic", "merge_atomic"}:
            candidates = [_atomic(item) for item in replacements]
            if operation == "revise_atomic_contract":
                if len(refs) != 1 or len(candidates) != 1:
                    raise ValueError("Atomic revise requires one target and replacement")
                return TypedRepairEngine.propose_atomic_revision(
                    refs[0], candidates[0], evidence, source_failure_ids,
                )
            if operation == "split_atomic":
                if len(refs) != 1 or len(candidates) < 2:
                    raise ValueError("Atomic split requires one target and multiple replacements")
                return TypedRepairEngine.propose_atomic_split(
                    refs[0], candidates, evidence, source_failure_ids,
                )
            if len(refs) < 2 or len(candidates) != 1:
                raise ValueError("Atomic merge requires multiple targets and one replacement")
            return TypedRepairEngine.propose_atomic_merge(
                refs, candidates[0], evidence, source_failure_ids,
            )

        candidates = [_implementation(item) for item in replacements]
        if len(refs) != 1 or len(candidates) != 1:
            raise ValueError("Implementation repair requires one target and replacement")
        if operation == "revise_implementation_mapping":
            return TypedRepairEngine.propose_implementation_mapping_revision(
                refs[0], candidates[0], evidence, source_failure_ids,
            )
        if operation == "revise_grounding_constraint":
            return TypedRepairEngine.propose_implementation_constraint_revision(
                refs[0], candidates[0], evidence, source_failure_ids,
            )
        if operation == "specialize_implementation":
            return TypedRepairEngine.propose_implementation_specialization(
                refs[0], candidates[0], evidence, source_failure_ids,
            )
        raise ValueError(f"unsupported typed repair operation: {operation}")

    @staticmethod
    def propose_atomic_revision(
        source_ref: SkillRef | str,
        replacement: AbstractAtomicSkill,
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        return TypedRepairEngine._proposal(
            source_ref, "atomic", "revise_atomic_contract", "atomic.revise.v1",
            [replacement], evidence, source_failure_ids,
        )

    @staticmethod
    def propose_atomic_split(
        source_ref: SkillRef | str,
        replacements: Sequence[AbstractAtomicSkill],
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        return TypedRepairEngine._proposal(
            source_ref, "atomic", "split_atomic", "atomic.split.v1",
            replacements, evidence, source_failure_ids,
        )

    @staticmethod
    def propose_atomic_merge(
        source_refs: Sequence[SkillRef | str],
        replacement: AbstractAtomicSkill,
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        refs = [str(SkillRef.parse(item)) for item in source_refs]
        if not refs:
            raise ValueError("atomic merge requires source refs")
        return TypedRepairEngine._proposal(
            refs[0], "atomic", "merge_atomic", "atomic.merge.v1",
            [replacement], evidence, source_failure_ids, target_refs=refs,
        )

    @staticmethod
    def propose_implementation_mapping_revision(
        source_ref: SkillRef | str,
        replacement: ImplementationAtom,
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        return TypedRepairEngine._proposal(
            source_ref, "implementation", "revise_implementation_mapping",
            "implementation.revise_mapping.v1", [replacement], evidence,
            source_failure_ids,
        )

    @staticmethod
    def propose_implementation_constraint_revision(
        source_ref: SkillRef | str,
        replacement: ImplementationAtom,
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        return TypedRepairEngine._proposal(
            source_ref, "implementation", "revise_grounding_constraint",
            "implementation.revise_constraint.v1", [replacement], evidence,
            source_failure_ids,
        )

    @staticmethod
    def propose_implementation_specialization(
        source_ref: SkillRef | str,
        replacement: ImplementationAtom,
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        source_failure_ids: Sequence[str] = (),
    ) -> RepairProposal:
        return TypedRepairEngine._proposal(
            source_ref, "implementation", "specialize_implementation",
            "implementation.specialize.v1", [replacement], evidence,
            source_failure_ids,
        )

    @staticmethod
    def _proposal(
        target_ref: SkillRef | str,
        layer: str,
        operation: str,
        schema: str,
        replacements: Sequence[Artifact],
        evidence: Sequence[RepairEvidence | Mapping[str, Any]],
        failures: Sequence[str],
        *,
        target_refs: Sequence[str] = (),
    ) -> RepairProposal:
        patch = {
            "typed_schema": schema,
            "replacements": [to_primitive(item) for item in replacements],
            "source_cases": [to_primitive(RepairEvidence.from_value(item)) for item in evidence],
            "requires_concrete_patch": False,
        }
        if target_refs:
            patch["target_refs"] = list(target_refs)
        return RepairProposal.create(
            str(SkillRef.parse(target_ref)), layer, operation, patch, list(failures),
        )

    def execute(
        self,
        proposal: RepairProposal,
        *,
        replay: ReplayCallback,
        validate: ValidationCallback,
        admit: AdmissionCallback,
    ) -> TypedRepairResult:
        """Close a proposal as admitted/rejected, or propagate unexpected errors.

        A propagated error intentionally leaves the proposal in ``replaying`` so
        checkpoint resume cannot mistake an interrupted edit for a rejection.
        """
        if proposal.status not in {"proposed", "replaying"}:
            raise ValueError("only proposed/replaying repairs can execute")
        if proposal.status == "proposed":
            self.store.save(proposal)
        proposal.status = "replaying"
        self.store.save(proposal)
        try:
            candidates, sources, cases = self._materialize_and_gate(proposal)
            replay_details = self._replay_all(candidates, cases, replay)
            validation_details = []
            admitted: list[Artifact] = []
            for candidate in candidates:
                validation = validate(candidate)
                validation_details.append(_result_details(validation))
                if not _result_passed(validation):
                    raise SemanticRepairRejection(
                        "validator_rejected", "candidate validator rejected",
                        details={"candidate_ref": str(candidate.ref)},
                    )
                decision = admit(candidate)
                admitted_candidate = decision if isinstance(
                    decision, (AbstractAtomicSkill, ImplementationAtom)
                ) else candidate
                if not _result_passed(decision) and decision is not admitted_candidate:
                    raise SemanticRepairRejection(
                        "admission_rejected", "candidate admission rejected",
                        details={"candidate_ref": str(candidate.ref), "admission": _result_details(decision)},
                    )
                if admitted_candidate.ref != candidate.ref:
                    raise SemanticRepairRejection(
                        "admission_ref_mutation", "admission changed immutable candidate ref",
                    )
                if _semantic(admitted_candidate) != _semantic(candidate):
                    raise SemanticRepairRejection(
                        "admission_semantic_mutation", "admission changed candidate semantics",
                    )
                if admitted_candidate.status is not SkillStatus.CANDIDATE:
                    details = admitted_candidate.quality if isinstance(
                        admitted_candidate, ImplementationAtom
                    ) else admitted_candidate.metadata
                    raise SemanticRepairRejection(
                        "admission_rejected", "admission did not return Candidate",
                        details={"candidate_ref": str(candidate.ref), "admission": details},
                    )
                admitted.append(admitted_candidate)

            # Registration is last: no artifact is visible before every
            # candidate in a split has passed replay, validation, and admission.
            for candidate in admitted:
                self.skills.register(candidate)
            lineage = self._lineage(proposal, admitted, sources)
            proposal.replay_result = {
                "passed": True,
                "source_replays": replay_details,
                "validation": validation_details,
                "admitted_refs": [str(item.ref) for item in admitted],
                "lineage": [to_primitive(item) for item in lineage],
            }
            proposal.status = "admitted"
            self.store.save(proposal)
            return TypedRepairResult(
                proposal,
                tuple(str(item.ref) for item in admitted),
                tuple(lineage),
            )
        except SemanticRepairRejection as exc:
            proposal.status = "rejected"
            proposal.replay_result = {
                "passed": False,
                "failure_code": exc.code,
                "message": str(exc),
                "details": exc.details,
            }
            self.store.save(proposal)
            return TypedRepairResult(proposal)

    def _materialize_and_gate(
        self, proposal: RepairProposal,
    ) -> tuple[list[Artifact], list[Artifact], list[RepairEvidence]]:
        expected_schema = _SCHEMAS.get((proposal.target_layer, proposal.operation))
        if expected_schema is None or proposal.proposed_patch.get("typed_schema") != expected_schema:
            raise SemanticRepairRejection("typed_proposal_invalid", "operation/layer/schema mismatch")
        raw_candidates = proposal.proposed_patch.get("replacements")
        raw_cases = proposal.proposed_patch.get("source_cases")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise SemanticRepairRejection("typed_proposal_invalid", "replacement candidates are required")
        if not isinstance(raw_cases, list):
            raise SemanticRepairRejection("typed_proposal_invalid", "source_cases must be a list")
        try:
            cases = [RepairEvidence.from_value(item) for item in raw_cases]
        except (KeyError, TypeError, ValueError) as exc:
            raise SemanticRepairRejection("source_evidence_invalid", str(exc)) from exc

        target_refs = proposal.proposed_patch.get("target_refs") or [proposal.target_ref]
        if not isinstance(target_refs, list) or not target_refs:
            raise SemanticRepairRejection("typed_proposal_invalid", "target refs are required")
        if proposal.target_ref != str(target_refs[0]):
            raise SemanticRepairRejection("typed_proposal_invalid", "primary target ref mismatch")
        try:
            parsed_refs = [SkillRef.parse(item) for item in target_refs]
            sources: list[Artifact]
            if proposal.target_layer == "atomic":
                sources = [self.skills.get_atomic(ref) for ref in parsed_refs]
                raw = [_atomic(item) for item in raw_candidates]
                kind = "atomic"
            else:
                sources = [self.skills.get_implementation(ref) for ref in parsed_refs]
                raw = [_implementation(item) for item in raw_candidates]
                kind = "implementation"
        except KeyError as exc:
            raise SemanticRepairRejection("repair_target_missing", str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise SemanticRepairRejection("typed_proposal_invalid", str(exc)) from exc

        expected_count = 2 if proposal.operation == "split_atomic" else 1
        if len(raw) < expected_count or (proposal.operation != "split_atomic" and len(raw) != 1):
            raise SemanticRepairRejection("candidate_count_invalid", "operation candidate cardinality is invalid")
        if proposal.operation == "merge_atomic" and len(sources) < 2:
            raise SemanticRepairRejection("merge_sources_insufficient", "merge requires at least two source artifacts")
        if proposal.operation != "merge_atomic" and len(sources) != 1:
            raise SemanticRepairRejection("repair_sources_invalid", "operation requires exactly one source artifact")

        candidates = [
            replace(
                item,
                ref=self._next_ref(kind, self._candidate_logical_id(proposal, item, sources)),
                status=SkillStatus.CANDIDATE,
            )
            for item in raw
        ]
        if len({str(item.ref) for item in candidates}) != len(candidates):
            raise SemanticRepairRejection("candidate_ref_collision", "candidate logical ids must be distinct")
        self._operation_gate(proposal, candidates, sources)
        self._evidence_gate(candidates, cases)
        return candidates, sources, cases

    @staticmethod
    def _candidate_logical_id(
        proposal: RepairProposal, candidate: Artifact, sources: Sequence[Artifact],
    ) -> str:
        source_id = sources[0].ref.logical_id
        if proposal.operation in {
            "revise_atomic_contract", "revise_implementation_mapping",
            "revise_grounding_constraint",
        }:
            return source_id
        if candidate.ref.logical_id in {item.ref.logical_id for item in sources}:
            raise SemanticRepairRejection(
                "specialized_identity_required",
                "split/merge/specialize candidates require a distinct logical id",
            )
        return candidate.ref.logical_id

    def _next_ref(self, kind: str, logical_id: str) -> SkillRef:
        versions = [
            ref.version for ref in self.skills.list_refs(kind)
            if ref.logical_id == logical_id
        ]
        return SkillRef(
            logical_id,
            "1.0.0" if not versions else bump_version(max(versions, key=_version_key)),
        )

    @staticmethod
    def _operation_gate(
        proposal: RepairProposal, candidates: Sequence[Artifact], sources: Sequence[Artifact],
    ) -> None:
        operation = proposal.operation
        candidate = candidates[0]
        source = sources[0]
        if operation in {"revise_atomic_contract", "revise_implementation_mapping", "revise_grounding_constraint"}:
            if _semantic(candidate) == _semantic(source):
                raise SemanticRepairRejection("semantic_edit_empty", "replacement has no semantic change")
        if operation == "revise_atomic_contract":
            assert isinstance(candidate, AbstractAtomicSkill) and isinstance(source, AbstractAtomicSkill)
            if content_hash(candidate.metadata) != content_hash(source.metadata):
                raise SemanticRepairRejection(
                    "atomic_revision_scope_invalid",
                    "Atomic revision cannot rewrite provenance metadata",
                )
            contract_changed = _atomic_contract(candidate) != _atomic_contract(source)
            guideline_changed = content_hash(candidate.guideline) != content_hash(source.guideline)
            if not contract_changed and not guideline_changed:
                raise SemanticRepairRejection(
                    "semantic_edit_empty",
                    "Atomic revision requires a contract or evidence-backed guideline change",
                )
        elif operation == "revise_implementation_mapping":
            assert isinstance(candidate, ImplementationAtom) and isinstance(source, ImplementationAtom)
            if len(candidate.tool_bindings) != len(source.tool_bindings):
                raise SemanticRepairRejection(
                    "mapping_revision_scope_invalid", "mapping revision changed ToolBinding cardinality",
                )
            restored_bindings = []
            for before, after in zip(source.tool_bindings, candidate.tool_bindings, strict=True):
                if (
                    after.tool_ref != before.tool_ref
                    or after.role != before.role
                    or after.order != before.order
                ):
                    raise SemanticRepairRejection(
                        "mapping_revision_scope_invalid",
                        "mapping revision changed Tool ref, role, or order",
                    )
                restored_bindings.append(replace(after, parameter_mapping=before.parameter_mapping))
            candidate_policy = dict(candidate.execution_policy)
            source_output_mapping = source.execution_policy.get("output_mapping")
            if source_output_mapping is None:
                candidate_policy.pop("output_mapping", None)
            else:
                candidate_policy["output_mapping"] = source_output_mapping
            unchanged = replace(
                candidate,
                ref=source.ref,
                tool_bindings=restored_bindings,
                execution_policy=candidate_policy,
            )
            if _semantic(unchanged) != _semantic(source):
                raise SemanticRepairRejection("mapping_revision_scope_invalid", "mapping revision changed non-mapping fields")
        elif operation == "revise_grounding_constraint":
            assert isinstance(candidate, ImplementationAtom) and isinstance(source, ImplementationAtom)
            unchanged = replace(candidate, ref=source.ref, grounding_constraints=source.grounding_constraints)
            if _semantic(unchanged) != _semantic(source):
                raise SemanticRepairRejection("constraint_revision_scope_invalid", "constraint revision changed other fields")
        elif operation == "specialize_implementation":
            assert isinstance(candidate, ImplementationAtom) and isinstance(source, ImplementationAtom)
            if candidate.abstract_ref != source.abstract_ref:
                raise SemanticRepairRejection("specialization_abstract_mismatch", "specialization changed Atomic contract")
            compatibility_valid, compatibility_strict = _compatibility_narrows(
                source.compatibility, candidate.compatibility,
            )
            constraints_valid, constraints_strict = _constraints_strengthen(
                source.grounding_constraints, candidate.grounding_constraints,
            )
            if not compatibility_valid or not constraints_valid:
                raise SemanticRepairRejection(
                    "specialization_domain_broadened",
                    "specialization broadened compatibility or weakened GroundingConstraint",
                )
            if not compatibility_strict and not constraints_strict:
                raise SemanticRepairRejection(
                    "specialization_domain_missing", "specialization did not define a narrower domain",
                )
        elif operation == "split_atomic":
            assert isinstance(source, AbstractAtomicSkill)
            if len(source.effects) < 2:
                raise SemanticRepairRejection("split_boundary_missing", "Atomic source has fewer than two independent effects")
            source_effects = {_effect_key(item) for item in source.effects}
            partitions = []
            for item in candidates:
                assert isinstance(item, AbstractAtomicSkill)
                effect_set = {_effect_key(effect) for effect in item.effects}
                if not effect_set or not effect_set <= source_effects:
                    raise SemanticRepairRejection("split_effect_partition_invalid", "split effect is empty or not from source")
                partitions.append(effect_set)
            if set().union(*partitions) != source_effects or sum(map(len, partitions)) != len(source_effects):
                raise SemanticRepairRejection("split_effect_partition_invalid", "split effects must be a disjoint source partition")
        elif operation == "merge_atomic":
            atomics = [item for item in sources if isinstance(item, AbstractAtomicSkill)]
            if len(atomics) != len(sources) or len({_atomic_contract(item) for item in atomics}) != 1:
                raise SemanticRepairRejection("merge_equivalence_missing", "Atomic merge sources are not contract-equivalent")
            assert isinstance(candidate, AbstractAtomicSkill)
            if _atomic_contract(candidate) != _atomic_contract(atomics[0]):
                raise SemanticRepairRejection("merge_equivalence_missing", "merged candidate changed the equivalent contract")

    @staticmethod
    def _evidence_gate(candidates: Sequence[Artifact], cases: Sequence[RepairEvidence]) -> None:
        for candidate in candidates:
            selected = [
                item for item in cases
                if not item.candidate_keys or candidate.ref.logical_id in item.candidate_keys
            ]
            if any(
                item.agent_parameter_error
                or item.failure_layer == "runtime_agent"
                or item.failure_code.startswith("runtime_agent_")
                for item in selected
            ):
                raise SemanticRepairRejection("agent_parameter_error_not_artifact_evidence", "Agent parameter error cannot justify artifact evolution")
            if len({item.evidence_id for item in selected}) < 2:
                raise SemanticRepairRejection("independent_support_insufficient", "candidate requires two evidence records")
            if len({item.task_id for item in selected}) < 2 or len({item.trace_id for item in selected}) < 2:
                raise SemanticRepairRejection("independent_support_insufficient", "candidate evidence must come from independent tasks/traces")
            clusters = {item.cluster_key for item in selected}
            if "" in clusters or len(clusters) != 1:
                raise SemanticRepairRejection("heterogeneous_failure_cluster", "candidate evidence must form one stable cluster")

    @staticmethod
    def _replay_all(
        candidates: Sequence[Artifact],
        cases: Sequence[RepairEvidence],
        replay: ReplayCallback,
    ) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for candidate in candidates:
            selected = [
                item for item in cases
                if not item.candidate_keys or candidate.ref.logical_id in item.candidate_keys
            ]
            for evidence in selected:
                passed = bool(replay(candidate, dict(evidence.replay_case)))
                details.append({
                    "candidate_ref": str(candidate.ref),
                    "evidence_id": evidence.evidence_id,
                    "task_id": evidence.task_id,
                    "trace_id": evidence.trace_id,
                    "passed": passed,
                })
                if not passed:
                    raise SemanticRepairRejection(
                        "source_replay_failed", "required source replay failed",
                        details=details[-1],
                    )
        return details

    @staticmethod
    def _lineage(
        proposal: RepairProposal,
        candidates: Sequence[Artifact],
        sources: Sequence[Artifact],
    ) -> list[LineageRecord]:
        relation = {
            "split_atomic": GlobalRelationType.SPLIT_FROM,
            "merge_atomic": GlobalRelationType.MERGED_FROM,
        }.get(proposal.operation, GlobalRelationType.DERIVED_FROM)
        return [
            LineageRecord(
                source_ref=str(candidate.ref),
                target_ref=str(source.ref),
                relation=relation,
                operation=proposal.operation,
                proposal_id=proposal.proposal_id,
            )
            for candidate in candidates
            for source in sources
        ]
