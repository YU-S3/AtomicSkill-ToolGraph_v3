"""Atomic, implementation, and composite version registry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, TypeVar

from ..core.bindings import BindingExpression, GroundingConstraint, ToolBinding
from ..core.contracts import (
    AbstractAtomicSkill, CompositeOccurrence, CompositeSkill, IdentityConstraint,
    ImplementationAtom, ParameterSpec, SemanticPredicate, TaskContract,
)
from ..core.edges import GraphEdge
from ..core.refs import SkillRef
from ..core.status import RuntimeMode, SkillStatus, skill_status_usable
from .artifact_store import ArtifactStore
from .database import StateDatabase

SkillObject = AbstractAtomicSkill | ImplementationAtom | CompositeSkill


def _ref(value: Any) -> SkillRef:
    return value if isinstance(value, SkillRef) else SkillRef.from_dict(value)


def _predicate(value: Any) -> SemanticPredicate:
    if isinstance(value, SemanticPredicate):
        return value
    args = {}
    for key, item in value.get("args", {}).items():
        args[key] = BindingExpression.from_dict(item) if isinstance(item, dict) and "kind" in item else item
    return SemanticPredicate(value["predicate"], args, int(value.get("cardinality", 1)), str(value.get("distinct_by", "")))


def _parameter(value: Any) -> ParameterSpec:
    return value if isinstance(value, ParameterSpec) else ParameterSpec(**value)


def _contract(value: Any) -> TaskContract:
    if isinstance(value, TaskContract):
        return value
    return TaskContract(
        target_effects=[_predicate(item) for item in value.get("target_effects", [])],
        cardinality_constraints=list(value.get("cardinality_constraints", [])),
        identity_constraints=[IdentityConstraint(**item) for item in value.get("identity_constraints", [])],
        source=value.get("source", "planner_proposed"),
        confidence=float(value.get("confidence", 0.0)),
        validator_id=str(value.get("validator_id", "")),
    )


class SkillRegistry:
    def __init__(self, store: ArtifactStore, database: StateDatabase) -> None:
        self.store = store
        self.database = database

    def register_atomic(self, artifact: AbstractAtomicSkill) -> None:
        self.store.put("atomic", artifact)

    def register_implementation(self, artifact: ImplementationAtom) -> None:
        self.store.put("implementation", artifact)

    def register_composite(self, artifact: CompositeSkill) -> None:
        self.store.put("composite", artifact)

    def register(self, artifact: SkillObject) -> None:
        if isinstance(artifact, AbstractAtomicSkill):
            self.register_atomic(artifact)
        elif isinstance(artifact, ImplementationAtom):
            self.register_implementation(artifact)
        elif isinstance(artifact, CompositeSkill):
            self.register_composite(artifact)
        else:
            raise TypeError(type(artifact).__name__)

    def _payload(self, ref: SkillRef, kind: str) -> tuple[dict[str, Any], SkillStatus]:
        row = self.database.execute(
            "SELECT artifact_kind,status FROM artifact_index WHERE artifact_ref=?", (str(ref),)
        ).fetchone()
        if row is None or row["artifact_kind"] != kind:
            raise KeyError(str(ref))
        return self.store.get_payload(str(ref)), SkillStatus(row["status"])

    def get_atomic(self, ref: SkillRef | str) -> AbstractAtomicSkill:
        ref = SkillRef.parse(ref)
        payload, status = self._payload(ref, "atomic")
        return AbstractAtomicSkill(
            ref=ref, summary=payload["summary"],
            inputs=[_parameter(item) for item in payload.get("inputs", [])],
            outputs=[_parameter(item) for item in payload.get("outputs", [])],
            preconditions=[_predicate(item) for item in payload.get("preconditions", [])],
            effects=[_predicate(item) for item in payload.get("effects", [])],
            validator_spec=payload.get("validator_spec", {}),
            failure_modes=payload.get("failure_modes", []),
            guideline=payload.get("guideline", {}), metadata=payload.get("metadata", {}), status=status,
        )

    def find_equivalent_atomic(
        self,
        candidate: AbstractAtomicSkill,
        *,
        statuses: Iterable[SkillStatus | str] = (
            SkillStatus.CANDIDATE,
            SkillStatus.ACTIVE,
        ),
    ) -> SkillRef | None:
        """Read-only lookup using the same canonical contract as Aligner.

        This is intentionally an equivalence query, not registration.  It is
        safe to call while staging E2 authority and cannot create an Artifact
        row or advance a semantic version.
        """

        # Local import keeps the knowledge registry independent from the
        # stateful Aligner while sharing its pure canonical identity function.
        from ..evolution.contract_canonicalizer import atomic_contract_signature

        allowed = {SkillStatus(value) for value in statuses}
        signature = atomic_contract_signature(candidate)
        matches = [
            atomic
            for atomic in self.atomics()
            if atomic.status in allowed
            and atomic_contract_signature(atomic) == signature
        ]
        if not matches:
            return None

        def version_key(value: AbstractAtomicSkill) -> tuple[int, int, int, int, str]:
            try:
                major, minor, patch = (
                    int(piece) for piece in value.ref.version.split(".")
                )
                parsed = (major, minor, patch)
            except (TypeError, ValueError):
                parsed = (-1, -1, -1)
            return (
                1 if value.status is SkillStatus.ACTIVE else 0,
                *parsed,
                str(value.ref),
            )

        return max(matches, key=version_key).ref

    def get_implementation(self, ref: SkillRef | str) -> ImplementationAtom:
        ref = SkillRef.parse(ref)
        payload, status = self._payload(ref, "implementation")
        return ImplementationAtom(
            ref=ref, abstract_ref=_ref(payload["abstract_ref"]),
            tool_bindings=[ToolBinding(**item) for item in payload.get("tool_bindings", [])],
            grounding_constraints=[GroundingConstraint(**item) for item in payload.get("grounding_constraints", [])],
            execution_policy=payload.get("execution_policy", {}), compatibility=payload.get("compatibility", {}),
            quality=payload.get("quality", {}), status=status,
            metadata=payload.get("metadata", {}),
        )

    def get_composite(self, ref: SkillRef | str) -> CompositeSkill:
        ref = SkillRef.parse(ref)
        payload, status = self._payload(ref, "composite")
        occurrences = [
            CompositeOccurrence(
                step_id=item["step_id"], occurrence_id=item["occurrence_id"], node_ref=_ref(item["node_ref"]),
                binding_specs={key: BindingExpression.from_dict(value) for key, value in item.get("binding_specs", {}).items()},
            ) for item in payload.get("occurrences", [])
        ]
        return CompositeSkill(
            ref=ref, summary=payload["summary"], occurrences=occurrences,
            control_sequence=list(payload.get("control_sequence", [])),
            data_edges=[GraphEdge(**item) for item in payload.get("data_edges", [])],
            dependency_edges=[GraphEdge(**item) for item in payload.get("dependency_edges", [])],
            goal_contract=_contract(payload.get("goal_contract", {})), guideline=payload.get("guideline", {}),
            insight=payload.get("insight", {}), validator_spec=payload.get("validator_spec", {}),
            metadata=payload.get("metadata", {}), status=status,
        )

    def list_refs(self, kind: str, *, mode: RuntimeMode | str | None = None) -> list[SkillRef]:
        rows = self.database.rows(
            "SELECT logical_id,version,status FROM artifact_index WHERE artifact_kind=? ORDER BY logical_id,version",
            (kind,),
        )
        return [
            SkillRef(row["logical_id"], row["version"]) for row in rows
            if mode is None or skill_status_usable(row["status"], mode)
        ]

    def atomics(self, *, mode: RuntimeMode | str | None = None) -> list[AbstractAtomicSkill]:
        return [self.get_atomic(ref) for ref in self.list_refs("atomic", mode=mode)]

    def implementations(self, *, mode: RuntimeMode | str | None = None) -> list[ImplementationAtom]:
        return [self.get_implementation(ref) for ref in self.list_refs("implementation", mode=mode)]

    def composites(self, *, mode: RuntimeMode | str | None = None) -> list[CompositeSkill]:
        return [self.get_composite(ref) for ref in self.list_refs("composite", mode=mode)]

    def implementations_for(self, abstract_ref: SkillRef, *, mode: RuntimeMode | str) -> list[ImplementationAtom]:
        return [item for item in self.implementations(mode=mode) if item.abstract_ref == abstract_ref]

    def update_status(self, ref: SkillRef | str, status: SkillStatus | str) -> None:
        if self.database.readonly:
            raise RuntimeError("frozen registry is read-only")
        status = SkillStatus(status)
        cursor = self.database.execute(
            "UPDATE artifact_index SET status=? WHERE artifact_ref=? AND artifact_kind IN ('atomic','implementation','composite')",
            (status.value, str(SkillRef.parse(ref))),
        )
        if cursor.rowcount != 1:
            raise KeyError(str(ref))
        self.database.connection.commit()

    def set_recommended(self, ref: SkillRef | str) -> None:
        if self.database.readonly:
            raise RuntimeError("frozen registry is read-only")
        ref = SkillRef.parse(ref)
        self.database.execute(
            "INSERT INTO recommended_pointers(logical_id,artifact_ref) VALUES(?,?) "
            "ON CONFLICT(logical_id) DO UPDATE SET artifact_ref=excluded.artifact_ref",
            (ref.logical_id, str(ref)),
        )
        self.database.connection.commit()
