"""Contract alignment and immutable version registration for all four layers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from ..core.contracts import AbstractAtomicSkill, CompositeSkill, ImplementationAtom, ToolAsset
from ..core.refs import SkillRef, ToolRef, bump_version, content_hash
from ..core.status import SkillStatus, ToolStatus
from ..knowledge.skill_registry import SkillRegistry
from ..knowledge.tool_registry import ToolRegistry
from .contract_canonicalizer import (
    AtomicContractCanonicalizer,
    CanonicalizedAtomicBundle,
    aligned_role_maps,
    atomic_contract_signature,
    canonical_atomic_contract,
    composite_structure_payload,
)
from .identity_matching import IdentityProof, match_atomic, match_tool, match_implementation, bounded_matches, MAX_SEARCH_STATES


def _canonical_atomic_contract(value: AbstractAtomicSkill) -> dict[str, Any]:
    """Compatibility wrapper around the sole contract canonicalizer."""

    return canonical_atomic_contract(value)


def _atomic_signature(value: AbstractAtomicSkill) -> str:
    """Compatibility wrapper used by maintenance and diagnosis modules."""

    return atomic_contract_signature(value)


def _tool_signature(value: ToolAsset) -> str:
    """Legacy exact-byte replay/cache key, NOT an equivalence proof.

    Alpha identity consumers must call match_tool and apply its explicit maps.
    This key intentionally remains conservative for historical case APIs until
    their source-specific certificate has verified the full executable bytes.
    """
    # Replay cases and provenance are evidence for an immutable executable,
    # not part of executable identity.  They are credited through the Ledger.
    return content_hash({
        "signature": value.signature,
        "interface": value.interface,
        "artifact_kind": value.artifact_kind,
        "artifact": value.artifact,
        "safety": value.safety,
    })


def _implementation_signature(value: ImplementationAtom) -> str:
    return content_hash({
        "abstract": str(value.abstract_ref),
        "tools": value.tool_bindings,
        "constraints": value.grounding_constraints,
        "policy": value.execution_policy,
        "compatibility": value.compatibility,
    })


def _composite_signature(value: CompositeSkill) -> str:
    """Hash workflow semantics without trace-local occurrence/edge identifiers."""
    return content_hash(composite_structure_payload(
        occurrences=value.occurrences,
        control_sequence=value.control_sequence,
        data_edges=value.data_edges,
        dependency_edges=value.dependency_edges,
        goal_contract=value.goal_contract,
        validator_spec=value.validator_spec,
    ))


def _version_key(version: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = (int(piece) for piece in version.split("."))
    except Exception as exc:
        raise ValueError(f"semantic version required for alignment: {version!r}") from exc
    return major, minor, patch


@dataclass(frozen=True)
class AtomicAlignment:
    ref: SkillRef
    role_map: dict[str, str]
    reused: bool


@dataclass(frozen=True)
class ToolAlignmentResult:
    ref: ToolRef
    source_ref: ToolRef | None = None
    operation: str = "reuse"
    admitted: bool = True
    admission_failures: tuple[str, ...] = ()
    identity_proof: IdentityProof | None = None


class Aligner:
    def __init__(self, skills: SkillRegistry, tools: ToolRegistry) -> None:
        self.skills, self.tools = skills, tools
        self.atomic_canonicalizer = AtomicContractCanonicalizer()

    def resolve_atomic(self, candidate: AbstractAtomicSkill) -> AtomicAlignment:
        """Resolve alpha-equivalence and expose candidate-to-persisted roles."""

        existing_ref = self.skills.find_equivalent_atomic(candidate)
        if existing_ref is not None:
            persisted = self.skills.get_atomic(existing_ref)
            input_roles, output_roles = aligned_role_maps(candidate, persisted)
            return AtomicAlignment(
                existing_ref,
                {**output_roles, **input_roles},
                True,
            )
        neutral = self.atomic_canonicalizer.canonicalize(candidate)
        ref = self._next_skill_ref(
            neutral.atomic.ref,
            "atomic",
        )
        # A newly discovered Atomic establishes the persistent role schema.
        # Keep those semantic role names intact; alpha-normalization is only
        # an identity calculation.  Later equivalent candidates are mapped
        # onto these actual persisted names by ``aligned_role_maps``.
        input_roles = {item.name: item.name for item in candidate.inputs}
        output_roles = {item.name: item.name for item in candidate.outputs}
        return AtomicAlignment(
            ref,
            {**output_roles, **input_roles},
            False,
        )

    def stage_atomic(
        self,
        candidate: AbstractAtomicSkill,
        tool: ToolAsset | None = None,
        implementation: ImplementationAtom | None = None,
    ) -> CanonicalizedAtomicBundle:
        """Resolve the eventual persistent Atomic ref without writing state.

        E2 can use the returned canonical roles and ``bundle.atomic.ref`` for
        authoritative existing-edge lookup.  A later ``align_atomic`` call
        will choose the same ref as long as no concurrent writer mutates the
        single-process experiment bank.
        """

        alignment = self.resolve_atomic(candidate)
        if alignment.reused:
            persisted = self.skills.get_atomic(alignment.ref)
            input_roles, output_roles = aligned_role_maps(candidate, persisted)
            return self.atomic_canonicalizer.canonicalize(
                candidate,
                tool,
                implementation,
                input_role_map=input_roles,
                output_role_map=output_roles,
                atomic_ref=alignment.ref,
            )
        return self.atomic_canonicalizer.canonicalize(
            candidate,
            tool,
            implementation,
            input_role_map={item.name: item.name for item in candidate.inputs},
            output_role_map={item.name: item.name for item in candidate.outputs},
            atomic_ref=alignment.ref,
        )

    def align_atomic(self, candidate: AbstractAtomicSkill) -> SkillRef:
        alignment = self.resolve_atomic(candidate)
        if alignment.reused:
            return alignment.ref
        staged = self.stage_atomic(candidate)
        admitted = replace(
            staged.atomic,
            status=SkillStatus.CANDIDATE,
        )
        self.skills.register_atomic(admitted)
        return admitted.ref

    def align_tool(self, candidate: ToolAsset) -> ToolRef:
        signature = _tool_signature(candidate)
        for existing, result in bounded_matches(candidate, self.tools.tools(), match_tool):
            if result.status == "exact":
                # Identity is independent of availability: failed programs
                # cannot acquire a clean lifecycle by changing their names.
                return existing.ref
        ref = self._next_tool_ref(
            ToolRef(f"tool_{signature[:24]}", "1.0.0")
        )
        self.tools.register(replace(candidate, ref=ref))
        return ref

    def existing_tool_with_replay_cases(
        self,
        candidate: ToolAsset,
    ) -> ToolAsset | None:
        """Return an executable only when current-authority certificates cover it."""
        from ..governance.ledger import EvidenceLedger
        from .replay_certificates import ReplayCertificates
        certificates = ReplayCertificates(EvidenceLedger(self.tools.database))

        if candidate.status not in {
            ToolStatus.ADMISSION_PENDING,
            ToolStatus.CANDIDATE,
        }:
            return None
        candidate_cases = {content_hash(item) for item in candidate.tests}
        if not candidate_cases:
            return None
        matches = [
            item
            for item, result in bounded_matches(candidate, self.tools.tools(), match_tool)
            if result.status == "exact"
            and item.status in {
                ToolStatus.CANDIDATE,
                ToolStatus.ACTIVE,
                ToolStatus.PREFERRED,
            }
            and all(certificates.lookup(_tool_signature(item), case, tool=item) is not None
                    for case in candidate.tests)
        ]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda item: (_version_key(item.ref.version), str(item.ref)),
            reverse=True,
        )[0]

    def align_tool_with_replays(
        self,
        candidate: ToolAsset,
        *,
        admission: Any,
        replay: Callable[[ToolAsset, dict[str, Any]], bool],
    ) -> ToolAlignmentResult:
        """Align an admitted executable; replay evidence never changes its ref.

        Admission runs on the incoming cases before this method. The caller's
        certificate-aware replay boundary performs novel cases and records
        their outcomes separately from the immutable ToolAsset.
        """
        signature = _tool_signature(candidate)
        results = list(bounded_matches(candidate, self.tools.tools(), match_tool))
        matches = [item for item, result in results if result.status == "exact"]
        proofs = {str(item.ref): result.proof for item, result in results if result.status == "exact"}
        if candidate.status is not ToolStatus.CANDIDATE:
            if matches:
                existing = sorted(
                    matches,
                    key=lambda item: (_version_key(item.ref.version), str(item.ref)),
                    reverse=True,
                )[0]
                return ToolAlignmentResult(
                    existing.ref,
                    existing.ref,
                    "add_replay",
                    False,
                    tuple(map(
                        str,
                        candidate.metadata.get("admission_failure") or [],
                    )),
                    proofs[str(existing.ref)],
                )
            # Admission failure is itself immutable diagnostic knowledge, but
            # never a validated discovery. Preserve a SHADOW only when there
            # is no older usable executable to retain as the immutable target.
            ref = self._next_tool_ref(
                ToolRef(f"tool_{signature[:24]}", "1.0.0")
            )
            rejected = replace(candidate, ref=ref, status=ToolStatus.SHADOW)
            self.tools.register(rejected)
            return ToolAlignmentResult(
                ref,
                operation="discover",
                admitted=False,
                admission_failures=tuple(map(
                    str, rejected.metadata.get("admission_failure") or [],
                )),
            )
        if not matches:
            ref = self._next_tool_ref(
                ToolRef(f"tool_{signature[:24]}", "1.0.0")
            )
            self.tools.register(replace(candidate, ref=ref))
            return ToolAlignmentResult(ref, operation="discover")
        existing = sorted(
            matches,
            key=lambda item: (_version_key(item.ref.version), str(item.ref)),
            reverse=True,
        )[0]
        usable = existing.status in {ToolStatus.CANDIDATE, ToolStatus.ACTIVE, ToolStatus.PREFERRED}
        return ToolAlignmentResult(existing.ref, existing.ref, "add_replay", usable,
            () if usable else ("equivalent_executable_unavailable",), proofs[str(existing.ref)])

    def replay_target_ref(self, candidate: ToolAsset) -> ToolRef:
        """Resolve the same eventual ref as alignment without registering it."""
        signature = _tool_signature(candidate)
        matches = [item for item, result in bounded_matches(candidate, self.tools.tools(), match_tool)
                   if result.status == "exact"]
        if matches:
            return max(matches, key=lambda item: (_version_key(item.ref.version), str(item.ref))).ref
        return self._next_tool_ref(ToolRef(f"tool_{signature[:24]}", "1.0.0"))

    def align_implementation(self, candidate: ImplementationAtom, atomic_ref: SkillRef, tool_ref: ToolRef,
                             *, source_tool: ToolAsset | None = None) -> SkillRef:
        if len(candidate.tool_bindings) != 1:
            raise ValueError("single-Tool alignment cannot retarget a multi-Tool implementation")
        if source_tool is not None:
            from ..core.bindings import BindingExprKind, BindingExpression
            destination = self.tools.get(tool_ref)
            result = match_tool(source_tool, destination)
            if result.status != "exact" or result.proof is None:
                # A newly registered raw-identical unsupported legacy Tool
                # needs no role remapping, but cannot share equivalence credit.
                if _tool_signature(source_tool) != _tool_signature(destination):
                    raise ValueError("Tool role retargeting requires a complete identity proof")
            else:
                proof = result.proof
                bindings = [replace(b, parameter_mapping={proof.input_role_map.get(k, k): v
                    for k, v in b.parameter_mapping.items()}) for b in candidate.tool_bindings]
                policy = dict(candidate.execution_policy)
                if "output_mapping" in policy:
                    mapping = {}
                    for k, raw in policy["output_mapping"].items():
                        expression = BindingExpression.from_dict(raw)
                        if expression.kind is BindingExprKind.TOOL_OUTPUT:
                            expression = replace(expression, source_role=proof.output_role_map.get(
                                expression.source_role, expression.source_role))
                        mapping[k] = expression
                    policy["output_mapping"] = mapping
                candidate = replace(candidate, tool_bindings=bindings, execution_policy=policy)
        candidate = replace(
            candidate, abstract_ref=atomic_ref,
            tool_bindings=[replace(item, tool_ref=tool_ref) for item in candidate.tool_bindings],
        )
        signature = _implementation_signature(candidate)
        atomic = self.skills.get_atomic(atomic_ref)
        source_tools = {str(b.tool_ref): self.tools.get(b.tool_ref) for b in candidate.tool_bindings}
        remaining = MAX_SEARCH_STATES
        for existing in self.skills.implementations():
            result = match_implementation(candidate, existing,
                source_atomic=atomic, target_atomic=self.skills.get_atomic(existing.abstract_ref),
                source_tools=source_tools,
                target_tools={str(b.tool_ref): self.tools.get(b.tool_ref) for b in existing.tool_bindings}, max_states=remaining)
            remaining -= result.search_states
            if result.status == "exact":
                return existing.ref
            if remaining <= 0:
                break
        ref = self._next_skill_ref(
            SkillRef(f"impl_{signature[:24]}", "1.0.0"),
            "implementation",
        )
        self.skills.register_implementation(replace(candidate, ref=ref))
        return ref

    def align_composite(self, candidate: CompositeSkill, atomic_refs: dict[str, SkillRef]) -> SkillRef:
        occurrences = [replace(item, node_ref=atomic_refs.get(item.occurrence_id, item.node_ref)) for item in candidate.occurrences]
        candidate = replace(candidate, occurrences=occurrences)
        signature = _composite_signature(candidate)
        for existing in self.skills.composites():
            if signature == _composite_signature(existing):
                if not (
                    candidate.status is SkillStatus.CANDIDATE
                    and existing.status not in {SkillStatus.CANDIDATE, SkillStatus.ACTIVE}
                ):
                    return existing.ref
        ref = self._next_skill_ref(
            SkillRef(f"composite_{signature[:24]}", "1.0.0"),
            "composite",
        )
        self.skills.register_composite(replace(candidate, ref=ref))
        return ref

    def _next_skill_ref(self, ref: SkillRef, kind: str) -> SkillRef:
        versions = [item.version for item in self.skills.list_refs(kind) if item.logical_id == ref.logical_id]
        return ref if not versions else SkillRef(ref.logical_id, bump_version(max(versions, key=_version_key)))

    def _next_tool_ref(self, ref: ToolRef) -> ToolRef:
        versions = [item.version for item in self.tools.list_refs() if item.tool_id == ref.tool_id]
        return ref if not versions else ToolRef(ref.tool_id, bump_version(max(versions, key=_version_key)))
