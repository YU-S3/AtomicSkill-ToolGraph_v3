"""Contract alignment and immutable version registration for all four layers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable

from ..core.contracts import AbstractAtomicSkill, CompositeSkill, ImplementationAtom, ToolAsset
from ..core.refs import SkillRef, ToolRef, bump_version, content_hash
from ..core.serialization import to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..knowledge.skill_registry import SkillRegistry
from ..knowledge.tool_registry import ToolRegistry


def _atomic_role_occurrences(value: AbstractAtomicSkill, role: str) -> tuple[tuple[str, str, str], ...]:
    """Describe a role by semantic use, without using its trace-local name."""

    occurrences: list[tuple[str, str, str]] = []
    for boundary, predicates in (("pre", value.preconditions), ("effect", value.effects)):
        for predicate in predicates:
            for argument_name, argument in predicate.args.items():
                source_role = getattr(argument, "source_role", "")
                if source_role == role or argument == f"${role}":
                    occurrences.append((boundary, predicate.predicate.casefold(), str(argument_name)))
    return tuple(sorted(occurrences))


def _normalize_atomic_value(value: Any, role_map: dict[str, str]) -> Any:
    primitive = to_primitive(value)
    if isinstance(primitive, str):
        if primitive in role_map:
            return role_map[primitive]
        if primitive.startswith("$") and primitive[1:] in role_map:
            return "$" + role_map[primitive[1:]]
        return primitive
    if isinstance(primitive, list):
        return [_normalize_atomic_value(item, role_map) for item in primitive]
    if isinstance(primitive, dict):
        return {
            str(key): _normalize_atomic_value(item, role_map)
            for key, item in primitive.items()
        }
    return primitive


def _canonical_atomic_contract(value: AbstractAtomicSkill) -> dict[str, Any]:
    """Return the alpha-normalized semantic identity of an Abstract Atomic.

    Summary/intent wording, guideline action sequences, failure prose,
    provenance, concrete source instances, and Tool bodies are deliberately
    outside this identity boundary.
    """

    described_roles: list[tuple[tuple[Any, ...], str, Any]] = []
    for boundary, specs in (("input", value.inputs), ("output", value.outputs)):
        for spec in specs:
            descriptor = (
                boundary,
                str(spec.semantic_type).casefold(),
                bool(spec.required),
                bool(spec.runtime_resolvable),
                str(spec.required_resolution).casefold(),
                _atomic_role_occurrences(value, spec.name),
            )
            described_roles.append((descriptor, spec.name, spec))
    described_roles.sort(key=lambda item: json.dumps(item[0], sort_keys=True))

    role_map: dict[str, str] = {}
    boundary_counts = {"input": 0, "output": 0}
    for descriptor, role, _spec in described_roles:
        boundary = str(descriptor[0])
        canonical = f"{boundary}_{boundary_counts[boundary]:03d}"
        boundary_counts[boundary] += 1
        role_map[role] = canonical

    def parameter(boundary: str, spec: Any) -> dict[str, Any]:
        return {
            "role": role_map[spec.name],
            "boundary": boundary,
            "semantic_type": str(spec.semantic_type).casefold(),
            "required": bool(spec.required),
            "runtime_resolvable": bool(spec.runtime_resolvable),
            "required_resolution": str(spec.required_resolution).casefold(),
        }

    def predicate(item: Any) -> dict[str, Any]:
        return {
            "predicate": str(item.predicate).casefold(),
            "args": {
                str(name): _normalize_atomic_value(argument, role_map)
                for name, argument in sorted(item.args.items())
            },
            "cardinality": int(item.cardinality),
            "distinct_by": str(item.distinct_by),
        }

    def sorted_predicates(items: list[Any]) -> list[dict[str, Any]]:
        normalized = [predicate(item) for item in items]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    return {
        "inputs": sorted(
            (parameter("input", item) for item in value.inputs),
            key=lambda item: item["role"],
        ),
        "outputs": sorted(
            (parameter("output", item) for item in value.outputs),
            key=lambda item: item["role"],
        ),
        "preconditions": sorted_predicates(value.preconditions),
        "effects": sorted_predicates(value.effects),
        "validator_contract": _normalize_atomic_value(value.validator_spec, role_map),
    }


def _atomic_signature(value: AbstractAtomicSkill) -> str:
    return content_hash(_canonical_atomic_contract(value))


def _tool_signature(value: ToolAsset) -> str:
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
    by_step = {item.step_id: item for item in value.occurrences}
    position = {step_id: index for index, step_id in enumerate(value.control_sequence)}

    def expression(raw: Any) -> Any:
        kind = getattr(raw, "kind", None)
        source_step = getattr(raw, "source_step", "")
        return {
            "kind": getattr(kind, "value", kind),
            "source_role": getattr(raw, "source_role", ""),
            "source_step": position.get(source_step, source_step),
            "constant": getattr(raw, "constant", None),
            "transform_id": getattr(raw, "transform_id", ""),
        }

    occurrences = []
    for step_id in value.control_sequence:
        occurrence = by_step[step_id]
        occurrences.append({
            "node_ref": str(occurrence.node_ref),
            "bindings": {
                role: expression(binding)
                for role, binding in sorted(occurrence.binding_specs.items())
            },
        })

    def edge(raw: Any) -> dict[str, Any]:
        return {
            "edge_type": raw.edge_type.value,
            "source": position[raw.source_step],
            "target": position[raw.target_step],
            "source_role": raw.source_role,
            "target_role": raw.target_role,
        }

    return content_hash({
        "occurrences": occurrences,
        "data_edges": sorted(
            (edge(item) for item in value.data_edges),
            key=lambda item: (item["source"], item["target"], item["source_role"], item["target_role"]),
        ),
        "dependency_edges": sorted(
            (edge(item) for item in value.dependency_edges),
            key=lambda item: (item["source"], item["target"], item["source_role"], item["target_role"]),
        ),
        "contract": value.goal_contract,
        "validator_spec": value.validator_spec,
    })


def _version_key(version: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = (int(piece) for piece in version.split("."))
    except Exception as exc:
        raise ValueError(f"semantic version required for alignment: {version!r}") from exc
    return major, minor, patch


@dataclass(frozen=True)
class ToolAlignmentResult:
    ref: ToolRef
    source_ref: ToolRef | None = None
    operation: str = "reuse"
    admitted: bool = True
    admission_failures: tuple[str, ...] = ()


class Aligner:
    def __init__(self, skills: SkillRegistry, tools: ToolRegistry) -> None:
        self.skills, self.tools = skills, tools

    def align_atomic(self, candidate: AbstractAtomicSkill) -> SkillRef:
        signature = _atomic_signature(candidate)
        for existing in self.skills.atomics():
            if _atomic_signature(existing) == signature:
                # align_atomic always promotes a validated canonical occurrence
                # to Candidate; never strand it behind an unusable old version.
                if existing.status in {SkillStatus.CANDIDATE, SkillStatus.ACTIVE}:
                    return existing.ref
        # Do not let whichever trace discovers the capability first leak its
        # intent wording into the persistent logical identity.
        identity_ref = SkillRef(f"atomic_{signature[:24]}", "1.0.0")
        ref = self._next_skill_ref(identity_ref, "atomic")
        admitted = replace(candidate, ref=ref, status=SkillStatus.CANDIDATE)
        self.skills.register_atomic(admitted)
        return ref

    def align_tool(self, candidate: ToolAsset) -> ToolRef:
        signature = _tool_signature(candidate)
        for existing in self.tools.tools():
            if _tool_signature(existing) == signature:
                if not (
                    candidate.status is ToolStatus.CANDIDATE
                    and existing.status not in {
                        ToolStatus.CANDIDATE, ToolStatus.ACTIVE, ToolStatus.PREFERRED,
                    }
                ):
                    return existing.ref
        ref = self._next_tool_ref(candidate.ref)
        self.tools.register(replace(candidate, ref=ref))
        return ref

    def align_tool_with_replays(
        self,
        candidate: ToolAsset,
        *,
        admission: Any,
        replay: Callable[[ToolAsset, dict[str, Any]], bool],
    ) -> ToolAlignmentResult:
        """Reuse an executable or immutably add independently observed replays."""
        if candidate.status is not ToolStatus.CANDIDATE:
            # Admission failure is itself immutable diagnostic knowledge, but
            # never a validated discovery.  Preserve the SHADOW Tool and keep
            # its paired Implementation on the same non-usable version.
            ref = self._next_tool_ref(candidate.ref)
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
        signature = _tool_signature(candidate)
        matches = [
            item for item in self.tools.tools()
            if _tool_signature(item) == signature
            and item.status in {
                ToolStatus.CANDIDATE,
                ToolStatus.ACTIVE,
                ToolStatus.PREFERRED,
            }
        ]
        if not matches:
            ref = self._next_tool_ref(candidate.ref)
            self.tools.register(replace(candidate, ref=ref))
            return ToolAlignmentResult(ref, operation="discover")
        existing = sorted(
            matches,
            key=lambda item: (_version_key(item.ref.version), str(item.ref)),
            reverse=True,
        )[0]
        existing_cases = {
            content_hash(item): item for item in existing.tests
        }
        novel = [
            item for item in candidate.tests
            if content_hash(item) not in existing_cases
        ]
        if not novel:
            return ToolAlignmentResult(existing.ref)
        merged_cases = [*existing.tests, *novel]
        next_ref = self._next_tool_ref(existing.ref)
        replacement = replace(
            existing,
            ref=next_ref,
            tests=merged_cases,
            provenance={
                **existing.provenance,
                "evolution_operation": "add_replay",
                "source_ref": str(existing.ref),
                "source_trace_ids": sorted({
                    str(item.get("trace_id", ""))
                    for item in merged_cases
                    if item.get("trace_id")
                }),
            },
            metadata={
                **existing.metadata,
                "batch_evolution": {
                    "operation": "add_replay",
                    "source_ref": str(existing.ref),
                    "added_replay_count": len(novel),
                },
            },
            status=ToolStatus.ADMISSION_PENDING,
        )
        admitted = admission.admit_tool(replacement, replay=replay)
        if admitted.status is not ToolStatus.CANDIDATE:
            return ToolAlignmentResult(
                existing.ref,
                existing.ref,
                "add_replay",
                False,
                tuple(map(str, admitted.metadata.get("admission_failure") or [])),
            )
        self.tools.register(admitted)
        return ToolAlignmentResult(admitted.ref, existing.ref, "add_replay")

    def align_implementation(self, candidate: ImplementationAtom, atomic_ref: SkillRef, tool_ref: ToolRef) -> SkillRef:
        candidate = replace(
            candidate, abstract_ref=atomic_ref,
            tool_bindings=[replace(item, tool_ref=tool_ref) for item in candidate.tool_bindings],
        )
        signature = _implementation_signature(candidate)
        for existing in self.skills.implementations():
            if signature == _implementation_signature(existing):
                if not (
                    candidate.status is SkillStatus.CANDIDATE
                    and existing.status not in {SkillStatus.CANDIDATE, SkillStatus.ACTIVE}
                ):
                    return existing.ref
        ref = self._next_skill_ref(candidate.ref, "implementation")
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
        ref = self._next_skill_ref(candidate.ref, "composite")
        self.skills.register_composite(replace(candidate, ref=ref))
        return ref

    def _next_skill_ref(self, ref: SkillRef, kind: str) -> SkillRef:
        versions = [item.version for item in self.skills.list_refs(kind) if item.logical_id == ref.logical_id]
        return ref if not versions else SkillRef(ref.logical_id, bump_version(max(versions, key=_version_key)))

    def _next_tool_ref(self, ref: ToolRef) -> ToolRef:
        versions = [item.version for item in self.tools.list_refs() if item.tool_id == ref.tool_id]
        return ref if not versions else ToolRef(ref.tool_id, bump_version(max(versions, key=_version_key)))
