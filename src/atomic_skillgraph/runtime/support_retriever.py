"""Generic Runtime support-Atomic retrieval.

Only the blocked Atomic contract is used: missing input roles are matched
against other normal Atomic outputs/effects.  No task type, object family, or
benchmark workflow may enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..core.contracts import AbstractAtomicSkill
from ..core.serialization import to_primitive
from ..core.support_authority import support_role_authority


@dataclass(frozen=True)
class SupportObligation:
    kind: str
    consumer_atomic_ref: str
    consumer_occurrence_id: str
    role: str = ""
    semantic_type: str = ""
    required_resolution: str = ""
    predicate: str = ""
    predicate_args: tuple[tuple[str, Any], ...] = ()
    effect_domain: str = ""
    cardinality: int = 1
    distinct_by: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"binding", "predicate"}:
            raise ValueError("invalid Support obligation kind")


def predicate_input_mapping(producer: Any, consumer: Any, obligation: SupportObligation) -> list[dict[str, str]]:
    """Exact predicate positions prove role identity, never role spelling/type alone."""
    from ..core.support_authority import _referenced_role
    from ..core.semantic_types import semantic_types_compatible
    producer_roles = {item.name: item for item in (*producer.inputs, *producer.outputs)}
    consumer_roles = {item.name: item for item in consumer.inputs}
    mappings = []
    for effect in producer.effects:
        if (effect.predicate != obligation.predicate
                or str(effect.effect_domain.value) != obligation.effect_domain
                or effect.cardinality < obligation.cardinality
                or effect.distinct_by != obligation.distinct_by
                or set(effect.args) != dict(obligation.predicate_args).keys()):
            continue
        mapping: dict[str, str] = {}
        valid = True
        for position, target in obligation.predicate_args:
            source = effect.args[position]
            left, right = _referenced_role(source), _referenced_role(target)
            if not left or not right:
                if to_primitive(source) != to_primitive(target):
                    valid = False
                continue
            p, c = producer_roles.get(left), consumer_roles.get(right)
            if p is None or c is None or not semantic_types_compatible(p.semantic_type, c.semantic_type):
                valid = False
                break
            if left in mapping and mapping[left] != right:
                valid = False
                break
            mapping[left] = right
        if valid and mapping not in mappings:
            mappings.append(mapping)
    return mappings


@dataclass(frozen=True)
class SupportRoleMapping:
    producer_role: str
    consumer_role: str
    semantic_type: str
    producer_resolution: str = "semantic"
    required_resolution: str = "semantic"
    effect_domain: str = ""


@dataclass(frozen=True)
class SupportCandidate:
    atomic_ref: str
    score: float
    supplied_roles: tuple[str, ...]
    output_roles: tuple[str, ...]
    effect_predicates: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]
    role_mappings: tuple[SupportRoleMapping, ...] = ()
    inputs: tuple[dict[str, Any], ...] = ()
    outputs: tuple[dict[str, Any], ...] = ()
    execution_available: bool = False
    missing_required_inputs: tuple[str, ...] = ()
    predicate_obligations: tuple[dict[str, Any], ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict)
    allowed_output_mappings: tuple[dict[str, str], ...] = ()
    mapping_previews: tuple[dict[str, Any], ...] = ()
    mapping_proven: bool = False
    input_ready: bool = False


def _predicate_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("predicate", ""))
    return str(getattr(value, "predicate", ""))


class SupportAtomicRetriever:
    """Return formal-compatible support candidates, never workflow choices."""

    def retrieve_for_task(self, *, query: str, atomics, execution_availability, top_k=5):
        """Task consumers have no parent-role obligation; offer their own interfaces."""
        from ..knowledge.query import lexical_similarity
        from ..agents.portable_support_view import portable_support_view
        candidates = []
        for raw in atomics:
            if not execution_availability.get(str(raw.ref), False):
                continue
            atomic = portable_support_view(raw)
            candidates.append(SupportCandidate(
                str(atomic.ref), lexical_similarity(query, atomic.summary), (),
                tuple(p.name for p in atomic.outputs), tuple(e.predicate for e in atomic.effects),
                ({"consumer_scope": "task"},), inputs=tuple(to_primitive(atomic.inputs)),
                outputs=tuple(to_primitive(atomic.outputs)), execution_available=True))
        return sorted(candidates, key=lambda c: (-c.score, c.atomic_ref))[:top_k]

    def retrieve(
        self,
        *,
        blocked_atomic: AbstractAtomicSkill,
        missing_roles: Iterable[str],
        atomics: Iterable[AbstractAtomicSkill],
        execution_availability: Mapping[str, bool] | None = None,
        top_k: int | None = 3,
        obligations: Iterable[SupportObligation] = (),
    ) -> list[SupportCandidate]:
        missing = {str(role) for role in missing_roles}
        obligations = tuple(obligations)
        if not missing and not obligations:
            return []
        blocked_inputs = {str(item.name): item for item in blocked_atomic.inputs}
        candidates: list[SupportCandidate] = []
        for atomic in atomics:
            from ..agents.portable_support_view import portable_support_view
            atomic = portable_support_view(atomic)
            if str(atomic.ref) == str(blocked_atomic.ref):
                continue
            execution_available = bool(
                execution_availability is None
                or execution_availability.get(str(atomic.ref), False)
            )
            if not execution_available:
                continue
            mappings: list[SupportRoleMapping] = []
            supplied_roles: list[str] = []
            diagnostics: list[dict[str, Any]] = []
            for output in atomic.outputs:
                for consumer_role in sorted(missing):
                    required = blocked_inputs.get(consumer_role)
                    if required is None:
                        continue
                    authority = support_role_authority(
                        atomic,
                        str(output.name),
                        blocked_atomic,
                        consumer_role,
                    )
                    diagnostics.append({
                        "producer_role": output.name,
                        "consumer_role": consumer_role,
                        "compatible": authority.authorized,
                        "authority_reason": authority.reason,
                        "semantic_role_authorized": bool(
                            authority.semantic_alias_authorized
                            or authority.relation_verified_exception
                        ),
                        "required_type": required.semantic_type,
                        "offered_type": output.semantic_type,
                        "producer_resolution": (
                            authority.producer_resolution
                        ),
                        "required_resolution": (
                            authority.required_resolution
                        ),
                        "effect_domain": authority.effect_domain,
                        "producer_aliases": list(
                            authority.producer_aliases
                        ),
                        "consumer_aliases": list(
                            authority.consumer_aliases
                        ),
                        "relation_verified_exception": (
                            authority.relation_verified_exception
                        ),
                    })
                    if not authority.authorized:
                        continue
                    mappings.append(SupportRoleMapping(
                        producer_role=str(output.name),
                        consumer_role=str(consumer_role),
                        semantic_type=authority.semantic_type,
                        producer_resolution=authority.producer_resolution,
                        required_resolution=authority.required_resolution,
                        effect_domain=authority.effect_domain,
                    ))
                    if output.name not in supplied_roles:
                        supplied_roles.append(output.name)
            predicate_obligations = tuple({"obligation": to_primitive(obligation), "input_mapping": mapping}
                for obligation in obligations if obligation.kind == "predicate"
                for mapping in predicate_input_mapping(atomic, blocked_atomic, obligation))
            if not mappings and not predicate_obligations:
                continue
            candidates.append(SupportCandidate(
                atomic_ref=str(atomic.ref),
                score=float(len(mappings) + len(predicate_obligations)),
                supplied_roles=tuple(sorted(supplied_roles)),
                output_roles=tuple(sorted(str(item.name) for item in atomic.outputs)),
                effect_predicates=tuple(sorted({
                    _predicate_name(item) for item in atomic.effects
                })),
                diagnostics=tuple(to_primitive(diagnostics)),
                role_mappings=tuple(mappings),
                inputs=tuple(
                    {
                        "name": str(item.name),
                        "semantic_type": str(item.semantic_type),
                        "required": bool(item.required),
                        "runtime_resolvable": bool(item.runtime_resolvable),
                        "required_resolution": str(item.required_resolution),
                    }
                    for item in atomic.inputs
                ),
                outputs=tuple(
                    {
                        "name": str(item.name),
                        "semantic_type": str(item.semantic_type),
                        "required": bool(item.required),
                    }
                    for item in atomic.outputs
                ),
                execution_available=True,
                predicate_obligations=predicate_obligations,
                # A retrieved Atomic has no support occurrence or prepared
                # argument bindings yet.  Required inputs remain explicitly
                # missing until the selected call passes ordinary preflight.
                missing_required_inputs=tuple(sorted(
                    str(item.name) for item in atomic.inputs if item.required
                )),
            ))
        candidates.sort(key=lambda item: (-item.score, item.atomic_ref))
        if top_k is None:
            return candidates
        return candidates[: max(0, int(top_k))]


__all__ = ["SupportAtomicRetriever", "SupportCandidate"]
