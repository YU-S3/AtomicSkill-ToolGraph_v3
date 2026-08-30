"""Validate reference-only E1 boundaries against transition certificates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import ParameterSpec, SemanticPredicate, TaskContract
from ..core.refs import SkillRef, content_hash
from ..validation.contract_matcher import ContractMatcher


@dataclass
class AtomicBoundaryProposal:
    phase_id: str
    intent: str
    event_start: int
    event_end_exclusive: int
    selected_effect_refs: list[str]
    selected_precondition_refs: list[str]
    output_role_mapping: dict[str, str]
    rationale: str


# Compatibility name for code that imports the previous public symbol.  The
# constructor contract is intentionally the new reference-only contract.
AtomicOccurrenceProposal = AtomicBoundaryProposal


@dataclass
class CanonicalAtomicOccurrence:
    occurrence_id: str
    phase_id: str
    intent: str
    event_start: int
    event_end_exclusive: int
    input_bindings: dict[str, Any]
    output_bindings: dict[str, Any]
    input_specs: list[ParameterSpec]
    output_specs: list[ParameterSpec]
    preconditions: list[SemanticPredicate]
    effects: list[SemanticPredicate]
    action_events: list[dict[str, Any]]
    prefix_events: list[dict[str, Any]]
    source_task: dict[str, Any]
    source_trace_id: str
    proposed_ref: SkillRef
    validation_refs: list[str] = field(default_factory=list)
    selected_effect_refs: list[str] = field(default_factory=list)
    selected_precondition_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_end(self) -> int:
        """Inclusive compatibility view; extractor authority remains half-open."""

        return self.event_end_exclusive - 1


def _semantic_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "entity" if isinstance(value, str) else "value"


def _value_key(value: Any) -> str:
    return repr(value)


def _fact(raw: Mapping[str, Any]) -> dict[str, Any]:
    fact_ref = str(raw.get("fact_ref", ""))
    predicate = str(raw.get("predicate", ""))
    arguments = dict(raw.get("args") or {})
    if not fact_ref or not predicate or not arguments:
        raise ValueError("transition certificate contains an invalid semantic fact")
    return {
        "fact_ref": fact_ref,
        "predicate": predicate,
        "args": arguments,
        "cardinality": max(1, int(raw.get("cardinality", 1))),
        "distinct_by": str(raw.get("distinct_by", "")),
    }


def _parameter_role(
    preferred: str,
    value: Any,
    bindings: dict[str, Any],
    role_by_value: dict[str, str],
) -> str:
    value_id = _value_key(value)
    if value_id in role_by_value:
        return role_by_value[value_id]
    base = str(preferred).strip() or "value"
    role = base
    suffix = 2
    while role in bindings and bindings[role] != value:
        role = f"{base}_{suffix}"
        suffix += 1
    bindings[role] = value
    role_by_value[value_id] = role
    return role


def _predicate_from_fact(
    raw: Mapping[str, Any], role_by_value: Mapping[str, str],
) -> SemanticPredicate:
    arguments: dict[str, Any] = {}
    for name, value in dict(raw.get("args") or {}).items():
        role = role_by_value.get(_value_key(value))
        if not role:
            raise ValueError(f"certificate fact argument lacks derived input role: {name}")
        arguments[str(name)] = BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role=role,
        )
    return SemanticPredicate(
        str(raw.get("predicate", "")),
        arguments,
        max(1, int(raw.get("cardinality", 1))),
        str(raw.get("distinct_by", "")),
    )


def _resolve_output_source(
    source: str,
    *,
    selected_events: list[dict[str, Any]],
    selected_effects: Mapping[str, dict[str, Any]],
) -> Any:
    if source.startswith("argument:"):
        argument = source.removeprefix("argument:")
        values = {
            _value_key(value): value
            for event in selected_events
            for name, value in dict(event.get("arguments") or {}).items()
            if name == argument
        }
        if len(values) != 1:
            raise ValueError(f"output source is not one unique action argument: {source}")
        return next(iter(values.values()))
    if source.startswith("fact:"):
        body = source.removeprefix("fact:")
        if ":" not in body:
            raise ValueError(f"output fact source lacks an argument name: {source}")
        fact_ref, argument = body.rsplit(":", 1)
        fact = selected_effects.get(fact_ref)
        if fact is None or argument not in dict(fact.get("args") or {}):
            raise ValueError(f"output source is not a selected effect argument: {source}")
        return dict(fact["args"])[argument]
    raise ValueError(f"unsupported output source: {source}")


def _resolved_predicate_args(
    predicate: SemanticPredicate, bindings: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw in predicate.args.items():
        if isinstance(raw, BindingExpression):
            result[name] = raw.constant if raw.kind is BindingExprKind.CONSTANT else bindings.get(raw.source_role)
        else:
            result[name] = raw
    return result


def _exact_contract_match(
    target: SemanticPredicate,
    offered: SemanticPredicate,
    offered_args: Mapping[str, Any],
) -> bool:
    return (
        target.predicate.casefold() == offered.predicate.casefold()
        and set(target.args) == set(offered_args)
        and all(target.args[name] == offered_args.get(name) for name in target.args)
    )


class Atomicizer:
    """Certificate reference validator; it contains no environment semantics."""

    def validate_proposed_subset(
        self,
        proposals: list[AtomicBoundaryProposal],
        normalized_trace: dict[str, Any],
        *,
        task_contract: TaskContract | None = None,
        contract_matcher: ContractMatcher | None = None,
    ) -> tuple[list[CanonicalAtomicOccurrence], list[dict[str, str]]]:
        validated: list[tuple[AtomicBoundaryProposal, CanonicalAtomicOccurrence]] = []
        rejections: list[dict[str, str]] = []
        seen_phases: set[str] = set()
        for proposal in proposals:
            if not proposal.phase_id or proposal.phase_id in seen_phases:
                rejections.append({
                    "phase_id": str(proposal.phase_id),
                    "error_type": "ValueError",
                    "error": f"duplicate/empty Atomic phase id: {proposal.phase_id!r}",
                })
                continue
            seen_phases.add(proposal.phase_id)
            try:
                validated.append((proposal, self._validate_one(proposal, normalized_trace)))
            except ValueError as exc:
                rejections.append({
                    "phase_id": str(proposal.phase_id),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

        contract = task_contract or _task_contract(normalized_trace)

        def contribution(item: CanonicalAtomicOccurrence) -> int:
            bindings = {**item.input_bindings, **item.output_bindings}
            score = 0
            for target in contract.target_effects:
                matched_cardinality = 0
                distinct_values: set[str] = set()
                for effect in item.effects:
                    arguments = _resolved_predicate_args(effect, bindings)
                    if contract_matcher is not None:
                        matched = contract_matcher.effect_covers_target(
                            offered_predicate=effect,
                            offered_arguments=arguments,
                            target_predicate=target,
                        )
                    else:
                        matched = _exact_contract_match(target, effect, arguments)
                    if not matched:
                        continue
                    matched_cardinality += max(1, int(effect.cardinality))
                    if target.distinct_by:
                        value = arguments.get(target.distinct_by)
                        if value not in (None, ""):
                            distinct_values.add(_value_key(value))
                available = (
                    len(distinct_values)
                    if target.distinct_by
                    else matched_cardinality
                )
                score += min(max(1, int(target.cardinality)), available)
            return score

        # Resolve overlap winners independently of Agent ordering.
        priority = sorted(
            validated,
            key=lambda pair: (
                -contribution(pair[1]),
                pair[0].event_end_exclusive - pair[0].event_start,
                -len(pair[0].selected_effect_refs),
                pair[0].event_start,
                pair[0].phase_id,
            ),
        )
        accepted: list[CanonicalAtomicOccurrence] = []
        for proposal, occurrence in priority:
            if any(
                proposal.event_start < current.event_end_exclusive
                and current.event_start < proposal.event_end_exclusive
                for current in accepted
            ):
                rejections.append({
                    "phase_id": proposal.phase_id,
                    "error_type": "OverlapResolution",
                    "error": "Atomic proposal lost deterministic overlap resolution",
                })
                continue
            accepted.append(occurrence)
        accepted.sort(key=lambda item: (
            item.event_start,
            item.event_end_exclusive,
            item.phase_id,
        ))
        if not accepted:
            detail = rejections[0]["error"] if rejections else "no proposals"
            raise ValueError(f"Extractor E1 produced no valid Atomic occurrences: {detail}")
        return accepted, rejections

    def validate_and_canonicalize(
        self,
        proposals: list[AtomicBoundaryProposal],
        normalized_trace: dict[str, Any],
    ) -> list[CanonicalAtomicOccurrence]:
        accepted, rejections = self.validate_proposed_subset(proposals, normalized_trace)
        if rejections:
            raise ValueError(rejections[0]["error"])
        return accepted

    def _validate_one(
        self,
        proposal: AtomicBoundaryProposal,
        normalized_trace: dict[str, Any],
    ) -> CanonicalAtomicOccurrence:
        events = list(normalized_trace.get("actions") or [])
        if not (
            0 <= proposal.event_start < proposal.event_end_exclusive <= len(events)
        ):
            raise ValueError(f"invalid half-open event range for {proposal.phase_id}")
        selected = events[proposal.event_start:proposal.event_end_exclusive]
        if not selected or not all(bool(item.get("accepted")) for item in selected):
            raise ValueError(f"Atomic proposal contains rejected/no events: {proposal.phase_id}")
        if any(
            int(item.get("after_revision", -1)) <= int(item.get("before_revision", -1))
            for item in selected
        ):
            raise ValueError(f"Atomic proposal lacks an executed revision transition: {proposal.phase_id}")

        selected_span_ids = {str(item.get("span_id", "")) for item in selected}
        if len(selected_span_ids) != 1 or "" in selected_span_ids:
            raise ValueError(f"Atomic proposal crosses incompatible RuntimeSpan: {proposal.phase_id}")
        spans = list(normalized_trace.get("runtime_spans") or [])
        if spans and not any(
            int(span.get("action_start", 0)) <= proposal.event_start
            and int(span.get("action_end", len(events))) >= proposal.event_end_exclusive
            and str(span.get("span_id", "")) in selected_span_ids
            for span in spans
        ):
            raise ValueError(f"Atomic proposal crosses incompatible RuntimeSpan: {proposal.phase_id}")

        effects_by_ref: dict[str, dict[str, Any]] = {}
        effect_event_index: dict[str, int] = {}
        required_by_ref: dict[str, dict[str, Any]] = {}
        certificate_evidence: list[str] = []
        produced_fact_ids: set[tuple[str, str]] = set()
        latest_negative_index: dict[tuple[str, str], int] = {}

        def semantic_fact_id(item: Mapping[str, Any]) -> tuple[str, str]:
            return (
                str(item.get("predicate", "")),
                repr(sorted(dict(item.get("args") or {}).items())),
            )

        for local_index, event in enumerate(selected):
            certificate = event.get("transition_certificate")
            if not isinstance(certificate, dict):
                raise ValueError(f"selected event lacks transition certificate: {proposal.phase_id}")
            if certificate.get("accepted") is not True:
                raise ValueError(f"certificate rejects selected event: {proposal.phase_id}")
            for raw in list(certificate.get("required_facts") or []):
                item = _fact(raw)
                if semantic_fact_id(item) in produced_fact_ids:
                    continue
                previous = required_by_ref.get(item["fact_ref"])
                if previous is not None and previous != item:
                    raise ValueError("one precondition reference resolves to conflicting facts")
                required_by_ref[item["fact_ref"]] = item
            for raw in [
                *list(certificate.get("positive_effects") or []),
                *list(certificate.get("terminal_effects") or []),
            ]:
                item = _fact(raw)
                previous = effects_by_ref.get(item["fact_ref"])
                if previous is not None and previous != item:
                    raise ValueError("one effect reference resolves to conflicting facts")
                effects_by_ref[item["fact_ref"]] = item
                effect_event_index[item["fact_ref"]] = local_index
                produced_fact_ids.add(semantic_fact_id(item))
            for raw in list(certificate.get("negative_effects") or []):
                item = _fact(raw)
                latest_negative_index[semantic_fact_id(item)] = local_index
            certificate_evidence.extend(map(str, certificate.get("evidence_refs") or []))

        if len(set(proposal.selected_effect_refs)) != len(proposal.selected_effect_refs):
            raise ValueError("selected effect references must be unique")
        if len(set(proposal.selected_precondition_refs)) != len(proposal.selected_precondition_refs):
            raise ValueError("selected precondition references must be unique")
        if not proposal.selected_effect_refs:
            raise ValueError(f"Atomic proposal requires selected effects: {proposal.phase_id}")
        missing_effects = sorted(set(proposal.selected_effect_refs) - set(effects_by_ref))
        missing_preconditions = sorted(
            set(proposal.selected_precondition_refs) - set(required_by_ref)
        )
        if missing_effects:
            raise ValueError(f"unknown/out-of-boundary effect references: {missing_effects}")
        if missing_preconditions:
            raise ValueError(
                f"unknown/out-of-boundary precondition references: {missing_preconditions}"
            )
        omitted_preconditions = sorted(
            set(required_by_ref) - set(proposal.selected_precondition_refs)
        )
        if omitted_preconditions:
            raise ValueError(
                f"required certificate preconditions were omitted: {omitted_preconditions}"
            )
        selected_effects = {
            ref: effects_by_ref[ref] for ref in proposal.selected_effect_refs
        }
        for ref, item in selected_effects.items():
            if latest_negative_index.get(semantic_fact_id(item), -1) > effect_event_index[ref]:
                raise ValueError(
                    f"selected effect is negated within the Atomic boundary: {ref}"
                )
        selected_preconditions = {
            ref: required_by_ref[ref] for ref in proposal.selected_precondition_refs
        }

        inputs: dict[str, Any] = {}
        role_by_value: dict[str, str] = {}
        for raw in [*selected_preconditions.values(), *selected_effects.values()]:
            for argument, value in dict(raw.get("args") or {}).items():
                _parameter_role(str(argument), value, inputs, role_by_value)
        for event in selected:
            for argument, value in dict(event.get("arguments") or {}).items():
                _parameter_role(str(argument), value, inputs, role_by_value)
        if not inputs:
            raise ValueError(f"certificate boundary derives no reusable inputs: {proposal.phase_id}")

        if not proposal.output_role_mapping:
            raise ValueError(f"Atomic proposal requires output role mappings: {proposal.phase_id}")
        outputs: dict[str, Any] = {}
        effect_values = {
            _value_key(value)
            for item in selected_effects.values()
            for value in dict(item.get("args") or {}).values()
        }
        for role, source in sorted(proposal.output_role_mapping.items()):
            if not role:
                raise ValueError("output role names must be non-empty")
            value = _resolve_output_source(
                source,
                selected_events=selected,
                selected_effects=selected_effects,
            )
            if _value_key(value) not in effect_values:
                raise ValueError(f"output source is not established by a selected effect: {source}")
            outputs[role] = value

        preconditions = [
            _predicate_from_fact(item, role_by_value)
            for item in selected_preconditions.values()
        ]
        effects = [
            _predicate_from_fact(item, role_by_value)
            for item in selected_effects.values()
        ]
        input_specs = [
            ParameterSpec(
                role,
                _semantic_type(value),
                True,
                True,
                "concrete" if isinstance(value, str) else "semantic",
            )
            for role, value in sorted(inputs.items())
        ]
        output_specs = [
            ParameterSpec(role, _semantic_type(value), True, False, "semantic")
            for role, value in sorted(outputs.items())
        ]
        signature = content_hash({
            "inputs": input_specs,
            "outputs": output_specs,
            "preconditions": preconditions,
            "effects": effects,
        })[:24]
        occurrence_id = (
            f"occ_{normalized_trace['trace_id']}_{proposal.event_start:04d}_"
            f"{content_hash(proposal.phase_id)[:8]}"
        )
        refs = list(dict.fromkeys([
            f"trace:{normalized_trace['trace_id']}:events:"
            f"{proposal.event_start}-{proposal.event_end_exclusive}",
            *proposal.selected_effect_refs,
            *proposal.selected_precondition_refs,
            *certificate_evidence,
        ]))
        return CanonicalAtomicOccurrence(
            occurrence_id=occurrence_id,
            phase_id=proposal.phase_id,
            intent=proposal.intent,
            event_start=proposal.event_start,
            event_end_exclusive=proposal.event_end_exclusive,
            input_bindings=inputs,
            output_bindings=outputs,
            input_specs=input_specs,
            output_specs=output_specs,
            preconditions=preconditions,
            effects=effects,
            action_events=selected,
            prefix_events=list(events[:proposal.event_start]),
            source_task=dict(normalized_trace.get("source_task") or {}),
            source_trace_id=str(normalized_trace["trace_id"]),
            proposed_ref=SkillRef(f"atomic_{signature}", "1.0.0"),
            validation_refs=refs,
            selected_effect_refs=list(proposal.selected_effect_refs),
            selected_precondition_refs=list(proposal.selected_precondition_refs),
        )


def _task_contract(normalized_trace: Mapping[str, Any]) -> TaskContract:
    raw = dict(normalized_trace.get("task_contract") or {})
    effects = [
        SemanticPredicate(
            str(item.get("predicate", "")),
            dict(item.get("args") or {}),
            int(item.get("cardinality", 1)),
            str(item.get("distinct_by", "")),
        )
        for item in raw.get("target_effects", [])
    ]
    return TaskContract(target_effects=effects)


__all__ = [
    "AtomicBoundaryProposal",
    "AtomicOccurrenceProposal",
    "Atomicizer",
    "CanonicalAtomicOccurrence",
]
