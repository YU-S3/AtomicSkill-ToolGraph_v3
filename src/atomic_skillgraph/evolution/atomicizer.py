"""Validate E1 event slices and canonicalize instance values into typed roles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import ParameterSpec, SemanticPredicate
from ..core.refs import SkillRef, content_hash


@dataclass
class AtomicOccurrenceProposal:
    phase_id: str
    intent: str
    event_start: int
    event_end: int
    input_roles: dict[str, Any]
    output_roles: dict[str, Any]
    preconditions: list[SemanticPredicate]
    effects: list[SemanticPredicate]
    rationale: str
    support_event_ids: list[str] = field(default_factory=list)
    shared_precondition_event_ids: list[str] = field(default_factory=list)
    precondition_witness_refs: list[str] = field(default_factory=list)
    effect_witness_refs: list[str] = field(default_factory=list)
    ordering_constraints: list[dict[str, Any]] = field(default_factory=list)
    input_provenance_refs: dict[str, Any] = field(default_factory=dict)
    output_derivations: dict[str, Any] = field(default_factory=dict)
    # Explicit migration boundary: only proposals transported through the
    # current E1 schema may claim the v3.2 authority contract.  Older internal
    # replay/promotion fixtures remain isolated on the legacy path.
    input_provenance_contract: str = "legacy_action_argument_v1"


@dataclass
class CanonicalAtomicOccurrence:
    occurrence_id: str
    phase_id: str
    intent: str
    event_start: int
    event_end: int
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
    support_event_ids: list[str] = field(default_factory=list)
    shared_precondition_event_ids: list[str] = field(default_factory=list)
    precondition_witness_refs: list[str] = field(default_factory=list)
    effect_witness_refs: list[str] = field(default_factory=list)
    ordering_constraints: list[dict[str, Any]] = field(default_factory=list)
    envelope_events: list[dict[str, Any]] = field(default_factory=list)
    input_provenance_refs: dict[str, Any] = field(default_factory=dict)
    output_derivations: dict[str, Any] = field(default_factory=dict)


_ACTION_EFFECTS: dict[str, tuple[tuple[str, dict[str, tuple[str, ...]]], ...]] = {
    "TAKE": (("agent.holds", {"object": ("object", "item")}),),
    "PUT": (("object.at_location", {
        "object": ("object", "item"), "location": ("destination", "location"),
    }),),
    "MOVE": (("object.at_location", {
        "object": ("object", "item"), "location": ("destination", "location"),
    }),),
    "HEAT": (("object.heated", {"object": ("object", "item")}),),
    "COOL": (("object.cooled", {"object": ("object", "item")}),),
    "CLEAN": (("object.cleaned", {"object": ("object", "item")}),),
    "SLICE": (("object.sliced", {"object": ("object", "item")}),),
    "GO_TO": (("agent.at_location", {"location": ("destination", "location")}),),
    "OPEN": (("container.open", {"container": ("object", "container")}),),
    "CLOSE": (("container.closed", {"container": ("object", "container")}),),
    "TOGGLE_ON": (("light.on", {"light": ("object", "light")}),),
    "TOGGLE_OFF": (("light.off", {"light": ("object", "light")}),),
    "EXAMINE": (("object.observed", {"object": ("object", "item")}),),
}

# Deterministic domains for the state reducer's harness-derived facts.  The
# current production E1 path additionally projects the Harness predicate
# vocabulary into each fact; this table keeps normalization of historical
# action-only traces explicit and deterministic rather than domainless.
_REDUCED_FACT_DOMAINS: dict[str, str] = {
    "agent.holds": "world",
    "agent.at_location": "world",
    "object.at_location": "world",
    "object.heated": "world",
    "object.cleaned": "world",
    "object.cooled": "world",
    "object.sliced": "world",
    "container.open": "world",
    "container.closed": "world",
    "light.on": "world",
    "light.off": "world",
    "object.observed": "evidence",
    "object.observed_with": "world",
}

# Contextual goal facts that the ALFWorld validator certifies at terminal but
# that cannot be reconstructed from the final action arguments alone.
def _semantic_type(role: str, value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "entity" if any(token in role for token in (
        "object", "source", "location", "station", "destination",
        "entity", "receptacle", "container", "light", "tool",
    )) else "string"


def _resolve(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, dict) and "kind" in value:
        value = BindingExpression.from_dict(value)
    if isinstance(value, BindingExpression):
        if value.kind is BindingExprKind.CONSTANT:
            return value.constant
        return bindings.get(value.source_role)
    if isinstance(value, str) and value.startswith("$"):
        return bindings.get(value[1:])
    return value


def _entity_family(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"(?:_|\s)\d+$", "", value.casefold()))


def _witness_value_equal(expected: Any, observed: Any, *, semantic_family: bool) -> bool:
    if expected == observed:
        return True
    if semantic_family and isinstance(expected, str) and isinstance(observed, str):
        return bool(_entity_family(expected)) and _entity_family(expected) == _entity_family(observed)
    return False


def _fact_matches(
    predicate: SemanticPredicate,
    fact: dict[str, Any],
    bindings: dict[str, Any],
    *,
    require_domain: bool = False,
) -> bool:
    if predicate.predicate.casefold() != str(fact.get("predicate", "")).casefold():
        return False
    expected_domain = str(getattr(predicate.effect_domain, "value", predicate.effect_domain))
    if require_domain and expected_domain.casefold() != str(
        fact.get("effect_domain", "")
    ).casefold():
        return False
    expected = {name: _resolve(value, bindings) for name, value in predicate.args.items()}
    observed = dict(fact.get("args") or {})
    semantic_family = bool(fact.get("semantic_family"))
    return bool(expected) and set(expected) == set(observed) and all(
        _witness_value_equal(value, observed.get(name), semantic_family=semantic_family)
        for name, value in expected.items()
    )


def _predicate_has_witnesses(
    predicate: SemanticPredicate,
    facts: list[dict[str, Any]],
    bindings: dict[str, Any],
    *,
    require_domain: bool = False,
) -> bool:
    matching = [
        fact for fact in facts
        if _fact_matches(
            predicate, fact, bindings, require_domain=require_domain,
        )
    ]
    needed = max(1, int(predicate.cardinality))
    if predicate.distinct_by:
        return len({
            dict(fact.get("args") or {}).get(predicate.distinct_by)
            for fact in matching
            if dict(fact.get("args") or {}).get(predicate.distinct_by) not in {None, ""}
        }) >= needed
    identities = {
        tuple(sorted(dict(fact.get("args") or {}).items()))
        for fact in matching
    }
    return len(identities) >= needed


def _argument(arguments: dict[str, Any], *aliases: str) -> Any:
    return next((arguments[name] for name in aliases if name in arguments), None)


def reduce_action_state(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct the current action-derived state without stale witnesses.

    TraceNormalizer intentionally does not expose private validator snapshots.
    Its fallback therefore has to be a reducer, not a monotonic bag of every
    historical action effect: TAKE→PUT no longer witnesses ``agent.holds``;
    HEAT→COOL no longer witnesses ``object.heated``; and opposing open/light
    transitions replace one another.  Each surviving fact retains the action
    index that most recently established it so an Atomic effect can be tied to
    the selected slice rather than to unrelated prefix history.
    """

    facts: dict[tuple[str, tuple[tuple[str, Any], ...]], dict[str, Any]] = {}
    agent_location: Any = None
    light_locations: dict[Any, Any] = {}

    def key(predicate: str, arguments: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
        return predicate, tuple(sorted(arguments.items()))

    def discard(predicate: str, **arguments: Any) -> None:
        facts.pop(key(predicate, arguments), None)

    def discard_where(predicate: str, role: str, value: Any) -> None:
        for fact_key in list(facts):
            name, items = fact_key
            if name == predicate and dict(items).get(role) == value:
                facts.pop(fact_key, None)

    def add(predicate: str, arguments: dict[str, Any], event: dict[str, Any], index: int) -> None:
        facts[key(predicate, arguments)] = {
            "predicate": predicate,
            "args": arguments,
            "effect_domain": _REDUCED_FACT_DOMAINS[predicate],
            "witness_ref": (
                f"action:{event.get('action_id', event.get('event_index', index))}:"
                f"revision:{event.get('after_revision', '')}"
            ),
            "event_index": int(event.get("event_index", index)),
        }

    def rebuild_observed_with(event: dict[str, Any], index: int) -> None:
        held = {
            dict(items).get("object")
            for predicate, items in facts
            if predicate == "agent.holds"
        }
        lights_on = {
            dict(items).get("light")
            for predicate, items in facts
            if predicate == "light.on"
        }
        desired = {
            key("object.observed_with", {"object": obj, "light": light})
            for obj in held
            for light in lights_on
            if (
                obj is not None
                and light is not None
                and agent_location is not None
                and light_locations.get(light) == agent_location
            )
        }
        current = {
            fact_key for fact_key in facts
            if fact_key[0] == "object.observed_with"
        }
        for stale in current - desired:
            facts.pop(stale, None)
        for novel in desired - current:
            add(
                "object.observed_with",
                dict(novel[1]),
                event,
                index,
            )

    for index, event in enumerate(events):
        if not event.get("accepted"):
            continue
        action = str(event.get("action_type", ""))
        arguments = dict(event.get("arguments") or {})
        obj = _argument(arguments, "object", "item")
        if action == "TAKE" and obj is not None:
            discard_where("object.at_location", "object", obj)
            add("agent.holds", {"object": obj}, event, index)
        elif action in {"PUT", "MOVE"} and obj is not None:
            destination = _argument(arguments, "destination", "location")
            discard("agent.holds", object=obj)
            discard_where("object.at_location", "object", obj)
            if destination is not None:
                add(
                    "object.at_location",
                    {"object": obj, "location": destination},
                    event,
                    index,
                )
        elif action == "HEAT" and obj is not None:
            discard("object.cooled", object=obj)
            add("object.heated", {"object": obj}, event, index)
        elif action == "COOL" and obj is not None:
            discard("object.heated", object=obj)
            add("object.cooled", {"object": obj}, event, index)
        elif action == "CLEAN" and obj is not None:
            add("object.cleaned", {"object": obj}, event, index)
        elif action == "SLICE" and obj is not None:
            add("object.sliced", {"object": obj}, event, index)
        elif action == "GO_TO":
            destination = _argument(arguments, "destination", "location")
            for fact_key in list(facts):
                if fact_key[0] == "agent.at_location":
                    facts.pop(fact_key, None)
            if destination is not None:
                agent_location = destination
                add("agent.at_location", {"location": destination}, event, index)
        elif action in {"OPEN", "CLOSE"} and obj is not None:
            current = "container.open" if action == "OPEN" else "container.closed"
            opposite = "container.closed" if action == "OPEN" else "container.open"
            discard(opposite, container=obj)
            add(current, {"container": obj}, event, index)
        elif action in {"TOGGLE_ON", "TOGGLE_OFF"} and obj is not None:
            current = "light.on" if action == "TOGGLE_ON" else "light.off"
            opposite = "light.off" if action == "TOGGLE_ON" else "light.on"
            discard(opposite, light=obj)
            add(current, {"light": obj}, event, index)
            if action == "TOGGLE_ON" and agent_location is not None:
                light_locations[obj] = agent_location
        elif action == "USE" and obj is not None:
            on_key = key("light.on", {"light": obj})
            if on_key in facts:
                discard("light.on", light=obj)
                add("light.off", {"light": obj}, event, index)
            else:
                discard("light.off", light=obj)
                add("light.on", {"light": obj}, event, index)
                if agent_location is not None:
                    light_locations[obj] = agent_location
        elif action == "EXAMINE" and obj is not None:
            add("object.observed", {"object": obj}, event, index)
        rebuild_observed_with(event, index)

    return sorted(
        facts.values(),
        key=lambda item: (
            str(item["predicate"]),
            tuple(sorted(dict(item["args"]).items())),
        ),
    )


def _normalized_state_facts(
    normalized_trace: dict[str, Any], *, key: str, revision: int,
) -> list[dict[str, Any]]:
    """Read optional canonical before/after facts when a richer Trace provides them."""
    result: list[dict[str, Any]] = []
    for item in normalized_trace.get(key, []):
        item_revision = int(item.get("revision", revision))
        if item_revision != revision:
            continue
        fact = {
            "predicate": str(item.get("predicate", "")),
            "args": dict(item.get("args") or {}),
            "witness_ref": str(item.get("witness_ref") or f"{key}:revision:{revision}"),
        }
        for field_name in (
            "cardinality", "distinct_by", "effect_domain", "event_index",
            "source_kind", "draft_id",
        ):
            if field_name in item:
                fact[field_name] = item[field_name]
        result.append(fact)
    return result


def _canonical_predicate(
    predicate: SemanticPredicate,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    output_derivations: Mapping[str, Any] | None = None,
) -> SemanticPredicate:
    arguments: dict[str, Any] = {}
    combined = {**inputs, **outputs}
    for name, value in predicate.args.items():
        if isinstance(value, BindingExpression):
            if value.kind is not BindingExprKind.CONSTANT and value.source_role not in combined:
                raise ValueError(f"predicate references unknown role: {value.source_role}")
            arguments[name] = value
            continue
        # When an input semantic anchor and a fresh output happen to have the
        # same episode value, plain value matching is ambiguous.  The explicit
        # EFFECT_WITNESS derivation is the authority for the predicate role.
        derived_matches = [
            str(role)
            for role, raw in dict(output_derivations or {}).items()
            if isinstance(raw, Mapping)
            and str(raw.get("kind", "")).casefold() == "effect_witness"
            and str(raw.get("predicate", "")).casefold()
            == predicate.predicate.casefold()
            and str(raw.get("argument_role", "")) == str(name)
            and outputs.get(str(role)) == value
        ]
        matches = derived_matches or [
            role for role, bound in combined.items() if bound == value
        ]
        if isinstance(value, str) and value.startswith("$"):
            matches = [value[1:]]
        if not matches and isinstance(value, str) and re.search(r"(?:_|\s)\d+$", value):
            raise ValueError(f"concrete instance in predicate is not bound to a role: {value}")
        arguments[name] = BindingExpression(BindingExprKind.SKILL_INPUT, source_role=matches[0]) if matches else value
    return SemanticPredicate(
        predicate.predicate, arguments, predicate.cardinality,
        predicate.distinct_by, predicate.effect_domain,
    )


def _authority_ref_identity(authority: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(authority.get("authority_ref", "")),
        str(authority.get("role", "")),
        repr(authority.get("value")),
    )


def _input_authorities(
    normalized_trace: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    through_event: int,
    proposal_event_start: int,
    selected_lineage: frozenset[str],
    support_indices: set[int],
    enforce_runtime_lineage: bool,
) -> list[dict[str, Any]]:
    """Return only code-supplied E1 input authorities.

    Current E1 proposals may not recover provenance by matching concrete
    values against action arguments.  The producer of the E1 context must
    project every admissible authority explicitly into ``boundary_authorities``.
    """

    authorities: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    event_indexes = {
        str(event.get("event_id", event.get("action_id", ""))): int(
            event.get("event_index", index)
        )
        for index, event in enumerate(events)
        if event.get("accepted") is True
        and str(event.get("event_id", event.get("action_id", "")))
    }
    boundary = dict(normalized_trace.get("boundary_authorities") or {})
    runtime_input_kinds = {
        "current_occurrence_anchor",
        "current_confirmed_binding",
        "current_candidate_binding",
        "confirmed_binding",
        "data_flow",
        "validated_output",
        "constant",
        "runtime_input",
    }
    for raw in list(boundary.get("inputs") or []):
        if not isinstance(raw, Mapping):
            continue
        authority = dict(raw)
        if not str(authority.get("authority_ref", "")):
            continue
        authority_kind = str(
            authority.get("kind", authority.get("source_kind", ""))
        ).casefold()
        if authority_kind == "action_argument":
            event_id = str(authority.get("event_id", ""))
            if (
                event_id not in event_indexes
                or event_indexes[event_id] > through_event
            ):
                continue
        is_runtime_input = (
            str(authority.get("authority_ref", "")).startswith(
                "runtime_input:"
            )
            or bool(str(authority.get("draft_id", "")))
            or authority_kind in runtime_input_kinds
        )
        if enforce_runtime_lineage and is_runtime_input:
            draft_id = str(authority.get("draft_id", ""))
            source_occurrence_id = str(
                authority.get("source_occurrence_id", "")
            )
            trial_event_start = authority.get("trial_event_start")
            trial_event_end = authority.get("trial_event_end")
            if (
                not draft_id
                or str(authority.get("authority_ref", ""))
                != f"runtime_input:{draft_id}:{authority.get('role', '')}"
                or not source_occurrence_id
                or source_occurrence_id not in selected_lineage
                or isinstance(trial_event_start, bool)
                or not isinstance(trial_event_start, int)
                or isinstance(trial_event_end, bool)
                or not isinstance(trial_event_end, int)
                or trial_event_start < 0
                or trial_event_end < trial_event_start
                or trial_event_end >= len(events)
                or proposal_event_start < trial_event_start
                or through_event > trial_event_end
                or not support_indices
                or any(
                    event_index < trial_event_start
                    or event_index > trial_event_end
                    for event_index in support_indices
                )
            ):
                continue
        identity = _authority_ref_identity(authority)
        if identity not in seen:
            seen.add(identity)
            authorities.append(authority)
    return authorities


def _legacy_action_input_authorities(
    events: list[dict[str, Any]],
    *,
    through_event: int,
) -> list[dict[str, Any]]:
    """Migration-only authority projection for pre-v3.2 internal proposals."""

    authorities: list[dict[str, Any]] = []
    for event in events[: through_event + 1]:
        if not event.get("accepted"):
            continue
        event_id = str(event.get("event_id", event.get("action_id", "")))
        for role, value in dict(event.get("arguments") or {}).items():
            base = {
                "kind": "action_argument",
                "source_kind": "action_argument",
                "role": str(role),
                "value": value,
                "event_id": event_id,
                "argument_role": str(role),
            }
            authorities.extend([
                {
                    **base,
                    "authority_ref": f"action_arg:{event_id}:{role}",
                },
                {
                    **base,
                    "authority_ref": (
                        f"action:{event_id}:revision:"
                        f"{event.get('after_revision', '')}"
                    ),
                },
            ])
    return authorities


def _resolve_input_authority(
    role: str,
    value: Any,
    ref: str,
    authorities: list[dict[str, Any]],
    *,
    phase_id: str,
) -> dict[str, Any]:
    matches = [
        item for item in authorities
        if str(item.get("authority_ref", "")) == str(ref)
    ]
    if len(matches) != 1:
        raise ValueError(f"input authority ref not found: {ref}")
    authority = dict(matches[0])
    if str(authority.get("role", "")) != str(role):
        raise ValueError(
            f"input authority role mismatch for {role}: {ref}"
        )
    if repr(authority.get("value")) != repr(value):
        raise ValueError(
            f"input authority value mismatch for {role}: {ref}"
        )
    return authority


def _normalize_output_derivations(
    proposal: Any,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    *,
    phase_id: str,
    require_explicit: bool = False,
) -> dict[str, Any]:
    derivations: dict[str, Any] = {}
    supplied = dict(getattr(proposal, "output_derivations", None) or {})
    if require_explicit and set(supplied) != set(outputs):
        raise ValueError(
            "Atomic output derivation roles must exactly match output roles: "
            f"{phase_id}"
        )
    for output_role, value in outputs.items():
        raw = supplied.get(output_role)
        if raw is None and not require_explicit:
            matches = [
                role for role, bound in inputs.items() if bound == value
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Atomic output lacks one authoritative derivation: "
                    f"{phase_id}.{output_role}"
                )
            derivations[output_role] = {
                "kind": "input_identity",
                "input_role": matches[0],
            }
            continue
        derivation = dict(raw) if isinstance(raw, Mapping) else {}
        kind = str(
            derivation.get("kind")
            if derivation.get("kind") is not None
            else (
                "" if require_explicit else derivation.get("type", "")
            )
        ).casefold()
        if kind == "input_identity":
            allowed_shape = (
                {"kind", "input_role"}
                if require_explicit
                else ({"kind", "input_role"}, {"type", "input_role"})
            )
            shape_valid = (
                set(derivation) == allowed_shape
                if require_explicit
                else set(derivation) in allowed_shape
            )
            if not shape_valid:
                raise ValueError(
                    "Atomic input_identity derivation shape invalid: "
                    f"{phase_id}.{output_role}"
                )
            input_role = str(derivation.get("input_role", ""))
            if input_role not in inputs or inputs[input_role] != value:
                raise ValueError(
                    f"Atomic input_identity derivation invalid: {phase_id}.{output_role}"
                )
            derivations[output_role] = {
                "kind": "input_identity",
                "input_role": input_role,
            }
        elif kind == "effect_witness":
            expected_shape = {"kind", "predicate", "argument_role"}
            legacy_shape = {"type", "predicate", "argument_role"}
            valid_shapes = (
                (expected_shape,)
                if require_explicit
                else (expected_shape, legacy_shape)
            )
            if set(derivation) not in valid_shapes:
                raise ValueError(
                    "Atomic effect_witness derivation shape invalid: "
                    f"{phase_id}.{output_role}"
                )
            derivations[output_role] = {
                "kind": "effect_witness",
                "predicate": str(derivation.get("predicate", "")),
                "argument_role": str(derivation.get("argument_role", "")),
            }
        else:
            raise ValueError(
                f"unsupported Atomic output derivation: {phase_id}.{output_role}"
            )
    return derivations


def _validate_effect_witness_derivations(
    derivations: dict[str, Any],
    proposal: Any,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    effect_facts: list[dict[str, Any]],
    *,
    phase_id: str,
    require_domain: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    validated: dict[str, Any] = {}
    declared_effects = {
        item.predicate.casefold(): item for item in proposal.effects
    }
    extra_refs: list[str] = []
    for output_role, derivation in derivations.items():
        if derivation.get("kind") != "effect_witness":
            validated[output_role] = dict(derivation)
            continue
        predicate = str(derivation.get("predicate", "")).casefold()
        argument_role = str(derivation.get("argument_role", ""))
        effect = declared_effects.get(predicate)
        if effect is None:
            raise ValueError(
                f"Atomic effect_witness derivation references undeclared Effect: {phase_id}.{output_role}"
            )
        if argument_role not in {str(role) for role in effect.args}:
            raise ValueError(
                f"Atomic effect_witness derivation role invalid: {phase_id}.{output_role}"
            )
        matching = [
            fact for fact in effect_facts
            if str(fact.get("predicate", "")).casefold() == predicate
            and (
                not require_domain
                or str(fact.get("effect_domain", "")).casefold()
                == str(
                    getattr(effect.effect_domain, "value", effect.effect_domain)
                ).casefold()
            )
            and repr(dict(fact.get("args") or {}).get(argument_role))
            == repr(outputs[output_role])
        ]
        if not matching:
            raise ValueError(
                f"Atomic effect_witness derivation lacks witness: {phase_id}.{output_role}"
            )
        refs = [
            str(fact.get("witness_ref", ""))
            for fact in matching
            if str(fact.get("witness_ref", ""))
        ]
        if refs:
            extra_refs.extend(refs)
        validated[output_role] = {
            "kind": "effect_witness",
            "predicate": effect.predicate,
            "argument_role": argument_role,
            "witness_refs": list(dict.fromkeys(refs)),
        }
    return validated, list(dict.fromkeys(extra_refs))


class Atomicizer:
    def validate_proposed_subset(
        self,
        proposals: list[AtomicOccurrenceProposal],
        normalized_trace: dict[str, Any],
    ) -> tuple[list[CanonicalAtomicOccurrence], list[dict[str, str]]]:
        """Reject invalid Agent proposals without fabricating replacement occurrences.

        A semantically invalid exploration proposal must not discard unrelated,
        code-validated causal occurrences from the same E1 submission.  The
        accepted proposals remain exactly Agent-proposed.  Composite admission
        later remains fail-closed when the accepted subset cannot cover the
        TaskContract.
        """

        accepted: list[AtomicOccurrenceProposal] = []
        canonical: list[CanonicalAtomicOccurrence] = []
        rejections: list[dict[str, str]] = []
        for proposal in proposals:
            try:
                candidate = self.validate_and_canonicalize(
                    [*accepted, proposal], normalized_trace,
                )
            except ValueError as exc:
                rejections.append({
                    "phase_id": str(proposal.phase_id),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            accepted.append(proposal)
            canonical = candidate
        if not canonical:
            detail = rejections[0]["error"] if rejections else "no proposals"
            raise ValueError(f"Extractor E1 produced no valid Atomic occurrences: {detail}")
        return canonical, rejections

    def validate_and_canonicalize(
        self, proposals: list[AtomicOccurrenceProposal], normalized_trace: dict[str, Any],
    ) -> list[CanonicalAtomicOccurrence]:
        events = normalized_trace.get("actions", [])
        spans = normalized_trace.get("runtime_spans", [])
        span_by_id = {
            str(span.get("span_id", "")): span
            for span in spans
            if str(span.get("span_id", ""))
        }

        def span_lineage(span_id: str) -> frozenset[str]:
            occurrences: set[str] = set()
            seen: set[str] = set()
            while span_id and span_id not in seen:
                seen.add(span_id)
                span = span_by_id.get(span_id)
                if span is None:
                    break
                occurrence_id = str(span.get("occurrence_id", ""))
                if occurrence_id:
                    occurrences.add(occurrence_id)
                parent_id = str(span.get("parent_span_id", "") or "")
                if not parent_id or parent_id == span_id:
                    break
                span_id = parent_id
            return frozenset(occurrences)

        result: list[CanonicalAtomicOccurrence] = []
        used_support_events: set[str] = set()
        used_effect_events: set[str] = set()
        used_effect_witnesses: set[str] = set()
        used_shared_precondition_events: set[str] = set()
        for proposal in proposals:
            current_e1_authority = (
                proposal.input_provenance_contract
                == "code_authority_v3_2"
            )
            if not proposal.phase_id or any(item.phase_id == proposal.phase_id for item in result):
                raise ValueError(f"duplicate/empty Atomic phase id: {proposal.phase_id!r}")
            if not (0 <= proposal.event_start <= proposal.event_end < len(events)):
                raise ValueError(f"invalid event range for {proposal.phase_id}")
            envelope_events = events[proposal.event_start: proposal.event_end + 1]
            if not envelope_events or not all(item.get("accepted") for item in envelope_events):
                raise ValueError(f"Atomic proposal contains rejected/no events: {proposal.phase_id}")
            if current_e1_authority and not proposal.support_event_ids:
                raise ValueError(
                    f"Atomic support_event_ids must be explicit: {proposal.phase_id}"
                )
            if len(set(map(str, proposal.support_event_ids))) != len(
                proposal.support_event_ids
            ):
                raise ValueError(
                    f"Atomic support_event_ids contain duplicates: {proposal.phase_id}"
                )
            if proposal.support_event_ids:
                envelope_by_id = {
                    str(item.get("event_id", item.get("action_id", ""))): item
                    for item in envelope_events
                }
                if not current_e1_authority:
                    envelope_by_id.update({
                        str(item.get("event_index", index)): item
                        for index, item in enumerate(
                            envelope_events, proposal.event_start,
                        )
                    })
                selected = []
                for event_id in proposal.support_event_ids:
                    event = envelope_by_id.get(str(event_id))
                    if event is None:
                        raise ValueError(
                            "support event outside evidence envelope: "
                            f"{event_id}"
                        )
                    selected.append(event)
            else:
                # Explicit legacy migration boundary. Current E1 submissions
                # cannot enter this envelope-as-support compatibility path.
                selected = list(envelope_events)
            if not selected or not all(item.get("accepted") for item in selected):
                raise ValueError(f"Atomic proposal contains rejected/no events: {proposal.phase_id}")
            owned_support_events = {
                str(item.get("event_id", item.get("action_id", "")))
                for item in selected
            }
            shared_precondition_events = {
                str(item) for item in proposal.shared_precondition_event_ids
            }
            if not shared_precondition_events.issubset(
                owned_support_events
            ):
                raise ValueError(
                    "shared precondition event must be one selected support "
                    f"event: {proposal.phase_id}"
                )
            overlap = owned_support_events & used_support_events
            unmarked_overlap = overlap - (
                shared_precondition_events
                | used_shared_precondition_events
            )
            if unmarked_overlap:
                raise ValueError(
                    "Atomic support events are already owned by another "
                    f"independent Atomic: {sorted(unmarked_overlap)}"
                )
            if any(
                int(item.get("after_revision", -1)) <= int(item.get("before_revision", -1))
                for item in selected
            ):
                raise ValueError(f"Atomic proposal lacks a real revision transition: {proposal.phase_id}")
            selected_span_ids = {str(item.get("span_id", "")) for item in selected}
            support_indices = {
                int(item.get("event_index", events.index(item)))
                for item in selected
            }
            if "" in selected_span_ids:
                raise ValueError(f"Atomic proposal lacks a RuntimeSpan: {proposal.phase_id}")
            orphan_span_ids = selected_span_ids - set(span_by_id)
            if orphan_span_ids:
                raise ValueError(
                    "noncontiguous_evidence_lineage_invalid: orphan "
                    f"RuntimeSpan(s) {sorted(orphan_span_ids)} for "
                    f"{proposal.phase_id}"
                )
            lineages = {span_lineage(span_id) for span_id in selected_span_ids}
            if len(lineages) > 1:
                raise ValueError(f"noncontiguous_evidence_lineage_invalid: {proposal.phase_id}")
            lineage = next(iter(lineages)) if lineages else frozenset()
            if len(lineage) > 1:
                raise ValueError(f"Atomic proposal crosses incompatible RuntimeSpan: {proposal.phase_id}")
            if not lineage and len(selected_span_ids) > 1:
                raise ValueError(f"noncontiguous_evidence_lineage_invalid: {proposal.phase_id}")
            containing = [
                span for span in spans
                if span.get("action_start", 0) <= proposal.event_start
                and span.get("action_end", len(events)) >= proposal.event_end + 1
                and (
                    str(span.get("span_id", "")) in selected_span_ids
                    or str(span.get("occurrence_id", "")) in lineage
                )
            ]
            if spans and not containing:
                raise ValueError(f"Atomic proposal crosses incompatible RuntimeSpan: {proposal.phase_id}")
            if not proposal.input_roles:
                raise ValueError("Atomic occurrence requires explicit input roles")
            inputs = dict(proposal.input_roles)
            outputs = dict(proposal.output_roles)
            if len({repr(value) for value in inputs.values()}) != len(inputs):
                raise ValueError(f"Atomic input identity is ambiguous: {proposal.phase_id}")
            authorities = _input_authorities(
                normalized_trace,
                events,
                through_event=proposal.event_end,
                proposal_event_start=proposal.event_start,
                selected_lineage=lineage,
                support_indices=support_indices,
                enforce_runtime_lineage=current_e1_authority,
            )
            supplied_input_refs = dict(
                proposal.input_provenance_refs or {}
            )
            if current_e1_authority and set(supplied_input_refs) != set(inputs):
                raise ValueError(
                    "Atomic input provenance roles must exactly match input "
                    f"roles: {proposal.phase_id}"
                )
            if not current_e1_authority:
                authorities.extend(_legacy_action_input_authorities(
                    events, through_event=proposal.event_end,
                ))
            input_provenance: dict[str, Any] = {}
            for role, value in inputs.items():
                ref = str(supplied_input_refs.get(role, ""))
                if not ref:
                    if current_e1_authority:
                        raise ValueError(
                            "Atomic input provenance ref is empty: "
                            f"{proposal.phase_id}.{role}"
                        )
                    matches = [
                        item for item in authorities
                        if repr(item.get("value")) == repr(value)
                    ]
                    if not matches:
                        raise ValueError(
                            "Atomic input lacks code authority: "
                            f"{proposal.phase_id}.{role}"
                        )
                    authority = dict(matches[0])
                else:
                    authority = _resolve_input_authority(
                        role, value, ref, authorities,
                        phase_id=proposal.phase_id,
                    )
                input_provenance[role] = authority
            output_derivations = _normalize_output_derivations(
                proposal, inputs, outputs, phase_id=proposal.phase_id,
                require_explicit=current_e1_authority,
            )

            bindings = {**inputs, **outputs}
            prefix_facts = reduce_action_state(list(events[:proposal.event_start]))
            prefix_facts += _normalized_state_facts(
                normalized_trace,
                key="before_state_facts",
                revision=int(selected[0].get("before_revision", 0)),
            )
            if len(set(proposal.precondition_witness_refs)) != len(
                proposal.precondition_witness_refs
            ):
                raise ValueError(
                    "Atomic precondition witness refs contain duplicates: "
                    f"{proposal.phase_id}"
                )
            explicit_precondition_facts = (
                [
                    fact for fact in prefix_facts
                    if str(fact.get("witness_ref", ""))
                    in set(proposal.precondition_witness_refs)
                ]
                if current_e1_authority
                else prefix_facts
            )
            if (
                current_e1_authority
                and proposal.preconditions
                and not proposal.precondition_witness_refs
            ):
                raise ValueError(
                    "Atomic precondition witnesses must be explicit: "
                    f"{proposal.phase_id}"
                )
            for precondition in proposal.preconditions:
                if not _predicate_has_witnesses(
                    precondition, explicit_precondition_facts, bindings,
                    require_domain=current_e1_authority,
                ):
                    raise ValueError(f"Atomic precondition lacks before-state witness: {proposal.phase_id}")
            for witness_ref in proposal.precondition_witness_refs:
                facts = [
                    fact for fact in explicit_precondition_facts
                    if str(fact.get("witness_ref", "")) == str(witness_ref)
                ]
                if not facts or not any(
                    _fact_matches(
                        precondition, fact, bindings,
                        require_domain=current_e1_authority,
                    )
                    for precondition in proposal.preconditions
                    for fact in facts
                ):
                    raise ValueError(
                        "evidence_witness_ref_invalid: precondition "
                        f"{witness_ref}"
                    )

            effect_facts = [
                fact
                for fact in reduce_action_state(
                    list(events[:proposal.event_end + 1])
                )
                if int(fact.get("event_index", -1)) in support_indices
            ]
            normalized_effect_facts = _normalized_state_facts(
                normalized_trace,
                key="after_state_facts",
                revision=int(selected[-1].get("after_revision", 0)),
            )
            for fact in normalized_effect_facts:
                if str(fact.get("source_kind", "")) == "runtime_trial_r1":
                    event_index = fact.get("event_index")
                    if (
                        not str(fact.get("draft_id", ""))
                        or not str(fact.get("witness_ref", ""))
                        or isinstance(event_index, bool)
                        or not isinstance(event_index, int)
                        or event_index not in support_indices
                    ):
                        continue
                effect_facts.append(fact)
            if len(set(proposal.effect_witness_refs)) != len(
                proposal.effect_witness_refs
            ):
                raise ValueError(
                    "Atomic effect witness refs contain duplicates: "
                    f"{proposal.phase_id}"
                )
            if current_e1_authority and not proposal.effect_witness_refs:
                raise ValueError(
                    f"Atomic effect witnesses must be explicit: {proposal.phase_id}"
                )
            if current_e1_authority:
                effect_facts = [
                    fact for fact in effect_facts
                    if str(fact.get("witness_ref", ""))
                    in set(proposal.effect_witness_refs)
                ]
            unused_witnesses = set(range(len(effect_facts)))
            effect_witness_indexes: list[int] = []
            for effect in proposal.effects:
                matching = [
                    fact_index
                    for fact_index in sorted(unused_witnesses)
                    if _fact_matches(
                        effect,
                        effect_facts[fact_index],
                        bindings,
                        require_domain=current_e1_authority,
                    )
                ]
                required_witnesses = max(1, int(effect.cardinality))
                if len(matching) < required_witnesses:
                    effect_witness_indexes = []
                    break
                selected_witnesses = matching[:required_witnesses]
                effect_witness_indexes.extend(selected_witnesses)
                unused_witnesses.difference_update(selected_witnesses)
            if not proposal.effects or not effect_witness_indexes:
                raise ValueError(f"Atomic effect lacks accepted state/validator witness: {proposal.phase_id}")
            selected_event_by_index = {
                int(item.get("event_index", events.index(item))): str(
                    item.get("event_id", item.get("action_id", ""))
                )
                for item in selected
            }
            effect_event_ids = {
                selected_event_by_index[event_index]
                for fact_index in effect_witness_indexes
                if (
                    (event_index := int(
                        effect_facts[fact_index].get("event_index", -1)
                    ))
                    in selected_event_by_index
                )
            }
            duplicate_effect_owners = effect_event_ids & used_effect_events
            if duplicate_effect_owners:
                raise ValueError(
                    "shared precondition evidence cannot duplicate Effect "
                    "ownership across independent Atomics: "
                    f"{sorted(duplicate_effect_owners)}"
                )

            output_derivations, derivation_effect_refs = (
                _validate_effect_witness_derivations(
                    output_derivations,
                    proposal,
                    inputs,
                    outputs,
                    effect_facts,
                    phase_id=proposal.phase_id,
                    require_domain=current_e1_authority,
                )
            )

            for witness_ref in proposal.effect_witness_refs:
                facts = [
                    fact for fact in effect_facts
                    if str(fact.get("witness_ref", "")) == str(witness_ref)
                ]
                if not facts or not any(
                    _fact_matches(
                        effect, fact, bindings,
                        require_domain=current_e1_authority,
                    )
                    for effect in proposal.effects
                    for fact in facts
                ):
                    raise ValueError(f"evidence_witness_ref_invalid: effect {witness_ref}")

            effect_witness_owners = {
                str(effect_facts[fact_index].get("witness_ref", ""))
                for fact_index in effect_witness_indexes
                if str(effect_facts[fact_index].get("witness_ref", ""))
            }
            effect_witness_owners.update(
                str(ref) for ref in derivation_effect_refs if str(ref)
            )
            effect_witness_owners.update(
                str(ref) for ref in proposal.effect_witness_refs if str(ref)
            )
            duplicate_effect_witnesses = (
                effect_witness_owners & used_effect_witnesses
            )
            if duplicate_effect_witnesses:
                raise ValueError(
                    "shared precondition evidence cannot duplicate Effect "
                    "witness ownership across independent Atomics: "
                    f"{sorted(duplicate_effect_witnesses)}"
                )

            support_event_by_id = {
                str(item.get("event_id", item.get("action_id", ""))): item
                for item in selected
            }
            for ordering in proposal.ordering_constraints:
                before_id = str(ordering.get("before_event_id", ""))
                after_id = str(ordering.get("after_event_id", ""))
                before_event = support_event_by_id.get(before_id)
                after_event = support_event_by_id.get(after_id)
                if before_event is None or after_event is None:
                    raise ValueError(
                        "ordering_constraints may only reference support events: "
                        f"{proposal.phase_id}"
                    )
                if int(before_event.get("after_revision", -1)) >= int(
                    after_event.get("after_revision", -1)
                ):
                    raise ValueError(
                        f"ordering constraint is not revision-ordered: {proposal.phase_id}"
                    )

            # A validated occurrence must also be independently compilable.
            # Episode-local entity instances cannot become constants in a
            # reusable Tool, so every such accepted action argument must be
            # owned by exactly one explicit Atomic input role.  Checking this
            # inside the per-proposal validation boundary prevents one bad E1
            # occurrence from failing compilation for all otherwise valid
            # occurrences in the same extraction.
            input_values = list(inputs.values())
            output_values = list(outputs.values())
            for event in selected:
                for argument, value in dict(event.get("arguments") or {}).items():
                    if not (
                        isinstance(value, str)
                        and re.search(r"(?:_|\s)\d+$", value)
                    ):
                        continue
                    input_owned = input_values.count(value) == 1
                    fresh_output_owned = (
                        output_values.count(value) == 1
                        and any(
                            derivation.get("kind") == "effect_witness"
                            and repr(outputs.get(output_role)) == repr(value)
                            for output_role, derivation in output_derivations.items()
                        )
                    )
                    if not (input_owned or fresh_output_owned):
                        raise ValueError(
                            "Atomic concrete action argument lacks one reusable "
                            f"input/output derivation: {proposal.phase_id}.{argument}={value}"
                        )

            validation_refs: list[str] = []
            after_revision = int(selected[-1].get("after_revision", 0))
            for validation in normalized_trace.get("validations", []):
                if str(validation.get("level", "")) != "atomic" or int(validation.get("revision", -1)) != after_revision:
                    continue
                validation_result = dict(validation.get("result") or {})
                if validation_result.get("passed") is not True:
                    raise ValueError(f"Atomic validator rejected proposed boundary: {proposal.phase_id}")
                validation_refs.extend(map(str, validation_result.get("witness_refs", [])))
            validation_refs.extend(
                str(effect_facts[fact_index]["witness_ref"])
                for fact_index in effect_witness_indexes
            )
            validation_refs.extend(derivation_effect_refs)
            preconditions = [
                _canonical_predicate(
                    item, inputs, outputs, output_derivations,
                )
                for item in proposal.preconditions
            ]
            effects = [
                _canonical_predicate(
                    item, inputs, outputs, output_derivations,
                )
                for item in proposal.effects
            ]
            input_specs = [
                ParameterSpec(role, _semantic_type(role, value), True, True, "concrete" if _semantic_type(role, value) == "entity" else "semantic")
                for role, value in sorted(inputs.items())
            ]
            output_specs = [ParameterSpec(role, _semantic_type(role, value), True, False, "semantic") for role, value in sorted(outputs.items())]
            logical_id = "atomic_" + re.sub(r"[^a-z0-9]+", "_", proposal.intent.casefold()).strip("_")[:40]
            signature = content_hash({
                "intent": proposal.intent, "inputs": input_specs,
                "outputs": output_specs, "preconditions": preconditions,
                "effects": effects,
            })[:12]
            result.append(CanonicalAtomicOccurrence(
                "", proposal.phase_id, proposal.intent, proposal.event_start, proposal.event_end,
                inputs, outputs, input_specs, output_specs, preconditions, effects, selected,
                list(events[:proposal.event_start]), dict(normalized_trace.get("source_task") or {}),
                normalized_trace["trace_id"], SkillRef(f"{logical_id}_{signature}", "1.0.0"),
                list(dict.fromkeys([
                    f"trace:{normalized_trace['trace_id']}:events:{proposal.event_start}-{proposal.event_end}",
                    *validation_refs,
                ])),
                support_event_ids=[str(item.get("event_id", item.get("action_id", ""))) for item in selected],
                shared_precondition_event_ids=sorted(
                    shared_precondition_events
                ),
                precondition_witness_refs=list(proposal.precondition_witness_refs),
                effect_witness_refs=list(proposal.effect_witness_refs),
                ordering_constraints=[dict(item) for item in proposal.ordering_constraints],
                envelope_events=envelope_events,
                input_provenance_refs=dict(input_provenance),
                output_derivations=dict(output_derivations),
            ))
            used_support_events.update(owned_support_events)
            used_effect_events.update(effect_event_ids)
            used_effect_witnesses.update(effect_witness_owners)
            used_shared_precondition_events.update(
                shared_precondition_events
            )
        if not result:
            raise ValueError("Extractor E1 produced no canonical Atomic occurrence")
        result.sort(key=lambda item: (
            item.event_start, item.event_end, item.phase_id,
        ))
        for index, occurrence in enumerate(result):
            occurrence.occurrence_id = (
                f"occ_{normalized_trace['trace_id']}_{index:03d}"
            )
        return result
