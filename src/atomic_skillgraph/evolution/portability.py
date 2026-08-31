"""Generic portability checks and canonical capability labels.

The validator derives episode-specific terms from concrete bindings.  It has
no benchmark object catalogue and never decides whether a state transition is
valid; it only controls long-term semantic labels after deterministic effect
validation has succeeded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.serialization import to_primitive
from ..core.status import SkillStatus


_GENERIC_ROLE_TERMS = frozenset({
    "object",
    "source",
    "destination",
    "container",
    "location",
    "device",
    "resource",
    "entity",
    "input",
    "output",
    "item",
    "target",
    "station",
    "tool",
    "light",
})
_GENERIC_SEMANTIC_TYPES = frozenset({
    "array",
    "boolean",
    "entity",
    "integer",
    "number",
    "string",
})
_GLOBAL_FORBIDDEN_TERMS = frozenset({
    "alfworld",
    "benchmark",
    "task_family",
    "source_episode",
    "canonical_control_sequence",
    "validated_node",
    "validated_nodes",
})
_SEQUENCE_MARKERS = frozenset({
    "and",
    "then",
    "and_then",
    "followed_by",
    "after_that",
    "before_then",
})
_SOURCE_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "then",
    "to",
    "with",
})


@dataclass(frozen=True)
class PortabilityResult:
    passed: bool
    offending_terms: list[str]
    normalized_text: str


@dataclass(frozen=True)
class CanonicalCapabilityLabel:
    canonical_intent: str
    display_summary: str
    source: str


@dataclass(frozen=True)
class KnownAtomicContractView:
    atomic_ref: str
    canonical_intent: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    preconditions: list[dict[str, Any]]
    effects: list[dict[str, Any]]


def normalize_portable_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            to_primitive(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.casefold())).strip("_")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _strings(item)


def episode_specific_terms(
    input_bindings: Mapping[str, Any],
    output_bindings: Mapping[str, Any],
    role_names: Iterable[str],
    semantic_types: Iterable[str],
) -> set[str]:
    """Derive concrete instance and family terms without an ontology list."""

    allowed = set(_GENERIC_ROLE_TERMS)
    # Role names originate in the E1 proposal, so an arbitrary role such as
    # ``cellphone`` must not be able to whitelist the concrete family
    # ``cellphone_1``.  Only the fixed generic vocabulary and the small set of
    # code-derived scalar/entity types may extend the allow set.
    for value in role_names:
        normalized = normalize_portable_text(value)
        if normalized in _GENERIC_ROLE_TERMS:
            allowed.add(normalized)
    for value in semantic_types:
        normalized = normalize_portable_text(value)
        if normalized in _GENERIC_SEMANTIC_TYPES:
            allowed.add(normalized)

    result: set[str] = set()
    for raw in _strings({
        "inputs": dict(input_bindings),
        "outputs": dict(output_bindings),
    }):
        normalized = normalize_portable_text(raw)
        if not normalized:
            continue
        family = re.sub(r"(?:_\d+)+$", "", normalized).strip("_")
        for candidate in (normalized, family):
            if (
                candidate
                and candidate not in allowed
                and not candidate.isdigit()
            ):
                result.add(candidate)
    return result


def validate_portability(
    text: Any,
    *,
    episode_terms: Iterable[str] = (),
    additional_forbidden_terms: Iterable[str] = (),
    require_intent: bool = False,
) -> PortabilityResult:
    normalized = normalize_portable_text(text)
    forbidden = {
        normalize_portable_text(item)
        for item in (
            *tuple(_GLOBAL_FORBIDDEN_TERMS),
            *tuple(episode_terms),
            *tuple(additional_forbidden_terms),
        )
        if normalize_portable_text(item)
    }
    padded = f"_{normalized}_"
    offending = sorted({
        term
        for term in forbidden
        if f"_{term}_" in padded
    })
    if require_intent:
        raw = str(text)
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", raw):
            offending.append("intent_not_lower_snake_case")
        if (
            re.search(r"(?:^|_)\d+(?:_|$)", raw)
            or re.search(r"[a-z]\d+(?:_|$)", raw)
        ):
            offending.append("instance_identifier")
        if any(
            f"_{marker}_" in padded
            for marker in _SEQUENCE_MARKERS
        ):
            offending.append("multiple_intents")
    offending = sorted(set(offending))
    return PortabilityResult(bool(normalized) and not offending, offending, normalized)


def contract_label(effects: Iterable[Any], outputs: Iterable[Any] = ()) -> str:
    del outputs  # reserved for deterministic disambiguation in future schemas
    predicates = sorted({
        str(
            item.get("predicate", "")
            if isinstance(item, Mapping)
            else getattr(item, "predicate", "")
        )
        for item in effects
        if str(
            item.get("predicate", "")
            if isinstance(item, Mapping)
            else getattr(item, "predicate", "")
        )
    })
    primary = normalize_portable_text(predicates[0]) if predicates else "verified_effect"
    return f"establish_{primary}"


def occurrence_terms(occurrence: Any) -> set[str]:
    specs = [
        *list(getattr(occurrence, "input_specs", ()) or ()),
        *list(getattr(occurrence, "output_specs", ()) or ()),
    ]
    return episode_specific_terms(
        getattr(occurrence, "input_bindings", {}) or {},
        getattr(occurrence, "output_bindings", {}) or {},
        [str(getattr(item, "name", "")) for item in specs],
        [str(getattr(item, "semantic_type", "")) for item in specs],
    )


def source_forbidden_terms(occurrence: Any) -> set[str]:
    source = dict(getattr(occurrence, "source_task", {}) or {})
    # SourceTask already contains the available source goal/provenance text;
    # action observations are intentionally absent from the authoritative E1
    # view.  Traverse the complete source payload so task signatures, goal
    # wording, and nested context/metadata cannot become persistent labels.
    values = [
        value
        for value in _strings(source)
        if normalize_portable_text(value)
    ]
    result = set(values)
    ignored = (
        _SOURCE_STOPWORDS
        | _GENERIC_ROLE_TERMS
        | _GENERIC_SEMANTIC_TYPES
    )
    for value in values:
        tokens = [
            token
            for token in normalize_portable_text(value).split("_")
            if token and token not in ignored
        ]
        # Multi-token fragments catch source task-family/goal wording without
        # banning reusable single capability words such as ``cool`` or
        # ``place``.  This is lexical and task-agnostic; it has no ALFWorld
        # taxonomy or entity dictionary.
        for width in range(2, min(5, len(tokens)) + 1):
            result.update(
                "_".join(tokens[index:index + width])
                for index in range(len(tokens) - width + 1)
            )
    return result


def resolve_capability_label(
    occurrence: Any,
    atomic: Any,
    *,
    existing_atomic: Any | None = None,
) -> CanonicalCapabilityLabel:
    terms = occurrence_terms(occurrence)
    extra = source_forbidden_terms(occurrence)
    if existing_atomic is not None:
        existing = str(
            dict(getattr(existing_atomic, "metadata", {}) or {}).get(
                "canonical_intent", "",
            )
        )
        if existing and validate_portability(
            existing,
            episode_terms=terms,
            additional_forbidden_terms=extra,
            require_intent=True,
        ).passed:
            return CanonicalCapabilityLabel(
                existing,
                existing.replace("_", " "),
                "existing_contract",
            )

    proposed = str(getattr(occurrence, "intent", ""))
    result = validate_portability(
        proposed,
        episode_terms=terms,
        additional_forbidden_terms=extra,
        require_intent=True,
    )
    if result.passed:
        return CanonicalCapabilityLabel(
            proposed,
            proposed.replace("_", " "),
            "llm_portable",
        )
    fallback = contract_label(
        getattr(atomic, "effects", ()) or (),
        getattr(atomic, "outputs", ()) or (),
    )
    return CanonicalCapabilityLabel(
        fallback,
        fallback.replace("_", " "),
        "contract_fallback",
    )


def resolve_capability_label_group(
    candidates: Iterable[tuple[Any, Any]],
    *,
    existing_atomic: Any | None = None,
) -> CanonicalCapabilityLabel:
    """Resolve one deterministic label for an aligned Atomic candidate set.

    All candidates have the same code-derived contract identity.  The
    authority order is persistent contract, any portable LLM proposal, then
    the contract fallback; ordering the candidate rows cannot change it.
    """

    rows = list(candidates)
    if not rows:
        raise ValueError("canonical capability label requires a candidate")
    episode_terms = set().union(*(
        occurrence_terms(occurrence)
        for occurrence, _atomic in rows
    ))
    source_terms = set().union(*(
        source_forbidden_terms(occurrence)
        for occurrence, _atomic in rows
    ))
    if existing_atomic is not None:
        existing = str(
            dict(getattr(existing_atomic, "metadata", {}) or {}).get(
                "canonical_intent", "",
            )
        )
        if existing and validate_portability(
            existing,
            episode_terms=episode_terms,
            additional_forbidden_terms=source_terms,
            require_intent=True,
        ).passed:
            return CanonicalCapabilityLabel(
                existing,
                existing.replace("_", " "),
                "existing_contract",
            )

    portable = sorted({
        str(getattr(occurrence, "intent", ""))
        for occurrence, _atomic in rows
        if validate_portability(
            str(getattr(occurrence, "intent", "")),
            episode_terms=episode_terms,
            additional_forbidden_terms=source_terms,
            require_intent=True,
        ).passed
    })
    if portable:
        return CanonicalCapabilityLabel(
            portable[0],
            portable[0].replace("_", " "),
            "llm_portable",
        )
    fallback = contract_label(
        getattr(rows[0][1], "effects", ()) or (),
        getattr(rows[0][1], "outputs", ()) or (),
    )
    return CanonicalCapabilityLabel(
        fallback,
        fallback.replace("_", " "),
        "contract_fallback",
    )


def relevant_known_atomic_contracts(
    trace: Mapping[str, Any],
    skills: Any,
    *,
    limit: int = 20,
) -> list[KnownAtomicContractView]:
    if isinstance(limit, bool) or int(limit) <= 0:
        raise ValueError("known Atomic contract limit must be positive")
    trace_predicates = {
        str(effect.get("predicate", ""))
        for event in trace.get("actions", ())
        for field in (
            "authoritative_positive_effects",
            "authoritative_terminal_effect_certificates",
        )
        for effect in event.get(field, ())
        if effect.get("predicate")
    }
    candidates: list[tuple[int, int, int, str, Any, str]] = []
    for atomic in skills.atomics():
        if atomic.status not in {SkillStatus.ACTIVE, SkillStatus.CANDIDATE}:
            continue
        overlap = len(
            trace_predicates
            & {str(effect.predicate) for effect in atomic.effects}
        )
        if not overlap:
            continue
        canonical_intent = str(
            dict(atomic.metadata or {}).get("canonical_intent", "")
        )
        if not validate_portability(
            canonical_intent,
            require_intent=True,
        ).passed:
            canonical_intent = contract_label(atomic.effects, atomic.outputs)
        raw_trace_ids = dict(atomic.metadata or {}).get(
            "source_trace_ids", (),
        )
        if isinstance(raw_trace_ids, str):
            raw_trace_ids = (raw_trace_ids,)
        support = len(set(map(str, raw_trace_ids)))
        try:
            row = skills.database.execute(
                "SELECT projection_json FROM lifecycle_projection "
                "WHERE artifact_ref=?",
                (str(atomic.ref),),
            ).fetchone()
            if row is not None:
                projection = json.loads(str(row["projection_json"]))
                support = max(
                    support,
                    int(projection.get("validated_count", 0)),
                )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        status_rank = 1 if atomic.status is SkillStatus.ACTIVE else 0
        candidates.append((
            -status_rank,
            -overlap,
            -support,
            str(atomic.ref),
            atomic,
            canonical_intent,
        ))
    candidates.sort(key=lambda item: item[:4])

    def parameter(item: Any) -> dict[str, Any]:
        return {
            "name": str(item.name),
            "semantic_type": str(item.semantic_type),
            "required": bool(item.required),
            "runtime_resolvable": bool(item.runtime_resolvable),
            "required_resolution": str(item.required_resolution),
        }

    return [
        KnownAtomicContractView(
            atomic_ref=str(atomic.ref),
            canonical_intent=canonical_intent,
            inputs=[parameter(item) for item in atomic.inputs],
            outputs=[parameter(item) for item in atomic.outputs],
            preconditions=[to_primitive(item) for item in atomic.preconditions],
            effects=[to_primitive(item) for item in atomic.effects],
        )
        for _, _, _, _, atomic, canonical_intent in candidates[: int(limit)]
    ]


def composite_fallback_summary(
    canonical_intents: Iterable[str],
    *,
    structure_digest: str,
) -> str:
    intents = [str(item) for item in canonical_intents if str(item)]
    candidate = "compose_" + "_then_".join(intents[:4])
    return candidate if len(candidate) <= 120 else f"compose_{structure_digest[:24]}"


def portable_guideline_fallback(canonical_intents: Iterable[str]) -> dict[str, Any]:
    return {
        "ordered_capabilities": [str(item) for item in canonical_intents],
        "parameter_flow": "preserve_declared_data_flow",
        "dependency_order": "follow_control_order",
    }


__all__ = [
    "CanonicalCapabilityLabel",
    "KnownAtomicContractView",
    "PortabilityResult",
    "composite_fallback_summary",
    "contract_label",
    "episode_specific_terms",
    "occurrence_terms",
    "portable_guideline_fallback",
    "relevant_known_atomic_contracts",
    "resolve_capability_label",
    "resolve_capability_label_group",
    "source_forbidden_terms",
    "validate_portability",
]
