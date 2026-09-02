"""Occurrence evidence continuity and compact exploration memory.

The objects in this module are task-local code authority.  They never parse
Agent prose and never infer benchmark workflows.  Current facts come only from
the validator channel after real environment transitions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


FactKey = tuple[str, tuple[tuple[str, str], ...]]


def fact_key(value: Mapping[str, Any]) -> FactKey:
    """Return a deterministic identity for one validator fact."""

    predicate = str(value.get("predicate", ""))
    arguments = tuple(sorted(
        (str(role), str(argument))
        for role, argument in dict(value.get("args") or {}).items()
    ))
    return predicate, arguments


def normalized_facts(snapshot: Mapping[str, Any] | None) -> dict[FactKey, dict[str, Any]]:
    """Project a validator snapshot to serializable positive facts only."""

    result: dict[FactKey, dict[str, Any]] = {}
    for raw in list(dict(snapshot or {}).get("facts") or []):
        if not isinstance(raw, Mapping):
            continue
        key = fact_key(raw)
        if not key[0]:
            continue
        result[key] = {
            "predicate": key[0],
            "args": dict(key[1]),
        }
    return result


@dataclass
class OccurrenceAtomicEvidenceState:
    """Accepted-action facts owned by one Runtime occurrence.

    ``_current_world`` is the last authoritative validator snapshot.  A fact
    enters ``_owned`` only when an accepted action creates/re-establishes it
    during this occurrence.  Owned facts remain usable across Agent turns and
    sessions while they remain true in the current world.
    """

    occurrence_id: str
    started_revision: int
    _current_world: dict[FactKey, dict[str, Any]] = field(default_factory=dict)
    _owned: dict[FactKey, dict[str, Any]] = field(default_factory=dict)
    _first_seen_revision: dict[FactKey, int] = field(default_factory=dict)
    _invalidated_revision: dict[FactKey, int] = field(default_factory=dict)

    @classmethod
    def begin(
        cls,
        occurrence_id: str,
        revision: int,
        validator_snapshot: Mapping[str, Any] | None,
    ) -> "OccurrenceAtomicEvidenceState":
        return cls(
            occurrence_id=str(occurrence_id),
            started_revision=int(revision),
            _current_world=normalized_facts(validator_snapshot),
        )

    def reconcile(
        self,
        validator_snapshot: Mapping[str, Any] | None,
        *,
        revision: int,
        accepted: bool,
    ) -> None:
        """Merge one accepted transition and invalidate facts no longer true."""

        current = normalized_facts(validator_snapshot)
        if accepted:
            for key in current.keys() - self._current_world.keys():
                self._owned[key] = copy.deepcopy(current[key])
                self._first_seen_revision.setdefault(key, int(revision))
                self._invalidated_revision.pop(key, None)
        for key in set(self._owned) - set(current):
            self._invalidated_revision.setdefault(key, int(revision))
        # Re-established facts become authoritative again only when a real
        # accepted transition added them back to the current world above.
        self._current_world = current

    def authoritative_facts(self) -> list[dict[str, Any]]:
        """Return every occurrence-owned fact that is still true now."""

        return [
            copy.deepcopy(self._owned[key])
            for key in sorted(set(self._owned) & set(self._current_world))
        ]

    def invalidated_facts(self) -> list[dict[str, Any]]:
        return [
            {
                **copy.deepcopy(self._owned[key]),
                "invalidated_at_revision": self._invalidated_revision[key],
            }
            for key in sorted(self._invalidated_revision)
            if key not in self._current_world
        ]

    def full_state(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "started_revision": self.started_revision,
            "active_facts": self.authoritative_facts(),
            "invalidated_facts": self.invalidated_facts(),
        }


@dataclass
class ExplorationMemory:
    """Structured historical memory; never a replacement for current truth."""

    visited: set[str] = field(default_factory=set)
    inspected: set[str] = field(default_factory=set)
    opened: set[str] = field(default_factory=set)
    discovered: dict[str, dict[str, Any]] = field(default_factory=dict)
    negative_observations: dict[str, Any] = field(default_factory=dict)
    accepted_actions_since_grounding_change: int = 0
    _last_grounding_signature: Any = None

    @staticmethod
    def _catalog_values(catalog: Iterable[Any]) -> set[str]:
        values: set[str] = set()
        for spec in catalog:
            arguments = (
                dict(spec.get("arguments") or {})
                if isinstance(spec, Mapping)
                else dict(getattr(spec, "arguments", {}) or {})
            )
            for value in arguments.values():
                if isinstance(value, str) and value:
                    values.add(value)
        return values

    def observe_catalog(
        self,
        catalog: Iterable[Any],
        *,
        revision: int,
        current_facts: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        current_values = self._catalog_values(catalog)
        locations: dict[str, str] = {}
        for fact in current_facts:
            arguments = dict(fact.get("args") or {})
            current_values.update(
                value
                for value in arguments.values()
                if isinstance(value, str) and value
            )
            if str(fact.get("predicate", "")) != "object.at_location":
                continue
            obj = str(arguments.get("object", ""))
            location = str(arguments.get("location", ""))
            if obj and location:
                locations[obj] = location
                current_values.update((obj, location))
        for value, entry in self.discovered.items():
            entry["evidence_status"] = (
                "observed" if value in current_values else "historical"
            )
        for value in sorted(current_values):
            entry = self.discovered.setdefault(value, {})
            entry["evidence_status"] = "observed"
            entry["last_seen_revision"] = int(revision)
            if value in locations:
                entry["last_known_location"] = locations[value]

    def record_action(
        self,
        record: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None,
        catalog: Iterable[Any],
        revision: int,
        current_facts: Iterable[Mapping[str, Any]],
    ) -> None:
        if not bool(record.get("accepted")):
            return
        self.accepted_actions_since_grounding_change += 1
        action_type = str(record.get("action_type", "")).upper()
        arguments = dict(record.get("arguments") or {})
        if action_type == "GO_TO":
            destination = str(arguments.get("destination", ""))
            if destination:
                self.visited.add(destination)
        if action_type in {"LOOK", "EXAMINE", "INVENTORY"}:
            for value in arguments.values():
                if isinstance(value, str) and value:
                    self.inspected.add(value)
        if action_type == "OPEN":
            opened = str(
                arguments.get("object")
                or arguments.get("container")
                or arguments.get("destination")
                or ""
            )
            if opened:
                self.opened.add(opened)

        # Negative observations must be structured adapter metadata.  Natural
        # language observations are deliberately never parsed here.
        negative = dict(metadata or {}).get("negative_observations")
        if isinstance(negative, Mapping):
            for key, value in negative.items():
                self.negative_observations[str(key)] = copy.deepcopy(value)
        self.observe_catalog(
            catalog,
            revision=revision,
            current_facts=current_facts,
        )

    def note_grounding_state(self, signature: Any) -> None:
        if self._last_grounding_signature is None:
            self._last_grounding_signature = copy.deepcopy(signature)
            return
        if signature != self._last_grounding_signature:
            self.accepted_actions_since_grounding_change = 0
            self._last_grounding_signature = copy.deepcopy(signature)

    def policy_view(self) -> dict[str, Any]:
        return {
            "visited": sorted(self.visited),
            "inspected": sorted(self.inspected),
            "opened": sorted(self.opened),
            "discovered": {
                key: copy.deepcopy(self.discovered[key])
                for key in sorted(self.discovered)
            },
            "negative_observations": {
                key: copy.deepcopy(self.negative_observations[key])
                for key in sorted(self.negative_observations)
            },
            "progress_since_last_grounding_change": {
                "accepted_actions": self.accepted_actions_since_grounding_change,
            },
        }


__all__ = [
    "ExplorationMemory",
    "FactKey",
    "OccurrenceAtomicEvidenceState",
    "fact_key",
    "normalized_facts",
]
