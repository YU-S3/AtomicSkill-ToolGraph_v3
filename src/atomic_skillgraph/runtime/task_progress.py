"""Deterministic task progress derived only from validator-visible facts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..core.bindings import BindingExpression
from ..core.contracts import IdentityRelation, SemanticPredicate, TaskContract
from ..core.serialization import to_primitive
from ..planner.multiplicity import normalize_task_contract
from ..traces.schema import TaskProgressRecord


@dataclass(frozen=True)
class TargetProgress:
    constraint_id: str
    predicate: str
    required_count: int
    satisfied_count: int
    remaining_count: int
    distinct_by: str
    satisfied_witnesses: tuple[dict[str, Any], ...]
    used_distinct_values: tuple[Any, ...]
    shared_values: dict[str, Any]


@dataclass(frozen=True)
class TaskProgressSnapshot:
    revision: int
    targets: tuple[TargetProgress, ...]
    unsatisfied_identity_constraints: tuple[dict[str, Any], ...]
    progress_digest: str


def _base_entity(value: Any) -> str:
    return re.sub(r"(?:_|\s)\d+$", "", str(value).strip().casefold())


def _expected_value(value: Any) -> Any:
    if isinstance(value, BindingExpression):
        return None
    if isinstance(value, dict) and "kind" in value:
        return None
    return value


def _matches(effect: SemanticPredicate, fact: dict[str, Any]) -> bool:
    if str(fact.get("predicate", "")).casefold() != effect.predicate.casefold():
        return False
    args = dict(fact.get("args") or {})
    for role, raw in effect.args.items():
        expected = _expected_value(raw)
        if expected in (None, ""):
            continue
        actual = args.get(role)
        if actual in (None, "") or _base_entity(actual) != _base_entity(expected):
            return False
    return True


class TaskProgressTracker:
    def __init__(
        self,
        contract: TaskContract,
        validator_channel: Any,
        *,
        trace_builder: Any | None = None,
    ) -> None:
        self.contract = normalize_task_contract(contract)
        self.validator_channel = validator_channel
        self.trace_builder = trace_builder
        self._revision = 0
        self._last_digest = ""

    def snapshot(self) -> TaskProgressSnapshot:
        raw = self.validator_channel.snapshot()
        revision = int(raw.get("revision", self._revision))
        facts = [
            dict(value) for value in raw.get("facts", ()) if isinstance(value, dict)
        ]
        constraints_by_predicate = {
            str(value["predicate"]).casefold(): value
            for value in self.contract.cardinality_constraints
        }
        targets: list[TargetProgress] = []
        witness_by_role: dict[str, set[Any]] = {}
        for index, effect in enumerate(self.contract.target_effects):
            constraint = constraints_by_predicate.get(effect.predicate.casefold(), {})
            required_count = max(
                int(effect.cardinality), int(constraint.get("count", 1)),
            )
            distinct_by = str(
                constraint.get("distinct_by") or effect.distinct_by or ""
            )
            matches = [
                dict(fact.get("args") or {}) for fact in facts if _matches(effect, fact)
            ]
            if distinct_by:
                deduplicated: dict[str, dict[str, Any]] = {}
                for witness in matches:
                    value = witness.get(distinct_by)
                    if value not in (None, ""):
                        deduplicated[str(value)] = witness
                matches = [deduplicated[key] for key in sorted(deduplicated)]
            satisfied = min(required_count, len(matches))
            shared_roles = list(constraint.get("shared_roles", ()))
            shared_values = {
                role: sorted({
                    witness[role] for witness in matches if witness.get(role) not in (None, "")
                })
                for role in shared_roles
            }
            for witness in matches:
                for role, value in witness.items():
                    witness_by_role.setdefault(role, set()).add(value)
            targets.append(TargetProgress(
                constraint_id=str(
                    constraint.get("constraint_id")
                    or f"target::{index}::{effect.predicate}"
                ),
                predicate=effect.predicate,
                required_count=required_count,
                satisfied_count=satisfied,
                remaining_count=max(0, required_count - satisfied),
                distinct_by=distinct_by,
                satisfied_witnesses=tuple(matches[:required_count]),
                used_distinct_values=tuple(
                    witness[distinct_by]
                    for witness in matches[:required_count]
                    if distinct_by and witness.get(distinct_by) not in (None, "")
                ),
                shared_values=shared_values,
            ))

        unsatisfied: list[dict[str, Any]] = []
        for identity in self.contract.identity_constraints:
            left_role = re.sub(r"_\d+$", "", identity.left_role)
            right_role = re.sub(r"_\d+$", "", identity.right_role)
            left = witness_by_role.get(left_role, set())
            right = witness_by_role.get(right_role, set())
            passed = False
            if identity.relation is IdentityRelation.SAME_AS:
                passed = bool(left & right)
            else:
                values = left | right
                passed = len(values) >= 2 if left_role == right_role else bool(left and right and left.isdisjoint(right))
            if not passed:
                unsatisfied.append(to_primitive(identity))

        digest_payload = {
            "targets": [to_primitive(value) for value in targets],
            "unsatisfied_identity_constraints": unsatisfied,
        }
        digest = hashlib.sha256(json.dumps(
            digest_payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return TaskProgressSnapshot(
            revision=revision,
            targets=tuple(targets),
            unsatisfied_identity_constraints=tuple(unsatisfied),
            progress_digest=digest,
        )

    def record(self, source: str) -> TaskProgressSnapshot:
        snapshot = self.snapshot()
        if snapshot.progress_digest != self._last_digest:
            self._revision += 1
            record = TaskProgressRecord(
                revision=self._revision,
                source=str(source),
                snapshot=to_primitive(snapshot),
            )
            if self.trace_builder is not None:
                self.trace_builder.trace.task_progress_records.append(record)
            self._last_digest = snapshot.progress_digest
        return snapshot


__all__ = ["TargetProgress", "TaskProgressSnapshot", "TaskProgressTracker"]
