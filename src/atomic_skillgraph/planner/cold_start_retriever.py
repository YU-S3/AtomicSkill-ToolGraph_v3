"""C0 deterministic retrieval from the isolated failure-side bank."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..core.contracts import CapabilityRequirement, TaskContract
from ..core.serialization import to_primitive
from .multiplicity import (
    RequirementExpansion,
    RequirementInstance,
    normalize_task_contract,
    requirement_instance_shape_id,
)


@dataclass(frozen=True)
class ProvisionalAtomicCandidate:
    provisional_ref: str
    status: str
    score: float
    canonical_intent: str
    atomic_contract: dict[str, Any]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FailureExperienceView:
    experience_id: str
    status: str
    requirement_shape_ids: tuple[str, ...]
    validated_prefix_shape_ids: tuple[str, ...]
    first_unrecovered_divergence: dict[str, Any]
    remaining_requirement_shape_ids: tuple[str, ...]
    negative_suffix_summary: dict[str, Any]
    avoid_pattern_codes: tuple[str, ...]
    support_count: int


def _predicate(value: Any) -> tuple[str, set[str], int]:
    if isinstance(value, dict):
        return (
            str(value.get("predicate", "")).casefold(),
            set(map(str, (value.get("args") or {}))),
            int(value.get("cardinality", 1)),
        )
    return (
        str(getattr(value, "predicate", "")).casefold(),
        set(map(str, getattr(value, "args", {}))),
        int(getattr(value, "cardinality", 1)),
    )


def provisional_contract_compatible(
    requirement: CapabilityRequirement,
    atomic_contract: dict[str, Any],
) -> bool:
    offered = [
        _predicate(value) for value in atomic_contract.get("effects", ())
    ]
    for wanted_raw in requirement.desired_effects:
        wanted_name, wanted_roles, wanted_count = _predicate(wanted_raw)
        # Requirement templates inside RepeatBlocks are units.  An isolated
        # provisional never gains aggregate task-level cardinality authority.
        if wanted_count != 1:
            return False
        if not any(
            name == wanted_name and wanted_roles.issubset(roles) and count >= 1
            for name, roles, count in offered
        ):
            return False
    required_types = {
        item.semantic_type for item in requirement.expected_inputs if item.required
    }
    available_types = {
        str(item.get("semantic_type", ""))
        for item in atomic_contract.get("inputs", ())
        if isinstance(item, dict)
    }
    return not required_types or required_types.issubset(available_types)


def task_cluster_signature(
    contract: TaskContract,
    harness_profile: str,
    requirement_expansion: RequirementExpansion | None = None,
) -> str:
    normalized = normalize_task_contract(contract)
    payload = {
        "targets": [
            {
                "predicate": effect.predicate.casefold(),
                "roles": sorted(map(str, effect.args)),
                "cardinality": int(effect.cardinality),
                "distinct_by": str(effect.distinct_by),
            }
            for effect in normalized.target_effects
        ],
        "cardinality": [
            {
                key: value for key, value in constraint.items()
                if key != "constraint_id"
            }
            for constraint in normalized.cardinality_constraints
        ],
        "identity": sorted(
            (item.left_role, item.relation.value, item.right_role, item.scope)
            for item in normalized.identity_constraints
        ),
        # A task cluster describes the semantic boundary shape, not the
        # Planner's incidental role spelling.  Keep a sorted *multiset* so
        # repeated parameters remain significant while aliases such as
        # ``item``/``object`` do not split otherwise identical clusters.
        "semantic_parameter_types": sorted(
            (
                boundary,
                parameter.semantic_type,
                bool(parameter.required),
                parameter.required_resolution,
            )
            for requirement in (
                getattr(requirement_expansion, "templates", ())
                if requirement_expansion is not None else ()
            )
            for boundary, parameters in (
                ("input", requirement.expected_inputs),
                ("output", requirement.expected_outputs),
            )
            for parameter in parameters
        ),
        "harness_profile": harness_profile,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _portable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _portable_value(item)
            for key, item in value.items()
            if str(key) not in {
                "source_task_id", "source_trace_id", "action", "actions",
                "action_list", "concrete_bindings", "source_span",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_portable_value(item) for item in value]
    if isinstance(value, str) and re.search(r"(?:_|\s)\d+\b", value):
        return "<entity>"
    return value


class ProvisionalAtomicRetriever:
    def __init__(self, store: Any, *, top_k: int = 3) -> None:
        self.store = store
        self.top_k = int(top_k)

    def retrieve(
        self,
        missing_instances: list[RequirementInstance],
        *,
        harness_profile: str,
        top_k: int | None = None,
    ) -> dict[str, list[ProvisionalAtomicCandidate]]:
        allowed = {"trial_ready", "trial_supported"}
        records = self.store.list_provisionals(statuses=allowed)
        result: dict[str, list[ProvisionalAtomicCandidate]] = {}
        limit = self.top_k if top_k is None else int(top_k)
        for instance in missing_instances:
            ranked: list[tuple[float, str, Any]] = []
            for record in records:
                status = str(getattr(getattr(record, "status", ""), "value", getattr(record, "status", "")))
                metadata = dict(getattr(record, "metadata", {}) or {})
                if status not in allowed:
                    continue
                if getattr(record, "harness_profile", "") != harness_profile:
                    continue
                contract = dict(getattr(record, "atomic_contract", {}) or {})
                if not provisional_contract_compatible(instance.requirement, contract):
                    continue
                if metadata.get("portable_contract_valid", True) is not True:
                    continue
                score = (
                    2.0 if status == "trial_supported" else 1.0
                ) + 0.1 * int(metadata.get("independent_source_replay_support", 0)) + 0.1 * int(
                    metadata.get("independent_local_trial_success", 0)
                )
                ranked.append((score, str(record.provisional_ref), record))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            result[instance.instance_id] = [
                ProvisionalAtomicCandidate(
                    provisional_ref=str(record.provisional_ref),
                    status=str(getattr(getattr(record, "status", ""), "value", getattr(record, "status", ""))),
                    score=score,
                    canonical_intent=str(record.canonical_intent),
                    atomic_contract=dict(record.atomic_contract),
                    reasons=("unit_contract_compatible", "failure_side_isolated"),
                )
                for score, _, record in ranked[:limit]
            ]
        return result


class FailureExperienceRetriever:
    def __init__(self, store: Any, *, top_k: int = 2) -> None:
        self.store = store
        self.top_k = int(top_k)

    def retrieve(
        self,
        task_contract: TaskContract,
        requirement_expansion: RequirementExpansion,
        *,
        harness_profile: str,
        top_k: int | None = None,
    ) -> list[FailureExperienceView]:
        signature = task_cluster_signature(
            task_contract, harness_profile, requirement_expansion,
        )
        active = {"observed", "confirmed"}
        current_shape_ids = {
            requirement_instance_shape_id(
                item,
                requirement_expansion,
                task_contract,
            )
            for item in requirement_expansion.instances
        }
        ranked: list[tuple[int, int, int, str, Any]] = []
        for record in self.store.list_failure_experiences(statuses=active):
            status = str(getattr(getattr(record, "status", ""), "value", getattr(record, "status", "")))
            if record.cluster_signature != signature or record.harness_profile != harness_profile:
                continue
            support = len(set(getattr(record, "support_trace_ids", ())))
            metadata = dict(getattr(record, "metadata", {}) or {})
            raw_remaining_shapes = metadata.get(
                "remaining_requirement_shape_ids", (),
            )
            record_remaining_shape_ids = (
                set(map(str, raw_remaining_shapes))
                if (
                    metadata.get("semantic_shape_version") == 1
                    and isinstance(raw_remaining_shapes, (list, tuple))
                )
                else set()
            )
            overlap = len(current_shape_ids & record_remaining_shape_ids)
            ranked.append((1 if status == "confirmed" else 0, support, overlap, str(record.experience_id), record))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        limit = self.top_k if top_k is None else int(top_k)
        return [
            FailureExperienceView(
                experience_id=str(record.experience_id),
                status=str(getattr(getattr(record, "status", ""), "value", getattr(record, "status", ""))),
                requirement_shape_ids=tuple(map(
                    str,
                    dict(record.metadata or {}).get(
                        "requirement_shape_ids", (),
                    ),
                )),
                validated_prefix_shape_ids=tuple(map(
                    str,
                    dict(record.metadata or {}).get(
                        "validated_prefix_shape_ids", (),
                    ),
                )),
                first_unrecovered_divergence=dict(_portable_value(record.first_unrecovered_divergence)),
                remaining_requirement_shape_ids=tuple(map(
                    str,
                    dict(record.metadata or {}).get(
                        "remaining_requirement_shape_ids", (),
                    ),
                )),
                negative_suffix_summary=dict(_portable_value(record.negative_suffix_summary)),
                avoid_pattern_codes=tuple(record.avoid_pattern_codes),
                support_count=len(set(record.support_trace_ids)),
            )
            for *_, record in ranked[:limit]
        ]


__all__ = [
    "FailureExperienceRetriever",
    "FailureExperienceView",
    "ProvisionalAtomicCandidate",
    "ProvisionalAtomicRetriever",
    "provisional_contract_compatible",
    "task_cluster_signature",
]
