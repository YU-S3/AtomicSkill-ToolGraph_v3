"""Pure, lossless R5 transformations of policy-safe Runtime payloads.

Only top-level Support diagnostics are removed from the Agent-facing view.
Downstream deduplication is used only when it round-trips exactly and reduces
the canonical UTF-8 representation. This module does not retrieve candidates,
validate facts, ground bindings, select actions, or mutate its input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


FORMAT = "runtime_downstream_dedup_v1"
TOP_FIELDS = {"current_step", "output_obligations", "remaining_method_outline"}
EDGE_FIELDS = (
    "producer_step",
    "producer_output_role",
    "edge_id",
    "consumer_step",
    "consumer_input_role",
)
SHARED_FIELDS = (
    "consumer_summary",
    "consumer_preconditions",
    "consumer_effects",
    "consumer_known_semantic_anchors",
)
OBLIGATION_FIELDS = (
    set(EDGE_FIELDS) | set(SHARED_FIELDS) | {"consumer_input_contract"}
)


def canonical_bytes(value: Any) -> bytes:
    """Return the deterministic JSON bytes used only for audit and comparison."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """Return the canonical policy-view SHA-256 digest."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def unpack_downstream_context(packed: dict[str, Any]) -> dict[str, Any]:
    """Expand an R5 downstream view for tests and audit, never execution."""

    if packed.get("context_format") != FORMAT:
        return copy.deepcopy(packed)
    table = packed["consumer_contracts"]
    obligations = []
    for edge in packed["output_obligations"]:
        consumer = table[edge["consumer_step"]]
        obligation = copy.deepcopy(edge)
        for field in SHARED_FIELDS:
            obligation[field] = copy.deepcopy(consumer[field])
        obligation["consumer_input_contract"] = copy.deepcopy(
            consumer["consumer_input_contracts"][edge["consumer_input_role"]]
        )
        obligations.append(obligation)
    return {
        "current_step": copy.deepcopy(packed["current_step"]),
        "output_obligations": obligations,
        "remaining_method_outline": copy.deepcopy(
            packed["remaining_method_outline"]
        ),
    }


def pack_downstream_context(raw: Any) -> tuple[Any, str]:
    """Deduplicate an exact supported shape, otherwise return it unchanged."""

    original = copy.deepcopy(raw)
    if not isinstance(raw, dict):
        return original, "original_unrecognized_shape"
    if raw.get("context_format") == FORMAT:
        return original, "already_packed"
    if set(raw) != TOP_FIELDS or not isinstance(raw["output_obligations"], list):
        return original, "original_unrecognized_shape"
    obligations = raw["output_obligations"]
    if len(obligations) < 2:
        return original, "original_no_repeated_consumer"

    table: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    for obligation in obligations:
        if not isinstance(obligation, dict) or set(obligation) != OBLIGATION_FIELDS:
            return original, "original_unrecognized_shape"
        if any(not isinstance(obligation[field], str) for field in EDGE_FIELDS):
            return original, "original_unrecognized_shape"
        step = obligation["consumer_step"]
        role = obligation["consumer_input_role"]
        contract = obligation["consumer_input_contract"]
        if not step or not role or not isinstance(contract, dict):
            return original, "original_unrecognized_shape"
        shared = {
            field: copy.deepcopy(obligation[field]) for field in SHARED_FIELDS
        }
        if step not in table:
            table[step] = {**shared, "consumer_input_contracts": {}}
        else:
            prior_shared = {field: table[step][field] for field in SHARED_FIELDS}
            if canonical_bytes(prior_shared) != canonical_bytes(shared):
                return original, "original_conflicting_consumer_view"
        contracts = table[step]["consumer_input_contracts"]
        if (
            role in contracts
            and canonical_bytes(contracts[role]) != canonical_bytes(contract)
        ):
            return original, "original_conflicting_input_contract"
        contracts[role] = copy.deepcopy(contract)
        edges.append({
            field: copy.deepcopy(obligation[field]) for field in EDGE_FIELDS
        })
        occurrences[step] = occurrences.get(step, 0) + 1
    if max(occurrences.values(), default=0) < 2:
        return original, "original_no_repeated_consumer"

    packed = {
        "context_format": FORMAT,
        "current_step": copy.deepcopy(raw["current_step"]),
        "consumer_contracts": table,
        "output_obligations": edges,
        "remaining_method_outline": copy.deepcopy(
            raw["remaining_method_outline"]
        ),
    }
    if canonical_bytes(unpack_downstream_context(packed)) != canonical_bytes(raw):
        return original, "original_roundtrip_mismatch"
    if len(canonical_bytes(packed)) >= len(canonical_bytes(raw)):
        return original, "original_no_size_benefit"
    return packed, "deduplicated"


def project_runtime_payload(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an Agent-facing copy and a separate Trace-only projection audit."""

    out = copy.deepcopy(raw)
    removed: list[str] = []
    support_raw = raw.get("support_atomic_candidates")
    if isinstance(support_raw, list):
        for index, item in enumerate(out["support_atomic_candidates"]):
            if isinstance(item, dict) and "diagnostics" in item:
                removed.append(
                    f"support_atomic_candidates[{index}].diagnostics"
                )
                del item["diagnostics"]

    reason = "not_present"
    raw_downstream = None
    state = out.get("current_state_snapshot")
    if isinstance(state, dict) and "downstream_obligations" in state:
        raw_downstream = copy.deepcopy(state["downstream_obligations"])
        state["downstream_obligations"], reason = pack_downstream_context(
            raw_downstream
        )

    audit = {
        "projection_version": "v3.2-r5",
        "removed_fields": removed,
        "downstream_projection": reason,
        "raw_support_candidates": copy.deepcopy(support_raw),
        "raw_downstream_obligations": raw_downstream,
        "before_payload_sha256": digest(raw),
        "after_payload_sha256": digest(out),
        "before_payload_utf8_bytes": len(canonical_bytes(raw)),
        "after_payload_utf8_bytes": len(canonical_bytes(out)),
        "byte_count_is_not_token_count": True,
    }
    return out, audit


__all__ = [
    "FORMAT",
    "canonical_bytes",
    "digest",
    "pack_downstream_context",
    "project_runtime_payload",
    "unpack_downstream_context",
]
