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
from typing import Any, Iterable

from .protocol import NativeToolSpec, ONE_NATIVE_CALL_PREFIX


FORMAT = "runtime_downstream_dedup_v1"
TOP_FIELDS = {"current_step", "output_obligations", "remaining_method_outline"}
EDGE_FIELDS = (
    "producer_step",
    "producer_output_role",
    "edge_id",
    "consumer_step",
    "consumer_input_role",
)
RELATION_FIELDS = (
    "relation_predicate",
    "effect_domain",
    "relevant_anchor_roles",
    "public_relation_status",
    "public_evidence_refs",
)
SHARED_FIELDS = (
    "consumer_summary",
    "consumer_preconditions",
    "consumer_effects",
    "consumer_known_semantic_anchors",
)
LEGACY_OBLIGATION_FIELDS = (
    set(EDGE_FIELDS) | set(SHARED_FIELDS) | {"consumer_input_contract"}
)
OBLIGATION_FIELDS = LEGACY_OBLIGATION_FIELDS | set(RELATION_FIELDS)
CANDIDATE_FIELDS = {"producer_value_context"}


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
        if (
            not isinstance(obligation, dict)
            or frozenset(obligation) not in {
                frozenset(LEGACY_OBLIGATION_FIELDS),
                frozenset(OBLIGATION_FIELDS),
                frozenset(OBLIGATION_FIELDS | CANDIDATE_FIELDS),
            }
        ):
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
            field: copy.deepcopy(obligation[field])
            for field in (*EDGE_FIELDS, *RELATION_FIELDS, *sorted(CANDIDATE_FIELDS))
            if field in obligation
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


def _native_table(specs: Iterable[NativeToolSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        if not isinstance(spec, NativeToolSpec):
            raise ValueError("native specs must use the provider protocol adapter")
        # Compare the same representation the provider actually sends.
        rows.append(copy.deepcopy(spec.to_openai()["function"]))
    return rows


def restore_native_interfaces(projected, native_tool_specs, audit):
    """Audit-only inverse. Reject a changed table, payload or replacement proof."""
    if digest(projected) != audit["projected_payload_hash"]:
        raise ValueError("expression payload integrity mismatch")
    result = copy.deepcopy(projected)
    if audit["replacements"]:
        rows = _native_table(native_tool_specs)
        if digest(rows) != audit["native_specs_hash"]:
            raise ValueError("expression native table integrity mismatch")
        seen = set()
        for replacement in audit["replacements"]:
            index = replacement["index"]
            items = result["allowed_implementation_invocations"]
            if type(index) is not int or index < 0 or index >= len(items) or index in seen:
                raise ValueError("invalid expression replacement")
            seen.add(index)
            item = items[index]
            matching = [row for row in rows if row["name"] == replacement["native_name"]]
            if len(matching) != 1 or item.get("name") != replacement["native_name"]:
                raise ValueError("ambiguous expression replacement")
            if "description" in item or "input_schema" in item:
                raise ValueError("expression restoration would overwrite data")
            description = matching[0]["description"]
            rule = replacement["description_rule"]
            if rule == "one_native_call_prefix_v1" and description.startswith(ONE_NATIVE_CALL_PREFIX):
                description = description[len(ONE_NATIVE_CALL_PREFIX):]
            elif rule != "identity":
                raise ValueError("unknown expression description wrapper")
            item.update(description=description, input_schema=copy.deepcopy(matching[0]["parameters"]))
    if digest(result) != audit["source_payload_hash"]:
        raise ValueError("expression roundtrip mismatch")
    return result


def compact_native_interfaces(raw, native_tool_specs=None):
    """Pure, reversible substitution of definitions offered in THIS request.

    The caller supplies an already policy-safe view. No interface is created,
    filtered or made ready here; unknown fields and unsupported shapes survive.
    """
    out = copy.deepcopy(raw)
    specs = list(native_tool_specs) if native_tool_specs is not None else None
    audit = dict(projection_version="r103.expression.v1", source_payload_hash=digest(raw),
                 native_specs_hash=None, replacements=[], removed_paths=[],
                 original_fallback_reasons=[], required_surface_equal=False)
    rows = None
    if specs is not None:
        try:
            rows = _native_table(specs)
            audit["native_specs_hash"] = digest(rows)
        except (ValueError, TypeError):
            audit["original_fallback_reasons"].append({"reason": "native_table_unrecognized"})
    else:
        audit["original_fallback_reasons"].append({"reason": "native_table_absent"})
    items = out.get("allowed_implementation_invocations")
    if rows is not None and isinstance(items, list):
        for index, item in enumerate(items):
            reason = "definition_fields_absent_or_unrecognized"
            if isinstance(item, dict) and isinstance(item.get("description"), str) and isinstance(item.get("input_schema"), dict):
                matching = [row for row in rows if row["name"] == item.get("name")]
                reason = "native_not_unique_in_actual_request"
                if len(matching) == 1:
                    native = matching[0]
                    rule = ("identity" if native["description"] == item["description"] else
                            "one_native_call_prefix_v1" if native["description"] == ONE_NATIVE_CALL_PREFIX + item["description"] else None)
                    reason = "description_differs" if rule is None else "schema_differs"
                    if rule is not None and canonical_bytes(native["parameters"]) == canonical_bytes(item["input_schema"]):
                        del item["description"], item["input_schema"]
                        audit["replacements"].append(dict(index=index, native_name=item["name"], description_rule=rule))
                        audit["removed_paths"].extend(f"allowed_implementation_invocations[{index}].{key}" for key in ("description", "input_schema"))
                        reason = ""
            if reason:
                audit["original_fallback_reasons"].append(dict(index=index, reason=reason))
    guidance = raw.get("current_state_snapshot", {}).get("current_atomic", {}).get("skill_guidance")
    audit.update(projected_payload_hash=digest(out), guidance_input_hash=digest(guidance),
                 guidance_output_hash=digest(out.get("current_state_snapshot", {}).get("current_atomic", {}).get("skill_guidance")),
                 before_utf8_bytes=len(canonical_bytes(raw)), after_utf8_bytes=len(canonical_bytes(out)),
                 byte_count_is_not_token_count=True)
    restored = restore_native_interfaces(out, specs, audit)
    audit["required_surface_equal"] = canonical_bytes(restored) == canonical_bytes(raw)
    return out, audit


def project_support_presentation(raw, summaries=None):
    """Same original summaries and exact preview dedup for both profiles."""
    out = copy.deepcopy(raw)
    audit = {'version': 'r103.runtime-feedback.v2', 'summary_source_hashes': {},
             'missing_summary_refs': [], 'preview_sources': [], 'removed_exact_previews': 0}
    for path, rows in [('support_atomic_candidates', out.get('support_atomic_candidates', [])),
                      ('task_runtime_frame.capability_candidates', out.get('task_runtime_frame', {}).get('capability_candidates', []))]:
        for i, row in enumerate(rows):
            ref = row.get('atomic_ref')
            if ref in (summaries or {}):
                row['summary'] = summaries[ref]
                audit['summary_source_hashes'][ref] = digest(summaries[ref])
            else:
                audit['missing_summary_refs'].append(ref)
            if 'mapping_previews' in row:
                audit['preview_sources'].append({'path': path, 'index': i,
                    'mapping_previews': copy.deepcopy(row['mapping_previews'])})
                seen, unique = set(), []
                for item in row['mapping_previews']:
                    key = canonical_bytes(item)
                    if key not in seen:
                        seen.add(key)
                        unique.append(item)
                audit['removed_exact_previews'] += len(row['mapping_previews']) - len(unique)
                row['mapping_previews'] = unique
    audit.update(before_bytes=len(canonical_bytes(raw)), after_bytes=len(canonical_bytes(out)))
    return out, audit


def project_runtime_payload(
    raw: dict[str, Any], *, native_tool_specs: Iterable[NativeToolSpec] | None = None,
    expression_enabled: bool = True,
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
    out, audit["expression_audit"] = compact_native_interfaces(out, native_tool_specs if expression_enabled else None)
    audit["runtime_presentation"] = "new" if expression_enabled else "old"
    audit["after_payload_sha256"] = digest(out)
    audit["after_payload_utf8_bytes"] = len(canonical_bytes(out))
    return out, audit


__all__ = [
    "FORMAT",
    "canonical_bytes",
    "digest",
    "pack_downstream_context",
    "project_runtime_payload",
    "unpack_downstream_context",
]
