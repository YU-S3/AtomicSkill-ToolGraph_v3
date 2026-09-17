"""Typed E1 verification over recorded public evidence, not executable IR."""
from __future__ import annotations

import copy
from typing import Any

from ..agents.protocol import validate_schema_instance
from ..core.bindings import resolution_satisfies
from ..core.semantic_types import semantic_types_compatible
from ..core.serialization import json_values_equal, json_value_key, to_primitive
from ..tooling.entry_contract import parameter_schema
from ..traces.canonical import canonical_trace_records


def public_value_authorities(trace: Any) -> list[dict]:
    """Project public bindings/catalog evidence, never hidden goal state."""
    result = []
    for index, raw in enumerate(canonical_trace_records(trace, "binding_changes") if hasattr(trace, "binding_changes") else []):
        change = to_primitive(raw)
        binding = dict(change.get("current") or {})
        if not binding or binding.get("status") != "grounded":
            continue
        result.append({
            "authority_ref": f"binding:{trace.trace_id}:{index}",
            "kind": "public_binding", "source_kind": "public_binding",
            "role": binding["role"], "value": binding["value"],
            "semantic_type": binding["semantic_type"], "resolution": binding["resolution"],
            "available_revision": int(change["revision"]),
            "source_occurrence_id": change.get("occurrence_id", ""),
            "trace_id": trace.trace_id,
        })
    for raw in (canonical_trace_records(trace, "grounding_evidence_changes") if hasattr(trace, "grounding_evidence_changes") else []):
        change = to_primitive(raw)
        evidence = dict(change.get("payload") or {})
        payload = dict(evidence.get("payload") or {})
        evidence_type = evidence.get("evidence_type")
        if change.get("operation") != "add" or evidence_type not in {"entity_concrete", "validated_tool_output", "task_binding"}:
            continue
        if evidence_type == "validated_tool_output" and not all(k in payload for k in ('semantic_type', 'resolution', 'validation_refs', 'source_occurrence_id')):
            continue
        # entity_concrete is the existing explicit identity certificate, not
        # a type inferred from an argument name or instance suffix.
        result.append({
            "authority_ref": str(evidence["evidence_id"]),
            "kind": "public_catalog" if evidence_type == 'entity_concrete' else 'public_binding',
            "source_kind": evidence_type,
            "role": str(payload["role"]), "value": payload["value"],
            "semantic_type": str(payload.get("semantic_type", "entity")),
            "resolution": 'concrete' if evidence_type == 'entity_concrete' else str(payload.get('resolution', 'semantic')),
            "available_revision": int(evidence["observed_at_revision"]),
            "trace_id": trace.trace_id, "source_occurrence_id": str(payload.get('source_occurrence_id', '')),
        })
    # Repeated catalog exposure does not need another copy in the E1 prompt.
    # Relation authority is revision-scoped, so never deduplicate its revisions.
    unique = {}
    for item in sorted(result, key=lambda a: a['available_revision']):
        key = (item['kind'], item['source_occurrence_id'], item['role'], item['semantic_type'],
               item['resolution'], json_value_key(item['value']),
               item['available_revision'] if item['resolution'] == 'relation_verified' else None)
        unique.setdefault(key, item)
    return list(unique.values())


def _available(authority: dict, revision: int) -> bool:
    known = authority.get("available_revision")
    return isinstance(known, int) and not isinstance(known, bool) and 0 <= known <= revision


def validate_input_specs(proposal, authorities: dict, entry: dict):
    inputs, outputs = copy.deepcopy(proposal.input_specs), copy.deepcopy(proposal.output_specs)
    for specs, values in ((inputs, proposal.input_roles), (outputs, proposal.output_roles)):
        if len({p.name for p in specs}) != len(specs) or {p.name for p in specs} != set(values):
            raise ValueError("typed boundary spec/sample roles must match exactly")
        validate_schema_instance(values, parameter_schema(specs))
    for spec in inputs:
        authority = authorities[spec.name]
        if not _available(authority, int(entry["before_revision"])):
            raise ValueError(f"input authority was not available at entry: {spec.name}")
        if not semantic_types_compatible(str(authority.get("semantic_type", "")), spec.semantic_type):
            raise ValueError(f"input authority type mismatch: {spec.name}")
        actual_resolution = str(authority.get("resolution", "semantic"))
        if actual_resolution == 'relation_verified' and authority['available_revision'] != int(entry['before_revision']):
            actual_resolution = 'concrete'
        if not resolution_satisfies(actual_resolution, spec.required_resolution):
            raise ValueError(f"input authority resolution mismatch: {spec.name}")
    in_specs, out_specs = {p.name: p for p in inputs}, {p.name: p for p in outputs}
    for role, constraint in proposal.output_semantic_constraints.items():
        if (role not in out_specs or not isinstance(constraint, dict)
                or set(constraint) != {"compatible_with_input"}
                or constraint["compatible_with_input"] not in in_specs):
            raise ValueError("invalid output semantic constraint roles")
        source = in_specs[constraint["compatible_with_input"]]
        if not semantic_types_compatible(source.semantic_type, out_specs[role].semantic_type):
            raise ValueError("output semantic constraint type mismatch")
    return inputs, outputs


def validate_local_authorities(proposal, authorities, normalized):
    refs = proposal.local_value_authority_refs
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate local authority reference")
    result = []
    for ref in refs:
        matches = [a for a in authorities if a.get("authority_ref") == ref]
        if len(matches) != 1:
            raise ValueError(f"local authority is not uniquely supplied: {ref}")
        authority = matches[0]
        if (authority.get("kind") not in {"public_binding", "public_catalog"}
                or authority.get("trace_id") != normalized["trace_id"]):
            raise ValueError(f"local authority has invalid public scope: {ref}")
        result.append(authority)
    return result


def validate_operand_authority(value, event, inputs, locals_, normalized):
    revision = int(event["before_revision"])
    if any(json_values_equal(value, a.get("value")) and _available(a, revision)
           for a in [*inputs.values(), *locals_]):
        return
    for literal in normalized.get("stable_operand_constants", []):
        if json_values_equal(value, literal):
            return
    raise ValueError(f"action operand lacks pre-use input/local authority: {event.get('event_id', '')}")
