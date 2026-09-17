"""Public, task-local Runtime automation interface and input resolution.

This module projects only Harness schemas and bindings already authorized for
the current occurrence.  It deliberately does not read validator snapshots,
Tool programs, persistent cases, or bindings owned by another occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.bindings import (
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
    resolution_satisfies,
)
from ..core.semantic_types import semantic_types_compatible
from ..core.serialization import to_primitive
from .ir import (
    ACTION_CATALOG_ENTRY_FIELDS,
    CONDITION_OPERATORS,
    CONDITION_SOURCES,
)


RUNTIME_AUTOMATION_INTERFACE_VERSION = "runtime_automation_interface_v1"
RUNTIME_INPUT_BINDING_KINDS = (
    "current_occurrence_anchor",
    "current_confirmed_binding",
    "current_candidate_binding",
    "data_flow",
    "constant",
)

_INPUT_BINDING_SOURCE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "kind": "current_occurrence_anchor",
        "fields": ["source_role"],
        "description": (
            "Read a semantic Task/DataFlow anchor of this occurrence. The "
            "anchor may guide search but is not automatically a concrete entity."
        ),
    },
    {
        "kind": "current_confirmed_binding",
        "fields": ["source_role"],
        "description": (
            "Read a currently grounded binding whose resolution satisfies the "
            "draft input contract."
        ),
    },
    {
        "kind": "current_candidate_binding",
        "fields": ["source_role"],
        "description": (
            "Read a current Agent candidate. A candidate is not a confirmed fact "
            "and must still satisfy the R0 resolution rules."
        ),
    },
    {
        "kind": "data_flow",
        "fields": ["source_role"],
        "description": (
            "Read a validated DataFlow value already available to this occurrence; "
            "values owned only by another occurrence are inaccessible."
        ),
    },
    {
        "kind": "constant",
        "fields": ["value"],
        "description": (
            "Use a portable semantic literal accepted by R0. Episode concrete "
            "entity identifiers are forbidden."
        ),
    },
)


_TOOL_IR_COLLECTION_SOURCE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "source": "action_catalog",
        "description": (
            "Enumerate values projected from the current public admissible "
            "primitive-action catalog at FOR_EACH entry. This exposes "
            "candidates only; it does not select an action or prove an effect."
            " A filtered/projected FOR_EACH with zero matches aborts the entire "
            "Tool with tool_ir_selector_no_match. Optional candidates need an "
            "IF condition.match guard before enumeration; false skips that "
            "lookup and permits subsequent work."
        ),
        "entry_fields": list(ACTION_CATALOG_ENTRY_FIELDS),
        "where": {
            "action_type": (
                "optional exact action_type from primitive_actions"
            ),
            "argument_role": (
                "required with semantic_compatible_with and names an exact "
                "argument role from that primitive action"
            ),
            "semantic_compatible_with": {
                "source": list(CONDITION_SOURCES),
                "field": (
                    "required field or declared role in the selected source"
                ),
                "semantic_type": "optional public semantic type",
            },
        },
        "direct_argument_filter_encoding": (
            "Put each optional exact portable primitive argument value directly "
            "at where.<argument_role>. Do not wrap argument filters in another "
            "object. Every role is checked against the selected action_type's "
            "public primitive signature."
        ),
        "projection": {
            "field": (
                "a top-level action_catalog entry field, used instead of project"
            ),
            "project": {
                "kind": ["field", "argument"],
                "field": "required when kind=field",
                "role": (
                    "required when kind=argument and names a primitive "
                    "argument role"
                ),
            },
        },
        "distinct": "optional boolean",
        "refresh_each_iteration": (
            "Optional boolean, FOR_EACH action_catalog only. False (default) "
            "uses the entry snapshot. True queries the current revision before "
            "each iteration and takes the first unseen projected value; vanished "
            "values are skipped and new values are visible. Empty after progress "
            "ends the loop; strict empty entry still rejects. Bounds are unchanged."
        ),
    },
)


RUNTIME_OUTPUT_DERIVATION_RULES = (
    "A required same-name input/output uses input_identity only when required, "
    "semantic type and minimum resolution agree; a semantic class cannot be "
    "upgraded to a concrete instance by renaming. Each fresh required output "
    "must have exactly one distinct (predicate, argument_role) derivation in "
    "the declared final Effects. Repeated references to the same pair count "
    "once. Final Effects describe the capability's promised outcome; Builder "
    "puts intermediate action effects in ACTION.expected_effects. Do not "
    "delete necessary final effects to pass R0 or arbitrarily select a witness. "
    "Future output witnesses need not exist at R0, but every input source "
    "must already be authorized. Ambiguous derivations are explicitly rejected. "
    "A unique predicate/argument derivation is not a unique concrete witness: "
    "final validation also requires one jointly consistent current witness assignment. "
    "Role names are neutral symbols; predicate argument names never imply a binding lookup. "
    "Use explicit $role references. Ordinary strings are literal values. A semantic target "
    "is not a concrete identity. To return a concrete member, declare a distinct concrete "
    "output role with output_semantic_constraints: {output_role: {compatible_with_input: input_role}}. "
    "The input must be required and type-compatible. R1 checks Harness compatibility, then "
    "compares against that input's actual VALUE, not its semantic_type label. "
    "Do not invent a generic category literal solely to constrain a relation-derived output: "
    "that literal must satisfy the same Harness family matcher. Only declare membership "
    "constraints required by the capability; other concrete outputs remain effect-derived. R1 "
    "uses declared concrete RETURN candidates only to filter current authoritative facts. "
    "Candidates and prose never create witnesses; missing, stale or incompatible results fail. "
    "Without an explicit effect reference or output semantic constraint, a descriptive input "
    "does not constrain the result. Invalid same-name upgrades are rejected, never auto-renamed. "
    "Declare only necessary final Effects supported by the public predicate validation_source. "
    "A policy projection is not automatically validator authority."
)


_TOOL_IR_CONDITION_CONTRACT: dict[str, Any] = {
    "capabilities": ["selector_condition_v1"],
    "match_shape": {
        "op": "exists or not_exists; defaults to exists",
        "match": {
            "source": "action_catalog",
            "where": "required public action_type; direct argument filters and optional semantic_compatible_with",
            "project": {"kind": "argument", "role": "an argument of the selected public primitive"},
            "distinct": "optional boolean",
        },
    },
    "match_rules": (
        "Use either match/op or the legacy shape, never both. Match compares "
        "current public action candidates; semantic comparison source/field "
        "must name a declared tool_input or in-scope local_variable and uses "
        "the Harness matcher. Empty matches mean false (not_exists true), not "
        "selector failure or a global absence fact. Querying creates no witness "
        "or binding. Invalid selectors/references are errors. No field paths, "
        "nested match, private facts, or new operators. Required RETURN and "
        "filtered FOR_EACH keep their strict no-result failure semantics."
    ),
    "shape": {
        "source": "one of sources",
        "field": "required source field or declared role",
        "op": "one of operators; defaults to exists",
        "value": "used only by equals, not_equals, and contains",
    },
    "sources": list(CONDITION_SOURCES),
    "operators": sorted(CONDITION_OPERATORS),
    "source_field_contracts": {
        "tool_input": "field is an exact declared Tool input role",
        "local_variable": "field is an exact definitely-in-scope local role",
        "action_catalog": (
            "field is action_id, revision, action_type, arguments, or one of "
            "length/count/size; reads the current public catalog"
        ),
        "semantic_evidence": (
            "field is a top-level public fact field such as predicate, args, or "
            "effect_domain, or one of length/count/size"
        ),
        "binding_evidence": (
            "field is a top-level public binding-evidence field, or one of "
            "length/count/size"
        ),
    },
    "semantics": {
        "exists": "true when the resolved value is non-empty",
        "not_exists": "true when the resolved value is empty",
        "equals": "resolved value equals value",
        "not_equals": "resolved value does not equal value",
        "contains": "value occurs in the resolved list/set/tuple or string",
        "empty": "resolved value is empty",
        "non_empty": "resolved value is non-empty",
    },
}


@dataclass
class RuntimeAutomationInputResolution:
    """One authoritative interpretation shared by R0 and the trial preflight."""

    values: dict[str, Any] = field(default_factory=dict)
    binding_updates: list[RuntimeBinding] = field(default_factory=list)
    input_authorities: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_codes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failure_codes


def public_predicate_schema(harness: Any) -> list[dict[str, Any]]:
    """Return the complete public predicate signature without hidden truth."""

    method = getattr(harness, "semantic_predicate_schema", None)
    if not callable(method):
        return []
    result: list[dict[str, Any]] = []
    for raw in method():
        item = to_primitive(raw)
        if not isinstance(item, Mapping):
            raise TypeError("semantic predicate schema entries must be mappings")
        normalized = {
            "predicate": str(item.get("predicate", "")),
            "effect_domain": str(item.get("effect_domain", "")),
            "validation_source": str(item.get("validation_source", "")),
            "argument_roles": [str(role) for role in item.get("argument_roles", ())],
            "argument_semantic_types": {
                str(role): str(value)
                for role, value in dict(
                    item.get("argument_semantic_types") or {}
                ).items()
            },
        }
        if (
            not normalized["predicate"]
            or normalized["effect_domain"] not in {"world", "evidence"}
            or not normalized["argument_roles"]
            or set(normalized["argument_roles"])
            != set(normalized["argument_semantic_types"])
        ):
            raise ValueError("semantic predicate schema entry is incomplete")
        result.append(normalized)
    return result


def public_primitive_action_schema(harness: Any) -> list[dict[str, Any]]:
    """Return the public primitive action signatures only."""

    method = getattr(harness, "primitive_action_schema", None)
    if not callable(method):
        return []
    result: list[dict[str, Any]] = []
    for raw in method():
        item = to_primitive(raw)
        if not isinstance(item, Mapping):
            raise TypeError("primitive action schema entries must be mappings")
        action_type = str(item.get("action_type", ""))
        argument_roles = [str(role) for role in item.get("argument_roles", ())]
        if not action_type:
            raise ValueError("primitive action schema entry lacks action_type")
        result.append({
            "action_type": action_type,
            "argument_roles": argument_roles,
        })
    return result


def public_tool_ir_collection_sources() -> list[dict[str, Any]]:
    """Return code-owned Tool IR selector contracts exposed to ToolBuilder."""

    return to_primitive(_TOOL_IR_COLLECTION_SOURCE_DEFINITIONS)


def public_tool_ir_condition_contract() -> dict[str, Any]:
    """Return the exact existing IF/STOP_WHEN read-only condition contract."""

    return to_primitive(_TOOL_IR_CONDITION_CONTRACT)


def _binding_projection(binding: RuntimeBinding, *, source_role: str) -> dict[str, Any]:
    return {
        "source_role": str(source_role),
        "value": to_primitive(binding.value),
        "semantic_type": str(binding.semantic_type),
        "status": str(binding.status.value),
        "resolution": str(binding.resolution.value),
        "source": str(binding.source.value),
        "world_revision": int(binding.world_revision),
        "evidence_refs": [str(ref) for ref in binding.evidence_refs],
    }


def _candidate_snapshot(binding_store: Any, occurrence: Any) -> dict[str, RuntimeBinding]:
    method = getattr(binding_store, "candidate_snapshot_for_node", None)
    if callable(method):
        return dict(method(occurrence))
    return {}


def _coerce_public_binding(
    value: Any,
    *,
    role: str,
    default_source: BindingSource,
) -> RuntimeBinding | None:
    """Normalize a code-owned binding view without accepting raw Agent data."""

    if isinstance(value, RuntimeBinding):
        return value
    concrete = getattr(value, "value", None)
    if concrete in (None, ""):
        return None
    try:
        return RuntimeBinding(
            role=str(getattr(value, "role", "") or role),
            value=concrete,
            semantic_type=str(
                getattr(value, "semantic_type", "") or "entity"
            ),
            source=getattr(value, "source", default_source),
            status=getattr(value, "status", BindingStatus.GROUNDED),
            resolution=getattr(
                value, "resolution", BindingResolution.SEMANTIC,
            ),
            evidence_refs=[
                str(ref) for ref in getattr(value, "evidence_refs", ())
            ],
            world_revision=int(getattr(value, "world_revision", 0)),
        )
    except (TypeError, ValueError):
        return None


def _current_source_bindings(
    binding_store: Any,
    occurrence: Any,
) -> tuple[
    dict[str, RuntimeBinding],
    dict[str, RuntimeBinding],
    dict[str, RuntimeBinding],
    dict[str, RuntimeBinding],
]:
    snapshot_method = getattr(binding_store, "snapshot_for_node", None)
    snapshot = (
        dict(snapshot_method(occurrence))
        if callable(snapshot_method)
        else {}
    )
    candidates = _candidate_snapshot(binding_store, occurrence)
    occurrence_roles = {
        str(role) for role in dict(getattr(occurrence, "binding_specs", {}) or {})
    }
    occurrence_roles.update(snapshot)
    anchors: dict[str, RuntimeBinding] = {}
    anchor_method = getattr(binding_store, "semantic_anchor_for", None)
    if callable(anchor_method):
        for role in sorted(occurrence_roles):
            anchor = _coerce_public_binding(
                anchor_method(occurrence, role),
                role=role,
                default_source=BindingSource.TASK,
            )
            if anchor is not None:
                anchors[role] = anchor
    confirmed = {
        role: binding for role, binding in snapshot.items()
        if isinstance(binding, RuntimeBinding)
        and binding.status is BindingStatus.GROUNDED
    }
    data_flow = {
        role: binding for role, binding in snapshot.items()
        if isinstance(binding, RuntimeBinding)
        and binding.status is BindingStatus.GROUNDED
        and binding.source is BindingSource.DATA_FLOW
    }
    output_method = getattr(binding_store, "validated_outputs", None)
    if callable(output_method):
        for role, binding in dict(
            output_method(str(occurrence.occurrence_id))
        ).items():
            if isinstance(binding, RuntimeBinding):
                data_flow.setdefault(str(role), binding)
    return anchors, confirmed, candidates, data_flow


def build_runtime_automation_interface(
    harness: Any,
    occurrence: Any,
    binding_store: Any,
) -> dict[str, Any]:
    """Project the complete public task-local self-tooling contract."""

    dynamic = build_runtime_automation_interface_update(
        occurrence, binding_store,
    )
    return {
        "schema_version": RUNTIME_AUTOMATION_INTERFACE_VERSION,
        "consumer_scope": getattr(occurrence, "consumer_scope", "node"),
        "parent_atomic_ref": "" if getattr(occurrence, "consumer_scope", "node") == "task" else str(occurrence.node_ref),
        "source_occurrence_id": dynamic["source_occurrence_id"],
        "primitive_actions": public_primitive_action_schema(harness),
        "predicate_vocabulary": public_predicate_schema(harness),
        "input_binding_kinds": [
            dict(item) for item in _INPUT_BINDING_SOURCE_DEFINITIONS
        ],
        "current_input_sources": dynamic["current_input_sources"],
        "fresh_output_rules": {
            "derivation_contract": RUNTIME_OUTPUT_DERIVATION_RULES,
            "future_effect_witness_allowed": True,
            "existing_output_witness_required_at_r0": False,
            "outputs_must_be_validated_after_trial": True,
        },
        "trial_scope": "task_local",
    }


def build_runtime_automation_interface_update(
    occurrence: Any,
    binding_store: Any,
) -> dict[str, Any]:
    """Project only the task-local interface fields that can change by turn."""

    anchors, confirmed, candidates, data_flow = _current_source_bindings(
        binding_store, occurrence,
    )
    return {
        "source_occurrence_id": str(occurrence.occurrence_id),
        "current_input_sources": {
            "current_occurrence_anchor": [
                _binding_projection(binding, source_role=role)
                for role, binding in sorted(anchors.items())
            ],
            "current_confirmed_binding": [
                _binding_projection(binding, source_role=role)
                for role, binding in sorted(confirmed.items())
            ],
            "current_candidate_binding": [
                _binding_projection(binding, source_role=role)
                for role, binding in sorted(candidates.items())
            ],
            "data_flow": [
                _binding_projection(binding, source_role=role)
                for role, binding in sorted(data_flow.items())
            ],
            "constant": {
                "portable_semantic_literal_only": True,
                "episode_concrete_identifiers_forbidden": True,
            },
        },
    }


def _is_episode_concrete_literal(value: Any) -> bool:
    # Import lazily because validator owns the portability rule and imports
    # this module for the runtime input resolver.
    from .validator import episode_literal_matches

    return bool(episode_literal_matches(value, annotation=False))


def resolve_runtime_automation_inputs(
    draft: Any,
    ctx: Any,
    occurrence: Any,
) -> RuntimeAutomationInputResolution:
    """Resolve draft inputs once for both R0 and runtime-trial validation."""

    result = RuntimeAutomationInputResolution()

    def fail(message: str) -> None:
        result.failure_codes.append("runtime_automation_input_binding_invalid")
        result.messages.append(message)

    specs = dict(getattr(draft, "input_binding_specs", None) or {})
    inputs = {str(item.name): item for item in list(getattr(draft, "inputs", ())) }
    required = {role for role, item in inputs.items() if bool(item.required)}
    spec_roles = {str(role) for role in specs}
    missing = sorted(required - spec_roles)
    unexpected = sorted(spec_roles - set(inputs))
    if missing:
        fail(f"required input_binding_specs missing roles {missing}")
    if unexpected:
        fail(f"input_binding_specs contain undeclared roles {unexpected}")

    binding_store = getattr(ctx, "binding_store", None)
    if binding_store is None:
        fail("task-local binding store is unavailable")
        return result
    try:
        anchors, confirmed, candidates, data_flow = _current_source_bindings(
            binding_store, occurrence,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        fail(f"task-local input sources are unavailable: {exc}")
        return result
    source_maps = {
        "current_occurrence_anchor": anchors,
        "current_confirmed_binding": confirmed,
        "current_candidate_binding": candidates,
        "data_flow": data_flow,
    }

    for role, raw in specs.items():
        role = str(role)
        if role not in inputs:
            continue
        if not isinstance(raw, Mapping):
            fail(f"input_binding_specs.{role} must be an object")
            continue
        spec = dict(raw)
        kind = str(spec.get("kind", "")).casefold()
        if kind not in RUNTIME_INPUT_BINDING_KINDS:
            fail(f"input_binding_specs.{role} has unsupported kind {kind}")
            continue
        input_spec = inputs[role]
        source_role = str(spec.get("source_role", ""))
        source_binding: RuntimeBinding | None = None
        if kind == "constant":
            if set(spec) - {"kind", "value"}:
                fail(f"{role}: constant has unsupported fields")
                continue
            value = spec.get("value")
            if value in (None, "") or _is_episode_concrete_literal(value):
                fail(f"{role}: invalid episode concrete constant")
                continue
            if str(input_spec.required_resolution) != "semantic":
                fail(
                    f"{role}: semantic literal cannot satisfy "
                    f"{input_spec.required_resolution} resolution"
                )
                continue
            resolved_binding = RuntimeBinding(
                role=role,
                value=value,
                semantic_type=str(input_spec.semantic_type),
                source=BindingSource.AGENT_PROPOSED,
                status=BindingStatus.GROUNDED,
                resolution=BindingResolution.SEMANTIC,
                evidence_refs=[],
                world_revision=int(getattr(ctx, "world_revision", 0)),
            )
        else:
            if set(spec) - {"kind", "source_role"}:
                fail(f"{role}: {kind} has unsupported fields")
                continue
            if not source_role:
                fail(f"{role}: {kind} requires source_role")
                continue
            source_binding = source_maps[kind].get(source_role)
            if source_binding is None and kind == "current_occurrence_anchor":
                anchor_method = getattr(binding_store, "semantic_anchor_for", None)
                if callable(anchor_method):
                    source_binding = _coerce_public_binding(
                        anchor_method(occurrence, source_role),
                        role=source_role,
                        default_source=BindingSource.TASK,
                    )
            if source_binding is None:
                fail(f"{role}: {kind}.{source_role} unavailable")
                continue
            required_type = str(input_spec.semantic_type or "entity")
            offered_type = str(source_binding.semantic_type or "")
            if not offered_type or not semantic_types_compatible(
                required_type, offered_type,
            ):
                fail(
                    f"{role}: semantic type {offered_type or '<unknown>'} "
                    f"is incompatible with {required_type}"
                )
                continue
            try:
                resolution_ok = resolution_satisfies(
                    source_binding.resolution,
                    input_spec.required_resolution,
                )
            except (KeyError, TypeError, ValueError):
                resolution_ok = False
            if not resolution_ok:
                fail(
                    f"{role}: resolution {source_binding.resolution.value} does "
                    f"not satisfy {input_spec.required_resolution}"
                )
                continue
            resolved_binding = RuntimeBinding(
                role=role,
                value=source_binding.value,
                semantic_type=source_binding.semantic_type,
                source=source_binding.source,
                status=source_binding.status,
                resolution=source_binding.resolution,
                evidence_refs=list(source_binding.evidence_refs),
                world_revision=int(source_binding.world_revision),
            )
        result.values[role] = resolved_binding.value
        result.binding_updates.append(resolved_binding)
        result.input_authorities[role] = {
            "kind": kind,
            "source_occurrence_id": str(occurrence.occurrence_id),
            "source_role": source_role,
            "value": to_primitive(resolved_binding.value),
            "semantic_type": str(resolved_binding.semantic_type),
            "status": str(resolved_binding.status.value),
            "resolution": str(resolved_binding.resolution.value),
            "source": str(resolved_binding.source.value),
            "world_revision": int(resolved_binding.world_revision),
            "available_revision": int(ctx.world_revision),
            "evidence_refs": [str(ref) for ref in resolved_binding.evidence_refs],
            "authority_ref": f"runtime_input:{draft.draft_id}:{role}",
        }
    return result


__all__ = [
    "RUNTIME_AUTOMATION_INTERFACE_VERSION",
    "RUNTIME_INPUT_BINDING_KINDS",
    "RuntimeAutomationInputResolution",
    "build_runtime_automation_interface",
    "build_runtime_automation_interface_update",
    "public_predicate_schema",
    "public_primitive_action_schema",
    "public_tool_ir_condition_contract",
    "public_tool_ir_collection_sources",
    "resolve_runtime_automation_inputs",
]
