"""State-scoped deterministic rejection evidence, not a new policy oracle."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..core.refs import content_hash
from ..core.serialization import to_primitive


@dataclass(frozen=True)
class RejectedRuntimeCandidate:
    occurrence_id: str
    role: str
    value: Any
    failure_code: str
    revision: int
    scope: str
    evidence_refs: tuple[str, ...] = ()


def state_signature(ctx, occurrence):
    # Revision alone is insufficient after rollback/replay. Include the
    # authoritative world, current role bindings, and committed Repeat state.
    return content_hash(to_primitive({
        "revision": ctx.world_revision,
        "atomic_ref": str(occurrence.node_ref),
        "step_id": occurrence.step_id,
        "world": ctx.harness.validator_channel().snapshot(),
        "bindings": ctx.binding_store.snapshot_for_node(occurrence),
        "semantic_anchors": sorted(ctx.binding_store._semantic_anchors.items()),
        "grounding_evidence": ctx.evidence_store._evidence,
        "repeat": ctx.binding_store.repeat_state,
    }))


def query_key(occurrence, call):
    return content_hash({"occurrence": occurrence.occurrence_id,
                         "tool": call.name, "arguments": call.arguments})


def cached_rejection(ctx, occurrence, call):
    signature = state_signature(ctx, occurrence)
    entry = ctx.rejected_runtime_candidates.get(query_key(occurrence, call) + ":" + signature)
    if entry:
        return {**copy.deepcopy(entry["payload"]), "deterministic_rejection_cache_hit": True}
    return None


def remember_rejection(ctx, occurrence, call, payload):
    if not isinstance(payload, dict) or payload.get("deterministic_rejection_cache_hit"):
        return
    # Only validator/preflight results, never a provider error, tool execution
    # failure, speculative environment observation, or model refusal.
    validation = payload.get("validation", {}) if call.name == "validate_current_atomic" else payload
    typed_preflight = call.name.startswith("invoke_impl_") and payload.get("failure_layer") in {
        "runtime_binding", "runtime_agent",
    } and not payload.get("started", False)
    if not (call.name == "validate_current_atomic" or typed_preflight):
        return
    if not isinstance(validation, dict) or validation.get("passed") is not False or not validation.get("failure_code"):
        return
    claims = call.arguments.get("candidate_bindings", {}) if call.name == "validate_current_atomic" else call.arguments
    key = query_key(occurrence, call)
    refs = tuple(validation.get("witness_refs", ()))
    candidates = [RejectedRuntimeCandidate(
        occurrence.occurrence_id, role, copy.deepcopy(value), validation["failure_code"],
        ctx.world_revision, "exact_call_and_authoritative_state", refs,
    ) for role, value in claims.items()]
    entry = {"query_key": key, "state_signature": state_signature(ctx, occurrence),
             "tool": call.name, "candidate_group": to_primitive(candidates),
             "payload": copy.deepcopy(payload)}
    ctx.rejected_runtime_candidates[key + ":" + entry["state_signature"]] = entry
    ctx.trace_builder.trace.metadata.setdefault("runtime_candidate_rejections", []).append({
        k: copy.deepcopy(v) for k, v in entry.items() if k != "payload"})


def current_rejections(ctx, occurrence):
    signature = state_signature(ctx, occurrence)
    return [{"tool": value["tool"], "scope": "exact_call_and_authoritative_state",
             "candidate_group": copy.deepcopy(value["candidate_group"])}
            for value in ctx.rejected_runtime_candidates.values()
            if value["state_signature"] == signature
            and any(item["occurrence_id"] == occurrence.occurrence_id for item in value["candidate_group"])]
