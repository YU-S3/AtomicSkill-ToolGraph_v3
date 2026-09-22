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


def consumer_step_identity(occurrence):
    # Support UUIDs are execution audit identities, not distinct obligations.
    # Atomic contract, formal bindings, world and Repeat remain in the key.
    return "support" if occurrence.step_id.startswith("support::") else occurrence.step_id


def state_signature(ctx, occurrence):
    # Revision alone is insufficient after rollback/replay. Include the
    # authoritative world, current role bindings, and committed Repeat state.
    return content_hash(to_primitive({
        "revision": ctx.world_revision,
        "atomic_ref": str(occurrence.node_ref),
        "step_id": consumer_step_identity(occurrence),
        "world": _semantic_state(ctx.harness.validator_channel().snapshot()),
        "bindings": {k: _binding(v) for k, v in ctx.binding_store.snapshot_for_node(occurrence).items()},
        "semantic_anchors": {role: _binding(value) for (owner, role), value in ctx.binding_store._semantic_anchors.items()
                             if owner in {occurrence.occurrence_id, "task"}},
        "repeat": ctx.binding_store.repeat_state,
    }))


def _binding(value):
    return {key: to_primitive(getattr(value, key)) for key in ('value', 'semantic_type', 'source', 'status', 'resolution')}


def _semantic_state(value):
    value = to_primitive(value)
    if isinstance(value, dict):
        return {key: _semantic_state(item) for key, item in value.items()
                if key not in {'evidence_id', 'evidence_refs', 'witness_refs', 'event_id', 'timestamp', 'observed_at'}}
    if isinstance(value, list):
        return sorted((_semantic_state(item) for item in value), key=lambda item: content_hash(item))
    return value


def query_key(occurrence, call, ctx=None):
    arguments = call.arguments
    if call.name == 'environment_action' and ctx is not None:
        # First validate the current action identity. An invalid/stale id may
        # never become legal merely because an older call was cached.
        spec = next((item for item in ctx.action_catalog
                     if item.action_id == arguments['action_id'] and item.revision == ctx.world_revision), None)
        if spec is None:
            raise KeyError(arguments['action_id'])
        arguments = {'action_type': spec.action_type, 'arguments': spec.arguments, 'intent': arguments['intent'],
                     'candidate_bindings': arguments.get('candidate_bindings', {}),
                     'candidate_outputs': arguments.get('candidate_outputs', {})}
    return content_hash({"occurrence": occurrence.occurrence_id,
                         "tool": call.name, "arguments": arguments})


def cached_rejection(ctx, occurrence, call):
    signature = state_signature(ctx, occurrence)
    try:
        key = query_key(occurrence, call, ctx)
    except (KeyError, ValueError):
        return None  # Ordinary handler reports the original protocol error.
    entry = ctx.rejected_runtime_candidates.get(key + ":" + signature)
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
    support_rejection = call.name == 'invoke_support_atomic' and payload.get('accepted') is False
    env_rejection = call.name == 'environment_action' and payload.get('repeat_preflight_rejected') is True
    if not (call.name == "validate_current_atomic" or typed_preflight or support_rejection or env_rejection):
        return
    if support_rejection or env_rejection:
        validation = {**payload, 'passed': False, 'failure_code': payload.get('preflight_failure_code') or payload.get('failure_code') or payload.get('error')}
    if not isinstance(validation, dict) or validation.get("passed") is not False or not validation.get("failure_code"):
        return
    claims = call.arguments.get("candidate_bindings", {}) if call.name == "validate_current_atomic" else call.arguments
    key = query_key(occurrence, call, ctx)
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
    selected = {}
    for value in ctx.rejected_runtime_candidates.values():
        payload = value.get('payload', {})
        stable = payload.get('constraint_scope') == 'stable_schema'
        if not stable and not (value.get('state_signature') == signature and any(
                item['occurrence_id'] == occurrence.occurrence_id for item in value['candidate_group'])):
            continue
        # Stable constraints survive sessions/world revisions, but repeated
        # malformed values do not replicate their large original argument lists.
        key = content_hash({k: payload.get(k) for k in ('support_atomic_ref',
            'error_code', 'argument_path', 'expected_constraint', 'actual_summary')}) if stable else value['query_key']
        selected[key] = value
    return [{"tool": value["tool"], "scope": value.get('payload', {}).get('constraint_scope', 'exact_call_and_authoritative_state'),
             "constraint_feedback": {k: copy.deepcopy(value['payload'][k]) for k in
                 ('error_code', 'argument_path', 'expected_constraint', 'actual_summary', 'support_atomic_ref')
                 if k in value.get('payload', {})},
             "candidate_group": [] if value.get('payload', {}).get('constraint_scope') == 'stable_schema'
                 else copy.deepcopy(value["candidate_group"])}
            for value in selected.values()]
