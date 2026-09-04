"""Build the state-transition-authoritative extractor view of a TraceRecord."""

from __future__ import annotations

import re
from typing import Any

from ..core.serialization import to_primitive
from .atomicizer import reduce_action_state


def _fact_identity(fact: dict[str, Any]) -> tuple[str, str]:
    return (
        str(fact.get("predicate", "")),
        repr(sorted(dict(fact.get("args") or {}).items())),
    )


def _state_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before_ids = {_fact_identity(item) for item in before}
    after_ids = {_fact_identity(item) for item in after}
    return (
        [item for item in after if _fact_identity(item) not in before_ids],
        [item for item in before if _fact_identity(item) not in after_ids],
    )


def _entity_family(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(
        r"[^a-z0-9]", "",
        re.sub(r"(?:_|\s)\d+$", "", value.casefold()),
    )


def _terminal_certificate_projection(
    certificates: list[dict[str, Any]],
    *,
    action: dict[str, Any],
    before_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach exact, action/state-derived candidates to narrow certificates."""

    available = [
        value
        for value in dict(action.get("arguments") or {}).values()
        if isinstance(value, str) and value
    ]
    available.extend(
        value
        for fact in before_facts
        for value in dict(fact.get("args") or {}).values()
        if isinstance(value, str) and value
    )
    projected: list[dict[str, Any]] = []
    for certificate in certificates:
        candidates: dict[str, list[str]] = {}
        for role, family_value in dict(certificate.get("args") or {}).items():
            family = _entity_family(family_value)
            matching = sorted({
                value for value in available
                if family and _entity_family(value) == family
            })
            if matching:
                candidates[str(role)] = matching
        projected.append({
            **certificate,
            "concrete_binding_candidates": candidates,
        })
    return projected


class TraceNormalizer:
    def build(self, trace: Any) -> dict[str, Any]:
        task_contract = to_primitive(trace.task_contract)
        actions = []
        for index, record in enumerate(trace.environment_actions):
            value = to_primitive(record)
            actions.append({
                "event_index": index, "action_id": value["action_id"],
                "action_type": value["action_type"], "arguments": value["arguments"],
                "accepted": value["accepted"],
                "before_revision": value["revision"], "after_revision": value["new_revision"],
                "done": value["done"], "won": value["won"], "span_id": value["span_id"],
            })
        for index, action in enumerate(actions):
            before = reduce_action_state(actions[:index])
            after = reduce_action_state(actions[:index + 1])
            positive, negative = _state_delta(before, after)
            action.update({
                # The E1 transport uses the normal Python half-open interval.
                # AtomicOccurrenceProposal stores the converted inclusive end.
                "extractor_event_start": index,
                "extractor_event_end_exclusive": index + 1,
                "input_role_candidates": dict(action.get("arguments") or {}),
                "authoritative_before_state_facts": before,
                "authoritative_positive_effects": positive,
                "authoritative_negative_effects": negative,
                # Reserved for adapter-provided, state-derived certificates.
                # Official ``won`` is never used to synthesize a semantic fact.
                "authoritative_terminal_effect_certificates": [],
            })
        spans = [to_primitive(item) for item in trace.runtime_spans if item.learnable]
        validations = [to_primitive(item) for item in trace.validations]
        input_authorities: list[dict[str, Any]] = []
        seen_input_authorities: set[tuple[str, str, str]] = set()
        for action in actions:
            if action.get("accepted") is not True:
                continue
            event_id = str(
                action.get("event_id", action.get("action_id", ""))
            )
            if not event_id:
                continue
            for raw_role, value in dict(
                action.get("arguments") or {}
            ).items():
                role = str(raw_role)
                authority_ref = f"action_arg:{event_id}:{role}"
                identity = (authority_ref, role, repr(value))
                if identity in seen_input_authorities:
                    continue
                seen_input_authorities.add(identity)
                input_authorities.append({
                    "authority_ref": authority_ref,
                    "event_id": event_id,
                    "argument_role": role,
                    "kind": "action_argument",
                    "source_kind": "action_argument",
                    "role": role,
                    "value": value,
                })
        return {
            "trace_id": trace.trace_id, "task_goal": trace.task.goal,
            "source_task": {
                "task_id": trace.task.task_id,
                "task_signature": trace.task.task_signature,
                "goal": trace.task.goal,
                "benchmark": trace.task.benchmark,
                "task_type": trace.task.task_type,
                "context": {
                    "env_index": trace.task.metadata.get("env_index"),
                    "game_file": trace.task.metadata.get("game_file", ""),
                },
                "metadata": dict(trace.task.metadata),
            },
            "task_contract": task_contract, "benchmark_success": trace.benchmark_success,
            "actions": actions, "runtime_spans": spans, "validations": validations,
            "node_records": [to_primitive(item) for item in trace.node_records],
            "implementation_invocations": [to_primitive(item) for item in trace.implementation_invocations],
            "tool_executions": [to_primitive(item) for item in trace.tool_executions],
            "boundary_authorities": {
                "inputs": input_authorities,
                "effects": [],
            },
        }
