"""Compact policy-facing prompts for Planner/Runtime/Extractor sessions.

This module accepts only the fields the design permits an Agent to observe.  It
does not accept a validator snapshot, hidden benchmark state, a Tool body, or a
whole persistent graph, keeping those channels separated by construction.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any, Iterable, Mapping

from ..core.serialization import to_primitive


_ATOMIC_RUNTIME_FIELDS = ("summary", "inputs", "outputs", "preconditions", "effects")
_ATOMIC_SEEDED_FIELDS = (*_ATOMIC_RUNTIME_FIELDS, "guideline")
_INVOCATION_FIELDS = ("name", "description", "input_schema", "implementation_ref", "atomic_ref")
_ACTION_HISTORY_FIELDS = (
    "action_id",
    "action_type",
    "accepted",
    "observation",
    "done",
    "won",
    "revision",
    "new_revision",
)
_FORBIDDEN_POLICY_KEYS = {
    "validator_only",
    "validator_snapshot",
    "hidden_state",
    "hidden_pddl_state",
    "oracle_answer",
    "benchmark_answer",
    "tool_body",
    "source_code",
}


class ContextBuilder:
    """Build deterministic, compact user inputs for v3 Agent sessions."""

    def runtime_node(
        self,
        *,
        task_goal: str,
        atomic_contract: Any,
        certified_bindings: Mapping[str, Any] | None = None,
        missing_required_arguments: Iterable[str] | None = None,
        semantic_anchors: Mapping[str, Any] | None = None,
        execution_ready_bindings: Mapping[str, Any] | None = None,
        missing_or_insufficient_bindings: Iterable[str] | None = None,
        observation: str,
        action_catalog: Iterable[Any],
        relevant_action_history: Iterable[Any],
        remaining_budget: Mapping[str, Any],
        implementation_invocations: Iterable[Any],
    ) -> str:
        invocations = [
            _project(value, _INVOCATION_FIELDS) for value in implementation_invocations
        ]
        if len(invocations) > 3:
            raise ValueError("RuntimePreparationSession may expose at most 3 implementations")
        ready = (
            dict(execution_ready_bindings)
            if execution_ready_bindings is not None
            else dict(certified_bindings or {})
        )
        missing = (
            list(missing_or_insufficient_bindings)
            if missing_or_insufficient_bindings is not None
            else list(missing_required_arguments or ())
        )
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "current_atomic_contract": _project(atomic_contract, _ATOMIC_RUNTIME_FIELDS),
            "semantic_anchors": _policy_value(dict(semantic_anchors or {})),
            "execution_ready_bindings": _policy_value(ready),
            "missing_or_insufficient_bindings": [str(value) for value in missing],
            "current_observation": _text(observation, "observation"),
            "current_action_catalog": _compact_catalog(action_catalog),
            "relevant_action_history": _compact_history(relevant_action_history),
            "remaining_budget": _policy_value(dict(remaining_budget)),
            "allowed_implementation_invocations": invocations,
        }
        return _render(
            "Prepare and execute only the current Atomic occurrence. Use only native tools; "
            "never encode an action in prose.",
            payload,
        )

    def seeded_node(
        self,
        *,
        task_goal: str,
        atomic_contract: Any,
        certified_bindings: Mapping[str, Any] | None = None,
        semantic_anchors: Mapping[str, Any] | None = None,
        execution_ready_bindings: Mapping[str, Any] | None = None,
        missing_or_insufficient_bindings: Iterable[str] = (),
        observation: str,
        action_catalog: Iterable[Any],
        relevant_action_history: Iterable[Any],
        remaining_budget: Mapping[str, Any],
    ) -> str:
        ready = (
            dict(execution_ready_bindings)
            if execution_ready_bindings is not None
            else dict(certified_bindings or {})
        )
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "current_atomic_contract_and_guideline": _project(
                atomic_contract, _ATOMIC_SEEDED_FIELDS
            ),
            "semantic_anchors": _policy_value(dict(semantic_anchors or {})),
            "execution_ready_bindings": _policy_value(ready),
            "missing_or_insufficient_bindings": [
                str(value) for value in missing_or_insufficient_bindings
            ],
            "current_observation": _text(observation, "observation"),
            "current_action_catalog": _compact_catalog(action_catalog),
            "relevant_real_action_history": _compact_history(relevant_action_history),
            "remaining_budget": _policy_value(dict(remaining_budget)),
        }
        return _render(
            "Solve only the current Atomic occurrence with environment_action. This is a fresh "
            "Seeded session and contains no failed Tool body or failed Implementation mapping.",
            payload,
        )

    def dynamic_task(
        self,
        *,
        task_goal: str,
        observation: str,
        action_catalog: Iterable[Any],
        relevant_action_history: Iterable[Any],
        remaining_budget: Mapping[str, Any],
    ) -> str:
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "current_observation": _text(observation, "observation"),
            "current_action_catalog": _compact_catalog(action_catalog),
            "relevant_action_history": _compact_history(relevant_action_history),
            "remaining_budget": _policy_value(dict(remaining_budget)),
        }
        return _render(
            "Solve the task through native environment_action calls. The orchestrator, not prose, "
            "determines completion.",
            payload,
        )

    def planner_requirements(
        self,
        *,
        task_goal: str,
        task_contract: Any,
        semantic_hints: Iterable[Any] = (),
    ) -> str:
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "task_contract": _policy_value(task_contract),
            "semantic_hints": _policy_value(list(semantic_hints)),
        }
        return _render(
            "Submit CapabilityRequirements with the offered native submit tool. Do not claim formal completeness "
            "beyond the supplied TaskContract authority.",
            payload,
        )

    def planner_workflow(
        self,
        *,
        task_goal: str,
        task_contract: Any,
        requirements: Iterable[Any],
        atomic_search_results: Iterable[Any],
        existing_edge_evidence: Iterable[Any] = (),
    ) -> str:
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "task_contract": _policy_value(task_contract),
            "requirements": _policy_value(list(requirements)),
            "atomic_search_results": _policy_value(list(atomic_search_results)),
            "existing_edge_evidence": _policy_value(list(existing_edge_evidence)),
        }
        return _render(
            "Propose one strictly linear control sequence and forward data/dependency edges as "
            "a native submit tool call. Code will validate the proposal.",
            payload,
        )

    def extractor_e1(self, *, canonical_trace: Any) -> str:
        return _render(
            "Propose reusable Atomic occurrences from this canonical successful trace with the offered "
            "native submit tool. Follow this exact E1 authority contract: event_start is inclusive and "
            "event_end is EXCLUSIVE, so one event i is [i,i+1). Prefer the smallest non-overlapping "
            "causal slice, normally one accepted state-changing event. Select only the minimal causal "
            "chain that establishes the TaskContract; do not turn search/exploration detours into skills. "
            "A navigation/open event belongs only when it establishes a concrete precondition for a later "
            "selected causal event. Omit LOOK, INVENTORY, and every "
            "event for which both authoritative_positive_effects and "
            "authoritative_terminal_effect_certificates are empty. input_roles must be non-empty, have "
            "unique concrete values, and copy those values exactly from input_role_candidates or the "
            "arguments of authoritative state/effect facts in the selected event context; never invent "
            "agent/player/search/inventory bindings. output_roles must be "
            "non-empty and each output value must exactly repeat one input value; publish the affected "
            "object, container, light, or reached location under a reusable output role so later steps "
            "can consume it. Preconditions may be empty and otherwise must copy exact predicates and "
            "concrete args from authoritative_before_state_facts at event_start. Effects must copy exact "
            "predicates and concrete args from authoritative_positive_effects of the selected slice, or "
            "use the exact predicate of an explicitly listed terminal certificate and choose each effect "
            "argument only from that role's concrete_binding_candidates; every chosen candidate must also "
            "be present in input_roles. "
            "Do not use observation prose, aliases such as agent.holding/object.in_inventory/player.at, "
            "bare role names, placeholders, or any unlisted fact as evidence. Code validates each proposed "
            "occurrence independently, rejects invalid proposals, and passes only the validated subset to E2.",
            {"canonical_trace": _policy_value(canonical_trace)},
        )

    def extractor_e2(self, *, canonical_occurrences: Iterable[Any]) -> str:
        prefix = (
            "The following canonical occurrences were validated by code and are authoritative. "
            "Discard or correct any conflicting memory from the previous turn."
        )
        return _render(
            prefix
            + " Propose the canonical control sequence and edge references with the native submit tool. "
            "Use every authoritative occurrence exactly once in the supplied chronological order. "
            "Copy existing_edges only from known_edge_evidence. Add a new data_flow edge whenever an "
            "earlier output binding value is reused by a later required input, using the exact occurrence "
            "IDs and role names. Each required target role has at most one authoritative producer; when "
            "multiple earlier outputs carry the same binding identity, select only the nearest preceding "
            "producer in control_sequence and do not add duplicate producers. The only dependency wire "
            "value is edge_type=requires_skill. Do not "
            "invent requires_skill edges merely to represent control order; the control_sequence already "
            "carries order and occurrences need not be edge-connected.",
            {"canonical_occurrences": _policy_value(list(canonical_occurrences))},
        )


def _render(instruction: str, payload: dict[str, Any]) -> str:
    return instruction + "\n\nPOLICY_CONTEXT_JSON\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    primitive = to_primitive(value) if is_dataclass(value) else value
    if not isinstance(primitive, Mapping):
        raise TypeError("context object must be a mapping or dataclass")
    return {str(key): item for key, item in primitive.items()}


def _project(value: Any, fields: Iterable[str]) -> dict[str, Any]:
    mapping = _as_mapping(value)
    return {
        name: _policy_value(mapping[name])
        for name in fields
        if name in mapping
    }


def _policy_value(value: Any) -> Any:
    primitive = to_primitive(value)
    _reject_forbidden_keys(primitive)
    try:
        json.dumps(primitive, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy context must be JSON serializable") from exc
    return primitive


def _reject_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_POLICY_KEYS:
                raise ValueError(f"validator-only or executable field is forbidden in policy context: {path}.{key}")
            _reject_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, path=f"{path}[{index}]")


def _compact_catalog(values: Iterable[Any]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        mapping = _as_mapping(value)
        action_id = str(mapping.get("action_id", ""))
        if not action_id:
            raise ValueError("action catalog entry requires action_id")
        if action_id in seen:
            continue
        seen.add(action_id)
        compact = {
            "action_id": action_id,
            "action_type": str(mapping.get("action_type", "")),
            "display_text": str(mapping.get("display_text", "")),
            "revision": mapping.get("revision"),
        }
        catalog.append(_policy_value(compact))
    return catalog


def _compact_history(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [_project(value, _ACTION_HISTORY_FIELDS) for value in values]


__all__ = ["ContextBuilder"]
