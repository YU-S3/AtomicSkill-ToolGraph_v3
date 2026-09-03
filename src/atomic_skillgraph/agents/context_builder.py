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
_INVOCATION_FIELDS = ("name", "description", "input_schema")
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
        task_semantic_context: Mapping[str, Any] | None = None,
        current_occurrence_semantic_anchors: Mapping[str, Any] | None = None,
        execution_ready_bindings: Mapping[str, Any] | None = None,
        missing_or_insufficient_bindings: Iterable[str] | None = None,
        observation: str,
        action_catalog: Iterable[Any],
        relevant_action_history: Iterable[Any],
        remaining_budget: Mapping[str, Any],
        implementation_invocations: Iterable[Any],
        downstream_plan_context: Mapping[str, Any] | None = None,
        current_state_snapshot: Mapping[str, Any] | None = None,
        exploration_memory: Mapping[str, Any] | None = None,
        recent_failed_learned_invocation: Mapping[str, Any] | None = None,
        support_atomic_candidates: Iterable[Any] = (),
        runtime_automation_drafts: Iterable[Any] = (),
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
        state = dict(current_state_snapshot or {
            "current_atomic": _project(
                atomic_contract, _ATOMIC_RUNTIME_FIELDS,
            ),
            "semantic_anchors": dict(
                current_occurrence_semantic_anchors or {}
            ),
            "confirmed_bindings": ready,
            "candidate_bindings": {},
            "missing_bindings": [str(value) for value in missing],
            "invalidated_bindings": {},
            "preconditions": [],
            "effect_witness_status": {},
            "learned_invocation_ready": False,
            "blocking_reasons": [],
            "downstream_obligations": dict(downstream_plan_context or {}),
            "remaining_budget": _compact_budget(remaining_budget),
        })
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "task_semantic_context": _policy_value(
                dict(task_semantic_context or {})
            ),
            "current_state_snapshot": _policy_value(state),
            "current_observation": _text(observation, "observation"),
            "current_action_catalog": _compact_catalog(action_catalog),
            "exploration_memory": _policy_value(dict(exploration_memory or {})),
            "recent_accepted_actions": _compact_history(
                relevant_action_history,
            ),
            "recent_failed_learned_invocation": _policy_value(
                dict(recent_failed_learned_invocation)
                if recent_failed_learned_invocation is not None
                else None
            ),
            "allowed_implementation_invocations": invocations,
            "support_atomic_candidates": [
                _policy_value(item) for item in support_atomic_candidates
            ],
            "runtime_automation_drafts": [
                _policy_value(item) for item in runtime_automation_drafts
            ],
        }
        return _render(
            "Prepare and execute only the current Atomic occurrence. Mark each "
            "environment_action as explore or attempt_current_atomic. An explore action "
            "collects public evidence and never commits the current Atomic merely because "
            "its effect happens to be true. Use validate_current_atomic when the current "
            "accepted-action-derived state already proves the Atomic and no new environment "
            "action is needed. current_state_snapshot.downstream_obligations describes how current outputs are "
            "consumed by the already-validated Runtime plan. It is semantic intent, never "
            "current evidence, and you decide which concrete current entities satisfy it. "
            "Use cannot_resolve only when this occurrence may still be valid but public "
            "evidence is insufficient or search is incomplete. current_state_snapshot is "
            "the code-authoritative current status; exploration_memory is historical and "
            "must not be treated as current truth. Use plan_conflict only "
            "when the formal occurrence, a hard semantic anchor, or a downstream obligation "
            "conflicts with public evidence and the same rigid graph cannot solve the task. "
            "Use give_up to terminate this route without asserting such a formal conflict. "
            "task_semantic_context describes the whole-task goal; only "
            "current_state_snapshot.semantic_anchors constrains "
            "learned invocation arguments. Stored Atomic summaries and learned-implementation "
            "descriptions are portable semantic guidance, never current bindings or evidence. "
            "Resolve missing unanchored arguments by "
            "instantiating that relational intent with task_semantic_context and current environment "
            "evidence; do not copy a same-named task field or the task's final destination merely "
            "because its name or type matches. Explore first when the required current relation is "
            "not yet evidenced. This prohibition applies only to roles absent from "
            "current_state_snapshot.semantic_anchors. When a role is explicitly anchored there, ground a "
            "compatible concrete current entity for that anchor, including when it is the task's final "
            "destination. Before executing a repetitive, mechanical, low-semantic-value action "
            "sequence, prefer propose_runtime_automation_atomic when the loop object, condition, "
            "and stop condition can be expressed by the current structured action/evidence "
            "interface. The proposal is an Atomic draft, never source code. "
            "For a learned invocation, call the exact native-tool name shown in "
            "allowed_implementation_invocations.name; never derive or extend a tool name from an "
            "artifact description or identifier, and copy canonical values exactly from the latest "
            "public catalog arguments: current_action_catalog.actions[].arguments initially, "
            "then action_catalog.actions[].arguments in environment tool results. Use only "
            "native tools; "
            "never encode an action in prose.",
            payload,
            sort_keys=False,
        )

    def seeded_node(
        self,
        *,
        task_goal: str,
        atomic_contract: Any,
        certified_bindings: Mapping[str, Any] | None = None,
        task_semantic_context: Mapping[str, Any] | None = None,
        current_occurrence_semantic_anchors: Mapping[str, Any] | None = None,
        execution_ready_bindings: Mapping[str, Any] | None = None,
        missing_or_insufficient_bindings: Iterable[str] = (),
        observation: str,
        action_catalog: Iterable[Any],
        relevant_action_history: Iterable[Any],
        remaining_budget: Mapping[str, Any],
        downstream_plan_context: Mapping[str, Any] | None = None,
        current_state_snapshot: Mapping[str, Any] | None = None,
        exploration_memory: Mapping[str, Any] | None = None,
        recent_failed_learned_invocation: Mapping[str, Any] | None = None,
    ) -> str:
        ready = (
            dict(execution_ready_bindings)
            if execution_ready_bindings is not None
            else dict(certified_bindings or {})
        )
        state = dict(current_state_snapshot or {
            "current_atomic": _project(
                atomic_contract, _ATOMIC_SEEDED_FIELDS,
            ),
            "semantic_anchors": dict(
                current_occurrence_semantic_anchors or {}
            ),
            "confirmed_bindings": ready,
            "candidate_bindings": {},
            "missing_bindings": [
                str(value) for value in missing_or_insufficient_bindings
            ],
            "invalidated_bindings": {},
            "preconditions": [],
            "effect_witness_status": {},
            "learned_invocation_ready": False,
            "blocking_reasons": [],
            "downstream_obligations": dict(downstream_plan_context or {}),
            "remaining_budget": _compact_budget(remaining_budget),
        })
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "task_semantic_context": _policy_value(dict(task_semantic_context or {})),
            "current_state_snapshot": _policy_value(state),
            "current_observation": _text(observation, "observation"),
            "current_action_catalog": _compact_catalog(action_catalog),
            "exploration_memory": _policy_value(dict(exploration_memory or {})),
            "recent_accepted_actions": _compact_history(
                relevant_action_history,
            ),
            "recent_failed_learned_invocation": _policy_value(
                dict(recent_failed_learned_invocation)
                if recent_failed_learned_invocation is not None
                else None
            ),
        }
        return _render(
            "Solve only the current Atomic occurrence with environment_action and "
            "validate_current_atomic. Mark every environment action as explore or "
            "attempt_current_atomic. Exploration never commits the Atomic merely because "
            "its effect happens to be true. current_state_snapshot.downstream_obligations describes how current "
            "outputs are consumed by the already-validated Runtime plan; use it as semantic "
            "intent, never current evidence, and choose concrete entities yourself. "
            "task_semantic_context describes the whole-task goal; only "
            "current_state_snapshot.semantic_anchors constrains this occurrence. "
            "current_state_snapshot is current authority; exploration_memory is historical. "
            "Stored Atomic summaries "
            "and guidelines are portable semantic guidance, never current bindings or evidence. "
            "Resolve missing unanchored arguments by "
            "instantiating that relational intent with task_semantic_context and current environment "
            "evidence; do not copy a same-named task field or the task's final destination merely "
            "because its name or type matches. Explore first when the required current relation is "
            "not yet evidenced. This prohibition applies only to roles absent from "
            "current_state_snapshot.semantic_anchors. When a role is explicitly anchored there, ground a "
            "compatible concrete current entity for that anchor, including when it is the task's final "
            "destination. This is a fresh Seeded session and contains no failed Tool body or "
            "failed Implementation mapping.",
            payload,
            sort_keys=False,
        )

    def dynamic_task(
        self,
        *,
        task_goal: str,
        observation: str,
        action_catalog: Iterable[Any],
        relevant_action_history: Iterable[Any],
        remaining_budget: Mapping[str, Any],
        task_progress: Mapping[str, Any] | None = None,
        rescue_method_guidance: Mapping[str, Any] | None = None,
        exploration_memory: Mapping[str, Any] | None = None,
        recent_failed_learned_invocation: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "task_goal": _text(task_goal, "task_goal"),
            "current_observation": _text(observation, "observation"),
            "current_action_catalog": _compact_catalog(action_catalog),
            "current_state_snapshot": {
                "task_progress": _policy_value(dict(task_progress or {})),
                "remaining_budget": _compact_budget(remaining_budget),
            },
            "exploration_memory": _policy_value(dict(exploration_memory or {})),
            "recent_accepted_actions": _compact_history(
                relevant_action_history,
            ),
            "recent_failed_learned_invocation": _policy_value(
                dict(recent_failed_learned_invocation)
                if recent_failed_learned_invocation is not None
                else None
            ),
        }
        if rescue_method_guidance is not None:
            payload["rescue_method_guidance"] = _policy_value(
                dict(rescue_method_guidance)
            )
        return _render(
            "Solve the task through native environment_action calls. The orchestrator, not prose, "
            "determines completion. task_progress is descriptive validator-backed state, "
            "not an action policy. Choose actions yourself. When public state and current "
            "actions make it possible to complete an already-started unsatisfied obligation, "
            "prefer making measurable progress before unrelated exploration. "
            "rescue_method_guidance, when present, is portable non-binding method context and "
            "must not be treated as current evidence or a hard concrete anchor.",
            payload,
            sort_keys=False,
        )

    def tool_builder(
        self,
        *,
        atomic: Any,
        provenance: Any,
        evidence_support: Iterable[Any] | None = None,
        semantic_delta: Mapping[str, Any] | None = None,
        harness_interface: Mapping[str, Any] | None = None,
        near_match_interfaces: Iterable[Any] | None = None,
        local_failures: Iterable[Any] | None = None,
    ) -> str:
        atomic_view = _project(
            atomic,
            ("summary", "inputs", "outputs", "preconditions", "effects"),
        )
        # The frozen ToolBuilder context never includes the complete task goal,
        # full trace, full planner history, full skill bank, or old Tool bodies.
        payload = {
            "canonical_atomic": atomic_view,
            "atomic_evidence_support": [
                _policy_value(item) for item in evidence_support or ()
            ],
            "semantic_delta": _policy_value(dict(semantic_delta or {})),
            "harness_interface": _policy_value(dict(harness_interface or {})),
            "tool_ir_schema": {
                "schema_version": 1,
                "opcodes": ["ACTION", "IF", "FOR_EACH", "STOP_WHEN", "RETURN"],
            },
            "safety_portability": {
                "no_python": True,
                "no_shell": True,
                "no_filesystem": True,
                "no_network": True,
                "no_task_id_constants": True,
                "no_episode_entity_constants": True,
                "no_benchmark_family_branch": True,
                "no_hidden_llm_call": True,
                "bounded_max_actions": True,
                "evidence_backed_outputs": True,
            },
            "near_match_interfaces": [
                _policy_value(item) for item in near_match_interfaces or ()
            ],
            "local_failure_facts": [
                _policy_value(item) for item in local_failures or ()
            ],
            "source_kind": (
                "success_evolution"
                if getattr(provenance, "source", "") == "success_evolution"
                else "runtime_automation"
                if getattr(provenance, "source", "") == "runtime_automation"
                else str(getattr(provenance, "source", "unknown"))
            ),
        }
        return _render(
            "You are ToolBuilder, the only v3.2 Tool Program author. "
            "You implement one already-proposed Atomic. "
            "The Atomic contract is authoritative. "
            "The source trace is evidence, not a program to replay. "
            "Use the minimal reusable procedure needed to realize the Atomic. "
            "Do not copy every event in the evidence envelope. "
            "Do not add task-specific workflow knowledge. "
            "Every branch, output, and expected effect must be representable in "
            "the supplied Harness interface. "
            "Return decision=no_tool if no safe reusable bounded implementation "
            "is justified. "
            "Call only the offered create_tool submission.",
            payload,
            sort_keys=False,
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

    def extractor_e1(
        self,
        *,
        canonical_trace: Any,
        known_atomic_contracts: Iterable[Any] = (),
        required_task_contract_witnesses: Any = (),
        runtime_automation_drafts: Iterable[Any] = (),
        runtime_tool_trials: Iterable[Any] = (),
    ) -> str:
        return _render(
            """Propose the smallest sufficient set of reusable Atomic capability occurrences
from the supplied code-authoritative successful trace.

The trace is the only factual authority. Do not assume a benchmark taxonomy,
task type, operation catalogue, or predefined workflow.

An Atomic occurrence is one independently meaningful and independently
verifiable state transition with:
- one coherent reusable intent;
- explicit external input/output identities;
- a minimal causal accepted-event slice;
- at least one authoritative positive Effect or narrow terminal certificate.

Use state-transition evidence rather than action wording as authority.

Do not extract:
- pure observation with no authoritative transition;
- repeated checks;
- failed attempts;
- loops;
- recovery actions;
- incidental search/exploration detours that have no reusable, validated
  evidence-domain Effect.
A bounded Runtime-created automation that has passed R1 and produces an
authoritative reusable evidence-domain Effect is not an incidental detour;
review it as an ordinary Atomic candidate.

A setup/helper action belongs inside an occurrence only when it is necessary
to replay the occurrence's core transition. A durable independently useful
transition should remain a separate occurrence. Do not merge distinct
effect-producing boundaries merely to reduce the number of occurrences.

intent requirements:
- concise lower_snake_case;
- describes exactly one reusable transition;
- remains correct after replacing every concrete entity with another entity
  having the same semantic role;
- contains no instance identifier;
- contains no source-episode object, location, receptacle, device, or task
  wording;
- contains no sequence of multiple intents.

If known_atomic_contracts contains an equivalent validated contract, reuse its
canonical_intent. Otherwise propose a new portable intent.

All episode-specific values belong only in input_roles/output_roles.
Never copy a concrete value into intent, rationale intended as a long-term
summary, or any reusable guideline.

event_start is inclusive and event_end is exclusive.
Ranges must be ordered and non-overlapping.

input_roles:
- non-empty;
- unique role-to-concrete-value bindings;
- every input must reference one supplied input_provenance_refs authority.

output_roles:
- non-empty;
- every required output must have exactly one code-verifiable derivation;
- INPUT_IDENTITY: exactly the same concrete identity as one declared input; or
- EFFECT_WITNESS: a concrete argument of one declared authoritative Effect witness.
Do not invent an output value.
Do not derive an output from observation prose.
Use only supplied boundary_authorities / effect witness refs.

preconditions:
- may be empty;
- include only facts semantically necessary for the core transition;
- copy only code-authoritative before-state facts.

effects:
- non-empty;
- copy only code-authoritative positive Effects or explicitly supplied narrow
  terminal certificates;
- never infer a fact from observation prose.

Code will independently validate every proposal. Invalid proposals are
discarded and cannot change the persistent graph.

This is a strict-success learning trace. The code-authoritative target witness
section identifies TaskContract effects already proven by accepted,
state-derived facts. Your complete proposal must preserve enough valid causal
transitions for the validated occurrences to collectively cover every supplied
target witness. Do not invent an Effect merely because the TaskContract
requires it; use only the supplied witness facts and their causal event slices.
When a witness is state-derived from earlier accepted transitions, select a
minimal occurrence slice whose declared Effect is exactly that authoritative
positive fact. Search/navigation detours remain non-learnable unless causally
required inside that occurrence.

  event_start/event_end are the temporal evidence envelope only. Explicitly
  select support_event_ids. Support events may be non-contiguous within one
  causal occurrence lineage. Do not include unrelated actions merely to make
  the interval contiguous. Precondition and effect witnesses must be explicit.
  Only extract causal capabilities supported before benchmark terminal success.

Call the offered native submission tool exactly once.""",
            {
                "canonical_trace": _policy_value(canonical_trace),
                "known_atomic_contracts": _policy_value(
                    list(known_atomic_contracts)
                ),
                "required_task_contract_witnesses": _policy_value(
                    required_task_contract_witnesses
                ),
                "runtime_created_atomic_drafts": _policy_value(
                    list(runtime_automation_drafts)
                ),
                "runtime_created_tool_proposals": [
                    _policy_value(item.get("proposal"))
                    for item in runtime_tool_trials
                    if isinstance(item, Mapping) and item.get("proposal")
                ],
                "runtime_tool_trials": _policy_value(
                    list(runtime_tool_trials)
                ),
                "boundary_authorities": _policy_value(
                    dict(
                        _policy_value(canonical_trace).get(
                            "boundary_authorities", {}
                        )
                    )
                ),
            },
        )

    def extractor_e2(
        self,
        *,
        canonical_occurrences: Iterable[Any],
        canonical_control_sequence: Iterable[str],
        known_existing_edge_evidence: Iterable[Any] = (),
        new_edge_candidates: Iterable[Any] = (),
    ) -> str:
        return _render(
            """The Composite represents the minimal reusable causal method, not a narration
of the source episode.

Use only the code-authoritative occurrences. Discard or correct any
conflicting memory from the previous turn.

The control sequence is code-authoritative and is not yours to rewrite.
Existing edges are code-authoritative; select only their supplied IDs. New
edge candidates have already passed deterministic endpoint, role,
binding-identity/type, or effect-precondition eligibility checks. Select only
the candidates semantically required by the reusable composition. Do not
invent an endpoint, role, edge ID, edge type, or provenance.

Do not describe:
- the benchmark;
- the task family;
- the number of validated nodes;
- source-episode entity names;
- validation mechanics such as "canonical control sequence".

summary:
- concise;
- reusable across entity substitutions;
- describes the capability composition, not the source task sentence.

guideline:
- contains only reusable ordering, dependency, and parameter-flow guidance;
- contains no concrete entity or location.

insight:
- may explain why the composition is reusable;
- may not invent facts or dependencies.

Do not select requires_skill solely to express temporal order; the canonical
control sequence already carries order and occurrences need not be
edge-connected.

  Use only validated canonical Atomics. Runtime-created support Atomics are
  ordinary candidates. Prefer a causally sufficient minimal subgraph. Do not
  retain planned-but-unexecuted post-terminal nodes. Keep a support Atomic if
  its evidence/output is actually consumed.

Call the offered native submission tool exactly once.""",
            {
                "canonical_occurrences": _policy_value(
                    list(canonical_occurrences)
                ),
                "canonical_control_sequence": _policy_value(
                    list(canonical_control_sequence)
                ),
                "known_existing_edge_evidence": _policy_value(
                    list(known_existing_edge_evidence)
                ),
                "new_edge_candidates": _policy_value(
                    list(new_edge_candidates)
                ),
            },
        )


def _render(
    instruction: str,
    payload: dict[str, Any],
    *,
    sort_keys: bool = True,
) -> str:
    return instruction + "\n\nPOLICY_CONTEXT_JSON\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=sort_keys,
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


def _compact_catalog(values: Iterable[Any]) -> dict[str, Any]:
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    revision: Any = None
    for value in values:
        mapping = _as_mapping(value)
        action_id = str(mapping.get("action_id", ""))
        if not action_id:
            raise ValueError("action catalog entry requires action_id")
        if action_id in seen:
            continue
        seen.add(action_id)
        entry_revision = mapping.get("revision")
        if revision is None:
            revision = entry_revision
        elif entry_revision != revision:
            raise ValueError("action catalog entries must share one revision")
        compact = {
            "action_id": action_id,
            "action_type": str(mapping.get("action_type", "")),
            "arguments": _policy_value(dict(mapping.get("arguments") or {})),
        }
        catalog.append(_policy_value(compact))
    return _policy_value({"revision": revision, "actions": catalog})


def _compact_history(values: Iterable[Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for value in values:
        mapping = _as_mapping(value)
        if mapping.get("accepted") is False:
            continue
        compact = {
            "action_type": str(mapping.get("action_type", "")),
            "arguments": _policy_value(dict(mapping.get("arguments") or {})),
            "observation": str(mapping.get("observation", "")),
            "revision": int(
                mapping.get("new_revision", mapping.get("revision", 0))
            ),
            "done": bool(mapping.get("done", False)),
            "won": bool(mapping.get("won", False)),
            "origin": str(mapping.get("origin", "")),
        }
        if mapping.get("intent"):
            compact["intent"] = str(mapping["intent"])
        history.append(_policy_value(compact))
    return history[-5:]


def _compact_budget(value: Mapping[str, Any]) -> dict[str, int]:
    mapping = dict(value)
    result = {
        "remaining_global_actions": max(
            0, int(mapping.get("remaining_global_actions", 0))
        ),
    }
    if bool(mapping.get("node_budget_active")):
        result["remaining_node_actions"] = max(
            0, int(mapping.get("remaining_node_actions", 0))
        )
    return result


__all__ = ["ContextBuilder"]
