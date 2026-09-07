"""Compact policy-facing prompts for Planner/Runtime/Extractor sessions.

This module accepts only the fields the design permits an Agent to observe.  It
does not accept a validator snapshot, hidden benchmark state, a Tool body, or a
whole persistent graph, keeping those channels separated by construction.
"""

from __future__ import annotations

import copy
import json
from dataclasses import is_dataclass
from typing import Any, Iterable, Mapping

from ..core.serialization import to_primitive
from .runtime_policy_projection import project_runtime_payload
from .runtime_prompt_texts import (
    DYNAMIC_PROMPT,
    PREPARATION_PROMPT,
    SEEDED_PROMPT,
)


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
        projection_audit: dict[str, Any] | None = None,
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
        projected, audit = project_runtime_payload(payload)
        if projection_audit is not None:
            projection_audit.update(copy.deepcopy(audit))
        return _render(
            PREPARATION_PROMPT,
            projected,
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
        projection_audit: dict[str, Any] | None = None,
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
        projected, audit = project_runtime_payload(payload)
        if projection_audit is not None:
            projection_audit.update(copy.deepcopy(audit))
        return _render(
            SEEDED_PROMPT,
            projected,
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
        projection_audit: dict[str, Any] | None = None,
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
        projected, audit = project_runtime_payload(payload)
        if projection_audit is not None:
            projection_audit.update(copy.deepcopy(audit))
        return _render(
            DYNAMIC_PROMPT,
            projected,
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
        atomic_mapping = _as_mapping(atomic)
        provenance_mapping = _as_mapping(provenance)
        atomic_ref = provenance_mapping.get("atomic_ref")
        if not isinstance(atomic_ref, str) or not atomic_ref.strip():
            raise ValueError("ToolBuilder provenance atomic_ref must be non-empty")
        atomic_view = _project(
            atomic,
            ("summary", "inputs", "outputs", "preconditions", "effects"),
        )
        atomic_evidence_support = _compact_tool_builder_evidence(
            evidence_support or ()
        )
        atomic_effect_witness_refs = list(dict.fromkeys(
            str(fact.get("witness_ref", ""))
            for event in atomic_evidence_support
            for fact in event.get("authoritative_positive_effects", ())
            if str(fact.get("witness_ref", ""))
        ))
        source_kind = (
            "success_evolution"
            if provenance_mapping.get("source") == "success_evolution"
            else "runtime_automation"
            if provenance_mapping.get("source") == "runtime_automation"
            else str(provenance_mapping.get("source", "unknown"))
        )
        # The frozen ToolBuilder context never includes the complete task goal,
        # full trace, full planner history, full skill bank, or old Tool bodies.
        payload = {
            "canonical_atomic": atomic_view,
            "atomic_ref": atomic_ref,
            "atomic_output_derivations": _policy_value(dict(
                dict(atomic_mapping.get("validator_spec") or {}).get(
                    "output_derivations"
                ) or {}
            )),
            "atomic_evidence_support": atomic_evidence_support,
            "atomic_effect_witness_refs": atomic_effect_witness_refs,
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
            "historical_loop_evidence_required": (
                source_kind == "success_evolution"
            ),
            "source_kind": source_kind,
        }
        return _render(
            """You are the ToolBuilder. Submit exactly one native create_tool call for the supplied Atomic, or submit decision=no_tool. Do not execute environment actions. Do not return a program as prose, Markdown, or standalone JSON.

AUTHORITY AND IMMUTABLE FIELDS
The supplied canonical_atomic defines the capability. The supplied atomic_ref is its identity. Echo atomic_ref exactly. Echo canonical_atomic.inputs and canonical_atomic.outputs exactly, including each role name, semantic_type, required, runtime_resolvable, and required_resolution. Do not add, delete, rename, or reinterpret a boundary role.
Only the supplied Harness action/predicate interfaces and structured evidence may justify the implementation. Do not invent an action, predicate, argument role, current fact, or output. Do not copy an episode entity identifier into a reusable program constant.

KEEP THREE DIFFERENT THINGS SEPARATE
1. ACTION.argument_mapping says how to call a primitive now.
2. ACTION.expected_effects says what must be true immediately after that particular action.
3. ToolProposal.final_effects is the Atomic's final contract, not a restatement in your preferred variable names.
For a create proposal, copy canonical_atomic.effects into final_effects without changing predicate names, argument keys, formal source_role values, cardinality, distinct_by, or effect_domain. Do not rename a final output role to an input role just because RETURN will give them the same concrete value. Do not add extra final effects.
For example, when the supplied final Effect uses source_role="result" and RETURN maps result to input item, the final Effect must still use "result", not "item". A step-level Effect may use "item" when that is the value justified immediately after the step. The example names are illustrative; use only this Atomic's actual roles.

ACTION ARGUMENTS AND STEP EFFECTS
Use an action_type from harness_interface.primitive_actions and exactly its declared argument roles. In ACTION.argument_mapping, use {"kind":"skill_input","source_role":"<declared input>"}; a loop-local value inside its valid loop body uses {"kind":"local_variable","source_role":"<iteration variable>"}. Use constant only for a genuinely portable literal allowed by the interface, never for an episode entity or task identifier.
Every ACTION needs a non-empty expected_effects list. Each effect must use the supplied predicate vocabulary, its exact argument roles, and its declared effect_domain. State only effects justified after that exact action; do not place later effects on an earlier step.
Within ACTION.expected_effects, formal references use {"kind":"skill_input","source_role":"<formal role>"} or the supported $role notation; a definitely defined loop-local value may use {"kind":"local_variable","source_role":"<iteration variable>"}. A declared fresh output may be referenced only when the current action's structured evidence can resolve it. Prefer a currently known input/local value when it already names the affected entity.
Do NOT use kind=tool_output, kind=data_flow, or kind=adapter_transform in ACTION.expected_effects. Adding source_step does not make tool_output valid in this field. Do not confuse the broader graph BindingExpression vocabulary with the narrower Tool IR field rules.

RETURN AND OUTPUT DERIVATIONS
RETURN.output_sources is an object keyed by the actual Atomic output roles. Do not put a selector directly at the top level of output_sources.
For an input identity, use:
{"<output_role>":{"source":"tool_input","field":"<input_role>"}}
For a loop local, use source=local_variable and field=<iteration variable> within its valid scope.
For a value read from structured semantic evidence, use:
{"<output_role>":{"source":"semantic_evidence","where":{"predicate":"<supplied predicate>"},"project":{"kind":"argument","role":"<predicate argument role>"}}}
Add the supported filters required to select the correct value. A selector must resolve the required identity; do not rely on an arbitrary first match. Do not infer selector results from prose.
Follow atomic_output_derivations. An input_identity must return that input's exact identity. An effect_witness must agree with the declared predicate argument and the real witness; an output name alone is not evidence. A direct input return is permissible only when the implementation and contract prove it is that same witnessed identity.
Do not use kind=skill_input, kind=tool_output, or kind=data_flow as a RETURN source. RETURN uses source/field/project, not argument_mapping's kind/source_role form.
Every successful return path must supply the required outputs. Loop variables do not exist outside their loop bodies. Put shared work after IF branches only when its required values are available on both branches.

BOUNDING AND SAFETY
Use only ACTION, IF, FOR_EACH, STOP_WHEN, RETURN. Give nodes unique non-empty node_id values. Keep nesting within the existing maximum of four levels. Set a positive max_actions that bounds actual primitive actions; control nodes do not count as primitive actions. Bound every FOR_EACH with a positive max_iterations.
In success_evolution, propose FOR_EACH only when the supplied evidence contains at least two structurally isomorphic distinct repetitions. In runtime_automation, loop behavior may instead earn evidence through the existing task-local R1 trial. Do not change this distinction.
Use only supported condition/selector sources and operators. No Python, shell, filesystem, network, hidden model calls, task-family branches, or episode-specific constants.
Evidence_outputs are optional: use [] when unnecessary. When supplied, each entry must name an actual output role and a supported source. Path expectations describe what must be verified; never claim that an unexecuted path has already passed.

NO_TOOL IS A VALID DECISION, NOT A FAKE EXECUTABLE
If no safe, reusable, bounded implementation can satisfy the supplied contract, submit decision=no_tool with a specific rationale. Still include every field required by the offered native-tool schema: proposal_version="1", a non-empty summary, the supplied atomic_ref, the supplied input/output lists, program=[], max_actions=1, final_effects=[], evidence_outputs=[], path_expectations=[], and rationale. The value 1 is a schema-compatible placeholder, not permission to execute an action. Code does not compile or run a no_tool proposal.

FINAL CHECK BEFORE THE SINGLE SUBMISSION
Check the immutable boundary, exact final_effects copy, each ACTION's argument/step-effect rules, required RETURN output keys, local scopes, bounds, and portability. If any required value or effect cannot be justified, choose no_tool rather than inventing it. Do this within the existing call and token budget; do not request an additional repair turn.""",
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

event_start is inclusive and event_end is exclusive in this submission. A single event at index i uses [i, i+1). Code performs the exclusive-to-inclusive conversion; do not subtract one yourself.
The precondition boundary is exactly canonical_trace.actions[event_start].authoritative_before_state_facts. It is not the state before an arbitrary support event and not any historical state inside the envelope.
Choose the smallest evidence envelope that contains the necessary support events without moving the entry boundary earlier than the stated preconditions. A fact created by a support/setup event inside the envelope is an intermediate fact, not an entry precondition. Either describe the occurrence from an earlier valid entry state with that setup included, or start the core occurrence after setup and cite its actual entry-state witnesses. Do not erase necessary preconditions merely to pass validation; do not attach later witnesses to an earlier start.
An event at index i normally creates its after-state at that action's after_revision. The event index and world revision are different fields. Copy the supplied references; never build a witness string by arithmetic.
Temporal envelopes may overlap, but support_event_ids determine event ownership. Preserve the existing shared-precondition and independent-Effect ownership rules below.
Temporal evidence envelopes may overlap when Atomics share prerequisite context.
support_event_ids, not envelope overlap, define effect-producing event ownership.
Do not assign the same effect-producing support event to multiple independent
Atomics. shared_precondition_event_ids is not a general list of prerequisite
events: every listed event must also be selected in support_event_ids by this
occurrence and by at least one other proposed occurrence. Use it only to mark
that shared support event as precondition evidence in every owner. Ordinary
prerequisites belong in the temporal envelope and precondition_witness_refs,
not shared_precondition_event_ids. Code accepts shared support ownership only
when the event is not claimed as an Effect witness by two independent Atomics.

input_roles:
- non-empty; every role has a concrete value supported by one supplied authority;
- unique role-to-concrete-value bindings;
- input_provenance_refs keys must exactly equal input_roles keys;
- for every input role r, select exactly one supplied boundary_authorities.inputs entry a with a.role == r and a.value == input_roles[r]; then copy a.authority_ref exactly;
- do not rename an input to a more descriptive alias while citing an authority for a different role. If a.role is object, using light or container as the input key with that same reference is invalid under this interface;
- this equality applies to the Atomic input role and the input authority role, not to a predicate's argument name. A predicate argument such as location may legitimately refer to an input named destination;
- do not invent a new input authority, derive one from prose, or borrow an authority outside the supplied event/lineage boundary;
- if no permitted authority supplies a required input, revise the proposed occurrence using the actual evidence; do not fabricate a match.

output_roles:
- non-empty;
- every required output must have exactly one code-verifiable derivation;
- INPUT_IDENTITY: exactly the same concrete identity as one declared input; or
- EFFECT_WITNESS: a concrete argument of one declared authoritative Effect witness.
Do not invent an output value.
Do not derive an output from observation prose.
Use only supplied boundary_authorities / effect witness refs.

preconditions:
- may be empty only when the proposed transition genuinely needs no declared entry facts;
- include only necessary facts present in canonical_trace.actions[event_start].authoritative_before_state_facts;
- precondition_witness_refs must name those exact entry-state certificates, with matching predicate, arguments, and effect_domain;
- a fact may persist across revisions, but its certificate at a later revision is not interchangeable with the certificate at the entry boundary;
- facts established inside the selected envelope are not entry preconditions;
- preserve the supplied predicate argument keys and effect_domain; do not infer a precondition from task wording or observation prose.

Boundary example, for syntax only: if event 5 establishes fact P, event 6 uses P to establish Q, and P was absent before event 5, an occurrence starting at 5 cannot cite P as an entry precondition. An occurrence selecting event 6 alone uses [6,7) and may cite the P certificate supplied in actions[6].authoritative_before_state_facts. Use the actual event IDs/revisions and facts from this trace, not these example numbers.

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

Before the one native submission, verify every proposed occurrence independently: [event_start,event_end) contains its support_event_ids; each input's authority has the same role and value; every precondition reference belongs to the exact entry snapshot; every Effect reference belongs to the selected support events and matches the declared predicate/domain; every output has one legal input_identity or effect_witness derivation. Do not change correct sibling occurrences to hide an invalid one. This self-check adds no tool call and no retry.

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


def _compact_tool_builder_evidence(values: Iterable[Any]) -> list[dict[str, Any]]:
    """Expose only the bounded Atomic occurrence's structured authorities."""

    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        mapping = _as_mapping(value)
        positive_effects = [
            {
                key: _policy_value(fact[key])
                for key in (
                    "predicate", "args", "cardinality", "distinct_by",
                    "effect_domain", "witness_ref", "revision",
                )
                if key in fact
            }
            for raw in mapping.get("authoritative_positive_effects", ())
            if isinstance(raw, Mapping)
            for fact in [dict(raw)]
        ]
        result.append(_policy_value({
            "event_id": str(
                mapping.get("event_id", mapping.get("action_id", index))
            ),
            "action_type": str(mapping.get("action_type", "")),
            "arguments": dict(mapping.get("arguments") or {}),
            "accepted": bool(mapping.get("accepted", True)),
            "before_revision": int(mapping.get("before_revision", 0)),
            "after_revision": int(
                mapping.get("after_revision", mapping.get("new_revision", 0))
            ),
            "authoritative_positive_effects": positive_effects,
        }))
    return result


__all__ = ["ContextBuilder"]
