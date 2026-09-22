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
from ..tooling.capability_boundary import CAPABILITY_BOUNDARY_RULES
from ..tooling.runtime_interface import public_tool_ir_condition_contract, public_tool_ir_collection_sources, OUTPUT_SEMANTIC_CONSTRAINT_RULES
from .runtime_policy_projection import project_runtime_payload
from .protocol import NativeToolSpec
from .runtime_prompt_texts import (
    DYNAMIC_PROMPT,
    R10_STEP_PROMPT,
    AUTOMATION_DRAFT_PROMPT,
)


_ATOMIC_RUNTIME_FIELDS = ("summary", "inputs", "outputs", "preconditions", "effects")
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


EXTRACTOR_COMPOSITE_PROMPT = """Compose only the supplied code-authoritative Atomic occurrences. Submit one submit_extractor_composite call using the offered schema. The canonical control sequence is fixed; do not create, delete, reorder, or reinterpret occurrences, their roles, or contracts.

Select existing edges and new candidate edges only by their supplied IDs, and keep each list in its correct namespace. An eligible candidate is not automatically required: select the relationships semantically needed by this composition. Do not infer a data-flow mapping from matching role names, lexical similarity, or an imagined task recipe. Do not generate a new mapping, tool body, or proof.

Use only the current canonical occurrences, their declared interfaces/identity relationships, fixed sequence, and edge candidates. Do not reconstruct E1 from memory. Write the required summary, guideline, and insight as portable descriptions of the selected composition, not as episode answers. Empty edge selections are allowed when the schema and the actual composition permit them; do not invent edges to fill a list.

Before submitting, check selected IDs, duplicates, source/target meaning, forward direction, and preservation of the fixed occurrence sequence. Code performs graph validation. Do not claim validation success or request another planning turn.
Do not select requires_skill merely for order: the fixed sequence already expresses it. Descriptions must not narrate benchmark families, source entity names, node counts, or validation mechanics."""


class ContextBuilder:
    """Build deterministic, compact user inputs for v3 Agent sessions."""

    def selected_support_summaries(self, skills, candidates):
        from ..evolution.portability import validate_portability
        result = {}
        for candidate in candidates:
            try:
                atomic = skills.get_atomic(candidate.atomic_ref)
            except KeyError:
                continue  # The projection records this as a missing summary.
            if validate_portability(atomic.summary).passed:
                result[candidate.atomic_ref] = atomic.summary
        return result

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
        runtime_automation_interface: Mapping[str, Any] | None = None,
        projection_audit: dict[str, Any] | None = None,
        runtime_step_mode: str | None = None,
        rejected_candidates: Iterable[Any] = (),
        execution_frame: Mapping[str, Any] | None = None,
        native_tool_specs: Iterable[NativeToolSpec] | None = None,
        support_summary_lookup: Mapping[str, str] | None = None,
    ) -> str:
        invocations = [
            _project(value, _INVOCATION_FIELDS) for value in implementation_invocations
        ]
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
        state = copy.deepcopy(current_state_snapshot or {
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
        from .skill_guidance import guidance_view
        state["current_atomic"]["skill_guidance"] = guidance_view(atomic_contract)
        # Progress/resources are represented once, in the execution frame.
        if execution_frame is not None:
            state.pop("remaining_budget", None)
            state.pop("last_step_feedback", None)
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
            "runtime_automation_interface": _policy_value(
                dict(runtime_automation_interface or {})
            ),
            "runtime_automation_drafts": _compact_runtime_automation_drafts(
                runtime_automation_drafts
            ),
        }
        if runtime_step_mode is not None:
            payload["runtime_step_mode"] = runtime_step_mode
            payload["rejected_candidates"] = _policy_value(list(rejected_candidates))
            payload["execution_frame"] = _policy_value(dict(execution_frame or {}))
        original_presentation = getattr(self, "runtime_presentation", "new") == "old"
        projected, audit = project_runtime_payload(payload, native_tool_specs=native_tool_specs,
                                                  expression_enabled=not original_presentation)
        projected, lean_instruction = self._release_projection(projected, audit, support_summary_lookup)
        if projection_audit is not None:
            projection_audit.update(copy.deepcopy(audit))
        rendered = _render(
            self._runtime_instruction("node") + lean_instruction + ("\n\n" + self._runtime_instruction("draft") + "\n\n"
                + OUTPUT_SEMANTIC_CONSTRAINT_RULES if runtime_automation_interface else ""),
            projected,
            sort_keys=original_presentation,
        )
        if projection_audit is not None:
            import hashlib
            projection_audit['final_render_hash'] = hashlib.sha256(rendered.encode()).hexdigest()
        return rendered

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
        task_runtime_frame: Mapping[str, Any] | None = None,
        native_tool_specs: Iterable[NativeToolSpec] | None = None,
        exploration_memory: Mapping[str, Any] | None = None,
        recent_failed_learned_invocation: Mapping[str, Any] | None = None,
        projection_audit: dict[str, Any] | None = None,
        support_summary_lookup: Mapping[str, str] | None = None,
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
        if task_runtime_frame is not None:
            payload["task_runtime_frame"] = _policy_value(dict(task_runtime_frame))
            if task_runtime_frame.get("runtime_automation_interface"):
                payload["runtime_automation_interface"] = payload["task_runtime_frame"].pop("runtime_automation_interface")
        original_presentation = getattr(self, "runtime_presentation", "new") == "old"
        projected, audit = project_runtime_payload(payload, native_tool_specs=native_tool_specs,
                                                  expression_enabled=not original_presentation)
        projected, lean_instruction = self._release_projection(projected, audit, support_summary_lookup)
        if projection_audit is not None:
            projection_audit.update(copy.deepcopy(audit))
        rendered = _render(
            self._runtime_instruction("dynamic") + lean_instruction + ("\n\n" + self._runtime_instruction("draft") + "\n\n"
                + OUTPUT_SEMANTIC_CONSTRAINT_RULES if payload.get("runtime_automation_interface") else ""),
            projected,
            sort_keys=original_presentation,
        )
        if projection_audit is not None:
            import hashlib
            projection_audit['final_render_hash'] = hashlib.sha256(rendered.encode()).hexdigest()
        return rendered

    def _release_projection(self, projected, audit, summaries):
        from .runtime_policy_projection import project_support_presentation
        projected, audit['support_presentation'] = project_support_presentation(projected, summaries)
        if getattr(self, 'presentation_profile', 'current') != 'lean':
            return projected, ''
        from .runtime_expression_codec import project_lean, ROWS_HELP
        result, details = project_lean(projected)
        audit['release_expression'] = details
        explanation = '\n\n' + ROWS_HELP if any(t['applied'] for t in details['transforms'].values()) else ''
        return result, explanation

    def _runtime_instruction(self, scope):
        if getattr(self, 'presentation_profile', 'current') == 'lean' and scope in {'node','dynamic'}:
            from .lean_runtime_prompt_texts import NODE, DYNAMIC
            from ..runtime.search_history import HISTORY_HELP
            return (NODE if scope == 'node' else DYNAMIC) + '\n\n' + HISTORY_HELP
        from . import runtime_prompt_texts as current
        from . import baseline_runtime_prompt_texts as original
        source = original if getattr(self, "runtime_presentation", "new") == "old" else current
        text = getattr(source, {"node": "R10_STEP_PROMPT", "dynamic": "DYNAMIC_PROMPT",
                                "draft": "AUTOMATION_DRAFT_PROMPT"}[scope])
        if scope in {'node', 'dynamic'}:
            from ..runtime.search_history import HISTORY_HELP
            text += '\n\n' + HISTORY_HELP
        return text

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
        additional_evidence_sources: Iterable[Any] | None = None,
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
            **({"additional_evidence_sources": _policy_value(list(additional_evidence_sources))}
               if additional_evidence_sources else {}),
            "output_semantic_constraints": _policy_value(dict(
                dict(atomic_mapping.get("validator_spec") or {}).get("output_semantic_constraints") or {}
            )),
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
                "condition_contract": public_tool_ir_condition_contract(),
                "collection_sources": public_tool_ir_collection_sources(),
                "evidence_selector_contract": {
                    "source": "semantic_evidence",
                    "where": {
                        "predicate": "exact public predicate",
                        "argument_role": "predicate argument compared with semantic_compatible_with",
                        "semantic_compatible_with": {
                            "source": "tool_input or in-scope local_variable",
                            "field": "declared input or local role",
                            "semantic_type": "optional public semantic type",
                        },
                    },
                    "project": {"kind": "argument", "role": "exact predicate argument"},
                    "availability": "Current structured public evidence is refreshed after accepted actions. A selector only reads facts; it does not create or guarantee a witness. Use in RETURN or FOR_EACH, not condition.match.",
                },
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
            """Implement exactly the supplied Atomic capability. Submit one native create_tool call with decision=create or decision=no_tool. ToolBuilder is the program author; code independently checks schema, static validity, replay/trial, and admission. Do not execute actions or output a program outside the native submission.

IMMUTABLE BOUNDARY
Use proposal_version="2". Echo the supplied atomic_ref, canonical_atomic.inputs, and canonical_atomic.outputs exactly. For create, copy canonical_atomic.effects into final_effects with the same predicates, argument keys, formal references, domains, cardinality, and distinctness. Preserve the supplied output-semantic constraints and derivations; do not relabel an output as an input merely because their returned values coincide. Do not repair an invalid capability contract by deleting its conditions or inventing different outputs.

EXTERNAL ENTRY VERSUS INTERNAL WORK
Provide entry_contract.conditions and entry_contract.grounding_constraints explicitly. Use only the grounding-constraint kinds offered by this request's schema. Express a supported state or relation precondition in entry_contract.conditions with the public predicate and explicit bindings. A predicate name is not a registered verifier ID. Do not invent callbacks or duplicate an Atomic precondition through an unavailable verifier. Keep genuine requirements; empty arrays are valid when no additional external requirement is needed. Do not elevate an optional branch's action requirement or an intermediate result to the whole tool's entrance. Atomic preconditions remain binding.
The call schema checks input presence and types. argument_exists is not a synonym for a supplied key: it requires the applicable current grounding evidence. argument_concrete authenticates a given identity, not its current location or affordance. A discovery tool accepting a category must not require the still-unknown discovered instance at entry. Empty arrays do not waive genuine requirements.

PROGRAM AND SYMBOLS
Use only the supplied Tool IR and public Harness vocabulary. Keep ACTION argument names separate from Atomic formal roles. ACTION.argument_mapping uses the supported kind/source_role expression for a declared input or in-scope local; constants must be portable and allowed. Graph data_flow/tool_output/adapter_transform are not Tool ACTION operands. ACTION.expected_effects are non-empty, evidence-justified immediate effects of that action; they are not the final contract copied onto every step. Use the field-specific reference forms in the syntax table. A declared fresh result can be referenced only where the actual action evidence can resolve it.

Use unique node IDs, existing nesting limits, positive max_iterations, and a positive max_actions bound. These are limits on actual work, not evidence that the program has executed. Keep local variables in lexical scope and ensure required values are defined on every path that uses them. Never turn an episode location or object seen in the example into a reusable constant.

SOURCE AND REPLAY
For success_evolution, semantic_delta.source_boundary identifies the declared capability entry. semantic_delta.before_facts contains the recorded, authorized source facts at that entry, not the first selected support event or the current live environment. source_input_bindings contains typed source-example values for this Atomic's formal inputs; use them only to interpret the source evidence, never as reusable program constants.
atomic_evidence_support lists the selected accepted events in their real Trace order, including their before/after revisions and positive/negative fact changes. These are per-event changes, not a complete final-state snapshot. Sparse support does not imply that unselected work has already executed in your new program. A missing projected fact does not prove absence or persistence, and later evidence cannot authorize an entry input or precondition.
Design against the declared entry, input boundary, supported causal work, and public primitive semantics. Do not repeat a historical preparation action unconditionally when the source entry already satisfies its purpose. Do not add obligatory actions merely to make a program look complete. Independently valid generalization remains allowed under the existing static and replay rules; the program need not copy the source action list. Required action failure remains failure.
For runtime_automation, use the supplied current public invocation context and existing R1 contract instead of requiring a historical source occurrence. Missing pre-trial final witnesses alone is not a reason for no_tool. Code, not this prompt or the source example, determines whether execution and output validation pass. Do not treat possible primitive signatures as currently executable actions.

CHANGING CANDIDATES
Use only collection and condition operators actually supplied by the interface. Understand snapshot versus refresh_each_iteration exactly as documented; neither mode guarantees that a previously collected value still has a required affordance after state changes. Guard genuinely optional work with an authored current query. Required action failure must remain failure, not be skipped by the executor. A filtered required collection or RETURN selector with no match is not successful progress. A visit alone does not prove discovery or absence. Apply the target constraints to the chosen candidates, stop condition, and returned values; an unrelated witness cannot complete the capability.

RETURN AND TERMINAL
RETURN.output_sources is keyed by actual output roles and uses source/field/project, not ACTION's kind/source_role notation. Return an identity from its declared input, a definitely defined in-scope local, or a supported structured evidence selector. Correlated outputs must agree with one valid relation, not independent arbitrary first matches. Do not rely on a transient witness still being available after unrelated later actions.
After official terminal, no new environment action or online model call is allowed. Only the original program's actually reachable pure tail may finish using available values. Do not jump over a pending action to reach RETURN. An unexecuted branch or unmet final effect cannot be reported as complete.

NO_TOOL
Choose no_tool when no supported safe bounded implementation can satisfy the supplied contract. Do not choose it merely because a contract needs multiple steps or has semantic inputs. Use the existing complete no_tool field shape supplied below; do not fabricate an executable or a PASS record. evidence_outputs and path_expectations may use their existing empty forms when unnecessary.

BEFORE SUBMISSION
Check: (1) exact boundary/final contract; (2) genuine entry requirements; (3) action vocabulary, arguments, and immediate effects; (4) mandatory source path or trial assumptions; (5) locals and bounds; (6) empty/stale/irrelevant candidate cases; (7) all required RETURN values and correlations; (8) portability. Perform this check within the single existing call; do not add a review turn or output extra self-check fields.

FIELD SYNTAX
ACTION input: {"kind":"skill_input","source_role":"<input>"}; scoped local: {"kind":"local_variable","source_role":"<local>"}. Expected-effect values use supported kind/source_role or $role references, never graph-only bindings. RETURN: {"<output>":{"source":"tool_input","field":"<input>"}} or scoped local_variable/structured semantic_evidence source/where/project. evidence_outputs entries use role, not output_role, with that selector shape.
Entry references use inputs or portable constants only. Maximum nesting is four; success_evolution loops require at least two structurally isomorphic distinct repetitions. Snapshot is the default; action_catalog refresh_each_iteration=true re-queries each iteration for the first unseen projected value. Neither mode skips a required failure. Public catalog relations describe argument mappings and availability, not routes or validator truth.
Complete no_tool shape: proposal_version="2", decision="no_tool", non-empty summary/rationale, exact atomic_ref and inputs/outputs, entry_contract={"conditions":[],"grounding_constraints":[]}, program=[], max_actions=1, final_effects=[], evidence_outputs=[], path_expectations=[]. The action bound is only a placeholder; no_tool is not executed.""" + "\n\n" + CAPABILITY_BOUNDARY_RULES + "\n\n" + OUTPUT_SEMANTIC_CONSTRAINT_RULES,
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
        generalization_groups: Iterable[Any] = (),
    ) -> str:
        groups = list(generalization_groups)
        return _render(
            """Extract independently useful, reusable Atomic capabilities from the supplied canonical successful experience. Submit exactly one submit_extractor_atomics call. You declare capability contracts and their evidence, not Tool programs. An empty occurrences array is valid when no eligible capability is supported. Do not invent a capability to complete a graph or meet a tool-count target.

CAPABILITY AND SOURCE
One Atomic has one coherent intent and independently verifiable results. It may include multiple native steps and necessary internal preparation. A loop is not a capability merely because it repeats; bounded exploration is eligible when it establishes a supplied authoritative world/evidence transition. Preserve the source's allowed ownership and causal boundaries. Distinct short provider sessions do not by themselves require distinct Atomics. Use the supplied coverage information to distinguish new preparation experience from work already covered by a completed tool. Reuse an equivalent known contract when appropriate; do not merge an entire task into one capability.

ENTRY, INTERNAL VALUES, AND OUTPUTS
Declare boundary_schema_version="2", input_specs, output_specs, local_value_authority_refs, and output_semantic_constraints. An external input must be available at the declared event_start. Its required_resolution and semantic_type describe the capability requirement; neither an identifier-looking string nor a role name proves them. A semantic input is not the concrete instance found later. A value obtained inside the slice may remain internal if its supplied local authority is available before each actual use. Local evidence references are not Tool variable declarations. ToolBuilder writes the bounded program and its local variables later.

For each formal input, input_provenance_refs[formal_role] cites {authority_ref, source_role} only from the supplied typed boundary_authorities.inputs. source_role must equal the cited authority's role field, not its optional source_role ancestry metadata. The formal name may differ through this explicit mapping. An action argument or a post-action alias in the history is not an entry certificate. A listed authority must still satisfy your declared entry time, type, resolution and ownership. Internal values may use the supplied local authorities at their actual time of use. Do not invent or repair a reference, strip instance suffixes, or derive authority from observation prose.
For each input, explicitly choose runtime_resolvable for future use, not for the historical sample. Use true when a Runtime Agent is allowed to obtain and certify the value through the offered public task/environment interfaces. Use false only when this capability requires its caller or an explicit upstream binding to supply that value instead. A concrete value known at source entry, and a Tool that does not itself search, are not reasons to set false. Required inputs must still be present at invocation and meet required_resolution; source-entry evidence, identity, scope and Repeat checks remain mandatory. Do not replace an existing binding merely because this flag is true. Reuse a known contract only when its input-resolution responsibility is also appropriate. The same intent name with a different responsibility is not an identical contract. Do not edit an existing stored contract to fit this trace.

TIME AND OWNERSHIP
Use [event_start,event_end): start is inclusive and end is exclusive. An event i uses [i,i+1). The entry event is the canonical_trace.actions record whose event_index equals event_start. event_start is an immutable Trace coordinate, not necessarily the current list position. Use that record's authoritative_before_state_facts. Do not move the entry to the first selected support event. Select explicit, unique accepted support_event_ids within that envelope. Their real Trace order is authoritative. Do not absorb rolled-back events or independent owners. Shared prerequisite evidence does not authorize duplicate ownership of an effect-producing event. Each phase_id is unique in this submission; it need not copy a Runtime owner ID.

RESULTS AND CONDITIONS
Every declared effect and precondition must have matching supplied witnesses, argument keys, domain, and correct time. Preserve cardinality, distinctness, and existing identity obligations. Conditions created by internal preparation are not entry conditions. Do not invent an effect because the task asks for it.
Every required output has one explicit derivation. input_identity returns exactly a declared input's typed value and its actually certified identity level. effect_witness identifies the actual argument of a declared effect. Fresh is relative to this capability's external inputs, not necessarily never seen anywhere in the episode. An entity output equal to an existing explicit entity input must use input_identity, not be relabeled fresh. The sparse output-semantic rule supplied below governs category compatibility; a joint location or containment relation instead requires its declared relational evidence.

Write portable guideline steps and notes within the offered schema limits. These are soft experience suggestions, not a mandatory action script. Keep episode values in the source evidence/value fields, not in reusable intent or guideline. Do not change a rejected sibling to make another proposal appear valid.

BEFORE SUBMISSION
Check: (1) unique phase IDs and valid half-open ranges; (2) input availability at the declared entry; (3) exact formal-to-authority mapping; (4) local availability and source ownership; (5) matching precondition/effect witnesses; (6) output identity/derivation and sparse compatibility; (7) portable guideline and unchanged known equivalent contracts. Check within this call; do not output a separate checklist, request an extra review, or claim that code validation has already passed.
Use a portable lower_snake_case intent; preserve the canonical_intent of an equivalent known contract. Input/output role values are source-example values, not new authority.""" + "\n\n" + CAPABILITY_BOUNDARY_RULES + "\n\n" + OUTPUT_SEMANTIC_CONSTRAINT_RULES + (
"\nYou may propose at most one optional generalizations sidecar for the offered group. Give exactly one current and one history source_proposal using the same occurrence schema. Each source keeps its own values, half-open event coordinates, owner and authority refs. Both proposals must express one identical reusable contract; code validates each separately. Do not intersect conditions, import prior evidence into current, or insert the sidecar into occurrences merely to fill the current graph. Prefer a coherent capability including necessary preparation, often 2-4 actions when supported, not a forced action count. Empty generalizations is valid." if groups else ""),
            {
                "canonical_trace": _policy_value(canonical_trace),
                **({"generalization_groups": _policy_value(groups)} if groups else {}),
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
            EXTRACTOR_COMPOSITE_PROMPT,
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
    from .portable_support_view import portable_support_view
    if hasattr(value, 'effects'):
        value = portable_support_view(value)
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


def _compact_runtime_automation_drafts(
    values: Iterable[Any],
) -> list[dict[str, Any]]:
    """Keep task-local decisions while excluding Tool programs and mappings."""

    projected: list[dict[str, Any]] = []
    for value in values:
        mapping = _as_mapping(value)
        draft_raw = mapping.get("draft", mapping)
        draft = _as_mapping(draft_raw)
        item = {
            key: _policy_value(draft[key])
            for key in ("draft_id", "intent", "source_occurrence_id")
            if key in draft
        }
        for key in (
            "stage", "r0_passed", "static_passed", "r1_passed",
            "failure_code", "message",
        ):
            if key in mapping:
                item[key] = _policy_value(mapping[key])
        trial = mapping.get("trial")
        if isinstance(trial, Mapping):
            item["trial"] = {
                key: _policy_value(trial[key])
                for key in ("r1_outputs", "r1", "terminal_interrupted")
                if key in trial
            }
        projected.append(item)
    return projected


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
    result["node_actions_used"] = int(mapping.get("used_node_actions", 0))
    return result


def _project_tool_builder_facts(values: Iterable[Any]) -> list[dict[str, Any]]:
    """The existing capability-local fact whitelist; never a raw snapshot."""
    return [{key: copy.deepcopy(fact[key]) for key in (
        "predicate", "args", "cardinality", "distinct_by", "effect_domain",
        "witness_ref", "revision",
    ) if key in fact} for fact in values]


def _compact_tool_builder_evidence(values: Iterable[Any]) -> list[dict[str, Any]]:
    """Expose only the bounded Atomic occurrence's structured authorities."""

    result: list[dict[str, Any]] = []
    for value in values:
        mapping = _as_mapping(value)
        result.append(_policy_value({
            "event_index": mapping["event_index"],
            "event_id": mapping.get("event_id") or mapping["action_id"],
            "action_type": mapping["action_type"],
            "arguments": mapping["arguments"],
            "accepted": mapping["accepted"],
            "before_revision": mapping["before_revision"],
            "after_revision": mapping["after_revision"],
            "authoritative_positive_effects": _project_tool_builder_facts(mapping["authoritative_positive_effects"]),
            "authoritative_negative_effects": _project_tool_builder_facts(mapping["authoritative_negative_effects"]),
        }))
    return result


__all__ = ["ContextBuilder"]
