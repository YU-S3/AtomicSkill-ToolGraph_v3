"""Gate36: the runtime-created self-tool must close the full System loop.

This test deliberately mutates the bank only through
``AtomicSkillGraphSystem.run_task``.  The Harness and provider are deterministic
protocol implementations, but Success Extraction, Atomicization, ToolBuilder,
Admission, Registry persistence, planning, support retrieval, and execution are
the production objects wired by ``AtomicSkillGraphSystem``.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.core.errors import AtomicSkillGraphError, FailureLayer
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.contract_canonicalizer import (
    atomic_contract_signature,
    canonical_atomic_contract,
)
from atomic_skillgraph.harness.protocol import HarnessActionResult
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments.fakes import FakeProviderRequest, FakeReply, ScriptedAgentProvider

from test_v32_r21_cross_task_reuse import (
    LocatingHarness,
    _LOCATE_PROGRAM,
    _locate_draft_call,
    _locating_task,
    _room_for,
)


class Gate36Harness(LocatingHarness):
    """Separate Atomic completion from benchmark terminal completion.

    ``TAKE`` establishes the target predicate but remains non-terminal so a
    declarative Tool can execute its required RETURN node during admission
    replay.  The independent ``FINISH`` action then latches benchmark success.
    This models the common non-final Atomic case without weakening production
    replay rules for terminal-interrupted Tools.
    """

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        if self._task is None:
            raise RuntimeError("Gate36Harness must be reset before execution")
        if self._done or self._won:
            raise AtomicSkillGraphError(
                "harness_terminal_latched",
                "benchmark terminal is latched; no further env.step is allowed",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        spec = self._catalog.get(action_id, revision)
        target = str(self._task.context["target_item"])
        room = _room_for(target)
        accepted = False
        observation = "Nothing happens."
        if spec.action_type == "SEARCH" and spec.arguments == {"target": target}:
            accepted = True
            observation = f"You search and find {target} at {room}."
        elif (
            spec.action_type == "TAKE"
            and spec.arguments == {"object": target, "location": room}
            and not self._held
        ):
            accepted = True
            self._held = True
            observation = f"You take {target}; the task can now be finalized."
        elif spec.action_type == "FINISH" and self._held:
            accepted = True
            self._won = self._done = True
            observation = "Task finalized."
        self._revision += 1
        catalog = self._replace_catalog()
        self._validator.record(
            spec,
            accepted=accepted,
            revision=self._revision,
            done=self._done,
            won=self._won,
            observation=observation,
            catalog=catalog,
        )
        return HarnessActionResult(
            accepted,
            observation,
            self._done,
            self._won,
            self._revision,
            catalog,
            {"action_type": spec.action_type},
        )

    def _replace_catalog(self):
        if self._task is None or not self._held:
            return super()._replace_catalog()
        return self._catalog.replace([{
            "action_type": "FINISH",
            "arguments": {},
            "display_text": "finish task",
        }], self._revision)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "method_patch": "3.2",
        "data_dir": str(tmp_path / "bank"),
        "trace_data_dir": str(tmp_path / "traces"),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "deterministic-gate36",
            "api_key_env": "MODEL_API_KEY",
            "planner": {"max_turns": 4, "max_total_tokens_per_task": 120000},
            "runtime": {
                "max_total_tokens_per_node": 80000,
                "max_total_tokens_per_task": 300000,
                "protocol_repair_limit": 1,
                "learned_toolcall_repair_limit": 2,
            },
            "extractor": {"max_turns": 2, "max_total_tokens_per_task": 262144},
        },
        "planner": {
            "max_repeat_count": 4,
            "max_runtime_occurrences": 16,
        },
        "runtime": {
            "global_action_budget": 100,
            "node_action_budget": 35,
        },
        "extraction": {
            "extract_full_dynamic_success": True,
            "extract_task_rescue_success": True,
            "extract_novel_seeded_success": True,
            "skip_stable_direct_success": True,
        },
        "cold_start": {"enabled": False},
        "experiment": {
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "allow_long_term_knowledge_writes": True,
            "output_dir": str(tmp_path / "run"),
        },
    }


def _derived_locate_atomic(
    witness_ref: str,
    *,
    kind: str = "effect_witness",
    predicate: str = "entity.discovered_at",
    argument_role: str = "location",
) -> AbstractAtomicSkill:
    derivation = (
        {
            "kind": "input_identity",
            "input_role": "target",
            "witness_refs": [witness_ref],
        }
        if kind == "input_identity"
        else {
            "kind": kind,
            "predicate": predicate,
            "argument_role": argument_role,
            "witness_refs": [witness_ref],
        }
    )
    return AbstractAtomicSkill(
        SkillRef("gate36_locate_contract", "1.0.0"),
        "locate target",
        [ParameterSpec("target", "entity")],
        [ParameterSpec("location", "location")],
        [],
        [SemanticPredicate(
            "entity.discovered_at",
            {
                "entity": BindingExpression(
                    BindingExprKind.SKILL_INPUT, source_role="target",
                ),
                "location": BindingExpression(
                    BindingExprKind.SKILL_INPUT, source_role="location",
                ),
            },
            effect_domain="evidence",
        )],
        {
            "validator_id": "harness_atomic_effect",
            "output_derivations": {"location": derivation},
        },
        [],
        {},
        {"source_trace_ids": [witness_ref]},
        SkillStatus.CANDIDATE,
    )


def test_gate36_atomic_identity_excludes_only_output_witness_provenance() -> None:
    first = _derived_locate_atomic("witness:task-a")
    second = replace(
        _derived_locate_atomic("witness:task-b"),
        ref=SkillRef("gate36_locate_other_trace", "1.0.0"),
    )

    assert atomic_contract_signature(first) == atomic_contract_signature(second)
    identity_derivations = canonical_atomic_contract(first)[
        "validator_contract"
    ]["output_derivations"]
    assert all(
        "witness_refs" not in item
        for item in identity_derivations.values()
    )
    # Audit provenance remains on the persisted artifact contract itself.
    assert first.validator_spec["output_derivations"]["location"]["witness_refs"] == [
        "witness:task-a"
    ]

    semantic_variants = [
        _derived_locate_atomic(
            "witness:task-b", predicate="object.at_location",
        ),
        _derived_locate_atomic(
            "witness:task-b", argument_role="entity",
        ),
        _derived_locate_atomic(
            "witness:task-b", kind="input_identity",
        ),
    ]
    assert all(
        atomic_contract_signature(first) != atomic_contract_signature(variant)
        for variant in semantic_variants
    )


def _input_specs(atomic: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(item)) for item in atomic["inputs"]]


def _output_specs(atomic: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(item)) for item in atomic["outputs"]]


def _tool_builder_reply(request: FakeProviderRequest) -> dict[str, Any]:
    """Build exactly the Tool for the code-supplied canonical Atomic."""

    atomic = request.policy_context["canonical_atomic"]
    effects = copy.deepcopy(list(atomic["effects"]))
    predicate = str(effects[0]["predicate"])
    common = {
        "proposal_version": "1",
        "decision": "create",
        "atomic_ref": "skill://gate36_builder_boundary@1.0.0",
        "inputs": _input_specs(atomic),
        "outputs": _output_specs(atomic),
        "max_actions": 1,
        "final_effects": effects,
        "evidence_outputs": [],
        "path_expectations": [],
    }
    if predicate == "entity.discovered_at":
        return {
            **common,
            "summary": "locate target entity",
            "program": copy.deepcopy(_LOCATE_PROGRAM),
            "rationale": "Perform one bounded target search and return its evidence.",
        }
    if predicate != "agent.holds":
        raise AssertionError(f"unexpected Gate36 Atomic predicate: {predicate!r}")
    return {
        **common,
        "summary": "take target object",
        "program": [
            {
                "node_id": "take",
                "op": "ACTION",
                "action_type": "TAKE",
                "argument_mapping": {
                    "object": {"kind": "skill_input", "source_role": "object"},
                    "location": {"kind": "skill_input", "source_role": "location"},
                },
                "expected_effects": effects,
            },
            {
                "node_id": "ret",
                "op": "RETURN",
                "output_sources": {
                    "held_object": {"source": "tool_input", "field": "object"},
                },
            },
        ],
        "rationale": "Perform one bounded take action and return the held identity.",
    }


class _RequestAwareToolReply(FakeReply):
    """Materialize a native ``create_tool`` call from its request context."""

    def materialize(self, *, call_id, tools, request=None):
        if request is None:
            raise AssertionError("Gate36 ToolBuilder reply requires its request")
        return FakeReply.tool(
            "create_tool", _tool_builder_reply(request),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
        ).materialize(call_id=call_id, tools=tools, request=request)


def _e1_reply(request: FakeProviderRequest) -> dict[str, Any]:
    """Select every causal accepted action using only supplied authorities."""

    context = request.policy_context
    actions = list(context["canonical_trace"]["actions"])
    boundary_effects = list(context["boundary_authorities"]["effects"])
    witnesses_by_event: dict[str, list[dict[str, Any]]] = {}
    for fact in boundary_effects:
        event_id = str(fact.get("event_id", ""))
        if not event_id:
            witness_ref = str(fact.get("witness_ref", ""))
            for action in actions:
                candidate = str(action.get("event_id", action.get("action_id", "")))
                if candidate and (
                    witness_ref == candidate
                    or candidate in witness_ref
                    or witness_ref in {
                        str(item.get("witness_ref", ""))
                        for item in action.get("authoritative_positive_effects", [])
                    }
                ):
                    event_id = candidate
                    break
        witnesses_by_event.setdefault(event_id, []).append(fact)

    occurrences: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if action.get("accepted") is not True:
            continue
        event_id = str(action.get("event_id", action.get("action_id", "")))
        action_type = str(action["action_type"])
        arguments = dict(action.get("arguments") or {})
        candidates = witnesses_by_event.get(event_id, [])
        if not candidates:
            candidates = [
                dict(item)
                for item in action.get("authoritative_positive_effects", [])
            ]
        if action_type == "SEARCH":
            fact = next(
                item for item in candidates
                if item.get("predicate") == "entity.discovered_at"
            )
            target = str(arguments["target"])
            concrete_args = dict(fact["args"])
            occurrences.append({
                "phase_id": f"locate_{index}",
                "intent": "locate target entity",
                "event_start": index,
                "event_end": index + 1,
                "support_event_ids": [event_id],
                "input_roles": {"target": target},
                "input_provenance_refs": {
                    "target": f"action_arg:{event_id}:target",
                },
                "output_roles": {
                    "entity": concrete_args["entity"],
                    "location": concrete_args["location"],
                },
                "output_derivations": {
                    "entity": {
                        "kind": "effect_witness",
                        "predicate": "entity.discovered_at",
                        "argument_role": "entity",
                    },
                    "location": {
                        "kind": "effect_witness",
                        "predicate": "entity.discovered_at",
                        "argument_role": "location",
                    },
                },
                "preconditions": [],
                "precondition_witness_refs": [],
                "effects": [{
                    "predicate": "entity.discovered_at",
                    "args": concrete_args,
                    "effect_domain": "evidence",
                }],
                "effect_witness_refs": [str(fact["witness_ref"])],
                "rationale": "The accepted search produced the supplied discovery fact.",
            })
        elif action_type == "TAKE":
            fact = next(
                item for item in candidates if item.get("predicate") == "agent.holds"
            )
            target = str(arguments["object"])
            location = str(arguments["location"])
            occurrences.append({
                "phase_id": f"take_{index}",
                "intent": "take target object",
                "event_start": index,
                "event_end": index + 1,
                "support_event_ids": [event_id],
                "input_roles": {"object": target, "location": location},
                "input_provenance_refs": {
                    "object": f"action_arg:{event_id}:object",
                    "location": f"action_arg:{event_id}:location",
                },
                "output_roles": {"held_object": target},
                "output_derivations": {
                    "held_object": {"kind": "input_identity", "input_role": "object"},
                },
                "preconditions": [],
                "precondition_witness_refs": [],
                "effects": [{
                    "predicate": "agent.holds",
                    "args": dict(fact["args"]),
                    "effect_domain": "world",
                }],
                "effect_witness_refs": [str(fact["witness_ref"])],
                "rationale": "The accepted take produced the supplied possession fact.",
            })
    if not occurrences:
        raise AssertionError("Gate36 Extractor received no accepted causal action")
    return {"occurrences": occurrences}


def _e2_reply(request: FakeProviderRequest) -> dict[str, Any]:
    candidates = request.policy_context["new_edge_candidates"]
    return {
        "selected_existing_edge_ids": [],
        "selected_new_edge_candidate_ids": [
            str(item["candidate_id"])
            for item in candidates
            if item["edge_type"] == "data_flow"
        ],
        "summary": "locate and take a target object",
        "guideline": {
            "ordering": "locate before taking when the location is unresolved",
        },
        "insight": {"source": "gate36_authoritative_actions"},
    }


def _providers() -> tuple[dict[str, ScriptedAgentProvider], dict[str, ScriptedAgentProvider]]:
    providers = {
        name: ScriptedAgentProvider(provider_id=name)
        for name in (
            "planner", "runtime_preparation", "runtime_seeded",
            "runtime_dynamic", "extractor", "tool_builder", "default",
        )
    }
    # Bootstrap uses one Dynamic action.  Task A creates and trials its runtime
    # self-tool, then finishes the stored take node.  Task B invokes the learned
    # support Atomic for a different entity and finishes the same stored node.
    providers["runtime_dynamic"].enqueue(
        FakeReply.tool("environment_action", {"action_id": "r000_a002"}),
        FakeReply.tool("environment_action", {"action_id": "r001_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r002_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r002_a001"}),
    )
    providers["runtime_preparation"].enqueue(
        FakeReply.tool("propose_runtime_automation_atomic", _locate_draft_call()),
        FakeReply.tool("environment_action", {
            "action_id": "r001_a002", "intent": "attempt_current_atomic",
        }),
        # Task B's support ref is not known until Task A is persisted; this
        # provider is extended at that explicit phase boundary below.
    )
    # Bootstrap evolution, Task A runtime ToolBuilder, and Task A evolution.
    providers["tool_builder"].enqueue(
        _RequestAwareToolReply(),
        _RequestAwareToolReply(),
        _RequestAwareToolReply(),
    )
    # Bootstrap and Task A each perform E1 then E2.  Task B may also be
    # extraction-eligible; exact executable reuse guarantees no target builder.
    providers["extractor"].enqueue(
        FakeReply.structured(_e1_reply), FakeReply.structured(_e2_reply),
        FakeReply.structured(_e1_reply), FakeReply.structured(_e2_reply),
        FakeReply.structured(_e1_reply), FakeReply.structured(_e2_reply),
    )
    return providers, providers


def _find_locate_asset(system: AtomicSkillGraphSystem) -> tuple[str, str, str]:
    matches = []
    for raw_ref in system.skills.list_refs("atomic"):
        atomic = system.skills.get_atomic(raw_ref)
        if [item.predicate for item in atomic.effects] == ["entity.discovered_at"]:
            matches.append(atomic)
    assert len(matches) == 1
    atomic = matches[0]
    implementations = system.skills.implementations_for(
        atomic.ref, mode=RuntimeMode.ONLINE,
    )
    assert len(implementations) == 1
    implementation = implementations[0]
    assert len(implementation.tool_bindings) == 1
    tool_ref = str(implementation.tool_bindings[0].tool_ref)
    return str(atomic.ref), str(implementation.ref), tool_ref


def test_gate36_full_system_runtime_self_tool_persists_and_reuses(tmp_path: Path) -> None:
    providers, injected = _providers()
    harness = Gate36Harness()
    with AtomicSkillGraphSystem(
        _config(tmp_path), harness=harness, provider=injected,
    ) as system:
        # Bootstrap: run_task alone evolves the first reusable take graph into
        # the previously empty bank.
        assert system.skills.list_refs("atomic") == []
        bootstrap = system.run_task(
            _locating_task("gate36_bootstrap", "orange_1")
        )
        assert bootstrap.benchmark_success is True
        assert bootstrap.runtime_plan["source"] == "full_dynamic"
        assert bootstrap.metadata["extraction"]["applied"] is True

        # Task A: runtime proposes a self-tool, R0/R1 execute it, and the normal
        # success-evolution path persists its cross-task Atomic/Tool/Impl.
        task_a = _locating_task("gate36_task_a", "apple_1")
        trace_a = system.run_task(task_a)
        assert trace_a.benchmark_success is True
        assert trace_a.runtime_plan["source"] == "stored_composite"
        assert trace_a.metadata["semantic_authority_source"] == (
            "validator_snapshot_v3_2"
        )
        draft = dict(trace_a.metadata["runtime_automation_drafts"])["locate_1"]
        assert draft["r0_passed"] is True
        assert draft["r1_passed"] is True
        assert draft["trial"]["r1_outputs"] == {
            "entity": "apple_1",
            "location": _room_for("apple_1"),
        }
        assert trace_a.metadata["extraction"]["applied"] is True
        locate_atomic_ref, locate_impl_ref, locate_tool_ref = _find_locate_asset(system)
        assert system.skills.get_atomic(locate_atomic_ref).status is SkillStatus.CANDIDATE
        assert system.skills.get_implementation(locate_impl_ref).status is SkillStatus.CANDIDATE
        learned_locate_tool = system.tools.get(locate_tool_ref)
        assert learned_locate_tool.status is ToolStatus.CANDIDATE
        assert learned_locate_tool.safety["zero_llm"] is True
        assert learned_locate_tool.provenance["source"] == "success_evolution"
        assert learned_locate_tool.provenance["source_trace_id"] == trace_a.trace_id

        # Task B: same persisted bank, different task/entity.  The only write
        # entry point remains run_task; the agent selects the production support
        # candidate and its learned zero-LLM Tool supplies the missing location.
        task_b = _locating_task(
            "gate36_task_b", "mug_1", expose_object=False,
        )
        providers["runtime_preparation"].enqueue(
            FakeReply.tool("invoke_support_atomic", {
                "support_atomic_ref": locate_atomic_ref,
                "arguments": {"target": "mug_1"},
                "output_mapping": {"location": "location"},
            }),
            FakeReply.tool("environment_action", {
                "action_id": "r001_a002", "intent": "attempt_current_atomic",
            }),
        )
        builder_request_start = len(providers["tool_builder"].requests)
        trace_b = system.run_task(task_b)
        assert task_b.task_id != task_a.task_id
        assert task_b.context["target_item"] != task_a.context["target_item"]
        assert trace_b.benchmark_success is True

        augmentations = list(trace_b.metadata.get("runtime_graph_augmentation") or [])
        assert any(
            item["support_atomic_ref"] == locate_atomic_ref
            and item["output_mapping"] == {"location": "location"}
            for item in augmentations
        )
        target_executions = [
            item for item in trace_b.tool_executions
            if str(item.tool_ref) == locate_tool_ref
        ]
        assert len(target_executions) == 1
        assert target_executions[0].result["executed_step_count"] == 1
        assert target_executions[0].result["atomic_effect_passed"] is True
        target_search = next(
            action for action in trace_b.environment_actions
            if action.action_type == "SEARCH"
            and action.arguments == {"target": "mug_1"}
        )
        assert target_search.span_id == target_executions[0].span_id
        # There is no nested Agent session for the support occurrence: the
        # SEARCH transition is executed wholly inside the persisted zero-LLM
        # Tool after the outer Runtime Agent selected invoke_support_atomic.
        assert not any(
            session.occurrence_id == target_executions[0].occurrence_id
            for session in trace_b.agent_sessions
        )
        assert any(
            change.get("reason") == "validated_output_published"
            and change.get("role") == "location"
            and dict(change.get("current") or {}).get("value") == _room_for("mug_1")
            for change in trace_b.binding_changes
        )
        current_occurrence_id = str(trace_b.node_records[0].occurrence_id)
        downstream_span = next(
            span for span in trace_b.runtime_spans
            if span.kind == "runtime_preparation"
            and span.occurrence_id == current_occurrence_id
        )
        downstream_take = next(
            action for action in trace_b.environment_actions
            if action.action_type == "TAKE"
        )
        assert downstream_take.span_id == downstream_span.span_id
        assert downstream_take.arguments == {
            "object": "mug_1",
            "location": _room_for("mug_1"),
        }

        # This is target-specific: inspect only Task B ToolBuilder requests
        # whose canonical Atomic is the learned locate capability.
        task_b_builder_requests = providers["tool_builder"].requests[
            builder_request_start:
        ]
        target_builder_requests = [
            request for request in task_b_builder_requests
            if [
                item.get("predicate")
                for item in request.policy_context["canonical_atomic"]["effects"]
            ] == ["entity.discovered_at"]
        ]
        assert target_builder_requests == []

        # Three public calls yielded three persisted, integrity-verifiable traces.
        assert len(list(system.traces.iter_payloads())) == 3
        system.artifacts.verify_all()

    for provider in providers.values():
        provider.assert_exhausted()
