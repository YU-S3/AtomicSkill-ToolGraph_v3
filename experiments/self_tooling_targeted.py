"""Isolated R9.2.1 route driver; never imported by formal experiment code.

Only provider submissions and fixture worlds are controlled here. Production
sessions, R0, static checks, execution, R1 and parent validation own results.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill, ContractSource, ParameterSpec, SemanticPredicate, TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeLinearPlan, RuntimeOccurrence
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.harness.protocol import HarnessActionResult, HarnessTask
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.tooling.proposal import ToolProvenance, tool_proposal_from_dict
from experiments.fakes import FakeProviderRequest, FakeReply
from experiments.protocol import hash_config


FIXTURE_VERSION = "r921-core-two-v1"


class FixtureSetupError(ValueError):
    """The fixed case cannot exercise its declared route."""


@dataclass(frozen=True)
class RouteCase:
    case_id: str
    route: str = "runtime_preparation"
    target: str = "egg"
    locations: tuple[str, ...] = ("countertop_1", "countertop_2", "cabinet_1")
    openable: bool = True
    absent: bool = False
    variant: str = "positive"
    stop_when: bool = False
    terminal_action: str = ""
    terminal_won: bool = True
    parent_role: str = "object"
    nested_local: bool = False
    repeat_parent: bool = False


CORE_TWO = (
    RouteCase("locate_across_empty_and_distractor", openable=False),
    RouteCase("locate_inside_openable_candidate", route="runtime_seeded", target="apple",
              locations=("desk_2", "shelf_3", "drawer_2")),
)


class CandidateHarness(AlfWorldAdapter):
    """Finite world using the unmodified ALFWorld parser/matcher/validator."""
    profile_name = "r921_controlled_alfworld_contract"

    def __init__(self, case: RouteCase):
        super().__init__(split="train")
        self.case = case
        self.timeline: list[dict[str, Any]] = []

    def initialize(self):
        return 1

    def reset(self, task):
        self._current_task = task
        self._revision = 0
        self._done = self._won = False
        self.location = ""
        self.opened = False
        self.held = False
        self._validator.reset()
        catalog = self._replace_catalog()
        self._validator.set_catalog(catalog)
        self._observation = "Public candidate locations are available. The target location is unresolved."
        return HarnessActionResult(True, self._observation, False, False, 0, catalog)

    def _replace_catalog(self):
        raw = ["go to " + location.replace("_", " ") for location in self.case.locations]
        if self.location == self.case.locations[-1] and self.case.openable and not self.opened:
            raw += ["open " + self.location.replace("_", " ")]
        if self.location == self.case.locations[1]:
            raw += ["take cup 1 from " + self.location.replace("_", " ")]
        if (self.location == self.case.locations[-1] and not self.case.absent and not self.held
                and (not self.case.openable or self.opened)):
            raw += [f"take {self.case.target} 1 from " + self.location.replace("_", " ")]
        if self.held:
            raw += [f"put {self.case.target} 1 in/on " + self.location.replace("_", " ")]
        return self._catalog.replace([] if self._done else raw, self._revision)

    def execute_action(self, action_id, revision):
        if self._done:
            raise RuntimeError("fixture environment called after terminal")
        action = self._catalog.get(action_id, revision)
        if action.action_type == "GO_TO":
            self.location = action.arguments["destination"]
        elif action.action_type == "OPEN":
            self.opened = True
        elif action.action_type == "TAKE":
            self.held = action.arguments["object"] == self.case.target + "_1"
        elif action.action_type == "PUT":
            self.held = False
            self._done = self._won = True
        else:
            raise FixtureSetupError("unexpected fixture action")
        if action.action_type == self.case.terminal_action:
            self._done, self._won = True, self.case.terminal_won
        self._revision += 1
        catalog = self._replace_catalog()
        self._observation = "Accepted " + action.display_text
        self._validator.record(action, accepted=True, revision=self._revision,
                               done=self._done, won=self._won, observation=self._observation, catalog=catalog)
        self.timeline.append({"kind": "action", "action": to_primitive(action), "after_revision": self._revision})
        return HarnessActionResult(True, self._observation, self._done, self._won, self._revision, catalog)

    def task_contract(self, task):
        return TaskContract([SemanticPredicate("agent.holds", {"object": self.case.target})],
                            source=ContractSource.ADAPTER_DERIVED, confidence=1.0,
                            validator_id="r921_fixture_action_derived")


def fixture_task(case: RouteCase):
    return HarnessTask(case.case_id, f"Find and take a {case.target}.", "r921_fixture", "targeted_locate", {
        "semantic_bindings": {case.parent_role: case.target},
        "binding_types": {case.parent_role: "entity"},
    }, {"task_signature": hash_config(to_primitive(case)), "fixture_controlled": True})


def parent_atomic(case: RouteCase):
    role = case.parent_role
    return AbstractAtomicSkill(SkillRef("r921_parent_take", "1.0.0"), "take the required object",
        [ParameterSpec(role, "entity", runtime_resolvable=True, required_resolution="concrete"),
         ParameterSpec("source", "entity", runtime_resolvable=True, required_resolution="concrete")],
        [ParameterSpec(role, "entity", required_resolution="concrete")], [],
        [SemanticPredicate("agent.holds", {"object": "$" + role})],
        {"validator_id": "harness_atomic_effect", "output_derivations": {
            role: {"kind": "input_identity", "input_role": role}}}, [], {},
        {"fixture_controlled": True}, SkillStatus.ACTIVE)


def fixed_draft(request: FakeProviderRequest, case: RouteCase):
    interface = request.policy_context["runtime_automation_interface"]
    anchors = interface["current_input_sources"]["current_occurrence_anchor"]
    anchor = next((a for a in anchors if a["source_role"] == case.parent_role), None)
    if anchor is None:
        raise FixtureSetupError("required occurrence semantic anchor is unavailable")
    draft = {
        "draft_id": case.case_id + "_locate", "intent": "locate_matching_entity",
        "inputs": [{"name": "target", "semantic_type": "entity", "required": True,
                    "runtime_resolvable": False, "required_resolution": "semantic"}],
        "outputs": [{"name": "location", "semantic_type": "entity", "required": True,
                     "runtime_resolvable": False, "required_resolution": "concrete"}],
        "preconditions": [], "effects": [{"predicate": "entity.discovered_at",
            "args": {"entity": "$target", "location": "$location"}, "effect_domain": "evidence"}],
        "rationale": "Resolve the target location through a bounded check of public candidates.",
        "source_occurrence_id": interface["source_occurrence_id"],
        "input_binding_specs": {"target": {"kind": "current_occurrence_anchor",
                                            "source_role": anchor["source_role"]}},
    }
    if case.variant == "r0_negative":
        draft["effects"].append({"predicate": "agent.at_location", "args": {"location": "$location"},
                                 "effect_domain": "world"})
    return draft


def selector(action_type, role, *, target_role="", target_source="tool_input", projection=""):
    where = {"action_type": action_type}
    if target_role:
        where.update(argument_role=role, semantic_compatible_with={
            "source": target_source, "field": target_role, "semantic_type": "entity"})
    return {"source": "action_catalog", "where": where,
            "project": {"kind": "argument", "role": projection or role}, "distinct": True}


def scripted_proposal(atomic, case: RouteCase):
    """Test transport input only. Never included in the live Builder prompt."""
    target, output = atomic["inputs"][0]["name"], atomic["outputs"][0]["name"]
    target_query = {"match": selector("TAKE", "object", target_role=target)}
    location_source = {"source": "semantic_evidence", "where": {
        "predicate": "entity.discovered_at", "argument_role": "entity",
        "semantic_compatible_with": {"source": "tool_input", "field": target, "semantic_type": "entity"}},
        "project": {"kind": "argument", "role": "location"}, "distinct": True}
    ret = {"op": "RETURN", "node_id": "return_location", "output_sources": {output: location_source}}
    body = [
        {"op": "ACTION", "node_id": "visit", "action_type": "GO_TO",
         "argument_mapping": {"destination": {"kind": "local_variable", "source_role": "candidate"}},
         "expected_effects": [{"predicate": "agent.at_location", "args": {
             "location": {"kind": "local_variable", "source_role": "candidate"}}, "effect_domain": "world"}]},
        {"op": "IF", "node_id": "open_if_available", "condition": {"match": selector(
            "OPEN", "object", target_role="candidate", target_source="local_variable")},
         "then_branch": [{"op": "ACTION", "node_id": "open", "action_type": "OPEN",
             "argument_mapping": {"object": {"kind": "local_variable", "source_role": "candidate"}},
             "expected_effects": [{"predicate": "container.open", "args": {"container": {
                 "kind": "local_variable", "source_role": "candidate"}}, "effect_domain": "world"}]}]},
    ]
    if case.nested_local:
        body.insert(1, {"op": "FOR_EACH", "node_id": "shadow_candidate", "iteration_variable": "candidate",
            "collection_source": selector("GO_TO", "destination"), "max_iterations": 1,
            "body": [{"op": "STOP_WHEN", "node_id": "inner_query", "condition": {"match": selector(
                "GO_TO", "destination", target_role="candidate", target_source="local_variable")}}]})
    if case.stop_when:
        body.append({"op": "STOP_WHEN", "node_id": "found", "condition": target_query})
    else:
        body.append({"op": "IF", "node_id": "found", "condition": target_query, "then_branch": [ret]})
    program = [{"op": "FOR_EACH", "node_id": "candidates", "iteration_variable": "candidate",
        "collection_source": selector("GO_TO", "destination"), "max_iterations": 16, "body": body},
        {**copy.deepcopy(ret), "node_id": "final_return"}]
    if case.variant == "static_negative":
        body[1]["condition"]["match"]["where"]["semantic_compatible_with"]["field"] = "not_in_scope"
    if case.variant == "wrong_output":
        for node in (ret, program[-1]):
            node["output_sources"][output] = {"source": "tool_input", "field": target}
    return {"proposal_version": "2", "entry_contract": {"conditions": [], "grounding_constraints": []}, "decision": "no_tool" if case.variant == "no_tool" else "create",
        "summary": "locate matching target", "atomic_ref": str(atomic["ref"]),
        "inputs": atomic["inputs"], "outputs": atomic["outputs"],
        "program": [] if case.variant == "no_tool" else program, "max_actions": 32,
        "final_effects": atomic["effects"], "evidence_outputs": [], "path_expectations": [],
        "rationale": "Explicit fixture NO_TOOL." if case.variant == "no_tool" else "Bounded public candidate traversal."}


def safe_messages(messages):
    # Only chat payloads; no transport headers/credentials or private reasoning.
    return [{k: copy.deepcopy(v) for k, v in message.items() if k != "reasoning_content"}
            for message in messages]


class RouteProvider:
    """Instance-only provider boundary. The real delegate receives exact input."""
    def __init__(self, case, stage, timeline, *, delegate=None, audit=None):
        self.case, self.stage, self.timeline = case, stage, timeline
        self.delegate, self.audit = delegate, audit
        self.requests = []
        self.replies = []
        self.forced = False

    @property
    def request_record_count(self):
        return int(getattr(self.delegate, "request_record_count", 0))

    def request_records_since(self, index):
        method = getattr(self.delegate, "request_records_since", None)
        return method(index) if method else ()

    def snapshot(self):
        return {"provider": "r921_targeted_boundary", "stage": self.stage,
                "fixture_generated": self.delegate is None, "request_count": len(self.requests),
                "delegate": self.delegate.snapshot() if self.delegate else None}

    def set_request_context(self, **kwargs):
        if self.delegate is not None:
            method = getattr(self.delegate, "set_request_context", None)
            if method:
                method(**kwargs)

    def complete(self, messages, *, tools=None):
        tools = tools or []
        request = FakeProviderRequest(tuple(copy.deepcopy(messages)), tuple(tools))
        payload = {"messages": safe_messages(messages), "tools": [tool.to_openai() for tool in tools]}
        self.requests.append(payload)
        self.timeline.append({"kind": "provider", "stage": self.stage, "request": len(self.requests),
                              "external": self.delegate is not None and (self.stage == "tool_builder" or self.forced)})
        if self.audit:
            self.audit(self.stage, len(self.requests), "request", {**payload, "fingerprint": hash_config(payload)})
        names = {tool.name for tool in tools}
        if self.stage == "tool_builder" and self.delegate is None:
            atomic = {**request.policy_context["canonical_atomic"], "ref": request.policy_context["atomic_ref"]}
            reply = FakeReply.tool("create_tool", scripted_proposal(atomic, self.case),
                                   prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)
        elif self.stage != "tool_builder" and not self.forced:
            if "request_runtime_automation" in names:
                reply = FakeReply.tool("request_runtime_automation", {
                    "reason": "Explicit targeted self-tooling route fixture",
                    "intended_capability": "Find the supplied target using public observations"},
                    prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)
            elif "propose_runtime_automation_atomic" in names:
                self.forced = True
                reply = FakeReply.tool("propose_runtime_automation_atomic", fixed_draft(request, self.case),
                                       prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)
            else:
                raise FixtureSetupError("runtime automation request/draft interface unavailable")
        elif self.delegate is None:
            catalog = newest_catalog(request)
            target_actions = [a for a in catalog if a.get("action_type") == "TAKE"
                              and str(a.get("arguments", {}).get("object", "")).rsplit("_", 1)[0] == self.case.target]
            if self.case.variant != "positive" or not target_actions:
                reply = FakeReply.tool("report_runtime_status", {"status": "cannot_resolve",
                    "detail": "Targeted fixture ends after the observed route outcome."},
                    prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)
            else:
                reply = FakeReply.tool("environment_action", {"action_id": target_actions[0]["action_id"],
                    "intent": "attempt_current_atomic", "candidate_bindings": {
                        self.case.parent_role: target_actions[0]["arguments"]["object"],
                        "source": target_actions[0]["arguments"]["source"]},
                    "candidate_outputs": {self.case.parent_role: target_actions[0]["arguments"]["object"]}},
                    prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)
        else:
            reply = None
        if reply is not None:
            turn = reply.materialize(call_id=f"fixture_{len(self.requests)}", tools=tools, request=request)
            turn.provider_metadata = {"provider": "r921_fixture_transport", "fixture_generated": True,
                                      "external_tokens": 0, "request_id": f"fixture_{len(self.requests)}"}
            turn.latency_ms = 0
        else:
            turn = self.delegate.complete(messages=messages, tools=tools)
        record = {"tool_calls": to_primitive(turn.tool_calls), "content": turn.content,
                  "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens,
                  "total_tokens": turn.total_tokens, "reasoning_tokens": turn.reasoning_tokens,
                  "latency_ms": turn.latency_ms, "provider_metadata": to_primitive(turn.provider_metadata)}
        self.replies.append(record)
        if self.audit:
            self.audit(self.stage, len(self.requests), "reply", record)
        return turn


def newest_catalog(request):
    for message in reversed(request.messages):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        state = payload.get("current_state_snapshot", {})
        catalog = state.get("action_catalog") or payload.get("action_catalog")
        if catalog is not None:
            return catalog.get("actions", []) if isinstance(catalog, dict) else catalog
    context = request.policy_context
    catalog = (context.get("current_action_catalog")
        or context.get("current_state_snapshot", {}).get("action_catalog") or context.get("action_catalog", []))
    return catalog.get("actions", []) if isinstance(catalog, dict) else catalog


def install_parent(system, case):
    atomic = parent_atomic(case)
    if system.readonly:
        registered = system.skills.get_atomic(atomic.ref)
        if to_primitive(registered) != to_primitive(atomic):
            raise FixtureSetupError("frozen fixture parent differs from its declared contract")
        implementations = system.skills.implementations_for(atomic.ref, mode=system.mode)
        return registered, [item.ref for item in implementations]
    system.skills.register_atomic(atomic)
    if case.route != "runtime_preparation":
        return atomic, []
    role = case.parent_role
    raw = {"proposal_version": "2", "entry_contract": {"conditions": [], "grounding_constraints": []}, "decision": "create", "summary": "take object from source",
        "atomic_ref": str(atomic.ref), "inputs": to_primitive(atomic.inputs), "outputs": to_primitive(atomic.outputs),
        "program": [{"node_id": "take", "op": "ACTION", "action_type": "TAKE", "argument_mapping": {
            "object": {"kind": "skill_input", "source_role": role},
            "source": {"kind": "skill_input", "source_role": "source"}}, "expected_effects": to_primitive(atomic.effects)},
            {"node_id": "return", "op": "RETURN", "output_sources": {role: {"source": "tool_input", "field": role}}}],
        "max_actions": 1, "final_effects": to_primitive(atomic.effects), "evidence_outputs": [],
        "path_expectations": [], "rationale": "Declared fixture parent implementation; not a learned asset."}
    proposal = tool_proposal_from_dict(raw)
    static = system.tool_static_validator.validate_proposal(proposal, atomic, system.harness)
    if not static.passed:
        raise FixtureSetupError("fixture parent static rejection: " + repr(to_primitive(static)))
    canonical = CanonicalAtomicOccurrence("fixture_parent", "fixture", atomic.summary, 0, 0,
        {role: case.target + "_1", "source": case.locations[-1]}, {role: case.target + "_1"},
        atomic.inputs, atomic.outputs, [], atomic.effects, [], [], to_primitive(fixture_task(case)), "fixture", atomic.ref)
    compiled = system.tool_compiler.compile_proposal(canonical, atomic, proposal,
        ToolProvenance(source="r921_fixture", atomic_ref=str(atomic.ref),
                       source_trace_id="fixture", occurrence_id="fixture_parent"))
    compiled.implementation.status = SkillStatus.ACTIVE
    compiled.tool.status = ToolStatus.ACTIVE
    system.tools.register(compiled.tool)
    system.skills.register_implementation(compiled.implementation)
    return atomic, [compiled.implementation.ref]


def make_plan(task, atomic, implementations, harness):
    occurrence = RuntimeOccurrence("parent", "parent", atomic.ref, [], {
        atomic.inputs[0].name: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=atomic.inputs[0].name)},
        implementations, atomic.effects)
    return RuntimeLinearPlan(task.task_id, "stored_composite", "skill://r921_fixture_parent@1.0.0",
        [occurrence], [occurrence.step_id], [], [], harness.task_contract(task), {"fixture_controlled": True})


def run_node_case(system, case, *, live=False, task=None, audit=None, action_budget=None):
    """Exercise native route sessions, not the coordinator directly."""
    task = task or fixture_task(case)
    atomic, implementations = install_parent(system, case)
    plan = make_plan(task, atomic, implementations, system.harness)
    occurrence = plan.occurrences[0]
    if case.repeat_parent:
        from atomic_skillgraph.core.results import RuntimeRepeatConstraint
        plan.repeat_constraints = [RuntimeRepeatConstraint(
            "repeat_fixture", 2, ((occurrence.step_id,), ("next_parent",)), ("item_identity",), (),
            {occurrence.step_id: {"item_identity": case.parent_role},
             "next_parent": {"item_identity": case.parent_role}})]
    ctx = TaskRuntimeContext.create(task, plan, system.harness, system.orchestrator.create_trace_builder(task),
        RuntimeBudget(global_action_budget=int(system.config.get("runtime", {}).get("global_action_budget", 100)),
                      node_action_budget=action_budget or int(system.config.get("runtime", {}).get("node_action_budget", 35))))
    ctx.budget.begin_node(occurrence.occurrence_id)
    ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
    ctx.begin_occurrence(occurrence)
    timeline = getattr(system.harness, "timeline", [])
    original = system._provider_override
    runtime = RouteProvider(case, case.route, timeline, delegate=system._provider(case.route) if live else None, audit=audit)
    builder = RouteProvider(case, "tool_builder", timeline, delegate=system._provider("tool_builder") if live else None, audit=audit)
    before = system.knowledge_digest()
    system._current_task_id = task.task_id
    system._current_task_usage_start = len(system.usage.events)
    system._provider_override = {case.route: runtime, "tool_builder": builder}
    start = time.monotonic()
    try:
        executor = system.orchestrator.node_executor
        invocations = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=task.task_id)
        if case.route == "runtime_preparation":
            if not invocations or executor.try_autonomous(occurrence, invocations, ctx) is not None:
                raise FixtureSetupError("Preparation requires an implementation with genuinely unresolved input")
            result = executor.run_agent_node(occurrence, ctx, mode="preparation")
        else:
            if invocations:
                raise FixtureSetupError("Seeded fixture unexpectedly has a compatible implementation")
            result = executor.run_agent_node(occurrence, ctx, mode="seeded")
    finally:
        system._provider_override = original
        system.orchestrator._persist_v32_task_local_assets(ctx)
        if audit:
            audit("internal", 0, "trace", to_primitive(ctx.trace_builder.trace))
            audit("internal", 0, "usage", [event.to_dict() for event in system.usage.events])
    trace = ctx.trace_builder.trace
    trace.metadata["targeted_fixture"] = {"case": to_primitive(case), "is_formal_experiment": False,
                                         "test_level": "node_integration"}
    return {"ctx": ctx, "trace": trace, "result": result, "runtime": runtime, "builder": builder,
            "timeline": timeline, "elapsed_seconds": time.monotonic() - start,
            "long_term_unchanged": before == system.knowledge_digest(),
            "usage": [event.to_dict() for event in system.usage.events]}


def summarize_case(case, outcome):
    """Classify evidence, never infer a trial from fixture intent."""
    ctx = outcome["ctx"]
    drafts = list(ctx.runtime_automation_drafts.values())
    draft = drafts[0] if len(drafts) == 1 else {}
    trial = draft.get("trial") or {}
    trial = ctx.runtime_tool_trials.get(trial.get("draft_id"), trial)
    r1 = trial.get("r1") or {}
    actions = outcome["trace"].environment_actions
    start, end = int(trial.get("trial_event_start", 0)), int(trial.get("trial_event_end", -1))
    trial_actions = actions[start:end + 1] if trial else []
    accepted = sum(bool(a.accepted) for a in trial_actions)
    actual = [s.session_type for s in outcome["trace"].agent_sessions]
    expected_session = "RuntimePreparationSession" if case.route == "runtime_preparation" else "SeededSession"
    entry = bool(outcome["runtime"].forced and expected_session in actual and len(drafts) == 1)
    expected_rejection = {"r0_negative": "r0_rejected", "no_tool": "builder_no_tool",
                          "static_negative": "static_rejected", "absent": "r1_rejected",
                          "wrong_output": "r1_rejected"}.get(case.variant)
    parent = bool(outcome["result"].atomic_effect_passed)
    if expected_rejection:
        passed = entry and draft.get("stage") == expected_rejection and not draft.get("r1_passed")
    else:
        passed = bool(entry and draft.get("r1_passed") and accepted >= 2 and parent
                      and outcome["long_term_unchanged"])
    acceptance_checks = {
        "expected_native_route": entry,
        "r1_passed": bool(draft.get("r1_passed")),
        "at_least_two_trial_actions": accepted >= 2,
        "parent_completed": parent,
        "long_term_unchanged": outcome["long_term_unchanged"],
    }
    costs = summarize_usage(outcome["usage"])
    return {"case_id": case.case_id, "case_passed": passed, "expected_rejection": expected_rejection,
        "positive_acceptance_checks": acceptance_checks,
        "positive_acceptance_failures": [key for key, value in acceptance_checks.items() if not value],
        "route_entry_reached": entry, "route_last_stage": draft.get("stage", "not_reached"),
        "expected_route": expected_session, "actual_route": actual,
        "route_failure_code": draft.get("failure_code", "route_not_reached"),
        "r0_result": draft.get("r0_report", {}), "builder_decision": (draft.get("proposal") or {}).get("decision"),
        "static_result": draft.get("static_report", {}), "trial_started": bool(r1.get("started")),
        "trial_action_count": accepted, "trial_actions": to_primitive(trial_actions), "r1_result": r1,
        "r1_passed": bool(draft.get("r1_passed")), "parent_completed": parent,
        "parent_result": to_primitive(outcome["result"]),
        "terminal_reconciled": bool(outcome["result"].terminal_effect_reconciled),
        "long_term_state_unchanged_when_required": outcome["long_term_unchanged"],
        "elapsed_seconds": outcome["elapsed_seconds"], "costs_by_provider_and_bucket": costs,
        "targeted_external_tokens": sum(b["tokens"] for b in costs.values() if b["external"]),
        "fixture_tokens": sum(b["tokens"] for b in costs.values() if not b["external"]),
        "runtime_selection_forced": True, "builder_submission_scripted": outcome["builder"].delegate is None,
        "test_level": "node_integration", "is_formal_experiment": False}


def summarize_usage(events):
    costs = {}
    for event in events:
        fixture = bool(event.get("provider_metadata", {}).get("fixture_generated"))
        provider = "fixture" if fixture else str(event.get("provider", "external"))
        key = provider + ":" + event["bucket"]
        bucket = costs.setdefault(key, {"tokens": 0, "calls": 0, "latency_ms": 0, "external": not fixture})
        bucket["tokens"] += event["total_tokens"]
        bucket["calls"] += event["call_count"]
        bucket["latency_ms"] += event["latency_ms"]
    return costs


def summarize_suite(cases, *, expected_case_count=None):
    expected = len(cases) if expected_case_count is None else expected_case_count
    complete = len(cases) == expected
    return {"phase": "targeted_smoke", "is_formal_experiment": False,
        "complete": complete, "expected_case_count": expected,
        "passed": complete and bool(cases) and all(c.get("case_passed") is True for c in cases),
        "positive_trial_passes": sum(c.get("case_passed") is True and not c.get("expected_rejection") for c in cases),
        "expected_rejection_passes": sum(c.get("case_passed") is True and bool(c.get("expected_rejection")) for c in cases),
        "targeted_external_tokens": sum(c.get("targeted_external_tokens", 0) for c in cases),
        "fixture_tokens": sum(c.get("fixture_tokens", 0) for c in cases), "cases": cases}
