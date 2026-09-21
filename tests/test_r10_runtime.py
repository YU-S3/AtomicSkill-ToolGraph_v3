"""R10 uses ordinary typed contracts; no formal benchmark decision fixtures."""
import copy
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
from atomic_skillgraph.core.errors import AtomicSkillGraphError, BudgetExhausted
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.runtime_step import run_runtime_step
from atomic_skillgraph.runtime.composite_executor import VerifiedCompositeExecutor
from atomic_skillgraph.runtime.checkpoint import capture, restore
from atomic_skillgraph.traces.canonical import canonical_action_indices
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments import self_tooling_targeted as route
from experiments.fakes import FakeProviderRequest, FakeReply
from fixtures.r921_self_tooling_cases import fixture_config


class CheckpointHarness(route.CandidateHarness):
    profile_name = "fake_v3"
    def reset(self, task):
        result = super().reset(task)
        self._runtime_accepted_prefix = []
        return result

    def execute_action(self, action_id, revision):
        action = self._catalog.get(action_id, revision)
        result = super().execute_action(action_id, revision)
        self._runtime_accepted_prefix.append({"action_type": action.action_type,
                                              "arguments": copy.deepcopy(action.arguments)})
        return result


class StepProvider:
    def __init__(self, choose):
        self.choose = choose
        self.requests = []

    def snapshot(self):
        return {"provider": "r10_test", "fixture_generated": True}

    def complete(self, messages, *, tools):
        request = FakeProviderRequest(tuple(copy.deepcopy(messages)), tuple(tools))
        self.requests.append(request)
        name, args = self.choose(request, len(self.requests))
        return FakeReply.tool(name, args, prompt_tokens=10, completion_tokens=10, reasoning_tokens=0).materialize(
            call_id=f"step_{len(self.requests)}", tools=tools, request=request,
        )


def setup(tmp_path, choose, *, case_id="r10_step"):
    case = route.RouteCase(case_id)
    harness = CheckpointHarness(case)
    config = fixture_config(tmp_path)
    config["experiment"]["task_manifest_path"] = None
    config["runtime"]["rollback_automatic_execution_failure"] = True
    provider = StepProvider(choose)
    system = AtomicSkillGraphSystem(config, harness=harness, provider=provider)
    atomic, impls = route.install_parent(system, case)
    task = route.fixture_task(case)
    plan = route.make_plan(task, atomic, impls, harness)
    ctx = TaskRuntimeContext.create(task, plan, harness, system.orchestrator.create_trace_builder(task), RuntimeBudget())
    ctx.runtime_config = config["runtime"]
    occurrence = plan.occurrences[0]
    ctx.budget.begin_node(occurrence.occurrence_id)
    ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
    ctx.begin_occurrence(occurrence)
    system._current_task_id = task.task_id
    system._current_task_usage_start = 0
    invocations = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=task.task_id)
    return system, ctx, occurrence, invocations, provider


def action(request, action_type, **arguments):
    catalog = route.newest_catalog(request)
    if not catalog:
        catalog = request.policy_context.get("current_action_catalog", {})
        catalog = catalog.get("actions", []) if isinstance(catalog, dict) else catalog
    assert catalog, request.policy_context
    matches = [a for a in catalog if a["action_type"] == action_type
               and all(a["arguments"].get(k) == v for k, v in arguments.items())]
    assert matches, request.policy_context
    spec = matches[0]
    return "environment_action", {"action_id": spec["action_id"], "intent": "explore"}


def test_steps_are_fresh_and_lazy_and_charge_shared_task_budget(tmp_path):
    system, ctx, occurrence, invocations, provider = setup(
        tmp_path, lambda request, count: action(request, "GO_TO", destination="cabinet_1"))
    ex = system.orchestrator.node_executor
    for _ in range(2):
        run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    assert len(provider.requests) == 2
    assert all([m["role"] for m in request.messages] == ["system", "user"] for request in provider.requests)
    assert all("request_runtime_automation" in {t.name for t in request.tools} for request in provider.requests)
    assert all("propose_runtime_automation_atomic" not in {t.name for t in request.tools} for request in provider.requests)
    system.config["llm"]["runtime"]["max_total_tokens_per_task"] = 40
    with pytest.raises(AtomicSkillGraphError) as error:
        run_runtime_step(ex, "seeded", occurrence, ctx, invocations, [])
    assert error.value.code == "runtime_task_token_budget_exhausted"
    assert len(provider.requests) == 2


def test_checkpoint_restores_world_and_logic_not_usage_or_trace(tmp_path):
    system, ctx, occurrence, invocations, provider = setup(
        tmp_path, lambda request, count: action(request, "GO_TO", destination="cabinet_1"))
    before = ctx.harness._runtime_state_digest()
    saved = capture(ctx, occurrence.occurrence_id)
    run_runtime_step(system.orchestrator.node_executor, "preparation", occurrence, ctx, invocations, [])
    assert ctx.harness._runtime_state_digest() != before
    restore(ctx, saved, "test_failure")
    assert ctx.harness._runtime_state_digest() == before
    assert ctx.world_revision == 0
    assert len(ctx.trace_builder.trace.environment_actions) == 1
    assert canonical_action_indices(ctx.trace_builder.trace) == []
    assert ctx.budget.used_global_actions == 1
    assert len(system.usage.events) == 1


def test_automatic_partial_tool_budget_exhaustion_restores_checkpoint(tmp_path, monkeypatch):
    from experiments.r10_world_checks import install_fixture, bind
    from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
    from atomic_skillgraph.core.contracts import SemanticPredicate
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from atomic_skillgraph.runtime import checkpoint as checkpoints
    from atomic_skillgraph.core.serialization import to_primitive

    system, ctx, _, _, provider = setup(tmp_path, lambda *args: pytest.fail("automatic Tool must not call policy"))
    expr = BindingExpression(BindingExprKind.SKILL_INPUT, source_role="target")
    opened = SemanticPredicate("container.open", {"container": expr})
    atomic, impl = install_fixture(system, "budget_partial", [], [opened],
                                  [("GO_TO", "destination"), ("OPEN", "object")], source_target="cabinet_1")
    occurrence = RuntimeOccurrence("budget_partial", "budget_partial", atomic.ref, [], {},
                                   [str(impl.ref)], atomic.effects)
    ctx.begin_occurrence(occurrence)
    ctx.budget.begin_node(occurrence.occurrence_id)
    bind(ctx, occurrence, "cabinet_1")
    invocations = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store,
                                                               task_id=ctx.task.task_id)
    ctx.budget.global_action_budget = 1
    usage_before = tuple(system.usage.events)
    initial_digest = ctx.harness._runtime_state_digest()
    saved = []
    def record_capture(*args, **kwargs):
        value = capture(*args, **kwargs)
        saved.append(value)
        return value
    monkeypatch.setattr(checkpoints, "capture", record_capture)

    with pytest.raises(BudgetExhausted) as error:
        system.orchestrator.node_executor.try_autonomous(occurrence, invocations, ctx)

    expected_code = "episode_action_budget_exhausted"
    assert error.value.code == expected_code
    assert len(saved) == 1
    assert ctx.harness._runtime_state_digest() == initial_digest
    checkpoint = saved[0]
    for name, value in checkpoint.logical.items():
        assert to_primitive(getattr(ctx, name)) == to_primitive(value), name
    for store, state in ((ctx.binding_store, checkpoint.binding_state), (ctx.evidence_store, checkpoint.evidence_state)):
        assert {k: v for k, v in vars(store).items() if not callable(v)} == state
    trace = ctx.trace_builder.trace
    assert len(trace.environment_actions) == 1
    assert canonical_action_indices(trace) == []
    assert ctx.budget.used_global_actions == ctx.budget.used_node_actions == 1
    assert tuple(system.usage.events) == usage_before
    assert not provider.requests
    assert trace.metadata["r10_metrics"]["runtime_rollback_count"] == 1
    assert trace.metadata["r10_metrics"]["llm_free_environment_action_count"] == 1
    assert trace.metadata["runtime_rollbacks"][0]["failure_code"] == expected_code
    ctx.trace_builder.finish()
    assert not ctx.trace_builder._open_spans
    assert trace.runtime_spans
    assert all(span.action_end >= span.action_start for span in trace.runtime_spans)




@pytest.mark.parametrize("mode", ["preparation", "seeded"])
@pytest.mark.parametrize("rejected", [False, True])
def test_O1_O2_explore_retains_owner_before_cache_and_feedback(tmp_path, monkeypatch, mode, rejected):
    system, ctx, occurrence, invocations, provider = setup(
        tmp_path, lambda request, count: action(request, "GO_TO", destination="cabinet_1"))
    ex = system.orchestrator.node_executor
    if rejected:
        monkeypatch.setattr(ex, "_execute_environment_call", lambda *a, **kw: (
            {"accepted": False, "failure_code": "fixture_affordance_rejected"}, None))
    first = run_runtime_step(ex, mode, occurrence, ctx, invocations, [])
    second = run_runtime_step(ex, mode, occurrence, ctx, invocations, [])
    assert first.result is second.result is None
    assert len(provider.requests) == 2
    assert not any(record.result['started'] for record in ctx.trace_builder.trace.implementation_invocations)
    assert all(row["accepted_semantic_turn_count"] == 1 for row in ctx.trace_builder.trace.metadata["runtime_steps"])
    if rejected:
        assert "fixture_affordance_rejected" in str(provider.requests[1].policy_context)
        assert not ctx.trace_builder.trace.environment_actions


@pytest.mark.parametrize("handoff", ["validate_current_atomic", "invoke_support_atomic", "implementation", "request_runtime_automation", "attempt_current_atomic"])
def test_explicit_failed_calls_return_to_same_agent(tmp_path, monkeypatch, handoff):
    from atomic_skillgraph.agents.protocol import NativeToolSpec
    def choose(request, count):
        if handoff == "implementation":
            return next(t.name for t in request.tools if t.name.startswith("invoke_impl_")), {"object": "apple_1", "source": "cabinet_1"}
        if handoff == "attempt_current_atomic":
            name, args = action(request, "GO_TO", destination="cabinet_1")
            return name, {**args, "intent": "attempt_current_atomic"}
        if handoff == "validate_current_atomic":
            return handoff, {"candidate_bindings": {}}
        return handoff, ({"reason": "test", "intended_capability": "test"} if handoff == "request_runtime_automation" else {})
    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose)
    ex = system.orchestrator.node_executor
    selected = []
    if handoff == "invoke_support_atomic":
        original_tools = ex._node_tools
        monkeypatch.setattr(ex, "_node_tools", lambda *a, **kw: [
            *[t for t in original_tools(*a, **kw) if t.name != handoff],
            NativeToolSpec(handoff, "fixture dispatch seam", {"type": "object", "properties": {}})])
        monkeypatch.setattr(ex, "_invoke_support_atomic_call", lambda *a, **kw: selected.append(True) or {"accepted": False})
    result = run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    assert result.result is None
    assert not any(record.result['started'] for record in ctx.trace_builder.trace.implementation_invocations)
    if handoff == "invoke_support_atomic":
        assert selected == [True]  # Selected route still reaches the normal handler.








def test_typed_rejection_survives_fresh_steps_but_not_state_changes(tmp_path, monkeypatch):
    def choose(request, count):
        name = next(tool.name for tool in request.tools if tool.name.startswith("invoke_impl_"))
        return name, {"object": "apple_1", "source": "cabinet_1"}
    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose)
    compiler = system.invocation_compiler
    original = compiler.prepare_arguments
    attempts = []
    def prepare(*args, **kwargs):
        if kwargs["call_id"] != "autonomous":
            attempts.append(kwargs["arguments"])
        return original(*args, **kwargs)
    monkeypatch.setattr(compiler, "prepare_arguments", prepare)
    ex = system.orchestrator.node_executor
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    run_runtime_step(ex, "seeded", occurrence, ctx, invocations, [])
    assert len(provider.requests) == 2 and len(attempts) == 1
    assert not ctx.trace_builder.trace.environment_actions
    assert ctx.rejected_runtime_candidates
    entry = next(iter(ctx.rejected_runtime_candidates.values()))
    assert entry["candidate_group"][0]["failure_code"] == "runtime_semantic_anchor_mismatch"
    assert "apple_1" in str(provider.requests[-1].policy_context["rejected_candidates"])
    from atomic_skillgraph.runtime.negative_memory import current_rejections
    saved = capture(ctx, occurrence.occurrence_id)
    # A revision change expires the exact-state exclusion, not a permanent
    # semantic blacklist. Rollback restores its applicability without erasing it.
    ctx.world_revision += 1
    assert not current_rejections(ctx, occurrence)
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    assert len(attempts) == 2
    # A later revision's entry must not erase the earlier exact-state memory.
    restore(ctx, saved, "fixture")
    assert ctx.rejected_runtime_candidates
    assert current_rejections(ctx, occurrence)


def test_rejection_cache_includes_repeat_state_and_exact_argument_group(tmp_path):
    from atomic_skillgraph.runtime.negative_memory import remember_rejection, cached_rejection
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *args: pytest.fail("no model"))
    call = NativeToolCall("negative", "validate_current_atomic", {"candidate_bindings": {"object": "egg_1"}})
    remember_rejection(ctx, occurrence, call, {"passed": False, "validation": {
        "passed": False, "failure_code": "runtime_repetition_distinctness_violation"}})
    assert cached_rejection(ctx, occurrence, call)
    other = NativeToolCall("other", "validate_current_atomic", {"candidate_bindings": {"object": "egg_2"}})
    assert cached_rejection(ctx, occurrence, other) is None
    ctx.binding_store.repeat_state.committed_distinct_values["test_repeat"] = {0: "egg_2"}
    assert cached_rejection(ctx, occurrence, call) is None


def test_support_identity_requires_bound_consumer_but_discovery_accepts_anchor(tmp_path):
    from atomic_skillgraph.core.contracts import ParameterSpec
    from atomic_skillgraph.core.bindings import BindingResolution
    from atomic_skillgraph.runtime.support_request import mapped_support_bindings
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *args: pytest.fail("no Agent"))
    producer = SimpleNamespace(inputs=[ParameterSpec("destination", "entity", required_resolution="concrete")])
    # A location input absent from the parent must never be grounded from
    # whatever unrelated PUT/GO_TO affordance the helper happens to expose.
    assert mapped_support_bindings(producer, {"destination": "source"}, set(), occurrence, ctx.binding_store) is None
    producer.inputs = [ParameterSpec("target_family", "entity", required_resolution="semantic")]
    values = mapped_support_bindings(producer, {"target_family": "object"}, {"target_family"}, occurrence, ctx.binding_store)
    assert values["target_family"].value == "egg"
    assert values["target_family"].resolution == BindingResolution.SEMANTIC
    producer.inputs = [ParameterSpec("target_family", "entity", required_resolution="concrete")]
    assert mapped_support_bindings(producer, {"target_family": "object"}, {"target_family"}, occurrence, ctx.binding_store) is None


def test_binding_support_not_rejected_by_unselected_predicate_route():
    from dataclasses import replace
    from test_r92_support_and_public_memory import _support_call_fixture, _support_call
    executor, runner, ctx, session, parent, blocked, candidate = _support_call_fixture(
        executable=True, preflight_passed=False)
    candidate = replace(candidate, predicate_obligations=({"input_mapping": {"destination": "missing_parent_entity"}},))
    # The fixture intentionally has no parent binding lookup: the Agent chose
    # the authorized output->input route, not an input-identity predicate route.
    payload = executor._invoke_support_atomic_call(
        _support_call(candidate, arguments={"destination": "desk_2"}), session, parent, ctx, blocked, [candidate])
    assert payload["error"] == "support_not_execution_ready"
    assert runner.calls == 0


def test_sparse_atomicizer_does_not_replay_rolled_back_prefix():
    from test_v32_r61_extractor_e2_repair import _normalized_trace, _atomic_proposal
    from atomic_skillgraph.evolution.atomicizer import Atomicizer
    normalized, proposal = _normalized_trace(), _atomic_proposal()
    from fixtures.typed_e1 import declare_take_fixture
    declare_take_fixture(proposal, normalized)
    normalized["raw_action_count"] = 2
    normalized["actions"][0]["event_index"] = 1
    normalized["actions"][0]["authoritative_positive_effects"][0]["event_index"] = 1
    normalized["runtime_spans"][0].update(action_start=1, action_end=2)
    proposal.event_start = proposal.event_end = 1
    canonical = Atomicizer(legacy_source_replay=True).validate_and_canonicalize([proposal], normalized)
    assert canonical[0].prefix_events == []
    assert canonical[0].event_start == 1
    proposal.event_start = 0
    with pytest.raises(ValueError, match="rolled-back"):
        Atomicizer(legacy_source_replay=True).validate_and_canonicalize([proposal], normalized)


def test_negative_replay_retains_failure_time_world_not_final_success_branch():
    from atomic_skillgraph.traces.canonical import failure_prefix_action_indices
    trace = {"environment_actions": [{}, {}, {}, {}, {}], "metadata": {"runtime_rollbacks": [
        {"discarded_action_start": 1, "discarded_action_end": 2},
        {"discarded_action_start": 2, "discarded_action_end": 5}]}}
    assert canonical_action_indices(trace) == [0]
    assert failure_prefix_action_indices(trace, 3) == {0, 2}


def test_canonical_validator_output_keys_and_predicate_positions_are_distinct():
    from atomic_skillgraph.evolution.contract_canonicalizer import _rewrite_validator_spec
    original = {"output_derivations": {
        "object": {"kind": "effect_witness", "predicate": "entity.discovered_at", "argument_role": "entity"},
        "source": {"kind": "input_identity", "input_role": "entity"}},
        "output_semantic_constraints": {"object": {"compatible_with_input": "entity"}}}
    actual = _rewrite_validator_spec(original, {"entity": "input_000"}, {"object": "output_000", "source": "output_001"})
    assert set(actual["output_derivations"]) == {"output_000", "output_001"}
    assert actual["output_derivations"]["output_000"]["argument_role"] == "entity"
    assert actual["output_derivations"]["output_001"]["input_role"] == "input_000"
    assert actual["output_semantic_constraints"] == {"output_000": {"compatible_with_input": "input_000"}}
    assert _rewrite_validator_spec(actual, {"input_000": "input_000"}, {"output_000": "output_000", "output_001": "output_001"}) == actual


def test_semantic_tool_input_reuses_formal_anchor_without_concrete_upgrade(tmp_path):
    from dataclasses import replace
    from atomic_skillgraph.core.bindings import BindingResolution
    system, ctx, occurrence, invocations, _ = setup(tmp_path, lambda *args: pytest.fail("no Agent"))
    compiled = copy.deepcopy(invocations[0])
    compiled.atomic.inputs = [replace(compiled.atomic.inputs[0], required_resolution="semantic")]
    compiled.spec.input_schema = {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"], "additionalProperties": False}
    compiled.spec.grounding_constraints = []
    def prepare(value):
        return system.invocation_compiler.prepare_arguments(compiled, call_name=compiled.spec.name,
            call_id="semantic", arguments={"object": value}, occurrence=occurrence,
            binding_store=ctx.binding_store, evidence_store=ctx.evidence_store, revision=ctx.world_revision)
    result = prepare("egg")
    assert result.passed
    assert result.binding_updates[0].resolution is BindingResolution.SEMANTIC
    assert not prepare("apple").passed
    compiled.atomic.inputs[0].required_resolution = "concrete"
    assert not prepare("egg").passed


@pytest.mark.parametrize("source", ["stored_composite", "atomic_composition"])
def test_graph_bootstrap_then_two_unassisted_nodes_same_executor(tmp_path, source):
    from dataclasses import replace
    from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
    from atomic_skillgraph.core.contracts import SemanticPredicate
    from atomic_skillgraph.core.results import RuntimeOccurrence
    from experiments.r10_world_checks import install_fixture
    system, ctx, occurrence, _, provider = setup(tmp_path,
        lambda request, count: ("environment_action", {**action(request, "GO_TO", destination="cabinet_1")[1], "intent": "attempt_current_atomic", "candidate_bindings": {"target": "cabinet_1"}}) if count == 1 else pytest.fail(str(request.policy_context.get("current_state_snapshot"))))
    system.harness.case = replace(system.harness.case, terminal_action="TAKE")
    expr = BindingExpression(BindingExprKind.SKILL_INPUT, source_role="target")
    at = SemanticPredicate("agent.at_location", {"location": expr})
    opened = SemanticPredicate("container.open", {"container": expr})
    go, go_impl = install_fixture(system, "generic_go", [], [at], [("GO_TO", "destination")], source_target="cabinet_1")
    op, op_impl = install_fixture(system, "generic_open", [at], [opened], [("OPEN", "object")], source_target="cabinet_1")
    def node(name, atomic, implementation):
        return RuntimeOccurrence(name, name, atomic.ref, [], {"target": BindingExpression(
            BindingExprKind.CONSTANT, constant="cabinet_1")}, [implementation.ref], atomic.effects)
    plan = copy.deepcopy(ctx.plan)
    plan.source = source
    occurrence = copy.deepcopy(occurrence)
    occurrence.binding_specs = {role: BindingExpression(BindingExprKind.CONSTANT, constant=value)
                                for role, value in {"object": "egg_1", "source": "cabinet_1"}.items()}
    plan.occurrences = [node("go", go, go_impl), node("open", op, op_impl), occurrence]
    plan.control_sequence = [item.step_id for item in plan.occurrences]
    system.planner.build_plan = lambda *args, **kwargs: plan
    trace = system.orchestrator.run_task(ctx.task)
    from atomic_skillgraph.runtime.r10_metrics import finalize
    finalize(trace, system.config)
    assert trace.benchmark_success
    assert len(provider.requests) == 1
    assert [a.action_type for a in trace.environment_actions] == ["GO_TO", "OPEN", "TAKE"]
    assert trace.node_records[-1].direct_result["validated_outputs"] == {'object': 'egg_1'}
    assert trace.node_records[-1].direct_result['completed']  # Original RETURN tail is executed.
    assert trace.metadata["r10_metrics"]["post_bootstrap_llm_free_node_count"] == 2, [(n.occurrence_id, str(n.status), n.direct_result.get("atomic_effect_passed"), n.direct_result.get("failure_code")) for n in trace.node_records]


def test_canonical_learning_keeps_original_event_coordinates(tmp_path):
    from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
    system, ctx, occurrence, invocations, provider = setup(
        tmp_path, lambda request, count: action(request, "GO_TO", destination="cabinet_1"))
    saved = capture(ctx, occurrence.occurrence_id)
    ex = system.orchestrator.node_executor
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    restore(ctx, saved, "rejected_program")
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    trace = ctx.trace_builder.trace
    normalized = TraceNormalizer().build(trace)
    assert len(trace.environment_actions) == 2
    assert [item["event_index"] for item in normalized["actions"]] == [1]
    assert normalized["actions"][0]["extractor_event_start"] == 1
    assert normalized["raw_action_count"] == 2


def test_terminal_world_is_never_rolled_back(tmp_path):
    system, ctx, occurrence, invocations, provider = setup(
        tmp_path, lambda request, count: action(request, "GO_TO", destination="cabinet_1"))
    saved = capture(ctx, occurrence.occurrence_id)
    run_runtime_step(system.orchestrator.node_executor, "preparation", occurrence, ctx, invocations, [])
    ctx.terminal_latched = True
    restore(ctx, saved, "interrupted")
    assert ctx.world_revision == 1
    assert not ctx.trace_builder.trace.metadata.get("runtime_rollbacks")




def staged_observation(tmp_path, case_id="r10_step", *, repair_revision=None):
    from atomic_skillgraph.evolution.runtime_support_promotion import collect_observations
    from atomic_skillgraph.knowledge.runtime_support_store import RuntimeSupportStore
    case = route.RouteCase(case_id)
    def choose(request, count):
        if "propose_runtime_automation_atomic" in {tool.name for tool in request.tools}:
            return "propose_runtime_automation_atomic", route.fixed_draft(request, case)
        return "request_runtime_automation", {"reason": "systematic public search", "intended_capability": "locate target"}
    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose, case_id=case_id)
    if repair_revision:
        ctx.trace_builder.trace.metadata["repair_revision"] = repair_revision
    ctx.runtime_config["persistent_runtime_support_promotion"] = True
    system.runtime_support_store = RuntimeSupportStore(system.database)
    system._provider_override = {"runtime_preparation": provider, "runtime_seeded": provider,
        "tool_builder": route.RouteProvider(case, "tool_builder", [])}
    executor = system.orchestrator.node_executor
    step = run_runtime_step(executor, "preparation", occurrence, ctx, invocations, [], bootstrap=True)
    assert not ctx.runtime_tool_trials
    draft_step = run_runtime_step(executor, "preparation", occurrence, ctx, invocations, [], draft_request=step.automation_request)
    assert len(ctx.runtime_tool_trials) == 1
    trial = next(iter(ctx.runtime_tool_trials.values()))
    assert trial["r1"]["admission_eligible"], trial
    assert "promotion_bundle" in trial
    assert not trial["parent_completed_after_trial"]
    provider.choose = lambda request, count: (next(t.name for t in request.tools if t.name.startswith("invoke_impl_")), {"object": "egg_1", "source": "cabinet_1"})
    result = executor.run_agent_node(occurrence, ctx)
    assert result.atomic_effect_passed
    assert trial["parent_completed_after_trial"]
    system.orchestrator._persist_v32_task_local_assets(ctx)
    trace = ctx.trace_builder.trace
    trace.learning_eligible = False
    assert collect_observations(system, trace) == []
    trace.learning_eligible = trace.benchmark_success = trace.task_contract_success = True
    observations = collect_observations(system, trace)
    assert len(observations) == 1
    assert system.runtime_support_store.append(observations[0])
    assert not system.runtime_support_store.append(observations[0])
    assert len(system.runtime_support_store.observations(observations[0]["contract_signature"])) == 1
    return system, ctx, observations[0]


def test_lazy_draft_r1_collects_only_after_parent_and_strict_success(tmp_path):
    from atomic_skillgraph.evolution.runtime_support_promotion import collect_observations
    system, ctx, observation = staged_observation(tmp_path)
    system.readonly = True
    assert collect_observations(system, ctx.trace_builder.trace) == []


def test_support_observation_uses_benchmark_success_not_contract_diagnostic(tmp_path):
    from atomic_skillgraph.evolution.runtime_support_promotion import collect_observations
    system, ctx, _ = staged_observation(tmp_path)
    trace = ctx.trace_builder.trace
    trace.benchmark_success = trace.strict_task_success = trace.learning_eligible = True
    trace.infrastructure_failure = False
    trace.task_contract_success = False
    observations = collect_observations(system, trace)
    assert len(observations) == 1
    assert observations[0]["trial"]["r1"]["admission_eligible"]
    assert observations[0]["trial"]["parent_completed_after_trial"]
    assert observations[0]["bundle"]
    assert trace.task_contract_success is False  # Preserve the diagnostic.
    for field, denied_value in (("benchmark_success", False), ("learning_eligible", False),
                                ("infrastructure_failure", True)):
        before = getattr(trace, field)
        setattr(trace, field, denied_value)
        assert collect_observations(system, trace) == []
        setattr(trace, field, before)


def test_two_source_support_promotion_uses_cross_case_replay(tmp_path):
    from atomic_skillgraph.evolution.runtime_support_promotion import prepare_and_apply
    first, ctx1, obs1 = staged_observation(tmp_path / "first", "independent_1")
    second, ctx2, obs2 = staged_observation(tmp_path / "second", "independent_2")
    assert obs1["contract_signature"] == obs2["contract_signature"]
    assert not prepare_and_apply(first, ctx1.trace_builder.trace, ctx1.task, [obs1])
    first.traces.save_atomic(ctx1.trace_builder.trace)
    first.traces.save_atomic(ctx2.trace_builder.trace)
    events = prepare_and_apply(first, ctx2.trace_builder.trace, ctx2.task, [obs2])
    assert events, ctx2.trace_builder.trace.metadata
    promotions = ctx2.trace_builder.trace.metadata["runtime_support_promotions"]
    assert len(promotions) == 1
    assert len(ctx2.trace_builder.trace.metadata["tool_replay_results"]) == 2
    assert all(item["passed"] for item in ctx2.trace_builder.trace.metadata["tool_replay_results"])
    atomic_ref, implementation_ref, tool_ref = promotions[0]["refs"]
    assert first.skills.get_atomic(atomic_ref).status.value == "candidate"
    assert first.skills.get_implementation(implementation_ref).status.value == "candidate"
    assert first.tools.get(tool_ref).status.value == "candidate"


def test_r10_formal_configs_registered_and_guarded():
    from pathlib import Path
    from atomic_skillgraph.system import load_config
    from experiments.run_v3_train import _train_protocol, _validate_formal_config as train_guard
    from experiments.run_v3_frozen_eval import _frozen_protocol, _validate_formal_config as eval_guard
    root = Path(__file__).resolve().parents[1]
    for name, protocol, guard, expected in (
        ("alfworld_train_full_120_r10_seed42", _train_protocol, train_guard, ("r10_full120", 42, 20, 120)),
        ("alfworld_frozen_eval_134_r10_seed42", _frozen_protocol, eval_guard, ("r10_frozen134", 42, 0, 134)),
    ):
        config = load_config(root / "configs" / (name + ".yaml"))
        assert protocol(config) == expected
        guard(config, root / config["experiment"]["output_dir"])


def test_cross_case_failure_keeps_staging_without_assets(tmp_path, monkeypatch):
    from atomic_skillgraph.evolution.runtime_support_promotion import prepare_and_apply
    first, ctx1, obs1 = staged_observation(tmp_path / "first", "independent_1")
    second, ctx2, obs2 = staged_observation(tmp_path / "second", "independent_2")
    before = (len(first.skills.atomics()), len(first.skills.implementations()), len(first.tools.tools()))
    calls = []
    def failed_replay(tool, case, **kwargs):
        calls.append(case["source_task"]["task_id"])
        return case["source_task"]["task_id"] == "independent_1"
    monkeypatch.setattr(first, "_replay_case_with_source_authority", failed_replay)
    assert not prepare_and_apply(first, ctx2.trace_builder.trace, ctx2.task, [obs2])
    assert set(calls) == {"independent_1", "independent_2"}
    assert before == (len(first.skills.atomics()), len(first.skills.implementations()), len(first.tools.tools()))


def test_tool_ir_alpha_rename_preserves_local_names_and_literals():
    from atomic_skillgraph.evolution.contract_canonicalizer import _rewrite_tool_ir
    source = {"op": "BIND", "name": "target", "source": {"source": "tool_input", "field": "target"}}
    returned = {"op": "RETURN", "output_sources": {"found": {"source": "local_variable", "field": "target"}}}
    result = _rewrite_tool_ir([source, returned], {"target": "input_000"}, {"found": "output_000"})
    assert result[0]["name"] == "target"
    assert result[0]["source"]["field"] == "input_000"
    assert result[1]["output_sources"] == {"output_000": {"source": "local_variable", "field": "target"}}


def test_nested_checkpoint_keeps_successful_child_and_failed_audit(tmp_path):
    def choose(request, count):
        return action(request, "GO_TO", destination="cabinet_1") if count == 1 else action(request, "OPEN")
    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose)
    ex = system.orchestrator.node_executor
    outer = capture(ctx, occurrence.occurrence_id)
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    child_digest = ctx.harness._runtime_state_digest()
    nearest = capture(ctx, occurrence.occurrence_id)
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    restore(ctx, nearest, "parent_failed")
    assert ctx.harness._runtime_state_digest() == child_digest
    assert canonical_action_indices(ctx.trace_builder.trace) == [0]
    assert ctx.budget.used_global_actions == 2
    assert len(system.usage.events) == 2
    assert ctx.runtime_step_feedback[occurrence.occurrence_id]["tool"] == "environment_action"


def test_failed_automation_trial_rolls_back_without_refunding(tmp_path):
    case = route.RouteCase("r10_step", variant="wrong_output")
    def choose(request, count):
        if "propose_runtime_automation_atomic" in {t.name for t in request.tools}:
            return "propose_runtime_automation_atomic", route.fixed_draft(request, case)
        return "request_runtime_automation", {"reason": "search", "intended_capability": "locate"}
    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose)
    system._provider_override = {"runtime_preparation": provider,
        "tool_builder": route.RouteProvider(case, "tool_builder", [])}
    initial = ctx.harness._runtime_state_digest()
    ex = system.orchestrator.node_executor
    step = run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [])
    run_runtime_step(ex, "preparation", occurrence, ctx, invocations, [], draft_request=step.automation_request)
    trial = next(iter(ctx.runtime_tool_trials.values()))
    assert not trial["r1"]["admission_eligible"]
    assert ctx.harness._runtime_state_digest() == initial
    assert ctx.trace_builder.trace.metadata["runtime_rollbacks"]
    assert not canonical_action_indices(ctx.trace_builder.trace)
    assert ctx.budget.used_global_actions > 0




def test_runtime_support_staging_included_in_both_frozen_digests(tmp_path):
    from experiments.protocol import hash_knowledge
    system, ctx, obs = staged_observation(tmp_path)
    before_system = system.knowledge_digest()
    before_protocol = hash_knowledge(system.data_dir, database=system.database)
    changed = copy.deepcopy(obs)
    changed.update(task_id="independent_2", observation_id="different_id")
    system.runtime_support_store.append(changed)
    assert system.knowledge_digest() != before_system
    assert hash_knowledge(system.data_dir, database=system.database) != before_protocol


def test_builder_tokens_charge_task_not_occurrence(tmp_path):
    from atomic_skillgraph.agents.usage import UsageBucket
    # The existing provider ledger remains the sole allocation source.
    system, ctx, occurrence, invocations, provider = setup(tmp_path, lambda request, count: action(request, "GO_TO", destination="cabinet_1"))
    run_runtime_step(system.orchestrator.node_executor, "preparation", occurrence, ctx, invocations, [])
    event = copy.deepcopy(system.usage.events[0])
    from dataclasses import replace
    system.usage.append(replace(event, event_id="fixture_builder", bucket=UsageBucket.TOOL_BUILDER_RUNTIME, session_id="builder"))
    system.config["llm"]["runtime"].update(max_total_tokens_per_node=30, max_total_tokens_per_task=40)
    with pytest.raises(AtomicSkillGraphError) as error:
        run_runtime_step(system.orchestrator.node_executor, "seeded", occurrence, ctx, invocations, [])
    assert error.value.code == "runtime_task_token_budget_exhausted"
