"""R9.2.1 native Session -> R0 -> Builder -> multi-action trial -> R1."""
import copy
import json
from dataclasses import replace

import pytest

from atomic_skillgraph.system import AtomicSkillGraphSystem
from fixtures.r921_self_tooling_cases import (
    CandidateHarness, RouteCase, RouteProvider, fixture_config, fixture_task, install_parent, run_node_case,
)


@pytest.mark.parametrize("route", ["runtime_preparation", "runtime_seeded"])
def test_shared_node_budget_exhaustion_cannot_buy_parent_continuation(tmp_path, monkeypatch, route):
    from atomic_skillgraph.core.errors import BudgetExhausted
    original = RouteProvider.complete
    seen = []
    def exhaust_after_trial(self, messages, *, tools=None):
        turn = original(self, messages, tools=tools)
        if self.stage == route:
            seen.append(self)
            if len(self.requests) == 3:
                turn.prompt_tokens = turn.total_tokens = 100001
        return turn
    monkeypatch.setattr(RouteProvider, "complete", exhaust_after_trial)
    case = RouteCase("continuation", route=route)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        with pytest.raises(BudgetExhausted) as raised:
            run_node_case(system, case)
        assert raised.value.code == "runtime_node_token_budget_exhausted"
        assert len(seen[-1].requests) == 3
        assert sum(e.to_dict()["total_tokens"] for e in system.usage.events) == 100001
        assert not system.harness.validator_channel().won


@pytest.mark.parametrize("claimed_destination,passed", [("cabinet_1", True), ("countertop_1", False)])
def test_final_effect_argument_alias_keeps_resolved_input(tmp_path, claimed_destination, passed):
    from atomic_skillgraph.core.contracts import ToolAsset
    from atomic_skillgraph.core.refs import ToolRef
    from atomic_skillgraph.runtime.tool_runner import ToolRunner
    from atomic_skillgraph.validation.tool_validator import ToolValidator
    case = RouteCase("final_alias")
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        tool = ToolAsset(ToolRef("final_alias", "1.0.0"), "navigation", {
            "type": "object", "properties": {"destination": {"type": "string"},
                "claim": {"type": "string"}, "location": {"type": "string"}},
            "required": ["destination", "claim", "location"],
        }, {"entry_contract": {"conditions": [], "grounding_constraints": []}, "output_schema": {"type": "object", "properties": {}}}, "tool_ir_v1", {
            "max_actions": 1, "program": [
                {"node_id": "move", "op": "ACTION", "action_type": "GO_TO",
                 "argument_mapping": {"destination": {"kind": "skill_input", "source_role": "destination"}}},
                {"node_id": "return", "op": "RETURN", "output_sources": {}},
            ], "final_effects": [{"predicate": "agent.at_location", "args": {"location": "$claim"},
                                    "effect_domain": "world"}],
        }, [], {"reviewed": True}, {}, {})
        bindings = {"destination": "cabinet_1", "claim": claimed_destination, "location": "egg"}
        result = ToolRunner(ToolValidator()).run(tool, bindings, outcome["ctx"], occurrence_id="parent")
        assert result.completed, result
        assert result.atomic_effect_passed is passed
        assert bindings["location"] == "egg"


@pytest.mark.parametrize("route", ["runtime_preparation", "runtime_seeded"])
@pytest.mark.parametrize("stop_when", [False, True])
def test_multicandidate_native_route_reaches_real_r1(tmp_path, route, stop_when):
    case = RouteCase("route", route=route, stop_when=stop_when)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        ctx = outcome["ctx"]
        assert len(ctx.runtime_automation_drafts) == 1, ctx.runtime_automation_drafts
        draft = next(iter(ctx.runtime_automation_drafts.values()))
        assert draft["r0_passed"], draft
        assert draft["static_passed"], __import__("json").dumps(draft, indent=2)
        assert draft["r1_passed"], __import__("json").dumps(draft, indent=2)
        assert outcome["result"].atomic_effect_passed, outcome["result"]
        assert outcome["long_term_unchanged"]
        trial = next(iter(ctx.runtime_tool_trials.values()))
        assert trial["r1_outputs"] == {"location": "cabinet_1"}
        assert trial["trial_bindings"] == {"target": "egg"}
        assert trial["parent_completed_after_trial"]
        persisted = outcome["trace"].metadata
        assert persisted["runtime_automation_drafts"][draft["draft"]["draft_id"]]["r1_passed"]
        assert persisted["runtime_tool_trials"][trial["draft_id"]]["parent_completed_after_trial"]
        assert ctx.validated_outputs.get("parent", {}).get("location") is None
        actions = outcome["trace"].environment_actions
        assert [a.action_type for a in actions] == ["GO_TO", "GO_TO", "GO_TO", "OPEN", "TAKE"]
        assert [a.new_revision for a in actions] == list(range(1, 6))
        assert ctx.budget.remaining_node_actions == 30
        builder_payload = json.loads(outcome["builder"].requests[0]["messages"][-1]["content"].split(
            "POLICY_CONTEXT_JSON\n", 1)[1])
        entry = builder_payload["harness_interface"]["runtime_entry"]
        assert entry["remaining_resources"] == {"node_actions": 35, "task_actions": 100,
            "node_tokens": system._stage_config('runtime').get('max_total_tokens_per_node', 80000),
            "task_tokens": system._stage_config('runtime').get('max_total_tokens_per_task', 300000)}
        assert entry["consumer_obligation"] == {"atomic_ref": "skill://r921_parent_take@1.0.0", "step_id": "parent", "repeat": None}
        assert {key: entry[key] for key in ("revision", "input_values", "action_catalog")} == {"revision": 0, "input_values": {"target": "egg"}, "action_catalog": [
            {"action_type": "GO_TO", "arguments": {"destination": location}} for location in case.locations]}
        assert "semantic_compatible_with" in builder_payload["tool_ir_schema"]["evidence_selector_contract"]["where"]
        assert builder_payload["semantic_delta"] == {}
        assert builder_payload["atomic_evidence_support"] == []
        assert builder_payload["harness_interface"]["public_catalog_relations"] == system.harness.public_catalog_relation_schema()
        timeline = outcome["timeline"]
        action_positions = [i for i, item in enumerate(timeline) if item["kind"] == "action"]
        assert all(item["kind"] == "action" for item in timeline[action_positions[0]:action_positions[3] + 1])
        assert trial["result"]["tool_results"][0]["tool_path_evidence"]["final_effect_result"]["passed"]
        public_reply = outcome["runtime"].requests[2]["messages"][-1]
        payload = json.loads(public_reply["content"].split("POLICY_CONTEXT_JSON\n", 1)[1])
        assert payload["current_action_catalog"]["revision"] == 4
        assert any(a["action_type"] == "TAKE" and a["arguments"]["object"] == "egg_1"
                   for a in payload["current_action_catalog"]["actions"])


@pytest.mark.parametrize("variant,stage,actions,builders", [
    ("r0_negative", "r0_rejected", 0, 0),
    ("no_tool", "builder_no_tool", 0, 1),
    ("static_negative", "static_rejected", 0, 1),
    ("absent", "r1_rejected", 4, 1),
    ("wrong_output", "r1_rejected", 4, 1),
])
def test_native_negative_route_reports_actual_rejection(tmp_path, variant, stage, actions, builders):
    case = RouteCase("negative", variant=variant, absent=variant == "absent")
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        drafts = outcome["ctx"].runtime_automation_drafts
        draft = next(iter(drafts.values()))
        assert draft["stage"] == stage
        assert not draft["r1_passed"]
        assert not outcome["result"].atomic_effect_passed
        assert len(outcome["trace"].environment_actions) == actions
        assert len(outcome["builder"].requests) == builders
        assert outcome["long_term_unchanged"]
        assert len(outcome["runtime"].requests) == 3
        assert all(event["total_tokens"] == 0 and event["provider_metadata"]["fixture_generated"]
                   for event in outcome["usage"])


def test_native_trial_keeps_original_action_budget(tmp_path):
    case = RouteCase("budget", route="runtime_seeded")
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        from atomic_skillgraph.core.errors import BudgetExhausted
        audits = {}
        with pytest.raises(BudgetExhausted) as failure:
            run_node_case(system, case, action_budget=2,
                audit=lambda stage, index, kind, payload: audits.update({kind: payload}))
        assert failure.value.code == "runtime_node_action_budget_exhausted"
        trace = audits["trace"]
        assert len(trace["environment_actions"]) == 2
        assert not any(trial.get("r1", {}).get("admission_eligible")
                       for trial in trace["metadata"].get("runtime_tool_trials", {}).values())
        assert len(audits["usage"]) == 3  # request + draft + Builder; no extra parent turn


def test_nested_loop_restores_local_before_outer_query(tmp_path):
    case = RouteCase("nested", nested_local=True, stop_when=True)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        draft = next(iter(outcome["ctx"].runtime_automation_drafts.values()))
        assert draft["r1_passed"], draft["static_report"]
        assert outcome["result"].atomic_effect_passed
        assert [a.action_type for a in outcome["trace"].environment_actions] == ["GO_TO"] * 3 + ["OPEN", "TAKE"]


def test_repeat_parent_renamed_role_is_committed_only_after_parent_action(tmp_path):
    case = RouteCase("repeat", parent_role="payload", repeat_parent=True)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        assert outcome["result"].atomic_effect_passed
        trial = next(iter(outcome["ctx"].runtime_tool_trials.values()))
        assert trial["trial_bindings"] == {"target": "egg"}
        assert trial["r1_outputs"] == {"location": "cabinet_1"}
        commits = [v for v in outcome["trace"].validations if v.level == "runtime_repeat_commit"]
        assert commits and all(v.revision == 5 for v in commits)


def test_full_system_frozen_multicandidate_route_and_next_task_isolation(tmp_path):
    from atomic_skillgraph.core.contracts import CompositeOccurrence, CompositeSkill
    from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
    from atomic_skillgraph.core.refs import SkillRef
    from atomic_skillgraph.core.status import SkillStatus

    case = RouteCase("frozen_full_system", terminal_action="TAKE")
    config = fixture_config(tmp_path)
    with AtomicSkillGraphSystem(config, harness=CandidateHarness(case)) as setup:
        atomic, _ = install_parent(setup, case)
        setup.skills.register_composite(CompositeSkill(
            SkillRef("r921_fixture_workflow", "1.0.0"), "find and take an egg",
            [CompositeOccurrence("parent", "parent", atomic.ref, {"object": BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role="object")})], ["parent"], [], [],
            setup.harness.task_contract(fixture_task(case)), {}, {}, {}, {"fixture_controlled": True}, SkillStatus.ACTIVE))
        setup.freeze(tmp_path / "frozen_bank")
    config["data_dir"] = str(tmp_path / "frozen_bank")
    config["experiment"].update(runtime_mode="frozen", freeze_skills=True, allow_long_term_knowledge_writes=False)
    config["cold_start"]["enabled"] = False
    for index in range(2):
        task_case = replace(case, case_id=f"frozen_full_system_{index}")
        harness = CandidateHarness(task_case)
        runtime = RouteProvider(task_case, case.route, harness.timeline)
        builder = RouteProvider(task_case, "tool_builder", harness.timeline)
        with AtomicSkillGraphSystem(config, harness=harness, readonly=True,
                provider={case.route: runtime, "tool_builder": builder}) as system:
            before = system.knowledge_digest()
            trace = system.run_task(fixture_task(task_case))
            assert trace.runtime_plan["source"] == "stored_composite"
            assert trace.benchmark_success, trace.failure_summary
            draft = next(iter(trace.metadata["runtime_automation_drafts"].values()))
            assert draft["r1_passed"]
            assert len(trace.environment_actions) == 5
            assert len(builder.requests) == 1  # no temporary tool survives to skip this Builder
            assert before == system.knowledge_digest()
            assert all("task_local" not in str(ref) for ref in system.skills.list_refs("atomic"))
@pytest.mark.parametrize("won", [False, True])
def test_trial_terminal_stops_provider_and_environment_calls(tmp_path, won):
    case = RouteCase("terminal", terminal_action="OPEN", terminal_won=won)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=CandidateHarness(case)) as system:
        outcome = run_node_case(system, case)
        assert system.harness.validator_channel().snapshot()["done"]
        assert outcome["ctx"].terminal_latched is won
        # Won finalizes without another request. Done-without-won retains the
        # existing failure-return policy, with no additional environment step.
        assert len(outcome["runtime"].requests) == 2
        assert len(outcome["builder"].requests) == 1
        assert len(outcome["trace"].environment_actions) == 4
        if won:
            assert outcome["timeline"][-1]["kind"] == "action"
        trial = next(iter(outcome["ctx"].runtime_tool_trials.values()))
        assert trial["r1"]["terminal_interrupted"] is won
        assert not trial["r1"]["admission_eligible"]
