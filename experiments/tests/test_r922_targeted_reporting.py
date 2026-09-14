"""Model draft corrections remain visible and cannot erase a successful trial."""
from experiments import self_tooling_targeted as route
from experiments.fakes import FakeProviderRequest, FakeReply
from experiments.run_v3_r922_targeted import summarize_boundary_outcome
from atomic_skillgraph.system import AtomicSkillGraphSystem
from fixtures.r921_self_tooling_cases import fixture_config


def test_r0_correction_accounts_for_both_native_submissions(tmp_path, monkeypatch):
    class CorrectingProvider(route.RouteProvider):
        def complete(self, messages, *, tools=None):
            if self.stage != "tool_builder" and not self.requests:
                draft = route.fixed_draft(FakeProviderRequest(tuple(messages), tuple(tools)), self.case)
                draft["draft_id"] += "_invalid"
                draft["preconditions"] = [{"predicate": "nonexistent", "args": {}}]
                self.requests.append({"fixture": "initial_invalid_draft"})
                reply = FakeReply.tool("propose_runtime_automation_atomic", draft,
                    prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)
                return reply.materialize(call_id="initial_rejection", tools=tools)
            return super().complete(messages, tools=tools)
    monkeypatch.setattr(route, "RouteProvider", CorrectingProvider)
    case = route.RouteCase("corrected", route="runtime_seeded", openable=False)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=route.CandidateHarness(case)) as system:
        outcome = route.run_node_case(system, case)
    result = summarize_boundary_outcome(case, outcome)
    assert result["case_passed"]
    assert result["draft_count"] == 2
    assert result["r0_rejection_count"] == result["actual_trial_count"] == result["r1_pass_count"] == 1
    assert [item["route_last_stage"] for item in result["draft_results"]] == ["r0_rejected", "r1_passed"]
    assert result["trial_action_count"] >= 2
    assert result["targeted_external_tokens"] == 0
