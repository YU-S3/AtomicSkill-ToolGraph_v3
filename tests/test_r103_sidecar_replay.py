"""Two-source admission with original source reconstruction and interpreter.

These controlled sources test G07/G08 execution, not real-model generalization.
The separate generalization tests cover proposal/source ownership validation.
"""
from copy import deepcopy
from dataclasses import replace
import pytest

from atomic_skillgraph.core.status import ToolStatus, SkillStatus
from atomic_skillgraph.evolution.generalization import apply_sidecar
from atomic_skillgraph.evolution.tool_compiler import build_occurrence_replay_case
from test_r1021_i import source_case, author


@pytest.mark.parametrize("bad_second", [False, True])
def test_G07_G08_actual_two_source_replay_no_online_credit(tmp_path, bad_second):
    first, ctx, provider, c, n = source_case(tmp_path / "first", case_id="source_one")
    second, ctx2, _, c2, n2 = source_case(tmp_path / "second", sparse=True, case_id="source_two")
    try:
        ctx2.trace_builder.trace.finish()
        first.traces.save_atomic(ctx2.trace_builder.trace)
        provider.choose = lambda r, _: author(r, create=True)
        before = len(provider.requests)
        compiled, _ = first._build_tool_for_occurrence(c, first._canonical_atomic_for_occurrence(c), n,
            ctx.trace_builder.trace, source_task=ctx.task,
            additional_evidence_sources=[{"source_slot":"history",
                "source_boundary":first._build_builder_source_context(c2,n2),
                "evidence_support":c2.action_events}], allow_exact_reuse=False)
        assert compiled is not None
        case2 = build_occurrence_replay_case(c2, compiled.atomic, source_task=ctx2.task, kind="tool_proposal_replay")
        assert compiled.tool.tests[0]["source_task"]["task_id"] != case2["source_task"]["task_id"]
        if bad_second:
            case2 = deepcopy(case2)
            case2["bindings"]["container"] = "unobserved_container"
        item = replace(compiled, tool=replace(compiled.tool, tests=[compiled.tool.tests[0],case2]))
        trace = ctx.trace_builder.trace
        state = {"item":item,"attempt":{"attempt_key":"controlled_two_source"},"audit":{}}
        calls = []
        original = first._replay_case_with_source_authority
        def observed(tool, case, **kwargs):
            value = original(tool, case, **kwargs)
            calls.append((case["source_task"]["task_id"],value))
            return value
        first._replay_case_with_source_authority = observed
        events = apply_sidecar(first,trace,ctx.task,state)
        assert len(provider.requests) == before + 1
        assert len(calls) == 2
        assert calls[0][0] != calls[1][0]
        assert calls[0][1].passed
        assert calls[1][1].passed is (not bad_second)
        assert all(e.event.value in {"proposed","validated"} for e in events)
        if bad_second:
            assert state["attempt"]["result_status"] == "replay_rejected_atomic_retained"
        else:
            assert state["attempt"]["result_status"] == "candidate_admitted", state
            impl_ref, tool_ref = state["audit"]["refs"][1:]
            assert first.skills.get_implementation(impl_ref).status is SkillStatus.CANDIDATE
            assert first.tools.get(tool_ref).status is ToolStatus.CANDIDATE
    finally:
        first.close()
        second.close()
