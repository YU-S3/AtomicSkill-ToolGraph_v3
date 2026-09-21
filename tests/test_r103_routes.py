"""Route diagnostics use production compilation; no preflight PASS stubs."""
import copy
from dataclasses import replace

from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import SkillStatus, ToolStatus, RuntimeMode
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.governance.lifecycle import CandidateUsePolicy
from atomic_skillgraph.system import AtomicSkillGraphSystem
from test_r10_runtime import CheckpointHarness, StepProvider, route
from fixtures.r921_self_tooling_cases import fixture_config


def setup(tmp_path):
    case = route.RouteCase("route_diagnostic")
    harness = CheckpointHarness(case)
    config = fixture_config(tmp_path)
    config["repair_revision"] = "R10.3"
    config["experiment"]["task_manifest_path"] = None
    system = AtomicSkillGraphSystem(config, harness=harness, provider=StepProvider(lambda *a: None))
    atomic, refs = route.install_parent(system, case)
    task = route.fixture_task(case)
    plan = route.make_plan(task, atomic, refs, harness)
    ctx = TaskRuntimeContext.create(task, plan, harness, system.orchestrator.create_trace_builder(task), RuntimeBudget())
    occurrence = plan.occurrences[0]
    ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
    ctx.begin_occurrence(occurrence)
    system.invocation_compiler.candidate_policy = CandidateUsePolicy(0.0)
    return system, ctx, occurrence, system.skills.get_implementation(refs[0])


def compile_routes(system, ctx, occurrence):
    return system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store,
        task_id=ctx.task.task_id, evidence_store=ctx.evidence_store, revision=ctx.world_revision,
        task_contract=ctx.task_contract)


def test_active_representative_uses_its_own_history_not_candidate_alias(tmp_path):
    import json
    from atomic_skillgraph.core.refs import ToolRef
    from atomic_skillgraph.core.serialization import to_primitive
    from atomic_skillgraph.governance.projections import ArtifactStats
    system, ctx, occ, active = setup(tmp_path)
    try:
        candidate = replace(active, ref=SkillRef('first_candidate_alias', '1.0.0'), status=SkillStatus.CANDIDATE)
        system.skills.register_implementation(candidate)
        tool = system.tools.get(active.tool_bindings[0].tool_ref)
        alternative_tool = replace(tool, ref=ToolRef('different_route', '1.0.0'),
            artifact={**tool.artifact, 'max_actions': tool.artifact['max_actions']+1})
        system.tools.register(alternative_tool)
        alternative = replace(active, ref=SkillRef('second_active', '1.0.0'),
            tool_bindings=[replace(active.tool_bindings[0], tool_ref=alternative_tool.ref)])
        system.skills.register_implementation(alternative)
        for impl, failures in ((candidate, 50), (active, 0), (alternative, 5)):
            stats = ArtifactStats(str(impl.ref), 'implementation', intrinsic_failure_count=failures)
            system.database.execute('INSERT OR REPLACE INTO lifecycle_projection VALUES(?,?,?)',
                (str(impl.ref), json.dumps(to_primitive(stats)), 0))
        occ.implementation_candidates = [candidate.ref, alternative.ref, active.ref]
        selected = compile_routes(system, ctx, occ)
        assert [c.implementation.ref for c in selected] == [active.ref, alternative.ref]
    finally:
        system.close()


def test_D_incompatible_active_cannot_suppress_candidate_or_mutate_bindings(tmp_path):
    system, ctx, occurrence, impl = setup(tmp_path)
    try:
        bad = replace(impl, ref=SkillRef("incompatible", "1.0.0"), compatibility={"harness_profiles": ["other_profile"]})
        candidate = replace(impl, ref=SkillRef("candidate", "1.0.0"), status=SkillStatus.CANDIDATE)
        system.skills.register_implementation(bad)
        system.skills.register_implementation(candidate)
        occurrence.implementation_candidates = [bad.ref, candidate.ref]
        before = copy.deepcopy(ctx.binding_store.snapshot_for_node(occurrence))
        selected = compile_routes(system, ctx, occurrence)
        assert [r.implementation.ref for r in selected] == [candidate.ref]
        assert ctx.binding_store.snapshot_for_node(occurrence) == before
        assert not ctx.trace_builder.trace.environment_actions
        assert not system.usage.events
        assert compile_routes(system, ctx, occurrence)[0].implementation.ref == candidate.ref
        assert system.invocation_compiler.route_diagnostics[0]["state"] == "preparable"
    finally:
        system.close()


def test_D_equivalent_aliases_one_route_and_frozen_no_candidate(tmp_path):
    system, ctx, occurrence, impl = setup(tmp_path)
    try:
        refs = []
        for n in range(4):
            candidate = replace(impl, ref=SkillRef(f"alias_{n}", "1.0.0"), status=SkillStatus.CANDIDATE)
            system.skills.register_implementation(candidate)
            refs.append(candidate.ref)
        occurrence.implementation_candidates = refs
        assert len(compile_routes(system, ctx, occurrence)) == 1
        system.invocation_compiler.mode = RuntimeMode.FROZEN
        assert compile_routes(system, ctx, occurrence) == []
        system.invocation_compiler.mode = RuntimeMode.ONLINE
        system.tools.update_status(impl.tool_bindings[0].tool_ref, ToolStatus.SUPPRESSED)
        assert compile_routes(system, ctx, occurrence) == []
    finally:
        system.close()


def test_D_helper_uses_same_route_quota_and_records_offered_route(tmp_path):
    from types import SimpleNamespace
    from atomic_skillgraph.agents.protocol import NativeToolSpec
    system,ctx,occ,impl = setup(tmp_path)
    try:
        ex = system.orchestrator.node_executor
        atomic = system.skills.get_atomic(occ.node_ref)
        old = copy.deepcopy(ctx.binding_store.snapshot_for_node(occ))
        availability = ex._support_execution_availability([atomic],ctx)
        assert availability[str(atomic.ref)]
        ex._record_support_display(ctx,"helper_session",[SimpleNamespace(atomic_ref=str(atomic.ref))],
                                   [NativeToolSpec("invoke_support_atomic","helper",{"type":"object"})])
        records = ctx.trace_builder.trace.metadata["candidate_route_exposures"]
        assert records[0]["implementation_ref"] == str(impl.ref)
        assert ctx.binding_store.snapshot_for_node(occ) == old
        assert not ctx.trace_builder.trace.environment_actions
        incompatible = replace(impl,compatibility={"harness_profiles":["incompatible"]},ref=SkillRef("incompatible_helper","1.0.0"))
        system.skills.register_implementation(incompatible)
        system.tools.update_status(impl.tool_bindings[0].tool_ref,ToolStatus.SUPPRESSED)
        assert not ex._support_execution_availability([atomic],ctx)[str(atomic.ref)]
    finally:
        system.close()


def test_D06_D07_one_route_quota_and_dedup_before_reserved_slot(tmp_path):
    from atomic_skillgraph.core.refs import ToolRef
    system,ctx,occ,impl = setup(tmp_path)
    try:
        refs = [impl.ref]
        for i in range(3):
            alias = replace(impl,ref=SkillRef(f"reliable_alias_{i}","1.0.0"))
            system.skills.register_implementation(alias)
            refs.append(alias.ref)
        original = system.tools.get(impl.tool_bindings[0].tool_ref)
        candidate_tool = replace(original,ref=ToolRef("candidate_program","1.0.0"),status=ToolStatus.CANDIDATE,
            artifact={**original.artifact,"max_actions":original.artifact["max_actions"]+1})
        system.tools.register(candidate_tool)
        candidate = replace(impl,ref=SkillRef("candidate_route","1.0.0"),status=SkillStatus.CANDIDATE,
            tool_bindings=[replace(impl.tool_bindings[0],tool_ref=candidate_tool.ref)])
        system.skills.register_implementation(candidate)
        occ.implementation_candidates = refs+[candidate.ref]
        class CountingPolicy:
            def __init__(self): self.calls=[]
            def allows(self,**kwargs):
                self.calls.append(kwargs)
                return CandidateUsePolicy(1.0).allows(**kwargs)
        policy = CountingPolicy()
        system.invocation_compiler.candidate_policy = policy
        selected = compile_routes(system,ctx,occ)
        assert len(selected) == 2 and any(r.implementation.ref==candidate.ref for r in selected)
        assert len(policy.calls) == 1  # Not once for Impl plus again for Tool.
        assert policy.calls[0]["artifact_kind"] == "implementation"
        assert not ctx.trace_builder.trace.environment_actions
    finally:
        system.close()
