import json

import pytest

from atomic_skillgraph.runtime.interventions import DeploymentIntervention
from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.agents.protocol import NativeToolSpec
from atomic_skillgraph.runtime.runtime_step import run_runtime_step
from test_r10_runtime import setup, StepProvider


@pytest.mark.parametrize("condition,programs,automatic,presentation", [
    ("C11",True,True,"new"), ("C01",True,True,"old"), ("C10",False,False,"new"),
    ("C00",False,False,"old"), ("A0",True,False,"new")])
def test_CAU_explicit_diagnostic_masks(condition,programs,automatic,presentation):
    config = {"repair_revision":"R10.3", "experiment":{"experiment_kind":"diagnostic"},
              "r103_interventions":{"condition":condition}}
    p = DeploymentIntervention.from_config(config)
    assert (p.programs,p.automatic_entry,p.presentation) == (programs,automatic,presentation)
    config["experiment"]["experiment_kind"] = "formal"
    with pytest.raises(ValueError,match="diagnostic"):
        DeploymentIntervention.from_config(config)


def test_CAU_old_presentation_keeps_same_current_semantic_surface():
    from atomic_skillgraph.agents.baseline_runtime_prompt_texts import R10_STEP_PROMPT
    from atomic_skillgraph.agents.runtime_policy_projection import restore_native_interfaces
    spec = NativeToolSpec("invoke_example", "Current legal capability.", {"type":"object","properties":{}})
    args = dict(task_goal="public goal", atomic_contract={"summary":"a","inputs":[],"outputs":[],"preconditions":[],"effects":[]},
        observation="public", action_catalog=[], relevant_action_history=[], remaining_budget={},
        implementation_invocations=[spec], native_tool_specs=[spec])
    builder, audit = ContextBuilder(), {}
    new = builder.runtime_node(**args,projection_audit=audit)
    builder.runtime_presentation = "old"
    old = builder.runtime_node(**args)
    assert old.startswith(R10_STEP_PROMPT)
    get = lambda text: json.loads(text.split("\n\nPOLICY_CONTEXT_JSON\n")[1])
    assert restore_native_interfaces(get(new),[spec],audit["expression_audit"]) == get(old)


@pytest.mark.parametrize("condition", ["C10","A0"])
def test_CAU_node_mask_applied_before_native_request_not_runner_noop(tmp_path,condition):
    def choose(request,n):
        names = {t.name for t in request.tools}
        assert "environment_action" in names and "validate_current_atomic" in names
        if condition == "C10":
            assert not any(n.startswith("invoke_") for n in names)
            assert not any("automation" in n for n in names)
        else:
            assert any(n.startswith("invoke_impl_") for n in names)
            assert "request_runtime_automation" in names
        return "report_runtime_status", {"status":"cannot_resolve", "detail":"Controlled interface inspection."}
    system,ctx,occ,invocations,provider = setup(tmp_path,choose)
    mask = DeploymentIntervention.from_config({"repair_revision":"R10.3",
        "experiment":{"experiment_kind":"diagnostic"},"r103_interventions":{"condition":condition}})
    ctx.runtime_config["r103_deployment_intervention"] = mask.to_dict()
    try:
        assert system.orchestrator.node_executor.try_autonomous(occ,invocations,ctx) is None
        run_runtime_step(system.orchestrator.node_executor,"preparation",occ,ctx,invocations,[])
        assert len(provider.requests) == 1
        assert not ctx.trace_builder.trace.tool_executions
    finally:
        system.close()


def test_CAU_A0_retains_actual_agent_selected_program_execution(tmp_path):
    from experiments.r10_world_checks import install_fixture
    from atomic_skillgraph.core.contracts import SemanticPredicate
    from atomic_skillgraph.runtime.task_runtime import run_dynamic
    selected = {}
    def choose(request, count):
        if count == 1:
            return "invoke_support_atomic", {"support_atomic_ref":selected["ref"],
                                             "arguments":{"target":"cabinet_1"}}
        return "report_runtime_status", {"status":"give_up","detail":"controlled boundary"}
    system,ctx,occ,invocations,provider = setup(tmp_path, choose)
    ctx.runtime_config["r103_deployment_intervention"] = DeploymentIntervention(
        "A0", "new", True, False).to_dict()
    try:
        atomic,_ = install_fixture(system,"a0_nav",[],
            [SemanticPredicate("agent.at_location",{"location":"$target"})],[("GO_TO","destination")])
        selected["ref"] = str(atomic.ref)
        assert system.orchestrator.node_executor.try_autonomous(occ,invocations,ctx) is None
        run_dynamic(system.orchestrator.node_executor,ctx)
        assert any(t.result.get("started") for t in ctx.trace_builder.trace.tool_executions)
        assert len(ctx.trace_builder.trace.environment_actions) == 1
    finally:
        system.close()


@pytest.mark.parametrize("rescue,continuation",[(False,False),(True,False),(False,True)])
def test_CAU_C10_task_channels_closed_with_existing_local_draft(tmp_path,rescue,continuation):
    from atomic_skillgraph.runtime.task_runtime import run_dynamic
    def choose(request,count):
        assert {t.name for t in request.tools} == {"environment_action","report_runtime_status"}
        return "report_runtime_status", {"status":"give_up","detail":"controlled boundary"}
    system,ctx,_,_,provider = setup(tmp_path,choose)
    ctx.runtime_config["r103_deployment_intervention"] = DeploymentIntervention(
        "C10","new",False,False).to_dict()
    ctx.runtime_automation_drafts["preexisting"] = {"state":"validated"}
    try:
        run_dynamic(system.orchestrator.node_executor,ctx,rescue=rescue,cold_start_continuation=continuation)
        assert len(provider.requests) == 1
        assert not ctx.trace_builder.trace.tool_executions
    finally:
        system.close()


def test_CAU_learning_ablation_identity_keeps_baseline_and_no_policy_leak():
    from atomic_skillgraph.evolution.learning_interventions import identity_scope, enhanced
    from atomic_skillgraph.evolution.identity_matching import match_atomic, match_tool, verify_atomic_proof
    from test_r103_identity import atomic, tool
    a,b = atomic(),atomic(("c","d"),("p","q"))
    assert match_tool(tool(),tool("thing","answer","pick","element")).status == "exact"
    with identity_scope(False):
        proof = match_atomic(a,b)
        assert proof.status == "exact" and verify_atomic_proof(a,b,proof.proof)
        assert match_tool(tool(),tool()).status == "exact"
        assert match_tool(tool(),tool("thing","answer","pick","element")).status == "different"
        assert match_atomic(a,b,compatible_mapping=lambda i,o:i["a"]=="d").status == "different"
    assert enhanced()
    assert match_atomic(a,b,compatible_mapping=lambda i,o:i["a"]=="d").status == "exact"


@pytest.mark.parametrize("condition",["L-identity-support","L-generalization"])
def test_CAU_learning_banks_cannot_import_full_or_change_condition(tmp_path,condition):
    from atomic_skillgraph.system import AtomicSkillGraphSystem
    from test_r10_runtime import CheckpointHarness,route
    from fixtures.r921_self_tooling_cases import fixture_config
    from test_r103_identity import atomic
    from atomic_skillgraph.evolution.learning_interventions import bind_bank
    config = fixture_config(tmp_path)
    config.update(repair_revision="R10.3",r103_learning_intervention=condition)
    config["experiment"].update(experiment_kind="learning_ablation",task_manifest_path=None)
    with AtomicSkillGraphSystem(config,harness=CheckpointHarness(route.RouteCase("learning")),
                                provider=StepProvider(lambda *a:None)) as system:
        assert system.is_empty_knowledge_bank()
        system.skills.register_atomic(atomic())
        system.config.pop("r103_learning_intervention")
        with pytest.raises(RuntimeError,match="condition"):
            bind_bank(system)
    config.pop("r103_learning_intervention")
    with pytest.raises(RuntimeError,match="condition"):
        AtomicSkillGraphSystem(config,harness=CheckpointHarness(route.RouteCase("learning")),
                              provider=StepProvider(lambda *a:None))
