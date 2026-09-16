"""Declared generic contract fixtures on a real ALFWorld world (no policy LLM)."""
from __future__ import annotations

import argparse
from pathlib import Path
import uuid

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind, RuntimeBinding, BindingSource, BindingStatus, BindingResolution
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeOccurrence, RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.support_closure import SupportClosure
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.tooling.proposal import tool_proposal_from_dict, ToolProvenance
from atomic_skillgraph.traces.canonical import canonical_action_indices
from experiments.run_v3_self_tooling_targeted import isolated_config


def install_fixture(system, name, preconditions, effects, actions, *, source_target=""):
    role = BindingExpression(BindingExprKind.SKILL_INPUT, source_role="target")
    atomic = AbstractAtomicSkill(SkillRef(name, "1.0.0"), "generic contract fixture",
        [ParameterSpec("target", "entity", runtime_resolvable=True, required_resolution="concrete")],
        [], preconditions, effects, {}, [], {}, {"acceptance_fixture": True}, SkillStatus.ACTIVE)
    system.skills.register_atomic(atomic)
    if not actions:
        return atomic, None
    proposal = tool_proposal_from_dict({"proposal_version": "1", "decision": "create",
        "summary": "declared world contract fixture", "atomic_ref": str(atomic.ref),
        "inputs": to_primitive(atomic.inputs), "outputs": [], "max_actions": len(actions),
        "program": [{"op": "ACTION", "node_id": f"a{i}", "action_type": kind,
                     "argument_mapping": {argument: to_primitive(role)} if argument else {},
                     "expected_effects": []} for i, (kind, argument) in enumerate(actions)]
                    + [{"op": "RETURN", "node_id": "end", "output_sources": {}}],
        "final_effects": to_primitive(effects), "evidence_outputs": [], "path_expectations": [], "rationale": "acceptance only"})
    source = CanonicalAtomicOccurrence(name, name, name, 0, 0, {"target": source_target}, {}, atomic.inputs,
        [], preconditions, effects, [], [], to_primitive(system.harness._current_task), "fixture", atomic.ref)
    compiled = system.tool_compiler.compile_proposal(source, atomic, proposal,
        ToolProvenance(source="r10_acceptance_fixture", atomic_ref=str(atomic.ref), source_trace_id="fixture", occurrence_id=name))
    compiled.tool.status = ToolStatus.ACTIVE
    compiled.implementation.status = SkillStatus.ACTIVE
    system.tools.register(compiled.tool)
    system.skills.register_implementation(compiled.implementation)
    return atomic, compiled.implementation


def bind(ctx, occurrence, value):
    # Value comes from an actual, recorded public destination affordance.
    spec = next(a for a in ctx.action_catalog if a.action_type == "GO_TO" and a.arguments.get("destination") == value)
    ctx.binding_store.commit_grounded(occurrence.occurrence_id, {"target": RuntimeBinding(
        "target", value, "entity", BindingSource.HARNESS_EVIDENCE, BindingStatus.GROUNDED,
        BindingResolution.CONCRETE, [f"action_catalog:{spec.action_id}:revision:{spec.revision}"], ctx.world_revision)})


def run(config_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = isolated_config(load_config(config_path), output)
    config["experiment"]["task_manifest_path"] = None
    with AtomicSkillGraphSystem(config) as system:
        harness = system.harness
        task = harness.load_tasks(limit=1)[0]
        initial = harness.reset(task)
        targets = [a.arguments["destination"] for a in initial.catalog if a.action_type == "GO_TO"]
        openable = unavailable = None
        inspected = []
        for target in targets:
            catalog = harness.reset(task).catalog
            action = next(a for a in catalog if a.action_type == "GO_TO" and a.arguments["destination"] == target)
            result = harness.execute_action(action.action_id, action.revision)
            has_open = any(a.action_type == "OPEN" and a.arguments.get("object") == target for a in result.catalog)
            inspected.append({"target": target, "catalog": to_primitive(result.catalog), "has_open": has_open})
            if has_open and openable is None:
                openable = target
            elif not has_open and unavailable is None:
                unavailable = target
            if openable and unavailable:
                break
        if not openable or not unavailable:
            raise RuntimeError("fixture requires two public destinations with different current OPEN affordances")
        expr = BindingExpression(BindingExprKind.SKILL_INPUT, source_role="target")
        at = SemanticPredicate("agent.at_location", {"location": expr})
        opened = SemanticPredicate("container.open", {"container": expr})
        parent, _ = install_fixture(system, "r10_fixture_parent", [opened], [at], [])
        install_fixture(system, "r10_fixture_go", [], [at], [("GO_TO", "destination")], source_target=openable)
        install_fixture(system, "r10_fixture_open", [at], [opened], [("OPEN", "object")], source_target=openable)
        occurrence = RuntimeOccurrence("parent", "parent", parent.ref, [], {}, [], parent.effects)
        plan = RuntimeLinearPlan(task.task_id, "atomic_composition", "", [occurrence], ["parent"], [], [], harness.task_contract(task), {})
        ctx = TaskRuntimeContext.create(task, plan, harness, system.orchestrator.create_trace_builder(task), RuntimeBudget())
        ctx.runtime_config = config["runtime"]
        ctx.budget.begin_node("parent")
        ctx.begin_occurrence(occurrence)
        bind(ctx, occurrence, openable)
        closed = SupportClosure(system.orchestrator.node_executor).close(occurrence, ctx, [])
        atomic_write_json(output / "support_trace.json", to_primitive(ctx.trace_builder.trace))
        atomic_write_json(output / "support_state.json", to_primitive(ctx.grounding_state_by_occurrence))
        assert closed and ctx.trace_builder.trace.metadata["r10_metrics"]["support_closure_success_count"] == 2
        before = harness._runtime_state_digest()
        actions_before = len(ctx.trace_builder.trace.environment_actions)
        failure_atomic, failure_impl = install_fixture(system, "r10_fixture_partial", [], [opened], [("GO_TO", "destination"), ("LOOK", None), ("OPEN", "object")], source_target=unavailable)
        failure = RuntimeOccurrence("partial", "partial", failure_atomic.ref, [], {}, [str(failure_impl.ref)], failure_atomic.effects)
        ctx.begin_occurrence(failure)
        bind(ctx, failure, unavailable)
        invocations = system.invocation_compiler.compile_candidates(failure, ctx.binding_store, task_id=task.task_id)
        result = system.orchestrator.node_executor.try_autonomous(failure, invocations, ctx)
        assert result.started and not result.atomic_effect_passed
        assert before == harness._runtime_state_digest()
        assert canonical_action_indices(ctx.trace_builder.trace) == list(range(actions_before))
        assert ctx.budget.used_global_actions > actions_before
        assert len(ctx.trace_builder.trace.environment_actions) - actions_before == 2
        summary = {"passed": True, "real_alfworld": True, "fixture_assets": True,
            "task_id": task.task_id, "recursive_support_successes": 2,
            "restored_digest": before, "r10_metrics": ctx.trace_builder.trace.metadata["r10_metrics"],
            "failure": to_primitive(result), "provider_calls": len(system.usage.events)}
        atomic_write_json(output / "trace.json", to_primitive(ctx.trace_builder.trace))
        atomic_write_json(output / "public_fixture_selection.json", inspected)
        atomic_write_json(output / "summary.json", summary)
        print(summary, flush=True)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alfworld_train_full_120_r10_seed42.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
