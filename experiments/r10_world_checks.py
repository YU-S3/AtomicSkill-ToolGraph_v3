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
    proposal = tool_proposal_from_dict({"proposal_version": "2", "entry_contract": {"conditions": [], "grounding_constraints": []}, "decision": "create",
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
    raise ValueError("R10 automatic Support acceptance was retired; use R10.2 explicit-node acceptance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alfworld_train_full_120_r10_seed42.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
