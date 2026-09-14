"""R9.2.2 isolated acceptance; never a formal train or frozen-test launcher."""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.results import RuntimeOccurrence, RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.core.status import ToolStatus
from atomic_skillgraph.evolution.replay import ReplaySourceAuthority
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.traces.store import TraceStore
from experiments import self_tooling_targeted as route
from experiments.protocol import hash_code, hash_config, validate_deepseek_formal_llm
from experiments.run_v3_self_tooling_targeted import isolated_config, load_alfworld_case, read_fixture_manifest


class ChangingMultiHarness(route.CandidateHarness):
    """Finite, declared scene; uses real parser, authority and all runtime code."""
    def _replace_catalog(self):
        locations = {0: ("countertop_1", "countertop_2", "shelf_1"),
                     1: ("shelf_1", "cabinet_1"), 2: ("cabinet_1",)}.get(self._revision, ())
        raw = ["go to " + value.replace("_", " ") for value in locations]
        if self.location == "cabinet_1" and not self.held:
            raw += [f"take apple {number} from cabinet 1" for number in (1, 2)]
        return self._catalog.replace(raw, self._revision)


class ModelDraftProvider(route.RouteProvider):
    """Force only the targeted entry; every draft/program/value is model output."""
    def complete(self, messages, *, tools=None):
        messages = copy.deepcopy(messages)
        selected = list(tools or [])
        first = self.stage != "tool_builder" and not self.forced
        if first:
            selected = [tool for tool in selected if tool.name == "propose_runtime_automation_atomic"]
            if not selected:
                raise route.FixtureSetupError("native automation entry unavailable")
            messages[0]["content"] += (
                "\nTARGETED ACCEPTANCE: propose a bounded discovery helper now. "
                "The target is a semantic family and may have multiple instances. "
                "Use distinct concrete entity and location outputs with the public "
                "output_semantic_constraints contract, and only the minimal supported "
                "discovery Effect. Select no concrete instance until current facts "
                "support it. The helper must search the changing current catalog; "
                "cross-action catalog loops should explicitly refresh_each_iteration."
            )
            self.forced = True
        payload = {"messages": route.safe_messages(messages), "tools": [tool.to_openai() for tool in selected]}
        self.requests.append(payload)
        if self.audit:
            self.audit(self.stage, len(self.requests), "request", payload)
        turn = self.delegate.complete(messages=messages, tools=selected)
        result = to_primitive(turn)
        self.replies.append(result)
        if self.audit:
            self.audit(self.stage, len(self.requests), "reply", result)
        return turn


def summarize_boundary_outcome(case, outcome):
    """Account for native R0 correction without hiding rejected proposals.

    R9.2.1's fixture summary requires exactly one scripted draft. Here the real
    Runtime may correct its proposal within the existing session allowance.
    """
    attempts = []
    for draft_id, draft in outcome["ctx"].runtime_automation_drafts.items():
        context = copy.copy(outcome["ctx"])
        context.runtime_automation_drafts = {draft_id: draft}
        row = route.summarize_case(case, {**outcome, "ctx": context})
        attempts.append((draft_id, row))
    result = next((dict(row) for _, row in attempts if row["case_passed"]),
                  dict(attempts[-1][1]) if attempts else route.summarize_case(case, outcome))
    result["draft_count"] = len(attempts)
    result["draft_results"] = [{"draft_id": draft_id, **{key: row[key] for key in (
        "case_passed", "route_last_stage", "route_failure_code", "r0_result", "static_result",
        "trial_action_count", "r1_passed", "r1_result")}} for draft_id, row in attempts]
    result["r0_rejection_count"] = sum(row["route_last_stage"] == "r0_rejected" for _, row in attempts)
    result["actual_trial_count"] = sum(row["trial_started"] for _, row in attempts)
    result["r1_pass_count"] = sum(row["r1_passed"] for _, row in attempts)
    return result


def live_boundaries(source, root, fixture_manifest=None, real_env_index=None):
    config = isolated_config(source, root)
    if real_env_index is not None:
        if real_env_index < 0:
            raise route.FixtureSetupError("env_index must be nonnegative")
        harness = AlfWorldAdapter(split="train")
        task = harness.load_tasks(limit=real_env_index+1)[-1]
        case = route.RouteCase(f"train_{real_env_index}_multi", route="runtime_seeded",
                              target=task.context["semantic_bindings"]["object"])
        atomic_write_json(root / "source_task.json", to_primitive(task))
    elif fixture_manifest:
        fixed = read_fixture_manifest(fixture_manifest)["cases"][0]
        case, harness, task = load_alfworld_case(fixed, source)
    else:
        case = route.RouteCase("changing_multi", route="runtime_seeded", target="apple", openable=False)
        harness, task = ChangingMultiHarness(case), None
    snapshots = []
    execute = harness.execute_action
    def observed_action(*args, **kwargs):
        snapshots.append({"before": to_primitive(harness.action_catalog())})
        result = execute(*args, **kwargs)
        snapshots[-1]["after"] = to_primitive(harness.action_catalog())
        return result
    harness.execute_action = observed_action
    original = route.RouteProvider
    route.RouteProvider = ModelDraftProvider
    try:
        with AtomicSkillGraphSystem(config, harness=harness) as system:
            def audit(stage, index, kind, payload):
                atomic_write_json(root / f"{stage}_{index:03d}_{kind}.json", payload)
            outcome = route.run_node_case(system, case, live=True, task=task, audit=audit)
            result = summarize_boundary_outcome(case, outcome)
            result.update(runtime_draft_model_generated=True, builder_submission_scripted=False,
                          environment_kind="alfworld" if fixture_manifest or real_env_index is not None else "controlled_dynamic_multi")
            result["multiple_current_target_instances"] = any(
                len({action["arguments"].get("object") for action in snapshot[key]
                     if action["action_type"] == "TAKE" and harness.semantic_value_compatible(
                         role="object", concrete_value=action["arguments"].get("object"),
                         semantic_anchor=case.target, semantic_type="entity")}) >= 2
                for snapshot in snapshots for key in ("before", "after"))
            result["passed"] = result["case_passed"] and result["multiple_current_target_instances"]
            atomic_write_json(root / "public_catalog_snapshots.json", snapshots)
            atomic_write_json(root / "trace.json", to_primitive(outcome["trace"]))
            return result
    finally:
        route.RouteProvider = original


def replay_scaling(source, root, source_root):
    source_root = Path(source_root).resolve(strict=True)
    bank = source_root / "data_v3"
    old_database = StateDatabase(bank / "state.sqlite3", readonly=True)
    try:
        old_artifacts = ArtifactStore(bank, old_database)
        old_tools = ToolRegistry(old_artifacts, old_database)
        old_skills = SkillRegistry(old_artifacts, old_database)
        # Deterministic evidence fixture: the most independently observed
        # executable in the supplied bank, not a task-family rule.
        old_tool = max(old_tools.tools(), key=lambda tool: len(tool.tests))
        old_impl = next(impl for impl in old_skills.implementations()
                        if any(binding.tool_ref == old_tool.ref for binding in impl.tool_bindings))
        atomic = old_skills.get_atomic(old_impl.abstract_ref)
        cases = copy.deepcopy(old_tool.tests[:10])
    finally:
        old_database.close()
    if len(cases) != 10 or len({case["case_id"] for case in cases}) != 10:
        raise route.FixtureSetupError("ten distinct source replay cases required")
    config = isolated_config(source, root)
    config["experiment"]["task_manifest_path"] = None
    authority = ReplaySourceAuthority(TraceStore(source_root, readonly=True), allowed_split="train")
    harness = AlfWorldAdapter(split="train")
    counts = {"reset": 0, "initialize": 0}
    for name in counts:
        original = getattr(harness, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        setattr(harness, name, counted)
    rows = []
    with AtomicSkillGraphSystem(config, harness=harness) as system:
        system._replay_source_authority = lambda: authority
        atomic_ref = system.aligner.align_atomic(atomic)
        refs, impls = set(), set()
        for index, case in enumerate([*cases, cases[4]]):
            task = authority.resolve(case, current_task=None, current_trace=None)
            trace = system.orchestrator.create_trace_builder(task).trace
            candidate = replace(old_tool, tests=[case], status=ToolStatus.ADMISSION_PENDING)
            admitted = system.admission.admit_tool(candidate, atomic=atomic, harness=harness,
                replay=lambda tool, item: system._replay_case_with_source_authority(
                    replace(tool, ref=system.aligner.replay_target_ref(tool)), item,
                    current_task=None, current_trace=None, audit_trace=trace))
            alignment = system.aligner.align_tool_with_replays(admitted, admission=system.admission, replay=None)
            refs.add(str(alignment.ref))
            implementation = system.admission.admit_implementation(old_impl, admitted, atomic=atomic, harness=harness)
            impl_ref = system.aligner.align_implementation(implementation, atomic_ref, alignment.ref)
            impls.add(str(impl_ref))
            system._capture_replay_bank_metrics(trace)
            trace.finish()
            system.traces.save_atomic(trace)
            system._commit_replay_certificates(trace)
            row = {"case_id": case["case_id"], "passed": alignment.admitted,
                   "implementation_admitted": implementation.status.value == "candidate",
                   **trace.metadata.get("replay_accounting", {}), "reset_counts": dict(counts),
                   "tool_refs": sorted(refs), "implementation_refs": sorted(impls)}
            rows.append(row)
            atomic_write_json(root / "progress.json", rows)
            print(json.dumps({"case": index+1, **row}), flush=True)
        scaling_counts = dict(counts)
        # Real Direct on the same learned Atomic with explicit task-local
        # concrete inputs; no new provider or preferred-candidate override.
        task = authority.resolve(cases[0], current_task=None, current_trace=None)
        occurrence = RuntimeOccurrence("direct", "direct", atomic_ref, [],
            {role: BindingExpression(BindingExprKind.CONSTANT, constant=value)
             for role, value in cases[0]["bindings"].items()},
            [impl_ref], atomic.effects)
        plan = RuntimeLinearPlan(task.task_id, "stored_composite", "", [occurrence], ["direct"], [], [], harness.task_contract(task), {})
        ctx = TaskRuntimeContext.create(task, plan, harness, system.orchestrator.create_trace_builder(task),
            RuntimeBudget(global_action_budget=100, node_action_budget=35))
        ctx.budget.begin_node("direct")
        ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
        ctx.begin_occurrence(occurrence)
        invocations = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=task.task_id)
        direct = system.orchestrator.node_executor.try_autonomous(occurrence, invocations, ctx)
        atomic_write_json(root / "direct_trace.json", to_primitive(ctx.trace_builder.trace))
        direct_passed = bool(direct is not None and direct.atomic_effect_passed and
                             direct.node_status.value == "direct_autonomous_success")
        return {"passed": all(row["passed"] and row["implementation_admitted"] for row in rows)
                and scaling_counts == {"reset": 10, "initialize": 10}
                and len(refs) == len(impls) == 1 and direct_passed and len(invocations) == 1,
                "rows": rows, "scaling_reset_counts": scaling_counts,
                "direct_passed": direct_passed, "direct_candidate_count": len(invocations),
                "direct_result": to_primitive(direct), "bank": trace.metadata["replay_bank_metrics"],
                "source_root": str(source_root), "source_tool": str(old_tool.ref)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("replay-scaling", "live-boundaries"), required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--fixture-manifest")
    parser.add_argument("--real-env-index", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = load_config(args.config)
    validate_deepseek_formal_llm(source)
    atomic_write_json(root / "manifest.json", {"is_formal_experiment": False,
        "config_hash": hash_config(source), "code_hash": hash_code(Path(__file__).resolve().parents[1]),
        "mode": args.mode, "source_root": args.source_root, "fixture_manifest": args.fixture_manifest})
    started = time.monotonic()
    try:
        result = (replay_scaling(source, root, args.source_root) if args.mode == "replay-scaling"
                  else live_boundaries(source, root, args.fixture_manifest, args.real_env_index))
    except Exception as exc:
        atomic_write_json(root / "summary.json", {"passed": False, "error_type": type(exc).__name__,
                          "error": str(exc), "elapsed_seconds": time.monotonic()-started})
        raise
    result["elapsed_seconds"] = time.monotonic()-started
    atomic_write_json(root / "summary.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result.get("passed", result.get("case_passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
