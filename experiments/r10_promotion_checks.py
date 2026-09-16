"""Real provider/ALFWorld promotion route, with an explicitly fixed test entry.

Only the initial native menu and the single parent contract are fixtures.
Drafts, Tool IR, all concrete values, and all environment decisions are model
output. This is isolated route acceptance, not a reported benchmark experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from dataclasses import replace

from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.core.contracts import SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.evolution.runtime_support_promotion import collect_observations, prepare_and_apply
from atomic_skillgraph.runtime.orchestrator import refresh_learning_eligibility
from atomic_skillgraph.runtime.r10_metrics import finalize
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config, _reconcile_events, _require_formal_usage
from experiments import self_tooling_targeted as route
from experiments.run_v3_self_tooling_targeted import isolated_config
from experiments.protocol import hash_code


class DraftEntryProvider(route.RouteProvider):
    def complete(self, messages, *, tools):
        messages = copy.deepcopy(messages)
        selected = list(tools)
        if not self.forced:
            selected = [tool for tool in selected if tool.name == "request_runtime_automation"]
            if not selected:
                raise RuntimeError("R10 lightweight request entry is missing")
            self.forced = True
        if any(tool.name == "propose_runtime_automation_atomic" for tool in selected):
            messages[0]["content"] += (
                "\nTARGETED ACCEPTANCE: author a reusable bounded discovery helper for the "
                "current parent target. The input is the semantic target family. Expose both "
                "a discovered concrete entity and its discovered concrete location, with the "
                "public interface's output semantic constraints. Do not take/place the entity. "
                "For this declared acceptance boundary use one required semantic entity input "
                "(runtime_resolvable=true), two required concrete entity outputs "
                "(runtime_resolvable=false), no preconditions, and the single evidence Effect "
                "entity.discovered_at(entity, location) with cardinality=1 and distinct_by=''. "
                "Constrain the discovered entity to the semantic input. Choose your own role "
                "names and author the full draft under R0; the Tool program is built independently."
            )
        index = len(self.requests)
        self.requests.append({"messages": route.safe_messages(messages), "tools": [t.to_openai() for t in selected]})
        if self.audit:
            self.audit(self.stage, index, "request", self.requests[-1])
        turn = self.delegate.complete(messages=messages, tools=selected)
        if self.audit:
            self.audit(self.stage, index, "result", {"calls": [t.to_dict() if hasattr(t, "to_dict") else {"name": t.name, "arguments": t.arguments} for t in turn.tool_calls],
                "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens})
        return turn


def run(config_path, output, indexes):
    start_hash = hash_code(Path(__file__).resolve().parents[1])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = isolated_config(load_config(config_path), output)
    config["experiment"]["task_manifest_path"] = None
    rows = []
    with AtomicSkillGraphSystem(config) as system:
        parent_fixture = None
        tasks = {task.context["env_index"]: task for task in system.harness.load_tasks(limit=max(indexes) + 1)}
        for index in indexes:
            task = tasks[index]
            started = time.monotonic()
            root = output / task.task_id
            root.mkdir()
            case = route.RouteCase(task.task_id, target=task.context["semantic_bindings"]["object"])
            if parent_fixture is None:
                base, base_impls = route.install_parent(system, case)
                # Explicit, task-generic fixture precondition supplies the
                # entity/location predicate-position authority for Support.
                parent = replace(base, ref=SkillRef("r10_parent_discovered_take", "1.0.0"),
                    preconditions=[SemanticPredicate("entity.discovered_at",
                        {"entity": "$object", "location": "$source"}, effect_domain="evidence")])
                system.skills.register_atomic(parent)
                implementation = replace(system.skills.get_implementation(base_impls[0]),
                    ref=SkillRef("r10_parent_discovered_take_impl", "1.0.0"), abstract_ref=parent.ref)
                system.skills.register_implementation(implementation)
                parent_fixture = parent, [implementation.ref]
            parent, implementations = parent_fixture
            plan = route.make_plan(task, parent, implementations, system.harness)
            plan.source, plan.source_composite_ref = "atomic_composition", None
            # Explicit acceptance fixture; no production planner changes.
            system.planner.build_plan = lambda *args, **kwargs: plan
            def audit(stage, sequence, kind, payload):
                atomic_write_json(root / f"{stage}_{sequence:03d}_{kind}.json", payload)
            provider = DraftEntryProvider(case, "runtime", [], delegate=system._provider("runtime_preparation"), audit=audit)
            system._provider_override = {"runtime_preparation": provider, "runtime_seeded": provider,
                "tool_builder": system._provider("tool_builder"), "runtime_dynamic": system._provider("runtime_dynamic")}
            system._current_task_id = task.task_id
            start_usage, start_sessions = len(system.usage.events), len(system._observed_sessions)
            system._current_task_usage_start = start_usage
            offsets = system._provider_request_offsets()
            trace = system.orchestrator.run_task(task)
            system._attach_external_sessions(trace, system._observed_sessions[start_sessions:])
            usage = system.usage.events[start_usage:]
            trace.llm_usage = [event.to_dict() for event in usage]
            trace.metadata["usage_reconciliation"] = _reconcile_events(usage)
            system._attach_provider_requests(trace, offsets)
            system._require_resource_usage_complete(trace)
            _require_formal_usage(usage, trace.agent_turns)
            refresh_learning_eligibility(trace)
            observations = collect_observations(system, trace)
            runtime_events = system.credit.assign(trace)
            events = [*runtime_events, *prepare_and_apply(system, trace, task, observations)]
            trace.evidence_event_refs = list(dict.fromkeys([*trace.evidence_event_refs, *[e.event_id for e in events]]))
            finalize(trace, config)
            system.traces.save_atomic(trace)
            for observation in observations:
                system.runtime_support_store.append(observation)
            system._commit_replay_certificates(trace)
            system._commit_evidence(events)
            system._provider_override = None
            row = {"task_id": task.task_id, "trace_id": trace.trace_id, "strict_success": trace.benchmark_success and trace.task_contract_success,
                "learning_eligible": trace.learning_eligible, "observation_signatures": [o["contract_signature"] for o in observations],
                "promotions": trace.metadata.get("runtime_support_promotions", []),
                "rejections": trace.metadata.get("runtime_support_promotion_rejections", []),
                "duration_seconds": time.monotonic() - started}
            rows.append(row)
            atomic_write_json(output / "progress.json", rows)
            print(json.dumps(row), flush=True)
        atomic_write_json(output / "summary.json", {"cases": rows, "real_provider": True, "real_alfworld": True,
            "code_hash": start_hash, "code_unchanged": start_hash == hash_code(Path(__file__).resolve().parents[1])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alfworld_train_full_120_r10_seed42.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-index", action="append", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.env_index)
