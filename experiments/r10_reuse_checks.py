"""Real deployments of promoted Candidates; never force lifecycle status."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time

from atomic_skillgraph.core.serialization import atomic_write_json, read_json, to_primitive
from atomic_skillgraph.runtime.orchestrator import refresh_learning_eligibility
from atomic_skillgraph.runtime.r10_metrics import finalize
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config, _reconcile_events, _require_formal_usage
from experiments import self_tooling_targeted as route
from experiments.protocol import hash_code
from experiments.run_v3_self_tooling_targeted import isolated_config


class ReuseEntryProvider(route.RouteProvider):
    automatic = False

    def complete(self, messages, *, tools):
        messages, selected = copy.deepcopy(messages), list(tools)
        if not self.forced:
            name = "environment_action" if self.automatic else "invoke_support_atomic"
            selected = [tool for tool in tools if tool.name == name]
            if not selected:
                raise RuntimeError(f"Required declared acceptance entry is unavailable: {name}")
            messages[0]["content"] += (
                "\nTARGETED ACCEPTANCE: for this bootstrap only, perform the current public LOOK action "
                "with intent=explore. Then the ordinary executor will test deterministic Support reuse."
                if self.automatic else
                "\nTARGETED ACCEPTANCE: invoke the presented persistent discovery Support with the "
                "current target's semantic family input. Map its validated entity and location "
                "outputs to the parent's object and source. Do not build a new Tool."
            )
            self.forced = True
        request = {"messages": route.safe_messages(messages), "tools": [tool.to_openai() for tool in selected]}
        self.requests.append(request)
        if self.audit:
            self.audit(self.stage, len(self.requests), "request", request)
        turn = self.delegate.complete(messages=messages, tools=selected)
        return turn


def run(config_path, source, output, indexes):
    source, output = Path(source).resolve(), Path(output).resolve()
    start_hash = hash_code(Path(__file__).resolve().parents[1])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("bank", "trace_store"):
        shutil.copytree(source / name, output / name)
    config = isolated_config(load_config(config_path), output)
    config["experiment"].update(task_manifest_path=None, initialize_v3_bank="existing")
    promotions = [p for row in read_json(source / "progress.json") for p in row["promotions"]]
    if not promotions:
        raise RuntimeError("Source acceptance produced no normal promoted Candidate")
    refs = promotions[-1]["refs"]
    rows = []
    with AtomicSkillGraphSystem(config) as system:
        parent = system.skills.get_atomic("skill://r10_parent_discovered_take@1.0.0")
        impls = [item.ref for item in system.skills.implementations_for(parent.ref, mode=system.mode)]
        tasks = {t.context["env_index"]: t for t in system.harness.load_tasks(limit=max(indexes) + 1)}
        for index in indexes:
            task, started = tasks[index], time.monotonic()
            status_before = [system.skills.get_atomic(refs[0]).status.value,
                             system.skills.get_implementation(refs[1]).status.value,
                             system.tools.get(refs[2]).status.value]
            automatic = all(status in {"active", "preferred"} for status in status_before)
            root = output / task.task_id
            root.mkdir()
            plan = route.make_plan(task, parent, impls, system.harness)
            plan.source, plan.source_composite_ref = "atomic_composition", None
            system.planner.build_plan = lambda *args, **kwargs: plan
            provider = ReuseEntryProvider(route.RouteCase(task.task_id), "runtime", [],
                delegate=system._provider("runtime_preparation"), audit=lambda stage, i, kind, payload:
                atomic_write_json(root / f"{stage}_{i:03d}_{kind}.json", payload))
            provider.automatic = automatic
            system._provider_override = {"runtime_preparation": provider, "runtime_seeded": provider,
                "tool_builder": system._provider("tool_builder"), "runtime_dynamic": system._provider("runtime_dynamic")}
            system._current_task_id = task.task_id
            usage_start, sessions_start = len(system.usage.events), len(system._observed_sessions)
            system._current_task_usage_start = usage_start
            trace = system.orchestrator.run_task(task)
            system._attach_external_sessions(trace, system._observed_sessions[sessions_start:])
            usage = system.usage.events[usage_start:]
            trace.llm_usage = [event.to_dict() for event in usage]
            trace.metadata["usage_reconciliation"] = _reconcile_events(usage)
            system._require_resource_usage_complete(trace)
            _require_formal_usage(usage, trace.agent_turns)
            refresh_learning_eligibility(trace)
            events = system.credit.assign(trace)
            trace.evidence_event_refs = [event.event_id for event in events]
            finalize(trace, config)
            system.traces.save_atomic(trace)
            system._commit_evidence(events)
            review = system.lifecycle.review(artifact_refs=refs)
            system._provider_override = None
            row = {"task_id": task.task_id, "trace_id": trace.trace_id, "status_before": status_before,
                "automatic_entry": automatic, "strict_success": trace.benchmark_success and trace.task_contract_success,
                "lifecycle_review": to_primitive(review), "r10_metrics": trace.metadata["r10_metrics"],
                "tool_builder_calls": sum(event.bucket.value == "tool_builder_runtime" for event in usage),
                "duration_seconds": time.monotonic() - started}
            rows.append(row)
            atomic_write_json(output / "progress.json", rows)
            print(row, flush=True)
        atomic_write_json(output / "summary.json", {"cases": rows, "refs": refs,
            "code_hash": start_hash, "code_unchanged": start_hash == hash_code(Path(__file__).resolve().parents[1]),
            "candidate_direct_deployments_not_r1_credit": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alfworld_train_full_120_r10_seed42.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-index", action="append", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.source, args.output, args.env_index)
