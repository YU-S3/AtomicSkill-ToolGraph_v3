"""Fixed training-side R10.3 validation, not formal Train-120/Frozen-134 scores.

No task-family execution policy or forced provider choices are installed. The
dev16 learning run uses the original manifest/attempt/checkpoint boundaries.
Deployment comparisons share its readonly bank and current legal state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from atomic_skillgraph.core.serialization import atomic_create_json, atomic_write_json, to_primitive
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.knowledge.r103_protocol import METADATA, validate_config
from experiments.protocol import (RunManifest, RunState, ManifestStore, AttemptTraceLedger,
    TaskCheckpointStore, artifact_audit_snapshot, hash_code, hash_config,
    validate_deepseek_formal_llm, capture_execution_manifest, task_signature)
from experiments.run_v3_train import _task_manifests, ensure_task_manifest
from experiments.run_v3_r102_targeted import DEV_IDS, row_for as original_row_for

REPO = Path(__file__).resolve().parents[1]


def row_for(trace, elapsed):
    """Keep unreported reasoning/attempt usage distinct from measured zero."""
    row = original_row_for(trace, elapsed)
    from experiments.r103_metrics import trace_metrics
    row["r103_diagnostics"] = trace_metrics(trace)
    usage = trace.llm_usage
    reasoning_known = all(item.get("reasoning_tokens") is not None for item in usage)
    row["completion_tokens"] = sum(item["completion_tokens"] for item in usage)
    row["reasoning_unavailable_calls"] = sum(item.get("reasoning_tokens") is None for item in usage)
    if not reasoning_known:
        row["reasoning_tokens"] = None
        row["non_reasoning_completion_tokens"] = None
    requests = trace.provider_requests
    row["provider_attempts"] = len(requests)
    row["unknown_usage_requests"] = sum(item.usage_status != "reported" for item in requests)
    row["cost_scope"] = "recorded task usage including known retries; provider probe reported separately"
    return row


def declared_entries():
    reference = json.loads((REPO / "data/baseline_manifests/train_120.json").read_text(encoding="utf-8"))
    indexed = {int(item["task_id"].split("_")[2]): item for item in reference["tasks"]}
    return [indexed[number] for number in DEV_IDS]


def resolve_tasks(system, entries):
    tasks = {task.task_id: task for task in system.harness.load_tasks(limit=max(e["env_index"] for e in entries)+1)}
    selected = []
    for entry in entries:
        task = tasks[entry["task_id"]]
        digest = hashlib.sha256(Path(task.context["game_file"]).read_bytes()).hexdigest()
        if (task.context["env_index"] != entry["env_index"] or digest != entry["gamefile_sha256"]
                or task_signature(task) != entry["task_signature"]):
            raise RuntimeError("fixed training-side task identity differs from declared source")
        selected.append(task)
    return selected


def capture_usage(system, output):
    append = system.usage.append
    def persist(event):
        result = append(event)
        atomic_write_json(output / "all_usage.json", [e.to_dict() for e in system.usage.events])
        return result
    system.usage.append = persist


def capture_requests(system, output):
    """Observe the actual adapter payload; never alter messages/tools/caps."""
    import uuid
    factory = system._provider
    seen = set()
    def observed(stage):
        provider = factory(stage)
        if id(provider) in seen:
            return provider
        seen.add(id(provider))
        build, complete = provider._build_payload, provider.complete
        def payload(messages, tools):
            value = build(messages, tools)
            atomic_create_json(output / "provider_payloads" / f"{stage}_{uuid.uuid4().hex}.json",
                {"stage":stage,"payload":value,"payload_hash":hash_config(value),
                 "messages_hash":hash_config(value["messages"]),"tools_hash":hash_config(value.get("tools",[]))})
            return value
        def request(messages, *, tools=None):
            offset = provider.request_record_count
            try:
                return complete(messages, tools=tools)
            finally:
                atomic_create_json(output / "provider_attempts" / f"{stage}_{uuid.uuid4().hex}.json",
                    {"stage":stage,"attempts":list(provider.request_records_since(offset))})
        provider._build_payload, provider.complete = payload, request
        return provider
    system._provider = observed


def prepare(config_path, output):
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    validate_deepseek_formal_llm(config)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return config, output


def capability(config, output, code):
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    result = ensure_provider_capability(config, output_dir=output, code_hash=code,
        config_hash=hash_config(config), run_if_missing=True)
    capture_execution_manifest(REPO, output, hash_config(config), result)
    return result


def train_dev16(config_path, output, learning_condition="Full"):
    started = time.monotonic()
    config, output = prepare(config_path, output)
    run_id = "r103_dev16_" + output.name.replace("-", "_")
    config["data_dir"], config["trace_data_dir"] = str(output / "bank"), str(output)
    config["experiment"].update(name=run_id, phase="train", runtime_mode="online", freeze_skills=False,
        initialize_v3_bank="empty", output_dir=str(output), task_manifest_path=str(output / "task_manifest.json"),
        frozen_snapshot_dir=str(output / "frozen_bank"))
    # Formal method/attempt authority within this independent finite TRAIN
    # selection, not a formal benchmark score or an imported diagnostic bank.
    config["experiment"]["experiment_kind"] = "formal"
    if learning_condition != "Full":
        config["r103_learning_intervention"] = learning_condition
        config["experiment"]["experiment_kind"] = "learning_ablation"
    code, entries = hash_code(REPO), declared_entries()
    declared = {"kind":"finite_training_validation", "formal_experiment":False, "empty_bank":True,
        "declared_before_execution":True, "entries":entries, "code_hash":code, "config_hash":hash_config(config),
        "source_manifest":"data/baseline_manifests/train_120.json", "protocol":METADATA,
        "learning_condition":learning_condition}
    atomic_create_json(output / "declared_manifest.json", declared)
    atomic_create_json(output / "config.json", config)
    capability(config, output, code)
    rows = []
    with AtomicSkillGraphSystem(config) as system:
        capture_usage(system, output)
        capture_requests(system, output)
        initial = system.knowledge_digest()
        if artifact_audit_snapshot(system.database)["artifact_index"]["total"] != 0:
            raise RuntimeError("finite training validation requires an empty bank")
        tasks = resolve_tasks(system, entries)
        manifest = RunManifest.create(run_id=run_id, phase="train", config_hash=hash_config(config),
            code_commit=code, knowledge_digest=initial, tasks=_task_manifests(tasks,"train",initial),
            metadata={**METADATA,"repair_revision":"R10.3","formal_experiment":False,
                "experiment_role":"finite_training_validation", "learning_condition":learning_condition,
                "llm_config_hash":hash_config(config["llm"])})
        store = ManifestStore(output / "manifests", system.database)
        store.persist_before_run(manifest)
        ensure_task_manifest(output / "task_manifest.json", manifest)
        store.mark_run_state(run_id, RunState.RUNNING)
        ledger = AttemptTraceLedger(output / "attempt_history", output / "traces")
        checkpoint = TaskCheckpointStore(output / ".task_checkpoint", system.data_dir)
        for item, task in zip(manifest.tasks,tasks):
            if hash_code(REPO) != code:
                raise RuntimeError("code changed during fixed dev16; do not mix revisions")
            before = system.knowledge_digest()
            sequence = store.mark_task_running(run_id, item.task_id, max_attempts=1)
            attempt = ledger.begin(run_id=run_id,task_id=item.task_id,task_signature=item.task_signature,
                attempt_kind="task",sequence=sequence,
                expected_periodic_milestone=system.expected_periodic_maintenance_milestone_after_success())
            checkpoint.create(system.database, run_id=run_id,task_id=item.task_id,before_digest=before,
                              config_hash=manifest.config_hash,code_commit=code)
            tick = time.monotonic()
            try:
                trace = system.run_task(task,attempt_id=attempt.attempt_id)
                ledger.capture(attempt,reason="run_task_returned")
                row = row_for(trace,time.monotonic()-tick)
                row.update(knowledge_digest_before=before,knowledge_digest_after=system.knowledge_digest())
                artifact_audit_snapshot(system.database)
                if trace.infrastructure_failure or row["action_source_audit"]["unattributed_world_actions"]:
                    raise RuntimeError("dev16 infrastructure/action-source integrity failure")
                store.mark_task_completed(run_id,item.task_id,trace_id=trace.trace_id,result=row)
                checkpoint.clear()
                rows.append(row)
                atomic_write_json(output / "progress.json",rows)
                print(json.dumps({"completed":len(rows),"total":16,"task_id":task.task_id,
                    "official_won":row["official_won"],"total_tokens":row["total_tokens"],
                    "duration_seconds":row["duration_seconds"]}),flush=True)
            except Exception as exc:
                from experiments.protocol import sanitize_error_text
                store.mark_task_failed(run_id,item.task_id,infrastructure=True,
                    result={"type":type(exc).__name__,"error":sanitize_error_text(exc)})
                store.mark_run_state(run_id,RunState.INFRASTRUCTURE_FAILED)
                raise
        maintenance = system.run_maintenance(triggering_task_id=rows[-1]["task_id"],
            milestone="r103_dev16_final_batch",finalize_pending=True)
        if maintenance.pending_count:
            raise RuntimeError("finite training maintenance left pending repairs")
        artifact_audit_snapshot(system.database)
        final_digest = system.knowledge_digest()
        source = {"source_run_id":run_id,"source_code_commit":code,"source_config_hash":manifest.config_hash,
            "source_task_manifest_hash":manifest.task_manifest_hash,"source_final_knowledge_digest":final_digest,
            "source_llm_config_hash":hash_config(config["llm"]),"experiment_role":"finite_training_validation",
            "learning_condition":learning_condition,**METADATA}
        system.freeze(output / "frozen_bank",provenance=source)
        if system.knowledge_digest() != final_digest:
            raise RuntimeError("freeze mutated source knowledge")
        store.mark_run_state(run_id,RunState.COMPLETED)
        result = {"complete":True,"formal_experiment":False,"tasks":len(rows),
            "official_successes":sum(r["official_won"] for r in rows),"cases":rows,
            "wall_seconds":time.monotonic()-started,"code_hash":code,"knowledge_digest":final_digest,
            "maintenance":to_primitive(maintenance),"total_tokens":sum(e.usage.total_tokens for e in system.usage.events)}
        from experiments.r103_metrics import bank_metrics, aggregate
        result["bank_execution_support"] = bank_metrics(system.database)
        result["r103_diagnostics"] = aggregate(rows)
        atomic_create_json(output / "summary.json",result)
    return result


def deploy(config_path, output, source_run, condition, diagnostic_tasks=None):
    config, output = prepare(config_path, output)
    source_run = Path(source_run).expanduser().resolve()
    source = json.loads((source_run / "declared_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((source_run / "frozen_bank/freeze_manifest.json").read_text(encoding="utf-8"))
    source_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    code = hash_code(REPO)
    if source["code_hash"] != code or frozen["provenance"]["source_code_commit"] != code:
        raise RuntimeError("diagnostic code must match source training code")
    if hash_config(source_config["llm"]) != hash_config(config["llm"]):
        raise RuntimeError("deployment diagnosis cannot change source provider/resources")
    if not json.loads((source_run / "summary.json").read_text(encoding="utf-8")).get("complete"):
        raise RuntimeError("diagnostic source dev16 is incomplete")
    if source.get("learning_condition", "Full") != "Full":
        raise RuntimeError("paired Full deployment conditions cannot import a learning-ablation bank")
    entries = source["entries"]
    if diagnostic_tasks is not None:
        if not 1 <= diagnostic_tasks <= len(entries):
            raise ValueError("diagnostic task count must be a nonempty prefix of the declared dev set")
        entries = entries[:diagnostic_tasks]
    config["data_dir"],config["trace_data_dir"] = str(source_run / "frozen_bank"),str(output)
    config["cold_start"]["enabled"] = False
    config["experiment"].update(output_dir=str(output),runtime_mode="frozen",freeze_skills=True,
        phase="diagnostic",experiment_kind="diagnostic",allow_long_term_knowledge_writes=False,task_manifest_path=None)
    config["r103_interventions"] = {"condition":condition}
    from atomic_skillgraph.runtime.interventions import DeploymentIntervention
    mask = DeploymentIntervention.from_config(config)
    manifest = {"experiment_kind":"diagnostic","formal_experiment":False,"condition":mask.to_dict(),
        "source_run":str(source_run),"bank_digest":frozen["knowledge_digest"],"source_provenance":frozen["provenance"],
        "code_hash":code,"config_hash":hash_config(config),"task_entries":entries,
        "selection":"fixed declared dev prefix; never selected by observed outcome",
        "task_manifest_hash":frozen["provenance"]["source_task_manifest_hash"]}
    atomic_create_json(output / "intervention_manifest.json",manifest)
    atomic_create_json(output / "config.json",config)
    capability(config,output,code)
    rows,started = [],time.monotonic()
    with AtomicSkillGraphSystem(config,readonly=True) as system:
        capture_usage(system,output)
        capture_requests(system,output)
        if system.knowledge_digest() != frozen["knowledge_digest"]:
            raise RuntimeError("diagnostic source bank digest mismatch")
        for task in resolve_tasks(system,entries):
            if hash_code(REPO) != code:
                raise RuntimeError("code changed during paired deployment")
            tick = time.monotonic()
            trace = system.run_task(task)
            row = row_for(trace,time.monotonic()-tick)
            if system.knowledge_digest() != frozen["knowledge_digest"]:
                raise RuntimeError("diagnostic modified frozen knowledge")
            if not mask.programs and any(r.result.get("started") is True for r in trace.tool_executions):
                raise RuntimeError("disabled program channel started a Tool")
            if not mask.automatic_entry and row["automatic_graph_nodes"]:
                raise RuntimeError("disabled automatic entry executed a node")
            rows.append(row)
            atomic_write_json(output / "progress.json",rows)
            print(json.dumps({"condition":condition,"completed":len(rows),"task_id":task.task_id,
                              "official_won":row["official_won"]}),flush=True)
        result = {**manifest,"complete":True,"cases":rows,"wall_seconds":time.monotonic()-started,
                  "total_tokens":sum(e.usage.total_tokens for e in system.usage.events)}
        from experiments.r103_metrics import aggregate
        result["r103_diagnostics"] = aggregate(rows)
        atomic_create_json(output / "summary.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/alfworld_train_full_120_r103_seed42.yaml")
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=["dev16","deployment"],required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--condition",choices=["C11","C01","C10","C00","A0"],default="C11")
    parser.add_argument("--learning-condition",choices=["Full","L-identity-support","L-generalization"],default="Full")
    parser.add_argument("--diagnostic-tasks",type=int,help="Predeclare a fixed dev-prefix size for paired deployment only.")
    args = parser.parse_args()
    if args.mode == "dev16":
        if args.diagnostic_tasks is not None:
            parser.error("dev16 must run all sixteen declared tasks")
        train_dev16(args.config,args.output,args.learning_condition)
    elif not args.source_run:
        parser.error("deployment requires --source-run")
    else:
        deploy(args.config,args.output,args.source_run,args.condition,args.diagnostic_tasks)


if __name__ == "__main__":
    main()
