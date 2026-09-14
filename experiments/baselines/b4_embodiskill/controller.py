"""Matched Train chunks, immutable checkpoints and readonly snapshot selection.

No upstream algorithm runs in this controller; the external worker invokes
TeamSolver/EmbodiSkill directly. Post-hoc contracts never flow back to it.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from experiments.baselines.b4_embodiskill import METHOD_ID
from experiments.baselines.b4_embodiskill.manifest_adapter import train_chunks
from experiments.baselines.b4_embodiskill.state import (copy_state, read_json, read_jsonl,
    write_json, load_checkpoint, publish_checkpoint)
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import freeze_files, FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.post_evaluator import TaskRow, summarize_rows, write_rows_jsonl, write_evaluated_episodes_jsonl


def usage(events):
    result = dict(physical_attempts=len(events), logical_calls=len({e["logical_call_id"] for e in events}),
        provider_retries=sum(e["attempt"] > 1 for e in events),
        failed_attempts=sum(e["status"] != "succeeded" for e in events),
        usage_complete=all(e.get("usage_status") == "reported" for e in events),
        api_cost=None, api_cost_unpriced=True)
    for role in ("target", "evolution"):
        rows = [e for e in events if e["role"] == role]
        bucket = {"calls": len({e["logical_call_id"] for e in rows}), "physical_attempts": len(rows)}
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            values = [e.get(key) for e in rows]
            bucket[key] = sum(values) if all(isinstance(v, int) and v >= 0 for v in values) else None
            bucket[key + "_known_subtotal"] = sum(v for v in values if isinstance(v, int) and v >= 0)
        result[role] = bucket
    return result


class WorkerFailure(RuntimeError):
    def __init__(self, message, failure_kind):
        super().__init__(message)
        self.failure_kind = failure_kind


class SeedController:
    def __init__(self, spec, seed, manifests):
        self.spec, self.seed, self.manifests = spec, seed, manifests
        self.root = Path(spec["output"]) / f"seed_{seed}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = spec["config"]

    def operation(self, name, phase, *, source=None, task=None, epoch=0, rate=0):
        receipt = load_checkpoint(self.root, name)
        if receipt:
            return receipt
        attempt = self.root / "attempts" / name / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        state = attempt / "state"
        before = copy_state(source, state)
        job = {**self.spec, "run_id": f'{self.spec["campaign_id"]}_seed{self.seed}',
            "run_seed": self.seed, "operation": name, "phase": phase, "epoch": epoch,
            "output": str(attempt), "state": str(state), "train_success_rate": rate,
            "task": task.to_dict() if task else {}}
        write_json(attempt / "job.json", job)
        environment = worker_environment(Path(self.spec["repo"]))
        started = time.monotonic()
        with (attempt / "worker.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run([self.spec["worker_python"], "-m",
                "experiments.baselines.b4_embodiskill.worker", "--job", str(attempt / "job.json")],
                cwd=self.spec["repo"], env=environment, stdout=log, stderr=subprocess.STDOUT)
        if source is not None and digest_directory(source) != before:
            raise RuntimeError("Worker changed its source snapshot")
        if completed.returncode or not (attempt / "result.json").exists():
            failure_path = attempt / "rollout_failure.json"
            failure = read_json(failure_path) if failure_path.exists() else {}
            kind = failure.get("failure_kind", "infrastructure_failure")
            write_json(self.root / "failure.json", dict(operation=name, phase=phase,
                attempt=str(attempt), exit_code=completed.returncode, failure_kind=kind,
                usage=usage(read_jsonl(attempt / "provider_calls.jsonl"))))
            raise WorkerFailure(f"B4 {name} worker failed; see {failure_path}", kind)
        result = read_json(attempt / "result.json")
        result["process_wall_time_ms"] = int((time.monotonic()-started)*1000)
        result["source_digest"] = before
        write_json(attempt / "result.json", result)
        if phase in {"test", "validation"}:
            events = read_jsonl(attempt / "method_events.jsonl")
            if any(e.get("method") in {"save_task_context", "reflect_episode", "revise_manual"} for e in events):
                raise RuntimeError("Readonly evaluation attempted a learning hook")
            # Only this task's disposable copy is removed; evidence remains flat in attempt.
            if state.parent != attempt or state.name != "state":
                raise RuntimeError("Unsafe disposable state target")
            shutil.rmtree(state)
            state = source
        receipt = publish_checkpoint(self.root, name, attempt, state)
        return receipt

    def evaluate(self, phase, tasks, source, epoch):
        workers = self.config["parallel"]["test_workers_per_seed" if phase == "test" else "episode_workers_per_seed"]
        def one(pair):
            index, task = pair
            return self.operation(f"{phase}_{epoch:02d}_{index:03d}", phase,
                source=source, task=task, epoch=epoch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, enumerate(tasks)))

    def run(self):
        train, val, test = self.manifests
        cfg = self.config
        manifest = dict(protocol="protocol-faithful-matched-train-v2.1", method=METHOD_ID,
            run_seed=self.seed, controller_commit=self.spec["git_state"]["commit"],
            external_repo=self.spec["source_receipt"]["repo"],
            external_commit=self.spec["source_receipt"]["declared_commit"],
            train_manifest_hash=train.digest, validation_manifest_hash=val.digest, test_manifest_hash=test.digest,
            alfworld_package_version=self.spec["identity"]["dependencies"]["distributions"]["alfworld"],
            alfworld_data_signature=hashlib.sha256("".join(m.digest for m in self.manifests).encode()).hexdigest(),
            model=cfg["model"]["model"], reasoning_effort=cfg["model"]["reasoning_effort"],
            max_environment_actions=cfg["max_environment_actions"], **cfg["parallel"],
            method_specific_alfworld_prior=True, num_epochs=cfg["train"]["num_epochs"],
            train_chunk_size=cfg["train"]["train_chunk_size"], validation_policy="epoch_snapshot_read_only",
            best_snapshot_metric="official_won_rate", team_solver=True, static_few_shots=1,
            embedding_model=cfg["embedding"]["model"], formal=cfg["experiment_kind"]=="formal")
        manifest_path = self.root / "run_manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise RuntimeError("Seed run manifest changed on resume")
        else:
            write_json(manifest_path, manifest)
        train_tasks = train.tasks[:cfg["train"]["train_size"]]
        val_tasks = val.tasks[:cfg["selection"]["validation_size"]]
        test_tasks = test.tasks[:cfg["selection"].get("test_size", len(test.tasks))]
        chunks = train_chunks(train_tasks, seed=self.seed, epochs=cfg["train"]["num_epochs"],
                              chunk_size=cfg["train"]["train_chunk_size"])
        current, best, best_rate, best_epoch = None, None, -1., None
        trains, validations, revisions, history = [], [], [], []
        for epoch, chunk in enumerate(chunks):
            epoch_rows = []
            for index, task in enumerate(chunk):
                receipt = self.operation(f"train_{epoch:02d}_{index:03d}", "train",
                    source=current, task=task, epoch=epoch)
                current = Path(receipt["state"])
                epoch_rows.append(receipt)
                trains.append(receipt)
                write_json(self.root / "progress.json", dict(phase="train", epoch=epoch,
                    train_completed=len(trains), train_total=len(train_tasks), task_id=task.task_id))
            revision = self.operation(f"revision_{epoch:02d}", "revision", source=current,
                epoch=epoch, rate=sum(r["result"]["official_success"] for r in epoch_rows)/len(epoch_rows))
            revisions.append(revision)
            current = Path(revision["state"])
            rows = self.evaluate("validation", val_tasks, current, epoch)
            validations.extend(rows)
            rate = sum(r["result"]["official_success"] for r in rows)/len(rows)
            if rate > best_rate:
                best, best_rate, best_epoch = current, rate, epoch
            history.append(dict(epoch=epoch, validation_official_rate=rate, best_epoch=best_epoch,
                snapshot_digest=revision["state_digest"], manual_version=revision["result"]["manual_after"]["version"]))
            write_json(self.root / "selection.json", history)
        frozen_dir = self.root / "frozen"
        if frozen_dir.exists():
            frozen = FrozenArtifact.load(frozen_dir)
            if frozen.metadata["best_epoch"] != best_epoch or frozen.metadata["state_digest"] != digest_directory(best):
                raise RuntimeError("Existing frozen snapshot disagrees with selection")
        else:
            provenance = self.root / "provenance"
            write_json(provenance / "config.json", cfg)
            write_json(provenance / "source.json", self.spec["source_receipt"])
            write_json(provenance / "campaign_lock.json", self.spec)
            write_json(provenance / "run_manifest.json", manifest)
            for name, manifest in zip(("train", "validation", "test"), self.manifests):
                write_json(provenance / f"{name}_manifest.json", manifest.to_dict())
            files = {"state/"+p.relative_to(best).as_posix(): p for p in best.rglob("*") if p.is_file()}
            files.update({"provenance/"+p.name: p for p in provenance.iterdir() if p.is_file()})
            pending_frozen = self.root / (".frozen_" + uuid.uuid4().hex)
            freeze_files(method_id=METHOD_ID, source_files=files, destination=pending_frozen,
                source_train_manifest_hash=train.digest, source_validation_manifest_hash=val.digest,
                metadata=dict(run_seed=self.seed, best_epoch=best_epoch, best_validation_rate=best_rate,
                    state_digest=digest_directory(best), source_run_id=self.spec["campaign_id"],
                    method_specific_alfworld_prior=True))
            FrozenArtifact.load(pending_frozen)
            os.replace(pending_frozen, frozen_dir)
            frozen = FrozenArtifact.load(frozen_dir)
        tests = self.evaluate("test", test_tasks, frozen.root / "state", best_epoch)
        assert_frozen_unchanged(frozen)
        summary = dict(passed=True, seed=self.seed, best_epoch=best_epoch, best_validation_rate=best_rate,
            train_episodes=len(trains), unique_train_tasks=len({r["result"]["task"]["task_id"] for r in trains}),
            validation_episodes=len(validations), test_episodes=len(tests),
            frozen_digest=frozen.digest, frozen_unchanged=True, selection=history,
            final_assets=revisions[-1]["result"], frozen_assets=read_json(best / "skill/current.json"))
        for phase, receipts in (("train", trains), ("validation", validations), ("test", tests)):
            summary[phase] = self.report(phase, receipts)
        events = [e for p in (self.root / "attempts").rglob("provider_calls.jsonl") for e in read_jsonl(p)]
        summary["all_attempts_usage"] = usage(events)
        summary["all_attempts_embedding_calls"] = sum(e["calls"]
            for p in (self.root/"attempts").rglob("embedding_calls.jsonl") for e in read_jsonl(p))
        summary["training_cost"] = dict(
            usage=usage([e for e in events if e["phase"] in {"train", "revision", "validation"}]),
            train_episodes=len(trains), validation_episodes=len(validations),
            environment_actions=sum(len(r["result"]["actions"]) for r in trains+validations),
            embedding_calls=sum(r["result"]["embedding_calls"] for r in trains+revisions+validations))
        summary["method_metrics"] = method_metrics(trains, revisions, validations, tests,
            best=best, best_epoch=best_epoch, best_rate=best_rate)
        summary["evolution_revision_usage"] = usage([e for r in revisions for e in read_jsonl(Path(r["attempt"]) / "provider_calls.jsonl")])
        summary["smoke_checks"] = smoke_checks(trains, revisions, validations, tests)
        if cfg["experiment_kind"] == "smoke":
            summary["passed"] = all(summary["smoke_checks"].values())
        write_json(self.root / "summary.json", summary)
        if summary["passed"]:
            write_json(self.root / "test_report.json", dict(passed=True, phase="test", method=METHOD_ID,
                run_id=f'{self.spec["campaign_id"]}_seed{self.seed}',
                test_cost=dict(api_cost=None, api_cost_unpriced=True), frozen_digest=frozen.digest))
            write_json(self.root / "completion.json", dict(passed=True, phase="test", report="test_report.json",
                run_id=f'{self.spec["campaign_id"]}_seed{self.seed}'))
        return summary

    def report(self, phase, receipts):
        from experiments.baselines.common.manifest import ManifestTask
        from experiments.baselines.common.task_authority import StrictTaskEvaluator
        evaluator = StrictTaskEvaluator(self.spec["alfworld_data"])
        records = []
        for receipt in receipts:
            r = receipt["result"]
            task = ManifestTask.from_dict(r["task"])
            cost = usage(read_jsonl(Path(receipt["attempt"]) / "provider_calls.jsonl"))
            flags = evaluator.evaluate(task, r["actions"], official_success=r["official_success"])
            if flags.replayed_terminal_won != r["official_success"] or flags.environment_actions != len(r["actions"]):
                raise RuntimeError("Posthoc replay differs from actual episode")
            record = CommonEpisodeRecord(method=METHOD_ID, phase=phase, run_seed=self.seed,
                task_id=task.task_id, task_type=task.task_type, manifest_index=task.index,
                gamefile=r["gamefile"], gamefile_hash=task.gamefile_sha256,
                official_success=r["official_success"], environment_actions=len(r["actions"]),
                invalid_actions=flags.invalid_actions, command_turns=len(r["actions"]),
                termination_reason=r["termination_reason"], embedding_calls=r["embedding_calls"],
                wall_time_ms=r["process_wall_time_ms"], artifact_digest_before=r["source_digest"] or "",
                artifact_digest_after=receipt["state_digest"] if phase == "train" else (r["source_digest"] or ""),
                method_metrics=dict(usage=cost, manual_version=r["manual_before"]["version"],
                    method_specific_alfworld_prior=True, trajectory_records=r["trajectory_records"],
                    attempt=receipt["attempt"]))
            record.set_posthoc_outcome(contract_consistency=flags.task_contract_success)
            for role in ("target", "evolution"):
                setattr(record, role+"_llm_calls", cost[role]["calls"])
                for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                    # Existing common row schema is integer-only. Keep explicit completeness
                    # and null authoritative totals alongside known subtotals, never claim exact zero.
                    setattr(record, role+"_"+key, cost[role][key+"_known_subtotal"])
            records.append(record)
        out = self.root / phase
        out.mkdir(exist_ok=True)
        for name in ("episodes.jsonl", "evaluated_common_episodes.jsonl"):
            destination = out / name
            if destination.exists():
                if read_jsonl(destination) != [r.to_dict() for r in records]:
                    raise RuntimeError("Existing report differs from immutable episode evidence")
            else:
                write_evaluated_episodes_jsonl(records, destination)
        rows = [TaskRow.from_episode(r) for r in records]
        write_rows_jsonl(rows, out / "task_rows.jsonl")
        result = summarize_rows(rows, task_types=sorted({r.task_type for r in records}))
        write_json(out / "summary.json", result)
        return result


def smoke_checks(trains, revisions, validations, tests):
    events = [e for r in trains+revisions for e in read_jsonl(Path(r["attempt"])/"method_events.jsonl")]
    exited = {e["method"] for e in events if e["event"] == "upstream_exit"}
    revision_calls = [e for r in revisions for e in read_jsonl(Path(r["attempt"])/"provider_calls.jsonl")]
    return dict(real_train=bool(trains), reflection_called="reflect_episode" in exited,
        manual_revision_called="revise_manual" in exited,
        manual_revision_provider_called=bool(revision_calls),
        manual_version_advanced=any(r["result"]["manual_after"]["version"] > r["result"]["manual_before"]["version"] for r in revisions),
        readonly_val=bool(validations) and all(not r["result"]["update_skill"] for r in validations),
        readonly_test=bool(tests) and all(not r["result"]["update_skill"] for r in tests),
        won_authority=all(
            isinstance(r["result"]["official_success"], bool)
            and r["result"]["official_success"] == bool(
                (read_jsonl(Path(r["attempt"])/"environment_actions.jsonl") or [{"won":False}])[-1]["won"])
            for r in trains+validations+tests))


def method_metrics(trains, revisions, validations, tests, *, best, best_epoch, best_rate):
    events = [e for r in trains+revisions+validations+tests
              for e in read_jsonl(Path(r["attempt"])/"method_events.jsonl")]
    reflections = [e.get("result") for e in events
                   if e.get("event") == "upstream_exit" and e.get("method") == "reflect_episode"]
    manual = read_json(best / "skill/current.json")
    calls = [e for r in trains+revisions+validations+tests
             for e in read_jsonl(Path(r["attempt"])/"provider_calls.jsonl")]
    return dict(manual_version_count=len(list(best.glob("skill/versions/skill_v*.json"))),
        manual_rule_count=sum(len(section.get("items", [])) for section in manual.get("sections", [])),
        execution_note_count=len(manual.get("execution_notes", [])),
        episode_reflection_count=len(reflections),
        reflection_type_counts={kind:sum((r.get("reflection_type") if r else "empty") == kind for r in reflections)
            for kind in ("S_NEW","S_BETTER","FAIL_SKILL","FAIL_EXECUTION","empty")},
        retrieved_success_trajectory_count=sum(len(e.get("retrieved_successful", [])) for e in events),
        stuck_recovery_call_count=len({e["logical_call_id"] for e in calls if e["stage"] == "stuck_recovery"}),
        best_epoch=best_epoch, best_validation_score=best_rate, best_snapshot_digest=digest_directory(best))


def worker_environment(repo):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(repo), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false", ANONYMIZED_TELEMETRY="False")
    # Own worker environment; never install Ours or import its facts into the method.
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_BASE_URL", None)
    return env
