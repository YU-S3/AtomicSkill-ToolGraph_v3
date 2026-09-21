"""One independently resumable test repetition; no training entry point."""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl, write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.manifest import TaskManifestSet, sha256_json
from experiments.baselines.common.formal_validation import verify_formal_manifest
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.post_evaluator import (
    TaskRow, summarize_rows, write_rows_jsonl, write_evaluated_episodes_jsonl, _write_jsonl_atomic)
from experiments.baselines.common.source_identity import sanitize_error_text
from .authority import (REPO, METHODS, audit_source, assert_source_unchanged, file_hash,
                        code_identity, guard_environment)


@contextmanager
def round_lease(root):
    # Deliberately NOT the single-method campaign lease: user explicitly chose
    # independent cross-method repeated inference. Only duplicate same-run
    # controllers are excluded. A sibling's failure/exit never cancels this run.
    with (root/".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This repetition is already running") from exc
        try:
            yield
        except Exception as exc:
            write_json(root/f"failure_{time.time_ns()}.json",dict(passed=False,
                error_type=type(exc).__name__,error=sanitize_error_text(exc)))
            raise
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def successful_usage(events, *, b4=False):
    """Accepted target responses only, distinct from total physical billing."""
    successes=[e for e in events if e["status"]=="succeeded" and e["role"]=="target"]
    keys=("prompt_tokens","completion_tokens","reasoning_tokens")
    totals={k:0 for k in keys}
    ids=set()
    for event in successes:
        call_id=event["logical_call_id"]
        if call_id in ids:
            raise ValueError("Duplicate successful logical response")
        ids.add(call_id)
        physical=event.get("physical_attempt_usage")
        accepted=physical[-1] if physical else event
        for key in keys:
            value=accepted.get(key)
            if not isinstance(value,int) or value<0:
                raise ValueError("Accepted response has unknown token usage")
            totals[key]+=value
    totals.update(logical_calls=len(ids), total_tokens=totals["prompt_tokens"]+totals["completion_tokens"],
                  scope="successful_target_responses_excluding_infrastructure_failures",
                  reasoning_tokens_in_completion=True)
    return totals


def tasks_for_mode(mode):
    name="train_120" if mode=="smoke" else "test_ood_full_134"
    manifest=TaskManifestSet.load(REPO/f"data/baseline_manifests/{name}.json")
    verify_formal_manifest(manifest,alfworld_data=os.environ["ALFWORLD_DATA"],
        role="train" if mode=="smoke" else "test",profile="formal_v2")
    # Same independent engineering task in both smoke repetitions; never a
    # formal Test task and never counted as an additional formal measurement.
    return manifest, list(manifest.tasks[:1] if mode=="smoke" else manifest.tasks)


def copy_frozen(authority, lane):
    original=Path(authority["original_frozen"])
    target=lane/"frozen"
    if not target.exists():
        pending=lane/f".frozen_pending_{os.getpid()}_{time.time_ns()}"
        shutil.copytree(original,pending)
        if FrozenArtifact.load(pending).digest!=authority["frozen_digest"]:
            raise ValueError("Frozen copy differs from original")
        os.replace(pending,target)
    frozen=FrozenArtifact.load(target)
    if frozen.digest!=authority["frozen_digest"] or frozen.method_id!=authority["method"]:
        raise ValueError("Existing repetition frozen asset changed")
    return frozen


def verify_qualification(path, method, authority, code):
    receipt=read_json(path)
    if (not receipt.get("passed") or receipt["identity"]["code"]!=code
            or receipt["identity"]["sources"].get(method)!=sha256_json(authority)
            or receipt.get("independent_smoke_rounds")!=10):
        raise ValueError("Ten-lane smoke qualification missing or source/config changed")
    for relative,expected in receipt["evidence_hashes"].items():
        target=(Path(path).parent/relative).resolve()
        target.relative_to(Path(path).parent.resolve())
        if file_hash(target)!=expected:
            raise ValueError("Smoke qualification evidence changed")


def worker_environment(authority):
    env=dict(os.environ)
    env.update(PYTHONPATH=f'{REPO}/src:{REPO}:{authority["source"]}',
        PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    return env


def text_evaluate(spec, lane, tasks, frozen):
    from experiments.baselines.b0_dynamic.campaign import load_record
    authority=spec["authority"]
    for index,task in enumerate(tasks):
        episode=lane/"test/episodes"/f"task_{task.index:04d}"
        if (episode/"completion.json").exists():
            record=load_record(episode,task,42,frozen.digest,"test")
            if record.method!=authority["method"]:
                raise ValueError("Completed record belongs to another method")
        else:
            episode.mkdir(parents=True,exist_ok=True)
            job=dict(authority=authority,repeat=spec["repeat"],episode_dir=str(episode),
                task=task.to_dict(),frozen=str(lane/"frozen"),source=authority["source"],
                seed=42,phase="test",alfworld_data=spec["alfworld_data"],
                gate=str(Path(spec["output"])/"provider_gate"),cap=1,campaign_id=spec["run_id"])
            write_json(episode/"job.json",job)
            with (episode/f"worker_{time.time_ns()}.log").open("w") as log:
                result=subprocess.run([authority["python"],"-m","experiments.baselines.repeat_test.text_worker",
                    "--job",str(episode/"job.json")],cwd=REPO,env=worker_environment(authority),
                    stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Episode worker failed; resume this round after fixing infra: {episode}")
            record=load_record(episode,task,42,frozen.digest,"test")
        write_json(Path(spec["output"])/"progress.json",dict(
            method=authority["method"],repeat=spec["repeat"],completed=index+1,total=len(tasks)))
        print(f'{authority["method"]} repeat={spec["repeat"]} {index+1}/{len(tasks)} won={record.official_success}',flush=True)
    records=[load_record(lane/"test/episodes"/f"task_{t.index:04d}",t,42,frozen.digest,"test") for t in tasks]
    target=lane/"test"
    path=target/"evaluated_common_episodes.jsonl"
    if not path.exists():
        write_evaluated_episodes_jsonl(records,path)
    rows=[TaskRow.from_episode(r) for r in records]
    write_rows_jsonl(rows,target/"task_rows.jsonl")
    events=[e for p in sorted(target.glob("episodes/*/attempts/*/provider_calls.jsonl")) for e in read_jsonl(p)]
    _write_jsonl_atomic(target/"provider_calls.jsonl",events,overwrite=True)
    return summarize_rows(rows,task_types=list(dict.fromkeys(t.task_type for t in tasks))),events


def b4_evaluate(spec,lane,tasks,frozen):
    from experiments.baselines.b4_embodiskill.controller import SeedController
    authority=spec["authority"]
    b4=copy.deepcopy(authority["b4_spec"])
    b4.update(repo=str(REPO),output=spec["output"],campaign_id=spec["run_id"],
        gate_dir=str(Path(spec["output"])/"provider_gate"),provider_cap=1,
        alfworld_data=spec["alfworld_data"],
        report_recovery_progress=str(Path(spec["output"])/"replay_progress.json"))
    controller=SeedController(b4,42,())
    receipts=[]
    epoch=frozen.metadata["best_epoch"]
    for index,task in enumerate(tasks):
        receipt=controller.operation(f"test_{epoch:02d}_{index:03d}","test",
            source=frozen.root/"state",task=task,epoch=epoch)
        receipts.append(receipt)
        assert_frozen_unchanged(frozen)
        write_json(Path(spec["output"])/"progress.json",dict(
            method=authority["method"],repeat=spec["repeat"],completed=index+1,total=len(tasks)))
        print(f'B4 repeat={spec["repeat"]} {index+1}/{len(tasks)} won={receipt["result"]["official_success"]}',flush=True)
    summary=controller.report("test",receipts)
    events=[e for p in sorted((lane/"attempts").glob("*/*/provider_calls.jsonl")) for e in read_jsonl(p)]
    _write_jsonl_atomic(lane/"test/provider_calls.jsonl",events,overwrite=True)
    return summary,events


def execute(args):
    guard_environment()
    authority=audit_source(args.method)
    code=code_identity()
    root=Path(args.output).absolute()
    if args.repeat not in (1,2):
        raise ValueError("Only two additional rounds are requested")
    mode="smoke" if args.smoke else "test"
    manifest,tasks=tasks_for_mode(mode)
    if mode=="test":
        if not args.qualification:
            raise ValueError("Formal repeat requires a ten-lane smoke qualification")
        verify_qualification(Path(args.qualification).absolute(),args.method,authority,code)
    if args.resume:
        if not root.is_dir():
            raise ValueError("Resume requires an existing round")
    else:
        root.mkdir(parents=True,exist_ok=False)
    with round_lease(root):
        spec=dict(authority=authority,code=code,repeat=args.repeat,seed=42,mode=mode,output=str(root),
            run_id=f"{root.parent.name}_{root.name}",alfworld_data=str(Path(os.environ["ALFWORLD_DATA"]).resolve()),
            task_manifest_digest=manifest.digest,task_ids=[t.task_id for t in tasks])
        if args.resume:
            if read_json(root/"repeat_lock.json")!=spec:
                raise ValueError("Repeat source/config/runtime/data or output path changed")
            if (root/"completion.json").exists():
                completion=read_json(root/"completion.json")
                if file_hash(root/"test_report.json")!=completion["report_sha256"]:
                    raise ValueError("Completed report changed")
                if digest_directory(root/"seed_42")!=completion["evidence_digest"]:
                    raise ValueError("Completed repeat evidence changed")
                assert_source_unchanged(authority)
                print("Already completed; no model requests",flush=True)
                return 0
        else:
            write_json(root/"repeat_lock.json",spec)
        lane=root/"seed_42"
        lane.mkdir(exist_ok=True)
        frozen=copy_frozen(authority,lane)
        write_json(lane/"run_manifest.json",dict(method=args.method,run_seed=42,repeat=args.repeat,
            test_manifest_hash=manifest.digest,frozen_digest=frozen.digest,identity=spec,
            training_performed=False,model=authority["model"]))
        started=time.monotonic()
        summary,events=(b4_evaluate if args.method=="b4_embodiskill" else text_evaluate)(spec,lane,tasks,frozen)
        if any(e["role"]!="target" for e in events):
            raise ValueError("Frozen test invoked learning/evolution")
        if summary["tasks"]!=len(tasks) or summary["infrastructure_failed_episodes"]:
            raise ValueError("Repeat is not manifest-complete")
        if code_identity()!=code:
            raise ValueError("Source changed during repeat")
        assert_frozen_unchanged(frozen)
        assert_source_unchanged(authority)
        if audit_source(args.method)!=authority:
            raise ValueError("Historical source or dependency changed during repeat")
        from experiments.baselines.b0_dynamic.driver import usage_totals
        from experiments.baselines.b4_embodiskill.controller import usage
        total=usage(events) if args.method=="b4_embodiskill" else usage_totals(events)
        effective=successful_usage(events)
        report=dict(passed=True,method=args.method,repeat=args.repeat,run_seed=42,phase="test",
            engineering_smoke=args.smoke,test=summary,successful_response_usage=effective,
            all_attempt_usage=total,frozen_digest=frozen.digest,frozen_unchanged=True,
            original_result=authority["original_result"],original_test=authority["original_test"],
            training_performed=False,duration_seconds=time.monotonic()-started,
            test_cost=dict(api_cost=None,api_cost_unpriced=True))
        write_json(root/"test_report.json",report)
        write_json(lane/"test_report.json",report)
        lines=[f"# {args.method} seed42 repeat {args.repeat}","",
            f"Completed: {summary['tasks']} tasks",
            f"Official: {summary['official_success']}/{summary['tasks']}",
            f"Strict: {summary['common_strict_success']}/{summary['tasks']}",
            f"Successful-response tokens/task: {effective['total_tokens']/len(tasks):.2f}",
            "Frozen asset unchanged; no training or validation selection.",
            "This is an inference repetition of seed42, not a new training seed."]
        (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        from experiments.baselines.common.integrity import assert_no_secrets_on_disk
        assert_no_secrets_on_disk(root,api_key_env=authority["model"]["api_key_env"])
        write_json(root/"completion.json",dict(passed=True,report_sha256=file_hash(root/"test_report.json"),
            evidence_digest=digest_directory(lane)))
        print(f"Complete: {root}",flush=True)
        return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=METHODS,required=True)
    p.add_argument("--repeat",type=int,choices=(1,2),required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--qualification")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args(argv)
    try:
        return execute(args)
    except Exception as exc:
        # Do not write into an existing live round before its lease is held.
        print(f"Repeat stopped: {type(exc).__name__}: {sanitize_error_text(exc)}",flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
