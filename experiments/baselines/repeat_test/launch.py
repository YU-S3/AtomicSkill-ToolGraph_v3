"""Launch ten detached rounds, or qualify that same ten-process topology."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl, write_json
from experiments.baselines.common.manifest import sha256_json
from .authority import (REPO,METHODS,audit_source,qualification_identity,guard_environment,
                        file_hash,code_identity)
from .run import verify_qualification


def round_commands(root, *, smoke, qualification=None, methods=METHODS):
    commands=[]
    for method in methods:
        for repeat in (1,2):
            output=root/f"{method}_repeat_{repeat}"
            cmd=[sys.executable,"-m","experiments.baselines.repeat_test.run",
                 "--method",method,"--repeat",str(repeat),"--output",str(output)]
            if smoke:
                cmd.append("--smoke")
            else:
                cmd+=["--qualification",str(qualification)]
            commands.append((method,repeat,output,cmd))
    return commands


def launch_process(cmd,log_path):
    env=dict(os.environ,PYTHONPATH=f"{REPO}/src:{REPO}",PYTHONUNBUFFERED="1",
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             TOKENIZERS_PARALLELISM="false")
    with log_path.open("x") as log:
        return subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)


def memory():
    return {line.split(":")[0]:int(line.split(":")[1].split()[0])*1024
            for line in Path("/proc/meminfo").read_text().splitlines()}


def run(args):
    guard_environment()
    root=Path(args.output).absolute()
    audits={method:audit_source(method) for method in METHODS}
    identity=qualification_identity(audits)
    if args.mode=="formal":
        if not args.qualification:
            raise ValueError("Formal needs the passing ten-lane smoke receipt")
        for method,authority in audits.items():
            verify_qualification(Path(args.qualification).absolute(),method,authority,identity["code"])
    initial=memory()
    reserve=max(2*1024**3,int(initial["MemTotal"]*.15))
    if initial["MemAvailable"]<reserve:
        raise MemoryError("Not enough memory reserve for ten independent rounds")
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/"source_audit.json",dict(identity=identity,sources=audits,
        protocol="same_seed42_assets_two_additional_independent_inference_rounds",
        original_is_repetition_zero=True,formal_training_started=False))
    processes=[]
    launches=[]
    for method,repeat,output,cmd in round_commands(root,smoke=args.mode=="smoke",
            qualification=Path(args.qualification).absolute() if args.qualification else None):
        log=root/f"{method}_repeat_{repeat}.log"
        process=launch_process(cmd,log)
        processes.append(process)
        launches.append(dict(method=method,repeat=repeat,pid=process.pid,output=str(output),log=str(log),command=cmd))
        write_json(root/"launches.json",dict(launched=launches,independent=True,wait_for_siblings=False))
        print(f"Started {method} repeat={repeat} pid={process.pid} log={log}",flush=True)
    if args.mode=="formal":
        # No parent wait, sibling cancellation, shared provider gate or global
        # method lease. Each child owns final reporting and exits independently.
        return 0
    minimum=initial["MemAvailable"]
    while any(p.poll() is None for p in processes):
        minimum=min(minimum,memory()["MemAvailable"])
        time.sleep(2)
    passed=all(p.returncode==0 for p in processes)
    checks={}
    all_ids=[]
    for item,process in zip(launches,processes):
        out=Path(item["output"])
        report_path=out/"test_report.json"
        if process.returncode or not report_path.exists():
            checks[out.name]=False
            continue
        report=read_json(report_path)
        events=read_jsonl(out/"seed_42/test/provider_calls.jsonl")
        ids={e["logical_call_id"] for e in events}
        all_ids.extend(ids)
        checks[out.name]=bool(report["passed"] and report["engineering_smoke"]
            and report["frozen_unchanged"] and not report["training_performed"]
            and report["test"]["tasks"]==1 and events and all(e["role"]=="target" for e in events)
            and report["successful_response_usage"]["logical_calls"]>0)
    checks["distinct_requests_between_rounds"]=len(all_ids)==len(set(all_ids))
    checks["ten_rounds"]=len(launches)==10
    checks["memory_reserve"]=minimum>=reserve
    checks["identity_unchanged"]=qualification_identity({m:audit_source(m) for m in METHODS})==identity
    passed=passed and all(checks.values())
    write_json(root/"smoke_checks.json",dict(passed=passed,checks=checks,
        minimum_mem_available=minimum,reserve_bytes=reserve,exit_codes=[p.returncode for p in processes]))
    if passed:
        evidence={p.relative_to(root).as_posix():file_hash(p)
            for p in root.rglob("*") if p.is_file() and p.suffix in {".json",".jsonl"}}
        write_json(root/"smoke_qualification.json",dict(passed=True,identity=identity,
            independent_smoke_rounds=10,evidence_hashes=evidence))
    print(f"Ten-lane smoke passed={passed}: {root}",flush=True)
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","formal"))
    p.add_argument("--output",required=True)
    p.add_argument("--qualification")
    args=p.parse_args()
    try:
        return run(args)
    except Exception as exc:
        from experiments.baselines.common.source_identity import sanitize_error_text
        print(f"Launch stopped: {type(exc).__name__}: {sanitize_error_text(exc)}",flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
