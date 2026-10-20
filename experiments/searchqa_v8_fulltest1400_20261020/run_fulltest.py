"""Full canonical Test1400 for the two unchanged, previously tested V8 Banks."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "searchqa_accuracy_v7_20261019"
V8 = HERE.parent / "searchqa_accuracy_v8_20261020"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
import campaign as c
from atomic_skillgraph.empirical.budget_governor import billing_blocked

r = c.r
PARENT = Path("/home/yangchengyu/searchqa_accuracy_v8_seed42_20261020")
REFERENCE = Path("/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020")
ROOT = Path("/home/yangchengyu/searchqa_v8_fulltest1400_seed42_20261020")
STAGE = "fulltest1400"
ARMS = ("v8_guidance", "v8_programmatic")
ALLOCATION = {"tokens": 16000000, "http": 6000}
c.CAMPAIGN = ROOT


def prepare():
    assert not ROOT.exists(), "Preserve existing runs; use check or explicit resume"
    previous = PARENT / "comparison192"
    assert r.read(previous / "result.json")["status"] == "completed"
    old_plan = r.read(previous / "plan.json")
    assert r.digest(old_plan) == r.read(previous / "plan_identity.json")["sha256"]
    assert old_plan["code"] == r.code_identity()
    for path, expected in old_plan["files"].items():
        assert r.sha(path) == expected, path
    receipt = r.read(V8 / "offline_acceptance.json")
    assert receipt["passed"] and receipt["driver_sha256"] == r.sha(c.__file__)
    assert receipt["controller_sha256"] == r.sha(V8 / "run_experiment.py")
    assert r.read(REFERENCE / STAGE / "result.json")["status"] == "completed"
    parent = r.read(PARENT / "budget.json")
    assert not billing_blocked(parent)
    carry = r.read(PARENT / "cost_carryforward_manifest.json")
    charge = sum(a.get("accounted_tokens", a["reserved_tokens"]) for a in parent["attempts"].values())
    limits = dict(token_limit=charge + ALLOCATION["tokens"],
                  request_limit=len(parent["attempts"]) + ALLOCATION["http"], finish_reserve=0)
    authority, reserve = carry["original_authority_limits"], carry["r3_remaining_reservation"]
    assert limits["token_limit"] + reserve["tokens"] <= authority["token_limit"]
    assert limits["request_limit"] + reserve["http"] <= authority["request_limit"]
    cfg = r.read(previous / ARMS[0] / "config.json")
    canonical = Path(cfg["manifest"])
    r.verify(canonical.parent.parent)
    public = r.read(r.DATASETS / "test.json")
    tasks = public["tasks"]
    assert len(tasks) == len({t["task_id"] for t in tasks}) == len({t["physical_key"] for t in tasks}) == 1400
    assert [t["task_id"] for t in tasks] == [t["task_id"] for t in r.read(canonical)["tasks"]]
    assert public["split_sha256"] == r.sha(canonical)
    assert public["authority_sha256"] == r.sha(canonical.parent.parent / "manifest.json")
    assert all(t["split"] == "test" for t in tasks)
    arms = {a: {"package": str(previous / a / "input"), "kind": "native"} for a in ARMS}
    for arm, item in arms.items():
        assert r.tree_identity(item["package"]) == old_plan["input_identity"][arm] == receipt["packages"][arm]
        assert r.tree_identity(previous / arm / "frozen_bank") == old_plan["frozen"][arm]
    ledger = deepcopy(parent)
    ledger["limits"] = limits
    for name in ("budget.json", "inherited_budget.json"):
        r.write_json(ROOT / name, ledger)
    r.write_json(ROOT / "authorization.json", {
        "status": "approved", "at": r.utc(),
        "user_instruction": "新结果试试test呢？局部题不能代表全部的整体效果",
        "scope": "Both unchanged V8 Banks, each full canonical Test1400; no training, edits or automatic promotion.",
        "allocation": ALLOCATION, "original_cumulative_authority_unchanged": True})
    r.write_json(ROOT / "cost_carryforward_manifest.json", {
        "parent": str(PARENT), "parent_budget_sha256": r.sha(PARENT / "budget.json"),
        "historical_charged_tokens": charge, "historical_http": len(parent["attempts"]),
        "allocation": ALLOCATION, "limits": limits, "original_authority_limits": authority,
        "r3_remaining_reservation": reserve,
        "old_unknown_reservation_preserved": ledger.get("inherited_reservation_recovery")})
    r.write_json(HERE / "spec.json", {
        "tasks": str(r.DATASETS / "test.json"), "arms": arms,
        "purpose": "Authorized full Test1400, V8 guidance versus V8 conditional Program. Same DeepSeek parameters, public input and scorer; canonical task order with seed42 arm interleaving. Post-Test development evaluation, not blind. Prior V7 full Test is a historical reference, not a concurrent arm."})
    c.prepare(STAGE, HERE / "spec.json")
    root = ROOT / STAGE
    plan = r.read(root / "plan.json")
    for arm in ARMS:
        path = root / arm / "config.json"
        config = r.read(path)
        old_config = r.read(previous / arm / "config.json")
        assert config["llm"] == old_config["llm"] and config["runtime"] == old_config["runtime"]
        assert config["budget"]["eval_task_tokens"] == 96000
        bundle = r.read(root / arm / "frozen_bank/imported_bank.json")
        assert bundle == r.read(previous / arm / "frozen_bank/imported_bank.json")
        assert plan["frozen"][arm] == old_plan["frozen"][arm]
        config["experiment"]["authored_bank_runtime"] = bundle["method_version"]
        r.write_json(path, config)
        plan["files"][str(path)] = r.sha(path)
    locked = [Path(__file__), V8 / "offline_acceptance.json", V8 / "run_experiment.py",
              canonical, canonical.parent.parent / "manifest.json", r.DATASETS / "test.json",
              PARENT / "budget.json", previous / "plan.json", previous / "result.json"]
    plan["files"].update({str(p): r.sha(p) for p in locked})
    plan.update(canonical_order=True, automatic_promotion=False,
                historical_reference=str(REFERENCE / STAGE / "v7_text_compiler"),
                known_limitation="Unchanged V8 field parser can return punctuation-only placeholders; evaluate as tested, do not hot-edit.",
                reporting="Report complete EM/F1, paired gains/losses, costs and field usefulness. Historical V7 comparison includes run variability.")
    r.write_json(root / "plan.json", plan)
    r.write_json(root / "plan_identity.json", {"sha256": r.digest(plan)})
    result = c.check(STAGE)
    r.write_json(ROOT / "preflight.json", {
        **result, "at": r.utc(), "real_http": 0, "allocation": ALLOCATION,
        "banks_match_comparison192": True, "llm_and_runtime_unchanged": True,
        "canonical_manifest_verified": True, "source_unchanged": True,
        "historical_charge": charge, "historical_http": len(parent["attempts"])})
    return result


def start(resume=False):
    with (ROOT / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.check(STAGE)
        if (ROOT / "pid").exists():
            pid = int((ROOT / "pid").read_text())
            proc = Path("/proc") / str(pid) / "cmdline"
            assert not proc.exists() or str(Path(__file__)) not in proc.read_text(), "Already active"
            assert resume, "Explicit --resume required"
        assert not (ROOT / STAGE / "result.json").exists() or resume
        assert not (ROOT / "STOP_AFTER_TASK").exists() and not (ROOT / STAGE / "STOP_AFTER_TASK").exists()
        cmd = [sys.executable, "-B", "-u", str(Path(__file__)), "run"]
        with (ROOT / "run.log").open("ab") as log:
            child = subprocess.Popen(cmd, cwd=r.SOURCE, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / "pid").write_text(str(child.pid) + "\n")
        launch = {"pid": child.pid, "at": r.utc(), "command": cmd, "resume": resume,
                  "plan_sha256": r.read(ROOT / STAGE / "plan_identity.json")["sha256"]}
        r.write_json(ROOT / ("resume_launch.json" if resume else "launch.json"), launch)
        time.sleep(2)
        assert child.poll() is None, "Inspect run.log"
        return launch


def monitor():
    value = c.status(STAGE)
    pid = int((ROOT / "pid").read_text()) if (ROOT / "pid").exists() else None
    proc = Path("/proc") / str(pid) / "cmdline"
    alive = bool(pid and proc.exists() and str(Path(__file__)) in proc.read_text())
    print(f"{value['at']} status={value['status']} process_alive={alive} pid={pid}")
    for arm, row in value["arms"].items():
        n = row["completed"]
        em = f"{row['correct']/n:.2%}" if n else "-"
        print(f"{arm}: {n}/1400 correct={row['correct']} EM={em} tokens={row['tokens']:,}"
              f" program_ok={row['program_ok']}/{row['program_calls']}")
    print(f"New tokens={value['campaign_tokens']:,}/{ALLOCATION['tokens']:,}; "
          f"HTTP={value['campaign_http']}/{ALLOCATION['http']}; "
          f"inflight_or_unmetered={value['inflight_or_unmetered']}")
    for name in ("current.json", "result.json"):
        path = ROOT / STAGE / name
        if path.exists():
            data = r.read(path)
            if name == "current.json":
                print("Current:", json.dumps(data))
            elif data.get("error"):
                print("Error:", json.dumps(data["error"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "check", "start", "run", "status"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        monitor()
    else:
        value = prepare() if args.action == "prepare" else c.check(STAGE) if args.action == "check" else start(args.resume) if args.action == "start" else c.execute(STAGE)
        print(json.dumps(value, ensure_ascii=False), flush=True)
