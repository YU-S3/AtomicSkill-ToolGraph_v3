"""Two independent repetitions of the unchanged V7 full Test1400."""
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
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
import campaign as c
from atomic_skillgraph.empirical.budget_governor import billing_blocked

r = c.r
PARENT = Path("/home/yangchengyu/searchqa_accuracy_v8_seed42_20261020")
REFERENCE = Path("/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020")
SIBLING = Path("/home/yangchengyu/searchqa_v8_fulltest1400_seed42_20261020")
ROOT = Path("/home/yangchengyu/searchqa_v7_repeat2_test1400_seed42_20261021")
STAGE = "fulltest1400"
ARMS = ("v7_repeat_1", "v7_repeat_2")
ALLOCATION = {"tokens": 16000000, "http": 6000}
c.CAMPAIGN = ROOT


def prepare():
    assert not ROOT.exists(), "Preserve existing run; use check or explicit resume"
    reference = REFERENCE / STAGE
    original = reference / "v7_text_compiler"
    old_plan = r.read(reference / "plan.json")
    assert r.read(reference / "result.json")["status"] == "completed"
    assert r.digest(old_plan) == r.read(reference / "plan_identity.json")["sha256"]
    assert old_plan["code"] == r.code_identity()
    for path, expected in old_plan["files"].items():
        assert r.sha(path) == expected, path
    assert r.tree_identity(original / "input") == old_plan["input_identity"]["v7_text_compiler"]
    assert r.tree_identity(original / "frozen_bank") == old_plan["frozen"]["v7_text_compiler"]
    reference_rows = c.completed(reference, "v7_text_compiler")
    assert len(reference_rows) == 1400
    assert sum(bool(row["score"]["hard"]) for row in reference_rows.values()) == 1190
    receipts = [PRIOR / name for name in ("offline_acceptance.json", "text_compiler_acceptance.json")]
    for path in receipts:
        receipt = r.read(path)
        assert receipt["passed"] and receipt["driver_sha256"] == r.sha(c.__file__)
    assert r.read(PARENT / "comparison192/result.json")["status"] == "completed"
    parent = r.read(PARENT / "budget.json")
    assert not billing_blocked(parent)
    carry = r.read(PARENT / "cost_carryforward_manifest.json")
    sibling_carry = r.read(SIBLING / "cost_carryforward_manifest.json")
    assert sibling_carry["parent"] == str(PARENT)
    assert sibling_carry["parent_budget_sha256"] == r.sha(PARENT / "budget.json")
    sibling_reserved = r.read(SIBLING / "authorization.json")["allocation"]
    assert sibling_reserved == sibling_carry["allocation"]
    assert sibling_carry["limits"] == r.read(SIBLING / "budget.json")["limits"]
    charge = sum(a.get("accounted_tokens", a["reserved_tokens"]) for a in parent["attempts"].values())
    limits = dict(token_limit=charge + ALLOCATION["tokens"],
                  request_limit=len(parent["attempts"]) + ALLOCATION["http"], finish_reserve=0)
    authority, r3 = carry["original_authority_limits"], carry["r3_remaining_reservation"]
    assert limits["token_limit"] + r3["tokens"] + sibling_reserved["tokens"] <= authority["token_limit"]
    assert limits["request_limit"] + r3["http"] + sibling_reserved["http"] <= authority["request_limit"]
    old_config = r.read(original / "config.json")
    canonical = Path(old_config["manifest"])
    r.verify(canonical.parent.parent)
    public = r.read(r.DATASETS / "test.json")
    tasks = public["tasks"]
    assert len(tasks) == len({t["task_id"] for t in tasks}) == len({t["physical_key"] for t in tasks}) == 1400
    assert [t["task_id"] for t in tasks] == [t["task_id"] for t in r.read(canonical)["tasks"]]
    assert public["split_sha256"] == r.sha(canonical)
    assert public["authority_sha256"] == r.sha(canonical.parent.parent / "manifest.json")
    assert tasks == r.read(reference / "tasks.json")["tasks"]
    ledger = deepcopy(parent)
    ledger["limits"] = limits
    for name in ("budget.json", "inherited_budget.json"):
        r.write_json(ROOT / name, ledger)
    r.write_json(ROOT / "authorization.json", {
        "status": "approved", "at": r.utc(), "user_instruction": "V7再启动两次重复的test",
        "scope": "Two fresh, independent full Test1400 repetitions of the V7 Bank that scored 1190/1400. Same model, input, prompt, parameters and frozen source. No training or Bank edits.",
        "allocation": ALLOCATION, "original_cumulative_authority_unchanged": True})
    r.write_json(ROOT / "cost_carryforward_manifest.json", {
        "parent": str(PARENT), "parent_budget_sha256": r.sha(PARENT / "budget.json"),
        "historical_charged_tokens": charge, "historical_http": len(parent["attempts"]),
        "allocation": ALLOCATION, "limits": limits, "original_authority_limits": authority,
        "r3_remaining_reservation": r3,
        "concurrent_sibling": {
            "root": str(SIBLING), "allocation_reserved_in_full": sibling_reserved,
            "authorization_sha256": r.sha(SIBLING / "authorization.json"),
            "accounting": "Stable common-ancestor charges inherited once. The entire V8 allocation is reserved separately, not fabricated as measured usage. Combine unique physical attempts for final cumulative totals."},
        "authority_envelope": {
            "tokens": limits["token_limit"] + r3["tokens"] + sibling_reserved["tokens"],
            "http": limits["request_limit"] + r3["http"] + sibling_reserved["http"]},
        "old_unknown_reservation_preserved": ledger.get("inherited_reservation_recovery")})
    arms = {a: {"package": str(original / "input"), "kind": "native"} for a in ARMS}
    r.write_json(HERE / "spec.json", {
        "tasks": str(r.DATASETS / "test.json"), "arms": arms,
        "purpose": "Two independent full Test1400 repetitions of identical V7. Canonical task order, seed42 interleaving, separate requests/results/Frozen copies. Same DeepSeek settings; generation seed unsupported. Measure EM/F1/cost and per-task prediction/correctness flips against both repetitions and original1190/1400, without outcome-based edits."})
    c.prepare(STAGE, HERE / "spec.json")
    root = ROOT / STAGE
    plan = r.read(root / "plan.json")
    for index, arm in enumerate(ARMS, 1):
        path = root / arm / "config.json"
        config = r.read(path)
        assert config["llm"] == old_config["llm"] and config["runtime"] == old_config["runtime"]
        assert config["budget"]["eval_task_tokens"] == 96000
        assert config["experiment"]["seed"] == old_config["experiment"]["seed"] == 42
        assert plan["input_identity"][arm] == old_plan["input_identity"]["v7_text_compiler"]
        assert plan["frozen"][arm] == old_plan["frozen"]["v7_text_compiler"]
        bundle = r.read(root / arm / "frozen_bank/imported_bank.json")
        assert bundle == r.read(original / "frozen_bank/imported_bank.json")
        config["experiment"]["authored_bank_runtime"] = bundle["method_version"]
        r.write_json(path, config)
        plan["files"][str(path)] = r.sha(path)
    locked = [Path(__file__), *receipts, canonical, canonical.parent.parent / "manifest.json",
              r.DATASETS / "test.json", PARENT / "budget.json", reference / "plan.json",
              reference / "result.json", original / "config.json",
              SIBLING / "authorization.json", SIBLING / "cost_carryforward_manifest.json"]
    plan["files"].update({str(path): r.sha(path) for path in locked})
    plan.update(canonical_order=True, automatic_promotion=False, reference=str(original),
                reference_correct=1190, repetitions=2, same_seed_and_parameters=True,
                generation_seed_supported=False, learned_assets_mutable=False)
    r.write_json(root / "plan.json", plan)
    r.write_json(root / "plan_identity.json", {"sha256": r.digest(plan)})
    result = c.check(STAGE)
    r.write_json(ROOT / "preflight.json", {
        **result, "at": r.utc(), "real_http": 0, "allocation": ALLOCATION,
        "identical_original_v7_bank": True, "identical_model_and_runtime": True,
        "canonical_manifest_verified": True, "source_unchanged": True,
        "both_repetitions_have_empty_task_storage": all(not (root / a / "test").exists() for a in ARMS),
        "concurrent_v8_fully_reserved": sibling_reserved})
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
