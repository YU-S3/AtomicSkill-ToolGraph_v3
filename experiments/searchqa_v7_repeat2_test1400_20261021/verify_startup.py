"""Inspect fresh paid prefixes without any additional model requests."""
from collections import Counter
import json
import run_repeats as run

c, r = run.c, run.r
c.check(run.STAGE)
root = run.ROOT / run.STAGE
reference = c.completed(run.REFERENCE / run.STAGE, "v7_text_compiler")
tasks = {t["task_id"]: t for t in r.read(root / "tasks.json")["tasks"]}
audit = {}
physical_ids = {}
for arm in run.ARMS:
    rows = c.completed(root, arm)
    assert len(rows) >= 3, "Wait for at least three completed tasks per repetition"
    models, fingerprints = Counter(), Counter()
    ids = set()
    for key, row in rows.items():
        assert row["execution"]["all_authored_nodes_completed"]
        assert all(a["status"] == "ok" for a in row["execution"].get("attempts", []))
        assert len(row["requests"]) == len(row["usage"]) == 1
        req, original = row["requests"][0], reference[key]["requests"][0]
        assert req["messages"] == original["messages"], "Actual visible messages changed"
        assert req.get("tools") == original.get("tools")
        payload = json.loads(req["messages"][1]["content"])
        assert payload["goal"] == tasks[key]["goal"] and payload["inputs"] == tasks[key]["inputs"]
        assert req["provider_finish_reason"] == "stop"
        assert row["usage"][0]["model"] == "deepseek-v4-flash"
        meta = row["usage"][0]["provider_metadata"]
        models[meta["response_model"]] += 1
        fingerprints[str(meta.get("system_fingerprint"))] += 1
        for attempt in req["http_attempts"]:
            params = attempt["final_payload_audit"]["serialized_parameters"]
            assert params["model"] == "deepseek-v4-flash" and params["max_tokens"] == 32768
            assert params["reasoning_effort"] == "high" and params["thinking"]["type"] == "enabled"
            assert attempt["usage_status"] == "reported"
            assert attempt["request_id"]
            ids.add(attempt["request_id"])
    physical_ids[arm] = ids
    audit[arm] = {
        "completed": len(rows), "actual_messages_match_original_v7": True,
        "all_native_nodes_completed": True, "all_programs_ok": True,
        "model_calls_per_task": 1, "public_input_unchanged": True,
        "response_models": dict(models), "system_fingerprints": dict(fingerprints)}
assert physical_ids[run.ARMS[0]].isdisjoint(physical_ids[run.ARMS[1]])
result = {
    "passed": True, "at": r.utc(), "source": r.code_identity(),
    "plan_sha256": r.sha(root / "plan.json"), "arms": audit,
    "independent_call_ids": True, "status": c.status(run.STAGE),
    "real_http_from_this_check": 0, "note": "Engineering prefix only, not an accuracy conclusion."}
r.write_json(run.ROOT / "startup_verification.json", result)
r.write_json(run.HERE / "startup_verification.json", result)
print(json.dumps(result, ensure_ascii=False))
