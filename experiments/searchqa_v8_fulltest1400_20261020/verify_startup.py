"""Read-only verification of the actual Test1400 prefix; no model requests."""
from collections import Counter
import json
import run_fulltest as run

c, r = run.c, run.r
c.check(run.STAGE)
root = run.ROOT / run.STAGE
tasks = {t["task_id"]: t for t in r.read(root / "tasks.json")["tasks"]}
audit = {}
for arm in run.ARMS:
    rows = c.completed(root, arm)
    assert len(rows) >= 3, "Wait for three completed tasks per arm"
    models = Counter()
    delivered = 0
    for key, row in rows.items():
        assert row["execution"]["all_authored_nodes_completed"]
        assert all(a["status"] == "ok" for a in row["execution"].get("attempts", []))
        req = row["requests"][0]
        payload = json.loads(req["messages"][1]["content"])
        assert payload["goal"] == tasks[key]["goal"] and payload["inputs"] == tasks[key]["inputs"]
        assert len(row["usage"]) == 1 and row["usage"][0]["model"] == "deepseek-v4-flash"
        models[row["usage"][0]["provider_metadata"]["response_model"]] += 1
        assert req["provider_finish_reason"] == "stop"
        for attempt in req["http_attempts"]:
            settings = attempt["final_payload_audit"]["serialized_parameters"]
            assert settings["model"] == "deepseek-v4-flash" and settings["max_tokens"] == 32768
            assert settings["reasoning_effort"] == "high" and settings["thinking"]["type"] == "enabled"
            assert attempt["usage_status"] == "reported"
        for path in (run.V8 / "guidance").glob("*.txt"):
            assert req["messages"][0]["content"].count(path.read_text().strip()) == 1
        mapping = req["node_response_mapping"]
        delivered += bool(mapping.get("source_fields_delivered"))
        assert ("source_fields" in payload) == bool(mapping.get("source_fields_delivered"))
    audit[arm] = {
        "completed": len(rows), "source_fields_delivered": delivered,
        "all_native_nodes_completed": True, "all_programs_ok": True,
        "model_calls_per_task": 1, "public_input_unchanged": True, "response_models": dict(models)}
assert audit["v8_programmatic"]["source_fields_delivered"] >= 1
result = {
    "passed": True, "at": r.utc(), "source": r.code_identity(),
    "plan_sha256": r.sha(root / "plan.json"), "arms": audit,
    "status": c.status(run.STAGE), "real_http_from_this_check": 0,
    "note": "Engineering prefix verification only; no accuracy conclusion."}
r.write_json(run.ROOT / "startup_verification.json", result)
r.write_json(run.HERE / "startup_verification.json", result)
print(json.dumps(result, ensure_ascii=False))
