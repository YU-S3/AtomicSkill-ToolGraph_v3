import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from openai import OpenAI

from experiments.baselines.b4_embodiskill.campaign import load_config
from experiments.baselines.b4_embodiskill.controller import SeedController, WorkerFailure, usage
from experiments.baselines.b4_embodiskill.state import read_json, write_json
from experiments.baselines.common.model_client import AuditedChatClient, ProviderFailure


def good():
    return {"id": "response", "object": "chat.completion", "created": 0, "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "look"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                  "completion_tokens_details": {"reasoning_tokens": 18}}}


@pytest.mark.parametrize("bad", ["html", "json", "empty_choices", "null_message", "invalid_content"])
def test_real_sdk_malformed_envelope_retries_same_request(tmp_path, monkeypatch, bad):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if bad == "html":
                return httpx.Response(200, text="<html>upstream unavailable</html>", headers={"content-type": "text/html"})
            if bad == "json":
                return httpx.Response(200, json={"error": {"message": "upstream unavailable"}})
            body = good()
            if bad == "empty_choices":
                body["choices"] = []
            elif bad == "null_message":
                body["choices"][0]["message"] = None
            else:
                body["choices"][0]["message"]["content"] = {"bad": True}
            return httpx.Response(200, json=body)
        return httpx.Response(200, json=good())
    monkeypatch.setattr("experiments.baselines.common.model_client.time.sleep", lambda _: None)
    sdk = OpenAI(api_key="test-secret", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    path = tmp_path / "provider_calls.jsonl"
    client = AuditedChatClient(output=path, identity={}, model=load_config(False)["model"], client=sdk)
    assert client.chat(messages=[{"role": "user", "content": "next"}], stage="solver", role="target") == "look"
    assert requests[0] == requests[1] and requests[0]["max_tokens"] == 65536
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[0]["failure_code"] == "invalid_response" and rows[0]["retryable"]
    assert rows[0]["error_phase"] == "response_envelope"
    assert rows[0]["http_status"] == 200 and "response_body_excerpt" in rows[0]
    assert rows[0]["logical_call_id"] == rows[1]["logical_call_id"]
    assert rows[0]["provider_attempt_id"] != rows[1]["provider_attempt_id"]
    assert rows[1]["status"] == "succeeded"
    if bad not in {"html", "json"}:
        assert usage(rows)["target"]["completion_tokens"] == 40
    else:
        assert usage(rows)["target"]["completion_tokens"] is None
    sdk.close()


@pytest.mark.parametrize("kind,expected_count,retryable", [("503", 5, True), ("401", 1, False), ("bug", 1, False)])
def test_retry_boundary_and_error_evidence(tmp_path, monkeypatch, kind, expected_count, retryable):
    calls = []
    def handler(request):
        calls.append(request)
        if kind == "bug":
            raise AttributeError("local bug")
        return httpx.Response(int(kind), json={"error": {"message": "api_key=secret-placeholder"}})
    monkeypatch.setattr("experiments.baselines.common.model_client.time.sleep", lambda _: None)
    sdk = OpenAI(api_key="test-secret", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    # Test an actual local implementation bug outside the SDK separately from
    # HTTP errors (the SDK wraps exceptions inside httpx as connection errors).
    if kind == "bug":
        sdk = NS(chat=NS(completions=NS(create=lambda **_: (_ for _ in ()).throw(AttributeError("local bug")))))
    path = tmp_path / "provider_calls.jsonl"
    client = AuditedChatClient(output=path, identity={}, model=load_config(False)["model"], client=sdk)
    with pytest.raises(ProviderFailure) as error:
        client.chat(messages=[], stage="solver", role="target")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == expected_count and error.value.retryable == retryable
    assert "error_traceback" in rows[0] and "secret-placeholder" not in path.read_text()


def test_operation_recovery_restarts_committed_state_and_preserves_attempts(tmp_path, monkeypatch):
    controller = SeedController({"output": str(tmp_path), "config": load_config(True)}, 42, [])
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "manual.json", {"version": 1})
    outputs = []
    def worker(command, **kwargs):
        job = read_json(command[-1])
        output = Path(job["output"])
        outputs.append(output)
        assert read_json(Path(job["state"]) / "manual.json") == {"version": 1}
        if len(outputs) == 1:
            write_json(Path(job["state"]) / "manual.json", {"version": 999})
            write_json(output / "rollout_failure.json", {"failure_kind": "infrastructure_failure", "retryable": True})
            return NS(returncode=1)
        write_json(output / "result.json", {"official_success": False})
        return NS(returncode=0)
    controller.spec.update(repo=str(tmp_path), campaign_id="test", worker_python="python")
    monkeypatch.setattr("experiments.baselines.b4_embodiskill.controller.subprocess.run", worker)
    monkeypatch.setattr("experiments.baselines.b4_embodiskill.controller.time.sleep", lambda _: None)
    receipt = controller.operation("train_00_000", "train", source=source)
    assert len(outputs) == 2 and outputs[0] != outputs[1]
    assert receipt["result"]["official_success"] is False  # no retry of task failure
    assert (outputs[0] / "rollout_failure.json").exists()
    assert read_json(source / "manual.json") == {"version": 1}
    assert controller.operation("train_00_000", "train", source=source) == receipt
    assert len(outputs) == 2


def test_empty_model_evolution_does_not_expand_to_operation_recovery(tmp_path, monkeypatch):
    response = NS(choices=[NS(message=NS(content=""), finish_reason="stop")], usage=NS(
        prompt_tokens=10, completion_tokens=20, completion_tokens_details=NS(reasoning_tokens=20)))
    sdk = NS(chat=NS(completions=NS(create=lambda **_: response)))
    monkeypatch.setattr("experiments.baselines.common.model_client.time.sleep", lambda _: None)
    path = tmp_path / "provider_calls.jsonl"
    client = AuditedChatClient(output=path, identity={}, model=load_config(False)["model"], client=sdk)
    with pytest.raises(ProviderFailure) as error:
        client.chat(messages=[], stage="episode_reflection", role="evolution")
    assert not error.value.retryable
    assert len(path.read_text().splitlines()) == 5  # preserves original empty-response policy


@pytest.mark.parametrize("kind,retryable,count", [("infrastructure_failure", True, 3),
    ("infrastructure_failure", False, 1), ("protocol_failure", True, 1)])
def test_operation_recovery_is_bounded_and_never_retries_protocol(tmp_path, monkeypatch, kind, retryable, count):
    controller = SeedController({"output": str(tmp_path), "config": load_config(True)}, 42, [])
    calls = []
    def fail(*a, **k):
        calls.append(1)
        raise WorkerFailure("test", kind, retryable=retryable)
    monkeypatch.setattr(controller, "_operation_once", fail)
    monkeypatch.setattr("experiments.baselines.b4_embodiskill.controller.time.sleep", lambda _: None)
    with pytest.raises(WorkerFailure):
        controller.operation("operation", "train")
    assert len(calls) == count


def test_transport_resume_preserves_original_and_rejects_other_changes(tmp_path, monkeypatch):
    from experiments.baselines.b4_embodiskill import transport_repair as repair
    from experiments.baselines.common.manifest import sha256_json
    locked = dict(identity=dict(code_hash="old", config_hash="cfg", manifests=["tasks"]),
        git_state=dict(commit="old-commit"), config=load_config(False))
    locked["resolved_config_hash"] = sha256_json(locked["config"])
    current = copy.deepcopy(locked)
    current["identity"]["code_hash"] = "new"
    current["git_state"]["commit"] = "new-commit"
    original = copy.deepcopy(locked)
    changed = ["experiments/baselines/common/model_client.py"]
    monkeypatch.setattr(repair, "git", lambda repo, *a: b"" if a[0] == "status" else "\n".join(changed).encode())
    monkeypatch.setattr(repair, "historical_code_hash", lambda *a: "old")
    checks = []
    monkeypatch.setattr("experiments.baselines.b4_embodiskill.campaign.verify_smoke_qualification", lambda *a: checks.append(a))
    smoke = tmp_path / "smoke.json"
    write_json(smoke, {"passed": True})
    recovered = repair.prepare_transport_resume(tmp_path, tmp_path, locked, current, smoke)
    assert locked == original and recovered["identity"] == locked["identity"]
    assert recovered["git_state"] == locked["git_state"] and recovered["execution_identity"] == current["identity"]
    assert checks and recovered["transport_recovery"]["original_lock_sha256"] == sha256_json(locked)
    assert repair.prepare_transport_resume(tmp_path, tmp_path, locked, current, smoke) == recovered
    changed.append(".external/embodiskill/tasks/workflow/team/team.py")
    with pytest.raises(RuntimeError, match="outside audited"):
        repair.prepare_transport_resume(tmp_path, tmp_path, locked, current, smoke)
    changed.pop()
    current["identity"]["manifests"] = ["different"]
    with pytest.raises(RuntimeError, match="cannot change"):
        repair.prepare_transport_resume(tmp_path, tmp_path, locked, current, smoke)


def test_failed_evaluation_does_not_cancel_other_queued_tasks(tmp_path, monkeypatch):
    cfg = load_config(True)
    cfg["parallel"]["test_workers_per_seed"] = 1
    controller = SeedController({"output": str(tmp_path), "config": cfg}, 42, [])
    seen = []
    def operation(name, phase, **kwargs):
        seen.append(kwargs["task"])
        if kwargs["task"] == 0:
            raise WorkerFailure("offline", "infrastructure_failure", retryable=True)
        return kwargs["task"]
    monkeypatch.setattr(controller, "operation", operation)
    with pytest.raises(WorkerFailure):
        controller.evaluate("test", list(range(20)), tmp_path, 0)
    assert seen == list(range(20))
