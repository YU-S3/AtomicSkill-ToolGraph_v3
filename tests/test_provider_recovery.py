import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from atomic_skillgraph.agents.provider import AgentProviderError, OpenAICompatibleConfig, OpenAICompatibleProvider
from atomic_skillgraph.agents.provider_audit import decision_usage_auditable
from atomic_skillgraph.core.serialization import atomic_write_json
from experiments.provider_recovery import (RELEASE_FILE, RECEIPT_FILE, release_inventory_hash,
    prepare_recovery, read_recovery, checkpoint_identity)
from experiments.protocol import ProtocolError, _file_hashes, hash_code
from experiments.report import validate_formal_usage


GOOD = {"choices": [{"finish_reason": "stop", "message": {"content": "ok", "reasoning_content": "reason"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.ok = status == 200
        self.headers = {"x-request-id": "same-upstream-request-id"}
        self.text = "untrusted secret fixture-key"
        self.content = self.text.encode()

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def provider(monkeypatch, responses, max_retries=2):
    monkeypatch.setenv("RECOVERY_KEY", "fixture-key")
    queue = iter(responses)
    calls = []
    def post(*args, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        value = next(queue)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr("atomic_skillgraph.agents.provider.requests.post", post)
    p = OpenAICompatibleProvider(OpenAICompatibleConfig(
        base_url="https://api.deepseek.com", model="deepseek-v4-flash", api_key_env="RECOVERY_KEY",
        max_completion_tokens=32768, max_retries=max_retries, retry_backoff_seconds=0, max_retry_after_seconds=0))
    return p, calls


@pytest.mark.parametrize("bad", [None, [], "", ValueError("secret fixture-key"), {}, {"choices": []}])
def test_bad_envelope_retries_without_changing_prompt_or_fabricating_usage(monkeypatch, bad):
    p,calls = provider(monkeypatch, [Response(bad), Response(GOOD)])
    result = p.complete([{"role": "user", "content": "request"}])
    assert result.total_tokens == 3
    assert len(calls) == 2 and calls[0]["json"] == calls[1]["json"]
    first, last = p.request_records
    assert first["usage"] is None and first["usage_status"] == "unavailable"
    assert last["usage"]["total_tokens"] == 3 and last["retry_count"] == 1
    assert first["request_id"] != last["request_id"]
    assert "fixture-key" not in json.dumps(p.request_records)
    assert first["response_diagnostic"]["body_sha256"]


def test_mixed_transport_and_envelope_failures_share_retry_limit(monkeypatch):
    p,calls = provider(monkeypatch, [requests.exceptions.Timeout("timeout"), Response(None), Response([])])
    with pytest.raises(AgentProviderError, match="JSON object"):
        p.complete([{"role": "user", "content": "request"}])
    assert len(calls) == len(p.request_records) == 3
    assert [r["retry_count"] for r in p.request_records] == [0,1,2]


def test_auth_and_metered_model_errors_are_not_silently_retried(monkeypatch):
    p,calls = provider(monkeypatch, [Response({}, 401)])
    with pytest.raises(AgentProviderError):
        p.complete([{"role": "user", "content": "request"}])
    assert len(calls) == 1
    p,calls = provider(monkeypatch, [Response({"usage": GOOD["usage"], "choices": []})])
    with pytest.raises(AgentProviderError) as exc:
        p.complete([{"role": "user", "content": "request"}])
    assert len(calls) == 1 and exc.value.usage_turn.total_tokens == 3


def error_trace():
    return {"trace_id": "failed_http", "task": {"task_id": "t"}, "resource_usage_complete": False,
        "infrastructure_failure": True, "provider_requests": [{"request_id": "r1", "outcome": "error",
        "usage_status": "unavailable", "error_code": "provider_invalid_response", "http_status": 200,
        "payload_fingerprint": "hash", "started_at": 1.0, "ended_at": 2.0}], "llm_usage": [], "agent_turns": []}


def test_infra_uncertainty_is_reported_as_lower_bound_not_zero_or_complete():
    trace = error_trace()
    result = validate_formal_usage([trace])
    assert result["provider_unknown_usage_request_count"] == 1
    assert result["token_totals_are_lower_bounds"] is True
    assert result["resource_usage_complete"] is False and result["unknown_provider_tokens"] is None
    assert result["cost_usd"] is None
    obj = SimpleNamespace(resource_usage_complete=False, provider_requests=[SimpleNamespace(**trace["provider_requests"][0])])
    assert decision_usage_auditable(obj)
    obj.provider_requests=[]
    assert not decision_usage_auditable(obj)


@pytest.mark.parametrize("mutation", ["success", "unrecorded", "model_error", "missing_turn_usage"])
def test_unknown_successful_turn_or_unrecorded_failure_still_fails_closed(mutation):
    trace = error_trace()
    if mutation == "success": trace["provider_requests"][0]["outcome"] = "success"
    if mutation == "unrecorded": trace["provider_requests"] = []
    if mutation == "model_error": trace["provider_requests"][0]["error_code"] = "invalid_native_tool_call"
    if mutation == "missing_turn_usage": trace["agent_turns"] = [{"session_id": "model"}]
    with pytest.raises(ValueError): validate_formal_usage([trace])


def test_receipt_keeps_original_manifest_and_rejects_code_or_checkpoint_drift(tmp_path):
    repo, output = tmp_path, tmp_path / "runs/example"
    atomic_write_json(output / "run_manifest.json", {"run_id": "example", "code_commit": "old-code", "config_hash": "cfg"})
    checkpoint = output / ".task_checkpoint"
    checkpoint.mkdir()
    atomic_write_json(checkpoint / "checkpoint_manifest.json", {"run_id": "example", "code_commit": "old-code",
        "config_hash": "cfg", "task_id": "next", "files": []})
    atomic_write_json(repo / RELEASE_FILE, {"original_code_hash": "old-code", "patched_inventory_hash": release_inventory_hash(repo)})
    original = (output / "run_manifest.json").read_bytes()
    original_checkpoint = (checkpoint / "checkpoint_manifest.json").read_bytes()
    current = hash_code(repo)
    with pytest.raises(ProtocolError, match="configuration"):
        prepare_recovery(repo, output, current, "other", enabled=True)
    receipt = prepare_recovery(repo, output, current, "cfg", enabled=True)
    assert prepare_recovery(repo, output, current, "cfg", enabled=False) == receipt
    assert (output / "run_manifest.json").read_bytes() == original
    assert (checkpoint / "checkpoint_manifest.json").read_bytes() == original_checkpoint
    assert checkpoint_identity(output, current, receipt) == "old-code"
    atomic_write_json(repo / "unreviewed.json", {"changed": True})
    with pytest.raises(ProtocolError, match="exact reviewed"):
        read_recovery(repo, output, current)


def test_checkpoint_corruption_blocks_recovery(tmp_path):
    output = tmp_path / "runs/example"
    atomic_write_json(output / "run_manifest.json", {"run_id": "example", "code_commit": "old-code", "config_hash": "cfg"})
    atomic_write_json(output / ".task_checkpoint/checkpoint_manifest.json", {"run_id": "example", "code_commit": "old-code",
        "config_hash": "cfg", "task_id": "next", "files": [{"path": "state.sqlite3", "sha256": "wrong"}]})
    atomic_write_json(tmp_path / RELEASE_FILE, {"original_code_hash": "old-code", "patched_inventory_hash": release_inventory_hash(tmp_path)})
    with pytest.raises(ProtocolError, match="intact checkpoint"):
        prepare_recovery(tmp_path, output, hash_code(tmp_path), "cfg", enabled=True)
    assert not (output / RECEIPT_FILE).exists()
