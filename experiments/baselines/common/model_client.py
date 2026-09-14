"""Audited OpenAI-compatible transport; persists every physical attempt.

Unknown usage remains null. Empty target text is model behavior; empty
evolution text cannot be consumed as a successful structured update.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path


def append_event(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ProviderFailure(RuntimeError):
    failure_kind = "infrastructure_failure"


class AuditedChatClient:
    def __init__(self, *, output: Path, identity: dict, model: dict, gate=None, client=None):
        self.output, self.identity, self.model, self.gate = output, identity, model, gate
        if model["model"] != "deepseek-v4-flash" or model["reasoning_effort"] != "high":
            raise ValueError("Frozen model identity mismatch")
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ[model.get("api_key_env", "MODEL_API_KEY")],
                base_url=model["base_url"], max_retries=0, timeout=180)
        self.client = client

    def chat(self, *, messages, stage, role, max_tokens, temperature=0.1, stop=None):
        call_id = uuid.uuid4().hex
        payload = dict(model=self.model["model"], messages=messages,
                       reasoning_effort="high", max_tokens=int(max_tokens),
                       temperature=temperature, extra_body={"thinking": {"type": "enabled"}})
        if stop:
            payload["stop"] = stop
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        for attempt in range(1, 6):
            event = {**self.identity, "event": "provider_attempt", "logical_call_id": call_id,
                     "attempt": attempt, "role": role, "stage": stage,
                     "model": self.model["model"], "reasoning_effort": "high",
                     "payload_sha256": digest, "requested_max_tokens": int(max_tokens),
                     "prompt_tokens": None, "completion_tokens": None,
                     "reasoning_tokens": None, "usage_status": "unavailable"}
            response, content, retry, failure = None, None, False, None
            context = self.gate.acquire(run_id=self.identity["run_id"], seed=self.identity["run_seed"],
                role=role, stage=stage, logical_call_id=call_id) if self.gate else nullcontext(None)
            started = time.monotonic()
            try:
                with context as lease:
                    if lease:
                        event.update(provider_slot_id=lease.slot_id, provider_queue_wait_ms=lease.queue_wait_ms)
                    response = self.client.chat.completions.create(**payload)
                event["provider_request_id"] = getattr(response, "_request_id", None)
                usage = response.usage
                if usage is not None:
                    event.update(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                        reasoning_tokens=getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
                        usage_status="reported")
                if not response.choices or response.choices[0].message is None:
                    failure, retry = "invalid_response", True
                else:
                    event["finish_reason"] = response.choices[0].finish_reason
                    content = response.choices[0].message.content or ""
                    event["empty_content"] = not content.strip()
                    event["completion_budget_exhausted"] = event["finish_reason"] == "length"
                    if not content.strip() and role == "evolution":
                        failure, retry = "empty_evolution_message", True
                    elif usage is None:
                        failure, retry = "missing_usage", True
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                event["http_status"] = status
                failure = type(exc).__name__
                retry = status in (408, 429) or (status is not None and status >= 500) or isinstance(exc, (TimeoutError, ConnectionError)) or failure in {"APIConnectionError", "APITimeoutError"}
            event.update(latency_ms=int((time.monotonic() - started) * 1000),
                         status="failed" if failure else "succeeded", failure_code=failure)
            append_event(self.output, event)
            if content is not None:
                append_event(self.output.with_name("model_responses.jsonl"), {
                    **self.identity, "logical_call_id": call_id, "role": role,
                    "attempt": attempt, "stage": stage, "messages": messages, "content": content,
                    "finish_reason": event.get("finish_reason"), "failure_code": failure,
                })
            if not failure:
                return content
            if not retry or attempt == 5:
                raise ProviderFailure(f"Provider {stage} failed: {failure}; attempts={attempt}")
            time.sleep((2, 5, 10, 20)[attempt - 1])
        raise AssertionError("unreachable")
