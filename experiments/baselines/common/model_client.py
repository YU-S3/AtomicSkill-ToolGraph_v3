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
from experiments.baselines.common.reasoning_budget import policy_metadata, response_evidence


def append_event(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ProviderFailure(RuntimeError):
    failure_kind = "infrastructure_failure"


class CompletionBudgetExhausted(ProviderFailure):
    failure_kind = "protocol_failure"
    failure_code = "COMPLETION_BUDGET_EXHAUSTED"


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

    def chat(self, *, messages, stage, role, max_tokens=None, method_output_token_hint=None,
             temperature=0.1, stop=None, content_parser=None):
        hint = method_output_token_hint if method_output_token_hint is not None else max_tokens
        cap = policy_metadata(self.model)["provider_completion_cap"]
        call_id = uuid.uuid4().hex
        payload = dict(model=self.model["model"], messages=messages,
                       reasoning_effort="high", max_tokens=cap,
                       temperature=temperature, extra_body={"thinking": {"type": "enabled"}})
        if stop:
            payload["stop"] = stop
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        for attempt in range(1, 6):
            event = {**self.identity, "event": "provider_attempt", "logical_call_id": call_id,
                     "attempt": attempt, "role": role, "stage": stage,
                     "model": self.model["model"], "reasoning_effort": "high",
                     "provider_attempt_id": uuid.uuid4().hex,
                     "http_token_limit_field": "max_tokens",
                     "payload_sha256": digest, "requested_max_tokens": hint,
                     **response_evidence(None, cap=cap, hint=hint),
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
                event.update(response_evidence(response, cap=cap, hint=hint, parser=content_parser))
                if not response.choices or response.choices[0].message is None:
                    failure, retry = "invalid_response", True
                else:
                    event["finish_reason"] = response.choices[0].finish_reason
                    content = response.choices[0].message.content or ""
                    event["empty_content"] = not content.strip()
                    event["completion_budget_exhausted"] = event["budget_exhausted"]
                    if event["budget_exhausted"]:
                        failure, retry = "COMPLETION_BUDGET_EXHAUSTED", False
                    elif not content.strip() and (role == "evolution" or stage == "trajectory_reranking"):
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
                    "provider_attempt_id": event["provider_attempt_id"],
                    "attempt": attempt, "stage": stage, "messages": messages, "content": content,
                    "finish_reason": event.get("finish_reason"), "failure_code": failure,
                })
            if not failure:
                return content
            if failure == "COMPLETION_BUDGET_EXHAUSTED":
                raise CompletionBudgetExhausted(f"{failure}: {stage}; fixed cap={cap}; attempts={attempt}")
            if not retry or attempt == 5:
                raise ProviderFailure(f"Provider {stage} failed: {failure}; attempts={attempt}")
            time.sleep((2, 5, 10, 20)[attempt - 1])
        raise AssertionError("unreachable")
