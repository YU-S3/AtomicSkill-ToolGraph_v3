"""Audited OpenAI-compatible transport; persists every physical attempt.

Unknown usage remains null. Empty target text is model behavior; empty
evolution text cannot be consumed as a successful structured update.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from experiments.baselines.common.reasoning_budget import policy_metadata, response_evidence
from experiments.baselines.common.source_identity import sanitize_error_text


def append_event(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ProviderFailure(RuntimeError):
    failure_kind = "infrastructure_failure"

    def __init__(self, message, *, retryable=False, failure_code=None):
        super().__init__(message)
        self.retryable = retryable
        if failure_code is not None:
            self.failure_code = failure_code


class InvalidProviderResponse(ValueError):
    """An unusable transport envelope, not an unsuccessful model action."""


def validate_response(response):
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise InvalidProviderResponse("Response has no nonempty choices array")
    message = getattr(choices[0], "message", None)
    if message is None or not hasattr(message, "content"):
        raise InvalidProviderResponse("First choice has no message/content field")
    if message.content is not None and not isinstance(message.content, str):
        raise InvalidProviderResponse("Message content is neither text nor null")


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
            raw = None
            error_phase = "request"
            context = self.gate.acquire(run_id=self.identity["run_id"], seed=self.identity["run_seed"],
                role=role, stage=stage, logical_call_id=call_id) if self.gate else nullcontext(None)
            started = time.monotonic()
            try:
                with context as lease:
                    if lease:
                        event.update(provider_slot_id=lease.slot_id, provider_queue_wait_ms=lease.queue_wait_ms)
                    endpoint = self.client.chat.completions
                    raw_endpoint = getattr(endpoint, "with_raw_response", None)
                    if raw_endpoint is not None:
                        raw = raw_endpoint.create(**payload)
                        event["http_status"] = raw.status_code
                        event["provider_request_id"] = raw.headers.get("x-request-id")
                        event["response_content_type"] = raw.headers.get("content-type")
                        error_phase = "response_decode"
                        response = raw.parse()
                    else:
                        response = endpoint.create(**payload)
                error_phase = "response_envelope"
                event["provider_request_id"] = getattr(response, "_request_id", None) or event.get("provider_request_id")
                # Even a malformed envelope may carry billed usage.
                event.update(response_evidence(SimpleNamespace(usage=getattr(response, "usage", None)),
                    cap=cap, hint=hint))
                validate_response(response)
                usage = getattr(response, "usage", None)
                error_phase = "response_evidence"
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
                if status is not None:
                    event["http_status"] = status
                failure = type(exc).__name__
                malformed = isinstance(exc, InvalidProviderResponse) or (
                    error_phase == "response_decode" and failure in {
                        "JSONDecodeError", "ValidationError", "APIResponseValidationError"})
                retry = malformed or status in (408, 429) or (status is not None and status >= 500) or isinstance(exc, (TimeoutError, ConnectionError)) or failure in {"APIConnectionError", "APITimeoutError"}
                event.update(error_type=failure, error_phase=error_phase,
                    error_message=sanitize_error_text(exc),
                    error_traceback=sanitize_error_text(traceback.format_exc()),
                    response_type=type(response).__name__)
                if malformed:
                    failure = "invalid_response"
                # Store only failed transport bodies; never expose credentials or
                # successful hidden reasoning to the method or its response log.
                if raw is not None and malformed:
                    event["response_body_excerpt"] = sanitize_error_text(raw.text)
                elif isinstance(response, (str, bytes)):
                    event["response_body_excerpt"] = sanitize_error_text(response)
                for field in ("error_message", "error_traceback", "response_body_excerpt"):
                    if field in event:
                        for name in (self.model.get("api_key_env", "MODEL_API_KEY"), "OPENAI_API_KEY"):
                            secret = os.environ.get(name)
                            if secret:
                                event[field] = event[field].replace(secret, "[REDACTED]")
            event.update(latency_ms=int((time.monotonic() - started) * 1000),
                         status="failed" if failure else "succeeded", failure_code=failure,
                         retryable=bool(failure and retry))
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
                raise ProviderFailure(f"Provider {stage} failed: {failure}; attempts={attempt}",
                    retryable=retry and failure != "empty_evolution_message", failure_code=failure)
            time.sleep((2, 5, 10, 20)[attempt - 1])
        raise AssertionError("unreachable")
