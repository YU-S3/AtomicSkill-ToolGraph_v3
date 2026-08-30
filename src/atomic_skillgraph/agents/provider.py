"""DeepSeek V4 Chat adapter for the provider-independent Agent protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from ..core.errors import AgentProtocolError, AtomicSkillGraphError, FailureLayer
from .protocol import AgentMessage, AgentTurn, NativeToolCall, NativeToolSpec, parse_json_strict


class AgentProviderError(AtomicSkillGraphError):
    """A classified infrastructure failure at the LLM provider boundary."""

    def __init__(self, code: str, message: str, *, http_status: int | None = None) -> None:
        super().__init__(code, message, layer=FailureLayer.INFRASTRUCTURE)
        self.http_status = http_status


class ProviderProtocolError(AgentProviderError):
    """A metered response envelope that is invalid for formal Agent use."""

    def __init__(self, code: str, message: str, *, usage_turn: AgentTurn) -> None:
        super().__init__(code, message)
        self.usage_turn = usage_turn


class ProviderAgentProtocolError(AgentProtocolError):
    """A metered HTTP-200 turn whose model-authored tool call is malformed."""

    def __init__(self, code: str, message: str, *, usage_turn: AgentTurn) -> None:
        super().__init__(code, message, layer=FailureLayer.RUNTIME_AGENT)
        self.usage_turn = usage_turn


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for the single formal HTTP dialect used by this experiment."""

    base_url: str
    model: str
    api_key_env: str
    max_completion_tokens: int
    dialect: str = "deepseek_v4_chat"
    thinking_type: str = "enabled"
    reasoning_effort: str = "high"
    connect_timeout_seconds: float = 15.0
    request_timeout_seconds: float = 120.0
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    max_retry_after_seconds: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or fragments")
        if self.dialect != "deepseek_v4_chat":
            raise ValueError("formal provider dialect must be deepseek_v4_chat")
        if not self.model.strip() or not self.api_key_env.strip():
            raise ValueError("model and api_key_env must be non-empty")
        if isinstance(self.max_completion_tokens, bool) or self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.thinking_type not in {"enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled or disabled")
        if self.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
        if self.connect_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.max_retries < 0 or self.retry_backoff_seconds < 0:
            raise ValueError("retry settings must be non-negative")
        if self.max_retry_after_seconds < 0:
            raise ValueError("max_retry_after_seconds must be non-negative")
        for name in self.extra_headers:
            lowered = name.lower()
            if (
                lowered in {"authorization", "proxy-authorization", "cookie", "set-cookie"}
                or ("api" in lowered and "key" in lowered)
                or lowered.endswith("subscription-key")
                or lowered.endswith("access-token")
            ):
                raise ValueError("authentication headers cannot be supplied directly; use api_key_env")

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def resolve_api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise AgentProviderError(
                "provider_auth_error",
                f"LLM API key environment variable {self.api_key_env!r} is not set",
            )
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "dialect": self.dialect,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "max_completion_tokens": self.max_completion_tokens,
            "http_token_limit_field": "max_tokens",
            "thinking_type": self.thinking_type,
            "reasoning_effort": self.reasoning_effort,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
        }


class OpenAICompatibleProvider:
    """Synchronous DeepSeek Chat Completions provider with audited retries."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self._request_records: list[dict[str, Any]] = []
        self._records_lock = threading.RLock()
        self._request_context = threading.local()

    def snapshot(self) -> dict[str, Any]:
        value = self.config.snapshot()
        value["request_record_count"] = self.request_record_count
        return value

    @property
    def request_record_count(self) -> int:
        with self._records_lock:
            return len(self._request_records)

    @property
    def request_records(self) -> tuple[dict[str, Any], ...]:
        with self._records_lock:
            return tuple(copy.deepcopy(self._request_records))

    def request_records_since(self, start_index: int) -> tuple[dict[str, Any], ...]:
        if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
            raise ValueError("request record start index must be a non-negative integer")
        with self._records_lock:
            return tuple(copy.deepcopy(self._request_records[start_index:]))

    def set_request_context(self, *, session_id: str, stage: str) -> None:
        """Set thread-local audit attribution without changing the formal protocol."""
        self._request_context.value = {"session_id": str(session_id), "stage": str(stage)}

    def _build_payload(
        self, messages: list[AgentMessage], tools: list[NativeToolSpec] | None,
    ) -> dict[str, Any]:
        normalized_messages = _validate_deepseek_messages(messages)
        normalized_tools = list(tools or [])
        if not all(isinstance(tool, NativeToolSpec) for tool in normalized_tools):
            raise TypeError("tools must contain NativeToolSpec values")
        if len({tool.name for tool in normalized_tools}) != len(normalized_tools):
            raise ValueError("native tool names must be unique within a provider request")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": normalized_messages,
            "max_tokens": self.config.max_completion_tokens,
            "thinking": {"type": self.config.thinking_type},
            "reasoning_effort": self.config.reasoning_effort,
        }
        if normalized_tools:
            payload["tools"] = [tool.to_openai() for tool in normalized_tools]
        return payload

    def complete(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn:
        normalized_tools = list(tools or [])
        payload = self._build_payload(messages, normalized_tools)
        api_key = self.config.resolve_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        payload_fingerprint = _sha256_json(payload)
        payload_fields = sorted(payload)
        # Provider/transport exceptions and HTTP bodies are untrusted.  A
        # server or proxy can echo prior messages, including DeepSeek's
        # provider-private reasoning_content.  Treat every replay value as a
        # secret for diagnostics and never persist raw response bodies.
        private_replay_values = tuple(
            str(message.get("reasoning_content", ""))
            for message in payload["messages"]
            if message.get("role") == "assistant"
            and isinstance(message.get("reasoning_content"), str)
        )
        diagnostic_secrets = (api_key, *private_replay_values)
        started_all = time.perf_counter()
        retry_count = 0
        while True:
            audit_id = f"provider_request_{uuid.uuid4().hex}"
            started_at = time.time()
            try:
                response = requests.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(self.config.connect_timeout_seconds, self.config.request_timeout_seconds),
                )
            except Exception as exc:  # noqa: BLE001
                code = (
                    "provider_timeout"
                    if isinstance(exc, requests.exceptions.Timeout)
                    else "provider_transport_error"
                )
                message = _sanitize(
                    f"{type(exc).__name__}: {exc}", secrets=diagnostic_secrets,
                )
                self._append_request_record(
                    audit_id, started_at, "error", None, retry_count, None, code, message,
                    payload_fingerprint, payload_fields, "",
                )
                if _is_transient_transport_error(exc) and retry_count < self.config.max_retries:
                    retry_count += 1
                    self._backoff(retry_count)
                    continue
                raise AgentProviderError(code, message) from exc

            provider_request_id = _provider_request_id(response, {})
            if not response.ok:
                code = _http_error_code(response.status_code, response.text)
                message = f"HTTP {response.status_code} from LLM provider"
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count, None,
                    code, message, payload_fingerprint, payload_fields, provider_request_id,
                )
                if _is_transient_status(response.status_code) and retry_count < self.config.max_retries:
                    retry_count += 1
                    self._backoff(retry_count, response=response)
                    continue
                raise AgentProviderError(code, message, http_status=response.status_code)

            latency_ms = (time.perf_counter() - started_all) * 1000.0
            try:
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                message = "LLM provider returned invalid JSON"
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count, None,
                    "provider_invalid_response", message, payload_fingerprint, payload_fields,
                    provider_request_id,
                )
                raise AgentProviderError(
                    "provider_invalid_response", message, http_status=response.status_code,
                ) from exc
            if not isinstance(data, dict):
                message = "LLM provider response must be a JSON object"
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count, None,
                    "provider_invalid_response", message, payload_fingerprint, payload_fields,
                    provider_request_id,
                )
                raise AgentProviderError("provider_invalid_response", message)
            provider_request_id = _provider_request_id(response, data)
            try:
                turn = self._parse_response(
                    data,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    response=response,
                    tools_requested=bool(normalized_tools),
                )
            except (AgentProviderError, ProviderAgentProtocolError) as exc:
                usage_turn = getattr(exc, "usage_turn", None)
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count,
                    usage_turn if isinstance(usage_turn, AgentTurn) else None,
                    exc.code, _sanitize(str(exc), secrets=diagnostic_secrets), payload_fingerprint,
                    payload_fields, provider_request_id,
                )
                raise
            self._append_request_record(
                audit_id, started_at, "success", response.status_code, retry_count, turn, "", "",
                payload_fingerprint, payload_fields, provider_request_id,
            )
            return turn

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        latency_ms: float,
        retry_count: int,
        response: requests.Response,
        tools_requested: bool,
    ) -> AgentTurn:
        try:
            usage, usage_metadata = _parse_usage(data.get("usage"))
        except (TypeError, ValueError) as exc:
            raise AgentProviderError(
                "provider_usage_missing",
                f"LLM provider returned missing or invalid usage metadata: {exc}",
                http_status=response.status_code,
            ) from exc
        metadata: dict[str, Any] = {
            "provider": "openai_compatible",
            "dialect": self.config.dialect,
            "model": self.config.model,
            "response_id": str(data.get("id", "")),
            "response_model": str(data.get("model", "")),
            "system_fingerprint": str(data.get("system_fingerprint", "") or ""),
            "created": data.get("created"),
            "retry_count": retry_count,
            **usage_metadata,
        }
        request_id = _provider_request_id(response, data)
        if request_id:
            metadata["request_id"] = request_id
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            self._raise_protocol(
                "provider_invalid_response", "provider response has no valid first choice",
                usage, latency_ms, metadata,
            )
        choice = choices[0]
        finish_reason = str(choice.get("finish_reason", "") or "")
        message = choice.get("message")
        if not isinstance(message, dict):
            self._raise_protocol(
                "provider_invalid_response", "provider choice has no assistant message",
                usage, latency_ms, metadata,
            )
        raw_content = message.get("content")
        if raw_content is None:
            content = ""
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            self._raise_protocol(
                "provider_invalid_response", "assistant content must be a string or null",
                usage, latency_ms, metadata,
            )
        raw_reasoning = message.get("reasoning_content")
        if isinstance(raw_reasoning, str):
            reasoning_content = raw_reasoning
        elif tools_requested and self.config.thinking_type == "enabled":
            self._raise_protocol(
                "provider_reasoning_content_missing",
                "thinking+tools response is missing string reasoning_content",
                usage, latency_ms, metadata, content=content, finish_reason=finish_reason,
            )
        elif raw_reasoning is None:
            reasoning_content = ""
        else:
            self._raise_protocol(
                "provider_invalid_response", "assistant reasoning_content must be a string",
                usage, latency_ms, metadata, content=content, finish_reason=finish_reason,
            )
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            self._raise_protocol(
                "provider_invalid_response", "assistant tool_calls must be a list",
                usage, latency_ms, metadata, content=content,
                reasoning_content=reasoning_content, finish_reason=finish_reason,
            )
        calls: list[NativeToolCall] = []
        try:
            for raw_call in raw_calls:
                calls.append(_parse_native_tool_call(raw_call))
        except (TypeError, ValueError) as exc:
            self._raise_agent_protocol(
                "runtime_agent_schema_error", f"invalid native tool call: {exc}",
                usage, latency_ms, metadata, content=content,
                reasoning_content=reasoning_content, finish_reason=finish_reason,
            )
        replay_message: AgentMessage = {
            "role": "assistant", "content": content, "reasoning_content": reasoning_content,
        }
        if raw_calls:
            replay_message["tool_calls"] = copy.deepcopy(raw_calls)
        return AgentTurn(
            content=content,
            tool_calls=calls,
            finish_reason=finish_reason,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            provider_metadata=metadata,
            reasoning_content=reasoning_content,
            replay_assistant_message=replay_message,
        )

    def _raise_protocol(
        self,
        code: str,
        message: str,
        usage: dict[str, Any],
        latency_ms: float,
        metadata: dict[str, Any],
        *,
        content: str = "",
        reasoning_content: str = "",
        finish_reason: str = "",
    ) -> None:
        usage_turn = AgentTurn(
            content=content,
            tool_calls=[],
            finish_reason=finish_reason,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            provider_metadata=metadata,
            reasoning_content=reasoning_content,
        )
        raise ProviderProtocolError(code, message, usage_turn=usage_turn)

    def _raise_agent_protocol(
        self,
        code: str,
        message: str,
        usage: dict[str, Any],
        latency_ms: float,
        metadata: dict[str, Any],
        *,
        content: str = "",
        reasoning_content: str = "",
        finish_reason: str = "",
    ) -> None:
        usage_turn = AgentTurn(
            content=content,
            tool_calls=[],
            finish_reason=finish_reason,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            provider_metadata=metadata,
            reasoning_content=reasoning_content,
        )
        raise ProviderAgentProtocolError(code, message, usage_turn=usage_turn)

    def _append_request_record(
        self,
        audit_id: str,
        started_at: float,
        outcome: str,
        http_status: int | None,
        retry_count: int,
        usage: AgentTurn | None,
        error_code: str,
        sanitized_error: str,
        payload_fingerprint: str,
        payload_fields: list[str],
        provider_request_id: str,
    ) -> None:
        context = getattr(self._request_context, "value", {})
        reasoning = usage.reasoning_content if usage is not None else ""
        record = {
            "request_id": audit_id,
            "provider_request_id": provider_request_id,
            "session_id": str(context.get("session_id", "")),
            "stage": str(context.get("stage", "")),
            "endpoint": self.config.endpoint,
            "started_at": started_at,
            "ended_at": time.time(),
            "outcome": outcome,
            "http_status": http_status,
            "retry_count": retry_count,
            "usage_status": (
                str(usage.provider_metadata.get("usage_status", "reported"))
                if usage is not None else "unavailable"
            ),
            "usage": (
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                }
                if usage is not None else None
            ),
            "error_code": error_code,
            "sanitized_error": sanitized_error,
            "payload_fingerprint": payload_fingerprint,
            "payload_field_names": list(payload_fields),
            "reasoning_content_present": bool(reasoning),
            "reasoning_content_chars": len(reasoning),
            "reasoning_content_sha256": (
                hashlib.sha256(reasoning.encode("utf-8")).hexdigest() if reasoning else ""
            ),
        }
        with self._records_lock:
            self._request_records.append(record)

    def _backoff(self, retry_index: int, *, response: requests.Response | None = None) -> None:
        retry_after = _retry_after_seconds(response, self.config.max_retry_after_seconds)
        base = retry_after if retry_after is not None else (
            self.config.retry_backoff_seconds * (2 ** max(retry_index - 1, 0))
        )
        jitter = random.SystemRandom().random() * min(0.2 * max(base, 1.0), 1.0)
        time.sleep(min(base + jitter, self.config.max_retry_after_seconds))


def _validate_deepseek_messages(messages: list[AgentMessage]) -> list[AgentMessage]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("provider request requires a non-empty messages list")
    normalized = copy.deepcopy(messages)
    for index, message in enumerate(normalized):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has an invalid role")
        if "content" not in message or not isinstance(message["content"], str):
            raise TypeError(f"message {index} content must be a string")
        allowed = {
            "system": {"role", "content"},
            "user": {"role", "content"},
            "assistant": {"role", "content", "reasoning_content", "tool_calls"},
            "tool": {"role", "tool_call_id", "content"},
        }[str(role)]
        extras = set(message) - allowed
        if extras:
            raise ValueError(f"message {index} has unsupported DeepSeek fields: {sorted(extras)}")
        if role == "assistant" and "reasoning_content" in message and not isinstance(
            message["reasoning_content"], str
        ):
            raise TypeError(f"message {index} reasoning_content must be a string")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"message {index} tool_call_id must be non-empty")
    try:
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider messages must be JSON serializable") from exc
    return normalized


def _parse_native_tool_call(raw: Any) -> NativeToolCall:
    if not isinstance(raw, dict):
        raise TypeError("tool call must be an object")
    if raw.get("type", "function") != "function":
        raise ValueError("only native function tool calls are supported")
    function = raw.get("function")
    if not isinstance(function, dict):
        raise TypeError("tool call function must be an object")
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, str):
        arguments = parse_json_strict(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = copy.deepcopy(raw_arguments)
    else:
        raise TypeError("tool call arguments must be a JSON object string")
    if not isinstance(arguments, dict):
        raise TypeError("decoded tool call arguments must be a JSON object")
    return NativeToolCall(
        call_id=str(raw.get("id", "")), name=str(function.get("name", "")), arguments=arguments,
    )


def _parse_usage(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("usage object is missing")
    required_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    missing_fields = [name for name in required_fields if name not in raw]
    if missing_fields:
        raise ValueError("missing required provider usage fields: " + ", ".join(missing_fields))
    prompt = _nonnegative_int(raw["prompt_tokens"], "prompt_tokens")
    completion = _nonnegative_int(raw["completion_tokens"], "completion_tokens")
    total = _nonnegative_int(raw["total_tokens"], "total_tokens")
    details = raw.get("completion_tokens_details")
    reasoning: int | None = None
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        reasoning = _nonnegative_int(details["reasoning_tokens"], "reasoning_tokens")
    metadata: dict[str, Any] = {
        "usage_status": "reported",
        "missing_usage_fields": [],
        "reasoning_tokens_status": "reported" if reasoning is not None else "unavailable",
        "reasoning_tokens_source": (
            "completion_tokens_details.reasoning_tokens" if reasoning is not None else "unavailable"
        ),
        "reasoning_tokens_in_completion": True if reasoning is not None else None,
    }
    for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cache_hit_tokens", "cache_miss_tokens"):
        if raw.get(name) is not None:
            metadata[name] = _nonnegative_int(raw[name], name)
    prompt_details = raw.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens") is not None:
        metadata["prompt_cached_tokens"] = _nonnegative_int(
            prompt_details["cached_tokens"], "prompt_tokens_details.cached_tokens",
        )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
    }, metadata


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"provider {field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"provider {field_name} must be a non-negative integer") from exc
    if parsed < 0 or (isinstance(value, str) and value.strip() != str(parsed)):
        raise ValueError(f"provider {field_name} must be a non-negative integer")
    return parsed


def _provider_request_id(response: requests.Response, data: dict[str, Any]) -> str:
    return str(
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or data.get("id")
        or ""
    )


def _http_error_code(status_code: int, response_text: str) -> str:
    if status_code in {401, 403}:
        return "provider_auth_error"
    if status_code == 429:
        return "provider_rate_limited"
    if status_code in {408, 504}:
        return "provider_timeout"
    lowered = response_text.lower()
    if status_code == 400 and any(
        marker in lowered for marker in ("unavailable", "unsupported", "not support", "unknown parameter")
    ):
        return "provider_capability_mismatch"
    if 400 <= status_code < 500:
        return "provider_invalid_request"
    return "provider_transport_error"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_transient_transport_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
            requests.exceptions.ChunkedEncodingError,
        ),
    )


def _is_transient_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or 500 <= status_code < 600


def _retry_after_seconds(response: requests.Response | None, maximum: float) -> float | None:
    if response is None:
        return None
    value = str(response.headers.get("retry-after", "") or "").strip()
    if not value:
        return None
    try:
        return max(0.0, min(float(value), maximum))
    except ValueError:
        return None


def _sanitize(text: str, *, secrets: tuple[str, ...] = ()) -> str:
    sanitized = text
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED_API_KEY]")
    sanitized = re.sub(r"Bearer\s+[^\s,;]+", "Bearer [REDACTED_TOKEN]", sanitized, flags=re.I)
    sanitized = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", sanitized)
    return sanitized[:800]


__all__ = [
    "AgentProviderError",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "ProviderProtocolError",
    "ProviderAgentProtocolError",
]
