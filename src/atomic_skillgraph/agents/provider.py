"""OpenAI-compatible HTTP adapter for the provider-independent Agent protocol."""

from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from ..core.errors import AgentProtocolError, AtomicSkillGraphError, FailureLayer
from .protocol import (
    AgentMessage,
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    parse_json_strict,
)


class AgentProviderError(AtomicSkillGraphError):
    """An HTTP, authentication, or invalid provider-envelope failure."""


class ProviderProtocolError(AgentProtocolError):
    """A provider response that cannot be admitted into the Agent protocol.

    ``usage_turn`` contains metering only, so the failed provider call can still
    be charged without treating malformed response content as an action.
    """

    def __init__(self, message: str, *, usage_turn: AgentTurn) -> None:
        super().__init__(
            "runtime_agent_schema_error",
            message,
            layer=FailureLayer.RUNTIME_AGENT,
        )
        self.usage_turn = usage_turn


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key_env: str
    max_completion_tokens: int
    reasoning_effort: str | None = None
    temperature: float | None = None
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
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env must be non-empty")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.temperature is not None and not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be numeric or None")
        if self.connect_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.max_retries < 0 or self.retry_backoff_seconds < 0:
            raise ValueError("retry settings must be non-negative")
        if self.max_retry_after_seconds < 0:
            raise ValueError("max_retry_after_seconds must be non-negative")
        forbidden = set()
        for name in self.extra_headers:
            lowered = name.lower()
            if (
                lowered in {"authorization", "proxy-authorization", "cookie", "set-cookie"}
                or ("api" in lowered and "key" in lowered)
                or lowered.endswith("subscription-key")
                or lowered.endswith("access-token")
            ):
                forbidden.add(lowered)
        if forbidden:
            raise ValueError(
                "authentication headers cannot be supplied directly; use api_key_env only"
            )

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def resolve_api_key(self) -> str:
        """Read the secret only from the configured environment variable."""
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise AgentProviderError(
                "infrastructure_failure",
                f"LLM API key environment variable {self.api_key_env!r} is not set",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        return value

    def snapshot(self) -> dict[str, Any]:
        """Return auditable, secret-free provider configuration."""
        return {
            "provider": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
        }


class OpenAICompatibleProvider:
    """Synchronous Chat Completions provider with bounded transient retries."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def snapshot(self) -> dict[str, Any]:
        return self.config.snapshot()

    def complete(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[NativeToolSpec] | None = None,
        structured_output_schema: dict[str, Any] | None = None,
    ) -> AgentTurn:
        normalized_messages = _validate_messages(messages)
        normalized_tools = list(tools or [])
        if len({tool.name for tool in normalized_tools}) != len(normalized_tools):
            raise ValueError("native tool names must be unique within a provider request")
        if structured_output_schema is not None and not isinstance(structured_output_schema, dict):
            raise TypeError("structured_output_schema must be a mapping or None")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": normalized_messages,
            "max_completion_tokens": self.config.max_completion_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if normalized_tools:
            payload["tools"] = [tool.to_openai() for tool in normalized_tools]
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        if structured_output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "schema": copy.deepcopy(structured_output_schema),
                },
            }

        api_key = self.config.resolve_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        started = time.perf_counter()
        retry_count = 0
        while True:
            try:
                response = requests.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(
                        self.config.connect_timeout_seconds,
                        self.config.request_timeout_seconds,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - transport boundary
                if _is_transient_transport_error(exc) and retry_count < self.config.max_retries:
                    retry_count += 1
                    self._backoff(retry_count)
                    continue
                message = _sanitize(f"{type(exc).__name__}: {exc}", secrets=(api_key,))
                raise AgentProviderError(
                    "llm_error",
                    message,
                    layer=FailureLayer.INFRASTRUCTURE,
                ) from exc

            if not response.ok:
                message = _sanitize(
                    f"HTTP {response.status_code} from LLM provider: {response.text[:600]}",
                    secrets=(api_key,),
                )
                if _is_transient_status(response.status_code) and retry_count < self.config.max_retries:
                    retry_count += 1
                    self._backoff(retry_count, response=response)
                    continue
                raise AgentProviderError(
                    "llm_error",
                    message,
                    layer=FailureLayer.INFRASTRUCTURE,
                )

            latency_ms = (time.perf_counter() - started) * 1000.0
            try:
                data = response.json()
            except Exception as exc:  # noqa: BLE001 - provider boundary
                message = _sanitize(
                    f"LLM provider returned invalid JSON: {response.text[:300]}",
                    secrets=(api_key,),
                )
                raise AgentProviderError(
                    "llm_error",
                    message,
                    layer=FailureLayer.INFRASTRUCTURE,
                ) from exc
            if not isinstance(data, dict):
                raise AgentProviderError(
                    "llm_error",
                    "LLM provider response must be a JSON object",
                    layer=FailureLayer.INFRASTRUCTURE,
                )
            return self._parse_response(
                data,
                latency_ms=latency_ms,
                retry_count=retry_count,
                response=response,
            )

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        latency_ms: float,
        retry_count: int,
        response: requests.Response,
    ) -> AgentTurn:
        try:
            usage, usage_metadata = _parse_usage(data.get("usage"))
        except (TypeError, ValueError) as exc:
            raise AgentProviderError(
                "llm_error",
                f"LLM provider returned invalid usage metadata: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        metadata: dict[str, Any] = {
            "provider": "openai_compatible",
            "model": self.config.model,
            "response_id": str(data.get("id", "")),
            "response_model": str(data.get("model", "")),
            "system_fingerprint": str(data.get("system_fingerprint", "") or ""),
            "created": data.get("created"),
            "retry_count": retry_count,
            **usage_metadata,
        }
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        if request_id:
            metadata["request_id"] = str(request_id)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            self._raise_protocol("provider response has no valid first choice", usage, latency_ms, metadata)
        choice = choices[0]
        finish_reason = str(choice.get("finish_reason", "") or "")
        message = choice.get("message")
        if not isinstance(message, dict):
            self._raise_protocol("provider choice has no assistant message", usage, latency_ms, metadata)

        raw_content = message.get("content")
        if raw_content is None:
            content = ""
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            self._raise_protocol(
                "assistant content must be a string or null",
                usage,
                latency_ms,
                metadata,
            )

        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            self._raise_protocol(
                "assistant tool_calls must be a list",
                usage,
                latency_ms,
                metadata,
                content=content,
                finish_reason=finish_reason,
            )
        calls: list[NativeToolCall] = []
        try:
            for raw_call in raw_calls:
                calls.append(_parse_native_tool_call(raw_call))
        except (TypeError, ValueError) as exc:
            self._raise_protocol(
                f"invalid native tool call: {exc}",
                usage,
                latency_ms,
                metadata,
                content=content,
                finish_reason=finish_reason,
            )

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
        )

    def _raise_protocol(
        self,
        message: str,
        usage: dict[str, Any],
        latency_ms: float,
        metadata: dict[str, Any],
        *,
        content: str = "",
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
        )
        raise ProviderProtocolError(message, usage_turn=usage_turn)

    def _backoff(self, retry_index: int, *, response: requests.Response | None = None) -> None:
        retry_after = _retry_after_seconds(response, self.config.max_retry_after_seconds)
        base = (
            retry_after
            if retry_after is not None
            else self.config.retry_backoff_seconds * (2 ** max(retry_index - 1, 0))
        )
        # Infrastructure retry jitter must not perturb the experiment's seeded
        # global RNG (candidate exploration and task ordering depend on it).
        jitter = random.SystemRandom().random() * min(0.2 * max(base, 1.0), 1.0)
        time.sleep(min(base + jitter, self.config.max_retry_after_seconds))


def _validate_messages(messages: list[AgentMessage]) -> list[AgentMessage]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("provider request requires a non-empty messages list")
    normalized = copy.deepcopy(messages)
    for index, message in enumerate(normalized):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be a mapping")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has an invalid role")
        if "content" not in message:
            raise ValueError(f"message {index} is missing content")
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
        arguments = raw_arguments
    else:
        raise TypeError("tool call arguments must be a JSON object string")
    if not isinstance(arguments, dict):
        raise TypeError("decoded tool call arguments must be a JSON object")
    return NativeToolCall(
        call_id=str(raw.get("id", "")),
        name=str(function.get("name", "")),
        arguments=arguments,
    )


def _parse_usage(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = raw if isinstance(raw, dict) else {}
    required_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    missing_fields = [name for name in required_fields if name not in usage]
    if missing_fields:
        raise ValueError(
            "missing required provider usage fields: " + ", ".join(missing_fields)
        )
    prompt = _nonnegative_int(usage.get("prompt_tokens", 0), "prompt_tokens")
    completion = _nonnegative_int(usage.get("completion_tokens", 0), "completion_tokens")
    total = _nonnegative_int(usage.get("total_tokens", 0), "total_tokens")

    details = usage.get("completion_tokens_details")
    reasoning: int | None = None
    reasoning_source = "unavailable"
    included: bool | None = None
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        reasoning = _nonnegative_int(details["reasoning_tokens"], "reasoning_tokens")
        reasoning_source = "completion_tokens_details.reasoning_tokens"
        included = True
    elif usage.get("reasoning_tokens") is not None:
        reasoning = _nonnegative_int(usage["reasoning_tokens"], "reasoning_tokens")
        reasoning_source = "usage.reasoning_tokens"
        reported_included = usage.get("reasoning_tokens_in_completion")
        included = reported_included if isinstance(reported_included, bool) else None

    metadata = {
        "usage_status": "reported",
        "missing_usage_fields": [],
        "reasoning_tokens_status": "reported" if reasoning is not None else "unavailable",
        "reasoning_tokens_source": reasoning_source,
        "reasoning_tokens_in_completion": included,
    }
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
    }, metadata


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"provider {field_name} must be a non-negative integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"provider {field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"provider {field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"provider {field_name} must be a non-negative integer")
    if isinstance(value, str) and value.strip() != str(parsed):
        raise ValueError(f"provider {field_name} must be a non-negative integer")
    return parsed


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
]
