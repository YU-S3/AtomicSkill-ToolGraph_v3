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
    input_modalities: tuple[str, ...] = ('text',)
    token_limit_field: str = 'max_tokens'
    generation_seed: int | None = None
    capability_profile: dict | None = None
    image_input: dict | None = None
    stream: bool = False
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        from .image_input import validate_policy
        validate_policy(self.image_input)
        if type(self.stream) is not bool:
            raise ValueError('stream must be boolean')
        if self.thinking_budget is not None and (self.dialect!='qwen_38_chat' or
                self.thinking_type!='enabled' or type(self.thinking_budget) is not int or
                not 1<=self.thinking_budget<=262144):
            raise ValueError('Explicit thinking budget requires enabled Qwen3.8 thinking in 1..262144')
        if self.image_input is not None and 'image' not in self.input_modalities:
            raise ValueError('Image input policy requires a visual capability lock')
        defaults = {'tools':True, 'thinking':self.dialect in {'deepseek_v4_chat', 'moonshot_k2_chat', 'glm_53_chat', 'minimax_m3_chat', 'qwen_38_chat'},
                    'reasoning_effort':self.dialect in {'deepseek_v4_chat','glm_53_chat','qwen_38_chat'},
                    'generation_seed':self.dialect=='qwen_38_chat'}
        if self.capability_profile is not None and (not isinstance(self.capability_profile,dict) or
                set(self.capability_profile)-set(defaults) or any(type(v) is not bool for v in self.capability_profile.values())):
            raise ValueError('Invalid provider capability profile')
        object.__setattr__(self, 'capability_profile', {**defaults, **(self.capability_profile or {})})
        if self.generation_seed is not None and not self.capability_profile['generation_seed']:
            raise ValueError('Provider generation seed unsupported')
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or fragments")
        if self.dialect not in {'deepseek_v4_chat', 'openai_chat', 'moonshot_k2_chat', 'glm_53_chat', 'minimax_m3_chat', 'qwen_38_chat'}:
            raise ValueError('Unsupported provider dialect')
        if self.dialect == 'moonshot_k2_chat' and (self.capability_profile['reasoning_effort'] or self.capability_profile['generation_seed']):
            raise ValueError('Moonshot K2 has no configured reasoning_effort or generation seed support')
        if self.dialect == 'glm_53_chat':
            if self.model not in {'glm-5.3','glm-5.3-flash','glm-5.3-flashx'}:
                raise ValueError('GLM 5.3 dialect requires a documented GLM 5.3 model ID')
            if self.thinking_type!='enabled' or not self.capability_profile['thinking']:
                raise ValueError('GLM 5.3 requires enabled thinking, including learning requests')
            if self.reasoning_effort not in {'low','high','max'} or not self.capability_profile['reasoning_effort']:
                raise ValueError('GLM 5.3 reasoning_effort must be low, high or max')
            if self.capability_profile['generation_seed'] or self.generation_seed is not None:
                raise ValueError('GLM 5.3 generation seed is not established by the official API contract')
            if self.token_limit_field!='max_tokens' or type(self.max_completion_tokens) is not int or not 1<=self.max_completion_tokens<=131072:
                raise ValueError('GLM 5.3 requires max_tokens in 1..131072')
        if self.dialect in {'minimax_m3_chat','qwen_38_chat'}:
            expected = 'MiniMax-M3' if self.dialect=='minimax_m3_chat' else 'qwen3.8-flash'
            maximum = 524288 if self.dialect=='minimax_m3_chat' else 131072
            if self.model!=expected or not self.capability_profile['thinking']:
                raise ValueError('Model-specific dialect requires '+expected+' with explicit thinking control')
            fields={'max_completion_tokens','max_tokens'} if self.dialect=='qwen_38_chat' else {'max_completion_tokens'}
            if self.token_limit_field not in fields or type(self.max_completion_tokens) is not int or not 1<=self.max_completion_tokens<=maximum:
                raise ValueError(expected+' requires a total max_completion_tokens cap in 1..'+str(maximum))
            if self.dialect=='minimax_m3_chat' and (self.capability_profile['reasoning_effort'] or self.capability_profile['generation_seed']):
                raise ValueError('MiniMax-M3 has no effective reasoning_effort or documented generation seed')
            if self.dialect=='qwen_38_chat':
                if self.token_limit_field=='max_tokens' and self.thinking_type=='enabled' and self.thinking_budget is None:
                    raise ValueError('Qwen body-only cap requires an explicit separately reserved thinking budget')
                if not self.capability_profile['reasoning_effort']:
                    raise ValueError('Qwen3.8 reasoning effort must be explicitly supported')
                if self.generation_seed is not None and (type(self.generation_seed) is not int or not 0<=self.generation_seed<2**31):
                    raise ValueError('Qwen generation seed must be an integer in 0..2^31-1')
        if self.token_limit_field not in {'max_tokens', 'max_completion_tokens'}:
            raise ValueError('Unsupported token limit field')
        if not self.model.strip() or not self.api_key_env.strip():
            raise ValueError("model and api_key_env must be non-empty")
        if isinstance(self.max_completion_tokens, bool) or self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.thinking_type not in {"enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled or disabled")
        if self.reasoning_effort not in ({'low','medium','high','max','xhigh'} if self.dialect=='qwen_38_chat' else {"low", 'medium', "high", "max"}):
            raise ValueError('Unsupported reasoning effort')
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
            "http_token_limit_field": self.token_limit_field,
            "thinking_type": self.thinking_type if self.capability_profile['thinking'] else None,
            "reasoning_effort": self.wire_reasoning_effort if self.capability_profile['reasoning_effort'] and self.thinking_budget is None else None,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "capability_profile": dict(self.capability_profile),
            **({'stream':True} if self.stream else {}),
            **({'glm_protocol':{'version':'bigmodel.glm53.chat.v1','clear_thinking':False,
                'temperature':1.0,'top_p':0.95,'tool_choice':'auto only; single named submission mapped and host validated',
                'tool_stream':self.stream,'stream_usage':'native final chunk; no stream_options'}} if self.dialect=='glm_53_chat' else {}),
            **({'minimax_protocol':{'version':'minimax.m3.chat.v1','thinking_type':'adaptive' if self.thinking_type=='enabled' else 'disabled',
                'reasoning_split':True,'service_tier':'standard','fixed_temperature':1,'fixed_top_p':0.95,
                'tool_choice':'omitted: native tools with host-validated single submission',
                'output_cap_includes_reasoning':True}} if self.dialect=='minimax_m3_chat' else {}),
            **({'qwen_protocol':{'version':'qwen.3.8.flash.body-budget.v2' if self.thinking_budget is not None else 'qwen.3.8.flash.chat.v1','enable_thinking':self.thinking_type=='enabled',
                'preserve_thinking':True,'requested_reasoning_effort':self.reasoning_effort,
                'sent_reasoning_effort':self.wire_reasoning_effort if self.thinking_type=='enabled' and self.thinking_budget is None else None,
                'tool_choice':'auto for a single named submission while thinking; host validation unchanged',
                'output_cap_includes_reasoning':self.token_limit_field=='max_completion_tokens','temperature':0.6 if self.thinking_type=='enabled' else None,
                **({'thinking_budget':self.thinking_budget,'body_token_cap':self.max_completion_tokens if self.token_limit_field=='max_tokens' else None,
                    'effort_control':'thinking_budget; reasoning_effort omitted to avoid conflicting parameters'} if self.thinking_budget is not None else {})}}
                if self.dialect=='qwen_38_chat' else {}),
            **({'image_input':dict(self.image_input)} if self.image_input else {}),
        }

    @property
    def wire_reasoning_effort(self):
        if self.dialect=='qwen_38_chat' and self.reasoning_effort in {'high','max'}:
            return 'xhigh'
        return self.reasoning_effort


class OpenAICompatibleProvider:
    """Synchronous DeepSeek Chat Completions provider with audited retries."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self._request_records: list[dict[str, Any]] = []
        self._records_lock = threading.RLock()
        self._request_context = threading.local()
        self.budget_governor = None

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

    def set_request_context(self, *, session_id: str, stage: str,
                            request_sequence: int | None = None, repair: bool | None = None, **attribution) -> None:
        """Set thread-local audit attribution without changing the formal protocol."""
        self._request_context.value = {"session_id": str(session_id), "stage": str(stage),
            "request_sequence": request_sequence, "repair": repair, **attribution}

    def _build_payload(
        self, messages: list[AgentMessage], tools: list[NativeToolSpec] | None,
        tool_choice: dict | str | None = None,
    ) -> dict[str, Any]:
        normalized_messages = _validate_deepseek_messages(messages, allow_images='image' in self.config.input_modalities)
        if self.config.dialect not in {'deepseek_v4_chat', 'moonshot_k2_chat', 'glm_53_chat', 'minimax_m3_chat', 'qwen_38_chat'}:
            normalized_messages = [{k:v for k,v in m.items() if k != 'reasoning_content'} for m in normalized_messages]
        normalized_tools = list(tools or [])
        if not all(isinstance(tool, NativeToolSpec) for tool in normalized_tools):
            raise TypeError("tools must contain NativeToolSpec values")
        if len({tool.name for tool in normalized_tools}) != len(normalized_tools):
            raise ValueError("native tool names must be unique within a provider request")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": normalized_messages,
            self.config.token_limit_field: self.config.max_completion_tokens,
        }
        if self.config.stream:
            payload['stream']=True
            if self.config.dialect!='glm_53_chat':payload['stream_options']={'include_usage':True}
        if self.config.capability_profile['reasoning_effort'] and self.config.thinking_budget is None and not (self.config.dialect=='qwen_38_chat' and self.config.thinking_type=='disabled'):
            payload['reasoning_effort'] = self.config.wire_reasoning_effort
        if self.config.capability_profile['thinking']:
            payload['thinking'] = {'type': self.config.thinking_type}
        if self.config.dialect=='glm_53_chat':
            payload['thinking']['clear_thinking']=False
            payload.update(temperature=1.0,top_p=0.95)
        if self.config.dialect=='minimax_m3_chat':
            payload['thinking']={'type':'adaptive' if self.config.thinking_type=='enabled' else 'disabled'}
            payload.update(reasoning_split=True,service_tier='standard')
        if self.config.dialect=='qwen_38_chat':
            payload.pop('thinking',None)
            payload.update(enable_thinking=self.config.thinking_type=='enabled',preserve_thinking=True)
            if self.config.thinking_type=='enabled':payload['temperature']=0.6
            if self.config.thinking_budget is not None:payload['thinking_budget']=self.config.thinking_budget
        if normalized_tools:
            if not self.config.capability_profile['tools']: raise ValueError('Provider native tools unsupported')
            payload["tools"] = self._wire_tools(normalized_tools)
            if self.config.dialect=='glm_53_chat' and self.config.stream:payload['tool_stream']=True
        if tool_choice is not None:
            if self.config.dialect=='minimax_m3_chat':
                if tool_choice!='auto' and not (len(normalized_tools)==1 and tool_choice=={'type':'function','function':{'name':normalized_tools[0].name}}):
                    raise ValueError('MiniMax-M3 supports native automatic selection or one matching submission Schema')
                # The M3 OpenAI contract documents tools, but not forced tool_choice.
                # Omit that unsupported parameter; retain the exact Schema and host checks.
            elif self.config.dialect=='qwen_38_chat':
                if isinstance(tool_choice,str) and tool_choice in {'auto','none'}:
                    payload['tool_choice']=tool_choice
                elif (isinstance(tool_choice,dict) and tool_choice in [
                        {'type':'function','function':{'name':t.name}} for t in normalized_tools]):
                    if self.config.thinking_type=='enabled':
                        if len(normalized_tools)!=1:raise ValueError('Qwen thinking cannot force selection among multiple tools')
                        payload['tool_choice']='auto'
                    else:payload['tool_choice']=copy.deepcopy(tool_choice)
                else:raise ValueError('Qwen3.8 does not support required or an unknown named tool')
            elif self.config.dialect=='glm_53_chat':
                if tool_choice=='auto':
                    payload['tool_choice']='auto'
                elif (len(normalized_tools)==1 and tool_choice=={'type':'function','function':{'name':normalized_tools[0].name}}):
                    # Auto-only API: keep one submission Schema and host validation.
                    payload['tool_choice']='auto'
                else:raise ValueError('GLM 5.3 supports auto or a single matching submission tool only')
            elif self.config.dialect == 'moonshot_k2_chat' and isinstance(tool_choice, str):
                if tool_choice not in {'auto', 'none'}:
                    raise ValueError('Moonshot K2 string tool_choice must be auto or none')
                payload['tool_choice'] = tool_choice
            else:
                if (not isinstance(tool_choice, dict) or set(tool_choice) != {'type', 'function'} or
                    tool_choice.get('type') != 'function' or not isinstance(tool_choice.get('function'), dict) or
                    set(tool_choice['function']) != {'name'} or
                    tool_choice['function']['name'] not in {tool.name for tool in normalized_tools}):
                    raise ValueError('Named tool_choice must identify an actual request tool')
                if self.config.dialect in {'deepseek_v4_chat', 'moonshot_k2_chat'} and self.config.thinking_type != 'disabled':
                    raise ValueError('Named tool_choice requires disabled thinking')
                payload['tool_choice'] = copy.deepcopy(tool_choice)
        if self.config.generation_seed is not None:
            payload['seed'] = self.config.generation_seed
        return payload

    def _wire_tools(self, tools):
        values = [tool.to_openai() for tool in tools]
        if self.config.dialect == 'moonshot_k2_chat':
            from .moonshot_schema import wire_schema
            for value in values:
                value['function']['parameters'] = wire_schema(value['function']['parameters'])
        return values

    def input_measure(self, payload):
        from .image_input import measure
        return measure(payload, self.config.image_input)

    def input_token_bound(self, payload):
        return self.input_measure(payload)['input_token_bound']

    def complete(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[NativeToolSpec] | None = None,
        max_completion_tokens: int | None = None,
        tool_choice: dict | str | None = None,
    ) -> AgentTurn:
        normalized_tools = list(tools or [])
        payload = self._build_payload(messages, normalized_tools, tool_choice) if tool_choice is not None else self._build_payload(messages, normalized_tools)
        if max_completion_tokens is not None:
            if type(max_completion_tokens) is not int or max_completion_tokens <= 0:
                raise ValueError('Request completion cap must be positive')
            if self.config.dialect=='glm_53_chat' and max_completion_tokens>131072:
                raise ValueError('GLM 5.3 max_tokens exceeds 131072')
            if self.config.dialect in {'minimax_m3_chat','qwen_38_chat'} and max_completion_tokens>(524288 if self.config.dialect=='minimax_m3_chat' else 131072):
                raise ValueError('Model output cap exceeds documented maximum')
            payload[self.config.token_limit_field] = max_completion_tokens
        image_audit = self.input_measure(payload)
        payload_bytes = image_audit['text_wire_bytes']
        if self.config.image_input is not None:
            self._request_context.value = {**getattr(self._request_context, 'value', {}),
                'image_input': self.config.image_input, 'image_audit': image_audit}
        maximum = getattr(self._request_context, 'value', {}).get('payload_max_bytes')
        if maximum is not None and payload_bytes > maximum:
            scope = 'provider text payload' if self.config.image_input else 'complete provider payload'
            error = ValueError('material_too_large: ' + scope + ' ' + str(payload_bytes) + ' > ' + str(maximum))
            error.code = 'material_too_large'
            raise error
        # This observes the final serialized HTTP surface, after all Session
        # edits. Store only public policy segments, schema and hashes; never
        # headers or provider-private assistant reasoning/envelopes.
        policy_segments = []
        for message in payload['messages']:
            content = message.get('content', '')
            if message.get('role') == 'user' and isinstance(content, str) and '\nPOLICY_CONTEXT_JSON\n' in content:
                public = content.split('\nPOLICY_CONTEXT_JSON\n', 1)[1]
                try:
                    policy_segments.append(json.loads(public))
                except (TypeError, ValueError):
                    policy_segments.append({'parse_failed': True, 'sha256': hashlib.sha256(public.encode()).hexdigest()})
        self._request_context.final_payload_audit = {
            'serialized_parameters': {k:copy.deepcopy(v) for k,v in payload.items()
                                      if k not in {'messages','tools'}},
            'provider_snapshot': self.config.snapshot(),
            'messages': [{k: copy.deepcopy(v) for k, v in m.items() if k != 'reasoning_content'}
                         for m in payload['messages']],
            'private_reasoning_redacted': any('reasoning_content' in m for m in payload['messages']),
            'messages_sha256': _sha256_json(payload['messages']),
            'tools_sha256': _sha256_json(payload.get('tools', [])),
            'messages_utf8_bytes': len(json.dumps(payload['messages'], ensure_ascii=False).encode()),
            'tools_utf8_bytes': len(json.dumps(payload.get('tools', []), ensure_ascii=False).encode()),
            'policy_contexts': policy_segments,
            'policy_blocks': [{key: {'sha256': hashlib.sha256(json.dumps(value,
                ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                'utf8_bytes': len(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())}
                for key, value in segment.items()} for segment in policy_segments],
            # Persist the hash BEFORE Trace's canonical JSON writer sorts
            # object keys. Re-hashing the loaded dict loses the sent order.
            'policy_context_sha256': [hashlib.sha256(json.dumps(segment,
                ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                for segment in policy_segments],
            'tools': copy.deepcopy(payload.get('tools', [])),
            'captured_after_build_payload': True,
            'serialized_payload_utf8_bytes': image_audit['wire_bytes'],
            **({'text_material_utf8_bytes':payload_bytes,'image_audit':image_audit} if self.config.image_input else {}),
        }
        if tool_choice is not None:
            self._request_context.final_payload_audit['tool_choice'] = copy.deepcopy(payload.get('tool_choice'))
            if payload.get('tool_choice')!=tool_choice:
                self._request_context.final_payload_audit['tool_choice_adaptation']={
                    'requested':copy.deepcopy(tool_choice),'sent':payload.get('tool_choice'),
                    'reason':'Provider automatic tool selection; single submission Schema with unchanged host validation'}
        requested_tools = [tool.to_openai() for tool in normalized_tools]
        if payload.get('tools', []) != self._wire_tools(normalized_tools):
            raise ValueError('Final provider tools differ from requested tools')
        if payload.get('tools', []) != requested_tools:
            self._request_context.final_payload_audit['schema_adaptation'] = {
                'version':'moonshot.schema-syntax.v1','requested_tools':requested_tools,
                'host_validation':'Original schema retained; only wire syntax uses typed enums and anyOf.'}
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
            self._request_context.response_diagnostic = {}
            self._request_context.raw_usage = None
            self._request_context.public_response = None
            if self.budget_governor:
                self.budget_governor.admit(audit_id, payload, getattr(self._request_context, 'value', {}))
            try:
                stream_data = None
                response = requests.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(self.config.connect_timeout_seconds, self.config.request_timeout_seconds),
                    **({'stream':True} if self.config.stream else {}),
                )
                if self.config.stream and response.ok:
                    try:
                        stream_data = self._read_stream(response)
                    finally:
                        response.close()
            except Exception as exc:  # noqa: BLE001
                code = (
                    "provider_timeout"
                    if isinstance(exc, requests.exceptions.Timeout)
                    else "provider_invalid_response" if isinstance(exc,ValueError) else "provider_transport_error"
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

            body = b'' if stream_data is not None else getattr(response, "content", b"")
            self._request_context.response_diagnostic.update({
                "body_bytes": len(body) if isinstance(body, bytes) else None,
                "body_sha256": hashlib.sha256(body).hexdigest() if isinstance(body, bytes) and body else "",
            })
            provider_request_id = _provider_request_id(response, {})
            if not response.ok:
                try:
                    error = response.json().get('error', {})
                    public_error = {}
                    for field in ('code', 'type', 'message'):
                        if field in error:
                            value = str(error[field]).replace(api_key, '[redacted]')
                            value = re.sub(r'(?i)(?:bearer\s+\S+|sk-[A-Za-z0-9_-]+)', '[redacted]', value)
                            public_error[field] = value[:1024]
                            if len(value) > 1024: public_error['truncated'] = True
                    self._request_context.response_diagnostic['error'] = public_error
                except (ValueError, TypeError, AttributeError):
                    self._request_context.response_diagnostic['error_unparseable'] = True
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
                data = stream_data if stream_data is not None else response.json()
            except Exception as exc:  # noqa: BLE001
                message = "LLM provider returned invalid JSON"
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count, None,
                    "provider_invalid_response", message, payload_fingerprint, payload_fields,
                    provider_request_id,
                )
                if retry_count < self.config.max_retries:
                    retry_count += 1
                    self._backoff(retry_count, response=response)
                    continue
                raise AgentProviderError(
                    "provider_invalid_response", message, http_status=response.status_code,
                ) from exc
            if not isinstance(data, dict):
                self._request_context.response_diagnostic["json_type"] = type(data).__name__
                message = "LLM provider response must be a JSON object"
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count, None,
                    "provider_invalid_response", message, payload_fingerprint, payload_fields,
                    provider_request_id,
                )
                if retry_count < self.config.max_retries:
                    retry_count += 1
                    self._backoff(retry_count, response=response)
                    continue
                raise AgentProviderError("provider_invalid_response", message, http_status=response.status_code)
            provider_request_id = _provider_request_id(response, data)
            self._request_context.raw_usage = copy.deepcopy(data.get('usage'))
            self._request_context.public_response = {
                **{k: copy.deepcopy(data[k]) for k in ('id', 'model', 'created', 'system_fingerprint') if k in data},
                'choices': [{**{k: copy.deepcopy(v) for k, v in c.items() if k != 'message'},
                             'message': {k: copy.deepcopy(v) for k, v in c.get('message', {}).items()
                                         if k != 'reasoning_content'}}
                            for c in (data.get('choices') if isinstance(data.get('choices'), list) else []) if isinstance(c, dict)
                            and isinstance(c.get('message', {}), dict)]}
            try:
                turn = self._parse_response(
                    data,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    response=response,
                    tools_requested=bool(normalized_tools),
                )
            except (AgentProviderError, ProviderAgentProtocolError) as exc:
                usage_turn = getattr(exc, "usage_turn", None) or getattr(exc, 'billing_turn', None)
                self._append_request_record(
                    audit_id, started_at, "error", response.status_code, retry_count,
                    usage_turn if isinstance(usage_turn, AgentTurn) else None,
                    exc.code, _sanitize(str(exc), secrets=diagnostic_secrets), payload_fingerprint,
                    payload_fields, provider_request_id,
                )
                # Only retry an unusable, unmetered provider envelope here.
                # Metered/model-authored invalid turns retain the existing
                # session accounting and protocol-repair path.
                if (not isinstance(usage_turn, AgentTurn)
                        and exc.code == "provider_usage_missing"
                        and retry_count < self.config.max_retries):
                    retry_count += 1
                    self._backoff(retry_count, response=response)
                    continue
                raise
            self._append_request_record(
                audit_id, started_at, "success", response.status_code, retry_count, turn, "", "",
                payload_fingerprint, payload_fields, provider_request_id,
            )
            return turn

    def _read_stream(self, response):
        """Reassemble Chat SSE without losing reasoning, tool arguments or final usage."""
        from pathlib import Path
        from ..empirical.checkpoint import save_json
        output=getattr(self._request_context,'value',{}).get('stream_output_dir')
        folder=Path(output) if output else None
        if folder:folder.mkdir(parents=True,exist_ok=True)
        journal=(folder/'stream_events.jsonl').open('w',encoding='utf-8') if folder else None
        data={};message={'role':'assistant','content':'','reasoning_content':''};calls={}
        finish=None;done=False;count=0;size=0;checksum=hashlib.sha256();last=0
        def report(complete=False):
            value={'updated_at':time.time(),'events':count,'bytes':size,'response_id':data.get('id'),
                'response_model':data.get('model'),'content_chars':len(message['content']),
                'reasoning_chars':len(message['reasoning_content']),'finish_reason':finish,
                'usage_received':data.get('usage') is not None,'completed':complete,
                'events_sha256':checksum.hexdigest()}
            self._request_context.response_diagnostic.update(stream=value)
            if folder:save_json(folder/'stream_progress.json',value)
        try:
            for line in response.iter_lines(chunk_size=1024):
                if not line or line.startswith(b':'):continue
                if not line.startswith(b'data:'):raise ValueError('Invalid SSE data line')
                raw=line[5:].strip()
                if raw==b'[DONE]':done=True;break
                chunk=json.loads(raw)
                if not isinstance(chunk,dict) or 'error' in chunk:raise ValueError('Invalid provider SSE event')
                count+=1;size+=len(raw);checksum.update(raw+b'\n')
                if size>64*1024*1024:raise ValueError('Provider SSE envelope exceeds 64 MiB')
                if journal:journal.write(raw.decode('utf-8')+'\n')
                for key in ('id','model','created','system_fingerprint'):
                    if key not in chunk:continue
                    if key=='created' and self.config.dialect in {'minimax_m3_chat','qwen_38_chat'}:
                        # These APIs emit event timestamps, including a possibly earlier
                        # final usage timestamp. Response id/model remain strict identities.
                        stamp=chunk[key]
                        if type(stamp) is not int or stamp<0:raise ValueError('Invalid SSE created timestamp')
                        audit=data.setdefault('_stream_created',{'first':stamp,'last':stamp,
                            'min':stamp,'max':stamp,'changes':0})
                        audit['changes']+=int(audit['last']!=stamp)
                        audit.update(last=stamp,min=min(audit['min'],stamp),max=max(audit['max'],stamp))
                        data.setdefault('created',stamp)
                        continue
                    if key in data and data[key]!=chunk[key]:raise ValueError('Provider SSE identity changed: '+key)
                    data[key]=chunk[key]
                if chunk.get('usage') is not None:
                    data['usage']=chunk['usage'];self._request_context.raw_usage=copy.deepcopy(chunk['usage'])
                if self.config.dialect=='minimax_m3_chat' and 'base_resp' in chunk:
                    data['base_resp']=copy.deepcopy(chunk['base_resp'])
                    if not isinstance(data['base_resp'],dict):raise ValueError('Invalid MiniMax base_resp')
                    if data['base_resp'].get('status_code',0)!=0:
                        # Pass the native error to the same classifier as non-stream responses.
                        # Retain any reported usage; never turn an error event into a scored answer.
                        report()
                        return data
                for choice in chunk.get('choices',[]):
                    if choice.get('index',0)!=0:raise ValueError('Unexpected additional streamed choice')
                    delta=choice.get('delta',{})
                    for key in ('content','reasoning_content'):
                        fragment=delta.get(key)
                        if fragment is not None:
                            if not isinstance(fragment,str):raise ValueError('Invalid SSE text fragment')
                            message[key]+=fragment
                    for fragment in delta.get('tool_calls',[]):
                        index=fragment.get('index')
                        if type(index) is not int or index<0:raise ValueError('Invalid streamed tool index')
                        call=calls.setdefault(index,{'id':'','type':'function','function':{'name':'','arguments':''}})
                        if fragment.get('id'):call['id']+=fragment['id']
                        if fragment.get('type') not in (None,'function'):raise ValueError('Invalid streamed tool type')
                        for key in ('name','arguments'):
                            piece=fragment.get('function',{}).get(key)
                            if piece is not None:
                                if not isinstance(piece,str):raise ValueError('Invalid streamed tool fragment')
                                call['function'][key]+=piece
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
                if time.monotonic()-last>=1:
                    if journal:journal.flush()
                    report();last=time.monotonic()
            if not done and self.config.dialect=='minimax_m3_chat' and finish is not None:
                # MiniMax documents native final-chunk termination. Graceful EOF is
                # sufficient only with an explicit finish reason AND complete usage.
                _parse_usage(data.get('usage'))
                done=True
            if not done or finish is None:raise ValueError('Incomplete provider SSE termination')
            if calls:message['tool_calls']=[calls[i] for i in sorted(calls)]
            data['choices']=[{'index':0,'message':message,'finish_reason':finish}]
            report(True)
            return data
        finally:
            if journal:journal.close()
            if not done:report()

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        latency_ms: float,
        retry_count: int,
        response: requests.Response,
        tools_requested: bool,
    ) -> AgentTurn:
        if self.config.dialect=='minimax_m3_chat':
            self._check_minimax_status(data,latency_ms,response)
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        finish = choice.get("finish_reason")
        outcomes = {"content_filter": "provider_content_filter"}
        if self.config.dialect == "glm_53_chat":
            outcomes.update(sensitive="provider_content_filter", network_error="provider_inference_error",
                            model_context_window_exceeded="provider_context_window_exceeded")
        if finish in outcomes:
            # An explicit provider outcome remains identifiable even without billing.
            error = AgentProviderError(outcomes[finish], "Provider returned finish_reason=" + finish,
                                       http_status=response.status_code)
            try:
                billed, metadata = _parse_usage(data.get("usage"))
            except (TypeError, ValueError):
                pass
            else:
                error.usage_turn = AgentTurn(content="", tool_calls=[], latency_ms=latency_ms,
                    finish_reason=finish, prompt_tokens=billed["prompt_tokens"],
                    completion_tokens=billed["completion_tokens"], total_tokens=billed["total_tokens"],
                    reasoning_tokens=billed["reasoning_tokens"],
                    provider_metadata={**metadata, "response_model": data.get("model"),
                                       "response_id": data.get("id"), "provider_outcome": outcomes[finish]})
            raise error
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
        if '_stream_created' in data:
            metadata['stream_created']=copy.deepcopy(data['_stream_created'])
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
        elif tools_requested and self.config.dialect == 'deepseek_v4_chat' and self.config.thinking_type == "enabled":
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

    def _check_minimax_status(self, data, latency_ms, response):
        status=data.get('base_resp')
        if status is None:return
        if not isinstance(status,dict) or type(status.get('status_code')) is not int:
            raise AgentProviderError('provider_invalid_response','Invalid MiniMax base_resp',http_status=response.status_code)
        native=status['status_code']
        if native==0:return
        code={1000:'provider_inference_error',1001:'provider_timeout',1002:'provider_rate_limited',
              1004:'provider_auth_error',1008:'provider_quota_exhausted',1013:'provider_inference_error',
              1027:'provider_invalid_response',1039:'provider_context_window_exceeded',
              2013:'provider_invalid_request'}.get(native,'provider_invalid_request')
        self._request_context.response_diagnostic['minimax_status_code']=native
        error=AgentProviderError(code,'MiniMax returned status_code='+str(native),http_status=response.status_code)
        try:usage,metadata=_parse_usage(data.get('usage'))
        except (TypeError,ValueError):pass
        else:
            error.billing_turn=AgentTurn(content='',tool_calls=[],latency_ms=latency_ms,finish_reason='provider_error',
                prompt_tokens=usage['prompt_tokens'],completion_tokens=usage['completion_tokens'],
                total_tokens=usage['total_tokens'],reasoning_tokens=usage['reasoning_tokens'],
                provider_metadata={**metadata,'response_model':data.get('model'),'minimax_status_code':native})
        raise error

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
            "request_context": copy.deepcopy(context),
            "stage": str(context.get("stage", "")),
            "request_sequence": context.get('request_sequence'),
            "repair_in_progress": context.get('repair'),
            "final_payload_audit": copy.deepcopy(getattr(self._request_context, 'final_payload_audit', {})),
            "endpoint": self.config.endpoint,
            "model_id": self.config.model,
            "raw_usage": copy.deepcopy(getattr(self._request_context, 'raw_usage', None)),
            "public_response": copy.deepcopy(getattr(self._request_context, 'public_response', None)),
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
            "response_diagnostic": dict(getattr(self._request_context, "response_diagnostic", {})),
            "attempt_latency_ms": round((time.time() - started_at) * 1000, 3),
        }
        with self._records_lock:
            self._request_records.append(record)
        if self.budget_governor:
            self.budget_governor.complete(record)

    def _backoff(self, retry_index: int, *, response: requests.Response | None = None) -> None:
        retry_after = _retry_after_seconds(response, self.config.max_retry_after_seconds)
        base = retry_after if retry_after is not None else (
            self.config.retry_backoff_seconds * (2 ** max(retry_index - 1, 0))
        )
        jitter = random.SystemRandom().random() * min(0.2 * max(base, 1.0), 1.0)
        time.sleep(min(base + jitter, self.config.max_retry_after_seconds))


def _validate_deepseek_messages(messages: list[AgentMessage], *, allow_images=False) -> list[AgentMessage]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("provider request requires a non-empty messages list")
    normalized = copy.deepcopy(messages)
    for index, message in enumerate(normalized):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has an invalid role")
        content = message.get('content')
        if isinstance(content, list) and role == 'user' and allow_images:
            for part in content:
                if not isinstance(part, dict) or part.get('type') not in {'text', 'image_url'}:
                    raise ValueError('Unsupported content part')
                if part['type'] == 'text' and not isinstance(part.get('text'), str):
                    raise TypeError('Text part needs text')
                if part['type'] == 'image_url' and not isinstance(part.get('image_url', {}).get('url'), str):
                    raise TypeError('Image part needs a real image URL')
        elif not isinstance(content, str):
            raise TypeError(f'message {index} content requires text or supported image parts')
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
