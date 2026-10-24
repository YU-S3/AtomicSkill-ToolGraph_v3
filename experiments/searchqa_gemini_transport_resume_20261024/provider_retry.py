"""Bounded read-only inference retries; unknown usage remains fully reserved."""
from contextlib import contextmanager
from copy import deepcopy
import time

import requests

from atomic_skillgraph.agents.provider import AgentProviderError, OpenAICompatibleProvider
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import digest
from provider_outcomes import minimax_code, refusal_evidence, retain_reservation

POLICY = {'version': 'bounded-provider-retry.v3', 'max_retries': 3,
          'dialects': ['minimax_m3_chat', 'glm_53_chat', 'rootflow_gemini38_chat'],
          'backoff_seconds': [30, 90, 180], 'minimum_completion_cap': 1024,
          'http_statuses': [408, 429, 500, 502, 503, 504, 529],
          'stream_error_messages': ['Incomplete provider SSE termination'],
          'minimax_codes': [1000, 1001, 1002, 1013, 1024, 1033, 1041, 1045, 2062, 2064, 2066],
          'billing': 'Every physical attempt is retained; absent usage keeps its full reservation.',
          'payload': 'Same messages/tools/model and original output cap; failed transport remains globally charged, outside productive task tokens.',
          'exhaustion': 'Stop without scoring; never reset the retry allowance on resume.'}


class ProviderRetryStopped(BaseException):
    """An infrastructure stop must bypass model repair and answer finalization."""


def transient(attempt, error):
    if refusal_evidence(attempt):
        return False
    code = minimax_code(attempt)
    if code is not None:
        return code in POLICY['minimax_codes']
    diagnostic = (attempt.get('response_diagnostic') or {}).get('error') or {}
    kind = str(diagnostic.get('type', '')).lower()
    if any(word in kind for word in ('quota', 'balance', 'authentication', 'permission')):
        return False
    if attempt.get('http_status') in POLICY['http_statuses']:
        return True
    cause = error.__cause__
    stream = (attempt.get('response_diagnostic') or {}).get('stream') or {}
    if (attempt.get('http_status') in (None, 200) and stream.get('completed') is False and isinstance(cause, ValueError)
            and str(cause) in POLICY['stream_error_messages']):
        return True
    return (attempt.get('http_status') is None
            and isinstance(cause, (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                                   requests.exceptions.ChunkedEncodingError))
            and not isinstance(cause, requests.exceptions.SSLError))


def fit_cap(provider, messages, kwargs, context):
    """Conserve both the original completion ceiling and all paid/unknown charges."""
    payload = provider._build_payload(messages, list(kwargs.get('tools') or []), kwargs.get('tool_choice'))
    field = provider.config.token_limit_field
    original = kwargs.get('max_completion_tokens', payload[field])
    payload[field] = original
    headroom = provider.budget_governor.headroom(context)
    if headroom['unknown_billing'] or headroom['requests'] <= context.get('finish_request_reserve', 0):
        raise ProviderRetryStopped('Retry admission blocked by billing or request budget')
    cap = min(original, headroom['tokens'] - input_token_bound(payload))
    payload[field] = cap
    cap = min(cap, headroom['tokens'] - input_token_bound(payload))
    if cap < POLICY['minimum_completion_cap']:
        raise ProviderRetryStopped('Retry cannot fit the remaining task/global budget')
    return cap


def bound_context(governor, provider, messages, kwargs, context):
    binding = governor.state.get('transport_continuations', {}).get(context.get('parent_task_id'))
    if not binding:
        return context
    result = {**context, 'parent_scope': binding['parent_scope'],
              'continuation_origin': {'parent_scope': context['parent_scope'],
                                      'logical_decision_id': context['logical_decision_id']}}
    entry = governor.state['provider_retries'][binding['retry_key']]
    if not entry.get('completed'):
        assert context['purpose'] == binding['purpose'], 'Unexpected interrupted decision'
        wire = provider._build_payload(messages, list(kwargs.get('tools') or []), kwargs.get('tool_choice'))
        field = provider.config.token_limit_field
        wire[field] = kwargs.get('max_completion_tokens', wire[field])
        assert digest(wire) == binding['payload_digest'], 'Continuation payload changed'
        result['logical_decision_id'] = binding['logical_decision_id']
    return result


def retry_preflight(governor, task_id):
    """A pending transport stop must not become an Agent final-answer fallback on restart."""
    for entry in governor.state.get('provider_retries', {}).values():
        if entry['context']['parent_task_id'] != task_id or not entry.get('next_retry'):
            continue
        if entry.get('stopped'):
            raise ProviderRetryStopped(entry['stopped'])
        headroom = governor.headroom(entry['context'])
        required = entry['retry_required_tokens']
        if (headroom['unknown_billing'] or headroom['tokens'] < required
                or headroom['requests'] <= entry['context'].get('finish_request_reserve', 0)):
            raise ProviderRetryStopped('Pending transport retry cannot fit the unchanged global/productive budget')


@contextmanager
def retry_guard(governor, directory, stop_requested=lambda: False):
    original = OpenAICompatibleProvider.complete

    def complete(provider, messages, **kwargs):
        if provider.config.dialect not in POLICY['dialects']:
            return original(provider, messages, **kwargs)
        assert provider.config.max_retries == 0, 'Do not stack two retry loops'
        original_context = deepcopy(getattr(provider._request_context, 'value', {}))
        context = bound_context(governor, provider, messages, kwargs, original_context)
        key = digest({k: context.get(k) for k in ('parent_scope', 'parent_task_id', 'logical_decision_id', 'purpose')})
        assert context.get('logical_decision_id') and context.get('parent_task_id')
        entry = governor.state.get('provider_retries', {}).get(key)
        if entry and entry.get('stopped'):
            raise ProviderRetryStopped(entry['stopped'])
        index = len(entry['failures']) if entry else 0
        try:
            while True:
                if stop_requested():
                    raise ProviderRetryStopped('Stop requested before provider attempt')
                if entry and entry.get('next_retry'):
                    # Never lower the model's output budget because transport failed.
                    cap = fit_cap(provider, messages, kwargs, context)
                    payload = provider._build_payload(messages, list(kwargs.get('tools') or []), kwargs.get('tool_choice'))
                    expected = kwargs.get('max_completion_tokens', payload[provider.config.token_limit_field])
                    if cap < expected:
                        raise ProviderRetryStopped('Original completion cap cannot fit global/productive budget')
                    while time.time() < entry.get('retry_not_before', 0):
                        if stop_requested():
                            raise ProviderRetryStopped('Stop requested during retry backoff')
                        time.sleep(max(0, min(1, entry['retry_not_before'] - time.time())))
                provider._request_context.value = {**context, **({'infrastructure_retry_index': index} if index else {})}
                offset = provider.request_record_count
                try:
                    turn = original(provider, messages, **kwargs)
                except AgentProviderError as exc:
                    records = provider.request_records_since(offset)
                    if len(records) != 1 or not transient(records[-1], exc):
                        raise
                    attempt = records[-1]
                    save_json(directory / (attempt['request_id'] + '.json'), attempt)
                    with governor.lock:
                        entry = governor.state.setdefault('provider_retries', {}).setdefault(key,
                            {'policy': deepcopy(POLICY), 'context': context, 'failures': [], 'next_retry': False})
                        entry['failures'].append(attempt['request_id'])
                        entry.setdefault('retry_required_tokens', governor.state['attempts'][attempt['request_id']]['reserved_tokens'])
                        index = len(entry['failures'])
                        entry['next_retry'] = True
                        if index > POLICY['max_retries']:
                            entry['stopped'] = 'Transient failure persisted after three retries'
                        # Persist the finite retry count in the same reservation-recovery write.
                        retain_reservation(governor, attempt, 'bounded_transient_retry')
                        governor.transport_failure(attempt, str(directory / (attempt['request_id'] + '.json')))
                        entry['retry_not_before'] = time.time() + POLICY['backoff_seconds'][min(index, 3) - 1]
                        save_json(governor.path, governor.state)
                    if entry.get('stopped'):
                        raise ProviderRetryStopped(entry['stopped']) from exc
                    continue
                if entry:
                    with governor.lock:
                        entry['next_retry'] = False
                        entry['completed'] = True
                        save_json(governor.path, governor.state)
                    turn.provider_metadata['infrastructure_retry_count'] = index
                return turn
        finally:
            provider._request_context.value = original_context

    OpenAICompatibleProvider.complete = complete
    try:
        yield
    finally:
        OpenAICompatibleProvider.complete = original
