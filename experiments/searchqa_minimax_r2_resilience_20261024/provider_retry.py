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

POLICY = {'version': 'bounded-provider-retry.v1', 'max_retries': 3,
          'backoff_seconds': [10, 30, 60], 'minimum_completion_cap': 1024,
          'http_statuses': [408, 429, 500, 502, 503, 504, 529],
          'minimax_codes': [1000, 1001, 1002, 1013, 1024, 1033, 1041, 1045, 2062, 2064, 2066],
          'billing': 'Every physical attempt is retained; absent usage keeps its full reservation.',
          'payload': 'Same messages/tools/model; retry output cap can only shrink to fit existing task/global budgets.',
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


@contextmanager
def retry_guard(governor, directory, stop_requested=lambda: False):
    original = OpenAICompatibleProvider.complete

    def complete(provider, messages, **kwargs):
        if provider.config.dialect != 'minimax_m3_chat':
            return original(provider, messages, **kwargs)
        assert provider.config.max_retries == 0, 'Do not stack two retry loops'
        context = deepcopy(getattr(provider._request_context, 'value', {}))
        key = digest({k: context.get(k) for k in ('parent_scope', 'parent_task_id', 'logical_decision_id', 'purpose')})
        assert context.get('logical_decision_id') and context.get('parent_task_id')
        entry = governor.state.get('provider_retries', {}).get(key)
        if entry and entry.get('stopped'):
            raise ProviderRetryStopped(entry['stopped'])
        if entry and entry.get('next_retry'):
            # A process restart retains the previous failed-attempt count and cap.
            kwargs['max_completion_tokens'] = fit_cap(provider, messages, kwargs, context)
        index = len(entry['failures']) if entry else 0
        try:
            while True:
                if stop_requested():
                    raise ProviderRetryStopped('Stop requested before provider attempt')
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
                        index = len(entry['failures'])
                        entry['next_retry'] = True
                        if index > POLICY['max_retries']:
                            entry['stopped'] = 'Transient failure persisted after three retries'
                        # Persist the finite retry count in the same reservation-recovery write.
                        retain_reservation(governor, attempt, 'bounded_transient_retry')
                        save_json(governor.path, governor.state)
                    if entry.get('stopped'):
                        raise ProviderRetryStopped(entry['stopped']) from exc
                    try:
                        cap = fit_cap(provider, messages, kwargs, context)
                    except ProviderRetryStopped as stopped:
                        entry['stopped'] = str(stopped)
                        save_json(governor.path, governor.state)
                        raise
                    kwargs['max_completion_tokens'] = cap
                    entry['retry_completion_cap'] = cap
                    save_json(governor.path, governor.state)
                    for _ in range(POLICY['backoff_seconds'][index - 1]):
                        if stop_requested():
                            raise ProviderRetryStopped('Stop requested during retry backoff')
                        time.sleep(1)
                    continue
                if entry:
                    with governor.lock:
                        entry['next_retry'] = False
                        entry['completed'] = True
                        save_json(governor.path, governor.state)
                    turn.provider_metadata['infrastructure_retry_count'] = index
                return turn
        finally:
            provider._request_context.value = context

    OpenAICompatibleProvider.complete = complete
    try:
        yield
    finally:
        OpenAICompatibleProvider.complete = original
