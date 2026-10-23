"""Opt-in, durable retries for inference transport failures in frozen evaluations."""
from contextlib import contextmanager
from copy import deepcopy
import time

import requests

from atomic_skillgraph.agents.provider import AgentProviderError, OpenAICompatibleProvider, _sha256_json
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.core.errors import BudgetExhausted

VERSION = 'frozen-inference-retry.v1'
DELAYS = [10, 30, 60]
STATUSES = {408, 429, 500, 502, 503, 504, 529}


def validate_policy(policy):
    if (policy.get('version') != VERSION or not policy.get('authorization')
            or policy.get('max_retries_per_task') != 3 or policy.get('backoff_seconds') != DELAYS
            or policy.get('per_attempt_task_tokens') != 96000):
        raise ValueError('Expected explicit frozen inference retry policy')


def transient(attempt, error):
    detail = (attempt.get('response_diagnostic') or {}).get('error') or {}
    codes = ' '.join(str(detail.get(k, '')) for k in ('code', 'type', 'message')).lower()
    if any(x in codes for x in ('quota', 'balance', 'authentication', 'permission', 'insufficient')):
        return False
    if attempt.get('http_status') in STATUSES:
        return True
    return (attempt.get('http_status') is None and
        isinstance(error.__cause__, (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                                    requests.exceptions.ChunkedEncodingError)) and
        not isinstance(error.__cause__, requests.exceptions.SSLError))


class RetryGovernor(BudgetGovernor):
    def __init__(self, path, *, policy, **limits):
        validate_policy(policy)
        self.policy = deepcopy(policy)
        super().__init__(path, **limits)
        if self.state.setdefault('inference_retry_policy', self.policy) != self.policy:
            raise ValueError('Frozen retry policy changed')
        save_json(self.path, self.state)

    def _context(self, context):
        adjusted = dict(context)
        entry = self.state.get('inference_retries', {}).get(context.get('parent_task_id'), {})
        failed = [self.state['attempts'][key] for key in entry.get('failures', [])]
        # The extra allowance pays only recorded transient failures. It never
        # changes generation caps, ordinary task spending, or the global ceiling.
        credit = sum(row.get('accounted_tokens', row['reserved_tokens']) for row in failed
                     if row['context'].get('parent_scope') == context.get('parent_scope'))
        if credit and context.get('parent_token_limit') is not None:
            if len(failed) > self.policy['max_retries_per_task']:
                raise AgentProviderError('provider_retry_exhausted', 'Task retry allowance exhausted')
            adjusted.update(normal_parent_token_limit=context['parent_token_limit'],
                infrastructure_retry_allowance=credit, parent_token_limit=context['parent_token_limit'] + credit)
        return adjusted

    def headroom(self, context):
        return super().headroom(self._context(context))

    def admit(self, audit_id, payload, context):
        entry = self.state.get('inference_retries', {}).get(context.get('parent_task_id'), {})
        if entry.get('pending'):
            if len(entry['failures']) > self.policy['max_retries_per_task']:
                raise AgentProviderError('provider_retry_exhausted', 'Task retry allowance exhausted')
            if _sha256_json(payload) != entry['payload_fingerprint']:
                raise ValueError('Retry payload changed; refusing a different inference request')
        return super().admit(audit_id, payload, self._context(context))

    def retain_failure(self, attempt):
        with self.lock:
            key = attempt['request_id']
            row = self.state['attempts'][key]
            task = row['context']['parent_task_id']
            if row['status'] not in {'metered', 'unknown_billing'} or row['outcome'] != 'error':
                raise ValueError('Only a finished failed HTTP attempt can be retained')
            if row.get('accounted_tokens', row['reserved_tokens']) > self.policy['per_attempt_task_tokens']:
                raise ValueError('Failed attempt exceeds the authorized retry allowance')
            entry = self.state.setdefault('inference_retries', {}).setdefault(task, {'failures': []})
            if key not in entry['failures']:
                entry['failures'].append(key)
            entry.update(pending=True, payload_fingerprint=attempt['payload_fingerprint'])
            count = len(entry['failures'])
            entry['retry_not_before'] = time.time() + DELAYS[min(count, len(DELAYS)) - 1]
            if row['status'] == 'unknown_billing':
                receipt = self.state.setdefault('inherited_reservation_recovery', {
                    'schema': 'billing-reservation-recovery.v1',
                    'authorization': self.policy['authorization'], 'attempts': {}})
                receipt['attempts'][key] = {'entry_hash': digest(row), 'reserved_tokens': row['reserved_tokens']}
            save_json(self.path, self.state)
            return entry

    def complete(self, record):
        super().complete(record)
        if record['outcome'] == 'success':
            task = self.state['attempts'][record['request_id']]['context'].get('parent_task_id')
            entry = self.state.get('inference_retries', {}).get(task)
            if entry:
                entry.update(pending=False, successful_attempt=record['request_id'])
                save_json(self.path, self.state)


@contextmanager
def retry_guard(governor, directory, stop_requested=lambda: False):
    original = OpenAICompatibleProvider.complete

    def complete(provider, messages, **kwargs):
        if provider.budget_governor is not governor:
            return original(provider, messages, **kwargs)
        if provider.config.max_retries != 0:
            raise ValueError('Do not stack provider and durable retry loops')
        context = deepcopy(getattr(provider._request_context, 'value', {}))
        task = context.get('parent_task_id')
        if not task or context.get('train_parent') or context.get('purpose') not in {'runtime', 'finish_only'}:
            raise ValueError('Retry policy is limited to frozen runtime inference')
        try:
            while True:
                entry = governor.state.get('inference_retries', {}).get(task, {})
                count = len(entry.get('failures', []))
                if entry.get('pending') and count > governor.policy['max_retries_per_task']:
                    raise AgentProviderError('provider_retry_exhausted', 'Transient failure persisted after three retries')
                deadline = entry.get('retry_not_before', 0) if entry.get('pending') else 0
                while time.time() < deadline:
                    if stop_requested():
                        raise SystemExit(75)
                    time.sleep(min(1, deadline - time.time()))
                if stop_requested():
                    raise SystemExit(75)
                provider._request_context.value = {**context, 'infrastructure_retry_index': count}
                offset = provider.request_record_count
                try:
                    return original(provider, messages, **kwargs)
                except BudgetExhausted as exc:
                    if entry.get('pending'):
                        raise AgentProviderError('provider_retry_budget_exhausted',
                            'Retry cannot fit the remaining cumulative budget') from exc
                    raise
                except AgentProviderError as exc:
                    attempts = provider.request_records_since(offset)
                    if len(attempts) != 1 or not transient(attempts[0], exc):
                        raise
                    attempt = attempts[0]
                    save_json(directory / (attempt['request_id'] + '.json'), attempt)
                    governor.retain_failure(attempt)
        finally:
            provider._request_context.value = context

    OpenAICompatibleProvider.complete = complete
    try:
        yield
    finally:
        OpenAICompatibleProvider.complete = original
