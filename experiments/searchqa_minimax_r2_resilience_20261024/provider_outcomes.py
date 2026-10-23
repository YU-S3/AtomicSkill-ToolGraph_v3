"""Terminal refusals for frozen evaluation, with conservative, auditable billing."""
from contextlib import contextmanager
from copy import deepcopy
import json
import re
from pathlib import Path

from atomic_skillgraph.agents.provider import AgentProviderError
from atomic_skillgraph.empirical.answer_executor import AnswerExecutor
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import digest

VERSION = 'provider-terminal-refusal.v3'
POLICY = {
    'version': VERSION,
    'authorization': '用户要求补齐拒答处理和有限重试后续跑；保留原额度、原始记录及未知费用预留。',
    'refusal': 'Official empty-answer scoring; keep in denominator; no retry or prompt alteration.',
    'billing': 'Missing usage stays unknown; retain full original reservation, never fabricate zero usage.',
    'other_errors': 'Stop without scoring; unknown transport/inference failures are not refusals.',
}


def minimax_code(attempt):
    if attempt.get('final_payload_audit', {}).get('provider_snapshot', {}).get('dialect') != 'minimax_m3_chat':
        return None
    diagnostic = attempt.get('response_diagnostic') or {}
    code = diagnostic.get('minimax_status_code')
    error = diagnostic.get('error') or {}
    if code is None:
        code = error.get('code')
    if str(code).isdigit():
        return int(code)
    match = re.search(r'\((\d{4})\)\s*$', error.get('message', ''))
    return int(match[1]) if match else None


def refusal_evidence(attempt):
    dialect = attempt.get('final_payload_audit', {}).get('provider_snapshot', {}).get('dialect')
    diagnostic = attempt.get('response_diagnostic') or {}
    error = diagnostic.get('error') or {}
    code = minimax_code(attempt)
    if code in (1026, 1027) and attempt.get('http_status') in (200, 400, 422):
        return {'source': 'minimax_content_filter', 'http_status': attempt['http_status'],
                'error_type': error.get('type'), 'error_code': code, 'finish_reason': None}
    if (dialect == 'moonshot_k2_chat' and attempt.get('http_status') == 400
            and error.get('type') == 'content_filter'):
        return {'source': 'http_error', 'http_status': 400, 'error_type': 'content_filter',
                'error_code': error.get('code'), 'finish_reason': None}
    choices = (attempt.get('public_response') or {}).get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    finish = choices[0].get('finish_reason')
    if not (finish == 'content_filter' or (finish == 'sensitive' and dialect == 'glm_53_chat')):
        return None
    stream = (attempt.get('response_diagnostic') or {}).get('stream')
    if attempt.get('http_status') != 200 or (stream and stream.get('completed') is not True):
        return None
    return {'source': 'completion', 'http_status': 200, 'error_type': None, 'finish_reason': finish}


def last_refusal(audit, task_id):
    attempts = [a for request in audit.get('requests', []) for a in request.get('http_attempts', [])]
    if not attempts or refusal_evidence(attempts[-1]) is None:
        return None
    attempt = attempts[-1]
    if attempt.get('request_context', {}).get('parent_task_id') != task_id:
        raise ValueError('Refusal task identity mismatch')
    return attempt


def retain_reservation(governor, attempt, reason='terminal_refusal'):
    """Keep the immutable charge while recording the authorized recovery reason."""
    with governor.lock:
        entry = governor.state['attempts'][attempt['request_id']]
        if entry['status'] == 'metered':
            return {'status': 'metered', 'actual_usage': deepcopy(entry['actual_usage'])}
        assert entry['status'] == 'unknown_billing' and 'accounted_tokens' not in entry
        assert entry['reserved_tokens'] > 0 and entry['context'] == attempt['request_context']
        assert attempt.get('usage') is None
        receipt = governor.state.setdefault('inherited_reservation_recovery', {
            'schema': 'billing-reservation-recovery.v1',
            'authorization': deepcopy(POLICY), 'attempts': {}})
        expected = {'entry_hash': digest(entry), 'reserved_tokens': entry['reserved_tokens']}
        prior = receipt['attempts'].setdefault(attempt['request_id'], expected)
        assert prior == expected
        governor.state.setdefault('provider_recovery_events', {}).setdefault(attempt['request_id'],
            {'policy': VERSION, 'reason': reason, 'attempt_digest': digest(attempt)})
        save_json(governor.path, governor.state)
        return {'status': 'unknown_billing', 'reserved_tokens': entry['reserved_tokens'],
                'actual_usage': None, 'reservation_receipt': str(governor.path)}


def recover_saved_refusals(governor, paths):
    """A crash after a refusal receipt never causes the same request to be resent."""
    for path in paths:
        audit = json.loads(Path(path).read_text())
        attempts = [a for q in audit.get('requests', []) for a in q.get('http_attempts', [])]
        if not attempts:
            continue
        task_id = attempts[-1].get('request_context', {}).get('parent_task_id')
        attempt = last_refusal(audit, task_id)
        if attempt:
            retain_reservation(governor, attempt)


@contextmanager
def refusal_guard(governor, replay_sources=None):
    original = AnswerExecutor.run
    replay_sources = replay_sources or {}

    def require_available():
        if governor.billing_blocked:
            raise AgentProviderError('provider_billing_unresolved',
                                     'Unresolved nonterminal provider billing; stop without scoring')

    def terminal(system, task, audit, attempt, origin=None):
        assert system.readonly and task.split != 'train'
        billing = retain_reservation(governor, attempt)
        require_available()
        result = deepcopy(getattr(system.executor, 'partial_execution', {}))
        result.setdefault('attempts', [])
        evidence = refusal_evidence(attempt)
        assert evidence is not None
        result.update(prediction=None, reason='provider_content_filter', answer_status='provider_refusal',
                      empty_answer=True, completion_truncated=False, all_authored_nodes_completed=False,
                      provider_finish_reason=evidence['finish_reason'],
                      provider_refusal_evidence=evidence,
                      provider_request_id=attempt['provider_request_id'],
                      refusal_policy=VERSION, refusal_billing=billing, refused_request_id=attempt['request_id'])
        if origin:
            result['recovered_request_audit'] = origin
        current_ids = {v['id'] for v in system.requests[system._request_start:]}
        system.requests.extend(deepcopy(v) for v in audit['requests'] if v['id'] not in current_ids)
        system._save_requests()
        if system.checkpoint:
            system.checkpoint.advance(system.checkpoint.state['stage'], terminal_refusal=result)
        return result

    def guarded(executor, task, *args, **kwargs):
        system = executor.system
        if system.checkpoint and system.checkpoint.state.get('terminal_refusal'):
            require_available()
            audit = json.loads(Path(system.audit_path).read_text())
            system.requests.extend(deepcopy(audit['requests']))
            return deepcopy(system.checkpoint.state['terminal_refusal'])
        source = replay_sources.get(task.task_id)
        path = Path(source['path']) if source else Path(system.audit_path)
        if path.exists():
            audit = json.loads(path.read_text())
            if source:
                assert digest(audit) == source['audit_digest']
            refusal = last_refusal(audit, task.task_id)
            if refusal:
                return terminal(system, task, audit, refusal, source)
            if source:
                raise ValueError('Inherited response is not a completed provider refusal')
        require_available()
        try:
            result = original(executor, task, *args, **kwargs)
        except Exception:
            audit = json.loads(Path(system.audit_path).read_text()) if Path(system.audit_path).exists() else {}
            refusal = last_refusal(audit, task.task_id)
            if refusal:
                return terminal(system, task, audit, refusal)
            require_available()
            raise
        require_available()
        return result

    AnswerExecutor.run = guarded
    try:
        yield
    finally:
        AnswerExecutor.run = original
