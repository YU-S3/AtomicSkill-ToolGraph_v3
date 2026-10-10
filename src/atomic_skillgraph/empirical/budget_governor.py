"""Durable admission and billing for actual HTTP attempts, including retries."""
import json
from pathlib import Path
from threading import RLock

from ..core.errors import BudgetExhausted, FailureLayer
from .checkpoint import save_json


def input_token_bound(payload):
    # Byte fallback is conservative for text BPE; no model tokenizer is assumed.
    return len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


class BudgetGovernor:
    def __init__(self, path, *, token_limit=5000000, finish_reserve=500000, request_limit=205,
                 proposal_repair_limit=None, validation_limit=None):
        self.path, self.lock = Path(path), RLock()
        self.limits = dict(token_limit=token_limit, finish_reserve=finish_reserve, request_limit=request_limit)
        if proposal_repair_limit is not None:
            self.limits['proposal_repair_limit'] = proposal_repair_limit
        if validation_limit is not None: self.limits['validation_limit'] = validation_limit
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'limits': self.limits, 'attempts': {}, 'unknown_billing': False}
        if self.state['limits'] != self.limits: raise ValueError('Governor identity changed')
        if any(a['status'] == 'admitted' for a in self.state['attempts'].values()):
            self.state['unknown_billing'] = True
            save_json(self.path,self.state)

    def admit(self, audit_id, payload, context):
        with self.lock:
            attempts = self.state['attempts']
            if audit_id in attempts: raise ValueError('Duplicate HTTP audit identity')
            if context.get('decision_purpose') == 'repair' and not self.repair_available():
                raise ValueError('Batch proposal repair limit reached')
            reserved = input_token_bound(payload) + payload.get('max_tokens', payload.get('max_completion_tokens', 0))
            used = sum(a.get('accounted_tokens', a['reserved_tokens']) for a in attempts.values())
            cap = self.limits['token_limit'] - (0 if context.get('purpose') == 'finish_only' else self.limits['finish_reserve'])
            scope = context.get('episode_scope')
            per_episode = context.get('episode_request_limit')
            exhausted = (self.state['unknown_billing'] or len(attempts) >= self.limits['request_limit']
                or used + reserved > cap or (per_episode is not None and sum(
                    a['context'].get('episode_scope') == scope for a in attempts.values()) >= per_episode))
            parent = context.get('parent_scope')
            parent_cap = context.get('parent_token_limit')
            parent_used = sum(a.get('accounted_tokens',a['reserved_tokens']) for a in attempts.values()
                              if parent and a['context'].get('parent_scope') == parent)
            if parent_cap is not None and parent_used + reserved + context.get('parent_finish_reserve',0) > parent_cap:
                raise BudgetExhausted('parent_task_budget_exhausted','Parent task HTTP admission refused')
            if exhausted:
                raise BudgetExhausted('diagnostic_budget_exhausted', 'HTTP admission refused', layer=FailureLayer.RUNTIME_AGENT)
            attempts[audit_id] = {'reserved_tokens': reserved, 'input_estimate': input_token_bound(payload),
                'estimate_method': 'utf8_byte_upper_bound_text', 'context': dict(context), 'status': 'admitted'}
            save_json(self.path, self.state)

    def admit_validation(self, identity):
        with self.lock:
            rows = self.state.setdefault('worker_validations', [])
            if identity in rows: raise RuntimeError('Unresolved local validation execution; no blind replay')
            if self.limits.get('validation_limit') is not None and len(rows) >= self.limits['validation_limit']:
                raise BudgetExhausted('local_validation_budget_exhausted','Local validation execution limit reached')
            rows.append(identity)
            save_json(self.path,self.state)

    def validation_available(self):
        limit = self.limits.get('validation_limit')
        return limit is None or len(self.state.get('worker_validations', [])) < limit

    def repair_available(self):
        limit = self.limits.get('proposal_repair_limit')
        return limit is None or sum(a['context'].get('decision_purpose') == 'repair'
                                   for a in self.state['attempts'].values()) < limit

    def complete(self, record):
        with self.lock:
            entry = self.state['attempts'][record['request_id']]
            if entry['status'] != 'admitted': return
            usage = record.get('raw_usage')
            total = usage.get('total_tokens') if isinstance(usage, dict) else None
            if not isinstance(total, int) or total < 0:
                entry['status'] = 'unknown_billing'
                self.state['unknown_billing'] = True
            else:
                entry.update(status='metered', accounted_tokens=total, actual_usage=usage,
                    estimate_error=total-entry['reserved_tokens'])
            entry['outcome'] = record['outcome']
            save_json(self.path, self.state)

