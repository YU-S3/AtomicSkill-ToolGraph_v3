"""Separate verified failed transport from productive task tokens, never from global cost."""
from copy import deepcopy

from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import digest


class TransportBudgetGovernor(BudgetGovernor):
    def transport_failure(self, attempt, evidence):
        """Called only after the retry classifier accepts a terminal transport failure."""
        with self.lock:
            key = attempt['request_id']
            row = self.state['attempts'][key]
            assert row['outcome'] == attempt['outcome'] == 'error'
            assert row['status'] in ('metered', 'unknown_billing')
            assert row['context'] == attempt['request_context']
            receipt = {'entry_hash': digest(row), 'attempt_hash': digest(attempt),
                       'charge': row.get('accounted_tokens', row['reserved_tokens']),
                       'evidence': evidence, 'policy': 'transport-task-accounting.v1'}
            prior = self.state.setdefault('transport_failures', {}).setdefault(key, receipt)
            assert prior == receipt
            save_json(self.path, self.state)

    def _effective(self, context):
        result = deepcopy(context)
        binding = self.state.get('transport_continuations', {}).get(context.get('parent_task_id'))
        if binding:
            result['parent_scope'] = binding['parent_scope']
        rows = {}
        for key, receipt in self.state.get('transport_failures', {}).items():
            row = self.state['attempts'][key]
            assert digest(row) == receipt['entry_hash'], 'Transport accounting identity changed'
            assert row.get('accounted_tokens', row['reserved_tokens']) == receipt['charge']
            if row['context'].get('parent_scope') == result.get('parent_scope'):
                rows[key] = row
        credit = sum(row.get('accounted_tokens', row['reserved_tokens']) for row in rows.values())
        if result.get('parent_token_limit') is not None:
            result['parent_token_limit'] += credit
        if result.get('budget_pool_limit') is not None:
            result['budget_pool_limit'] += sum(row.get('accounted_tokens', row['reserved_tokens'])
                for row in rows.values() if row['context'].get('budget_pool') == result.get('budget_pool'))
        return result, {'failure_ids': list(rows), 'tokens': credit,
                        'productive_parent_limit': context.get('parent_token_limit'),
                        'effective_parent_limit': result.get('parent_token_limit'),
                        'global_charge_unchanged': True}

    def headroom(self, context):
        with self.lock:
            effective, adjustment = self._effective(context)
            result = super().headroom(effective)
            result['transport_adjustment'] = adjustment
            return result

    def admit(self, audit_id, payload, context):
        with self.lock:
            effective, adjustment = self._effective(context)
            # Durable admission occurs before HTTP. The second write restores the
            # original audit context; an interruption between writes fails closed.
            super().admit(audit_id, payload, effective)
            row = self.state['attempts'][audit_id]
            row['context'] = deepcopy(context)
            if adjustment['tokens']:
                row['transport_budget_adjustment'] = adjustment
            save_json(self.path, self.state)
