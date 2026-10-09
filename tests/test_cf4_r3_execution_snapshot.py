"""Budget snapshots through production state modules; no HTTP or source execution."""
from copy import deepcopy

import pytest
import requests

from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import object_schema
from test_cf2_contracts import office
from test_empirical import program


@pytest.mark.parametrize('consumption,hard,checkpoint', [
    ('used', True, True), ('unused', True, False),
    ('replaced', True, True), ('used', False, True),
])
def test_budget_snapshot_preserves_actual_contribution(tmp_path, monkeypatch, consumption, hard, checkpoint):
    monkeypatch.setattr(requests, 'post', lambda *a, **k: pytest.fail('No model HTTP allowed'))
    s = office(tmp_path)
    if checkpoint: s.checkpoint = TaskCheckpoint(tmp_path/'cp')
    (s.adapter.corpus/'b.txt').write_text('independent fallback')
    asset = s.bank.put('program', program(
        "def run(ctx, inputs):\n    raise AssertionError('Fixture source must never execute')",
        outputs=object_schema({'path': {'type': 'string'}}, ['path'])))
    plan = {'goal': 'find target', 'nodes': [
        {'id': 'producer', 'execution_mode': 'dynamic', 'goal': 'find path', 'args': {}},
        {'id': 'consumer', 'execution_mode': 'dynamic', 'goal': 'read path',
         'args': {'path': {'from': 'producer', 'field': 'path'}}},
    ], 'outputs': {'path': {'from': 'producer', 'field': 'path'}}}
    s.planner.plan = lambda *a: deepcopy(plan)
    workers, scores, learning, records = [], [], [], []
    def inert_worker(p, arguments, broker):
        workers.append(p['id'])
        return {'status': 'ok', 'outputs': {'path': 'a.txt'}}
    monkeypatch.setattr(s.worker, 'execute', inert_worker)
    actions = iter([
        {'action': 'call_program', 'name': asset['id'], 'arguments': {}},
        {'action': 'call_tool', 'name': 'read',
         'arguments': {'path': 'a.txt' if consumption == 'used' else 'b.txt', 'offset': 0},
         'used_inputs': ['path'] if consumption != 'unused' else [],
         'replaced_inputs': ['path'] if consumption == 'replaced' else []},
    ])
    def runtime(*a, **k):
        try: return next(actions)
        except StopIteration:
            raise BudgetExhausted('empirical_token_budget_exhausted', 'synthetic role budget')
    s.executor.agent = runtime
    s.adapter.evaluate = lambda sealed: (scores.append(sealed) or {'hard': hard, 'raw_score': float(hard)})
    s.learner.learn = lambda *a: (learning.append(True) or {'decision': 'no_change'})
    record = s.bank.record
    def capture(attempt):
        records.append(deepcopy(attempt))
        record(attempt)
    monkeypatch.setattr(s.bank, 'record', capture)
    try:
        trace = s.run_task(s.adapter.task, learn=True)
        execution = trace['execution']
        attempt = execution['attempts'][0]
        used, positive = consumption == 'used', consumption == 'used' and hard
        assert len(scores) == len(learning) == len(records) == len(workers) == 1
        assert execution['prediction'] is None and execution['reason'] == 'token_budget_exhausted'
        assert attempt['outputs_consumed'] == used
        assert not attempt['submission_by_program'] and not attempt['terminal_by_program']
        assert attempt['output_contract_status'] == 'valid' and attempt['local_check'] == 'unavailable'
        assert (attempt['outcome'], attempt['basis']) == (('positive', 'task_outcome') if positive else ('normal', None))
        assert records == [attempt] == s.bank.attempts(asset['id'])
        assert s.bank.get(asset['id'])['state'] == 'candidate'
        assert execution['plan'] == plan and execution['values'][0]['node'] == 'producer'
        assert [h.get('program') or h.get('tool') for h in execution['history']] == [asset['id'], 'read']
        assert execution['input_reads'] == (['consumer'] if used else [])
        assert execution['replaced_inputs'] == ({'consumer': ['path']} if consumption == 'replaced' else {})
        assert execution['submission_producer_attempt_id'] is None
        assert not s.requests and not s.usage.events
        assert trace['native_call_attempts'] == 1 and trace['tools'][0]['result']['accepted']
        if checkpoint:
            state = s.checkpoint.state['executor_state']
            assert (attempt['id'] in state['consumed']) == used
            assert s.checkpoint.state['stage'] == 'learning_finished'
            assert s.checkpoint.state['trace']['execution'] == execution
    finally: s.close()
