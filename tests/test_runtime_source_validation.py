"""Model syntax errors remain bounded Runtime feedback, before tool dispatch."""
import pytest

from atomic_skillgraph.empirical import OFFICE_TRACE_METHOD_VERSION
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.planner import dynamic
from atomic_skillgraph.empirical.program_worker import validate_context_source
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import office


@pytest.mark.parametrize('source', ['print(', 'if True:\nprint(1)', 'print("unterminated)'])
def test_invalid_source_returns_feedback_without_dispatch(tmp_path, monkeypatch, source):
    system = office(tmp_path)
    system.config['experiment']['officeqa_method_version'] = OFFICE_TRACE_METHOD_VERSION
    system.executor.checkpoint = TaskCheckpoint(tmp_path/'checkpoint')
    seen = []
    class Agent:
        config = system.config
        def call(self, stage, prompt, materials, *args, **kwargs):
            seen.append(materials)
            return {'action': 'call_tool', 'name': 'execute_python', 'arguments': {'source': source}} if len(seen) == 1 else {'action': 'finish', 'answer': 'unchanged-fixture'}
    system.executor.agent = Agent().call
    monkeypatch.setattr(system.adapter, 'call_with_broker', lambda *a, **k: pytest.fail('Invalid source reached Worker'))
    broker = Broker(system.adapter, 24, context=system.task_context)
    try:
        result = system.executor.run(system.adapter.task, system.adapter, broker, dynamic(system.adapter.task))
        assert result['reason'] == 'agent_submitted' and result['prediction'] == 'unchanged-fixture'
        assert len(seen) == 2 and not broker.events and broker.remaining_calls() == 24
        assert 'python_syntax_error' in str(seen[-1])
        assert result['history'][0]['count'] == 1
        assert system.executor.checkpoint.state['executor_state']['pending_step'] is None
    finally:
        system.close()


def test_validation_preserves_api_checks_and_infrastructure_errors(monkeypatch):
    with pytest.raises(ValueError, match='context_api_misuse'):
        validate_context_source('ctx = {}')
    validate_context_source('result = ctx.call("read", {})')
    def fail(*args, **kwargs): raise OSError('host failure')
    monkeypatch.setattr('ast.parse', fail)
    with pytest.raises(OSError, match='host failure'):
        validate_context_source('print(1)')
