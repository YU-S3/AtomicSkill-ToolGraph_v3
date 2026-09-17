"""Acceptance driver bookkeeping preserves failures without bypassing the engine."""
import json
import pytest
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments import r102_dataflow_check as driver
from experiments import self_tooling_targeted as route
from experiments.fakes import FakeReply, ScriptedAgentProvider
from fixtures.r921_self_tooling_cases import fixture_config


def test_builder_setup_exception_keeps_usage_and_immutable_trace(tmp_path, monkeypatch):
    checkout = tmp_path / 'code'
    checkout.mkdir()
    monkeypatch.setattr(driver, 'REPO', checkout)
    harness = route.CandidateHarness(route.RouteCase('driver_failure'))
    task = route.fixture_task(harness.case)
    harness.load_tasks = lambda **kwargs: [task]
    runtime = ScriptedAgentProvider([FakeReply.tool('select_fixture_input', {'destination': 'cabinet_1'})])
    class FailedBuilder:
        def snapshot(self):
            return {'provider': 'fixture', 'fixture_generated': True}
        def complete(self, *args, **kwargs):
            raise RuntimeError('injected setup failure')
    monkeypatch.setattr(driver, 'AtomicSkillGraphSystem', lambda config:
        AtomicSkillGraphSystem(config, harness=harness, provider={
            'runtime_preparation': runtime, 'tool_builder': FailedBuilder()}))
    monkeypatch.setattr('atomic_skillgraph.agents.provider_probe.ensure_provider_capability',
        lambda config, **kwargs: {'passed': True, 'code_hash': kwargs['code_hash'], 'config_hash': kwargs['config_hash']})
    output = tmp_path / 'audit'
    with pytest.raises(RuntimeError, match='injected setup failure'):
        driver.run(fixture_config(tmp_path), output)
    usage = json.loads((output / 'all_usage.json').read_text())
    assert len(usage) == 1 and usage[0]['total_tokens'] > 0
    traces = list((output / 'trace_store' / 'traces').glob('trace_*.json'))
    traces = [path for path in traces if not path.name.endswith(('.claim.json', '.owner.json'))]
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text())
    assert trace['llm_usage'] == usage
    assert trace['metadata']['diagnostic_setup_or_execution_error']['type'] == 'RuntimeError'
    assert not trace['environment_actions']


def test_diagnostic_finishes_pending_maintenance_before_freeze(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import run_v3_r102_targeted as targeted
    calls = []
    class API:
        database = object()
        def run_maintenance(self, **kwargs):
            calls.append(('maintenance', kwargs))
            return SimpleNamespace(pending_count=0)
        def knowledge_digest(self):
            return 'stable'
        def freeze(self, destination):
            assert [name for name, _ in calls] == ['maintenance', 'audit']
            calls.append(('freeze', destination))
    monkeypatch.setattr(targeted, 'artifact_audit_snapshot',
                        lambda database: calls.append(('audit', database)))
    result = targeted.finalize_diagnostic(API(), tmp_path, [{'task_id': 'last_saved_task'}])
    assert calls[0][1] == {'triggering_task_id': 'last_saved_task',
        'milestone': 'r102_diagnostic_final_batch', 'finalize_pending': True}
    assert result['source_digest_unchanged']
