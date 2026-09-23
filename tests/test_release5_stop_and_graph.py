import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import pytest

from experiments.release_control import ReleaseStopRequested, check_stop
from atomic_skillgraph.core.serialization import atomic_write_json, dataclass_from_dict, to_primitive
from atomic_skillgraph.core.contracts import CompositeSkill
from atomic_skillgraph.deployment.oldfirst_revision import revise_graph
from atomic_skillgraph.deployment.oldfirst_plan import compile_plan, workflow_jobs
from atomic_skillgraph.deployment.release_protocol import ReleaseError

FIXTURES = Path(__file__).parent/'fixtures/release5'


def test_CG01_02_05_graph_rebuilt_without_stale_ids_or_double_producer():
    source = dataclass_from_dict(CompositeSkill, json.loads((FIXTURES/'graph_source.json').read_text()))
    delta = json.loads((FIXTURES/'graph_delta.json').read_text())
    old = to_primitive(source)
    new = revise_graph(dict(delta, action='revise'), source)
    assert to_primitive(source) == old
    assert to_primitive(new.goal_contract) == old['goal_contract']
    assert [o.node_ref for o in new.occurrences] == [o.node_ref for o in source.occurrences]
    assert set(new.control_sequence) == {n['step_id'] for n in delta['target_nodes']}
    producers = [(e.target_step, e.target_role) for e in new.data_edges]
    assert len(producers) == len(set(producers))
    assert [(e.source_step, e.source_role) for e in new.data_edges if e.target_step == 'cool' and e.target_role == 'station'] == [('station_nav', 'location')]
    assert [(e.source_step, e.source_role) for e in new.data_edges if e.target_step == 'deliver' and e.target_role == 'object'] == [('cool', 'object')]
    assert 'destination' not in new.occurrences[1].binding_specs
    assert not set(source.control_sequence) & set(new.control_sequence)
    # Relabeling is structural only, never an episode-specific dispatch rule.
    renamed = copy.deepcopy(delta)
    mapping = {n['step_id']: f'step_{i}' for i, n in enumerate(renamed['target_nodes'])}
    for n in renamed['target_nodes']:
        n['step_id'] = mapping[n['step_id']]
        for b in n['binding_specs'].values():
            if b['source_step']:
                b['source_step'] = mapping[b['source_step']]
    other = revise_graph(dict(renamed, action='revise'), source)
    assert other.goal_contract == new.goal_contract
    assert [(mapping[e.source_step], mapping[e.target_step], e.source_role, e.target_role) for e in new.data_edges] == [
        (e.source_step, e.target_step, e.source_role, e.target_role) for e in other.data_edges]


def test_BK01_plan_preserves_all_prior_revisions_and_only_appends_allowed_graph(tmp_path):
    delta = json.loads((FIXTURES/'graph_delta.json').read_text())
    base = {'seed': 42, 'workflow_targets': [{'target_ref': delta['source_ref'], 'goal_group': 'cool'}],
        'derived_revision_jobs': [], 'deployment_preferences_patch': {'prefer_workflow_ref': 'unchanged'},
        'manual_publications': {'old': {'target_effective_status': 'active'}}}
    path = tmp_path/'base.json'; atomic_write_json(path, base)
    result = compile_plan(path, FIXTURES/'graph_delta.json', tmp_path/'new.json')
    assert result['manual_publications'] == base['manual_publications']
    assert result['workflow_targets'] == base['workflow_targets']
    assert workflow_jobs(result)[0]['target_ref'] == delta['target_ref']
    with pytest.raises(FileExistsError):
        compile_plan(path, FIXTURES/'graph_delta.json', tmp_path/'new.json')


def test_OP01_real_task_loop_settles_first_task_before_pause_and_never_begins_second(tmp_path, monkeypatch):
    import experiments.run_v3_frozen_eval as loop
    from experiments.protocol import TaskManifest, AttemptTraceLedger
    from atomic_skillgraph.core.serialization import atomic_create_json
    from test_release5_native_contract import session, tool
    output = tmp_path/'eval/seed42/rep01'; output.mkdir(parents=True)
    frozen = tmp_path/'seed42/frozen/release_manifest.json'
    config = {'experiment': {'task_manifest_path': str(output/'task_manifest.json')},
              'bank_release': {'release_manifest': str(frozen), 'source_seed': 42}}
    config_path = tmp_path/'config.json'; atomic_write_json(config_path, config)
    items = tuple(TaskManifest(i, f't{i}', f'sig{i}', 'frozen:digest', 'fake', 'test', '{}') for i in range(2))
    events = []
    s, p = session([{}])
    def run_task(task, **kwargs):
        events.append(('request', task.task_id))
        s.next_turn('one', tools=[tool()])
        atomic_write_json(tmp_path/'control/stop_all.json', {'operator_requested': True})
        atomic_create_json(output/'traces/trace_first.json', {'schema_version': 3, 'trace_id': 'trace_first',
            'task': {'task_id': task.task_id, 'task_signature': 'sig0', 'task_type': 'fixture'},
            'metadata': {'attempt_id': kwargs['attempt_id']}, 'llm_usage': [e.to_dict() for e in s.usage_ledger.events]})
        return SimpleNamespace(trace_id='trace_first', benchmark_success=True, task_contract_success=True,
            strict_task_success=True, learning_eligible=False, graph_self_sufficient_success=True, infrastructure_failure=False)
    class Ledger(AttemptTraceLedger):
        def begin(self, **kwargs):
            events.append(('begin', kwargs['task_id']))
            return super().begin(**kwargs)
        def capture(self, attempt, **kwargs):
            result = super().capture(attempt, **kwargs)
            events.append(('capture', attempt.task_id))
            return result
    system = SimpleNamespace(database=None, run_task=run_task, knowledge_digest=lambda: 'digest')
    monkeypatch.setattr(loop, 'artifact_audit_snapshot', lambda db: {})
    monkeypatch.setattr(loop, 'artifact_growth_audit', lambda a, b: {})
    with pytest.raises(ReleaseStopRequested):
        loop.run_frozen_tasks(system=system, config=config, config_path=config_path,
            tasks=[SimpleNamespace(task_id=i.task_id) for i in items], task_items=items, output_dir=output,
            resume=False, run_id='stop_fixture', phase='test', digest_before='digest', max_task_attempts=1,
            attempt_ledger=Ledger(output/'attempt_history', output/'traces'), train_manifest=SimpleNamespace(run_id='source', code_commit='code', metadata={}),
            invocation_started_at=datetime.now(timezone.utc), invocation_started_monotonic=0,
            expected_total=2, labels=[], per_type=0, experiment_seed=42, protocol='r103_frozen134',
            source_train_replay=False, reference_train_manifest=None, reference_test_manifest=None, counts={}, reference_disjoint_audit=None)
    assert events == [('begin', 't0'), ('request', 't0'), ('capture', 't0')]
    assert len(list((output/'attempt_history').glob('*.capture.json'))) == 1
    assert len(p.requests) == 1
    with sqlite3.connect(output/'run_state.sqlite3') as db:
        assert db.execute('SELECT task_id,state FROM run_tasks ORDER BY task_id').fetchall() == [('t0', 'completed'), ('t1', 'pending')]
    assert json.loads((output/'progress.json').read_text())['state'] == 'paused'


def test_OP01_supervisor_stops_all_repetitions(tmp_path, monkeypatch):
    import experiments.released_stream as stream
    import experiments.run_v3_released_frozen as runner
    monkeypatch.setattr(stream, 'validate_plan', lambda root: [
        {'seed': 42, 'rep': i, 'config': str(tmp_path/f'config{i}'), 'output': str(tmp_path/f'rep{i}')} for i in (1, 2, 3)])
    started = []
    def run(path, **kwargs):
        started.append(path)
        atomic_write_json(tmp_path/'control/stop_seed42.json', {})
        check_stop(tmp_path, 42)
    monkeypatch.setattr(runner, 'run', run)
    stream.run_stream(tmp_path, 42)
    assert len(started) == 1
    stream.run_stream(tmp_path, 42, resume=True)
    assert len(started) == 1  # explicit --resume does not silently clear the request
