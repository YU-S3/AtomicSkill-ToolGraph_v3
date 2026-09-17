"""R10.1 A: rejected proposal audit is not registered lifecycle evidence."""
import copy
from dataclasses import replace

import pytest

from atomic_skillgraph.core.errors import AtomicSkillGraphError
from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.aligner import _tool_signature
from atomic_skillgraph.evolution.replay import ReplayCaseResult, replay_case_id
from atomic_skillgraph.evolution.replay_certificates import ReplayCertificates
from atomic_skillgraph.evolution.replay_publication import resolve
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments.fakes import FakeHarness, fake_task
from experiments.protocol import artifact_audit_snapshot
from test_batch_evolution import _system_config
from test_r92_replay_source_authority import _source


def observation(system, trace, tool, case_number, passed):
    case = copy.deepcopy(tool.tests[0])
    case['case_id'] = f'case_{case_number}'
    result = ReplayCaseResult(replay_case_id(case), case['trace_id'], 'source',
        trace.task.task_id, 'source', 'final_validation', passed,
        failure_code='' if passed else 'tool_ir_action_unavailable')
    event = ReplayCertificates(system.ledger).certificate(_tool_signature(tool), case, result,
        artifact_ref=str(tool.ref), trace_id=trace.trace_id, task_id=trace.task.task_id)
    trace.metadata.setdefault('replay_certificate_events', []).append(to_primitive(event))
    trace.metadata.setdefault('tool_replay_results', []).append(to_primitive(result))
    trace.evidence_event_refs.append(event.event_id)
    return event


@pytest.mark.parametrize('registered', [False, True])
def test_A01_A02_A03_mixed_replay_registration(tmp_path, registered):
    with AtomicSkillGraphSystem(_system_config(tmp_path), harness=FakeHarness()) as system:
        trace = system.orchestrator.create_trace_builder(fake_task('mixed', 'apple_1')).trace
        tool = ToolCompiler().compile([_source('source', 'source_trace', 'object', 'apple_1')])[0].tool
        a = replace(tool, ref=ToolRef('rejected', '1.0.0'))
        observation(system, trace, a, 1, True)
        observation(system, trace, a, 2, False)
        b = copy.deepcopy(tool)
        b.ref = ToolRef('accepted', '1.0.0')
        b.artifact['steps'] = [*b.artifact['steps'], *copy.deepcopy(b.artifact['steps'])]
        if registered:
            system.tools.register(a)
        system.tools.register(b)
        observation(system, trace, b, 1, True)
        observation(system, trace, b, 2, True)
        resolve(system, trace)
        trace.finish()
        system.traces.save_atomic(trace)
        system._commit_replay_certificates(trace)
        artifact_audit_snapshot(system.database)
        assert len(trace.metadata['tool_replay_results']) == 4
        assert len(trace.metadata['replay_certificate_events']) == (4 if registered else 2)
        assert len(ReplayCertificates(system.ledger).events(_tool_signature(a))) == (2 if registered else 0)
        before = system.ledger.count()
        system._commit_replay_certificates(trace)
        assert system.ledger.count() == before


def test_A04_canonical_identity_is_regenerated(tmp_path):
    with AtomicSkillGraphSystem(_system_config(tmp_path), harness=FakeHarness()) as system:
        trace = system.orchestrator.create_trace_builder(fake_task('alias', 'apple_1')).trace
        tool = ToolCompiler().compile([_source('source', 'source_trace', 'object', 'apple_1')])[0].tool
        system.tools.register(tool)
        event = observation(system, trace, replace(tool, ref=ToolRef('temporary', '1.0.0')), 1, False)
        resolve(system, trace)
        published = trace.metadata['replay_certificate_events'][0]
        assert published['artifact_ref'] == str(tool.ref)
        assert published['event_id'] != event.event_id
        assert event.event_id not in trace.evidence_event_refs
        system._commit_replay_certificates(trace)
        assert system.ledger.count() == 1


def test_A02_all_unregistered_failures_remain_trace_only(tmp_path):
    with AtomicSkillGraphSystem(_system_config(tmp_path), harness=FakeHarness()) as system:
        trace = system.orchestrator.create_trace_builder(fake_task('all_failed', 'apple_1')).trace
        tool = ToolCompiler().compile([_source('source', 'source_trace', 'object', 'apple_1')])[0].tool
        observation(system, trace, tool, 1, False)
        before = artifact_audit_snapshot(system.database)
        resolve(system, trace)
        trace.finish()
        system.traces.save_atomic(trace)
        system._commit_replay_certificates(trace)
        assert not trace.metadata['replay_certificate_events']
        assert len(trace.metadata['replay_candidate_observations']) == 1
        assert not trace.metadata['tool_replay_results'][0]['passed']
        assert system.ledger.count() == 0
        assert artifact_audit_snapshot(system.database) == before


def test_A06_batch_precheck_is_atomic(tmp_path):
    with AtomicSkillGraphSystem(_system_config(tmp_path), harness=FakeHarness()) as system:
        trace = system.orchestrator.create_trace_builder(fake_task('batch', 'apple_1')).trace
        tool = ToolCompiler().compile([_source('source', 'source_trace', 'object', 'apple_1')])[0].tool
        system.tools.register(tool)
        observation(system, trace, tool, 1, True)
        observation(system, trace, replace(tool, ref=ToolRef('unresolved', '1.0.0')), 2, False)
        with pytest.raises(AtomicSkillGraphError, match='unregistered replay target'):
            system._commit_replay_certificates(trace)
        assert system.ledger.count() == 0


def test_A06_existing_identity_with_wrong_signature_is_not_rebound(tmp_path):
    with AtomicSkillGraphSystem(_system_config(tmp_path), harness=FakeHarness()) as system:
        trace = system.orchestrator.create_trace_builder(fake_task('conflict', 'apple_1')).trace
        tool = ToolCompiler().compile([_source('source', 'source_trace', 'object', 'apple_1')])[0].tool
        system.tools.register(tool)
        altered = copy.deepcopy(tool)
        altered.artifact['steps'] *= 2
        observation(system, trace, altered, 1, True)
        with pytest.raises(AtomicSkillGraphError, match='signature differs'):
            resolve(system, trace)
        assert system.ledger.count() == 0


def test_A08_mixed_candidates_close_through_full_run_task(tmp_path, monkeypatch):
    from test_v32_r3_full_system_self_tool_reuse import _providers, _config, Gate36Harness, _locating_task
    from atomic_skillgraph.evolution import runtime_support_promotion
    providers, injected = _providers()
    config = _config(tmp_path)
    config['runtime']['persistent_runtime_support_promotion'] = True
    with AtomicSkillGraphSystem(config, harness=Gate36Harness(), provider=injected) as system:
        rejected_ref = ToolRef('mixed_rejected', '1.0.0')
        accepted_ref = ToolRef('mixed_accepted', '1.0.0')

        def mixed_candidates(owner, trace, task, observations):
            # Fixed replay outcomes at the promotion seam; all task execution,
            # immutable persistence, publication and projection are production.
            a = ToolCompiler().compile([_source('mixed', 'mixed_trace', 'object', 'apple_1')])[0].tool
            a = replace(a, ref=rejected_ref)
            b = copy.deepcopy(a)
            b.ref = accepted_ref
            b.artifact['steps'] *= 2
            owner.tools.register(b)
            for tool, outcomes in ((a, (True, False)), (b, (True, True))):
                for index, passed in enumerate(outcomes):
                    observation(owner, trace, tool, index, passed)
            return []

        monkeypatch.setattr(runtime_support_promotion, 'prepare_and_apply', mixed_candidates)
        trace = system.run_task(_locating_task('mixed_pipeline', 'orange_1'))
        assert trace.benchmark_success and not trace.infrastructure_failure
        assert len(trace.metadata['replay_candidate_observations']) >= 4
        rejected = [row for row in trace.metadata['replay_publication_decisions'] if row['candidate_ref'] == str(rejected_ref)]
        assert len(rejected) == 2
        assert all(row['disposition'] == 'trace_only_rejected' for row in rejected)
        events = trace.metadata['replay_certificate_events']
        assert sum(row['artifact_ref'] == str(accepted_ref) for row in events) == 2
        assert not any(row['artifact_ref'] == str(rejected_ref) for row in events)
        persisted = system.traces.load(trace.trace_id)
        assert persisted.metadata['replay_publication_decisions'] == trace.metadata['replay_publication_decisions']
        artifact_audit_snapshot(system.database)
