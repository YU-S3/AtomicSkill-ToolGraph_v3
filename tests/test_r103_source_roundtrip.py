"""Persisted source normalization uses the original evidence, without mocks."""
import copy

import pytest

from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.traces.store import TraceStore
from test_r1021_i import source_case


def test_same_source_rephrased_e1_is_idempotent_but_evidence_conflicts_are_not(tmp_path):
    from atomic_skillgraph.knowledge.learning_sources import LearningSourceStore
    system, ctx, _, occurrence, normalized, proposal = source_case(tmp_path, include_proposal=True)
    try:
        trace = ctx.trace_builder.trace
        trace.benchmark_success = trace.learning_eligible = True
        trace.metadata['execution_source'] = dict(run_id='r', benchmark='synthetic', task_id=trace.task.task_id,
            task_signature=trace.task.task_signature, attempt_id='a', manifest_ordinal=0,
            attempt_ordinal=1, split='train', experiment_kind='formal')
        store = LearningSourceStore(system.database, system.data_dir)
        from atomic_skillgraph.knowledge.r103_protocol import SOURCE_DDL
        system.database.connection.executescript(SOURCE_DDL)
        atomic = system._canonical_atomic_for_occurrence(occurrence)
        first = store.stage(trace, normalized, occurrence, atomic, system.harness.profile_name, proposal=proposal)
        from test_r1021_boundaries import typed_atomicizer
        other_proposal = copy.deepcopy(proposal)
        other_proposal.phase_id = 'different_author_label'
        other_proposal.intent = 'same evidence rephrased'
        other_proposal.guideline = {'steps': ['Prepare the required input and check the effect.']}
        other, = typed_atomicizer().validate_and_canonicalize([other_proposal], normalized)
        other.occurrence_id += '_sidecar'
        other_atomic = system._canonical_atomic_for_occurrence(other)
        second = store.stage(trace, normalized, other, other_atomic, system.harness.profile_name, proposal=other_proposal)
        assert first['sample_key'] == second['sample_key'] and first['capsule_hash'] != second['capsule_hash']
        with system.database.transaction() as connection:
            store.commit(connection, first, 'parent')
            store.commit(connection, second, 'parent')
        with system.database.transaction() as connection:
            store.commit(connection, second, 'parent')
        rows = system.database.rows('SELECT * FROM learning_source_index')
        assert len(rows) == 1 and rows[0]['capsule_hash'] == first['capsule_hash']
        assert store.read(second)['occurrence']['intent'] == other.intent
        for field in ('prefix_events', 'effect_witness_refs'):
            payload = copy.deepcopy(store.read(second))
            payload['occurrence'][field] = []
            assert payload != store.read(second)
            damaged = {**second, **store._write(payload)}
            with pytest.raises(RuntimeError, match='conflicting learning source'):
                with system.database.transaction() as connection:
                    store.commit(connection, damaged, 'parent')
        with pytest.raises(RuntimeError, match='conflicting learning source'):
            with system.database.transaction() as connection:
                store.commit(connection, second, 'different_parent')
    finally:
        system.close()


@pytest.mark.parametrize("sparse", [False, True])
def test_learning_source_normalization_survives_trace_roundtrip(tmp_path, sparse):
    system, ctx, _, _, _ = source_case(tmp_path, sparse=sparse)
    try:
        trace = ctx.trace_builder.trace
        expected = system._normalized_learning_source(trace)
        assert expected["runtime_spans"]
        system.traces.save_atomic(trace)
        payload = system.traces.load_payload(trace.trace_id)
        before = copy.deepcopy(payload)
        reloaded = TraceStore.from_payload(payload)
        assert isinstance(reloaded.runtime_spans[0], dict)
        assert system._normalized_learning_source(reloaded) == expected
        assert to_primitive(reloaded) == before == payload
    finally:
        system.close()


def test_source_singleton_recheck_preserves_original_batch_id_and_checks_evidence(tmp_path):
    from atomic_skillgraph.evolution.identity_matching import raw_hash
    from atomic_skillgraph.knowledge.learning_sources import LearningSourceStore
    system, ctx, _, _, _, proposal = source_case(tmp_path, include_proposal=True)
    try:
        trace = ctx.trace_builder.trace
        trace.benchmark_success = trace.learning_eligible = True
        trace.metadata["execution_source"] = {
            "run_id": "source-run", "benchmark": "synthetic", "task_id": trace.task.task_id,
            "task_signature": trace.task.task_signature, "attempt_id": "attempt-1",
            "manifest_ordinal": 0, "attempt_ordinal": 1, "split": "train", "experiment_kind": "formal"}
        normalized = system._normalized_learning_source(trace)
        occurrence, = system.atomicizer.validate_and_canonicalize([proposal], normalized)
        occurrence.occurrence_id = f"occ_{trace.trace_id}_007"
        atomic = system._canonical_atomic_for_occurrence(occurrence)
        sources = LearningSourceStore(system.database, system.data_dir)
        reference = sources.stage(trace, normalized, occurrence, atomic,
                                  system.harness.profile_name, proposal=proposal)
        system.traces.save_atomic(trace)
        reference["source_trace_hash"] = raw_hash(system.traces.load_payload(trace.trace_id))
        assert sources.verified(system, reference)["occurrence"]["occurrence_id"] == occurrence.occurrence_id
        with pytest.raises(RuntimeError, match="occurrence index identity"):
            sources.verified(system, dict(reference, source_occurrence_id="different"))
        altered = copy.deepcopy(sources.read(reference))
        altered["occurrence"]["prefix_events"] = []
        changed_reference = {**reference, **sources._write(altered)}
        with pytest.raises(RuntimeError, match="differs from revalidated evidence"):
            sources.verified(system, changed_reference)
    finally:
        system.close()
