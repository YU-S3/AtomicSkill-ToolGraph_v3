"""Persisted source normalization uses the original evidence, without mocks."""
import copy

import pytest

from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.traces.store import TraceStore
from test_r1021_i import source_case


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
