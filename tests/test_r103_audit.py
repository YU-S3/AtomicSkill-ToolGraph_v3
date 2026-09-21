"""Observability must not change matching or convert missing usage to zero."""
from copy import deepcopy
from types import SimpleNamespace

from atomic_skillgraph.evolution.identity_audit import capture, snapshot
from atomic_skillgraph.evolution.identity_matching import match_atomic, raw_hash
from test_r103_identity import atomic


def test_identity_audit_snapshot_is_detached_and_nested_scope_isolated():
    a, b = atomic(), atomic(("c", "d"), ("p", "q"))
    before = raw_hash(a), raw_hash(b)
    expected = match_atomic(a, b)
    assert snapshot() == {}
    with capture():
        assert match_atomic(a, b) == expected
        saved = snapshot()
        untouched = deepcopy(saved)
        with capture():
            assert snapshot()["identity_source_pairs_checked"] == 0
            match_atomic(a, b, max_states=0)
            assert snapshot()["identity_exact_different_unknown_by_layer"]["atomic"]["unknown"] == 1
        match_atomic(a, b)
        assert snapshot()["identity_exact_different_unknown_by_layer"]["atomic"]["exact"] == 2
        assert snapshot()["identity_source_pairs_checked"] == 1
        assert saved == untouched
    assert snapshot() == {}
    assert before == (raw_hash(a), raw_hash(b))


def test_validation_usage_preserves_unknown_reasoning(monkeypatch):
    from experiments import run_v3_r103_validation as driver
    from experiments import r103_metrics
    monkeypatch.setattr(r103_metrics, "trace_metrics", lambda _: None)
    monkeypatch.setattr(driver, "original_row_for", lambda *a: {
        "reasoning_tokens": 0, "non_reasoning_completion_tokens": 4})
    trace = SimpleNamespace(llm_usage=[{"completion_tokens":4,"reasoning_tokens":None}],
                            provider_requests=[SimpleNamespace(usage_status="unavailable")])
    row = driver.row_for(trace, 1)
    assert row["reasoning_tokens"] is None
    assert row["non_reasoning_completion_tokens"] is None
    assert row["reasoning_unavailable_calls"] == row["unknown_usage_requests"] == 1
    assert row["completion_tokens"] == 4


def test_CAU04_paired_cost_keeps_failures_in_denominator_and_missing_is_not_zero():
    import pytest
    from experiments.r103_metrics import paired_results, trace_metrics, aggregate
    old = [{"task_id":"a","official_won":True,"total_tokens":10},
           {"task_id":"b","official_won":False,"total_tokens":90}]
    new = [{"task_id":"a","official_won":False,"total_tokens":80},
           {"task_id":"b","official_won":True,"total_tokens":20}]
    result = paired_results(old,new)
    assert result["newly_solved"] == ["b"] and result["newly_failed"] == ["a"]
    assert result["new_mean_recorded_tokens_all_tasks"] == 50
    assert result["retained_success_recorded_token_delta_mean"] is None
    with pytest.raises(ValueError): paired_results(old,new[:1])
    value = trace_metrics({"metadata":{"repair_revision":"R10.3"}})
    assert value["exact_reuse_builder_avoided"] is None
    assert value["expression"]["required_surface_equal"] is None
    assert aggregate([{"r103_diagnostics":value}])["exact_reuse_builder_avoided_observed"] is None


def test_bank_metrics_reads_projection_without_writing(tmp_path):
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.governance.projections import ArtifactStats
    from experiments.r103_metrics import bank_metrics
    import json
    with StateDatabase(tmp_path / "state.sqlite3",r103=True) as db:
        stats = ArtifactStats("tool://fixture@1.0.0","tool")
        db.execute("INSERT INTO lifecycle_projection VALUES(?,?,?)",(stats.artifact_ref,json.dumps(stats.to_dict()),0))
        before = db.connection.total_changes
        result = bank_metrics(db)
        assert result[stats.artifact_ref]["registered_direct_success_count"] == 0
        assert db.connection.total_changes == before
