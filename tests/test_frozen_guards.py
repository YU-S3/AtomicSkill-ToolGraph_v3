from __future__ import annotations

from pathlib import Path

import pytest

from atomic_skillgraph.knowledge import (
    ProvisionalAtomicRecord,
    ProvisionalStatus,
    provisional_ref_for,
)
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments.fakes import FakeHarness
from experiments.report import validate_frozen_v31_guards


def _config(
    data_dir: Path,
    trace_dir: Path,
    *,
    frozen: bool,
) -> dict:
    return {
        "schema_version": 3,
        "method_patch": "3.1",
        "data_dir": str(data_dir),
        "trace_data_dir": str(trace_dir),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "model",
            "api_key_env": "MODEL_API_KEY",
        },
        "planner": {
            "max_repeat_count": 4,
            "max_runtime_occurrences": 16,
            "cold_start_c1_repair_limit": 1,
        },
        "cold_start": {"enabled": not frozen},
        "experiment": {
            "benchmark": "alfworld",
            "condition": "full",
            "runtime_mode": "frozen" if frozen else "online",
            "freeze_skills": frozen,
            "allow_long_term_knowledge_writes": not frozen,
            "output_dir": str(trace_dir),
        },
    }


def test_frozen_constructs_no_failure_side_components_and_digest_is_stable(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source" / "data_v3"
    snapshot_dir = tmp_path / "frozen" / "data_v3"
    with AtomicSkillGraphSystem(
        _config(source_dir, tmp_path / "source-traces", frozen=False),
        harness=FakeHarness(),
    ) as source:
        assert source.failure_knowledge is not None
        signature = "frozen_digest_contract"
        source.failure_knowledge.upsert_provisional(ProvisionalAtomicRecord(
            provisional_ref=provisional_ref_for(signature),
            contract_signature=signature,
            canonical_intent="acquire_target_object",
            atomic_contract={
                "inputs": [{"name": "object", "semantic_type": "entity"}],
                "outputs": [{"name": "held_object", "semantic_type": "entity"}],
                "preconditions": [],
                "effects": [{
                    "predicate": "agent.holds",
                    "args": {"object": "$object"},
                    "cardinality": 1,
                    "distinct_by": "",
                }],
                "validator_spec": {"validator_id": "harness_atomic_effect"},
            },
            seeded_guideline={
                "intent": "establish the declared local Effect",
                "parameter_flow": "preserve declared role identity",
            },
            harness_profile="alfworld_v3",
            source_trace_id="trace-source",
            source_task_id="task-source",
            source_span={"event_start": 1, "event_end": 2},
            source_replay={"passed": True, "witness_refs": ["witness-1"]},
            aligned_plan_step_ids=("step-acquire",),
            progress_relation="consumed_prerequisite",
            status=ProvisionalStatus.TRIAL_READY,
        ))
        # A failure-side ledger fact must participate in the snapshot digest,
        # even though Frozen never constructs a semantic failure-side store.
        source.database.execute(
            "INSERT INTO cold_start_evidence("
            "event_id,task_id,trace_id,subject_ref,subject_kind,event_type,"
            "sequence_no,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                "event-1",
                "task-1",
                "trace-1",
                "provisional:test",
                "provisional_atomic",
                "source_replay_passed",
                0,
                "{}",
            ),
        )
        source.database.connection.commit()
        source_digest = source.knowledge_digest()
        source.freeze(snapshot_dir)

    frozen_trace_dir = tmp_path / "frozen-eval-traces"
    with AtomicSkillGraphSystem(
        _config(snapshot_dir, frozen_trace_dir, frozen=True),
        harness=FakeHarness(),
    ) as frozen:
        assert frozen.failure_knowledge is None
        assert frozen.provisional_retriever is None
        assert frozen.failure_experience_retriever is None
        assert frozen.planner.provisional_retriever is None
        assert frozen.planner.failure_experience_retriever is None
        assert frozen.knowledge_digest() == source_digest
        row = frozen.database.execute(
            "SELECT file_path FROM provisional_artifacts"
        ).fetchone()
        assert row is not None
        payload_path = Path(str(row["file_path"]))
        assert snapshot_dir in payload_path.parents
        assert payload_path.is_file()

        # Eval outputs live outside the snapshot and cannot perturb knowledge.
        frozen_trace_dir.mkdir(parents=True, exist_ok=True)
        (frozen_trace_dir / "trace-probe.json").write_text(
            '{"schema_version":3}', encoding="utf-8"
        )
        assert frozen.knowledge_digest() == source_digest
        with pytest.raises(RuntimeError, match="read-only"):
            frozen.database.set_metadata("forbidden", "write")


def test_frozen_report_requires_zero_failure_reads_and_provisional_selection() -> None:
    base = {
        "trace_id": "trace-1",
        "schema_version": 3,
        "metadata": {},
        "runtime_plan": {"source": "full_dynamic"},
        "llm_usage": [],
    }
    assert validate_frozen_v31_guards([base]) == {
        "failure_side_read_count": 0,
        "provisional_selected_count": 0,
        "cold_start_provider_call_count": 0,
    }

    read = {**base, "metadata": {"failure_side_read_count": 1}}
    with pytest.raises(ValueError, match="failure_side_read_count"):
        validate_frozen_v31_guards([read])

    selected = {**base, "metadata": {"provisional_selected_count": 1}}
    with pytest.raises(ValueError, match="provisional_selected_count"):
        validate_frozen_v31_guards([selected])
