"""Formal protocol gates for the v3.2-R8 three-seed experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.system import load_config
from experiments.protocol import (
    ProtocolError,
    active_composite_frozen_closure_audit,
    require_active_composite_frozen_closure,
    write_run_observability,
)
from experiments.run_v3_frozen_eval import (
    _frozen_protocol,
    _validate_formal_config as _validate_frozen_config,
)
from experiments.run_v3_train import (
    _train_protocol,
    _validate_formal_config as _validate_train_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_r8_three_seed_train_and_eval_configs_are_formally_isolated(seed: int) -> None:
    train_path = (
        REPO_ROOT / "configs" / f"alfworld_train_full_120_r8_seed{seed}.yaml"
    )
    train = load_config(train_path)
    train_output = REPO_ROOT / "runs" / train["experiment"]["name"]
    assert _train_protocol(train) == ("r8_full120", seed, 20, 120)
    assert train["lifecycle"][
        "composite_candidate_zero_success_trial_limit"
    ] == 3
    _validate_train_config(train, train_output)

    frozen_path = (
        REPO_ROOT / "configs" / f"alfworld_frozen_eval_134_r8_seed{seed}.yaml"
    )
    frozen = load_config(frozen_path)
    frozen_output = REPO_ROOT / "runs" / frozen["experiment"]["name"]
    assert _frozen_protocol(frozen) == ("r8_frozen134", seed, 0, 134)
    assert frozen["experiment"]["source_train_run_dir"].endswith(
        f"alfworld_train_full_120_r8_seed{seed}"
    )
    assert frozen["lifecycle"][
        "composite_candidate_zero_success_trial_limit"
    ] == 3
    _validate_frozen_config(frozen, frozen_output)


def test_r7_formal_protocols_remain_accepted() -> None:
    train = load_config(
        REPO_ROOT / "configs" / "alfworld_train_full_120_r7_seed42.yaml"
    )
    assert _train_protocol(train) == ("r7_full120", 42, 20, 120)
    _validate_train_config(
        train, REPO_ROOT / "runs" / "alfworld_train_full_120_r7_seed42"
    )

    frozen = load_config(
        REPO_ROOT / "configs" / "alfworld_frozen_eval_134_r7_seed42.yaml"
    )
    assert _frozen_protocol(frozen) == ("r7_frozen134", 42, 0, 134)
    _validate_frozen_config(
        frozen, REPO_ROOT / "runs" / "alfworld_frozen_eval_134_r7_seed42"
    )


@pytest.mark.parametrize("invalid_limit", [True, 3.0, 4])
def test_r8_formal_config_rejects_threshold_or_seed_pairing_drift(
    invalid_limit: object,
) -> None:
    train = load_config(
        REPO_ROOT / "configs" / "alfworld_train_full_120_r8_seed42.yaml"
    )
    train["lifecycle"][
        "composite_candidate_zero_success_trial_limit"
    ] = invalid_limit
    with pytest.raises(ProtocolError, match="zero_success_trial_limit"):
        _validate_train_config(
            train, REPO_ROOT / "runs" / "alfworld_train_full_120_r8_seed42"
        )

    frozen = load_config(
        REPO_ROOT / "configs" / "alfworld_frozen_eval_134_r8_seed42.yaml"
    )
    frozen["lifecycle"][
        "composite_candidate_zero_success_trial_limit"
    ] = invalid_limit
    with pytest.raises(ProtocolError, match="zero_success_trial_limit"):
        _validate_frozen_config(
            frozen, REPO_ROOT / "runs" / "alfworld_frozen_eval_134_r8_seed42"
        )

    frozen = load_config(
        REPO_ROOT / "configs" / "alfworld_frozen_eval_134_r8_seed42.yaml"
    )
    frozen["experiment"]["source_train_run_dir"] = (
        "runs/alfworld_train_full_120_r8_seed43"
    )
    frozen["experiment"]["source_frozen_snapshot_dir"] = (
        "runs/alfworld_train_full_120_r8_seed43/frozen/data_v3"
    )
    frozen["data_dir"] = frozen["experiment"]["source_frozen_snapshot_dir"]
    with pytest.raises(ProtocolError, match="must use source"):
        _validate_frozen_config(
            frozen, REPO_ROOT / "runs" / "alfworld_frozen_eval_134_r8_seed42"
        )


def _register_index_row(
    database: StateDatabase, ref: str, kind: str, status: str,
) -> None:
    logical_id, version = ref.split(":", 1)[1].split("@", 1)
    database.execute(
        "INSERT INTO artifact_index VALUES(?,?,?,?,?,?,?,?)",
        (ref, kind, logical_id, version, "hash", status, "fixture", 3),
    )
    database.connection.commit()


def test_formal_freeze_closure_audit_is_fail_closed_and_then_passes(
    tmp_path: Path,
) -> None:
    composite_ref = "composite:place@1.0.0"
    atomic_ref = "atomic:put@1.0.0"
    registry = SimpleNamespace(
        get_composite=lambda _ref: SimpleNamespace(
            occurrences=[SimpleNamespace(node_ref=atomic_ref)]
        )
    )
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        _register_index_row(database, composite_ref, "composite", "active")
        _register_index_row(database, atomic_ref, "atomic", "candidate")

        missing_relation = active_composite_frozen_closure_audit(
            database, registry,
        )
        assert missing_relation["active_composite_frozen_closure_passed"] is False
        assert {item["reason"] for item in missing_relation["violations"]} == {
            "missing_contains_relation",
            "child_atomic_not_frozen_usable",
        }
        with pytest.raises(ProtocolError, match="Frozen dependency closure failed"):
            require_active_composite_frozen_closure(database, registry)

        database.execute(
            "INSERT INTO graph_edges VALUES(?,?,?,?,?)",
            ("edge:contains", composite_ref, atomic_ref, "contains", "{}"),
        )
        database.execute(
            "UPDATE artifact_index SET status='active' WHERE artifact_ref=?",
            (atomic_ref,),
        )
        database.connection.commit()
        passed = require_active_composite_frozen_closure(database, registry)
        assert passed["active_composite_frozen_closure_passed"] is True
        assert passed["active_composite_count"] == 1
        assert passed["runtime_child_occurrence_count"] == 1


def test_run_observability_sidecar_has_required_wall_clock_fields(
    tmp_path: Path,
) -> None:
    target = write_run_observability(
        tmp_path / "reports" / "formal_run.json",
        run_id="formal_r8_seed42",
        run_started_at="2026-09-11T00:00:00+00:00",
        run_ended_at="2026-09-11T01:00:00+00:00",
        run_elapsed_seconds=3600.25,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["run_started_at"] == "2026-09-11T00:00:00+00:00"
    assert payload["run_ended_at"] == "2026-09-11T01:00:00+00:00"
    assert payload["run_elapsed_seconds"] == 3600.25
