"""Integrity checks: secrets on disk, usage evidence, failure separation."""

from __future__ import annotations

import os

import pytest

from experiments.baselines.common.integrity import (
    assert_no_secrets_on_disk,
    run_manifest_payload,
    split_infrastructure_failures,
    validate_episode_usage,
)
from experiments.baselines.common.schema import CommonEpisodeRecord


def _episode(*, task_id: str, infra: bool = False, calls: int = 1) -> CommonEpisodeRecord:
    return CommonEpisodeRecord(
        method="b3_skillopt",
        phase="train",
        run_seed=42,
        task_id=task_id,
        task_type="pick_and_place_simple",
        manifest_index=0,
        gamefile="json_2.1.1/train/game.tw-pddl",
        gamefile_hash="a" * 64,
        official_success=False,
        target_llm_calls=calls,
        infrastructure_failure=infra,
    )


def test_secret_scan_fail_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-key-0123456789abcdef")
    (tmp_path / "leak.json").write_text('{"key": "sk-test-key-0123456789abcdef"}')
    with pytest.raises(RuntimeError, match="credentials"):
        assert_no_secrets_on_disk(tmp_path, api_key_env="MODEL_API_KEY")


def test_secret_pattern_scan(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    (tmp_path / "log.txt").write_text("Bearer sk-Abcdefghijklmnopqrstuvwxyz123456")
    with pytest.raises(RuntimeError, match="credentials"):
        assert_no_secrets_on_disk(tmp_path, api_key_env="MODEL_API_KEY")


def test_missing_usage_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="usage"):
        validate_episode_usage([_episode(task_id="t", calls=0)])


def test_infra_failure_not_counted_as_task_failure() -> None:
    normal, failed = split_infrastructure_failures([
        _episode(task_id="ok"),
        _episode(task_id="crash", infra=True, calls=0),
    ])
    assert [episode.task_id for episode in normal] == ["ok"]
    assert [episode.task_id for episode in failed] == ["crash"]
    validate_episode_usage(normal)


def test_run_manifest_provenance_fields() -> None:
    payload = run_manifest_payload(
        method="b3_skillopt",
        external_repo="https://github.com/microsoft/SkillOpt",
        external_commit="c" * 40,
        ours_controller_commit="d" * 40,
        train_manifest_hash="a" * 64,
        validation_manifest_hash="b" * 64,
        test_manifest_hash=None,
        alfworld_version="0.4.2",
        alfworld_data_signature="{}",
        model="deepseek-v4-flash",
        provider="openai_compatible",
        method_specific_decoding={"train": {"num_epochs": 4}},
        max_environment_actions=100,
        run_seed=42,
        phase="train",
        output_dir="runs/baselines/pilot/b3_skillopt/42",
    )
    for field in (
        "method", "external_repo", "external_commit", "ours_controller_commit",
        "train_manifest_hash", "validation_manifest_hash", "alfworld_version",
        "model", "provider", "max_environment_actions", "run_seed",
    ):
        assert field in payload
    assert payload["max_environment_actions"] == 100
    assert payload["run_seed"] == 42
