"""Formal epoch-boundary resume tests for the B3 SkillOpt worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.baselines.common.manifest import sha256_json
from experiments.baselines.common.resume import (
    ResumeCheckpointError,
    ResumeEligibilityError,
    ResumeIdentityError,
    prepare_epoch_boundary_resume,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_source(tmp_path: Path, *, last_completed: int = 11) -> tuple[
    Path, dict[str, object], dict[str, object], str
]:
    source = tmp_path / "parent_attempt"
    train = source / "train"
    initial = b"# Initial Skill\n"
    initial_digest = hashlib.sha256(initial).hexdigest()
    config: dict[str, object] = {
        "protocol_profile": "formal_v2",
        "train": {
            "num_epochs": 4,
            "train_size": 120,
            "batch_size": 40,
            "accumulation": 1,
        },
        "optimizer": {"use_meta_skill": True, "use_slow_update": True},
    }
    identity: dict[str, object] = {
        "config_digest": sha256_json(config),
        "train_manifest_digest": "1" * 64,
        "validation_manifest_digest": "2" * 64,
        "test_manifest_digest": "3" * 64,
        "controller_code_digest": "4" * 64,
        "skillopt_runtime_digest": "5" * 64,
        "model_identity_digest": "6" * 64,
        "alfworld_data_digest": "7" * 64,
    }
    policy: dict[str, object] = {
        "campaign_concurrency_policy": {
            "seed_lanes": 3,
            "campaign_provider_max_inflight": 16,
        },
        "provider_retry_policy": {
            "attempts": 5,
            "delays": [2, 5, 10, 20],
            "jitter_ratio": 0.1,
        },
    }
    manifest = {
        "schema_version": 3,
        "formal": True,
        "phase": "train",
        "method": "b3_skillopt",
        "run_seed": 42,
        "run_id": "parent_run_42",
        "identity": identity,
        "initial_skill_sha256": initial_digest,
        **policy,
    }
    _write_json(source / "run_manifest.json", manifest)
    _write_json(source / "config_resolved.json", config)
    _write_json(source / "run_state.json", {
        "state": "failed",
        "phase": "train",
        "run_id": "parent_run_42",
        "failure_kind": "infrastructure_failure",
    })
    _write_json(source / "failure.json", {
        "passed": False,
        "phase": "train",
        "method": "b3_skillopt",
        "run_id": "parent_run_42",
        "failure_kind": "infrastructure_failure",
    })
    for step in range(last_completed + 1):
        content = initial if step == 0 else f"# Skill {step}\n".encode()
        path = train / "skills" / f"skill_v{step:04d}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    history = []
    for step in range(1, last_completed + 1):
        best_step = 7 if step >= 7 else step
        history.append({
            "step": step,
            "epoch": ((step - 1) // 3) + 1,
            "step_in_epoch": (step - 1) % 3,
            "current_score": step / 20,
            "best_score": best_step / 20,
            "best_step": best_step,
            "current_origin": f"step_{step:04d}",
            "best_origin": f"step_{best_step:04d}",
        })
    _write_json(train / "history.json", history)
    runtime_best_step = min(7, last_completed)
    _write_json(train / "runtime_state.json", {
        "last_completed_step": last_completed,
        "current_skill_path": str(train / "skills" / f"skill_v{last_completed:04d}.md"),
        "current_score": last_completed / 20,
        "current_origin": f"step_{last_completed:04d}",
        "best_skill_path": str(train / "best_skill.md"),
        "best_score": runtime_best_step / 20,
        "best_step": runtime_best_step,
        "best_origin": (
            f"step_{runtime_best_step:04d}"
            if runtime_best_step else "initial_skill"
        ),
    })
    (train / "best_skill.md").write_bytes(
        (train / "skills" / f"skill_v{runtime_best_step:04d}.md").read_bytes()
    )
    for epoch in range(1, 5):
        _write_json(
            train / "meta_skill" / f"epoch_{epoch:02d}" / "meta_skill_result.json",
            {"epoch": epoch, "action": "done"},
        )
        _write_json(
            train / "slow_update" / f"epoch_{epoch:02d}" / "slow_result.json",
            {"epoch": epoch, "action": "reject"},
        )
    for step in (9, 10, 11):
        _write_json(
            train
            / "steps"
            / f"step_{step:04d}"
            / "rollout"
            / "episodes"
            / f"task_{step}"
            / "receipt.json",
            {"status": "completed", "step": step},
        )
    return source, identity, policy, initial_digest


def _prepare(
    source: Path,
    destination: Path,
    identity: dict[str, object],
    policy: dict[str, object],
    initial_digest: str,
    **kwargs: object,
):
    return prepare_epoch_boundary_resume(
        source_run=source,
        destination_train_dir=destination,
        expected_method="b3_skillopt",
        expected_seed=42,
        expected_identity=identity,
        expected_policy=policy,
        expected_initial_skill_sha256=initial_digest,
        steps_per_epoch=3,
        **kwargs,
    )


def test_prepares_new_attempt_at_last_complete_epoch_and_copies_cache(tmp_path: Path) -> None:
    source, identity, policy, initial_digest = _fixture_source(tmp_path)
    incomplete = (
        source / "train" / "steps" / "step_0011" / "rollout"
        / "episodes" / "task_incomplete"
    )
    incomplete.mkdir(parents=True)
    (incomplete / "partial.tmp").write_text("not committed", encoding="utf-8")
    destination = tmp_path / "new_attempt" / "train"
    destination.mkdir(parents=True)
    _write_json(destination / "worker_wire.json", {"schema_version": 2})

    prepared = _prepare(
        source,
        destination,
        identity,
        policy,
        initial_digest,
        copy_episode_cache=True,
    )

    assert prepared.source_last_completed_step == 11
    assert prepared.checkpoint_step == 9
    assert prepared.resume_from_step == 10
    assert prepared.resume_from_epoch == 4
    assert prepared.best_step == 7
    assert sorted(path.name for path in (destination / "skills").iterdir()) == [
        f"skill_v{step:04d}.md" for step in range(10)
    ]
    history = json.loads((destination / "history.json").read_text(encoding="utf-8"))
    assert [row["step"] for row in history] == list(range(1, 10))
    runtime = json.loads((destination / "runtime_state.json").read_text(encoding="utf-8"))
    assert runtime["last_completed_step"] == 9
    assert runtime["current_skill_path"] == str(
        (destination / "skills" / "skill_v0009.md").resolve()
    )
    assert runtime["best_skill_path"] == str((destination / "best_skill.md").resolve())
    assert (destination / "best_skill.md").read_bytes() == (
        destination / "skills" / "skill_v0007.md"
    ).read_bytes()

    for epoch in range(1, 4):
        assert (destination / "meta_skill" / f"epoch_{epoch:02d}").is_dir()
        assert (destination / "slow_update" / f"epoch_{epoch:02d}").is_dir()
    assert not (destination / "meta_skill" / "epoch_04").exists()
    assert not (destination / "slow_update" / "epoch_04").exists()
    assert not (destination / "steps" / "step_0009").exists()
    for step in (10, 11):
        assert (
            destination / "steps" / f"step_{step:04d}" / "rollout" / "episodes"
        ).is_dir()
    assert not (
        destination / "steps" / "step_0011" / "rollout" / "episodes"
        / "task_incomplete"
    ).exists()

    resume = json.loads((destination / "resume.json").read_text(encoding="utf-8"))
    assert resume["status"] == "prepared"
    assert resume["checkpoint_step"] == 9
    assert resume["parent_run_id"] == "parent_run_42"
    assert resume["expected_identity"] == identity
    assert resume["expected_policy"] == policy
    assert sorted(resume["inventory"]["slow_update"]) == [
        f"slow_update/epoch_{epoch:02d}" for epoch in range(1, 4)
    ]
    assert sorted(resume["inventory"]["episode_cache_dirs"]) == [
        f"steps/step_{step:04d}/rollout/episodes" for step in (10, 11)
    ]


def test_step_boundary_without_epoch_finalizers_rolls_back_one_epoch(
    tmp_path: Path,
) -> None:
    source, identity, policy, initial_digest = _fixture_source(
        tmp_path, last_completed=12,
    )
    # The optimizer step committed, but epoch 4's meta finalizer did not.  The
    # numerical step boundary is therefore not yet a formal-safe boundary.
    (source / "train" / "meta_skill" / "epoch_04" / "meta_skill_result.json").unlink()
    destination = tmp_path / "resumed" / "train"

    prepared = _prepare(
        source,
        destination,
        identity,
        policy,
        initial_digest,
    )

    assert prepared.source_last_completed_step == 12
    assert prepared.checkpoint_step == 9
    assert prepared.resume_from_step == 10
    assert prepared.resume_from_epoch == 4
    assert not (destination / "skills" / "skill_v0010.md").exists()
    history = json.loads((destination / "history.json").read_text(encoding="utf-8"))
    assert [row["step"] for row in history] == list(range(1, 10))
    assert not (destination / "meta_skill" / "epoch_04").exists()
    assert not (destination / "slow_update" / "epoch_04").exists()


def test_provider_outage_before_baseline_checkpoint_resumes_from_fresh_step_zero(
    tmp_path: Path,
) -> None:
    source, identity, policy, initial_digest = _fixture_source(
        tmp_path, last_completed=0,
    )
    (source / "train" / "runtime_state.json").unlink()
    (source / "train" / "history.json").unlink()

    destination = tmp_path / "baseline_resume" / "train"
    prepared = _prepare(
        source, destination, identity, policy, initial_digest,
    )

    assert prepared.checkpoint_step == 0
    assert prepared.resume_from_step == 1
    assert prepared.resume_from_epoch == 1
    runtime = json.loads(
        (destination / "runtime_state.json").read_text(encoding="utf-8")
    )
    assert runtime["last_completed_step"] == 0
    assert runtime["current_score"] == -1.0
    assert runtime["best_score"] == -1.0
    assert runtime["best_step"] == 0


def test_first_step_outage_allows_step_zero_runtime_with_missing_history(
    tmp_path: Path,
) -> None:
    source, identity, policy, initial_digest = _fixture_source(
        tmp_path, last_completed=0,
    )
    (source / "train" / "history.json").unlink()
    runtime_path = source / "train" / "runtime_state.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.update({"current_score": 0.5, "best_score": 0.5})
    _write_json(runtime_path, runtime)

    destination = tmp_path / "first_step_resume" / "train"
    prepared = _prepare(
        source, destination, identity, policy, initial_digest,
    )

    assert prepared.checkpoint_step == 0
    resumed_runtime = json.loads(
        (destination / "runtime_state.json").read_text(encoding="utf-8")
    )
    assert resumed_runtime["current_score"] == 0.5
    assert resumed_runtime["best_score"] == 0.5
    assert resumed_runtime["best_step"] == 0


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("manifest", "phase", "test"),
        ("state", "state", "completed"),
        ("state", "failure_kind", "protocol_failure"),
        ("failure", "failure_kind", "protocol_failure"),
    ],
)
def test_rejects_non_infrastructure_parent(
    tmp_path: Path, target: str, field: str, value: object
) -> None:
    source, identity, policy, initial_digest = _fixture_source(tmp_path)
    paths = {
        "manifest": source / "run_manifest.json",
        "state": source / "run_state.json",
        "failure": source / "failure.json",
    }
    payload = json.loads(paths[target].read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(paths[target], payload)

    with pytest.raises(ResumeEligibilityError):
        _prepare(source, tmp_path / "new" / "train", identity, policy, initial_digest)


def test_rejects_identity_or_policy_mismatch(tmp_path: Path) -> None:
    source, identity, policy, initial_digest = _fixture_source(tmp_path)
    wrong_identity = {**identity, "model_identity_digest": "f" * 64}
    with pytest.raises(ResumeIdentityError, match="immutable identity"):
        _prepare(
            source, tmp_path / "identity" / "train", wrong_identity, policy, initial_digest
        )
    wrong_policy = {
        **policy,
        "provider_retry_policy": {
            "attempts": 4,
            "delays": [2, 5, 10],
            "jitter_ratio": 0.1,
        },
    }
    with pytest.raises(ResumeIdentityError, match="policy field"):
        _prepare(
            source, tmp_path / "policy" / "train", identity, wrong_policy, initial_digest
        )


def test_rejects_incomplete_completed_epoch_state(tmp_path: Path) -> None:
    source, identity, policy, initial_digest = _fixture_source(tmp_path)
    (source / "train" / "slow_update" / "epoch_03" / "slow_result.json").unlink()

    with pytest.raises(ResumeCheckpointError, match="completed epoch 3"):
        _prepare(source, tmp_path / "new" / "train", identity, policy, initial_digest)


def test_rejects_cross_skill_or_stale_destination(tmp_path: Path) -> None:
    source, identity, policy, initial_digest = _fixture_source(tmp_path)
    destination = tmp_path / "new" / "train"
    destination.mkdir(parents=True)
    (destination / "history.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ResumeCheckpointError, match="not fresh"):
        _prepare(source, destination, identity, policy, initial_digest)

    (destination / "history.json").unlink()
    bad_initial = source / "train" / "skills" / "skill_v0000.md"
    bad_initial.write_text("changed", encoding="utf-8")
    with pytest.raises(ResumeIdentityError, match="skill_v0000"):
        _prepare(source, destination, identity, policy, initial_digest)
