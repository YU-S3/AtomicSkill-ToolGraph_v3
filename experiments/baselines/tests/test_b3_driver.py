"""Controller-side B3 train/freeze/train_eval protocol tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES, sha256_json
from experiments.baselines.b3_skillopt import driver as driver_module
from experiments.baselines.b3_skillopt.driver import (
    SkillOptBaselineDriver,
    _load_action_texts,
)
from experiments.baselines.common.driver import RunContext
from experiments.baselines.common.freeze import freeze_files
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.trace import CommonSidecarWriter
from experiments.baselines.common.usage import UsageSnapshot


def _manifest(tmp_path: Path, *, role: str) -> tuple[Path, TaskManifestSet]:
    manifest_id = "train_30" if role == "train" else "validation_30"
    split = "train" if role == "train" else "valid_seen"
    tasks: list[ManifestTask] = []
    index = 0
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        for family_index in range(5):
            relative = (
                f"json_2.1.1/{split}/{task_type}/trial_{family_index}/"
                "game.tw-pddl"
            )
            tasks.append(ManifestTask(
                index=index,
                task_id=f"{manifest_id}_{index}",
                task_type=task_type,
                source_split=split,
                env_index=index,
                gamefile_rel=relative,
                gamefile_sha256=hashlib.sha256(relative.encode()).hexdigest(),
                task_signature=hashlib.sha256(
                    f"signature:{manifest_id}:{index}".encode()
                ).hexdigest(),
            ))
            index += 1
    manifest = TaskManifestSet.create(
        manifest_id=manifest_id,
        benchmark="alfworld",
        source_split=split,
        seed=42,
        tasks=tuple(tasks),
    )
    path = manifest.save(tmp_path / f"{manifest_id}.json")
    return path, manifest


def _context(
    tmp_path: Path,
    *,
    train_path: Path,
    train: TaskManifestSet,
    validation_path: Path,
    validation: TaskManifestSet,
) -> RunContext:
    output = tmp_path / "run"
    output.mkdir()
    resolved_config = output / "config_resolved.json"
    resolved_config.write_text("{}\n", encoding="utf-8")
    config_digest = sha256_json({})
    data = tmp_path / "alfworld"
    data.mkdir()
    return RunContext(
        campaign_id="fixture",
        method_id="b3_skillopt",
        run_seed=42,
        output_dir=output,
        repo_root=tmp_path,
        external_repo=tmp_path / "external",
        external_commit="c" * 40,
        model_config=ModelConfig(
            provider="openai_compatible",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key_env="MODEL_API_KEY",
            reasoning_effort="high",
        ),
        max_environment_actions=100,
        alfworld_data=data,
        config_hash=config_digest,
        code_hash="b" * 64,
        run_id="run_20260907T010203Z_fixture",
        resolved_config_path=resolved_config.resolve(),
        identity={
            "config_digest": config_digest,
            "train_manifest_digest": train.digest,
            "validation_manifest_digest": validation.digest,
        },
        train_manifest_path=train_path.resolve(),
        validation_manifest_path=validation_path.resolve(),
        test_manifest_path=None,
    )


def _driver(tmp_path: Path) -> SkillOptBaselineDriver:
    worker_python = tmp_path / "worker-python"
    worker_python.touch()
    external = tmp_path / "external"
    external.mkdir(exist_ok=True)
    return SkillOptBaselineDriver(
        config={},
        repo_root=tmp_path,
        external_root=external,
        lock={},
        worker_python=worker_python,
    )


def _episode(
    task: ManifestTask,
    *,
    phase: str,
    artifact_digest: str,
    actions: int = 2,
) -> CommonEpisodeRecord:
    return CommonEpisodeRecord(
        method="b3_skillopt",
        phase=phase,
        run_seed=42,
        task_id=task.task_id,
        task_type=task.task_type,
        manifest_index=task.index,
        gamefile=task.gamefile_rel,
        gamefile_hash=task.gamefile_sha256,
        official_success=bool(task.index % 2),
        task_contract_success=None,
        strict_success=None,
        environment_actions=actions,
        target_llm_calls=1,
        artifact_digest_before=artifact_digest,
        artifact_digest_after=artifact_digest,
    )


def test_smoke_isolated_v2_wire_requires_evidence_not_task_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )
    captured = {}

    def fake_run_worker(*, wire, worker_module, python, wire_dir):
        captured.update(wire=wire, wire_dir=wire_dir)
        smoke_out = ctx.output_dir / "smoke"
        smoke_out.mkdir()
        usage = UsageSnapshot()
        usage.target.calls = 1
        usage.save(smoke_out / "usage.json")
        episode = _episode(
            train.tasks[0],
            phase="smoke",
            artifact_digest="1" * 64,
            actions=1,
        )
        episode.official_success = False
        writer = CommonSidecarWriter(smoke_out / "rollout")
        writer.write_episodes([episode])
        writer.write_environment_actions([{
            "episode_task_id": episode.task_id,
            "step_index": 0,
            "action": "look",
            "env_feedback": "fixture",
            "reward": 0.0,
            "done": False,
        }])
        return {"passed": True, "episodes": 1, "rows": 1}

    monkeypatch.setattr(driver_module, "run_worker", fake_run_worker)
    smoke = _driver(tmp_path).smoke(ctx, train)

    assert captured["wire"].phase == "smoke"
    assert captured["wire"].run_id == ctx.run_id
    assert captured["wire"].identity == ctx.identity
    assert captured["wire"].test_manifest_path is None
    assert captured["wire_dir"] == ctx.output_dir / "smoke"
    assert len(smoke.episodes) == 1
    assert smoke.episodes[0].official_success is False
    assert smoke.usage.target.calls == 1


def test_smoke_rejects_infrastructure_failure(tmp_path: Path, monkeypatch) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )

    def fake_run_worker(**kwargs):
        smoke_out = ctx.output_dir / "smoke"
        smoke_out.mkdir()
        UsageSnapshot().save(smoke_out / "usage.json")
        episode = _episode(
            train.tasks[0],
            phase="smoke",
            artifact_digest="1" * 64,
            actions=1,
        )
        episode.infrastructure_failure = True
        writer = CommonSidecarWriter(smoke_out / "rollout")
        writer.write_episodes([episode])
        writer.write_environment_actions([{
            "episode_task_id": episode.task_id,
            "step_index": 0,
            "action": "look",
        }])
        return {"passed": True, "episodes": 1, "rows": 1}

    monkeypatch.setattr(driver_module, "run_worker", fake_run_worker)
    with pytest.raises(RuntimeError, match="infrastructure failures"):
        _driver(tmp_path).smoke(ctx, train)


def test_train_uses_v2_identity_and_reads_only_train_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )
    captured = {}

    def fake_run_worker(*, wire, worker_module, python, wire_dir):
        captured.update(wire=wire, wire_dir=wire_dir)
        train_out = ctx.output_dir / "train"
        train_out.mkdir()
        (train_out / "best_skill.md").write_text("# best\n", encoding="utf-8")
        UsageSnapshot().save(train_out / "usage.json")
        episodes = [
            _episode(train.tasks[0], phase="train", artifact_digest="1" * 64),
            _episode(
                validation.tasks[0],
                phase="validation",
                artifact_digest="1" * 64,
            ),
        ]
        CommonSidecarWriter(train_out / "rollout").write_episodes(episodes)
        return {
            "passed": True,
            "episodes": {"total": 2, "train": 1, "validation": 1},
            "train_summary": {"best_step": 1},
        }

    monkeypatch.setattr(driver_module, "run_worker", fake_run_worker)
    result = _driver(tmp_path).train(ctx, train, validation)

    wire = captured["wire"]
    assert wire.phase == "train"
    assert wire.run_id == ctx.run_id
    assert wire.result_path == str(ctx.output_dir / "train" / "worker_result.json")
    assert wire.config_path == str(ctx.resolved_config_path)
    assert wire.identity == ctx.identity
    assert wire.test_manifest_path is None
    assert captured["wire_dir"] == ctx.output_dir / "train"
    assert len(result.episodes) == len(result.validation_episodes) == 1
    assert result.method_metrics == {"best_step": 1}


def test_train_rejects_cross_phase_sidecar(tmp_path: Path, monkeypatch) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )

    def fake_run_worker(**kwargs):
        train_out = ctx.output_dir / "train"
        train_out.mkdir()
        (train_out / "best_skill.md").write_text("# best\n", encoding="utf-8")
        UsageSnapshot().save(train_out / "usage.json")
        CommonSidecarWriter(train_out / "rollout").write_episodes([
            _episode(train.tasks[0], phase="train", artifact_digest="1" * 64),
            _episode(train.tasks[1], phase="train_eval", artifact_digest="1" * 64),
        ])
        return {
            "passed": True,
            "episodes": {"total": 2, "train": 1, "validation": 1},
        }

    monkeypatch.setattr(driver_module, "run_worker", fake_run_worker)
    with pytest.raises(ValueError, match="unexpected phases"):
        _driver(tmp_path).train(ctx, train, validation)


def test_train_rejects_context_manifest_identity_mismatch(
    tmp_path: Path,
) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )
    object.__setattr__(ctx, "identity", {
        **ctx.identity,
        "train_manifest_digest": "0" * 64,
    })
    with pytest.raises(ValueError, match="Train manifest identity"):
        _driver(tmp_path).train(ctx, train, validation)


def test_train_eval_is_identity_bound_read_only_and_strictly_replayed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )
    skill = tmp_path / "best_skill.md"
    skill.write_text("# frozen best\n", encoding="utf-8")
    frozen = freeze_files(
        method_id="b3_skillopt",
        source_files={"best_skill.md": skill},
        destination=ctx.output_dir / "frozen",
        source_train_manifest_hash=train.digest,
        source_validation_manifest_hash=validation.digest,
    )
    captured = {}

    def fake_run_worker(*, wire, worker_module, python, wire_dir):
        captured.update(wire=wire, wire_dir=wire_dir)
        phase_out = ctx.output_dir / "train_eval" / "rollout"
        episodes = [
            _episode(
                task,
                phase="train_eval",
                artifact_digest=frozen.digest,
            )
            for task in train.tasks
        ]
        writer = CommonSidecarWriter(phase_out)
        writer.write_episodes(episodes)
        writer.write_environment_actions([
            {
                "episode_task_id": task.task_id,
                "step_index": step,
                "action": "look" if step == 0 else "inventory",
                "env_feedback": "fixture",
                "reward": 0.0,
                "done": False,
            }
            for task in train.tasks
            for step in range(2)
        ])
        return {
            "passed": True,
            "episodes": 30,
            "rows": 30,
            "frozen_unchanged": True,
            "frozen_digest_before": frozen.digest,
            "frozen_digest_after": frozen.digest,
        }

    replayed = []

    class FakeStrictEvaluator:
        def __init__(self, alfworld_data):
            assert Path(alfworld_data) == ctx.alfworld_data

        def evaluate(self, entry, action_texts, *, official_success):
            replayed.append((entry.task_id, list(action_texts)))
            return SimpleNamespace(
                task_contract_success=True,
                strict_success=bool(official_success),
                invalid_actions=0,
            )

    monkeypatch.setattr(driver_module, "run_worker", fake_run_worker)
    monkeypatch.setattr(driver_module, "StrictTaskEvaluator", FakeStrictEvaluator)
    episodes = _driver(tmp_path).evaluate_train(ctx, frozen, train)

    wire = captured["wire"]
    assert wire.phase == "train_eval"
    assert wire.test_manifest_path is None
    assert wire.frozen_artifact_path == str(ctx.output_dir / "frozen")
    assert wire.result_path == str(
        ctx.output_dir / "train_eval" / "worker_result.json"
    )
    assert wire.identity["frozen_artifact_digest"] == frozen.digest
    assert wire.identity["evaluation_manifest_digest"] == train.digest
    assert captured["wire_dir"] == ctx.output_dir / "train_eval"
    assert len(episodes) == len(replayed) == 30
    assert all(actions == ["look", "inventory"] for _, actions in replayed)
    assert all(isinstance(episode.strict_success, bool) for episode in episodes)


def test_train_eval_rejects_worker_digest_drift(tmp_path: Path, monkeypatch) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    ctx = _context(
        tmp_path,
        train_path=train_path,
        train=train,
        validation_path=validation_path,
        validation=validation,
    )
    skill = tmp_path / "best_skill.md"
    skill.write_text("# frozen best\n", encoding="utf-8")
    frozen = freeze_files(
        method_id="b3_skillopt",
        source_files={"best_skill.md": skill},
        destination=ctx.output_dir / "frozen",
        source_train_manifest_hash=train.digest,
        source_validation_manifest_hash=validation.digest,
    )
    monkeypatch.setattr(driver_module, "run_worker", lambda **kwargs: {
        "passed": True,
        "frozen_unchanged": True,
        "frozen_digest_before": "0" * 64,
        "frozen_digest_after": frozen.digest,
    })
    with pytest.raises(RuntimeError, match="frozen_digest_before"):
        _driver(tmp_path).evaluate_train(ctx, frozen, train)


def test_action_sidecar_requires_unique_contiguous_steps(tmp_path: Path) -> None:
    task = ManifestTask(
        index=0,
        task_id="task_0",
        task_type="pick_and_place_simple",
        source_split="train",
        env_index=0,
        gamefile_rel="json_2.1.1/train/task_0/game.tw-pddl",
        gamefile_sha256="a" * 64,
        task_signature="b" * 64,
    )
    episode = _episode(
        task,
        phase="train_eval",
        artifact_digest="f" * 64,
    )
    sidecar = tmp_path / "common_environment_actions.jsonl"

    def write_steps(indexes):
        sidecar.write_text("".join(
            json.dumps({
                "episode_task_id": task.task_id,
                "step_index": index,
                "action": "look",
            }) + "\n"
            for index in indexes
        ), encoding="utf-8")

    write_steps([0, 0])
    with pytest.raises(ValueError, match="duplicates"):
        _load_action_texts(tmp_path, episodes=[episode])

    write_steps([0, 2])
    with pytest.raises(ValueError, match="contiguous"):
        _load_action_texts(tmp_path, episodes=[episode])

    write_steps([0, 1])
    assert _load_action_texts(tmp_path, episodes=[episode]) == {
        task.task_id: ["look", "look"]
    }
