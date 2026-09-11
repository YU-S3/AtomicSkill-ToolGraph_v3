"""Top-level Train30/Validation30/frozen-Train30 orchestration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES
from experiments.baselines import run_method
from experiments.baselines.common.driver import TrainResult
from experiments.baselines.common.freeze import freeze_files
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.usage import RoleUsage, UsageSnapshot


def _manifest(tmp_path: Path, *, role: str) -> tuple[Path, TaskManifestSet]:
    manifest_id = "train_30" if role == "train" else "validation_30"
    split = "train" if role == "train" else "valid_seen"
    tasks = []
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        for family_index in range(5):
            index = len(tasks)
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
                    f"{manifest_id}:signature:{index}".encode()
                ).hexdigest(),
            ))
    manifest = TaskManifestSet.create(
        manifest_id=manifest_id,
        benchmark="alfworld",
        source_split=split,
        seed=42,
        tasks=tuple(tasks),
    )
    return manifest.save(tmp_path / f"{manifest_id}.json"), manifest


def _episodes(
    manifest: TaskManifestSet,
    *,
    phase: str,
    digest: str,
    strict: bool,
) -> list[CommonEpisodeRecord]:
    return [
        CommonEpisodeRecord(
            method="b3_skillopt",
            phase=phase,
            run_seed=42,
            task_id=task.task_id,
            task_type=task.task_type,
            manifest_index=task.index,
            gamefile=task.gamefile_rel,
            gamefile_hash=task.gamefile_sha256,
            official_success=task.index % 2 == 0,
            task_contract_success=(task.index % 2 == 0) if strict else None,
            strict_success=(task.index % 2 == 0) if strict else None,
            environment_actions=2,
            invalid_actions=0 if strict else None,
            target_llm_calls=1,
            target_prompt_tokens=10,
            target_completion_tokens=2,
            wall_time_ms=5,
            artifact_digest_before=digest,
            artifact_digest_after=digest,
        )
        for task in manifest.tasks
    ]


def test_top_level_train_is_train_val_freeze_train_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path, train = _manifest(tmp_path, role="train")
    validation_path, validation = _manifest(tmp_path, role="validation")
    output = tmp_path / "formal-output"
    data = tmp_path / "alfworld"
    data.mkdir()
    monkeypatch.setenv("ALFWORLD_DATA", str(data))
    monkeypatch.setenv("MODEL_API_KEY", "fixture-provider-key")
    monkeypatch.setattr(run_method, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_method, "verify_formal_manifest", lambda *args, **kwargs: {
        "passed": True,
    })
    monkeypatch.setattr(run_method, "_alfworld_data_signature", lambda root: {
        "resolved_data_root": str(root),
        "logic_sha256": "a" * 64,
        "grammar_sha256": "b" * 64,
        "dataset_gamefile_count": 60,
        "dataset_content_merkle_sha256": "c" * 64,
    })
    monkeypatch.setattr(run_method, "_controller_git_state", lambda: {
        "commit": "d" * 40,
        "branch": "fixture",
        "dirty": False,
        "dirty_status_sha256": "e" * 64,
    })

    config = {
        "schema_version": 1,
        "campaign_id": "fixture",
        "run_seed": 42,
        "max_environment_actions": 100,
        "worker_python": ".venv_b3_skillopt/bin/python",
        "model": {
            "provider": "openai_compatible",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
                "api_key_env": "MODEL_API_KEY",
                "reasoning_effort": "high",
            },
        "train": {"num_epochs": 4},
        "gradient": {"minibatch_size": 8, "merge_batch_size": 8},
        "optimizer": {"learning_rate": 4},
        "evaluation": {"use_gate": True},
        "env": {"max_steps": 100},
    }
    lock = {
        "schema_version": 1,
        "skillopt": {
            "repo": "https://github.com/microsoft/SkillOpt",
            "commit": "f" * 40,
            "version": "0.2.0",
            "runtime_tree": {"sha256": "1" * 64, "file_count": 155},
        },
    }

    class FakeDriver:
        def __init__(self) -> None:
            self.config = config
            self.train_called = 0
            self.eval_called = 0

        def preflight(self, ctx) -> None:
            assert ctx.config_hash == ctx.identity["config_digest"]
            assert ctx.test_manifest_path is None

        def train(self, ctx, train_manifest, validation_manifest) -> TrainResult:
            self.train_called += 1
            best = ctx.output_dir / "train" / "best_skill.md"
            best.parent.mkdir()
            best.write_text("# learned\n", encoding="utf-8")
            usage = UsageSnapshot(
                target=RoleUsage(calls=60, prompt_tokens=600, completion_tokens=120),
                evolution=RoleUsage(calls=3, prompt_tokens=30, completion_tokens=6),
            )
            return TrainResult(
                episodes=_episodes(
                    train_manifest, phase="train", digest="2" * 64, strict=False,
                ),
                validation_episodes=_episodes(
                    validation_manifest,
                    phase="validation",
                    digest="2" * 64,
                    strict=False,
                ),
                usage=usage,
                persistent_artifact_files={"best_skill.md": best},
                method_metrics={"best_step": 2, "best_validation_hard": 0.5},
            )

        def freeze(self, ctx, train_result):
            return freeze_files(
                method_id="b3_skillopt",
                source_files=train_result.persistent_artifact_files,
                destination=ctx.output_dir / "frozen",
                source_train_manifest_hash=train.digest,
                source_validation_manifest_hash=validation.digest,
            )

        def evaluate_train(self, ctx, frozen, train_manifest):
            self.eval_called += 1
            usage = UsageSnapshot(
                target=RoleUsage(calls=30, prompt_tokens=300, completion_tokens=60)
            )
            phase = ctx.output_dir / "train_eval"
            phase.mkdir()
            usage.save(phase / "usage.json")
            (phase / "worker_result.json").write_text(
                json.dumps({
                    "provider_evidence": {
                        "calls": 30,
                        "reasoning_token_status_counts": {"reported": 30},
                    }
                }) + "\n",
                encoding="utf-8",
            )
            return _episodes(
                train_manifest,
                phase="train_eval",
                digest=frozen.digest,
                strict=True,
            )

    driver = FakeDriver()
    monkeypatch.setattr(
        run_method,
        "load_lock_and_driver",
        lambda **kwargs: (lock, driver),
    )
    exit_code = run_method.main([
        "--phase", "train",
        "--train-manifest", str(train_path),
        "--validation-manifest", str(validation_path),
        "--config", str(tmp_path / "ignored-by-fixture.yaml"),
        "--output-dir", str(output),
    ])

    assert exit_code == 0
    assert driver.train_called == driver.eval_called == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["protocol"]["evaluation_scope"] == "train_resubstitution"
    assert report["protocol"]["held_out"] is False
    assert report["protocol"]["generalization_claim"] is False
    assert report["comparison_metrics"]["official_success"] == 15
    assert report["effectiveness"]["train_resubstitution"]["tasks"] == 30
    assert report["training_cost"]["unique_validation_tasks"] == 30
    assert json.loads((output / "task_manifest.json").read_text())["test"] is None
    assert json.loads((output / "run_state.json").read_text())["state"] == "completed"


def test_training_membership_rejects_missing_or_infrastructure(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path, role="train")
    episodes = _episodes(
        manifest, phase="train", digest="a" * 64, strict=False,
    )
    with pytest.raises(RuntimeError, match="missing"):
        run_method._validate_training_episode_membership(
            episodes[:-1], manifest, phase="train", run_seed=42,
        )
    episodes[0].infrastructure_failure = True
    with pytest.raises(RuntimeError, match="infrastructure_failure"):
        run_method._validate_training_episode_membership(
            episodes, manifest, phase="train", run_seed=42,
        )


def test_safe_error_does_not_replace_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert run_method._safe_error(RuntimeError("ordinary failure")) == "ordinary failure"


def test_atomic_json_refuses_stale_output(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    run_method._write_json_atomic(path, {"value": 1})
    with pytest.raises(FileExistsError):
        run_method._write_json_atomic(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
