"""Exact-gamefile construction for controller-side strict ALFWorld replay."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from experiments.baselines.common import task_authority
from experiments.baselines.common.manifest import ManifestTask


def _install_fake_alfworld(monkeypatch, environment_module: ModuleType) -> None:
    alfworld = ModuleType("alfworld")
    agents = ModuleType("alfworld.agents")
    alfworld.agents = agents
    agents.environment = environment_module
    monkeypatch.setitem(sys.modules, "alfworld", alfworld)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)


def test_adapter_registers_only_specific_manifest_gamefile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "json_2.1.1" / "valid_unseen" / "target" / "game.tw-pddl"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    initialized_with: list[list[str]] = []

    class FakeTwEnv:
        def __init__(self, config, train_eval):
            self.game_files = ["wrong-a", "wrong-b"]
            self.num_games = 2

        def init_env(self, *, batch_size):
            assert batch_size == 1
            initialized_with.append(list(self.game_files))
            return object()

    environment = ModuleType("alfworld.agents.environment")
    environment.get_environment = lambda name: FakeTwEnv
    _install_fake_alfworld(monkeypatch, environment)

    adapter = AlfWorldAdapter(
        split="eval_out_of_distribution",
        alfworld_data=str(tmp_path),
        specific_gamefiles=[str(target)],
    )

    assert adapter.initialize() == 1
    assert initialized_with == [[str(target.resolve())]]
    assert adapter._tw_env.game_files == [str(target.resolve())]
    assert adapter._tw_env.num_games == 1


def test_strict_probe_uses_exact_gamefile_at_local_index_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative = "json_2.1.1/valid_unseen/family/trial/game.tw-pddl"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    captured = {}

    class FakeAdapter:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def reset(self, task):
            captured["probe"] = task

    monkeypatch.setattr(task_authority, "AlfWorldAdapter", FakeAdapter)
    entry = ManifestTask(
        index=0,
        task_id="alfworld_eval_out_of_distribution_55_look_at_obj_in_light",
        task_type="look_at_obj_in_light",
        source_split="valid_unseen",
        env_index=55,
        gamefile_rel=relative,
        gamefile_sha256="a" * 64,
        task_signature="b" * 64,
    )

    _, probe = task_authority.StrictTaskEvaluator(tmp_path)._build_probe_task(entry)

    assert captured["kwargs"]["specific_gamefiles"] == [str(target.resolve())]
    assert captured["probe"] is probe
    assert probe.context == {
        "env_index": 0,
        "game_file": str(target.resolve()),
    }


def test_strict_evaluate_closes_environment_when_validation_raises(
    tmp_path: Path,
    monkeypatch,
) -> None:
    closed = []

    class FakeEnv:
        def close(self):
            closed.append(True)

    class FakeAdapter:
        _env = FakeEnv()
        _observation = "Your task is to: inspect the object"
        _done = False
        _won = False
        _revision = 0

        def validator_channel(self):
            def fail_validation(contract):
                raise RuntimeError("validation failed")

            return SimpleNamespace(validate_task_contract=fail_validation)

        def task_contract(self, task):
            return object()

    entry = ManifestTask(
        index=0,
        task_id="task",
        task_type="look_at_obj_in_light",
        source_split="valid_unseen",
        env_index=55,
        gamefile_rel="json_2.1.1/valid_unseen/family/trial/game.tw-pddl",
        gamefile_sha256="a" * 64,
        task_signature="b" * 64,
    )
    evaluator = task_authority.StrictTaskEvaluator(tmp_path)
    monkeypatch.setattr(
        evaluator,
        "_build_probe_task",
        lambda manifest_entry: (FakeAdapter(), object()),
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        evaluator.evaluate(entry, [], official_success=False)

    assert closed == [True]
