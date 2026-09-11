from __future__ import annotations

import copy
from pathlib import Path

import pytest

from atomic_skillgraph.system import load_config
from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES, ProtocolError
from experiments.run_v3_frozen_eval import (
    _frozen_protocol,
    _r7_reference_manifests,
    _selection as frozen_selection,
    _validate_formal_config as validate_frozen_config,
)
from experiments.run_v3_train import (
    _r7_train_reference_manifest,
    _selection as train_selection,
    _train_protocol,
    _validate_formal_config as validate_train_config,
)


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
R9_LIFECYCLE = {
    "composite_active_deployment_successes": 2,
    "composite_candidate_zero_success_trial_limit": 3,
    "composite_candidate_activation_trial_limit": 5,
    "composite_active_consecutive_deployment_unsuccessful_limit": 3,
}


def _config(name: str) -> dict[str, object]:
    return load_config(ROOT / "configs" / name)


def _output(config: dict[str, object]) -> Path:
    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    return (ROOT / str(experiment["output_dir"])).resolve()


def _r9_expected_from_r8(
    r8: dict[str, object], *, seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    planner = copy.deepcopy(dict(r8["planner"]))  # type: ignore[arg-type]
    planner["literal_authorities"] = {}
    lifecycle = copy.deepcopy(dict(r8["lifecycle"]))  # type: ignore[arg-type]
    lifecycle.update(R9_LIFECYCLE)
    lifecycle["candidate_exploration_seed"] = seed
    return planner, lifecycle


@pytest.mark.parametrize("seed", SEEDS)
def test_r9_train_configs_are_exact_isolated_full120_protocols(seed: int) -> None:
    config = _config(f"alfworld_train_full_120_r9_seed{seed}.yaml")
    r8 = _config(f"alfworld_train_full_120_r8_seed{seed}.yaml")

    assert _train_protocol(config) == ("r9_full120", seed, 20, 120)
    labels, per_type, total = train_selection(config)
    assert tuple(labels) == ALFWORLD_FORMAL_TASK_TYPES
    assert (per_type, total) == (20, 120)
    validate_train_config(config, _output(config))

    reference = _r7_train_reference_manifest(config)
    assert reference is not None
    assert (reference.manifest_id, len(reference.tasks)) == ("train_120", 120)

    for section in ("llm", "cold_start", "runtime", "extraction", "harness"):
        assert config[section] == r8[section]
    expected_planner, expected_lifecycle = _r9_expected_from_r8(r8, seed=seed)
    assert config["planner"] == expected_planner
    assert config["lifecycle"] == expected_lifecycle

    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    assert experiment["name"] == f"alfworld_train_full_120_r9_seed{seed}"
    assert experiment["seed"] == seed
    assert experiment["initialize_v3_bank"] == "empty"
    assert experiment["output_dir"] == (
        f"runs/alfworld_train_full_120_r9_seed{seed}"
    )
    assert "_r8_" not in str(config["data_dir"])
    assert "_r8_" not in str(config["trace_data_dir"])


@pytest.mark.parametrize("seed", SEEDS)
def test_r9_frozen_configs_are_exact_isolated_fixed134_protocols(seed: int) -> None:
    config = _config(f"alfworld_frozen_eval_134_r9_seed{seed}.yaml")
    r8 = _config(f"alfworld_frozen_eval_134_r8_seed{seed}.yaml")

    assert _frozen_protocol(config) == ("r9_frozen134", seed, 0, 134)
    labels, per_type, total = frozen_selection(config)
    assert tuple(labels) == ALFWORLD_FORMAL_TASK_TYPES
    assert (per_type, total) == (0, 134)
    validate_frozen_config(config, _output(config))

    references = _r7_reference_manifests(config)
    assert references is not None
    train, test = references
    assert (train.manifest_id, len(train.tasks)) == ("train_120", 120)
    assert (test.manifest_id, len(test.tasks)) == ("test_ood_full_134", 134)

    for section in ("llm", "cold_start", "runtime", "extraction", "harness"):
        assert config[section] == r8[section]
    expected_planner, expected_lifecycle = _r9_expected_from_r8(r8, seed=seed)
    assert config["planner"] == expected_planner
    assert config["lifecycle"] == expected_lifecycle

    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    assert experiment["name"] == f"alfworld_frozen_eval_134_r9_seed{seed}"
    assert experiment["seed"] == seed
    assert experiment["source_train_run_dir"] == (
        f"runs/alfworld_train_full_120_r9_seed{seed}"
    )
    assert "_r8_" not in str(config["data_dir"])
    assert "_r8_" not in str(config["trace_data_dir"])


@pytest.mark.parametrize("field,wanted", tuple(R9_LIFECYCLE.items()))
@pytest.mark.parametrize("kind", ("train", "frozen"))
def test_r9_formal_configs_reject_lifecycle_threshold_drift(
    field: str, wanted: int, kind: str,
) -> None:
    if kind == "train":
        original = _config("alfworld_train_full_120_r9_seed42.yaml")
        validator = validate_train_config
    else:
        original = _config("alfworld_frozen_eval_134_r9_seed42.yaml")
        validator = validate_frozen_config
    changed = copy.deepcopy(original)
    changed["lifecycle"][field] = wanted + 1  # type: ignore[index]

    with pytest.raises(ProtocolError, match=rf"lifecycle\.{field}"):
        validator(changed, _output(changed))


@pytest.mark.parametrize(
    "name,validator",
    (
        ("alfworld_train_full_120_r9_seed42.yaml", validate_train_config),
        ("alfworld_frozen_eval_134_r9_seed42.yaml", validate_frozen_config),
    ),
)
def test_r9_formal_configs_reject_any_literal_authority(
    name: str, validator: object,
) -> None:
    changed = copy.deepcopy(_config(name))
    changed["planner"]["literal_authorities"] = {"mode": ["fast"]}  # type: ignore[index]

    with pytest.raises(ProtocolError, match="literal_authorities must be empty"):
        validator(changed, _output(changed))  # type: ignore[operator]


def test_r9_formal_configs_preserve_one_exact_llm_protocol() -> None:
    configs = [
        _config(f"alfworld_train_full_120_r9_seed{seed}.yaml")
        for seed in SEEDS
    ] + [
        _config(f"alfworld_frozen_eval_134_r9_seed{seed}.yaml")
        for seed in SEEDS
    ]
    assert all(config["llm"] == configs[0]["llm"] for config in configs[1:])


def test_r9_frozen_config_cannot_point_at_an_r8_source() -> None:
    changed = copy.deepcopy(_config("alfworld_frozen_eval_134_r9_seed42.yaml"))
    changed["data_dir"] = "runs/alfworld_train_full_120_r8_seed42/frozen/data_v3"
    changed["experiment"]["source_train_run_dir"] = (  # type: ignore[index]
        "runs/alfworld_train_full_120_r8_seed42"
    )
    changed["experiment"]["source_frozen_snapshot_dir"] = changed["data_dir"]  # type: ignore[index]

    with pytest.raises(ProtocolError, match="must use source"):
        validate_frozen_config(changed, _output(changed))
