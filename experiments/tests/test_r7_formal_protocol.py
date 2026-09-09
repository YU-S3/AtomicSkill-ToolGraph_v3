from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.system import load_config
from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES, ProtocolError
from experiments.run_v3_frozen_eval import (
    _r7_reference_manifests,
    _selection as frozen_selection,
    _validate_formal_config as validate_frozen_config,
    _verify_source_train,
)
from experiments.run_v3_train import (
    _final_maintenance_milestone,
    _r7_train_reference_manifest,
    _selection as train_selection,
    _validate_formal_config as validate_train_config,
)
from experiments.protocol import RunManifest, TaskManifest
from experiments.reference_manifest import load_formal_reference_manifest


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def _config(name: str) -> dict[str, object]:
    return load_config(ROOT / "configs" / name)


def _output(config: dict[str, object]) -> Path:
    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    return (ROOT / str(experiment["output_dir"])).resolve()


def _without_candidate_seed(payload: object) -> dict[str, object]:
    result = copy.deepcopy(dict(payload))  # type: ignore[arg-type]
    result.pop("candidate_exploration_seed", None)
    return result


@pytest.mark.parametrize("seed", SEEDS)
def test_r7_train_configs_are_exact_full120_protocol_repeats(seed: int) -> None:
    config = _config(f"alfworld_train_full_120_r7_seed{seed}.yaml")
    legacy = _config("alfworld_train_full_30_r6.yaml")

    labels, per_type, total = train_selection(config)
    assert tuple(labels) == ALFWORLD_FORMAL_TASK_TYPES
    assert (per_type, total) == (20, 120)
    validate_train_config(config, _output(config))
    reference = _r7_train_reference_manifest(config)
    assert reference is not None
    assert (reference.manifest_id, len(reference.tasks)) == ("train_120", 120)
    assert config["experiment"]["seed"] == seed  # type: ignore[index]
    assert config["lifecycle"]["candidate_exploration_seed"] == seed  # type: ignore[index]
    assert config["experiment"]["initialize_v3_bank"] == "empty"  # type: ignore[index]

    for section in ("llm", "planner", "cold_start", "runtime", "extraction"):
        assert config[section] == legacy[section]
    assert _without_candidate_seed(config["lifecycle"]) == legacy["lifecycle"]


@pytest.mark.parametrize("seed", SEEDS)
def test_r7_eval_configs_are_exact_fixed134_protocol_repeats(seed: int) -> None:
    config = _config(f"alfworld_frozen_eval_134_r7_seed{seed}.yaml")
    legacy = _config("alfworld_frozen_eval_60_r6.yaml")

    labels, per_type, total = frozen_selection(config)
    assert tuple(labels) == ALFWORLD_FORMAL_TASK_TYPES
    assert (per_type, total) == (0, 134)
    validate_frozen_config(config, _output(config))
    references = _r7_reference_manifests(config)
    assert references is not None
    train, test = references
    assert (train.manifest_id, len(train.tasks)) == ("train_120", 120)
    assert (test.manifest_id, len(test.tasks)) == ("test_ood_full_134", 134)
    assert config["experiment"]["seed"] == seed  # type: ignore[index]
    assert config["lifecycle"]["candidate_exploration_seed"] == seed  # type: ignore[index]

    for section in ("llm", "planner", "cold_start", "runtime", "extraction"):
        assert config[section] == legacy[section]
    assert _without_candidate_seed(config["lifecycle"]) == legacy["lifecycle"]


def test_r7_train_rejects_tampered_count_seed_name_manifest_id_and_path() -> None:
    original = _config("alfworld_train_full_120_r7_seed42.yaml")

    changed = copy.deepcopy(original)
    changed["harness"]["task_selection"]["total_tasks"] = 119  # type: ignore[index]
    with pytest.raises(ProtocolError, match="120"):
        train_selection(changed)

    changed = copy.deepcopy(original)
    changed["experiment"]["seed"] = 43  # type: ignore[index]
    with pytest.raises(ProtocolError, match="experiment.seed"):
        validate_train_config(changed, _output(changed))

    changed = copy.deepcopy(original)
    changed["experiment"]["name"] = "alfworld_train_full_120_r7_seed45"  # type: ignore[index]
    with pytest.raises(ProtocolError, match="allowed formal train"):
        train_selection(changed)

    changed = copy.deepcopy(original)
    changed["harness"]["task_selection"]["reference_manifest_id"] = "other"  # type: ignore[index]
    with pytest.raises(ProtocolError, match="reference_manifest_id"):
        validate_train_config(changed, _output(changed))

    changed = copy.deepcopy(original)
    changed["harness"]["task_selection"]["reference_manifest_path"] = (  # type: ignore[index]
        "data/baseline_manifests/test_ood_full_134.json"
    )
    with pytest.raises(ProtocolError, match="reference_manifest_path"):
        validate_train_config(changed, _output(changed))


def test_r7_eval_rejects_tampered_count_seed_name_manifest_and_source() -> None:
    original = _config("alfworld_frozen_eval_134_r7_seed42.yaml")

    changed = copy.deepcopy(original)
    changed["harness"]["task_selection"]["tasks_per_type"] = 22  # type: ignore[index]
    with pytest.raises(ProtocolError, match="tasks_per_type=0"):
        frozen_selection(changed)

    changed = copy.deepcopy(original)
    changed["experiment"]["seed"] = 43  # type: ignore[index]
    with pytest.raises(ProtocolError, match="experiment.seed"):
        validate_frozen_config(changed, _output(changed))

    changed = copy.deepcopy(original)
    changed["experiment"]["name"] = "alfworld_frozen_eval_134_r7_seed45"  # type: ignore[index]
    with pytest.raises(ProtocolError, match="allowed formal frozen"):
        frozen_selection(changed)

    changed = copy.deepcopy(original)
    changed["harness"]["task_selection"]["reference_manifest_id"] = "other"  # type: ignore[index]
    with pytest.raises(ProtocolError, match="reference_manifest_id"):
        validate_frozen_config(changed, _output(changed))

    changed = copy.deepcopy(original)
    changed["harness"]["task_selection"]["train_reference_manifest_path"] = (  # type: ignore[index]
        "data/baseline_manifests/test_ood_full_134.json"
    )
    with pytest.raises(ProtocolError, match="train_reference_manifest_path"):
        validate_frozen_config(changed, _output(changed))

    changed = copy.deepcopy(original)
    changed["data_dir"] = "runs/alfworld_train_full_120_r7_seed43/frozen/data_v3"
    changed["experiment"]["source_train_run_dir"] = (  # type: ignore[index]
        "runs/alfworld_train_full_120_r7_seed43"
    )
    changed["experiment"]["source_frozen_snapshot_dir"] = changed["data_dir"]  # type: ignore[index]
    with pytest.raises(ProtocolError, match="must use source"):
        validate_frozen_config(changed, _output(changed))


def test_r7_final_maintenance_milestone_is_protocol_metadata_authoritative() -> None:
    task = TaskManifest(0, "task-1", "signature-1", "initial:empty")
    legacy = RunManifest.create(
        run_id="legacy",
        phase="train",
        config_hash="config",
        code_commit="code",
        knowledge_digest="empty",
        tasks=(task,),
    )
    r7 = RunManifest.create(
        run_id="r7",
        phase="train",
        config_hash="config",
        code_commit="code",
        knowledge_digest="empty",
        tasks=(task,),
        metadata={
            "final_batch_maintenance_milestone": "formal_full_120_final_batch"
        },
    )
    assert _final_maintenance_milestone(legacy) == "formal_full_30_final_batch"
    assert _final_maintenance_milestone(r7) == "formal_full_120_final_batch"


def test_r7_frozen_source_verifier_rejects_task_identity_substitution(
    tmp_path: Path,
) -> None:
    reference = load_formal_reference_manifest(
        ROOT / "data/baseline_manifests/train_120.json",
        manifest_id="train_120",
    )
    family_counts = {
        label: sum(task.task_type == label for task in reference.tasks)
        for label in ALFWORLD_FORMAL_TASK_TYPES
    }
    items = [
        TaskManifest(
            task.index,
            task.task_id,
            task.task_signature,
            "initial:empty" if task.index == 0 else f"after:{task.index - 1}",
            "alfworld",
            "train",
            json.dumps({
                "task_type": task.task_type,
                "env_index": task.env_index,
                "game_file": f"/alfworld/{task.gamefile_rel}",
            }, sort_keys=True),
        )
        for task in reference.tasks
    ]
    first = items[0]
    items[0] = TaskManifest(
        first.ordinal,
        "substituted-task-id",
        first.task_signature,
        first.knowledge_milestone,
        first.benchmark,
        first.split,
        first.metadata_json,
    )
    manifest = RunManifest.create(
        run_id="alfworld_train_full_120_r7_seed42",
        phase="train",
        config_hash="config",
        code_commit="code",
        knowledge_digest="empty",
        tasks=items,
        metadata={
            "condition": "full",
            "tasks_per_type": 20,
            "total_tasks": 120,
            "reference_manifest_id": reference.manifest_id,
            "reference_manifest_digest": reference.digest,
            "reference_manifest_seed": reference.seed,
            "reference_manifest_task_count": len(reference.tasks),
            "reference_manifest_family_counts": family_counts,
            "seed": 42,
            "final_batch_maintenance_milestone": "formal_full_120_final_batch",
        },
    )

    with pytest.raises(ProtocolError, match="task identity differs from reference"):
        _verify_source_train(
            train_run_dir=tmp_path / manifest.run_id,
            train_manifest=manifest,
            freeze_manifest={},
            frozen_digest="frozen",
            current_code_digest="code",
            current_llm_hash="llm",
            reference_train_manifest=reference,
            expected_experiment_seed=42,
        )
