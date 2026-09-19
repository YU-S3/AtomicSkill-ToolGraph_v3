import pytest

from atomic_skillgraph.system import load_config
from experiments.check_r1021_launch import check
from experiments import run_v3_train as train, run_v3_frozen_eval as frozen
from experiments.protocol import ProtocolError


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_real_formal_entry_gates_and_manifests(seed):
    check(seed)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_cross_seed_frozen_source_still_rejected(seed):
    config = load_config(frozen.REPO_ROOT / f"configs/alfworld_frozen_eval_134_r1021_seed{seed}.yaml")
    other = 43 if seed == 42 else 42
    source = f"runs/alfworld_train_full_120_r1021_seed{other}"
    config["experiment"]["source_train_run_dir"] = source
    config["experiment"]["source_frozen_snapshot_dir"] = source + "/frozen/data_v3"
    config["data_dir"] = source + "/frozen/data_v3"
    with pytest.raises(ProtocolError, match="must use source"):
        frozen._validate_formal_config(config, frozen._path(config["experiment"]["output_dir"]))


@pytest.mark.parametrize("module,phase", [(train, "train_full_120"), (frozen, "frozen_eval_134")])
def test_seed_name_mismatch_still_rejected(module, phase):
    config = load_config(module.REPO_ROOT / f"configs/alfworld_{phase}_r1021_seed43.yaml")
    config["experiment"]["seed"] = 42
    with pytest.raises(ProtocolError, match="experiment.seed"):
        module._validate_formal_config(config, module._path(config["experiment"]["output_dir"]))
