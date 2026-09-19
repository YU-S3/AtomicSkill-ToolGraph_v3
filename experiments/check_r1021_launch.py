"""Read-only formal entry/config checks, before paying for provider probes."""
import argparse

from atomic_skillgraph.system import load_config
from experiments import run_v3_train as train, run_v3_frozen_eval as frozen


def check(seed: int) -> None:
    training = load_config(train.REPO_ROOT / f"configs/alfworld_train_full_120_r1021_seed{seed}.yaml")
    evaluation = load_config(frozen.REPO_ROOT / f"configs/alfworld_frozen_eval_134_r1021_seed{seed}.yaml")
    for module, config in ((train, training), (frozen, evaluation)):
        module._selection(config)
        module._validate_formal_config(config, module._path(config["experiment"]["output_dir"]))
    train._r7_train_reference_manifest(training)
    frozen._r7_reference_manifests(evaluation)
    print(f"seed{seed}: formal train/test config, source pairing and fixed manifests passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    check(parser.parse_args().seed)
